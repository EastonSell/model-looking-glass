"""CPU-only trusted GPT-2 tracing under a resident worker's execution lock.

This is an owner-side building block, not a client endpoint or GPU admission
system. Whole-batch outputs must be routed by the trusted batch assembler.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import threading

import torch

from .capture_dispatcher import CaptureAbortedError, CaptureDispatcher, CapturedSlice
from .model_ownership import ModelLease
from .resident_worker import (ForwardResult, InvalidRequest, RequestCancelled, ResidentWorker,
                              WorkerExecutionError, WorkerOutOfMemory, WorkerUnavailable)
from .trace_schema import validate_trace


def _describe_failure(error):
    """Retain bounded type/message diagnostics without traceback objects."""
    try:
        message = str(error)
    except BaseException:
        message = "<message unavailable>"
    return f"{type(error).__name__}: {message[:512]}"


def _interrupt_with_diagnostics(error, message):
    """Keep non-Exception interruptions interrupting after safe cleanup."""
    try:
        error.args = (message,)
    except BaseException:
        pass
    return error


class TraceOwnershipError(WorkerExecutionError):
    """The model or its hook ownership is quarantined after an uncertain state."""


@dataclass(frozen=True, slots=True)
class TraceSite:
    """A predefined GPT-2 layer and boundary role, never an arbitrary hook."""
    layer_index: int
    module_kind: str

    def __post_init__(self):
        if type(self.layer_index) is not int or self.layer_index < 0:
            raise ValueError("layer_index must be a nonnegative integer")
        if type(self.module_kind) is not str or self.module_kind not in ("block", "attn", "mlp"):
            raise ValueError("module_kind must be block, attn, or mlp")


@dataclass(frozen=True, slots=True)
class TracedForwardResult:
    """A trusted owner's whole-batch result and separately owned slices."""
    forward_result: ForwardResult
    events: tuple[CapturedSlice, ...]


class TracedGPT2Worker(ResidentWorker):
    """A resident GPT-2 LM-head worker with bounded CPU capture.

    All execution through this subclass holds one process-local model lease;
    traced calls additionally install predefined hooks only while the inherited
    worker lock is held. CUDA tracing is unavailable until admission is wired.
    """

    def __init__(self, model_id, model, *, device="cpu", **worker_limits):
        if torch.device(device).type != "cpu":
            raise ValueError("traced worker is CPU-only until GPU admission is integrated")
        # Imported lazily so the preexisting synthetic worker remains torch-only.
        from transformers import GPT2LMHeadModel
        if type(model) is not GPT2LMHeadModel:
            raise TypeError("model must be an unmodified GPT2LMHeadModel")
        if getattr(model.config, "add_cross_attention", False):
            raise ValueError("cross-attention GPT-2 is unsupported")
        if model.config.use_cache is not False or model.is_gradient_checkpointing:
            raise ValueError("disable model caching and gradient checkpointing")
        if not isinstance(model.transformer.h, torch.nn.ModuleList) or not model.transformer.h:
            raise ValueError("GPT-2 must have a standard block list")
        self._check_layout(model)
        self._check_aliases(model)
        super().__init__(model_id, model, device=device, **worker_limits)
        self._trace_gate = threading.Lock()
        self._pending_plan = None
        self._pending_events = None
        self._hook_cleanup_failed = False
        self._hook_cleanup_diagnostic = None
        self._registration_indeterminate = False
        self._model_state_uncertain = False
        self._quarantined_lease = None
        self._last_forward_failure = None

    def _poison(self):
        # Preserve the owner's lease if a failed forward changed model state.
        # Ordinary OOM on an unchanged CPU model can still release ownership.
        interruption = None
        if self._model is not None:
            try:
                if self._model_state() != self._baseline:
                    self._model_state_uncertain = True
            except BaseException as exc:
                self._model_state_uncertain = True
                if not isinstance(exc, Exception):
                    interruption = exc
        super()._poison()
        if interruption is not None:
            raise interruption from None

    @staticmethod
    def _check_layout(model):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP
        for block in model.transformer.h:
            if (type(block) is not GPT2Block or type(block.attn) is not GPT2Attention
                    or type(block.mlp) is not GPT2MLP):
                raise ValueError("GPT-2 trace boundaries must use standard module types")

    @staticmethod
    def _check_aliases(model):
        counts = Counter(id(module) for _, module in model.named_modules(remove_duplicate=False))
        for block in model.transformer.h:
            for module in (block, block.attn, block.mlp):
                if counts[id(module)] != 1:
                    raise ValueError("aliased GPT-2 trace modules are unsupported")

    def _check_available(self):
        if self._closed or self.poisoned or self._model is None:
            raise WorkerUnavailable("worker is closed or poisoned")

    def close(self):
        # The outer gate precedes the worker lock on every path in this class.
        with self._trace_gate:
            return super().close()

    @staticmethod
    def _has_registered_hooks(model):
        global_hooks = torch.nn.modules.module
        if any(getattr(global_hooks, name, None) for name in (
                "_global_forward_hooks", "_global_forward_pre_hooks",
                "_global_backward_hooks", "_global_backward_pre_hooks")):
            return True
        return any(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                   or module._backward_pre_hooks for module in model.modules())

    def _release_lease(self, lease):
        # Release is atomic and token-checked. If an interruption lands around
        # registry publication, one retry either finishes releasing this lease
        # or safely leaves a later owner's distinct token alone.
        failure = retry_failure = poison_failure = None
        try:
            lease.release()
            return
        except BaseException as exc:
            failure = exc
        try:
            lease.release()
        except BaseException as exc:
            retry_failure = exc
        if retry_failure is None:
            raise failure from None

        # A second failure makes lease state uncertain. Retain the handle and
        # disable this worker rather than silently stranding a live owner token.
        self._quarantined_lease = lease
        try:
            with self._lock:
                self._poison()
        except BaseException as exc:
            poison_failure = exc
        description = (f"model lease release failed: {_describe_failure(failure)}; "
                       f"release retry failed: {_describe_failure(retry_failure)}")
        if poison_failure is not None:
            description += f"; poison failed: {_describe_failure(poison_failure)}"
        for exc in (failure, retry_failure, poison_failure):
            if exc is not None and not isinstance(exc, Exception):
                raise _interrupt_with_diagnostics(exc, description) from None
        raise TraceOwnershipError(description)

    def _finish_lease(self, lease, acquired):
        if not acquired:
            self._release_lease(lease)
            return
        reasons = []
        if self._hook_cleanup_failed:
            reasons.append(self._hook_cleanup_diagnostic or "hook cleanup failed")
        if self._registration_indeterminate:
            reasons.append("hook registration was indeterminate")
        interruption = None
        # The base worker's postcheck can itself be interrupted, before it
        # poisons an altered model. Recheck while ownership and lock are held.
        try:
            with self._lock:
                if self._model is lease.model:
                    if self._model_state() != self._baseline:
                        self._model_state_uncertain = True
                elif self._model is not None:
                    self._model_state_uncertain = True
                if self.quarantined:
                    reasons.append("execution process is quarantined")
        except BaseException as exc:
            self._model_state_uncertain = True
            reasons.append(f"registered state verification failed: {_describe_failure(exc)}")
            if not isinstance(exc, Exception):
                interruption = exc
        if self._model_state_uncertain:
            reasons.append("registered model state changed or could not be verified")
        try:
            lease.verify_tree()
            if self._has_registered_hooks(lease.model):
                reasons.append("registered hooks remain")
        except BaseException as exc:
            reasons.append(f"model verification failed: {_describe_failure(exc)}")
            if interruption is None and not isinstance(exc, Exception):
                interruption = exc
        if reasons:
            # Preserve ownership and model references until process teardown.
            self._quarantined_lease = lease
            try:
                with self._lock:
                    self._poison()
            except BaseException as exc:
                reasons.append(f"poison failed: {_describe_failure(exc)}")
                if interruption is None and not isinstance(exc, Exception):
                    interruption = exc
            if interruption is not None:
                raise _interrupt_with_diagnostics(
                    interruption, f"model lease quarantined: {'; '.join(reasons)}") from None
            raise TraceOwnershipError(f"model lease quarantined: {'; '.join(reasons)}")
        self._release_lease(lease)

    def execute(self, request, cancel_event=None):
        """Run the original untraced API while excluding other model owners."""
        with self._trace_gate:
            lease = result = None
            acquired = False
            failure = None
            primary_failure = None
            self._last_forward_failure = None
            try:
                self._check_available()
                lease = ModelLease(self._model)
                lease.acquire()
                acquired = True
                result = super().execute(request, cancel_event)
            except BaseException as exc:
                primary_failure = _describe_failure(exc)
                if (self._last_forward_failure is not None
                        and self._last_forward_failure not in primary_failure):
                    primary_failure += f"; initial worker failure: {self._last_forward_failure}"
                request = result = None
                failure = (_interrupt_with_diagnostics(exc, primary_failure)
                           if not isinstance(exc, Exception) else exc)
            finally:
                try:
                    if lease is not None:
                        self._finish_lease(lease, acquired)
                except BaseException as exc:
                    description = _describe_failure(exc)
                    request = result = None
                    if failure is not None and not isinstance(failure, Exception):
                        failure = _interrupt_with_diagnostics(
                            failure, f"{primary_failure}; cleanup: {description}")
                    elif not isinstance(exc, Exception):
                        failure = _interrupt_with_diagnostics(
                            exc, f"{description}; initial failure: {primary_failure}" if primary_failure
                            else description)
                    elif primary_failure is not None:
                        failure = TraceOwnershipError(f"{description}; initial failure: {primary_failure}")
                    else:
                        failure = exc
                finally:
                    lease = None
                    self._last_forward_failure = None
            if failure is not None:
                failure.__traceback__ = failure.__context__ = failure.__cause__ = None
                raise failure from None
            return result

    @staticmethod
    def _snapshot_sites(sites):
        if type(sites) is not tuple or not sites:
            raise ValueError("sites must be a nonempty tuple of distinct TraceSite values")
        if any(type(site) is not TraceSite for site in sites):
            raise ValueError("sites must contain distinct TraceSite values")
        # Copy and revalidate: frozen dataclasses can still be modified with
        # object.__setattr__, and the caller retains the original objects.
        copied = tuple(TraceSite(site.layer_index, site.module_kind) for site in sites)
        if len(set(copied)) != len(copied):
            raise ValueError("sites must contain distinct TraceSite values")
        return copied

    def _resolve_sites(self, sites):
        sites = self._snapshot_sites(sites)
        model = self._model
        if model.config.use_cache is not False or model.is_gradient_checkpointing:
            raise ValueError("disable model caching and gradient checkpointing")
        self._check_layout(model)
        self._check_aliases(model)
        selected = []
        for site in sites:
            if site.layer_index >= len(model.transformer.h):
                raise ValueError("trace layer is outside the model")
            block = model.transformer.h[site.layer_index]
            module = block if site.module_kind == "block" else getattr(block, site.module_kind)
            path = f"transformer.h.{site.layer_index}"
            if site.module_kind != "block":
                path += f".{site.module_kind}"
            selected.append((module, path, site))
        return selected

    def execute_traced(self, request, *, captures, sites, sequence_starts,
                       max_events, max_trace_bytes, cancel_event=None):
        """Return a private batch result only after all worker postchecks pass.

        The trusted owner must supply truthful input-row ownership, coordinate
        run-wide sequence/forward indices, and route each row and slice.
        """
        with self._trace_gate:
            snapshot = dispatcher = lease = result = events = published = None
            acquired = False
            failure = None
            primary_failure = None
            self._last_forward_failure = None
            try:
                if cancel_event is not None and type(cancel_event) is not threading.Event:
                    raise InvalidRequest("cancel_event must be a threading.Event")
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelled("request cancelled before execution")
                if self.device.type != "cpu":
                    raise ValueError("traced worker is CPU-only until GPU admission is integrated")
                self._check_available()
                # Ownership must precede snapshots: snapshot OOM can poison a
                # worker and must not leave an uncertain model unprotected.
                lease = ModelLease(self._model)
                lease.acquire()
                acquired = True
                self._hook_cleanup_failed = False
                self._hook_cleanup_diagnostic = None
                self._registration_indeterminate = False
                # Snapshot before deriving sizes; the base worker snapshots once more.
                snapshot = self.snapshot_request(request)
                request = None
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelled("request cancelled before execution")
                batch_size, sequence_length = snapshot.inputs["input_ids"].shape
                sites = self._snapshot_sites(sites)
                dispatcher = CaptureDispatcher(
                    captures, batch_size=batch_size, sequence_length=sequence_length,
                    max_events=max_events, max_trace_bytes=max_trace_bytes,
                    capture_device="cpu", sequence_starts=sequence_starts)
                self._resolve_sites(sites)  # preflight without installing hooks
                self._pending_plan = (dispatcher, sites)
                self._pending_events = None
                # ResidentWorker acquires _lock before calling our _perform and
                # publishes logits only after cleanup, checks and cancellation.
                result = super().execute(snapshot, cancel_event)
                events = self._pending_events
                if events is None:
                    raise RuntimeError("completed trace did not produce events")
                published = TracedForwardResult(result, events)
            except BaseException as exc:
                primary_failure = _describe_failure(exc)
                if (self._last_forward_failure is not None
                        and self._last_forward_failure not in primary_failure):
                    primary_failure += f"; initial worker failure: {self._last_forward_failure}"
                result = events = published = snapshot = dispatcher = request = None
                # Nested validation/execution frames can otherwise retain private
                # input tensors through a caller-retained exception traceback.
                failure = (_interrupt_with_diagnostics(exc, primary_failure)
                           if not isinstance(exc, Exception) else exc)
            finally:
                self._pending_events = None
                self._pending_plan = None
                try:
                    if lease is not None:
                        self._finish_lease(lease, acquired)
                except BaseException as exc:
                    description = _describe_failure(exc)
                    # Tracebacks may outlive a failed call. Do not let this frame
                    # retain private outputs or input snapshots after cleanup fails.
                    result = events = published = snapshot = dispatcher = request = None
                    if failure is not None and not isinstance(failure, Exception):
                        failure = _interrupt_with_diagnostics(
                            failure, f"{primary_failure}; cleanup: {description}")
                    elif not isinstance(exc, Exception):
                        failure = _interrupt_with_diagnostics(
                            exc, f"{description}; initial failure: {primary_failure}" if primary_failure
                            else description)
                    elif primary_failure is not None:
                        failure = TraceOwnershipError(f"{description}; initial failure: {primary_failure}")
                    else:
                        failure = exc
                finally:
                    lease = None
                    self._last_forward_failure = None
            if failure is not None:
                # Raise outside active exception handlers, so Python cannot
                # silently reattach the discarded exception chain.
                failure.__traceback__ = failure.__context__ = failure.__cause__ = None
                raise failure from None
            return published

    def _perform(self, request):
        if self._pending_plan is None:
            try:
                outcome = super()._perform(request)
            except BaseException as exc:
                self._last_forward_failure = _describe_failure(exc)
                raise
            if outcome[1] is not None:
                kind, message = outcome[1]
                self._last_forward_failure = f"{kind.__name__}: {message[:512]}"
            return outcome
        if self.device.type != "cpu":
            raise ValueError("traced worker is CPU-only until GPU admission is integrated")
        dispatcher, sites = self._pending_plan
        selected = self._resolve_sites(sites)
        model = self._model
        handles = []
        guard_lock = threading.Lock()
        frames = defaultdict(list)
        guard = {"active": False, "entries": 0, "completed": 0, "failed": False}
        owner_thread = threading.get_ident()

        def begin(_module, _args):
            thread = threading.get_ident()
            with guard_lock:
                frames[thread].append(False)
                if thread != owner_thread or guard["active"] or guard["entries"]:
                    guard["failed"] = True
                    raise RuntimeError("exactly one nonreentrant root forward is required")
                frames[thread][-1] = True
                guard["active"] = True
                guard["entries"] += 1

        def end(_module, _args, output):
            thread = threading.get_ident()
            with guard_lock:
                stack = frames.get(thread)
                if not stack:
                    return None
                entered = stack.pop()
                if not stack:
                    frames.pop(thread, None)
                if entered:
                    guard["active"] = False
                    if output is not None:
                        guard["completed"] += 1
            return None

        def site_hook(path, site):
            def capture(_module, _args, output):
                if type(output) is torch.Tensor:
                    tensor, output_path = output, ()
                elif isinstance(output, (tuple, list)) and output and type(output[0]) is torch.Tensor:
                    tensor, output_path = output[0], (0,)
                else:
                    raise TypeError("unsupported GPT-2 boundary output")
                dispatcher.record(path, layer_index=site.layer_index,
                                  module_kind=site.module_kind, output=tensor,
                                  output_path=output_path)
                return None
            return capture

        base_outcomes = []
        primary_failure = None
        def perform_once():
            outcome = super(TracedGPT2Worker, self)._perform(request)
            base_outcomes.append(outcome)
            if outcome[1] is not None:
                kind, message = outcome[1]
                self._last_forward_failure = f"{kind.__name__}: {message[:512]}"
            return outcome

        try:
            def register(method, *args, **kwargs):
                # If registration raises before returning a handle, the hook may
                # already exist and cannot necessarily be removed safely.
                self._registration_indeterminate = True
                handle = method(*args, **kwargs)
                handles.append(handle)
                self._registration_indeterminate = False

            register(model.register_forward_pre_hook, begin, prepend=True)
            register(model.register_forward_hook, end, always_call=True)
            for module, path, site in selected:
                register(module.register_forward_hook, site_hook(path, site))
            try:
                outcome, events = dispatcher.run(perform_once)
            except CaptureAbortedError:
                # Base _perform converts hook/OOM failures to a classified result.
                if base_outcomes and base_outcomes[0][1] is not None:
                    kind, message = base_outcomes[0][1]
                    primary_failure = f"{kind.__name__}: {message[:512]}"
                    return base_outcomes[0]
                raise
            if outcome[1] is not None:
                # The dispatcher considers a returned failure a normal result.
                # Drop private captures; the base worker handles the failure.
                kind, message = outcome[1]
                primary_failure = f"{kind.__name__}: {message[:512]}"
                events = None
                return outcome
            if guard["failed"] or guard["entries"] != 1 or guard["completed"] != 1:
                raise RuntimeError("exactly one completed root forward is required")
            if {event.module_path for event in events} != {path for _, path, _ in selected}:
                raise RuntimeError("not all requested trace sites were observed")
            validate_trace(event.to_trace_event() for event in events)
            self._pending_events = events
            return outcome
        except (MemoryError, torch.OutOfMemoryError) as exc:
            primary_failure = _describe_failure(exc)
            self._last_forward_failure = primary_failure
            return None, (WorkerOutOfMemory, f"allocation failed: {exc}")
        except BaseException as exc:
            primary_failure = _describe_failure(exc)
            if self._last_forward_failure is None:
                self._last_forward_failure = primary_failure
            raise
        finally:
            cleanup_error = None
            cleanup_interrupt = None
            for handle in reversed(handles):
                try:
                    handle.remove()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    if cleanup_interrupt is None and not isinstance(exc, Exception):
                        cleanup_interrupt = exc
            if cleanup_error is not None:
                self._hook_cleanup_failed = True
                self._hook_cleanup_diagnostic = _describe_failure(cleanup_error)
                description = f"trace hook cleanup failed: {self._hook_cleanup_diagnostic}"
                if primary_failure is not None:
                    description += f"; initial failure: {primary_failure}"
                cleanup_error = None
                if cleanup_interrupt is not None:
                    raise _interrupt_with_diagnostics(cleanup_interrupt, description) from None
                raise RuntimeError(description) from None
