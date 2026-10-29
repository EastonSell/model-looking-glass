"""Serialized execution of bounded forwards on one trusted resident model.

This is not a sandbox, scheduler, model loader, or trace dispatcher. See
``docs/RESIDENT_WORKER.md`` for the ownership and reproducibility contract.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import threading
from types import MappingProxyType

import torch
from torch import nn

from .grouped_observation import (_enter_forward, _finish_forward, _result_forward,
                                  _returned_forward)


class InvalidRequest(ValueError):
    """A request is outside the supported, bounded forward contract."""


class RequestCancelled(RuntimeError):
    """The request was cancelled before execution or after device completion."""


class WorkerUnavailable(RuntimeError):
    """The model has been closed or its state can no longer be trusted."""


class WorkerExecutionError(RuntimeError):
    """A model forward or its output violated the execution contract."""


class WorkerConfigurationError(WorkerExecutionError):
    """The execution thread has unsupported ambient runtime settings."""


class WorkerStateError(WorkerExecutionError):
    """Post-forward model state or cache behavior violated its contract."""


class WorkerOutOfMemory(WorkerExecutionError):
    """Allocation failed; the worker must be replaced."""


class WorkerSynchronizationError(WorkerExecutionError):
    """Device completion could not be established; quarantine the process."""


class OutputLimitExceeded(WorkerExecutionError):
    """Logits exceed the allowed per-request retained-output size."""


@dataclass(frozen=True)
class ForwardRequest:
    request_id: str
    inputs: Mapping[str, torch.Tensor]
    seed: int = 0


@dataclass(frozen=True)
class ForwardResult:
    request_id: str
    logits: torch.Tensor


# PyTorch's default generators are process-global. This lock coordinates this
# module's workers, not arbitrary other threads that access those generators.
_rng_lock = threading.Lock()
_process_quarantined = threading.Event()
# This pre-existing global is set before the Event notification. Assigning an
# existing module binding does not require allocating a new retention object.
# Binding it to the Event identity also keeps test-isolated Event objects apart.
_process_quarantine_latch = None


def _is_process_quarantined():
    return (_process_quarantine_latch is _process_quarantined
            or _process_quarantined.is_set())


def _mark_process_quarantined():
    global _process_quarantine_latch
    _process_quarantine_latch = _process_quarantined
    try:
        _process_quarantined.set()
    except BaseException:
        # Notification is secondary; the process-wide latch remains set.
        pass
# Keep known live resources until the owner terminates this execution process.
# Model internals may have released additional temporaries before an error.
_quarantine_references = []
_mapping_proxy_type = type(MappingProxyType({}))
_input_names = frozenset(("input_ids", "attention_mask", "token_type_ids", "position_ids"))


def _describe_failure(exc, fallback):
    """Keep useful diagnostics without depending on an exception's __str__."""
    try:
        return f"{type(exc).__name__}: {str(exc)[:1024]}"
    except BaseException:
        return fallback


def registered_model_state(model, device):
    """Snapshot registered identities and versions; this is not an integrity proof.

    The snapshot is meaningful only in this process and can miss ``.data``
    edits, unregistered mutable state, and arbitrary model-code side effects.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a trusted torch.nn.Module")
    device = torch.device(device)
    global_hooks = nn.modules.module
    if any(getattr(global_hooks, name, None) for name in (
            "_global_forward_hooks", "_global_forward_pre_hooks",
            "_global_backward_hooks", "_global_backward_pre_hooks")):
        raise ValueError("process-global module hooks are not supported")
    modules = []
    for name, module in model.named_modules():
        if module.training:
            raise ValueError("every model module must be in evaluation mode")
        if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or module._backward_pre_hooks):
            raise ValueError("model hooks are not supported by this worker")
        modules.append((name, id(module)))
    tensors = []
    for kind, collection in (("parameter", model.named_parameters()),
                             ("buffer", model.named_buffers())):
        for name, tensor in collection:
            if tensor.device != device:
                raise ValueError("all model parameters and buffers must be on the worker device")
            try:
                version = tensor._version
            except RuntimeError as exc:
                raise ValueError("model tensors must track in-place changes") from exc
            tensors.append((kind, name, id(tensor), version, tuple(tensor.shape),
                            str(tensor.dtype), str(tensor.device)))
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        cache_policy = config.use_cache
        if cache_policy is not False:
            raise ValueError("model cache must be explicitly disabled")
    else:
        cache_policy = None
    return tuple(modules), tuple(tensors), (id(config), cache_policy)


class ResidentWorker:
    """Own a preconfigured model and execute at most one forward at a time."""

    cancellation_exception = RequestCancelled

    def __init__(self, model_id, model, *, device="cpu", max_batch_size=1,
                 max_sequence_length=512, max_input_bytes=1024 * 1024,
                 max_output_bytes=8 * 1024 * 1024):
        if _is_process_quarantined():
            raise WorkerSynchronizationError("execution process is quarantined")
        if type(model_id) is not str or not model_id or len(model_id) > 256:
            raise ValueError("model_id must be a nonempty string of at most 256 characters")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a trusted torch.nn.Module")
        for name, value in (("max_batch_size", max_batch_size),
                            ("max_sequence_length", max_sequence_length),
                            ("max_input_bytes", max_input_bytes),
                            ("max_output_bytes", max_output_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        resolved_device = torch.device(device)
        if resolved_device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA devices are supported")
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        self._model_id = model_id
        self._device = resolved_device
        self._max_batch_size = max_batch_size
        self._max_sequence_length = max_sequence_length
        self._max_input_bytes = max_input_bytes
        self._max_output_bytes = max_output_bytes
        self._lock = threading.Lock()
        self._model = model
        # Allocate fixed custody slots before any work can be launched. If a
        # device drain fails, retain known inputs/outputs on this worker even
        # when a failure-time append to the process hold list cannot allocate.
        self._live_buffers = [None, None, None, None, None]
        # Set before any device work and cleared only after a successful drain.
        # A missing receipt is conservatively treated as an uncertain drain.
        self._drain_pending = False
        self._poison_pending = threading.Event()
        self._closed = False
        self._baseline = self._model_state()

    @property
    def model_id(self):
        return self._model_id

    @property
    def device(self):
        return self._device

    @property
    def max_batch_size(self):
        return self._max_batch_size

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def max_input_bytes(self):
        return self._max_input_bytes

    @property
    def max_output_bytes(self):
        return self._max_output_bytes

    @property
    def poisoned(self):
        return self._poison_pending.is_set() or _is_process_quarantined()

    @property
    def quarantined(self):
        """A device drain failed; this execution process requires teardown."""
        return _is_process_quarantined()

    def _model_state(self):
        return registered_model_state(self._model, self.device)

    def validate_request(self, request):
        """Validate without copying inputs or modifying model state."""
        error = self._request_error(request)
        request = None
        if error is not None:
            raise InvalidRequest(error)

    def _request_error(self, request):
        if type(request) is not ForwardRequest:
            return "request must be a ForwardRequest"
        if type(request.request_id) is not str or not request.request_id or len(request.request_id) > 256:
            return "request_id must be a nonempty string of at most 256 characters"
        if type(request.seed) is not int or not 0 <= request.seed < 2**63:
            return "seed must be an integer in [0, 2**63)"
        if type(request.inputs) not in (dict, _mapping_proxy_type):
            return "inputs must be a plain tensor dictionary"
        if any(type(name) is not str for name in request.inputs):
            return "input names must be plain strings"
        if "input_ids" not in request.inputs or not set(request.inputs).issubset(_input_names):
            return "inputs must contain input_ids and only supported names"
        shape = None
        total_bytes = 0
        for name, value in request.inputs.items():
            if type(value) is not torch.Tensor or value.device.type != "cpu" or value.layout != torch.strided:
                return "inputs must be ordinary dense CPU tensors"
            allowed = (torch.int32, torch.int64, torch.bool) if name == "attention_mask" else (torch.int32, torch.int64)
            if value.dtype not in allowed or value.ndim != 2:
                return "input tensors must be two-dimensional integer token tensors"
            if shape is None:
                shape = tuple(value.shape)
            elif shape != tuple(value.shape):
                return "all input tensors must have the same shape"
            total_bytes += value.numel() * value.element_size()
        if not (0 < shape[0] <= self.max_batch_size and 0 < shape[1] <= self.max_sequence_length):
            return "input batch or sequence length exceeds its bound"
        if total_bytes > self.max_input_bytes:
            return "input tensor bytes exceed their bound"
        return None

    def snapshot_request(self, request):
        """Make a bounded CPU snapshot; caller must not mutate during copying."""
        if _is_process_quarantined():
            request = None
            raise WorkerSynchronizationError("execution process is quarantined")
        invalid = None
        try:
            self.validate_request(request)
        except InvalidRequest as exc:
            invalid = str(exc)
        if invalid is not None:
            request = None
            raise InvalidRequest(invalid) from None
        oom = False
        try:
            inputs = {name: tensor.detach().clone(memory_format=torch.contiguous_format)
                      for name, tensor in request.inputs.items()}
        except (MemoryError, torch.OutOfMemoryError):
            self._poison_pending.set()
            oom = True
        if oom:
            request = None
            with self._lock:
                self._poison()
            raise WorkerOutOfMemory("input snapshot allocation failed") from None
        return ForwardRequest(request.request_id, MappingProxyType(inputs), request.seed)

    def _poison(self):
        self._poison_pending.set()
        self._release_model()

    def _release_model(self):
        if (_is_process_quarantined() or self._drain_pending) and self._model is not None:
            try:
                _quarantine_references.append(self._model)
            except BaseException:
                pass
            # A rooted worker itself remains an allocation-free fallback.
            return
        self._model = None

    def close(self):
        """Wait for an active call and release the worker's model reference."""
        with self._lock:
            self._closed = True
            self._release_model()

    def _perform(self, request):
        """Return outcome data, not exceptions carrying model/tensor tracebacks."""
        self._live_buffers[0] = request
        try:
            return self._perform_with_custody(request)
        finally:
            if self._drain_pending:
                # A failed/interrupted cleanup may have skipped its first
                # attempt to publish the process latch; retry before clearing.
                _mark_process_quarantined()
            elif not _is_process_quarantined():
                for index in range(len(self._live_buffers)):
                    self._live_buffers[index] = None

    def _perform_with_custody(self, request):
        outcome, failure, sync_failure, interrupt = None, None, None, None
        device_inputs, output, logits = None, None, None
        devices = [self.device.index] if self.device.type == "cuda" else []
        sync_attempted = False
        if devices:
            self._drain_pending = True

        def quarantine():
            nonlocal sync_failure
            # Do this before diagnostics or retention: both can fail in an OOM.
            sync_failure = "target device synchronization failed"
            _mark_process_quarantined()
            # Recover values that were returned to local variables but whose
            # normal custody-slot assignment was interrupted. No new list is
            # allocated on this failure path.
            try:
                self._live_buffers[0] = request
                self._live_buffers[1] = device_inputs
                self._live_buffers[2] = output
                self._live_buffers[3] = logits
                self._live_buffers[4] = outcome
            except BaseException:
                # Repeated asynchronous interruption requires process teardown.
                pass
            try:
                _quarantine_references.append((self._model, request, device_inputs, output, logits, outcome))
            except BaseException:
                # The process must still be quarantined when reference retention fails.
                pass

        try:
            with _rng_lock:
                try:
                    if _is_process_quarantined():
                        return None, (WorkerSynchronizationError, "execution process is quarantined")
                    try:
                        with torch.random.fork_rng(devices=devices):
                            try:
                                torch.random.default_generator.manual_seed(request.seed)
                                if devices:
                                    torch.cuda.default_generators[self.device.index].manual_seed(request.seed)
                                with torch.inference_mode():
                                    device_inputs = {}
                                    self._live_buffers[1] = device_inputs
                                    for name, tensor in request.inputs.items():
                                        device_inputs[name] = tensor.to(self.device)
                                    _enter_forward(self, request, device_inputs)
                                    output = self._model(**device_inputs)
                                    self._live_buffers[2] = output
                                    _returned_forward()
                                    if isinstance(output, Mapping) and output.get("past_key_values") is not None:
                                        raise WorkerStateError("model returned an unsupported key/value cache")
                                    logits = output if type(output) is torch.Tensor else output.get("logits") if isinstance(output, Mapping) else None
                                    self._live_buffers[3] = logits
                                    if type(logits) is not torch.Tensor or not logits.is_floating_point() or logits.ndim != 3:
                                        raise ValueError("model must return rank-three floating logits")
                                    if tuple(logits.shape[:2]) != tuple(request.inputs["input_ids"].shape):
                                        raise ValueError("logits must match input batch and sequence dimensions")
                                    if logits.numel() * logits.element_size() > self.max_output_bytes:
                                        raise OutputLimitExceeded("logits exceed the configured output byte bound")
                                    with torch.inference_mode(False):
                                        outcome = logits.detach().to(device="cpu", copy=True,
                                                                     memory_format=torch.contiguous_format)
                                    self._live_buffers[4] = outcome
                                    if (not outcome.is_contiguous() or outcome.storage_offset() != 0
                                            or outcome.untyped_storage().nbytes()
                                            != outcome.numel() * outcome.element_size()):
                                        raise WorkerExecutionError("copied logits do not own exact contiguous storage")
                            finally:
                                if devices:
                                    sync_attempted = True
                                    try:
                                        torch.cuda.synchronize(self.device)
                                        _finish_forward(True)
                                        self._drain_pending = False
                                    except BaseException:
                                        quarantine()
                                else:
                                    _finish_forward(True)
                    finally:
                        if devices and not sync_attempted:
                            try:
                                torch.cuda.synchronize(self.device)
                                _finish_forward(True)
                                self._drain_pending = False
                            except BaseException:
                                quarantine()
                finally:
                    # No peer may pass the shared execution lock while a
                    # device drain remains unverified and unlatchable.
                    if self._drain_pending:
                        quarantine()
        except BaseException as exc:
            # The original synchronization failure remains authoritative even
            # if publishing quarantine was itself interrupted once. Recover
            # the local references before this frame can unwind.
            if self._drain_pending:
                quarantine()
            if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                failure = (WorkerOutOfMemory, _describe_failure(exc, "allocation failed"))
            elif isinstance(exc, OutputLimitExceeded):
                failure = (OutputLimitExceeded, _describe_failure(exc, "output byte bound exceeded"))
            elif isinstance(exc, WorkerStateError):
                failure = (WorkerStateError, _describe_failure(exc, "model state contract violated"))
            elif isinstance(exc, Exception):
                failure = (WorkerExecutionError, _describe_failure(exc, "model execution failed"))
            else:
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                interrupt = exc
        _result_forward(failure, sync_failure is not None, interrupt is not None)
        if sync_failure is not None:
            return None, (WorkerSynchronizationError, f"device synchronization failed: {sync_failure}")
        if interrupt is not None:
            raise interrupt
        return outcome, failure

    def execute(self, request, cancel_event=None):
        """Execute a validated request, cooperatively discarding cancelled work."""
        if _is_process_quarantined():
            request = None
            raise WorkerSynchronizationError("execution process is quarantined")
        if cancel_event is not None and type(cancel_event) is not threading.Event:
            request = None
            raise InvalidRequest("cancel_event must be a threading.Event")
        if cancel_event is not None and cancel_event.is_set():
            request = None
            raise RequestCancelled("request cancelled before execution")
        if self._closed or self.poisoned:
            request = None
            raise WorkerUnavailable("worker is closed or poisoned")
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            request = None
            raise WorkerConfigurationError("ambient autocast is incompatible with resident execution")
        snapshot_error = None
        try:
            snapshot = self.snapshot_request(request)
        except (InvalidRequest, WorkerOutOfMemory, WorkerSynchronizationError) as exc:
            snapshot_error = (type(exc), str(exc))
        except BaseException as exc:
            request = None
            exc.__traceback__ = None
            exc.__context__ = None
            exc.__cause__ = None
            raise exc
        if snapshot_error is not None:
            request = None
            raise snapshot_error[0](snapshot_error[1]) from None
        request = None
        if cancel_event is not None and cancel_event.is_set():
            snapshot = None
            raise RequestCancelled("request cancelled before execution")
        with self._lock:
            if self._closed or self.poisoned:
                snapshot = None
                raise WorkerUnavailable("worker is closed or poisoned")
            if cancel_event is not None and cancel_event.is_set():
                snapshot = None
                raise RequestCancelled("request cancelled before execution")
            try:
                state_ok = self._model_state() == self._baseline
            except Exception:
                state_ok = False
            except BaseException as exc:
                snapshot = None
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                self._poison_pending.set()
                self._poison()
                raise exc
            if not state_ok:
                snapshot = None
                self._poison_pending.set()
                self._poison()
                raise WorkerUnavailable("model state changed outside the worker")
            try:
                logits, failure = self._perform(snapshot)
            except BaseException as exc:
                snapshot = None
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                self._poison_pending.set()
                self._poison()
                raise exc
            # An unverified CUDA drain is more serious than any state check.
            # The process-wide latch already prevents subsequent work.
            if failure is not None and failure[0] is WorkerSynchronizationError:
                snapshot = None
                logits = None
                self._poison_pending.set()
                self._poison()
                raise failure[0](failure[1]) from None
            # A drained failed forward may still have mutated model state. Check
            # before dropping the model so its owner can quarantine the lease.
            try:
                state_ok = self._model_state() == self._baseline
            except Exception:
                state_ok = False
            except BaseException as exc:
                snapshot = None
                logits = None
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                self._poison_pending.set()
                self._poison()
                raise exc
            if not state_ok:
                snapshot = None
                logits = None
                self._poison_pending.set()
                self._poison()
                detail = "model state changed or could not be verified during forward"
                if failure is not None:
                    detail += f"; original failure: {failure[0].__name__}: {failure[1]}"
                raise WorkerStateError(detail) from None
            if self.poisoned:
                snapshot = None
                logits = None
                self._poison()
                raise WorkerUnavailable("worker was poisoned during execution")
            if failure is not None:
                snapshot = None
                logits = None
                if failure[0] is not OutputLimitExceeded:
                    self._poison_pending.set()
                    self._poison()
                raise failure[0](failure[1]) from None
            if cancel_event is not None and cancel_event.is_set():
                snapshot = None
                logits = None
                raise RequestCancelled("request cancelled after execution")
            return ForwardResult(snapshot.request_id, logits)
