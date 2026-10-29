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


class InvalidRequest(ValueError):
    """A request is outside the supported, bounded forward contract."""


class RequestCancelled(RuntimeError):
    """The request was cancelled before execution or after device completion."""


class WorkerUnavailable(RuntimeError):
    """The model has been closed or its state can no longer be trusted."""


class WorkerExecutionError(RuntimeError):
    """A model forward or its output violated the execution contract."""


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
# Keep known live resources until the owner terminates this execution process.
# Model internals may have released additional temporaries before an error.
_quarantine_references = []
_mapping_proxy_type = type(MappingProxyType({}))
_input_names = frozenset(("input_ids", "attention_mask", "token_type_ids", "position_ids"))


class ResidentWorker:
    """Own a preconfigured model and execute at most one forward at a time."""

    cancellation_exception = RequestCancelled

    def __init__(self, model_id, model, *, device="cpu", max_batch_size=1,
                 max_sequence_length=512, max_input_bytes=1024 * 1024,
                 max_output_bytes=8 * 1024 * 1024):
        if _process_quarantined.is_set():
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
        self.model_id = model_id
        self.device = resolved_device
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self._lock = threading.Lock()
        self._model = model
        self._poison_pending = threading.Event()
        self._closed = False
        self._baseline = self._model_state()

    @property
    def poisoned(self):
        return self._poison_pending.is_set() or _process_quarantined.is_set()

    @property
    def quarantined(self):
        """A device drain failed; this execution process requires teardown."""
        return _process_quarantined.is_set()

    def _model_state(self):
        global_hooks = nn.modules.module
        if any(getattr(global_hooks, name, None) for name in (
                "_global_forward_hooks", "_global_forward_pre_hooks",
                "_global_backward_hooks", "_global_backward_pre_hooks")):
            raise ValueError("process-global module hooks are not supported")
        modules = []
        for name, module in self._model.named_modules():
            if module.training:
                raise ValueError("every model module must be in evaluation mode")
            if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                    or module._backward_pre_hooks):
                raise ValueError("model hooks are not supported by this worker")
            modules.append((name, id(module)))
        tensors = []
        for kind, collection in (("parameter", self._model.named_parameters()),
                                 ("buffer", self._model.named_buffers())):
            for name, tensor in collection:
                if tensor.device != self.device:
                    raise ValueError("all model parameters and buffers must be on the worker device")
                try:
                    version = tensor._version
                except RuntimeError as exc:
                    raise ValueError("model tensors must track in-place changes") from exc
                tensors.append((kind, name, id(tensor), version, tuple(tensor.shape),
                                str(tensor.dtype), str(tensor.device)))
        return tuple(modules), tuple(tensors)

    def validate_request(self, request):
        """Validate without copying inputs or modifying model state."""
        if type(request) is not ForwardRequest:
            raise InvalidRequest("request must be a ForwardRequest")
        if type(request.request_id) is not str or not request.request_id or len(request.request_id) > 256:
            raise InvalidRequest("request_id must be a nonempty string of at most 256 characters")
        if type(request.seed) is not int or not 0 <= request.seed < 2**63:
            raise InvalidRequest("seed must be an integer in [0, 2**63)")
        if type(request.inputs) not in (dict, _mapping_proxy_type):
            raise InvalidRequest("inputs must be a plain tensor dictionary")
        if any(type(name) is not str for name in request.inputs):
            raise InvalidRequest("input names must be plain strings")
        if "input_ids" not in request.inputs or not set(request.inputs).issubset(_input_names):
            raise InvalidRequest("inputs must contain input_ids and only supported names")
        shape = None
        total_bytes = 0
        for name, value in request.inputs.items():
            if type(value) is not torch.Tensor or value.device.type != "cpu" or value.layout != torch.strided:
                raise InvalidRequest("inputs must be ordinary dense CPU tensors")
            allowed = (torch.int32, torch.int64, torch.bool) if name == "attention_mask" else (torch.int32, torch.int64)
            if value.dtype not in allowed or value.ndim != 2:
                raise InvalidRequest("input tensors must be two-dimensional integer token tensors")
            if shape is None:
                shape = tuple(value.shape)
            elif shape != tuple(value.shape):
                raise InvalidRequest("all input tensors must have the same shape")
            total_bytes += value.numel() * value.element_size()
        if not (0 < shape[0] <= self.max_batch_size and 0 < shape[1] <= self.max_sequence_length):
            raise InvalidRequest("input batch or sequence length exceeds its bound")
        if total_bytes > self.max_input_bytes:
            raise InvalidRequest("input tensor bytes exceed their bound")

    def snapshot_request(self, request):
        """Make a bounded CPU snapshot; caller must not mutate during copying."""
        if _process_quarantined.is_set():
            raise WorkerSynchronizationError("execution process is quarantined")
        self.validate_request(request)
        oom = None
        try:
            inputs = {name: tensor.detach().clone(memory_format=torch.contiguous_format)
                      for name, tensor in request.inputs.items()}
        except (MemoryError, torch.OutOfMemoryError) as exc:
            self._poison_pending.set()
            oom = str(exc)
        if oom is not None:
            with self._lock:
                self._poison()
            request = None
            raise WorkerOutOfMemory(f"input snapshot failed: {oom}") from None
        return ForwardRequest(request.request_id, MappingProxyType(inputs), request.seed)

    def _poison(self):
        self._poison_pending.set()
        self._release_model()

    def _release_model(self):
        if _process_quarantined.is_set() and self._model is not None:
            try:
                _quarantine_references.append(self._model)
            except BaseException:
                pass
        self._model = None

    def close(self):
        """Wait for an active call and release the worker's model reference."""
        with self._lock:
            self._closed = True
            self._release_model()

    def _perform(self, request):
        """Return outcome data, not exceptions carrying model/tensor tracebacks."""
        outcome, failure, sync_failure, interrupt = None, None, None, None
        device_inputs, output, logits = None, None, None
        devices = [self.device.index] if self.device.type == "cuda" else []
        sync_attempted = False

        def quarantine():
            nonlocal sync_failure
            # Do this before diagnostics or retention: both can fail in an OOM.
            sync_failure = "target device synchronization failed"
            _process_quarantined.set()
            try:
                _quarantine_references.append((self._model, request, device_inputs, output, logits, outcome))
            except BaseException:
                # The process must still be quarantined when reference retention fails.
                pass

        try:
            with _rng_lock:
                if _process_quarantined.is_set():
                    return None, (WorkerSynchronizationError, "execution process is quarantined")
                try:
                    with torch.random.fork_rng(devices=devices):
                        try:
                            torch.random.default_generator.manual_seed(request.seed)
                            if devices:
                                torch.cuda.default_generators[self.device.index].manual_seed(request.seed)
                            with torch.inference_mode():
                                device_inputs = {name: tensor.to(self.device) for name, tensor in request.inputs.items()}
                                output = self._model(**device_inputs)
                                logits = output if type(output) is torch.Tensor else output.get("logits") if isinstance(output, Mapping) else None
                                if type(logits) is not torch.Tensor or not logits.is_floating_point() or logits.ndim != 3:
                                    raise ValueError("model must return rank-three floating logits")
                                if tuple(logits.shape[:2]) != tuple(request.inputs["input_ids"].shape):
                                    raise ValueError("logits must match input batch and sequence dimensions")
                                if logits.numel() * logits.element_size() > self.max_output_bytes:
                                    raise OutputLimitExceeded("logits exceed the configured output byte bound")
                                with torch.inference_mode(False):
                                    outcome = logits.detach().to(device="cpu", copy=True)
                        finally:
                            if devices:
                                sync_attempted = True
                                try:
                                    torch.cuda.synchronize(self.device)
                                except BaseException:
                                    quarantine()
                finally:
                    if devices and not sync_attempted:
                        try:
                            torch.cuda.synchronize(self.device)
                        except BaseException:
                            quarantine()
        except BaseException as exc:
            if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                failure = (WorkerOutOfMemory, f"allocation failed: {exc}")
            elif isinstance(exc, OutputLimitExceeded):
                failure = (OutputLimitExceeded, str(exc))
            elif isinstance(exc, Exception):
                failure = (WorkerExecutionError, f"{type(exc).__name__}: {exc}")
            else:
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                interrupt = exc
        if sync_failure is not None:
            return None, (WorkerSynchronizationError, f"device synchronization failed: {sync_failure}")
        if interrupt is not None:
            raise interrupt
        return outcome, failure

    def execute(self, request, cancel_event=None):
        """Execute a validated request, cooperatively discarding cancelled work."""
        if _process_quarantined.is_set():
            request = None
            raise WorkerSynchronizationError("execution process is quarantined")
        if cancel_event is not None and type(cancel_event) is not threading.Event:
            raise InvalidRequest("cancel_event must be a threading.Event")
        if cancel_event is not None and cancel_event.is_set():
            request = None
            raise RequestCancelled("request cancelled before execution")
        if self._closed or self.poisoned:
            request = None
            raise WorkerUnavailable("worker is closed or poisoned")
        snapshot = self.snapshot_request(request)
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
            if not state_ok:
                self._poison()
                snapshot = None
                raise WorkerUnavailable("model state changed outside the worker")
            try:
                logits, failure = self._perform(snapshot)
            except BaseException as exc:
                self._poison()
                snapshot = None
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                raise exc
            if failure is not None and failure[0] is not OutputLimitExceeded:
                self._poison()
                snapshot = None
                logits = None
                raise failure[0](failure[1]) from None
            if self.poisoned:
                self._poison()
                snapshot = None
                logits = None
                raise WorkerUnavailable("worker was poisoned during execution")
            try:
                state_ok = self._model_state() == self._baseline
            except Exception:
                state_ok = False
            if not state_ok:
                self._poison()
                snapshot = None
                logits = None
                raise WorkerExecutionError("model changed registered state during forward")
            if failure is not None:
                snapshot = None
                logits = None
                raise failure[0](failure[1]) from None
            if cancel_event is not None and cancel_event.is_set():
                snapshot = None
                logits = None
                raise RequestCancelled("request cancelled after execution")
            return ForwardResult(snapshot.request_id, logits)
