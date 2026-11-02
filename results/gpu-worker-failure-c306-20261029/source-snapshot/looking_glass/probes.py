"""Identity observation points and two equivalent value-reporting channels.

The hook always returns None. Recording never replaces the main-path tensor.
Sessions are local, single-use and single-threaded. They are not a general
profiler and do not support concurrent use on the same model.
"""
from collections import defaultdict
from dataclasses import dataclass
from threading import RLock, current_thread
from typing import Optional

import torch
from torch import nn

from .model_ownership import ModelLease


def _validate_tensor(a):
    if not isinstance(a, torch.Tensor) or not a.is_floating_point():
        raise TypeError("capture requires one floating-point tensor")
    if not bool(torch.isfinite(a).all()):
        raise ValueError("capture requires finite activations")


def gradient_readout(a: torch.Tensor) -> torch.Tensor:
    """Return a detached copy via grad_q sum(stopgrad(a)*q), where q=0.

    The separate q is ephemeral, not a model parameter, and never optimized.
    Local contexts also permit this isolated computation during inference mode.
    """
    _validate_tensor(a)
    with torch.inference_mode(False), torch.enable_grad():
        saved = a.detach().clone()
        q = torch.zeros_like(saved, requires_grad=True)
        report = (saved * q).sum()
        (value,) = torch.autograd.grad(report, q, create_graph=False)
        return value.detach().clone()


class TapPoint(nn.Identity):
    """A genuine identity operation with an explicit instrumentation label."""
    def __init__(self, node_id: str, group: str = "dense"):
        super().__init__()
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a nonempty string")
        if group not in ("sparse", "dense"):
            raise ValueError("group must be sparse or dense")
        self.node_id = node_id
        self.group = group


@dataclass
class TraceEvent:
    run_id: str
    node_id: str
    sequence: int
    occurrence: int
    shape: tuple
    dtype: str
    device: str
    value: Optional[torch.Tensor]

    @property
    def tensor_bytes(self) -> int:
        return 0 if self.value is None else self.value.numel() * self.value.element_size()

    def metadata(self) -> dict:
        return {"run_id": self.run_id, "node_id": self.node_id,
                "sequence": self.sequence, "occurrence": self.occurrence,
                "shape": list(self.shape), "dtype": self.dtype,
                "device": self.device, "tensor_bytes": self.tensor_bytes}


class TraceSession:
    """Record labeled identity outputs, without replacing values or gradients.

    max_events is an explicit bound; exceeding it raises rather than silently
    dropping events. max_tensor_bytes, when set, is a cumulative allowance for
    the logical payload bytes of successfully captured tensor snapshots. The
    configured limit is read-only for the session. The allowance
    does not replenish if the caller removes or modifies events, and it does
    not account for tensors inserted or resized by the caller. ID-only events
    consume no tensor bytes. Bounded value capture requires non-nested strided
    tensors, whose logical payload can be estimated before copying.

    The session holds a process-local lease on the registered model tree while
    hooks are installed. Cooperating hook sessions with overlapping modules are
    rejected. Changing the registered module tree during a session is not
    supported; validation at forward and capture boundaries detects changes.
    Root forwards and captured tap calls must run on the entering thread. After
    a detected capture, forward, validation or cleanup failure, ``failed``
    remains true; further traced forwards and normal context exit are rejected.
    PyTorch can skip completion hooks for BaseException, delaying detection
    until exit; callers must stop use immediately after such an interruption.
    Events retained after a failure are incomplete evidence.

    The allowance is not a RAM or VRAM peak bound: it excludes metadata,
    allocator rounding and reservations, model state and activations, finite
    validation temporaries, and gradient-readout intermediates. tensor_bytes
    reports the current logical payload bytes in the public events list.
    """
    def __init__(self, model: nn.Module, mode="direct", placement="dense",
                 run_id="run", max_events=10000, max_tensor_bytes=None):
        if mode not in ("direct", "gradient", "id"):
            raise ValueError("mode must be direct, gradient, or id")
        if placement not in ("sparse", "dense"):
            raise ValueError("placement must be sparse or dense")
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        if max_tensor_bytes is not None and (
            type(max_tensor_bytes) is not int or max_tensor_bytes < 0
        ):
            raise ValueError("max_tensor_bytes must be a nonnegative integer or None")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        self.model, self.mode, self.placement = model, mode, placement
        self.run_id, self.max_events = run_id, max_events
        self._max_tensor_bytes = max_tensor_bytes
        self._captured_tensor_bytes = 0
        self.events = []
        self._handles = []
        self._pending_handle = None
        self._registration_uncertain = False
        self._lease = None
        self._counts = defaultdict(int)
        self._used = False
        self._failed = False
        self._forward_stack = []
        self._owner_thread = None
        self._entry_owner = None
        self._context_active = False
        self._entering = False
        self._closing = False
        self._lifecycle_lock = RLock()

    @property
    def tensor_bytes(self):
        return sum(e.tensor_bytes for e in self.events)

    @property
    def max_tensor_bytes(self):
        return self._max_tensor_bytes

    @property
    def failed(self):
        """Whether any session operation or traced forward has failed."""
        return self._failed

    def __enter__(self):
        entry_owner = object()
        try:
            with self._lifecycle_lock:
                if self._used:
                    raise RuntimeError("TraceSession is single-use")
                self._used = True
                self._entry_owner = entry_owner
                self._owner_thread = current_thread()
                self._context_active = True
                self._entering = True
            lease = ModelLease(self.model)
            self._lease = lease
            lease.acquire()
            lease.verify_tree()
            points = [m for m in self.model.modules() if isinstance(m, TapPoint)
                      and (self.placement == "dense" or m.group == "sparse")]
            labels = [m.node_id for m in points]
            if len(set(labels)) != len(labels):
                raise ValueError("distinct tap modules must have distinct node IDs")
            self._install_hook(self.model.register_forward_pre_hook, self._before_forward)
            for module in points:
                self._install_hook(module.register_forward_hook, self._capture)
            self._install_hook(self.model.register_forward_hook, self._after_forward)
            self._install_hook(
                lambda callback: self.model.register_forward_hook(callback, always_call=True),
                self._finish_forward,
            )
            lease.verify_tree()
            with self._lifecycle_lock:
                if self._failed:
                    raise RuntimeError("TraceSession failed during entry")
                self._entering = False
            return self
        except BaseException as error:
            if self._entry_owner is entry_owner:
                self._failed = True
                try:
                    self._cleanup(verify_tree=False)
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                finally:
                    with self._lifecycle_lock:
                        self._entering = False
                        self._context_active = False
            raise

    def _install_hook(self, register, callback):
        # Registration may mutate a hook dictionary before returning a handle.
        # If it raises first, public APIs cannot prove whether a hook remains.
        self._registration_uncertain = True
        handle = register(callback)
        if not callable(getattr(handle, "remove", None)):
            raise RuntimeError("hook registration returned an invalid removal handle")
        # Keep the returned handle in a preexisting slot until list insertion
        # succeeds, including if recording the handle raises an OOM error.
        self._pending_handle = handle
        self._registration_uncertain = False
        self._handles.append(handle)
        self._pending_handle = None

    def _before_forward(self, module, inputs):
        try:
            with self._lifecycle_lock:
                self._check_thread()
                if self._closing or self._entering or not self._context_active:
                    raise RuntimeError("TraceSession is not active for a forward")
                self._forward_stack.append(False)
                if self._failed:
                    raise RuntimeError("TraceSession failed; its events are incomplete")
            self._lease.verify_tree()
        except BaseException:
            self._failed = True
            raise

    def _after_forward(self, module, inputs, output):
        try:
            with self._lifecycle_lock:
                self._check_thread()
                if self._failed:
                    raise RuntimeError("TraceSession failed; its events are incomplete")
            self._lease.verify_tree()
            with self._lifecycle_lock:
                if self._failed:
                    raise RuntimeError("TraceSession failed; its events are incomplete")
                self._forward_stack[-1] = True
        except BaseException:
            self._failed = True
            raise
        return None

    def _finish_forward(self, module, inputs, output):
        with self._lifecycle_lock:
            if current_thread() is not self._owner_thread:
                self._failed = True
                return None
            if not self._forward_stack or not self._forward_stack.pop():
                self._failed = True
        return None

    def _capture(self, module, inputs, output):
        try:
            with self._lifecycle_lock:
                self._check_thread()
                if self._closing or self._entering or not self._context_active:
                    raise RuntimeError("TraceSession is not active for capture")
                if self._failed:
                    raise RuntimeError("TraceSession failed; its events are incomplete")
            self._lease.verify_tree()
            return self._capture_value(module, inputs, output)
        except BaseException:
            self._failed = True
            raise

    def _capture_value(self, module, inputs, output):
        if len(self.events) >= self.max_events:
            raise RuntimeError("trace capacity exceeded; increase max_events explicitly")
        if not isinstance(output, torch.Tensor):
            raise TypeError("tap output must be one tensor")
        value_bytes = 0
        if self.mode != "id" and self.max_tensor_bytes is not None:
            if not output.is_floating_point():
                raise TypeError("capture requires one floating-point tensor")
            if output.layout != torch.strided or output.is_nested:
                raise ValueError("bounded value capture requires non-nested strided tensors")
            value_bytes = output.numel() * output.element_size()
            if self._captured_tensor_bytes + value_bytes > self.max_tensor_bytes:
                raise RuntimeError(
                    "trace tensor capacity exceeded: "
                    f"{self._captured_tensor_bytes} captured + {value_bytes} requested "
                    f"exceeds max_tensor_bytes={self.max_tensor_bytes}"
                )
        if self.mode == "id":
            value = None
        elif self.mode == "gradient":
            value = gradient_readout(output)
        else:
            _validate_tensor(output)
            value = output.detach().clone()
        count = self._counts[module.node_id]
        self._counts[module.node_id] += 1
        self.events.append(TraceEvent(self.run_id, module.node_id,
            len(self.events), count, tuple(output.shape), str(output.dtype),
            str(output.device), value))
        if self.max_tensor_bytes is not None:
            self._captured_tensor_bytes += value_bytes
        return None

    def _check_thread(self):
        if self._owner_thread is not None and current_thread() is not self._owner_thread:
            self._failed = True
            raise RuntimeError("TraceSession must be used on its entering thread")

    def _check_close(self):
        if self._entering:
            self._failed = True
            raise RuntimeError("cannot close TraceSession during session entry")
        if self._closing:
            self._failed = True
            raise RuntimeError("TraceSession cleanup is already in progress")
        if self._forward_stack:
            self._failed = True
            raise RuntimeError("cannot close TraceSession during an active forward")
        if (self._context_active and self._owner_thread is not None and
                current_thread() is not self._owner_thread):
            self._failed = True
            raise RuntimeError("active TraceSession must be closed on its entering thread")

    def _cleanup(self, *, verify_tree):
        verification_error = None
        if verify_tree and self._lease is not None:
            try:
                self._lease.verify_tree()
            except BaseException as error:
                self._failed = True
                verification_error = error
        removal_error = None
        if self._pending_handle is not None:
            # An asynchronous interruption may have arrived just after the
            # returned handle was stored but before uncertainty was cleared.
            self._registration_uncertain = False
        # A signal can arrive after append succeeds but before the pending slot
        # is cleared. In that case the handle is already tracked in the list.
        if self._pending_handle is not None:
            for handle in self._handles:
                if handle is self._pending_handle:
                    self._pending_handle = None
                    break
        if self._pending_handle is not None:
            try:
                self._pending_handle.remove()
            except BaseException as error:
                self._failed = True
                removal_error = error
            else:
                self._pending_handle = None
        # Delete only successfully removed handles in place, so a memory error
        # cannot lose a handle while assembling a separate remaining list.
        for index in range(len(self._handles) - 1, -1, -1):
            try:
                self._handles[index].remove()
            except BaseException as error:
                self._failed = True
                if removal_error is None:
                    removal_error = error
            else:
                del self._handles[index]
        # Keep ownership if any hook might still be installed. A caller that
        # retains the session may retry close() after resolving the failure.
        if (not self._handles and self._pending_handle is None and
                not self._registration_uncertain and self._lease is not None):
            self._lease.release()
            self._lease = None
        if self._registration_uncertain:
            self._failed = True
            error = RuntimeError("hook registration may have left an unknown hook; "
                                 "discard this model or worker")
            if removal_error is not None:
                raise error from removal_error
            if verification_error is not None:
                raise error from verification_error
            raise error
        if verification_error is not None:
            if removal_error is not None:
                raise verification_error from removal_error
            raise verification_error
        if removal_error is not None:
            raise removal_error

    def close(self):
        closing_owner = object()
        try:
            try:
                with self._lifecycle_lock:
                    self._check_close()
                    self._closing = closing_owner
                self._cleanup(verify_tree=True)
            finally:
                with self._lifecycle_lock:
                    if self._closing is closing_owner:
                        self._closing = False
        except BaseException:
            self._failed = True
            raise

    def __exit__(self, exc_type, exc, tb):
        closing_owner = object()
        try:
            with self._lifecycle_lock:
                self._check_thread()
                self._check_close()
                self._closing = closing_owner
                if exc_type is not None:
                    self._failed = True
            self._cleanup(verify_tree=exc_type is None)
        except BaseException as cleanup_error:
            self._failed = True
            if exc is not None and current_thread() is self._owner_thread:
                raise exc.with_traceback(tb) from cleanup_error
            raise
        finally:
            with self._lifecycle_lock:
                if self._closing is closing_owner:
                    self._context_active = False
                    self._closing = False
                failed = self._failed
        if exc_type is None and failed:
            raise RuntimeError("TraceSession failed; its events are incomplete")
        return False
