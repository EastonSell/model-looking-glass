"""Bounded single-forward activation capture and intervention primitives.

These hooks require exclusive ownership of a model for the duration of the
session; they are not a request-local dispatcher for a shared model broker.
"""

from copy import deepcopy
from threading import Lock, current_thread

import torch
from torch import nn

from .model_ownership import ModelLease


_DEFAULT_MAX_TENSOR_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_INVOCATIONS = 1024
def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _validate_tensor(tensor, max_tensor_bytes):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError("activation must be a floating-point tensor")
    if _tensor_bytes(tensor) > max_tensor_bytes:
        raise ValueError("activation exceeds max_tensor_bytes")
    try:
        finite = bool(torch.isfinite(tensor).all())
    except (RuntimeError, NotImplementedError) as error:
        raise ValueError("activation cannot be checked for finite values") from error
    if not finite:
        raise ValueError("activation must contain only finite values")


class _ActivationSession:
    def __init__(self, model, module_path, *, invocation, batch_index, token_index,
                 max_tensor_bytes, max_invocations):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(module_path, str):
            raise TypeError("module_path must be a string")
        _nonnegative_int(invocation, "invocation")
        _positive_int(max_tensor_bytes, "max_tensor_bytes")
        _positive_int(max_invocations, "max_invocations")
        if invocation >= max_invocations:
            raise ValueError("invocation must be below max_invocations")
        if (batch_index is None) != (token_index is None):
            raise ValueError("batch_index and token_index must both be set or omitted")
        if batch_index is not None:
            _nonnegative_int(batch_index, "batch_index")
            _nonnegative_int(token_index, "token_index")
        self.model = model
        self.module_path = module_path
        self.module = model.get_submodule(module_path)
        self.invocation = invocation
        self.batch_index = batch_index
        self.token_index = token_index
        self.max_tensor_bytes = max_tensor_bytes
        self.max_invocations = max_invocations
        self._lifecycle_lock = Lock()
        self._used = False
        self._entry_token = None
        self._entering = False
        self._entered = False
        self._closing = False
        self._owner_thread = None
        self._lease = ModelLease(model)
        self._aliases = ()
        # Allocate all bookkeeping before registering any hook. Appending a
        # returned handle can otherwise fail after the hook was installed.
        self._handles = [None] * 4
        self._registration_uncertain = False
        self._in_forward = False
        self._forward_count = 0
        self._invocations_seen = 0
        self._completed = False
        self._applied = False
        self._invalid = False
        self._committed = False
        self._metadata = None

    def __enter__(self):
        token = object()
        try:
            with self._lifecycle_lock:
                if self._used:
                    raise RuntimeError("activation sessions are single-use")
                self._entry_token = token
                self._entering = True
                self._used = True
                self._owner_thread = current_thread()
            try:
                current_target = self.model.get_submodule(self.module_path)
            except AttributeError as error:
                raise RuntimeError("module path changed before session entry") from error
            if current_target is not self.module:
                raise RuntimeError("module path changed before session entry")
            self._lease.acquire()
            self._verify_model_tree()
            self._aliases = tuple(path for path, module in self._lease.registered_paths
                                  if module is self.module)
            if not self._aliases:
                raise RuntimeError("target module is not in the leased module tree")
            self._register(0, self.model.register_forward_pre_hook, self._before_forward)
            self._register(1, self.module.register_forward_hook, self._target_hook)
            self._register(2, self.model.register_forward_hook, self._complete_forward)
            self._register(3, self.model.register_forward_hook, self._finish_forward, always_call=True)
            with self._lifecycle_lock:
                if self._invalid:
                    raise RuntimeError("activation session was invalidated during entry")
                self._entered = True
                self._entering = False
            return self
        except BaseException as error:
            with self._lifecycle_lock:
                owns_entry = self._entry_token is token
                if owns_entry:
                    self._entering = True
                    self._used = True
            if not owns_entry:
                raise
            try:
                try:
                    self._cleanup()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            finally:
                with self._lifecycle_lock:
                    self._entering = False
                    self._entered = False
            raise

    def _register(self, slot, register, hook, **kwargs):
        # PyTorch or a custom module can install a hook, then raise before
        # returning its handle. Without that handle cleanup cannot prove safety.
        self._registration_uncertain = True
        handle = register(hook, **kwargs)
        if not callable(getattr(handle, "remove", None)):
            raise TypeError("hook registration returned an invalid handle")
        self._handles[slot] = handle
        self._registration_uncertain = False

    def _verify_model_tree(self):
        try:
            self._lease.verify_tree()
        except RuntimeError as error:
            self._mark_invalid()
            raise RuntimeError("model module tree changed during the session") from error
        try:
            target = self.model.get_submodule(self.module_path)
        except AttributeError as error:
            self._mark_invalid()
            raise RuntimeError("module path changed during the session") from error
        if target is not self.module:
            self._mark_invalid()
            raise RuntimeError("module path changed during the session")

    def _before_forward(self, module, inputs):
        self._check_owner()
        if self._forward_count != 0 or self._in_forward:
            self._mark_invalid()
            raise RuntimeError("session permits exactly one root model forward")
        self._verify_model_tree()
        self._forward_count = 1
        self._in_forward = True

    def _complete_forward(self, module, inputs, output):
        self._check_owner()
        self._verify_model_tree()
        self._completed = True
        return None

    def _finish_forward(self, module, inputs, output):
        if current_thread() is not self._owner_thread:
            self._mark_invalid()
            return None
        if not self._completed:
            self._mark_invalid()
        self._in_forward = False
        return None

    def _target_hook(self, module, inputs, output):
        try:
            self._check_owner()
            if not self._in_forward:
                raise RuntimeError("target module must run within the root model forward")
            self._verify_model_tree()
            count = self._invocations_seen
            self._invocations_seen += 1
            if count >= self.max_invocations:
                raise RuntimeError("max_invocations exceeded")
            if count != self.invocation:
                return None
            if isinstance(output, torch.Tensor):
                tensor = output
            elif type(output) is tuple and output and isinstance(output[0], torch.Tensor):
                tensor = output[0]
            else:
                raise TypeError("module output must be a tensor or a tuple beginning with a tensor")
            _validate_tensor(tensor, self.max_tensor_bytes)
            if self.batch_index is None:
                selected = tensor
            else:
                if tensor.ndim != 3:
                    raise ValueError("token selection requires [batch, token, hidden] output")
                if self.batch_index >= tensor.shape[0] or self.token_index >= tensor.shape[1]:
                    raise IndexError("batch_index or token_index is out of range")
                selected = tensor[self.batch_index, self.token_index]
            replacement = self._apply(output, tensor, selected)
            self._metadata = {
                "module_path": self.module_path,
                "module_aliases": list(self._aliases),
                "call_site_known": False,
                "forward_index": 0,
                "invocation": count,
                "batch_index": self.batch_index,
                "token_index": self.token_index,
                "output_shape": list(tensor.shape),
                "selected_shape": list(selected.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            self._applied = True
            return replacement
        except BaseException:
            self._mark_invalid()
            raise

    def _mark_invalid(self):
        with self._lifecycle_lock:
            self._invalid = True
            self._committed = False

    def _check_owner(self):
        with self._lifecycle_lock:
            if current_thread() is not self._owner_thread:
                self._invalid = True
                self._committed = False
                raise RuntimeError("activation session must run in its owning thread")
            if self._invalid:
                raise RuntimeError("activation session is already invalid")

    @property
    def metadata(self):
        if not self._committed:
            raise RuntimeError("session has no successfully completed result")
        return deepcopy(self._metadata)

    def _cleanup(self):
        first_error = None
        for index, handle in enumerate(self._handles):
            if handle is None:
                continue
            try:
                handle.remove()
                self._handles[index] = None
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is None and not self._registration_uncertain:
            self._lease.release()
        if first_error is not None:
            raise first_error
        if self._registration_uncertain:
            raise RuntimeError("hook registration was indeterminate; discard this model and worker")

    def close(self):
        """Retry removal of known handles; indeterminate registration remains quarantined."""
        token = object()
        try:
            with self._lifecycle_lock:
                if not self._used:
                    raise RuntimeError("close is only available after session entry was attempted")
                if self._entering or self._entered or self._closing:
                    raise RuntimeError("cannot close an active activation session")
                self._closing = token
            self._cleanup()
        finally:
            with self._lifecycle_lock:
                if self._closing is token:
                    self._closing = False

    def abort(self):
        """Discard an entered session after the owner has ensured quiescence.

        This is recovery for an interrupted context-entry return, not a way to
        cancel a running forward. Foreign callers invalidate but cannot release.
        """
        with self._lifecycle_lock:
            if current_thread() is not self._owner_thread:
                if self._entering or self._entered or self._closing:
                    self._invalid = True
                    self._committed = False
                raise RuntimeError("only the owning thread can abort an activation session")
            if not self._entered or self._entering or self._closing:
                raise RuntimeError("no entered activation session to abort")
            if self._in_forward:
                self._invalid = True
                self._committed = False
                raise RuntimeError("cannot abort while a forward may be active")
            self._invalid = True
            self._committed = False
        self.__exit__(RuntimeError, None, None)

    def __exit__(self, exc_type, exc, tb):
        token = object()
        verification_error = None
        preparation_error = None
        cleanup_error = None
        success = False
        try:
            with self._lifecycle_lock:
                if self._closing:
                    if current_thread() is not self._owner_thread:
                        self._invalid = True
                        self._committed = False
                    raise RuntimeError("activation session cleanup is in progress")
                if not self._entering and not self._entered:
                    raise RuntimeError("activation session is not in an active context")
                if current_thread() is not self._owner_thread:
                    self._invalid = True
                    raise RuntimeError("activation session must exit in its owning thread")
                if self._entering:
                    raise RuntimeError("activation session is not in an active context")
                self._closing = token
                self._entered = False
            if exc_type is None:
                try:
                    self._verify_model_tree()
                except BaseException as error:
                    verification_error = error
        except BaseException as error:
            with self._lifecycle_lock:
                owns_cleanup = self._closing is token
            if not owns_cleanup:
                raise
            preparation_error = error
        finally:
            with self._lifecycle_lock:
                owns_cleanup = self._closing is token
            if owns_cleanup:
                try:
                    self._cleanup()
                except BaseException as error:
                    cleanup_error = error
                finally:
                    with self._lifecycle_lock:
                        success = (exc_type is None and verification_error is None
                                   and preparation_error is None and cleanup_error is None
                                   and not self._invalid and self._forward_count == 1
                                   and self._completed and self._applied and not self._in_forward)
                        if success:
                            self._committed = True
                        self._entered = False
                        self._closing = False
        if not success:
            self._discard()
            self._metadata = None
            if cleanup_error is not None:
                if exc is not None:
                    raise exc.with_traceback(tb) from cleanup_error
                if preparation_error is not None:
                    raise preparation_error from cleanup_error
                if verification_error is not None:
                    raise verification_error from cleanup_error
                raise cleanup_error
            if preparation_error is not None:
                if exc is not None:
                    raise exc.with_traceback(tb) from preparation_error
                raise preparation_error
            if verification_error is not None:
                raise verification_error
            if exc_type is None:
                raise RuntimeError("session did not complete the requested intervention or capture")
        return False

    def _discard(self):
        pass


class ActivationCapture(_ActivationSession):
    """Capture exactly one module output or one [batch, token] hidden vector.

    The detached snapshot and metadata become available only after one complete,
    successful root-model forward and clean context exit.
    """
    def __init__(self, model, module_path, *, invocation=0, batch_index=None,
                 token_index=None, max_tensor_bytes=_DEFAULT_MAX_TENSOR_BYTES,
                 max_invocations=_DEFAULT_MAX_INVOCATIONS):
        super().__init__(model, module_path, invocation=invocation,
                         batch_index=batch_index, token_index=token_index,
                         max_tensor_bytes=max_tensor_bytes, max_invocations=max_invocations)
        self._value = None

    def _apply(self, output, tensor, selected):
        self._value = selected.detach().clone()
        return None

    def _discard(self):
        self._value = None

    @property
    def value(self):
        if not self._committed:
            raise RuntimeError("capture has no successfully completed value")
        return self._value


class ActivationIntervention(_ActivationSession):
    """Patch, zero, add a fixed delta, or perform a null hook control.

    Supplied values are detached, validated and cloned at construction. A null
    control clones and writes back the current selection without changing it.
    """
    def __init__(self, model, module_path, *, operation, value=None, invocation=0,
                 batch_index=None, token_index=None,
                 max_tensor_bytes=_DEFAULT_MAX_TENSOR_BYTES,
                 max_invocations=_DEFAULT_MAX_INVOCATIONS):
        super().__init__(model, module_path, invocation=invocation,
                         batch_index=batch_index, token_index=token_index,
                         max_tensor_bytes=max_tensor_bytes, max_invocations=max_invocations)
        if operation not in ("patch", "zero", "add", "null"):
            raise ValueError("operation must be patch, zero, add, or null")
        if operation in ("patch", "add"):
            if value is None:
                raise ValueError("patch and add require a value")
            _validate_tensor(value, max_tensor_bytes)
            self._source = value.detach().clone()
        else:
            if value is not None:
                raise ValueError("zero and null do not accept a value")
            self._source = None
        self.operation = operation

    def __enter__(self):
        try:
            return super().__enter__()
        except BaseException:
            self._clear_source_if_inactive()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self._clear_source_if_inactive()

    def _clear_source_if_inactive(self):
        # A rejected nested or foreign call must not mutate the active owner.
        with self._lifecycle_lock:
            if not self._entering and not self._entered and not self._closing:
                self._source = None

    def _apply(self, output, tensor, selected):
        if self._source is not None:
            if self._source.shape != selected.shape:
                raise ValueError("replacement shape must exactly match the selected activation")
            if self._source.dtype != selected.dtype:
                raise TypeError("replacement dtype must match the selected activation")
            if self._source.device != selected.device:
                raise ValueError("replacement device must match the selected activation")
        if self.operation == "patch":
            replacement = self._source
        elif self.operation == "zero":
            replacement = torch.zeros_like(selected)
        elif self.operation == "add":
            replacement = selected + self._source
        else:
            replacement = selected
        _validate_tensor(replacement, self.max_tensor_bytes)
        modified = tensor.clone()
        if self.batch_index is None:
            modified[...] = replacement
        else:
            modified[self.batch_index, self.token_index] = replacement
        if type(output) is tuple:
            return (modified,) + output[1:]
        return modified
