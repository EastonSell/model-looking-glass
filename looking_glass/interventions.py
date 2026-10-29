"""Bounded single-forward activation capture and intervention primitives.

These hooks require exclusive ownership of a model for the duration of the
session; they are not a request-local dispatcher for a shared model broker.
"""

from copy import deepcopy
from threading import Lock
from weakref import WeakSet

import torch
from torch import nn


_DEFAULT_MAX_TENSOR_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_INVOCATIONS = 1024
_MAX_REGISTERED_PATHS = 65536
_active_modules = WeakSet()
_active_lock = Lock()


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _snapshot_tree(model):
    # Include duplicate module paths: aliases carry distinct lookup names even
    # though a forward hook cannot identify the Python call site that used one.
    paths = []
    try:
        for path, module in model.named_modules(remove_duplicate=False):
            if len(paths) >= _MAX_REGISTERED_PATHS:
                raise ValueError("model has too many registered module paths")
            paths.append((path, module))
    except RecursionError as error:
        raise ValueError("cyclic module trees are not supported") from error
    return tuple(paths)


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
        self._used = False
        self._owned_modules = ()
        self._tree = ()
        self._aliases = ()
        self._handles = []
        self._in_forward = False
        self._forward_count = 0
        self._invocations_seen = 0
        self._completed = False
        self._applied = False
        self._invalid = False
        self._committed = False
        self._metadata = None

    def __enter__(self):
        if self._used:
            raise RuntimeError("activation sessions are single-use")
        self._used = True
        tree = _snapshot_tree(self.model)
        modules = tuple(self.model.modules())
        try:
            current_target = self.model.get_submodule(self.module_path)
        except AttributeError as error:
            raise RuntimeError("module path changed before session entry") from error
        if current_target is not self.module or not any(m is self.module for m in modules):
            raise RuntimeError("module path changed before session entry")
        with _active_lock:
            if any(module in _active_modules for module in modules):
                raise RuntimeError("overlapping activation sessions require model isolation")
            for module in modules:
                _active_modules.add(module)
            self._owned_modules = modules
            self._tree = tree
            self._aliases = tuple(path for path, module in tree if module is self.module)
        try:
            self._verify_model_tree()
            self._handles.append(self.model.register_forward_pre_hook(self._before_forward))
            self._handles.append(self.module.register_forward_hook(self._target_hook))
            self._handles.append(self.model.register_forward_hook(self._complete_forward))
            self._handles.append(self.model.register_forward_hook(self._finish_forward, always_call=True))
        except BaseException:
            self._cleanup()
            raise
        return self

    def _verify_model_tree(self):
        try:
            current = _snapshot_tree(self.model)
        except ValueError as error:
            self._invalid = True
            raise RuntimeError("model module tree changed during the session") from error
        if (len(current) != len(self._tree)
                or any(a_path != b_path or a_module is not b_module
                       for (a_path, a_module), (b_path, b_module) in zip(current, self._tree))):
            self._invalid = True
            raise RuntimeError("model module tree changed during the session")
        try:
            target = self.model.get_submodule(self.module_path)
        except AttributeError as error:
            self._invalid = True
            raise RuntimeError("module path changed during the session") from error
        if target is not self.module:
            self._invalid = True
            raise RuntimeError("module path changed during the session")

    def _before_forward(self, module, inputs):
        if self._forward_count != 0 or self._in_forward:
            self._invalid = True
            raise RuntimeError("session permits exactly one root model forward")
        self._verify_model_tree()
        self._forward_count = 1
        self._in_forward = True

    def _complete_forward(self, module, inputs, output):
        self._verify_model_tree()
        self._completed = True
        return None

    def _finish_forward(self, module, inputs, output):
        if not self._completed:
            self._invalid = True
        self._in_forward = False
        return None

    def _target_hook(self, module, inputs, output):
        try:
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
            self._invalid = True
            raise

    @property
    def metadata(self):
        if not self._committed:
            raise RuntimeError("session has no successfully completed result")
        return deepcopy(self._metadata)

    def _cleanup(self):
        try:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
        finally:
            with _active_lock:
                for module in self._owned_modules:
                    _active_modules.discard(module)
                self._owned_modules = ()

    def __exit__(self, exc_type, exc, tb):
        success = (exc_type is None and not self._invalid and self._forward_count == 1
                   and self._completed and self._applied and not self._in_forward)
        self._cleanup()
        if success:
            self._committed = True
        else:
            self._discard()
            self._metadata = None
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
            self._source = None
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
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
