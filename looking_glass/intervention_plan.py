"""Ordered, bounded multi-site activation operations in one model forward.

The model is exclusively leased for cooperating Python hook APIs. CUDA and
cross-process resource lifetimes remain the caller or broker's responsibility.
"""

from copy import deepcopy
from threading import Lock, current_thread

import torch
from torch import nn

from .interventions import _nonnegative_int, _positive_int, _tensor_bytes, _validate_tensor
from .model_ownership import ModelLease


class ActivationPlan:
    """Execute explicitly ordered named operations and publish captures atomically.

    Each step is a dict with name, operation, module_path, optional invocation,
    batch_index, token_index, value, and source_capture. Operations are capture, patch, zero,
    add and null. Same-module/same-invocation operations must be consecutive;
    other sites must execute in the specified model hook-completion order.
    """

    def __init__(self, model, steps, *, max_tensor_bytes=16 * 1024 * 1024,
                 max_capture_bytes=16 * 1024 * 1024, max_source_bytes=16 * 1024 * 1024,
                 max_steps=128, max_invocations=1024, _parent_lease=None):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        for name, value in (("max_tensor_bytes", max_tensor_bytes),
                            ("max_capture_bytes", max_capture_bytes),
                            ("max_source_bytes", max_source_bytes),
                            ("max_steps", max_steps), ("max_invocations", max_invocations)):
            _positive_int(value, name)
        if not isinstance(steps, (tuple, list)) or not steps or len(steps) > max_steps:
            raise ValueError("steps must be a nonempty bounded list")
        self.model = model
        self.max_tensor_bytes = max_tensor_bytes
        self.max_capture_bytes = max_capture_bytes
        self.max_invocations = max_invocations
        self._borrow_release_key = None
        if _parent_lease is None:
            self._lease = ModelLease(model)
        elif type(_parent_lease) is ModelLease and _parent_lease.model is model:
            # Trusted case owners use an exclusive child scope while keeping
            # the same parent reservation across multiple model forwards.
            self._borrow_release_key = object()
            self._lease = _parent_lease._borrow_for_plan(
                self._borrow_release_key, self._borrow_cleanup_complete)
        else:
            raise ValueError("parent lease must be the exact lease for this model")
        self._steps = []
        self._groups = []
        self._group_by_site = {}
        self._modules = {}
        self._sources = {}
        self._source_bytes = 0
        self._captures = {}
        self._capture_bytes = 0
        self._events = []
        self._counts = {}
        self._pending = {}
        self._active_sites = set()
        self._processed_sites = set()
        self._next_group = 0
        self._lifecycle_lock = Lock()
        self._used = False
        self._entry_token = None
        self._entering = False
        self._entered = False
        self._closing = False
        self._owner_thread = None
        self._borrow_cleanup_owner = None
        self._registration_uncertain = False
        self._in_forward = False
        self._sync_forward = None
        self._sync_unsafe = False
        self._sync_hook_pending = False
        self._sync_observation_pending = False
        self._sync_observed = None
        self._forward_count = 0
        self._completed = False
        self._invalid = False
        self._committed = False

        allowed = {"name", "operation", "module_path", "invocation",
                   "batch_index", "token_index", "value", "source_capture"}
        names = set()
        previous_steps = {}
        raw_sources = []
        last_site = None
        for raw in steps:
            if not isinstance(raw, dict) or set(raw) - allowed:
                raise ValueError("each step must contain only supported fields")
            name = raw.get("name")
            path = raw.get("module_path")
            operation = raw.get("operation")
            if not isinstance(name, str) or not name or len(name) > 128 or name in names:
                raise ValueError("step names must be nonempty, bounded and unique")
            if not isinstance(path, str) or len(path) > 4096:
                raise ValueError("module_path must be a bounded string")
            if operation not in ("capture", "patch", "zero", "add", "null"):
                raise ValueError("unsupported step operation")
            names.add(name)
            invocation = raw.get("invocation", 0)
            _nonnegative_int(invocation, "invocation")
            if invocation >= max_invocations:
                raise ValueError("invocation must be below max_invocations")
            batch_index, token_index = raw.get("batch_index"), raw.get("token_index")
            if (batch_index is None) != (token_index is None):
                raise ValueError("batch_index and token_index must both be set or omitted")
            if batch_index is not None:
                _nonnegative_int(batch_index, "batch_index")
                _nonnegative_int(token_index, "token_index")
            module = model.get_submodule(path)
            value, source_capture = raw.get("value"), raw.get("source_capture")
            if source_capture is not None:
                prior = previous_steps.get(source_capture) if isinstance(source_capture, str) else None
                if (operation != "patch" or value is not None or prior is None
                        or prior["operation"] != "capture" or prior["module"] is not module
                        or prior["invocation"] != invocation
                        or prior["batch_index"] != batch_index or prior["token_index"] != token_index):
                    raise ValueError("source_capture must name an earlier capture at the exact same site")
            elif operation in ("patch", "add"):
                if value is None:
                    raise ValueError("patch and add require a fixed tensor value")
                _validate_tensor(value, max_tensor_bytes)
                self._source_bytes += _tensor_bytes(value)
                raw_sources.append((name, value))
            elif value is not None:
                raise ValueError("capture, zero and null do not accept a value")
            site = (id(module), invocation)
            if site != last_site:
                if site in self._group_by_site:
                    raise ValueError("steps for one module invocation must be consecutive")
                self._group_by_site[site] = len(self._groups)
                self._groups.append([])
            step = {"name": name, "operation": operation, "module_path": path,
                    "module": module, "invocation": invocation,
                    "batch_index": batch_index, "token_index": token_index,
                    "source_capture": source_capture}
            self._groups[-1].append(step)
            self._steps.append(step)
            previous_steps[name] = step
            self._modules[id(module)] = module
            last_site = site
        if self._source_bytes > max_source_bytes:
            raise ValueError("aggregate source values exceed max_source_bytes")
        for name, value in raw_sources:
            self._sources[name] = value.detach().clone()
        # Every slot is allocated before any hook is installed. Index assignment
        # cannot trigger a list-growth allocation after registration returns.
        self._handles = [None] * (3 * len(self._modules) + 3)

    def __enter__(self):
        token = object()
        try:
            with self._lifecycle_lock:
                if self._used:
                    raise RuntimeError("activation plans are single-use")
                self._entry_token = token
                self._entering = True
                self._used = True
                self._owner_thread = current_thread()
            self._lease.acquire()
            self._verify_tree()
            for step in self._steps:
                if self.model.get_submodule(step["module_path"]) is not step["module"]:
                    raise RuntimeError("step module path changed before plan entry")
            aliases = self._lease.registered_paths
            self._aliases = {
                key: [path for path, module in aliases if module is target]
                for key, target in self._modules.items()
            }
            if any(not paths for paths in self._aliases.values()):
                raise RuntimeError("a target module is not in the leased tree")
            slot = 0
            self._register(slot, self.model.register_forward_pre_hook, self._before_forward)
            slot += 1
            for module in self._modules.values():
                self._register(slot, module.register_forward_pre_hook, self._before_target)
                self._register(slot + 1, module.register_forward_hook, self._target_hook)
                self._register(slot + 2, module.register_forward_hook, self._finish_target, always_call=True)
                slot += 3
            self._register(slot, self.model.register_forward_hook, self._complete_forward)
            self._register(slot + 1, self.model.register_forward_hook, self._finish_forward, always_call=True)
            self._verify_tree()
            with self._lifecycle_lock:
                if self._invalid:
                    raise RuntimeError("activation plan was invalidated during entry")
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
            source_error = None
            try:
                try:
                    try:
                        self._sources.clear()
                    except BaseException as clear_error:
                        source_error = clear_error
                        self._sources = {}
                finally:
                    try:
                        self._cleanup()
                    except BaseException as cleanup_error:
                        if source_error is not None:
                            cleanup_error.__cause__ = source_error
                        raise error from cleanup_error
                if source_error is not None:
                    raise error from source_error
            finally:
                with self._lifecycle_lock:
                    self._entering = False
                    self._entered = False
            raise

    def _register(self, slot, register, hook, **kwargs):
        self._registration_uncertain = True
        handle = register(hook, **kwargs)
        if not callable(getattr(handle, "remove", None)):
            raise TypeError("hook registration returned an invalid handle")
        self._handles[slot] = handle
        self._registration_uncertain = False

    def _verify_tree(self):
        try:
            self._lease.verify_tree()
        except RuntimeError:
            self._mark_invalid()
            raise

    def _before_forward(self, module, inputs):
        self._check_owner()
        if self._forward_count or self._in_forward:
            with self._lifecycle_lock:
                if self._sync_forward is not None:
                    self._sync_unsafe = True
            self._mark_invalid()
            raise RuntimeError("plan permits exactly one root model forward")
        self._verify_tree()
        self._forward_count = 1
        self._in_forward = True

    def _before_target(self, module, inputs):
        try:
            self._check_owner()
            if not self._in_forward:
                raise RuntimeError("target must run inside the root forward")
            self._verify_tree()
            key = id(module)
            if key in self._active_sites:
                with self._lifecycle_lock:
                    if self._sync_forward is not None:
                        self._sync_unsafe = True
                raise RuntimeError("reentrant target modules are not supported")
            invocation = self._counts.get(key, 0)
            if invocation >= self.max_invocations:
                raise RuntimeError("max_invocations exceeded")
            self._counts[key] = invocation + 1
            self._pending[key] = invocation
            self._active_sites.add(key)
        except BaseException:
            self._mark_invalid()
            raise

    def _target_hook(self, module, inputs, output):
        try:
            self._check_owner()
            self._begin_synchronous_hook()
            self._check_synchronous_value(output)
            self._verify_tree()
            key = id(module)
            if not self._in_forward or key not in self._pending:
                raise RuntimeError("target hook is outside an active model invocation")
            site = (key, self._pending[key])
            group_index = self._group_by_site.get(site)
            if group_index is None:
                self._processed_sites.add(key)
                self._end_synchronous_hook()
                return None
            if group_index != self._next_group:
                raise RuntimeError("model execution differs from the declared plan order")
            result = output
            changed = False
            preceding = []
            for step in self._groups[group_index]:
                result, modified = self._apply_step(step, result, preceding)
                changed = changed or modified
                preceding.append(step["name"])
            self._next_group += 1
            self._processed_sites.add(key)
            self._end_synchronous_hook()
            return result if changed else None
        except BaseException as original:
            try:
                try:
                    self._fail_synchronous_hook()
                finally:
                    self._mark_invalid()
            except BaseException as secondary:
                raise original.with_traceback(original.__traceback__) from secondary
            raise

    def _apply_step(self, step, output, preceding):
        if isinstance(output, torch.Tensor):
            tensor, selector = output, []
        elif type(output) is tuple and output and isinstance(output[0], torch.Tensor):
            tensor, selector = output[0], [0]
        else:
            raise TypeError("target output must be a tensor or a tuple beginning with one")
        _validate_tensor(tensor, self.max_tensor_bytes)
        batch, token = step["batch_index"], step["token_index"]
        if batch is None:
            selected = tensor
        else:
            if tensor.ndim != 3:
                raise ValueError("token selection requires [batch, token, hidden] output")
            if batch >= tensor.shape[0] or token >= tensor.shape[1]:
                raise IndexError("batch_index or token_index is out of range")
            selected = tensor[batch, token]
        operation, name = step["operation"], step["name"]
        changed = operation != "capture"
        if operation == "capture":
            size = _tensor_bytes(selected)
            if self._capture_bytes + size > self.max_capture_bytes:
                raise ValueError("aggregate captures exceed max_capture_bytes")
            self._captures[name] = selected.detach().clone()
            self._capture_bytes += size
            result = output
        else:
            if step["source_capture"] is not None:
                source = self._captures.get(step["source_capture"])
                if source is None:
                    raise RuntimeError("referenced capture was not completed")
            else:
                source = self._sources.get(name)
            if source is not None:
                if source.shape != selected.shape:
                    raise ValueError("source shape must exactly match the selected activation")
                if source.dtype != selected.dtype:
                    raise TypeError("source dtype must match the selected activation")
                if source.device != selected.device:
                    raise ValueError("source device must match the selected activation")
            if operation == "patch":
                replacement = source
            elif operation == "add":
                replacement = selected + source
            elif operation == "zero":
                replacement = torch.zeros_like(selected)
            else:
                replacement = selected
            _validate_tensor(replacement, self.max_tensor_bytes)
            modified = tensor.clone()
            if batch is None:
                modified[...] = replacement
            else:
                modified[batch, token] = replacement
            result = (modified,) + output[1:] if type(output) is tuple else modified
        self._events.append({
            "name": name, "operation": operation, "order": len(self._events),
            "module_path": step["module_path"],
            "module_aliases": list(self._aliases[id(step["module"])]),
            "call_site_known": False, "forward_index": 0,
            "invocation": step["invocation"], "batch_index": batch,
            "token_index": token, "output_selector": selector,
            "output_shape": list(tensor.shape), "selected_shape": list(selected.shape),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "source_capture": step["source_capture"],
            "preceding_same_site_actions": list(preceding),
        })
        return result, changed

    def _finish_target(self, module, inputs, output):
        if current_thread() is not self._owner_thread:
            self._mark_invalid()
            return None
        key = id(module)
        if key in self._active_sites and key not in self._processed_sites:
            self._mark_invalid()
        self._active_sites.discard(key)
        self._pending.pop(key, None)
        self._processed_sites.discard(key)
        return None

    def _complete_forward(self, module, inputs, output):
        self._check_owner()
        self._verify_tree()
        if self._active_sites or self._pending or self._processed_sites:
            self._mark_invalid()
            raise RuntimeError("target invocation did not complete")
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

    def _mark_invalid(self):
        with self._lifecycle_lock:
            self._invalid = True
            self._committed = False

    def _check_owner(self):
        with self._lifecycle_lock:
            if current_thread() is not self._owner_thread:
                self._invalid = True
                self._committed = False
                if self._sync_forward is not None:
                    self._sync_unsafe = True
                raise RuntimeError("activation plan must run in its owning thread")
            if self._invalid:
                raise RuntimeError("activation plan is already invalid")

    def run_synchronous_cpu(self, *args, **kwargs):
        """Call the leased model once under an exclusive, synchronous CPU contract.

        Call only inside this plan's context on its owning thread. The caller
        must ensure that the model, inputs, native operations and any other
        threads do not start or retain asynchronous or device work. CPU checks
        reject known non-CPU tensors but cannot establish that precondition.
        This is only Python hook/lease cleanup, not device or tensor recovery.
        """
        with self._lifecycle_lock:
            self._check_synchronous_entry()
        self._reject_known_non_cpu(args, kwargs)
        token = object()
        with self._lifecycle_lock:
            self._check_synchronous_entry()
            self._sync_forward = token
        try:
            output = self.model(*args, **kwargs)
        except BaseException as original:
            try:
                self._finish_synchronous(token, failed=True)
            except BaseException as secondary:
                raise original.with_traceback(original.__traceback__) from secondary
            raise
        self._check_synchronous_value(output)
        self._finish_synchronous(token, failed=False)
        return output

    def _check_synchronous_entry(self):
        # The lifecycle lock is held by the caller. Only an active wrapper's
        # owner can retire its marker; rejected attempts never acquire it.
        if current_thread() is not self._owner_thread:
            if self._entered or self._entering or self._closing:
                self._invalid = True
                self._committed = False
                self._sync_unsafe = True
            raise RuntimeError("synchronous forward must run on the owning thread")
        if not self._entered or self._entering or self._closing:
            raise RuntimeError("synchronous forward requires an entered plan")
        if self._sync_forward is not None or self._in_forward:
            self._invalid = True
            self._committed = False
            self._sync_unsafe = True
            raise RuntimeError("another forward may be active")
        if self._sync_unsafe or self._invalid or self._forward_count:
            self._invalid = True
            self._committed = False
            raise RuntimeError("plan cannot begin another forward")

    def _reject_known_non_cpu(self, args, kwargs):
        pending = [args, kwargs, tuple(self.model.parameters()),
                   tuple(self.model.buffers()), tuple(self._sources.values())]
        self._reject_non_cpu_values(pending)

    @staticmethod
    def _reject_non_cpu_values(pending):
        visited = set()
        while pending:
            value = pending.pop()
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("synchronous CPU forward rejects known non-CPU tensors")
            elif (type(value) in (tuple, list) or isinstance(value, dict)) and id(value) not in visited:
                visited.add(id(value))
                if isinstance(value, dict):
                    pending.extend(dict.keys(value))
                    pending.extend(dict.values(value))
                else:
                    pending.extend(value)

    def _check_synchronous_value(self, value):
        # Keep a pending marker during inspection. If a transition is
        # interrupted, the trusted owner must also retain the exception graph.
        with self._lifecycle_lock:
            if self._sync_forward is None:
                return
            if self._sync_observation_pending:
                self._sync_unsafe = True
                raise RuntimeError("synchronous output observation is uncertain")
            self._sync_observation_pending = True
            self._sync_observed = value
        self._reject_non_cpu_values([value])
        with self._lifecycle_lock:
            if self._sync_unsafe:
                raise RuntimeError("synchronous output observation is uncertain")
            self._sync_observed = None
            self._sync_observation_pending = False

    def _begin_synchronous_hook(self):
        with self._lifecycle_lock:
            if self._sync_forward is None:
                return
            if self._sync_hook_pending:
                self._sync_unsafe = True
                raise RuntimeError("synchronous target hook is reentrant")
            self._sync_hook_pending = True

    def _end_synchronous_hook(self):
        with self._lifecycle_lock:
            if self._sync_forward is None:
                return
            if self._sync_unsafe:
                raise RuntimeError("synchronous target hook is uncertain")
            self._sync_hook_pending = False

    def _fail_synchronous_hook(self):
        with self._lifecycle_lock:
            if self._sync_forward is not None:
                self._sync_unsafe = True

    def _finish_synchronous(self, token, *, failed):
        # This method is reached only after this wrapper's direct model call
        # returned or raised. Publish token retirement last: an interrupted
        # transition must leave the context unable to release the lease.
        with self._lifecycle_lock:
            if (current_thread() is not self._owner_thread or self._sync_forward is not token
                    or not self._entered or self._entering or self._closing or self._sync_unsafe
                    or self._sync_hook_pending or self._sync_observation_pending):
                self._invalid = True
                self._committed = False
                self._sync_unsafe = True
                if failed:
                    return
                raise RuntimeError("synchronous forward retirement is uncertain")
            if failed:
                self._invalid = True
                self._committed = False
                self._in_forward = False
            elif self._in_forward or not self._completed:
                self._invalid = True
                self._committed = False
                self._sync_unsafe = True
                raise RuntimeError("synchronous forward completion is uncertain")
            self._sync_forward = None

    def _cleanup(self):
        owner = current_thread()
        try:
            with self._lifecycle_lock:
                if self._borrow_cleanup_owner is not None:
                    raise RuntimeError("plan cleanup is already in progress")
                if not self._entering and not self._closing:
                    raise RuntimeError("plan cleanup requires an owned lifecycle transition")
                self._borrow_cleanup_owner = owner
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
                if self._borrow_release_key is None:
                    self._lease.release()
                else:
                    self._lease.release(self._borrow_release_key)
            if first_error is not None:
                raise first_error
            if self._registration_uncertain:
                raise RuntimeError("hook registration was indeterminate; discard this model and worker")
        finally:
            with self._lifecycle_lock:
                if self._borrow_cleanup_owner is owner:
                    self._borrow_cleanup_owner = None

    def _borrow_cleanup_complete(self):
        return (self._borrow_cleanup_owner is current_thread()
                and not self._in_forward and not self._registration_uncertain
                and self._sync_forward is None and not self._sync_unsafe
                and not self._sync_hook_pending
                and not self._sync_observation_pending and self._sync_observed is None
                and all(handle is None for handle in self._handles))

    def close(self):
        """Retry hook removal after an unsuccessful exit or registration."""
        token = object()
        try:
            with self._lifecycle_lock:
                if not self._used:
                    raise RuntimeError("close is only available after plan entry was attempted")
                if self._sync_forward is not None:
                    self._invalid = True
                    self._committed = False
                    self._sync_unsafe = True
                if (self._entering or self._entered or self._closing or self._in_forward
                        or self._sync_forward is not None or self._sync_unsafe):
                    raise RuntimeError("cannot close an active plan")
                self._closing = token
            self._cleanup()
        finally:
            with self._lifecycle_lock:
                if self._closing is token:
                    self._closing = False

    def abort(self):
        """Discard an entered plan after the owner has ensured quiescence."""
        with self._lifecycle_lock:
            if current_thread() is not self._owner_thread:
                if self._entering or self._entered or self._closing:
                    self._invalid = True
                    self._committed = False
                    if self._sync_forward is not None:
                        self._sync_unsafe = True
                raise RuntimeError("only the owning thread can abort an activation plan")
            if not self._entered or self._entering or self._closing:
                raise RuntimeError("no entered activation plan to abort")
            if self._in_forward or self._sync_forward is not None or self._sync_unsafe:
                self._invalid = True
                self._committed = False
                if self._sync_forward is not None:
                    self._sync_unsafe = True
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
                        if self._sync_forward is not None:
                            self._sync_unsafe = True
                    raise RuntimeError("activation plan cleanup is in progress")
                if not self._entering and not self._entered:
                    raise RuntimeError("activation plan is not in an active context")
                if current_thread() is not self._owner_thread:
                    self._invalid = True
                    if self._sync_forward is not None:
                        self._sync_unsafe = True
                    raise RuntimeError("activation plan must exit in its owning thread")
                if self._entering:
                    raise RuntimeError("activation plan is not in an active context")
                if self._in_forward or self._sync_forward is not None or self._sync_unsafe:
                    self._invalid = True
                    self._committed = False
                    if self._sync_forward is not None:
                        self._sync_unsafe = True
                    if exc is not None:
                        return False
                    raise RuntimeError("cannot exit while a forward may be active; quarantine the worker")
                self._closing = token
                self._entered = False
            if exc_type is None:
                try:
                    self._verify_tree()
                except BaseException as error:
                    verification_error = error
            self._sources.clear()
        except BaseException as error:
            with self._lifecycle_lock:
                owns_cleanup = self._closing is token
            if not owns_cleanup:
                raise
            preparation_error = error
            self._sources = {}
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
                                   and self._completed and not self._in_forward
                                   and not self._active_sites and not self._pending
                                   and not self._processed_sites
                                   and self._next_group == len(self._groups))
                        if success:
                            self._committed = True
                        self._entered = False
                        self._closing = False
        if not success:
            self._captures.clear()
            self._events.clear()
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
                raise RuntimeError("plan did not complete every requested step")
        return False

    @property
    def captures(self):
        if not self._committed:
            raise RuntimeError("plan has no successfully completed captures")
        return dict(self._captures)

    @property
    def metadata(self):
        if not self._committed:
            raise RuntimeError("plan has no successfully completed metadata")
        return {"steps": deepcopy(self._events), "capture_bytes": self._capture_bytes,
                "source_bytes": self._source_bytes}
