"""Process-local exclusive ownership of registered PyTorch module trees.

All cooperating hook APIs must use the same lease. It does not prevent custom
hooks, model-tree mutation, or access from another process.
"""

from threading import RLock
from weakref import ref

from torch import nn


_lock = RLock()
# id(module) -> (weak module reference, unique lease token). Active leases
# themselves retain the modules, so object IDs cannot be reused while held.
_owners = {}
_MAX_REGISTERED_PATHS = 65536


def _module_tree(model):
    paths = []
    try:
        for path, module in model.named_modules(remove_duplicate=False):
            if len(paths) >= _MAX_REGISTERED_PATHS:
                raise ValueError("model has too many registered module paths")
            paths.append((path, module))
    except RecursionError as error:
        raise ValueError("cyclic module trees are not supported") from error
    return tuple(paths)


def _module_ref(module):
    # Stale weak entries are removed from the next private registry snapshot.
    # A weakref callback must not mutate a registry during atomic publication.
    return ref(module)


class ModelLease:
    """Reserve a model's registered module tree for one in-process owner.

    Acquisition is nonblocking except for the short registry mutex. Each lease
    is single-use and nonreentrant; release is idempotent and owner-checked.
    The lease covers cooperating Python hook ownership. It does not synchronize
    CUDA or enforce device-work lifetime: a broker must separately hold its GPU
    resource lease through synchronization and device cleanup.
    """

    def __init__(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self._token = object()
        self._used = False
        self._held = False
        self._modules = ()
        self._tree = ()

    def acquire(self):
        global _owners
        tree = _module_tree(self.model)
        seen = set()
        modules = []
        for _, module in tree:
            if id(module) not in seen:
                seen.add(id(module))
                modules.append(module)
        modules = tuple(modules)
        references = tuple(_module_ref(module) for module in modules)
        with _lock:
            if self._used:
                raise RuntimeError("model leases are single-use")
            self._used = True
            replacement = _owners.copy()
            for key, entry in tuple(replacement.items()):
                if entry[0]() is None:
                    del replacement[key]
            for module in modules:
                entry = replacement.get(id(module))
                if entry is not None:
                    existing_module = entry[0]()
                    if existing_module is None:
                        del replacement[id(module)]
                    else:
                        raise RuntimeError("model module tree is already leased")
            previous = _owners
            try:
                for module, module_ref in zip(modules, references):
                    key = id(module)
                    replacement[key] = (module_ref, self._token)
                self._modules = modules
                self._tree = tree
                self._held = True
                self._verify_tree_snapshot()
                _owners = replacement
            except BaseException:
                _owners = previous
                self._modules = ()
                self._tree = ()
                self._held = False
                raise
        return self

    @property
    def registered_paths(self):
        """Return the acquisition-time paths and module objects while held."""
        with _lock:
            if not self._owns_tree():
                raise RuntimeError("model lease is not held")
            return self._tree

    def _owns_tree(self):
        if not self._held or not self._modules:
            return False
        for module in self._modules:
            entry = _owners.get(id(module))
            if entry is None or entry[0]() is not module or entry[1] is not self._token:
                return False
        return True

    def _verify_tree_snapshot(self):
        try:
            current = _module_tree(self.model)
        except ValueError as error:
            raise RuntimeError("model module tree changed during the lease") from error
        if (len(current) != len(self._tree)
                or any(a_path != b_path or a_module is not b_module
                       for (a_path, a_module), (b_path, b_module)
                       in zip(current, self._tree))):
            raise RuntimeError("model module tree changed during the lease")

    def verify_tree(self):
        """Reject added, removed, renamed, replaced or re-aliased modules."""
        with _lock:
            if not self._owns_tree():
                raise RuntimeError("model lease is not held")
            self._verify_tree_snapshot()

    def release(self):
        global _owners
        with _lock:
            if not self._held:
                return
            replacement = _owners.copy()
            for module in self._modules:
                key = id(module)
                entry = replacement.get(key)
                if entry is not None and entry[0]() is module and entry[1] is self._token:
                    del replacement[key]
            _owners = replacement
            self._modules = ()
            self._tree = ()
            self._held = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
