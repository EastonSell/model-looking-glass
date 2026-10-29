"""Process-local exclusive ownership of registered PyTorch module trees.

All cooperating hook APIs must use the same lease. It does not prevent custom
hooks, model-tree mutation, or access from another process.
"""

from threading import RLock, current_thread
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

    Retain the lease object and put acquisition inside an outer try/finally that
    calls release when cancellation is possible. Python can interrupt a context
    manager at the call/return boundary after ownership has been published.
    If release itself fails, retry or quarantine the model/worker.
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
        self._owner_thread = None
        self._borrower = None
        self._quarantined = False

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
                self._owner_thread = current_thread()
                self._verify_tree_snapshot()
                _owners = replacement
            except BaseException:
                _owners = previous
                self._modules = ()
                self._tree = ()
                self._held = False
                self._owner_thread = None
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
            if self._borrower is not None or self._quarantined:
                raise RuntimeError("model lease has an active or quarantined child scope")
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

    def _borrow_for_plan(self, release_key, cleanup_complete):
        """Create a restricted child capability for one trusted plan.

        This internal interface is not a general reentrant lease. The child
        acquires under the same registry lock, can only verify this lease's
        exact token/tree, and never releases the parent's reservation.
        """
        return _PlanLease(self, release_key, cleanup_complete)

    def _quarantine(self):
        """Permanently prevent release when the owning worker is uncertain."""
        with _lock:
            if not self._owns_tree():
                raise RuntimeError("model lease is not held")
            self._quarantined = True

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        try:
            self.release()
        except BaseException as cleanup_error:
            if exc is not None:
                raise exc.with_traceback(tb) from cleanup_error
            raise
        return False


class _PlanLease:
    """Single-use private borrow from a continuously held ModelLease."""

    def __init__(self, parent, release_key, cleanup_complete):
        self._parent = parent
        self._token = object()
        self._held = False
        self._used = False
        self._release_key = release_key
        self._cleanup_complete = cleanup_complete

    def acquire(self):
        parent = self._parent
        with _lock:
            if self._used:
                raise RuntimeError("plan lease scopes are single-use")
            self._used = True
            if current_thread() is not parent._owner_thread:
                raise RuntimeError("plan scope must run on the parent lease owner thread")
            if not parent._owns_tree() or parent._quarantined:
                raise RuntimeError("parent model lease is not available")
            if parent._borrower is not None:
                raise RuntimeError("another plan scope is active")
            try:
                parent._verify_tree_snapshot()
            except BaseException:
                parent._quarantined = True
                raise
            # Publish the borrow last. If interrupted after publication, the
            # retained child handle can release only its own token.
            self._held = True
            parent._borrower = self._token
        return self

    def _owns_scope(self):
        parent = self._parent
        return (self._held and parent._owns_tree()
                and parent._borrower is self._token and not parent._quarantined)

    @property
    def registered_paths(self):
        with _lock:
            if not self._owns_scope():
                raise RuntimeError("plan lease scope is not held")
            return self._parent._tree

    def verify_tree(self):
        with _lock:
            if current_thread() is not self._parent._owner_thread:
                raise RuntimeError("plan scope must run on the parent lease owner thread")
            if not self._owns_scope():
                raise RuntimeError("plan lease scope is not held")
            try:
                self._parent._verify_tree_snapshot()
            except BaseException:
                self._parent._quarantined = True
                raise

    def release(self, release_key=None):
        # A supervisor can retry cleanup of an exited, quiescent plan. The
        # plan's lifecycle owns that precondition; a stale child cannot remove
        # a newer child's marker or release the parent's registry reservation.
        with _lock:
            if release_key is not self._release_key:
                raise RuntimeError("only plan cleanup can release a borrowed scope")
            if not self._cleanup_complete():
                raise RuntimeError("plan hooks or forward may still be active")
            if self._held and self._parent._borrower is self._token:
                try:
                    if not self._parent._owns_tree():
                        raise RuntimeError("parent model lease is not held")
                    self._parent._verify_tree_snapshot()
                except BaseException:
                    self._parent._quarantined = True
                    raise
                self._parent._borrower = None
            self._held = False
