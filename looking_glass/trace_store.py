"""Process-local, bounded ownership of CPU trace payloads.

The store's quota covers its claimed logical tensor payload, including pending
copies. It is not a bound on total host memory or external tensor references.
"""
from dataclasses import dataclass
from threading import RLock

import torch


class TraceStoreFull(RuntimeError):
    """A byte or item reservation would exceed the store's limit."""


class TraceStoreCancelled(RuntimeError):
    """A request or an in-flight reservation was cancelled."""


class TraceStoreNotReady(RuntimeError):
    """An item has not yet been committed to the store."""


def _nonnegative(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _payload_info(value):
    if value is None:
        return 0, None, None
    # Tensor subclasses can override copy/size operations and are not trusted
    # to honor the store's preflight or independent-copy contract.
    if type(value) is not torch.Tensor:
        raise TypeError("trace payload must be a plain Tensor or None")
    if value.device.type != "cpu":
        raise ValueError("trace payload must already be on the CPU")
    if value.layout != torch.strided or value.is_nested or value.is_quantized:
        raise ValueError("trace payload must be a non-nested, non-quantized strided tensor")
    return value.numel() * value.element_size(), tuple(value.shape), value.dtype


def _clone_tensor(value):
    try:
        if value is None:
            return None
        return value.detach().clone(memory_format=torch.contiguous_format)
    finally:
        # Avoid retaining the input in this helper's exception traceback.
        value = None


@dataclass(frozen=True)
class TraceStoreUsage:
    max_tensor_bytes: int
    max_items: int
    claimed_bytes: int
    claimed_items: int
    ready_bytes: int
    ready_items: int


class HostTraceStore:
    """Thread-safe logical-byte and item quota shared by request handles.

    Both limits include pending reservations and in-flight copies. This
    object does not interact with MemoryLease; a broker must coordinate that
    separate accounting through the lifetime of its consumer-owned buffers.
    """

    def __init__(self, *, max_tensor_bytes, max_items):
        self._max_tensor_bytes = _nonnegative(max_tensor_bytes, "max_tensor_bytes")
        self._max_items = _nonnegative(max_items, "max_items")
        self._lock = RLock()
        self._entries = {}
        self._claimed_bytes = 0
        self._claimed_items = 0

    def request(self, request_id):
        if (type(request_id) is not str or len(request_id) > 256
                or not request_id.strip()):
            raise ValueError("request_id must be a nonempty string of at most 256 characters")
        return TraceRequest(self, request_id)

    def usage(self):
        with self._lock:
            ready = [entry for entry in self._entries.values() if entry._state == "ready"]
            return TraceStoreUsage(
                self._max_tensor_bytes, self._max_items,
                self._claimed_bytes, self._claimed_items,
                sum(entry._nbytes for entry in ready), len(ready),
            )

    def _remove_locked(self, entry, state, *, transfer=False):
        key = (entry._request, entry._item_id)
        if self._entries.get(key) is not entry:
            raise RuntimeError("trace store reservation is not active")
        remaining_bytes = remaining_items = None
        value = entry._value if transfer else None
        try:
            entry._state = "releasing"
            # Drop the store's reference while its claim still counts.
            # On take(), value deliberately transfers it to the caller.
            entry._value = None
            # Disposal can invoke reentrant weakref callbacks that change the
            # totals, so calculate the remaining counts only afterwards.
            remaining_bytes = self._claimed_bytes - entry._nbytes
            remaining_items = self._claimed_items - 1
            del self._entries[key]
            self._claimed_bytes = remaining_bytes
            self._claimed_items = remaining_items
            entry._state = state
            return value
        except BaseException:
            value = None
            entry._state = "releasing"
            entry._value = None
            if self._entries.get(key) is entry:
                del self._entries[key]
            # An interrupted take can release its local transfer reference
            # here, triggering callbacks after the earlier totals were read.
            # Rebuild from active entries so those reservations remain charged.
            self._claimed_bytes = sum(item._nbytes for item in self._entries.values())
            self._claimed_items = len(self._entries)
            entry._state = state
            raise


class TraceRequest:
    """A single request generation; cancellation permanently closes it."""

    def __init__(self, store, request_id):
        self._store = store
        self._request_id = request_id
        self._cancelled = False

    @property
    def request_id(self):
        return self._request_id

    def reserve(self, item_id, nbytes):
        _nonnegative(nbytes, "nbytes")
        if not ((type(item_id) is int and item_id >= 0)
                or (type(item_id) is str and len(item_id) <= 256 and item_id.strip())):
            raise ValueError("item_id must be a nonnegative integer or nonempty string of at most 256 characters")
        store = self._store
        entry = TraceReservation(self, item_id, nbytes)
        key = (self, item_id)
        with store._lock:
            if self._cancelled:
                raise TraceStoreCancelled("request was cancelled")
            if key in store._entries:
                raise ValueError("item_id already has an active reservation")
            old_items, old_bytes = store._claimed_items, store._claimed_bytes
            new_items, new_bytes = old_items + 1, old_bytes + nbytes
            if new_items > store._max_items or new_bytes > store._max_tensor_bytes:
                raise TraceStoreFull("trace store capacity exceeded")
            try:
                store._entries[key] = entry
                store._claimed_items = new_items
                store._claimed_bytes = new_bytes
                return entry
            except BaseException:
                if store._entries.get(key) is entry:
                    del store._entries[key]
                store._claimed_items, store._claimed_bytes = old_items, old_bytes
                entry._state = "failed"
                raise

    def put(self, item_id, value):
        nbytes, _, _ = _payload_info(value)
        with self.reserve(item_id, nbytes) as entry:
            entry.commit(value)
            return entry

    def cancel(self):
        store = self._store
        with store._lock:
            self._cancelled = True
            for entry in list(store._entries.values()):
                if entry._request is not self:
                    continue
                if entry._state in ("reserved", "ready"):
                    store._remove_locked(entry, "cancelled")
                elif entry._state == "copying":
                    entry._state = "cancelling"


class TraceReservation:
    """One claimed slot, later used to read, take, or drop the snapshot."""

    def __init__(self, request, item_id, nbytes):
        self._request = request
        self._store = request._store
        self._item_id = item_id
        self._nbytes = nbytes
        self._state = "reserved"
        self._value = None
        self._copy_owner = None

    @property
    def item_id(self):
        return self._item_id

    def commit(self, value):
        store = self._store
        owner = object()
        snapshot = None
        try:
            with store._lock:
                if self._state in ("cancelled", "cancelling"):
                    raise TraceStoreCancelled("reservation was cancelled")
                if self._state != "reserved" or self._copy_owner is not None:
                    raise RuntimeError("reservation cannot be committed twice")
                if store._entries.get((self._request, self._item_id)) is not self:
                    raise RuntimeError("reservation is not active in this store")
                # Mark ownership first, so an interrupted transition cannot
                # be claimed by a second committer before cleanup runs.
                self._copy_owner = owner
                self._state = "copying"
            nbytes, shape, dtype = _payload_info(value)
            if nbytes != self._nbytes:
                raise ValueError("payload size does not match its reservation")
            with store._lock:
                if self._state == "cancelling":
                    raise TraceStoreCancelled("reservation was cancelled before copying")
            snapshot = _clone_tensor(value)
            actual_bytes, actual_shape, actual_dtype = _payload_info(snapshot)
            if (actual_bytes != self._nbytes or actual_shape != shape or actual_dtype != dtype
                    or (snapshot is not None and
                        snapshot.untyped_storage().nbytes() != self._nbytes)):
                raise ValueError("copied payload does not match its reservation")
            if (snapshot is not None and self._nbytes and
                    snapshot.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()):
                raise ValueError("copied payload must not alias the source")
            with store._lock:
                if self._state == "cancelling":
                    raise TraceStoreCancelled("reservation was cancelled during copying")
                self._value = snapshot
                self._state = "ready"
            return self
        except BaseException:
            # A retained exception traceback keeps this frame alive. Do not let
            # it also retain our rejected copy after its claim is released.
            snapshot = None
            with store._lock:
                if (self._copy_owner is owner and
                        store._entries.get((self._request, self._item_id)) is self):
                    state = "cancelled" if self._state == "cancelling" else "failed"
                    store._remove_locked(self, state)
            raise

    def _ready_locked(self):
        if self._state in ("reserved", "copying", "cancelling"):
            raise TraceStoreNotReady("trace item is not ready")
        if self._state != "ready":
            raise KeyError("trace item is no longer retained")

    def read(self):
        """Return an independent caller-owned copy without removing the item."""
        with self._store._lock:
            self._ready_locked()
            return _clone_tensor(self._value)

    def take(self):
        """Transfer the stored payload to the caller and release the store claim."""
        with self._store._lock:
            self._ready_locked()
            return self._store._remove_locked(self, "taken", transfer=True)

    def cancel(self):
        """Cancel only an uncommitted claim; use drop for a ready item."""
        store = self._store
        with store._lock:
            if self._state == "reserved":
                store._remove_locked(self, "cancelled")
                return True
            if self._state == "copying":
                self._state = "cancelling"
                return True
            return False

    def drop(self):
        """Discard a ready item or cancel an uncommitted one, idempotently."""
        store = self._store
        with store._lock:
            if self._state in ("reserved", "ready"):
                store._remove_locked(self, "dropped")
            elif self._state == "copying":
                self._state = "cancelling"
                return True
            else:
                return False
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.cancel()
        else:
            self.drop()
        return False
