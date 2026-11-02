"""Cooperative host-memory custody accounting; no allocations are intercepted.

Callers reserve before allocating, keep claims during every overlapping use,
and release only after storage is disposed. This module is not an OS memory
limit or a security boundary between Python code in the same process.
"""

from dataclasses import dataclass, replace
from threading import RLock


class HostBudgetExceeded(RuntimeError):
    """A byte, item, request, operation, or transfer limit would be exceeded."""


class HostBusy(RuntimeError):
    """Resources are still claimed or an operation is using them."""


class HostRequestCancelled(RuntimeError):
    """The request is cancelled and cannot start new ordinary work."""


class InvalidHostHandle(RuntimeError):
    """A handle is stale, closed, foreign, forged, or already finished."""


def _text(value, name):
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError(f"{name} must be a nonempty string of at most 128 characters")
    return value


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


@dataclass(frozen=True)
class ClaimInfo:
    claim_id: int
    request_id: str
    request_generation: int
    stage: str
    owners: tuple[str, ...]
    cleanup_transfers: tuple[bool, ...]
    nbytes: int
    items: int
    active_operations: int
    binding: str | None = None


@dataclass(frozen=True)
class RequestInfo:
    request_id: str
    generation: int
    cancelled: bool


@dataclass(frozen=True)
class OperationInfo:
    operation_id: int
    request_id: str
    request_generation: int
    label: str
    claim_ids: tuple[int, ...]


@dataclass(frozen=True)
class HostBudgetSnapshot:
    max_bytes: int
    max_items: int
    max_requests: int
    max_operations: int
    max_transfers: int
    used_bytes: int
    used_items: int
    requests: tuple[RequestInfo, ...]
    claims: tuple[ClaimInfo, ...]
    operations: tuple[OperationInfo, ...]


@dataclass(slots=True)
class _RequestRecord:
    handle: object
    generation: int
    cancelled: bool = False
    guard: object = None


@dataclass(frozen=True)
class _ClaimRecord:
    handle: object
    request: object
    stage: str
    owners: tuple[str, ...]
    cleanup_transfers: tuple[bool, ...]
    nbytes: int
    items: int
    binding: str | None = None
    guard: object = None


@dataclass(frozen=True)
class _OperationRecord:
    handle: object
    request: object
    label: str
    claim_ids: tuple[int, ...]


class HostBudget:
    """Bound explicitly declared live host bytes, items, and ledger metadata."""

    def __init__(self, *, max_bytes, max_items, max_requests, max_operations,
                 max_transfers=32):
        self._max_bytes = _integer(max_bytes, "max_bytes", 0)
        self._max_items = _integer(max_items, "max_items", 1)
        self._max_requests = _integer(max_requests, "max_requests", 1)
        self._max_operations = _integer(max_operations, "max_operations", 1)
        self._max_transfers = _integer(max_transfers, "max_transfers", 0)
        self._lock = RLock()
        self._requests = {}
        self._claims = {}
        self._operations = {}
        self._next_claim = 0
        self._next_operation = 0
        self._next_request = 0

    @property
    def max_bytes(self):
        return self._max_bytes

    @property
    def max_items(self):
        return self._max_items

    @property
    def max_requests(self):
        return self._max_requests

    @property
    def max_operations(self):
        return self._max_operations

    @property
    def max_transfers(self):
        return self._max_transfers

    def _request(self, handle):
        if type(handle) is not HostRequest or handle._budget is not self:
            raise InvalidHostHandle("request belongs to another ledger")
        record = self._requests.get(handle._request_id)
        if record is None or record.handle is not handle:
            raise InvalidHostHandle("request is closed, stale, or unregistered")
        return record

    def _claim(self, handle):
        if type(handle) is not HostClaim or handle._budget is not self:
            raise InvalidHostHandle("claim belongs to another ledger")
        record = self._claims.get(handle._claim_id)
        if record is None or record.handle is not handle:
            raise InvalidHostHandle("claim is released, stale, or unregistered")
        return record

    def _operation_count(self, claim_id):
        return sum(claim_id in record.claim_ids for record in self._operations.values())

    def _claim_info(self, claim_id, record):
        generation = self._request(record.request).generation
        return ClaimInfo(claim_id, record.request._request_id, generation, record.stage,
                         record.owners, record.cleanup_transfers, record.nbytes,
                         record.items, self._operation_count(claim_id), record.binding)

    def open_request(self, request_id):
        """Open a bounded request lifecycle; a closed ID can later be reused."""
        _text(request_id, "request_id")
        with self._lock:
            if request_id in self._requests:
                raise ValueError("request_id is already open")
            if len(self._requests) >= self.max_requests:
                raise HostBudgetExceeded("request limit exceeded")
            handle = HostRequest(self, request_id, _issued=True)
            generation = self._next_request
            self._next_request += 1
            record = _RequestRecord(handle, generation)
            try:
                self._requests[request_id] = record
                return handle
            except BaseException:
                if self._requests.get(request_id) is record:
                    del self._requests[request_id]
                raise

    def request_guard(self, request_id):
        """Create an unregistered guard; no ledger resource is opened yet."""
        return HostRequestGuard(self, request_id)

    def active_guards(self):
        """Return guards retained by live request records for trusted recovery."""
        with self._lock:
            return tuple(record.guard for record in self._requests.values()
                         if record.guard is not None)

    def active_requests(self):
        """Return retained request handles for a trusted cleanup coordinator."""
        with self._lock:
            return tuple(record.handle for record in self._requests.values())

    def active_claims(self):
        """Return current owner capabilities; stale transferred handles are excluded."""
        with self._lock:
            return tuple(record.handle for record in self._claims.values())

    def active_claim_guards(self):
        """Return claim guards retained by live records for trusted recovery."""
        with self._lock:
            guards = tuple(record.guard for record in self._claims.values() if record.guard is not None)
        return guards

    def active_operations(self):
        """Return retained operation handles for verified cleanup or recovery."""
        with self._lock:
            return tuple(record.handle for record in self._operations.values())

    def snapshot(self):
        """Return an immutable snapshot of declared commitments and provenance."""
        with self._lock:
            claims = tuple(self._claim_info(key, value) for key, value in self._claims.items())
            requests = tuple(RequestInfo(key, value.generation, value.cancelled)
                             for key, value in self._requests.items())
            operations = tuple(OperationInfo(key, value.request._request_id,
                                             self._request(value.request).generation,
                                             value.label, value.claim_ids)
                               for key, value in self._operations.items())
            return HostBudgetSnapshot(self.max_bytes, self.max_items, self.max_requests,
                                      self.max_operations, self.max_transfers, sum(c.nbytes for c in claims),
                                      sum(c.items for c in claims), requests, claims, operations)


class HostRequestGuard:
    """Retain an exact request generation through open/adoption interruptions.

    Constructing the guard is side-effect free.  Keep it before calling
    ``open``; after registration the ledger also retains it until the request
    closes.  The guard never releases claims or finishes operations.
    """

    def __init__(self, budget, request_id):
        if type(budget) is not HostBudget:
            raise ValueError("budget must be a HostBudget")
        self._budget = budget
        self._request_id = _text(request_id, "request_id")
        self._handle = None
        self._closed = False

    def open(self):
        """Register at most one generation, retaining its handle before insert."""
        budget = self._budget
        with budget._lock:
            if self._closed:
                raise InvalidHostHandle("guard is closed")
            current = budget._requests.get(self._request_id)
            if self._handle is not None:
                if (current is not None and current.handle is self._handle
                        and current.guard is self):
                    return None
                # A guarded handle is never reused, including if interruption
                # occurred before its record was inserted.  A new guard is
                # required; the old one cannot attach to a reused request ID.
                raise InvalidHostHandle("guard has no live request generation")
            if current is not None:
                raise ValueError("request_id is already open")
            if len(budget._requests) >= budget.max_requests:
                raise HostBudgetExceeded("request limit exceeded")
            if self._handle is None:
                self._handle = HostRequest(budget, self._request_id, _issued=True)
            generation = budget._next_request
            budget._next_request += 1
            record = _RequestRecord(self._handle, generation, guard=self)
            budget._requests[self._request_id] = record
            return None  # opened

    def _record(self):
        current = self._budget._requests.get(self._request_id)
        if (self._closed or self._handle is None or current is None
                or current.handle is not self._handle or current.guard is not self):
            raise InvalidHostHandle("guard has no live request generation")
        return current

    def adopt(self):
        """Return the request alias while retaining the recovery handle."""
        with self._budget._lock:
            self._record()
            return self._handle

    @property
    def generation(self):
        with self._budget._lock:
            return self._record().generation

    def cancel_and_close_empty(self):
        """Cancel and close this exact request, but never dispose its claims."""
        budget = self._budget
        with budget._lock:
            if self._closed:
                return False
            current = budget._requests.get(self._request_id)
            if (self._handle is None or current is None
                    or current.handle is not self._handle or current.guard is not self):
                self._closed = True
                return False
            self._handle.cancel()
            closed = self._handle.close()  # HostBusy preserves all live claims.
            self._closed = True
            return closed


class HostRequest:
    """A request's lifetime, including its cancellation tombstone."""

    def __init__(self, budget, request_id, *, _issued=False):
        self._budget = budget
        self._request_id = request_id
        self._issued = _issued
        self._closed = False

    def reserve(self, owner, stage, nbytes, *, items=1):
        """Commit capacity before creating or retaining the declared storage."""
        _text(owner, "owner")
        _text(stage, "stage")
        _integer(nbytes, "nbytes", 0)
        _integer(items, "items", 1)
        budget = self._budget
        with budget._lock:
            state = budget._request(self)
            if state.cancelled:
                raise HostRequestCancelled("request is cancelled")
            if (sum(c.nbytes for c in budget._claims.values()) + nbytes > budget.max_bytes
                    or sum(c.items for c in budget._claims.values()) + items > budget.max_items):
                raise HostBudgetExceeded("host byte or item limit exceeded")
            claim_id = budget._next_claim
            budget._next_claim += 1
            handle = HostClaim(budget, claim_id, _issued=True)
            record = _ClaimRecord(handle, self, stage, (owner,), (), nbytes, items)
            try:
                budget._claims[claim_id] = record
                return handle
            except BaseException:
                if budget._claims.get(claim_id) is record:
                    del budget._claims[claim_id]
                raise

    def claim_guard(self, owner, stage, nbytes, *, items=1):
        """Create an unregistered, recoverable claim reservation guard."""
        return HostClaimGuard(self, owner, stage, nbytes, items=items)

    def begin_operation(self, label, claims):
        """Pin all source/destination claims for a copy, send, or other use."""
        _text(label, "label")
        if type(claims) not in (list, tuple) or not claims:
            raise ValueError("claims must be a nonempty list or tuple")
        if len(claims) > self._budget.max_items:
            raise ValueError("operation contains too many claims")
        claims = tuple(claims)
        budget = self._budget
        with budget._lock:
            if budget._request(self).cancelled:
                raise HostRequestCancelled("request is cancelled")
            if len(budget._operations) >= budget.max_operations:
                raise HostBudgetExceeded("operation limit exceeded")
            ids = []
            seen = set()
            for claim in claims:
                record = budget._claim(claim)
                if record.request is not self:
                    raise InvalidHostHandle("operation claims must belong to this request")
                if claim._claim_id in seen:
                    raise ValueError("operation cannot contain a duplicate claim")
                seen.add(claim._claim_id)
                ids.append(claim._claim_id)
            operation_id = budget._next_operation
            budget._next_operation += 1
            handle = HostOperation(budget, operation_id)
            record = _OperationRecord(handle, self, label, tuple(ids))
            try:
                budget._operations[operation_id] = record
                return handle
            except BaseException:
                if budget._operations.get(operation_id) is record:
                    del budget._operations[operation_id]
                raise

    def cancel(self):
        """Tombstone this request without releasing any claim or operation."""
        budget = self._budget
        with budget._lock:
            state = budget._request(self)
            if state.cancelled:
                return False
            state.cancelled = True
            return True

    def close(self):
        """Remove the tombstone only after all operations and claims are gone."""
        budget = self._budget
        with budget._lock:
            if self._closed:
                return False
            existing = budget._requests.get(self._request_id)
            if self._issued and (existing is None or existing.handle is not self):
                self._closed = True
                return False
            budget._request(self)
            if (any(c.request is self for c in budget._claims.values())
                    or any(o.request is self for o in budget._operations.values())):
                raise HostBusy("request still has claims or operations")
            try:
                del budget._requests[self._request_id]
            finally:
                if self._request_id not in budget._requests:
                    self._closed = True
            return True


class HostClaimGuard:
    """Retain one exact claim capability across reservation interruptions.

    Creating a guard does not reserve. The trusted owner retains it before
    calling reserve; after insertion, the ledger retains it until transfer or
    release. It does not dispose storage or release a claim automatically.
    """

    def __init__(self, request, owner, stage, nbytes, *, items=1):
        if type(request) is not HostRequest or type(request._budget) is not HostBudget:
            raise ValueError('request must be an exact HostRequest')
        self._request = request
        self._budget = request._budget
        self._owner = _text(owner, 'owner')
        self._stage = _text(stage, 'stage')
        self._nbytes = _integer(nbytes, 'nbytes', 0)
        self._items = _integer(items, 'items', 1)
        self._handle = None

    def reserve(self):
        """Register once or confirm the same still-live guarded claim."""
        budget = self._budget
        with budget._lock:
            state = budget._request(self._request)
            if self._handle is not None:
                current = budget._claims.get(self._handle._claim_id)
                if current is None or current.handle is not self._handle or current.guard is not self:
                    raise InvalidHostHandle('guard has no live claim')
            else:
                if state.cancelled:
                    raise HostRequestCancelled('request is cancelled')
                if (sum(c.nbytes for c in budget._claims.values()) + self._nbytes > budget.max_bytes
                        or sum(c.items for c in budget._claims.values()) + self._items > budget.max_items):
                    raise HostBudgetExceeded('host byte or item limit exceeded')
                claim_id = budget._next_claim
                budget._next_claim += 1
                self._handle = HostClaim(budget, claim_id, _issued=True)
                record = _ClaimRecord(self._handle, self._request, self._stage, (self._owner,), (),
                                      self._nbytes, self._items, guard=self)
                budget._claims[claim_id] = record
        return None

    def adopt(self):
        """Return the same live claim; cancellation still permits recovery."""
        budget = self._budget
        with budget._lock:
            budget._request(self._request)
            if self._handle is None:
                raise InvalidHostHandle('guard has no live claim')
            record = budget._claims.get(self._handle._claim_id)
            if record is None or record.handle is not self._handle or record.guard is not self:
                raise InvalidHostHandle('guard has no live claim')
            claim = self._handle
        return claim


class HostClaim:
    """An indivisible owner capability for a declared storage commitment."""

    def __init__(self, budget, claim_id, *, _issued=False):
        self._budget = budget
        self._claim_id = claim_id
        self._issued = _issued
        self._stale = False
        self._released = False

    def info(self):
        budget = self._budget
        with budget._lock:
            record = budget._claim(self)
            return budget._claim_info(self._claim_id, record)

    def bind_once(self, binding):
        """Bind this claim to one trusted purpose, including across transfer.

        Repeating the same binding is an idempotent recovery check. Bindings
        are labels, not credentials; the coordinator must issue unique values.
        """
        _text(binding, "binding")
        budget = self._budget
        bound = False
        with budget._lock:
            record = budget._claim(self)
            if budget._request(record.request).cancelled:
                raise HostRequestCancelled("request is cancelled")
            if record.binding is not None:
                if record.binding != binding:
                    raise ValueError("claim is already bound to another purpose")
            else:
                if budget._operation_count(self._claim_id):
                    raise HostBusy("claim is used by an in-flight operation")
                replacement = replace(record, binding=binding)
                try:
                    budget._claims[self._claim_id] = replacement
                    bound = True
                except BaseException:
                    if budget._claims.get(self._claim_id) is replacement:
                        budget._claims[self._claim_id] = record
                    raise
        return bound

    def transfer(self, new_owner, *, cleanup=False):
        """Atomically replace this capability; a cancelled claim permits cleanup only."""
        _text(new_owner, "new_owner")
        if type(cleanup) is not bool:
            raise ValueError("cleanup must be a boolean")
        budget = self._budget
        with budget._lock:
            record = budget._claim(self)
            if budget._operation_count(self._claim_id):
                raise HostBusy("claim is used by an in-flight operation")
            cancelled = budget._request(record.request).cancelled
            if cancelled and not cleanup:
                raise HostRequestCancelled("cancelled claims can only transfer for cleanup")
            if cleanup and not cancelled:
                raise ValueError("cleanup transfer requires a cancelled request")
            if new_owner == record.owners[-1]:
                raise ValueError("new owner must differ from the current owner")
            regular_transfers = len(record.cleanup_transfers) - sum(record.cleanup_transfers)
            if (cleanup and any(record.cleanup_transfers)) or (not cleanup and regular_transfers >= budget.max_transfers):
                raise HostBudgetExceeded("claim transfer history limit exceeded")
            handle = HostClaim(budget, self._claim_id, _issued=True)
            replacement = replace(
                record, handle=handle, owners=record.owners + (new_owner,),
                cleanup_transfers=record.cleanup_transfers + (cleanup,), guard=None)
            try:
                budget._claims[self._claim_id] = replacement
                self._stale = True
                return handle
            except BaseException:
                self._stale = False
                if budget._claims.get(self._claim_id) is replacement:
                    budget._claims[self._claim_id] = record
                raise

    def release(self):
        """Release after the current owner has disposed every covered reference."""
        budget = self._budget
        with budget._lock:
            if self._released:
                return False
            if self._issued and not self._stale and self._claim_id not in budget._claims:
                self._released = True
                return False
            budget._claim(self)
            if budget._operation_count(self._claim_id):
                raise HostBusy("claim is used by an in-flight operation")
            try:
                del budget._claims[self._claim_id]
            finally:
                if self._claim_id not in budget._claims:
                    self._released = True
            return True


class HostOperation:
    """An in-flight use pinning all participating claims until it finishes."""

    def __init__(self, budget, operation_id):
        self._budget = budget
        self._operation_id = operation_id

    def finish(self):
        """End the use and report request liveness at this instant only."""
        budget = self._budget
        with budget._lock:
            record = budget._operations.get(self._operation_id)
            if record is None or record.handle is not self:
                raise InvalidHostHandle("operation is finished or unregistered")
            live = not budget._request(record.request).cancelled
            try:
                del budget._operations[self._operation_id]
                return live
            except BaseException:
                if self._operation_id not in budget._operations:
                    budget._operations[self._operation_id] = record
                raise
