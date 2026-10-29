"""Bounded, recoverable control-plane receipts for trusted trace-job owners.

This module retains no trace payloads or memory claims and does not execute,
authorize, publish, transport, or dispose model results. See its documentation
for the required external custody and late-cancellation responsibilities.
"""
from dataclasses import dataclass, replace
import threading

_MAX_INT = 2**63 - 1
_FACTORY = object()


class JobRegistryFull(RuntimeError):
    """The bounded registry cannot admit another live job."""


class JobStateError(RuntimeError):
    """A requested transition is not valid for this job."""


class StaleReceipt(JobStateError):
    """The receipt no longer identifies the current registered generation."""


def _integer(value, name, minimum=1):
    if type(value) is not int or not minimum <= value <= _MAX_INT:
        raise ValueError(f"{name} must be a plain bounded integer")


def _hex_id(value, name):
    if (type(value) is not str or len(value) != 32
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a lowercase 32-character hex identifier")


def _recipient(value):
    if type(value) is not str or len(value) > 128 or not value.strip():
        raise ValueError("recipient must be a nonempty plain string of at most 128 characters")


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Tensor-free status for one current in-process registry record."""

    broker_id: str
    process_generation: str
    job_id: int
    registration_id: str
    worker_request_id: str
    record_generation: int
    phase: str
    cancel_requested: bool
    recipient: str | None


@dataclass(frozen=True, slots=True)
class _Record:
    job_id: int
    registration_id: str
    generation: int
    token: object
    cancel_signal: threading.Event
    phase: str = "preparing"
    recipient: str | None = None
    handoff_token: object | None = None


class _Noncopyable:
    __slots__ = ()

    def __copy__(self):
        raise TypeError("job capabilities cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("job capabilities cannot be copied")

    def __reduce__(self):
        raise TypeError("job capabilities cannot be serialized")

    def __reduce_ex__(self, protocol):
        raise TypeError("job capabilities cannot be serialized")


class _Receipt(_Noncopyable):
    __slots__ = ("_registry", "_job_id", "_generation", "_token")

    def __init__(self, factory, registry, record):
        if factory is not _FACTORY:
            raise TypeError("receipts are issued by their registry")
        self._registry = registry
        self._job_id = record.job_id
        self._generation = record.generation
        self._token = record.token


class _Handoff(_Noncopyable):
    __slots__ = ("_registry", "_job_id", "_generation", "_token", "_handoff_token")

    def __init__(self, factory, registry, record):
        if factory is not _FACTORY:
            raise TypeError("handoffs are issued by their registry")
        self._registry = registry
        self._job_id = record.job_id
        self._generation = record.generation
        self._token = record.token
        self._handoff_token = record.handoff_token


def _new_receipt(registry, record):
    return _Receipt(_FACTORY, registry, record)


def _new_handoff(registry, record):
    return _Handoff(_FACTORY, registry, record)


class TraceJobRegistry:
    """Coordinate bounded job receipts for one trusted broker/process owner.

    State transitions are control-plane attestations only. The coordinator
    must independently own and account for every resource and actual alias.
    """

    def __init__(self, *, broker_id, process_generation, max_jobs):
        _hex_id(broker_id, "broker_id")
        _hex_id(process_generation, "process_generation")
        _integer(max_jobs, "max_jobs")
        self._broker_id = broker_id
        self._process_generation = process_generation
        self._max_jobs = max_jobs
        self._next_generation = 0
        self._records = {}
        self._lock = threading.Lock()

    def _current(self, capability, expected_type):
        if type(capability) is not expected_type or capability._registry is not self:
            raise StaleReceipt("foreign or forged job capability")
        record = self._records.get(capability._job_id)
        if (record is None or record.generation != capability._generation
                or record.token is not capability._token):
            raise StaleReceipt("job capability is stale")
        return record

    def _snapshot(self, record):
        return JobSnapshot(self._broker_id, self._process_generation, record.job_id,
                           record.registration_id,
                           f"lg-{self._broker_id}-{record.job_id}", record.generation,
                           record.phase, record.cancel_signal.is_set(), record.recipient)

    def register(self, job_id, *, registration_id):
        """Register before allocation using a caller-known unique identifier."""
        _integer(job_id, "job_id")
        _hex_id(registration_id, "registration_id")
        with self._lock:
            if job_id in self._records:
                raise JobStateError("job ID is already registered")
            if len(self._records) >= self._max_jobs:
                raise JobRegistryFull("job registry is full")
            if self._next_generation > _MAX_INT:
                raise JobRegistryFull("job generation space is exhausted")
            record = _Record(job_id, registration_id, self._next_generation, object(),
                             threading.Event())
            self._next_generation += 1
            self._records[job_id] = record
            receipt = _new_receipt(self, record)
        return receipt

    def recover(self, job_id, *, registration_id):
        """Reissue a capability for the exact still-live registration."""
        _integer(job_id, "job_id")
        _hex_id(registration_id, "registration_id")
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.registration_id != registration_id:
                raise StaleReceipt("job registration is absent or stale")
            receipt = _new_receipt(self, record)
        return receipt

    def active_jobs(self):
        with self._lock:
            jobs = tuple(sorted(self._records))
        return jobs

    def inspect(self, receipt):
        with self._lock:
            snapshot = self._snapshot(self._current(receipt, _Receipt))
        return snapshot

    def cancel(self, receipt):
        """Signal cancellation; no resource is stopped, disposed, or released."""
        with self._lock:
            record = self._current(receipt, _Receipt)
        # Do not hold the publication lock while signalling or doing cleanup.
        # A record's signal persists across its immutable state replacements.
        record.cancel_signal.set()
        with self._lock:
            self._current(receipt, _Receipt)

    def seal(self, receipt):
        """Attest readiness; the caller must validate the complete job first."""
        with self._lock:
            record = self._current(receipt, _Receipt)
            if record.phase != "preparing" or record.cancel_signal.is_set():
                raise JobStateError("job cannot be sealed")
            self._records[record.job_id] = replace(record, phase="sealed")

    def begin_handoff(self, receipt, recipient):
        """Record a recoverable tentative handoff, without releasing custody."""
        _recipient(recipient)
        with self._lock:
            record = self._current(receipt, _Receipt)
            if record.phase != "sealed" or record.cancel_signal.is_set():
                raise JobStateError("job cannot begin handoff")
            updated = replace(record, phase="handoff_pending", recipient=recipient,
                              handoff_token=object())
            self._records[record.job_id] = updated
            handoff = _new_handoff(self, updated)
        return handoff

    def recover_handoff(self, receipt, recipient):
        _recipient(recipient)
        with self._lock:
            record = self._current(receipt, _Receipt)
            if (record.phase not in ("handoff_pending", "owned")
                    or record.recipient != recipient):
                raise JobStateError("no matching handoff is recoverable")
            handoff = _new_handoff(self, record)
        return handoff

    def ack_handoff(self, handoff):
        """A trusted recipient accepts handoff; duplicate acknowledgements are safe."""
        with self._lock:
            record = self._current(handoff, _Handoff)
            if record.handoff_token is not handoff._handoff_token:
                raise StaleReceipt("handoff token is stale")
            if record.phase == "owned":
                changed = False
            else:
                if record.phase != "handoff_pending" or record.cancel_signal.is_set():
                    raise JobStateError("handoff cannot be acknowledged")
                self._records[record.job_id] = replace(record, phase="owned")
                changed = True
        return changed

    def begin_disposal(self, receipt):
        """Start an owner-managed cleanup; no resource is disposed here."""
        with self._lock:
            record = self._current(receipt, _Receipt)
            if record.phase == "disposing":
                changed = False
            else:
                if record.phase == "quarantined":
                    raise JobStateError("quarantined jobs require external reconciliation")
                self._records[record.job_id] = replace(record, phase="disposing")
                changed = True
        return changed

    def finish_disposal(self, receipt):
        """Attest external disposal completed, then retire the registry entry."""
        with self._lock:
            record = self._current(receipt, _Receipt)
            if record.phase != "disposing":
                raise JobStateError("disposal has not begun")
            del self._records[record.job_id]

    def quarantine(self, receipt):
        """Keep an uncertain job registered until external reconciliation."""
        with self._lock:
            record = self._current(receipt, _Receipt)
            if record.phase != "quarantined":
                self._records[record.job_id] = replace(record, phase="quarantined")
