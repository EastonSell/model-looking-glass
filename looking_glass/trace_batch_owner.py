"""Private CPU trace-batch custody; this is not a broker or transport API."""
from dataclasses import dataclass, fields, is_dataclass
import threading
import time

from .host_budget import HostBudget, HostBudgetExceeded
from .trace_delivery_guard import DeliverySlotRegistry
from .trace_job_validation import make_trace_job_layout, _strip_failure_frames
from .trace_schema import TraceEvent
from .trace_store import HostTraceStore
from .trace_transport import DeliveryIdentity, DeliveryManifest
from .traced_broker_worker import BoundTracePlan


class BatchStateError(RuntimeError):
    """The batch is unavailable, incomplete, cancelled, or needs supervision."""


def _hex(value):
    return (type(value) is str and len(value) == 32
            and all(char in "0123456789abcdef" for char in value))


def _same_exact(left, right):
    """Compare validated metadata without bool/int or equality spoofing."""
    if type(left) is not type(right):
        return False
    if type(right) is tuple:
        return len(left) == len(right) and all(_same_exact(a, b) for a, b in zip(left, right))
    if is_dataclass(right):
        return all(_same_exact(getattr(left, item.name), getattr(right, item.name))
                   for item in fields(right))
    return left == right


@dataclass(frozen=True, slots=True)
class BatchManifest:
    broker_id: str
    process_generation: str
    job_id: int
    registration_id: str
    generation: int
    events: tuple[DeliveryManifest, ...]


@dataclass(frozen=True, slots=True)
class _Pending:
    registration_id: str
    generation: int


class TraceBatchRegistry:
    """Root bounded job contexts before any child can allocate a claim."""

    def __init__(self, *, budget, store, broker_id, process_generation,
                 max_jobs, max_slots_per_job):
        if (type(budget) is not HostBudget or type(store) is not HostTraceStore
                or not _hex(broker_id) or not _hex(process_generation)
                or type(max_jobs) is not int or not 1 <= max_jobs <= 65536
                or type(max_slots_per_job) is not int
                or not 1 <= max_slots_per_job <= 65536):
            raise ValueError("invalid trusted batch registry configuration")
        self._budget = budget
        self._store = store
        self._broker_id = broker_id
        self._process_generation = process_generation
        self._max_jobs = max_jobs
        self._max_slots = max_slots_per_job
        self._jobs = {}
        self._last_generation = 0
        self._lock = threading.Lock()

    def register(self, plan, *, registration_id, generation, batch_size, sequence_length,
                 hidden_width, layer_count, max_event_bytes):
        """Install a caller-keyed context; never start capture during registration."""
        if (type(plan) is not BoundTracePlan or not _hex(registration_id)
                or type(generation) is not int or not 1 <= generation < 2**63):
            raise ValueError("invalid plan or registration ID")
        identity = plan.identity
        if (type(identity.job_id) is not int or not 1 <= identity.job_id < 2**63
                or identity.broker_id != self._broker_id):
            raise ValueError("plan identity does not belong to this registry")
        pending = _Pending(registration_id, generation)
        try:
            with self._lock:
                if identity.job_id in self._jobs or len(self._jobs) >= self._max_jobs:
                    raise BatchStateError("job is active or registry is full")
                if generation <= self._last_generation:
                    raise BatchStateError("registration generation is not fresh")
                self._last_generation = generation
                self._jobs[identity.job_id] = pending
            layout = make_trace_job_layout(plan, batch_size=batch_size,
                                           sequence_length=sequence_length,
                                           hidden_width=hidden_width, layer_count=layer_count,
                                           max_slots=self._max_slots,
                                           max_event_bytes=max_event_bytes)
            request = plan.capture_request()
            if layout._identity != (request.run_id, request.job_id, request.request_id):
                raise ValueError("plan identity changed during construction")
            job = _TraceBatch(self, identity.broker_id, identity.job_id, identity.team_id,
                              request.run_id, request.request_id, registration_id,
                              generation, layout)
            with self._lock:
                if self._jobs.get(identity.job_id) is not pending:
                    raise BatchStateError("registration ownership changed")
                self._jobs[identity.job_id] = job
        except BaseException:
            with self._lock:
                if self._jobs.get(identity.job_id) is pending:
                    del self._jobs[identity.job_id]
            raise

    def lookup(self, job_id, *, registration_id, generation):
        if (type(job_id) is not int or not _hex(registration_id)
                or type(generation) is not int or not 1 <= generation < 2**63):
            raise ValueError("invalid job key")
        with self._lock:
            job = self._jobs.get(job_id)
            if (type(job) is not _TraceBatch or job._registration_id != registration_id
                    or job._generation != generation):
                raise BatchStateError("unknown or stale job")
        return job

    def active_jobs(self):
        with self._lock:
            jobs = tuple(sorted(self._jobs))
        return jobs

    def retire(self, job):
        if type(job) is not _TraceBatch or job._registry is not self:
            raise BatchStateError("foreign job")
        with job._lock:
            if job._phase != "disposed" or job._active:
                raise BatchStateError("job has not drained")
            with self._lock:
                if self._jobs.get(job._job_id) is not job:
                    raise BatchStateError("stale job")
                del self._jobs[job._job_id]
                job._phase = "retired"


class _TraceBatch:
    def __init__(self, registry, broker_id, job_id, team_id, run_id, request_id,
                 registration_id, generation, layout):
        self._registry = registry
        self._broker_id, self._job_id, self._team_id = broker_id, job_id, team_id
        self._run_id, self._request_id = run_id, request_id
        self._registration_id, self._layout = registration_id, layout
        self._generation = generation
        self._slots = layout.slots
        # One registry per rooted job. IDs/generations exist before any child
        # registration and remain recoverable through the rooted job object.
        self._keys = tuple((f"event-{slot.slot_index}", slot.slot_index + 1)
                           for slot in self._slots)
        self._deliveries = DeliverySlotRegistry(max_slots=len(self._slots))
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._phase = "registered"
        self._active = None
        self._source_status = ["unattempted"] * len(self._slots)
        self._metadata = None
        self._manifest = None
        self._sources_disposed = False

    def __copy__(self):
        raise TypeError("batch capabilities cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("batch capabilities cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("batch capabilities cannot be serialized")

    @property
    def state(self):
        with self._lock:
            phase = self._phase
        return phase

    def _begin(self, expected, phase, token):
        with self._lock:
            if self._active or self._phase not in expected or self._cancel.is_set():
                raise BatchStateError("job is not in the required live state")
            self._active = token
            self._phase = phase

    def _fail(self, token):
        with self._lock:
            if self._active is token:
                self._cancel.set()
                self._phase = "failed"
                self._active = None

    def prepare(self):
        """Reserve each child source before the trusted owner starts capture."""
        token = object()
        try:
            self._begin(("registered",), "preparing", token)
            for slot, key in zip(self._slots, self._keys):
                if self._cancel.is_set():
                    raise BatchStateError("job was cancelled")
                self._deliveries.register(*key)
                child_slot = self._deliveries.lookup(*key)
                child_slot.construct(self._registry._budget, self._registry._store,
                                     broker_id=self._broker_id, job_id=self._job_id,
                                     team_id=self._team_id, run_id=self._run_id,
                                     request_id=self._request_id, sample_id=slot.sample_id,
                                     batch_position=slot.batch_position,
                                     max_payload_bytes=slot.payload_bytes)
                self._source_status[slot.slot_index] = "pending"
                try:
                    child_slot.adopt().reserve_source(slot.payload_bytes)
                except HostBudgetExceeded:
                    # This concrete delivery API reports this error only when
                    # no source claim was committed. Other failures remain
                    # pending and require supervisor reconciliation.
                    self._source_status[slot.slot_index] = "absent"
                    raise
                self._source_status[slot.slot_index] = "reserved"
            with self._lock:
                if self._cancel.is_set():
                    raise BatchStateError("job was cancelled")
                self._phase = "prepared"
                self._active = None
        except BaseException:
            self._fail(token)
            raise

    def stage(self, events):
        """Validate and privately clone every exact slot; no payload escapes."""
        by_key = event = metadata = failure = None
        token = object()
        try:
            self._begin(("prepared",), "staging", token)
            metadata = self._layout.validate(events)
            by_key = {(event.batch_position, event.layer_index, event.module_kind): event
                      for event in events}
            for slot, key in zip(self._slots, self._keys):
                if self._cancel.is_set():
                    raise BatchStateError("job was cancelled")
                event = by_key[(slot.batch_position, slot.layer_index, slot.module_kind)]
                self._deliveries.lookup(*key).adopt().stage(event)
                event = None
            with self._lock:
                if self._cancel.is_set():
                    raise BatchStateError("job was cancelled")
                self._metadata = metadata
                self._phase = "staged"
                self._active = None
        except BaseException as error:
            self._fail(token)
            failure = error
            try:
                _strip_failure_frames(error)
            except BaseException:
                pass
        finally:
            event = events = by_key = metadata = None
        if failure is not None:
            try:
                raise failure from None
            except BaseException:
                try:
                    failure.__context__ = failure.__cause__ = None
                except BaseException:
                    pass
                raise

    def confirm_sources_disposed(self):
        """Attest that every source and every producer alias is already gone."""
        token = object()
        uncertain = False
        try:
            with self._lock:
                cancelled_early = (self._cancel.is_set()
                                   and self._phase in ("registered", "prepared"))
                if (self._active or self._sources_disposed
                        or (self._phase not in ("staged", "failed") and not cancelled_early)):
                    raise BatchStateError("source disposal cannot be confirmed")
                self._active = token
            for slot_id, generation, state in self._deliveries.active_slots():
                index = generation - 1
                status = self._source_status[index]
                if status == "pending":
                    uncertain = True
                    raise BatchStateError("source reservation outcome requires supervision")
                slot = self._deliveries.lookup(slot_id, generation)
                if state == "failed_clean" or state == "reserved":
                    continue
                if state != "ready":
                    raise BatchStateError("slot requires supervision")
                child = slot.adopt()
                if self._cancel.is_set():
                    child.cancel()
                if status == "reserved":
                    child.confirm_source_disposed()
            with self._lock:
                self._sources_disposed = True
                self._active = None
        except BaseException:
            if uncertain:
                with self._lock:
                    if self._active is token:
                        self._cancel.set()
                        self._phase = "quarantined"
                        self._active = None
            else:
                self._fail(token)
            raise

    def seal(self):
        """Commit complete metadata after trusted source disposal/postchecks."""
        with self._lock:
            if not self._sources_disposed:
                raise BatchStateError("source disposal is not confirmed")
        token = object()
        try:
            self._begin(("staged",), "sealing", token)
            manifests = []
            delivery_ids = set()
            for slot, key, expected in zip(self._slots, self._keys, self._metadata):
                if self._cancel.is_set():
                    raise BatchStateError("job was cancelled")
                child = self._deliveries.lookup(*key).adopt()
                manifest = child.publish()
                identity = manifest.identity
                if (type(manifest) is not DeliveryManifest
                        or type(identity) is not DeliveryIdentity
                        or not _same_exact(identity, child.identity) or not _hex(identity.delivery_id)
                        or identity.delivery_id in delivery_ids
                        or not _same_exact(
                            (identity.broker_id, identity.job_id, identity.team_id,
                             identity.run_id, identity.request_id, identity.sample_id,
                             identity.batch_position),
                            (self._broker_id, self._job_id, self._team_id, self._run_id,
                             self._request_id, slot.sample_id, slot.batch_position))
                        or not _same_exact(manifest.event, expected)
                        or type(manifest.payload_bytes) is not int
                        or manifest.payload_bytes != slot.payload_bytes
                        or type(manifest.byte_order) is not str or manifest.byte_order != "little"):
                    raise BatchStateError("child manifest does not match its exact slot")
                delivery_ids.add(identity.delivery_id)
                manifests.append(manifest)
            complete = BatchManifest(self._broker_id, self._registry._process_generation,
                                     self._job_id, self._registration_id, self._generation,
                                     tuple(manifests))
            with self._lock:
                if self._cancel.is_set():
                    raise BatchStateError("job was cancelled")
                self._manifest = complete
                self._phase = "sealed"
                self._active = None
        except BaseException:
            self._fail(token)
            raise

    def manifest(self):
        """Return detached metadata only after the whole batch commits."""
        with self._lock:
            if self._phase != "sealed" or self._active or self._cancel.is_set():
                raise BatchStateError("batch is not available")
            current = self._manifest
        items = tuple(DeliveryManifest(DeliveryIdentity(
            item.identity.broker_id, item.identity.job_id, item.identity.team_id,
            item.identity.run_id, item.identity.request_id, item.identity.sample_id,
            item.identity.batch_position, item.identity.delivery_id),
            TraceEvent.from_json(item.event.to_json()), item.payload_bytes, item.byte_order)
                      for item in current.events)
        return BatchManifest(current.broker_id, current.process_generation, current.job_id,
                             current.registration_id, current.generation, items)

    def cancel(self):
        """Signal promptly, then serialize with short phase-commit sections."""
        self._cancel.set()
        with self._lock:
            pass

    def dispose(self, *, timeout):
        """Cancel children and drain only after truthful source disposal."""
        if type(timeout) not in (int, float) or not 0 <= timeout <= 86400:
            raise ValueError("invalid timeout")
        self._cancel.set()
        token = object()
        deadline = time.monotonic() + timeout
        drained = False
        try:
            with self._lock:
                if self._active or not self._sources_disposed:
                    return False
                if self._phase == "disposed":
                    return True
                if self._phase in ("retired", "quarantined"):
                    return False
                self._active = token
                self._phase = "disposing"
            for slot_id, generation, state in self._deliveries.active_slots():
                slot = self._deliveries.lookup(slot_id, generation)
                if state == "quarantined":
                    with self._lock:
                        self._phase = "quarantined"
                    return False
                if state == "failed_clean" or state == "reserved":
                    if not slot.retire():
                        return False
                    continue
                child = slot.owned_delivery()
                slot.cancel()
                if not child.drain(max(0.0, deadline - time.monotonic())):
                    return False
                if not slot.retire():
                    return False
            drained = True
            return True
        except BaseException:
            with self._lock:
                if self._active is token:
                    self._phase = "quarantined"
            raise
        finally:
            with self._lock:
                if self._active is token:
                    self._active = None
                    if drained:
                        self._phase = "disposed"
