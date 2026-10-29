"""Trusted in-process custody for one sealed CPU trace batch.

This is not a broker, wire protocol, authentication boundary, or CUDA path.
"""
from dataclasses import dataclass
import threading

from .cpu_trace_owner import CpuTraceOwner, _CpuTraceJob
from .host_budget import HostBudget, HostBudgetExceeded
from .trace_batch_owner import BatchManifest, _same_exact
from .trace_transport import DeliveryIdentity, DeliveryManifest


class TraceHandoffStateError(RuntimeError):
    """The handoff is stale, unavailable, incomplete or requires supervision."""


def _hex(value):
    return (type(value) is str and len(value) == 32
            and all(character in "0123456789abcdef" for character in value))


def _text(value):
    return type(value) is str and 0 < len(value) <= 128 and bool(value.strip())


@dataclass(frozen=True, slots=True)
class EventKey:
    broker_id: str
    process_generation: str
    job_id: int
    registration_id: str
    generation: int
    transfer_id: str
    slot_id: str
    slot_generation: int
    delivery: DeliveryIdentity
    sequence: int


@dataclass(frozen=True, slots=True)
class CompleteMarker:
    broker_id: str
    process_generation: str
    job_id: int
    registration_id: str
    generation: int
    transfer_id: str
    event_count: int


_FACTORY = object()


class _Noncopyable:
    def __copy__(self):
        raise TypeError("handoff capabilities cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("handoff capabilities cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("handoff capabilities cannot be serialized")


class SealedBatchRegistry:
    """Root a bounded set of trusted sealed CPU jobs before handoff."""

    def __init__(self, *, budget, broker_id, process_generation, max_jobs, max_events):
        if (type(budget) is not HostBudget or not _hex(broker_id)
                or not _hex(process_generation) or type(max_jobs) is not int
                or not 1 <= max_jobs <= 65536 or type(max_events) is not int
                or not 1 <= max_events <= 256):
            raise ValueError("invalid sealed-batch registry configuration")
        self._budget = budget
        self._broker_id = broker_id
        self._process_generation = process_generation
        self._max_jobs = max_jobs
        self._max_events = max_events
        self._last_generation = 0
        self._jobs = {}
        self._lock = threading.Lock()

    def register(self, owner, job, *, transfer_id, recipient_id):
        """Root a sealed job; caller assigns all identities before this call."""
        if (type(owner) is not CpuTraceOwner or type(job) is not _CpuTraceJob
                or job._owner is not owner or owner._budget is not self._budget
                or owner._broker_id != self._broker_id
                or owner._process_generation != self._process_generation
                or not _hex(transfer_id) or not _text(recipient_id)):
            raise ValueError("invalid trusted owner or handoff identity")
        with job._lock:
            with owner._lock:
                if (owner._job is not job or job._phase != "sealed" or job._active is not None
                        or job._cancel.is_set() or job._batch is None
                        or getattr(job, "_trace_handoff", None) is not None):
                    raise TraceHandoffStateError("job is not a live sealed owner")
            count = len(job._batch._keys)
            if not 1 <= count <= self._max_events:
                raise ValueError("job exceeds the event limit")
            with self._lock:
                if job._job_id in self._jobs or len(self._jobs) >= self._max_jobs:
                    raise TraceHandoffStateError("job already registered or registry is full")
                if job._generation <= self._last_generation:
                    raise TraceHandoffStateError("job generation is not fresh")
                previous_generation = self._last_generation
                bridge = None
                try:
                    self._last_generation = job._generation
                    bridge = _HandoffJob(self, owner, job, transfer_id, recipient_id, count)
                    # Fence another registry using the exact same live job.
                    # This is cooperative and does not prevent direct calls
                    # through an independently held job alias.
                    job._trace_handoff = bridge
                    self._jobs[job._job_id] = bridge
                except BaseException:
                    if bridge is None or self._jobs.get(job._job_id) is not bridge:
                        if bridge is not None and getattr(job, "_trace_handoff", None) is bridge:
                            job._trace_handoff = None
                        self._last_generation = previous_generation
                    raise

    def lookup(self, job_id, *, registration_id, generation, transfer_id):
        if (type(job_id) is not int or not 1 <= job_id < 2**63
                or not _hex(registration_id) or type(generation) is not int
                or not 1 <= generation < 2**63 or not _hex(transfer_id)):
            raise TraceHandoffStateError("invalid or stale handoff key")
        with self._lock:
            bridge = self._jobs.get(job_id)
            if (bridge is None or bridge._registration_id != registration_id
                    or bridge._generation != generation or bridge._transfer_id != transfer_id):
                raise TraceHandoffStateError("unknown or stale handoff")
        return bridge

    def retire(self, bridge):
        if type(bridge) is not _HandoffJob or bridge._registry is not self:
            raise TraceHandoffStateError("foreign handoff")
        with bridge._lock:
            if bridge._phase != "disposed" or bridge._active is not None:
                raise TraceHandoffStateError("handoff has not drained")
            with bridge._job._lock:
                with self._lock:
                    if (self._jobs.get(bridge._job_id) is not bridge
                            or getattr(bridge._job, "_trace_handoff", None) is not bridge):
                        raise TraceHandoffStateError("stale handoff")
                    del self._jobs[bridge._job_id]
                    bridge._job._trace_handoff = None
                    bridge._phase = "retired"


class _HandoffJob(_Noncopyable):
    def __init__(self, registry, owner, job, transfer_id, recipient_id, count):
        self._registry, self._owner, self._job = registry, owner, job
        self._job_id, self._registration_id, self._generation = (
            job._job_id, job._registration_id, job._generation)
        self._transfer_id, self._recipient_id, self._count = transfer_id, recipient_id, count
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._phase = "registered"
        self._active = None
        self._active_child = None
        self._recipient = None
        self._host_guard = self._host_request = self._manifest_guard = None
        self._manifest_claim = self._manifest_op = None
        self._expected = self._published = self._expected_keys = self._keys = None
        self._key_index = None
        self._marker = self._expected_marker = None
        self._states = ["pending"] * count
        self._manifest_received = False
        self._manifest_exposed = False

    @property
    def state(self):
        with self._lock:
            state = self._phase
        return state

    def _begin(self, expected, phase, token, *, allow_cancel=False):
        with self._lock:
            if (self._active is not None or self._phase not in expected
                    or (self._cancel.is_set() and not allow_cancel)):
                raise TraceHandoffStateError("handoff is not in the required state")
            self._active = token
            self._phase = phase

    def _fail(self, token):
        with self._lock:
            if self._active is token:
                self._cancel.set()
                self._phase = "quarantined"
                self._active = None
                self._active_child = None

    def prepare(self):
        """Reserve manifest representation/workspace before copying metadata."""
        token = object()
        copy_started = False
        try:
            self._begin(("registered",), "preparing", token)
            budget = self._registry._budget
            self._host_guard = budget.request_guard(f"handoff-{self._transfer_id}")
            self._host_guard.open()
            self._host_request = self._host_guard.adopt()
            # Two persistent metadata copies plus one JSON workspace envelope.
            self._manifest_guard = self._host_request.claim_guard(
                "handoff", "manifest", (2 * self._count + 1) * 65536,
                items=2 * self._count + 1)
            self._manifest_guard.reserve()
            self._manifest_claim = self._manifest_guard.adopt()
            self._manifest_op = self._host_request.begin_operation(
                f"handoff-manifest-{self._transfer_id}", (self._manifest_claim,))
            copy_started = True
            self._expected = self._job.manifest()
            self._published = self._job.manifest()
            if (type(self._expected) is not BatchManifest
                    or not _same_exact(self._published, self._expected)
                    or len(self._expected.events) != self._count):
                raise TraceHandoffStateError("sealed manifest changed or is incomplete")
            self._expected_keys = self._make_keys(self._expected)
            self._keys = self._make_keys(self._published)
            if not _same_exact(self._keys, self._expected_keys):
                raise TraceHandoffStateError("sealed slot identities differ")
            self._key_index = {(key.slot_id, key.slot_generation): index
                               for index, key in enumerate(self._expected_keys)}
            if len(self._key_index) != self._count:
                raise TraceHandoffStateError("sealed slot keys are not unique")
            self._expected_marker = CompleteMarker(
                self._registry._broker_id, self._registry._process_generation, self._job_id,
                self._registration_id, self._generation, self._transfer_id, self._count)
            self._marker = CompleteMarker(
                self._registry._broker_id, self._registry._process_generation, self._job_id,
                self._registration_id, self._generation, self._transfer_id, self._count)
            live = self._manifest_op.finish()
            self._manifest_op = None
            if not live:
                raise TraceHandoffStateError("manifest request was cancelled")
            with self._lock:
                if self._cancel.is_set():
                    raise TraceHandoffStateError("handoff was cancelled")
                self._phase = "prepared"
                self._active = None
        except HostBudgetExceeded:
            if not copy_started:
                with self._lock:
                    if self._active is token:
                        self._cancel.set()
                        self._phase = "admission_failed"
                        self._active = None
            else:
                self._fail(token)
            raise
        except BaseException:
            self._fail(token)
            raise

    def _make_keys(self, manifest):
        if (not _same_exact((manifest.broker_id, manifest.process_generation,
                             manifest.job_id, manifest.registration_id, manifest.generation),
                            (self._registry._broker_id, self._registry._process_generation,
                             self._job_id, self._registration_id, self._generation))
                or type(manifest.events) is not tuple or len(manifest.events) != self._count):
            raise TraceHandoffStateError("manifest does not match this job")
        keys = []
        for pair, event in zip(self._job._batch._keys, manifest.events):
            if (type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
                    or type(pair[1]) is not int or type(event) is not DeliveryManifest
                    or type(event.identity) is not DeliveryIdentity):
                raise TraceHandoffStateError("invalid child manifest or slot")
            keys.append(EventKey(self._registry._broker_id, self._registry._process_generation,
                                 self._job_id, self._registration_id, self._generation,
                                 self._transfer_id, pair[0], pair[1], event.identity,
                                 event.event.sequence))
        return tuple(keys)

    def offer(self):
        with self._lock:
            if self._phase != "prepared" or self._active is not None or self._cancel.is_set():
                raise TraceHandoffStateError("handoff cannot be offered")
            self._phase = "offered"

    def accept(self, *, recipient_id, transfer_id):
        if not _text(recipient_id) or not _hex(transfer_id):
            raise TraceHandoffStateError("invalid recipient identity")
        with self._lock:
            if (self._phase != "offered" or self._cancel.is_set()
                    or recipient_id != self._recipient_id or transfer_id != self._transfer_id):
                raise TraceHandoffStateError("handoff cannot be accepted")
            if self._recipient is None:
                self._recipient = _Recipient(_FACTORY, self, object())
            self._phase = "owned"
            recipient = self._recipient
        return recipient

    def recover_recipient(self, *, recipient_id, transfer_id):
        if not _text(recipient_id) or not _hex(transfer_id):
            raise TraceHandoffStateError("invalid recipient identity")
        with self._lock:
            if (self._phase not in ("owned", "completed") or self._cancel.is_set()
                    or recipient_id != self._recipient_id or transfer_id != self._transfer_id
                    or self._recipient is None):
                raise TraceHandoffStateError("recipient is not recoverable")
            recipient = self._recipient
        return recipient

    def disconnect(self):
        """Signal lost recipient; keep claims until supervised reconciliation."""
        self._cancel.set()
        with self._lock:
            if self._phase not in ("retired", "disposed"):
                if (self._recipient is None and self._active is None
                        and self._phase in ("registered", "prepared", "offered", "admission_failed",
                                            "abort_pending")):
                    self._phase = "cancelled_unowned"
                else:
                    self._phase = "quarantined"
            child = self._active_child
        try:
            if child is not None:
                child.cancel()
            self._job.cancel()
        except BaseException:
            # A failed signal is not evidence that child custody is quiescent.
            with self._lock:
                self._phase = "quarantined"
            raise

    def abort_before_transfer(self, *, timeout):
        """Drain a job whose metadata was never exposed to a recipient."""
        if type(timeout) not in (int, float) or not 0 <= timeout <= 86400:
            raise ValueError("invalid timeout")
        token = object()
        values = None
        try:
            self._begin(("registered", "prepared", "offered", "admission_failed",
                         "cancelled_unowned", "abort_pending"),
                        "aborting", token, allow_cancel=True)
            self._cancel.set()
            with self._lock:
                if self._recipient is not None:
                    raise TraceHandoffStateError("recipient may have observed the job")
                values = (self._expected, self._published, self._expected_keys,
                          self._keys, self._expected_marker, self._marker, self._key_index)
                self._expected = self._published = self._expected_keys = self._keys = None
                self._expected_marker = self._marker = None
                self._key_index = None
            values = None
            self._job.cancel()
            if not self._job.dispose(timeout=timeout):
                with self._lock:
                    self._phase = "abort_pending"
                    self._active = None
                return False
            self._owner.retire(self._job)
            if self._manifest_claim is not None:
                self._manifest_claim.release()
                self._manifest_claim = None
            if self._host_guard is not None:
                self._host_guard.cancel_and_close_empty()
                self._host_request = None
            with self._lock:
                self._phase = "disposed"
                self._active = None
            return True
        except BaseException:
            self._fail(token)
            raise

    def dispose(self, *, timeout):
        """Drain after all release ACKs and explicit recipient disposal."""
        if type(timeout) not in (int, float) or not 0 <= timeout <= 86400:
            raise ValueError("invalid timeout")
        token = object()
        values = None
        try:
            self._begin(("disposal_confirmed",), "disposing", token)
            with self._lock:
                values = (self._expected, self._published, self._expected_keys,
                          self._keys, self._expected_marker, self._marker, self._key_index)
                self._expected = self._published = self._expected_keys = self._keys = None
                self._expected_marker = self._marker = None
                self._key_index = None
            values = None  # possible finalizers run outside the control lock
            if not self._job.dispose(timeout=timeout):
                with self._lock:
                    self._phase = "disposal_confirmed"
                    self._active = None
                return False
            self._owner.retire(self._job)
            self._manifest_claim.release()
            self._manifest_claim = None
            self._host_request.close()
            self._host_request = None
            with self._lock:
                self._phase = "disposed"
                self._active = None
            return True
        except BaseException:
            self._fail(token)
            raise


class _Recipient(_Noncopyable):
    def __init__(self, factory, bridge, token):
        if factory is not _FACTORY:
            raise TypeError("recipients are issued by the handoff owner")
        self._bridge, self._token = bridge, token

    def _check(self, *, completed=False):
        bridge = self._bridge
        allowed = ("owned", "completed") if completed else ("owned",)
        if (bridge._recipient is not self or bridge._phase not in allowed
                or bridge._cancel.is_set()):
            raise TraceHandoffStateError("recipient is unavailable")
        return bridge

    def manifest(self):
        bridge = self._bridge
        with bridge._lock:
            self._check(completed=True)
            if not _same_exact(bridge._published, bridge._expected):
                raise TraceHandoffStateError("manifest changed")
            manifest = bridge._published
            bridge._manifest_exposed = True
        return manifest

    def events(self):
        bridge = self._bridge
        with bridge._lock:
            self._check(completed=True)
            if not _same_exact(bridge._keys, bridge._expected_keys):
                raise TraceHandoffStateError("event keys changed")
            keys = bridge._keys
        return keys

    def ack_manifest(self):
        """Attest that the recipient has received the complete metadata."""
        bridge = self._bridge
        with bridge._lock:
            self._check(completed=True)
            if (not bridge._manifest_exposed
                    or not _same_exact(bridge._published, bridge._expected)):
                raise TraceHandoffStateError("manifest changed")
            changed = not bridge._manifest_received
            if changed:
                bridge._manifest_received = True
        return changed

    def _slot_index(self, key):
        bridge = self._bridge
        if (type(key) is not EventKey or type(key.slot_id) is not str
                or not 1 <= len(key.slot_id) <= 128 or type(key.slot_generation) is not int):
            raise TraceHandoffStateError("invalid event key")
        index = bridge._key_index.get((key.slot_id, key.slot_generation))
        if index is None or not _same_exact(key, bridge._expected_keys[index]):
            raise TraceHandoffStateError("event key does not belong to this job")
        return index

    def delivery_state(self, key):
        """Read the local outcome for this exact event without resending it."""
        bridge = self._bridge
        with bridge._lock:
            self._check(completed=True)
            index = self._slot_index(key)
            state = bridge._states[index]
        return state

    def send(self, key, sink):
        if not callable(sink):
            raise ValueError("sink must be callable")
        bridge = self._bridge
        token = object()
        child = None
        try:
            with bridge._lock:
                self._check()
                index = self._slot_index(key)
                if (not bridge._manifest_received or bridge._active is not None
                        or bridge._states[index] != "pending"):
                    raise TraceHandoffStateError("slot cannot be sent")
                bridge._active = token
                bridge._states[index] = "sending"
            expected = bridge._expected_keys[index]
            slot = bridge._job._batch._deliveries.lookup(expected.slot_id, expected.slot_generation)
            child = slot.adopt()
            if not _same_exact(child.identity, expected.delivery):
                raise TraceHandoffStateError("child identity changed")
            with bridge._lock:
                if bridge._cancel.is_set():
                    raise TraceHandoffStateError("handoff was cancelled")
                bridge._active_child = child
            try:
                child.send(sink)
            except HostBudgetExceeded:
                # The exact TraceDelivery implementation raises this only for
                # admission before the sink runs, and restores its ready state.
                # Any cancelled or ambiguous outcome remains quarantined.
                with bridge._lock:
                    if bridge._active is token and not bridge._cancel.is_set():
                        bridge._states[index] = "pending"
                        bridge._active_child = None
                        bridge._active = None
                raise
            with bridge._lock:
                if bridge._cancel.is_set():
                    raise TraceHandoffStateError("handoff was cancelled")
                bridge._states[index] = "sent"
                bridge._active_child = None
                bridge._active = None
        except BaseException:
            bridge._fail(token)
            raise
        finally:
            sink = child = None

    def ack_received(self, key):
        return self._ack(key, release=False)

    def ack_release(self, key):
        return self._ack(key, release=True)

    def _ack(self, key, *, release):
        bridge = self._bridge
        token = object()
        try:
            with bridge._lock:
                self._check(completed=True)
                index = self._slot_index(key)
                expected_state = "received" if release else "sent"
                next_state = "released" if release else "received"
                if bridge._active is not None:
                    raise TraceHandoffStateError("another slot operation is active")
                duplicate = ((release and bridge._states[index] == "released")
                             or (not release and bridge._states[index] in ("received", "released")))
                if not duplicate and bridge._states[index] != expected_state:
                    raise TraceHandoffStateError("slot is not ready for this acknowledgement")
                if not duplicate and release and bridge._phase != "completed":
                    raise TraceHandoffStateError("complete marker has not committed")
                if not duplicate:
                    bridge._active = token
            if duplicate:
                return False
            expected = bridge._expected_keys[index]
            child = bridge._job._batch._deliveries.lookup(
                expected.slot_id, expected.slot_generation).adopt()
            if not _same_exact(child.identity, expected.delivery):
                raise TraceHandoffStateError("child identity changed")
            changed = child.ack_release(expected.delivery) if release else child.ack_received(expected.delivery)
            with bridge._lock:
                if bridge._cancel.is_set():
                    raise TraceHandoffStateError("handoff was cancelled")
                bridge._states[index] = next_state
                bridge._active = None
            return changed
        except BaseException:
            bridge._fail(token)
            raise

    def complete(self):
        bridge = self._bridge
        with bridge._lock:
            self._check(completed=True)
            valid_states = ("received", "released") if bridge._phase == "completed" else ("received",)
            if (bridge._active is not None or not bridge._manifest_received
                    or any(state not in valid_states for state in bridge._states)
                    or not _same_exact(bridge._marker, bridge._expected_marker)):
                raise TraceHandoffStateError("complete receipt is unavailable")
            bridge._phase = "completed"
            marker = bridge._marker
        return marker

    def confirm_disposed(self):
        bridge = self._bridge
        with bridge._lock:
            self._check(completed=True)
            if (bridge._phase != "completed" or bridge._active is not None
                    or any(state != "released" for state in bridge._states)):
                raise TraceHandoffStateError("recipient disposal cannot be confirmed")
            bridge._phase = "disposal_confirmed"
