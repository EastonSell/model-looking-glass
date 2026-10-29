"""Trusted, CPU-only reconciliation of a narrowly scoped quarantined handoff.

The caller must exclusively control the complete local owner and prove that
all relevant calls and external aliases have stopped before attestation.
"""
from dataclasses import dataclass
import math

from .cpu_trace_owner import CpuTraceOwner, _CpuTraceJob
from .host_budget import HostBudgetExceeded, HostClaim, HostOperation, HostRequest
from .trace_batch_handoff import EventKey, SealedBatchRegistry, _HandoffJob, _Noncopyable, _hex
from .trace_batch_owner import _same_exact
from .trace_transport import TraceDelivery


class RecoveryStateError(RuntimeError):
    """The recovery is stale, incomplete, or requires process supervision."""


@dataclass(frozen=True, slots=True)
class RecoveryScope:
    broker_id: str
    process_generation: str
    job_id: int
    registration_id: str
    generation: int
    transfer_id: str
    recovery_id: str
    manifest_request_id: str
    manifest_request_generation: int
    manifest_claim_id: int
    inventory_claim_id: int
    operation_id: int
    events: tuple[EventKey, ...]


_FACTORY = object()


def register_recovery(bridge, *, broker_id, process_generation, job_id,
                      registration_id, generation, transfer_id, recovery_id):
    """Root one caller-keyed recovery on an exact quarantined bridge.

    The supervisor must prevent any new calls through job, child, sink, or
    recipient aliases. The later attestation is required before disposal.
    """
    if (type(bridge) is not _HandoffJob or not _hex(broker_id)
            or not _hex(process_generation) or type(job_id) is not int
            or not 1 <= job_id < 2**63 or not _hex(registration_id)
            or type(generation) is not int or not 1 <= generation < 2**63
            or not _hex(transfer_id) or not _hex(recovery_id)):
        raise RecoveryStateError("invalid recovery identity")
    registry = bridge._registry
    if type(registry) is not SealedBatchRegistry:
        raise RecoveryStateError("invalid handoff registry")
    with bridge._lock:
        if (bridge._phase != "quarantined" or not bridge._cancel.is_set()
                or bridge._active is not None or bridge._active_child is not None
                or bridge._recipient is None or bridge._manifest_op is not None
                or type(bridge._count) is not int or not 1 <= bridge._count <= 256
                or type(bridge._host_request) is not HostRequest
                or type(bridge._manifest_claim) is not HostClaim
                or getattr(bridge, "_recovery", None) is not None
                or (registry._broker_id, registry._process_generation, bridge._job_id,
                    bridge._registration_id, bridge._generation, bridge._transfer_id)
                != (broker_id, process_generation, job_id, registration_id,
                    generation, transfer_id)):
            raise RecoveryStateError("handoff is not eligible for recovery")
        job, owner = bridge._job, bridge._owner
        if type(job) is not _CpuTraceJob or type(owner) is not CpuTraceOwner:
            raise RecoveryStateError("invalid exact job or owner")
        with job._lock:
            if job._phase != "sealed" or job._active is not None or job._batch is None:
                raise RecoveryStateError("job is not a quiescent sealed owner")
            with owner._lock:
                if (owner._job is not job or job._owner is not owner
                        or owner._budget is not registry._budget
                        or job._batch._registry is not owner._batches):
                    raise RecoveryStateError("job owner changed")
                with registry._lock:
                    if (registry._jobs.get(job_id) is not bridge
                            or getattr(job, "_trace_handoff", None) is not bridge):
                        raise RecoveryStateError("handoff root changed")
                    recovery = _Recovery(
                        _FACTORY, bridge, recovery_id,
                        (broker_id, process_generation, job_id, registration_id,
                         generation, transfer_id))
                    # Install before returning so the caller's known ID can
                    # recover an interrupted return without making a second owner.
                    bridge._recovery = recovery


def lookup_recovery(bridge, *, recovery_id):
    if type(bridge) is not _HandoffJob or not _hex(recovery_id):
        raise RecoveryStateError("invalid recovery key")
    with bridge._lock:
        recovery = getattr(bridge, "_recovery", None)
        if (type(recovery) is not _Recovery or recovery._bridge is not bridge
                or recovery._recovery_id != recovery_id or bridge._phase == "retired"):
            raise RecoveryStateError("unknown or stale recovery")
        recovery._check_binding_locked()
        with recovery._registry._lock:
            if recovery._registry._jobs.get(bridge._job_id) is not bridge:
                raise RecoveryStateError("recovery root is no longer registered")
    return recovery


class _Recovery(_Noncopyable):
    def __init__(self, factory, bridge, recovery_id, identity):
        if factory is not _FACTORY:
            raise TypeError("recovery capabilities are issued by a rooted handoff")
        self._bridge, self._recovery_id = bridge, recovery_id
        self._identity = identity
        self._registry, self._owner, self._job = bridge._registry, bridge._owner, bridge._job
        self._batch, self._deliveries = bridge._job._batch, bridge._job._batch._deliveries
        self._request, self._manifest_claim, self._count = (
            bridge._host_request, bridge._manifest_claim, bridge._count)
        self._state = "unbound"
        self._guard = self._binding_claim = self._operation = None
        self._children = self._index = self._scope = self._expected_scope = None
        self._scope_exposed = self._scope_acknowledged = False
        self._drain_attempted = False
        self._error = None

    @property
    def state(self):
        bridge = self._bridge
        with bridge._lock:
            state = self._state
        return state

    def _check_binding_locked(self):
        bridge = self._bridge
        if (getattr(bridge, "_recovery", None) is not self
                or bridge._registry is not self._registry or bridge._owner is not self._owner
                or bridge._job is not self._job or type(bridge._count) is not int
                or bridge._count != self._count
                or (self._state != "disposed" and bridge._active is None
                    and (bridge._host_request is not self._request
                         or bridge._manifest_claim is not self._manifest_claim))):
            self._state = "uncertain"
            bridge._phase = "quarantined"
            raise RecoveryStateError("rooted recovery identity or custody changed")
        actual = (self._registry._broker_id, self._registry._process_generation,
                  bridge._job_id, bridge._registration_id, bridge._generation,
                  bridge._transfer_id)
        if not _same_exact(actual, self._identity):
            self._state = "uncertain"
            bridge._phase = "quarantined"
            raise RecoveryStateError("rooted recovery identity or custody changed")

    def _begin(self, states, token):
        bridge = self._bridge
        with bridge._lock:
            self._check_binding_locked()
            if (getattr(bridge, "_recovery", None) is not self
                    or bridge._phase != "quarantined" or not bridge._cancel.is_set()
                    or bridge._active is not None or self._state not in states):
                raise RecoveryStateError("recovery is not in the required state")
            with self._registry._lock:
                if self._registry._jobs.get(bridge._job_id) is not bridge:
                    self._state = "uncertain"
                    raise RecoveryStateError("recovery root is no longer registered")
            bridge._active = token
            bridge._phase = "recovering"

    def _uncertain(self, token, error):
        bridge = self._bridge
        with bridge._lock:
            if bridge._active is token:
                # The original error may retain an arbitrary object graph.
                # Root it and keep every remaining claim for supervision.
                self._error = error
                self._state = "uncertain"
                bridge._phase = "quarantined"
                bridge._active = None

    def _capacity_denied(self, token):
        bridge = self._bridge
        with bridge._lock:
            if bridge._active is not token or bridge._phase != "recovering":
                raise RecoveryStateError("recovery ownership changed during admission")
            self._state = "unbound"
            bridge._phase = "quarantined"
            bridge._active = None

    def bind(self):
        """Reserve and pin inventory before copying the exact child set.

        A normal capacity denial returns False and can be retried. The owner
        must already have stopped concurrent uses of the rooted handoff.
        """
        bridge = self._bridge
        token = object()
        try:
            self._begin(("unbound",), token)
            request, claim = self._request, self._manifest_claim
            budget = self._registry._budget
            with budget._lock:
                request_record = budget._request(request)
                claim_record = budget._claim(claim)
                if (request_record.cancelled or claim_record.request is not request
                        or claim_record.stage != "manifest"
                        or claim_record.nbytes != (2 * bridge._count + 1) * 65536
                        or claim_record.items != 2 * bridge._count + 1
                        or budget._operation_count(claim._claim_id) != 0):
                    raise RecoveryStateError("manifest claim is not uniquely recoverable")
                request_generation = request_record.generation
            if self._guard is None:
                self._guard = request.claim_guard(
                    "recovery", "inventory", (3 * bridge._count + 4) * 65536,
                    items=3 * bridge._count + 4)
            try:
                self._guard.reserve()
            except HostBudgetExceeded:
                # A return can be interrupted after the guard committed. Only
                # a guard with no issued handle proves pre-commit denial.
                with budget._lock:
                    if self._guard._handle is not None:
                        raise
                self._guard = None
                self._capacity_denied(token)
                return False
            self._binding_claim = self._guard.adopt()
            try:
                # Both claims belong to this exact request generation. A lost
                # return here is uncertain and is never guessed to be unpinned.
                self._operation = request.begin_operation(
                    f"handoff-recovery-{self._recovery_id}",
                    (claim, self._binding_claim))
            except HostBudgetExceeded:
                # A patched or interrupted return may have committed an
                # operation. Verify neither exact claim is pinned before
                # treating this as a pre-copy denial.
                with budget._lock:
                    if (budget._operation_count(claim._claim_id)
                            or budget._operation_count(self._binding_claim._claim_id)):
                        raise
                self._binding_claim.release()
                self._binding_claim = None
                self._guard = None
                self._capacity_denied(token)
                return False
            if type(self._operation) is not HostOperation:
                raise RecoveryStateError("invalid exact recovery operation")
            if (type(bridge._expected_keys) is not tuple
                    or len(bridge._expected_keys) != bridge._count
                    or not _same_exact(bridge._keys, bridge._expected_keys)):
                raise RecoveryStateError("handoff keys changed")
            batch = self._job._batch
            if (batch is not self._batch or batch._deliveries is not self._deliveries
                    or not batch._sources_disposed
                    or type(batch._keys) is not tuple or len(batch._keys) != bridge._count):
                raise RecoveryStateError("batch source custody changed")
            children = []
            for key, pair in zip(bridge._expected_keys, batch._keys):
                if (type(key) is not EventKey or type(pair) is not tuple or len(pair) != 2
                        or not _same_exact(pair, (key.slot_id, key.slot_generation))):
                    raise RecoveryStateError("slot identity changed")
                slot = batch._deliveries.lookup(*pair)
                child = slot.owned_delivery()
                if type(child) is not TraceDelivery or not _same_exact(child.identity, key.delivery):
                    raise RecoveryStateError("delivery identity changed")
                children.append((pair[0], pair[1], slot, child))
            self._children = tuple(children)
            children = None
            self._index = {item[0]: item for item in self._children}
            if len(self._index) != bridge._count:
                raise RecoveryStateError("slot identities are not unique")
            arguments = (bridge._registry._broker_id, bridge._registry._process_generation,
                         bridge._job_id, bridge._registration_id, bridge._generation,
                         bridge._transfer_id, self._recovery_id, request._request_id,
                         request_generation, claim._claim_id,
                         self._binding_claim._claim_id, self._operation._operation_id)
            self._expected_scope = RecoveryScope(*arguments, bridge._expected_keys)
            self._scope = RecoveryScope(*arguments, bridge._keys)
            if not _same_exact(self._scope, self._expected_scope):
                raise RecoveryStateError("recovery scope changed")
            with bridge._lock:
                if bridge._active is not token or bridge._phase != "recovering":
                    raise RecoveryStateError("recovery ownership changed")
                self._state = "bound"
                bridge._phase = "quarantined"
                bridge._active = None
            return True
        except BaseException as error:
            self._uncertain(token, error)
            raise

    def scope(self):
        """Expose the pinned identity inventory to the trusted supervisor."""
        bridge = self._bridge
        with bridge._lock:
            self._check_binding_locked()
            if (bridge._phase != "quarantined" or bridge._active is not None
                    or self._state != "bound" or not _same_exact(self._scope, self._expected_scope)):
                raise RecoveryStateError("recovery scope is unavailable")
            self._scope_exposed = True
            scope = self._scope
        return scope

    def ack_scope(self, scope):
        bridge = self._bridge
        with bridge._lock:
            self._check_binding_locked()
            if (bridge._phase != "quarantined" or bridge._active is not None
                    or self._state != "bound" or not self._scope_exposed
                    or not _same_exact(scope, self._expected_scope)
                    or not _same_exact(self._scope, self._expected_scope)):
                raise RecoveryStateError("recovery scope does not match")
            changed = not self._scope_acknowledged
            self._scope_acknowledged = True
        return changed

    def confirm_disposed(self):
        """Attest quiescence and disposal or separately charged aliases."""
        bridge = self._bridge
        with bridge._lock:
            self._check_binding_locked()
            if (bridge._phase != "quarantined" or bridge._active is not None
                    or self._state != "bound" or not self._scope_acknowledged):
                raise RecoveryStateError("recovery disposal cannot be attested")
            self._state = "attested"

    def _verify_children(self):
        """Reject replacement and require proof for each previously retired slot."""
        batch = self._job._batch
        if batch is not self._batch or batch._deliveries is not self._deliveries:
            raise RecoveryStateError("batch ownership changed")
        registry = batch._deliveries
        retired = []
        with registry._lock:
            if any(slot_id not in self._index for slot_id in registry._slots):
                raise RecoveryStateError("unexpected delivery slot")
            for index, (slot_id, generation, slot, child) in enumerate(self._children):
                current = registry._slots.get(slot_id)
                if (slot._generation != generation or slot._delivery is not child
                        or not _same_exact(child.identity, self._expected_scope.events[index].delivery)):
                    raise RecoveryStateError("delivery capability changed")
                if current is None:
                    if not self._drain_attempted or slot._state != "retired":
                        raise RecoveryStateError("delivery retirement is uncertain")
                    retired.append(child)
                elif current is not slot:
                    raise RecoveryStateError("delivery capability changed")
        for child in retired:
            if not child.drain(0):
                return False
        return True

    def drain(self, *, timeout):
        """Drain a certified local job; a normal incomplete drain is retryable."""
        if (type(timeout) not in (int, float) or not 0 <= timeout <= 86400
                or not math.isfinite(timeout)):
            raise ValueError("invalid timeout")
        bridge = self._bridge
        token = object()
        values = None
        try:
            self._begin(("attested", "retryable"), token)
            if not self._verify_children():
                with bridge._lock:
                    self._state = "retryable"
                    bridge._phase = "quarantined"
                    bridge._active = None
                return False
            self._drain_attempted = True
            if not self._job.dispose(timeout=timeout):
                with bridge._lock:
                    self._state = "retryable"
                    bridge._phase = "quarantined"
                    bridge._active = None
                return False
            self._owner.retire(self._job)
            with bridge._lock:
                if bridge._active is not token or bridge._phase != "recovering":
                    raise RecoveryStateError("recovery ownership changed during drain")
                values = (bridge._expected, bridge._published, bridge._expected_keys,
                          bridge._keys, bridge._expected_marker, bridge._marker,
                          bridge._key_index, self._children, self._index,
                          self._scope, self._expected_scope, self._batch,
                          self._deliveries)
                bridge._expected = bridge._published = bridge._expected_keys = bridge._keys = None
                bridge._expected_marker = bridge._marker = bridge._key_index = None
                self._children = self._index = self._scope = self._expected_scope = None
                self._batch = self._deliveries = None
            values = None  # destructors run outside the control lock, while pinned
            live = self._operation.finish()
            self._operation = None
            if not live:
                raise RecoveryStateError("manifest request was cancelled during recovery")
            self._binding_claim.release()
            self._binding_claim = None
            bridge._manifest_claim.release()
            bridge._manifest_claim = None
            bridge._host_request.close()
            bridge._host_request = None
            with bridge._lock:
                if bridge._active is not token or bridge._phase != "recovering":
                    raise RecoveryStateError("recovery ownership changed during release")
                self._state = "disposed"
                bridge._phase = "disposed"
                bridge._active = None
            return True
        except BaseException as error:
            self._uncertain(token, error)
            raise
