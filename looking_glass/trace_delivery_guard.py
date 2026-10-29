"""Bounded, trusted process-local adoption of single-event deliveries.

The registry retains the exact object before its constructor can claim memory.
It does not provide a broker, IPC transport, or automatic claim cleanup.
"""

from threading import RLock

from .host_budget import HostBudget
from .trace_store import HostTraceStore
from .trace_transport import DeliveryConstructionError, TraceDelivery


class DeliverySlotError(RuntimeError):
    """A slot is stale, unavailable, or needs supervised reconciliation."""


class DeliverySlotFull(DeliverySlotError):
    """The registry has no free bounded slot."""


def _valid_key(slot_id, generation):
    return (type(slot_id) is str and 0 < len(slot_id) <= 128
            and bool(slot_id.strip()) and type(generation) is int
            and 0 < generation < 2**63)


class DeliverySlotRegistry:
    """Bound the number of slots within one trusted registry lifetime.

    The caller records each slot ID and monotonically increasing generation
    before registration; the pair identifies a possibly interrupted operation.
    """

    def __init__(self, *, max_slots):
        if type(max_slots) is not int or not 0 < max_slots <= 65536:
            raise ValueError("max_slots must be an integer between 1 and 65536")
        self._max_slots = max_slots
        self._lock = RLock()
        self._slots = {}
        self._last_generation = 0

    def register(self, slot_id, generation):
        if not _valid_key(slot_id, generation):
            raise ValueError("invalid slot identity")
        with self._lock:
            if slot_id in self._slots:
                raise DeliverySlotError("slot ID is already active")
            if len(self._slots) >= self._max_slots:
                raise DeliverySlotFull("delivery slot capacity exhausted")
            if generation <= self._last_generation:
                raise DeliverySlotError("slot generation is not fresh")
            # Consume the generation even if slot installation is interrupted.
            # There is no claim-bearing delivery before insertion.
            self._last_generation = generation
            slot = _DeliverySlot(self, slot_id, generation)
            self._slots[slot_id] = slot

    def lookup(self, slot_id, generation):
        if not _valid_key(slot_id, generation):
            raise ValueError("invalid slot identity")
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None or slot._generation != generation:
                raise DeliverySlotError("unknown or stale delivery slot")
        return slot

    def active_slots(self):
        """Return bounded identity/state metadata, not child ownership."""
        with self._lock:
            slots = tuple((slot._slot_id, slot._generation,
                           "quarantined" if slot._quarantined else slot._state)
                          for slot in self._slots.values())
        return slots


class _DeliverySlot:
    """Capability issued by its registry; never construct one directly."""

    def __init__(self, registry, slot_id, generation):
        self._registry = registry
        self._slot_id = slot_id
        self._generation = generation
        self._state = "reserved"
        self._delivery = None
        self._failure_guard = None
        self._active = None
        self._cancel_requested = False
        self._quarantined = False

    def _check_locked(self):
        if self._registry._slots.get(self._slot_id) is not self:
            raise DeliverySlotError("stale or foreign delivery slot")

    @property
    def state(self):
        with self._registry._lock:
            self._check_locked()
            state = "quarantined" if self._quarantined else self._state
        return state

    @property
    def failure_guard(self):
        with self._registry._lock:
            self._check_locked()
            guard = self._failure_guard
        return guard

    def owned_delivery(self):
        """Return the exact object for a trusted supervisor after quiescence.

        A quarantined object may be partially initialized; its methods are not
        necessarily safe. This accessor does not establish claim disposal.
        """
        with self._registry._lock:
            self._check_locked()
            if self._active is not None or self._delivery is None:
                raise DeliverySlotError("delivery is not quiescent")
            child = self._delivery
        return child

    def adopt(self):
        """Retrieve the same ready object; caller-assignment loss is retryable."""
        with self._registry._lock:
            self._check_locked()
            if (self._state != "ready" or self._active is not None
                    or self._cancel_requested or self._quarantined):
                raise DeliverySlotError("delivery is not available for adoption")
            child = self._delivery
        return child

    def construct(self, budget, store, *, broker_id, job_id, team_id, run_id,
                  request_id, sample_id, batch_position, max_payload_bytes,
                  metadata_reserve_bytes=65536):
        if type(budget) is not HostBudget or type(store) is not HostTraceStore:
            raise ValueError("expected trusted concrete budget and trace store")
        token = object()
        failure = None
        try:
            with self._registry._lock:
                self._check_locked()
                if (self._state != "reserved" or self._active is not None
                        or self._quarantined):
                    raise DeliverySlotError("slot is not available for construction")
                self._active = token
                self._state = "constructing"
                # Install the exact object before its constructor can reserve.
                self._delivery = object.__new__(TraceDelivery)
                child = self._delivery
            TraceDelivery.__init__(child, budget, store, broker_id=broker_id,
                                   job_id=job_id, team_id=team_id, run_id=run_id,
                                   request_id=request_id, sample_id=sample_id,
                                   batch_position=batch_position,
                                   max_payload_bytes=max_payload_bytes,
                                   metadata_reserve_bytes=metadata_reserve_bytes)
            with self._registry._lock:
                self._check_locked()
                if self._quarantined:
                    raise DeliverySlotError("slot requires supervision")
                if self._cancel_requested:
                    self._state = "cancelling"
                    cancelling = True
                else:
                    self._state = "ready"
                    self._active = None
                    cancelling = False
            if cancelling:
                child.cancel()
                with self._registry._lock:
                    self._check_locked()
                    if self._quarantined:
                        raise DeliverySlotError("slot requires supervision")
                    self._state = "cancelled"
                    self._active = None
                raise DeliverySlotError("delivery construction was cancelled")
        except BaseException as error:
            with self._registry._lock:
                if (self._registry._slots.get(self._slot_id) is self
                        and self._active is token):
                    if isinstance(error, DeliveryConstructionError):
                        self._failure_guard = error.guard
                    if (not self._quarantined
                            and not isinstance(error, DeliveryConstructionError)
                            and (self._delivery is None
                                 or getattr(self._delivery, "_construction_clean", False) is True)):
                        self._state = "failed_clean"
                    else:
                        self._quarantined = True
                        self._state = "quarantined"
                    self._active = None
            # Do not retain exception frames that could contain host payloads.
            try:
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
            except BaseException:
                pass
            failure = "delivery construction failed or was interrupted"
        if failure is not None:
            raise DeliverySlotError(failure) from None

    def cancel(self):
        """Record cancellation and signal a completed child outside the lock.

        Child signalling can itself block. During construction the signal is
        deferred until the constructor returns. A request is not a drain.
        """
        token = object()
        entered = False
        failure = False
        try:
            with self._registry._lock:
                self._check_locked()
                if self._quarantined:
                    raise DeliverySlotError("slot requires supervision")
                if self._cancel_requested:
                    disposition = "already"
                else:
                    if self._state not in ("reserved", "constructing", "ready"):
                        raise DeliverySlotError("slot cannot be cancelled")
                    entered = True
                    self._cancel_requested = True
                    if self._state == "constructing":
                        disposition = "deferred"
                    elif self._state == "reserved":
                        self._state = "cancelled"
                        disposition = "deferred"
                    else:
                        if self._active is not None:
                            raise DeliverySlotError("slot cannot be cancelled")
                        self._active = token
                        self._state = "cancelling"
                        child = self._delivery
                        disposition = "signal"
            if disposition == "already":
                return False
            if disposition == "deferred":
                return True
            child.cancel()
            with self._registry._lock:
                self._check_locked()
                if self._quarantined:
                    raise DeliverySlotError("slot requires supervision")
                self._state = "cancelled"
                self._active = None
            return True
        except BaseException:
            with self._registry._lock:
                if entered and self._registry._slots.get(self._slot_id) is self:
                    self._quarantined = True
                    self._state = "quarantined"
                    if self._active is token:
                        self._active = None
            failure = True
        if failure:
            raise DeliverySlotError("slot cancellation failed or was interrupted") from None

    def retire(self):
        """Retire only a slot with no child or a confirmed drained child."""
        with self._registry._lock:
            self._check_locked()
            if (self._active is not None or self._quarantined
                    or self._state in ("constructing", "cancelling")):
                raise DeliverySlotError("delivery is active or quarantined")
            child = self._delivery
            if child is None or self._state == "failed_clean":
                del self._registry._slots[self._slot_id]
                self._state = "retired"
                retired = True
            else:
                retired = False
        if retired:
            return True
        # A timeout or failed drain does not dispose anything or free a slot.
        if not child.drain(0):
            return False
        with self._registry._lock:
            self._check_locked()
            if (self._active is not None or self._quarantined
                    or self._delivery is not child):
                retired = False
            else:
                del self._registry._slots[self._slot_id]
                self._state = "retired"
                retired = True
        return retired
