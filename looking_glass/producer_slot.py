"""Bounded, process-local adoption of a trusted CPU producer envelope.

The slot retains the exact child and host guards before their registrations;
it does not provide a broker, scheduler, transport, or physical memory limit.
"""

from threading import RLock
from uuid import uuid4

from .host_budget import HostBudget, InvalidHostHandle
from .trace_producer import ProducerEnvelope, ProducerProfile, _FACTORY_KEY, _STAGES


class ProducerSlotError(RuntimeError):
    """A slot is stale, unavailable, or requires supervised reconciliation."""


class ProducerSlotFull(ProducerSlotError):
    """The bounded slot registry has no free slot."""


def _valid_key(slot_id, generation):
    return (type(slot_id) is str and 0 < len(slot_id) <= 128
            and bool(slot_id.strip()) and type(generation) is int
            and 0 < generation < 2**63)


class ProducerSlotRegistry:
    """Retain a bounded set of caller-identified producer lifecycles."""

    def __init__(self, *, max_slots):
        if type(max_slots) is not int or not 0 < max_slots <= 65536:
            raise ValueError("max_slots must be an integer between 1 and 65536")
        self._max_slots = max_slots
        self._lock = RLock()
        self._slots = {}
        self._last_generation = 0

    def register(self, slot_id, generation):
        """Install a slot; caller records the ID and generation first."""
        if not _valid_key(slot_id, generation):
            raise ValueError("invalid producer slot identity")
        with self._lock:
            if slot_id in self._slots:
                raise ProducerSlotError("slot ID is already active")
            if len(self._slots) >= self._max_slots:
                raise ProducerSlotFull("producer slot capacity exhausted")
            if generation <= self._last_generation:
                raise ProducerSlotError("slot generation is not fresh")
            self._last_generation = generation
            slot = _ProducerSlot(self, slot_id, generation)
            self._slots[slot_id] = slot

    def lookup(self, slot_id, generation):
        if not _valid_key(slot_id, generation):
            raise ValueError("invalid producer slot identity")
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None or slot._generation != generation:
                raise ProducerSlotError("unknown or stale producer slot")
        return slot

    def active_slots(self):
        with self._lock:
            slots = tuple((slot._slot_id, slot._generation,
                           "quarantined" if slot._quarantined else slot._state)
                          for slot in self._slots.values())
        return slots


class _ProducerSlot:
    """A registry-issued capability; identity is checked on every operation."""

    def __init__(self, registry, slot_id, generation):
        self._registry = registry
        self._slot_id = slot_id
        self._generation = generation
        self._state = "reserved"
        self._active = None
        self._cancel_requested = False
        self._quarantined = False
        self._construction_clean = True
        self._envelope = None
        self._request_guard = None
        self._claim_guards = {}

    def _check_locked(self):
        if self._registry._slots.get(self._slot_id) is not self:
            raise ProducerSlotError("stale or foreign producer slot")

    @property
    def state(self):
        with self._registry._lock:
            self._check_locked()
            state = "quarantined" if self._quarantined else self._state
        return state

    def guards(self):
        """Expose retained capabilities to a trusted recovery coordinator."""
        with self._registry._lock:
            self._check_locked()
            if self._active is not None:
                raise ProducerSlotError("producer is not quiescent")
            request_guard = self._request_guard
            claim_guards = tuple(self._claim_guards.items())
        return request_guard, claim_guards

    def owned_envelope(self):
        """Return the exact quiescent child, which may be partially initialized."""
        with self._registry._lock:
            self._check_locked()
            if self._active is not None or self._envelope is None:
                raise ProducerSlotError("producer is not quiescent")
            child = self._envelope
        return child

    def adopt(self):
        """Return the same ready object; caller assignment can be retried."""
        with self._registry._lock:
            self._check_locked()
            if (self._state != "ready" or self._active is not None
                    or self._cancel_requested or self._quarantined):
                raise ProducerSlotError("producer is not available for adoption")
            child = self._envelope
        return child

    def _prepare(self, token, budget, request_id, profile):
        """Admit one preinstalled child, retaining every guard before use."""
        if type(budget) is not HostBudget or type(profile) is not ProducerProfile:
            raise ValueError("a HostBudget and ProducerProfile are required")
        guard = budget.request_guard(request_id)
        owner = f"producer-{uuid4().hex}"
        claims = {}
        try:
            with self._registry._lock:
                self._check_locked()
                if self._active is not token:
                    raise ProducerSlotError("construction ownership changed")
                self._request_guard = guard
                self._construction_clean = False
            guard.open()
            request = guard.adopt()
            for name in _STAGES:
                limit = getattr(profile, name)
                claim_guard = request.claim_guard(owner, name, limit.nbytes,
                                                  items=limit.items)
                with self._registry._lock:
                    self._check_locked()
                    if self._active is not token:
                        raise ProducerSlotError("construction ownership changed")
                    self._claim_guards[name] = claim_guard
                claim_guard.reserve()
                claims[name] = claim_guard.adopt()
            ProducerEnvelope.__init__(self._envelope, _FACTORY_KEY, budget,
                                      request, profile, claims)
        except BaseException:
            # No forward has started. Release only the capabilities retained by
            # this slot; transferred/stale guards cannot release a new owner.
            try:
                try:
                    request = guard.adopt()
                except InvalidHostHandle:
                    guard.cancel_and_close_empty()
                else:
                    request.cancel()
                    with self._registry._lock:
                        guarded_claims = tuple(self._claim_guards.values())
                    for claim_guard in guarded_claims:
                        try:
                            claim = claim_guard.adopt()
                        except InvalidHostHandle:
                            continue
                        claim.release()
                    guard.cancel_and_close_empty()
                with self._registry._lock:
                    if self._registry._slots.get(self._slot_id) is self:
                        self._construction_clean = True
            except BaseException:
                # Do not infer that a failed cleanup disposed any storage.
                pass
            raise

    def construct(self, budget, request_id, profile):
        """Construct in place after the caller has registered a known slot."""
        token = object()
        failure = None
        try:
            with self._registry._lock:
                self._check_locked()
                if (self._state != "reserved" or self._active is not None
                        or self._quarantined):
                    raise ProducerSlotError("slot is not available for construction")
                self._active = token
                self._state = "constructing"
                self._envelope = object.__new__(ProducerEnvelope)
                child = self._envelope
            self._prepare(token, budget, request_id, profile)
            with self._registry._lock:
                self._check_locked()
                if self._quarantined:
                    raise ProducerSlotError("slot requires supervision")
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
                        raise ProducerSlotError("slot requires supervision")
                    self._state = "cancelled"
                    self._active = None
                raise ProducerSlotError("producer construction was cancelled")
        except BaseException as error:
            with self._registry._lock:
                if (self._registry._slots.get(self._slot_id) is self
                        and self._active is token):
                    if self._construction_clean and not self._quarantined:
                        self._state = "failed_clean"
                    else:
                        self._quarantined = True
                        self._state = "quarantined"
                    self._active = None
            try:
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
            except BaseException:
                pass
            failure = "producer construction failed or was interrupted"
        if failure is not None:
            raise ProducerSlotError(failure) from None

    def cancel(self):
        """Block adoption and cooperatively signal a completed producer."""
        token = object()
        entered = False
        failure = False
        try:
            with self._registry._lock:
                self._check_locked()
                if self._quarantined:
                    raise ProducerSlotError("slot requires supervision")
                if self._cancel_requested:
                    disposition = "already"
                else:
                    if self._state not in ("reserved", "constructing", "ready"):
                        raise ProducerSlotError("slot cannot be cancelled")
                    entered = True
                    self._cancel_requested = True
                    if self._state == "constructing":
                        disposition = "deferred"
                    elif self._state == "reserved":
                        self._state = "cancelled"
                        disposition = "deferred"
                    else:
                        self._active = token
                        self._state = "cancelling"
                        child = self._envelope
                        disposition = "signal"
            if disposition == "already":
                return False
            if disposition == "deferred":
                return True
            child.cancel()
            with self._registry._lock:
                self._check_locked()
                if self._quarantined:
                    raise ProducerSlotError("slot requires supervision")
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
            raise ProducerSlotError("slot cancellation failed or was interrupted") from None

    def retire(self):
        """Retire only an empty slot or an explicitly closed producer."""
        with self._registry._lock:
            self._check_locked()
            if (self._active is not None or self._quarantined
                    or self._state in ("constructing", "cancelling")):
                raise ProducerSlotError("producer is active or quarantined")
            child = self._envelope
            if child is None or self._state == "failed_clean":
                del self._registry._slots[self._slot_id]
                self._state = "retired"
                return True
        if not child.closed:
            raise ProducerSlotError("producer has not been explicitly closed")
        with self._registry._lock:
            self._check_locked()
            if (self._active is not None or self._quarantined
                    or self._envelope is not child):
                raise ProducerSlotError("producer changed during retirement")
            del self._registry._slots[self._slot_id]
            self._state = "retired"
        return True
