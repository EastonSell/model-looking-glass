"""Experimental bounded waiting layer for a trusted atomic handoff target."""

from collections.abc import Mapping
from dataclasses import dataclass
from queue import Full
from threading import Lock, get_ident
from uuid import uuid4
from weakref import WeakKeyDictionary

from .fair_admission import AdmissionEmpty, AdmissionFull, FairAdmissionQueue, UnknownPrincipal


HANDOFF_PROTOCOL = 'looking-glass.atomic-admission.v1'


class AdapterClosed(RuntimeError):
    """The waiting adapter no longer accepts requests."""


class HandoffProtocolError(RuntimeError):
    """The trusted target violated the synchronous one-claim handoff contract."""


class HandoffFailure(RuntimeError):
    """A handoff ended after claim; the owner must reconcile its outcome."""

    def __init__(self, ticket):
        super().__init__('handoff ended after a waiter was claimed; reconcile its outcome')
        self.ticket = ticket


class FatalHandoffFailure(BaseException):
    """A fatal handoff error after claim; the owner must reconcile it."""

    def __init__(self, ticket):
        super().__init__('fatal handoff error after a waiter was claimed')
        self.ticket = ticket


@dataclass(frozen=True)
class WaitingTicket:
    namespace: str
    sequence: int


@dataclass(frozen=True)
class TransferReceipt:
    ticket: WaitingTicket
    target_receipt: object


class RegistrationGuard:
    """Trusted owner's pre-retained handle for one authenticated submission."""

    def __init__(self, adapter, principal, payload):
        self._adapter = adapter

    def submit(self):
        try:
            self._adapter._submit_guard(self)
        except BaseException:
            self._adapter._mark_transition_unknown(self)
            raise

    def adopt(self):
        return self._adapter._adopt_guard(self)


@dataclass
class _Registration:
    principal: str
    payload: bytes | None
    ticket: WaitingTicket | None = None


class ExperimentalFairAdapter:
    """A waiting queue; its target must own atomic broker resource handoff."""

    def __init__(self, principal_teams, *, verifier, target, max_waiters,
                 max_waiters_per_team, max_waiting_bytes, max_item_bytes):
        if not callable(verifier):
            raise TypeError('a trusted credential verifier is required')
        if (getattr(target, 'handoff_protocol', None) != HANDOFF_PROTOCOL or
                not callable(getattr(target, 'accept_next', None))):
            raise TypeError('target lacks the atomic handoff contract')
        for value in (max_waiting_bytes, max_item_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('byte limits must be positive integers')
        if max_item_bytes > max_waiting_bytes:
            raise ValueError('item byte limit cannot exceed total waiting byte limit')
        target_limit = getattr(target, 'max_item_bytes', None)
        if target_limit is not None:
            if type(target_limit) is not int or target_limit <= 0:
                raise ValueError('target item byte limit must be a positive integer')
            if max_item_bytes > target_limit:
                raise ValueError('item byte limit exceeds the target limit')
        if not isinstance(principal_teams, Mapping):
            raise ValueError('principal-to-team mapping must be a mapping')
        principal_snapshot = dict(principal_teams)
        self._queue = FairAdmissionQueue(principal_snapshot, max_waiters, max_waiters_per_team)
        self._principals = frozenset(principal_snapshot)
        self._principal_teams = principal_snapshot
        self._verifier = verifier
        self._target = target
        self._max_waiting_bytes = max_waiting_bytes
        self._max_item_bytes = max_item_bytes
        self._namespace = uuid4().hex
        self._lock = Lock()
        self._owners = {}
        self._sizes = {}
        self._tickets = {}
        self._registration_roots = {}
        self._inflight_registration = None
        self._issued_registrations = WeakKeyDictionary()
        self._transition = None
        self._closing_roots = ()
        self._closed_tickets = ()
        self._closed = False
        self._indeterminate = False
        # A concrete broker target can reserve itself for exactly one waiting
        # adapter and validate the canonical team set. This hook is trusted,
        # like the target's atomic handoff implementation itself.
        binder = getattr(target, 'bind_waiting_queue', None)
        if binder is not None:
            if not callable(binder):
                raise TypeError('target waiting-queue binder must be callable')
            binder(frozenset(principal_snapshot.values()), max_item_bytes)

    def submit(self, credential, payload):
        guard = self.registration_guard(credential, payload)
        guard.submit()
        return guard.adopt()

    def registration_guard(self, credential, payload, *, team_id=None):
        """Create a handle without mutating the queue; retain it before submit."""
        principal = self._verify(credential)
        if team_id is not None and (type(team_id) is not str or self._principal_teams[principal] != team_id):
            raise PermissionError('verified principal does not belong to the required team')
        if type(payload) is not bytes:
            raise TypeError('waiting payload must be immutable bytes')
        if len(payload) > self._max_item_bytes:
            raise Full('waiting payload byte limit reached')
        guard = RegistrationGuard(self, principal, payload)
        with self._lock:
            self._issued_registrations[guard] = _Registration(principal, payload)
        return guard

    def _submit_guard(self, guard):
        with self._lock:
            self._check_transition_locked()
            if type(guard) is not RegistrationGuard or guard._adapter is not self:
                raise ValueError('foreign registration guard')
            registration = self._issued_registrations.get(guard)
            if registration is None:
                raise ValueError('unissued registration guard')
            if self._indeterminate:
                raise AdapterClosed('admission is indeterminate; process replacement required')
            if registration.ticket is not None:
                return
            if self._closed:
                raise AdapterClosed('waiting admission is closed')
            payload = registration.payload
            principal = registration.principal
            size = len(payload)
            if size > self._max_item_bytes or sum(self._sizes.values()) + size > self._max_waiting_bytes:
                raise Full('waiting payload byte limit reached')
            self._transition = guard
            self._inflight_registration = guard
            try:
                sequence = self._queue.submit(principal, payload)
            except AdmissionFull:
                self._inflight_registration = None
                self._transition = None
                raise
            except BaseException:
                self._indeterminate = True
                self._closed = True
                raise
            try:
                ticket = WaitingTicket(self._namespace, sequence)
                self._owners[sequence] = principal
                self._sizes[sequence] = size
                self._tickets[sequence] = ticket
                self._registration_roots[sequence] = guard
                registration.ticket = ticket
                registration.payload = None
                self._inflight_registration = None
                self._transition = None
            except BaseException:
                try:
                    self._owners.pop(sequence, None)
                    self._sizes.pop(sequence, None)
                    self._tickets.pop(sequence, None)
                    self._registration_roots.pop(sequence, None)
                    if not self._queue.cancel(sequence):
                        raise RuntimeError('unable to roll back pending ticket')
                    registration.ticket = None
                    registration.payload = payload
                    self._inflight_registration = None
                    self._transition = None
                except BaseException:
                    self._indeterminate = True
                    self._closed = True
                    raise
                raise

    def _mark_transition_unknown(self, marker):
        # This boundary intentionally takes no lock: an interrupted CPython
        # context-manager exit may have stranded that lock. The service owner
        # must stop and let a supervisor replace an unresponsive process.
        if self._transition is marker:
            self._indeterminate = True
            self._closed = True

    def _check_transition_locked(self):
        if self._transition is not None:
            self._indeterminate = True
            self._closed = True
            raise AdapterClosed('prior admission transition is indeterminate')

    def _adopt_guard(self, guard):
        with self._lock:
            if type(guard) is not RegistrationGuard or guard._adapter is not self:
                raise ValueError('foreign registration guard')
            registration = self._issued_registrations.get(guard)
            if registration is None:
                raise ValueError('unissued registration guard')
            if self._indeterminate:
                raise AdapterClosed('admission is indeterminate; process replacement required')
            ticket = registration.ticket
            if ticket is None:
                raise ValueError('registration has not completed')
        return ticket

    @property
    def indeterminate(self):
        """Sticky best-effort signal; an unresponsive lock needs a supervisor."""
        return self._indeterminate

    @property
    def inflight_registration(self):
        """Trusted owner recovery hook; never a cancellation or disposal proof."""
        with self._lock:
            guard = self._inflight_registration
        return guard

    def registration_roots(self):
        """Return the bounded roots retained for waiting or unknown work."""
        with self._lock:
            roots = tuple(self._registration_roots.values())
            roots += self._closing_roots
            if self._inflight_registration is not None:
                roots += (self._inflight_registration,)
        return roots

    def _verify(self, credential):
        principal = self._verifier(credential)
        if type(principal) is not str or principal not in self._principals:
            raise UnknownPrincipal('verified principal is not configured')
        return principal

    def cancel(self, credential, ticket):
        principal = self._verify(credential)
        if not self._owns_ticket_namespace(ticket):
            return False
        try:
            return self._cancel_verified(principal, ticket)
        except BaseException:
            self._mark_transition_unknown(ticket)
            raise

    def _cancel_verified(self, principal, ticket):
        with self._lock:
            self._check_transition_locked()
            if self._indeterminate:
                raise AdapterClosed('claim state is unknown; close and reconcile outstanding tickets')
            owner = self._owners.get(ticket.sequence)
            if owner is None:
                return False
            if owner != principal:
                raise PermissionError('ticket belongs to another principal')
            self._transition = ticket
            try:
                if not self._queue.cancel(ticket.sequence):
                    raise RuntimeError('pending ticket missing from admission queue')
                del self._owners[ticket.sequence]
                del self._sizes[ticket.sequence]
                del self._tickets[ticket.sequence]
                del self._registration_roots[ticket.sequence]
                self._transition = None
            except BaseException:
                self._indeterminate = True
                self._closed = True
                raise
        return True

    def _owns_ticket_namespace(self, ticket):
        if (type(ticket) is not WaitingTicket or type(ticket.namespace) is not str or
                type(ticket.sequence) is not int or ticket.sequence <= 0):
            raise ValueError('invalid waiting ticket')
        return ticket.namespace == self._namespace

    def recover_receipt(self, ticket):
        """Trusted service owner lookup after an interrupted handoff."""
        if not self._owns_ticket_namespace(ticket):
            return None
        recover = getattr(self._target, 'recover_receipt', None)
        if not callable(recover):
            raise TypeError('target does not support recoverable receipts')
        return recover(ticket.sequence)

    def acknowledge_receipt(self, ticket, handle):
        """Trusted owner confirms it has separately budgeted result custody."""
        if not self._owns_ticket_namespace(ticket):
            return False
        acknowledge = getattr(self._target, 'acknowledge_receipt', None)
        if not callable(acknowledge):
            raise TypeError('target does not support recoverable receipts')
        return acknowledge(ticket.sequence, handle)

    def transfer_next(self):
        with self._lock:
            self._check_transition_locked()
            if self._indeterminate:
                raise AdapterClosed('claim state is unknown; close and reconcile outstanding tickets')
            if self._closed:
                return None
        state_lock = Lock()
        handoff_thread = get_ident()
        active = True
        called = False
        claimed = None
        claim_marker = object()

        def claim(eligible):
            nonlocal called, claimed
            if get_ident() != handoff_thread:
                raise HandoffProtocolError('claim must run on the handoff thread')
            with state_lock:
                if not active or called:
                    raise HandoffProtocolError('claim must be synchronous and called at most once')
                called = True
                with self._lock:
                    self._check_transition_locked()
                    if self._closed:
                        return None
                    self._transition = claim_marker
                    try:
                        request = self._queue.claim_next(eligible)
                        claimed = self._tickets[request.ticket_id]
                        del self._sizes[request.ticket_id]
                        del self._owners[request.ticket_id]
                        del self._tickets[request.ticket_id]
                        del self._registration_roots[request.ticket_id]
                        self._transition = None
                        return request
                    except AdmissionEmpty:
                        self._transition = None
                        return None
                    except BaseException:
                        # The queue might have removed a waiter before an
                        # asynchronous interruption or failed allocation. Do
                        # not offer more work or guess which ticket moved. The
                        # owner must close and reconcile all outstanding IDs.
                        self._indeterminate = True
                        self._closed = True
                        raise

        try:
            receipt = self._target.accept_next(claim)
        except BaseException as exc:
            self._mark_transition_unknown(claim_marker)
            with state_lock:
                active = False
                ticket = claimed
            if ticket is not None:
                if isinstance(exc, Exception):
                    raise HandoffFailure(ticket) from exc
                raise FatalHandoffFailure(ticket) from exc
            raise
        finally:
            with state_lock:
                active = False
        if claimed is None:
            if receipt is not None:
                raise HandoffProtocolError('target returned a receipt without claiming a waiter')
            return None
        if receipt is None:
            raise HandoffFailure(claimed)
        try:
            return TransferReceipt(claimed, receipt)
        except Exception as exc:
            raise HandoffFailure(claimed) from exc
        except BaseException as exc:
            raise FatalHandoffFailure(claimed) from exc

    def close(self):
        """Cancel waiters and return their tickets for terminal bookkeeping."""
        marker = object()
        try:
            return self._close_waiters(marker)
        except BaseException:
            self._mark_transition_unknown(marker)
            raise

    def _close_waiters(self, marker):
        with self._lock:
            self._check_transition_locked()
            if self._indeterminate:
                raise AdapterClosed('admission is indeterminate; process replacement required')
            if self._closed:
                return ()
            tickets = tuple(self._tickets.values())
            self._closing_roots = tuple(self._registration_roots.values())
            self._transition = marker
            removed = self._queue.drain()
            self._closed = True
            self._closed_tickets = tickets
            self._owners.clear()
            self._sizes.clear()
            self._tickets.clear()
            self._registration_roots.clear()
            self._transition = None
            self._closing_roots = ()
        del removed
        return tickets

    @property
    def closed_tickets(self):
        """Trusted recovery snapshot after an orderly close, not after quarantine."""
        with self._lock:
            if self._indeterminate:
                raise AdapterClosed('admission is indeterminate; process replacement required')
            tickets = self._closed_tickets
        return tickets

    def __len__(self):
        with self._lock:
            return len(self._owners)

    @property
    def waiting_bytes(self):
        with self._lock:
            return sum(self._sizes.values())
