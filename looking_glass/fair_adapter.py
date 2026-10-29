"""Experimental bounded waiting layer for a trusted atomic handoff target."""

from collections.abc import Mapping
from dataclasses import dataclass
from queue import Full
from threading import Lock, get_ident
from uuid import uuid4

from .fair_admission import AdmissionEmpty, FairAdmissionQueue, UnknownPrincipal


HANDOFF_PROTOCOL = 'looking-glass.atomic-admission.v1'


class AdapterClosed(RuntimeError):
    """The waiting adapter no longer accepts requests."""


class HandoffProtocolError(RuntimeError):
    """The trusted target violated the synchronous one-claim handoff contract."""


class HandoffFailure(RuntimeError):
    """A target failed after claiming a ticket; its owner must record failure."""

    def __init__(self, ticket):
        super().__init__('handoff failed after a waiter was claimed')
        self.ticket = ticket


class FatalHandoffFailure(BaseException):
    """A fatal target error after claim; supervisors must terminally fail it."""

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
        self._verifier = verifier
        self._target = target
        self._max_waiting_bytes = max_waiting_bytes
        self._max_item_bytes = max_item_bytes
        self._namespace = uuid4().hex
        self._lock = Lock()
        self._owners = {}
        self._sizes = {}
        self._tickets = {}
        self._closed = False
        # A concrete broker target can reserve itself for exactly one waiting
        # adapter and validate the canonical team set. This hook is trusted,
        # like the target's atomic handoff implementation itself.
        binder = getattr(target, 'bind_waiting_queue', None)
        if binder is not None:
            if not callable(binder):
                raise TypeError('target waiting-queue binder must be callable')
            binder(frozenset(principal_snapshot.values()), max_item_bytes)

    def submit(self, credential, payload):
        principal = self._verify(credential)
        if type(payload) is not bytes:
            raise TypeError('waiting payload must be immutable bytes')
        with self._lock:
            if self._closed:
                raise AdapterClosed('waiting admission is closed')
            size = len(payload)
            if size > self._max_item_bytes or sum(self._sizes.values()) + size > self._max_waiting_bytes:
                raise Full('waiting payload byte limit reached')
            sequence = self._queue.submit(principal, payload)
            try:
                ticket = WaitingTicket(self._namespace, sequence)
                self._owners[sequence] = principal
                self._sizes[sequence] = size
                self._tickets[sequence] = ticket
            except BaseException:
                self._owners.pop(sequence, None)
                self._sizes.pop(sequence, None)
                self._tickets.pop(sequence, None)
                if not self._queue.cancel(sequence):
                    raise RuntimeError('unable to roll back pending ticket')
                raise
            return ticket

    def _verify(self, credential):
        principal = self._verifier(credential)
        if type(principal) is not str or principal not in self._principals:
            raise UnknownPrincipal('verified principal is not configured')
        return principal

    def cancel(self, credential, ticket):
        principal = self._verify(credential)
        if (type(ticket) is not WaitingTicket or type(ticket.namespace) is not str or
                type(ticket.sequence) is not int or ticket.sequence <= 0):
            raise ValueError('invalid waiting ticket')
        if ticket.namespace != self._namespace:
            return False
        with self._lock:
            owner = self._owners.get(ticket.sequence)
            if owner is None:
                return False
            if owner != principal:
                raise PermissionError('ticket belongs to another principal')
            if not self._queue.cancel(ticket.sequence):
                raise RuntimeError('pending ticket missing from admission queue')
            del self._owners[ticket.sequence]
            del self._sizes[ticket.sequence]
            del self._tickets[ticket.sequence]
            return True

    def transfer_next(self):
        with self._lock:
            if self._closed:
                return None
        state_lock = Lock()
        handoff_thread = get_ident()
        active = True
        called = False
        claimed = None

        def claim(eligible):
            nonlocal called, claimed
            if get_ident() != handoff_thread:
                raise HandoffProtocolError('claim must run on the handoff thread')
            with state_lock:
                if not active or called:
                    raise HandoffProtocolError('claim must be synchronous and called at most once')
                called = True
                with self._lock:
                    if self._closed:
                        return None
                    try:
                        request = self._queue.claim_next(eligible)
                    except AdmissionEmpty:
                        return None
                    claimed = self._tickets[request.ticket_id]
                    del self._sizes[request.ticket_id]
                    del self._owners[request.ticket_id]
                    del self._tickets[request.ticket_id]
                    return request

        try:
            receipt = self._target.accept_next(claim)
        except BaseException as exc:
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
        with self._lock:
            tickets = tuple(self._tickets.values())
            removed = self._queue.drain()
            self._closed = True
            self._owners.clear()
            self._sizes.clear()
            self._tickets.clear()
        del removed
        return tickets

    def __len__(self):
        with self._lock:
            return len(self._owners)

    @property
    def waiting_bytes(self):
        with self._lock:
            return sum(self._sizes.values())
