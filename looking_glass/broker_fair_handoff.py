"""Owner-configured, experimental fair admission into one resident broker.

The budget charges declared commitments; it is not an OS memory limit. The
decoder and worker are trusted and must obey the bounds documented by the
service owner. The broker owns this object and disables its ordinary submit
path before making it available.
"""

from dataclasses import replace
from time import perf_counter_ns

import torch

from .broker import BrokerUnavailable, RequestHandle, _Job, _validate_forward_snapshot
from .fair_adapter import HANDOFF_PROTOCOL
from .fair_admission import AdmissionRequest
from .host_budget import HostBudget, HostBudgetExceeded


class FairPreparationError(RuntimeError):
    """A claimed waiter could not be decoded or snapshotted."""


class FatalFairPreparation(BaseException):
    """A fatal error occurred after a waiter was claimed."""


class _InputCustody:
    """A host-ledger claim owned by a job until its input is disposed."""

    def __init__(self, request, claim):
        self._request = request
        self._claim = claim

    def release(self):
        self._claim.release()
        self._request.close()


class _ReceiptRecord:
    """A result commitment and recoverable handle, held until owner ACK."""

    def __init__(self, handle, custody):
        self.handle = handle
        self.custody = custody
        self.published = False
        self.sequence = None
        self.acknowledged = False
        self.retired = False


class _HandoffState:
    """Identity-tracked preparation, guarded by the broker condition."""

    def __init__(self):
        self.active = False


class BrokerFairHandoff:
    """A single-adapter synchronous target, constructed by its broker only."""

    handoff_protocol = HANDOFF_PROTOCOL

    def __init__(self, broker, teams, decoder, max_item_bytes, max_decoded_bytes,
                 preparation_bytes, max_preparation_bytes, max_result_bytes,
                 max_unacknowledged):
        if type(teams) not in (tuple, list) or not teams:
            raise ValueError('teams must be a nonempty tuple or list')
        if any(type(team) is not str or not team or len(team) > 128 for team in teams):
            raise ValueError('team IDs must be bounded nonempty strings')
        self._teams = frozenset(teams)
        if len(self._teams) != len(teams):
            raise ValueError('team IDs must be distinct')
        if not callable(decoder):
            raise TypeError('decoder must be trusted owner-supplied code')
        for value in (max_item_bytes, max_decoded_bytes, preparation_bytes,
                      max_preparation_bytes, max_result_bytes, max_unacknowledged):
            if type(value) is not int or value <= 0:
                raise ValueError('host-byte limits must be positive integers')
        input_limit = broker._worker.max_input_bytes
        output_limit = broker._worker.max_output_bytes
        if (type(input_limit) is not int or input_limit <= 0 or
                type(output_limit) is not int or output_limit <= 0):
            raise ValueError('worker must expose positive input and output tensor limits')
        # Cover three overlapping input snapshots and both model logits and
        # the worker's CPU result copy, in addition to the retained result
        # commitment. General model intermediates need their own budget.
        minimum = max_item_bytes + max_decoded_bytes + 3 * input_limit + 2 * output_limit
        if preparation_bytes < minimum or max_preparation_bytes < preparation_bytes:
            raise ValueError('host commitment is below the configured input bounds')
        if max_result_bytes < output_limit:
            raise ValueError('result commitment is below the worker output limit')
        self._broker = broker
        self._decoder = decoder
        self._max_item_bytes = max_item_bytes
        self._max_decoded_bytes = max_decoded_bytes
        self._max_result_bytes = max_result_bytes
        self._input_limit = input_limit
        self._output_limit = output_limit
        self._commitment = preparation_bytes
        self._budget = HostBudget(max_bytes=max_preparation_bytes,
                                  max_items=broker._max_pending + 1,
                                  max_requests=broker._max_pending + 1,
                                  max_operations=1)
        self._receipt_budget = HostBudget(max_bytes=max_unacknowledged * max_result_bytes,
                                          max_items=max_unacknowledged,
                                          max_requests=max_unacknowledged,
                                          max_operations=1)
        self._receipts = {}
        # Broker-owned roots for claimed inputs until cleanup or publication.
        self._inflight = {}
        self._bound = False

    @property
    def teams(self):
        return self._teams

    @property
    def max_item_bytes(self):
        return self._max_item_bytes

    @property
    def max_decoded_bytes(self):
        return self._max_decoded_bytes

    @property
    def max_result_bytes(self):
        return self._max_result_bytes

    def bind_waiting_queue(self, teams, max_item_bytes):
        broker = self._broker
        with broker._condition:
            if broker._fair_handoff is not self:
                raise BrokerUnavailable('target is not owned by this broker')
            if broker._closed or broker._unavailable or broker._worker.poisoned:
                raise BrokerUnavailable('broker is closed or worker is unavailable')
            if teams != self.teams or max_item_bytes > self.max_item_bytes:
                raise ValueError('adapter teams or item limit do not match the target')
            if self._bound:
                raise BrokerUnavailable('target already has a waiting adapter')
            self._bound = True
            broker._condition.notify_all()

    @property
    def committed_bytes(self):
        return self._budget.snapshot().used_bytes

    @property
    def receipt_committed_bytes(self):
        """Declared result bytes held for unacknowledged receipt custody."""
        return self._receipt_budget.snapshot().used_bytes

    def wait_for_capacity(self, timeout=None):
        """Wait for declared byte and global slot capacity, not a team turn.

        Use a finite timeout when an external actor can poison the worker; that
        state change need not notify the broker condition.
        """
        broker = self._broker
        with broker._condition:
            broker._condition.wait_for(
                lambda: broker._closed or broker._unavailable or broker._worker.poisoned or
                (self._bound and broker._pending_slots < broker._max_pending and
                 self.committed_bytes + self._commitment <= self._budget.max_bytes and
                 self._receipt_budget.snapshot().used_items < self._receipt_budget.max_items), timeout)
            return (self._bound and not broker._closed and not broker._unavailable and not broker._worker.poisoned and
                    broker._pending_slots < broker._max_pending and
                    self.committed_bytes + self._commitment <= self._budget.max_bytes and
                    self._receipt_budget.snapshot().used_items < self._receipt_budget.max_items)

    def _reserve_locked(self, budget, job_id, stage, amount):
        request = budget.open_request(f'lg-{self._broker.broker_id}-{job_id}')
        try:
            custody = _InputCustody(request, None)
            custody._claim = request.reserve('broker', stage, amount)
            return custody
        except BaseException:
            try:
                request.close()
            except BaseException as cleanup_exc:
                self._broker._mark_cleanup_failed_locked()
                if not isinstance(cleanup_exc, Exception):
                    raise
            raise

    def _release(self, custody):
        """Call only after all input references covered by the claim are gone."""
        try:
            custody.release()
        except BaseException as exc:
            with self._broker._condition:
                self._broker._mark_cleanup_failed_locked()
            if not isinstance(exc, Exception):
                raise
            raise BrokerUnavailable('input custody cleanup failed') from None
        with self._broker._condition:
            self._broker._condition.notify_all()

    def recover_receipt(self, sequence):
        """Trusted owner lookup; the adapter checks ticket namespace first."""
        with self._broker._condition:
            record = self._receipts.get(sequence)
            if record is None:
                return None
            if record.acknowledged:
                return None
            if not record.published:
                raise BrokerUnavailable('receipt publication outcome is unknown')
            return record.handle

    def acknowledge_receipt(self, sequence, handle):
        """Transfer result custody to an already-budgeted trusted owner."""
        broker = self._broker
        mutating = False
        try:
            with broker._condition:
                record = self._receipts.get(sequence)
                if record is None:
                    return False
                if record.handle is not handle:
                    raise ValueError('receipt handle does not match')
                if not record.published or broker._fatal_cleanup_failure:
                    raise BrokerUnavailable('receipt outcome is unknown')
                if record.acknowledged:
                    return False
                mutating = True
                record.acknowledged = True
                if not record.retired:
                    return True
                del self._receipts[sequence]
                record.handle = None
            self._release(record.custody)
            return True
        except BaseException:
            if mutating:
                with broker._condition:
                    broker._mark_cleanup_failed_locked()
            raise

    def retire_receipt(self, job):
        """Complete result retirement after broker-local aliases are dropped."""
        broker = self._broker
        held_handle = None
        custody = None
        try:
            with broker._condition:
                record = job.receipt_record
                if record is None:
                    return
                if broker._fatal_cleanup_failure:
                    return
                record.retired = True
                job.receipt_record = None
                if record.acknowledged:
                    del self._receipts[record.sequence]
                    held_handle = record.handle
                    record.handle = None
                    custody = record.custody
            del held_handle
            if custody is not None:
                self._release(custody)
        except BaseException:
            with broker._condition:
                broker._mark_cleanup_failed_locked()
            raise

    def validate_result_storage(self, result):
        """Reject result storage beyond the declared per-receipt commitment."""
        if result.logits.untyped_storage().nbytes() > self._max_result_bytes:
            raise ValueError('result storage exceeds the receipt commitment')

    def accept_next(self, claim):
        state = _HandoffState()
        try:
            return self._accept_next(claim, state)
        except BaseException:
            if state.active:
                with self._broker._condition:
                    # A stage transition may have been interrupted after its
                    # mutation. Retain all uncertain slots and ledger claims.
                    self._broker._mark_cleanup_failed_locked()
                    self._broker._fair_preparations.discard(state)
                    self._broker._condition.notify_all()
            raise

    def _accept_next(self, claim, state):
        broker = self._broker
        request = None
        with broker._condition:
            if broker._fair_handoff is not self or not self._bound:
                raise BrokerUnavailable('fair handoff target is not bound')
            if broker._closed or broker._unavailable or broker._worker.poisoned:
                raise BrokerUnavailable('broker is closed or worker is unavailable')
            if (broker._worker.max_input_bytes != self._input_limit or
                    broker._worker.max_output_bytes != self._output_limit):
                broker._mark_cleanup_failed_locked()
                raise BrokerUnavailable('worker input or output limit changed')
            if broker._pending_slots >= broker._max_pending:
                return None
            eligible = frozenset(team for team in self.teams
                                 if broker._team_slots.get(team, 0) < broker._max_pending_per_team)
            if not eligible:
                return None
            next_team_counts = {team: broker._team_slots.get(team, 0) + 1 for team in eligible}
            next_id = broker._next_job_id + 1
            # Guard starts before the first ledger mutation. An interruption
            # during any later stage must fail closed and retain uncertainty.
            state.active = True
            broker._fair_preparations.add(state)
            try:
                custody = self._reserve_locked(self._budget, next_id, 'fair-input', self._commitment)
            except HostBudgetExceeded:
                if broker._fatal_cleanup_failure:
                    raise BrokerUnavailable('host admission cleanup failed') from None
                broker._fair_preparations.discard(state)
                state.active = False
                return None
            try:
                result_custody = self._reserve_locked(
                    self._receipt_budget, next_id, 'fair-result', self._max_result_bytes)
            except BaseException as exc:
                try:
                    custody.release()
                except BaseException as cleanup_exc:
                    broker._mark_cleanup_failed_locked()
                    if not isinstance(cleanup_exc, Exception):
                        raise
                if broker._fatal_cleanup_failure:
                    if not isinstance(exc, Exception):
                        raise
                    raise BrokerUnavailable('host admission cleanup failed') from None
                if isinstance(exc, HostBudgetExceeded):
                    broker._fair_preparations.discard(state)
                    state.active = False
                    return None
                raise
            # Allocate everything possible before allowing the adapter to
            # consume a waiter. A callback failure leaves accounting unknown;
            # retain the reservation and stop the broker rather than guess.
            try:
                handle = RequestHandle(broker, next_id, '', perf_counter_ns())
                job = _Job(handle, None, custody)
                receipt_record = _ReceiptRecord(handle, result_custody)
                self._inflight[next_id] = job
                previous_pending = broker._pending_slots
                next_pending = previous_pending + 1
            except BaseException:
                try:
                    custody.release()
                    result_custody.release()
                except BaseException as cleanup_exc:
                    broker._mark_cleanup_failed_locked()
                    if not isinstance(cleanup_exc, Exception):
                        raise
                raise
            broker._pending_slots = next_pending
            # Even an empty claim may retain its provisional host request until
            # after releasing the condition; never reuse that request ID.
            broker._next_job_id = next_id
            try:
                request = claim(eligible)
                if request is not None:
                    if (type(request) is not AdmissionRequest or request.team_id not in eligible or
                            type(request.payload) is not bytes or
                            len(request.payload) > self.max_item_bytes):
                        raise ValueError('claimed request violates handoff limits')
                    team_id = request.team_id
                    ticket_id = request.ticket_id
                    receipt_record.sequence = ticket_id
                    job.staged_raw = request
                    handle._team_id = team_id
                    broker._team_slots[team_id] = next_team_counts[team_id]
            except BaseException:
                broker._mark_cleanup_failed_locked()
                raise
            if request is None:
                broker._pending_slots = previous_pending
                del self._inflight[next_id]
                broker._fair_preparations.discard(state)
                broker._condition.notify_all()
        if request is None:
            self._release(custody)
            self._release(result_custody)
            state.active = False
            return None

        decoded = None
        snapshot = None
        failure = None
        poisoned = False
        try:
            decoded = self._decoder(request.payload)
            job.staged_decoded = decoded
            try:
                _validate_forward_snapshot(decoded)
            except BaseException:
                # A decoder must reject malformed wire data by raising; an
                # unsupported returned object may own unrecognized resources.
                # It is already rooted in the preallocated inflight job.
                with broker._condition:
                    broker._mark_cleanup_failed_locked()
                raise
            request_id = decoded.request_id
            try:
                snapshot = broker._worker.snapshot_request(decoded)
            except BaseException as exc:
                # Every preparation has its own pre-rooted job. A previous
                # preparation may already have latched quarantine, but this
                # exception can independently own partially prepared inputs.
                job.retained_error = exc
                if not isinstance(exc, Exception) or isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                    with broker._condition:
                        broker._mark_cleanup_failed_locked()
                raise
            job.request = snapshot
            try:
                _validate_forward_snapshot(snapshot)
                if snapshot.request_id != request_id:
                    raise ValueError('worker snapshot changed the caller request_id')
                worker_request_id = f'lg-{broker._identity_prefix}-{handle.job_id}'
                snapshot = replace(snapshot, request_id=worker_request_id)
            except BaseException:
                # job.request roots the exact rejected worker snapshot.
                with broker._condition:
                    broker._mark_cleanup_failed_locked()
                raise
            job.request = snapshot
        except BaseException as exc:
            job.retained_error = exc
            failure = (not isinstance(exc, Exception), type(exc).__name__)
            try:
                poisoned = bool(broker._worker.poisoned)
            except BaseException as status_exc:
                poisoned = True
                job.retained_status_error = status_exc
                if not isinstance(status_exc, Exception):
                    with broker._condition:
                        broker._mark_cleanup_failed_locked()
                    raise
            if poisoned:
                # A poisoned worker can retain unknown preparation inputs even
                # when it raises an ordinary Exception such as WorkerOutOfMemory.
                with broker._condition:
                    broker._mark_cleanup_failed_locked()
        request = None
        decoded = None

        if failure is not None:
            snapshot = None
            removed = []
            with broker._condition:
                broker._fair_preparations.discard(state)
                try:
                    if not broker._fatal_cleanup_failure and not broker.quarantined:
                        broker._release_slot_locked(team_id)
                        if poisoned:
                            broker._unavailable = 'worker failed while preparing fair request'
                            removed = broker._fail_pending_locked()
                except BaseException as cleanup_exc:
                    broker._mark_cleanup_failed_locked()
                    if not isinstance(cleanup_exc, Exception):
                        raise
                broker._condition.notify_all()
            if not broker._fatal_cleanup_failure:
                # A poisoned worker may retain references in quarantine.
                if not poisoned:
                    job.retained_error = None
                    job.retained_status_error = None
                    job.staged_raw = None
                    job.staged_decoded = None
                    job.request = None
                    with broker._condition:
                        del self._inflight[next_id]
                    self._release(custody)
                    self._release(result_custody)
                broker._retire_items(removed)
                state.active = False
            if failure[0]:
                raise FatalFairPreparation(failure[1])
            raise FairPreparationError(failure[1])

        job.staged_raw = None
        job.staged_decoded = None
        publication_failure = None
        publication_started = False
        with broker._condition:
            try:
                if broker._closed or broker._unavailable or broker._worker.poisoned:
                    raise BrokerUnavailable('broker closed while preparing fair request')
                handle._request_id = request_id
                handle._worker_request_id = worker_request_id
                job.request = snapshot
                job.receipt_record = receipt_record
                handle._job = job
                handle._ready_ns = perf_counter_ns()
                publication_started = True
                self._receipts[ticket_id] = receipt_record
                try:
                    broker._queue.put(team_id, handle.job_id, job)
                    receipt_record.published = True
                    # The owner can execute and retire the job as soon as this
                    # condition is released. It must be the only holder of the
                    # input snapshot by then, even if this thread is paused.
                    snapshot = None
                    del self._inflight[next_id]
                    broker._fair_preparations.discard(state)
                    # The receipt ledger, not the dispatch frame, now owns the
                    # handle. Retained interruption tracebacks must not pin a
                    # completed result after its custody has been transferred.
                    handle = None
                    receipt_record = None
                    job = None
                    state.active = False
                except BaseException as exc:
                    broker._mark_cleanup_failed_locked()
                    if not isinstance(exc, Exception):
                        raise
                    raise BrokerUnavailable('queue publication failed; accounting is unknown') from None
            except BaseException as exc:
                publication_failure = (not isinstance(exc, Exception), type(exc).__name__)
                if publication_started and not broker._fatal_cleanup_failure:
                    broker._mark_cleanup_failed_locked()
                try:
                    poisoned = bool(broker._worker.poisoned)
                except BaseException as status_exc:
                    poisoned = True
                    if not isinstance(status_exc, Exception):
                        broker._retain_unknown(status_exc)
                        raise
                if not broker._fatal_cleanup_failure:
                    try:
                        broker._release_slot_locked(team_id)
                    except BaseException as cleanup_exc:
                        broker._mark_cleanup_failed_locked()
                        if not isinstance(cleanup_exc, Exception):
                            raise
            finally:
                broker._condition.notify_all()
        if publication_failure is not None:
            if not broker._fatal_cleanup_failure and not poisoned:
                snapshot = None
                job.request = None
                handle._job = None
                with broker._condition:
                    del self._inflight[next_id]
                self._release(custody)
                self._release(result_custody)
                with broker._condition:
                    broker._fair_preparations.discard(state)
                state.active = False
            if publication_failure[0]:
                raise FatalFairPreparation(publication_failure[1])
            raise FairPreparationError(publication_failure[1])
        # Return only an opaque identifier. The trusted owner retrieves the
        # handle from the receipt ledger after this handoff frame has unwound.
        return ticket_id
