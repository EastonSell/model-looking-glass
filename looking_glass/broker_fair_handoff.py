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


def _failure_kind(exc):
    if isinstance(exc, KeyboardInterrupt):
        return 1
    if isinstance(exc, SystemExit):
        return 2
    if isinstance(exc, Exception):
        return 0
    return 3


def _failure_identity(exc):
    """Conservatively preserve fatal interruptions in a bounded error chain."""
    original = exc
    for _ in range(64):
        kind = _failure_kind(exc)
        if kind:
            return kind, type(exc).__name__
        # Python records an exception being handled as the new exception's
        # context even when a display-only ``from None`` suppresses it.
        context = BaseException.__getattribute__(exc, '__context__')
        cause = BaseException.__getattribute__(exc, '__cause__')
        next_error = context if context is not None else cause
        if next_error is None or next_error is exc:
            break
        exc = next_error
    return 0, type(original).__name__


def _raise_sanitized_failure(kind, name):
    """Raise a fresh error without exposing a privately retained traceback."""
    try:
        if kind == 1:
            raise KeyboardInterrupt(name) from None
        if kind == 2:
            raise SystemExit(name) from None
        if kind == 3:
            raise FatalFairPreparation(name) from None
        raise FairPreparationError(name) from None
    except BaseException as public_error:
        # An exception active in the caller would otherwise become the new
        # error's implicit context even when we raise outside our handler.
        public_error.__context__ = None
        raise


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
        self.retirement = handle._retirement
        handle._retirement_supported = True
        self.published = False
        self.sequence = None
        self.acknowledged = False
        self.retired = False


class _HandoffState:
    """Identity-tracked preparation, guarded by the broker condition."""

    def __init__(self):
        self.active = False
        self.job = None
        self.retained_error = None
        self.retained_quarantine_error = None
        self.retained_notification_error = None
        self.retained_classification_error = None
        self.retained_root_error = None
        self.retained_root_retry_error = None
        self.fatal_kind = 0
        self.fatal_name = None

    def remember_fatal(self, exc):
        kind = _failure_kind(exc)
        if kind and not self.fatal_kind:
            self.fatal_kind = kind
            self.fatal_name = type(exc).__name__


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
        input_limit, output_limit = broker._worker_limits_locked()
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
        # Retain each concurrent failed handoff independently. The slots cover
        # pending preparations, receipt positions, and one registration fault.
        self._failure_states = [None] * (broker._max_pending + max_unacknowledged + 1)
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
            if broker._closed or broker._unavailable or broker.quarantined or broker._worker_poisoned_locked():
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
                lambda: broker._closed or broker._unavailable or broker.quarantined or broker._worker_poisoned_locked() or
                (self._bound and broker._pending_slots < broker._max_pending and
                 any(job is None for job in broker._job_roots) and
                 self.committed_bytes + self._commitment <= self._budget.max_bytes and
                 self._receipt_budget.snapshot().used_items < self._receipt_budget.max_items), timeout)
            return (self._bound and not broker._closed and not broker._unavailable and not broker.quarantined and
                    not broker._worker_poisoned_locked() and
                    broker._pending_slots < broker._max_pending and
                    any(job is None for job in broker._job_roots) and
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
                if not record.published or broker._fatal_cleanup_failure or broker.quarantined:
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
                if broker._fatal_cleanup_failure or broker.quarantined:
                    return
                retirement = record.retirement
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
            # This signal does not own the handle. Publish only after every
            # execution-thread/job alias has been dropped and cleanup finished.
            # An unacknowledged receipt can still retain the handle itself.
            with broker._condition:
                if not broker._fatal_cleanup_failure and not broker.quarantined:
                    retirement.complete = True
                    broker._condition.notify_all()
        except BaseException:
            with broker._condition:
                broker._mark_cleanup_failed_locked()
            raise

    def validate_result_storage(self, result):
        """Reject result storage beyond the declared per-receipt commitment."""
        if result.logits.untyped_storage().nbytes() > self._max_result_bytes:
            raise ValueError('result storage exceeds the receipt commitment')

    def _retire_unpublished(self, job, job_id, result_custody):
        """Dispose an unpublished job only if cleanup wins against quarantine."""
        broker = self._broker
        with broker._condition:
            if broker._fatal_cleanup_failure or broker.quarantined:
                return False
            # Construct the guard before detaching any staged references. The
            # job remains in _inflight until both ledger releases complete.
            staged = (job.staged_raw, job.staged_decoded, job.retained_error,
                      job.retained_status_error)
            job.retiring_staged = staged
            job.staged_raw = None
            job.staged_decoded = None
            job.retained_error = None
            job.retained_status_error = None
            # This locked transition commits cleanup before a later
            # quarantine; the guard prevents finalizers under the condition.
            job.retiring_staged = None
        staged = None
        if not broker._retire_job(job):
            return False
        self._release(result_custody)
        job.pending_result_custody = None
        with broker._condition:
            del self._inflight[job_id]
        return True

    def _drop_success_staged(self, job):
        """Commit disposal of copied request inputs against quarantine."""
        broker = self._broker
        with broker._condition:
            if broker._fatal_cleanup_failure or broker.quarantined:
                return False
            staged = (job.staged_raw, job.staged_decoded)
            job.retiring_staged = staged
            job.staged_raw = None
            job.staged_decoded = None
            job.retiring_staged = None
        staged = None
        return True

    def _root_failure_state_locked(self, state):
        for index, existing in enumerate(self._failure_states):
            if existing is state:
                return
            if existing is None:
                self._failure_states[index] = state
                return
        # A trusted owner must bound concurrent transfer calls. Retain an
        # unexpected excess in the broker's separate bounded quarantine pool.
        self._broker._store_quarantine_value_locked(state)

    def accept_next(self, claim):
        state = _HandoffState()
        try:
            return self._accept_next(claim, state)
        except BaseException as exc:
            if not state.active:
                raise
            with self._broker._condition:
                state.retained_error = exc
                if self._broker._fair_orphan_state is None:
                    # Also cover a failure between removing a completed
                    # preparation from the set and returning to this guard.
                    self._broker._fair_orphan_state = state
                if state.job is not None:
                    # If a cleanup transition was interrupted, its traceback
                    # can still hold detached local references.
                    state.job.retained_cleanup_error = exc
                # Latch before inspecting the exception: classification can
                # itself fail while the original private traceback owns data.
                try:
                    self._broker._quarantine_latched = True
                finally:
                    self._broker._quarantine_latched = True
                    self._broker._fatal_cleanup_failure = True
                    self._broker._unavailable = 'broker cleanup failed; request status is unknown'
                try:
                    self._root_failure_state_locked(state)
                except BaseException as root_error:
                    state.retained_root_error = root_error
                    try:
                        self._root_failure_state_locked(state)
                    except BaseException as retry_error:
                        state.retained_root_retry_error = retry_error
                        # The shared quarantine pool is a separate bounded
                        # fallback if the preallocated target table fails.
                        self._broker._store_quarantine_value_locked(state)
                try:
                    kind, name = _failure_identity(exc)
                    if state.fatal_kind:
                        kind, name = state.fatal_kind, state.fatal_name
                except BaseException as classification_error:
                    state.retained_classification_error = classification_error
                    kind, name = _failure_kind(exc), type(exc).__name__
                    if state.fatal_kind:
                        kind, name = state.fatal_kind, state.fatal_name
                    elif kind == 0:
                        kind = _failure_kind(classification_error)
                        name = type(classification_error).__name__
                for root_error in (state.retained_root_error, state.retained_root_retry_error):
                    if root_error is not None and kind == 0 and _failure_kind(root_error) != 0:
                        kind, name = _failure_kind(root_error), type(root_error).__name__
                root_error = None
                # A stage transition may have been interrupted after its
                # mutation. Retain all uncertain slots and ledger claims.
                mark_failed = False
                try:
                    self._broker._mark_cleanup_failed_locked()
                except BaseException as mark_error:
                    # Even an interrupted or repeatedly failing notification
                    # must not expose the private exception chain or leave the
                    # broker available. The pre-existing state is broker-rooted.
                    state.retained_quarantine_error = mark_error
                    self._broker._quarantine_latched = True
                    self._broker._fatal_cleanup_failure = True
                    self._broker._unavailable = 'broker cleanup failed; request status is unknown'
                    mark_failed = True
                    if kind == 0 and _failure_kind(mark_error) != 0:
                        name, kind = type(mark_error).__name__, _failure_kind(mark_error)
                try:
                    self._broker._condition.notify_all()
                except BaseException as notification_error:
                    state.retained_notification_error = notification_error
                    self._broker._quarantine_latched = True
                    self._broker._fatal_cleanup_failure = True
                    self._broker._unavailable = 'broker cleanup failed; request status is unknown'
                    mark_failed = True
                    if kind == 0 and _failure_kind(notification_error) != 0:
                        name, kind = type(notification_error).__name__, _failure_kind(notification_error)
                if not mark_failed:
                    self._broker._fair_preparations.discard(state)
            # Do not expose the private retained exception: a caller can
            # clear its traceback frames, releasing unknown worker-owned data.
        # Raise outside the handler so __context__ does not point back to the
        # private exception. Preserve the built-in fatal interruption kinds.
        _raise_sanitized_failure(kind, name)

    def _accept_next(self, claim, state):
        broker = self._broker
        request = None
        with broker._condition:
            if broker._fair_handoff is not self or not self._bound:
                raise BrokerUnavailable('fair handoff target is not bound')
            if broker._closed or broker._unavailable or broker.quarantined or broker._worker_poisoned_locked():
                raise BrokerUnavailable('broker is closed or worker is unavailable')
            if broker._worker_limits_locked() != (self._input_limit, self._output_limit):
                broker._mark_cleanup_failed_locked()
                raise BrokerUnavailable('worker input or output limit changed')
            if broker._pending_slots >= broker._max_pending:
                return None
            if all(job is not None for job in broker._job_roots):
                return None
            eligible = frozenset(team for team in self.teams
                                 if broker._team_slots.get(team, 0) < broker._max_pending_per_team)
            if not eligible:
                return None
            next_team_counts = {team: broker._team_slots.get(team, 0) + 1 for team in eligible}
            next_id = broker._next_job_id + 1
            # Guard starts before the first ledger mutation. An interruption
            # during any later stage must fail closed and retain uncertainty.
            try:
                state.active = True
                broker._fair_preparations.add(state)
            except BaseException as registration_error:
                # Reject new admissions before releasing this condition. The
                # preallocated orphan root covers a failure before set.add.
                state.active = True
                state.retained_error = registration_error
                broker._fair_orphan_state = state
                broker._quarantine_latched = True
                broker._fatal_cleanup_failure = True
                broker._unavailable = 'broker cleanup failed; request status is unknown'
                raise
            try:
                custody = self._reserve_locked(self._budget, next_id, 'fair-input', self._commitment)
            except HostBudgetExceeded:
                if broker._fatal_cleanup_failure or broker.quarantined:
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
                if broker._fatal_cleanup_failure or broker.quarantined:
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
                job = _Job(handle, input_custody=custody)
                state.job = job
                job.pending_result_custody = result_custody
                receipt_record = _ReceiptRecord(handle, result_custody)
                self._inflight[next_id] = job
                # The target root exists before the shared broker root is
                # installed; both precede the claim and queue publication.
                broker._root_job_locked(job)
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
                broker._fair_preparations.discard(state)
                broker._condition.notify_all()
        if request is None:
            if not self._retire_unpublished(job, next_id, result_custody):
                raise BrokerUnavailable('empty claim cleanup is unknown')
            state.job = None
            state.active = False
            return None

        decoded = None
        snapshot = None
        failure = None
        poisoned = False
        try:
            try:
                decoded = self._decoder(request.payload)
            except BaseException as exc:
                state.remember_fatal(exc)
                # OOM or interruption can leave partially constructed inputs
                # in the decoder's traceback. Keep that private traceback on
                # the already-rooted job until owner-process replacement.
                job.retained_error = exc
                if not isinstance(exc, Exception) or isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                    job.quarantine_required = True
                    with broker._condition:
                        broker._mark_cleanup_failed_locked()
                raise
            job.staged_decoded = decoded
            try:
                _validate_forward_snapshot(decoded)
            except BaseException:
                # A decoder must reject malformed wire data by raising; an
                # unsupported returned object may own unrecognized resources.
                # It is already rooted in the preallocated inflight job.
                job.quarantine_required = True
                with broker._condition:
                    broker._mark_cleanup_failed_locked()
                raise
            request_id = decoded.request_id
            try:
                snapshot = broker._worker.snapshot_request(decoded)
            except BaseException as exc:
                state.remember_fatal(exc)
                # Every preparation has its own pre-rooted job. A previous
                # preparation may already have latched quarantine, but this
                # exception can independently own partially prepared inputs.
                job.retained_error = exc
                if not isinstance(exc, Exception) or isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                    job.quarantine_required = True
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
                job.quarantine_required = True
                with broker._condition:
                    broker._mark_cleanup_failed_locked()
                raise
            job.request = snapshot
        except BaseException as exc:
            state.remember_fatal(exc)
            previous_error = job.retained_error
            job.retained_error = exc
            # A failed quarantine call can replace an earlier interruption.
            # Preserve that fatal category while retaining both exceptions via
            # their private context chain on the rooted job.
            reported = (previous_error if previous_error is not None and
                        not isinstance(previous_error, Exception) and isinstance(exc, Exception) else exc)
            failure = (_failure_kind(reported), type(reported).__name__)
            reported = None
            previous_error = None
            if broker.quarantined:
                poisoned = True
            elif (job.quarantine_required or not isinstance(exc, Exception) or
                    isinstance(exc, (MemoryError, torch.OutOfMemoryError))):
                with broker._condition:
                    broker._mark_cleanup_failed_locked()
                poisoned = True
            else:
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
            if not broker._fatal_cleanup_failure and not broker.quarantined:
                # A poisoned worker may retain references in quarantine.
                if not poisoned:
                    if not self._retire_unpublished(job, next_id, result_custody):
                        raise BrokerUnavailable('failed preparation cleanup is unknown')
                broker._retire_items(removed)
                state.job = None
                state.active = False
            _raise_sanitized_failure(*failure)

        if not self._drop_success_staged(job):
            raise BrokerUnavailable('preparation cleanup is unknown')
        publication_failure = None
        publication_started = False
        with broker._condition:
            try:
                if broker._closed or broker._unavailable or broker.quarantined or broker._worker_poisoned_locked():
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
                    state.job = None
                    state.active = False
                except BaseException as exc:
                    state.remember_fatal(exc)
                    broker._mark_cleanup_failed_locked()
                    if not isinstance(exc, Exception):
                        raise
                    raise BrokerUnavailable('queue publication failed; accounting is unknown') from None
            except BaseException as exc:
                state.remember_fatal(exc)
                publication_failure = (_failure_kind(exc), type(exc).__name__)
                if publication_started and not broker._fatal_cleanup_failure:
                    broker._mark_cleanup_failed_locked()
                if broker.quarantined:
                    poisoned = True
                else:
                    try:
                        poisoned = bool(broker._worker.poisoned)
                    except BaseException as status_exc:
                        poisoned = True
                        if job is None:
                            # Publication may already have succeeded before an
                            # interruption cleared this frame's job reference.
                            broker._retain_unknown(status_exc)
                        else:
                            job.retained_status_error = status_exc
                            broker._mark_cleanup_failed_locked()
                        if not isinstance(status_exc, Exception):
                            raise
                if not broker._fatal_cleanup_failure and not broker.quarantined:
                    try:
                        broker._release_slot_locked(team_id)
                    except BaseException as cleanup_exc:
                        broker._mark_cleanup_failed_locked()
                        if not isinstance(cleanup_exc, Exception):
                            raise
            finally:
                broker._condition.notify_all()
        if publication_failure is not None:
            if not broker._fatal_cleanup_failure and not broker.quarantined and not poisoned:
                snapshot = None
                handle._job = None
                if not self._retire_unpublished(job, next_id, result_custody):
                    raise BrokerUnavailable('publication cleanup is unknown')
                with broker._condition:
                    broker._fair_preparations.discard(state)
                state.job = None
                state.active = False
            _raise_sanitized_failure(*publication_failure)
        # Return only an opaque identifier. The trusted owner retrieves the
        # handle from the receipt ledger after this handoff frame has unwound.
        return ticket_id
