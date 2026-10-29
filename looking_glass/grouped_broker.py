"""Experimental trusted grouped coordinator; not yet enabled by the service.

This draft needs an independently reviewed GroupedWorker, live-runtime
collector, whole-batch profiles and supervised process custody before use.
"""

from dataclasses import dataclass
import math
import re
import threading
from time import perf_counter
from uuid import uuid4

from .certified_batch_policy import CertifiedBatchPolicy
from .fair_queue import FairTeamQueue, QueueEmpty
from .grouped_broker_state import CohortMember, CohortState, _Open
from .grouped_observation import GroupedObservation
from .grouped_worker import GroupedWorker
from .host_budget import HostBudget, HostBusy
from .microbatch import BatchProfile, MicrobatchRequest, _DTYPE_BYTES
from .resident_worker import ForwardRequest, WorkerOutOfMemory


class GroupedBrokerUnavailable(RuntimeError):
    """The grouped coordinator cannot safely accept or resolve work."""


@dataclass(frozen=True, slots=True)
class SnapshotReservation:
    compatibility_key: bytes
    profile_id: str
    nbytes: int
    items: int


@dataclass(frozen=True, slots=True)
class ValidationReservation:
    compatibility_key: bytes
    request_count: int
    profile_id: str
    nbytes: int
    items: int


@dataclass(frozen=True, slots=True)
class GroupedResult:
    job_id: str
    team_id: str
    request_id: str
    status: str
    cancel_requested: bool
    error_code: str | None
    logits: object = None


class _Job:
    def __init__(self, job_id, team_id, request_id, worker_request_id):
        self.job_id, self.team_id, self.request_id = job_id, team_id, request_id
        self.worker_request_id = worker_request_id
        self.status = 'preparing'
        self.cancel_requested = False
        self.ready_at = None
        self.request_guard = self.host_request = None
        self.claim_guard = self.claim = None
        self.snapshot = self.group = self.single = None
        self.disposed = False
        self.deliveries = set()
        self.cancel_active = False
        self.dispose_active = False
        self.selection_ns = self.terminal_ns = None


class _Group:
    def __init__(self, group_id, jobs, state):
        self.group_id, self.jobs, self.state = group_id, jobs, state
        self.request_guard = self.host_request = None
        self.claim_guard = self.claim = None
        self.validation_guard = self.validation_claim = None
        self.plan = self.launch = None
        self.execution_finished = False
        self.retiring = self.retired = False
        self.quarantined = False
        self.selection_ns = None


_owners = set()
_owners_lock = threading.Lock()


def _name(value, limit=128):
    return type(value) is str and bool(value) and len(value) <= limit


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{label} must be a positive integer')


def _reservation(record, expected_type, key):
    if type(record) is not expected_type or type(record.compatibility_key) is not bytes or record.compatibility_key != key:
        raise ValueError('reservation must match the exact policy key')
    if type(record.profile_id) is not str or re.fullmatch(r'[0-9a-f]{64}', record.profile_id) is None:
        raise ValueError('reservation requires a separately verified profile digest')
    _positive(record.nbytes, 'reservation bytes')
    _positive(record.items, 'reservation items')


class GroupedHandle:
    def __init__(self, broker, job):
        self._broker, self._job = broker, job

    def cancel(self):
        return self._broker._cancel(self._job)

    def delivery(self):
        """Reserve one bounded, independently released result handoff."""
        return self._broker._new_delivery(self._job)

    def active_deliveries(self):
        """Trusted recovery access after an ambiguous ticket handoff."""
        return self._broker._active_deliveries(self._job)

    def dispose(self):
        """Attest all caller-owned result aliases have been disposed."""
        return self._broker._dispose(self._job)


class GroupedDelivery:
    """One consumer's result custody; release only after its aliases are gone."""

    def __init__(self, broker, job):
        self._broker, self._job = broker, job
        self._wait_token = None
        self._consumer_thread = None
        self._delivered = False
        self._receipt = None
        self._confirmed = self._abandoned = False
        self._released = False

    def wait(self, timeout=None):
        return self._broker._wait(self, timeout)

    def confirm(self, result):
        """Confirm this exact result reached the consumer before disposal."""
        return self._broker._confirm_delivery(self, result)

    def abandon(self):
        """Attest an unreceived result and all error aliases are disposed."""
        return self._broker._abandon_delivery(self)

    def release(self):
        return self._broker._release_delivery(self)


class GroupedBroker:
    """One-owner grouped dispatch for trusted local clients (draft only)."""

    def __init__(self, worker, profile, policy, *, host_budget, identity, collect_runtime,
                 snapshot_reservation, validation_reservations, max_pending,
                 max_pending_per_team, max_coalesce_wait, max_lifetime_jobs,
                 max_active_deliveries_per_job=4, observation=None):
        if observation is not None and type(observation) is not GroupedObservation:
            raise ValueError('an exact trusted grouped observation is required')
        if (type(worker) is not GroupedWorker or type(profile) is not BatchProfile
                or type(policy) is not CertifiedBatchPolicy or type(host_budget) is not HostBudget):
            raise ValueError('exact grouped worker, profile, evidence policy and host ledger are required')
        if worker._host is not host_budget or worker._profiles.get(profile.profile_id) != profile:
            raise ValueError('worker must use this exact ledger and configured profile')
        if (type(identity) is not dict or set(identity) !=
                {'model_fingerprint', 'checkpoint_sha256', 'adapter_sha256', 'backend'}
                or not callable(collect_runtime)):
            raise ValueError('trusted live identity and runtime collector are required')
        for value, label in ((max_pending, 'max_pending'), (max_pending_per_team, 'max_pending_per_team'),
                             (max_lifetime_jobs, 'max_lifetime_jobs'),
                             (max_active_deliveries_per_job, 'max_active_deliveries_per_job')):
            _positive(value, label)
        if max_active_deliveries_per_job > 32:
            raise ValueError('at most 32 active deliveries per job are supported')
        if (type(max_coalesce_wait) not in (int, float) or not math.isfinite(max_coalesce_wait)
                or not 0 <= max_coalesce_wait <= 60):
            raise ValueError('coalescing interval must be finite and at most 60 seconds')
        _reservation(snapshot_reservation, SnapshotReservation, policy.compatibility_key)
        logical_input = math.prod(profile.input_shape) * sum(_DTYPE_BYTES[dtype] for _, dtype in profile.input_schema)
        if snapshot_reservation.nbytes < logical_input or snapshot_reservation.items < len(profile.input_schema):
            raise ValueError('snapshot reservation cannot cover its logical input')
        if type(validation_reservations) is not tuple or not validation_reservations:
            raise ValueError('profiled validation reservations are required')
        validation = {}
        for record in validation_reservations:
            _reservation(record, ValidationReservation, policy.compatibility_key)
            policy.require_count(record.request_count)
            logical_scratch = math.prod(profile.input_shape) * profile.output_width + 1
            if record.nbytes < logical_scratch or record.items < 2:
                raise ValueError('validation reservation cannot cover its minimum logical scratch')
            if record.request_count in validation:
                raise ValueError('duplicate validation reservation')
            validation[record.request_count] = record
        if set(validation) != set(policy.allowed_request_counts):
            raise ValueError('every admitted size requires a validation reservation')
        if any(size > max_pending for size in policy.allowed_request_counts):
            raise ValueError('queue cannot hold an admitted cohort')
        # Jobs occupy slots until all cohort-owned input/output tensors are
        # disposed.  Thus floor(max_pending / minimum cohort size) bounds
        # simultaneously retained groups, even if callers retain old results.
        # Reserve capacity for every staged snapshot and group before any job
        # is accepted; a separately designated, exclusively used ledger is a
        # trusted integration requirement, not an OS/RSS limit.
        max_groups = max_pending // min(policy.allowed_request_counts)
        group_bytes, group_items = worker.required_host_capacity(profile.profile_id)
        _positive(group_bytes, 'group reservation bytes')
        _positive(group_items, 'group reservation items')
        required = (
            max_pending * snapshot_reservation.nbytes
            + max_groups * (group_bytes + max(v.nbytes for v in validation.values())),
            max_pending * snapshot_reservation.items
            + max_groups * (group_items + max(v.items for v in validation.values())),
            max_pending + max_groups,
            max_groups,
        )
        if any(need > cap for need, cap in zip(required, (host_budget.max_bytes, host_budget.max_items,
                                                           host_budget.max_requests, host_budget.max_operations))):
            raise ValueError('host ledger cannot cover pending snapshots and retained cohorts')
        policy.require_live_binding(profile, runtime_manifest=collect_runtime(), **identity)
        self._worker, self._profile, self._policy, self._host = worker, profile, policy, host_budget
        self._identity, self._collect_runtime = dict(identity), collect_runtime
        self._snapshot_reservation, self._validation = snapshot_reservation, validation
        self._queue = FairTeamQueue(max_pending, max_pending_per_team)
        self._condition = threading.Condition()
        self._owner_lock = threading.Lock()
        self._broker_id = uuid4().hex
        self._observation = observation
        if observation is not None:
            observation._bind_broker(self._broker_id)
        self._next_job = self._next_claim = 0
        self._max_lifetime_jobs = max_lifetime_jobs
        self._max_active_deliveries_per_job = max_active_deliveries_per_job
        self._max_pending = max_pending
        self._max_pending_per_team = max_pending_per_team
        self._max_coalesce_wait = float(max_coalesce_wait)
        self._slots = 0
        self._team_slots = {}
        self._jobs, self._groups, self._tokens = {}, {}, {}
        self._pending_claim = None
        self._fatal = self._quarantined = False
        self._closed = False
        self._failure = None
        with _owners_lock:
            _owners.add(self)
        with host_budget._lock:
            if (getattr(host_budget, '_grouped_broker_owner', None) is not None
                    or host_budget._requests or host_budget._claims or host_budget._operations):
                with _owners_lock:
                    _owners.discard(self)
                raise ValueError('host ledger must be empty and exclusive to this broker')
            host_budget._grouped_broker_owner = self

    @property
    def quarantined(self):
        # A health probe must not wait for a damaged condition or cohort lock.
        if self._quarantined:
            return True
        try:
            if self._worker._worker.quarantined:
                self._quarantined = True
        except BaseException as exc:
            # Unknown device state cannot become healthy merely because a
            # later read succeeds.  Keep the signal and first diagnostic.
            self._quarantined = True
            if self._failure is None:
                self._failure = exc
        return self._quarantined

    @property
    def failed(self):
        """Nonblocking health signal; an unreadable device state is quarantined."""
        if self._fatal or self.quarantined:
            return True
        try:
            return bool(self._worker._worker.failed)
        except BaseException as exc:
            self._quarantined = True
            if self._failure is None:
                self._failure = exc
            return True

    def _check_live(self):
        if self.failed:
            raise GroupedBrokerUnavailable('the underlying device owner is unavailable')
        self._policy.require_live_binding(self._profile, runtime_manifest=self._collect_runtime(), **self._identity)

    def _fail_closed(self, exc, *, quarantine=True):
        # Publish the sticky health signal before taking any potentially
        # damaged lock or touching a cohort's failure path.
        self._fatal = True
        if quarantine:
            self._quarantined = True
        if self._failure is None:
            self._failure = exc  # Retain traceback and potentially live buffers.
        with self._condition:
            for group in self._groups.values():
                group.state.mark_unknown()
            self._condition.notify_all()

    def submit(self, team_id, request):
        if not _name(team_id) or type(request) is not ForwardRequest:
            raise ValueError('a bounded team ID and exact forward request are required')
        self._check_live()
        self._policy.require_request(request)
        token = team_id, request.request_id
        with self._condition:
            if self._fatal or self.quarantined or self._closed:
                raise GroupedBrokerUnavailable('grouped coordinator is unavailable')
            if token in self._tokens:
                raise ValueError('team/request identity has already been used')
            if self._next_job >= self._max_lifetime_jobs:
                raise GroupedBrokerUnavailable('lifetime job limit reached; replace the owner')
            if self._slots >= self._max_pending or self._team_slots.get(team_id, 0) >= self._max_pending_per_team:
                raise GroupedBrokerUnavailable('pending capacity is exhausted')
            sequence = self._next_job
            self._next_job += 1
            job_id = f'{self._broker_id}-j-{sequence}'
            job = _Job(job_id, team_id, request.request_id, f'{self._broker_id}-r-{sequence}')
            handle = GroupedHandle(self, job)
            self._jobs[job_id] = job
            self._tokens[token] = handle
            self._slots += 1
            self._team_slots[team_id] = self._team_slots.get(team_id, 0) + 1
        failed = False
        ready_ns = None
        try:
            job.request_guard = self._host.request_guard(job_id)
            job.request_guard.open()
            job.host_request = job.request_guard.adopt()
            reserve = self._snapshot_reservation
            job.claim_guard = job.host_request.claim_guard(job_id, 'input-snapshot', reserve.nbytes,
                                                            items=reserve.items)
            job.claim_guard.reserve()
            job.claim = job.claim_guard.adopt()
            internal = ForwardRequest(job.worker_request_id, request.inputs, request.seed)
            snapshot = self._worker.snapshot_request(internal)
            self._policy.require_request(snapshot)
            if snapshot.request_id != job.worker_request_id:
                raise ValueError('worker snapshot identity mismatch')
            for tensor in snapshot.inputs.values():
                logical_bytes = tensor.numel() * tensor.element_size()
                if (not tensor.is_contiguous() or tensor.storage_offset() != 0
                        or tensor.untyped_storage().nbytes() != logical_bytes):
                    raise ValueError('worker snapshot must own exact contiguous tensor storage')
            job.snapshot = snapshot
            # Queue publication permits another thread to complete and retire
            # this job immediately.  The submit frame must not keep additional
            # aliases to its owned snapshot after that point.
            tensor = snapshot = internal = None
            with self._condition:
                if self._fatal or self.quarantined or self._closed:
                    raise GroupedBrokerUnavailable('owner became unavailable during snapshot')
                self._queue.put(team_id, job_id, job, compatibility_key=self._policy.compatibility_key)
                if self._observation is not None:
                    ready_ns = self._observation._ready_stamp(job)
                job.status = 'queued'
                job.ready_at = perf_counter()
                self._condition.notify_all()
            if self._observation is not None:
                self._observation._ready(job, ready_ns)
        except BaseException as exc:
            if self._observation is not None:
                self._observation._fault()
            self._fail_closed(exc)
            failed = True
        if failed:
            # Raise outside the exception handler: `from None` alone still
            # exposes the original exception through __context__.
            raise GroupedBrokerUnavailable('submission outcome is uncertain; replace the owner')
        return handle

    def lookup(self, team_id, request_id):
        """Recover a handle after an ambiguous submit return in this owner epoch."""
        if not _name(team_id) or not _name(request_id, 256):
            raise ValueError('invalid team/request identity')
        with self._condition:
            handle = self._tokens.get((team_id, request_id))
        return handle

    def _release_job(self, job):
        # No local tensor alias may survive the claim release; callers must
        # already have retired every group/consumer reference.
        job.snapshot = None
        if job.claim is not None:
            job.claim.release()
            job.claim = None
        if job.host_request is not None:
            job.host_request.close()
            job.host_request = None
        with self._condition:
            if job.status != 'released':
                job.status = 'released'
                self._slots -= 1
                self._team_slots[job.team_id] -= 1
                if self._team_slots[job.team_id] == 0:
                    del self._team_slots[job.team_id]
                self._condition.notify_all()

    def _cancel(self, job):
        cleanup = False
        failed = False
        group = None
        expected_busy = False
        try:
            with self._condition:
                if job.disposed:
                    return False
                if self._fatal or self.quarantined:
                    raise GroupedBrokerUnavailable('job status is unknown')
                if job.group is not None:
                    if job.cancel_active or job.dispose_active:
                        expected_busy = True
                        raise HostBusy('another control handoff is active')
                    # A raised cancellation may retain the cohort publication
                    # in its traceback.  Leave this marker set on failure so a
                    # concurrent disposer cannot retire it before quarantine.
                    job.cancel_active = True
                    group = job.group
                elif job.status == 'queued':
                    if not self._queue.cancel(job.job_id):
                        raise GroupedBrokerUnavailable('queued identity is missing')
                    job.cancel_requested = True
                    job.single = GroupedResult(job.job_id, job.team_id, job.request_id, 'cancelled', True, None)
                    if self._observation is not None:
                        job.terminal_ns = self._observation._stamp()
                    job.status = 'terminal'
                    accepted, cleanup = True, True
                    self._condition.notify_all()
                else:
                    accepted = False
            if group is not None:
                # Do not hold the broker condition while awaiting a cohort's
                # independent publication lock.  The job pin above prevents
                # disposal until this handoff is known complete.
                accepted = group.state.cancel(job.job_id)
                with self._condition:
                    job.cancel_active = False
                    self._condition.notify_all()
        except HostBusy as exc:
            # A concurrent, already-pinned control operation is normal.
            if expected_busy:
                raise
            self._fail_closed(exc)
            failed = True
        except BaseException as exc:
            self._fail_closed(exc)
            failed = True
        if failed:
            raise GroupedBrokerUnavailable('cancellation outcome is uncertain; replace the owner')
        if cleanup:
            try:
                if self._observation is not None:
                    self._observation._terminal(job, 'cancelled', job.terminal_ns)
                self._release_job(job)
            except BaseException as exc:
                if self._observation is not None:
                    self._observation._fault()
                self._fail_closed(exc)
                failed = True
            if failed:
                raise GroupedBrokerUnavailable('cancel cleanup failed')
        return accepted

    def _claim(self, timeout):
        deadline = None if timeout is None else perf_counter() + timeout
        with self._condition:
            while True:
                if self._fatal or self.quarantined or self._closed:
                    raise GroupedBrokerUnavailable('grouped coordinator is unavailable')
                now = perf_counter()
                try:
                    head_id = self._queue.peek_head_id()
                except QueueEmpty:
                    head_id = None
                if head_id is not None:
                    head = self._jobs[head_id]
                    claim_id = f'{self._broker_id}-c-{self._next_claim}'
                    self._next_claim += 1
                    self._pending_claim = claim_id
                    selected = self._queue.pop_compatible(
                        self._policy.compatibility_key, claim_id=claim_id,
                        allowed_sizes=self._policy.allowed_request_counts,
                        max_per_team=self._profile.max_per_owner)
                    if not selected and now >= head.ready_at + self._max_coalesce_wait:
                        selected = self._queue.pop_compatible(
                            self._policy.compatibility_key, claim_id=claim_id,
                            allowed_sizes=(1,), max_per_team=1)
                    if selected:
                        jobs = tuple(item.payload for item in selected)
                        selection_ns = (self._observation._stamp()
                                        if self._observation is not None else None)
                        if len(jobs) == 1:
                            job = jobs[0]
                            job.single = GroupedResult(job.job_id, job.team_id, job.request_id,
                                                       'failed', False, 'no_compatible_cohort')
                            if self._observation is not None:
                                job.terminal_ns = self._observation._stamp()
                            job.status = 'terminal'
                            claimed = job
                        else:
                            members = tuple(CohortMember(job.job_id, job.team_id, job.worker_request_id)
                                            for job in jobs)
                            group_id = f'{self._broker_id}-g-{self._next_claim}'
                            state = CohortState(group_id, members, self._profile,
                                                observation=self._observation)
                            claimed = _Group(group_id, jobs, state)
                            self._groups[group_id] = claimed
                            for job in jobs:
                                job.group, job.status = claimed, 'staged'
                        claimed.selection_ns = selection_ns
                        self._queue.acknowledge_claim(claim_id)
                        self._pending_claim = None
                        self._condition.notify_all()
                        selected = jobs = head = None
                        break
                    self._pending_claim = None
                    remaining_head = max(0., head.ready_at + self._max_coalesce_wait - now)
                else:
                    remaining_head = None
                remaining = None if deadline is None else deadline - now
                if remaining is not None and remaining <= 0:
                    claimed = None
                    break
                wait = remaining if remaining_head is None else remaining_head if remaining is None else min(remaining, remaining_head)
                self._condition.wait(wait)
        return claimed

    def run_once(self, timeout=None):
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0):
            raise ValueError('timeout must be finite and nonnegative')
        if not self._owner_lock.acquire(blocking=False):
            raise GroupedBrokerUnavailable('only one dispatcher may own the grouped worker')
        failed = False
        try:
            try:
                claimed = self._claim(timeout)
                if claimed is None:
                    return False
                if type(claimed) is _Job:
                    if self._observation is not None:
                        self._observation._selected((claimed,), claimed.selection_ns)
                        self._observation._terminal(claimed, 'failed', claimed.terminal_ns)
                    self._release_job(claimed)
                    return True
                if self._observation is not None:
                    self._observation._selected(claimed.jobs, claimed.selection_ns)
                self._execute(claimed)
                return True
            except BaseException as exc:
                if self._observation is not None:
                    self._observation._abandon_forward(None)
                self._fail_closed(exc)
                failed = True
        finally:
            self._owner_lock.release()
        if failed:
            raise GroupedBrokerUnavailable('dispatch outcome is uncertain; replace the owner')

    def _execute(self, group):
        self._check_live()
        self._policy.require_count(len(group.jobs))
        count = len(group.jobs)
        group.request_guard = self._host.request_guard(group.group_id)
        group.request_guard.open()
        group.host_request = group.request_guard.adopt()
        nbytes, items = self._worker.required_host_capacity(self._profile.profile_id)
        _positive(nbytes, 'group reservation bytes')
        _positive(items, 'group reservation items')
        group.claim_guard = group.host_request.claim_guard(group.group_id, 'group-envelope', nbytes, items=items)
        group.claim_guard.reserve()
        group.claim = group.claim_guard.adopt()
        validation = self._validation[count]
        group.validation_guard = group.host_request.claim_guard(group.group_id, 'output-validation',
                                                                validation.nbytes, items=validation.items)
        group.validation_guard.reserve()
        group.validation_claim = group.validation_guard.adopt()
        if not group.state.try_commit_launch():
            publication = group.state.finish_cancelled_before_launch()
            if self._observation is not None:
                self._observation._publication(group, publication)
            publication = None
            with self._condition:
                group.execution_finished = True
                self._condition.notify_all()
            self._maybe_retire(group)
            return
        requests = tuple(MicrobatchRequest(job.job_id, job.team_id, job.worker_request_id,
                                           self._profile.contract, job.snapshot.inputs) for job in group.jobs)
        group.plan = self._worker.prepare(self._profile.profile_id, requests, group.host_request,
                                          group.claim, group.group_id)
        requests = None
        group.launch = group.plan.launch()
        scope = None
        try:
            if self._observation is not None:
                scope = self._observation._start_forward(group, group.launch)
            outcome = self._worker.execute(group.plan, group.launch, cancel_event=None)
        except WorkerOutOfMemory as exc:
            group.quarantined = True
            self._fatal = self._quarantined = True
            if self._failure is None:
                self._failure = exc
            with self._condition:
                self._condition.notify_all()
            publication = group.state.publish_failure('out_of_memory', quarantine_required=True)
            if self._observation is not None:
                self._observation._finish_forward(scope)
                self._observation._publication(group, publication)
            return
        except BaseException:
            if self._observation is not None:
                self._observation._abandon_forward(scope)
            raise
        if self._observation is not None:
            self._observation._finish_forward(scope)
        if group.plan.outcome() is not outcome:
            raise ValueError('group outcome is not the exact plan-issued object')
        if outcome.quarantine_required:
            group.quarantined = True
            self._fatal = self._quarantined = True
            with self._condition:
                self._condition.notify_all()
            publication = group.state.publish_failure('execution_failed', quarantine_required=True)
            if self._observation is not None:
                self._observation._publication(group, publication)
            return
        publication = group.state.publish_success(outcome)
        if self._observation is not None:
            self._observation._publication(group, publication)
        publication = None
        outcome = None
        group.launch = None
        group.validation_claim.release()
        group.validation_claim = None
        with self._condition:
            group.execution_finished = True
            self._condition.notify_all()
        self._maybe_retire(group)

    def _new_delivery(self, job):
        with self._condition:
            if type(job) is not _Job or self._jobs.get(job.job_id) is not job:
                raise ValueError('job belongs to another broker')
            if job.disposed or job.dispose_active:
                raise GroupedBrokerUnavailable('job is disposed')
            if len(job.deliveries) >= self._max_active_deliveries_per_job:
                raise HostBusy('delivery limit reached for this job')
            ticket = GroupedDelivery(self, job)
            job.deliveries.add(ticket)
        return ticket

    def _active_deliveries(self, job):
        with self._condition:
            if type(job) is not _Job or self._jobs.get(job.job_id) is not job:
                raise ValueError('job belongs to another broker')
            return tuple(job.deliveries)

    def _release_delivery(self, ticket):
        if type(ticket) is not GroupedDelivery:
            raise ValueError('expected an exact delivery ticket')
        job = ticket._job
        with self._condition:
            if ticket._broker is not self:
                raise ValueError('delivery belongs to another broker')
            if ticket._released:
                job.deliveries.discard(ticket)
                return False
            if ticket not in job.deliveries:
                raise ValueError('delivery is forged or unregistered')
            if ticket._wait_token is not None:
                raise HostBusy('delivery wait is in progress')
            if ticket._consumer_thread is not None and ticket._consumer_thread is not threading.current_thread():
                raise HostBusy('delivery must be released by its consumer thread')
            if ticket._delivered and not (ticket._confirmed or ticket._abandoned):
                raise HostBusy('confirm or explicitly abandon the delivered result first')
            ticket._receipt = None
            ticket._released = True
            job.deliveries.remove(ticket)
            self._condition.notify_all()
        return True

    def _confirm_delivery(self, ticket, result):
        if type(ticket) is not GroupedDelivery:
            raise ValueError('expected an exact delivery ticket')
        with self._condition:
            if (ticket._broker is not self or ticket._released or ticket not in ticket._job.deliveries
                    or ticket._consumer_thread is not threading.current_thread()):
                raise GroupedBrokerUnavailable('delivery is unavailable to this consumer')
            if ticket._wait_token is not None:
                raise HostBusy('delivery wait is in progress')
            if not ticket._delivered or ticket._abandoned or ticket._receipt is not result:
                raise ValueError('result is not the exact un-abandoned ticket receipt')
            if ticket._confirmed:
                return False
            ticket._confirmed = True
        return True

    def _abandon_delivery(self, ticket):
        if type(ticket) is not GroupedDelivery:
            raise ValueError('expected an exact delivery ticket')
        with self._condition:
            if (ticket._broker is not self or ticket._released or ticket not in ticket._job.deliveries
                    or ticket._consumer_thread is not threading.current_thread()):
                raise GroupedBrokerUnavailable('delivery is unavailable to this consumer')
            if ticket._wait_token is not None:
                raise HostBusy('delivery wait is in progress')
            if not ticket._delivered or ticket._confirmed:
                raise ValueError('only an unconfirmed delivered result can be abandoned')
            if ticket._abandoned:
                return False
            ticket._abandoned = True
        return True

    def _wait(self, ticket, timeout):
        if type(ticket) is not GroupedDelivery:
            raise ValueError('expected an exact delivery ticket')
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0):
            raise ValueError('timeout must be finite and nonnegative')
        job = ticket._job
        deadline = None if timeout is None else perf_counter() + timeout
        token = object()
        publication = None
        failed = False
        try:
            with self._condition:
                if (ticket._broker is not self or ticket._released or ticket not in job.deliveries
                        or job.disposed):
                    raise GroupedBrokerUnavailable('delivery is unavailable')
                if ticket._wait_token is not None or ticket._delivered:
                    raise HostBusy('delivery is already waiting or was delivered')
                current = threading.current_thread()
                if ticket._consumer_thread is not None and ticket._consumer_thread is not current:
                    raise HostBusy('delivery is bound to another consumer thread')
                ticket._consumer_thread = current
                ticket._wait_token = token
                while job.group is None and job.single is None:
                    if self._fatal or self.quarantined:
                        raise GroupedBrokerUnavailable('job status is unknown')
                    remaining = None if deadline is None else deadline - perf_counter()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError('job is still pending')
                    self._condition.wait(remaining)
                group, single = job.group, job.single
            if single is not None:
                result = single
            else:
                # Read the publication in this frame under its state lock.
                # A nested wait() may be interrupted on its return event after
                # producing a publication but before our local assignment;
                # its exception traceback could then retain uncharged logits.
                # Polling while open never moves the publication into another
                # frame, and a real timeout has no captured output.
                timed_out = False
                state = group.state
                with state._condition:
                    while type(state._state) is _Open:
                        remaining = None if deadline is None else deadline - perf_counter()
                        if remaining is not None and remaining <= 0:
                            timed_out = True
                            break
                        state._condition.wait(0.1 if remaining is None else min(0.1, remaining))
                    if not timed_out:
                        publication = state._state
                        state._check_state(publication)
                if timed_out:
                    raise TimeoutError('cohort has not been published')
                member = next(result for result in publication.members if result.member.job_id == job.job_id)
                result = GroupedResult(job.job_id, job.team_id, job.request_id, member.status,
                                       member.cancel_requested, publication.error_code, member.logits)
            with self._condition:
                ticket._receipt = result
                ticket._delivered = True
            return result
        except BaseException as exc:
            # Once a publication was obtained, any construction or handoff
            # error can keep its logits alive in an exception traceback.  Keep
            # every claim and expose only a fresh error, not that traceback.
            if publication is None and type(exc) in (TimeoutError, HostBusy, GroupedBrokerUnavailable):
                raise
            self._fail_closed(exc)
            failed = True
        finally:
            with self._condition:
                if ticket._wait_token is token:
                    ticket._wait_token = None
                    self._condition.notify_all()
        if failed:
            raise GroupedBrokerUnavailable('delivery outcome is uncertain; replace the owner')

    def _dispose(self, job):
        already_disposed = False
        group = expected_error = None
        failed = False
        try:
            with self._condition:
                if job.disposed:
                    group, already_disposed = job.group, True
                elif self._fatal or self.quarantined:
                    expected_error = GroupedBrokerUnavailable('unhealthy owner requires process teardown')
                elif job.cancel_active or job.dispose_active:
                    expected_error = HostBusy('a control handoff is still active or uncertain')
                elif job.deliveries:
                    expected_error = HostBusy('all delivery tickets must be released before disposal')
                elif job.group is None:
                    if job.single is None:
                        expected_error = RuntimeError('cannot dispose a pending job')
                    else:
                        job.disposed = True
                else:
                    group = job.group
                    job.dispose_active = True
            if expected_error is None and group is not None and not already_disposed:
                phase = group.state.phase
                with self._condition:
                    if self._fatal or self.quarantined or group.quarantined:
                        expected_error = GroupedBrokerUnavailable('quarantined group requires process teardown')
                    elif phase != 'terminal':
                        job.dispose_active = False
                        expected_error = RuntimeError('cannot dispose an unfinished cohort')
                    else:
                        job.disposed, job.dispose_active = True, False
        except BaseException as exc:
            self._fail_closed(exc)
            failed = True
        if failed:
            raise GroupedBrokerUnavailable('disposal state is uncertain; replace the owner')
        if expected_error is not None:
            raise expected_error
        if group is not None:
            # A previous disposal can commit before this call is interrupted.
            # Retrying the same handle resumes retirement of this exact group.
            self._maybe_retire(group)
        return not already_disposed

    def _maybe_retire(self, group):
        failed = False
        try:
            with self._condition:
                if (self._fatal or self.quarantined or group.retiring or group.retired
                        or group.quarantined or not group.execution_finished
                        or not all(job.disposed for job in group.jobs)):
                    return
                group.retiring = True
            group.state.retire()
            if group.plan is not None:
                group.plan.dispose()
                group.plan = None
                group.claim = None
            elif group.claim is not None:
                group.claim.release()
                group.claim = None
            if group.validation_claim is not None:
                group.validation_claim.release()
                group.validation_claim = None
            group.host_request.close()
            group.host_request = None
            for job in group.jobs:
                self._release_job(job)
            with self._condition:
                group.retired = True
                self._condition.notify_all()
        except BaseException as exc:
            self._fail_closed(exc)
            failed = True
        if failed:
            raise GroupedBrokerUnavailable('group disposal is uncertain')

    def close(self):
        """Retire an idle healthy broker and release its exclusive ledger.

        Every handle and delivery must already be disposed.  A quarantined
        owner and its ledger remain rooted until supervised process teardown.
        """
        if not self._owner_lock.acquire(blocking=False):
            raise HostBusy('dispatcher is still active')
        try:
            with self._condition:
                if self._closed:
                    return False
                if self._fatal or self.quarantined:
                    raise GroupedBrokerUnavailable('unhealthy owner requires process teardown')
                if (self._slots or any(not job.disposed or job.deliveries for job in self._jobs.values())
                        or any(not group.retired for group in self._groups.values())):
                    raise HostBusy('all handles, deliveries and groups must be disposed')
                with self._host._lock:
                    if (getattr(self._host, '_grouped_broker_owner', None) is not self
                            or self._host._requests or self._host._claims or self._host._operations):
                        self._fatal = self._quarantined = True
                        raise GroupedBrokerUnavailable('exclusive ledger state is uncertain')
                    self._closed = True
                    self._host._grouped_broker_owner = None
                with _owners_lock:
                    _owners.discard(self)
                self._condition.notify_all()
            return True
        finally:
            self._owner_lock.release()
