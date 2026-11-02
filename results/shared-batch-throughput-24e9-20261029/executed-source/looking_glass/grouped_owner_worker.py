"""Experimental fixed-workspace owner for one- and two-row execution.

This is a trusted in-process custody boundary, not a device-memory limit or a
GPU certification. Both device lease objects remain rooted until process exit;
their commitments normally remain active for that lifetime. A separately
reviewed one-shot may retire inert accounting only after verified drain,
zero allocator counters and host claims, and disposal of model/input/output,
trace and callback custody, under its still-held physical lease, with no
further CUDA operation before clean attestation and process exit. The
caller must precharge input snapshots and retain host claims until every covered
result, input, exception, traceback and consumer alias has been disposed.
The budget telemetry callback must guard each device query and anchor every
reported total to the verified physical runtime's device total.
"""

import threading

import torch

from .budgeted_worker import (
    DeviceAdmission, DeviceQuarantined, ProfileUnavailable, WorkerContractError,
    _model_cache_binding, _require_no_autocast, _validate_profile, _worker_binding,
)
from .grouped_observation import _bind_forward, _bind_serial_forward
from .host_budget import HostBusy
from .memory_budget import AdmissionError, MemoryBudget, MemoryLease
from .owner_completion import OwnerCompletionAuthority
from .resident_worker import (
    ForwardResult, InvalidRequest, RequestCancelled, ResidentWorker, WorkerOutOfMemory,
    WorkerUnavailable, _ResidentResultCustody,
)


class _GroupedBindingBusy(HostBusy):
    """The exact grouped worker has a still-live producer or operation."""


class GroupedWorkspaceWorker:
    """A single resident worker with exactly two fixed workload signatures."""

    cancellation_exception = RequestCancelled

    def __init__(self, worker, budget, resident_lease, workspace_lease, *, profiles, authority, seed):
        if (type(worker) is not ResidentWorker or type(budget) is not MemoryBudget
                or type(resident_lease) is not MemoryLease or type(workspace_lease) is not MemoryLease
                or type(authority) is not OwnerCompletionAuthority):
            raise TypeError('expected exact resident worker, budget, leases and authority')
        if (resident_lease is workspace_lease or resident_lease._budget is not budget
                or workspace_lease._budget is not budget):
            raise ValueError('leases must be distinct capabilities of the supplied budget')
        if worker.device.type == 'cuda' and authority._device_guard is None:
            raise ValueError('CUDA workers require an exact owner lease guard')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if type(profiles) is not tuple or len(profiles) != 2:
            raise ValueError('exact serial and two-row profiles are required')
        admission = DeviceAdmission(budget, device=worker.device,
                                    max_estimated_incremental_bytes=workspace_lease.estimated_incremental_bytes)
        selected = {}
        for profile in profiles:
            signature = _validate_profile(profile, worker, admission)
            if signature in selected:
                raise ValueError('profile signatures must be distinct')
            selected[signature] = profile
        shapes = {profile.shape for profile in profiles}
        if (len(shapes) != 2 or {shape[0] for shape in shapes} != {1, 2}
                or len({shape[1] for shape in shapes}) != 1
                or profiles[0].input_dtypes != profiles[1].input_dtypes
                or profiles[0].profile_id == profiles[1].profile_id):
            raise ValueError('profiles must cover matching one- and two-row inputs')
        self._worker, self._model, self._budget = worker, worker._model, budget
        self._resident_lease, self._workspace_lease = resident_lease, workspace_lease
        self._resident_bytes = resident_lease.estimated_incremental_bytes
        self._workspace_bytes = workspace_lease.estimated_incremental_bytes
        if (type(self._resident_bytes) is not int or self._resident_bytes <= 0
                or type(self._workspace_bytes) is not int or self._workspace_bytes <= 0):
            raise ValueError('fixed lease commitments must be positive integers')
        self._authority, self._device_guard, self._seed = authority, authority._device_guard, seed
        self._binding, self._cache_binding = _worker_binding(worker), _model_cache_binding(worker)
        self._profiles = selected
        self._lock, self._closed = threading.Lock(), False
        # The shared authority permits a separately profiled CPU recorder for
        # its fixed serial owner; grouped observation remains unsupported.
        self._cpu_observation = self._cpu_observation_binding = None
        self._observation_bound = False
        self._preflight_lock = threading.Lock()
        self._preflight_failure = self._preflight_notification_failure = None
        self._broker_construction_lock = threading.Lock()
        self._broker_construction_failure = self._broker_construction_notification_failure = None
        self._teardown_broker = self._teardown_failure = self._teardown_notification_failure = None
        self._custody = _ResidentResultCustody()
        self._retiring_transfer = self._transfer_error = self._group_cleanup_error = None
        self._retiring_cancel = self._cancel_cleanup_error = None
        if worker._owner_completion is not None:
            raise ValueError('resident worker is already bound')
        authority._bind_worker(self)
        try:
            worker._owner_completion = authority
            self._check_leases()
        except BaseException:
            authority.mark_unsafe()
            raise

    @property
    def quarantined(self):
        return self._authority.unsafe or self._worker.quarantined

    @property
    def failed(self):
        return self.quarantined or self._worker.poisoned

    @property
    def poisoned(self):
        return self._closed or self.failed

    def _check_available(self):
        if self.quarantined or self._closed or self._worker.poisoned:
            raise DeviceQuarantined('owner is unavailable; replace the process')
        try:
            matches = (_worker_binding(self._worker) == self._binding
                       and _model_cache_binding(self._worker) == self._cache_binding
                       and self._worker._owner_completion is self._authority
                       and self._authority._device_guard is self._device_guard)
        except BaseException:
            self._authority.mark_unsafe()
            raise
        if not matches:
            self._authority.mark_unsafe()
            raise WorkerUnavailable('worker binding changed after profiling')

    def _check_leases(self):
        try:
            active = self._budget.active_leases()
            expected = (self._resident_lease, self._workspace_lease)
            if (len(active) != 2 or set(active) != set(expected)
                    or any(lease.released or lease.trace_bytes('gpu') or lease.trace_bytes('host')
                           or lease._trace_limits != {'gpu': 0, 'host': 0} for lease in expected)
                    or self._resident_lease.estimated_incremental_bytes != self._resident_bytes
                    or self._workspace_lease.estimated_incremental_bytes != self._workspace_bytes):
                raise WorkerContractError('fixed workspace leases changed')
            self._authority._before_device()
            snapshot = self._budget.snapshot()
            self._authority._before_device()
            total = self._resident_bytes + self._workspace_bytes
            committed = snapshot.committed_incremental_bytes + snapshot.reserve_bytes
            if (snapshot.active_leases != 2 or snapshot.committed_incremental_bytes != total
                    or snapshot.gpu_trace_bytes or snapshot.host_trace_bytes
                    or committed > snapshot.telemetry.device_free_bytes
                    or snapshot.telemetry.process_reserved_bytes + committed > snapshot.telemetry.device_total_bytes):
                raise AdmissionError('fixed workspace is not covered by live headroom')
            if self._authority.unsafe:
                raise DeviceQuarantined('owner became unsafe during telemetry')
        except BaseException:
            self._authority.mark_unsafe()
            raise

    def snapshot_request(self, request):
        self._check_available()
        snapshot = self._worker.snapshot_request(request)
        if snapshot.seed != self._seed:
            raise InvalidRequest('request seed differs from the trusted profile')
        return snapshot

    def execute(self, request, cancel_event=None):
        """Return a serial result held in producer custody until confirmed."""
        return self._execute(request, cancel_event=cancel_event)

    def execute_control(self, request):
        """Execute a trusted profiled one- or two-row validation control."""
        return self._execute(request, control=True)

    def _execute_grouped(self, request, group, launch, cancel_event=None):
        from .grouped_worker import GroupPlan
        if type(group) is not GroupPlan or group._owner._worker is not self:
            raise ValueError('group does not belong to this worker')
        if cancel_event is not None:
            raise InvalidRequest('one member cannot cancel the shared forward')
        try:
            return self._execute(request, group=group, launch=launch)
        except BaseException as exc:
            # Launch ownership is committed before the resident call. If
            # that handoff is interrupted, it is not proof no work began.
            if group._started:
                self._authority.mark_unsafe()
                if self._custody.error is None:
                    self._custody.error = exc
            raise

    def _execute(self, request, cancel_event=None, *, group=None, launch=None, control=False):
        # A direct control's snapshot/signature work is serialized with the
        # forward: a competing preflight fault cannot race an active result.
        # Its caller retains the separately charged request/exception aliases.
        failed = False
        was_oom = False
        was_quarantined = False
        with self._preflight_lock:
            if self._authority.unsafe or self._preflight_failure is not None:
                raise DeviceQuarantined('owner is unavailable; replace the process')
            try:
                return self._execute_inner(request, cancel_event, group=group, launch=launch, control=control)
            except BaseException as exc:
                if (type(exc) in (InvalidRequest, ProfileUnavailable)
                        and (group is None or not group._started)):
                    raise
                if (type(exc) is RequestCancelled and not self._authority.unsafe
                        and all(value is None for value in (self._custody.request, self._custody.logits,
                                                           self._custody.result, self._custody.error,
                                                           self._custody.producer_error,
                                                           self._custody.forward_error,
                                                           self._custody.state_error))):
                    raise
                try:
                    try:
                        if self._preflight_failure is None:
                            self._preflight_failure = exc
                    finally:
                        if self._preflight_failure is None:
                            self._preflight_failure = exc
                        self._authority._unsafe = True
                except BaseException as notification_error:
                    if self._preflight_notification_failure is None:
                        self._preflight_notification_failure = notification_error
                was_oom = isinstance(exc, WorkerOutOfMemory)
                was_quarantined = isinstance(exc, DeviceQuarantined)
                failed = True
        if failed:
            if was_oom:
                raise WorkerOutOfMemory('grouped owner failed; replace the process')
            if was_quarantined:
                raise DeviceQuarantined('grouped owner became unsafe; replace the process')
            raise WorkerUnavailable('grouped owner outcome is uncertain; replace the process')

    def _execute_inner(self, request, cancel_event=None, *, group=None, launch=None, control=False):
        if cancel_event is not None and type(cancel_event) is not threading.Event:
            raise InvalidRequest('cancel_event must be a threading.Event')
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelled('request cancelled before execution')
        _require_no_autocast()
        snapshot = self.snapshot_request(request)
        signature = (tuple(snapshot.inputs['input_ids'].shape),
                     tuple(sorted((name, str(value.dtype)) for name, value in snapshot.inputs.items())))
        profile = self._profiles.get(signature)
        if (profile is None or (group is None and not control and profile.shape[0] != 1)
                or (group is not None and profile.shape[0] != 2)):
            raise ProfileUnavailable('no trusted profile matches this request')
        with self._lock:
            self._check_available()
            if any(getattr(self._custody, name) is not None for name in
                   ('result', 'logits', 'producer_error', 'forward_error', 'state_error')):
                self._authority.mark_unsafe()
                raise WorkerContractError('previous result custody was not transferred')
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled('request cancelled before execution')
            self._check_leases()
            _require_no_autocast()
            if group is not None:
                all_cancelled = group._abort_if_all_cancelled(launch)
                if all_cancelled is not None:
                    return all_cancelled
                group._start_under_gate(launch, profile, snapshot)
                _bind_forward(self._worker, profile, snapshot, launch)
            else:
                _bind_serial_forward(self._worker, profile, snapshot)
            self._custody.generation += 1
            self._custody.request, self._custody.error = snapshot, None
            try:
                result = self._worker._execute_with_custody(snapshot, cancel_event, self._custody)
                self._check_available()
                self._check_leases()
                if (type(result) is not ForwardResult or result is not self._custody.result
                        or result.request_id != snapshot.request_id or type(result.logits) is not torch.Tensor
                        or result.logits.device.type != 'cpu' or result.logits.requires_grad
                        or result.logits.is_inference() or not result.logits.is_floating_point()
                        or result.logits.ndim != 3 or tuple(result.logits.shape[:2]) != profile.shape
                        or result.logits.numel() * result.logits.element_size() > profile.output_limit_bytes):
                    raise WorkerContractError('resident worker returned an unsupported result')
                if group is None:
                    return result
                outcome = group._finish_under_gate(launch, result)
                if outcome.quarantine_required:
                    raise WorkerContractError('grouped result failed validation or splitting')
                self._transfer_result(result, group=group)
                return outcome
            except RequestCancelled as exc:
                if group is not None:
                    self._custody.error = exc
                    self._authority.mark_unsafe()
                    raise
                try:
                    unhealthy = False
                    with self._authority._decision_lock:
                        if self._authority.unsafe or self._worker.quarantined or self._worker.poisoned:
                            self._authority.mark_unsafe()
                            unhealthy = True
                        else:
                            self._retiring_cancel = (self._custody.request, self._custody.logits,
                                                     self._custody.result, self._custody.error,
                                                     self._custody.producer_error, self._custody.forward_error,
                                                     self._custody.state_error)
                            self._custody.request = self._custody.logits = self._custody.result = self._custody.error = None
                            self._custody.producer_error = self._custody.forward_error = self._custody.state_error = None
                    if unhealthy:
                        raise DeviceQuarantined('owner became unsafe during cancellation') from exc
                    if not self._authority.unsafe:
                        self._retiring_cancel = None
                except BaseException as cleanup_exc:
                    self._cancel_cleanup_error = cleanup_exc
                    self._authority.mark_unsafe()
                    raise
                request = snapshot = None
            except BaseException as exc:
                self._custody.error = exc
                self._authority.mark_unsafe()
                if group is not None and group._started:
                    try:
                        group._fail_under_gate(launch, exc)
                    except BaseException as cleanup_exc:
                        self._group_cleanup_error = cleanup_exc
                raise
            # Raise a fresh cancellation outside the original exception
            # handler so the producer traceback cannot retain its snapshot.
            raise RequestCancelled('request cancelled after execution')

    def _transfer_result(self, result, *, group=None):
        try:
            with self._authority._decision_lock:
                if self._authority.unsafe or self._custody.result is not result:
                    raise WorkerContractError('unexpected or unsafe result custody generation')
                if group is not None:
                    group._owned_result = result
                self._retiring_transfer = (self._custody.request, self._custody.logits,
                                           self._custody.result, self._custody.error,
                                           self._custody.producer_error, self._custody.forward_error,
                                           self._custody.state_error)
                self._custody.request = self._custody.logits = self._custody.result = self._custody.error = None
                self._custody.producer_error = self._custody.forward_error = self._custody.state_error = None
            if not self._authority.unsafe:
                self._retiring_transfer = None
        except BaseException as exc:
            self._transfer_error = exc
            self._authority.mark_unsafe()
            raise

    def confirm_control_result(self, result):
        """Attest that the exact ungrouped result is rooted under separate host custody."""
        with self._lock:
            self._transfer_result(result)

    def _bind_grouped_broker(self, broker):
        """Bind the exact broker while its constructor owns the preflight gate."""
        from .grouped_broker import GroupedBroker
        if type(broker) is not GroupedBroker or not self._preflight_lock.locked():
            raise WorkerContractError('grouped broker requires the exact execution gate')
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._authority._before_device()
            with self._authority._decision_lock:
                if (self._authority._worker is not self or self._authority._worker_retired
                        or self._authority._broker is not None or self._authority.unsafe
                        or self._worker._owner_completion is not self._authority
                        or self._authority._device_guard is not self._device_guard
                        or self._cpu_observation is not None or self._cpu_observation_binding is not None
                        or self._observation_bound or self._teardown_broker is not None
                        or self._closed or self._worker.poisoned):
                    raise WorkerContractError('grouped worker authority binding changed')
                if (any(getattr(self._custody, name) is not None for name in
                        ('request', 'logits', 'result', 'error', 'producer_error', 'forward_error', 'state_error'))
                        or self._retiring_transfer is not None or self._retiring_cancel is not None
                        or self._worker._drain_pending or self._worker._snapshot_error is not None
                        or any(value is not None for value in self._worker._live_buffers)
                        or any(value is not None for value in self._worker._retiring_buffers)):
                    return False
                self._authority._broker = broker
                return True
        finally:
            self._lock.release()

    def _allow_retirement(self):
        # This is only the retirement decision. Reference drops and ledger
        # operations happen after releasing the authority lock. If a broker
        # health read is in progress, wait outside the decision lock so an
        # indeterminate observation cannot authorize a new release.
        from .grouped_broker import GroupedBroker
        broker = self._authority._broker
        health_gate = broker._health_lock if type(broker) is GroupedBroker else None
        acquired = False
        try:
            if health_gate is not None:
                acquired = health_gate.acquire()
            with self._authority._decision_lock:
                if (self._authority.unsafe or self._worker.quarantined or self._worker.poisoned
                        or (broker is not None and (broker._fatal or broker._quarantined))):
                    self._authority.mark_unsafe()
                    raise DeviceQuarantined('unsafe owner requires process teardown')
        finally:
            if acquired:
                health_gate.release()

    def close(self):
        # The process owner, not this adapter, disposes the model and leases.
        self._closed = True


def retire_grouped_owner(worker, broker, *, callers_quiescent):
    """Detach one closed grouped owner after trusted caller quiescence.

    ``callers_quiescent`` attests that dispatch, control, ticket, cancellation,
    disposal and health callers have stopped and that their aliases, including
    any prior teardown exception graphs, are gone.
    The caller must root the worker, broker, physical lease and both device
    lease objects until process exit. Normally both commitments remain active
    until exit. A separately reviewed one-shot may release inert accounting
    only after verified drain, zero allocator counters and host claims, and
    disposal of model/input/output, trace and callback custody, holding the
    physical lease and committing to no further CUDA work before attestation
    and exit. This
    is not proof of external alias or native-memory disposal, nor does this
    helper release a device lease.
    """
    from .grouped_broker import GroupedBroker
    from .grouped_broker_state import _RETIRED
    from .host_budget import HostBusy
    if type(worker) is not GroupedWorkspaceWorker or type(broker) is not GroupedBroker:
        raise TypeError('exact grouped worker and broker are required')
    if callers_quiescent is not True:
        raise ValueError('the trusted owner must attest caller quiescence')
    authority = worker._authority
    if (worker._teardown_broker is broker and authority._worker_retired
            and authority._worker is None and authority._broker is None and not authority.unsafe):
        return False
    held = []
    commit_started = False
    checked_busy = False
    failed = False
    rejected = None
    try:
        authority._before_device()
        # Nonblocking acquisition verifies quiescence without waiting while
        # holding the authority decision lock. The owner attestation is still
        # required because a new caller could otherwise start after a check.
        locks = (worker._preflight_lock, worker._lock, broker._owner_lock,
                 broker._admission_lock, broker._runtime_lock, broker._health_lock,
                 broker._condition, broker._queue._lock, broker._host._lock)
        for lock in locks:
            if not lock.acquire(blocking=False):
                checked_busy = True
                raise HostBusy('a grouped owner operation is still active')
            held.append(lock)
        resident = worker._worker
        custody_names = ('request', 'logits', 'result', 'error', 'producer_error', 'forward_error', 'state_error')
        if (authority._worker is not worker or authority._broker is not broker
                or broker._worker._worker is not worker or authority._worker_retired
                or not worker._closed or not resident._closed or resident._model is not None
                or not broker._closed or broker._fatal or broker._quarantined or broker._slots
                or broker._pending_claim is not None or broker._queue._size or broker._queue._claimed_size
                or broker._queue._claims or broker._queue._claimed_jobs or broker._queue._job_teams
                or broker._queue._teams or broker._queue._rotation
                or getattr(broker._host, '_grouped_broker_owner', None) is not None
                or broker._host._requests or broker._host._claims or broker._host._operations
                or resident._drain_pending or resident.poisoned or resident._snapshot_error is not None
                or any(value is not None for value in resident._live_buffers)
                or any(value is not None for value in resident._retiring_buffers)
                or any(getattr(worker._custody, name) is not None for name in custody_names)
                or worker._retiring_transfer is not None or worker._retiring_cancel is not None
                or worker._transfer_error is not None or worker._group_cleanup_error is not None
                or worker._cancel_cleanup_error is not None or worker._preflight_failure is not None
                or worker._broker_construction_failure is not None
                or any((name.endswith('failure') or name.endswith('error')) and value is not None
                       for name, value in vars(worker).items())
                or any(name.endswith('failure') and value is not None
                       for name, value in vars(broker).items())
                or any(not job.disposed or job.deliveries or job.snapshot is not None
                       or job.cancel_active or job.dispose_active or job.status != 'released'
                       or (job.single is not None and job.single.logits is not None)
                       or any(name.endswith('failure') and value is not None for name, value in vars(job).items())
                       for job in broker._jobs.values())
                or any(not group.retired or group.state._state is not _RETIRED
                       or group.plan is not None or group.claim is not None
                       or group.validation_claim is not None or group.host_request is not None
                       or any(name.endswith('failure') and value is not None for name, value in vars(group).items())
                       for group in broker._groups.values())):
            checked_busy = True
            raise HostBusy('grouped owner custody is not fully retired')
        with authority._decision_lock:
            if authority.unsafe or resident.poisoned or broker._fatal or broker._quarantined:
                raise DeviceQuarantined('owner became unsafe before retirement')
            # Both exact objects remain in caller and worker roots while the
            # permanent one-shot authority binding is detached.
            commit_started = True
            worker._teardown_broker = broker
            authority._worker_retired = True
            authority._broker = None
            authority._worker = None
    except BaseException as exc:
        if type(exc) is HostBusy and checked_busy and not commit_started:
            rejected = exc
        else:
            try:
                try:
                    if worker._teardown_failure is None:
                        worker._teardown_failure = exc
                finally:
                    if worker._teardown_failure is None:
                        worker._teardown_failure = exc
                    authority._unsafe = True
            except BaseException as notification_error:
                if worker._teardown_notification_failure is None:
                    worker._teardown_notification_failure = notification_error
            failed = True
    finally:
        try:
            for lock in reversed(held):
                lock.release()
        except BaseException as exc:
            try:
                if worker._teardown_notification_failure is None:
                    worker._teardown_notification_failure = exc
            finally:
                authority._unsafe = True
            failed = True
    if failed:
        raise WorkerUnavailable('grouped retirement is uncertain; replace the process')
    if rejected is not None:
        raise rejected
    return True
