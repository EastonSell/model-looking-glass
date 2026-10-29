"""Fixed-workspace custody for one trusted, serialized service owner.

This is not a general device-memory limit. Its leases are cooperative policy
commitments and are intentionally retained through the supervised process exit.
"""

import threading

import torch

from .budgeted_worker import (
    DeviceAdmission, DeviceQuarantined, ProfileUnavailable, WorkerContractError,
    _model_cache_binding, _require_no_autocast, _validate_profile, _worker_binding,
)
from .memory_budget import AdmissionError, MemoryBudget, MemoryLease
from .resident_worker import (
    ForwardResult, InvalidRequest, RequestCancelled, ResidentWorker,
    WorkerUnavailable, _ResidentResultCustody,
)


class OwnerCompletionAuthority:
    """Order one service's success decisions against its sticky unsafe signal.

    ``mark_unsafe`` is a nonblocking CPython reference assignment. A success
    decision holds ``_decision_lock`` and linearizes at its read of that flag.
    A later mark may therefore precede visible delivery of an already-decided
    result. All relevant unsafe sources must mark this exact authority; polling
    an independent flag only orders its detection, not its original change.
    A trusted device guard must independently retain any resource-bearing
    partials and their claims after failure; its exception traceback can be
    exposed to callers and is not an authority-owned custody root.
    """

    def __init__(self, *, device_guard=None):
        if device_guard is not None and not callable(device_guard):
            raise TypeError('device guard must be trusted callable owner code')
        self._unsafe = False
        self._decision_lock = threading.Lock()
        self._device_guard = device_guard
        self._guard_binding = device_guard
        self._worker = None
        self._worker_bound = False
        self._worker_retired = False
        self._broker = None

    def mark_unsafe(self):
        self._unsafe = True

    @property
    def unsafe(self):
        return self._unsafe

    def _assert_recording_disabled(self):
        from .grouped_observation import _local
        if (getattr(_local, 'forward', None) is not None
                or (self._broker is not None
                    and getattr(self._broker, '_serial_observation', None) is not None)):
            raise WorkerContractError('fixed service requires recording to be disabled')

    def _before_device(self):
        """Commit one device operation after checking the actual owner lease."""
        try:
            self._assert_recording_disabled()
            if self._unsafe or self._device_guard is not self._guard_binding:
                raise DeviceQuarantined('owner is unsafe before device operation')
            # OwnerSession.check can perform I/O; do not hold the decision
            # lock while it runs. A separately revoked session is observable
            # at this check, not atomically ordered with external revocation.
            if self._device_guard is not None:
                self._device_guard()
            with self._decision_lock:
                self._assert_recording_disabled()
                if self._unsafe or self._device_guard is not self._guard_binding:
                    raise DeviceQuarantined('owner is unsafe before device operation')
        except BaseException:
            self.mark_unsafe()
            raise

    def _bind_worker(self, worker):
        if self._worker_bound or self._unsafe:
            raise ValueError('completion authority is already bound or unsafe')
        self._worker_bound = True
        self._worker = worker

    def _bind_broker(self, broker, worker):
        if self._worker is not worker or self._broker is not None or self._unsafe:
            raise ValueError('completion authority does not match this worker')
        self._broker = broker

    def retire_unbrokered_worker(self, worker):
        """Detach one closed collector worker after its producers have stopped.

        The caller retains its own worker/model roots and memory claims until
        actual disposal. On any failure or interrupted return, it must mark
        this authority unsafe and retain those roots through process exit: an
        interrupt at the return boundary can follow a completed detachment.
        This does not prove process quiescence or attestation.
        """
        if type(worker) is not LifetimeWorkspaceWorker or self._worker is not worker:
            raise ValueError('worker does not belong to this completion authority')
        if not worker._lock.acquire(blocking=False):
            raise WorkerUnavailable('worker execution has not stopped')
        try:
            try:
                with self._decision_lock:
                    if worker._worker.poisoned:
                        # Resident/process poison is an unsafe source even if
                        # another signaler has not yet marked this authority.
                        self._unsafe = True
                    if (self._unsafe or self._broker is not None or self._worker_retired
                            or not worker._closed or not worker._worker._closed
                            or worker._worker._model is not None or worker._worker._drain_pending
                            or worker._worker._snapshot_error is not None
                            or any(value is not None for value in worker._worker._live_buffers)
                            or any(value is not None for value in worker._worker._retiring_buffers)
                            or any(getattr(worker._custody, name) is not None
                                   for name in ('request', 'logits', 'result', 'error', 'producer_error',
                                                'forward_error', 'state_error'))
                            or worker._warmup_result is not None
                            or worker._retiring_cancel is not None
                            or worker._retiring_transfer is not None
                            or worker._cancel_cleanup_error is not None
                            or worker._transfer_cleanup_error is not None):
                        raise WorkerUnavailable('unbrokered worker custody is not fully retired')
                    # The exact worker argument is a strong root while this lock is
                    # held; permanent one-shot binding survives detachment.
                    self._worker_retired = True
                    self._worker = None
            finally:
                worker._lock.release()
        except BaseException:
            if self._worker_retired:
                try:
                    self.mark_unsafe()
                finally:
                    self._unsafe = True
            raise


class LifetimeWorkspaceWorker:
    """One resident worker using a real, process-lifetime workspace lease.

    The owner alone supplies the trusted profiles and retains both leases.
    This worker never releases either lease or disposes the resident model.
    """

    cancellation_exception = RequestCancelled

    def __init__(self, worker, budget, resident_lease, workspace_lease, *, profiles, authority):
        if (type(worker) is not ResidentWorker or type(budget) is not MemoryBudget
                or type(resident_lease) is not MemoryLease or type(workspace_lease) is not MemoryLease
                or type(authority) is not OwnerCompletionAuthority):
            raise TypeError('expected exact resident worker, budget, leases and authority')
        if (resident_lease is workspace_lease or resident_lease._budget is not budget
                or workspace_lease._budget is not budget):
            raise ValueError('leases must be distinct capabilities of the supplied budget')
        if worker.device.type == 'cuda' and authority._device_guard is None:
            raise ValueError('CUDA workers require an exact owner lease guard')
        profiles = tuple(profiles)
        if len(profiles) not in (1, 2):
            raise ValueError('one warm profile or one cold/warm pair is required')
        admission = DeviceAdmission(budget, device=worker.device,
            max_estimated_incremental_bytes=workspace_lease.estimated_incremental_bytes)
        self._worker = worker
        self._model = worker._model
        self._budget = budget
        self._resident_lease = resident_lease
        self._workspace_lease = workspace_lease
        self._resident_bytes = resident_lease.estimated_incremental_bytes
        self._workspace_bytes = workspace_lease.estimated_incremental_bytes
        if (type(self._resident_bytes) is not int or self._resident_bytes <= 0
                or type(self._workspace_bytes) is not int or self._workspace_bytes <= 0):
            raise ValueError('fixed lease commitments must be positive integers')
        self._authority = authority
        self._device_guard = authority._device_guard
        self._binding = _worker_binding(worker)
        self._cache_binding = _model_cache_binding(worker)
        self._profiles = {}
        self._warm_profiles = {}
        for profile in profiles:
            signature = _validate_profile(profile, worker, admission)
            if signature in self._profiles:
                prior = self._profiles[signature]
                if (profile.model_id != prior.model_id or profile.device != prior.device
                        or profile.output_limit_bytes != prior.output_limit_bytes
                        or profile.estimated_incremental_bytes != prior.estimated_incremental_bytes
                        or profile.profile_id == prior.profile_id):
                    raise ValueError('conflicting profiles for one workload signature')
                self._warm_profiles[signature] = profile
            else:
                self._profiles[signature] = profile
        if len(profiles) == 2 and (len(self._profiles) != 1 or len(self._warm_profiles) != 1):
            raise ValueError('cold and warm profiles must cover the same signature')
        self._cold_pending = len(profiles) == 2
        self._warmup_result = None
        self._lock = threading.Lock()
        self._closed = False
        self._custody = _ResidentResultCustody()
        self._retiring_cancel = None
        self._retiring_transfer = None
        self._cancel_cleanup_error = None
        self._transfer_cleanup_error = None
        if worker._owner_completion is not None:
            raise ValueError('resident worker is already bound')
        authority._bind_worker(self)
        worker._owner_completion = authority
        self._check_leases()

    @property
    def quarantined(self):
        if self._worker.quarantined or self._recorder_attached():
            self._authority.mark_unsafe()
        return self._authority.unsafe

    @property
    def failed(self):
        if self.quarantined:
            return True
        if self._worker.poisoned:
            self._authority.mark_unsafe()
            return True
        return False

    def _recorder_attached(self):
        broker = self._authority._broker
        return broker is not None and getattr(broker, '_serial_observation', None) is not None

    @property
    def poisoned(self):
        return self._closed or self.failed

    @property
    def max_input_bytes(self):
        self._check_available()
        return self._binding[4]

    @property
    def max_output_bytes(self):
        self._check_available()
        return self._binding[5]

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
                raise WorkerContractError('fixed service leases changed')
            # The budget callback is trusted owner code and, for CUDA, must
            # itself guard each telemetry query and retain any resource-bearing
            # partials/claims on failure. These outer checks catch a session
            # loss before or during the snapshot as well; the callback's
            # exception traceback is not a private custody root.
            self._authority._before_device()
            snapshot = self._budget.snapshot()
            self._authority._before_device()
            total = self._resident_bytes + self._workspace_bytes
            committed = snapshot.committed_incremental_bytes + snapshot.reserve_bytes
            if (snapshot.active_leases != 2 or snapshot.committed_incremental_bytes != total
                    or snapshot.gpu_trace_bytes or snapshot.host_trace_bytes
                    or committed > snapshot.telemetry.device_free_bytes
                    or snapshot.telemetry.process_reserved_bytes + committed
                    > snapshot.telemetry.device_total_bytes):
                raise AdmissionError('fixed workspace is not covered by live headroom')
            if self._authority.unsafe:
                raise DeviceQuarantined('owner became unsafe during telemetry')
        except BaseException:
            self._authority.mark_unsafe()
            raise

    def snapshot_request(self, request):
        self._check_available()
        return self._worker.snapshot_request(request)

    def execute(self, request, cancel_event=None):
        if cancel_event is not None and type(cancel_event) is not threading.Event:
            raise InvalidRequest('cancel_event must be a threading.Event')
        _require_no_autocast()
        self._check_available()
        snapshot = self._worker.snapshot_request(request)
        signature = (tuple(snapshot.inputs['input_ids'].shape),
                     tuple(sorted((name, str(value.dtype)) for name, value in snapshot.inputs.items())))
        profile = self._profiles.get(signature)
        if profile is None:
            raise ProfileUnavailable('no trusted profile matches this request')
        with self._lock:
            self._check_available()
            if (self._custody.result is not None or self._custody.logits is not None
                    or self._custody.producer_error is not None or self._custody.forward_error is not None
                    or self._custody.state_error is not None):
                self._authority.mark_unsafe()
                raise WorkerContractError('previous result custody was not transferred')
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled('request cancelled before execution')
            self._check_leases()
            _require_no_autocast()
            from .grouped_observation import _local
            if getattr(_local, 'forward', None) is not None:
                self._authority.mark_unsafe()
                raise WorkerContractError('fixed service requires recording to be disabled')
            self._custody.generation += 1
            self._custody.request = snapshot
            self._custody.error = None
            try:
                result = self._worker._execute_with_custody(snapshot, cancel_event, self._custody)
                if self._authority.unsafe or self._worker.quarantined or self._worker.poisoned:
                    self._authority.mark_unsafe()
                    raise DeviceQuarantined('owner became unsafe during execution')
                if self._recorder_attached() or getattr(_local, 'forward', None) is not None:
                    raise WorkerContractError('fixed service recording policy changed during execution')
                if (_worker_binding(self._worker) != self._binding
                        or _model_cache_binding(self._worker) != self._cache_binding
                        or self._worker._owner_completion is not self._authority
                        or self._authority._device_guard is not self._device_guard):
                    raise WorkerContractError('worker binding changed during execution')
                # A trusted owner must not alter these commitments while a
                # forward runs. Detect that violation before returning output.
                self._check_leases()
                if (type(result) is not ForwardResult or result is not self._custody.result
                        or result.request_id != snapshot.request_id or type(result.logits) is not torch.Tensor
                        or result.logits.device.type != 'cpu' or result.logits.requires_grad
                        or result.logits.is_inference() or not result.logits.is_floating_point()
                        or result.logits.ndim != 3 or tuple(result.logits.shape[:2]) != profile.shape
                        or result.logits.numel() * result.logits.element_size() > profile.output_limit_bytes):
                    raise WorkerContractError('resident worker returned an unsupported result')
                return result
            except RequestCancelled:
                try:
                    with self._authority._decision_lock:
                        if (self._authority.unsafe or self._worker.quarantined
                                or self._worker.poisoned):
                            self._authority.mark_unsafe()
                        else:
                            self._retiring_cancel = (self._custody.request, self._custody.logits,
                                                     self._custody.error, self._custody.producer_error,
                                                     self._custody.forward_error, self._custody.state_error)
                            self._custody.request = None
                            self._custody.logits = None
                            self._custody.error = None
                            self._custody.producer_error = None
                            self._custody.forward_error = None
                            self._custody.state_error = None
                    # Last references are released outside the decision lock.
                    if not self._authority.unsafe:
                        self._retiring_cancel = None
                except BaseException as exc:
                    self._cancel_cleanup_error = exc
                    self._authority.mark_unsafe()
                    raise
                raise
            except BaseException as exc:
                self._custody.error = exc
                self._authority.mark_unsafe()
                raise

    def _transfer_result(self, result):
        """Drop producer custody only after the broker roots this exact result."""
        try:
            with self._authority._decision_lock:
                self._stage_transfer_locked(result)
            if not self._authority.unsafe:
                self._retiring_transfer = None
        except BaseException as exc:
            self._transfer_cleanup_error = exc
            self._authority.mark_unsafe()
            raise

    def _stage_transfer_locked(self, result):
        if self._authority.unsafe or self._custody.result is not result:
            raise WorkerContractError('unexpected or unsafe result custody generation')
        # The authority read is the transfer/retirement decision. Stage the
        # worker's separate snapshot as well as the exact result and error.
        self._retiring_transfer = (self._custody.result, self._custody.request,
                                   self._custody.logits, self._custody.error, self._custody.producer_error,
                                   self._custody.forward_error, self._custody.state_error)
        self._custody.result = None
        self._custody.request = None
        self._custody.logits = None
        self._custody.error = None
        self._custody.producer_error = None
        self._custody.forward_error = None
        self._custody.state_error = None

    def _finish_cold(self, result):
        """Advance to the warm profile after private cold-result validation."""
        with self._lock:
            self._warmup_result = result
            try:
                with self._authority._decision_lock:
                    if self._authority.unsafe or not self._cold_pending:
                        raise WorkerContractError('cold warmup is not ready to retire')
                    if self._custody.result is not result:
                        raise WorkerContractError('cold result identity changed')
                    self._profiles = self._warm_profiles
                    self._stage_transfer_locked(result)
                    self._cold_pending = False
                if not self._authority.unsafe:
                    self._retiring_transfer = None
                    self._warmup_result = None
            except BaseException:
                self._authority.mark_unsafe()
                raise

    def close(self):
        # Child process exit, not broker closure, disposes the model and leases.
        self._closed = True

