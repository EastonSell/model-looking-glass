"""Trusted, opt-in observations of the grouped execution path.

The recorder stores metadata only.  A caller must stop and join all producers
before accepting a report; no observation is a device-safety certificate.
"""

import re
import threading
from time import perf_counter_ns


_local = threading.local()


class GroupedObservation:
    """One bounded recorder for one trusted grouped-broker owner epoch."""

    def __init__(self, recorder):
        from .batched_recorder import BatchedEventRecorder
        if type(recorder) is not BatchedEventRecorder:
            raise ValueError('an exact service-owned BatchedEventRecorder is required')
        self._recorder = recorder
        self._faulted = False
        self._broker_id = None
        self._seen = set()

    def _bind_broker(self, broker_id):
        if self._broker_id is not None:
            raise ValueError('an observation belongs to one broker owner epoch')
        self._broker_id = broker_id

    def _fault(self):
        # This latch is independent of the recorder's potentially damaged
        # lock.  A supervisor must still reject a stranded producer thread.
        self._faulted = True

    def _stamp(self):
        try:
            value = perf_counter_ns()
            if type(value) is not int or not 0 <= value < 2 ** 63:
                raise ValueError('invalid source timestamp')
            return value
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
            return None

    def record(self, event):
        """Record a driver observation without allowing recorder errors to escape."""
        try:
            accepted = self._recorder.record(event)
            if accepted is not True:
                self._fault()
                return False
            return True
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
            return False

    def _ready_stamp(self, job):
        try:
            # Event V1 uses request_id as the driver's globally unique logical
            # job ID.  Duplicate IDs are legal for the broker, but cannot form
            # valid evidence under that schema.
            if (type(job.request_id) is not str or not job.request_id
                    or len(job.request_id) > 128 or job.request_id in self._seen
                    or len(self._seen) >= 4096):
                raise ValueError('logical job identity is not unique and bounded')
            self._seen.add(job.request_id)
            return self._stamp()
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
            return None

    def _ready(self, job, stamp):
        try:
            if stamp is not None:
                self.record({'kind': 'ready', 'time_ns': stamp, 'job_id': job.request_id,
                             'team_id': job.team_id, 'broker_id': self._broker_id,
                             'broker_job_id': job.job_id})
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise

    def _selected(self, jobs, stamp):
        try:
            if stamp is not None:
                for job in jobs:
                    self.record({'kind': 'selected', 'time_ns': stamp, 'job_id': job.request_id})
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise

    def _terminal(self, job, status, stamp):
        try:
            state = {'ok': 'succeeded', 'failed': 'failed', 'cancelled': 'cancelled'}[status]
            if stamp is not None:
                self.record({'kind': 'broker_terminal', 'time_ns': stamp,
                             'job_id': job.request_id, 'state': state})
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise

    def _publication(self, group, publication):
        try:
            if (len(group.jobs) != len(publication.members)
                    or any(job.job_id != result.member.job_id
                           for job, result in zip(group.jobs, publication.members))):
                raise ValueError('published membership differs from the selected cohort')
            stamp = group.state._published_ns
            for job, result in zip(group.jobs, publication.members):
                self._terminal(job, result.status, stamp)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise

    def _start_forward(self, group, launch):
        scope = None
        try:
            if getattr(_local, 'forward', None) is not None:
                raise RuntimeError('nested forward observation is unsupported')
            scope = _ForwardScope(self, group, launch)
            _local.forward = scope
            return scope
        except BaseException as exc:
            self._fault()
            if scope is not None:
                self._abandon_forward(scope)
            if not isinstance(exc, Exception):
                raise
            return None

    def _finish_forward(self, scope):
        if scope is None:
            return
        try:
            if getattr(_local, 'forward', None) is not scope:
                raise RuntimeError('forward observation context changed')
            _local.forward = None
            scope.emit()
        except BaseException as exc:
            self._fault()
            self._abandon_forward(scope)
            if not isinstance(exc, Exception):
                raise

    def _abandon_forward(self, scope):
        self._fault()
        try:
            current = getattr(_local, 'forward', None)
            if current is scope or (scope is None and type(current) is _ForwardScope and current.owner is self):
                _local.forward = None
        except BaseException:
            self._fault()

    def seal(self, *, producers_stopped):
        """Return a detached report after every producer was stopped and joined."""
        self._recorder.seal(producers_stopped=producers_stopped)
        report = self._recorder.report()
        if self._faulted:
            report['events_complete'] = False
        return report


class _ForwardScope:
    def __init__(self, owner, group, launch):
        jobs = group.jobs
        if (not jobs or tuple(job.job_id for job in jobs) != launch.job_ids
                or group.group_id != group.plan.group_id
                or group.plan._profile.input_shape[0] != 1):
            raise ValueError('unsupported or mismatched launched cohort')
        self.owner = owner
        self.group_id = group.group_id
        self.call_id = launch.launch_id
        self.job_ids = tuple(job.job_id for job in jobs)
        self.members = [{'job_id': job.request_id, 'team_id': job.team_id,
                         'broker_id': owner._broker_id, 'broker_job_id': job.job_id,
                         'row_offset': index} for index, job in enumerate(jobs)]
        self.profile_id = self.shape = self.worker_id = None
        self.schema = self.device = None
        self.entry_ns = self.finish_ns = None
        self.returned = False
        self.state = self.error = None
        self.synchronized = False

    def bind(self, worker, profile, snapshot, launch):
        if (self.profile_id is not None or launch.job_ids != self.job_ids
                or launch.launch_id != self.call_id or snapshot.request_id != self.group_id
                or type(profile.profile_id) is not str
                or re.fullmatch(r'[0-9a-f]{64}', profile.profile_id) is None
                or tuple(snapshot.inputs['input_ids'].shape) != profile.shape
                or profile.shape[0] != len(self.members)):
            raise ValueError('selected whole-batch workload does not match the launch')
        self.profile_id = profile.profile_id
        self.shape = tuple(profile.shape)
        self.schema = tuple(profile.input_dtypes)
        self.device = profile.device
        self.worker_id = id(worker)

    def enter(self, worker, request, device_inputs):
        if (self.entry_ns is not None or self.profile_id is None or id(worker) != self.worker_id
                or request.request_id != self.group_id
                or tuple(request.inputs['input_ids'].shape) != self.shape
                or tuple(sorted((name, str(tensor.dtype)) for name, tensor in device_inputs.items())) != self.schema
                or any(tuple(tensor.shape) != self.shape or str(tensor.device) != self.device
                       for tensor in device_inputs.values())):
            raise ValueError('actual resident forward does not match the admitted launch')
        self.entry_ns = self.owner._stamp()

    def finish(self, synchronized):
        if self.entry_ns is not None:
            self.finish_ns = self.owner._stamp()
            self.synchronized = synchronized

    def result(self, failure, synchronization_failed, interrupted):
        if self.entry_ns is None:
            return
        if synchronization_failed or interrupted or not self.returned or self.finish_ns is None:
            self.state, self.error = 'uncertain', 'execution_or_synchronization_uncertain'
        elif failure is not None:
            self.state, self.error = 'failed', 'resident_forward_failed'
        else:
            self.state, self.error = 'succeeded', None

    def emit(self):
        if self.entry_ns is None:
            self.owner._fault()
            return
        if self.profile_id is None or self.state is None:
            self.owner._fault()
            return
        self.owner.record({'kind': 'forward_entry', 'time_ns': self.entry_ns,
                           'call_id': self.call_id, 'group_id': self.group_id,
                           'profile_sha256': self.profile_id, 'input_shape': list(self.shape),
                           'members': self.members})
        if self.state == 'uncertain':
            stamp = self.finish_ns if self.finish_ns is not None else self.owner._stamp()
            if stamp is not None:
                self.owner.record({'kind': 'forward_uncertain', 'time_ns': stamp,
                                   'call_id': self.call_id, 'error': self.error})
        else:
            self.owner.record({'kind': 'forward_finished', 'time_ns': self.finish_ns,
                               'call_id': self.call_id, 'state': self.state,
                               'synchronized': self.synchronized, 'error': self.error})


def _invoke(method, *args):
    """Update thread-local scalar metadata without exposing observer failures."""
    scope = None
    try:
        scope = getattr(_local, 'forward', None)
        if scope is not None:
            getattr(scope, method)(*args)
    except BaseException as exc:
        if scope is not None:
            scope.owner._fault()
        if not isinstance(exc, Exception):
            raise


def _bind_forward(worker, profile, snapshot, launch):
    _invoke('bind', worker, profile, snapshot, launch)


def _enter_forward(worker, request, device_inputs):
    _invoke('enter', worker, request, device_inputs)


def _returned_forward():
    scope = None
    try:
        scope = getattr(_local, 'forward', None)
        if scope is not None:
            scope.returned = True
    except BaseException as exc:
        if scope is not None:
            scope.owner._fault()
        if not isinstance(exc, Exception):
            raise


def _finish_forward(synchronized):
    _invoke('finish', synchronized)


def _result_forward(failure, synchronization_failed, interrupted):
    _invoke('result', failure, synchronization_failed, interrupted)
