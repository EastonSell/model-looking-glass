"""Bounded, owner-thread source stamps for a prospective grouped measurement.

The caller must bind the exact source, resident model, broker, physical owner,
and cleanup policy separately. This module makes no device call and does not
enable GroupedObservation. Profiling adds overhead throughout the instrumented
owner-thread interval. A Python return event can also mean exception unwinding;
the finish hook is only an observation of a source call and its argument. GPU
completion requires the separately verified resident source and valid outcome.
Successful records contain scalar facts only. Escaping exceptions and their
tracebacks may retain frames or tensors; the caller must retain their existing
owner, lease and failure roots until supervised recovery.
"""

from contextlib import contextmanager
import copy
import math
import sys
import threading
from time import perf_counter_ns
from types import CodeType

from .grouped_broker import GroupedBroker, _Group, _Job
from .grouped_observation import _finish_forward


class SourceStamps:
    """Record scalar facts; a failure traceback can retain live frames."""

    def __init__(self, model, *, broker=None, max_model_calls=16, max_claims=16):
        code = getattr(getattr(type(model), 'forward', None), '__code__', None)
        if (type(code) is not CodeType or (broker is not None and type(broker) is not GroupedBroker)
                or type(max_model_calls) is not int or not 1 <= max_model_calls <= 64
                or type(max_claims) is not int or not 1 <= max_claims <= 64):
            raise ValueError('a fixed Python model and bounded source observer are required')
        self._model_id = id(model)
        self._broker_id = None if broker is None else id(broker)
        self._model_code = code
        self._claim_code = GroupedBroker._claim.__code__
        self._finish_code = _finish_forward.__code__
        self._limits = (max_model_calls, max_claims)
        self._next = 'serial'
        self._owner_thread = threading.get_ident()
        self._windows = {}
        self._active = None
        self._installed = None
        self._pending_model = None
        self._model_frame_id = None
        self._fault = False

    @staticmethod
    def _name(value):
        if type(value) is not str or not value or len(value) > 128:
            raise RuntimeError('invalid source-observed job identity')
        return value

    def _claim(self, claimed):
        if type(claimed) is _Group:
            kind, group_id, jobs = 'group', self._name(claimed.group_id), claimed.jobs
            if type(jobs) is not tuple or len(jobs) != 2:
                raise RuntimeError('invalid source-observed group membership')
        elif type(claimed) is _Job:
            kind, group_id, jobs = 'single', None, (claimed,)
        else:
            raise RuntimeError('invalid source-observed broker claim')
        members = []
        for index, job in enumerate(jobs):
            if type(job) is not _Job or type(job.ready_at) is not float or not math.isfinite(job.ready_at):
                raise RuntimeError('invalid source-observed ready time')
            members.append({'broker_job_id': self._name(job.job_id), 'team_id': self._name(job.team_id),
                            'request_id': self._name(job.request_id), 'row_index': index,
                            'ready_at_hex': job.ready_at.hex()})
        return {'claim_kind': kind, 'group_id': group_id, 'members': members,
                'claim_return_event_ns': perf_counter_ns()}

    def _profile(self, frame, event, arg):
        try:
            if threading.get_ident() != self._owner_thread or self._active is None:
                raise RuntimeError('source observer ran outside its owner thread')
            if frame.f_code is self._claim_code and event == 'return' and arg is not None:
                if self._broker_id is not None and id(frame.f_locals.get('self')) == self._broker_id:
                    if len(self._active['claims']) >= self._limits[1]:
                        raise RuntimeError('source claim stamp bound exceeded')
                    self._active['claims'].append(self._claim(arg))
            elif frame.f_code is self._model_code and id(frame.f_locals.get('self')) == self._model_id:
                if event == 'call':
                    if self._pending_model is not None or len(self._active['model_calls']) >= self._limits[0]:
                        raise RuntimeError('source model stamp bound or ordering exceeded')
                    value = frame.f_locals.get('input_ids')
                    shape = list(value.shape) if value is not None else None
                    if shape not in ([1, 4], [2, 4]) or any(type(item) is not int for item in shape):
                        raise RuntimeError('source model input shape is outside the fixed workload')
                    record = {'shape': shape, 'entry_ns': perf_counter_ns(),
                              'python_return_event_ns': None, 'python_return_event_is_success': None,
                              'finish_hook_ns': None, 'finish_hook_synchronized_argument': None}
                    self._active['model_calls'].append(record)
                    self._pending_model = record
                    self._model_frame_id = id(frame)
                elif event == 'return' and id(frame) == self._model_frame_id:
                    self._pending_model['python_return_event_ns'] = perf_counter_ns()
                    self._model_frame_id = None
            elif frame.f_code is self._finish_code and event == 'call':
                if (self._pending_model is None or self._pending_model['python_return_event_ns'] is None
                        or frame.f_locals.get('synchronized') is not True):
                    raise RuntimeError('source finish hook is unexpected or unsynchronized')
                self._pending_model['finish_hook_ns'] = perf_counter_ns()
                self._pending_model['finish_hook_synchronized_argument'] = True
                self._pending_model = None
        except BaseException:
            self._fault = True
            raise

    @contextmanager
    def capture(self, mode):
        """Capture one source interval in serial then grouped order.

        Normal block return and an intact profile hook establish capture only;
        the caller must reconcile counts, outcomes, timestamps, and retained
        records before treating the measurement as complete.
        """
        if (type(mode) is not str or mode not in ('serial', 'grouped') or self._next is None
                or threading.get_ident() != self._owner_thread or self._active is not None
                or self._fault or mode != self._next or sys.getprofile() is not None):
            raise RuntimeError('source profile is unavailable or out of order')
        record = {'capture_started_ns': perf_counter_ns(), 'capture_finished_ns': None,
                  'capture_completed': False, 'model_calls': [], 'claims': []}
        self._windows[mode] = record
        self._active = record
        callback = self._profile
        self._installed = callback
        failed = True
        intact = False
        try:
            sys.setprofile(callback)
            yield
            intact = sys.getprofile() is callback
            if not intact or self._fault:
                raise RuntimeError('source profile was lost during capture')
            if self._pending_model is not None:
                raise RuntimeError('source finish hook was not observed')
            failed = False
        finally:
            sys.setprofile(None)
            record['capture_finished_ns'] = perf_counter_ns()
            record['capture_completed'] = not failed and intact and not self._fault
            self._active = None
            self._installed = None
            if record['capture_completed']:
                self._next = 'grouped' if mode == 'serial' else None
            else:
                self._fault = True

    def current_counts(self):
        """Return scalar event cursors only while the exact profile is live."""
        if (self._active is None or threading.get_ident() != self._owner_thread
                or self._fault or sys.getprofile() is not self._installed):
            if self._active is not None:
                self._fault = True
            raise RuntimeError('source profile is not active and intact')
        return len(self._active['model_calls']), len(self._active['claims'])

    def current_model_shapes(self, start):
        """Copy only observed shape scalars after a checked event cursor."""
        count, _ = self.current_counts()
        if type(start) is not int or not 0 <= start <= count:
            raise ValueError('invalid source model event cursor')
        return [list(item['shape']) for item in self._active['model_calls'][start:]]

    def snapshot(self):
        if self._active is not None:
            raise RuntimeError('source profile capture is still active')
        return {'schema': 'grouped-throughput-source-stamps-v1', 'windows': copy.deepcopy(self._windows),
                'limitations': ['owner-thread Python profiling adds measurement overhead',
                                'claim return follows selection and is not the queue-pop instant',
                                'model return events can also occur during exception unwinding',
                                'finish-hook argument requires pinned resident source and valid owner outcome']}
