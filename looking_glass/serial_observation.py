"""Trusted serial-broker observations for the batched-event protocol.

The wrapper requires a single owner and the existing budgeted resident worker;
it neither grants device admission nor certifies model identity or timing.
"""

import threading
import weakref

from .broker import Broker, Outcome, QueueFull, RequestHandle, RequestState
from .budgeted_worker import BudgetedWorker
from .grouped_observation import GroupedObservation, _SerialForwardScope, _local
from .resident_worker import ForwardRequest, RequestCancelled


class SerialObservation(GroupedObservation):
    """One recorder and bounded identity registry for one serial broker epoch."""

    def __init__(self, recorder):
        super().__init__(recorder)
        self._broker = None
        self._registration_lock = threading.Lock()
        self._registered = {}
        self._completed = set()
        self._capturing = set()
        self._executed = set()
        self._window_ended = False

    def record(self, event):
        # Serial terminal stamps are buffered at wait().  A producer must
        # finish copying them before the timed window ends, including when a
        # wait and window_end race on separate threads.
        try:
            if type(event) is dict and all(type(key) is str for key in event):
                kind = event.get('kind')
                if type(kind) is str and kind == 'window_end':
                    self._check_cut(end_window=True)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
        return super().record(event)

    def _check_cut(self, *, end_window):
        broker = self._broker
        if broker is None:
            with self._registration_lock:
                if end_window:
                    self._window_ended = True
            self._fault()
            return
        # Broker IDs are monotonic and never reused within this epoch.  A
        # count plus range check catches direct submissions, including jobs
        # cancelled before execution, without retaining their handles.
        with broker._condition:
            with self._registration_lock:
                if end_window:
                    self._window_ended = True
                allocated = broker._next_job_id
                if (type(allocated) is not int or allocated != len(self._registered)
                        or any(job_id < 1 or job_id > allocated for job_id in self._registered)
                        or self._capturing or self._registered.keys() != self._completed
                        or not self._executed.issubset(self._registered)):
                    self._fault()

    def _attach(self, broker):
        if self._broker is not None:
            raise ValueError('an observation belongs to one broker owner epoch')
        self._bind_broker(broker.broker_id)
        self._broker = broker

    def submit(self, team_id, request):
        """Submit through the bound broker and register its exact handle."""
        if self._broker is None:
            raise RuntimeError('serial observation is not bound')
        try:
            return self._submit(team_id, request)
        except QueueFull as exc:
            # The broker guarantees this rejection happened before acceptance.
            if type(exc) is not QueueFull:
                self._fault()
            raise
        except BaseException:
            self._fault()
            raise

    def _submit(self, team_id, request):
        if type(request) is not ForwardRequest:
            # Preserve the broker's exact-type boundary before reading any
            # attribute on a caller-provided object.
            return self._broker.submit(team_id, request)
        request_id = request.request_id
        handle = self._broker.submit(team_id, request)
        try:
            self._register(handle, team_id, request_id)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
        return handle

    def _register(self, handle, team_id, request_id):
        if (type(handle) is not RequestHandle or handle._broker is not self._broker
                or handle.team_id != team_id or handle.request_id != request_id
                or type(handle.job_id) is not int or not 0 < handle.job_id < 2 ** 63):
            raise ValueError('broker returned a mismatched handle')
        with self._registration_lock:
            if handle.job_id in self._registered or len(self._registered) >= 4096:
                raise ValueError('serial observation registration is not unique or bounded')
            if self._window_ended:
                self._fault()
            if (type(request_id) is not str or not request_id or len(request_id) > 128
                    or request_id in self._seen):
                self._fault()
            else:
                self._seen.add(request_id)
            self._registered[handle.job_id] = (weakref.ref(handle), team_id, request_id)

    def wait(self, handle, timeout=None):
        """Wait and copy the original broker-stamped events before returning."""
        if type(handle) is not RequestHandle or handle._broker is not self._broker:
            raise ValueError('handle belongs to another serial broker')
        mismatch = False
        try:
            with self._registration_lock:
                registration = self._registered.get(handle.job_id)
                mismatch = registration is not None and registration[0]() is not handle
                if registration is None:
                    # Preserve the real outcome if metadata registration failed.
                    self._fault()
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
        if mismatch:
            self._fault()
            raise ValueError('handle does not match its registered submission')
        try:
            outcome = handle.wait(timeout)
        except BaseException:
            # A timeout or ambiguous handoff is not evidence of a terminal.
            self._fault()
            raise
        try:
            self._capture(handle, outcome)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
        return outcome

    def _capture(self, handle, outcome):
        with self._broker._condition:
            if (type(outcome) is not Outcome or handle._outcome is not outcome
                    or outcome.broker_id != self._broker_id or outcome.job_id != handle.job_id
                    or outcome.team_id != handle.team_id or outcome.request_id != handle.request_id
                    or outcome.ready_ns != handle._ready_ns or outcome.started_ns != handle._started_ns
                    or outcome.state not in (RequestState.SUCCEEDED, RequestState.FAILED, RequestState.CANCELLED)):
                raise ValueError('terminal outcome is not the exact broker-issued record')
            job_id, team_id, request_id = handle.job_id, handle.team_id, handle.request_id
            ready_ns, selected_ns, terminal_ns = outcome.ready_ns, outcome.started_ns, outcome.finished_ns
            state = outcome.state.value
        if (type(ready_ns) is not int or type(terminal_ns) is not int
                or not 0 <= ready_ns <= terminal_ns < 2 ** 63
                or (selected_ns is not None and
                    (type(selected_ns) is not int or not ready_ns <= selected_ns <= terminal_ns))):
            raise ValueError('broker source stamps are incomplete or out of order')
        with self._registration_lock:
            registration = self._registered.get(job_id)
            if registration is None or registration[0]() is not handle or registration[1:] != (team_id, request_id):
                raise ValueError('handle does not match its registered submission')
            if job_id in self._completed:
                return
            if job_id in self._capturing:
                # Concurrent waits may return the real outcome, but a timed
                # report cannot claim its metadata was copied before both.
                self._fault()
                return
            self._capturing.add(job_id)
        self.record({'kind': 'ready', 'time_ns': ready_ns, 'job_id': request_id,
                     'team_id': team_id, 'broker_id': self._broker_id, 'broker_job_id': job_id})
        if selected_ns is not None:
            self.record({'kind': 'selected', 'time_ns': selected_ns, 'job_id': request_id})
        self.record({'kind': 'broker_terminal', 'time_ns': terminal_ns, 'job_id': request_id, 'state': state})
        with self._registration_lock:
            if self._window_ended:
                self._fault()
            self._capturing.remove(job_id)
            self._completed.add(job_id)

    def _note_execution(self, job_id):
        try:
            with self._registration_lock:
                if job_id in self._executed or len(self._executed) >= 4096:
                    raise ValueError('serial execution identity is repeated or unbounded')
                if self._window_ended:
                    self._fault()
                self._executed.add(job_id)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise

    def _start_serial(self, identity):
        scope = None
        try:
            if getattr(_local, 'forward', None) is not None:
                raise RuntimeError('nested forward observation is unsupported')
            scope = _SerialForwardScope(self, *identity)
            _local.forward = scope
            return scope
        except BaseException as exc:
            self._fault()
            if scope is not None:
                self._abandon_forward(scope)
            if not isinstance(exc, Exception):
                raise
            return None

    def _settle_serial(self, scope, *, failed):
        if scope is None:
            if failed:
                self._abandon_forward(None)
            return
        try:
            if scope.entry_ns is None:
                if not failed:
                    self._fault()
                if getattr(_local, 'forward', None) is not scope:
                    raise RuntimeError('serial observation context changed')
                _local.forward = None
                return
            self._finish_forward(scope)
        except BaseException as exc:
            self._fault()
            self._abandon_forward(scope)
            if not isinstance(exc, Exception):
                raise

    def seal(self, *, producers_stopped):
        if producers_stopped is not True:
            raise ValueError('all event producers must be stopped before sealing')
        try:
            self._check_cut(end_window=False)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise
        return super().seal(producers_stopped=producers_stopped)


class SerialObservedWorker:
    """Owner-private broker proxy around an exact budgeted resident worker."""

    cancellation_exception = RequestCancelled

    def __init__(self, worker, observation):
        if type(worker) is not BudgetedWorker or type(observation) is not SerialObservation:
            raise ValueError('an exact budgeted worker and serial observation are required')
        self._worker, self._observation, self._broker = worker, observation, None

    def bind_broker(self, broker):
        if type(broker) is not Broker:
            raise ValueError('an exact serial broker is required')
        with broker._condition:
            if (self._broker is not None or broker._worker is not self or broker._next_job_id != 0
                    or broker._pending_slots or broker._running is not None or broker._closed or broker.quarantined):
                raise ValueError('bind before any broker submission and keep exclusive ownership')
            self._observation._attach(broker)
            self._broker = broker

    @property
    def poisoned(self):
        return self._worker.poisoned

    @property
    def quarantined(self):
        return self._worker.quarantined

    @property
    def failed(self):
        return self._worker.failed

    @property
    def max_input_bytes(self):
        return self._worker.max_input_bytes

    @property
    def max_output_bytes(self):
        return self._worker.max_output_bytes

    def snapshot_request(self, request):
        return self._worker.snapshot_request(request)

    def _identity(self, request):
        broker = self._broker
        if broker is None or threading.current_thread() is not broker._thread:
            raise RuntimeError('only the bound broker owner can execute this worker')
        with broker._condition:
            handle = broker._running
            if (broker._worker is not self or type(handle) is not RequestHandle
                    or handle._state is not RequestState.RUNNING
                    or handle._worker_request_id != request.request_id):
                raise RuntimeError('running broker identity does not match the worker request')
            identity = (handle.job_id, handle.team_id, handle.request_id, handle._worker_request_id)
            handle = None
        return identity

    def execute(self, request, cancel_event=None):
        if self._broker is None or threading.current_thread() is not self._broker._thread:
            self._observation._fault()
            raise RuntimeError('only the bound broker owner can execute this worker')
        scope = None
        try:
            try:
                identity = self._identity(request)
            except Exception:
                # A read-only instrumentation failure cannot change the
                # broker's actual request or its worker outcome.
                self._observation._abandon_forward(None)
                return self._worker.execute(request, cancel_event)
            self._observation._note_execution(identity[0])
            scope = self._observation._start_serial(identity)
            result = self._worker.execute(request, cancel_event)
        except BaseException:
            self._observation._settle_serial(scope, failed=True)
            raise
        self._observation._settle_serial(scope, failed=False)
        return result

    def close(self):
        return self._worker.close()
