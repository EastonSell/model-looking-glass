"""Serial ownership, bounded admission, and fair dispatch for a resident worker.

This is a cooperative coordination primitive for trusted local clients, not a
process isolation, VRAM budgeting, batching, or concurrent CUDA execution API.
"""

from dataclasses import dataclass, replace
from enum import Enum
from threading import Condition, Event, Thread, current_thread
from time import perf_counter_ns
from typing import Any, Optional
from uuid import uuid4

from .fair_queue import FairTeamQueue, QueueFull


class BrokerUnavailable(RuntimeError):
    """The broker is closing or its worker can no longer accept requests."""


class BrokerShutdownError(RuntimeError):
    """The worker's close method failed."""


class RequestState(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass(frozen=True)
class Outcome:
    """A terminal status snapshot. The result itself is client-owned and mutable."""

    broker_id: str
    job_id: int
    team_id: str
    request_id: str
    state: RequestState
    submitted_ns: int
    ready_ns: int
    started_ns: Optional[int]
    finished_ns: int
    result: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class _Job:
    handle: 'RequestHandle'
    request: Any = None


class RequestHandle:
    """Observe one accepted request, request cancellation, or wait for its outcome."""

    def __init__(self, broker, job_id, team_id, submitted_ns):
        self._broker = broker
        self._job_id, self._team_id, self._submitted_ns = job_id, team_id, submitted_ns
        self._request_id = None
        self._worker_request_id = None
        self._state = RequestState.PENDING
        self._ready_ns = None
        self._started_ns = None
        self._outcome = None
        self._cancel = Event()
        self._job = None

    @property
    def job_id(self):
        return self._job_id

    @property
    def team_id(self):
        return self._team_id

    @property
    def request_id(self):
        return self._request_id

    @property
    def submitted_ns(self):
        return self._submitted_ns

    @property
    def state(self):
        with self._broker._condition:
            return self._state

    def cancel(self):
        """Request cancellation; return false if already requested or terminal.

        Pending jobs finish immediately. A running job remains running until
        the worker returns, even when cancellation has been requested.
        """
        job = None
        broker = self._broker
        with broker._condition:
            if self._state is RequestState.PENDING:
                job = self._job  # Keep payload alive until after lock release.
                if not broker._queue.cancel(self.job_id):
                    raise RuntimeError('pending request missing from broker queue')
                broker._release_slot_locked(self.team_id)
                self._job = None
                self._cancel.set()
                broker._finish_locked(self, RequestState.CANCELLED)
                accepted = True
            elif self._state is RequestState.RUNNING and not self._cancel.is_set():
                self._cancel.set()
                accepted = True
            else:
                accepted = False
        del job
        return accepted

    def wait(self, timeout=None):
        """Return a terminal Outcome, or raise TimeoutError if still in flight."""
        if timeout is not None and (isinstance(timeout, bool) or timeout < 0):
            raise ValueError('timeout must be nonnegative or None')
        with self._broker._condition:
            if not self._broker._condition.wait_for(lambda: self._outcome is not None, timeout):
                raise TimeoutError('request is still in flight')
            return self._outcome


class Broker:
    """Queue requests for exactly one trusted, preconstructed resident worker.

    The worker must supply snapshot_request, execute, poisoned, and close, as
    implemented by ResidentWorker. Clients must not otherwise access its model.
    """

    def __init__(self, worker, *, max_pending, max_pending_per_team):
        self._queue = FairTeamQueue(max_pending, max_pending_per_team)
        cancellation_exception = getattr(worker, 'cancellation_exception', None)
        if (not isinstance(cancellation_exception, type)
                or not issubclass(cancellation_exception, Exception)
                or cancellation_exception is Exception):
            raise TypeError('worker must expose a cancellation_exception type')
        self._cancellation_exception = cancellation_exception
        self._condition = Condition()
        self._worker = worker
        self._next_job_id = 0
        self._identity_prefix = uuid4().hex
        self._max_pending = max_pending
        self._max_pending_per_team = max_pending_per_team
        self._pending_slots = 0
        self._team_slots = {}
        self._preparing = 0
        self._running = None
        self._closed = False
        self._unavailable = None
        self._shutdown_error = None
        self._thread = Thread(target=self._run, name='looking-glass-worker', daemon=True)
        self._thread.start()

    @property
    def broker_id(self):
        """Unique identifier for this broker lifetime, for artifact correlation."""
        return self._identity_prefix

    @property
    def available(self):
        """Whether this broker can currently accept requests."""
        with self._condition:
            return not self._closed and not self._unavailable and not self._worker.poisoned

    def submit(self, team_id, request):
        """Submit a supported request or raise QueueFull/BrokerUnavailable.

        Snapshot validation errors propagate and do not occupy queue capacity.
        The calling client must not mutate its input during snapshotting.
        """
        submitted_ns = perf_counter_ns()
        if type(team_id) is not str or not team_id:
            raise ValueError('team_id must be a nonempty string')
        with self._condition:
            if self._closed or self._unavailable or self._worker.poisoned:
                raise BrokerUnavailable('broker is closed or worker is unavailable')
            if self._pending_slots >= self._max_pending:
                raise QueueFull('global pending-job limit reached')
            if self._team_slots.get(team_id, 0) >= self._max_pending_per_team:
                raise QueueFull('team pending-job limit reached')
            self._pending_slots += 1
            self._team_slots[team_id] = self._team_slots.get(team_id, 0) + 1
            self._preparing += 1
            self._next_job_id += 1
            handle = RequestHandle(self, self._next_job_id, team_id, submitted_ns)
        # Preparation holds a bounded slot, but not the condition. A slow CPU
        # copy cannot prevent cancellation of a different running request.
        try:
            snapshot = self._worker.snapshot_request(request)
            request_id = snapshot.request_id
            worker_request_id = f'lg-{self._identity_prefix}-{handle.job_id}'
            snapshot = replace(snapshot, request_id=worker_request_id)
        except BaseException:
            try:
                poisoned = bool(self._worker.poisoned)
            except BaseException:
                poisoned = True
            removed = []
            with self._condition:
                self._preparing -= 1
                self._release_slot_locked(team_id)
                if poisoned:
                    self._unavailable = 'worker failed while preparing request'
                    removed = self._fail_pending_locked()
                self._condition.notify_all()
            del removed
            raise
        with self._condition:
            try:
                if self._closed or self._unavailable or self._worker.poisoned:
                    raise BrokerUnavailable('broker closed while preparing request')
                handle._request_id = request_id
                handle._worker_request_id = worker_request_id
                job = _Job(handle, snapshot)
                self._queue.put(team_id, handle.job_id, job)
                handle._job = job
                handle._ready_ns = perf_counter_ns()
            except BaseException:
                self._release_slot_locked(team_id)
                raise
            finally:
                self._preparing -= 1
                self._condition.notify_all()
            return handle

    def _release_slot_locked(self, team_id):
        self._pending_slots -= 1
        self._team_slots[team_id] -= 1
        if not self._team_slots[team_id]:
            del self._team_slots[team_id]

    def _finish_locked(self, handle, state, result=None, error_type=None, error_message=None):
        handle._state = state
        handle._outcome = Outcome(
            broker_id=self.broker_id, job_id=handle.job_id, team_id=handle.team_id, request_id=handle.request_id,
            state=state, submitted_ns=handle.submitted_ns, ready_ns=handle._ready_ns,
            started_ns=handle._started_ns, finished_ns=perf_counter_ns(),
            result=result, error_type=error_type, error_message=error_message,
        )
        self._condition.notify_all()

    def _run(self):
        removed = []
        try:
            self._serve()
        except BaseException as exc:
            failure = _describe_exception(exc)
            with self._condition:
                self._unavailable = 'worker owner failed unexpectedly'
                if self._running is not None and self._running._outcome is None:
                    self._finish_locked(self._running, RequestState.FAILED,
                                        error_type=failure[0], error_message=failure[1])
                self._running = None
                removed = self._fail_pending_locked()
                self._condition.notify_all()
        finally:
            # A snapshot in progress may still use the worker. Never close it
            # until all accepted preparation reservations have been released.
            with self._condition:
                self._condition.wait_for(lambda: self._preparing == 0)
            del removed
            try:
                self._worker.close()
            except BaseException as exc:
                with self._condition:
                    self._shutdown_error = _describe_exception(exc)
                    self._condition.notify_all()

    def _serve(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: len(self._queue) or self._closed or self._unavailable)
                if self._closed or self._unavailable:
                    return
                item = self._queue.pop()
                self._release_slot_locked(item.team_id)
                job = item.payload
                handle = job.handle
                handle._job = None
                handle._state = RequestState.RUNNING
                handle._started_ns = perf_counter_ns()
                self._running = handle
            result = None
            failure = None
            cancellation = False
            fatal = False
            removed = []
            try:
                result = self._worker.execute(job.request, handle._cancel)
            except BaseException as exc:
                failure = _describe_exception(exc)
                cancellation = isinstance(exc, self._cancellation_exception)
                fatal = not isinstance(exc, Exception)
            else:
                try:
                    if type(result.request_id) is not str or result.request_id != handle._worker_request_id:
                        raise ValueError('result identity mismatch')
                    result = replace(result, request_id=handle.request_id)
                except BaseException:
                    failure = ('InvalidWorkerResult', 'worker returned an invalid request identity or result')
                    fatal = True
                    result = None
            try:
                poisoned = self._worker.poisoned
            except BaseException as exc:
                poisoned = True
                if failure is None or cancellation:
                    failure = _describe_exception(exc)
                    cancellation = False
            with self._condition:
                if failure is not None and (not cancellation or poisoned):
                    self._finish_locked(handle, RequestState.FAILED, error_type=failure[0], error_message=failure[1])
                elif cancellation or handle._cancel.is_set():
                    self._finish_locked(handle, RequestState.CANCELLED)
                else:
                    self._finish_locked(handle, RequestState.SUCCEEDED, result=result)
                self._running = None
                if poisoned or fatal:
                    self._unavailable = 'worker failed and cannot be reused'
                    removed = self._fail_pending_locked()
            # Do not retain finished inputs, exceptions, or results in the owner
            # thread while waiting for the next job.
            del item, job, handle, result, failure, removed

    def _fail_pending_locked(self):
        removed = []
        while len(self._queue):
            item = self._queue.pop()
            self._release_slot_locked(item.team_id)
            removed.append(item)
            handle = item.payload.handle
            handle._job = None
            self._finish_locked(handle, RequestState.FAILED,
                                error_type='WorkerUnavailable', error_message=self._unavailable)
        return removed

    def close(self, *, wait=True, timeout=None):
        """Close admission, cancel queued and running jobs, and optionally join.

        Return whether the owner has stopped. A timeout does not preempt work;
        callers may call close again to join once the worker has returned.
        """
        if wait and current_thread() is self._thread:
            raise RuntimeError('owner thread cannot join itself')
        removed = []
        with self._condition:
            self._closed = True
            while len(self._queue):
                item = self._queue.pop()
                self._release_slot_locked(item.team_id)
                removed.append(item)
                handle = item.payload.handle
                handle._job = None
                handle._cancel.set()
                self._finish_locked(handle, RequestState.CANCELLED)
            if self._running is not None:
                self._running._cancel.set()
            self._condition.notify_all()
        del removed
        if wait:
            self._thread.join(timeout)
        stopped = not self._thread.is_alive()
        if stopped and self._shutdown_error is not None:
            raise BrokerShutdownError(f'worker close failed: {self._shutdown_error[0]}: {self._shutdown_error[1]}')
        return stopped

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _describe_exception(exc):
    try:
        message = str(exc)
    except BaseException:
        message = '<exception message unavailable>'
    return type(exc).__name__, message
