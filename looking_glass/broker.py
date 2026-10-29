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
            if broker._fatal_cleanup_failure and self._outcome is None:
                raise BrokerUnavailable('request status is unknown after broker cleanup failure')
            if self._state is RequestState.PENDING:
                job = self._job  # Keep payload alive until after lock release.
                try:
                    if not broker._queue.cancel(self.job_id):
                        raise RuntimeError('pending request missing from broker queue')
                    self._cancel.set()
                    broker._finish_locked(self, RequestState.CANCELLED)
                    self._job = None
                    broker._release_slot_locked(self.team_id)
                except BaseException:
                    broker._mark_cleanup_failed_locked()
                    raise BrokerUnavailable('request cancellation cleanup failed') from None
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
            if not self._broker._condition.wait_for(
                    lambda: self._outcome is not None or self._broker._fatal_cleanup_failure, timeout):
                raise TimeoutError('request is still in flight')
            if self._outcome is None:
                raise BrokerUnavailable('request status is unknown after broker cleanup failure')
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
        self._fatal_cleanup_failure = False
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

    @property
    def failed(self):
        """Whether a broker failure occurred, distinct from orderly closure."""
        with self._condition:
            return bool(self._unavailable or self._fatal_cleanup_failure)

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
            next_pending = self._pending_slots + 1
            next_team = self._team_slots.get(team_id, 0) + 1
            next_preparing = self._preparing + 1
            next_job_id = self._next_job_id + 1
            handle = RequestHandle(self, next_job_id, team_id, submitted_ns)
            self._team_slots[team_id] = next_team
            self._pending_slots = next_pending
            self._preparing = next_preparing
            self._next_job_id = next_job_id
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
                try:
                    self._release_slot_locked(team_id)
                except BaseException:
                    self._mark_cleanup_failed_locked()
                    raise BrokerUnavailable('snapshot cleanup failed; accounting is unknown') from None
                if poisoned:
                    self._unavailable = 'worker failed while preparing request'
                    try:
                        removed = self._fail_pending_locked()
                    except BaseException:
                        self._mark_cleanup_failed_locked()
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
                handle._job = job
                handle._ready_ns = perf_counter_ns()
                try:
                    self._queue.put(team_id, handle.job_id, job)
                except BaseException:
                    # If queue publication itself fails, do not assume its
                    # rollback completed under host-memory pressure.
                    self._mark_cleanup_failed_locked()
                    raise BrokerUnavailable('queue publication failed; accounting is unknown') from None
            except BaseException:
                if not self._fatal_cleanup_failure:
                    try:
                        self._release_slot_locked(team_id)
                    except BaseException:
                        self._mark_cleanup_failed_locked()
                        raise BrokerUnavailable('preparation cleanup failed') from None
                raise
            finally:
                self._preparing -= 1
                self._condition.notify_all()
            return handle

    def _release_slot_locked(self, team_id):
        team_count = self._team_slots.get(team_id, 0)
        if self._pending_slots <= 0 or team_count <= 0:
            raise RuntimeError('invalid pending-slot accounting')
        pending = self._pending_slots - 1
        remaining = team_count - 1
        if remaining:
            self._team_slots[team_id] = remaining
        else:
            del self._team_slots[team_id]
        self._pending_slots = pending

    def _mark_cleanup_failed_locked(self):
        # Do not discard pending requests or decrement unverified reservations.
        self._fatal_cleanup_failure = True
        self._unavailable = 'broker cleanup failed; request status is unknown'
        self._condition.notify_all()

    def _finish_locked(self, handle, state, result=None, error_type=None, error_message=None):
        outcome = Outcome(
            broker_id=self.broker_id, job_id=handle.job_id, team_id=handle.team_id, request_id=handle.request_id,
            state=state, submitted_ns=handle.submitted_ns, ready_ns=handle._ready_ns,
            started_ns=handle._started_ns, finished_ns=perf_counter_ns(),
            result=result, error_type=error_type, error_message=error_message,
        )
        handle._state = state
        handle._outcome = outcome
        self._condition.notify_all()

    def _run(self):
        removed = []
        try:
            self._serve()
        except BaseException as exc:
            failure = _describe_exception(exc)
            with self._condition:
                self._unavailable = 'worker owner failed unexpectedly'
                try:
                    if self._running is not None and self._running._outcome is None:
                        self._finish_locked(self._running, RequestState.FAILED,
                                            error_type=failure[0], error_message=failure[1])
                        self._running._job = None
                    self._running = None
                    removed = self._fail_pending_locked()
                except BaseException:
                    self._mark_cleanup_failed_locked()
                finally:
                    # An unexpected owner failure can occur between dequeue,
                    # accounting, and terminal publication. Even if draining
                    # appears to work, do not claim the accounting is complete.
                    self._mark_cleanup_failed_locked()
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
                job = item.payload
                handle = job.handle
                self._running = handle
                self._release_slot_locked(item.team_id)
                handle._job = None
                handle._state = RequestState.RUNNING
                handle._started_ns = perf_counter_ns()
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
            removed.append(item)
            handle = item.payload.handle
            self._finish_locked(handle, RequestState.FAILED,
                                error_type='WorkerUnavailable', error_message=self._unavailable)
            handle._job = None
            self._release_slot_locked(item.team_id)
        return removed

    def close(self, *, wait=True, timeout=None):
        """Close admission, cancel queued and running jobs, and optionally join.

        Return whether the owner has stopped. A timeout does not preempt work;
        callers may call close again to join once the worker has returned.
        """
        if wait and current_thread() is self._thread:
            raise RuntimeError('owner thread cannot join itself')
        removed = []
        cleanup_failed = False
        with self._condition:
            self._closed = True
            try:
                while len(self._queue) and not self._fatal_cleanup_failure:
                    item = self._queue.pop()
                    removed.append(item)
                    handle = item.payload.handle
                    handle._cancel.set()
                    self._finish_locked(handle, RequestState.CANCELLED)
                    handle._job = None
                    self._release_slot_locked(item.team_id)
            except BaseException:
                self._mark_cleanup_failed_locked()
                cleanup_failed = True
            if self._running is not None:
                self._running._cancel.set()
            self._condition.notify_all()
        del removed
        if cleanup_failed:
            raise BrokerShutdownError('broker could not complete pending-request cleanup')
        if wait:
            self._thread.join(timeout)
        stopped = not self._thread.is_alive()
        if self._fatal_cleanup_failure:
            raise BrokerShutdownError('broker cleanup failed; request status and accounting are unknown')
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
