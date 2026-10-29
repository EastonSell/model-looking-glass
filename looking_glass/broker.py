"""Serial ownership, bounded admission, and fair dispatch for a resident worker.

This is a cooperative coordination primitive for trusted local clients, not a
process isolation, VRAM budgeting, batching, or concurrent CUDA execution API.
"""

from dataclasses import dataclass, replace
from enum import Enum
from threading import Condition, Event, Lock, Thread, current_thread
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any, Optional
from uuid import uuid4

import torch

from .fair_queue import FairTeamQueue, QueueFull
from .resident_worker import ForwardRequest, ForwardResult, InvalidRequest


# Register live owners before work starts. A quarantined owner remains strongly
# referenced until process exit even when clients drop their Broker references.
_owner_registry = set()
_owner_registry_lock = Lock()
_mapping_proxy_type = type(MappingProxyType({}))
_input_names = frozenset(('input_ids', 'attention_mask', 'token_type_ids', 'position_ids'))


def _validate_forward_snapshot(request):
    """Enforce the shared forward-only boundary for requests and snapshots.

    Worker-specific tensor bounds and defensive copying remain the worker's
    responsibility. Trace-bearing request types need a separate custody API.
    """
    if type(request) is not ForwardRequest:
        raise InvalidRequest('broker requires an exact ForwardRequest')
    if type(request.request_id) is not str or not request.request_id or len(request.request_id) > 256:
        raise InvalidRequest('request_id must be a nonempty string of at most 256 characters')
    if type(request.seed) is not int or not 0 <= request.seed < 2**63:
        raise InvalidRequest('seed must be an integer in [0, 2**63)')
    if type(request.inputs) not in (dict, _mapping_proxy_type):
        raise InvalidRequest('inputs must be a plain tensor dictionary')
    if len(request.inputs) > len(_input_names):
        raise InvalidRequest('too many input names')
    if (any(type(name) is not str for name in request.inputs)
            or 'input_ids' not in request.inputs or not set(request.inputs).issubset(_input_names)):
        raise InvalidRequest('unsupported input names')
    shape = None
    total_bytes = 0
    for name, tensor in request.inputs.items():
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                or tensor.layout != torch.strided or tensor.ndim != 2):
            raise InvalidRequest('inputs must be plain dense two-dimensional CPU tensors')
        allowed = (torch.int32, torch.int64, torch.bool) if name == 'attention_mask' else (torch.int32, torch.int64)
        if tensor.dtype not in allowed:
            raise InvalidRequest('input tensor dtype is unsupported')
        if shape is None:
            shape = tuple(tensor.shape)
        elif shape != tuple(tensor.shape):
            raise InvalidRequest('input tensor shapes differ')
        total_bytes += tensor.numel() * tensor.element_size()
    if not (0 < shape[0] <= 256 and 0 < shape[1] <= 8192) or total_bytes > 64 * 1024 * 1024:
        raise InvalidRequest('input shape or bytes exceed the broker ceiling')
    return request


def _validate_forward_result(result, worker_request_id):
    """Accept only the forward result schema and its matching internal ID."""
    if type(result) is not ForwardResult:
        raise ValueError('broker requires an exact ForwardResult')
    if type(result.request_id) is not str or result.request_id != worker_request_id:
        raise ValueError('result identity mismatch')
    if (type(result.logits) is not torch.Tensor or result.logits.device.type != 'cpu'
            or result.logits.requires_grad or result.logits.is_inference()):
        raise ValueError('logits must be a mutable detached plain CPU tensor')
    return result


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
            if (broker._fatal_cleanup_failure or broker.quarantined) and self._outcome is None:
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
                    lambda: self._outcome is not None or self._broker._fatal_cleanup_failure
                    or self._broker.quarantined, timeout):
                raise TimeoutError('request is still in flight')
            if self._outcome is None:
                raise BrokerUnavailable('request status is unknown after broker cleanup failure')
            return self._outcome


class Broker:
    """Queue requests for exactly one trusted, preconstructed resident worker.

    The worker must supply snapshot_request, execute, poisoned, and close, as
    implemented by ResidentWorker. Requests/snapshots and results use the exact
    forward-only types. Clients must not otherwise access the worker's model.
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
        self._quarantine_latched = False
        self._startup_decided = False
        # At most max_pending preparations and one execution can concurrently
        # return unknown objects. Reserve one more slot for an owner exception
        # whose traceback can retain another in-flight object.
        self._quarantine_empty = object()
        self._quarantine_objects = [self._quarantine_empty] * (max_pending + 2)
        self._thread = Thread(target=self._run, name='looking-glass-worker', daemon=True)
        with _owner_registry_lock:
            _owner_registry.add(self)
        try:
            self._thread.start()
        except BaseException as exc:
            # Thread.start can start the owner and then raise. The caller has no
            # usable handle, so signal shutdown but retain the owner/worker and
            # require process teardown instead of assuming startup did not run.
            self._retain_unknown(exc)
            with self._condition:
                self._closed = True
            raise BrokerUnavailable('owner startup outcome is unknown; terminate the owner process') from None
        finally:
            with self._condition:
                self._startup_decided = True
                self._condition.notify_all()

    @property
    def broker_id(self):
        """Unique identifier for this broker lifetime, for artifact correlation."""
        return self._identity_prefix

    @property
    def available(self):
        """Whether this broker can currently accept requests."""
        with self._condition:
            return (not self._closed and not self._unavailable and not self.quarantined
                    and not self._worker.poisoned)

    @property
    def failed(self):
        """Whether a broker failure occurred, distinct from orderly closure."""
        with self._condition:
            return bool(self._unavailable or self._fatal_cleanup_failure or self.quarantined)

    @property
    def quarantined(self):
        """Unknown worker-owned data requires terminating the owner process."""
        return self._quarantine_latched

    def _retain_unknown(self, value):
        # Retain before signaling. A failed notification must not release the
        # object; quarantine remains independently visible to admission/close.
        with self._condition:
            try:
                try:
                    self._store_quarantine_value_locked(value)
                except BaseException:
                    # Retry once while this frame still owns the value. The
                    # identity check avoids consuming two slots if interruption
                    # occurred just after the first assignment.
                    self._store_quarantine_value_locked(value)
                    raise
            finally:
                self._quarantine_latched = True
                self._mark_cleanup_failed_locked()

    def _store_quarantine_value_locked(self, value):
        # No counter: interruption after assignment cannot let another
        # preparation overwrite an already-retained object.
        if any(existing is value for existing in self._quarantine_objects):
            return
        for index, existing in enumerate(self._quarantine_objects):
            if existing is self._quarantine_empty:
                self._quarantine_objects[index] = value
                return
        raise RuntimeError('quarantine retention slots exhausted')

    def submit(self, team_id, request):
        """Submit a supported request or raise QueueFull/BrokerUnavailable.

        Snapshot validation errors propagate and do not occupy queue capacity.
        The calling client must not mutate its input during snapshotting.
        """
        submitted_ns = perf_counter_ns()
        if type(team_id) is not str or not team_id:
            raise ValueError('team_id must be a nonempty string')
        _validate_forward_snapshot(request)
        request_id = request.request_id
        with self._condition:
            if self._closed or self._unavailable or self.quarantined or self._worker.poisoned:
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
            try:
                snapshot = self._worker.snapshot_request(request)
            except BaseException as exc:
                if not isinstance(exc, Exception) or isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                    self._retain_unknown(exc)
                raise
            try:
                _validate_forward_snapshot(snapshot)
                if snapshot.request_id != request_id:
                    raise InvalidRequest('worker snapshot changed the caller request_id')
                worker_request_id = f'lg-{self._identity_prefix}-{handle.job_id}'
                snapshot = replace(snapshot, request_id=worker_request_id)
            except BaseException:
                self._retain_unknown(snapshot)
                raise
        except BaseException:
            try:
                poisoned = bool(self._worker.poisoned)
            except BaseException:
                poisoned = True
            removed = []
            with self._condition:
                self._preparing -= 1
                if self._fatal_cleanup_failure or self.quarantined:
                    # Another thread already found accounting uncertain. Do not
                    # release any reservation or drain the queue on that basis.
                    self._condition.notify_all()
                    raise BrokerUnavailable('snapshot status is unknown after broker cleanup failure') from None
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
                if self._closed or self._unavailable or self.quarantined or self._worker.poisoned:
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
                if not self._fatal_cleanup_failure and not self.quarantined:
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
        self._quarantine_latched = True
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
            # Do not close/unregister before the constructor knows whether
            # Thread.start returned, including a start-then-raise interruption.
            with self._condition:
                self._condition.wait_for(lambda: self._startup_decided)
            self._serve()
        except BaseException as exc:
            self._retain_unknown(exc)
            failure = _describe_exception(exc)
            with self._condition:
                self._unavailable = 'worker owner failed unexpectedly'
                try:
                    if self._running is not None and self._running._outcome is None:
                        self._finish_locked(self._running, RequestState.FAILED,
                                            error_type=failure[0], error_message=failure[1])
                        self._running._job = None
                    self._running = None
                    if not self._fatal_cleanup_failure and not self.quarantined:
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
            if not self.quarantined:
                try:
                    self._worker.close()
                except BaseException as exc:
                    self._retain_unknown(exc)
                    with self._condition:
                        self._shutdown_error = _describe_exception(exc)
                        self._condition.notify_all()
                finally:
                    if not self.quarantined:
                        with _owner_registry_lock:
                            _owner_registry.discard(self)

    def _serve(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: len(self._queue) or self._closed or self._unavailable
                                         or self.quarantined)
                if self._closed or self._unavailable or self.quarantined:
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
                if not isinstance(exc, Exception) or isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                    self._retain_unknown(exc)
                failure = _describe_exception(exc)
                cancellation = isinstance(exc, self._cancellation_exception)
                fatal = not isinstance(exc, Exception)
            else:
                try:
                    _validate_forward_result(result, handle._worker_request_id)
                    result = replace(result, request_id=handle.request_id)
                except BaseException:
                    self._retain_unknown(result)
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
                if (poisoned or fatal) and not self._fatal_cleanup_failure and not self.quarantined:
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
                while len(self._queue) and not self._fatal_cleanup_failure and not self.quarantined:
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
        if self._fatal_cleanup_failure or self.quarantined:
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
