"""Serial ownership, bounded admission, and fair dispatch for a resident worker.

This is a cooperative coordination primitive for trusted local clients, not a
process isolation, VRAM budgeting, batching, or concurrent CUDA execution API.
"""

from dataclasses import dataclass, replace
from enum import Enum
import sys
from threading import Condition, Event, Lock, Thread, current_thread
from time import monotonic, perf_counter_ns
from types import MappingProxyType, TracebackType
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
_forward_request_fields = frozenset(('request_id', 'inputs', 'seed'))
_forward_result_fields = frozenset(('request_id', 'logits'))


def _supports_bound_owner_thread_hook():
    """Restrict the private per-thread hook to reviewed CPython runtimes."""
    if sys.implementation.name != 'cpython' or not (3, 10) <= sys.version_info[:2] <= (3, 13):
        return False
    gil_enabled = getattr(sys, '_is_gil_enabled', None)
    return gil_enabled is None or gil_enabled() is True


def _has_exact_fields(value, fields):
    """Reject extra instance-owned state before rewriting a dataclass."""
    attributes = vars(value)
    return (type(attributes) is dict and len(attributes) == len(fields) and
            all(type(key) is str and key in fields for key in attributes))


def _validate_forward_snapshot(request):
    """Enforce the shared forward-only boundary for requests and snapshots.

    Worker-specific tensor bounds and defensive copying remain the worker's
    responsibility. Trace-bearing request types need a separate custody API.
    """
    if type(request) is not ForwardRequest:
        raise InvalidRequest('broker requires an exact ForwardRequest')
    if not _has_exact_fields(request, _forward_request_fields):
        raise InvalidRequest('broker requires exactly the ForwardRequest fields')
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
    if not _has_exact_fields(result, _forward_result_fields):
        raise ValueError('broker requires exactly the ForwardResult fields')
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
class BrokerHealth:
    """Observational broker state for a separately supervised owner."""

    failed: bool
    quarantined: bool
    closed: bool
    owner_alive: bool


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
    preparation_token: Any = None
    input_custody: Any = None
    receipt_record: Any = None
    staged_raw: Any = None
    staged_decoded: Any = None
    retained_error: Any = None
    retained_status_error: Any = None
    retained_result: Any = None
    retained_cleanup_error: Any = None
    retiring_staged: Any = None
    retiring_completion: Any = None
    pending_result_custody: Any = None
    quarantine_required: bool = False


class _RetirementSignal:
    """A handle-independent signal, guarded by the broker condition."""

    def __init__(self):
        self.complete = False


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
        self._retirement = _RetirementSignal()
        self._retirement_supported = False

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
        try:
            return self._cancel_request()
        except BaseException as exc:
            if not self._broker.quarantined:
                # Cancellation may have removed the job before interruption,
                # including just before the retirement helper is called.
                self._broker._retain_unknown(exc)
            public_error = _sanitized_failure(exc, BrokerUnavailable)
        # Raise outside the exception handler. ``from None`` alone only hides
        # the original in the display; __context__ remains externally mutable.
        _raise_sanitized(public_error)

    def _cancel_request(self):
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
                except BaseException as exc:
                    broker._retain_unknown(exc)
                    if not isinstance(exc, Exception):
                        raise
                    raise BrokerUnavailable('request cancellation cleanup failed') from None
                accepted = True
            elif self._state is RequestState.RUNNING and not self._cancel.is_set():
                self._cancel.set()
                accepted = True
            else:
                accepted = False
        if job is not None:
            broker._retire_job(job)
        del job
        return accepted

    def wait(self, timeout=None):
        """Return a terminal Outcome, or raise TimeoutError if still in flight."""
        if timeout is not None and (isinstance(timeout, bool) or timeout < 0):
            raise ValueError('timeout must be nonnegative or None')
        with self._broker._condition:
            if not self._broker._wait_condition_locked(
                    lambda: self._outcome is not None or self._broker._fatal_cleanup_failure
                    or self._broker.quarantined, timeout):
                raise TimeoutError('request is still in flight')
            if self._outcome is None:
                raise BrokerUnavailable('request status is unknown after broker cleanup failure')
            outcome = self._outcome
        broker = self._broker
        # A terminal already committed before quarantine remains readable.
        # Do not repeatedly enter an observer after owner health is unknown.
        if broker.quarantined:
            return outcome
        try:
            observation = broker._service_observer()
        except BaseException as exc:
            try:
                notification_failure = broker._retain_observation_failure()
            except BaseException as interrupted:
                if broker._observation_cleanup_failure is None:
                    broker._observation_cleanup_failure = interrupted
                broker._quarantine_latched = True
                if broker._completion_authority is not None:
                    broker._completion_authority._unsafe = True
                notification_failure = interrupted
            public_error = _sanitized_failure(notification_failure if notification_failure is not None else exc,
                                              BrokerUnavailable)
        else:
            if observation is not None:
                try:
                    observation._capture_bounded(self, outcome)
                except Exception:
                    # Ordinary metadata failures do not alter the terminal
                    # result. The observation's own latch is independent.
                    observation._fault()
                    return outcome
                except BaseException as exc:
                    # The single pre-rooted callback state holds its exact
                    # handle, outcome and failure. One slot covers all errors
                    # from that state rather than one per caller.
                    try:
                        notification_failure = broker._retain_observation_failure()
                    except BaseException as interrupted:
                        if broker._observation_cleanup_failure is None:
                            broker._observation_cleanup_failure = interrupted
                        broker._quarantine_latched = True
                        if broker._completion_authority is not None:
                            broker._completion_authority._unsafe = True
                        notification_failure = interrupted
                    public_error = _sanitized_failure(notification_failure if notification_failure is not None else exc,
                                                      BrokerUnavailable)
                else:
                    return outcome
            else:
                return outcome
        _raise_sanitized(public_error)

    def wait_for_retirement(self, timeout=None):
        """Wait until fair-handoff execution aliases have been retired.

        The receipt may still retain this handle until the service owner ACKs.
        The caller must independently account for all of its own result aliases.
        This signal is currently available only for fair-handoff handles.
        """
        if timeout is not None and (isinstance(timeout, bool) or timeout < 0):
            raise ValueError('timeout must be nonnegative or None')
        broker = self._broker
        with broker._condition:
            if not self._retirement_supported:
                raise BrokerUnavailable('retirement tracking requires fair handoff')
            if not broker._wait_condition_locked(
                    lambda: self._retirement.complete or broker._fatal_cleanup_failure
                    or broker.quarantined, timeout):
                raise TimeoutError('request retirement is still in flight')
            if broker._fatal_cleanup_failure or broker.quarantined:
                raise BrokerUnavailable('request retirement is unknown after broker cleanup failure')
            return True


class Broker:
    """Queue requests for exactly one trusted, preconstructed resident worker.

    The worker must supply snapshot_request, execute, poisoned, and close, as
    implemented by ResidentWorker. Requests/snapshots and results use the exact
    forward-only types. Clients must not otherwise access the worker's model.
    """

    def __init__(self, worker, *, max_pending, max_pending_per_team, completion_authority=None):
        from .owner_completion import OwnerCompletionAuthority
        try:
            self._initialize_guarded(worker, max_pending=max_pending,
                                     max_pending_per_team=max_pending_per_team,
                                     completion_authority=completion_authority)
            return
        except BaseException as exc:
            if (type(completion_authority) is OwnerCompletionAuthority
                    and completion_authority._broker is self):
                try:
                    if completion_authority._broker_construction_error is None:
                        original = BaseException.__getattribute__(exc, '__context__')
                        completion_authority._broker_construction_error = (
                            original if isinstance(original, BaseException) else exc)
                    elif completion_authority._broker_construction_error is not exc:
                        completion_authority._broker_construction_cleanup_error = exc
                finally:
                    completion_authority._unsafe = True
            elif not getattr(self, '_quarantine_latched', False):
                raise
            public_error = _sanitized_failure(exc, BrokerUnavailable)
        _raise_sanitized(public_error)

    def _initialize_guarded(self, worker, *, max_pending, max_pending_per_team, completion_authority):
        from .owner_completion import OwnerCompletionAuthority
        try:
            self._initialize(worker, max_pending=max_pending, max_pending_per_team=max_pending_per_team,
                             completion_authority=completion_authority)
        except BaseException as exc:
            if (type(completion_authority) is OwnerCompletionAuthority
                    and completion_authority._broker is self):
                try:
                    completion_authority._broker_construction_error = exc
                finally:
                    completion_authority._unsafe = True
            raise

    def _initialize(self, worker, *, max_pending, max_pending_per_team, completion_authority):
        from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
        if type(worker) is LifetimeWorkspaceWorker:
            if (type(completion_authority) is not OwnerCompletionAuthority
                    or worker._authority is not completion_authority):
                raise ValueError('fixed-workspace worker requires its completion authority')
            if worker._cold_pending:
                raise ValueError('cold warmup must finish before broker startup')
            if not _supports_bound_owner_thread_hook():
                raise RuntimeError('fixed worker requires a supported owner thread failure hook')
        elif completion_authority is not None:
            raise ValueError('completion authority requires the fixed-workspace worker')
        # Retain the original authority as an invariant of this broker. An
        # attempted replacement must not turn later decisions into unbound
        # publication or retirement.
        self._completion_authority_binding = completion_authority
        self._serial_observation = None
        self._observation_owner = (worker._cpu_observation if completion_authority is not None else None)
        self._observation_failure_owner = None
        self._observation_cleanup_failure = None
        if completion_authority is not None:
            completion_authority._bind_broker(self, worker)
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
        self._fair_preparations = set()
        self._running = None
        # Every preparing, queued, executing, or retiring job has a durable
        # slot before its request can become visible to the queue. Queue
        # removal cannot then orphan it, even if Python is interrupted before
        # the returned item is assigned to a local variable.
        self._job_roots = [None] * (max_pending + 1)
        self._closed = False
        self._unavailable = None
        self._fatal_cleanup_failure = False
        self._shutdown_error = None
        self._fair_handoff = None
        self._fair_running_job = None
        # One condition-serialized handoff can fail before creating a job or
        # joining the preparation set. Retain that state through quarantine.
        self._fair_orphan_state = None
        self._quarantine_latched = False
        # One preflight status probe can fail before any job exists. The
        # condition serializes probes, and the latch rejects later probes.
        self._status_error = None
        self._status_cleanup_error = None
        self._status_recovery_error = None
        # Unexpected owner failures must be observable and retained even when
        # another thread holds the admission condition indefinitely.
        self._owner_failure = None
        self._owner_escaping_failure = None
        self._owner_hook_failure = None
        self._owner_hook_traceback = None
        self._owner_hook_capture_complete = False
        self._owner_hook_recovery_failure = None
        self._startup_decided = False
        # Every rooted job can have a concurrent cleanup exception and a
        # detached input whose disposal was interrupted. Reserve separate
        # fixed slots for all of those plus owner/shutdown exception paths.
        self._quarantine_empty = object()
        self._quarantine_objects = [self._quarantine_empty] * (3 * (max_pending + 1) + 4)
        self._thread = Thread(target=self._run, name='looking-glass-worker', daemon=True)
        if completion_authority is not None:
            # CPython invokes this per-thread hook when an exception escapes
            # the target, including at its return boundary. Keep the exact
            # failure private; an application hook could mutate its traceback.
            if not hasattr(self._thread, '_invoke_excepthook'):
                raise RuntimeError('the owner thread failure hook is unavailable')
            self._thread._invoke_excepthook = self._capture_owner_thread_failure
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
            if not isinstance(exc, Exception):
                raise
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
    def _completion_authority(self):
        return self._completion_authority_binding

    @_completion_authority.setter
    def _completion_authority(self, value):
        authority = self._completion_authority_binding
        if value is authority:
            return
        try:
            if authority is not None:
                authority._unsafe = True
        finally:
            self._quarantine_latched = True
            if authority is not None:
                authority._unsafe = True
        raise RuntimeError('broker completion authority cannot be replaced')

    @_completion_authority.deleter
    def _completion_authority(self):
        authority = self._completion_authority_binding
        try:
            if authority is not None:
                authority._unsafe = True
        finally:
            self._quarantine_latched = True
            if authority is not None:
                authority._unsafe = True

    def _capture_owner_thread_failure(self, thread):
        """Root an escaped bound-owner failure before thread completion."""
        try:
            self._capture_owner_thread_failure_inner(thread)
        except BaseException as interrupted:
            # If capture itself is interrupted before sys.exc_info() returns,
            # the original target exception is the interruption's context.
            # Use base descriptors so an exception subclass cannot spoof it.
            try:
                self._owner_hook_recovery_failure = interrupted
                original = BaseException.__dict__['__context__'].__get__(interrupted, BaseException)
                if not isinstance(original, BaseException):
                    original = interrupted
                if self._owner_escaping_failure is None:
                    self._owner_escaping_failure = original
                elif self._owner_escaping_failure is not original and self._owner_hook_failure is None:
                    self._owner_hook_failure = original
                traceback = BaseException.__dict__['__traceback__'].__get__(original, BaseException)
                if type(traceback) is TracebackType:
                    self._owner_hook_traceback = traceback
            finally:
                self._quarantine_latched = True
                authority = self._completion_authority_binding
                if authority is not None:
                    authority._unsafe = True

    def _capture_owner_thread_failure_inner(self, thread):
        """Capture from the active CPython exception context before notification."""
        exc = None
        captured_traceback = None
        try:
            _, exc, captured_traceback = sys.exc_info()
            if not isinstance(exc, BaseException) or type(captured_traceback) is not TracebackType:
                raise RuntimeError('owner thread failure capture is unavailable')
            if self._owner_escaping_failure is None:
                self._owner_escaping_failure = exc
            else:
                self._owner_hook_failure = exc
            self._owner_hook_traceback = captured_traceback
            self._owner_hook_capture_complete = True
        finally:
            authority = self._completion_authority_binding
            try:
                try:
                    if authority is not None:
                        authority._unsafe = True
                finally:
                    self._quarantine_latched = True
            finally:
                if authority is not None:
                    authority._unsafe = True
                self._quarantine_latched = True
                if self._owner_escaping_failure is None:
                    self._owner_escaping_failure = exc
                elif self._owner_escaping_failure is not exc and self._owner_hook_failure is None:
                    self._owner_hook_failure = exc
                if type(captured_traceback) is TracebackType and self._owner_hook_traceback is None:
                    self._owner_hook_traceback = captured_traceback

    def _wait_condition_locked(self, predicate, timeout=None):
        """Poll the lock-free owner signal without blocking its publisher."""
        if self._completion_authority is None:
            return self._condition.wait_for(predicate, timeout)
        deadline = None if timeout is None else monotonic() + max(0, timeout)
        while not predicate():
            interval = 0.05 if deadline is None else min(0.05, max(0, deadline - monotonic()))
            if interval <= 0:
                return False
            self._condition.wait(interval)
        return True

    @property
    def health_snapshot(self):
        """Observe broker-owned latches without waiting for admission or a worker.

        These fields are not a synchronized snapshot and may change while they
        are read. They cannot authorize admission or prove device completion.
        Supervisors must separately monitor worker and transport health.
        """
        quarantined = self.quarantined
        return BrokerHealth(
            failed=bool(self._unavailable or self._fatal_cleanup_failure or quarantined),
            quarantined=quarantined,
            closed=self._closed,
            owner_alive=self._thread.is_alive(),
        )

    @property
    def available(self):
        """Whether this broker can currently accept requests."""
        with self._condition:
            return (not self._closed and not self._unavailable and not self.quarantined
                    and not self._worker_poisoned_locked())

    @property
    def failed(self):
        """Whether a broker failure occurred, distinct from orderly closure."""
        with self._condition:
            return bool(self._unavailable or self._fatal_cleanup_failure or self.quarantined)

    @property
    def quarantined(self):
        """Unknown worker-owned data requires terminating the owner process."""
        return (self._quarantine_latched or
                (self._completion_authority is not None and self._completion_authority.unsafe))

    def _signal_completion_unsafe(self):
        if self._completion_authority is not None:
            self._completion_authority.mark_unsafe()

    def _service_observer(self):
        """Return the exact optional CPU observer or fail its owner closed."""
        try:
            observer = self._serial_observation
            authority = self._completion_authority
            if authority is None:
                if observer is not None or self._observation_owner is not None:
                    raise RuntimeError('serial observation requires a fixed owner')
                return None
            worker = self._worker
            expected = worker._cpu_observation
            if (self._observation_owner is not expected or worker._cpu_observation_binding is not expected
                    or worker._authority is not authority or authority._worker is not worker
                    or authority._broker is not self):
                raise RuntimeError('fixed owner observation policy changed')
            if expected is None:
                if observer is not None or worker._observation_bound is not False:
                    raise RuntimeError('fixed owner requires recording to be disabled')
                return None
            if (observer is not expected or worker._observation_bound is not True
                    or worker._worker.device.type != 'cpu'):
                raise RuntimeError('CPU observation owner identity changed')
            return observer
        except BaseException:
            self._signal_completion_unsafe()
            raise

    def _retain_observation_failure(self):
        """Latch a dedicated, preallocated root for observation failures."""
        try:
            owner = self._observation_owner
            # The broker was registered before its thread started. Its
            # dedicated field roots the exact observer and both callback
            # states, without consuming the job/owner quarantine slots.
            self._observation_failure_owner = owner if owner is not None else self
            self._quarantine_latched = True
            if self._completion_authority is not None:
                self._completion_authority.mark_unsafe()
            if owner is not None:
                owner._faulted = True
            with self._condition:
                self._mark_cleanup_failed_locked()
        except BaseException as exc:
            # The supervisor must replace an owner if this transition or its
            # notification is interrupted, even when its lock is stranded.
            if self._observation_cleanup_failure is None:
                self._observation_cleanup_failure = exc
            self._quarantine_latched = True
            if self._completion_authority is not None:
                self._completion_authority._unsafe = True
            return exc
        return None

    def _worker_poisoned_locked(self):
        """Probe trusted worker status while preserving unknown failures."""
        return self._worker_property_locked('poisoned')

    def _worker_property_locked(self, name):
        """Read a trusted worker property and quarantine unknown failures."""
        try:
            return self._worker_poisoned_probe_locked(name)
        except BaseException as exc:
            # A second interruption may occur before the inner probe records
            # its original error. This preallocated slot retains that outer
            # exception and its private context before returning to a caller.
            self._status_cleanup_error = exc
            try:
                self._mark_cleanup_failed_locked()
            except BaseException as recovery_exc:
                self._status_recovery_error = recovery_exc
                self._signal_completion_unsafe()
                self._quarantine_latched = True
                self._fatal_cleanup_failure = True
                self._unavailable = 'broker cleanup failed; request status is unknown'
                if not isinstance(recovery_exc, Exception):
                    exc = recovery_exc
            public_error = _sanitized_failure(exc, BrokerUnavailable)
        _raise_sanitized(public_error)

    def _worker_poisoned_probe_locked(self, name='poisoned'):
        try:
            value = getattr(self._worker, name)
            return bool(value) if name == 'poisoned' else value
        except BaseException as exc:
            self._status_error = exc
            name = type(exc).__name__
            if isinstance(exc, KeyboardInterrupt):
                kind = 1
            elif isinstance(exc, SystemExit):
                kind = 2
            elif isinstance(exc, Exception):
                kind = 0
            else:
                kind = 3
            try:
                self._mark_cleanup_failed_locked()
            except BaseException as cleanup_exc:
                self._status_cleanup_error = cleanup_exc
                self._signal_completion_unsafe()
                self._quarantine_latched = True
                self._fatal_cleanup_failure = True
                self._unavailable = 'broker cleanup failed; request status is unknown'
                if not isinstance(cleanup_exc, Exception):
                    name = type(cleanup_exc).__name__
                    kind = 1 if isinstance(cleanup_exc, KeyboardInterrupt) else (
                        2 if isinstance(cleanup_exc, SystemExit) else 3)
        # Preserve only the fatal category in a fresh exception. In
        # particular, its mutable traceback must not be the retained one.
        try:
            if kind == 1:
                raise KeyboardInterrupt(name) from None
            if kind == 2:
                raise SystemExit(name) from None
            if kind == 3:
                raise BaseException(name) from None
            raise BrokerUnavailable('worker status is unknown; terminate the owner process') from None
        except BaseException as public_error:
            public_error.__context__ = None
            raise

    def _worker_limits_locked(self):
        """Read and validate per-snapshot/result caps under the condition."""
        limits = []
        for name in ('max_input_bytes', 'max_output_bytes'):
            value = self._worker_property_locked(name)
            if type(value) is not int or value <= 0:
                # An unsupported value may itself own opaque resources.
                self._retain_unknown(value)
                raise BrokerUnavailable('worker tensor limit is invalid; terminate the owner process') from None
            limits.append(value)
        return limits[0], limits[1]

    def _retain_unknown(self, value):
        # Retain before signaling. A failed notification must not release the
        # object; quarantine remains independently visible to admission/close.
        # Signal the fixed owner before waiting for an occupied condition; the
        # producer frame still owns value until its durable slot is populated.
        try:
            self._signal_completion_unsafe()
        finally:
            # Even an interruption before the helper enters must leave the
            # fixed owner's nonblocking signal latched.
            if self._completion_authority is not None:
                self._completion_authority.mark_unsafe()
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

    def enable_fair_handoff(self, teams, *, decoder, max_item_bytes,
                            max_decoded_bytes, preparation_bytes, max_preparation_bytes,
                            max_result_bytes, max_unacknowledged):
        """Enable a trusted waiting adapter while idle, disabling direct submit.

        The decoder and its declared bounds are owner-controlled. This mode is
        experimental and has not been wired into the IPC server.
        """
        from .broker_fair_handoff import BrokerFairHandoff

        with self._condition:
            if self._closed or self._unavailable or self.quarantined or self._worker_poisoned_locked():
                raise BrokerUnavailable('broker is closed or worker is unavailable')
            if (self._fair_handoff is not None or self._running is not None or
                    self._pending_slots or self._preparing or len(self._queue) or
                    any(job is not None for job in self._job_roots)):
                raise BrokerUnavailable('fair mode requires an idle, unbound broker')
            target = BrokerFairHandoff(self, teams, decoder, max_item_bytes,
                                       max_decoded_bytes, preparation_bytes,
                                       max_preparation_bytes, max_result_bytes,
                                       max_unacknowledged)
            team_slots = {team: 0 for team in target.teams}
            self._team_slots = team_slots
            self._fair_handoff = target
            return target

    def _root_job_locked(self, job):
        """Reserve a durable slot under the condition before publishing work."""
        if any(existing is job for existing in self._job_roots):
            raise RuntimeError('job is already rooted')
        for index, existing in enumerate(self._job_roots):
            if existing is None:
                self._job_roots[index] = job
                return
        raise QueueFull('broker job-retirement capacity reached')

    def _retire_job(self, job):
        """Release a known-clean job root, retaining it on uncertain cleanup.

        The caller must still own ``job`` and must have completed all work
        using it. The actual last reference is released outside the condition.
        This does not prove disposal of aliases held by clients or a worker.
        """
        retiring_request = None
        try:
            with self._condition:
                if self._fatal_cleanup_failure or self.quarantined:
                    return False
                if not any(existing is job for existing in self._job_roots):
                    raise RuntimeError('job has no broker-owned root')
                # Commit disposal against quarantine under the same lock.
                # The local reference prevents a finalizer from running here.
                if self._completion_authority is not None:
                    # Pre-root every result/error alias before the authority's
                    # retirement decision. The unsafe flag read linearizes
                    # this cleanup before or after a concurrent unsafe mark.
                    job.retiring_completion = (job.retained_result, job.retained_error,
                                                job.retained_status_error)
                    with self._completion_authority._decision_lock:
                        if self._completion_authority.unsafe:
                            return False
                        retiring_request = job.request
                        job.request = None
                        job.retained_result = None
                        job.retained_error = None
                        job.retained_status_error = None
                else:
                    retiring_request = job.request
                    job.request = None
            # Retirement has won against later quarantine. Keep the root
            # occupied until releasing the input really returns, including
            # while a finalizer blocks; no finalizer runs under the condition.
            retiring_request = None
            # Last-reference finalizers run outside the broker condition.
            job.retiring_completion = None
            self._retire_job_input(job)
            self._retire_job_result(job)
            with self._condition:
                if self._fatal_cleanup_failure or self.quarantined:
                    return False
                for index, existing in enumerate(self._job_roots):
                    if existing is job:
                        self._job_roots[index] = None
                        self._condition.notify_all()
                        return True
                raise RuntimeError('job has no broker-owned root')
        except BaseException as exc:
            # Assignment or notification can be interrupted after removal.
            # The caller and this frame still own the job at this boundary.
            if retiring_request is not None:
                self._retain_unknown(retiring_request)
            self._retain_unknown(job)
            self._retain_unknown(exc)
            raise

    def _retire_items(self, items):
        for item in items:
            self._retire_job(item.payload)

    def submit(self, team_id, request):
        """Submit a supported request or raise QueueFull/BrokerUnavailable.

        Snapshot validation errors propagate and do not occupy queue capacity.
        The calling client must not mutate its input during snapshotting.
        """
        preparation_token = object()
        handle = None
        observation = None
        try:
            handle = self._submit(team_id, request, preparation_token)
            try:
                observation = self._service_observer()
                if observation is not None:
                    observation._register_bounded(handle)
            except BaseException as exc:
                try:
                    notification_failure = self._retain_observation_failure()
                except BaseException as interrupted:
                    if self._observation_cleanup_failure is None:
                        self._observation_cleanup_failure = interrupted
                    self._quarantine_latched = True
                    if self._completion_authority is not None:
                        self._completion_authority._unsafe = True
                    notification_failure = interrupted
                if notification_failure is not None:
                    raise notification_failure
                raise
            return handle
        except BaseException as exc:
            with self._condition:
                unresolved = any(job is not None and job.preparation_token is preparation_token
                                 for job in self._job_roots)
            if unresolved and handle is None:
                # Also covers interruption after slot release but before a
                # failed preparation's root has been retired.
                self._retain_unknown(exc)
            elif handle is not None and not self.quarantined:
                # An interruption can occur after acceptance but before the
                # observation helper begins. No metadata callback ran yet.
                try:
                    notification_failure = self._retain_observation_failure()
                except BaseException as interrupted:
                    if self._observation_cleanup_failure is None:
                        self._observation_cleanup_failure = interrupted
                    self._quarantine_latched = True
                    if self._completion_authority is not None:
                        self._completion_authority._unsafe = True
                    notification_failure = interrupted
                if notification_failure is not None:
                    exc = notification_failure
            if not self.quarantined:
                raise
            public_error = _sanitized_failure(exc, BrokerUnavailable)
        _raise_sanitized(public_error)

    def _submit(self, team_id, request, preparation_token):
        submitted_ns = perf_counter_ns()
        if type(team_id) is not str or not team_id:
            raise ValueError('team_id must be a nonempty string')
        _validate_forward_snapshot(request)
        request_id = request.request_id
        with self._condition:
            if self._closed or self._unavailable or self.quarantined or self._worker_poisoned_locked():
                raise BrokerUnavailable('broker is closed or worker is unavailable')
            if self._fair_handoff is not None:
                raise BrokerUnavailable('direct submission is disabled in fair mode')
            if self._pending_slots >= self._max_pending:
                raise QueueFull('global pending-job limit reached')
            if self._team_slots.get(team_id, 0) >= self._max_pending_per_team:
                raise QueueFull('team pending-job limit reached')
            if all(job is not None for job in self._job_roots):
                raise QueueFull('broker job-retirement capacity reached')
            next_pending = self._pending_slots + 1
            next_team = self._team_slots.get(team_id, 0) + 1
            next_preparing = self._preparing + 1
            next_job_id = self._next_job_id + 1
            handle = RequestHandle(self, next_job_id, team_id, submitted_ns)
            job = _Job(handle, preparation_token=preparation_token)
            try:
                self._root_job_locked(job)
                self._team_slots[team_id] = next_team
                self._pending_slots = next_pending
                self._preparing = next_preparing
                self._next_job_id = next_job_id
            except BaseException as exc:
                # A slot or accounting mutation might already have happened.
                self._retain_unknown(job)
                self._retain_unknown(exc)
                raise
        # Preparation holds a bounded slot, but not the condition. A slow CPU
        # copy cannot prevent cancellation of a different running request.
        try:
            try:
                snapshot = self._worker.snapshot_request(request)
                job.request = snapshot
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
                job.request = snapshot
                # After validation, the rooted job is the sole broker-owned
                # snapshot reference even if later publication fails.
                snapshot = None
            except BaseException:
                self._retain_unknown(snapshot)
                raise
        except BaseException as exc:
            removed = []
            with self._condition:
                self._preparing -= 1
                if self._fatal_cleanup_failure or self.quarantined:
                    # Another thread already found accounting uncertain. Do not
                    # release any reservation or drain the queue on that basis.
                    self._condition.notify_all()
                    if not isinstance(exc, Exception):
                        raise
                    raise BrokerUnavailable('snapshot status is unknown after broker cleanup failure') from None
                poisoned = self._worker_poisoned_locked()
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
            self._retire_items(removed)
            self._retire_job(job)
            del removed
            raise
        publication_started = False
        try:
            with self._condition:
                try:
                    if self._closed or self._unavailable or self.quarantined or self._worker_poisoned_locked():
                        raise BrokerUnavailable('broker closed while preparing request')
                    handle._request_id = request_id
                    handle._worker_request_id = worker_request_id
                    handle._job = job
                    handle._ready_ns = perf_counter_ns()
                    try:
                        # After this point the queue may own the job even if
                        # an interruption prevents submit from returning.
                        publication_started = True
                        self._queue.put(team_id, handle.job_id, job)
                    except BaseException as exc:
                        # If queue publication itself fails, do not assume its
                        # rollback completed under host-memory pressure.
                        self._retain_unknown(exc)
                        if not isinstance(exc, Exception):
                            raise
                        raise BrokerUnavailable('queue publication failed; accounting is unknown') from None
                except BaseException as exc:
                    if publication_started:
                        self._retain_unknown(exc)
                    elif not self._fatal_cleanup_failure and not self.quarantined:
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
        except BaseException as exc:
            if publication_started:
                self._retain_unknown(exc)
            else:
                self._retire_job(job)
            raise

    def _release_slot_locked(self, team_id):
        team_count = self._team_slots.get(team_id, 0)
        if self._pending_slots <= 0 or team_count <= 0:
            raise RuntimeError('invalid pending-slot accounting')
        pending = self._pending_slots - 1
        remaining = team_count - 1
        if remaining or self._fair_handoff is not None:
            self._team_slots[team_id] = remaining
        else:
            del self._team_slots[team_id]
        self._pending_slots = pending

    def _mark_cleanup_failed_locked(self):
        # Do not discard pending requests or decrement unverified reservations.
        try:
            self._signal_completion_unsafe()
        finally:
            if self._completion_authority is not None:
                self._completion_authority.mark_unsafe()
            self._quarantine_latched = True
            self._fatal_cleanup_failure = True
            self._unavailable = 'broker cleanup failed; request status is unknown'
            self._condition.notify_all()

    def _retire_job_input(self, job):
        """Dispose a finished job's input before releasing its host claim."""
        custody = job.input_custody
        if custody is None:
            return
        job.request = None
        job.input_custody = None
        self._fair_handoff._release(custody)

    def _retire_job_result(self, job):
        """Dispose broker result aliases before marking a receipt retired."""
        if job.receipt_record is None:
            return
        target = self._fair_handoff
        with self._condition:
            held_handle = job.handle
            job.handle = None
        # Last-reference finalizers may execute here, never under the condition.
        del held_handle
        target.retire_receipt(job)

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
        try:
            self._run_inner()
        except BaseException as exc:
            # A failure in the owner error/cleanup path must not escape to a
            # thread exception hook that could clear its private traceback.
            self._owner_escaping_failure = exc
            if self._owner_failure is None:
                original = BaseException.__getattribute__(exc, '__context__')
                self._owner_failure = original if isinstance(original, BaseException) else exc
            self._signal_completion_unsafe()
            self._quarantine_latched = True

    def _run_inner(self):
        removed = []
        serve_returned = False
        try:
            # Do not close/unregister before the constructor knows whether
            # Thread.start returned, including a start-then-raise interruption.
            with self._condition:
                self._condition.wait_for(lambda: self._startup_decided)
            self._serve()
            serve_returned = True
        except BaseException as exc:
            # A stalled handoff may hold the condition. Signal the supervisor
            # and root the private exception before waiting for that condition.
            # Repeated assignments preserve both on a one-shot interruption.
            try:
                self._owner_failure = exc
            finally:
                try:
                    self._signal_completion_unsafe()
                    self._quarantine_latched = True
                finally:
                    self._owner_failure = exc
                    self._signal_completion_unsafe()
                    self._quarantine_latched = True
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
            if not serve_returned:
                # This also covers an interruption before the failure handler
                # could root its exception or notify the supervisor.
                self._signal_completion_unsafe()
                self._quarantine_latched = True
                escaping = sys.exc_info()[1]
                if escaping is not None:
                    self._owner_escaping_failure = escaping
                    if self._owner_failure is None:
                        original = BaseException.__getattribute__(escaping, '__context__')
                        self._owner_failure = original if isinstance(original, BaseException) else escaping
            # A snapshot in progress may still use the worker. Never close it
            # until all accepted preparation reservations have been released.
            with self._condition:
                self._wait_condition_locked(lambda: self._preparing == 0 and not self._fair_preparations and
                                            (self.quarantined or all(job is None for job in self._job_roots)))
            del removed
            if not self.quarantined:
                try:
                    self._worker.close()
                except BaseException as exc:
                    self._retain_unknown(exc)
                    with self._condition:
                        self._shutdown_error = _describe_exception(exc)
                        self._condition.notify_all()
                else:
                    # An interrupted error handler must leave the owner rooted
                    # until the outer guard can latch and retain the failure.
                    # A bound owner stays registered through process exit,
                    # including after an external caller joins its thread.
                    if not self.quarantined and self._completion_authority is None:
                        with _owner_registry_lock:
                            _owner_registry.discard(self)

    def _serve(self):
        while True:
            with self._condition:
                self._wait_condition_locked(lambda: len(self._queue) or self._closed or self._unavailable
                                            or self.quarantined)
                if self._closed or self._unavailable or self.quarantined:
                    return
                item = self._queue.pop()
                job = item.payload
                handle = job.handle
                self._running = handle
                if job.input_custody is not None:
                    # Keep the actual input owner rooted if execution becomes
                    # uncertain, not just the numerical ledger commitment.
                    self._fair_running_job = job
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
                observation = self._service_observer()
                if observation is None:
                    result = self._worker.execute(job.request, handle._cancel)
                else:
                    result = observation._execute(self._worker, handle, job.request, handle._cancel)
            except BaseException as exc:
                job.retained_error = exc
                if not isinstance(exc, Exception) or isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                    self._retain_unknown(exc)
                failure = _describe_exception(exc)
                cancellation = isinstance(exc, self._cancellation_exception)
                fatal = not isinstance(exc, Exception)
            else:
                # Root the producer's exact object before any validation or
                # relabeling can fail. The fixed worker independently retains
                # it until this job root has taken custody.
                original_result = result
                job.retained_result = result
                try:
                    _validate_forward_result(result, handle._worker_request_id)
                    if self._fair_handoff is not None:
                        self._fair_handoff.validate_result_storage(result)
                    result = replace(result, request_id=handle.request_id)
                    job.retained_result = result
                    if self._completion_authority is not None and not self.quarantined:
                        self._worker._transfer_result(original_result)
                except BaseException:
                    self._retain_unknown(result)
                    failure = ('InvalidWorkerResult', 'worker returned an invalid request identity or result')
                    fatal = True
                    result = None
            # Keep the exact returned object rooted if the status probe itself
            # fails; its fields may own resources beyond numerical ledgers.
            job.retained_result = result
            if self._completion_authority is not None and failure is None:
                try:
                    # Recheck the supervised lease without holding the broker
                    # condition. External revocation can still race afterward;
                    # only the authority flag is ordered with publication.
                    self._completion_authority._before_device()
                except BaseException as exc:
                    job.retained_status_error = exc
                    failure = _describe_exception(exc)
                    fatal = not isinstance(exc, Exception)
            with self._condition:
                if self.quarantined:
                    poisoned = True
                else:
                    try:
                        poisoned = self._worker_poisoned_locked()
                    except BaseException as exc:
                        poisoned = True
                        job.retained_status_error = exc
                        if failure is None or cancellation:
                            failure = _describe_exception(exc)
                            cancellation = False
                        if not isinstance(exc, Exception):
                            fatal = True
                if poisoned and job.input_custody is not None:
                    self._mark_cleanup_failed_locked()
                    if failure is None:
                        failure = ('WorkerUnavailable', 'worker poisoned during execution')
                        result = None
                        cancellation = False
                if self.quarantined and failure is None:
                    failure = ('WorkerUnavailable', 'worker status is unknown after quarantine')
                    result = None
                    cancellation = False
                if failure is not None and (not cancellation or poisoned):
                    self._finish_locked(handle, RequestState.FAILED, error_type=failure[0], error_message=failure[1])
                elif cancellation or handle._cancel.is_set():
                    self._finish_locked(handle, RequestState.CANCELLED)
                else:
                    if self._completion_authority is None:
                        self._finish_locked(handle, RequestState.SUCCEEDED, result=result)
                    else:
                        authority = self._completion_authority
                        # All validation, cancellation checks and job rooting
                        # precede this read. A later unsafe request is ordered
                        # after this success decision even if delivery follows.
                        with authority._decision_lock:
                            if authority.unsafe:
                                self._mark_cleanup_failed_locked()
                                self._finish_locked(handle, RequestState.FAILED,
                                    error_type='WorkerUnavailable',
                                    error_message='owner became unsafe before success')
                            else:
                                self._finish_locked(handle, RequestState.SUCCEEDED, result=result)
                self._running = None
                if (poisoned or fatal) and not self._fatal_cleanup_failure and not self.quarantined:
                    self._unavailable = 'worker failed and cannot be reused'
                    if not self._fatal_cleanup_failure:
                        removed = self._fail_pending_locked()
                # A waiting client can wake when the condition is released.
                # Drop this thread's result/handle aliases before retirement.
                item = None
                handle = None
                result = None
                original_result = None
            # Do not retain finished inputs, exceptions, or results in the owner
            # thread while waiting for the next job.
            if not self._fatal_cleanup_failure and not self.quarantined:
                if not poisoned and not fatal:
                    if self._completion_authority is None:
                        job.retained_error = None
                        job.retained_status_error = None
                        job.retained_result = None
                    self._retire_job(job)
                    with self._condition:
                        if not self._fatal_cleanup_failure and not self.quarantined:
                            self._fair_running_job = None
                elif job.input_custody is None:
                    # The base trusted-worker contract permits clean disposal
                    # after an ordinary poisoned direct execution failure.
                    if self._completion_authority is None:
                        job.retained_error = None
                        job.retained_status_error = None
                        job.retained_result = None
                    self._retire_job(job)
                self._retire_items(removed)
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
        try:
            return self._close(wait=wait, timeout=timeout)
        except BaseException as exc:
            if not self.quarantined:
                # A close can be interrupted after popping jobs but before
                # handing them to the retirement helper.
                self._retain_unknown(exc)
            public_error = _sanitized_failure(exc, BrokerShutdownError)
        _raise_sanitized(public_error)

    def _close(self, *, wait=True, timeout=None):
        removed = []
        cleanup_failure = None
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
            except BaseException as exc:
                self._retain_unknown(exc)
                cleanup_failure = exc
            if self._running is not None:
                self._running._cancel.set()
            handle = None
            self._condition.notify_all()
        if cleanup_failure is None and not self._fatal_cleanup_failure and not self.quarantined:
            try:
                self._retire_items(removed)
            except BrokerUnavailable:
                raise BrokerShutdownError('broker could not release input custody') from None
        del removed
        if cleanup_failure is not None:
            if not isinstance(cleanup_failure, Exception):
                raise cleanup_failure
            raise BrokerShutdownError('broker could not complete pending-request cleanup')
        if self._fatal_cleanup_failure or self.quarantined:
            raise BrokerShutdownError('broker cleanup failed; request status and accounting are unknown')
        if wait:
            deadline = None if timeout is None else monotonic() + max(0, timeout)
            while self._thread.is_alive():
                if self._fatal_cleanup_failure or self.quarantined:
                    raise BrokerShutdownError('broker cleanup failed; request status and accounting are unknown')
                interval = 0.05 if deadline is None else min(0.05, max(0, deadline - monotonic()))
                if interval <= 0:
                    break
                self._thread.join(interval)
        stopped = not self._thread.is_alive()
        if self._fatal_cleanup_failure or self.quarantined:
            raise BrokerShutdownError('broker cleanup failed; request status and accounting are unknown')
        if stopped and self._shutdown_error is not None:
            raise BrokerShutdownError(f'worker close failed: {self._shutdown_error[0]}: {self._shutdown_error[1]}')
        if stopped and self._completion_authority is not None:
            authority = self._completion_authority
            with authority._decision_lock:
                if self._quarantine_latched or authority.unsafe:
                    raise BrokerShutdownError('broker owner exit is uncertain')
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


def _sanitized_failure(exc, ordinary_type):
    """Return a fresh public error without disclosing retained tracebacks."""
    message = 'broker ownership is uncertain; terminate the owner process'
    if isinstance(exc, KeyboardInterrupt):
        return KeyboardInterrupt(message)
    if isinstance(exc, SystemExit):
        return SystemExit(message)
    if isinstance(exc, GeneratorExit):
        return GeneratorExit(message)
    if not isinstance(exc, Exception):
        return BaseException(message)
    return ordinary_type(message)


def _raise_sanitized(error):
    """Raise without attaching a caller's active exception as mutable context."""
    try:
        raise error from None
    except BaseException:
        error.__context__ = None
        error.__cause__ = None
        raise
