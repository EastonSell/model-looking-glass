"""Lifecycle of the trusted GPU-owner service, independent of transport setup.

The enclosing process owns the cross-process lease and durable recovery
marker. Only a separate supervisor may clear a marker after this process exits.
"""
from dataclasses import dataclass
from enum import Enum
import sys
from threading import Event, Thread
from time import monotonic
from types import TracebackType


class OwnerStatus(str, Enum):
    CLEAN = 'clean'
    QUARANTINED = 'quarantined'
    WORKER_FAILED = 'worker_failed'
    SHUTDOWN_INCOMPLETE = 'shutdown_incomplete'


@dataclass(frozen=True)
class OwnerResult:
    status: OwnerStatus
    error_type: str | None = None
    error_message: str | None = None
    teardown_verified: bool = False


@dataclass(frozen=True)
class OwnerHealth:
    """One cooperative, nonblocking observation; never an admission grant."""
    quarantined: bool
    worker_unavailable: bool
    worker_failed: bool
    broker_failed: bool
    broker_closed: bool
    broker_owner_alive: bool


def make_gpt2_request_factory(*, vocab_size, max_positions, max_batch, max_tokens):
    """Validate wire token matrices against a trusted pinned GPT-2 config."""
    for name, value, maximum in (
        ('vocab_size', vocab_size, 2**31), ('max_positions', max_positions, 2**31),
        ('max_batch', max_batch, 256), ('max_tokens', max_tokens, 8192),
    ):
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(f'invalid {name}')
    if max_tokens > max_positions:
        raise ValueError('max_tokens exceeds model positions')
    limits = {'input_ids': vocab_size, 'token_type_ids': vocab_size,
              'position_ids': max_positions, 'attention_mask': 2}

    def factory(request_id, inputs, seed):
        if type(inputs) is not dict or 'input_ids' not in inputs or not set(inputs) <= set(limits):
            raise ValueError('unsupported input names')
        shape = None
        for name, matrix in inputs.items():
            if type(matrix) is not list or not 1 <= len(matrix) <= max_batch:
                raise ValueError('invalid input batch')
            width = None
            for row in matrix:
                if type(row) is not list or not 1 <= len(row) <= max_tokens:
                    raise ValueError('invalid token sequence')
                if width is None:
                    width = len(row)
                if len(row) != width:
                    raise ValueError('ragged token sequence')
                if any(type(value) is not int or not 0 <= value < limits[name] for value in row):
                    raise ValueError('input value outside model bounds')
                if name == 'attention_mask' and not any(row):
                    raise ValueError('attention mask must expose at least one token')
            actual_shape = (len(matrix), width)
            if shape is None:
                shape = actual_shape
            elif shape != actual_shape:
                raise ValueError('input shapes differ')
        import torch
        from .resident_worker import ForwardRequest
        return ForwardRequest(request_id, {name: torch.tensor(matrix, dtype=torch.long)
                                           for name, matrix in inputs.items()}, seed)

    return factory


def _error(exc):
    try:
        message = str(exc)
    except BaseException:
        message = '<exception message unavailable>'
    return type(exc).__name__, message


class _LifecycleFailureSlot:
    """Preallocated roots for one trusted lifecycle producer."""

    def __init__(self, authority):
        self.authority = authority
        self.error = self.traceback = None
        self.followup_error = self.followup_traceback = None
        self.hook_error = self.hook_traceback = None
        self.hook_recovery_error = None

    def _capture(self, exc):
        if self.error is None or self.error is exc:
            self.error = exc
            target = 'traceback'
        elif self.followup_error is None or self.followup_error is exc:
            self.followup_error = exc
            target = 'followup_traceback'
        else:
            raise RuntimeError('lifecycle failure custody is exhausted')
        traceback = BaseException.__dict__['__traceback__'].__get__(exc, BaseException)
        if type(traceback) is TracebackType:
            setattr(self, target, traceback)

    def record(self, exc):
        # No broker condition or notification lock is needed for this signal.
        try:
            self.authority._unsafe = True
        finally:
            try:
                self._capture(exc)
            finally:
                self.authority._unsafe = True
                self._capture(exc)

    def capture_thread_failure(self):
        """Root the active target exception without calling a global hook."""
        try:
            self.authority._unsafe = True
            _, error, traceback = sys.exc_info()
            if not isinstance(error, BaseException) or type(traceback) is not TracebackType:
                raise RuntimeError('lifecycle thread failure capture is unavailable')
            self.hook_error = error
            self.hook_traceback = traceback
        except BaseException as interrupted:
            # If capture is interrupted, CPython chains the original target
            # failure as context. Base descriptors reject spoofed properties.
            self.hook_recovery_error = interrupted
            original = BaseException.__dict__['__context__'].__get__(interrupted, BaseException)
            if isinstance(original, BaseException):
                self.hook_error = original
                traceback = BaseException.__dict__['__traceback__'].__get__(original, BaseException)
                if type(traceback) is TracebackType:
                    self.hook_traceback = traceback
        finally:
            self.authority._unsafe = True

    def recover_hook_interruption(self, interrupted):
        """Recover an interrupted hook call while its target error is active."""
        try:
            self.hook_recovery_error = interrupted
            original = BaseException.__dict__['__context__'].__get__(interrupted, BaseException)
            if isinstance(original, BaseException):
                self.hook_error = original
                traceback = BaseException.__dict__['__traceback__'].__get__(original, BaseException)
                if type(traceback) is TracebackType:
                    self.hook_traceback = traceback
        finally:
            self.authority._unsafe = True


class _ServiceLifecycleCustody:
    """One startup and one shutdown producer, rooted by their owner."""

    def __init__(self, authority):
        from .broker import _supports_bound_owner_thread_hook
        from .owner_completion import OwnerCompletionAuthority
        if type(authority) is not OwnerCompletionAuthority:
            raise TypeError('an exact owner completion authority is required')
        if not _supports_bound_owner_thread_hook():
            raise RuntimeError('lifecycle failure observation requires supported CPython')
        self.authority = authority
        self.startup = _LifecycleFailureSlot(authority)
        self.shutdown = _LifecycleFailureSlot(authority)


def _record_lifecycle_failure(slot, exc):
    try:
        slot.record(exc)
    finally:
        # Retry at a one-shot interruption to the record call or its signal.
        slot.authority._unsafe = True
        slot._capture(exc)


def _observe_thread_exception(thread, failed, on_failure=None, custody=None):
    """Latch an escaped target exception before CPython reports it.

    This per-thread hook covers exceptions escaping the target's return, as
    well as its body, without changing the process-wide threading hook. It is
    a CPython-specific best effort; interruption of thread bootstrap, this
    hook, or the interpreter's finalizer still requires process recovery.
    """
    original = getattr(thread, '_invoke_excepthook', None)
    if not callable(original):
        raise RuntimeError('thread exception observation is unavailable')

    def notify(owner_thread):
        if custody is not None:
            try:
                try:
                    custody.authority._unsafe = True
                    custody.capture_thread_failure()
                except BaseException as interrupted:
                    custody.recover_hook_interruption(interrupted)
            finally:
                custody.authority._unsafe = True
                failed[0] = True
            return
        failed[0] = True
        if on_failure is not None:
            try:
                on_failure()
            finally:
                on_failure()
        original(owner_thread)

    thread._invoke_excepthook = notify


class _CpuServiceWorker:
    """Adapt the resident worker's close state to the owner health contract."""

    def __init__(self, worker):
        self._worker = worker
        self._closed = False
        self.cancellation_exception = worker.cancellation_exception

    @property
    def poisoned(self):
        return self._closed or self._worker.poisoned

    @property
    def failed(self):
        return self._worker.poisoned

    @property
    def quarantined(self):
        return self._worker.quarantined

    def snapshot_request(self, request):
        return self._worker.snapshot_request(request)

    def execute(self, request, cancel_event=None):
        return self._worker.execute(request, cancel_event)

    def close(self):
        self._worker.close()
        self._closed = True


def serve_cpu_model(model, *, model_id, team_ids, stop_event, announce,
                    max_tokens=2, max_pending=8, max_pending_per_team=2,
                    max_connections=8, shutdown_timeout=10):
    """Run a trusted CPU model behind per-team local IPC; no GPU is used."""
    from .broker import Broker
    from .ipc import IPCServer
    from .resident_worker import ResidentWorker

    config = getattr(model, 'config', None)
    factory = make_gpt2_request_factory(vocab_size=getattr(config, 'vocab_size', None),
                                       max_positions=getattr(config, 'n_positions', None),
                                       max_batch=1, max_tokens=max_tokens)
    output_limit = 2 * 1024 * 1024
    response_limit = 4 * 1024 * 1024
    max_by_tensor = output_limit // (8 * config.vocab_size)
    max_by_wire = (response_limit - 512) // (32 * config.vocab_size)
    if max_tokens > min(max_by_tensor, max_by_wire):
        raise ValueError('max_tokens exceeds bounded full-logit response capacity')
    resident = ResidentWorker(model_id, model, device='cpu', max_batch_size=1,
                              max_sequence_length=max_tokens, max_input_bytes=64 * 1024,
                              max_output_bytes=output_limit)
    worker = _CpuServiceWorker(resident)
    broker = Broker(worker, max_pending=max_pending, max_pending_per_team=max_pending_per_team)
    try:
        server = IPCServer(broker, team_ids=team_ids, request_factory=factory,
                           max_connections=max_connections, max_batch=1, max_tokens=max_tokens,
                           max_response_bytes=response_limit)
    except BaseException:
        broker.close(timeout=shutdown_timeout)
        raise
    return serve_owner(server, broker, worker, stop_event, announce=announce,
                       verify_drain=lambda: True, shutdown_timeout=shutdown_timeout)


def _notify_health_uncertainty(callback):
    # The trusted latch is idempotent. If a one-shot interruption arrives at
    # callback entry, the finally attempt still has a chance to publish it.
    try:
        callback()
    finally:
        callback()


def serve_owner(server, broker, worker, stop_event, *, announce, verify_drain,
                poll_interval=0.1, shutdown_timeout=10, health_provider=None,
                on_health_uncertain=None, failure_custody=None):
    """Run an assembled owner and report whether its teardown was verified.

    ``announce(server)`` must keep credentials private. ``verify_drain()``
    must verify completed device work and leases after shutdown and return
    exactly True. Workload failure can coexist with verified teardown.

    The optional ``health_provider`` and ``on_health_uncertain`` are a paired
    trusted interface. The provider must return an exact ``OwnerHealth`` with
    bool fields, without blocking on the broker condition. Its
    ``worker_unavailable`` field means persistently closed or poisoned, not
    transient backpressure. It must synchronously signal resource uncertainty
    to the owner authority and retain any resource-bearing partials or escaped
    exception graphs independently; scalar-only observations are preferable.
    The paired callback must be nonblocking and idempotent, and mark that same
    authority unsafe. It can be invoked more than once, including during error
    handling. A malformed, failed, or uncertain observation cannot establish
    verified teardown. In this mode quarantine leaves cleanup to process
    recovery instead of starting best-effort cleanup threads.

    Health observations are non-atomic, can be stale, and do not grant
    admission or order a concurrent fault with startup, result publication, or
    retirement. All fault sources need their own coordinated authority signal.
    An external process watchdog must bound stalled or repeatedly interrupted
    callbacks. A verified result joins this helper's lifecycle threads; it
    does not establish quiescence of separately owned threads or protect every
    possible interpreter bootstrap, exception-hook, or finalizer interruption.
    By default these helpers format callback errors and delegate escaped thread
    errors; callbacks must independently retain resource-bearing partials.
    An owner may supply its private preallocated ``failure_custody`` to retain
    startup and shutdown errors and escaped target failures until process exit.
    This is bounded cooperative custody, not protection against arbitrary
    repeated interruptions, unassigned native temporaries, or faults in the
    interpreter's thread bootstrap, exception hook, or finalizer.
    """
    options = dict(announce=announce, verify_drain=verify_drain,
                   poll_interval=poll_interval, shutdown_timeout=shutdown_timeout,
                   health_provider=health_provider, on_health_uncertain=on_health_uncertain,
                   failure_custody=failure_custody)
    if failure_custody is not None:
        if type(failure_custody) is not _ServiceLifecycleCustody or health_provider is None:
            raise TypeError('lifecycle failure custody requires the bound health interface')
    if health_provider is None:
        return _serve_owner_impl(server, broker, worker, stop_event, **options)
    if not callable(health_provider) or not callable(on_health_uncertain):
        raise TypeError('health provider and uncertainty callback must both be callable')
    verified = False
    try:
        result = _serve_owner_impl(server, broker, worker, stop_event, **options)
        verified = result.teardown_verified is True
        return result
    finally:
        if not verified:
            # An interrupted observation/handler must not return to the owner
            # while the common authority remains clear. The source callback is
            # trusted and idempotent; repeated interruption requires recovery.
            try:
                on_health_uncertain()
            finally:
                on_health_uncertain()


def _serve_owner_impl(server, broker, worker, stop_event, *, announce, verify_drain,
                      poll_interval=0.1, shutdown_timeout=10, health_provider=None,
                      on_health_uncertain=None, failure_custody=None):
    """Run an already-configured owner and report if teardown was verified.

    `announce(server)` must deliver credentials only to trusted recipients.
    `verify_drain()` must verify all device work and memory leases are complete
    after server/broker shutdown and return exactly True. Only a result with
    teardown_verified=True permits a clean-shutdown attestation; workload failure
    may coexist with verified teardown and must still be reported as failure.
    If QUARANTINED, the owner process must terminate promptly while retaining
    its lease until process exit.

    A verified result waits for both lifecycle threads and rejects exceptions
    escaping their targets via CPython's per-thread exception hook. It does
    not certify arbitrary interruptions of CPython thread bootstrap/finalizers
    or quiescence of independently owned monitor and transport threads.

    A trusted nonblocking ``health_provider`` may replace the legacy worker
    and broker property reads. Worker unavailability means persistent closed
    or poisoned state, not transient backpressure. The provider must report
    observed resource uncertainty to its owner authority synchronously and
    retain any resource-bearing partials or exception graph independently.
    The idempotent paired ``on_health_uncertain`` callback marks that authority
    when an observation raises, is malformed, reports quarantine, or observes
    a closed/dead broker during service. A process watchdog must still bound a
    provider or notification that stalls or is repeatedly interrupted.
    In this opt-in mode an uncertain shutdown is left for process recovery;
    no new best-effort cleanup threads are started after quarantine.
    The pre-start health check is observational and does not replace a
    source-time authority guard against a concurrent fault during startup.
    """
    if not callable(announce) or not callable(verify_drain):
        raise TypeError('announce and verify_drain must be callable')
    if ((health_provider is None) != (on_health_uncertain is None)
            or (health_provider is not None and (not callable(health_provider)
                                                  or not callable(on_health_uncertain)))):
        raise TypeError('health provider and uncertainty callback must both be callable')
    if type(poll_interval) not in (int, float) or not 0 < poll_interval <= 1:
        raise ValueError('poll_interval must be in (0, 1]')
    if type(shutdown_timeout) not in (int, float) or not 0 <= shutdown_timeout <= 300:
        raise ValueError('shutdown_timeout must be in [0, 300]')

    broker_close_started = [False]

    def notify_uncertainty_best_effort(custody=None):
        if health_provider is not None:
            try:
                _notify_health_uncertainty(on_health_uncertain)
            except BaseException as exc:
                if custody is not None:
                    try:
                        _record_lifecycle_failure(custody, exc)
                    finally:
                        custody.authority._unsafe = True
                        custody._capture(exc)

    def quarantine_cleanup():
        if health_provider is None:
            _request_quarantine_cleanup(server, broker)

    def observation():
        accepted = False
        try:
            value = health_provider()
            fields = ('quarantined', 'worker_unavailable', 'worker_failed',
                      'broker_failed', 'broker_closed', 'broker_owner_alive')
            if type(value) is not OwnerHealth or any(type(getattr(value, field)) is not bool for field in fields):
                raise ValueError('malformed owner health observation')
            accepted = True
            return value
        finally:
            if not accepted:
                _notify_health_uncertainty(on_health_uncertain)

    def quarantine_seen():
        if health_provider is None:
            return bool(worker.quarantined or broker.quarantined)
        value = observation()
        if (value.quarantined or (not broker_close_started[0]
                and (value.broker_closed or not value.broker_owner_alive))):
            _notify_health_uncertainty(on_health_uncertain)
            return True
        return False

    def serving_status():
        if health_provider is None:
            if worker.quarantined or broker.quarantined:
                return OwnerStatus.QUARANTINED
            if worker.poisoned or not broker.available:
                return OwnerStatus.WORKER_FAILED
            return None
        value = observation()
        if value.quarantined or value.broker_closed or not value.broker_owner_alive:
            _notify_health_uncertainty(on_health_uncertain)
            return OwnerStatus.QUARANTINED
        if value.worker_unavailable or value.worker_failed or value.broker_failed:
            return OwnerStatus.WORKER_FAILED
        return None

    def terminal_status():
        if health_provider is None:
            if worker.quarantined or broker.quarantined:
                return OwnerStatus.QUARANTINED
            return OwnerStatus.WORKER_FAILED if worker.failed or broker.failed else OwnerStatus.CLEAN
        value = observation()
        if value.quarantined or not value.broker_closed or value.broker_owner_alive:
            _notify_health_uncertainty(on_health_uncertain)
            return OwnerStatus.QUARANTINED
        # Closed/dead is expected only here, after successful broker close/join.
        return OwnerStatus.WORKER_FAILED if value.worker_failed or value.broker_failed else OwnerStatus.CLEAN

    if health_provider is not None:
        try:
            prestart = serving_status()
        except BaseException as exc:
            return OwnerResult(OwnerStatus.QUARANTINED, *_error(exc))
        if prestart is not None:
            if prestart is not OwnerStatus.QUARANTINED:
                notify_uncertainty_best_effort()
            return OwnerResult(OwnerStatus.QUARANTINED)

    status = None
    failure = None
    startup_done = Event()
    startup_errors = []
    # A startup exception handler can itself be interrupted. Begin with an
    # unknown outcome and publish one only after the path has completed.
    startup_outcome = [None]
    startup_terminal = [False]
    startup_thread_failed = [False]
    startup_custody = failure_custody.startup if failure_custody is not None else None

    def startup():
        try:
            server.start()
            announce(server)
        except BaseException as exc:
            try:
                if startup_custody is not None:
                    try:
                        _record_lifecycle_failure(startup_custody, exc)
                    finally:
                        startup_custody.authority._unsafe = True
                        startup_custody._capture(exc)
                if health_provider is not None:
                    _notify_health_uncertainty(on_health_uncertain)
                startup_errors.append(('OwnerLifecycleFailure', 'startup callback failed')
                                      if startup_custody is not None else _error(exc))
                startup_outcome[0] = 'failed'
            except BaseException as recovery:
                if startup_custody is not None:
                    _record_lifecycle_failure(startup_custody, recovery)
        else:
            startup_outcome[0] = 'succeeded'
        startup_terminal[0] = True

    def startup_wrapper():
        completed = False
        try:
            startup()
            completed = startup_outcome[0] == 'succeeded'
        finally:
            # This separate frame protects even an interruption at entry to
            # startup's exception handler before it reaches its first try.
            if startup_custody is not None and not completed:
                startup_custody.authority._unsafe = True
            try:
                startup_done.set()
            except BaseException as exc:
                startup_terminal[0] = False
                if startup_custody is not None:
                    _record_lifecycle_failure(startup_custody, exc)
                return

    startup_thread = Thread(target=startup_wrapper, name='looking-glass-owner-startup', daemon=True)
    _observe_thread_exception(startup_thread, startup_thread_failed, on_health_uncertain,
                              custody=startup_custody)
    startup_thread.start()
    startup_deadline = monotonic() + shutdown_timeout
    while not startup_done.wait(poll_interval):
        try:
            status = serving_status()
            if status is not None:
                break
        except BaseException as exc:
            status = OwnerStatus.QUARANTINED
            failure = _error(exc)
            break
        if stop_event.is_set() or monotonic() >= startup_deadline:
            status = OwnerStatus.SHUTDOWN_INCOMPLETE
            break

    # An Event can be visible before its producing thread has returned. The
    # clean path must join that producer; a stuck or uncertain exit is never a
    # verified device drain. Continue observing quarantine while joining.
    while status is None and startup_done.is_set() and startup_thread.is_alive():
        try:
            status = serving_status()
            if status is not None:
                break
        except BaseException as exc:
            status = OwnerStatus.QUARANTINED
            failure = _error(exc)
            break
        remaining = startup_deadline - monotonic()
        if remaining <= 0:
            status = OwnerStatus.SHUTDOWN_INCOMPLETE
            break
        startup_thread.join(min(poll_interval, remaining))

    if status is None and startup_outcome[0] == 'failed':
        status = OwnerStatus.SHUTDOWN_INCOMPLETE
        if startup_errors:
            failure = startup_errors[0]
    if status is None and (startup_thread_failed[0] or not startup_terminal[0]
                           or startup_outcome[0] is None):
        status = OwnerStatus.SHUTDOWN_INCOMPLETE
    if status is None:
        while True:
            try:
                status = serving_status()
                if status is not None:
                    break
            except BaseException as exc:
                status = OwnerStatus.QUARANTINED
                failure = _error(exc)
                break
            if stop_event.wait(poll_interval):
                break

    if status is OwnerStatus.QUARANTINED:
        quarantine_cleanup()
        return OwnerResult(status, *(failure or (None, None)))

    if health_provider is not None and startup_outcome[0] == 'failed':
        return OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE, *(failure or (None, None)))

    # A startup thread that has not finished can still create or announce a
    # server after cleanup. Request shutdown, but never attest that teardown.
    if (not startup_done.is_set() or startup_thread.is_alive() or startup_thread_failed[0] or not startup_terminal[0]
            or startup_outcome[0] is None):
        notify_uncertainty_best_effort()
        quarantine_cleanup()
        return OwnerResult(status or OwnerStatus.SHUTDOWN_INCOMPLETE,
                           *(failure or (None, None)))

    deadline = monotonic() + shutdown_timeout
    done = Event()
    results = []
    finish_terminal = [False]
    finish_thread_failed = [False]
    finish_custody = failure_custody.shutdown if failure_custody is not None else None

    def finish():
        try:
            if status is OwnerStatus.WORKER_FAILED:
                # Let already accepted jobs receive their terminal failures.
                # Admission is closed first; a bounded grace cannot delay a
                # concurrent quarantine, which the owner thread still polls.
                server.stop_accepting(timeout=max(0, min(1.0, deadline - monotonic())))
                server.drain(timeout=max(0, min(1.0, deadline - monotonic())))
            server_stopped = server.close(timeout=max(0, deadline - monotonic()))
            if health_provider is not None and server_stopped is not True:
                _notify_health_uncertainty(on_health_uncertain)
                result = OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE)
            elif health_provider is not None and quarantine_seen():
                result = OwnerResult(OwnerStatus.QUARANTINED)
            else:
                broker_close_started[0] = True
                broker_stopped = broker.close(timeout=max(0, deadline - monotonic()))
                if quarantine_seen():
                    result = OwnerResult(OwnerStatus.QUARANTINED)
                elif (not server_stopped or not broker_stopped
                      or (health_provider is not None and broker_stopped is not True)):
                    notify_uncertainty_best_effort(finish_custody)
                    result = OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE)
                elif health_provider is not None and terminal_status() is OwnerStatus.QUARANTINED:
                    result = OwnerResult(OwnerStatus.QUARANTINED)
                elif verify_drain() is not True:
                    notify_uncertainty_best_effort(finish_custody)
                    result = OwnerResult(OwnerStatus.QUARANTINED)
                else:
                    final_health = terminal_status()
                    if final_health is OwnerStatus.QUARANTINED:
                        result = OwnerResult(OwnerStatus.QUARANTINED)
                    else:
                        final_status = status or final_health
                        result = OwnerResult(final_status, *(failure or (None, None)), teardown_verified=True)
        except BaseException as exc:
            if finish_custody is not None:
                try:
                    _record_lifecycle_failure(finish_custody, exc)
                finally:
                    finish_custody.authority._unsafe = True
                    finish_custody._capture(exc)
            notify_uncertainty_best_effort(finish_custody)
            details = (('OwnerLifecycleFailure', 'shutdown callback failed')
                       if finish_custody is not None else _error(exc))
            result = OwnerResult(OwnerStatus.QUARANTINED, *details)
        results.append(result)
        try:
            done.set()
        except BaseException as exc:
            if finish_custody is not None:
                _record_lifecycle_failure(finish_custody, exc)
            return
        finish_terminal[0] = True

    finish_thread = Thread(target=finish, name='looking-glass-owner-shutdown', daemon=True)
    _observe_thread_exception(finish_thread, finish_thread_failed, on_health_uncertain,
                              custody=finish_custody)
    finish_thread.start()
    while True:
        try:
            if quarantine_seen():
                quarantine_cleanup()
                return OwnerResult(OwnerStatus.QUARANTINED)
        except BaseException as exc:
            quarantine_cleanup()
            return OwnerResult(OwnerStatus.QUARANTINED, *_error(exc))
        remaining = deadline - monotonic()
        if done.wait(max(0, min(poll_interval, remaining))):
            while finish_thread.is_alive():
                try:
                    if quarantine_seen():
                        quarantine_cleanup()
                        return OwnerResult(OwnerStatus.QUARANTINED)
                except BaseException as exc:
                    quarantine_cleanup()
                    return OwnerResult(OwnerStatus.QUARANTINED, *_error(exc))
                remaining = deadline - monotonic()
                if remaining <= 0:
                    notify_uncertainty_best_effort()
                    return OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE)
                finish_thread.join(min(poll_interval, remaining))
            if finish_thread_failed[0] or not finish_terminal[0]:
                notify_uncertainty_best_effort()
                return OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE)
            try:
                final_health = terminal_status()
                if final_health is OwnerStatus.QUARANTINED:
                    return OwnerResult(OwnerStatus.QUARANTINED)
                if results[0].status is OwnerStatus.CLEAN and final_health is OwnerStatus.WORKER_FAILED:
                    return OwnerResult(OwnerStatus.WORKER_FAILED, teardown_verified=True)
            except BaseException as exc:
                return OwnerResult(OwnerStatus.QUARANTINED, *_error(exc))
            return results[0]
        if remaining <= 0:
            notify_uncertainty_best_effort()
            return OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE)


def _request_quarantine_cleanup(server, broker):
    """Best-effort signals; a failed device drain must not block process exit."""
    def close_server():
        try:
            server.close(timeout=0)
        except BaseException:
            pass

    def close_broker():
        try:
            broker.close(wait=False)
        except BaseException:
            pass

    Thread(target=close_server, name='looking-glass-close-ipc', daemon=True).start()
    Thread(target=close_broker, name='looking-glass-close-broker', daemon=True).start()
