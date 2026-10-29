"""CPU-only lifecycle tests; no model, CUDA, or actual service credentials."""

from dataclasses import dataclass
import dis
import gc
import inspect
import sys
import threading
from threading import Event, Lock, Thread
from time import monotonic, sleep
import weakref

from looking_glass.fair_service_lifecycle import _FairOwnerLifecycle
from looking_glass.service import OwnerStatus, serve_owner


@dataclass(frozen=True)
class Health:
    failed: bool = False
    quarantined: bool = False
    closed: bool = False
    owner_alive: bool = True


class FakeWorker:
    poisoned = False
    failed = False
    quarantined = False


class FakeBroker:
    def __init__(self, order):
        self.order = order
        self.health_snapshot = Health()
        self.close_result = True

    @property
    def quarantined(self):
        return self.health_snapshot.quarantined

    def close(self, *, wait=True, timeout=None):
        self.order.append('broker-close')
        return self.close_result


class FakeThread:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


class FakeBridge:
    def __init__(self, order):
        self.order = order
        self.quarantined = False
        self._thread = None
        self.close_result = True

    @property
    def dispatcher_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        self.order.append('bridge-start')
        self._thread = FakeThread()

    def close(self, timeout=3.0):
        self.order.append('bridge-close')
        if self.close_result:
            self._thread.alive = False
        return self.close_result


class FakeServer:
    def __init__(self, order):
        self.order = order
        self.custody_quarantined = False
        self.close_result = True
        self.address = ('127.0.0.1', 12345)
        self.broker_id = 'a' * 32
        self.team_tokens = {'a': 'credential'}

    def start(self):
        self.order.append('server-start')

    def stop_accepting(self, timeout):
        self.order.append('stop-accepting')
        return True

    def drain(self, timeout):
        self.order.append('drain-server')
        return True

    def close(self, timeout):
        self.order.append('server-close')
        return self.close_result


def make_lifecycle():
    order = []
    worker, broker = FakeWorker(), FakeBroker(order)
    bridge, server = FakeBridge(order), FakeServer(order)
    return _FairOwnerLifecycle(worker, broker, bridge, server,
                               owner_quarantined=lambda: False), order


def test_owner_latch_callback_is_mandatory():
    order = []
    try:
        _FairOwnerLifecycle(FakeWorker(), FakeBroker(order), FakeBridge(order), FakeServer(order))
        assert False, 'owner latch callback was not required'
    except TypeError:
        pass
    assert order == []


def run(lifecycle, stop, order):
    def announce(server):
        assert server.address == ('127.0.0.1', 12345)
        order.append('announce')

    def verify():
        order.append('verify-drain')
        return True

    return serve_owner(lifecycle.server_view, lifecycle.broker_view, lifecycle.worker, stop,
                       announce=announce, verify_drain=verify, poll_interval=.01,
                       shutdown_timeout=.5)


def test_normal_lifecycle_closes_server_then_bridge_then_actual_broker():
    lifecycle, order = make_lifecycle()
    stop = Event()
    stop.set()
    result = run(lifecycle, stop, order)
    assert result.status is OwnerStatus.CLEAN and result.teardown_verified
    assert order == ['bridge-start', 'server-start', 'announce', 'server-close',
                     'bridge-close', 'broker-close', 'verify-drain']


def test_undrained_server_never_closes_bridge_or_broker_or_attests():
    lifecycle, order = make_lifecycle()
    lifecycle.server.close_result = False
    stop = Event()
    stop.set()
    result = run(lifecycle, stop, order)
    assert not result.teardown_verified
    assert 'bridge-close' not in order and 'broker-close' not in order
    assert 'verify-drain' not in order


def test_undrained_bridge_never_closes_broker_or_attests():
    lifecycle, order = make_lifecycle()
    lifecycle.bridge.close_result = False
    stop = Event()
    stop.set()
    result = run(lifecycle, stop, order)
    assert not result.teardown_verified
    assert 'bridge-close' in order and 'broker-close' not in order
    assert 'verify-drain' not in order


def test_bridge_or_server_quarantine_is_seen_without_a_connected_client():
    for component, attr in (('bridge', 'quarantined'), ('server', 'custody_quarantined')):
        lifecycle, order = make_lifecycle()
        stop, finished = Event(), Event()
        results = []
        def launch():
            results.append(run(lifecycle, stop, order))
            finished.set()
        thread = Thread(target=launch)
        thread.start()
        deadline = monotonic() + 2
        while 'announce' not in order and monotonic() < deadline:
            sleep(.005)
        assert 'announce' in order
        setattr(getattr(lifecycle, component), attr, True)
        assert finished.wait(2)
        thread.join(1)
        assert results[0].status is OwnerStatus.QUARANTINED
        assert not results[0].teardown_verified
        assert 'verify-drain' not in order


def test_lost_dispatcher_is_quarantined_before_clean_shutdown():
    lifecycle, order = make_lifecycle()
    stop, ready = Event(), Event()
    results = []
    def announce(_server):
        ready.set()
    def launch():
        results.append(serve_owner(lifecycle.server_view, lifecycle.broker_view,
                                   lifecycle.worker, stop, announce=announce,
                                   verify_drain=lambda: True, poll_interval=.01,
                                   shutdown_timeout=.5))
    thread = Thread(target=launch)
    thread.start()
    assert ready.wait(2)
    lifecycle.bridge._thread.alive = False
    thread.join(2)
    assert not thread.is_alive()
    assert results[0].status is OwnerStatus.QUARANTINED
    assert not results[0].teardown_verified


def test_non_boolean_dispatcher_observation_fails_closed():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()

    class InvalidBridge:
        quarantined = False
        dispatcher_alive = 1

    lifecycle.bridge = InvalidBridge()
    assert lifecycle.quarantined is True
    assert isinstance(lifecycle._errors[8], RuntimeError)


def test_return_boundary_interruption_latches_unknown_server_close():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()
    code = lifecycle.stop_server.__func__.__code__
    interrupted = False
    def trace(frame, event, arg):
        nonlocal interrupted
        if event == 'call' and frame.f_code is code:
            frame.f_trace_opcodes = True
        elif event == 'opcode' and frame.f_code is code and not interrupted:
            instruction = next((item for item in dis.get_instructions(code)
                                if item.offset == frame.f_lasti), None)
            if instruction and instruction.opname == 'STORE_FAST' and instruction.argval == 'stopped':
                interrupted = True
                raise KeyboardInterrupt('after server close returned')
        return trace
    sys.settrace(trace)
    try:
        try:
            lifecycle.server_view.close(timeout=1)
            assert False, 'the return boundary was not interrupted'
        except KeyboardInterrupt:
            pass
    finally:
        sys.settrace(None)
    assert interrupted and lifecycle.quarantined
    assert not lifecycle.broker_view.close(timeout=1)
    assert 'broker-close' not in order


def test_blocked_bridge_close_cannot_retire_broker_after_timeout():
    lifecycle, order = make_lifecycle()
    entered, release, owner_unsafe = Event(), Event(), Event()
    lifecycle._owner_quarantined = owner_unsafe.is_set
    def blocked_close(timeout):
        order.append('bridge-close')
        entered.set()
        release.wait(3)
        lifecycle.bridge._thread.alive = False
        return True
    lifecycle.bridge.close = blocked_close
    stop = Event()
    stop.set()
    results = []
    thread = Thread(target=lambda: results.append(run(lifecycle, stop, order)))
    thread.start()
    assert entered.wait(2)
    thread.join(2)
    assert not thread.is_alive()
    assert not results[0].teardown_verified
    owner_unsafe.set()
    release.set()
    deadline = monotonic() + 2
    while lifecycle._bridge_thread.is_alive() and monotonic() < deadline:
        sleep(.005)
    assert not lifecycle._bridge_thread.is_alive()
    assert 'broker-close' not in order and 'verify-drain' not in order


def test_quarantine_while_server_close_blocks_interrupts_watchdog_wait():
    lifecycle, order = make_lifecycle()
    entered, release = Event(), Event()
    def blocked_close(timeout):
        order.append('server-close')
        entered.set()
        release.wait(3)
        return True
    lifecycle.server.close = blocked_close
    stop = Event()
    stop.set()
    results = []
    thread = Thread(target=lambda: results.append(run(lifecycle, stop, order)))
    thread.start()
    assert entered.wait(2)
    lifecycle.bridge.quarantined = True
    thread.join(2)
    assert not thread.is_alive()
    assert results[0].status is OwnerStatus.QUARANTINED
    assert not results[0].teardown_verified
    release.set()
    deadline = monotonic() + 2
    while lifecycle._server_thread.is_alive() and monotonic() < deadline:
        sleep(.005)
    assert not lifecycle._server_thread.is_alive()
    assert 'broker-close' not in order and 'verify-drain' not in order


def test_unexpected_broker_exit_or_owner_latch_prevents_attestation():
    for failure in ('dead', 'closed', 'owner'):
        lifecycle, order = make_lifecycle()
        stop, ready = Event(), Event()
        results = []
        thread = Thread(target=lambda: results.append(serve_owner(
            lifecycle.server_view, lifecycle.broker_view, lifecycle.worker, stop,
            announce=lambda server: ready.set(), verify_drain=lambda: True,
            poll_interval=.01, shutdown_timeout=.5)))
        thread.start()
        assert ready.wait(2)
        if failure == 'dead':
            lifecycle.broker.health_snapshot = Health(owner_alive=False)
        elif failure == 'closed':
            lifecycle.broker.health_snapshot = Health(closed=True)
        else:
            lifecycle._owner_quarantined = lambda: True
        thread.join(2)
        assert not thread.is_alive()
        assert results[0].status is OwnerStatus.QUARANTINED
        assert not results[0].teardown_verified


def test_known_worker_failure_is_distinct_from_verified_teardown():
    lifecycle, order = make_lifecycle()
    stop, ready = Event(), Event()
    results = []
    thread = Thread(target=lambda: results.append(serve_owner(
        lifecycle.server_view, lifecycle.broker_view, lifecycle.worker, stop,
        announce=lambda server: ready.set(), verify_drain=lambda: True,
        poll_interval=.01, shutdown_timeout=.5)))
    thread.start()
    assert ready.wait(2)
    lifecycle.worker.poisoned = True
    lifecycle.worker.failed = True
    lifecycle.broker.health_snapshot = Health(failed=True, owner_alive=False)
    thread.join(2)
    assert not thread.is_alive()
    assert results[0].status is OwnerStatus.WORKER_FAILED
    assert results[0].teardown_verified
    assert order[-5:] == ['stop-accepting', 'drain-server', 'server-close',
                          'bridge-close', 'broker-close']


def test_ambiguous_start_keeps_graph_quarantined():
    lifecycle, order = make_lifecycle()
    def fail_start():
        order.append('server-start')
        raise RuntimeError('listener start outcome is unknown')
    lifecycle.server.start = fail_start
    stop = Event()
    stop.set()
    result = run(lifecycle, stop, order)
    assert not result.teardown_verified
    assert lifecycle.quarantined
    assert lifecycle._errors[2] is not None
    assert 'broker-close' not in order and 'verify-drain' not in order


def test_startup_fault_stops_before_listener_or_announcement():
    for fault in ('bridge', 'server'):
        lifecycle, order = make_lifecycle()
        unsafe = Event()
        lifecycle._owner_quarantined = unsafe.is_set
        if fault == 'bridge':
            original_start = lifecycle.bridge.start
        else:
            original_start = lifecycle.server.start

        def start_and_fault():
            original_start()
            unsafe.set()

        if fault == 'bridge':
            lifecycle.bridge.start = start_and_fault
        else:
            lifecycle.server.start = start_and_fault
        stop = Event()
        stop.set()
        result = run(lifecycle, stop, order)
        assert lifecycle.quarantined and not result.teardown_verified
        assert 'announce' not in order and 'verify-drain' not in order
        if fault == 'bridge':
            assert 'server-start' not in order


def test_dead_dispatcher_during_startup_prevents_listener():
    lifecycle, order = make_lifecycle()
    original_start = lifecycle.bridge.start

    def start_and_exit():
        original_start()
        lifecycle.bridge._thread.alive = False

    lifecycle.bridge.start = start_and_exit
    stop = Event()
    stop.set()
    result = run(lifecycle, stop, order)
    assert lifecycle.quarantined and not result.teardown_verified
    assert 'server-start' not in order and 'announce' not in order


def test_failure_custody_survives_interruption_during_first_error_store():
    lifecycle, order = make_lifecycle()
    refs, injected = [], Event()
    class Payload:
        pass
    def fail_close(timeout):
        payload = Payload()
        refs.append(weakref.ref(payload))
        error = RuntimeError('private close failure')
        error.payload = payload
        raise error
    lifecycle.server.close = fail_close
    code = lifecycle._fail.__func__.__code__
    first_store = next(item.offset for item in dis.get_instructions(code)
                       if item.opname == 'STORE_SUBSCR')
    def trace(frame, event, arg):
        if event == 'call' and frame.f_code is code:
            frame.f_trace_opcodes = True
        elif (event == 'opcode' and frame.f_code is code and frame.f_lasti == first_store
              and not injected.is_set()):
            injected.set()
            raise KeyboardInterrupt('interrupt original error retention')
        return trace
    stop = Event()
    stop.set()
    threading.settrace(trace)
    try:
        result = run(lifecycle, stop, order)
    finally:
        threading.settrace(None)
    assert injected.is_set() and not result.teardown_verified
    gc.collect()
    assert refs and refs[0]() is not None
    assert lifecycle._errors[0].payload is refs[0]()
    assert isinstance(lifecycle._errors[1], KeyboardInterrupt)


def test_interruption_before_failure_recorder_keeps_original_exception():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()
    refs = []
    class Payload:
        pass
    def fail(timeout):
        payload = Payload()
        refs.append(weakref.ref(payload))
        error = RuntimeError('private server failure')
        error.payload = payload
        raise error
    lifecycle.server.stop_accepting = fail
    code = lifecycle._fail.__func__.__code__
    lines, start = inspect.getsourcelines(lifecycle._fail.__func__)
    target = start + next(i for i, line in enumerate(lines) if line.strip() == 'try:')
    injected = False
    def trace(frame, event, arg):
        nonlocal injected
        if frame.f_code is code and event == 'line' and frame.f_lineno == target and not injected:
            injected = True
            raise KeyboardInterrupt('before failure recorder')
        return trace
    sys.settrace(trace)
    try:
        try:
            lifecycle.server_view.stop_accepting(timeout=1)
            assert False, 'failure was not propagated'
        except KeyboardInterrupt:
            pass
    finally:
        sys.settrace(None)
    gc.collect()
    assert injected and lifecycle.quarantined
    assert refs[0]() is not None and lifecycle._errors[0].payload is refs[0]()
    assert isinstance(lifecycle._errors[1], KeyboardInterrupt)


def test_no_new_server_operation_is_issued_after_quarantine():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()
    calls, refs = [], []
    class Payload:
        pass
    def fail(timeout):
        calls.append('stop')
        payload = Payload()
        refs.append(weakref.ref(payload))
        error = RuntimeError('server failure')
        error.payload = payload
        raise error
    lifecycle.server.stop_accepting = fail
    try:
        lifecycle.server_view.stop_accepting(timeout=1)
        assert False, 'failure was not propagated'
    except RuntimeError:
        pass
    assert lifecycle.server_view.stop_accepting(timeout=1) is False
    assert lifecycle.server_view.drain(timeout=1) is False
    assert lifecycle.server_view.close(timeout=1) is False
    gc.collect()
    assert calls == ['stop'] and refs[0]() is not None


def test_simultaneous_health_and_server_failures_keep_separate_roots():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()
    entered, release = Event(), Event()
    refs = {}
    class Payload:
        pass
    class BrokenWorker:
        @property
        def quarantined(self):
            payload = Payload()
            refs['health'] = weakref.ref(payload)
            error = RuntimeError('health probe failed')
            error.payload = payload
            raise error
    def fail(timeout):
        payload = Payload()
        refs['server'] = weakref.ref(payload)
        entered.set()
        release.wait(2)
        error = RuntimeError('server failure')
        error.payload = payload
        raise error
    lifecycle.server.stop_accepting = fail
    def serve():
        try:
            lifecycle.server_view.stop_accepting(timeout=2)
        except RuntimeError:
            pass
    thread = Thread(target=serve)
    thread.start()
    assert entered.wait(2)
    lifecycle.worker = BrokenWorker()
    assert lifecycle.quarantined
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    gc.collect()
    assert refs['health']() is not None and refs['server']() is not None
    assert lifecycle._errors[8].payload is refs['health']()
    assert lifecycle._errors[0].payload is refs['server']()


def test_overlapping_healthy_observations_do_not_quarantine():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()
    entered, release, lock = Event(), Event(), Lock()
    calls = []
    class SlowWorker:
        @property
        def quarantined(self):
            with lock:
                calls.append(1)
                first = len(calls) == 1
            if first:
                entered.set()
                release.wait(2)
            return False
    lifecycle.worker = SlowWorker()
    values = []
    thread = Thread(target=lambda: values.append(lifecycle.quarantined))
    thread.start()
    assert entered.wait(2)
    assert lifecycle.quarantined is False
    release.set()
    thread.join(2)
    assert not thread.is_alive() and values == [False]
    assert not lifecycle._unsafe


def test_health_probe_capacity_exhaustion_fails_closed_without_waiting():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()
    all_entered, release, lock = Event(), Event(), Lock()
    calls = []
    class SlowWorker:
        @property
        def quarantined(self):
            with lock:
                calls.append(1)
                if len(calls) == 4:
                    all_entered.set()
            release.wait(2)
            return False
    lifecycle.worker = SlowWorker()
    threads = [Thread(target=lambda: lifecycle.quarantined) for _ in range(4)]
    for thread in threads:
        thread.start()
    assert all_entered.wait(2)
    assert lifecycle.quarantined is True
    release.set()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()


def test_older_healthy_snapshot_cannot_overwrite_a_newer_failure():
    lifecycle, order = make_lifecycle()
    lifecycle.server_view.start()
    entered, release, lock = Event(), Event(), Lock()
    class OrderedBroker:
        quarantined = False
        def __init__(self):
            self.calls = 0
        @property
        def health_snapshot(self):
            with lock:
                self.calls += 1
                call = self.calls
            if call == 1:
                entered.set()
                release.wait(2)
                return Health()
            return Health(failed=call == 2)
    lifecycle.broker = OrderedBroker()
    values = []
    thread = Thread(target=lambda: values.append(lifecycle.broker_view.available))
    thread.start()
    assert entered.wait(2)
    assert lifecycle.broker_view.available is False
    release.set()
    thread.join(2)
    assert not thread.is_alive() and values == [False]
    assert lifecycle.broker_view.failed is True
