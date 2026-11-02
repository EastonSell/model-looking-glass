"""CPU-only exercises of the fixed serial-service comparison core."""
import threading
import time
import gc
import weakref
from itertools import count

import pytest

from test_direct_concurrent_clients import live_cpu_owner
from tools import serial_service_throughput_v1 as throughput
from tools.serial_service_throughput_v1 import FixedSharedLoad, _validate_memory, run_live_service


class SyntheticMemory:
    """Point values for exercising evidence handling; not device measurements."""
    source = 'synthetic_cpu'

    def __init__(self):
        self.calls = []

    def begin_window(self, mode):
        self.calls.append(('begin', mode))
        return self.snapshot()

    def end_window(self, mode):
        self.calls.append(('end', mode))
        return self.snapshot()

    def snapshot(self):
        return {'device_total_bytes': 1000, 'device_free_bytes': 800,
                'process_allocated_bytes': 100, 'process_reserved_bytes': 200,
                'peak_allocated_bytes': 110, 'peak_reserved_bytes': 210}


def test_two_real_cpu_ipc_clients_against_one_serial_broker(live_cpu_owner):
    service, model, authority, budget, resident, workspace = live_cpu_owner
    model.release.set()
    probe = SyntheticMemory()
    run = FixedSharedLoad(service.client, teams=('alpha', 'beta'), memory_probe=probe)
    result = run.run()
    assert result['status'] == 'complete'
    assert result['owner_teardown_verified'] is False
    assert result['comparison_valid'] is True
    assert len(result['plan_sha256']) == 64 and result['shared_over_serial_rate_ratio'] > 0
    assert result['scope'] == 'one serial owner; no batching or device overlap asserted'
    assert [window['mode'] for window in result['windows']] == ['serial', 'shared']
    for window in result['windows']:
        assert window['summary']['attempted'] == 8
        assert window['summary']['succeeded'] == 8
        assert window['summary']['completed_input_positions'] == 16
        assert window['summary']['requests_per_second'] > 0
        assert window['summary']['input_positions_per_second'] > 0
        assert window['summary']['queue_wait_ns']['n'] == 8
        assert window['summary']['service_ns']['n'] == 8
        assert all(window['summary']['by_team'][team]['succeeded'] == 4 for team in ('alpha', 'beta'))
        assert window['memory']['source'] == 'synthetic_cpu'
        assert window['memory']['before_call_ns'][1] <= window['start_ns']
        assert window['end_ns'] <= window['memory']['after_call_ns'][0]
        assert len({row['job_id'] for row in window['rows']}) == 8
    assert model.calls == 18  # two warmups plus identical eight-request windows
    assert model.maximum == 1 and not authority.unsafe
    assert set(budget.active_leases()) == {resident, workspace}
    assert 'logits' not in repr(result) and '127.0.0.1' not in repr(result)
    assert probe.calls == [('begin', 'serial'), ('end', 'serial'),
                           ('begin', 'shared'), ('end', 'shared')]


def test_failed_submission_stops_new_work_and_requires_reconciliation():
    class Client:
        def submit(self, **kwargs):
            raise OSError('private transport detail')
    run = FixedSharedLoad(lambda team: Client(), teams=('alpha', 'beta'), memory_probe=SyntheticMemory())
    result = run.run()
    assert result['status'] == 'incomplete'
    assert result['requires_owner_reconciliation'] is True
    assert result['owner_teardown_verified'] is False
    assert result['comparison_valid'] is False
    assert result['windows'] == []
    assert run.failure_count > 0 and 'private transport detail' not in repr(result)
    with pytest.raises(RuntimeError):
        run.run()


def test_stalled_client_is_rooted_and_later_drainable():
    entered, release = threading.Event(), threading.Event()
    class Client:
        def submit(self, **kwargs):
            entered.set()
            release.wait(5)
            raise OSError('submission outcome unknown')
    run = FixedSharedLoad(lambda team: Client(), teams=('alpha', 'beta'),
                          memory_probe=SyntheticMemory(), timeout=0.05)
    try:
        result = run.run()
        assert entered.is_set()
        assert result['status'] == 'incomplete' and result['requires_owner_reconciliation'] is True
        assert any(thread.is_alive() for thread in run.threads)
    finally:
        release.set()
    assert run.drain_clients(timeout=5) is True


def test_memory_snapshot_rejects_missing_or_inconsistent_counters():
    sample = SyntheticMemory().snapshot()
    assert _validate_memory(sample) == sample
    for changes in ({'process_allocated_bytes': True}, {'device_free_bytes': 1001},
                    {'process_allocated_bytes': 201}, {'peak_reserved_bytes': 199},
                    {'unexpected': 1}):
        with pytest.raises(ValueError):
            _validate_memory({**sample, **changes})


def test_thread_start_failure_after_launch_does_not_submit(monkeypatch):
    submitted = []
    class Client:
        def submit(self, **kwargs):
            submitted.append(kwargs)
            raise AssertionError('no request should be submitted')
    original = threading.Thread.start
    def start_then_fail(thread):
        original(thread)
        raise RuntimeError('injected start return failure')
    monkeypatch.setattr(threading.Thread, 'start', start_then_fail)
    run = FixedSharedLoad(lambda team: Client(), teams=('alpha', 'beta'), memory_probe=SyntheticMemory())
    result = run.run()
    assert result['status'] == 'incomplete' and result['requires_owner_reconciliation'] is True
    assert run.drain_clients(timeout=5) is True and not submitted and run.failure_count > 0


def test_parent_interruption_before_go_releases_rooted_gate(monkeypatch):
    entered, captured, submitted = threading.Event(), [], []
    class Client:
        def submit(self, **kwargs):
            submitted.append(kwargs)
            raise AssertionError('no request should be submitted')
    run = FixedSharedLoad(lambda team: Client(), teams=('alpha', 'beta'),
                          memory_probe=SyntheticMemory())
    original_serve = run._serve
    def tracked_serve(streams, count, go):
        captured.append(go)
        entered.set()
        original_serve(streams, count, go)
    monkeypatch.setattr(run, '_serve', tracked_serve)
    original_clock = time.perf_counter_ns
    fired = []
    def interrupt_once():
        if threading.current_thread() is threading.main_thread() and not fired:
            assert entered.wait(5)
            fired.append(True)
            raise KeyboardInterrupt('injected before gate release')
        return original_clock()
    monkeypatch.setattr(throughput.time, 'perf_counter_ns', interrupt_once)
    try:
        with pytest.raises(KeyboardInterrupt, match='before gate release'):
            run.run()
        drained = run.drain_clients(timeout=0.1)
    finally:
        # Keep a failed predecessor from leaving a non-daemon thread stuck.
        for gate in captured:
            gate.set()
        run.drain_clients(timeout=5)
    assert drained and not submitted


def test_first_client_is_rooted_if_second_factory_fails():
    clients = []
    class Client:
        pass
    def factory(team):
        if team == 'beta':
            raise OSError('second factory failed')
        client = Client()
        clients.append(weakref.ref(client))
        return client
    run = FixedSharedLoad(factory, teams=('alpha', 'beta'), memory_probe=SyntheticMemory())
    result = run.run()
    gc.collect()
    assert result['status'] == 'incomplete'
    assert clients[0]() is not None


def test_live_gpu_entry_is_disabled_before_any_action():
    with pytest.raises(RuntimeError, match='not enabled|requires reviewed'):
        run_live_service(lambda: pytest.fail('owner must not start'))


def test_client_interruption_propagates_after_client_thread_is_joined():
    interruption = KeyboardInterrupt('private interruption')
    class Client:
        def submit(self, **kwargs):
            raise interruption
    run = FixedSharedLoad(lambda team: Client(), teams=('alpha', 'beta'), memory_probe=SyntheticMemory())
    with pytest.raises(KeyboardInterrupt) as caught:
        run.run()
    assert caught.value is interruption and run.failure_count > 0
    assert run.drain_clients(timeout=5) is True


def test_successful_rows_from_different_brokers_do_not_form_a_comparison(monkeypatch):
    jobs = count(1)
    class Handle:
        def __init__(self, team):
            self.team, self.job_id = team, next(jobs)
            self.broker_id = ('a' if team == 'alpha' else 'b') * 32
        def wait(self, timeout):
            stamp = time.perf_counter_ns()
            return {'status': 'succeeded', 'submitted_ns': stamp, 'ready_ns': stamp,
                    'started_ns': stamp, 'finished_ns': stamp, 'broker_id': self.broker_id,
                    'job_id': self.job_id}
        def close(self):
            pass
    class Client:
        def __init__(self, team): self.team = team
        def submit(self, **kwargs): return Handle(self.team)
    monkeypatch.setattr(throughput, '_result_identity',
                        lambda response, team: (response['broker_id'], response['job_id'], '0' * 64))
    run = FixedSharedLoad(Client, teams=('alpha', 'beta'), memory_probe=SyntheticMemory())
    result = run.run()
    assert result['status'] == 'invalid_comparison' and result['comparison_valid'] is False
    assert result['requires_owner_reconciliation'] is True
    assert result['shared_over_serial_rate_ratio'] is None


def test_non_synthetic_memory_provider_is_rejected_before_acquiring_clients():
    probe = SyntheticMemory()
    probe.source = 'live_owner_guarded'
    with pytest.raises(RuntimeError, match='not enabled'):
        FixedSharedLoad(lambda team: pytest.fail('client acquisition'),
                        teams=('alpha', 'beta'), memory_probe=probe)
