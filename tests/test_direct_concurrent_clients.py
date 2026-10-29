"""CPU-only checks of the gated concurrent direct-service client."""
import dis
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from looking_glass import direct_concurrent_clients as clients
from looking_glass import gpu_service
from looking_glass.broker import Broker, BrokerShutdownError
from looking_glass.budgeted_worker import WorkloadProfile
from looking_glass.fixed_service_request import FixedServiceRequestFactory
from looking_glass.ipc import IPCClient, IPCHandle, IPCServer, CancelAckObservation
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from looking_glass.owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
from looking_glass.resident_worker import ResidentWorker
from looking_glass.two_token_service import _PreparedService


@pytest.fixture
def contracts():
    with patch.object(gpu_service, 'DIRECT_SERVICE_OWNER_CONTRACT',
                      'two-team-direct-service-v1', create=True), \
         patch.object(gpu_service, 'OWNER_COMPLETION_CUSTODY_CONTRACT',
                      'service-result-retirement-v1', create=True):
        yield


def test_owner_gates_precede_client_acquisition():
    service = object.__new__(_PreparedService)
    service._teams = ('alpha', 'beta')
    service.client = lambda team: pytest.fail('client must not be acquired')
    with patch.object(gpu_service, 'DIRECT_SERVICE_OWNER_CONTRACT', None, create=True):
        with pytest.raises(clients.ConcurrentClientError):
            clients.ConcurrentClients(service)


def test_plan_rejects_unprofiled_seeds_and_excess_cancel_flags(contracts):
    service = object.__new__(_PreparedService)
    service._teams = ('alpha', 'beta')
    service.client = lambda team: pytest.fail('invalid plan acquired a client')
    for seeds, cancel in [((29, True), (False, False)), ((29, 42), (False, False)),
                          ((29, 43, 71), (False, False)), ((29, 43), (False, 0))]:
        with pytest.raises(ValueError):
            clients.ConcurrentClients(service, seeds=seeds, cancel=cancel)


def test_partial_thread_start_is_reported_as_unresolved_without_submitting(contracts):
    service = object.__new__(_PreparedService)
    service._teams = ('alpha', 'beta')
    service.client = lambda team: IPCClient(('127.0.0.1', 12345), 'test-secret')
    pilot = clients.ConcurrentClients(service)
    start = threading.Thread.start
    calls = []
    def partial_start(thread):
        calls.append(thread)
        if len(calls) == 2:
            raise MemoryError('private startup detail')
        return start(thread)
    with patch.object(threading.Thread, 'start', partial_start):
        with pytest.raises(MemoryError):
            pilot.start()
    report = pilot.wait(timeout=5)
    assert report is not None and report.requires_owner_reconciliation
    assert [row.status for row in report.requests] == ['unresolved', 'unresolved']
    assert pilot.wait_admitted('alpha', timeout=0.01) is False
    with pytest.raises(ValueError):
        pilot.wait_admitted('other', timeout=0.01)
    assert 'test-secret' not in repr(report) and 'private startup detail' not in repr(report)
    with pytest.raises(clients.ConcurrentClientError):
        pilot.start()


class BlockingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.active = self.maximum = self.calls = 0
        self.raise_oom = False

    def forward(self, input_ids):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.maximum = max(self.maximum, self.active)
        self.entered.set()
        if not self.release.wait(10):
            raise RuntimeError('test forward was not released')
        if self.raise_oom:
            raise torch.OutOfMemoryError('test-only private message')
        with self.lock:
            self.active -= 1
        return {'logits': torch.zeros((1, 2, 50257), dtype=torch.float32)}


@pytest.fixture
def live_cpu_owner(monkeypatch):
    import looking_glass.resident_worker as resident_module
    monkeypatch.setattr(resident_module, '_process_quarantined', threading.Event())
    monkeypatch.setattr(resident_module, '_process_quarantine_latch', None)
    monkeypatch.setattr(resident_module, '_quarantine_references', [])
    model = BlockingModel().eval()
    resident = ResidentWorker('a' * 64, model, device='cpu', max_batch_size=1,
                              max_sequence_length=2, max_input_bytes=64,
                              max_output_bytes=512 * 1024)
    budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                          max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    load = budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    workspace = budget.acquire(1024 * 1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    profile = WorkloadProfile('test', 'a' * 64, 'cpu', (1, 2),
                              (('input_ids', 'torch.int64'),), 512 * 1024, 1024 * 1024)
    authority = OwnerCompletionAuthority()
    worker = LifetimeWorkspaceWorker(resident, budget, load, workspace,
                                     profiles=(profile,), authority=authority)
    broker = Broker(worker, max_pending=2, max_pending_per_team=1,
                    completion_authority=authority)
    from looking_glass.service import make_gpt2_request_factory
    factory = FixedServiceRequestFactory(make_gpt2_request_factory(
        vocab_size=50257, max_positions=1024, max_batch=1, max_tokens=2))
    server = IPCServer(broker, team_ids=('alpha', 'beta'), request_factory=factory,
                       completion_authority=authority, max_connections=2,
                       max_request_bytes=4096, max_response_bytes=4 * 1024 * 1024,
                       max_batch=1, max_tokens=2, io_timeout=5, job_timeout=15).start()
    service = object.__new__(_PreparedService)
    service._teams = ('alpha', 'beta')
    service._test_server = server
    service.client = lambda team: IPCClient(server.address, server.team_tokens[team], io_timeout=5)
    try:
        yield service, model, authority, budget, load, workspace
    finally:
        model.release.set()
        server_closed = server.close(timeout=10)
        try:
            broker_closed = broker.close(timeout=10)
        except BrokerShutdownError:
            broker_closed = False
            broker._thread.join(5)
        if not authority.unsafe:
            assert server_closed is True and broker_closed is True
        else:
            assert not broker._thread.is_alive()


def test_real_two_client_queue_is_serial_and_returns_only_bounded_metadata(contracts, live_cpu_owner):
    service, model, authority, budget, load, workspace = live_cpu_owner
    pilot = clients.ConcurrentClients(service)
    pilot.start()
    assert model.entered.wait(5)
    assert all(pilot.wait_admitted(team, timeout=5) for team in service._teams)
    assert pilot.wait(timeout=0.01) is None
    model.release.set()
    report = pilot.wait(timeout=20)
    assert report is not None
    assert [row.status for row in report.requests] == ['succeeded', 'succeeded']
    assert report.serial_execution_observed is True
    assert report.queueing_observed is True
    assert report.fairness_under_backlog == 'not_measured'
    assert report.activation_trace == 'disabled'
    assert not report.requires_owner_reconciliation
    assert model.maximum == 1 and model.calls == 2
    assert not authority.unsafe
    assert set(budget.active_leases()) == {load, workspace}
    assert 'logits' not in repr(report) and '127.0.0.1' not in repr(report)
    assert all(row.float32_sha256 is not None for row in report.requests)


def test_wait_cannot_finalize_while_start_is_still_preparing_threads(contracts, live_cpu_owner):
    service, model, _, _, _, _ = live_cpu_owner
    entered, release, errors = threading.Event(), threading.Event(), []
    original = threading.Thread.start
    def gated_start(thread):
        if thread.name == 'looking-glass-direct-client' and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original(thread)
    pilot = clients.ConcurrentClients(service)
    def start_pilot():
        try:
            pilot.start()
        except BaseException as exc:
            errors.append(exc)
    with patch.object(threading.Thread, 'start', gated_start):
        starter = threading.Thread(target=start_pilot)
        starter.start()
        try:
            assert entered.wait(5)
            assert pilot.wait(timeout=0.05) is None
        finally:
            release.set()
        starter.join(5)
    assert not starter.is_alive() and not errors
    assert all(pilot.wait_admitted(team, timeout=5) for team in service._teams)
    model.release.set()
    report = pilot.wait(timeout=20)
    assert report is not None and not report.requires_owner_reconciliation
    assert [row.status for row in report.requests] == ['succeeded', 'succeeded']


def test_running_and_queued_cancellations_are_observed_separately(contracts, live_cpu_owner):
    service, model, _, _, _, _ = live_cpu_owner
    acks = {'alpha': threading.Event(), 'beta': threading.Event()}
    original = IPCHandle._capture_cancel_ack
    def capture(handle, accepted):
        original(handle, accepted)
        acks[handle.team_id].set()
    with patch.object(IPCHandle, '_capture_cancel_ack', capture):
        pilot = clients.ConcurrentClients(service)
        pilot.start()
        assert model.entered.wait(5)
        assert all(pilot.wait_admitted(team, timeout=5) for team in service._teams)
        pilot.cancel('alpha')
        pilot.cancel('beta')
        assert all(event.wait(5) for event in acks.values())
        model.release.set()
        report = pilot.wait(timeout=20)
    assert report is not None
    assert [row.status for row in report.requests] == ['cancelled', 'cancelled']
    assert [row.cancel_ack for row in report.requests] == [True, True]
    assert not report.requires_owner_reconciliation
    assert model.maximum == 1 and model.calls == 1


def test_report_waits_for_a_concurrent_cancel_send_failure(contracts, live_cpu_owner):
    service, model, _, _, _, _ = live_cpu_owner
    entered, release, failures = threading.Event(), threading.Event(), []
    def blocked_cancel(handle):
        entered.set()
        assert release.wait(10)
        raise OSError('private cancel failure')
    def cancel():
        try:
            pilot.cancel('alpha')
        except OSError:
            failures.append(True)
    with patch.object(IPCHandle, 'cancel', blocked_cancel):
        pilot = clients.ConcurrentClients(service)
        pilot.start()
        assert model.entered.wait(5)
        assert all(pilot.wait_admitted(team, timeout=5) for team in service._teams)
        sender = threading.Thread(target=cancel)
        sender.start()
        assert entered.wait(5)
        model.release.set()
        assert pilot.wait(timeout=20) is None
        release.set()
        sender.join(5)
        assert not sender.is_alive() and failures == [True]
        report = pilot.wait(timeout=5)
    assert report is not None and report.requires_owner_reconciliation
    assert [row.status for row in report.requests] == ['succeeded', 'succeeded']
    assert 'private cancel failure' not in repr(report)
    with pytest.raises(clients.ConcurrentClientError, match='already final'):
        pilot.cancel('alpha')
    assert pilot.wait(timeout=5) is report


def test_terminal_failure_and_timeout_require_reconciliation_without_leaking_error_text():
    slot = clients._Slot('alpha', 29, True)
    slot.handle = SimpleNamespace(broker_id='a' * 32, job_id=1,
        cancel_ack_observation=CancelAckObservation(True, 1),
        cancel_ack_observation_failed=False)
    failure = {'status': 'failed', 'submitted_ns': 1, 'ready_ns': 2,
               'started_ns': 3, 'finished_ns': 4, 'error': 'WorkerOutOfMemory'}
    result = clients._terminal(slot, failure)
    assert result.status == 'failed' and result.error == 'out_of_memory'
    assert result.cancel_ack is True
    result = clients._terminal(slot, {'status': 'timeout', 'error': 'private text'})
    assert result.status == 'unresolved' and result.error == 'timeout'
    assert 'private text' not in repr(result)


def test_cancel_capture_failure_and_invalid_source_times_are_not_accepted():
    slot = clients._Slot('alpha', 29, True)
    slot.handle = SimpleNamespace(broker_id='a' * 32, job_id=1,
        cancel_ack_observation=None, cancel_ack_observation_failed=True)
    outcome = {'status': 'cancelled', 'submitted_ns': 1, 'ready_ns': 2,
               'started_ns': 3, 'finished_ns': 4}
    with pytest.raises(clients.ConcurrentClientError):
        clients._terminal(slot, outcome)
    slot.handle.cancel_ack_observation_failed = False
    outcome['finished_ns'] = 2
    with pytest.raises(clients.ConcurrentClientError):
        clients._terminal(slot, outcome)


def test_real_worker_oom_never_becomes_client_success(contracts, live_cpu_owner):
    service, model, authority, budget, load, workspace = live_cpu_owner
    model.raise_oom = True
    pilot = clients.ConcurrentClients(service)
    pilot.start()
    assert model.entered.wait(5)
    assert all(pilot.wait_admitted(team, timeout=5) for team in service._teams)
    model.release.set()
    report = pilot.wait(timeout=20)
    assert report is not None and report.requires_owner_reconciliation
    assert all(row.status != 'succeeded' for row in report.requests)
    assert authority.unsafe
    assert set(budget.active_leases()) == {load, workspace}
    assert 'test-only private message' not in repr(report)


def test_rejection_after_factory_failure_still_requires_owner_reconciliation(contracts, live_cpu_owner):
    service, model, authority, budget, load, workspace = live_cpu_owner
    server = service._test_server
    factory = server.request_factory
    allow_failure = threading.Event()
    def faulty(request_id, inputs, seed):
        if seed == 43:
            assert allow_failure.wait(10)
            raise ValueError('private uncertain factory failure')
        return factory(request_id, inputs, seed)
    server.request_factory = faulty
    pilot = clients.ConcurrentClients(service, seeds=(29, 43))
    try:
        pilot.start()
        assert model.entered.wait(5)
        model.release.set()
        pilot._slots[0].thread.join(10)
        assert pilot._slots[0].observation.status == 'succeeded'
    finally:
        allow_failure.set()
    report = pilot.wait(timeout=20)
    assert report is not None and report.requires_owner_reconciliation
    assert [row.status for row in report.requests] == ['succeeded', 'rejected']
    assert report.requests[1].error == 'invalid_request'
    assert authority.unsafe
    assert set(budget.active_leases()) == {load, workspace}
    assert 'private uncertain factory failure' not in repr(report)


def test_lost_submit_return_is_unresolved_and_never_retried(contracts, live_cpu_owner):
    service, model, _, _, _, _ = live_cpu_owner
    submit = IPCClient.submit
    lost, calls, saved = threading.Event(), [], []
    def interrupted(client, **kwargs):
        calls.append(kwargs['team_id'])
        handle = submit(client, **kwargs)
        if kwargs['team_id'] == 'beta':
            saved.append(handle)
            lost.set()
            raise TimeoutError('private lost return')
        return handle
    with patch.object(IPCClient, 'submit', interrupted):
        pilot = clients.ConcurrentClients(service)
        pilot.start()
        assert model.entered.wait(5) and lost.wait(5)
        model.release.set()
        report = pilot.wait(timeout=20)
    assert report is not None and report.requires_owner_reconciliation
    assert report.requests[1].status == 'unresolved'
    assert sorted(calls) == ['alpha', 'beta']
    assert 'private lost return' not in repr(report)
    for handle in saved:
        handle.close()


def test_escaped_client_return_fault_cannot_be_a_clean_report(contracts, live_cpu_owner):
    service, model, _, _, _, _ = live_cpu_owner
    fired = threading.Event()
    offsets = {item.offset for item in dis.get_instructions(clients.ConcurrentClients._run)
               if item.opname == 'RETURN_VALUE'}
    def trace(frame, event, arg):
        if frame.f_code is clients.ConcurrentClients._run.__code__:
            if event == 'call':
                frame.f_trace_opcodes = True
            elif event == 'opcode' and frame.f_lasti in offsets and not fired.is_set():
                fired.set()
                raise KeyboardInterrupt('private client return fault')
        return trace
    previous = threading.gettrace()
    with patch.object(threading, 'excepthook', lambda args: None):
        threading.settrace(trace)
        try:
            pilot = clients.ConcurrentClients(service)
            pilot.start()
        finally:
            threading.settrace(previous)
        assert all(pilot.wait_admitted(team, timeout=5) for team in service._teams)
        model.release.set()
        report = pilot.wait(timeout=20)
    assert fired.is_set() and report is not None
    assert report.requires_owner_reconciliation
    assert any(row.status == 'unresolved' for row in report.requests)
    assert 'private client return fault' not in repr(report)
