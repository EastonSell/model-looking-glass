"""CPU checks for the disabled source-bound concurrent-owner composition."""
import hashlib
from pathlib import Path
from threading import Lock
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from looking_glass import gpu_service
from looking_glass import supervised_direct_concurrent as run_module
from looking_glass.direct_concurrent_clients import (ConcurrentClientReport, ConcurrentClients,
                                                      RequestObservation)
from looking_glass.gpu_service import ProfileFiles
from looking_glass.supervisor import SupervisionResult
from looking_glass.two_token_service import _PreparedService
from looking_glass.trusted_service_profile import TrustedServiceSource


_canonical_methods = run_module._canonical_methods


@pytest.fixture
def owner_contracts():
    with patch.object(gpu_service, 'DIRECT_SERVICE_OWNER_CONTRACT',
                      'two-team-direct-service-v1', create=True), \
         patch.object(gpu_service, 'OWNER_COMPLETION_CUSTODY_CONTRACT',
                      'service-result-retirement-v1', create=True), \
         patch.object(run_module, '_canonical_methods', return_value=True):
        yield


def prepared_stub(tmp_path):
    service = object.__new__(_PreparedService)
    service._teams = ('alpha', 'beta')
    service._repo = str(tmp_path)
    service._source = TrustedServiceSource(str(tmp_path), str(tmp_path), b'source', b'assets', ())
    service._profiles = tuple(ProfileFiles(tmp_path / f'p{i}', tmp_path / f'e{i}',
                                           tmp_path / f'm{i}', str(i) * 64)
                              for i in range(1, 4))
    service._identity = SimpleNamespace(marker_path=tmp_path / 'unsafe')
    service._controller = SimpleNamespace(_lock=Lock(), _claimed=True, shutdown=Mock(), wait=Mock())
    service._launch = object()
    service.start = Mock()
    service.shutdown, service.wait = service._controller.shutdown, service._controller.wait
    return service


def completed_clients(service):
    clients = ConcurrentClients(service)
    clients.start = Mock()
    clients.wait = Mock(return_value=ConcurrentClientReport(
        (RequestObservation('alpha', 29, 'cancelled'), RequestObservation('beta', 29, 'cancelled')),
        None, None, False))
    return clients


def test_unmeasured_host_envelope_blocks_owner_before_start(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    clients = ConcurrentClients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with pytest.raises(run_module.ConcurrentServiceError, match='host'):
        run.run()
    service.start.assert_not_called()
    service.shutdown.assert_not_called()


def test_changed_source_or_profile_binding_rejects_before_owner_start(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    clients = ConcurrentClients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    service._profiles = tuple(reversed(service._profiles))
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(run_module.ConcurrentServiceError, match='binding'):
            run.run()
    service.start.assert_not_called()


@pytest.mark.parametrize('target,method', [
    ('service', 'start'), ('service', 'client'), ('clients', 'start'), ('clients', 'wait'),
])
def test_instance_method_overrides_cannot_authorize_source_bound_run(owner_contracts, tmp_path, target, method):
    service = prepared_stub(tmp_path)
    del service.start
    clients = ConcurrentClients(service)
    assert _canonical_methods(service, clients)
    setattr(service if target == 'service' else clients, method, Mock())
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_canonical_methods', _canonical_methods), \
         patch.object(run_module, '_require_measured_host_envelope') as admission:
        with pytest.raises(run_module.ConcurrentServiceError, match='binding'):
            run.run()
    admission.assert_not_called()


def test_canonical_function_bound_to_another_service_is_rejected(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    del service.start
    clients = ConcurrentClients(service)
    other = object.__new__(_PreparedService)
    service.start = MethodType(_PreparedService.start, other)
    assert not _canonical_methods(service, clients)


def test_binding_change_after_start_stops_the_original_controller(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    original = service._controller
    original.shutdown.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    replacement = SimpleNamespace(_lock=Lock(), _claimed=True, shutdown=Mock(), wait=Mock())
    service.start.side_effect = lambda **kwargs: setattr(service, '_controller', replacement)
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(run_module.ConcurrentServiceError, match='binding'):
            run.run()
    original.shutdown.assert_called_once()
    replacement.shutdown.assert_not_called()
    clients.start.assert_not_called()
    expected = SupervisionResult('run', 123, 0, True, False, 'clean')
    original.wait.return_value = expected
    assert run.wait_supervisor(timeout=1) is expected
    replacement.wait.assert_not_called()


def test_actual_prepared_start_receives_pinned_profiles_and_source(owner_contracts, tmp_path):
    from looking_glass import two_token_service as service_module
    service = prepared_stub(tmp_path)
    del service.start  # Exercise the real _PreparedService.start method.
    service._identity = SimpleNamespace(workspace_root=tmp_path, lock_path=tmp_path / 'lock',
        marker_path=tmp_path / 'unsafe', validate_paths=Mock())
    service._launch = SimpleNamespace(command=('fixed-child',), environment={}, verify_ready=Mock())
    service.shutdown.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None), \
         patch.object(run_module, 'prepare_trusted_source', return_value=service._source), \
         patch.object(service_module, '_check_owner_contract'), \
         patch.object(service_module, '_source_checkout', return_value=tmp_path), \
         patch.object(service_module, '_workspace_root'), \
         patch.object(service_module, 'prepare_trusted_source', return_value=service._source), \
         patch.object(service_module, '_load_profile_registry') as registry, \
         patch.object(service_module, '_start_owner_controller') as start:
        result = run.run()
    assert result.teardown_verified and result.clients is clients.wait.return_value
    assert registry.call_args.args == (service._profiles, service._identity)
    assert start.call_args.args == (service._controller, service._launch.command)
    assert start.call_args.kwargs['identity'] is service._identity
    assert service._launch.verify_ready.call_count == 1
    service.shutdown.assert_called_once()


def test_client_timeout_requests_shutdown_without_publishing_success(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    service.shutdown.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    clients = completed_clients(service)
    clients.wait.return_value = None
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(run_module.ConcurrentServiceError, match='unresolved'):
            run.run(client_timeout=0.01)
    service.shutdown.assert_called_once()
    assert run._client_report is None


def test_clean_owner_does_not_erase_unresolved_client_status(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    service.shutdown.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    clients = completed_clients(service)
    clients.wait.return_value = ConcurrentClientReport(
        (RequestObservation('alpha', 29, 'succeeded'),
         RequestObservation('beta', 29, 'rejected', error='invalid_request')),
        None, None, True)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None), \
         patch.object(run_module, 'prepare_trusted_source', return_value=service._source):
        result = run.run()
    assert result.teardown_verified is True
    assert result.clients.requires_owner_reconciliation is True
    assert result.clients.requests[1].status == 'rejected'


def test_start_and_shutdown_failures_are_retained_and_supervisor_rechecked(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    first, shutdown = RuntimeError('private start'), OSError('private shutdown')
    service.start.side_effect = first
    service.shutdown.side_effect = shutdown
    service.wait.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(RuntimeError) as caught:
            run.run()
    assert caught.value is first
    assert run._run_failure is first and run._shutdown_failure is shutdown
    service.wait.assert_called_once()
    clients.start.assert_not_called()


def test_shutdown_interrupt_wins_over_ordinary_start_failure(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    first, interruption = RuntimeError('private start'), KeyboardInterrupt('private shutdown')
    service.start.side_effect = first
    service.shutdown.side_effect = interruption
    service.wait.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(KeyboardInterrupt) as caught:
            run.run()
    assert caught.value is interruption
    assert run._run_failure is first and run._shutdown_failure is interruption
    service.wait.assert_called_once()


def test_wait_interrupt_wins_over_ordinary_shutdown_failure(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    shutdown, interruption = OSError('private shutdown'), KeyboardInterrupt('private wait')
    service.shutdown.side_effect = shutdown
    service.wait.side_effect = interruption
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(KeyboardInterrupt) as caught:
            run.run()
    assert caught.value is interruption
    assert run._shutdown_failure is shutdown and run._wait_failure is interruption


def test_unsafe_marker_prevents_verified_teardown(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    service.shutdown.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    service._identity.marker_path.write_text('unsafe')
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(run_module.ConcurrentServiceError, match='teardown'):
            run.run()


def test_source_change_after_teardown_does_not_publish_result(owner_contracts, tmp_path):
    service = prepared_stub(tmp_path)
    service.shutdown.return_value = SupervisionResult('run', 123, 0, True, False, 'clean')
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    changed = TrustedServiceSource(str(tmp_path), str(tmp_path), b'changed', b'assets', ())
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None), \
         patch.object(run_module, 'prepare_trusted_source', return_value=changed):
        with pytest.raises(run_module.ConcurrentServiceError, match='source'):
            run.run()
    service.shutdown.assert_called_once()
    assert run._verification_failure is not None


@pytest.mark.parametrize('result', [
    SupervisionResult('run', 123, 1, True, False, 'nonzero'),
    SupervisionResult('run', 123, 0, False, False, 'unsafe'),
    SupervisionResult('run', 123, False, True, False, 'invalid'),
    None,
])
def test_unclean_supervision_never_publishes_result(owner_contracts, tmp_path, result):
    service = prepared_stub(tmp_path)
    service.shutdown.return_value = result
    clients = completed_clients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    with patch.object(run_module, '_require_measured_host_envelope', return_value=None):
        with pytest.raises(run_module.ConcurrentServiceError, match='teardown'):
            run.run()


def test_actual_supervisor_and_loopback_return_source_bound_summary(owner_contracts, tmp_path):
    # A fake CPU model and patched compatibility/host gates exercise the real
    # supervisor, IPC clients, server and broker, not GPU admission or profiling.
    import os
    import sys
    import sysconfig
    import textwrap
    from looking_glass import gpu_service_control as control

    service = prepared_stub(tmp_path)
    source = service._source
    repo = Path(__file__).resolve().parents[1]
    stop, device, marker = tmp_path / 'stop', tmp_path / 'device.lock', tmp_path / 'device.unsafe.json'
    program = r'''
import os, sys, threading, torch
from types import SimpleNamespace
from torch import nn
from looking_glass.device_lease import DeviceLease
from looking_glass.gpu_service_control import _encode_ready, _monitor_stop
from looking_glass.service import OwnerStatus, serve_cpu_model
from looking_glass.supervisor import OwnerSession
session = OwnerSession.from_environment()
lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
session.check(lease)
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=50257, n_positions=1024, use_cache=False)
    def forward(self, input_ids):
        return {'logits': torch.zeros((1, 2, 50257), dtype=torch.float32)}
stop = threading.Event()
monitor, failed, _errors = _monitor_stop(sys.argv[1], stop)
def announce(server):
    session.send_startup(lease, _encode_ready(server, ('alpha', 'beta')))
result = serve_cpu_model(TinyModel().eval(), model_id='test-only',
    team_ids=('alpha', 'beta'), stop_event=stop, announce=announce,
    max_tokens=2, max_pending=2, max_pending_per_team=1,
    max_connections=2, shutdown_timeout=5)
monitor.join(2)
if result.status is OwnerStatus.CLEAN and result.teardown_verified and not failed[0] and not monitor.is_alive():
    session.attest_clean(lease)
    os._exit(0)
os._exit(31)
'''
    command = [sys._base_executable, '-c', textwrap.dedent(program), str(stop)]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(repo), sysconfig.get_paths()['purelib']))
    env['CUDA_VISIBLE_DEVICES'] = ''
    controller = control._prepare_owner_controller(stop)
    service._controller = controller
    service._identity = SimpleNamespace(marker_path=marker)
    service.client = lambda team: controller.client(team)
    service.start = lambda **kwargs: control._start_owner_controller(
        controller, command, device_lease_path=device, marker_path=marker,
        identity=None, env=env, cwd=repo, teams=service._teams,
        startup_timeout=15, owner_timeout=45, lease_timeout=3, terminate_grace=2)
    service.shutdown = lambda *, timeout: controller.shutdown(timeout=timeout)
    service.wait = lambda *, timeout: controller.wait(timeout=timeout)
    clients = ConcurrentClients(service)
    run = run_module.SupervisedConcurrentRun(service, clients)
    try:
        with patch.object(run_module, '_require_measured_host_envelope', return_value=None), \
             patch.object(run_module, 'prepare_trusted_source', return_value=source):
            result = run.run(client_timeout=20, shutdown_timeout=15)
        assert result.teardown_verified is True
        assert [row.status for row in result.clients.requests] == ['succeeded', 'succeeded']
        assert result.clients.serial_execution_observed is True
        assert result.clients.fairness_under_backlog == 'not_measured'
        assert result.clients.activation_trace == 'disabled'
        assert result.source_base_sha256 == hashlib.sha256(b'source').hexdigest()
        assert result.approved_profile_ids == ('1' * 64, '2' * 64, '3' * 64)
        assert not marker.exists() and controller._done.is_set()
        assert 'test-only' not in repr(result) and str(marker) not in repr(result)
    finally:
        if not controller._done.is_set() and controller._channel is not None:
            controller._channel.abort()
            controller._done.wait(5)
