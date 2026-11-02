"""CPU-only verification of the gated two-team service smoke."""
import hashlib
from pathlib import Path
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from looking_glass import two_team_service_smoke as smoke
from looking_glass.supervisor import SupervisionResult
from looking_glass.two_token_service import _PreparedService


class _Handle:
    def __init__(self, response, service):
        self.response, self.service = response, service
        self.events = service.events

    def wait(self, timeout):
        self.events.append(('wait', self.response['team_id'], timeout))
        return self.response

    def close(self):
        self.events.append(('close', self.response['team_id']))
        if self.service.close_error and self.response['team_id'] == 'alpha':
            raise self.service.close_error


class _Client:
    def __init__(self, team, service):
        self.team, self.service = team, service

    def submit(self, **request):
        self.service.events.append(('submit', request))
        assert request == {'team_id': self.team, 'request_id': smoke._REQUEST_ID,
                           'inputs': {'input_ids': [[15496, 995]]}, 'seed': 29}
        if self.service.fail_submit == self.team:
            raise RuntimeError('private failure detail')
        return _Handle(self.service.responses[self.team], self.service)


class _Service:
    def __init__(self, root):
        self.events = []
        self.unsafe_marker_path = root / 'unsafe'
        self.fail_submit = None
        self.start_error = None
        self.close_error = None
        self.shutdown_error = None
        self.wait_error = None
        self.result = SupervisionResult('run', 123, 0, True, False, 'clean')
        self.responses = {}
        for index, team in enumerate(('alpha', 'beta'), start=1):
            self.responses[team] = {
                'version': 1, 'status': 'succeeded', 'team_id': team,
                'request_id': smoke._REQUEST_ID, 'broker_id': 'a' * 32, 'job_id': index,
                'result': {'shape': [1, 2, 50257], 'dtype': 'torch.float32',
                           'logits': [0.0] * smoke._NUM_VALUES},
            }

    def start(self, **kwargs):
        self.events.append(('start', kwargs))
        if self.start_error:
            raise self.start_error

    def client(self, team):
        return _Client(team, self)

    def shutdown(self, *, timeout):
        self.events.append(('shutdown', timeout))
        if self.shutdown_error:
            raise self.shutdown_error
        return self.result

    def wait(self, *, timeout):
        self.events.append(('reconcile', timeout))
        if self.wait_error:
            raise self.wait_error
        return self.result


class TwoTeamSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service = _Service(Path(self.tmp.name))
        self.teams = ('alpha', 'beta')

    def test_absent_fair_owner_gate_precedes_any_start(self):
        prepared = object.__new__(_PreparedService)
        prepared.start = Mock()
        with patch.object(smoke._owner_module, 'FAIR_SERVICE_OWNER_CONTRACT', None, create=True):
            with self.assertRaisesRegex(smoke.ServiceSmokeError, 'fair service owner'):
                smoke.run_two_team_smoke(prepared)
        prepared.start.assert_not_called()
        with self.assertRaises(TypeError):
            smoke.run_two_team_smoke(self.service)

    def test_direct_smoke_requires_its_own_owner_and_completion_gates(self):
        prepared = object.__new__(_PreparedService)
        prepared.start = Mock()
        with patch.object(smoke._owner_module, 'FAIR_SERVICE_OWNER_CONTRACT',
                          'two-team-fair-service-v1', create=True), \
             patch.object(smoke._owner_module, 'DIRECT_SERVICE_OWNER_CONTRACT', None, create=True), \
             patch.object(smoke._owner_module, 'OWNER_COMPLETION_CUSTODY_CONTRACT',
                          'service-result-retirement-v1', create=True):
            with self.assertRaisesRegex(smoke.ServiceSmokeError, 'direct service owner'):
                smoke.run_direct_two_team_smoke(prepared)
        with patch.object(smoke._owner_module, 'DIRECT_SERVICE_OWNER_CONTRACT',
                          'two-team-direct-service-v1', create=True), \
             patch.object(smoke._owner_module, 'OWNER_COMPLETION_CUSTODY_CONTRACT', None, create=True):
            with self.assertRaisesRegex(smoke.ServiceSmokeError, 'completion custody'):
                smoke.run_direct_two_team_smoke(prepared)
        prepared.start.assert_not_called()
        with self.assertRaises(TypeError):
            smoke.run_direct_two_team_smoke(self.service)

    def test_direct_smoke_exercises_two_teams_without_claiming_fairness(self):
        prepared = object.__new__(_PreparedService)
        prepared._teams = self.teams
        prepared._identity = SimpleNamespace(marker_path=self.service.unsafe_marker_path)
        prepared.start, prepared.client = self.service.start, self.service.client
        prepared.shutdown, prepared.wait = self.service.shutdown, self.service.wait
        with patch.object(smoke._owner_module, 'DIRECT_SERVICE_OWNER_CONTRACT',
                          'two-team-direct-service-v1', create=True), \
             patch.object(smoke._owner_module, 'OWNER_COMPLETION_CUSTODY_CONTRACT',
                          'service-result-retirement-v1', create=True), \
             patch.object(smoke._owner_module, 'FAIR_SERVICE_OWNER_CONTRACT', None, create=True):
            result = smoke.run_direct_two_team_smoke(prepared)
            self.assertEqual(result.requests_succeeded, 2)
            self.assertTrue(result.teardown_verified)
            self.assertEqual([event[0] for event in self.service.events].count('submit'), 2)
            self.assertEqual(self.service.events[-1][0], 'shutdown')
        with patch.object(smoke._owner_module, 'FAIR_SERVICE_OWNER_CONTRACT',
                          'two-team-fair-service-v1', create=True):
            with self.assertRaisesRegex(smoke.ServiceSmokeError, 'already claimed'):
                smoke.run_two_team_smoke(prepared)

    def test_direct_smoke_uses_real_cpu_controller_and_loopback(self):
        # This supervised child uses a small CPU model and patched owner gates;
        # it does not validate GPU service startup or measured profiles.
        import os
        import sys
        import sysconfig
        import textwrap
        from looking_glass import gpu_service_control as control

        root = Path(self.tmp.name)
        stop, device, marker = root / 'stop', root / 'device.lock', root / 'device.unsafe.json'
        repo = Path(__file__).resolve().parents[1]
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
        owner = control._prepare_owner_controller(stop)
        prepared = object.__new__(_PreparedService)
        prepared._controller, prepared._teams = owner, self.teams
        prepared._identity = SimpleNamespace(marker_path=marker)
        startup_errors = []
        def start(**_kwargs):
            try:
                return control._start_owner_controller(owner, command, device_lease_path=device,
                    marker_path=marker, identity=None, env=env, cwd=repo, teams=self.teams,
                    startup_timeout=15, owner_timeout=45, lease_timeout=3, terminate_grace=2)
            except BaseException as exc:
                startup_errors.append((type(exc).__name__, str(exc)))
                raise
        prepared.start = start
        try:
            with patch.object(smoke._owner_module, 'DIRECT_SERVICE_OWNER_CONTRACT',
                              'two-team-direct-service-v1', create=True), \
                 patch.object(smoke._owner_module, 'OWNER_COMPLETION_CUSTODY_CONTRACT',
                              'service-result-retirement-v1', create=True):
                try:
                    result = smoke.run_direct_two_team_smoke(prepared)
                except smoke.ServiceSmokeError as exc:
                    self.fail(f'{exc}; startup={startup_errors}; owner={owner._error_type}; result={owner._result}')
            self.assertEqual(result.requests_succeeded, 2)
            self.assertEqual(result.float32_sha256, hashlib.sha256(bytes(smoke._NUM_VALUES * 4)).hexdigest())
            self.assertTrue(result.teardown_verified)
            self.assertFalse(marker.exists())
            self.assertTrue(owner._done.is_set())
        finally:
            if not owner._done.is_set():
                owner._channel.abort()
                owner._done.wait(5)

    def test_two_fixed_requests_succeed_only_after_verified_teardown(self):
        result = smoke._exercise(self.service, self.teams)
        self.assertEqual(result.requests_succeeded, 2)
        self.assertEqual(result.logits_shape, (1, 2, 50257))
        self.assertEqual(result.float32_sha256, hashlib.sha256(bytes(smoke._NUM_VALUES * 4)).hexdigest())
        self.assertTrue(result.teardown_verified)
        actions = [event[0] for event in self.service.events]
        self.assertLess(actions.index('submit'), actions.index('wait'))
        self.assertEqual(actions.count('submit'), 2)
        self.assertLess(max(i for i, action in enumerate(actions) if action == 'submit'),
                        min(i for i, action in enumerate(actions) if action == 'wait'))
        self.assertEqual(actions[-1], 'shutdown')
        self.assertNotIn('alpha', repr(result))

    def test_distinct_jobs_and_bitwise_equal_logits_are_required(self):
        for mutate in (
            lambda: self.service.responses['beta'].update(job_id=1),
            lambda: self.service.responses['beta'].update(broker_id='b' * 32),
            lambda: self.service.responses['beta']['result']['logits'].__setitem__(0, -0.0),
        ):
            with self.subTest(mutate=mutate):
                self.service = _Service(Path(self.tmp.name))
                mutate()
                with self.assertRaisesRegex(smoke.ServiceSmokeError, 'result validation'):
                    smoke._exercise(self.service, self.teams)
                self.assertEqual(self.service.events[-1][0], 'shutdown')

    def test_invalid_logits_are_rejected_and_owner_still_shuts_down(self):
        for value in (float('nan'), float('inf'), True, 1.0000000001):
            with self.subTest(value=value):
                self.service = _Service(Path(self.tmp.name))
                self.service.responses['alpha']['result']['logits'][0] = value
                with self.assertRaisesRegex(smoke.ServiceSmokeError, 'result validation'):
                    smoke._exercise(self.service, self.teams)
                self.assertEqual(self.service.events[-1][0], 'shutdown')

    def test_unclean_or_timed_out_owner_cannot_report_success(self):
        for result in (SupervisionResult('run', 123, 0, False, False, 'unsafe'),
                       SupervisionResult('run', 123, 1, True, False, 'failed'),
                       SupervisionResult('run', 123, False, True, False, 'malformed'),
                       SupervisionResult('run', 123, 0.0, True, False, 'malformed'),
                       SupervisionResult('run', 123, 0, True, True, 'timeout')):
            self.service.result = result
            with self.subTest(result=result), self.assertRaisesRegex(smoke.ServiceSmokeError, 'teardown'):
                smoke._exercise(self.service, self.teams)
        self.service.result = SupervisionResult('run', 123, 0, True, False, 'clean')
        self.service.unsafe_marker_path.touch()
        with self.assertRaisesRegex(smoke.ServiceSmokeError, 'teardown'):
            smoke._exercise(self.service, self.teams)

    def test_failed_submission_reports_sanitized_error_and_closes_owner(self):
        self.service.fail_submit = 'beta'
        with self.assertRaises(smoke.ServiceSmokeError) as caught:
            smoke._exercise(self.service, self.teams)
        self.assertIn('submission', str(caught.exception))
        self.assertNotIn('private failure detail', str(caught.exception))
        self.assertIn(('close', 'alpha'), self.service.events)
        self.assertEqual(self.service.events[-1][0], 'shutdown')

    def test_shutdown_exception_is_not_mistaken_for_success(self):
        self.service.shutdown_error = RuntimeError('private failure detail')
        with self.assertRaisesRegex(smoke.ServiceSmokeError, 'teardown'):
            smoke._exercise(self.service, self.teams)
        self.assertEqual(self.service.events[-1][0], 'reconcile')

    def test_interrupted_start_still_requests_shutdown(self):
        self.service.start_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            smoke._exercise(self.service, self.teams)
        self.assertEqual(self.service.events[-1][0], 'shutdown')

    def test_cleanup_interruptions_are_preserved_while_cleanup_continues(self):
        for kind in (KeyboardInterrupt, SystemExit):
            for phase in ('close', 'shutdown', 'wait'):
                with self.subTest(kind=kind.__name__, phase=phase):
                    self.service = _Service(Path(self.tmp.name))
                    interrupted = kind('interrupted')
                    if phase == 'close':
                        self.service.close_error = interrupted
                    else:
                        self.service.shutdown_error = (interrupted if phase == 'shutdown'
                                                       else RuntimeError('shutdown failed'))
                        if phase == 'wait':
                            self.service.wait_error = interrupted
                    with self.assertRaises(kind) as caught:
                        smoke._exercise(self.service, self.teams)
                    self.assertIs(caught.exception, interrupted)
                    self.assertIn(('close', 'alpha'), self.service.events)
                    self.assertIn(('close', 'beta'), self.service.events)
                    self.assertIn(('shutdown', 30), self.service.events)
                    if phase != 'close':
                        self.assertIn(('reconcile', 30), self.service.events)
                    detail = ' '.join(getattr(interrupted, '__notes__', ()))
                    if not detail and interrupted.__context__ is not None:
                        detail = str(interrupted.__context__)
                    self.assertIn('teardown was not verified', detail)

    def test_first_interruption_survives_later_cleanup_interruption(self):
        first = KeyboardInterrupt('start')
        self.service.start_error = first
        self.service.shutdown_error = SystemExit('shutdown')
        with self.assertRaises(KeyboardInterrupt) as caught:
            smoke._exercise(self.service, self.teams)
        self.assertIs(caught.exception, first)
        self.assertIn(('reconcile', 30), self.service.events)

    def test_second_smoke_cannot_shutdown_an_active_smoke(self):
        prepared = object.__new__(_PreparedService)
        prepared._teams = self.teams
        entered, release, completed = Event(), Event(), []
        def exercise(service, teams):
            entered.set()
            release.wait(3)
            completed.append((service, teams))
        with patch.object(smoke._owner_module, 'FAIR_SERVICE_OWNER_CONTRACT',
                          smoke._REQUIRED_FAIR_CONTRACT, create=True), \
             patch.object(smoke, '_exercise', side_effect=exercise):
            thread = Thread(target=smoke.run_two_team_smoke, args=(prepared,))
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaisesRegex(smoke.ServiceSmokeError, 'already claimed'):
                    smoke.run_two_team_smoke(prepared)
            finally:
                release.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(completed, [(prepared, self.teams)])


if __name__ == '__main__':
    unittest.main()
