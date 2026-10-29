"""CPU-only tests for the private supervised-service control plane."""
import json
import os
from pathlib import Path
import sys
import sysconfig
import tempfile
import textwrap
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from looking_glass import gpu_service_control as control


TEAM_TOKENS = {"alpha": "a" * 64, "beta": "b" * 64}
BROKER_ID = "c" * 32


class ServiceControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = Path(__file__).resolve().parents[1]
        self.stop = self.root / "stop"
        self.device = self.root / "device.lock"
        self.marker = self.root / "device.unsafe.json"

    def ready(self):
        server = SimpleNamespace(address=("127.0.0.1", 40000), broker_id=BROKER_ID,
                                 team_tokens=TEAM_TOKENS)
        return control._encode_ready(server, ("alpha", "beta"))

    def test_ready_frame_binds_endpoint_model_and_distinct_team_tokens(self):
        frame = self.ready()
        address, broker, tokens = control._decode_ready(frame, ("alpha", "beta"))
        self.assertEqual(address, ("127.0.0.1", 40000))
        self.assertEqual((broker, tokens), (BROKER_ID, TEAM_TOKENS))
        for change in ({"address": ["0.0.0.0", 40000]}, {"device": "cpu"},
                       {"model_id": "other"}, {"broker_id": "c" * 31},
                       {"team_tokens": {"alpha": "a" * 64, "beta": "a" * 64}},
                       {"team_tokens": {"alpha": "a" * 64}},
                       {"extra": "not allowed"}):
            value = json.loads(frame)
            value.update(change)
            with self.subTest(change=list(change)), self.assertRaisesRegex(
                    ValueError, "invalid private service readiness frame"):
                control._decode_ready(json.dumps(value).encode(), ("alpha", "beta"))

    def test_bad_frames_do_not_echo_credentials_and_reject_duplicate_keys(self):
        secret = "sensitive-marker"
        for payload in (b"{\"schema\":1,\"schema\":2}",
                        ("{" + secret).encode(), b"[]", b"x" * 65537):
            with self.subTest(payload=payload[:10]), self.assertRaises(ValueError) as caught:
                control._decode_ready(payload, ("alpha", "beta"))
            self.assertNotIn(secret, str(caught.exception))
            self.assertIsNone(caught.exception.__context__)

    def child(self, *, invalid=False, ignore_stop=False, controller=None):
        program = """
            import json, os, secrets, sys, time
            from pathlib import Path
            from looking_glass.device_lease import DeviceLease
            from looking_glass.supervisor import OwnerSession
            from looking_glass.gpu_service_control import MODEL_ID, READY_SCHEMA
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            tokens = {name: secrets.token_hex(32) for name in ('alpha', 'beta')}
            assert all(token not in repr(sys.argv) and token not in repr(dict(os.environ))
                       for token in tokens.values())
            frame = {'schema': READY_SCHEMA, 'address': ['127.0.0.1', 40000],
                     'broker_id': 'c' * 32, 'team_tokens': tokens,
                     'model_id': MODEL_ID, 'device': 'cuda:0'}
            payload = b'bad' if sys.argv[2] == 'invalid' else json.dumps(frame).encode()
            session.send_startup(lease, payload)
            until = time.monotonic() + 12
            while time.monotonic() < until:
                if Path(sys.argv[1]).exists() and sys.argv[3] != 'ignore':
                    session.attest_clean(lease)
                    raise SystemExit(0)
                time.sleep(.02)
        """
        command = [sys._base_executable, "-c", textwrap.dedent(program), str(self.stop),
                   "invalid" if invalid else "valid", "ignore" if ignore_stop else "normal"]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.repo)
        env["CUDA_VISIBLE_DEVICES"] = ""
        controller = controller or control._prepare_owner_controller(self.stop)
        return control._start_owner_controller(
            controller, command, device_lease_path=self.device, marker_path=self.marker,
            identity=None, env=env, cwd=self.repo,
            teams=("alpha", "beta"), startup_timeout=5, owner_timeout=12,
            lease_timeout=3, terminate_grace=1)

    def test_real_cpu_supervisor_private_ready_and_clean_shutdown(self):
        owner = self.child()
        alpha = owner.client("alpha").token
        beta = owner.client("beta").token
        self.assertEqual((len(alpha), len(beta)), (64, 64))
        self.assertNotEqual(alpha, beta)
        self.assertNotIn(alpha, repr(owner))
        with self.assertRaises(RuntimeError):
            owner.client("missing")
        result = owner.shutdown(timeout=5)
        self.assertTrue(result.clean)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(self.marker.exists())
        with self.assertRaises(RuntimeError):
            owner.client("alpha")
        with self.assertRaises(RuntimeError):
            owner.shutdown(timeout=1)

    def test_bad_startup_aborts_child_and_preserves_unsafe_marker(self):
        with self.assertRaisesRegex(ValueError, "invalid private service readiness"):
            self.child(invalid=True)
        self.assertTrue(self.marker.exists())

    def test_shutdown_timeout_aborts_then_reports_actual_supervisor_result(self):
        owner = self.child(ignore_stop=True)
        with self.assertRaises(TimeoutError):
            owner.shutdown(timeout=.05)
        result = owner.wait(timeout=5)
        self.assertFalse(result.clean)
        self.assertTrue(self.marker.exists())

    def test_exited_owner_revokes_controller_credentials(self):
        owner = self.child()
        self.stop.touch()
        self.assertTrue(owner.wait(timeout=5).clean)
        with self.assertRaisesRegex(RuntimeError, 'credential is unavailable'):
            owner.client('alpha')
        self.assertEqual(owner._tokens, {})

    def test_interrupted_start_return_keeps_caller_recovery_handle(self):
        owner = control._prepare_owner_controller(self.stop)
        def interrupt(frame, event, arg):
            if frame.f_code is control._start_owner_controller.__code__ and event == 'return':
                raise KeyboardInterrupt()
            return interrupt
        sys.settrace(interrupt)
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.child(controller=owner)
        finally:
            sys.settrace(None)
        self.assertTrue(owner.shutdown(timeout=5).clean)
        self.assertFalse(self.marker.exists())

    def test_cancelling_real_owner_during_startup_keeps_unsafe_marker(self):
        program = """
            import time
            from looking_glass.device_lease import DeviceLease
            from looking_glass.supervisor import OwnerSession
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            time.sleep(10)
        """
        env = dict(os.environ)
        env['PYTHONPATH'] = str(self.repo)
        env['CUDA_VISIBLE_DEVICES'] = ''
        owner = control._prepare_owner_controller(self.stop)
        failures = []
        def start():
            try:
                control._start_owner_controller(owner,
                    [sys._base_executable, '-c', textwrap.dedent(program)],
                    device_lease_path=self.device, marker_path=self.marker,
                    identity=None, env=env, cwd=self.repo, teams=('alpha', 'beta'),
                    startup_timeout=8, owner_timeout=12, lease_timeout=3, terminate_grace=1)
            except BaseException as exc:
                failures.append(type(exc).__name__)
        thread = threading.Thread(target=start)
        thread.start()
        deadline = time.monotonic() + 5
        while not self.marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(self.marker.exists())
        try:
            outcome = owner.shutdown(timeout=5)
            self.assertFalse(outcome.clean)
        except RuntimeError:
            pass  # The supervisor can fail before receiving its startup frame.
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertTrue(owner._done.wait(5))
        self.assertTrue(self.marker.exists())
        self.assertNotIn(owner, control._CONTROLLERS)
        self.assertEqual(owner._tokens, {})

    def test_failed_thread_start_revokes_launch_without_retaining_slot(self):
        class FailedThread:
            ident = None
            def __init__(self, *args, **kwargs):
                pass
            def start(self):
                raise RuntimeError('synthetic thread start failure')
        owner = control._prepare_owner_controller(self.stop)
        with patch.object(control, 'Thread', FailedThread):
            with self.assertRaisesRegex(RuntimeError, 'thread start failure'):
                self.child(controller=owner)
        self.assertTrue(owner._done.is_set())
        self.assertNotIn(owner, control._CONTROLLERS)
        self.assertFalse(self.marker.exists())
        with self.assertRaisesRegex(RuntimeError, 'StartupCancelled'):
            owner.wait(timeout=1)

    def test_thread_start_then_raise_cannot_admit_untracked_work(self):
        class StartedThenFailed:
            def __init__(self, *args, **kwargs):
                self.thread = threading.Thread(*args, **kwargs)
            @property
            def ident(self):
                return self.thread.ident
            def start(self):
                self.thread.start()
                raise RuntimeError('start returned uncertain')
            def join(self, timeout):
                self.thread.join(timeout)
        owner = control._prepare_owner_controller(self.stop)
        with patch.object(control, 'Thread', StartedThenFailed):
            with self.assertRaisesRegex(RuntimeError, 'start returned uncertain'):
                self.child(controller=owner)
        self.assertTrue(owner._done.wait(5))
        self.assertEqual(owner._tokens, {})
        self.assertNotIn(owner, control._CONTROLLERS)
        if self.marker.exists():
            self.assertFalse(owner._result and owner._result.clean)

    def test_real_cpu_broker_serves_two_teams_through_private_controller(self):
        program = """
            import sys, threading, torch
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
                    self.config = SimpleNamespace(vocab_size=8, n_positions=4)
                def forward(self, input_ids):
                    return input_ids.float().unsqueeze(-1) + torch.tensor([0., 1.])
            stop = threading.Event()
            monitor, failed, errors = _monitor_stop(sys.argv[1], stop)
            def announce(server):
                session.send_startup(lease, _encode_ready(server, ('alpha', 'beta')))
            result = serve_cpu_model(TinyModel().eval(), model_id='test-only',
                team_ids=('alpha', 'beta'), stop_event=stop, announce=announce,
                max_tokens=2, shutdown_timeout=5)
            monitor.join(2)
            if result.teardown_verified:
                session.attest_clean(lease)
            if result.status is not OwnerStatus.CLEAN or not result.teardown_verified or failed[0]:
                raise SystemExit(31)
        """
        command = [sys._base_executable, '-c', textwrap.dedent(program), str(self.stop)]
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(self.repo), sysconfig.get_paths()['purelib']))
        env['CUDA_VISIBLE_DEVICES'] = ''
        owner = control._prepare_owner_controller(self.stop)
        control._start_owner_controller(owner, command, device_lease_path=self.device,
            marker_path=self.marker, identity=None, env=env, cwd=self.repo,
            teams=('alpha', 'beta'), startup_timeout=10, owner_timeout=25,
            lease_timeout=3, terminate_grace=2)
        try:
            from looking_glass.ipc import IPCRejected
            with self.assertRaises(IPCRejected):
                owner.client('alpha').submit(team_id='beta', request_id='bad-team',
                    inputs={'input_ids': [[1]]})
            barrier = threading.Barrier(2)
            def request(team, token):
                client = owner.client(team)
                barrier.wait(timeout=5)
                handle = client.submit(team_id=team, request_id='same',
                    inputs={'input_ids': [[token]]})
                return handle.wait(timeout=10)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(request, 'alpha', 1), pool.submit(request, 'beta', 2)]
                results = [future.result(timeout=12) for future in futures]
            self.assertEqual([item['status'] for item in results], ['succeeded', 'succeeded'])
            self.assertEqual([item['team_id'] for item in results], ['alpha', 'beta'])
            self.assertEqual([item['result']['logits'] for item in results], [[1.0, 2.0], [2.0, 3.0]])
            self.assertTrue(owner.shutdown(timeout=8).clean)
        finally:
            if not owner._done.is_set():
                owner._channel.abort()
                owner._done.wait(5)

    def test_stop_monitor_detects_signal_and_rejects_preexisting_marker(self):
        stop = threading.Event()
        thread, failed, errors = control._monitor_stop(self.stop, stop)
        self.stop.touch()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(stop.is_set())
        self.assertFalse(failed[0])
        self.assertEqual(errors, [])
        with self.assertRaises(ValueError):
            control._monitor_stop(self.stop, threading.Event())

    def test_invalid_stop_marker_fails_closed_and_reports_type(self):
        stop = threading.Event()
        thread, failed, errors = control._monitor_stop(self.stop, stop)
        self.stop.mkdir()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(stop.is_set())
        self.assertTrue(failed[0])
        self.assertEqual(errors, ["RuntimeError"])

    def test_stop_is_signalled_when_failure_diagnostics_cannot_be_allocated(self):
        class BadErrorName(type):
            def __getattribute__(cls, name):
                if name == '__name__':
                    raise MemoryError('diagnostic allocation')
                return super().__getattribute__(name)
        class BadError(Exception, metaclass=BadErrorName):
            pass
        stop = threading.Event()
        with patch.object(control, '_same_parent', side_effect=BadError()):
            thread, failed, errors = control._monitor_stop(self.stop, stop)
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(stop.is_set())
        self.assertTrue(failed[0])
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
