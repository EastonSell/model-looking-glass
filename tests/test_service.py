import threading
import time
import unittest
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

try:
    from looking_glass.service import OwnerStatus, make_gpt2_request_factory, serve_cpu_model, serve_owner
except ModuleNotFoundError as exc:
    if exc.name != 'looking_glass.service':
        raise
    serve_owner = None


class TestServer:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def close(self, timeout=0):
        self.stopped = True
        return True


class TestBroker:
    def __init__(self):
        self.available = True
        self.failed = False
        self.waited = None

    def close(self, *, wait=True, timeout=None):
        self.waited = wait
        self.available = False
        return True


class TestWorker:
    def __init__(self):
        self.poisoned = False
        self.quarantined = False

    @property
    def failed(self):
        return self.poisoned or self.quarantined


class OwnerTests(unittest.TestCase):
    def test_cpu_cli_announces_private_endpoint_and_handles_shutdown(self):
        from looking_glass.cpu_service import main
        import torch
        from torch import nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=8, n_positions=4)

            def forward(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1) + torch.tensor([0.0, 1.0])

        bundle = SimpleNamespace(model=TinyModel().eval(), metadata={'model_id': 'tiny',
                                                                      'model_revision': 'revision'})
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('looking_glass.cpu_service.load_distilgpt2', return_value=bundle), \
                patch('sys.stdin', io.StringIO('shutdown\n')), \
                patch('sys.stdout', stdout), patch('sys.stderr', stderr):
            code = main(['--cache-dir', 'unused', '--team', 'alpha'])
        self.assertEqual(code, 0, stderr.getvalue())
        endpoint = json.loads(stdout.getvalue())
        self.assertEqual(endpoint['address'][0], '127.0.0.1')
        self.assertEqual(set(endpoint['team_tokens']), {'alpha'})

    def test_cpu_cli_rejects_unsupported_length_before_loading_checkpoint(self):
        from looking_glass.cpu_service import main
        with patch('looking_glass.cpu_service.load_distilgpt2') as load, \
                patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--cache-dir', 'unused', '--team', 'alpha', '--max-tokens', '3'])
        load.assert_not_called()

    def test_real_cpu_service_serves_one_model_and_stops_cleanly(self):
        import torch
        from torch import nn
        from looking_glass.ipc import IPCClient

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=8, n_positions=4)

            def forward(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1) + torch.tensor([0.0, 1.0])

        stop, ready = threading.Event(), threading.Event()
        server_box, results = [], []

        def announce(server):
            server_box.append(server)
            ready.set()

        owner = threading.Thread(target=lambda: results.append(serve_cpu_model(
            TinyModel().eval(), model_id='tiny-test', team_ids=('alpha', 'beta'),
            stop_event=stop, announce=announce, max_tokens=2)), daemon=True)
        owner.start()
        try:
            self.assertTrue(ready.wait(5))
            server = server_box[0]
            a = IPCClient(server.address, server.team_tokens['alpha']).submit(
                team_id='alpha', request_id='same', inputs={'input_ids': [[1]]})
            b = IPCClient(server.address, server.team_tokens['beta']).submit(
                team_id='beta', request_id='same', inputs={'input_ids': [[2]]})
            self.assertEqual(a.wait(5)['result']['logits'], [1.0, 2.0])
            self.assertEqual(b.wait(5)['result']['logits'], [2.0, 3.0])
        finally:
            stop.set()
            owner.join(5)
        self.assertFalse(owner.is_alive())
        self.assertEqual(results[0].status, OwnerStatus.CLEAN)
        self.assertTrue(results[0].teardown_verified)

    def test_separate_client_receives_ordinary_worker_oom_before_owner_stops(self):
        import torch
        from torch import nn
        from looking_glass.broker import Broker
        from looking_glass.ipc import IPCServer
        from looking_glass.resident_worker import ResidentWorker
        from looking_glass.service import _CpuServiceWorker

        entered, release, ready, stop = (threading.Event() for _ in range(4))

        class GatedOomModel(nn.Module):
            def forward(self, input_ids):
                entered.set()
                if not release.wait(10):
                    raise RuntimeError('test gate timed out')
                raise torch.OutOfMemoryError('simulated CPU OOM')

        resident = ResidentWorker('cpu-failure-test', GatedOomModel().eval(), device='cpu')
        worker = _CpuServiceWorker(resident)
        broker = Broker(worker, max_pending=4, max_pending_per_team=2)
        server = IPCServer(broker, team_ids=['alpha'], job_timeout=10)
        owner_results = []
        owner = threading.Thread(target=lambda: owner_results.append(serve_owner(
            server, broker, worker, stop, announce=lambda _: ready.set(),
            verify_drain=lambda: True, poll_interval=0.001, shutdown_timeout=5)), daemon=True)
        child = None
        owner.start()
        try:
            self.assertTrue(ready.wait(5))
            client_code = '''
import json, sys
from looking_glass.ipc import IPCClient
spec = json.loads(sys.stdin.readline())
handle = IPCClient(tuple(spec['address']), spec['token']).submit(
    team_id='alpha', request_id='oom', inputs={'input_ids': [[1]]})
print(json.dumps(handle.wait(timeout=10)))
'''
            child = subprocess.Popen([sys.executable, '-c', client_code], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child.stdin.write(json.dumps({'address': server.address,
                                          'token': server.team_tokens['alpha']}) + '\n')
            child.stdin.close()
            child.stdin = None
            self.assertTrue(entered.wait(10))
            release.set()
            stdout, stderr = child.communicate(timeout=15)
            self.assertEqual(child.returncode, 0, stderr)
            outcome = json.loads(stdout)
            self.assertEqual(outcome['status'], 'failed')
            self.assertEqual(outcome['error'], 'WorkerOutOfMemory')
            self.assertEqual(outcome['team_id'], 'alpha')
            self.assertEqual(outcome['request_id'], 'oom')
            owner.join(5)
            self.assertFalse(owner.is_alive())
            self.assertEqual(owner_results[0].status, OwnerStatus.WORKER_FAILED)
            self.assertTrue(owner_results[0].teardown_verified)
        finally:
            release.set()
            stop.set()
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
            owner.join(5)
            server.close(timeout=5)
            broker.close(timeout=5)

    def test_model_specific_request_factory_rejects_out_of_range_ids(self):
        factory = make_gpt2_request_factory(vocab_size=5, max_positions=4,
                                            max_batch=1, max_tokens=4)
        request = factory('safe', {'input_ids': [[0, 4]], 'attention_mask': [[1, 1]]}, 0)
        self.assertEqual(request.inputs['input_ids'].tolist(), [[0, 4]])
        invalid = (
            {'input_ids': [[5]]},
            {'input_ids': [[0]], 'token_type_ids': [[5]]},
            {'input_ids': [[0]], 'position_ids': [[4]]},
            {'input_ids': [[0]], 'attention_mask': [[2]]},
            {'input_ids': [[0]], 'attention_mask': [[0]]},
            {'input_ids': [[0, 0, 0, 0, 0]]},
            {'input_ids': [[0]], 'position_ids': [[False]]},
        )
        for inputs in invalid:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                factory('unsafe', inputs, 0)

    def test_cpu_service_rejects_unencodable_model_length_before_inference(self):
        from torch import nn

        class WideModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=50257, n_positions=128)
                self.calls = 0

            def forward(self, input_ids):
                self.calls += 1
                raise AssertionError('must not execute')

        model = WideModel().eval()
        stop = threading.Event()
        stop.set()
        with self.assertRaises(ValueError):
            serve_cpu_model(model, model_id='wide', team_ids=('alpha',),
                            stop_event=stop, announce=lambda _: None,
                            max_tokens=3)
        self.assertEqual(model.calls, 0)

    def test_clean_stop_drains_before_reporting_clean(self):
        self.assertIsNotNone(serve_owner, 'owner service not implemented')
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        observed = []

        def verify_drain():
            observed.append((server.stopped, broker.waited))
            return True

        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=verify_drain, poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.CLEAN)
        self.assertTrue(result.teardown_verified)
        self.assertEqual(observed, [(True, True)])

    def test_quarantine_is_observed_without_any_client(self):
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        result = []
        thread = threading.Thread(target=lambda: result.append(
            serve_owner(server, broker, worker, stop, announce=lambda _: None,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=1)))
        thread.start()
        deadline = time.monotonic() + 2
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.001)
        worker.quarantined = True
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0].status, OwnerStatus.QUARANTINED)
        self.assertFalse(broker.waited)

    def test_failed_final_device_drain_is_not_clean(self):
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=lambda: False, poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)

    def test_poison_during_shutdown_is_not_reported_clean(self):
        worker = TestWorker()

        class PoisonOnClose(TestServer):
            def close(self, timeout=0):
                worker.poisoned = True
                return super().close(timeout=timeout)

        stop = threading.Event()
        stop.set()
        result = serve_owner(PoisonOnClose(), TestBroker(), worker, stop,
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.WORKER_FAILED)

    def test_broker_failure_during_shutdown_is_not_reported_clean(self):
        class FailingBroker(TestBroker):
            def close(self, *, wait=True, timeout=None):
                self.failed = True
                return super().close(wait=wait, timeout=timeout)

        stop = threading.Event()
        stop.set()
        result = serve_owner(TestServer(), FailingBroker(), TestWorker(), stop,
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.WORKER_FAILED)
        self.assertTrue(result.teardown_verified)

    def test_worker_failure_stops_admission_without_clean_attestation(self):
        class DrainServer(TestServer):
            def __init__(self):
                super().__init__()
                self.calls = []

            def stop_accepting(self, timeout=0):
                self.calls.append('stop_accepting')
                return True

            def drain(self, timeout=0):
                self.calls.append('drain')
                return True

            def close(self, timeout=0):
                self.calls.append('close')
                return super().close(timeout=timeout)

        server, broker, worker = DrainServer(), TestBroker(), TestWorker()
        worker.poisoned = True
        stop = threading.Event()
        verified = []
        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=lambda: (verified.append(True), True)[1], poll_interval=0.01,
                             shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.WORKER_FAILED)
        self.assertTrue(result.teardown_verified)
        self.assertTrue(server.stopped)
        self.assertEqual(server.calls, ['stop_accepting', 'drain', 'close'])
        self.assertEqual(verified, [True])

    def test_normal_worker_close_is_not_a_failure(self):
        worker = TestWorker()

        class CloseableWorker(TestWorker):
            @property
            def failed(self):
                return False

        worker = CloseableWorker()

        class CloseWorkerBroker(TestBroker):
            def close(self, *, wait=True, timeout=None):
                worker.poisoned = True  # admission is closed, but not failed
                return super().close(wait=wait, timeout=timeout)

        stop = threading.Event()
        stop.set()
        result = serve_owner(TestServer(), CloseWorkerBroker(), worker, stop,
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.CLEAN)
        self.assertTrue(result.teardown_verified)

    def test_unfinished_broker_shutdown_cannot_be_attested(self):
        class StuckBroker(TestBroker):
            def close(self, *, wait=True, timeout=None):
                return False

        server, broker, worker = TestServer(), StuckBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=0.1)
        self.assertEqual(result.status, OwnerStatus.SHUTDOWN_INCOMPLETE)

    def test_quarantine_interrupts_a_slow_shutdown(self):
        class SlowServer(TestServer):
            def __init__(self):
                super().__init__()
                self.closing = threading.Event()
                self.release = threading.Event()

            def close(self, timeout=0):
                self.closing.set()
                self.release.wait()
                return False

        server, broker, worker = SlowServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        result = []
        thread = threading.Thread(target=lambda: result.append(
            serve_owner(server, broker, worker, stop, announce=lambda _: None,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=3)))
        thread.start()
        self.assertTrue(server.closing.wait(2))
        try:
            worker.quarantined = True
            thread.join(0.5)
            self.assertFalse(thread.is_alive(), 'quarantine waited for slow shutdown')
            self.assertEqual(result[0].status, OwnerStatus.QUARANTINED)
        finally:
            server.release.set()
            thread.join(2)

    def test_quarantine_interrupts_a_blocked_startup_announcement(self):
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        announce_started, announce_release = threading.Event(), threading.Event()
        result = []

        def announce(_):
            announce_started.set()
            announce_release.wait()

        thread = threading.Thread(target=lambda: result.append(
            serve_owner(server, broker, worker, threading.Event(), announce=announce,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=3)))
        thread.start()
        self.assertTrue(announce_started.wait(2))
        try:
            worker.quarantined = True
            thread.join(0.5)
            self.assertFalse(thread.is_alive(), 'quarantine waited for blocked announcement')
            self.assertEqual(result[0].status, OwnerStatus.QUARANTINED)
        finally:
            announce_release.set()
            thread.join(2)

    def test_unfinished_startup_never_certifies_teardown(self):
        class SlowStart(TestServer):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def start(self):
                self.entered.set()
                self.release.wait()
                return super().start()

        server, broker, worker = SlowStart(), TestBroker(), TestWorker()
        stop = threading.Event()
        results = []
        owner = threading.Thread(target=lambda: results.append(
            serve_owner(server, broker, worker, stop, announce=lambda _: None,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=1)))
        owner.start()
        self.assertTrue(server.entered.wait(2))
        try:
            stop.set()
            owner.join(0.5)
            self.assertFalse(owner.is_alive())
            self.assertEqual(results[0].status, OwnerStatus.SHUTDOWN_INCOMPLETE)
            self.assertFalse(results[0].teardown_verified)
        finally:
            server.release.set()
            owner.join(2)

    def test_health_read_failure_cannot_be_attested(self):
        class UnknownHealth(TestWorker):
            @property
            def quarantined(self):
                raise RuntimeError('health unavailable')

            @quarantined.setter
            def quarantined(self, value):
                pass

        result = serve_owner(TestServer(), TestBroker(), UnknownHealth(), threading.Event(),
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertEqual(result.error_type, 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
