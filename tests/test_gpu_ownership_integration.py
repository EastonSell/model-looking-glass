"""Cross-process ownership tests with fake GPU callbacks, never CUDA."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import time
import unittest


GPU_A = "GPU-12345678-1234-5678-9abc-1234567890ab"
GPU_B = "GPU-abcdef12-3456-789a-bcde-f01234567890"

# This process models a GPU-owning service or isolated benchmark without
# importing torch or calling CUDA. Its only device operation is a file write.
OWNER = textwrap.dedent("""
    import json, os, sys, time
    from pathlib import Path
    from looking_glass.device_lease import DeviceLease
    from looking_glass.supervisor import OwnerSession

    action, callback_file, ready_file, release_file = sys.argv[1:]
    session = OwnerSession.from_environment()
    lock_path = session.device_lease_path
    if action == 'wrong_lock':
        lock_path = lock_path.with_name('wrong.lock')
    lease = DeviceLease.acquire(lock_path, timeout=5)
    try:
        session.check(lease)
    except Exception:
        raise SystemExit(27)
    if action == 'wrong_lock':
        Path(ready_file).write_text('fake runtime query ran', encoding='utf-8')
    uuid = 'GPU-abcdef12-3456-789a-bcde-f01234567890' if action == 'wrong_uuid' else 'GPU-12345678-1234-5678-9abc-1234567890ab'
    try:
        session.verify_runtime_binding(lease, runtime_uuid=uuid, device_count=1)
    except Exception:
        raise SystemExit(21)
    if action == 'swap_lease':
        lease.release()
        lease = DeviceLease.acquire(session.device_lease_path, timeout=5)
        try:
            session.attest_clean(lease)
        except Exception:
            raise SystemExit(22)
        raise SystemExit(0)
    Path(callback_file).write_text('fake callback ran', encoding='utf-8')
    if action == 'crash':
        os._exit(23)
    if action == 'uncertain_drain':
        raise SystemExit(24)
    if action == 'no_attestation':
        raise SystemExit(0)
    if action == 'malformed_attestation':
        session.attestation_path.write_text('{invalid', encoding='utf-8')
        raise SystemExit(0)
    if action == 'wrong_run_attestation':
        session.attest_clean(lease)
        data = json.loads(session.attestation_path.read_text(encoding='utf-8'))
        data['run_id'] = '0' * 32 if session.run_id != '0' * 32 else '1' * 32
        session.attestation_path.write_text(json.dumps(data), encoding='utf-8')
        raise SystemExit(0)
    if action == 'hold':
        Path(ready_file).write_text('ready', encoding='utf-8')
        deadline = time.monotonic() + 30
        while not Path(release_file).exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not Path(release_file).exists():
            raise SystemExit(25)
    session.attest_clean(lease)
    if action == 'scientific_failure_clean_drain':
        raise SystemExit(26)
""")

# Directly launched using the base interpreter so the supervised owner is an
# actual direct child on Windows venvs that use a launcher process.
SUPERVISOR = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    from looking_glass.device_identity import resolve_device_identity
    from looking_glass.supervisor import run_supervised

    root, callback, ready, release, outcome = sys.argv[1:]
    uuid = 'GPU-12345678-1234-5678-9abc-1234567890ab'
    identity = resolve_device_identity(root, uuid, physical_uuids=(uuid,))
    result = run_supervised(
        [sys.executable, '-c', %r, 'hold', callback, ready, release],
        device_lease_path=identity.lock_path,
        marker_path=identity.marker_path,
        identity=identity,
        timeout=30, lease_timeout=5, terminate_grace=3,
    )
    Path(outcome).write_text(json.dumps({'clean': result.clean, 'exit_code': result.exit_code}), encoding='utf-8')
""" % OWNER)


# Exercise the real service, IPC transport, and broker using a CPU-only fake
# worker and drain callback. The child's real CUDA entry points are blocked.
SERVICE_OWNER = textwrap.dedent("""
    import json, sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from threading import Event, Lock
    import torch
    assert not torch.cuda.is_initialized()
    def forbidden_cuda(*args, **kwargs):
        raise RuntimeError('real CUDA access is disabled in this test')
    torch.cuda._lazy_init = forbidden_cuda
    torch._C._cuda_init = forbidden_cuda
    torch._C._cuda_getDeviceCount = forbidden_cuda
    from looking_glass.broker import Broker
    from looking_glass.device_lease import DeviceLease
    from looking_glass.ipc import IPCClient, IPCServer
    from looking_glass.resident_worker import ForwardRequest, ForwardResult
    from looking_glass.service import serve_owner
    from looking_glass.supervisor import OwnerSession

    action, report_path = sys.argv[1:]
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=5)
    session.check(lease)
    session.verify_runtime_binding(
        lease, runtime_uuid='GPU-12345678-1234-5678-9abc-1234567890ab', device_count=1)

    class Cancelled(Exception):
        pass

    worker_started = Event()
    both_handles_active = Event()
    count_lock = Lock()
    count = 0

    class Worker:
        cancellation_exception = Cancelled
        poisoned = False
        quarantined = False
        failed = False
        closed = False

        def snapshot_request(self, request):
            return request

        def execute(self, request, cancel_event):
            worker_started.set()
            if not both_handles_active.wait(8):
                raise TimeoutError('both teams did not obtain handles while work was active')
            if action == 'science_failure':
                raise RuntimeError('synthetic scientific failure')
            return ForwardResult(request.request_id, torch.tensor([[0.0]], device='cpu'))

        def close(self):
            self.closed = True

    class Server(IPCServer):
        closed = False

        def close(self, timeout=5):
            self.closed = super().close(timeout=timeout)
            return self.closed

    worker = Worker()
    broker = Broker(worker, max_pending=4, max_pending_per_team=2)
    server = Server(broker, team_ids=('team-a', 'team-b'),
                    request_factory=lambda request_id, inputs, seed: ForwardRequest(
                        request_id, {'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.int64,
                                                               device='cpu')}, seed),
                    result_encoder=lambda result: {'ok': True}, io_timeout=5, job_timeout=10)
    stop = Event()
    outcomes = []

    def announce(server):
        def request(team):
            global count
            handle = IPCClient(server.address, server.team_tokens[team], io_timeout=5).submit(
                team_id=team, request_id=team, inputs={'input_ids': [[1]]})
            with count_lock:
                count += 1
                if count == 2:
                    if not worker_started.wait(5):
                        raise TimeoutError('worker did not start before both handles arrived')
                    both_handles_active.set()
            return handle.wait(10)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(request, ('team-a', 'team-b')))
        outcomes.extend(results)
        stop.set()

    def verify_drain():
        return server.closed and worker.closed and action != 'uncertain_drain'

    result = serve_owner(server, broker, worker, stop,
                         announce=announce, verify_drain=verify_drain,
                         poll_interval=0.01, shutdown_timeout=12)
    Path(report_path).write_text(json.dumps({
        'outcomes': [item['status'] for item in outcomes],
        'team_ids': [item['team_id'] for item in outcomes],
        'broker_ids': [item['broker_id'] for item in outcomes],
        'job_ids': [item['job_id'] for item in outcomes],
        'both_handles_active': both_handles_active.is_set(),
        'status': result.status.value,
        'teardown_verified': result.teardown_verified,
        'worker_closed': worker.closed, 'server_closed': server.closed,
    }), encoding='utf-8')
    if result.teardown_verified:
        session.attest_clean(lease)
    assert not torch.cuda.is_initialized()
    if action == 'science_failure':
        raise SystemExit(31)
    if not result.teardown_verified:
        raise SystemExit(32)
""")


class GPUOwnershipIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.supervisor = importlib.import_module("looking_glass.supervisor")
        except ModuleNotFoundError as exc:
            if exc.name == "looking_glass.supervisor":
                self.fail("the supervised ownership component is not implemented")
            raise
        self.identity_api = importlib.import_module("looking_glass.device_identity")
        self.lease_api = importlib.import_module("looking_glass.device_lease")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package_root = Path(importlib.import_module("looking_glass").__file__).resolve().parent.parent
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = GPU_A
        self.env["PYTHONPATH"] = os.pathsep.join(
            [str(self.package_root), sysconfig.get_path("purelib"), self.env.get("PYTHONPATH", "")]
        )
        self.identity = self.identity_api.resolve_device_identity(
            self.root, GPU_A, physical_uuids=(GPU_A,), environ=self.env,
        )
        self.callback = self.root / "callback"
        self.ready = self.root / "ready"
        self.release = self.root / "release"
        self.outcome = self.root / "outcome.json"

    def run_owner(self, action, callback=None, lease_timeout=2, timeout=10):
        return self.supervisor.run_supervised(
            [sys._base_executable, "-c", OWNER, action, str(callback or self.callback), str(self.ready), str(self.release)],
            device_lease_path=self.identity.lock_path,
            marker_path=self.identity.marker_path,
            identity=self.identity,
            timeout=timeout,
            lease_timeout=lease_timeout,
            terminate_grace=3,
            cwd=self.package_root,
            env=self.env,
        )

    def run_service_owner(self, action):
        return self.supervisor.run_supervised(
            [sys._base_executable, "-c", SERVICE_OWNER, action, str(self.outcome)],
            device_lease_path=self.identity.lock_path,
            marker_path=self.identity.marker_path,
            identity=self.identity,
            timeout=40, lease_timeout=3, terminate_grace=3,
            cwd=self.package_root, env=self.env,
        )

    def assert_successor_blocked(self):
        next_callback = self.root / "next-callback"
        with self.assertRaises(self.supervisor.UnsafeOwnerError):
            self.run_owner("clean", callback=next_callback)
        self.assertFalse(next_callback.exists())

    def start_owner_supervisor(self):
        process = subprocess.Popen(
            [sys._base_executable, "-c", SUPERVISOR, str(self.root), str(self.callback), str(self.ready), str(self.release), str(self.outcome)],
            cwd=self.package_root, env=self.env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        def cleanup():
            self.release.touch()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        self.addCleanup(cleanup)
        deadline = time.monotonic() + 15
        while not self.ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail(f"supervisor exited before owner was ready: {process.returncode}")
            time.sleep(0.02)
        self.assertTrue(self.ready.exists(), "owner was not ready before deadline")
        return process

    def test_wrong_runtime_uuid_fails_before_fake_gpu_callback(self):
        result = self.run_owner("wrong_uuid")
        self.assertFalse(result.clean)
        self.assertEqual(result.exit_code, 21)
        self.assertFalse(self.callback.exists())
        self.assertTrue(self.identity.marker_path.exists())
        self.assert_successor_blocked()

    def test_wrong_active_lock_rejected_before_fake_runtime_query(self):
        result = self.run_owner("wrong_lock")
        self.assertEqual(result.exit_code, 27)
        self.assertFalse(result.clean)
        self.assertFalse(self.ready.exists(), "fake runtime query must not run")
        self.assertFalse(self.callback.exists())
        self.assertTrue(self.identity.marker_path.exists())
        self.assert_successor_blocked()

    def test_success_without_attestation_is_not_clean(self):
        result = self.run_owner("no_attestation")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.clean)
        self.assertTrue(self.callback.exists())
        self.assertTrue(self.identity.marker_path.exists())
        self.assert_successor_blocked()

    def test_malformed_attestation_is_not_clean(self):
        result = self.run_owner("malformed_attestation")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.clean)
        self.assertTrue(self.identity.marker_path.exists())
        self.assert_successor_blocked()

    def test_wrong_run_attestation_is_not_clean(self):
        result = self.run_owner("wrong_run_attestation")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.clean)
        self.assertTrue(self.identity.marker_path.exists())
        self.assert_successor_blocked()

    def test_active_owner_timeout_keeps_quarantine(self):
        result = self.run_owner("hold", timeout=5)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.clean)
        self.assertTrue(self.ready.exists(), "the timed-out owner must have started work")
        self.assertTrue(self.callback.exists())
        self.assertTrue(self.identity.marker_path.exists())
        lease = self.lease_api.DeviceLease.acquire(self.identity.lock_path, timeout=5)
        lease.release()
        self.assert_successor_blocked()

    def test_real_service_science_failure_with_verified_fake_drain(self):
        result = self.run_service_owner("science_failure")
        report = json.loads(self.outcome.read_text(encoding="utf-8"))
        self.assertEqual(report["outcomes"], ["failed", "failed"])
        self.assertEqual(report["team_ids"], ["team-a", "team-b"])
        self.assertEqual(len(set(report["broker_ids"])), 1)
        self.assertEqual(len(set(report["job_ids"])), 2)
        self.assertTrue(report["both_handles_active"])
        self.assertEqual(report["status"], "clean")
        self.assertTrue(report["worker_closed"] and report["server_closed"])
        self.assertTrue(report["teardown_verified"])
        self.assertEqual(result.exit_code, 31)
        self.assertTrue(result.clean, "clean describes teardown, not task success")
        self.assertFalse(self.identity.marker_path.exists())
        self.assertTrue(self.run_owner("clean").clean)

    def test_real_service_uncertain_fake_drain_keeps_quarantine(self):
        result = self.run_service_owner("uncertain_drain")
        report = json.loads(self.outcome.read_text(encoding="utf-8"))
        self.assertEqual(report["outcomes"], ["succeeded", "succeeded"])
        self.assertEqual(report["team_ids"], ["team-a", "team-b"])
        self.assertEqual(len(set(report["broker_ids"])), 1)
        self.assertEqual(len(set(report["job_ids"])), 2)
        self.assertTrue(report["both_handles_active"])
        self.assertEqual(report["status"], "quarantined")
        self.assertTrue(report["worker_closed"] and report["server_closed"])
        self.assertFalse(report["teardown_verified"])
        self.assertEqual(result.exit_code, 32)
        self.assertFalse(result.clean)
        self.assertTrue(self.identity.marker_path.exists())
        self.assert_successor_blocked()

    def test_failed_science_with_fake_confirmed_drain_is_not_successful_work(self):
        result = self.run_owner("scientific_failure_clean_drain")
        self.assertEqual(result.exit_code, 26)
        self.assertTrue(result.clean, "clean describes device teardown only")
        self.assertTrue(self.callback.exists())
        self.assertFalse(self.identity.marker_path.exists())
        next_callback = self.root / "next-callback"
        successor = self.run_owner("clean", callback=next_callback)
        self.assertEqual(successor.exit_code, 0)
        self.assertTrue(successor.clean)
        self.assertTrue(next_callback.exists())

    def test_uncertain_drain_keeps_marker_and_blocks_successor(self):
        result = self.run_owner("uncertain_drain")
        self.assertEqual(result.exit_code, 24)
        self.assertFalse(result.clean)
        self.assertTrue(self.callback.exists())
        self.assertTrue(self.identity.marker_path.exists())
        next_callback = self.root / "next-callback"
        with self.assertRaises(self.supervisor.UnsafeOwnerError):
            self.run_owner("clean", callback=next_callback)
        self.assertFalse(next_callback.exists())

    def test_owner_crash_keeps_marker_and_releases_os_lock(self):
        result = self.run_owner("crash")
        self.assertEqual(result.exit_code, 23)
        self.assertFalse(result.clean)
        self.assertTrue(self.identity.marker_path.exists())
        lease = self.lease_api.DeviceLease.acquire(self.identity.lock_path, timeout=3)
        lease.release()
        with self.assertRaises(self.supervisor.UnsafeOwnerError):
            self.run_owner("clean")

    def test_simulated_owner_and_benchmark_cannot_overlap(self):
        process = self.start_owner_supervisor()
        self.assertTrue(self.callback.exists())
        self.assertTrue(self.identity.marker_path.exists())
        with self.assertRaises(self.lease_api.LeaseTimeoutError):
            self.lease_api.DeviceLease.acquire(self.identity.lock_path, timeout=0)
        benchmark_callback = self.root / "benchmark-callback"
        with self.assertRaises(self.lease_api.LeaseTimeoutError):
            self.run_owner("clean", callback=benchmark_callback, lease_timeout=0.2)
        self.assertFalse(benchmark_callback.exists())
        self.release.touch()
        self.assertEqual(process.wait(timeout=15), 0)
        outcome = json.loads(self.outcome.read_text(encoding="utf-8"))
        self.assertEqual(outcome, {"clean": True, "exit_code": 0})
        self.assertFalse(self.identity.marker_path.exists())
        result = self.run_owner("clean", callback=benchmark_callback)
        self.assertTrue(result.clean)
        self.assertTrue(benchmark_callback.exists())

    def test_killed_supervisor_stops_owner_and_preserves_quarantine(self):
        process = self.start_owner_supervisor()
        process.terminate()
        process.wait(timeout=10)
        self.assertTrue(self.identity.marker_path.exists())
        lease = self.lease_api.DeviceLease.acquire(self.identity.lock_path, timeout=5)
        lease.release()
        callback = self.root / "blocked-benchmark"
        with self.assertRaises(self.supervisor.UnsafeOwnerError):
            self.run_owner("clean", callback=callback)
        self.assertFalse(callback.exists())

    def test_reacquiring_a_different_lease_cannot_reuse_runtime_verification(self):
        result = self.run_owner("swap_lease")
        self.assertFalse(result.clean)
        self.assertEqual(result.exit_code, 22)
        self.assertTrue(self.identity.marker_path.exists())
        self.assertFalse(self.callback.exists())

    def test_marker_path_cannot_be_changed_for_the_same_device(self):
        with self.assertRaises((ValueError, self.supervisor.SupervisorError)):
            self.supervisor.run_supervised(
                [sys._base_executable, "-c", OWNER, "clean", str(self.callback), str(self.ready), str(self.release)],
                device_lease_path=self.identity.lock_path,
                marker_path=self.root / "alternate.unsafe.json",
                identity=self.identity,
                timeout=5, lease_timeout=1, cwd=self.package_root, env=self.env,
            )
        self.assertFalse(self.callback.exists())


if __name__ == "__main__":
    unittest.main()
