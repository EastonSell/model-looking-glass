"""CPU-only process test of broker quarantine after a client disconnects.

The lease, marker, supervisor, broker and IPC are real. The GPU identity,
worker and device drain are simulated; no GPU allocation or claim is tested.
"""

import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import tempfile
import textwrap
import unittest


GPU_UUID = "GPU-12345678-1234-5678-9abc-1234567890ab"

CLIENT = textwrap.dedent("""
    import json, sys
    from looking_glass.ipc import IPCClient

    spec = json.loads(sys.stdin.readline())
    handle = IPCClient(tuple(spec['address']), spec['token'], io_timeout=5).submit(
        team_id='team-a', request_id='same-id', inputs={'input_ids': [[1]]})
    print('accepted', flush=True)
    if sys.stdin.readline().strip() != 'disconnect':
        raise RuntimeError('missing disconnect command')
    handle.close()
    print('disconnected', flush=True)
""")

OWNER = textwrap.dedent("""
    import gc, json, subprocess, sys, time, weakref
    from pathlib import Path
    from threading import Event
    import torch
    assert not torch.cuda.is_initialized()
    def forbidden_cuda(*args, **kwargs):
        raise RuntimeError('real CUDA access is disabled in this test')
    torch.cuda._lazy_init = forbidden_cuda
    torch._C._cuda_init = forbidden_cuda
    torch._C._cuda_getDeviceCount = forbidden_cuda
    from looking_glass.broker import Broker
    from looking_glass.device_lease import DeviceLease
    from looking_glass.host_budget import HostBudget
    from looking_glass.ipc import IPCServer
    from looking_glass.resident_worker import ForwardRequest, ForwardResult
    from looking_glass.service import serve_owner
    from looking_glass.supervisor import OwnerSession

    report_path, client_code = sys.argv[1:]
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=5)
    session.check(lease)
    session.verify_runtime_binding(
        lease, runtime_uuid='GPU-12345678-1234-5678-9abc-1234567890ab', device_count=1)

    budget = HostBudget(max_bytes=16, max_items=1, max_requests=1, max_operations=1)
    reservation = budget.open_request('trace-source')
    claim = reservation.reserve('worker', 'source', 16)
    started, release = Event(), Event()
    client_disconnected, cancel_observed = Event(), Event()

    class ClaimOwner:
        def __del__(self):
            claim.release()

    class Cancelled(Exception):
        pass

    class Worker:
        cancellation_exception = Cancelled
        poisoned = False
        quarantined = False
        failed = False
        closed = False
        alias = None
        claim_ref = None

        def snapshot_request(self, request):
            return ForwardRequest(request.request_id,
                                  {name: value.clone() for name, value in request.inputs.items()},
                                  request.seed)

        def execute(self, request, cancel_event):
            started.set()
            if not release.wait(8):
                raise RuntimeError('test release timed out')
            owner = ClaimOwner()
            self.claim_ref = weakref.ref(owner)
            logits = torch.ones(4, dtype=torch.float32)
            logits.owner = owner
            self.alias = logits.detach()
            # The wrong ID forces validation failure even though the class and
            # CPU tensor types look ordinary. The producer retains an alias.
            return ForwardResult('wrong-worker-id', logits)

        def close(self):
            self.closed = True

    worker = Worker()
    broker = Broker(worker, max_pending=2, max_pending_per_team=1)
    server = IPCServer(broker, team_ids=('team-a',),
                       request_factory=lambda request_id, inputs, seed: ForwardRequest(
                           request_id, {'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.int64)}, seed),
                       io_timeout=5, job_timeout=10)
    child = None

    def announce(server):
        global child
        child = subprocess.Popen([sys.executable, '-c', client_code], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            child.stdin.write(json.dumps({'address': server.address,
                                          'token': server.team_tokens['team-a']}) + '\\n')
            child.stdin.flush()
            if child.stdout.readline().strip() != 'accepted' or not started.wait(5):
                raise RuntimeError('client did not start an accepted forward')
            child.stdin.write('disconnect\\n')
            child.stdin.flush()
            if child.stdout.readline().strip() != 'disconnected':
                raise RuntimeError('client did not disconnect')
            client_disconnected.set()
            child.wait(timeout=5)
            if child.returncode != 0:
                raise RuntimeError('client exited unsuccessfully')
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with broker._condition:
                    running = broker._running
                    cancelled = running is not None and running._cancel.is_set()
                if cancelled:
                    cancel_observed.set()
                    break
                time.sleep(0.01)
            if not cancel_observed.is_set():
                raise RuntimeError('IPC disconnect did not request cancellation')
        finally:
            release.set()
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    outcome = serve_owner(server, broker, worker, Event(), announce=announce,
                          verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=12)
    gc.collect()
    report = {
        'status': outcome.status.value,
        'teardown_verified': outcome.teardown_verified,
        'broker_quarantined': broker.quarantined,
        'worker_closed': worker.closed,
        'client_disconnected': client_disconnected.is_set(),
        'cancel_observed': cancel_observed.is_set(),
        'claim_alive': worker.claim_ref is not None and worker.claim_ref() is not None,
        'claimed_bytes': budget.snapshot().used_bytes,
        'alias_values': worker.alias.tolist() if worker.alias is not None else None,
    }
    Path(report_path).write_text(json.dumps(report), encoding='utf-8')
    assert not torch.cuda.is_initialized()
    raise SystemExit(32)
""")


class ServiceQuarantineIntegrationTests(unittest.TestCase):
    def test_disconnected_client_cannot_hide_unknown_result_quarantine(self):
        from looking_glass.device_identity import resolve_device_identity
        from looking_glass.supervisor import UnsafeOwnerError, run_supervised

        package_root = Path(importlib.import_module('looking_glass').__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / 'report.json'
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = GPU_UUID
            env['PYTHONPATH'] = os.pathsep.join(
                (str(package_root), sysconfig.get_path('purelib'), env.get('PYTHONPATH', '')))
            identity = resolve_device_identity(root, GPU_UUID, physical_uuids=(GPU_UUID,), environ=env)
            result = run_supervised(
                [sys._base_executable, '-c', OWNER, str(report_path), CLIENT],
                device_lease_path=identity.lock_path, marker_path=identity.marker_path,
                identity=identity, timeout=30, lease_timeout=3, terminate_grace=3,
                cwd=package_root, env=env)
            self.assertEqual(result.exit_code, 32)
            self.assertFalse(result.clean)
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'quarantined')
            self.assertFalse(report['teardown_verified'])
            self.assertTrue(report['broker_quarantined'])
            self.assertFalse(report['worker_closed'])
            self.assertTrue(report['client_disconnected'] and report['cancel_observed'])
            self.assertTrue(report['claim_alive'])
            self.assertEqual(report['claimed_bytes'], 16)
            self.assertEqual(report['alias_values'], [1.0] * 4)
            self.assertTrue(identity.marker_path.exists())
            with self.assertRaises(UnsafeOwnerError):
                run_supervised([sys._base_executable, '-c', 'raise SystemExit(0)'],
                               device_lease_path=identity.lock_path, marker_path=identity.marker_path,
                               identity=identity, timeout=5, lease_timeout=3, terminate_grace=3,
                               cwd=package_root, env=env)


if __name__ == '__main__':
    unittest.main()
