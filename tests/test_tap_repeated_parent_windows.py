"""CPU-only end-to-end exercise of the disabled parent/child barrier seam."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import pytest

from looking_glass import tap_repeated_parent as parent
from looking_glass.device_identity import resolve_device_identity
from looking_glass.supervisor import StartupChannel
from looking_glass.tap_repeated_activity import ParentActivity
from looking_glass import tap_repeated_protocol as protocol
from scripts.v2_source_guards import WindowsJob


_GPU = 'GPU-00000000-0000-0000-0000-000000000001'
_TOTAL = 8 * 1024**3


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='requires the pinned Windows QPC runtime')
@pytest.mark.parametrize('fail_after_query', [False, True])
def test_real_supervised_fake_owner_crosses_both_private_barriers(monkeypatch, tmp_path, fail_after_query):
    code = textwrap.dedent('''
        import json, os, pathlib, sys, time
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession, _supervisor_birth
        from looking_glass.tap_repeated_child_barrier import ChildBarriers, wait_frame

        folder = pathlib.Path(sys.argv[1])
        gpu = 'GPU-00000000-0000-0000-0000-000000000001'
        total = 8 * 1024**3
        challenge = 'b' * 64
        clock_info = time.get_clock_info('perf_counter')
        clock = {'name':'perf_counter_ns', 'monotonic':clock_info.monotonic,
                 'adjustable':clock_info.adjustable, 'implementation':clock_info.implementation,
                 'resolution_seconds':clock_info.resolution}
        runtime = {'platform':'win32', 'implementation':'cpython', 'python_version':'3.11.15'}
        session = OwnerSession.from_environment()
        lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
        session.verify_runtime_binding(lease, runtime_uuid=gpu, device_count=1)
        ready = {'schema':'tap-repeated-owner-ready-v1', 'session_index':2, 'owner_pid':os.getpid(),
                 'owner_birth':_supervisor_birth(), 'supervisor_run_id':session.run_id,
                 'challenge':challenge, 'gpu_uuid':gpu, 'cuda_total_bytes':total,
                 'ready_ns':time.perf_counter_ns(), 'clock':clock, 'runtime':runtime}
        barriers = ChildBarriers(ready)
        session.send_startup(lease, json.dumps(ready).encode())
        grant_raw, seen = wait_frame(folder / 'start-grant.json', timeout=10)
        barriers.accept_start(grant_raw, observed_ns=seen)
        lower = time.perf_counter_ns()
        upper = lower + 250_000_000
        barriers.authorize_preflight(lower, upper)
        start = time.perf_counter_ns()
        if not lower <= start <= upper:
            raise RuntimeError('preflight missed its authorized interval')
        rows = json.loads((folder / 'template.json').read_bytes())
        rows[0].update(gate_lower_ns=lower, gate_upper_ns=upper,
                       memory_snapshot_started_ns=seen, memory_snapshot_ended_ns=lower)
        offset = start - rows[0]['started_ns']
        for row in rows:
            for key in ('started_ns', 'ended_ns', 'completed_ns', 'completion_ns',
                        'guarded_started_ns', 'guarded_ended_ns', 'postcheck_completed_ns'):
                if key in row:
                    row[key] += offset
        completed = rows[-1]['completed_ns']
        while time.perf_counter_ns() <= completed:
            time.sleep(.001)
        with (folder / 'journal.jsonl').open('xb') as output:
            for row in rows:
                output.write((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\\n').encode())
            output.flush()
            os.fsync(output.fileno())
        barriers.publish_finished(folder / 'finished.json', preflight_started_ns=start,
                                  workload_ended_ns=completed)
        cleanup_raw, cleanup_seen = wait_frame(folder / 'cleanup-grant.json', timeout=10)
        barriers.accept_cleanup(cleanup_raw, observed_ns=cleanup_seen)
        session.attest_clean(lease)
        os._exit(0)
    ''')
    monkeypatch.setattr(parent, '_PARENT_LAUNCH_ENABLED', True)
    fixture_path = Path(__file__).with_name('test_tap_repeated_protocol.py')
    spec = importlib.util.spec_from_file_location('_repeated_protocol_fixture', fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    schedule = protocol.load_schedule(fixture.SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    rows = fixture._attempt(schedule, 2, binding)['journal']
    (tmp_path / 'template.json').write_text(json.dumps(rows), encoding='utf-8')
    identity = resolve_device_identity(tmp_path, _GPU, physical_uuids=(_GPU,), environ={})
    info = time.get_clock_info('perf_counter')
    clock_info = {'name': 'perf_counter_ns', 'monotonic': info.monotonic, 'adjustable': info.adjustable,
                  'implementation': info.implementation, 'resolution_seconds': info.resolution}
    runtime = {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}
    channel = StartupChannel(max_bytes=4096)
    job = WindowsJob()
    repo = Path(__file__).resolve().parent.parent
    env = {'SystemRoot': os.environ['SystemRoot'], 'PYTHONPATH': str(repo)}
    supervision = parent._Supervision([sys._base_executable, '-B', '-c', code, str(tmp_path)],
        channel=channel, identity=identity, device_lease_path=identity.lock_path,
        marker_path=identity.marker_path, cwd=repo, env=env, timeout=20, process_job=job)
    process_queries = 0
    def query(kind):
        nonlocal process_queries
        if kind == 'process':
            process_queries += 1
            if fail_after_query and process_queries == 2:
                raise OSError('synthetic after-query failure')
            if fail_after_query:
                return f'{_GPU},29999,foreign\n'.encode()
            return b''
        return f'{_GPU},8192,1024,7168,0\n'.encode()
    activity = ParentActivity(gpu_uuid=_GPU, query=query)
    journal = parent._PrivateJournal(tmp_path / 'activity.jsonl')
    try:
        result = parent._drive_session(session_index=2, challenge='b' * 64, identity=identity,
            channel=channel, supervision=supervision, activity=activity, journal=journal,
            attempt_dir=tmp_path, expected_smi_total=_TOTAL, schedule=schedule, parent_clock=clock_info,
            parent_runtime=runtime, exclusive=True, clock=time.perf_counter_ns, sleep=time.sleep,
            startup_timeout=8, workload_timeout=8, reap_timeout=5)
        assert result['supervision'].clean is True and result['supervision'].exit_code == 0
        if fail_after_query:
            assert result['state'] == 'incomplete'
            assert result['screen']['availability'] == 'unverified'
            assert result['screen']['status'] == 'contaminated'
            assert result['screen']['foreign_process_ids'] == [29999]
            assert result['screen_state'] == 'unverified' and result['summary'] is None
        else:
            assert result['state'] == 'pending_owner_evidence'
            assert result['summary']['warmup_samples'] == 6
            assert result['summary']['measured_samples'] == 36
            assert len(result['summary']['paired_rounds']) == 12
            assert result['screen']['availability'] == 'complete'
            assert result['screen_state'] == 'no_contamination_observed'
        assert not identity.marker_path.exists() and not identity.attestation_path.exists()
        assert not parent.pending_parent_attempts() and not parent.pending_activity_pollers()
    finally:
        channel.abort()
        job.close()


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='requires the pinned Windows QPC runtime')
def test_visible_grant_interruption_stops_real_fake_owner_and_retains_marker(monkeypatch, tmp_path):
    code = textwrap.dedent('''
        import json, os, pathlib, sys, time
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession, _supervisor_birth
        folder = pathlib.Path(sys.argv[1])
        gpu = 'GPU-00000000-0000-0000-0000-000000000001'
        clock_info = time.get_clock_info('perf_counter')
        clock = {'name':'perf_counter_ns', 'monotonic':clock_info.monotonic,
                 'adjustable':clock_info.adjustable, 'implementation':clock_info.implementation,
                 'resolution_seconds':clock_info.resolution}
        runtime = {'platform':'win32', 'implementation':'cpython', 'python_version':'3.11.15'}
        session = OwnerSession.from_environment()
        lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
        session.verify_runtime_binding(lease, runtime_uuid=gpu, device_count=1)
        ready = {'schema':'tap-repeated-owner-ready-v1', 'session_index':2, 'owner_pid':os.getpid(),
                 'owner_birth':_supervisor_birth(), 'supervisor_run_id':session.run_id,
                 'challenge':'b'*64, 'gpu_uuid':gpu, 'cuda_total_bytes':8*1024**3,
                 'ready_ns':time.perf_counter_ns(), 'clock':clock, 'runtime':runtime}
        session.send_startup(lease, json.dumps(ready).encode())
        deadline = time.monotonic() + 10
        while not (folder / 'start-grant.json').exists() and time.monotonic() < deadline:
            time.sleep(.005)
        if not (folder / 'start-grant.json').exists():
            raise RuntimeError('missing start grant')
        (folder / 'grant-seen').write_text('seen', encoding='ascii')
        while True:
            time.sleep(.05)
    ''')
    monkeypatch.setattr(parent, '_PARENT_LAUNCH_ENABLED', True)
    real_publish = parent._publish_barrier
    def publish_then_interrupt(path, payload):
        real_publish(path, payload)
        deadline = time.monotonic() + 2
        while not (tmp_path / 'grant-seen').exists() and time.monotonic() < deadline:
            time.sleep(.005)
        assert (tmp_path / 'grant-seen').exists()
        raise KeyboardInterrupt()
    monkeypatch.setattr(parent, '_publish_barrier', publish_then_interrupt)
    identity = resolve_device_identity(tmp_path, _GPU, physical_uuids=(_GPU,), environ={})
    info = time.get_clock_info('perf_counter')
    clock_info = {'name': 'perf_counter_ns', 'monotonic': info.monotonic, 'adjustable': info.adjustable,
                  'implementation': info.implementation, 'resolution_seconds': info.resolution}
    runtime = {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}
    channel, job = StartupChannel(max_bytes=4096), WindowsJob()
    repo = Path(__file__).resolve().parent.parent
    env = {'SystemRoot': os.environ['SystemRoot'], 'PYTHONPATH': str(repo)}
    supervision = parent._Supervision([sys._base_executable, '-B', '-c', code, str(tmp_path)],
        channel=channel, identity=identity, device_lease_path=identity.lock_path,
        marker_path=identity.marker_path, cwd=repo, env=env, timeout=15, process_job=job)
    activity = ParentActivity(gpu_uuid=_GPU,
        query=lambda kind: f'{_GPU},8192,1024,7168,0\n'.encode() if kind == 'gpu' else b'')
    journal = parent._PrivateJournal(tmp_path / 'activity.jsonl')
    try:
        with pytest.raises(KeyboardInterrupt):
            parent._drive_session(session_index=2, challenge='b' * 64, identity=identity,
                channel=channel, supervision=supervision, activity=activity, journal=journal,
                attempt_dir=tmp_path, expected_smi_total=_TOTAL, schedule={}, parent_clock=clock_info,
                parent_runtime=runtime, exclusive=True, clock=time.perf_counter_ns, sleep=time.sleep,
                startup_timeout=8, workload_timeout=8, reap_timeout=5)
        assert (tmp_path / 'grant-seen').exists()
        assert supervision.done and supervision._result is not None
        assert supervision._result.clean is False and supervision._result.exit_code is not None
        assert identity.marker_path.exists() and not identity.attestation_path.exists()
        assert not parent.pending_parent_attempts() and not parent.pending_activity_pollers()
    finally:
        channel.abort()
        job.close()
