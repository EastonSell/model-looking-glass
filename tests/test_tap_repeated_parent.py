import hashlib
import json
import os
import threading

import pytest

from looking_glass import tap_repeated_parent as parent
from looking_glass.device_identity import DeviceIdentityError, resolve_device_identity
from looking_glass.supervisor import SupervisionResult, _identity_spec, _marker, _supervisor_birth
from looking_glass.tap_repeated_activity import ParentActivity

_GPU = 'GPU-00000000-0000-0000-0000-000000000001'
_SMI_TOTAL = 1000 * 1024**2
_CUDA_TOTAL = _SMI_TOTAL - 327_680


def _clock():
    return {
        'name': 'perf_counter_ns', 'monotonic': True, 'adjustable': False,
        'implementation': 'QueryPerformanceCounter()', 'resolution_seconds': 1e-9,
    }


def _runtime():
    return {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}


def _ready():
    return {
        'schema': 'tap-repeated-owner-ready-v1', 'session_index': 2,
        'owner_pid': 1234, 'owner_birth': '123456789',
        'supervisor_run_id': 'a' * 32, 'challenge': 'b' * 64,
        'gpu_uuid': _GPU, 'cuda_total_bytes': _CUDA_TOTAL,
        'ready_ns': 100, 'clock': _clock(), 'runtime': _runtime(),
    }


def test_ready_frame_is_strict_and_bound_to_slot_and_challenge():
    raw = json.dumps(_ready()).encode()
    assert parent._decode_ready(raw, session_index=2, challenge='b' * 64) == _ready()
    with pytest.raises(ValueError):
        parent._decode_ready(raw, session_index=1, challenge='b' * 64)
    with pytest.raises(ValueError):
        parent._decode_ready(raw, session_index=2, challenge='c' * 64)
    with pytest.raises(ValueError):
        parent._decode_ready(raw[:-1] + b', "owner_pid": 4321}',
                             session_index=2, challenge='b' * 64)
    for field, value in [('owner_pid', True), ('owner_birth', '0'),
                         ('supervisor_run_id', 'A' * 32), ('ready_ns', True),
                         ('gpu_uuid', 'GPU-short'), ('cuda_total_bytes', True),
                         ('cuda_total_bytes', 0)]:
        row = _ready()
        row[field] = value
        with pytest.raises(ValueError):
            parent._decode_ready(json.dumps(row).encode(), session_index=2, challenge='b' * 64)


def test_clock_handshake_rejects_reversed_or_different_clock():
    ready = _ready()
    parent._check_ready_clock(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=120)
    with pytest.raises(ValueError):
        parent._check_ready_clock(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=99)
    with pytest.raises(ValueError):
        parent._check_ready_clock(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=101)
    other = _clock()
    other['implementation'] = 'different-clock'
    with pytest.raises(ValueError):
        parent._check_ready_clock(ready, parent_clock=other, parent_runtime=_runtime(), received_ns=120)
    with pytest.raises(ValueError):
        parent._check_ready_clock(ready, parent_clock=_clock(),
                                  parent_runtime=dict(_runtime(), python_version='3.12.0'), received_ns=120)


def test_ready_must_match_the_supervisors_registered_child(monkeypatch):
    class Capability:
        pid, birth, run_id = 9876, '123456789', 'a' * 32
        closed = False
        def check_live(self):
            pass
        def close(self):
            self.closed = True
    capability = Capability()
    monkeypatch.setattr(parent, 'inspect_live_child', lambda _run_id: capability)
    with pytest.raises(RuntimeError, match='registered child'):
        parent._bind_ready(_ready(), identity=object())
    assert capability.closed


def test_ready_must_match_the_supervised_device(monkeypatch):
    class Identity:
        gpu_uuid = 'GPU-00000000-0000-0000-0000-000000000002'
    class Capability:
        pid, birth, run_id = 1234, '123456789', 'a' * 32
        closed = False
        def check_live(self):
            pass
        def close(self):
            self.closed = True
    capability = Capability()
    monkeypatch.setattr(parent, 'DeviceIdentity', Identity)
    monkeypatch.setattr(parent, 'inspect_live_child', lambda _run_id: capability)
    with pytest.raises(RuntimeError, match='supervised device'):
        parent._bind_ready(_ready(), identity=Identity())
    assert capability.closed


def test_marker_must_match_the_exact_supervisor_run(monkeypatch, tmp_path):
    class Identity:
        def __init__(self):
            self.lock_path = tmp_path / 'device.lock'
            self.marker_path = tmp_path / 'device.unsafe.json'
            self.attestation_path = tmp_path / 'device.unsafe.json.attestation'
        def validate_paths(self):
            pass
    identity = Identity()
    monkeypatch.setattr(parent, 'DeviceIdentity', Identity)
    monkeypatch.setattr(parent, '_identity_spec', lambda _identity: {'test': True})
    marker = {'version': 1, 'run_id': 'a' * 32, 'device_lease_path': str(identity.lock_path),
              'device_identity': {'test': True}}
    identity.marker_path.write_text(json.dumps(marker), encoding='utf-8')
    parent._check_marker(identity, 'a' * 32)
    with pytest.raises(RuntimeError, match='marker'):
        parent._check_marker(identity, 'b' * 32)
    identity.attestation_path.write_text('{}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='attestation'):
        parent._check_marker(identity, 'a' * 32)


def test_marker_check_does_not_dispatch_to_a_shadowed_identity_method(tmp_path):
    identity = resolve_device_identity(tmp_path, _GPU, physical_uuids=(_GPU,), environ={})
    object.__setattr__(identity, 'validate_paths', lambda: None)
    object.__setattr__(identity, 'marker_path', tmp_path / 'alternate.unsafe.json')
    identity.marker_path.write_text(json.dumps(_marker('a' * 32, identity.lock_path,
                                                       _identity_spec(identity))), encoding='utf-8')
    with pytest.raises(DeviceIdentityError, match='coordination paths'):
        parent._check_marker(identity, 'a' * 32)


def test_barrier_publication_is_atomic_and_create_only(tmp_path):
    path = tmp_path / 'start-grant.json'
    payload = {'challenge': 'b' * 64, 'owner_pid': 1234, 'issued_ns': 200}
    parent._publish_barrier(path, payload)
    assert json.loads(path.read_bytes()) == payload
    with pytest.raises(FileExistsError):
        parent._publish_barrier(path, {'different': True})
    assert json.loads(path.read_bytes()) == payload


def test_interrupted_barrier_write_never_publishes_partial_grant(monkeypatch, tmp_path):
    path = tmp_path / 'start-grant.json'
    def fail(_fd):
        raise OSError('simulated flush failure')
    monkeypatch.setattr(parent.os, 'fsync', fail)
    with pytest.raises(OSError, match='simulated'):
        parent._publish_barrier(path, {'grant': True})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_short_barrier_write_cannot_publish_truncated_grant(monkeypatch, tmp_path):
    class ShortWriter:
        def __init__(self, fd):
            self.fd = fd
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            return False
        def write(self, raw):
            os.write(self.fd, raw[:1])
            return 1
        def flush(self):
            pass
    monkeypatch.setattr(parent.os, 'fdopen', lambda fd, *args, **kwargs: ShortWriter(fd))
    path = tmp_path / 'start-grant.json'
    with pytest.raises(OSError, match='short'):
        parent._publish_barrier(path, {'grant': True})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != 'nt', reason='Windows create-only rename contract')
def test_barrier_is_readable_at_the_instant_it_becomes_visible(monkeypatch, tmp_path):
    observed = []
    real_rename, real_link = os.rename, os.link
    def wrap(operation):
        def publish(src, dst):
            operation(src, dst)
            observed.append(json.loads(parent._read_plain_frame(dst)))
        return publish
    monkeypatch.setattr(parent.os, 'rename', wrap(real_rename))
    monkeypatch.setattr(parent.os, 'link', wrap(real_link))
    parent._publish_barrier(tmp_path / 'start.json', {'grant': True})
    assert observed == [{'grant': True}]


def test_controller_is_disabled_before_any_supervisor_or_query(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(parent, 'run_supervised', lambda *a, **k: calls.append('spawn'))
    with pytest.raises(RuntimeError, match='disabled'):
        parent.run_session(attempt_dir=tmp_path / 'attempt')
    assert calls == []
    assert not (tmp_path / 'attempt').exists()


def _snapshot():
    def point(start):
        return {'gpu_uuid': _GPU, 'query_started_ns': start,
                'query_ended_ns': start + 10, 'observed_ns': start + 10,
                'total_bytes': 1000 * 1024**2, 'used_bytes': 100 * 1024**2,
                'free_bytes': 900 * 1024**2,
                'utilization_percent': 0}
    return {'observations': [point(8_500_000_000), point(8_800_000_000),
                             point(9_000_000_000)],
            'process_queries': [{'phase': 'before', 'ok': True,
                                 'query_started_ns': 9_100_000_000,
                                 'query_ended_ns': 9_100_000_010, 'processes': []}],
            'errors': [], 'error_count': 0, 'complete': True, 'in_flight': False, 'sealed': False}


def test_start_grant_binds_all_initial_timestamps_and_expires_before_staleness():
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    grant = parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000,
        snapshot=_snapshot(), gpu_uuid=_GPU, expected_smi_total=1000 * 1024**2, issued_ns=9_200_000_000)
    assert grant['initial_oldest_query_started_ns'] == 8_500_000_000
    assert grant['initial_latest_query_ended_ns'] == 9_000_000_010
    assert grant['initial_process_started_ns'] == 9_100_000_000
    assert grant['initial_process_ended_ns'] == 9_100_000_010
    assert grant['issued_ns'] < grant['expires_ns'] <= 10_500_000_000
    assert grant['owner_pid'] == ready['owner_pid'] and grant['challenge'] == ready['challenge']
    assert grant['gpu_uuid'] == _GPU
    assert grant['cuda_total_bytes'] == _CUDA_TOTAL
    assert grant['activity_smi_total_bytes'] == _SMI_TOTAL


def test_start_grant_rejects_wrong_device_or_cross_provider_total():
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    options = dict(parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000,
                   snapshot=_snapshot(), gpu_uuid=_GPU, expected_smi_total=_SMI_TOTAL,
                   issued_ns=9_200_000_000)
    for changed in (dict(ready, gpu_uuid='GPU-00000000-0000-0000-0000-000000000002'),
                    dict(ready, cuda_total_bytes=_SMI_TOTAL - 2 * 1024**2 - 1),
                    dict(ready, cuda_total_bytes=True)):
        with pytest.raises(ValueError):
            parent._make_start_grant(changed, **options)


def test_start_grant_rejects_oldest_stale_point_or_inflight_producer():
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    snapshot = _snapshot()
    snapshot['observations'][0]['query_started_ns'] = 7_000_000_000
    snapshot['observations'][0]['query_ended_ns'] = 7_000_000_010
    snapshot['observations'][0]['observed_ns'] = 7_000_000_010
    with pytest.raises(ValueError):
        parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000,
            snapshot=snapshot, gpu_uuid=_GPU, expected_smi_total=1000 * 1024**2, issued_ns=9_200_000_000)
    snapshot = _snapshot()
    snapshot['in_flight'] = True
    with pytest.raises(ValueError):
        parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000,
            snapshot=snapshot, gpu_uuid=_GPU, expected_smi_total=1000 * 1024**2, issued_ns=9_200_000_000)


def test_start_grant_rejects_conflicting_earlier_initial_point():
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    snapshot = _snapshot()
    conflicting = dict(snapshot['observations'][0], query_started_ns=8_200_000_000,
                       query_ended_ns=8_200_000_010, observed_ns=8_200_000_010,
                       total_bytes=999 * 1024**2)
    snapshot['observations'].insert(0, conflicting)
    with pytest.raises(ValueError):
        parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000,
            snapshot=snapshot, gpu_uuid=_GPU, expected_smi_total=1000 * 1024**2,
            issued_ns=9_200_000_000)
def test_initial_screen_failure_never_publishes_a_grant(tmp_path):
    class Activity:
        def __init__(self):
            self.calls = 0
        def sample_gpu(self):
            self.calls += 1
            return self.calls != 2
        def sample_processes(self, phase):
            raise AssertionError('process query must not run after failed telemetry')
        def snapshot(self):
            return {'calls': self.calls}
    class Live:
        def check_live(self):
            pass
    saved = []
    with pytest.raises(RuntimeError, match='initial GPU telemetry'):
        parent._initial_screen(Activity(), ready=_ready(), live=Live(), marker_check=lambda: None,
            persist=lambda row: saved.append(row), grant_path=tmp_path / 'start.json',
            parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000, gpu_uuid=_GPU,
            expected_smi_total=1000 * 1024**2, clock=lambda: 9_200_000_000)
    assert saved == [{'calls': 1}, {'calls': 2}]
    assert not (tmp_path / 'start.json').exists()


def test_initial_process_query_rejects_owner_exit_and_retains_the_query(tmp_path):
    class Live:
        alive = True
        def check_live(self):
            if not self.alive:
                raise RuntimeError('owner exited')
    live = Live()
    class Activity:
        def sample_gpu(self):
            return True
        def sample_processes(self, phase):
            live.alive = False
            return True
        def snapshot(self):
            return {'query_seen': not live.alive}
    saved = []
    with pytest.raises(RuntimeError, match='owner exited'):
        parent._initial_screen(Activity(), ready=_ready(), live=live, marker_check=lambda: None,
            persist=lambda row: saved.append(row), grant_path=tmp_path / 'start.json',
            parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000, gpu_uuid=_GPU,
            expected_smi_total=1000 * 1024**2, clock=lambda: 9_200_000_000)
    assert saved[-1] == {'query_seen': True}
    assert not (tmp_path / 'start.json').exists()


def test_finished_barrier_binds_actual_preflight_and_rejects_late_or_wrong_owner():
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    grant = parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(), received_ns=8_100_000_000,
        snapshot=_snapshot(), gpu_uuid=_GPU, expected_smi_total=1000 * 1024**2, issued_ns=9_200_000_000)
    finish = {'schema': 'tap-repeated-workload-finished-v1',
              **{key: grant[key] for key in ('session_index', 'owner_pid', 'owner_birth',
                  'supervisor_run_id', 'challenge', 'gpu_uuid', 'cuda_total_bytes',
                  'activity_smi_total_bytes', 'clock', 'runtime')},
              'grant_seen_ns': 9_250_000_000, 'preflight_started_ns': 9_260_000_000,
              'workload_ended_ns': 9_800_000_000}
    assert parent._decode_finished(json.dumps(finish).encode(), grant=grant,
                                   received_ns=9_810_000_000) == finish
    for field, value in [('owner_pid', 9876), ('challenge', 'c' * 64),
                         ('gpu_uuid', 'GPU-00000000-0000-0000-0000-000000000002'),
                         ('cuda_total_bytes', _CUDA_TOTAL + 1), ('activity_smi_total_bytes', _SMI_TOTAL + 1),
                         ('preflight_started_ns', grant['expires_ns'] + 1),
                         ('workload_ended_ns', 9_900_000_000)]:
        invalid = dict(finish)
        invalid[field] = value
        with pytest.raises(ValueError):
            parent._decode_finished(json.dumps(invalid).encode(), grant=grant,
                                     received_ns=9_810_000_000)


def test_private_journal_retains_prior_records_and_latches_failed_flush(monkeypatch, tmp_path):
    path = tmp_path / 'activity.jsonl'
    journal = parent._PrivateJournal(path)
    journal.append({'kind': 'gpu', 'value': 1})
    def fail(_fd):
        raise OSError('simulated journal flush failure')
    monkeypatch.setattr(parent.os, 'fsync', fail)
    with pytest.raises(OSError, match='simulated'):
        journal.append({'kind': 'gpu', 'value': 2})
    assert journal.healthy is False
    with pytest.raises(RuntimeError):
        journal.append({'kind': 'gpu', 'value': 3})
    journal.close()
    assert json.loads(path.read_bytes().splitlines()[0]) == {'kind': 'gpu', 'value': 1}


def test_activity_recorder_persists_only_new_points_and_rejects_lost_history(tmp_path):
    snapshot = _snapshot()
    class Activity:
        def snapshot(self):
            return snapshot
    journal = parent._PrivateJournal(tmp_path / 'activity.jsonl')
    recorder = parent._ActivityRecorder(Activity(), journal)
    recorder.persist()
    snapshot['observations'].append(dict(snapshot['observations'][-1], observed_ns=9_200_000_000))
    recorder.persist()
    rows = [json.loads(row) for row in (tmp_path / 'activity.jsonl').read_bytes().splitlines()]
    assert len(rows[0]['observations']) == 3 and len(rows[1]['observations']) == 1
    assert rows[1]['process_queries'] == []
    snapshot['observations'].clear()
    with pytest.raises(RuntimeError, match='history'):
        recorder.persist()
    with pytest.raises(RuntimeError, match='unverified'):
        recorder.persist()
    journal.close()


def test_stalled_telemetry_producer_stays_rooted_until_it_really_joins():
    entered, release = threading.Event(), threading.Event()
    class Activity:
        def sample_gpu(self):
            entered.set()
            release.wait(5)
            return True
    class Live:
        def check_live(self):
            pass
    class Recorder:
        def persist(self):
            pass
    poller = parent._ActivityPoller(Activity(), Live(), lambda: None, Recorder(),
                                    lambda: None, interval=0.01)
    try:
        poller.start()
        assert entered.wait(2)
        assert poller.stop_and_join(timeout=0.01) is False
        assert poller in parent.pending_activity_pollers()
        release.set()
        assert poller.stop_and_join(timeout=2) is True
        assert poller not in parent.pending_activity_pollers()
    finally:
        release.set()
        poller.stop_and_join(timeout=2)


def test_telemetry_failure_aborts_the_owner_and_preserves_error():
    aborted = threading.Event()
    class Activity:
        def sample_gpu(self):
            return False
    class Live:
        def check_live(self):
            pass
    class Recorder:
        def persist(self):
            pass
    poller = parent._ActivityPoller(Activity(), Live(), lambda: None, Recorder(),
                                    aborted.set, interval=0.01)
    poller.start()
    assert aborted.wait(2)
    assert poller.stop_and_join(timeout=2) is True
    assert isinstance(poller.error, RuntimeError)
    assert poller.completed_cleanly is False


def test_telemetry_failure_still_attempts_abort_if_stop_notification_fails():
    aborted = threading.Event()
    class Activity:
        def sample_gpu(self):
            return False
    class Live:
        def check_live(self):
            pass
    class Recorder:
        def persist(self):
            pass
    class BrokenStop:
        def is_set(self):
            return False
        def set(self):
            raise OSError('stop notification failed')
    poller = parent._ActivityPoller(Activity(), Live(), lambda: None, Recorder(), aborted.set, interval=0.01)
    poller._stop = BrokenStop()
    try:
        poller.start()
        assert aborted.wait(2)
        assert poller.error is not None and not poller.completed_cleanly
    finally:
        poller._stop = threading.Event()
        poller.stop_and_join(timeout=2)


def test_after_process_query_cannot_exclude_a_dead_owner_pid(tmp_path):
    class Live:
        alive = True
        def check_live(self):
            if not self.alive:
                raise RuntimeError('owner exited during process query')
    live = Live()
    class Activity:
        def sample_processes(self, phase):
            assert phase == 'after'
            live.alive = False
            return True
    class Poller:
        completed_cleanly = True
        def stop_and_join(self, *, timeout):
            return True
    class Recorder:
        calls = 0
        def persist(self):
            self.calls += 1
    recorder = Recorder()
    finish = {'workload_ended_ns': 10_000_000_000}
    grant = {'clock': _clock(), 'runtime': _runtime(), 'gpu_uuid': _GPU,
             'activity_smi_total_bytes': _SMI_TOTAL}
    with pytest.raises(RuntimeError, match='owner exited'):
        parent._finish_screen(Activity(), Poller(), recorder=recorder, live=live,
            marker_check=lambda: None, finish=finish, grant=grant,
            finish_path=tmp_path / 'finish-grant.json', gpu_uuid=_GPU,
            expected_smi_total=1000 * 1024**2, exclusive=True,
            clock=lambda: 10_100_000_000, sleep=lambda _seconds: None)
    assert recorder.calls == 1
    assert not (tmp_path / 'finish-grant.json').exists()


def test_finish_screen_records_bracketed_process_and_delayed_idle_points(tmp_path):
    class Clock:
        now = 10_100_000_000
        def __call__(self):
            return self.now
        def sleep(self, seconds):
            self.now += max(1, round(seconds * 1_000_000_000))
    clock = Clock()
    snapshot = _snapshot()
    for timestamp in (9_500_000_000, 9_800_000_000, 9_990_000_000):
        snapshot['observations'].append(dict(snapshot['observations'][-1],
            query_started_ns=timestamp, query_ended_ns=timestamp + 10, observed_ns=timestamp + 10))
    class Activity:
        def sample_processes(self, phase):
            snapshot['process_queries'].append({'phase': phase, 'ok': True,
                'query_started_ns': clock.now, 'query_ended_ns': clock.now + 10, 'processes': []})
            clock.now += 100_000_000
            return True
        def sample_gpu(self):
            snapshot['observations'].append(dict(snapshot['observations'][-1],
                query_started_ns=clock.now, query_ended_ns=clock.now + 10, observed_ns=clock.now + 10))
            clock.now += 100_000_000
            return True
        def seal(self, *, producers_stopped):
            assert producers_stopped is True
            snapshot['sealed'] = True
            return snapshot
    class Poller:
        completed_cleanly = True
        def stop_and_join(self, *, timeout):
            return True
    class Live:
        def check_live(self):
            pass
    class Recorder:
        def persist(self):
            pass
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    grant = parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(),
        received_ns=8_100_000_000, snapshot=_snapshot(), gpu_uuid=_GPU,
        expected_smi_total=1000 * 1024**2, issued_ns=9_200_000_000)
    finish = {'preflight_started_ns': 9_260_000_000, 'workload_ended_ns': 10_000_000_000}
    screen, cleanup = parent._finish_screen(Activity(), Poller(), recorder=Recorder(), live=Live(),
        marker_check=lambda: None, finish=finish, grant=grant, finish_path=tmp_path / 'finish.json',
        gpu_uuid=_GPU, expected_smi_total=1000 * 1024**2, exclusive=True,
        clock=clock, sleep=clock.sleep)
    assert screen['availability'] == 'complete' and screen['status'] == 'no_contamination_observed'
    assert len(screen['after_idle_samples']) == 3
    assert snapshot['process_queries'][-1]['query_started_ns'] >= finish['workload_ended_ns']
    assert json.loads((tmp_path / 'finish.json').read_bytes()) == cleanup


def test_finish_screen_cannot_promote_a_failed_process_query(monkeypatch, tmp_path):
    class Clock:
        now = 10_100_000_000
        def __call__(self):
            return self.now
        def sleep(self, seconds):
            self.now += max(1, round(seconds * 1_000_000_000))
    clock = Clock()
    snapshot = _snapshot()
    class Activity:
        def sample_processes(self, phase):
            return False
        def sample_gpu(self):
            return True
        def seal(self, *, producers_stopped):
            snapshot['sealed'] = True
            return snapshot
    class Poller:
        completed_cleanly = True
        def stop_and_join(self, *, timeout):
            return True
    class Live:
        def check_live(self):
            pass
    class Recorder:
        def persist(self):
            pass
    seen = []
    def classify(**kwargs):
        seen.extend(kwargs['errors'])
        return {'availability': 'unverified' if kwargs['errors'] else 'complete'}
    monkeypatch.setattr(parent, 'classify_activity', classify)
    ready = _ready(); ready['ready_ns'] = 8_000_000_000
    grant = parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(),
        received_ns=8_100_000_000, snapshot=_snapshot(), gpu_uuid=_GPU,
        expected_smi_total=1000 * 1024**2, issued_ns=9_200_000_000)
    screen, cleanup = parent._finish_screen(Activity(), Poller(), recorder=Recorder(), live=Live(),
        marker_check=lambda: None, finish={'preflight_started_ns': 9_260_000_000,
        'workload_ended_ns': 10_000_000_000}, grant=grant, finish_path=tmp_path / 'finish.json',
        gpu_uuid=_GPU, expected_smi_total=1000 * 1024**2, exclusive=True,
        clock=clock, sleep=clock.sleep)
    assert seen and screen['availability'] == 'unverified'
    assert cleanup['screen_availability'] == 'unverified'


def test_finish_screen_rejects_a_different_device_or_smi_total(tmp_path):
    grant, finish = _finished_fixture()
    class Poller:
        def stop_and_join(self, *, timeout):
            raise AssertionError('must reject mismatched screen before querying')
    class Live:
        def check_live(self):
            pass
    for uuid, total in (('GPU-00000000-0000-0000-0000-000000000002', _SMI_TOTAL),
                        (_GPU, _SMI_TOTAL + 1024**2)):
        with pytest.raises(ValueError, match='bound device'):
            parent._finish_screen(object(), Poller(), recorder=object(), live=Live(),
                marker_check=lambda: None, finish=finish, grant=grant,
                finish_path=tmp_path / 'finish.json', gpu_uuid=uuid, expected_smi_total=total,
                exclusive=True, clock=lambda: 10_100_000_000, sleep=lambda _seconds: None)


def test_supervisor_uses_strict_exit_and_retains_running_capability(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def supervise(command, **kwargs):
        calls.append((command, kwargs))
        entered.set()
        release.wait(5)
        return {'clean': True}
    class Channel:
        aborted = False
        def abort(self):
            self.aborted = True
        def wait_closed(self, *, timeout):
            return True
    monkeypatch.setattr(parent, 'run_supervised', supervise)
    channel = Channel()
    owner = parent._Supervision(['fake-owner'], channel=channel, identity=object(),
        device_lease_path=tmp_path / 'device.lock', marker_path=tmp_path / 'device.unsafe.json',
        cwd=tmp_path, env={}, timeout=5, process_job=object())
    try:
        owner.start()
        assert entered.wait(2)
        assert owner.release_if_quiescent(timeout=0.01) is False
        assert owner in parent.pending_parent_supervisions()
        assert calls[0][1]['require_zero_exit'] is True
        assert calls[0][1]['inherit_environment'] is False
        release.set()
        assert owner.wait(timeout=2) == {'clean': True}
        assert owner.release_if_quiescent(timeout=2) is True
        assert owner not in parent.pending_parent_supervisions()
    finally:
        release.set()
        owner.release_if_quiescent(timeout=2)


def test_interrupted_supervisor_thread_start_revokes_spawn(monkeypatch, tmp_path):
    calls = []
    class Channel:
        def abort(self):
            calls.append('abort')
        def wait_closed(self, *, timeout):
            return True
    monkeypatch.setattr(parent, 'run_supervised', lambda *args, **kwargs: calls.append('spawn'))
    original = threading.Thread.start
    def start_then_interrupt(thread):
        original(thread)
        raise KeyboardInterrupt()
    monkeypatch.setattr(threading.Thread, 'start', start_then_interrupt)
    owner = parent._Supervision(['fake-owner'], channel=Channel(), identity=object(),
        device_lease_path=tmp_path / 'device.lock', marker_path=tmp_path / 'device.unsafe.json',
        cwd=tmp_path, env={}, timeout=5, process_job=object())
    with pytest.raises(KeyboardInterrupt):
        owner.start()
    assert owner.release_if_quiescent(timeout=2) is True
    assert calls == ['abort']


def test_terminal_supervision_requires_the_same_reaped_owner_and_zero_exit():
    ready = _ready()
    class Live:
        run_id, pid, birth = ready['supervisor_run_id'], ready['owner_pid'], ready['owner_birth']
    good = SupervisionResult(ready['supervisor_run_id'], ready['owner_pid'], 0, True, False,
                             'clean device teardown')
    parent._check_supervision_result(good, ready=ready, live=Live())
    for bad in (SupervisionResult('c' * 32, ready['owner_pid'], 0, True, False, 'clean device teardown'),
                SupervisionResult(ready['supervisor_run_id'], 9999, 0, True, False, 'clean device teardown'),
                SupervisionResult(ready['supervisor_run_id'], ready['owner_pid'], 1, True, False, 'clean device teardown'),
                SupervisionResult(ready['supervisor_run_id'], ready['owner_pid'], 0, False, False, 'missing'),
                SupervisionResult(ready['supervisor_run_id'], ready['owner_pid'], 0, True, True, 'timeout')):
        with pytest.raises(RuntimeError):
            parent._check_supervision_result(bad, ready=ready, live=Live())
    mismatch = Live()
    mismatch.birth = '999'
    with pytest.raises(RuntimeError):
        parent._check_supervision_result(good, ready=ready, live=mismatch)


def _finished_fixture():
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    grant = parent._make_start_grant(ready, parent_clock=_clock(), parent_runtime=_runtime(),
        received_ns=8_100_000_000, snapshot=_snapshot(), gpu_uuid=_GPU,
        expected_smi_total=_SMI_TOTAL, issued_ns=9_200_000_000)
    finish = {'schema': 'tap-repeated-workload-finished-v1',
        **{key: grant[key] for key in ('session_index', 'owner_pid', 'owner_birth',
            'supervisor_run_id', 'challenge', 'gpu_uuid', 'cuda_total_bytes',
            'activity_smi_total_bytes', 'clock', 'runtime')},
        'grant_seen_ns': 9_250_000_000, 'preflight_started_ns': 9_260_000_000,
        'workload_ended_ns': 10_000_000_000}
    return grant, finish


def test_finished_file_is_read_only_while_the_registered_owner_is_alive(tmp_path):
    grant, finish = _finished_fixture()
    path = tmp_path / 'finished.json'
    path.write_text(json.dumps(finish), encoding='utf-8')
    checks = []
    class Live:
        def check_live(self):
            checks.append('live')
    result = parent._await_finished(path, grant=grant, live=Live(), marker_check=lambda: checks.append('marker'),
        supervisor_done=lambda: False, clock=lambda: 10_100_000_000, sleep=lambda _seconds: None,
        timeout=1)
    assert result == finish
    assert checks == ['live', 'marker', 'live', 'marker']


def test_finished_file_rejects_owner_death_during_read(monkeypatch, tmp_path):
    grant, finish = _finished_fixture()
    path = tmp_path / 'finished.json'
    path.write_text(json.dumps(finish), encoding='utf-8')
    class Live:
        checks = 0
        def check_live(self):
            self.checks += 1
            if self.checks == 2:
                raise RuntimeError('owner died during read')
    with pytest.raises(RuntimeError, match='owner died'):
        parent._await_finished(path, grant=grant, live=Live(), marker_check=lambda: None,
            supervisor_done=lambda: False, clock=lambda: 10_100_000_000,
            sleep=lambda _seconds: None, timeout=1)


def test_missing_finished_file_has_bounded_wait_and_detects_supervisor_exit(tmp_path):
    grant, _finish = _finished_fixture()
    class Clock:
        now = 10_100_000_000
        def __call__(self):
            return self.now
        def sleep(self, seconds):
            self.now += round(seconds * 1_000_000_000)
    class Live:
        def check_live(self):
            pass
    clock = Clock()
    with pytest.raises(TimeoutError):
        parent._await_finished(tmp_path / 'missing', grant=grant, live=Live(), marker_check=lambda: None,
            supervisor_done=lambda: False, clock=clock, sleep=clock.sleep, timeout=0.1)
    with pytest.raises(RuntimeError, match='supervisor exited'):
        parent._await_finished(tmp_path / 'missing', grant=grant, live=Live(), marker_check=lambda: None,
            supervisor_done=lambda: True, clock=clock, sleep=clock.sleep, timeout=0.1)


def test_closed_journal_anchors_and_bytes_are_bound_before_cleanup(tmp_path):
    grant, finish = _finished_fixture()
    rows = [{'stage': 'preflight', 'observation_id': 'preflight', 'started_ns': finish['preflight_started_ns'],
             'gate_lower_ns': 9_255_000_000, 'gate_upper_ns': 9_265_000_000,
             'memory_snapshot_started_ns': 9_251_000_000, 'memory_snapshot_ended_ns': 9_254_000_000},
            {'stage': 'postcheck', 'observation_id': 'preflight', 'status': 'passed'}]
    rows += [{'stage': 'other'} for _ in range(82)]
    rows += [{'stage': 'measured', 'observation_id': 'measured:11:2', 'ended_ns': 9_999_000_000},
             {'stage': 'postcheck', 'observation_id': 'measured:11:2', 'status': 'passed',
              'device_work_complete': True, 'completed_ns': finish['workload_ended_ns']}]
    raw = b''.join((json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                   for row in rows)
    path = tmp_path / 'journal.jsonl'
    path.write_bytes(raw)
    read_rows, digest = parent._read_bound_journal(path, finish=finish, grant=grant)
    assert read_rows == rows and digest == hashlib.sha256(raw).hexdigest()
    assert parent._recheck_bound_journal(path, finish=finish, grant=grant, expected_sha256=digest) == rows
    path.write_bytes(raw.replace(b'{"stage":"other"}', b'{"stage":"other2"}', 1))
    with pytest.raises(ValueError, match='changed'):
        parent._recheck_bound_journal(path, finish=finish, grant=grant, expected_sha256=digest)
    path.write_bytes(raw)
    bad_finish = dict(finish, workload_ended_ns=finish['workload_ended_ns'] + 1)
    with pytest.raises(ValueError, match='anchor'):
        parent._read_bound_journal(path, finish=bad_finish, grant=grant)
    rows[0]['gate_upper_ns'] = grant['expires_ns'] + 1
    path.write_bytes(b''.join((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode()
                              for row in rows))
    with pytest.raises(ValueError, match='preflight'):
        parent._read_bound_journal(path, finish=finish, grant=grant)
    rows[0]['gate_upper_ns'] = 9_265_000_000
    for key, value in (('memory_snapshot_started_ns', finish['grant_seen_ns'] - 1),
                       ('memory_snapshot_started_ns', True),
                       ('memory_snapshot_ended_ns', rows[0]['gate_lower_ns'] + 1),
                       ('memory_snapshot_ended_ns', None)):
        original = rows[0][key]
        rows[0][key] = value
        path.write_bytes(b''.join((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode()
                                  for row in rows))
        with pytest.raises(ValueError, match='memory snapshot'):
            parent._read_bound_journal(path, finish=finish, grant=grant)
        rows[0][key] = original
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError):
        parent._read_bound_journal(path, finish=finish, grant=grant)


def _driver_fixture(monkeypatch, tmp_path, query):
    monkeypatch.setattr(parent, '_PARENT_LAUNCH_ENABLED', True)
    ready = _ready()
    ready['ready_ns'] = 8_000_000_000
    class Channel:
        def receive(self, *, timeout):
            return json.dumps(ready).encode()
    class Identity:
        gpu_uuid = _GPU
    class Supervision:
        def __init__(self):
            self._channel, self._identity, self._process_job = channel, identity, object()
            self.aborted = False
            self.releasable = True
        def start(self):
            pass
        def abort(self):
            self.aborted = True
        def release_if_quiescent(self, *, timeout):
            return self.releasable
        @property
        def done(self):
            return False
    class Live:
        closed = False
        def check_live(self):
            pass
        def close(self):
            self.closed = True
    channel, identity, live = Channel(), Identity(), Live()
    supervision = Supervision()
    monkeypatch.setattr(parent, '_bind_ready', lambda _ready, *, identity: live)
    monkeypatch.setattr(parent, '_check_marker', lambda _identity, _run_id: None)
    journal = parent._PrivateJournal(tmp_path / 'activity.jsonl')
    activity = ParentActivity(gpu_uuid=_GPU, query=query, clock=lambda: 8_200_000_000)
    kwargs = dict(session_index=2, challenge='b' * 64, identity=identity, channel=channel,
        supervision=supervision, activity=activity, journal=journal, attempt_dir=tmp_path,
        expected_smi_total=_SMI_TOTAL, schedule={}, parent_clock=_clock(), parent_runtime=_runtime(),
        exclusive=True, clock=lambda: 8_100_000_000, sleep=lambda _seconds: None,
        startup_timeout=1, workload_timeout=1, reap_timeout=1)
    return supervision, live, journal, kwargs


def test_driver_aborts_and_saves_a_real_interrupted_query(monkeypatch, tmp_path):
    def query(_kind):
        raise KeyboardInterrupt()
    supervision, live, journal, kwargs = _driver_fixture(monkeypatch, tmp_path, query)
    kwargs['startup_timeout'] = 180
    with pytest.raises(KeyboardInterrupt):
        parent._drive_session(**kwargs)
    records = [json.loads(line) for line in (tmp_path / 'activity.jsonl').read_bytes().splitlines()]
    assert supervision.aborted and live.closed
    assert records[-1]['error_count'] == 1 and records[-1]['complete'] is False
    assert records[-1]['errors'] == [{'code': 'query_or_parse_failed'}]
    assert journal._fd is None and not parent.pending_parent_attempts()


def test_driver_keeps_custody_when_grant_might_already_be_visible(monkeypatch, tmp_path):
    supervision, live, journal, kwargs = _driver_fixture(monkeypatch, tmp_path, lambda _kind: b'')
    supervision.releasable = False
    def interrupted_grant(_activity, **options):
        options['grant_path'].write_bytes(b'{"possibly_consumed":true}\n')
        raise KeyboardInterrupt()
    monkeypatch.setattr(parent, '_initial_screen', interrupted_grant)
    with pytest.raises(KeyboardInterrupt):
        parent._drive_session(**kwargs)
    assert supervision.aborted and (tmp_path / 'start-grant.json').exists()
    assert not live.closed and journal._fd is not None
    pending = parent.pending_parent_attempts()
    assert len(pending) == 1
    supervision.releasable = True
    assert pending[0].retire_if_quiescent(timeout=1) is True
    assert live.closed and journal._fd is None and not parent.pending_parent_attempts()


def test_driver_keeps_unjoined_telemetry_and_owner_custody(monkeypatch, tmp_path):
    supervision, live, journal, kwargs = _driver_fixture(monkeypatch, tmp_path, lambda _kind: b'')
    grant, _finish = _finished_fixture()
    class BlockedPoller:
        releasable = False
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop_and_join(self, *, timeout):
            return self.releasable
    monkeypatch.setattr(parent, '_initial_screen', lambda *args, **kwargs: grant)
    monkeypatch.setattr(parent, '_ActivityPoller', BlockedPoller)
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(parent, '_await_finished', interrupt)
    with pytest.raises(KeyboardInterrupt):
        parent._drive_session(**kwargs)
    pending = parent.pending_parent_attempts()
    assert supervision.aborted and len(pending) == 1
    assert not live.closed and journal._fd is not None
    assert (tmp_path / 'activity.jsonl').read_bytes() == b''
    pending[0].poller.releasable = True
    assert pending[0].retire_if_quiescent(timeout=1) is True
    assert live.closed and journal._fd is None and not parent.pending_parent_attempts()
