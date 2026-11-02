import json
import os
import sys
import time

import pytest

from looking_glass import tap_repeated_child_barrier as child
from looking_glass import tap_repeated_parent as parent
from looking_glass.supervisor import _supervisor_birth


_GPU = 'GPU-00000000-0000-0000-0000-000000000001'
_TOTAL = 8 * 1024**3


def _ready():
    info = time.get_clock_info('perf_counter')
    return {'schema': 'tap-repeated-owner-ready-v1', 'session_index': 2, 'owner_pid': os.getpid(),
            'owner_birth': _supervisor_birth(), 'supervisor_run_id': 'a' * 32, 'challenge': 'b' * 64,
            'gpu_uuid': _GPU, 'cuda_total_bytes': _TOTAL, 'ready_ns': 8_000_000_000,
            'clock': {'name': 'perf_counter_ns', 'monotonic': info.monotonic, 'adjustable': info.adjustable,
                      'implementation': info.implementation, 'resolution_seconds': info.resolution},
            'runtime': {'platform': sys.platform, 'implementation': sys.implementation.name,
                        'python_version': '.'.join(map(str, sys.version_info[:3]))}}


def _grant(ready):
    def point(start):
        return {'gpu_uuid': _GPU, 'query_started_ns': start, 'query_ended_ns': start + 10,
                'observed_ns': start + 10, 'total_bytes': _TOTAL, 'used_bytes': 1024**3,
                'free_bytes': 7 * 1024**3, 'utilization_percent': 0}
    snapshot = {'observations': [point(8_500_000_000), point(8_800_000_000), point(9_000_000_000)],
                'process_queries': [{'phase': 'before', 'ok': True, 'query_started_ns': 9_100_000_000,
                                     'query_ended_ns': 9_100_000_010, 'processes': []}],
                'errors': [], 'error_count': 0, 'complete': True, 'in_flight': False}
    return parent._make_start_grant(ready, parent_clock=ready['clock'], parent_runtime=ready['runtime'],
        received_ns=8_100_000_000, snapshot=snapshot, gpu_uuid=_GPU,
        expected_smi_total=_TOTAL, issued_ns=9_200_000_000)


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='requires the pinned Windows QPC runtime')
def test_child_gate_binds_actual_preflight_and_publishes_finished_once(tmp_path):
    ready = _ready()
    grant = _grant(ready)
    barriers = child.ChildBarriers(ready)
    barriers.accept_start(json.dumps(grant).encode(), observed_ns=9_250_000_000)
    barriers.authorize_preflight(9_255_000_000, 9_265_000_000)
    with pytest.raises(RuntimeError):
        barriers.authorize_preflight(9_255_000_000, 9_265_000_000)
    path = tmp_path / 'finished.json'
    barriers.publish_finished(path, preflight_started_ns=9_260_000_000, workload_ended_ns=10_000_000_000)
    finish = parent._decode_finished(path.read_bytes(), grant=grant, received_ns=10_100_000_000)
    assert finish['preflight_started_ns'] == 9_260_000_000
    with pytest.raises(RuntimeError):
        barriers.publish_finished(path, preflight_started_ns=9_260_000_000, workload_ended_ns=10_000_000_000)
    cleanup = {'schema': 'tap-repeated-cleanup-grant-v1',
               **{key: grant[key] for key in ('session_index', 'owner_pid', 'owner_birth',
                   'supervisor_run_id', 'challenge', 'gpu_uuid', 'cuda_total_bytes',
                   'activity_smi_total_bytes', 'clock', 'runtime')},
               'workload_ended_ns': 10_000_000_000, 'issued_ns': 10_100_000_000,
               'expires_ns': 15_100_000_000, 'screen_availability': 'unverified'}
    barriers.accept_cleanup(json.dumps(cleanup).encode(), observed_ns=10_200_000_000)
    assert barriers.cleanup_authorized is True
    with pytest.raises(RuntimeError):
        barriers.accept_cleanup(json.dumps(cleanup).encode(), observed_ns=10_200_000_000)


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='requires the pinned Windows QPC runtime')
def test_child_gate_rejects_mismatched_identity_stale_interval_and_cleanup(tmp_path):
    ready = _ready()
    grant = _grant(ready)
    bad = dict(grant, activity_smi_total_bytes=_TOTAL + 3 * 1024**2)
    with pytest.raises(ValueError):
        child.ChildBarriers(ready).accept_start(json.dumps(bad).encode(), observed_ns=9_250_000_000)
    for lower, upper in ((9_255_000_000, 9_505_000_001),
                         (9_255_000_000, grant['expires_ns'] + 1),
                         (9_100_000_000, 9_100_000_001)):
        barriers = child.ChildBarriers(ready)
        barriers.accept_start(json.dumps(grant).encode(), observed_ns=9_250_000_000)
        with pytest.raises(ValueError):
            barriers.authorize_preflight(lower, upper)
        with pytest.raises(RuntimeError):
            barriers.authorize_preflight(9_255_000_000, 9_265_000_000)
    barriers = child.ChildBarriers(ready)
    barriers.accept_start(json.dumps(grant).encode(), observed_ns=9_250_000_000)
    barriers.authorize_preflight(9_255_000_000, 9_265_000_000)
    with pytest.raises(ValueError):
        barriers.publish_finished(tmp_path / 'bad.json', preflight_started_ns=9_270_000_000,
                                  workload_ended_ns=10_000_000_000)
    assert not (tmp_path / 'bad.json').exists()


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='requires the pinned Windows QPC runtime')
def test_child_rejects_a_stale_or_wrong_owner_cleanup_grant(tmp_path):
    ready, grant = _ready(), _grant(_ready())
    for field, value in (('owner_pid', os.getpid() + 1), ('expires_ns', 10_150_000_000)):
        barriers = child.ChildBarriers(ready)
        barriers.accept_start(json.dumps(grant).encode(), observed_ns=9_250_000_000)
        barriers.authorize_preflight(9_255_000_000, 9_265_000_000)
        barriers.publish_finished(tmp_path / (field + '.json'),
            preflight_started_ns=9_260_000_000, workload_ended_ns=10_000_000_000)
        cleanup = {'schema': 'tap-repeated-cleanup-grant-v1',
                   **{key: grant[key] for key in ('session_index', 'owner_pid', 'owner_birth',
                       'supervisor_run_id', 'challenge', 'gpu_uuid', 'cuda_total_bytes',
                       'activity_smi_total_bytes', 'clock', 'runtime')},
                   'workload_ended_ns': 10_000_000_000, 'issued_ns': 10_100_000_000,
                   'expires_ns': 15_100_000_000, 'screen_availability': 'complete'}
        cleanup[field] = value
        with pytest.raises(ValueError):
            barriers.accept_cleanup(json.dumps(cleanup).encode(), observed_ns=10_200_000_000)
        assert barriers.cleanup_authorized is False


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='requires the pinned Windows QPC runtime')
def test_interrupted_visible_finished_publication_cannot_be_reused(monkeypatch, tmp_path):
    ready, grant = _ready(), _grant(_ready())
    barriers = child.ChildBarriers(ready)
    barriers.accept_start(json.dumps(grant).encode(), observed_ns=9_250_000_000)
    barriers.authorize_preflight(9_255_000_000, 9_265_000_000)
    real_publish = child._publish_barrier
    def publish_then_interrupt(path, payload):
        real_publish(path, payload)
        raise KeyboardInterrupt()
    monkeypatch.setattr(child, '_publish_barrier', publish_then_interrupt)
    path = tmp_path / 'finished.json'
    with pytest.raises(KeyboardInterrupt):
        barriers.publish_finished(path, preflight_started_ns=9_260_000_000, workload_ended_ns=10_000_000_000)
    assert path.exists()
    with pytest.raises(RuntimeError):
        barriers.publish_finished(path, preflight_started_ns=9_260_000_000, workload_ended_ns=10_000_000_000)


def test_wait_frame_retries_only_transient_sharing_failures(monkeypatch):
    class Clock:
        now = 1_000_000_000
        def __call__(self):
            return self.now
        def sleep(self, seconds):
            self.now += round(seconds * 1_000_000_000)
    clock = Clock()
    sharing = PermissionError('sharing violation')
    sharing.winerror = 32
    attempts = []
    def read(_path):
        attempts.append(1)
        if len(attempts) == 1:
            raise sharing
        return b'{}'
    monkeypatch.setattr(child, '_read_plain_frame', read)
    raw, observed = child.wait_frame('unused', timeout=1, clock=clock, sleep=clock.sleep)
    assert raw == b'{}' and observed > 1_000_000_000 and len(attempts) == 2
    denied = PermissionError('denied')
    denied.winerror = 5
    monkeypatch.setattr(child, '_read_plain_frame', lambda _path: (_ for _ in ()).throw(denied))
    with pytest.raises(PermissionError):
        child.wait_frame('unused', timeout=1, clock=clock, sleep=clock.sleep)
