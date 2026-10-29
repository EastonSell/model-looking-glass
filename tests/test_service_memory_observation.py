"""CPU-only contracts for a bounded serving-memory observation."""

import importlib.util

import pytest

from looking_glass.memory_budget import CudaMemorySnapshot


class FakeCuda:
    def __init__(self):
        self.calls = []
        self.free = 800
        self.total = 1000
        self.allocated = 100
        self.reserved = 150
        self.peak_allocated = 260
        self.peak_reserved = 300

    def synchronize(self, device):
        assert device == 0
        self.calls.append('synchronize')

    def reset_peak_memory_stats(self, device):
        assert device == 0
        self.calls.append('reset')

    def mem_get_info(self, device):
        assert device == 0
        self.calls.append('free')
        return self.free, self.total

    def memory_allocated(self, device):
        assert device == 0
        self.calls.append('allocated')
        return self.allocated

    def memory_reserved(self, device):
        assert device == 0
        self.calls.append('reserved')
        return self.reserved

    def max_memory_allocated(self, device):
        assert device == 0
        self.calls.append('peak_allocated')
        return self.peak_allocated

    def max_memory_reserved(self, device):
        assert device == 0
        self.calls.append('peak_reserved')
        return self.peak_reserved


def make_observation(cuda, *, limit=4, fail=None):
    assert importlib.util.find_spec('looking_glass.service_memory_observation') is not None
    from looking_glass.service_memory_observation import ServingMemoryObservation

    marks = []
    ticks = iter(range(100, 200))

    def guard():
        if fail is not None and fail():
            raise RuntimeError('lease lost')
        marks.append('guard')

    observer = ServingMemoryObservation(cuda=cuda, guard=guard, mark_unsafe=lambda: marks.append('unsafe'),
                                        expected_total_bytes=1000, clock=lambda: next(ticks), max_samples=limit)
    return observer, marks


def test_serving_window_records_guarded_points_and_process_peaks():
    cuda = FakeCuda()
    observer, marks = make_observation(cuda)
    observer.start()
    observer.record(CudaMemorySnapshot(700, 1000, 180, 220))
    cuda.free, cuda.allocated, cuda.reserved = 720, 160, 220
    result = observer.finish()

    assert cuda.calls == ['synchronize', 'reset', 'free', 'allocated', 'reserved',
                          'synchronize', 'free', 'allocated', 'reserved',
                          'peak_allocated', 'peak_reserved']
    assert len(marks) == 2 * len(cuda.calls)
    assert result['baseline']['device_free_bytes'] == 800
    assert result['samples'][0]['device_free_bytes'] == 700
    assert result['endpoint']['device_free_bytes'] == 720
    assert result['process_peak_allocated_bytes'] == 260
    assert result['process_peak_reserved_bytes'] == 300
    assert result['baseline']['time_ns'] < result['samples'][0]['time_ns'] < result['endpoint']['time_ns']
    assert result['dropped_samples'] == 0


def test_sample_capacity_is_bounded_and_reports_drops():
    observer, _ = make_observation(FakeCuda(), limit=2)
    observer.start()
    for free in (750, 740, 730):
        observer.record(CudaMemorySnapshot(free, 1000, 150, 200))
    result = observer.finish()
    assert [s['device_free_bytes'] for s in result['samples']] == [750, 740]
    assert result['dropped_samples'] == 1


def test_lost_lease_makes_the_observation_failed_and_cannot_finish():
    cuda = FakeCuda()
    failed = [False]
    observer, marks = make_observation(cuda, fail=lambda: failed[0])
    observer.start()
    failed[0] = True
    with pytest.raises(RuntimeError, match='lease lost'):
        observer.finish()
    assert marks[-1] == 'unsafe'
    assert cuda.calls[-1] == 'reserved'  # no CUDA operation after failed guard
    with pytest.raises(RuntimeError):
        observer.finish()


def test_changed_device_total_fails_and_marks_owner_unsafe():
    observer, marks = make_observation(FakeCuda())
    observer.start()
    with pytest.raises(ValueError):
        observer.record(CudaMemorySnapshot(700, 1200, 180, 220))
    assert marks[-1] == 'unsafe'
    with pytest.raises(RuntimeError):
        observer.finish()


def test_invalid_device_free_stops_before_allocator_queries():
    cuda = FakeCuda()
    cuda.free = 1001
    observer, marks = make_observation(cuda)
    with pytest.raises(ValueError):
        observer.start()
    assert cuda.calls == ['synchronize', 'reset', 'free']
    assert marks[-1] == 'unsafe'


def test_baseline_total_must_match_profile_before_allocator_queries():
    cuda = FakeCuda()
    cuda.total = 1200
    observer, marks = make_observation(cuda)
    with pytest.raises(ValueError):
        observer.start()
    assert cuda.calls == ['synchronize', 'reset', 'free']
    assert marks[-1] == 'unsafe'


def test_invalid_allocated_stops_before_reserved_query():
    cuda = FakeCuda()
    cuda.allocated = -1
    observer, marks = make_observation(cuda)
    with pytest.raises(ValueError):
        observer.start()
    assert cuda.calls == ['synchronize', 'reset', 'free', 'allocated']
    assert marks[-1] == 'unsafe'


def test_changed_endpoint_total_stops_before_allocator_queries():
    cuda = FakeCuda()
    observer, marks = make_observation(cuda)
    observer.start()
    cuda.total = 1200
    with pytest.raises(ValueError):
        observer.finish()
    assert cuda.calls[-2:] == ['synchronize', 'free']
    assert marks[-1] == 'unsafe'


def test_invalid_peak_allocated_stops_before_reserved_peak_query():
    cuda = FakeCuda()
    observer, marks = make_observation(cuda)
    observer.start()
    cuda.peak_allocated = 1
    with pytest.raises(ValueError):
        observer.finish()
    assert cuda.calls[-1] == 'peak_allocated'
    assert marks[-1] == 'unsafe'


def test_concrete_backend_saves_source_bound_observation_in_private_file(tmp_path, monkeypatch):
    import json
    import torch
    from looking_glass.gpu_service import _TorchServiceBackend

    cuda = FakeCuda()
    for name in ('synchronize', 'reset_peak_memory_stats', 'mem_get_info',
                 'memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, getattr(cuda, name))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    class Session:
        def check(self, lease):
            pass
    binding = {'source_base_sha256': 'a' * 64, 'asset_manifest_sha256': 'b' * 64,
               'profile_ids': ['1' * 64, '2' * 64, '3' * 64], 'owner_pid': 123}
    path = tmp_path / 'serving-memory.json'
    backend = _TorchServiceBackend(session=Session(), device_lease=object(), cache_dir=tmp_path,
        team_ids=('alpha',), live_key_builder=lambda _: (), verify_loaded_model=lambda _: True,
        verify_device_model=lambda *_: True, serving_memory_path=path, serving_memory_binding=binding)
    from types import SimpleNamespace
    backend._resolved_keys = (SimpleNamespace(device_total_bytes=1000),)
    backend._start_serving_memory()
    cuda.free, cuda.allocated, cuda.reserved = 700, 180, 220
    backend.telemetry()
    backend._finish_serving_memory()
    record = json.loads(path.read_text())
    assert record['binding'] == binding
    assert record['baseline']['device_free_bytes'] == 800
    assert record['samples'][0]['device_free_bytes'] == 700
    assert record['process_peak_reserved_bytes'] == 300


@pytest.mark.parametrize('free,total,allocated,expected_calls', [
    (1001, 1000, 100, ['free']),
    (800, 1200, 100, ['free']),
    (800, 1000, -1, ['free', 'allocated']),
])
def test_backend_admission_checks_invalid_telemetry_before_later_cuda_calls(
        tmp_path, monkeypatch, free, total, allocated, expected_calls):
    import torch
    from types import SimpleNamespace
    from looking_glass import gpu_service
    from looking_glass.gpu_service import _TorchServiceBackend
    from looking_glass.memory_budget import TelemetryError

    cuda = FakeCuda()
    cuda.free, cuda.total, cuda.allocated = free, total, allocated
    for name in ('mem_get_info', 'memory_allocated', 'memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, getattr(cuda, name))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    marks = []
    monkeypatch.setattr(gpu_service, '_mark_unsafe', lambda: marks.append('unsafe'))
    class Session:
        def check(self, lease):
            pass
    backend = _TorchServiceBackend(session=Session(), device_lease=object(), cache_dir=tmp_path,
        team_ids=('alpha',), live_key_builder=lambda _: (), verify_loaded_model=lambda _: True,
        verify_device_model=lambda *_: True)
    backend._resolved_keys = (SimpleNamespace(device_total_bytes=1000),)
    with pytest.raises(TelemetryError):
        backend.telemetry()
    assert cuda.calls == expected_calls
    assert marks == ['unsafe']
