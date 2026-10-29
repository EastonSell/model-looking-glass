import subprocess
import inspect
import sys
import threading

import pytest

from looking_glass import tap_repeated_activity as activity
from looking_glass import tap_repeated_protocol as protocol


UUID = 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
OTHER = 'GPU-ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee'
GPU = f'{UUID}, 8192, 2048, 6144, 1\n'.encode()
PROCESSES = f'{UUID}, 101, "a process, with comma"\n{OTHER}, 103, other\n'.encode()


def test_parse_actual_point_shapes_and_private_processes():
    gpu = activity.parse_gpu(GPU, gpu_uuid=UUID, started_ns=1, ended_ns=3)
    assert gpu == {'gpu_uuid': UUID, 'query_started_ns': 1, 'query_ended_ns': 3, 'observed_ns': 3,
                   'total_bytes': 8 * protocol.GIB, 'used_bytes': 2 * protocol.GIB,
                   'free_bytes': 6 * protocol.GIB, 'utilization_percent': 1}
    processes = activity.parse_processes(PROCESSES, gpu_uuid=UUID, phase='before', started_ns=4, ended_ns=5)
    assert processes['processes'] == [{'gpu_uuid': UUID, 'pid': 101, 'name': 'a process, with comma'}]
    assert activity.parse_processes(b'', gpu_uuid=UUID, phase='after', started_ns=6, ended_ns=7)['processes'] == []


@pytest.mark.parametrize('raw', [pytest.param(GPU + GPU, id='duplicate'), pytest.param(b'', id='missing'),
    pytest.param(b'not,a,gpu,row\n', id='malformed'),
    pytest.param(f'{UUID}, 8192, 8192, 8192, 1\n'.encode(), id='impossible-memory'),
    pytest.param(b'x' * (64 * 1024 + 1), id='oversized')])
def test_invalid_gpu_queries_are_not_promoted(raw):
    with pytest.raises(ValueError):
        activity.parse_gpu(raw, gpu_uuid=UUID, started_ns=1, ended_ns=2)


@pytest.mark.parametrize('raw', [b'[Not Supported]\n', f'{UUID}, -1, name\n'.encode(),
                                  f'{UUID}, 9, \n'.encode(), b'\xff'])
def test_invalid_process_queries_are_not_promoted(raw):
    with pytest.raises((ValueError, UnicodeDecodeError)):
        activity.parse_processes(raw, gpu_uuid=UUID, phase='after', started_ns=1, ended_ns=2)


def test_smi_uses_fixed_shell_free_commands_and_bounded_retained_output(tmp_path, monkeypatch):
    executable = tmp_path / 'nvidia-smi.exe'
    executable.write_bytes(b'cpu fake')
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=GPU)

    monkeypatch.setattr(subprocess, 'run', run)
    smi = activity._NvidiaSmi(executable)
    assert smi('gpu') == GPU
    assert calls[0][0][1:] == ['--query-gpu=' + activity._GPU_FIELDS, '--format=csv,noheader,nounits']
    assert calls[0][1]['stdin'] == subprocess.DEVNULL and calls[0][1]['timeout'] == 1.5
    assert 'shell' not in calls[0][1]
    with pytest.raises(ValueError):
        activity._NvidiaSmi('relative-smi')


def test_parent_inventory_captures_source_times_and_rejects_late_or_overflow():
    values = iter(range(1, 50))
    collector = activity.ParentActivity(gpu_uuid=UUID,
        query=lambda kind: GPU if kind == 'gpu' else PROCESSES, clock=lambda: next(values), max_samples=3)
    assert collector.sample_processes('before')
    assert all(collector.sample_gpu() for _ in range(3))
    assert not collector.sample_gpu()
    assert collector.sample_processes('after')
    report = collector.seal(producers_stopped=True)
    assert not report['complete'] and report['error_count'] == 1
    assert len(report['observations']) == 3 and len(report['process_queries']) == 2
    assert report['observations'][0]['query_started_ns'] == 3
    assert report['observations'][0]['query_ended_ns'] == 4
    assert not collector.sample_gpu()
    assert collector.seal(producers_stopped=True)['error_count'] == 3


def test_parent_query_failure_is_sticky_and_does_not_expose_exception_text():
    def query(kind):
        raise OSError('PRIVATE-GPU-UUID-OR-PATH')

    collector = activity.ParentActivity(gpu_uuid=UUID, query=query)
    assert not collector.sample_gpu()
    report = collector.seal(producers_stopped=True)
    assert not report['complete'] and report['errors'] == [{'code': 'query_or_parse_failed'}]
    assert 'PRIVATE-GPU-UUID-OR-PATH' not in repr(report)


def test_parent_rejects_overlap_and_inflight_seal_without_losing_evidence():
    entered, release = threading.Event(), threading.Event()

    def query(kind):
        entered.set()
        assert release.wait(5)
        return GPU

    collector = activity.ParentActivity(gpu_uuid=UUID, query=query)
    thread = threading.Thread(target=collector.sample_gpu)
    thread.start()
    try:
        assert entered.wait(5)
        assert not collector.sample_processes('before')
        report = collector.seal(producers_stopped=True)
        assert not report['complete'] and report['error_count'] == 2
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert collector.seal(producers_stopped=True)['error_count'] >= 3


def test_control_flow_interrupt_is_rethrown_after_marking_incomplete():
    def query(kind):
        raise KeyboardInterrupt()

    collector = activity.ParentActivity(gpu_uuid=UUID, query=query)
    with pytest.raises(KeyboardInterrupt):
        collector.sample_gpu()
    assert not collector.seal(producers_stopped=True)['complete']


def test_snapshot_is_detached_and_reflects_pending_and_sealed_states():
    entered, release = threading.Event(), threading.Event()

    def query(kind):
        if kind == 'gpu':
            entered.set()
            assert release.wait(5)
            return GPU
        return PROCESSES

    collector = activity.ParentActivity(gpu_uuid=UUID, query=query)
    assert collector.sample_processes('before')
    copy = collector.snapshot()
    copy['process_queries'][0]['processes'][0]['name'] = 'changed'
    assert collector.snapshot()['process_queries'][0]['processes'][0]['name'] != 'changed'
    thread = threading.Thread(target=collector.sample_gpu)
    thread.start()
    try:
        assert entered.wait(5)
        pending = collector.snapshot()
        assert pending['in_flight'] and not pending['complete'] and not pending['sealed']
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert collector.snapshot()['complete'] and not collector.snapshot()['in_flight']
    sealed = collector.seal(producers_stopped=True)
    assert sealed['complete'] and sealed['sealed'] and not sealed['in_flight']


@pytest.mark.parametrize('stage', ['publish', 'failure'])
def test_interruption_before_publication_or_failure_latch_leaves_pending(stage):
    def query(kind):
        if stage == 'failure':
            raise ValueError('private')
        return GPU

    collector = activity.ParentActivity(gpu_uuid=UUID, query=query)
    lines, first = inspect.getsourcelines(activity.ParentActivity._collect)
    needle = "if self._sealed:" if stage == 'publish' else "self._fail_locked('query_or_parse_failed')"
    line = first + next(i for i, text in enumerate(lines) if needle in text)

    def trace(frame, event, arg):
        if frame.f_code is activity.ParentActivity._collect.__code__ and event == 'line' and frame.f_lineno == line:
            raise KeyboardInterrupt('injected')
        return trace

    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt):
            collector.sample_gpu()
    finally:
        sys.settrace(None)
    report = collector.seal(producers_stopped=True)
    assert report['in_flight'] and report['sealed'] and not report['complete']
    assert report['observations'] == []
