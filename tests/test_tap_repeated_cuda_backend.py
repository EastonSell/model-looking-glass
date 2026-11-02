import gc
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from looking_glass import tap_repeated_admission as admission
from looking_glass import tap_repeated_cuda_backend as backend
from looking_glass import tap_repeated_disposal as disposal
from looking_glass import tap_repeated_lean as lean
from looking_glass import tap_repeated_model as fixed_model
from looking_glass import tap_repeated_owner as owner
from looking_glass import tap_repeated_preflight as preflight
from looking_glass import tap_repeated_timing as timing
from looking_glass import tap_repeated_gpu_owner as gpu_owner
from looking_glass import tap_repeated_protocol as protocol
from looking_glass.tap_repeated_protocol import EXPERIMENT_PROTOCOL, GIB, _valid_preflight


class Identity:
    gpu_uuid = 'GPU-private-test'
    workspace_root = 'test-workspace'


class Lease:
    pass


class Session:
    def __init__(self, lease):
        self.identity = Identity()
        self._runtime_verified = True
        self._bound_lease = lease
        self.valid = True
        self.checks = 0

    def check(self, lease):
        self.checks += 1
        if not self.valid or lease is not self._bound_lease:
            raise RuntimeError('lost owner')


@pytest.fixture
def stack(monkeypatch):
    was_enabled = gc.isenabled()
    gc.disable()
    monkeypatch.setattr(owner, '_OPERATIONS_ENABLED', True)
    monkeypatch.setattr(backend, '_CUDA_BACKEND_ENABLED', True)
    monkeypatch.setattr(owner, 'OwnerSession', Session)
    monkeypatch.setattr(owner, 'DeviceLease', Lease)
    monkeypatch.setattr(owner, 'DeviceIdentity', Identity)
    lease, calls = Lease(), []
    session = Session(lease)
    row = {'free_bytes': 6 * GIB, 'total_bytes': 8 * GIB,
           'allocated_bytes': 128 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}
    admitted = admission._Admission(check_owner=lambda: session.check(lease),
        snapshot=lambda: dict(row), total_bytes=8 * GIB, cap_bytes=2 * GIB)
    admitted.acquire()
    operations = owner._OwnerOperations(session, lease, admitted)
    operations.bind_session(0)
    operations.require_disabled_gc()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True
            calls.append('event')

        def record(self, selected_stream):
            assert selected_stream is stream
            calls.append('record')
        def elapsed_time(self, other): calls.append('elapsed'); return 1.0

    class Stream:
        cuda_stream = 0
        device = SimpleNamespace(index=0)

    stream = Stream()
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction',
                        lambda value, device: calls.append(('set', value, device)))
    monkeypatch.setattr(torch.cuda, 'get_per_process_memory_fraction',
                        lambda device: calls.append('get') or 0.25)
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda device: calls.append('default') or stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: calls.append('current') or Stream())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append('sync'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: calls.append('reset'))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: calls.append('free') or (6 * GIB, 8 * GIB))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: calls.append('allocated') or 128 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: calls.append('reserved') or 512 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: calls.append('peak_a') or 128 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: calls.append('peak_r') or 512 * 1024 ** 2)
    try:
        yield operations, session, admitted, calls, Event
    finally:
        if was_enabled: gc.enable()


def test_backend_is_disabled_and_import_does_not_load_torch_in_fresh_process():
    code = '''
import sys
from looking_glass import tap_repeated_cuda_backend as backend
from looking_glass import tap_repeated_model as fixed_model
assert "torch" not in sys.modules
try:
    backend._CudaRuntime(None)
except RuntimeError:
    pass
else:
    raise AssertionError("CUDA backend was unexpectedly enabled")
assert "torch" not in sys.modules
try:
    fixed_model._FixedModelSetup(None)
except RuntimeError:
    pass
else:
    raise AssertionError("fixed model setup was unexpectedly enabled")
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_fake_cuda_calls_bind_cap_stream_events_and_counter_sources(stack):
    operations, _, _, calls, _ = stack
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    assert calls == [('set', 0.25, 0), 'get', 'default', 'event', 'event', 'record', 'record', 'sync']
    operations.call(runtime.current_stream, root='current_stream_before')
    assert operations.call(runtime.is_default_stream, 'current_stream_before') is True
    assert operations.call(runtime.record_start) is None
    assert operations.call(runtime.record_end) is None
    assert operations.call(runtime.synchronize) is None
    assert operations.call(runtime.elapsed_ms) == 1.0
    assert operations.call(runtime.reset_peaks) is None
    assert operations.call(runtime.free_total_bytes) == (6 * GIB, 8 * GIB)
    assert operations.call(runtime.allocated_bytes) == 128 * 1024 ** 2
    assert operations.call(runtime.reserved_bytes) == 512 * 1024 ** 2
    assert operations.call(runtime.peak_allocated_bytes) == 128 * 1024 ** 2
    assert operations.call(runtime.peak_reserved_bytes) == 512 * 1024 ** 2
    assert not torch.cuda.is_initialized()


def test_incorrect_cap_readback_stops_before_stream_or_events(stack, monkeypatch):
    operations, _, admitted, calls, _ = stack
    monkeypatch.setattr(torch.cuda, 'get_per_process_memory_fraction', lambda device: 0.5)
    runtime = backend._CudaRuntime(operations)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        runtime.initialize()
    assert operations.unsafe and runtime.state == 'failed'
    assert calls == [('set', 0.25, 0)]
    assert operations.roots.start_event is None and operations.roots.end_event is None
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_replaced_event_is_rejected_before_recording(stack):
    operations, _, admitted, calls, _ = stack
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    operations.roots.start_event = object()
    before = list(calls)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.call(runtime.record_start)
    assert operations.unsafe and calls == before
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_owner_loss_after_event_assignment_retains_both_event_roots(stack, monkeypatch):
    operations, session, admitted, calls, Event = stack
    count = []

    def event(*, enable_timing):
        result = Event(enable_timing=enable_timing)
        count.append(True)
        if len(count) == 2: session.valid = False
        return result

    monkeypatch.setattr(torch.cuda, 'Event', event)
    runtime = backend._CudaRuntime(operations)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        runtime.initialize()
    assert operations.unsafe and runtime.state == 'failed'
    assert operations.roots.start_event is not None and operations.roots.end_event is not None
    assert calls.count('event') == 2 and admitted.budget.active_leases() == (admitted.lease,)


def test_failed_priming_sync_retains_events_and_never_becomes_ready(stack, monkeypatch):
    operations, _, admitted, calls, _ = stack

    def fail(device):
        calls.append('failed-sync')
        raise RuntimeError('private priming failure')

    monkeypatch.setattr(torch.cuda, 'synchronize', fail)
    runtime = backend._CudaRuntime(operations)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        runtime.initialize()
    assert runtime.state == 'failed' and operations.unsafe
    assert calls[-3:] == ['record', 'record', 'failed-sync']
    assert operations.roots.start_event is not None and operations.roots.end_event is not None
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_owner_loss_in_current_stream_getter_retains_assigned_wrapper(stack, monkeypatch):
    operations, session, admitted, calls, _ = stack
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    dropped = []

    class Stream:
        cuda_stream = 0
        device = SimpleNamespace(index=0)

        def __del__(self):
            dropped.append(not session.valid)

    def current(device):
        calls.append('current-lost')
        result = Stream()
        session.valid = False
        return result

    monkeypatch.setattr(torch.cuda, 'current_stream', current)
    with pytest.raises(RuntimeError, match='owner operation failed') as raised:
        operations.call(runtime.current_stream, root='current_stream_before')
    retained = weakref.ref(operations.roots.current_stream_before)
    raised.value.__traceback__ = None
    raised.value.__context__ = None
    assert retained() is operations.roots.current_stream_before
    assert retained() is not None and not dropped
    assert operations.unsafe and admitted.budget.active_leases() == (admitted.lease,)


def test_interrupted_stream_root_assignment_retains_wrapper_independently(stack, monkeypatch):
    operations, _, admitted, _, _ = stack
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    refs, fired = [], []

    class Stream:
        cuda_stream = 0
        device = SimpleNamespace(index=0)

    def current(device):
        stream = Stream()
        refs.append(weakref.ref(stream))
        return stream

    monkeypatch.setattr(torch.cuda, 'current_stream', current)
    lines, first = inspect.getsourcelines(owner._OwnerOperations.call)
    root_line = first + next(index for index, line in enumerate(lines)
                             if 'setattr(self.roots, root, result)' in line)

    def interrupt(frame, event, arg):
        if (event == 'line' and frame.f_code is owner._OwnerOperations.call.__code__
                and frame.f_lineno == root_line and not fired):
            fired.append(True)
            raise KeyboardInterrupt('private stream assignment interruption')
        return interrupt

    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt) as raised:
            operations.call(runtime.current_stream, root='current_stream_before')
    finally:
        sys.settrace(None)
    raised.value.__traceback__ = None
    raised.value.__context__ = None
    operations.roots.failure = None
    assert fired == [True] and operations.roots.current_stream_before is None
    assert refs[0]() is operations.roots.pending_return
    assert operations.unsafe and admitted.budget.active_leases() == (admitted.lease,)


def test_current_wrappers_survive_completion_sync_and_release_for_all_modes(stack, monkeypatch):
    transformers = pytest.importorskip('transformers')
    operations, session, _, _, Event = stack
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    runtime.device, runtime.vocab_size, runtime.hidden_size = 'cpu', 128, 16
    monkeypatch.setattr(Event, 'elapsed_time', lambda self, other: 0.0)
    monkeypatch.setattr(timing, '_TIMING_ENABLED', True)
    sync_roots = []

    def observe_sync(device):
        roots = operations.roots
        sync_roots.append((roots.current_stream_before is not None,
                           roots.current_stream_after is not None))

    monkeypatch.setattr(torch.cuda, 'synchronize', observe_sync)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        clock = {'name': 'perf_counter_ns', 'implementation': 'fake', 'monotonic': True,
                 'adjustable': False, 'resolution_seconds': 1e-7}
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=clock)
        counts = []
        for mode in ('none', 'direct', 'gradient'):
            before = session.checks
            runner.sample(mode)
            counts.append(session.checks - before)
            assert operations.roots.current_stream_before is None
            assert operations.roots.current_stream_after is None
        assert counts[0] == counts[1] == counts[2]
        assert sync_roots == [(False, False), (True, True), (False, False)] * 3
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)


def test_fake_backend_and_disposal_release_the_rooted_default_stream(stack, monkeypatch):
    operations, _, admitted, _, _ = stack

    class Stream:
        cuda_stream = 0
        device = SimpleNamespace(index=0)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True
        def record(self, selected_stream):
            assert selected_stream is operations.roots.default_stream

    class Resource:
        pass

    monkeypatch.setattr(torch.cuda, 'default_stream', lambda device: Stream())
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    operations.roots.model = Resource()
    operations.roots.converted = operations.roots.model
    operations.roots.input_ids = Resource()
    refs = [weakref.ref(getattr(operations.roots, name)) for name in
            ('model', 'converted', 'input_ids', 'default_stream', 'start_event', 'end_event')]
    coordinator = disposal._Disposal(operations, synchronize=runtime.synchronize,
                                      cleanup=(), final_counters=(lambda: 0, lambda: 0))
    coordinator.finish()
    assert all(ref() is None for ref in refs)
    assert admitted.state == 'released' and admitted.budget.active_leases() == ()
    assert not torch.cuda.is_initialized()


def test_fixed_model_setup_rejects_an_unverified_bundle_before_transfer(stack, monkeypatch):
    from looking_glass import pretrained
    operations, _, admitted, calls, _ = stack
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    runtime.device = 'cpu'
    monkeypatch.setattr(timing, '_TIMING_ENABLED', True)
    monkeypatch.setattr(fixed_model, '_MODEL_SETUP_ENABLED', True)
    monkeypatch.setattr(fixed_model, '_TARGET_DEVICE', 'cpu')
    monkeypatch.setattr(pretrained, 'load_distilgpt2', lambda **kwargs: object())
    clock = {'name': 'perf_counter_ns', 'implementation': 'fake', 'monotonic': True,
             'adjustable': False, 'resolution_seconds': 1e-7}
    sample = timing._TimedSample(operations, runtime, session=0, clock_info=clock)
    setup = fixed_model._FixedModelSetup(sample)
    before = list(calls)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        setup.run('private-cache')
    assert calls == before and operations.unsafe and setup.state == 'failed'
    assert operations.roots.model_bundle is not None and operations.roots.model is None
    assert admitted.budget.active_leases() == (admitted.lease,)
    assert not torch.cuda.is_initialized()


def test_fixed_model_setup_rechecks_three_gib_immediately_before_transfer(stack, monkeypatch):
    from looking_glass import pretrained
    operations, _, admitted, _, _ = stack
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    runtime.device = 'cpu'
    monkeypatch.setattr(timing, '_TIMING_ENABLED', True)
    monkeypatch.setattr(fixed_model, '_MODEL_SETUP_ENABLED', True)
    monkeypatch.setattr(fixed_model, '_TARGET_DEVICE', 'cpu')
    transfers = []
    class Model:
        def to(self, **_kwargs):
            transfers.append(True)
            raise AssertionError('transfer must not start')
    bundle = SimpleNamespace(model=Model())
    monkeypatch.setattr(pretrained, 'load_distilgpt2', lambda **kwargs: bundle)
    monkeypatch.setattr(fixed_model, '_verify_bundle', lambda _bundle: {'verified': True})
    # These counters satisfy the remaining-cap rule but violate the distinct
    # 3 GiB pre-transfer floor, after admission had already succeeded.
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (5 * GIB // 2, 8 * GIB))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: GIB)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 3 * GIB // 2)
    clock = {'name': 'perf_counter_ns', 'implementation': 'fake', 'monotonic': True,
             'adjustable': False, 'resolution_seconds': 1e-7}
    sample = timing._TimedSample(operations, runtime, session=0, clock_info=clock)
    setup = fixed_model._FixedModelSetup(sample)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        setup.run('private-cache')
    assert not transfers and operations.roots.model is None
    assert operations.roots.model_bundle is bundle and operations.unsafe
    assert admitted.budget.active_leases() == (admitted.lease,)
    assert setup.partial['memory_before_transfer']['free_bytes'] == 5 * GIB // 2
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(not os.environ.get('LG_PINNED_CACHE'), reason='offline pinned checkpoint not provided')
def test_real_pinned_checkpoint_setup_on_cpu_with_cuda_denied(stack, monkeypatch, tmp_path):
    operations, _, _, _, _ = stack
    # Do not let the fake backend itself retain stream wrappers: final
    # disposal must establish that every owner-rooted wrapper is released.
    class EphemeralStream:
        cuda_stream = 0
        device = SimpleNamespace(index=0)
    class EphemeralEvent:
        def __init__(self, *, enable_timing): assert enable_timing is True
        def record(self, stream): assert stream.cuda_stream == 0
        def elapsed_time(self, _other): return 1.0
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda device: EphemeralStream())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: EphemeralStream())
    monkeypatch.setattr(torch.cuda, 'Event', EphemeralEvent)
    runtime = backend._CudaRuntime(operations)
    runtime.initialize()
    runtime.device = 'cpu'
    monkeypatch.setattr(timing, '_TIMING_ENABLED', True)
    monkeypatch.setattr(fixed_model, '_MODEL_SETUP_ENABLED', True)
    monkeypatch.setattr(fixed_model, '_TARGET_DEVICE', 'cpu')
    clock = {'name': 'perf_counter_ns', 'implementation': 'fake', 'monotonic': True,
             'adjustable': False, 'resolution_seconds': 1e-7}
    sample = timing._TimedSample(operations, runtime, session=0, clock_info=clock)
    setup = fixed_model._FixedModelSetup(sample)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        identity = setup.run(os.environ['LG_PINNED_CACHE'])
        assert identity['model_revision'] == '2290a62682d06624634c1f46a6ad5be0f47f38aa'
        assert identity['parameter_count'] == 81912576 and identity['input_ids'] == [15496, 995]
        assert operations.roots.model is operations.roots.converted
        assert operations.roots.input_ids.tolist() == [[15496, 995]]
        assert operations.roots.model_bundle is None and not operations.unsafe
        runner = preflight._FixedPreflight(sample)
        runner.prepare()
        row, post = runner.run(lambda lower, upper: None)
        assert _valid_preflight(row, 'cpu', 0, EXPERIMENT_PROTOCOL)
        assert row['checks']['value_tensor_bytes'] == 3072
        assert post['completed_ns'] >= row['ended_ns'] and not operations.unsafe
        monkeypatch.setattr(lean, '_LEAN_ENABLED', True)
        runner = lean._LeanTimedSample(sample)
        schedule = protocol.load_schedule(Path(__file__).resolve().parents[1]
                                          / 'evidence/gpu-tap-latency-repeated-v1/schedule.json')
        journal = gpu_owner._Journal(tmp_path / 'journal.jsonl')
        journal.append(row)
        journal.append(post)
        for item in protocol.session_order(schedule, 0):
            timed = runner.sample_mode(item['mode'])
            assert timed['output_shape'] == [1, 2, 50257]
            assert timed['wall_ns'] == timed['completion_ns'] - timed['started_ns']
            journal.append({**item, 'session': 0, 'postcheck_status': 'pending', **timed})
            journal.append({'stage': 'postcheck', 'observation_id': item['observation_id'],
                            'status': 'passed', 'completed_ns': timed['postcheck_completed_ns'],
                            'memory_after': timed['memory_after'], 'device_work_complete': True})
        journal.close()
        rows = [json.loads(line) for line in (tmp_path / 'journal.jsonl').read_bytes().splitlines()]
        summary = protocol.summarize_session(schedule, 0, rows, clock=clock, total_bytes=8 * GIB,
            cap_bytes=2 * GIB, device='cpu', run_protocol=EXPERIMENT_PROTOCOL)
        assert summary['warmup_samples'] == 6 and summary['measured_samples'] == 36
        assert len(summary['paired_rounds']) == 12
        monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda _device: 0)
        monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda _device: 0)
        cleanup = disposal._Disposal(operations, synchronize=runtime.synchronize,
            cleanup=(lambda: None, lambda: None, lambda: None),
            final_counters=(lambda: torch.cuda.memory_allocated(0), lambda: torch.cuda.memory_reserved(0)))
        cleanup.finish()
        assert cleanup.state == 'released' and cleanup.counters == {'allocated_bytes': 0, 'reserved_bytes': 0}
        assert operations.admitted.state == 'released' and not operations.admitted.budget.active_leases()
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
