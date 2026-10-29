import gc
import inspect
import sys
import traceback
import weakref

import pytest
import torch

from looking_glass import tap_repeated_admission as admission
from looking_glass import tap_repeated_owner as owner
from looking_glass import tap_repeated_preflight as preflight
from looking_glass import tap_repeated_lean as lean
from looking_glass import tap_repeated_timing as timing
from looking_glass.tap_repeated_protocol import GIB


CLOCK = {'name': 'perf_counter_ns', 'implementation': 'fake', 'monotonic': True,
         'adjustable': False, 'resolution_seconds': 1e-7}


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
        self.checks = 0
        self.valid = True

    def check(self, lease):
        self.checks += 1
        if not self.valid or lease is not self._bound_lease:
            raise RuntimeError('lost owner')


class Runtime:
    device, vocab_size, hidden_size = 'cpu', 128, 16

    def __init__(self, operations):
        self.operations, self.calls, self.sync_roots = operations, [], []

    def current_stream(self):
        self.calls.append('current')
        class Stream:
            pass
        return Stream()
    def is_default_stream(self, slot): self.calls.append('default'); return True
    def record_start(self): self.calls.append('start')
    def record_end(self): self.calls.append('end')
    def elapsed_ms(self): self.calls.append('elapsed'); return 0.0
    def reset_peaks(self): self.calls.append('reset')
    def free_total_bytes(self): self.calls.append('free'); return 6 * GIB, 8 * GIB
    def allocated_bytes(self): self.calls.append('allocated'); return 128 * 1024 ** 2
    def reserved_bytes(self): self.calls.append('reserved'); return 512 * 1024 ** 2
    def peak_allocated_bytes(self): self.calls.append('peak_allocated'); return 128 * 1024 ** 2
    def peak_reserved_bytes(self): self.calls.append('peak_reserved'); return 512 * 1024 ** 2

    def synchronize(self):
        self.calls.append('sync')
        roots = self.operations.roots
        self.sync_roots.append([getattr(roots, name) is not None
                                for name in ('model_result', 'output', 'capture', 'trace', 'trace_events')])


def make_stack(monkeypatch):
    monkeypatch.setattr(owner, '_OPERATIONS_ENABLED', True)
    monkeypatch.setattr(timing, '_TIMING_ENABLED', True)
    monkeypatch.setattr(owner, 'OwnerSession', Session)
    monkeypatch.setattr(owner, 'DeviceLease', Lease)
    monkeypatch.setattr(owner, 'DeviceIdentity', Identity)
    lease = Lease()
    session = Session(lease)
    row = {'free_bytes': 6 * GIB, 'total_bytes': 8 * GIB,
           'allocated_bytes': 128 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}
    admitted = admission._Admission(check_owner=lambda: session.check(lease),
        snapshot=lambda: dict(row), total_bytes=8 * GIB, cap_bytes=2 * GIB)
    admitted.acquire()
    operations = owner._OwnerOperations(session, lease, admitted)
    return operations, session, admitted


def test_timing_entry_is_disabled_before_owner_or_device_access():
    assert timing._TIMING_ENABLED is False
    with pytest.raises(RuntimeError, match='disabled'):
        timing._TimedSample(None, None, session=0, clock_info=CLOCK)
    assert not torch.cuda.is_initialized()


def test_real_cpu_model_uses_matched_guards_and_completion_boundaries(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, session, _ = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        results, counts = [], []
        for mode in ('none', 'direct', 'gradient'):
            before = session.checks
            results.append(runner.sample(mode))
            counts.append(session.checks - before)
        assert counts[0] == counts[1] == counts[2]
        assert [row['events'] for row in results] == [0, 1, 1]
        assert [row['recorded_tensor_bytes'] for row in results] == [0, 64, 64]
        assert all(row['output_shape'] == [1, 2, 128] and row['wall_ns'] > 0 for row in results)
        expected = ['sync', 'reset', 'free', 'allocated', 'reserved', 'current', 'default', 'start',
                    'current', 'default', 'end', 'sync', 'elapsed', 'sync', 'peak_allocated', 'peak_reserved',
                    'free', 'allocated', 'reserved']
        assert runtime.calls == expected * 3
        assert runtime.sync_roots == [[False] * 5, [True, True, False, False, False], [False] * 5,
                                      [False] * 5, [True] * 5, [False] * 5,
                                      [False] * 5, [True] * 5, [False] * 5]
        assert all(getattr(operations.roots, name) is None
                   for name in ('model_result', 'output', 'capture', 'trace', 'trace_events'))
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


def test_real_cpu_four_path_preflight_holds_every_root_through_sync(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, _ = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        forwards = []
        operations.roots.model.register_forward_pre_hook(lambda *args: forwards.append(True))
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        observed, grants = [], []

        def sync():
            observed.append(tuple(getattr(operations.roots, name) is not None
                                  for name in preflight._PREFLIGHT_NAMES))
            runtime.calls.append('sync')

        runtime.synchronize = sync
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        runner = preflight._FixedPreflight(sample)
        duplicate = preflight._FixedPreflight(sample)
        runner.prepare()
        row, post = runner.run(lambda lower, upper: grants.append((lower, upper)))
        assert len(grants) == 1 and grants[0][0] <= row['started_ns'] <= grants[0][1]
        assert (row['gate_lower_ns'], row['gate_upper_ns']) == grants[0]
        assert (row['memory_snapshot_started_ns'] <= row['memory_snapshot_ended_ns']
                <= row['gate_lower_ns'] <= row['started_ns'])
        assert row['started_ns'] - row['memory_snapshot_started_ns'] <= 2_000_000_000
        assert runner.state == 'complete'
        assert row['checks']['preflight_modes'] == ['none', 'id', 'direct', 'gradient']
        assert row['checks']['outputs_exact'] and row['checks']['values_exact']
        assert row['checks']['id_value_absent'] and row['checks']['value_tensor_bytes'] == 64
        assert row['checks']['output_shape'] == [1, 2, 128]
        assert row['ended_ns'] <= post['completed_ns']
        with pytest.raises(ValueError, match='exact ready timing owner'):
            preflight._FixedPreflight(sample)
        assert observed == [(False,) * len(preflight._PREFLIGHT_NAMES),
                            (True,) * len(preflight._PREFLIGHT_NAMES),
                            (False,) * len(preflight._PREFLIGHT_NAMES)]
        assert not operations.unsafe and not torch.cuda.is_initialized()
        assert len(forwards) == 4
        with pytest.raises(RuntimeError, match='owner operation failed'):
            duplicate.prepare()
        assert len(forwards) == 4 and operations.unsafe
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


def test_lean_timing_uses_one_outer_guard_and_separate_wall_intervals(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, session, _ = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        first = preflight._FixedPreflight(sample)
        first.prepare()
        first.run(lambda lower, upper: None)
        monkeypatch.setattr(lean, '_LEAN_ENABLED', True)
        runner = lean._LeanTimedSample(sample)
        checks = []

        def start():
            checks.append(('start', session.checks))
        def end():
            checks.append(('end', session.checks))

        runtime.record_start, runtime.record_end = start, end
        for mode in ('none', 'direct', 'gradient'):
            row = runner.sample_mode(mode)
            assert row['started_ns'] < row['completion_ns'] <= row['ended_ns']
            assert row['wall_ns'] == row['completion_ns'] - row['started_ns']
            assert row['guarded_wall_ns'] == row['guarded_ended_ns'] - row['guarded_started_ns']
            assert row['guarded_started_ns'] <= row['started_ns']
            assert row['guarded_ended_ns'] >= row['postcheck_completed_ns'] >= row['ended_ns']
            assert all(getattr(operations.roots, name) is None for name in lean._TRANSIENTS)
        assert [name for name, _ in checks] == ['start', 'end'] * 3
        assert all(checks[index][1] == checks[index + 1][1] for index in (0, 2, 4))
        assert not operations.unsafe and not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


def test_lean_postcheck_detects_loss_and_keeps_synchronized_results(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, session, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        first = preflight._FixedPreflight(sample)
        first.prepare()
        first.run(lambda lower, upper: None)
        monkeypatch.setattr(lean, '_LEAN_ENABLED', True)
        runner = lean._LeanTimedSample(sample)
        real_forward = sample._forward

        def forward_then_revoke():
            result = real_forward()
            session.valid = False
            return result

        sample._forward = forward_then_revoke
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.sample_mode('direct')
        assert operations.unsafe and runner.state == 'failed'
        assert all(getattr(operations.roots, name) is not None
                   for name in ('model_result', 'output', 'trace', 'trace_events', 'capture'))
        assert runtime.calls[-1] == 'elapsed'
        assert admitted.budget.active_leases() == (admitted.lease,)
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


@pytest.mark.parametrize('fault', ('assignment', 'unexpected_record_result'))
def test_lean_preserves_assigned_return_on_failure_and_stops(monkeypatch, fault):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        first = preflight._FixedPreflight(sample)
        first.prepare()
        first.run(lambda lower, upper: None)
        monkeypatch.setattr(lean, '_LEAN_ENABLED', True)
        runner = lean._LeanTimedSample(sample)
        refs, fired = [], []

        class Resource:
            pass

        def resource():
            value = Resource()
            refs.append(weakref.ref(value))
            return value

        if fault == 'assignment':
            runtime.current_stream = resource
            lines, first_line = inspect.getsourcelines(lean._LeanTimedSample._capture)
            target = first_line + next(index for index, line in enumerate(lines)
                                       if 'setattr(roots, slot, value)' in line)

            def interrupt(frame, event, arg):
                if (event == 'line' and frame.f_code is lean._LeanTimedSample._capture.__code__
                        and frame.f_lineno == target and not fired):
                    fired.append(True)
                    raise KeyboardInterrupt('private assignment interruption')
                return interrupt
        else:
            runtime.record_start = resource
            interrupt = None
        try:
            if interrupt is not None:
                sys.settrace(interrupt)
            with pytest.raises(KeyboardInterrupt if fault == 'assignment' else RuntimeError) as raised:
                runner.sample_mode('none')
        finally:
            sys.settrace(None)
        raised.value.__traceback__ = None
        raised.value.__context__ = None
        operations.roots.failure = None
        runner.failure = None
        assert refs[0]() is operations.roots.pending_return and refs[0]() is not None
        assert operations.unsafe and runner.state == 'failed'
        assert admitted.budget.active_leases() == (admitted.lease,)
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


@pytest.mark.parametrize('phase', ('start', 'forward', 'validation', 'end', 'completion_sync'))
def test_lean_internal_failure_stops_without_further_device_calls(monkeypatch, phase):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        first = preflight._FixedPreflight(sample)
        first.prepare()
        first.run(lambda lower, upper: None)
        monkeypatch.setattr(lean, '_LEAN_ENABLED', True)
        runner = lean._LeanTimedSample(sample)
        runtime.calls.clear()

        def fail(name):
            runtime.calls.append(name)
            raise RuntimeError('private measured failure')

        if phase == 'start':
            runtime.record_start = lambda: fail('failed-start')
        elif phase == 'forward':
            real_forward = sample._forward
            def fail_forward():
                result = real_forward()
                assert result.logits is not None
                fail('failed-forward')
            sample._forward = fail_forward
        elif phase == 'validation':
            sample._validate = lambda mode: fail('failed-validation')
        elif phase == 'end':
            runtime.record_end = lambda: fail('failed-end')
        else:
            sync_count = []
            def fail_second_sync():
                sync_count.append(True)
                if len(sync_count) == 2:
                    fail('failed-completion-sync')
                runtime.calls.append('sync')
            runtime.synchronize = fail_second_sync
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.sample_mode('direct')
        assert operations.unsafe and runner.state == 'failed'
        assert runtime.calls[-1].startswith('failed-')
        if phase in ('validation', 'end', 'completion_sync'):
            assert all(getattr(operations.roots, name) is not None
                       for name in ('model_result', 'output', 'trace', 'trace_events', 'capture'))
        if phase == 'forward':
            assert operations.roots.trace is not None and operations.roots.trace_events is not None
            assert operations.roots.failure is not None
        before = list(runtime.calls)
        with pytest.raises(RuntimeError, match='not ready'):
            runner.sample_mode('none')
        assert runtime.calls == before and admitted.budget.active_leases() == (admitted.lease,)
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


@pytest.mark.parametrize('response', ('raise', 'false'))
def test_preflight_rejected_gate_stops_before_first_model_call(monkeypatch, response):
    transformers = pytest.importorskip('transformers')
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        operations.roots.model.forward = lambda **kwargs: pytest.fail('the model must not run')
        runner = preflight._FixedPreflight(sample)
        runner.prepare()

        def reject(lower, upper):
            if response == 'raise':
                raise RuntimeError('start grant unavailable')
            return False

        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.run(reject)
        assert runner.state == 'failed' and operations.unsafe
        assert operations.roots.preflight_result_0 is None
        assert operations.roots.current_stream_before is not None
        assert runtime.calls.count('sync') == 1 and admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


def test_cuda_timing_rejects_missing_preflight_before_any_device_callback(monkeypatch):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        runtime.device = 'cuda:0'
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError, match='owner operation failed'):
            sample.sample('none')
        assert runtime.calls == [] and operations.unsafe
        assert admitted.budget.active_leases() == (admitted.lease,)
        assert not torch.cuda.is_initialized()
    finally:
        if was_enabled: gc.enable()


def test_preflight_external_alias_stops_before_disposal_sync(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime, alias, syncs = Runtime(operations), [], []

        def sync():
            syncs.append(True)
            if len(syncs) == 2:
                alias.append(operations.roots.preflight_output_0)

        runtime.synchronize = sync
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        runner = preflight._FixedPreflight(sample)
        runner.prepare()
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.run(lambda lower, upper: None)
        assert len(syncs) == 2 and alias and runner.state == 'failed' and operations.unsafe
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


@pytest.mark.parametrize('failure', ('revoked_owner', 'delayed_gate'))
def test_preflight_rechecks_owner_and_freshness_after_gate(monkeypatch, failure):
    transformers = pytest.importorskip('transformers')
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, session, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        times = iter((100, 101, 102, 102 + preflight._START_GATE_MAX_NS + 1))
        sample = timing._TimedSample(operations, Runtime(operations), session=0,
                                     clock_info=CLOCK, clock=lambda: next(times))
        operations.roots.model.forward = lambda **kwargs: pytest.fail('a stale or revoked grant must not run the model')
        runner = preflight._FixedPreflight(sample)
        runner.prepare()

        def gate(lower, upper):
            assert upper - lower == preflight._START_GATE_MAX_NS
            if failure == 'revoked_owner':
                session.valid = False

        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.run(gate)
        assert operations.unsafe and operations.roots.preflight_result_0 is None
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


def test_preflight_enters_contexts_before_checking_grant_freshness(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.model.forward = lambda **kwargs: pytest.fail('a stale grant must not run the model')
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        now, checked = [100], []

        class DelayedAutocast:
            def __enter__(self):
                now[0] += 2_000_000_000
            def __exit__(self, *args):
                return False

        monkeypatch.setattr(torch, 'autocast', lambda **kwargs: DelayedAutocast())
        sample = timing._TimedSample(operations, Runtime(operations), session=0,
                                     clock_info=CLOCK, clock=lambda: now[0])
        runner = preflight._FixedPreflight(sample)
        runner.prepare()

        def gate(lower, upper):
            checked.append(lower)
            if lower > 500_000_000:
                raise RuntimeError('grant expired')

        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.run(gate)
        assert checked == [2_000_000_100] and operations.unsafe
        assert operations.roots.preflight_result_0 is None
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


@pytest.mark.parametrize('bad_counter', ('free', 'allocated'))
def test_preflight_rejects_fresh_bad_counter_before_later_queries_or_forward(monkeypatch, bad_counter):
    transformers = pytest.importorskip('transformers')
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.model.forward = lambda **kwargs: pytest.fail('invalid headroom must stop the model')
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        runner = preflight._FixedPreflight(sample)
        runner.prepare()
        before = len(runtime.calls)
        if bad_counter == 'free':
            runtime.free_total_bytes = lambda: runtime.calls.append('bad_free') or (6 * GIB, 7 * GIB)
        else:
            runtime.allocated_bytes = lambda: runtime.calls.append('bad_allocated') or -1
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.run(lambda lower, upper: None)
        assert runtime.calls[before:] == (['bad_free'] if bad_counter == 'free' else ['free', 'bad_allocated'])
        assert operations.unsafe and operations.roots.preflight_result_0 is None
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


def test_preflight_rechecks_owner_after_fresh_memory_snapshot(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, session, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.model.forward = lambda **kwargs: pytest.fail('lost owner must stop the model')
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        sample = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        runner = preflight._FixedPreflight(sample)
        runner.prepare()

        def free_then_revoke():
            runtime.calls.append('free-and-revoke')
            session.valid = False
            return 6 * GIB, 8 * GIB

        runtime.free_total_bytes = free_then_revoke
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.run(lambda lower, upper: None)
        assert operations.unsafe and operations.roots.preflight_result_0 is None
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


def test_failed_completion_sync_retains_outputs_and_forbids_later_sample(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)

        def fail():
            runtime.calls.append('sync')
            if runtime.calls.count('sync') == 2:
                raise KeyboardInterrupt('private sync failure')

        runtime.synchronize = fail
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(KeyboardInterrupt):
            runner.sample('direct')
        assert operations.unsafe and runner.state == 'failed'
        assert all(getattr(operations.roots, name) is not None
                   for name in ('model_result', 'output', 'capture', 'trace', 'trace_events'))
        assert admitted.budget.active_leases() == (admitted.lease,)
        with pytest.raises(RuntimeError, match='not ready'):
            runner.sample('none')
        assert runtime.calls.count('sync') == 2 and not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


def test_gc_reenabled_during_completion_sync_stops_before_next_device_operation(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)

        def sync_then_enable_gc():
            runtime.calls.append('sync')
            if runtime.calls.count('sync') == 2:
                gc.enable()

        runtime.synchronize = sync_then_enable_gc
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.sample('direct')
        assert operations.unsafe and runner.state == 'failed'
        assert runtime.calls[-1] == 'sync' and 'elapsed' not in runtime.calls
        assert operations.roots.output is not None and operations.roots.trace_events is not None
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()
        else: gc.disable()


@pytest.mark.parametrize('mode', ('none', 'direct', 'gradient'))
@pytest.mark.parametrize('slot', ('preflight_capture_0', 'preflight_output_0'))
def test_timed_mode_rejects_live_preflight_resource_before_any_device_call(monkeypatch, mode, slot):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        operations.bind_session(0)
        operations.require_disabled_gc()
        setattr(operations.roots, slot, object())
        runtime = Runtime(operations)
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError):
            runner.sample(mode)
        assert runtime.calls == [] and operations.unsafe
        assert getattr(operations.roots, slot) is not None
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


def test_external_capture_alias_blocks_reuse_and_second_sync(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime, aliases = Runtime(operations), []

        def sync_with_alias():
            runtime.calls.append('sync')
            if runtime.calls.count('sync') == 2:
                aliases.append(operations.roots.capture)

        runtime.synchronize = sync_with_alias
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.sample('direct')
        assert operations.unsafe and aliases[0] is not None
        assert runtime.calls.count('sync') == 2 and 'peak_allocated' not in runtime.calls
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


@pytest.mark.parametrize('failure', ['sync', 'reset'])
def test_pre_sample_failure_never_starts_event_or_forward(monkeypatch, failure):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)

        def fail():
            runtime.calls.append('failed-' + failure)
            raise RuntimeError('private pre-sample failure')

        setattr(runtime, 'synchronize' if failure == 'sync' else 'reset_peaks', fail)
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.sample('none')
        assert operations.unsafe and runner.partial['stage'] == 'pre_sample'
        assert 'start' not in runtime.calls and 'default' not in runtime.calls
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


def test_unexpected_assigned_device_return_is_privately_retained(monkeypatch):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime, refs = Runtime(operations), []

        def bad_sync():
            value = torch.tensor([123.0])
            refs.append(weakref.ref(value))
            return value

        runtime.synchronize = bad_sync
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        try:
            runner.sample('none')
        except RuntimeError as public:
            assert public.__context__ is None
            traceback.clear_frames(public.__traceback__)
            public.__traceback__ = None
        else:
            pytest.fail('unexpected device callback return must fail')
        gc.collect()  # CPU-only fault probe; the real owner must not collect.
        assert refs[0]() is runner.pending_return and operations.unsafe
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


@pytest.mark.parametrize('damage', ['changed_total', 'low_free', 'invalid_free', 'invalid_allocated'])
def test_invalid_memory_read_stops_before_later_queries(monkeypatch, damage):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operations, _, admitted = make_stack(monkeypatch)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)

        def free_total():
            runtime.calls.append('free')
            return {'changed_total': (6 * GIB, 8 * GIB + 1),
                    'low_free': (0, 8 * GIB), 'invalid_free': (True, 8 * GIB),
                    'invalid_allocated': (6 * GIB, 8 * GIB)}[damage]

        def allocated():
            runtime.calls.append('allocated')
            return True

        runtime.free_total_bytes = free_total
        runtime.allocated_bytes = allocated
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError):
            runner.sample('none')
        assert operations.unsafe and 'reserved' not in runtime.calls and 'start' not in runtime.calls
        if damage != 'invalid_allocated': assert 'allocated' not in runtime.calls
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled: gc.enable()


@pytest.mark.parametrize('damage', ['short', 'event_too_long'])
def test_inconsistent_timing_is_retained_as_partial_and_stops(monkeypatch, damage):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        if damage == 'event_too_long': runtime.elapsed_ms = lambda: 1_000_000.0
        ticks = iter((100, 101 if damage == 'short' else 100_100))
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK, clock=lambda: next(ticks))
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.sample('none')
        assert operations.unsafe and runner.state == 'failed'
        assert runner.partial['stage'] == 'postcheck_pending' and runner.partial['wall_ns'] > 0
        if damage == 'event_too_long': assert runner.partial['device_event_ms'] == 1_000_000.0
        assert 'peak_allocated' not in runtime.calls
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


def test_returned_cache_is_rejected_before_sample_success(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        model = transformers.GPT2LMHeadModel(config).eval()
        original = model.forward

        def returning_cache(*args, **kwargs):
            output = original(*args, **kwargs)
            output.past_key_values = ('unexpected-cache',)
            return output

        model.forward = returning_cache
        operations.roots.model = model
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError, match='owner operation failed'):
            runner.sample('none')
        assert operations.unsafe and operations.roots.model_result.past_key_values == ('unexpected-cache',)
        assert 'end' not in runtime.calls and admitted.budget.active_leases() == (admitted.lease,)
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()


def test_invalid_peak_allocated_stops_before_peak_reserved_query(monkeypatch):
    transformers = pytest.importorskip('transformers')
    was_enabled, threads = gc.isenabled(), torch.get_num_threads()
    gc.disable()
    torch.set_num_threads(1)
    try:
        operations, _, admitted = make_stack(monkeypatch)
        config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
        config._attn_implementation = 'eager'
        operations.roots.model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.input_ids = torch.tensor([[10, 11]], dtype=torch.int64)
        operations.bind_session(0)
        operations.require_disabled_gc()
        runtime = Runtime(operations)

        def bad_peak():
            runtime.calls.append('peak_allocated')
            return True

        runtime.peak_allocated_bytes = bad_peak
        runner = timing._TimedSample(operations, runtime, session=0, clock_info=CLOCK)
        with pytest.raises(RuntimeError):
            runner.sample('none')
        assert operations.unsafe and 'peak_reserved' not in runtime.calls
        assert runner.partial['memory_before']['total_bytes'] == 8 * GIB
        assert runner.partial['metadata']['output_finite'] is True
        assert 'peak_allocated_bytes' not in runner.partial
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        torch.set_num_threads(threads)
        if was_enabled: gc.enable()
