import gc
import inspect
import subprocess
import sys
import traceback
import weakref

import pytest
import torch

from looking_glass import tap_repeated_admission as admission
from looking_glass import tap_repeated_owner as owner
from looking_glass.tap_repeated_protocol import GIB


class FakeIdentity:
    gpu_uuid = 'GPU-private-test'
    workspace_root = 'test-workspace'


class FakeLease:
    pass


class FakeSession:
    def __init__(self, lease):
        self.identity = FakeIdentity()
        self._runtime_verified = True
        self._bound_lease = lease
        self.valid = True
        self.checks = 0

    def check(self, lease):
        self.checks += 1
        if not self.valid or lease is not self._bound_lease:
            raise RuntimeError('supervised owner lost')


def stack(monkeypatch):
    monkeypatch.setattr(owner, '_OPERATIONS_ENABLED', True)
    monkeypatch.setattr(owner, 'OwnerSession', FakeSession)
    monkeypatch.setattr(owner, 'DeviceLease', FakeLease)
    monkeypatch.setattr(owner, 'DeviceIdentity', FakeIdentity)
    lease = FakeLease()
    session = FakeSession(lease)
    row = {'free_bytes': 6 * GIB, 'total_bytes': 8 * GIB,
           'allocated_bytes': 128 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}
    admitted = admission._Admission(check_owner=lambda: session.check(lease),
        snapshot=lambda: dict(row), total_bytes=8 * GIB, cap_bytes=2 * GIB)
    admitted.acquire()
    return owner._OwnerOperations(session, lease, admitted), session, admitted


def test_operations_are_disabled_before_owner_or_device_access():
    assert owner._OPERATIONS_ENABLED is False
    with pytest.raises(RuntimeError, match='disabled'):
        owner._OwnerOperations(None, None, None)
    assert torch.cuda.is_initialized() is False


def test_owner_module_import_does_not_import_torch_in_a_fresh_process():
    code = "import sys; import looking_glass.tap_repeated_owner; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_lost_owner_stops_trace_binding_before_torch_import_in_a_fresh_process():
    code = '''
import sys
from looking_glass import tap_repeated_owner as owner
operations = object.__new__(owner._OwnerOperations)
operations.unsafe = False
operations.roots = owner._Roots()
def denied():
    raise RuntimeError("lost owner")
operations._check = denied
try:
    operations.bind_trace_events("trace", session=0)
except RuntimeError:
    pass
else:
    raise AssertionError("owner loss was not detected")
assert operations.unsafe
assert "looking_glass.transformer_trace" not in sys.modules
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_assigned_output_is_rooted_before_owner_postcheck(monkeypatch):
    operations, session, admitted = stack(monkeypatch)
    references = []

    def allocate_then_lose_owner():
        value = torch.tensor([42.0])
        references.append(weakref.ref(value))
        session.valid = False
        return value

    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.call(allocate_then_lose_owner, root='output')
    assert str(operations.roots.failure) == 'supervised owner lost'
    operations.roots.failure = None
    gc.collect()
    assert operations.unsafe and references[0]() is operations.roots.output
    assert admitted.budget.active_leases() == (admitted.lease,)
    called = []
    with pytest.raises(RuntimeError, match='unverified'):
        operations.call(lambda: called.append(True))
    assert called == [] and torch.cuda.is_initialized() is False


def test_interrupted_designated_root_assignment_retains_python_return(monkeypatch):
    operations, _, admitted = stack(monkeypatch)
    lines, first_line = inspect.getsourcelines(owner._OwnerOperations.call)
    root_line = first_line + next(i for i, line in enumerate(lines)
                                  if 'setattr(self.roots, root, result)' in line)
    references, interrupted = [], []

    def allocate():
        value = torch.tensor([9.0])
        references.append(weakref.ref(value))
        return value

    def trace(frame, event, arg):
        if frame.f_code is owner._OwnerOperations.call.__code__ and event == 'line' and frame.f_lineno == root_line:
            interrupted.append(True)
            raise KeyboardInterrupt('root assignment interrupted')
        return trace

    try:
        sys.settrace(trace)
        try:
            operations.call(allocate, root='output')
        except KeyboardInterrupt as exc:
            traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None
    finally:
        sys.settrace(None)
    operations.roots.failure = None
    gc.collect()
    assert interrupted == [True] and operations.unsafe and operations.roots.output is None
    assert references[0]() is operations.roots.pending_return
    assert admitted.budget.active_leases() == (admitted.lease,)
    with pytest.raises(RuntimeError, match='unverified'):
        operations.call(lambda: pytest.fail('no later operation'))


def test_identity_or_trace_claim_change_prevents_operation(monkeypatch):
    operations, session, admitted = stack(monkeypatch)
    session.identity = FakeIdentity()
    session.identity.gpu_uuid = 'GPU-other'
    called = []
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.call(lambda: called.append(True))
    assert str(operations.roots.failure) == 'supervised physical binding changed'
    assert operations.unsafe and called == []

    operations, _, admitted = stack(monkeypatch)
    admitted.lease.release_trace('gpu', 1)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.call(lambda: called.append(True))
    assert 'custody differs' in str(operations.roots.failure)
    assert operations.unsafe and called == []


def test_identity_changed_during_admission_check_prevents_operation(monkeypatch):
    operations, session, admitted = stack(monkeypatch)
    original = admitted.verify

    def verify_then_rebind():
        original()
        session.identity.gpu_uuid = 'GPU-other'

    admitted.verify = verify_then_rebind
    called = []
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.call(lambda: called.append(True))
    assert str(operations.roots.failure) == 'supervised physical binding changed'
    assert operations.unsafe and called == []


def test_root_overwrite_is_rejected_and_custody_is_retained(monkeypatch):
    operations, _, admitted = stack(monkeypatch)
    value = operations.call(lambda: torch.tensor([1.0]), root='output')
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.call(lambda: pytest.fail('must not overwrite'), root='output')
    assert 'occupied' in str(operations.roots.failure)
    assert operations.unsafe and operations.roots.output is value
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_clearing_roots_is_a_trusted_post_sync_step(monkeypatch):
    operations, _, _ = stack(monkeypatch)
    references = []

    def allocate():
        value = torch.tensor([1.0])
        references.append(weakref.ref(value))
        return value

    operations.call(allocate, root='output')
    # This CPU fixture models an already completed synchronization; the helper
    # does not itself prove the real CUDA synchronization or disposal.
    operations.clear_after_synchronization('output')
    gc.collect()
    assert references[0]() is None and not operations.unsafe


def test_owner_loss_during_one_disposal_retains_remaining_roots(monkeypatch):
    operations, session, admitted = stack(monkeypatch)
    disposed, references = [], []

    class First:
        def __del__(self):
            disposed.append('first')
            session.valid = False

    class Second:
        def __del__(self):
            disposed.append('second')

    def second():
        value = Second()
        references.append(weakref.ref(value))
        return value

    operations.call(First, root='output')
    operations.call(second, root='capture')
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.clear_after_synchronization('output', 'capture')
    assert str(operations.roots.failure) == 'supervised owner lost'
    assert operations.unsafe and disposed == ['first']
    assert references[0]() is operations.roots.capture
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_preflight_roots_coexist_across_a_fake_completion_sync(monkeypatch):
    operations, _, _ = stack(monkeypatch)
    slots = ([f'preflight_result_{index}' for index in range(4)]
             + [f'preflight_trace_{index}' for index in range(3)]
             + [f'preflight_capture_{index}' for index in range(2)])
    references, observed = [], []

    def allocate():
        value = torch.tensor([1.0])
        references.append(weakref.ref(value))
        return value

    for slot in slots:
        operations.call(allocate, root=slot)
    operations.call(lambda: observed.append(sum(ref() is not None for ref in references)))
    operations.clear_after_synchronization(*slots)
    gc.collect()
    operations.call(lambda: observed.append(sum(ref() is not None for ref in references)))
    assert observed == [9, 0] and not operations.unsafe


def test_public_error_cannot_clear_private_failure_frames_or_partial_tensor(monkeypatch):
    operations, _, admitted = stack(monkeypatch)
    references = []

    def allocate_then_fail():
        partial = torch.tensor([42.0])
        references.append(weakref.ref(partial))
        raise MemoryError('private operation failure')

    try:
        operations.call(allocate_then_fail)
    except MemoryError as public:
        assert public is not operations.roots.failure
        assert public.__context__ is None and public.__cause__ is None
        traceback.clear_frames(public.__traceback__)
        public.__traceback__ = None
    else:
        pytest.fail('the failure must escape')
    gc.collect()
    assert operations.unsafe and str(operations.roots.failure) == 'private operation failure'
    assert references[0]() is not None
    assert admitted.budget.active_leases() == (admitted.lease,)


@pytest.mark.parametrize('mode', ['direct', 'gradient'])
def test_failed_real_cpu_trace_retains_appended_value_independent_of_session(monkeypatch, mode):
    transformers = pytest.importorskip('transformers')
    from looking_glass.transformer_trace import GPT2TraceSession
    operations, _, admitted = stack(monkeypatch)
    config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
    config._attn_implementation = 'eager'
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = transformers.GPT2LMHeadModel(config).eval()
        operations.roots.model = model
        operations.bind_session(0)
        trace = operations.call(lambda: GPT2TraceSession(model, run_id=f'{owner.EXPERIMENT_PROTOCOL}-session-0', mode=mode,
            targets=('block',), layers=(0,), samples=(0,), tokens=(1,), capture_device='source',
            max_events=2, max_trace_bytes=64 * 1024), root='trace')
        operations.bind_trace_events('trace', session=0)
        retained = operations.roots.trace_events
        assert trace.events is retained and len(retained) == 0
        references = []

        def forward_then_fail():
            with torch.no_grad(), trace:
                model(input_ids=torch.tensor([[10, 11]]), use_cache=False, return_dict=True)
                assert len(trace.events) == 1
                references.append(weakref.ref(trace.events[0].value))
                raise RuntimeError('private failure after capture')

        try:
            operations.call(forward_then_fail)
        except RuntimeError as public:
            assert str(public) == 'owner operation failed' and public.__context__ is None
            traceback.clear_frames(public.__traceback__)
            public.__traceback__ = None
        else:
            pytest.fail('the failed trace must not become a success')
        assert not trace.successful and operations.unsafe
        # Prove the separate container owns the event after the session and
        # private exception root are dropped in this CPU-only test.
        operations.roots.trace = None
        operations.roots.failure = None
        del trace
        gc.collect()
        assert references[0]() is retained[0].value
        assert admitted.budget.active_leases() == (admitted.lease,)
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(previous)


def test_trace_event_inventory_rejects_non_events_and_excess():
    from looking_glass.transformer_trace import GPT2TraceEvent
    retained = owner._RetainedEvents(GPT2TraceEvent)
    with pytest.raises(RuntimeError, match='inventory'):
        retained.append(object())
    assert retained == []


def test_trace_binding_requires_a_fresh_pre_rooted_session(monkeypatch):
    operations, _, admitted = stack(monkeypatch)
    operations.bind_session(0)
    operations.roots.trace = object()
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.bind_trace_events('trace', session=0)
    assert operations.unsafe and operations.roots.trace_events is None
    assert 'fresh bounded owner trace' in str(operations.roots.failure)
    assert admitted.budget.active_leases() == (admitted.lease,)


@pytest.mark.parametrize('damage', ['third_value_trace', 'wrong_token', 'wrong_run', 'overlapping_preflight',
                                    'overlapping_capture', 'overlapping_result', 'overlapping_output',
                                    'session_mismatch'])
def test_fixed_trace_claim_rejects_wrong_mode_selection_or_overlap(monkeypatch, damage):
    transformers = pytest.importorskip('transformers')
    from looking_glass.transformer_trace import GPT2TraceSession
    operations, _, admitted = stack(monkeypatch)
    config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
    config._attn_implementation = 'eager'
    model = transformers.GPT2LMHeadModel(config).eval()
    operations.roots.model = model
    operations.bind_session(0)
    trace_slot = 'trace' if damage in ('overlapping_preflight', 'overlapping_capture', 'overlapping_result', 'overlapping_output') else 'preflight_trace_0'
    if damage == 'overlapping_preflight': operations.roots.preflight_events_2 = object()
    if damage == 'overlapping_capture': operations.roots.preflight_capture_0 = object()
    if damage == 'overlapping_result': operations.roots.preflight_result_0 = object()
    if damage == 'overlapping_output': operations.roots.preflight_output_0 = object()
    mode = 'direct' if damage in ('third_value_trace', 'overlapping_preflight',
                                  'overlapping_capture', 'overlapping_result', 'overlapping_output') else 'id'
    token = (0,) if damage == 'wrong_token' else (1,)
    requested_session = 1 if damage == 'session_mismatch' else 0
    run_id = 'wrong' if damage == 'wrong_run' else f'{owner.EXPERIMENT_PROTOCOL}-session-{requested_session}'
    trace = operations.call(lambda: GPT2TraceSession(model, run_id=run_id, mode=mode,
        targets=('block',), layers=(0,), samples=(0,), tokens=token, capture_device='source',
        max_events=2, max_trace_bytes=64 * 1024), root=trace_slot)
    with pytest.raises(RuntimeError, match='owner operation failed'):
        operations.bind_trace_events(trace_slot, session=requested_session)
    assert operations.unsafe and trace._used is False
    assert admitted.budget.active_leases() == (admitted.lease,)
    assert not torch.cuda.is_initialized()


def test_preflight_slots_allow_only_one_id_direct_and_gradient_session(monkeypatch):
    transformers = pytest.importorskip('transformers')
    from looking_glass.transformer_trace import GPT2TraceSession
    operations, _, _ = stack(monkeypatch)
    config = transformers.GPT2Config(vocab_size=128, n_embd=16, n_layer=1, n_head=2, n_positions=8)
    config._attn_implementation = 'eager'
    model = transformers.GPT2LMHeadModel(config).eval()
    operations.roots.model = model
    operations.bind_session(0)
    for index, mode in enumerate(('id', 'direct', 'gradient')):
        trace = operations.call(lambda: GPT2TraceSession(model,
            run_id=f'{owner.EXPERIMENT_PROTOCOL}-session-0', mode=mode,
            targets=('block',), layers=(0,), samples=(0,), tokens=(1,), capture_device='source',
            max_events=2, max_trace_bytes=64 * 1024), root=f'preflight_trace_{index}')
        operations.bind_trace_events(f'preflight_trace_{index}', session=0)
        assert trace.events is getattr(operations.roots, f'preflight_events_{index}')
    assert not operations.unsafe and not torch.cuda.is_initialized()
