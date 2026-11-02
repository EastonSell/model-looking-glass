"""CPU-only lifetime checks for the proposed repeated-tap teardown."""
import gc
import importlib
import inspect
import sys
import weakref

import pytest

from looking_glass import tap_repeated_admission as admission
from looking_glass import tap_repeated_owner as owner
from looking_glass.tap_repeated_protocol import GIB


@pytest.fixture(autouse=True)
def _disable_automatic_gc():
    was_enabled = gc.isenabled()
    gc.disable()
    yield
    if was_enabled:
        gc.enable()


class _Identity:
    gpu_uuid = 'GPU-test-fixture'
    workspace_root = 'workspace-fixture'


class _Lease:
    pass


class _Session:
    def __init__(self, lease):
        self.identity = _Identity()
        self._runtime_verified = True
        self._bound_lease = lease
        self.valid = True

    def check(self, lease):
        if not self.valid or lease is not self._bound_lease:
            raise RuntimeError('supervised owner lost')


class _Resource:
    def __init__(self, on_delete=None):
        self.on_delete = on_delete

    def __del__(self):
        if self.on_delete is not None:
            self.on_delete()


class _CyclicResource(_Resource):
    def __init__(self, on_delete=None):
        super().__init__(on_delete)
        self.cycle = self


def _stack(monkeypatch):
    monkeypatch.setattr(owner, '_OPERATIONS_ENABLED', True)
    monkeypatch.setattr(owner, 'OwnerSession', _Session)
    monkeypatch.setattr(owner, 'DeviceLease', _Lease)
    monkeypatch.setattr(owner, 'DeviceIdentity', _Identity)
    device = _Lease()
    session = _Session(device)
    row = {'free_bytes': 6 * GIB, 'total_bytes': 8 * GIB,
           'allocated_bytes': 128 * 1024**2, 'reserved_bytes': 512 * 1024**2}
    admitted = admission._Admission(check_owner=lambda: session.check(device),
        snapshot=lambda: dict(row), total_bytes=8 * GIB, cap_bytes=2 * GIB)
    admitted.acquire()
    operations = owner._OwnerOperations(session, device, admitted)
    return operations, session, admitted


def _populate(operations, *, aliased=True):
    model, converted = _Resource(), _Resource()
    if aliased:
        converted = model
    operations.roots.model = model
    operations.roots.converted = converted
    operations.roots.input_ids = _Resource()
    operations.roots.default_stream = _Resource()
    operations.roots.start_event = _Resource()
    operations.roots.end_event = _Resource()
    refs = tuple(weakref.ref(getattr(operations.roots, name)) for name in
                 ('model', 'converted', 'input_ids', 'default_stream', 'start_event', 'end_event'))
    return refs


def test_disposal_keeps_claims_until_three_drains_and_empty_allocator(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    refs = _populate(operations)
    calls = []

    def synchronize():
        calls.append('sync')
        assert admitted.budget.active_leases() == (admitted.lease,)
        if len(calls) == 1:
            assert all(ref() is not None for ref in refs)

    def clean():
        calls.append('clean')
        assert all(ref() is None for ref in refs)
        assert admitted.budget.active_leases() == (admitted.lease,)

    def allocated():
        calls.append('allocated')
        assert admitted.budget.active_leases() == (admitted.lease,)
        return 0

    def reserved():
        calls.append('reserved')
        assert admitted.budget.active_leases() == (admitted.lease,)
        return 0

    coordinator = disposal._Disposal(operations, synchronize=synchronize,
                                     cleanup=(clean,), final_counters=(allocated, reserved))
    coordinator.finish()
    gc.collect()
    assert calls == ['sync', 'sync', 'clean', 'sync', 'allocated', 'reserved']
    assert all(ref() is None for ref in refs)
    assert admitted.budget.active_leases() == ()
    assert admitted.state == 'released'


@pytest.mark.parametrize('failure_at', [1, 2, 3])
@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_failed_sync_stops_without_releasing_claim(monkeypatch, failure_at, failure_type):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    refs = _populate(operations)
    calls = []

    def sync():
        calls.append('sync')
        if calls.count('sync') == failure_at:
            raise failure_type('drain failed')

    coordinator = disposal._Disposal(operations, synchronize=sync,
        cleanup=(lambda: calls.append('clean'),),
        final_counters=(lambda: calls.append('allocated'), lambda: calls.append('reserved')))
    with pytest.raises(failure_type):
        coordinator.finish()
    assert str(operations.roots.failure) == 'drain failed'
    assert calls.count('sync') == failure_at
    assert 'allocated' not in calls and 'reserved' not in calls
    if failure_at == 1:
        assert all(ref() is not None for ref in refs)
    assert admitted.budget.active_leases() == (admitted.lease,)
    assert admitted.lease.trace_bytes('gpu') == admission.GPU_TRACE_CLAIM
    assert operations.unsafe is True
    with pytest.raises(RuntimeError, match='single-use'):
        coordinator.finish()


@pytest.mark.parametrize('slot', ['pending_return', 'failure', 'cleanup_failure',
                                  'trace', 'trace_events', 'preflight_events_2',
                                  'preflight_result_2', 'preflight_output_2', 'capture'])
def test_occupied_failure_or_transient_root_blocks_first_sync(monkeypatch, slot):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    refs = _populate(operations)
    retained = _Resource()
    setattr(operations.roots, slot, retained)
    calls = []
    coordinator = disposal._Disposal(operations, synchronize=lambda: calls.append('sync'),
        cleanup=(), final_counters=(lambda: 0, lambda: 0))
    with pytest.raises(RuntimeError):
        coordinator.finish()
    assert calls == []
    assert all(ref() is not None for ref in refs)
    assert getattr(operations.roots, slot) is retained
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_owner_loss_in_first_finalizer_keeps_remaining_roots(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, session, admitted = _stack(monkeypatch)
    _populate(operations, aliased=False)
    operations.roots.model = _Resource(lambda: setattr(session, 'valid', False))
    names = ('converted', 'input_ids', 'start_event', 'end_event')
    refs = tuple(weakref.ref(getattr(operations.roots, name)) for name in names)
    calls = []
    coordinator = disposal._Disposal(operations, synchronize=lambda: calls.append('sync'),
        cleanup=(), final_counters=(lambda: calls.append('allocated'), lambda: calls.append('reserved')))
    with pytest.raises(RuntimeError):
        coordinator.finish()
    assert calls == ['sync']
    assert all(ref() is not None for ref in refs)
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_sync_that_changes_root_stops_before_dropping_other_resources(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    _populate(operations, aliased=False)
    names = ('converted', 'input_ids', 'start_event', 'end_event')
    refs = tuple(weakref.ref(getattr(operations.roots, name)) for name in names)
    calls = []

    def sync():
        calls.append('sync')
        operations.roots.model = None

    coordinator = disposal._Disposal(operations, synchronize=sync, cleanup=(),
                                      final_counters=(lambda: calls.append('allocated'), lambda: calls.append('reserved')))
    with pytest.raises(RuntimeError, match='root'):
        coordinator.finish()
    assert calls == ['sync']
    assert all(ref() is not None for ref in refs)
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_unweakrefable_or_externally_retained_resource_never_releases(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    _populate(operations)
    operations.roots.model = object()
    calls = []
    coordinator = disposal._Disposal(operations, synchronize=lambda: calls.append('sync'),
        cleanup=(), final_counters=(lambda: calls.append('allocated'), lambda: calls.append('reserved')))
    with pytest.raises(RuntimeError, match='weak-referenced'):
        coordinator.finish()
    assert not calls
    assert admitted.budget.active_leases() == (admitted.lease,)

    operations, _, admitted = _stack(monkeypatch)
    _populate(operations)
    external = operations.roots.input_ids
    calls = []
    coordinator = disposal._Disposal(operations, synchronize=lambda: calls.append('sync'),
        cleanup=(), final_counters=(lambda: calls.append('allocated'), lambda: calls.append('reserved')))
    with pytest.raises(RuntimeError, match='still reachable'):
        coordinator.finish()
    assert external is not None
    assert calls == ['sync']
    assert admitted.budget.active_leases() == (admitted.lease,)


@pytest.mark.parametrize('counters', [
    (False, 0),
    (0, 1),
    (0.0, 0),
    (0, None),
])
def test_bad_final_counters_leave_admission_charged(monkeypatch, counters):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    _populate(operations)
    coordinator = disposal._Disposal(operations, synchronize=lambda: None, cleanup=(),
                                      final_counters=(lambda: counters[0], lambda: counters[1]))
    with pytest.raises(RuntimeError, match='owner operation failed'):
        coordinator.finish()
    assert 'exactly zero' in str(operations.roots.failure)
    assert admitted.budget.active_leases() == (admitted.lease,)
    assert admitted.lease.trace_bytes('gpu') == admission.GPU_TRACE_CLAIM


def test_cleanup_exception_stops_following_callbacks_and_keeps_claim(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    _populate(operations)
    calls = []

    def fail():
        calls.append('cleanup')
        raise KeyboardInterrupt('cleanup interrupted')

    coordinator = disposal._Disposal(operations, synchronize=lambda: calls.append('sync'),
        cleanup=(fail, lambda: calls.append('unexpected')),
        final_counters=(lambda: calls.append('allocated'), lambda: calls.append('reserved')))
    with pytest.raises(KeyboardInterrupt):
        coordinator.finish()
    assert str(operations.roots.failure) == 'cleanup interrupted'
    assert calls == ['sync', 'sync', 'cleanup']
    assert admitted.budget.active_leases() == (admitted.lease,)
    assert admitted.lease.trace_bytes('gpu') == admission.GPU_TRACE_CLAIM


def test_unexpected_callback_result_is_retained_and_stops(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    _populate(operations)
    refs = []

    def sync():
        value = _Resource()
        refs.append(weakref.ref(value))
        return value

    coordinator = disposal._Disposal(operations, synchronize=sync, cleanup=(),
        final_counters=(lambda: 0, lambda: 0))
    with pytest.raises(RuntimeError) as caught:
        coordinator.finish()
    caught.value.__traceback__ = None
    gc.collect()
    assert refs[0]() is coordinator.pending_return
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_lost_owner_between_allocator_queries_skips_second_query(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, session, admitted = _stack(monkeypatch)
    _populate(operations)
    queries = []

    def allocated():
        queries.append('allocated')
        session.valid = False
        return 0

    def reserved():
        queries.append('reserved')
        return 0

    coordinator = disposal._Disposal(operations, synchronize=lambda: None,
                                     cleanup=(), final_counters=(allocated, reserved))
    with pytest.raises(RuntimeError):
        coordinator.finish()
    assert queries == ['allocated']
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_cyclic_roots_do_not_run_unchecked_global_finalizers(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, session, admitted = _stack(monkeypatch)
    _populate(operations, aliased=False)
    actions = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        def lose_owner():
            actions.append('owner_lost')
            session.valid = False

        def later_finalizer():
            if not session.valid:
                actions.append('device_after_loss')

        operations.roots.model = _CyclicResource(lose_owner)
        operations.roots.converted = operations.roots.model
        operations.roots.input_ids = _CyclicResource(later_finalizer)
        syncs = []
        coordinator = disposal._Disposal(operations, synchronize=lambda: syncs.append('sync'),
            cleanup=(), final_counters=(lambda: 0, lambda: 0))
        with pytest.raises(RuntimeError):
            coordinator.finish()
        assert actions == []
        assert syncs == ['sync']
        assert admitted.budget.active_leases() == (admitted.lease,)
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_automatic_gc_must_be_disabled_before_disposal(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    refs = _populate(operations)
    syncs = []
    coordinator = disposal._Disposal(operations, synchronize=lambda: syncs.append('sync'),
        cleanup=(), final_counters=(lambda: 0, lambda: 0))
    gc.enable()
    try:
        with pytest.raises(RuntimeError, match='garbage collection'):
            coordinator.finish()
    finally:
        gc.disable()
    assert not syncs
    assert all(ref() is not None for ref in refs)
    assert admitted.budget.active_leases() == (admitted.lease,)


def test_automatic_gc_reenabled_by_sync_stops_before_root_drop(monkeypatch):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    refs = _populate(operations)
    calls = []

    def sync():
        calls.append('sync')
        gc.enable()

    coordinator = disposal._Disposal(operations, synchronize=sync, cleanup=(),
                                     final_counters=(lambda: 0, lambda: 0))
    try:
        with pytest.raises(RuntimeError, match='garbage collection'):
            coordinator.finish()
    finally:
        gc.disable()
    assert calls == ['sync']
    assert all(ref() is not None for ref in refs)
    assert admitted.budget.active_leases() == (admitted.lease,)


@pytest.mark.parametrize('kind,method,nested', [
    ('sync', '_none_call', 'run'),
    ('counter', '_capture_counter', 'read'),
])
def test_interrupt_after_callback_return_keeps_assigned_resource(monkeypatch, kind, method, nested):
    disposal = importlib.import_module('looking_glass.tap_repeated_disposal')
    operations, _, admitted = _stack(monkeypatch)
    _populate(operations)
    refs = []

    def resource():
        value = _Resource()
        refs.append(weakref.ref(value))
        return value

    sync = resource if kind == 'sync' else lambda: None
    allocated = resource if kind == 'counter' else lambda: 0
    coordinator = disposal._Disposal(operations, synchronize=sync, cleanup=(),
                                      final_counters=(allocated, lambda: 0))
    lines, first = inspect.getsourcelines(getattr(disposal._Disposal, method))
    target = first + next(i for i, line in enumerate(lines)
                          if line.strip() == 'self.pending_return = result')
    fired = False

    def trace(frame, event, arg):
        nonlocal fired
        if (not fired and event == 'line' and frame.f_code.co_name == nested
                and frame.f_code.co_filename == disposal.__file__
                and frame.f_lineno == target):
            fired = True
            raise KeyboardInterrupt('injected after return')
        return trace

    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            coordinator.finish()
    finally:
        sys.settrace(None)
    assert fired
    caught.value.__traceback__ = None
    caught.value.__context__ = None
    caught.value.__cause__ = None
    gc.collect()
    assert refs[0]() is coordinator.pending_return
    assert refs[0]() is not None
    assert admitted.budget.active_leases() == (admitted.lease,)
    assert admitted.lease.trace_bytes('gpu') == admission.GPU_TRACE_CLAIM
