from copy import deepcopy
import inspect
import sys

import pytest

from looking_glass import tap_repeated_admission as admission
from looking_glass.memory_budget import MemoryBudget, MemoryLease
from looking_glass.tap_repeated_protocol import GIB, HEADROOM, OUTSIDE_CAP


TOTAL, CAP = 8 * GIB, 2 * GIB
ROW = {'free_bytes': 6 * GIB, 'total_bytes': TOTAL,
       'allocated_bytes': 128 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}


class Owner:
    def __init__(self):
        self.valid = True
        self.checks = 0

    def check(self):
        self.checks += 1
        if not self.valid:
            raise RuntimeError('supervised owner lost')


class Telemetry:
    def __init__(self):
        self.row = deepcopy(ROW)
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        return dict(self.row)


def make_admission():
    owner, telemetry = Owner(), Telemetry()
    value = admission._Admission(check_owner=owner.check, snapshot=telemetry.snapshot,
                                 total_bytes=TOTAL, cap_bytes=CAP)
    return value, owner, telemetry


def test_whole_run_and_all_trace_are_reserved_before_work():
    value, owner, telemetry = make_admission()
    first = value.acquire()
    assert first == ROW and value.state == 'active'
    assert value.budget.active_leases() == (value.lease,)
    assert value.lease.estimated_incremental_bytes == CAP + HEADROOM + OUTSIDE_CAP
    assert value.lease.trace_bytes('gpu') == 128 * 1024
    assert value.lease.trace_bytes('host') == 0
    assert value.check_snapshot() == ROW and owner.checks >= 6 and telemetry.calls >= 3
    with pytest.raises(RuntimeError, match='single-use'):
        value.acquire()


def test_pressure_and_lost_owner_stop_later_sampling():
    value, owner, telemetry = make_admission()
    value.acquire()
    telemetry.row['free_bytes'] = GIB
    with pytest.raises(ValueError):
        value.check_snapshot()
    before = telemetry.calls
    assert value.state == 'failed' and value.budget.active_leases() == (value.lease,)
    assert value.lease.trace_bytes('gpu') == admission.GPU_TRACE_CLAIM
    with pytest.raises(RuntimeError, match='not active'):
        value.check_snapshot()
    assert telemetry.calls == before

    other, other_owner, other_telemetry = make_admission()
    other.acquire()
    original = other_telemetry.snapshot

    def revoke_during_snapshot():
        result = original()
        other_owner.valid = False
        return result

    other.snapshot = revoke_during_snapshot
    with pytest.raises(RuntimeError, match='owner lost'):
        other.check_snapshot()
    before = other_telemetry.calls
    assert other.state == 'failed' and other.budget.active_leases() == (other.lease,)
    with pytest.raises(RuntimeError, match='not active'):
        other.check_snapshot()
    assert other_telemetry.calls == before


def test_interrupted_verify_failure_handler_leaves_nonactive_marker():
    value, owner, telemetry = make_admission()
    value.acquire()
    owner.valid = False
    lines, first_line = inspect.getsourcelines(admission._Admission.verify)
    failure_line = first_line + next(i for i, line in enumerate(lines) if "self.state = 'failed'" in line)
    interrupted = []

    def trace(frame, event, arg):
        if frame.f_code is admission._Admission.verify.__code__ and event == 'line' and frame.f_lineno == failure_line:
            interrupted.append(True)
            raise KeyboardInterrupt('failure handler interrupted')
        return trace

    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt, match='failure handler interrupted'):
            value.verify()
    finally:
        sys.settrace(None)
    before = telemetry.calls
    assert interrupted == [True] and value.state == 'verifying'
    assert value.budget.active_leases() == (value.lease,)
    owner.valid = True
    with pytest.raises(RuntimeError, match='not active'):
        value.check_snapshot()
    assert telemetry.calls == before


@pytest.mark.parametrize('exception_type', [MemoryError, KeyboardInterrupt])
def test_lost_acquire_return_keeps_issued_lease_rooted(exception_type, monkeypatch):
    value, _, _ = make_admission()
    original = MemoryBudget.acquire

    def acquire_then_fail(budget, *args, **kwargs):
        original(budget, *args, **kwargs)
        raise exception_type('lost admission return')

    monkeypatch.setattr(MemoryBudget, 'acquire', acquire_then_fail)
    with pytest.raises(exception_type):
        value.acquire()
    assert value.state == 'failed' and isinstance(value.failure, exception_type)
    assert value.lease is None and len(value.budget.active_leases()) == 1
    assert value.budget.active_leases()[0].estimated_incremental_bytes == CAP + HEADROOM + OUTSIDE_CAP
    with pytest.raises(RuntimeError, match='single-use'):
        value.acquire()


@pytest.mark.parametrize('exception_type', [MemoryError, KeyboardInterrupt])
def test_lost_trace_reservation_return_preserves_full_claim(exception_type, monkeypatch):
    value, _, _ = make_admission()
    original = MemoryLease.reserve_trace

    def reserve_then_fail(lease, *args, **kwargs):
        original(lease, *args, **kwargs)
        raise exception_type('lost trace return')

    monkeypatch.setattr(MemoryLease, 'reserve_trace', reserve_then_fail)
    with pytest.raises(exception_type):
        value.acquire()
    assert value.state == 'failed' and value.budget.active_leases() == (value.lease,)
    assert value.lease.trace_bytes('gpu') == admission.GPU_TRACE_CLAIM
    with pytest.raises(RuntimeError, match='not active'):
        value.verify()


def test_release_is_single_use_after_trusted_caller_disposal():
    value, _, _ = make_admission()
    value.acquire()
    # CPU fixture only: the real owner must synchronize and dispose its rooted
    # model, input, events, outputs and traces before calling this method.
    value.release_after_disposal()
    assert value.state == 'released' and value.budget.active_leases() == ()
    assert value.lease.trace_bytes('gpu') == 0
    with pytest.raises(RuntimeError, match='not active'):
        value.release_after_disposal()


@pytest.mark.parametrize('exception_type', [MemoryError, KeyboardInterrupt])
def test_ambiguous_trace_release_stops_without_releasing_the_whole_lease(exception_type, monkeypatch):
    value, _, _ = make_admission()
    value.acquire()
    original = MemoryLease.release_trace

    def release_then_fail(lease, *args, **kwargs):
        original(lease, *args, **kwargs)
        raise exception_type('lost release return')

    monkeypatch.setattr(MemoryLease, 'release_trace', release_then_fail)
    with pytest.raises(exception_type):
        value.release_after_disposal()
    assert value.state == 'failed' and value.budget.active_leases() == (value.lease,)
    with pytest.raises(RuntimeError, match='not active'):
        value.release_after_disposal()


@pytest.mark.parametrize('exception_type', [MemoryError, KeyboardInterrupt])
def test_lost_whole_lease_release_return_is_not_retried(exception_type, monkeypatch):
    value, owner, _ = make_admission()
    value.acquire()
    original = MemoryLease.release

    def release_then_fail(lease):
        original(lease)
        raise exception_type('lost whole-lease release return')

    monkeypatch.setattr(MemoryLease, 'release', release_then_fail)
    with pytest.raises(exception_type):
        value.release_after_disposal()
    # The ledger transition really committed, so it would be incorrect to
    # claim the whole-run lease remains active or to retry its release.
    assert value.state == 'failed' and value.budget.active_leases() == ()
    assert value.lease.trace_bytes('gpu') == 0 and owner.valid
    with pytest.raises(RuntimeError, match='not active'):
        value.release_after_disposal()
