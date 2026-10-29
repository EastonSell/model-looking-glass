import inspect
import subprocess
import sys
import textwrap
from queue import Full

import pytest

from looking_glass.fair_adapter import AdapterClosed, ExperimentalFairAdapter, HANDOFF_PROTOCOL, RegistrationGuard
from looking_glass.fair_admission import AdmissionEmpty, FairAdmissionQueue


class Target:
    handoff_protocol = HANDOFF_PROTOCOL

    def __init__(self):
        self.requests = []

    def accept_next(self, claim):
        request = claim({'A', 'B'})
        if request is None:
            return None
        self.requests.append(request)
        return request.ticket_id


def make_adapter():
    target = Target()
    adapter = ExperimentalFairAdapter(
        {'alice': 'A', 'bob': 'B'}, verifier=lambda credential: credential,
        target=target, max_waiters=3, max_waiters_per_team=2,
        max_waiting_bytes=64, max_item_bytes=32,
    )
    return adapter, target


def interrupt_return(function, call):
    code = function.__code__

    def trace(frame, event, arg):
        if frame.f_code is code and event == 'return':
            sys.settrace(None)
            raise KeyboardInterrupt('return interrupted')
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            call()
    finally:
        sys.settrace(None)


def test_guard_recovers_ticket_after_lost_submission_return():
    adapter, _ = make_adapter()
    guard = adapter.registration_guard('alice', b'payload')
    interrupt_return(type(guard).submit, guard.submit)
    ticket = guard.adopt()
    assert guard.adopt() is ticket
    assert adapter.waiting_bytes == 7
    assert adapter.cancel('alice', ticket)
    assert guard.adopt() is ticket


def test_guard_binds_verified_principal_and_exact_adapter():
    first, _ = make_adapter()
    second, _ = make_adapter()
    guard = first.registration_guard('alice', b'data')
    guard.submit()
    ticket = guard.adopt()
    with pytest.raises(PermissionError):
        first.cancel('bob', ticket)
    assert not second.cancel('alice', ticket)
    assert first.cancel('alice', ticket)


def test_guard_can_require_the_authenticated_team_binding():
    adapter, _ = make_adapter()
    with pytest.raises(PermissionError):
        adapter.registration_guard('alice', b'data', team_id='B')
    guard = adapter.registration_guard('alice', b'data', team_id='A')
    guard.submit()
    assert adapter.cancel('alice', guard.adopt())


def test_unissued_guard_cannot_bypass_verifier():
    adapter, _ = make_adapter()
    forged = RegistrationGuard(adapter, 'bob', b'payload')
    with pytest.raises(ValueError):
        forged.submit()
    assert len(adapter) == 0


def test_guard_submission_is_idempotent_even_after_claim():
    adapter, target = make_adapter()
    guard = adapter.registration_guard('alice', b'payload')
    guard.submit()
    first = guard.adopt()
    guard.submit()
    assert adapter.transfer_next().ticket is first
    guard.submit()
    assert guard.adopt() is first
    assert len(adapter) == 0
    assert len(target.requests) == 1


def test_guard_creation_rejects_oversized_payload_before_retaining_it():
    adapter, _ = make_adapter()
    with pytest.raises(Full):
        adapter.registration_guard('alice', b'x' * 33)
    assert len(adapter) == 0


def test_stale_transition_marker_quarantines_before_another_mutation():
    adapter, _ = make_adapter()
    guard = adapter.registration_guard('alice', b'payload')
    adapter._transition = object()  # Simulates a lost transition's outer return.
    with pytest.raises(AdapterClosed):
        guard.submit()
    assert adapter.indeterminate
    assert len(adapter._queue) == 0


def test_nested_lock_interrupt_sets_nonblocking_quarantine_signal():
    # Isolate the interpreter-lock failure mode: a stranded lock cannot hang
    # the main test process, and the signal itself must not acquire that lock.
    script = textwrap.dedent('''
        import inspect, sys
        from looking_glass.fair_adapter import ExperimentalFairAdapter, HANDOFF_PROTOCOL
        class Target:
            handoff_protocol = HANDOFF_PROTOCOL
            def accept_next(self, claim): return None
        a = ExperimentalFairAdapter({'alice':'A'}, verifier=lambda x:x, target=Target(),
            max_waiters=1, max_waiters_per_team=1, max_waiting_bytes=8, max_item_bytes=8)
        g = a.registration_guard('alice', b'data')
        src, first = inspect.getsourcelines(a._submit_guard)
        line = first + next(i for i, s in enumerate(src) if 'self._inflight_registration = guard' in s) + 1
        def trace(frame, event, arg):
            if frame.f_code is a._submit_guard.__func__.__code__ and event == 'line' and frame.f_lineno == line:
                sys.settrace(None)
                raise KeyboardInterrupt()
            return trace
        sys.settrace(trace)
        try:
            g.submit()
        except KeyboardInterrupt:
            pass
        finally:
            sys.settrace(None)
        assert a.indeterminate
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stdout + result.stderr


def test_interrupted_inner_queue_return_latches_unknown_and_retains_guard():
    adapter, _ = make_adapter()
    guard = adapter.registration_guard('alice', b'payload')
    interrupt_return(FairAdmissionQueue.submit, guard.submit)
    assert adapter.indeterminate
    assert adapter.inflight_registration is guard
    with pytest.raises(AdapterClosed):
        guard.adopt()
    with pytest.raises(AdapterClosed):
        adapter.transfer_next()
    with pytest.raises(AdapterClosed):
        adapter.close()


def test_interrupted_inner_cancel_latches_unknown_and_preserves_ticket_root():
    adapter, _ = make_adapter()
    guard = adapter.registration_guard('alice', b'payload')
    guard.submit()
    ticket = guard.adopt()
    interrupt_return(FairAdmissionQueue.cancel, lambda: adapter.cancel('alice', ticket))
    assert adapter.indeterminate
    assert guard in adapter.registration_roots()
    with pytest.raises(AdapterClosed):
        adapter.close()


def test_interrupted_queue_drain_latches_unknown_and_preserves_ticket_root():
    adapter, _ = make_adapter()
    guard = adapter.registration_guard('alice', b'payload')
    guard.submit()
    interrupt_return(FairAdmissionQueue.drain, adapter.close)
    assert adapter.indeterminate
    assert guard in adapter.registration_roots()
    with pytest.raises(AdapterClosed):
        adapter.close()


def test_lost_close_return_can_recover_exact_ticket_snapshot():
    adapter, _ = make_adapter()
    guard = adapter.registration_guard('alice', b'payload')
    guard.submit()
    ticket = guard.adopt()
    interrupt_return(type(adapter).close, adapter.close)
    assert adapter.closed_tickets == (ticket,)
    assert not adapter.indeterminate


def test_queue_publication_interrupted_before_counter_commit_rolls_back():
    queue = FairAdmissionQueue({'alice': 'A'}, 2, 2)
    source, first = inspect.getsourcelines(FairAdmissionQueue.submit)
    line = first + next(i for i, value in enumerate(source) if 'self._next_ticket = next_ticket' in value)

    def trace(frame, event, arg):
        if frame.f_code is FairAdmissionQueue.submit.__code__ and event == 'line' and frame.f_lineno == line:
            sys.settrace(None)
            raise KeyboardInterrupt('counter interrupted')
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            queue.submit('alice', b'first')
    finally:
        sys.settrace(None)
    assert len(queue) == 0
    with pytest.raises(AdmissionEmpty):
        queue.claim_next()
    ticket = queue.submit('alice', b'second')
    assert queue.claim_next().payload == b'second'
    assert ticket == 1
