"""CPU loopback tests for authority-bound acceptor fault custody."""
import gc
import inspect
import socket
import sys
import threading
import weakref

import pytest
import torch

from looking_glass.ipc import IPCError, IPCServer
from looking_glass.broker import BrokerUnavailable, RequestState
from test_ipc import FakeBroker
from test_ipc_completion_authority import make_fair_owner, root_server_for_fault_injection, stop_owner
from test_owner_completion import make_owner


class Payload:
    pass


class SpoofedTracebackError(RuntimeError):
    @property
    def __traceback__(self):
        return 'not a traceback'


def accept_return_trace(references):
    code = IPCServer._accept.__code__
    injected = False

    def trace(frame, event, arg):
        nonlocal injected
        if frame.f_code is code and event == 'return' and not injected:
            injected = True
            payload = Payload()
            references.append(weakref.ref(payload))
            raise SpoofedTracebackError('acceptor return failed')
        return trace
    return trace


def test_acceptor_return_fault_is_rooted_and_latched_before_existing_hook():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    references, seen, reported = [], [], threading.Event()
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def hook(args):
        try:
            fault = server.retained_acceptor_fault
            seen.append((fault.capture_complete, fault.exception is args.exc_value,
                         fault.traceback is args.exc_traceback, authority.unsafe,
                         server.custody_quarantined))
        except BaseException as error:
            seen.append(type(error))
        finally:
            reported.set()

    try:
        threading.settrace(accept_return_trace(references))
        threading.excepthook = hook
        server.start()
        assert server.close(timeout=3) is False
        assert reported.wait(3)
        assert seen == [(True, True, True, True, True)]
        assert not server._accept_thread.is_alive()
        gc.collect()
        assert references[0]() is not None
        fault = server.retained_acceptor_fault
        assert fault.capture_complete and type(fault.exception) is SpoofedTracebackError
        assert fault.traceback is not None
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        server.close(timeout=3)
        stop_owner(model, broker)


def test_acceptor_fault_capture_failure_marks_unknown_and_retains_slot(monkeypatch):
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    references, failures, reported = [], [], threading.Event()
    previous_trace, previous_unraisable = threading.gettrace(), sys.unraisablehook

    def fail_capture():
        raise MemoryError('injected acceptor fault capture failure')

    def unraisable(args):
        failures.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(ipc, '_current_handler_exception', fail_capture)
        threading.settrace(accept_return_trace(references))
        sys.unraisablehook = unraisable
        server.start()
        assert server.close(timeout=3) is False
        assert reported.wait(3) and failures == [MemoryError]
        assert authority.unsafe and server.custody_quarantined
        fault = server.retained_acceptor_fault
        assert fault is not None and not fault.capture_complete
        assert fault.exception is None and fault.traceback is None
    finally:
        threading.settrace(previous_trace)
        sys.unraisablehook = previous_unraisable
        server.close(timeout=3)
        stop_owner(model, broker)


def test_interrupted_owner_signal_retains_unknown_fault_and_marks_unsafe(monkeypatch):
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    references, failures, reported = [], [], threading.Event()
    previous_trace, previous_unraisable = threading.gettrace(), sys.unraisablehook

    def interrupted_signal():
        raise KeyboardInterrupt('injected interruption while signaling')

    def unraisable(args):
        failures.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(authority, 'mark_unsafe', interrupted_signal)
        threading.settrace(accept_return_trace(references))
        sys.unraisablehook = unraisable
        server.start()
        assert server.close(timeout=3) is False
        assert reported.wait(3) and failures == [KeyboardInterrupt]
        assert authority.unsafe and server.custody_quarantined
        fault = server.retained_acceptor_fault
        assert fault is not None and not fault.capture_complete
        assert fault.exception is None and fault.traceback is None
    finally:
        threading.settrace(previous_trace)
        sys.unraisablehook = previous_unraisable
        server.close(timeout=3)
        stop_owner(model, broker)


def test_accept_loop_failure_is_retained_and_shutdown_stays_incomplete(monkeypatch):
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    references, seen, reported = [], [], threading.Event()
    previous_hook = threading.excepthook

    def fail_accept():
        payload = Payload()
        references.append(weakref.ref(payload))
        raise RuntimeError('injected accept-loop failure')

    def hook(args):
        fault = server.retained_acceptor_fault
        seen.append((fault.capture_complete, fault.exception is args.exc_value,
                     fault.traceback is args.exc_traceback, authority.unsafe))
        reported.set()

    try:
        monkeypatch.setattr(server, '_accept_loop', fail_accept)
        threading.excepthook = hook
        server.start()
        assert reported.wait(3)
        assert seen == [(True, True, True, True)]
        assert server._accept_failure is server.retained_acceptor_fault.exception
        assert server.close(timeout=3) is False
        gc.collect()
        assert references[0]() is not None and server.custody_quarantined
    finally:
        threading.excepthook = previous_hook
        server.close(timeout=3)
        stop_owner(model, broker)


def test_reentrant_failing_hook_uses_no_server_lock_and_keeps_original_fault():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    references, seen, reported = [], [], threading.Event()
    previous_trace, previous_hook, previous_sys_hook = threading.gettrace(), threading.excepthook, sys.excepthook

    def hook(args):
        fault = server.retained_acceptor_fault
        seen.append((fault.exception is args.exc_value, authority.unsafe))
        raise ValueError('injected application hook failure')

    def sys_hook(error_type, error, traceback):
        seen.append(error_type)
        reported.set()

    try:
        threading.settrace(accept_return_trace(references))
        threading.excepthook = hook
        sys.excepthook = sys_hook
        server.start()
        with server._lock:
            server._accepting_stopped.set()
            assert reported.wait(3)
        assert seen == [(True, True), ValueError]
        assert server.close(timeout=3) is False
        assert type(server.retained_acceptor_fault.exception) is SpoofedTracebackError
        gc.collect()
        assert references[0]() is not None
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        sys.excepthook = previous_sys_hook
        server.close(timeout=3)
        stop_owner(model, broker)


def test_fair_bound_acceptor_uses_the_same_owner_authority():
    model, broker, authority, bridge = make_fair_owner()
    model.release.set()
    server = IPCServer(bridge, team_ids=['team'], custody_guardian=bridge,
                       custody_epoch='b' * 32, completion_authority=authority)
    root_server_for_fault_injection(bridge, server)
    references, seen, reported = [], [], threading.Event()
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def hook(args):
        fault = server.retained_acceptor_fault
        seen.append((fault.exception is args.exc_value, authority.unsafe, server.custody_quarantined))
        reported.set()

    try:
        threading.settrace(accept_return_trace(references))
        threading.excepthook = hook
        server.start()
        assert server.close(timeout=3) is False
        assert reported.wait(3) and seen == [(True, True, True)]
        assert server.retained_acceptor_fault.capture_complete
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        server.close(timeout=3)
        bridge.close(timeout=1)
        stop_owner(model, broker)


def test_normal_acceptor_exit_leaves_no_fault_and_legacy_mode_unchanged():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    try:
        server.start()
        assert server.close(timeout=3) is True
        assert server.retained_acceptor_fault is None
        assert not authority.unsafe and not server.custody_quarantined
    finally:
        server.close(timeout=3)
        stop_owner(model, broker)
    legacy = IPCServer(FakeBroker())
    try:
        legacy.start()
        assert legacy.close(timeout=3) is True
        with pytest.raises(IPCError):
            _ = legacy.retained_acceptor_fault
    finally:
        legacy.close(timeout=3)


@pytest.mark.parametrize('escaping_target', ['acceptor', 'handler'])
def test_escaped_thread_fault_marks_owner_before_blocked_capture(monkeypatch, escaping_target):
    """An in-flight job cannot succeed while fault custody capture is stalled."""
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, request = make_owner()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    handle = broker.submit('team', request)
    entered, resume, fired, reported = (threading.Event() for _ in range(4))
    original_capture = ipc._current_handler_exception
    target_code = (IPCServer._accept if escaping_target == 'acceptor'
                   else IPCServer._run_handler).__code__
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def blocked_capture():
        entered.set()
        if not resume.wait(8):
            raise RuntimeError('fault-capture gate timed out')
        return original_capture()

    def trace(frame, event, arg):
        if frame.f_code is target_code and event == 'return' and not fired.is_set():
            fired.set()
            raise RuntimeError('injected thread return failure')
        return trace

    def hook(args):
        reported.set()

    try:
        assert model.entered.wait(3)
        monkeypatch.setattr(ipc, '_current_handler_exception', blocked_capture)
        threading.excepthook = hook
        threading.settrace(trace)
        server.start()
        if escaping_target == 'acceptor':
            server._accepting_stopped.set()
        else:
            with socket.create_connection(server.address, timeout=3):
                pass
        assert entered.wait(3) and fired.is_set()
        threading.settrace(previous_trace)
        assert authority.unsafe
        model.release.set()
        try:
            outcome = handle.wait(3)
        except BrokerUnavailable:
            pass
        else:
            assert outcome.state is not RequestState.SUCCEEDED
        resume.set()
        assert reported.wait(3)
        assert server.custody_quarantined
    finally:
        threading.settrace(previous_trace)
        resume.set()
        model.release.set()
        server.close(timeout=3)
        if fired.is_set():
            reported.wait(3)
        threading.excepthook = previous_hook
        stop_owner(model, broker)


def test_accept_loop_failure_marks_owner_before_blocked_fault_processing(monkeypatch):
    """The protected accept path must suppress a later owner success too."""
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, request = make_owner()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    handle = broker.submit('team', request)
    entered, resume, fired, reported = (threading.Event() for _ in range(4))
    original_select = ipc.select.select
    original_mark = server._mark_custody_fault
    previous_hook = threading.excepthook

    def fail_select(*args, **kwargs):
        if threading.current_thread() is server._accept_thread:
            raise RuntimeError('injected accept-loop failure')
        return original_select(*args, **kwargs)

    def blocked_mark():
        if threading.current_thread() is server._accept_thread and not fired.is_set():
            fired.set()
            entered.set()
            if not resume.wait(8):
                raise RuntimeError('fault-processing gate timed out')
        return original_mark()

    try:
        assert model.entered.wait(3)
        monkeypatch.setattr(ipc.select, 'select', fail_select)
        monkeypatch.setattr(server, '_mark_custody_fault', blocked_mark)
        threading.excepthook = lambda args: reported.set()
        server.start()
        assert entered.wait(3)
        assert authority.unsafe
        model.release.set()
        try:
            outcome = handle.wait(3)
        except BrokerUnavailable:
            pass
        else:
            assert outcome.state is not RequestState.SUCCEEDED
        resume.set()
        assert reported.wait(3)
        assert server.custody_quarantined and server._accept_failure is not None
    finally:
        resume.set()
        model.release.set()
        server.close(timeout=3)
        if fired.is_set():
            reported.wait(3)
        threading.excepthook = previous_hook
        stop_owner(model, broker)


def test_interrupted_first_signal_retains_chained_target_custody(monkeypatch):
    """A one-shot signal interruption must not lose the original claim graph."""
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    references, aliases, fired, errors = [], [], [], []
    reported = threading.Event()
    previous_trace, previous_unraisable = threading.gettrace(), sys.unraisablehook
    lines, first_line = inspect.getsourcelines(IPCServer._install_thread_fault_observer)
    signal_line = first_line + next(i for i, line in enumerate(lines)
                                    if line.strip() == 'authority._unsafe = True')

    def fail_target():
        claim = Payload()
        tensor = torch.ones(1)
        tensor.claim = claim
        references.append(weakref.ref(claim))
        aliases.append(tensor.detach())
        raise RuntimeError('injected acceptor target failure')

    def trace(frame, event, arg):
        if (frame.f_code.co_name == 'observe' and event == 'line'
                and frame.f_lineno == signal_line and not fired):
            fired.append(True)
            raise KeyboardInterrupt('injected first signal interruption')
        return trace

    def unraisable(args):
        errors.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(server, '_accept', fail_target)
        threading.settrace(trace)
        sys.unraisablehook = unraisable
        server.start()
        assert reported.wait(3) and fired == [True] and errors == [KeyboardInterrupt]
        server._accept_thread.join(3)
        assert not server._accept_thread.is_alive() and authority.unsafe
        fault = server.retained_acceptor_fault
        assert fault is not None and not fault.capture_complete
        assert type(fault.observer_failure) is KeyboardInterrupt
        assert type(fault.observer_failure.__context__) is RuntimeError
        gc.collect()
        assert references[0]() is not None
        assert aliases[0].tolist() == [1.0]
    finally:
        threading.settrace(previous_trace)
        sys.unraisablehook = previous_unraisable
        server.close(timeout=3)
        stop_owner(model, broker)


def test_interrupted_accept_loop_signal_retains_original_failure(monkeypatch):
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    references, fired, seen, reported = [], [], [], threading.Event()
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook
    original_select = ipc.select.select
    lines, first_line = inspect.getsourcelines(IPCServer._accept)
    signal_line = first_line + next(i for i, line in enumerate(lines)
                                    if line.strip() == 'binding[0]._unsafe = True')

    def fail_select(*args, **kwargs):
        if threading.current_thread() is server._accept_thread:
            payload = Payload()
            references.append(weakref.ref(payload))
            raise RuntimeError('injected accept-loop failure')
        return original_select(*args, **kwargs)

    def trace(frame, event, arg):
        if (frame.f_code is IPCServer._accept.__code__ and event == 'line'
                and frame.f_lineno == signal_line and not fired):
            fired.append(True)
            raise KeyboardInterrupt('injected first accept signal interruption')
        return trace

    def hook(args):
        seen.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(ipc.select, 'select', fail_select)
        threading.settrace(trace)
        threading.excepthook = hook
        server.start()
        assert reported.wait(3) and fired == [True] and seen == [KeyboardInterrupt]
        server._accept_thread.join(3)
        assert authority.unsafe and server.custody_quarantined
        assert type(server._accept_failure) is RuntimeError
        fault = server.retained_acceptor_fault
        assert fault.capture_complete and type(fault.exception) is KeyboardInterrupt
        gc.collect()
        assert references[0]() is not None
    finally:
        threading.settrace(previous_trace)
        server.close(timeout=3)
        threading.excepthook = previous_hook
        stop_owner(model, broker)


def test_interrupted_binding_lookup_signals_before_stalled_exception_retention(monkeypatch):
    """A single interrupted first signal cannot let a pending job succeed."""
    model, _, _, _, _, authority, _, broker, request = make_owner()
    server = IPCServer(broker, team_ids=['team'], completion_authority=authority)
    handle = broker.submit('team', request)
    entered, resume, reported, fired = threading.Event(), threading.Event(), threading.Event(), []
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook
    lines, first_line = inspect.getsourcelines(IPCServer._accept)
    binding_line = first_line + next(i for i, line in enumerate(lines)
                                     if line.strip() == 'binding = self._completion_binding')
    original_setattr = IPCServer.__setattr__

    def blocked_assignment(obj, name, value):
        if obj is server and name == '_accept_failure' and isinstance(value, BaseException):
            entered.set()
            if not resume.wait(8):
                raise RuntimeError('fault-retention gate timed out')
        return original_setattr(obj, name, value)

    def fail_loop():
        raise RuntimeError('injected accept-loop failure')

    def trace(frame, event, arg):
        if (frame.f_code is IPCServer._accept.__code__ and event == 'line'
                and frame.f_lineno == binding_line and not fired):
            fired.append(True)
            raise KeyboardInterrupt('injected binding lookup interruption')
        return trace

    try:
        assert model.entered.wait(3)
        monkeypatch.setattr(IPCServer, '__setattr__', blocked_assignment)
        monkeypatch.setattr(server, '_accept_loop', fail_loop)
        threading.settrace(trace)
        threading.excepthook = lambda args: reported.set()
        server.start()
        assert entered.wait(3) and fired == [True]
        threading.settrace(previous_trace)
        assert authority.unsafe
        model.release.set()
        try:
            outcome = handle.wait(3)
        except BrokerUnavailable:
            pass
        else:
            assert outcome.state is not RequestState.SUCCEEDED
        resume.set()
        assert reported.wait(3)
        assert type(server._accept_failure) is RuntimeError
    finally:
        threading.settrace(previous_trace)
        resume.set()
        model.release.set()
        server.close(timeout=3)
        if fired:
            reported.wait(3)
        threading.excepthook = previous_hook
        stop_owner(model, broker)
