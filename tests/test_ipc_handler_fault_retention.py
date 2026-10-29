"""CPU loopback checks for authority-bound handler fault custody."""

import gc
import sys
import threading
import traceback
import weakref

import pytest

from looking_glass.ipc import IPCClient, IPCError, IPCRejected, IPCServer
from test_ipc import FakeBroker
from test_ipc_completion_authority import stop_owner
from test_owner_completion import make_owner


class _Payload:
    pass


def _return_fault_trace(references, *, attach=True, error_type=RuntimeError):
    """Inject at the wrapper return, outside its protected handler call."""
    injected = False

    def trace(frame, event, arg):
        nonlocal injected
        if frame.f_code is IPCServer._run_handler.__code__ and event == "return" and not injected:
            injected = True
            payload = _Payload()
            references.append(weakref.ref(payload))
            error = error_type("handler wrapper return failed")
            if attach:
                error.payload = payload
            raise error
        return trace

    return trace


class _SpoofedTracebackError(RuntimeError):
    @property
    def __traceback__(self):
        return "not the real traceback"


def test_traceback_capture_rejects_a_different_active_exception():
    from looking_glass.ipc import _current_handler_traceback

    try:
        raise RuntimeError("active")
    except RuntimeError:
        with pytest.raises(IPCError, match="context changed"):
            _current_handler_traceback(RuntimeError("different"))


def test_subclass_traceback_property_cannot_spoof_fault_custody():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    references, observed, reported = [], [], threading.Event()
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def hook(args):
        fault = server.retained_handler_faults[0]
        observed.append((fault.capture_complete, fault.exception is args.exc_value,
                         fault.traceback is args.exc_traceback, authority.unsafe,
                         server.custody_quarantined))
        reported.set()

    try:
        threading.settrace(_return_fault_trace(references, attach=False, error_type=_SpoofedTracebackError))
        threading.excepthook = hook
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert reported.wait(3)
        assert observed == [(True, True, True, True, True)]
        assert server.close(timeout=3)
        gc.collect()
        assert references[0]() is not None  # only the actual traceback roots it
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        server.close(timeout=3)
        stop_owner(model, broker)


def test_return_fault_is_rooted_and_latched_before_the_existing_hook():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority, max_connections=1)
    references, observed = [], []
    reported = threading.Event()
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def hook(args):
        faults = server.retained_handler_faults
        observed.append((args.exc_type, authority.unsafe, server.custody_quarantined,
                         len(faults), faults[0].capture_complete,
                         faults[0].exception is args.exc_value,
                         faults[0].traceback is args.exc_traceback))
        reported.set()

    try:
        threading.settrace(_return_fault_trace(references))
        threading.excepthook = hook
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert reported.wait(3)
        assert observed == [(RuntimeError, True, True, 1, True, True, True)]
        with server._lock:
            thread = next(iter(server._handler_fault_slots))
        thread.join(timeout=3)
        assert not thread.is_alive()
        # The dead fault retains its original bounded connection slot.
        with pytest.raises(IPCRejected) as rejected:
            client.submit(team_id="team", request_id="second", inputs={"input_ids": [[1, 2]]})
        assert rejected.value.code == "busy"
        assert server.close(timeout=3)
        gc.collect()
        assert references and references[0]() is not None
        faults = server.retained_handler_faults
        assert len(faults) == 1 and type(faults[0].exception) is RuntimeError
        assert faults[0].exception.payload is references[0]()
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        server.close(timeout=3)
        stop_owner(model, broker)


def test_capture_failure_retains_an_unknown_slot_and_marks_authority(monkeypatch):
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    injected, reported, references, failures = threading.Event(), threading.Event(), [], []
    previous_trace, previous_unraisable = threading.gettrace(), sys.unraisablehook

    def fail_capture():
        injected.set()
        raise MemoryError("exception capture failed")

    def unraisable(args):
        failures.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(ipc, "_current_handler_exception", fail_capture)
        threading.settrace(_return_fault_trace(references))
        sys.unraisablehook = unraisable
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert injected.wait(3)
        assert server.close(timeout=3)
        assert reported.wait(3) and failures == [MemoryError]
        assert authority.unsafe and server.custody_quarantined
        faults = server.retained_handler_faults
        assert len(faults) == 1 and not faults[0].capture_complete
        assert faults[0].exception is None and faults[0].traceback is None
    finally:
        threading.settrace(previous_trace)
        sys.unraisablehook = previous_unraisable
        server.close(timeout=3)
        stop_owner(model, broker)


def test_reentrant_failing_application_hook_preserves_original_fault():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    references, seen, reported = [], [], threading.Event()
    previous_trace, previous_hook, previous_sys_hook = threading.gettrace(), threading.excepthook, sys.excepthook

    def hook(args):
        faults = server.retained_handler_faults
        seen.append((faults[0].exception is args.exc_value, authority.unsafe))
        raise ValueError("application exception hook failed")

    def sys_hook(error_type, error, traceback):
        seen.append(error_type)
        reported.set()

    try:
        threading.settrace(_return_fault_trace(references))
        threading.excepthook = hook
        sys.excepthook = sys_hook
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert reported.wait(3)
        assert server.close(timeout=3)
        assert seen == [(True, True), ValueError]
        assert type(server.retained_handler_faults[0].exception) is RuntimeError
        assert references[0]() is not None
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        sys.excepthook = previous_sys_hook
        server.close(timeout=3)
        stop_owner(model, broker)


@pytest.mark.parametrize("clear_frames", [False, True])
def test_hook_traceback_mutation_has_the_documented_boundary(clear_frames):
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    references, reported = [], threading.Event()
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def hook(args):
        if clear_frames:
            traceback.clear_frames(args.exc_traceback)
        else:
            args.exc_value.__traceback__ = None
        reported.set()

    try:
        threading.settrace(_return_fault_trace(references, attach=False))
        threading.excepthook = hook
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert reported.wait(3)
        assert server.close(timeout=3)
        gc.collect()
        fault = server.retained_handler_faults[0]
        assert fault.capture_complete and fault.traceback is not None
        if clear_frames:
            # This deliberately violates the trusted-hook contract: a rooted
            # traceback cannot prevent its frame locals from being cleared.
            assert references[0]() is None
        else:
            assert fault.exception.__traceback__ is None
            assert references[0]() is not None
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        server.close(timeout=3)
        stop_owner(model, broker)


def test_traceback_capture_failure_keeps_exception_and_incomplete_status(monkeypatch):
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    injected, reported, references, failures = threading.Event(), threading.Event(), [], []
    previous_trace, previous_unraisable = threading.gettrace(), sys.unraisablehook

    def fail_traceback(error):
        injected.set()
        raise MemoryError("traceback capture failed")

    def unraisable(args):
        failures.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(ipc, "_current_handler_traceback", fail_traceback)
        threading.settrace(_return_fault_trace(references))
        sys.unraisablehook = unraisable
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert injected.wait(3)
        assert server.close(timeout=3)
        assert reported.wait(3) and failures == [MemoryError]
        fault = server.retained_handler_faults[0]
        assert not fault.capture_complete and type(fault.exception) is RuntimeError
        assert fault.traceback is None and references[0]() is not None
        assert authority.unsafe and server.custody_quarantined
    finally:
        threading.settrace(previous_trace)
        sys.unraisablehook = previous_unraisable
        server.close(timeout=3)
        stop_owner(model, broker)


@pytest.mark.parametrize("bad_traceback", [None, "not a traceback"])
def test_invalid_traceback_capture_keeps_an_incomplete_slot(monkeypatch, bad_traceback):
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    reported, references, failures = threading.Event(), [], []
    previous_trace, previous_unraisable = threading.gettrace(), sys.unraisablehook

    def unraisable(args):
        failures.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(ipc, "_current_handler_traceback", lambda error: bad_traceback)
        threading.settrace(_return_fault_trace(references))
        sys.unraisablehook = unraisable
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert reported.wait(3) and failures == [IPCError]
        assert server.close(timeout=3)
        fault = server.retained_handler_faults[0]
        assert not fault.capture_complete and type(fault.exception) is RuntimeError
        assert fault.traceback is None and authority.unsafe and server.custody_quarantined
    finally:
        threading.settrace(previous_trace)
        sys.unraisablehook = previous_unraisable
        server.close(timeout=3)
        stop_owner(model, broker)


def test_interrupted_first_slot_assignment_still_retains_unknown(monkeypatch):
    import looking_glass.ipc as ipc

    class InterruptedSlot(ipc._HandlerFaultSlot):
        __slots__ = ("failed_once",)

        def __init__(self):
            super().__init__()
            self.failed_once = False

        def __setattr__(self, name, value):
            if name == "seen" and value is True and not self.failed_once:
                self.failed_once = True
                raise MemoryError("first slot write interrupted")
            super().__setattr__(name, value)

    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    references, reported, failures = [], threading.Event(), []
    previous_trace, previous_unraisable = threading.gettrace(), sys.unraisablehook

    def unraisable(args):
        failures.append(args.exc_type)
        reported.set()

    try:
        monkeypatch.setattr(ipc, "_HandlerFaultSlot", InterruptedSlot)
        threading.settrace(_return_fault_trace(references))
        sys.unraisablehook = unraisable
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert reported.wait(3) and failures == [MemoryError]
        assert server.close(timeout=3)
        assert authority.unsafe and server.custody_quarantined
        faults = server.retained_handler_faults
        assert len(faults) == 1 and not faults[0].capture_complete
        assert faults[0].exception is None and faults[0].traceback is None
    finally:
        threading.settrace(previous_trace)
        sys.unraisablehook = previous_unraisable
        server.close(timeout=3)
        stop_owner(model, broker)


def test_fault_inventory_is_nonblocking_if_server_lock_is_held():
    model, _, _, _, _, _, _, broker, _ = make_owner()
    server = IPCServer(broker, team_ids=["team"], completion_authority=broker._completion_authority)
    try:
        with server._lock:
            with pytest.raises(RuntimeError, match="fault inventory"):
                _ = server.retained_handler_faults
    finally:
        server.close(timeout=1)
        stop_owner(model, broker)


def test_handler_fault_signal_does_not_wait_for_server_lock():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    reached, release, reported = threading.Event(), threading.Event(), threading.Event()
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def trace(frame, event, arg):
        if frame.f_code is IPCServer._run_handler.__code__ and event == "return":
            reached.set()
            if not release.wait(3):
                raise RuntimeError("test did not release handler")
            raise RuntimeError("fault while server lock is held")
        return trace

    def hook(args):
        reported.set()

    try:
        threading.settrace(trace)
        threading.excepthook = hook
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert reached.wait(3)
        with server._lock:
            release.set()
            assert reported.wait(3)
            assert authority.unsafe and server.custody_quarantined
        assert server.close(timeout=3)
        assert len(server.retained_handler_faults) == 1
    finally:
        release.set()
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        server.close(timeout=3)
        stop_owner(model, broker)


def test_clean_handlers_recycle_slots_without_retaining_faults():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority, max_connections=1)
    try:
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        for index in range(4):
            assert client.submit(team_id="team", request_id=f"case-{index}", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
            with server._lock:
                threads = tuple(server._handler_fault_slots)
            for thread in threads:
                thread.join(timeout=3)
                assert not thread.is_alive()
        assert server.close(timeout=3)
        assert not authority.unsafe and not server.custody_quarantined
        assert server.retained_handler_faults == ()
        assert not server._handler_fault_slots
    finally:
        server.close(timeout=3)
        stop_owner(model, broker)


def test_authority_binding_rejects_unsupported_thread_runtime(monkeypatch):
    import looking_glass.ipc as ipc

    model, _, _, _, _, authority, _, broker, _ = make_owner()
    try:
        monkeypatch.setattr(ipc, "_supported_handler_runtime", lambda: False)
        with pytest.raises(ValueError, match="supported CPython"):
            IPCServer(broker, team_ids=["team"], completion_authority=authority)
        assert not authority.unsafe
    finally:
        stop_owner(model, broker)


def test_free_threaded_cpython_is_rejected(monkeypatch):
    import looking_glass.ipc as ipc

    original = ipc.sysconfig.get_config_var
    monkeypatch.setattr(ipc.sysconfig, "get_config_var",
                        lambda name: 1 if name == "Py_GIL_DISABLED" else original(name))
    assert not ipc._supported_handler_runtime()


def test_unbound_server_does_not_claim_fault_observation():
    server = IPCServer(FakeBroker())
    try:
        with pytest.raises(RuntimeError, match="observation is not configured"):
            _ = server.retained_handler_faults
    finally:
        server.close(timeout=1)
