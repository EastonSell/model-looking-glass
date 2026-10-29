"""CPU-only checks that IPC shutdown waits for actual handler thread exit."""

import inspect
import threading

import pytest

from looking_glass.ipc import IPCClient, IPCRejected, IPCServer
from test_ipc_completion_authority import stop_owner
from test_owner_completion import make_owner


@pytest.mark.parametrize("method", ["close", "drain"])
def test_shutdown_does_not_report_quiescent_before_a_handler_exits(method):
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority, max_connections=1)
    reached, release = threading.Event(), threading.Event()
    tracked = []
    lines, start = inspect.getsourcelines(IPCServer._serve)
    target_line = start + next(i for i, line in enumerate(lines)
                               if "if cancellation_exception is not None and not pending_exception:" in line)
    previous = threading.gettrace()

    def trace(frame, event, arg):
        if frame.f_code is IPCServer._serve.__code__ and event == "line" and frame.f_lineno == target_line:
            tracked.append(threading.current_thread())
            reached.set()
            if not release.wait(5):
                raise AssertionError("test did not release the completed handler")
        return trace

    try:
        threading.settrace(trace)
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)
        assert result["status"] == "succeeded"
        assert reached.wait(3) and len(tracked) == 1 and tracked[0].is_alive()
        # The connection slot was released, but a second handler must not be
        # started while the first handler still counts toward the bound.
        with pytest.raises(IPCRejected) as caught:
            client.submit(team_id="team", request_id="other", inputs={"input_ids": [[1, 2]]})
        assert caught.value.code == "busy"
        assert not authority.unsafe and not server.custody_quarantined
        if method == "drain":
            assert server.stop_accepting(timeout=3)
        assert getattr(server, method)(timeout=0.05) is False
        assert tracked[0].is_alive()
    finally:
        release.set()
        if tracked:
            tracked[0].join(timeout=3)
        threading.settrace(previous)
        server.close(timeout=3)
        stop_owner(model, broker)
    assert tracked and not tracked[0].is_alive()
    assert server.close(timeout=1) is True


def test_unexpected_failure_in_final_handler_tail_marks_owner_before_exit():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    failed, hook_called = threading.Event(), threading.Event()
    seen = []
    lines, start = inspect.getsourcelines(IPCServer._serve)
    target_line = start + next(i for i, line in enumerate(lines)
                               if "if cancellation_exception is not None and not pending_exception:" in line)
    previous_trace, previous_hook = threading.gettrace(), threading.excepthook

    def trace(frame, event, arg):
        if frame.f_code is IPCServer._serve.__code__ and event == "line" and frame.f_lineno == target_line:
            failed.set()
            raise RuntimeError("injected handler-tail failure")
        return trace

    def hook(args):
        seen.append((args.exc_type, authority.unsafe, server.custody_quarantined))
        hook_called.set()

    try:
        threading.settrace(trace)
        threading.excepthook = hook
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        assert client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert failed.wait(3) and hook_called.wait(3)
        assert seen == [(RuntimeError, True, True)]
        assert server.close(timeout=3)
        assert authority.unsafe and server.custody_quarantined
    finally:
        threading.settrace(previous_trace)
        threading.excepthook = previous_hook
        server.close(timeout=3)
        stop_owner(model, broker)
