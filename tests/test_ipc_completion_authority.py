"""CPU-only checks for IPC faults reaching the actual owner authority."""

import threading
import socket
import struct
import json
import time
import gc
import inspect
import sys
import weakref
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch

import pytest

from looking_glass.broker import BrokerShutdownError, BrokerUnavailable, RequestHandle, RequestState
from looking_glass.fair_adapter import ExperimentalFairAdapter
from looking_glass.ipc import IPCClient, IPCRejected, IPCServer
from looking_glass.ipc_fair_bridge import FairIPCBridge, ForwardCodec, _ServerConstruction
from looking_glass.owner_completion import OwnerCompletionAuthority
from test_owner_completion import make_owner


def make_fair_owner(*, max_connections=2):
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    codec = ForwardCodec(max_batch=1, max_tokens=2, max_encoded_bytes=512, max_decoded_bytes=512)
    target = broker.enable_fair_handoff(
        ("team",), decoder=codec.decode, max_item_bytes=512, max_decoded_bytes=512,
        preparation_bytes=2048, max_preparation_bytes=4096,
        max_result_bytes=128, max_unacknowledged=4)
    adapter = ExperimentalFairAdapter(
        {"alice": "team"}, verifier=lambda token: "alice", target=target,
        max_waiters=4, max_waiters_per_team=3, max_waiting_bytes=2048, max_item_bytes=512)
    bridge = FairIPCBridge(adapter, codec=codec, team_credentials={"team": b"secret"},
                           custody_epoch="b" * 32, max_connections=max_connections,
                           max_request_bytes=2048, max_response_bytes=4096,
                           connection_commitment_bytes=2_000_000)
    return model, broker, authority, bridge


def stop_owner(model, broker):
    model.release.set()
    try:
        broker.close(timeout=2)
    except BrokerShutdownError:
        broker._thread.join(2)


def root_server_for_fault_injection(bridge, server):
    # Mirror the existing guarded factory's final rooted state. The separate
    # bridge integration suite exercises its actual make_server() path.
    bridge._server_construction = _ServerConstruction(server, None)
    bridge._server_created = True
    bridge._server = server


def test_request_factory_failure_marks_owner_before_rejection():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    observed = []

    def failed_factory(*_args):
        raise MemoryError("uncertain input allocation")

    server = IPCServer(broker, team_ids=["team"], request_factory=failed_factory,
                       completion_authority=authority)
    quarantine_order = []

    class RecordingFlag:
        def __init__(self):
            self.event = threading.Event()

        def set(self):
            quarantine_order.append(authority.unsafe)
            self.event.set()

        def is_set(self):
            return self.event.is_set()

    server._custody_quarantine = RecordingFlag()
    original_reject = server._try_reject

    def checked_reject(*args):
        observed.append((authority.unsafe, server.custody_quarantined))
        return original_reject(*args)

    server._try_reject = checked_reject
    try:
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        with pytest.raises(IPCRejected):
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert observed == [(True, True)]
        assert quarantine_order and all(quarantine_order)
        assert authority.unsafe and broker.quarantined
    finally:
        server.close()
        model.release.set()
        try:
            broker.close(timeout=2)
        except BrokerShutdownError:
            broker._thread.join(2)


def test_ipc_fault_precedes_a_blocked_brokers_success_decision():
    model, _, _, _, _, authority, _, broker, request = make_owner()

    def failed_factory(*_args):
        raise MemoryError("uncertain input allocation")

    server = IPCServer(broker, team_ids=["team"], request_factory=failed_factory,
                       completion_authority=authority)
    handle = broker.submit("team", request)
    try:
        assert model.entered.wait(2)
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        with pytest.raises(IPCRejected):
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert authority.unsafe
        model.release.set()
        try:
            outcome = handle.wait(3)
        except BrokerUnavailable:
            outcome = None
        assert outcome is None or outcome.state is not RequestState.SUCCEEDED
    finally:
        server.close(timeout=2)
        stop_owner(model, broker)


def test_completion_authority_must_match_the_actual_broker():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    wrong = OwnerCompletionAuthority()
    try:
        with pytest.raises(ValueError):
            IPCServer(broker)
        with pytest.raises(ValueError):
            IPCServer(broker, completion_authority=wrong)
        assert not authority.unsafe
        assert not wrong.unsafe
        server = IPCServer(broker, completion_authority=authority)
        assert server.close(timeout=1)
        assert not authority.unsafe
    finally:
        model.release.set()
        broker.close(timeout=2)


def test_server_will_not_start_after_the_owner_is_already_unsafe():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    try:
        authority.mark_unsafe()
        with pytest.raises(IPCRejected):
            server.start()
        assert server.address is None
    finally:
        server.close(timeout=1)
        stop_owner(model, broker)


def test_binding_removal_marks_the_pinned_owner_before_server_start():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    try:
        del broker._completion_authority
        with pytest.raises(Exception):
            server.start()
        assert authority.unsafe and broker.quarantined
        # No socket or transport custody was acquired before this denial.
        assert not server.custody_quarantined
    finally:
        broker._completion_authority = authority
        server.close(timeout=1)
        stop_owner(model, broker)


def test_interrupted_server_construction_marks_bound_authority():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    try:
        with patch("looking_glass.ipc.secrets.token_hex", side_effect=MemoryError("token")):
            with pytest.raises(MemoryError):
                IPCServer(broker, completion_authority=authority)
        assert authority.unsafe and broker.quarantined
    finally:
        model.release.set()
        try:
            broker.close(timeout=2)
        except BrokerShutdownError:
            broker._thread.join(2)


def test_interrupted_authority_signal_still_marks_constructor_fault():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    try:
        with patch("looking_glass.ipc.secrets.token_hex", side_effect=MemoryError("token")), \
                patch.object(authority, "mark_unsafe", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                IPCServer(broker, completion_authority=authority)
        assert authority.unsafe and broker.quarantined
    finally:
        stop_owner(model, broker)


def test_interrupted_authority_signal_still_latches_ipc_fault():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    try:
        with patch.object(authority, "mark_unsafe", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                server._mark_custody_fault()
        assert authority.unsafe and server.custody_quarantined
    finally:
        server.close(timeout=1)
        stop_owner(model, broker)


def test_timeout_of_close_is_retryable_without_marking_owner_unsafe():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    entered, release = threading.Event(), threading.Event()

    def hold_lifecycle():
        with server._lifecycle:
            entered.set()
            assert release.wait(3)

    thread = threading.Thread(target=hold_lifecycle)
    thread.start()
    try:
        assert entered.wait(2)
        assert not server.close(timeout=0.01)
        assert not authority.unsafe
    finally:
        release.set()
        thread.join(2)
        server.close(timeout=2)
        model.release.set()
        broker.close(timeout=2)


def test_startup_failure_marks_owner_before_listener_cleanup():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    observations = []

    class BrokenListener:
        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            raise OSError("bind failure")

        def close(self):
            observations.append(authority.unsafe)

    try:
        with patch("looking_glass.ipc.socket.socket", return_value=BrokenListener()):
            with pytest.raises(OSError):
                server.start()
        assert observations == [True]
        assert server.custody_quarantined
    finally:
        server.close()
        model.release.set()
        try:
            broker.close(timeout=2)
        except BrokerShutdownError:
            broker._thread.join(2)


def test_interrupted_startup_cleanup_keeps_partial_listener_rooted():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    references = []

    class Listener:
        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            raise OSError("failed bind")

        def close(self):
            pass

    def make_listener(*_args):
        listener = Listener()
        references.append(weakref.ref(listener))
        return listener

    def exercise():
        try:
            server.start()
        except KeyboardInterrupt:
            return
        raise AssertionError("signal interruption did not occur")

    try:
        with patch("looking_glass.ipc.socket.socket", side_effect=make_listener), \
                patch.object(authority, "mark_unsafe", side_effect=KeyboardInterrupt):
            exercise()
        gc.collect()
        assert authority.unsafe and server.custody_quarantined
        assert len(references) == 1 and references[0]() is not None
        assert not server.close(timeout=0.1)
    finally:
        # The fake listener has no OS resource; production cannot infer this.
        server._startup_uncertain = False
        server._startup_failure = None
        server.close(timeout=1)
        stop_owner(model, broker)


def test_listener_close_failure_marks_owner_before_returning_error():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority).start()
    listener = server._listener

    class BrokenClose:
        def fileno(self):
            return listener.fileno()

        def close(self):
            raise OSError("unknown listener close")

    try:
        server._listener = BrokenClose()
        with pytest.raises(OSError):
            server.close(timeout=1)
        assert authority.unsafe and server.custody_quarantined
    finally:
        server._listener = listener
        server.close(timeout=2)
        model.release.set()
        try:
            broker.close(timeout=2)
        except BrokerShutdownError:
            broker._thread.join(2)


def test_handler_thread_start_failure_marks_owner_and_retains_connection():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority).start()
    original_start = threading.Thread.start
    finished = threading.Event()
    escaped = []

    def failed_start(thread):
        if thread.name == "looking-glass-ipc-client":
            raise RuntimeError("handler start failed")
        return original_start(thread)

    def observed_exception(args):
        escaped.append(args.exc_type)
        finished.set()

    sock = None
    try:
        with patch.object(threading.Thread, "start", failed_start), \
                patch.object(threading, "excepthook", observed_exception):
            sock = socket.create_connection(server.address, timeout=2)
            assert finished.wait(2)
        assert escaped == [RuntimeError]
        assert authority.unsafe and server.custody_quarantined
        assert not server.close(timeout=0.1)
    finally:
        if sock is not None:
            sock.close()
        # This fixture has proved the handler never started. Production must
        # not infer that fact from a failed Thread.start return.
        with server._lock:
            server._handlers.clear()
            server._accept_orphan = None
            server._accept_uncertain = False
        server.close(timeout=2)
        stop_owner(model, broker)


def test_unexpected_listener_poll_failure_stops_bound_service():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    try:
        with patch("looking_glass.ipc.select.select", side_effect=OSError("poll failure")):
            server.start()
            server._accept_thread.join(2)
        assert not server._accept_thread.is_alive()
        assert authority.unsafe and server.custody_quarantined
        assert not server.close(timeout=1)
    finally:
        # A disposable CPU fixture can establish no accepted client existed.
        server._accept_uncertain = False
        server.close(timeout=2)
        stop_owner(model, broker)


def test_interrupted_accept_handoff_keeps_escaped_client_rooted():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    references = []

    class FakeClient:
        pass

    class Listener:
        def accept(self):
            client = FakeClient()
            references.append(weakref.ref(client))
            return client, ("127.0.0.1", 1)

    server._listener = Listener()
    source, first_line = inspect.getsourcelines(IPCServer._accept_loop)
    offsets = [i for i, line in enumerate(source) if "monitored = self._completion_binding" in line]
    assert len(offsets) == 1
    target_line = first_line + offsets[0]
    previous_trace = sys.gettrace()

    def trace(frame, event, _arg):
        if frame.f_code is IPCServer._accept_loop.__code__ and event == "line" and frame.f_lineno == target_line:
            sys.settrace(None)
            raise KeyboardInterrupt("accept handoff interrupted")
        return trace

    def exercise():
        try:
            sys.settrace(trace)
            with patch("looking_glass.ipc.select.select", return_value=([server._listener], [], [])):
                server._accept()
        except KeyboardInterrupt:
            return
        finally:
            sys.settrace(previous_trace)
        raise AssertionError("interruption was not injected")

    try:
        exercise()
        gc.collect()
        assert authority.unsafe and server.custody_quarantined
        assert len(references) == 1 and references[0]() is not None
    finally:
        server._listener = None
        server._accept_uncertain = False
        server._accept_failure = None
        server.close(timeout=1)
        stop_owner(model, broker)


def test_serialization_failure_signals_before_terminal_failure_frame():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()

    def encoder(_result):
        raise MemoryError("uncertain serialization")

    server = IPCServer(broker, team_ids=["team"], result_encoder=encoder,
                       completion_authority=authority)
    sent = []
    original_send = server._send

    def checked_send(client, message):
        if message.get("status") == "failed":
            sent.append((authority.unsafe, server.custody_quarantined))
        return original_send(client, message)

    server._send = checked_send
    try:
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(3)
        assert result["status"] == "failed"
        assert sent == [(True, True)]
    finally:
        server.close(timeout=2)
        try:
            broker.close(timeout=2)
        except BrokerShutdownError:
            broker._thread.join(2)


def test_result_identity_mismatch_signals_before_failure_frame():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, completion_authority=authority)
    sent = []
    server._send = lambda _client, message: sent.append(
        (authority.unsafe, server.custody_quarantined, message["error"]))
    outcome = SimpleNamespace(broker_id="0" * 32, request_id="case", team_id="team", job_id=1)
    try:
        assert not server._send_outcome(None, outcome, SimpleNamespace(job_id=1), "case", "team")
        assert sent == [(True, True, "invalid_result")]
    finally:
        server.close(timeout=1)
        stop_owner(model, broker)


def test_direct_lost_submit_return_marks_before_rejection():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    original_submit = broker.submit
    original_reject = server._try_reject
    observed = []

    def lost_return(team, request):
        original_submit(team, request)
        raise RuntimeError("submit returned to an interrupted caller")

    def checked_reject(*args):
        observed.append(authority.unsafe)
        return original_reject(*args)

    server._try_reject = checked_reject
    try:
        with patch.object(broker, "submit", lost_return):
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            with pytest.raises(IPCRejected):
                client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert observed == [True]
        assert broker.quarantined
    finally:
        server.close(timeout=2)
        stop_owner(model, broker)


def test_direct_missing_handle_after_submission_marks_owner_unsafe():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    original_submit = broker.submit
    observed = []
    original_reject = server._try_reject

    def missing_handle(team, request):
        original_submit(team, request)
        return None

    def checked_reject(*args):
        observed.append(authority.unsafe)
        return original_reject(*args)

    server._try_reject = checked_reject
    try:
        with patch.object(broker, "submit", missing_handle):
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            with pytest.raises(IPCRejected):
                client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert observed == [True]
        assert authority.unsafe and broker.quarantined
    finally:
        server.close(timeout=2)
        stop_owner(model, broker)


def test_wrong_broker_handle_is_never_cancelled_by_this_server():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    other_model, _, _, _, _, other_authority, _, other_broker, other_request = make_owner()
    other_handle = other_broker.submit("team", replace(other_request, request_id="case"))
    assert other_model.entered.wait(2)
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    try:
        with patch.object(broker, "submit", return_value=other_handle):
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            with pytest.raises(IPCRejected):
                client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert authority.unsafe and not other_authority.unsafe
        assert not other_handle._cancel.is_set()
    finally:
        server.close(timeout=2)
        stop_owner(model, broker)
        stop_owner(other_model, other_broker)


def test_direct_queue_full_is_clean_backpressure():
    model, _, _, _, _, authority, _, broker, request = make_owner()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    try:
        broker.submit("team", request)
        assert model.entered.wait(2)
        broker.submit("team", request)
        broker.submit("team", request)
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        with pytest.raises(IPCRejected) as caught:
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert caught.value.code == "queue_full"
        assert not authority.unsafe
        assert not server.custody_quarantined
    finally:
        server.close(timeout=2)
        stop_owner(model, broker)


def test_malformed_cancel_return_signals_before_any_acknowledgment():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    original_cancel = RequestHandle.cancel
    original_send = server._send
    sent = []

    def malformed_cancel(handle):
        original_cancel(handle)
        return None

    def checked_send(client, message):
        sent.append(message.get("status"))
        return original_send(client, message)

    server._send = checked_send
    client_handle = None
    try:
        with patch.object(RequestHandle, "cancel", malformed_cancel):
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            client_handle = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
            assert model.entered.wait(2)
            client_handle.cancel()
            deadline = time.monotonic() + 2
            while not authority.unsafe and time.monotonic() < deadline:
                time.sleep(0.01)
        assert authority.unsafe and server.custody_quarantined
        assert "cancel_requested" not in sent and "cancel_not_accepted" not in sent
    finally:
        if client_handle is not None:
            client_handle.close()
        server.close(timeout=2)
        stop_owner(model, broker)


def test_fair_reservation_fault_keeps_claim_and_signals_before_response():
    model, broker, authority, bridge = make_fair_owner()
    original_reserve = bridge.reserve
    observed = []

    def failed_reserve(key):
        original_reserve(key)
        raise RuntimeError("reservation return is uncertain")

    bridge.reserve = failed_reserve
    server = IPCServer(bridge, team_ids=["team"], custody_guardian=bridge,
                       custody_epoch="b" * 32, completion_authority=authority)
    root_server_for_fault_injection(bridge, server)
    original_reject = server._try_reject

    def checked_reject(*args):
        observed.append((authority.unsafe, server.custody_quarantined))
        return original_reject(*args)

    server._try_reject = checked_reject
    try:
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        with pytest.raises(IPCRejected):
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert observed == [(True, True)]
        assert bridge.committed_bytes == 2_000_000
        assert len(bridge._records) == 1
    finally:
        server.close(timeout=2)
        bridge.close(timeout=0.1)
        stop_owner(model, broker)


def test_fair_missing_proxy_keeps_its_registered_claim_quarantined():
    model, broker, authority, bridge = make_fair_owner()
    server = IPCServer(bridge, team_ids=["team"], custody_guardian=bridge,
                       custody_epoch="b" * 32, completion_authority=authority)
    root_server_for_fault_injection(bridge, server)
    original_submit = bridge.submit_registered

    def missing_proxy(key, team, request):
        original_submit(key, team, request)
        return None

    try:
        with patch.object(bridge, "submit_registered", missing_proxy):
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            with pytest.raises(IPCRejected):
                client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert authority.unsafe and server.custody_quarantined
        assert bridge.committed_bytes == 2_000_000
        record = next(iter(bridge._records.values()))
        assert record.proxy is not None and not record.handler_closed
    finally:
        server.close(timeout=2)
        bridge.close(timeout=0.1)
        stop_owner(model, broker)


def test_fair_known_full_capacity_stays_retryable_and_owner_safe():
    model, broker, authority, bridge = make_fair_owner(max_connections=1)
    server = IPCServer(bridge, team_ids=["team"], custody_guardian=bridge,
                       custody_epoch="b" * 32, completion_authority=authority,
                       max_connections=2)
    root_server_for_fault_injection(bridge, server)
    first = None
    try:
        bridge.start()
        server.start()
        first = socket.create_connection(server.address, timeout=2)
        first.settimeout(2)
        header = b""
        while len(header) < 4:
            header += first.recv(4 - len(header))
        size = struct.unpack("!I", header)[0]
        data = b""
        while len(data) < size:
            data += first.recv(size - len(data))
        assert json.loads(data)["status"] == "ready"
        client = IPCClient(server.address, server.team_tokens["team"])
        with pytest.raises(IPCRejected) as caught:
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert caught.value.code == "busy"
        assert not authority.unsafe and not server.custody_quarantined
        assert bridge.committed_bytes == 2_000_000
        first.close()
        first = None
        deadline = time.monotonic() + 3
        while bridge.committed_bytes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert bridge.committed_bytes == 0
    finally:
        if first is not None:
            first.close()
        server.close(timeout=2)
        bridge.close(timeout=2)
        stop_owner(model, broker)


def test_real_fair_proxy_is_accepted_and_result_is_delivered():
    model, broker, authority, bridge = make_fair_owner()
    server = IPCServer(bridge, team_ids=["team"], custody_guardian=bridge,
                       custody_epoch="b" * 32, completion_authority=authority)
    root_server_for_fault_injection(bridge, server)
    model.release.set()
    try:
        bridge.start()
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]}).wait(5)
        assert result["status"] == "succeeded"
        assert result["request_id"] == "case" and result["team_id"] == "team"
        assert not authority.unsafe and not server.custody_quarantined
        deadline = time.monotonic() + 3
        while bridge.committed_bytes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert bridge.committed_bytes == 0
    finally:
        server.close(timeout=2)
        bridge.close(timeout=2)
        stop_owner(model, broker)


def test_fair_binding_rechecks_pinned_target_before_server_start():
    model, broker, authority, bridge = make_fair_owner()
    server = IPCServer(bridge, team_ids=["team"], custody_guardian=bridge,
                       custody_epoch="b" * 32, completion_authority=authority)
    root_server_for_fault_injection(bridge, server)
    original = bridge._target_broker
    try:
        bridge._target_broker = object()
        with pytest.raises(IPCRejected):
            server.start()
        assert authority.unsafe and server.custody_quarantined
    finally:
        bridge._target_broker = original
        server.close(timeout=2)
        bridge.close(timeout=0.1)
        stop_owner(model, broker)


def test_unrooted_fair_server_cannot_bypass_bridge_factory():
    model, broker, authority, bridge = make_fair_owner()
    server = IPCServer(bridge, team_ids=["team"], custody_guardian=bridge,
                       custody_epoch="b" * 32, completion_authority=authority)
    try:
        with pytest.raises(IPCRejected):
            server.start()
        assert server.address is None
        assert authority.unsafe and server.custody_quarantined
    finally:
        server.close(timeout=1)
        bridge.close(timeout=0.1)
        stop_owner(model, broker)
