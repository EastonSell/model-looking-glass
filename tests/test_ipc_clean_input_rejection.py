"""The exact trusted pre-allocation rejection has a narrow IPC scope."""

import time
from unittest.mock import patch

import pytest

from looking_glass.ipc import IPCClient, IPCError, IPCInputRejected, IPCRejected, IPCServer, _default_request_factory
from test_ipc_completion_authority import make_fair_owner, root_server_for_fault_injection, stop_owner
from test_owner_completion import make_owner


def checked_factory(request_id, inputs, seed):
    if inputs["input_ids"] != [[1, 2]]:
        raise IPCInputRejected("private details must not reach the client")
    return _default_request_factory(request_id, inputs, seed)


def test_clean_rejection_does_not_poison_a_running_or_later_direct_job():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    server = IPCServer(broker, team_ids=["team"], request_factory=checked_factory,
                       completion_authority=authority)
    first = None
    try:
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        first = client.submit(team_id="team", request_id="first", inputs={"input_ids": [[1, 2]]})
        assert model.entered.wait(2)
        with pytest.raises(IPCRejected) as caught:
            client.submit(team_id="team", request_id="bad", inputs={"input_ids": [[3, 4]]})
        assert caught.value.code == "invalid_request"
        assert "private" not in str(caught.value)
        assert not authority.unsafe and not server.custody_quarantined
        model.release.set()
        assert first.wait(3)["status"] == "succeeded"
        assert client.submit(team_id="team", request_id="last", inputs={"input_ids": [[1, 2]]}).wait(3)["status"] == "succeeded"
        assert not authority.unsafe and not broker.quarantined
    finally:
        model.release.set()
        if first is not None:
            first.close()
        server.close(timeout=2)
        stop_owner(model, broker)


def test_clean_rejection_retires_fair_guardian_connection_before_later_work():
    model, broker, authority, bridge = make_fair_owner()
    server = IPCServer(bridge, team_ids=["team"], request_factory=checked_factory,
                       custody_guardian=bridge, custody_epoch="b" * 32,
                       completion_authority=authority)
    root_server_for_fault_injection(bridge, server)
    model.release.set()
    try:
        bridge.start()
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        with pytest.raises(IPCRejected) as caught:
            client.submit(team_id="team", request_id="bad", inputs={"input_ids": [[3, 4]]})
        assert caught.value.code == "invalid_request"
        deadline = time.monotonic() + 3
        while bridge.committed_bytes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert bridge.committed_bytes == 0
        assert not bridge.quarantined and not authority.unsafe and not server.custody_quarantined
        assert client.submit(team_id="team", request_id="good", inputs={"input_ids": [[1, 2]]}).wait(5)["status"] == "succeeded"
    finally:
        server.close(timeout=2)
        bridge.close(timeout=2)
        stop_owner(model, broker)


def test_subclass_and_ordinary_factory_errors_remain_uncertain():
    class DerivedRejection(IPCInputRejected):
        pass

    for failure in (DerivedRejection("not the exact signal"), ValueError("unknown allocation")):
        model, _, _, _, _, authority, _, broker, _ = make_owner()

        def broken_factory(*_args):
            raise failure

        server = IPCServer(broker, team_ids=["team"], request_factory=broken_factory,
                           completion_authority=authority)
        try:
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            with pytest.raises(IPCRejected):
                client.submit(team_id="team", request_id="bad", inputs={"input_ids": [[1, 2]]})
            assert authority.unsafe and server.custody_quarantined
        finally:
            server.close(timeout=2)
            stop_owner(model, broker)


def test_the_exact_signal_from_result_encoder_is_not_a_clean_input_rejection():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    model.release.set()

    def broken_encoder(_result):
        raise IPCInputRejected("after execution")

    server = IPCServer(broker, team_ids=["team"], result_encoder=broken_encoder,
                       completion_authority=authority)
    try:
        server.start()
        client = IPCClient(server.address, server.team_tokens["team"])
        handle = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        try:
            result = handle.wait(3)
        except (EOFError, IPCError):
            pass  # An uncertain serializer may close without a terminal frame.
        else:
            assert result["status"] == "failed"
        assert authority.unsafe and server.custody_quarantined
    finally:
        server.close(timeout=2)
        stop_owner(model, broker)


def test_the_exact_signal_after_a_committed_direct_submit_is_uncertain():
    model, _, _, _, _, authority, _, broker, _ = make_owner()
    original = broker.submit
    retained = []

    def committed_then_rejected(*args):
        retained.append(original(*args))
        assert model.entered.wait(2)
        raise IPCInputRejected("submit already committed")

    server = IPCServer(broker, team_ids=["team"], completion_authority=authority)
    try:
        with patch.object(broker, "submit", side_effect=committed_then_rejected):
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            with pytest.raises(IPCRejected):
                client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert len(retained) == 1
        assert authority.unsafe and server.custody_quarantined
    finally:
        model.release.set()
        server.close(timeout=2)
        stop_owner(model, broker)


def test_the_exact_signal_after_fair_authentication_is_uncertain():
    model, broker, authority, bridge = make_fair_owner()
    server = IPCServer(bridge, team_ids=["team"], custody_guardian=bridge,
                       custody_epoch="b" * 32, completion_authority=authority)
    root_server_for_fault_injection(bridge, server)
    original = bridge.authenticate

    def authenticated_then_rejected(*args):
        original(*args)
        raise IPCInputRejected("authentication already committed")

    try:
        with patch.object(bridge, "authenticate", side_effect=authenticated_then_rejected):
            bridge.start()
            server.start()
            client = IPCClient(server.address, server.team_tokens["team"])
            with pytest.raises(IPCRejected):
                client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1, 2]]})
        assert authority.unsafe and server.custody_quarantined
        assert bridge.committed_bytes == 2_000_000
        assert len(bridge._records) == 1
        assert not next(iter(bridge._records.values())).handler_closed
    finally:
        model.release.set()
        server.close(timeout=2)
        bridge.close(timeout=2)
        stop_owner(model, broker)
