"""Real experiment lifetimes when a client disconnects and submits again."""
import json
import socket
import struct
import sys
import threading

import torch

from looking_glass.app import create_server
from test_app_http import call
from test_app_sensitivity import state


def test_all_experiment_types_are_reusable_without_retiring_retained_data(state):
    from test_app_cross_patch import pair
    reference = state.post("pin-reference", {"capture_id": state._capture_id, "event": 0, "vector": 1,
                                             "expected_reference_id": None})
    manual = state.post("intervene", {"capture_id": state._capture_id, "event": 0, "vector": 1,
                                    "operation": "zero"})
    capture, comparison = state.store.summary(), state.experiment_store.summary()
    capture_bits = [value.view(torch.int32).clone() for value in state.store._values]
    comparison_bits = [value.view(torch.int32).clone() for value in state.experiment_store._values]
    reference_bits = state.reference._value.view(torch.int32).clone()
    weights = {name: value.view(torch.int32).clone() for name, value in state.runtime.model.state_dict().items()}
    binding = state.runtime.capture_binding()
    one = {"capture_id": state._capture_id, "event": 0, "vector": 1, "layers": [0], "sites": ["attention"]}
    repeated = {"capture_id": state._capture_id, "layers": [0], "sites": ["attention"],
                "prompts": ["hello world", "world hello"]}
    for _ in range(2):
        assert state.post("sensitivity", one)["status"] == "complete"
        assert state.post("sensitivity-repeat", repeated)["status"] == "complete"
        assert state.post("patch-cross-prompt", pair(state))["status"] == "complete"
        assert state.store.summary() == capture and state.experiment_store.summary() == comparison
        assert state._experiment_id == manual["experiment_id"]
        assert state._reference_id == reference["reference_id"]
        assert state.runtime.capture_binding() is binding is state._live_binding
        assert all(torch.equal(value.view(torch.int32), expected)
                   for value, expected in zip(state.store._values, capture_bits))
        assert all(torch.equal(value.view(torch.int32), expected)
                   for value, expected in zip(state.experiment_store._values, comparison_bits))
        assert torch.equal(state.reference._value.view(torch.int32), reference_bits)
        assert all(torch.equal(value.view(torch.int32), weights[name])
                   for name, value in state.runtime.model.state_dict().items())
        assert not any(module._forward_hooks or module._forward_pre_hooks for module in state.runtime.model.modules())


def test_disconnected_sweep_finishes_safely_and_a_later_request_succeeds(state, monkeypatch):
    reference = state.post("pin-reference", {"capture_id": state._capture_id, "event": 0, "vector": 1,
                                             "expected_reference_id": None})
    manual = state.post("intervene", {"capture_id": state._capture_id, "event": 0, "vector": 1,
                                    "operation": "zero"})
    capture_summary, comparison_summary = state.store.summary(), state.experiment_store.summary()
    capture_bits = [value.view(torch.int32).clone() for value in state.store._values]
    comparison_bits = [value.view(torch.int32).clone() for value in state.experiment_store._values]
    reference_bits = state.reference._value.view(torch.int32).clone()
    weights = {name: value.view(torch.int32).clone() for name, value in state.runtime.model.state_dict().items()}
    binding = state.runtime.capture_binding()

    entered, release, controlled_cell_completed, request_finished = (threading.Event() for _ in range(4))
    original_intervene = state.runtime.intervene_fixed_forward

    def blocked_cell(*args, **kwargs):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("test did not release the controlled cell")
        result = original_intervene(*args, **kwargs)
        controlled_cell_completed.set()
        return result

    monkeypatch.setattr(state.runtime, "intervene_fixed_forward", blocked_cell)
    server = create_server(state)
    original_process = server.process_request_thread

    def process(request, address):
        try:
            return original_process(request, address)
        finally:
            request_finished.set()

    server.process_request_thread = process
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    client = None
    try:
        worker.start()
        client = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
        payload = {"capture_id": state._capture_id, "event": 0, "vector": 1,
                   "layers": [0], "sites": ["attention"]}
        body = json.dumps(payload).encode("utf-8")
        header = (f"POST /api/sensitivity HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                  f"X-MLG-Token: {state.token}\r\nContent-Type: application/json\r\n"
                  f"Content-Length: {len(body)}\r\n\r\n").encode("ascii")
        client.sendall(header + body)
        assert entered.wait(10)
        # A TCP reset models a lost browser connection while the test seam holds
        # the server at a safe boundary before the controlled cell begins.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                          struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0))
        client.close()
        client = None
        release.set()
        assert controlled_cell_completed.wait(20)
        assert request_finished.wait(10)
        assert state.runtime.capture_binding() is binding is state._live_binding
        assert state.runtime.info()["state"] == "ready"
        assert state.store.summary() == capture_summary
        assert state.experiment_store.summary() == comparison_summary
        assert state._experiment_id == manual["experiment_id"]
        assert state._reference_id == reference["reference_id"]
        assert torch.equal(state.reference._value.view(torch.int32), reference_bits)
        assert all(torch.equal(value.view(torch.int32), expected)
                   for value, expected in zip(state.store._values, capture_bits))
        assert all(torch.equal(value.view(torch.int32), expected)
                   for value, expected in zip(state.experiment_store._values, comparison_bits))
        assert all(torch.equal(value.view(torch.int32), weights[name])
                   for name, value in state.runtime.model.state_dict().items())
        assert not any(module._forward_hooks or module._forward_pre_hooks for module in state.runtime.model.modules())

        # Closing a client is not cancellation; a later authenticated request is
        # still usable after the original cell has fully completed.
        code, response, _ = call((state, server), "/api/sensitivity", "POST", payload)
        assert code == 200 and json.loads(response)["status"] == "complete"
    finally:
        release.set()
        if client is not None:
            client.close()
        server.shutdown()
        worker.join(5)
        server.server_close()
        assert not worker.is_alive()
