"""Authenticated loopback tests for cooperative experiment stop jobs."""

import http.client
import json
import socket
import struct
import sys
import threading

import pytest

from looking_glass.app import AppState, create_server
from looking_glass.app_experiment_jobs import ExperimentJobs


@pytest.fixture
def endpoint(tmp_path):
    state = AppState(runtime=object(), export_root=tmp_path, download_root=tmp_path / "downloads")
    server = create_server(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield state, server
    finally:
        server.shutdown()
        worker.join(5)
        server.server_close()
        assert not worker.is_alive()


def request(endpoint, path, data, *, token=True, headers=None, timeout=3):
    state, server = endpoint
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=timeout)
    hdrs = {"Content-Type": "application/json"}
    if token:
        hdrs["X-MLG-Token"] = state.token
    hdrs.update(headers or {})
    try:
        connection.request("POST", path, json.dumps(data), hdrs)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def reserve(endpoint, action="sensitivity", capture="capture-test"):
    status, result = request(endpoint, "/api/experiment-reserve", {"action": action, "capture_id": capture})
    assert status == 200
    assert result["action"] == action and result["status"] == "reserved"
    return result["job_id"]


def test_authentication_action_binding_single_use_and_legacy_compatibility(endpoint):
    state, _ = endpoint
    calls = []

    def post(action, data, *, cancel_event=None):
        calls.append((action, data, cancel_event))
        return {"status": "complete"}

    state.post = post
    body = {"action": "sensitivity", "capture_id": "capture-test"}
    assert request(endpoint, "/api/experiment-reserve", body, token=False)[0] == 403
    assert request(endpoint, "/api/experiment-reserve", body, headers={"Origin": "https://attacker.invalid"})[0] == 403
    assert request(endpoint, "/api/experiment-reserve", body, headers={"Host": "attacker.invalid"})[0] == 403
    assert request(endpoint, "/api/experiment-reserve", {**body, "unknown": 1})[0] == 400
    assert request(endpoint, "/api/experiment-reserve", {**body, "action": "run"})[0] == 400
    job_id = reserve(endpoint)
    assert request(endpoint, "/api/experiment-reserve", body)[0] == 400
    assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id}, token=False)[0] == 403
    assert request(endpoint, "/api/experiment-status", {"job_id": job_id}, token=False)[0] == 403
    assert request(endpoint, "/api/sensitivity-repeat", {"capture_id": "capture-test", "job_id": job_id})[0] == 400
    assert request(endpoint, "/api/sensitivity", {"capture_id": "other", "job_id": job_id})[0] == 400
    assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[1]["status"] == "reserved"
    status, result = request(endpoint, "/api/sensitivity", {"capture_id": "capture-test", "job_id": job_id})
    assert status == 200 and result["status"] == "complete"
    assert len(calls) == 1 and calls[0][0:2] == ("sensitivity", {"capture_id": "capture-test"})
    assert isinstance(calls[0][2], threading.Event)
    assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[0] == 400
    assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id})[0] == 400
    assert request(endpoint, "/api/sensitivity", {"capture_id": "capture-test", "job_id": job_id})[0] == 400
    # Existing clients do not reserve and still reach exactly the old path.
    assert request(endpoint, "/api/sensitivity", {"capture_id": "legacy"})[0] == 200
    assert calls[-1] == ("sensitivity", {"capture_id": "legacy"}, None)


def test_orphan_reservation_expires_and_stop_before_claim_is_retained(endpoint):
    state, _ = endpoint
    now = [100.0]
    state.experiment_jobs = ExperimentJobs(reservation_ttl=1, clock=lambda: now[0])
    job_id = reserve(endpoint)
    assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id})[1]["status"] == "stopping"
    assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[1]["status"] == "stopping"
    now[0] += 1.1
    assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[0] == 400
    assert request(endpoint, "/api/sensitivity", {"capture_id": "capture-test", "job_id": job_id})[0] == 400
    assert reserve(endpoint) != job_id


def test_queued_job_can_be_stopped_without_waiting_for_model_lock(endpoint):
    state, _ = endpoint
    claimed = threading.Event()
    original_claim = state.experiment_jobs.claim

    def claim(*args):
        lease = original_claim(*args)
        claimed.set()
        return lease

    state.experiment_jobs.claim = claim
    observed, response = [], []

    def post(_action, data, *, cancel_event=None):
        observed.append((data, cancel_event.is_set()))
        return {"status": "cancelled" if cancel_event.is_set() else "complete"}

    state.post = post
    job_id = reserve(endpoint)
    state.lock.acquire()
    worker = threading.Thread(target=lambda: response.append(request(endpoint, "/api/sensitivity", {
        "capture_id": "capture-test", "job_id": job_id}, timeout=10)), daemon=True)
    try:
        worker.start()
        assert claimed.wait(5)
        assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[1]["status"] == "queued"
        assert request(endpoint, "/api/sensitivity", {"capture_id": "capture-test", "job_id": job_id})[0] == 400
        assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id})[1]["status"] == "stopping"
        assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[1]["status"] == "stopping"
    finally:
        state.lock.release()
        worker.join(10)
    assert not worker.is_alive()
    assert response == [(200, {"status": "cancelled"})]
    assert observed == [({"capture_id": "capture-test"}, True)]
    assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[0] == 400


def test_stop_running_job_returns_promptly_and_does_not_override_completion(endpoint):
    state, _ = endpoint
    started, finish = threading.Event(), threading.Event()
    observed, response, events = [], [], []

    def post(_action, _data, *, cancel_event=None):
        events.append(cancel_event)
        started.set()
        assert cancel_event.wait(5)
        observed.append(cancel_event.is_set())
        assert finish.wait(5)
        # Simulate a stop arriving after the last safe checkpoint.
        return {"status": "complete"}

    state.post = post
    job_id = reserve(endpoint, "patch-cross-prompt")
    worker = threading.Thread(target=lambda: response.append(request(endpoint, "/api/patch-cross-prompt", {
        "capture_id": "capture-test", "job_id": job_id}, timeout=10)), daemon=True)
    try:
        worker.start()
        assert started.wait(5)
        assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[1]["status"] == "running"
        assert request(endpoint, "/api/experiment-reserve", {"action": "sensitivity", "capture_id": "capture-test"})[0] == 400
        assert request(endpoint, "/api/patch-cross-prompt", {"capture_id": "capture-test", "job_id": job_id})[0] == 400
        forged = "0" * len(job_id) if job_id != "0" * len(job_id) else "1" * len(job_id)
        assert request(endpoint, "/api/experiment-cancel", {"job_id": forged})[0] == 400
        assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id, "extra": 1})[0] == 400
        assert not events[0].is_set()
        assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id})[1]["status"] == "stopping"
        assert worker.is_alive()  # Current safe operation has not yet finished.
    finally:
        finish.set()
        worker.join(10)
    assert not worker.is_alive() and observed == [True]
    assert response == [(200, {"status": "complete"})]
    assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id})[0] == 400


def test_failed_and_disconnected_jobs_release_the_reservation(endpoint):
    state, server = endpoint

    def failed(_action, _data, *, cancel_event=None):
        raise ValueError("synthetic validation failure")

    state.post = failed
    job_id = reserve(endpoint)
    assert request(endpoint, "/api/sensitivity", {"capture_id": "capture-test", "job_id": job_id})[0] == 400
    assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[0] == 400

    started, release = threading.Event(), threading.Event()
    released = threading.Event()
    old_release = state.experiment_jobs.release

    def tracked_release(lease):
        try:
            return old_release(lease)
        finally:
            released.set()

    state.experiment_jobs.release = tracked_release

    def delayed(_action, _data, *, cancel_event=None):
        started.set()
        assert release.wait(5)
        return {"status": "complete"}

    state.post = delayed
    job_id = reserve(endpoint)
    client = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
    body = json.dumps({"capture_id": "capture-test", "job_id": job_id}).encode("ascii")
    headers = (f"POST /api/sensitivity HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\n"
               f"X-MLG-Token: {state.token}\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode("ascii")
    try:
        client.sendall(headers + body)
        assert started.wait(5)
        linger = struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        client.close()
        release.set()
        assert released.wait(5)
        assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[0] == 400
        assert reserve(endpoint) != job_id
    finally:
        release.set()
        client.close()


def test_preclaimed_stop_preserves_real_capture_and_comparison(tmp_path, monkeypatch):
    """Exercise the real model/runner through the authenticated stop route."""
    from looking_glass.app_runtime import LocalModelRuntime
    from test_app_runtime import local_checkpoint

    checkpoint = tmp_path / "model"
    local_checkpoint(checkpoint, "gpt2")
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports",
                     download_root=tmp_path / "downloads")
    state.post("load", {"path": str(checkpoint), "device": "cpu"})
    capture = state.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    comparison = state.post("intervene", {"capture_id": capture["capture_id"],
                                          "event": 0, "vector": 1, "operation": "zero"})
    original = state.store.summary()
    comparison_before = state.experiment_store.summary()
    binding = state._live_binding
    server = create_server(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        endpoint = (state, server)
        job_id = reserve(endpoint, capture=capture["capture_id"])
        assert request(endpoint, "/api/experiment-cancel", {"job_id": job_id})[1]["status"] == "stopping"

        def forbidden(*_args, **_kwargs):
            pytest.fail("pre-claimed stop must prevent every model forward")

        monkeypatch.setattr(state.runtime.model, "forward", forbidden)
        status, result = request(endpoint, "/api/sensitivity", {
            "capture_id": capture["capture_id"], "event": 0, "vector": 1,
            "layers": [0], "sites": ["attention"], "job_id": job_id}, timeout=10)
        assert status == 200 and result["status"] == "cancelled"
        assert result["completed_runs"] == 0
        assert all(row["status"] == "skipped" for row in result["results"])
        assert state.store.summary() == original
        assert state.experiment_store.summary() == comparison_before
        assert state._experiment_id == comparison["experiment_id"]
        assert state._live_binding is binding is state.runtime.capture_binding()
        assert request(endpoint, "/api/experiment-status", {"job_id": job_id})[0] == 400
    finally:
        server.shutdown()
        worker.join(5)
        server.server_close()
        assert not worker.is_alive()
