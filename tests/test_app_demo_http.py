"""Explicit demo loading and real, no-checkpoint onboarding over loopback HTTP."""
import json
import threading

import pytest

from looking_glass.app import AppState, create_server
from test_app_http import call, FakeRuntime


class DemoRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.demo_calls = 0
        self.fail_demo = False

    def load_demo(self):
        self.demo_calls += 1
        if self.fail_demo:
            raise ValueError("demo parameter budget exceeded")
        self.loaded = {"family": "gpt2", "device": "cpu", "state": "ready",
                       "path": None, "provenance": {"kind": "synthetic_untrained_demo"}}
        return self.info()


@pytest.fixture
def demo_server(tmp_path):
    state = AppState(runtime=DemoRuntime(), export_root=tmp_path / "exports")
    httpd = create_server(state)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield state, httpd
    finally:
        httpd.shutdown(); worker.join(5); httpd.server_close()
        assert not worker.is_alive()


def test_demo_requires_explicit_authenticated_post(demo_server):
    state, _ = demo_server
    assert call(demo_server, "/api/demo", "POST", {}, token=False)[0] == 403
    assert call(demo_server, "/api/demo", "POST", {}, headers={"Origin": "https://evil.invalid"})[0] == 403
    assert call(demo_server, "/api/demo")[0] == 404
    assert state.runtime.demo_calls == 0
    code, body, _ = call(demo_server, "/api/demo", "POST", {})
    assert code == 200, body
    result = json.loads(body)
    assert result["model"]["provenance"]["kind"] == "synthetic_untrained_demo"
    assert result["model"]["device"] == "cpu" and result["run"] is None
    assert state.runtime.demo_calls == 1
    assert not state.store.export_root.exists()


@pytest.mark.parametrize("payload", [{"device": "cuda:0"}, {"seed": 99}, {"path": "elsewhere"}, {"config": {"n_layer": 1000}}])
def test_demo_rejects_hidden_configuration_before_runtime(demo_server, payload):
    assert call(demo_server, "/api/demo", "POST", payload)[0] == 400
    assert demo_server[0].runtime.demo_calls == 0


def test_demo_success_replaces_captures_but_failure_preserves_them(demo_server):
    state, _ = demo_server
    assert call(demo_server, "/api/run", "POST", {"prompt": "original"})[0] == 200
    state.experiment_store.publish(state.runtime.run("comparison", {}))
    state._experiment_id = "comparison-id"
    before = state.status()
    state.runtime.fail_demo = True
    assert call(demo_server, "/api/demo", "POST", {})[0] == 400
    assert state.status()["run"] == before["run"]
    assert state.status()["experiment"] == before["experiment"]
    state.runtime.fail_demo = False
    assert call(demo_server, "/api/demo", "POST", {})[0] == 200
    assert state.status()["run"] is None and state.status()["experiment"] is None
    assert state._capture_id is state._experiment_id is state._live_binding is None


def test_real_demo_query_edit_export_and_reopen_without_checkpoint(tmp_path):
    from looking_glass.app_runtime import LocalModelRuntime
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports")
    httpd = create_server(state)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    endpoint = state, httpd
    try:
        code, body, _ = call(endpoint, "/api/demo", "POST", {})
        assert code == 200, body
        model = json.loads(body)["model"]
        assert model["provenance"]["kind"] == "synthetic_untrained_demo" and model["path"] is None
        code, body, _ = call(endpoint, "/api/run", "POST", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
        assert code == 200, body
        run = json.loads(body)
        assert run["model"]["provenance"] == model["provenance"] and run["intervention_available"]
        capture_id = run["capture_id"]
        source_url = f"/api/tensor?event=0&vector=0&limit=4096&capture_id={capture_id}"
        raw = json.loads(call(endpoint, source_url)[1])
        assert raw["coordinate"]["status"] == "known"
        code, body, _ = call(endpoint, "/api/intervene", "POST", {"capture_id": capture_id, "event": 0, "vector": 0, "operation": "zero"})
        assert code == 200, body
        comparison = json.loads(body)
        assert comparison["model"]["provenance"] == model["provenance"]
        controls = comparison["experiment"]["controls"]
        assert controls["null_full_logits_bitwise_equal"] and controls["restoration_full_logits_bitwise_equal"]
        assert json.loads(call(endpoint, source_url)[1]) == raw
        assert not state.store.export_root.exists()
        saved = json.loads(call(endpoint, "/api/save", "POST", {"name": "demo", "capture_id": capture_id})[1])
        assert call(endpoint, "/api/unload", "POST", {})[0] == 200
        code, body, _ = call(endpoint, "/api/open", "POST", {"path": saved["path"]})
        assert code == 200, body
        opened = json.loads(body)
        assert opened["model"]["provenance"] == model["provenance"]
        assert not opened["intervention_available"]
        assert json.loads(call(endpoint, f'/api/tensor?event=0&vector=0&limit=4096&capture_id={opened["capture_id"]}')[1]) == raw
    finally:
        httpd.shutdown(); worker.join(5); httpd.server_close()
        assert not worker.is_alive()
