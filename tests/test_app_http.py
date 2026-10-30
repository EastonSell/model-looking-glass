"""Real loopback HTTP boundary tests, without loading a model checkpoint."""
import http.client
import json
import threading
from types import SimpleNamespace

import pytest
import torch

from looking_glass.app import AppState, create_server, MAX_BODY


class FakeRuntime:
    def __init__(self): self.loaded = None
    def info(self): return self.loaded
    def inspect(self, path): return {"path": path, "family": "gpt2"}
    def capabilities(self): return {"devices": [{"id": "cpu", "name": "CPU", "available": True}], "cuda": {"available": False}}
    def load(self, path, **kwargs): self.loaded = {"name": path, "family": "gpt2", "targets": ["block"], "layers": 1}; return self.loaded
    def run(self, prompt, settings, image_bytes=None):
        if prompt == "fail": raise RuntimeError("synthetic failure")
        event = SimpleNamespace(value=torch.ones(1, 2, 4), metadata=lambda: {"module_path": "block.0"})
        return {"answer": prompt, "task": "synthetic", "events": [event], "coverage": "fixture"}


@pytest.fixture
def server(tmp_path):
    state = AppState(runtime=FakeRuntime(), export_root=tmp_path)
    httpd = create_server(state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    yield state, httpd
    httpd.shutdown(); thread.join(5); httpd.server_close()
    assert not thread.is_alive()


def call(server, path="/api/status", method="GET", data=None, headers=None, token=True):
    state, httpd = server
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    base = {"X-MLG-Token": state.token} if token else {}
    if data is not None: base["Content-Type"] = "application/json"
    base.update(headers or {})
    conn.request(method, path, body=None if data is None else json.dumps(data), headers=base)
    response = conn.getresponse(); body = response.read(); status = response.status; response_headers = dict(response.getheaders()); conn.close()
    return status, body, response_headers


def test_static_page_token_and_safe_headers(server):
    code, body, headers = call(server, "/", token=False)
    assert code == 200 and server[0].token.encode() in body and b"__MLG_TOKEN__" not in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    assert call(server, "/../../pyproject.toml")[0] == 404


def test_auth_origin_and_host_guards(server):
    assert call(server, token=False)[0] == 403
    assert call(server, headers={"X-MLG-Token": "wrong"})[0] == 403
    assert call(server, headers={"Host": "attacker.invalid"})[0] == 403
    assert call(server, headers={"Origin": "https://attacker.invalid"})[0] == 403
    assert call(server, "/api/load", "POST", {"path": "x"}, token=False)[0] == 403
    assert call(server, headers={"Origin": f"http://127.0.0.1:{server[1].server_port}"})[0] == 200


def test_real_http_run_browse_save_reopen(server):
    assert call(server, "/api/load", "POST", {"path": "local-model"})[0] == 200
    status, body, _ = call(server, "/api/run", "POST", {"prompt": "<script>not HTML</script>", "settings": {}})
    assert status == 200 and json.loads(body)["event_count"] == 1
    assert call(server, "/api/events")[0] == 200
    assert json.loads(call(server, "/api/tensor?event=0&vector=1&limit=2")[1])["values"] == [1., 1.]
    saved = json.loads(call(server, "/api/save", "POST", {"name": "sample"})[1])
    assert call(server, "/api/clear", "POST", {})[0] == 200
    assert json.loads(call(server)[1])["run"] is None
    assert call(server, "/api/open", "POST", {"path": saved["path"]})[0] == 200


def test_failed_run_drops_stale_result(server):
    assert call(server, "/api/run", "POST", {"prompt": "ok", "settings": {}})[0] == 200
    assert call(server, "/api/run", "POST", {"prompt": "fail", "settings": {}})[0] == 500
    assert json.loads(call(server)[1])["run"] is None


def test_limits_and_bad_requests(server):
    assert call(server, "/api/run", "POST", {"prompt": "x", "settings": {"max_trace_bytes": 2**40}})[0] == 400
    assert call(server, "/api/run", "POST", {"prompt": "x", "image_base64": "not base64"})[0] == 400
    assert call(server, "/api/events?offset=0&offset=1")[0] == 400
    assert call(server, "/api/run", "POST", {}, headers={"Content-Type": "text/plain"})[0] == 415
    assert call(server, "/api/save", "POST", {"name": "../escape"})[0] == 400
    assert call(server, "/api/status", "POST", {})[0] == 404


@pytest.mark.parametrize("family", ["gpt2", "distilbert", "t5", "vit"])
def test_actual_local_checkpoint_through_http_and_reopen(tmp_path, family):
    import base64
    from test_app_runtime import local_checkpoint, image_bytes
    from looking_glass.app_runtime import LocalModelRuntime
    checkpoint = tmp_path / "model"
    local_checkpoint(checkpoint, family)
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports")
    httpd = create_server(state); thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    endpoint = (state, httpd)
    try:
        assert call(endpoint, "/api/load", "POST", {"path": str(checkpoint), "device": "cpu"})[0] == 200
        request = {"prompt": "hello [MASK]" if family == "distilbert" else "hello world", "settings": {"max_new_tokens": 2}}
        if family == "vit": request["image_base64"] = base64.b64encode(image_bytes()).decode()
        code, body, _ = call(endpoint, "/api/run", "POST", request)
        assert code == 200, body.decode()
        report = json.loads(body)
        assert report["model"]["family"] == family and report["complete"]
        assert report["settings"]["max_new_tokens"] == 2
        raw = json.loads(call(endpoint, "/api/tensor?event=0&vector=0&limit=4096")[1])
        saved = json.loads(call(endpoint, "/api/save", "POST", {"name": family})[1])
        assert call(endpoint, "/api/clear", "POST", {})[0] == 200
        assert call(endpoint, "/api/open", "POST", {"path": saved["path"]})[0] == 200
        assert json.loads(call(endpoint, "/api/tensor?event=0&vector=0&limit=4096")[1]) == raw
    finally:
        httpd.shutdown(); thread.join(5); httpd.server_close()


def test_device_inventory_and_load_options_are_explicit(server):
    state = server[0]
    state.runtime.capabilities = lambda: {"devices": [{"id": "cpu", "name": "CPU", "available": True}], "cuda": {"available": False, "reason": "CPU-only fixture"}, "supported_cuda_families": ["gpt2", "t5"]}
    received = []
    def load(path, **kwargs):
        received.append((path, kwargs)); return {"family": "gpt2", "device": kwargs["device"]}
    state.runtime.load = load
    body = json.loads(call(server)[1])
    assert body["cuda"]["available"] is False and body["devices"][0]["id"] == "cpu"
    response = call(server, "/api/load", "POST", {"path": "local", "device": "cuda:1", "max_vram_bytes": 2**30})
    assert response[0] == 200
    assert received == [("local", {"device": "cuda:1", "max_vram_bytes": 2**30})]
    for budget in (0, True, 2**50, "100"):
        assert call(server, "/api/load", "POST", {"path": "local", "max_vram_bytes": budget})[0] == 400
    assert len(received) == 1


def test_cuda_unavailable_is_not_a_successful_cpu_load(tmp_path, monkeypatch):
    from test_app_runtime import local_checkpoint
    from looking_glass.app_runtime import LocalModelRuntime
    checkpoint = tmp_path / "model"; local_checkpoint(checkpoint, "gpt2")
    runtime = LocalModelRuntime()
    monkeypatch.setattr(LocalModelRuntime, "capabilities", staticmethod(lambda: {
        "devices": [{"id": "cpu", "name": "CPU", "available": True}],
        "cuda": {"available": False, "reason": "Synthetic unavailable device"},
        "supported_cuda_families": ["gpt2", "t5"]}))
    state = AppState(runtime=runtime, export_root=tmp_path / "exports")
    with pytest.raises(ValueError, match="unavailable"):
        state.post("load", {"path": str(checkpoint), "device": "cuda:0"})
    assert runtime.info() is None and state.store.summary() is None
    info = state.post("load", {"path": str(checkpoint), "device": "cpu"})
    assert info["device"] == "cpu"


def test_lowered_gpu_server_allowance_defaults_apply_to_cpu_request(tmp_path):
    received = []
    fake = FakeRuntime()
    fake.load = lambda path, **kwargs: received.append(kwargs) or {"device": kwargs["device"]}
    state = AppState(runtime=fake, export_root=tmp_path, max_vram_bytes=512 * 1024**2)
    state.post("load", {"path": "model", "device": "cpu"})
    assert received == [{"device": "cpu", "max_vram_bytes": 512 * 1024**2}]
