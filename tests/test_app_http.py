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



@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_failed_open_preserves_current_capture_revision_and_values(server, tmp_path, failure):
    from pathlib import Path
    status, body, _ = call(server, "/api/run", "POST", {"prompt": "keep this capture", "settings": {}})
    assert status == 200
    original = json.loads(body); capture = original["capture_id"]
    url = f"/api/tensor?event=0&vector=1&capture_id={capture}"
    raw = json.loads(call(server, url)[1])
    path = tmp_path / "missing-export"
    if failure == "corrupt":
        saved = json.loads(call(server, "/api/save", "POST", {"name": "corrupt", "capture_id": capture})[1])
        path = Path(saved["path"])
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tensor_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    code, body, _ = call(server, "/api/open", "POST", {"path": str(path)})
    assert code == 400 and "error" in json.loads(body)
    assert json.loads(call(server)[1])["run"] == original
    assert json.loads(call(server, url)[1]) == raw
    saved = json.loads(call(server, "/api/save", "POST", {"name": "still-here", "capture_id": capture})[1])
    code, body, _ = call(server, "/api/open", "POST", {"path": saved["path"]})
    assert code == 200 and json.loads(body)["capture_id"] != capture
    assert call(server, url)[0] == 400  # Successful replacement, unlike failed Open, retires old reads.


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
        coordinates = json.loads(call(endpoint, "/api/vectors?event=0&limit=100")[1])
        assert coordinates["vectors"][0]["coordinate"]["status"] == "known"
        assert raw["coordinate"] == coordinates["vectors"][0]["coordinate"]
        stack = "encoder" if family == "t5" else "image" if family == "vit" else "input"
        filtered_url = f"/api/events?token_position=1&forward=0&stack={stack}"
        code, body, _ = call(endpoint, filtered_url)
        assert code == 200, body
        filtered = json.loads(body)
        assert filtered["total"] == 6  # two layers, three supported roles at one source position
        for row in filtered["events"]:
            assert row["matched_vector_index"] == (1 if family == "gpt2" else 0)
            selected = json.loads(call(endpoint, f"/api/tensor?event={row['index']}&vector={row['matched_vector_index']}")[1])
            assert selected["coordinate"] == row["source_summary"]["coordinate"]
            assert selected["coordinate"]["token_position"] == 1
            assert selected["coordinate"]["stack"] == stack
        saved = json.loads(call(endpoint, "/api/save", "POST", {"name": family})[1])
        assert call(endpoint, "/api/clear", "POST", {})[0] == 200
        assert call(endpoint, "/api/open", "POST", {"path": saved["path"]})[0] == 200
        assert json.loads(call(endpoint, "/api/tensor?event=0&vector=0&limit=4096")[1]) == raw
        assert json.loads(call(endpoint, "/api/vectors?event=0&limit=100")[1]) == coordinates
        assert json.loads(call(endpoint, filtered_url)[1]) == filtered
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


@pytest.mark.parametrize("family", ["gpt2", "distilbert", "vit"])
def test_cuda_unavailable_is_not_a_successful_cpu_load(tmp_path, monkeypatch, family):
    from test_app_runtime import local_checkpoint
    from looking_glass.app_runtime import LocalModelRuntime
    checkpoint = tmp_path / "model"; local_checkpoint(checkpoint, family)
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


def test_model_browser_explicit_root_containment_and_no_implicit_home(tmp_path):
    root = tmp_path / "models"; root.mkdir()
    child = root / "checkpoint"; child.mkdir(); (child / "config.json").write_text('{}')
    (root / "other.txt").write_text('not read')
    outside = tmp_path / "outside"; outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    state = AppState(runtime=FakeRuntime(), export_root=tmp_path / "exports")
    with pytest.raises(ValueError, match="explicit"):
        state.post("browse-models", {"path": ""})
    result = state.post("browse-models", {"path": str(root)})
    assert result["root"] == result["path"] == str(root) and result["parent"] is None
    assert result["folders"] == [{"name": "checkpoint", "path": str(child), "is_checkpoint": True}]
    nested = state.post("browse-models", {"path": str(child), "root": str(root)})
    assert nested["parent"] == str(root)
    with pytest.raises(ValueError, match="root"):
        state.post("browse-models", {"path": str(outside), "root": str(root)})
    restricted = AppState(runtime=FakeRuntime(), export_root=tmp_path / "exports", models_dir=root)
    assert restricted.post("browse-models", {"path": ""})["root"] == str(root)
    with pytest.raises(ValueError, match="root"):
        restricted.post("browse-models", {"path": str(outside), "root": str(outside)})
    assert not (tmp_path / "exports").exists()


def test_browse_and_saved_routes_require_auth_and_unload_retains_trace(server, tmp_path):
    state = server[0]
    # Model discovery and exported captures have separate roots in this scenario.
    state.store.export_root = tmp_path / "exports"
    state.runtime.unload = lambda: setattr(state.runtime, "loaded", None)
    assert call(server, "/api/load", "POST", {"path": "model"})[0] == 200
    assert call(server, "/api/run", "POST", {"prompt": "capture", "settings": {}})[0] == 200
    before = state.store.summary()
    assert call(server, "/api/unload", "POST", {})[0] == 200
    assert state.store.summary() == before and state.runtime.info() is None
    root=tmp_path / "models"; root.mkdir()
    assert call(server, "/api/browse-models", "POST", {"path": str(root)}, token=False)[0] == 403
    assert call(server, "/api/browse-models", "POST", {"path": str(root)})[0] == 200
    assert call(server, "/api/saved", token=False)[0] == 403
    assert json.loads(call(server, "/api/saved")[1])["runs"] == []
    assert call(server, "/api/save", "POST", {"name": "catalog-item"})[0] == 200
    runs=json.loads(call(server, "/api/saved")[1])["runs"]
    assert runs[0]["name"] == "catalog-item" and runs[0]["status"] == "complete_unverified"
    result=json.loads(call(server, "/api/events?q=BLOCK.0")[1])
    assert result["total"] == result["total_all"] == 1
    assert json.loads(call(server, "/api/events?q=unmatched")[1])["total"] == 0


def test_failed_unload_keeps_capture_and_visible_runtime_state(server):
    state=server[0]
    assert call(server, "/api/load", "POST", {"path": "model"})[0] == 200
    assert call(server, "/api/run", "POST", {"prompt": "capture", "settings": {}})[0] == 200
    before=state.store.summary()
    def failure():
        state.runtime.loaded["state"]="quarantined"
        raise RuntimeError("cleanup uncertain; restart")
    state.runtime.unload=failure
    assert call(server, "/api/unload", "POST", {})[0] == 500
    assert state.store.summary() == before and state.runtime.info()["state"] == "quarantined"


def test_optional_discovery_failure_does_not_hide_runtime_status(tmp_path, monkeypatch):
    state=AppState(runtime=FakeRuntime(), export_root=tmp_path / "exports", models_dir=tmp_path / "missing")
    assert state.status()["model_discovery_warning"]
    def denied(data): raise PermissionError("synthetic denied directory")
    monkeypatch.setattr(state,"browse_models",denied)
    status=state.status()
    assert status["model"] is None and status["models"] == [] and status["cuda"]["available"] is False
    assert status["model_discovery_warning"]


def test_model_browser_listing_is_bounded_and_does_not_read_config(tmp_path):
    root=tmp_path / "models"; root.mkdir()
    for index in range(205):
        path=root / f"model-{index:03}"; path.mkdir(); (path / "config.json").write_bytes(b'invalid JSON is only a candidate')
    state=AppState(runtime=FakeRuntime(),export_root=tmp_path / "exports")
    result=state.post("browse-models",{"path":str(root)})
    assert len(result["folders"]) == 200 and result["truncated"]
    assert all(row["is_checkpoint"] for row in result["folders"])


def test_vector_navigation_api_is_authenticated_bounded_and_read_only(server):
    state = server[0]
    assert call(server, '/api/vectors?event=0')[0] == 400
    assert call(server, '/api/run', 'POST', {'prompt': 'capture', 'settings': {}})[0] == 200
    before = state.store.summary()
    assert call(server, '/api/vectors?event=0', token=False)[0] == 403
    code, body, _ = call(server, '/api/vectors?event=0&offset=1&limit=1')
    assert code == 200, body
    page = json.loads(body)
    assert page['total'] == 2 and [row['index'] for row in page['vectors']] == [1]
    # Generic legacy metadata has no grounded source coordinate; do not invent one.
    assert page['vectors'][0]['coordinate']['status'] == 'unknown'
    for query in ('event=-1', 'event=0&limit=201', 'event=0&offset=-1', 'event=0&event=0'):
        assert call(server, '/api/vectors?' + query)[0] == 400
    assert state.store.summary() == before


def test_source_event_filters_are_forwarded_and_revision_guarded(server):
    from test_app_storage import coordinate_result
    state = server[0]
    state.store.publish(coordinate_result(positions=[3, 1, 0]))
    state._capture_id = "source-filter-fixture"
    url = "/api/events?token_position=1&forward=0&stack=input&capture_id=source-filter-fixture"
    code, body, _ = call(server, url)
    assert code == 200, body
    result = json.loads(body)
    assert result["total"] == 1
    row = result["events"][0]
    assert row["index"] == 0 and row["matched_vector_index"] == 1
    assert row["source_summary"]["coordinate"]["token_position"] == 1
    assert json.loads(call(server, "/api/events?token_position=2")[1])["total"] == 0
    assert call(server, url.replace("source-filter-fixture", "stale"))[0] == 400
    assert call(server, url, token=False)[0] == 403
    assert state._capture_id == "source-filter-fixture"


@pytest.mark.parametrize("query", ["token_position=-1", "token_position=true", "forward=1.5", "forward=-1",
    "token_position=999999999999999999999999", "stack=unknown", "token_position=0&token_position=1"])
def test_invalid_source_event_filter_returns_actionable_http_error(server, query):
    from test_app_storage import coordinate_result
    server[0].store.publish(coordinate_result())
    code, body, _ = call(server, "/api/events?" + query)
    assert code == 400 and json.loads(body)["error"]
    assert server[0].store.summary()["event_count"] == 1
