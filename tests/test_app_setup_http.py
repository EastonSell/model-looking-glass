"""Checkpoint input preparation fails before losing the current live capture."""
import json
import threading

import pytest
import transformers

from looking_glass.app import AppState, create_server
from looking_glass.app_runtime import LocalModelRuntime, FAMILIES
from test_app_http import call
from test_app_runtime import local_checkpoint


@pytest.mark.parametrize("family", ["gpt2", "distilbert", "t5", "vit"])
def test_input_setup_failure_preserves_live_owner_and_capture_over_http(tmp_path, monkeypatch, family):
    checkpoint = tmp_path / "checkpoint"
    local_checkpoint(checkpoint, family)
    assets = {p: p.read_bytes() for p in checkpoint.iterdir()
              if p.name != "config.json" and p.suffix != ".safetensors"}
    for path in assets:
        path.unlink()
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports")
    server = create_server(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = state, server
    try:
        assert call(endpoint, "/api/demo", "POST", {})[0] == 200
        code, body, _ = call(endpoint, "/api/run", "POST", {"prompt": "hello", "settings": {"max_new_tokens": 1}})
        assert code == 200, body
        original = json.loads(body)
        owner, binding = state.runtime.model, state.runtime.capture_binding()
        source_url = "/api/tensor?event=0&vector=0&capture_id=" + original["capture_id"]
        raw = json.loads(call(endpoint, source_url)[1])
        model_class = getattr(transformers, FAMILIES[family]["architecture"])
        def forbidden(*args, **kwargs):
            raise AssertionError("input preflight must not load model weights")
        with monkeypatch.context() as guard:
            guard.setattr(model_class, "from_pretrained", forbidden)
            for action in ("inspect", "load"):
                code, body, _ = call(endpoint, "/api/" + action, "POST", {"path": str(checkpoint), "device": "cpu"})
                assert code == 400, body
                assert "tokenizer" in json.loads(body)["error"].lower() or "processor" in json.loads(body)["error"].lower()
                assert state.runtime.model is owner and state.runtime.capture_binding() is binding
                status = json.loads(call(endpoint)[1])
                assert status["run"]["capture_id"] == original["capture_id"]
                assert status["run"]["intervention_available"]
                assert json.loads(call(endpoint, source_url)[1]) == raw
            for path, data in assets.items():
                path.write_bytes(data)
            code, body, _ = call(endpoint, "/api/inspect", "POST", {"path": str(checkpoint)})
            assert code == 200, body
            inspection = json.loads(body)
            assert inspection["supported_devices"] == (["cpu"] if family == "vit" else ["cpu", "cuda"])
            checks = inspection["inspection"]
            assert checks["metadata_checked"] and checks["weight_headers_checked"] and checks["input_assets_loadable"]
            assert checks["weights_loaded"] is False and checks["inference_tested"] is False
            assert inspection["input_summary"]["kind"] == ("image_processor" if family == "vit" else "fast_tokenizer")
            assert state.runtime.model is owner and state.runtime.capture_binding() is binding
            assert json.loads(call(endpoint, source_url)[1]) == raw
        code, body, _ = call(endpoint, "/api/load", "POST", {"path": str(checkpoint), "device": "cpu"})
        assert code == 200, body
        status = json.loads(call(endpoint)[1])
        assert status["model"]["family"] == family and status["run"] is None
        assert state.runtime.capture_binding() is not binding
        assert not state.store.export_root.exists()
    finally:
        server.shutdown(); worker.join(5); server.server_close()
        assert not worker.is_alive()
