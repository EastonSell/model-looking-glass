"""A sensitivity report replays real captured GPT-2 boundaries with controls."""
import pytest
import torch

from looking_glass.app import AppState
from looking_glass.app_runtime import LocalModelRuntime
from test_app_runtime import local_checkpoint


@pytest.fixture
def state(tmp_path):
    path = tmp_path / "model"
    local_checkpoint(path, "gpt2")
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports",
                     download_root=tmp_path / "downloads")
    state.post("load", {"path": str(path), "device": "cpu"})
    state.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    return state


def request(state, **kwargs):
    return {"capture_id": state._capture_id, "event": 0, "vector": 1, **kwargs}


def test_real_sweep_reports_controls_and_preserves_capture_and_comparison(state):
    state.post("intervene", {**request(state), "operation": "zero"})
    comparison_id = state._experiment_id
    before = state.store.summary()
    weights = {key: value.clone() for key, value in state.runtime.model.state_dict().items()}

    report = state.post("sensitivity", request(state))

    assert report["capture_id"] == state._capture_id
    assert report["status"] == "complete"
    assert report["layers"] == [0, 1]
    assert report["sites"] == ["attention", "mlp", "block"]
    assert (report["planned_runs"], report["completed_runs"]) == (6, 6)
    assert (report["completed_model_forwards"], report["model_forwards_upper_bound"]) == (24, 24)
    assert report["anchor"]["token_position"] == 1
    assert report["anchor"]["input_token_ids"] == [5, 6]
    assert [(row["layer"], row["site"]) for row in report["results"]] == [
        (layer, site) for layer in (0, 1) for site in ("attention", "mlp", "block")]
    for row in report["results"]:
        assert row["status"] == "ok" and all(row["controls"].values())
        assert 0 <= row["total_variation"] <= 1
        token = row["baseline_token"]
        assert token["probability_delta"] == pytest.approx(
            token["intervention_probability"] - token["baseline_probability"])
        assert row["next_token_distribution"]["normalization"] == "full vocabulary"
    assert state.store.summary() == before and state._experiment_id == comparison_id
    assert all(torch.equal(value.view(torch.uint8), state.runtime.model.state_dict()[key].view(torch.uint8))
               for key, value in weights.items())


def test_uncaptured_sites_are_unsupported_not_zero(state):
    state.post("run", {"prompt": "hello world", "settings": {
        "targets": ["block"], "layers": [0], "max_new_tokens": 1}})
    report = state.post("sensitivity", request(state))
    assert report["status"] == "partial"
    assert [row["status"] for row in report["results"]] == ["unsupported", "unsupported", "ok"]
    for row in report["results"][:2]:
        assert row["reason"] and "baseline_token" not in row
    assert report["completed_runs"] == 1


def test_real_zero_vector_is_reported_as_a_successful_no_effect(state):
    # This is a fresh synthetic checkpoint in a temporary directory.
    with torch.no_grad():
        projection = state.runtime.model.transformer.h[0].mlp.c_proj
        projection.weight.zero_()
        projection.bias.zero_()
    state.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    report = state.post("sensitivity", request(state, layers=[0], sites=["mlp"]))
    row = report["results"][0]
    assert report["status"] == "complete" and row["status"] == "ok"
    assert not row["activation_changed"] and all(row["controls"].values())
    assert row["baseline_token"]["probability_delta"] == 0
    assert row["total_variation"] == row["last_position_max_abs_delta"] == 0


@pytest.mark.parametrize("changes", [
    {"layers": []}, {"layers": [0, 0]}, {"layers": [True]}, {"layers": [2]},
    {"sites": []}, {"sites": ["attn"]}, {"sites": ["mlp", "mlp"]},
    {"unknown": 1}, {"capture_id": "stale"},
])
def test_invalid_request_rejected_before_any_replay(state, monkeypatch, changes):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid request must not replay the model")
    monkeypatch.setattr(state.runtime, "intervene_fixed_forward", forbidden)
    with pytest.raises(ValueError):
        state.post("sensitivity", request(state, **changes))


@pytest.mark.parametrize("source", ["unloaded", "imported"])
def test_requires_live_model_binding(state, source, monkeypatch):
    if source == "imported":
        saved = state.post("save", {"name": "source"})
        state.post("open", {"path": saved["path"]})
    else:
        state.post("unload", {})
    def forbidden(*args, **kwargs):
        pytest.fail("unbound or imported capture must not replay the model")
    monkeypatch.setattr(state.runtime, "intervene_fixed_forward", forbidden)
    with pytest.raises(ValueError, match="live|ready|fresh"):
        state.post("sensitivity", request(state))


def test_control_failure_stops_remaining_sites_and_is_inconclusive(state, monkeypatch):
    import looking_glass.app_interventions as module
    execute = module.execute
    original = state.runtime.model.forward
    calls = []

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(None)
        if len(calls) == 2:
            result.logits = result.logits + 1
        return result

    def after_preflight(owner, validated):
        monkeypatch.setattr(owner.model, "forward", changed)
        return execute(owner, validated)

    monkeypatch.setattr(module, "execute", after_preflight)
    report = state.post("sensitivity", request(state, layers=[0]))
    assert report["status"] == "inconclusive"
    assert [row["status"] for row in report["results"]] == ["failed", "skipped", "skipped"]
    assert "null" in report["results"][0]["reason"].lower()
    assert "baseline_token" not in report["results"][0]
    assert report["completed_runs"] == 0
    assert state.runtime.info()["state"] == "quarantined"


def test_http_requires_token_and_current_capture_and_returns_report(state):
    import json
    import threading
    from looking_glass.app import create_server
    from test_app_http import call

    server = create_server(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        endpoint = (state, server)
        data = request(state, layers=[0], sites=["attention"])
        assert call(endpoint, "/api/sensitivity", "POST", data, token=False)[0] == 403
        assert call(endpoint, "/api/sensitivity", "POST", {**data, "capture_id": "stale"})[0] == 400
        status, body, _ = call(endpoint, "/api/sensitivity", "POST", data)
        assert status == 200, body
        result = json.loads(body)
        assert result["capture_id"] == data["capture_id"] and result["results"][0]["status"] == "ok"
    finally:
        server.shutdown()
        worker.join(5)
        server.server_close()
        assert not worker.is_alive()


def test_llama_sweep_matches_all_captured_boundaries(tmp_path):
    """Exercise the sweep's Llama path mapping, not just the manual replay."""
    from test_app_llama_http import LLAMA3_SCALING, llama_checkpoint

    checkpoint = tmp_path / "llama"
    llama_checkpoint(checkpoint, 2, LLAMA3_SCALING)
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports",
                     download_root=tmp_path / "downloads")
    state.post("load", {"path": str(checkpoint), "device": "cpu"})
    capture = state.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    original = state.store.summary()
    position = len(capture["steps"][0]["input_token_ids"]) - 1
    report = state.post("sensitivity", {
        "capture_id": capture["capture_id"], "event": 0, "vector": position,
        "layers": [0, 1], "sites": ["attention", "mlp", "block"],
    })
    assert report["status"] == "complete" and report["completed_runs"] == 6
    assert report["anchor"]["token_position"] == position
    assert all(row["status"] == "ok" and all(row["controls"].values())
               and row["controls"]["static_rotary_state_unchanged"]
               and row["controls"]["fixed_input_ids_mask_unchanged"]
               for row in report["results"])
    assert state.store.summary() == original
