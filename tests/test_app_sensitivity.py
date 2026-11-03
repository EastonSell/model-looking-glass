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


def test_sweep_preserves_unavailable_baseline_label_flags(state, monkeypatch):
    import looking_glass.app_interventions as engine
    monkeypatch.setattr(engine, "_token_text", lambda *_args: ("", True, False))
    report = state.post("sensitivity", request(state, layers=[0], sites=["attention"]))
    assert report["status"] == "complete"
    token = report["results"][0]["baseline_token"]
    assert token["text_unavailable"] and not token["text_truncated"]


def test_full_baseline_digest_detects_non_top_logit_permutation():
    from looking_glass.app_interventions import _full_logits_digest
    first = torch.tensor([[[1., 2., 10., -1.]]])
    second = first[:, :, [1, 0, 2, 3]].contiguous()
    assert first.argmax() == second.argmax()
    assert first.double().softmax(-1)[0, 0, 2] == second.double().softmax(-1)[0, 0, 2]
    assert _full_logits_digest(first) != _full_logits_digest(second)
    assert _full_logits_digest(first) != _full_logits_digest(first.reshape(1, 2, 2))


def test_sweep_detects_real_non_top_logit_drift_with_unchanged_top_probability(state, monkeypatch):
    import looking_glass.app_interventions as engine
    model = state.runtime.model
    step = state.store.summary()["steps"][0]
    ids = torch.tensor([step["input_token_ids"]], dtype=torch.long)
    mask = torch.tensor([step["input_attention_mask"]], dtype=torch.long)
    with torch.inference_mode():
        initial = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0, -1].double()
    top = int(initial.argmax())
    probability = initial.log_softmax(-1)[top].exp()
    pair = None
    for left in range(initial.numel()):
        for right in range(left + 1, initial.numel()):
            if top in (left, right) or initial[left] == initial[right]:
                continue
            changed = initial.clone()
            changed[left], changed[right] = initial[right], initial[left]
            if changed.log_softmax(-1)[top].exp() == probability:
                pair = (left, right)
                break
        if pair is not None:
            break
    assert pair is not None, "fixture must have distinct non-top logits with an unchanged top probability"
    execute = engine.execute
    forward = model.forward
    baseline_keys = []
    digests = []

    def after_preflight(owner, validated):
        if baseline_keys:
            def permuted(*args, **kwargs):
                output = forward(*args, **kwargs)
                logits = output.logits.clone()
                left, right = pair
                logits[0, -1, left], logits[0, -1, right] = output.logits[0, -1, right], output.logits[0, -1, left]
                output.logits = logits
                return output
            monkeypatch.setattr(owner.model, "forward", permuted)
        try:
            result = execute(owner, validated)
        finally:
            monkeypatch.setattr(owner.model, "forward", forward)
        experiment = result["experiment"]
        token_id = experiment["comparison"]["baseline_next_token_id"]
        token = next(row for row in experiment["next_token_distribution"]["rows"] if row["token_id"] == token_id)
        baseline_keys.append((token_id, token["baseline_logit"], token["baseline_probability"]))
        digests.append(result["_baseline_digest"])
        return result

    monkeypatch.setattr(engine, "execute", after_preflight)
    report = state.post("sensitivity", request(state, layers=[0]))
    assert len(baseline_keys) == 2 and baseline_keys[0] == baseline_keys[1]
    assert digests[0] != digests[1]
    assert report["status"] == "inconclusive"
    assert [row["status"] for row in report["results"]] == ["ok", "failed", "skipped"]
    assert report["results"][1]["failure_kind"] == "baseline_mismatch"


def test_sweep_rejects_inconsistent_full_baselines_without_exporting_hash(state, monkeypatch):
    import json
    original = state.runtime.intervene_fixed_forward
    hashes = []
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        hashes.append(result["_baseline_digest"])
        if len(hashes) == 2:
            result["_baseline_digest"] = "0" * 64 if hashes[0] != "0" * 64 else "1" * 64
        return result
    monkeypatch.setattr(state.runtime, "intervene_fixed_forward", changed)
    report = state.post("sensitivity", request(state, layers=[0]))
    assert report["status"] == "inconclusive"
    assert [row["status"] for row in report["results"]] == ["ok", "failed", "skipped"]
    assert report["results"][1]["failure_kind"] == "baseline_mismatch"
    assert "Full baseline logits differed" in report["results"][1]["reason"]
    serialized = json.dumps(report)
    assert "_baseline_digest" not in serialized and all(value not in serialized for value in hashes)


def test_manual_comparison_and_export_omit_internal_digest(state):
    import json
    from pathlib import Path
    comparison = state.post("intervene", {**request(state), "operation": "zero"})
    assert "_baseline_digest" not in json.dumps(comparison)
    saved = state.post("save-experiment", {"experiment_id": comparison["experiment_id"], "name": "manual"})
    assert "_baseline_digest" not in (Path(saved["path"]) / "manifest.json").read_text(encoding="utf-8")


def test_missing_display_labels_do_not_quarantine_manual_intervention(state, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise ValueError("synthetic display failure")
    monkeypatch.setattr(state.runtime.tokenizer, "decode", unavailable)
    comparison = state.post("intervene", {**request(state), "operation": "zero"})
    assert state.runtime.info()["state"] == "ready"
    experiment = comparison["experiment"]
    assert all(row["text_unavailable"] for row in experiment["next_token_distribution"]["rows"])
    assert all(row["text_unavailable"] for row in experiment["baseline_top_next_tokens"])


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
