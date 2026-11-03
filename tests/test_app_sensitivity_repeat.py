"""Repeated prompt tests run on one live owner without replacing its capture."""
import pytest
import torch

from test_app_sensitivity import state


def request(state, **changes):
    return {"capture_id": state._capture_id, "layers": [0], "sites": ["attention", "mlp", "block"],
            "prompts": ["hello world", "world hello"], **changes}


def test_real_repeated_sweep_preserves_retained_data_and_owner(state):
    reference = state.post("pin-reference", {"capture_id": state._capture_id, "event": 0, "vector": 1,
                                             "expected_reference_id": None})
    reference_value = state.reference._value.clone()
    state.post("intervene", {"capture_id": state._capture_id, "event": 0, "vector": 1, "operation": "zero"})
    original = state.store.summary()
    original_values = [value.clone() for value in state.store._values]
    comparison = state.experiment_store.summary()
    comparison_values = [value.clone() for value in state.experiment_store._values]
    weights = {key: value.clone() for key, value in state.runtime.model.state_dict().items()}
    binding, experiment_id = state._live_binding, state._experiment_id
    report = state.post("sensitivity-repeat", request(state))
    assert report["status"] == "complete" and report["completed_prompts"] == 2
    assert report["completed_runs"] == 6 and report["completed_model_forwards"] == 26
    assert report["model_forwards_upper_bound"] == 26
    assert [row["prompt"] for row in report["results"]] == ["hello world", "world hello"]
    for row in report["results"]:
        evidence = row["report"]
        assert "capture_id" not in evidence and evidence["source"].startswith("temporary")
        assert evidence["anchor"]["token_position"] == len(evidence["anchor"]["input_token_ids"]) - 1
        assert all(item["status"] == "ok" and all(item["controls"].values()) for item in evidence["results"])
    assert state.store.summary() == original and state.experiment_store.summary() == comparison
    assert state._live_binding is binding is state.runtime.capture_binding()
    assert state._experiment_id == experiment_id
    assert state._reference_id == reference["reference_id"] and torch.equal(state.reference._value, reference_value)
    assert all(torch.equal(a, b) for a, b in zip(original_values, state.store._values))
    assert all(torch.equal(a, b) for a, b in zip(comparison_values, state.experiment_store._values))
    assert all(torch.equal(value, state.runtime.model.state_dict()[key]) for key, value in weights.items())


@pytest.mark.parametrize("changes", [
    {"prompts": []}, {"prompts": ["one"]}, {"prompts": ["x"] * 5},
    {"prompts": ["x", " "]}, {"prompts": ["x", "y" * 513]},
    {"layers": [0, 1], "prompts": ["x"] * 3}, {"sites": []},
    {"capture_id": "stale"}, {"unknown": 1},
])
def test_invalid_request_does_not_run(state, monkeypatch, changes):
    monkeypatch.setattr(state.runtime, "run", lambda *a, **k: pytest.fail("must reject before any forward"))
    with pytest.raises(ValueError):
        state.post("sensitivity-repeat", request(state, **changes))


def test_long_input_is_not_truncated_and_stops_following_prompt(state):
    report = state.post("sensitivity-repeat", request(state, prompts=["hello " * 70, "hello world"]))
    assert report["status"] == "inconclusive" and report["completed_runs"] == 0
    assert report["results"][0]["status"] == "failed"
    assert "max_input_tokens" in report["results"][0]["reason"]
    assert report["results"][1]["status"] == "skipped"
    assert state.runtime.capture_binding() is state._live_binding


def test_forward_failure_quarantines_owner_and_skips_remaining_prompt(state, monkeypatch):
    original = state.store.summary()
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic forward failure")
    monkeypatch.setattr(state.runtime.model, "forward", failed)
    report = state.post("sensitivity-repeat", request(state))
    assert report["status"] == "inconclusive"
    assert [row["status"] for row in report["results"]] == ["failed", "skipped"]
    assert state.runtime.info()["state"] == "quarantined"
    assert state.store.summary() == original


def test_failed_control_quarantines_and_skips_remaining_work(state, monkeypatch):
    import looking_glass.app_interventions as module
    execute, original = module.execute, state.runtime.model.forward
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
    report = state.post("sensitivity-repeat", request(state))
    assert report["status"] == "inconclusive" and report["completed_runs"] == 0
    assert [row["status"] for row in report["results"]] == ["inconclusive", "skipped"]
    assert [row["status"] for row in report["results"][0]["report"]["results"]] == ["failed", "skipped", "skipped"]
    assert state.runtime.info()["state"] == "quarantined"


def test_retained_payloads_count_against_temporary_allowance(state, monkeypatch):
    state.post("pin-reference", {"capture_id": state._capture_id, "event": 0, "vector": 1,
                                 "expected_reference_id": None})
    state.post("intervene", {"capture_id": state._capture_id, "event": 0, "vector": 1, "operation": "zero"})
    retained = state.store.summary()["total_bytes"] + state.reference.total_bytes + state.experiment_store.summary()["total_bytes"]
    state.max_trace_bytes = retained + 1
    monkeypatch.setattr(state.runtime, "run", lambda *a, **k: pytest.fail("must reject before any forward"))
    with pytest.raises(ValueError, match="allowance"):
        state.post("sensitivity-repeat", request(state))


def test_repeated_llama_static_rope_controls(tmp_path):
    from looking_glass.app import AppState
    from looking_glass.app_runtime import LocalModelRuntime
    from test_app_llama_http import LLAMA3_SCALING, llama_checkpoint
    checkpoint = tmp_path / "llama"
    llama_checkpoint(checkpoint, 2, LLAMA3_SCALING)
    owner = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports",
                     download_root=tmp_path / "downloads")
    owner.post("load", {"path": str(checkpoint), "device": "cpu"})
    capture = owner.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    report = owner.post("sensitivity-repeat", {"capture_id": capture["capture_id"], "layers": [0],
        "sites": ["attention", "mlp", "block"], "prompts": ["hello", "world"]})
    assert report["status"] == "complete" and report["completed_runs"] == 6
    for prompt in report["results"]:
        assert all(row["controls"]["static_rotary_state_unchanged"]
                   and row["controls"]["fixed_input_ids_mask_unchanged"]
                   for row in prompt["report"]["results"])


def test_repeated_report_preserves_truncated_final_token_label(state, monkeypatch):
    import looking_glass.app_runtime as module
    original = module._token_labels
    def truncated(tokenizer, steps):
        labels, note = original(tokenizer, steps)
        for label in labels.values():
            label["truncated"] = True
        return labels, note
    monkeypatch.setattr(module, "_token_labels", truncated)
    report = state.post("sensitivity-repeat", request(state, sites=["attention"]))
    assert report["status"] == "complete"
    assert all(item["report"]["anchor"]["token_label_truncated"] for item in report["results"])


def test_demo_warning_and_provenance_survive_both_json_reports(tmp_path):
    import json
    from looking_glass.app import AppState
    from looking_glass.app_runtime import LocalModelRuntime
    owner = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports",
                     download_root=tmp_path / "downloads")
    owner.post("demo", {})
    capture = owner.post("run", {"prompt": "Hi", "settings": {"max_new_tokens": 1}})
    one = owner.post("sensitivity", {"capture_id": capture["capture_id"], "event": 0, "vector": 0,
                                   "layers": [0], "sites": ["attention"]})
    repeat = owner.post("sensitivity-repeat", {"capture_id": capture["capture_id"], "layers": [0],
                        "sites": ["attention"], "prompts": ["Hi", "Bye"]})
    for report in [one, repeat, *(item["report"] for item in repeat["results"])]:
        saved = json.loads(json.dumps(report))
        assert saved["model"]["provenance"]["kind"] == "synthetic_untrained_demo"
        assert "untrained" in saved["model"]["warning"]
        assert saved["model"]["architecture"] == "GPT2LMHeadModel"
        assert saved["model"]["config_summary"]["n_layer"] == 2
        assert "cryptographic" in saved["model"]["identity_note"]
        assert "path" not in saved["model"]


def test_loopback_route_requires_token_and_current_capture(state):
    import json
    import threading
    from looking_glass.app import create_server
    from test_app_http import call
    server = create_server(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        endpoint = (state, server)
        assert call(endpoint, "/api/sensitivity-repeat", "POST", request(state), token=False)[0] == 403
        assert call(endpoint, "/api/sensitivity-repeat", "POST", request(state, capture_id="stale"))[0] == 400
        status, body, _ = call(endpoint, "/api/sensitivity-repeat", "POST", request(state, sites=["attention"]))
        assert status == 200, body
        assert json.loads(body)["status"] == "complete"
    finally:
        server.shutdown()
        worker.join(5)
        server.server_close()
        assert not worker.is_alive()
