"""Actual donor-vector transfer with independent direct-hook evidence."""
import copy
import pytest
import torch
from test_app_sensitivity import state


def pair(state, **changes):
    data = {"capture_id": state._capture_id, "donor_prompt": "hello world", "receiver_prompt": "world hello"}
    preview = state.post("patch-preview", data)
    donor, receiver = preview["donor"], preview["receiver"]
    return {**data, "donor_position": 0, "receiver_position": 1,
            "donor_token_ids": donor["token_ids"], "receiver_token_ids": receiver["token_ids"],
            "donor_token_id": donor["token_ids"][0], "receiver_token_id": receiver["token_ids"][1],
            "layer": 0, "site": "attention", **changes}


def test_transfer_matches_independent_direct_hook_and_preserves_original(state):
    request = pair(state)
    original = state.store.summary()
    before_values = [x.clone() for x in state.store._values]
    weights = {k: v.clone() for k, v in state.runtime.model.state_dict().items()}
    result = state.post("patch-cross-prompt", request)
    assert result["status"] == "complete" and all(result["controls"].values())
    assert result["total_model_forwards"] == 10
    assert result["donor"]["position"] == 0 and result["receiver"]["position"] == 1
    assert state.store.summary() == original and all(torch.equal(a, b) for a, b in zip(before_values, state.store._values))
    assert all(torch.equal(v, state.runtime.model.state_dict()[k]) for k, v in weights.items())
    model = state.runtime.model
    donor_ids = torch.tensor([request["donor_token_ids"]]); receiver_ids = torch.tensor([request["receiver_token_ids"]])
    captured = []
    def capture(_m, _i, output):
        captured.append(output[0][0, 0].detach().clone())
    hook = model.transformer.h[0].attn.register_forward_hook(capture)
    try:
        with torch.inference_mode(): clean = model(input_ids=donor_ids, use_cache=False).logits[0, -1].double()
    finally: hook.remove()
    with torch.inference_mode(): baseline = model(input_ids=receiver_ids, use_cache=False).logits[0, -1].double()
    def transfer(_m, _i, output):
        value = output[0].clone(); value[0, 1] = captured[0]; return (value, *output[1:])
    hook = model.transformer.h[0].attn.register_forward_hook(transfer)
    try:
        with torch.inference_mode(): patched = model(input_ids=receiver_ids, use_cache=False).logits[0, -1].double()
    finally: hook.remove()
    clean_id = int(clean.argmax()); p, q = baseline.softmax(-1), patched.softmax(-1)
    assert result["clean_baseline"]["token_id"] == clean_id
    assert result["clean_baseline"]["probability"] == pytest.approx(float(clean.softmax(-1)[clean_id]), abs=1e-12)
    assert result["corrupted_baseline"]["probability"] == pytest.approx(float(p[clean_id]), abs=1e-12)
    assert result["patched"]["probability"] == pytest.approx(float(q[clean_id]), abs=1e-12)
    assert result["metrics"]["total_variation"] == pytest.approx(float((p-q).abs().sum()/2), abs=1e-12)


@pytest.mark.parametrize("changes", [
    {"donor_position": True}, {"receiver_position": 99}, {"donor_token_id": 999},
    {"donor_token_ids": [999]}, {"capture_id": "stale"}, {"layer": True}, {"site": "unknown"},
])
def test_invalid_preview_or_alignment_rejects_before_forward(state, monkeypatch, changes):
    request = pair(state, **changes)
    monkeypatch.setattr(state.runtime, "run", lambda *a, **k: pytest.fail("must reject before model forwarding"))
    with pytest.raises(ValueError): state.post("patch-cross-prompt", request)


def test_watched_token_outside_top_union_is_full_vocabulary(state):
    from looking_glass.app_interventions import _next_token_distribution
    a = torch.arange(30, dtype=torch.float32).reshape(1, 1, 30)
    b = a.flip(-1).contiguous()
    distribution = _next_token_distribution(a, b, state.runtime.tokenizer, [12])
    assert 12 not in {row["token_id"] for row in distribution["rows"]}
    row = distribution["watched_tokens"][0]
    assert row["baseline_probability"] == pytest.approx(float(a[0, 0].double().softmax(-1)[12]))
    assert row["intervention_probability"] == pytest.approx(float(b[0, 0].double().softmax(-1)[12]))


def test_preview_marks_unavailable_and_truncated_token_labels(state, monkeypatch):
    tokenizer = state.runtime.tokenizer
    monkeypatch.setattr(tokenizer, "decode", lambda *_args, **_kwargs: "x" * 513)
    monkeypatch.setattr(tokenizer, "convert_ids_to_tokens", lambda *_args, **_kwargs: None)
    preview = state.post("patch-preview", {"capture_id": state._capture_id,
                         "donor_prompt": "hello world", "receiver_prompt": "world hello"})
    for side in ("donor", "receiver"):
        for token in preview[side]["tokens"]:
            assert token["text"] == "x" * 512 and token["text_truncated"]
            assert not token["text_unavailable"]
            assert token["piece"] == "" and token["piece_unavailable"]
            assert not token["piece_truncated"]


def test_preview_does_not_hide_resource_exhaustion(state, monkeypatch):
    def exhausted(*_args, **_kwargs):
        raise MemoryError("simulated allocation failure")
    monkeypatch.setattr(state.runtime.tokenizer, "decode", exhausted)
    with pytest.raises(MemoryError):
        state.post("patch-preview", {"capture_id": state._capture_id,
                   "donor_prompt": "hello world", "receiver_prompt": "world hello"})


def test_clean_baseline_preserves_missing_or_truncated_label_flags(state, monkeypatch):
    import looking_glass.app_interventions as engine
    request = pair(state)
    monkeypatch.setattr(engine, "_token_text", lambda *_args: ("x" * 512, False, True))
    report = state.post("patch-cross-prompt", request)
    assert report["clean_baseline"]["text_truncated"]
    assert not report["clean_baseline"]["text_unavailable"]
    assert len(report["clean_baseline"]["text"]) == 512


@pytest.mark.parametrize("failing_trial", [1, 2])
def test_control_failure_quarantines_and_preserves_capture(state, monkeypatch, failing_trial):
    import looking_glass.app_interventions as engine
    request = pair(state)
    original = state.store.summary()
    execute, forward = engine.execute, state.runtime.model.forward
    trials, calls = [], []
    def wrapped(owner, validated):
        trials.append(None)
        if len(trials) == failing_trial:
            def corrupt(*args, **kwargs):
                result = forward(*args, **kwargs); calls.append(None)
                if len(calls) == 2: result.logits = result.logits + 1
                return result
            monkeypatch.setattr(owner.model, "forward", corrupt)
        return execute(owner, validated)
    monkeypatch.setattr(engine, "execute", wrapped)
    with pytest.raises(ValueError, match="null control"):
        state.post("patch-cross-prompt", request)
    assert state.runtime.info()["state"] == "quarantined"
    assert state.store.summary() == original


def test_same_prompt_same_position_is_verified_noop(state):
    request = pair(state, receiver_prompt="hello world", receiver_position=0)
    # Obtain the exact receiver preview for the revised prompt.
    preview = state.post("patch-preview", {k: request[k] for k in ("capture_id", "donor_prompt", "receiver_prompt")})
    request["receiver_token_ids"] = preview["receiver"]["token_ids"]
    request["receiver_token_id"] = request["receiver_token_ids"][0]
    report = state.post("patch-cross-prompt", request)
    assert not report["metrics"]["activation_changed"] and report["metrics"]["total_variation"] == 0
    assert all(report["controls"].values())


def test_llama_cross_prompt_checks_rotary_controls(tmp_path):
    from looking_glass.app import AppState
    from looking_glass.app_runtime import LocalModelRuntime
    from test_app_llama_http import LLAMA3_SCALING, llama_checkpoint
    path = tmp_path / "llama"; llama_checkpoint(path, 2, LLAMA3_SCALING)
    owner = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports", download_root=tmp_path / "downloads")
    owner.post("load", {"path": str(path), "device": "cpu"})
    owner.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    report = owner.post("patch-cross-prompt", pair(owner))
    assert report["status"] == "complete" and all(report["controls"].values())
    assert report["controls"]["donor_static_rotary_state_unchanged"]
    assert report["controls"]["receiver_static_rotary_state_unchanged"]


def test_http_rejects_missing_token_and_stale_preview(state):
    import threading
    from looking_glass.app import create_server
    from test_app_http import call
    server = create_server(state); worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        endpoint = (state, server); request = pair(state)
        assert call(endpoint, "/api/patch-cross-prompt", "POST", request, token=False)[0] == 403
        assert call(endpoint, "/api/patch-cross-prompt", "POST", {**request, "donor_token_id": 999})[0] == 400
        assert call(endpoint, "/api/patch-preview", "POST", {"capture_id": "stale", "donor_prompt": "x", "receiver_prompt": "y"})[0] == 400
    finally:
        server.shutdown(); worker.join(5); server.server_close()
        assert not worker.is_alive()
