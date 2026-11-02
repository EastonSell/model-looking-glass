"""Two-phase reconstruction tests use fabricated models and token IDs only."""
import hashlib
import json

import numpy as np
import pytest
import torch

from test_targeted_edit_projected_development import guarded_model
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_projected_development as development
from looking_glass import targeted_edit_projected_execution as execution
from looking_glass import targeted_edit_projected_math as projected


def prepared(tmp_path):
    _model, guard = guarded_model()
    directory = tmp_path / "development"
    candidate = development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, directory)
    digest = hashlib.sha256((directory / "pre-evaluation-candidate.json").read_bytes()).hexdigest()
    return guard, directory, candidate, digest


def test_reconstructs_saved_controls_before_evaluation(tmp_path):
    guard, directory, candidate, digest = prepared(tmp_path)
    handoff = execution.reconstruct(guard, directory, digest, (3,), ((1,), (2,)), 3, 1)
    assert handoff.candidate == candidate
    assert set(handoff.updates) == set(projected.CONDITIONS)
    for name, delta in handoff.updates.items():
        assert v1._tensor_sha(delta) == candidate["construction"]["conditions"][name]["intended_delta_sha256"]
    guard.verify()


def test_altered_saved_artifact_stops_before_model_forward(tmp_path, monkeypatch):
    guard, directory, _candidate, digest = prepared(tmp_path)
    artifact = directory / "full-gradient.f32"
    data = bytearray(artifact.read_bytes())
    data[0] ^= 1
    artifact.write_bytes(data)
    calls = []
    monkeypatch.setattr(guard.model, "forward", lambda **kw: calls.append(kw))
    with pytest.raises(execution.ExecutionError, match="artifact differs"):
        execution.reconstruct(guard, directory, digest, (3,), ((1,), (2,)), 3, 1)
    assert calls == [] and guard.poisoned


def test_wrong_candidate_identity_stops_before_model_forward(tmp_path, monkeypatch):
    guard, directory, _candidate, _digest = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(guard.model, "forward", lambda **kw: calls.append(kw))
    with pytest.raises(execution.ExecutionError, match="digest differs"):
        execution.reconstruct(guard, directory, "0" * 64, (3,), ((1,), (2,)), 3, 1)
    assert calls == [] and guard.poisoned


def test_evaluation_records_each_condition_and_restores(tmp_path, monkeypatch):
    guard, directory, _candidate, digest = prepared(tmp_path)
    handoff = execution.reconstruct(guard, directory, digest, (3,), ((1,), (2,)), 3, 1)
    cases = (("a", "target", "a", (1,), None), ("b", "locality", "b", (2,), ("x", 1)))
    monkeypatch.setattr(execution.score, "_cases", lambda _document: cases)
    monkeypatch.setattr(execution.score, "evaluate", lambda matrices, _document, _candidate: {
        "conditions": list(matrices), "provisional_numerical_decision": "negative"})
    def fake_row(model, ids):
        with torch.no_grad():
            tokens = torch.tensor([ids], dtype=torch.int64)
            values = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits[0, -1]
            row = np.zeros(v1.VOCAB_SIZE, dtype="<f4")
            row[:values.numel()] = values.numpy()
            return row.tobytes()
    monkeypatch.setattr(v1, "_logit_row", fake_row)
    result = execution.evaluate(handoff, b"synthetic", tmp_path / "evaluation")
    assert result["conditions"] == list(execution.score.CONDITIONS)
    observations = [json.loads(line) for line in (tmp_path / "evaluation" / "observations.jsonl").read_bytes().splitlines()]
    assert [(row["condition"], row["case_id"]) for row in observations] == [
        (condition, case[0]) for condition in execution.score.CONDITIONS for case in cases]
    for condition in execution.score.CONDITIONS:
        state = json.loads((tmp_path / "evaluation" / ("state-" + condition + ".json")).read_bytes())
        assert state["cleanup"] == "verified"
        assert state["weight_sha256"] == guard.weight_sha
        assert state["addition_denormal_probe_before"] == state["addition_denormal_probe_after"]
    guard.verify()
    with pytest.raises(execution.ExecutionError, match="fresh reconstructed"):
        execution.evaluate(handoff, b"synthetic", tmp_path / "again")


def test_forged_update_cannot_produce_report(tmp_path, monkeypatch):
    guard, directory, _candidate, digest = prepared(tmp_path)
    handoff = execution.reconstruct(guard, directory, digest, (3,), ((1,), (2,)), 3, 1)
    handoff.updates["projected"].add_(1)
    calls = []
    monkeypatch.setattr(guard.model, "forward", lambda **kw: calls.append(kw))
    monkeypatch.setattr(execution.score, "_cases", lambda _document: (("a", "target", "a", (1,), None),))
    monkeypatch.setattr(v1, "_logit_row", lambda _model, _ids: np.zeros(v1.VOCAB_SIZE, dtype="<f4").tobytes())
    with pytest.raises(execution.ExecutionError, match="intended update"):
        execution.evaluate(handoff, b"synthetic", tmp_path / "failure")
    assert guard.poisoned and calls == []
    assert not (tmp_path / "failure" / "provisional-report.json").exists()


def test_direct_evaluation_without_reconstruction_is_rejected(tmp_path):
    with pytest.raises(execution.ExecutionError, match="fresh reconstructed"):
        execution.evaluate(object(), b"synthetic", tmp_path / "failure")
    assert not (tmp_path / "failure").exists()


def test_rehashed_forged_scalar_is_rejected(tmp_path):
    guard, directory, _candidate, _digest = prepared(tmp_path)
    path = directory / "pre-evaluation-candidate.json"
    candidate = json.loads(path.read_bytes())
    candidate["calibration_shifts"]["projected"]["max_output_abs"] += 1
    raw = v1._canonical(candidate)
    path.write_bytes(raw)
    with pytest.raises(execution.ExecutionError, match="calibration measurement differs"):
        execution.reconstruct(guard, directory, hashlib.sha256(raw).hexdigest(), (3,), ((1,), (2,)), 3, 1)
    assert guard.poisoned


def test_rehashed_forged_discovery_change_is_rejected(tmp_path):
    guard, directory, _candidate, _digest = prepared(tmp_path)
    path = directory / "pre-evaluation-candidate.json"
    candidate = json.loads(path.read_bytes())
    candidate["discovery_changes"]["projected"]["change"] += 1
    raw = v1._canonical(candidate)
    path.write_bytes(raw)
    with pytest.raises(execution.ExecutionError, match="discovery measurement differs"):
        execution.reconstruct(guard, directory, hashlib.sha256(raw).hexdigest(), (3,), ((1,), (2,)), 3, 1)
    assert guard.poisoned


def test_rounded_away_delta_mutation_is_rejected_before_rows(tmp_path, monkeypatch):
    guard, directory, _candidate, digest = prepared(tmp_path)
    handoff = execution.reconstruct(guard, directory, digest, (3,), ((1,), (2,)), 3, 1)
    delta = handoff.updates["projected"]
    original_applied = v1._tensor_sha(guard.original + delta)
    # Find a one-ULP mutation whose addition rounds to the same float32 weight.
    found = False
    for index in range(delta.numel()):
        flat = delta.view(-1)
        old = flat[index].clone()
        flat[index] = torch.nextafter(old, torch.tensor(float("inf")))
        if v1._tensor_sha(guard.original + delta) == original_applied:
            found = True
            break
        flat[index] = old
    assert found
    calls = []
    monkeypatch.setattr(guard.model, "forward", lambda **kw: calls.append(kw))
    with pytest.raises(execution.ExecutionError, match="intended update"):
        execution.evaluate(handoff, b"synthetic", tmp_path / "failure")
    assert calls == [] and guard.poisoned
