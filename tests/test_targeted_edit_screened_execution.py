import hashlib
import json

import numpy as np
import pytest

from test_targeted_edit_projected_development import guarded_model
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_screened_development as development
from looking_glass import targeted_edit_screened_execution as execution
from looking_glass import targeted_edit_screened_screen as screen


def fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 5)
    monkeypatch.setattr(v1, "LONDON", 3)
    monkeypatch.setattr(v1, "PARIS", 1)
    targets = [{"case_id": f"t{i}", "role": "target", "stratum": f"s{i // 8 + 1}",
                "token_ids": (3,), "prompt": f"target {i}"} for i in range(32)]
    localities = [{"case_id": f"l{i}", "role": "locality", "stratum": f"l{i // 4}",
                   "token_ids": [2], "prompt": f"locality {i}",
                   "continuations": [{"text": " x", "ids": [2]}]} for i in range(12)]
    document = json.dumps({"cases": [{**row, "token_ids": list(row["token_ids"])} for row in targets]
                           + localities}).encode()
    monkeypatch.setattr(screen, "CASE_SHA256", hashlib.sha256(document).hexdigest())
    monkeypatch.setattr(screen, "_targets", lambda _doc: targets)
    model, guard = guarded_model()
    candidate_dir = tmp_path / "candidate"
    development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, candidate_dir)
    candidate_sha = hashlib.sha256((candidate_dir / "pre-evaluation-candidate.json").read_bytes()).hexdigest()
    calls = []
    def fake_row(_model, ids):
        calls.append(tuple(ids))
        return np.array([0., 1., 0., 0., -1.], dtype="<f4").tobytes()
    monkeypatch.setattr(v1, "_logit_row", fake_row)
    screen_dir = tmp_path / "screen"
    binding = {"source_commit": "a" * 40, "manifest_sha256": "b" * 64,
               "development_record_sha256": "c" * 64, "candidate_sha256": candidate_sha}
    screen.screen(guard, document, binding, screen_dir)
    screen_sha = hashlib.sha256((screen_dir / "screen-selection.json").read_bytes()).hexdigest()
    calls.clear()
    return model, guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document, calls


def test_evaluation_reuses_screen_rows_and_applies_only_frozen_conditions(monkeypatch, tmp_path):
    _model, guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document, calls = fixture(monkeypatch, tmp_path)
    prepared = execution.reconstruct(guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document)
    result = execution.evaluate(prepared, tmp_path / "evaluation")
    conditions = 5 + int(prepared.match["available"] and not prepared.match["shared_equal_norm_arm"])
    assert len(calls) == 12 + 20 * conditions
    assert result["selected_count"] == 8
    raw = (tmp_path / "evaluation/original.f32").read_bytes()
    saved = (screen_dir / "screen.f32").read_bytes()
    row_bytes = 5 * 4
    assert raw[:8 * row_bytes] == b"".join(saved[i * row_bytes:(i + 1) * row_bytes] for i in prepared.offsets)
    journal = [json.loads(line) for line in (tmp_path / "evaluation/observations.jsonl").read_bytes().splitlines()]
    assert [r["source"] for r in journal[:8]] == ["screen"] * 8
    assert all(r["source"] == "forward" for r in journal[8:])
    assert not guard.poisoned
    with pytest.raises(execution.ExecutionError, match="fresh"):
        execution.evaluate(prepared, tmp_path / "second")


def test_mutated_control_fences_before_any_new_forward(monkeypatch, tmp_path):
    _model, guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document, calls = fixture(monkeypatch, tmp_path)
    prepared = execution.reconstruct(guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document)
    prepared.updates["nuisance"][0, 0] += .01
    with pytest.raises(execution.ExecutionError, match="intended update"):
        execution.evaluate(prepared, tmp_path / "evaluation")
    assert calls == [] and guard.poisoned


def test_missing_control_fences_before_any_new_forward(monkeypatch, tmp_path):
    _model, guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document, calls = fixture(monkeypatch, tmp_path)
    prepared = execution.reconstruct(guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document)
    del prepared.updates["nuisance"]
    with pytest.raises(execution.ExecutionError, match="handoff changed"):
        execution.evaluate(prepared, tmp_path / "evaluation")
    assert calls == [] and guard.poisoned


def test_shared_match_reuses_equal_norm_rows_without_extra_forward(monkeypatch, tmp_path):
    _model, guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document, calls = fixture(monkeypatch, tmp_path)
    original = execution.reconstruct(guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document)
    updates = {k: v for k, v in original.updates.items() if k != "matched"}
    attestations = {k: v for k, v in original.attestations.items() if k != "matched"}
    prepared = execution._Prepared(guard, original.candidate, updates, attestations, document,
                                   original.offsets, original.baseline,
                                   {"available": True, "shared_equal_norm_arm": True})
    execution.evaluate(prepared, tmp_path / "evaluation")
    assert len(calls) == 12 + 20 * 5
    assert (tmp_path / "evaluation/matched.f32").read_bytes() == (tmp_path / "evaluation/unprojected.f32").read_bytes()
    journal = [json.loads(line) for line in (tmp_path / "evaluation/observations.jsonl").read_bytes().splitlines()]
    assert [row["source"] for row in journal[-20:]] == ["unprojected"] * 20


def test_changed_screen_row_fences_before_reconstruction_or_forward(monkeypatch, tmp_path):
    _model, guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document, calls = fixture(monkeypatch, tmp_path)
    path = screen_dir / "screen.f32"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(execution.ExecutionError, match="screen artifact"):
        execution.reconstruct(guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document)
    assert calls == [] and guard.poisoned
