import hashlib
import json

import numpy as np
import pytest

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import answer_format_feasibility as numerical
from looking_glass import answer_format_feasibility_worker as worker


class Guard:
    def __init__(self):
        self.model = object()
        self.weight_sha = "a" * 64
        self.state_sha = "b" * 64
        self.buffer_sha = "c" * 64
        self.runtime_settings = {"synthetic": True}
        self.last_cleanup = "not_attempted"
        self.poisoned = False

    def verify(self):
        if self.poisoned:
            raise RuntimeError("poisoned")

    def condition(self, delta, applied, callback):
        assert delta is None and applied == self.weight_sha
        callback()
        self.last_cleanup = "verified"
        self.poisoned = True

    def commit_condition(self):
        assert self.poisoned and self.last_cleanup == "verified"
        self.poisoned = False


@pytest.fixture
def fake_cases(monkeypatch):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 6)
    monkeypatch.setattr(v1, "_denormal_probe", lambda: {"synthetic": True})
    rows = []
    for stage in numerical.SEEDS:
        for fact in numerical.FACTS:
            for family in numerical.FAMILIES:
                for prefix in numerical.PREFIXES:
                    ids = [1, len(rows) % 6]
                    labels = {name: {"token_id": i, "full_token_ids": ids + [i]}
                              for i, name in enumerate(numerical.LABELS)}
                    rows.append({"case_id": f"{stage}-{fact}-{family}-{prefix}", "role": stage,
                                 "fact_id": fact, "family": family, "prefix": prefix,
                                 "prefix_text": "", "suffix_text": "fixture", "prompt": "fixture",
                                 "prefix_token_ids": [], "suffix_token_ids": ids, "token_ids": ids,
                                 "labels": labels})
    doc = {"protocol_sha256": worker.PROTOCOL_SHA, "scope_sha256": worker.SCOPE_SHA,
           "model_forward": False, "seeds": numerical.SEEDS,
           "execution_permutations": {s: numerical.execution_order(s) for s in numerical.SEEDS},
           "cases": rows}
    raw = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()
    monkeypatch.setattr(worker, "CASE_SHA", hashlib.sha256(raw).hexdigest())
    return raw, rows, {"source_commit": "d" * 40, "manifest_sha256": "e" * 64}


def test_both_phases_save_shuffled_rows_and_replay_development(fake_cases, tmp_path, monkeypatch):
    raw, rows, identity = fake_cases
    calls = []
    def logit(_model, ids):
        calls.append(ids)
        return np.array([4, 0, 1, 2, -1, -2], dtype="<f4").tobytes()
    monkeypatch.setattr(v1, "_logit_row", logit)
    dev_dir = tmp_path / "development"
    dev = worker.run(Guard(), raw, identity, "development", dev_dir)
    expected = [tuple(rows[i]["token_ids"]) for i in numerical.execution_order("development")]
    assert calls == expected and dev["score"]["selected_family"] == "statement"
    assert len((dev_dir / "journal.jsonl").read_bytes().splitlines()) == 32
    matrix, result = (dev_dir / "matrix.f32").read_bytes(), (dev_dir / "results.json").read_bytes()
    prior = {"matrix": matrix, "result": result,
             "matrix_sha256": worker.sha(matrix), "result_sha256": worker.sha(result)}
    calls.clear()
    held = worker.run(Guard(), raw, identity, "heldout", tmp_path / "heldout", prior=prior)
    assert calls == [tuple(rows[32 + i]["token_ids"]) for i in numerical.execution_order("heldout")]
    assert held["score"]["status"] == "positive" and held["score"]["selected_family"] == "statement"
    assert held["development_matrix_sha256"] == prior["matrix_sha256"]


def test_rehashed_forged_development_selection_fences_before_forward(fake_cases, tmp_path, monkeypatch):
    raw, _rows, identity = fake_cases
    monkeypatch.setattr(v1, "_logit_row", lambda *_: np.array([4, 0, 1, 2, -1, -2], dtype="<f4").tobytes())
    dev_dir = tmp_path / "development"
    worker.run(Guard(), raw, identity, "development", dev_dir)
    matrix = (dev_dir / "matrix.f32").read_bytes()
    saved = json.loads((dev_dir / "results.json").read_bytes())
    saved["score"]["families"][0]["robust_facts"] = []
    saved["score"]["families"][0]["eligible"] = False
    saved["score"]["selected_family"] = "question"
    forged = json.dumps(saved, separators=(",", ":"), sort_keys=True).encode()
    prior = {"matrix": matrix, "result": forged,
             "matrix_sha256": worker.sha(matrix), "result_sha256": worker.sha(forged)}
    monkeypatch.setattr(v1, "_logit_row", lambda *_: pytest.fail("held-out forward should be fenced"))
    guard = Guard()
    with pytest.raises(worker.StudyError, match="does not replay"):
        worker.run(guard, raw, identity, "heldout", tmp_path / "bad", prior=prior)
    assert guard.poisoned and not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ("boolean_as_integer", "duplicate_key"))
def test_rehashed_typed_or_duplicate_json_fences_before_heldout(fake_cases, tmp_path, monkeypatch, kind):
    raw, _rows, identity = fake_cases
    monkeypatch.setattr(v1, "_logit_row", lambda *_: np.array([4, 0, 1, 2, -1, -2], dtype="<f4").tobytes())
    dev_dir = tmp_path / "development"
    worker.run(Guard(), raw, identity, "development", dev_dir)
    matrix, original = (dev_dir / "matrix.f32").read_bytes(), (dev_dir / "results.json").read_bytes()
    if kind == "boolean_as_integer":
        saved = json.loads(original)
        saved["score"]["families"][0]["eligible"] = 1
        changed = json.dumps(saved, separators=(",", ":"), sort_keys=True).encode()
    else:
        changed = b'{"stage":"development",' + original[1:]
    prior = {"matrix": matrix, "result": changed,
             "matrix_sha256": worker.sha(matrix), "result_sha256": worker.sha(changed)}
    monkeypatch.setattr(v1, "_logit_row", lambda *_: pytest.fail("held-out forward should be fenced"))
    guard = Guard()
    with pytest.raises(worker.StudyError):
        worker.run(guard, raw, identity, "heldout", tmp_path / "bad", prior=prior)
    assert guard.poisoned and not (tmp_path / "bad").exists()


def test_partial_forward_is_preserved_and_no_result_is_committed(fake_cases, tmp_path, monkeypatch):
    raw, _rows, identity = fake_cases
    calls = []
    def interrupted(_model, ids):
        calls.append(ids)
        if len(calls) == 9:
            raise KeyboardInterrupt("fabricated interruption")
        return np.array([4, 0, 1, 2, -1, -2], dtype="<f4").tobytes()
    monkeypatch.setattr(v1, "_logit_row", interrupted)
    guard, output = Guard(), tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        worker.run(guard, raw, identity, "development", output)
    assert guard.poisoned and len((output / "journal.jsonl").read_bytes().splitlines()) == 8
    assert not (output / "results.json").exists()


def test_scoring_denormal_drift_prevents_complete_result(fake_cases, tmp_path, monkeypatch):
    raw, _rows, identity = fake_cases
    monkeypatch.setattr(v1, "_logit_row", lambda *_: np.array([4, 0, 1, 2, -1, -2], dtype="<f4").tobytes())
    drift = {"active": False}
    original = numerical.summarize
    def score(*args, **kwargs):
        result = original(*args, **kwargs)
        drift["active"] = True
        return result
    def probe():
        if drift["active"]:
            raise RuntimeError("fabricated denormal drift")
        return {"synthetic": True}
    monkeypatch.setattr(numerical, "summarize", score)
    monkeypatch.setattr(v1, "_denormal_probe", probe)
    guard, output = Guard(), tmp_path / "drift"
    with pytest.raises(RuntimeError, match="denormal drift"):
        worker.run(guard, raw, identity, "development", output)
    assert guard.poisoned and not (output / "results.json").exists()

