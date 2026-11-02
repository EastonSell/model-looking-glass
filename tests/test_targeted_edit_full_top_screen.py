"""Fabricated writer tests; no checkpoint or study prompt is forwarded."""

import hashlib
import json

import numpy as np
import pytest

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_full_top_screen as screen


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
def fixture(monkeypatch):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 5)
    monkeypatch.setattr(v1, "_denormal_probe", lambda: {"synthetic": True})
    rows = []
    for fact_id in screen.FACTS:
        for index, role in enumerate(screen.ROLES):
            ids = [1 + index % 3]
            rows.append({"case_id": f"{fact_id}-{index}", "fact_id": fact_id, "role": role,
                         "token_ids": ids, "factual": {"token_id": 1, "full_token_ids": ids + [1]},
                         "counterfactual": {"token_id": 2, "full_token_ids": ids + [2]}})
    rows.extend({"case_id": f"locality-{i}", "role": "locality"} for i in range(12))
    doc = {"protocol_sha256": screen.PROTOCOL_SHA256, "model_forward": False,
           "fact_order": list(screen.FACTS), "cases": rows}
    raw = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()
    monkeypatch.setattr(screen, "CASE_SHA256", hashlib.sha256(raw).hexdigest())
    identity = {"source_commit": "d" * 40, "manifest_sha256": "e" * 64}
    return raw, rows, identity


def test_writer_saves_all_rows_and_the_fixed_selection(fixture, tmp_path, monkeypatch):
    raw, rows, identity = fixture
    calls = []

    def logit(_model, ids):
        calls.append(ids)
        return np.array([0, 3, -1, -2, -3], dtype="<f4").tobytes()

    monkeypatch.setattr(v1, "_logit_row", logit)
    guard = Guard()
    output = tmp_path / "screen"
    result = screen.screen(guard, raw, identity, output)
    assert calls == [tuple(row["token_ids"]) for row in rows[:36]]
    assert result["selection"]["fact_index"] == 0
    assert result["selection"]["evaluation_offsets"] == [3, 4, 5, 6]
    matrix = (output / "screen.f32").read_bytes()
    assert len(matrix) == 36 * 5 * 4
    assert len((output / "observations.jsonl").read_bytes().splitlines()) == 36
    for row in result["selection"]["observations"]:
        offset = row["logit_byte_offset"]
        assert row["logit_sha256"] == hashlib.sha256(matrix[offset:offset + row["logit_bytes"]]).hexdigest()
    assert json.loads((output / "screen-selection.json").read_bytes()) == result
    assert not guard.poisoned


def test_case_mismatch_fences_before_forward(fixture, tmp_path, monkeypatch):
    raw, _rows, identity = fixture
    monkeypatch.setattr(v1, "_logit_row", lambda *_args: pytest.fail("unexpected forward"))
    guard = Guard()
    with pytest.raises(screen.ScreenError, match="case bytes"):
        screen.screen(guard, raw + b" ", identity, tmp_path / "bad")
    assert guard.poisoned and not (tmp_path / "bad").exists()


def test_interruption_keeps_partial_journal_without_selection(fixture, tmp_path, monkeypatch):
    raw, _rows, identity = fixture
    calls = []

    def logit(_model, ids):
        calls.append(ids)
        if len(calls) == 12:
            raise KeyboardInterrupt("fabricated interruption")
        return np.array([0, 3, -1, -2, -3], dtype="<f4").tobytes()

    monkeypatch.setattr(v1, "_logit_row", logit)
    guard = Guard()
    output = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt, match="fabricated"):
        screen.screen(guard, raw, identity, output)
    assert guard.poisoned
    assert len((output / "observations.jsonl").read_bytes().splitlines()) == 11
    assert not (output / "screen-selection.json").exists()


def test_denormal_drift_during_selection_never_completes_record(fixture, tmp_path, monkeypatch):
    raw, _rows, identity = fixture
    monkeypatch.setattr(v1, "_logit_row", lambda *_args: np.array([0, 3, -1, -2, -3], dtype="<f4").tobytes())
    drift = {"active": False}
    original_select = screen.numerical.select_screen

    def probe():
        if drift["active"]:
            raise v1.StudyError("fabricated denormal drift")
        return {"synthetic": True}

    def select(*args):
        result = original_select(*args)
        drift["active"] = True
        return result

    monkeypatch.setattr(v1, "_denormal_probe", probe)
    monkeypatch.setattr(screen.numerical, "select_screen", select)
    guard = Guard()
    output = tmp_path / "partial-score"
    with pytest.raises(v1.StudyError, match="denormal drift"):
        screen.screen(guard, raw, identity, output)
    assert guard.poisoned
    assert len((output / "observations.jsonl").read_bytes().splitlines()) == 36
    assert not (output / "screen-selection.json").exists()
