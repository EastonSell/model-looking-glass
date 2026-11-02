"""Synthetic screening writer tests; no checkpoint or public pool forward."""

import hashlib
import json

import numpy as np
import pytest

from test_targeted_edit_projected_development import guarded_model
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_screened_screen as screen
from looking_glass import targeted_edit_screened_selection as selector


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 5)
    monkeypatch.setattr(v1, "LONDON", 3)
    monkeypatch.setattr(v1, "PARIS", 1)
    targets = [{"case_id": f"s{group}-{index}", "role": "target", "stratum": f"s{group}",
                "token_ids": [1 + (index % 4)]}
               for group in range(1, 5) for index in range(8)]
    cases = {"cases": targets + [{"case_id": f"locality-{i}", "role": "locality"} for i in range(12)]}
    data = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    monkeypatch.setattr(screen, "CASE_SHA256", hashlib.sha256(data).hexdigest())
    binding = {"source_commit": "a" * 40, "manifest_sha256": "b" * 64,
               "development_record_sha256": "c" * 64, "candidate_sha256": "d" * 64}
    return data, targets, binding


def test_screen_saves_exact_original_rows_and_recomputable_sign_selection(fixture, tmp_path):
    case_bytes, targets, binding = fixture
    model, guard = guarded_model()
    calls = []
    forward = model.forward

    def record(**kwargs):
        calls.append(tuple(kwargs["input_ids"][0].tolist()))
        return forward(**kwargs)

    model.forward = record
    output = tmp_path / "screen"
    result = screen.screen(guard, case_bytes, binding, output)
    assert calls == [tuple(row["token_ids"]) for row in targets]
    matrix = (output / "screen.f32").read_bytes()
    assert len(matrix) == 32 * 5 * 4
    rows = np.frombuffer(matrix, dtype="<f4").reshape(32, 5).astype(np.float64)
    expected = selector.select_screen(targets, [float(row[3] - row[1]) for row in rows])
    for row in result["selection"]["rows"]:
        expected_row = expected["rows"][row["offset"]]
        assert row["baseline_margin"] == expected_row["baseline_margin"]
        assert row["eligible"] == expected_row["eligible"] and row["selected"] == expected_row["selected"]
        start = row["logit_byte_offset"]
        assert row["logit_sha256"] == hashlib.sha256(matrix[start:start + row["logit_bytes"]]).hexdigest()
    assert result["selection"]["selected_offsets"] == expected["selected_offsets"]
    assert len((output / "observations.jsonl").read_bytes().splitlines()) == 32
    assert json.loads((output / "screen-selection.json").read_bytes()) == result
    assert not guard.poisoned
    guard.verify()


def test_wrong_public_bytes_fence_before_any_forward(fixture, tmp_path):
    case_bytes, _targets, binding = fixture
    model, guard = guarded_model()
    model.forward = lambda **_kwargs: pytest.fail("unexpected model forward")
    with pytest.raises(screen.ScreenError, match="case bytes"):
        screen.screen(guard, case_bytes + b" ", binding, tmp_path / "bad")
    assert guard.poisoned and not (tmp_path / "bad").exists()


def test_interrupted_forward_leaves_only_partial_screen_and_fences(fixture, tmp_path):
    case_bytes, _targets, binding = fixture
    model, guard = guarded_model()
    forward = model.forward
    count = 0

    def interrupted(**kwargs):
        nonlocal count
        count += 1
        if count == 12:
            raise KeyboardInterrupt("synthetic screen interruption")
        return forward(**kwargs)

    model.forward = interrupted
    output = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        screen.screen(guard, case_bytes, binding, output)
    assert guard.poisoned
    assert len((output / "observations.jsonl").read_bytes().splitlines()) == 11
    assert not (output / "screen-selection.json").exists()
