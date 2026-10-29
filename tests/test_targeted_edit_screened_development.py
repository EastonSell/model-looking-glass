"""Discovery-only development checks with a fabricated CPU model."""

import hashlib
import json

import pytest
import torch

from test_targeted_edit_projected_development import guarded_model
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_screened_development as development
from looking_glass import targeted_edit_screened_math as math_helper


def test_development_records_every_fixed_condition_and_discovery_grid(tmp_path):
    model, guard = guarded_model()
    original = guard.original.clone()
    calls = []
    forward = model.forward

    def record(**kwargs):
        calls.append(tuple(kwargs["input_ids"][0].tolist()))
        return forward(**kwargs)

    model.forward = record
    output = tmp_path / "development"
    candidate = development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert calls == [(3,), (1,), (2,), (3,)] + [(1,), (2,), (3,)] * 11
    assert candidate["status"] == "private-candidate"
    assert candidate["version"] == "targeted-edit-screened-development-v1"
    assert len(candidate["discovery_changes"]) == len(candidate["calibration_shifts"]) == 11
    assert [item["dose"] for item in candidate["grid_measurements"]] == list(math_helper.DOSE_GRID)
    assert candidate["grid_measurements"][4]["condition"] == "unprojected"
    assert candidate["grid_measurements"][4]["gain"] == candidate["discovery_changes"]["unprojected"]["change"]
    assert not any("grid-05" in name for name in candidate["artifacts"])
    expected_match = math_helper.select_discovery_match(
        candidate["discovery_changes"]["projected"]["change"],
        [{"dose": item["dose"], "gain": item["gain"]} for item in candidate["grid_measurements"]],
    )
    assert candidate["discovery_match"] == expected_match
    assert json.loads((output / "pre-evaluation-candidate.json").read_bytes()) == candidate
    for name, record in candidate["artifacts"].items():
        data = (output / name).read_bytes()
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
    for condition, shifts in candidate["calibration_shifts"].items():
        assert len(shifts["max_output_abs_by_token"]) == 2
        assert len(shifts["max_algebraic_abs_by_token"]) == 2
        assert candidate["artifacts"][f"calibration-{condition}-algebraic.f64"]["shape"] == [2, guard.original.shape[1]]
        state = json.loads((output / f"state-{condition}.json").read_bytes())
        assert state["condition"] == condition and state["cleanup"] == "verified"
    assert torch.equal(model.projection.weight, original) and not guard.poisoned
    guard.verify()


def test_mutated_grid_update_fences_before_any_edited_forward(monkeypatch, tmp_path):
    model, guard = guarded_model()
    calls = []
    forward = model.forward
    build_grid = math_helper.prepare_unprojected_grid

    def record(**kwargs):
        calls.append(tuple(kwargs["input_ids"][0].tolist()))
        return forward(**kwargs)

    def tamper(*args):
        result = build_grid(*args)
        result["updates"][math_helper.DOSE_GRID[0]][0, 0] += 0.01
        return result

    model.forward = record
    monkeypatch.setattr(math_helper, "prepare_unprojected_grid", tamper)
    output = tmp_path / "partial"
    with pytest.raises(development.DevelopmentError, match="intended update changed"):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert calls == [(3,), (1,), (2,), (3,)]
    assert guard.poisoned and not (output / "pre-evaluation-candidate.json").exists()


def test_interrupted_grid_forward_fences_and_keeps_partial_evidence(tmp_path):
    model, guard = guarded_model()
    forward = model.forward
    count = 0

    def interrupted(**kwargs):
        nonlocal count
        count += 1
        if count == 21:  # First non-equal grid condition, after the five controls.
            raise KeyboardInterrupt("synthetic interruption")
        return forward(**kwargs)

    model.forward = interrupted
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert guard.poisoned and not (output / "pre-evaluation-candidate.json").exists()
    assert (output / "state-unprojected.json").exists()
    assert not (output / "state-unprojected-grid-01.json").exists()
