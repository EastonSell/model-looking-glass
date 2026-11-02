"""Synthetic checks for the screened follow-on; no model or study inputs."""
import hashlib

import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_screened_math as screened


def inputs():
    weight = torch.linspace(-1, 1, 24, dtype=torch.float32).reshape(6, 4)
    calibration = torch.zeros((2, 6), dtype=torch.float32)
    calibration[0, 0] = 1
    calibration[1, 1] = 1
    h = torch.arange(1, 7, dtype=torch.float32)
    g = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    return weight, calibration, h, g, torch.outer(h, g)


def rows(gains):
    return [{"dose": d, "gain": gain} for d, gain in zip(screened.DOSE_GRID, gains)]


def test_fresh_seed_controls_and_grid_are_finite_and_reproducible():
    args = inputs()
    assert torch.set_flush_denormal(False)
    state = torch.get_rng_state().clone()
    base = screened.prepare_projected_edits(*args)
    grid = screened.prepare_unprojected_grid(*args, base)
    assert torch.equal(state, torch.get_rng_state())
    assert (base["attestation"]["random_seed"], base["attestation"]["nuisance_seed"]) == (20261103, 20261104)
    generator = torch.Generator(device="cpu").manual_seed(20261103)
    left = torch.randn(6, generator=generator, dtype=torch.float64)
    right = torch.randn(4, generator=generator, dtype=torch.float64)
    assert torch.equal(base["factors"]["random_input"], left)
    assert torch.equal(base["factors"]["random_output"], right)
    assert torch.equal(base["updates"]["opposite"], -base["updates"]["projected"])
    assert torch.equal(grid["updates"][0.001], base["updates"]["unprojected"])
    assert [row["dose"] for row in grid["attestation"]["candidates"]] == list(screened.DOSE_GRID)
    for item in grid["attestation"]["candidates"]:
        delta = grid["updates"][item["dose"]]
        changed = args[0] + delta
        actual = changed.to(torch.float64) - args[0].to(torch.float64)
        assert item["intended_delta_sha256"] == v1._tensor_sha(delta)
        assert item["realized_weight_sha256"] == v1._tensor_sha(changed)
        assert item["realized_delta_float64_sha256"] == hashlib.sha256(
            actual.contiguous().numpy().astype("<f8", copy=False).tobytes()).hexdigest()
        assert 0.999 <= item["realized_to_planned"] <= 1.001
        assert item["realized_prediction"]["full_gradient_dot"] == pytest.approx(
            float(torch.sum(args[4].to(torch.float64) * actual)), abs=1e-15)


def test_grid_rejects_changed_input_and_changed_equal_norm_update():
    args = inputs()
    assert torch.set_flush_denormal(False)
    base = screened.prepare_projected_edits(*args)
    changed = list(args)
    changed[2] = changed[2].clone()
    changed[2][0] += 1
    with pytest.raises(screened.ProjectionError, match="differ"):
        screened.prepare_unprojected_grid(*changed, base)
    base["updates"]["unprojected"][0, 0] += 1
    with pytest.raises(screened.ProjectionError, match="equal-norm"):
        screened.prepare_unprojected_grid(*args, base)


def test_match_uses_floor_tolerance_tie_break_and_shared_arm():
    tied = rows([0.1, 0.9375, 1.0625, 1.8, 1.9, 2.0, 2.1])
    choice = screened.select_discovery_match(1.0, tied)
    assert choice["available"] and choice["selected_dose"] == 0.000625
    assert choice["nearest_discrepancy"] == 0.0625
    shared = screened.select_discovery_match(1.0, rows([0.25, 0.3, 0.4, 0.5, 1.0, 1.4, 1.8]))
    assert shared["selected_dose"] == 0.001 and shared["shared_equal_norm_arm"]


def test_no_match_is_null_and_nearest_is_retained():
    result = screened.select_discovery_match(1.0, rows([0.25, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35]))
    assert result["selected_dose"] is None and result["nearest_dose"] == 0.00125
    assert result["reason"] == "nearest_discrepancy_exceeds_tolerance"
    none = screened.select_discovery_match(1.0, rows([0.25] * 7))
    assert none["nearest_dose"] is None and none["reason"] == "no_unprojected_gain_above_floor"
    floor = screened.select_discovery_match(0.25, rows([0.3] * 7))
    assert floor["selected_dose"] is None and floor["reason"] == "projected_gain_not_above_floor"


@pytest.mark.parametrize("failure", ["nan", "boolean", "missing", "reordered", "extra"])
def test_bad_or_incomplete_grid_is_not_a_no_match(failure):
    grid = rows([0.5] * 7)
    gain = 1.0
    if failure == "nan":
        grid[0]["gain"] = float("nan")
    elif failure == "boolean":
        gain = True
    elif failure == "missing":
        grid.pop()
    elif failure == "reordered":
        grid[0], grid[1] = grid[1], grid[0]
    else:
        grid[0]["extra"] = 1
    with pytest.raises(screened.ProjectionError):
        screened.select_discovery_match(gain, grid)


def test_unexpected_denormal_mode_is_not_silently_reset():
    args = inputs()
    assert torch.set_flush_denormal(True)
    try:
        with pytest.raises(v1.StudyError, match="denormal"):
            screened.prepare_projected_edits(*args)
    finally:
        assert torch.set_flush_denormal(False)
