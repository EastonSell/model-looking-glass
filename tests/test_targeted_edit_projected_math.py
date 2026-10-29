"""Synthetic-only numerical checks; no checkpoint, model, or new case."""
import hashlib
import math

import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_projected_math as projected


def inputs():
    weight = torch.linspace(-1, 1, 24, dtype=torch.float32).reshape(6, 4)
    calibration = torch.zeros((2, 6), dtype=torch.float32)
    calibration[0, 0] = 1
    calibration[1, 1] = 1
    h = torch.arange(1, 7, dtype=torch.float32)
    g = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    return weight, calibration, h, g, torch.outer(h, g)


def test_fixed_controls_predictions_and_float32_realization():
    weight, calibration, h, g, gradient = inputs()
    original = weight.clone()
    state = torch.get_rng_state().clone()
    result = projected.prepare_projected_edits(weight, calibration, h, g, gradient)
    info = result["attestation"]
    assert torch.equal(weight, original) and torch.equal(state, torch.get_rng_state())
    assert tuple(result["updates"]) == projected.CONDITIONS
    assert torch.equal(result["updates"]["opposite"], -result["updates"]["projected"])
    assert info["rank"] == 2
    expected = math.sqrt(float(torch.dot(h[2:], h[2:])) / float(torch.dot(h, h)))
    assert info["retained_fraction"] == pytest.approx(expected, abs=1e-14)
    assert info["retained_fraction"] ** 2 + info["discarded_fraction"] ** 2 == pytest.approx(1)
    assert info["calibration_relative_residual"] == 0
    assert info["gradient_relative_residual"] == 0
    assert info["conditions"]["projected"]["calibration_algebraic_max_abs"] == 0
    assert info["conditions"]["unprojected"]["calibration_algebraic_max_abs"] > 0
    assert info["planned_hash_format"] == "contiguous C-order little-endian float64"
    assert (info["random_seed"], info["nuisance_seed"]) == (20261102, 20261101)
    factors = result["factors"]
    pairs = {
        "projected": (factors["projected_input"], g.to(torch.float64)),
        "unprojected": (h.to(torch.float64), g.to(torch.float64)),
        "projected_random": (factors["projected_random_input"], factors["random_output"]),
        "nuisance": (factors["nuisance_input"], g.to(torch.float64)),
    }
    plans = {}
    for name, (left, right) in pairs.items():
        outer = torch.outer(left, right)
        plans[name] = outer * (info["planned_delta_norm"] / float(torch.linalg.vector_norm(outer)))
    plans["opposite"] = -plans["projected"]
    for name, planned in plans.items():
        digest = hashlib.sha256(planned.contiguous().numpy().astype("<f8", copy=False).tobytes()).hexdigest()
        assert info["conditions"][name]["planned_float64_sha256"] == digest
    projected_plan = info["conditions"]["projected"]["planned_prediction"]["full_gradient_dot"]
    unprojected_plan = info["conditions"]["unprojected"]["planned_prediction"]["full_gradient_dot"]
    assert projected_plan / unprojected_plan == pytest.approx(expected, abs=1e-14)
    for name, update in result["updates"].items():
        changed = weight + update
        actual = changed.to(torch.float64) - weight.to(torch.float64)
        record = info["conditions"][name]
        assert record["intended_delta_sha256"] == v1._tensor_sha(update)
        assert record["realized_weight_sha256"] == v1._tensor_sha(changed)
        assert record["realized_delta_norm"] == pytest.approx(float(torch.linalg.vector_norm(actual)))
        assert 0.999 <= record["realized_to_planned"] <= 1.001
        assert 0.999 <= record["realized_to_projected"] <= 1.001
        assert record["addition_denormal_probe_before"] == record["addition_denormal_probe_after"]
        prediction = record["realized_prediction"]
        assert prediction["denormal_probe_before"] == prediction["denormal_probe_after"]
        assert prediction["full_gradient_dot"] == pytest.approx(
            float(torch.sum(gradient.to(torch.float64) * actual)), abs=1e-15
        )
        assert abs(prediction["full_gradient_dot"] - prediction["rank_one_dot"]) <= (
            prediction["residual_bound"] + 1e-14
        )


def test_seeded_controls_repeat_without_consuming_global_rng():
    args = inputs()
    torch.manual_seed(7)
    state = torch.get_rng_state().clone()
    first = projected.prepare_projected_edits(*args)
    assert torch.equal(state, torch.get_rng_state())
    torch.manual_seed(99)
    second_state = torch.get_rng_state().clone()
    second = projected.prepare_projected_edits(*args)
    assert torch.equal(second_state, torch.get_rng_state())
    assert all(torch.equal(first["updates"][name], second["updates"][name])
               for name in projected.CONDITIONS)
    assert first["attestation"] == second["attestation"]


def test_truncated_svd_records_a_nonzero_approximate_null():
    weight = torch.linspace(-1, 1, 12, dtype=torch.float32).reshape(6, 2)
    calibration = torch.zeros((2, 6), dtype=torch.float32)
    calibration[0, 0], calibration[1, 1] = 1, 1e-11
    h = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)
    g = torch.tensor([1, 2], dtype=torch.float32)
    result = projected.prepare_projected_edits(weight, calibration, h, g, torch.outer(h, g))
    assert result["attestation"]["rank"] == 1
    assert 0 < result["attestation"]["calibration_relative_residual"] < 1e-8


def test_full_gradient_prediction_can_reverse_the_ideal_tiny_projection():
    weight = torch.ones((2, 1), dtype=torch.float32)
    tiny = 2e-6
    calibration = torch.tensor([[-math.sqrt(1 - tiny * tiny), tiny]], dtype=torch.float32)
    h = torch.tensor([1, 0], dtype=torch.float32)
    g = torch.tensor([1], dtype=torch.float32)
    gradient = torch.tensor([[1], [-4e-6]], dtype=torch.float32)
    result = projected.prepare_projected_edits(weight, calibration, h, g, gradient)
    info = result["attestation"]
    prediction = info["conditions"]["projected"]["planned_prediction"]
    assert 1e-6 < info["retained_fraction"] < 3e-6
    assert info["gradient_relative_residual"] < 1e-5
    assert prediction["rank_one_dot"] > 0 > prediction["full_gradient_dot"]
    assert abs(prediction["full_gradient_dot"] - prediction["rank_one_dot"]) <= (
        prediction["residual_bound"] + 1e-14
    )


def test_failed_post_prediction_probe_returns_no_construction(monkeypatch):
    args = inputs()
    original_weight = args[0].clone()
    probe = projected.v1._denormal_probe
    calls = []

    def fail_during_prediction():
        calls.append(True)
        if len(calls) == 5:
            raise v1.StudyError("synthetic denormal change")
        return probe()

    monkeypatch.setattr(projected.v1, "_denormal_probe", fail_during_prediction)
    with pytest.raises(v1.StudyError, match="denormal"):
        projected.prepare_projected_edits(*args)
    assert len(calls) == 5 and torch.equal(args[0], original_weight)


@pytest.mark.parametrize("failure", ["zero", "nonfinite", "shape", "gradient", "collapsed"])
def test_invalid_constructions_fail_without_mutating_weight(failure):
    weight, calibration, h, g, gradient = inputs()
    original = weight.clone()
    if failure == "zero":
        weight.zero_()
        original = weight.clone()
    elif failure == "nonfinite":
        calibration[0, 0] = float("nan")
    elif failure == "shape":
        h = h[:-1]
    elif failure == "gradient":
        gradient[0, 0] += 1
    else:
        h = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32)
        gradient = torch.outer(h, g)
    with pytest.raises(projected.ProjectionError):
        projected.prepare_projected_edits(weight, calibration, h, g, gradient)
    assert torch.equal(weight, original)
