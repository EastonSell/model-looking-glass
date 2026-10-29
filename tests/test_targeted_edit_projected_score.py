"""Synthetic score checks; no checkpoint or evaluation case is forwarded."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_projected_score as score


@pytest.fixture
def sample(monkeypatch):
    torch.set_flush_denormal(False)
    monkeypatch.setattr(v1, "VOCAB_SIZE", 16)
    monkeypatch.setattr(v1, "LONDON", 3)
    monkeypatch.setattr(v1, "PARIS", 1)
    cases = tuple((f"target-{i}", "target", "", (i,), None) for i in range(4)) + tuple(
        (f"locality-{i}", "locality", "", (i + 4,), (" expected", 2 + i)) for i in range(9))
    baseline = np.zeros((13, 16), dtype="<f4")
    baseline[:4, 1] = 1
    logits = {name: baseline.copy() for name in score.CONDITIONS}
    logits["projected"][:4, 3] = 2
    logits["unprojected"][4:, 0] = np.float32(0.1)
    development = {"status": "private-candidate", "discovery_margin": -1.0,
                   "calibration_shifts": {"projected": {"max_output_abs": 1e-7},
                                          "unprojected": {"max_output_abs": 2e-6}}}
    return cases, logits, development


def test_published_case_bytes_are_pinned():
    path = Path(__file__).resolve().parents[1] / "experiments/targeted_edit_projected_public_cases_v1.json"
    raw = path.read_bytes()
    cases = score._cases(raw)
    assert len(cases) == 13 and cases[0][0] == "projected-target-1"
    assert cases[-1][0] == "projected-locality-9" and cases[-1][-1] == (" euro", 11063)
    with pytest.raises(v1.StudyError, match="case bytes"):
        score._cases(raw + b" ")


def test_full_public_case_shape_without_model():
    torch.set_flush_denormal(False)
    path = Path(__file__).resolve().parents[1] / "experiments/targeted_edit_projected_public_cases_v1.json"
    original = np.zeros((13, v1.VOCAB_SIZE), dtype="<f4")
    original[:4, v1.PARIS] = 1
    logits = {name: original for name in score.CONDITIONS}
    development = {"status": "private-candidate", "discovery_margin": -1.0,
                   "calibration_shifts": {"projected": {"max_output_abs": 1e-7},
                                          "unprojected": {"max_output_abs": 2e-6}}}
    result = score.evaluate(logits, path.read_bytes(), development)
    assert result["provisional_numerical_decision"] == "negative"
    assert result["conditions"]["projected"]["targets"][0]["case_id"] == "projected-target-1"
    assert result["conditions"]["projected"]["locality"][-1]["case_id"] == "projected-locality-9"


def test_all_condition_metrics_and_success_are_provisional(sample):
    cases, logits, development = sample
    result = score._evaluate(logits, cases, development)
    assert result["provisional_numerical_decision"] == "success"
    assert set(result["conditions"]) == set(score.CONDITIONS)
    assert all(result["eligibility"].values()) and all(result["criteria"].values())
    projected = result["conditions"]["projected"]
    assert len(projected["targets"]) == 4 and len(projected["locality"]) == 9
    assert projected["mean_target_change"] == 2 and projected["reversed_targets"] == 4
    target = projected["targets"][0]
    assert (target["original_top"], target["condition_top"]) == (1, 3)
    base_logz = math.log(15 + math.e)
    edit_logz = math.log(14 + math.e + math.exp(2))
    assert target["original_london_nll"] == pytest.approx(base_logz)
    assert target["original_paris_nll"] == pytest.approx(base_logz - 1)
    assert target["condition_london_nll"] == pytest.approx(edit_logz - 2)
    assert target["condition_paris_nll"] == pytest.approx(edit_logz - 1)
    expected_target_kl = edit_logz - base_logz - 2 / (15 + math.e)
    assert target["kl"] == pytest.approx(expected_target_kl, abs=1e-15)
    expected_kl = math.log((15 + math.exp(float(np.float32(0.1)))) / 16) - float(np.float32(0.1)) / 16
    assert result["conditions"]["unprojected"]["mean_kl"] == pytest.approx(expected_kl, abs=1e-15)
    assert result["denormal_probe_before"] == result["denormal_probe_after"]


@pytest.mark.parametrize("kind", ("discovery", "targets", "calibration"))
def test_ineligible_precedes_numerical_criteria(sample, kind):
    cases, logits, development = sample
    if kind == "discovery":
        development["discovery_margin"] = 0.0
    elif kind == "targets":
        for matrix in logits.values():
            matrix[2:4, 1] = 0
    else:
        development["calibration_shifts"]["unprojected"]["max_output_abs"] = 1e-6
    result = score._evaluate(logits, cases, development)
    assert result["provisional_numerical_decision"] == "ineligible"


def test_strict_target_gain_and_inclusive_random_advantage(sample):
    cases, logits, development = sample
    for matrix in logits.values():
        matrix[:4, 1] = 0.25
    logits["projected"][:4, 3] = 0.5
    result = score._evaluate(logits, cases, development)
    assert result["conditions"]["projected"]["mean_target_change"] == 0.5
    assert not result["criteria"]["mean_target_gain"]
    assert result["criteria"]["random_control_advantage"]
    assert result["provisional_numerical_decision"] == "negative"


def test_calibration_ratio_boundary_is_inclusive(sample):
    cases, logits, development = sample
    development["calibration_shifts"]["projected"]["max_output_abs"] = 0.1 * 2e-6
    assert score._evaluate(logits, cases, development)["criteria"]["calibration_shift_ratio"]
    development["calibration_shifts"]["projected"]["max_output_abs"] = 0.1001 * 2e-6
    assert not score._evaluate(logits, cases, development)["criteria"]["calibration_shift_ratio"]


def test_locality_kl_advantage_on_both_sides_of_threshold(sample):
    cases, logits, development = sample

    def oracle(value):
        value = float(value)
        return math.log1p(math.expm1(value) / 16) - value / 16

    low, high = 0.0, 0.1
    for _ in range(60):
        middle = (low + high) / 2
        if oracle(middle) < 1e-6:
            low = middle
        else:
            high = middle
    below = np.float32(low)
    while oracle(below) >= 1e-6:
        below = np.nextafter(below, np.float32(-np.inf))
    above = np.nextafter(below, np.float32(np.inf))
    while oracle(above) < 1e-6:
        above = np.nextafter(above, np.float32(np.inf))
    for value, passes in ((below, False), (above, True)):
        logits["unprojected"][4:, 0] = value
        result = score._evaluate(logits, cases, development)
        assert result["conditions"]["unprojected"]["mean_kl"] == pytest.approx(oracle(value), abs=2e-15)
        assert result["criteria"]["locality_kl_advantage"] is passes


@pytest.mark.parametrize("bad", (True, float("nan"), float("inf"), 10**1000))
def test_invalid_development_measurements_fail_closed(sample, bad):
    cases, logits, development = sample
    development["discovery_margin"] = bad
    with pytest.raises(v1.StudyError, match="measurement"):
        score._evaluate(logits, cases, development)


def test_post_score_denormal_probe_is_required(sample, monkeypatch):
    cases, logits, development = sample
    actual_probe = v1._denormal_probe
    count = 0

    def fail_after_scores():
        nonlocal count
        count += 1
        if count == 3:
            raise v1.StudyError("injected post-score denormal failure")
        return actual_probe()

    monkeypatch.setattr(v1, "_denormal_probe", fail_after_scores)
    with pytest.raises(v1.StudyError, match="post-score"):
        score._evaluate(logits, cases, development)
