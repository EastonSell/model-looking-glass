from pathlib import Path

import numpy as np
import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_screened_score as score


CASES = (Path(__file__).resolve().parents[1] / "experiments/targeted_edit_screened_public_cases_v1.json").read_bytes()


def matrices(n, matched=True):
    shape = (n + 12, v1.VOCAB_SIZE)
    original = np.full(shape, -20, dtype="<f4")
    original[:, 0] = 2
    original[:, 1] = 0
    original[:n, v1.LONDON] = -0.5
    original[:n, v1.PARIS] = 0
    values = {}
    for name in score.BASE_CONDITIONS + (("matched",) if matched else ()):
        item = original.copy()
        gain = {"original": 0, "projected": 0.9, "unprojected": 1.0,
                "projected_random": 0.1, "opposite": -0.9, "nuisance": 0.2, "matched": 1.0}[name]
        item[:n, v1.LONDON] += gain
        item[n:, 1] += {"original": 0, "projected": .09, "unprojected": .1,
                        "projected_random": 0, "opposite": 0, "nuisance": 0, "matched": .1}[name]
        values[name] = item
    return values


def test_equal_and_discovery_matched_conditions_report_the_fixed_comparison():
    torch.set_flush_denormal(False)
    result = score.evaluate(matrices(8), CASES, list(range(8)), True)
    assert result["provisional_equal_norm_decision"] == "success"
    assert result["provisional_matched_discovery_decision"] == "success"
    p = result["conditions"]["projected"]
    assert p["reversed_targets"] == 8
    assert p["mean_target_change"] == pytest.approx(.9, abs=1e-6)
    assert p["mean_kl"] == pytest.approx(.0004350446, abs=1e-8)
    assert set(p["locality_strata"]) == {"other-country-capital", "other-france-fact", "unrelated-fact"}
    assert all({"original_margin", "condition_margin", "original_london_nll", "condition_london_nll",
                "original_paris_nll", "condition_paris_nll"} <= set(row) for row in p["locality"])
    assert result["matched_discovery_criteria"]["evaluated_target_gap"] == pytest.approx(.1, abs=1e-6)


@pytest.mark.parametrize("count", [0, 7])
def test_underfilled_screen_remains_ineligible_and_zero_means_are_null(count):
    torch.set_flush_denormal(False)
    result = score.evaluate(matrices(count, matched=False), CASES, list(range(count)), False)
    assert result["provisional_equal_norm_decision"] == "ineligible"
    assert result["provisional_matched_discovery_decision"] == "ineligible"
    assert len(result["conditions"]["projected"]["locality"]) == 12
    if count == 0:
        assert result["conditions"]["projected"]["mean_target_change"] is None
        assert result["conditions"]["projected"]["reversed_targets"] is None
        assert result["equal_norm_criteria"]["mean_target_gain"] is None


def test_target_sign_and_full_matrix_finiteness_are_enforced():
    torch.set_flush_denormal(False)
    values = matrices(8)
    for array in values.values():
        array[0, v1.LONDON] = 0
    with pytest.raises(v1.StudyError, match="baseline-negative"):
        score.evaluate(values, CASES, list(range(8)), True)
    values = matrices(8)
    values["matched"][0, 10] = np.nan
    with pytest.raises(v1.StudyError, match="nonfinite"):
        score.evaluate(values, CASES, list(range(8)), True)
