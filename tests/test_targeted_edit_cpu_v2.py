"""Small synthetic checks; these tests never load a pretrained model."""

import copy
import math

import numpy as np
import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2


def test_public_rows_have_no_complete_row_prefix_overlap():
    assert v2.check_split_rows() == 26


def test_public_case_identity_collision_is_rejected(monkeypatch):
    cases = list(v2.EVALUATION_CASES)
    cases[0] = (v1.CASES[1][0], *cases[0][1:])
    monkeypatch.setattr(v2, "EVALUATION_CASES", tuple(cases))
    with pytest.raises(v1.StudyError):
        v2.check_split_rows()


@pytest.fixture
def tiny_vocab(monkeypatch):
    assert torch.set_flush_denormal(False)
    monkeypatch.setattr(v1, "VOCAB_SIZE", 16)
    monkeypatch.setattr(v1, "LONDON", 1)
    monkeypatch.setattr(v1, "PARIS", 2)

    def cases_with_small_label(cases):
        return tuple((*case[:4], (" label", 3) if case[1] == "locality" else None)
                     for case in cases)

    monkeypatch.setattr(v1, "CASES", cases_with_small_label(v1.CASES))
    monkeypatch.setattr(v2, "EVALUATION_CASES", cases_with_small_label(v2.EVALUATION_CASES))


def matrices(cases, names):
    baseline = np.zeros((len(cases), 16), dtype="<f4")
    for index, case in enumerate(cases):
        if case[1] in ("discovery", "target"):
            baseline[index, 1] = -1.0
    return {name: baseline.copy() for name in names}


def dev_arrays():
    return matrices(v1.CASES, ("original", *(name for name, _ in v2.GRID)))


def eval_arrays():
    return matrices(v2.EVALUATION_CASES, v2.EVALUATION_CONDITIONS)


def set_development_gain(array, gain):
    # Baseline contrast is -1, so a gain of two reverses discovery and targets.
    for index, case in enumerate(v1.CASES):
        if case[1] in ("discovery", "target"):
            array[index, 1] = -1.0 + gain


def test_development_selection_ranks_gains_and_breaks_exact_ties_by_smaller_dose(tiny_vocab):
    data = dev_arrays()
    set_development_gain(data["dose-00005"], 2.0)
    set_development_gain(data["dose-0001"], 3.0)
    set_development_gain(data["dose-0002"], 3.0)
    report = v2.select_development(data)
    assert report["development_qualified"] is True
    assert report["selected_condition"] == "dose-0001"
    assert report["selected_relative_norm"] == 0.001
    assert report["point_prediction"] == 3.0
    assert report["sign_prediction"] == "positive"
    assert not report["criteria"]["dose-000025"]["discovery_reversal"]


def test_no_viable_dose_keeps_the_smallest_diagnostic_fallback(tiny_vocab):
    report = v2.select_development(dev_arrays())
    assert report["selected_condition"] == "dose-000025"
    assert report["selected_relative_norm"] == 0.00025
    assert report["development_qualified"] is False
    assert report["point_prediction"] == 0.0


@pytest.mark.parametrize("mutation", ["qualified", "dose", "point", "criterion", "extra"])
def test_development_selection_must_recompute_from_all_saved_conditions(tiny_vocab, mutation):
    data = dev_arrays()
    set_development_gain(data["dose-0001"], 3.0)
    valid = v2.select_development(data)
    assert v2.verify_development_selection(data, valid) == valid
    forged = copy.deepcopy(valid)
    if mutation == "qualified":
        forged["development_qualified"] = False
    elif mutation == "dose":
        forged["selected_condition"] = "dose-0002"
        forged["selected_relative_norm"] = 0.002
    elif mutation == "point":
        forged["point_prediction"] += 0.1
    elif mutation == "criterion":
        forged["criteria"]["dose-0001"]["mean_target_gain"] = False
    else:
        forged["unfrozen_note"] = True
    with pytest.raises(v1.StudyError):
        v2.verify_development_selection(data, forged)


def selection(qualified=True, point=2.0):
    return {"selected_condition": "dose-0001" if qualified else "dose-000025",
            "selected_relative_norm": 0.001 if qualified else 0.00025,
            "development_qualified": qualified, "point_prediction": point,
            "sign_prediction": "positive"}


def test_evaluation_reports_controls_predictions_and_original_v1_limits(tiny_vocab):
    data = eval_arrays()
    data["selected"][:4, 1] = 1.0
    data["v1_1_percent"][:4, 1] = 3.0
    report = v2.evaluate(data, selection())
    assert report["decision"] == "success"
    assert report["criteria"]["random_control_advantage"] is True
    assert report["prediction"] == {"sign": "positive", "point": 2.0,
                                    "observed_changes": [2.0] * 4,
                                    "strictly_positive_count": 4,
                                    "denominator": 4, "mean_absolute_error": 0.0}
    assert report["conditions"]["v1_1_percent"]["mean_target_change"] == 4.0
    assert report["v1_scalar_thresholds_on_new_sample"]["at_least_three_targets_reversed"]


def test_fallback_is_negative_and_zero_change_does_not_count_as_positive(tiny_vocab):
    data = eval_arrays()
    data["selected"][:3, 1] = 1.0
    report = v2.evaluate(data, selection(qualified=False, point=0.0))
    assert report["decision"] == "negative"
    assert report["development_qualified"] is False
    assert report["prediction"]["observed_changes"] == [2.0, 2.0, 2.0, 0.0]
    assert report["prediction"]["strictly_positive_count"] == 3
    assert report["prediction"]["mean_absolute_error"] == 1.5


def test_ineligible_baseline_precedes_unqualified_fallback(tiny_vocab):
    data = eval_arrays()
    for array in data.values():
        array[2:4, 1] = 1.0
    report = v2.evaluate(data, selection(qualified=False))
    assert report["decision"] == "ineligible"
    assert not report["criteria"]["baseline_three_targets_negative"]
    assert not report["criteria"]["development_qualified"]


def test_locality_metrics_agree_with_an_independent_closed_form(tiny_vocab):
    data = eval_arrays()
    # Original distribution is uniform across 16 tokens. Raising token 5 by
    # log(2) gives unnormalized weights (1, ..., 2, ..., 1).
    data["selected"][4, 5] = np.float32(math.log(2))
    report = v2.evaluate(data, selection())
    row = report["conditions"]["selected"]["locality"][0]
    x = float(np.float32(math.log(2)))
    normalizer = 15.0 + math.exp(x)
    assert row["original_top"] == 0  # smallest ID wins ties
    assert row["condition_top"] == 5
    assert row["original_nll"] == pytest.approx(math.log(16), abs=1e-13)
    assert row["condition_nll"] == pytest.approx(math.log(normalizer), abs=1e-13)
    assert row["kl"] == pytest.approx(math.log(normalizer / 16) - x / 16, abs=1e-13)
    assert not report["criteria"]["all_locality_top_preserved"]


@pytest.mark.parametrize("mutation", ["missing", "wrong_dtype", "nonfinite", "wrong_shape"])
def test_scoring_rejects_incomplete_or_invalid_matrices(tiny_vocab, mutation):
    data = eval_arrays()
    if mutation == "missing":
        del data["random"]
    elif mutation == "wrong_dtype":
        data["random"] = data["random"].astype("<f8")
    elif mutation == "nonfinite":
        data["random"][0, 0] = np.nan
    else:
        data["random"] = data["random"][:-1]
    with pytest.raises(v1.StudyError):
        v2.evaluate(data, selection())


@pytest.mark.parametrize("mutation", ["boolean_dose", "wrong_dose", "nan_point", "nonboolean_qualified", "unhashable_name", "wrong_fallback"])
def test_evaluation_rejects_malformed_selection(tiny_vocab, mutation):
    chosen = selection()
    if mutation == "boolean_dose":
        chosen["selected_relative_norm"] = True
    elif mutation == "wrong_dose":
        chosen["selected_relative_norm"] = 0.004
    elif mutation == "nan_point":
        chosen["point_prediction"] = float("nan")
    elif mutation == "nonboolean_qualified":
        chosen["development_qualified"] = 1
    elif mutation == "unhashable_name":
        chosen["selected_condition"] = []
    else:
        chosen["development_qualified"] = False
    with pytest.raises(v1.StudyError):
        v2.evaluate(eval_arrays(), chosen)
