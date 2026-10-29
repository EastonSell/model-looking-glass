"""Small synthetic checks; these tests never load a pretrained model."""

import copy
import hashlib
import math
from pathlib import Path

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


def test_tied_extreme_logits_keep_the_log_normalizer_and_correct_kl(tiny_vocab, monkeypatch):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 4)
    row = np.asarray([[2**60, 2**60, -(2**60), -(2**60)]], dtype="<f4")
    uniform = np.zeros((1, 4), dtype="<f4")
    scores = v2._scores({"extreme": row, "uniform": uniform}, (None,), ("extreme", "uniform"))
    _z, logp, top = scores["extreme"][0]
    _uz, uniform_logp, _top = scores["uniform"][0]
    assert top == 0
    assert logp[0] == pytest.approx(-math.log(2), abs=1e-15)
    assert logp[1] == pytest.approx(-math.log(2), abs=1e-15)
    assert -logp[0] == pytest.approx(math.log(2), abs=1e-15)
    assert float(np.exp(logp).sum()) == pytest.approx(1.0, abs=1e-15)
    kl = float(np.sum(np.exp(logp) * (logp - uniform_logp), dtype=np.float64))
    assert kl == pytest.approx(math.log(2), abs=1e-15)


def test_ordinary_logits_match_an_independent_float64_normalization(tiny_vocab, monkeypatch):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 4)
    row = np.asarray([[-2.0, 0.0, 1.0, 3.0]], dtype="<f4")
    _z, logp, top = v2._scores({"test": row}, (None,), ("test",))["test"][0]
    denominator = math.fsum(math.exp(float(value) - 3.0) for value in row[0])
    expected = [float(value) - 3.0 - math.log(denominator) for value in row[0]]
    assert top == 3
    assert np.allclose(logp, expected, rtol=0, atol=1e-15)


def test_saved_v1_logits_remain_numerically_close_under_v2_normalization():
    assert torch.set_flush_denormal(False)
    root = Path(__file__).resolve().parents[1] / "results" / "targeted-edit-cpu-v1-20261028" / "reference"
    pins = {"original": "947518214fd7c8bdfebd347a1944cc4ce5cb97ac9e622846cc7e21308f96c309",
            "gradient": "43e94ab1a840df8c0d4fb65e56966befb06110a05edae9094e3817515cf339f7",
            "random": "1d7fd5030e7d54bbab5ce3796f1a2e719e82c1e136de64294e061babf1c9fef7"}
    old, new = {}, {}
    for condition, pin in pins.items():
        raw = (root / f"{condition}.f32").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin
        rows = np.frombuffer(raw, dtype="<f4").reshape(len(v1.CASES), v1.VOCAB_SIZE)
        old[condition] = [v1._score(row)[1] for row in rows]
        new[condition] = [v2._score(row)[1] for row in rows]
    for condition in pins:
        assert max(float(np.max(np.abs(a - b))) for a, b in zip(old[condition], new[condition])) < 1e-12
    for condition in ("gradient", "random"):
        for index in range(5, len(v1.CASES)):
            kl_old = np.sum(np.exp(old["original"][index]) *
                            (old["original"][index] - old[condition][index]), dtype=np.float64)
            kl_new = np.sum(np.exp(new["original"][index]) *
                            (new["original"][index] - new[condition][index]), dtype=np.float64)
            assert abs(float(kl_old - kl_new)) < 1e-12


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
