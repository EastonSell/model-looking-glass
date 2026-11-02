import math
import random

import numpy as np
import pytest

from looking_glass import answer_format_feasibility as study


def cases(stage):
    return [{"case_id": f"{stage}-{fact}-{family}-{prefix}", "role": stage,
             "fact_id": fact, "family": family, "prefix": prefix,
             "labels": {name: {"token_id": i} for i, name in enumerate(study.LABELS)}}
            for fact in study.FACTS for family in study.FAMILIES for prefix in study.PREFIXES]


def matrix(stage, robust, tie=None):
    rows = []
    for fact in study.FACTS:
        for family in study.FAMILIES:
            for prefix in study.PREFIXES:
                x = np.full(10, -5, dtype=np.float32)
                x[0] = 4
                if prefix == "spain":
                    x[2] = 3
                if prefix == "sweden":
                    x[3] = 3
                if (fact, family) not in robust and prefix == "neutral":
                    x[4] = 5
                if (fact, family, prefix) == tie:
                    x[4] = 4
                rows.append(x)
    return np.stack([rows[i] for i in study.execution_order(stage)])


def test_seeded_orders_are_exact_shuffles_and_repeatable():
    for stage, seed in study.SEEDS.items():
        expected = list(range(32))
        random.Random(seed).shuffle(expected)
        assert study.execution_order(stage) == expected == study.execution_order(stage)
    assert study.execution_order("development") != study.execution_order("heldout")


def test_development_selection_and_all_pair_metrics():
    robust = {(f, "statement") for f in study.FACTS[:3]}
    result = study.summarize("development", matrix("development", robust), cases("development"))
    assert result["status"] == "eligible"
    assert result["selected_family"] == "statement"
    assert result["families"][0]["robust_facts"] == list(study.FACTS[:3])
    assert len(result["observations"]) == 32 and len(result["pairs"]) == 48
    assert len(result["contrasts"]) == 8
    assert all(x["d"] == pytest.approx(16) for x in result["contrasts"])
    pair = next(x for x in result["pairs"] if x["fact_id"] == "france"
                and x["family"] == "statement" and (x["reference"], x["comparison"]) == ("none", "spain"))
    a = np.array([4, -5, -5, -5, -5, -5, -5, -5, -5, -5], dtype=np.float64)
    b = a.copy()
    b[2] = 3
    pa, pb = np.exp(a - np.log(np.exp(a).sum())), np.exp(b - np.log(np.exp(b).sum()))
    assert pair["kl"] == pytest.approx(float(np.sum(pa * np.log(pa / pb))), abs=1e-13)
    assert pair["total_variation"] == pytest.approx(float(np.abs(pa - pb).sum() / 2), abs=1e-13)


def test_heldout_reuses_selected_development_facts_without_reselection():
    dev_robust = {(f, "statement") for f in study.FACTS[:3]}
    dev = study.summarize("development", matrix("development", dev_robust), cases("development"))
    heldout_robust = {(f, "statement") for f in study.FACTS[:2]} | {(f, "question") for f in study.FACTS}
    held = study.summarize("heldout", matrix("heldout", heldout_robust), cases("heldout"), development=dev)
    assert held["selected_family"] == "statement"
    assert held["reproduced_development_facts"] == list(study.FACTS[:2])
    assert held["status"] == "negative"
    assert held["descriptive_direction_met"] is True
    held = study.summarize("heldout", matrix("heldout", dev_robust), cases("heldout"), development=dev)
    assert held["status"] == "positive"


def test_underfill_still_reports_all_heldout_contrasts():
    dev = study.summarize("development", matrix("development", set()), cases("development"))
    assert dev["status"] == "ineligible" and dev["selected_family"] is None
    held = study.summarize("heldout", matrix("heldout", set()), cases("heldout"), development=dev)
    assert held["status"] == "ineligible"
    assert held["selected_family"] is None
    assert len(held["observations"]) == 32 and len(held["heldout_fact_means"]) == 4
    assert held["descriptive_direction_met"] is True


def test_ties_are_not_unique_factual_and_inconsistent_metadata_fails():
    robust = {(f, "statement") for f in study.FACTS[:3]}
    raw = matrix("development", robust, tie=("france", "statement", "none"))
    result = study.summarize("development", raw, cases("development"))
    item = next(r for r in result["observations"] if r["fact_id"] == "france"
                and r["family"] == "statement" and r["prefix"] == "none")
    assert item["unique_top_id"] is None and item["top_tie_count"] == 2
    assert item["factual_top"] is False and result["status"] == "ineligible"
    invalid = cases("development")
    invalid[1]["labels"]["factual"]["token_id"] = 4
    with pytest.raises(ValueError):
        study.summarize("development", raw, invalid)


def test_nonfinite_rows_and_forged_selection_fail_closed():
    raw = matrix("development", set())
    raw[0, 0] = np.nan
    with pytest.raises(ValueError):
        study.summarize("development", raw, cases("development"))
    dev = study.summarize("development", matrix("development", set()), cases("development"))
    dev["selected_family"] = "statement"
    with pytest.raises(ValueError):
        study.summarize("heldout", matrix("heldout", set()), cases("heldout"), development=dev)


def test_extreme_finite_tie_has_normalized_log_probability():
    x = np.full(10, -1e20, dtype=np.float32)
    x[0] = x[1] = np.float32(1e20)
    values = study.full_top._rows32(x.reshape(1, -1), 1)[0]
    logs, _maximum, _logsum = study._log_distribution(values)
    assert logs[0] == pytest.approx(-math.log(2))
    assert logs[1] == pytest.approx(-math.log(2))


def test_example_contrast_retains_sign_with_huge_unrelated_top():
    stage = "heldout"
    canonical = []
    for fact in study.FACTS:
        for family in study.FAMILIES:
            for prefix in study.PREFIXES:
                x = np.zeros(10, dtype=np.float32)
                x[4] = np.float32(2**60)
                if prefix == "spain":
                    x[2], x[3] = 1, 0
                elif prefix == "sweden":
                    x[2], x[3] = 0, 1
                canonical.append(x)
    raw = np.stack([canonical[i] for i in study.execution_order(stage)])
    dev = study.summarize("development", matrix("development", set()), cases("development"))
    result = study.summarize(stage, raw, cases(stage), development=dev)
    assert all(x["d"] == 2 for x in result["contrasts"])
    assert result["descriptive_direction_met"] is True
    pair = next(p for p in result["pairs"] if p["fact_id"] == "france"
                and p["family"] == "statement" and p["reference"] == "none" and p["comparison"] == "spain")
    assert pair["logp_delta"]["madrid"] == 1


def test_fact_mean_preserves_small_residual_across_large_opposite_contrasts():
    stage = "heldout"
    canonical = []
    for fact in study.FACTS:
        for family in study.FAMILIES:
            for prefix in study.PREFIXES:
                x = np.zeros(10, dtype=np.float32)
                x[4] = np.float32(2**61)
                if family == "statement" and prefix == "spain":
                    x[2] = np.float32(2**60)
                if family == "question" and prefix == "sweden":
                    x[2], x[3] = np.float32(2**60), 1
                canonical.append(x)
    raw = np.stack([canonical[i] for i in study.execution_order(stage)])
    dev = study.summarize("development", matrix("development", set()), cases("development"))
    result = study.summarize(stage, raw, cases(stage), development=dev)
    assert all(x["mean_d"] == 0.5 for x in result["heldout_fact_means"])
    assert result["descriptive_direction_met"] is True

