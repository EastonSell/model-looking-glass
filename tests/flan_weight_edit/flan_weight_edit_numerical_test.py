"""Boundary checks for the FLAN weight-change numerical contract."""

import math

import numpy as np

import flan_weight_edit_numerical as science


def row(margin, target, negative=True):
    return {"margin_gain": margin, "target_gain": target, "negative": negative}


def test_per_case_balance_rejects_a_matching_average_and_ties_choose_lower_dose():
    primary = [row(2.0, 1.0), row(2.0, 1.0)]
    candidates = [
        {"dose": 0.25, "rows": [row(1.0, 0.0), row(3.0, 2.0)]},
        {"dose": 1.0, "rows": [row(1.9, 0.96), row(2.1, 1.04)]},
        {"dose": 0.5, "rows": [row(1.9, 0.96), row(2.1, 1.04)]},
    ]
    result = science.select_control(primary, candidates)
    assert result["matched"] is True
    assert result["dose"] == 0.5
    assert math.isclose(result["maximum_ratio"], 0.5)


def test_failed_primary_floor_and_greedy_gate_keep_nearest_unmatched():
    primary = [row(0.49, 1.0), row(1.0, 1.0)]
    candidates = [
        {"dose": 1.0, "rows": [row(0.49, 1.0), row(1.0, 1.0)]},
        {"dose": 0.5, "rows": [row(0.49, 1.0), row(1.0, 1.0, False)]},
    ]
    result = science.select_control(primary, candidates)
    assert result["matched"] is False
    assert result["dose"] == 0.5
    assert result["maximum_ratio"] == 0.0
    assert result["primary_eligible"] is False


def test_stable_score_effect_avoids_subtracting_rounded_large_logprobs():
    huge = float(2**60)
    before = [(0.0, huge, 0.0)]
    after = [(1.0, huge, 0.0)]
    assert science.sequence_score(after) == science.sequence_score(before)
    assert science.sequence_effect(after, before) == 1.0
    assert science.margin_effect(after, before, before, before) == 1.0


def test_frozen_random_direction_is_reproducible_orthogonal_and_same_norm():
    wonderful = np.arange(512, dtype=np.float32) / 100
    awful = wonderful + np.linspace(-1, 1, 512, dtype=np.float32)
    excellent = wonderful + 0.1
    head_negative = np.full(512, 0.25, dtype=np.float32)
    head_positive = np.full(512, -0.5, dtype=np.float32)
    a = science.directions(wonderful, awful, excellent, head_negative, head_positive)
    b = science.directions(wonderful, awful, excellent, head_negative, head_positive)
    assert np.array_equal(a["random"], b["random"])
    assert abs(math.fsum(float(x) * float(y) for x, y in zip(a["random"], a["target"]))) < 1e-12
    assert math.isclose(science.norm(a["random"]), science.norm(a["target"]), rel_tol=1e-14)
    assert np.array_equal(science.donor_row(awful), awful)
    assert np.array_equal(science.donor_row(excellent), excellent)
    assert np.array_equal(science.construct_row(head_negative, a["broad"], 1.0), head_positive)


def test_primary_replacement_copies_donor_bits_even_when_direction_rounding_loses_them():
    wonderful = np.full(512, 1e30, dtype=np.float32)
    awful = np.full(512, 1e-20, dtype=np.float32)
    direction = np.array([math.fsum((float(a), -float(w))) for w, a in zip(wonderful, awful)])
    assert not np.array_equal(science.construct_row(wonderful, direction, 1.0), awful)
    actual = science.donor_row(awful)
    assert np.array_equal(actual.view(np.uint32), awful.view(np.uint32))
    assert actual is not awful


def test_fixed_selector_rejects_an_incomplete_grid():
    import pytest

    primary = [row(2.0, 1.0), row(2.0, 1.0)]
    with pytest.raises(ValueError, match="complete fixed grid"):
        science.select_control_fixed(primary, [{"dose": 0.25, "rows": primary}])
