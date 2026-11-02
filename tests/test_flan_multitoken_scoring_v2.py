"""Hand-calculated, model-free scoring checks for the proposed successor."""

import pytest

from looking_glass.flan_multitoken_scoring_v2 import group_decision, primary_interval, holdout_interval


RECEIVERS = ("P1", "N1", "P2", "N2")
CONTROLS = ("same_adjective", "opposite_instruction", "opposite_subject", "orthogonal")


def patched_from_movements(clean, movements):
    return {r: clean[r] - w if r.startswith("P") else clean[r] + w
            for r, w in zip(RECEIVERS, movements)}


def test_orthogonal_control_cannot_cancel_before_absolute_aggregation():
    clean = dict(zip(RECEIVERS, (1.0, -1.0, 1.0, -1.0)))
    arms = {"opposite_adjective": patched_from_movements(clean, (0.16, 0.20, 0.12, 0.0))}
    arms.update({name: patched_from_movements(clean, (0.0, 0.0, 0.0, 0.0))
                 for name in CONTROLS})
    arms["orthogonal"] = patched_from_movements(clean, (0.12, -0.12, 0.12, -0.12))
    result = group_decision(clean, arms)
    assert result["A"] == pytest.approx(0.12)
    assert result["B3"] == 0.0
    assert result["B4"] == pytest.approx(0.12)
    assert result["positive_count"] == 3
    assert result["passes_B3"] is True
    assert result["passes_B4"] is False


def test_boundaries_include_equal_values_and_require_three_positive_directions():
    clean = {r: 0.0 for r in RECEIVERS}
    arms = {"opposite_adjective": patched_from_movements(clean, (0.1, 0.1, 0.1, 0.1))}
    arms.update({name: patched_from_movements(clean, (0.05, 0.05, 0.05, 0.05))
                 for name in CONTROLS})
    result = group_decision(clean, arms)
    assert result["A"] == 0.1
    assert result["B4"] == 0.05
    assert result["passes_B4"] is True


@pytest.mark.parametrize("outcomes,expected", [
    (("pass", "pass", "fail", "unmeasured"), (2, 3, (2, 3), "undetermined")),
    (("pass", "pass", "pass", "unmeasured"), (3, 3, (3, 4), "met")),
    (("fail", "fail", "unmeasured", "unmeasured"), (0, 2, (0, 2), "not-met")),
    (("pass", "pass", "pass", "fail"), (3, 4, (3, 3), "met")),
])
def test_planned_four_slot_interval(outcomes, expected):
    result = primary_interval(outcomes, attempt_complete=True)
    assert (result["passes"], result["evaluable"], result["interval"], result["status"]) == expected
    assert result["partial"] is (result["evaluable"] < 4)


def test_incomplete_execution_overrides_an_otherwise_met_interval():
    result = primary_interval(("pass", "pass", "pass", "unmeasured"), attempt_complete=False)
    assert result["status"] == "incomplete"
    assert result["interval"] is None


def test_discovery_reports_a_nullable_observed_pass_rate():
    empty = primary_interval(("unmeasured",) * 4, attempt_complete=True)
    assert empty["rate"] is None
    partial = primary_interval(("pass", "fail", "unmeasured", "unmeasured"), attempt_complete=True)
    assert partial["rate"] == 0.5


def test_invalid_or_nonfinite_values_are_not_silently_scored():
    clean = {r: 0.0 for r in RECEIVERS}
    arms = {name: dict(clean) for name in ("opposite_adjective", *CONTROLS)}
    arms["orthogonal"]["P1"] = float("nan")
    with pytest.raises(ValueError):
        group_decision(clean, arms)
    with pytest.raises(ValueError):
        primary_interval(("pass", "pass", "pass", "missing"), attempt_complete=True)


def test_finite_inputs_with_overflowing_movements_cannot_pass():
    clean = dict(zip(RECEIVERS, (1e308, -1e308, 1e308, -1e308)))
    arms = {name: dict(clean) for name in ("opposite_adjective", *CONTROLS)}
    arms["opposite_adjective"] = {r: -clean[r] for r in RECEIVERS}
    with pytest.raises(ValueError):
        group_decision(clean, arms)


def test_unavailable_orthogonal_arm_preserves_only_three_control_result():
    clean = {r: 0.0 for r in RECEIVERS}
    arms = {"opposite_adjective": patched_from_movements(clean, (0.2, 0.2, 0.2, 0.2))}
    arms.update({name: patched_from_movements(clean, (0.0, 0.0, 0.0, 0.0))
                 for name in CONTROLS[:3]})
    result = group_decision(clean, arms)
    assert result["passes_B3"] is True
    assert result["B4"] is None
    assert result["passes_B4"] is None


@pytest.mark.parametrize("outcomes,expected", [
    (("pass", "pass"), (2, 2, (2, 2), "met")),
    (("pass", "unmeasured"), (1, 1, (1, 2), "undetermined")),
    (("fail", "unmeasured"), (0, 1, (0, 1), "not-met")),
    (("unmeasured", "unmeasured"), (0, 0, (0, 2), "undetermined")),
])
def test_two_slot_holdout_interval_is_independent_of_discovery(outcomes, expected):
    result = holdout_interval(outcomes, attempt_complete=True)
    assert (result["passes"], result["evaluable"], result["interval"], result["status"]) == expected
    assert result["partial"] is (result["evaluable"] < 2)
    assert result["rate"] == (None if result["evaluable"] == 0 else result["passes"] / result["evaluable"])


def test_holdout_incomplete_overrides_a_met_result():
    result = holdout_interval(("pass", "pass"), attempt_complete=False)
    assert result["status"] == "incomplete"
    assert result["interval"] is None


def test_holdout_rejects_compacted_or_discovery_sized_inputs():
    with pytest.raises(ValueError):
        holdout_interval(("pass",), attempt_complete=True)
    with pytest.raises(ValueError):
        holdout_interval(("pass", "pass", "fail", "unmeasured"), attempt_complete=True)
