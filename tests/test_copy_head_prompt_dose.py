"""Synthetic numerical and decision boundaries; these tests load no model."""
import math
import struct

import pytest

from looking_glass.copy_head_prompt_dose import (
    MIN_DOSE, SHAPE_BYTES, VOCAB, action_order, compare, decision, family_summary,
    logit_stats, plan_prompt,
)


def witness(head_values, *, other=0.0):
    values = [other] * (SHAPE_BYTES // 4)
    for head, value in head_values.items():
        values[10 * 768 + head * 64] = value
    return struct.pack(f"<{len(values)}f", *values)


def logits(changes):
    values = [0.0] * VOCAB
    for index, value in changes.items():
        values[index] = value
    return struct.pack(f"<{VOCAB}f", *values)


@pytest.mark.parametrize("smallest", [0.0, -0.0, MIN_DOSE])
def test_floor_gate_precedes_division(smallest):
    plan = plan_prompt(witness({2: smallest, 10: 1.0, 7: 2.0}))
    assert plan["status"] == "dose-ineligible"
    assert plan["reason"] == "dose floor"
    assert plan["plans"] is None
    assert plan["q"] == abs(smallest)
    assert len(plan["original"][2]) == 256


def test_equal_realized_doses_and_full_slice_bits():
    plan = plan_prompt(witness({2: 1.0, 10: 2.0, 7: 4.0}))
    assert plan["status"] == "eligible"
    assert plan["q"] == 1.0
    assert [plan["plans"][h]["fraction"] for h in (2, 10, 7)] == [1.0, .5, .25]
    assert [plan["plans"][h]["achieved"] for h in (2, 10, 7)] == [1.0, 1.0, 1.0]
    assert all(len(plan["plans"][h]["delivered"]) == 256 for h in (2, 10, 7))


def test_rounding_failure_retains_all_plans_without_false_match():
    plan = plan_prompt(witness({2: 1.0, 10: float(2**30), 7: float(2**30)}))
    assert plan["status"] == "dose-ineligible"
    assert plan["reason"] == "realized dose mismatch"
    assert plan["plans"][10]["achieved"] == 0.0
    assert set(plan["plans"]) == {2, 10, 7}


def test_nonfinite_untouched_projection_input_invalidates():
    with pytest.raises(ValueError, match="nonfinite"):
        plan_prompt(witness({2: 1.0, 10: 2.0, 7: 3.0}, other=float("nan")))


def test_full_distribution_and_tied_top():
    clean = logit_stats(logits({}), 0)
    arm = logit_stats(logits({0: 1.0}), 0)
    result = compare(clean, arm)
    z = math.e + VOCAB - 1
    assert clean["top_tie_count"] == VOCAB
    assert arm["top_unique_id"] == 0
    assert result["top_change"] == "tied/indeterminate"
    assert result["target_log_probability_delta"] == pytest.approx(1 - math.log(z / VOCAB), abs=1e-14)
    assert result["kl_clean_to_arm"] == pytest.approx(math.log(z / VOCAB) - 1 / VOCAB, abs=1e-14)
    assert result["total_variation"] == pytest.approx(math.e / z - 1 / VOCAB, abs=1e-14)
    null = compare(clean, clean)
    assert null["kl_clean_to_arm"] == 0.0
    assert null["total_variation"] == 0.0


def test_extreme_common_top_does_not_erase_target_delta():
    clean = logit_stats(logits({5: float(2**60)}), 0)
    arm = logit_stats(logits({5: float(2**60), 0: 1.0}), 0)
    assert clean["target_log_probability"] == arm["target_log_probability"]
    assert compare(clean, arm)["target_log_probability_delta"] == 1.0


def test_family_denominator_and_directional_criterion():
    def row(h2, h10, q=1.0):
        return {"status": "eligible", "q": q, "comparisons": {
            "h2": {"target_log_probability_delta": h2, "target_logit_delta": h2},
            "h10": {"target_log_probability_delta": h10, "target_logit_delta": h10},
            "h7": {"target_log_probability_delta": 0.0, "target_logit_delta": 0.0}}}
    family = family_summary([row(-2, -1, 2), row(0, 0), row(0, 0), row(0, 0)])
    assert family["contrast"] == 1.0
    assert family["dose_normalized_contrast"] == 0.5
    bad = family_summary([row(0, 0), row(0, 0), row(0, 0), {"status": "dose-ineligible", "q": 0.0}])
    assert bad["contrast"] is None
    assert decision([family] * 5 + [bad] * 3)["status"] == "ineligible"
    assert decision([family] * 6 + [bad] * 2)["status"] == "criterion-met"
    zero = {"status": "evaluable", "contrast": 0.0}
    assert decision([family] * 4 + [zero] * 2 + [bad] * 2)["status"] == "criterion-not-met"
    assert action_order(0, 0) == ("h2", "h10", "h7")
    assert action_order(0, 1) == ("h10", "h7", "h2")


def test_invalid_status_and_nonfinite_cannot_be_reclassified():
    with pytest.raises(ValueError, match="status"):
        family_summary([{"status": "invalid"}] + [{"status": "dose-ineligible", "q": 0.0}] * 3)
    with pytest.raises(ValueError, match="status"):
        decision([{"status": "typo"}] + [{"status": "dose-ineligible", "contrast": None}] * 7)
    with pytest.raises(ValueError, match="contrast"):
        decision([{"status": "evaluable", "contrast": float("nan")}]
                 + [{"status": "dose-ineligible", "contrast": None}] * 7)
    row = {"status": "eligible", "q": float("nan"), "comparisons": {}}
    with pytest.raises(ValueError, match="dose"):
        family_summary([row] * 4)
    row = {"status": "eligible", "q": 1.0,
           "comparisons": {arm: {"target_log_probability_delta": float("nan"), "target_logit_delta": 0.0}
                           for arm in ("h2", "h10", "h7")}}
    with pytest.raises(ValueError, match="delta"):
        family_summary([row] * 4)
    with pytest.raises(ValueError, match="delta"):
        family_summary([row] * 3 + [{"status": "dose-ineligible", "q": 0.0}])
