"""Fixed primary, balance and substitution conditions on fabricated outputs."""

import pytest

import flan_weight_edit_decision as decisions


def effects(margin, target, label):
    return {"margin_gain": margin, "target_gain": target, "strict_label": label}


def test_primary_underfill_and_failed_calibration_are_ineligible():
    cases = [{"baseline_eligible": True, "edited_label": "negative"}] * 2 + [
        {"baseline_eligible": False, "edited_label": "negative"}] * 2
    assert decisions.primary_outcome(cases, calibration_eligible=True)["status"] == "ineligible"
    cases = [{"baseline_eligible": True, "edited_label": "negative"}] * 4
    assert decisions.primary_outcome(cases, calibration_eligible=False)["status"] == "ineligible"
    assert decisions.primary_outcome(cases, calibration_eligible=True)["status"] == "positive"
    cases[0] = {"baseline_eligible": True, "edited_label": "positive"}
    cases[1] = {"baseline_eligible": True, "edited_label": "positive"}
    assert decisions.primary_outcome(cases, calibration_eligible=True)["status"] == "negative"


def test_matched_locality_requires_every_eligible_case_to_balance():
    primary = [effects(2, 1, "negative"), effects(2, 1, "negative")]
    control = [effects(1.9, .96, "negative"), effects(2.1, 1.04, "negative")]
    assert decisions.evaluation_balanced(primary, control) is True
    control[1] = effects(2.4, 1.04, "negative")
    assert decisions.evaluation_balanced(primary, control) is False
    control[1] = effects(2.0, 1.0, "positive")
    assert decisions.evaluation_balanced(primary, control) is False


def test_substitution_rejects_mismatched_same_prefix_and_allows_later_divergence():
    def record(sha, prefix):
        return {"sha256": sha, "metadata": {"decoder_prefix": prefix}}
    target = {"label_spans": {"positive": record("p", [0, 1465]), "negative": record("n", [0, 2841])},
              "greedy_spans": [record("x", [0]), record("y", [0, 12])], "greedy_ids": [12, 1]}
    donor = {"label_spans": {"positive": record("p", [0, 1465]), "negative": record("n", [0, 2841])},
             "greedy_spans": [record("x", [0]), record("y", [0, 12])], "greedy_ids": [12, 1]}
    assert decisions.check_substitution(target, donor) is True
    donor["label_spans"]["negative"] = record("changed", [0, 2841])
    with pytest.raises(decisions.SubstitutionMismatch):
        decisions.check_substitution(target, donor)
