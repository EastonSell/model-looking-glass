"""Boundaries and aggregation of signed, matched-recipient context outcomes."""
import math

import pytest

from looking_glass import bert_cue_context_signed as signed
from looking_glass.bert_cue_context_transfer import CUES, SITES, donors


def row(*, q=0.5, baseline=True, cross=0.2, same=0.02, same_class=0.01, gap=1.0):
    baselines = [{"context": ci, "cue": cue, "any_minus_some":
                  (2.0 if cue in CUES[:2] else 0.0) + ci * gap}
                 for ci in range(2) for cue in CUES]
    feasible = q >= 0.05
    effects = []
    if feasible:
        for site in SITES:
            for ci in range(2):
                for cue in CUES:
                    for di, donor in donors(ci, cue):
                        if di != ci and donor == cue:
                            value = same
                        elif di != ci and donor == signed.same_class_other(cue):
                            value = same_class
                        else:
                            value = (-cross if cue in CUES[:2] else cross)
                        effects.append({"site": site, "target_context": ci, "target_cue": cue,
                                        "donor_context": di, "donor_cue": donor, "delta": value})
    return {"baseline_eligible": baseline, "dose_eligible": feasible, "q": q,
            "baselines": baselines, "effects": effects}


def test_signed_orientations_controls_and_same_cue_directions():
    outcome = signed.summarize_pair(row())
    assert outcome["eligible"] and outcome["primary_status"] == "pass"
    assert outcome["sensitivity_status"] == "pass"
    assert all(item["primary_margin"] == pytest.approx(0.18) for item in outcome["recipients"])
    assert all(len(item["limiting_opposite"]) == 2 for item in outcome["recipients"])
    assert all(item["limiting_control"] == ["same_cue"] for item in outcome["recipients"])
    assert [item["secondary_donorward"] for item in outcome["recipients"]] == [0.02] * 4 + [-0.02] * 4
    assert outcome["secondary"]["toward_count"] == 4 and outcome["secondary"]["exceed_count"] == 4
    # Cross-context clean gaps need not agree with the class-directed sign.
    assert outcome["recipients"][0]["opposite_clean_gaps"]["always"] == -1.0
    assert outcome["recipients"][0]["same_class_clean_gap"] == 1.0


def test_strict_primary_and_sensitivity_boundaries():
    zero = signed.summarize_pair(row(cross=0.01, same=0.01, same_class=0.01))
    assert zero["primary_raw_pass"] is False and zero["primary_status"] == "fail"
    threshold = signed.summarize_pair(row(cross=0.05, same=0.0, same_class=0.0))
    assert threshold["primary_status"] == "pass" and threshold["sensitivity_status"] == "fail"
    above = signed.summarize_pair(row(cross=math.nextafter(0.05, math.inf), same=0.0, same_class=0.0))
    assert above["sensitivity_status"] == "pass"
    backwards = signed.summarize_pair(row(cross=-0.2))
    assert backwards["primary_status"] == "fail"


def test_gap_floor_common_dose_and_ineligible_raw_outcomes():
    floor = signed.summarize_pair(row(gap=0.5))
    assert floor["secondary"]["status"] == "evaluable"
    assert floor["secondary"]["qualified_recipients"] == 8
    below = signed.summarize_pair(row(gap=0.25))
    assert below["secondary"]["status"] == "unevaluable"
    assert below["secondary"]["qualified_recipients"] == 0
    assert below["secondary"]["raw_toward_count"] == below["secondary"]["raw_exceed_count"] == 0
    assert below["secondary"]["mean_donorward"] is None
    low = signed.summarize_pair(row(q=0.4))
    assert low["primary_raw_pass"] and low["primary_status"] == "unevaluable"
    assert low["secondary"]["status"] == "unevaluable"
    baseline = signed.summarize_pair(row(baseline=False))
    assert baseline["primary_raw_pass"] and baseline["primary_status"] == "unevaluable"
    zero = signed.summarize_pair(row(q=0))
    assert zero["primary_raw_pass"] is None and zero["primary_status"] == "unevaluable"
    assert all(item["primary_margin"] is None for item in zero["recipients"])
    assert zero["secondary"]["raw_toward_count"] is None


def test_aggregation_excludes_ineligible_raw_pass_and_checks_pair_order():
    cases = [row(), row(), row(cross=0.0), row(q=0.4)]
    result = [{"id": identity, "split": "discovery" if i == 0 else "evaluation",
               "signed": signed.summarize_pair(value)}
              for i, (identity, value) in enumerate(zip(signed.PAIR_IDS, cases))]
    summary = signed.evaluate(result)
    assert summary["primary"] == {"status": "fail", "eligible_pairs": 2, "passing_pairs": 1}
    assert summary["secondary"]["evaluable_pairs"] == 2
    result[2]["signed"] = signed.summarize_pair(row(q=0.4))
    assert signed.evaluate(result)["primary"]["status"] == "unevaluable"
    result.reverse()
    with pytest.raises(ValueError, match="order"):
        signed.evaluate(result)


def test_rejects_missing_duplicate_and_nonfinite_effects():
    incomplete = row()
    incomplete["effects"].pop()
    with pytest.raises(ValueError, match="donor"):
        signed.summarize_pair(incomplete)
    duplicate = row()
    duplicate["effects"].append(duplicate["effects"][0])
    with pytest.raises(ValueError, match="duplicate"):
        signed.summarize_pair(duplicate)
    nonfinite = row()
    nonfinite["effects"][0]["delta"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        signed.summarize_pair(nonfinite)
