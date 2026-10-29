"""Model-free behavioral edit reports and their incomplete-evidence boundaries."""

import json
import math

import pytest

from looking_glass.behavioral_edit_evaluation import (
    BehavioralCase, BehavioralObservation, CategoricalDistribution,
    RollbackEvidence, ScalarComparison, StateDigest, evaluate_behavioral_edit,
)


def observation(value, *, ids=None, text=None, termination=None):
    return BehavioralObservation(value=value, token_ids=ids, decoded_text=text, termination=termination)


def dist(probabilities, support=(0, 1, 2), context="first-step"):
    return CategoricalDistribution(support=support, probabilities=probabilities, context=context)


def test_reports_target_and_locality_without_interpreting_damage():
    cases = [
        BehavioralCase("target-1", "target", observation("Paris", ids=(7, 1), text="Paris", termination="eos"),
                       observation("London", ids=(8, 1), text="London", termination="eos"),
                       input_id="prompt-1", references=("Paris",), targets=("London",),
                       before_distribution=dist((.7, .2, .1)), after_distribution=dist((.2, .7, .1)),
                       measures=(ScalarComparison("sequence_margin", -1.7, -.8),
                                 ScalarComparison("target_log_probability", -2.0, -1.0))),
        BehavioralCase("local-1", "locality", observation("wrong"), observation("different-wrong"),
                       references=("correct",), before_distribution=dist((.4, .3, .3)),
                       after_distribution=dist((.3, .4, .3))),
        BehavioralCase("local-2", "locality", observation("correct"), observation("wrong"),
                       references=("correct",)),
        BehavioralCase("local-3", "locality", observation("wrong"), observation("correct"),
                       references=("correct",)),
    ]
    report = evaluate_behavioral_edit(cases)
    target, local1, local2, local3 = report["cases"]
    assert target["input_id"] == "prompt-1"
    assert target["before"]["token_ids"] == [7, 1]
    assert target["after"]["decoded_text"] == "London"
    assert target["after"]["termination"] == "eos"
    assert target["before_reference_match"] is True
    assert target["after_reference_match"] is False
    assert target["before_target_match"] is False
    assert target["after_target_match"] is True
    assert target["distribution"]["total_variation"] == pytest.approx(.5)
    assert target["measures"]["sequence_margin"]["delta"] == pytest.approx(.9)
    assert target["measures"]["target_log_probability"]["delta"] == pytest.approx(1.0)
    assert (local1["before_reference_match"], local1["after_reference_match"]) == (False, False)
    assert (local2["before_reference_match"], local2["after_reference_match"]) == (True, False)
    assert (local3["before_reference_match"], local3["after_reference_match"]) == (False, True)
    summary = report["summaries"]["locality"]
    assert summary["paired_reference_count"] == 3
    assert summary["before_reference_matches"] == 1
    assert summary["after_reference_matches"] == 1
    assert summary["reference_match_delta"] == 0
    assert summary["lost_reference_match"] == summary["gained_reference_match"] == 1
    assert summary["outcome_value_changed_count"] == 3
    assert "damage" not in str(report).lower() and "success" not in report
    assert report["rollback"]["status"] == "not_observed"


def test_unchanged_outcome_can_have_a_changed_full_distribution():
    case = BehavioralCase("local", "locality", observation(0), observation(0), references=(0,),
                          before_distribution=dist((.5, .3, .2)), after_distribution=dist((.5, .1, .4)))
    row = evaluate_behavioral_edit([case])["cases"][0]
    assert row["outcome_value_changed"] is False
    assert row["distribution"]["status"] == "compared"
    assert row["distribution"]["support_size"] == 3
    assert row["distribution"]["total_variation"] == pytest.approx(.2)


def test_missing_tied_and_invalid_evidence_do_not_become_failures_or_zero_rates():
    cases = [
        BehavioralCase("tied", "target", BehavioralObservation(status="tied"), observation("b"),
                       references=("a",), targets=("b",), before_distribution=dist((.5, .5, 0)),
                       after_distribution=dist((.4, .6, 0))),
        BehavioralCase("invalid", "target", observation("a"), BehavioralObservation(status="invalid", reason="nonfinite"),
                       references=("a",), measures=(ScalarComparison("score", 1.0, math.nan),)),
        BehavioralCase("missing", "locality", observation("a"), None, references=("a",),
                       before_distribution=dist((1, 0, 0)), measures=(ScalarComparison("score", None, 3.0),)),
    ]
    report = evaluate_behavioral_edit(cases)
    tied, invalid, missing = report["cases"]
    assert tied["before"]["status"] == "tied" and tied["before_reference_match"] is None
    assert tied["distribution"]["total_variation"] == pytest.approx(.1)
    assert invalid["after"]["reason"] == "nonfinite" and invalid["after_reference_match"] is None
    assert invalid["measures"]["score"]["after_status"] == "invalid_nonfinite"
    assert invalid["measures"]["score"]["delta"] is None
    assert missing["after"]["status"] == "missing"
    assert missing["distribution"]["status"] == "missing_after"
    assert missing["measures"]["score"]["delta"] is None
    for group in ("target", "locality"):
        summary = report["summaries"][group]
        assert summary["paired_reference_count"] == 0
        assert summary["before_reference_rate"] is summary["after_reference_rate"] is None
        assert summary["reference_match_delta"] is None


def test_token_sequence_including_eos_and_decoded_text_are_retained():
    before = observation((5, 1), ids=(5, 1), text="answer", termination="eos")
    after = observation((5,), ids=(5,), text="answer", termination="length_limit")
    case = BehavioralCase("sequence", "locality", before, after, references=((5, 1),), restored=before)
    row = evaluate_behavioral_edit([case])["cases"][0]
    assert row["before"]["value"] == row["before"]["token_ids"] == [5, 1]
    assert row["after"]["decoded_text"] == "answer"
    assert row["outcome_value_changed"] is True and row["after_reference_match"] is False
    assert row["restored_value_matches_before"] is True


@pytest.mark.parametrize("after, status", [
    (dist((.5, .5), support=(1, 0)), "incomparable_support"),
    (dist((.5, .3, .2), context="second-step"), "incomparable_context"),
    (dist((math.nan, .5, .5)), "invalid_after"),
    (dist((.4, -.1, .7)), "invalid_after"),
    (dist((.1, .2, .3)), "invalid_after"),
])
def test_unusable_distribution_never_produces_tv(after, status):
    support = (0, 1) if status == "incomparable_support" else (0, 1, 2)
    before = dist((.5, .5), support=support) if len(support) == 2 else dist((.5, .3, .2))
    case = BehavioralCase("row", "locality", observation(0), observation(0),
                          before_distribution=before, after_distribution=after)
    distribution = evaluate_behavioral_edit([case])["cases"][0]["distribution"]
    assert distribution["status"] == status
    assert distribution["total_variation"] is None


def test_disjoint_probabilities_and_one_sided_missing_are_reported():
    first = BehavioralCase("a", "target", observation(0), observation(1),
                           before_distribution=dist((1, 0, 0)), after_distribution=dist((0, 1, 0)))
    second = BehavioralCase("b", "target", observation(0), observation(1),
                            after_distribution=dist((0, 1, 0)))
    report = evaluate_behavioral_edit([first, second])
    assert report["cases"][0]["distribution"]["total_variation"] == 1.0
    assert report["cases"][1]["distribution"]["status"] == "missing_before"
    assert report["summaries"]["target"]["distribution_compared_count"] == 1
    assert report["summaries"]["target"]["mean_total_variation"] == 1.0
    assert report["summaries"]["locality"]["mean_total_variation"] is None


@pytest.mark.parametrize("rollback, expected", [
    (None, "not_observed"),
    (RollbackEvidence(False), "not_attempted"),
    (RollbackEvidence(True, StateDigest("parameters-and-buffers", "a" * 64), None), "incomplete"),
    (RollbackEvidence(True, StateDigest("parameters", "a" * 64), StateDigest("parameters", "a" * 64)), "reported_match"),
    (RollbackEvidence(True, StateDigest("parameters", "a" * 64), StateDigest("parameters", "b" * 64)), "reported_mismatch"),
    (RollbackEvidence(True, StateDigest("parameters", "a" * 64), StateDigest("buffers", "a" * 64)), "incomparable_scope"),
])
def test_rollback_status_is_limited_to_caller_supplied_state_scope(rollback, expected):
    case = BehavioralCase("case", "target", observation("a"), observation("b"), restored=observation("a"))
    report = evaluate_behavioral_edit([case], rollback=rollback)
    assert report["rollback"]["status"] == expected
    assert report["cases"][0]["restored_value_matches_before"] is True


@pytest.mark.parametrize("case", [
    BehavioralCase("", "target", observation("a"), observation("b")),
    BehavioralCase("id", "other", observation("a"), observation("b")),
    BehavioralCase("id", "target", observation(True), observation(1)),
    BehavioralCase("id", "target", observation((1, 2), ids=(1,)), observation((1, 2))),
    BehavioralCase("id", "target", observation(1), observation("1")),
    BehavioralCase("id", "target", observation(1), observation(1), references=(True,)),
    BehavioralCase("id", "target", observation(1), observation(1),
                   before_distribution=dist((.5, .5), support=(1, 1))),
])
def test_malformed_or_ambiguous_case_is_rejected(case):
    with pytest.raises(ValueError):
        evaluate_behavioral_edit([case])


def test_duplicate_case_and_measure_names_are_rejected():
    case = BehavioralCase("id", "target", observation("a"), observation("b"))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_behavioral_edit([case, case])
    repeated = BehavioralCase("row", "target", observation("a"), observation("b"),
                              measures=(ScalarComparison("score", 1, 2), ScalarComparison("score", 2, 3)))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_behavioral_edit([repeated])


def test_nonfinite_inputs_are_reported_without_nonstandard_json_numbers():
    case = BehavioralCase("invalid", "target", observation("a"), BehavioralObservation(status="invalid", reason="nonfinite"),
                          before_distribution=dist((.5, .3, .2)), after_distribution=dist((math.inf, 0, 0)),
                          measures=(ScalarComparison("nan", math.nan, 1),
                                    ScalarComparison("overflow", -1e308, 1e308)))
    report = evaluate_behavioral_edit([case])
    row = report["cases"][0]
    assert row["distribution"]["status"] == "invalid_after"
    assert row["distribution"]["after_invalid_reason"] == "nonfinite"
    assert row["measures"]["nan"]["before_status"] == "invalid_nonfinite"
    assert row["measures"]["overflow"]["delta_status"] == "unrepresentable"
    json.dumps(report, allow_nan=False)


def test_inconsistent_rollback_records_are_rejected():
    case = BehavioralCase("id", "target", observation("a"), observation("b"))
    with pytest.raises(ValueError, match="not-attempted"):
        evaluate_behavioral_edit([case], rollback=RollbackEvidence(False, after=StateDigest("weights", "a" * 64)))
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_behavioral_edit([case], rollback=RollbackEvidence(True, StateDigest("weights", "invalid")))


def test_distribution_normalization_is_recorded_and_small_rounding_is_adjusted():
    case = BehavioralCase("id", "locality", observation(0), observation(0),
                          before_distribution=dist((.5, .3, .1999998)),
                          after_distribution=dist((.5, .3, .2)))
    compared = evaluate_behavioral_edit([case])["cases"][0]["distribution"]
    assert compared["status"] == "compared"
    assert compared["before_probability_sum"] == pytest.approx(.9999998)
    assert compared["total_variation"] > 0


def test_outcome_unavailability_is_not_hidden_by_a_paired_reference_rate():
    case = BehavioralCase("id", "locality", observation("correct"), BehavioralObservation(status="tied"),
                          references=("correct",))
    row = evaluate_behavioral_edit([case])
    assert row["cases"][0]["before_reference_match"] is True
    assert row["cases"][0]["after_reference_match"] is None
    assert row["summaries"]["locality"]["paired_reference_count"] == 0
    assert row["summaries"]["locality"]["reference_match_delta"] is None


def test_invalid_finite_probability_sum_overflow_stays_a_reported_invalid_record():
    case = BehavioralCase("id", "target", observation(0), observation(1),
                          before_distribution=dist((.5, .5), support=(0, 1)),
                          after_distribution=dist((1e308, 1e308), support=(0, 1)))
    distribution = evaluate_behavioral_edit([case])["cases"][0]["distribution"]
    assert distribution["status"] == "invalid_after"
    assert distribution["after_invalid_reason"] == "outside_unit_interval"
    assert distribution["after_probability_sum"] is None
    assert distribution["total_variation"] is None


def test_equal_value_does_not_hide_changed_tokens_or_termination_after_rollback():
    before = observation("answer", ids=(10, 1), text="answer", termination="eos")
    edited = observation("answer", ids=(10,), text="answer", termination="length_limit")
    restored = observation("answer", ids=(10,), text="answer", termination="length_limit")
    case = BehavioralCase("id", "locality", before, edited, restored=restored)
    row = evaluate_behavioral_edit([case])["cases"][0]
    assert row["outcome_value_changed"] is False
    assert row["token_ids_changed"] is True
    assert row["decoded_text_changed"] is False
    assert row["termination_changed"] is True
    assert row["restored_value_matches_before"] is True
    assert row["restored_token_ids_match_before"] is False
    assert row["restored_decoded_text_matches_before"] is True
    assert row["restored_termination_matches_before"] is False


def test_absent_or_invalid_metadata_is_not_reported_as_equal():
    before = observation("answer", ids=(10, 1), text="answer", termination="eos")
    after = observation("answer", text="answer")
    restored = BehavioralObservation(status="invalid", token_ids=(10, 1), decoded_text="answer")
    row = evaluate_behavioral_edit([BehavioralCase("id", "locality", before, after, restored=restored)])["cases"][0]
    assert row["outcome_value_changed"] is False
    assert row["token_ids_changed"] is None and row["termination_changed"] is None
    assert row["restored_token_ids_match_before"] is None
    assert row["restored_decoded_text_matches_before"] is None


def test_boolean_outcomes_are_compared_only_with_boolean_labels():
    case = BehavioralCase("id", "target", observation(False), observation(True),
                          references=(False,), targets=(True,))
    row = evaluate_behavioral_edit([case])["cases"][0]
    assert row["outcome_value_changed"] is True
    assert row["before_reference_match"] is True
    assert row["after_reference_match"] is False
    assert row["after_target_match"] is True
    with pytest.raises(ValueError, match="representation"):
        evaluate_behavioral_edit([BehavioralCase("mixed", "target", observation(True), observation(1))])
