"""Exercise the two fixed study schedules without a checkpoint."""

from contextlib import contextmanager

import flan_weight_edit_cases as cases
import flan_weight_edit_study as study


def fake_cases():
    rows = cases.fixed_texts()
    for row in rows:
        row["input_ids"] = [1627] if "wonderful" in row["text"] else [4]
    return rows


class FakeSession:
    def __init__(self, failed_floor=False, underfill=False, wrong_clean_score=False):
        self.failed_floor = failed_floor
        self.underfill = underfill
        self.wrong_clean_score = wrong_clean_score
        self.calls = []
        self.arms = []

    @contextmanager
    def arm(self, name, dose):
        self.arms.append((name, dose))
        yield

    def score(self, name, function):
        return function()

    def evaluate(self, arm, case):
        self.calls.append((arm, case["id"]))
        donor = case["role"].endswith("-donor")
        changed = arm not in ("baseline", "same") and not donor
        if self.failed_floor and arm == "primary":
            gain = 0.25
        elif arm == "baseline" and donor:
            gain = 2.0
        elif changed:
            gain = 2.0
        else:
            gain = 0.0
        if self.wrong_clean_score and arm == "baseline" and case["id"] == "discovery-1-target":
            gain = 3.0
        label = "negative" if donor or changed else "positive"
        if self.underfill and arm == "baseline" and case["id"] in ("evaluation-1-target", "evaluation-2-target"):
            label = "negative"
        def span(sha, prefix):
            return {"sha256": sha, "metadata": {"decoder_prefix": prefix}}
        return {"case_id": case["id"], "arm": arm, "strict_label": label,
                "strict_correct": label == case["expected"],
                "label_terms": {"positive": [(1., 0., 0.)], "negative": [(gain, 0., 0.)]},
                "label_spans": {"positive": span("p", [0, 1465]), "negative": span("n", [0, 2841])},
                "greedy_spans": [span("g", [0])], "greedy_ids": [1],
                "candidate_span": None if case["expected"] in ("positive", "negative") else span("c", [0, 1627])}


def test_development_runs_complete_fixed_grid_and_selects_per_case():
    session = FakeSession()
    result = study.development(fake_cases(), session)
    assert result["status"] == "complete"
    assert result["calibration_eligible"] is True
    assert all(result["selection"][arm]["matched"] for arm in ("broad", "random", "opposite"))
    assert len(session.calls) == 68
    assert len(session.arms) == 29


def test_failed_calibration_still_preserves_grid_and_evaluation():
    session = FakeSession(failed_floor=True, underfill=True)
    development = study.development(fake_cases(), session)
    assert development["calibration_eligible"] is False
    assert len(session.calls) == 68
    result = study.evaluation(fake_cases(), session, development)
    assert result["primary"]["status"] == "ineligible"
    assert len(session.calls) == 68 + 76
    assert len(result["targets"]) == 4 and len(result["localities"]) == 8


def test_correct_greedy_but_wrong_full_sequence_margin_blocks_all_edits():
    session = FakeSession(wrong_clean_score=True)
    result = study.development(fake_cases(), session)
    assert result["status"] == "ineligible"
    assert len(session.calls) == 8
    assert session.arms == []


def test_token_absent_locality_mismatch_stops_later_cases():
    import pytest
    class Mismatch(FakeSession):
        def evaluate(self, arm, case):
            result = super().evaluate(arm, case)
            if arm == "primary" and case["id"] == "locality-1":
                result["greedy_spans"][0]["sha256"] = "different"
            return result
    session = Mismatch()
    fixed = fake_cases()
    development = study.development(fixed, session)
    with pytest.raises(ValueError, match="structural identity"):
        study.evaluation(fixed, session, development)
    assert session.calls[-1] == ("primary", "locality-1")
