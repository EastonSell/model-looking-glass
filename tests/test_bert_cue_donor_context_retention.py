"""Boundaries and denominators for matched donor-context retention."""
import importlib.util
import math
from pathlib import Path

import pytest

from looking_glass import bert_cue_donor_context_retention as study
from looking_glass.bert_cue_context_transfer import CUES, SITES, donors


def row(*, q=0.5, baseline=True, within=0.04, alternate=0.03):
    baselines = [{"context": ci, "cue": cue, "any_minus_some":
                  (2.0 if cue in CUES[:2] else 0.0) + ci}
                 for ci in range(2) for cue in CUES]
    distances, effects = [], []
    for site in SITES:
        for ci in range(2):
            for cue in CUES:
                for di, donor in donors(ci, cue):
                    key = {"site": site, "target_context": ci, "target_cue": cue,
                           "donor_context": di, "donor_cue": donor}
                    distances.append({**key, "distance": 0.8})
                    if q >= 0.05:
                        is_opposite = (cue in CUES[:2]) != (donor in CUES[:2])
                        direction = -1 if cue in CUES[:2] else 1
                        value = direction * (within if di == ci else alternate) if is_opposite else 0.01
                        effects.append({**key, "delta": value})
    return {"baseline_eligible": baseline, "dose_eligible": q >= 0.05, "q": q,
            "baselines": baselines, "distances": distances, "effects": effects}


def set_first_edges(case, *, within=None, alternate=None, count=1):
    matches = []
    for item in case["effects"]:
        if (item["site"] == "cue2" and item["donor_context"] == item["target_context"]
                and ((item["target_cue"] in CUES[:2]) != (item["donor_cue"] in CUES[:2]))):
            matches.append(item)
    for item in matches[:count]:
        direction = -1 if item["target_cue"] in CUES[:2] else 1
        if within is not None:
            item["delta"] = direction * within
        if alternate is not None:
            counterpart = next(v for v in case["effects"]
                               if v["site"] == "cue2" and v["target_context"] == item["target_context"]
                               and v["target_cue"] == item["target_cue"] and v["donor_cue"] == item["donor_cue"]
                               and v["donor_context"] == 1 - item["target_context"])
            counterpart["delta"] = direction * alternate


def test_exact_edge_boundaries_and_class_orientation():
    result = study.summarize_pair(row(within=0.02, alternate=0.01))
    assert result["evaluable"] and result["informative_edges"] == 16
    assert result["retained_edges"] == 16 and result["primary_status"] == "pass"
    assert result["sensitivity_retained_edges"] == 0 and result["sensitivity_status"] == "fail"
    assert all(edge["within_signed"] == 0.02 and edge["alternate_signed"] == 0.01 for edge in result["edges"])
    full = study.summarize_pair(row(within=0.02, alternate=0.02))
    assert full["sensitivity_status"] == "pass"
    below = study.summarize_pair(row(within=math.nextafter(0.02, 0.0)))
    assert below["informative_edges"] == 0 and all(edge["exclusion"] == "below_floor" for edge in below["edges"])
    zero = study.summarize_pair(row(within=0.0))
    assert zero["informative_edges"] == 0 and all(edge["exclusion"] == "nonpositive" for edge in zero["edges"])


def test_informative_and_passing_edge_denominators():
    case = row()
    set_first_edges(case, within=0.0, count=5)
    assert study.summarize_pair(case)["primary_status"] == "unevaluable"
    case = row()
    set_first_edges(case, within=0.0, count=4)
    result = study.summarize_pair(case)
    assert result["informative_edges"] == result["retained_edges"] == 12
    assert result["primary_status"] == "pass" and result["retained_per_informative"] == 1
    case = row()
    set_first_edges(case, alternate=0.0, count=5)
    assert study.summarize_pair(case)["primary_status"] == "fail"
    case = row()
    set_first_edges(case, alternate=0.0, count=4)
    result = study.summarize_pair(case)
    assert result["retained_edges"] == 12 and result["primary_status"] == "pass"
    assert result["retained_per_16"] == result["retained_per_informative"] == 0.75


def test_baseline_dose_exclusions_keep_raw_values():
    for case in (row(baseline=False), row(q=0.4)):
        result = study.summarize_pair(case)
        assert result["primary_status"] == "unevaluable" and result["retained_edges"] == 16
        assert all(edge["within_signed"] == 0.04 for edge in result["edges"])
    result = study.summarize_pair(row(q=0))
    assert result["primary_status"] == "unevaluable" and result["informative_edges"] is None
    assert all(edge["exclusion"] == "dose_ineligible" and edge["within_signed"] is None for edge in result["edges"])


def test_aggregate_counts_only_evaluable_pairs():
    cases = [row(), row(), row(alternate=0.0), row(q=0.4)]
    pairs = [{"id": ident, "split": "discovery" if i == 0 else "evaluation",
              "retention": study.summarize_pair(case)}
             for i, (ident, case) in enumerate(zip(study.PAIR_IDS, cases))]
    assert study.evaluate(pairs)["primary"] == {"status": "fail", "evaluable_pairs": 2, "passing_pairs": 1}
    pairs[2]["retention"] = study.summarize_pair(row(q=0.4))
    assert study.evaluate(pairs)["primary"]["status"] == "unevaluable"
    pairs.reverse()
    with pytest.raises(ValueError, match="order"):
        study.evaluate(pairs)


def test_incomplete_duplicate_or_nonfinite_witnesses_reject():
    case = row()
    case["distances"].pop()
    with pytest.raises(ValueError, match="distances"):
        study.summarize_pair(case)
    case = row()
    case["effects"].append(case["effects"][0])
    with pytest.raises(ValueError, match="duplicate"):
        study.summarize_pair(case)
    case = row()
    case["effects"][0]["delta"] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        study.summarize_pair(case)
    for collection, field in (("baselines", "context"), ("distances", "target_context"),
                              ("effects", "donor_context")):
        case = row()
        case[collection][0][field] = False
        with pytest.raises(ValueError, match="integer"):
            study.summarize_pair(case)


def test_matched_geometry_known_vectors_and_zero_norm():
    path = Path(__file__).resolve().parents[1] / "scripts/bert_cue_donor_context_geometry.py"
    spec = importlib.util.spec_from_file_location("donor_geometry_test", path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    assert geometry.matched_geometry([0, 0], [3, 4], [-4, 3]) == (5.0, 5.0, 0.0)
    assert geometry.matched_geometry([0, 0], [3, 4], [-3, -4]) == (5.0, 5.0, -1.0)
    assert geometry.matched_geometry([0, 0], [3, 4], [0, 0]) == (5.0, 0.0, None)
    with pytest.raises(ValueError, match="dimensions"):
        geometry.matched_geometry([0, 0], [3], [0, 0])
    edges = study.summarize_pair(row(q=0))["edges"]
    geometry.validate_edges(edges)
    forged = [dict(edge) for edge in edges]
    forged[-1] = dict(forged[0])
    with pytest.raises(ValueError, match="ordered"):
        geometry.validate_edges(forged)
    forged = [dict(edge) for edge in edges]
    forged[0]["target_context"] = False
    with pytest.raises(ValueError, match="type"):
        geometry.validate_edges(forged)
    geometry.numeric_mode()
