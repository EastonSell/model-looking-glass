"""Model-free locality-distribution summaries."""
import math

import numpy as np

import flan_weight_edit_results as results


def test_equal_distributions_have_zero_drift():
    row = np.array([2.0, -2.0, 1.0], dtype=np.float32)
    assert results.categorical_drift(row, row) == {"kl": 0.0, "tv": 0.0}


def test_two_way_distribution_drift_matches_closed_form():
    baseline = np.array([0.0, 0.0], dtype=np.float32)
    changed = np.array([math.log(3.0), 0.0], dtype=np.float32)
    outcome = results.categorical_drift(baseline, changed)
    p = math.exp(float(changed[0])) / (math.exp(float(changed[0])) + 1)
    assert abs(outcome["tv"] - (p - 0.5)) < 1e-14
    assert abs(outcome["kl"] - (-math.log(2) - 0.5 * math.log(p * (1 - p)))) < 1e-14


def test_structural_scope_stops_at_divergence_or_edited_decoder_token():
    def span(prefix, sha):
        return {"metadata": {"decoder_prefix": prefix}, "sha256": sha}
    before = [span([0], "a"), span([0, 1627], "b"), span([0, 1627, 1], "c")]
    after = [span([0], "a"), span([0, 1627], "b"), span([0, 1627, 1], "c")]
    checked = results.greedy_scope(before, after, structural_input=True)
    assert [item["scope"] for item in checked] == ["matched", "edited-token-in-prefix", "after-noncomparable"]
    assert checked[0]["equal_raw_logits"] is True
    changed = [span([0], "a"), span([0, 4], "b")]
    checked = results.greedy_scope(before, changed, structural_input=True)
    assert checked[1]["scope"] == "divergent-prefix"
    assert results.greedy_scope(before, after, structural_input=False)[0]["scope"] == "no-structural-claim"


def test_structural_unequal_logits_are_incomplete():
    import pytest
    with pytest.raises(ValueError, match="structural identity"):
        results.require_structural([{"scope": "matched", "equal_raw_logits": False}])
    results.require_structural([{"scope": "divergent-prefix", "equal_raw_logits": None}])


def test_unrelated_locality_requires_both_candidate_spans():
    import pytest
    case = {"expected": "wonderful", "input_ids": [1627]}
    with pytest.raises(ValueError, match="candidate logits are missing"):
        results.check_locality_identity(case, "primary", {"candidate_span": None}, {"candidate_span": None})
