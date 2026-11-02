"""Provisional scoring for the fixed screened projected-edit comparison.

The caller must bind the published baseline screen, fixed cases and discovery
match, own all arrays while they are copied, and separately verify execution
and cleanup. This module does not authorize model forwards or certify a run.
"""
import hashlib
import json
import math

import numpy as np

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2
from . import targeted_edit_screened_screen as screen


BASE_CONDITIONS = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")


def cases(document, offsets):
    """Return the frozen selected targets followed by all twelve localities."""
    targets = screen._targets(document)
    if (type(offsets) is not list or len(offsets) > 8 or len(set(offsets)) != len(offsets)
            or any(type(i) is not int or not 0 <= i < 32 for i in offsets)):
        raise v1.StudyError("selected target offsets differ")
    rows = json.loads(document)["cases"]
    result = []
    for i in offsets:
        row = rows[i]
        result.append((row["case_id"], "target", row["stratum"], row["prompt"],
                       tuple(targets[i]["token_ids"]), None))
    for row in rows[32:]:
        labels = row["continuations"]
        ids = row["token_ids"]
        if (row["role"] != "locality" or type(ids) is not list or not ids
                or any(type(x) is not int or not 0 <= x < v1.VOCAB_SIZE for x in ids)
                or len(labels) != 1 or len(labels[0]["ids"]) != 1
                or type(labels[0]["ids"][0]) is not int
                or not 0 <= labels[0]["ids"][0] < v1.VOCAB_SIZE):
            raise v1.StudyError("fixed locality differs")
        result.append((row["case_id"], "locality", row["stratum"], row["prompt"],
                       tuple(ids), (labels[0]["text"], labels[0]["ids"][0])))
    if len(result) != len(offsets) + 12 or len({r[0] for r in result}) != len(result):
        raise v1.StudyError("fixed evaluation rows differ")
    return tuple(result)


def _mean(values):
    return None if not values else float(math.fsum(values) / len(values))


def _summary(scores, fixed, name):
    targets, locality = [], []
    for index, (case_id, role, stratum, _prompt, _ids, label) in enumerate(fixed):
        base_z, base_logp, base_top = scores["original"][index]
        edit_z, edit_logp, edit_top = scores[name][index]
        kl = float(np.sum(np.exp(base_logp) * (base_logp - edit_logp), dtype=np.float64))
        original_margin = float(base_z[v1.LONDON] - base_z[v1.PARIS])
        condition_margin = float(edit_z[v1.LONDON] - edit_z[v1.PARIS])
        common = {"case_id": case_id, "stratum": stratum, "original_top": base_top,
                  "condition_top": edit_top, "kl": kl,
                  "original_margin": original_margin, "condition_margin": condition_margin,
                  "original_london_nll": float(-base_logp[v1.LONDON]),
                  "condition_london_nll": float(-edit_logp[v1.LONDON]),
                  "original_paris_nll": float(-base_logp[v1.PARIS]),
                  "condition_paris_nll": float(-edit_logp[v1.PARIS])}
        if role == "target":
            initial, changed = original_margin, condition_margin
            if not initial < 0:
                raise v1.StudyError("selected target is not baseline-negative")
            targets.append({**common, "change": changed - initial, "reversed": changed > 0})
        else:
            initial = float(-base_logp[label[1]])
            changed = float(-edit_logp[label[1]])
            locality.append({**common, "expected_text": label[0], "expected_id": label[1],
                             "original_nll": initial, "condition_nll": changed,
                             "nll_deterioration": changed - initial})
    if len(locality) != 12:
        raise v1.StudyError("locality count differs")
    strata = {}
    for stratum in sorted({r["stratum"] for r in locality}):
        group = [r for r in locality if r["stratum"] == stratum]
        strata[stratum] = {"count": len(group), "mean_kl": _mean([r["kl"] for r in group]),
                           "mean_nll_deterioration": _mean([r["nll_deterioration"] for r in group])}
    return {"targets": targets, "locality": locality, "locality_strata": strata,
            "mean_target_change": _mean([r["change"] for r in targets]),
            "reversed_targets": None if not targets else sum(r["reversed"] for r in targets),
            "all_top_preserved": all(r["original_top"] == r["condition_top"] for r in locality),
            "mean_kl": _mean([r["kl"] for r in locality]),
            "max_kl": max(r["kl"] for r in locality),
            "mean_nll_deterioration": _mean([r["nll_deterioration"] for r in locality]),
            "max_nll_deterioration": max(r["nll_deterioration"] for r in locality)}


def evaluate(logits, document, offsets, match_available):
    """Compute the preset two decisions from private, caller-owned matrices."""
    if type(match_available) is not bool:
        raise v1.StudyError("discovery-match status differs")
    fixed = cases(document, offsets)
    names = BASE_CONDITIONS + (("matched",) if match_available else ())
    before = v1._denormal_probe()
    # _scores snapshots the input arrays and validates all full-vocabulary rows.
    scored = v2._scores(logits, fixed, names)
    summaries = {name: _summary(scored, fixed, name) for name in names}
    p, u, random = (summaries[name] for name in ("projected", "unprojected", "projected_random"))
    common = {"at_least_two_reversals": None if not offsets else p["reversed_targets"] >= 2,
              "mean_target_gain": None if not offsets else p["mean_target_change"] > 0.25,
              "random_advantage": None if not offsets else p["mean_target_change"] - random["mean_target_change"] >= 0.10,
              "all_locality_top_preserved": p["all_top_preserved"], "mean_locality_kl": p["mean_kl"] <= 0.02,
              "max_locality_kl": p["max_kl"] <= 0.10,
              "mean_locality_nll_deterioration": p["mean_nll_deterioration"] <= 0.05,
              "max_locality_nll_deterioration": p["max_nll_deterioration"] <= 0.20}
    def advantage(control):
        return {"projected_kl_ratio": p["mean_kl"] <= 0.9 * control["mean_kl"],
                "projected_kl_difference": control["mean_kl"] - p["mean_kl"] >= 1e-6}
    equal = {**common, **advantage(u)}
    matched = None
    if match_available:
        m = summaries["matched"]
        if offsets:
            gap = abs(p["mean_target_change"] - m["mean_target_change"])
            tolerance = max(0.05, 0.25 * max(abs(p["mean_target_change"]), abs(m["mean_target_change"])))
            gain_floor, close = m["mean_target_change"] > 0.25, gap <= tolerance
        else:
            gap = tolerance = gain_floor = close = None
        matched = {**common, **advantage(m), "unprojected_target_gain": gain_floor,
                   "evaluated_target_gain_match": close, "evaluated_target_gap": gap,
                   "evaluated_target_tolerance": tolerance}
    eligible = len(offsets) == 8
    equal_decision = "ineligible" if not eligible else ("success" if all(equal.values()) else "negative")
    matched_decision = ("ineligible" if not eligible or not match_available else
                        ("success" if all(value for key, value in matched.items()
                                          if key not in ("evaluated_target_gap", "evaluated_target_tolerance")) else "negative"))
    after = v1._denormal_probe()
    return {"version": "targeted-edit-screened-score-v1", "case_sha256": hashlib.sha256(document).hexdigest(),
            "selected_count": len(offsets), "target_count_eligible": eligible,
            "discovery_match_available": match_available, "conditions": summaries,
            "equal_norm_criteria": equal, "matched_discovery_criteria": matched,
            "provisional_equal_norm_decision": equal_decision,
            "provisional_matched_discovery_decision": matched_decision,
            "denormal_probe_before": before, "denormal_probe_after": after,
            "scope": "numerical result only; execution and cleanup require independent verification"}
