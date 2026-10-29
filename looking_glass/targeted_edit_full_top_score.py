"""Provisional full-vocabulary score for the prospective full-top study.

These calculations do not attest publication, intervention, cleanup, source
identity or custody. The caller must bind and exclusively own the saved rows.
"""

from __future__ import annotations

import math

import numpy as np

from . import targeted_edit_full_top as study


CONDITIONS = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")


def _logsoftmax(row):
    maximum = float(np.max(row))
    shifted = row - maximum
    denominator = math.log(float(np.exp(shifted).sum(dtype=np.float64)))
    result = shifted - denominator
    if not bool(np.isfinite(result).all()):
        raise study.StudyError("nonfinite log-softmax")
    return result


def _accepted_nll(logp, accepted):
    selected = logp[np.array(accepted, dtype=np.int64)]
    maximum = float(np.max(selected))
    return -(maximum + math.log(float(np.exp(selected - maximum).sum(dtype=np.float64))))


def _locality(locality, width):
    if type(locality) not in (list, tuple) or len(locality) != 12:
        raise study.StudyError("twelve frozen locality descriptions are required")
    result = []
    seen = set()
    for entry in locality:
        if type(entry) is not dict:
            raise study.StudyError("malformed locality description")
        name, stratum = entry.get("case_id"), entry.get("stratum")
        expected, accepted = entry.get("expected_id"), entry.get("accepted_ids")
        if (type(name) is not str or not name or name in seen or type(stratum) is not str or not stratum
                or type(expected) is not int or not 0 <= expected < width
                or type(accepted) not in (list, tuple) or not accepted
                or any(type(token) is not int or not 0 <= token < width for token in accepted)
                or len(set(accepted)) != len(accepted) or expected not in accepted):
            raise study.StudyError("invalid or repeated locality label")
        seen.add(name)
        result.append({"case_id": name, "stratum": stratum,
                       "expected_id": expected, "accepted_ids": tuple(accepted)})
    return result


def evaluate(matrices, true_id, alternative_id, locality, *, development_top_achieved, match_available):
    """Score four selected targets and twelve fixed locality rows.

    Terminal labels are provisional: the supervised owner and independent
    reader must give source, process, cleanup and publication failures priority.
    """
    if type(development_top_achieved) is not bool or type(match_available) is not bool:
        raise study.StudyError("development and match status must be explicit booleans")
    names = CONDITIONS + (("matched",) if match_available else ())
    if type(matrices) is not dict or set(matrices) != set(names):
        raise study.StudyError("condition inventory differs")
    decoded = {name: study._rows32(matrices[name], 16) for name in names}
    width = decoded["original"].shape[1]
    if any(values.shape != (16, width) for values in decoded.values()):
        raise study.StudyError("condition shapes differ")
    study._token_pair(true_id, alternative_id, width)
    fixed = _locality(locality, width)
    base = decoded["original"]
    if any(study._top(row) != true_id for row in base[:4]):
        raise study.StudyError("selected target baseline is not uniquely factual")
    base_logp = [_logsoftmax(row) for row in base]
    base_top = [study._top_details(row) for row in base]

    summaries = {}
    for name in names:
        rows, target_results, locality_results = decoded[name], [], []
        for i, (row, before) in enumerate(zip(rows, base)):
            logp = _logsoftmax(row)
            kl = float(np.sum(np.exp(base_logp[i]) * (base_logp[i] - logp), dtype=np.float64))
            if not math.isfinite(kl):
                raise study.StudyError("nonfinite locality or target KL")
            details = study._top_details(row)
            before_margin = float(before[alternative_id] - before[true_id])
            margin = float(row[alternative_id] - row[true_id])
            common = {"offset": i, "baseline_top": base_top[i], "condition_top": details,
                      "factual_logit": float(row[true_id]), "alternative_logit": float(row[alternative_id]),
                      "factual_logp": float(logp[true_id]), "alternative_logp": float(logp[alternative_id]),
                      "alternative_logp_gain": float(logp[alternative_id] - base_logp[i][alternative_id]),
                      "baseline_margin": before_margin, "margin": margin, "margin_gain": margin - before_margin,
                      "kl": kl}
            if i < 4:
                target_results.append({**common, "alternative_unique_top": details["unique_top_id"] == alternative_id,
                                       "top_class": study._category(row, true_id, alternative_id)})
                continue
            case = fixed[i - 4]
            original_expected_nll = -float(base_logp[i][case["expected_id"]])
            expected_nll = -float(logp[case["expected_id"]])
            original_accepted_nll = _accepted_nll(base_logp[i], case["accepted_ids"])
            accepted_nll = _accepted_nll(logp, case["accepted_ids"])
            locality_results.append({**common, **case,
                                     "baseline_accepted": base_top[i]["unique_top_id"] in case["accepted_ids"],
                                     "condition_accepted": details["unique_top_id"] in case["accepted_ids"],
                                     "original_expected_nll": original_expected_nll, "expected_nll": expected_nll,
                                     "expected_nll_change": expected_nll - original_expected_nll,
                                     "original_accepted_nll": original_accepted_nll, "accepted_nll": accepted_nll,
                                     "accepted_nll_change": accepted_nll - original_accepted_nll})
        mean_kl = math.fsum(item["kl"] for item in locality_results) / len(locality_results)
        strata = {}
        for stratum in sorted({item["stratum"] for item in locality_results}):
            group = [item for item in locality_results if item["stratum"] == stratum]
            strata[stratum] = {"count": len(group), "mean_kl": math.fsum(item["kl"] for item in group) / len(group),
                               "mean_expected_nll_change": math.fsum(item["expected_nll_change"] for item in group) / len(group),
                               "mean_accepted_nll_change": math.fsum(item["accepted_nll_change"] for item in group) / len(group)}
        correct = [item for item in locality_results if item["baseline_accepted"]]
        summaries[name] = {"targets": target_results, "locality": locality_results,
                           "locality_strata": strata,
                           "alternative_top_count": sum(item["alternative_unique_top"] for item in target_results),
                           "baseline_accepted_count": len(correct),
                           "all_baseline_accepted_preserved": (all(item["condition_accepted"] for item in correct)
                                                               if correct else None),
                           "mean_locality_kl": mean_kl,
                           "max_locality_kl": max(item["kl"] for item in locality_results),
                           "max_accepted_nll_change": max(item["accepted_nll_change"] for item in locality_results)}

    p, u, random = (summaries[name] for name in ("projected", "unprojected", "projected_random"))
    common = {"development_top_achieved": development_top_achieved,
              "at_least_two_counterfactual_tops": p["alternative_top_count"] >= 2,
              "more_tops_than_random": p["alternative_top_count"] > random["alternative_top_count"],
              "baseline_accepted_preserved": p["all_baseline_accepted_preserved"],
              "mean_kl_at_most_0_02": p["mean_locality_kl"] <= 0.02,
              "max_kl_at_most_0_10": p["max_locality_kl"] <= 0.10,
              "max_accepted_nll_change_at_most_0_20": p["max_accepted_nll_change"] <= 0.20}

    def comparison(control):
        return {**common, "kl_ratio": p["mean_locality_kl"] <= 0.9 * control["mean_locality_kl"],
                "kl_difference": control["mean_locality_kl"] - p["mean_locality_kl"] >= 1e-6}

    equal_criteria = comparison(u)
    equal_reasons = ([] if p["baseline_accepted_count"] else ["no baseline-accepted locality"])
    equal_status = "ineligible" if equal_reasons else ("positive" if all(equal_criteria.values()) else "negative")
    match_reasons = equal_reasons.copy()
    balance = None
    matched_criteria = None
    if not match_available:
        match_reasons.append("no development match")
    else:
        balance = study.evaluated_strength_balance(matrices["original"][:4], matrices["projected"][:4],
                                                   matrices["matched"][:4], true_id, alternative_id)
        if not balance["balanced"]:
            match_reasons.append("evaluated strength imbalance")
        matched_criteria = comparison(summaries["matched"])
    matched_status = ("ineligible" if match_reasons else
                      ("positive" if all(matched_criteria.values()) else "negative"))
    return {"version": "targeted-edit-full-top-score-v1", "conditions": summaries,
            "equal_norm": {"status": equal_status, "ineligibility_reasons": equal_reasons, "criteria": equal_criteria},
            "matched": {"status": matched_status, "ineligibility_reasons": match_reasons,
                        "criteria": matched_criteria, "evaluated_balance": balance},
            "scope": "provisional saved-row math; execution and cleanup require independent verification"}
