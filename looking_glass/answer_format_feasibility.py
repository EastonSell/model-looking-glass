"""Pure saved-logit comparisons for the answer-format feasibility study.

This module does not load a model or authenticate input provenance. Callers
must bind the fixed cases, source and each phase's saved rows before use.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np

from . import targeted_edit_full_top as full_top


FACTS = ("france", "japan", "italy", "uk")
FAMILIES = ("statement", "question")
PREFIXES = ("none", "neutral", "spain", "sweden")
LABELS = ("factual", "counterfactual", "madrid", "stockholm")
SEEDS = {"development": 20261031, "heldout": 20261101}
PAIRS = tuple(itertools.combinations(PREFIXES, 2))


class StudyError(ValueError):
    """Invalid evidence is incomplete, not a numerical outcome."""


def execution_order(stage):
    if stage not in SEEDS:
        raise StudyError("unknown phase")
    result = list(range(32))
    random.Random(SEEDS[stage]).shuffle(result)
    return result


def _validate_cases(rows, stage, width):
    if stage not in SEEDS or type(rows) not in (list, tuple) or len(rows) != 32:
        raise StudyError("a phase requires exactly 32 ordered cases")
    expected = [(fact, family, prefix) for fact in FACTS
                for family in FAMILIES for prefix in PREFIXES]
    ids, fact_labels = [], {}
    for row, triple in zip(rows, expected):
        if (type(row) is not dict or row.get("role") != stage
                or (row.get("fact_id"), row.get("family"), row.get("prefix")) != triple
                or type(row.get("case_id")) is not str or not row["case_id"]):
            raise StudyError("case role, identity or order differs")
        labels = row.get("labels")
        if type(labels) is not dict or set(labels) != set(LABELS):
            raise StudyError("case label inventory differs")
        token_ids = []
        for label in LABELS:
            item = labels[label]
            if (type(item) is not dict or type(item.get("token_id")) is not int
                    or not 0 <= item["token_id"] < width):
                raise StudyError("a label token ID differs")
            token_ids.append(item["token_id"])
        if len(set(token_ids)) != 4:
            raise StudyError("the four compared labels must be distinct")
        if row["fact_id"] in fact_labels and fact_labels[row["fact_id"]] != token_ids:
            raise StudyError("a fact's label token IDs vary across cases")
        fact_labels[row["fact_id"]] = token_ids
        ids.append(row["case_id"])
    if len(set(ids)) != 32:
        raise StudyError("case identifiers repeat")


def _log_distribution(row):
    top = float(np.max(row))
    shifted = row - top
    total = float(np.exp(shifted).sum(dtype=np.float64))
    if not math.isfinite(total) or total < 1:
        raise StudyError("invalid full-vocabulary normalization")
    log_total = math.log(total)
    return shifted - log_total, top, log_total


def _pair(reference, comparison):
    with np.errstate(under="ignore"):
        p, q = np.exp(reference), np.exp(comparison)
    kl = float(np.sum(p * (reference - comparison), dtype=np.float64))
    tv = float(np.sum(np.abs(p - q), dtype=np.float64) / 2.0)
    if not math.isfinite(kl) or not math.isfinite(tv) or kl < -1e-12 or not 0 <= tv <= 1 + 1e-12:
        raise StudyError("invalid distribution comparison")
    return max(0.0, kl), min(1.0, tv)


def _validate_development(previous):
    if (type(previous) is not dict or previous.get("stage") != "development"
            or type(previous.get("families")) is not list or len(previous["families"]) != 2):
        raise StudyError("development selection is unavailable")
    for family, row in zip(FAMILIES, previous["families"]):
        if type(row) is not dict:
            raise StudyError("development family summary differs")
        facts = row.get("robust_facts")
        if (row.get("family") != family or type(facts) is not list
                or len(set(facts)) != len(facts) or any(f not in FACTS for f in facts)
                or facts != [f for f in FACTS if f in facts]
                or row.get("eligible") is not (len(facts) >= 3)):
            raise StudyError("development family summary differs")
    chosen = next((row for row in previous["families"] if row["eligible"]), None)
    expected = None if chosen is None else chosen["family"]
    if previous.get("selected_family") != expected:
        raise StudyError("development selection differs")
    return chosen


def summarize(stage, raw_rows, cases, *, development=None):
    """Score saved f32 rows in fixed execution order, never run a model."""
    values = full_top._rows32(raw_rows, 32)
    _validate_cases(cases, stage, values.shape[1])
    order = execution_order(stage)
    logps, normalizers, observations, by_key = {}, {}, [], {}
    for execution_index, case_index in enumerate(order):
        case, row = cases[case_index], values[execution_index]
        logs, maximum, log_total = _log_distribution(row)
        details = full_top._top_details(row)
        label_ids = {k: case["labels"][k]["token_id"] for k in LABELS}
        scores = {k: {"token_id": token, "logit": float(row[token]),
                      "logp": float(logs[token])} for k, token in label_ids.items()}
        if not all(math.isfinite(x) for v in scores.values() for x in (v["logit"], v["logp"])):
            raise StudyError("nonfinite label score")
        observation = {"execution_index": execution_index, "canonical_index": case_index,
                       "case_id": case["case_id"], "fact_id": case["fact_id"],
                       "family": case["family"], "prefix": case["prefix"],
                       **details, "scores": scores,
                       "factual_top": details["unique_top_id"] == label_ids["factual"]}
        observations.append(observation)
        key = (case["fact_id"], case["family"], case["prefix"])
        by_key[key], logps[key], normalizers[key] = observation, logs, (maximum, log_total)
    families = []
    for family in FAMILIES:
        robust = [fact for fact in FACTS if all(
            by_key[(fact, family, prefix)]["factual_top"] for prefix in PREFIXES)]
        families.append({"family": family, "robust_facts": robust, "eligible": len(robust) >= 3})
    pairs, contrasts, contrast_terms = [], [], {}
    for fact in FACTS:
        for family in FAMILIES:
            for left, right in PAIRS:
                a, b = by_key[(fact, family, left)], by_key[(fact, family, right)]
                la, lb = logps[(fact, family, left)], logps[(fact, family, right)]
                max_a, norm_a = normalizers[(fact, family, left)]
                max_b, norm_b = normalizers[(fact, family, right)]
                kl, tv = _pair(la, lb)
                relation = ("tied" if a["unique_top_id"] is None or b["unique_top_id"] is None
                            else "same" if a["unique_top_id"] == b["unique_top_id"] else "changed")
                pairs.append({"fact_id": fact, "family": family, "reference": left, "comparison": right,
                              "reference_case_id": a["case_id"], "comparison_case_id": b["case_id"],
                              "top_relation": relation, "kl": kl, "total_variation": tv,
                              "logp_delta": {label: math.fsum((b["scores"][label]["logit"],
                                  -a["scores"][label]["logit"], -max_b, max_a, -norm_b, norm_a))
                                  for label in LABELS}})
            s, p = by_key[(fact, family, "spain")], by_key[(fact, family, "sweden")]
            # The two log-softmax normalizers cancel. Summing the exact f32
            # logits after promotion avoids losing the sign to cancellation
            # when an unrelated vocabulary logit is extremely large.
            terms = (s["scores"]["madrid"]["logit"], -p["scores"]["madrid"]["logit"],
                     -s["scores"]["stockholm"]["logit"], p["scores"]["stockholm"]["logit"])
            contrast_terms[(fact, family)] = terms
            d = math.fsum(terms)
            if not math.isfinite(d):
                raise StudyError("nonfinite example contrast")
            contrasts.append({"fact_id": fact, "family": family, "d": d,
                              "spain_top_madrid": s["unique_top_id"] == s["scores"]["madrid"]["token_id"],
                              "sweden_top_stockholm": p["unique_top_id"] == p["scores"]["stockholm"]["token_id"]})
    result = {"stage": stage, "execution_order": order, "observations": observations,
              "families": families, "pairs": pairs, "contrasts": contrasts}
    if stage == "development":
        chosen = next((row for row in families if row["eligible"]), None)
        result["selected_family"] = None if chosen is None else chosen["family"]
        result["status"] = "eligible" if chosen else "ineligible"
    else:
        chosen = _validate_development(development)
        result["selected_family"] = None if chosen is None else chosen["family"]
        heldout_robust = [] if chosen is None else next(
            row["robust_facts"] for row in families if row["family"] == chosen["family"])
        reproduced = [] if chosen is None else [f for f in chosen["robust_facts"] if f in heldout_robust]
        result["reproduced_development_facts"] = reproduced
        result["status"] = "ineligible" if chosen is None else ("positive" if len(reproduced) >= 3 else "negative")
        means = [{"fact_id": fact, "mean_d": math.fsum(
                      term for family in FAMILIES for term in contrast_terms[(fact, family)]) / len(FAMILIES)}
                 for fact in FACTS]
        result["heldout_fact_means"] = means
        result["descriptive_direction_met"] = sum(row["mean_d"] > 0 for row in means) >= 3
    return result

