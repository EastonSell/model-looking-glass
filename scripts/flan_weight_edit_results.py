"""Descriptive locality drift from already saved first-step vocabulary rows."""
import hashlib
import math
from pathlib import Path

import numpy as np

import flan_weight_edit_numerical as science


def categorical_drift(before, after):
    if (before.ndim != 1 or after.shape != before.shape or not np.isfinite(before).all()
            or not np.isfinite(after).all()):
        raise ValueError("finite aligned full-vocabulary rows are required")
    a, b = before.astype(np.float64), after.astype(np.float64)
    max_a, max_b = float(np.max(a)), float(np.max(b))
    exp_a, exp_b = np.exp(a - max_a), np.exp(b - max_b)
    sum_a, sum_b = math.fsum(map(float, exp_a)), math.fsum(map(float, exp_b))
    log_a, log_b = math.log(sum_a), math.log(sum_b)
    p, q = exp_a / sum_a, exp_b / sum_b
    tv = 0.5 * math.fsum(abs(float(x) - float(y)) for x, y in zip(p, q))
    kl = math.fsum(float(x) * math.fsum((float(za), -max_a, -log_a, -float(zb), max_b, log_b))
                   for x, za, zb in zip(p, a, b))
    return {"kl": kl, "tv": tv}


def greedy_scope(before, after, *, structural_input):
    result, comparable = [], bool(structural_input)
    for index in range(max(len(before), len(after))):
        a = before[index] if index < len(before) else None
        b = after[index] if index < len(after) else None
        if not structural_input:
            status = "no-structural-claim"
        elif not comparable:
            status = "after-noncomparable"
        elif a is None or b is None:
            status, comparable = "missing-step", False
        elif a["metadata"]["decoder_prefix"] != b["metadata"]["decoder_prefix"]:
            status, comparable = "divergent-prefix", False
        elif 1627 in a["metadata"]["decoder_prefix"]:
            status, comparable = "edited-token-in-prefix", False
        else:
            status = "matched"
        result.append({"step": index, "scope": status,
                       "baseline_prefix": None if a is None else a["metadata"]["decoder_prefix"],
                       "edited_prefix": None if b is None else b["metadata"]["decoder_prefix"],
                       "equal_raw_logits": a["sha256"] == b["sha256"] if status == "matched" else None})
    return result


def require_structural(comparisons):
    if any(item["scope"] == "matched" and item["equal_raw_logits"] is not True for item in comparisons):
        raise ValueError("structural identity logits differ")


def check_locality_identity(case, arm, baseline, changed):
    structural = 1627 not in case.get("input_ids", []) and arm in ("primary", "same", "random", "opposite")
    if case["expected"] not in ("positive", "negative") and (baseline.get("candidate_span") is None
                                                                  or changed.get("candidate_span") is None):
        raise ValueError("locality candidate logits are missing")
    greedy = greedy_scope(baseline["greedy_spans"], changed["greedy_spans"], structural_input=structural)
    teachers = {}
    pairs = [(label, baseline["label_spans"][label], changed["label_spans"][label])
             for label in ("positive", "negative")]
    if baseline.get("candidate_span") is not None and changed.get("candidate_span") is not None:
        pairs.append(("candidate", baseline["candidate_span"], changed["candidate_span"]))
    for label, before, after in pairs:
        teachers[label] = greedy_scope([before], [after], structural_input=structural)[0]
    require_structural(greedy)
    require_structural(list(teachers.values()))
    return structural, greedy, teachers


def summarize(cases, evaluation, logit_path):
    def row(stream, span):
        if span.get("shape") != [32128] or span.get("bytes") != 32128 * 4:
            raise ValueError("first-step logit span differs")
        stream.seek(span["offset"])
        raw = stream.read(span["bytes"])
        if len(raw) != span["bytes"] or hashlib.sha256(raw).hexdigest() != span["sha256"]:
            raise ValueError("first-step logit bytes differ")
        return np.frombuffer(raw, dtype="<f4")
    def expected_score(case, outcome):
        terms = outcome["label_terms"][case["expected"]] if case["expected"] in ("positive", "negative") else outcome["candidate_terms"]
        return science.sequence_score(terms)
    result = []
    with Path(logit_path).open("rb") as stream:
        for case in cases[16:]:
            baseline = evaluation["baseline"][case["id"]]
            original = row(stream, baseline["greedy_spans"][0])
            for name, outcomes in evaluation["arms"].items():
                changed = outcomes[case["id"]]
                metric = categorical_drift(original, row(stream, changed["greedy_spans"][0]))
                structural, greedy_comparisons, teachers = check_locality_identity(case, name, baseline, changed)
                result.append({"case_id": case["id"], "arm": name,
                               "input_contains_edited_token": 1627 in case["input_ids"],
                               "first_step_structural_identity_expected": structural,
                               "greedy_prefix_comparisons": greedy_comparisons,
                               "teacher_prefix_comparisons": teachers,
                               "baseline_correct": baseline["strict_correct"],
                               "changed_correct": changed["strict_correct"],
                               "baseline_expected_log_probability": expected_score(case, baseline),
                               "changed_expected_log_probability": expected_score(case, changed),
                               "baseline_ids": baseline["greedy_ids"], "changed_ids": changed["greedy_ids"],
                               **metric})
    return result
