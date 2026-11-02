"""Per-prompt matched-dose arithmetic and scores for the copying follow-up."""
from __future__ import annotations

import math
import struct

from looking_glass.copy_head_factorial import (
    MIN_DOSE, SHAPE_BYTES, _bits, _norm, _values, f64_to_f32_bits, head_slice,
)

HEADS = (2, 10, 7)
ARMS = ("h2", "h10", "h7")
TOLERANCE = 1e-5
VOCAB = 50_257


def plan_prompt(noop_before: bytes) -> dict:
    """Return an explicit floor/mismatch decision and the exact delivered bits.

    No division or planned slice is computed until all source values and
    norms are finite and the prompt's common q is strictly above the floor.
    Planned deliveries survive a rounding mismatch for later inspection.
    """
    if type(noop_before) is not bytes or len(noop_before) != SHAPE_BYTES:
        raise ValueError("unexpected no-op input")
    _values(noop_before)  # Check the complete witness, including untouched heads.
    original = {head: head_slice(noop_before, head) for head in HEADS}
    values = {head: _values(original[head]) for head in HEADS}
    norms = {head: _norm(values[head]) for head in HEADS}
    q = min(norms.values())
    if not math.isfinite(q):
        raise ValueError("nonfinite common dose")
    result = {"q": q, "norms": norms, "original": original,
              "status": "dose-ineligible", "reason": "dose floor", "plans": None}
    if q <= MIN_DOSE:
        return result
    plans = {}
    for head in HEADS:
        fraction = q / norms[head]
        after_bits = tuple(f64_to_f32_bits((1.0 - fraction) * value) for value in values[head])
        delivered = struct.pack("<64I", *after_bits)
        after_values = _values(delivered)
        achieved = _norm(after - before for after, before in zip(after_values, values[head]))
        before_bits = _bits(original[head])
        plans[head] = {
            "fraction": fraction, "delivered": delivered, "achieved": achieved,
            "changed_elements": sum(before != after for before, after in zip(before_bits, after_bits)),
            "changed_bits": sum((before ^ after).bit_count() for before, after in zip(before_bits, after_bits)),
        }
    achieved = [plans[head]["achieved"] for head in HEADS]
    valid = (all(abs(value - q) <= TOLERANCE * q for value in achieved)
             and max(achieved) - min(achieved) <= TOLERANCE * q)
    result.update({"status": "eligible" if valid else "dose-ineligible",
                   "reason": None if valid else "realized dose mismatch", "plans": plans})
    return result


def action_order(family_index: int, variant: int) -> tuple[str, ...]:
    if (type(family_index) is not int or not 0 <= family_index < 8
            or type(variant) is not int or not 0 <= variant < 4):
        raise ValueError("invalid family or variant")
    shift = (4 * family_index + variant) % 3
    return ARMS[shift:] + ARMS[:shift]


def logit_stats(raw: bytes, target: int) -> dict:
    values = _values(raw)
    if len(values) != VOCAB or type(target) is not int or not 0 <= target < VOCAB:
        raise ValueError("invalid vocabulary logits or target")
    maximum = max(values)
    logz_shift = math.log(math.fsum(math.exp(value - maximum) for value in values))
    top_count = sum(value == maximum for value in values)
    representative = values.index(maximum)
    return {"values": values, "maximum": maximum, "logz_shift": logz_shift,
            "target_logit": values[target],
            "target_log_probability": (values[target] - maximum) - logz_shift,
            "target_rank": 1 + sum(value > values[target] for value in values),
            "top_unique_id": representative if top_count == 1 else None,
            "top_representative_id": representative, "top_tie_count": top_count}


def compare(clean: dict, arm: dict) -> dict:
    """Compare two finite full-vocabulary rows, preserving the directional KL."""
    if len(clean["values"]) != VOCAB or len(arm["values"]) != VOCAB:
        raise ValueError("invalid probability rows")
    delta = math.fsum((arm["target_logit"], -clean["target_logit"], clean["maximum"],
                       -arm["maximum"], clean["logz_shift"], -arm["logz_shift"]))
    kl_terms, tv_terms = [], []
    for a, b in zip(clean["values"], arm["values"]):
        lp = (a - clean["maximum"]) - clean["logz_shift"]
        lq = (b - arm["maximum"]) - arm["logz_shift"]
        p, q = math.exp(lp), math.exp(lq)
        log_ratio = math.fsum((a, -b, -clean["maximum"], arm["maximum"],
                               -clean["logz_shift"], arm["logz_shift"]))
        kl_terms.append(p * log_ratio)
        tv_terms.append(abs(p - q))
    if clean["top_unique_id"] is None or arm["top_unique_id"] is None:
        top_change = "tied/indeterminate"
    else:
        top_change = "changed" if clean["top_unique_id"] != arm["top_unique_id"] else "unchanged"
    return {"target_logit_delta": arm["target_logit"] - clean["target_logit"],
            "target_log_probability_delta": delta,
            "kl_clean_to_arm": math.fsum(kl_terms),
            "total_variation": 0.5 * math.fsum(tv_terms), "top_change": top_change}


def output_delta_l2(noop_output: bytes, arm_output: bytes) -> float:
    if (type(noop_output) is not bytes or type(arm_output) is not bytes
            or len(noop_output) != SHAPE_BYTES or len(arm_output) != SHAPE_BYTES):
        raise ValueError("invalid projection outputs")
    before = _values(noop_output)
    after = _values(arm_output)
    start = 10 * 768
    return _norm(b - a for a, b in zip(before[start:], after[start:]))


def family_summary(variants: list[dict]) -> dict:
    """Summarize four variants; a family requires four eligible dose plans."""
    if len(variants) != 4:
        raise ValueError("expected four variants")
    if any(row.get("status") not in ("eligible", "dose-ineligible") for row in variants):
        raise ValueError("unknown or invalid prompt status")
    for row in variants:
        if not math.isfinite(row["q"]) or row["q"] < 0:
            raise ValueError("invalid prompt dose")
        if row["status"] == "eligible":
            if row["q"] <= MIN_DOSE:
                raise ValueError("invalid eligible prompt dose")
            if any(not math.isfinite(row["comparisons"][arm][field])
                   for arm in ARMS for field in ("target_log_probability_delta", "target_logit_delta")):
                raise ValueError("nonfinite prompt delta")
    if any(row["status"] != "eligible" for row in variants):
        return {"status": "dose-ineligible", "contrast": None}
    d, d_logit, case_deltas = {}, {}, {}
    for arm in ARMS:
        delta = [row["comparisons"][arm]["target_log_probability_delta"] for row in variants]
        logit_delta = [row["comparisons"][arm]["target_logit_delta"] for row in variants]
        d[arm] = math.fsum((delta[0], -math.fsum(delta[1:]) / 3))
        d_logit[arm] = math.fsum((logit_delta[0], -math.fsum(logit_delta[1:]) / 3))
        case_deltas[arm] = {"target_log_probability": delta, "target_logit": logit_delta}
    normalized = [(row["comparisons"]["h10"]["target_log_probability_delta"]
                   - row["comparisons"]["h2"]["target_log_probability_delta"]) / row["q"]
                  for row in variants]
    return {"status": "evaluable", "D": d, "target_logit_D": d_logit, "case_deltas": case_deltas,
            "contrast": d["h10"] - d["h2"], "target_logit_contrast": d_logit["h10"] - d_logit["h2"],
            "h10_minus_h7": d["h10"] - d["h7"], "h2_minus_h7": d["h2"] - d["h7"],
            "dose_normalized_contrast": math.fsum((normalized[0], -math.fsum(normalized[1:]) / 3))}


def decision(families: list[dict]) -> dict:
    if len(families) != 8:
        raise ValueError("expected eight families")
    if any(row.get("status") not in ("evaluable", "dose-ineligible") for row in families):
        raise ValueError("unknown or invalid family status")
    if any((not math.isfinite(row["contrast"])) if row["status"] == "evaluable"
           else row.get("contrast") is not None for row in families):
        raise ValueError("invalid family contrast")
    evaluable = [row for row in families if row["status"] == "evaluable"]
    count = len(evaluable)
    positives = sum(row["contrast"] > 0 for row in evaluable)
    mean = math.fsum(row["contrast"] for row in evaluable) / count if count else None
    status = ("ineligible" if count < 6 else
              "criterion-met" if mean > 0 and 4 * positives >= 3 * count else "criterion-not-met")
    return {"status": status, "evaluable_families": count,
            "strictly_positive": positives, "mean_contrast": mean}
