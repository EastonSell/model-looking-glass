"""Independent model-free primitives for the prospective FLAN sentiment audit.

These functions consume saved numeric bytes, never load a model, and do not
establish that an execution took place.  This is a work-in-progress core;
the per-pass journal and manifest bindings will be added after their schema is
fixed.  Exact float32 log-softmax intentionally uses the same pinned CPU torch
primitive as the protocol, rather than claiming bitwise equivalence of a
binary64 approximation.
"""

from __future__ import annotations

import math
import random
import struct
import unicodedata

RECEIVERS = ("P1", "N1", "P2", "N2")
OPPOSITE = "opposite_adjective"
CONTROLS = ("same_adjective", "opposite_subject", "opposite_instruction", "orthogonal")


def floats32(raw: bytes, expected: int) -> tuple[float, ...]:
    if type(raw) is not bytes or type(expected) is not int or expected < 1 or len(raw) != 4 * expected:
        raise ValueError("unexpected float32 payload")
    values = tuple(x[0] for x in struct.iter_unpack("<f", raw))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite saved float32")
    return values


def norm64(values) -> float:
    result = math.sqrt(math.fsum(value ** 2 for value in values))
    if not math.isfinite(result):
        raise ValueError("nonfinite vector norm")
    return result


def cosine64(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("vector widths differ")
    a, b = norm64(left), norm64(right)
    if not a or not b:
        return None
    result = math.fsum(x * y for x, y in zip(left, right, strict=True)) / (a * b)
    if not math.isfinite(result):
        raise ValueError("nonfinite cosine")
    return result


def donor_distance(base_raw: bytes, donor_raw: bytes) -> float:
    if len(base_raw) != len(donor_raw) or len(base_raw) % 4:
        raise ValueError("donor width differs")
    base = floats32(base_raw, len(base_raw) // 4)
    donor = floats32(donor_raw, len(donor_raw) // 4)
    return norm64(tuple(y - x for x, y in zip(base, donor, strict=True)))


def donor_landing(base_raw: bytes, donor_raw: bytes, radius: float):
    """Independently reconstruct one donor interpolation and its evidence."""
    if type(radius) is not float or not math.isfinite(radius) or radius <= 0:
        raise ValueError("invalid dose")
    if len(base_raw) != len(donor_raw) or len(base_raw) % 4:
        raise ValueError("donor width differs")
    base = floats32(base_raw, len(base_raw) // 4)
    donor = floats32(donor_raw, len(donor_raw) // 4)
    delta = tuple(y - x for x, y in zip(base, donor, strict=True))
    distance = norm64(delta)
    if not distance or radius > distance:
        raise ValueError("unavailable natural direction")
    fraction = radius / distance
    landed_raw = struct.pack("<" + str(len(base)) + "f", *(x + fraction * d for x, d in zip(base, delta, strict=True)))
    landed = floats32(landed_raw, len(base))
    movement = tuple(y - x for x, y in zip(base, landed, strict=True))
    achieved = norm64(movement)
    return {"bytes": landed_raw, "distance": distance, "fraction": fraction,
            "achieved_norm": achieved, "intended_cosine": cosine64(movement, delta),
            "tolerance": min(max(1e-5, 1e-5 * radius), 0.01 * radius)}


def orthogonal_landing(base_raw: bytes, donor_raw: bytes, radius: float, seed: int):
    """Recreate both seeded binary64 witness vectors and one rounded landing."""
    if type(seed) is not int or type(radius) is not float or radius <= 0 or not math.isfinite(radius):
        raise ValueError("invalid orthogonal identity")
    if len(base_raw) != len(donor_raw) or len(base_raw) % 4:
        raise ValueError("donor width differs")
    base = floats32(base_raw, len(base_raw) // 4)
    donor = floats32(donor_raw, len(donor_raw) // 4)
    delta = tuple(y - x for x, y in zip(base, donor, strict=True))
    distance = norm64(delta)
    if not distance or radius > distance:
        raise ValueError("unavailable natural direction")
    rng = random.Random(seed)
    source = tuple(rng.gauss(0, 1) for _ in base)
    coefficient = math.fsum(g * d for g, d in zip(source, delta, strict=True)) / math.fsum(d * d for d in delta)
    residual = tuple(g - coefficient * d for g, d in zip(source, delta, strict=True))
    residual_norm = norm64(residual)
    source_raw = struct.pack("<" + str(len(base)) + "d", *source)
    residual_raw = struct.pack("<" + str(len(base)) + "d", *residual)
    if not residual_norm:
        return {"source": source_raw, "residual": residual_raw, "status": "unavailable",
                "failure": "zero orthogonal residual"}
    scale = radius / residual_norm
    landed_raw = struct.pack("<" + str(len(base)) + "f", *(x + scale * d for x, d in zip(base, residual, strict=True)))
    landed = floats32(landed_raw, len(base))
    movement = tuple(y - x for x, y in zip(base, landed, strict=True))
    return {"source": source_raw, "residual": residual_raw, "coefficient": coefficient,
            "residual_norm": residual_norm, "donor_norm": distance, "bytes": landed_raw,
            "achieved_norm": norm64(movement), "intended_cosine": cosine64(movement, residual),
            "donor_cosine": cosine64(movement, delta),
            "tolerance": min(max(1e-5, 1e-5 * radius), 0.01 * radius)}


def greedy_step(raw: bytes, vocabulary: int):
    values = floats32(raw, vocabulary)
    best = max(values)
    token = values.index(best)
    return token, values.count(best) == 1


def strict_label(generated, expected, unique_steps, special_ids, decode):
    if (expected not in ("positive", "negative") or type(generated) not in (tuple, list)
            or type(unique_steps) not in (tuple, list) or len(generated) != len(unique_steps)
            or not generated or generated[-1] != 1 or generated.count(1) != 1
            or any(type(x) is not int or x in special_ids for x in generated[:-1])
            or any(x is not True for x in unique_steps)):
        return False
    text = decode(generated[:-1])
    if type(text) is not str:
        raise ValueError("decoder must return text")
    return unicodedata.normalize("NFC", text).strip().casefold() == expected


def first_step_tv(raw_a: bytes, raw_b: bytes, vocabulary: int) -> float:
    a, b = floats32(raw_a, vocabulary), floats32(raw_b, vocabulary)
    am, bm = max(a), max(b)
    ea, eb = tuple(math.exp(x - am) for x in a), tuple(math.exp(x - bm) for x in b)
    za, zb = math.fsum(ea), math.fsum(eb)
    return 0.5 * math.fsum(abs(x / za - y / zb) for x, y in zip(ea, eb, strict=True))


def teacher_score(raw: bytes, target_ids, vocabulary: int, torch):
    """Recompute selected CPU float32 log-softmax bits and exact ranks."""
    if type(target_ids) not in (list, tuple) or not target_ids:
        raise ValueError("missing target sequence")
    if any(type(i) is not int or not 0 <= i < vocabulary for i in target_ids):
        raise ValueError("invalid target ID")
    values = floats32(raw, len(target_ids) * vocabulary)
    matrix = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(len(target_ids), vocabulary)
    logprob = torch.log_softmax(matrix, dim=-1)
    selected = tuple(float(logprob[i, token].item()) for i, token in enumerate(target_ids))
    ranks = tuple(1 + sum(x > values[i * vocabulary + token] for x in values[i * vocabulary:(i + 1) * vocabulary])
                  for i, token in enumerate(target_ids))
    if not all(math.isfinite(x) for x in selected):
        raise ValueError("nonfinite selected log-probability")
    return {"selected": selected, "ranks": ranks, "score": math.fsum(selected)}


def group_rule(clean, measured):
    """Compute B3/B4 independently; an unavailable arm remains unmeasured."""
    if type(clean) is not dict or set(clean) != set(RECEIVERS) or type(measured) is not dict:
        raise ValueError("invalid receiver set")
    if any(type(v) is not float or not math.isfinite(v) for v in clean.values()):
        raise ValueError("nonfinite clean score")
    permitted = {OPPOSITE, *CONTROLS}
    if not set(measured) <= permitted:
        raise ValueError("invalid arm")
    effects = {}
    for arm, row in measured.items():
        if type(row) is not dict or set(row) != set(RECEIVERS):
            raise ValueError("arm is missing a receiver")
        if any(type(v) is not float or not math.isfinite(v) for v in row.values()):
            raise ValueError("nonfinite arm score")
        effects[arm] = {r: (clean[r] - row[r] if r[0] == "P" else row[r] - clean[r]) for r in RECEIVERS}
    means = {arm: math.fsum(abs(effects[arm][r]) for r in RECEIVERS) / 4.0
             for arm in CONTROLS if arm in effects}
    if OPPOSITE not in effects:
        return {"movements": effects, "control_means": means, "A": None, "B3": None, "B4": None,
                "positive_count": None, "passes_B3": None, "passes_B4": None}
    opposite = [effects[OPPOSITE][r] for r in RECEIVERS]
    a, positives = math.fsum(opposite) / 4.0, sum(value > 0 for value in opposite)
    b3 = max(means[x] for x in CONTROLS[:3]) if all(x in means for x in CONTROLS[:3]) else None
    b4 = max(means.values()) if all(x in means for x in CONTROLS) else None
    def passes(b):
        return None if b is None else a >= 0.10 and a - b >= 0.05 and positives >= 3
    return {"movements": effects, "control_means": means, "A": a, "B3": b3, "B4": b4,
            "positive_count": positives, "passes_B3": passes(b3), "passes_B4": passes(b4)}


def planned_interval(outcomes, planned: int, required: int, complete: bool):
    if (type(outcomes) not in (list, tuple) or type(planned) is not int or type(required) is not int
            or len(outcomes) != planned or not 1 <= required <= planned or type(complete) is not bool
            or any(type(v) is not str or v not in ("pass", "fail", "unmeasured") for v in outcomes)):
        raise ValueError("invalid planned outcomes")
    passes = outcomes.count("pass")
    evaluable = passes + outcomes.count("fail")
    lo, hi = passes, passes + planned - evaluable
    status = ("incomplete" if not complete else "met" if lo >= required
              else "not-met" if hi < required else "undetermined")
    return {"status": status, "passes": passes, "evaluable": evaluable, "planned": planned,
            "rate": passes / evaluable if evaluable else None,
            "interval": [lo, hi] if complete else None, "partial": evaluable < planned or not complete}


def classify_unit(groups, expected_subjects):
    """Apply only the fixed high-dose eligibility rule to one planned slot.

    This does not bless a pass as scientifically replayed: callers may supply
    a group only after the corresponding complete saved pass has been checked.
    Missing groups, unsupported clean baselines and absent high-dose B4 are
    unmeasured, independently of low-dose outcomes.
    """
    if (type(groups) is not list or type(expected_subjects) not in (list, tuple)
            or not expected_subjects or len(set(expected_subjects)) != len(expected_subjects)
            or any(s not in ("story", "concert", "painting") for s in expected_subjects)):
        raise ValueError("invalid unit groups")
    expected = {(subject, instruction) for subject in expected_subjects for instruction in (0, 1)}
    actual = {}
    for group in groups:
        if type(group) is not dict or type(group.get("instruction")) is not int:
            raise ValueError("invalid group")
        key = (group.get("subject"), group["instruction"])
        if key not in expected or key in actual:
            raise ValueError("duplicate or unexpected group")
        actual[key] = group
    reasons = []
    for key in sorted(expected):
        if key not in actual:
            reasons.append({"subject": key[0], "instruction": key[1], "reason": "missing-group"})
            continue
        group = actual[key]
        if type(group.get("baseline_supported")) is not bool:
            raise ValueError("baseline support is not Boolean")
        if not group["baseline_supported"]:
            reasons.append({"subject": key[0], "instruction": key[1], "reason": "baseline-unsupported"})
        doses = group.get("doses")
        if type(doses) is not dict or type(doses.get("high")) is not dict:
            raise ValueError("missing high-dose record")
        high = doses["high"]
        if high.get("status") != "measured" or type(high.get("passes_B4")) is not bool:
            reasons.append({"subject": key[0], "instruction": key[1], "reason": "high-dose-B4-unmeasured"})
    if reasons:
        return {"status": "unmeasured", "reasons": reasons}
    return {"status": "pass" if all(group["doses"]["high"]["passes_B4"] for group in actual.values()) else "fail",
            "reasons": []}
