"""Model-free full-vocabulary baseline scorer for the prospective copy screen."""
import math
import struct

VOCAB = 50_257
ROW_BYTES = VOCAB * 4
GAP_FLOOR = 0.5


def decode_float32(bits):
    # Every finite binary32 value, including its subnormals, is normal-range
    # binary64. Decode integer fields without a hardware binary32 conversion.
    sign = -1.0 if bits >> 31 else 1.0
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7fffff
    if exponent == 255:
        raise ValueError("baseline row contains a nonfinite logit")
    magnitude = (math.ldexp(float(fraction), -149) if exponent == 0 else
                 math.ldexp(float((1 << 23) | fraction), exponent - 150))
    return sign * magnitude

def score_row(raw, target):
    if type(raw) is not bytes or len(raw) != ROW_BYTES:
        raise ValueError("a complete little-endian float32 vocabulary row is required")
    if type(target) is not int or not 0 <= target < VOCAB:
        raise ValueError("target token ID is outside the vocabulary")
    values = tuple(decode_float32(bits[0]) for bits in struct.iter_unpack("<I", raw))
    maximum = max(values)
    normalizer = math.log(math.fsum(math.exp(value - maximum) for value in values))
    count = sum(value == maximum for value in values)
    first = values.index(maximum)
    return {"target_id": target, "target_logit": values[target], "maximum": maximum,
            "logz_shift": normalizer,
            "target_log_probability": (values[target] - maximum) - normalizer,
            "target_rank": 1 + sum(value > values[target] for value in values),
            "top_unique_id": first if count == 1 else None,
            "top_representative_id": first, "top_tie_count": count}


def target_gap(repeat, shuffle):
    if repeat["target_id"] != shuffle["target_id"]:
        raise ValueError("a repeat and its shuffle must score the same token")
    return math.fsum((repeat["target_logit"], -shuffle["target_logit"],
                      -repeat["maximum"], shuffle["maximum"],
                      -repeat["logz_shift"], shuffle["logz_shift"]))


def score_family(cases, raw_rows):
    if (type(cases) is not list or type(raw_rows) is not list
            or len(cases) != 8 or len(raw_rows) != 8):
        raise ValueError("a family needs exactly eight prompt rows")
    group = cases[0]["group"]
    if (type(group) is not int or not 1 <= group <= 24
            or any(case["group"] != group or case["variant"] != i for i, case in enumerate(cases))
            or len({case["case_id"] for case in cases}) != 8):
        raise ValueError("family identities or order differ")
    targets = [case["target_id"] for case in cases]
    if (len(set(targets[:4])) != 1 or len(set(targets[4:])) != 1
            or targets[0] == targets[4]):
        raise ValueError("rotations require two distinct fixed targets")
    scores = [score_row(raw, case["target_id"]) for case, raw in zip(cases, raw_rows)]
    gaps, reasons = [], []
    for rotation in range(2):
        start = 4 * rotation
        repeat = scores[start]
        if repeat["top_unique_id"] != repeat["target_id"]:
            reasons.append({"rotation": rotation, "reason": "repeat-target-not-unique-top"})
        for shuffle in range(1, 4):
            gap = target_gap(repeat, scores[start + shuffle])
            gaps.append({"rotation": rotation, "shuffle": shuffle, "gap": gap})
            if gap < GAP_FLOOR:
                reasons.append({"rotation": rotation, "shuffle": shuffle, "reason": "gap-below-0.5"})
    return {"group": group, "split": "development" if group % 2 else "held-out",
            "baseline_status": "supported" if not reasons else "unsupported",
            "reasons": reasons, "gaps": gaps,
            "rows": [{"case_id": case["case_id"], "variant": case["variant"], **score}
                     for case, score in zip(cases, scores)]}


def score_screen(cases, raw_rows):
    if len(cases) != 192 or len(raw_rows) != 192:
        raise ValueError("complete screen requires all 192 rows")
    if len({case["case_id"] for case in cases}) != 192:
        raise ValueError("duplicate case identity")
    families = [score_family(cases[i:i+8], raw_rows[i:i+8]) for i in range(0, 192, 8)]
    if [family["group"] for family in families] != list(range(1, 25)):
        raise ValueError("family ordering differs from the frozen split")
    return {"families": families,
            "supported_development": sum(f["baseline_status"] == "supported" and f["split"] == "development"
                                         for f in families),
            "supported_held_out": sum(f["baseline_status"] == "supported" and f["split"] == "held-out"
                                      for f in families)}
