"""Independent numerical replay of the public pair/dose copying protocol.

This module analyzes caller-supplied saved bytes; it does not execute a model,
authenticate a source, verify an execution chronology, or establish causation.
It deliberately does not import the experiment runner.
"""

from __future__ import annotations

import math
import hashlib
import re
import struct


ARMS = (
    "h2_low", "h7_low", "h10_low", "h2_h7_low", "h2_h10_low", "h7_h10_low",
    "h2_match", "h7_match", "h10_match",
    "h2_high", "h7_high", "h10_high", "h2_h7_high", "h2_h10_high", "h7_h10_high",
)
PAIRS = (("h2", "h7"), ("h2", "h10"), ("h7", "h10"))


def _float32_bits(bits: int) -> float:
    sign = -1.0 if bits >> 31 else 1.0
    exponent = (bits >> 23) & 0xff
    mantissa = bits & 0x7fffff
    if exponent == 0xff:
        raise ValueError("nonfinite float32 value")
    if exponent == 0:
        magnitude = math.ldexp(mantissa, -149)
    else:
        magnitude = math.ldexp((1 << 23) | mantissa, exponent - 150)
    return math.copysign(magnitude, sign)


def _nearest_float32_bits(value: float) -> int:
    """Convert binary64 to binary32 using integer nearest-even rounding."""
    if not math.isfinite(value):
        raise ValueError("nonfinite conversion")
    sign = 0x80000000 if math.copysign(1.0, value) < 0 else 0
    value = abs(value)
    if value == 0:
        return sign
    numerator, denominator = value.as_integer_ratio()
    exponent = math.frexp(value)[1] - 1
    shift = 149 if exponent < -126 else 23 - exponent
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    quotient, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or (remainder * 2 == denominator and quotient & 1):
        quotient += 1
    if exponent < -126:
        return sign | quotient  # Rounding may produce the smallest normal.
    if quotient == 1 << 24:
        quotient >>= 1
        exponent += 1
    if exponent > 127:
        raise ValueError("float32 overflow")
    return sign | ((exponent + 127) << 23) | (quotient - (1 << 23))


def _norm(values) -> float:
    return math.sqrt(math.fsum(value ** 2 for value in values))


def validate_delivery_family(sources, deliveries) -> dict:
    """Recompute all requested doses and compare every post-action bit.

    Sources and deliveries contain four ordered complete 11-by-768
    little-endian float32 projection inputs. Each delivery mapping must have
    all 15 arm names. This enforces the exact single/pair correspondence and
    unchanged non-target input bits, but does not itself check saved pre-action
    inputs or the authenticity of any input. Callers must supply the verified
    same-prompt no-op input as each source.
    """
    if type(sources) not in (tuple, list) or len(sources) != 4:
        raise ValueError("exactly four source inputs required")
    if type(deliveries) not in (tuple, list) or len(deliveries) != 4:
        raise ValueError("exactly four delivery mappings required")
    if any(type(source) is not bytes or len(source) != 11 * 768 * 4 for source in sources):
        raise ValueError("invalid projection input")
    if any(type(row) is not dict or set(row) != set(ARMS) for row in deliveries):
        raise ValueError("all and only fixed deliveries are required")
    heads = {"h2": 2, "h7": 7, "h10": 10}
    offsets = {name: (10 * 768 + 64 * head) * 4 for name, head in heads.items()}
    source_slices = []
    norms = []
    for source in sources:
        # Check all coordinates, not only the selected head slices, for finiteness.
        _decode(source, 11 * 768)
        slices = {name: source[offset:offset + 256] for name, offset in offsets.items()}
        values = {name: _decode(raw, 64) for name, raw in slices.items()}
        row_norms = {name: _norm(item) for name, item in values.items()}
        if any(not math.isfinite(n) or n <= 0 for n in row_norms.values()):
            raise ValueError("invalid source norm")
        source_slices.append((slices, values))
        norms.append(row_norms)
    minimum = min(n for row in norms for n in row.values())
    if minimum <= 2.0 ** -19:
        raise ValueError("minimum head norm is too small")
    q = {"low": 0.5 * minimum, "high": minimum, "match": math.sqrt(2.0) * (0.5 * minimum)}
    variants = []
    for variant, source in enumerate(sources):
        slices, values = source_slices[variant]
        planned = {}
        single_info = {}
        for name in heads:
            for dose in ("low", "match", "high"):
                key = f"{name}_{dose}"
                fraction = q[dose] / norms[variant][name]
                scale = 1.0 - fraction
                bits = [_nearest_float32_bits(scale * value) for value in values[name]]
                after = b"".join(struct.pack("<I", item) for item in bits)
                after_values = [_float32_bits(item) for item in bits]
                differences = [new - old for new, old in zip(after_values, values[name])]
                achieved = _norm(differences)
                if not math.isfinite(achieved) or abs(achieved - q[dose]) > 1e-5 * q[dose]:
                    raise ValueError("achieved single delivery norm mismatch")
                original_bits = [item[0] for item in struct.iter_unpack("<I", slices[name])]
                planned[key] = (after, differences)
                single_info[key] = {
                    "requested": q[dose], "fraction": fraction, "achieved": achieved,
                    "changed_elements": sum(a != b for a, b in zip(bits, original_bits)),
                    "changed_bit_count": sum((a ^ b).bit_count() for a, b in zip(bits, original_bits)),
                }
        for arm in ARMS:
            parts = arm.split("_")
            dose = parts[-1]
            expected = bytearray(source)
            for name in parts[:-1]:
                offset = offsets[name]
                expected[offset:offset + 256] = planned[f"{name}_{dose}"][0]
            actual = deliveries[variant][arm]
            if type(actual) is not bytes or actual != expected:
                raise ValueError(f"delivery bits mismatch for {arm}")
        pairs = {}
        for first, second in PAIRS:
            for dose in ("low", "high"):
                key = f"{first}_{second}_{dose}"
                combined = _norm(planned[f"{first}_{dose}"][1] + planned[f"{second}_{dose}"][1])
                if not math.isfinite(combined):
                    raise ValueError("nonfinite pair delivery norm")
                if dose == "low" and abs(combined - q["match"]) > 1e-5 * q["match"]:
                    raise ValueError("matched pair delivery norm mismatch")
                pair_info = {"combined_norm": combined}
                if dose == "low":
                    pair_info["match_a_norm"] = single_info[f"{first}_match"]["achieved"]
                    pair_info["match_b_norm"] = single_info[f"{second}_match"]["achieved"]
                    pair_info["pair_minus_match_a_norm"] = combined - pair_info["match_a_norm"]
                    pair_info["pair_minus_match_b_norm"] = combined - pair_info["match_b_norm"]
                pairs[key] = pair_info
        variants.append({"norms": norms[variant], "singles": single_info, "pairs": pairs})
    return {"minimum_norm": minimum, "q_low": q["low"], "q_match": q["match"], "q_high": q["high"], "variants": variants}


def _decode(raw: bytes, vocabulary_size: int) -> list[float]:
    if type(raw) is not bytes or len(raw) != 4 * vocabulary_size:
        raise ValueError("invalid float32 logit vector length or type")
    return [_float32_bits(bits) for (bits,) in struct.iter_unpack("<I", raw)]


def _score(logits: list[float], target: int) -> dict[str, float | int]:
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("nonfinite logit")
    maximum = max(logits)
    target_value = logits[target]
    log_probability = (target_value - maximum) - math.log(
        math.fsum(math.exp(x - maximum) for x in logits)
    )
    if not math.isfinite(log_probability):
        raise ValueError("nonfinite score")
    return {
        "target_logit": target_value,
        "target_log_probability": log_probability,
        "target_rank": 1 + sum(x > target_value for x in logits),
        "top_token_id": logits.index(maximum),
    }


def _contrast(values: list[float]) -> float:
    return values[0] - math.fsum(values[1:]) / 3.0


def analyze_family(cases: list[dict], *, vocabulary_size: int = 50257) -> dict:
    """Score a complete ordered repeated-plus-three-shuffle family.

    Each case supplies an integer target_id, exact little-endian float32
    ``clean`` bytes, and an ``arms`` dictionary with all 15 names in ARMS.
    This numerical-only entry does not validate execution order, no-op inputs,
    deliveries, receipt, or source provenance. Production callers must first
    bind and validate those independently. The optional smaller vocabulary is
    for fabricated tests; the study's actual vocabulary contains 50,257 IDs.
    """
    if type(vocabulary_size) is not int or not 1 <= vocabulary_size <= 50257:
        raise ValueError("invalid vocabulary size")
    if type(cases) not in (tuple, list) or len(cases) != 4:
        raise ValueError("exactly four ordered cases required")

    rows = []
    shared_target = None
    for case in cases:
        if type(case) is not dict or set(case) != {"target_id", "clean", "arms"}:
            raise ValueError("invalid case")
        target = case["target_id"]
        if type(target) is not int or not 0 <= target < vocabulary_size:
            raise ValueError("invalid target")
        if shared_target is None:
            shared_target = target
        elif shared_target != target:
            raise ValueError("variants must share one target")
        arms = case["arms"]
        if type(arms) is not dict or set(arms) != set(ARMS):
            raise ValueError("all and only fixed arms are required")

        decoded = {"clean": _decode(case["clean"], vocabulary_size)}
        decoded.update({name: _decode(arms[name], vocabulary_size) for name in ARMS})
        scores = {name: _score(row, target) for name, row in decoded.items()}
        additive = {}
        for first, second in PAIRS:
            for dose in ("low", "high"):
                key = f"{first}_{second}_{dose}"
                a, b = decoded[f"{first}_{dose}"], decoded[f"{second}_{dose}"]
                clean = decoded["clean"]
                z_add = [(x - base) + y for x, base, y in zip(a, clean, b)]
                additive[key] = _score(z_add, target)
        rows.append({"target_id": target, "scores": scores, "additive": additive})

    metrics = ("target_log_probability", "target_logit")
    arms_out = {}
    for name in ARMS:
        arms_out[name] = {}
        for metric in metrics:
            deltas = [row["scores"][name][metric] - row["scores"]["clean"][metric] for row in rows]
            arms_out[name][metric] = {"deltas": deltas, "contrast": _contrast(deltas)}

    pairs_out = {}
    for first, second in PAIRS:
        for dose in ("low", "high"):
            key = f"{first}_{second}_{dose}"
            pairs_out[key] = {}
            for metric, public_name in (("target_log_probability", "log_probability"), ("target_logit", "target_logit")):
                a = arms_out[f"{first}_{dose}"][metric]["contrast"]
                b = arms_out[f"{second}_{dose}"][metric]["contrast"]
                pair = arms_out[key][metric]["contrast"]
                deltas_add = [row["additive"][key][metric] - row["scores"]["clean"][metric] for row in rows]
                d_add = _contrast(deltas_add)
                da = arms_out[f"{first}_{dose}"][metric]["deltas"]
                db = arms_out[f"{second}_{dose}"][metric]["deltas"]
                dp = arms_out[key][metric]["deltas"]
                item = {
                    "single_a": a,
                    "single_b": b,
                    "pair": pair,
                    "interaction": (pair - a) - b,
                    "additive_deltas": deltas_add,
                    "additive_contrast": d_add,
                    "additive_interaction": (d_add - a) - b,
                    "pair_minus_additive": pair - d_add,
                    "prompt_interactions": [(p - x) - y for p, x, y in zip(dp, da, db)],
                    "additive_prompt_interactions": [(p - x) - y for p, x, y in zip(deltas_add, da, db)],
                }
                if dose == "low":
                    match_a = arms_out[f"{first}_match"][metric]["contrast"]
                    match_b = arms_out[f"{second}_match"][metric]["contrast"]
                    item.update(match_a=match_a, match_b=match_b,
                                pair_minus_match_a=pair - match_a, pair_minus_match_b=pair - match_b)
                pairs_out[key][public_name] = item

    comparisons = {}
    for metric in ("log_probability", "target_logit"):
        per_dose = {}
        for dose in ("low", "high"):
            j210 = pairs_out[f"h2_h10_{dose}"][metric]["interaction"]
            per_dose[dose] = {
                "h2_h10_minus_h2_h7": j210 - pairs_out[f"h2_h7_{dose}"][metric]["interaction"],
                "h2_h10_minus_h7_h10": j210 - pairs_out[f"h7_h10_{dose}"][metric]["interaction"],
            }
        per_dose["high_minus_low"] = {
            f"{a}_{b}": pairs_out[f"{a}_{b}_high"][metric]["interaction"] - pairs_out[f"{a}_{b}_low"][metric]["interaction"]
            for a, b in PAIRS
        }
        comparisons[metric] = per_dose
    return {"rows": rows, "arms": arms_out, "pairs": pairs_out, "comparisons": comparisons}


def audit_saved_family(cases, records, raw_files, *, family_index: int, vocabulary_size: int = 50257) -> dict:
    """Replay one complete family's saved records and landing witnesses.

    ``raw_files`` maps each fixed case ID to its complete immutable bytes. This
    function requires contiguous hashed slices in the saved action order and
    recomputes scores, controls, doses and projection-output diagnostics. It
    does not check a study-level receipt, source binding, process execution or
    case provenance; those require a separate outer evidence boundary. The
    optional smaller vocabulary is only for fabricated tests.
    """
    if type(family_index) is not int or not 0 <= family_index < 8:
        raise ValueError("invalid family index")
    if type(vocabulary_size) is not int or not 1 <= vocabulary_size <= 50257:
        raise ValueError("invalid vocabulary size")
    if type(cases) is not list or len(cases) != 4 or type(records) is not list or len(records) != 72:
        raise ValueError("incomplete case or record set")
    for variant, case in enumerate(cases):
        if (type(case) is not dict or case.get("case_id") != f"copy-pairs-{family_index + 1:02d}-{variant}"
                or type(case.get("group")) is not int or case["group"] != family_index + 1
                or type(case.get("variant")) is not int or case["variant"] != variant
                or type(case.get("target_id")) is not int or not 0 <= case["target_id"] < vocabulary_size):
            raise ValueError("invalid saved case identity")
    ids = [case["case_id"] for case in cases]
    expected_file_bytes = 18 * 4 * vocabulary_size + 16 * 3 * 11 * 768 * 4
    if (type(raw_files) is not dict or set(raw_files) != set(ids)
            or any(type(raw_files[key]) is not bytes or len(raw_files[key]) != expected_file_bytes for key in ids)):
        raise ValueError("incomplete or oversized case payload")
    expected = [(cases[v], action) for v in range(4) for action in ("clean", "noop")]
    for variant, case in enumerate(cases):
        shift = (4 * family_index + variant) % len(ARMS)
        expected.extend((case, action) for action in (*ARMS[shift:], *ARMS[:shift], "clean_after"))

    offsets = {key: 0 for key in ids}
    saved = {key: {} for key in ids}
    for record, (case, action) in zip(records, expected):
        if type(record) is not dict or record.get("action") != action:
            raise ValueError("saved record action order differs")
        for key in ("case_id", "group", "variant", "target_id"):
            if type(record.get(key)) is not type(case[key]) or record[key] != case[key]:
                raise ValueError("saved record identity differs")
        field_sizes = {"logits": vocabulary_size * 4}
        if action not in ("clean", "clean_after"):
            field_sizes.update(cproj_before=11 * 768 * 4, cproj_after=11 * 768 * 4, cproj_output=11 * 768 * 4)
        raw = record.get("raw")
        if type(raw) is not dict or set(raw) != set(field_sizes):
            raise ValueError("missing or unexpected saved slice")
        chunks = {}
        for name, size in field_sizes.items():
            item = raw[name]
            if (type(item) is not dict or set(item) != {"offset", "bytes", "sha256"}
                    or type(item["offset"]) is not int or item["offset"] != offsets[case["case_id"]]
                    or type(item["bytes"]) is not int or item["bytes"] != size
                    or type(item["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
                raise ValueError("invalid, missing, or overlapping saved slice")
            end = item["offset"] + size
            chunk = raw_files[case["case_id"]][item["offset"]:end]
            if len(chunk) != size or hashlib.sha256(chunk).hexdigest() != item["sha256"]:
                raise ValueError("saved slice digest differs")
            offsets[case["case_id"]] = end
            chunks[name] = chunk
        score = _score(_decode(chunks["logits"], vocabulary_size), case["target_id"])
        recorded = record.get("score")
        if type(recorded) is not dict or set(recorded) != set(score):
            raise ValueError("saved score is incomplete")
        for name, value in score.items():
            saved_value = recorded[name]
            if name in ("target_rank", "top_token_id"):
                if type(saved_value) is not int or saved_value != value:
                    raise ValueError("saved rank or top token differs")
            elif type(saved_value) not in (int, float) or not math.isfinite(saved_value) or saved_value != value:
                raise ValueError("saved score differs")
        if "cproj_output" in chunks:
            _decode(chunks["cproj_output"], 11 * 768)  # Reject nonfinite values at every position.
        saved[case["case_id"]][action] = chunks
    if any(offsets[key] != expected_file_bytes for key in ids):
        raise ValueError("unconsumed or incomplete payload")

    sources, deliveries, numerical_cases, output_diagnostics = [], [], [], []
    for case in cases:
        row = saved[case["case_id"]]
        clean = row["clean"]["logits"]
        if row["noop"]["logits"] != clean or row["clean_after"]["logits"] != clean:
            raise ValueError("clean/no-op/clean-after logits differ")
        source = row["noop"]["cproj_before"]
        if row["noop"]["cproj_after"] != source:
            raise ValueError("no-op input differs")
        if any(row[arm]["cproj_before"] != source for arm in ARMS):
            raise ValueError("pre-action input differs")
        sources.append(source)
        deliveries.append({arm: row[arm]["cproj_after"] for arm in ARMS})
        numerical_cases.append({"target_id": case["target_id"], "clean": clean,
                                "arms": {arm: row[arm]["logits"] for arm in ARMS}})
        final = {name: _decode(row[name]["cproj_output"][-768 * 4:], 768) for name in ("noop", *ARMS)}
        per_pair = {}
        for first, second in PAIRS:
            for dose in ("low", "high"):
                key = f"{first}_{second}_{dose}"
                base, a, b, pair = final["noop"], final[f"{first}_{dose}"], final[f"{second}_{dose}"], final[key]
                per_pair[key] = {
                    "single_a_output_delta_norm": _norm(x - y for x, y in zip(a, base)),
                    "single_b_output_delta_norm": _norm(x - y for x, y in zip(b, base)),
                    "pair_output_delta_norm": _norm(x - y for x, y in zip(pair, base)),
                    "pair_minus_singles_output_residual_norm": _norm(((x - y) - z) + n for x, y, z, n in zip(pair, a, b, base)),
                }
        output_diagnostics.append({
            "output_delta_norms": {name: _norm(x - y for x, y in zip(final[name], final["noop"])) for name in ARMS},
            "pair_residual_norms": {name: item["pair_minus_singles_output_residual_norm"] for name, item in per_pair.items()},
            "pairs": per_pair,
        })
    delivery_report = validate_delivery_family(sources, deliveries)
    numerical = analyze_family(numerical_cases, vocabulary_size=vocabulary_size)
    numerical["top_one_counts"] = {
        name: sum(row["scores"][name]["top_token_id"] == row["target_id"] for row in numerical["rows"])
        for name in ("clean", *ARMS)
    }
    return {"case_ids": ids, "delivery": delivery_report, "numerical": numerical,
            "output_diagnostics": output_diagnostics}


def summarize_study(families) -> dict:
    """Aggregate exactly eight ordered family reports, retaining all rows.

    This aggregation is not a receipt or provenance verifier. A caller must
    obtain every family from its complete saved-data audit and bind the whole
    study's receipt and source separately before describing the run as complete.
    """
    if type(families) not in (list, tuple) or len(families) != 8:
        raise ValueError("expected eight complete families")
    for group, family in enumerate(families):
        if (type(family) is not dict or family.get("case_ids") !=
                [f"copy-pairs-{group + 1:02d}-{variant}" for variant in range(4)]):
            raise ValueError("family identities are missing or out of order")
    try:
        numerical = [family["numerical"] for family in families]
        if any(len(item["rows"]) != 4 for item in numerical):
            raise ValueError("family rows are incomplete")
        mean_arms = {}
        for arm in ARMS:
            mean_arms[arm] = {}
            for metric in ("target_log_probability", "target_logit"):
                rows = [item["arms"][arm][metric] for item in numerical]
                mean_arms[arm][metric] = {
                    "contrast": math.fsum(row["contrast"] for row in rows) / 8,
                    "repeat_delta": math.fsum(row["deltas"][0] for row in rows) / 8,
                    "shuffle_mean_delta": math.fsum(math.fsum(row["deltas"][1:]) / 3.0 for row in rows) / 8,
                }
        mean_pairs = {}
        for first, second in PAIRS:
            for dose in ("low", "high"):
                key = f"{first}_{second}_{dose}"
                mean_pairs[key] = {}
                for metric in ("log_probability", "target_logit"):
                    rows = [item["pairs"][key][metric] for item in numerical]
                    fields = ("single_a", "single_b", "pair", "interaction", "additive_contrast",
                              "additive_interaction", "pair_minus_additive")
                    if dose == "low":
                        fields += ("match_a", "match_b", "pair_minus_match_a", "pair_minus_match_b")
                    mean_pairs[key][metric] = {field: math.fsum(row[field] for row in rows) / 8 for field in fields}
        mean_comparisons = {}
        for metric in ("log_probability", "target_logit"):
            mean_comparisons[metric] = {}
            for dose in ("low", "high", "high_minus_low"):
                keys = numerical[0]["comparisons"][metric][dose]
                mean_comparisons[metric][dose] = {
                    key: math.fsum(item["comparisons"][metric][dose][key] for item in numerical) / 8
                    for key in keys
                }
        top_one = {}
        for arm in ("clean", *ARMS):
            hits = [[row["scores"][arm]["top_token_id"] == row["target_id"] for row in item["rows"]]
                    for item in numerical]
            top_one[arm] = {"repeated": sum(row[0] for row in hits),
                            "shuffled": sum(sum(row[1:]) for row in hits)}
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("incomplete family report") from error
    return {"families": list(families), "mean_arms": mean_arms, "mean_pairs": mean_pairs,
            "mean_comparisons": mean_comparisons, "top_one": top_one}


def replay_result(cases, result, read_case, *, vocabulary_size: int = 50257) -> dict:
    """Replay a complete result, reading at most four case files at a time.

    ``read_case(id)`` must return the complete immutable per-case bytes, and
    is responsible for bounding its own input read. This function checks the
    fixed 576-record schedule and witnesses. It does not authenticate the
    result, receipt, source revision, cases, supervisor, or actual execution;
    a separate evidence-binding caller is required before any completion claim.
    A smaller vocabulary is accepted only for fabricated tests.
    """
    if type(vocabulary_size) is not int or not 1 <= vocabulary_size <= 50257:
        raise ValueError("invalid vocabulary size")
    if type(cases) is not list or len(cases) != 32 or not callable(read_case):
        raise ValueError("expected 32 ordered cases and a case reader")
    for group in range(8):
        for variant in range(4):
            case = cases[4 * group + variant]
            if (type(case) is not dict or case.get("case_id") != f"copy-pairs-{group + 1:02d}-{variant}"
                    or type(case.get("group")) is not int or case["group"] != group + 1
                    or type(case.get("variant")) is not int or case["variant"] != variant
                    or type(case.get("target_id")) is not int or not 0 <= case["target_id"] < vocabulary_size):
                raise ValueError("saved case identity differs")
    if (type(result) is not dict or result.get("schema") != "distilgpt2-copy-head-pairs-dose-result-v1"
            or result.get("status") != "complete" or type(result.get("cases")) is not list
            or len(result["cases"]) != 576):
        raise ValueError("result record set is incomplete")
    for key in ("submitted_forwards", "model_call_returns_observed", "returned_forwards"):
        if type(result.get(key)) is not int or result[key] != 576:
            raise ValueError("forward counter differs")
    file_size = 18 * 4 * vocabulary_size + 16 * 3 * 11 * 768 * 4
    if type(result.get("raw_payload_bytes")) is not int or result["raw_payload_bytes"] != 32 * file_size:
        raise ValueError("result payload count differs")
    families = []
    for group in range(8):
        family_cases = cases[group * 4:group * 4 + 4]
        files = {case["case_id"]: read_case(case["case_id"]) for case in family_cases}
        families.append(audit_saved_family(family_cases, result["cases"][group * 72:(group + 1) * 72],
                                           files, family_index=group, vocabulary_size=vocabulary_size))
    aggregate = summarize_study(families)
    aggregate["scope"] = "saved-record numerical and landing-witness consistency only; receipt, source and execution are not authenticated"
    return aggregate
