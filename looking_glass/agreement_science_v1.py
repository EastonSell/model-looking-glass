"""Model-free arithmetic for the public agreement-direction protocol."""
import math
import struct
import hashlib
from pathlib import Path
import re
import stat


_TARGET_ARMS = ("same_diff", "opp_same", "opp_diff", "u_plus", "u_minus",
                "v1_plus", "v1_minus", "v2_plus", "v2_minus", "l1_u_plus",
                "l1_u_minus", "subject_u_plus", "subject_u_minus", "center_u",
                "center_v1", "center_v2")
_CONTEXT_ARMS = ("u_plus", "u_minus", "v1_plus", "v1_minus", "v2_plus", "v2_minus")


def same_recorded_value(a, b):
    """Compare JSON values without treating bools, numbers or signed zero as aliases."""
    if type(a) is not type(b):
        return False
    if type(a) is float:
        return math.isfinite(a) and math.isfinite(b) and struct.pack("<d", a) == struct.pack("<d", b)
    if type(a) is dict:
        return set(a) == set(b) and all(same_recorded_value(a[key], b[key]) for key in a)
    if type(a) is list:
        return len(a) == len(b) and all(same_recorded_value(x, y) for x, y in zip(a, b))
    return type(a) in (str, int, bool, type(None)) and a == b


def expected_schedule(cases):
    """Derive the protocol's fixed forward order without importing the owner."""
    if type(cases) not in (list, tuple) or len(cases) != 40:
        raise ValueError("expected the complete ordered 40-case inventory")
    names = set()
    for index, case in enumerate(cases):
        if type(case) is not dict:
            raise ValueError("invalid case")
        name = case.get("case_id")
        if type(name) is not str or not name or name in names:
            raise ValueError("invalid or duplicate case ID")
        names.add(name)
        family = index // 4 if index < 32 else index - 32
        if type(case.get("family_index")) is not int or case["family_index"] != family:
            raise ValueError("case family order differs")
        if index < 32:
            variant = index % 4
            if (case.get("role") != "target" or type(case.get("variant")) is not int
                    or case["variant"] != variant
                    or case.get("subject_number") != ("singular" if variant % 2 == 0 else "plural")):
                raise ValueError("target variant or number differs")
        elif case.get("role") != "context" or case.get("subject_number") is not None:
            raise ValueError("context identity differs")

    schedule = []
    def add(case, action, slot=None):
        schedule.append({"index": len(schedule), "case_id": case["case_id"],
                         "role": case["role"], "family_index": case["family_index"],
                         "subject_number": case.get("subject_number"), "action": action, "slot": slot})
    for family in range(8):
        members = cases[family * 4:family * 4 + 4]
        for case in members:
            for action in ("clean", "clean_repeat", "noop_l3_final", "noop_l3_subject", "noop_l1_final"):
                add(case, action)
        for case in members:
            offset = (family + 4 * case["variant"]) % 16
            for slot in range(16):
                add(case, _TARGET_ARMS[(offset + slot) % 16], slot)
            add(case, "clean_after")
        context = cases[32 + family]
        for action in ("clean", "clean_repeat", "noop_l3_final"):
            add(context, action)
        for slot in range(6):
            add(context, _CONTEXT_ARMS[(family + slot) % 6], slot)
        add(context, "clean_after")
    return schedule


def validate_forward_inventory(events, cases):
    """Check all forward identities and actual input rows in protocol order."""
    schedule = expected_schedule(cases)
    if type(events) not in (list, tuple) or len(events) != len(schedule):
        raise ValueError("forward inventory is incomplete")
    by_id = {case["case_id"]: case for case in cases}
    for event, expected in zip(events, schedule):
        if type(event) is not dict or event.get("kind") != "forward":
            raise ValueError("invalid forward event")
        for field, value in expected.items():
            if field not in event or type(event[field]) is not type(value) or event[field] != value:
                raise ValueError("forward identity or order differs")
        expected_tokens = by_id[expected["case_id"]].get("token_ids")
        actual_tokens, mask = event.get("input_ids"), event.get("attention_mask")
        if (type(expected_tokens) is not list or not expected_tokens
                or any(type(value) is not int or not 0 <= value < 50257 for value in expected_tokens)
                or type(actual_tokens) is not list or len(actual_tokens) != len(expected_tokens)
                or any(type(a) is not int or a != b for a, b in zip(actual_tokens, expected_tokens))
                or type(mask) is not list or len(mask) != len(expected_tokens)
                or any(type(value) is not int or value != 1 for value in mask)
                or event.get("all_sequence_logits_finite") is not True):
            raise ValueError("forward input or output status differs")
    return len(schedule)


def validate_journal_layout(records, cases):
    """Require the complete interleaving of forwards and geometry records."""
    schedule = expected_schedule(cases)
    if type(records) not in (list, tuple) or len(records) != 809:
        raise ValueError("journal event inventory is incomplete")
    for index, record in enumerate(records):
        if (type(record) is not dict or set(record) != {"index", "event", "tensors"}
                or type(record["index"]) is not int or record["index"] != index
                or type(record["event"]) is not dict or type(record["tensors"]) is not dict):
            raise ValueError("journal record identity differs")
    first = records[0]
    if (first["event"].get("kind") != "run_contract" or first["tensors"]
            or type(first["event"].get("expected_forwards")) is not int
            or first["event"]["expected_forwards"] != 784
            or type(first["event"].get("seed")) is not int or first["event"]["seed"] != 20261030):
        raise ValueError("run contract differs")
    cursor = 1
    forwards = []
    def accept_forward(expected):
        nonlocal cursor
        record = records[cursor]
        cursor += 1
        event, tensors = record["event"], record["tensors"]
        hooked = expected["action"] not in ("clean", "clean_repeat", "clean_after")
        names = ({"last_logits", "whole_before", "before", "after", "whole_after"}
                 if hooked else {"last_logits"})
        if set(tensors) != names:
            raise ValueError("forward tensor inventory differs")
        forwards.append(event)
    for family in range(8):
        group = schedule[family * 98:(family + 1) * 98]
        for expected in group[:20]:
            accept_forward(expected)
        event, tensors = records[cursor]["event"], records[cursor]["tensors"]
        cursor += 1
        if (event.get("kind") != "family_geometry" or type(event.get("family_index")) is not int
                or event["family_index"] != family or tensors):
            raise ValueError("family geometry record differs")
        for expected in group[20:88]:
            accept_forward(expected)
        for variant in (0, 2):
            event, tensors = records[cursor]["event"], records[cursor]["tensors"]
            cursor += 1
            if (event.get("kind") != "centering_result" or type(event.get("family_index")) is not int
                    or event["family_index"] != family or type(event.get("singular_variant")) is not int
                    or event["singular_variant"] != variant or tensors):
                raise ValueError("centering record differs")
        for expected in group[88:]:
            accept_forward(expected)
    if cursor != len(records):
        raise ValueError("unexpected journal records")
    return validate_forward_inventory(forwards, cases)


def _finite(*values):
    try:
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError("expected finite numbers")
    except OverflowError:
        raise ValueError("expected finite numbers") from None


def scalar_response(*, clean, plus, minus, q):
    _finite(clean, plus, minus, q)
    if q < 0:
        raise ValueError("dose must be nonnegative")
    if q == 0 and (clean != plus or clean != minus):
        raise ValueError("zero-dose response must match clean")
    return {
        "plus_change": plus - clean,
        "minus_change": minus - clean,
        "odd": (plus - minus) / 2,
        "even": (plus + minus) / 2 - clean,
        "odd_per_unit": ((plus - minus) / 2) / q if q else None,
    }


def projection_loss(clean_singular, clean_plural, arm_singular, arm_plural):
    _finite(clean_singular, clean_plural, arm_singular, arm_plural)
    clean_gap = clean_singular - clean_plural
    arm_gap = arm_singular - arm_plural
    return {"clean_gap": clean_gap, "arm_gap": arm_gap, "loss": clean_gap - arm_gap}


def donor_weakening(clean, patched, number):
    _finite(clean, patched)
    if number not in ("singular", "plural"):
        raise ValueError("unrecognized target number")
    return (1 if number == "singular" else -1) * (clean - patched)


def fixed_margins(logits):
    """Calculate the prespecified contrasts from one full finite vocabulary."""
    if type(logits) not in (list, tuple) or len(logits) != 50257:
        raise ValueError("expected the complete 50257-logit vector")
    _finite(*logits)
    return {"present": logits[318] - logits[389], "past": logits[373] - logits[547],
            "the_vs_a": logits[262] - logits[257], "in_vs_on": logits[287] - logits[319]}


def donor_comparisons(*, clean, same_diff, opposite_same, opposite_diff, number):
    """Keep all cases; a strict sign reversal excludes ties at zero."""
    _finite(clean, same_diff, opposite_same, opposite_diff)
    arms = {"same_diff": same_diff, "opposite_same": opposite_same, "opposite_diff": opposite_diff}
    weakening = {name: donor_weakening(clean, value, number) for name, value in arms.items()}
    sign = 1 if number == "singular" else -1
    return {"clean_correct": sign * clean > 0,
            "weakening": weakening,
            "raw_change": {name: value - clean for name, value in arms.items()},
            "absolute_change": {name: abs(value - clean) for name, value in arms.items()},
            "contrasts": {"opposite_same_minus_same_diff": weakening["opposite_same"] - weakening["same_diff"],
                          "opposite_diff_minus_same_diff": weakening["opposite_diff"] - weakening["same_diff"]},
            "margin_reversal": {name: (clean > 0 > value or clean < 0 < value)
                                for name, value in arms.items()}}


def _ordered(rows, dimension, variants):
    if type(rows) is not list or len(rows) != 8 * len(variants):
        raise ValueError("incomplete observation inventory")
    index = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"family", dimension, "u", "v1", "v2"}:
            raise ValueError("unexpected observation fields")
        family, variant = row["family"], row[dimension]
        if type(family) is not int or family not in range(8) or type(variant) is not type(variants[0]) or variant not in variants:
            raise ValueError("unexpected observation identity")
        _finite(row["u"], row["v1"], row["v2"])
        if (family, variant) in index:
            raise ValueError("duplicate observation identity")
        index[family, variant] = row
    return [index[f, v] for f in range(8) for v in variants]


def _summary(rows, floor):
    n = len(rows)
    mean_u = math.fsum(row["u"] for row in rows) / n
    mean_abs_v1 = math.fsum(abs(row["v1"]) for row in rows) / n
    mean_abs_v2 = math.fsum(abs(row["v2"]) for row in rows) / n
    return {"n": n, "mean_u": mean_u, "mean_abs_v1": mean_abs_v1,
            "mean_abs_v2": mean_abs_v2,
            "passed": mean_u >= floor and mean_u > mean_abs_v1 and mean_u > mean_abs_v2}


def evaluate_signed(rows):
    ordered = _ordered(rows, "variant", (0, 1, 2, 3))
    classes = {"singular": _summary([r for r in ordered if r["variant"] in (0, 2)], 0.05),
               "plural": _summary([r for r in ordered if r["variant"] in (1, 3)], 0.05)}
    families = [{"family": f, **_summary([r for r in ordered if r["family"] == f], 0.05)}
                for f in range(8)]
    count = sum(item["passed"] for item in families)
    return {"classes": classes, "families": families, "families_passed": count,
            "passed": all(item["passed"] for item in classes.values()) and count >= 6}


def evaluate_projection(rows):
    ordered = _ordered(rows, "morphology", ("regular", "irregular"))
    aggregate = _summary(ordered, 0.10)
    per_pair = [{**r, "passed": r["u"] >= 0.05 and r["u"] > abs(r["v1"]) and r["u"] > abs(r["v2"])}
                for r in ordered]
    count = sum(item["passed"] for item in per_pair)
    return {**aggregate, "pairs": per_pair, "pairs_passed": count,
            "passed": aggregate["passed"] and count >= 12}


_MARGINS = ("present", "past", "the_vs_a", "in_vs_on")


def _response_group(rows, directions):
    if not rows:
        raise ValueError("empty summary group")
    n = len(rows)
    result = {"n": n, "directions": {}}
    for direction in directions:
        result["directions"][direction] = {}
        for margin in _MARGINS:
            observed = [row["responses"][direction][margin] for row in rows]
            averages = {key: math.fsum(item[key] for item in observed) / n
                        for key in ("plus_change", "minus_change", "odd", "even")}
            slopes = [item["odd_per_unit"] for item in observed]
            averages["odd_per_unit"] = (None if any(value is None for value in slopes)
                                        else math.fsum(slopes) / n)
            result["directions"][direction][margin] = averages
    if "donors" in rows[0]:
        names = ("same_diff", "opposite_same", "opposite_diff")
        donors = [row["donors"] for row in rows]
        result["donors"] = {"n": n, "clean_correct": sum(item["clean_correct"] for item in donors),
            "mean_weakening": {name: math.fsum(item["weakening"][name] for item in donors) / n for name in names},
            "mean_raw_change": {name: math.fsum(item["raw_change"][name] for item in donors) / n for name in names},
            "mean_absolute_change": {name: math.fsum(item["absolute_change"][name] for item in donors) / n for name in names},
            "margin_reversals": {name: sum(item["margin_reversal"][name] for item in donors) for name in names},
            "mean_contrasts": {name: math.fsum(item["contrasts"][name] for item in donors) / n
                               for name in ("opposite_same_minus_same_diff", "opposite_diff_minus_same_diff")}}
    return result


def evaluate_study(cases, scores, doses):
    """Reduce complete fixed scalar observations; full-logit checks are separate.

    ``scores`` maps (case_id, action) to all four finite margin values. A
    separate saved-byte reader must check actual raw payloads, zero-dose bit
    equality, witness landing, source/case identity and external observation.
    """
    schedule = expected_schedule(cases)
    expected = {(item["case_id"], item["action"]) for item in schedule}
    if type(scores) is not dict or set(scores) != expected:
        raise ValueError("scalar result inventory differs")
    if type(doses) is not dict or set(doses) != {case["case_id"] for case in cases}:
        raise ValueError("dose inventory differs")
    for value in scores.values():
        if type(value) is not dict or set(value) != set(_MARGINS):
            raise ValueError("scalar margin inventory differs")
        _finite(*(value[name] for name in _MARGINS))
    for case in cases:
        dose = doses[case["case_id"]]
        if (type(dose) is not float or not math.isfinite(dose) or dose < 0
                or (dose == 0 and math.copysign(1.0, dose) < 0)
                or (case["role"] == "target" and 0 < dose <= 2 ** -19)):
            raise ValueError("invalid nominal dose")
    for family in range(8):
        rows = cases[4 * family:4 * family + 4]
        if (any(row.get("morphology") != ("regular" if i < 2 else "irregular")
                for i, row in enumerate(rows))
                or rows[0].get("attractor_number") not in ("singular", "plural")
                or any(row.get("attractor_number") != rows[0]["attractor_number"] for row in rows)):
            raise ValueError("target pairing differs")
        q = [doses[row["case_id"]] for row in rows]
        if q[0] != q[1] or q[2] != q[3] or doses[cases[32 + family]["case_id"]] != math.fsum((q[0], q[2])) / 2:
            raise ValueError("pair or context doses differ")

    targets, contexts = [], []
    for case in cases:
        name = case["case_id"]
        target = case["role"] == "target"
        clean = scores[name, "clean"]
        noops = ("clean_repeat", "clean_after", "noop_l3_final")
        if target:
            noops += ("noop_l3_subject", "noop_l1_final")
        if any(scores[name, action] != clean for action in noops):
            raise ValueError("clean or no-op margins differ")
        if target and doses[name] == 0 and any(scores[name, action] != clean
                                              for action in ("same_diff", "opp_same", "opp_diff")):
            raise ValueError("zero-dose donor margins differ")
        directions = ("u", "v1", "v2", "l1_u", "subject_u") if target else ("u", "v1", "v2")
        responses = {direction: {margin: scalar_response(
                        clean=clean[margin], plus=scores[name, direction + "_plus"][margin],
                        minus=scores[name, direction + "_minus"][margin], q=doses[name])
                    for margin in _MARGINS} for direction in directions}
        record = {"case_id": name, "family": case["family_index"], "dose": doses[name],
                  "clean": dict(clean), "responses": responses,
                  "arm_margins": {item["action"]: dict(scores[name, item["action"]])
                                  for item in schedule if item["case_id"] == name}}
        if target:
            record.update({"variant": case["variant"], "subject_number": case["subject_number"],
                           "morphology": case["morphology"], "attractor_number": case["attractor_number"],
                           "donors": donor_comparisons(clean=clean["present"],
                               same_diff=scores[name, "same_diff"]["present"],
                               opposite_same=scores[name, "opp_same"]["present"],
                               opposite_diff=scores[name, "opp_diff"]["present"],
                               number=case["subject_number"])})
            targets.append(record)
        else:
            contexts.append(record)

    signed_rows = [{"family": row["family"], "variant": row["variant"],
                    **{direction: row["responses"][direction]["present"]["odd"]
                       for direction in ("u", "v1", "v2")}} for row in targets]
    pairs, projection_rows = [], []
    for family in range(8):
        for variant, morphology in ((0, "regular"), (2, "irregular")):
            singular, plural = cases[4 * family + variant:4 * family + variant + 2]
            left, right = singular["case_id"], plural["case_id"]
            losses = {direction: {margin: projection_loss(
                        scores[left, "clean"][margin], scores[right, "clean"][margin],
                        scores[left, "center_" + direction][margin],
                        scores[right, "center_" + direction][margin])
                      for margin in _MARGINS} for direction in ("u", "v1", "v2")}
            pairs.append({"family": family, "morphology": morphology,
                          "singular_case_id": left, "plural_case_id": right, "losses": losses})
            projection_rows.append({"family": family, "morphology": morphology,
                                    **{direction: losses[direction]["present"]["loss"]
                                       for direction in ("u", "v1", "v2")}})

    target_directions = ("u", "v1", "v2", "l1_u", "subject_u")
    groups = {"families": [_response_group([r for r in targets if r["family"] == f], target_directions)
                           for f in range(8)],
              "subject_number": {v: _response_group([r for r in targets if r["subject_number"] == v], target_directions)
                                 for v in ("singular", "plural")},
              "morphology": {v: _response_group([r for r in targets if r["morphology"] == v], target_directions)
                             for v in ("regular", "irregular")},
              "attractor_number": {v: _response_group([r for r in targets if r["attractor_number"] == v], target_directions)
                                   for v in ("singular", "plural")},
              "contexts": _response_group(contexts, ("u", "v1", "v2"))}
    return {"targets": targets, "contexts": contexts, "pairs": pairs, "groups": groups,
            "signed_prediction": evaluate_signed(signed_rows),
            "projection_prediction": evaluate_projection(projection_rows)}


def decode_float32(raw, *, expected_count):
    if type(raw) is not bytes or type(expected_count) is not int or expected_count < 1 or len(raw) != 4 * expected_count:
        raise ValueError("float32 payload length differs")
    result = []
    for (bits,) in struct.iter_unpack("<I", raw):
        exponent, mantissa = (bits >> 23) & 0xff, bits & 0x7fffff
        if exponent == 0xff:
            raise ValueError("nonfinite float32 payload")
        value = (math.ldexp(float(mantissa), -149) if exponent == 0
                 else math.ldexp(float((1 << 23) + mantissa), exponent - 150))
        result.append(math.copysign(value, -1.0 if bits >> 31 else 1.0))
    return tuple(result)


def family_geometry(captures, u):
    """Recompute the three donor distances and pair-common dose per row."""
    if (type(captures) not in (list, tuple) or len(captures) != 4
            or type(u) not in (list, tuple) or not u or len(u) > 768
            or any(type(value) is not float or not math.isfinite(value) for value in u)
            or abs(math.sqrt(math.fsum(value * value for value in u)) - 1.0) > 1e-12):
        raise ValueError("invalid family captures or direction")
    rows = [decode_float32(raw, expected_count=len(u)) for raw in captures]
    distances = []
    for i in range(4):
        distances.append([math.sqrt(math.fsum((a - b) * (a - b)
                           for a, b in zip(rows[i ^ mask], rows[i]))) for mask in (2, 1, 3)])
    doses, centers, gaps = [0.0] * 4, [0.0] * 4, []
    for singular in (0, 2):
        plural = singular + 1
        q = min(distances[singular] + distances[plural]) / 2.0
        if q <= 2 ** -19:
            q = 0.0
        doses[singular] = doses[plural] = q
        gap = (math.fsum(a * b for a, b in zip(rows[singular], u))
               - math.fsum(a * b for a, b in zip(rows[plural], u)))
        gaps.append(gap)
        center = gap / 2.0
        if abs(center) <= 2 ** -19:
            center = 0.0
        centers[singular], centers[plural] = center, -center if center else 0.0
    return {"distances": distances, "doses": doses, "centering": centers, "centering_gaps": gaps}


def float32_bits(value):
    """Round one finite binary64 value to finite binary32, ties to even."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("expected a finite binary64 value")
    sign = 0x80000000 if math.copysign(1.0, value) < 0 else 0
    if value == 0:
        return sign
    numerator, denominator = abs(value).as_integer_ratio()
    exponent = numerator.bit_length() - denominator.bit_length()
    if ((exponent >= 0 and numerator < (denominator << exponent))
            or (exponent < 0 and (numerator << -exponent) < denominator)):
        exponent -= 1

    def rounded_quotient(n, d):
        quotient, remainder = divmod(n, d)
        if 2 * remainder > d or (2 * remainder == d and quotient & 1):
            quotient += 1
        return quotient

    if exponent < -126:
        significand = rounded_quotient(numerator << 149, denominator)
        return sign | significand
    shift = 23 - exponent
    significand = (rounded_quotient(numerator << shift, denominator) if shift >= 0
                   else rounded_quotient(numerator, denominator << -shift))
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        raise ValueError("rounded value is not finite binary32")
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))


def _delivery(base, original, proposed, requested, extra):
    if requested == 0.0:
        raw = base
    else:
        words = [float32_bits(float(value)) for value in proposed]
        if len(words) != len(original):
            raise ValueError("delivery width differs")
        raw = struct.pack("<" + str(len(words)) + "I", *words)
    actual = decode_float32(raw, expected_count=len(original))
    before_words = [word for (word,) in struct.iter_unpack("<I", base)]
    after_words = [word for (word,) in struct.iter_unpack("<I", raw)]
    realized = math.sqrt(math.fsum((a - b) * (a - b) for a, b in zip(actual, original)))
    source_norm = math.sqrt(math.fsum(value * value for value in original))
    relative_error = None if requested == 0 else abs(realized - requested) / requested
    metadata = {"requested_norm": requested, "realized_norm": realized,
                "dose_valid": raw == base if requested == 0 else relative_error <= 1e-5,
                "relative_error": relative_error, "source_norm": source_norm,
                "realized_over_source_norm": None if source_norm == 0 else realized / source_norm,
                "changed_elements": sum(a != b for a, b in zip(actual, original)),
                "changed_words": sum(a != b for a, b in zip(after_words, before_words)),
                "changed_bit_count": sum((a ^ b).bit_count() for a, b in zip(after_words, before_words)),
                **extra}
    return raw, metadata


def donor_delivery(base, donor, dose):
    if (type(base) is not bytes or not base or len(base) % 4 or len(base) > 768 * 4
            or type(donor) is not bytes or len(donor) != len(base)
            or type(dose) is not float or not math.isfinite(dose) or dose < 0):
        raise ValueError("invalid donor delivery")
    count = len(base) // 4
    original, source = decode_float32(base, expected_count=count), decode_float32(donor, expected_count=count)
    differences = [a - b for a, b in zip(source, original)]
    distance = math.sqrt(math.fsum(value * value for value in differences))
    if dose == 0:
        return _delivery(base, original, (), 0.0, {"donor_distance": distance, "fraction": None})
    if distance == 0 or dose > distance:
        raise ValueError("donor dose exceeds the segment")
    fraction = dose / distance
    proposed = (value + fraction * delta for value, delta in zip(original, differences))
    return _delivery(base, original, proposed, dose, {"donor_distance": distance, "fraction": fraction})


def direction_delivery(base, direction, amount):
    if (type(base) is not bytes or not base or len(base) % 4 or len(base) > 768 * 4
            or type(direction) not in (list, tuple) or len(direction) != len(base) // 4
            or any(type(value) is not float or not math.isfinite(value) for value in direction)
            or abs(math.sqrt(math.fsum(value * value for value in direction)) - 1.0) > 1e-12
            or type(amount) is not float or not math.isfinite(amount)):
        raise ValueError("invalid direction delivery")
    original = decode_float32(base, expected_count=len(direction))
    if amount == 0:
        return _delivery(base, original, (), 0.0, {"signed_amount": 0.0})
    proposed = (value + amount * change for value, change in zip(original, direction))
    return _delivery(base, original, proposed, abs(amount), {"signed_amount": amount})


def verify_hook_witness(before, after, whole_before, whole_after, *, width, sequence_length, token_index):
    if (type(width) is not int or not 1 <= width <= 768
            or type(sequence_length) is not int or not 1 <= sequence_length <= 32
            or type(token_index) is not int or not 0 <= token_index < sequence_length):
        raise ValueError("invalid hook coordinates")
    for payload, count in ((before, width), (after, width),
                           (whole_before, width * sequence_length), (whole_after, width * sequence_length)):
        decode_float32(payload, expected_count=count)
    start, end = token_index * width * 4, (token_index + 1) * width * 4
    if (before != whole_before[start:end] or after != whole_after[start:end]
            or whole_before[:start] != whole_after[:start]
            or whole_before[end:] != whole_after[end:]):
        raise ValueError("hook slice or untouched positions differ")
    return True


def expected_hook_plan(action, *, sequence_length, width):
    """Construct the fixed hook site's declared five-step capture/patch plan."""
    noops = ("noop_l3_final", "noop_l3_subject", "noop_l1_final")
    if (type(action) is not str or action not in set(_TARGET_ARMS + _CONTEXT_ARMS + noops)
            or type(sequence_length) is not int or not 2 <= sequence_length <= 32
            or type(width) is not int or not 1 <= width <= 768):
        raise ValueError("invalid hook plan request")
    layer = 1 if action == "noop_l1_final" or action.startswith("l1_") else 3
    role = "subject" if action == "noop_l3_subject" or action.startswith("subject_") else "last"
    module = f"transformer.h.{layer}.attn"
    site = {"id": f"{layer}:attn:{role}", "module_path": module, "layer": layer,
            "component": "attn", "token_role": role, "output_selector": [0]}
    names = ("whole_before", "before", "action", "after", "whole_after")
    shape, token = [1, sequence_length, width], (1 if role == "subject" else sequence_length - 1)
    steps = []
    for index, name in enumerate(names):
        whole = name in ("whole_before", "whole_after")
        operation = ("null" if action in noops else "patch") if name == "action" else "capture"
        steps.append({"batch_index": None if whole else 0, "call_site_known": False,
            "device": "cpu", "dtype": "torch.float32", "forward_index": 0, "invocation": 0,
            "module_aliases": [module], "module_path": module, "name": name,
            "operation": operation, "order": index, "output_selector": [0], "output_shape": shape,
            "preceding_same_site_actions": list(names[:index]), "selected_shape": shape if whole else [width],
            "source_capture": None, "token_index": None if whole else token})
    return site, {"capture_bytes": 2 * (sequence_length * width + width) * 4,
                  "source_bytes": 0 if action in noops else width * 4, "steps": steps}


def read_saved_tensor(directory, reference, expected_shape):
    """Read one bounded content-addressed float32 payload without model code."""
    if (type(reference) is not dict or set(reference) != {"sha256", "shape", "dtype", "bytes"}
            or type(expected_shape) is not tuple or not 1 <= len(expected_shape) <= 8
            or any(type(dim) is not int or dim <= 0 for dim in expected_shape)):
        raise ValueError("invalid tensor reference")
    digest, shape, size = reference["sha256"], reference["shape"], reference["bytes"]
    if (type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest, re.ASCII) is None
            or type(shape) is not list or any(type(dim) is not int for dim in shape)
            or tuple(shape) != expected_shape or type(size) is not int
            or size != 4 * math.prod(expected_shape) or size > 1024 * 1024
            or reference["dtype"] != "float32-le"):
        raise ValueError("tensor reference does not match the expected shape")
    try:
        path = Path(directory) / (digest + ".f32le")
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                or info.st_size != size):
            raise ValueError("tensor file is absent, aliased or has the wrong size")
        with path.open("rb") as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("tensor bytes differ from their reference")
        return raw
    except OSError as exc:
        raise ValueError("tensor file could not be read") from exc


def vocabulary_diagnostics(clean, arm, target_ids, *, top_n=20):
    if (not clean or len(clean) != len(arm) or type(target_ids) not in (list, tuple)
            or type(top_n) is not int or top_n < 1):
        raise ValueError("invalid vocabulary inputs")
    _finite(*clean, *arm)
    if any(type(i) is not int or i < 0 or i >= len(clean) for i in target_ids):
        raise ValueError("invalid target token")

    def log_probabilities(values):
        maximum = max(values)
        normalizer = math.log(math.fsum(math.exp(value - maximum) for value in values))
        return [(value - maximum) - normalizer for value in values]

    def ranks(values):
        return [1 + sum(value > values[i] or (value == values[i] and j < i)
                        for j, value in enumerate(values)) for i in target_ids]

    clean_logp, arm_logp = log_probabilities(clean), log_probabilities(arm)
    kl = math.fsum(math.exp(p) * (p - q) if math.exp(p) else 0.0
                   for p, q in zip(clean_logp, arm_logp))
    top_clean = sorted(range(len(clean)), key=lambda i: (-clean[i], i))[:top_n]
    top_arm = sorted(range(len(arm)), key=lambda i: (-arm[i], i))[:top_n]
    return {"top_clean": top_clean, "top_arm": top_arm,
            "top_clean_logits": [clean[i] for i in top_clean], "top_arm_logits": [arm[i] for i in top_arm],
            "target_logits_clean": [clean[i] for i in target_ids], "target_logits_arm": [arm[i] for i in target_ids],
            "ranks_clean": ranks(clean), "ranks_arm": ranks(arm), "kl_clean_to_arm": kl}
