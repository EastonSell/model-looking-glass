"""Model-free replay of the prospective masked-antonym baseline screen.

The caller must independently establish the manifest's source and chronology,
then supply its exact SHA-256. This validates saved records and screen math;
it does not attest actual execution, source loading, exposure, or later edits.
"""

import argparse
import base64
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

MAX_JOURNAL = 128 * 1024 * 1024
MAX_LINE = 2 * 1024 * 1024
ORDER = ("box", "rope", "drink", "room")
ROLES = ("discovery", "calibration-01", "calibration-02")
SPEC = {
    "box": ("small", "large", ("small", "little")),
    "rope": ("short", "long", ("short",)),
    "drink": ("cold", "hot", ("cold",)),
    "room": ("dark", "bright", ("dark", "dim")),
}
THRESHOLD = Fraction(1, 10)
TAIL_CUTOFF = Decimal(-1200)
START_KEYS = {"action", "attention_mask", "case_id", "condition", "input_ids", "kind", "mask_index", "object", "phase", "seq"}
COMPLETE_KEYS = START_KEYS | {"logits", "parameter", "parameter_sha256", "realized_norm"}


def require_denormal_support():
    """Fail closed if this reader thread flushes either signed f32/f64 tail."""
    positive = bytes.fromhex("0100000000000000")
    negative = bytes.fromhex("0100000000000080")
    pos = struct.unpack("<d", positive)[0]
    neg = struct.unpack("<d", negative)[0]
    if not (pos > 0 and neg < 0 and struct.pack("<d", pos * 1.) == positive
            and struct.pack("<d", neg * 1.) == negative
            and struct.pack("<d", float("5e-324")) == positive
            and struct.pack("<d", float("-5e-324")) == negative
            and struct.unpack("<f", bytes.fromhex("01000000"))[0] == 2. ** -149
            and struct.unpack("<f", bytes.fromhex("01000080"))[0] == -(2. ** -149)):
        raise ValueError("binary32/64 subnormals are not preserved")


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def _exact_int(value, expected):
    return type(value) is int and value == expected


def _exact_ids(value, expected):
    return (type(value) is list and type(expected) is list and len(value) == len(expected)
            and all(type(a) is int and type(b) is int and a == b for a, b in zip(value, expected)))


def _f32(envelope, count):
    if (type(envelope) is not dict or set(envelope) != {"encoding", "bytes", "sha256", "data"}
            or envelope["encoding"] != "f32le" or not _exact_int(envelope["bytes"], count * 4)
            or type(envelope["sha256"]) is not str or type(envelope["data"]) is not str):
        raise ValueError("invalid saved full-logit envelope")
    try:
        raw = base64.b64decode(envelope["data"], validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid base64 full logits") from error
    if len(raw) != 4 * count or hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
        raise ValueError("full-logit byte length or digest differs")
    values = struct.unpack("<" + str(count) + "f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite full-logit value")
    return values


def decimal_summary(values, ids):
    """Compute selected full-vocabulary summaries, retaining tiny tails.

    Values are exact finite float32s decoded to binary64. Terms below -1200
    nats are omitted and an absolute bound on the missing normalizer correction
    is retained. Finite precision is a cross-check, not a global certificate.
    """
    if not values or not all(type(x) is float and math.isfinite(x) for x in values):
        raise ValueError("expected finite float32-decoded values")
    if any(type(i) is not int or i < 0 or i >= len(values) for i in ids.values()):
        raise ValueError("invalid token id")
    peak = max(values)
    below = max((value for value in values if value < peak), default=None)
    # Every exact f32 can be converted to Decimal exactly; 220 digits also
    # suffice for the difference of any two finite f32 values. Increase the
    # precision when the first correction to the peak is far below one.
    gap = min(1200.0, peak - below) if below is not None else 0.0
    precision = max(220, int(math.ceil(gap / math.log(10.0))) + 120)
    if precision > 700:
        raise ValueError("adaptive precision exceeds bounded contract")
    with localcontext() as ctx:
        ctx.prec = precision
        exact = [Decimal.from_float(value) for value in values]
        maximum = max(exact)
        deltas = [value - maximum for value in exact]
        kept = [delta.exp() for delta in deltas if delta >= TAIL_CUTOFF]
        omitted = len(deltas) - len(kept)
        denominator = sum(kept, Decimal(0))
        if denominator < 1:
            raise ValueError("invalid shifted partition")
        log_normalizer = denominator.ln()
        # log(1 + omitted_weight / kept_weight) <= omitted_weight because the
        # kept partition includes at least one peak term. Decimal.exp is
        # rounded: advance one representable value and round multiplication up.
        ctx.rounding = ROUND_CEILING
        omission_bound = TAIL_CUTOFF.exp().next_plus() * omitted
        ctx.rounding = ROUND_HALF_EVEN
        selected = {}
        for name, token in ids.items():
            logp = deltas[token] - log_normalizer
            probability = Decimal(0) if logp < TAIL_CUTOFF else logp.exp()
            selected[name] = {
                "id": token,
                "logit": values[token],
                "logit_f32le": struct.pack("<f", values[token]).hex(),
                "rank": 1 + sum(value > values[token] for value in values),
                "log_probability": float(logp),
                "probability": float(probability),
            }
        first = deltas[ids["factual"]] - log_normalizer
        second = deltas[ids["counterfactual"]] - log_normalizer
        pair = sum((x.exp() if x >= TAIL_CUTOFF else Decimal(0) for x in (first, second)), Decimal(0))
        correction_text = str(log_normalizer)
        bound_text = str(omission_bound)
    top_ids = [i for i, value in enumerate(values) if value == peak]
    top = top_ids[0] if len(top_ids) == 1 else None
    return {
        "top_ids": top_ids,
        "unique_top": top,
        "top_representative": top_ids[0],
        "top_five_ids": sorted(range(len(values)), key=lambda i: (-values[i], i))[:5],
        "tokens": selected,
        "pair_probability_mass": float(pair),
        "decimal_precision_digits": precision,
        "shifted_log_normalizer_decimal": correction_text,
        "omitted_tail_terms": omitted,
        "log_normalizer_omission_upper_bound_decimal": bound_text,
    }


def _target_row(row, candidate, role, vocabulary):
    factual, cue, accepted = SPEC[candidate]
    if (type(row) is not dict or row.get("object") != candidate or row.get("role") != role
            or type(row.get("id")) is not str or not row["id"]
            or row.get("factual") != factual or row.get("counterfactual") != cue
            or row.get("wrong_target") != "red" or row.get("accepted") != list(accepted)):
        raise ValueError("target row does not match fixed antonym specification")
    token_map = row.get("word_ids")
    if type(token_map) is not dict:
        raise ValueError("missing token ids")
    words = tuple(dict.fromkeys((factual, cue, "red", *accepted)))
    if any(type(token_map.get(word)) is not int or not 0 <= token_map[word] < vocabulary for word in words):
        raise ValueError("invalid focal or accepted token id")
    if vocabulary == 30522 and any(token_map[word] in {0, 100, 101, 102, 103} for word in words):
        raise ValueError("focal and accepted tokens must be non-special")
    if len({token_map[word] for word in words}) != len(words):
        raise ValueError("distinct target words alias the same token id")
    input_ids, mask = row.get("input_ids"), row.get("attention_mask")
    position = row.get("mask_index")
    if (type(input_ids) is not list or not input_ids or
            any(type(token) is not int or not 0 <= token < 30522 for token in input_ids) or
            type(mask) is not list or len(mask) != len(input_ids) or
            any(type(bit) is not int or bit not in (0, 1) for bit in mask) or
            type(position) is not int or not 0 <= position < len(input_ids) or
            input_ids[position] != 103 or input_ids.count(103) != 1 or mask[position] != 1):
        raise ValueError("invalid token, attention mask, or unique mask position")
    return {"factual": token_map[factual], "counterfactual": token_map[cue],
            "wrong_target": token_map["red"], **{f"accepted_{i}": token_map[word]
                                                     for i, word in enumerate(accepted)}}


def audit_stream(manifest_raw, stream, *, expected_manifest_sha, vocabulary):
    require_denormal_support()
    if (type(expected_manifest_sha) is not str or re.fullmatch("[0-9a-f]{64}", expected_manifest_sha) is None
            or hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha):
        raise ValueError("manifest digest mismatch")
    if type(vocabulary) is not int or not 4 <= vocabulary <= 30522:
        raise ValueError("invalid vocabulary size")
    manifest = _json(manifest_raw)
    if manifest.get("candidate_order") != list(ORDER) or type(manifest.get("cases")) is not list:
        raise ValueError("fixed candidate order or cases differ")
    if vocabulary == 30522 and len(manifest["cases"]) != 55:
        raise ValueError("production manifest must contain the 55 fixed cases")
    if any(type(row) is not dict or type(row.get("id")) is not str for row in manifest["cases"]):
        raise ValueError("invalid case record")
    ids = [row["id"] for row in manifest["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case identity")
    rows = []
    for candidate in ORDER:
        for role in ROLES:
            matched = [row for row in manifest["cases"] if row.get("object") == candidate and row.get("role") == role]
            if len(matched) != 1:
                raise ValueError("missing or duplicate candidate/role")
            rows.append(matched[0])

    digest = hashlib.sha256()
    byte_count = 0
    candidates = {name: True for name in ORDER}
    observed = []
    for action, row in enumerate(rows):
        candidate = ORDER[action // 3]
        role = ROLES[action % 3]
        token_ids = _target_row(row, candidate, role, vocabulary)
        pair = []
        for offset, kind in enumerate(("start", "complete")):
            line = stream.readline(MAX_LINE + 1)
            byte_count += len(line)
            if not line or len(line) > MAX_LINE or not line.endswith(b"\n") or byte_count > MAX_JOURNAL:
                raise ValueError("incomplete or oversized event journal")
            digest.update(line)
            event = _json(line)
            expected_keys = START_KEYS if kind == "start" else COMPLETE_KEYS
            if (type(event) is not dict or set(event) != expected_keys
                    or not _exact_int(event.get("seq"), 2 * action + offset)
                    or event.get("kind") != kind or not _exact_int(event.get("action"), action)
                    or event.get("phase") != "screen" or event.get("condition") != "original"
                    or event.get("object") != candidate or event.get("case_id") != row["id"]
                    or not _exact_ids(event.get("input_ids"), row["input_ids"])
                    or not _exact_ids(event.get("attention_mask"), row["attention_mask"])
                    or not _exact_int(event.get("mask_index"), row["mask_index"])):
                raise ValueError("ordered event or model-entry identity differs")
            pair.append(event)
        done = pair[1]
        if (not {"parameter", "parameter_sha256", "realized_norm"}.issubset(done)
                or done["parameter"] is not None or done["parameter_sha256"] is not None
                or type(done["realized_norm"]) is not float or done["realized_norm"] != 0.0):
            raise ValueError("screen must retain its unedited zero-norm state")
        values = _f32(done.get("logits"), vocabulary)
        details = decimal_summary(values, token_ids)
        exact = Fraction.from_float(values[token_ids["factual"]]) - Fraction.from_float(values[token_ids["counterfactual"]])
        margin_pass = exact > THRESHOLD
        discovery_top_pass = role != "discovery" or details["unique_top"] == token_ids["factual"]
        eligible = margin_pass and discovery_top_pass
        candidates[candidate] &= eligible
        observed.append({"case_id": row["id"], "object": candidate, "role": role,
                         "accepted": row["accepted"], "margin": float(exact),
                         "margin_exact": {"numerator": exact.numerator, "denominator": exact.denominator},
                         "margin_pass": margin_pass, "discovery_top_pass": discovery_top_pass,
                         "eligible": eligible, **details})
    if stream.read(1):
        raise ValueError("unexpected extra journal data")
    selected = next((candidate for candidate in ORDER if candidates[candidate]), None)
    require_denormal_support()
    return {"schema": "independent-antonym-screen-review-v1", "manifest_sha256": expected_manifest_sha,
            "journal_sha256": digest.hexdigest(), "journal_bytes": byte_count,
            "candidate_eligible": candidates, "selected_object": selected,
            "status": "eligible" if selected is not None else "baseline-ineligible", "rows": observed,
            "selection_rule": "discovery factual unique top and all three exact f32 margins > 1/10",
            "scope": "Saved screen records and numerical cross-check only; source, chronology, exposure and execution require separate evidence. Later edit/locality gates require a separate reader."}


def main():
    require_denormal_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.journal.open("rb") as journal:
        report = audit_stream(args.manifest.read_bytes(), journal,
                              expected_manifest_sha=args.manifest_sha256, vocabulary=30522)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as output:
        if output.write(raw) != len(raw):
            raise OSError("short report write")
    print("independent screen report SHA-256", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
