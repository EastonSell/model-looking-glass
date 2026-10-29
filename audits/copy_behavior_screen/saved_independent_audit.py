"""Independent, model-free replay of the fixed copying baseline screen.

The fixed decision and double-precision gap arithmetic follow the public
protocol; no model, tokenizer, or project modules are imported. Source and
process observations are checked separately from numerical reconstruction.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


VOCABULARY = 50_257
ROW_BYTES = 4 * VOCABULARY
CASE_COUNT = 192
CASE_SHA256 = "96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734"
PROTOCOL_SHA256 = "78907b74bf3edb44a47c794508a9385da23b2bf2f26d07df4fefe74542b58875"
AMENDMENT_SHA256 = "832d115a81dc58359f5ba5189222253ea002036fa46a37f99b1813b5f51a8f01"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
WEIGHTS_SHA256 = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"


def _binary32_value(word):
    """Convert IEEE bits to a normal-range Python float, including subnormals."""
    exponent = (word >> 23) & 255
    fraction = word & 0x7fffff
    if exponent == 255:
        raise ValueError("nonfinite saved logit")
    coefficient = fraction if exponent == 0 else (1 << 23) | fraction
    power = -149 if exponent == 0 else exponent - 150
    number = math.ldexp(coefficient, power)
    return -number if word >> 31 else number


def summarize_row(payload, target):
    if not isinstance(payload, bytes) or len(payload) != ROW_BYTES:
        raise ValueError("saved row must contain exactly one vocabulary")
    if type(target) is not int or not 0 <= target < VOCABULARY:
        raise ValueError("invalid target token")
    values = [_binary32_value(x[0]) for x in struct.iter_unpack("<I", payload)]
    top = max(values)
    tied = [index for index, value in enumerate(values) if value == top]
    chosen = values[target]
    shifted_log_sum = math.log(math.fsum(math.exp(value - top) for value in values))
    return {
        "target_id": target,
        "target_logit": chosen,
        "maximum": top,
        "shifted_log_sum": shifted_log_sum,
        "target_log_probability": (chosen - top) - shifted_log_sum,
        "target_rank": 1 + sum(value > chosen for value in values),
        "top_representative_id": tied[0],
        "top_tie_count": len(tied),
        "top_unique_id": tied[0] if len(tied) == 1 else None,
    }


def gap(repeated, shuffled):
    if repeated["target_id"] != shuffled["target_id"]:
        raise ValueError("paired rows must score the same target")
    return math.fsum((repeated["target_logit"], -shuffled["target_logit"],
                      -repeated["maximum"], shuffled["maximum"],
                      -repeated["shifted_log_sum"], shuffled["shifted_log_sum"]))


def check_case_order(cases):
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("the fixed screen requires all 192 cases")
    identifiers = set()
    for index, case in enumerate(cases):
        group, variant = divmod(index, 8)
        group += 1
        if (case.get("group") != group or case.get("variant") != variant
                or case.get("rotation") != variant // 4
                or case.get("shuffle") != variant % 4
                or case.get("split") != ("development" if group % 2 else "held-out")):
            raise ValueError("case group, order, rotation, shuffle or split differs")
        ident = case.get("case_id")
        if not isinstance(ident, str) or ident in identifiers:
            raise ValueError("duplicate or invalid case identifier")
        identifiers.add(ident)
        ids = case.get("token_ids")
        if (not isinstance(ids, list) or len(ids) != 11
                or any(type(token) is not int or not 0 <= token < VOCABULARY for token in ids)):
            raise ValueError("invalid fixed input token IDs")
        target = case.get("target_id")
        if type(target) is not int or not 0 <= target < VOCABULARY:
            raise ValueError("invalid fixed target")
    for first in range(0, CASE_COUNT, 8):
        targets = [case["target_id"] for case in cases[first:first + 8]]
        if (len(set(targets[:4])) != 1 or len(set(targets[4:])) != 1
                or targets[0] == targets[4]):
            raise ValueError("the two rotations must score two fixed distinct targets")


def reconstruct(cases, raw_rows):
    check_case_order(cases)
    if len(raw_rows) != CASE_COUNT:
        raise ValueError("incomplete screen: 192 raw rows are required")
    rows = []
    for case, payload in zip(cases, raw_rows):
        score = summarize_row(payload, case["target_id"])
        rows.append({"case_id": case["case_id"], "variant": case["variant"], **score})
    families = []
    for start in range(0, CASE_COUNT, 8):
        group_rows = rows[start:start + 8]
        failures, gaps = [], []
        for rotation in range(2):
            repeat = group_rows[4 * rotation]
            if repeat["top_unique_id"] != repeat["target_id"]:
                failures.append({"rotation": rotation, "reason": "repeat-target-not-unique-top"})
            for shuffle in range(1, 4):
                distance = gap(repeat, group_rows[4 * rotation + shuffle])
                gaps.append({"rotation": rotation, "shuffle": shuffle, "gap": distance})
                if distance < 0.5:
                    failures.append({"rotation": rotation, "shuffle": shuffle,
                                     "reason": "gap-below-0.5"})
        group = start // 8 + 1
        families.append({"group": group, "split": "development" if group % 2 else "held-out",
                         "baseline_status": "unsupported" if failures else "supported",
                         "reasons": failures, "gaps": gaps, "rows": group_rows})
    return {"families": families,
            "supported_development": sum(f["baseline_status"] == "supported"
                                         for f in families if f["split"] == "development"),
            "supported_held_out": sum(f["baseline_status"] == "supported"
                                      for f in families if f["split"] == "held-out")}


def load_fixed_cases(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA256:
        raise ValueError("case bytes do not match the frozen public proposal")
    cases = json.loads(raw)["cases"]
    check_case_order(cases)
    return cases


def _read_pinned(path, expected, limit):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("a SHA-256 pin is required")
    if path.stat().st_size > limit:
        raise ValueError(f"metadata exceeds its bound: {path.name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected.lower():
        raise ValueError(f"saved bytes differ from the supplied pin: {path.name}")
    return data


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {token}")))


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _close(actual, expected):
    if type(actual) not in (int, float) or not math.isfinite(actual):
        return False
    return abs(actual - expected) <= max(1e-12, 4 * math.ulp(expected))


def _match_summary(rebuilt, saved):
    for key in ("supported_development", "supported_held_out"):
        if saved.get(key) != rebuilt[key]:
            raise ValueError(f"saved {key} differs from raw logits")
    actual_families = saved.get("families")
    if not isinstance(actual_families, list) or len(actual_families) != 24:
        raise ValueError("saved family summary is incomplete")
    for reference, observed in zip(rebuilt["families"], actual_families):
        for key in ("group", "split", "baseline_status", "reasons"):
            if observed.get(key) != reference[key]:
                raise ValueError(f"family {reference['group']} differs: {key}")
        if len(observed.get("gaps", [])) != 6 or len(observed.get("rows", [])) != 8:
            raise ValueError("saved family gaps or rows are incomplete")
        for expected_gap, saved_gap in zip(reference["gaps"], observed["gaps"]):
            if (saved_gap.get("rotation") != expected_gap["rotation"]
                    or saved_gap.get("shuffle") != expected_gap["shuffle"]
                    or not _close(saved_gap.get("gap"), expected_gap["gap"])):
                raise ValueError("saved repeat/shuffle gap differs")
        for expected_row, saved_row in zip(reference["rows"], observed["rows"]):
            for key in ("case_id", "variant", "target_id", "target_rank", "top_unique_id",
                        "top_representative_id", "top_tie_count", "target_logit", "maximum"):
                if saved_row.get(key) != expected_row[key]:
                    raise ValueError(f"saved row differs: {key}")
            for saved_key, our_key in (("logz_shift", "shifted_log_sum"),
                                       ("target_log_probability", "target_log_probability")):
                if not _close(saved_row.get(saved_key), expected_row[our_key]):
                    raise ValueError(f"saved row differs: {saved_key}")


def audit_directory(directory, cases, manifest_pin, journal_pin, result_pin, *, synthetic=False):
    """Read a closed run once. Synthetic mode must be explicitly marked by its fixture."""
    manifest_raw = _read_pinned(directory / "manifest.json", manifest_pin, 1024 * 1024)
    journal_raw = _read_pinned(directory / "attempt.jsonl", journal_pin, 32 * 1024 * 1024)
    result_raw = _read_pinned(directory / "result.json", result_pin, 32 * 1024 * 1024)
    manifest, result = _json(manifest_raw), _json(result_raw)
    if result.get("synthetic_fixture") is True:
        if not synthetic or manifest.get("synthetic_fixture") is not True:
            raise ValueError("synthetic fixture is not an actual model run")
    elif synthetic:
        raise ValueError("synthetic mode requires an explicitly marked fixture")
    elif manifest.get("synthetic_fixture") is True:
        raise ValueError("a synthetic manifest cannot be treated as an actual run")
    if ((directory / "failure.json").exists()
            or manifest.get("schema") != "distilgpt2-copy-behavior-screen-run-v1"
            or result.get("schema") != "distilgpt2-copy-behavior-screen-result-v1"
            or result.get("status") != "complete"
            or result.get("manifest_sha256") != manifest_pin.lower()
            or manifest.get("cases_sha256") != CASE_SHA256):
        raise ValueError("run is incomplete or its manifest binding differs")
    if not synthetic and (manifest.get("protocol_sha256") != PROTOCOL_SHA256
                          or manifest.get("amendment_sha256") != AMENDMENT_SHA256
                          or manifest.get("raw_limit") != CASE_COUNT * ROW_BYTES
                          or manifest.get("model_id") != "distilbert/distilgpt2"
                          or manifest.get("model_revision") != MODEL_REVISION
                          or manifest.get("weights_sha256") != WEIGHTS_SHA256
                          or result.get("checkpoint", {}).get("weights_sha256") != WEIGHTS_SHA256):
        raise ValueError("checkpoint identity differs")
    if (result.get("counts") != {"submitted": CASE_COUNT, "model_returns": CASE_COUNT,
                                 "returned": CASE_COUNT, "retained": CASE_COUNT * ROW_BYTES}
            or not _is_sha(result.get("state_sha256_before"))
            or not _is_sha(result.get("buffers_sha256_before"))
            or result.get("state_sha256_before") != result.get("state_sha256_after")
            or result.get("buffers_sha256_before") != result.get("buffers_sha256_after")):
        raise ValueError("counts or state/buffer checks differ")
    events = [_json(line) for line in journal_raw.splitlines()]
    if (len(events) != 1 + 3 * CASE_COUNT or events[0].get("event") != "runtime_ready"
            or events[0].get("runtime") != result.get("runtime")):
        raise ValueError("journal length or initial runtime differs")
    raw_path = directory / "screen.f32"
    if raw_path.stat().st_size != CASE_COUNT * ROW_BYTES:
        raise ValueError("raw screen length differs")
    raw = raw_path.read_bytes()
    if (len(raw) != CASE_COUNT * ROW_BYTES
            or hashlib.sha256(raw).hexdigest() != result.get("raw_sha256")):
        raise ValueError("raw screen length or digest differs")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != CASE_COUNT:
        raise ValueError("saved records are incomplete")
    rows = []
    for index, (case, recorded) in enumerate(zip(cases, records)):
        begin, saved, complete = events[1 + 3 * index:4 + 3 * index]
        payload = raw[index * ROW_BYTES:(index + 1) * ROW_BYTES]
        raw_ref = {"logits": {"offset": index * ROW_BYTES, "bytes": ROW_BYTES,
                              "sha256": hashlib.sha256(payload).hexdigest()}}
        expected_record = {"case_id": case["case_id"], "index": index, "group": case["group"],
                           "variant": case["variant"], "target_id": case["target_id"], "raw": raw_ref}
        if (begin != {"event": "start", "case_id": case["case_id"], "index": index}
                or saved != {"event": "raw_recorded", "case_id": case["case_id"], "index": index,
                             "raw": raw_ref}
                or complete != {"event": "complete", **expected_record}
                or recorded != expected_record):
            raise ValueError(f"journal or row binding differs at index {index}")
        rows.append(payload)
    summary = reconstruct(cases, rows)
    _match_summary(summary, result["summary"])
    return {"schema": "copy-behavior-screen-independent-replay-v1", "synthetic_fixture": synthetic,
            "cases_sha256": CASE_SHA256, "manifest_sha256": manifest_pin.lower(),
            "journal_sha256": journal_pin.lower(), "result_sha256": result_pin.lower(),
            "raw_sha256": result["raw_sha256"], "row_count": CASE_COUNT,
            "raw_bytes": len(raw), "summary": summary,
            "scope": "saved-byte reconstruction only; source, process and chronology are separate observations"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--synthetic", action="store_true", help="only accept an explicitly marked fake fixture")
    args = parser.parse_args()
    report = audit_directory(args.run, load_fixed_cases(args.cases), args.manifest_sha256,
                             args.journal_sha256, args.result_sha256, synthetic=args.synthetic)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
