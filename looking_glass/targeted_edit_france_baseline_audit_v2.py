"""Independently reconstruct saved baseline logits using only the stdlib.

This checks closed saved bytes and reported identities. It does not load a
model, authenticate execution or publication chronology, prove source or
public-input custody, validate a protected split, or attest process cleanup.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

CASE_SHA256 = "c8dd418bea948c8ca2b4d1e39e696374a28bd2f911bb16707415c38170fbba82"
CASE_IDS = ("fit", "calibration-1", "calibration-2") + tuple(
    f"T{i:02d}" for i in range(1, 9)) + tuple(f"L{i:02d}" for i in range(1, 17))
EXPOSED = frozenset(("L01", "L02", "L05"))
PROBE = {"binary32": {"normal_times_half_bits": 4194304,
                      "numpy_cast_preserves_min_subnormal": True, "subnormal_times_one_bits": 1},
         "binary64_bits": [1, 0x8000000000000001], "signed_binary32_bits": [1, -2147483647]}


def _reject(value):
    raise ValueError("nonfinite JSON number: " + value)


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON float")
    return parsed


def _pairs(entries):
    result = {}
    for key, value in entries:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_reject)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid JSON") from error


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def _read(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or linked evidence file")
    with path.open("rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("evidence file exceeds its bound")
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _same(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(_same(a[key], b[key]) for key in a)
    if type(a) is list:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_word(word):
    """Decode finite little-endian IEEE binary32 bits without a float32 cast."""
    if type(word) is not int or not 0 <= word <= 0xffffffff:
        raise ValueError("invalid float32 bit word")
    sign, exponent, fraction = word >> 31, (word >> 23) & 255, word & 0x7fffff
    if exponent == 255:
        raise ValueError("nonfinite float32 bits")
    if exponent == 0:
        value = math.ldexp(float(fraction), -149)
    else:
        value = math.ldexp(float((1 << 23) + fraction), exponent - 150)
    return -value if sign else value


def top_details(words):
    maximum = second = -math.inf
    representative, count, length = None, 0, 0
    for index, word in enumerate(words):
        value = decode_word(word)
        length += 1
        if value > maximum:
            maximum, second, representative, count = value, maximum, index, 1
        elif value == maximum:
            count += 1
            second = maximum
        elif value > second:
            second = value
    if length < 2:
        raise ValueError("top-token row is too short")
    return {"unique_top_id": representative if count == 1 else None,
            "top_tie_count": count, "top_representative_id": representative,
            "top_gap": maximum - second}


def _fixed(rows, locality, true_id, alternative_id, vocabulary):
    _require(type(vocabulary) is int and vocabulary >= 3 and type(true_id) is int
             and type(alternative_id) is int and true_id != alternative_id
             and 0 <= true_id < vocabulary and 0 <= alternative_id < vocabulary,
             "invalid fixed vocabulary or target IDs")
    _require(type(rows) is list and len(rows) == 27, "wrong fixed row count")
    fixed = []
    for expected, row in zip(CASE_IDS, rows):
        _require(type(row) is dict and set(row) == {"case_id", "token_ids", "attention_mask"}
                 and row["case_id"] == expected and type(row["token_ids"]) is list
                 and 1 <= len(row["token_ids"]) <= 128
                 and all(type(x) is int and 0 <= x < vocabulary for x in row["token_ids"])
                 and type(row["attention_mask"]) is list
                 and len(row["attention_mask"]) == len(row["token_ids"])
                 and all(type(x) is int and x == 1 for x in row["attention_mask"]), "invalid fixed row")
        fixed.append({"case_id": expected, "token_ids": list(row["token_ids"]),
                      "attention_mask": list(row["attention_mask"])})
    _require(len({tuple(row["token_ids"]) for row in fixed}) == 27, "duplicate fixed row")
    _require(type(locality) in (list, tuple) and len(locality) == 16, "wrong locality count")
    labels = []
    for index, item in enumerate(locality, 1):
        name = f"L{index:02d}"
        _require(type(item) is dict and set(item) == {"case_id", "accepted_ids"}
                 and item["case_id"] == name and type(item["accepted_ids"]) in (list, tuple)
                 and len(item["accepted_ids"]) > 0
                 and all(type(x) is int and 0 <= x < vocabulary for x in item["accepted_ids"])
                 and len(set(item["accepted_ids"])) == len(item["accepted_ids"]), "invalid locality labels")
        labels.append({"case_id": name, "accepted_ids": list(item["accepted_ids"]), "new": name not in EXPOSED})
    return fixed, labels


def audit_rows(output, rows, locality, true_id, alternative_id, vocabulary):
    """Audit one closed saved bundle using caller-supplied fixed descriptions."""
    rows, locality = _fixed(rows, locality, true_id, alternative_id, vocabulary)
    output = Path(output)
    _require(output.is_dir() and not output.is_symlink(), "invalid evidence directory")
    expected_files = {"started.json", "events.jsonl", "screen.f32", "provisional.json", "completed.json"}
    _require({item.name for item in output.iterdir()} == expected_files, "incomplete or unexpected evidence inventory")
    started_bytes = _read(output / "started.json", 1_000_000)
    result_bytes = _read(output / "provisional.json", 2_000_000)
    completed_bytes = _read(output / "completed.json", 100_000)
    started, result, completed = map(_decode, (started_bytes, result_bytes, completed_bytes))
    _require(all(raw == _canonical(doc) for raw, doc in ((started_bytes, started),
             (result_bytes, result), (completed_bytes, completed))), "noncanonical saved JSON")
    _require(type(started) is dict and set(started) == {"identity", "rows", "vocabulary", "scope"}
             and type(started["identity"]) is dict and type(started["rows"]) is int and started["rows"] == 27
             and type(started["vocabulary"]) is int and started["vocabulary"] == vocabulary,
             "started record differs")
    _require(type(completed) is dict and set(completed) == {"identity", "provisional_sha256", "status", "cleanup"}
             and _same(completed["identity"], started["identity"])
             and completed["status"] == "provisional-complete" and completed["cleanup"] == "verified"
             and completed["provisional_sha256"] == _sha(result_bytes), "completion record differs")
    _require(type(result) is dict and set(result) == {"identity", "baseline", "row_sha256", "screen_sha256",
             "journal_sha256", "original_state_sha256", "original_weight_sha256", "original_buffer_sha256", "scope"}
             and _same(result["identity"], started["identity"]), "provisional result differs")
    for name in ("original_state_sha256", "original_weight_sha256", "original_buffer_sha256"):
        digest = result[name]
        _require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                 "reported state digest differs")
    raw = _read(output / "screen.f32", 27 * vocabulary * 4)
    events_bytes = _read(output / "events.jsonl", 1_000_000)
    _require(len(raw) == 27 * vocabulary * 4 and _sha(raw) == result["screen_sha256"]
             and _sha(events_bytes) == result["journal_sha256"] and events_bytes.endswith(b"\n"),
             "saved matrix or journal digest differs")
    lines = events_bytes.splitlines(keepends=True)
    _require(len(lines) == 54, "forward journal count differs")
    events = [_decode(line) for line in lines]
    _require(all(line == _canonical(event) for line, event in zip(lines, events)), "noncanonical journal")
    hashes, tops = [], []
    for index, row in enumerate(rows):
        start, done = events[2 * index:2 * index + 2]
        _require(type(start) is dict and set(start) == {"kind", "index", "case_id", "planned_input_ids",
                 "planned_attention_mask", "denormal_probe_before"}, "forward-start structure differs")
        _require(_same(start, {"kind": "forward_start", "index": index, "case_id": row["case_id"],
                 "planned_input_ids": row["token_ids"], "planned_attention_mask": row["attention_mask"],
                 "denormal_probe_before": PROBE}), "forward-start identity differs")
        offset = index * vocabulary * 4
        row_data = raw[offset:offset + vocabulary * 4]
        digest = _sha(row_data)
        expected_done = {"kind": "forward_complete", "index": index, "case_id": row["case_id"],
                         "actual_input_ids": row["token_ids"], "actual_attention_mask": row["attention_mask"],
                         "offset": offset, "bytes": len(row_data), "sha256": digest,
                         "returned_metadata": {"device": "cpu", "dtype": "torch.float32",
                                               "shape": [1, len(row["token_ids"]), vocabulary]},
                         "denormal_probe_after": PROBE}
        _require(_same(done, expected_done), "forward-completion identity or bytes differ")
        hashes.append(digest)
        tops.append(top_details(word[0] for word in struct.iter_unpack("<I", row_data)))
    _require(_same(result["row_sha256"], hashes), "per-row digest list differs")
    dev = [{"offset": i, **tops[i], "supported": tops[i]["unique_top_id"] == true_id} for i in range(3)]
    targets = [{"offset": i, **tops[i], "supported": tops[i]["unique_top_id"] == true_id} for i in range(3, 11)]
    local = [{**item, "offset": 11+i, **tops[11+i],
              "supported": tops[11+i]["unique_top_id"] in item["accepted_ids"]}
             for i, item in enumerate(locality)]
    E, H = sum(row["supported"] for row in targets), sum(row["supported"] for row in local)
    H_new = sum(row["supported"] and row["new"] for row in local)
    reasons = []
    if not all(row["supported"] for row in dev):
        reasons.append("development baseline")
    if E < 4:
        reasons.append("target baseline count")
    if H < 8:
        reasons.append("locality baseline count")
    if H_new < 4:
        reasons.append("new locality baseline count")
    expected = {"status": "ineligible" if reasons else "eligible", "reasons": reasons,
                "E": E, "H": H, "H_new": H_new, "development": dev, "targets": targets,
                "locality": local, "binary64_probe_before": [1, 0x8000000000000001],
                "binary64_probe_after": [1, 0x8000000000000001]}
    _require(_same(result["baseline"], expected), "saved baseline science differs from raw logits")
    return {"status": "saved-records-consistent", "forward_completions": 27, "E": E, "H": H,
            "H_new": H_new, "baseline_status": expected["status"], "reasons": reasons,
            "identity": started["identity"], "screen_sha256": _sha(raw),
            "journal_sha256": _sha(events_bytes), "provisional_sha256": _sha(result_bytes),
            "completed_sha256": _sha(completed_bytes),
            "scope": "saved-byte consistency only; source, public-input custody, execution and cleanup are separate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    case_bytes = _read(args.cases, 200_000)
    _require(_sha(case_bytes) == CASE_SHA256, "fixed case digest differs")
    cases = _decode(case_bytes)
    _require(cases.get("schema") == "france-full-top-fixed-cases-v3", "case schema differs")
    source_rows = cases["rows"]
    rows = [{key: item[key] for key in ("case_id", "token_ids", "attention_mask")} for item in source_rows]
    labels = [{"case_id": item["case_id"],
               "accepted_ids": [answer["token_id"] for answer in item["continuations"]]}
              for item in source_rows[11:]]
    report = audit_rows(args.run, rows, labels, 6342, 3576, 50257)
    report["cases_sha256"] = CASE_SHA256
    payload = _canonical(report)
    with open(args.output, "xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("short audit write")
        stream.flush()
        os.fsync(stream.fileno())
    print("sha256=" + _sha(payload))


if __name__ == "__main__":
    main()
