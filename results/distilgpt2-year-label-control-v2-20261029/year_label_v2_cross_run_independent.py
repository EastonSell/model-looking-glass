"""Compare saved v1 and v2 baseline bytes for the 36 reused public cases.

This validates the named segment bytes and v2's recorded comparison; it does
not rerun a model, establish execution provenance, or gate the v2 outcome.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"Digest mismatch: {path.name}")
    return data


def records(path, expected):
    raw = read_pinned(path / "attempt.jsonl", expected)
    out = {}
    for line in raw.splitlines():
        e = json.loads(line)
        if e.get("event") == "complete":
            k = (e["case_id"], e["action"])
            if k in out:
                raise ValueError("Duplicate complete event")
            out[k] = e
    return out


def segment(root, record, name):
    m = record["raw"][name]
    if type(m["offset"]) is not int or m["offset"] < 0 or type(m["bytes"]) is not int or m["bytes"] <= 0:
        raise ValueError("Invalid segment extent")
    with (root / (record["case_id"] + ".f32")).open("rb") as f:
        f.seek(m["offset"])
        b = f.read(m["bytes"])
    if len(b) != m["bytes"] or digest(b) != m["sha256"]:
        raise ValueError("Invalid segment bytes")
    return b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1-run", required=True, type=Path)
    p.add_argument("--v2-run", required=True, type=Path)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--v2-journal-sha256", required=True)
    p.add_argument("--v2-result-sha256", required=True)
    args = p.parse_args()
    case_sha = "07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08"
    v1_journal_sha = "a2f5ba4f980bde155bf4eb5729f154f7422956eb9c0920421f13327038e057f5"
    v1_manifest_sha = "d43d181e1c7046c8e97e6310bd3edd38fe85089e0e69347cd2ddb2728d29d47a"
    cases = json.loads(read_pinned(args.cases, case_sha))["cases"]
    if len(cases) != 48:
        raise ValueError("Wrong case count")
    rows = cases[12:]
    v1_manifest = json.loads(read_pinned(args.v1_run / "manifest.json", v1_manifest_sha))
    if v1_manifest["cases"] != cases:
        raise ValueError("V1 case binding differs")
    v1, v2 = records(args.v1_run, v1_journal_sha), records(args.v2_run, args.v2_journal_sha256)
    v2_result = json.loads(read_pinned(args.v2_run / "result.json", args.v2_result_sha256))
    comparisons, projection = [], []
    for row in rows:
        case = row["case_id"]
        same = {}
        for action, name in (("clean", "logits"), ("noop", "logits"),
                             ("noop", "block_before"), ("noop", "block_after")):
            left, right = (segment(root, data[case, action], name)
                           for root, data in ((args.v1_run, v1), (args.v2_run, v2)))
            same[action + "_" + name] = left == right
        comparisons.append({"case_id": case, **same})
        projection.append({"case_id": case, "clean_logits_match_v1": same["clean_logits"],
                           "noop_block_output_matches_v1": same["noop_block_before"]})
    if v2_result["cross_run_baseline_comparison"] != projection:
        raise ValueError("Recorded v2 comparison does not match raw bytes")
    keys = ("clean_logits", "noop_logits", "noop_block_before", "noop_block_after")
    print(json.dumps({"status": "saved_byte_comparison_complete", "case_count": len(rows),
                      "matches": {key: sum(r[key] for r in comparisons) for key in keys},
                      "rows": comparisons, "v1_journal_sha256": v1_journal_sha,
                      "v2_journal_sha256": args.v2_journal_sha256,
                      "v2_result_sha256": args.v2_result_sha256,
                      "scope": "Saved-byte comparison only; differences do not gate v2, and execution or provenance is not attested."},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
