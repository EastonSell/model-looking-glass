"""Build the fixed natural-transfer schedule from the public case bundle; no model access."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

CASE_SHA256 = "50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827"
DETERMINER_IDS = {"this": 428, "these": 777}
SPLITS = ("discovery", "evaluation")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def build(source):
    raw = Path(source).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CASE_SHA256, "case-byte hash mismatch")
    data = json.loads(raw)
    rows = data["cases"]
    require(len(rows) == 68, "case count")
    counts = Counter((r["split"], r["role"]) for r in rows)
    expected = {(s, role): n for s in SPLITS for role, n in
                (("target", 24), ("fragment", 6), ("demonstrative", 4))}
    require(counts == expected, "case role or split counts")
    require(len({r["case_id"] for r in rows}) == len(rows), "duplicate case IDs")
    targets = [r for r in rows if r["role"] == "target"]
    by_key = {}
    for r in targets:
        key = (r["split"], r["frame_index"], r["context_index"], r["noun_index"], r["determiner"])
        require(key not in by_key, "duplicate target key")
        require(r["split"] in SPLITS and r["frame_index"] in (0, 1)
                and r["context_index"] in (0, 1, 2) and r["noun_index"] in (0, 1)
                and r["determiner"] in DETERMINER_IDS, "unexpected target coordinate")
        tokens = r["token_ids"]
        di, ni, fi = r["determiner_index"], r["subject_index"], r["final_index"]
        require(isinstance(tokens, list) and all(type(x) is int and x >= 0 for x in tokens), "bad token row")
        require(0 <= di < len(tokens) and 0 <= ni < len(tokens) and di != ni
                and fi == len(tokens) - 1 and tokens[di] == DETERMINER_IDS[r["determiner"]], "bad positions")
        by_key[key] = r
    expected_keys = {(s, f, c, n, d) for s in SPLITS for f in (0, 1)
                     for c in (0, 1, 2) for n in (0, 1) for d in DETERMINER_IDS}
    require(set(by_key) == expected_keys, "incomplete target grid")

    mappings = []
    for split in SPLITS:
        for r in (x for x in targets if x["split"] == split):
            f, c, n, d = r["frame_index"], r["context_index"], r["noun_index"], r["determiner"]
            other_d = "these" if d == "this" else "this"
            q1 = by_key[(split, f, c, 1 - n, d)]
            q2 = by_key[(split, f, c, n, other_d)]
            q4 = by_key[(split, f, c, 1 - n, other_d)]
            di, ni, fi = r["determiner_index"], r["subject_index"], r["final_index"]
            for donor, differences in ((q1, {ni}), (q2, {di}), (q4, {di, ni})):
                require(len(donor["token_ids"]) == len(r["token_ids"]), "unequal token lengths")
                require((donor["determiner_index"], donor["subject_index"], donor["final_index"])
                        == (di, ni, fi), "positions mismatch")
                observed = {i for i, (a, b) in enumerate(zip(r["token_ids"], donor["token_ids"])) if a != b}
                require(observed == differences, "unexpected donor token differences")
            mappings.append({
                "split": split,
                "receiver_case_id": r["case_id"],
                "receiver_token_ids": r["token_ids"],
                "frame_index": f,
                "context_index": c,
                "noun_index": n,
                "determiner": d,
                "determiner_index": di,
                "subject_index": ni,
                "final_index": fi,
                "orientation": 1 if d == "these" else -1,
                "candidates": {
                    "Q1": {"operation": "scaled-same-determiner-other-noun", "donor_case_id": q1["case_id"]},
                    "Q2": {"operation": "exact-same-noun-opposite-determiner", "donor_case_id": q2["case_id"]},
                    "Q3": {"operation": "cyclic-shift-Q2-displacement", "source_donor_case_id": q2["case_id"]},
                    "Q4": {"operation": "scaled-opposite-determiner-other-noun", "donor_case_id": q4["case_id"]},
                },
            })
    require(len(mappings) == 48, "mapping count")
    return {"schema": "natural-invariant-donor-transfer-mapping-v1", "case_sha256": CASE_SHA256,
            "case_revision": "3e998f6535dc41ec4e0bb45463164fe1b46c2611",
            "ordered_action_candidates": ["Q1", "Q2", "Q3", "Q4"],
            "new_forwards_per_phase": 96, "new_forwards_total": 192,
            "model_forward_performed_by_builder": False, "receivers": mappings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    document = build(args.cases)
    content = (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with open(args.output, "xb") as out:
        out.write(content)
    print(json.dumps({"path": args.output, "sha256": hashlib.sha256(content).hexdigest(),
                      "receivers": len(document["receivers"]),
                      "forwards_per_phase": document["new_forwards_per_phase"]}, sort_keys=True))


if __name__ == "__main__":
    main()
