"""Compare the fixed suffix cases against explicitly named public inputs.

This tokenizer-ID and source-text check does not load model weights or query
private data.  Pending inputs are supplied as a separately SHA-bound file.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess


CASE_SHA = "1da4f8e4773009fedc750480c21597fc60ecaa0c990dcff0af15a5c7101c3dca"
PREVIOUS_REVIEW_SHA = "b9cc99838fa98c99d1640ff09482ba4e6604d4390e4448d5aa19b7570690d695"
PENDING_SHA = "4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7"
EXTRA = [
    ("experiments/cross_model_agreement_cases_v1.json", "8e2cb4f815a12ffb8149f4afe6d3d073df9c170f", "b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503", 32),
    ("experiments/bert_color_preference_public_cases_v2.json", "f79dc502ff85b467b0682f4e8d168fbeb776cadf", "f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10", 37),
    ("experiments/bert_color_edit_public_exposures_v2.json", "8c03137ccee6664e4ace2dd324f7b0108ab88a28", "2129d9642548b69928e9e3f4bcb8d044e0383b9671a3f01fe9bb0754ef3a7824", 37),
]
TERMS = ("scout", "scouts", "monk", "monks", "poet", "poets", "guide", "guides",
         "visitor", "visitors", "tourist", "tourists")
FRAMES = ("At the station", "Beyond the gate")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(row):
    ids, mask = row["input_ids"], row["attention_mask"]
    if (type(ids) is not list or not 2 <= len(ids) <= 64 or
            any(type(x) is not int or not 0 <= x < 30522 for x in ids) or
            ids[0] != 101 or ids[-1] != 102 or ids.count(103) != 1 or
            type(mask) is not list or mask != [1] * len(ids)):
        raise ValueError("unexpected BERT input or attention mask")
    return tuple(ids)


def extract(name, document):
    if name == "bert_negation_public_cases_v1.json":
        position = document["indices"]["cue"]
        result = []
        for case in document["cases"]:
            for cue in ("not", "never", "just", "also"):
                ids = list(case["just_ids"])
                if ids[position] != document["cues"]["just"]:
                    raise ValueError("negation cue mismatch")
                ids[position] = document["cues"][cue]
                result.append(checked({"input_ids": ids, "attention_mask": document["attention_mask"]}))
        return result
    if name == "bert_cue_context_transfer_cases_v1.json":
        return [checked(row) for pair in document["pairs"]
                for context in pair["contexts"] for row in context["rows"].values()]
    if name.startswith("bert_cue_") and name.endswith("_cases_v1.json"):
        return [checked(row) for case in document["cases"] for row in case["rows"].values()]
    if name in ("bert_color_edit_public_cases_v1.json", "bert_color_preference_public_cases_v2.json"):
        return [checked(row) for row in document["cases"]]
    if name == "bert_color_edit_public_exposures_v2.json":
        return [checked(row) for row in document["records"]]
    if name == "cross_model_agreement_cases_v1.json":
        return [checked(case["bert"]) for case in document["cases"]]
    return [checked(row) for row in document["rows"]]


def collision(a, b):
    # Check full and one-SEP-removed sequences in all four combinations.
    return any(x[:len(y)] == y or y[:len(x)] == x
               for x in (a, a[:-1]) for y in (b, b[:-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--pending-context", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raw = args.cases.read_bytes()
    if sha(raw) != CASE_SHA:
        raise ValueError("case manifest hash mismatch")
    document = json.loads(raw)
    new = [(case["case_id"], variant, checked(case["variants"][variant]))
           for case in document["cases"] for variant in ("terminal", "continued")]
    if len(new) != 64 or len({(x[0], x[1]) for x in new}) != 64:
        raise ValueError("unexpected proposed cases")
    expected, unexpected = [], []
    for a, b in itertools.combinations(new, 2):
        if collision(a[2], b[2]):
            (expected if a[0] == b[0] and a[1] != b[1] else unexpected).append([a[:2], b[:2]])
    if len(expected) != 32 or unexpected:
        raise ValueError("unexpected within-proposal prefix relationship")

    prior = (args.repo / "audits/cross_model_agreement_v1/bert-public-exposure-review.json").read_bytes()
    if sha(prior) != PREVIOUS_REVIEW_SHA:
        raise ValueError("previous inventory pin differs")
    sources = [(s["path"], s["commit"], s["sha256"], s["rows"])
               for s in json.loads(prior)["sources"]] + EXTRA
    pending = args.pending_context.read_bytes()
    if sha(pending) != PENDING_SHA:
        raise ValueError("pending context pin differs")
    sources.append(("bert_cue_context_transfer_cases_v1.json", "pending-local", PENDING_SHA, 24))
    reports, hits = [], []
    for path, commit, wanted, expected_rows in sources:
        source = pending if commit == "pending-local" else subprocess.check_output(
            ["git", "-C", str(args.repo), "show", commit + ":" + path])
        if sha(source) != wanted:
            raise ValueError("source digest mismatch: " + path)
        rows = extract(Path(path).name, json.loads(source))
        if len(rows) != expected_rows:
            raise ValueError("source row count mismatch: " + path)
        collisions = [(case, variant, j) for case, variant, ids in new
                      for j, old in enumerate(rows) if collision(ids, old)]
        hits.extend((path, *item) for item in collisions)
        text = source.decode("utf-8")
        terms = [word for word in TERMS if re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE)]
        frames = [frame for frame in FRAMES if frame.casefold() in text.casefold()]
        reports.append({"path": path, "commit": commit, "sha256": wanted,
                        "rows": len(rows), "collisions": len(collisions),
                        "matched_whole_noun_forms": terms, "matched_surface_frames": frames})
    report = {"schema": "bert-agreement-suffix-public-comparison-v1", "case_sha256": CASE_SHA,
              "new_rows": len(new), "within_pairs_compared": len(new) * (len(new) - 1) // 2,
              "expected_within_pair_relations": len(expected), "unexpected_within_relations": unexpected,
              "source_count": len(reports), "source_row_references": sum(x["rows"] for x in reports),
              "cross_bundle_pairs_compared": len(new) * sum(x["rows"] for x in reports),
              "collisions": hits, "sources": reports,
              "scope": "Only listed public or fixed pending inputs and exact surface strings; no private, unknown, or pretraining-disjointness guarantee."}
    encoded = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with args.output.open("xb") as out:
        out.write(encoded)
    print(f"report_sha256={sha(encoded)} rows={len(new)} sources={len(reports)} collisions={len(hits)}")
    if hits:
        raise SystemExit("public input collision")


if __name__ == "__main__":
    main()
