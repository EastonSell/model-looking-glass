"""Record public cue rows observed in two closed journals, without model use."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

MANIFEST = "6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a"
JOURNALS = {
    "original_incomplete": "ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05",
    "continuation": "67c022d36d7427840334390602706e286aab60168c50c42dad72b7e711b2c87f",
}
EXPECTED = {
    "original_incomplete": {"exposed-calibration-prior", "exposed-calibration-new", "fresh-01-prior",
                            "fresh-01-new", "fresh-02-prior"},
    "continuation": {"fresh-02-prior", "fresh-02-new", "fresh-03-prior", "fresh-03-new"},
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("original", type=Path)
    p.add_argument("continuation", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--published-result-commit", required=True)
    args = p.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.published_result_commit):
        raise ValueError("a published result commit is required")
    paths = {"original_incomplete": args.original, "continuation": args.continuation}
    if sha(args.manifest) != MANIFEST or {name: sha(path) for name, path in paths.items()} != JOURNALS:
        raise ValueError("fixed manifest or journal digest differs")
    manifest = json.loads(args.manifest.read_bytes())
    cases = {case["id"]: case for case in manifest["cases"]}
    if len(cases) != 8:
        raise ValueError("fixed manifest case count differs")
    seen = {}
    for source, path in paths.items():
        with gzip.open(path, "rb") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("kind") != "clean":
                    continue
                case_id, cue = event.get("case_id"), event.get("cue")
                if case_id not in EXPECTED[source] or cue not in cases[case_id]["rows"]:
                    raise ValueError("unexpected clean case or cue")
                key = (case_id, cue)
                if source in seen.get(key, []):
                    raise ValueError("duplicate clean witness")
                row = cases[case_id]["rows"][cue]
                if event.get("input_ids") != row["input_ids"]:
                    raise ValueError("saved input IDs differ from fixed manifest")
                seen.setdefault(key, []).append(source)
    rows = []
    for case in manifest["cases"]:
        cue_set = manifest["cue_sets"][case["cue_set"]]
        for cue in cue_set["negative"] + cue_set["affirmative"]:
            key = (case["id"], cue)
            if not seen.get(key):
                raise ValueError("manifest variant lacks a saved clean witness")
            row = case["rows"][cue]
            rows.append({"case_id": case["id"], "cue": cue, "split": case["split"],
                         "cue_set": case["cue_set"], "status": "observed-saved",
                         "observed_in": seen[key], "text": row["text"],
                         "input_ids": row["input_ids"], "attention_mask": row["attention_mask"]})
    if len(rows) != 32 or sum("original_incomplete" in row["observed_in"] for row in rows) != 20 or \
            sum("continuation" in row["observed_in"] for row in rows) != 16 or \
            sum(len(row["observed_in"]) == 2 for row in rows) != 4:
        raise ValueError("observed row or overlap counts differ")
    if sha(args.manifest) != MANIFEST or {name: sha(path) for name, path in paths.items()} != JOURNALS:
        raise ValueError("input changed during exposure inventory")
    output = {"schema": "bert-cue-dose-public-exposures-v2", "model": "distilbert/distilbert-base-uncased",
              "model_revision": "12040accade4e8a0f71eabdb258fecc2e7e948be", "manifest_sha256": MANIFEST,
              "original_source_commit": "b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4",
              "continuation_source_commit": "cdc9630e24542fed84b0c9556bb7f37c60d722d1",
              "published_continuation_result_commit": args.published_result_commit,
              "original_attempt": "incomplete", "journal_sha256": JOURNALS,
              "original_failure_sha256_commitment": "12b861407be2e4050b7f5e7b792847867f7b7c588cf75857783fc36d1bb8317a",
              "original_exit_sha256": "f5499d680cbd2ea1d2ebdfdb7268ce30598ad8aa7d6af9ad8ef0ff806ed54f34",
              "continuation_result_sha256": "cce115669ca1a957428d000950aa36e04c8087a0a96a87dc78492cdff59662d5",
              "continuation_exit_sha256": "a932efaf2be7b8a8cf72456c73e9bbcb8dd4623e1fa370ef6fe56384dd38f042",
              "record_counts": {"rows": 32, "original_incomplete": 20, "continuation": 16,
                                "observed_in_both": 4},
              "scope": "Input IDs are checked against saved clean witnesses; text, split and all-one attention masks come from the fixed manifest. This is not an execution attestation or a claim of complete prior/pretraining exposure.",
              "rows": rows}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": 32, "original": 20, "continuation": 16, "both": 4,
                      "output_sha256": sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
