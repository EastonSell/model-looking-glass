"""Render compact tables from the fixed saved cue-context result; no model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MANIFEST_HASH = "4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7"
RESULT_HASH = "4f3eca14fffea1df6d8504d79017c2980ba03f8f31fd3fd7b7d1812f3697cbc8"


def read_fixed(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("fixed input hash differs")
    return json.loads(data)


def table(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("result", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    manifest, result = read_fixed(args.manifest, MANIFEST_HASH), read_fixed(args.result, RESULT_HASH)
    if result.get("source_commit") != "40752bde00f11b69832066c2d16ce52c49f96810":
        raise ValueError("recorded execution source differs")
    args.output.mkdir(parents=False, exist_ok=False)
    sources = {p["id"]: p for p in manifest["pairs"]}
    pairs, baselines, effects, recipients, tv = [], [], [], [], []
    for pair in result["pairs"]:
        case = sources[pair["id"]]
        base = {"pair": pair["id"], "split": pair["split"]}
        context_names = ["/".join((c["role"], c["verb"], c["object"])) for c in case["contexts"]]
        comp = pair["comparison"]
        pairs.append({**base, "context_a": context_names[0], "context_b": context_names[1],
                      "baseline_eligible": pair["baseline_eligible"], "dose_eligible": pair["dose_eligible"],
                      "q": pair["q"], "minimum_distance": pair["minimum_distance"],
                      "same_cue_mean": comp["same_cue_mean"], "same_class_mean": comp["same_class_mean"],
                      "half_difference": comp["half_difference"], "status": comp["status"]})
        for row in pair["baselines"]:
            baselines.append({**base, **{k: v for k, v in row.items() if k != "top_five"},
                              "context_name": context_names[row["context"]],
                              "top_five": json.dumps(row["top_five"], separators=(",", ":"))})
        for row in pair["effects"]:
            effects.append({**base, **row, "target_context_name": context_names[row["target_context"]],
                            "donor_context_name": context_names[row["donor_context"]]})
        for row in comp["recipients"]:
            recipients.append({**base, **row, "target_context_name": context_names[row["target_context"]]})
        for row in pair["tv_summary"]:
            tv.append({**base, **row})
    table(args.output / "pairs.csv", ["pair", "split", "context_a", "context_b", "baseline_eligible",
          "dose_eligible", "q", "minimum_distance", "same_cue_mean", "same_class_mean", "half_difference", "status"], pairs)
    table(args.output / "baselines.csv", ["pair", "split", "context", "context_name", "cue", "any_minus_some",
          "any_logit", "some_logit", "any_probability", "some_probability", "any_log_probability",
          "some_log_probability", "target_pair_probability", "top_five"], baselines)
    table(args.output / "effects.csv", ["pair", "split", "site", "target_context", "target_context_name", "target_cue",
          "donor_context", "donor_context_name", "donor_cue", "delta", "full_tv", "excluded_tv",
          "clean_pair_probability", "patched_pair_probability"], effects)
    table(args.output / "recipients.csv", ["pair", "split", "target_context", "target_context_name", "target_cue",
          "same_cue_delta", "same_class_delta"], recipients)
    table(args.output / "tv_summary.csv", ["pair", "split", "site", "metric", "same_cue_mean", "same_class_mean"], tv)
    print(json.dumps({"pairs": len(pairs), "baselines": len(baselines), "effects": len(effects),
                      "recipients": len(recipients), "tv_summaries": len(tv)}, sort_keys=True))


if __name__ == "__main__":
    main()
