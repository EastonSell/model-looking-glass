"""Render the fixed signed cue-context saved result without a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MANIFEST_HASH = "843649c3f105e26185d96f31d769b0585d219bb7a45ba445cf2c2548598927df"
RESULT_HASH = "0a788dfb0655ac8da55d94b1154fa240860850bf9464928b391d8cfa3327f62a"
SOURCE = "91f9960e419cf20df9df156a6ebc1980d1b6a864"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest, result = read_fixed(args.manifest, MANIFEST_HASH), read_fixed(args.result, RESULT_HASH)
    if result.get("source_commit") != SOURCE:
        raise ValueError("recorded source differs")
    args.output.mkdir(parents=False, exist_ok=False)
    sources = {pair["id"]: pair for pair in manifest["pairs"]}
    pairs, baselines, effects, recipients, tv = [], [], [], [], []
    for pair in result["pairs"]:
        contexts = sources[pair["id"]]["contexts"]
        names = ["/".join((row["role"], row["verb"], row["object"])) for row in contexts]
        base = {"pair": pair["id"], "split": pair["split"]}
        summary = pair["signed"]
        secondary = summary["secondary"]
        margins = [row["primary_margin"] for row in summary["recipients"] if row["primary_margin"] is not None]
        pairs.append({**base, "context_a": names[0], "context_b": names[1],
                      "baseline_eligible": pair["baseline_eligible"], "dose_eligible": pair["dose_eligible"],
                      "q": pair["q"], "minimum_distance": pair["minimum_distance"],
                      "primary_eligible": summary["eligible"], "minimum_D": min(margins) if margins else None,
                      "primary_raw_pass": summary["primary_raw_pass"], "primary_status": summary["primary_status"],
                      "sensitivity_raw_pass": summary["sensitivity_raw_pass"],
                      "sensitivity_status": summary["sensitivity_status"],
                      **{"secondary_" + key: value for key, value in secondary.items()}})
        for row in pair["baselines"]:
            baselines.append({**base, **{key: value for key, value in row.items() if key != "top_five"},
                              "context_name": names[row["context"]],
                              "top_five": json.dumps(row["top_five"], separators=(",", ":"))})
        for row in pair["effects"]:
            effects.append({**base, **row, "target_context_name": names[row["target_context"]],
                            "donor_context_name": names[row["donor_context"]]})
        for row in summary["recipients"]:
            serial = {key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                      if isinstance(value, (dict, list)) else value for key, value in row.items()}
            recipients.append({**base, **serial, "target_context_name": names[row["target_context"]]})
        for row in pair["tv_summary"]:
            tv.append({**base, **row})
    pair_columns = ["pair", "split", "context_a", "context_b", "baseline_eligible", "dose_eligible", "q",
                    "minimum_distance", "primary_eligible", "minimum_D", "primary_raw_pass", "primary_status",
                    "sensitivity_raw_pass", "sensitivity_status", "secondary_status", "secondary_qualified_recipients",
                    "secondary_raw_toward_count", "secondary_raw_exceed_count", "secondary_toward_count",
                    "secondary_exceed_count", "secondary_mean_donorward", "secondary_mean_excess"]
    base_columns = ["pair", "split", "context", "context_name", "cue", "any_minus_some", "any_logit",
                    "some_logit", "any_probability", "some_probability", "any_log_probability",
                    "some_log_probability", "target_pair_probability", "top_five"]
    effect_columns = ["pair", "split", "site", "target_context", "target_context_name", "target_cue",
                      "donor_context", "donor_context_name", "donor_cue", "delta", "full_tv", "excluded_tv",
                      "clean_pair_probability", "patched_pair_probability"]
    recipient_columns = ["pair", "split", "target_context", "target_context_name", "target_cue", "class_direction",
                         "same_cue_gap", "gap_qualified", "same_cue_delta", "same_class_cue", "same_class_clean_gap",
                         "same_class_delta", "opposite_clean_gaps", "opposite_delta", "opposite_class_directed",
                         "limiting_opposite", "limiting_control", "primary_margin", "secondary_donorward", "secondary_excess"]
    table(args.output / "pairs.csv", pair_columns, pairs)
    table(args.output / "baselines.csv", base_columns, baselines)
    table(args.output / "effects.csv", effect_columns, effects)
    table(args.output / "recipients.csv", recipient_columns, recipients)
    table(args.output / "tv_summary.csv", ["pair", "split", "site", "metric", "same_cue_mean", "same_class_mean"], tv)
    print(json.dumps({"pairs": len(pairs), "baselines": len(baselines), "effects": len(effects),
                      "recipients": len(recipients), "tv_summaries": len(tv)}, sort_keys=True))


if __name__ == "__main__":
    main()
