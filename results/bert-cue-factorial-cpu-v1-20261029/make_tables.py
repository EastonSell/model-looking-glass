"""Build descriptive tables from a closed factorial result and pinned vocabulary."""
import csv
import json
import sys
from pathlib import Path


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(result_path, manifest_path, vocab_path, output_path, *, expected_vocab=30522):
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    vocab = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    if len(vocab) != expected_vocab:
        raise ValueError("vocabulary size differs")
    by_id = {row["id"]: row for row in manifest["cases"]}
    if (len(by_id) != 12 or len(result["cases"]) != 12 or
            [row["id"] for row in result["cases"]] != [row["id"] for row in manifest["cases"]]):
        raise ValueError("fixed case identities or order differ")
    sites = ("cue2", "object2", "mask2", "cue5")
    cases, baselines, effects, targets = [], [], [], []
    for row in result["cases"]:
        fixed = by_id[row["id"]]
        common = {key: row[key] for key in ("id", "triple_id", "frame", "cue_set")}
        if any(row[key] != fixed[key] for key in ("triple_id", "frame", "cue_set")):
            raise ValueError("case metadata differs")
        group = manifest["cue_sets"][row["cue_set"]]
        cues = group["negative"] + group["affirmative"]
        summary = dict(common, role=fixed["role"], verb=fixed["verb"], object=fixed["object"],
                       baseline_eligible=row["baseline_eligible"], dose_eligible=row["dose_eligible"],
                       min_baseline_gap=min(gap["gap"] for gap in row["baseline_gaps"]),
                       minimum_distance=row["minimum_distance"], radius=row["radius"],
                       cue_min_margin=row["cue_min_margin"], limiting_targets="|".join(row["limiting_targets"]),
                       specificity_passed=row["specificity_passed"])
        for index, cue in enumerate(cues, 1):
            summary[f"cue_{index}"] = cue
            summary[f"margin_{index}"] = row["specificity"]["cue2"]["targets"][cue]["margin"] if row["dose_eligible"] else ""
            baseline = row["baselines"][cue]
            bases = dict(common, cue=cue, any_minus_some=baseline["any_minus_some"],
                         any_probability=baseline["any_probability"], some_probability=baseline["some_probability"])
            for rank, item in enumerate(baseline["top_five"], 1):
                bases[f"top{rank}_id"] = item["token_id"]
                bases[f"top{rank}_token"] = vocab[item["token_id"]]
                bases[f"top{rank}_logit"] = item["logit"]
            baselines.append(bases)
        cases.append(summary)
        if not row["dose_eligible"]:
            continue
        for site in sites:
            for effect in row["effects"][site]:
                effects.append(dict(common, site=site, **effect))
            for target in cues:
                info = row["specificity"][site]["targets"][target]
                donors = list(info["cross_signed"])
                targets.append(dict(common, site=site, target=target,
                                    cross_donor_1=donors[0], cross_signed_1=info["cross_signed"][donors[0]],
                                    cross_donor_2=donors[1], cross_signed_2=info["cross_signed"][donors[1]],
                                    cross_min=info["cross_min"], same_donor=info["same_donor"],
                                    same_delta=info["same_delta"], same_abs=info["same_abs"],
                                    margin=info["margin"], passed=info["passed"]))
    evaluation = result["evaluation"]
    cells = [{"cell": name, **data} for name, data in evaluation["cells"].items()]
    contrasts = []
    for kind, mapping in (("cue", evaluation["cue_contrasts"]), ("frame", evaluation["frame_contrasts"]),
                          ("interaction", {"all": evaluation["interaction"]})):
        for name, value in mapping.items():
            if value["values"]:
                for item in value["values"]:
                    contrasts.append({"kind": kind, "contrast": name, "status": value["status"],
                                      "eligible": value["eligible"], "mean": value["mean"], **item})
            else:
                contrasts.append({"kind": kind, "contrast": name, "status": value["status"],
                                  "eligible": value["eligible"], "mean": value["mean"], "triple_id": "", "value": ""})
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    common = ["id", "triple_id", "frame", "cue_set"]
    case_fields = common + ["role", "verb", "object", "baseline_eligible", "dose_eligible", "min_baseline_gap",
                            "minimum_distance", "radius", "cue_min_margin", "limiting_targets", "specificity_passed"]
    case_fields += [field for n in range(1, 5) for field in (f"cue_{n}", f"margin_{n}")]
    baseline_fields = common + ["cue", "any_minus_some", "any_probability", "some_probability"]
    for rank in range(1, 6):
        baseline_fields += [f"top{rank}_id", f"top{rank}_token", f"top{rank}_logit"]
    effect_fields = common + ["site", "donor", "target", "delta"]
    target_fields = common + ["site", "target", "cross_donor_1", "cross_signed_1", "cross_donor_2",
                              "cross_signed_2", "cross_min", "same_donor", "same_delta", "same_abs", "margin", "passed"]
    for name, fields, rows in (("cases.csv", case_fields, cases), ("baselines.csv", baseline_fields, baselines),
                               ("effects.csv", effect_fields, effects), ("target_checks.csv", target_fields, targets),
                               ("cells.csv", ["cell", "total", "eligible", "passing", "status"], cells),
                               ("contrasts.csv", ["kind", "contrast", "status", "eligible", "mean", "triple_id", "value"], contrasts)):
        write_csv(out / name, fields, rows)
    print(json.dumps({"cases": len(cases), "baselines": len(baselines), "effects": len(effects),
                      "target_checks": len(targets), "cells": len(cells), "contrast_rows": len(contrasts)}, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: build_cue_factorial_tables.py RESULT MANIFEST VOCAB OUTPUT_DIR")
    main(*sys.argv[1:])
