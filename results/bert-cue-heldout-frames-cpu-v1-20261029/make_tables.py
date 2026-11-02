"""Build descriptive CSV tables from a closed held-out-frame result and vocabulary."""
import csv
import json
import sys
from pathlib import Path


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(result_path, manifest_path, vocab_path, output_path):
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    vocab = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    if len(vocab) != 30522:
        raise ValueError("expected the pinned 30522-token vocabulary")
    by_id = {row["id"]: row for row in manifest["cases"]}
    cues = ("not", "hardly", "really", "certainly")
    sites = ("cue2", "object2", "mask2", "cue5")
    cases, baselines, effects, targets = [], [], [], []
    for row in result["cases"]:
        fixed = by_id[row["id"]]
        common = {"case_id": row["id"], "split": row["split"], "frame": row["frame"]}
        summary = dict(common, role=fixed["role"], verb=fixed["verb"], object=fixed["object"],
                       baseline_eligible=row["baseline_eligible"], dose_eligible=row["dose_eligible"],
                       min_baseline_gap=min(gap["gap"] for gap in row["baseline_gaps"]),
                       radius=row["radius"], specificity_passed=row["specificity_passed"])
        for cue in cues:
            summary["margin_" + cue] = row["specificity"]["cue2"]["targets"][cue]["margin"] if row["dose_eligible"] else ""
            baseline = row["baselines"][cue]
            top = baseline["top_five"]
            bases = dict(common, cue=cue, any_minus_some=baseline["any_minus_some"],
                         any_probability=baseline["any_probability"], some_probability=baseline["some_probability"])
            for rank, item in enumerate(top, 1):
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
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    common = ["case_id", "split", "frame"]
    case_fields = common + ["role", "verb", "object", "baseline_eligible", "dose_eligible",
                            "min_baseline_gap", "radius", "specificity_passed"] + ["margin_" + cue for cue in cues]
    baseline_fields = common + ["cue", "any_minus_some", "any_probability", "some_probability"]
    for rank in range(1, 6):
        baseline_fields += [f"top{rank}_id", f"top{rank}_token", f"top{rank}_logit"]
    effect_fields = common + ["site", "donor", "target", "delta"]
    target_fields = common + ["site", "target", "cross_donor_1", "cross_signed_1", "cross_donor_2",
                              "cross_signed_2", "cross_min", "same_donor", "same_delta", "same_abs",
                              "margin", "passed"]
    for name, fields, rows in (("cases.csv", case_fields, cases), ("baselines.csv", baseline_fields, baselines),
                               ("effects.csv", effect_fields, effects), ("target_checks.csv", target_fields, targets)):
        write_csv(out / name, fields, rows)
    print(json.dumps({"cases": len(cases), "baselines": len(baselines), "effects": len(effects),
                      "target_checks": len(targets), "evaluation": result["evaluation"]}, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: build_cue_heldout_frames_tables.py RESULT MANIFEST VOCAB OUTPUT_DIR")
    main(*sys.argv[1:])
