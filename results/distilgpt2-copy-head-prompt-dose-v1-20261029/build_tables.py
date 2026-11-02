"""Produce compact tables from the saved matched-dose result and public cases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result_raw, case_raw, audit_raw = (p.read_bytes() for p in (args.result, args.cases, args.audit))
    assert digest(result_raw) == "229a030bcb6c63549cc5a8229844cbd350edf5d6493ca4ed47898b0d831299b4"
    assert digest(case_raw) == "b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"
    assert digest(audit_raw) == "571ee3136cccc5e2581fdffa18d1db08161ff1daf8c95bf78bd752788236f021"
    result, cases, audit = json.loads(result_raw), json.loads(case_raw)["cases"], json.loads(audit_raw)
    assert audit["status"] == "complete_saved_byte_pass" and audit["result_sha256"] == digest(result_raw)
    assert result["decision"] == audit["decision"] and result["cases"] == audit["cases"]
    assert result["families"] == audit["families"]
    source = {row["case_id"]: row for row in cases}
    records = {(row["case_id"], row["action"]): row for row in result["records"]}
    by_case = []
    for row in result["cases"]:
        c = source[row["case_id"]]
        clean = records[(c["case_id"], "clean")]["score"]
        for head in (2, 10, 7):
            arm = "h" + str(head)
            r = records[(c["case_id"], arm)]
            p = row["plan"]["heads"][str(head)]
            a = r["comparison"]
            by_case.append({"case_id": c["case_id"], "family": c["group"], "variant": c["variant"],
                            "prompt": c["text"], "target_token_id": c["target_id"], "arm": arm,
                            "common_q": row["q"], "source_l2": p["source_l2"],
                            "removed_fraction": p["fraction"], "realized_l2": p["achieved"],
                            "projection_output_delta_l2": r["projection_output_delta_l2"],
                            "baseline_target_rank": clean["target_rank"],
                            "baseline_unique_top_id": clean["top_unique_id"],
                            "edited_target_rank": r["score"]["target_rank"],
                            "edited_unique_top_id": r["score"]["top_unique_id"],
                            "edited_top_tie_count": r["score"]["top_tie_count"],
                            "top_change": a["top_change"],
                            "target_log_probability_delta": a["target_log_probability_delta"],
                            "target_logit_delta": a["target_logit_delta"],
                            "kl_clean_to_arm": a["kl_clean_to_arm"], "total_variation": a["total_variation"]})
    by_family = []
    for i, f in enumerate(result["families"], 1):
        by_family.append({"family": i, "status": f["status"], "D_h2": f["D"]["h2"],
                          "D_h10": f["D"]["h10"], "D_h7": f["D"]["h7"],
                          "contrast_h10_minus_h2": f["contrast"],
                          "dose_normalized_contrast": f["dose_normalized_contrast"],
                          "target_logit_D_h2": f["target_logit_D"]["h2"],
                          "target_logit_D_h10": f["target_logit_D"]["h10"],
                          "target_logit_D_h7": f["target_logit_D"]["h7"],
                          "target_logit_contrast": f["target_logit_contrast"]})
    assert len(by_case) == 96 and len(by_family) == 8
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "prompt-effects.csv", by_case)
    write_csv(args.out / "family-effects.csv", by_family)
    print(json.dumps({"case_rows": len(by_case), "family_rows": len(by_family),
                      "case_csv_sha256": digest((args.out / "prompt-effects.csv").read_bytes()),
                      "family_csv_sha256": digest((args.out / "family-effects.csv").read_bytes())}))


if __name__ == "__main__":
    main()
