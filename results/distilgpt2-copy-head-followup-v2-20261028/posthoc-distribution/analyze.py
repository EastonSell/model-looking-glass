"""Post-hoc distributional analysis of saved DistilGPT2 copying-head logits."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_logits(directory, entry):
    record = entry["raw"]["logits"]
    with (directory / (entry["case_id"] + ".f32")).open("rb") as source:
        source.seek(record["offset"])
        payload = source.read(record["bytes"])
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("Saved logit byte count or hash mismatch")
    values = np.frombuffer(payload, dtype="<f4").astype(np.float64)
    if values.shape != (50257,) or not np.isfinite(values).all():
        raise ValueError("Unexpected saved logit vector")
    return values


def log_probabilities(logits):
    shift = logits - logits.max()
    return shift - math.log(float(np.exp(shift).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.result_directory / "run"
    data = json.loads((raw / "result.json").read_text(encoding="utf-8"))
    indexed = {(row["case_id"], row["action"]): row for row in data["cases"]}
    arms = ("h10_final", "h2_final", "h7_final", "h3_final", "h1_final", "h10_target")
    rows = []
    for group in range(1, 9):
        for variant in range(4):
            case_id = f"copy-followup-{group:02}-{variant}"
            clean = indexed[(case_id, "clean")]
            logp = log_probabilities(read_logits(raw, clean))
            p = np.exp(logp)
            for arm in arms:
                entry = indexed[(case_id, arm)]
                logq = log_probabilities(read_logits(raw, entry))
                q = np.exp(logq)
                rows.append({
                    "case_id": case_id, "group": group, "variant": variant,
                    "arm": arm,
                    "tv": float(np.abs(p - q).sum() / 2),
                    "kl_clean_to_ablated": float((p * (logp - logq)).sum()),
                    "target_log_probability_delta": float(logq[entry["target_id"]] - logp[entry["target_id"]]),
                    "target_probability_delta": float(q[entry["target_id"]] - p[entry["target_id"]]),
                    "top_token_changed": int(np.argmax(p)) != int(np.argmax(q)),
                    "pre_head_l2": entry["intervention"]["pre_head_l2"],
                })
    aggregates = {}
    for arm in arms:
        aggregates[arm] = {}
        for label, test in (("repeated", lambda r: r["variant"] == 0),
                            ("shuffled", lambda r: r["variant"] != 0)):
            selected = [r for r in rows if r["arm"] == arm and test(r)]
            aggregates[arm][label] = {
                "count": len(selected),
                "mean_tv": math.fsum(r["tv"] for r in selected) / len(selected),
                "median_tv": float(np.median([r["tv"] for r in selected])),
                "mean_kl_clean_to_ablated": math.fsum(r["kl_clean_to_ablated"] for r in selected) / len(selected),
                "mean_target_probability_delta": math.fsum(r["target_probability_delta"] for r in selected) / len(selected),
                "top_token_changed_count": sum(r["top_token_changed"] for r in selected),
            }
    output = {"analysis": "posthoc-full-vocabulary-copy-head-v1", "numpy": np.__version__,
              "source_result_sha256": hashlib.sha256((raw / "result.json").read_bytes()).hexdigest(),
              "arms": aggregates, "rows": rows}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
