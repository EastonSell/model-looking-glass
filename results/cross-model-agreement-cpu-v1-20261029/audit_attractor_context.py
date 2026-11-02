"""Post-hoc, read-only public evidence audit; no model or tokenizer execution."""
import hashlib
import json
import statistics
from pathlib import Path
import sys

import numpy as np


if len(sys.argv) != 4:
    raise SystemExit('usage: audit_attractor_context.py WORK_ROOT CASE_MANIFEST OUTPUT')
ROOT = Path(sys.argv[1]).resolve()
MANIFEST = Path(sys.argv[2]).resolve()
OUTPUT = Path(sys.argv[3]).resolve()
EXPECTED_CASE_SHA = "b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values):
    values = list(values)
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def audit(key, manifest):
    directory = ROOT / f"cross-model-{key}-cpu-v1-20261029-01"
    receipt = read_json(ROOT / f"cross-model-{key}-controller-v1-20261029-01" / "exit-observation.json")
    complete_path = directory / "complete.json"
    complete = read_json(complete_path)
    assert receipt["status"] == "complete" and receipt["exit_code"] == 0
    assert sha(complete_path) == receipt["complete_sha256"]
    for field, filename in (("events_sha256", "events.jsonl"), ("logits_sha256", "logits.f32"),
                            ("witness_sha256", "witness.f32"), ("metadata_sha256", "metadata.json")):
        assert sha(directory / filename) == complete[field]
    assert read_json(directory / "metadata.json")["case_sha256"] == EXPECTED_CASE_SHA
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 192
    vocab = {"gpt": 50257, "bert": 30522}[key]
    assert (directory / "logits.f32").stat().st_size == 192 * vocab * 4
    assert (directory / "witness.f32").stat().st_size == 192 * 2 * 768 * 4
    vectors = np.memmap(directory / "witness.f32", dtype="<f4", mode="r", shape=(192, 2, 768))
    logits = np.memmap(directory / "logits.f32", dtype="<f4", mode="r", shape=(192, vocab))
    clean = {}
    for index, event in enumerate(events):
        assert event["event_index"] == index
        assert event["witness_offset"] == index * 2 * 768 * 4
        assert event["logits_offset"] == index * vocab * 4
        if event["arm"] == "clean":
            assert event["hook_count"] == 1 and event["returned_status"] == "returned"
            assert event["case_id"] not in clean
            clean[event["case_id"]] = index
    assert len(clean) == 32
    cases = manifest["cases"]
    lookup = {(c["frame"], c["attractor_number"], c["lemma"], c["number"]): c for c in cases}
    assert len(lookup) == 32
    target = manifest["targets"][key]
    pairs = []
    for frame in (0, 1):
        for lemma in range(4):
            for number in (0, 1):
                one = lookup[(frame, 0, lemma, number)]
                many = lookup[(frame, 1, lemma, number)]
                one_row, many_row = one[key], many[key]
                one_ids, many_ids = one_row["input_ids"], many_row["input_ids"]
                assert len(one_ids) == len(many_ids)
                assert one_row["attention_mask"] == many_row["attention_mask"]
                assert one_row["subject_position"] == many_row["subject_position"]
                assert one_row["output_position"] == many_row["output_position"]
                differences = [i for i, (x, y) in enumerate(zip(one_ids, many_ids)) if x != y]
                assert len(differences) == 1 and differences[0] > one_row["subject_position"]
                a, b = clean[one["case_id"]], clean[many["case_id"]]
                va, vb = vectors[a, 0], vectors[b, 0]
                assert np.isfinite(va).all() and np.isfinite(vb).all()
                assert np.isfinite(logits[a]).all() and np.isfinite(logits[b]).all()
                ma = float(logits[a, target["is"]]) - float(logits[a, target["are"]])
                mb = float(logits[b, target["is"]]) - float(logits[b, target["are"]])
                assert ma != 0.0 and mb != 0.0
                pairs.append({
                    "singular_attractor_case": one["case_id"],
                    "plural_attractor_case": many["case_id"],
                    "frame": frame, "lemma": lemma, "subject_number": number,
                    "attractor_token_position": differences[0],
                    "subject_vector_byte_identical": va.tobytes() == vb.tobytes(),
                    "subject_vector_l2": float(np.linalg.norm(va.astype(np.float64) - vb.astype(np.float64))),
                    "singular_attractor_is_minus_are": ma,
                    "plural_attractor_is_minus_are": mb,
                    "margin_plural_minus_singular_attractor": mb - ma,
                    "binary_sign_changes": (ma > 0.0) != (mb > 0.0),
                })
    return {
        "scope": "post-hoc matched public input-token pairs; descriptive and nonmechanistic",
        "sources": {"complete_sha256": sha(complete_path), "events_sha256": complete["events_sha256"],
                    "witness_sha256": complete["witness_sha256"], "logits_sha256": complete["logits_sha256"]},
        "pairs": len(pairs),
        "identical_subject_vectors": sum(p["subject_vector_byte_identical"] for p in pairs),
        "subject_vector_l2": summarize(p["subject_vector_l2"] for p in pairs),
        "margin_plural_minus_singular_attractor": summarize(p["margin_plural_minus_singular_attractor"] for p in pairs),
        "negative_margin_changes": sum(p["margin_plural_minus_singular_attractor"] < 0 for p in pairs),
        "zero_margin_changes": sum(p["margin_plural_minus_singular_attractor"] == 0 for p in pairs),
        "positive_margin_changes": sum(p["margin_plural_minus_singular_attractor"] > 0 for p in pairs),
        "binary_sign_changes": sum(p["binary_sign_changes"] for p in pairs),
        "pair_results": pairs,
    }


def main():
    assert sha(MANIFEST) == EXPECTED_CASE_SHA
    manifest = read_json(MANIFEST)
    report = {"schema": "cross-model-attractor-context-posthoc-audit-v1",
              "source_sha256": sha(Path(__file__)), "case_manifest_sha256": sha(MANIFEST),
              "gpt": audit("gpt", manifest), "bert": audit("bert", manifest)}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("source_sha256", sha(Path(__file__)))
    print("report_sha256", sha(OUTPUT))
    for key in ("gpt", "bert"):
        print(key, json.dumps({k: v for k, v in report[key].items() if k != "pair_results"}, sort_keys=True))


if __name__ == "__main__":
    main()
