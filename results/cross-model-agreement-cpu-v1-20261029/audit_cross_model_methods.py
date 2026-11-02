"""Read-only audit of public, closed cross-model agreement evidence; no model execution."""
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys

import numpy as np


if len(sys.argv) != 4:
    raise SystemExit('usage: audit_cross_model_methods.py WORK_ROOT CASE_MANIFEST OUTPUT')
ROOT = Path(sys.argv[1]).resolve()
MANIFEST = Path(sys.argv[2]).resolve()
OUTPUT = Path(sys.argv[3]).resolve()
EXPECTED_CASE_SHA = "b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503"
EXPECTED_SOURCE = "763402e8311e2402c49eb865c1812c5a8b5380a2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary(values):
    values = list(values)
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def same_bits(left, right):
    return left.shape == right.shape and left.tobytes() == right.tobytes()


def audit(key, manifest):
    directory = ROOT / f"cross-model-{key}-cpu-v1-20261029-01"
    receipt_file = ROOT / f"cross-model-{key}-controller-v1-20261029-01" / "exit-observation.json"
    replay_file = ROOT / f"cross-model-{key}-cpu-replay-v1.json"
    complete_file = directory / "complete.json"
    complete, receipt = read_json(complete_file), read_json(receipt_file)
    metadata = read_json(directory / "metadata.json")
    replay = read_json(replay_file)
    assert receipt["exit_code"] == 0 and receipt["status"] == "complete"
    assert receipt["source_commit"] == metadata["source_commit"] == EXPECTED_SOURCE
    assert receipt["complete_sha256"] == sha(complete_file)
    assert receipt["metadata_sha256"] == sha(directory / "metadata.json")
    assert metadata["case_sha256"] == EXPECTED_CASE_SHA
    assert complete["status"] == "complete" and complete["events"] == 192
    for field, filename in {
        "events_sha256": "events.jsonl", "logits_sha256": "logits.f32",
        "witness_sha256": "witness.f32", "memory_sha256": "memory.jsonl",
        "metadata_sha256": "metadata.json", "starts_sha256": "starts.jsonl",
        "retention_errors_sha256": "retention-errors.jsonl",
    }.items():
        assert complete[field] == sha(directory / filename), (key, field)

    cases = manifest["cases"]
    assert len(cases) == 32
    vocab = {"gpt": 50257, "bert": 30522}[key]
    logits_path, witness_path = directory / "logits.f32", directory / "witness.f32"
    assert logits_path.stat().st_size == 192 * vocab * 4
    assert witness_path.stat().st_size == 192 * 2 * 768 * 4
    logits = np.memmap(logits_path, dtype="<f4", mode="r", shape=(192, vocab))
    witnesses = np.memmap(witness_path, dtype="<f4", mode="r", shape=(192, 2, 768))
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 192
    indexed = {}
    for i, event in enumerate(events):
        assert event["event_index"] == i and event["logits_offset"] == i * vocab * 4
        assert event["witness_offset"] == i * 2 * 768 * 4
        assert event["hook_count"] == 1 and event["returned_status"] == "returned"
        assert event["full_logits_finite"] and np.isfinite(logits[i]).all()
        assert np.isfinite(witnesses[i]).all()
        case_index = event["case_index"]
        assert cases[case_index]["case_id"] == event["case_id"]
        arm_key = (case_index, event["arm"])
        assert arm_key not in indexed
        indexed[arm_key] = i
    required = {"clean", "self", "clean_after", "opposite", "same", "random"}
    for n in range(32):
        assert {a for i, a in indexed if i == n} == required

    target_ids = manifest["targets"][key]
    replay_cases = {case["case_id"]: case for case in replay["cases"]}
    results = []
    norm_errors = []
    direction_errors = []
    for i, case in enumerate(cases):
        positions = {a: indexed[(i, a)] for a in required}
        clean_i = positions["clean"]
        base = np.asarray(witnesses[clean_i, 0], dtype=np.float64)
        assert same_bits(witnesses[clean_i, 0], witnesses[clean_i, 1])
        for arm in required:
            assert same_bits(witnesses[positions[arm], 0], witnesses[clean_i, 0])
        for arm in ("self", "clean_after"):
            assert same_bits(witnesses[positions[arm], 1], witnesses[clean_i, 1])
            assert same_bits(logits[positions[arm]], logits[clean_i])

        effects = {}
        ci = np.asarray(logits[clean_i], dtype=np.float64)
        clean_margin = float(ci[target_ids["is"]] - ci[target_ids["are"]])
        sign = -1 if case["number"] == 0 else 1
        correct = clean_margin > 0 if case["number"] == 0 else clean_margin < 0
        assert clean_margin != 0
        top = int(np.argmax(ci))
        top_ties = int(np.count_nonzero(ci == ci[top]))
        exp = np.exp(ci - ci[top])
        mass = float((exp[target_ids["is"]] + exp[target_ids["are"]]) / np.sum(exp))
        distances = {}
        actual_norms = {}
        for arm in ("opposite", "same", "random"):
            event_i = positions[arm]
            event = events[event_i]
            assert event["requested_q"] == 2.0
            delta = np.asarray(witnesses[event_i, 1], dtype=np.float64) - base
            actual = float(np.linalg.norm(delta))
            actual_norms[arm] = actual
            norm_errors.append(abs(actual - 2.0))
            assert abs(actual - event["actual_l2"]) < 1e-12
            assert abs(actual - 2.0) <= 2e-4
            if arm != "random":
                donor = cases[event["donor_index"]]
                assert donor["frame"] == case["frame"]
                assert donor["attractor_number"] == case["attractor_number"]
                if arm == "opposite":
                    assert donor["number"] == 1 - case["number"] and donor["lemma"] == case["lemma"]
                else:
                    assert donor["number"] == case["number"] and donor["lemma"] == case["same_number_donor_lemma"]
                donor_i = positions["clean"] if event["donor_index"] == i else indexed[(event["donor_index"], "clean")]
                direction = np.asarray(witnesses[donor_i, 0], dtype=np.float64) - base
                norm = float(np.linalg.norm(direction))
                distances[arm] = norm
                assert abs(norm - event[f"{arm}_raw_norm"]) < 1e-10
            else:
                assert event["random_seed"] == 20261029 + i
                direction = np.random.Generator(np.random.PCG64(event["random_seed"])).standard_normal(base.shape)
                norm = float(np.linalg.norm(direction))
            error = float(np.linalg.norm(delta - 2.0 * direction / norm) / 2.0)
            direction_errors.append(error)
            assert error <= 1e-4
            edited = np.asarray(logits[event_i], dtype=np.float64)
            margin = float(edited[target_ids["is"]] - edited[target_ids["are"]])
            effect = sign * (margin - clean_margin)
            effects[arm] = effect
            replay_effect = replay_cases[case["case_id"]]["effects"][arm]["donorward_effect"]
            assert abs(effect - replay_effect) < 1e-12
        opposite_i = positions["opposite"]
        opp = np.asarray(logits[opposite_i], dtype=np.float64)
        opposite_margin = float(opp[target_ids["is"]] - opp[target_ids["are"]])
        opposite_correct = opposite_margin > 0 if case["number"] == 0 else opposite_margin < 0
        assert opposite_margin != 0
        opp_top = int(np.argmax(opp))
        opp_top_ties = int(np.count_nonzero(opp == opp[opp_top]))
        assert abs(clean_margin - replay_cases[case["case_id"]]["clean_margin"]["is_are"]) < 1e-12
        results.append({
            "case_id": case["case_id"], "frame": case["frame"], "lemma": case["lemma"],
            "subject_number": case["number"], "attractor_number": case["attractor_number"],
            "clean_correct": correct, "clean_margin": clean_margin,
            "opposite_margin": opposite_margin, "opposite_correct": opposite_correct,
            "effects": effects, "p_is_or_are": mass, "clean_top_id": top,
            "clean_top_ties": top_ties, "opposite_top_id": opp_top,
            "opposite_top_ties": opp_top_ties, "donor_distances": distances,
            "actual_patch_norms": actual_norms,
        })

    opposite_effects = [r["effects"]["opposite"] for r in results]
    return {
        "source_hashes": {"receipt": sha(receipt_file), "complete": sha(complete_file),
                          "events": sha(directory / "events.jsonl"), "logits": sha(logits_path),
                          "witness": sha(witness_path), "replay": sha(replay_file)},
        "cases": len(results), "clean_binary_correct": sum(r["clean_correct"] for r in results),
        "clean_correct_by_subject_attractor": {
            f'{s}-{a}': {"correct": sum(r["clean_correct"] for r in results if r["subject_number"] == s and r["attractor_number"] == a),
                         "cases": sum(r["subject_number"] == s and r["attractor_number"] == a for r in results)}
            for s in (0, 1) for a in (0, 1)},
        "opposite_beats_both": sum(r["effects"]["opposite"] > r["effects"]["same"] and
                                    r["effects"]["opposite"] > r["effects"]["random"] for r in results),
        "opposite_donorward_positive": sum(x > 0 for x in opposite_effects),
        "opposite_effect_range": summary(opposite_effects),
        "correct_to_wrong": sum(r["clean_correct"] and not r["opposite_correct"] for r in results),
        "wrong_to_correct": sum(not r["clean_correct"] and r["opposite_correct"] for r in results),
        "full_vocab_top_changes": sum(r["clean_top_id"] != r["opposite_top_id"] for r in results),
        "clean_top_ties": sum(r["clean_top_ties"] > 1 for r in results),
        "opposite_top_ties": sum(r["opposite_top_ties"] > 1 for r in results),
        "target_verb_top": sum(r["clean_top_id"] in (target_ids["is"], target_ids["are"]) for r in results),
        "clean_top_ids": {str(t): sum(r["clean_top_id"] == t for r in results)
                          for t in sorted({r["clean_top_id"] for r in results})},
        "p_is_plus_p_are": summary(r["p_is_or_are"] for r in results),
        "opposite_donor_norm": summary(r["donor_distances"]["opposite"] for r in results),
        "same_donor_norm": summary(r["donor_distances"]["same"] for r in results),
        "maximum_abs_patch_l2_error": max(norm_errors),
        "maximum_relative_direction_error": max(direction_errors),
        "case_results": results,
    }


def main():
    assert sha(MANIFEST) == EXPECTED_CASE_SHA
    manifest = read_json(MANIFEST)
    assert manifest["schema"] == "cross-model-agreement-cases-v1"
    report = {"schema": "independent-cross-model-methods-audit-v1",
              "scope": "saved public inputs and raw outputs only; no model forwards",
              "audit_source_sha256": sha(Path(__file__)), "case_manifest_sha256": sha(MANIFEST),
              "gpt": audit("gpt", manifest), "bert": audit("bert", manifest)}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("audit_sha256", sha(OUTPUT))
    print("source_sha256", sha(Path(__file__)))
    for key in ("gpt", "bert"):
        d = {k: v for k, v in report[key].items() if k != "case_results"}
        print(key, json.dumps(d, sort_keys=True))


if __name__ == "__main__":
    main()
