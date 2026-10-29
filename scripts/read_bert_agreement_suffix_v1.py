"""Replay the fixed paired BERT suffix study from retained bytes, without a model."""
import sys

if __name__ == "__main__" and (not sys.flags.isolated or not sys.flags.no_site
                               or not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError("direct reader requires -I -S -B and a fresh pycache prefix")

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


CASE_SHA = "1da4f8e4773009fedc750480c21597fc60ecaa0c990dcff0af15a5c7101c3dca"
SOURCE_PATHS = ("scripts/bert_agreement_suffix_v1.py", "scripts/read_bert_agreement_suffix_v1.py",
                "scripts/observe_bert_agreement_suffix_v1.py", "scripts/cross_model_agreement_v1.py",
                "scripts/read_cross_model_agreement_v1.py", "docs/BERT_AGREEMENT_SUFFIX_EXECUTION_V1.md",
                "docs/BERT_AGREEMENT_SUFFIX_V1.md",
                "experiments/bert_agreement_suffix_cases_v1.json")
TARGETS = {"is": 2003, "are": 2024, "was": 2001, "were": 2020}
ASSETS = {"config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
          "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
          "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
          "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
          "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"}
VOCAB = 30522
SEED = 20261029
PENDING_NAME = "observer-pending.json"


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(repo, commit, git):
    if (not re.fullmatch("[0-9a-f]{40}", commit) or not git.is_absolute()
            or not git.is_file() or repo.resolve() != Path(__file__).resolve().parents[1]):
        raise ValueError("invalid source commit, checkout or Git executable")
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PATH") if key in os.environ}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"})
    if git.resolve().is_relative_to(repo.resolve()):
        raise ValueError("Git executable must be selected outside the checkout")
    kind = subprocess.run([str(git), "-C", str(repo), "cat-file", "-t", commit], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env).stdout.strip()
    if kind != b"commit":
        raise ValueError("source identifier does not name a commit")
    result = {}
    for name in SOURCE_PATHS:
        data = subprocess.run([str(git), "-C", str(repo), "cat-file", "blob", commit + ":" + name], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env).stdout
        if (repo / name).read_bytes() != data:
            raise ValueError("reader or source differs: " + name)
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def bootstrap():
    if (__spec__ is not None or not sys.flags.isolated or not sys.flags.no_site
            or not sys.dont_write_bytecode or not sys.pycache_prefix
            or sys._xoptions.get("pycache_prefix") != sys.pycache_prefix
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()):
        raise RuntimeError("run direct base Python with -I -S -B and a fresh empty pycache prefix")
    prefix = Path(sys.pycache_prefix)
    if not prefix.is_absolute() or not prefix.is_dir() or prefix.is_symlink() or any(prefix.iterdir()):
        raise RuntimeError("invalid pycache prefix")
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("site", "repo", "git"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args, _ = parser.parse_known_args()
    if not args.site.is_absolute() or not args.site.is_dir() or args.site.is_symlink():
        raise ValueError("invalid dependency site")
    hashes = source_hashes(args.repo, args.source_commit, args.git)
    git_sha = file_sha(args.git)
    sys.path.append(str(args.site.resolve()))
    return hashes, git_sha


if __name__ == "__main__":
    EARLY_SOURCE, EARLY_GIT = bootstrap()
else:
    EARLY_SOURCE = EARLY_GIT = None

import numpy as np
if __name__ == "__main__":
    helper_path = Path(__file__).resolve().with_name("read_cross_model_agreement_v1.py")
    spec = importlib.util.spec_from_file_location("_bert_suffix_verified_math", helper_path)
    mathutil = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mathutil)
    if Path(mathutil.__file__).resolve() != helper_path:
        raise RuntimeError("math helper module origin differs")
else:
    from scripts import read_cross_model_agreement_v1 as mathutil


def read_jsonl(path):
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("partial journal")
    return [json.loads(row) for row in raw.splitlines()]


def compare_bits(a, b):
    return np.asarray(a, dtype="<f4").tobytes() == np.asarray(b, dtype="<f4").tobytes()


def correct(margin, number):
    return margin > 0 if number == 0 else margin < 0


def clean_score(logits, number):
    mathutil.require_subnormals()
    x = np.asarray(logits, dtype=np.float64)
    margin = float(x[TARGETS["is"]] - x[TARGETS["are"]])
    weights = np.exp(x - float(np.max(x)))
    unresolved = bool(np.any(weights == 0))
    denominator = math.fsum(float(value) for value in weights)
    top = int(np.argmax(x))
    ties = int(np.count_nonzero(x == x[top]))
    if unresolved:
        p_is = p_are = pair = None
    else:
        p_is, p_are = float(weights[TARGETS["is"]] / denominator), float(weights[TARGETS["are"]] / denominator)
        pair = p_is + p_are
        if p_is == 0 or p_are == 0 or pair == 0:
            unresolved = True
            p_is = p_are = pair = None
    return {"is_are_margin": margin, "was_were_margin": float(x[TARGETS["was"]] - x[TARGETS["were"]]),
            "binary_correct": bool(correct(margin, number)), "binary_tie": margin == 0,
            "p_is": p_is, "p_are": p_are, "p_is_plus_p_are": pair,
            "probability_numerically_unresolved": unresolved, "top_id": top, "top_ties": ties}


def arm_score(logits, baseline, number):
    base = clean_score(baseline, number)
    arm = clean_score(logits, number)
    sign = -1 if number == 0 else 1
    return {"donorward_effect": sign * (arm["is_are_margin"] - base["is_are_margin"]),
            "was_were_margin": arm["was_were_margin"], "binary_correct": arm["binary_correct"],
            "binary_tie": arm["binary_tie"], **mathutil.distribution(baseline, logits)}


def close_number(a, b):
    if type(a) not in (int, float) or type(b) not in (int, float):
        return False
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-12)


def probability_ratio(new, old):
    if old is None or new is None:
        return None, "probability_unresolved"
    if old == 0:
        return None, "zero_denominator"
    try:
        value = new / old
    except OverflowError:
        return None, "overflow"
    if not math.isfinite(value):
        return None, "overflow"
    return value, "finite"


def valid_memory(value, device):
    if device == "cpu":
        return value is None
    keys = {"cuda_free", "cuda_total", "allocated", "reserved", "peak_allocated", "peak_reserved"}
    return (type(value) is dict and set(value) == keys
            and all(type(value[k]) is int and value[k] >= 0 for k in keys)
            and value["cuda_free"] <= value["cuda_total"]
            and value["allocated"] <= value["peak_allocated"]
            and value["reserved"] <= value["peak_reserved"])


def donor_indices(cases, case):
    index = {(c["frame"], c["attractor_number"], c["lemma"], c["number"]): c["index"] for c in cases}
    return (index[case["frame"], case["attractor_number"], case["lemma"], 1 - case["number"]],
            index[case["frame"], case["attractor_number"], case["same_number_donor_lemma"], case["number"]])


def intended_landing(base, direction, q):
    base64 = np.asarray(base, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0 or not math.isfinite(q) or q <= 0:
        raise ValueError("invalid donor direction or dose")
    intended = q * (direction / norm)
    value = np.asarray(base64 + intended, dtype="<f4")
    actual_delta = value.astype(np.float64) - base64
    actual = float(np.linalg.norm(actual_delta))
    error = float(np.linalg.norm(actual_delta - intended)) / q
    if not math.isfinite(actual) or actual <= 0 or abs(actual - q) > 1e-4 * q or error > 1e-4:
        raise ValueError("invalid rounded landing")
    return value, actual, error


def group_summary(rows):
    n = len(rows)
    t = [r["terminal"] for r in rows]
    c = [r["continued"] for r in rows]
    changes = [r["mass_change"] for r in rows]
    means = lambda name, variant: math.fsum(row[variant][name] for row in rows) / n
    return {"cases": n, "terminal_correct": sum(x["clean"]["binary_correct"] for x in t),
            "continued_correct": sum(x["clean"]["binary_correct"] for x in c),
            "mean_mass_change": None if any(x is None for x in changes) else math.fsum(changes) / n,
            "terminal_mean_opposite_minus_same": means("opposite_minus_same", "terminal"),
            "continued_mean_opposite_minus_same": means("opposite_minus_same", "continued"),
            "terminal_mean_opposite_minus_random": means("opposite_minus_random", "terminal"),
            "continued_mean_opposite_minus_random": means("opposite_minus_random", "continued")}


def summarize(rows):
    if len(rows) != 32:
        raise ValueError("all 32 paired cases are required")
    summary = {"cases": len(rows)}
    changes = [r["mass_change"] for r in rows]
    summary["mass_increases"] = None if any(x is None for x in changes) else sum(x > 0 for x in changes)
    summary["mean_mass_change"] = None if any(x is None for x in changes) else math.fsum(changes) / len(rows)
    summary["mass_hypothesis_passes"] = (None if summary["mass_increases"] is None else
                                         summary["mass_increases"] >= 24 and summary["mean_mass_change"] > 0)
    for variant in ("terminal", "continued"):
        selected = [r[variant] for r in rows]
        clean_correct = sum(x["clean"]["binary_correct"] for x in selected)
        mean_same = math.fsum(x["opposite_minus_same"] for x in selected) / len(rows)
        mean_random = math.fsum(x["opposite_minus_random"] for x in selected) / len(rows)
        positive = sum(x["opposite_minus_same"] > 0 and x["opposite_minus_random"] > 0 for x in selected)
        summary[variant] = {"correct": clean_correct, "baseline_limited": clean_correct < 24,
                            "mean_opposite_minus_same": mean_same, "mean_opposite_minus_random": mean_random,
                            "positive_against_both": positive,
                            "donor_hypothesis_passes": mean_same > 0 and mean_random > 0 and positive >= 24}
        for arm in ("opposite", "same", "random") + (("fixed_terminal_opposite",) if variant == "continued" else ()):
            summary[variant][arm] = {
                "binary_correct_after": sum(x["arms"][arm]["binary_correct"] for x in selected),
                "strict_binary_sign_flips": sum(not x["clean"]["binary_tie"] and not x["arms"][arm]["binary_tie"]
                                                and x["clean"]["binary_correct"] != x["arms"][arm]["binary_correct"]
                                                for x in selected),
                "unique_top_changes": sum(x["arms"][arm]["unique_top1_changed"] is True for x in selected),
                "top_change_unresolved_due_to_ties": sum(x["arms"][arm]["unique_top1_changed"] is None for x in selected),
            }
    summary["correct_in_both"] = sum(r["terminal"]["clean"]["binary_correct"] and
                                     r["continued"]["clean"]["binary_correct"] for r in rows)
    summary["continued_accuracy_not_lower"] = summary["continued"]["correct"] >= summary["terminal"]["correct"]
    summary["descriptive_baseline_correct_subsets"] = {}
    for subset in ("terminal", "continued", "both"):
        selected = [r for r in rows if (r["terminal"]["clean"]["binary_correct"] if subset == "terminal"
                                        else r["continued"]["clean"]["binary_correct"] if subset == "continued"
                                        else r["terminal"]["clean"]["binary_correct"] and r["continued"]["clean"]["binary_correct"])]
        summary["descriptive_baseline_correct_subsets"][subset] = {
            "case_ids": [r["case_id"] for r in selected], "cases": len(selected),
            "terminal_mean_opposite_minus_same": None if not selected else math.fsum(r["terminal"]["opposite_minus_same"] for r in selected) / len(selected),
            "continued_mean_opposite_minus_same": None if not selected else math.fsum(r["continued"]["opposite_minus_same"] for r in selected) / len(selected),
        }
    summary["dose"] = {"min": min(r["requested_q"] for r in rows), "median": statistics.median(r["requested_q"] for r in rows),
                       "max": max(r["requested_q"] for r in rows)}
    for field in ("context_recipe_delta_same", "context_recipe_delta_random", "fixed_direction_context_delta",
                  "shared_random_context_delta", "fixed_minus_random_context_delta"):
        values = [r[field] for r in rows]
        summary[field] = {"mean": math.fsum(values) / len(values), "median": statistics.median(values),
                          "positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values)}
    summary["transitions"] = {key: sum(r["binary_transition"] == key for r in rows)
                              for key in ("correct_to_wrong", "wrong_to_correct", "correct_to_correct", "wrong_to_wrong")}
    summary["by_frame"] = {str(v): group_summary([r for r in rows if r["frame"] == v])
                           for v in sorted({r["frame"] for r in rows})}
    summary["by_lemma"] = {str(v): group_summary([r for r in rows if r["lemma"] == v])
                           for v in sorted({r["lemma"] for r in rows})}
    return summary


def replay(attempt, manifest, receipt, receipt_sha, complete_sha, expected_source, expected_git, expected_commit):
    mathutil.require_subnormals()
    if (attempt / "failure.txt").exists() or (receipt.parent / PENDING_NAME).exists() or file_sha(receipt) != receipt_sha:
        raise ValueError("failure or observer-pending sentinel present, or receipt differs")
    completed_path, metadata_path = attempt / "complete.json", attempt / "metadata.json"
    if file_sha(completed_path) != complete_sha:
        raise ValueError("completion digest differs")
    completed, metadata, exit_record = (json.loads(p.read_bytes()) for p in (completed_path, metadata_path, receipt))
    if (completed.get("status") != "complete" or type(completed.get("events")) is not int
            or completed.get("events") != 416 or completed.get("model_weakref_dead") is not True
            or exit_record.get("schema") != "bert-agreement-suffix-direct-child-exit-v1"
            or exit_record.get("status") != "complete" or exit_record.get("exit_code") != 0
            or type(exit_record.get("exit_code")) is not int or type(exit_record.get("child_pid")) is not int
            or exit_record.get("interrupted") is not None or exit_record.get("child_pid") != metadata.get("pid")
            or exit_record.get("attempt_name") != attempt.name or exit_record.get("complete_sha256") != complete_sha
            or exit_record.get("metadata_sha256") != file_sha(metadata_path)
            or exit_record.get("source_commit") != metadata.get("source_commit")
            or metadata.get("source_commit") != expected_commit
            or metadata.get("sources_sha256") != expected_source or metadata.get("git_sha256") != expected_git
            or metadata.get("assets_sha256") != ASSETS
            or metadata.get("schema") != "bert-agreement-suffix-attempt-v1"
            or type(metadata.get("pid")) is not int or metadata.get("pid") <= 0
            or type(metadata.get("seed")) is not int or metadata.get("seed") != SEED
            or metadata.get("device") not in ("cpu", "cuda")
            or metadata.get("base_interpreter_direct") is not True or metadata.get("isolated") is not True
            or any(k not in metadata for k in ("initial_memory", "gpu_name", "cublas_workspace_config"))
            or any(type(metadata.get(k)) is not str or not metadata[k].strip() or len(metadata[k]) > 512
                   for k in ("python", "torch", "transformers", "numpy"))
            or type(metadata.get("start_unix")) not in (int, float) or not math.isfinite(metadata["start_unix"])
            or metadata["start_unix"] <= 0 or not valid_memory(metadata.get("initial_memory"), metadata.get("device"))
            or (metadata.get("device") == "cpu" and metadata.get("gpu_name") is not None)
            or (metadata.get("device") == "cuda" and (type(metadata.get("gpu_name")) is not str or not metadata["gpu_name"].strip()))
            or metadata.get("deterministic_algorithms") is not True or metadata.get("tf32") is not False
            or metadata.get("cudnn_benchmark") is not False or metadata.get("cudnn_deterministic") is not True
            or type(metadata.get("intra_threads")) is not int or metadata.get("intra_threads") != 1
            or type(metadata.get("inter_threads")) is not int or metadata.get("inter_threads") != 1
            or metadata.get("cublas_workspace_config") != (":4096:8" if metadata.get("device") == "cuda" else None)
            or metadata.get("case_sha256") != CASE_SHA):
        raise ValueError("completion, receipt or source metadata differs")
    for name in ("metadata.json", "loading.json", "starts.jsonl", "events.jsonl", "memory.jsonl",
                 "retention-errors.jsonl", "logits.f32", "witness.f32"):
        if completed.get(name.replace(".", "_") + "_sha256") != file_sha(attempt / name):
            raise ValueError("retained file digest differs: " + name)
    loading = json.loads((attempt / "loading.json").read_bytes())
    if (set(loading) != {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
            or any(type(value) is not list or len(value) != 0 for value in loading.values())):
        raise ValueError("checkpoint loading diagnostics are not empty")
    if (attempt / "retention-errors.jsonl").stat().st_size:
        raise ValueError("retention errors exist")
    starts, events = read_jsonl(attempt / "starts.jsonl"), read_jsonl(attempt / "events.jsonl")
    if len(starts) != 416 or len(events) != 416:
        raise ValueError("incomplete event count")
    memory = read_jsonl(attempt / "memory.jsonl")
    if len(memory) != 416:
        raise ValueError("incomplete per-forward memory samples")
    for i, sample in enumerate(memory):
        if set(sample) != {"event_index", "memory"} or type(sample.get("event_index")) is not int or sample["event_index"] != i:
            raise ValueError("memory sample order differs")
        if not valid_memory(sample.get("memory"), metadata["device"]):
            raise ValueError("per-forward memory sample differs")
    if (attempt / "logits.f32").stat().st_size != 416 * VOCAB * 4 or (attempt / "witness.f32").stat().st_size != 416 * 2 * 768 * 4:
        raise ValueError("saved tensor byte counts differ")
    logits = np.memmap(attempt / "logits.f32", dtype="<f4", mode="r", shape=(416, VOCAB))
    witnesses = np.memmap(attempt / "witness.f32", dtype="<f4", mode="r", shape=(416, 2, 768))
    cases, order = manifest["cases"], []
    for case in cases:
        for variant in ("terminal", "continued"):
            order.append((case, variant, "clean"))
    for case in cases:
        for variant in ("terminal", "continued"):
            arms = ["self", "opposite", "same", "random"]
            if variant == "continued":
                arms.append("fixed_terminal_opposite")
            arms.append("clean_after")
            order.extend((case, variant, arm) for arm in arms)
    saved = {}
    for i, (case, variant, arm) in enumerate(order):
        mathutil.require_subnormals()
        start, event, row = starts[i], events[i], case["variants"][variant]
        fixed = {"case_id": case["case_id"], "case_index": case["index"], "variant": variant,
                 "arm": arm, "event_index": i}
        if any(start.get(k) != v or event.get(k) != v for k, v in fixed.items()):
            raise ValueError("forward order differs")
        if any(type(record.get(key)) is not int for record in (start, event) for key in ("case_index", "event_index")):
            raise ValueError("forward index type differs")
        if (start.get("input_ids") != row["input_ids"] or start.get("attention_mask") != row["attention_mask"]
                or any(type(x) is not int for x in start.get("input_ids", []) + start.get("attention_mask", []))
                or start.get("subject_position") != row["subject_position"] or start.get("output_position") != row["output_position"]
                or type(start.get("subject_position")) is not int or type(start.get("output_position")) is not int
                or type(event.get("hook_count")) is not int or event.get("hook_count") != 1
                or type(event.get("logits_offset")) is not int or type(event.get("witness_offset")) is not int
                or event.get("returned_status") != "returned"
                or event.get("full_logits_finite") is not True or event.get("logits_offset") != i * VOCAB * 4
                or event.get("witness_offset") != i * 2 * 768 * 4):
            raise ValueError("forward metadata differs")
        if not np.isfinite(logits[i]).all() or not np.isfinite(witnesses[i]).all():
            raise ValueError("nonfinite saved tensor")
        saved[case["index"], variant, arm] = (event, logits[i], witnesses[i, 0], witnesses[i, 1])
    rows = []
    for case in cases:
        index, (opp, same) = case["index"], donor_indices(cases, case)
        base = {v: saved[index, v, "clean"][2] for v in ("terminal", "continued")}
        directions, norms = {}, {}
        for variant in ("terminal", "continued"):
            if not compare_bits(base[variant], saved[index, variant, "clean"][3]):
                raise ValueError("clean capture changed its site")
            for arm, donor in (("opposite", opp), ("same", same)):
                delta = saved[donor, variant, "clean"][2].astype(np.float64) - base[variant].astype(np.float64)
                norm = float(np.linalg.norm(delta))
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("invalid donor norm")
                directions[variant, arm], norms[variant + "_" + arm] = delta, norm
        q = min(2.0, *norms.values())
        cosines = {arm: float(np.dot(directions["terminal", arm], directions["continued", arm]) /
                              (norms["terminal_" + arm] * norms["continued_" + arm]))
                   for arm in ("opposite", "same")}
        random = np.random.Generator(np.random.PCG64(SEED + index)).standard_normal(768)
        row_result = {"case_id": case["case_id"], "frame": case["frame"], "lemma": case["lemma"],
                      "number": case["number"], "attractor_number": case["attractor_number"],
                      "requested_q": q, "natural_norms": norms, "direction_cosines": cosines}
        for variant in ("terminal", "continued"):
            _, baseline, _, _ = saved[index, variant, "clean"]
            arms = ["self", "opposite", "same", "random"]
            if variant == "continued":
                arms.append("fixed_terminal_opposite")
            arms.append("clean_after")
            outcomes = {}
            for arm in arms:
                event, values, before, after = saved[index, variant, arm]
                if (not compare_bits(before, base[variant]) or not close_number(event["requested_q"], q)
                        or any(not close_number(event["natural_norms"][k], v) for k, v in norms.items())
                        or any(not close_number(event["direction_cosines"][k], v) for k, v in cosines.items())):
                    raise ValueError("recorded dose, donor geometry or source differs")
                if arm in ("self", "clean_after"):
                    expected_donor = index if arm == "self" else None
                    if (not compare_bits(before, after) or not compare_bits(values, baseline)
                            or event.get("donor_index") != expected_donor
                            or (arm == "self" and type(event.get("donor_index")) is not int)
                            or event.get("random_seed") is not None
                            or type(event.get("actual_l2")) not in (int, float) or event["actual_l2"] != 0
                            or type(event.get("relative_direction_error")) not in (int, float)
                            or event["relative_direction_error"] != 0):
                        raise ValueError("self or clean-after evidence differs from clean")
                    continue
                direction = (random if arm == "random" else directions["terminal", "opposite"]
                             if arm == "fixed_terminal_opposite" else directions[variant, arm])
                expected, actual, error = intended_landing(base[variant], direction, q)
                donor = None if arm == "random" else same if arm == "same" else opp
                if (not compare_bits(after, expected) or event["donor_index"] != donor
                        or (donor is not None and type(event["donor_index"]) is not int)
                        or event["random_seed"] != (SEED + index if arm == "random" else None)
                        or (arm == "random" and type(event["random_seed"]) is not int)
                        or not close_number(event["actual_l2"], actual)
                        or not close_number(event["relative_direction_error"], error)):
                    raise ValueError("recorded donor or rounded landing differs")
                outcomes[arm] = arm_score(values, baseline, case["number"])
            row_result[variant] = {"clean": clean_score(baseline, case["number"]), "arms": outcomes,
                                   "opposite_minus_same": outcomes["opposite"]["donorward_effect"] - outcomes["same"]["donorward_effect"],
                                   "opposite_minus_random": outcomes["opposite"]["donorward_effect"] - outcomes["random"]["donorward_effect"]}
        t, c = row_result["terminal"], row_result["continued"]
        old_mass, new_mass = t["clean"]["p_is_plus_p_are"], c["clean"]["p_is_plus_p_are"]
        row_result["mass_change"] = None if old_mass is None or new_mass is None else new_mass - old_mass
        row_result["mass_ratio"], row_result["mass_ratio_status"] = probability_ratio(new_mass, old_mass)
        tc, cc = t["clean"]["binary_correct"], c["clean"]["binary_correct"]
        row_result["binary_transition"] = ("correct" if tc else "wrong") + "_to_" + ("correct" if cc else "wrong")
        row_result["context_recipe_delta_same"] = c["opposite_minus_same"] - t["opposite_minus_same"]
        row_result["context_recipe_delta_random"] = c["opposite_minus_random"] - t["opposite_minus_random"]
        fixed_delta = c["arms"]["fixed_terminal_opposite"]["donorward_effect"] - t["arms"]["opposite"]["donorward_effect"]
        random_delta = c["arms"]["random"]["donorward_effect"] - t["arms"]["random"]["donorward_effect"]
        row_result["fixed_direction_context_delta"] = fixed_delta
        row_result["shared_random_context_delta"] = random_delta
        row_result["fixed_minus_random_context_delta"] = fixed_delta - random_delta
        rows.append(row_result)
    mathutil.require_subnormals()
    return {"schema": "bert-agreement-suffix-replay-v1", "case_sha256": CASE_SHA,
            "receipt_sha256": receipt_sha, "complete_sha256": complete_sha,
            "source_commit": metadata["source_commit"], "device": metadata["device"],
            "summary": summarize(rows), "cases": rows,
            "scope": "Retained selected-position tensors and cooperative exit receipt; no execution or unsaved-position attestation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "repo", "git", "attempt", "cases", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "receipt-sha256", "complete-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if file_sha(args.cases) != CASE_SHA or args.cases.resolve() != (args.repo / SOURCE_PATHS[-1]).resolve():
        raise ValueError("case bytes or location differ")
    hashes = source_hashes(args.repo, args.source_commit, args.git)
    if hashes != EARLY_SOURCE or file_sha(args.git) != EARLY_GIT:
        raise ValueError("source or Git executable changed after early verification")
    report = replay(args.attempt, json.loads(args.cases.read_bytes()), args.receipt,
                    args.receipt_sha256, args.complete_sha256, hashes, EARLY_GIT, args.source_commit)
    if source_hashes(args.repo, args.source_commit, args.git) != hashes or file_sha(args.git) != EARLY_GIT:
        raise ValueError("source or Git executable changed during replay")
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as out:
        out.write(encoded)
    print("report_sha256", hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    main()
