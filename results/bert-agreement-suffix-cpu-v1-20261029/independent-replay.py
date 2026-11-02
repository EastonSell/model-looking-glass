"""Independently replay the CPU BERT suffix study from quiescent saved bytes.

This checker does not import a model or execute a forward. It checks the
recorded selected-position tensors, fixed intervention mathematics, public
source identities, and a separately supplied direct-child exit receipt.
These observations do not attest actual execution or unsaved positions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

if not __debug__:
    raise RuntimeError("run without -O or -OO")

if __name__ == "__main__":
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise RuntimeError("use an isolated -I -S -B interpreter")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--site", required=True, type=Path)
    initial, _ = pre.parse_known_args()
    if not initial.site.is_absolute() or not initial.site.is_dir() or initial.site.is_symlink():
        raise ValueError("dependency site must be an absolute ordinary directory")
    sys.path.append(str(initial.site.resolve()))

import numpy as np

CASE_SHA = "1da4f8e4773009fedc750480c21597fc60ecaa0c990dcff0af15a5c7101c3dca"
ASSETS = {"config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
          "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
          "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
          "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
          "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"}
SOURCE_PATHS = ("scripts/bert_agreement_suffix_v1.py", "scripts/read_bert_agreement_suffix_v1.py",
                "scripts/observe_bert_agreement_suffix_v1.py", "scripts/cross_model_agreement_v1.py",
                "scripts/read_cross_model_agreement_v1.py", "docs/BERT_AGREEMENT_SUFFIX_EXECUTION_V1.md",
                "docs/BERT_AGREEMENT_SUFFIX_V1.md", "experiments/bert_agreement_suffix_cases_v1.json")
TARGETS = {"is": 2003, "are": 2024, "was": 2001, "were": 2020}
SEED = 20261029


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def floating(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    def constant(value):
        raise ValueError("nonfinite JSON constant " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_float=floating, parse_constant=constant)


def jsonl(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("missing JSONL line terminator: " + str(path))
    return [strict_json(line) for line in raw.splitlines()]


def exact_int(value, expected):
    return type(value) is int and value == expected


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite_number(a) and finite_number(b) and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)


def bits(x):
    return np.asarray(x, dtype="<f4").tobytes()


def require_subnormals():
    """Check the CPU thread used for this replay; this says nothing about CUDA."""
    with np.errstate(under="ignore"):
        a = np.array([np.finfo(np.float32).tiny], dtype=np.float32) / np.float32(2)
        b = np.array([np.finfo(np.float64).tiny], dtype=np.float64) / np.float64(2)
        c = np.array([1], dtype=np.uint32).view(np.float32) * np.float32(1)
        d = np.array([1], dtype=np.uint64).view(np.float64) * np.float64(1)
    if (a.view(np.uint32)[0] != 1 << 22 or b.view(np.uint64)[0] != 1 << 51
            or c.view(np.uint32)[0] != 1 or d.view(np.uint64)[0] != 1):
        raise RuntimeError("replay thread does not preserve CPU subnormals")


def length(x):
    return math.sqrt(math.fsum(float(a) * float(a) for a in x))


def probability_ratio(new, old):
    if old is None or new is None:
        return None, "probability_unresolved"
    if old == 0:
        return None, "zero_denominator"
    value = new / old
    return (value, "finite") if math.isfinite(value) else (None, "overflow")


def clean_score(vector, number, ids):
    """Float64 full-vocabulary probabilities; mark any underflow unresolved."""
    values = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite selected logits")
    top = float(np.max(values))
    tops = np.flatnonzero(values == top)
    with np.errstate(under="ignore"):
        w = np.exp(values - top)
    unresolved = bool(np.any(w == 0))
    total = math.fsum(map(float, w))
    with np.errstate(under="ignore"):
        probs = w / total
    unresolved = unresolved or bool(np.any(probs == 0))
    m = float(values[ids["is"]] - values[ids["are"]])
    return {"is_are_margin": m, "was_were_margin": float(values[ids["was"]] - values[ids["were"]]),
            "binary_correct": bool(m > 0 if number == 0 else m < 0), "binary_tie": m == 0,
            "p_is": None if unresolved else float(probs[ids["is"]]),
            "p_are": None if unresolved else float(probs[ids["are"]]),
            "p_is_plus_p_are": None if unresolved else float(probs[ids["is"]] + probs[ids["are"]]),
            "probability_numerically_unresolved": unresolved,
            "top_id": int(tops[0]), "top_ties": int(len(tops))}


def distribution(base, edited):
    """TV and KL(clean || edited), with a stable small-change formulation."""
    a, b = [float(x) for x in base], [float(x) for x in edited]
    ta, tb = max(a), max(b)
    ia = [i for i, x in enumerate(a) if x == ta]
    ib = [i for i, x in enumerate(b) if x == tb]
    result = {"clean_top1_id": ia[0], "arm_top1_id": ib[0], "clean_top1_ties": len(ia), "arm_top1_ties": len(ib),
              "representative_top1_id_changed": ia[0] != ib[0],
              "unique_top1_changed": (ia[0] != ib[0] if len(ia) == len(ib) == 1 else None)}
    wa, wb = [math.exp(x - ta) for x in a], [math.exp(x - tb) for x in b]
    if not all(wa) or not all(wb):
        return {**result, "tv": None, "kl_clean_to_arm": None, "numerically_unresolved": True}
    za, zb = math.fsum(wa), math.fsum(wb)
    p = [w / za for w in wa]
    if not all(p) or any(w / zb == 0 for w in wb):
        return {**result, "tv": None, "kl_clean_to_arm": None, "numerically_unresolved": True}
    differences = [y - x for x, y in zip(a, b)]
    anchor = differences[ia[0]]
    relative = [x - anchor for x in differences]
    if max(map(abs, relative)) > 600:
        return {**result, "tv": None, "kl_clean_to_arm": None, "numerically_unresolved": True}
    delta_z = math.fsum(x * math.expm1(y) for x, y in zip(p, relative))
    if not math.isfinite(delta_z) or delta_z <= -1:
        return {**result, "tv": None, "kl_clean_to_arm": None, "numerically_unresolved": True}
    log_z = math.log1p(delta_z)
    ratios = [x - log_z for x in relative]
    tv = math.fsum(x * abs(math.expm1(y)) for x, y in zip(p, ratios)) / 2
    def kl_term(x):
        if abs(x) < 1e-4:
            return x*x*(0.5 + x*(1/6 + x*(1/24 + x/120)))
        return max(0.0, math.expm1(x) - x)
    kl = math.fsum(x * kl_term(y) for x, y in zip(p, ratios))
    return {**result, "tv": tv, "kl_clean_to_arm": kl, "numerically_unresolved": False}


def action_order(cases):
    order = [(c, v, "clean") for c in cases for v in ("terminal", "continued")]
    for c in cases:
        for v in ("terminal", "continued"):
            arms = ["self", "opposite", "same", "random"]
            if v == "continued":
                arms.append("fixed_terminal_opposite")
            order.extend((c, v, a) for a in (*arms, "clean_after"))
    return order


def audit_semantics(cases, starts, events, memory, logits, witness, *, ids=TARGETS, dimensions=768):
    """Pure record/math check. Alternate dimensions/IDs are for synthetic tests only."""
    require_subnormals()
    if len(cases) != 32 or len(starts) != 416 or len(events) != 416 or len(memory) != 416:
        raise ValueError("the fixed case/record counts differ")
    if logits.shape[0] != 416 or witness.shape != (416, 2, dimensions):
        raise ValueError("saved tensor layout differs")
    if max(ids.values()) >= logits.shape[1]:
        raise ValueError("target IDs outside vocabulary")
    vocab = logits.shape[1]
    lookup = {(c["frame"], c["attractor_number"], c["lemma"], c["number"]): c["index"] for c in cases}
    if len(lookup) != 32:
        raise ValueError("case-factor keys repeat")
    saved = {}
    for i, (case, variant, arm) in enumerate(action_order(cases)):
        start, event, sample, row = starts[i], events[i], memory[i], case["variants"][variant]
        for item in (start, event):
            if (not exact_int(item.get("event_index"), i) or not exact_int(item.get("case_index"), case["index"])
                    or item.get("case_id") != case["case_id"] or item.get("variant") != variant or item.get("arm") != arm):
                raise ValueError("forward identity/order differs at " + str(i))
        if (type(sample) is not dict or set(sample) != {"event_index", "memory"}
                or not exact_int(sample.get("event_index"), i) or sample.get("memory") is not None):
            raise ValueError("CPU memory sample sequence differs at " + str(i))
        for name in ("input_ids", "attention_mask"):
            got = start.get(name)
            if type(got) is not list or any(type(x) is not int for x in got) or got != row[name]:
                raise ValueError("start input differs at " + str(i))
        if (not exact_int(start.get("subject_position"), row["subject_position"])
                or not exact_int(start.get("output_position"), row["output_position"])
                or not exact_int(event.get("hook_count"), 1) or event.get("returned_status") != "returned"
                or event.get("full_logits_finite") is not True
                or not exact_int(event.get("logits_offset"), i * vocab * 4)
                or not exact_int(event.get("witness_offset"), i * 2 * dimensions * 4)):
            raise ValueError("forward metadata differs at " + str(i))
        if not np.isfinite(logits[i]).all() or not np.isfinite(witness[i]).all():
            raise ValueError("nonfinite saved tensors at " + str(i))
        saved[(case["index"], variant, arm)] = (event, logits[i], witness[i, 0], witness[i, 1])
    rows = []
    for case in cases:
        i, frame, attractor, lemma, number = (case[k] for k in ("index", "frame", "attractor_number", "lemma", "number"))
        opp = lookup[(frame, attractor, lemma, 1 - number)]
        same = lookup[(frame, attractor, case["same_number_donor_lemma"], number)]
        bases = {v: saved[(i, v, "clean")][2] for v in ("terminal", "continued")}
        dirs, norms = {}, {}
        for v in ("terminal", "continued"):
            if bits(bases[v]) != bits(saved[(i, v, "clean")][3]):
                raise ValueError("clean capture changed its site")
            for arm, donor in (("opposite", opp), ("same", same)):
                d = saved[(donor, v, "clean")][2].astype(np.float64) - bases[v].astype(np.float64)
                n = length(d)
                if not math.isfinite(n) or n <= 0:
                    raise ValueError("invalid donor displacement")
                dirs[v, arm], norms[v + "_" + arm] = d, n
        q = min(2.0, *norms.values())
        cosines = {arm: math.fsum(float(a) * float(b) for a, b in zip(dirs["terminal", arm], dirs["continued", arm])) /
                         (norms["terminal_" + arm] * norms["continued_" + arm]) for arm in ("opposite", "same")}
        random_dir = np.random.Generator(np.random.PCG64(SEED + i)).standard_normal(dimensions)
        result = {"case_id": case["case_id"], "frame": frame, "lemma": lemma, "number": number,
                  "attractor_number": attractor, "requested_q": q, "natural_norms": norms,
                  "direction_cosines": cosines}
        for v in ("terminal", "continued"):
            baseline = saved[(i, v, "clean")][1]
            clean = clean_score(baseline, number, ids)
            outcomes = {}
            arms = ["self", "opposite", "same", "random"]
            if v == "continued":
                arms.append("fixed_terminal_opposite")
            for arm in (*arms, "clean_after"):
                event, value, before, after = saved[(i, v, arm)]
                if bits(before) != bits(bases[v]) or not close(event.get("requested_q"), q):
                    raise ValueError("base state or shared dose differs")
                recorded_norms, recorded_cos = event.get("natural_norms"), event.get("direction_cosines")
                if (type(recorded_norms) is not dict or set(recorded_norms) != set(norms)
                        or type(recorded_cos) is not dict or set(recorded_cos) != set(cosines)
                        or any(not close(recorded_norms[k], x) for k, x in norms.items())
                        or any(not close(recorded_cos[k], x) for k, x in cosines.items())):
                    raise ValueError("donor geometry differs")
                if arm in ("self", "clean_after"):
                    if (bits(before) != bits(after) or bits(value) != bits(baseline)
                            or event.get("random_seed") is not None
                            or not close(event.get("actual_l2"), 0.0)
                            or not close(event.get("relative_direction_error"), 0.0)
                            or (not exact_int(event.get("donor_index"), i) if arm == "self" else event.get("donor_index") is not None)):
                        raise ValueError("self/clean-after control differs")
                    continue
                d = (random_dir if arm == "random" else dirs["terminal", "opposite"]
                     if arm == "fixed_terminal_opposite" else dirs[v, arm])
                donor = None if arm == "random" else same if arm == "same" else opp
                if ((event.get("donor_index") is not None if donor is None else not exact_int(event.get("donor_index"), donor))
                        or (not exact_int(event.get("random_seed"), SEED + i) if arm == "random"
                            else event.get("random_seed") is not None)):
                    raise ValueError("donor identity or paired random seed differs")
                intended = q * d / length(d)
                delta = after.astype(np.float64) - before.astype(np.float64)
                realized = length(delta)
                error = length(delta - intended) / q
                if (not math.isfinite(realized) or realized <= 0 or abs(realized - q) > 1e-4*q
                        or not math.isfinite(error) or error > 1e-4
                        or not close(event.get("actual_l2"), realized)
                        or not close(event.get("relative_direction_error"), error)):
                    raise ValueError("rounded displacement or recorded landing differs")
                # The owner uses NumPy's norm for its single float32 rounding.
                rounded = np.asarray(before.astype(np.float64) + q * d / np.linalg.norm(d), dtype="<f4")
                if bits(after) != bits(rounded):
                    raise ValueError("rounded action does not match intended landing")
                changed = clean_score(value, number, ids)
                sign = -1 if number == 0 else 1
                outcomes[arm] = {"donorward_effect": sign * (changed["is_are_margin"] - clean["is_are_margin"]),
                                 "was_were_margin": changed["was_were_margin"],
                                 "binary_correct": changed["binary_correct"], "binary_tie": changed["binary_tie"],
                                 **distribution(baseline, value)}
            result[v] = {"clean": clean, "arms": outcomes,
                         "opposite_minus_same": outcomes["opposite"]["donorward_effect"] - outcomes["same"]["donorward_effect"],
                         "opposite_minus_random": outcomes["opposite"]["donorward_effect"] - outcomes["random"]["donorward_effect"]}
        t, c = result["terminal"], result["continued"]
        old, new = t["clean"]["p_is_plus_p_are"], c["clean"]["p_is_plus_p_are"]
        result["mass_change"] = None if old is None or new is None else new - old
        result["mass_ratio"], result["mass_ratio_status"] = probability_ratio(new, old)
        result["binary_transition"] = ("correct" if t["clean"]["binary_correct"] else "wrong") + "_to_" + ("correct" if c["clean"]["binary_correct"] else "wrong")
        result["context_recipe_delta_same"] = c["opposite_minus_same"] - t["opposite_minus_same"]
        result["context_recipe_delta_random"] = c["opposite_minus_random"] - t["opposite_minus_random"]
        fixed = c["arms"]["fixed_terminal_opposite"]["donorward_effect"] - t["arms"]["opposite"]["donorward_effect"]
        random = c["arms"]["random"]["donorward_effect"] - t["arms"]["random"]["donorward_effect"]
        result.update(fixed_direction_context_delta=fixed, shared_random_context_delta=random,
                      fixed_minus_random_context_delta=fixed - random)
        rows.append(result)
    require_subnormals()
    return rows


def summarize(rows):
    n = len(rows)
    if n != 32:
        raise ValueError("32 paired outcomes required")
    changes = [r["mass_change"] for r in rows]
    mass_increases = None if any(x is None for x in changes) else sum(x > 0 for x in changes)
    mean_mass = None if mass_increases is None else math.fsum(changes) / n
    out = {"cases": n, "mass_increases": mass_increases, "mean_mass_change": mean_mass,
           "mass_hypothesis_passes": None if mean_mass is None else mean_mass > 0 and mass_increases >= 24}
    for v in ("terminal", "continued"):
        data = [r[v] for r in rows]
        same = math.fsum(x["opposite_minus_same"] for x in data) / n
        random = math.fsum(x["opposite_minus_random"] for x in data) / n
        positive = sum(x["opposite_minus_same"] > 0 and x["opposite_minus_random"] > 0 for x in data)
        correct_count = sum(x["clean"]["binary_correct"] for x in data)
        out[v] = {"correct": correct_count, "baseline_limited": correct_count < 24,
                  "mean_opposite_minus_same": same, "mean_opposite_minus_random": random,
                  "positive_against_both": positive,
                  "donor_hypothesis_passes": same > 0 and random > 0 and positive >= 24}
        for arm in ("opposite", "same", "random") + (("fixed_terminal_opposite",) if v == "continued" else ()):
            out[v][arm] = {"binary_correct_after": sum(x["arms"][arm]["binary_correct"] for x in data),
                           "strict_binary_sign_flips": sum(not x["clean"]["binary_tie"] and not x["arms"][arm]["binary_tie"]
                                                            and x["clean"]["binary_correct"] != x["arms"][arm]["binary_correct"]
                                                            for x in data),
                           "unique_top_changes": sum(x["arms"][arm]["unique_top1_changed"] is True for x in data),
                           "top_change_unresolved_due_to_ties": sum(x["arms"][arm]["unique_top1_changed"] is None for x in data)}
    out["correct_in_both"] = sum(r["terminal"]["clean"]["binary_correct"] and r["continued"]["clean"]["binary_correct"] for r in rows)
    out["continued_accuracy_not_lower"] = out["continued"]["correct"] >= out["terminal"]["correct"]
    out["transitions"] = {k: sum(r["binary_transition"] == k for r in rows) for k in
                          ("correct_to_wrong", "wrong_to_correct", "correct_to_correct", "wrong_to_wrong")}
    out["descriptive_baseline_correct_subsets"] = {}
    for name in ("terminal", "continued", "both"):
        selected = [r for r in rows if (r["terminal"]["clean"]["binary_correct"] if name == "terminal" else
                                        r["continued"]["clean"]["binary_correct"] if name == "continued" else
                                        r["terminal"]["clean"]["binary_correct"] and r["continued"]["clean"]["binary_correct"])]
        summary = {"case_ids": [r["case_id"] for r in selected], "cases": len(selected)}
        for variant in ("terminal", "continued"):
            for control in ("same", "random"):
                key = variant + "_mean_opposite_minus_" + control
                summary[key] = (None if not selected else
                                math.fsum(r[variant]["opposite_minus_" + control] for r in selected) / len(selected))
        out["descriptive_baseline_correct_subsets"][name] = summary
    def group(selected):
        size = len(selected)
        values = [r["mass_change"] for r in selected]
        info = {"cases": size, "mean_mass_change": None if any(x is None for x in values) else math.fsum(values) / size}
        for variant in ("terminal", "continued"):
            info[variant + "_correct"] = sum(r[variant]["clean"]["binary_correct"] for r in selected)
            for control in ("same", "random"):
                info[variant + "_mean_opposite_minus_" + control] = math.fsum(r[variant]["opposite_minus_" + control] for r in selected) / size
        return info
    for factor in ("frame", "lemma"):
        out["by_" + factor] = {str(v): group([r for r in rows if r[factor] == v]) for v in sorted({r[factor] for r in rows})}
    out["dose"] = {"min": min(r["requested_q"] for r in rows), "median": statistics.median(r["requested_q"] for r in rows),
                   "max": max(r["requested_q"] for r in rows)}
    for key in ("context_recipe_delta_same", "context_recipe_delta_random", "fixed_direction_context_delta",
                "shared_random_context_delta", "fixed_minus_random_context_delta"):
        values = [r[key] for r in rows]
        out[key] = {"mean": math.fsum(values) / n, "median": statistics.median(values),
                    "positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values)}
    return out


def source_hashes(git, repo, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not git.is_absolute() or not git.is_file():
        raise ValueError("expected commit or Git executable differs")
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PATH") if k in os.environ}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    def run(*argv):
        return subprocess.check_output([str(git), "-C", str(repo), "--no-replace-objects", *argv], env=env)
    if run("cat-file", "-t", commit).strip() != b"commit":
        raise ValueError("source is not a commit")
    return {name: hashlib.sha256(run("cat-file", "blob", commit + ":" + name)).hexdigest() for name in SOURCE_PATHS}


def audit_files(args):
    attempt = args.attempt
    if sha(args.cases) != CASE_SHA or sha(args.receipt) != args.receipt_sha256:
        raise ValueError("manifest or externally supplied receipt digest differs")
    if ((attempt / "failure.txt").exists() or (args.receipt.parent / "observer-pending.json").exists()
            or sha(attempt / "complete.json") != args.complete_sha256):
        raise ValueError("failed attempt or completion digest differs")
    names = ("metadata.json", "loading.json", "starts.jsonl", "events.jsonl", "memory.jsonl",
             "retention-errors.jsonl", "logits.f32", "witness.f32")
    expected_entries = set(names) | {"complete.json"}
    entries = list(attempt.iterdir())
    if ({p.name for p in entries} != expected_entries
            or any(p.is_symlink() or not p.is_file() for p in entries)
            or sum(p.stat().st_size for p in entries) > 128 * 1024**2):
        raise ValueError("attempt file inventory or output cap differs")
    manifest = strict_json(args.cases.read_bytes())
    meta = strict_json((attempt / "metadata.json").read_bytes())
    completed = strict_json((attempt / "complete.json").read_bytes())
    receipt = strict_json(args.receipt.read_bytes())
    expected_sources = source_hashes(args.git, args.repo, args.source_commit)
    if (meta.get("schema") != "bert-agreement-suffix-attempt-v1" or meta.get("device") != "cpu"
            or type(meta.get("pid")) is not int or meta["pid"] <= 0
            or not exact_int(receipt.get("child_pid"), meta["pid"])
            or meta.get("case_sha256") != CASE_SHA or meta.get("assets_sha256") != ASSETS
            or meta.get("source_commit") != args.source_commit or meta.get("sources_sha256") != expected_sources
            or meta.get("git_sha256") != sha(args.git) or not exact_int(meta.get("seed"), SEED)
            or meta.get("base_interpreter_direct") is not True or meta.get("isolated") is not True
            or meta.get("deterministic_algorithms") is not True or meta.get("tf32") is not False
            or meta.get("cudnn_benchmark") is not False or meta.get("cudnn_deterministic") is not True
            or not exact_int(meta.get("intra_threads"), 1) or not exact_int(meta.get("inter_threads"), 1)
            or "initial_memory" not in meta or meta["initial_memory"] is not None
            or "gpu_name" not in meta or meta["gpu_name"] is not None
            or "cublas_workspace_config" not in meta or meta["cublas_workspace_config"] is not None
            or not finite_number(meta.get("start_unix"))):
        raise ValueError("recorded source, asset, device or numerical policy differs")
    for key in ("python", "torch", "transformers", "numpy"):
        if type(meta.get(key)) is not str or not meta[key]:
            raise ValueError("missing runtime identity " + key)
    if (completed.get("status") != "complete" or not exact_int(completed.get("events"), 416)
            or completed.get("model_weakref_dead") is not True
            or "final_memory" not in completed or completed["final_memory"] is not None
            or not finite_number(completed.get("end_unix"))
            or receipt.get("schema") != "bert-agreement-suffix-direct-child-exit-v1"
            or receipt.get("status") != "complete" or not exact_int(receipt.get("exit_code"), 0)
            or receipt.get("interrupted") is not None or receipt.get("source_commit") != args.source_commit
            or type(receipt.get("controller_name")) is not str or receipt["controller_name"] != args.receipt.parent.name
            or receipt.get("attempt_name") != attempt.name or receipt.get("complete_sha256") != args.complete_sha256
            or receipt.get("metadata_sha256") != sha(attempt / "metadata.json")):
        raise ValueError("completion or direct-child exit identity differs")
    recorded = {}
    for name in names:
        recorded[name] = sha(attempt / name)
        if completed.get(name.replace(".", "_") + "_sha256") != recorded[name]:
            raise ValueError("recorded file digest differs: " + name)
    loading = strict_json((attempt / "loading.json").read_bytes())
    keys = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if type(loading) is not dict or set(loading) != keys or any(type(loading[k]) is not list or loading[k] for k in keys):
        raise ValueError("checkpoint diagnostics do not contain four empty lists")
    if (attempt / "retention-errors.jsonl").stat().st_size != 0:
        raise ValueError("retention-error record exists")
    if (attempt / "logits.f32").stat().st_size != 416 * 30522 * 4 or (attempt / "witness.f32").stat().st_size != 416 * 2 * 768 * 4:
        raise ValueError("selected tensor lengths differ")
    logits = np.memmap(attempt / "logits.f32", dtype="<f4", mode="r", shape=(416, 30522))
    witness = np.memmap(attempt / "witness.f32", dtype="<f4", mode="r", shape=(416, 2, 768))
    rows = audit_semantics(manifest["cases"], jsonl(attempt / "starts.jsonl"), jsonl(attempt / "events.jsonl"),
                           jsonl(attempt / "memory.jsonl"), logits, witness)
    if any(sha(attempt / name) != digest for name, digest in recorded.items()):
        raise ValueError("an input changed while replaying")
    return {"schema": "independent-bert-agreement-suffix-replay-v1", "case_sha256": CASE_SHA,
            "source_commit": args.source_commit, "receipt_sha256": args.receipt_sha256,
            "complete_sha256": args.complete_sha256, "summary": summarize(rows), "cases": rows,
            "scope": "Quiescent saved selected-position tensors and cooperative direct-child receipt; no model execution, loaded-asset, unseen-position or operating-system-wide cleanup attestation."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "attempt", "cases", "receipt", "repo", "git", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt-sha256", "complete-sha256", "source-commit"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--synthetic-fixture", action="store_true", help="label a deliberately fabricated fixture")
    args = p.parse_args()
    result = audit_files(args)
    result["synthetic_fixture"] = bool(args.synthetic_fixture)
    if args.synthetic_fixture:
        result["scope"] = "Fabricated test records only; " + result["scope"]
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as f:
        f.write(raw)
    print("independent_saved_byte_replay", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
