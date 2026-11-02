"""Independent saved-byte replay of the fixed CPU multisuffix study.

Run with a quiescent original attempt and controller. The source/receipt
checks are cooperative recorded evidence, not an execution, process-tree,
chronology or unsaved-position attestation. No model or checkpoint is loaded.
Explicit synthetic mode permits fabricated data only and labels it accordingly.
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
if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
    raise RuntimeError("use isolated Python -I -S -B")
pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--site", required=True, type=Path)
initial, _ = pre.parse_known_args()
if not initial.site.is_absolute() or not initial.site.is_dir() or initial.site.is_symlink():
    raise ValueError("site must be an absolute ordinary directory")
sys.path.append(str(initial.site.resolve()))
import numpy as np

CASE_SHA = "bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e"
ASSETS = {"config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
          "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
          "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
          "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
          "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"}
TARGETS = {"is": 2003, "are": 2024, "was": 2001, "were": 2020}
VARIANTS = ("terminal", "period", "present", "outside", "awake")
SEED = 20261029
SOURCE_PATHS = ("scripts/bert_multisuffix_agreement_v1.py", "scripts/read_bert_multisuffix_agreement_v1.py",
                "scripts/observe_bert_multisuffix_agreement_v1.py", "scripts/cross_model_agreement_v1.py",
                "scripts/read_cross_model_agreement_v1.py", "docs/BERT_MULTISUFFIX_AGREEMENT_V1.md",
                "docs/BERT_MULTISUFFIX_EXECUTION_V1.md", "experiments/bert_multisuffix_agreement_cases_v1.json")
FILES = ("metadata.json", "loading.json", "starts.jsonl", "events.jsonl", "memory.jsonl",
         "retention-errors.jsonl", "logits.f32", "witness.f32", "plan.json", "complete.json")


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

def norm(vector):
    return math.sqrt(math.fsum(float(x) * float(x) for x in vector))

def donors(cases):
    if len(cases) != 32:
        raise ValueError("32 base cases are required")
    lookup = {}
    for i, c in enumerate(cases):
        if type(c.get("index")) is not int or c["index"] != i:
            raise ValueError("case index/order differs")
        fields = tuple(c.get(k) for k in ("frame", "attractor_number", "lemma", "number"))
        if any(type(x) is not int for x in fields) or fields in lookup:
            raise ValueError("case factors invalid or repeated")
        f, a, lemma, number = fields
        if f not in (0, 1) or a not in (0, 1) or lemma not in range(4) or number not in (0, 1):
            raise ValueError("case factor out of range")
        lookup[fields] = i
    result = []
    for c in cases:
        f, a, lemma, number = (c[k] for k in ("frame", "attractor_number", "lemma", "number"))
        paired = lemma ^ 1
        same = lookup[(f, a, paired, number)]
        opposite = lookup[(f, a, paired, 1 - number)]
        if (type(c.get("same_donor_index")) is not int or c["same_donor_index"] != same
                or type(c.get("opposite_donor_index")) is not int or c["opposite_donor_index"] != opposite):
            raise ValueError("fixed other-lemma donor identity differs")
        result.append((opposite, same))
    return result

def plan_all(cases, clean_vectors):
    clean = np.asarray(clean_vectors)
    if (clean.dtype != np.float32 or clean.ndim != 3 or clean.shape[:2] != (32, 5)
            or clean.shape[2] < 1 or not np.isfinite(clean).all()):
        raise ValueError("clean state tensor differs or is nonfinite")
    paired = donors(cases)
    plans = []
    for i, (opposite, same) in enumerate(paired):
        vectors, norms, cosines = {}, {}, {}
        for vi, variant in enumerate(VARIANTS):
            for arm, donor in (("opposite", opposite), ("same", same)):
                d = clean[donor, vi].astype(np.float64) - clean[i, vi].astype(np.float64)
                n = float(np.linalg.norm(d))
                independently_summed = norm(d)
                if (not math.isfinite(n) or not math.isfinite(independently_summed)
                        or not math.isclose(n, independently_summed, rel_tol=1e-12, abs_tol=0.0)):
                    raise ValueError("nonfinite donor displacement")
                vectors[variant, arm], norms[f"{variant}_{arm}"] = d, n
        for variant in VARIANTS:
            for arm in ("opposite", "same"):
                left, right = vectors["terminal", arm], vectors[variant, arm]
                a, b = norms[f"terminal_{arm}"], norms[f"{variant}_{arm}"]
                cosines[f"{variant}_{arm}"] = (None if a == 0 or b == 0 else
                    math.fsum(float(x) * float(y) for x, y in zip(left, right)) / (a * b))
        q = min(2.0, *norms.values())
        result = {"index": i, "q": q, "natural_norms": norms,
                  "direction_cosines": cosines, "random_seed": SEED + i,
                  "eligible": q > 0, "failure": None, "landings": {}}
        if q == 0:
            result["failure"] = {"kind": "zero_dose"}
            plans.append(result)
            continue
        random_direction = np.random.Generator(np.random.PCG64(SEED + i)).standard_normal(clean.shape[2])
        for vi, variant in enumerate(VARIANTS):
            actions = (("opposite", vectors[variant, "opposite"]),
                       ("same", vectors[variant, "same"]), ("random", random_direction))
            if variant != "terminal":
                actions += (("fixed_terminal_opposite", vectors["terminal", "opposite"]),)
            for arm, direction in actions:
                direction_norm = float(np.linalg.norm(direction))
                if not math.isfinite(direction_norm) or direction_norm <= 0:
                    raise ValueError("invalid active direction")
                base = clean[i, vi].astype(np.float64)
                intended = q * (direction / direction_norm)
                landed = np.asarray(base + intended, dtype=np.float32)
                if not np.isfinite(landed).all():
                    raise ValueError("nonfinite rounded landing")
                actual_delta = landed.astype(np.float64) - base
                realized = norm(actual_delta)
                error = norm(actual_delta - intended) / q
                failed = (not math.isfinite(realized) or realized <= 0 or abs(realized - q) > 1e-4 * q
                          or not math.isfinite(error) or error > 1e-4)
                result["landings"][variant, arm] = {"value": landed, "actual_l2": realized,
                                                     "relative_direction_error": error, "valid": not failed}
                if failed and result["failure"] is None:
                    result["eligible"] = False
                    result["failure"] = {"kind": "rounded_landing", "variant": variant, "arm": arm}
        plans.append(result)
    return plans

def score_all(cases, clean_logits, plans, changed, *, ids):
    logits = np.asarray(clean_logits)
    if (len(cases) != 32 or len(plans) != 32 or logits.dtype != np.float32 or logits.ndim != 3
            or logits.shape[:2] != (32, 5) or len(set(ids.values())) != 4
            or max(ids.values()) >= logits.shape[2] or min(ids.values()) < 0
            or not np.isfinite(logits).all()):
        raise ValueError("invalid clean logits or target IDs")
    rows = []
    expected_keys = set()
    for i, case in enumerate(cases):
        plan = plans[i]
        if (type(plan.get("eligible")) is not bool or type(plan.get("index")) is not int
                or plan["index"] != i):
            raise ValueError("invalid plan order or eligibility")
        row = {"case_id": case["case_id"], "frame": case["frame"], "lemma": case["lemma"],
               "number": case["number"], "eligible": plan["eligible"], "variants": {}}
        for vi, variant in enumerate(VARIANTS):
            clean = clean_score(logits[i, vi], case["number"], ids)
            outcome = {"clean": clean, "arms": {}}
            if plan["eligible"]:
                arms = ("opposite", "same", "random") + (("fixed_terminal_opposite",) if vi else ())
                for arm in arms:
                    key = (i, variant, arm)
                    expected_keys.add(key)
                    if key not in changed:
                        raise ValueError("missing eligible intervention")
                    new = np.asarray(changed[key])
                    if new.dtype != np.float32 or new.shape != (logits.shape[2],) or not np.isfinite(new).all():
                        raise ValueError("invalid intervention logits")
                    scored = clean_score(new, case["number"], ids)
                    sign = -1 if case["number"] == 0 else 1
                    outcome["arms"][arm] = {"donorward_effect": sign * (scored["is_are_margin"] - clean["is_are_margin"]),
                        "binary_correct": scored["binary_correct"], "binary_tie": scored["binary_tie"],
                        "was_were_margin": scored["was_were_margin"], **distribution(logits[i, vi], new)}
                outcome["opposite_minus_same"] = (outcome["arms"]["opposite"]["donorward_effect"]
                                                    - outcome["arms"]["same"]["donorward_effect"])
                outcome["opposite_minus_random"] = (outcome["arms"]["opposite"]["donorward_effect"]
                                                      - outcome["arms"]["random"]["donorward_effect"])
            row["variants"][variant] = outcome
        baseline = row["variants"]["period"]["clean"]["p_is_plus_p_are"]
        row["suffix_vs_period"] = {}
        for variant in ("present", "outside", "awake"):
            value = row["variants"][variant]["clean"]["p_is_plus_p_are"]
            ratio, reason = probability_ratio(value, baseline)
            row["suffix_vs_period"][variant] = {"difference": None if value is None or baseline is None else value - baseline,
                                                   "ratio": ratio, "ratio_status": reason}
        row["context_comparisons"] = {}
        for variant in VARIANTS[1:]:
            for reference in (("terminal",) if variant == "period" else ("terminal", "period")):
                old = row["variants"][reference]["clean"]
                new = row["variants"][variant]["clean"]
                old_mass, new_mass = old["p_is_plus_p_are"], new["p_is_plus_p_are"]
                ratio, reason = probability_ratio(new_mass, old_mass)
                key = variant + "_minus_" + reference
                row["context_comparisons"][key] = {
                    "mass_difference": None if old_mass is None or new_mass is None else new_mass - old_mass,
                    "mass_ratio": ratio, "mass_ratio_status": reason,
                    "unique_top_changed": (old["top_id"] != new["top_id"]
                                           if old["top_ties"] == new["top_ties"] == 1 else None)}
        if plan["eligible"]:
            terminal = row["variants"]["terminal"]
            row["context_interventions"] = {}
            for variant in VARIANTS[1:]:
                current = row["variants"][variant]
                fixed = (current["arms"]["fixed_terminal_opposite"]["donorward_effect"]
                         - terminal["arms"]["opposite"]["donorward_effect"])
                random = current["arms"]["random"]["donorward_effect"] - terminal["arms"]["random"]["donorward_effect"]
                row["context_interventions"][variant] = {
                    "fixed_direction_delta": fixed, "shared_random_delta": random,
                    "fixed_minus_random_delta": fixed - random,
                    "recipe_delta_same": current["opposite_minus_same"] - terminal["opposite_minus_same"],
                    "recipe_delta_random": current["opposite_minus_random"] - terminal["opposite_minus_random"]}
        else:
            row["context_interventions"] = None
        rows.append(row)
    if set(changed) != expected_keys:
        raise ValueError("extra intervention or intervention for ineligible base")
    return rows

def summarize(rows):
    if len(rows) != 32 or any(type(r["eligible"]) is not bool for r in rows):
        raise ValueError("32 rows and explicit eligibility are required")
    groups = {(frame, lemma): [r for r in rows if r["frame"] == frame and r["lemma"] == lemma]
              for frame in (0, 1) for lemma in range(4)}
    if any(len(group) != 4 for group in groups.values()):
        raise ValueError("frame/lemma group structure differs")
    output = {"cases": 32, "eligible": sum(r["eligible"] for r in rows), "suffix_vs_period": {}, "variants": {}}
    for variant in ("present", "outside", "awake"):
        differences = [r["suffix_vs_period"][variant]["difference"] for r in rows]
        if any(x is None for x in differences):
            output["suffix_vs_period"][variant] = {"criterion": None, "reason": "probability_unresolved"}
            continue
        group_means = {f"{f}:{l}": math.fsum(r["suffix_vs_period"][variant]["difference"] for r in group) / 4
                       for (f, l), group in groups.items()}
        mean = math.fsum(differences) / 32
        positive = sum(x > 0 for x in differences)
        positive_groups = sum(x > 0 for x in group_means.values())
        output["suffix_vs_period"][variant] = {"mean": mean, "positive_cases": positive,
            "group_means": group_means, "positive_groups": positive_groups,
            "criterion": mean > 0 and positive >= 24 and positive_groups >= 6}
    for variant in VARIANTS:
        clean = [r["variants"][variant]["clean"] for r in rows]
        eligible = [r["variants"][variant] for r in rows if r["eligible"]]
        correct = sum(c["binary_correct"] for c in clean)
        same = None if not eligible else math.fsum(x["opposite_minus_same"] for x in eligible) / len(eligible)
        random = None if not eligible else math.fsum(x["opposite_minus_random"] for x in eligible) / len(eligible)
        positive = sum(x["opposite_minus_same"] > 0 and x["opposite_minus_random"] > 0 for x in eligible)
        criterion = (None if len(eligible) != 32 else same > 0 and random > 0 and positive >= 24)
        output["variants"][variant] = {"clean_correct": correct, "clean_ties": sum(c["binary_tie"] for c in clean),
            "baseline_limited": correct < 24, "eligible_count": len(eligible),
            "eligible_mean_opposite_minus_same": same, "eligible_mean_opposite_minus_random": random,
            "eligible_positive_against_both": positive, "full_set_donor_criterion": criterion}
        arm_summary = {}
        for arm in ("opposite", "same", "random") + (("fixed_terminal_opposite",) if variant != "terminal" else ()):
            pairs = [(r["variants"][variant]["clean"], r["variants"][variant]["arms"][arm])
                     for r in rows if r["eligible"]]
            arm_summary[arm] = {"eligible": len(pairs),
                "strict_binary_sign_flips": sum(not a["binary_tie"] and not b["binary_tie"]
                                                and a["binary_correct"] != b["binary_correct"] for a, b in pairs),
                "binary_tie_unresolved": sum(a["binary_tie"] or b["binary_tie"] for a, b in pairs),
                "unique_top_changes": sum(b["unique_top1_changed"] is True for _, b in pairs),
                "top_tie_unresolved": sum(b["unique_top1_changed"] is None for _, b in pairs),
                "distribution_unresolved": sum(b["numerically_unresolved"] for _, b in pairs)}
        output["variants"][variant]["arms"] = arm_summary
    output["by_factor"] = {}
    for factor, values in (("frame", (0, 1)), ("lemma", range(4))):
        output["by_factor"][factor] = {}
        for value in values:
            selected = [r for r in rows if r[factor] == value]
            info = {"cases": len(selected), "variants": {}}
            for variant in VARIANTS:
                clean = [r["variants"][variant]["clean"] for r in selected]
                masses = [x["p_is_plus_p_are"] for x in clean]
                info["variants"][variant] = {"correct": sum(x["binary_correct"] for x in clean),
                    "ties": sum(x["binary_tie"] for x in clean),
                    "mean_pair_mass": None if any(x is None for x in masses) else math.fsum(masses) / len(masses)}
            output["by_factor"][factor][str(value)] = info
    subsets = {variant: [r for r in rows if r["variants"][variant]["clean"]["binary_correct"]] for variant in VARIANTS}
    subsets["all_variants"] = [r for r in rows if all(r["variants"][v]["clean"]["binary_correct"] for v in VARIANTS)]
    output["descriptive_correct_subsets"] = {}
    for name, selected in subsets.items():
        info = {"case_ids": [r["case_id"] for r in selected], "cases": len(selected), "variants": {}}
        for variant in VARIANTS:
            eligible = [r["variants"][variant] for r in selected if r["eligible"]]
            info["variants"][variant] = {"eligible": len(eligible),
                "mean_opposite_minus_same": None if not eligible else math.fsum(x["opposite_minus_same"] for x in eligible) / len(eligible),
                "mean_opposite_minus_random": None if not eligible else math.fsum(x["opposite_minus_random"] for x in eligible) / len(eligible)}
        output["descriptive_correct_subsets"][name] = info
    output["context_interventions"] = {}
    for variant in VARIANTS[1:]:
        records = [r["context_interventions"][variant] for r in rows if r["eligible"]]
        entry = {"eligible": len(records)}
        entry.update({key: None if not records else math.fsum(r[key] for r in records) / len(records)
                      for key in ("fixed_direction_delta", "shared_random_delta", "fixed_minus_random_delta",
                                  "recipe_delta_same", "recipe_delta_random")})
        output["context_interventions"][variant] = entry
    return output

def check(condition, reason):
    if not condition:
        raise ValueError(reason)

def geometry_close(actual, expected):
    """Relative comparison that does not erase tiny nonzero doses or norms."""
    return (finite_number(actual) and finite_number(expected)
            and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=0.0))

def ordered(cases, plans):
    out = [(case, variant, "clean") for case in cases for variant in VARIANTS]
    for case, plan in zip(cases, plans):
        if not plan["eligible"]:
            continue
        for variant in VARIANTS:
            arms = ["self", "opposite", "same", "random"]
            if variant != "terminal":
                arms.append("fixed_terminal_opposite")
            out.extend((case, variant, arm) for arm in (*arms, "clean_after"))
    return out

def gpu_sample(value):
    keys = {"cuda_free", "cuda_total", "allocated", "reserved", "peak_allocated", "peak_reserved"}
    check(type(value) is dict and set(value) == keys and
          all(type(value[k]) is int and value[k] >= 0 for k in keys), "invalid GPU memory sample")
    check(value["cuda_total"] > 0 and value["cuda_free"] <= value["cuda_total"] and
          value["allocated"] <= value["reserved"] and value["allocated"] <= value["peak_allocated"] and
          value["reserved"] <= value["peak_reserved"] and value["peak_allocated"] <= value["peak_reserved"],
          "inconsistent GPU memory sample")
    return value["cuda_total"]

def audit_payload(root, cases_path, *, device):
    check(device in ("cpu", "cuda"), "unsupported recorded device")
    root = Path(root)
    check(sha(cases_path) == CASE_SHA, "fixed case bytes differ")
    entries = list(root.iterdir())
    check({x.name for x in entries} == set(FILES) and all(x.is_file() and not x.is_symlink() for x in entries)
          and sum(x.stat().st_size for x in entries) <= 192 * 1024**2, "inventory or cap differs")
    source_hash = {name: sha(root / name) for name in FILES}
    manifest = strict_json(Path(cases_path).read_bytes())
    cases = manifest["cases"]
    check(manifest.get("schema") == "bert-multisuffix-agreement-cases-v1"
          and manifest.get("variants") == list(VARIANTS), "manifest schema differs")
    donors(cases)
    raw_plan = strict_json((root / "plan.json").read_bytes())
    check(type(raw_plan) is dict and set(raw_plan) == {"schema", "cases"} and
          raw_plan["schema"] == "bert-multisuffix-dose-plan-v1" and
          type(raw_plan["cases"]) is list and len(raw_plan["cases"]) == 32, "plan structure differs")
    starts, events, memory = [jsonl(root / name) for name in ("starts.jsonl", "events.jsonl", "memory.jsonl")]
    count = len(starts)
    check(160 <= count <= 1088 and len(events) == len(memory) == count, "journal counts differ")
    check((root / "retention-errors.jsonl").stat().st_size == 0, "retention failure present")
    check((root / "logits.f32").stat().st_size == count * 30522 * 4 and
          (root / "witness.f32").stat().st_size == count * 2 * 768 * 4, "tensor lengths differ")
    logits = np.memmap(root / "logits.f32", dtype="<f4", mode="r", shape=(count, 30522))
    witness = np.memmap(root / "witness.f32", dtype="<f4", mode="r", shape=(count, 2, 768))
    require_subnormals()
    clean_vectors = np.asarray(witness[:160, 0]).reshape(32, 5, 768)
    plans = plan_all(cases, clean_vectors)
    for i, (case, planned, recomputed) in enumerate(zip(cases, raw_plan["cases"], plans)):
        keys = {"case_id", "case_index", "eligible", "reason", "requested_q", "natural_norms",
                "direction_cosines", "random_seed"}
        check(type(planned) is dict and set(planned) == keys and planned["case_id"] == case["case_id"]
              and exact_int(planned["case_index"], i) and type(planned["eligible"]) is bool
              and planned["eligible"] == recomputed["eligible"] and geometry_close(planned["requested_q"], recomputed["q"])
              and exact_int(planned["random_seed"], SEED + i), "plan identity/dose differs")
        failure = recomputed["failure"]
        reason = (None if failure is None else "zero_donor_distance" if failure["kind"] == "zero_dose" else
                  "rounded_landing:" + failure["variant"] + ":" + failure["arm"])
        check(planned["reason"] == reason, "ineligible reason differs")
        norms = planned["natural_norms"]
        check(type(norms) is dict and set(norms) == set(recomputed["natural_norms"]) and
              all(geometry_close(norms[k], v) for k, v in recomputed["natural_norms"].items()), "donor norms differ")
        cosine = planned["direction_cosines"]
        check(type(cosine) is dict and set(cosine) == set(VARIANTS[1:]), "cosine variants differ")
        for variant in VARIANTS[1:]:
            check(type(cosine[variant]) is dict and set(cosine[variant]) == {"opposite", "same"}, "cosine arms differ")
            for arm in ("opposite", "same"):
                expected = recomputed["direction_cosines"][f"{variant}_{arm}"]
                actual = cosine[variant][arm]
                check(actual is None if expected is None else close(actual, expected), "direction cosine differs")
    schedule = ordered(cases, plans)
    check(len(schedule) == count == 160 + 29 * sum(p["eligible"] for p in plans), "eligible schedule differs")
    changed, gpu_total = {}, None
    for i, (case, variant, arm) in enumerate(schedule):
        s, e, sample, row = starts[i], events[i], memory[i], case["variants"][variant]
        identity = {"event_index", "case_id", "case_index", "variant", "arm"}
        start_keys = identity | {"input_ids", "attention_mask", "subject_position", "output_position"}
        event_keys = identity | {"hook_count", "full_logits_finite", "returned_status", "logits_offset", "witness_offset"}
        if arm != "clean":
            event_keys |= {"donor_index", "requested_q", "actual_l2", "relative_direction_error", "random_seed"}
        check(type(s) is dict and set(s) == start_keys and type(e) is dict and set(e) == event_keys,
              "forward record fields differ")
        for item in (s, e):
            check(type(item) is dict and exact_int(item.get("event_index"), i)
                  and exact_int(item.get("case_index"), case["index"])
                  and item.get("case_id") == case["case_id"] and item.get("variant") == variant
                  and item.get("arm") == arm, "action identity/order differs")
        for key in ("input_ids", "attention_mask"):
            check(type(s.get(key)) is list and all(type(x) is int for x in s[key]) and s[key] == row[key],
                  "input IDs or mask differ")
        check(exact_int(s.get("subject_position"), row["subject_position"])
              and exact_int(s.get("output_position"), row["output_position"])
              and exact_int(e.get("hook_count"), 1) and e.get("returned_status") == "returned"
              and e.get("full_logits_finite") is True and exact_int(e.get("logits_offset"), i * 30522 * 4)
              and exact_int(e.get("witness_offset"), i * 2 * 768 * 4), "forward metadata differs")
        check(type(sample) is dict and set(sample) == {"event_index", "memory"}
              and exact_int(sample["event_index"], i), "memory sample order differs")
        if device == "cpu":
            check(sample["memory"] is None, "CPU memory sample differs")
        else:
            total = gpu_sample(sample["memory"])
            if gpu_total is None:
                gpu_total = total
            check(total == gpu_total, "GPU total differs within journal")
        before, after = witness[i]
        check(np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(logits[i]).all(),
              "nonfinite selected tensor")
        ci, vi = case["index"], VARIANTS.index(variant)
        clean = witness[ci * 5 + vi]
        check(bits(before) == bits(clean[0]), "pre-action state differs from clean")
        if arm == "clean":
            check(bits(before) == bits(after), "clean witness differs")
            continue
        plan = plans[ci]
        check(geometry_close(e.get("requested_q"), plan["q"]), "action dose differs")
        if arm in ("self", "clean_after"):
            check(bits(before) == bits(after) and bits(logits[i]) == bits(logits[ci * 5 + vi])
                  and e.get("random_seed") is None and geometry_close(e.get("actual_l2"), 0.0)
                  and geometry_close(e.get("relative_direction_error"), 0.0)
                  and (exact_int(e.get("donor_index"), ci) if arm == "self" else e.get("donor_index") is None),
                  "self or clean-after control differs")
            continue
        predicted = plan["landings"][variant, arm]
        donor = None if arm == "random" else case["same_donor_index"] if arm == "same" else case["opposite_donor_index"]
        check((e.get("donor_index") is None if donor is None else exact_int(e.get("donor_index"), donor))
              and (exact_int(e.get("random_seed"), SEED + ci) if arm == "random" else e.get("random_seed") is None)
              and bits(after) == bits(predicted["value"])
              and geometry_close(e.get("actual_l2"), predicted["actual_l2"])
              and close(e.get("relative_direction_error"), predicted["relative_direction_error"]),
              "donor or rounded action differs")
        changed[ci, variant, arm] = logits[i]
    rows = score_all(cases, np.asarray(logits[:160]).reshape(32, 5, 30522), plans, changed,
                            ids=TARGETS)
    summary = summarize(rows)
    for row, case, recorded, plan in zip(rows, cases, raw_plan["cases"], plans):
        row.update(attractor_number=case["attractor_number"], requested_q=plan["q"],
                   same_donor_index=case["same_donor_index"], opposite_donor_index=case["opposite_donor_index"],
                   ineligible_reason=recorded["reason"], natural_norms=plan["natural_norms"],
                   direction_cosines={variant: {arm: plan["direction_cosines"][variant + "_" + arm]
                                               for arm in ("opposite", "same")}
                                      for variant in VARIANTS[1:]})
    doses = [p["q"] for p in plans]
    summary["dose"] = {"min": min(doses), "median": float(np.median(doses)), "max": max(doses)}
    require_subnormals()
    check(all(sha(root / name) == digest for name, digest in source_hash.items()), "input changed during replay")
    return {"schema": "independent-multisuffix-synthetic-payload-v1", "synthetic_unattested": True,
            "events": count, "eligible_cases": sum(p["eligible"] for p in plans),
            "case_sha256": CASE_SHA, "file_sha256": source_hash,
            "summary": summary, "cases": rows,
            "scope": "Fabricated quiescent saved tensors only; no owner receipt, process, source, assets or execution attestation."}

def audit_envelope(args):
    if (args.receipt is None or args.repo is None or args.git is None or args.receipt_sha256 is None
            or args.complete_sha256 is None or args.source_commit is None):
        raise ValueError("receipt, repo, Git, source commit and expected digests are required")
    if (not re.fullmatch("[0-9a-f]{64}", args.receipt_sha256)
            or not re.fullmatch("[0-9a-f]{64}", args.complete_sha256)):
        raise ValueError("expected digest syntax differs")
    if (not args.git.is_absolute() or not args.git.is_file() or args.git.is_symlink()
            or args.git.resolve().is_relative_to(args.repo.resolve())):
        raise ValueError("Git executable must be an external absolute ordinary file")
    if (not args.receipt.is_file() or args.receipt.is_symlink()
            or (args.receipt.parent / "observer-pending.json").exists()
            or (args.attempt / "failure.txt").exists()
            or sha(args.receipt) != args.receipt_sha256
            or sha(args.attempt / "complete.json") != args.complete_sha256):
        raise ValueError("receipt, pending marker, failure or completion digest differs")
    receipt = strict_json(args.receipt.read_bytes())
    meta = strict_json((args.attempt / "metadata.json").read_bytes())
    complete = strict_json((args.attempt / "complete.json").read_bytes())
    source = source_hashes(args.git, args.repo, args.source_commit)
    if sha(Path(__file__)) != source["scripts/read_bert_multisuffix_agreement_v1.py"]:
        raise ValueError("executing reader bytes differ from committed reader")
    if (type(meta) is not dict or meta.get("schema") != "bert-multisuffix-agreement-attempt-v1"
            or meta.get("device") != "cpu" or type(meta.get("pid")) is not int or meta["pid"] <= 0
            or meta.get("case_sha256") != CASE_SHA or meta.get("assets_sha256") != ASSETS
            or meta.get("source_commit") != args.source_commit or meta.get("sources_sha256") != source
            or meta.get("git_sha256") != sha(args.git) or not exact_int(meta.get("seed"), SEED)
            or meta.get("base_interpreter_direct") is not True or meta.get("isolated") is not True
            or meta.get("deterministic_algorithms") is not True or meta.get("tf32") is not False
            or meta.get("cudnn_benchmark") is not False or meta.get("cudnn_deterministic") is not True
            or not exact_int(meta.get("intra_threads"), 1) or not exact_int(meta.get("inter_threads"), 1)
            or "initial_memory" not in meta or meta["initial_memory"] is not None
            or "gpu_name" not in meta or meta["gpu_name"] is not None
            or "cublas_workspace_config" not in meta or meta["cublas_workspace_config"] is not None
            or not finite_number(meta.get("start_unix"))):
        raise ValueError("recorded CPU runtime, source, asset or numerical policy differs")
    for key in ("python", "torch", "transformers", "numpy"):
        if type(meta.get(key)) is not str or not meta[key]:
            raise ValueError("missing recorded runtime identity")
    if (type(receipt) is not dict or receipt.get("schema") != "bert-multisuffix-direct-child-exit-v1"
            or receipt.get("status") != "complete" or not exact_int(receipt.get("exit_code"), 0)
            or not exact_int(receipt.get("child_pid"), meta["pid"]) or receipt.get("interrupted") is not None
            or receipt.get("source_commit") != args.source_commit or receipt.get("attempt_name") != args.attempt.name
            or type(receipt.get("controller_name")) is not str or receipt["controller_name"] != args.receipt.parent.name
            or receipt.get("complete_sha256") != args.complete_sha256
            or receipt.get("metadata_sha256") != sha(args.attempt / "metadata.json")
            or not finite_number(receipt.get("start_unix")) or not finite_number(receipt.get("end_unix"))):
        raise ValueError("direct-child receipt differs")
    if (type(complete) is not dict or complete.get("status") != "complete"
            or type(complete.get("events")) is not int or type(complete.get("eligible_cases")) is not int
            or not 0 <= complete["eligible_cases"] <= 32
            or complete["events"] != 160 + 29 * complete["eligible_cases"]
            or complete.get("model_weakref_dead") is not True or "final_memory" not in complete
            or complete["final_memory"] is not None or not finite_number(complete.get("end_unix"))):
        raise ValueError("completion status or counts differ")
    for name in FILES:
        if name != "complete.json" and complete.get(name.replace(".", "_") + "_sha256") != sha(args.attempt / name):
            raise ValueError("completed file hash differs: " + name)
    loading = strict_json((args.attempt / "loading.json").read_bytes())
    keys = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if type(loading) is not dict or set(loading) != keys or any(type(loading[k]) is not list or loading[k] for k in keys):
        raise ValueError("checkpoint diagnostics differ")
    return complete, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "attempt", "cases", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt", "repo", "git"):
        parser.add_argument("--" + name, type=Path)
    for name in ("receipt-sha256", "complete-sha256", "source-commit"):
        parser.add_argument("--" + name)
    parser.add_argument("--synthetic-unattested", action="store_true")
    args = parser.parse_args()
    if args.synthetic_unattested:
        report = audit_payload(args.attempt, args.cases, device="cpu")
    else:
        complete, source = audit_envelope(args)
        report = audit_payload(args.attempt, args.cases, device="cpu")
        if (not exact_int(complete["events"], report["events"])
                or not exact_int(complete["eligible_cases"], report["eligible_cases"])
                or sha(args.receipt) != args.receipt_sha256
                or sha(args.attempt / "complete.json") != args.complete_sha256
                or sha(args.cases) != CASE_SHA or sha(args.git) != strict_json((args.attempt / "metadata.json").read_bytes())["git_sha256"]
                or sha(Path(__file__)) != source["scripts/read_bert_multisuffix_agreement_v1.py"]):
            raise ValueError("recorded counts or input bytes changed during replay")
        report.update(schema="independent-multisuffix-cpu-replay-v1", synthetic_unattested=False,
                      source_commit=args.source_commit, sources_sha256=source,
                      receipt_sha256=args.receipt_sha256, complete_sha256=args.complete_sha256,
                      scope="Quiescent recorded CPU tensors and cooperative original direct-child receipt; no execution, source chronology, process-tree or unsaved-position attestation.")
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short replay report write")
    print("multisuffix_saved_byte_replay", hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
