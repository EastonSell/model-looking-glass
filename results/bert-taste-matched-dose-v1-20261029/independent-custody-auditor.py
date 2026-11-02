"""Independently inspect one closed public matched-dose taste run without a model.

The parent/child, source and full-output observations are saved attestations.
Inputs must be quiescent. Patched full-vocabulary logits were not retained.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys

FRAMES = ("coffee", "milk", "soup", "cake", "pudding", "cocoa")
MODES = (("ingredient_donor", 7, False), ("ingredient_self", 7, True),
         ("control_self", 1, True), ("ingredient_matched", 7, False), ("control_matched", 1, False))
SOURCE_COMMIT = "74227e9909e4a485296065aced0368a1ccafba64"
LOCAL_EQUIVALENT = "5744734af59fa0d1022d90517ae43139fb749691"
CASE_SHA = "983be330b6691ce23986f26e50a345f77ca5cf46dd28a660a3a012a2cf598a7c"
RESULT_SHA = "cad47fbb5599f5ec44161d59ddbfc6d26bd904b399f23c68089b709921ec194e"
OBSERVER_SHA = "64553741dfe735f482ec0541ea1aa1b85195236f0edd8ddd6be7449b5012a760"
EXIT_SHA = "2913352a8e8d7427b5ecd135ce61d05491de5d085353adff2ab07ce7cdf4846d"
POST_EXIT_SHA = "953efa50564aae45d939a6dcd71982443f3abc84d3129801c1dc1210032c3d2c"
EXIT_PROJECTION_SHA = "85e0e0af8b2928ed27130b48ebb9cbdc16ca2d25aab8d6810ad12869fb29dbfc"
START_PROJECTION_SHA = "809b1a5679bd0c723490a611f487e53fa36264852a7577cfc27b0ab98c4a3404"
START_ORIGINAL_SHA = "a38976dcfdb8cc131a0403db4eafdec701bd18538835b8cb58c3747aa59436ff"
EXPECTED_ASSETS = {
    "config.json": {"sha256": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83", "size": 483},
    "model.safetensors": {"sha256": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063", "size": 267954768},
    "tokenizer.json": {"sha256": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98", "size": 466062},
    "tokenizer_config.json": {"sha256": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10", "size": 48},
    "vocab.txt": {"sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", "size": 231508},
}
SOURCE_PATHS = (
    "scripts/bert_taste_matched_dose_v1.py", "scripts/bert_taste_matched_parent_v1.py",
    "scripts/bert_taste_matched_independent.py", "docs/BERT_TASTE_MATCHED_DOSE_V1.md",
    "experiments/bert_taste_matched_dose_public_cases_v1.json", "experiments/bert_agreement_exposures_v6.json",
    "experiments/bert_matched_public_cases_v1.json", "experiments/bert_taste_public_exposures_v2.json",
    "scripts/bert_taste_patch_v1.py", "looking_glass/__init__.py", "looking_glass/pretrained.py",
    "looking_glass/bert_agreement_study.py", "looking_glass/bert_agreement_runner.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py", "looking_glass/model_ownership.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_exit_fields(record):
    require(type(record) is dict and record.get("status") == "observed-zero-exit"
            and type(record.get("exit_code")) is int and record["exit_code"] == 0
            and record.get("parent_prefix_empty") is True and record.get("source_commit") == SOURCE_COMMIT
            and record.get("observer_sha256") == OBSERVER_SHA and record.get("result_sha256") == RESULT_SHA,
            "parent exit observation differs")


def verify_private_exit(record):
    _verify_exit_fields(record)
    require(type(record.get("process_id")) is int and record["process_id"] > 0,
            "parent process identity is invalid")


def verify_public_exit_projection(record):
    _verify_exit_fields(record)
    require("process_id" not in record and record.get("original_sha256") == EXIT_SHA,
            "public exit projection differs")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path, maximum):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        info = component.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
                "redirected evidence path")
    require(stat.S_ISREG(path.lstat().st_mode), "evidence is not an ordinary file")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "evidence exceeds its bound")
    return raw


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)


def float32(bits):
    require(type(bits) is int and 0 <= bits <= 0xffffffff, "invalid binary32 word")
    sign, exp, fraction = bits >> 31, (bits >> 23) & 255, bits & 0x7fffff
    require(exp != 255, "nonfinite binary32 value")
    magnitude = math.ldexp(fraction, -149) if exp == 0 else math.ldexp((1 << 23) | fraction, exp - 150)
    return -magnitude if sign else magnitude


def selected_margin(summary):
    require(type(summary) is dict, "missing selected logits")
    sweet, sour = float32(summary["sweet_bits"]), float32(summary["sour_bits"])
    for key, expected in (("sweet_logit", sweet), ("sour_logit", sour)):
        actual = summary[key]
        require(type(actual) in (int, float) and math.isfinite(actual)
                and struct.pack("<d", float(actual)) == struct.pack("<d", expected), "selected logit differs from bits")
    margin = math.fsum((sweet, -sour))
    require(type(summary["sweet_minus_sour"]) in (int, float) and summary["sweet_minus_sour"] == margin,
            "selected margin differs")
    return margin


def direction_summary(differences, eligible, *, sign):
    require(type(differences) is dict and set(differences) == set(eligible) and sign in (-1, 1),
            "invalid directional inputs")
    values = {}
    for name, (ingredient, control) in differences.items():
        require(type(eligible[name]) is bool and math.isfinite(ingredient) and math.isfinite(control),
                "invalid directional value")
        values[name] = math.fsum((sign * ingredient, -sign * control))
    selected = [values[name] for name in differences if eligible[name]]
    return {**values, "n": len(selected), "mean": math.fsum(selected) / len(selected) if selected else None}


def verify_environment(result):
    model, runtime = result["model"], result["runtime"]
    expected_model = {"model": "distilbert/distilbert-base-uncased",
                      "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be", "device": "cpu", "dtype": "float32",
                      "model_class": "DistilBertForMaskedLM", "tokenizer_class": "DistilBertTokenizerFast",
                      "attention_implementation": "eager", "parameter_count": 66985530,
                      "pytorch_cuda_initialized": False, "assets": EXPECTED_ASSETS,
                      "torch_version": "2.8.0+cu128", "transformers_version": "4.57.3"}
    expected_runtime = {"python": "3.11.15", "torch": "2.8.0+cu128", "threads": 1, "interop_threads": 1,
                        "deterministic_algorithms": True, "cuda_initialized": False, "default_dtype": "torch.float32",
                        "mkldnn_enabled": True, "omp_num_threads": "1", "mkl_num_threads": "1", "mkl_cbwr": "AUTO",
                        "aten_cpu_capability": None, "denormal_probe": {"smallest_times_one_bits": 1,
                                                                         "normal_half_bits": 4194304},
                        "rounding_probe": "nearest-even-halfway-cases-pass",
                        "torch_build_sha256": "9835a6eed581f616d1e294cc42162b701425ea24bf4723678b8233e78f30e8c6"}
    expected_exposure = {"prior_row_counts": [160, 64, 8], "exact_or_complete_prefix_collisions": 0,
                         "complete_exposure_history_certified": False}
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(canonical(model) == canonical(expected_model) and canonical(runtime) == canonical(expected_runtime)
            and canonical(result["exposure"]) == canonical(expected_exposure), "recorded environment differs")
    require(type(result["model_state_sha256_before_after"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", result["model_state_sha256_before_after"]) is not None,
            "recorded model state hash differs")


def _fixed_source(repo, source_ref):
    require(source_ref in (LOCAL_EQUIVALENT, SOURCE_COMMIT), "unsupported source ref")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    def blob(name):
        return subprocess.run(["git", "--no-replace-objects", "-C", str(repo), "show",
                               source_ref + ":" + name], env=env, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    content = {name: blob(name) for name in SOURCE_PATHS}
    cases = content["experiments/bert_taste_matched_dose_public_cases_v1.json"]
    require(sha(cases) == CASE_SHA, "frozen case digest differs")
    return {"commit": SOURCE_COMMIT, "selected_source_sha256": {name: sha(raw) for name, raw in content.items()}}, parse(cases)


def _schedule(frame):
    out = []
    def call(row, layer, position, mode, donor):
        identity = {"row_id": frame + "-" + row, "layer": layer, "position": position,
                    "mode": mode, "donor": None if donor is None else frame + "-" + donor}
        out.extend(({"event": "forward_start", **identity}, {"event": "forward_complete", **identity}))
    for layer in range(6):
        for row in ("sugar", "lemon"):
            call(row, layer, 7, "capture_ingredient", None)
            call(row, layer, 1, "capture_control", None)
        out.append({"event": "dose_plan", "frame": frame, "row_id": frame + "-sugar", "layer": layer})
        for row, other in (("sugar", "lemon"), ("lemon", "sugar")):
            for mode, position, self_patch in MODES:
                call(row, layer, position, mode, row if self_patch else other)
    for row in ("sugar", "lemon"):
        call(row, 0, 7, "final_clean", None)
    return out


def _baseline(summary, encoded):
    raw = base64.b64decode(encoded, validate=True)
    require(len(raw) == 30522 * 4 and sha(raw) == summary["mask_logits_sha256"], "baseline logit bytes differ")
    words = [value[0] for value in struct.iter_unpack("<I", raw)]
    values = [float32(word) for word in words]
    require(words[4086] == summary["sweet_bits"] and words[14768] == summary["sour_bits"],
            "baseline selected bits differ")
    top = sorted(range(len(values)), key=lambda index: (-values[index], index))[:5]
    require(top == summary["top_five_ids"], "baseline top-five differs")
    peak = max(values)
    total = math.fsum(math.exp(value - peak) for value in values)
    for index, key in ((4086, "sweet_probability"), (14768, "sour_probability")):
        require(math.isclose(summary[key], math.exp(values[index] - peak) / total, rel_tol=1e-12, abs_tol=1e-12),
                "baseline probability differs")
    return {"margin": selected_margin(summary), "top_five_ids": top,
            "sweet_probability": summary["sweet_probability"], "sour_probability": summary["sour_probability"]}


def audit_numeric(cases, result, journals):
    rows = cases["rows"]
    require(type(rows) is list and [row["id"] for row in rows] ==
            [frame + "-" + word for frame in FRAMES for word in ("sugar", "lemon")], "case order differs")
    require(cases["tokens"]["sweet"] == 4086 and cases["tokens"]["sour"] == 14768, "target tokens differ")
    by_id = {row["id"]: row for row in rows}
    require(type(journals) is dict and set(journals) == {frame + ".jsonl" for frame in FRAMES}, "journal set differs")
    baselines, effects = {}, {}
    complete_count = 0
    for frame in FRAMES:
        raw = journals[frame + ".jsonl"]
        require(raw.endswith(b"\n"), "journal is truncated")
        lines, schedule = raw.splitlines(), _schedule(frame)
        require(len(lines) == len(schedule), "journal schedule length differs")
        for line, identity in zip(lines, schedule):
            event = parse(line)
            require(type(event) is dict and all(type(event.get(key)) is type(value) and event.get(key) == value
                                                for key, value in identity.items()), "journal schedule differs")
            if identity["event"] != "forward_complete":
                continue
            complete_count += 1
            row, layer, mode = identity["row_id"], identity["layer"], identity["mode"]
            require(event["model_entry"] == {"input_ids": by_id[row]["input_ids"],
                                             "attention_mask": by_id[row]["attention_mask"]}, "model inputs differ")
            summary = event["summary"]
            require(type(summary.get("mask_logits_sha256")) is str and
                    re.fullmatch(r"[0-9a-f]{64}", summary["mask_logits_sha256"]) is not None, "logit hash differs")
            margin = selected_margin(summary)
            if mode == "capture_ingredient" and layer == 0:
                require(row not in baselines, "baseline is repeated")
                base = _baseline(summary, event["baseline_mask_logits_float32_le_base64"])
                base["mask_logits_sha256"] = summary["mask_logits_sha256"]
                baselines[row] = base
                require(result["baselines"][row] == summary, "result baseline differs")
            require(row in baselines, "baseline is missing")
            if mode.startswith("capture_") or mode == "final_clean" or mode.endswith("_self") or layer == 5:
                require(summary["mask_logits_sha256"] == baselines[row]["mask_logits_sha256"]
                        and margin == baselines[row]["margin"], "clean or control logits differ")
            if mode in {name for name, _position, _self in MODES}:
                direction = "lemon_to_sugar" if row.endswith("-sugar") else "sugar_to_lemon"
                key = frame, layer, direction, mode
                require(key not in effects, "duplicate intervention")
                effects[key] = {"recipient": row, "donor": identity["donor"], "position": identity["position"],
                                "baseline_margin": baselines[row]["margin"], "patched_margin": margin,
                                "effect": math.fsum((margin, -baselines[row]["margin"])), "summary": summary}
    require(complete_count == 516 and len(effects) == 360 and len(baselines) == 12, "completed-forward inventory differs")
    eligible = {}
    for frame in FRAMES:
        sugar, lemon = baselines[frame + "-sugar"], baselines[frame + "-lemon"]
        targets = {4086, 14768}
        eligible[frame] = (sugar["margin"] > 0 and lemon["margin"] < 0
                           and bool(targets.intersection(sugar["top_five_ids"]))
                           and bool(targets.intersection(lemon["top_five_ids"])))
    require(result["eligibility"] == eligible and all(type(v) is bool for v in result["eligibility"].values()),
            "eligibility differs")
    stored_effects = {}
    for item in result["effects"]:
        key = item["frame"], item["layer"], item["direction"], item["mode"]
        require(key not in stored_effects and key in effects, "result intervention identity differs")
        expected = effects[key]
        require(all(item[name] == value for name, value in expected.items()), "result intervention differs")
        stored_effects[key] = item
    require(set(stored_effects) == set(effects), "result interventions are incomplete")
    comparisons, layers = [], []
    for layer in range(6):
        group = {"layer": layer}
        for direction, sign in (("sugar_to_lemon", 1), ("lemon_to_sugar", -1)):
            differences = {frame: (effects[frame, layer, direction, "ingredient_matched"]["effect"],
                                   effects[frame, layer, direction, "control_matched"]["effect"]) for frame in FRAMES}
            summary = direction_summary(differences, eligible, sign=sign)
            comparisons.extend({"frame": frame, "layer": layer, "direction": direction,
                                "eligible": eligible[frame], "signed_advantage": summary[frame]} for frame in FRAMES)
            group[direction] = {"n": summary["n"], "mean": summary["mean"]}
        layers.append(group)
    require(result["comparisons"] == comparisons and result["layer_summary"] == layers,
            "directional comparisons or means differ")
    return {"completed_forwards": complete_count, "interventions": len(effects), "baselines": baselines,
            "eligibility": eligible, "comparisons": comparisons, "layer_summary": layers}


def evidence_paths(directory, *, published_package=False):
    root = Path(directory)
    if published_package:
        return {"run": root / "run", "observer": root / "receipts/observer.json",
                "exit": root / "receipts/operator-exit-projection.json",
                "start": root / "receipts/operator-start-projection.json",
                "post_exit": root / "receipts/post-exit-observation.json",
                "repo": root.parents[1], "source_ref": SOURCE_COMMIT}
    return {"run": root / "bert-taste-matched-context-20261029-01",
            "observer": root / "bert-taste-matched-context-20261029-01.observer.json",
            "exit": root / "bert-taste-matched-operator-20261029-01/exit.json",
            "post_exit": root / "bert-taste-matched-operator-20261029-01/post-exit-observation.json",
            "repo": root / "teams/bert_taste_matched_dose", "source_ref": LOCAL_EQUIVALENT}


def audit_closed(directory, *, published_package=False):
    paths = evidence_paths(directory, published_package=published_package)
    run, observer_path, exit_path, post_exit_path = (paths[key] for key in ("run", "observer", "exit", "post_exit"))
    root_info = run.lstat()
    require(stat.S_ISDIR(root_info.st_mode) and not (getattr(root_info, "st_file_attributes", 0) & 0x400),
            "run root is not an ordinary directory")
    require({p.name for p in run.iterdir()} == {"started.json", "result.json", *(f + ".jsonl" for f in FRAMES)},
            "closed run file set differs")
    result_raw, observer_raw, exit_raw, post_exit_raw = (read_regular(run / "result.json", 2_000_000),
        read_regular(observer_path, 250_000), read_regular(exit_path, 250_000), read_regular(post_exit_path, 250_000))
    expected_exit_sha = EXIT_PROJECTION_SHA if published_package else EXIT_SHA
    require((sha(result_raw), sha(observer_raw), sha(exit_raw), sha(post_exit_raw)) ==
            (RESULT_SHA, OBSERVER_SHA, expected_exit_sha, POST_EXIT_SHA), "reported result or observation digest differs")
    result, observer, exit_record, post_exit = (parse(raw) for raw in (result_raw, observer_raw, exit_raw, post_exit_raw))
    if published_package:
        verify_public_exit_projection(exit_record)
        start_raw = read_regular(paths["start"], 250_000)
        require(sha(start_raw) == START_PROJECTION_SHA, "start projection digest differs")
        start_projection = parse(start_raw)
        require(type(start_projection) is dict and "process_id" not in start_projection
                and start_projection.get("status") == "launched"
                and start_projection.get("source_commit") == SOURCE_COMMIT
                and start_projection.get("original_sha256") == START_ORIGINAL_SHA,
                "start projection differs")
    else:
        verify_private_exit(exit_record)
    require(post_exit["direct_process_record_sha256"] == EXIT_SHA
            and type(post_exit["matching_python_commandlines_at_observation"]) is int
            and post_exit["matching_python_commandlines_at_observation"] == 0
            and post_exit["observed_at_utc"] == "2026-10-29T07:22:59.5064323Z",
            "post-exit operator observation differs")
    require(observer["status"] == "complete" and observer["child_reaped"] is True
            and type(observer["child_exit_code"]) is int and observer["child_exit_code"] == 0
            and observer["bytecode_prefix_empty"] is True and observer["source_unchanged"] is True
            and observer["source_commit"] == SOURCE_COMMIT and observer["result_sha256"] == RESULT_SHA,
            "child observer receipt differs")
    require(result["status"] == "complete" and type(result["completed_forwards"]) is int
            and result["completed_forwards"] == 516 and result["seed"] == 20261030,
            "result completion differs")
    verify_environment(result)
    source, cases = _fixed_source(paths["repo"], paths["source_ref"])
    started = parse(read_regular(run / "started.json", 250_000))
    require(started == {"status": "started", "source": source,
                        "cases_path": "experiments/bert_taste_matched_dose_public_cases_v1.json", "seed": 20261030}
            and result["source"] == source, "saved source or started identity differs")
    journals = {frame + ".jsonl": read_regular(run / (frame + ".jsonl"), 4_000_000) for frame in FRAMES}
    journal_hashes = {name: {"size": len(raw), "sha256": sha(raw)} for name, raw in journals.items()}
    require(result["journals"] == observer["journals"] == journal_hashes, "saved journal hashes differ")
    numeric = audit_numeric(cases, result, journals)
    report = {"schema": "independent-matched-taste-custody-review-v1", "status": "matching_saved_records",
            "source_commit": SOURCE_COMMIT, "local_equivalent_tree_commit": LOCAL_EQUIVALENT,
            "cases_sha256": CASE_SHA, "result_sha256": RESULT_SHA, "observer_sha256": OBSERVER_SHA,
            "parent_exit_sha256": EXIT_SHA, "post_exit_observation_sha256": POST_EXIT_SHA,
            "journal_sha256": {k: v["sha256"] for k, v in journal_hashes.items()},
            **numeric, "limitations": ["Exit, cleanup, source and untouched-coordinate observations are not independently authenticated execution proof.",
                "The parent exit file records no observation timestamp or whole-process-tree proof.",
                "Patched selected bits are saved observations; without the full patched logits their association with the full-logit hash cannot be verified.",
                "Only clean baseline full-vocabulary logits were retained; patched probabilities and top-five are owner-reported.",
                "These public convenience cases are not a held-out or population sample."]}
    if published_package:
        report["replay_source_ref"] = paths["source_ref"]
        report["status"] = "matching_public_saved_records"
        report.pop("parent_exit_sha256")
        report["parent_exit_projection_sha256"] = EXIT_PROJECTION_SHA
        report["parent_exit_original_sha256_reported"] = EXIT_SHA
        report["parent_start_projection_sha256"] = START_PROJECTION_SHA
        report["parent_start_original_sha256_reported"] = START_ORIGINAL_SHA
        report["limitations"].append("The public projections do not establish the original operator records or their process identifiers; an earlier private review checked the originals.")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-package", action="store_true", help="read the adjacent public run and receipts")
    arguments = parser.parse_args()
    try:
        result = audit_closed(Path(__file__).resolve().parent, published_package=arguments.published_package)
    except Exception:
        sys.stdout.buffer.write(b'{"status":"incomplete"}\n')
        raise
    sys.stdout.buffer.write((json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
