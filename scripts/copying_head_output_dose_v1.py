"""Run the fixed family-wide projection-output-dose comparison on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback

if __name__ == "__main__" and (
        not sys.dont_write_bytecode or not sys.pycache_prefix
        or not Path(sys.pycache_prefix).is_absolute() or not Path(sys.pycache_prefix).is_dir()
        or any(Path(sys.pycache_prefix).iterdir()) or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or any(name in os.environ for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY"))
        or not os.environ.get("COPY_OUTPUT_DOSE_BOOTSTRAP_SHA256") or "torch" in sys.modules):
    raise RuntimeError("use the reviewed bootstrap with fresh bytecode and CPU/offline flags")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network

GUARD_EVENTS = []
SAFE_QUERIES = {"cuda_is_available": 0, "cuda_device_count": 0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)

from scripts import copying_head_prompt_dose_v1 as old
from scripts.build_copy_output_dose_cases_v1 import build as rebuild_cases
from looking_glass.copy_head_factorial import _norm, _values, binary64_rounding_probe, head_slice
from looking_glass.copy_head_output_dose import HEADS, WEIGHT_BYTES, inspect_actual, plan_family, project, weights
from looking_glass.copy_head_intervention import assert_no_forward_hooks

prior, base = old.prior, old.base
base.GUARD_EVENTS = GUARD_EVENTS
PROTOCOL = "docs/DISTILGPT2_COPY_OUTPUT_DOSE_V1.md"
CASES = "experiments/distilgpt2_copy_output_dose_cases_v1.json"
MANIFEST = "experiments/distilgpt2_copy_output_dose_manifest_v1.json"
PROTOCOL_SHA = "52bea66f131668ded270f95506b5911719dcd4b967dbc83320124f418187e485"
CASES_SHA = "e45fe8cfe8b28a4f4308a4d3b2206f35cebe8154c7b4301ac1356c28258b135e"
MAX_ACTION_BYTES = 80_604_288
MAX_WEIGHT_BYTES = 589_824
MAX_MANIFEST_BYTES = 1_048_576
MAX_JOURNAL_BYTES = 10_485_760
MAX_RESULT_BYTES = 5_242_880
MAX_FAILURE_BYTES = 1_048_576
ORDER = ("out_h2", "out_h10", "out_h7", "in_h2", "in_h10", "in_h7")


class BoundedJournal:
    """Reserve a fixed cumulative metadata budget, including partial failure."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        if self.stream.tell() + len(data) > MAX_JOURNAL_BYTES:
            raise OSError("journal metadata limit exceeded")
        size = self.stream.write(data)
        if size != len(data):
            raise OSError("short journal write")
        return size

    def flush(self):
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()


def bounded_json(path, value, maximum):
    if len(base.canonical(value)) > maximum:
        raise OSError("JSON metadata limit exceeded")
    return base.durable_json(path, value)


def git(*args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.check_output(["git", *args], cwd=ROOT, env=env, timeout=60)


def validate_source(manifest, manifest_path):
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("checkout is not clean")
    if manifest_path.resolve() != (ROOT / MANIFEST).resolve():
        raise RuntimeError("manifest path differs")
    revision = git("rev-parse", "HEAD").decode().strip()
    git("merge-base", "--is-ancestor", manifest["source_revision"], revision)
    if (git("diff", "--name-only", manifest["source_revision"], revision, "--").decode().splitlines() != [MANIFEST]
            or git("show", f"{revision}:{MANIFEST}") != manifest_path.read_bytes()):
        raise RuntimeError("manifest is not the sole committed source change")
    sources = manifest["source_sha256"]
    required = {PROTOCOL, CASES, "looking_glass/copy_head_output_dose.py",
                "looking_glass/pretrained_exposures_v36.py", "looking_glass/pretrained_exposures_v37.py",
                "scripts/copying_head_output_dose_v1.py", "scripts/copying_head_output_dose_bootstrap_v1.py",
                *old.REQUIRED_SOURCE_PATHS, *(f"looking_glass/pretrained_exposures_v{i}.py" for i in range(32, 36))}
    if not required.issubset(sources):
        raise RuntimeError("source map omits an imported source")
    for path, expected in sources.items():
        raw = (ROOT / path).read_bytes()
        if base.sha(raw) != expected or git("show", f"{manifest['source_revision']}:{path}") != raw:
            raise RuntimeError(f"source changed: {path}")
    if (sources[PROTOCOL] != PROTOCOL_SHA or sources[CASES] != CASES_SHA
            or git("show", "-s", "--format=%T", manifest["source_revision"]).decode().strip() != manifest["source_tree"]):
        raise RuntimeError("protocol, cases or source tree differ")
    return {"revision": revision, "source_revision": manifest["source_revision"],
            "source_tree": manifest["source_tree"], "sha256": sources}


def verify_reservation(cases, manifest):
    from looking_glass import pretrained_exposures_v36 as predecessor
    from looking_glass import pretrained_exposures_v37 as reservation
    previous, previous_sha = predecessor._checked_inventory()
    current, current_sha = reservation._checked_inventory()
    if (previous_sha != manifest["predecessor_inventory_sha256"]
            or current_sha != manifest["reservation_inventory_sha256"]
            or current["records"][:len(previous["records"])] != previous["records"]
            or len(current["records"]) != len(previous["records"]) + 32):
        raise RuntimeError("reservation predecessor or size differs")
    if len({row["record_id"] for row in current["records"]}) != len(current["records"]):
        raise RuntimeError("duplicate reservation identities")
    for case, row in zip(cases, current["records"][len(previous["records"]):]):
        if (row.get("record_id") != "planned-" + case["case_id"] or row.get("category") != "planned"
                or row.get("token_ids") != case["token_ids"] or row.get("target_token_id") != case["target_id"]
                or row.get("public_text") != case["text"]):
            raise RuntimeError("planned input or target identity differs")
    return {"predecessor": previous_sha, "reservation": current_sha,
            "predecessor_rows": len(previous["records"]), "reservation_rows": len(current["records"])}


def output_description(plan):
    result = {"status": plan["status"], "reason": plan["reason"], "q_out": plan["q_out"],
              "source_output_norms": [{str(h): row[h] for h in HEADS} for row in plan["source_output_norms"]],
              "minima": plan["minima"], "plans": None}
    if plan["plans"] is not None:
        result["plans"] = [{str(h): {"fraction": row[h]["fraction"], "source_bits_hex": row[h]["source"].hex(),
                           "delivered_bits_hex": row[h]["delivered"].hex(),
                           "input_removed_l2": row[h]["input_removed_l2"],
                           "predicted_output_l2": row[h]["predicted_output_l2"],
                           "changed_elements": row[h]["changed_elements"], "changed_bits": row[h]["changed_bits"]}
                            for h in HEADS} for row in plan["plans"]]
    return result


def rotation(family, variant):
    shift = (4 * (family - 1) + variant) % 6
    return ORDER[shift:] + ORDER[:shift]


def predicted_actual(cache, head, delivery, output, matrix):
    source = _values(head_slice(cache["before"], head))
    landed = _values(delivery)
    predicted = project(tuple(b - a for a, b in zip(source, landed)), matrix[head])
    before, after = _values(cache["noop_output"]), _values(output)
    actual = tuple(b - a for a, b in zip(before[10 * 768:], after[10 * 768:]))
    return {"input_removed_l2": _norm(b - a for a, b in zip(source, landed)),
            "predicted_output_l2": _norm(predicted), "actual_output_l2": _norm(actual),
            "predicted_vector_residual_l2": _norm(a - b for a, b in zip(actual, predicted))}


def direction_cosines(cache, prefix):
    if any(prefix + "_h" + str(h) not in cache.get("arm_outputs", {}) for h in HEADS):
        return None
    baseline = _values(cache["noop_output"])[10 * 768:]
    vectors = {h: tuple(b - a for a, b in zip(baseline,
               _values(cache["arm_outputs"][prefix + "_h" + str(h)])[10 * 768:])) for h in HEADS}
    norms = {h: _norm(vectors[h]) for h in HEADS}
    return {f"{a}-{b}": None if norms[a] <= 2**-8 or norms[b] <= 2**-8 else
            math.fsum(x * y for x, y in zip(vectors[a], vectors[b])) / (norms[a] * norms[b])
            for a, b in ((2, 10), (2, 7), (10, 7))}


def perform_action(case, kind, cache, head, delivery, *, model, runtime_before,
                   matrix, output, log, counts, activity, completed):
    case_id = case["case_id"]
    observation = {}
    marker = {"recorded": False, "partial": False, "partial_save_error": None}
    activity.update({"phase": "forward", "case_id": case_id, "action": kind,
                     "observation": observation, "marker": marker})
    ids = torch.tensor([case["token_ids"]], dtype=torch.int64, device="cpu")
    base._append_event(log, {"event": "start", "case_id": case_id, "action": kind})
    if prior.runtime(model) != runtime_before:
        raise RuntimeError("runtime changed before forward")
    path = output / (case_id + ".f32")
    start_offset = path.stat().st_size if path.exists() else 0
    before_returns, before_bound, before_retained = counts["model_returns"], counts["returned"], counts["retained"]
    sentinel, payload = object(), None
    payload = sentinel
    counts["submitted"] += 1
    try:
        action = kind if kind in ("clean", "noop", "clean_after") else "h" + str(head)
        payload = prior.run_forward(model, ids, action, expected_before=cache.get("before"),
                                    deliveries=None if head is None else {head: delivery}, observation=observation)
        observation["helper_return_observed"] = True
        counts["model_returns"] = before_returns + int(observation.get("model_call_return_observed", False))
        counts["returned"] = before_bound + 1
        counts["retained"] = before_retained + sum(map(len, payload.values()))
        if counts["retained"] > MAX_ACTION_BYTES:
            raise RuntimeError("raw payload exceeds fixed bound")
        activity["phase"] = "persist_and_validate"
        with path.open("xb" if kind == "clean" else "ab") as raw:
            offsets = base._write_payload(raw, payload)
        base._append_event(log, {"event": "raw_recorded", "case_id": case_id, "action": kind, "raw": offsets})
        marker["recorded"] = True
        if ids.tolist() != [case["token_ids"]] or prior.runtime(model) != runtime_before:
            raise RuntimeError("input or runtime changed")
        old.validate_returned_payload(payload, action, cache, head, delivery)
        score = old.logit_stats(payload["logits"], case["target_id"])
        record = {"case_id": case_id, "group": case["group"], "variant": case["variant"],
                  "target_id": case["target_id"], "action": kind, "score": old.public_score(score), "raw": offsets}
        if kind == "clean":
            cache.update({"clean": payload["logits"], "clean_score": score})
        elif kind == "noop":
            cache.update({"before": payload["cproj_before"], "noop_output": payload["cproj_output"]})
        elif head is not None:
            if payload["cproj_output"][:10 * 768 * 4] != cache["noop_output"][:10 * 768 * 4]:
                raise RuntimeError("untouched token-position output changed")
            record["comparison"] = old.compare(cache["clean_score"], score)
            record["dose"] = predicted_actual(cache, head, delivery, payload["cproj_output"], matrix)
            cache.setdefault("comparisons", {})[kind] = record["comparison"]
            cache.setdefault("arm_outputs", {})[kind] = payload["cproj_output"]
        base._append_event(log, {"event": "complete", **record})
        completed.append(record)
    except BaseException:
        bound = payload is not sentinel
        observation["helper_return_observed"] = bound
        observation["helper_return_unbound_uncertain"] = bool(observation.get("model_call_return_observed", False) and not bound)
        counts["model_returns"] = before_returns + int(observation.get("model_call_return_observed", False))
        counts["returned"] = before_bound + int(bound)
        partial = observation.get("payload", {})
        size = sum(map(len, partial.values()))
        counts["retained"] = before_retained + size if marker["recorded"] else before_retained
        if partial and not marker["recorded"]:
            try:
                if before_retained + size > MAX_ACTION_BYTES:
                    raise RuntimeError("partial payload exceeds limit")
                if old.recover_payload(log, path, case_id, kind, partial, start_offset,
                                       observation.get("model_call_return_observed", False)):
                    marker["partial"] = True
                    counts["retained"] = before_retained + size
            except Exception as error:
                marker["partial_save_error"] = type(error).__name__
        raise


def execute_family(batch, perform, plan_start, log, weight_raw):
    if len(batch) != 4 or any(c["group"] != batch[0]["group"] or c["variant"] != i for i, c in enumerate(batch)):
        raise ValueError("family case identities or variant order differ")
    group = batch[0]["group"]
    caches = [{} for _ in range(4)]
    for case, cache in zip(batch, caches):
        perform(case, "clean", cache)
        perform(case, "noop", cache)
    plan_start()
    out_plan = plan_family(tuple(cache["before"] for cache in caches), weight_raw)
    in_plans = [old.plan_prompt(cache["before"]) for cache in caches]
    base._append_event(log, {"event": "dose_plan", "group": group,
                            "output": output_description(out_plan),
                            "input": [old.plan_description(p) for p in in_plans]})
    for variant, (case, cache, in_plan) in enumerate(zip(batch, caches, in_plans)):
        for kind in rotation(group, variant):
            scheme, head_text = kind.split("_")
            head = int(head_text[1:])
            plan = out_plan if scheme == "out" else in_plan
            if plan["status"] != "eligible":
                base._append_event(log, {"event": "skipped", "case_id": case["case_id"],
                                        "action": kind, "reason": plan["reason"]})
                continue
            delivery = (out_plan["plans"][variant][head]["delivered"] if scheme == "out"
                        else in_plan["plans"][head]["delivered"])
            perform(case, kind, cache, head, delivery)
        perform(case, "clean_after", cache)
    if out_plan["status"] == "eligible":
        assessment = inspect_actual(tuple(c["noop_output"] for c in caches),
                                    tuple({h: c["arm_outputs"]["out_h" + str(h)] for h in HEADS} for c in caches),
                                    out_plan)
    else:
        assessment = {"status": "dose-ineligible", "reason": out_plan["reason"], "q_out": out_plan["q_out"]}
    regime_cases = {"output": [], "input": []}
    for case, cache, in_plan in zip(batch, caches, in_plans):
        for label, prefix, status, q in (("output", "out", assessment["status"], out_plan["q_out"]),
                                          ("input", "in", in_plan["status"], in_plan["q"])):
            comparisons = {"h" + str(h): cache.get("comparisons", {}).get(prefix + "_h" + str(h)) for h in HEADS}
            regime_cases[label].append({"status": status, "q": q, "comparisons": comparisons})
    summary = {label: old.family_summary(rows) for label, rows in regime_cases.items()}
    clean = [c["clean_score"]["target_log_probability"] for c in caches]
    family = {"group": group, "output_plan": output_description(out_plan),
              "input_plans": [old.plan_description(p) for p in in_plans],
              "output_assessment": assessment, "summaries": summary,
              "input_output_cosines": [direction_cosines(c, "in") for c in caches],
              "clean_target_ranks": [c["clean_score"]["target_rank"] for c in caches],
              "clean_repeat_minus_shuffle": math.fsum((clean[0], -math.fsum(clean[1:]) / 3))}
    base._append_event(log, {"event": "family_complete", **family})
    return family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if any(not path.is_absolute() for path in (args.cache_dir, args.output_dir, args.manifest)):
        raise ValueError("all paths must be absolute")
    raw_manifest = args.manifest.read_bytes()
    if os.environ.get("COPY_OUTPUT_DOSE_BOOTSTRAP_SHA256") != base.sha(raw_manifest):
        raise RuntimeError("bootstrap and manifest binding differ")
    manifest = json.loads(raw_manifest)
    if manifest.get("schema") != "distilgpt2-copy-output-dose-manifest-v1":
        raise ValueError("wrong manifest schema")
    source = validate_source(manifest, args.manifest)
    raw_cases = (ROOT / CASES).read_bytes()
    if base.sha(raw_cases) != CASES_SHA:
        raise ValueError("case bytes changed")
    case_data = json.loads(raw_cases)
    if rebuild_cases(args.cache_dir / old.TOKENIZER_REL) != case_data:
        raise RuntimeError("frozen tokenizer-only case construction differs")
    cases = case_data["cases"]
    if len(cases) != 32 or any(c["group"] != i // 4 + 1 or c["variant"] != i % 4 for i, c in enumerate(cases)):
        raise ValueError("case order differs")
    exposure = verify_reservation(cases, manifest)
    output = args.output_dir
    parent = output.parent.resolve(strict=True)
    if output.exists() or ROOT == parent or ROOT in parent.parents or parent.drive.upper() != "E:":
        raise ValueError("output must be new and outside source checkout")
    needed = MAX_ACTION_BYTES + MAX_WEIGHT_BYTES + 16_777_216 + 1_048_576
    if shutil.disk_usage(parent).free < needed or shutil.disk_usage("C:\\").free < 1_073_741_824:
        raise OSError("insufficient E or C disk headroom")
    torch.set_num_interop_threads(1)
    initial_probe = base.numeric_policy(initialize=True)
    binary64_rounding_probe()
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    output.mkdir()
    run_manifest = {"schema": "distilgpt2-copy-output-dose-run-v1", "source": source,
                    "submitted_manifest_sha256": base.sha(raw_manifest), "dependencies": base.dependency_identity(),
                    "exposure": exposure, "cases_sha256": CASES_SHA, "protocol_sha256": PROTOCOL_SHA,
                    "model_id": base.MODEL_ID, "model_revision": base.MODEL_REVISION,
                    "weights_sha256": base.WEIGHTS_SHA256, "initial_numeric_probe": initial_probe,
                    "maximum_action_bytes": MAX_ACTION_BYTES, "maximum_weight_bytes": MAX_WEIGHT_BYTES,
                    "maximum_manifest_bytes": MAX_MANIFEST_BYTES, "maximum_journal_bytes": MAX_JOURNAL_BYTES,
                    "maximum_result_bytes": MAX_RESULT_BYTES, "maximum_failure_bytes": MAX_FAILURE_BYTES,
                    "scope": "public exploratory; nonblind; known-input exclusions only"}
    run_manifest_sha = bounded_json(output / "manifest.json", run_manifest, MAX_MANIFEST_BYTES)
    completed, family_results = [], []
    counts = {"submitted": 0, "model_returns": 0, "returned": 0, "retained": 0}
    activity = {"phase": "model_load", "case_id": None, "action": None,
                "observation": {}, "marker": {"recorded": False, "partial": False, "partial_save_error": None}}
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        if bundle.metadata["weights_sha256"] != base.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint weights changed")
        model = bundle.model
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        state_before, buffers_before, runtime_before = base.state_hash(model), base.buffer_hash(model), prior.runtime(model)
        assert_no_forward_hooks(model)
        weight = model.transformer.h[3].attn.c_proj.weight
        if tuple(weight.shape) != (768, 768) or weight.dtype != torch.float32 or weight.device.type != "cpu":
            raise RuntimeError("checkpoint projection geometry differs")
        weight_raw = {h: base._raw(weight[64*h:64*(h+1)].detach().contiguous(), WEIGHT_BYTES) for h in HEADS}
        matrix = weights(weight_raw)
        weight_path = output / "projection-weights.f32"
        with weight_path.open("xb") as stream:
            raw = b"".join(weight_raw[h] for h in HEADS)
            if len(raw) != MAX_WEIGHT_BYTES or stream.write(raw) != len(raw):
                raise OSError("weight slice write failed")
            stream.flush(); os.fsync(stream.fileno())
        with (output / "attempt.jsonl").open("xb") as stream:
            log = BoundedJournal(stream)
            base._append_event(log, {"event": "runtime_ready", "actual_runtime": runtime_before,
                                    "weight_order": list(HEADS), "weight_sha256": base.sha(raw),
                                    "weight_head_sha256": {str(h): base.sha(weight_raw[h]) for h in HEADS}})
            def perform(case, kind, cache, head=None, delivery=None):
                return perform_action(case, kind, cache, head, delivery, model=model, runtime_before=runtime_before,
                                      matrix=matrix, output=output, log=log, counts=counts,
                                      activity=activity, completed=completed)
            def plan_start():
                activity.update({"phase": "dose_plan", "case_id": None, "action": None,
                                 "observation": {}, "marker": {"recorded": False, "partial": False, "partial_save_error": None}})
            for group in range(8):
                batch = cases[4*group:4*group+4]
                family_results.append(execute_family(batch, perform, plan_start, log, weight_raw))
        activity["phase"] = "verification"
        in_count = sum(p["status"] == "eligible" for f in family_results for p in f["input_plans"])
        out_count = sum(f["output_plan"]["status"] == "eligible" for f in family_results)
        forwards = 96 + 3 * in_count + 12 * out_count
        raw_bytes = 32 * (3 * base.LOGITS_BYTES + 3 * base.WITNESS_BYTES) + (3 * in_count + 12 * out_count) * (base.LOGITS_BYTES + 3 * base.WITNESS_BYTES)
        if (len(family_results) != 8 or len(completed) != forwards or counts["submitted"] != forwards
                or counts["model_returns"] != forwards or counts["returned"] != forwards or counts["retained"] != raw_bytes
                or base.state_hash(model) != state_before or base.buffer_hash(model) != buffers_before
                or prior.runtime(model) != runtime_before or validate_source(manifest, args.manifest) != source
                or GUARD_EVENTS or torch.cuda.is_initialized()):
            raise RuntimeError("final count, state, runtime or source verification failed")
        assert_no_forward_hooks(model)
        result = {"schema": "distilgpt2-copy-output-dose-result-v1", "status": "complete",
                  "manifest_sha256": run_manifest_sha, "checkpoint": bundle.metadata,
                  "weight_sha256": base.sha(raw), "weight_head_sha256": {str(h): base.sha(weight_raw[h]) for h in HEADS},
                  "state_sha256_before": state_before, "state_sha256_after": base.state_hash(model),
                  "buffers_sha256_before": buffers_before, "buffers_sha256_after": base.buffer_hash(model),
                  "runtime": runtime_before, "guard_safe_queries": SAFE_QUERIES,
                  "submitted_forwards": counts["submitted"], "model_call_returns_observed": counts["model_returns"],
                  "returned_forwards": counts["returned"], "raw_payload_bytes": counts["retained"],
                  "records": completed, "families": family_results,
                  "decisions": {label: old.decision([f["summaries"][label] for f in family_results]) for label in ("output", "input")}}
        bounded_json(output / "result.json", result, MAX_RESULT_BYTES)
    except BaseException as error:
        attempt = output / "attempt.jsonl"
        if attempt.is_file():
            try:
                with attempt.open("ab") as stream:
                    log = BoundedJournal(stream)
                    base._append_event(log, {"event": "failed", "case_id": activity["case_id"], "action": activity["action"],
                                             "phase": activity["phase"], "error_type": type(error).__name__, "marker": activity["marker"]})
            except OSError:
                pass
        bounded_json(output / "failure.json", {"schema": "distilgpt2-copy-output-dose-failure-v1",
            "status": "incomplete", "manifest_sha256": run_manifest_sha, "phase": activity["phase"],
            "case_id": activity["case_id"], "action": activity["action"], "marker": activity["marker"],
            "model_call_return_observed": activity["observation"].get("model_call_return_observed", False),
            "helper_return_observed": activity["observation"].get("helper_return_observed", False),
            "helper_return_unbound_uncertain": activity["observation"].get("helper_return_unbound_uncertain", False),
            "submitted_forwards": counts["submitted"], "model_call_returns_observed": counts["model_returns"],
            "returned_forwards": counts["returned"], "completed_records": len(completed),
            "admitted_payload_bytes": counts["retained"], "raw_files": base.raw_file_inventory(output, cases),
            "weight_sha256": base.sha((output / "projection-weights.f32").read_bytes())
                             if (output / "projection-weights.f32").is_file() else None,
            "error_type": type(error).__name__,
            "traceback_frames": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                 for frame in traceback.extract_tb(error.__traceback__)]}, MAX_FAILURE_BYTES)
        raise
    print(json.dumps({"status": "complete", "forwards": counts["submitted"], "raw_payload_bytes": counts["retained"],
                      "decisions": result["decisions"]}, allow_nan=False))


if __name__ == "__main__":
    main()
