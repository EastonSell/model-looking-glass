"""Execute the fixed public per-prompt matched-dose study on one CPU owner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback

if __name__ == "__main__" and (
        not sys.dont_write_bytecode or not sys.pycache_prefix
        or not Path(sys.pycache_prefix).is_absolute()
        or not Path(sys.pycache_prefix).is_dir() or any(Path(sys.pycache_prefix).iterdir())
        or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
    raise RuntimeError("use -B, a fresh absolute -X pycache_prefix and offline flags before imports")
if __name__ == "__main__" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("set CUDA_VISIBLE_DEVICES=-1 before starting")
if __name__ == "__main__" and not os.environ.get("COPY_HEAD_PROMPT_BOOTSTRAP_SHA256"):
    raise RuntimeError("launch through the reviewed stdlib-only source bootstrap")
if __name__ == "__main__" and "torch" in sys.modules:
    raise RuntimeError("torch was imported before the backend environment snapshot")
ENV_NAMES = ("MKL_CBWR", "ATEN_CPU_CAPABILITY")
ENV_AT_IMPORT = {name: {"present": name in os.environ, "value": os.environ.get(name)} for name in ENV_NAMES}
EXPECTED_ENV = {name: {"present": False, "value": None} for name in ENV_NAMES}
if __name__ == "__main__" and ENV_AT_IMPORT != EXPECTED_ENV:
    raise RuntimeError("CPU backend override variables must be absent")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network  # noqa: E402

GUARD_EVENTS = []
SAFE_QUERIES = {"cuda_is_available": 0, "cuda_device_count": 0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)

from scripts import copying_head_factorial_v1 as prior  # noqa: E402
from scripts.build_copy_head_prompt_dose_cases_v1 import build as rebuild_cases  # noqa: E402
from looking_glass.copy_head_intervention import assert_no_forward_hooks  # noqa: E402
from looking_glass.copy_head_factorial import binary64_rounding_probe  # noqa: E402
from looking_glass.copy_head_prompt_dose import (  # noqa: E402
    ARMS, HEADS, action_order, compare, decision, family_summary, logit_stats,
    output_delta_l2, plan_prompt,
)

base = prior.base
base.GUARD_EVENTS = GUARD_EVENTS
PROTOCOL = "docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md"
CASES = "experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json"
PROTOCOL_SHA = "40d1aac13f0faa82c048db033f8a4152d8c6efb564274afa829b58042bac5d6d"
CASES_SHA = "b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"
MANIFEST = "experiments/distilgpt2_copy_head_prompt_dose_manifest_v1.json"
TOKENIZER_REL = "models--distilbert--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa/tokenizer.json"
MAX_RAW_BYTES = 51_573_504
REQUIRED_SOURCE_PATHS = tuple(dict.fromkeys((*prior.SOURCES, PROTOCOL, CASES, "scripts/copying_head_prompt_dose_v1.py",
    "scripts/copying_head_prompt_dose_bootstrap_v1.py",
    "scripts/build_copy_head_prompt_dose_cases_v1.py", "looking_glass/copy_head_prompt_dose.py",
    "docs/COPY_HEAD_PROMPT_DOSE_RUNNER_V1.md", "tests/test_copy_head_prompt_dose.py",
    "tests/test_copying_head_prompt_dose_v1.py",
    "looking_glass/copy_head_factorial.py", "looking_glass/copy_head_intervention.py",
    "scripts/copying_head_factorial_v1.py", "scripts/copying_head_followup_v2.py",
    "scripts/copying_head_eval_v1.py", "scripts/copying_dev_v1.py",
    "scripts/distilgpt2_intervention_dev_bootstrap.py", "looking_glass/__init__.py",
    "looking_glass/pretrained.py", "looking_glass/pretrained_exposures.py",
    *(f"looking_glass/pretrained_exposures_v{i}.py" for i in range(2, 32)),
    "experiments/natural_agreement_original_public_support_v1.json",
    "experiments/natural_agreement_successor_inputs_v1.json")))


def git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.check_output(["git", *args], cwd=ROOT, env=env, timeout=60)


def validate_source(manifest, manifest_path):
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("checkout is not clean")
    if manifest_path.resolve() != (ROOT / MANIFEST).resolve():
        raise RuntimeError("manifest must be the published checkout file")
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    git("merge-base", "--is-ancestor", manifest["source_revision"], revision)
    changed = git("diff", "--name-only", manifest["source_revision"], revision, "--").decode().splitlines()
    if changed != [MANIFEST] or git("show", f"{revision}:{MANIFEST}") != manifest_path.read_bytes():
        raise RuntimeError("manifest is not the sole source-to-HEAD tree change")
    sources = manifest["source_sha256"]
    if not set(REQUIRED_SOURCE_PATHS).issubset(sources):
        raise RuntimeError("manifest omits a required source")
    for path, expected in sources.items():
        raw = (ROOT / path).read_bytes()
        if base.sha(raw) != expected or git("show", f"{manifest['source_revision']}:{path}") != raw:
            raise RuntimeError(f"source bytes changed: {path}")
    if (sources[PROTOCOL] != PROTOCOL_SHA or sources[CASES] != CASES_SHA
            or git("show", "-s", "--format=%T", manifest["source_revision"]).decode().strip()
            != manifest["source_tree"]):
        raise RuntimeError("source tree, protocol or case binding changed")
    return {"revision": revision, "source_revision": manifest["source_revision"],
            "source_tree": manifest["source_tree"], "sha256": sources}


def verify_reservation(cases, manifest):
    from looking_glass import pretrained_exposures_v30 as predecessor
    from looking_glass import pretrained_exposures_v31 as reservation
    old, old_digest = predecessor._checked_inventory()
    payload, digest = reservation._checked_inventory()
    if (old_digest != manifest["predecessor_inventory_sha256"]
            or digest != manifest["reservation_inventory_sha256"]):
        raise RuntimeError("exposure inventory binding changed")
    rows = tuple(tuple(case["token_ids"]) for case in cases)
    if rows != reservation.COPY_PROMPT_DOSE_ROWS:
        raise RuntimeError("reservation does not bind the exact 32 model inputs")
    if (payload["records"][:len(old["records"])] != old["records"]
            or len(payload["records"]) != len(old["records"]) + 34):
        raise RuntimeError("unexpected reservation record count")
    added = payload["records"][len(old["records"]):]
    identities = [record["record_id"] for record in payload["records"]]
    if len(set(identities)) != len(identities):
        raise RuntimeError("reservation record identities repeat")
    for case, record in zip(cases, added[:32]):
        if (record.get("record_id") != "planned-" + case["case_id"] or record.get("category") != "planned"
                or record.get("token_ids") != case["token_ids"]
                or record.get("target_token_id") != case["target_id"]
                or record.get("public_text") != case["text"]):
            raise RuntimeError("planned copying identity, input, label or status differs")
    for (identity, row), record in zip(reservation.SYNTHETIC_SERVICE_CASES, added[32:]):
        if (record.get("record_id") != identity or record.get("category") != "planned"
                or record.get("token_ids") != list(row) or record.get("attention_mask") != [1] * len(row)):
            raise RuntimeError("planned synthetic input or mask differs")
    return {"predecessor": old_digest, "reservation": digest,
            "predecessor_rows": len(old["records"]), "reservation_rows": len(payload["records"])}


def plan_description(plan):
    heads = {}
    for head in HEADS:
        row = {"source_l2": plan["norms"][head], "source_bits_hex": plan["original"][head].hex()}
        if plan["plans"] is not None:
            planned = plan["plans"][head]
            row.update({key: planned[key] for key in ("fraction", "achieved", "changed_elements", "changed_bits")})
            row["delivered_bits_hex"] = planned["delivered"].hex()
        heads[str(head)] = row
    return {"status": plan["status"], "reason": plan["reason"], "q": plan["q"], "heads": heads,
            "plan_unavailable": "plan unavailable: dose floor" if plan["plans"] is None else None}


def public_score(score):
    return {key: value for key, value in score.items() if key != "values"}


def validate_returned_payload(payload, kind, cache, head, delivery):
    for name, data in payload.items():
        if any((item[0] & 0x7f800000) == 0x7f800000 for item in struct.iter_unpack("<I", data)):
            raise RuntimeError(f"nonfinite returned {name}")
    if kind in ("noop", "clean_after") and payload["logits"] != cache["clean"]:
        raise RuntimeError("clean, no-op and clean-after logits differ")
    if kind == "noop" and payload["cproj_before"] != payload["cproj_after"]:
        raise RuntimeError("no-op changed projection input bits")
    if kind in ARMS:
        if payload["cproj_before"] != cache["before"]:
            raise RuntimeError("arm pre-action input differs from the prompt no-op")
        expected = bytearray(cache["before"])
        start = (10 * 768 + 64 * head) * 4
        expected[start:start + 256] = delivery
        if payload["cproj_after"] != bytes(expected):
            raise RuntimeError("planned delivery or an untouched coordinate differs")


def recover_payload(log, path, case_id, action, payload, start_offset, model_returned):
    """Complete a partially written returned payload without duplicating bytes.

    A mismatching or unexpectedly long partial file is left untouched and the
    attempt fails. This is best-effort evidence retention, not proof of process
    chronology or recovery from a hard process termination.
    """
    if not payload:
        return False
    expected = b"".join(payload.values())
    mode = "r+b" if path.exists() else "x+b"
    with path.open(mode) as raw:
        raw.seek(0, os.SEEK_END)
        end = raw.tell()
        if end < start_offset or end - start_offset > len(expected):
            raise RuntimeError("partial raw payload has an unexpected length")
        raw.seek(start_offset)
        prefix = raw.read(end - start_offset)
        if prefix != expected[:len(prefix)]:
            raise RuntimeError("partial raw payload differs from returned bytes")
        raw.seek(0, os.SEEK_END)
        suffix = expected[len(prefix):]
        if raw.write(suffix) != len(suffix):
            raise OSError("short recovered payload write")
        raw.flush()
        os.fsync(raw.fileno())
    offsets, offset = {}, start_offset
    for name, data in payload.items():
        offsets[name] = {"offset": offset, "bytes": len(data), "sha256": base.sha(data)}
        offset += len(data)
    base._append_event(log, {"event": "raw_recorded_recovered", "case_id": case_id,
                             "action": action, "model_call_return_observed": model_returned,
                             "raw": offsets})
    return True


def execute_case(case, perform, plan_start, plan_record):
    """Execute the fixed one-prompt schedule with an injectable forward owner."""
    cache = {}
    perform(case, "clean", cache)
    perform(case, "noop", cache)
    plan_start(case)
    plan = plan_prompt(cache["before"])
    desc = plan_description(plan)
    plan_record(case, desc)
    if plan["status"] == "eligible":
        for kind in action_order(case["group"] - 1, case["variant"]):
            head = int(kind[1:])
            perform(case, kind, cache, head, plan["plans"][head]["delivered"])
    perform(case, "clean_after", cache)
    return {"case_id": case["case_id"], "group": case["group"], "variant": case["variant"],
            "status": plan["status"], "q": plan["q"], "plan": desc,
            "comparisons": cache.get("comparisons", {})}


def perform_action(case, kind, cache, head, delivery, *, model, runtime_before,
                   output, log, counts, activity, completed):
    """Run one action and preserve bounded returned bytes on any later failure."""
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
    before_model_returns, before_returned, before_retained = (counts["model_returns"],
                                                              counts["returned"], counts["retained"])
    sentinel = object()
    payload = sentinel
    counts["submitted"] += 1
    try:
        payload = prior.run_forward(model, ids, kind, expected_before=cache.get("before"),
                                    deliveries=None if head is None else {head: delivery},
                                    observation=observation)
        observation["helper_return_observed"] = True
        counts["model_returns"] = before_model_returns + int(observation.get("model_call_return_observed", False))
        counts["returned"] = before_returned + 1
        counts["retained"] = before_retained + sum(map(len, payload.values()))
        if counts["retained"] > MAX_RAW_BYTES:
            raise RuntimeError("raw payload exceeds fixed bound")
        activity["phase"] = "persist_and_validate"
        with path.open("xb" if kind == "clean" else "ab") as raw:
            offsets = base._write_payload(raw, payload)
        base._append_event(log, {"event": "raw_recorded", "case_id": case_id,
                                 "action": kind, "raw": offsets})
        marker["recorded"] = True
        if ids.tolist() != [case["token_ids"]] or prior.runtime(model) != runtime_before:
            raise RuntimeError("input or runtime changed after forward")
        validate_returned_payload(payload, kind, cache, head, delivery)
        score = logit_stats(payload["logits"], case["target_id"])
        record = {"case_id": case_id, "group": case["group"], "variant": case["variant"],
                  "target_id": case["target_id"], "action": kind,
                  "score": public_score(score), "raw": offsets}
        if kind == "clean":
            cache.update({"clean": payload["logits"], "clean_score": score})
        elif kind == "noop":
            cache.update({"before": payload["cproj_before"], "noop_output": payload["cproj_output"]})
        elif kind in ARMS:
            record["comparison"] = compare(cache["clean_score"], score)
            record["projection_output_delta_l2"] = output_delta_l2(cache["noop_output"], payload["cproj_output"])
            cache.setdefault("comparisons", {})[kind] = record["comparison"]
        base._append_event(log, {"event": "complete", **record})
        completed.append(record)
    except BaseException:
        helper_bound = payload is not sentinel
        observation["helper_return_observed"] = helper_bound
        observation["helper_return_unbound_uncertain"] = bool(
            observation.get("model_call_return_observed", False) and not helper_bound)
        counts["model_returns"] = before_model_returns + int(observation.get("model_call_return_observed", False))
        counts["returned"] = before_returned + int(helper_bound)
        partial = observation.get("payload", {})
        size = sum(map(len, partial.values()))
        counts["retained"] = before_retained + size if marker["recorded"] else before_retained
        if partial and not marker["recorded"]:
            try:
                if before_retained + size > MAX_RAW_BYTES:
                    raise RuntimeError("partial payload exceeds fixed bound")
                if recover_payload(log, path, case_id, kind, partial, start_offset,
                                   observation.get("model_call_return_observed", False)):
                    marker["partial"] = True
                    counts["retained"] = before_retained + size
            except Exception as error:
                marker["partial_save_error"] = type(error).__name__
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if (not sys.dont_write_bytecode or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or any(Path(sys.pycache_prefix).iterdir()) or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise RuntimeError("use -B, a fresh absolute -X pycache_prefix and offline flags")
    if not args.cache_dir.is_absolute() or not args.output_dir.is_absolute() or not args.manifest.is_absolute():
        raise ValueError("cache, output and manifest paths must be absolute")
    if ENV_AT_IMPORT != EXPECTED_ENV or any(name in os.environ for name in ENV_NAMES):
        raise RuntimeError("CPU backend environment changed")
    manifest_bytes = args.manifest.read_bytes()
    if os.environ.get("COPY_HEAD_PROMPT_BOOTSTRAP_SHA256") != base.sha(manifest_bytes):
        raise RuntimeError("bootstrap and owner manifest bytes differ")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "distilgpt2-copy-head-prompt-dose-manifest-v1":
        raise ValueError("unexpected manifest")
    source = validate_source(manifest, args.manifest)
    case_bytes = (ROOT / CASES).read_bytes()
    case_data = json.loads(case_bytes)
    if base.sha(case_bytes) != CASES_SHA:
        raise ValueError("case bytes changed")
    if rebuild_cases(args.cache_dir / TOKENIZER_REL) != case_data:
        raise RuntimeError("case construction does not reproduce the frozen cases")
    cases = case_data["cases"]
    exposure = verify_reservation(cases, manifest)
    dependencies = base.dependency_identity()
    output = args.output_dir
    parent = output.parent.resolve(strict=True)
    if output.exists() or ROOT == parent or ROOT in parent.parents:
        raise ValueError("output must be a new directory outside the checkout")
    if shutil.disk_usage(parent).free < MAX_RAW_BYTES + 16_777_216:
        raise OSError("insufficient space for the fixed payload limit")
    torch.set_num_interop_threads(1)
    initial_probe = base.numeric_policy(initialize=True)
    binary64_rounding_probe()
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    output.mkdir()
    run_manifest = {"schema": "distilgpt2-copy-head-prompt-dose-run-v1", "source": source,
                    "submitted_manifest_sha256": base.sha(manifest_bytes), "dependencies": dependencies,
                    "exposure": exposure, "cases_sha256": CASES_SHA, "protocol_sha256": PROTOCOL_SHA,
                    "model_id": base.MODEL_ID, "model_revision": base.MODEL_REVISION,
                    "weights_sha256": base.WEIGHTS_SHA256, "maximum_raw_payload_bytes": MAX_RAW_BYTES,
                    "initial_numeric_probe": initial_probe, "cpu_backend_environment": ENV_AT_IMPORT,
                    "counter_semantics": {"submitted_forwards": "locally admitted helper attempts; interruption can precede the actual call",
                        "model_call_returns_observed": "returns observed inside the helper",
                        "returned_forwards": "helper returns whose payload was bound by the caller; an unbound return can be uncertain",
                        "raw_payload_bytes": "admitted payload bytes; actual partial-file inventory is authoritative after failure"},
                    "scope": "public exploratory; nonblind; known-input exclusions only"}
    run_manifest_sha = base.durable_json(output / "manifest.json", run_manifest)
    completed, case_results = [], []
    counts = {"submitted": 0, "model_returns": 0, "returned": 0, "retained": 0}
    activity = {"phase": "model_load", "case_id": None, "action": None,
                "observation": {}, "marker": {"recorded": False, "partial": False, "partial_save_error": None}}
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        if bundle.metadata["weights_sha256"] != base.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint weight hash changed")
        model = bundle.model
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        state_before, buffers_before = base.state_hash(model), base.buffer_hash(model)
        runtime_before = prior.runtime(model)
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("interop thread count differs")
        assert_no_forward_hooks(model)
        with (output / "attempt.jsonl").open("xb") as log:
            base._append_event(log, {"event": "runtime_ready", "actual_runtime": runtime_before})

            def perform(case, kind, cache, head=None, delivery=None):
                return perform_action(case, kind, cache, head, delivery, model=model,
                                      runtime_before=runtime_before, output=output, log=log,
                                      counts=counts, activity=activity, completed=completed)

            def plan_start(case):
                activity.update({"phase": "dose_plan", "case_id": case["case_id"], "action": None,
                                 "observation": {}, "marker": {"recorded": False, "partial": False,
                                                                    "partial_save_error": None}})

            def plan_record(case, desc):
                base._append_event(log, {"event": "dose_plan", "case_id": case["case_id"], **desc})

            for case in cases:
                case_results.append(execute_case(case, perform, plan_start, plan_record))
        activity["phase"] = "verification"
        eligible = sum(row["status"] == "eligible" for row in case_results)
        expected_forwards = 3 * len(cases) + 3 * eligible
        expected_bytes = (len(cases) * (3 * base.LOGITS_BYTES + 3 * base.WITNESS_BYTES)
                          + eligible * (3 * base.LOGITS_BYTES + 9 * base.WITNESS_BYTES))
        if (len(case_results) != 32 or len(completed) != expected_forwards or counts["submitted"] != expected_forwards
                or counts["model_returns"] != expected_forwards or counts["returned"] != expected_forwards
                or counts["retained"] != expected_bytes
                or base.state_hash(model) != state_before or base.buffer_hash(model) != buffers_before
                or prior.runtime(model) != runtime_before or validate_source(manifest, args.manifest) != source
                or GUARD_EVENTS or torch.cuda.is_initialized()):
            raise RuntimeError("final count, state, runtime or source verification failed")
        assert_no_forward_hooks(model)
        families = [family_summary(case_results[4 * group:4 * group + 4]) for group in range(8)]
        result = {"schema": "distilgpt2-copy-head-prompt-dose-result-v1", "status": "complete",
                  "manifest_sha256": run_manifest_sha, "checkpoint": bundle.metadata,
                  "state_sha256_before": state_before, "state_sha256_after": base.state_hash(model),
                  "buffers_sha256_before": buffers_before, "buffers_sha256_after": base.buffer_hash(model),
                  "runtime": runtime_before, "guard_safe_queries": SAFE_QUERIES,
                  "submitted_forwards": counts["submitted"], "model_call_returns_observed": counts["model_returns"],
                  "returned_forwards": counts["returned"], "raw_payload_bytes": counts["retained"], "records": completed,
                  "cases": case_results, "families": families, "decision": decision(families)}
        base.durable_json(output / "result.json", result)
    except BaseException as error:
        attempt = output / "attempt.jsonl"
        if attempt.is_file():
            try:
                with attempt.open("ab") as log:
                    base._append_event(log, {"event": "failed", "case_id": activity["case_id"], "action": activity["action"],
                        "phase": activity["phase"], "error_type": type(error).__name__, "marker": activity["marker"]})
            except OSError:
                pass
        base.durable_json(output / "failure.json", {"schema": "distilgpt2-copy-head-prompt-dose-failure-v1",
            "status": "incomplete", "manifest_sha256": run_manifest_sha, "phase": activity["phase"],
            "case_id": activity["case_id"], "action": activity["action"], "marker": activity["marker"],
            "model_call_return_observed": activity["observation"].get("model_call_return_observed", False),
            "helper_return_observed": activity["observation"].get("helper_return_observed", False),
            "helper_return_unbound_uncertain": activity["observation"].get("helper_return_unbound_uncertain", False),
            "submitted_forwards": counts["submitted"], "model_call_returns_observed": counts["model_returns"],
            "returned_forwards": counts["returned"], "completed_records": len(completed),
            "admitted_payload_bytes": counts["retained"],
            "raw_files": base.raw_file_inventory(output, cases), "error_type": type(error).__name__,
            "traceback_frames": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                 for frame in traceback.extract_tb(error.__traceback__)]})
        raise
    print(json.dumps({"status": "complete", "forwards": counts["submitted"], "raw_payload_bytes": counts["retained"],
                      "decision": result["decision"]}, allow_nan=False))


if __name__ == "__main__":
    main()
