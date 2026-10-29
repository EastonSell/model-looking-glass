"""Fixed public CPU follow-up with rotated head and position ablations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

if __name__ == "__main__" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("set CUDA_VISIBLE_DEVICES=-1 before starting")
if __name__ == "__main__" and "torch" in sys.modules:
    raise RuntimeError("torch must not be imported before the CPU backend environment snapshot")

CPU_ENV_NAMES = ("MKL_CBWR", "ATEN_CPU_CAPABILITY")
CPU_ENV_AT_TORCH_IMPORT = {name: {"present": name in os.environ, "value": os.environ.get(name)}
                           for name in CPU_ENV_NAMES}
EXPECTED_CPU_ENV = {name: {"present": False, "value": None} for name in CPU_ENV_NAMES}
if __name__ == "__main__" and CPU_ENV_AT_TORCH_IMPORT != EXPECTED_CPU_ENV:
    raise RuntimeError("MKL_CBWR and ATEN_CPU_CAPABILITY must be absent before importing torch")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network  # noqa: E402

GUARD_EVENTS = []
SAFE_QUERIES = {"cuda_is_available": 0, "cuda_device_count": 0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)

from scripts import copying_head_eval_v1 as base  # noqa: E402
from looking_glass.copy_head_intervention import HeadProjectionWitness, assert_no_forward_hooks  # noqa: E402
from looking_glass.copy_head_followup import ARMS, ARM_MAP, action_order, arm_witness, summarize  # noqa: E402

base.GUARD_EVENTS = GUARD_EVENTS
PROTOCOL_PATH = "docs/DISTILGPT2_COPY_HEAD_FOLLOWUP_V2.md"
CASE_PATH = "experiments/distilgpt2_copy_head_followup_v2.json"
# The separately published protocol, cases and both exposure successors must
# match these exact pins. Missing or changed artifacts block model forwarding.
PROTOCOL_SHA256 = "70616bb01aa7ccb8440da9dc503c274a3d5cb7b24dd3965fd985de6c645ca68c"
PROTOCOL_REVISION = "a9f29a5421ba40404708d48913c7d599be69a429"
CASE_SHA256 = "3831bfd1a48aaa3fcb67fbdb3cdc10f4422149b1fed90fd5267d3c73525255a8"
V8_EXPOSURE_SHA256 = "22b1392ed43568fcac574b55ae6cf0212766771d8439ac39e599f7aa112b298f"
V9_EXPOSURE_SHA256 = "e0c394786b316ea61b9f3697063eb86c7cf290d898c3ed84549880df8af91c4d"
V8_EXPOSURE_VERSION = "distilgpt2-development-exposures-v8"
V9_EXPOSURE_VERSION = "distilgpt2-development-exposures-v9"
PRIOR_RESULT_SHA256 = "e36732e0538f7b5a069fed337fd86eb725744663b3801d1d953907d0edb93a02"
PRIOR_SUPERVISOR_SHA256 = "2935b0dfbecf10a8171cd62d05a0856f86994c7cce975d9090022a2d74610c9a"
PRIOR_SOURCE_REVISION = "5458c5d54ddba5ca024e495e236519d376971583"
PRIOR_ROOT = "results/distilgpt2-copy-head-eval-v1-20261028"
MAX_RAW_BYTES = 81_000_000
EXPECTED_RAW_BYTES = 32 * (9 * base.LOGITS_BYTES + 21 * base.WITNESS_BYTES)
MAX_FORWARD_BYTES = base.LOGITS_BYTES + 3 * base.WITNESS_BYTES
SOURCES = ("scripts/copying_head_followup_v2.py", "scripts/copying_head_eval_v1.py",
           "scripts/copying_dev_v1.py", "scripts/distilgpt2_intervention_dev_bootstrap.py",
           "looking_glass/__init__.py", "looking_glass/copy_head_intervention.py",
           "looking_glass/copy_head_followup.py", "looking_glass/pretrained.py",
           "looking_glass/pretrained_exposures.py", "looking_glass/pretrained_exposures_v2.py",
           "looking_glass/pretrained_exposures_v3.py", "looking_glass/pretrained_exposures_v4.py",
           "looking_glass/pretrained_exposures_v5.py", "looking_glass/pretrained_exposures_v6.py",
           "looking_glass/pretrained_exposures_v7.py", "looking_glass/pretrained_exposures_v8.py",
           "looking_glass/pretrained_exposures_v9.py", PROTOCOL_PATH, CASE_PATH,
           base.INVENTORY_PATH, "docs/DISTILGPT2_COPY_HEAD_EVAL_V1.md",
           "docs/COPYING_HEAD_FOLLOWUP_RUNNER_V2.md", "tests/test_copy_head_followup.py",
           "tests/test_copying_head_followup_v2.py", "tests/test_copy_head_intervention.py",
           PRIOR_ROOT + "/run/result.json",
           PRIOR_ROOT + "/supervisor.json", base.SCREEN_ROOT + "/run/result.json",
           base.SCREEN_ROOT + "/run/attentions.f32", base.SCREEN_ROOT + "/supervisor.json")


def check_cpu_backend_environment():
    current = {name: {"present": name in os.environ, "value": os.environ.get(name)}
               for name in CPU_ENV_NAMES}
    if CPU_ENV_AT_TORCH_IMPORT != EXPECTED_CPU_ENV or current != CPU_ENV_AT_TORCH_IMPORT:
        raise RuntimeError("CPU backend environment changed or was not frozen before torch import")
    return current


def runtime(model):
    return {**base.runtime(model), "cpu_backend_environment": check_cpu_backend_environment()}


def validate_cases(data):
    if (type(data) is not dict or data.get("schema") != "distilgpt2-copy-head-followup-cases-v2"
            or data.get("model_id") != base.MODEL_ID or data.get("model_revision") != base.MODEL_REVISION
            or data.get("protocol_revision") != PROTOCOL_REVISION
            or data.get("protocol_sha256") != PROTOCOL_SHA256
            or data.get("known_exposure_baseline_v7_sha256") != base.EXPOSURE_SHA256
            or data.get("known_exposure_current_version") != V8_EXPOSURE_VERSION
            or data.get("known_exposure_current_sha256") != V8_EXPOSURE_SHA256
            or data.get("tokenizer_sha256") != "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
            or data.get("arms") != [{"layer": 3, "head": head, "position": position}
                                     for _, head, position in ARMS]
            or data.get("rotation") != "(zero_based_family + 2 * variant) % 6, left"
            or data.get("expected_forwards") != 288 or data.get("raw_tensor_bytes") != EXPECTED_RAW_BYTES
            or data.get("permutations") != [list(row) for row in base.PERMUTATIONS]):
        raise ValueError("case inventory does not match the fixed follow-up")
    groups, cases = data.get("groups"), data.get("cases")
    if type(groups) is not list or len(groups) != 8 or type(cases) is not list or len(cases) != 32:
        raise ValueError("expected eight public groups and 32 cases")
    seen = []
    for group_index, group in enumerate(groups):
        if type(group) is not dict or set(group) != {"words", "token_ids"}:
            raise ValueError("malformed group")
        words, tokens = group["words"], group["token_ids"]
        if (type(words) is not list or len(words) != 6
                or any(type(word) is not str or not word or len(word) > 80 for word in words)
                or type(tokens) is not list or len(tokens) != 6
                or any(type(token) is not int or not 0 <= token < base.VOCAB for token in tokens)):
            raise ValueError("malformed words or token IDs")
        seen.extend(tokens)
        for variant, permutation in enumerate(base.PERMUTATIONS):
            indices = permutation + (0, 1, 2, 3, 4)
            expected = {"case_id": f"copy-followup-{group_index + 1:02d}-{variant}",
                        "group": group_index + 1, "variant": variant,
                        "text": "".join(" " + words[index] for index in indices),
                        "token_ids": [tokens[index] for index in indices], "target_id": tokens[5]}
            if cases[4 * group_index + variant] != expected:
                raise ValueError("expanded case differs from its group")
    _, prior_cases = base.inventory()
    previous = {value for _, tokens in base.GROUPS for value in tokens}
    previous.update(value for row in prior_cases for value in row["token_ids"])
    previous.update(row["target_id"] for row in prior_cases)
    if len(seen) != 48 or len(set(seen)) != 48 or set(seen) & previous:
        raise ValueError("new token IDs must be distinct from prior public screens")
    return cases


def inventory():
    if (PROTOCOL_SHA256 is None or CASE_SHA256 is None
            or V8_EXPOSURE_SHA256 is None or V9_EXPOSURE_SHA256 is None):
        raise RuntimeError("the protocol, cases and exposure reservation are not all frozen")
    protocol = (ROOT / PROTOCOL_PATH).read_bytes()
    case_data = (ROOT / CASE_PATH).read_bytes()
    if base.sha(protocol) != PROTOCOL_SHA256 or base.sha(case_data) != CASE_SHA256:
        raise RuntimeError("the frozen follow-up protocol or cases changed")
    return validate_cases(json.loads(case_data))


def check_exposure(cases):
    if V8_EXPOSURE_SHA256 is None or V9_EXPOSURE_SHA256 is None:
        raise RuntimeError("the reviewed exposure reservation is not bound")
    from looking_glass import pretrained_exposures_v7 as v7, pretrained_exposures_v8 as v8
    from looking_glass import pretrained_exposures_v9 as v9
    if (v7.inventory_digest() != base.EXPOSURE_SHA256 or v8.INVENTORY_VERSION != V8_EXPOSURE_VERSION
            or v8.inventory_digest() != V8_EXPOSURE_SHA256 or v9.INVENTORY_VERSION != V9_EXPOSURE_VERSION
            or v9.inventory_digest() != V9_EXPOSURE_SHA256):
        raise RuntimeError("exposure reservation digest changed")
    proposed = tuple(tuple(row["token_ids"]) for row in cases)
    old = tuple(tuple(row["token_ids"]) for row in v8.inventory_payload()["records"])
    reject_prefix_overlap(proposed, old)
    if (proposed != v9.FOLLOWUP_ROWS
            or tuple(row["target_id"] for row in cases)
            != tuple(v9.FOLLOWUP_GROUPS[index // 4][5] for index in range(32))):
        raise RuntimeError("cases differ from the reviewed exposure reservation")
    return {"v7": base.EXPOSURE_SHA256, "v8": V8_EXPOSURE_SHA256, "v9": V9_EXPOSURE_SHA256}


def reject_prefix_overlap(proposed, old):
    def related(left, right):
        return left[:len(right)] == right or right[:len(left)] == left
    if (any(not row for row in proposed) or any(not row for row in old)
            or any(related(row, prior) for row in proposed for prior in old)
            or any(related(left, right) for index, left in enumerate(proposed)
                   for right in proposed[index + 1:])):
        raise RuntimeError("public rows overlap a known row or each other by full-row prefix")


def verify_previous():
    discovery = base.verify_discovery()
    prior_screen = json.loads((ROOT / base.SCREEN_ROOT / "run/result.json").read_bytes())
    if [(row["layer"], row["head"]) for row in prior_screen["attention_ranking"][:2]] != [(3, 10), (3, 2)]:
        raise RuntimeError("saved discovery ranking no longer selects H10 and H2")
    directory = ROOT / PRIOR_ROOT
    raw = (directory / "run/result.json").read_bytes()
    supervisor_raw = (directory / "supervisor.json").read_bytes()
    supervisor = json.loads(supervisor_raw)
    result = json.loads(raw)
    if (base.sha(raw) != PRIOR_RESULT_SHA256 or base.sha(supervisor_raw) != PRIOR_SUPERVISOR_SHA256
            or supervisor.get("schema") != "copying-head-process-observation-v1"
            or supervisor.get("source_revision") != PRIOR_SOURCE_REVISION
            or supervisor.get("interrupted") is not False or supervisor.get("process_exit_code") != 0
            or supervisor.get("failure_file_present") is not False
            or supervisor.get("run_sha256", {}).get("result.json") != PRIOR_RESULT_SHA256
            or result.get("status") != "complete" or result.get("submitted_forwards") != 160
            or result.get("returned_forwards") != 160
            or result.get("model_state_sha256_before") != result.get("model_state_sha256_after")
            or result.get("model_buffers_sha256_before") != result.get("model_buffers_sha256_after")):
        raise RuntimeError("prior public head evaluation is not the fixed complete result")
    return {"discovery": discovery, "prior_result_sha256": PRIOR_RESULT_SHA256,
            "prior_supervisor_sha256": PRIOR_SUPERVISOR_SHA256, "prior_source_revision": PRIOR_SOURCE_REVISION,
            "prior_candidate_mean": result["summary"]["mean_candidate"],
            "prior_control_mean": result["summary"]["mean_control"]}


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=30)
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    tree = git("show", "-s", "--format=%T", revision).decode("ascii").strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("source checkout is not clean")
    sources = {}
    for path in SOURCES:
        disk = (ROOT / path).read_bytes()
        if git("show", f"{revision}:{path}") != disk:
            raise RuntimeError(f"recorded source differs from the commit: {path}")
        sources[path] = base.sha(disk)
    return {"revision": revision, "tree": tree, "sha256": sources}


def run_forward(model, ids, action):
    assert_no_forward_hooks(model)
    if action not in ("clean", "noop", "clean_after") and action not in ARM_MAP:
        raise ValueError("unknown action")
    projection = model.transformer.h[3].attn.c_proj
    witness = None
    with torch.inference_mode():
        if action in ("clean", "clean_after"):
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           output_attentions=False, output_hidden_states=False, return_dict=True)
        else:
            head, position = ARM_MAP[action] if action in ARM_MAP else (None, None)
            mode = "zero" if action in ARM_MAP else "noop"
            with HeadProjectionWitness(projection, sequence_length=11, width=768, heads=12,
                                       mode=mode, head=head, position=position) as witness:
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               output_attentions=False, output_hidden_states=False, return_dict=True)
    assert_no_forward_hooks(model)
    logits = result.logits
    if (result.past_key_values is not None or result.attentions is not None or result.hidden_states is not None
            or type(logits) is not torch.Tensor or logits.device.type != "cpu"
            or logits.dtype != torch.float32 or tuple(logits.shape) != (1, 11, base.VOCAB)
            or not bool(torch.isfinite(logits).all())):
        raise RuntimeError("model returned an invalid logit tensor")
    payload = {"logits": base._raw(logits[0, -1], base.LOGITS_BYTES)}
    if witness is not None:
        payload.update({"cproj_before": base._raw(witness.before, base.WITNESS_BYTES),
                        "cproj_after": base._raw(witness.after, base.WITNESS_BYTES),
                        "cproj_output": base._raw(witness.output, base.WITNESS_BYTES)})
    if sum(len(value) for value in payload.values()) > MAX_FORWARD_BYTES:
        raise RuntimeError("retained forward payload exceeds its cap")
    return payload


def persist_action(log, raw_file, case, action, payload, clean_logits, noop_input, postcheck, marker):
    raw = base._write_payload(raw_file, payload)
    base._append_event(log, {"event": "raw_recorded", "case_id": case["case_id"],
                             "action": action, "raw": raw})
    marker["recorded"] = True
    postcheck()
    if action in ("noop", "clean_after") and payload["logits"] != clean_logits:
        raise RuntimeError("unhooked clean and no-op logit bits differ")
    if action in ARM_MAP and payload["cproj_before"] != noop_input:
        raise RuntimeError("pre-ablation projection input differs from no-op")
    record = {"case_id": case["case_id"], "group": case["group"], "variant": case["variant"],
              "target_id": case["target_id"], "action": action,
              "score": base.statistics(payload["logits"], case["target_id"]),
              "intervention": arm_witness(action, payload), "raw": raw}
    base._append_event(log, {"event": "complete", **record})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (not sys.dont_write_bytecode or not sys.pycache_prefix
            or not Path(sys.pycache_prefix).is_absolute() or any(Path(sys.pycache_prefix).iterdir())
            or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise RuntimeError("use -B, a fresh absolute -X pycache_prefix and offline flags")
    source = source_identity()
    cpu_environment = check_cpu_backend_environment()
    dependencies = base.dependency_identity()
    cases = inventory()
    previous = verify_previous()
    exposure = check_exposure(cases)
    if EXPECTED_RAW_BYTES != 80_604_288 or EXPECTED_RAW_BYTES > MAX_RAW_BYTES:
        raise RuntimeError("declared raw payload cap is inconsistent")
    output = args.output_dir
    if not output.is_absolute() or not args.cache_dir.is_absolute():
        raise ValueError("cache and output paths must be absolute")
    parent = output.parent.resolve(strict=True)
    if ROOT == parent or ROOT in parent.parents or output.exists():
        raise ValueError("output must be a new directory outside the checkout")
    if shutil.disk_usage(parent).free < MAX_RAW_BYTES + 16_777_216:
        raise OSError("insufficient space for the declared raw payload")
    pre_probe = base.numeric_policy(initialize=True)
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    output.mkdir()
    manifest = {"schema": "distilgpt2-copy-head-followup-run-v2", "source": source,
                "dependencies": dependencies, "protocol_sha256": PROTOCOL_SHA256,
                "cases_sha256": CASE_SHA256, "exposure_inventories": exposure,
                "previous": previous, "cases": cases, "arms": ARMS,
                "rotation": "(zero_based_family + 2 * variant) % 6",
                "model_id": base.MODEL_ID, "model_revision": base.MODEL_REVISION,
                "weights_sha256": base.WEIGHTS_SHA256, "raw_payload_bytes": EXPECTED_RAW_BYTES,
                "max_raw_bytes": MAX_RAW_BYTES, "max_retained_bytes_per_forward": MAX_FORWARD_BYTES,
                "initial_numeric_probe": pre_probe,
                "cpu_backend_environment_at_torch_import": cpu_environment,
                "intended_runtime": {"device": "cpu", "dtype": "float32", "model_eval": True,
                                     "attention_implementation": "eager", "use_cache": False,
                                     "intraop_threads": 1, "interop_threads": torch.get_num_interop_threads(),
                                     "deterministic_algorithms": False, "mkldnn_enabled": torch.backends.mkldnn.enabled,
                                     "float32_matmul_precision": "highest", "cpu_rng_seed": base.CPU_RNG_SEED},
                "counter_semantics": {"submitted_forwards": "attempts entering the helper; an early error can precede a model call",
                                      "returned_forwards": "complete helper payloads; a model call may return before helper validation fails",
                                      "payload_recorded": "returned payload and its raw_recorded event were durably written"},
                "evidence_status": "public disjoint exploratory; not a blind held-out evaluation"}
    manifest_hash = base.durable_json(output / "manifest.json", manifest)
    completed, submitted, returned, retained = [], 0, 0, 0
    phase, case_id, action, payload_returned = "model_load", None, None, False
    marker = {"recorded": False}
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        if bundle.metadata["weights_sha256"] != base.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint weight hash changed")
        for case in cases:
            if bundle.tokenizer(case["text"], add_special_tokens=False)["input_ids"] != case["token_ids"]:
                raise RuntimeError("pinned tokenizer does not reproduce a public case")
        model = bundle.model
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        model_before, buffers_before = base.state_hash(model), base.buffer_hash(model)
        runtime_before = runtime(model)
        assert_no_forward_hooks(model)
        with (output / "attempt.jsonl").open("xb") as log:
            base._append_event(log, {"event": "runtime_ready", "cpu_rng_seed": base.CPU_RNG_SEED,
                                     "actual_runtime": runtime_before})
            for case in cases:
                case_id = case["case_id"]
                ids = torch.tensor([case["token_ids"]], dtype=torch.int64, device="cpu")
                clean_logits, noop_input = None, None
                with (output / (case_id + ".f32")).open("xb") as raw_file:
                    for action in action_order(case["group"] - 1, case["variant"]):
                        phase, payload_returned, marker = "forward", False, {"recorded": False}
                        base._append_event(log, {"event": "start", "case_id": case_id, "action": action})
                        if runtime(model) != runtime_before:
                            raise RuntimeError("runtime changed before a forward")
                        submitted += 1
                        payload = run_forward(model, ids, action)
                        returned += 1
                        payload_returned = True
                        retained += sum(map(len, payload.values()))
                        if retained > EXPECTED_RAW_BYTES:
                            raise RuntimeError("aggregate raw payload exceeds its limit")
                        phase = "persist_and_validate"
                        def postcheck():
                            if ids.tolist() != [case["token_ids"]]:
                                raise RuntimeError("model modified the input IDs")
                            if runtime(model) != runtime_before:
                                raise RuntimeError("runtime or CPU RNG changed after a forward")
                        record = persist_action(log, raw_file, case, action, payload,
                                                clean_logits, noop_input, postcheck, marker)
                        if action == "clean":
                            clean_logits = payload["logits"]
                        if action == "noop":
                            noop_input = payload["cproj_before"]
                        completed.append(record)
        phase = "verification"
        if (len(completed) != 288 or submitted != 288 or returned != 288
                or retained != EXPECTED_RAW_BYTES or base.state_hash(model) != model_before
                or base.buffer_hash(model) != buffers_before or runtime(model) != runtime_before
                or source_identity() != source or GUARD_EVENTS or torch.cuda.is_initialized()):
            raise RuntimeError("final count, model, runtime or source verification failed")
        assert_no_forward_hooks(model)
        result = {"schema": "distilgpt2-copy-head-followup-result-v2", "status": "complete",
                  "manifest_sha256": manifest_hash, "checkpoint": bundle.metadata,
                  "model_state_sha256_before": model_before, "model_state_sha256_after": base.state_hash(model),
                  "model_buffers_sha256_before": buffers_before,
                  "model_buffers_sha256_after": base.buffer_hash(model), "runtime": runtime_before,
                  "guard_safe_queries": SAFE_QUERIES, "submitted_forwards": submitted,
                  "returned_forwards": returned, "raw_payload_bytes": retained,
                  "cases": completed, "summary": summarize(completed)}
        base.durable_json(output / "result.json", result)
    except BaseException as error:
        attempt_log = output / "attempt.jsonl"
        if attempt_log.is_file():
            try:
                with attempt_log.open("ab") as log:
                    base._append_event(log, {"event": "failed", "case_id": case_id, "action": action,
                                             "phase": phase, "error_type": type(error).__name__,
                                             "payload_returned": payload_returned,
                                             "payload_recorded": marker["recorded"]})
            except OSError:
                pass
        base.durable_json(output / "failure.json", {"schema": "distilgpt2-copy-head-followup-failure-v2",
                         "status": "incomplete", "manifest_sha256": manifest_hash,
                         "phase": phase, "case_id": case_id, "action": action,
                         "payload_returned": payload_returned, "payload_recorded": marker["recorded"],
                         "submitted_forwards": submitted, "returned_forwards": returned,
                         "completed_records": len(completed), "admitted_payload_bytes": retained,
                         "raw_files": base.raw_file_inventory(output, cases), "error_type": type(error).__name__,
                         "traceback_frames": [{"file": Path(frame.filename).name, "line": frame.lineno,
                                               "function": frame.name}
                                              for frame in traceback.extract_tb(error.__traceback__)]})
        raise
    print(json.dumps({"status": "complete", "forwards": submitted, "raw_payload_bytes": retained,
                      "summary": result["summary"]}, allow_nan=False))


if __name__ == "__main__":
    main()
