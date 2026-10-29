"""Independently check saved public head-ablation evidence; no model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess


ACTIONS = ("clean", "noop", "candidate_zero", "control_zero", "clean_after")
LOGITS_BYTES = 50257 * 4
WITNESS_BYTES = 11 * 768 * 4
INVENTORY_PATH = "experiments/distilgpt2_copying_head_eval_v1.json"
INVENTORY_SHA256 = "34bbb68a5acb8615e856a20b77bc3ec93787cc82afa11c9dabb83d7c209baca9"
EXPOSURE_SHA256 = "e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344"
PROTOCOL_SHA256 = "b6ebfcc0ead38bb120eafbcfd62b29e026be127375e2bee0a0950e91059b10cc"
SCREEN_RESULT_SHA256 = "8914c014bb57e37bd662f73f9f87f1c2431ad6f04a084a2920e44b68b89f0e5d"
SCREEN_ATTENTION_SHA256 = "ba90feee7214d646c36cdcad27ef767232f7799ba85e3ca55ebd248f6c6056ee"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
WEIGHTS_SHA256 = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
SOURCE_PATHS = {INVENTORY_PATH, "docs/DISTILGPT2_COPY_HEAD_EVAL_V1.md",
                "scripts/copying_head_eval_v1.py", "scripts/copying_dev_v1.py", "docs/COPYING_HEAD_RUNNER_V1.md",
                "scripts/distilgpt2_intervention_dev_bootstrap.py", "looking_glass/__init__.py",
                "looking_glass/copy_head_intervention.py", "looking_glass/pretrained_exposures_v7.py",
                "looking_glass/pretrained.py", "tests/test_copy_head_intervention.py",
                "tests/test_copying_head_eval_v1.py", "results/distilgpt2-copying-dev-v1-20261028/run/result.json",
                "results/distilgpt2-copying-dev-v1-20261028/run/attentions.f32",
                "results/distilgpt2-copying-dev-v1-20261028/supervisor.json"}
NUMERIC_PROBE = {"input_preserved": True, "half_min_normal_bits": 4194304, "numpy_cast_preserved": True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stats(raw, target):
    values = struct.unpack("<50257f", raw)
    require(all(map(math.isfinite, values)), "nonfinite logits")
    maximum = max(values)
    target_logit = values[target]
    return {"target_logit": target_logit,
            "target_log_probability": target_logit - maximum - math.log(math.fsum(
                math.exp(value - maximum) for value in values)),
            "target_rank": 1 + sum(value > target_logit for value in values),
            "top_token_id": values.index(maximum)}


def check_witness(action, payload, reported):
    before, after, projection = (payload[name] for name in ("cproj_before", "cproj_after", "cproj_output"))
    before_bits, after_bits = struct.unpack("<8448I", before), struct.unpack("<8448I", after)
    require(all(map(math.isfinite, struct.unpack("<8448f", before)))
            and all(map(math.isfinite, struct.unpack("<8448f", projection))), "nonfinite witness")
    if action == "noop":
        require(before == after and reported is None, "invalid no-op witness")
        return
    head = {"candidate_zero": 10, "control_zero": 7}[action]
    start, end = 10 * 768 + head * 64, 10 * 768 + (head + 1) * 64
    require(before_bits[:start] == after_bits[:start] and before_bits[end:] == after_bits[end:],
            "untouched coordinates changed")
    require(all(value == 0 for value in after_bits[start:end]), "ablation was not delivered")
    values = struct.unpack("<8448f", before)[start:end]
    expected = {"layer": 3, "head": head, "position": 10, "channels": [head * 64, (head + 1) * 64],
                "pre_head_l2": math.sqrt(math.fsum(float(value) ** 2 for value in values)),
                "changed_float32_bit_patterns": sum(a != b for a, b in zip(before_bits[start:end],
                                                                            after_bits[start:end]))}
    require(reported == expected, "intervention metadata differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True, help="directory containing run and supervisor.json")
    parser.add_argument("--repository", type=Path, required=True, help="checkout containing the recorded revision")
    parser.add_argument("--source-revision", required=True, help="public source revision recorded before launch")
    args = parser.parse_args()
    attempt, run = args.attempt, args.attempt / "run"
    manifest_raw = (run / "manifest.json").read_bytes()
    result_raw = (run / "result.json").read_bytes()
    manifest, result = json.loads(manifest_raw), json.loads(result_raw)
    supervisor = json.loads((attempt / "supervisor.json").read_bytes())
    pre_raw = (attempt / "pre-run.json").read_bytes()
    pre = json.loads(pre_raw)
    require(supervisor["schema"] == "copying-head-process-observation-v1"
            and supervisor["process_exit_code"] == 0 and supervisor["interrupted"] is False
            and not supervisor["failure_file_present"]
            and not (run / "failure.json").exists(), "process did not complete cleanly")
    require(supervisor["run_sha256"]["manifest.json"] == digest(manifest_raw)
            and supervisor["run_sha256"]["result.json"] == digest(result_raw)
            and result["manifest_sha256"] == digest(manifest_raw)
            and supervisor["pre_run_sha256"] == digest(pre_raw), "manifest/result/pre-run digest mismatch")
    source = manifest["source"]
    require(manifest["schema"] == "distilgpt2-copy-head-run-v1"
            and result["schema"] == "distilgpt2-copy-head-result-v1"
            and re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is not None
            and supervisor["source_revision"] == source["revision"] == pre["source_revision"] == args.source_revision
            and set(source["sha256"]) == SOURCE_PATHS
            and pre["schema"] == "copying-head-launch-v1"
            and pre["protocol_sha256"] == PROTOCOL_SHA256
            and pre["inventory_sha256"] == INVENTORY_SHA256
            and pre["exposure_inventory_sha256"] == EXPOSURE_SHA256
            and pre["device"] == "cpu"
            and pre["environment"] == {"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1",
                 "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            and pre["intended_runtime"] == {"dtype": "float32", "attention_implementation": "eager",
                "use_cache": False, "intraop_threads": 1, "deterministic_algorithms": False,
                "float32_matmul_precision": "highest", "cpu_rng_seed": 20261029}
            and manifest["inventory_sha256"] == INVENTORY_SHA256
            and manifest["exposure_inventory_sha256"] == EXPOSURE_SHA256,
            "source, case or exposure binding differs")
    source_bytes = {}
    for path, expected in source["sha256"].items():
        raw = subprocess.check_output(["git", "show", source["revision"] + ":" + path], cwd=args.repository)
        require(digest(raw) == expected, "committed source bytes differ: " + path)
        source_bytes[path] = raw
    cases, rows = manifest["cases"], result["cases"]
    inventory = json.loads(source_bytes[INVENTORY_PATH])
    require(digest(source_bytes[INVENTORY_PATH]) == INVENTORY_SHA256
            and digest(source_bytes["docs/DISTILGPT2_COPY_HEAD_EVAL_V1.md"]) == PROTOCOL_SHA256
            and digest(source_bytes["results/distilgpt2-copying-dev-v1-20261028/run/result.json"]) == SCREEN_RESULT_SHA256
            and digest(source_bytes["results/distilgpt2-copying-dev-v1-20261028/run/attentions.f32"]) == SCREEN_ATTENTION_SHA256
            and cases == inventory["cases"] and manifest["actions"] == list(ACTIONS)
            and len(cases) == 32 and len(rows) == 160 and result["status"] == "complete"
            and result["submitted_forwards"] == result["returned_forwards"] == 160,
            "unexpected case or forward count")
    require(manifest["model_id"] == "distilbert/distilgpt2" and manifest["model_revision"] == MODEL_REVISION
            and manifest["weights_sha256"] == WEIGHTS_SHA256
            and manifest["initial_numeric_probe"] == NUMERIC_PROBE
            and manifest["discovery"] == {"result_sha256": SCREEN_RESULT_SHA256,
                "attention_sha256": SCREEN_ATTENTION_SHA256,
                "selected": {"layer": 3, "head": 10, "mean_attention_difference": 0.8940028093044626}},
            "model, numeric probe or development evidence differs")
    deps, runtime, checkpoint = manifest["dependencies"], result["runtime"], result["checkpoint"]
    require(deps["python"] == "3.11.15" and deps["torch"] == "2.8.0+cu128"
            and deps["numpy"] == "2.4.6" and deps["transformers"] == "4.57.3"
            and deps["gpt2_source_sha256"] == "b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80"
            and deps["conv1d_source_sha256"] == "e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86"
            and deps["byteorder"] == "little", "dependency identity differs")
    require(runtime["device"] == "cpu" and runtime["dtype"] == "float32"
            and runtime["intraop_threads"] == 1 and runtime["cpu_autocast"] is False
            and runtime["cuda_visibility"] == "-1" and runtime["cuda_initialized"] is False
            and runtime["attention_implementation"] == "eager" and runtime["use_cache"] is False
            and runtime["numeric_probe"] == NUMERIC_PROBE and runtime["model_eval"] is True
            and runtime["guard_blocked_calls"] == [] and runtime["cpu_rng_seed"] == 20261029
            and runtime["deterministic_algorithms"] is False
            and runtime["float32_matmul_precision"] == "highest", "recorded runtime differs")
    require(checkpoint["model_id"] == "distilbert/distilgpt2" and checkpoint["model_revision"] == MODEL_REVISION
            and checkpoint["weights_sha256"] == WEIGHTS_SHA256
            and checkpoint["asset_sha256"]["model.safetensors"] == WEIGHTS_SHA256
            and checkpoint["asset_sha256"]["tokenizer.json"] == "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
            and checkpoint["tokenizer_revision"] == MODEL_REVISION
            and checkpoint["device"] == "cpu" and checkpoint["dtype"] == "float32",
            "checkpoint or tokenizer identity differs")
    require(all(type(result[key]) is str and re.fullmatch(r"[0-9a-f]{64}", result[key]) is not None
                for key in ("model_state_sha256_before", "model_state_sha256_after",
                            "model_buffers_sha256_before", "model_buffers_sha256_after"))
            and result["model_state_sha256_before"] == result["model_state_sha256_after"]
            and result["model_state_sha256_before"] == json.loads(
                source_bytes["results/distilgpt2-copying-dev-v1-20261028/run/result.json"])["model_state_sha256_before"]
            and result["model_buffers_sha256_before"] == result["model_buffers_sha256_after"],
            "model state or buffer hashes differ")
    events = [json.loads(line) for line in (run / "attempt.jsonl").read_bytes().splitlines()]
    require(supervisor["run_sha256"]["attempt.jsonl"] == digest((run / "attempt.jsonl").read_bytes())
            and len(events) == 481 and events[0] == {"event": "runtime_ready", "cpu_rng_seed": 20261029,
                                                     "actual_runtime": runtime},
            "incomplete or mismatched event log")
    seen = []
    total = 0
    for index, case in enumerate(cases):
        raw = (run / (case["case_id"] + ".f32")).read_bytes()
        require(supervisor["run_sha256"][case["case_id"] + ".f32"] == digest(raw),
                "raw file digest differs")
        offset, clean, noop_input = 0, None, None
        for k, action in enumerate(ACTIONS):
            row = rows[index * 5 + k]
            require((row["case_id"], row["group"], row["variant"], row["target_id"], row["action"])
                    == (case["case_id"], case["group"], case["variant"], case["target_id"], action),
                    "case/action order differs")
            event_index = 1 + 3 * (index * 5 + k)
            require(events[event_index] == {"event": "start", "case_id": case["case_id"], "action": action}
                    and events[event_index + 1] == {"event": "raw_recorded", "case_id": case["case_id"],
                                                    "action": action, "raw": row["raw"]}
                    and events[event_index + 2] == {"event": "complete", **row},
                    "start/completion pairing differs")
            names = ["logits"] + ([] if action in ("clean", "clean_after") else
                                   ["cproj_before", "cproj_after", "cproj_output"])
            require(set(row["raw"]) == set(names), "raw payload names differ")
            payload = {}
            for name in names:
                item = row["raw"][name]
                expected_length = LOGITS_BYTES if name == "logits" else WITNESS_BYTES
                require(item["offset"] == offset and item["bytes"] == expected_length,
                        "raw payload offset or length differs")
                payload[name] = raw[offset:offset + expected_length]
                require(len(payload[name]) == expected_length and digest(payload[name]) == item["sha256"],
                        "raw payload digest differs")
                offset += expected_length
            require(row["score"] == stats(payload["logits"], case["target_id"]), "logit score differs")
            if action == "clean":
                clean = payload["logits"]
            elif action in ("noop", "clean_after"):
                require(clean == payload["logits"], "clean/no-op/post-clean logits differ")
            if action not in ("clean", "clean_after"):
                check_witness(action, payload, row["intervention"])
                if action == "noop":
                    noop_input = payload["cproj_before"]
                else:
                    require(noop_input == payload["cproj_before"], "pre-ablation projection input differs")
            else:
                require(row["intervention"] is None, "unexpected clean intervention metadata")
            seen.append(row)
        require(offset == len(raw), "raw file has a missing or extra tail")
        total += len(raw)
    require(total == manifest["raw_payload_bytes"] == result["raw_payload_bytes"] == 41896576,
            "raw payload total differs")
    families = []
    for group in range(8):
        grouped = [seen[(group * 4 + variant) * 5:(group * 4 + variant + 1) * 5] for variant in range(4)]
        row = {"group": group + 1}
        for name, action_index in (("candidate", 2), ("control", 3)):
            deltas = [case[action_index]["score"]["target_log_probability"] -
                      case[0]["score"]["target_log_probability"] for case in grouped]
            row[name] = {"case_deltas": deltas, "family_contrast": deltas[0] - math.fsum(deltas[1:]) / 3}
        row["candidate_minus_control"] = row["candidate"]["family_contrast"] - row["control"]["family_contrast"]
        families.append(row)
    expected = {"families": families,
                "mean_candidate": math.fsum(x["candidate"]["family_contrast"] for x in families) / 8,
                "mean_control": math.fsum(x["control"]["family_contrast"] for x in families) / 8,
                "mean_candidate_minus_control": math.fsum(x["candidate_minus_control"] for x in families) / 8}
    require(result["summary"] == expected, "family summary differs")
    stdout = (attempt / "stdout.bin").read_bytes()
    stderr = (attempt / "stderr.bin").read_bytes()
    require(len(stdout) == supervisor["stdout_bytes"] and digest(stdout) == supervisor["stdout_sha256"]
            and len(stderr) == supervisor["stderr_bytes"] and digest(stderr) == supervisor["stderr_sha256"]
            and json.loads(stdout) == {"status": "complete", "forwards": 160,
                                       "raw_payload_bytes": total, "summary": expected},
            "process output or observation differs")
    print(json.dumps({"status": "pass", "forwards": 160, "raw_payload_bytes": total,
                      "summary": expected, "scope": "saved-byte consistency; no model replay"},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
