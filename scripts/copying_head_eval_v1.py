"""Public, fixed DistilGPT-2 head-ablation follow-up; CPU only, not a holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

if __name__ == "__main__" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("set CUDA_VISIBLE_DEVICES=-1 before starting")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network  # noqa: E402

GUARD_EVENTS = []
SAFE_QUERIES = {"cuda_is_available": 0, "cuda_device_count": 0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)

import numpy as np  # noqa: E402
import transformers  # noqa: E402
from transformers.models.gpt2 import modeling_gpt2  # noqa: E402
from transformers.pytorch_utils import Conv1D  # noqa: E402

from looking_glass.copy_head_intervention import (  # noqa: E402
    HeadProjectionWitness, assert_no_forward_hooks,
)
from looking_glass.pretrained import load_distilgpt2  # noqa: E402
from scripts.copying_dev_v1 import (  # noqa: E402
    GROUPS, PERMUTATIONS, canonical, durable_json, numeric_policy, sha,
    state_hash, statistics,
)


INVENTORY_PATH = "experiments/distilgpt2_copying_head_eval_v1.json"
INVENTORY_SHA256 = "34bbb68a5acb8615e856a20b77bc3ec93787cc82afa11c9dabb83d7c209baca9"
EXPOSURE_SHA256 = "e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344"
GPT2_SOURCE_SHA256 = "b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80"
CONV1D_SOURCE_SHA256 = "e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86"
MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
WEIGHTS_SHA256 = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
SCREEN_REVISION = "55d11a53eb34a13eb6b14e19a5bd0c47172edda5"
SCREEN_RESULT_SHA256 = "8914c014bb57e37bd662f73f9f87f1c2431ad6f04a084a2920e44b68b89f0e5d"
CONTROL_DIGEST = "7a864639743da70e88dcd2c5f3b6bf79cf2942c70a27d8889395fedc57c309fc"
ACTIONS = ("clean", "noop", "candidate_zero", "control_zero", "clean_after")
LENGTH, WIDTH, HEADS, VOCAB = 11, 768, 12, 50257
LOGITS_BYTES = VOCAB * 4
WITNESS_BYTES = LENGTH * WIDTH * 4
MAX_FORWARD_RETAINED_BYTES = LOGITS_BYTES + 3 * WITNESS_BYTES
EXPECTED_RAW_BYTES = 32 * (5 * LOGITS_BYTES + 9 * WITNESS_BYTES)
MAX_RAW_BYTES = 42_000_000
CPU_RNG_SEED = 20261029
SCREEN_ROOT = "results/distilgpt2-copying-dev-v1-20261028"
SOURCES = ("scripts/copying_head_eval_v1.py", "scripts/copying_dev_v1.py",
           "scripts/distilgpt2_intervention_dev_bootstrap.py", "looking_glass/__init__.py",
           "looking_glass/copy_head_intervention.py", "looking_glass/pretrained.py",
           "looking_glass/pretrained_exposures_v7.py", INVENTORY_PATH,
           "docs/DISTILGPT2_COPY_HEAD_EVAL_V1.md", "docs/COPYING_HEAD_RUNNER_V1.md",
           "tests/test_copy_head_intervention.py",
           "tests/test_copying_head_eval_v1.py", SCREEN_ROOT + "/run/result.json",
           SCREEN_ROOT + "/run/attentions.f32", SCREEN_ROOT + "/supervisor.json")


def selected_control():
    raw = b"MLG-copy-head-control-v1\0" + (20261029).to_bytes(8, "big")
    digest = hashlib.sha256(raw).digest()
    choices = [head for head in range(12) if head != 10]
    return digest.hex(), choices[int.from_bytes(digest, "big") % len(choices)]


def validate_inventory(data):
    """Validate the published rows; this function does not create new cases."""
    if type(data) is not dict:
        raise ValueError("inventory must be an object")
    candidate = {"layer": 3, "head": 10, "channel_start": 640, "channel_end_exclusive": 704}
    control = data.get("layer_matched_random_control")
    if (data.get("schema") != "distilgpt2-copy-head-eval-cases-v1"
            or data.get("model_id") != MODEL_ID or data.get("model_revision") != MODEL_REVISION
            or data.get("screen_source_revision") != SCREEN_REVISION
            or data.get("screen_result_sha256") != SCREEN_RESULT_SHA256
            or data.get("candidate") != candidate
            or type(control) is not dict or control.get("layer") != 3
            or control.get("head") != 7 or control.get("channel_start") != 448
            or control.get("channel_end_exclusive") != 512
            or control.get("seed") != 20261029 or control.get("sha256") != CONTROL_DIGEST
            or selected_control() != (CONTROL_DIGEST, 7)
            or data.get("actions") != list(ACTIONS) or data.get("expected_forwards") != 160
            or data.get("permutations") != [list(row) for row in PERMUTATIONS]):
        raise ValueError("inventory does not match the fixed protocol")
    groups, cases = data.get("groups"), data.get("cases")
    if type(groups) is not list or len(groups) != 8 or type(cases) is not list or len(cases) != 32:
        raise ValueError("expected eight groups and 32 public cases")
    seen = []
    for group_index, group in enumerate(groups):
        if type(group) is not dict or set(group) != {"words", "token_ids"}:
            raise ValueError("malformed group")
        words, tokens = group["words"], group["token_ids"]
        if (type(words) is not list or len(words) != 6
                or any(type(word) is not str or not word or len(word) > 80 for word in words)
                or type(tokens) is not list or len(tokens) != 6
                or any(type(token) is not int or not 0 <= token < VOCAB for token in tokens)):
            raise ValueError("malformed group words or token IDs")
        seen.extend(tokens)
        for variant, permutation in enumerate(PERMUTATIONS):
            indices = permutation + (0, 1, 2, 3, 4)
            expected = {"case_id": f"copy-eval-{group_index + 1:02d}-{variant}",
                        "group": group_index + 1, "variant": variant,
                        "text": "".join(" " + words[index] for index in indices),
                        "token_ids": [tokens[index] for index in indices], "target_id": tokens[5]}
            if cases[4 * group_index + variant] != expected:
                raise ValueError("expanded case differs from its fixed group")
    discovery = {value for _, tokens in GROUPS for value in tokens}
    if len(seen) != 48 or len(set(seen)) != 48 or set(seen) & discovery:
        raise ValueError("evaluation token IDs must be distinct from the discovery screen")
    return cases


def inventory():
    raw = (ROOT / INVENTORY_PATH).read_bytes()
    if sha(raw) != INVENTORY_SHA256:
        raise RuntimeError("fixed public case inventory hash changed")
    data = json.loads(raw)
    return data, validate_inventory(data)


def check_exposure(cases):
    from looking_glass import pretrained_exposures_v7 as exposure
    if exposure.inventory_digest() != EXPOSURE_SHA256:
        raise RuntimeError("reviewed exposure inventory digest changed")
    if (tuple(tuple(row["token_ids"]) for row in cases) != exposure.PUBLIC_EVALUATION_ROWS
            or tuple(row["target_id"] for row in cases)
            != tuple(exposure.PUBLIC_EVALUATION_GROUPS[index // 4][5] for index in range(32))):
        raise RuntimeError("public cases differ from the exposure reservation")
    return EXPOSURE_SHA256


def verify_discovery():
    """Bind the chosen head to the saved screen and its raw attention values."""
    directory = ROOT / SCREEN_ROOT
    raw_result = (directory / "run/result.json").read_bytes()
    raw_attention = (directory / "run/attentions.f32").read_bytes()
    supervisor = json.loads((directory / "supervisor.json").read_bytes())
    result = json.loads(raw_result)
    if (sha(raw_result) != SCREEN_RESULT_SHA256 or supervisor.get("process_exit_code") != 0
            or supervisor.get("failure_file_present") is not False
            or supervisor.get("source_revision") != SCREEN_REVISION
            or supervisor.get("run_sha256", {}).get("result.json") != SCREEN_RESULT_SHA256
            or supervisor.get("run_sha256", {}).get("attentions.f32") != sha(raw_attention)
            or result.get("source_revision") != SCREEN_REVISION
            or result.get("attentions_sha256") != sha(raw_attention)
            or result.get("submitted_forwards") != 33 or result.get("root_outputs_returned") != 33
            or result.get("captured_logits_bitwise_match") is not True
            or result.get("model_state_sha256_before") != result.get("model_state_sha256_after")
            or len(raw_attention) != 32 * 6 * 12 * 11 * 11 * 4):
        raise RuntimeError("saved discovery evidence or supervisor record differs")
    attention = np.frombuffer(raw_attention, dtype="<f4").reshape(32, 6, 12, 11, 11)
    if not bool(np.isfinite(attention).all()):
        raise RuntimeError("saved discovery attention is nonfinite")
    ranking = sorted(({"layer": layer, "head": head,
                       "mean_attention_difference": math.fsum(
                           float(attention[4 * group, layer, head, -1, 5])
                           - float(attention[4 * group + control, layer, head, -1, 5])
                           for group in range(8) for control in (1, 2, 3)) / 24}
                      for layer in range(6) for head in range(12)),
                     key=lambda row: (-row["mean_attention_difference"], row["layer"], row["head"]))
    if (result.get("attention_ranking") != ranking
            or ranking[0]["layer"] != 3 or ranking[0]["head"] != 10):
        raise RuntimeError("raw discovery ranking does not select layer 3 head 10")
    return {"result_sha256": SCREEN_RESULT_SHA256, "attention_sha256": sha(raw_attention),
            "selected": ranking[0]}


def buffer_hash(model):
    """Hash every named buffer, including nonpersistent causal-mask buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.named_buffers(remove_duplicate=False)):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or not tensor.is_contiguous():
            raise RuntimeError("unexpected model buffer")
        header = canonical({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        data = memoryview(tensor.detach().numpy()).cast("B")
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


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
        sources[path] = sha(disk)
    return {"revision": revision, "tree": tree, "sha256": sources}


def dependency_identity():
    gpt2_path = Path(modeling_gpt2.__file__).resolve(strict=True)
    conv_path = Path(sys.modules[Conv1D.__module__].__file__).resolve(strict=True)
    if (sys.version_info[:3] != (3, 11, 15) or torch.__version__ != "2.8.0+cu128"
            or np.__version__ != "2.4.6" or transformers.__version__ != "4.57.3"
            or sha(gpt2_path.read_bytes()) != GPT2_SOURCE_SHA256
            or sha(conv_path.read_bytes()) != CONV1D_SOURCE_SHA256 or sys.byteorder != "little"):
        raise RuntimeError("runtime or GPT-2 source differs from the fixed CPU protocol")
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "transformers": transformers.__version__, "gpt2_module": modeling_gpt2.__name__,
            "gpt2_source_sha256": GPT2_SOURCE_SHA256, "conv1d_module": Conv1D.__module__,
            "conv1d_source_sha256": CONV1D_SOURCE_SHA256, "system": platform.system(),
            "machine": platform.machine(), "byteorder": sys.byteorder}


def runtime(model):
    attention = model.transformer.h[3].attn
    projection = attention.c_proj
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized()
            or GUARD_EVENTS or torch.get_num_threads() != 1 or torch.is_autocast_enabled("cpu")
            or torch.are_deterministic_algorithms_enabled()
            or torch.get_float32_matmul_precision() != "highest"
            or torch.get_default_dtype() != torch.float32
            or torch.random.default_generator.initial_seed() != CPU_RNG_SEED
            or type(model) is not modeling_gpt2.GPT2LMHeadModel
            or type(attention) is not modeling_gpt2.GPT2Attention or type(projection) is not Conv1D
            or model.forward.__func__ is not modeling_gpt2.GPT2LMHeadModel.forward
            or attention.forward.__func__ is not modeling_gpt2.GPT2Attention.forward
            or projection.forward.__func__ is not Conv1D.forward
            or model.training or any(module.training for module in model.modules())
            or model.config._attn_implementation != "eager" or model.config.use_cache is not False
            or model.config.n_embd != WIDTH or model.config.n_head != HEADS
            or len(model.transformer.h) != 6 or model.config.vocab_size != VOCAB
            or attention.num_heads != HEADS or attention.head_dim != 64 or attention.pruned_heads
            or attention.layer_idx != 3 or attention.is_cross_attention
            or not attention.scale_attn_weights or attention.scale_attn_by_inverse_layer_idx
            or attention.reorder_and_upcast_attn
            or tuple(projection.weight.shape) != (WIDTH, WIDTH)
            or tuple(projection.bias.shape) != (WIDTH,)
            or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                   for parameter in model.parameters())):
        raise RuntimeError("model or runtime differs from the fixed CPU protocol")
    return {"numeric_probe": numeric_policy(), "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "model_config_sha256": sha(canonical(model.config.to_dict())),
            "cpu_rng_state_sha256": sha(torch.random.default_generator.get_state().numpy().tobytes()),
            "cpu_rng_seed": CPU_RNG_SEED, "device": "cpu", "dtype": "float32", "cpu_autocast": False,
            "model_eval": True, "guard_blocked_calls": list(GUARD_EVENTS),
            "cuda_visibility": "-1", "cuda_initialized": False,
            "attention_implementation": "eager", "use_cache": False}


def _raw(tensor, expected_bytes):
    raw = tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes()
    if len(raw) != expected_bytes:
        raise RuntimeError("retained tensor exceeds its fixed shape")
    return raw


def run_forward(model, ids, action):
    if action not in ACTIONS:
        raise ValueError("unknown action")
    assert_no_forward_hooks(model)
    projection = model.transformer.h[3].attn.c_proj
    spec = {"noop": ("noop", None), "candidate_zero": ("zero", 10),
            "control_zero": ("zero", 7)}.get(action)
    witness = None
    with torch.inference_mode():
        if spec is None:
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           output_attentions=False, output_hidden_states=False, return_dict=True)
        else:
            with HeadProjectionWitness(projection, sequence_length=LENGTH, width=WIDTH,
                                       heads=HEADS, mode=spec[0], head=spec[1]) as witness:
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               output_attentions=False, output_hidden_states=False, return_dict=True)
    assert_no_forward_hooks(model)
    logits = result.logits
    if (result.past_key_values is not None or result.attentions is not None or result.hidden_states is not None
            or type(logits) is not torch.Tensor or logits.device.type != "cpu"
            or logits.dtype != torch.float32 or tuple(logits.shape) != (1, LENGTH, VOCAB)
            or not bool(torch.isfinite(logits).all())):
        raise RuntimeError("model returned an invalid full logit tensor")
    payload = {"logits": _raw(logits[0, -1], LOGITS_BYTES)}
    if witness is not None:
        payload.update({"cproj_before": _raw(witness.before, WITNESS_BYTES),
                        "cproj_after": _raw(witness.after, WITNESS_BYTES),
                        "cproj_output": _raw(witness.output, WITNESS_BYTES)})
    if sum(len(value) for value in payload.values()) > MAX_FORWARD_RETAINED_BYTES:
        raise RuntimeError("retained forward payload exceeds its cap")
    return payload


def action_witness(action, payload):
    if action not in ("candidate_zero", "control_zero"):
        return None
    head = 10 if action == "candidate_zero" else 7
    before = np.frombuffer(payload["cproj_before"], dtype="<f4").reshape(1, LENGTH, WIDTH)
    before_bits = np.frombuffer(payload["cproj_before"], dtype="<u4").reshape(1, LENGTH, WIDTH)
    after_bits = np.frombuffer(payload["cproj_after"], dtype="<u4").reshape(1, LENGTH, WIDTH)
    start, end = head * 64, (head + 1) * 64
    values = before[0, -1, start:end].astype("<f8")
    return {"layer": 3, "head": head, "position": LENGTH - 1, "channels": [start, end],
            "pre_head_l2": math.sqrt(math.fsum(float(value) ** 2 for value in values)),
            "changed_float32_bit_patterns": int(np.count_nonzero(
                before_bits[0, -1, start:end] != after_bits[0, -1, start:end]))}


def summarize(records):
    if len(records) != 160:
        raise ValueError("summary requires all 160 fixed forward records")
    families = []
    for group in range(8):
        cases = [records[(group * 4 + variant) * 5:(group * 4 + variant + 1) * 5]
                 for variant in range(4)]
        if any([row["action"] for row in case] != list(ACTIONS) for case in cases):
            raise ValueError("records do not follow the fixed action order")
        result = {"group": group + 1}
        for name, action_index in (("candidate", 2), ("control", 3)):
            deltas = [case[action_index]["score"]["target_log_probability"]
                      - case[0]["score"]["target_log_probability"] for case in cases]
            result[name] = {"case_deltas": deltas,
                            "family_contrast": deltas[0] - math.fsum(deltas[1:]) / 3}
        result["candidate_minus_control"] = (result["candidate"]["family_contrast"]
                                             - result["control"]["family_contrast"])
        families.append(result)
    return {"families": families,
            "mean_candidate": math.fsum(row["candidate"]["family_contrast"] for row in families) / 8,
            "mean_control": math.fsum(row["control"]["family_contrast"] for row in families) / 8,
            "mean_candidate_minus_control": math.fsum(row["candidate_minus_control"] for row in families) / 8}


def _append_event(stream, record):
    raw = canonical(record) + b"\n"
    if len(raw) > 65536:
        raise RuntimeError("attempt event exceeds its metadata bound")
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())


def _write_payload(stream, payload):
    result = {}
    for name, raw in payload.items():
        offset = stream.tell()
        if stream.write(raw) != len(raw):
            raise OSError("short raw tensor write")
        result[name] = {"offset": offset, "bytes": len(raw), "sha256": sha(raw)}
    stream.flush()
    os.fsync(stream.fileno())
    return result


def persist_action(log, raw_file, case, action, payload, clean_logits, noop_input, postcheck, marker):
    """Durably save a returned payload before evaluating controls or runtime."""
    raw = _write_payload(raw_file, payload)
    _append_event(log, {"event": "raw_recorded", "case_id": case["case_id"],
                        "action": action, "raw": raw})
    marker["recorded"] = True
    postcheck()
    if action in ("noop", "clean_after") and payload["logits"] != clean_logits:
        raise RuntimeError("unhooked clean and no-op logit bits differ")
    if action in ("candidate_zero", "control_zero") and payload["cproj_before"] != noop_input:
        raise RuntimeError("pre-ablation projection input bits differ from no-op")
    record = {"case_id": case["case_id"], "group": case["group"], "variant": case["variant"],
              "target_id": case["target_id"], "action": action,
              "score": statistics(payload["logits"], case["target_id"]),
              "intervention": action_witness(action, payload), "raw": raw}
    _append_event(log, {"event": "complete", **record})
    return record


def raw_file_inventory(output, cases):
    """Record bounded partial files after an unsuccessful attempt."""
    found = []
    for case in cases:
        path = output / (case["case_id"] + ".f32")
        if path.is_file():
            raw = path.read_bytes()
            found.append({"case_id": case["case_id"], "bytes": len(raw), "sha256": sha(raw)})
    return found


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
    dependencies = dependency_identity()
    data, cases = inventory()
    discovery = verify_discovery()
    exposure = check_exposure(cases)
    if EXPECTED_RAW_BYTES != 41_896_576 or EXPECTED_RAW_BYTES > MAX_RAW_BYTES:
        raise RuntimeError("declared raw payload limit is inconsistent")
    output = args.output_dir
    if not output.is_absolute() or not args.cache_dir.is_absolute():
        raise ValueError("cache and output paths must be absolute")
    parent = output.parent.resolve(strict=True)
    if ROOT == parent or ROOT in parent.parents or output.exists():
        raise ValueError("output must be a new directory outside the checkout")
    if shutil.disk_usage(parent).free < MAX_RAW_BYTES + 8_388_608:
        raise OSError("insufficient space for the declared raw payload")
    pre_probe = numeric_policy(initialize=True)
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    output.mkdir()
    manifest = {"schema": "distilgpt2-copy-head-run-v1", "source": source,
                "dependencies": dependencies, "inventory_sha256": INVENTORY_SHA256,
                "discovery": discovery,
                "exposure_inventory_sha256": exposure, "cases": cases,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "weights_sha256": WEIGHTS_SHA256, "actions": ACTIONS,
                "raw_payload_bytes": EXPECTED_RAW_BYTES, "max_raw_bytes": MAX_RAW_BYTES,
                "max_retained_bytes_per_forward": MAX_FORWARD_RETAINED_BYTES,
                "initial_numeric_probe": pre_probe,
                "counter_semantics": {
                    "submitted_forwards": "attempts entering the run_forward helper; an early helper error can precede a model call",
                    "returned_forwards": "complete payloads returned by the helper; a model call may return before helper validation fails",
                    "payload_recorded": "the complete returned payload and its raw_recorded event were durably written"},
                "intended_runtime": {"device": "cpu", "dtype": "float32",
                                     "attention_implementation": "eager", "use_cache": False,
                                     "intraop_threads": 1, "deterministic_algorithms": False,
                                     "interop_threads": torch.get_num_interop_threads(),
                                     "mkldnn_enabled": torch.backends.mkldnn.enabled,
                                     "float32_matmul_precision": "highest", "cpu_rng_seed": CPU_RNG_SEED},
                "evidence_status": "public disjoint exploratory; not blind or independently preregistered"}
    manifest_hash = durable_json(output / "manifest.json", manifest)
    completed, submitted, returned, retained = [], 0, 0, 0
    phase, case_id, action, payload_returned = "model_load", None, None, False
    marker = {"recorded": False}
    try:
        bundle = load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        if bundle.metadata["weights_sha256"] != WEIGHTS_SHA256:
            raise RuntimeError("checkpoint weight hash changed")
        for case in cases:
            if bundle.tokenizer(case["text"], add_special_tokens=False)["input_ids"] != case["token_ids"]:
                raise RuntimeError("pinned tokenizer does not reproduce a public case")
        model = bundle.model
        torch.random.default_generator.manual_seed(CPU_RNG_SEED)
        model_before = state_hash(model)
        buffers_before = buffer_hash(model)
        runtime_before = runtime(model)
        assert_no_forward_hooks(model)
        with (output / "attempt.jsonl").open("xb") as log:
            _append_event(log, {"event": "runtime_ready", "cpu_rng_seed": CPU_RNG_SEED,
                                "actual_runtime": runtime_before})
            for case in cases:
                case_id = case["case_id"]
                ids = torch.tensor([case["token_ids"]], dtype=torch.int64, device="cpu")
                clean_logits, noop_input = None, None
                with (output / (case_id + ".f32")).open("xb") as raw_file:
                    for action in ACTIONS:
                        phase = "forward"
                        payload_returned = False
                        marker = {"recorded": False}
                        _append_event(log, {"event": "start", "case_id": case_id, "action": action})
                        if runtime(model) != runtime_before:
                            raise RuntimeError("runtime changed before a forward")
                        submitted += 1
                        payload = run_forward(model, ids, action)
                        returned += 1
                        payload_returned = True
                        retained += sum(map(len, payload.values()))
                        if retained > EXPECTED_RAW_BYTES:
                            raise RuntimeError("aggregate raw payload exceeds its fixed limit")
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
        if (len(completed) != 160 or submitted != 160 or returned != 160
                or retained != EXPECTED_RAW_BYTES or state_hash(model) != model_before
                or buffer_hash(model) != buffers_before
                or runtime(model) != runtime_before or source_identity() != source
                or GUARD_EVENTS or torch.cuda.is_initialized()):
            raise RuntimeError("final count, model, runtime or source verification failed")
        assert_no_forward_hooks(model)
        result = {"schema": "distilgpt2-copy-head-result-v1", "status": "complete",
                  "manifest_sha256": manifest_hash, "checkpoint": bundle.metadata,
                  "model_state_sha256_before": model_before, "model_state_sha256_after": state_hash(model),
                  "model_buffers_sha256_before": buffers_before,
                  "model_buffers_sha256_after": buffer_hash(model),
                  "runtime": runtime_before, "guard_safe_queries": SAFE_QUERIES,
                  "submitted_forwards": submitted, "returned_forwards": returned,
                  "raw_payload_bytes": retained, "cases": completed,
                  "summary": summarize(completed)}
        durable_json(output / "result.json", result)
    except BaseException as error:
        attempt_log = output / "attempt.jsonl"
        if attempt_log.is_file():
            try:
                with attempt_log.open("ab") as log:
                    _append_event(log, {"event": "failed", "case_id": case_id, "action": action,
                                        "phase": phase, "error_type": type(error).__name__,
                                        "payload_returned": payload_returned,
                                        "payload_recorded": marker["recorded"]})
            except OSError:
                pass
        durable_json(output / "failure.json", {"schema": "distilgpt2-copy-head-failure-v1",
                     "status": "incomplete", "manifest_sha256": manifest_hash,
                     "phase": phase, "case_id": case_id, "action": action,
                     "payload_returned": payload_returned, "payload_recorded": marker["recorded"],
                     "submitted_forwards": submitted, "returned_forwards": returned,
                     "completed_records": len(completed), "admitted_payload_bytes": retained,
                     "raw_files": raw_file_inventory(output, cases),
                     "error_type": type(error).__name__,
                     "traceback_frames": [{"file": Path(frame.filename).name,
                                           "line": frame.lineno, "function": frame.name}
                                          for frame in traceback.extract_tb(error.__traceback__)]})
        raise
    print(json.dumps({"status": "complete", "forwards": submitted, "raw_payload_bytes": retained,
                      "summary": result["summary"]}, allow_nan=False))


if __name__ == "__main__":
    main()
