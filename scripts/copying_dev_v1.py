"""Fixed public DistilGPT-2 sequence-copying screen; no held-out inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

if __name__ == "__main__" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("set CUDA_VISIBLE_DEVICES=-1 before starting the CPU screen")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network  # noqa: E402

GUARD_EVENTS = []
SAFE_QUERIES = {"cuda_is_available": 0, "cuda_device_count": 0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)
from looking_glass.pretrained import load_distilgpt2  # noqa: E402


GROUPS = (
    (("apple", "river", "music", "green", "horse", "violet"), (17180, 7850, 2647, 4077, 8223, 45725)),
    (("silver", "cloud", "garden", "train", "candle", "lemon"), (8465, 6279, 11376, 4512, 26839, 18873)),
    (("forest", "button", "window", "ocean", "yellow", "paper"), (8222, 4936, 4324, 9151, 7872, 3348)),
    (("crystal", "pencil", "mirror", "thunder", "bottle", "purple"), (15121, 21613, 10162, 18355, 9294, 14032)),
    (("rabbit", "shadow", "basket", "orange", "castle", "marble"), (22746, 9082, 7988, 10912, 16669, 30623)),
    (("winter", "cookie", "copper", "velvet", "engine", "star"), (7374, 19751, 15317, 47750, 3113, 3491)),
    (("balloon", "feather", "jacket", "coffee", "island", "piano"), (21190, 17972, 15224, 6891, 7022, 19132)),
    (("pepper", "rocket", "sugar", "turtle", "lantern", "flower"), (13385, 10701, 7543, 28699, 41207, 15061)),
)
PERMUTATIONS = ((0, 1, 2, 3, 4, 5), (4, 2, 0, 3, 1, 5),
                (1, 3, 4, 2, 0, 5), (3, 0, 4, 1, 2, 5))
SOURCES = ("docs/DISTILGPT2_COPYING_DEV_V1.md", "scripts/copying_dev_v1.py",
           "scripts/distilgpt2_intervention_dev_bootstrap.py", "looking_glass/__init__.py",
           "looking_glass/pretrained.py")
VOCAB, LAYERS, HEADS, LENGTH = 50257, 6, 12, 11


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def durable_json(path, value):
    data = canonical(value)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError("result file already exists")
    os.replace(temporary, path)
    return sha(data)


def event(stream, value):
    stream.write(canonical(value) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("source checkout is not clean")
    sources = {}
    for path in SOURCES:
        disk = (ROOT / path).read_bytes()
        if git("show", f"{revision}:{path}") != disk:
            raise RuntimeError("recorded source bytes differ from the commit")
        sources[path] = sha(disk)
    return revision, sources


def numeric_policy(*, initialize=False):
    if initialize:
        torch.set_num_threads(1)
        if not torch.set_flush_denormal(False):
            raise RuntimeError("denormal preservation unavailable")
    if torch.get_num_threads() != 1:
        raise RuntimeError("intra-op thread count changed")
    subnormal = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    normal = torch.tensor([0x00800000], dtype=torch.int32).view(torch.float32)
    result = {"input_preserved": not bool(torch.equal(subnormal, torch.zeros_like(subnormal))),
              "half_min_normal_bits": int((normal * 0.5).view(torch.int32)[0].item()),
              "numpy_cast_preserved": float(np.array([1], dtype="<u4").view("<f4").astype("<f8")[0]) == 2.0 ** -149}
    if result != {"input_preserved": True, "half_min_normal_bits": 0x00400000, "numpy_cast_preserved": True}:
        raise RuntimeError("denormal preservation probe failed")
    return result


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise RuntimeError("unexpected model-state tensor")
        header = canonical({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        data = memoryview(tensor.detach().numpy()).cast("B")
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def runtime_checks(model):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized() or GUARD_EVENTS
            or torch.get_num_threads() != 1 or torch.is_autocast_enabled("cpu")
            or model.training or any(module.training for module in model.modules())
            or model.config._attn_implementation != "eager" or model.config.use_cache is not False
            or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                   for parameter in model.parameters())):
        raise RuntimeError("model runtime does not match the fixed CPU execution policy")
    return numeric_policy()


def planned_cases():
    ids = [token for _, group in GROUPS for token in group]
    if len(ids) != 48 or len(set(ids)) != 48:
        raise RuntimeError("development token IDs are not distinct")
    rows = []
    for group_index, (words, tokens) in enumerate(GROUPS):
        for variant, order in enumerate(PERMUTATIONS):
            indices = order + (0, 1, 2, 3, 4)
            rows.append({"case_id": f"copy-dev-{group_index + 1:02d}-{variant}",
                         "group": group_index + 1, "variant": variant,
                         "text": "".join(" " + words[index] for index in indices),
                         "token_ids": [tokens[index] for index in indices], "target_id": tokens[5]})
    return rows


def statistics(raw, target):
    values = np.frombuffer(raw, dtype="<f4").astype("<f8")
    maximum = float(values.max())
    total = math.fsum(math.exp(float(value) - maximum) for value in values)
    target_value = float(values[target])
    return {"target_logit": target_value, "target_log_probability": target_value - maximum - math.log(total),
            "target_rank": 1 + int(np.count_nonzero(values > target_value)),
            "top_token_id": int(values.argmax())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    revision, sources = source_identity()
    if not sys.dont_write_bytecode or not sys.pycache_prefix:
        raise RuntimeError("use -B and a fresh private -X pycache_prefix")
    if list(Path(sys.pycache_prefix).iterdir()):
        raise RuntimeError("bytecode cache prefix must begin empty")
    probe_before = numeric_policy(initialize=True)
    cases = planned_cases()
    args.output_dir.mkdir(parents=False, exist_ok=False)
    manifest = {"schema": "distilgpt2-copying-dev-v1", "source_revision": revision,
                "source_sha256": sources, "cases": cases, "control_permutations": PERMUTATIONS,
                "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                "dtype": "float32", "device": "cpu", "intraop_threads": 1,
                "expected_forwards": 33, "score": "target log probability from saved float32 final logits",
                "rank": "one plus count of strictly greater logits", "pre_probe": probe_before}
    manifest_hash = durable_json(args.output_dir / "manifest.json", manifest)
    completed = []
    submitted, root_returned = 0, 0
    try:
        bundle = load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        for case in cases:
            actual = bundle.tokenizer(case["text"], add_special_tokens=False)["input_ids"]
            if actual != case["token_ids"] or len(actual) != LENGTH:
                raise RuntimeError("pinned tokenizer did not produce the fixed token sequence")
        model = bundle.model
        if runtime_checks(model) != probe_before:
            raise RuntimeError("initial model runtime differs")
        before = state_hash(model)
        with (args.output_dir / "attempt.jsonl").open("xb") as log, \
             (args.output_dir / "last_logits.f32").open("xb") as logits_file, \
             (args.output_dir / "attentions.f32").open("xb") as attention_file:
            first = cases[0]
            event(log, {"event": "start", "case_id": first["case_id"], "mode": "uncaptured"})
            first_ids = torch.tensor([first["token_ids"]], dtype=torch.int64)
            submitted += 1
            runtime_checks(model)
            with torch.inference_mode():
                plain = model(input_ids=first_ids, attention_mask=torch.ones_like(first_ids),
                              use_cache=False, output_attentions=False, return_dict=True)
            root_returned += 1
            runtime_checks(model)
            if (plain.past_key_values is not None or type(plain.logits) is not torch.Tensor
                    or plain.logits.device.type != "cpu" or plain.logits.dtype != torch.float32
                    or tuple(plain.logits.shape) != (1, LENGTH, VOCAB)
                    or not bool(torch.isfinite(plain.logits[0, -1]).all())):
                raise RuntimeError("invalid uncaptured control")
            plain_raw = plain.logits[0, -1].detach().contiguous().numpy().astype("<f4", copy=False).tobytes()
            with (args.output_dir / "uncaptured_logits.f32").open("xb") as plain_file:
                plain_file.write(plain_raw); plain_file.flush(); os.fsync(plain_file.fileno())
            event(log, {"event": "complete", "case_id": first["case_id"], "mode": "uncaptured",
                        "logits_sha256": sha(plain_raw)})
            for case_index, case in enumerate(cases):
                event(log, {"event": "start", "case_id": case["case_id"]})
                ids = torch.tensor([case["token_ids"]], dtype=torch.int64)
                submitted += 1
                runtime_checks(model)
                with torch.inference_mode():
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   use_cache=False, output_attentions=True, return_dict=True)
                root_returned += 1
                runtime_checks(model)
                if (output.past_key_values is not None or type(output.logits) is not torch.Tensor
                        or output.logits.device.type != "cpu" or output.logits.dtype != torch.float32
                        or tuple(output.logits.shape) != (1, LENGTH, VOCAB)
                        or type(output.attentions) not in (tuple, list) or len(output.attentions) != LAYERS
                        or any(type(item) is not torch.Tensor or item.device.type != "cpu"
                               or item.dtype != torch.float32 for item in output.attentions)):
                    raise RuntimeError("unexpected output shape or cache")
                last = output.logits[0, -1].detach().contiguous()
                attention = torch.stack([item[0] for item in output.attentions]).detach().contiguous()
                if (tuple(attention.shape) != (LAYERS, HEADS, LENGTH, LENGTH)
                        or not bool(torch.isfinite(last).all()) or not bool(torch.isfinite(attention).all())):
                    raise RuntimeError("nonfinite or malformed output")
                normalization_error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
                future_attention = float(attention.triu(diagonal=1).abs().max().item())
                if (normalization_error > 1e-5 or future_attention != 0.0
                        or bool((attention < 0).any()) or bool((attention > 1).any())):
                    raise RuntimeError("attention normalization or causal mask differs")
                logits_raw = last.numpy().astype("<f4", copy=False).tobytes()
                attention_raw = attention.numpy().astype("<f4", copy=False).tobytes()
                if case_index == 0 and logits_raw != plain_raw:
                    raise RuntimeError("captured and uncaptured logit bits differ")
                logits_file.write(logits_raw); logits_file.flush(); os.fsync(logits_file.fileno())
                attention_file.write(attention_raw); attention_file.flush(); os.fsync(attention_file.fileno())
                record = {"case_id": case["case_id"], "group": case["group"], "variant": case["variant"],
                          "target_id": case["target_id"], "logits_sha256": sha(logits_raw),
                          "attention_sha256": sha(attention_raw),
                          "logits_bytes": len(logits_raw), "attention_bytes": len(attention_raw),
                          "attention_normalization_max_error": normalization_error,
                          "attention_future_max": future_attention,
                          "attention_last_to_earlier_target": attention[:, :, -1, 5].tolist(),
                          **statistics(logits_raw, case["target_id"])}
                completed.append(record)
                event(log, {"event": "complete", "case_id": case["case_id"],
                            "logits_sha256": record["logits_sha256"], "attention_sha256": record["attention_sha256"]})
        after = state_hash(model)
        post_probe = runtime_checks(model)
        if before != after or post_probe != probe_before or torch.cuda.is_initialized():
            raise RuntimeError("model state or execution policy changed")
        pairs = []
        for index in range(8):
            repeated, controls = completed[index * 4], completed[index * 4 + 1:index * 4 + 4]
            differences = [repeated["target_log_probability"] - row["target_log_probability"] for row in controls]
            pairs.append({"group": index + 1, "control_differences": differences,
                          "mean_log_probability_difference": math.fsum(differences) / 3})
        ranking = sorted(({"layer": layer, "head": head,
                           "mean_attention_difference": math.fsum(
                               completed[index * 4]["attention_last_to_earlier_target"][layer][head] -
                               completed[index * 4 + control]["attention_last_to_earlier_target"][layer][head]
                               for index in range(8) for control in (1, 2, 3)) / 24}
                          for layer in range(LAYERS) for head in range(HEADS)),
                         key=lambda row: (-row["mean_attention_difference"], row["layer"], row["head"]))
        means = [item["mean_log_probability_difference"] for item in pairs]
        result = {"schema": "distilgpt2-copying-dev-result-v1", "manifest_sha256": manifest_hash,
                  "source_revision": revision, "model_metadata": bundle.metadata,
                  "model_state_sha256_before": before, "model_state_sha256_after": after,
                  "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                              "system": platform.system(), "machine": platform.machine(),
                              "byteorder": sys.byteorder, "intraop_threads": torch.get_num_threads(),
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "cuda_visibility": "-1", "cuda_initialized": torch.cuda.is_initialized(),
                              "cpu_autocast": torch.is_autocast_enabled("cpu"), "model_eval": not model.training,
                              "attention_implementation": model.config._attn_implementation,
                              "use_cache": model.config.use_cache, "post_probe": post_probe,
                              "guard_blocked_calls": list(GUARD_EVENTS), "guard_safe_queries": dict(SAFE_QUERIES)},
                  "logits_sha256": sha((args.output_dir / "last_logits.f32").read_bytes()),
                  "attentions_sha256": sha((args.output_dir / "attentions.f32").read_bytes()),
                  "uncaptured_logits_sha256": sha(plain_raw), "captured_logits_bitwise_match": True,
                  "submitted_forwards": submitted, "root_outputs_returned": root_returned,
                  "cases": completed, "pairs": pairs, "mean_log_probability_difference": math.fsum(means) / 8,
                  "positive_group_means": sum(value > 0 for value in means),
                  "repeated_top_one": sum(row["top_token_id"] == row["target_id"] for row in completed[::4]),
                  "control_top_one": sum(row["top_token_id"] == row["target_id"] for row in completed if row["variant"]),
                  "attention_ranking": ranking}
        durable_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": "complete", "forwards": len(completed) + 1,
                          "positive_group_means": result["positive_group_means"],
                          "mean_log_probability_difference": result["mean_log_probability_difference"],
                          "repeated_top_one": result["repeated_top_one"],
                          "control_top_one": result["control_top_one"]}))
    except BaseException as error:
        durable_json(args.output_dir / "failure.json", {"status": "incomplete", "submitted_forwards": submitted,
                     "root_outputs_returned": root_returned, "complete_recorded_cases": len(completed),
                     "error_type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
