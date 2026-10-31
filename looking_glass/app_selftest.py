"""Bounded, synthetic, actual-CUDA acceptance check; no downloaded checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory


EXPECTED_STAGES = ("load", "same_device_full_logits_and_boundaries", "app_capture_offload",
                   "temporary_export_reopen", "repeated_forward_cleanup")

def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _checkpoint(path, family):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration, PreTrainedTokenizerFast
    path.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        if family == "gpt2":
            config = GPT2Config(vocab_size=8, n_positions=32, n_ctx=32, n_embd=8, n_layer=2,
                                n_head=2, eos_token_id=None, bos_token_id=2)
            cls = GPT2LMHeadModel
        else:
            config = T5Config(vocab_size=8, d_model=8, d_kv=4, d_ff=16, num_layers=2,
                              num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                              pad_token_id=0, eos_token_id=None)
            cls = T5ForConditionalGeneration
        config._attn_implementation = "eager"
        cls(config).eval().save_pretrained(path, safe_serialization=True)
    vocab = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[BOS]", "[EOS]", "hello", "world", "blue", "red"])}
    base = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=base, unk_token="[UNK]", pad_token="[PAD]",
                           bos_token="[BOS]", eos_token="[EOS]").save_pretrained(path)


def _bits_equal(left, right):
    import torch
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.detach().cpu().contiguous().view(torch.int32), right.detach().cpu().contiguous().view(torch.int32))


def _family(root, family, device, budget, stages):
    import torch
    from .app_runtime import LocalModelRuntime, _cuda_operation
    from .app_storage import TraceStore
    from .transformer_trace import GPT2TraceSession
    from .trace_schema import GenerationContext
    from .flan_t5_trace import T5TracePoint, trace_t5_forward

    runtime = LocalModelRuntime()
    handles = []
    stage = "allocator_prerequisite"
    try:
        previous = torch.cuda.get_per_process_memory_fraction(device)
        stage = "fixture"
        path = root / family
        _checkpoint(path, family)
        stage = "load"
        info = runtime.load(str(path), device=str(device), max_vram_bytes=budget)
        _require(info["device"] == str(device) and next(runtime.model.parameters()).device == device, "placement mismatch")
        _require(torch.cuda.get_per_process_memory_fraction(device) == previous, "load policy not restored")
        stages.append({"family": family, "stage": stage, "status": "passed"})
        stage = "same_device_full_logits_and_boundaries"
        expected = {}
        with _cuda_operation(device, budget) as comparison_memory, torch.inference_mode():
            ids = runtime.tokenizer("hello world", return_tensors="pt")["input_ids"].to(device)
            args = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            if family == "t5":
                decoder = torch.zeros((1, 1), dtype=torch.int64, device=device)
                args.update(decoder_input_ids=decoder, decoder_attention_mask=torch.ones_like(decoder))
            baseline = runtime.model(**args).logits.detach().cpu()
            if family == "gpt2":
                paths = [f"transformer.h.{layer}" + suffix for layer in range(2) for suffix in (".attn", ".mlp", "")]
            else:
                paths = [f"{stack}.block.{layer}" + suffix for stack, depth in (("encoder", 2), ("decoder", 1))
                         for layer in range(depth) for suffix in (".layer.0", ".layer.1" if stack == "encoder" else ".layer.2", "")]
            for name in paths:
                def hook(_module, _args, out, name=name):
                    expected[name] = (out[0] if isinstance(out, tuple) else out).detach().cpu().clone()
                handles.append(runtime.model.get_submodule(name).register_forward_hook(hook))
            observed = runtime.model(**args).logits
            _require(_bits_equal(baseline, observed), "independent hook changed logits")
            del observed
            for handle in handles: handle.remove()
            handles.clear()
            if family == "gpt2":
                with GPT2TraceSession(runtime.model, run_id="selftest", targets=("attn", "mlp", "block"),
                        layers=(0, 1), samples=None, tokens=None, mode="direct", capture_device="cpu",
                        max_events=1000, max_trace_bytes=1024**2,
                        forward_context=GenerationContext(stage="forward", use_cache=False)) as trace:
                    traced = runtime.model(**args).logits
                records = trace.events
            else:
                points = [T5TracePoint(stack, layer, role, 0, token)
                    for stack, depth, length in (("encoder", 2, ids.shape[1]), ("decoder", 1, 1))
                    for layer in range(depth) for role in ("self_attention", "feed_forward", "block") for token in range(length)]
                output, records = trace_t5_forward(runtime.model, input_ids=ids, attention_mask=args["attention_mask"],
                    decoder_input_ids=decoder, decoder_attention_mask=args["decoder_attention_mask"], points=points,
                    run_id="selftest", max_events=1000, max_trace_bytes=1024**2, max_tensor_bytes=1024**2,
                    max_input_tokens=32, max_logits_bytes=1024**2, synchronous_cpu=False)
                traced = output.logits
            _require(_bits_equal(baseline, traced), "adapter changed full logits")
            _require(len(records) > 0, "adapter captured no boundaries")
            del traced, records
            if family == "t5": del output
        _require(torch.cuda.get_per_process_memory_fraction(device) == previous, "comparison policy not restored")
        stages.append({"family": family, "stage": stage, "status": "passed", "comparison": "complete float32 logits, raw bits, same device",
                       "memory": {key: comparison_memory[key] for key in ("peak_allocated_bytes", "peak_reserved_bytes", "effective_limit_bytes")}})
        stage = "app_capture_offload"
        result = runtime.run("hello world", {"max_new_tokens": 1})
        _require(result["generated_token_ids"] == [int(baseline[0, -1].argmax())], "app prediction differs")
        seen = set()
        coordinates = Counter()
        expected_count = len(paths) if family == "gpt2" else 3 * (2 * ids.shape[1] + 1)
        _require(len(result["events"]) == expected_count, "capture count mismatch")
        for event in result["events"]:
            metadata = event.metadata()
            _require(metadata["device"] == str(device) and metadata["saved_device"] == "cpu" and event.value.device.type == "cpu", "offload placement mismatch")
            wanted = expected[metadata["module_path"]]
            if family == "t5": wanted = wanted[metadata["batch_positions"][0], metadata["token_positions"][0]]
            _require(_bits_equal(event.value, wanted), "captured boundary differs")
            seen.add(metadata["module_path"])
            coordinates[(metadata["module_path"], tuple(metadata["batch_positions"]), tuple(metadata["token_positions"]))] += 1
        _require(seen == set(paths), "missing boundary")
        expected_coordinates = Counter()
        for name in paths:
            length = ids.shape[1] if family == "gpt2" or name.startswith("encoder.") else 1
            if family == "gpt2": expected_coordinates[(name, (0,), tuple(range(length)))] += 1
            else:
                for token in range(length): expected_coordinates[(name, (0,), (token,))] += 1
        _require(coordinates == expected_coordinates, "boundary coordinates duplicated or missing")
        stages.append({"family": family, "stage": stage, "status": "passed", "events": expected_count})
        stage = "temporary_export_reopen"
        store = TraceStore(root / (family + "-exports"))
        store.publish(result)
        original = [[store.tensor(i, vector=j, limit=4096)["raw_bits"] for j in range(event.value.numel() // event.value.shape[-1])]
                    for i, event in enumerate(result["events"])]
        saved = store.save("selftest")
        store.clear(); store.open(saved["path"])
        for i, vectors in enumerate(original):
            for j, bits in enumerate(vectors):
                _require(store.tensor(i, vector=j, limit=4096)["raw_bits"] == bits, "export bits differ")
        _require(store.summary()["model"]["device"] == str(device), "export model metadata differs")
        stages.append({"family": family, "stage": stage, "status": "passed"})
        stage = "repeated_forward_cleanup"
        again = runtime.run("hello world", {"max_new_tokens": 2})
        _require(again["coverage"]["forward_count"] == 2, "repeat forward incomplete")
        for report in (result["gpu_memory"], again["gpu_memory"]):
            _require(report["synchronized"] and report["peak_reserved_bytes"] <= report["effective_limit_bytes"], "allocator report invalid")
        _require(all(not module._forward_hooks and not module._forward_pre_hooks for module in runtime.model.modules()), "residual hooks")
        _require(torch.cuda.get_per_process_memory_fraction(device) == previous, "run policy not restored")
        numerics = again["model"]["execution_numerics"]
        peaks = {key: again["gpu_memory"][key] for key in ("peak_allocated_bytes", "peak_reserved_bytes", "effective_limit_bytes")}
        stage = "repeated_forward_cleanup"
        runtime.unload()
        _require(torch.cuda.get_per_process_memory_fraction(device) == previous, "unload policy changed")
        stages.append({"family": family, "stage": stage, "status": "passed", "memory": peaks, "numerics": numerics})
    except Exception as error:
        stages.append({"family": family, "stage": stage, "status": "failed", "error_type": type(error).__name__,
                       "reason": "Self-test assertion or device operation failed; no success or cleanup guarantee."})
        raise
    finally:
        for handle in handles: handle.remove()
        # On failure, do not launch further CUDA cleanup in a possibly poisoned context.
        # CLI exits nonzero; no remaining family runs or cleanup-success claim is made.


def run_selftest(device="cuda:0", max_vram_mib=1024):
    report = {"schema": "looking-glass-gpu-selftest-v1", "status": "failed", "device": device,
              "dtype": "float32", "max_vram_mib": max_vram_mib, "stages": [],
              "scope": "Tiny synthetic GPT-2/T5 on one actual CUDA device; not arbitrary checkpoints, CPU/GPU equivalence, or full interpretability. Temporary synthetic exports are deleted."}
    try:
        if not isinstance(device, str) or not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
            raise ValueError("explicit cuda:N required")
        if type(max_vram_mib) is not int or not 64 <= max_vram_mib <= 65536:
            raise ValueError("budget must be 64..65536 MiB")
        import torch
        report["torch_version"], report["cuda_runtime"] = str(torch.__version__), torch.version.cuda
        if not torch.cuda.is_available() or int(device.split(":")[1]) >= torch.cuda.device_count():
            report.update(status="skipped", reason="Requested actual CUDA device/build unavailable; no GPU validation performed.")
            report["stages"].append({"stage": "cuda_prerequisite", "status": "skipped"})
            return report
        selected = torch.device(device)
        free, total = torch.cuda.mem_get_info(selected)
        if free < max(256 * 1024**2, math.ceil(total * .10)) + 64 * 1024**2:
            report.update(status="skipped", reason="Insufficient free CUDA headroom for bounded self-test.")
            report["stages"].append({"stage": "cuda_prerequisite", "status": "skipped"})
            return report
        with TemporaryDirectory(prefix="looking-glass-selftest-") as temporary:
            for family in ("gpt2", "t5"):
                _family(Path(temporary), family, selected, max_vram_mib * 1024**2, report["stages"])
        expected = [(family, stage, "passed") for family in ("gpt2", "t5") for stage in EXPECTED_STAGES]
        actual = [(row.get("family"), row.get("stage"), row.get("status")) for row in report["stages"]]
        _require(actual == expected, "incomplete, unordered, duplicated, or unsuccessful stage ledger")
        report["status"] = "passed"
    except Exception as error:
        if not report["stages"] or report["stages"][-1]["status"] != "failed":
            report["stages"].append({"stage": "validation_or_final_cleanup", "status": "failed", "error_type": type(error).__name__})
        report.update(status="failed", error_type=type(error).__name__, reason="Validation, execution, or cleanup failed; remaining stages stopped.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-vram-mib", type=int, default=1024)
    args = parser.parse_args(argv)
    with redirect_stdout(sys.stderr):
        report = run_selftest(args.device, args.max_vram_mib)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return {"passed": 0, "failed": 1, "skipped": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
