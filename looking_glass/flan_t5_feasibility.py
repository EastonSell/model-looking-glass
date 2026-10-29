"""Fixed CPU-only FLAN-T5 feasibility study; see the v1 protocol."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROTOCOL = REPO / "docs/FLAN_T5_CAUSAL_FEASIBILITY_V1.md"
CASES = REPO / "docs/FLAN_T5_PUBLIC_CASES_V1.json"
REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
MANIFEST_HASH = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
CASE_HASH = "e00d77cf45885b10e19fe689b8cdb13d2d36dd038c7ab2c1701f9367eb2fe9fc"
MAX_BYTES = 128 * 1024 * 1024
MAX_FORWARDS = 76


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    paths = ("looking_glass/flan_t5_feasibility.py", "looking_glass/interventions.py",
             "looking_glass/model_ownership.py", "docs/FLAN_T5_CAUSAL_FEASIBILITY_V1.md",
             "docs/FLAN_T5_PUBLIC_CASES_V1.json")
    hashes = {}
    for name in paths:
        actual = (REPO / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
        if actual != committed:
            raise ValueError(f"source differs from committed blob: {name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return {"commit": commit, "source_sha256": hashes}


def verify_assets(directory):
    manifest_path = directory / "asset-manifest.json"
    if digest_file(manifest_path) != MANIFEST_HASH:
        raise ValueError("asset manifest differs from predeclared identity")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["repo"] != "google/flan-t5-small" or manifest["revision"] != REVISION:
        raise ValueError("checkpoint revision differs")
    required = {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
                "spiece.model", "tokenizer.json", "tokenizer_config.json"}
    if set(manifest["files"]) != required:
        raise ValueError("checkpoint asset set differs")
    for name, item in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != item["bytes"] or digest_file(path) != item["sha256"]:
            raise ValueError(f"checkpoint asset identity differs: {name}")
    return manifest


def check_tokenizer(tokenizer, cases):
    if type(tokenizer).__name__ != "T5TokenizerFast" or tokenizer.pad_token_id != 0 or tokenizer.eos_token_id != 1:
        raise ValueError("tokenizer class or special IDs differ")
    for row in [cases["neutral"], *(x for pair in cases["pairs"] for x in pair["sources"] + pair["targets"])]:
        current = tokenizer(row["text"], return_offsets_mapping=True, return_special_tokens_mask=True)
        for key, old_key in (("input_ids", "input_ids"), ("attention_mask", "attention_mask"),
                             ("special_tokens_mask", "special_tokens_mask"), ("offset_mapping", "offsets")):
            if old_key in row:
                value = [list(v) if isinstance(v, tuple) else v for v in current[key]]
                if value != row[old_key]:
                    raise ValueError(f"tokenizer alignment differs: {key}")
        if not row["input_ids"] or row["input_ids"][-1] != 1:
            raise ValueError("missing target/source EOS")
    eligibility = []
    for pair in cases["pairs"]:
        left, right = pair["sources"]
        a, b = left["input_ids"], right["input_ids"]
        reasons = []
        too_short = any(len(row[key]) <= 6 for row in (left, right)
                        for key in ("input_ids", "special_tokens_mask", "offsets"))
        if (too_short or len(a) != len(b) or left["attention_mask"] != right["attention_mask"] or len(a) > 32
                or [i for i, (x, y) in enumerate(zip(a, b)) if x != y] != [6]
                or left["word_position"] != 6 or right["word_position"] != 6
                or (not too_short and (left["special_tokens_mask"][0] or left["special_tokens_mask"][6]
                                       or right["special_tokens_mask"][0] or right["special_tokens_mask"][6]))):
            reasons.append("source-token-alignment")
        if not too_short:
            for source in (left, right):
                start, end = source["offsets"][6]
                segment = source["text"][start:end]
                if segment.strip() != source["text"].split()[-1]:
                    reasons.append("source-word-offset")
        if any(len(target["input_ids"]) > 8 for target in pair["targets"]):
            reasons.append("target-length")
        eligibility.append(reasons)
    return eligibility


class Evidence:
    def __init__(self, path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=False)
        self.used = 0

    def write(self, name, raw, reserve=65536):
        if self.used + len(raw) + reserve > MAX_BYTES:
            raise ValueError("retained evidence limit exceeded")
        dest = self.path / name
        with dest.open("xb") as f:
            written = f.write(raw)
            if written != len(raw):
                raise OSError("short evidence write")
            f.flush()
            os.fsync(f.fileno())
        self.used += len(raw)
        return hashlib.sha256(raw).hexdigest()

    def json(self, name, value, reserve=65536):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        return self.write(name, raw, reserve)


def memory_bytes():
    if sys.platform != "win32":
        return {"available": False}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),
                                                     ctypes.byref(values), values.cb):
        raise OSError("process memory observation unavailable")
    return {"available": True, "working_set_bytes": values.WorkingSetSize,
            "peak_working_set_bytes": values.PeakWorkingSetSize}


def run(asset_dir, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise ValueError("CPU run requires CUDA_VISIBLE_DEVICES empty or -1")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("offline flags must be explicitly set")
    if digest_file(CASES) != CASE_HASH:
        raise ValueError("case manifest differs from fixed input")
    cases = json.loads(CASES.read_bytes())
    source_before = source_identity()
    assets_before = verify_assets(asset_dir)
    evidence = Evidence(output)
    evidence.json("started.json", {"schema": "mlg-flant5-attempt-v1", "status": "started",
                                   "source": source_before, "asset_manifest_sha256": MANIFEST_HASH,
                                   "case_manifest_sha256": CASE_HASH, "evidence_cap": MAX_BYTES,
                                   "max_forwards": MAX_FORWARDS})
    active_forward = {"name": None, "index": None, "phase": "startup"}
    try:
        import torch
        import tokenizers
        import transformers
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from .interventions import ActivationCapture, ActivationIntervention

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        flush_supported = torch.set_flush_denormal(False)
        subnormal_bits = int((torch.tensor([2.0 ** -149], dtype=torch.float32) * 1.0).view(torch.int32)[0])
        if not flush_supported or subnormal_bits != 1:
            raise ValueError("CPU denormal behavior differs from the fixed setting")
        torch.manual_seed(20261028)
        tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True,
                                                  use_fast=True, trust_remote_code=False)
        eligibility = check_tokenizer(tokenizer, cases)
        before_load = memory_bytes()
        model, load_info = T5ForConditionalGeneration.from_pretrained(
            str(asset_dir), local_files_only=True, use_safetensors=True, trust_remote_code=False,
            output_loading_info=True, dtype=torch.float32)
        if any(load_info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("model loading diagnostics are not empty")
        model.eval()
        model.config.use_cache = False
        if (type(model).__name__ != "T5ForConditionalGeneration" or model.config.d_model != 512
                or model.config.vocab_size != 32128 or model.config.decoder_start_token_id != 0
                or model.config.eos_token_id != 1 or model.config.pad_token_id != 0):
            raise ValueError("model configuration differs")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("model parameters are not CPU float32")
        if any(b.device.type != "cpu" or (b.is_floating_point() and b.dtype != torch.float32)
               for b in model.buffers()):
            raise ValueError("model buffers are not on CPU with the expected dtype")
        after_load = memory_bytes()
        environment = {"python": platform.python_version(), "torch": torch.__version__,
                       "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                       "device": "cpu", "dtype": "float32", "num_threads": torch.get_num_threads(),
                       "interop_threads": torch.get_num_interop_threads(),
                       "deterministic": torch.are_deterministic_algorithms_enabled(), "seed": 20261028,
                       "system": platform.system(), "system_release": platform.release(),
                       "processor": platform.processor(), "cpu_count": os.cpu_count(),
                       "mkldnn_enabled": torch.backends.mkldnn.enabled,
                       "torch_build_config": torch.__config__.show(),
                       "subnormal_multiply_bits": subnormal_bits,
                       "before_load": before_load, "after_load": after_load,
                       "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}
        evidence.json("environment.json", environment)
        records = []
        def hook_snapshot():
            return tuple((id(m), tuple(m._forward_pre_hooks.items()), tuple(m._forward_hooks.items()),
                          tuple(m._backward_hooks.items())) for m in model.modules())
        hooks_before = hook_snapshot()

        def equal(a, b):
            return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))

        def save_failed_output(logits, state, name, reason):
            def bounded(value, limit):
                if not isinstance(value, torch.Tensor):
                    return b"", None
                value = value.detach().contiguous()
                shape = list(value.shape)
                if value.device.type != "cpu" or value.dtype != torch.float32 or not 0 < value.numel() <= limit:
                    return b"", shape
                return value.numpy().astype("<f4", copy=False).tobytes(), shape
            log_raw, log_shape = bounded(logits, 8 * 32128)
            state_raw, state_shape = bounded(state, 32 * 512)
            row = {"name": name, "status": reason, "logits_bytes": len(log_raw),
                   "encoder_bytes": len(state_raw), "logits_shape": log_shape, "encoder_shape": state_shape,
                   "logits_available": bool(log_raw), "encoder_available": bool(state_raw)}
            if log_raw or state_raw:
                row["file"] = f"forward-{len(records):03d}-failed.bin"
                row["sha256"] = evidence.write(row["file"], log_raw + state_raw)
            evidence.json(f"forward-{len(records):03d}-failed.json", row)

        def run_forward(source, target, name, condition, value=None, token_index=None):
            if len(records) >= MAX_FORWARDS:
                raise ValueError("forward cap exceeded")
            source_ids = torch.tensor([source["input_ids"]], dtype=torch.long)
            source_mask = torch.tensor([source["attention_mask"]], dtype=torch.long)
            target_ids = target["input_ids"]
            decoder_ids = [0] + target_ids[:-1]
            decoder = torch.tensor([decoder_ids], dtype=torch.long)
            decoder_mask = torch.ones_like(decoder)
            active_forward.update(name=name, index=len(records), phase="forward")
            kwargs = dict(input_ids=source_ids, attention_mask=source_mask, decoder_input_ids=decoder,
                          decoder_attention_mask=decoder_mask, use_cache=False, return_dict=True)
            with torch.no_grad():
                captured = None
                if condition == "clean":
                    out = model(**kwargs)
                elif condition == "capture":
                    with ActivationCapture(model, "encoder.final_layer_norm") as session:
                        out = model(**kwargs)
                    captured = session.value
                else:
                    action = "null" if condition == "null" else "zero" if condition == "zero" else "patch"
                    selection = {} if token_index is None else {"batch_index": 0, "token_index": token_index}
                    with ActivationIntervention(model, "encoder.final_layer_norm", operation=action,
                                                value=value, **selection):
                        out = model(**kwargs)
                raw_logits = getattr(out, "logits", None)
                raw_state = getattr(out, "encoder_last_hidden_state", None)
                if not isinstance(raw_logits, torch.Tensor) or not isinstance(raw_state, torch.Tensor):
                    save_failed_output(raw_logits, raw_state, name, "non-tensor-output")
                    raise ValueError("non-tensor model output")
                logits = raw_logits.detach().contiguous()
                state = raw_state.detach().contiguous()
                if (logits.dtype != torch.float32 or state.dtype != torch.float32
                        or logits.shape != (1, len(target_ids), 32128)
                        or state.shape != (1, len(source["input_ids"]), 512)
                        or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(state).all())):
                    save_failed_output(logits, state, name, "malformed-or-nonfinite")
                    raise ValueError("nonfinite or malformed model output")
                selected = torch.log_softmax(logits[0], dim=-1)[torch.arange(len(target_ids)),
                                                               torch.tensor(target_ids, dtype=torch.long)]
                per_token = [float(x) for x in selected]
                if not all(math.isfinite(x) for x in per_token):
                    save_failed_output(logits, state, name, "nonfinite-logprob")
                    raise ValueError("nonfinite selected log probabilities")
                score = math.fsum(per_token)
            log_bytes = logits.numpy().astype("<f4", copy=False).tobytes(order="C")
            state_bytes = state.numpy().astype("<f4", copy=False).tobytes(order="C")
            index = len(records)
            filename = f"forward-{index:03d}.bin"
            sha = evidence.write(filename, log_bytes + state_bytes)
            row = {"index": index, "name": name, "condition": condition,
                   "source_ids": source["input_ids"], "source_mask": source["attention_mask"],
                   "target_ids": target_ids, "decoder_ids": decoder_ids,
                   "decoder_mask": [1] * len(decoder_ids), "token_index": token_index,
                   "selected_logprobs": per_token, "score": score, "file": filename,
                   "sha256": sha, "dtype": "little-endian-float32", "logits_shape": list(logits.shape),
                   "logits_bytes": len(log_bytes), "encoder_shape": list(state.shape),
                   "encoder_bytes": len(state_bytes)}
            evidence.json(f"forward-{index:03d}.json", row)
            records.append(row)
            active_forward.update(phase="control-check")
            if captured is not None and not equal(captured, state):
                raise ValueError("captured encoder state differs from root-model output")
            return score, logits, state

        summaries = []
        for pair_index, pair in enumerate(cases["pairs"]):
            if eligibility[pair_index]:
                summary = {"pair_index": pair_index, "status": "ineligible",
                           "reasons": eligibility[pair_index]}
                summaries.append(summary)
                evidence.json(f"pair-{pair_index}-decision.json", summary)
                continue
            targets = pair["targets"]
            neutral = [run_forward(cases["neutral"], t, f"pair{pair_index}-context-target{j}", "clean")[0]
                       for j, t in enumerate(targets)]
            base = {}
            states = {}
            scores = {}
            for side in range(2):
                for target_index in range(2):
                    score, logits, state = run_forward(pair["sources"][side], targets[target_index],
                                                      f"pair{pair_index}-base-{side}-{target_index}", "clean")
                    base[side, target_index] = logits
                    scores["base", side, target_index] = score
                    if side in states and not equal(states[side], state):
                        raise ValueError("encoder state varies with target decoder inputs")
                    states[side] = state
            margin = [math.fsum((scores["base", s, 0], -scores["base", s, 1])) for s in range(2)]
            delta = margin[0] - margin[1]
            context_margin = neutral[0] - neutral[1]
            summary = {"pair_index": pair_index, "baseline_margins": margin, "delta": delta,
                       "context_only_scores": neutral, "context_only_margin": context_margin,
                       "context_adjusted_margins": [x - context_margin for x in margin],
                       "baseline_supported": margin[0] > 0 and margin[1] < 0 and delta >= 1.0}
            if summary["baseline_supported"]:
                conditions = ("repeat", "null", "token_self", "full_self", "full_swap",
                              "word_swap", "zero", "instruction_swap")
                distances = {}
                patches = {}
                for condition in conditions:
                    for side in range(2):
                        own, donor = states[side], states[1 - side]
                        token_index = 6 if condition in ("null", "token_self", "word_swap", "zero") else \
                                      0 if condition == "instruction_swap" else None
                        action = "capture" if condition == "repeat" else "null" if condition == "null" else \
                                 "zero" if condition == "zero" else "patch"
                        value = None
                        if action == "patch":
                            if condition == "full_self": value = own
                            elif condition == "full_swap": value = donor
                            elif condition == "token_self": value = own[0, token_index]
                            else: value = donor[0, token_index]
                        if token_index is not None:
                            expected = (torch.zeros_like(own[0, token_index]) if action == "zero" else
                                        own[0, token_index] if action == "null" else value)
                            diff = (expected.double() - own[0, token_index].double()).flatten()
                            distances[f"{condition}-{side}"] = math.sqrt(math.fsum(float(x) ** 2 for x in diff))
                        elif condition in ("full_self", "full_swap"):
                            diff = (value.double() - own.double()).flatten()
                            distances[f"{condition}-{side}"] = math.sqrt(math.fsum(float(x) ** 2 for x in diff))
                        if condition not in ("repeat",):
                            patches[f"{condition}-{side}"] = {
                                "destination_side": side,
                                "donor_side": (1 - side if condition in ("full_swap", "word_swap", "instruction_swap")
                                               else None if condition == "zero" else side),
                                "destination_position": token_index,
                                "donor_position": None if condition == "zero" else token_index,
                                "destination_special": (None if token_index is None else
                                                        pair["sources"][side]["special_tokens_mask"][token_index]),
                                "donor_special": (None if token_index is None or condition == "zero" else
                                                  pair["sources"][1 - side if condition in ("word_swap", "instruction_swap") else side]
                                                  ["special_tokens_mask"][token_index]),
                                "l2": distances.get(f"{condition}-{side}")}
                        for target_index in range(2):
                            score, logits, state = run_forward(pair["sources"][side], targets[target_index],
                                f"pair{pair_index}-{condition}-{side}-{target_index}", action, value, token_index)
                            scores[condition, side, target_index] = score
                            if condition in ("repeat", "null", "token_self", "full_self") and not equal(logits, base[side, target_index]):
                                raise ValueError(f"identity control failed: {condition}")
                            if condition == "full_swap" and not equal(logits, base[1 - side, target_index]):
                                raise ValueError("complete encoder swap control failed")
                            if condition in ("repeat", "full_self") and not equal(state, own):
                                raise ValueError(f"complete encoder state control failed: {condition}")
                            if condition == "full_swap" and not equal(state, donor):
                                raise ValueError("complete encoder swap state differs from donor")
                            if token_index is not None:
                                if not equal(state[0, token_index], expected):
                                    raise ValueError("patched vector does not match expected source")
                                for i in range(state.shape[1]):
                                    if i != token_index and not equal(state[0, i], own[0, i]):
                                        raise ValueError("unselected encoder position changed")
                def condition_margin(condition, side):
                    return math.fsum((scores[condition, side, 0], -scores[condition, side, 1]))
                word = [margin[0] - condition_margin("word_swap", 0),
                        condition_margin("word_swap", 1) - margin[1]]
                instruction = [margin[0] - condition_margin("instruction_swap", 0),
                               condition_margin("instruction_swap", 1) - margin[1]]
                positive = all(word[i] >= 0.20 * delta and word[i] >= instruction[i] + 0.05 * delta
                               for i in range(2))
                summary.update({"status": "positive" if positive else "negative", "word_movements": word,
                                "instruction_movements": instruction, "patch_distances_l2": distances,
                                "patch_metadata": patches,
                                "condition_margins": {c: [condition_margin(c, s) for s in range(2)] for c in conditions}})
            else:
                summary["status"] = "baseline-unsupported"
            summaries.append(summary)
            evidence.json(f"pair-{pair_index}-decision.json", summary)
            print(json.dumps({"pair_index": pair_index, "status": summary["status"],
                              "baseline_margins": margin, "forward_count": len(records)}), flush=True)
        source_after = source_identity()
        assets_after = verify_assets(asset_dir)
        if source_before != source_after or assets_before != assets_after:
            raise ValueError("source or checkpoint changed during execution")
        if hook_snapshot() != hooks_before:
            raise ValueError("model hooks were not restored")
        completion = {"schema": "mlg-flant5-causal-feasibility-v1", "status": "complete",
                      "source": source_before, "asset_manifest_sha256": MANIFEST_HASH,
                      "case_manifest_sha256": CASE_HASH, "model": "google/flan-t5-small",
                      "revision": REVISION, "forwards": len(records), "pairs": summaries,
                      "environment": environment, "final_memory": memory_bytes(),
                      "model_hooks_restored": True,
                      "bytes_before_terminal": evidence.used,
                      "limitations": ["Public exploratory pairs, not blinded or pretraining-independent.",
                                      "Sequence scores include EOS and are sensitive to target length and priors.",
                                      "An encoder-state intervention does not establish a general translation mechanism."]}
        sha = evidence.json("completed.json", completion, reserve=0)
        print(json.dumps({"status": "complete", "result_sha256": sha, "forwards": len(records),
                          "retained_bytes": evidence.used}), flush=True)
        return completion
    except BaseException as error:
        try:
            evidence.json("failure.json", {"schema": "mlg-flant5-attempt-v1", "status": "incomplete",
                                           "error_type": type(error).__name__, "failed_forward": active_forward,
                                           "retained_bytes_before_failure": evidence.used},
                          reserve=0)
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.asset_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
