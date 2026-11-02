"""Public, CPU-only taste-context activation-patching exploration."""

from dataclasses import dataclass
from contextlib import ExitStack
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from looking_glass.bert_agreement_runner import (
    _has_hooks as has_hooks,
    decode_float32_bits,
    float32_bytes,
)
from looking_glass.intervention_plan import ActivationPlan


MODEL = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
SEED = 20261029
CASES_PATH = "experiments/bert_taste_public_cases_v1.json"
PRIOR_INVENTORIES = {
    "experiments/bert_agreement_exposures_v5.json": "34db577a211d6036e7803b8cdb6aaab35687b56974de8cf1ac6bccbf3272fb3c",
    "experiments/bert_matched_public_cases_v1.json": "5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb",
}
SOURCE_PATHS = (
    "scripts/bert_taste_patch_v1.py", "scripts/bert_taste_parent_v1.py",
    "docs/BERT_TASTE_CONTEXT_V1.md", "experiments/bert_taste_public_cases_v1.json",
    "experiments/bert_agreement_exposures_v5.json", "experiments/bert_matched_public_cases_v1.json",
    "looking_glass/__init__.py", "looking_glass/pretrained.py",
    "looking_glass/bert_agreement_study.py", "looking_glass/bert_agreement_runner.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
    "looking_glass/model_ownership.py",
)


@dataclass(frozen=True)
class Observation:
    mask_logits: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor | None
    full_before_sha256: str
    full_after_sha256: str | None
    untouched_positions_equal: bool | None
    events: tuple


def _sha_tensor(tensor):
    return hashlib.sha256(float32_bytes(tensor)).hexdigest()


def denormal_probe():
    smallest = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    normal = torch.tensor([0x00800000], dtype=torch.int32).view(torch.float32)
    result = {"smallest_times_one_bits": int((smallest * torch.ones(1)).view(torch.int32)[0]),
              "normal_half_bits": int((normal * torch.tensor([0.5])).view(torch.int32)[0])}
    if result != {"smallest_times_one_bits": 1, "normal_half_bits": 0x00400000}:
        raise RuntimeError("denormal-preservation probe failed")
    return result


def observe(model, ids, *, layer, position, mask_index, source=None, self_patch=False):
    """Run one block-output capture/patch with a whole-output bit witness."""
    if (not isinstance(model, torch.nn.Module) or any(m.training for m in model.modules())
            or has_hooks(model)):
        raise ValueError("a hook-free evaluation model is required")
    if (type(ids) is not tuple or len(ids) != 10 or any(type(i) is not int for i in ids)
            or type(layer) is not int or not 0 <= layer < len(model.distilbert.transformer.layer)
            or type(position) is not int or not 0 <= position < 10
            or type(mask_index) is not int or not 0 <= mask_index < 10):
        raise ValueError("invalid input or layer coordinates")
    if (any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
            for parameter in model.parameters())
            or any(buffer.device.type != "cpu" for buffer in model.buffers())):
        raise ValueError("the model must be CPU float32")
    hidden, vocab = model.config.dim, model.config.vocab_size
    if any(i < 0 or i >= vocab for i in ids):
        raise ValueError("input token is outside the vocabulary")
    if type(self_patch) is not bool or (self_patch and source is not None):
        raise ValueError("self patch and external source are mutually exclusive")
    if source is not None and (type(source) is not torch.Tensor or source.device.type != "cpu"
                               or source.dtype != torch.float32 or tuple(source.shape) != (hidden,)
                               or not bool(torch.isfinite(source).all())):
        raise ValueError("source must be a finite CPU float32 hidden vector")

    path = f"distilbert.transformer.layer.{layer}"
    whole = {"module_path": path, "invocation": 0}
    selected = {**whole, "batch_index": 0, "token_index": position}
    steps = [{"name": "full_before", "operation": "capture", **whole},
             {"name": "before", "operation": "capture", **selected}]
    patched = self_patch or source is not None
    if patched:
        action = {"name": "patch", "operation": "patch", **selected}
        action["source_capture" if self_patch else "value"] = "before" if self_patch else source
        steps.extend((action, {"name": "after", "operation": "capture", **selected},
                      {"name": "full_after", "operation": "capture", **whole}))
    plan = ActivationPlan(model, steps, max_tensor_bytes=10 * hidden * 4,
                          max_capture_bytes=(22 if patched else 11) * hidden * 4,
                          max_source_bytes=hidden * 4, max_steps=5, max_invocations=1)
    inputs = torch.tensor([ids], dtype=torch.int64)
    attention = torch.ones_like(inputs)
    with torch.inference_mode(), plan:
        logits = model(input_ids=inputs, attention_mask=attention,
                       output_attentions=False, output_hidden_states=False,
                       return_dict=True).logits
        if (tuple(inputs[0].tolist()) != ids or tuple(attention[0].tolist()) != (1,) * 10
                or type(logits) is not torch.Tensor or tuple(logits.shape) != (1, 10, vocab)
                or logits.dtype != torch.float32 or logits.device.type != "cpu"
                or not bool(torch.isfinite(logits).all())):
            raise RuntimeError("model input or output is invalid")
        mask_logits = logits[0, mask_index].detach().view(torch.int32).clone().view(torch.float32)
    if has_hooks(model):
        raise RuntimeError("hook cleanup is uncertain; discard the model")

    captures, metadata = plan.captures, plan.metadata
    events = metadata["steps"]
    expected_names = [step["name"] for step in steps]
    if (len(events) != len(steps) or [event["name"] for event in events] != expected_names
            or any(event["operation"] != step["operation"] or event["order"] != index
                   or event["module_path"] != path or event["module_aliases"] != [path]
                   or event["forward_index"] != 0 or event["invocation"] != 0
                   or event["batch_index"] != step.get("batch_index")
                   or event["token_index"] != step.get("token_index")
                   or event["output_selector"] != [0]
                   or event["output_shape"] != [1, 10, hidden]
                   or event["selected_shape"] != ([hidden] if "token_index" in step else [1, 10, hidden])
                   or event["dtype"] != "torch.float32" or event["device"] != "cpu"
                   or event["source_capture"] != step.get("source_capture")
                   or event["preceding_same_site_actions"] != expected_names[:index]
                   for index, (event, step) in enumerate(zip(events, steps)))):
        raise RuntimeError("observed intervention coordinates differ")
    full_before, before = captures["full_before"], captures["before"]
    if not torch.equal(full_before[0, position].contiguous().view(torch.int32),
                       before.contiguous().view(torch.int32)):
        raise RuntimeError("selected pre-action bits differ from the full output")
    after = captures.get("after")
    full_after = captures.get("full_after")
    untouched = None
    if patched:
        wanted = before if self_patch else source
        expected = full_before.contiguous().view(torch.int32).clone()
        expected[0, position] = wanted.contiguous().view(torch.int32)
        untouched = torch.equal(expected, full_after.contiguous().view(torch.int32))
        if (not untouched or not torch.equal(after.contiguous().view(torch.int32),
                                              wanted.contiguous().view(torch.int32))):
            raise RuntimeError("patch failed its full-output bit witness")
    return Observation(mask_logits, before, after, _sha_tensor(full_before),
                       None if full_after is None else _sha_tensor(full_after),
                       untouched, tuple(events))


def summarize_logits(logits, *, sweet_id, sour_id):
    """Report raw binary32 logits and full-vocabulary softmax probabilities."""
    if (type(logits) is not torch.Tensor or logits.device.type != "cpu"
            or logits.dtype != torch.float32 or logits.ndim != 1 or logits.numel() < 5
            or type(sweet_id) is not int or type(sour_id) is not int or sweet_id == sour_id
            or not 0 <= sweet_id < logits.numel() or not 0 <= sour_id < logits.numel()
            or not bool(torch.isfinite(logits).all())):
        raise ValueError("invalid mask logits or target IDs")
    raw = logits.contiguous().view(torch.int32)
    sweet_bits, sour_bits = int(raw[sweet_id]) & 0xffffffff, int(raw[sour_id]) & 0xffffffff
    sweet, sour = decode_float32_bits(sweet_bits), decode_float32_bits(sour_bits)
    probabilities = torch.softmax(logits.to(torch.float64), dim=0)
    return {"sweet_bits": sweet_bits, "sour_bits": sour_bits,
            "sweet_logit": sweet, "sour_logit": sour,
            "sweet_minus_sour": math.fsum((sweet, -sour)),
            "sweet_probability": float(probabilities[sweet_id]),
            "sour_probability": float(probabilities[sour_id]),
            "top_five_ids": [int(i) for i in torch.topk(logits, 5).indices],
            "mask_logits_sha256": _sha_tensor(logits)}


def compare_exposures(rows, prior_sets):
    """Reject exact or complete-row-prefix matches against stated public inventories."""
    def coordinates(row):
        ids, mask = row["input_ids"], row["attention_mask"]
        if (type(ids) is not list or not ids or type(mask) is not list or len(ids) != len(mask)
                or any(type(x) is not int or x < 0 for x in ids)
                or any(type(x) is not int or x not in (0, 1) for x in mask)):
            raise ValueError("invalid exposure row")
        return ids, mask

    for row in rows:
        ids, mask = coordinates(row)
        for prior in prior_sets:
            for known in prior:
                old_ids, old_mask = coordinates(known)
                length = min(len(ids), len(old_ids))
                if ids[:length] == old_ids[:length] and mask[:length] == old_mask[:length]:
                    raise ValueError("exposure collision with a cited public row")
    return 0


def run_cases(model, rows, *, sweet_id, sour_id, emit):
    """Execute each adjacent sugar/lemon pair and retain every directional effect."""
    if type(rows) is not list or not rows or len(rows) % 2 or not callable(emit):
        raise ValueError("expected complete adjacent pairs and a journal callback")
    baselines, effects = {}, []
    count = 0

    def forward(row, *, layer, position, mode, donor=None, source=None, self_patch=False):
        nonlocal count
        identity = {"row_id": row["id"], "layer": layer, "position": position,
                    "mode": mode, "donor": donor}
        emit({"event": "forward_start", **identity})
        numeric_before = denormal_probe()
        observation = observe(model, tuple(row["input_ids"]), layer=layer, position=position,
                              mask_index=4, source=source, self_patch=self_patch)
        numeric_after = denormal_probe()
        summary = summarize_logits(observation.mask_logits, sweet_id=sweet_id, sour_id=sour_id)
        emit({"event": "forward_complete", **identity, "summary": summary,
              "denormal_before": numeric_before, "denormal_after": numeric_after,
              "baseline_mask_logits_float32_le_base64": (
                  base64.b64encode(float32_bytes(observation.mask_logits)).decode("ascii")
                  if mode == "capture_ingredient" and layer == 0 else None),
              "before_sha256": _sha_tensor(observation.before),
              "after_sha256": None if observation.after is None else _sha_tensor(observation.after),
              "source_sha256": None if source is None else _sha_tensor(source),
              "before_float32_le_base64": base64.b64encode(float32_bytes(observation.before)).decode("ascii"),
              "after_float32_le_base64": None if observation.after is None else
                  base64.b64encode(float32_bytes(observation.after)).decode("ascii"),
              "source_float32_le_base64": None if source is None else
                  base64.b64encode(float32_bytes(source)).decode("ascii"),
              "full_before_sha256": observation.full_before_sha256,
              "full_after_sha256": observation.full_after_sha256,
              "untouched_positions_equal": observation.untouched_positions_equal,
              "events": list(observation.events)})
        count += 1
        return observation, summary

    for offset in range(0, len(rows), 2):
        pair = rows[offset:offset + 2]
        references = {}
        for layer in range(len(model.distilbert.transformer.layer)):
            captures = {}
            for row in pair:
                for position, role in ((7, "ingredient"), (1, "control")):
                    observed, summary = forward(row, layer=layer, position=position,
                                                mode=f"capture_{role}")
                    captures[(row["id"], position)] = observed
                    if row["id"] not in references:
                        references[row["id"]] = observed.mask_logits
                        baselines[row["id"]] = summary
                    elif not torch.equal(references[row["id"]].view(torch.int32),
                                         observed.mask_logits.view(torch.int32)):
                        raise RuntimeError("clean baseline changed between captures")
            for index, recipient in enumerate(pair):
                donor = pair[1 - index]
                base = baselines[recipient["id"]]["sweet_minus_sour"]
                for mode, position, is_self in (("ingredient_swap", 7, False),
                                                ("ingredient_self", 7, True),
                                                ("control_swap", 1, False)):
                    source = None if is_self else captures[(donor["id"], position)].before
                    observed, summary = forward(recipient, layer=layer, position=position, mode=mode,
                                                donor=recipient["id"] if is_self else donor["id"],
                                                source=source, self_patch=is_self)
                    expected_before = captures[(recipient["id"], position)].before
                    if not torch.equal(observed.before.view(torch.int32), expected_before.view(torch.int32)):
                        raise RuntimeError("pre-patch capture differs from the matching clean input")
                    if (is_self or layer == len(model.distilbert.transformer.layer) - 1) and not torch.equal(
                            observed.mask_logits.view(torch.int32), references[recipient["id"]].view(torch.int32)):
                        raise RuntimeError("self or final-block control changed mask logits")
                    patched = summary["sweet_minus_sour"]
                    effects.append({"recipient": recipient["id"], "donor": recipient["id"] if is_self else donor["id"],
                                    "layer": layer, "position": position, "mode": mode,
                                    "baseline_margin": base, "patched_margin": patched,
                                    "effect": math.fsum((patched, -base))})
        for row in pair:
            observed, _ = forward(row, layer=0, position=7, mode="final_clean")
            if not torch.equal(observed.mask_logits.view(torch.int32), references[row["id"]].view(torch.int32)):
                raise RuntimeError("final clean logits differ from the original baseline")
    return {"baselines": baselines, "effects": effects, "completed_forwards": count}


def _git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1"})
    return subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def source_identity():
    """Bind selected execution files to their actual checkout and Git blobs."""
    commit = _git("rev-parse", "HEAD").decode("ascii").strip()
    hashes = {}
    for name in SOURCE_PATHS:
        data = (ROOT / name).read_bytes()
        if data != _git("show", f"{commit}:{name}"):
            raise RuntimeError("selected checkout source differs from its Git blob")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"commit": commit, "selected_source_sha256": hashes}


def verify_cases(tokenizer):
    """Verify the public reservation, its cited inventories, and actual tokenization."""
    manifest = json.loads((ROOT / CASES_PATH).read_bytes())
    if (manifest.get("schema") != "bert-taste-public-cases-v1" or manifest.get("model") != MODEL
            or manifest.get("revision") != REVISION or len(manifest.get("rows", [])) != 8):
        raise ValueError("public case identity differs")
    if manifest.get("prior_public_inventories") != PRIOR_INVENTORIES:
        raise ValueError("cited public inventories differ")
    expected_tokens = {"mask": 103, "sugar": 5699, "lemon": 14380,
                       "sweet": 4086, "sour": 14768, "unchanged_control": 1996}
    if manifest.get("tokens") != expected_tokens or manifest.get("coordinates") != {
            "mask": 4, "ingredient": 7, "unchanged_control": 1, "sequence_length": 10}:
        raise ValueError("public token coordinates differ")
    for token in ("sugar", "lemon", "sweet", "sour"):
        if tokenizer.encode(token, add_special_tokens=False) != [expected_tokens[token]]:
            raise ValueError("actual tokenizer differs from the fixed target words")
    if tokenizer.mask_token_id != expected_tokens["mask"]:
        raise ValueError("actual mask token differs")
    rows = manifest["rows"]
    for offset, name in enumerate(("tea", "drink", "sauce", "water")):
        pair = rows[2 * offset:2 * offset + 2]
        if [row.get("id") for row in pair] != [name + "-sugar", name + "-lemon"]:
            raise ValueError("public pair order differs")
        for row, word in zip(pair, ("sugar", "lemon")):
            ids, mask = row["input_ids"], row["attention_mask"]
            actual = tokenizer(row["text"], add_special_tokens=True, padding=False,
                               truncation=False, return_attention_mask=True)
            if (len(ids) != 10 or mask != [1] * 10 or ids[4] != 103 or ids[1] != 1996
                    or ids[7] != expected_tokens[word] or ids.count(103) != 1
                    or actual["input_ids"] != ids or actual["attention_mask"] != mask):
                raise ValueError("actual tokenizer or fixed row differs")
        if any(a != b for index, (a, b) in enumerate(zip(pair[0]["input_ids"], pair[1]["input_ids"]))
               if index != 7):
            raise ValueError("matched rows differ beyond the ingredient")
    prior = []
    for path, digest in manifest["prior_public_inventories"].items():
        raw = (ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("cited public exposure inventory differs")
        previous = json.loads(raw)
        if previous["identity"]["model"] != MODEL or previous["identity"]["revision"] != REVISION:
            raise ValueError("cited public exposure identity differs")
        prior.append(previous["rows"])
    compare_exposures(rows, prior)
    return manifest, {"prior_row_counts": [len(items) for items in prior],
                      "exact_or_complete_prefix_collisions": 0,
                      "complete_exposure_history_certified": False}


def _write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        write_all(stream, raw)
        stream.flush()
        os.fsync(stream.fileno())


def write_all(stream, raw):
    remaining = memoryview(raw)
    while remaining:
        written = stream.write(remaining)
        if type(written) is not int or not 0 < written <= len(remaining):
            raise OSError("write did not make valid progress")
        remaining = remaining[written:]


def record_incomplete(output, phase, counts, original):
    try:
        _write_json(output / "incomplete.json", {"status": "incomplete", "phase": phase,
                                                 "exception_type": type(original).__name__, "counts": counts})
    except BaseException as secondary:
        if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(
                original, (KeyboardInterrupt, SystemExit)):
            raise secondary from original


def _layer_summary(effects, layers):
    result = []
    for layer in range(layers):
        entry = {"layer": layer}
        for mode in ("ingredient_swap", "ingredient_self", "control_swap"):
            entry[mode] = {}
            directions = (("lemon_self", "-lemon"), ("sugar_self", "-sugar")) if mode == "ingredient_self" else (
                ("sugar_to_lemon", "-lemon"), ("lemon_to_sugar", "-sugar"))
            for direction, suffix in directions:
                values = [item["effect"] for item in effects if item["layer"] == layer
                          and item["mode"] == mode and item["recipient"].endswith(suffix)]
                if len(values) != 4:
                    raise RuntimeError("directional effect inventory is incomplete")
                entry[mode][direction] = {"n": len(values), "mean_effect": math.fsum(values) / len(values)}
        result.append(entry)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="local Hugging Face hub cache")
    parser.add_argument("--output", required=True, help="new output directory outside the source checkout")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("hide CUDA before starting the CPU process")
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("output must be outside the source checkout")
    source_before = source_identity()
    output.mkdir(parents=False, exist_ok=False)
    _write_json(output / "started.json", {"status": "started", "source": source_before,
                                          "cases_path": CASES_PATH, "seed": SEED})
    counts = {"started": 0, "completed": 0}
    phase = "runtime_setup"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise RuntimeError("denormal preservation is unavailable")
        denormal_probe()
        torch.random.default_generator.manual_seed(SEED)
        random.seed(SEED)
        from looking_glass.bert_agreement_study import load_pinned_distilbert, capture_model_state, verify_model_state
        phase = "offline_model_load"
        loaded = load_pinned_distilbert(args.cache_dir)
        model, tokenizer = loaded.model, loaded.tokenizer
        phase = "case_and_exposure_check"
        manifest, exposure = verify_cases(tokenizer)
        witness = capture_model_state(model)
        phase = "public_forwards"
        with ExitStack() as stack:
            journals = {name: stack.enter_context((output / f"{name}.jsonl").open("xb"))
                        for name in ("tea", "drink", "sauce", "water")}

            def emit(record):
                family = record["row_id"].split("-", 1)[0]
                stream = journals[family]
                raw = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                write_all(stream, raw)
                stream.flush()
                counts["started" if record["event"] == "forward_start" else "completed"] += 1

            result = run_cases(model, manifest["rows"], sweet_id=4086, sour_id=14768, emit=emit)
            for stream in journals.values():
                stream.flush()
                os.fsync(stream.fileno())
        journal_files = {}
        for name in ("tea", "drink", "sauce", "water"):
            path = output / f"{name}.jsonl"
            data = path.read_bytes()
            journal_files[path.name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if counts != {"started": 248, "completed": 248} or result["completed_forwards"] != 248:
            raise RuntimeError("forward inventory is incomplete")
        phase = "final_checks"
        verify_model_state(model, witness)
        if has_hooks(model) or torch.cuda.is_initialized():
            raise RuntimeError("model cleanup or CPU-only state is uncertain")
        source_after = source_identity()
        if source_before != source_after:
            raise RuntimeError("selected source changed during execution")
        for baseline in result["baselines"].values():
            baseline["top_five_tokens"] = tokenizer.convert_ids_to_tokens(baseline["top_five_ids"])
        result.update({"status": "complete", "schema": "bert-taste-context-result-v1",
                       "source": source_before, "model": loaded.metadata, "seed": SEED,
                       "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                                   "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                   "default_dtype": str(torch.get_default_dtype()),
                                   "mkldnn_enabled": torch.backends.mkldnn.enabled,
                                   "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode("utf-8")).hexdigest(),
                                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                                   "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                                   "mkl_cbwr": os.environ.get("MKL_CBWR"),
                                   "aten_cpu_capability": os.environ.get("ATEN_CPU_CAPABILITY"),
                                   "denormal_probe": denormal_probe(),
                                   "cuda_initialized": torch.cuda.is_initialized()},
                       "exposure": exposure, "model_state_sha256_before_after": witness.sha256,
                       "journals": journal_files,
                       "layer_summary": _layer_summary(result["effects"], len(model.distilbert.transformer.layer)),
                       "limitations": ["Public exploratory inputs; no held-out or generalization claim.",
                           "Full-output untouched-coordinate equality is owner-observed; only its hashes are saved.",
                           "Full baseline mask-logit bits are saved; patched full-vocabulary summaries are owner-observed.",
                           "Source hashes and local execution records do not independently attest execution."]})
        _write_json(output / "result.json", result)
        return 0
    except BaseException as exc:
        record_incomplete(output, phase, counts, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
