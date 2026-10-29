"""Public CPU matched-dose taste-context follow-up; no downloads."""

import sys

if __name__ in ("__main__", "<run_path>") and (
        not sys.flags.isolated or not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError("isolated no-bytecode child with a fresh prefix is required")

import argparse
import base64
from contextlib import ExitStack
import hashlib
import heapq
import importlib.machinery
import json
import math
import os
from pathlib import Path
import platform
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261030
CASES_PATH = "experiments/bert_taste_matched_dose_public_cases_v1.json"
FRAMES = ("coffee", "milk", "soup", "cake", "pudding", "cocoa")
PRIOR_INVENTORIES = {
    "experiments/bert_agreement_exposures_v6.json": "0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb",
    "experiments/bert_matched_public_cases_v1.json": "5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb",
    "experiments/bert_taste_public_exposures_v2.json": "a766d20778191280122a6b498b9bcb34907bfac761152ae7c3cf4df0fb9c5aa1",
}
SOURCE_PATHS = (
    "scripts/bert_taste_matched_dose_v1.py", "scripts/bert_taste_matched_parent_v1.py",
    "scripts/bert_taste_matched_independent.py",
    "docs/BERT_TASTE_MATCHED_DOSE_V1.md", "experiments/bert_taste_matched_dose_public_cases_v1.json",
    "experiments/bert_agreement_exposures_v6.json", "experiments/bert_matched_public_cases_v1.json",
    "experiments/bert_taste_public_exposures_v2.json", "scripts/bert_taste_patch_v1.py",
    "looking_glass/__init__.py", "looking_glass/pretrained.py", "looking_glass/bert_agreement_study.py",
    "looking_glass/bert_agreement_runner.py", "looking_glass/intervention_plan.py",
    "looking_glass/interventions.py", "looking_glass/model_ownership.py",
)


class _CheckoutImportGuard:
    """Allow only declared checkout Python source through ordinary import."""
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        project = fullname == "scripts" or fullname.startswith("scripts.") or fullname == "looking_glass" or fullname.startswith("looking_glass.")
        if spec is None:
            if project:
                raise ImportError("declared project module cannot be resolved")
            return None
        if spec.origin not in (None, "built-in", "frozen"):
            lexical = Path(spec.origin).absolute()
            resolved = lexical.resolve()
            in_checkout = lexical.is_relative_to(ROOT) or resolved.is_relative_to(ROOT)
            if in_checkout:
                allowed = {ROOT / name for name in SOURCE_PATHS if name.endswith(".py")}
                if (resolved not in allowed or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)):
                    raise ImportError("checkout import is not an allowed source file")
                return spec
            if project:
                raise ImportError("project module resolved outside the checkout")
        elif spec.submodule_search_locations is not None:
            allowed_dirs = {ROOT / "scripts", ROOT / "looking_glass"}
            for item in spec.submodule_search_locations:
                location = Path(item).resolve()
                if location.is_relative_to(ROOT) and location not in allowed_dirs:
                    raise ImportError("checkout namespace is not allowed")
            if project:
                if fullname != "scripts" or [Path(item).resolve() for item in spec.submodule_search_locations] != [ROOT / "scripts"]:
                    raise ImportError("project namespace differs")
                return spec
        elif project:
            raise ImportError("project import has no source origin")
        return None


# Guard the one-shot child only; normal library imports must not change the
# host process's import policy. The prior module sees this exact root already
# present in the child and therefore does not prepend it.
if __name__ in ("__main__", "<run_path>"):
    sys.path[:] = [entry for entry in sys.path
                   if not Path(entry or os.getcwd()).resolve().is_relative_to(ROOT)]
    sys.path.append(str(ROOT))
    if any(name == "scripts" or name.startswith("scripts.") or name == "looking_glass" or name.startswith("looking_glass.")
           for name in sys.modules):
        raise ImportError("project modules were loaded before the checkout guard")
    sys.meta_path.insert(0, _CheckoutImportGuard())

import torch

from scripts import bert_taste_patch_v1 as previous

if Path(previous.__file__).resolve() != ROOT / "scripts/bert_taste_patch_v1.py":
    raise ImportError("prior taste source origin differs")
for _name in ("bert_agreement_runner", "intervention_plan", "interventions", "model_ownership"):
    _module = sys.modules.get("looking_glass." + _name)
    if _module is None or Path(_module.__file__).resolve() != ROOT / "looking_glass" / (_name + ".py"):
        raise ImportError("loaded project source origin differs")

MODEL = previous.MODEL
REVISION = previous.REVISION


def _values(vector):
    if (type(vector) is not torch.Tensor or vector.device.type != "cpu"
            or vector.dtype != torch.float32 or vector.ndim != 1 or vector.numel() == 0):
        raise ValueError("expected a nonempty CPU float32 vector")
    raw = previous.float32_bytes(vector)
    return [previous.decode_float32_bits(bits) for (bits,) in struct.iter_unpack("<I", raw)]


def _distance(left, right):
    if len(left) != len(right):
        raise ValueError("vector lengths differ")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))


def rounding_probe():
    # Halfway cases distinguish round-to-nearest-even from directed rounding.
    if (struct.pack("<f", 1 + 2 ** -24) != struct.pack("<I", 0x3f800000)
            or struct.pack("<f", 1 + 3 * 2 ** -24) != struct.pack("<I", 0x3f800002)):
        raise RuntimeError("float32 nearest-even rounding is unavailable")
    return "nearest-even-halfway-cases-pass"


def _as_float32(values):
    try:
        raw = b"".join(struct.pack("<f", value) for value in values)
    except (OverflowError, struct.error) as exc:
        raise ValueError("nonfinite interpolated source") from exc
    result = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    _values(result)
    return result


def matched_sources(sugar_ingredient, lemon_ingredient, sugar_control, lemon_control):
    """Construct all four non-extrapolating sources and verify their doses."""
    previous.denormal_probe()
    rounding_probe()
    values = {"sugar": {"ingredient": _values(sugar_ingredient), "control": _values(sugar_control)},
              "lemon": {"ingredient": _values(lemon_ingredient), "control": _values(lemon_control)}}
    distances = {site: _distance(values["sugar"][site], values["lemon"][site])
                 for site in ("ingredient", "control")}
    radius = min(distances.values())
    if not all(math.isfinite(x) and x >= 1e-5 for x in distances.values()):
        raise ValueError("matched dose is nonfinite or below the fixed minimum")
    fractions = {site: radius / distances[site] for site in distances}
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in fractions.values()):
        raise ValueError("matched interpolation fraction is invalid")
    actions = {}
    for direction, recipient, donor in (("sugar_to_lemon", "lemon", "sugar"),
                                        ("lemon_to_sugar", "sugar", "lemon")):
        actions[direction] = {}
        for site in ("ingredient", "control"):
            before, other, fraction = values[recipient][site], values[donor][site], fractions[site]
            requested = [math.fsum((a, fraction * (b - a))) for a, b in zip(before, other)]
            if any(not math.isfinite(x) for x in requested):
                raise ValueError("matched source is nonfinite")
            source = _as_float32(requested)
            delivered = _distance(_values(source), before)
            if not math.isfinite(delivered) or delivered == 0 or abs(delivered - radius) > 1e-6 + 1e-4 * radius:
                raise ValueError("rounded matched dose is outside tolerance")
            base_norm = _distance(before, [0.0] * len(before))
            actions[direction][site] = {"source": source, "fraction": fraction,
                                        "requested_norm": radius, "delivered_norm": delivered,
                                        "relative_error": abs(delivered - radius) / radius,
                                        "recipient_norm": base_norm,
                                        "recipient_norm_status": "zero" if base_norm == 0 else "positive",
                                        "dose_to_recipient_norm": None if base_norm == 0 else delivered / base_norm}
    previous.denormal_probe()
    return {"radius": radius, "distances": distances, "fractions": fractions, "actions": actions}


def top_five(logits):
    values = _values(logits)
    if len(values) < 5:
        raise ValueError("at least five logits are required")
    return heapq.nsmallest(5, range(len(values)), key=lambda index: (-values[index], index))


def eligible(sugar, lemon, sweet_id, sour_id):
    targets = {sweet_id, sour_id}
    return (sugar["sweet_minus_sour"] > 0 and lemon["sweet_minus_sour"] < 0
            and bool(targets.intersection(sugar["top_five_ids"]))
            and bool(targets.intersection(lemon["top_five_ids"])))


def observe_entry(model, ids, *, layer, position, mask_index, source=None, self_patch=False):
    """Record actual arguments at the model-call boundary and restore the method."""
    if "forward" in model.__dict__:
        raise ValueError("model forward override is already present")
    original = model.forward
    entries = []

    def observed(*args, **kwargs):
        if args or entries or set(kwargs) != {"input_ids", "attention_mask", "output_attentions",
                                               "output_hidden_states", "return_dict"}:
            raise RuntimeError("model entry arguments differ")
        inputs, mask = kwargs["input_ids"], kwargs["attention_mask"]
        if (type(inputs) is not torch.Tensor or type(mask) is not torch.Tensor
                or inputs.dtype != torch.int64 or mask.dtype != torch.int64
                or inputs.device.type != "cpu" or mask.device.type != "cpu"
                or tuple(inputs.shape) != (1, 10) or tuple(mask.shape) != (1, 10)
                or tuple(inputs[0].tolist()) != ids or tuple(mask[0].tolist()) != (1,) * 10
                or kwargs["output_attentions"] is not False or kwargs["output_hidden_states"] is not False
                or kwargs["return_dict"] is not True):
            raise RuntimeError("actual model entry differs from the fixed call")
        entries.append({"input_ids": inputs[0].tolist(), "attention_mask": mask[0].tolist()})
        return original(*args, **kwargs)

    model.forward = observed
    try:
        result = previous.observe(model, ids, layer=layer, position=position,
                                  mask_index=mask_index, source=source, self_patch=self_patch)
    finally:
        del model.forward
    if len(entries) != 1:
        raise RuntimeError("model entry did not occur exactly once")
    return result, entries[0]


def _bits(tensor):
    return base64.b64encode(previous.float32_bytes(tensor)).decode("ascii")


def _same(left, right):
    return previous.float32_bytes(left) == previous.float32_bytes(right)


def _summary(logits, sweet_id, sour_id, tokenizer):
    result = previous.summarize_logits(logits, sweet_id=sweet_id, sour_id=sour_id)
    result["top_five_ids"] = top_five(logits)
    result["top_five_tokens"] = tokenizer.convert_ids_to_tokens(result["top_five_ids"])
    return result


def run_cases(model, rows, *, sweet_id, sour_id, tokenizer, emit):
    """Run all pairs in fixed order; eligibility does not gate any forward."""
    if type(rows) is not list or not rows or len(rows) % 2 or not callable(emit):
        raise ValueError("expected adjacent pairs and a journal callback")
    baselines, effects, doses, eligibility = {}, [], [], {}
    completed = 0

    def forward(row, layer, position, mode, *, donor=None, source=None, self_patch=False,
                expected_before=None, donor_clean=None, dose=None):
        nonlocal completed
        identity = {"row_id": row["id"], "layer": layer, "position": position,
                    "mode": mode, "donor": donor}
        emit({"event": "forward_start", **identity})
        before_probe = previous.denormal_probe()
        observed, entry = observe_entry(model, tuple(row["input_ids"]), layer=layer,
                                        position=position, mask_index=4, source=source,
                                        self_patch=self_patch)
        after_probe = previous.denormal_probe()
        if expected_before is not None and not _same(observed.before, expected_before):
            raise RuntimeError("same-invocation pre-action differs from the clean recipient")
        actual_norm = None
        if dose is not None:
            actual_norm = _distance(_values(observed.after), _values(observed.before))
            if not math.isclose(actual_norm, dose["delivered_norm"], rel_tol=0, abs_tol=0):
                raise RuntimeError("actual intervention norm differs from planned float32 dose")
        summary = _summary(observed.mask_logits, sweet_id, sour_id, tokenizer)
        score_probe = previous.denormal_probe()
        emit({"event": "forward_complete", **identity, "model_entry": entry, "summary": summary,
              "denormal_before": before_probe, "denormal_after": after_probe,
              "denormal_after_scoring": score_probe,
              "baseline_mask_logits_float32_le_base64": _bits(observed.mask_logits)
                  if mode == "capture_ingredient" and layer == 0 else None,
              "before_float32_le_base64": _bits(observed.before),
              "after_float32_le_base64": None if observed.after is None else _bits(observed.after),
              "source_float32_le_base64": None if source is None else _bits(source),
              "donor_clean_float32_le_base64": None if donor_clean is None else _bits(donor_clean),
              "full_before_sha256": observed.full_before_sha256,
              "full_after_sha256": observed.full_after_sha256,
              "untouched_positions_equal": observed.untouched_positions_equal,
              "actual_pre_post_norm": actual_norm,
              "dose": None if dose is None else {k: v for k, v in dose.items() if k != "source"},
              "events": list(observed.events)})
        completed += 1
        return observed, summary

    for offset in range(0, len(rows), 2):
        pair = rows[offset:offset + 2]
        references = {}
        frame = pair[0]["id"].rsplit("-", 1)[0]
        for layer in range(len(model.distilbert.transformer.layer)):
            captures = {}
            for row in pair:
                for position, site in ((7, "ingredient"), (1, "control")):
                    observed, summary = forward(row, layer, position, "capture_" + site)
                    captures[(row["id"], site)] = observed.before
                    if row["id"] not in references:
                        references[row["id"]] = observed.mask_logits
                        baselines[row["id"]] = summary
                    elif not _same(references[row["id"]], observed.mask_logits):
                        raise RuntimeError("clean mask logits changed between captures")
            if frame not in eligibility:
                eligibility[frame] = eligible(baselines[pair[0]["id"]], baselines[pair[1]["id"]],
                                              sweet_id, sour_id)
            try:
                plan = matched_sources(captures[(pair[0]["id"], "ingredient")],
                                       captures[(pair[1]["id"], "ingredient")],
                                       captures[(pair[0]["id"], "control")],
                                       captures[(pair[1]["id"], "control")])
            except (ValueError, RuntimeError) as exc:
                emit({"event": "dose_unavailable", "row_id": pair[0]["id"], "layer": layer,
                      "exception_type": type(exc).__name__, "reason": str(exc)})
                raise
            dose_record = {"frame": frame, "layer": layer, "radius": plan["radius"],
                           "distances": plan["distances"], "fractions": plan["fractions"]}
            emit({"event": "dose_plan", "row_id": pair[0]["id"], **dose_record})
            doses.append(dose_record)
            for index, recipient in enumerate(pair):
                donor = pair[1 - index]
                direction = "lemon_to_sugar" if index == 0 else "sugar_to_lemon"
                base = baselines[recipient["id"]]["sweet_minus_sour"]
                for mode, site, is_self, matched in (
                        ("ingredient_donor", "ingredient", False, False),
                        ("ingredient_self", "ingredient", True, False),
                        ("control_self", "control", True, False),
                        ("ingredient_matched", "ingredient", False, True),
                        ("control_matched", "control", False, True)):
                    position = 7 if site == "ingredient" else 1
                    donor_clean = None if is_self else captures[(donor["id"], site)]
                    action = plan["actions"][direction][site] if matched else None
                    source = None if is_self else action["source"] if matched else donor_clean
                    observed, summary = forward(recipient, layer, position, mode,
                                                donor=recipient["id"] if is_self else donor["id"],
                                                source=source, self_patch=is_self,
                                                expected_before=captures[(recipient["id"], site)],
                                                donor_clean=donor_clean, dose=action)
                    if (is_self or layer == len(model.distilbert.transformer.layer) - 1) and not _same(
                            observed.mask_logits, references[recipient["id"]]):
                        raise RuntimeError("self or final-block action changed mask logits")
                    patched = summary["sweet_minus_sour"]
                    effects.append({"frame": frame, "recipient": recipient["id"],
                                    "donor": recipient["id"] if is_self else donor["id"],
                                    "layer": layer, "position": position, "direction": direction,
                                    "mode": mode, "baseline_margin": base, "patched_margin": patched,
                                    "effect": math.fsum((patched, -base)), "summary": summary})
        for row in pair:
            observed, _ = forward(row, 0, 7, "final_clean")
            if not _same(observed.mask_logits, references[row["id"]]):
                raise RuntimeError("final clean logits changed")

    indexed = {(e["frame"], e["layer"], e["direction"], e["mode"]): e["effect"] for e in effects}
    comparisons, layers = [], []
    for layer in range(len(model.distilbert.transformer.layer)):
        group = {"layer": layer}
        for direction, sign in (("sugar_to_lemon", 1), ("lemon_to_sugar", -1)):
            values = []
            for frame in eligibility:
                ingredient = indexed[(frame, layer, direction, "ingredient_matched")]
                control = indexed[(frame, layer, direction, "control_matched")]
                contrast = math.fsum((sign * ingredient, -sign * control))
                comparisons.append({"frame": frame, "layer": layer, "direction": direction,
                                    "eligible": eligibility[frame], "signed_advantage": contrast})
                if eligibility[frame]:
                    values.append(contrast)
            group[direction] = {"n": len(values), "mean": None if not values else math.fsum(values) / len(values)}
        layers.append(group)
    return {"baselines": baselines, "eligibility": eligibility, "effects": effects,
            "doses": doses, "comparisons": comparisons, "layer_summary": layers,
            "completed_forwards": completed}


def source_identity():
    commit = previous._git("rev-parse", "HEAD").decode("ascii").strip()
    hashes = {}
    for name in SOURCE_PATHS:
        data = (ROOT / name).read_bytes()
        if data != previous._git("show", f"{commit}:{name}"):
            raise RuntimeError("selected source differs from its Git blob")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"commit": commit, "selected_source_sha256": hashes}


def verify_cases(tokenizer):
    manifest = json.loads((ROOT / CASES_PATH).read_bytes())
    if (manifest.get("schema") != "bert-taste-matched-dose-public-cases-v1"
            or manifest.get("model") != MODEL or manifest.get("revision") != REVISION
            or manifest.get("prior_public_inventories") != PRIOR_INVENTORIES
            or manifest.get("coordinates") != {"mask": 4, "ingredient": 7, "unchanged_control": 1,
                                                "sequence_length": 10}
            or manifest.get("tokens") != {"mask": 103, "sugar": 5699, "lemon": 14380,
                                           "sweet": 4086, "sour": 14768, "unchanged_control": 1996}):
        raise ValueError("public case metadata differs")
    if tokenizer.mask_token_id != 103:
        raise ValueError("mask token differs")
    for token, expected in (("sugar", 5699), ("lemon", 14380), ("sweet", 4086), ("sour", 14768)):
        if tokenizer.encode(token, add_special_tokens=False) != [expected]:
            raise ValueError("target tokenization differs")
    rows = manifest.get("rows")
    if type(rows) is not list or len(rows) != 12:
        raise ValueError("public case count differs")
    for index, frame in enumerate(FRAMES):
        pair = rows[index * 2:index * 2 + 2]
        if [r.get("id") for r in pair] != [frame + "-sugar", frame + "-lemon"]:
            raise ValueError("public case order differs")
        for row, word in zip(pair, ("sugar", "lemon")):
            ids, mask = row["input_ids"], row["attention_mask"]
            actual = tokenizer(row["text"], add_special_tokens=True, padding=False,
                               truncation=False, return_attention_mask=True)
            if (len(ids) != 10 or mask != [1] * 10 or ids[1] != 1996 or ids[4] != 103
                    or ids[7] != {"sugar": 5699, "lemon": 14380}[word] or ids.count(103) != 1
                    or actual["input_ids"] != ids or actual["attention_mask"] != mask):
                raise ValueError("fixed input or tokenizer differs")
        if any(a != b for i, (a, b) in enumerate(zip(pair[0]["input_ids"], pair[1]["input_ids"])) if i != 7):
            raise ValueError("pair differs outside ingredient position")
    priors = []
    for name, expected in PRIOR_INVENTORIES.items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("public inventory digest differs")
        inventory = json.loads(raw)
        identity = inventory.get("identity", inventory)
        if identity.get("model") != MODEL or identity.get("revision") != REVISION:
            raise ValueError("public inventory model differs")
        priors.append(inventory["rows"])
    previous.compare_exposures(rows, priors)
    for index, row in enumerate(rows):
        previous.compare_exposures([row], [rows[index + 1:]])
    return manifest, {"prior_row_counts": [len(x) for x in priors],
                      "exact_or_complete_prefix_collisions": 0,
                      "complete_exposure_history_certified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("hide CUDA before starting the CPU process")
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("output must be outside the source checkout")
    source_before = source_identity()
    output.mkdir(parents=False, exist_ok=False)
    previous._write_json(output / "started.json", {"status": "started", "source": source_before,
                                                     "cases_path": CASES_PATH, "seed": SEED})
    counts = {"started": 0, "completed": 0}
    phase = "runtime_setup"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise RuntimeError("denormal preservation is unavailable")
        previous.denormal_probe()
        rounding_probe()
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
            journals = {frame: stack.enter_context((output / f"{frame}.jsonl").open("xb")) for frame in FRAMES}

            def emit(record):
                stream = journals[record["row_id"].rsplit("-", 1)[0]]
                raw = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                previous.write_all(stream, raw)
                stream.flush()
                if record["event"] == "forward_start":
                    counts["started"] += 1
                elif record["event"] == "forward_complete":
                    counts["completed"] += 1

            result = run_cases(model, manifest["rows"], sweet_id=4086, sour_id=14768, tokenizer=tokenizer, emit=emit)
            for stream in journals.values():
                stream.flush()
                os.fsync(stream.fileno())
        journal_files = {}
        for frame in FRAMES:
            path = output / (frame + ".jsonl")
            raw = path.read_bytes()
            journal_files[path.name] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if counts != {"started": 516, "completed": 516} or result["completed_forwards"] != 516:
            raise RuntimeError("forward inventory is incomplete")
        phase = "final_checks"
        verify_model_state(model, witness)
        if previous.has_hooks(model) or torch.cuda.is_initialized():
            raise RuntimeError("model cleanup or CPU-only state is uncertain")
        source_after = source_identity()
        if source_before != source_after:
            raise RuntimeError("selected source changed during execution")
        result.update({"status": "complete", "schema": "bert-taste-matched-dose-result-v1",
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
                                   "denormal_probe": previous.denormal_probe(), "rounding_probe": rounding_probe(),
                                   "cuda_initialized": torch.cuda.is_initialized()},
                       "exposure": exposure, "model_state_sha256_before_after": witness.sha256,
                       "journals": journal_files,
                       "limitations": ["Public convenience frames; no held-out or generalization claim.",
                           "Whole-output untouched coordinates and patched full-vocabulary summaries are owner-observed.",
                           "Source hashes and direct-child observation do not authenticate execution or whole-tree cleanup."]})
        previous._write_json(output / "result.json", result)
        return 0
    except BaseException as exc:
        previous.record_incomplete(output, phase, counts, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
