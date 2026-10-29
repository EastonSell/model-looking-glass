"""Fixed CPU-only matched cue-class experiment and its strict decision."""
import itertools
import math

import torch

from .bert_negation_cue import (_bits, _has_hooks, _margin, _same_bits, baseline_summary,
                                distance, make_patch)
from .intervention_plan import ActivationPlan

CUES = ("never", "rarely", "always", "often")
NEGATIVE = ("never", "rarely")
AFFIRMATIVE = ("always", "often")
SITES = ("cue2", "object2", "mask2", "cue5")
PAIRS = tuple(itertools.combinations(CUES, 2))
TRANSFERS = tuple((donor, target) for target in CUES for donor in CUES if donor != target)
FRAMES = ("main", "embedded", "relative")


def baseline_screen(margins):
    if type(margins) is not dict or set(margins) != set(CUES):
        raise ValueError("four clean cue margins required")
    gaps = []
    for negative in NEGATIVE:
        for affirmative in AFFIRMATIVE:
            value = math.fsum((margins[negative], -margins[affirmative]))
            if not math.isfinite(value):
                raise ValueError("nonfinite baseline gap")
            gaps.append({"negative": negative, "affirmative": affirmative, "gap": value})
    return {"eligible": all(item["gap"] > 0.5 for item in gaps), "gaps": gaps}


def context_specificity(effects):
    if type(effects) is not dict or set(effects) != set(TRANSFERS):
        raise ValueError("all 12 directed effects are required")
    targets = {}
    for target in CUES:
        own = NEGATIVE if target in NEGATIVE else AFFIRMATIVE
        opposite = AFFIRMATIVE if target in NEGATIVE else NEGATIVE
        sign = -1 if target in NEGATIVE else 1
        other = next(cue for cue in own if cue != target)
        cross = {cue: sign * effects[cue, target] for cue in opposite}
        same = effects[other, target]
        if any(not math.isfinite(value) for value in (*cross.values(), same)):
            raise ValueError("nonfinite transfer effect")
        minimum = min(cross.values())
        margin = math.fsum((minimum, -abs(same)))
        targets[target] = {"cross_signed": cross, "cross_min": minimum, "same_donor": other,
                           "same_delta": same, "same_abs": abs(same), "margin": margin,
                           "passed": margin > 0.05}
    return {"targets": targets, "passed": all(row["passed"] for row in targets.values())}


def evaluate(rows):
    if type(rows) is not list or len(rows) != 9:
        raise ValueError("the evaluation requires all nine fixed contexts")
    if (len({row["id"] for row in rows}) != 9 or any(row.get("split") != "evaluation" for row in rows)
            or any(row.get("frame") not in FRAMES for row in rows)):
        raise ValueError("evaluation case identities or frames differ")
    frames = {}
    for frame in FRAMES:
        cases = [row for row in rows if row["frame"] == frame]
        if len(cases) != 3:
            raise ValueError("each frame requires three evaluation contexts")
        eligible = [row for row in cases if row["baseline_eligible"] and row["dose_eligible"]]
        passing = [row for row in eligible if row["specificity_passed"]]
        frames[frame] = {"total": 3, "eligible": len(eligible), "passing": len(passing)}
    if any(value["eligible"] < 2 for value in frames.values()):
        status = "unevaluable"
    elif all(value["passing"] >= 2 for value in frames.values()):
        status = "passed"
    else:
        status = "failed"
    return {"status": status, "frames": frames}


def choose_radius(captures):
    distances = [{"left": a, "right": b, "site": site,
                  "distance": distance(captures[a][site], captures[b][site])}
                 for a, b in PAIRS for site in SITES]
    return min(row["distance"] for row in distances), distances


class CompletedObservationFailure(RuntimeError):
    def __init__(self, phase, payload):
        super().__init__(f"completed forward failed {phase} control")
        self.phase = phase
        self.payload = payload


def _sites(indices):
    return {"cue2": (2, indices["cue"]), "object2": (2, indices["object"]),
            "mask2": (2, indices["mask"]), "cue5": (5, indices["cue"])}


def _logits_finite(logits):
    return bool(torch.isfinite(logits).all())


def observe(model, ids, indices, *, site=None, source=None, self_patch=False, expected_vocab=None, expected_width=None):
    if (not isinstance(model, torch.nn.Module) or any(m.training for m in model.modules()) or _has_hooks(model)
            or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or any(b.device.type != "cpu" for b in model.buffers()) or torch.is_autocast_enabled("cpu")):
        raise ValueError("expected an unhooked CPU float32 evaluation model")
    if (type(ids) is not torch.Tensor or ids.device.type != "cpu" or ids.dtype != torch.int64
            or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 64):
        raise ValueError("invalid input tensor")
    if (type(indices) is not dict or set(indices) != {"cue", "mask", "object"}
            or len(set(indices.values())) != 3 or any(type(i) is not int or not 0 <= i < ids.shape[1]
                                                       for i in indices.values())):
        raise ValueError("invalid fixed token indices")
    if type(self_patch) is not bool or (self_patch and source is not None):
        raise ValueError("invalid patch mode")
    if expected_vocab is not None and (type(expected_vocab) is not int or not 1 <= expected_vocab <= 100000):
        raise ValueError("invalid expected vocabulary width")
    if expected_width is not None and (type(expected_width) is not int or not 1 <= expected_width <= 4096):
        raise ValueError("invalid expected hidden width")
    sites = _sites(indices)
    if site is None:
        if source is not None or self_patch:
            raise ValueError("clean observation cannot patch")
        steps = [{"name": name, "operation": "capture", "module_path": f"distilbert.transformer.layer.{layer}",
                  "invocation": 0, "batch_index": 0, "token_index": token}
                 for name, (layer, token) in sites.items()]
    else:
        if site not in sites or (source is None and not self_patch):
            raise ValueError("invalid patch site or source")
        layer, token = sites[site]
        whole = {"module_path": f"distilbert.transformer.layer.{layer}", "invocation": 0}
        one = {**whole, "batch_index": 0, "token_index": token}
        patch = {"name": "patch", "operation": "patch", **one}
        patch.update({"source_capture": "before"} if self_patch else {"value": source})
        steps = [{"name": "full_before", "operation": "capture", **whole},
                 {"name": "before", "operation": "capture", **one}, patch,
                 {"name": "after", "operation": "capture", **one},
                 {"name": "full_after", "operation": "capture", **whole}]
    original, attention = ids.clone(), torch.ones_like(ids)
    plan = ActivationPlan(model, steps, max_tensor_bytes=1024 * 1024, max_capture_bytes=2 * 1024 * 1024,
                          max_source_bytes=1024 * 1024, max_steps=5, max_invocations=1)
    output, returned = None, False
    try:
        with torch.inference_mode(), plan:
            output = model(input_ids=ids, attention_mask=attention, output_attentions=False,
                           output_hidden_states=False, return_dict=True)
            returned = True
    except BaseException as exc:
        if not returned:
            raise
        # Plan exit may clear captures even after a model has returned. Preserve
        # only an independently bounded mask-logit slice and label the missing
        # captures; do not treat the failed plan as a successful observation.
        payload = {}
        try:
            logits = output.logits
            if (type(logits) is torch.Tensor and logits.device.type == "cpu" and logits.dtype == torch.float32
                    and logits.ndim == 3 and tuple(logits.shape[:2]) == tuple(ids.shape)
                    and 1 <= logits.shape[2] <= 100000):
                payload["mask_logits"] = logits[0, indices["mask"]].detach().clone()
        except Exception:
            pass
        raise CompletedObservationFailure("plan_exit", payload) from exc
    captures = plan.captures
    payload = {"captures": captures} if site is None else {
        "before": captures["before"], "after": captures["after"],
        "full_before": captures["full_before"], "full_after": captures["full_after"]}
    try:
        logits = output.logits
        if (type(logits) is not torch.Tensor or logits.device.type != "cpu" or logits.dtype != torch.float32
                or logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(ids.shape)
                or not 1 <= logits.shape[2] <= 100000):
            raise ValueError("invalid or oversized model logits")
        payload["mask_logits"] = logits[0, indices["mask"]].detach().clone()
        if expected_vocab is not None and logits.shape[2] != expected_vocab:
            raise ValueError("model vocabulary width differs from the fixed checkpoint")
        if not _logits_finite(logits):
            raise ValueError("nonfinite model logits")
    except BaseException as exc:
        raise CompletedObservationFailure("logits_validation", payload) from exc
    if expected_width is not None:
        if site is None:
            valid_shapes = all(tuple(captures[name].shape) == (expected_width,) for name in SITES)
        else:
            valid_shapes = (tuple(captures["before"].shape) == tuple(captures["after"].shape) == (expected_width,)
                            and tuple(captures["full_before"].shape) == tuple(captures["full_after"].shape)
                            == (1, ids.shape[1], expected_width))
        if not valid_shapes:
            raise CompletedObservationFailure("capture_shape", payload)
    if _has_hooks(model) or not torch.equal(ids, original) or not torch.equal(attention, torch.ones_like(attention)):
        raise CompletedObservationFailure("hook_or_input", payload)
    if site is None:
        return payload
    expected = captures["before"] if self_patch else source
    if not _same_bits(captures["after"], expected):
        raise CompletedObservationFailure("landing", payload)
    if any(not _same_bits(captures["full_before"][:, i], captures["full_after"][:, i])
           for i in range(ids.shape[1]) if i != sites[site][1]):
        raise CompletedObservationFailure("other_tokens", payload)
    return payload


def _emit_witness_failure(emit, meta, ids, phase, payload, **extra):
    record = {"kind": "forward_failed", "phase": phase, "input_ids": ids[0].tolist(), **meta, **extra}
    record["omitted_components"] = [] if "mask_logits" in payload else ["mask_logits_unavailable_or_unbounded"]
    if phase == "plan_exit":
        record["omitted_components"].append("plan_captures_unavailable_or_cleared")
    for key, value in payload.items():
        if key == "captures":
            record["captures_f32le"] = {name: _bits(tensor) for name, tensor in value.items()}
        else:
            record[key + "_f32le"] = _bits(value)
    emit(record)


def _observed(model, ids, indices, emit, meta, **kwargs):
    try:
        return observe(model, ids, indices, **kwargs)
    except CompletedObservationFailure as exc:
        extra = {"submitted_f32le": _bits(kwargs["source"])} if kwargs.get("source") is not None else {}
        _emit_witness_failure(emit, meta, ids, exc.phase, exc.payload, **extra)
        raise


def run_case(model, case, emit, *, any_id, some_id, expected_vocab=None, expected_width=None):
    case_id, indices = case["id"], case["indices"]
    rows = {cue: torch.tensor([case["rows"][cue]["input_ids"]], dtype=torch.int64) for cue in CUES}
    clean, captures, margins, baselines = {}, {}, {}, {}
    forwards = 0
    for cue in CUES:
        observed = _observed(model, rows[cue], indices, emit, {"case_id": case_id, "cue": cue},
                             expected_vocab=expected_vocab, expected_width=expected_width)
        forwards += 1
        clean[cue], captures[cue] = observed["mask_logits"], observed["captures"]
        emit({"kind": "clean", "case_id": case_id, "cue": cue, "input_ids": rows[cue][0].tolist(),
              "mask_logits_f32le": _bits(clean[cue]),
              "captures_f32le": {name: _bits(captures[cue][name]) for name in SITES}})
        baselines[cue] = baseline_summary(clean[cue], any_id=any_id, some_id=some_id)
        margins[cue] = baselines[cue]["any_minus_some"]
    baseline = baseline_screen(margins)
    emit({"kind": "baseline", "case_id": case_id, **baseline})
    radius, distances = choose_radius(captures)
    if not math.isfinite(radius):
        raise ValueError("nonfinite common radius")
    eligible_dose = radius >= 0.001
    emit({"kind": "radius", "case_id": case_id, "radius": radius, "distances": distances,
          "binding_edges": [item for item in distances if item["distance"] == radius],
          "dose_eligible": eligible_dose})
    all_effects = {}
    if not eligible_dose:
        emit({"kind": "dose_ineligible", "case_id": case_id, "radius": radius})
    else:
        for site in SITES:
            for cue in CUES:
                meta = {"case_id": case_id, "site": site, "target": cue}
                observed = _observed(model, rows[cue], indices, emit, meta, site=site, self_patch=True,
                                     expected_vocab=expected_vocab, expected_width=expected_width)
                forwards += 1
                emit({"kind": "self", **meta, "input_ids": rows[cue][0].tolist(),
                      **{key + "_f32le": _bits(value) for key, value in observed.items()}})
                if not _same_bits(observed["before"], captures[cue][site]) or not _same_bits(observed["mask_logits"], clean[cue]):
                    raise RuntimeError("self or clean-state control failed")
            site_effects = {}
            for donor, target in TRANSFERS:
                submitted, planned, fraction = make_patch(captures[target][site], captures[donor][site], radius)
                meta = {"case_id": case_id, "site": site, "donor": donor, "target": target}
                observed = _observed(model, rows[target], indices, emit, meta, site=site, source=submitted,
                                     expected_vocab=expected_vocab, expected_width=expected_width)
                forwards += 1
                try:
                    actual = distance(observed["before"], observed["after"])
                except BaseException:
                    _emit_witness_failure(emit, meta, rows[target], "dose_measurement", observed,
                                          submitted_f32le=_bits(submitted), requested_radius=radius,
                                          requested_fraction=fraction, planned_norm=planned)
                    raise
                emit({"kind": "transfer", **meta, "input_ids": rows[target][0].tolist(),
                      "requested_radius": radius, "requested_fraction": fraction, "planned_norm": planned,
                      "actual_norm": actual, "relative_error": abs(actual - radius) / radius,
                      "submitted_f32le": _bits(submitted),
                      **{key + "_f32le": _bits(value) for key, value in observed.items()}})
                if (not _same_bits(observed["before"], captures[target][site]) or
                        abs(actual - radius) / radius > 0.001 or
                        (site == "cue5" and not _same_bits(observed["mask_logits"], clean[target]))):
                    raise RuntimeError("transfer witness, dose or final-cue control failed")
                patched = _margin(observed["mask_logits"], any_id, some_id)
                site_effects[donor, target] = math.fsum((patched, -margins[target]))
            all_effects[site] = site_effects
    for cue in CUES:
        observed = _observed(model, rows[cue], indices, emit, {"case_id": case_id, "cue": cue},
                             expected_vocab=expected_vocab, expected_width=expected_width)
        forwards += 1
        emit({"kind": "repeat", "case_id": case_id, "cue": cue, "input_ids": rows[cue][0].tolist(),
              "mask_logits_f32le": _bits(observed["mask_logits"]),
              "captures_f32le": {name: _bits(observed["captures"][name]) for name in SITES}})
        if (not _same_bits(observed["mask_logits"], clean[cue]) or
                any(not _same_bits(observed["captures"][name], captures[cue][name]) for name in SITES)):
            raise RuntimeError("post-intervention clean repeat differs")
    if forwards != (72 if eligible_dose else 8):
        raise RuntimeError("case schedule incomplete")
    specificity = {site: context_specificity(all_effects[site]) for site in SITES} if eligible_dose else {}
    return {"id": case_id, "split": case["split"], "frame": case["frame"], "forwards": forwards,
            "baseline_eligible": baseline["eligible"], "baseline_gaps": baseline["gaps"],
            "dose_eligible": eligible_dose, "radius": radius, "distances": distances,
            "clean_margins": margins, "baselines": baselines,
            "effects": {site: [{"donor": donor, "target": target, "delta": values[donor, target]}
                               for donor, target in TRANSFERS] for site, values in all_effects.items()},
            "specificity": specificity,
            "specificity_passed": specificity["cue2"]["passed"] if eligible_dose else None}


def run_study(model, manifest, emit, *, expected_vocab=None, expected_width=None):
    cases = manifest["cases"]
    if (len(cases) != 12 or [row["split"] for row in cases] != ["development"] * 3 + ["evaluation"] * 9
            or len({row["id"] for row in cases}) != 12):
        raise ValueError("fixed case list differs")
    results = []
    for case in cases:
        try:
            results.append(run_case(model, case, emit, any_id=manifest["scored_tokens"]["any"],
                                    some_id=manifest["scored_tokens"]["some"], expected_vocab=expected_vocab,
                                    expected_width=expected_width))
        except BaseException as exc:
            try:
                emit({"kind": "case_failed", "case_id": case["id"], "error_type": type(exc).__name__})
            except Exception:
                pass
            raise
    decision = evaluate([row for row in results if row["split"] == "evaluation"])
    return {"status": "complete", "forwards": sum(row["forwards"] for row in results),
            "cases": results, "evaluation": decision}
