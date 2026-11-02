"""CPU-only, matched-dose cue-transfer primitives for the fixed study."""

import base64
import math
import sys

import torch

from .intervention_plan import ActivationPlan


SITES = (("cue2", 2, 4), ("object2", 2, 9), ("mask2", 2, 6), ("cue5", 5, 4))
PAIRS = (("not", "just"), ("never", "just"), ("also", "just"),
         ("also", "not"), ("also", "never"))
TRANSFERS = (("not", "just"), ("just", "not"), ("never", "just"),
             ("just", "never"), ("also", "just"), ("just", "also"),
             ("also", "not"), ("also", "never"))
CUES = ("not", "never", "just", "also")


class RadiusFailure(ValueError):
    def __init__(self, radius, distances):
        super().__init__("common radius below the fixed floor")
        self.radius = radius
        self.distances = distances


class ObservationFailure(RuntimeError):
    """A completed forward failed a local control; retain its bounded witness."""
    def __init__(self, phase, payload):
        super().__init__(f"forward {phase} control failed")
        self.phase = phase
        self.payload = payload


def _vector(value):
    if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.float32
            or value.ndim != 1 or not value.numel() or not bool(torch.isfinite(value).all())):
        raise ValueError("expected a finite one-dimensional CPU float32 vector")
    return value


def distance(left, right):
    left, right = _vector(left), _vector(right)
    if left.shape != right.shape:
        raise ValueError("vector shapes differ")
    squares = ((float(a) - float(b)) ** 2 for a, b in zip(left.tolist(), right.tolist()))
    result = math.sqrt(math.fsum(squares))
    if not math.isfinite(result):
        raise ValueError("nonfinite distance")
    return result


def choose_radius(states):
    if type(states) is not dict or set(states) != {"not", "never", "just", "also"}:
        raise ValueError("expected all four cue states")
    distances = []
    for left, right in PAIRS:
        for site, _, _ in SITES:
            value = distance(states[left][site], states[right][site])
            distances.append({"left": left, "right": right, "site": site, "distance": value})
    radius = min(item["distance"] for item in distances)
    if radius < 0.001:
        raise RadiusFailure(radius, distances)
    return radius, distances


def make_patch(target, donor, radius):
    target, donor = _vector(target), _vector(donor)
    if type(radius) not in (int, float) or isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    separation = distance(target, donor)
    if separation <= 0 or radius > separation:
        raise ValueError("requested radius would amplify or has no direction")
    fraction = radius / separation
    # Two float64 operations are intentional: no float32 arithmetic until the
    # single final rounding to the target representation.
    delta = donor.to(torch.float64) - target.to(torch.float64)
    scaled = delta * fraction
    patched = (target.to(torch.float64) + scaled).to(torch.float32)
    actual = distance(target, patched)
    if abs(actual - radius) / radius > 0.001:
        raise ValueError("delivered dose exceeds the fixed relative tolerance")
    return patched, actual, fraction


def transfer_score(actions):
    values = []
    for cue in ("not", "never"):
        forward = actions[(cue, "just")]
        reverse = actions[("just", cue)]
        if (type(forward) not in (int, float) or type(reverse) not in (int, float)
                or not math.isfinite(forward) or not math.isfinite(reverse)):
            raise ValueError("nonfinite directional transfer")
        values.append(math.fsum((forward, -reverse)) / 2.0)
    return math.fsum(values) / 2.0


def evaluate_prediction(case_scores):
    if (type(case_scores) not in (list, tuple) or len(case_scores) != 6
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in case_scores)):
        raise ValueError("all six finite evaluation scores are required")
    mean = math.fsum(case_scores) / 6.0
    positive = sum(value > 0 for value in case_scores)
    return {"mean": mean, "positive_cases": positive, "passed": mean > 0.02 and positive >= 4}


def _has_hooks(model):
    for module in model.modules():
        if any(getattr(module, name, {}) for name in
               ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")):
            return True
    registry = torch.nn.modules.module
    return any(getattr(registry, name, {}) for name in
               ("_global_forward_pre_hooks", "_global_forward_hooks", "_global_backward_pre_hooks",
                "_global_backward_hooks"))


def _landing_equal(left, right):
    return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def observe(model, input_ids, *, site=None, source=None, self_patch=False):
    """Run one clean capture or one patch with complete selected-block witnesses."""
    if (not isinstance(model, torch.nn.Module) or any(module.training for module in model.modules())
            or _has_hooks(model)):
        raise ValueError("expected an evaluation model without existing hooks")
    if (type(input_ids) is not torch.Tensor or input_ids.device.type != "cpu"
            or input_ids.dtype != torch.int64 or tuple(input_ids.shape) != (1, 12)):
        raise ValueError("expected one CPU int64 row of 12 tokens")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("all parameters must be CPU float32")
    if any(b.device.type != "cpu" for b in model.buffers()):
        raise ValueError("all buffers must remain on CPU")
    if torch.is_autocast_enabled("cpu"):
        raise ValueError("CPU autocast is not supported")
    if type(self_patch) is not bool or (self_patch and source is not None):
        raise ValueError("self and fixed-source patches are mutually exclusive")
    if site is None:
        if source is not None or self_patch:
            raise ValueError("clean observation cannot patch")
        steps = [{"name": name, "operation": "capture", "module_path": f"distilbert.transformer.layer.{layer}",
                  "invocation": 0, "batch_index": 0, "token_index": token} for name, layer, token in SITES]
    else:
        choices = {name: (layer, token) for name, layer, token in SITES}
        if site not in choices or (source is None and not self_patch):
            raise ValueError("invalid patch site or source")
        if source is not None:
            _vector(source)
        layer, token = choices[site]
        path = f"distilbert.transformer.layer.{layer}"
        whole = {"module_path": path, "invocation": 0}
        one = {**whole, "batch_index": 0, "token_index": token}
        patch = {"name": "patch", "operation": "patch", **one}
        patch.update({"source_capture": "before"} if self_patch else {"value": source})
        steps = [{"name": "full_before", "operation": "capture", **whole},
                 {"name": "before", "operation": "capture", **one}, patch,
                 {"name": "after", "operation": "capture", **one},
                 {"name": "full_after", "operation": "capture", **whole}]
    original = input_ids.clone()
    attention = torch.ones_like(input_ids)
    plan = ActivationPlan(model, steps, max_tensor_bytes=1024 * 1024,
                          max_capture_bytes=2 * 1024 * 1024, max_source_bytes=1024 * 1024,
                          max_steps=5, max_invocations=1)
    with torch.inference_mode(), plan:
        output = model(input_ids=input_ids, attention_mask=attention, output_attentions=False,
                       output_hidden_states=False, return_dict=True).logits
        if (type(output) is not torch.Tensor or output.device.type != "cpu" or output.dtype != torch.float32
                or output.ndim != 3 or tuple(output.shape[:2]) != (1, 12)
                or not bool(torch.isfinite(output).all())):
            raise RuntimeError("invalid model logits")
        mask_logits = output[0, 6].detach().clone()
    captures = plan.captures
    payload = {"mask_logits": mask_logits, "captures": captures} if site is None else {
        "mask_logits": mask_logits, "before": captures["before"], "after": captures["after"],
        "full_before": captures["full_before"], "full_after": captures["full_after"]}
    if _has_hooks(model) or not torch.equal(input_ids, original) or not torch.equal(attention, torch.ones_like(attention)):
        raise ObservationFailure("hook_or_input", payload)
    if site is None:
        return payload
    before, after = captures["before"], captures["after"]
    expected = before if self_patch else source
    before_full, after_full = captures["full_before"], captures["full_after"]
    if not _landing_equal(after, expected):
        raise ObservationFailure("landing", payload)
    unchanged = all(torch.equal(before_full[:, i].contiguous().view(torch.int32),
                                after_full[:, i].contiguous().view(torch.int32))
                    for i in range(12) if i != token)
    if not unchanged:
        raise ObservationFailure("other_tokens", payload)
    payload["other_tokens_equal"] = unchanged
    return payload


def _bits(value):
    if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.float32
            or sys.byteorder != "little"):
        raise ValueError("expected little-endian CPU float32 tensor")
    return base64.b64encode(value.detach().contiguous().view(torch.int32).numpy().tobytes()).decode("ascii")


def _same_bits(left, right):
    return left.shape == right.shape and torch.equal(left.contiguous().view(torch.int32),
                                                      right.contiguous().view(torch.int32))


def _margin(logits, any_id, some_id):
    if (type(any_id) is not int or type(some_id) is not int or any_id == some_id
            or not 0 <= any_id < logits.numel() or not 0 <= some_id < logits.numel()):
        raise ValueError("invalid scored token IDs")
    return math.fsum((float(logits[any_id]), -float(logits[some_id])))


def _observed_or_record_failure(model, ids, emit, metadata, **kwargs):
    try:
        return observe(model, ids, **kwargs)
    except ObservationFailure as exc:
        witness = {"kind": "forward_failed", "phase": exc.phase, **metadata,
                   "input_ids": ids[0].tolist()}
        for name, value in exc.payload.items():
            if name == "captures":
                witness["captures_f32le"] = {key: _bits(tensor) for key, tensor in value.items()}
            else:
                witness[name + "_f32le"] = _bits(value)
        emit(witness)
        raise


def baseline_summary(logits, *, any_id, some_id):
    if (type(logits) is not torch.Tensor or logits.device.type != "cpu" or logits.dtype != torch.float32
            or logits.ndim != 1 or logits.numel() < 5 or not bool(torch.isfinite(logits).all())):
        raise ValueError("expected finite CPU float32 vocabulary logits")
    margin = _margin(logits, any_id, some_id)
    values = logits.tolist()
    peak = max(values)
    denominator = math.fsum(math.exp(value - peak) for value in values)
    top = sorted(range(len(values)), key=lambda index: (-values[index], index))[:5]
    return {"any_minus_some": margin, "any_probability": math.exp(values[any_id] - peak) / denominator,
            "some_probability": math.exp(values[some_id] - peak) / denominator,
            "top_five": [{"token_id": index, "logit": values[index]} for index in top]}


def run_case(model, case_id, rows, emit, *, any_id, some_id):
    """Execute the fixed 56-forward schedule, emitting each completed witness."""
    if type(case_id) is not str or not case_id or type(rows) is not dict or set(rows) != set(CUES) or not callable(emit):
        raise ValueError("invalid case inputs")
    clean, captures, margins, baselines = {}, {}, {}, {}
    forwards = 0
    for cue in CUES:
        observed = _observed_or_record_failure(model, rows[cue], emit, {"case_id": case_id, "cue": cue})
        forwards += 1
        clean[cue], captures[cue] = observed["mask_logits"], observed["captures"]
        emit({"kind": "clean", "case_id": case_id, "cue": cue, "input_ids": rows[cue][0].tolist(),
              "mask_logits_f32le": _bits(clean[cue]),
              "captures_f32le": {name: _bits(captures[cue][name]) for name, _, _ in SITES}})
        margins[cue] = _margin(clean[cue], any_id, some_id)
        baselines[cue] = baseline_summary(clean[cue], any_id=any_id, some_id=some_id)
    try:
        radius, distances = choose_radius(captures)
    except RadiusFailure as exc:
        emit({"kind": "radius_failed", "case_id": case_id, "radius": exc.radius, "distances": exc.distances})
        raise
    emit({"kind": "radius", "case_id": case_id, "radius": radius, "distances": distances,
          "binding_edges": [item for item in distances if item["distance"] == radius]})
    effects = {}
    for site, _, _ in SITES:
        for cue in CUES:
            observed = _observed_or_record_failure(model, rows[cue], emit,
                                                   {"case_id": case_id, "site": site, "target": cue},
                                                   site=site, self_patch=True)
            forwards += 1
            record = {"kind": "self", "case_id": case_id, "site": site, "target": cue,
                      "input_ids": rows[cue][0].tolist(), "mask_logits_f32le": _bits(observed["mask_logits"]),
                      "before_f32le": _bits(observed["before"]), "after_f32le": _bits(observed["after"]),
                      "full_before_f32le": _bits(observed["full_before"]),
                      "full_after_f32le": _bits(observed["full_after"])}
            emit(record)
            if (not _same_bits(observed["before"], captures[cue][site])
                    or not _same_bits(observed["mask_logits"], clean[cue])):
                raise RuntimeError("self patch or clean-state control failed")
        effects[site] = {}
        for donor, target in TRANSFERS:
            submitted, planned_norm, fraction = make_patch(captures[target][site], captures[donor][site], radius)
            observed = _observed_or_record_failure(model, rows[target], emit,
                                                   {"case_id": case_id, "site": site,
                                                    "donor": donor, "target": target},
                                                   site=site, source=submitted)
            forwards += 1
            try:
                actual = distance(observed["before"], observed["after"])
            except BaseException:
                emit({"kind": "forward_failed", "phase": "dose_measurement", "case_id": case_id,
                      "site": site, "donor": donor, "target": target,
                      "input_ids": rows[target][0].tolist(), "requested_radius": radius,
                      "requested_fraction": fraction, "planned_norm": planned_norm,
                      "mask_logits_f32le": _bits(observed["mask_logits"]),
                      "submitted_f32le": _bits(submitted), "before_f32le": _bits(observed["before"]),
                      "after_f32le": _bits(observed["after"]),
                      "full_before_f32le": _bits(observed["full_before"]),
                      "full_after_f32le": _bits(observed["full_after"])})
                raise
            record = {"kind": "transfer", "case_id": case_id, "site": site, "donor": donor,
                      "target": target, "input_ids": rows[target][0].tolist(), "requested_radius": radius,
                      "requested_fraction": fraction, "planned_norm": planned_norm, "actual_norm": actual,
                      "relative_error": abs(actual - radius) / radius,
                      "mask_logits_f32le": _bits(observed["mask_logits"]), "submitted_f32le": _bits(submitted),
                      "before_f32le": _bits(observed["before"]), "after_f32le": _bits(observed["after"]),
                      "full_before_f32le": _bits(observed["full_before"]),
                      "full_after_f32le": _bits(observed["full_after"])}
            emit(record)
            if (not _same_bits(observed["before"], captures[target][site])
                    or abs(actual - radius) / radius > 0.001
                    or (site == "cue5" and not _same_bits(observed["mask_logits"], clean[target]))):
                raise RuntimeError("transfer witness, dose or final-block control failed")
            effects[site][(donor, target)] = math.fsum((_margin(observed["mask_logits"], any_id, some_id),
                                                         -margins[target]))
    for cue in CUES:
        observed = _observed_or_record_failure(model, rows[cue], emit, {"case_id": case_id, "cue": cue})
        forwards += 1
        emit({"kind": "repeat", "case_id": case_id, "cue": cue, "input_ids": rows[cue][0].tolist(),
              "mask_logits_f32le": _bits(observed["mask_logits"]),
              "captures_f32le": {name: _bits(observed["captures"][name]) for name, _, _ in SITES}})
        if (not _same_bits(observed["mask_logits"], clean[cue])
                or any(not _same_bits(observed["captures"][name], captures[cue][name]) for name, _, _ in SITES)):
            raise RuntimeError("post-intervention clean repeat differs")
    if forwards != 56:
        raise RuntimeError("case forward schedule is incomplete")
    return {"case_id": case_id, "forwards": forwards, "radius": radius, "distances": distances,
            "clean_margins": margins, "baselines": baselines,
            "effects": {site: [{"donor": a, "target": b, "delta": effects[site][(a, b)]}
                               for a, b in TRANSFERS] for site, _, _ in SITES},
            "directional_score": transfer_score(effects["cue2"])}


def run_study(model, manifest, emit):
    if (type(manifest) is not dict or type(manifest.get("cases")) is not list
            or len(manifest["cases"]) != 8 or not callable(emit)):
        raise ValueError("the fixed study requires eight cases")
    cases = manifest["cases"]
    if ([case.get("split") for case in cases] != ["development"] * 2 + ["evaluation"] * 6
            or any(type(case.get("id")) is not str or not case["id"] for case in cases)
            or len({case["id"] for case in cases}) != 8
            or type(manifest.get("cues")) is not dict or set(manifest["cues"]) != set(CUES)
            or any(type(value) is not int or value < 0 for value in manifest["cues"].values())
            or len(set(manifest["cues"].values())) != 4):
        raise ValueError("fixed split or cue identities differ")
    scores = manifest.get("scored_tokens")
    if type(scores) is not dict or set(scores) != {"any", "some"}:
        raise ValueError("scored token IDs are missing")
    results = []
    for case in cases:
        ids = case.get("just_ids")
        if (type(ids) is not list or len(ids) != 12 or any(type(value) is not int or value < 0 for value in ids)
                or ids[4] != manifest["cues"]["just"]):
            raise ValueError("case input does not match the fixed cue position")
        rows = {}
        for cue in CUES:
            row = ids.copy()
            row[4] = manifest["cues"][cue]
            rows[cue] = torch.tensor([row], dtype=torch.int64)
        try:
            result = run_case(model, case["id"], rows, emit, any_id=scores["any"], some_id=scores["some"])
        except BaseException as exc:
            try:
                emit({"kind": "case_failed", "case_id": case["id"], "error_type": type(exc).__name__})
            except Exception:
                pass
            raise
        result["split"] = case["split"]
        results.append(result)
    evaluation = [result for result in results if result["split"] == "evaluation"]
    decision = evaluate_prediction([result["directional_score"] for result in evaluation])
    return {"status": "complete", "forwards": sum(result["forwards"] for result in results),
            "cases": results, "evaluation_cases": [result["case_id"] for result in evaluation],
            "directional_prediction": decision}
