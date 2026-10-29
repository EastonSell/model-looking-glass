"""Bounded, descriptive donor-state interventions in an eager CPU ViT classifier."""

from dataclasses import dataclass, field
import hashlib
import math
import sys

import torch

from .agreement_numeric import decode_f32_le
from .intervention_plan import ActivationPlan
from .vit_trace import ViTTracePoint, _pair, trace_vit_classifier


@dataclass(frozen=True)
class ViTInterventionResult:
    """JSON-compatible report and bounded detached CPU tensor witnesses."""

    report: dict
    witnesses: dict = field(repr=False)


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _same_bits(a, b):
    return (a.dtype == b.dtype == torch.float32 and a.shape == b.shape
            and torch.equal(a.view(torch.int32).contiguous(), b.view(torch.int32).contiguous()))


def _decoded(value):
    if value.dtype != torch.float32 or value.device.type != "cpu" or sys.byteorder != "little":
        raise ValueError("little-endian CPU float32 values are required")
    return decode_f32_le(value.detach().view(torch.int32).contiguous().numpy().tobytes())


def _l2(a, b):
    if a.shape != b.shape:
        raise ValueError("displacement vectors have different shapes")
    return math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(_decoded(a), _decoded(b))))


def _digest(value):
    bits = value.detach().view(torch.int32).contiguous()
    return hashlib.sha256(memoryview(bits.numpy()).cast("B")).hexdigest()


def _site(point, record):
    return {"layer": point.layer, "module": point.module, "batch": point.batch,
            "token": point.token, "invocation": point.invocation, "token_kind": record.token_kind,
            "patch_row": record.patch_row, "patch_column": record.patch_column,
            "processed_pixel_box_xyxy": (list(record.processed_pixel_box_xyxy)
                                         if record.processed_pixel_box_xyxy is not None else None)}


def _dose_status(dose, target, tolerance):
    if target == 0:
        return {"l2": dose, "active": dose > 0, "ratio_to_target": None, "status": "zero_target"}
    if dose == 0:
        return {"l2": dose, "active": False, "ratio_to_target": 0.0, "status": "zero"}
    ratio = dose / target
    return {"l2": dose, "active": True, "ratio_to_target": ratio,
            "status": "matched" if abs(ratio - 1) <= tolerance else "unmatched"}


def compare_vit_donor_patch(model, *, receiver_pixels, donor_pixels, target, control, readout_class,
                            run_id="vit-donor-patch", relative_dose_tolerance=0.05,
                            max_input_bytes=16 * 1024 * 1024, max_tensor_bytes=16 * 1024 * 1024,
                            max_logits_bytes=4 * 1024 * 1024, max_retained_bytes=16 * 1024 * 1024):
    """Capture and compare one same-site donor patch with explicit controls.

    Inputs are equally shaped, already processed single-image CPU float32
    tensors. Target and control are distinct CLS or patch tokens at the same
    boundary. The reflected same-site control is active only when its rounded
    float32 displacement is finite and nonzero; its realized dose is reported.
    Callers must own the model, its execution thread, and pairing semantics.
    """
    if sys.byteorder != "little":
        raise ValueError("only little-endian CPU execution is supported")
    for name, value in (("max_input_bytes", max_input_bytes), ("max_tensor_bytes", max_tensor_bytes),
                        ("max_logits_bytes", max_logits_bytes), ("max_retained_bytes", max_retained_bytes)):
        _positive(value, name)
    if (type(target) is not ViTTracePoint or type(control) is not ViTTracePoint
            or (target.layer, target.module, target.batch, target.invocation) !=
               (control.layer, control.module, control.batch, control.invocation)
            or target.batch != 0 or target.token == control.token):
        raise ValueError("target and control must be distinct tokens at one batch-zero boundary")
    if type(run_id) is not str or not run_id.strip() or len(run_id) > 100:
        raise ValueError("run_id must be a nonempty string of at most 100 characters")
    if (type(relative_dose_tolerance) is not float or not math.isfinite(relative_dose_tolerance)
            or not 0 <= relative_dose_tolerance <= 1):
        raise ValueError("relative_dose_tolerance must be a finite float between zero and one")
    inputs = (receiver_pixels, donor_pixels)
    if any(type(t) is not torch.Tensor or t.device.type != "cpu" or t.dtype != torch.float32
           or t.ndim != 4 or t.shape[0] != 1 or not t.numel() or t.requires_grad for t in inputs):
        raise ValueError("inputs must be nonempty single-image CPU float32 tensors")
    if receiver_pixels.shape != donor_pixels.shape:
        raise ValueError("receiver and donor pixel shapes must match")
    if sum(t.numel() * t.element_size() for t in inputs) > max_input_bytes:
        raise ValueError("aggregate input pixel byte budget exceeded")
    from transformers import ViTForImageClassification
    if type(model) is not ViTForImageClassification:
        raise TypeError("model must be a standard ViTForImageClassification")
    image_size, patch_size = _pair(model.config.image_size, "image_size"), _pair(model.config.patch_size, "patch_size")
    hidden, intermediate, labels = model.config.hidden_size, model.config.intermediate_size, model.config.num_labels
    if (any(type(x) is not int or x <= 0 for x in (hidden, intermediate, labels))
            or type(readout_class) is not int or not 0 <= readout_class < labels
            or tuple(receiver_pixels.shape[2:]) != image_size or receiver_pixels.shape[1] != model.config.num_channels
            or any(size % patch for size, patch in zip(image_size, patch_size))):
        raise ValueError("model dimensions, readout class or input geometry are unsupported")
    sequence = 1 + (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    if target.layer >= len(model.vit.encoder.layer) or max(target.token, control.token) >= sequence:
        raise ValueError("selected layer or token is out of range")
    width = intermediate if target.module == "intermediate" else hidden
    vector_bytes, full_bytes, logits_bytes = width * 4, sequence * width * 4, labels * 4
    retained_bound = 10 * full_bytes + 6 * vector_bytes + 7 * logits_bytes
    if (vector_bytes > 1024 * 1024 or full_bytes > max_tensor_bytes or logits_bytes > max_logits_bytes
            or retained_bound > max_retained_bytes):
        raise ValueError("tensor, logits, numeric-vector or retained-evidence budget exceeded")
    receiver_pixels, donor_pixels = (t.detach().view(torch.int32).contiguous().clone().view(torch.float32)
                                      for t in inputs)
    if any(not bool(torch.isfinite(t).all()) for t in (receiver_pixels, donor_pixels)):
        raise ValueError("pixel inputs must be finite")
    digests = tuple(_digest(t) for t in (receiver_pixels, donor_pixels))
    pixel_shape = tuple(receiver_pixels.shape)
    dimensions = (hidden, intermediate, labels, image_size, patch_size, len(model.vit.encoder.layer))

    def check_inputs():
        current = (receiver_pixels, donor_pixels)
        now = (model.config.hidden_size, model.config.intermediate_size, model.config.num_labels,
               _pair(model.config.image_size, "image_size"), _pair(model.config.patch_size, "patch_size"),
               len(model.vit.encoder.layer))
        if (any(t.dtype != torch.float32 or t.device.type != "cpu" or tuple(t.shape) != pixel_shape for t in current)
                or tuple(_digest(t) for t in current) != digests or now != dimensions):
            raise RuntimeError("pixel inputs or model dimensions changed during the workflow")

    def check_logits(output):
        check_inputs()
        logits = output.logits
        if (type(logits) is not torch.Tensor or logits.dtype != torch.float32 or logits.device.type != "cpu"
                or tuple(logits.shape) != (1, labels) or not bool(torch.isfinite(logits).all())):
            raise RuntimeError("expected finite CPU float32 classifier logits")
        return logits.detach()

    def trace(pixels, name):
        check_inputs()
        result = trace_vit_classifier(model, pixels, points=[target, control], run_id=f"{run_id}/{name}",
            max_events=2, max_trace_bytes=2 * vector_bytes, max_tensor_bytes=max_tensor_bytes,
            max_input_bytes=max_input_bytes, max_logits_bytes=max_logits_bytes)
        logits = check_logits(result.output)
        records = {r.event.token_positions[0]: r for r in result.records}
        if set(records) != {target.token, control.token}:
            raise RuntimeError("trace omitted a selected token")
        for point in (target, control):
            r = records[point.token]
            expected_path = f"vit.encoder.layer.{point.layer}"
            if point.module != "block":
                expected_path += f".{point.module}"
            if (r.event.invocation != 0 or r.event.batch_positions != (0,) or r.event.layer_index != point.layer
                    or r.event.module_path != expected_path or r.event.module_kind != f"vit_{point.module}"
                    or r.event.output_path != ()
                    or tuple(r.event.shape) != (1, sequence, width) or tuple(r.value.shape) != (width,)
                    or r.value.dtype != torch.float32 or r.value.device.type != "cpu"):
                raise RuntimeError("trace metadata or selected shape differs")
            _decoded(r.value)
        return logits, records, result.image_metadata

    receiver_logits, receiver, geometry = trace(receiver_pixels, "receiver")
    donor_logits, donor, donor_geometry = trace(donor_pixels, "donor")
    if donor_geometry != geometry:
        raise RuntimeError("receiver and donor geometry differ")
    for point in (target, control):
        a, b = receiver[point.token].event, donor[point.token].event
        if (a.module_path != b.module_path or a.output_path != b.output_path or a.shape != b.shape
                or a.module_kind != b.module_kind or a.token_positions != b.token_positions):
            raise RuntimeError("receiver and donor capture policies differ")
    path, selector = receiver[target.token].event.module_path, list(receiver[target.token].event.output_path)
    if (receiver[control.token].event.module_path != path
            or list(receiver[control.token].event.output_path) != selector):
        raise RuntimeError("target and control do not share one boundary")
    limits = {"max_tensor_bytes": max_tensor_bytes, "max_capture_bytes": 2 * full_bytes + vector_bytes,
              "max_source_bytes": vector_bytes, "max_steps": 4, "max_invocations": 1}

    def arm(kind, point, replacement=None):
        base = {"module_path": path, "invocation": 0}
        selected = dict(base, batch_index=0, token_index=point.token)
        steps = [dict(base, name="before", operation="capture")]
        if kind == "self":
            steps += [dict(selected, name="self_source", operation="capture"),
                      dict(selected, name="patch", operation="patch", source_capture="self_source")]
        elif kind == "null":
            steps.append(dict(selected, name="patch", operation="null"))
        else:
            steps.append(dict(selected, name="patch", operation="patch", value=replacement))
        steps.append(dict(base, name="after", operation="capture"))
        check_inputs()
        plan = ActivationPlan(model, steps, **limits)
        with torch.inference_mode(), plan:
            output = model(pixel_values=receiver_pixels, interpolate_pos_encoding=False, output_attentions=False,
                           output_hidden_states=False, return_dict=True)
        logits = check_logits(output)
        captures, metadata = plan.captures, plan.metadata
        if (len(metadata["steps"]) != len(steps)
                or set(captures) != {s["name"] for s in steps if s["operation"] == "capture"}
                or metadata["source_bytes"] != (0 if kind in ("self", "null") else vector_bytes)
                or metadata["capture_bytes"] != sum(x.numel() * x.element_size() for x in captures.values())):
            raise RuntimeError("intervention did not retain all requested actions")
        for index, (wanted, actual) in enumerate(zip(steps, metadata["steps"], strict=True)):
            selected_step = "token_index" in wanted
            if (actual["name"] != wanted["name"] or actual["operation"] != wanted["operation"]
                    or actual["order"] != index or actual["module_path"] != path
                    or actual["module_aliases"] != [path] or actual["invocation"] != 0
                    or actual["forward_index"] != 0 or actual["output_selector"] != selector
                    or actual["batch_index"] != wanted.get("batch_index")
                    or actual["token_index"] != wanted.get("token_index")
                    or actual["source_capture"] != wanted.get("source_capture")
                    or actual["output_shape"] != [1, sequence, width]
                    or actual["selected_shape"] != ([width] if selected_step else [1, sequence, width])
                    or actual["dtype"] != "torch.float32" or actual["device"] != "cpu"):
                raise RuntimeError("intervention action metadata differs from its requested site")
        before, after = captures["before"], captures["after"]
        source = captures["self_source"] if kind == "self" else (
            before[0, point.token] if kind == "null" else replacement)
        expected = before.view(torch.int32).clone()
        expected[0, point.token] = source.view(torch.int32)
        if (not _same_bits(before[0, point.token], receiver[point.token].value)
                or (kind == "self" and not _same_bits(source, before[0, point.token]))
                or not torch.equal(after.view(torch.int32), expected)):
            raise RuntimeError("patch failed landing, receiver matching or untouched-position checks")
        return logits, captures, metadata

    target_dose = _l2(donor[target.token].value, receiver[target.token].value)
    off_dose = _l2(donor[control.token].value, receiver[control.token].value)
    reflected, reflected_reason = None, None
    if target_dose == 0:
        reflected_reason = "zero_target"
    else:
        values = [2 * float(r) - float(d) for r, d in
                  zip(_decoded(receiver[target.token].value), _decoded(donor[target.token].value))]
        if any(not math.isfinite(value) or abs(value) > torch.finfo(torch.float32).max for value in values):
            reflected_reason = "outside_float32_range"
        else:
            reflected = torch.tensor(values, dtype=torch.float32)
            if not bool(torch.isfinite(reflected).all()):
                reflected, reflected_reason = None, "nonfinite_after_rounding"
            elif _l2(reflected, receiver[target.token].value) == 0:
                reflected, reflected_reason = None, "zero_after_rounding"

    arms = {"self": (target, None), "null": (target, None),
            "target": (target, donor[target.token].value), "off_target": (control, donor[control.token].value)}
    if reflected is not None:
        arms["reflected"] = (target, reflected)
    logits = {"receiver": receiver_logits, "donor": donor_logits}
    captures, metadata = {}, {}
    for kind, (point, value) in arms.items():
        logits[kind], captures[kind], metadata[kind] = arm(kind, point, value)
    repeat_logits, repeat, repeat_geometry = trace(receiver_pixels, "repeat")
    controls = {"self_matches_receiver_logits": _same_bits(logits["self"], receiver_logits),
                "null_matches_receiver_logits": _same_bits(logits["null"], receiver_logits),
                "repeat_matches_receiver_logits": _same_bits(repeat_logits, receiver_logits),
                "repeat_matches_receiver_sites": (repeat_geometry == geometry and
                    all(_same_bits(repeat[p.token].value, receiver[p.token].value) for p in (target, control))),
                "all_patch_pre_states_match": all(_same_bits(state["before"], captures["self"]["before"])
                                                  for state in captures.values())}
    if not all(controls.values()):
        raise RuntimeError("self, null, repeat or pre-patch consistency control failed")
    reflected_dose = ({"l2": None, "active": False, "ratio_to_target": None, "status": reflected_reason}
                      if reflected is None else _dose_status(_l2(reflected, receiver[target.token].value),
                                                            target_dose, relative_dose_tolerance))
    witnesses = {"receiver": {str(k): v.value for k, v in receiver.items()},
                 "donor": {str(k): v.value for k, v in donor.items()}, "plans": captures,
                 "readout_logits": logits, "reflected_replacement": reflected}
    retained = (sum(t.numel() * t.element_size() for group in (witnesses["receiver"], witnesses["donor"], logits)
                    for t in group.values())
                + sum(t.numel() * t.element_size() for group in captures.values() for t in group.values())
                + (0 if reflected is None else reflected.numel() * reflected.element_size()))
    if retained > retained_bound:
        raise RuntimeError("retained witnesses exceed the preflight bound")
    baseline = float(_decoded(receiver_logits[0, readout_class:readout_class + 1])[0])
    measurements = {}
    for name, values in logits.items():
        value = float(_decoded(values[0, readout_class:readout_class + 1])[0])
        measurements[name] = {"logit": value, "delta_from_receiver": value - baseline}
    report = {"schema": "vit-donor-patch-v1", "run_id": run_id, "forwards": 3 + len(arms),
              "inputs": {"receiver_sha256": digests[0], "donor_sha256": digests[1], "pixel_shape": list(pixel_shape)},
              "geometry": geometry, "sites": {"target": _site(target, receiver[target.token]),
                                               "off_target": _site(control, receiver[control.token])},
              "trace_events": {"receiver": {str(k): v.event.to_dict() for k, v in receiver.items()},
                               "donor": {str(k): v.event.to_dict() for k, v in donor.items()}},
              "plan_metadata": metadata, "controls": controls,
              "doses": {"target": {"l2": target_dose, "active": target_dose > 0},
                        "off_target": _dose_status(off_dose, target_dose, relative_dose_tolerance),
                        "reflected": reflected_dose, "relative_tolerance": relative_dose_tolerance},
              "readout": {"class_id": readout_class, "measurements": measurements},
              "retained_tensor_bytes": retained, "retained_tensor_byte_bound": retained_bound,
              "limits": {"max_input_bytes": max_input_bytes, "max_tensor_bytes": max_tensor_bytes,
                         "max_logits_bytes": max_logits_bytes, "max_retained_bytes": max_retained_bytes}}
    return ViTInterventionResult(report, witnesses)
