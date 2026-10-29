"""Fixed, bounded synthetic ViT embedding-injection mechanics study.

This composes the public ViT trace and ActivationPlan primitives. The selector
is label-blind at evaluation; discovery uses the known injection labels.
"""

from dataclasses import replace
import hashlib
import json
import math
import random
import struct
import sys

import torch

from .agreement_numeric import decode_f32_le
from .intervention_plan import ActivationPlan
from .trace_schema import RunIdentity
from .vit_trace import ViTTracePoint, ViTTraceResult, trace_vit_classifier
from .vit_trace_comparison import compare_vit_traces


RANDOM_CHOICES = (166, 166, 52, 166)
SEEDS = (11, 12, 13, 14, 21, 22, 23, 24)
TOKENS = (16, 52, 103, 166)
INJECTIONS = (16, 52, 103, 166, 103, 166, 16, 52)
_PATH = "vit.encoder.layer.0"


def synthetic_pixels(seed, image_size):
    if type(seed) is not int or type(image_size) is not int or image_size <= 0:
        raise ValueError("invalid synthetic input parameters")
    c = torch.arange(3, dtype=torch.int64)[:, None, None]
    y = torch.arange(image_size, dtype=torch.int64)[None, :, None]
    x = torch.arange(image_size, dtype=torch.int64)[None, None, :]
    return ((((17*x + 31*y + 47*c + 97*seed) % 257) - 128).to(torch.float32) / 128).unsqueeze(0)


def _raw(value):
    if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.float32
            or sys.byteorder != "little"):
        raise ValueError("expected little-endian CPU float32 tensor")
    return value.detach().view(torch.int32).contiguous().numpy().tobytes()


def _values(value):
    return decode_f32_le(_raw(value))


def _same(a, b):
    return a.shape == b.shape and _raw(a) == _raw(b)


def _distance(a, b):
    if a.shape != b.shape:
        raise ValueError("incompatible displacement shapes")
    return math.sqrt(math.fsum((float(x)-float(y))**2 for x, y in zip(_values(a), _values(b), strict=True)))


def _delta(a, b):
    if a.shape != b.shape:
        raise ValueError("incompatible candidate shapes")
    return tuple(float(x)-float(y) for x, y in zip(_values(a), _values(b), strict=True))


def _norm(a):
    return math.sqrt(math.fsum(x*x for x in a))


def choose(direction, candidates):
    """Choose from ordered (token, delta) pairs; injection labels are not inputs."""
    if not direction or not candidates or any(len(vector) != len(direction) for _, vector in candidates):
        raise ValueError("incomplete fitted direction or candidate vectors")
    if not all(math.isfinite(x) for x in direction) or any(
            not math.isfinite(x) for _, vector in candidates for x in vector):
        raise ValueError("nonfinite selection data")
    if len({token for token, _ in candidates}) != len(candidates):
        raise ValueError("duplicate candidate tokens")
    scores = [(token, math.fsum(x*y for x, y in zip(direction, vector, strict=True)))
              for token, vector in candidates]
    magnitudes = [(token, _norm(vector)) for token, vector in candidates]
    if not all(math.isfinite(value) for _, value in scores + magnitudes):
        raise ValueError("nonfinite selection score")
    return {"directional": min(scores, key=lambda row: (-row[1], row[0]))[0],
            "magnitude": min(magnitudes, key=lambda row: (-row[1], row[0]))[0],
            "scores": [[token, score] for token, score in scores],
            "magnitudes": [[token, value] for token, value in magnitudes]}


def _assert_no_hooks(model):
    if (torch.nn.modules.module._global_forward_pre_hooks or torch.nn.modules.module._global_forward_hooks
            or any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())):
        raise RuntimeError("pre-existing or retained forward hooks")


def _plan_forward(model, pixels, injection_token, *, patch_token=None, replacement=None, self_patch=False):
    """Inject once and optionally patch one block output in the same root call."""
    _assert_no_hooks(model)
    pixel_bits = _raw(pixels)
    width = model.config.hidden_size
    injection = torch.zeros(width, dtype=torch.float32)
    injection[7] = 4.0
    emb = {"module_path": "vit.embeddings", "invocation": 0}
    block = {"module_path": _PATH, "invocation": 0}
    steps = [dict(emb, name="embedding_before", operation="capture"),
             dict(emb, name="injection", operation="add", batch_index=0,
                  token_index=injection_token, value=injection),
             dict(emb, name="embedding_after", operation="capture"),
             dict(block, name="block_before", operation="capture")]
    if patch_token is not None:
        selected = dict(block, batch_index=0, token_index=patch_token)
        if self_patch:
            steps += [dict(selected, name="self_source", operation="capture"),
                      dict(selected, name="patch", operation="patch", source_capture="self_source")]
        else:
            steps += [dict(selected, name="patch", operation="patch", value=replacement)]
    steps += [dict(block, name="block_after", operation="capture")]
    plan = ActivationPlan(model, steps, max_steps=8, max_invocations=1,
                          max_tensor_bytes=4*1024*1024, max_capture_bytes=8*1024*1024,
                          max_source_bytes=1024*1024)
    with torch.inference_mode(), plan:
        output = model(pixel_values=pixels, interpolate_pos_encoding=False, output_attentions=False,
                       output_hidden_states=False, return_dict=True)
    _assert_no_hooks(model)
    if _raw(pixels) != pixel_bits:
        raise RuntimeError("model forward mutated its pixel input")
    captures, metadata = plan.captures, plan.metadata
    if len(metadata["steps"]) != len(steps):
        raise RuntimeError("plan omitted an action")
    sequence = 1 + (pixels.shape[-1] // model.config.patch_size) ** 2
    for index, (wanted, actual) in enumerate(zip(steps, metadata["steps"], strict=True)):
        selected = "token_index" in wanted
        if (actual["name"] != wanted["name"] or actual["order"] != index
                or actual["operation"] != wanted["operation"] or actual["module_path"] != wanted["module_path"]
                or actual["module_aliases"] != [wanted["module_path"]] or actual["invocation"] != 0
                or actual["forward_index"] != 0 or actual["batch_index"] != wanted.get("batch_index")
                or actual["token_index"] != wanted.get("token_index") or actual["output_selector"] != []
                or actual["source_capture"] != wanted.get("source_capture")
                or actual["output_shape"] != [1, sequence, width]
                or actual["selected_shape"] != ([width] if selected else [1, sequence, width])
                or actual["dtype"] != "torch.float32" or actual["device"] != "cpu"):
            raise RuntimeError("plan action metadata does not match the fixed site")
    before, after = captures["embedding_before"], captures["embedding_after"]
    expected_bits = before.view(torch.int32).clone()
    rounded = before[0, injection_token, 7] + injection[7]
    expected_bits[0, injection_token, 7] = rounded.view(torch.int32)
    if (not torch.equal(after.view(torch.int32), expected_bits)
            or before[0, injection_token, 7].view(torch.int32) == after[0, injection_token, 7].view(torch.int32)):
        raise RuntimeError("invalid injection landing or unchanged treatment")
    injection_l2 = _distance(before[0, injection_token], after[0, injection_token])
    if injection_l2 == 0:
        raise RuntimeError("invalid zero-dose injection")
    block_before, block_after = captures["block_before"], captures["block_after"]
    expected = block_before.clone()
    if patch_token is not None:
        source = captures["self_source"] if self_patch else replacement
        expected[0, patch_token] = source
    if not _same(block_after, expected):
        raise RuntimeError("patch landing or untouched-position check failed")
    logits = output.logits.detach()
    if tuple(logits.shape) != (1, model.config.num_labels):
        raise RuntimeError("classifier returned an unexpected logit shape")
    _values(logits.flatten())
    return {"captures": captures, "metadata": metadata, "logits": logits,
            "injection_l2": injection_l2}


def paired_case(model, pixels, candidates, injection_token, run_id):
    points = [ViTTracePoint(0, "block", 0, token) for token in (0, *candidates)]
    clean = trace_vit_classifier(model, pixels, points=points, run_id=run_id + "/clean")
    injected = _plan_forward(model, pixels, injection_token)
    block = injected["captures"]["block_before"]
    records = tuple(replace(record, event=replace(record.event, identity=RunIdentity(run_id + "/injected")),
                            value=block[0, point.token].clone())
                    for point, record in zip(points, clean.records, strict=True))
    candidate = ViTTraceResult(None, records, clean.image_metadata,
                               sum(record.value.numel()*4 for record in records))
    comparison = compare_vit_traces(clean, candidate, points=points, pairing_verified=True)
    if comparison["status"] != "compared":
        raise RuntimeError("paired trace comparison was inconclusive")
    return {"clean": clean, "injected": injected, "comparison": comparison,
            "deltas": [(point.token, _delta(record.value, clean_record.value))
                       for point, record, clean_record in zip(points[1:], records[1:], clean.records[1:], strict=True)]}


def _replacement(receiver, donor, target_dose, reflected=False):
    a, b = _values(receiver), _values(donor)
    natural = _distance(receiver, donor)
    if natural == 0 or target_dose == 0:
        return None, "zero_direction"
    scale = -1.0 if reflected else target_dose / natural
    numbers = [float(x) + scale*(float(y)-float(x)) for x, y in zip(a, b, strict=True)]
    if any(not math.isfinite(x) or abs(x) > torch.finfo(torch.float32).max for x in numbers):
        return None, "nonfinite_or_out_of_range"
    result = torch.tensor(numbers, dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        return None, "nonfinite_after_rounding"
    return result, None


def intervention_arms(model, pixels, pair, injection_token, target_token, control_token, on_arm=None):
    clean, baseline = pair["clean"], pair["injected"]
    clean_vectors = {record.event.token_positions[0]: record.value for record in clean.records}
    corrupt = baseline["captures"]["block_before"]
    target = clean_vectors[target_token]
    dose = _distance(corrupt[0, target_token], target)
    reflected, reflected_error = _replacement(corrupt[0, target_token], target, dose, reflected=True)
    off, off_error = _replacement(corrupt[0, control_token], clean_vectors[control_token], dose)
    results, arms = {}, [("self", target_token, None, True), ("target", target_token, target, False)]
    if reflected is not None:
        arms.append(("reflected", target_token, reflected, False))
    if off is not None:
        arms.append(("off_target", control_token, off, False))
    for name, token, value, self_patch in arms:
        arm = _plan_forward(model, pixels, injection_token, patch_token=token,
                            replacement=value, self_patch=self_patch)
        if (not _same(arm["captures"]["embedding_before"], baseline["captures"]["embedding_before"])
                or not _same(arm["captures"]["embedding_after"], baseline["captures"]["embedding_after"])
                or not _same(arm["captures"]["block_before"], corrupt)):
            raise RuntimeError("injection or pre-patch state differs across arms")
        if name == "self" and not _same(arm["logits"], baseline["logits"]):
            raise RuntimeError("self patch changed logits")
        if name == "target" and not _same(arm["captures"]["block_after"][0, token], target):
            raise RuntimeError("clean donor did not land at selected site")
        actual = _distance(arm["captures"]["block_before"][0, token], arm["captures"]["block_after"][0, token])
        ratio = actual/dose if dose else None
        results[name] = {"forward": arm, "dose": actual, "ratio_to_target": ratio,
                         "dose_matched": bool(ratio is not None and actual > 0 and abs(ratio-1) <= 0.05),
                         "logit_distance_from_clean": _distance(arm["logits"], clean.output.logits)}
        if on_arm is not None:
            on_arm(name, results[name])
    return {"arms": results, "target_dose": dose, "off_target_token": control_token,
            "reflected_unavailable": reflected_error, "off_target_unavailable": off_error,
            "injected_logit_distance_from_clean": _distance(baseline["logits"], clean.output.logits)}


def fit_direction(pairs, injection_tokens):
    rows = [dict(pair["deltas"])[token] for pair, token in zip(pairs, injection_tokens, strict=True)]
    direction = tuple(math.fsum(row[index] for row in rows)/len(rows) for index in range(len(rows[0])))
    return direction


def direction_bytes(direction):
    if not direction or any(not math.isfinite(value) for value in direction):
        raise ValueError("invalid discovery direction")
    return struct.pack("<" + "d"*len(direction), *direction)


def fixed_random_choices():
    values = tuple(random.Random(218741).choices(list(TOKENS), k=4))
    if values != RANDOM_CHOICES:
        raise RuntimeError("runtime random sequence differs from the frozen protocol")
    return values
