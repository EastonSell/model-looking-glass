"""Explicit paired context for selected, uncached eager T5 forwards.

The context records producer-supplied inputs and a tap policy. It does not
attest model identity, execution, semantic token alignment, or generation.
"""
from dataclasses import dataclass, field, replace

import torch

from .flan_t5_trace import T5TracePoint, trace_t5_forward
from .trace_schema import GenerationContext, RunIdentity, TraceEvent


T5_CONTRACT = "t5-uncached-forward-eager-v1"
_ORDER = {"self_attention": 0, "feed_forward": 1, "block": 2}
_TRANSFORM = "restore_singleton_batch_and_token_axes"
_MAX_INPUT_TOKENS = 4096
_MAX_POINTS = 128


def _point_order(point):
    return (point.stack == "decoder", point.layer, _ORDER[point.module], point.batch, point.token)


def _rows(name, rows, *, maximum):
    if (type(rows) is not tuple or not rows
            or any(type(row) is not tuple or not row for row in rows)):
        raise ValueError(f"{name} must be nonempty immutable rows")
    width = len(rows[0])
    if (any(len(row) != width for row in rows)
            or any(type(value) is not int or value < 0 or value >= maximum
                   for row in rows for value in row)):
        raise ValueError(f"{name} must be rectangular with values in range")


@dataclass(frozen=True, slots=True)
class T5ForwardContext:
    """Immutable producer record for one CPU, uncached T5 forward."""

    contract: str
    run_id: str
    input_ids: tuple[tuple[int, ...], ...] = field(repr=False)
    attention_mask: tuple[tuple[int, ...], ...] = field(repr=False)
    decoder_input_ids: tuple[tuple[int, ...], ...] = field(repr=False)
    decoder_attention_mask: tuple[tuple[int, ...], ...] = field(repr=False)
    num_encoder_layers: int
    num_decoder_layers: int
    hidden_size: int
    vocab_size: int
    points: tuple[T5TracePoint, ...]

    def __post_init__(self):
        if type(self.contract) is not str or not self.contract.strip():
            raise ValueError("contract must be a nonempty string")
        RunIdentity(self.run_id)
        for name in ("num_encoder_layers", "num_decoder_layers", "hidden_size", "vocab_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("input_ids", "decoder_input_ids"):
            _rows(name, getattr(self, name), maximum=self.vocab_size)
        for name in ("attention_mask", "decoder_attention_mask"):
            _rows(name, getattr(self, name), maximum=2)
        enc, dec = self.input_ids, self.decoder_input_ids
        if (len(enc) != len(dec) or len(enc) != len(self.attention_mask)
                or len(dec) != len(self.decoder_attention_mask)
                or len(enc[0]) != len(self.attention_mask[0])
                or len(dec[0]) != len(self.decoder_attention_mask[0])):
            raise ValueError("each stack requires matching ID and mask shapes and batch size")
        if len(enc) * (len(enc[0]) + len(dec[0])) > _MAX_INPUT_TOKENS:
            raise ValueError("input token budget exceeded")
        if (type(self.points) is not tuple or not self.points or len(self.points) > _MAX_POINTS
                or any(type(point) is not T5TracePoint for point in self.points)
                or len(set(self.points)) != len(self.points)
                or self.points != tuple(sorted(self.points, key=_point_order))):
            raise ValueError("points must be distinct and in selected execution order")
        for point in self.points:
            mask = self.attention_mask if point.stack == "encoder" else self.decoder_attention_mask
            layers = self.num_encoder_layers if point.stack == "encoder" else self.num_decoder_layers
            if (point.layer >= layers or point.batch >= len(mask) or point.token >= len(mask[0])
                    or mask[point.batch][point.token] != 1):
                raise ValueError("point is outside its stack or selects a masked token")


@dataclass(frozen=True, slots=True)
class T5ComparableForward:
    output: object
    events: tuple
    t5_context: T5ForwardContext


def _dimensions(model):
    return (len(model.encoder.block), len(model.decoder.block),
            model.config.d_model, model.config.vocab_size)


def _tensor_rows(tensor):
    return tuple(tuple(row) for row in tensor.tolist())


def _inputs_match(inputs, shapes, snapshot):
    # Check metadata before reading values: an incompatible device may not
    # support tolist(), and Python considers equal int/float values equal.
    return (all(type(item) is torch.Tensor and item.device.type == "cpu"
                and item.dtype == torch.int64 and tuple(item.shape) == shape
                for item, shape in zip(inputs, shapes, strict=True))
            and tuple(_tensor_rows(item) for item in inputs) == snapshot)


def trace_t5_comparable_forward(model, *, input_ids, attention_mask, decoder_input_ids,
                                decoder_attention_mask, points, run_id,
                                max_events=128, max_trace_bytes=16 * 1024 * 1024,
                                max_tensor_bytes=16 * 1024 * 1024,
                                max_input_tokens=4096, max_logits_bytes=128 * 1024 * 1024):
    """Trace one bounded CPU forward and record immutable input/tap context.

    Inputs are cloned before the model is called. Persistent changes to those
    clones or model dimensions are rejected. This cooperative check cannot
    detect arbitrary transient mutation or prove that the model used them.
    """
    inputs = (input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
    if (type(max_input_tokens) is not int or not 0 < max_input_tokens <= _MAX_INPUT_TOKENS
            or type(max_events) is not int or not 0 < max_events <= _MAX_POINTS):
        raise ValueError("comparison input and event budgets exceed the supported bounds")
    if (any(type(item) is not torch.Tensor or item.device.type != "cpu"
            or item.dtype != torch.int64 or item.ndim != 2 or not item.numel() for item in inputs)
            or input_ids.numel() + decoder_input_ids.numel() > max_input_tokens
            or attention_mask.shape != input_ids.shape
            or decoder_attention_mask.shape != decoder_input_ids.shape
            or input_ids.shape[0] != decoder_input_ids.shape[0]):
        raise ValueError("inputs must be bounded nonempty CPU int64 matrices")
    if (type(points) not in (list, tuple) or not points or len(points) > max_events
            or any(type(point) is not T5TracePoint for point in points)
            or len(set(points)) != len(points)):
        raise ValueError("points must be a bounded list of distinct T5TracePoint values")
    clones = tuple(item.detach().clone() for item in inputs)
    shapes = tuple(tuple(item.shape) for item in clones)
    snapshot = tuple(_tensor_rows(item) for item in clones)
    dimensions = _dimensions(model)
    context = T5ForwardContext(T5_CONTRACT, run_id, *snapshot, *dimensions,
                               tuple(sorted(points, key=_point_order)))
    output, records = trace_t5_forward(
        model, input_ids=clones[0], attention_mask=clones[1], decoder_input_ids=clones[2],
        decoder_attention_mask=clones[3], points=context.points, run_id=run_id,
        max_events=max_events, max_trace_bytes=max_trace_bytes, max_tensor_bytes=max_tensor_bytes,
        max_input_tokens=max_input_tokens, max_logits_bytes=max_logits_bytes)
    if not _inputs_match(clones, shapes, snapshot) or _dimensions(model) != dimensions:
        raise ValueError("model changed the recorded inputs or dimensions during the forward")
    # A lazy import avoids a module cycle with the comparator's context checker.
    from .transformer_localization import TensorObservation

    observations = []
    for record in records:
        original_shape = record.event.saved_shape
        value = record.value.unsqueeze(0).unsqueeze(0)
        event = replace(record.event, saved_shape=tuple(value.shape))
        observations.append(TensorObservation(event, value, original_shape, _TRANSFORM))
    return T5ComparableForward(output, tuple(observations), context)


def _matches_events(context, observations):
    if context.contract != T5_CONTRACT or len(observations) != len(context.points):
        return False
    for sequence, (observation, point) in enumerate(zip(observations, context.points, strict=True)):
        event = getattr(observation, "event", None)
        if type(event) is not TraceEvent:
            return False
        path = f"{point.stack}.block.{point.layer}"
        if point.module == "self_attention":
            path += ".layer.0"
        elif point.module == "feed_forward":
            path += ".layer.1" if point.stack == "encoder" else ".layer.2"
        kind = "block_output" if point.module == "block" else f"{point.module}_residual"
        length = len(context.input_ids[0] if point.stack == "encoder" else context.decoder_input_ids[0])
        if (event.identity != RunIdentity(context.run_id) or event.sequence != sequence
                or event.forward_index != 0 or event.invocation != 0
                or event.module_path != path or event.module_kind != f"t5_{point.stack}_{kind}"
                or event.layer_index != point.layer
                or event.output_path != (() if point.module == "feed_forward" else (0,))
                or event.shape != (len(context.input_ids), length, context.hidden_size)
                or event.dtype != "torch.float32" or event.device != "cpu"
                or event.capture_mode != "direct" or event.batch_axis != 0 or event.token_axis != 1
                or event.batch_positions != (point.batch,) or event.token_positions != (point.token,)
                or event.generation != GenerationContext(stage="forward", use_cache=False)
                or event.saved_shape != (1, 1, context.hidden_size)
                or event.saved_dtype != "torch.float32" or event.saved_device != "cpu"
                or getattr(observation, "original_saved_shape", None) != (context.hidden_size,)
                or getattr(observation, "comparison_transform", None) != _TRANSFORM):
            return False
    return True


def _valid_differences(value):
    return (type(value) is tuple and all(type(pos) is tuple and len(pos) == 2
            and all(type(x) is int and x >= 0 for x in pos) for pos in value)
            and len(set(value)) == len(value))


def check_t5_pair(contexts, reference, candidate, expected_encoder_differences,
                  expected_decoder_differences):
    """Validate a paired policy without including token IDs or masks in the report."""
    report = {"verified": False, "issues": [], "actual_encoder_differences": None,
              "actual_decoder_differences": None}
    if (type(contexts) is not tuple or len(contexts) != 2
            or any(type(context) is not T5ForwardContext for context in contexts)):
        report["issues"].append("two T5 forward contexts are required")
        return report
    if not _valid_differences(expected_encoder_differences) or not _valid_differences(expected_decoder_differences):
        report["issues"].append("explicit distinct encoder and decoder difference coordinates are required")
        return report
    left, right = contexts
    if not _matches_events(left, reference):
        report["issues"].append("reference events do not match the supported T5 policy")
    if not _matches_events(right, candidate):
        report["issues"].append("candidate events do not match the supported T5 policy")
    if left.run_id == right.run_id:
        report["issues"].append("T5 run IDs are not distinct")
    if (left.contract, left.num_encoder_layers, left.num_decoder_layers, left.hidden_size,
            left.vocab_size, left.points) != (right.contract, right.num_encoder_layers,
            right.num_decoder_layers, right.hidden_size, right.vocab_size, right.points):
        report["issues"].append("T5 model dimensions or selected policies differ")
    for stack, ids_name, mask_name, expected in (
            ("encoder", "input_ids", "attention_mask", expected_encoder_differences),
            ("decoder", "decoder_input_ids", "decoder_attention_mask", expected_decoder_differences)):
        a, b = getattr(left, ids_name), getattr(right, ids_name)
        a_mask, b_mask = getattr(left, mask_name), getattr(right, mask_name)
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            report["issues"].append(f"{stack} input shapes differ")
            continue
        actual = tuple((row, token) for row in range(len(a)) for token in range(len(a[0]))
                       if a[row][token] != b[row][token])
        report[f"actual_{stack}_differences"] = [list(pos) for pos in actual]
        if set(actual) != set(expected):
            report["issues"].append(f"{stack} differences do not match declared coordinates")
        if a_mask != b_mask:
            report["issues"].append(f"{stack} attention masks differ")
        if any(a_mask[row][token] != 1 or b_mask[row][token] != 1 for row, token in actual):
            report["issues"].append(f"{stack} differences include masked tokens")
    report["verified"] = not report["issues"]
    return report
