"""Bounded eager traces for one uncached T5 encoder-decoder forward.

The caller supplies an already-loaded model and explicit masks. This module
does not download assets, run generation, manage GPU resources, or infer global
cache positions. Traces use the shared ordered ActivationPlan and TraceEvent.
"""

from dataclasses import dataclass

import torch

from .intervention_plan import ActivationPlan
from .trace_schema import GenerationContext, RunIdentity, TraceEvent


@dataclass(frozen=True, slots=True)
class T5TracePoint:
    """One coordinate at a block, self-attention or feed-forward boundary."""

    stack: str
    layer: int
    module: str
    batch: int
    token: int
    invocation: int = 0

    def __post_init__(self):
        if self.stack not in ("encoder", "decoder") or type(self.stack) is not str:
            raise ValueError("stack must be encoder or decoder")
        if type(self.module) is not str or self.module not in ("block", "self_attention", "feed_forward"):
            raise ValueError("unsupported T5 boundary")
        for name in ("layer", "batch", "token", "invocation"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.invocation != 0:
            raise ValueError("only the first invocation is supported")


@dataclass(frozen=True, slots=True)
class T5TraceRecord:
    event: TraceEvent
    value: torch.Tensor


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def trace_t5_forward(model, *, input_ids, attention_mask, decoder_input_ids,
                     decoder_attention_mask, points, run_id,
                     max_events=128, max_trace_bytes=16 * 1024 * 1024,
                     max_tensor_bytes=16 * 1024 * 1024, max_input_tokens=4096,
                     max_logits_bytes=128 * 1024 * 1024):
    """Return the model output and selected boundary-token records for one forward.

    This is inference-only and passes ``use_cache=False``. The positions are
    local tensor indices in the explicitly supplied encoder or decoder inputs;
    they are not generation steps or absolute cache positions. Trace budgets
    cover saved captures and individual boundary tensors, not all model memory.
    """
    from transformers import T5ForConditionalGeneration
    from transformers.models.t5.modeling_t5 import T5Block, T5LayerFF, T5LayerSelfAttention

    if type(model) is not T5ForConditionalGeneration:
        raise TypeError("model must be a standard T5ForConditionalGeneration")
    if any(module.training for module in model.modules()) or model.is_gradient_checkpointing:
        raise ValueError("model must be in evaluation mode without gradient checkpointing")
    if any(getattr(config, "_attn_implementation", None) != "eager"
           for config in (model.config, model.encoder.config, model.decoder.config)):
        raise ValueError("only eager T5 attention is supported")
    if getattr(model.forward, "__func__", None) is not T5ForConditionalGeneration.forward:
        raise ValueError("modified root forward is unsupported")
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise ValueError("pre-existing forward hooks are unsupported")
    for name, value in (("max_events", max_events), ("max_trace_bytes", max_trace_bytes),
                        ("max_tensor_bytes", max_tensor_bytes), ("max_input_tokens", max_input_tokens),
                        ("max_logits_bytes", max_logits_bytes)):
        _positive(value, name)
    identity = RunIdentity(run_id)
    if type(points) not in (list, tuple) or not points or len(points) > max_events:
        raise ValueError("points must be a nonempty bounded list")
    if any(type(point) is not T5TracePoint for point in points) or len(set(points)) != len(points):
        raise ValueError("points must be distinct T5TracePoint values")

    values = ((input_ids, attention_mask, "encoder"),
              (decoder_input_ids, decoder_attention_mask, "decoder"))
    if any(type(ids) is not torch.Tensor or ids.dtype != torch.long or ids.ndim != 2
           or not ids.numel() for ids, _, _ in values):
        raise ValueError("input IDs must be nonempty two-dimensional int64 tensors")
    if input_ids.shape[0] != decoder_input_ids.shape[0]:
        raise ValueError("encoder and decoder batch sizes differ")
    if input_ids.numel() + decoder_input_ids.numel() > max_input_tokens:
        raise ValueError("input token budget exceeded")
    parameter = next(model.parameters())
    if parameter.device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU and CUDA tensors are supported")
    if torch.is_autocast_enabled(parameter.device.type):
        raise ValueError("autocast must be disabled")
    if any(value.device != parameter.device or value.dtype != torch.float32 for value in model.parameters()):
        raise ValueError("all model parameters must be float32 on one device")
    for ids, mask, name in values:
        if (type(mask) is not torch.Tensor or mask.ndim != 2 or mask.shape != ids.shape
                or mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or ids.device != mask.device or ids.device != parameter.device):
            raise ValueError(f"{name} IDs and mask must have matching shapes and model device")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"{name} mask must contain only zero or one")
    head = model.get_output_embeddings().weight
    if head.dtype != torch.float32:
        raise ValueError("only float32 output logits are supported")
    if input_ids.shape[0] * decoder_input_ids.shape[1] * head.shape[0] * head.element_size() > max_logits_bytes:
        raise ValueError("logit output budget exceeded")

    order = {"self_attention": 0, "feed_forward": 1, "block": 2}
    ordered = sorted(points, key=lambda p: (p.stack == "decoder", p.layer, order[p.module], p.batch, p.token))
    aliases = {}
    for path, module in model.named_modules(remove_duplicate=False):
        aliases.setdefault(id(module), []).append(path)
    steps = []
    for index, point in enumerate(ordered):
        stack = getattr(model, point.stack)
        mask = attention_mask if point.stack == "encoder" else decoder_attention_mask
        if point.layer >= len(stack.block):
            raise ValueError("layer index outside selected stack")
        if point.batch >= mask.shape[0] or point.token >= mask.shape[1]:
            raise ValueError("token coordinate outside selected stack")
        if not bool(mask[point.batch, point.token]):
            raise ValueError("masked token coordinates are unsupported")
        path = f"{point.stack}.block.{point.layer}"
        expected_type = T5Block
        if point.module == "self_attention":
            path += ".layer.0"
            expected_type = T5LayerSelfAttention
        elif point.module == "feed_forward":
            path += ".layer.1" if point.stack == "encoder" else ".layer.2"
            expected_type = T5LayerFF
        module = model.get_submodule(path)
        if type(module) is not expected_type or aliases[id(module)] != [path]:
            raise ValueError("selected boundary must have one standard module path")
        steps.append({"name": f"capture_{index}", "operation": "capture", "module_path": path,
                      "invocation": point.invocation, "batch_index": point.batch, "token_index": point.token})

    plan = ActivationPlan(model, steps, max_steps=max_events, max_invocations=1,
                          max_tensor_bytes=max_tensor_bytes, max_capture_bytes=max_trace_bytes)
    with torch.inference_mode(), plan:
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                       use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
    if output.past_key_values is not None or tuple(output.logits.shape) != (
            input_ids.shape[0], decoder_input_ids.shape[1], head.shape[0]) or output.logits.dtype != torch.float32:
        raise ValueError("model output differs from the supported uncached forward")
    captures, metadata = plan.captures, plan.metadata["steps"]
    records = []
    for index, (point, item) in enumerate(zip(ordered, metadata, strict=True)):
        value = captures[f"capture_{index}"]
        kind = "block_output" if point.module == "block" else f"{point.module}_residual"
        event = TraceEvent(identity=identity, sequence=index, forward_index=0,
            module_path=item["module_path"], module_kind=f"t5_{point.stack}_{kind}",
            layer_index=point.layer, invocation=item["invocation"], output_path=tuple(item["output_selector"]),
            shape=tuple(item["output_shape"]), dtype=item["dtype"], device=item["device"], capture_mode="direct",
            batch_positions=(point.batch,), token_positions=(point.token,), batch_axis=0, token_axis=1,
            generation=GenerationContext(stage="forward", use_cache=False), saved_shape=tuple(value.shape),
            saved_dtype=str(value.dtype), saved_device=str(value.device))
        records.append(T5TraceRecord(event, value))
    return output, tuple(records)
