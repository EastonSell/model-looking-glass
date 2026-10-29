"""Bounded eager DistilBERT masked-LM observations, using the shared schema.

This CPU-only, single-forward adapter is built from the existing capture-only
ActivationPlan. It neither loads a checkpoint nor performs generation.
"""

from collections import Counter
from dataclasses import dataclass

import torch

from .intervention_plan import ActivationPlan
from .encoder_forward import DISTILBERT_CONTRACT, EncoderForwardContext
from .trace_schema import GenerationContext, RunIdentity, TraceEvent
from .transformer_localization import TensorObservation
from .transformer_trace import _indices


@dataclass(frozen=True)
class DistilBertTraceResult:
    """The MLM output, selected snapshots, and recorded encoder context.

    Each event is a TensorObservation. Its value is a view with restored
    singleton batch/token axes over the independently captured vector; the
    observation records that transformation and the original saved shape.
    """

    output: object
    events: tuple[TensorObservation, ...]
    attention_mask: tuple[tuple[int, ...], ...]
    tensor_bytes: int
    encoder_context: EncoderForwardContext | None = None


def trace_distilbert_masked_lm(model, input_ids, attention_mask, *, run_id,
                               layers=(0,), modules=("attention", "ffn", "block"),
                               samples=(0,), tokens=(0,), max_events=64,
                               max_trace_bytes=1024 * 1024,
                               max_tensor_bytes=16 * 1024 * 1024):
    """Run one eager, CPU float32 masked-LM forward with bounded snapshots.

    Selectors are ordered local indices; one event is retained per selected
    module, sample and token. Output attentions and hidden-state collections
    are forced off. The caller must exclusively own an unmodified evaluation-
    mode model. Caps cover captured tensors and individual module outputs,
    not model state, logits, metadata, temporary allocations or peak memory.
    """
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        raise RuntimeError("compiled execution is not supported")
    from transformers import DistilBertForMaskedLM
    from transformers.models.distilbert.modeling_distilbert import (
        FFN, MultiHeadSelfAttention, TransformerBlock,
    )

    if type(model) is not DistilBertForMaskedLM or model.config.model_type != "distilbert":
        raise TypeError("model must be an unmodified DistilBertForMaskedLM")
    if model.config._attn_implementation != "eager":
        raise ValueError("only eager DistilBERT attention is supported")
    if model.is_gradient_checkpointing or any(m.training for m in model.modules()):
        raise ValueError("model must be in evaluation mode without gradient checkpointing")
    if (any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or any(buffer.device.type != "cpu" for buffer in model.buffers())):
        raise ValueError("model parameters must be CPU float32 and buffers must be on CPU")
    if torch.is_autocast_enabled("cpu"):
        raise ValueError("CPU autocast is not supported")
    blocks = model.distilbert.transformer.layer
    if (not isinstance(blocks, torch.nn.ModuleList) or not blocks
            or any(type(block) is not TransformerBlock
                   or type(block.attention) is not MultiHeadSelfAttention
                   or type(block.ffn) is not FFN for block in blocks)):
        raise TypeError("DistilBERT blocks, attention and FFN must be standard classes")
    hook_state = torch.nn.modules.module
    if (hook_state._global_forward_pre_hooks or hook_state._global_forward_hooks
            or any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())):
        raise ValueError("model must have no existing local or global forward hooks")
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    for name, value in (("max_events", max_events), ("max_trace_bytes", max_trace_bytes),
                        ("max_tensor_bytes", max_tensor_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    layers, samples, tokens = (_indices("layers", layers), _indices("samples", samples),
                               _indices("tokens", tokens))
    if layers is None or samples is None or tokens is None:
        raise ValueError("layers, samples and tokens must be explicit nonempty selections")
    if (not isinstance(modules, (tuple, list)) or not modules
            or any(type(kind) is not str or kind not in ("attention", "ffn", "block") for kind in modules)
            or len(set(modules)) != len(modules)):
        raise ValueError("modules must contain distinct attention, ffn or block roles")
    if (not isinstance(input_ids, torch.Tensor) or input_ids.device.type != "cpu"
            or input_ids.dtype != torch.int64 or input_ids.ndim != 2
            or not input_ids.shape[0] or not input_ids.shape[1]):
        raise ValueError("input_ids must be a nonempty CPU int64 [batch, token] tensor")
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.device.type != "cpu"
            or attention_mask.dtype != torch.int64 or attention_mask.shape != input_ids.shape
            or bool(((attention_mask != 0) & (attention_mask != 1)).any())):
        raise ValueError("attention_mask must be a matching binary CPU int64 tensor")
    ids, mask = input_ids.detach().clone(), attention_mask.detach().clone()
    batch, length = ids.shape
    if (length > model.config.max_position_embeddings or bool((ids < 0).any())
            or bool((ids >= model.config.vocab_size).any())):
        raise ValueError("input IDs or sequence length exceed model bounds")
    if (any(layer >= len(blocks) for layer in layers)
            or any(sample >= batch for sample in samples)
            or any(token >= length for token in tokens)):
        raise ValueError("selected layer, sample or token is out of range")
    count = len(layers) * len(modules) * len(samples) * len(tokens)
    if count > max_events:
        raise ValueError("selected events exceed max_events")
    hidden = model.config.dim
    if count * hidden * 4 > max_trace_bytes:
        raise ValueError("selected snapshots exceed max_trace_bytes")
    if batch * length * hidden * 4 > max_tensor_bytes:
        raise ValueError("module output exceeds max_tensor_bytes")
    # Record the exact private clones before the call, and reject persistent
    # mutation by the called model before publishing an input context.
    id_rows = tuple(tuple(int(x) for x in row) for row in ids.tolist())
    mask_rows = tuple(tuple(int(x) for x in row) for row in mask.tolist())
    dimensions = (len(blocks), hidden, model.config.vocab_size, model.config.max_position_embeddings)

    # Hooks complete in layer order: attention, FFN, then enclosing block.
    # Group all selected slices of one invocation together for ActivationPlan.
    selections, steps = [], []
    for layer in sorted(layers):
        for kind in ("attention", "ffn", "block"):
            if kind not in modules:
                continue
            path = f"distilbert.transformer.layer.{layer}"
            if kind != "block":
                path += f".{kind}"
            for sample in samples:
                for token in tokens:
                    name = f"capture_{len(steps)}"
                    selections.append((layer, kind, path, sample, token, name))
                    steps.append({"name": name, "operation": "capture", "module_path": path,
                                  "invocation": 0, "batch_index": sample, "token_index": token})

    counts = Counter(id(module) for _, module in model.named_modules(remove_duplicate=False))
    if any(counts[id(model.get_submodule(path))] != 1 for _, _, path, _, _, _ in selections):
        raise ValueError("selected module aliases cannot be attributed to a unique path")

    plan = ActivationPlan(model, steps, max_steps=max_events, max_invocations=1,
                          max_tensor_bytes=max_tensor_bytes, max_capture_bytes=max_trace_bytes,
                          max_source_bytes=1)
    with torch.no_grad(), plan:
        output = model(input_ids=ids, attention_mask=mask, output_attentions=False,
                       output_hidden_states=False, return_dict=True)
    if (ids.device.type != "cpu" or mask.device.type != "cpu"
            or ids.dtype != torch.int64 or mask.dtype != torch.int64
            or tuple(tuple(int(x) for x in row) for row in ids.tolist()) != id_rows
            or tuple(tuple(int(x) for x in row) for row in mask.tolist()) != mask_rows
            or (len(model.distilbert.transformer.layer), model.config.dim,
                model.config.vocab_size, model.config.max_position_embeddings) != dimensions):
        raise RuntimeError("encoder inputs or model dimensions changed during the forward")

    steps_seen = plan.metadata["steps"]
    captures = plan.captures
    if len(steps_seen) != count or len(captures) != count:
        raise RuntimeError("plan did not retain all selected observations")
    observations = []
    for sequence, ((layer, kind, path, sample, token, name), step) in enumerate(zip(selections, steps_seen)):
        selector = [] if kind == "ffn" else [0]
        value = captures[name]
        if (step["name"] != name or step["order"] != sequence or step["operation"] != "capture"
                or step["module_path"] != path or step["forward_index"] != 0
                or step["invocation"] != 0 or step["batch_index"] != sample
                or step["token_index"] != token or step["output_selector"] != selector
                or step["output_shape"] != [batch, length, hidden]
                or step["selected_shape"] != [hidden] or step["dtype"] != "torch.float32"
                or step["device"] != "cpu" or value.shape != (hidden,)
                or value.dtype != torch.float32 or value.device.type != "cpu"):
            raise RuntimeError("captured output does not match the declared DistilBERT boundary")
        view = value.unsqueeze(0).unsqueeze(0)
        event = TraceEvent(
            identity=RunIdentity(run_id), sequence=sequence, forward_index=0,
            module_path=path, module_kind=kind, layer_index=layer, invocation=0,
            output_path=tuple(selector), shape=(batch, length, hidden),
            dtype=step["dtype"], device=step["device"], capture_mode="direct",
            batch_positions=(sample,), token_positions=(token,), batch_axis=0, token_axis=1,
            generation=GenerationContext(), saved_shape=tuple(view.shape),
            saved_dtype=str(view.dtype), saved_device=str(view.device),
        )
        observations.append(TensorObservation(
            event, view, original_saved_shape=tuple(value.shape),
            comparison_transform="restore_singleton_batch_and_token_axes"))
    size = sum(item.value.numel() * item.value.element_size() for item in observations)
    context = EncoderForwardContext(
        contract=DISTILBERT_CONTRACT, run_id=run_id,
        input_ids=id_rows, attention_mask=mask_rows, num_layers=dimensions[0], hidden_size=dimensions[1],
        vocab_size=dimensions[2], max_positions=dimensions[3],
        layers=tuple(sorted(layers)), modules=tuple(role for role in ("attention", "ffn", "block")
                                                   if role in modules),
        samples=samples, tokens=tokens,
    )
    return DistilBertTraceResult(output, tuple(observations), mask_rows, size, context)
