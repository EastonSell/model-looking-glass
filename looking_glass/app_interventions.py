"""Fixed-prefix, single-vector GPT-2/Llama experiments using existing lease-safe plans.

Display results are local intervention observations, not causal explanations.
"""
import copy
import math
import uuid

import torch

from .intervention_plan import ActivationPlan
from .model_ownership import ModelLease
from .trace_schema import GenerationContext, RunIdentity, TraceEvent
from .transformer_trace import GPT2TraceEvent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_bits(a, b):
    return (a.dtype == b.dtype == torch.float32 and a.shape == b.shape
            and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def validate_request(runtime, source, operation, feature, value, maximum):
    require(type(maximum) is int and 0 < maximum <= 512 * 1024**2, "paired experiment requires positive bounded memory allowance")
    from transformers import GPT2LMHeadModel, LlamaForCausalLM
    from collections import Counter
    model, family = runtime.model, runtime._info["family"]
    require((family == "gpt2" and type(model) is GPT2LMHeadModel)
            or (family == "llama" and type(model) is LlamaForCausalLM),
            "interventions support exact GPT2LMHeadModel or LlamaForCausalLM only")
    if family == "llama":
        from .llama_trace import validate_llama_model
        require(validate_llama_model(model, synchronous_cpu=runtime._device.type == "cpu") == runtime._device,
                "model placement changed")
    else:
        from transformers import GPT2Model
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
        model = runtime.model
        require(model.config._attn_implementation == "eager" and not model.config.add_cross_attention
                and not model.is_gradient_checkpointing, "only eager self-attention without checkpointing is supported")
        require(type(model.transformer) is GPT2Model and type(model.transformer.h) is torch.nn.ModuleList
                and len(model.transformer.h) == model.config.n_layer
                and all(type(block) is GPT2Block and type(block.attn) is GPT2Attention and type(block.mlp) is GPT2MLP
                        for block in model.transformer.h), "standard GPT2 module tree required")
        require(all(getattr(module.forward, "__func__", None) is type(module).forward for module in model.modules()),
                "standard unmodified forward methods required")
        require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"), "autocast is unsupported")
        hooks = torch.nn.modules.module
        require(not hooks._global_forward_pre_hooks and not hooks._global_forward_hooks
                and all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()), "pre-existing forward hooks are unsupported")
        require(all(b.device == runtime._device and (not b.is_floating_point() or b.dtype == torch.float32)
                    for b in model.buffers()), "model buffer placement/dtype changed")
    paths = Counter(id(module) for _, module in model.named_modules(remove_duplicate=False))
    require(all(count == 1 for count in paths.values()), "aliased modules are unsupported")
    config = model.config
    depth = config.n_layer if family == "gpt2" else config.num_hidden_layers
    positions = config.n_positions if family == "gpt2" else config.max_position_embeddings
    require(type(source) is dict and set(source) == {"coordinate", "step", "model", "reference_value"}, "invalid intervention source")
    coordinate, step, model = (source[key] for key in ("coordinate", "step", "model"))
    require(all(type(x) is dict for x in (coordinate, step, model)), "source metadata must be objects")
    require(coordinate.get("status") == "known" and coordinate.get("batch_position") == 0
            and type(coordinate.get("batch_position")) is int and coordinate.get("stack") == "input", "unknown or unsupported source coordinate")
    layer, token, path = (coordinate.get(key) for key in ("layer_index", "token_position", "module_path"))
    require(type(layer) is int and 0 <= layer < depth, "invalid source layer")
    prefix = f"transformer.h.{layer}" if family == "gpt2" else f"model.layers.{layer}"
    attention = ".attn" if family == "gpt2" else ".self_attn"
    require(path in (prefix, prefix + attention, prefix + ".mlp"), "source module is not a canonical adapter boundary")
    require(type(step.get("app_forward_index")) is int and step["app_forward_index"] >= 0
            and coordinate.get("app_forward_index") == step["app_forward_index"]
            and type(step.get("run_id")) is str and bool(step["run_id"])
            and step.get("use_cache") is False, "source forward provenance missing")
    ids, mask = step.get("input_token_ids"), step.get("input_attention_mask")
    require(type(ids) is list and 1 <= len(ids) <= 64 and len(ids) <= positions
            and all(type(i) is int and 0 <= i < runtime.model.config.vocab_size for i in ids), "fixed input requires 1..64 exact valid token IDs")
    require(type(mask) is list and len(mask) == len(ids) and all(type(i) is int and i in (0, 1) for i in mask)
            and any(mask), "exact input attention mask required")
    require(type(token) is int and 0 <= token < len(ids) and coordinate.get("token_id") == ids[token], "source token does not match saved input")
    numerics = runtime._numerics()
    require(model.get("family") == family and model.get("device") == str(runtime._device)
            and model.get("dtype") == "float32" and type(model.get("execution_numerics")) is dict
            and model["execution_numerics"] == numerics, "source execution device/numerics no longer match")
    require(not runtime.model.training and all(not m.training for m in runtime.model.modules()), "model must remain in evaluation mode")
    require(all(p.dtype == torch.float32 and p.device == runtime._device for p in runtime.model.parameters()), "model placement changed")
    reference = source["reference_value"]
    width = config.n_embd if family == "gpt2" else config.hidden_size
    require(type(reference) is torch.Tensor and reference.device.type == "cpu" and reference.dtype == torch.float32
            and reference.layout == torch.strided and reference.shape == (width,) and not reference.requires_grad
            and bool(torch.isfinite(reference).all()), "reference must be a finite CPU float32 full vector")
    require(operation in ("zero", "add", "replace"), "unsupported operation")
    if operation == "zero":
        require(feature is None and value is None, "zero targets the whole vector without a scalar")
    else:
        require(type(feature) is int and 0 <= feature < width, "invalid feature index")
        require(type(value) in (int, float) and math.isfinite(value) and abs(value) <= torch.finfo(torch.float32).max, "value must be finite float32-representable")
    logits_bytes = len(ids) * runtime.model.config.vocab_size * 4
    require(logits_bytes <= 16 * 1024**2, "full logits exceed 16 MiB limit")
    full_bytes = len(ids) * width * 4
    # Logical retained/copy/metric allowance, not a peak-RAM guarantee.
    if family == "gpt2":
        workspace = 3 * config.n_head * len(ids)**2 * 4 + 16 * len(ids) * (config.n_inner or 4 * width)
    else:
        workspace = (3 * config.num_attention_heads * len(ids)**2 * 4
            + 6 * len(ids) * config.intermediate_size * 4
            + 4 * len(ids) * ((config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim + width) * 4
            + config.head_dim * 2 + len(ids) * 48)
    planned = 10 * logits_bytes + 12 * full_bytes + 16 * width * 4 + workspace
    require(type(maximum) is int and 0 < maximum <= 512 * 1024**2 and planned <= maximum, "paired experiment exceeds logical memory allowance")
    return dict(family=family, coordinate=copy.deepcopy(coordinate), step=copy.deepcopy(step), reference=reference.clone(),
                path=path, layer=layer, token=token, ids=list(ids), mask=list(mask), operation=operation,
                feature=feature, value=value, width=width, logits_bytes=logits_bytes, full_bytes=full_bytes,
                planned_bytes=planned, maximum=maximum, numerics=numerics)


def replacement(fresh, operation, feature, value):
    """Apply scalar arithmetic only at its feature; preserve all other bits."""
    if operation == "zero":
        return None, torch.zeros_like(fresh)
    source = fresh.clone()
    source[feature] = fresh[feature] + value if operation == "add" else value
    return source, source.clone()


def execute(runtime, request):
    from .app_runtime import AppCapture, _token_labels
    model, device = runtime.model, runtime._device
    ids = torch.tensor([request["ids"]], dtype=torch.int64, device=device)
    mask = torch.tensor([request["mask"]], dtype=torch.int64, device=device)
    token, width = request["token"], request["width"]
    source = desired_cpu = None
    family = request["family"]
    rotary_snapshot = None
    if family == "llama":
        from .llama_trace import snapshot_llama_rotary, check_llama_rotary
    events, steps = [], []
    run = uuid.uuid4().hex
    kwargs = dict(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True,
                  output_attentions=False, output_hidden_states=False)
    if family == "llama": kwargs.update(past_key_values=None, logits_to_keep=0)
    expected_ids, expected_mask = ids.clone(), mask.clone()

    def logits(output):
        value = output.logits
        require(type(value) is torch.Tensor and value.device == device and value.shape == (1, len(request["ids"]), model.config.vocab_size)
                and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "invalid model logits")
        if family == "llama":
            require(output.past_key_values is None, "unexpected cached model output")
            require(torch.equal(ids, expected_ids) and torch.equal(mask, expected_mask), "fixed input IDs/mask changed")
            check_llama_rotary(model, rotary_snapshot)
        return value.detach().cpu().clone()

    def trial(lease, operation, index):
        nonlocal source, desired_cpu
        full = dict(module_path=request["path"], invocation=0)
        chosen = dict(full, batch_index=0, token_index=token)
        operation_step = dict(chosen, name="operation", operation=operation)
        if operation in ("add", "patch"): operation_step["value"] = source
        plan = ActivationPlan(model, [dict(full, name="before", operation="capture"), operation_step,
            dict(full, name="after", operation="capture")], max_tensor_bytes=request["full_bytes"],
            max_capture_bytes=2 * request["full_bytes"], max_source_bytes=width * 4,
            max_steps=3, max_invocations=1, _parent_lease=lease)
        with plan:
            output = plan.run_synchronous_cpu(**kwargs) if device.type == "cpu" else model(**kwargs)
            current = logits(output)
            del output
            if device.type == "cuda": torch.cuda.synchronize(device)
        captures = plan.captures
        expected_selector = ([0] if request["path"].endswith((".attn", ".self_attn"))
                             or (family == "gpt2" and not request["path"].endswith(".mlp")) else [])
        metadata = plan.metadata
        require(set(captures) == {"before", "after"} and len(metadata["steps"]) == 3,
                "intervention capture inventory incomplete")
        for order, (name, action) in enumerate((("before", "capture"), ("operation", operation), ("after", "capture"))):
            row = metadata["steps"][order]
            require(row["name"] == name and row["order"] == order and row["operation"] == action
                    and row["module_path"] == request["path"] and row["module_aliases"] == [request["path"]]
                    and row["forward_index"] == 0 and row["invocation"] == 0
                    and row["output_selector"] == expected_selector and row["output_shape"] == [1, len(request["ids"]), width]
                    and row["selected_shape"] == ([width] if name == "operation" else [1, len(request["ids"]), width])
                    and row["batch_index"] == (0 if name == "operation" else None)
                    and row["token_index"] == (token if name == "operation" else None)
                    and row["device"] == str(device) and row["dtype"] == "torch.float32",
                    "intervention boundary metadata differs from canonical contract")
        require(all(value.device == device and value.dtype == torch.float32
                    and tuple(value.shape) == (1, len(request["ids"]), width)
                    and bool(torch.isfinite(value).all()) for value in captures.values()),
                "invalid full boundary captures")
        before, after = captures["before"].detach().cpu(), captures["after"].detach().cpu()
        require(same_bits(before[0, token], request["reference"]), "fresh selected activation differs from original source")
        require(same_bits(before[:, :token], after[:, :token]) and same_bits(before[:, token + 1:], after[:, token + 1:]), "intervention changed other token positions at selected site")
        if operation == "null":
            fresh = captures["before"][0, token].detach().clone()
            source, desired = replacement(fresh, request["operation"], request["feature"], request["value"])
            require(bool(torch.isfinite(desired).all()), "operation produces nonfinite activation")
            desired_cpu = desired.detach().cpu().clone()
        elif request["operation"] != "zero":
            feature = request["feature"]
            require(same_bits(before[0, token, :feature], after[0, token, :feature])
                    and same_bits(before[0, token, feature + 1:], after[0, token, feature + 1:]), "operation changed another feature")
        expected = request["reference"] if operation == "null" else desired_cpu
        require(same_bits(after[0, token], expected), "operation did not land exactly")
        run_id = f"{run}-{index}"
        kind = ("attention" if request["path"].endswith(".self_attn") else
                "attn" if request["path"].endswith(".attn") else
                "mlp" if request["path"].endswith(".mlp") else "block")
        output_path = (0,) if kind in ("attn", "attention") or (family == "gpt2" and kind == "block") else ()
        for phase, full_value in (("before", before), ("after", after)):
            value = full_value[:, token:token + 1].clone().contiguous()
            if family == "gpt2":
                event = GPT2TraceEvent(run_id, len(events), 0, request["path"], kind, request["layer"], 0,
                    output_path, (0,), (token,), tuple(before.shape), "torch.float32", str(device), "direct", value,
                    GenerationContext(stage="forward", use_cache=False))
                metadata = event.metadata()
            else:
                event = TraceEvent(identity=RunIdentity(run_id), sequence=len(events), forward_index=0,
                    module_path=request["path"], module_kind=f"llama_{kind}", layer_index=request["layer"],
                    invocation=0, output_path=output_path, shape=tuple(before.shape), dtype="torch.float32",
                    device=str(device), capture_mode="direct", batch_positions=(0,), token_positions=(token,),
                    batch_axis=0, token_axis=1, generation=GenerationContext(stage="forward", use_cache=False),
                    saved_shape=tuple(value.shape), saved_dtype="torch.float32", saved_device="cpu")
                metadata = event.to_dict()
                if device.type == "cuda": metadata["app_offloaded_from_device"] = str(device)
            metadata.update(app_forward_index=index, app_sequence=len(events), experiment_phase=("null_" if index == 0 else "intervention_") + phase,
                            experiment_trial="null" if index == 0 else "intervention")
            events.append(AppCapture(value, metadata))
        steps.append({"app_forward_index": index, "run_id": run_id, "use_cache": False,
            "execution": "fixed saved prefix; single uncached intervention experiment forward", "event_end": len(events),
            "input_token_ids": list(request["ids"]), "input_attention_mask": list(request["mask"]), "decoder_token_ids": None})
        return current

    with torch.inference_mode(), ModelLease(model) as lease:
        if family == "llama": rotary_snapshot = snapshot_llama_rotary(model)
        baseline = logits(model(**kwargs))
        null = trial(lease, "null", 0)
        require(same_bits(baseline, null), "null control changed full logits")
        del null
        treatment = trial(lease, "zero" if request["operation"] == "zero" else "patch", 1)
        restoration = logits(model(**kwargs))
        require(same_bits(baseline, restoration), "unhooked restoration changed full logits")
        del restoration
        if device.type == "cuda": torch.cuda.synchronize(device)
    labels, labels_note = _token_labels(runtime.tokenizer, steps)
    difference = treatment.double() - baseline.double()
    displacement = desired_cpu.double() - request["reference"].double()
    def top(value):
        scores, indices = torch.topk(value[0, -1], min(5, value.shape[-1]))
        return [{"token_id": int(i), "text": runtime.tokenizer.decode([int(i)]), "logit": float(v)} for i, v in zip(indices, scores)]
    changed = not same_bits(desired_cpu, request["reference"])
    return {"answer": "Fixed-prefix intervention comparison completed", "task": "activation-intervention",
        "run_kind": "intervention_comparison", "events": events, "model": runtime.info(), "steps": steps, "token_labels": labels, "token_labels_note": labels_note,
        "coverage": {"complete": True, "captured_events": 4, "captured_bytes": 4 * width * 4,
            "full_feature_vectors": True, "forward_count": 2, "captured_forward_count": 2, "total_model_forwards": 4,
            "scope": "Selected-site before/after vectors from null and treatment trials only; four total model forwards including unhooked controls."},
        "experiment": {"source_coordinate": request["coordinate"], "source_step": request["step"],
            "input_prefix": request["ids"], "input_attention_mask": request["mask"],
            "operation": request["operation"], "feature_index": request["feature"], "requested_value": request["value"],
            "operation_semantics": "Zero whole vector, or change only the selected feature on a fresh baseline clone and patch it; other feature bits preserved.",
            "activation_changed": changed, "realized_displacement_l2": float(displacement.norm()),
            "realized_feature_value": None if request["feature"] is None else float(desired_cpu[request["feature"]]),
            "controls": {"null_full_logits_bitwise_equal": True, "restoration_full_logits_bitwise_equal": True,
                **({"static_rotary_state_unchanged": True, "fixed_input_ids_mask_unchanged": True} if family == "llama" else {}),
                "source_vector_bitwise_equal": True, "other_token_positions_at_selected_site_bitwise_equal": True,
                **({"other_features_at_selected_vector_bitwise_equal": True} if request["operation"] != "zero" else {})},
            "full_logits": {"shape": list(baseline.shape), "bitwise_equal": same_bits(baseline, treatment),
                "max_absolute_change": float(difference.abs().max()), "l2_change": float(difference.norm())},
            "comparison": {"full_logits_max_abs_delta": float(difference.abs().max()),
                "full_logits_l2_delta": float(difference.norm()), "last_position_max_abs_delta": float(difference[0, -1].abs().max()),
                "baseline_next_token_id": int(baseline[0, -1].argmax()), "intervention_next_token_id": int(treatment[0, -1].argmax()),
                "selected_vector_changed": changed},
            "readout_position": len(request["ids"]) - 1, "baseline_top_next_tokens": top(baseline),
            "intervention_top_next_tokens": top(treatment), "execution_numerics": request["numerics"],
            "logical_planned_bytes": request["planned_bytes"], "logical_allowance_bytes": request["maximum"],
            "scope": "One exact fixed input and next-token readout; no generated continuation, checkpoint edit, causal explanation or backdoor proof. Logical budgets are not peak-memory guarantees."}}
