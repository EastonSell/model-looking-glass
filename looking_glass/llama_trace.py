"""Bounded standard eager Llama observations; full outputs, no cache or chat templating."""
from dataclasses import dataclass
import math
import copy
import torch
from .intervention_plan import ActivationPlan
from .trace_schema import GenerationContext, RunIdentity, TraceEvent
from .transformer_localization import TensorObservation


@dataclass(frozen=True)
class LlamaTraceResult:
    output: object
    events: tuple[TensorObservation, ...]
    tensor_bytes: int


def validate_llama_config(config):
    """Reject unsupported RoPE/head geometry before construction or model retirement."""
    get = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    scaling = get('rope_scaling')
    if scaling is not None:
        keys = {'rope_type', 'factor', 'low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings'}
        if type(scaling) is not dict or set(scaling) != keys or scaling.get('rope_type') != 'llama3':
            raise ValueError('Llama RoPE requires default None or exact static llama3 keys; other scaling/aliases unsupported')
        for key in ('factor', 'low_freq_factor', 'high_freq_factor'):
            value = scaling[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'llama3 {key} must be finite numeric, not boolean')
        original = scaling['original_max_position_embeddings']
        maximum = get('max_position_embeddings')
        if (scaling['factor'] < 1 or not 0 < scaling['low_freq_factor'] < scaling['high_freq_factor']
                or type(original) is not int or original <= 0 or type(maximum) is not int or original >= maximum):
            raise ValueError('invalid static llama3 frequency/context range')
    partial = get('partial_rotary_factor', 1.0)
    if type(partial) not in (int, float) or partial != 1:
        raise ValueError('partial_rotary_factor must be exactly 1')
    for key in ('hidden_size', 'num_attention_heads', 'num_hidden_layers', 'intermediate_size', 'vocab_size', 'max_position_embeddings'):
        value = get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f'Llama {key} must be a positive integer')
    heads = get('num_attention_heads')
    kv = get('num_key_value_heads')
    kv = heads if kv is None else kv
    head_dim = get('head_dim')
    head_dim = get('hidden_size') // heads if head_dim is None else head_dim
    if (type(kv) is not int or kv <= 0 or heads % kv or type(head_dim) is not int
            or head_dim <= 0 or head_dim % 2 or get('hidden_size') % heads):
        raise ValueError('unsupported Llama attention/GQA head geometry')
    theta = get('rope_theta', 10000.0)
    if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
        raise ValueError('Llama rope_theta must be finite and positive')
    if type(get('pretraining_tp', 1)) is not int or get('pretraining_tp', 1) != 1:
        raise ValueError('tensor-parallel pretraining layouts are unsupported')


def trace_llama_causal_lm(model, input_ids, attention_mask, *, run_id,
        layers=(0,), modules=('attention', 'mlp', 'block'), max_events=128,
        max_trace_bytes=16*1024**2, max_tensor_bytes=16*1024**2,
        max_input_bytes=1024**2, max_logits_bytes=64*1024**2, synchronous_cpu=True):
    """One full uncached float32 forward; CPU default, explicit single-CUDA opt-in.

    Each selected boundary retains its entire [batch, token, hidden] output.
    Limits are logical payload bounds, not peak process/device-memory bounds.
    Caller owns CUDA failure recovery; no completed result escapes failed cleanup.
    """
    from transformers.models.llama.modeling_llama import (LlamaForCausalLM, LlamaModel,
        LlamaDecoderLayer, LlamaAttention, LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding)
    if type(model) is not LlamaForCausalLM or model.config.model_type != 'llama':
        raise TypeError('model must be a standard LlamaForCausalLM')
    validate_llama_config(model.config)
    if hasattr(torch, 'compiler') and torch.compiler.is_compiling():
        raise ValueError('compiled execution is unsupported')
    if type(synchronous_cpu) is not bool:
        raise ValueError('synchronous_cpu must be boolean')
    params = tuple(model.parameters())
    if not params:
        raise ValueError('standard model parameters required')
    device = params[0].device
    if (device.type not in ('cpu', 'cuda') or (synchronous_cpu and device.type != 'cpu')
            or any(p.device != device or p.dtype != torch.float32 for p in params)
            or any(b.device != device or b.is_complex() or b.is_quantized
                or (b.is_floating_point() and b.dtype != torch.float32) for b in model.buffers())):
        raise ValueError('one exact CPU/CUDA device and float32 floating parameters/buffers required; CPU is default')
    if (any(m.training for m in model.modules()) or model.is_gradient_checkpointing
            or torch.is_autocast_enabled(device.type)):
        raise ValueError('eval without checkpointing or selected-device autocast required')
    if (torch.nn.modules.module._global_forward_pre_hooks or torch.nn.modules.module._global_forward_hooks
            or any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())):
        raise ValueError('pre-existing forward hooks are unsupported')
    allowed = (LlamaForCausalLM, LlamaModel, LlamaDecoderLayer, LlamaAttention,
        LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding, torch.nn.Linear,
        torch.nn.Embedding, torch.nn.ModuleList)
    # Llama MLP's named activation is a standard Transformers activation module.
    from transformers.activations import ACT2FN
    activation_class = type(ACT2FN[model.config.hidden_act])
    for module in model.modules():
        cls = type(module)
        if cls not in allowed + (activation_class,):
            raise ValueError('unsupported Llama module tree')
        if cls is not torch.nn.ModuleList and getattr(module.forward, '__func__', None) is not cls.forward:
            raise ValueError('unmodified standard module forwards required')
    core = model.model
    expected_rope = 'default' if model.config.rope_scaling is None else 'llama3'
    if (type(core) is not LlamaModel or type(core.layers) is not torch.nn.ModuleList
            or len(core.layers) != model.config.num_hidden_layers
            or model.config._attn_implementation != 'eager' or core.config._attn_implementation != 'eager'
            or core.rotary_emb.rope_type != expected_rope):
        raise ValueError('standard eager Llama with default/static llama3 RoPE required')
    h, vocab = model.config.hidden_size, model.config.vocab_size
    head_dim = model.config.head_dim
    heads, kv = model.config.num_attention_heads, model.config.num_key_value_heads
    from transformers.modeling_rope_utils import _compute_default_rope_parameters, _compute_llama3_parameters
    rotary = core.rotary_emb
    expected_init = _compute_default_rope_parameters if expected_rope == 'default' else _compute_llama3_parameters
    if (rotary.config is not model.config or rotary.rope_init_fn is not expected_init
            or type(rotary.inv_freq) is not torch.Tensor or rotary.inv_freq.dtype != torch.float32
            or rotary.inv_freq.device != device or tuple(rotary.inv_freq.shape) != (head_dim // 2,)
            or not bool(torch.isfinite(rotary.inv_freq).all())
            or type(rotary.attention_scaling) not in (float, int) or rotary.attention_scaling != 1.0
            or rotary.max_seq_len_cached != model.config.max_position_embeddings
            or rotary.original_max_seq_len != model.config.max_position_embeddings):
        raise ValueError('unsupported static rotary implementation/state')
    if (tuple(core.embed_tokens.weight.shape) != (vocab, h)
            or tuple(model.lm_head.weight.shape) != (vocab, h)):
        raise ValueError('standard embedding/output vocabulary geometry required')
    for i, layer in enumerate(core.layers):
        if (type(layer) is not LlamaDecoderLayer or type(layer.self_attn) is not LlamaAttention
                or type(layer.mlp) is not LlamaMLP or layer.self_attn.layer_idx != i
                or layer.self_attn.config._attn_implementation != 'eager'
                or layer.self_attn.head_dim != head_dim
                or tuple(layer.self_attn.q_proj.weight.shape) != (heads * head_dim, h)
                or any(tuple(p.weight.shape) != (kv * head_dim, h) for p in (layer.self_attn.k_proj, layer.self_attn.v_proj))
                or tuple(layer.self_attn.o_proj.weight.shape) != (h, heads * head_dim)
                or tuple(layer.mlp.down_proj.weight.shape) != (h, model.config.intermediate_size)
                or any(tuple(p.weight.shape) != (model.config.intermediate_size, h) for p in (layer.mlp.gate_proj, layer.mlp.up_proj))):
            raise ValueError('standard Llama decoder/head geometry required')
    for name, value in [('max_events', max_events), ('max_trace_bytes', max_trace_bytes),
            ('max_tensor_bytes', max_tensor_bytes), ('max_input_bytes', max_input_bytes), ('max_logits_bytes', max_logits_bytes)]:
        if type(value) is not int or value <= 0: raise ValueError(f'{name} must be positive integer')
    identity = RunIdentity(run_id)
    if len(run_id) > 128: raise ValueError('run_id too long')
    if (type(layers) not in (tuple, list) or not layers
            or any(type(x) is not int or not 0 <= x < len(core.layers) for x in layers) or len(set(layers)) != len(layers)):
        raise ValueError('layers must be distinct valid layer indices')
    roles = ('attention', 'mlp', 'block')
    if (type(modules) not in (tuple, list) or not modules
            or any(type(x) is not str or x not in roles for x in modules) or len(set(modules)) != len(modules)):
        raise ValueError('modules must be distinct supported boundary names')
    for name, value in [('input_ids', input_ids), ('attention_mask', attention_mask)]:
        if (type(value) is not torch.Tensor or value.device != device or value.dtype != torch.int64
                or value.ndim != 2 or not value.numel() or value.requires_grad or value.layout != torch.strided):
            raise ValueError(f'{name} must be a nonempty same-device int64 matrix')
    batch, length = input_ids.shape
    if (attention_mask.shape != input_ids.shape or length > model.config.max_position_embeddings
            or not bool(((attention_mask == 0) | (attention_mask == 1)).all())
            or not bool(attention_mask.any(dim=1).all()) or not bool(((input_ids >= 0) & (input_ids < vocab)).all())):
        raise ValueError('invalid IDs/mask or sequence exceeds position limit')
    size = batch * length * h * 4
    count = len(layers) * len(modules)
    if count > max_events or count * size > max_trace_bytes or size > max_tensor_bytes:
        raise ValueError('capture budget exceeded')
    if input_ids.numel() * 16 + head_dim * 2 > max_input_bytes or batch * length * vocab * 4 > max_logits_bytes:
        raise ValueError('input/logit byte budget exceeded')
    ids, mask = input_ids.detach().clone(), attention_mask.detach().clone()
    before_ids, before_mask = ids.clone(), mask.clone()
    # Only the small rotary buffer is copied, never model parameters. CPU integer
    # encodings avoid numerical recomputation or a CPU/GPU equivalence claim.
    rotary_buffer = rotary.inv_freq
    rotary_bits = rotary_buffer.detach().view(torch.int32).to('cpu', non_blocking=False).clone()
    rotary_state = (rotary.rope_type, rotary.attention_scaling, rotary.max_seq_len_cached,
        rotary.original_max_seq_len, model.config.rope_theta, model.config.max_position_embeddings,
        model.config.head_dim, getattr(model.config, "partial_rotary_factor", 1.0),
        copy.deepcopy(model.config.rope_scaling))
    aliases = {}
    for path, module in model.named_modules(remove_duplicate=False):
        aliases.setdefault(id(module), []).append(path)
    steps, coordinates = [], []
    for layer in sorted(layers):
        for role in roles:
            if role not in modules: continue
            path = f'model.layers.{layer}' + ({'attention': '.self_attn', 'mlp': '.mlp', 'block': ''}[role])
            if aliases[id(model.get_submodule(path))] != [path]:
                raise ValueError('selected module must have unique path')
            steps.append(dict(name=f'capture_{len(steps)}', operation='capture', module_path=path, invocation=0))
            coordinates.append((layer, role, path))
    plan = ActivationPlan(model, steps, max_steps=max_events, max_invocations=1,
        max_tensor_bytes=max_tensor_bytes, max_capture_bytes=max_trace_bytes, max_source_bytes=1)
    with torch.inference_mode(), plan:
        output = model(input_ids=ids, attention_mask=mask, use_cache=False,
            past_key_values=None, logits_to_keep=0, output_attentions=False, output_hidden_states=False, return_dict=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
    if (rotary.inv_freq is not rotary_buffer or rotary.rope_init_fn is not expected_init
            or rotary.config is not model.config or rotary.inv_freq.device != device
            or rotary.inv_freq.dtype != torch.float32 or tuple(rotary.inv_freq.shape) != (head_dim // 2,)
            or not torch.equal(rotary.inv_freq.detach().view(torch.int32).to('cpu', non_blocking=False), rotary_bits)
            or (rotary.rope_type, rotary.attention_scaling, rotary.max_seq_len_cached,
                rotary.original_max_seq_len, model.config.rope_theta, model.config.max_position_embeddings,
                model.config.head_dim, getattr(model.config, "partial_rotary_factor", 1.0),
                model.config.rope_scaling) != rotary_state):
        raise RuntimeError('static rotary buffer or configuration changed during forward')
    if not torch.equal(ids, before_ids) or not torch.equal(mask, before_mask):
        raise RuntimeError('private input IDs/mask mutated')
    if (type(output.logits) is not torch.Tensor or output.logits.device != device
            or output.logits.dtype != torch.float32 or tuple(output.logits.shape) != (batch, length, vocab)
            or not bool(torch.isfinite(output.logits).all()) or output.past_key_values is not None):
        raise RuntimeError('invalid full uncached same-device float32 logits')
    captures, meta = plan.captures, plan.metadata
    if len(captures) != count or len(meta['steps']) != count:
        raise RuntimeError('missing Llama capture')
    records = []
    for index, ((layer, role, path), row) in enumerate(zip(coordinates, meta['steps'], strict=True)):
        name = f'capture_{index}'
        value = captures[name]
        selector = [0] if role == 'attention' else []
        if (row['name'] != name or row['order'] != index or row['module_path'] != path
                or row['module_aliases'] != [path] or row['invocation'] != 0 or row['forward_index'] != 0
                or row['output_selector'] != selector or row['output_shape'] != [batch, length, h]
                or row['selected_shape'] != [batch, length, h] or row['batch_index'] is not None
                or row['token_index'] is not None or row['dtype'] != 'torch.float32'
                or row['device'] != str(device) or value.device != device or value.dtype != torch.float32
                or tuple(value.shape) != (batch, length, h) or not bool(torch.isfinite(value).all())):
            raise RuntimeError('Llama boundary capture differs from declared contract')
        event = TraceEvent(identity=identity, sequence=index, forward_index=0, module_path=path,
            module_kind=f'llama_{role}', layer_index=layer, invocation=0, output_path=tuple(selector),
            shape=(batch, length, h), dtype=str(value.dtype), device=str(device), capture_mode='direct',
            batch_positions=tuple(range(batch)), token_positions=tuple(range(length)), batch_axis=0, token_axis=1,
            generation=GenerationContext(stage='forward', use_cache=False), saved_shape=tuple(value.shape),
            saved_dtype=str(value.dtype), saved_device=str(device))
        records.append(TensorObservation(event, value))
    if meta['capture_bytes'] != count * size: raise RuntimeError('capture accounting mismatch')
    if device.type == 'cuda': torch.cuda.synchronize(device)
    return LlamaTraceResult(output, tuple(records), count * size)
