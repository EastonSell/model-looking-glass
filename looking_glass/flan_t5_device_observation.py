"""Bounded FLAN-T5 observations for a separately supervised owner.

The CPU entry is for synthetic checks. This module is not a launcher and does
not establish a CUDA lease, numerical policy, memory reservation or cleanup.
"""

from dataclasses import dataclass
from contextlib import nullcontext
from importlib.metadata import version
import json
import os
import platform
from threading import Lock
from weakref import WeakSet

import torch
from transformers import T5ForConditionalGeneration

from .flan_t5_equal_dose import hook_snapshot, require_clean_hooks
from .intervention_plan import ActivationPlan
from .device_lease import DeviceLease
from .memory_budget import MemoryBudget, MemoryLease
from .supervisor import OwnerSession

GPU_TRACE_CLAIM = 512 * 1024
HOST_TRACE_CLAIM = 2 * 1024 * 1024
_CUDA_PERMIT = object()
_CONSUMED_CLAIMS = WeakSet()
_CLAIM_LOCK = Lock()


class ObservationCustody:
    """Owner-retained roots for one forward, including unsuccessful forwards."""

    def __init__(self):
        self.entered = False
        self.input_ids = self.attention_mask = None
        self.decoder_ids = self.decoder_mask = None
        self.source_bits = self.source = self.plan = None
        self.plan_sources = ()
        self.plan_captures = self.output = self.captures = None
        self.host_result = None
        self.host_logits = self.host_state = self.host_before = self.host_after = None
        self.host_logits_bits = self.host_state_bits = self.host_before_bits = self.host_after_bits = None


class _RetainedCaptures(dict):
    def clear(self):
        # ActivationPlan clears failed captures; an owner must retain them
        # until it can establish device completion or discard the process.
        pass


@dataclass(frozen=True)
class Observation:
    logits: torch.Tensor
    state: torch.Tensor
    before: torch.Tensor | None
    after: torch.Tensor | None
    metadata: dict | None
    source_ids: tuple
    source_mask: tuple
    decoder_ids: tuple
    decoder_mask: tuple


def _model_contract(model, device):
    if type(model) is not T5ForConditionalGeneration:
        raise ValueError('expected a T5ForConditionalGeneration model')
    config = model.config
    if (config.is_encoder_decoder is not True or config.use_cache is not False
            or config._attn_implementation != 'eager'
            or config.pad_token_id != 0 or config.decoder_start_token_id != 0
            or config.eos_token_id != 1):
        raise ValueError('model is not the fixed eager, uncached T5 configuration')
    if any(module.training is not False for module in model.modules()):
        raise ValueError('all model modules must be in evaluation mode')
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled(device.type):
        raise ValueError('autocast must be disabled')
    if any(p.device != device or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('model parameters must be float32 on the execution device')
    if any(b.device != device or (b.is_floating_point() and b.dtype != torch.float32)
           for b in model.buffers()):
        raise ValueError('model buffers must match the execution device')


def _ids(source, target, vocab):
    if type(source) is not dict or type(target) is not dict:
        raise ValueError('source and target rows must be dictionaries')
    ids, mask, targets = source.get('input_ids'), source.get('attention_mask'), target.get('input_ids')
    if (type(ids) is not list or type(mask) is not list or type(targets) is not list
            or not 2 <= len(ids) <= 32 or not 2 <= len(targets) <= 8
            or len(mask) != len(ids) or any(type(x) is not int or x != 1 for x in mask)
            or any(type(x) is not int or not 1 <= x < vocab for x in ids + targets)
            or ids[-1] != 1 or ids.count(1) != 1
            or targets[-1] != 1 or targets.count(1) != 1):
        raise ValueError('source, mask or target violates the bounded EOS contract')
    return tuple(ids), tuple(mask), tuple(targets)


def _host_float32(tensor, shape, device, rooted, slot):
    if (type(tensor) is not torch.Tensor or tensor.device != device
            or tensor.dtype != torch.float32 or tensor.layout != torch.strided
            or not tensor.is_contiguous() or tuple(tensor.shape) != shape):
        raise ValueError('returned tensor differs from the bounded float32 contract')
    # Copy integer encodings first, preserving signed zeros and subnormals.
    # Root the copy before the post-call owner check can reject the attempt.
    bits = rooted(slot, lambda: tensor.detach().view(torch.int32).to('cpu', copy=True))
    return bits.contiguous().view(torch.float32)


def _same_bits(left, right):
    return left.shape == right.shape and torch.equal(left.view(torch.int32), right.view(torch.int32))


def _validate_metadata(metadata, *, action, token_index, length, hidden, device):
    names = ['before'] if action == 'capture' else ['before', 'action', 'after']
    operations = ['capture'] if action == 'capture' else ['capture', action, 'capture']
    steps = []
    for order, (name, operation) in enumerate(zip(names, operations)):
        selected = name == 'action' and token_index is not None
        steps.append({'name': name, 'operation': operation, 'order': order,
                      'module_path': 'encoder.final_layer_norm',
                      'module_aliases': ['encoder.final_layer_norm'],
                      'call_site_known': False, 'forward_index': 0, 'invocation': 0,
                      'batch_index': 0 if selected else None,
                      'token_index': token_index if selected else None,
                      'output_selector': [], 'output_shape': [1, length, hidden],
                      'selected_shape': [hidden] if selected else [1, length, hidden],
                      'dtype': 'torch.float32', 'device': device, 'source_capture': None,
                      'preceding_same_site_actions': names[:order]})
    expected = {'steps': steps, 'capture_bytes': length * hidden * 4 * (1 if action == 'capture' else 2),
                'source_bytes': (hidden if token_index is not None else length * hidden) * 4
                if action == 'patch' else 0}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if canonical(metadata) != canonical(expected):
        raise ValueError('intervention metadata differs from the requested coordinates')


def observe_cpu(model, source, target, *, action, token_index=None, value=None,
                custody, check_owner, synchronize):
    """Exercise the device path with an actual T5 model on synthetic CPU inputs."""
    return _observe(model, source, target, action=action, token_index=token_index, value=value,
                    custody=custody, check_owner=check_owner, synchronize=synchronize, device='cpu')


def _cuda_owner_check(session, device_lease, memory_lease):
    if (type(session) is not OwnerSession or type(device_lease) is not DeviceLease
            or type(memory_lease) is not MemoryLease):
        raise ValueError('an exact supervised owner and device/memory leases are required')
    OwnerSession.check(session, device_lease)
    budget = memory_lease._budget
    if (session.identity is None or session._runtime_verified is not True
            or type(budget) is not MemoryBudget
            or not any(item is memory_lease for item in budget.active_leases())
            or memory_lease.estimated_incremental_bytes <= 0
            or memory_lease.trace_bytes('gpu') < GPU_TRACE_CLAIM
            or memory_lease.trace_bytes('host') < HOST_TRACE_CLAIM):
        raise RuntimeError('supervised device identity or active memory claims are missing')
    OwnerSession.check(session, device_lease)


def _consume_observation_claim(memory_lease):
    """Permanently bind a lease object to at most one observation attempt."""
    if type(memory_lease) is not MemoryLease:
        raise ValueError('a dedicated observation memory lease is required')
    with _CLAIM_LOCK:
        if memory_lease in _CONSUMED_CLAIMS:
            raise RuntimeError('observation memory lease was already used')
        _CONSUMED_CLAIMS.add(memory_lease)


def _cuda_numeric_state():
    if (os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
            or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
            or int((torch.tensor([2.0 ** -149], dtype=torch.float32, device='cpu') * 1.0).view(torch.int32)[0]) != 1):
        raise RuntimeError('CUDA numerical settings differ from the pinned contract')


def _cuda_numerics():
    versions = {'python': platform.python_version(), 'torch': str(torch.__version__),
                'cuda_build': str(torch.version.cuda), 'transformers': version('transformers'),
                'tokenizers': version('tokenizers'), 'safetensors': version('safetensors'),
                'numpy': version('numpy')}
    expected = {'python': '3.11.15', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
                'transformers': '4.57.3', 'tokenizers': '0.22.2', 'safetensors': '0.8.0',
                'numpy': '2.4.6'}
    if versions != expected:
        raise RuntimeError('CUDA numerical versions differ from the pinned contract')
    _cuda_numeric_state()
    return versions


def _record_cuda_stage(cuda, check_owner, stage, record_stage):
    """Record an obtained synchronized endpoint before checking its counters."""
    def guarded(function, *args):
        if check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        result = function(*args)
        if check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        return result
    guarded(cuda.synchronize, 0)
    free, total = guarded(cuda.mem_get_info, 0)
    snapshot = {'device_free_bytes': free, 'device_total_bytes': total,
                'allocated_bytes': guarded(cuda.memory_allocated, 0),
                'reserved_bytes': guarded(cuda.memory_reserved, 0),
                'peak_allocated_bytes': guarded(cuda.max_memory_allocated, 0),
                'peak_reserved_bytes': guarded(cuda.max_memory_reserved, 0)}
    recorded = snapshot.copy()
    if guarded(record_stage, stage, recorded) is not None:
        raise RuntimeError('memory recorder must return None or raise on failure')
    if (set(recorded) != set(snapshot) or any(type(key) is not str for key in recorded)
            or any(type(recorded[key]) is not int or recorded[key] != snapshot[key] for key in snapshot)):
        raise ValueError('memory recorder changed the obtained snapshot')
    if (any(type(value) is not int or value < 0 for value in snapshot.values())
            or total == 0 or free > total or snapshot['allocated_bytes'] > snapshot['reserved_bytes']
            or snapshot['reserved_bytes'] > total
            or snapshot['peak_allocated_bytes'] < snapshot['allocated_bytes']
            or snapshot['peak_reserved_bytes'] < snapshot['reserved_bytes']
            or snapshot['peak_allocated_bytes'] > snapshot['peak_reserved_bytes']
            or snapshot['peak_reserved_bytes'] > total):
        raise ValueError('invalid synchronized CUDA memory counters')
    return snapshot


def observe_cuda_owned(model, source, target, *, action, token_index=None, value=None,
                       custody, session, device_lease, memory_lease, record_stage):
    """Observe under a verified owner; this does not launch or attest the owner.

    The caller must hold its physical lease and model/allocator commitment,
    retain custody on every failure, and record partial evidence and cleanup.
    """
    if not callable(record_stage):
        raise TypeError('a trusted stage recorder is required')
    def check_owner():
        _cuda_owner_check(session, device_lease, memory_lease)
        _cuda_numeric_state()
    check_owner()
    _cuda_numerics()
    if (type(model) is not T5ForConditionalGeneration or model.config.d_model != 512
            or model.config.vocab_size != 32128 or model.config.num_layers != 6
            or model.config.num_decoder_layers != 6):
        raise ValueError('model dimensions differ from the pinned FLAN-T5 profile')
    _consume_observation_claim(memory_lease)
    check_owner()
    def synchronize(stage):
        _record_cuda_stage(torch.cuda, check_owner, stage, record_stage)
    return _observe(model, source, target, action=action, token_index=token_index, value=value,
                    custody=custody, check_owner=check_owner, synchronize=synchronize, device='cuda:0',
                    cuda_permit=_CUDA_PERMIT)


def _observe(model, source, target, *, action, token_index, value, custody,
             check_owner, synchronize, device, cuda_permit=None):
    if device not in ('cpu', 'cuda:0') or (device == 'cuda:0' and cuda_permit is not _CUDA_PERMIT):
        raise RuntimeError('CUDA observations require the supervised entry')
    if type(custody) is not ObservationCustody or custody.entered:
        raise ValueError('a fresh observation custody object is required')
    if not callable(check_owner) or not callable(synchronize):
        raise TypeError('owner and synchronization callbacks are required')
    custody.entered = True
    execution_device = torch.device(device)

    def require_owner():
        if check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')

    def guarded(function):
        require_owner()
        result = function()
        require_owner()
        return result

    def rooted(name, function):
        require_owner()
        result = function()
        setattr(custody, name, result)
        require_owner()
        return result

    require_owner()
    _model_contract(model, execution_device)
    require_clean_hooks(hook_snapshot(model))
    vocab, hidden = model.config.vocab_size, model.config.d_model
    if type(vocab) is not int or type(hidden) is not int or vocab <= 1 or hidden <= 0:
        raise ValueError('invalid model dimensions')
    ids, mask, targets = _ids(source, target, vocab)
    decoder_ids, decoder_mask = (0,) + targets[:-1], (1,) * len(targets)
    if action not in ('clean', 'capture', 'patch', 'null', 'zero'):
        raise ValueError('unsupported fixed action')
    if token_index is not None and (type(token_index) is not int or not 0 <= token_index < len(ids)):
        raise ValueError('invalid token coordinate')
    if action in ('clean', 'capture') and token_index is not None:
        raise ValueError('clean and capture observe the complete encoder state')
    selected_shape = (1, len(ids), hidden) if token_index is None else (hidden,)
    source_snapshot = None
    if action == 'patch':
        if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                or value.layout != torch.strided or value.dtype != torch.float32
                or tuple(value.shape) != selected_shape or not bool(torch.isfinite(value).all())):
            raise ValueError('patch source must be a finite bounded CPU float32 tensor')
        source_snapshot = value.detach().view(torch.int32).contiguous().clone()
    elif value is not None:
        raise ValueError('this action cannot accept a patch source')

    cpu_inputs = [torch.tensor([values], dtype=torch.long, device='cpu') for values in
                  (ids, mask, decoder_ids, decoder_mask)]
    names = ('input_ids', 'attention_mask', 'decoder_ids', 'decoder_mask')
    device_inputs = [rooted(name, lambda tensor=tensor: tensor.to(execution_device, copy=True))
                     for name, tensor in zip(names, cpu_inputs)]
    device_source_bits = None
    if source_snapshot is not None:
        device_source_bits = rooted('source_bits', lambda: source_snapshot.to(execution_device, copy=True))
        device_source = rooted('source', lambda: device_source_bits.view(torch.float32))
    else:
        device_source = None

    operation = steps = plan = None
    if action != 'clean':
        site = {'module_path': 'encoder.final_layer_norm', 'invocation': 0}
        steps = [dict(name='before', operation='capture', **site)]
        if action != 'capture':
            selected = dict(site)
            if token_index is not None:
                selected.update(batch_index=0, token_index=token_index)
            operation = dict(name='action', operation=action, **selected)
            if action == 'patch':
                operation['value'] = device_source
            steps.extend((operation, dict(name='after', operation='capture', **site)))
        require_owner()
        plan = ActivationPlan.__new__(ActivationPlan)
        custody.plan = plan
        full_bytes = len(ids) * hidden * 4
        ActivationPlan.__init__(plan, model, steps, max_tensor_bytes=full_bytes,
                                max_capture_bytes=2 * full_bytes, max_source_bytes=full_bytes,
                                max_steps=3, max_invocations=1)
        custody.plan_sources = tuple(plan._sources.values())
        custody.plan_captures = _RetainedCaptures()
        plan._captures = custody.plan_captures
        require_owner()

    with torch.inference_mode(), (plan if plan is not None else nullcontext()):
        _model_contract(model, execution_device)
        out = rooted('output', lambda: model(input_ids=device_inputs[0], attention_mask=device_inputs[1],
                     decoder_input_ids=device_inputs[2], decoder_attention_mask=device_inputs[3],
                     use_cache=False, return_dict=True))
        _model_contract(model, execution_device)
        guarded(lambda: synchronize('output_live'))

    captures, metadata = (None, None) if plan is None else (plan.captures, plan.metadata)
    custody.captures = captures
    require_clean_hooks(hook_snapshot(model))
    if metadata is not None:
        _validate_metadata(metadata, action=action, token_index=token_index, length=len(ids),
                           hidden=hidden, device=str(execution_device))
    for tensor, expected in zip(device_inputs, cpu_inputs):
        host = guarded(lambda tensor=tensor: tensor.to('cpu', copy=True))
        if not torch.equal(host, expected):
            raise ValueError('model input or mask changed')
    if getattr(out, 'past_key_values', None) is not None:
        raise ValueError('cached model output is not supported')
    logits = _host_float32(out.logits, (1, len(targets), vocab), execution_device, rooted, 'host_logits_bits')
    custody.host_logits = logits
    state = _host_float32(out.encoder_last_hidden_state, (1, len(ids), hidden), execution_device, rooted,
                          'host_state_bits')
    custody.host_state = state
    before = None if captures is None else _host_float32(captures['before'], (1, len(ids), hidden),
                                                          execution_device, rooted, 'host_before_bits')
    custody.host_before = before
    after = None if action in ('clean', 'capture') else _host_float32(captures['after'],
                                      (1, len(ids), hidden), execution_device, rooted, 'host_after_bits')
    custody.host_after = after
    outputs = (logits, state) + (() if before is None else (before,)) + (() if after is None else (after,))
    if not all(bool(torch.isfinite(t).all()) for t in outputs):
        raise ValueError('nonfinite output or capture')
    expected_state = before if after is None else after
    if expected_state is not None and not _same_bits(state, expected_state):
        raise ValueError('captured encoder state differs from root output')
    if action == 'patch':
        delivered = after if token_index is None else after[0, token_index]
        if not torch.equal(delivered.view(torch.int32), source_snapshot):
            raise ValueError('patch source did not land exactly')
    elif action == 'null' and not _same_bits(before, after):
        raise ValueError('null action changed the encoder state')
    elif action == 'zero':
        delivered = after if token_index is None else after[0, token_index]
        if not torch.equal(delivered.view(torch.int32), torch.zeros_like(delivered, dtype=torch.int32)):
            raise ValueError('zero action did not land exactly')
    if token_index is not None and action != 'capture':
        for index in range(len(ids)):
            if index != token_index and not _same_bits(before[0, index], after[0, index]):
                raise ValueError('an unselected encoder position changed')
    guarded(lambda: synchronize('captures_live'))
    result = Observation(logits, state, before, after, metadata, ids, mask, decoder_ids, decoder_mask)
    custody.host_result = result
    custody.input_ids = custody.attention_mask = custody.decoder_ids = custody.decoder_mask = None
    custody.source_bits = custody.source = custody.plan = custody.output = custody.captures = None
    custody.plan_sources = ()
    custody.plan_captures = None
    device_inputs = device_source = device_source_bits = plan = out = captures = None
    steps = operation = tensor = None
    guarded(lambda: synchronize('device_refs_disposed'))
    return result
