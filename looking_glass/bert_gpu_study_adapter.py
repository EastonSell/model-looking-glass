"""CPU-tested prototype for the fixed BERT family observer; CUDA is disabled.

This private prototype does not load a checkpoint or own a device. Before it
can execute on CUDA, a separately reviewed owner must bind the full family
schedule, model/device lease, resource reservations, output custody, and
failure/quarantine lifecycle. No current entry point enables CUDA execution.
"""

import json

import torch
from transformers import DistilBertForMaskedLM
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention

from .bert_agreement_runner import Observation, _has_hooks, decode_float32_bits, float32_bytes
from .intervention_plan import ActivationPlan


class _ObservationCustody:
    """Preallocated prototype roots; an external owner must retain this object."""
    __slots__ = ('entered', 'device_input', 'attention_mask', 'device_source_bits',
                 'device_source', 'plan', 'plan_source_roots', 'logits', 'captures',
                 'capture_before', 'capture_after')

    def __init__(self):
        self.entered = False
        self.device_input = self.attention_mask = self.device_source_bits = None
        self.device_source = self.plan = self.logits = self.captures = None
        self.capture_before = self.capture_after = None
        self.plan_source_roots = ()


def run_cuda_observation(*args, **kwargs):
    """The study's CUDA observer is unavailable until its owner is reviewed."""
    raise RuntimeError('BERT GPU study execution is disabled pending full-family calibration')


def _observe_cpu_prototype(model, input_ids, *, layer, token, mask_token, is_id, are_id,
                           source=None, self_patch=False, check_owner, synchronize, custody=None):
    """Exercise the prospective owned-observation path only on synthetic CPU inputs."""
    return _observe_owned(model, input_ids, layer=layer, token=token, mask_token=mask_token,
                          is_id=is_id, are_id=are_id, source=source, self_patch=self_patch,
                          device='cpu', check_owner=check_owner, synchronize=synchronize,
                          custody=_ObservationCustody() if custody is None else custody)


def _observe_owned(model, input_ids, *, layer, token, mask_token, is_id, are_id,
                   source, self_patch, device, check_owner, synchronize, custody):
    # A future CUDA owner must replace this gate only after reviewing and
    # calibrating the complete 140-forward family lifecycle.
    if type(device) is not str or device != 'cpu':
        raise RuntimeError('BERT GPU study execution is disabled pending full-family calibration')
    if not callable(check_owner) or not callable(synchronize):
        raise TypeError('trusted owner-check and synchronization callbacks are required')
    if type(custody) is not _ObservationCustody or custody.entered:
        raise ValueError('a fresh observation custody object is required')
    custody.entered = True

    def guarded(function):
        check_owner()
        value = function()
        check_owner()
        return value

    def rooted(name, function):
        check_owner()
        value = function()
        setattr(custody, name, value)
        check_owner()
        return value

    def completion(stage):
        guarded(lambda: synchronize(stage))

    check_owner()
    if type(model) is not DistilBertForMaskedLM or any(module.training is not False for module in model.modules()):
        raise ValueError('expected an evaluation-mode DistilBERT masked language model')
    if _has_hooks(model):
        raise RuntimeError('preexisting model or global hooks are unsupported')
    if (type(input_ids) is not torch.Tensor or input_ids.device.type != 'cpu'
            or input_ids.dtype != torch.int64 or tuple(input_ids.shape) != (1, 10)):
        raise ValueError('expected one CPU int64 input row of length ten')
    execution_device = torch.device(device)
    def check_configuration():
        if any(module.training is not False for module in model.modules()):
            raise RuntimeError('every model module must remain in evaluation mode')
        if (getattr(model.config, '_attn_implementation', None) != 'eager'
                or any(type(block.attention) is not MultiHeadSelfAttention
                       for block in model.distilbert.transformer.layer)
                or torch.is_autocast_enabled('cpu')
                or torch.is_autocast_enabled(execution_device.type)):
            raise RuntimeError('eager attention and disabled autocast are required')
    check_configuration()
    if (any(parameter.device != execution_device or parameter.dtype != torch.float32
            for parameter in model.parameters())
            or any(buffer.device != execution_device for buffer in model.buffers())):
        raise ValueError('model parameters and buffers must match the execution device')
    vocab, hidden = model.config.vocab_size, model.config.dim
    if (len(model.distilbert.transformer.layer) != 6 or type(layer) is not int
            or not 0 <= layer < 6 or type(token) is not int or not 0 <= token < 10
            or type(mask_token) is not int or not 0 <= mask_token < 10
            or type(is_id) is not int or type(are_id) is not int or is_id == are_id
            or not 0 <= is_id < vocab or not 0 <= are_id < vocab
            or bool((input_ids < 0).any()) or bool((input_ids >= vocab).any())):
        raise ValueError('invalid fixed-study tensor or vocabulary coordinates')
    if type(self_patch) is not bool or (self_patch and source is not None):
        raise ValueError('self and external patch sources are mutually exclusive')
    if source is not None and (type(source) is not torch.Tensor or source.device.type != 'cpu'
            or source.dtype != torch.float32 or tuple(source.shape) != (hidden,)
            or not bool(torch.isfinite(source).all())):
        raise ValueError('source must be a finite CPU float32 hidden vector')

    input_snapshot = tuple(input_ids[0].tolist())
    source_snapshot = None if source is None else float32_bytes(source)
    # Reinterpret before any copy to preserve float32 signed zeros/subnormals.
    device_input = rooted('device_input', lambda: input_ids.detach().to(execution_device, copy=True))
    attention_mask = rooted('attention_mask', lambda: torch.ones_like(device_input))
    if source is None:
        device_source = None
    else:
        source_bits = source.detach().view(torch.int32).contiguous()
        device_source_bits = rooted('device_source_bits', lambda: source_bits.to(execution_device, copy=True))
        device_source = rooted('device_source', lambda: device_source_bits.view(torch.float32))
    observed_ids = guarded(lambda: tuple(device_input[0].tolist()))
    observed_mask = guarded(lambda: tuple(attention_mask[0].tolist()))
    if observed_ids != input_snapshot or observed_mask != (1,) * 10:
        raise RuntimeError('model input transfer changed the declared values')

    path = f'distilbert.transformer.layer.{layer}'
    site = {'module_path': path, 'invocation': 0, 'batch_index': 0, 'token_index': token}
    steps = [{'name': 'before', 'operation': 'capture', **site}]
    if self_patch or device_source is not None:
        action = {'name': 'patch', 'operation': 'patch', **site}
        action['source_capture' if self_patch else 'value'] = 'before' if self_patch else device_source
        steps.extend((action, {'name': 'after', 'operation': 'capture', **site}))
        del action
    # Root the plan before its constructor can allocate a cloned source. Save
    # those exact clones separately before entry, because plan cleanup clears
    # its own source dictionary even when synchronization has failed.
    check_owner()
    plan = ActivationPlan.__new__(ActivationPlan)
    custody.plan = plan
    ActivationPlan.__init__(plan, model, steps, max_tensor_bytes=10 * hidden * 4,
        max_capture_bytes=2 * hidden * 4, max_source_bytes=hidden * 4,
        max_steps=3, max_invocations=1)
    custody.plan_source_roots = tuple(plan._sources.values())
    check_owner()

    def host_float32(tensor):
        # Copies are made through int32, avoiding float arithmetic on stored bits.
        detached = guarded(lambda: tensor.detach())
        integer_view = guarded(lambda: detached.view(torch.int32))
        host_bits = guarded(lambda: integer_view.to('cpu', copy=True))
        return host_bits.contiguous().view(torch.float32)

    with torch.inference_mode(), plan:
        try:
            check_owner()
            check_configuration()
            logits = model(input_ids=device_input, attention_mask=attention_mask,
                           output_attentions=False, output_hidden_states=False, return_dict=True).logits
            custody.logits = logits
            check_owner()
            check_configuration()
            if (guarded(lambda: tuple(device_input[0].tolist())) != observed_ids
                    or guarded(lambda: tuple(attention_mask[0].tolist())) != observed_mask):
                raise RuntimeError('model input tensors changed during execution')
            if (type(logits) is not torch.Tensor or logits.dtype != torch.float32
                    or logits.device != execution_device or tuple(logits.shape) != (1, 10, vocab)
                    or not guarded(lambda: bool(torch.isfinite(logits).all()))):
                raise RuntimeError('model returned invalid or nonfinite logits')
            mask_logits = host_float32(guarded(lambda: logits[0, mask_token]))
            completion('output_live')
        finally:
            # Plan exit can clear captures on an error, so retain any completed
            # device captures before it runs. A failed owner must retain the
            # custody object until its dedicated process has terminated.
            custody.capture_before = plan._captures.get('before')
            custody.capture_after = plan._captures.get('after')

    captures, metadata = plan.captures, plan.metadata
    custody.captures = captures
    if _has_hooks(model):
        raise RuntimeError('hook cleanup is uncertain; discard the owner')
    names = ['before', 'patch', 'after'] if len(steps) == 3 else ['before']
    ops = ['capture', 'patch', 'capture'] if len(steps) == 3 else ['capture']
    expected_events = []
    for index, (name, operation) in enumerate(zip(names, ops)):
        expected_events.append({'name': name, 'operation': operation, 'order': index,
            'module_path': path, 'module_aliases': [path], 'call_site_known': False,
            'forward_index': 0, 'invocation': 0, 'batch_index': 0, 'token_index': token,
            'output_selector': [0], 'output_shape': [1, 10, hidden], 'selected_shape': [hidden],
            'dtype': 'torch.float32', 'device': device,
            'source_capture': 'before' if self_patch and index == 1 else None,
            'preceding_same_site_actions': names[:index]})
    expected_metadata = {'steps': expected_events, 'capture_bytes': hidden * 4 * (2 if len(steps) == 3 else 1),
                         'source_bytes': hidden * 4 if source is not None else 0}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if canonical(metadata) != canonical(expected_metadata):
        raise RuntimeError('recorded action metadata differs from the requested site')
    before = host_float32(captures['before'])
    after = None if len(steps) == 1 else host_float32(captures['after'])
    if (not bool(torch.isfinite(before).all())
            or (after is not None and not bool(torch.isfinite(after).all()))):
        raise RuntimeError('intervention captured nonfinite values')
    if after is not None:
        expected = float32_bytes(before) if self_patch else source_snapshot
        if float32_bytes(after) != expected:
            raise RuntimeError('replacement bits did not land at the selected site')
    bits = mask_logits.view(torch.int32)
    is_bits, are_bits = int(bits[is_id]) & 0xffffffff, int(bits[are_id]) & 0xffffffff
    is_logit, are_logit = decode_float32_bits(is_bits), decode_float32_bits(are_bits)
    completion('captures_live')
    del logits, captures, plan, steps, device_source, device_input, attention_mask
    custody.logits = custody.captures = custody.plan = None
    custody.device_input = custody.attention_mask = custody.device_source = None
    custody.device_source_bits = None
    custody.plan_source_roots = ()
    custody.capture_before = custody.capture_after = None
    if source is not None:
        del device_source_bits
    completion('device_refs_disposed')
    return Observation(mask_logits, before, after, is_bits, are_bits, is_logit, are_logit,
                       tuple(metadata['steps']), metadata['capture_bytes'], metadata['source_bytes'],
                       observed_ids, observed_mask)
