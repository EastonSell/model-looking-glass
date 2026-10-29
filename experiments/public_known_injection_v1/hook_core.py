"""Ordered GPT-2 observation/intervention primitives for the public study."""
import math
import struct

import torch

from looking_glass.intervention_plan import ActivationPlan
from looking_glass.trace_schema import GenerationContext
from looking_glass.transformer_trace import GPT2TraceSession


SITES = (1, 2, 3, 4)


class UndeliveredInjection(ValueError):
    """The correctly rounded addition did not change the target float32 bits."""


def bitwise_equal(a, b):
    return (a.dtype == b.dtype == torch.float32 and a.shape == b.shape
            and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def _positioned(layer, token):
    return {'module_path': f'transformer.h.{layer}.mlp', 'batch_index': 0,
            'token_index': token}


def _full(layer):
    return {'module_path': f'transformer.h.{layer}.mlp'}


def _preserve_returned(output, on_partial, error):
    if output is not None and on_partial is not None:
        try:
            on_partial(output.logits[0, -1].detach(), error)
        except BaseException as save_error:
            if hasattr(error, 'add_note'):
                error.add_note('partial returned-logit save failed: ' + type(save_error).__name__)


def observe_clean(model, tokens, run_id, on_partial=None):
    token = tokens.shape[1] - 1
    session = GPT2TraceSession(model, run_id=run_id, targets=('mlp',), layers=SITES,
                               samples=(0,), tokens=(token,), mode='direct',
                               max_events=4, max_trace_bytes=1_000_000,
                               capture_device='cpu',
                               forward_context=GenerationContext(stage='forward', use_cache=False))
    output = None
    try:
        with session:
            output = model(input_ids=tokens, use_cache=False)
        if not session.successful or len(session.events) != 4:
            raise RuntimeError('incomplete clean trace')
        vectors = {event.layer_index: event.value[0, 0].detach().clone() for event in session.events}
        if tuple(vectors) != SITES:
            raise RuntimeError('clean trace order differs from fixed sites')
        return {'logits': output.logits[0, -1].detach().clone(), 'vectors': vectors,
                'events': [event.metadata() for event in session.events]}
    except BaseException as error:
        _preserve_returned(output, on_partial, error)
        raise


def perform_action(model, tokens, clean, injected_site, action, patch_site=None, on_partial=None):
    if injected_site not in (1, 4) or action not in ('injected', 'patch', 'self'):
        raise ValueError('unsupported injection/action')
    if (action == 'patch') != (patch_site in SITES):
        raise ValueError('patch action requires one fixed candidate site')
    token = tokens.shape[1] - 1
    delta = torch.zeros_like(clean[1])
    delta[0] = 4.0
    steps = []
    for layer in SITES:
        position = _positioned(layer, token)
        full = _full(layer)
        if layer == injected_site:
            steps.extend([dict(full, name='inject_before', operation='capture'),
                          dict(position, name='inject', operation='add', value=delta),
                          dict(full, name='inject_after', operation='capture')])
        if action == 'patch' and layer == patch_site:
            steps.extend([dict(full, name='patch_before', operation='capture'),
                          dict(position, name='patch', operation='patch', value=clean[layer]),
                          dict(full, name='patch_after', operation='capture')])
        if action == 'self' and layer == injected_site:
            steps.extend([dict(position, name='self_source', operation='capture'),
                          dict(full, name='self_before', operation='capture'),
                          dict(position, name='self_patch', operation='patch',
                               source_capture='self_source'),
                          dict(full, name='self_after', operation='capture')])
        # At the injection site this is strictly after the addition; any
        # same-site candidate/self patch is also complete before this capture.
        steps.append(dict(position, name=f'tap_{layer}', operation='capture'))
    plan = ActivationPlan(model, steps, max_tensor_bytes=1_000_000,
                          max_capture_bytes=4_000_000, max_source_bytes=1_000_000,
                          max_invocations=1)
    output = None
    try:
        with plan:
            output = model(input_ids=tokens, use_cache=False)
        return {'logits': output.logits[0, -1].detach().clone(),
                'captures': plan.captures, 'events': plan.metadata['steps']}
    except BaseException as error:
        _preserve_returned(output, on_partial, error)
        raise


def perform_null_self(model, tokens, on_partial=None):
    token = tokens.shape[1] - 1
    position, full = _positioned(1, token), _full(1)
    steps = [dict(full, name='before', operation='capture'),
             dict(position, name='null', operation='null'),
             dict(position, name='self_source', operation='capture'),
             dict(position, name='self_patch', operation='patch', source_capture='self_source'),
             dict(full, name='after', operation='capture')]
    plan = ActivationPlan(model, steps, max_tensor_bytes=1_000_000,
                          max_capture_bytes=4_000_000, max_source_bytes=1_000_000,
                          max_invocations=1)
    output = None
    try:
        with plan:
            output = model(input_ids=tokens, use_cache=False)
        return {'logits': output.logits[0, -1].detach().clone(),
                'captures': plan.captures, 'events': plan.metadata['steps']}
    except BaseException as error:
        _preserve_returned(output, on_partial, error)
        raise


def check_landing(result, action, clean=None, patch_site=None):
    """Validate saved full-tensor bits after the caller has persisted them."""
    captures = result['captures']
    before, after = captures['inject_before'], captures['inject_after']
    if before.dtype != torch.float32 or after.shape != before.shape or before.ndim != 3:
        raise ValueError('invalid injection witness shape or dtype')
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('nonfinite injection witness')
    expected = before.clone()
    value = float(before[0, -1, 0]) + 4.0
    # Python binary64 is exact for a binary32 value plus 4; pack performs the
    # single required binary32 rounding and rejects overflow.
    packed = struct.pack('<f', value)
    rounded = struct.unpack('<f', packed)[0]
    if not math.isfinite(rounded):
        raise ValueError('nonfinite injected coordinate')
    expected[0, -1, 0] = rounded
    if not bitwise_equal(expected, after):
        raise ValueError('injection landed differently or changed an untouched bit')
    if packed == struct.pack('<f', float(before[0, -1, 0])):
        raise UndeliveredInjection('rounded injection left the target coordinate unchanged')
    if action == 'patch':
        patch_before, patch_after = captures['patch_before'], captures['patch_after']
        expected_patch = patch_before.clone()
        expected_patch[0, -1] = clean[patch_site]
        if not bitwise_equal(expected_patch, patch_after):
            raise ValueError('candidate patch landed differently or changed an untouched bit')
    if action == 'self' and not bitwise_equal(captures['self_before'], captures['self_after']):
        raise ValueError('self patch changed a bit')
