"""Bounded same-forward early intervention and later DistilBERT captures."""
from dataclasses import dataclass

import torch

from looking_glass.intervention_plan import ActivationPlan
from scripts import bert_taste_patch_v1 as previous

LATE_POSITIONS = {'late_ingredient': 7, 'late_the': 1, 'late_tasted': 3, 'late_mask': 4}


@dataclass(frozen=True)
class Observation:
    mask_logits: torch.Tensor
    before: torch.Tensor
    after: object
    full_before_sha256: str
    full_after_sha256: object
    untouched_positions_equal: object
    later: dict
    late_full_sha256: object
    model_entry: dict
    events: tuple


def _same(left, right):
    return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def observe(model, ids, *, layer, position, mask_index, source=None, self_patch=False, capture_late=False):
    """Observe one CPU forward and optionally capture later block-3 positions."""
    if (not isinstance(model, torch.nn.Module) or any(module.training for module in model.modules())
            or previous.has_hooks(model) or 'forward' in model.__dict__):
        raise ValueError('a hook-free evaluation model with its original forward is required')
    if (type(ids) is not tuple or len(ids) != 10 or any(type(item) is not int for item in ids)
            or type(layer) is not int or not 0 <= layer < len(model.distilbert.transformer.layer)
            or type(position) is not int or not 0 <= position < 10
            or type(mask_index) is not int or not 0 <= mask_index < 10
            or type(self_patch) is not bool or type(capture_late) is not bool
            or (capture_late and layer != 0) or (self_patch and source is not None)):
        raise ValueError('invalid input, layer or action')
    if (any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32 for parameter in model.parameters())
            or any(buffer.device.type != 'cpu' for buffer in model.buffers())):
        raise ValueError('model must be CPU float32')
    hidden, vocab = model.config.dim, model.config.vocab_size
    if any(item < 0 or item >= vocab for item in ids):
        raise ValueError('token is outside vocabulary')
    if source is not None and (type(source) is not torch.Tensor or source.device.type != 'cpu'
                               or source.dtype != torch.float32 or tuple(source.shape) != (hidden,)
                               or not bool(torch.isfinite(source).all())):
        raise ValueError('invalid source vector')

    path = f'distilbert.transformer.layer.{layer}'
    whole = {'module_path': path, 'invocation': 0}
    selected = {**whole, 'batch_index': 0, 'token_index': position}
    steps = [{'name': 'full_before', 'operation': 'capture', **whole},
             {'name': 'before', 'operation': 'capture', **selected}]
    patched = self_patch or source is not None
    if patched:
        action = {'name': 'patch', 'operation': 'patch', **selected}
        action['source_capture' if self_patch else 'value'] = 'before' if self_patch else source
        steps.extend((action, {'name': 'after', 'operation': 'capture', **selected},
                      {'name': 'full_after', 'operation': 'capture', **whole}))
    if capture_late:
        late = {'module_path': 'distilbert.transformer.layer.3', 'invocation': 0}
        steps.append({'name': 'late_full', 'operation': 'capture', **late})
        steps.extend({'name': name, 'operation': 'capture', **late, 'batch_index': 0, 'token_index': index}
                     for name, index in LATE_POSITIONS.items())
    plan = ActivationPlan(model, steps, max_tensor_bytes=10 * hidden * 4,
                          max_capture_bytes=40 * hidden * 4, max_source_bytes=hidden * 4,
                          max_steps=10, max_invocations=1)
    inputs = torch.tensor([ids], dtype=torch.int64)
    attention = torch.ones_like(inputs)
    entries = []
    original = model.forward

    def watched(*args, **kwargs):
        if args or entries or set(kwargs) != {'input_ids', 'attention_mask', 'output_attentions',
                                              'output_hidden_states', 'return_dict'}:
            raise RuntimeError('model-entry arguments differ')
        observed_input, observed_mask = kwargs['input_ids'], kwargs['attention_mask']
        if (type(observed_input) is not torch.Tensor or type(observed_mask) is not torch.Tensor
                or observed_input.dtype != torch.int64 or observed_mask.dtype != torch.int64
                or observed_input.device.type != 'cpu' or observed_mask.device.type != 'cpu'
                or tuple(observed_input.shape) != (1, 10) or tuple(observed_mask.shape) != (1, 10)
                or tuple(observed_input[0].tolist()) != ids or tuple(observed_mask[0].tolist()) != (1,) * 10
                or kwargs['output_attentions'] is not False or kwargs['output_hidden_states'] is not False
                or kwargs['return_dict'] is not True):
            raise RuntimeError('actual model entry differs')
        entries.append({'input_ids': observed_input[0].tolist(), 'attention_mask': observed_mask[0].tolist()})
        return original(*args, **kwargs)

    primary = None
    try:
        model.forward = watched
        with torch.inference_mode(), plan:
            logits = model(input_ids=inputs, attention_mask=attention, output_attentions=False,
                           output_hidden_states=False, return_dict=True).logits
            if (tuple(inputs[0].tolist()) != ids or tuple(attention[0].tolist()) != (1,) * 10
                    or type(logits) is not torch.Tensor or tuple(logits.shape) != (1, 10, vocab)
                    or logits.dtype != torch.float32 or logits.device.type != 'cpu'
                    or not bool(torch.isfinite(logits).all())):
                raise RuntimeError('model input or output is invalid')
            mask_logits = logits[0, mask_index].detach().view(torch.int32).clone().view(torch.float32)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            if model.__dict__.get('forward') is watched:
                del model.forward
            elif 'forward' in model.__dict__:
                raise RuntimeError('model forward cleanup is uncertain')
        except BaseException as cleanup:
            if primary is None:
                raise
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(
                    primary, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
    if previous.has_hooks(model) or len(entries) != 1:
        raise RuntimeError('model cleanup or entry count is uncertain')

    captures, events = plan.captures, plan.metadata['steps']
    if len(events) != len(steps):
        raise RuntimeError('capture and intervention event count differs')
    prior = {}
    for index, (event, step) in enumerate(zip(events, steps)):
        module_path = step['module_path']
        expected_prior = prior.setdefault(module_path, [])
        if (event['name'] != step['name'] or event['operation'] != step['operation']
                or event['order'] != index or event['module_path'] != module_path
                or event['module_aliases'] != [module_path] or event['forward_index'] != 0
                or event['invocation'] != 0 or event['batch_index'] != step.get('batch_index')
                or event['token_index'] != step.get('token_index') or event['output_selector'] != [0]
                or event['output_shape'] != [1, 10, hidden]
                or event['selected_shape'] != ([hidden] if 'token_index' in step else [1, 10, hidden])
                or event['dtype'] != 'torch.float32' or event['device'] != 'cpu'
                or event['source_capture'] != step.get('source_capture')
                or event['preceding_same_site_actions'] != expected_prior):
            raise RuntimeError('observed action or capture coordinates differ')
        expected_prior.append(step['name'])
    full, before = captures['full_before'], captures['before']
    if not _same(full[0, position], before):
        raise RuntimeError('selected pre-action bits differ from the full output')
    after = captures.get('after')
    full_after = captures.get('full_after')
    untouched = None
    if patched:
        wanted = before if self_patch else source
        expected = full.contiguous().view(torch.int32).clone()
        expected[0, position] = wanted.contiguous().view(torch.int32)
        untouched = torch.equal(expected, full_after.contiguous().view(torch.int32))
        if not untouched or not _same(after, wanted):
            raise RuntimeError('patch failed its full-output bit witness')
    later = {}
    late_full_hash = None
    if capture_late:
        late_full = captures['late_full']
        for name, index in LATE_POSITIONS.items():
            later[name] = captures[name]
            if not _same(late_full[0, index], later[name]):
                raise RuntimeError('later selected bits differ from their full output')
        late_full_hash = previous._sha_tensor(late_full)
    return Observation(mask_logits, before, after, previous._sha_tensor(full),
                       None if full_after is None else previous._sha_tensor(full_after),
                       untouched, later, late_full_hash, entries[0], tuple(events))
