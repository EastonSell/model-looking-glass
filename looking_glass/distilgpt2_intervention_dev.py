"""Development-only CPU comparison of ordered intervention with a manual forward."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import torch

from .intervention_plan import ActivationPlan


PROTOCOL = "distilgpt2-intervention-dev-v1"
INPUT_IDS = (50256, 50256, 50256, 50256)
SITE = "transformer.h.2"
TOKEN = 2
ATOL = 1e-6
MIN_EFFECT = 1e-5
SEED = 1729


class SmokeFailure(RuntimeError):
    """A planned control or numerical comparison did not pass."""


def _bits(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
            or tensor.device.type != 'cpu' or sys.byteorder != 'little'):
        raise SmokeFailure('expected a little-endian CPU float32 tensor')
    # Reinterpret first, then copy as integers: an FTZ-sensitive floating
    # copy must not alter stored subnormal bit patterns.
    raw = tensor.detach().view(torch.int32).contiguous().numpy().reshape(-1).view(np.uint32)
    exponent = (raw >> np.uint32(23)) & np.uint32(255)
    if np.any(exponent == 255):
        raise SmokeFailure('nonfinite tensor')
    return raw


def _decoded(bits):
    """Decode IEEE binary32 fields as binary64 without FTZ-sensitive casts."""
    exponent = ((bits >> np.uint32(23)) & np.uint32(255)).astype(np.int16)
    fraction = bits & np.uint32(0x7fffff)
    significand = (fraction | np.where(exponent == 0, np.uint32(0),
                                        np.uint32(1 << 23))).astype(np.float64)
    powers = np.where(exponent == 0, -149, exponent - 150).astype(np.int16)
    values = np.ldexp(significand, powers)
    return np.where((bits >> np.uint32(31)) != 0, -values, values)


def max_stored_difference(left, right):
    """Binary64 max difference between stored float32 values, including subnormals."""
    if left.shape != right.shape or left.numel() == 0:
        raise SmokeFailure('tensor shape mismatch or empty tensor')
    return float(np.max(np.abs(_decoded(_bits(left)) - _decoded(_bits(right)))))


def _bit_equal(left, right):
    return left.shape == right.shape and bool(np.array_equal(_bits(left), _bits(right)))


def _summary(tensor):
    bits = _bits(tensor)
    return {'shape': list(tensor.shape), 'dtype': 'float32',
            'sha256_le_float32': hashlib.sha256(bits.tobytes()).hexdigest()}


def _hook_state(model):
    registry = torch.nn.modules.module
    names = ('_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_forward_hooks',
             '_forward_hooks_with_kwargs', '_forward_hooks_always_called',
             '_backward_pre_hooks', '_backward_hooks')
    local = tuple((id(module), *(tuple((key, id(hook)) for key, hook in
                                      getattr(module, name, {}).items()) for name in names))
                  for module in model.modules())
    global_names = ('_global_forward_pre_hooks', '_global_forward_hooks',
                    '_global_forward_hooks_with_kwargs', '_global_forward_hooks_always_called',
                    '_global_backward_pre_hooks', '_global_backward_hooks')
    return (local, *(tuple((key, id(hook)) for key, hook in getattr(registry, name, {}).items())
                     for name in global_names))


def _note(error, message):
    add_note = getattr(error, 'add_note', None)
    if callable(add_note):
        add_note(message)
    else:
        # Python 3.10 has no add_note. Preserve the original exception as the
        # raised error while making the cleanup diagnostic visible in its chain.
        diagnostic = SmokeFailure(message)
        previous = error.__cause__ or error.__context__
        if previous is not None and previous is not error:
            diagnostic.__cause__ = previous
        error.__cause__ = diagnostic


def _cleanup_status(model, baseline, error):
    """Retain the primary failure even when checking cleanup itself fails."""
    try:
        restored = _hook_state(model) == baseline
    except BaseException as inspection_error:
        try:
            _note(error, f'Hook inspection failed ({type(inspection_error).__name__}); '
                         'cleanup is unknown, discard the model.')
        except BaseException:
            pass  # Attaching diagnostics can itself fail under resource exhaustion.
        return None
    if not restored:
        try:
            _note(error, 'Hook cleanup is uncertain; discard the contaminated model.')
        except BaseException:
            pass
    return restored


def _target_delta(before, after):
    """Check the prescribed addition and its nonzero stored-value effect."""
    if before.shape != (1,) or after.shape != (1,):
        raise SmokeFailure('invalid target coordinate shape')
    if not _bit_equal(after, before + 1.0):
        raise SmokeFailure('target coordinate does not match float32 addition')
    realized = float(_decoded(_bits(after))[0] - _decoded(_bits(before))[0])
    if not realized > 0:
        raise SmokeFailure('target coordinate did not increase after addition')
    return realized


def _model_state_sha256(model):
    """Hash CPU state in bounded chunks, including names, shapes and dtypes."""
    digest = hashlib.sha256()
    # state_dict omits nonpersistent buffers (including GPT-2 causal masks).
    # Include all named buffers separately, including aliases, so those are
    # checked as well as parameters and persistent state.
    groups = (('state_dict', model.state_dict().items()),
              ('all_buffers', model.named_buffers(remove_duplicate=False)))
    for group, entries in groups:
        for name, tensor in entries:
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or not tensor.is_contiguous()):
                raise SmokeFailure('model state must be contiguous CPU tensors')
            header = json.dumps({'group': group, 'name': name, 'shape': list(tensor.shape),
                                 'dtype': str(tensor.dtype)},
                                sort_keys=True, separators=(',', ':')).encode('utf-8')
            digest.update(len(header).to_bytes(4, 'big'))
            digest.update(header)
            raw = tensor.detach().reshape(-1).view(torch.uint8)
            for start in range(0, raw.numel(), 1024 * 1024):
                digest.update(memoryview(raw[start:start + 1024 * 1024].numpy()))
    return digest.hexdigest()


def manual_forward(model, ids, *, mode='clean', donor=None):
    """Compute an unhooked GPT-2 forward by explicitly stepping its blocks."""
    if mode not in ('clean', 'inject', 'patch'):
        raise ValueError('unsupported manual mode')
    transformer = model.transformer
    length = ids.shape[1]
    positions = torch.arange(length, device='cpu')
    hidden = transformer.drop(transformer.wte(ids) + transformer.wpe(positions.unsqueeze(0)))
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    causal = torch.zeros((1, 1, length, length), dtype=torch.float32)
    causal.masked_fill_(upper, torch.finfo(torch.float32).min)
    vectors = {}
    for index, block in enumerate(transformer.h):
        output = block(hidden, cache_position=positions, attention_mask=causal,
                       use_cache=False, output_attentions=False)
        if type(output) is not tuple or not output or not isinstance(output[0], torch.Tensor):
            raise SmokeFailure('unexpected GPT-2 block output')
        hidden = output[0]
        if index == 2:
            vectors['before'] = hidden[0, TOKEN].detach().clone()
            if mode in ('inject', 'patch'):
                hidden = hidden.clone()
                hidden[0, TOKEN, 0] = hidden[0, TOKEN, 0] + 1.0
                vectors['after_injection'] = hidden[0, TOKEN].detach().clone()
            if mode == 'patch':
                if donor is None or donor.shape != hidden[0, TOKEN].shape:
                    raise SmokeFailure('invalid clean donor')
                hidden[0, TOKEN] = donor
                vectors['after_patch'] = hidden[0, TOKEN].detach().clone()
    return model.lm_head(transformer.ln_f(hidden)).detach().clone(), vectors


def _step(name, operation, **kwargs):
    return {'name': name, 'operation': operation, 'module_path': SITE,
            'invocation': 0, 'batch_index': 0, 'token_index': TOKEN, **kwargs}


def _plan_forward(model, ids, steps, baseline_hooks):
    plan = ActivationPlan(model, steps, max_tensor_bytes=2 * 1024 * 1024,
                          max_capture_bytes=128 * 1024, max_source_bytes=128 * 1024,
                          max_steps=16, max_invocations=1)
    try:
        with plan:
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False).logits.detach().clone()
    except BaseException as error:
        _cleanup_status(model, baseline_hooks, error)
        raise
    else:
        if _hook_state(model) != baseline_hooks:
            raise SmokeFailure('plan did not restore local hooks')
    metadata = plan.metadata
    events = metadata['steps']
    if len(events) != len(steps):
        raise SmokeFailure('missing intervention events')
    for index, (event, step) in enumerate(zip(events, steps)):
        if (event['name'] != step['name'] or event['operation'] != step['operation']
                or event['order'] != index or event['module_path'] != SITE
                or event['invocation'] != 0 or event['batch_index'] != 0
                or event['token_index'] != TOKEN or event['output_selector'] != [0]
                or event['forward_index'] != 0 or event['device'] != 'cpu'
                or event['dtype'] != 'torch.float32'
                or event['output_shape'] != [1, 4, model.config.n_embd]
                or event['selected_shape'] != [model.config.n_embd]
                or event['source_capture'] != step.get('source_capture')
                or event['preceding_same_site_actions'] != [prior['name'] for prior in steps[:index]]):
            raise SmokeFailure('unexpected intervention metadata')
    return result, plan.captures, metadata


def evaluate(model, ids, report=None):
    """Run the frozen comparisons with one model; test fixtures may use other IDs."""
    if report is None:
        report = {}
    if (not isinstance(ids, torch.Tensor) or ids.device.type != 'cpu'
            or ids.dtype != torch.int64 or tuple(ids.shape) != (1, 4)):
        raise SmokeFailure('expected a CPU int64 input row of length four')
    if (any(module.training for module in model.modules()) or len(model.transformer.h) != 6
            or model.config.use_cache is not False
            or model.config._attn_implementation != 'eager'):
        raise SmokeFailure('expected a six-block evaluation-mode GPT-2 model')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise SmokeFailure('expected CPU float32 model parameters')
    if any(buffer.device.type != 'cpu' or (buffer.is_floating_point() and buffer.dtype != torch.float32)
           for buffer in model.buffers()):
        raise SmokeFailure('expected CPU buffers with float32 floating values')
    hooks = _hook_state(model)
    if any(any(registry for registry in module[1:]) for module in hooks[0]) or any(hooks[1:]):
        raise SmokeFailure('preexisting local or global hooks are unsupported')
    report.update({'comparisons': {}, 'outputs': {}, 'activations': {}, 'plan_metadata': {}})
    report['model_state_sha256_before'] = _model_state_sha256(model)
    output_shape = (1, 4, model.config.vocab_size)

    def record_output(name, value):
        if tuple(value.shape) != output_shape:
            raise SmokeFailure('unexpected full logit shape')
        report['outputs'][name] = _summary(value)

    def check_hooks():
        if _hook_state(model) != hooks:
            raise SmokeFailure('forward hook registry changed')

    def compare(name, left, right):
        difference = max_stored_difference(left, right)
        report['comparisons'][name] = {'max_abs_difference_binary64': difference,
                                       'bit_equal': _bit_equal(left, right),
                                       'within_tolerance': difference <= ATOL}
        if difference > ATOL:
            raise SmokeFailure(f'{name} exceeds the fixed tolerance')

    def planned(name, steps):
        logits, captures, metadata = _plan_forward(model, ids, steps, hooks)
        record_output(name, logits)
        report['plan_metadata'][name] = metadata
        for key, value in captures.items():
            report['activations'][name + ':' + key] = _summary(value)
        return logits, captures

    try:
        with torch.inference_mode():
            clean = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          use_cache=False).logits.detach().clone()
            check_hooks()
            clean_repeat = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 use_cache=False).logits.detach().clone()
            check_hooks()
            ref_clean, clean_vectors = manual_forward(model, ids, mode='clean')
            check_hooks()
            donor = clean_vectors['before']
            ref_inject, injection_vectors = manual_forward(model, ids, mode='inject')
            check_hooks()
            ref_patch, patch_vectors = manual_forward(model, ids, mode='patch', donor=donor)
            check_hooks()
            for name, value in (('clean', clean), ('clean_repeat', clean_repeat),
                                ('manual_clean', ref_clean), ('manual_injected', ref_inject),
                                ('manual_patched', ref_patch)):
                record_output(name, value)
            for name, value in (('clean_donor', donor),
                                ('manual_before', injection_vectors['before']),
                                ('manual_after_injection', injection_vectors['after_injection']),
                                ('manual_after_patch', patch_vectors['after_patch'])):
                report['activations'][name] = _summary(value)
            compare('clean_manual', clean, ref_clean)
            compare('clean_repeat', clean_repeat, clean)
            compare('manual_donor_repeat', injection_vectors['before'], donor)

            delta = torch.zeros_like(donor)
            delta[0] = 1.0
            injected_steps = [_step('before', 'capture'), _step('inject', 'add', value=delta),
                              _step('after', 'capture')]
            injected, observed = planned('plan_injected', injected_steps)
            repeated, repeated_values = planned('plan_injected_repeat', injected_steps)
            compare('injected_manual_logits', injected, ref_inject)
            compare('injected_preceding_logits', injected[:, :TOKEN], clean[:, :TOKEN])
            compare('injected_manual_activation', observed['after'], injection_vectors['after_injection'])
            compare('injected_repeat_logits', repeated, injected)
            compare('injected_repeat_activation', repeated_values['after'], observed['after'])
            donor_input_equal = _bit_equal(observed['before'], donor)
            if not donor_input_equal:
                raise SmokeFailure('manual clean donor differs from the plan pre-injection value')
            other_equal = _bit_equal(observed['before'][1:], observed['after'][1:])
            if not other_equal:
                raise SmokeFailure('injection changed a non-target hidden coordinate')
            realized = _target_delta(observed['before'][:1], observed['after'][:1])
            effect = max_stored_difference(injected, clean)
            report['injection'] = {'nominal_coordinate_delta': 1.0,
                                   'realized_coordinate_delta_binary64': realized,
                                   'other_coordinates_bit_equal': other_equal,
                                   'target_matches_expected_add_bits': True,
                                   'manual_donor_bit_equal_before_injection': donor_input_equal,
                                   'max_logit_change': effect,
                                   'preceding_logits_bit_equal': _bit_equal(injected[:, :TOKEN],
                                                                           clean[:, :TOKEN])}
            if effect <= MIN_EFFECT:
                raise SmokeFailure('injection did not change logits above the fixed threshold')

            patch_steps = injected_steps + [_step('patch', 'patch', value=donor),
                                           _step('after_patch', 'capture')]
            patched, patched_values = planned('plan_patched', patch_steps)
            compare('patched_manual_logits', patched, ref_patch)
            compare('patched_clean_logits', patched, clean)
            compare('patched_preceding_logits', patched[:, :TOKEN], clean[:, :TOKEN])
            compare('patched_before_activation', patched_values['before'], donor)
            compare('patched_injected_activation', patched_values['after'], injection_vectors['after_injection'])
            compare('patched_after_activation', patched_values['after_patch'], patch_vectors['after_patch'])
            donor_exact = _bit_equal(patched_values['after_patch'], donor)
            report['injection']['clean_donor_bit_equal_after_patch'] = donor_exact
            if not donor_exact:
                raise SmokeFailure('post-patch activation differs from its donor bits')

            null, null_values = planned('plan_null', [
                _step('before_null', 'capture'), _step('null', 'null'),
                _step('after_null', 'capture')])
            clean_self, clean_values = planned('plan_clean_self_patch', [
                _step('clean_capture', 'capture'),
                _step('clean_self_patch', 'patch', source_capture='clean_capture'),
                _step('after_clean_self_patch', 'capture')])
            injected_self, injected_values = planned('plan_injected_self_patch', [
                _step('inject', 'add', value=delta), _step('snapshot', 'capture'),
                _step('self_patch', 'patch', source_capture='snapshot'),
                _step('after_self_patch', 'capture')])
            controls = {
                'null_activation_bit_equal': _bit_equal(null_values['before_null'], null_values['after_null']),
                'clean_self_patch_activation_bit_equal': _bit_equal(
                    clean_values['clean_capture'], clean_values['after_clean_self_patch']),
                'injected_self_patch_activation_bit_equal': _bit_equal(
                    injected_values['snapshot'], injected_values['after_self_patch']),
            }
            report['controls'] = controls
            if not all(controls.values()):
                raise SmokeFailure('control changed its source activation bits')
            final = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          use_cache=False).logits.detach().clone()
            check_hooks()
            record_output('final_clean', final)
            compare('null_clean_logits', null, clean)
            compare('clean_self_patch_logits', clean_self, clean)
            compare('injected_self_patch_logits', injected_self, injected)
            compare('final_clean_logits', final, clean)
            if _hook_state(model) != hooks:
                raise SmokeFailure('hooks changed after all forwards')
            report['hook_registries_unchanged'] = True
            report['model_state_sha256_after'] = _model_state_sha256(model)
            report['model_state_unchanged'] = (report['model_state_sha256_before'] ==
                                               report['model_state_sha256_after'])
            if not report['model_state_unchanged']:
                raise SmokeFailure('model parameters or buffers changed during smoke')
            return report
    except BaseException as error:
        report['hook_registries_unchanged'] = _cleanup_status(model, hooks, error)
        raise
