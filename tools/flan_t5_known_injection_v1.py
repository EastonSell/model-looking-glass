"""Bounded public FLAN-T5 known-injection study; no generation or downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import torch

from looking_glass.agreement_numeric import decode_f32_le
from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward
from looking_glass.intervention_plan import ActivationPlan

CASE_SHA = '1b21870b8d2c83a7c144ecbf2911728fcdc14a828bf2e8078fca43780df64c11'
RAW_LIMIT = 32 * 1024 * 1024
SITES = tuple(f'{stack}.block.{layer}' for stack in ('encoder', 'decoder') for layer in range(6))


def write_exact(stream, data):
    if stream.write(data) != len(data):
        raise OSError('short evidence write')


def raw_f32(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
            or tensor.device.type != 'cpu' or tensor.numel() == 0 or tensor.numel() * 4 > 1024 * 1024):
        raise ValueError('expected bounded CPU float32 evidence')
    return tensor.detach().contiguous().view(torch.int32).numpy().tobytes()


def equal_bits(left, right):
    return tuple(left.shape) == tuple(right.shape) and raw_f32(left) == raw_f32(right)


def distance(left, right):
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError('distance shape mismatch')
    a, b = decode_f32_le(raw_f32(left)), decode_f32_le(raw_f32(right))
    return math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def modified(clean):
    raw = raw_f32(clean)
    if clean.ndim != 1:
        raise ValueError('injection source must be one hidden vector')
    first = float(decode_f32_le(raw)[0])
    replacement = struct.unpack('<i', struct.pack('<f', first + 4.0))[0]
    result = clean.detach().clone()
    result.view(torch.int32)[0] = replacement
    if distance(clean, result) == 0.0:
        raise ValueError('injection rounded to zero displacement')
    return result


def random_guesses(case_id, stack):
    h = hashlib.sha256(('FLAN-T5-known-injection-random-v1|' + case_id + '|' + stack).encode('ascii')).digest()
    return int.from_bytes(h[:8], 'big') % 12, (0 if stack == 'encoder' else 6) + int.from_bytes(h[8:16], 'big') % 6


class Evidence:
    def __init__(self, root, limit=RAW_LIMIT):
        self.root, self.limit, self.total = root, limit, 0
        root.mkdir(parents=True, exist_ok=False)
        (root / 'tensors').mkdir()
        self.stream = (root / 'attempt.jsonl').open('xb')

    def event(self, value):
        write_exact(self.stream, (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def put(self, value):
        raw = raw_f32(value)
        digest = hashlib.sha256(raw).hexdigest()
        dest = self.root / 'tensors' / (digest + '.f32le')
        if dest.exists():
            if dest.read_bytes() != raw:
                raise ValueError('tensor digest collision')
        else:
            if self.total + len(raw) > self.limit:
                raise ValueError('raw evidence budget exceeded')
            with dest.open('xb') as stream:
                write_exact(stream, raw)
                stream.flush()
                os.fsync(stream.fileno())
            if dest.read_bytes() != raw:
                raise OSError('tensor evidence readback mismatch')
            self.total += len(raw)
        return {'sha256': digest, 'shape': list(value.shape), 'bytes': len(raw), 'path': 'tensors/' + dest.name}

    def close(self):
        self.stream.close()


def _finite(values):
    for tensor in values:
        decode_f32_le(raw_f32(tensor))


def _save_available(evidence, values):
    refs, unavailable = {}, []
    for name, value in values.items():
        try:
            refs[name] = evidence.put(value)
        except Exception as error:
            unavailable.append({'name': name, 'reason': type(error).__name__})
    return refs, unavailable


def _logits(output, decoder_length, vocab):
    logits = output.logits
    if (getattr(output, 'past_key_values', None) is not None or type(logits) is not torch.Tensor
            or logits.dtype != torch.float32 or logits.device.type != 'cpu'
            or tuple(logits.shape) != (1, decoder_length, vocab)):
        raise ValueError('unexpected uncached logits')
    return logits.detach()


def _points():
    return [T5TracePoint(stack, layer, 'block', 0, 1)
            for stack in ('encoder', 'decoder') for layer in range(6)]


def _check_inputs(inputs, snapshots):
    if any(value.dtype != torch.long or value.device.type != 'cpu' or not torch.equal(value, snapshots[name])
           for name, value in inputs.items()):
        raise ValueError('input IDs or masks changed during forward')


def _input_record(snapshots):
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'values': value.tolist()}
            for name, value in snapshots.items()}


def _raise_prioritizing_fatal(first, second):
    """Keep the first fatal interruption; otherwise retain the first failure."""
    if isinstance(first, Exception) and not isinstance(second, Exception):
        raise second.with_traceback(second.__traceback__) from first
    raise first.with_traceback(first.__traceback__) from second


def _record_incomplete(evidence, ctx, observed_return, error, return_scope):
    try:
        evidence.event({'event': 'forward_incomplete', **ctx, 'return_observed': observed_return,
                        'return_scope': return_scope, 'error_type': type(error).__name__,
                        'scope': 'previously saved evidence retained; unavailable captures are not inferred'})
    except BaseException as secondary:
        _raise_prioritizing_fatal(error, secondary)


def _trace_clean(model, inputs, case_id, label, evidence, vocab):
    ctx = {'case_id': case_id, 'arm': label}
    snapshots = {name: value.detach().clone() for name, value in inputs.items()}
    evidence.event({'event': 'forward_start', **ctx, 'inputs': _input_record(snapshots)})
    observed_return = None
    try:
        output, records = trace_t5_forward(model, **inputs, points=_points(), run_id=case_id + '/' + label,
                                          synchronous_cpu=True, max_events=12)
        observed_return = True
        evidence.event({'event': 'wrapper_return_observed', **ctx, 'return_scope': 'trace_wrapper'})
        values = [r.value for r in records]
        metadata = [r.event.to_dict() for r in records]
        refs, unavailable = _save_available(evidence, {'logits': getattr(output, 'logits', None),
                                                        **{f'trace_{i}': v for i, v in enumerate(values)}})
        evidence.event({'event': 'forward_observation', **ctx, 'refs': refs, 'unavailable': unavailable,
                        'trace_events': metadata})
        if unavailable or len(records) != 12:
            raise ValueError('incomplete clean trace or evidence')
        _check_inputs(inputs, snapshots)
        logits = _logits(output, inputs['decoder_input_ids'].shape[1], vocab)
        if any(r.event.module_path != SITES[i] or r.event.output_path != (0,)
               or r.event.token_positions != (1,) or r.event.invocation != 0 for i, r in enumerate(records)):
            raise ValueError('unexpected clean trace coordinates')
        _finite([logits, *values])
        evidence.event({'event': 'forward_complete', **ctx})
        return logits, values
    except BaseException as error:
        _record_incomplete(evidence, ctx, observed_return, error, 'trace_wrapper')
        raise


def _steps(stack, arm, clean, injected):
    target = 2 if stack == 'encoder' else 8
    alternate = target + 1
    steps = []
    for index, path in enumerate(SITES):
        full = {'module_path': path, 'invocation': 0}
        selected = {**full, 'batch_index': 0, 'token_index': 1}
        if index == target:
            steps += [{**full, 'name': 'inject_before', 'operation': 'capture'},
                      {**selected, 'name': 'inject', 'operation': 'patch', 'value': injected},
                      {**full, 'name': 'inject_after', 'operation': 'capture'}]
            if arm == 'self':
                steps += [{**selected, 'name': 'self_source', 'operation': 'capture'},
                          {**full, 'name': 'self_before', 'operation': 'capture'},
                          {**selected, 'name': 'self_patch', 'operation': 'patch', 'source_capture': 'self_source'},
                          {**full, 'name': 'self_after', 'operation': 'capture'}]
            if arm == 'restore':
                steps += [{**full, 'name': 'restore_before', 'operation': 'capture'},
                          {**selected, 'name': 'restore_patch', 'operation': 'patch', 'value': clean[target]},
                          {**full, 'name': 'restore_after', 'operation': 'capture'}]
        if index == alternate and arm == 'alternate':
            steps += [{**full, 'name': 'alternate_before', 'operation': 'capture'},
                      {**selected, 'name': 'alternate_patch', 'operation': 'patch', 'value': clean[alternate]},
                      {**full, 'name': 'alternate_after', 'operation': 'capture'}]
        steps.append({**selected, 'name': f'trace_{index}', 'operation': 'capture'})
    return steps


def _landed(before, after, expected):
    if before.ndim != 3 or after.shape != before.shape or before.shape[0] != 1 or before.shape[1] <= 1:
        return False
    predicted = before.detach().clone()
    predicted.view(torch.int32)[0, 1] = expected.view(torch.int32)
    return equal_bits(predicted, after)


def _trace_arm(model, inputs, case_id, stack, arm, clean, injected, evidence, vocab):
    ctx = {'case_id': case_id, 'stack': stack, 'arm': arm}
    snapshots = {name: value.detach().clone() for name, value in inputs.items()}
    evidence.event({'event': 'forward_start', **ctx, 'inputs': _input_record(snapshots)})
    observed_return = None
    try:
        plan = ActivationPlan(model, _steps(stack, arm, clean, injected), max_steps=24, max_invocations=1,
                              max_tensor_bytes=1024 * 1024, max_capture_bytes=2 * 1024 * 1024,
                              max_source_bytes=64 * 1024)
        with torch.inference_mode(), plan:
            output = plan.run_synchronous_cpu(**inputs, use_cache=False, output_attentions=False,
                                               output_hidden_states=False, return_dict=True)
        observed_return = True
        evidence.event({'event': 'wrapper_return_observed', **ctx, 'return_scope': 'plan_wrapper_and_context'})
        captures, metadata = plan.captures, plan.metadata
        refs, unavailable = _save_available(evidence, {'logits': getattr(output, 'logits', None), **captures})
        evidence.event({'event': 'forward_observation', **ctx, 'refs': refs, 'unavailable': unavailable,
                        'plan_metadata': metadata})
        if unavailable:
            raise ValueError('incomplete evidence: ' + unavailable[0]['reason'])
        _check_inputs(inputs, snapshots)
        logits = _logits(output, inputs['decoder_input_ids'].shape[1], vocab)
        _finite([logits, *captures.values()])
        traces = [captures[f'trace_{i}'] for i in range(12)]
        target = 2 if stack == 'encoder' else 8
        checks = {'inject_before_matches_clean': equal_bits(captures['inject_before'][0, 1], clean[target]),
                  'injection_landing_and_untouched': _landed(captures['inject_before'], captures['inject_after'], injected)}
        alternate_dose = None
        if arm == 'self':
            checks['self_landing_and_untouched'] = _landed(captures['self_before'], captures['self_after'], captures['self_source'])
            checks['self_source_is_injected'] = equal_bits(captures['self_source'], injected)
        elif arm == 'restore':
            checks['restore_landing_and_untouched'] = _landed(captures['restore_before'], captures['restore_after'], clean[target])
            checks['restore_full_equals_preinjection'] = equal_bits(captures['restore_after'], captures['inject_before'])
        elif arm == 'alternate':
            alternate_dose = distance(captures['alternate_before'][0, 1], clean[target + 1])
            checks['alternate_landing_and_untouched'] = _landed(captures['alternate_before'], captures['alternate_after'], clean[target + 1])
        if not all(checks.values()):
            raise ValueError('patch landing or untouched-position check failed')
        evidence.event({'event': 'forward_complete', **ctx, 'checks': checks, 'alternate_dose': alternate_dose})
        return logits, traces, alternate_dose
    except BaseException as error:
        _record_incomplete(evidence, ctx, observed_return, error, 'plan_wrapper_and_context')
        raise


def execute(model, cases, decoder_ids, root, *, vocab, hidden, raw_limit=RAW_LIMIT, metadata=None, postflight=None):
    if sys.byteorder != 'little' or type(raw_limit) is not int or not 0 < raw_limit <= RAW_LIMIT:
        raise ValueError('unsupported byte order or evidence limit')
    if (int(model.config.d_model) != hidden or int(model.get_output_embeddings().weight.shape[0]) != vocab
            or len(model.encoder.block) != 8 or len(model.decoder.block) != 8):
        raise ValueError('unexpected model dimensions')
    if not cases or len({c['case_id'] for c in cases}) != len(cases) or (metadata is not None and type(metadata) is not dict):
        raise ValueError('empty or duplicate cases')
    evidence = Evidence(root, raw_limit)
    trials = []
    try:
        evidence.event({'event': 'begin', 'case_sha256': CASE_SHA, 'case_count': len(cases),
                        'decoder_input_ids': list(decoder_ids), 'sites': list(SITES), 'raw_limit': raw_limit,
                        'model_metadata': metadata or {}})
        for case in cases:
            source = case['input_ids']
            if (len(source) < 3 or source[-1] != 1 or source[1] == 1
                    or len(decoder_ids) < 2 or decoder_ids[1] == 1):
                raise ValueError('inactive or EOS trace coordinate')
            inputs = {'input_ids': torch.tensor([source], dtype=torch.long),
                      'attention_mask': torch.ones((1, len(source)), dtype=torch.long),
                      'decoder_input_ids': torch.tensor([decoder_ids], dtype=torch.long),
                      'decoder_attention_mask': torch.ones((1, len(decoder_ids)), dtype=torch.long)}
            clean_logits, clean = _trace_clean(model, inputs, case['case_id'], 'clean', evidence, vocab)
            repeat_logits, repeat = _trace_clean(model, inputs, case['case_id'], 'clean_repeat', evidence, vocab)
            if not equal_bits(clean_logits, repeat_logits) or any(not equal_bits(a, b) for a, b in zip(clean, repeat, strict=True)):
                raise ValueError('clean repeat differed')
            for stack in ('encoder', 'decoder'):
                target = 2 if stack == 'encoder' else 8
                injection = modified(clean[target])
                inject_logits, injected, _ = _trace_arm(model, inputs, case['case_id'], stack, 'inject', clean, injection, evidence, vocab)
                self_logits, self_trace, _ = _trace_arm(model, inputs, case['case_id'], stack, 'self', clean, injection, evidence, vocab)
                restored_logits, restored, _ = _trace_arm(model, inputs, case['case_id'], stack, 'restore', clean, injection, evidence, vocab)
                alternate_logits, alternate, alternate_dose = _trace_arm(model, inputs, case['case_id'], stack, 'alternate', clean, injection, evidence, vocab)
                if (not equal_bits(inject_logits, self_logits)
                        or any(not equal_bits(a, b) for a, b in zip(injected, self_trace, strict=True))):
                    raise ValueError('self patch differed from injection')
                if (not equal_bits(clean_logits, restored_logits)
                        or any(not equal_bits(a, b) for a, b in zip(clean, restored, strict=True))):
                    raise ValueError('same-site clean restoration differed')
                distances = [distance(a, b) for a, b in zip(clean, injected, strict=True)]
                changed = [i for i, (a, b) in enumerate(zip(clean, injected, strict=True)) if not equal_bits(a, b)]
                if not changed:
                    raise ValueError('injection made no selected trace change')
                first = changed[0]
                largest = max(range(12), key=lambda i: (distances[i], -i))
                across, within = random_guesses(case['case_id'], stack)
                trial = {'case_id': case['case_id'], 'split': case['split'], 'stack': stack,
                         'target_index': target, 'first_change_index': first, 'magnitude_index': largest,
                         'random_across_index': across, 'random_within_index': within, 'known_site_index': target,
                         'trace_l2': distances, 'injection_l2': distance(clean[target], injection),
                         'alternate_l2': alternate_dose, 'alternate_active': alternate_dose > 0.0,
                         'injected_logit_l2': distance(clean_logits, inject_logits),
                         'alternate_logit_l2': distance(clean_logits, alternate_logits),
                         'same_site_restore_matches_clean': True, 'self_matches_injected': True}
                trials.append(trial)
                evidence.event({'event': 'trial_complete', **trial})
        groups = {}
        for split in ('discovery', 'heldout'):
            for stack in ('encoder', 'decoder'):
                selected = [t for t in trials if t['split'] == split and t['stack'] == stack]
                if selected:
                    groups[split + '/' + stack] = {'trials': len(selected), **{
                        key + '_hits': sum(t[key + '_index'] == t['target_index'] for t in selected)
                        for key in ('first_change', 'magnitude', 'random_across', 'random_within', 'known_site')}}
        evidence.event({'event': 'postflight_start', 'scope': 'asset and CUDA checks in production; injected callback in fabricated tests'})
        try:
            if postflight is not None:
                postflight()
        except BaseException as error:
            try:
                evidence.event({'event': 'postflight_incomplete', 'error_type': type(error).__name__})
            except BaseException as secondary:
                _raise_prioritizing_fatal(error, secondary)
            raise
        evidence.event({'event': 'postflight_passed', 'production_callback_supplied': postflight is not None})
        result = {'schema': 'flan-t5-known-injection-result-v1', 'case_sha256': CASE_SHA,
                  'forwards': len(cases) * 10, 'raw_tensor_bytes': evidence.total, 'trials': trials,
                  'groups': groups, 'model_metadata': metadata or {}}
        result_bytes = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        with (root / 'result.json').open('xb') as stream:
            write_exact(stream, result_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if (root / 'result.json').read_bytes() != result_bytes:
            raise OSError('result evidence readback mismatch')
        evidence.event({'event': 'complete', 'forwards': len(cases) * 10,
                        'result_sha256': hashlib.sha256(result_bytes).hexdigest(), 'raw_tensor_bytes': evidence.total})
    except BaseException as error:
        try:
            evidence.close()
        except BaseException as secondary:
            _raise_prioritizing_fatal(error, secondary)
        raise
    else:
        evidence.close()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.environ.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    from looking_glass.flan_t5_feasibility import verify_assets
    import tokenizers
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA:
        raise ValueError('case identity mismatch')
    obj = json.loads(raw)
    if len(obj['cases']) != 8 or obj['decoder_prefix']['decoder_input_ids'] != [0, 1368]:
        raise ValueError('unexpected public case count or decoder prefix')
    before = verify_assets(args.assets)
    tok = AutoTokenizer.from_pretrained(str(args.assets), local_files_only=True, use_fast=True, trust_remote_code=False)
    if tok.pad_token_id != 0 or tok.eos_token_id != 1 or tok('record', add_special_tokens=False)['input_ids'] != [1368]:
        raise ValueError('decoder tokenizer binding mismatch')
    for case in obj['cases']:
        if tok(case['text'])['input_ids'] != case['input_ids']:
            raise ValueError('source tokenization mismatch')
    model, info = AutoModelForSeq2SeqLM.from_pretrained(str(args.assets), local_files_only=True,
        use_safetensors=True, trust_remote_code=False, attn_implementation='eager',
        torch_dtype=torch.float32, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('checkpoint loading diagnostics')
    model.eval()
    torch.set_num_threads(1)
    metadata = {'model_id': 'google/flan-t5-small', 'revision': '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab',
                'asset_manifest_sha256': 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083',
                'asset_sha256': {name: item['sha256'] for name, item in before['files'].items()},
                'python_version': sys.version.split()[0], 'torch_version': torch.__version__,
                'transformers_version': transformers.__version__, 'tokenizers_version': tokenizers.__version__,
                'device': 'cpu', 'dtype': 'float32', 'attention': 'eager', 'torch_threads': torch.get_num_threads(),
                'use_cache': False, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def postflight():
        if torch.cuda.is_initialized() or verify_assets(args.assets) != before:
            raise RuntimeError('CUDA initialized or assets changed')
    result = execute(model, obj['cases'], [0, 1368], args.output, vocab=32128, hidden=512,
                     metadata=metadata, postflight=postflight)
    print(json.dumps({'forwards': result['forwards'], 'raw_tensor_bytes': result['raw_tensor_bytes'],
                      'result_sha256': hashlib.sha256((args.output / 'result.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
