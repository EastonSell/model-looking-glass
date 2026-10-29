"""Public, fixed two-attractor activation replacement using the existing plan API."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from looking_glass.intervention_plan import ActivationPlan

CASE_SHA = '6d0cbba21d15f9ab05df106617141c25a460f07be7ee57c53470f969eb79e518'
CONTINUATIONS = {'is': 318, 'are': 389, 'was': 373, 'were': 547}
SITES = ('transformer.h.1.attn', 'transformer.h.3.attn')
RAW_LIMIT = 16 * 1024 * 1024


def _bytes(value, shape):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.device.type != 'cpu' or tuple(value.shape) != tuple(shape)
            or not bool(torch.isfinite(value).all())):
        raise ValueError('invalid activation shape, device, dtype or finiteness')
    return value.detach().contiguous().numpy().tobytes(order='C')


def _scores(output):
    logits = output.logits
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
            or logits.device.type != 'cpu' or logits.ndim != 3
            or tuple(logits.shape) != (1, 8, 50257) or not bool(torch.isfinite(logits).all())
            or getattr(output, 'past_key_values', None) is not None):
        raise ValueError('invalid model output')
    values = [float(logits[0, -1, CONTINUATIONS[name]].item()) for name in CONTINUATIONS]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite selected logit')
    return {'values': dict(zip(CONTINUATIONS, values)), 'float32_le_hex': struct.pack('<4f', *values).hex()}


def _append(stream, value):
    stream.write((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    stream.flush()
    os.fsync(stream.fileno())


def _selected_bits(output):
    logits = getattr(output, 'logits', None)
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
            or logits.device.type != 'cpu' or logits.ndim != 3 or logits.shape[0] != 1
            or logits.shape[1] != 8 or logits.shape[2] <= 547):
        return None
    values = logits[0, -1, [318, 389, 373, 547]].detach().contiguous()
    return values.numpy().tobytes(order='C').hex()


def _incomplete(stream, evidence, plan, output, returned, context, error):
    """Best-effort bounded post-return evidence; never replace the failure."""
    refs, unavailable = {}, []
    try:
        captures = plan.captures
    except BaseException:
        captures = {}
        unavailable.append('plan captures unavailable')
    for name, tensor in captures.items():
        try:
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tensor.device.type != 'cpu' or tensor.numel() * 4 > 1024 * 1024):
                raise ValueError('unbounded capture')
            raw = tensor.detach().contiguous().numpy().tobytes(order='C')
            refs[name] = evidence.put(raw, tuple(tensor.shape))
        except BaseException:
            unavailable.append('capture ' + name)
    try:
        bits = _selected_bits(output) if output is not None else None
    except BaseException:
        bits = None
        unavailable.append('selected logit bits')
    try:
        _append(stream, {'event': 'forward_incomplete', **context,
                         'model_returned': returned, 'error_type': type(error).__name__,
                         'selected_float32_le_hex': bits, 'captures': refs,
                         'unavailable': unavailable})
    except BaseException as evidence_error:
        if hasattr(error, 'add_note'):
            error.add_note('Incomplete-forward evidence could not be fully recorded: '
                           + type(evidence_error).__name__)


class Evidence:
    def __init__(self, directory):
        self.directory = directory
        self.total = 0
        (directory / 'tensors').mkdir()

    def put(self, raw, shape):
        digest = hashlib.sha256(raw).hexdigest()
        file = self.directory / 'tensors' / (digest + '.f32le')
        if file.exists():
            if file.read_bytes() != raw:
                raise ValueError('tensor digest collision')
        else:
            if self.total + len(raw) > RAW_LIMIT:
                raise ValueError('raw tensor evidence exceeds 16 MiB')
            with file.open('xb') as out:
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            self.total += len(raw)
        return {'sha256': digest, 'shape': list(shape), 'bytes': len(raw), 'path': 'tensors/' + file.name}


def _units(cases):
    if len(cases) != 32 or len({c['case_id'] for c in cases}) != 32:
        raise ValueError('expected 32 distinct cases')
    units = defaultdict(dict)
    for c in cases:
        if (len(c['input_ids']) != 8 or c['subject_number'] not in ('singular', 'plural')
                or c['near_number'] not in ('singular', 'plural') or c['beside_number'] not in ('singular', 'plural')):
            raise ValueError('invalid case fields')
        key = (c['subject_pair'], c['order'], c['subject_number'])
        cell = (int(c['near_number'] != c['subject_number']), int(c['beside_number'] != c['subject_number']))
        if cell in units[key]:
            raise ValueError('duplicate cell')
        units[key][cell] = c
    if len(units) != 8 or any(set(x) != {(0, 0), (1, 0), (0, 1), (1, 1)} for x in units.values()):
        raise ValueError('incomplete factorial')
    for cells in units.values():
        target = cells[(0, 0)]['input_ids']
        for (a, b), case in cells.items():
            changed = {i for i, (x, y) in enumerate(zip(target, case['input_ids'])) if x != y}
            if changed != ({4} if a else set()) | ({7} if b else set()):
                raise ValueError('donor and target token alignment differs')
    return units


def _signed(scores, number, tense):
    sign = 1 if number == 'singular' else -1
    a, b = ('is', 'are') if tense == 'present' else ('was', 'were')
    return sign * (scores['values'][a] - scores['values'][b])


def _contrast(clean, effects, number, tense):
    m = {cell: _signed(value, number, tense) for cell, value in clean.items()}
    d = {name: _signed(value, number, tense) - m[(0, 0)] for name, value in effects.items()}
    patched = {name: _signed(value, number, tense) for name, value in effects.items()}
    clean_i = m[(1, 1)] - m[(1, 0)] - m[(0, 1)] + m[(0, 0)]
    patch_i = d['both'] - d['near'] - d['beside']
    return {'clean_margins': {str(cell[0]) + str(cell[1]): value for cell, value in m.items()},
            'clean_accuracy': {str(cell[0]) + str(cell[1]):
                               {'correct': value > 0, 'tie': value == 0} for cell, value in m.items()},
            'patched_margins': patched,
            'patched_accuracy': {name: {'correct': value > 0, 'tie': value == 0}
                                 for name, value in patched.items()},
            'patch_shifts': d, 'clean_interaction': clean_i,
            'patch_interaction': patch_i, 'patch_minus_clean': patch_i - clean_i}


def execute(model, cases, directory, metadata):
    """Run on an already loaded CPU model; return a complete saved result."""
    if sys.byteorder != 'little':
        raise ValueError('raw tensor evidence requires little endian')
    units = _units(cases)
    directory.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(directory)
    captures, clean_scores = {}, {}
    summary = []
    with (directory / 'attempt.jsonl').open('xb') as stream:
        _append(stream, {'event': 'begin', 'case_sha256': CASE_SHA, 'metadata': metadata})
        for case in cases:
            _append(stream, {'event': 'forward_start', 'kind': 'clean', 'case_id': case['case_id']})
            steps = [{'name': f'site_{i}', 'operation': 'capture', 'module_path': site,
                      'invocation': 0, 'batch_index': 0, 'token_index': 7}
                     for i, site in enumerate(SITES)]
            ids = torch.tensor([case['input_ids']], dtype=torch.long, device='cpu')
            plan = ActivationPlan(model, steps, max_tensor_bytes=1024 * 1024,
                                  max_capture_bytes=1024 * 1024, max_invocations=1)
            context = {'kind': 'clean', 'case_id': case['case_id']}
            output = None
            returned = False
            try:
                with torch.inference_mode(), plan:
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    returned = True
                    _append(stream, {'event': 'model_returned', **context})
                scores = _scores(output)
                refs = {}
                for i, site in enumerate(SITES):
                    vector = plan.captures[f'site_{i}']
                    raw = _bytes(vector, (768,))
                    captures[(case['case_id'], site)] = vector.detach().clone()
                    refs[site] = evidence.put(raw, (768,))
                clean_scores[case['case_id']] = scores
                _append(stream, {'event': 'forward_result', **context,
                                 'scores': scores, 'vectors': refs, 'plan_metadata': plan.metadata})
            except BaseException as error:
                _incomplete(stream, evidence, plan, output, returned, context, error)
                raise
        for key in sorted(units):
            subject, order, number = key
            cells = units[key]
            target = cells[(0, 0)]
            clean = {cell: clean_scores[c['case_id']] for cell, c in cells.items()}
            for site in SITES:
                effects = {}
                for label, cell in (('null', (0, 0)), ('self', (0, 0)),
                                    ('near', (1, 0)), ('beside', (0, 1)), ('both', (1, 1))):
                    donor = cells[cell]
                    _append(stream, {'event': 'forward_start', 'kind': label,
                                     'target': target['case_id'], 'donor': donor['case_id'], 'site': site})
                    common = {'module_path': site, 'invocation': 0, 'batch_index': 0, 'token_index': 7}
                    whole = {'module_path': site, 'invocation': 0}
                    action = {'name': 'action', 'operation': 'null' if label == 'null' else 'patch', **common}
                    if label == 'self':
                        action['source_capture'] = 'before'
                    elif label != 'null':
                        action['value'] = captures[(donor['case_id'], site)]
                    steps = [{'name': 'whole_before', 'operation': 'capture', **whole},
                             {'name': 'before', 'operation': 'capture', **common}, action,
                             {'name': 'after', 'operation': 'capture', **common},
                             {'name': 'whole_after', 'operation': 'capture', **whole}]
                    ids = torch.tensor([target['input_ids']], dtype=torch.long, device='cpu')
                    plan = ActivationPlan(model, steps, max_tensor_bytes=1024 * 1024,
                                          max_capture_bytes=1024 * 1024, max_source_bytes=128 * 1024,
                                          max_invocations=1)
                    context = {'kind': label, 'target': target['case_id'], 'donor': donor['case_id'], 'site': site}
                    output = None
                    returned = False
                    try:
                        with torch.inference_mode(), plan:
                            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                            returned = True
                            _append(stream, {'event': 'model_returned', **context})
                        scores = _scores(output)
                        values = plan.captures
                        before = _bytes(values['before'], (768,))
                        after = _bytes(values['after'], (768,))
                        whole_before = _bytes(values['whole_before'], (1, 8, 768))
                        whole_after = _bytes(values['whole_after'], (1, 8, 768))
                        expected = _bytes(captures[(donor['case_id'], site)], (768,))
                        target_clean = _bytes(captures[(target['case_id'], site)], (768,))
                        offset = 7 * 768 * 4
                        flags = {'target_matches_clean': before == target_clean,
                                 'landing_matches_donor': after == expected,
                                 'untouched_positions_equal': whole_before[:offset] == whole_after[:offset]
                                     and whole_before[offset + 768 * 4:] == whole_after[offset + 768 * 4:],
                                 'selected_slice_consistent': whole_before[offset:offset + 768 * 4] == before
                                     and whole_after[offset:offset + 768 * 4] == after,
                                 'control_logits_equal_clean': label not in ('null', 'self')
                                     or scores['float32_le_hex'] == clean[(0, 0)]['float32_le_hex']}
                        displacement = math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in
                            zip(values['before'].tolist(), captures[(donor['case_id'], site)].tolist())))
                        refs = {'whole_before': evidence.put(whole_before, (1, 8, 768)),
                                'whole_after': evidence.put(whole_after, (1, 8, 768)),
                                'donor': evidence.put(expected, (768,))}
                        _append(stream, {'event': 'forward_result', **context, 'scores': scores,
                                         'displacement_l2': displacement, 'checks': flags, 'tensors': refs,
                                         'plan_metadata': plan.metadata})
                        if not all(flags.values()):
                            raise ValueError('an action control or landing check failed')
                    except BaseException as error:
                        _incomplete(stream, evidence, plan, output, returned, context, error)
                        raise
                    if label in ('near', 'beside', 'both'):
                        effects[label] = scores
                contrast = {}
                for tense in ('present', 'past'):
                    contrast[tense] = _contrast(clean, effects, number, tense)
                summary.append({'subject_pair': subject, 'order': order, 'subject_number': number,
                                'target': target['case_id'], 'site': site, 'contrasts': contrast})
        if len(clean_scores) != 32 or len(summary) != 16:
            raise ValueError('incomplete schedule')
        groups = {'all': summary}
        for field in ('site', 'subject_pair', 'order', 'subject_number'):
            for unit in summary:
                groups.setdefault(f'{field}={unit[field]}', []).append(unit)
        means = {}
        for name, members in groups.items():
            means[name] = {'units': len(members)}
            for tense in ('present', 'past'):
                means[name][tense] = {
                    metric: math.fsum(u['contrasts'][tense][metric] for u in members) / len(members)
                    for metric in ('clean_interaction', 'patch_interaction', 'patch_minus_clean')}
                means[name][tense]['patch_shifts'] = {
                    label: math.fsum(u['contrasts'][tense]['patch_shifts'][label] for u in members) / len(members)
                    for label in ('near', 'beside', 'both')}
        result = {'schema': 'public-two-attractor-patch-result-v1', 'case_sha256': CASE_SHA,
                  'model_metadata': metadata, 'forwards': 112, 'raw_tensor_bytes': evidence.total,
                  'units': summary, 'means': means}
        result_bytes = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        with (directory / 'result.json').open('xb') as out:
            out.write(result_bytes)
            out.flush()
            os.fsync(out.fileno())
        _append(stream, {'event': 'complete', 'forwards': 112,
                         'result_sha256': hashlib.sha256(result_bytes).hexdigest(),
                         'raw_tensor_bytes': evidence.total})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA:
        raise ValueError('unexpected case file')
    obj = json.loads(raw)
    cases = obj['cases']
    _units(cases)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    from looking_glass.pretrained import load_distilgpt2
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    loaded = load_distilgpt2(cache_dir=str(args.cache), allow_download=False)
    if (loaded.metadata['weights_sha256'] != 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
            or loaded.metadata['asset_sha256']['tokenizer.json'] != obj['tokenizer_sha256']):
        raise ValueError('checkpoint or tokenizer mismatch')
    for case in cases:
        ids = loaded.tokenizer.encode(case['text'], add_special_tokens=False)
        if ids != case['input_ids']:
            raise ValueError('prompt tokenization mismatch')
        for name, token in CONTINUATIONS.items():
            if loaded.tokenizer.encode(case['text'] + ' ' + name, add_special_tokens=False) != ids + [token]:
                raise ValueError('continuation tokenization mismatch')
    result = execute(loaded.model, cases, args.output, loaded.metadata)
    print(json.dumps({'forwards': result['forwards'], 'result_sha256': hashlib.sha256((args.output / 'result.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
