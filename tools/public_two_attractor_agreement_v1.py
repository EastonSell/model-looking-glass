"""Run the fixed public two-attractor evaluation; no protected records are read."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

CASE_SHA256 = 'fcb79a2762a2fd9fa346347c5156f5eb87bba314b1c8a021fe5fb639827f2798'
CONTINUATIONS = {'is': 318, 'are': 389, 'was': 373, 'were': 547}


def _partition(case):
    s, f = case['subject_pair'], case['frame_pair']
    if s in (1, 2) and f == 1:
        return 'development'
    if s in (3, 4) and f == 2:
        return 'evaluation'
    if s in (1, 2) and f == 2:
        return 'cross_subject_development'
    return 'cross_subject_evaluation'


def _mean(values):
    if not values:
        raise ValueError('empty group')
    return sum(values) / len(values)


def aggregate(cases, rows):
    """Recompute all predeclared contrasts from saved case logits."""
    expected = {case['case_id']: case for case in cases}
    if len(expected) != 128 or len(rows) != 128:
        raise ValueError('expected all 128 distinct cases')
    by_id = {row['case_id']: row for row in rows}
    if len(by_id) != 128 or set(by_id) != set(expected):
        raise ValueError('saved rows do not match cases')
    units = defaultdict(dict)
    cases_out = []
    for case in cases:
        row = by_id[case['case_id']]
        scores = row['logits']
        if set(scores) != set(CONTINUATIONS) or any(
                type(scores[k]) not in (int, float) or not math.isfinite(scores[k]) for k in scores):
            raise ValueError('invalid saved logits')
        sign = 1 if case['subject_number'] == 'singular' else -1
        near_mismatch = int(case['near_number'] != case['subject_number'])
        beside_mismatch = int(case['beside_number'] != case['subject_number'])
        key = (case['subject_pair'], case['frame_pair'], case['order'], case['subject_number'])
        raw = {'present': float(scores['is']) - float(scores['are']),
               'past': float(scores['was']) - float(scores['were'])}
        cell = (near_mismatch, beside_mismatch)
        if cell in units[key]:
            raise ValueError('duplicate factorial cell')
        units[key][cell] = raw
        cases_out.append({'case_id': case['case_id'], 'partition': _partition(case),
                          'near_mismatch': near_mismatch, 'beside_mismatch': beside_mismatch,
                          'raw_margins': raw,
                          'signed_margins': {k: sign * v for k, v in raw.items()},
                          'correct': {k: sign * v > 0 for k, v in raw.items()}, 'logits': scores})
    unit_out = []
    for key in sorted(units):
        cells = units[key]
        if set(cells) != {(0, 0), (1, 0), (0, 1), (1, 1)}:
            raise ValueError('missing factorial cell')
        subject, pair, order, number = key
        sign = 1 if number == 'singular' else -1
        calc = {}
        for tense in ('present', 'past'):
            a, b, c, d = (sign * cells[cell][tense] for cell in ((0, 0), (1, 0), (0, 1), (1, 1)))
            # SS-SP-PS+PP uses absolute noun numbers; complementing both
            # coordinates for a plural subject leaves the cross-difference unchanged.
            raw_interaction = sign * (d - b - c + a)
            calc[tense] = {'all_congruent': a, 'near_only_mismatch': b,
                           'beside_only_mismatch': c, 'both_mismatch': d,
                           'near_shift': ((b - a) + (d - c)) / 2,
                           'beside_shift': ((c - a) + (d - b)) / 2,
                           'interaction': d - b - c + a,
                           'raw_number_interaction': raw_interaction}
        unit_out.append({'subject_pair': subject, 'frame_pair': pair, 'order': order,
                         'subject_number': number,
                         'partition': _partition({'subject_pair': subject, 'frame_pair': pair}),
                         'contrasts': calc})
    if len(unit_out) != 32:
        raise ValueError('expected 32 factorial units')
    grouping = {'all': unit_out}
    for field in ('partition', 'subject_number', 'subject_pair', 'frame_pair', 'order'):
        for unit in unit_out:
            grouping.setdefault(f'{field}={unit[field]}', []).append(unit)
    summary = {}
    for label, group in grouping.items():
        summary[label] = {'units': len(group)}
        for tense in ('present', 'past'):
            summary[label][tense] = {name: _mean([u['contrasts'][tense][name] for u in group])
                                     for name in ('all_congruent', 'near_only_mismatch',
                                                  'beside_only_mismatch', 'both_mismatch',
                                                  'near_shift', 'beside_shift', 'interaction')}
    condition = {}
    for near in (0, 1):
        for beside in (0, 1):
            rows_here = [r for r in cases_out if r['near_mismatch'] == near and r['beside_mismatch'] == beside]
            condition[f'{near}{beside}'] = {'cases': len(rows_here)}
            for tense in ('present', 'past'):
                condition[f'{near}{beside}'][tense] = {
                    'mean_signed_margin': _mean([r['signed_margins'][tense] for r in rows_here]),
                    'correct': sum(r['correct'][tense] for r in rows_here)}
    return {'case_results': cases_out, 'factorial_units': unit_out,
            'unit_summary': summary, 'condition_summary': condition}


def _append(stream, record):
    stream.write((json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    stream.flush()
    os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = args.cases.read_bytes()
    if hashlib.sha256(data).hexdigest() != CASE_SHA256:
        raise ValueError('unexpected fixed case file')
    obj = json.loads(data)
    cases = obj['cases']
    if len(cases) != 128 or len({c['case_id'] for c in cases}) != 128:
        raise ValueError('invalid case set')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    from looking_glass.pretrained import load_distilgpt2
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'attempt.jsonl').open('xb') as stream:
        _append(stream, {'event': 'loading', 'cases_sha256': CASE_SHA256})
        loaded = load_distilgpt2(cache_dir=str(args.cache), allow_download=False)
        model, tokenizer, metadata = loaded.model, loaded.tokenizer, loaded.metadata
        if metadata['weights_sha256'] != 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b':
            raise ValueError('unexpected checkpoint')
        if metadata['asset_sha256']['tokenizer.json'] != obj['tokenizer_sha256']:
            raise ValueError('unexpected tokenizer')
        for case in cases:
            ids = tokenizer.encode(case['text'], add_special_tokens=False)
            if ids != case['input_ids'] or not 0 < len(ids) <= 32:
                raise ValueError('tokenization mismatch')
            for name, token_id in CONTINUATIONS.items():
                if tokenizer.encode(case['text'] + ' ' + name, add_special_tokens=False) != ids + [token_id]:
                    raise ValueError('continuation mismatch')
        _append(stream, {'event': 'model_ready', 'metadata': metadata})
        rows = []
        for case in cases:
            _append(stream, {'event': 'forward_start', 'case_id': case['case_id']})
            ids = torch.tensor([case['input_ids']], dtype=torch.long, device='cpu')
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
            values = {name: float(logits[token_id].item()) for name, token_id in CONTINUATIONS.items()}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('nonfinite logit')
            row = {'case_id': case['case_id'], 'logits': values}
            _append(stream, {'event': 'forward_result', **row})
            rows.append(row)
        result = {'schema': 'public-two-attractor-agreement-result-v1',
                  'cases_sha256': CASE_SHA256, 'model_metadata': metadata,
                  'analysis': aggregate(cases, rows)}
        result_bytes = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        with (args.output / 'result.json').open('xb') as output:
            output.write(result_bytes)
            output.flush()
            os.fsync(output.fileno())
        _append(stream, {'event': 'complete', 'result_sha256': hashlib.sha256(result_bytes).hexdigest()})
    print(json.dumps({'cases': len(rows), 'result_sha256': hashlib.sha256(result_bytes).hexdigest()}))


if __name__ == '__main__':
    main()
