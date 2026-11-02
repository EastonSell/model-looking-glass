"""Replay one saved cross-model agreement attempt from retained bytes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

CASE_SHA = 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503'
VOCAB = {'gpt': 50257, 'bert': 30522}
ARMS = ('clean', 'self', 'opposite', 'same', 'random', 'clean_after')
TARGETS = {'gpt': {'is': 318, 'are': 389, 'was': 373, 'were': 547},
           'bert': {'is': 2003, 'are': 2024, 'was': 2001, 'were': 2020}}


def require_subnormals():
    """Fail closed if this thread flushes float32 or float64 subnormals."""
    with np.errstate(under='ignore'):
        f32_input = np.array([1], dtype=np.uint32).view(np.float32)
        f64_input = np.array([1], dtype=np.uint64).view(np.float64)
        f32_produced = np.array([np.finfo(np.float32).tiny], np.float32) / np.float32(2)
        f64_produced = np.array([np.finfo(np.float64).tiny], np.float64) / np.float64(2)
        checks = ((f32_input * np.float32(1)).view(np.uint32)[0] == 1,
                  (f64_input * np.float64(1)).view(np.uint64)[0] == 1,
                  f32_produced.view(np.uint32)[0] == 1 << 22,
                  f64_produced.view(np.uint64)[0] == 1 << 51)
    if not all(checks):
        raise RuntimeError('float32 or float64 subnormals are flushed on this thread')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(data)
    return hasher.hexdigest()


def read_jsonl(path):
    data = path.read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('partial JSONL')
    return [json.loads(line) for line in data.splitlines()]


def same_bits(a, b):
    return (np.asarray(a, dtype='<f4').shape == np.asarray(b, dtype='<f4').shape
            and np.asarray(a, dtype='<f4').tobytes() == np.asarray(b, dtype='<f4').tobytes())


def margins(logits, targets):
    return {'is_are': float(logits[targets['is']]) - float(logits[targets['are']]),
            'was_were': float(logits[targets['was']]) - float(logits[targets['were']])}


def distribution(clean, changed):
    require_subnormals()
    a, b = clean.astype(np.float64), changed.astype(np.float64)
    logp = a - np.logaddexp.reduce(a)
    logq = b - np.logaddexp.reduce(b)
    p, q = np.exp(logp), np.exp(logq)
    p /= p.sum()
    q /= q.sum()
    difference = b - a
    # A shared offset changes neither distribution. Center at the clean top
    # to keep expm1/log1p accurate even when a tail's mass is far below one ULP.
    relative = difference - difference[np.argmax(a)]
    unresolved = False
    if np.max(np.abs(relative)) <= 600 and not np.any((p == 0) | (q == 0)):
        weighted = float(np.sum(p * np.expm1(relative)))
        if weighted > -1 and math.isfinite(weighted):
            log_ratio = relative - math.log1p(weighted)
            exp_difference = np.expm1(log_ratio)
            tv = float(np.sum(p * np.abs(exp_difference)) / 2)
            # KL(p||q) = E_p[exp(log(q/p))-1-log(q/p)]; this form avoids
            # cancellation for nearly identical distributions.
            small = np.abs(log_ratio) < 1e-4
            term = exp_difference - log_ratio
            x = log_ratio[small]
            term[small] = x * x * (0.5 + x * (1/6 + x * (1/24 + x/120)))
            kl = float(np.sum(p * np.maximum(term, 0)))
        else:
            unresolved = True
    else:
        unresolved = True
    if unresolved:
        # Individual underflowed probabilities can sum to a representable
        # mass; do not present their naive float64 sum as a measured zero.
        tv = None
        kl = None
    clean_top, changed_top = int(np.argmax(a)), int(np.argmax(b))
    clean_ties = int(np.count_nonzero(a == a[clean_top]))
    changed_ties = int(np.count_nonzero(b == b[changed_top]))
    return {'tv': tv, 'kl_clean_to_arm': kl, 'numerically_unresolved': unresolved,
            'clean_top1_id': clean_top, 'arm_top1_id': changed_top,
            'clean_top1_ties': clean_ties, 'arm_top1_ties': changed_ties,
            'representative_top1_id_changed': clean_top != changed_top,
            'unique_top1_changed': (clean_top != changed_top if clean_ties == changed_ties == 1 else None)}


def score_case(case, logits_by_arm, targets):
    require_subnormals()
    base = margins(logits_by_arm['clean'], targets)
    sign = 1 if case['number'] == 1 else -1
    values = {}
    for arm in ('opposite', 'same', 'random'):
        scores = margins(logits_by_arm[arm], targets)
        values[arm] = {'is_are': scores['is_are'], 'was_were': scores['was_were'],
                       'donorward_effect': sign * (scores['is_are'] - base['is_are']),
                       **distribution(logits_by_arm['clean'], logits_by_arm[arm])}
    opposite = values['opposite']['donorward_effect']
    same = values['same']['donorward_effect']
    random = values['random']['donorward_effect']
    return {'case_id': case['case_id'], 'frame': case['frame'], 'lemma': case['lemma'],
            'number': case['number'], 'attractor_number': case['attractor_number'],
            'clean_margin': base, 'clean_correct': (base['is_are'] > 0 if case['number'] == 0
                                                   else base['is_are'] < 0),
            'effects': values, 'opposite_minus_same': opposite - same,
            'opposite_minus_random': opposite - random}


def summarize(rows):
    n = len(rows)
    if n != 32:
        raise ValueError('expected all 32 fixed cases')
    mean_same = math.fsum(row['opposite_minus_same'] for row in rows) / n
    mean_random = math.fsum(row['opposite_minus_random'] for row in rows) / n
    positive_both = sum(row['opposite_minus_same'] > 0 and row['opposite_minus_random'] > 0 for row in rows)
    correct = sum(row['clean_correct'] for row in rows)
    baseline_ok = correct >= 24
    prediction = mean_same > 0 and mean_random > 0 and positive_both >= 24
    def groups(field):
        result = {}
        for value in sorted({row[field] for row in rows}):
            selected = [row for row in rows if row[field] == value]
            count = len(selected)
            result[str(value)] = {'cases': count, 'correct_clean': sum(row['clean_correct'] for row in selected),
                                  'mean_opposite_minus_same': math.fsum(row['opposite_minus_same'] for row in selected) / count,
                                  'mean_opposite_minus_random': math.fsum(row['opposite_minus_random'] for row in selected) / count,
                                  'positive_against_both': sum(row['opposite_minus_same'] > 0 and row['opposite_minus_random'] > 0
                                                               for row in selected)}
        return result
    doses = [row['requested_q'] for row in rows]
    return {'cases': n, 'correct_clean': correct, 'baseline_adequate': baseline_ok,
            'mean_opposite_minus_same': mean_same, 'mean_opposite_minus_random': mean_random,
            'positive_against_both': positive_both, 'descriptive_prediction_passes': prediction,
            'by_frame': groups('frame'), 'by_lemma': groups('lemma'),
            'requested_q': {'min': min(doses), 'max': max(doses), 'mean': math.fsum(doses) / n},
            'interpretation': ('baseline-limited' if not baseline_ok else
                               'supports-fixed-descriptive-prediction' if prediction else
                               'does-not-support-fixed-descriptive-prediction')}


def replay(attempt, manifest, exit_observation):
    require_subnormals()
    metadata = json.loads((attempt / 'metadata.json').read_bytes())
    completed = json.loads((attempt / 'complete.json').read_bytes())
    model_key = metadata['model']
    if (model_key not in VOCAB or metadata.get('case_sha256') != CASE_SHA
            or digest(attempt / 'metadata.json') != completed.get('metadata_sha256')
            or completed.get('status') != 'complete' or completed.get('events') != 192
            or (attempt / 'failure.txt').exists()):
        raise ValueError('attempt is not complete')
    if (exit_observation.get('schema') != 'cross-model-direct-child-exit-v1'
            or exit_observation.get('status') != 'complete' or exit_observation.get('exit_code') != 0
            or exit_observation.get('interrupted') is not None
            or exit_observation.get('child_pid') != metadata.get('pid')
            or exit_observation.get('source_commit') != metadata.get('source_commit')
            or exit_observation.get('attempt_name') != attempt.name
            or exit_observation.get('complete_sha256') != digest(attempt / 'complete.json')
            or exit_observation.get('metadata_sha256') != digest(attempt / 'metadata.json')):
        raise ValueError('successful direct-child exit was not observed for this attempt')
    for name, field in (('starts.jsonl', 'starts_sha256'), ('events.jsonl', 'events_sha256'),
                        ('memory.jsonl', 'memory_sha256'), ('retention-errors.jsonl', 'retention_errors_sha256'),
                        ('logits.f32', 'logits_sha256'),
                        ('witness.f32', 'witness_sha256')):
        if digest(attempt / name) != completed.get(field):
            raise ValueError(f'saved evidence digest mismatch: {name}')
    if (attempt / 'retention-errors.jsonl').read_bytes():
        raise ValueError('attempt contains return-retention errors')
    starts, events, samples = (read_jsonl(attempt / name) for name in ('starts.jsonl', 'events.jsonl', 'memory.jsonl'))
    cases = manifest['cases']
    if len(cases) != 32 or len(starts) != 192 or len(events) != 192 or len(samples) != 192:
        raise ValueError('incomplete event, start, sample or case count')
    schedule = [(index, 'clean') for index in range(32)]
    schedule += [(index, arm) for index in range(32) for arm in ARMS[1:]]
    vocab = VOCAB[model_key]
    if ((attempt / 'logits.f32').stat().st_size != 192 * vocab * 4
            or (attempt / 'witness.f32').stat().st_size != 192 * 2 * 768 * 4):
        raise ValueError('saved tensor byte count differs')
    all_logits = np.memmap(attempt / 'logits.f32', dtype='<f4', mode='r', shape=(192, vocab))
    all_vectors = np.memmap(attempt / 'witness.f32', dtype='<f4', mode='r', shape=(192, 2, 768))
    grouped = [{} for _ in cases]
    for event_index, ((case_index, arm), start, event, sample) in enumerate(zip(schedule, starts, events, samples)):
        case = cases[case_index]
        row = case[model_key]
        if (start['event_index'] != event_index or event['event_index'] != event_index
                or sample['event_index'] != event_index or start['case_index'] != case_index
                or event['case_index'] != case_index or start['case_id'] != case['case_id']
                or event['case_id'] != case['case_id'] or start['arm'] != arm or event['arm'] != arm
                or start['input_ids'] != row['input_ids'] or start['attention_mask'] != row['attention_mask']
                or start['subject_position'] != row['subject_position']
                or start['output_position'] != row['output_position']
                or event['hook_count'] != 1 or event['full_logits_finite'] is not True
                or event['returned_status'] != 'returned'
                or event['logits_offset'] != event_index * vocab * 4
                or event['witness_offset'] != event_index * 2 * 768 * 4):
            raise ValueError('event, start or tensor order mismatch')
        if not np.isfinite(all_logits[event_index]).all() or not np.isfinite(all_vectors[event_index]).all():
            raise ValueError('saved tensor is nonfinite')
        grouped[case_index][arm] = {'event': event, 'logits': all_logits[event_index],
                                    'before': all_vectors[event_index, 0], 'after': all_vectors[event_index, 1]}
    lookup = {(case['frame'], case['attractor_number'], case['lemma'], case['number']): case['index']
              for case in cases}
    rows = []
    for case, arms in zip(cases, grouped):
        index = case['index']
        if index != len(rows) or set(arms) != set(ARMS):
            raise ValueError('case identity or arms differ')
        base, clean = arms['clean']['before'], arms['clean']['logits']
        if not same_bits(base, arms['clean']['after']):
            raise ValueError('clean witness differs')
        opposite_index = lookup[(case['frame'], case['attractor_number'], case['lemma'], 1 - case['number'])]
        same_index = lookup[(case['frame'], case['attractor_number'], case['same_number_donor_lemma'], case['number'])]
        donors = {'opposite': grouped[opposite_index]['clean']['before'],
                  'same': grouped[same_index]['clean']['before']}
        base64 = base.astype(np.float64)
        donor_norms = {name: float(np.linalg.norm(value.astype(np.float64) - base64))
                       for name, value in donors.items()}
        q = min(2.0, donor_norms['opposite'], donor_norms['same'])
        if not math.isfinite(q) or q <= 0:
            raise ValueError('invalid replayed dose')
        for arm in ARMS[1:]:
            value = arms[arm]
            if (not same_bits(value['before'], base) or value['event']['requested_q'] != q
                    or value['event']['opposite_raw_norm'] != donor_norms['opposite']
                    or value['event']['same_raw_norm'] != donor_norms['same']):
                raise ValueError('source witness or requested dose differs')
            if arm in ('self', 'clean_after'):
                if not same_bits(value['after'], base) or not same_bits(value['logits'], clean):
                    raise ValueError('self or clean-after mismatch')
                continue
            if arm == 'random':
                if value['event']['random_seed'] != 20261029 + index or value['event']['donor_index'] is not None:
                    raise ValueError('random-control identity differs')
                raw = np.random.Generator(np.random.PCG64(20261029 + index)).standard_normal(base.shape)
                direction = (base64 + raw) - base64
            else:
                expected_donor = opposite_index if arm == 'opposite' else same_index
                if value['event']['donor_index'] != expected_donor:
                    raise ValueError('donor identity differs')
                direction = donors[arm].astype(np.float64) - base64
            intended = (q / np.linalg.norm(direction)) * direction
            expected = (base64 + intended).astype('<f4')
            if not same_bits(value['after'], expected):
                raise ValueError('intervention witness differs from frozen construction')
            actual = float(np.linalg.norm(value['after'].astype(np.float64) - base64))
            direction_error = float(np.linalg.norm((value['after'].astype(np.float64) - base64) - intended)) / q
            if (actual <= 0 or abs(actual - q) > 1e-4 * q or direction_error > 1e-4
                    or actual != value['event']['actual_l2']):
                raise ValueError('actual dose or dose receipt differs')
        scored = score_case(case, {arm: value['logits'] for arm, value in arms.items()}, TARGETS[model_key])
        scored['requested_q'] = q
        scored['opposite_raw_norm'] = donor_norms['opposite']
        scored['same_raw_norm'] = donor_norms['same']
        rows.append(scored)
    return {'schema': 'cross-model-agreement-replay-v1', 'model': model_key,
            'summary': summarize(rows), 'cases': rows,
            'scope': 'Selected vocabulary logits and selected block-output vectors only; '
                     'full-sequence finiteness and nonselected-position preservation are owner-reported.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exit-observation', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    if digest(args.cases) != CASE_SHA:
        raise ValueError('case file differs')
    metadata = json.loads((args.attempt / 'metadata.json').read_bytes())
    if (not re.fullmatch('[0-9a-f]{40}', args.source_commit)
            or metadata.get('source_commit') != args.source_commit or metadata.get('device') != args.device):
        raise ValueError('external source or device binding differs')
    for relative in ('scripts/cross_model_agreement_v1.py', 'docs/CROSS_MODEL_AGREEMENT_V1.md',
                     'experiments/cross_model_agreement_cases_v1.json'):
        blob = subprocess.run(['git', '-C', str(args.repo), 'show', args.source_commit + ':' + relative],
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout
        if hashlib.sha256(blob).hexdigest() != metadata['sources_sha256'][relative]:
            raise ValueError('saved source hash differs from external Git object')
    observation = json.loads(args.exit_observation.read_bytes())
    result = replay(args.attempt, json.loads(args.cases.read_bytes()), observation)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with args.output.open('xb') as stream:
        stream.write(data)
    print(json.dumps(result['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
