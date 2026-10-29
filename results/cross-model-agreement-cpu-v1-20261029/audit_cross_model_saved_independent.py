"""Post-closure, model-free replay of the fixed cross-model saved evidence.

This script checks saved bytes and cooperative direct-child observations. It
does not attest model execution, exposure history, or unrecorded positions.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


CASE_SHA = 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503'
TARGETS = {'gpt': {'is': 318, 'are': 389, 'was': 373, 'were': 547},
           'bert': {'is': 2003, 'are': 2024, 'was': 2001, 'were': 2020}}
VOCAB = {'gpt': 50257, 'bert': 30522}
ARMS = ('self', 'opposite', 'same', 'random', 'clean_after')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(raw):
    def obj(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('duplicate JSON key')
            result[k] = v
        return result
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=obj, parse_constant=bad)


def journal(path):
    raw = path.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('partial JSONL: ' + path.name)
    return [load(line) for line in raw.splitlines()]


def subnormal_guard():
    with np.errstate(under='ignore'):
        a = np.array([1], dtype=np.uint32).view(np.float32) * np.float32(1)
        b = np.array([1], dtype=np.uint64).view(np.float64) * np.float64(1)
        c = np.array([np.finfo(np.float32).tiny], np.float32) / np.float32(2)
        d = np.array([np.finfo(np.float64).tiny], np.float64) / np.float64(2)
    if not (a.view(np.uint32)[0] == 1 and b.view(np.uint64)[0] == 1
            and c.view(np.uint32)[0] == 1 << 22 and d.view(np.uint64)[0] == 1 << 51):
        raise RuntimeError('subnormal arithmetic is unavailable on the audit thread')


def bits(a):
    return np.asarray(a, dtype='<f4').tobytes()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-13, abs_tol=0.0)


def length(values):
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def distribution(base, arm):
    """Stable full-vocabulary TV/KL; unavailable if individual tails underflow."""
    a, b = [float(x) for x in base], [float(x) for x in arm]
    topa, topb = max(a), max(b)
    ids_a = [i for i, x in enumerate(a) if x == topa]
    ids_b = [i for i, x in enumerate(b) if x == topb]
    weights = [math.exp(x - topa) for x in a]
    weights_b = [math.exp(x - topb) for x in b]
    result = {'top_clean': ids_a[0], 'top_arm': ids_b[0],
              'top_clean_ties': len(ids_a), 'top_arm_ties': len(ids_b),
              'unique_top_changed': (ids_a[0] != ids_b[0]
                                     if len(ids_a) == len(ids_b) == 1 else None)}
    if not all(weights) or not all(weights_b):
        return {**result, 'tv': None, 'kl': None, 'unresolved': True}
    total = math.fsum(weights)
    p = [x / total for x in weights]
    total_b = math.fsum(weights_b)
    if not all(p) or any(x / total_b == 0 for x in weights_b):
        return {**result, 'tv': None, 'kl': None, 'unresolved': True}
    difference = [y - x for x, y in zip(a, b)]
    anchor = difference[ids_a[0]]
    rel = [x - anchor for x in difference]
    if max(map(abs, rel)) > 600:
        return {**result, 'tv': None, 'kl': None, 'unresolved': True}
    w = math.fsum(x * math.expm1(y) for x, y in zip(p, rel))
    if not math.isfinite(w) or w <= -1:
        return {**result, 'tv': None, 'kl': None, 'unresolved': True}
    logz = math.log1p(w)
    ratios = [x - logz for x in rel]
    tv = math.fsum(x * abs(math.expm1(y)) for x, y in zip(p, ratios)) / 2
    def kterm(x):
        if abs(x) < 1e-4:
            return x*x*(0.5 + x*(1/6 + x*(1/24 + x/120)))
        return max(0.0, math.expm1(x) - x)
    kl = math.fsum(x * kterm(y) for x, y in zip(p, ratios))
    return {**result, 'tv': tv, 'kl': kl, 'unresolved': False}


def audit(attempt, cases_path, receipt_path, receipt_sha, complete_sha, *, vocab_override=None,
          target_override=None, expected_case_sha=CASE_SHA, source_commit=None, repo=None):
    subnormal_guard()
    attempt, cases_path, receipt_path = map(Path, (attempt, cases_path, receipt_path))
    if sha(cases_path) != expected_case_sha or sha(receipt_path) != receipt_sha:
        raise ValueError('external case or receipt pin differs')
    if sha(attempt / 'complete.json') != complete_sha or (attempt / 'failure.txt').exists():
        raise ValueError('completion pin differs or a failure is present')
    manifest = load(cases_path.read_bytes())
    metadata = load((attempt / 'metadata.json').read_bytes())
    complete = load((attempt / 'complete.json').read_bytes())
    receipt = load(receipt_path.read_bytes())
    model = metadata['model']
    if model not in VOCAB or metadata.get('case_sha256') != expected_case_sha:
        raise ValueError('model or manifest identity differs')
    if source_commit is not None:
        if (repo is None or metadata.get('source_commit') != source_commit or metadata.get('device') != 'cpu'
                or metadata.get('base_interpreter_direct') is not True or metadata.get('isolated') is not True):
            raise ValueError('source, CPU or interpreter metadata differs')
        source_paths = ('scripts/cross_model_agreement_v1.py', 'docs/CROSS_MODEL_AGREEMENT_V1.md',
                        'experiments/cross_model_agreement_cases_v1.json')
        source_bytes = {}
        for name in source_paths:
            raw = subprocess.run(['git', '-C', str(repo), 'show', source_commit + ':' + name],
                                 check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout
            if hashlib.sha256(raw).hexdigest() != metadata['sources_sha256'].get(name):
                raise ValueError('recorded source hash differs from pinned Git bytes')
            source_bytes[name] = raw
        if set(metadata['sources_sha256']) != set(source_paths):
            raise ValueError('recorded source set differs')
        tree = ast.parse(source_bytes[source_paths[0]])
        values = [ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ASSET_SHA'
                                                          for t in node.targets)]
        if len(values) != 1 or metadata.get('assets_sha256') != values[0][model]:
            raise ValueError('recorded asset identities differ from source pins')
    if (complete.get('status') != 'complete' or complete.get('events') != 192
            or complete.get('metadata_sha256') != sha(attempt / 'metadata.json')
            or receipt.get('schema') != 'cross-model-direct-child-exit-v1'
            or receipt.get('status') != 'complete' or type(receipt.get('exit_code')) is not int
            or receipt['exit_code'] != 0 or receipt.get('interrupted') is not None
            or type(receipt.get('child_pid')) is not int or receipt['child_pid'] <= 0
            or receipt['child_pid'] != metadata.get('pid')
            or receipt.get('source_commit') != metadata.get('source_commit')
            or receipt.get('attempt_name') != attempt.name
            or receipt.get('complete_sha256') != complete_sha
            or receipt.get('metadata_sha256') != sha(attempt / 'metadata.json')):
        raise ValueError('direct-child exit or terminal identity differs')
    files = {'starts.jsonl': 'starts_sha256', 'events.jsonl': 'events_sha256',
             'memory.jsonl': 'memory_sha256', 'retention-errors.jsonl': 'retention_errors_sha256',
             'logits.f32': 'logits_sha256', 'witness.f32': 'witness_sha256'}
    for path, key in files.items():
        if sha(attempt / path) != complete.get(key):
            raise ValueError('saved file digest differs: ' + path)
    if (attempt / 'retention-errors.jsonl').read_bytes():
        raise ValueError('returned evidence has a retention error')
    starts, events, memory = [journal(attempt / name) for name in
                              ('starts.jsonl', 'events.jsonl', 'memory.jsonl')]
    cases = manifest['cases']
    if len(cases) != 32 or len(starts) != 192 or len(events) != 192 or len(memory) != 192:
        raise ValueError('incomplete schedule')
    vocab = vocab_override or VOCAB[model]
    targets = target_override or TARGETS[model]
    if ((attempt / 'logits.f32').stat().st_size != 192 * vocab * 4
            or (attempt / 'witness.f32').stat().st_size != 192 * 2 * 768 * 4):
        raise ValueError('raw tensor lengths differ')
    logits = np.memmap(attempt / 'logits.f32', dtype='<f4', mode='r', shape=(192, vocab))
    vectors = np.memmap(attempt / 'witness.f32', dtype='<f4', mode='r', shape=(192, 2, 768))
    schedule = [(i, 'clean') for i in range(32)] + [(i, arm) for i in range(32) for arm in ARMS]
    actions = [{} for _ in cases]
    for n, ((i, arm), start, event, sample) in enumerate(zip(schedule, starts, events, memory)):
        case, row = cases[i], cases[i][model]
        if (type(start.get('event_index')) is not int or start['event_index'] != n
                or type(event.get('event_index')) is not int or event['event_index'] != n
                or sample.get('event_index') != n or start.get('case_index') != i
                or event.get('case_index') != i or start.get('case_id') != case['case_id']
                or event.get('case_id') != case['case_id'] or start.get('arm') != arm
                or event.get('arm') != arm or start.get('input_ids') != row['input_ids']
                or start.get('attention_mask') != row['attention_mask']
                or start.get('subject_position') != row['subject_position']
                or start.get('output_position') != row['output_position']
                or event.get('hook_count') != 1 or event.get('full_logits_finite') is not True
                or event.get('returned_status') != 'returned'
                or event.get('logits_offset') != n * vocab * 4
                or event.get('witness_offset') != n * 2 * 768 * 4
                or not np.isfinite(logits[n]).all() or not np.isfinite(vectors[n]).all()):
            raise ValueError('action or witness binding differs at ' + str(n))
        actions[i][arm] = (event, logits[n], vectors[n, 0], vectors[n, 1])
    lookup = {(c['frame'], c['attractor_number'], c['lemma'], c['number']): c['index'] for c in cases}
    if len(lookup) != 32:
        raise ValueError('case keys are not unique')
    rows = []
    for i, case in enumerate(cases):
        if case['index'] != i or set(actions[i]) != {'clean', *ARMS}:
            raise ValueError('case/arm set differs')
        action = actions[i]
        _, clean, base, after = action['clean']
        if bits(base) != bits(after):
            raise ValueError('clean capture changed the site')
        opp = lookup[(case['frame'], case['attractor_number'], case['lemma'], 1 - case['number'])]
        same = lookup[(case['frame'], case['attractor_number'], case['same_number_donor_lemma'], case['number'])]
        donor = {'opposite': actions[opp]['clean'][2], 'same': actions[same]['clean'][2]}
        a = base.astype(np.float64)
        d = {k: v.astype(np.float64) - a for k, v in donor.items()}
        norm = {k: length(v) for k, v in d.items()}
        q = min(2.0, norm['opposite'], norm['same'])
        if not math.isfinite(q) or q <= 0:
            raise ValueError('invalid donor dose')
        for arm in ARMS:
            event, output, before, actual = action[arm]
            if (bits(before) != bits(base) or not close(event['requested_q'], q)
                    or not close(event['opposite_raw_norm'], norm['opposite'])
                    or not close(event['same_raw_norm'], norm['same'])):
                raise ValueError('source or donor norm differs')
            if arm in ('self', 'clean_after'):
                if bits(actual) != bits(base) or bits(output) != bits(clean):
                    raise ValueError('self/no-op control differs')
                continue
            if arm == 'random':
                if event['donor_index'] is not None or event['random_seed'] != 20261029 + i:
                    raise ValueError('random donor identity differs')
                raw = np.random.Generator(np.random.PCG64(20261029 + i)).standard_normal(base.shape)
                direction = (a + raw) - a
                scale_norm = float(np.linalg.norm(direction))
            else:
                if event['donor_index'] != (opp if arm == 'opposite' else same):
                    raise ValueError('donor identity differs')
                direction = d[arm]
                scale_norm = float(event[arm + '_raw_norm'])
            intended = direction * (float(event['requested_q']) / scale_norm)
            planned = (a + intended).astype('<f4')
            if bits(planned) != bits(actual):
                raise ValueError('recorded f32 landing differs')
            actual_vector = actual.astype(np.float64) - a
            actual_norm = length(actual_vector)
            vector_error = length(actual_vector - intended) / q
            if (not close(event['actual_l2'], actual_norm) or actual_norm <= 0
                    or abs(actual_norm - q) / q > 1e-4 or vector_error > 1e-4):
                raise ValueError('delivered dose or direction differs')
        def margin(v, x, y):
            return float(v[targets[x]]) - float(v[targets[y]])
        clean_margin = margin(clean, 'is', 'are')
        sign = 1 if case['number'] == 1 else -1
        effects = {arm: sign * (margin(action[arm][1], 'is', 'are') - clean_margin)
                   for arm in ('opposite', 'same', 'random')}
        rows.append({'case_id': case['case_id'], 'frame': case['frame'], 'lemma': case['lemma'],
                     'number': case['number'], 'attractor_number': case['attractor_number'],
                     'clean_margin': clean_margin, 'clean_was_were': margin(clean, 'was', 'were'),
                     'was_were_by_arm': {arm: margin(action[arm][1], 'was', 'were')
                                         for arm in ('opposite', 'same', 'random')},
                     'clean_correct': clean_margin > 0 if case['number'] == 0 else clean_margin < 0,
                     'q': q, 'effects': effects,
                     'opposite_minus_same': effects['opposite'] - effects['same'],
                     'opposite_minus_random': effects['opposite'] - effects['random'],
                     'distribution': {arm: distribution(clean, action[arm][1])
                                      for arm in ('opposite', 'same', 'random')}})
    ns = math.fsum(row['opposite_minus_same'] for row in rows) / 32
    nr = math.fsum(row['opposite_minus_random'] for row in rows) / 32
    both = sum(row['opposite_minus_same'] > 0 and row['opposite_minus_random'] > 0 for row in rows)
    correct = sum(row['clean_correct'] for row in rows)
    subnormal_guard()
    return {'model': model, 'source_commit': metadata['source_commit'], 'receipt_sha256': receipt_sha,
            'complete_sha256': complete_sha, 'actions': 192, 'cases': rows,
            'summary': {'correct_clean': correct, 'baseline_limited': correct < 24,
                        'mean_opposite_minus_same': ns, 'mean_opposite_minus_random': nr,
                        'positive_against_both': both,
                        'descriptive_prediction_passes': ns > 0 and nr > 0 and both >= 24},
            'scope': 'Post-closure saved-byte consistency and recorded asset/source identities; '
                     'no model execution, loaded-asset, whole-sequence finiteness or nonselected-position attestation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'cases', 'receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--complete-sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.attempt, args.cases, args.receipt, args.receipt_sha256, args.complete_sha256,
                   source_commit=args.source_commit, repo=args.repo)
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short report write')
    print(json.dumps({'sha256': hashlib.sha256(data).hexdigest(), 'summary': report['summary']}, sort_keys=True))


if __name__ == '__main__':
    main()
