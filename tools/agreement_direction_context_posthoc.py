"""Post-hoc, model-free analysis of the fixed agreement-direction saved tensors.

This reads an exact published journal and content-addressed witnesses. It does
not load a model, authenticate the recorded execution, or turn the public cases
into an independent test. See the accompanying report for interpretation.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from looking_glass import agreement_science_saved_v1 as saved
from looking_glass import agreement_science_v1 as science


JOURNAL_SHA256 = '7c8be25d480d986b087052685720ba7d2813c53d27509dcc81e67f9e2049eca6'
TOKENIZER_SHA256 = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
STUDY_COMMIT = '7b28ab29817bc372a2104c155e406f8d85984a41'
BINDING = {'prelaunch_sha256': 'c367112faf54cf8321b15059fbc8bb65460f3f83c0b52c7e9eb58a81dad49e33',
           'reservation_sha256': '3df696303a06b2ec6ddf4113df1969897dbae89566375b7d3a9cecb85b3d59cd',
           'latest_exposure_sha256': '2549f1430ada47da088d2a909f8aa57ad5c08bf7c6d09f187bd34bb76b99f658'}
VOCABULARY = 50257
WIDTH = 768
DIRECTIONS = ('u', 'v1', 'v2')
MARGINS = {'is_minus_are': (318, 389), 'was_minus_were': (373, 547),
           'the_minus_a': (262, 257), 'in_minus_on': (287, 319)}


def decode_f32(raw, count):
    """Decode finite IEEE binary32 bits into binary64, including subnormals."""
    if type(raw) is not bytes or type(count) is not int or count < 1 or len(raw) != 4 * count:
        raise ValueError('invalid float32 payload')
    words = np.frombuffer(raw, dtype='<u4')
    exponent = ((words >> 23) & 255).astype(np.int32)
    mantissa = words & 0x7fffff
    if np.any(exponent == 255):
        raise ValueError('nonfinite float32 payload')
    significand = np.where(exponent == 0, mantissa, mantissa | 0x800000).astype(np.float64)
    powers = np.where(exponent == 0, -149, exponent - 150)
    values = np.ldexp(significand, powers)
    values = np.copysign(values, np.where(words >> 31, -1.0, 1.0))
    return values, int(np.count_nonzero((exponent == 0) & (mantissa != 0)))


def distribution(logits):
    """Return probabilities and log-probabilities with a shifted softmax."""
    if logits.ndim != 1 or not np.all(np.isfinite(logits)):
        raise ValueError('invalid logit vector')
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('invalid softmax denominator')
    return exponentials / denominator, shifted - math.log(denominator)


def distribution_change(clean, arm):
    p, lp = distribution(clean)
    q, lq = distribution(arm)
    differences = q - p
    l1 = float(np.sum(np.abs(differences), dtype=np.float64))
    share = float((abs(differences[318]) + abs(differences[389])) / l1) if l1 else 0.0
    return {'tv': l1 / 2.0, 'kl_clean_to_arm': float(np.sum(p * (lp - lq), dtype=np.float64)),
            'is_are_l1_share': share, 'top_increase_id': int(np.argmax(differences)),
            'top_decrease_id': int(np.argmin(differences)), 'top_clean_id': int(np.argmax(clean)),
            'top_arm_id': int(np.argmax(arm)), 'delta': differences,
            'clean_p': p, 'arm_p': q}


def odd_and_even(clean, plus, minus, q):
    if not math.isfinite(q) or q <= 0:
        raise ValueError('finite positive dose required')
    return (plus - minus) / (2 * q), (plus + minus) / 2 - clean


def centered_unit(vector):
    """Unit vector after removing a uniform vocabulary-wide logit offset."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError('invalid vocabulary response')
    centered = vector - np.mean(vector, dtype=np.float64)
    norm = math.sqrt(float(np.sum(centered * centered, dtype=np.float64)))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('zero or nonfinite centered response')
    return centered / norm


def require_same(actual, expected, message):
    if not science.same_recorded_value(actual, expected):
        raise ValueError(message)


def load_evidence(root, attempt):
    cases, directions = saved.load_frozen_inputs(
        root / 'experiments/distilgpt2_agreement_direction_cases_v1.json',
        root / 'results/distilgpt2-agreement-direction-training-v1')
    raw = saved.pinned_file(attempt / 'events.jsonl', JOURNAL_SHA256, limit=8 * 1024 * 1024)
    records = [saved.strict_json(line) for line in raw.splitlines()]
    science.validate_journal_layout(records, cases)
    saved.validate_recorded_runtime(records[0]['event'])
    require_same(records[0]['event']['external_binding'], BINDING, 'recorded study binding differs')
    by_id = {case['case_id']: case for case in cases}
    tensor_dir = attempt / 'tensors'
    captures, clean, doses, logits = {}, {}, {}, {}
    subnormals = 0
    read_logit_events = 0

    for record in records[1:]:
        event, refs = record['event'], record['tensors']
        if event['kind'] == 'family_geometry':
            family = event['family_index']
            members = cases[4 * family:4 * family + 4]
            expected = science.family_geometry([captures[row['case_id']] for row in members], directions['u'])
            require_same({k: v for k, v in event.items() if k not in ('kind', 'family_index')},
                         expected, 'family geometry differs')
            for i, case in enumerate(members):
                doses[case['case_id']] = expected['doses'][i]
            doses[cases[32 + family]['case_id']] = math.fsum((expected['doses'][0], expected['doses'][2])) / 2
            continue
        if event['kind'] != 'forward':
            continue
        case, action = by_id[event['case_id']], event['action']
        name, length = case['case_id'], len(case['token_ids'])
        relevant = action in ('clean', 'clean_repeat', 'clean_after', 'noop_l3_final') or action in {
            f'{direction}_{sign}' for direction in DIRECTIONS for sign in ('plus', 'minus')}
        if not relevant:
            continue
        raw_logits = science.read_saved_tensor(tensor_dir, refs['last_logits'], (VOCABULARY,))
        values, count = decode_f32(raw_logits, VOCABULARY)
        subnormals += count
        read_logit_events += 1
        if action == 'clean':
            clean[name] = raw_logits
            logits[name, action] = values
        elif action in ('clean_repeat', 'clean_after'):
            if clean.get(name) != raw_logits:
                raise ValueError('clean repeat differs in full-logit bits')
        else:
            site, plan = science.expected_hook_plan(action, sequence_length=length, width=WIDTH)
            require_same(event['site'], site, 'hook site differs')
            require_same(event['plan_metadata'], plan, 'hook plan differs')
            before = science.read_saved_tensor(tensor_dir, refs['before'], (WIDTH,))
            after = science.read_saved_tensor(tensor_dir, refs['after'], (WIDTH,))
            whole_before = science.read_saved_tensor(tensor_dir, refs['whole_before'], (1, length, WIDTH))
            whole_after = science.read_saved_tensor(tensor_dir, refs['whole_after'], (1, length, WIDTH))
            science.verify_hook_witness(before, after, whole_before, whole_after, width=WIDTH,
                                        sequence_length=length, token_index=length - 1)
            if action == 'noop_l3_final':
                if before != after or clean.get(name) != raw_logits or event['dose'] is not None or event['donor'] is not None:
                    raise ValueError('no-op differs from clean')
                captures[name] = before
            else:
                direction, sign = action.rsplit('_', 1)
                q = doses[name]
                if not math.isfinite(q) or q <= 0 or before != captures[name] or event['donor'] is not None:
                    raise ValueError('invalid intervention baseline or dose')
                landed, expected_dose = science.direction_delivery(before, directions[direction],
                                                                   q if sign == 'plus' else -q)
                if after != landed or expected_dose['dose_valid'] is not True:
                    raise ValueError('intervention landing differs')
                require_same(event['dose'], expected_dose, 'intervention dose differs')
                logits[name, action] = values
    if len(captures) != 40 or len(doses) != 40 or len(logits) != 280 or read_logit_events != 400:
        raise ValueError('relevant evidence is incomplete')
    return cases, doses, logits, {'read_logit_events': read_logit_events, 'subnormal_logit_values': subnormals,
                                  'signed_intervention_witnesses': 240}


def analyze(cases, doses, logits, tokenizer):
    rows, vectors = [], {}
    token_names = {i: tokenizer.decode([i], skip_special_tokens=False) for i in range(VOCABULARY)}
    for case in cases:
        name, q = case['case_id'], doses[case['case_id']]
        clean = logits[name, 'clean']
        for direction in DIRECTIONS:
            plus, minus = logits[name, direction + '_plus'], logits[name, direction + '_minus']
            normalized, even = odd_and_even(clean, plus, minus, q)
            vectors[name, direction] = centered_unit(normalized)
            row = {'case_id': name, 'text': case['text'], 'role': case['role'],
                   'family_index': case['family_index'], 'subject_number': case.get('subject_number'),
                   'attractor_number': case['attractor_number'], 'relation': case['relation'],
                   'token_count': len(case['token_ids']), 'direction': direction, 'q': q}
            for label, (a, b) in MARGINS.items():
                row[label + '_clean'] = float(clean[a] - clean[b])
                row[label + '_odd'] = float((plus[a] - plus[b] - minus[a] + minus[b]) / 2)
                row[label + '_odd_per_q'] = float(normalized[a] - normalized[b])
                row[label + '_even'] = float(even[a] - even[b])
            for sign, arm in (('plus', plus), ('minus', minus)):
                change = distribution_change(clean, arm)
                for field in ('tv', 'kl_clean_to_arm', 'is_are_l1_share', 'top_clean_id', 'top_arm_id',
                              'top_increase_id', 'top_decrease_id'):
                    row[sign + '_' + field] = change[field]
                for token in (318, 389, 373, 547):
                    row[sign + f'_p_{token}'] = float(change['arm_p'][token])
                    row[f'clean_p_{token}'] = float(change['clean_p'][token])
                for kind in ('top_increase', 'top_decrease'):
                    token = change[kind + '_id']
                    row[sign + '_' + kind + '_text'] = token_names[token]
                    row[sign + '_' + kind + '_probability_change'] = float(change['delta'][token])
            rows.append(row)

    pairs = []
    for direction in DIRECTIONS:
        for left, right in itertools.combinations(cases, 2):
            group = ('context-context' if left['role'] == right['role'] == 'context' else
                     'target-target' if left['role'] == right['role'] == 'target' else 'target-context')
            cosine = float(np.sum(vectors[left['case_id'], direction] * vectors[right['case_id'], direction],
                                  dtype=np.float64))
            pairs.append({'direction': direction, 'group': group, 'left_case_id': left['case_id'],
                          'right_case_id': right['case_id'], 'cosine': cosine})
    return rows, pairs


def summarize(cases, rows, pairs, evidence):
    def stats(values):
        values = list(values)
        if not values:
            raise ValueError('empty summary group')
        return {'n': len(values), 'mean': math.fsum(values) / len(values),
                'min': min(values), 'max': max(values)}

    groups = {'context': lambda r: r['role'] == 'context',
              'target_singular': lambda r: r['role'] == 'target' and r['subject_number'] == 'singular',
              'target_plural': lambda r: r['role'] == 'target' and r['subject_number'] == 'plural',
              'context_attractor_singular': lambda r: r['role'] == 'context' and r['attractor_number'] == 'singular',
              'context_attractor_plural': lambda r: r['role'] == 'context' and r['attractor_number'] == 'plural'}
    responses = {group: {direction: {label: stats(r[label + '_odd_per_q'] for r in rows
                    if r['direction'] == direction and include(r)) for label in MARGINS}
                    for direction in DIRECTIONS} for group, include in groups.items()}
    grouped_pairs = {direction: {group: stats(r['cosine'] for r in pairs
                    if r['direction'] == direction and r['group'] == group)
                    for group in ('context-context', 'target-target', 'target-context')}
                    for direction in DIRECTIONS}
    by_id = {r['case_id']: r for r in rows if r['direction'] == 'u'}
    comparisons = []
    for family in range(8):
        for variant in (0, 2):
            singular, plural = cases[4 * family + variant:4 * family + variant + 2]
            s, p = by_id[singular['case_id']], by_id[plural['case_id']]
            if s['q'] != p['q']:
                raise ValueError('matched pair doses differ')
            comparisons.append(s['is_minus_are_odd_per_q'] > p['is_minus_are_odd_per_q'])
    context = [r for r in rows if r['role'] == 'context' and r['direction'] == 'u']
    return {'analysis': 'post-hoc saved-logit analysis; not a new model run',
            'study_commit': STUDY_COMMIT, 'journal_sha256': JOURNAL_SHA256,
            'case_sha256': saved._CASE_SHA, 'direction_sha256': saved._VECTOR_SHA,
            'tokenizer_sha256': TOKENIZER_SHA256, 'evidence': evidence,
            'normalized_margin_responses': responses, 'centered_vocabulary_cosines': grouped_pairs,
            'same_lemma_target_pairs': {'n': len(comparisons), 'singular_greater_count': sum(comparisons)},
            'context_u_plus_tv': stats(r['plus_tv'] for r in context),
            'context_u_plus_is_are_l1_share': stats(r['plus_is_are_l1_share'] for r in context),
            'context_u_minus_tv': stats(r['minus_tv'] for r in context),
            'context_u_plus_is_probability_increased_count': sum(r['plus_p_318'] > r['clean_p_318'] for r in context),
            'context_u_plus_are_probability_decreased_count': sum(r['plus_p_389'] < r['clean_p_389'] for r in context)}


def write_csv(path, rows):
    if not rows:
        raise ValueError('no output rows')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--journal-directory', required=True)
    parser.add_argument('--tokenizer-json', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args(argv)
    root, attempt, output = Path(args.repo_root), Path(args.journal_directory), Path(args.output_directory)
    tokenizer_bytes = saved.pinned_file(Path(args.tokenizer_json), TOKENIZER_SHA256, limit=8 * 1024 * 1024)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
    if tokenizer.get_vocab_size() != VOCABULARY:
        raise ValueError('tokenizer vocabulary differs')
    cases, doses, logits, evidence = load_evidence(root, attempt)
    rows, pairs = analyze(cases, doses, logits, tokenizer)
    summary = summarize(cases, rows, pairs, evidence)
    output.mkdir(parents=False, exist_ok=False)
    write_csv(output / 'per-case.csv', rows)
    write_csv(output / 'pairwise-vocabulary-cosines.csv', pairs)
    raw = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    (output / 'summary.json').write_bytes(raw)
    print(json.dumps({'status': 'posthoc_complete', 'summary_sha256': hashlib.sha256(raw).hexdigest(),
                      'case_direction_rows': len(rows), 'pairwise_rows': len(pairs)}, sort_keys=True))


if __name__ == '__main__':
    main()
