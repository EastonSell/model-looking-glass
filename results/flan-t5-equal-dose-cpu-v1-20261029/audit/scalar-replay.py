"""Independent stdlib replay of saved FLAN-T5 matched-dose evidence.

This checks saved-byte consistency and a binary64 diagnostic, not execution,
secrecy, generalization, or the whole intervention lifecycle. The actual
decisions use the recorded binary32 per-token scores, as the protocol requires.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(b):
    return hashlib.sha256(b).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def floats(b):
    require(len(b) % 4 == 0, 'unaligned float payload')
    result = struct.unpack('<%df' % (len(b) // 4), b)
    require(all(math.isfinite(x) for x in result), 'nonfinite float payload')
    return result


def norm(a, b):
    require(len(a) == len(b), 'dose vector size differs')
    return math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b)))


def audit(root, case_path, expected_completed_sha, expected_commit):
    case_bytes = case_path.read_bytes()
    case_hash = 'a0e320c9d24a8940fbca05cbc4ed6e3f2661b5b44b2e62e1140832c0e99e7cd6'
    require(sha(case_bytes) == case_hash, 'fixed cases do not match')
    manifest = json.loads(case_bytes)
    completed_bytes = (root / 'completed.json').read_bytes()
    require(sha(completed_bytes) == expected_completed_sha, 'completed hash mismatch')
    completed = json.loads(completed_bytes)
    require(completed['status'] == 'complete' and completed['source']['commit'] == expected_commit,
            'unexpected status or source')
    require(completed['case_manifest_sha256'] == case_hash, 'case hash mismatch')
    cases = manifest['cases']
    summaries = completed['cases']
    require(len(cases) == len(summaries) == 6, 'wrong case count')
    require(not (root / 'failure.json').exists(), 'failure record present')
    conditions = ('repeat', 'null_word', 'self_word', 'self_instruction', 'self_punctuation',
                  'full_self', 'full_swap', 'full_word_swap', 'zero_word',
                  'matched_word', 'matched_instruction', 'matched_punctuation')
    expected = {}
    for case, summary in zip(cases, summaries):
        name = case['case_id']
        require(summary['case_id'] == name and summary['pair_id'] == case['pair_id']
                and summary['template_id'] == case['template_id'], 'case identity differs')
        if summary['status'] == 'alignment-ineligible':
            continue
        for target in range(2):
            expected[f'{name}-context-{target}'] = (case['neutral'], case['targets'][target])
        for side in range(2):
            for target in range(2):
                expected[f'{name}-base-{side}-{target}'] = (case['source'][side], case['targets'][target])
        if summary['status'] in ('positive', 'negative'):
            for condition in conditions:
                for side in range(2):
                    for target in range(2):
                        expected[f'{name}-{condition}-{side}-{target}'] = (case['source'][side], case['targets'][target])
    require(len(expected) == completed['forwards'] <= 324, 'forward count mismatch')
    rows = {}
    bases = {}
    max_difference = 0.0
    max_detail = None
    for index in range(completed['forwards']):
        row = json.loads((root / f'forward-{index:03d}.json').read_bytes())
        name = row['name']
        require(row['index'] == index and row['file'] == f'forward-{index:03d}.bin'
                and name in expected and name not in rows, 'unexpected or duplicate forward')
        source, target = expected[name]
        ids = target['input_ids']
        require(row['source_ids'] == source['input_ids'] and row['source_mask'] == source['attention_mask']
                and row['target_ids'] == ids and ids[-1] == 1 and ids.count(1) == 1
                and row['decoder_ids'] == [0] + ids[:-1] and row['decoder_mask'] == [1] * len(ids),
                'input or decoder binding differs')
        raw = (root / row['file']).read_bytes()
        require(sha(raw) == row['sha256'], 'raw hash mismatch')
        info = row['tensor_layout']['logits']
        require(info['available'] and info['shape'] == [1, len(ids), 32128]
                and info['bytes'] == len(ids) * 32128 * 4, 'logit shape differs')
        logits = floats(raw[info['offset']:info['offset']+info['bytes']])
        per_token = row['selected_logprobs']
        require(len(per_token) == len(ids) and math.fsum(per_token) == row['score'], 'stored score differs')
        for position, token_id in enumerate(ids):
            values = logits[position*32128:(position+1)*32128]
            maximum = max(values)
            value = values[token_id] - maximum - math.log(math.fsum(math.exp(x-maximum) for x in values))
            difference = abs(value - per_token[position])
            if difference > max_difference:
                max_difference, max_detail = difference, [index, position]
            require(difference <= 1e-3, 'binary64 diagnostic disagrees with recorded float32 score')
        if '-base-' in name:
            info = row['tensor_layout']['encoder']
            require(info['available'] and info['shape'] == [1, len(source['input_ids']), 512]
                    and info['bytes'] == len(source['input_ids'])*512*4, 'encoder shape differs')
            bases[name] = raw[info['offset']:info['offset']+info['bytes']]
        rows[name] = row['score']
    require(set(rows) == set(expected), 'forward set differs')
    reports = []
    for index, (case, summary) in enumerate(zip(cases, summaries)):
        name, status = case['case_id'], summary['status']
        if status == 'alignment-ineligible':
            reports.append({'case_id': name, 'status': status})
            continue
        context = [rows[f'{name}-context-{t}'] for t in range(2)]
        margins = [rows[f'{name}-base-{s}-0'] - rows[f'{name}-base-{s}-1'] for s in range(2)]
        delta = margins[0] - margins[1]
        require(context == summary['context_scores'] and context[0]-context[1] == summary['context_margin']
                and margins == summary['baseline_margins'] and delta == summary['delta'], 'baseline differs')
        supported = margins[0] > 0 and margins[1] < 0 and delta >= 1
        if not supported:
            require(status == 'baseline-unsupported', 'unsupported baseline mislabeled')
            reports.append({'case_id': name, 'status': status, 'baseline_margins': margins})
            continue
        require(status in ('positive', 'negative', 'dose-ineligible', 'dose-unmatchable'), 'wrong supported status')
        dose_record = json.loads((root / f'dose-{index:02d}.json').read_bytes())
        require(dose_record['case_id'] == name and dose_record['dose'] == summary['dose'], 'dose record differs')
        dose_raw = (root / dose_record['file']).read_bytes()
        require(sha(dose_raw) == dose_record['sha256'], 'dose payload hash differs')
        vectors = {}
        offset = 0
        for entry in dose_record['layout']:
            key = entry['site'], entry['side'], entry['role']
            require(key not in vectors and entry['offset'] == offset and entry['shape'] == [512]
                    and entry['bytes'] == 2048 and entry['dtype'] == 'little-endian-float32', 'dose layout differs')
            block = dose_raw[offset:offset+2048]
            vectors[key] = (block, floats(block))
            offset += 2048
        require(offset == len(dose_raw), 'unreferenced dose bytes')
        sites = {'word': case['word_position'], 'instruction': case['instruction_position'],
                 'punctuation': case['context_position']}
        distances, achieved = {}, {}
        for site, position in sites.items():
            for side in range(2):
                own = bases[f'{name}-base-{side}-0'][position*2048:(position+1)*2048]
                donor = bases[f'{name}-base-{1-side}-0'][position*2048:(position+1)*2048]
                require(vectors[site, side, 'own'][0] == own and vectors[site, side, 'donor'][0] == donor,
                        'dose vector not tied to clean encoder state')
            distances[site] = norm(vectors[site, 0, 'own'][1], vectors[site, 0, 'donor'][1])
            require(close(distances[site], summary['dose']['distances'][site]), 'distance differs')
        if any(x <= 1e-6 for x in distances.values()):
            require(status == 'dose-ineligible', 'ineligible dose mislabeled')
            reports.append({'case_id': name, 'status': status, 'baseline_margins': margins})
            continue
        q = min(distances.values())
        tolerance = min(max(1e-5, 1e-5*q), .01*q)
        require(close(q, summary['dose']['q']) and close(tolerance, summary['dose']['tolerance']), 'q differs')
        for site in sites:
            fraction = q / distances[site]
            require(close(fraction, summary['dose']['fractions'][site]), 'fraction differs')
            for side in range(2):
                own = vectors[site, side, 'own'][1]
                donor = vectors[site, side, 'donor'][1]
                delivered, values = vectors[site, side, 'delivered']
                reconstructed = b''.join(struct.pack('<f', a+fraction*(b-a)) for a, b in zip(own, donor))
                require(delivered == reconstructed, 'rounded delivery differs')
                actual = norm(values, own)
                achieved[f'{site}-{side}'] = actual
                require(close(actual, summary['dose']['achieved'][f'{site}-{side}']), 'achieved dose differs')
        matched = (all(x > 0 and abs(x-q) <= tolerance for x in achieved.values())
                   and all(abs(achieved[f'word-{s}']-achieved[f'{site}-{s}']) <= .01*achieved[f'word-{s}']
                           for s in range(2) for site in ('instruction', 'punctuation')))
        if not matched:
            require(status == 'dose-unmatchable', 'unmatchable dose mislabeled')
            reports.append({'case_id': name, 'status': status, 'baseline_margins': margins, 'q': q})
            continue
        require(status in ('positive', 'negative'), 'matchable dose mislabeled')
        movements = {}
        for condition in conditions:
            altered = [rows[f'{name}-{condition}-{s}-0']-rows[f'{name}-{condition}-{s}-1'] for s in range(2)]
            move = [margins[0]-altered[0], altered[1]-margins[1]]
            require(altered == summary['condition_margins'][condition]
                    and move == summary['signed_movements'][condition]
                    and [abs(x) for x in move] == summary['absolute_movements'][condition], 'movement differs')
            movements[condition] = move
        positive = all(movements['matched_word'][s] >= .20*delta and
                       all(movements['matched_word'][s] >= abs(movements[c][s])+.05*delta
                           for c in ('matched_instruction', 'matched_punctuation')) for s in range(2))
        require(status == ('positive' if positive else 'negative'), 'cell decision differs')
        reports.append({'case_id': name, 'status': status, 'baseline_margins': margins, 'delta': delta,
                        'q': q, 'achieved': achieved, 'matched_movements':
                        {k: movements[k] for k in ('matched_word', 'matched_instruction', 'matched_punctuation')}})
    pair_status = []
    for pair in range(3):
        pair_rows = [s['status'] for s in summaries if s['pair_id'] == pair]
        if len(pair_rows) == 2 and all(x in ('positive', 'negative') for x in pair_rows):
            pair_status.append('positive' if all(x == 'positive' for x in pair_rows) else 'negative')
        else:
            pair_status.append('unevaluable')
    aggregate = ('positive' if pair_status.count('positive') >= 2 else
                 'negative' if pair_status.count('negative') >= 2 else 'inconclusive')
    require(aggregate == completed['decision'], 'aggregate differs')
    return {'status': 'passed', 'scope': 'saved-byte scalar consistency only; not proof of execution or complete control landing',
            'completed_sha256': expected_completed_sha, 'source_commit': expected_commit,
            'case_sha256': case_hash, 'forwards': len(rows), 'f64_diagnostic_tolerance': 1e-3,
            'maximum_f64_vs_f32_per_token_difference': max_difference, 'maximum_index_and_position': max_detail,
            'pair_status': pair_status, 'aggregate': aggregate, 'cases': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--completed-sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.cases, args.completed_sha256, args.source_commit),
                     sort_keys=True, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
