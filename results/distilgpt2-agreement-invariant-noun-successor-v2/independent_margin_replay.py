"""Recompute the invariant study's fixed contrasts from saved logit bytes only.

This independent descriptive replay is not an execution, custody, full-vocabulary,
or hook-witness audit. Run only on a closed attempt and externally pinned files.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

AXES = ('u', 'v1', 'v2')
PAIRS = {'present': (318, 389), 'past': (373, 547),
         'the_vs_a': (262, 257), 'in_vs_on': (287, 319)}
ACTIONS = {'clean', 'noop', 'clean_after'} | {
    f'{axis}_{sign}' for axis in AXES for sign in ('plus', 'minus')}


def verified_bytes(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('independent input digest differs')
    return raw


def analyse(cases, records, tensors, vocabulary):
    if len(cases) != 68 or len(records) != 613:
        raise ValueError('incomplete fixed case or event count')
    output = []
    seen_ids = set()
    for case_index, case in enumerate(cases):
        cid = case['case_id']
        if cid in seen_ids:
            raise ValueError('duplicate case')
        seen_ids.add(cid)
        group = records[1 + 9 * case_index:1 + 9 * (case_index + 1)]
        values, raw_by_action = {}, {}
        for offset, record in enumerate(group):
            event = record['event']
            action = event['action']
            if (record['index'] != 1 + 9 * case_index + offset
                    or event['index'] != 9 * case_index + offset
                    or event['case_id'] != cid or event['case_index'] != case_index
                    or event['input_ids'] != case['token_ids']
                    or event['attention_mask'] != [1] * len(case['token_ids'])
                    or action not in ACTIONS or action in values):
                raise ValueError('saved action/case order differs')
            for source, target in (('split', 'split'), ('role', 'role'),
                                   ('frame_index', 'frame_index'), ('context_index', 'context_index'),
                                   ('noun_index', 'noun_index'), ('determiner', 'determiner')):
                if event[source] != case[target]:
                    raise ValueError('saved case metadata differs')
            ref = record['tensors']['last_logits']
            if (ref['dtype'] != 'float32-le' or ref['shape'] != [vocabulary]
                    or ref['bytes'] != vocabulary * 4):
                raise ValueError('saved logit shape differs')
            raw = verified_bytes(Path(tensors) / (ref['sha256'] + '.f32le'), ref['sha256'])
            if len(raw) != vocabulary * 4:
                raise ValueError('saved logit byte count differs')
            raw_by_action[action] = raw
            values[action] = {}
            for name, (left, right) in PAIRS.items():
                a = struct.unpack_from('<f', raw, left * 4)[0]
                b = struct.unpack_from('<f', raw, right * 4)[0]
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError('selected logit is nonfinite')
                values[action][name] = a - b
        if set(values) != ACTIONS or [group[i]['event']['action'] for i in (0, 1, 8)] != [
                'clean', 'noop', 'clean_after']:
            raise ValueError('case action set or baseline positions differ')
        if raw_by_action['clean'] != raw_by_action['noop'] or raw_by_action['clean'] != raw_by_action['clean_after']:
            raise ValueError('full baseline logits differ')
        odd, even = {}, {}
        for axis in AXES:
            plus, minus = values[axis + '_plus'], values[axis + '_minus']
            odd[axis] = {name: math.fsum((plus[name], -minus[name])) / 2 for name in PAIRS}
            even[axis] = {name: math.fsum((plus[name], minus[name])) / 2 - values['clean'][name]
                          for name in PAIRS}
        output.append({'case_id': cid, 'split': case['split'], 'role': case['role'],
                       'frame': case['frame_index'], 'context': case['context_index'],
                       'noun': case['noun_index'], 'determiner': case['determiner'],
                       'margins': values, 'odd': odd, 'even': even})

    partitions = {}
    for split in ('discovery', 'evaluation'):
        selected = {(x['frame'], x['context'], x['noun'], x['determiner']): x
                    for x in output if x['split'] == split and x['role'] == 'target'}
        if len(selected) != 24:
            raise ValueError('target cells are incomplete')
        frame_counts, noun_counts, context_counts = [0, 0], [0, 0], [0, 0, 0]
        ties = wrong = complete_cells = 0
        cells = []
        for frame in range(2):
            for context in range(3):
                group = [selected[frame, context, noun, det]
                         for noun in range(2) for det in ('this', 'these')]
                signs = []
                for i, row in enumerate(group):
                    margin = row['margins']['clean']['present']
                    correct = margin > 0 if i % 2 == 0 else margin < 0
                    signs.append(correct)
                    if correct:
                        frame_counts[frame] += 1
                        noun_counts[i // 2] += 1
                        context_counts[context] += 1
                    elif margin == 0:
                        ties += 1
                    else:
                        wrong += 1
                complete_cells += all(signs)
                results = {}
                for axis in AXES:
                    a, b, c, d = (row['odd'][axis]['present'] for row in group)
                    ka, kb = a - b, c - d
                    dn = math.fsum((abs(ka), abs(kb))) / 2
                    dl = math.fsum((abs(a - c), abs(b - d))) / 2
                    results[axis] = {'ka': ka, 'kb': kb, 's': math.fsum((ka, kb)) / 2,
                                     'd_num': dn, 'd_lex': dl, 'delta': dn - dl}
                cells.append({'frame': frame, 'context': context, 'axis': results})
        adequate = (sum(frame_counts) >= 18 and min(frame_counts) >= 9 and min(noun_counts) >= 9
                    and min(context_counts) >= 6 and complete_cells >= 4)
        baseline = {'overall_correct': sum(frame_counts), 'frame_correct': frame_counts,
                    'noun_correct': noun_counts, 'context_correct': context_counts,
                    'all_correct_cells': complete_cells, 'ties': ties, 'wrong_sign': wrong,
                    'adequate': adequate}
        criteria = {}
        for label, statistic, floor, require_nouns in (
                ('sensitivity', 's', 0.05, True), ('number_over_noun', 'delta', 0.02, False)):
            scores = [cell['axis']['u'][statistic] for cell in cells]
            average = math.fsum(scores) / 6
            controls = [math.fsum(abs(cell['axis'][axis][statistic]) for cell in cells) / 6
                        for axis in ('v1', 'v2')]
            frame_means = [math.fsum(scores[start:start + 3]) / 3 for start in (0, 3)]
            positive = sum(x > 0 for x in scores)
            nouns = [math.fsum(cell['axis']['u'][key] for cell in cells) / 6 for key in ('ka', 'kb')]
            passed = (average >= floor and all(average > x for x in controls) and positive >= 4
                      and all(x > 0 for x in frame_means) and (not require_nouns or all(x > 0 for x in nouns)))
            result = {'u_mean': average, 'v1_abs_mean': controls[0], 'v2_abs_mean': controls[1],
                      'positive_cells': positive, 'frame_means': frame_means, 'raw_pass': passed,
                      'decision': 'baseline_limited' if not adequate else 'pass' if passed else 'fail'}
            if require_nouns:
                result['noun_k_means'] = nouns
            criteria[label] = result
        partitions[split] = {'baseline': baseline, 'criteria': criteria, 'cells': cells}
    return {'cases': output, 'partitions': partitions,
            'scope': 'selected saved logits and fixed contrasts only; no model, custody or unsaved-output attestation'}


def compare_reference(ours, reference):
    lookup = {row['case_id']: row for row in reference['cases']}
    if len(lookup) != 68:
        raise ValueError('reference case count differs')
    for row in ours['cases']:
        other = lookup[row['case_id']]
        for field in ('split', 'role', 'frame', 'context', 'noun', 'determiner', 'margins', 'odd', 'even'):
            if row[field] != other[field]:
                raise ValueError(('reference case statistic differs', row['case_id'], field))
    for name, partition in ours['partitions'].items():
        baseline = reference['partitions'][name]['baseline']
        if any(baseline[key] != value for key, value in partition['baseline'].items()):
            raise ValueError(('reference baseline differs', name))
        if reference['partitions'][name]['criteria'] != partition['criteria']:
            raise ValueError(('reference criteria differ', name))
        for cell, other in zip(partition['cells'], reference['partitions'][name]['cells']):
            if (cell['frame'], cell['context']) != (other['frame'], other['context']):
                raise ValueError('reference cell order differs')
            for axis in AXES:
                target = other['statistics']['present'][axis]
                for mine, theirs in (('ka', 'k_a'), ('kb', 'k_b'), ('s', 's'), ('d_num', 'd_num'),
                                     ('d_lex', 'd_lex'), ('delta', 'delta')):
                    if cell['axis'][axis][mine] != target[theirs]:
                        raise ValueError(('reference cell statistic differs', name, axis, mine))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'case-sha256', 'events', 'events-sha256', 'tensors', 'compare-report', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--vocabulary', type=int, default=50257)
    args = parser.parse_args()
    cases = json.loads(verified_bytes(args.cases, args.case_sha256))['cases']
    events = verified_bytes(args.events, args.events_sha256)
    if not events.endswith(b'\n'):
        raise ValueError('event log is incomplete')
    result = analyse(cases, [json.loads(line) for line in events.splitlines()], args.tensors, args.vocabulary)
    compare_reference(result, json.loads(Path(args.compare_report).read_bytes()))
    result['inputs'] = {'cases_sha256': args.case_sha256, 'events_sha256': args.events_sha256,
                        'reference_report_sha256': hashlib.sha256(Path(args.compare_report).read_bytes()).hexdigest()}
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        raise ValueError('fresh absolute report path is required')
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')
    with output.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'cases': len(result['cases']),
                      'compared_fixed_criteria': 'pass'}))


if __name__ == '__main__':
    main()
