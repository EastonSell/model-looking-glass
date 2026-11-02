"""Independent saved-logit replay for the public two-attractor experiment."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

CASE_SHA = 'fcb79a2762a2fd9fa346347c5156f5eb87bba314b1c8a021fe5fb639827f2798'
WEIGHT_SHA = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def same(actual, expected, where='value'):
    if isinstance(expected, dict):
        check(type(actual) is dict and set(actual) == set(expected), where + ' keys differ')
        for key, value in expected.items():
            same(actual[key], value, where + '/' + str(key))
    elif isinstance(expected, list):
        check(type(actual) is list and len(actual) == len(expected), where + ' length differs')
        for index, value in enumerate(expected):
            same(actual[index], value, where + '/' + str(index))
    elif type(expected) is float:
        check(type(actual) in (int, float) and math.isfinite(actual)
              and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), where + ' differs')
    else:
        check(type(actual) is type(expected) and actual == expected, where + ' differs')


def mean(values):
    check(bool(values), 'empty group')
    return math.fsum(values) / len(values)


def partition(subject, noun_pair):
    if subject in (1, 2):
        return 'development' if noun_pair == 1 else 'cross_new_attractor_pair'
    return 'cross_new_subject' if noun_pair == 1 else 'evaluation'


def verify(case_bytes, journal_bytes, result_bytes, journal_sha, result_sha):
    check(hashlib.sha256(case_bytes).hexdigest() == CASE_SHA, 'case bytes differ')
    check(hashlib.sha256(journal_bytes).hexdigest() == journal_sha, 'journal digest differs')
    check(hashlib.sha256(result_bytes).hexdigest() == result_sha, 'result digest differs')
    bundle = json.loads(case_bytes)
    cases = bundle['cases']
    check(len(cases) == 128 and len({x['case_id'] for x in cases}) == 128, 'case set differs')
    check(journal_bytes.endswith(b'\n'), 'journal has a partial final line')
    events = [json.loads(line) for line in journal_bytes.splitlines()]
    check(len(events) == 259, 'journal does not contain a complete fixed run')
    same(events[0], {'event': 'loading', 'cases_sha256': CASE_SHA}, 'loading')
    check(set(events[1]) == {'event', 'metadata'} and events[1]['event'] == 'model_ready',
          'model-ready event differs')
    metadata = events[1]['metadata']
    check(type(metadata) is dict and metadata.get('weights_sha256') == WEIGHT_SHA
          and metadata.get('asset_sha256', {}).get('tokenizer.json') == TOKENIZER_SHA,
          'recorded model or tokenizer identity differs')
    same(events[-1], {'event': 'complete', 'result_sha256': result_sha}, 'completion')
    saved = json.loads(result_bytes)
    check(set(saved) == {'schema', 'cases_sha256', 'model_metadata', 'analysis'}
          and saved['schema'] == 'public-two-attractor-agreement-result-v1'
          and saved['cases_sha256'] == CASE_SHA, 'result identity differs')
    same(saved['model_metadata'], metadata, 'result model metadata')
    rows = {}
    for index, case in enumerate(cases):
        same(events[2 + 2 * index], {'event': 'forward_start', 'case_id': case['case_id']},
             'forward start')
        event = events[3 + 2 * index]
        check(type(event) is dict and set(event) == {'event', 'case_id', 'logits'}
              and event['event'] == 'forward_result' and event['case_id'] == case['case_id'],
              'forward result order differs')
        logits = event['logits']
        check(type(logits) is dict and set(logits) == {'is', 'are', 'was', 'were'}
              and all(type(x) in (int, float) and math.isfinite(x) for x in logits.values()),
              'saved logits differ')
        rows[case['case_id']] = logits
    analysis = saved['analysis']
    check(type(analysis) is dict and set(analysis) == {'case_results', 'factorial_units',
        'unit_summary', 'condition_summary', 'baseline_counts', 'subject_controls',
        'subject_control_summary'}, 'analysis inventory differs')
    case_outputs, units, subject_cells = [], defaultdict(dict), defaultdict(dict)
    for case in cases:
        s, f, order, number = (case[x] for x in ('subject_pair', 'frame_pair', 'order', 'subject_number'))
        near, beside = case['near_number'], case['beside_number']
        sign = 1 if number == 'singular' else -1
        n, b = int(near != number), int(beside != number)
        scores = rows[case['case_id']]
        raw = {'present': float(scores['is']) - float(scores['are']),
               'past': float(scores['was']) - float(scores['were'])}
        key = (s, f, order, number)
        check((n, b) not in units[key], 'duplicate factorial cell')
        units[key][(n, b)] = raw
        subject_key = (s, f, order, near, beside)
        check(number not in subject_cells[subject_key], 'duplicate subject cell')
        subject_cells[subject_key][number] = raw
        case_outputs.append({'case_id': case['case_id'], 'partition': partition(s, f),
            'subject_number': number, 'near_mismatch': n, 'beside_mismatch': b,
            'raw_margins': raw, 'signed_margins': {t: sign * x for t, x in raw.items()},
            'correct': {t: sign * x > 0 for t, x in raw.items()}, 'logits': scores})
    same(analysis['case_results'], case_outputs, 'case results')
    units_out = []
    for key in sorted(units):
        s, f, order, number = key
        cells = units[key]
        check(set(cells) == {(0, 0), (1, 0), (0, 1), (1, 1)}, 'factorial cell missing')
        sign = 1 if number == 'singular' else -1
        contrasts = {}
        for tense in ('present', 'past'):
            a, b, c, d = (sign * cells[q][tense] for q in ((0, 0), (1, 0), (0, 1), (1, 1)))
            cross = (d - b) - (c - a)
            contrasts[tense] = {'all_congruent': a, 'near_only_mismatch': b,
                'beside_only_mismatch': c, 'both_mismatch': d,
                'near_shift': ((b - a) + (d - c)) / 2.0,
                'beside_shift': ((c - a) + (d - b)) / 2.0,
                'interaction': cross, 'raw_number_interaction': sign * cross}
        units_out.append({'subject_pair': s, 'frame_pair': f, 'order': order,
            'subject_number': number, 'partition': partition(s, f), 'contrasts': contrasts})
    check(len(units_out) == 32, 'factorial unit count differs')
    same(analysis['factorial_units'], units_out, 'factorial units')
    grouped = {'all': units_out}
    for field in ('partition', 'subject_number', 'subject_pair', 'frame_pair', 'order'):
        for unit in units_out:
            grouped.setdefault(field + '=' + str(unit[field]), []).append(unit)
    summaries = {}
    names = ('all_congruent', 'near_only_mismatch', 'beside_only_mismatch', 'both_mismatch',
             'near_shift', 'beside_shift', 'interaction')
    for name, group in grouped.items():
        summaries[name] = {'units': len(group)}
        for tense in ('present', 'past'):
            summaries[name][tense] = {field: mean([u['contrasts'][tense][field] for u in group])
                                      for field in names}
    same(analysis['unit_summary'], summaries, 'unit summaries')
    condition = {}
    for near in (0, 1):
        for beside in (0, 1):
            group = [r for r in case_outputs if r['near_mismatch'] == near and r['beside_mismatch'] == beside]
            condition[str(near) + str(beside)] = {'cases': len(group)}
            for tense in ('present', 'past'):
                condition[str(near) + str(beside)][tense] = {
                    'mean_signed_margin': mean([r['signed_margins'][tense] for r in group]),
                    'correct': sum(r['correct'][tense] for r in group)}
    same(analysis['condition_summary'], condition, 'condition summaries')
    baselines, controls, control_summaries = {}, [], {}
    partitions = ('development', 'evaluation', 'cross_new_attractor_pair', 'cross_new_subject')
    for label in partitions:
        for number in ('singular', 'plural'):
            group = [r for r in case_outputs if r['partition'] == label and r['subject_number'] == number
                     and r['near_mismatch'] == 0 and r['beside_mismatch'] == 0]
            check(len(group) == 4, 'baseline group differs')
            baselines[label + '/' + number] = {}
            for tense in ('present', 'past'):
                margins = [r['signed_margins'][tense] for r in group]
                baselines[label + '/' + number][tense] = {'correct': sum(x > 0 for x in margins),
                    'ties': sum(x == 0 for x in margins), 'wrong_sign': sum(x < 0 for x in margins),
                    'incorrect_including_ties': sum(x <= 0 for x in margins)}
    same(analysis['baseline_counts'], baselines, 'baseline counts')
    for key in sorted(subject_cells):
        s, f, order, near, beside = key
        values = subject_cells[key]
        check(set(values) == {'singular', 'plural'}, 'subject control is incomplete')
        controls.append({'subject_pair': s, 'frame_pair': f, 'order': order,
            'near_number': near, 'beside_number': beside, 'partition': partition(s, f),
            'singular_minus_plural': {t: values['singular'][t] - values['plural'][t]
                                      for t in ('present', 'past')}})
    check(len(controls) == 64, 'subject-control count differs')
    same(analysis['subject_controls'], controls, 'subject controls')
    for label in partitions:
        for near in ('singular', 'plural'):
            for beside in ('singular', 'plural'):
                group = [r for r in controls if r['partition'] == label and
                         r['near_number'] == near and r['beside_number'] == beside]
                check(len(group) == 4, 'subject-control group differs')
                control_summaries[label + '/' + near + '/' + beside] = {
                    t: mean([r['singular_minus_plural'][t] for r in group]) for t in ('present', 'past')}
    same(analysis['subject_control_summary'], control_summaries, 'subject-control summaries')
    return {'status': 'saved_arithmetic_consistent', 'cases': 128, 'factorial_units': 32,
        'subject_controls': 64, 'journal_sha256': journal_sha, 'result_sha256': result_sha,
        'scope': 'saved selected logits only; process exit, unrecorded inputs and full logits are external'}


def main():
    if len(sys.argv) != 6:
        raise ValueError('usage: CASES JOURNAL RESULT EXPECTED_JOURNAL_SHA EXPECTED_RESULT_SHA')
    cases, journal, result = (Path(p).read_bytes() for p in sys.argv[1:4])
    output = verify(cases, journal, result, sys.argv[4], sys.argv[5])
    print(json.dumps(output, sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
