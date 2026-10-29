"""Independent selected-logit replay of a closed public substitution run.

Reads the existing files in place. It does not load a model or authenticate
process execution, exposure history, unsaved logits, or hook landings.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess


PINS = {
    'exit': 'a64a325d07e76ade152c8533dc4c1953c421c9b2e49de22f2e4b43b04b1df27e',
    'prelaunch': '2419e46d53c387a6861c77c3a0c42f9b4d62a2e76b30be3e25f82707ed590012',
    'supervisor': '273ba26b2d0640266374c75335bc0afd6e794d8ec2faac0992c103b866d3c874',
    'child': '51affc37a49a038849aa4f8638717bfbd728a14701a190ae5a0cf1e774042359',
    'entry': 'bb146fec401dc6654b2861b6a1991bb7f9dca3509782c98965b5837f0ad1ad82',
    'events': '7257fed9573c05b3710ca91c7c8dbb018508cf43307801a0be6d527b7eda65b1',
    'terminal': 'eff89a78a32f27bf8841475330ae0940802b360c0636045c4fbf6befa9d7320f',
    'cases': '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe',
}
REVISION = 'ca6c67f30b90f5d79df68e379cbb7cafe21acb08'
TREE = 'caa78a8bff55ed5c0adaeb07b8828057cf590d42'
START_SCOPE = 'public prospective agreement noun-substitution'
CONDITIONS = ('none', 'attractor_singular', 'attractor_plural')
ORDERS = ('As Ap Bs Bp', 'Ap As Bp Bs', 'Bs Bp As Ap',
          'Bp Bs Ap As', 'As Bs Ap Bp', 'Bp Ap Bs As')
SIGNS = (True, False, True, False, False, True)
AXES = ('u', 'v1', 'v2')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            check(key not in out, 'duplicate JSON key')
            out[key] = value
        return out

    def finite_float(value):
        number = float(value)
        check(math.isfinite(number), 'nonfinite JSON number')
        return number

    def bad_constant(value):
        raise ValueError('invalid JSON constant: ' + value)

    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                      parse_float=finite_float, parse_constant=bad_constant)


def read_pinned(path, digest, maximum):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    check(len(raw) <= maximum, 'input exceeds bound: ' + path.name)
    check(hashlib.sha256(raw).hexdigest() == digest, 'digest differs: ' + path.name)
    return raw


def f32(raw, offset):
    bits = struct.unpack_from('<I', raw, offset * 4)[0]
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    check(exponent != 255, 'selected logit is nonfinite')
    value = (math.ldexp(float(fraction), -149) if exponent == 0
             else math.ldexp(float((1 << 23) | fraction), exponent - 150))
    return -value if bits >> 31 else value


def mean(values):
    values = list(values)
    check(bool(values) and all(math.isfinite(v) for v in values), 'invalid mean')
    return math.fsum(values) / len(values)


def expected_schedule(cases):
    check(type(cases) is list and len(cases) == 54, 'wrong public case count')
    lookup = {}
    for row in cases:
        key = (row['frame_index'], row['condition_index'], row['morphology'],
               row['subject_key'], row['subject_number'])
        check(key not in lookup, 'duplicate case coordinates')
        lookup[key] = row
    result = []
    for frame in range(2):
        for condition in range(3):
            n = frame * 3 + condition
            morphs = ('regular', 'irregular') if (frame + condition) % 2 == 0 else ('irregular', 'regular')
            ordered = []
            for morph in morphs:
                keys = ('R1', 'R2') if morph == 'regular' else ('I1', 'I2')
                for name in ORDERS[n].split():
                    key = keys[0] if name[0] == 'A' else keys[1]
                    number = 'singular' if name[1] == 's' else 'plural'
                    ordered.append(lookup[(frame, condition, morph, key, number)])
            ordered.append(lookup[(frame, condition, None, None, None)])
            for row in ordered:
                plus_first = ((frame + condition) % 2 == 0 if row['role'] == 'context'
                              else SIGNS[n] != (row['morphology'] == 'irregular'))
                signs = ('plus', 'minus') if plus_first else ('minus', 'plus')
                base = [axis + '_' + sign for axis in AXES for sign in signs]
                rotation = 2 * condition
                arms = base[rotation:] + base[:rotation]
                for slot, action in [(None, 'clean'), (None, 'noop_l3_final')]:
                    result.append((row, action, slot))
                result.extend((row, action, slot) for slot, action in enumerate(arms))
                result.append((row, 'clean_after', None))
    check(len(result) == 486, 'wrong schedule length')
    return result


def factor(values):
    a_s, a_p, b_s, b_p = values
    opposite = mean((abs(a_s - b_p), abs(b_s - a_p)))
    same = mean((abs(a_s - b_s), abs(a_p - b_p)))
    return {'opposite': opposite, 'same': same, 'delta': opposite - same,
            'number': mean((a_s - a_p, b_s - b_p)),
            'lemma': mean((a_s - b_s, a_p - b_p)),
            'interaction': (a_s - a_p) - (b_s - b_p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.study
    attempt = root / 'study-attempt-v1'
    run = attempt / 'run'
    journal = run / 'attempt'

    exit_record = strict_json(read_pinned(root / 'study-inputs-v2/parent-exit.json', PINS['exit'], 64 * 1024))
    prelaunch = strict_json(read_pinned(run / 'prelaunch.json', PINS['prelaunch'], 128 * 1024))
    supervisor = strict_json(read_pinned(attempt / 'supervisor.json', PINS['supervisor'], 2 * 1024 * 1024))
    child = strict_json(read_pinned(attempt / 'child-custody.json', PINS['child'], 16 * 1024))
    entry = strict_json(read_pinned(run / 'entry-result.json', PINS['entry'], 64 * 1024))
    terminal = strict_json(read_pinned(journal / 'terminal.json', PINS['terminal'], 64 * 1024))
    cases = strict_json(read_pinned(run / 'cases.json', PINS['cases'], 2 * 1024 * 1024))['cases']
    events_raw = read_pinned(journal / 'events.jsonl', PINS['events'], 32 * 1024 * 1024)
    records = [strict_json(line) for line in events_raw.splitlines()]
    check(events_raw.endswith(b'\n') and len(records) == 487, 'event count or termination differs')
    check(exit_record['operator_observed_parent_exit_code'] == 0
          and type(exit_record['operator_observed_parent_exit_code']) is int
          and exit_record['journal_sha256'] == PINS['events']
          and exit_record['terminal_sha256'] == PINS['terminal']
          and exit_record['supervisor_sha256'] == PINS['supervisor']
          and exit_record['child_custody_sha256'] == PINS['child']
          and exit_record['entry_result_sha256'] == PINS['entry']
          and exit_record['prelaunch_sha256'] == PINS['prelaunch'], 'parent observation differs')
    sup = supervisor['supervision_result']
    check(supervisor['clean'] is True and supervisor['child_binding_matches'] is True
          and supervisor['post_source_check'] == 'matched' and supervisor['post_errors'] == []
          and supervisor['validation_only'] is False and supervisor['source_revision'] == REVISION
          and supervisor['source_tree'] == TREE and supervisor['child'] == child
          and sup['clean'] is True and type(sup['exit_code']) is int and sup['exit_code'] == 0
          and sup['timed_out'] is False and sup['run_id'] == child['run_id']
          and sup['pid'] == child['owner_pid'] == entry['owner_pid']
          and entry['owner_run_id'] == child['run_id'], 'supervision identities differ')
    check(entry['status'] == terminal['status'] == 'complete_candidate'
          and entry['model_disposed'] is True and entry['worker']['forwards'] == 486
          and terminal['events'] == 487 and terminal['detail']['forwards'] == 486
          and terminal['storage_accounting_complete'] is True, 'terminal is incomplete')
    inventory = {row['path']: row for row in supervisor['run_files']}
    check(len(inventory) == len(supervisor['run_files']), 'duplicate inventory path')
    for path, digest in (('prelaunch.json', PINS['prelaunch']), ('entry-result.json', PINS['entry']),
                         ('cases.json', PINS['cases']), ('attempt/events.jsonl', PINS['events']),
                         ('attempt/terminal.json', PINS['terminal'])):
        check(inventory[path]['sha256'] == digest, 'supervisor inventory pin differs')
    binding = {'prelaunch_sha256': PINS['prelaunch'],
               'reservation_sha256': prelaunch['exposure']['reservation_sha256'],
               'latest_exposure_sha256': prelaunch['exposure']['operator_sha256']}
    started = strict_json((journal / 'started.json').read_bytes())
    check(started == {'source_revision': REVISION, 'scope': START_SCOPE, 'external_binding': binding}
          and records[0]['event']['external_binding'] == binding and entry['binding'] == binding,
          'saved start binding differs')
    check(prelaunch['owner']['source_revision'] == REVISION and prelaunch['owner']['source_tree'] == TREE,
          'prelaunch source identity differs')
    tree = subprocess.check_output(['git', '-C', str(args.repo), 'show', '-s', '--format=%T', REVISION], text=True).strip()
    check(tree == TREE, 'source tree differs')
    for key in ('entry', 'executor'):
        path = prelaunch['owner']['entry_path' if key == 'entry' else 'executor_path']
        expected = prelaunch['owner']['entry_script_sha256' if key == 'entry' else 'executor_sha256']
        raw = subprocess.check_output(['git', '-C', str(args.repo), 'show', REVISION + ':' + path])
        check(hashlib.sha256(raw).hexdigest() == expected, 'source bytes differ')

    schedule = expected_schedule(cases)
    margins, digests = {}, {}
    for index, (case, action, slot) in enumerate(schedule):
        record, event = records[index + 1], records[index + 1]['event']
        check(record['index'] == index + 1 and event['index'] == index
              and event['kind'] == 'forward' and event['case_id'] == case['case_id']
              and event['action'] == action and event['slot'] == slot
              and event['input_ids'] == case['token_ids']
              and event['attention_mask'] == [1] * len(case['token_ids'])
              and event['all_sequence_logits_finite'] is True, 'forward identity differs')
        ref = record['tensors']['last_logits']
        check(ref['dtype'] == 'float32-le' and ref['shape'] == [50257] and ref['bytes'] == 201028,
              'logit tensor format differs')
        digest = ref['sha256']
        if digest not in digests:
            raw = read_pinned(journal / 'tensors' / (digest + '.f32le'), digest, 201028)
            check(len(raw) == 201028, 'logit tensor length differs')
            digests[digest] = f32(raw, 318) - f32(raw, 389)
        per_case = margins.setdefault(case['case_id'], {})
        check(action not in per_case, 'duplicate case/action')
        per_case[action] = {'margin': digests[digest], 'logits_sha256': digest}
    for case in cases:
        actions = margins[case['case_id']]
        check(len(actions) == 9 and actions['clean']['logits_sha256'] == actions['noop_l3_final']['logits_sha256']
              == actions['clean_after']['logits_sha256'], 'clean/no-op/repeat logits differ')

    lookup = {(c['frame_index'], c['condition_index'], c['morphology'], c['subject_key'],
               c['subject_number']): c for c in cases if c['role'] == 'target'}
    odd = {c['case_id']: {axis: mean((margins[c['case_id']][axis + '_plus']['margin'],
              -margins[c['case_id']][axis + '_minus']['margin'])) for axis in AXES} for c in cases}
    cells = []
    for frame in range(2):
        for cond, label in enumerate(CONDITIONS):
            for morph, keys in (('regular', ('R1', 'R2')), ('irregular', ('I1', 'I2'))):
                rows = [lookup[(frame, cond, morph, key, number)] for key, number in
                        ((keys[0], 'singular'), (keys[0], 'plural'),
                         (keys[1], 'singular'), (keys[1], 'plural'))]
                cells.append({'frame_index': frame, 'condition': label, 'morphology': morph,
                              'case_ids': [row['case_id'] for row in rows],
                              'axis': {axis: factor([odd[row['case_id']][axis] for row in rows]) for axis in AXES}})
    targets = [case for case in cases if case['role'] == 'target']
    def correct(row):
        value = margins[row['case_id']]['clean']['margin']
        return value > 0 if row['subject_number'] == 'singular' else value < 0
    counts = lambda rows: {'correct': sum(correct(row) for row in rows), 'total': len(rows)}
    baseline = {'overall': counts(targets),
                'morphology': {x: counts([r for r in targets if r['morphology'] == x]) for x in ('regular', 'irregular')},
                'frame': {str(x): counts([r for r in targets if r['frame_index'] == x]) for x in range(2)},
                'condition': {x: counts([r for r in targets if r['condition'] == x]) for x in CONDITIONS},
                'lemma': {x: counts([r for r in targets if r['subject_key'] == x]) for x in ('R1', 'R2', 'I1', 'I2')}}
    cases_by_id = {row['case_id']: row for row in cases}
    baseline['four_correct_cells'] = sum(all(correct(cases_by_id[x]) for x in cell['case_ids']) for cell in cells)
    baseline['adequate'] = (baseline['overall']['correct'] >= 36
        and all(x['correct'] >= 18 for x in baseline['morphology'].values())
        and all(x['correct'] >= 18 for x in baseline['frame'].values())
        and all(x['correct'] >= 12 for x in baseline['condition'].values())
        and all(x['correct'] >= 9 for x in baseline['lemma'].values())
        and baseline['four_correct_cells'] >= 8)
    def axes(rows):
        return {axis: {'mean_delta': mean(c['axis'][axis]['delta'] for c in rows),
                       'mean_absolute_delta': mean(abs(c['axis'][axis]['delta']) for c in rows)} for axis in AXES}
    total = axes(cells)
    morph = {m: axes([c for c in cells if c['morphology'] == m]) for m in ('regular', 'irregular')}
    positive = {m: sum(c['axis']['u']['delta'] > 0 for c in cells if c['morphology'] == m)
                for m in ('regular', 'irregular')}
    frame_means = {str(i): axes([c for c in cells if c['frame_index'] == i])['u']['mean_delta'] for i in range(2)}
    flags = {'overall': total['u']['mean_delta'] >= 0.05 and all(total['u']['mean_delta'] > total[x]['mean_absolute_delta'] for x in ('v1', 'v2')),
             'morphology': all(v['u']['mean_delta'] >= 0.02 and all(v['u']['mean_delta'] > v[x]['mean_absolute_delta'] for x in ('v1', 'v2')) for v in morph.values()),
             'positive': sum(positive.values()) >= 8 and all(v >= 4 for v in positive.values()),
             'frames': all(v > 0 for v in frame_means.values())}
    report = {'schema': 'independent-noun-substitution-selected-v1', 'pins': PINS, 'source_revision': REVISION,
              'source_tree': TREE, 'forward_count': 486, 'logit_payloads_rehashed': len(digests),
              'parent_exit_observation': 'zero exit as recorded by a cooperative operator',
              'baseline': baseline, 'overall': total, 'by_morphology': morph,
              'positive_cells_by_morphology': positive, 'frame_mean_delta_u': frame_means,
              'flags': flags, 'criterion_pass': baseline['adequate'] and all(flags.values()),
              'cases': [{'case_id': c['case_id'], 'odd': odd[c['case_id']], 'margins': margins[c['case_id']]} for c in cases],
              'cells': cells,
              'scope': 'Selected is/are logits only; no model execution, process authentication, full-vocabulary or landing audit.'}
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    with args.output.open('xb') as stream:
        check(stream.write(raw) == len(raw), 'short report write')
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                      'baseline': baseline, 'criterion_pass': report['criterion_pass'],
                      'overall': total, 'positive_cells_by_morphology': positive,
                      'frame_mean_delta_u': frame_means, 'flags': flags}, sort_keys=True))


if __name__ == '__main__':
    main()
