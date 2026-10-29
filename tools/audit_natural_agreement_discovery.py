"""Independent numerical and intervention-witness audit for a saved journal.

This uses no Model Looking Glass code. The synthetic option is solely for
testing this auditor; real usage requires the pinned public preflight manifest.
It checks saved evidence, not whether an untrusted process executed the model.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


MANIFEST_SHA = '1b1adf4d72163bfa3e0399ed8ee92eba6b5c0eb82a47edf28cecde5ffb1deafe'
SITES = [f'{layer}:{kind}:last' for layer in range(1, 5) for kind in ('attn', 'mlp')]
ALL_SITES = SITES + ['5:block:last', '5:block:subject']
CELLS = ('same_lemma_same_number', 'same_lemma_opposite_number',
         'different_lemma_same_number', 'different_lemma_opposite_number')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact_equal(actual, expected):
    """Compare JSON values without treating booleans as integer aliases."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            exact_equal(actual[key], value) for key, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            exact_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def fields(raw):
    require(bool(raw) and len(raw) % 4 == 0, 'invalid float32 byte payload')
    bits = np.frombuffer(raw, dtype='<u4')
    exponent = ((bits >> 23) & 255).astype(np.int16)
    require(not np.any(exponent == 255), 'nonfinite float32 field')
    fraction = bits & np.uint32(0x7fffff)
    mantissa = (fraction | np.where(exponent == 0, 0, 1 << 23).astype(np.uint32)).astype(np.float64)
    values = np.ldexp(mantissa, np.where(exponent == 0, -149, exponent - 150))
    return np.where(bits >> 31, -values, values)


def comparison(left, right):
    require(len(left) == len(right), 'tolerance comparison byte lengths differ')
    a, b = fields(left), fields(right)
    delta = np.abs(a - b)
    return {'bit_equal': left == right, 'max_abs_difference_binary64': float(np.max(delta)),
            'within_tolerance': bool(np.all(delta <= 1e-5 + 1e-5 * np.maximum(np.abs(a), np.abs(b))))}


def check_comparison(declared, left, right, message):
    computed = comparison(left, right)
    require(exact_equal(declared, computed) and computed['within_tolerance'], message)


def manifest_cases(path):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, 'preflight manifest digest mismatch')
    manifest = json.loads(raw)
    require(manifest['model_id'] == 'distilbert/distilgpt2' and
            manifest['model_revision'] == '2290a62682d06624634c1f46a6ad5be0f47f38aa' and
            manifest['targets'] == {' is': 318, ' are': 389, ' was': 373, ' were': 547},
            'wrong pinned model or contrast IDs')
    rows = manifest['rows']
    require(len(rows) == 16, 'expected 16 public preflight rows')
    return [{'case_id': f'c{index:02}', 'quartet': f'q{row["group"]}',
             'lemma': rows[index - index % 2]['subject'], 'number': row['subject_number'],
             'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
             'subject_index': row['subject_index']} for index, row in enumerate(rows)]


def inspect(directory, manifest, synthetic=False):
    directory = Path(directory)
    event_raw = (directory / 'events.jsonl').read_bytes()
    terminal_raw = (directory / 'terminal.json').read_bytes()
    started_raw = (directory / 'started.json').read_bytes()
    terminal = json.loads(terminal_raw)
    records = [json.loads(line) for line in event_raw.splitlines()]
    require(terminal['status'] == 'complete_candidate' and
            terminal.get('storage_accounting_complete') is True and
            exact_equal(terminal['events'], len(records)),
            'not a complete candidate journal')
    require(exact_equal(terminal['event_log_bytes'], len(event_raw)), 'event-byte accounting mismatch')
    require(exact_equal([r['index'] for r in records], list(range(len(records)))), 'event ordering mismatch')
    events = [r['event'] for r in records]
    require(events[0]['kind'] == 'run_contract' and events[-1]['kind'] == 'summary', 'missing endpoints')
    contract = events[0]
    cases = contract['cases']
    if not synthetic:
        require(exact_equal(cases, manifest_cases(manifest)), 'run cases differ from the pinned public manifest')
    case_map = {c['case_id']: c for c in cases}
    groups = defaultdict(list)
    for case in cases:
        groups[case['quartet']].append(case['case_id'])
    require(len(case_map) == len(cases) and len(groups) == 4 and
            all(len(g) == 4 for g in groups.values()), 'invalid case/quartet coverage')
    for group in groups.values():
        members = [case_map[k] for k in group]
        require(len({c['lemma'] for c in members}) == 2 and
                len({(c['lemma'], c['number']) for c in members}) == 4 and
                {c['number'] for c in members} == {'singular', 'plural'}, 'invalid factorial')
    require([s['id'] for s in contract['sites']] == ALL_SITES, 'unexpected site order')
    expected_width, expected_vocab = (8, 600) if synthetic else (768, 50257)
    if not synthetic:
        require(all(len(case['input_ids']) == 5 for case in cases), 'unexpected input sequence length')
    require(exact_equal(contract['cpu_rng_seed'], 20261028) and exact_equal(contract['cpu_threads'], 1) and
            (synthetic or contract['deterministic_algorithms'] is True), 'unexpected CPU execution contract')
    kinds = Counter(e['kind'] for e in events)
    expected = {'run_contract': 1, 'clean_capture': 16, 'clean_repeat': 16, 'null': 16,
                'patch': 512, 'delivery': 16, 'subject_position': 16, 'final_clean': 16,
                'present_site_metrics': 8, 'secondary_past_site_metrics': 8, 'summary': 1}
    require(kinds == expected, 'unexpected number of forward/control/summary records')
    planned = []
    for quartet, group in groups.items():
        for kind in ('clean_capture', 'clean_repeat'):
            planned.extend((kind, quartet, case, None, None) for case in group)
        planned.extend(('null', quartet, case, None, SITES[0]) for case in group)
        planned.extend(('patch', quartet, target, donor, site)
                       for site in SITES for target in group for donor in group)
        for target in group:
            target_case = case_map[target]
            donor = next(c for c in group if case_map[c]['lemma'] == target_case['lemma'] and
                         case_map[c]['number'] != target_case['number'])
            planned.extend((('delivery', quartet, target, donor, ALL_SITES[8]),
                            ('subject_position', quartet, target, donor, ALL_SITES[9])))
        planned.extend(('final_clean', quartet, case, None, None) for case in group)
    observed = []
    for event in events:
        kind = event['kind']
        if kind in ('clean_capture', 'clean_repeat', 'final_clean'):
            observed.append((kind, event['quartet'], event['case'], None, None))
        elif kind in ('null', 'patch', 'delivery', 'subject_position'):
            observed.append((kind, event['quartet'], event['target'], event.get('donor'), event['site']['id']))
    require(observed == planned, 'forward schedule or post-family cleanup order differs')
    summary = events[-1]['result']
    require(summary['status'] == 'complete' and exact_equal(summary['forwards'], 608) and
            summary['model_state_unchanged'] is True and
            exact_equal(summary['controls'], {'null': 16, 'self': 128, 'delivery': 16,
                                             'subject_position': 16, 'repeat': 16, 'final_clean': 16}),
            'summary state or control counts differ')
    require(exact_equal(summary['cpu_rng_seed'], contract['cpu_rng_seed']) and
            type(contract['seeded_cpu_rng_sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', contract['seeded_cpu_rng_sha256']) is not None and
            summary['seeded_cpu_rng_sha256'] == contract['seeded_cpu_rng_sha256'] and
            type(summary['model_state_sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', summary['model_state_sha256']) is not None,
            'summary and contract seed or state identity differs')

    cache, sizes = {}, {}
    def get(row, name):
        ref = row['tensors'][name]
        digest = ref['sha256']
        require(ref['dtype'] == 'float32-le' and exact_equal(ref['bytes'], 4 * math.prod(ref['shape'])),
                'invalid tensor reference')
        if digest not in cache:
            data = (directory / 'tensors' / (digest + '.f32le')).read_bytes()
            require(len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == digest,
                    'tensor digest/size mismatch')
            fields(data)  # independently reject nonfinite fields
            cache[digest] = data
            sizes[digest] = len(data)
        require(len(cache[digest]) == ref['bytes'], 'inconsistent shared tensor size')
        return cache[digest]
    for row in records:
        event, tensors = row['event'], row['tensors']
        kind = event['kind']
        if kind in ('clean_capture', 'clean_repeat'):
            expected_shapes = {f'capture_{i}': [expected_width] for i in range(10)}
            expected_shapes['last_logits'] = [expected_vocab]
        elif kind in ('null', 'patch', 'delivery', 'subject_position'):
            case = case_map[event['target']]
            expected_shapes = {'before': [expected_width], 'after': [expected_width],
                               'whole_before': [1, len(case['input_ids']), expected_width],
                               'whole_after': [1, len(case['input_ids']), expected_width]}
            name = 'contrast_logits' if kind == 'patch' and event['target'] != event['donor'] else 'last_logits'
            expected_shapes[name] = [4] if name == 'contrast_logits' else [expected_vocab]
        elif kind == 'final_clean':
            expected_shapes = {'last_logits': [expected_vocab]}
        else:
            expected_shapes = {}
        require(set(tensors) == set(expected_shapes) and
                all(exact_equal(tensors[name]['shape'], shape) for name, shape in expected_shapes.items()),
                'event tensor shapes differ from the declared model/forward')
        require(sum(ref['bytes'] for ref in tensors.values()) <= 1024 * 1024,
                'per-forward tensor budget exceeded')
        for name in row['tensors']:
            get(row, name)
    require(exact_equal(terminal['retained_tensor_bytes'], sum(sizes.values())), 'retained-byte accounting mismatch')
    require(sum(sizes.values()) <= 512 * 1024 * 1024 and len(event_raw) <= 64 * 1024 * 1024,
            'total tensor or event-log budget exceeded')
    require({p.name for p in (directory / 'tensors').iterdir()} ==
            {sha + '.f32le' for sha in cache}, 'unreferenced or missing tensor files')

    def unique(kind, key='case'):
        rows = [r for r in records if r['event']['kind'] == kind]
        result = {r['event'][key]: r for r in rows}
        require(len(result) == len(rows) == 16 and set(result) == set(case_map),
                'missing or duplicate ' + kind)
        return result
    clean = unique('clean_capture')
    repeat = unique('clean_repeat')
    null = unique('null', 'target')
    delivery = unique('delivery', 'target')
    subject = unique('subject_position', 'target')
    final = unique('final_clean')
    require(exact_equal(terminal['detail'], {'forwards': 608}), 'forward count mismatch')
    def input_hash(case):
        raw = json.dumps({'ids': case['input_ids'], 'subject_index': case['subject_index']},
                         sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()
    site_by_id = {item['id']: item for item in contract['sites']}
    for site in ALL_SITES:
        layer, component, token = site.split(':')
        expected_path = f'transformer.h.{layer}' + ('' if component == 'block' else '.' + component)
        expected_selector = [0] if component in ('attn', 'block') else []
        require(exact_equal(site_by_id[site], {'id': site, 'layer': int(layer), 'component': component,
                                              'token_role': token, 'module_path': expected_path,
                                              'output_selector': expected_selector}), 'site coordinate mismatch')

    def metadata(row, case, selected_site=None):
        event, tensors = row['event'], row['tensors']
        kind = event['kind']
        steps = event['plan_metadata']['steps']
        width = tensors['capture_0']['shape'][0] if selected_site is None else tensors['before']['shape'][0]
        if selected_site is None:
            requests = [(f'capture_{i}', 'capture', site, False, None) for i, site in enumerate(ALL_SITES)]
        else:
            operation = 'null' if kind == 'null' else 'patch'
            source = 'before' if kind == 'patch' and event['target'] == event['donor'] else None
            requests = [('whole_before', 'capture', selected_site, True, None),
                        ('before', 'capture', selected_site, False, None),
                        ('action', operation, selected_site, False, source),
                        ('after', 'capture', selected_site, False, None),
                        ('whole_after', 'capture', selected_site, True, None)]
        capture_bytes = (10 * width * 4 if selected_site is None else
                         2 * (len(case['input_ids']) + 1) * width * 4)
        source_bytes = width * 4 if (selected_site is not None and operation == 'patch' and
                                      not (kind == 'patch' and event['target'] == event['donor'])) else 0
        require(exact_equal(event['plan_metadata']['capture_bytes'], capture_bytes) and
                exact_equal(event['plan_metadata']['source_bytes'], source_bytes),
                'hook storage accounting mismatch')
        require(len(steps) == len(requests), 'wrong number of recorded hook steps')
        previous = []
        for index, (step, request) in enumerate(zip(steps, requests)):
            name, operation, site, whole, source = request
            descriptor = site_by_id[site]
            token = case['subject_index'] if site.endswith(':subject') else len(case['input_ids']) - 1
            prior = [n for path, n in previous if path == descriptor['module_path']]
            expected_values = {'name': name, 'operation': operation, 'order': index,
                               'module_path': descriptor['module_path'], 'invocation': 0, 'forward_index': 0,
                               'batch_index': None if whole else 0, 'token_index': None if whole else token,
                               'output_selector': descriptor['output_selector'],
                               'output_shape': [1, len(case['input_ids']), width],
                               'selected_shape': [1, len(case['input_ids']), width] if whole else [width],
                               'dtype': 'torch.float32', 'device': 'cpu', 'source_capture': source,
                               'preceding_same_site_actions': prior, 'call_site_known': False,
                               'module_aliases': [descriptor['module_path']]}
            require(exact_equal(step, expected_values), 'hook metadata mismatch')
            previous.append((descriptor['module_path'], name))

    for case in cases:
        key = case['case_id']
        require(clean[key]['event']['input_sha256'] == input_hash(case), 'clean input binding mismatch')
        metadata(clean[key], case)
        metadata(repeat[key], case)
        repeat_checks = repeat[key]['event']['checks']
        require(set(repeat_checks) == set(ALL_SITES) | {'last_logits'}, 'clean-repeat checks incomplete')
        check_comparison(repeat_checks['last_logits'], get(clean[key], 'last_logits'),
                         get(repeat[key], 'last_logits'), 'clean-repeat logits mismatch')
        for i in range(10):
            check_comparison(repeat_checks[ALL_SITES[i]], get(clean[key], f'capture_{i}'),
                             get(repeat[key], f'capture_{i}'), 'clean-repeat activation mismatch')
        check_comparison(final[key]['event']['logits_check'], get(clean[key], 'last_logits'),
                         get(final[key], 'last_logits'), 'post-cleanup logits mismatch')
        null_checks = null[key]['event']['checks']
        require(set(null_checks) == {'activation', 'logits'} and
                null[key]['event']['untouched_output_bits_equal'] is True, 'null control record mismatch')
        check_comparison(null_checks['activation'], get(null[key], 'before'),
                         get(null[key], 'after'), 'null activation mismatch')
        check_comparison(null_checks['logits'], get(clean[key], 'last_logits'),
                         get(null[key], 'last_logits'), 'null logit mismatch')
        require(exact_equal(null[key]['event']['site'], site_by_id[SITES[0]]), 'wrong null site')
        metadata(null[key], case, SITES[0])

    def witness(row, site, case, expected_source):
        before, after = get(row, 'before'), get(row, 'after')
        whole_before, whole_after = get(row, 'whole_before'), get(row, 'whole_after')
        hidden_bytes = len(before)
        pos = case['subject_index'] if site.endswith(':subject') else len(case['input_ids']) - 1
        begin, end = hidden_bytes * pos, hidden_bytes * (pos + 1)
        require(len(whole_before) == len(whole_after) == hidden_bytes * len(case['input_ids']),
                'whole-output shape mismatch')
        require(whole_before[begin:end] == before and whole_after[begin:end] == after and
                whole_before[:begin] == whole_after[:begin] and whole_before[end:] == whole_after[end:],
                'action modified an unselected output coordinate')
        require(after == expected_source, 'action did not land the named source bits')

    for case in cases:
        key = case['case_id']
        witness(null[key], SITES[0], case, get(null[key], 'before'))
        for controls, site, reference_donor in ((delivery, ALL_SITES[8], True),
                                                 (subject, ALL_SITES[9], False)):
            row = controls[key]
            event = row['event']
            donor_id = event['donor']
            donor = case_map[donor_id]
            require(event['site']['id'] == site and donor['quartet'] == case['quartet'] and
                    donor['lemma'] == case['lemma'] and donor['number'] != case['number'],
                    'position control donor/site mismatch')
            require(exact_equal(event['site'], site_by_id[site]) and
                    event['post_action_matches_donor_bits'] is True and
                    event['untouched_output_bits_equal'] is True, 'position control coordinate or status mismatch')
            metadata(row, case, site)
            witness(row, site, case, get(clean[donor_id], f'capture_{ALL_SITES.index(site)}'))
            reference = donor_id if reference_donor else key
            require(event['reference'] == reference, 'position control reference mismatch')
            check_comparison(event['logits_check'], get(clean[reference], 'last_logits'),
                             get(row, 'last_logits'), 'position control last-token logits mismatch')

    patches = {}
    for row in records:
        event = row['event']
        if event['kind'] != 'patch':
            continue
        target, donor, site = event['target'], event['donor'], event['site']['id']
        require(site in SITES and target in case_map and donor in case_map and
                case_map[target]['quartet'] == case_map[donor]['quartet'] == event['quartet'],
                'invalid patch case/site')
        require(exact_equal(event['site'], site_by_id[site]) and
                event['source_input_sha256'] == input_hash(case_map[donor]) and
                event['target_input_sha256'] == input_hash(case_map[target]) and
                clean[donor]['index'] < row['index'], 'patch coordinate/input binding mismatch')
        metadata(row, case_map[target], site)
        k = (site, target, donor)
        require(k not in patches, 'duplicate patch')
        self_patch = target == donor
        source = get(row, 'before') if self_patch else get(clean[donor], f'capture_{SITES.index(site)}')
        require(event['self_patch'] is self_patch and event['post_action_matches_donor_bits'] is True and
                event['untouched_output_bits_equal'] is True and
                event['source_tensor_sha256'] == hashlib.sha256(source).hexdigest(), 'wrong source identity')
        witness(row, site, case_map[target], source)
        if self_patch:
            require(set(event['checks']) == {'activation', 'logits'}, 'self-patch checks incomplete')
            check_comparison(event['checks']['activation'], get(row, 'before'),
                             get(row, 'after'), 'self-patch activation mismatch')
            check_comparison(event['checks']['logits'], get(clean[target], 'last_logits'),
                             get(row, 'last_logits'), 'self-patch output mismatch')
        else:
            require(event['checks'] == {}, 'non-self patch claims unexpected checks')
        patches[k] = row
    require(set(patches) == {(site, target, donor) for site in SITES for group in groups.values()
                             for target in group for donor in group}, 'patch coverage incomplete')

    def scores(row):
        if 'contrast_logits' in row['tensors']:
            return tuple(map(float, fields(get(row, 'contrast_logits'))))
        values = fields(get(row, 'last_logits'))
        return tuple(float(values[i]) for i in (318, 389, 373, 547))
    clean_scores = {key: scores(row) for key, row in clean.items()}
    metric_events = {(e['kind'], e['site']): e for e in events
                     if e['kind'] in ('present_site_metrics', 'secondary_past_site_metrics')}
    effects, family_effects, sign_flips, baselines = {}, {}, {}, {}
    for kind, a, b, labels in (('present_site_metrics', 0, 1, ('is', 'are')),
                                ('secondary_past_site_metrics', 2, 3, ('was', 'were'))):
        for site in SITES:
            event = metric_events[(kind, site)]
            expected_tokens = ((318, 389) if kind == 'present_site_metrics' else (373, 547))
            require(exact_equal(event['contrast'], {'first_token': ' ' + labels[0], 'first_id': expected_tokens[0],
                                                   'second_token': ' ' + labels[1], 'second_id': expected_tokens[1]}),
                    'contrast labels or token IDs do not match the raw logits')
            result = event['metrics']
            right = sum((1 if case_map[key]['number'] == 'singular' else -1) *
                        math.fsum((value[a], -value[b])) > 0 for key, value in clean_scores.items())
            ties = sum(math.fsum((value[a], -value[b])) == 0 for value in clean_scores.values())
            require(exact_equal(result['baseline'], {'n': 16, 'correct': right, 'incorrect': 16 - right, 'ties': ties}),
                    'baseline accuracy differs from raw logits')
            baselines[kind] = result['baseline']
            by_pair = {(p['target'], p['donor']): p for p in result['pairs']}
            require(len(result['pairs']) == len(by_pair) == 64, 'pair summary incomplete or duplicated')
            by_quartet = {q['quartet']: q for q in result['quartets']}
            require(len(result['quartets']) == len(by_quartet) == 4 and set(by_quartet) == set(groups),
                    'quartet summary incomplete or duplicated')
            family_ds, family_interactions = [], []
            site_family = {}
            flips = {'same_number': 0, 'opposite_number': 0}
            for group in sorted(groups):
                cells = {key: [] for key in CELLS}
                for target in sorted(groups[group]):
                    case = case_map[target]
                    c1, c2 = clean_scores[target][a], clean_scores[target][b]
                    cm = math.fsum((c1, -c2))
                    weight = 1 if case['number'] == 'singular' else -1
                    for donor in sorted(groups[group]):
                        p1, p2 = (scores(patches[(site, target, donor)])[i] for i in (a, b))
                        pm = math.fsum((p1, -p2))
                        shift = weight * math.fsum((cm, -pm))
                        dcase = case_map[donor]
                        cell = ('same_lemma' if case['lemma'] == dcase['lemma'] else 'different_lemma') + '_' + \
                               ('same_number' if case['number'] == dcase['number'] else 'opposite_number')
                        cells[cell].append(shift)
                        observed = by_pair[(target, donor)]
                        expected_values = {'site': site, 'quartet': group, 'cell': cell, 'weight': weight,
                                           'clean_' + labels[0]: c1, 'clean_' + labels[1]: c2,
                                           'patched_' + labels[0]: p1, 'patched_' + labels[1]: p2,
                                           'clean_margin': cm, 'patched_margin': pm, 'signed_target_shift': shift}
                        require(all(k in observed and exact_equal(observed[k], v) for k, v in expected_values.items()),
                                'saved pair effect differs from raw logits')
                        cs, ps = int(cm > 0) - int(cm < 0), int(pm > 0) - int(pm < 0)
                        require(exact_equal(observed['clean_margin_sign'], cs) and
                                exact_equal(observed['patched_margin_sign'], ps) and
                                observed['strict_sign_reversal'] is (cs * ps == -1) and
                                observed['zero_transition'] is (cs != ps and (cs == 0 or ps == 0)),
                                'saved margin sign classification differs')
                        flips['same_number' if case['number'] == dcase['number'] else 'opposite_number'] += int(cs * ps == -1)
                means = {k: math.fsum(v) / 4 for k, v in cells.items()}
                ss, so, ds, do = (means[k] for k in CELLS)
                d = math.fsum((so, -ss, do, -ds)) / 2
                interaction = math.fsum((so, -ss, -do, ds))
                observed = by_quartet[group]
                require(observed['site'] == site and exact_equal(observed['cell_means'], means) and
                        exact_equal(observed['cell_counts'], {k: 4 for k in CELLS}) and
                        exact_equal(observed['number_main_contrast'], d) and
                        exact_equal(observed['lemma_number_interaction'], interaction),
                        'saved family-level contrast differs')
                family_ds.append(d)
                family_interactions.append(interaction)
                site_family[group] = d
            avg = math.fsum(family_ds) / 4
            interaction_avg = math.fsum(family_interactions) / 4
            require(result['site_summary']['site'] == site and
                    exact_equal(result['site_summary']['n_quartets'], 4) and
                    exact_equal(result['site_summary']['mean_number_main_contrast'], avg) and
                    exact_equal(result['site_summary']['mean_lemma_number_interaction'], interaction_avg),
                    'saved site summary differs')
            effects[(kind, site)] = avg
            family_effects[(kind, site)] = site_family
            sign_flips[(kind, site)] = flips

    selection = events[-1]['result']['selection']
    predictions = {site: effects[('present_site_metrics', site)] for site in SITES}
    require(exact_equal(selection['predictions'], predictions), 'prediction differs from raw reconstruction')
    ranked = sorted(SITES, key=lambda site: -predictions[site])
    index = hashlib.sha256(b'MLG-number-v1-random\x00' + (20261028).to_bytes(8, 'big')).digest()[0] % 8
    require(selection['ranking'] == ranked and selection['selected_site'] ==
            (ranked[0] if predictions[ranked[0]] > 0 else None) and
            selection['earliest_site'] == SITES[0] and selection['random_site'] == SITES[index] and
            exact_equal(selection['random_index'], index), 'selection/random baseline mismatch')
    magnitudes = {site: [] for site in SITES}
    expected_pairs = {}
    for site in SITES:
        for group in groups:
            members = [case_map[c] for c in groups[group]]
            for lemma in sorted({c['lemma'] for c in members}):
                sing = next(c['case_id'] for c in members if c['lemma'] == lemma and c['number'] == 'singular')
                plural = next(c['case_id'] for c in members if c['lemma'] == lemma and c['number'] == 'plural')
                left = fields(get(clean[sing], f'capture_{SITES.index(site)}'))
                right = fields(get(clean[plural], f'capture_{SITES.index(site)}'))
                l2 = math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(left, right)))
                magnitudes[site].append(l2)
                expected_pairs[(site, group, lemma, sing, plural)] = l2
    actual_pairs = {(r['site'], r['quartet'], r['lemma'], r['singular'], r['plural']): r['l2']
                    for r in selection['magnitude_pairs']}
    require(exact_equal(actual_pairs, expected_pairs) and len(selection['magnitude_pairs']) == 64,
            'magnitude pair values differ from stored activations')
    means = {site: math.fsum(magnitudes[site]) / 8 for site in SITES}
    require(exact_equal(selection['magnitude_means'], means) and
            selection['magnitude_site'] == sorted(SITES, key=lambda site: -means[site])[0],
            'magnitude baseline mismatch')
    return {'schema': 'mlg-independent-natural-discovery-audit-v1',
            'status': 'independent_saved_evidence_pass', 'journal_event_sha256': hashlib.sha256(event_raw).hexdigest(),
            'journal_started_sha256': hashlib.sha256(started_raw).hexdigest(),
            'journal_terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(),
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'preflight_manifest_sha256': None if synthetic else MANIFEST_SHA,
            'tensor_count': len(cache), 'tensor_bytes': sum(sizes.values()), 'forwards': 608,
            'present_predictions': predictions, 'selected_site': selection['selected_site'],
            'present_baseline': baselines['present_site_metrics'],
            'past_baseline': baselines['secondary_past_site_metrics'],
            'present_family_effects': {site: family_effects[('present_site_metrics', site)] for site in SITES},
            'present_strict_sign_reversals': {site: sign_flips[('present_site_metrics', site)] for site in SITES},
            'random_site': selection['random_site'], 'earliest_site': selection['earliest_site'],
            'magnitude_site': selection['magnitude_site'],
            'secondary_past_predictions': {site: effects[('secondary_past_site_metrics', site)] for site in SITES},
            'scope': ('Fabricated synthetic data only.' if synthetic else
                      'Saved public development data only; no held-out or general causal claim.')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('journal')
    parser.add_argument('--preflight-manifest')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', help='new audit JSON file; never overwrites an existing file')
    args = parser.parse_args()
    require(args.synthetic or args.preflight_manifest, 'pinned manifest required')
    result = inspect(args.journal, args.preflight_manifest, args.synthetic)
    raw = json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False,
                     separators=(',', ':')).encode('utf-8')
    if args.output:
        with Path(args.output).open('xb') as stream:
            require(stream.write(raw) == len(raw), 'short audit output write')
            stream.flush()
            os.fsync(stream.fileno())
    else:
        sys.stdout.buffer.write(raw + b'\n')
