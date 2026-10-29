"""Independent saved-tensor audit after natural-agreement held-out reveal.

This uses no Model Looking Glass implementation and never runs a model. The
saved journal is audited against pre-existing pins; publication, faithful
execution, external custody and final cleanup require separate verification.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np


SITES = [f'{layer}:{kind}:last' for layer in range(1, 5) for kind in ('attn', 'mlp')]
ALL_SITES = SITES + ['5:block:last', '5:block:subject']
CELLS = ('same_lemma_same_number', 'same_lemma_opposite_number',
         'different_lemma_same_number', 'different_lemma_opposite_number')
MAX_EVENT_LOG_BYTES = 48 * 1024 * 1024
MAX_RETAINED_BYTES = 160 * 1024 * 1024


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



def prediction_errors(prediction, effects):
    """Score one frozen prediction and zero on exactly four family effects."""
    limit = 32 * 3.4028234663852886e38
    if (type(prediction) is not float or not math.isfinite(prediction) or abs(prediction) > limit
            or type(effects) not in (tuple, list) or len(effects) != 4
            or any(type(value) is not float or not math.isfinite(value) or abs(value) > limit for value in effects)):
        raise ValueError('four finite family effects and a finite prediction are required')
    errors = [math.fsum((effect, -prediction)) for effect in effects]

    def summarize(values):
        return {'mae': math.fsum(abs(value) for value in values) / 4,
                'rmse': math.sqrt(math.fsum(value * value for value in values) / 4)}

    return {'errors': errors, 'frozen_predictor': summarize(errors),
            'zero_predictor': summarize(effects)}


PROTOCOL_SHA = '95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee'
CHECKPOINT = '2290a62682d06624634c1f46a6ad5be0f47f38aa'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def decode_canonical(raw, limit):
    require(type(raw) is bytes and len(raw) <= limit, 'invalid or oversized JSON bytes')
    value = json.loads(raw)
    require(type(value) is dict and canonical(value) == raw, 'JSON is not canonical')
    return value


def read_bounded(path, limit):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400
            and info.st_nlink == 1, 'artifact is not an unaliased ordinary file')
    require(info.st_size <= limit, 'artifact exceeds its byte limit')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) == info.st_size, 'artifact changed while reading')
    return data


def check_directory(path):
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
            'artifact directory is a link or reparse point')


def is_hex(value, length):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def verify_binding(prediction_bytes, freeze_bytes, heldout_bytes, split_nonce,
                   expected_freeze_sha256, expected_commitment):
    require(type(split_nonce) is bytes and len(split_nonce) == 32, 'split nonce must be 32 bytes')
    require(is_hex(expected_freeze_sha256, 64) and is_hex(expected_commitment, 64), 'external pins are missing')
    freeze = decode_canonical(freeze_bytes, 1024 * 1024)
    require(hashlib.sha256(freeze_bytes).hexdigest() == expected_freeze_sha256, 'prediction freeze pin mismatch')
    require(set(freeze) == {'schema', 'protocol_sha256', 'checkpoint_revision', 'tracks'} and
            freeze['schema'] == 'mlg-number-prediction-freezes-v1' and
            freeze['protocol_sha256'] == PROTOCOL_SHA and freeze['checkpoint_revision'] == CHECKPOINT,
            'prediction freeze identity mismatch')
    tracks = freeze['tracks']
    require(type(tracks) is dict and 'natural' in tracks and set(tracks) <= {'natural', 'injection'},
            'natural track is not frozen')
    for track in tracks.values():
        require(type(track) is dict and set(track) == {'source_commit', 'runtime_sha256', 'prediction_sha256'} and
                is_hex(track['source_commit'], 40) and is_hex(track['runtime_sha256'], 64) and
                is_hex(track['prediction_sha256'], 64), 'invalid frozen track')
    prediction = decode_canonical(prediction_bytes, 8 * 1024 * 1024)
    prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
    require(prediction_sha == tracks['natural']['prediction_sha256'] and
            set(prediction) == {'schema', 'protocol_sha256', 'source_commit', 'checkpoint_revision',
                                'discovery_event_sha256', 'independent_audit_sha256', 'selection',
                                'present_site_metrics', 'secondary_past_site_metrics'} and
            prediction['schema'] == 'mlg-number-natural-predictions-v2' and
            prediction['protocol_sha256'] == PROTOCOL_SHA and prediction['checkpoint_revision'] == CHECKPOINT and
            prediction['source_commit'] == tracks['natural']['source_commit'] and
            is_hex(prediction['discovery_event_sha256'], 64) and is_hex(prediction['independent_audit_sha256'], 64),
            'frozen prediction identity mismatch')
    selection = prediction['selection']
    require(type(selection) is dict and type(selection.get('predictions')) is dict and
            set(selection['predictions']) == set(SITES) and
            all(type(value) is float and math.isfinite(value) and abs(value) <= 32 * 3.4028234663852886e38
                for value in selection['predictions'].values()) and
            selection.get('selected_site') in (None, *SITES) and
            all(selection.get(name) in SITES for name in ('earliest_site', 'random_site', 'magnitude_site')),
            'frozen predictions or site labels are invalid')
    require(selection['earliest_site'] == SITES[0], 'frozen earliest baseline differs')
    random_index = hashlib.sha256(b'MLG-number-v1-random\x00' + (20261028).to_bytes(8, 'big')).digest()[0] % 8
    require(selection['random_site'] == SITES[random_index] and exact_equal(selection.get('random_index'), random_index),
            'frozen random baseline differs')
    reveal = decode_canonical(heldout_bytes, 1024 * 1024)
    actual_commitment = hashlib.sha256(b'MLG-number-v1-heldout-commit\x00' + split_nonce + b'\x00' + heldout_bytes).hexdigest()
    require(actual_commitment == expected_commitment, 'held-out reveal commitment mismatch')
    require(set(reveal) == {'schema', 'protocol_sha256', 'source_commits', 'checkpoint_revision',
                            'prediction_freeze_sha256', 'exposure_inventory_sha256', 'families'} and
            reveal['schema'] == 'mlg-number-heldout-v1' and reveal['protocol_sha256'] == PROTOCOL_SHA and
            reveal['checkpoint_revision'] == CHECKPOINT and reveal['prediction_freeze_sha256'] == expected_freeze_sha256 and
            reveal['source_commits'] == {name: track['source_commit'] for name, track in tracks.items()} and
            is_hex(reveal['exposure_inventory_sha256'], 64), 'held-out reveal identity mismatch')
    families = reveal['families']
    require(type(families) is list and len(families) == 4, 'four revealed families are required')
    cases, all_rows = [], []
    for family_index, family in enumerate(families):
        require(type(family) is dict and set(family) == {'family_index', 'context', 'subjects', 'cases'} and
                exact_equal(family['family_index'], family_index) and type(family['context']) is str and
                type(family['subjects']) is list and len(family['subjects']) == 4 and
                all(type(subject) is str and subject for subject in family['subjects']) and
                type(family['cases']) is list and len(family['cases']) == 4, 'invalid revealed family')
        reference = None
        subject_tokens = []
        for item_index, item in enumerate(family['cases']):
            require(type(item) is dict and set(item) == {'text', 'input_ids', 'attention_mask', 'subject_index'} and
                    item['text'] == 'The ' + family['subjects'][item_index] + ' ' + family['context'],
                    'invalid revealed sentence')
            ids, mask, pos = item['input_ids'], item['attention_mask'], item['subject_index']
            require(type(ids) is list and 3 <= len(ids) <= 32 and
                    all(type(x) is int and 0 <= x <= 50256 for x in ids) and
                    type(mask) is list and len(mask) == len(ids) and all(type(x) is int and x == 1 for x in mask) and
                    type(pos) is int and 0 < pos < len(ids) - 1, 'invalid revealed token metadata')
            common = (len(ids), pos, ids[:pos] + ids[pos + 1:])
            if reference is None:
                reference = common
            require(common == reference, 'revealed family token positions differ')
            subject_tokens.append(ids[pos])
            all_rows.append(tuple(ids))
            cases.append({'case_id': f'heldout-{family_index}-{item_index}', 'quartet': f'heldout-{family_index}',
                          'lemma': f'lemma-{item_index // 2}',
                          'number': 'singular' if item_index % 2 == 0 else 'plural',
                          'input_ids': ids, 'attention_mask': mask, 'subject_index': pos})
        require(len(set(subject_tokens)) == 4, 'revealed subject tokens are not distinct')
    require(len(set(all_rows)) == 16, 'revealed token rows are duplicated')
    links = {'prediction_sha256': prediction_sha, 'prediction_freeze_sha256': expected_freeze_sha256,
             'heldout_commitment': expected_commitment, 'exposure_inventory_sha256': reveal['exposure_inventory_sha256'],
             'discovery_event_sha256': prediction['discovery_event_sha256'],
             'independent_audit_sha256': prediction['independent_audit_sha256']}
    return cases, selection, links

def inspect(directory, *, prediction_bytes, freeze_bytes, heldout_bytes, split_nonce,
            expected_freeze_sha256, expected_commitment, synthetic=False):
    expected_cases, selection, links = verify_binding(
        prediction_bytes, freeze_bytes, heldout_bytes, split_nonce,
        expected_freeze_sha256, expected_commitment)
    directory = Path(directory)
    check_directory(directory)
    require({p.name for p in directory.iterdir()} ==
            {'started.json', 'events.jsonl', 'terminal.json', 'tensors'},
            'unexpected or missing journal artifacts')
    check_directory(directory / 'tensors')
    event_raw = read_bounded(directory / 'events.jsonl', MAX_EVENT_LOG_BYTES)
    terminal_raw = read_bounded(directory / 'terminal.json', 1024 * 1024)
    started_raw = read_bounded(directory / 'started.json', 1024 * 1024)
    require(terminal_raw.endswith(b'\n') and started_raw.endswith(b'\n'), 'journal metadata is not newline terminated')
    terminal = decode_canonical(terminal_raw[:-1], 1024 * 1024)
    decode_canonical(started_raw[:-1], 1024 * 1024)
    require(event_raw.endswith(b'\n'), 'event journal is not newline terminated')
    records = [decode_canonical(line[:-1], 256 * 1024) for line in event_raw.splitlines(keepends=True)]
    require(all(set(row) == {'index', 'event', 'tensors'} for row in records), 'invalid journal row schema')
    require(set(terminal) == {'status', 'detail', 'events', 'retained_tensor_bytes',
                              'event_log_bytes', 'storage_accounting_complete'} and
            terminal['status'] == 'complete_candidate' and
            terminal.get('storage_accounting_complete') is True and
            exact_equal(terminal['events'], len(records)),
            'not a complete candidate journal')
    require(exact_equal(terminal['event_log_bytes'], len(event_raw)), 'event-byte accounting mismatch')
    require(type(terminal['retained_tensor_bytes']) is int and
            0 <= terminal['retained_tensor_bytes'] <= MAX_RETAINED_BYTES,
            'invalid retained-byte accounting')
    require(exact_equal([r['index'] for r in records], list(range(len(records)))), 'event ordering mismatch')
    events = [r['event'] for r in records]
    require(events[0]['kind'] == 'run_contract' and events[-1]['kind'] == 'summary', 'missing endpoints')
    contract = events[0]
    cases = contract['cases']
    require(exact_equal(cases, expected_cases), 'run cases differ from the revealed commitment')
    require(contract.get('phase') == 'heldout' and all(contract.get(key) == value for key, value in links.items()),
            'run contract differs from the externally pinned freeze or reveal')
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
    require(exact_equal(contract['cpu_rng_seed'], 20261028) and exact_equal(contract['cpu_threads'], 1) and
            (synthetic or contract['deterministic_algorithms'] is True), 'unexpected CPU execution contract')
    configuration = json.dumps(contract['execution_configuration'], ensure_ascii=True,
                               separators=(',', ':')).encode('ascii')
    require(hashlib.sha256(configuration).hexdigest() == contract['execution_configuration_sha256'],
            'execution configuration digest mismatch')
    kinds = Counter(e['kind'] for e in events)
    expected = {'run_contract': 1, 'clean_capture': 16, 'clean_repeat': 16, 'null': 16,
                'patch': 512, 'delivery': 16, 'subject_position': 16, 'final_clean': 16,
                'present_site_metrics': 8, 'secondary_past_site_metrics': 8, 'summary': 1}
    require(kinds == expected, 'unexpected number of forward/control/summary records')
    require([(event['kind'], event.get('site')) for event in events[-17:-1]] ==
            [(kind, site) for kind in ('present_site_metrics', 'secondary_past_site_metrics') for site in SITES],
            'site metric events are missing, reordered or interleaved')
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
            (('device_rng' not in summary and 'device_rng' not in contract) or
             ('device_rng' in summary and 'device_rng' in contract and
              exact_equal(summary['device_rng'], contract['device_rng']))) and
            type(contract['seeded_cpu_rng_sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', contract['seeded_cpu_rng_sha256']) is not None and
            summary['seeded_cpu_rng_sha256'] == contract['seeded_cpu_rng_sha256'] and
            type(summary['model_state_sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', summary['model_state_sha256']) is not None,
            'summary and contract seed or state identity differs')

    cache, sizes = {}, {}
    def get(row, name):
        ref = row['tensors'][name]
        require(type(ref) is dict and set(ref) == {'sha256', 'dtype', 'shape', 'bytes'}, 'invalid tensor reference schema')
        digest = ref['sha256']
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) is not None, 'invalid tensor digest')
        require(type(ref['bytes']) is int and ref['dtype'] == 'float32-le' and exact_equal(ref['bytes'], 4 * math.prod(ref['shape'])),
                'invalid tensor reference')
        if digest not in cache:
            data = read_bounded(directory / 'tensors' / (digest + '.f32le'), ref['bytes'])
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
    require(sum(sizes.values()) <= MAX_RETAINED_BYTES and len(event_raw) <= MAX_EVENT_LOG_BYTES,
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
            require(set(event) == {'kind', 'site', 'contrast', 'metrics'} and
                    type(result) is dict and set(result) == {'baseline', 'site_summary', 'quartets', 'pairs'},
                    'site metric event has an invalid schema')
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
                        require(set(observed) == set(expected_values) | {'target', 'donor', 'clean_margin_sign',
                                'patched_margin_sign', 'strict_sign_reversal', 'zero_transition'} and
                                all(k in observed and exact_equal(observed[k], v) for k, v in expected_values.items()),
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
                require(set(observed) == {'site', 'quartet', 'cell_means', 'cell_counts',
                                          'number_main_contrast', 'lemma_number_interaction'} and
                        observed['site'] == site and exact_equal(observed['cell_means'], means) and
                        exact_equal(observed['cell_counts'], {k: 4 for k in CELLS}) and
                        exact_equal(observed['number_main_contrast'], d) and
                        exact_equal(observed['lemma_number_interaction'], interaction),
                        'saved family-level contrast differs')
                family_ds.append(d)
                family_interactions.append(interaction)
                site_family[group] = d
            avg = math.fsum(family_ds) / 4
            interaction_avg = math.fsum(family_interactions) / 4
            require(set(result['site_summary']) == {'site', 'n_quartets', 'mean_number_main_contrast',
                                                     'mean_lemma_number_interaction'} and
                    result['site_summary']['site'] == site and
                    exact_equal(result['site_summary']['n_quartets'], 4) and
                    exact_equal(result['site_summary']['mean_number_main_contrast'], avg) and
                    exact_equal(result['site_summary']['mean_lemma_number_interaction'], interaction_avg),
                    'saved site summary differs')
            effects[(kind, site)] = avg
            family_effects[(kind, site)] = site_family
            sign_flips[(kind, site)] = flips

    site_rows = []
    for site in SITES:
        families = family_effects[('present_site_metrics', site)]
        names = sorted(groups)
        values = [families[name] for name in names]
        scored = prediction_errors(selection['predictions'][site], values)
        site_rows.append({'site': site, 'frozen_prediction': selection['predictions'][site],
                          'families': [{'quartet': name, 'effect': value, 'prediction_error': error}
                                       for name, value, error in zip(names, values, scored['errors'])],
                          'mean_effect': math.fsum(values) / 4,
                          'frozen_predictor': scored['frozen_predictor'],
                          'zero_predictor': scored['zero_predictor']})
    clean_cases = []
    for case in cases:
        c1, c2 = clean_scores[case['case_id']][:2]
        margin = math.fsum((c1, -c2))
        weight = 1 if case['number'] == 'singular' else -1
        clean_cases.append({'id': case['case_id'], 'quartet': case['quartet'], 'lemma': case['lemma'],
                            'number': case['number'], 'clean_is': c1, 'clean_are': c2,
                            'clean_margin': margin, 'correct': weight * margin > 0})
    evaluation = {'selected_site': selection['selected_site'], 'earliest_site': selection['earliest_site'],
                  'random_site': selection['random_site'], 'magnitude_site': selection['magnitude_site'],
                  'sites': site_rows, **links, 'clean_baseline': baselines['present_site_metrics'],
                  'clean_cases': clean_cases}
    require('selection' not in summary and exact_equal(summary.get('heldout_evaluation'), evaluation),
            'held-out results do not match frozen predictions and saved tensors')
    return {'schema': 'mlg-independent-natural-heldout-audit-v1',
            'status': 'independent_saved_evidence_pass', 'journal_event_sha256': hashlib.sha256(event_raw).hexdigest(),
            'journal_started_sha256': hashlib.sha256(started_raw).hexdigest(),
            'journal_terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(),
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **links, 'tensor_count': len(cache), 'tensor_bytes': sum(sizes.values()), 'forwards': 608,
            'selected_site': selection['selected_site'], 'earliest_site': selection['earliest_site'],
            'random_site': selection['random_site'], 'magnitude_site': selection['magnitude_site'],
            'sites': site_rows, 'clean_cases': clean_cases,
            'present_metrics': [metric_events[('present_site_metrics', site)] for site in SITES],
            'secondary_past_metrics': [metric_events[('secondary_past_site_metrics', site)] for site in SITES],
            'present_baseline': baselines['present_site_metrics'],
            'past_baseline': baselines['secondary_past_site_metrics'],
            'present_strict_sign_reversals': {site: sign_flips[('present_site_metrics', site)] for site in SITES},
            'secondary_past_mean_effects': {site: effects[('secondary_past_site_metrics', site)] for site in SITES},
            'scope': ('Fabricated synthetic data only.' if synthetic else
                      'Saved held-out evidence only; publication, execution and cleanup require separate verification.')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit saved evidence after held-out reveal; no model is run.')
    parser.add_argument('journal')
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--reveal', required=True)
    parser.add_argument('--nonce-file', required=True, help='32-byte split nonce, used only after reveal')
    parser.add_argument('--freeze-sha256', required=True, help='externally verified published freeze pin')
    parser.add_argument('--commitment', required=True, help='externally verified held-out commitment')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', help='new audit JSON file; never overwrites an existing file')
    args = parser.parse_args()
    if args.output:
        require(not Path(args.output).resolve().is_relative_to(Path(args.journal).resolve()),
                'audit output must be outside the saved journal')
    result = inspect(args.journal, prediction_bytes=read_bounded(Path(args.prediction), 8 * 1024 * 1024),
                     freeze_bytes=read_bounded(Path(args.freeze), 1024 * 1024),
                     heldout_bytes=read_bounded(Path(args.reveal), 1024 * 1024),
                     split_nonce=read_bounded(Path(args.nonce_file), 32),
                     expected_freeze_sha256=args.freeze_sha256, expected_commitment=args.commitment,
                     synthetic=args.synthetic)
    raw = canonical(result)
    if args.output:
        with Path(args.output).open('xb') as stream:
            require(stream.write(raw) == len(raw), 'short audit output write')
            stream.flush()
            os.fsync(stream.fileno())
    else:
        sys.stdout.buffer.write(raw + b'\n')
