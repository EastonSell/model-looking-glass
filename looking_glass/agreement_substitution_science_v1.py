"""Model-free replay of the fixed public noun-substitution agreement study.

This checks saved bytes and declared bookkeeping, not model execution, process
exit, unknown exposure or the finiteness of unsaved full-sequence logits. The
caller must independently verify this reader before executing it and bind the
published inputs, owner exit and live public/custody checks separately.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import ModuleType


CASE_SHA = '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe'
PROTOCOL_SHA = 'cc43267dbb412234a31844e837263541d0e16445eb9d2cb45fcb9ce7a9614796'
NUMERIC_SHA = '917694d448b796359db0002a630877e5226a2583d273092871892fade7a38ac4'
DIRECTION_SHA = {'u': 'd1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
    'v1': '4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
    'v2': '4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}
CONDITIONS = ('none', 'attractor_singular', 'attractor_plural')
PAIRS = (('R1', 'R2'), ('I1', 'I2'))
MARGINS = ('present', 'past', 'the_vs_a', 'in_vs_on')
AXES = ('u', 'v1', 'v2')
ARM_ORDER = ('As Ap Bs Bp', 'Ap As Bp Bs', 'Bs Bp As Ap',
             'Bp Bs Ap As', 'As Bs Ap Bp', 'Bp Ap Bs As')
REGULAR_PLUS_FIRST = (True, False, True, False, False, True)
TARGETS = (318, 389, 373, 547, 262, 257, 287, 319)


def bounded_file(path, limit):
    path = Path(path)
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                or info.st_size > limit):
            raise ValueError('evidence is not a bounded ordinary file')
        with path.open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit:
            raise ValueError('evidence changed or exceeded its bound')
        return raw
    except OSError as error:
        raise ValueError('evidence is unavailable') from error


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite JSON value')
        return result
    def reject(_):
        raise ValueError('nonfinite JSON constant')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_float=number, parse_constant=reject)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError('invalid JSON evidence') from error


def load_numeric(path):
    raw = bounded_file(path, 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != NUMERIC_SHA:
        raise ValueError('reviewed saved-tensor arithmetic differs')
    module = ModuleType('_verified_substitution_numeric')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module


def expected_schedule(cases):
    """Derive all case and arm placements directly from the frozen table."""
    if type(cases) not in (list, tuple) or len(cases) != 54:
        raise ValueError('the fixed 54 cases are required')
    seen, by_coordinate, contexts = set(), {}, {}
    for frame in range(2):
        for condition, label in enumerate(CONDITIONS):
            start = 27 * frame + 9 * condition
            for i, case in enumerate(cases[start:start + 9]):
                if type(case) is not dict:
                    raise ValueError('case is invalid')
                name = case.get('case_id')
                if (type(name) is not str or not name or len(name) > 128 or name in seen
                        or type(case.get('frame_index')) is not int or case['frame_index'] != frame
                        or type(case.get('condition_index')) is not int or case['condition_index'] != condition
                        or case.get('condition') != label):
                    raise ValueError('canonical case identity differs')
                seen.add(name)
                if i == 8:
                    wanted = ('context', None, None, None, None, None)
                    contexts[frame, condition] = (start + i, case, None)
                else:
                    morph, position = divmod(i, 4)
                    lemma, number = divmod(position, 2)
                    if type(case.get('morphology_index')) is not int or type(case.get('lemma_index')) is not int:
                        raise ValueError('case coordinates must be integers')
                    wanted = ('target', PAIRS[morph][lemma], ('singular', 'plural')[number],
                              ('regular', 'irregular')[morph], morph, lemma)
                    by_coordinate[frame, condition, morph, ('As', 'Ap', 'Bs', 'Bp')[position]] = (
                        start + i, case, morph)
                actual = tuple(case.get(key) for key in ('role', 'subject_key', 'subject_number',
                               'morphology', 'morphology_index', 'lemma_index'))
                if actual != wanted:
                    raise ValueError('canonical role, lemma or number order differs')
    schedule = []
    for frame in range(2):
        for condition, label in enumerate(CONDITIONS):
            n = 3 * frame + condition
            ordered = []
            for morph in ((0, 1) if (frame + condition) % 2 == 0 else (1, 0)):
                ordered.extend(by_coordinate[frame, condition, morph, name] for name in ARM_ORDER[n].split())
            ordered.append(contexts[frame, condition])
            for case_index, case, morph in ordered:
                plus_first = ((frame + condition) % 2 == 0 if morph is None
                              else REGULAR_PLUS_FIRST[n] != bool(morph))
                signs = ('plus', 'minus') if plus_first else ('minus', 'plus')
                slots = {((2 * i + j - 2 * condition) % 6): axis + '_' + sign
                         for i, axis in enumerate(AXES) for j, sign in enumerate(signs)}
                actions = [('clean', None), ('noop_l3_final', None)]
                actions += [(slots[slot], slot) for slot in range(6)]
                actions.append(('clean_after', None))
                for action, slot in actions:
                    schedule.append({'index': len(schedule), 'case_id': case['case_id'],
                        'case_index': case_index, 'frame_index': frame, 'condition': label,
                        'condition_index': condition, 'role': case['role'], 'morphology_index': morph,
                        'lemma_index': case.get('lemma_index'), 'subject_key': case.get('subject_key'),
                        'subject_number': case.get('subject_number'), 'morphology': case.get('morphology'),
                        'action': action, 'slot': slot})
    return tuple(schedule)


def validate_cases(cases, vocabulary):
    expected_schedule(cases)
    if type(vocabulary) is not int or vocabulary < 548:
        raise ValueError('invalid vocabulary')
    seen = []
    for case in cases:
        tokens = case.get('token_ids')
        if (type(tokens) is not list or not 1 <= len(tokens) <= 12
                or any(type(x) is not int or not 0 <= x < vocabulary for x in tokens)
                or type(case.get('final_index')) is not int or case['final_index'] != len(tokens) - 1
                or any(tokens[:len(other)] == other or other[:len(tokens)] == tokens for other in seen)):
            raise ValueError('input token row is invalid or prefix-overlapping')
        index = case.get('subject_index')
        if case['role'] == 'target':
            if type(index) is not int or not 0 < index < len(tokens) - 1:
                raise ValueError('target subject coordinate differs')
        elif index is not None:
            raise ValueError('subjectless case has a subject coordinate')
        seen.append(tokens)
    for frame in range(2):
        for cond in range(3):
            block = cases[frame * 27 + cond * 9:frame * 27 + cond * 9 + 8]
            first = block[0]
            for case in block[1:]:
                a, b, i = first['token_ids'], case['token_ids'], first['subject_index']
                if (case['subject_index'] != i or len(a) != len(b) or a[i] == b[i]
                        or a[:i] + a[i+1:] != b[:i] + b[i+1:]):
                    raise ValueError('target tokens differ outside subject')
        for i in range(9):
            a = cases[frame*27 + 9 + i]['token_ids']
            b = cases[frame*27 + 18 + i]['token_ids']
            if len(a) != len(b) or a[:-1] != b[:-1] or a[-1] == b[-1]:
                raise ValueError('attractor inputs differ outside the final noun')
    return True


def mean(values):
    values = list(values)
    if not values or any(type(x) is not float or not math.isfinite(x) for x in values):
        raise ValueError('finite nonempty sample required')
    return math.fsum(values) / len(values)


def factorial(values):
    if type(values) not in (tuple, list) or len(values) != 4:
        raise ValueError('four factorial values are required')
    a_s, a_p, b_s, b_p = values
    if any(type(x) is not float or not math.isfinite(x) for x in values):
        raise ValueError('nonfinite factorial value')
    cross = (a_s - b_p, b_s - a_p, a_s - b_s, a_p - b_p)
    opposite = mean((abs(cross[0]), abs(cross[1])))
    same = mean((abs(cross[2]), abs(cross[3])))
    return {'signed': {'a_s_minus_b_p': cross[0], 'b_s_minus_a_p': cross[1],
                       'a_s_minus_b_s': cross[2], 'a_p_minus_b_p': cross[3]},
            'opposite': opposite, 'same': same, 'delta': opposite - same,
            'number': mean((a_s - a_p, b_s - b_p)),
            'lemma': mean((a_s - b_s, a_p - b_p)),
            'interaction': (a_s - a_p) - (b_s - b_p)}


def score_cases(cases, margins):
    """Apply the prospective denominator, thresholds and all four contrasts."""
    expected_schedule(cases)
    if type(margins) is not dict or set(margins) != {row['case_id'] for row in cases}:
        raise ValueError('case margin inventory differs')
    per_case, lookup = [], {}
    action_names = {'clean', 'noop_l3_final', 'clean_after'} | {
        axis + '_' + sign for axis in AXES for sign in ('plus', 'minus')}
    for row in cases:
        actions = margins[row['case_id']]
        if type(actions) is not dict or set(actions) != action_names:
            raise ValueError('case action inventory differs')
        for value in actions.values():
            if (type(value) is not dict or set(value) != set(MARGINS)
                    or any(type(x) is not float or not math.isfinite(x) for x in value.values())):
                raise ValueError('case margin is malformed')
        clean = actions['clean']
        odd, even = {}, {}
        for axis in AXES:
            plus, minus = actions[axis + '_plus'], actions[axis + '_minus']
            odd[axis] = {key: mean((plus[key], -minus[key])) for key in MARGINS}
            even[axis] = {key: mean((plus[key], minus[key])) - clean[key] for key in MARGINS}
        result = {'case_id': row['case_id'], 'frame_index': row['frame_index'],
                  'condition': row['condition'], 'condition_index': row['condition_index'],
                  'role': row['role'], 'morphology': row['morphology'],
                  'subject_key': row['subject_key'], 'subject_number': row['subject_number'],
                  'margins': actions, 'odd': odd, 'even': even}
        per_case.append(result)
        if row['role'] == 'target':
            lookup[row['frame_index'], row['condition_index'], row['morphology'],
                   row['subject_key'], row['subject_number']] = result
    cells = []
    for frame in range(2):
        for cond, label in enumerate(CONDITIONS):
            for mi, morph in enumerate(('regular', 'irregular')):
                a, b = PAIRS[mi]
                members = [lookup[frame, cond, morph, key, number]
                           for key, number in ((a, 'singular'), (a, 'plural'),
                                               (b, 'singular'), (b, 'plural'))]
                values = {metric: {axis: factorial([item['odd'][axis][metric] for item in members])
                                   for axis in AXES} for metric in MARGINS}
                cells.append({'frame_index': frame, 'condition_index': cond, 'condition': label,
                              'morphology': morph, 'case_ids': [x['case_id'] for x in members],
                              'contrasts': values})
    targets = [x for x in per_case if x['role'] == 'target']
    def correct(item):
        margin = item['margins']['clean']['present']
        return margin > 0.0 if item['subject_number'] == 'singular' else margin < 0.0
    def counts(rows):
        return {'correct': sum(correct(x) for x in rows), 'total': len(rows)}
    by_morph = {name: counts([x for x in targets if x['morphology'] == name])
                for name in ('regular', 'irregular')}
    by_frame = {str(frame): counts([x for x in targets if x['frame_index'] == frame]) for frame in range(2)}
    by_condition = {label: counts([x for x in targets if x['condition'] == label]) for label in CONDITIONS}
    by_lemma = {key: counts([x for x in targets if x['subject_key'] == key])
                for key in ('R1', 'R2', 'I1', 'I2')}
    by_id = {x['case_id']: x for x in targets}
    complete_cells = sum(all(correct(by_id[case_id]) for case_id in c['case_ids']) for c in cells)
    overall = counts(targets)
    adequate = (overall['correct'] >= 36
        and all(x['correct'] >= 18 for x in by_morph.values())
        and all(x['correct'] >= 18 for x in by_frame.values())
        and all(x['correct'] >= 12 for x in by_condition.values())
        and all(x['correct'] >= 9 for x in by_lemma.values()) and complete_cells >= 8)
    baseline = {'overall': overall, 'by_morphology': by_morph, 'by_frame': by_frame,
                'by_condition': by_condition, 'by_lemma': by_lemma,
                'complete_four_correct_cells': complete_cells, 'adequate': adequate}
    def group_summary(group):
        return {metric: {axis: {**{key: mean(c['contrasts'][metric][axis][key] for c in group)
                                   for key in ('opposite', 'same', 'delta', 'number', 'lemma', 'interaction')},
                                'mean_absolute_delta': mean(abs(c['contrasts'][metric][axis]['delta'])
                                                            for c in group)} for axis in AXES}
                for metric in MARGINS}
    aggregates = {'overall': group_summary(cells),
        'by_frame': {str(i): group_summary([c for c in cells if c['frame_index'] == i]) for i in range(2)},
        'by_condition': {label: group_summary([c for c in cells if c['condition'] == label])
                         for label in CONDITIONS},
        'by_morphology': {name: group_summary([c for c in cells if c['morphology'] == name])
                          for name in ('regular', 'irregular')}}
    overall_axes = aggregates['overall']['present']
    morph_axes = {name: aggregates['by_morphology'][name]['present'] for name in ('regular', 'irregular')}
    positive = sum(c['contrasts']['present']['u']['delta'] > 0.0 for c in cells)
    positive_by_morph = {name: sum(c['contrasts']['present']['u']['delta'] > 0.0
                              for c in cells if c['morphology'] == name)
                         for name in ('regular', 'irregular')}
    flags = {'overall_mean_and_controls': overall_axes['u']['delta'] >= 0.05 and all(
                overall_axes['u']['delta'] > overall_axes[axis]['mean_absolute_delta'] for axis in ('v1', 'v2')),
             'morphology_means_and_controls': all(a['u']['delta'] >= 0.02 and all(
                a['u']['delta'] > a[axis]['mean_absolute_delta'] for axis in ('v1', 'v2'))
                for a in morph_axes.values()),
             'positive_counts': positive >= 8 and all(x >= 4 for x in positive_by_morph.values()),
             'frame_means_positive': all(aggregates['by_frame'][str(i)]['present']['u']['delta'] > 0.0
                                         for i in range(2))}
    criteria = {'flags': flags, 'positive_cells': positive, 'positive_by_morphology': positive_by_morph,
                'baseline_limited': not adequate, 'pass': adequate and all(flags.values()),
                'interpretation': ('baseline_limited' if not adequate else
                                   'criterion_pass' if all(flags.values()) else 'criterion_not_met')}
    return {'cases': per_case, 'cells': cells, 'fragments': [x for x in per_case if x['role'] == 'context'],
            'baseline': baseline, 'aggregates': aggregates, 'criterion': criteria}


def validate_directions(directions, width):
    if type(directions) is not dict or set(directions) != set(AXES):
        raise ValueError('three fixed directions are required')
    for direction in directions.values():
        if (type(direction) not in (tuple, list) or len(direction) != width
                or any(type(x) is not float or not math.isfinite(x) for x in direction)
                or abs(math.sqrt(math.fsum(x * x for x in direction)) - 1.0) > 1e-12):
            raise ValueError('direction is not finite and unit length')
    for left, right in (('u', 'v1'), ('u', 'v2'), ('v1', 'v2')):
        if abs(math.fsum(a*b for a, b in zip(directions[left], directions[right]))) > 1e-12:
            raise ValueError('directions are not orthogonal')


def validate_runtime(contract, science, width, vocabulary):
    environment = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None}
    numeric = [['deterministic_algorithms', True], ['deterministic_warn_only', False],
               ['cpu_threads', 1], ['cpu_interop_threads', 1], ['float32_matmul_precision', 'highest'],
               ['calling_thread_denormal_product_bits', 0x00400000]]
    configuration = contract['execution_configuration']
    if (not science.same_recorded_value(contract['environment'], environment)
            or type(configuration) is not list or len(configuration) != 2
            or not science.same_recorded_value(configuration[0], numeric)
            or type(configuration[1]) is not list or not configuration[1]):
        raise ValueError('recorded CPU runtime differs')
    root = configuration[1][0]
    if (type(root) is not list or len(root) != 3 or root[0] != ''
            or type(root[1]) is not list or type(root[2]) is not list):
        raise ValueError('recorded root configuration differs')
    fields = {}
    for entry in root[2]:
        if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str or entry[0] in fields:
            raise ValueError('duplicate or invalid recorded configuration field')
        fields[entry[0]] = entry[1]
    wanted = {'use_cache': ['bool', False], '_attn_implementation': ['str', 'eager'],
              'n_layer': ['int', 6], 'n_embd': ['int', width], 'vocab_size': ['int', vocabulary]}
    if any(name not in fields or not science.same_recorded_value(fields[name], value)
           for name, value in wanted.items()):
        raise ValueError('recorded model configuration differs')


def validate_budget(budget, width, vocabulary):
    keys = {'successful_logical_bytes_upper_bound', 'failed_return_bytes_reserve',
            'invalid_candidate_bytes_reserve', 'retained_tensor_bytes_limit',
            'per_forward_bytes_limit', 'event_log_bytes_limit'}
    success = 486 * vocabulary * 4 + 378 * (2 * 12 * width * 4 + 2 * width * 4)
    failure = 4 * (12 * vocabulary + 2 * 12 * width + 2 * width)
    candidate = 2 * width * 4
    if (type(budget) is not dict or set(budget) != keys or any(type(x) is not int for x in budget.values())
            or budget['successful_logical_bytes_upper_bound'] != success
            or budget['failed_return_bytes_reserve'] != failure
            or budget['invalid_candidate_bytes_reserve'] != candidate
            or not success + failure + candidate <= budget['retained_tensor_bytes_limit'] <= 512 * 1024**2
            or not failure <= budget['per_forward_bytes_limit'] <= 4 * 1024**2
            or not 32 * 1024**2 <= budget['event_log_bytes_limit'] <= 64 * 1024**2):
        raise ValueError('recorded evidence reservation differs')


def audit_records(records, cases, tensor_dir, directions, science, *, expected_binding,
                  width=768, vocabulary=50257, allow_synthetic=False):
    """Independently decode every saved logit and verify all hook witnesses."""
    if (type(allow_synthetic) is not bool or type(width) is not int or type(vocabulary) is not int
            or not 1 <= width <= 768 or not 548 <= vocabulary <= 50257
            or (not allow_synthetic and (width != 768 or vocabulary != 50257))):
        raise ValueError('unsupported audit dimensions')
    validate_directions(directions, width)
    validate_cases(cases, vocabulary)
    schedule = expected_schedule(cases)
    if type(records) not in (tuple, list) or len(records) != 487:
        raise ValueError('complete run contract and 486 forward records required')
    for index, record in enumerate(records):
        if (type(record) is not dict or set(record) != {'index', 'event', 'tensors'}
                or type(record['index']) is not int or record['index'] != index
                or type(record['event']) is not dict or type(record['tensors']) is not dict):
            raise ValueError('saved record index or fields differ')
    contract = records[0]['event']
    contract_fields = {'kind', 'expected_forwards', 'environment', 'seed', 'execution_configuration',
        'model_state_sha256', 'external_binding', 'schedule_sha256', 'arm_placement_counts', 'evidence_budget'}
    schedule_raw = json.dumps(schedule, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=True, allow_nan=False).encode('ascii')
    placement = {axis + '_' + sign: [sum(x['action'] == axis + '_' + sign and x['slot'] == slot
                                       for x in schedule) for slot in range(6)]
                 for axis in AXES for sign in ('plus', 'minus')}
    if (set(contract) != contract_fields or records[0]['tensors']
            or contract['kind'] != 'run_contract' or type(contract['expected_forwards']) is not int
            or contract['expected_forwards'] != 486 or type(contract['seed']) is not int
            or contract['seed'] != 20261030
            or not science.same_recorded_value(contract['external_binding'], expected_binding)
            or contract['schedule_sha256'] != hashlib.sha256(schedule_raw).hexdigest()
            or not science.same_recorded_value(contract['arm_placement_counts'], placement)
            or type(contract['model_state_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', contract['model_state_sha256']) is None):
        raise ValueError('saved run contract differs')
    validate_budget(contract['evidence_budget'], width, vocabulary)
    if not allow_synthetic:
        validate_runtime(contract, science, width, vocabulary)
    by_id = {case['case_id']: case for case in cases}
    seen, all_margins, details = {}, {}, {}
    def payload(reference, shape):
        raw = science.read_saved_tensor(tensor_dir, reference, shape)
        digest = reference['sha256']
        if digest in seen and seen[digest] != len(raw):
            raise ValueError('repeated saved tensor has inconsistent size')
        seen[digest] = len(raw)
        return raw
    def margin(values):
        return {'present': values[318] - values[389], 'past': values[373] - values[547],
                'the_vs_a': values[262] - values[257], 'in_vs_on': values[287] - values[319]}
    for begin in range(0, 486, 9):
        case_id = schedule[begin]['case_id']
        case = by_id[case_id]
        length = len(case['token_ids'])
        clean_raw = clean_values = source = None
        margins, vocab, deliveries = {}, {}, {}
        for offset in range(9):
            expected, record = schedule[begin + offset], records[begin + offset + 1]
            event, refs, action = record['event'], record['tensors'], expected['action']
            common = {'kind', 'input_ids', 'attention_mask', 'plan_metadata',
                      'all_sequence_logits_finite', 'dose'} | set(expected)
            hooked = action not in ('clean', 'clean_after')
            if (set(event) != (common | {'site', 'donor'} if hooked else common)
                    or event.get('kind') != 'forward'
                    or any(not science.same_recorded_value(event.get(key), value)
                           for key, value in expected.items())
                    or not science.same_recorded_value(event.get('input_ids'), case['token_ids'])
                    or not science.same_recorded_value(event.get('attention_mask'), [1] * length)
                    or event.get('all_sequence_logits_finite') is not True
                    or set(refs) != ({'last_logits', 'before', 'after', 'whole_before', 'whole_after'}
                                     if hooked else {'last_logits'})):
                raise ValueError('forward identity, input or tensor inventory differs')
            logits_raw = payload(refs['last_logits'], (vocabulary,))
            logits = science.decode_float32(logits_raw, expected_count=vocabulary)
            if action == 'clean':
                clean_raw, clean_values = logits_raw, logits
            elif clean_raw is None:
                raise ValueError('case lacks a preceding clean baseline')
            if action in ('noop_l3_final', 'clean_after') and logits_raw != clean_raw:
                raise ValueError('full clean, no-op or repeat logits differ')
            margins[action] = margin(logits)
            vocab[action] = science.vocabulary_diagnostics(clean_values, logits, TARGETS)
            if not hooked:
                if event['dose'] is not None or event['plan_metadata'] != {'steps': []}:
                    raise ValueError('unhooked action has a plan or dose')
                continue
            site, plan = science.expected_hook_plan(action, sequence_length=length, width=width)
            if (not science.same_recorded_value(event['site'], site)
                    or not science.same_recorded_value(event['plan_metadata'], plan)
                    or event['donor'] is not None):
                raise ValueError('hook site, plan or donor differs')
            before, after = payload(refs['before'], (width,)), payload(refs['after'], (width,))
            whole_before = payload(refs['whole_before'], (1, length, width))
            whole_after = payload(refs['whole_after'], (1, length, width))
            science.verify_hook_witness(before, after, whole_before, whole_after,
                                        width=width, sequence_length=length, token_index=length - 1)
            if action == 'noop_l3_final':
                if before != after or event['dose'] is not None:
                    raise ValueError('no-op capture differs')
                source = before
                continue
            axis, sign = action.rsplit('_', 1)
            if source is None or before != source:
                raise ValueError('arm source differs from no-op capture')
            amount = 2.0 if sign == 'plus' else -2.0
            landed, dose = science.direction_delivery(source, directions[axis], amount)
            original = science.decode_float32(source, expected_count=width)
            delivered = science.decode_float32(landed, expected_count=width)
            projection = math.fsum((a - b) * d for a, b, d in zip(delivered, original, directions[axis]))
            relative = abs(abs(projection) - 2.0) / 2.0
            dose['signed_projected_displacement'] = projection
            dose['projected_relative_error'] = relative
            dose['dose_valid'] = dose['dose_valid'] and projection * amount > 0.0 and relative <= 1e-5
            if after != landed or dose['dose_valid'] is not True or not science.same_recorded_value(event['dose'], dose):
                raise ValueError('delivered bits or realized dose differ')
            deliveries[action] = dose
        all_margins[case_id] = margins
        details[case_id] = {'vocabulary': vocab, 'deliveries': deliveries}
    report = score_cases(cases, all_margins)
    for row in report['cases']:
        row.update(details[row['case_id']])
    # Keep the subjectless list self-contained even after JSON serialization.
    report['fragments'] = [row for row in report['cases'] if row['role'] == 'context']
    report.update({'schedule': {'sha256': hashlib.sha256(schedule_raw).hexdigest(),
                                'arm_placement_counts': placement, 'actions': schedule},
                   'source_tensor_files': len(seen), 'source_tensor_bytes': sum(seen.values()),
                   'dimensions': {'width': width, 'vocabulary': vocabulary},
                   'scope': ('Saved-byte numerical consistency only; recorded full-sequence finiteness, '
                             'process exit and exposure are separately established.')})
    return report, seen


def ordinary_directory(path):
    try:
        info = Path(path).lstat()
        return (stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
                and not (getattr(info, 'st_file_attributes', 0)
                         & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)))
    except OSError:
        return False


def audit_attempt(directory, cases, directions, science, *, expected_binding, expected_source_revision,
                  expected_events_sha256, expected_scope, width=768, vocabulary=50257, allow_synthetic=False):
    """Replay a closed candidate with independently supplied owner/journal pins."""
    names = {'prelaunch_sha256', 'reservation_sha256', 'latest_exposure_sha256'}
    if (type(expected_binding) is not dict or set(expected_binding) != names
            or any(type(x) is not str or re.fullmatch(r'[0-9a-f]{64}', x) is None
                   for x in expected_binding.values())
            or type(expected_source_revision) is not str
            or re.fullmatch(r'[0-9a-f]{40}', expected_source_revision) is None
            or type(expected_events_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected_events_sha256) is None
            or type(expected_scope) is not str or not expected_scope or len(expected_scope) > 256
            or (not allow_synthetic and expected_scope != 'public prospective agreement noun-substitution')):
        raise ValueError('independent binding, source and event identities are required')
    root = Path(directory)
    if (not ordinary_directory(root)
            or {item.name for item in root.iterdir()} != {'started.json', 'events.jsonl', 'terminal.json', 'tensors'}
            or not ordinary_directory(root / 'tensors')):
        raise ValueError('attempt directory is incomplete or ambiguous')
    start_raw = bounded_file(root / 'started.json', 256 * 1024)
    events_raw = bounded_file(root / 'events.jsonl', 64 * 1024**2)
    terminal_raw = bounded_file(root / 'terminal.json', 256 * 1024)
    start, terminal = strict_json(start_raw), strict_json(terminal_raw)
    expected_start = {'scope': expected_scope, 'external_binding': expected_binding,
                      'source_revision': expected_source_revision}
    if (not science.same_recorded_value(start, expected_start)
            or type(terminal) is not dict or terminal.get('status') != 'complete_candidate'
            or not events_raw.endswith(b'\n')
            or hashlib.sha256(events_raw).hexdigest() != expected_events_sha256):
        raise ValueError('start, terminal or independently pinned event log differs')
    lines = events_raw.splitlines()
    if len(lines) != 487 or any(not line or len(line) > 256 * 1024 for line in lines):
        raise ValueError('event log does not contain the complete schedule')
    records = [strict_json(line) for line in lines]
    report, seen = audit_records(records, cases, root / 'tensors', directions, science,
        expected_binding=expected_binding, width=width, vocabulary=vocabulary, allow_synthetic=allow_synthetic)
    if {item.name for item in (root / 'tensors').iterdir()} != {digest + '.f32le' for digest in seen}:
        raise ValueError('tensor inventory contains a missing or unreferenced file')
    contract = records[0]['event']
    tensor_bytes = sum(seen.values())
    wanted_terminal = {'status': 'complete_candidate', 'events': 487,
        'event_log_bytes': len(events_raw), 'retained_tensor_bytes': tensor_bytes,
        'storage_accounting_complete': True, 'detail': {'status': 'complete_candidate', 'forwards': 486,
            'model_state_sha256_before_after': contract['model_state_sha256'],
            'retained_tensor_bytes': tensor_bytes, 'schedule_sha256': contract['schedule_sha256']}}
    if not science.same_recorded_value(terminal, wanted_terminal):
        raise ValueError('terminal or retained tensor accounting differs')
    report['journal'] = {'status': 'consistent_candidate', 'events': 487,
        'started_sha256': hashlib.sha256(start_raw).hexdigest(), 'events_sha256': expected_events_sha256,
        'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest()}
    return report


def pinned(path, expected, maximum):
    raw = bounded_file(path, maximum)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('published input differs from its fixed digest')
    return raw


def load_frozen_inputs(case_path, direction_directory):
    document = strict_json(pinned(case_path, CASE_SHA, 2 * 1024 * 1024))
    if (type(document) is not dict
            or document.get('schema') != 'distilgpt2-agreement-noun-substitution-cases-v1'
            or document.get('protocol_sha256') != PROTOCOL_SHA
            or document.get('model_id') != 'distilbert/distilgpt2'
            or document.get('model_revision') != '2290a62682d06624634c1f46a6ad5be0f47f38aa'
            or document.get('tokenizer_sha256') != '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'):
        raise ValueError('published case/protocol/model binding differs')
    cases = document.get('cases')
    validate_cases(cases, 50257)
    directions = {}
    for name, digest in DIRECTION_SHA.items():
        raw = pinned(Path(direction_directory) / (name + '.f64le'), digest, 768 * 8)
        if len(raw) != 768 * 8:
            raise ValueError('published direction width differs')
        directions[name] = struct.unpack('<768d', raw)
    validate_directions(directions, 768)
    return cases, directions


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'cases', 'directions', 'numeric-source', 'output', 'source-revision',
                 'prelaunch-sha256', 'reservation-sha256', 'latest-exposure-sha256',
                 'events-sha256', 'start-scope'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output, attempt = Path(args.output), Path(args.attempt)
    if (not output.is_absolute() or not output.parent.is_dir() or output.exists()
            or output == attempt or attempt in output.parents):
        raise ValueError('fresh report path outside the attempt is required')
    science = load_numeric(args.numeric_source)
    cases, directions = load_frozen_inputs(args.cases, args.directions)
    binding = {'prelaunch_sha256': args.prelaunch_sha256, 'reservation_sha256': args.reservation_sha256,
               'latest_exposure_sha256': args.latest_exposure_sha256}
    result = audit_attempt(attempt, cases, directions, science, expected_binding=binding,
        expected_source_revision=args.source_revision, expected_events_sha256=args.events_sha256,
        expected_scope=args.start_scope)
    result['public_inputs'] = {'case_sha256': CASE_SHA, 'protocol_sha256': PROTOCOL_SHA,
        'direction_sha256': dict(DIRECTION_SHA), 'numeric_source_sha256': NUMERIC_SHA,
        'owner_source_revision': args.source_revision, 'external_binding': binding}
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')) + '\n').encode('ascii')
    if len(raw) > 64 * 1024**2:
        raise ValueError('derived report exceeds its bound')
    with output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('derived report write was incomplete')
        stream.flush()
        os.fsync(stream.fileno())
    if bounded_file(output, 64 * 1024**2) != raw:
        raise ValueError('derived report changed after write')
    print(json.dumps({'status': 'saved_byte_audit', 'sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
