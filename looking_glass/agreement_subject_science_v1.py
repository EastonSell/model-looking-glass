"""Independent, model-free agreement-subject saved-evidence reader."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import ModuleType


CONDITIONS = ('none', 'mass', 'attractor_singular', 'attractor_plural')
FRAME_PAIRS = (('R1', 'I1'), ('R1', 'I2'), ('R2', 'I1'), ('R2', 'I2'))
FRAGMENT_OFFSETS = ((0, 0, 2, 2), (2, 2, 4, 4), (2, 2, 4, 4), (4, 4, 4, 4))
_CASE_SHA256 = '5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e'
_PROTOCOL_SHA256 = 'bd39b4556023110d0de88fb21381a0a3ce1e17cad862cdd0259b67849f540714'
_REFERENCE_SHA256 = 'bdb31d5b74b6cff17ed2a683415beab22e36b101220257d774909be6f9f804bd'
_SCIENCE_SHA256 = '917694d448b796359db0002a630877e5226a2583d273092871892fade7a38ac4'
_DIRECTION_SHAS = {'u': 'd1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
    'v1': '4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
    'v2': '4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}


def _load_science():
    """Compile the pinned helper bytes directly, without consulting bytecode."""
    path = Path(__file__).with_name('agreement_science_v1.py')
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != _SCIENCE_SHA256:
        raise ValueError('the independent numerical source differs')
    module = ModuleType('_verified_agreement_subject_numeric')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module


science = _load_science()


def expected_schedule(cases):
    """Independently derive the protocol's fixed case and arm ordering."""
    if type(cases) not in (tuple, list) or len(cases) != 80:
        raise ValueError('the fixed eighty cases are required')
    seen, schedule = set(), []
    for frame in range(4):
        for condition, label in enumerate(CONDITIONS):
            start = 20 * frame + 5 * condition
            block = cases[start:start + 5]
            for index, row in enumerate(block):
                if type(row) is not dict:
                    raise ValueError('case is invalid')
                name = row.get('case_id')
                if (type(name) is not str or not name or len(name) > 128 or name in seen
                        or type(row.get('frame_index')) is not int or row['frame_index'] != frame
                        or type(row.get('condition_index')) is not int or row['condition_index'] != condition
                        or row.get('condition') != label):
                    raise ValueError('canonical case order differs')
                seen.add(name)
                pair = index // 2 if index < 4 else None
                expected = ('context', None, None, None, None) if pair is None else (
                    'target', pair, FRAME_PAIRS[frame][pair],
                    ('singular', 'plural')[index % 2], ('regular', 'irregular')[pair])
                actual = tuple(row.get(key) for key in
                               ('role', 'pair_index', 'subject_key', 'subject_number', 'morphology'))
                if actual != expected or (pair is not None and type(row.get('pair_index')) is not int):
                    raise ValueError('canonical pair or role differs')
            ordered = []
            for pair in range(2):
                first = 1 if (frame + frame // 2 + pair) % 2 else 0
                ordered.extend((start + 2 * pair + number, pair) for number in (first, 1 - first))
            ordered.append((start + 4, None))
            for case_index, pair in ordered:
                row = cases[case_index]
                plus_first = (frame + condition + (pair or 0)) % 2 == 0
                signs = ('plus', 'minus') if plus_first else ('minus', 'plus')
                arms = tuple(name + '_' + sign for name in ('u', 'v1', 'v2') for sign in signs)
                offset = FRAGMENT_OFFSETS[frame][condition] if pair is None else 2 * ((4 * frame + condition) % 3)
                arms = arms[offset:] + arms[:offset]
                for action, slot in [('clean', None), ('noop_l3_final', None),
                                     *((arm, i) for i, arm in enumerate(arms)), ('clean_after', None)]:
                    schedule.append({'index': len(schedule), 'case_id': row['case_id'],
                        'case_index': case_index, 'frame_index': frame, 'condition': label,
                        'condition_index': condition, 'role': row['role'], 'pair_index': pair,
                        'subject_key': row['subject_key'], 'subject_number': row['subject_number'],
                        'morphology': row['morphology'], 'action': action, 'slot': slot})
    return tuple(schedule)


def validate_cases(cases, *, vocabulary):
    """Check fixed input identity, pair alignment and full-row disjointness."""
    expected_schedule(cases)
    if type(vocabulary) is not int or vocabulary < 548:
        raise ValueError('invalid vocabulary bound')
    seen = []
    for row in cases:
        tokens = row.get('token_ids')
        if (type(tokens) is not list or not 1 <= len(tokens) <= 12
                or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
                or type(row.get('final_index')) is not int or row['final_index'] != len(tokens) - 1
                or any(tokens[:len(other)] == other or other[:len(tokens)] == tokens for other in seen)):
            raise ValueError('invalid or prefix-overlapping case input')
        if row['role'] == 'target':
            if type(row.get('subject_index')) is not int or not 0 < row['subject_index'] < len(tokens) - 1:
                raise ValueError('invalid subject position')
        elif row.get('subject_index') is not None:
            raise ValueError('fragment unexpectedly has a subject position')
        seen.append(tokens)
    for frame in range(4):
        blocks = [cases[20 * frame + 5 * condition:20 * frame + 5 * condition + 5]
                  for condition in range(4)]
        for block in blocks:
            for pair in range(2):
                singular, plural = block[2 * pair:2 * pair + 2]
                a, b = singular['token_ids'], plural['token_ids']
                index = singular['subject_index']
                if (plural['subject_index'] != index or len(a) != len(b) or a[index] == b[index]
                        or a[:index] + a[index + 1:] != b[:index] + b[index + 1:]):
                    raise ValueError('paired inputs differ outside their subject')
        for position in range(4):
            rows = [blocks[index][position] for index in (1, 2, 3)]
            if (len({row['subject_index'] for row in rows}) != 1
                    or len({tuple(row['token_ids'][:-1]) for row in rows}) != 1
                    or len({row['token_ids'][-1] for row in rows}) != 3):
                raise ValueError('mass and count inputs differ outside their last noun')
        a, b = blocks[2][4]['token_ids'], blocks[3][4]['token_ids']
        if a[:-1] != b[:-1] or a[-1] == b[-1]:
            raise ValueError('count fragments differ outside their last noun')
    return True


def _mean(values):
    values = list(values)
    if not values or any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError('a nonempty finite sample is required')
    return math.fsum(values) / len(values)


def centered_cosine(values, reference):
    """Compare centered binary64 vectors and retain undefined zero norms."""
    if (type(values) not in (list, tuple) or type(reference) not in (list, tuple)
            or not values or len(values) != len(reference)
            or any(type(value) is not float or not math.isfinite(value)
                   for value in (*values, *reference))):
        raise ValueError('finite vectors of the same size are required')
    a_mean, b_mean = _mean(values), _mean(reference)
    a = [value - a_mean for value in values]
    b = [value - b_mean for value in reference]
    a_norm = math.sqrt(math.fsum(value * value for value in a))
    b_norm = math.sqrt(math.fsum(value * value for value in b))
    cosine = None if a_norm == 0.0 or b_norm == 0.0 else (
        math.fsum(x * y for x, y in zip(a, b)) / (a_norm * b_norm))
    return {'cosine': cosine, 'observed_centered_norm': a_norm, 'reference_centered_norm': b_norm}


def score_cases(cases, measurements, directions, *, reference_odd):
    """Compute all fixed-pair and aggregate criteria from audited case data."""
    expected_schedule(cases)
    if (type(measurements) is not dict or set(measurements) != {row['case_id'] for row in cases}
            or type(directions) is not dict or set(directions) != {'u', 'v1', 'v2'}
            or type(reference_odd) is not float or not math.isfinite(reference_odd)):
        raise ValueError('complete measurements and fixed directions are required')
    width = len(directions['u'])
    if not 1 <= width <= 768:
        raise ValueError('invalid direction width')
    for direction in directions.values():
        if (type(direction) not in (tuple, list) or len(direction) != width
                or any(type(value) is not float or not math.isfinite(value) for value in direction)
                or abs(math.sqrt(math.fsum(value * value for value in direction)) - 1.0) > 1e-12):
            raise ValueError('directions must be finite unit vectors')
    for left, right in (('u', 'v1'), ('u', 'v2'), ('v1', 'v2')):
        if abs(math.fsum(a * b for a, b in zip(directions[left], directions[right]))) > 1e-12:
            raise ValueError('control directions are not orthogonal')
    per_case = []
    by_id = {}
    for row in cases:
        observed = measurements[row['case_id']]
        if (type(observed) is not dict or type(observed.get('margins')) is not dict
                or type(observed.get('capture')) not in (tuple, list)
                or len(observed['capture']) != width
                or any(type(value) is not float or not math.isfinite(value) for value in observed['capture'])):
            raise ValueError('invalid case measurement')
        margins = observed['margins']
        if (type(margins.get('clean')) is not dict or 'present' not in margins['clean']
                or not margins['clean']):
            raise ValueError('the clean margin is required')
        keys = set(margins['clean'])
        if any(type(value) is not float or not math.isfinite(value) for value in margins['clean'].values()):
            raise ValueError('invalid clean margins')
        responses = {}
        for name in ('u', 'v1', 'v2'):
            plus, minus = margins.get(name + '_plus'), margins.get(name + '_minus')
            if (type(plus) is not dict or type(minus) is not dict or set(plus) != keys or set(minus) != keys
                    or any(type(value) is not float or not math.isfinite(value)
                           for value in (*plus.values(), *minus.values()))):
                raise ValueError('a signed margin is missing or nonfinite')
            responses[name] = {key: {'odd': (plus[key] - minus[key]) / 2.0,
                                     'even': (plus[key] + minus[key]) / 2.0 - margins['clean'][key],
                                     'plus_change': plus[key] - margins['clean'][key],
                                     'minus_change': minus[key] - margins['clean'][key]}
                               for key in sorted(keys)}
        result = {'case_id': row['case_id'], 'role': row['role'], 'frame_index': row['frame_index'],
                  'condition': row['condition'], 'subject_key': row['subject_key'],
                  'subject_number': row['subject_number'], 'morphology': row['morphology'],
                  'margins': margins, 'responses': responses,
                  'reference_odd_error': responses['u']['present']['odd'] - reference_odd}
        result['clean_correct'] = (None if row['role'] == 'context' else
            margins['clean']['present'] > 0.0 if row['subject_number'] == 'singular'
            else margins['clean']['present'] < 0.0)
        per_case.append(result)
        by_id[row['case_id']] = result
    pairs, contrasts = [], []
    for frame in range(4):
        for pair_index, subject in enumerate(FRAME_PAIRS[frame]):
            group = []
            for condition, label in enumerate(CONDITIONS):
                start = 20 * frame + 5 * condition + 2 * pair_index
                left, right = cases[start:start + 2]
                a, b = by_id[left['case_id']], by_id[right['case_id']]
                gap = {name: a['responses'][name]['present']['odd'] - b['responses'][name]['present']['odd']
                       for name in ('u', 'v1', 'v2')}
                av, bv = measurements[left['case_id']]['capture'], measurements[right['case_id']]['capture']
                geometry = {name: math.fsum(direction[i] * (av[i] - bv[i]) for i in range(width))
                            for name, direction in directions.items()}
                entry = {'frame_index': frame, 'condition': label, 'subject_key': subject,
                         'morphology': ('regular', 'irregular')[pair_index],
                         'singular_case': left['case_id'], 'plural_case': right['case_id'],
                         'clean_pair_correct': (a['margins']['clean']['present'] > 0.0
                                                and b['margins']['clean']['present'] < 0.0),
                         'k': gap, 'g': geometry}
                pairs.append(entry)
                group.append(entry)
            differences = {}
            for metric in ('k', 'g'):
                differences[metric] = {name: {
                    'attractor_singular_minus_none': group[2][metric][name] - group[0][metric][name],
                    'attractor_plural_minus_none': group[3][metric][name] - group[0][metric][name],
                    'attractor_plural_minus_singular': group[3][metric][name] - group[2][metric][name],
                    'mass_minus_none': group[1][metric][name] - group[0][metric][name]}
                    for name in ('u', 'v1', 'v2')}
            contrasts.append({'frame_index': frame, 'subject_key': subject, **differences})
    groups = {'frame': {str(index): [row for row in pairs if row['frame_index'] == index] for index in range(4)},
              'condition': {name: [row for row in pairs if row['condition'] == name] for name in CONDITIONS},
              'subject': {name: [row for row in pairs if row['subject_key'] == name]
                          for name in ('R1', 'R2', 'I1', 'I2')},
              'morphology': {name: [row for row in pairs if row['morphology'] == name]
                             for name in ('regular', 'irregular')}}
    accuracy = {category: {name: sum(row['clean_pair_correct'] for row in values)
                           for name, values in mapping.items()} for category, mapping in groups.items()}
    total_correct = sum(row['clean_pair_correct'] for row in pairs)
    adequate = total_correct >= 24 and all(value >= 5 for category in ('frame', 'condition', 'subject')
                                           for value in accuracy[category].values())
    target_cases = [row for row in per_case if row['role'] == 'target']
    def individual(field, values):
        return {str(value): {'correct': sum(row['clean_correct'] is True for row in target_cases
                                              if row[field] == value),
                             'total': sum(row[field] == value for row in target_cases)} for value in values}
    individual_number = individual('subject_number', ('singular', 'plural'))
    individual_groups = {'frame': individual('frame_index', range(4)),
                         'condition': individual('condition', CONDITIONS),
                         'subject': individual('subject_key', ('R1', 'R2', 'I1', 'I2')),
                         'morphology': individual('morphology', ('regular', 'irregular'))}
    abs_k = [abs(row['k']['u']) for row in pairs]
    constant = {'mean_abs_k': _mean(abs_k), 'within_0_10': sum(value <= 0.10 for value in abs_k)}
    constant['passes'] = constant['mean_abs_k'] <= 0.10 and constant['within_0_10'] >= 24
    errors = [abs(row['reference_odd_error']) for row in per_case]
    reference = {'mean_abs_error': _mean(errors), 'within_0_20': sum(value <= 0.20 for value in errors)}
    reference['passes'] = reference['mean_abs_error'] <= 0.20 and reference['within_0_20'] >= 60
    def reference_group(members):
        values = [abs(row['reference_odd_error']) for row in members]
        return {'mean_abs_error': _mean(values), 'within_0_20': sum(value <= 0.20 for value in values),
                'total': len(values)}
    reference['by_role'] = {role: reference_group(row for row in per_case if row['role'] == role)
                            for role in ('target', 'context')}
    reference['by_condition'] = {name: reference_group(row for row in per_case if row['condition'] == name)
                                 for name in CONDITIONS}
    def criterion(metric, floor, frame_floor):
        main = _mean(row[metric]['u'] for row in pairs)
        controls = {name: _mean(abs(row[metric][name]) for row in pairs) for name in ('v1', 'v2')}
        frames = []
        for frame in range(4):
            members = groups['frame'][str(frame)]
            value = _mean(row[metric]['u'] for row in members)
            control = {name: _mean(abs(row[metric][name]) for row in members) for name in ('v1', 'v2')}
            frames.append({'frame_index': frame, 'mean': value, 'mean_abs_controls': control,
                           'passes': value >= frame_floor and all(value > x for x in control.values())})
        return {'mean': main, 'mean_abs_controls': controls, 'frames': frames,
                'passes': main >= floor and all(main > x for x in controls.values())
                          and sum(row['passes'] for row in frames) >= 3}
    sensitivity = criterion('k', 0.10, 0.05)
    sensitivity['mean_k_u'] = sensitivity.pop('mean')
    sensitivity['interpretation'] = 'baseline_adequate' if adequate else 'baseline_limited'
    sensitivity['positive_zero_negative'] = [sum(row['k']['u'] > 0 for row in pairs),
        sum(row['k']['u'] == 0 for row in pairs), sum(row['k']['u'] < 0 for row in pairs)]
    geometry = criterion('g', 0.5, 0.5)
    geometry['mean_g_u'] = geometry.pop('mean')
    def aggregate(members, rows):
        by_role = {}
        for label, selected in (('singular', [r for r in rows if r['subject_number'] == 'singular']),
                                ('plural', [r for r in rows if r['subject_number'] == 'plural']),
                                ('context', [r for r in rows if r['role'] == 'context'])):
            if selected:
                by_role[label] = {'count': len(selected),
                    'mean_odd': {name: _mean(r['responses'][name]['present']['odd'] for r in selected)
                                 for name in ('u', 'v1', 'v2')},
                    'mean_even': {name: _mean(r['responses'][name]['present']['even'] for r in selected)
                                  for name in ('u', 'v1', 'v2')}}
        return {'pair_count': len(members), 'case_counts': {
                    'target': sum(row['role'] == 'target' for row in rows),
                    'context': sum(row['role'] == 'context' for row in rows)},
                'correct_pairs': sum(row['clean_pair_correct'] for row in members),
                'mean_k': {name: _mean(row['k'][name] for row in members) for name in ('u', 'v1', 'v2')},
                'mean_abs_k': {name: _mean(abs(row['k'][name]) for row in members) for name in ('u', 'v1', 'v2')},
                'mean_g': {name: _mean(row['g'][name] for row in members) for name in ('u', 'v1', 'v2')},
                'mean_abs_g': {name: _mean(abs(row['g'][name]) for row in members) for name in ('u', 'v1', 'v2')},
                'case_responses': by_role}
    condition_aggregates = {label: aggregate(groups['condition'][label],
                              [row for row in per_case if row['condition'] == label]) for label in CONDITIONS}
    frame_subject_aggregates = []
    for frame in range(4):
        for subject in FRAME_PAIRS[frame]:
            members = [row for row in pairs if row['frame_index'] == frame and row['subject_key'] == subject]
            rows = [row for row in per_case if row['frame_index'] == frame and row['subject_key'] == subject]
            frame_subject_aggregates.append({'frame_index': frame, 'subject_key': subject, **aggregate(members, rows)})
    return {'cases': per_case, 'pairs': pairs, 'contrasts': contrasts,
        'fragments': [row for row in per_case if row['role'] == 'context'],
        'baseline': {'correct_pairs': total_correct, 'adequate': adequate, 'by_group': accuracy,
                     'individual_by_number': individual_number, 'individual_by_group': individual_groups},
        'constant_additive': constant, 'specific_reference': reference,
        'sensitivity': sensitivity, 'geometry': geometry,
        'condition_aggregates': condition_aggregates, 'frame_subject_aggregates': frame_subject_aggregates}


def audit_records(records, cases, tensor_dir, directions, reference, *, expected_binding,
                  width=768, vocabulary=50257, allow_synthetic=False):
    """Replay every saved action; this does not attest a process or exposure."""
    if (type(allow_synthetic) is not bool or type(width) is not int or type(vocabulary) is not int
            or not 1 <= width <= 768 or not 548 <= vocabulary <= 50257
            or (not allow_synthetic and (width != 768 or vocabulary != 50257))
            or type(reference) not in (tuple, list) or len(reference) != vocabulary
            or any(type(value) is not float or not math.isfinite(value) for value in reference)):
        raise ValueError('invalid audit dimensions or reference')
    schedule = expected_schedule(cases)
    validate_cases(cases, vocabulary=vocabulary)
    if type(records) not in (list, tuple) or len(records) != 721:
        raise ValueError('the complete contract and 720 forwards are required')
    for index, record in enumerate(records):
        if (type(record) is not dict or set(record) != {'index', 'event', 'tensors'}
                or type(record['index']) is not int or record['index'] != index
                or type(record['event']) is not dict or type(record['tensors']) is not dict):
            raise ValueError('journal record identity differs')
    contract = records[0]['event']
    contract_fields = {'kind', 'expected_forwards', 'environment', 'seed', 'execution_configuration',
        'model_state_sha256', 'external_binding', 'schedule_sha256', 'arm_placement_counts', 'evidence_budget'}
    schedule_raw = json.dumps(schedule, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=True, allow_nan=False).encode('ascii')
    placement = {name: [sum(item['action'] == name and item['slot'] == slot for item in schedule)
                        for slot in range(6)] for name in
                 ('u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')}
    if (set(contract) != contract_fields or records[0]['tensors']
            or contract['kind'] != 'run_contract' or type(contract['expected_forwards']) is not int
            or contract['expected_forwards'] != 720 or type(contract['seed']) is not int
            or contract['seed'] != 20261030
            or not science.same_recorded_value(contract['external_binding'], expected_binding)
            or contract['schedule_sha256'] != hashlib.sha256(schedule_raw).hexdigest()
            or not science.same_recorded_value(contract['arm_placement_counts'], placement)
            or type(contract['model_state_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', contract['model_state_sha256']) is None):
        raise ValueError('the saved run contract differs')
    if not allow_synthetic:
        _validate_runtime(contract)
    by_id = {row['case_id']: row for row in cases}
    seen, measurements, details = {}, {}, {}
    targets = (318, 389, 373, 547, 262, 257, 287, 319)
    theoretical_odd = 2.0 * (reference[318] - reference[389])

    def payload(reference_item, shape):
        raw = science.read_saved_tensor(tensor_dir, reference_item, shape)
        digest = reference_item['sha256']
        if digest in seen and seen[digest] != len(raw):
            raise ValueError('repeated tensor identity differs')
        seen[digest] = len(raw)
        return raw

    def margins(values):
        return {'present': values[318] - values[389], 'past': values[373] - values[547],
                'the_vs_a': values[262] - values[257], 'in_vs_on': values[287] - values[319]}

    for begin in range(0, 720, 9):
        case_id = schedule[begin]['case_id']
        row = by_id[case_id]
        length = len(row['token_ids'])
        if (type(row['token_ids']) is not list or not 1 <= length <= 12
                or any(type(token) is not int or not 0 <= token < vocabulary for token in row['token_ids'])
                or type(row.get('final_index')) is not int or row['final_index'] != length - 1):
            raise ValueError('case input or final position differs')
        clean_raw = clean_values = base = None
        case_margins, logits_by_action, vocab_reports, deliveries = {}, {}, {}, {}
        for offset in range(9):
            expected, record = schedule[begin + offset], records[begin + offset + 1]
            event, refs, arm = record['event'], record['tensors'], expected['action']
            common = {'kind', 'input_ids', 'attention_mask', 'plan_metadata',
                      'all_sequence_logits_finite', 'dose'} | set(expected)
            hooked = arm not in ('clean', 'clean_after')
            if (set(event) != (common | {'site', 'donor'} if hooked else common)
                    or event.get('kind') != 'forward'
                    or any(not science.same_recorded_value(event.get(key), value) for key, value in expected.items())
                    or not science.same_recorded_value(event.get('input_ids'), row['token_ids'])
                    or not science.same_recorded_value(event.get('attention_mask'), [1] * length)
                    or event.get('all_sequence_logits_finite') is not True
                    or set(refs) != ({'last_logits', 'before', 'after', 'whole_before', 'whole_after'}
                                     if hooked else {'last_logits'})):
                raise ValueError('forward order, input or evidence inventory differs')
            logits_raw = payload(refs['last_logits'], (vocabulary,))
            logits = science.decode_float32(logits_raw, expected_count=vocabulary)
            if arm == 'clean':
                clean_raw, clean_values = logits_raw, logits
            elif clean_raw is None:
                raise ValueError('case has no clean baseline')
            if arm in ('noop_l3_final', 'clean_after') and logits_raw != clean_raw:
                raise ValueError('full clean or no-op logits differ')
            case_margins[arm] = margins(logits)
            if arm in ('u_plus', 'u_minus'):
                logits_by_action[arm] = logits
            vocab_reports[arm] = science.vocabulary_diagnostics(clean_values, logits, targets)
            if not hooked:
                if event['dose'] is not None or event['plan_metadata'] != {'steps': []}:
                    raise ValueError('an unhooked action has a plan or dose')
                continue
            site, plan = science.expected_hook_plan(arm, sequence_length=length, width=width)
            if (not science.same_recorded_value(event['site'], site)
                    or not science.same_recorded_value(event['plan_metadata'], plan)
                    or event['donor'] is not None):
                raise ValueError('hook site, plan or donor differs')
            before, after = payload(refs['before'], (width,)), payload(refs['after'], (width,))
            whole_before = payload(refs['whole_before'], (1, length, width))
            whole_after = payload(refs['whole_after'], (1, length, width))
            science.verify_hook_witness(before, after, whole_before, whole_after,
                                        width=width, sequence_length=length, token_index=length - 1)
            if arm == 'noop_l3_final':
                if before != after or event['dose'] is not None:
                    raise ValueError('the no-op capture differs')
                base = before
                continue
            name, sign = arm.rsplit('_', 1)
            if base is None or before != base:
                raise ValueError('the arm source differs from the no-op capture')
            amount = 2.0 if sign == 'plus' else -2.0
            landed, dose = science.direction_delivery(base, directions[name], amount)
            original = science.decode_float32(base, expected_count=width)
            delivered = science.decode_float32(landed, expected_count=width)
            projection = math.fsum((a - b) * d for a, b, d in zip(delivered, original, directions[name]))
            relative = abs(abs(projection) - 2.0) / 2.0
            dose['signed_projected_displacement'] = projection
            dose['projected_relative_error'] = relative
            dose['dose_valid'] = dose['dose_valid'] and projection * amount > 0.0 and relative <= 1e-5
            if after != landed or dose['dose_valid'] is not True or not science.same_recorded_value(event['dose'], dose):
                raise ValueError('delivered bits or dose differ')
            deliveries[arm] = dose
        capture = science.decode_float32(base, expected_count=width)
        plus = tuple(value + 2.0 * ref for value, ref in zip(clean_values, reference))
        minus = tuple(value - 2.0 * ref for value, ref in zip(clean_values, reference))
        reference_plus, reference_minus = margins(plus), margins(minus)
        realized_reference_odd = (reference_plus['present'] - reference_minus['present']) / 2.0
        slope = tuple((a - b) / 4.0 for a, b in zip(logits_by_action['u_plus'], logits_by_action['u_minus']))
        measurements[case_id] = {'margins': case_margins, 'capture': capture}
        details[case_id] = {'vocabulary': vocab_reports, 'deliveries': deliveries,
            'reference': {'plus_margins': reference_plus, 'minus_margins': reference_minus,
                'odd_present': realized_reference_odd, 'odd_roundoff': realized_reference_odd - theoretical_odd,
                'plus_vocabulary': science.vocabulary_diagnostics(clean_values, plus, targets),
                'minus_vocabulary': science.vocabulary_diagnostics(clean_values, minus, targets)},
            'centered_slope_alignment': centered_cosine(slope, reference)}
    report = score_cases(cases, measurements, directions, reference_odd=theoretical_odd)
    for row in report['cases']:
        row.update(details[row['case_id']])
    for pair in report['pairs']:
        pair['reference_k_residual'] = (details[pair['singular_case']]['reference']['odd_present']
                                        - details[pair['plural_case']]['reference']['odd_present'])
    valid_cosines = sum(row['centered_slope_alignment']['cosine'] is not None for row in report['cases'])
    report.update({'cosine_valid_count': valid_cosines, 'cosine_undefined_count': 80 - valid_cosines,
        'source_tensor_files': len(seen), 'source_tensor_bytes': sum(seen.values()),
        'dimensions': {'width': width, 'vocabulary': vocabulary},
        'scope': ('Saved-byte numerical consistency only; recorded full-sequence finiteness, process exit '
                  'and exposure are not independently established by this reader.')})
    return report, seen


def _validate_runtime(contract):
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
            raise ValueError('recorded configuration contains duplicate or invalid fields')
        fields[entry[0]] = entry[1]
    expected = {'use_cache': ['bool', False], '_attn_implementation': ['str', 'eager'],
                'n_layer': ['int', 6], 'n_embd': ['int', 768], 'vocab_size': ['int', 50257]}
    if any(name not in fields or not science.same_recorded_value(fields[name], value)
           for name, value in expected.items()):
        raise ValueError('recorded model configuration differs')
    budget = contract['evidence_budget']
    success = 720 * 50257 * 4 + 560 * (2 * 12 * 768 * 4 + 2 * 768 * 4)
    failure = 4 * (12 * 50257 + 2 * 12 * 768 + 2 * 768)
    candidate = 2 * 768 * 4
    required = {'successful_logical_bytes_upper_bound', 'failed_return_bytes_reserve',
                'invalid_candidate_bytes_reserve', 'retained_tensor_bytes_limit',
                'per_forward_bytes_limit', 'event_log_bytes_limit'}
    if (type(budget) is not dict or set(budget) != required
            or any(type(value) is not int for value in budget.values())
            or budget['successful_logical_bytes_upper_bound'] != success
            or budget['failed_return_bytes_reserve'] != failure
            or budget['invalid_candidate_bytes_reserve'] != candidate
            or not success + failure + candidate <= budget['retained_tensor_bytes_limit'] <= 512 * 1024**2
            or not failure <= budget['per_forward_bytes_limit'] <= 4 * 1024**2
            or budget['event_log_bytes_limit'] != 64 * 1024**2):
        raise ValueError('recorded evidence reservation differs')


def _bounded_file(path, limit):
    path = Path(path)
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                or info.st_size > limit):
            raise ValueError('evidence file is not a bounded ordinary file')
        with path.open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit:
            raise ValueError('evidence file changed or exceeded its bound')
        return raw
    except OSError as error:
        raise ValueError('evidence file is unavailable') from error


def _json(raw):
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
    def bad_constant(_):
        raise ValueError('nonfinite JSON constant')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_float=number, parse_constant=bad_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError('invalid JSON evidence') from error


def _ordinary_directory(path):
    try:
        item = Path(path).lstat()
        return (stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode)
                and not (getattr(item, 'st_file_attributes', 0)
                         & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)))
    except OSError:
        return False


def audit_attempt(directory, cases, directions, reference, *, expected_binding, expected_source_revision,
                  width=768, vocabulary=50257, allow_synthetic=False):
    """Replay a closed candidate; the caller separately observes process exit."""
    root = Path(directory)
    binding_keys = {'prelaunch_sha256', 'reservation_sha256', 'latest_exposure_sha256'}
    if (type(expected_binding) is not dict or set(expected_binding) != binding_keys
            or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None
                   for value in expected_binding.values())
            or type(expected_source_revision) is not str
            or re.fullmatch(r'[0-9a-f]{40}', expected_source_revision) is None):
        raise ValueError('independent source and binding identities are required')
    if (not _ordinary_directory(root)
            or {item.name for item in root.iterdir()} != {'started.json', 'events.jsonl', 'terminal.json', 'tensors'}
            or not _ordinary_directory(root / 'tensors')):
        raise ValueError('the attempt directory is incomplete or ambiguous')
    started_raw = _bounded_file(root / 'started.json', 256 * 1024)
    events_raw = _bounded_file(root / 'events.jsonl', 64 * 1024**2)
    terminal_raw = _bounded_file(root / 'terminal.json', 256 * 1024)
    started, terminal = _json(started_raw), _json(terminal_raw)
    expected_start = {'scope': 'public prospective agreement subject-dependence',
                      'external_binding': expected_binding, 'source_revision': expected_source_revision}
    if (not science.same_recorded_value(started, expected_start)
            or type(terminal) is not dict or terminal.get('status') != 'complete_candidate'
            or not events_raw.endswith(b'\n')):
        raise ValueError('start, terminal or event log is incomplete')
    lines = events_raw.splitlines()
    if len(lines) != 721 or any(not line or len(line) > 256 * 1024 for line in lines):
        raise ValueError('event log does not contain the complete schedule')
    records = [_json(line) for line in lines]
    report, seen = audit_records(records, cases, root / 'tensors', directions, reference,
        expected_binding=expected_binding, width=width, vocabulary=vocabulary, allow_synthetic=allow_synthetic)
    if {item.name for item in (root / 'tensors').iterdir()} != {digest + '.f32le' for digest in seen}:
        raise ValueError('tensor inventory contains a missing or unreferenced file')
    contract = records[0]['event']
    tensor_bytes = sum(seen.values())
    expected_terminal = {'status': 'complete_candidate', 'events': 721,
        'event_log_bytes': len(events_raw), 'retained_tensor_bytes': tensor_bytes,
        'storage_accounting_complete': True, 'detail': {'status': 'complete_candidate', 'forwards': 720,
            'model_state_sha256_before_after': contract['model_state_sha256'],
            'retained_tensor_bytes': tensor_bytes, 'schedule_sha256': contract['schedule_sha256']}}
    if not science.same_recorded_value(terminal, expected_terminal):
        raise ValueError('terminal or retained-byte accounting differs')
    report['journal'] = {'status': 'consistent_candidate', 'events': 721,
        'started_sha256': hashlib.sha256(started_raw).hexdigest(),
        'events_sha256': hashlib.sha256(events_raw).hexdigest(),
        'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest()}
    return report


def _pinned_file(path, digest, limit):
    raw = _bounded_file(path, limit)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('published input differs from its fixed digest')
    return raw


def load_frozen_inputs(case_path, direction_directory, reference_path):
    """Load the published cases, vectors and numerical helper by exact hash."""
    _pinned_file(Path(science.__file__), _SCIENCE_SHA256, 1024 * 1024)
    raw = _pinned_file(case_path, _CASE_SHA256, 2 * 1024 * 1024)
    document = _json(raw)
    if (type(document) is not dict
            or document.get('schema') != 'distilgpt2-agreement-subject-dependence-cases-v1'
            or document.get('protocol_sha256') != _PROTOCOL_SHA256
            or document.get('reference_sha256') != _REFERENCE_SHA256):
        raise ValueError('published case/protocol/reference binding differs')
    cases = document.get('cases')
    validate_cases(cases, vocabulary=50257)
    reference_raw = _pinned_file(reference_path, _REFERENCE_SHA256, 50257 * 8)
    if len(reference_raw) != 50257 * 8:
        raise ValueError('reference length differs')
    reference = struct.unpack('<50257d', reference_raw)
    if any(not math.isfinite(value) for value in reference):
        raise ValueError('reference contains nonfinite values')
    directions = {}
    for name, digest in _DIRECTION_SHAS.items():
        data = _pinned_file(Path(direction_directory) / (name + '.f64le'), digest, 768 * 8)
        if len(data) != 768 * 8:
            raise ValueError('direction width differs')
        direction = struct.unpack('<768d', data)
        if (any(not math.isfinite(value) for value in direction)
                or abs(math.sqrt(math.fsum(value * value for value in direction)) - 1.0) > 1e-12):
            raise ValueError('a published direction is not finite and unit length')
        directions[name] = direction
    for left, right in (('u', 'v1'), ('u', 'v2'), ('v1', 'v2')):
        if abs(math.fsum(a * b for a, b in zip(directions[left], directions[right]))) > 1e-12:
            raise ValueError('published directions are not orthogonal')
    return cases, directions, reference


def main(argv=None):
    """Write a candidate saved-byte report after independently pinned inputs."""
    import argparse
    parser = argparse.ArgumentParser(description='Independently replay saved agreement-subject evidence.')
    for name in ('attempt', 'cases', 'directions', 'reference', 'output', 'source-revision',
                 'prelaunch-sha256', 'reservation-sha256', 'latest-exposure-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output, attempt = Path(args.output), Path(args.attempt)
    if (not output.is_absolute() or not output.parent.is_dir() or output.exists()
            or output == attempt or attempt in output.parents):
        raise ValueError('a fresh report path outside the attempt is required')
    cases, directions, reference = load_frozen_inputs(args.cases, args.directions, args.reference)
    binding = {'prelaunch_sha256': args.prelaunch_sha256, 'reservation_sha256': args.reservation_sha256,
               'latest_exposure_sha256': args.latest_exposure_sha256}
    result = audit_attempt(attempt, cases, directions, reference, expected_binding=binding,
                           expected_source_revision=args.source_revision)
    result['public_inputs'] = {'cases_sha256': _CASE_SHA256, 'reference_sha256': _REFERENCE_SHA256,
        'protocol_sha256': _PROTOCOL_SHA256, 'direction_sha256': dict(_DIRECTION_SHAS),
        'numeric_source_sha256': _SCIENCE_SHA256, 'owner_source_revision': args.source_revision,
        'external_binding': binding}
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')) + '\n').encode('ascii')
    if len(raw) > 64 * 1024**2:
        raise ValueError('derived report exceeds its bound')
    with output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('derived report write was incomplete')
        stream.flush()
        os.fsync(stream.fileno())
    if _bounded_file(output, 64 * 1024**2) != raw:
        raise ValueError('derived report changed after writing')
    print(json.dumps({'status': 'saved_byte_audit', 'sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
