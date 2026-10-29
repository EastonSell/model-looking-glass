"""Model-free saved-byte replay for the fixed public invariant-noun successor study.

The caller independently verifies the reader source before executing it,
and establishes owner exit, exposure and launch identities separately. This
reader checks saved bytes; it cannot authenticate execution or inspect unsaved
full-sequence logits.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import ModuleType


MARGINS = ('present', 'past', 'the_vs_a', 'in_vs_on')
AXES = ('u', 'v1', 'v2')
TARGETS = (318, 389, 373, 547, 262, 257, 287, 319)
NUMERIC_SHA = '917694d448b796359db0002a630877e5226a2583d273092871892fade7a38ac4'
ACTION_SHA = 'f0625d27ee196eeeb1a85a3577238f0cbcad19017fc6596dfe429dcbc60c610d'
PROTOCOL_SHA = '58e8b4fb26ca04afe727b76938c5ed3ece58a9dd7f37e5a8a18fd7504ef80deb'
CASE_SHA = '188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b'
ROWS_SHA = '2064402fdfb262bcdbee7002d7693fe6b6a14032bd655efcc142713532469787'
DIRECTION_SHA = {'u': 'd1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
                 'v1': '4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
                 'v2': '4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}


def bounded_file(path, limit):
    """Read one bounded ordinary saved file, rejecting incomplete changes."""
    path = Path(path)
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                or info.st_size > limit):
            raise ValueError('saved evidence is not a bounded ordinary file')
        with path.open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit:
            raise ValueError('saved evidence changed or exceeded its bound')
        return raw
    except OSError as error:
        raise ValueError('saved evidence is unavailable') from error


def load_numeric(path):
    """Compile the exact verified helper bytes rather than importing cached code."""
    raw = bounded_file(path, 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != NUMERIC_SHA:
        raise ValueError('saved-tensor numeric helper differs')
    module = ModuleType('_verified_invariant_numeric')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module


def factorial(values):
    """Compute the fixed this/these and same-determiner noun contrasts."""
    if (type(values) not in (list, tuple) or len(values) != 4
            or any(type(x) is not float or not math.isfinite(x) for x in values)):
        raise ValueError('four finite factorial values are required')
    a_this, a_these, b_this, b_these = values
    k_a, k_b = a_this - a_these, b_this - b_these
    first, second = a_this - b_this, a_these - b_these
    d_num = math.fsum((abs(k_a), abs(k_b))) / 2
    d_lex = math.fsum((abs(first), abs(second))) / 2
    return {'k_a': k_a, 'k_b': k_b, 's': math.fsum((k_a, k_b)) / 2,
            'd_num': d_num, 'd_lex': d_lex, 'delta': d_num - d_lex,
            'l': math.fsum((first, second)) / 2, 'i': k_a - k_b,
            'a_this_minus_b_this': first, 'a_these_minus_b_these': second}


def assess_baseline(cells):
    """Apply the fixed, unfiltered baseline rules to six frame/context cells."""
    if type(cells) not in (list, tuple) or len(cells) != 6:
        raise ValueError('six baseline cells are required')
    indexed = {}
    for cell in cells:
        if type(cell) is not dict or set(cell) != {'frame', 'context', 'margins'}:
            raise ValueError('invalid baseline cell')
        frame, context, margins = cell['frame'], cell['context'], cell['margins']
        if (type(frame) is not int or frame not in range(2)
                or type(context) is not int or context not in range(3)
                or type(margins) not in (list, tuple) or len(margins) != 4
                or any(type(x) is not float or not math.isfinite(x) for x in margins)
                or (frame, context) in indexed):
            raise ValueError('invalid or duplicate baseline cell')
        indexed[frame, context] = margins

    frame_correct = [0, 0]
    noun_correct = [0, 0]
    context_correct = [0, 0, 0]
    details = []
    ties = wrong_sign = all_correct_cells = 0
    for frame in range(2):
        for context in range(3):
            margins = indexed[frame, context]
            statuses = []
            for index, margin in enumerate(margins):
                correct = margin > 0 if index % 2 == 0 else margin < 0
                status = 'correct' if correct else 'tie' if margin == 0 else 'wrong_sign'
                statuses.append(status)
                if correct:
                    frame_correct[frame] += 1
                    noun_correct[index // 2] += 1
                    context_correct[context] += 1
                elif status == 'tie':
                    ties += 1
                else:
                    wrong_sign += 1
            all_correct_cells += all(status == 'correct' for status in statuses)
            details.append({'frame': frame, 'context': context,
                            'margins': list(margins), 'statuses': statuses})
    overall = sum(frame_correct)
    adequate = (overall >= 18 and min(frame_correct) >= 9
                and min(noun_correct) >= 9 and min(context_correct) >= 6
                and all_correct_cells >= 4)
    return {'overall_correct': overall, 'frame_correct': frame_correct,
            'noun_correct': noun_correct, 'context_correct': context_correct,
            'all_correct_cells': all_correct_cells, 'ties': ties,
            'wrong_sign': wrong_sign, 'adequate': adequate, 'cells': details}


def assess_criteria(cells, baseline_adequate):
    """Compute both prespecified decisions from every fixed cell and control."""
    if (type(baseline_adequate) is not bool
            or type(cells) not in (list, tuple) or len(cells) != 6):
        raise ValueError('six cells and a baseline adequacy flag are required')
    indexed = {}
    for cell in cells:
        if (type(cell) is not dict
                or set(cell) != {'frame', 'context', 'u', 'v1', 'v2'}):
            raise ValueError('invalid criterion cell')
        frame, context = cell['frame'], cell['context']
        if (type(frame) is not int or frame not in range(2)
                or type(context) is not int or context not in range(3)
                or (frame, context) in indexed):
            raise ValueError('invalid or duplicate criterion cell')
        indexed[frame, context] = {axis: factorial(cell[axis])
                                   for axis in ('u', 'v1', 'v2')}

    ordered = [indexed[frame, context] for frame in range(2)
               for context in range(3)]

    def outcome(statistic, floor, noun_check):
        values = [cell['u'][statistic] for cell in ordered]
        average = math.fsum(values) / 6
        controls = [math.fsum(abs(cell[axis][statistic]) for cell in ordered) / 6
                    for axis in ('v1', 'v2')]
        frames = [math.fsum(values[frame * 3:frame * 3 + 3]) / 3
                  for frame in range(2)]
        positive = sum(value > 0 for value in values)
        nouns = [math.fsum(cell['u'][key] for cell in ordered) / 6
                 for key in ('k_a', 'k_b')]
        passed = (average >= floor and all(average > x for x in controls)
                  and positive >= 4 and all(x > 0 for x in frames)
                  and (not noun_check or all(x > 0 for x in nouns)))
        result = {'u_mean': average, 'v1_abs_mean': controls[0],
                  'v2_abs_mean': controls[1], 'positive_cells': positive,
                  'frame_means': frames, 'raw_pass': passed,
                  'decision': ('baseline_limited' if not baseline_adequate
                               else 'pass' if passed else 'fail')}
        if noun_check:
            result['noun_k_means'] = nouns
        return result

    return {'sensitivity': outcome('s', 0.05, True),
            'number_over_noun': outcome('delta', 0.02, False)}


def fixed_schedule():
    """Reconstruct the public protocol's complete case/action ordering."""
    seed = 'agreement-invariant-noun-v1:20261029'
    action_schedule = []
    def sort_key(split, purpose, identity):
        value = f'{seed}|{split}|{purpose}|{identity}'.encode('utf-8')
        return hashlib.sha256(value).digest(), identity

    def arms(number):
        signs = (('plus', 'minus') if number % 2 == 0 else ('minus', 'plus'))
        initial = tuple(f'{axis}_{sign}' for axis in ('u', 'v1', 'v2') for sign in signs)
        rotate = 2 * (number // 2)
        return initial[rotate:] + initial[:rotate]

    pair_counts = ((2, 2, 3, 2, 2, 3), (2, 2, 2, 3, 3, 2))
    fragment_counts = ((2, 1, 0, 1, 1, 1), (1, 2, 2, 0, 0, 1))
    for split_index, split in enumerate(('discovery', 'evaluation')):
        pairs = {}
        for frame in range(2):
            for context in range(3):
                for noun in range(2):
                    identity = f'{split}_target_f{frame}_c{context}_n{noun}'
                    pairs[identity] = ((frame + context + noun) % 2,
                                       (frame + noun + split_index) % 2)
            identity = f'{split}_demonstrative_f{frame}'
            pairs[identity] = (frame, (frame + split_index) % 2)
        fragment_ids = [f'{split}_fragment_f{frame}_c{context}'
                        for frame in range(2) for context in range(3)]
        assigned = {}
        for parity in (0, 1):
            members = sorted((name for name, (p, _) in pairs.items() if p == parity),
                             key=lambda name: sort_key(split, 'pair-pattern', name))
            pattern_numbers = [i for i in range(6) if i % 2 == parity
                               for _ in range(pair_counts[split_index][i])]
            if len(members) != len(pattern_numbers):
                raise ValueError('fixed pair allocation differs')
            assigned.update(zip(members, pattern_numbers))
        fragment_order = sorted(fragment_ids, key=lambda name: sort_key(split, 'fragment-pattern', name))
        fragment_patterns = [i for i in range(6)
                             for _ in range(fragment_counts[split_index][i])]
        assigned.update(zip(fragment_order, fragment_patterns))
        pair_order = sorted(pairs, key=lambda name: sort_key(split, 'pair-order', name))
        fragments = sorted(fragment_ids, key=lambda name: sort_key(split, 'fragment-order', name))

        def add_case(identity, pattern):
            action_schedule.extend((identity, action) for action in
                                   ('clean', 'noop', *arms(pattern), 'clean_after'))

        for index, identity in enumerate(pair_order, 1):
            determiners = ('these', 'this') if pairs[identity][1] else ('this', 'these')
            for determiner in determiners:
                add_case(f'{identity}_{determiner}', assigned[identity])
            if index in (2, 4, 6, 8, 10, 12):
                identity = fragments[index // 2 - 1]
                add_case(identity, assigned[identity])
    if len(action_schedule) != 612:
        raise ValueError('fixed schedule length differs')
    return [('v2_' + name if name.startswith('discovery_') else name, action)
            for name, action in action_schedule]


def _identity(name):
    if name.startswith('v2_discovery_'):
        name = name[3:]
    elif name.startswith('discovery_'):
        raise ValueError('v1 discovery identity is not a successor case')
    target = re.fullmatch(r'(discovery|evaluation)_target_f([01])_c([012])_n([01])_(this|these)', name)
    if target:
        split, frame, context, noun, determiner = target.groups()
        return {'split': split, 'role': 'target', 'frame': int(frame),
                'context': int(context), 'noun': int(noun), 'determiner': determiner}
    fragment = re.fullmatch(r'(discovery|evaluation)_fragment_f([01])_c([012])', name)
    if fragment:
        split, frame, context = fragment.groups()
        return {'split': split, 'role': 'fragment', 'frame': int(frame),
                'context': int(context), 'noun': None, 'determiner': None}
    demonstrative = re.fullmatch(r'(discovery|evaluation)_demonstrative_f([01])_(this|these)', name)
    if demonstrative:
        split, frame, determiner = demonstrative.groups()
        return {'split': split, 'role': 'demonstrative', 'frame': int(frame),
                'context': 0, 'noun': None, 'determiner': determiner}
    raise ValueError('unknown case identity')


def validate_cases(cases, *, vocabulary):
    """Verify the fixed case identities, token alignment and internal prefixes."""
    schedule = fixed_schedule()
    order = [schedule[index][0] for index in range(0, len(schedule), 9)]
    if (type(cases) not in (list, tuple) or len(cases) != 68
            or type(vocabulary) is not int or vocabulary < 778):
        raise ValueError('fixed case inventory and vocabulary are required')
    rows, groups = {}, []
    common = {'case_id', 'split', 'role', 'frame_index', 'context_index',
              'noun_index', 'determiner', 'text', 'token_ids', 'final_index'}
    for row, expected_id in zip(cases, order):
        if type(row) is not dict or row.get('case_id') != expected_id:
            raise ValueError('case order differs')
        meta = _identity(expected_id)
        extras = set() if meta['role'] == 'fragment' else {'determiner_index'}
        if meta['role'] == 'target':
            extras.add('subject_index')
        ids = row.get('token_ids')
        coordinates = (row.get('split'), row.get('role'), row.get('frame_index'),
                       row.get('context_index'), row.get('noun_index'), row.get('determiner'))
        expected = (meta['split'], meta['role'], meta['frame'], meta['context'],
                    meta['noun'], meta['determiner'])
        if (set(row) != common | extras or coordinates != expected
                or type(row.get('frame_index')) is not int
                or type(row.get('context_index')) is not int
                or (meta['role'] == 'target' and type(row.get('noun_index')) is not int)
                or type(row.get('text')) is not str or not row['text'] or len(row['text']) > 4096
                or type(ids) is not list or not 1 <= len(ids) <= 12
                or any(type(x) is not int or not 0 <= x < vocabulary for x in ids)
                or type(row.get('final_index')) is not int or row['final_index'] != len(ids) - 1):
            raise ValueError('case metadata or input differs')
        if meta['role'] != 'fragment':
            d = row['determiner_index']
            token = 428 if meta['determiner'] == 'this' else 777
            if (type(d) is not int or not 0 <= d < len(ids) - 1
                    or ids[d] != token or ids.count(token) != 1):
                raise ValueError('determiner position differs')
        if meta['role'] == 'target':
            n = row['subject_index']
            if type(n) is not int or not d < n < len(ids) - 1:
                raise ValueError('noun position differs')
        base = tuple(ids)
        groups.append([base] + [base + (token,) for token in TARGETS])
        rows[expected_id] = row
    for index, group in enumerate(groups):
        for prior in groups[:index]:
            if any(a[:len(b)] == b or b[:len(a)] == a for a in group for b in prior):
                raise ValueError('distinct case groups have a prefix collision')

    nouns = {}
    for split in ('discovery', 'evaluation'):
        case_split = 'v2_discovery' if split == 'discovery' else 'evaluation'
        for frame in range(2):
            for context in range(3):
                prefix = f'{case_split}_target_f{frame}_c{context}_n'
                group = [rows[f'{prefix}{noun}_{det}']
                         for noun in range(2) for det in ('this', 'these')]
                d, n = group[0]['determiner_index'], group[0]['subject_index']
                for row in group:
                    if (row['determiner_index'] != d or row['subject_index'] != n
                            or len(row['token_ids']) != len(group[0]['token_ids'])
                            or any(a != b for i, (a, b) in enumerate(zip(row['token_ids'], group[0]['token_ids']))
                                   if i not in (d, n))):
                        raise ValueError('factorial case alignment differs')
                    key = split, row['noun_index']
                    word = row['token_ids'][n]
                    if key in nouns and nouns[key] != word:
                        raise ValueError('noun token changes across fixed cells')
                    nouns[key] = word
                if group[0]['token_ids'][n] == group[2]['token_ids'][n]:
                    raise ValueError('noun controls are identical')
            for noun in range(2):
                for det in ('this', 'these'):
                    a = rows[f'{case_split}_target_f{frame}_c1_n{noun}_{det}']['token_ids']
                    b = rows[f'{case_split}_target_f{frame}_c2_n{noun}_{det}']['token_ids']
                    if a[:-1] != b[:-1] or a[-1] == b[-1]:
                        raise ValueError('attractor pair differs outside final token')
            a = rows[f'{case_split}_fragment_f{frame}_c1']['token_ids']
            b = rows[f'{case_split}_fragment_f{frame}_c2']['token_ids']
            if a[:-1] != b[:-1] or a[-1] == b[-1]:
                raise ValueError('fragment attractor pair differs')
            a = rows[f'{case_split}_demonstrative_f{frame}_this']
            b = rows[f'{case_split}_demonstrative_f{frame}_these']
            d = a['determiner_index']
            if (d != b['determiner_index'] or len(a['token_ids']) != len(b['token_ids'])
                    or a['token_ids'][:d] + a['token_ids'][d + 1:]
                    != b['token_ids'][:d] + b['token_ids'][d + 1:]):
                raise ValueError('demonstrative pair differs outside determiner')
    if len(set(nouns.values())) != 4:
        raise ValueError('discovery and evaluation noun tokens overlap')
    return True


def expected_records(cases, *, vocabulary):
    """Bind the independently reconstructed order to the public case metadata."""
    validate_cases(cases, vocabulary=vocabulary)
    result = []
    for index, (case_id, action) in enumerate(fixed_schedule()):
        case = cases[index // 9]
        result.append({'index': index, 'case_id': case_id, 'case_index': index // 9,
                       'split': case['split'], 'frame_index': case['frame_index'],
                       'context_index': case['context_index'], 'role': case['role'],
                       'noun_index': case['noun_index'], 'determiner': case['determiner'],
                       'action': action,
                       'slot': index % 9 - 2 if 2 <= index % 9 < 8 else None})
    return result


def score_margins(margins):
    """Score all fixed cases without filtering any baseline or control."""
    schedule = fixed_schedule()
    case_order = [schedule[index][0] for index in range(0, len(schedule), 9)]
    if type(margins) is not dict or set(margins) != set(case_order):
        raise ValueError('complete fixed case inventory is required')
    wanted_actions = {'clean', 'noop', 'clean_after'} | {
        f'{axis}_{sign}' for axis in AXES for sign in ('plus', 'minus')}
    per_case, targets = [], {}
    for name in case_order:
        actions = margins[name]
        if type(actions) is not dict or set(actions) != wanted_actions:
            raise ValueError('complete case action inventory is required')
        for value in actions.values():
            if (type(value) is not dict or set(value) != set(MARGINS)
                    or any(type(x) is not float or not math.isfinite(x) for x in value.values())):
                raise ValueError('finite case margins are required')
        if actions['clean'] != actions['noop'] or actions['clean'] != actions['clean_after']:
            raise ValueError('saved baseline margins differ')
        odd, even = {}, {}
        for axis in AXES:
            plus, minus = actions[f'{axis}_plus'], actions[f'{axis}_minus']
            odd[axis] = {metric: math.fsum((plus[metric], -minus[metric])) / 2
                         for metric in MARGINS}
            even[axis] = {metric: math.fsum((plus[metric], minus[metric])) / 2
                          - actions['clean'][metric] for metric in MARGINS}
        metadata = _identity(name)
        row = {'case_id': name, **metadata, 'margins': actions, 'odd': odd, 'even': even}
        per_case.append(row)
        if row['role'] == 'target':
            key = tuple(row[k] for k in ('split', 'frame', 'context', 'noun', 'determiner'))
            targets[key] = row

    def summary(values):
        return {'mean': math.fsum(values) / len(values),
                'positive': sum(x > 0 for x in values),
                'zero': sum(x == 0 for x in values),
                'negative': sum(x < 0 for x in values)}

    partitions = {}
    for split in ('discovery', 'evaluation'):
        cells, baseline_cells, criterion_cells = [], [], []
        for frame in range(2):
            for context in range(3):
                members = [targets[split, frame, context, noun, determiner]
                           for noun in range(2) for determiner in ('this', 'these')]
                statistics = {metric: {axis: factorial([row['odd'][axis][metric] for row in members])
                                       for axis in AXES} for metric in MARGINS}
                cells.append({'frame': frame, 'context': context,
                              'case_ids': [row['case_id'] for row in members],
                              'statistics': statistics})
                baseline_cells.append({'frame': frame, 'context': context,
                    'margins': tuple(row['margins']['clean']['present'] for row in members)})
                criterion_cells.append({'frame': frame, 'context': context,
                    **{axis: tuple(row['odd'][axis]['present'] for row in members) for axis in AXES}})
        baseline = assess_baseline(baseline_cells)

        def aggregate(group):
            return {metric: {axis: {statistic: summary([
                        row['statistics'][metric][axis][statistic] for row in group])
                        for statistic in ('k_a', 'k_b', 's', 'd_num', 'd_lex', 'delta',
                                          'l', 'i', 'a_this_minus_b_this', 'a_these_minus_b_these')}
                        for axis in AXES} for metric in MARGINS}

        all_aggregates = {'overall': aggregate(cells),
            'by_frame': {str(frame): aggregate([row for row in cells if row['frame'] == frame])
                         for frame in range(2)},
            'by_context': {str(context): aggregate([row for row in cells if row['context'] == context])
                           for context in range(3)}}
        all_aggregates['by_noun'] = {
            str(noun): {metric: {axis: summary([
                row['statistics'][metric][axis][('k_a', 'k_b')[noun]] for row in cells])
                for axis in AXES} for metric in MARGINS} for noun in range(2)}
        partitions[split] = {'cells': cells, 'baseline': baseline,
                             'criteria': assess_criteria(criterion_cells, baseline['adequate']),
                             'aggregates': all_aggregates}
    return {'cases': per_case, 'partitions': partitions,
            'fragments': [row for row in per_case if row['role'] == 'fragment'],
            'demonstratives': [row for row in per_case if row['role'] == 'demonstrative']}


def validate_directions(directions, width):
    if type(directions) is not dict or set(directions) != set(AXES):
        raise ValueError('three fixed directions are required')
    for values in directions.values():
        if (type(values) not in (tuple, list) or len(values) != width
                or any(type(x) is not float or not math.isfinite(x) for x in values)
                or abs(math.sqrt(math.fsum(x*x for x in values)) - 1.0) > 1e-12):
            raise ValueError('direction is not finite and unit length')
    for a, b in (('u', 'v1'), ('u', 'v2'), ('v1', 'v2')):
        if abs(math.fsum(x*y for x, y in zip(directions[a], directions[b]))) > 1e-12:
            raise ValueError('directions are not orthogonal')


def validate_budget(budget, width, vocabulary):
    keys = {'successful_logical_bytes_upper_bound', 'failed_return_bytes_reserve',
            'invalid_candidate_bytes_reserve', 'retained_tensor_bytes_limit',
            'per_forward_bytes_limit', 'event_log_bytes_limit'}
    success = 612 * vocabulary * 4 + 476 * (2 * 12 * width * 4 + 2 * width * 4)
    failure = 4 * (12 * vocabulary + 2 * 12 * width + 2 * width)
    if (type(budget) is not dict or set(budget) != keys
            or any(type(x) is not int for x in budget.values())
            or budget['successful_logical_bytes_upper_bound'] != success
            or budget['failed_return_bytes_reserve'] != failure
            or budget['invalid_candidate_bytes_reserve'] != 2 * width * 4
            or not success + failure + 2 * width * 4 <= budget['retained_tensor_bytes_limit'] <= 256 * 1024**2
            or not failure <= budget['per_forward_bytes_limit'] <= 4 * 1024**2
            or budget['event_log_bytes_limit'] != 32 * 1024**2):
        raise ValueError('saved evidence reservation differs')


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
        raise ValueError('recorded model configuration differs')
    fields = {}
    for entry in root[2]:
        if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str or entry[0] in fields:
            raise ValueError('invalid recorded model configuration')
        fields[entry[0]] = entry[1]
    wanted = {'use_cache': ['bool', False], '_attn_implementation': ['str', 'eager'],
              'n_layer': ['int', 6], 'n_embd': ['int', width], 'vocab_size': ['int', vocabulary]}
    if any(name not in fields or not science.same_recorded_value(fields[name], value)
           for name, value in wanted.items()):
        raise ValueError('recorded model configuration differs')


def require_binary64_subnormals():
    """Fail if this calling thread flushes binary64 subnormal arithmetic."""
    half = math.ldexp(1.0, -1022) * 0.5
    tiny = math.exp(-745.0) * 2.0
    bits = lambda value: struct.unpack('<Q', struct.pack('<d', value))[0]
    if bits(half) != 0x0008000000000000 or bits(tiny) != 0x0000000000000002:
        raise ValueError('calling-thread binary64 subnormal arithmetic differs')



def _normalizer_shift(clean, arm, p_peak, q_peak, p_normalizer, q_normalizer):
    """Compute log(Q/P), preserving small and individually underflowed changes."""
    direct, positive_logs, negative_logs = [], [], []
    smallest_normal = math.ldexp(1.0, -1022)
    for p, q in zip(clean, arm):
        change = math.fsum((q, -q_peak, -p, p_peak))
        if change == 0.0:
            continue
        p_log, q_log = math.fsum((p, -p_peak)), math.fsum((q, -q_peak))
        p_weight, q_weight = math.exp(p_log), math.exp(q_log)
        difference = (p_weight * math.expm1(change) if abs(change) < 0.5
                      else q_weight - p_weight)
        if (p_weight >= smallest_normal and q_weight >= smallest_normal
                and abs(difference) >= smallest_normal):
            direct.append(difference)
            continue
        if abs(change) < 0.5:
            log_magnitude = math.fsum((p_log, math.log(abs(math.expm1(change)))))
        elif change > 0:
            log_magnitude = math.fsum((q_log, math.log1p(-math.exp(-change))))
        else:
            log_magnitude = math.fsum((p_log, math.log1p(-math.exp(change))))
        (positive_logs if change > 0 else negative_logs).append(log_magnitude)

    def total(log_values):
        if not log_values:
            return 0.0
        peak = max(log_values)
        position = log_values.index(peak)
        rest = math.fsum(math.exp(value - peak) for i, value in enumerate(log_values) if i != position)
        return math.exp(peak + math.log1p(rest))

    mass_change = math.fsum((math.fsum(direct), total(positive_logs), -total(negative_logs)))
    relative_change = mass_change * math.exp(-p_normalizer)
    if -0.5 < relative_change < 0.5:
        return math.log1p(relative_change)
    return math.fsum((q_normalizer, -p_normalizer))


def stable_kl(clean, arm):
    """KL(clean||arm) from nonnegative log-domain terms, preserving tiny tails.

    For delta=log(q/p), sum p*(exp(delta)-1-delta); sum(q-p)=0 in the
    normalized distributions. The series avoids cancellation near zero, and
    the final flag records positive values below binary64's smallest number.
    """
    require_binary64_subnormals()
    if (type(clean) not in (list, tuple) or type(arm) not in (list, tuple)
            or not clean or len(clean) != len(arm)
            or any(type(x) is not float or not math.isfinite(x) for x in (*clean, *arm))):
        raise ValueError('finite matching logits are required')

    def normalizer(values):
        peak = max(values)
        peak_index = values.index(peak)
        remainder = math.fsum(math.exp(value - peak) for index, value in enumerate(values)
                              if index != peak_index)
        return peak, math.log1p(remainder)

    p_peak, p_normalizer = normalizer(clean)
    q_peak, q_normalizer = normalizer(arm)

    normalizer_shift = _normalizer_shift(clean, arm, p_peak, q_peak, p_normalizer, q_normalizer)

    terms = []
    for p, q in zip(clean, arm):
        delta = math.fsum((q, -q_peak, -p, p_peak, -normalizer_shift))
        if delta == 0.0:
            continue
        if abs(delta) < 0.01:
            x = delta
            factor = math.fsum((0.5, x/6, x*x/24, x**3/120, x**4/720,
                                x**5/5040, x**6/40320, x**7/362880))
            log_phi = 2 * math.log(abs(x)) + math.log(factor)
        elif delta > 50:
            correction = math.log1p(-(1 + delta) * math.exp(-delta))
            terms.append(math.fsum((q, -q_peak, -q_normalizer, correction)))
            continue
        else:
            log_phi = math.log(math.expm1(delta) - delta)
        terms.append(math.fsum((p, -p_peak, -p_normalizer, log_phi)))
    if not terms:
        require_binary64_subnormals()
        return 0.0, False
    peak = max(terms)
    peak_index = terms.index(peak)
    remainder = math.fsum(math.exp(value - peak) for index, value in enumerate(terms)
                          if index != peak_index)
    log_result = peak + math.log1p(remainder)
    try:
        value = math.exp(log_result)
    except OverflowError as error:
        raise ValueError('KL exceeds the supported numeric range') from error
    if not math.isfinite(value):
        raise ValueError('KL is not finite')
    require_binary64_subnormals()
    return value, value == 0.0


def audit_records(records, cases, tensor_dir, directions, science, *, expected_binding,
                  width=768, vocabulary=50257, allow_synthetic=False):
    """Check saved logits, fixed ordering and every delivery witness without a model."""
    require_binary64_subnormals()
    if (type(allow_synthetic) is not bool or type(width) is not int or type(vocabulary) is not int
            or not 1 <= width <= 768 or not 778 <= vocabulary <= 50257
            or (not allow_synthetic and (width != 768 or vocabulary != 50257))):
        raise ValueError('unsupported replay dimensions')
    validate_directions(directions, width)
    schedule = expected_records(cases, vocabulary=vocabulary)
    if type(records) not in (list, tuple) or len(records) != 613:
        raise ValueError('complete contract and 612 forwards are required')
    for index, record in enumerate(records):
        if (type(record) is not dict or set(record) != {'index', 'event', 'tensors'}
                or type(record['index']) is not int or record['index'] != index
                or type(record['event']) is not dict or type(record['tensors']) is not dict):
            raise ValueError('saved event index or fields differ')
    contract = records[0]['event']
    fields = {'kind', 'expected_forwards', 'environment', 'seed', 'execution_configuration',
              'model_state_sha256', 'external_binding', 'schedule_sha256', 'arm_placement_counts', 'evidence_budget'}
    placement = {axis + '_' + sign: [sum(row['action'] == axis + '_' + sign and row['slot'] == slot
                                        for row in schedule) for slot in range(6)]
                 for axis in AXES for sign in ('plus', 'minus')}
    if (set(contract) != fields or records[0]['tensors'] or contract['kind'] != 'run_contract'
            or type(contract['expected_forwards']) is not int or contract['expected_forwards'] != 612
            or type(contract['seed']) is not int or contract['seed'] != 20261030
            or not science.same_recorded_value(contract['external_binding'], expected_binding)
            or contract['schedule_sha256'] != ACTION_SHA
            or not science.same_recorded_value(contract['arm_placement_counts'], placement)
            or type(contract['model_state_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', contract['model_state_sha256']) is None):
        raise ValueError('saved run contract differs')
    validate_budget(contract['evidence_budget'], width, vocabulary)
    if not allow_synthetic:
        validate_runtime(contract, science, width, vocabulary)
    by_id = {row['case_id']: row for row in cases}
    seen, all_margins, details = {}, {}, {}

    def payload(reference, shape):
        raw = science.read_saved_tensor(tensor_dir, reference, shape)
        digest = reference['sha256']
        if digest in seen and seen[digest] != len(raw):
            raise ValueError('repeated tensor reference has inconsistent size')
        seen[digest] = len(raw)
        return raw

    for begin in range(0, 612, 9):
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
                    or any(not science.same_recorded_value(event.get(key), value) for key, value in expected.items())
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
            if action in ('noop', 'clean_after') and logits_raw != clean_raw:
                raise ValueError('full clean, no-op or repeat logits differ')
            margins[action] = {name: logits[left] - logits[right]
                               for name, left, right in zip(MARGINS, TARGETS[::2], TARGETS[1::2])}
            vocabulary_result = science.vocabulary_diagnostics(clean_values, logits, TARGETS)
            kl, underflow = stable_kl(clean_values, logits)
            vocabulary_result['kl_clean_to_arm'] = kl
            vocabulary_result['kl_underflowed_to_zero'] = underflow
            vocabulary_result['top_one_changed'] = vocabulary_result['top_clean'][0] != vocabulary_result['top_arm'][0]
            vocab[action] = vocabulary_result
            if not hooked:
                if event['dose'] is not None or event['plan_metadata'] != {'steps': []}:
                    raise ValueError('unhooked action has a plan or dose')
                continue
            site, plan = science.expected_hook_plan('noop_l3_final' if action == 'noop' else action,
                                                    sequence_length=length, width=width)
            if (not science.same_recorded_value(event['site'], site)
                    or not science.same_recorded_value(event['plan_metadata'], plan)
                    or event['donor'] is not None):
                raise ValueError('hook site, plan or donor differs')
            before, after = payload(refs['before'], (width,)), payload(refs['after'], (width,))
            whole_before = payload(refs['whole_before'], (1, length, width))
            whole_after = payload(refs['whole_after'], (1, length, width))
            science.verify_hook_witness(before, after, whole_before, whole_after,
                                        width=width, sequence_length=length, token_index=length - 1)
            if action == 'noop':
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
            projection = math.fsum((a-b)*d for a, b, d in zip(delivered, original, directions[axis]))
            relative = abs(abs(projection) - 2.0) / 2.0
            dose['signed_projected_displacement'] = projection
            dose['projected_relative_error'] = relative
            dose['dose_valid'] = dose['dose_valid'] and projection * amount > 0.0 and relative <= 1e-5
            if after != landed or dose['dose_valid'] is not True or not science.same_recorded_value(event['dose'], dose):
                raise ValueError('delivered vector or realized dose differs')
            deliveries[action] = dose
        all_margins[case_id] = margins
        details[case_id] = {'vocabulary': vocab, 'deliveries': deliveries}
    report = score_margins(all_margins)
    for row in report['cases']:
        row.update(details[row['case_id']])
    report.update({'schedule': {'sha256': ACTION_SHA, 'arm_placement_counts': placement, 'actions': schedule},
                   'source_tensor_files': len(seen), 'source_tensor_bytes': sum(seen.values()),
                   'dimensions': {'width': width, 'vocabulary': vocabulary},
                   'scope': ('Saved-byte numerical consistency only; unsaved sequence finiteness, '
                             'process exit and exposure are separately established.')})
    require_binary64_subnormals()
    return report, seen


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
            raise ValueError('nonfinite JSON number')
        return result

    def reject(_):
        raise ValueError('nonfinite JSON constant')

    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_float=number, parse_constant=reject)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError('invalid JSON evidence') from error


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
    """Check a closed candidate against independently observed launch/event pins."""
    require_binary64_subnormals()
    names = {'prelaunch_sha256', 'reservation_sha256', 'latest_exposure_sha256'}
    if (type(expected_binding) is not dict or set(expected_binding) != names
            or any(type(x) is not str or re.fullmatch(r'[0-9a-f]{64}', x) is None
                   for x in expected_binding.values())
            or type(expected_source_revision) is not str
            or re.fullmatch(r'[0-9a-f]{40}', expected_source_revision) is None
            or type(expected_events_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected_events_sha256) is None
            or type(expected_scope) is not str or not expected_scope or len(expected_scope) > 256
            or (not allow_synthetic and expected_scope != 'public prospective agreement invariant-noun')):
        raise ValueError('independent launch, source and event identities are required')
    root = Path(directory)
    if (not ordinary_directory(root)
            or {item.name for item in root.iterdir()} != {'started.json', 'events.jsonl', 'terminal.json', 'tensors'}
            or not ordinary_directory(root / 'tensors')):
        raise ValueError('attempt is incomplete or ambiguous')
    start_raw = bounded_file(root / 'started.json', 256 * 1024)
    events_raw = bounded_file(root / 'events.jsonl', 32 * 1024**2)
    terminal_raw = bounded_file(root / 'terminal.json', 256 * 1024)
    start, terminal = strict_json(start_raw), strict_json(terminal_raw)
    wanted_start = {'scope': expected_scope, 'external_binding': expected_binding,
                    'source_revision': expected_source_revision}
    if (not science.same_recorded_value(start, wanted_start)
            or type(terminal) is not dict or terminal.get('status') != 'complete_candidate'
            or not events_raw.endswith(b'\n')
            or hashlib.sha256(events_raw).hexdigest() != expected_events_sha256):
        raise ValueError('start, terminal or independently pinned event log differs')
    lines = events_raw.splitlines()
    if len(lines) != 613 or any(not line or len(line) > 256 * 1024 for line in lines):
        raise ValueError('event log does not contain the complete fixed schedule')
    records = [strict_json(line) for line in lines]
    report, seen = audit_records(records, cases, root / 'tensors', directions, science,
        expected_binding=expected_binding, width=width, vocabulary=vocabulary, allow_synthetic=allow_synthetic)
    if {item.name for item in (root / 'tensors').iterdir()} != {digest + '.f32le' for digest in seen}:
        raise ValueError('tensor inventory has a missing or unreferenced file')
    tensor_bytes = sum(seen.values())
    wanted_terminal = {'status': 'complete_candidate', 'events': 613,
        'event_log_bytes': len(events_raw), 'retained_tensor_bytes': tensor_bytes,
        'storage_accounting_complete': True, 'detail': {'status': 'complete_candidate', 'forwards': 612,
            'model_state_sha256_before_after': records[0]['event']['model_state_sha256'],
            'retained_tensor_bytes': tensor_bytes, 'schedule_sha256': ACTION_SHA}}
    if not science.same_recorded_value(terminal, wanted_terminal):
        raise ValueError('terminal or retained tensor accounting differs')
    report['journal'] = {'status': 'consistent_candidate', 'events': 613,
        'started_sha256': hashlib.sha256(start_raw).hexdigest(),
        'events_sha256': expected_events_sha256,
        'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest()}
    require_binary64_subnormals()
    return report


def pinned(path, expected, maximum):
    raw = bounded_file(path, maximum)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('published input differs from its fixed digest')
    return raw


def load_frozen_inputs(case_path, direction_directory):
    document = strict_json(pinned(case_path, CASE_SHA, 2 * 1024 * 1024))
    if (type(document) is not dict or document.get('schema') != 'agreement-invariant-noun-successor-cases-v2'
            or document.get('protocol_sha256') != PROTOCOL_SHA
            or document.get('protocol_revision') != '6f241157c7617be04fddf2624ccad6f24388cd85'
            or document.get('model_id') != 'distilbert/distilgpt2'
            or document.get('model_revision') != '2290a62682d06624634c1f46a6ad5be0f47f38aa'
            or document.get('tokenizer_sha256') != '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
            or document.get('actions_sha256') != ACTION_SHA
            or document.get('ordered_rows_sha256') != ROWS_SHA
            or type(document.get('expected_forwards')) is not int or document['expected_forwards'] != 612):
        raise ValueError('published case, protocol or model binding differs')
    cases = document.get('cases')
    validate_cases(cases, vocabulary=50257)
    rows = json.dumps([row['token_ids'] for row in cases], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    if hashlib.sha256(rows).hexdigest() != document.get('ordered_rows_sha256'):
        raise ValueError('published ordered input rows differ')
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
    require_binary64_subnormals()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'cases', 'directions', 'numeric-source', 'output', 'source-revision',
                 'prelaunch-sha256', 'reservation-sha256', 'latest-exposure-sha256',
                 'events-sha256', 'start-scope'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output, attempt = Path(args.output), Path(args.attempt)
    if not output.is_absolute() or not output.parent.is_dir() or os.path.lexists(output):
        raise ValueError('fresh report path outside the attempt is required')
    resolved_attempt, resolved_output = attempt.resolve(strict=True), output.resolve(strict=False)
    if resolved_output == resolved_attempt or resolved_attempt in resolved_output.parents:
        raise ValueError('fresh report path outside the attempt is required')
    science = load_numeric(args.numeric_source)
    cases, directions = load_frozen_inputs(args.cases, args.directions)
    binding = {'prelaunch_sha256': args.prelaunch_sha256, 'reservation_sha256': args.reservation_sha256,
               'latest_exposure_sha256': args.latest_exposure_sha256}
    result = audit_attempt(attempt, cases, directions, science, expected_binding=binding,
        expected_source_revision=args.source_revision, expected_events_sha256=args.events_sha256,
        expected_scope=args.start_scope)
    require_binary64_subnormals()
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
