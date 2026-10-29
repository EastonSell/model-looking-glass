"""Independent arithmetic checks for retained matched-dose BERT witnesses.

This module consumes supplied bytes only. It does not run a model, identify
the source of a witness, prove chronology, or establish execution provenance.
"""
import base64
import binascii
import copy
import hashlib
import json
import math
import os
import re
import stat
import struct

_SITES = ('block0', 'block2')
_DONORS = ('A_opposite', 'B_same', 'B_opposite')
_ACTIONS = ('self', *_DONORS, 'artificial')
_CELL_ORDER = (('singular', 'singular'), ('singular', 'plural'),
               ('plural', 'singular'), ('plural', 'plural'))
_HEX = re.compile(r'[0-9a-f]{64}\Z')


def _round_integer(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2 * remainder > denominator or
                       (2 * remainder == denominator and quotient % 2 == 1))


def _round_binary32(value):
    """Round a binary64 value to finite binary32 bits, ties to even."""
    if type(value) is not float:
        raise ValueError('expected a binary64 component')
    raw = int.from_bytes(struct.pack('<d', value), 'little')
    sign, exp = (raw >> 32) & 0x80000000, (raw >> 52) & 0x7ff
    fraction = raw & ((1 << 52) - 1)
    if exp == 0x7ff:
        raise ValueError('nonfinite component')
    significand = fraction | ((1 << 52) if exp else 0)
    if not significand:
        return sign
    power = exp - 1075 if exp else -1074
    top = significand.bit_length() - 1 + power
    shift = power + (149 if top < -126 else 23 - top)
    if shift >= 0:
        rounded = significand << shift
    else:
        rounded = _round_integer(significand, 1 << -shift)
    if top < -126:
        return sign | rounded
    if rounded == 1 << 24:
        rounded >>= 1
        top += 1
    if top > 127:
        raise ValueError('component overflows binary32')
    return sign | ((top + 127) << 23) | (rounded - (1 << 23))


def _decode(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError('invalid float32 bit pattern')
    exp, fraction = (bits >> 23) & 255, bits & 0x7fffff
    if exp == 255:
        raise ValueError('nonfinite float32 bit pattern')
    value = math.ldexp(float(fraction if exp == 0 else fraction | (1 << 23)),
                       -149 if exp == 0 else exp - 150)
    return -value if bits >> 31 else value


def _vector(raw):
    if type(raw) is not bytes or len(raw) != 3072:
        raise ValueError('expected exactly 768 saved float32 components')
    return tuple(_decode(item[0]) for item in struct.iter_unpack('<I', raw))


def _norm(values):
    result = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(result):
        raise ValueError('nonfinite norm')
    return result


def _payload(values):
    bits = [_round_binary32(value) for value in values]
    return struct.pack('<768I', *bits)


def _artificial(row_id, site, base, radius):
    prefix = b'bert-matched-v1\0' + row_id.encode('ascii') + b'\0' + site.encode('ascii') + b'\0'
    digest = b''.join(hashlib.sha256(prefix + index.to_bytes(4, 'big')).digest() for index in range(3))
    scale = radius / math.sqrt(768.0)
    return _payload(math.fsum((value, scale if (digest[i // 8] >> (7 - i % 8)) & 1 else -scale))
                    for i, value in enumerate(base))


def audit_witness(row_id, captures, patches):
    """Reconstruct one target's saved doses and verify exact action landing.

    Captures map each site to its target and three donor float32 byte vectors.
    Patches map each site/action to submitted, pre-action and post-action
    vectors. The caller must separately bind these to ordered journal events,
    inputs, source/model identity and full-logit control evidence.
    """
    if (type(row_id) is not str or not row_id or len(row_id) > 128 or not row_id.isascii()
            or type(captures) is not dict or set(captures) != set(_SITES)
            or type(patches) is not dict or set(patches) != set(_SITES)):
        raise ValueError('invalid witness identity')
    bases, directions, distances = {}, {}, {}
    for site in _SITES:
        capture = captures[site]
        if type(capture) is not dict or set(capture) != {'target', *_DONORS}:
            raise ValueError('incomplete donor captures')
        bases[site] = _vector(capture['target'])
        for donor in _DONORS:
            values = _vector(capture[donor])
            difference = tuple(math.fsum((value, -base)) for value, base in zip(values, bases[site]))
            distance = _norm(difference)
            if distance < 1e-4:
                raise ValueError('a donor distance is below the numerical floor')
            directions[site, donor] = difference
            distances[site, donor] = distance
    radius = min(distances.values())
    result = {'radius': radius, 'sites': {}}
    for site in _SITES:
        base, base_bytes = bases[site], captures[site]['target']
        if type(patches[site]) is not dict or set(patches[site]) != set(_ACTIONS):
            raise ValueError('incomplete action witnesses')
        base_norm = _norm(base)
        details = {}
        for action in _ACTIONS:
            witness = patches[site][action]
            if (type(witness) is not dict or set(witness) != {'submitted', 'pre', 'post'}
                    or any(type(value) is not bytes for value in witness.values())):
                raise ValueError('invalid action witness')
            if action == 'self':
                expected, distance, fraction = base_bytes, None, None
            elif action == 'artificial':
                expected, distance, fraction = _artificial(row_id, site, base, radius), None, None
            else:
                distance = distances[site, action]
                fraction = radius / distance
                if not math.isfinite(fraction) or not 0 < fraction <= 1:
                    raise ValueError('invalid dose fraction')
                expected = _payload(math.fsum((value, fraction * delta))
                                    for value, delta in zip(base, directions[site, action]))
            if (witness['pre'] != base_bytes or witness['submitted'] != expected
                    or witness['post'] != expected):
                raise ValueError('submitted or landed action differs')
            after = _vector(witness['post'])
            realized = _norm(math.fsum((value, -before)) for value, before in zip(after, base))
            target_radius = 0.0 if action == 'self' else radius
            absolute_error = abs(realized - target_radius)
            if action != 'self' and absolute_error > max(1e-6, 1e-5 * radius):
                raise ValueError('landed dose exceeds the fixed tolerance')
            details[action] = {'distance': distance, 'fraction': fraction, 'realized_norm': realized,
                'absolute_error': absolute_error, 'relative_error': None if action == 'self' else absolute_error / radius,
                'base_norm': base_norm, 'realized_over_base_norm': None if base_norm == 0 else realized / base_norm}
        result['sites'][site] = details
    return result


def _observation(value):
    if (type(value) is not dict or set(value) != {'bits', 'mask_sha256'}
            or type(value['bits']) is not list or len(value['bits']) != 2
            or type(value['mask_sha256']) is not str or _HEX.fullmatch(value['mask_sha256']) is None):
        raise ValueError('invalid saved mask observation')
    margin = math.fsum((_decode(value['bits'][0]), -_decode(value['bits'][1])))
    if not math.isfinite(margin):
        raise ValueError('nonfinite margin')
    return margin


def _summarize_records(records, family_numbers):
    if type(records) is not list or len(records) != 4 * len(family_numbers):
        raise ValueError('the complete ordered targets are required')
    seen, cells, cases, families = set(), {cell: [] for cell in _CELL_ORDER}, [], []
    for family_index, family in enumerate(family_numbers):
        group = []
        for index, cell in enumerate(_CELL_ORDER):
            row = records[family_index * 4 + index]
            if (type(row) is not dict or set(row) != {'family', 'row_id', 'subject_number',
                    'distractor_number', 'clean', 'final_mask_sha256', 'sites'}
                    or type(row['family']) is not int or row['family'] != family
                    or (row['subject_number'], row['distractor_number']) != cell
                    or type(row['row_id']) is not str or not row['row_id'] or row['row_id'] in seen
                    or type(row['sites']) is not dict or set(row['sites']) != set(_SITES)):
                raise ValueError('invalid ordered target identity')
            seen.add(row['row_id'])
            clean = _observation(row['clean'])
            if row['final_mask_sha256'] != row['clean']['mask_sha256']:
                raise ValueError('final clean mask differs')
            r = 1 if cell[0] == 'singular' else -1
            entry = {'family': family, 'row_id': row['row_id'], 'subject_number': cell[0],
                     'distractor_number': cell[1], 'clean_margin': clean,
                     'baseline_correct': r * clean > 0, 'analysis_input': row, 'sites': {}}
            cells[cell].append(entry)
            for site in _SITES:
                observed = row['sites'][site]
                if type(observed) is not dict or set(observed) != set(_ACTIONS):
                    raise ValueError('incomplete site observations')
                margins = {action: _observation(observed[action]) for action in _ACTIONS}
                # A/same is the identity self-patch, not a matched nonzero dose.
                margins['A_same'] = margins['self']
                if (observed['self']['mask_sha256'] != row['clean']['mask_sha256']
                        or observed['self']['bits'] != row['clean']['bits']):
                    raise ValueError('self-patch mask differs')
                delta_a = math.fsum((margins['A_opposite'], -margins['self']))
                delta_b = math.fsum((margins['B_opposite'], -margins['B_same']))
                artificial = math.fsum((margins['artificial'], -clean))
                entry['sites'][site] = {'margins': margins, 'primary_effect': -r * delta_b,
                    'secondary_a_identity': -r * delta_a,
                    'previous_two_donor_average': -r * (math.fsum((delta_a, delta_b)) / 2.0),
                    'artificial_shift': artificial, 'oriented_artificial_shift': -r * artificial,
                    **{f'{prefix}_opposite_sign_reversal':
                        (clean > 0 and margins[action] < 0) or (clean < 0 and margins[action] > 0)
                        for prefix, action in (('a', 'A_opposite'), ('b', 'B_opposite'))},
                    **{f'{prefix}_opposite_correct_to_wrong': r * clean > 0 and r * margins[action] < 0
                        for prefix, action in (('a', 'A_opposite'), ('b', 'B_opposite'))},
                    **{f'{prefix}_opposite_wrong_to_correct': r * clean < 0 and r * margins[action] > 0
                        for prefix, action in (('a', 'A_opposite'), ('b', 'B_opposite'))}}
            entry['paired_difference'] = math.fsum((entry['sites']['block0']['primary_effect'],
                                                   -entry['sites']['block2']['primary_effect']))
            cases.append(entry)
            group.append(entry)
        families.append({'family': family,
            'block0': math.fsum(row['sites']['block0']['primary_effect'] for row in group) / 4.0,
            'block2': math.fsum(row['sites']['block2']['primary_effect'] for row in group) / 4.0,
            'paired': math.fsum(row['paired_difference'] for row in group) / 4.0,
            'artificial': {site: math.fsum(row['sites'][site]['oriented_artificial_shift'] for row in group) / 4.0
                           for site in _SITES}})
    count = len(families)
    block0 = math.fsum(row['block0'] for row in families) / count
    block2 = math.fsum(row['block2'] for row in families) / count
    paired = math.fsum(row['paired'] for row in families) / count
    positive = sum(row['paired'] > 0 for row in families)
    if any(not math.isfinite(value) for value in (block0, block2, paired)):
        raise ValueError('nonfinite family aggregate')
    def cell_effects(rows):
        return {'count': len(rows), 'paired_mean': math.fsum(row['paired_difference'] for row in rows) / len(rows),
                'sites': {site: {
                    'primary_mean': math.fsum(row['sites'][site]['primary_effect'] for row in rows) / len(rows),
                    'secondary_a_identity_mean': math.fsum(row['sites'][site]['secondary_a_identity'] for row in rows) / len(rows),
                    'previous_two_donor_average_mean': math.fsum(row['sites'][site]['previous_two_donor_average'] for row in rows) / len(rows),
                    'oriented_artificial_mean': math.fsum(row['sites'][site]['oriented_artificial_shift'] for row in rows) / len(rows),
                    'a_opposite_sign_reversals': sum(row['sites'][site]['a_opposite_sign_reversal'] for row in rows),
                    'b_opposite_sign_reversals': sum(row['sites'][site]['b_opposite_sign_reversal'] for row in rows),
                    **{f'{prefix}_opposite_{direction}':
                        sum(row['sites'][site][f'{prefix}_opposite_{direction}'] for row in rows)
                        for prefix in ('a', 'b') for direction in ('correct_to_wrong', 'wrong_to_correct')}}
                    for site in _SITES}}
    return {'cases': cases, 'families': families, 'block0_mean': block0, 'block2_mean': block2,
        'paired_mean': paired, 'difference_of_site_means': math.fsum((block0, -block2)),
        'positive_families': positive,
        'criterion_met': None if count != 8 else block0 > .1 and paired > .1 and positive >= 6,
        'baseline_accuracy': sum(row['baseline_correct'] for row in cases) / len(cases),
        'baseline_by_cell': {f'{subject}/{distractor}': {
            'count': len(rows), 'accuracy': sum(row['baseline_correct'] for row in rows) / len(rows),
            'mean_margin': math.fsum(row['clean_margin'] for row in rows) / len(rows)}
            for (subject, distractor), rows in cells.items()},
        'effects_by_cell': {f'{subject}/{distractor}': cell_effects(rows)
                            for (subject, distractor), rows in cells.items()},
        'oriented_artificial_mean': {site: math.fsum(row['artificial'][site] for row in families) / count
                                     for site in _SITES}}


def summarize(records):
    """Recompute the complete fixed-order 8-family/32-target summaries.

    The caller maps bound journal events to these fields. The live runner
    checks selected bits against the full vector before hashing; a closed-file
    digest alone cannot independently authenticate or reconstruct that vector.
    """
    return _summarize_records(records, tuple(range(1, 9)))


def _same(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return set(actual) == set(expected) and all(_same(actual[key], expected[key]) for key in actual)
    if type(actual) is list:
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _expect(actual, expected, label):
    if not _same(actual, expected):
        raise ValueError(label + ' differs from the frozen schedule')


def _family_rows(manifest, split, family, mask_id):
    if (type(manifest) is not dict or type(manifest.get('rows')) is not list
            or type(manifest.get('targets')) is not list or len(manifest['rows']) > 128
            or len(manifest['targets']) > 64 or type(split) is not str or not split
            or type(family) is not int or not 1 <= family <= 8
            or type(mask_id) is not int or mask_id < 0):
        raise ValueError('invalid manifest identity')
    rows = [row for row in manifest['rows'] if type(row) is dict and row.get('split') == split
            and type(row.get('family')) is int and row['family'] == family]
    targets = [target for target in manifest['targets'] if type(target) is dict and target.get('split') == split
               and type(target.get('family')) is int and target['family'] == family]
    if len(rows) != 8 or len(targets) != 4:
        raise ValueError('family must contain eight rows and four targets')
    by_id, factors = {}, {}
    for row in rows:
        try:
            row_id, source = row['row_id'], row['subject_source']
            number, distractor = row['subject_number'], row['distractor_number']
            ids, attention = row['input_ids'], row['attention_mask']
            if (type(row_id) is not str or not row_id or len(row_id) > 128 or not row_id.isascii()
                    or row_id in by_id or source not in ('A', 'B') or number not in ('singular', 'plural')
                    or distractor not in ('singular', 'plural') or type(row['subject_lemma']) is not str
                    or type(row['distractor_lemma']) is not str or type(ids) is not list or len(ids) != 10
                    or any(type(value) is not int or value < 0 for value in ids)
                    or ids[6] != mask_id or ids.count(mask_id) != 1
                    or type(attention) is not list or len(attention) != 10
                    or any(type(value) is not int or value != 1 for value in attention)
                    or (source, number, distractor) in factors):
                raise ValueError('invalid family row')
            by_id[row_id] = row
            factors[source, number, distractor] = row_id
        except (KeyError, TypeError) as error:
            raise ValueError('invalid family row') from error
    if (len(factors) != 8 or len({row['distractor_lemma'] for row in rows}) != 1
            or any(len({row['subject_lemma'] for row in rows if row['subject_source'] == source}) != 1
                   for source in ('A', 'B'))
            or {row['subject_lemma'] for row in rows if row['subject_source'] == 'A'} ==
               {row['subject_lemma'] for row in rows if row['subject_source'] == 'B'}
            or len({tuple(row['input_ids']) for row in rows}) != 8
            or len({tuple(row['input_ids'][i] for i in (0, 1, 3, 4, 6, 7, 8, 9)) for row in rows}) != 1):
        raise ValueError('family factors or template differ')
    subject_ids = {(source, number): {row['input_ids'][2] for row in rows
                   if (row['subject_source'], row['subject_number']) == (source, number)}
                   for source in ('A', 'B') for number in ('singular', 'plural')}
    distractor_ids = {number: {row['input_ids'][5] for row in rows if row['distractor_number'] == number}
                      for number in ('singular', 'plural')}
    if (any(len(values) != 1 for values in (*subject_ids.values(), *distractor_ids.values()))
            or len({next(iter(values)) for values in subject_ids.values()}) != 4
            or len({next(iter(values)) for values in distractor_ids.values()}) != 2):
        raise ValueError('family token positions differ')
    seen_cases = set()
    for target, (number, distractor) in zip(targets, _CELL_ORDER):
        try:
            case_id, row_id, mapping = target['case_id'], target['target_row_id'], target['donor_row_ids']
            opposite = 'plural' if number == 'singular' else 'singular'
            expected = {'A_same': factors['A', number, distractor],
                        'A_opposite': factors['A', opposite, distractor],
                        'B_same': factors['B', number, distractor],
                        'B_opposite': factors['B', opposite, distractor]}
            if (type(case_id) is not str or not case_id or case_id in seen_cases
                    or row_id != expected['A_same'] or not _same(mapping, expected)):
                raise ValueError('target order or donor mapping differs')
            seen_cases.add(case_id)
        except (KeyError, TypeError) as error:
            raise ValueError('invalid family target') from error
    return rows, targets, by_id


def _vector_base64(value):
    if type(value) is not str or len(value) != 4096:
        raise ValueError('invalid vector encoding')
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('invalid vector encoding') from error
    if base64.b64encode(raw).decode('ascii') != value:
        raise ValueError('noncanonical vector encoding')
    _vector(raw)
    return raw


def _float_identical(value, bits):
    return type(value) is float and struct.pack('<d', value) == struct.pack('<d', _decode(bits))


def _events_metadata(events, *, site, kind):
    path = 'distilbert.transformer.layer.' + ('0' if site == 'block0' else '2')
    names = ('before',) if kind in ('capture', 'final_clean') else ('before', 'patch', 'after')
    operations = ('capture',) if len(names) == 1 else ('capture', 'patch', 'capture')
    expected = [{'name': name, 'operation': operation, 'order': index,
        'module_path': path, 'module_aliases': [path], 'call_site_known': False,
        'forward_index': 0, 'invocation': 0, 'batch_index': 0, 'token_index': 2,
        'output_selector': [0], 'output_shape': [1, 10, 768], 'selected_shape': [768],
        'dtype': 'torch.float32', 'device': 'cpu',
        'source_capture': 'before' if kind == 'self' and index == 1 else None,
        'preceding_same_site_actions': list(names[:index])}
        for index, (name, operation) in enumerate(zip(names, operations))]
    _expect(events, expected, 'action metadata')


def audit_family_events(manifest, records, *, split, family, mask_id):
    """Independently audit one complete saved family without a model forward.

    The caller must supply an externally bound, quiescent manifest and journal.
    This checks their consistency; their origin, execution and chronology are
    not attested. Full mask logits are retained only as digests, so their hash
    and selected-logit relationship cannot be independently reconstructed.
    """
    if type(records) is not list or len(records) != 144:
        raise ValueError('the complete 144-record family journal is required')
    # Snapshot the caller's mutable structures before checking or traversing.
    # A concurrent writer is outside the trusted-quiescent-input contract.
    try:
        manifest, records = copy.deepcopy(manifest), copy.deepcopy(records)
        if len(json.dumps([manifest, records], allow_nan=False).encode('utf-8')) > 16 * 1024 * 1024:
            raise ValueError('family journal exceeds the audit bound')
    except (TypeError, OverflowError, RecursionError) as error:
        raise ValueError('family journal is not finite JSON data') from error
    rows, targets, by_id = _family_rows(manifest, split, family, mask_id)
    cursor, count = 0, 0

    def take():
        nonlocal cursor
        if cursor >= len(records):
            raise ValueError('truncated family journal')
        result = records[cursor]
        cursor += 1
        return result

    def control(name, case_id, site):
        expected = {'status': 'control', 'split': split, 'family': family,
                    'name': name, 'case_id': case_id, 'site': site, 'passed': True}
        _expect(take(), expected, 'control')

    def forward(kind, row_id, site, *, case_id=None, donor_id=None):
        nonlocal count
        count += 1
        header = {'split': split, 'family': family, 'forward_index': count,
                  'kind': kind, 'row_id': row_id, 'donor_row_id': donor_id,
                  'case_id': case_id, 'site': site,
                  'layer': 0 if site == 'block0' else 2, 'token': 2}
        _expect(take(), {'status': 'started', **header}, 'started forward')
        row = take()
        fields = {'is_bits', 'are_bits', 'is_logit', 'are_logit', 'mask_logits_sha256',
                  'before_float32_le_base64', 'after_float32_le_base64',
                  'submitted_float32_le_base64', 'events', 'capture_bytes', 'source_bytes',
                  'model_input_ids', 'model_attention_mask'}
        if type(row) is not dict or set(row) != {'status', *header, *fields}:
            raise ValueError('completed forward fields differ')
        _expect({key: row[key] for key in ('status', *header)},
                {'status': 'completed', **header}, 'completed forward')
        _expect(row['model_input_ids'], by_id[row_id]['input_ids'], 'model input')
        _expect(row['model_attention_mask'], by_id[row_id]['attention_mask'], 'attention mask')
        _events_metadata(row['events'], site=site, kind=kind)
        is_action = kind not in ('capture', 'final_clean')
        source = kind in (*_DONORS, 'artificial')
        _expect(row['capture_bytes'], 6144 if is_action else 3072, 'capture bytes')
        _expect(row['source_bytes'], 3072 if source else 0, 'source bytes')
        if (not _float_identical(row['is_logit'], row['is_bits'])
                or not _float_identical(row['are_logit'], row['are_bits'])):
            raise ValueError('logged logits differ from their float32 bits')
        observation = {'bits': [row['is_bits'], row['are_bits']],
                       'mask_sha256': row['mask_logits_sha256']}
        _observation(observation)
        before = _vector_base64(row['before_float32_le_base64'])
        after = _vector_base64(row['after_float32_le_base64']) if is_action else None
        submitted = _vector_base64(row['submitted_float32_le_base64']) if source else None
        if ((not is_action and row['after_float32_le_base64'] is not None)
                or (not source and row['submitted_float32_le_base64'] is not None)):
            raise ValueError('unexpected action tensor')
        return {'observation': observation, 'before': before, 'after': after, 'submitted': submitted}

    captures, references = {site: {} for site in _SITES}, {}
    for site in _SITES:
        for row in rows:
            row_id = row['row_id']
            item = forward('capture', row_id, site)
            captures[site][row_id] = item['before']
            if site == 'block0':
                references[row_id] = item['observation']
            else:
                _expect(item['observation'], references[row_id], 'cross-site clean observation')
                control('clean_matches_block0', row_id, site)

    normalized, witnesses = [], []
    for target in targets:
        row_id, case_id = target['target_row_id'], target['case_id']
        row = by_id[row_id]
        plan = take()
        if (type(plan) is not dict or set(plan) != {'status', 'split', 'family', 'case_id', 'row_id', 'radius', 'sites'}
                or type(plan.get('sites')) is not dict or set(plan['sites']) != set(_SITES)):
            raise ValueError('invalid dose plan')
        _expect({key: plan[key] for key in ('status', 'split', 'family', 'case_id', 'row_id')},
                {'status': 'dose_plan', 'split': split, 'family': family, 'case_id': case_id, 'row_id': row_id},
                'dose plan')
        captured = {site: {'target': captures[site][row_id],
                           **{action: captures[site][target['donor_row_ids'][action]] for action in _DONORS}}
                    for site in _SITES}
        patches, observations = {}, {}
        for site in _SITES:
            patches[site], observations[site] = {}, {}
            for action in _ACTIONS:
                donor_id = (row_id if action == 'self' else
                            target['donor_row_ids'][action] if action in _DONORS else None)
                item = forward(action, row_id, site, case_id=case_id, donor_id=donor_id)
                if action == 'self':
                    _expect(item['observation'], references[row_id], 'self-patch observation')
                    control('self_patch', case_id, site)
                observations[site][action] = item['observation']
                patches[site][action] = {'pre': item['before'], 'post': item['after'],
                                         'submitted': item['before'] if action == 'self' else item['submitted']}
        calculated = audit_witness(row_id, captured, patches)
        _expect(plan['radius'], calculated['radius'], 'dose radius')
        for site in _SITES:
            if type(plan['sites'][site]) is not dict or set(plan['sites'][site]) != {*_DONORS, 'artificial'}:
                raise ValueError('dose plan actions differ')
            for action in (*_DONORS, 'artificial'):
                detail = calculated['sites'][site][action]
                expected = {key: value for key, value in detail.items()
                            if action != 'artificial' or key not in ('distance', 'fraction')}
                expected['value_float32_le_base64'] = base64.b64encode(patches[site][action]['submitted']).decode('ascii')
                _expect(plan['sites'][site][action], expected, 'dose metrics')
        normalized.append({'family': family, 'row_id': row_id, 'subject_number': row['subject_number'],
                           'distractor_number': row['distractor_number'], 'clean': references[row_id],
                           'final_mask_sha256': None, 'sites': observations})
        witnesses.append({'case_id': case_id, 'calculated': calculated})

    for target, item in zip(targets, normalized):
        row_id, case_id = target['target_row_id'], target['case_id']
        final = forward('final_clean', row_id, 'block2', case_id=case_id)
        _expect(final['observation'], references[row_id], 'final clean observation')
        if final['before'] != captures['block2'][row_id]:
            raise ValueError('final clean activation differs')
        control('final_clean', case_id, 'block2')
        item['final_mask_sha256'] = final['observation']['mask_sha256']
    if count != 60 or cursor != len(records):
        raise ValueError('family schedule is incomplete or has extra records')
    analysis = _summarize_records(normalized, (family,))
    for case, witness in zip(analysis['cases'], witnesses):
        case['case_id'] = witness['case_id']
        case['radius'] = witness['calculated']['radius']
        for site in _SITES:
            case['sites'][site]['doses'] = {action: witness['calculated']['sites'][site][action]
                                             for action in (*_DONORS, 'artificial')}
    return {'schema': 'bert-matched-family-audit-v1', 'split': split, 'family': family,
            'forward_count': count, 'cases': analysis['cases'], 'family_means': analysis['families'][0]}


def aggregate_families(families):
    """Recompute the fixed eight-family analysis from previously audited families.

    This cannot prove that a supplied family was actually audited. The caller
    must bind each summary to a successful audit of its immutable journal.
    """
    if type(families) is not list or len(families) != 8:
        raise ValueError('all eight ordered family audits are required')
    families = copy.deepcopy(families)
    normalized, retained, seen_cases = [], [], set()
    split = None
    for index, family in enumerate(families, 1):
        if (type(family) is not dict or set(family) != {'schema', 'split', 'family', 'forward_count',
                    'cases', 'family_means'} or family['schema'] != 'bert-matched-family-audit-v1'
                or type(family['family']) is not int or family['family'] != index
                or type(family['forward_count']) is not int or family['forward_count'] != 60
                or type(family['split']) is not str or not family['split']
                or type(family['cases']) is not list or len(family['cases']) != 4):
            raise ValueError('invalid family audit')
        if split is None:
            split = family['split']
        elif split != family['split']:
            raise ValueError('family splits differ')
        for case in family['cases']:
            if (type(case) is not dict or type(case.get('case_id')) is not str
                    or not case['case_id'] or case['case_id'] in seen_cases
                    or type(case.get('radius')) is not float or not math.isfinite(case['radius'])
                    or case['radius'] < 1e-4):
                raise ValueError('invalid audited case identity or radius')
            seen_cases.add(case['case_id'])
            radius, distances = case['radius'], []
            if type(case.get('sites')) is not dict or set(case['sites']) != set(_SITES):
                raise ValueError('invalid case sites')
            for site in _SITES:
                site_result = case['sites'][site]
                doses = site_result.get('doses') if type(site_result) is dict else None
                if type(doses) is not dict or set(doses) != {*_DONORS, 'artificial'}:
                    raise ValueError('invalid case dose metrics')
                base_norm = None
                for action in (*_DONORS, 'artificial'):
                    dose = doses[action]
                    keys = {'distance', 'fraction', 'realized_norm', 'absolute_error', 'relative_error',
                            'base_norm', 'realized_over_base_norm'}
                    if type(dose) is not dict or set(dose) != keys:
                        raise ValueError('invalid dose metrics')
                    for key in ('realized_norm', 'absolute_error', 'relative_error', 'base_norm'):
                        if type(dose[key]) is not float or not math.isfinite(dose[key]) or dose[key] < 0:
                            raise ValueError('nonfinite dose metric')
                    if base_norm is None:
                        base_norm = dose['base_norm']
                    elif dose['base_norm'] != base_norm:
                        raise ValueError('base norms differ at the same site')
                    error = abs(dose['realized_norm'] - radius)
                    ratio = None if base_norm == 0 else dose['realized_norm'] / base_norm
                    _expect(dose['absolute_error'], error, 'absolute dose error')
                    _expect(dose['relative_error'], error / radius, 'relative dose error')
                    _expect(dose['realized_over_base_norm'], ratio, 'relative base norm')
                    if error > max(1e-6, 1e-5 * radius):
                        raise ValueError('dose exceeds the fixed tolerance')
                    if action == 'artificial':
                        if dose['distance'] is not None or dose['fraction'] is not None:
                            raise ValueError('artificial direction is not a donor distance')
                    else:
                        distance = dose['distance']
                        if type(distance) is not float or not math.isfinite(distance) or distance < radius:
                            raise ValueError('invalid donor distance')
                        _expect(dose['fraction'], radius / distance, 'dose fraction')
                        distances.append(distance)
            if min(distances) != radius:
                raise ValueError('radius is not the six-distance minimum')
            normalized.append(case.get('analysis_input'))
            retained.append(case)
    computed = summarize(normalized)
    for saved, expected in zip(retained, computed['cases']):
        stripped = copy.deepcopy(saved)
        del stripped['case_id'], stripped['radius']
        for site in _SITES:
            del stripped['sites'][site]['doses']
        _expect(stripped, expected, 'case summary')
    for saved, expected in zip(families, computed['families']):
        _expect(saved['family_means'], expected, 'family summary')
    computed['cases'] = retained
    return {'schema': 'bert-matched-aggregate-audit-v1', 'split': split,
            'forward_count': 480, **computed}


def _json_record(line):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON object key')
            result[key] = value
        return result

    def no_constant(_value):
        raise ValueError('nonfinite JSON number')

    try:
        record = json.loads(line.decode('utf-8'), object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ValueError('invalid journal JSON record') from error
    if type(record) is not dict:
        raise ValueError('journal records must be JSON objects')
    return record


def audit_event_journal(manifest, path, *, expected_sha256, split='evaluation',
                        family_ids=tuple(range(1, 9)), mask_id=103):
    """Audit exact closed JSONL bytes one family at a time, without a model.

    ``expected_sha256`` must come from an independently retained custody record;
    supplying a self-declared hash does not attest provenance or execution.
    Only a single family or the complete ordered eight-family study is accepted.
    """
    if (type(expected_sha256) is not str or _HEX.fullmatch(expected_sha256) is None
            or type(family_ids) is not tuple or not family_ids
            or any(type(family) is not int or not 1 <= family <= 8 for family in family_ids)
            or (len(family_ids) != 1 and family_ids != tuple(range(1, 9)))):
        raise ValueError('invalid expected journal identity')
    digest, total, record_count, current, family_results = hashlib.sha256(), 0, 0, [], []
    family_position = 0
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
            raise ValueError('journal is not a bounded regular file')
        while True:
            line = stream.readline(65537)
            if not line:
                break
            total += len(line)
            if total > 16 * 1024 * 1024 or len(line) > 65536 or not line.endswith(b'\n'):
                raise ValueError('journal byte or line bound exceeded')
            digest.update(line)
            record_count += 1
            record = _json_record(line)
            actual_family = record.get('family')
            if type(actual_family) is not int:
                raise ValueError('invalid journal family')
            if actual_family != family_ids[family_position]:
                if (len(current) != 144 or family_position + 1 >= len(family_ids)
                        or actual_family != family_ids[family_position + 1]):
                    raise ValueError('missing, extra or reordered journal family')
                family_results.append(audit_family_events(manifest, current, split=split,
                    family=family_ids[family_position], mask_id=mask_id))
                family_position += 1
                current = []
            current.append(record)
            if len(current) > 144:
                raise ValueError('family journal has extra records')
        after = os.fstat(stream.fileno())
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or total != after.st_size):
            raise ValueError('journal changed while being audited')
    if family_position != len(family_ids) - 1 or len(current) != 144:
        raise ValueError('journal is incomplete')
    if digest.hexdigest() != expected_sha256:
        raise ValueError('journal digest differs from the retained expected digest')
    family_results.append(audit_family_events(manifest, current, split=split,
        family=family_ids[family_position], mask_id=mask_id))
    aggregate = aggregate_families(family_results) if len(family_results) == 8 else None
    return {'schema': 'bert-matched-journal-audit-v1', 'sha256': digest.hexdigest(), 'size_bytes': total,
            'record_count': record_count, 'forward_count': 60 * len(family_results),
            'family_ids': list(family_ids), 'families': family_results, 'aggregate': aggregate}
