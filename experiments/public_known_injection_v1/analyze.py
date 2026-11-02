"""Model-free saved-byte analysis for the fixed public injection study."""
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

from selector import discovery_selection, fixed_random_site


SITES = (1, 2, 3, 4)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _key(case_id, kind, cell=None, candidate=None):
    return (case_id, kind, cell, candidate)


def _expected(cases):
    schedule = []
    for case in cases:
        cid = case['case_id']
        schedule.extend([(cid, 'clean', None, None), (cid, 'null-self', None, None)])
        for cell in ('A', 'B'):
            schedule.append((cid, 'injected', cell, None))
            schedule.extend((cid, 'patch', cell, candidate) for candidate in SITES)
            schedule.append((cid, 'injected-self', cell, None))
        schedule.append((cid, 'clean-repeat', None, None))
    return schedule


def _expected_plan(kind, cell, candidate, final_token):
    if kind == 'null-self':
        return [(name, op, 1, selected) for name, op, selected in
                [('before', 'capture', False), ('null', 'null', True),
                 ('self_source', 'capture', True), ('self_patch', 'patch', True),
                 ('after', 'capture', False)]]
    injection = {'A': 1, 'B': 4}[cell]
    expected = []
    for site in SITES:
        if site == injection:
            expected.extend([('inject_before', 'capture', site, False),
                             ('inject', 'add', site, True),
                             ('inject_after', 'capture', site, False)])
        if kind == 'patch' and site == candidate:
            expected.extend([('patch_before', 'capture', site, False),
                             ('patch', 'patch', site, True),
                             ('patch_after', 'capture', site, False)])
        if kind == 'injected-self' and site == injection:
            expected.extend([('self_source', 'capture', site, True),
                             ('self_before', 'capture', site, False),
                             ('self_patch', 'patch', site, True),
                             ('self_after', 'capture', site, False)])
        expected.append((f'tap_{site}', 'capture', site, True))
    return expected


def _audit_metadata(record, active, token_count, hidden, vocab_size):
    _, kind, cell, candidate = active
    hooks, shapes = record['hooks'], record['shapes']
    expected_shapes = {'logits': [vocab_size]}
    if kind in ('clean', 'clean-repeat'):
        if len(hooks) != 4:
            raise ValueError('trace event count differs')
        for sequence, site in enumerate(SITES):
            event = hooks[sequence]
            if (event.get('sequence') != sequence or event.get('layer_index') != site
                    or event.get('module_path') != f'transformer.h.{site}.mlp'
                    or event.get('module_kind') != 'mlp' or event.get('invocation') != 0
                    or event.get('forward_index') != 0 or event.get('batch_positions') != [0]
                    or event.get('token_positions') != [token_count - 1]
                    or event.get('shape') != [1, token_count, hidden]
                    or event.get('saved_shape') != [1, 1, hidden]
                    or event.get('dtype') != 'torch.float32'
                    or event.get('generation', {}).get('use_cache') is not False):
                raise ValueError('trace event metadata differs')
            expected_shapes[f'tap_{site}'] = [hidden]
    else:
        plan = _expected_plan(kind, cell, candidate, token_count - 1)
        if len(hooks) != len(plan):
            raise ValueError('plan event count differs')
        for sequence, (name, operation, site, selected) in enumerate(plan):
            event = hooks[sequence]
            if (event.get('order') != sequence or event.get('name') != name
                    or event.get('operation') != operation
                    or event.get('module_path') != f'transformer.h.{site}.mlp'
                    or event.get('invocation') != 0 or event.get('forward_index') != 0
                    or event.get('dtype') != 'torch.float32'
                    or event.get('output_shape') != [1, token_count, hidden]
                    or event.get('batch_index') != (0 if selected else None)
                    or event.get('token_index') != (token_count - 1 if selected else None)
                    or event.get('selected_shape') != ([hidden] if selected else [1, token_count, hidden])):
                raise ValueError('plan event metadata or order differs')
            if name == 'self_patch' and event.get('source_capture') != 'self_source':
                raise ValueError('self patch source differs')
            if operation == 'capture':
                expected_shapes[name] = [hidden] if selected else [1, token_count, hidden]
    if shapes != expected_shapes:
        raise ValueError('saved tensor names or shapes differ from plan')


def _expected_checks(active):
    _, kind, cell, candidate = active
    if kind == 'clean':
        return {'trace_valid': True}
    if kind == 'clean-repeat':
        return {'logits_equal_clean': True, 'vectors_equal_clean': True}
    if kind == 'null-self':
        return {'before_after_equal': True, 'logits_equal_clean': True}
    if kind == 'injected':
        return {'landing_valid': True}
    if kind == 'injected-self':
        return {'landing_valid': True, 'logits_equal_injected': True}
    site = {'A': 1, 'B': 4}[cell]
    relationship = 'upstream' if candidate < site else 'same' if candidate == site else 'downstream'
    checks = {'landing_valid': True, 'relationship': relationship}
    if candidate < site:
        checks.update({'upstream_donor_equal': True, 'upstream_logits_equal_injected': True})
    if candidate == site:
        checks['same_site_logits_equal_clean'] = True
    return checks


def load_phase(directory, cases, *, vocab_size=50257, hidden=768):
    """Read every stored tensor, verify its hash, coverage, and action order."""
    root = Path(directory)
    journal = (root / 'events.jsonl').read_bytes()
    rows = [json.loads(line) for line in journal.splitlines()]
    raw = {case['case_id']: (root / (case['case_id'] + '.f32')).read_bytes() for case in cases}
    case_by_id = {case['case_id']: case for case in cases}
    positions = {cid: 0 for cid in raw}
    expected = _expected(cases)
    completed, active, record = {}, None, None
    seen_tensors = []
    for row in rows:
        kind = row['type']
        if kind == 'start':
            if active is not None or len(completed) >= len(expected):
                raise ValueError('unexpected action start')
            details = row['details']
            active = _key(row['case_id'], row['action'], details.get('cell'), details.get('candidate'))
            if active != expected[len(completed)] or row['ordinal'] != len(completed):
                raise ValueError('action schedule differs')
            if details.get('cell') in ('A', 'B'):
                if details.get('site') != {'A': 1, 'B': 4}[details['cell']]:
                    raise ValueError('declared injection site differs')
            record, seen_tensors = {'tensors': {}, 'shapes': {}, 'hooks': None}, []
        elif kind == 'tensor':
            if (active is None or row['ordinal'] != len(completed)
                    or (row['case_id'], row['action']) != active[:2]):
                raise ValueError('tensor without action')
            descriptor = row['tensor']
            cid = active[0]
            if descriptor['file'] != cid + '.f32' or descriptor['offset'] != positions[cid]:
                raise ValueError('non-contiguous raw segment')
            start, length = descriptor['offset'], descriptor['length']
            shape = descriptor['shape']
            if descriptor['dtype'] != 'float32-le' or length != 4 * math.prod(shape):
                raise ValueError('invalid tensor descriptor')
            payload = raw[cid][start:start + length]
            if len(payload) != length or _sha(payload) != descriptor['sha256']:
                raise ValueError('raw tensor differs from its hash')
            if any(not math.isfinite(value) for (value,) in struct.iter_unpack('<f', payload)):
                raise ValueError('nonfinite saved tensor')
            name = descriptor['name']
            if name in record['tensors']:
                raise ValueError('duplicate tensor name in one action')
            record['tensors'][name] = payload
            record['shapes'][name] = shape
            positions[cid] += length
            seen_tensors.append(descriptor)
        elif kind == 'raw-recorded':
            if active is None or row['tensors'] != seen_tensors or record['hooks'] is not None:
                raise ValueError('raw-recorded descriptors do not match')
            record['hooks'] = row['hook_events']
        elif kind == 'complete':
            if active is None or record['hooks'] is None or row['ordinal'] != len(completed):
                raise ValueError('incomplete action')
            if record['shapes'].get('logits') != [vocab_size]:
                raise ValueError('logit vector shape differs')
            _audit_metadata(record, active, len(case_by_id[active[0]]['token_ids']), hidden, vocab_size)
            expected_checks = _expected_checks(active)
            if (row['checks'] != expected_checks or any(type(row['checks'][key]) is not type(value)
                    for key, value in expected_checks.items())):
                raise ValueError('required recorded control failed or is missing')
            record['reported_checks'] = row['checks']
            completed[active] = record
            active, record = None, None
        else:
            raise ValueError('unexpected journal event in a complete phase')
    if active is not None or len(completed) != len(expected):
        raise ValueError('phase is incomplete')
    if any(positions[cid] != len(data) for cid, data in raw.items()):
        raise ValueError('unrecorded bytes remain in a case stream')
    for case in cases:
        _audit_case(completed, case, hidden)
    return completed, {'events_sha256': _sha(journal), 'actions': len(completed),
        'raw_files': {cid: {'sha256': _sha(data), 'bytes': len(data)} for cid, data in raw.items()},
        'raw_bytes': sum(len(data) for data in raw.values())}


def _audit_case(records, case, hidden):
    cid, tokens = case['case_id'], len(case['token_ids'])
    def t(kind, cell=None, candidate=None):
        return records[_key(cid, kind, cell, candidate)]['tensors']
    clean, repeat, null = t('clean'), t('clean-repeat'), t('null-self')
    if clean['logits'] != repeat['logits'] or clean['logits'] != null['logits'] or null['before'] != null['after']:
        raise ValueError('clean/null/repeat control failed')
    if any(clean[f'tap_{i}'] != repeat[f'tap_{i}'] for i in SITES):
        raise ValueError('clean repeat tap differs')
    for cell, site in (('A', 1), ('B', 4)):
        inject = t('injected', cell)
        for kind, candidate in [('injected', None)] + [('patch', x) for x in SITES] + [('injected-self', None)]:
            item = t(kind, cell, candidate)
            before, after = item['inject_before'], item['inject_after']
            if len(before) != tokens * hidden * 4 or len(after) != len(before):
                raise ValueError('injection witness shape differs')
            offset = (tokens - 1) * hidden * 4
            previous = struct.unpack_from('<f', before, offset)[0]
            intended = struct.pack('<f', previous + 4.0)
            expected = bytearray(before)
            expected[offset:offset + 4] = intended
            if bytes(expected) != after or intended == before[offset:offset + 4]:
                raise ValueError('injection missing or changed an untouched bit')
            if kind == 'injected' and item[f'tap_{site}'] != after[offset:offset + hidden * 4]:
                raise ValueError('injected tap was not captured after add')
            if kind == 'patch':
                donor = clean[f'tap_{candidate}']
                replacement = bytearray(item['patch_before'])
                replacement[offset:offset + hidden * 4] = donor
                if bytes(replacement) != item['patch_after']:
                    raise ValueError('candidate landing/untouched bits differ')
                if candidate < site and (item['patch_before'][offset:offset + hidden * 4] != donor
                                         or item['logits'] != inject['logits']):
                    raise ValueError('upstream control failed')
                if candidate == site and item['logits'] != clean['logits']:
                    raise ValueError('same-site control failed')
            if kind == 'injected-self' and (item['self_before'] != item['self_after']
                                             or item['logits'] != inject['logits']):
                raise ValueError('injected self control failed')


def _margin(raw, score_ids):
    return float(struct.unpack_from('<f', raw, score_ids[0] * 4)[0]) - float(struct.unpack_from('<f', raw, score_ids[1] * 4)[0])


def _norm_difference(left, right):
    if len(left) != len(right) or len(left) % 4:
        raise ValueError('incompatible float vectors')
    left_f, right_f = struct.iter_unpack('<f', left), struct.iter_unpack('<f', right)
    return math.sqrt(math.fsum((float(a[0]) - float(b[0])) ** 2 for a, b in zip(left_f, right_f)))


def discovery(records, cases, *, score_ids=(318, 389)):
    result = {}
    for cell in ('A', 'B'):
        margins, votes, lengths = [], Counter(), {site: [] for site in SITES}
        for case in cases:
            cid = case['case_id']
            clean = records[_key(cid, 'clean')]['tensors']
            injected = records[_key(cid, 'injected', cell)]['tensors']
            patched = [records[_key(cid, 'patch', cell, site)]['tensors']['logits'] for site in SITES]
            margins.append((_margin(clean['logits'], score_ids), _margin(injected['logits'], score_ids),
                            tuple(_margin(value, score_ids) for value in patched)))
            changed = [site for site in SITES if clean[f'tap_{site}'] != injected[f'tap_{site}']]
            if changed:
                votes[min(changed)] += 1
            for site in SITES:
                lengths[site].append(_norm_difference(clean[f'tap_{site}'], injected[f'tap_{site}']))
        selected = discovery_selection(margins)
        magnitude_scores = [math.fsum(lengths[site]) / len(cases) for site in SITES]
        magnitude_ranking = sorted(SITES, key=lambda s: (-magnitude_scores[s - 1], s))
        max_votes = max(votes.values(), default=0)
        first = min((site for site in SITES if votes[site] == max_votes), default=None) if max_votes else None
        result[cell] = {'selector': selected, 'first_changed_site': first,
                        'first_changed_votes': [votes[site] for site in SITES],
                        'magnitude_site': magnitude_ranking[0] if magnitude_scores[magnitude_ranking[0] - 1] > 0 else None,
                        'magnitude_scores': magnitude_scores, 'constant_site': 1,
                        'random_site': fixed_random_site(cell), 'uniform_site_reference': 0.25}
    return result


def heldout(records, cases, frozen, *, score_ids=(318, 389)):
    result = {}
    for cell in ('A', 'B'):
        prediction = frozen[cell]['selector']['signed_predictions']
        by_site = {}
        for site in SITES:
            observations = []
            for case in cases:
                cid = case['case_id']
                clean = records[_key(cid, 'clean')]['tensors']['logits']
                inject = records[_key(cid, 'injected', cell)]['tensors']['logits']
                patch = records[_key(cid, 'patch', cell, site)]['tensors']['logits']
                effect = _margin(patch, score_ids) - _margin(inject, score_ids)
                distance_injected = _norm_difference(inject, clean)
                distance_patch = _norm_difference(patch, clean)
                observations.append({'case_id': cid, 'actual_effect': effect,
                    'prediction': prediction[site - 1], 'absolute_error': abs(prediction[site - 1] - effect),
                    'zero_absolute_error': abs(effect), 'distance_injected': distance_injected,
                    'distance_patch': distance_patch,
                    'distance_ratio': distance_patch / distance_injected if distance_injected > 1e-12 else None})
            n = len(observations)
            mae = math.fsum(x['absolute_error'] for x in observations) / n
            zero_mae = math.fsum(x['zero_absolute_error'] for x in observations) / n
            d_injected = math.fsum(x['distance_injected'] for x in observations) / n
            d_patch = math.fsum(x['distance_patch'] for x in observations) / n
            by_site[str(site)] = {'cases': observations, 'prediction_mae': mae, 'zero_mae': zero_mae,
                'prediction_success': mae < zero_mae, 'mean_distance_injected': d_injected,
                'mean_distance_patch': d_patch,
                'restoration_success': None if d_injected <= 1e-12 else d_patch < d_injected,
                'undefined_ratio_count': sum(x['distance_ratio'] is None for x in observations)}
        result[cell] = {'by_site': by_site, 'choices': {
            'selector': frozen[cell]['selector']['selected_site'],
            'first_changed': frozen[cell]['first_changed_site'],
            'magnitude': frozen[cell]['magnitude_site'],
            'constant': frozen[cell]['constant_site'], 'random': frozen[cell]['random_site']}}
    return result
