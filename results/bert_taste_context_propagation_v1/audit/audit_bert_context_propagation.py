"""Independent, model-free saved-record replay for taste propagation v1.

The caller must separately bind the committed source, manifest, and observed
process exit. Whole activation tensors are not saved: their untouched-position
checks and the relationship of selected slices to full-tensor hashes remain
owner-observed. This reader uses only the standard library and an independent
Fraction-based binary32 converter; it imports no study code or model.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_independent_reader', HERE / 'audit_bert_template_direction.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
ref = prior.ref
TEMPLATES = ('after', 'with')
INGREDIENTS = ('sugar', 'lemon')
SITES = ('early_ingredient', 'late_ingredient', 'late_the', 'late_tasted')
LATER = ('late_ingredient', 'late_the', 'late_tasted', 'late_mask')
POSITIONS = {'early_ingredient': 7, 'late_ingredient': 7, 'late_the': 1, 'late_tasted': 3, 'late_mask': 4}
AXES = ('lex', 'context', 'orth1', 'orth2')
MODES = tuple(axis + '_' + sign for axis in AXES for sign in ('plus', 'minus')) + ('self',)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    prior.equal(actual, expected, message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def dec(encoded, width):
    return ref.vector(encoded, width)


def subtract(a, b):
    check(len(a) == len(b), 'vector widths differ')
    return tuple(x - y for x, y in zip(a, b))


def squared(a):
    return math.fsum(x ** 2 for x in a)


def norm(a):
    answer = math.sqrt(squared(a))
    check(math.isfinite(answer), 'nonfinite norm')
    return answer


def dot(a, b):
    check(len(a) == len(b), 'dot-product widths differ')
    return math.fsum(x * y for x, y in zip(a, b))


def cosine(a, b):
    denom = norm(a) * norm(b)
    check(denom > 0. and math.isfinite(denom), 'invalid cosine denominator')
    return dot(a, b) / denom


def signs(stem, template, site, index, width):
    prefix = (b'bert-taste-context-propagation-orth-v1\0' + stem.encode('utf-8') + b'\0'
              + template.encode('ascii') + b'\0' + site.encode('ascii') + b'\0' + str(index).encode('ascii'))
    seed = hashlib.sha256(prefix).digest()
    raw = b''.join(hashlib.sha256(seed + count.to_bytes(4, 'big')).digest()
                   for count in range((width + 255) // 256))
    return tuple((1. if byte & (1 << bit) else -1.)
                 for byte in raw for bit in range(7, -1, -1))[:width]


def orthogonals(stem, template, site, direction):
    basis = []
    for index in (1, 2):
        vector = signs(stem, template, site, index, len(direction))
        coefficient = dot(vector, direction) / squared(direction)
        vector = tuple(math.fsum((x, -coefficient * d)) for x, d in zip(vector, direction))
        if basis:
            coefficient = dot(vector, basis[0]) / squared(basis[0])
            vector = tuple(math.fsum((x, -coefficient * u)) for x, u in zip(vector, basis[0]))
        magnitude = norm(vector)
        check(magnitude >= 1e-8, 'degenerate orthogonal vector')
        basis.append(tuple(value / magnitude for value in vector))
    return tuple(basis)


def reconstruct_plan(stem, captures):
    distances = {}
    for site in SITES:
        distances[site] = {
            'lex_after': norm(subtract(captures['after', 'sugar'][site], captures['after', 'lemon'][site])),
            'lex_with': norm(subtract(captures['with', 'sugar'][site], captures['with', 'lemon'][site])),
            'context_sugar': norm(subtract(captures['after', 'sugar'][site], captures['with', 'sugar'][site])),
            'context_lemon': norm(subtract(captures['after', 'lemon'][site], captures['with', 'lemon'][site]))}
    radius = min(item for entry in distances.values() for item in entry.values())
    check(radius >= 1e-5 and math.isfinite(radius), 'dose below floor')
    actions, mutual, fractions = {}, {}, {}
    for template in TEMPLATES:
        actions[template], mutual[template], fractions[template] = {}, {}, {}
        other_template = 'with' if template == 'after' else 'after'
        for ingredient in INGREDIENTS:
            actions[template][ingredient], mutual[template][ingredient], fractions[template][ingredient] = {}, {}, {}
            other_ingredient = 'lemon' if ingredient == 'sugar' else 'sugar'
            for site in SITES:
                recipient = captures[template, ingredient][site]
                lexical = subtract(captures[template, other_ingredient][site], recipient)
                context = subtract(captures[other_template, ingredient][site], recipient)
                lexical_norm, context_norm = norm(lexical), norm(context)
                check(min(lexical_norm, context_norm) >= 1e-5, 'donor distance below floor')
                axis = subtract(captures[template, 'sugar'][site], captures[template, 'lemon'][site])
                orth1, orth2 = orthogonals(stem, template, site, axis)
                units = {'lex': tuple(x / lexical_norm for x in lexical),
                         'context': tuple(x / context_norm for x in context), 'orth1': orth1, 'orth2': orth2}
                fractions[template][ingredient][site] = {'lexical': radius / lexical_norm,
                    'context': radius / context_norm, 'lexical_context_cosine': cosine(lexical, context)}
                action, deltas = {}, {}
                for axis_name, unit in units.items():
                    for suffix, sign in (('plus', 1), ('minus', -1)):
                        mode = axis_name + '_' + suffix
                        intended = tuple(math.fsum((x, (sign * radius) * u)) for x, u in zip(recipient, unit))
                        raw = b''.join(struct.pack('<I', ref.f32_round(value)) for value in intended)
                        landed = tuple(ref.f32(bits[0]) for bits in struct.iter_unpack('<I', raw))
                        delta = subtract(landed, recipient)
                        deltas[mode] = delta
                        delivered = norm(delta)
                        donor = context if axis_name == 'context' else lexical
                        donor_cosine = cosine(delta, donor)
                        check(delivered > 0 and abs(delivered - radius) <= 1e-6 + 1e-4 * radius,
                              'delivered norm outside tolerance')
                        check(abs(donor_cosine) <= .01 if axis_name.startswith('orth') else sign * donor_cosine >= .99,
                              'delivered donor cosine differs')
                        base_norm = norm(recipient)
                        action[mode] = (raw, {'requested_norm': radius, 'delivered_norm': delivered,
                            'absolute_error': abs(delivered - radius), 'relative_error': abs(delivered - radius) / radius,
                            'recipient_norm': base_norm, 'recipient_norm_status': 'zero' if base_norm == 0 else 'positive',
                            'dose_to_recipient_norm': None if base_norm == 0 else delivered / base_norm,
                            'cosine_with_donor_direction': donor_cosine, 'lexical_cosine': cosine(delta, lexical),
                            'context_cosine': cosine(delta, context)})
                all_cosines = {}
                for first in ('plus', 'minus'):
                    for second in ('plus', 'minus'):
                        key = 'orth1_' + first + '__orth2_' + second
                        value = cosine(deltas['orth1_' + first], deltas['orth2_' + second])
                        check(abs(value) <= .01, 'mutual orthogonal cosine differs')
                        all_cosines[key] = value
                actions[template][ingredient][site] = action
                mutual[template][ingredient][site] = all_cosines
    return {'radius': radius, 'distances': distances, 'fractions': fractions, 'mutual_cosines': mutual}, actions


def odd_even(plus, minus, clean):
    check(len(plus) == len(minus) == len(clean), 'odd/even widths differ')
    return (tuple(math.fsum((a, -b)) / 2 for a, b in zip(plus, minus)),
            tuple(math.fsum((a, b, -2 * c)) / 2 for a, b, c in zip(plus, minus, clean)))


def centered(values):
    mean = math.fsum(values) / len(values)
    return tuple(math.fsum((value, -mean)) for value in values)


def projection(vector, direction):
    vnorm, dnorm = norm(vector), norm(direction)
    if dnorm < 1e-12:
        return {'coefficient': None, 'coefficient_status': 'clean direction below floor',
                'cosine': None, 'cosine_status': 'clean direction below floor',
                'vector_norm': vnorm, 'direction_norm': dnorm, 'residual_norm': None}
    product = dot(vector, direction)
    coefficient = product / squared(direction)
    residual = tuple(math.fsum((x, -coefficient * d)) for x, d in zip(vector, direction))
    return {'coefficient': coefficient, 'coefficient_status': 'defined',
            'cosine': None if vnorm < 1e-12 else product / (vnorm * dnorm),
            'cosine_status': 'response below floor' if vnorm < 1e-12 else 'defined',
            'vector_norm': vnorm, 'direction_norm': dnorm, 'residual_norm': norm(residual)}


def probabilities(logits):
    maximum = max(logits)
    values = tuple(math.exp(value - maximum) for value in logits)
    total = math.fsum(values)
    return tuple(value / total for value in values)


def tv(logits, clean):
    p, q = probabilities(logits), probabilities(clean)
    return math.fsum(abs(a - b) for a, b in zip(p, q)) / 2


def specificity(lex, orth1, orth2):
    return None if any(x is None for x in (lex, orth1, orth2)) else math.fsum((lex, -math.fsum((abs(orth1), abs(orth2))) / 2))


def float64_vector(vector):
    return base64.b64encode(b''.join(struct.pack('<d', value) for value in vector)).decode('ascii')


def metadata(events, layer, position, mode, capture_late, width):
    patched, self_patch = mode not in ('clean_capture', 'fresh_clean', 'final_capture'), mode == 'self'
    names = ['full_before', 'before'] + (['patch', 'after', 'full_after'] if patched else [])
    if capture_late:
        names += ['late_full', *LATER]
    check(type(events) is list and len(events) == len(names), 'intervention metadata count')
    earlier = {}
    for order, (event, name) in enumerate(zip(events, names)):
        late = name == 'late_full' or name in LATER
        path = f'distilbert.transformer.layer.{3 if late else layer}'
        whole = name in ('full_before', 'full_after', 'late_full')
        token = None if whole else (POSITIONS[name] if late else position)
        expected = {'name': name, 'operation': 'patch' if name == 'patch' else 'capture',
                    'order': order, 'forward_index': 0, 'invocation': 0, 'module_path': path,
                    'module_aliases': [path], 'batch_index': None if whole else 0, 'token_index': token,
                    'output_selector': [0], 'output_shape': [1, 10, width],
                    'selected_shape': [1, 10, width] if whole else [width], 'dtype': 'torch.float32',
                    'device': 'cpu', 'source_capture': 'before' if self_patch and name == 'patch' else None,
                    'preceding_same_site_actions': list(earlier.get(path, []))}
        for key, value in expected.items():
            equal(event.get(key), value, 'metadata ' + key)
        earlier.setdefault(path, []).append(name)


def required_mean(values):
    missing = sum(value is None for value in values)
    return {'value': None if missing else math.fsum(values) / len(values),
            'missing_count': missing, 'reason': 'required value unavailable' if missing else None}


def aggregate(cells, stems, groups, fields):
    result = {}
    for group in groups:
        result[group] = {}
        chosen = [cell for cell in cells if cell['group'] == group]
        for field in fields:
            stem_values = {}
            for stem in stems:
                entry = {}
                for template in TEMPLATES:
                    values = [cell[field] for cell in chosen if cell['stem'] == stem and cell['template'] == template]
                    check(len(values) == 2, 'missing or duplicate recipient aggregation')
                    entry[template] = required_mean(values)
                a, b = entry['after']['value'], entry['with']['value']
                entry['difference'] = None if a is None or b is None else math.fsum((a, -b))
                stem_values[stem] = entry
            means = {t: required_mean([stem_values[s][t]['value'] for s in stems]) for t in TEMPLATES}
            means['difference'] = required_mean([stem_values[s]['difference'] for s in stems])
            result[group][field] = {'stems': stem_values, 'means': means}
    return result


def validate_rows(rows):
    check(type(rows) is list and rows and len(rows) % 4 == 0, 'expected complete four-row stems')
    stems, seen_rows, seen_inputs = [], set(), set()
    for offset in range(0, len(rows), 4):
        group = rows[offset:offset + 4]
        stem = group[0].get('stem')
        check(type(stem) is str and stem not in stems, 'invalid or duplicate stem')
        stems.append(stem)
        for row, (template, ingredient) in zip(group, ((t, i) for t in TEMPLATES for i in INGREDIENTS)):
            check(row.get('stem') == stem and row.get('template') == template and row.get('ingredient') == ingredient,
                  'crossed row order differs')
            check(row.get('id') == stem + '-' + template + '-' + ingredient and row['id'] not in seen_rows,
                  'row identity differs or repeats')
            ids, mask = row.get('input_ids'), row.get('attention_mask')
            check(type(ids) is list and len(ids) == 10 and all(type(x) is int and x >= 0 for x in ids)
                  and type(mask) is list and mask == [1] * 10 and all(type(x) is int for x in mask), 'row input differs')
            check(tuple(ids) not in seen_inputs, 'repeated full input row')
            seen_rows.add(row['id'])
            seen_inputs.add(tuple(ids))
    return stems


def audit_stem(rows, events, width, vocab, sweet_id, sour_id):
    """Replay a complete single-stem journal without importing the producer."""
    validate_rows(rows)
    check(len(rows) == 4 and type(events) is list, 'one complete stem is required')
    check(all(all(0 <= x < vocab for x in row['input_ids']) for row in rows), 'token outside vocabulary')
    stem = rows[0]['stem']
    by_pair = {(row['template'], row['ingredient']): row for row in rows}
    offset = 0
    completed = 0

    def next_event():
        nonlocal offset
        check(offset < len(events), 'journal ended early')
        record = events[offset]
        offset += 1
        check(type(record) is dict, 'invalid event')
        return record

    def take(row, layer, position, mode, donor=None, full=False, later=False, expected_before=None,
             expected_source=None, expected_dose=None):
        nonlocal completed
        identity = {'row_id': row['id'], 'stem': stem, 'template': row['template'],
                    'ingredient': row['ingredient'], 'layer': layer, 'position': position, 'mode': mode, 'donor': donor}
        start, record = next_event(), next_event()
        for kind, item in (('forward_start', start), ('forward_complete', record)):
            equal(item.get('event'), kind, 'event order')
            for key, value in identity.items():
                equal(item.get(key), value, 'forward identity ' + key)
        equal(record.get('model_entry'), {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']},
              'actual recorded model entry')
        for field in ('denormal_before', 'denormal_after', 'denormal_after_scoring'):
            equal(record.get(field), {'smallest_times_one_bits': 1, 'normal_half_bits': 0x00400000}, field)
        for field in ('binary64_before', 'binary64_after_scoring'):
            equal(record.get(field), 'binary64-subnormals-preserved', field)
        prior.verify_summary(record.get('summary'), sweet_id, sour_id, vocab)
        mask_raw, mask_values = (None, None)
        if full:
            mask_raw, mask_values = dec(record.get('mask_logits_float32_le_base64'), vocab)
            prior.verify_baseline(mask_raw, record['summary'], vocab, sweet_id, sour_id)
        else:
            check(record.get('mask_logits_float32_le_base64') is None, 'unexpected or missing full-logit retention')
        before_raw, before_values = dec(record.get('before_float32_le_base64'), width)
        if expected_before is not None:
            check(before_raw == expected_before, 'pre-action bits differ from clean capture')
        patched = mode not in ('clean_capture', 'fresh_clean', 'final_capture')
        self_patch = mode == 'self'
        check(valid_hash(record.get('full_before_sha256')), 'invalid full before hash')
        if patched:
            after_raw, _ = dec(record.get('after_float32_le_base64'), width)
            check(valid_hash(record.get('full_after_sha256')) and record.get('untouched_positions_equal') is True,
                  'patch full-tensor witness missing')
            if self_patch:
                check(record.get('source_float32_le_base64') is None and after_raw == before_raw,
                      'self landing differs')
                equal(record.get('full_after_sha256'), record['full_before_sha256'], 'self full-tensor hash')
            else:
                source_raw, _ = dec(record.get('source_float32_le_base64'), width)
                check(source_raw == expected_source and after_raw == expected_source, 'source or landing bits differ')
        else:
            check(record.get('after_float32_le_base64') is None and record.get('source_float32_le_base64') is None
                  and record.get('full_after_sha256') is None and record.get('untouched_positions_equal') is None,
                  'unpatched metadata differs')
        if expected_dose is None:
            check(record.get('dose') is None and record.get('actual_pre_post_norm') is None, 'unexpected dose')
        else:
            equal(record.get('dose'), expected_dose, 'delivered dose metadata')
            equal(record.get('actual_pre_post_norm'), expected_dose['delivered_norm'], 'actual pre/post norm')
        later_data = record.get('later_float32_le_base64')
        check(type(later_data) is dict and set(later_data) == (set(LATER) if later else set()), 'later captures differ')
        later_raw = {name: dec(value, width)[0] for name, value in later_data.items()}
        if later:
            check(valid_hash(record.get('late_full_sha256')), 'later full-tensor hash missing')
        else:
            check(record.get('late_full_sha256') is None, 'unexpected later full hash')
        metadata(record.get('events'), layer, position, mode, later, width)
        completed += 1
        return {'raw': record, 'before_raw': before_raw, 'before': before_values,
                'mask_raw': mask_raw, 'mask': mask_values, 'later_raw': later_raw,
                'later': {name: dec(value, width)[1] for name, value in later_data.items()},
                'summary': record['summary']}

    clean, baselines = {}, {}
    for row in rows:
        capture = take(row, 0, 7, 'clean_capture', full=True, later=True)
        clean[row['id']] = capture
        baselines[row['id']] = capture['summary']
    captures = {}
    for row in rows:
        item = clean[row['id']]
        captures[row['template'], row['ingredient']] = {'early_ingredient': item['before'], **item['later']}
    plan, actions = reconstruct_plan(stem, captures)
    dose = next_event()
    equal(dose.get('event'), 'dose_plan', 'dose event')
    equal(dose.get('stem'), stem, 'dose stem')
    for key, value in plan.items():
        equal(dose.get(key), value, 'plan ' + key)
    for field in ('denormal_before_construction', 'denormal_after_construction'):
        equal(dose.get(field), {'smallest_times_one_bits': 1, 'normal_half_bits': 0x00400000}, field)
    for field in ('binary64_before_construction', 'binary64_after_construction'):
        equal(dose.get(field), 'binary64-subnormals-preserved', field)
    dose_result = {k: v for k, v in dose.items() if k != 'event'}
    eligibility = {}
    for template in TEMPLATES:
        sugar, lemon = (baselines[by_pair[template, ingredient]['id']] for ingredient in INGREDIENTS)
        eligibility[stem + '-' + template] = (sugar['sweet_minus_sour'] > 0 and lemon['sweet_minus_sour'] < 0
            and bool({sweet_id, sour_id}.intersection(sugar['top_five_ids']))
            and bool({sweet_id, sour_id}.intersection(lemon['top_five_ids'])))
    effects, cells, propagation = [], [], []
    for row in rows:
        template, ingredient = row['template'], row['ingredient']
        other = 'lemon' if ingredient == 'sugar' else 'sugar'
        other_template = 'with' if template == 'after' else 'after'
        lexical, context = by_pair[template, other], by_pair[other_template, ingredient]
        recipient = clean[row['id']]
        sign = 1 if other == 'sugar' else -1
        direction = 'sugar_to_lemon' if sign == 1 else 'lemon_to_sugar'
        clean_logits, donor_logits = recipient['mask'], clean[lexical['id']]['mask']
        output_direction = centered(subtract(donor_logits, clean_logits))
        for site in SITES:
            layer, position = (0 if site == 'early_ingredient' else 3), POSITIONS[site]
            before = recipient['before_raw'] if layer == 0 else recipient['later_raw'][site]
            paired = {}
            for axis in AXES:
                forward_pair = []
                for suffix in ('plus', 'minus'):
                    mode = axis + '_' + suffix
                    raw_source, expected_dose = actions[template][ingredient][site][mode]
                    donor = context['id'] if axis == 'context' else lexical['id']
                    observed = take(row, layer, position, mode, donor=donor, full=True, later=layer == 0,
                                    expected_before=before, expected_source=raw_source, expected_dose=expected_dose)
                    margin = observed['summary']['sweet_minus_sour']
                    baseline = recipient['summary']['sweet_minus_sour']
                    delta = math.fsum((margin, -baseline))
                    effects.append({'stem': stem, 'template': template, 'recipient': row['id'],
                        'direction': direction, 'site': site, 'mode': mode, 'baseline_margin': baseline,
                        'patched_margin': margin, 'effect': delta, 'summary': observed['summary']})
                    forward_pair.append((observed, delta))
                (plus, a), (minus, b) = forward_pair
                odd, even = math.fsum((a, -b)) / 2, math.fsum((a, b)) / 2
                output_odd, _ = odd_even(plus['mask'], minus['mask'], clean_logits)
                pair = {'odd': odd, 'even': even, 'signed_odd': sign * odd,
                        'tv_plus': tv(plus['mask'], clean_logits), 'tv_minus': tv(minus['mask'], clean_logits),
                        'output_projection': projection(centered(output_odd), output_direction),
                        'binary64_before_analysis': 'binary64-subnormals-preserved',
                        'binary64_after_analysis': 'binary64-subnormals-preserved'}
                if layer == 0:
                    pair['later'] = {}
                    for name in LATER:
                        odd_vec, even_vec = odd_even(plus['later'][name], minus['later'][name], recipient['later'][name])
                        clean_direction = subtract(clean[lexical['id']]['later'][name], recipient['later'][name])
                        pair['later'][name] = {'odd_float64_le_base64': float64_vector(odd_vec),
                            'even_float64_le_base64': float64_vector(even_vec),
                            'projection': projection(odd_vec, clean_direction)}
                paired[axis] = pair
            self_observed = take(row, layer, position, 'self', donor=row['id'], later=layer == 0, expected_before=before)
            equal(self_observed['summary'], recipient['summary'], 'self clean summary')
            if layer == 0:
                equal(self_observed['later_raw'], recipient['later_raw'], 'self later capture bits')
            effects.append({'stem': stem, 'template': template, 'recipient': row['id'], 'direction': direction,
                'site': site, 'mode': 'self', 'baseline_margin': recipient['summary']['sweet_minus_sour'],
                'patched_margin': self_observed['summary']['sweet_minus_sour'], 'effect': 0., 'summary': self_observed['summary']})
            cell = {'stem': stem, 'template': template, 'recipient': row['id'], 'direction': direction, 'group': site,
                    'eligible_descriptive': eligibility[stem + '-' + template], 'axes': paired}
            cell['taste_specificity'] = specificity(paired['lex']['signed_odd'], paired['orth1']['odd'], paired['orth2']['odd'])
            cell['context_specificity'] = math.fsum((paired['lex']['signed_odd'], -abs(paired['context']['odd'])))
            for axis in AXES:
                value, output = paired[axis], paired[axis]['output_projection']
                cell[axis + '_signed_odd'] = value['signed_odd']
                cell[axis + '_output_projection'] = output['coefficient']
                cell[axis + '_output_cosine'] = output['cosine']
                cell[axis + '_output_norm'] = output['vector_norm']
                cell[axis + '_output_residual'] = output['residual_norm']
                cell[axis + '_tv_plus'], cell[axis + '_tv_minus'] = value['tv_plus'], value['tv_minus']
            cells.append(cell)
            if layer == 0:
                for late in LATER:
                    item = {'stem': stem, 'template': template, 'recipient': row['id'], 'direction': direction, 'group': late}
                    for axis in AXES:
                        output = paired[axis]['later'][late]['projection']
                        item[axis + '_projection'], item[axis + '_cosine'], item[axis + '_norm'] = (
                            output['coefficient'], output['cosine'], output['vector_norm'])
                    item['propagation_specificity'] = specificity(item['lex_projection'], item['orth1_projection'], item['orth2_projection'])
                    propagation.append(item)
    for row in rows:
        fresh = take(row, 0, 7, 'fresh_clean', later=True)
        reference = clean[row['id']]
        equal(fresh['summary'], reference['summary'], 'fresh clean summary')
        check(fresh['before_raw'] == reference['before_raw'], 'fresh early capture differs')
        equal(fresh['later_raw'], reference['later_raw'], 'fresh later captures')
    final = {}
    for row in rows:
        result = take(row, 5, 7, 'final_capture')
        equal(result['summary'], clean[row['id']]['summary'], 'final clean summary')
        final[row['id']] = result['before_raw']
    for row in rows:
        other = 'lemon' if row['ingredient'] == 'sugar' else 'sugar'
        donor = by_pair[row['template'], other]
        result = take(row, 5, 7, 'final_donor', donor=donor['id'], expected_before=final[row['id']], expected_source=final[donor['id']])
        equal(result['summary'], clean[row['id']]['summary'], 'final donor clean summary')
    check(completed == 160 and offset == len(events), 'journal schedule incomplete or has extra events')
    return {'baselines': baselines, 'eligibility_descriptive': eligibility, 'doses': [dose_result],
            'effects': effects, 'comparisons': cells, 'propagation': propagation, 'completed_forwards': completed}


def audit(rows, journals, result, width, vocab, sweet_id, sour_id):
    prior.require_binary64_subnormals()
    for field in ('binary64_before_aggregation', 'binary64_after_aggregation'):
        equal(result.get(field), 'binary64-subnormals-preserved', field)
    stems = validate_rows(rows)
    check(set(journals) == set(stems), 'journal/stem set differs')
    collected = {'baselines': {}, 'eligibility_descriptive': {}, 'doses': [], 'effects': [],
                 'comparisons': [], 'propagation': [], 'completed_forwards': 0}
    for index, stem in enumerate(stems):
        part = audit_stem(rows[index * 4:index * 4 + 4], journals[stem], width, vocab, sweet_id, sour_id)
        for key in ('baselines', 'eligibility_descriptive'):
            collected[key].update(part[key])
        for key in ('doses', 'effects', 'comparisons', 'propagation'):
            collected[key].extend(part[key])
        collected['completed_forwards'] += part['completed_forwards']
    cell_fields = ['taste_specificity', 'context_specificity']
    for axis in AXES:
        cell_fields.extend(axis + suffix for suffix in ('_signed_odd', '_output_projection', '_output_cosine',
                                                        '_output_norm', '_output_residual', '_tv_plus', '_tv_minus'))
    prop_fields = ['propagation_specificity']
    for axis in AXES:
        prop_fields.extend(axis + suffix for suffix in ('_projection', '_cosine', '_norm'))
    collected['taste_aggregates'] = aggregate(collected['comparisons'], stems, SITES, cell_fields)
    collected['propagation_aggregates'] = aggregate(collected['propagation'], stems, LATER, prop_fields)
    for key, value in collected.items():
        equal(result.get(key), value, 'saved result ' + key)
    prior.require_binary64_subnormals()
    return {'status': 'pass', 'stems': stems, 'rows': len(rows), 'forwards': collected['completed_forwards'],
            'landings': len(stems) * 128, 'self_controls': len(stems) * 16,
            'fresh_controls': len(stems) * 4, 'final_block_controls': len(stems) * 4,
            'retained_full_logit_vectors': len(stems) * 132,
            'taste_aggregates': collected['taste_aggregates'], 'propagation_aggregates': collected['propagation_aggregates'],
            'scope': ['Saved-record arithmetic only; source, manifest and process-exit binding require separate review.',
                      'Selected captures do not independently verify full-tensor or untouched-coordinate witnesses.',
                      'Top-token string mapping requires an independently pinned tokenizer.',
                      'Source and process records are local observations, not execution attestation.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, help='fabricated JSON containing rows, events and result')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Test the auditor's own thread before decimal JSON values are parsed.
    prior.require_binary64_subnormals()
    if args.fixture:
        check(args.manifest is None and args.run_dir is None, 'choose fixture or closed-run mode')
        data = json.loads(args.fixture.read_bytes())
        rows, result = data['rows'], data['result']
        stems = validate_rows(rows)
        check(len(stems) == 1, 'fixture needs one stem')
        report = audit(rows, {stems[0]: data['events']}, result, 16, 64, 1, 2)
        report['input_sha256'] = sha(args.fixture.read_bytes())
        report['fixture'] = 'untrained/fabricated; not a scientific study result'
    else:
        check(args.manifest is not None and args.run_dir is not None, 'manifest and closed-run directory required')
        manifest_raw = args.manifest.read_bytes()
        manifest = json.loads(manifest_raw)
        rows = manifest['rows']
        stems = validate_rows(rows)
        check(len(stems) == 6 and manifest.get('selected_stems') == stems, 'six selected stems are required')
        result_raw = (args.run_dir / 'result.json').read_bytes()
        result = json.loads(result_raw)
        check(result.get('status') == 'complete' and result.get('schema') == 'bert-taste-context-propagation-result-v1',
              'result is not complete')
        journals, hashes = {}, {}
        check(set(result['journals']) == {stem + '.jsonl' for stem in stems}, 'journal inventory differs')
        for stem in stems:
            name = stem + '.jsonl'
            raw = (args.run_dir / name).read_bytes()
            check(0 < len(raw) <= 64 * 1024 * 1024 and raw.endswith(b'\n'), 'journal length/termination differs')
            equal(result['journals'][name], {'size': len(raw), 'sha256': sha(raw)}, 'journal inventory ' + name)
            check(all(len(line) <= 1024 * 1024 for line in raw.splitlines()), 'journal line too large')
            journals[stem] = [json.loads(line) for line in raw.splitlines()]
            hashes[name] = sha(raw)
        report = audit(rows, journals, result, 768, 30522, 4086, 14768)
        report.update({'manifest_sha256': sha(manifest_raw), 'result_sha256': sha(result_raw), 'journal_sha256': hashes})
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'forwards': report['forwards'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
