"""Independent, model-free replay of crossed-template taste study journals.

The caller must bind the manifest and result to the published source and the
separately observed process exit. No model, checkpoint or study implementation
is imported. Full untouched coordinates and patched full-logit vectors are not
retained and their corresponding owner observations cannot be independently
reconstructed here.
"""
import argparse
import base64
import hashlib
import heapq
import importlib.util
import json
import math
from pathlib import Path
import struct


def load_reference():
    path = Path(__file__).resolve().with_name('bert_taste_matched_independent.py')
    spec = importlib.util.spec_from_file_location('independent_taste_numeric', path)
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


ref = load_reference()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def require_binary64_subnormals():
    """Fail closed if this thread or its decimal parser flushes binary64 values."""
    smallest = bytes.fromhex('0100000000000000')
    value = struct.unpack('<d', smallest)[0]
    check(value > 0. and struct.pack('<d', value * 1.) == smallest
          and struct.pack('<d', float('5e-324')) == smallest,
          'auditor binary64 subnormal preservation is unavailable')


def probability_word(value):
    """Validate a nonnegative finite probability using its binary64 bits."""
    check(type(value) is float, 'invalid probability')
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    magnitude = bits & 0x7fffffffffffffff
    check((bits >> 63 == 0 or magnitude == 0) and magnitude <= 0x3ff0000000000000,
          'invalid probability')
    return magnitude


def equal(left, right, message):
    if type(left) is not type(right):
        raise ValueError(message + ': type')
    if isinstance(left, dict):
        check(set(left) == set(right), message + ': keys')
        for key in left:
            equal(left[key], right[key], message + '.' + key)
    elif isinstance(left, list):
        check(len(left) == len(right), message + ': length')
        for a, b in zip(left, right):
            equal(a, b, message)
    elif type(left) is float:
        check(math.isfinite(left) and math.isfinite(right)
              and struct.pack('<d', left) == struct.pack('<d', right), message)
    else:
        check(left == right, message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_string(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def decode(encoded, width):
    return ref.vector(encoded, width)


def distance(left, right):
    return norm(tuple(a - b for a, b in zip(left, right)))


def norm(values):
    answer = math.sqrt(math.fsum(value ** 2 for value in values))
    check(math.isfinite(answer), 'nonfinite norm')
    return answer


def signed_random(stem, template, width):
    seed = hashlib.sha256(b'bert-taste-orthogonal-v1\0' + stem.encode('utf-8')
                          + b'\0' + template.encode('ascii')).digest()
    digits = []
    for index in range((width + 255) // 256):
        block = hashlib.sha256(seed + index.to_bytes(4, 'big')).digest()
        digits.extend(1.0 if b & (1 << bit) else -1.0 for b in block for bit in range(7, -1, -1))
    return tuple(digits[:width])


def orthogonal(stem, template, axis):
    signs = signed_random(stem, template, len(axis))
    beta = math.fsum(a * b for a, b in zip(signs, axis)) / math.fsum(x ** 2 for x in axis)
    projected = tuple(math.fsum((a, -beta * b)) for a, b in zip(signs, axis))
    length = norm(projected)
    check(length >= 1e-8, 'degenerate orthogonal direction')
    return tuple(x / length for x in projected)


def reconstruct_plan(stem, captures):
    lengths = {template: {site: distance(captures[template, 'sugar', site][1],
                                         captures[template, 'lemon', site][1])
                          for site in ('ingredient', 'the')} for template in ('after', 'with')}
    radius = min(lengths[t][s] for t in ('after', 'with') for s in ('ingredient', 'the'))
    check(radius >= 1e-5, 'dose below protocol floor')
    fractions = {t: {s: radius / lengths[t][s] for s in ('ingredient', 'the')}
                 for t in ('after', 'with')}
    actions = {}
    for template in ('after', 'with'):
        sugar, lemon = (captures[template, i, 'ingredient'][1] for i in ('sugar', 'lemon'))
        axis = tuple(a - b for a, b in zip(sugar, lemon))
        orth = orthogonal(stem, template, axis)
        for ingredient, other in (('sugar', 'lemon'), ('lemon', 'sugar')):
            for site in ('ingredient', 'the'):
                base = captures[template, ingredient, site][1]
                donor = captures[template, other, site][1]
                direction = tuple(a - b for a, b in zip(donor, base))
                unit = tuple(x / lengths[template][site] for x in direction)
                pairs = [(site + '_toward', unit, 1), (site + '_away', unit, -1)]
                if site == 'ingredient':
                    pairs += [('orth_plus', orth, 1), ('orth_minus', orth, -1)]
                for label, vector, sign in pairs:
                    values = [math.fsum((a, (sign * radius) * b)) for a, b in zip(base, vector)]
                    raw = b''.join(struct.pack('<I', ref.f32_round(value)) for value in values)
                    landed = tuple(ref.f32(bits[0]) for bits in struct.iter_unpack('<I', raw))
                    delta = tuple(a - b for a, b in zip(landed, base))
                    delivered = norm(delta)
                    denominator = delivered * norm(direction)
                    check(delivered > 0 and abs(delivered - radius) <= 1e-6 + 1e-4 * radius,
                          'dose outside tolerance')
                    cosine = math.fsum(a * b for a, b in zip(delta, direction)) / denominator
                    check(abs(cosine) <= .01 if label.startswith('orth_') else sign * cosine >= .99,
                          'post-rounding cosine outside tolerance')
                    base_norm = norm(base)
                    metrics = {'requested_norm': radius, 'delivered_norm': delivered,
                               'absolute_error': abs(delivered-radius),
                               'relative_error': abs(delivered-radius) / radius,
                               'recipient_norm': base_norm,
                               'recipient_norm_status': 'zero' if base_norm == 0 else 'positive',
                               'dose_to_recipient_norm': None if base_norm == 0 else delivered / base_norm,
                               'cosine_with_donor_direction': cosine}
                    actions[template, ingredient, label] = (raw, metrics)
    return radius, lengths, fractions, actions


def read_json(path):
    return json.loads(path.read_bytes())


def verify_summary(summary, sweet_id, sour_id, vocab):
    require_binary64_subnormals()
    check(type(summary) is dict and type(summary.get('sweet_bits')) is int
          and type(summary.get('sour_bits')) is int, 'invalid selected logits')
    sweet, sour = ref.f32(summary['sweet_bits']), ref.f32(summary['sour_bits'])
    equal(summary.get('sweet_logit'), sweet, 'sweet logit')
    equal(summary.get('sour_logit'), sour, 'sour logit')
    equal(summary.get('sweet_minus_sour'), math.fsum((sweet, -sour)), 'score')
    check(hash_string(summary.get('mask_logits_sha256')), 'invalid mask-logit digest')
    ids = summary.get('top_five_ids')
    check(type(ids) is list and len(ids) == 5 and len(set(ids)) == 5
          and all(type(i) is int and 0 <= i < vocab for i in ids), 'invalid top-five IDs')
    tokens = summary.get('top_five_tokens')
    check(type(tokens) is list and len(tokens) == 5 and all(type(t) is str for t in tokens), 'invalid top-five tokens')
    p, q = summary['sweet_probability'], summary['sour_probability']
    p_word, q_word = probability_word(p), probability_word(q)
    check(p + q <= 1 + 1e-12, 'target probabilities exceed one')
    check(not (sweet > sour and p_word < q_word) and not (sour > sweet and q_word < p_word),
          'target probability order contradicts selected logits')
    if p > 1e-300 and q > 1e-300:
        check(abs(math.log(p) - math.log(q) - (sweet - sour)) < 1e-8, 'selected-logit/probability odds differ')


def verify_baseline(raw, summary, vocab, sweet_id, sour_id):
    check(len(raw) == vocab * 4 and sha(raw) == summary['mask_logits_sha256'], 'baseline full logits differ')
    bits = [p[0] for p in struct.iter_unpack('<I', raw)]
    values = [ref.f32(item) for item in bits]
    equal(summary['sweet_bits'], bits[sweet_id], 'baseline sweet bits')
    equal(summary['sour_bits'], bits[sour_id], 'baseline sour bits')
    ranking = heapq.nsmallest(5, range(vocab), key=lambda i: (-values[i], i))
    equal(summary['top_five_ids'], ranking, 'baseline ranking')
    maximum = max(values)
    expvalues = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(expvalues)
    for target, field in ((sweet_id, 'sweet_probability'), (sour_id, 'sour_probability')):
        value = expvalues[target] / denominator
        expected_word, actual_word = probability_word(value), probability_word(summary[field])
        if expected_word == 0:
            check(actual_word <= 2, 'baseline probability differs')
        elif expected_word < 0x0010000000000000:
            tolerance = max(2, math.ceil(expected_word * 1e-10))
            check(actual_word > 0 and abs(actual_word - expected_word) <= tolerance,
                  'baseline probability differs')
        else:
            check(math.isclose(summary[field], value, rel_tol=1e-10, abs_tol=0.),
                  'baseline probability differs')


def verify_metadata(events, layer, position, patched, self_patch, width):
    names = ['full_before', 'before'] + (['patch', 'after', 'full_after'] if patched else [])
    operations = ['capture', 'capture'] + (['patch', 'capture', 'capture'] if patched else [])
    check(type(events) is list and len(events) == len(names), 'action metadata count')
    for index, (event, name, operation) in enumerate(zip(events, names, operations)):
        check(type(event) is dict, 'action metadata type')
        expected = {'name': name, 'operation': operation, 'order': index, 'forward_index': 0,
                    'invocation': 0, 'module_path': f'distilbert.transformer.layer.{layer}',
                    'module_aliases': [f'distilbert.transformer.layer.{layer}'],
                    'batch_index': None if name.startswith('full_') else 0,
                    'token_index': None if name.startswith('full_') else position,
                    'output_selector': [0], 'output_shape': [1, 10, width],
                    'selected_shape': [1, 10, width] if name.startswith('full_') else [width],
                    'dtype': 'torch.float32', 'device': 'cpu',
                    'source_capture': 'before' if self_patch and name == 'patch' else None,
                    'preceding_same_site_actions': names[:index]}
        for key, value in expected.items():
            equal(event.get(key), value, 'action metadata ' + key)


def records(raw):
    check(len(raw) <= 32 * 1024 * 1024 and raw.endswith(b'\n'), 'journal size or termination')
    rows = []
    for line in raw.splitlines():
        check(len(line) <= 1024 * 1024, 'journal record too large')
        value = json.loads(line)
        check(type(value) is dict, 'invalid journal record')
        rows.append(value)
    return rows


def summarize_effects(stems, effects, eligibility):
    indexed = {(e['stem'], e['template'], e['direction'], e['mode']): e['effect'] for e in effects}
    comparisons, summaries = [], []
    for stem in stems:
        contrasts, specific = {}, {}
        for template in ('after', 'with'):
            for direction, sign in (('sugar_to_lemon', 1), ('lemon_to_sugar', -1)):
                def value(mode):
                    return indexed[stem, template, direction, mode]
                contrast = math.fsum((sign * value('ingredient_toward'), -sign * value('the_toward')))
                odd = {s: math.fsum((value(s + '_toward'), -value(s + '_away'))) / 2 for s in ('ingredient', 'the')}
                even = {s: math.fsum((value(s + '_toward'), value(s + '_away'))) / 2 for s in ('ingredient', 'the')}
                odd['orth'] = math.fsum((value('orth_plus'), -value('orth_minus'))) / 2
                even['orth'] = math.fsum((value('orth_plus'), value('orth_minus'))) / 2
                specificity = math.fsum((sign * odd['ingredient'], -abs(odd['orth'])))
                contrasts[template, direction], specific[template, direction] = contrast, specificity
                comparisons.append({'stem': stem, 'template': template, 'direction': direction,
                                    'eligible_descriptive': eligibility[stem + '-' + template],
                                    'signed_matched_position_contrast': contrast, 'odd': odd, 'even': even,
                                    'donor_odd_minus_abs_orth': specificity})
        direction_order = ('sugar_to_lemon', 'lemon_to_sugar')
        after = math.fsum(contrasts['after', d] for d in direction_order) / 2
        with_some = math.fsum(contrasts['with', d] for d in direction_order) / 2
        diff = {t: math.fsum((contrasts[t, direction_order[0]], -contrasts[t, direction_order[1]]))
                for t in ('after', 'with')}
        specific_after = math.fsum(specific['after', d] for d in direction_order) / 2
        specific_with = math.fsum(specific['with', d] for d in direction_order) / 2
        summaries.append({'stem': stem, 'after': after, 'with': with_some,
                          'template_difference': math.fsum((after, -with_some)),
                          'direction_main': math.fsum((diff['after'], diff['with'])) / 2,
                          'template_direction_interaction': math.fsum((diff['after'], -diff['with'])),
                          'direction_specificity_after': specific_after, 'direction_specificity_with': specific_with,
                          'direction_specificity_template_difference': math.fsum((specific_after, -specific_with))})
    means = {key: math.fsum(row[key] for row in summaries) / len(summaries) for key in summaries[0] if key != 'stem'}
    return comparisons, summaries, means


def audit(manifest, result, journal_by_stem, *, width=768, vocab=30522, sweet_id=4086, sour_id=14768):
    require_binary64_subnormals()
    rows = manifest['rows']
    stems = manifest['selected_stems']
    check(type(rows) is list and type(stems) is list and len(rows) == len(stems) * 4 and len(set(stems)) == len(stems),
          'invalid crossed manifest')
    check(set(journal_by_stem) == set(stems), 'journal set differs')
    baselines, eligibility, effects, dose_records = {}, {}, [], []
    by_hash, complete_count = {}, 0
    for group_index, stem in enumerate(stems):
        group = rows[4*group_index:4*group_index+4]
        check([(r['stem'], r['template'], r['ingredient']) for r in group] ==
              [(stem, 'after', 'sugar'), (stem, 'after', 'lemon'),
               (stem, 'with', 'sugar'), (stem, 'with', 'lemon')], 'crossed row order')
        check(all(type(r['input_ids']) is list and len(r['input_ids']) == 10
                  and all(type(x) is int and 0 <= x < vocab for x in r['input_ids'])
                  and type(r['attention_mask']) is list and len(r['attention_mask']) == 10
                  and all(type(x) is int and x == 1 for x in r['attention_mask']) for r in group),
              'invalid model inputs')
        journal = records(journal_by_stem[stem])
        cursor = 0
        references, captures, clean_full = {}, {}, {}

        def forward(row, layer, position, mode, donor=None, self_patch=False):
            nonlocal cursor, complete_count
            check(cursor + 1 < len(journal), 'missing forward records')
            start, complete = journal[cursor:cursor+2]
            cursor += 2
            identity = {'row_id': row['id'], 'stem': stem, 'template': row['template'],
                        'ingredient': row['ingredient'], 'layer': layer, 'position': position,
                        'mode': mode, 'donor': donor}
            for record, event in ((start, 'forward_start'), (complete, 'forward_complete')):
                equal(record.get('event'), event, 'record event')
                for key, value in identity.items():
                    equal(record.get(key), value, 'record identity ' + key)
            equal(complete.get('model_entry'), {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']},
                  'model input')
            summary = complete.get('summary')
            verify_summary(summary, sweet_id, sour_id, vocab)
            digest = summary['mask_logits_sha256']
            if digest in by_hash:
                equal(summary, by_hash[digest], 'identical-logit summary')
            else:
                by_hash[digest] = summary
            for field in ('denormal_before', 'denormal_after', 'denormal_after_scoring'):
                equal(complete.get(field), {'smallest_times_one_bits': 1, 'normal_half_bits': 0x400000}, 'denormal probe')
            patched = mode in ('ingredient_donor', 'ingredient_self', 'the_self', 'ingredient_toward',
                               'ingredient_away', 'orth_plus', 'orth_minus', 'the_toward', 'the_away', 'final_block_donor')
            verify_metadata(complete.get('events'), layer, position, patched, self_patch, width)
            before_raw, before = decode(complete.get('before_float32_le_base64'), width)
            check(hash_string(complete.get('full_before_sha256')), 'invalid full-before hash')
            if patched:
                after_raw, after = decode(complete.get('after_float32_le_base64'), width)
                check(complete.get('untouched_positions_equal') is True
                      and hash_string(complete.get('full_after_sha256')), 'missing full-output witness')
                if self_patch:
                    check(complete.get('source_float32_le_base64') is None
                          and complete.get('donor_clean_float32_le_base64') is None, 'self source mismatch')
                    equal(after_raw, before_raw, 'self landing')
                else:
                    source_raw, _ = decode(complete.get('source_float32_le_base64'), width)
                    equal(after_raw, source_raw, 'source landing')
            else:
                check(complete.get('after_float32_le_base64') is None
                      and complete.get('source_float32_le_base64') is None
                      and complete.get('donor_clean_float32_le_base64') is None
                      and complete.get('untouched_positions_equal') is None
                      and complete.get('full_after_sha256') is None, 'capture has patch evidence')
                after_raw, after = None, None
            complete_count += 1
            return complete, (before_raw, before), (after_raw, after)

        for row in group:
            for position, site in ((7, 'ingredient'), (1, 'the')):
                event, before, _ = forward(row, 3, position, 'capture_' + site)
                captures[row['template'], row['ingredient'], site] = before
                if site == 'ingredient':
                    encoded = event.get('baseline_mask_logits_float32_le_base64')
                    check(type(encoded) is str, 'missing baseline logits')
                    raw = base64.b64decode(encoded, validate=True)
                    check(base64.b64encode(raw).decode('ascii') == encoded, 'noncanonical baseline bits')
                    verify_baseline(raw, event['summary'], vocab, sweet_id, sour_id)
                    references[row['id']] = event['summary']['mask_logits_sha256']
                    clean_full[row['id']] = event['full_before_sha256']
                    baselines[row['id']] = event['summary']
                else:
                    check(event.get('baseline_mask_logits_float32_le_base64') is None, 'extra baseline logits')
                    equal(event['summary']['mask_logits_sha256'], references[row['id']], 'capture mask')
                    equal(event['full_before_sha256'], clean_full[row['id']], 'capture full output')
        for template in ('after', 'with'):
            sugar, lemon = (baselines[stem + '-' + template + '-' + i] for i in ('sugar', 'lemon'))
            targets = {sweet_id, sour_id}
            eligibility[stem + '-' + template] = (sugar['sweet_minus_sour'] > 0 and lemon['sweet_minus_sour'] < 0
                and bool(targets.intersection(sugar['top_five_ids'])) and bool(targets.intersection(lemon['top_five_ids'])))
        radius, lengths, fractions, plan = reconstruct_plan(stem, captures)
        check(cursor < len(journal), 'missing dose plan')
        record = journal[cursor]
        cursor += 1
        equal(record.get('event'), 'dose_plan', 'dose plan event')
        for key, value in {'stem': stem, 'layer': 3, 'radius': radius, 'distances': lengths, 'fractions': fractions}.items():
            equal(record.get(key), value, 'dose plan ' + key)
        for key in ('denormal_before_construction', 'denormal_after_construction'):
            equal(record.get(key), {'smallest_times_one_bits': 1, 'normal_half_bits': 0x400000}, 'dose probe')
        dose_records.append({k: v for k, v in record.items() if k != 'event'})
        for row in group:
            template, ingredient = row['template'], row['ingredient']
            other = 'lemon' if ingredient == 'sugar' else 'sugar'
            donor_id = stem + '-' + template + '-' + other
            direction = 'lemon_to_sugar' if ingredient == 'sugar' else 'sugar_to_lemon'
            modes = (('ingredient_donor', 'ingredient', False), ('ingredient_self', 'ingredient', True),
                     ('the_self', 'the', True), ('ingredient_toward', 'ingredient', False),
                     ('ingredient_away', 'ingredient', False), ('orth_plus', 'ingredient', False),
                     ('orth_minus', 'ingredient', False), ('the_toward', 'the', False), ('the_away', 'the', False))
            for mode, site, is_self in modes:
                event, before, after = forward(row, 3, 7 if site == 'ingredient' else 1, mode,
                                               row['id'] if is_self else donor_id, is_self)
                equal(before[0], captures[template, ingredient, site][0], 'pre-action capture')
                if not is_self:
                    donor, _ = decode(event.get('donor_clean_float32_le_base64'), width)
                    equal(donor, captures[template, other, site][0], 'donor clean capture')
                if is_self:
                    equal(event['summary']['mask_logits_sha256'], references[row['id']], 'self mask')
                if mode == 'ingredient_donor':
                    equal(after[0], captures[template, other, site][0], 'full donor landing')
                if mode in ('ingredient_donor', 'ingredient_self', 'the_self'):
                    check(event.get('dose') is None and event.get('actual_pre_post_norm') is None
                          and event.get('actual_donor_cosine') is None, 'unexpected dose')
                else:
                    expected_raw, metrics = plan[template, ingredient, mode]
                    equal(after[0], expected_raw, 'planned landing')
                    equal(event.get('dose'), metrics, 'dose metrics')
                    equal(event.get('actual_pre_post_norm'), metrics['delivered_norm'], 'observed dose')
                    equal(event.get('actual_donor_cosine'), metrics['cosine_with_donor_direction'], 'observed cosine')
                base = baselines[row['id']]['sweet_minus_sour']
                value = event['summary']['sweet_minus_sour']
                effects.append({'stem': stem, 'template': template, 'recipient': row['id'],
                                'donor': row['id'] if is_self else donor_id, 'direction': direction,
                                'layer': 3, 'position': 7 if site == 'ingredient' else 1, 'mode': mode,
                                'baseline_margin': base, 'patched_margin': value,
                                'effect': math.fsum((value, -base)), 'summary': event['summary']})
        for row in group:
            event, before, _ = forward(row, 3, 7, 'final_clean')
            equal(event['summary']['mask_logits_sha256'], references[row['id']], 'fresh clean mask')
            equal(before[0], captures[row['template'], row['ingredient'], 'ingredient'][0], 'fresh clean capture')
            equal(event['full_before_sha256'], clean_full[row['id']], 'fresh clean full output')
        final = {}
        for row in group:
            event, before, _ = forward(row, 5, 7, 'final_block_capture')
            equal(event['summary']['mask_logits_sha256'], references[row['id']], 'final block clean mask')
            final[row['template'], row['ingredient']] = before
        for row in group:
            template, ingredient = row['template'], row['ingredient']
            other = 'lemon' if ingredient == 'sugar' else 'sugar'
            donor_id = stem + '-' + template + '-' + other
            event, before, after = forward(row, 5, 7, 'final_block_donor', donor_id)
            equal(before[0], final[template, ingredient][0], 'final block pre-action')
            equal(after[0], final[template, other][0], 'final block landing')
            donor, _ = decode(event.get('donor_clean_float32_le_base64'), width)
            equal(donor, final[template, other][0], 'final block donor')
            equal(event['summary']['mask_logits_sha256'], references[row['id']], 'final block mask')
            base, value = baselines[row['id']]['sweet_minus_sour'], event['summary']['sweet_minus_sour']
            effects.append({'stem': stem, 'template': template, 'recipient': row['id'], 'donor': donor_id,
                            'direction': 'lemon_to_sugar' if ingredient == 'sugar' else 'sugar_to_lemon',
                            'layer': 5, 'position': 7, 'mode': 'final_block_donor', 'baseline_margin': base,
                            'patched_margin': value, 'effect': math.fsum((value, -base)), 'summary': event['summary']})
        check(cursor == len(journal), 'extra journal records')
    equal(result['completed_forwards'], 56 * len(stems), 'forward count')
    equal(complete_count, 56 * len(stems), 'independent forward count')
    equal(result['baselines'], baselines, 'baselines')
    equal(result['eligibility_descriptive'], eligibility, 'eligibility')
    equal(result['effects'], effects, 'effects')
    equal(result['doses'], dose_records, 'dose records')
    comparisons, summaries, means = summarize_effects(stems, effects, eligibility)
    equal(result['comparisons'], comparisons, 'comparisons')
    equal(result['stem_summaries'], summaries, 'stem summaries')
    equal(result['means'], means, 'means')
    require_binary64_subnormals()
    return {'status': 'saved-record-consistency-pass', 'stems': len(stems), 'forwards': complete_count,
            'matched_actions': 24 * len(stems), 'self_controls': 8 * len(stems),
            'final_block_controls': 4 * len(stems), 'eligibility_descriptive': eligibility,
            'stem_summaries': summaries, 'means': means,
            'limitations': ['Saved-record consistency does not independently attest execution.',
                'Whole-output untouched coordinates and patched full-logit vectors were not retained.',
                'The saved top-five token strings are not independently mapped to the pinned vocabulary.',
                'Recomputed baseline probabilities are checked with relative and near-underflow ULP tolerances.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    require_binary64_subnormals()
    manifest_raw = args.manifest.read_bytes()
    result_path = args.directory / 'result.json'
    result_raw = result_path.read_bytes()
    manifest, result = json.loads(manifest_raw), json.loads(result_raw)
    check(manifest.get('schema') == 'bert-taste-template-direction-public-cases-v1', 'manifest schema')
    check(result.get('status') == 'complete' and result.get('schema') == 'bert-taste-template-direction-result-v1',
          'result incomplete')
    check(len(manifest.get('selected_stems', [])) == 6, 'not six stems')
    journal = {}
    equal(set(result['journals']), {stem + '.jsonl' for stem in manifest['selected_stems']}, 'journal inventory set')
    for stem in manifest['selected_stems']:
        check(type(stem) is str and stem.isascii() and stem.isalpha(), 'unsafe stem')
        name = stem + '.jsonl'
        raw = (args.directory / name).read_bytes()
        equal(result['journals'].get(name), {'size': len(raw), 'sha256': sha(raw)}, 'journal inventory')
        journal[stem] = raw
    report = audit(manifest, result, journal)
    report['manifest_sha256'], report['result_sha256'] = sha(manifest_raw), sha(result_raw)
    report['journals'] = {name + '.jsonl': {'size': len(raw), 'sha256': sha(raw)} for name, raw in journal.items()}
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(data)
    print(sha(data))


if __name__ == '__main__':
    main()
