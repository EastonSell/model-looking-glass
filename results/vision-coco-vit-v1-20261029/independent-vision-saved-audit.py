"""Independent stdlib replay of bounded saved ViT/COCO intervention evidence.

The caller supplies the frozen case/donor manifests and a quiescent data
directory. This checks saved-byte consistency, not the authenticity of the
model, input images, source, supervision, publication timing, or execution.
Synthetic-only fixtures may use --synthetic; real splits require all fixed IDs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

STATE_SHAPE = [1, 197, 192]
STATE_BYTES = 197 * 192 * 4
TOKEN_BYTES = 192 * 4
EARLY = 'vit.encoder.layer.2'
LATE = 'vit.encoder.layer.11'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, limit=4_000_000):
    raw = path.read_bytes()
    require(0 < len(raw) <= limit, f'invalid JSON length: {path.name}')
    def unique_pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    return raw, value


def typed_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == \
           json.dumps(right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def close(a, b, message):
    require(type(a) in (int, float) and type(b) in (int, float) and not isinstance(a, bool)
            and not isinstance(b, bool) and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), message)


def decode_word(word):
    exponent, fraction = (word >> 23) & 255, word & 0x7fffff
    require(exponent != 255, 'nonfinite binary32')
    mantissa = fraction if exponent == 0 else (1 << 23) | fraction
    value = math.ldexp(float(mantissa), -149 if exponent == 0 else exponent - 150)
    return -value if word & 0x80000000 else value


def values(raw):
    require(len(raw) % 4 == 0, 'unaligned binary32 payload')
    return [decode_word(word[0]) for word in struct.iter_unpack('<I', raw)]


def _shift_round_even(number, shift):
    if shift >= 0:
        return number << shift
    divisor = 1 << (-shift)
    quotient, remainder = divmod(number, divisor)
    return quotient + (2 * remainder > divisor or (2 * remainder == divisor and quotient % 2 == 1))


def round_f32(value):
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    sign = (bits >> 32) & 0x80000000
    exponent, fraction = (bits >> 52) & 0x7ff, bits & ((1 << 52) - 1)
    require(exponent != 0x7ff, 'nonfinite planned value')
    mantissa = fraction if exponent == 0 else (1 << 52) | fraction
    if not mantissa:
        return sign
    power = -1074 if exponent == 0 else exponent - 1075
    high = mantissa.bit_length() - 1 + power
    if high < -126:
        return sign | _shift_round_even(mantissa, power + 149)
    significand = _shift_round_even(mantissa, power + 23 - high)
    if significand == (1 << 24):
        significand >>= 1
        high += 1
    require(high <= 127, 'planned binary32 overflows')
    return sign | ((high + 127) << 23) | (significand - (1 << 23))


def encode(values_):
    return b''.join(struct.pack('<I', round_f32(v)) for v in values_)


def token(state, index):
    require(len(state) == STATE_BYTES and type(index) is int and 1 <= index < 197, 'invalid state/token')
    return state[index * TOKEN_BYTES:(index + 1) * TOKEN_BYTES]


def block(state, tokens):
    require(type(tokens) is list and len(tokens) == len(set(tokens)) == 4, 'invalid token block')
    return b''.join(token(state, i) for i in tokens)


def distance(a, b):
    left, right = values(a), values(b)
    require(len(left) == len(right), 'distance shape mismatch')
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(left, right)))


def planned(own, donor, q, distance_):
    x, y = values(own), values(donor)
    require(len(x) == len(y) == 768 and distance_ > 0, 'invalid planned donor block')
    ratio = q / distance_
    return encode([a + ratio * (b - a) for a, b in zip(x, y)])


def random_planned(own, q, pair_id, side):
    x = values(own)
    require(len(x) == 768, 'invalid random block')
    scale = q / math.sqrt(768)
    signs = [1 if hashlib.sha256(f'vision-coco-random-v1:{pair_id}:{side}:{i}'.encode('ascii')).digest()[0] & 1
             else -1 for i in range(768)]
    return encode([a + scale * sign for a, sign in zip(x, signs)])


def summarize(raw, reference=None):
    z = values(raw)
    require(len(z) == 1000, 'invalid logit vector')
    dog, cat = z[151:269], z[281:286]
    off = z[:151] + z[269:281] + z[286:]
    def shifted(v):
        m = max(v)
        return m, math.fsum(math.exp(x - m) for x in v)
    dm, ds = shifted(dog)
    cm, cs = shifted(cat)
    om, os = shifted(off)
    tm, ts = shifted(z)
    conditional = [math.exp(x - om) / os for x in off]
    score = {'margin': (dm - cm) + math.log(ds) - math.log(cs),
             'dog_mass': math.exp(dm - tm) * ds / ts,
             'cat_mass': math.exp(cm - tm) * cs / ts,
             'off_mass': math.exp(om - tm) * os / ts,
             'top1_id': max(range(1000), key=lambda i: z[i])}
    if reference is not None:
        require(len(reference) == len(conditional), 'off-group reference length mismatch')
        score['off_conditional_tv'] = .5 * math.fsum(abs(x - y) for x, y in zip(conditional, reference))
    return score, conditional


def check_score(stored, computed):
    for key, expected in computed.items():
        if key == 'top1_id':
            require(type(stored.get(key)) is int and stored[key] == expected, 'top-1 mismatch')
        else:
            close(stored.get(key), expected, f'score mismatch: {key}')


def metadata_expected(row, path, tokens, action):
    events = row['metadata']['steps']
    if path is None:
        desired = [('early', 'capture', EARLY, None), ('late', 'capture', LATE, None)]
    else:
        desired = [('before_all', 'capture', path, None)]
        if tokens is None:
            desired.append(('action', action, path, None))
        else:
            for i, t in enumerate(tokens):
                desired.extend([(f'before_{i}', 'capture', path, t), (f'action_{i}', action, path, t),
                                (f'after_{i}', 'capture', path, t)])
        desired.append(('after_all', 'capture', path, None))
    require(type(events) is list and len(events) == len(desired), 'metadata action count mismatch')
    for order, (event, (name, operation, module, index)) in enumerate(zip(events, desired)):
        require(event.get('name') == name and event.get('operation') == operation and
                event.get('module_path') == module and typed_equal(event.get('order'), order) and
                typed_equal(event.get('invocation'), 0) and typed_equal(event.get('forward_index'), 0) and
                typed_equal(event.get('batch_index'), 0 if index is not None else None) and
                typed_equal(event.get('token_index'), index) and typed_equal(event.get('output_shape'), STATE_SHAPE) and
                typed_equal(event.get('selected_shape'), [192] if index is not None else STATE_SHAPE) and
                event.get('dtype') == 'torch.float32' and event.get('output_selector') == [] and
                module in event.get('module_aliases', []), 'metadata action identity/shape mismatch')


class Records:
    def __init__(self, root, count):
        self.root, self.count, self.next = root, count, 1
        self.hashes = []

    def take(self, label, path=None, tokens=None, action='capture'):
        index = self.next
        require(index <= self.count, 'missing forward')
        raw_json, row = read_json(self.root / f'forward-{index:03d}.json', 200_000)
        require(typed_equal(row.get('submission'), index) and row.get('label') == label and
                row.get('file') == f'forward-{index:03d}.bin' and
                row.get('dtype') == 'little-endian-float32', 'forward order/identity mismatch')
        packed = (self.root / row['file']).read_bytes()
        require(len(packed) <= 1_000_000 and sha(packed) == row.get('sha256'), 'forward bytes/hash mismatch')
        layout = row.get('layout')
        require(type(layout) is dict, 'missing tensor layout')
        ordered = sorted(layout.items(), key=lambda item: item[1]['offset'])
        offset, tensors = 0, {}
        for name, entry in ordered:
            shape, start, size = entry.get('shape'), entry.get('offset'), entry.get('bytes')
            require(type(name) is str and type(shape) is list and all(type(x) is int and x > 0 for x in shape)
                    and type(start) is int and start == offset and type(size) is int
                    and size == 4 * math.prod(shape) and start + size <= len(packed), 'invalid tensor layout')
            tensors[name] = packed[start:start + size]
            offset += size
        require(offset == len(packed) and layout.get('logits', {}).get('shape') == [1, 1000], 'payload coverage mismatch')
        metadata_expected(row, path, tokens, action)
        self.hashes.append({'submission': index, 'json_sha256': sha(raw_json), 'binary_sha256': sha(packed)})
        self.next += 1
        return row, tensors


def expect_shapes(row, tensors, shapes):
    require(set(tensors) == set(shapes), 'unexpected saved tensor set')
    for name, shape in shapes.items():
        require(row['layout'][name]['shape'] == shape, f'tensor shape mismatch: {name}')


def checked_score(row, tensors, reference=None):
    computed, conditional = summarize(tensors['logits'], reference)
    check_score(row['score'], computed)
    return computed, conditional


def initial(records, pair_id, side):
    row, tensors = records.take(f'{pair_id}:{side}:initial')
    expect_shapes(row, tensors, {'logits': [1, 1000], 'early': STATE_SHAPE, 'late': STATE_SHAPE})
    for name in ('early', 'late'):
        require(all((word[0] >> 23) & 255 != 255 for word in struct.iter_unpack('<I', tensors[name])),
                f'nonfinite initial {name} state')
    score_, conditional = checked_score(row, tensors)
    return {'logits': tensors['logits'], 'early': tensors['early'], 'late': tensors['late'],
            'score': score_, 'conditional': conditional}


def clean_forward(records, pair_id, side, name, baseline):
    row, tensors = records.take(f'{pair_id}:{side}:{name}')
    expect_shapes(row, tensors, {'logits': [1, 1000], 'early': STATE_SHAPE, 'late': STATE_SHAPE})
    require(all(tensors[k] == baseline[k] for k in ('logits', 'early', 'late')), f'{name} clean mismatch')
    score_, _ = checked_score(row, tensors, baseline['conditional'])
    return score_


def intervention(records, pair_id, side, name, path, tokens, before_state, expected_replacement,
                 baseline, expected_logits=None):
    action = 'null' if name.startswith('null_') else 'patch'
    row, tensors = records.take(f'{pair_id}:{side}:{name}', path, tokens, action)
    shapes = {'logits': [1, 1000], 'before_all': STATE_SHAPE, 'after_all': STATE_SHAPE}
    if tokens is not None:
        shapes.update({f'{kind}_{i}': [192] for i in range(4) for kind in ('before', 'after')})
    if action == 'patch':
        shapes['requested'] = STATE_SHAPE if tokens is None else [4, 192]
    expect_shapes(row, tensors, shapes)
    require(tensors['before_all'] == before_state, 'same-forward pre-state mismatch')
    if action == 'patch':
        require(tensors['requested'] == expected_replacement, f'planned replacement mismatch: {name}')
    if tokens is None:
        require(tensors['after_all'] == (before_state if action == 'null' else expected_replacement),
                'full-state landing mismatch')
    else:
        replacement = block(before_state, tokens) if action == 'null' else expected_replacement
        require(len(replacement) == 4 * TOKEN_BYTES, 'replacement length mismatch')
        expected_after = bytearray(before_state)
        for i, index in enumerate(tokens):
            prior = token(before_state, index)
            landed = replacement[i * TOKEN_BYTES:(i + 1) * TOKEN_BYTES]
            require(tensors[f'before_{i}'] == prior and tensors[f'after_{i}'] == landed,
                    'per-token capture/landing mismatch')
            expected_after[index * TOKEN_BYTES:(index + 1) * TOKEN_BYTES] = landed
        require(tensors['after_all'] == expected_after, 'untouched token or landing mismatch')
    if expected_logits is not None:
        require(tensors['logits'] == expected_logits, f'{name} logit control mismatch')
    score_, _ = checked_score(row, tensors, baseline['conditional'])
    return score_, tensors


def status_for(cat, dog):
    gap = dog['margin'] - cat['margin']
    reasons = []
    if cat['margin'] >= 0:
        reasons.append('cat-relative-margin')
    if dog['margin'] <= 0:
        reasons.append('dog-relative-margin')
    if gap < 1:
        reasons.append('gap-below-one')
    if cat['cat_mass'] < .2:
        reasons.append('cat-mass-below-0.2')
    if dog['dog_mass'] < .2:
        reasons.append('dog-mass-below-0.2')
    return {'eligible': not reasons, 'reasons': reasons, 'gap': gap}


def prepare(own, other, inside, outside, pair_id, side, same=None):
    inside_own, outside_own = block(own, inside), block(own, outside)
    inside_d = distance(inside_own, block(other, inside))
    outside_d = distance(outside_own, block(other, outside))
    result = {'inside_distance': inside_d, 'outside_distance': outside_d,
              'status': 'dose-ineligible', 'same_species_status': 'geometry-unavailable'}
    if min(inside_d, outside_d) <= 1e-6:
        return result, {}
    q = min(inside_d, outside_d)
    tolerance = min(max(1e-5, 1e-5 * q), .01 * q)
    requested = {'inside': planned(inside_own, block(other, inside), q, inside_d),
                 'outside': planned(outside_own, block(other, outside), q, outside_d),
                 'random': random_planned(inside_own, q, pair_id, side)}
    norms = {name: distance(value, outside_own if name == 'outside' else inside_own)
             for name, value in requested.items()}
    result.update(q=q, tolerance=tolerance, planned_norms=norms)
    if (any(norm <= 0 or abs(norm - q) > tolerance for norm in norms.values()) or
            any(abs(norms[key] - norms['inside']) > .01 * norms['inside'] for key in ('outside', 'random'))):
        result['status'] = 'dose-unmatchable'
        return result, requested
    result['status'] = 'matchable'
    if same is not None:
        same_d = distance(inside_own, block(same, inside))
        result['same_species_distance'] = same_d
        if same_d <= 1e-6 or same_d < q:
            result['same_species_status'] = 'distance-too-small'
        else:
            same_value = planned(inside_own, block(same, inside), q, same_d)
            same_norm = distance(inside_own, same_value)
            result.update(same_species_scale=q / same_d, same_species_planned_norm=same_norm)
            if (same_norm <= 0 or abs(same_norm - q) > tolerance or
                    abs(same_norm - norms['inside']) > .01 * norms['inside']):
                result['same_species_status'] = 'rounding-unmatchable'
            else:
                requested['same_species'] = same_value
                result['same_species_status'] = 'available'
    return result, requested


def compare_dose(stored, computed):
    require(set(stored) == set(computed), 'dose fields mismatch')
    for key, value in computed.items():
        if type(value) is dict:
            require(type(stored[key]) is dict and set(stored[key]) == set(value), 'planned dose fields mismatch')
            for name, number in value.items():
                close(stored[key][name], number, f'planned dose mismatch: {name}')
        elif type(value) is str:
            require(stored[key] == value, f'dose status mismatch: {key}')
        else:
            close(stored[key], value, f'dose value mismatch: {key}')


def classify_pair(base, dose_statuses, margins, baseline_margins):
    if not base['eligible']:
        return 'baseline-ineligible', None
    if any(value != 'matchable' for value in dose_statuses.values()):
        return ('dose-ineligible' if 'dose-ineligible' in dose_statuses.values() else 'dose-unmatchable'), None
    movements = {}
    for side in ('cat', 'dog'):
        sign = 1 if side == 'cat' else -1
        movements[side] = {site: sign * (margins[side]['matched_' + site] - baseline_margins[side])
                           for site in ('inside', 'outside')}
    passed = all(movements[side]['inside'] >= .02 * base['gap'] and
                 movements[side]['inside'] >= abs(movements[side]['outside']) + .01 * base['gap']
                 for side in ('cat', 'dog'))
    return ('positive' if passed else 'negative'), movements


def heldout_aggregate(pairs):
    expected_ids = {f'pair-{i:02d}' for i in range(4, 12)}
    valid = {'positive', 'negative', 'baseline-ineligible', 'dose-ineligible', 'dose-unmatchable'}
    require(len(pairs) == 8 and {p['pair_id'] for p in pairs} == expected_ids and
            all(p['status'] in valid for p in pairs), 'invalid held-out aggregate inputs')
    evaluable = [p for p in pairs if p['status'] in ('positive', 'negative')]
    positives = sum(p['status'] == 'positive' for p in evaluable)
    return {'status': 'inconclusive' if len(evaluable) < 4 else
            ('positive' if positives * 4 >= len(evaluable) * 3 else 'negative'),
            'selected': len(pairs), 'evaluable': len(evaluable), 'positive': positives}


def compare_condition(saved, score_, name, side, baseline, norm=None, per_token=None, receiver_norm=None, q=None):
    check_score(saved, score_)
    change = score_['margin'] - baseline['score']['margin']
    close(saved.get('raw_margin_change'), change, f'{name} raw margin change mismatch')
    close(saved.get('annotation_oriented_change'), change * (1 if side == 'cat' else -1),
          f'{name} oriented margin change mismatch')
    if norm is not None:
        close(saved.get('actual_dose'), norm, f'{name} actual dose mismatch')
        require(type(saved.get('token_doses')) is list and len(saved['token_doses']) == 4, 'token dose count mismatch')
        for actual, expected in zip(saved['token_doses'], per_token):
            close(actual, expected, 'token dose mismatch')
        close(saved.get('receiver_norm'), receiver_norm, 'receiver norm mismatch')
        if receiver_norm:
            close(saved.get('q_over_receiver_norm'), q / receiver_norm, 'relative dose mismatch')
        else:
            require(saved.get('q_over_receiver_norm') is None, 'zero receiver ratio mismatch')


def audit(manifest_path, donors_path, data, split, synthetic=False):
    manifest_raw, manifest = read_json(manifest_path, 5_000_000)
    donors_raw, donors = read_json(donors_path, 5_000_000)
    terminal_raw, terminal = read_json(data / 'terminal.json', 5_000_000)
    require(terminal.get('status') == 'complete' and terminal.get('split') == split and
            terminal.get('synthetic_rehearsal') is synthetic and not (data / 'failure.json').exists(),
            'terminal is incomplete or wrong scope')
    pairs = [p for p in manifest['pairs'] if p.get('split') == split]
    ids = [p.get('pair_id') for p in pairs]
    require(ids and len(ids) == len(set(ids)), 'pair IDs missing or repeated')
    if synthetic:
        require(manifest.get('schema') == 'fabricated-only' and split == 'development', 'synthetic scope mismatch')
    else:
        require(manifest.get('schema') == 'vision-coco-vit-spatial-cases-v1' and
                ids == [f'pair-{i:02d}' for i in (range(4) if split == 'development' else range(4, 12))],
                'real split is incomplete or out of order')
    image_ids = [p[s]['image_id'] for p in pairs for s in ('cat', 'dog')]
    require(len(image_ids) == len(set(image_ids)), 'image IDs repeat')
    by_pair = {p['pair_id']: p for p in pairs}
    for pair in pairs:
        a, b = pair['box_tokens'], pair['control_tokens']
        require(type(a) is list and type(b) is list and len(a) == len(set(a)) == len(b) == len(set(b)) == 4 and
                not set(a).intersection(b) and all(type(x) is int and 1 <= x < 197 for x in a + b), 'bad token sites')
    relevant_donors = [r for r in donors['rows'] if r['pair_id'] in by_pair]
    donor_rows = {(r['pair_id'], r['receiver_side']): r for r in relevant_donors}
    require(len(relevant_donors) == len(donor_rows) == 2 * len(pairs), 'missing or duplicate donor rows')
    for pair in pairs:
        for side in ('cat', 'dog'):
            row = donor_rows[(pair['pair_id'], side)]
            require(row['split'] == split and row['receiver_image_id'] == pair[side]['image_id'], 'donor receiver mismatch')
            if row['donor'] is not None:
                d = row['donor']
                require(d['donor_pair_id'] in by_pair and d['donor_pair_id'] != pair['pair_id'] and
                        by_pair[d['donor_pair_id']][side]['image_id'] == d['donor_image_id'], 'donor identity/split mismatch')
    count = terminal.get('submissions')
    recorded_forwards = terminal.get('saved_forwards', count if synthetic else None)
    require(type(count) is int and 0 < count <= 30 * len(pairs) and recorded_forwards == count,
            'terminal forward count mismatch')
    expected_files = {f'forward-{i:03d}.{suffix}' for i in range(1, count + 1) for suffix in ('bin', 'json')}
    actual_files = {p.name for p in data.glob('forward-*')}
    require(actual_files == expected_files, 'forward-file inventory mismatch')
    records = Records(data, count)
    baseline = {}
    for pair in pairs:
        for side in ('cat', 'dog'):
            baseline[pair[side]['image_id']] = initial(records, pair['pair_id'], side)
    terminal_pairs = terminal.get('pairs')
    require(type(terminal_pairs) is list and len(terminal_pairs) == len(pairs), 'terminal pair count mismatch')
    reported_pairs = {p['pair_id']: p for p in terminal_pairs}
    require(set(reported_pairs) == set(ids), 'terminal pair IDs mismatch')
    audit_pairs, pair_hashes = [], {}
    for pair in pairs:
        pair_id, inside, outside = pair['pair_id'], pair['box_tokens'], pair['control_tokens']
        pair_raw, saved = read_json(data / f'{pair_id}.json', 1_000_000)
        pair_hashes[pair_id] = sha(pair_raw)
        require(typed_equal(saved, reported_pairs[pair_id]), 'pair/terminal record mismatch')
        require(saved.get('pair_id') == pair_id and saved.get('split') == split, 'pair identity mismatch')
        base = status_for(baseline[pair['cat']['image_id']]['score'], baseline[pair['dog']['image_id']]['score'])
        require(saved['baseline_eligibility']['eligible'] is base['eligible'] and
                saved['baseline_eligibility']['reasons'] == base['reasons'], 'baseline gate mismatch')
        close(saved['baseline_eligibility']['gap'], base['gap'], 'baseline gap mismatch')
        side_doses, side_margins = {}, {}
        for side, opposite in (('cat', 'dog'), ('dog', 'cat')):
            own, other = baseline[pair[side]['image_id']], baseline[pair[opposite]['image_id']]
            donor = donor_rows[(pair_id, side)]['donor']
            same = None if donor is None else baseline[donor['donor_image_id']]['early']
            dose, requested = prepare(own['early'], other['early'], inside, outside, pair_id, side, same)
            side_doses[side] = dose
            info = saved['sides'][side]
            require(info['image_id'] == pair[side]['image_id'] and info['same_species_donor'] == donor, 'side/donor mismatch')
            check_score(info['baseline'], own['score'])
            compare_dose(info['dose'], dose)
            if donor is None:
                require(info['same_species_baseline'] is None, 'unexpected same-species baseline')
            else:
                check_score(info['same_species_baseline'], baseline[donor['donor_image_id']]['score'])
            conditions = info['conditions']
            repeat = clean_forward(records, pair_id, side, 'repeat', own)
            check_score(conditions['repeat'], repeat)
            plans = [('null_inside', EARLY, inside, None, own['logits']),
                     ('null_outside', EARLY, outside, None, own['logits']),
                     ('self_inside', EARLY, inside, block(own['early'], inside), own['logits']),
                     ('self_outside', EARLY, outside, block(own['early'], outside), own['logits']),
                     ('full_self', EARLY, None, own['early'], own['logits']),
                     ('full_swap', EARLY, None, other['early'], other['logits']),
                     ('natural_inside', EARLY, inside, block(other['early'], inside), None)]
            if dose['status'] == 'matchable':
                for name, indices in (('inside', inside), ('outside', outside), ('random', inside),
                                      ('same_species', inside)):
                    if name in requested:
                        plans.append(('matched_' + name, EARLY, indices, requested[name], None))
            plans.append(('final_inside', LATE, inside, block(other['late'], inside), own['logits']))
            require(set(conditions) == {'repeat', 'post_clean'} | {p[0] for p in plans}, 'condition set mismatch')
            side_margins[side] = {}
            inside_norm = None
            for name, path, tokens, replacement, expected_logits in plans:
                state = own['early'] if path == EARLY else own['late']
                score_, tensor = intervention(records, pair_id, side, name, path, tokens, state,
                                              replacement, own, expected_logits)
                norm = per_token = receiver_norm = None
                if name.startswith('matched_'):
                    pre, post = block(tensor['before_all'], tokens), block(tensor['after_all'], tokens)
                    norm = distance(pre, post)
                    per_token = [distance(pre[i * TOKEN_BYTES:(i + 1) * TOKEN_BYTES],
                                          post[i * TOKEN_BYTES:(i + 1) * TOKEN_BYTES]) for i in range(4)]
                    receiver_norm = distance(pre, bytes(len(pre)))
                    require(norm > 0 and abs(norm - dose['q']) <= dose['tolerance'], 'delivered dose differs')
                    if name == 'matched_inside':
                        inside_norm = norm
                    else:
                        require(inside_norm is not None and abs(norm - inside_norm) <= .01 * inside_norm,
                                'matched dose ratio differs')
                compare_condition(conditions[name], score_, name, side, own, norm, per_token,
                                  receiver_norm, dose.get('q'))
                side_margins[side][name] = score_['margin']
            post = clean_forward(records, pair_id, side, 'post_clean', own)
            check_score(conditions['post_clean'], post)
        expected_doses = {s: side_doses[s]['status'] for s in ('cat', 'dog')}
        require(saved['dose_matchability'] == expected_doses, 'pair dose-status mismatch')
        baseline_margins = {s: baseline[pair[s]['image_id']]['score']['margin'] for s in ('cat', 'dog')}
        status, movements = classify_pair(base, expected_doses, side_margins, baseline_margins)
        require(saved['status'] == status, 'pair primary status mismatch')
        if movements is not None:
            for side in ('cat', 'dog'):
                for site in ('inside', 'outside'):
                    close(saved['movements'][side][site], movements[side][site], 'pair movement mismatch')
        else:
            require('movements' not in saved, 'ineligible pair claims primary movements')
        audit_pairs.append({'pair_id': pair_id, 'status': status, 'baseline_eligibility': base,
                            'dose_statuses': expected_doses, 'movements': movements})
    require(records.next == count + 1, 'unexpected trailing forward records')
    aggregate = None
    if split == 'heldout':
        aggregate = heldout_aggregate(audit_pairs)
    require(terminal.get('aggregate') == aggregate, 'held-out aggregate mismatch')
    return {'schema': 'independent-vision-saved-evidence-v1', 'scope':
            'Saved logits, selected activation bytes, dose/control and outcome consistency only; '
            'not execution, source, image authenticity, held-out chronology or pretraining-independence attestation.',
            'synthetic': synthetic, 'split': split, 'manifest_sha256': sha(manifest_raw),
            'donors_sha256': sha(donors_raw), 'terminal_sha256': sha(terminal_raw),
            'forwards': count, 'forward_hashes': records.hashes, 'pair_hashes': pair_hashes,
            'pairs': audit_pairs, 'aggregate': aggregate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--donors', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'heldout'), required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.donors, args.data, args.split, args.synthetic)
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.report.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short report write')
    print(json.dumps({'report_sha256': sha(raw), 'forwards': report['forwards'],
                      'split': report['split'], 'aggregate': report['aggregate']}))


if __name__ == '__main__':
    main()
