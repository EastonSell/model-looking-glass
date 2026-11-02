"""Independent stdlib replay of screened v4 ViT/COCO saved evidence.

Checks byte, screen, selection, intervention and decision consistency only.
Evidence files must remain quiescent during the read. Does not attest source,
model, inputs, execution, supervision or timing. It checks nearest binary64
rounding at audit boundaries, not every operation or transient mode change.
Top-one IDs are checked; optional human-readable label strings are not.
"""
import argparse
import hashlib
import io
import re
import zipfile
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


def require_nearest_rounding():
    # Nonconstant expressions distinguish all four standard rounding modes.
    # This samples current process state, not concurrent transient changes.
    one = float.fromhex('0x1p0')
    delta = float.fromhex('0x1.8p-52')
    expected = float.fromhex('0x1.0000000000002p0')
    require(one + delta == expected and -one - delta == -expected,
            'binary64 rounding mode is not nearest-even')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bounded_read(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    require(len(raw) <= limit, f'oversized evidence file: {path.name}')
    return raw


def read_json(path, limit=4_000_000):
    raw = bounded_read(path,limit)
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
    signs = [1 if hashlib.sha256(f'vision-coco-small-dose-random-v2:{pair_id}:{side}:{i}'.encode('ascii')).digest()[0] & 1
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
    for key in ('dog_mass','cat_mass','off_mass','off_conditional_tv'):
        if key in computed:
            require(0 <= stored[key] <= 1, f'probability outside [0,1]: {key}')
    if all(key in computed for key in ('dog_mass','cat_mass','off_mass')):
        close(math.fsum(stored[key] for key in ('dog_mass','cat_mass','off_mass')),1.,
              'reported probability masses do not sum to one')


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
        packed = bounded_read(self.root / row['file'],1_000_000)
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
    if cat['margin'] >= 0: reasons.append('cat-relative-margin')
    if dog['margin'] <= 0: reasons.append('dog-relative-margin')
    if gap < 1: reasons.append('gap-below-one')
    if cat['cat_mass'] < .2: reasons.append('cat-mass-below-0.2')
    if dog['dog_mass'] < .2: reasons.append('dog-mass-below-0.2')
    return {'eligible': not reasons, 'reasons': reasons, 'gap': gap}


def prepare(own, opposite, same, ids, pair_id, side):
    receiver, other, donor = block(own, ids), block(opposite, ids), block(same, ids)
    receiver_norm = distance(receiver, bytes(len(receiver)))
    opposite_distance, same_distance = distance(receiver, other), distance(receiver, donor)
    result = {'receiver_norm': receiver_norm, 'opposite_distance': opposite_distance,
              'same_species_distance': same_distance, 'status': 'dose-ineligible', 'reasons': []}
    for name, value in (('receiver-norm', receiver_norm), ('opposite-distance', opposite_distance),
                        ('same-species-distance', same_distance)):
        if value <= 1e-6: result['reasons'].append(name)
    if result['reasons']: return result, {}
    q = min(.099 * receiver_norm, opposite_distance, same_distance)
    result['q'] = q
    if q <= 1e-6:
        result['reasons'].append('q')
        return result, {}
    requested = {'opposite': planned(receiver, other, q, opposite_distance),
                 'same_species': planned(receiver, donor, q, same_distance),
                 'random': random_planned(receiver, q, pair_id, side)}
    norms = {name: distance(value, receiver) for name, value in requested.items()}
    tolerance = min(max(1e-5, 1e-5*q), .01*q)
    result.update(tolerance=tolerance, planned_norms=norms,
                  donor_scales={'opposite':q/opposite_distance, 'same_species':q/same_distance})
    first = norms['opposite']
    if any(norm <= 0 or abs(norm-q) > tolerance or abs(norm-first) > .01*first
           or norm > .1*receiver_norm for norm in norms.values()):
        result.update(status='dose-unmatchable', reasons=['rounded-delivery'])
    else:
        result['status'] = 'matchable'
    return result, requested


def compare_dose(stored, expected):
    require(type(stored) is dict and set(stored) == set(expected), 'dose fields differ')
    for key, value in expected.items():
        actual = stored[key]
        if type(value) is dict:
            require(type(actual) is dict and set(actual) == set(value), 'dose nested fields differ')
            for name in value: close(actual[name], value[name], f'dose {key}/{name} differs')
        elif type(value) in (str, list):
            require(typed_equal(actual, value), f'dose {key} differs')
        else:
            close(actual, value, f'dose {key} differs')
    receiver, opposite, same = (stored[k] for k in ('receiver_norm','opposite_distance','same_species_distance'))
    require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in (receiver,opposite,same)),
            'reported dose distances invalid')
    reasons = [name for name,value in (('receiver-norm',receiver),('opposite-distance',opposite),
                                       ('same-species-distance',same)) if value <= 1e-6]
    if reasons:
        require(stored['status'] == 'dose-ineligible' and stored['reasons'] == reasons,
                'reported dose distances contradict eligibility')
        return
    q = stored['q']
    require(type(q) in (int,float) and math.isfinite(q) and q == min(.099*receiver,opposite,same),
            'reported dose does not equal its fixed bound')
    if q <= 1e-6:
        require(stored['status'] == 'dose-ineligible' and stored['reasons'] == ['q'],
                'reported q contradicts eligibility')
        return
    tolerance = stored['tolerance']
    require(type(tolerance) in (int,float) and math.isfinite(tolerance) and
            tolerance == min(max(1e-5,1e-5*q),.01*q), 'reported dose tolerance differs')
    norms = stored['planned_norms']
    require(set(norms) == {'opposite','same_species','random'} and
            all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in norms.values()),
            'reported planned doses invalid')
    first = norms['opposite']
    bad = any(v <= 0 or abs(v-q) > tolerance or abs(v-first) > .01*first or v > .1*receiver
              for v in norms.values())
    require(stored['status'] == ('dose-unmatchable' if bad else 'matchable') and
            stored['reasons'] == (['rounded-delivery'] if bad else []),
            'reported planned doses contradict matchability')


def classify(base, doses, movements):
    if not base['eligible']: return 'baseline-ineligible'
    if any(value != 'matchable' for value in doses.values()):
        return 'dose-ineligible' if 'dose-ineligible' in doses.values() else 'dose-unmatchable'
    passed = all(movements[side]['opposite'] >= .05 and
                 movements[side]['opposite'] >=
                 max(abs(movements[side]['same_species']), abs(movements[side]['random'])) + .025
                 for side in ('cat', 'dog'))
    return 'positive' if passed else 'negative'


def side_passes(movements):
    return (movements['opposite'] >= .05 and movements['opposite'] >=
            max(abs(movements['same_species']),abs(movements['random'])) + .025)


MODEL_STATE = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
MANIFEST_SHA = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'


def validate_images(images, synthetic=False):
    require(type(images) is list and len(images) == 64, 'candidate count differs')
    ids, jpeg_hashes, pixel_hashes = set(), set(), set()
    for index, row in enumerate(images):
        cat, rank = (17, index) if index < 32 else (18, index - 32)
        image_id = row.get('image_id')
        require(type(image_id) is int and image_id >= 0 and image_id not in ids and
                type(row.get('category_id')) is int and row['category_id'] == cat and
                type(row.get('rank')) is int and row['rank'] == rank and
                row.get('role') == ('donor' if rank % 4 == 0 else 'target') and
                row.get('file_name') == f'{image_id:012d}.jpg', 'candidate identity or role differs')
        ids.add(image_id)
        for key, previous in (('jpeg_sha256', jpeg_hashes), ('pixel_sha256', pixel_hashes)):
            value = row.get(key)
            require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None and
                    value not in previous, 'candidate image hash differs or is duplicated')
            previous.add(value)
        require(type(row.get('jpeg_bytes')) is int and 0 < row['jpeg_bytes'] <= 5_000_000,
                'candidate byte count differs')
        if not synthetic:
            rank_hash = sha(f'vision-coco-recognized-v4:{cat}:{image_id}'.encode('ascii'))
            require(row.get('selection_rank_sha256') == rank_hash, 'candidate rank hash differs')
            if rank:
                previous = images[index - 1]
                require((previous['selection_rank_sha256'], previous['image_id']) < (rank_hash, image_id),
                        'candidate order differs')
        sites = row.get('sites')
        require(type(sites) is list and sites, 'candidate has no qualified site')
        coords = []
        for site in sites:
            coord, counts = site.get('row_col'), site.get('token_mask_pixels')
            require(type(coord) is list and len(coord) == 2 and
                    all(type(x) is int and 0 <= x <= 12 for x in coord) and
                    type(counts) is list and len(counts) == 4 and
                    all(type(x) is int and 0 <= x <= 256 for x in counts), 'invalid site or occupancy')
            r, c = coord
            require(type(site.get('mask_pixels')) is int and site['mask_pixels'] == sum(counts) and
                    site['mask_pixels'] >= 512 and
                    typed_equal(site.get('token_ids'), [1 + 14*(r+dr) + c+dc for dr in range(2) for dc in range(2)]) and
                    type(site.get('squared_center_radius')) is int and
                    site['squared_center_radius'] == (r-6)**2 + (c-6)**2, 'site geometry differs')
            coords.append((r, c))
        require(len(coords) == len(set(coords)) and
                coords == sorted(coords, key=lambda p: ((p[0]-6)**2 + (p[1]-6)**2, p)),
                'site order differs')


def screen_scoring(raw, category):
    require(len(raw) == 4000 and category in (17, 18), 'screen vector or category differs')
    score, _ = summarize(raw)
    top = score['top1_id']
    top_matches = 281 <= top <= 285 if category == 17 else 151 <= top <= 268
    mass = score['cat_mass'] if category == 17 else score['dog_mass']
    reasons = ([] if top_matches else ['top1-outside-annotated-group']) + \
              ([] if mass >= .2 else ['group-mass-below-0.2'])
    return {**score, 'recognized': not reasons, 'reasons': reasons, 'logit_sha256': sha(raw)}


def reported_screen_gate(score, category):
    top = score['top1_id']
    matches = 281 <= top <= 285 if category == 17 else 151 <= top <= 268
    mass = score['cat_mass'] if category == 17 else score['dog_mass']
    reasons = ([] if matches else ['top1-outside-annotated-group']) + \
              ([] if mass >= .2 else ['group-mass-below-0.2'])
    return not reasons, reasons


def selection(images, observed):
    groups = {(category, role): [] for category in (17, 18) for role in ('target', 'donor')}
    require(len(images) == len(observed) == 64, 'screen inventory differs')
    for image, score in zip(images, observed):
        require(image['image_id'] == score['image_id'] and type(score['recognized']) is bool,
                'screen observation identity differs')
        groups[image['category_id'], image['role']].append((image, score))
    pairs, used, uses, rejects = [], set(), {}, []
    for cat, cat_score in groups[17, 'target']:
        if not cat_score['recognized']:
            continue
        for dog, dog_score in groups[18, 'target']:
            if dog['image_id'] in used or not dog_score['recognized']:
                continue
            attempted = {'cat_image_id': cat['image_id'], 'dog_image_id': dog['image_id']}
            if not (cat_score['margin'] < 0 and dog_score['margin'] > 0 and
                    dog_score['margin'] - cat_score['margin'] >= 1):
                rejects.append({**attempted, 'reason': 'baseline-margin-or-gap'})
                continue
            common = {tuple(s['row_col']) for s in cat['sites']} & {tuple(s['row_col']) for s in dog['sites']}
            if not common:
                rejects.append({**attempted, 'reason': 'no-common-qualified-site'})
                continue
            selected = None
            for coord in sorted(common, key=lambda p: ((p[0]-6)**2 + (p[1]-6)**2, p)):
                donors = []
                for category in (17, 18):
                    choices = [row for row, score in groups[category, 'donor']
                               if score['recognized'] and coord in {tuple(s['row_col']) for s in row['sites']}]
                    if not choices:
                        break
                    donors.append(min(choices, key=lambda x: (uses.get(x['image_id'], 0), x['rank'])))
                if len(donors) == 2:
                    selected = coord, donors
                    break
            if selected is None:
                rejects.append({**attempted, 'reason': 'no-recognized-same-species-donors-at-common-site'})
                continue
            (r, c), (cat_donor, dog_donor) = selected
            for donor in (cat_donor, dog_donor):
                uses[donor['image_id']] = uses.get(donor['image_id'], 0) + 1
            used.update((cat['image_id'], dog['image_id']))
            pairs.append({'pair_id': f'pair-{len(pairs):02d}', 'split': 'heldout',
                          'cat': cat, 'dog': dog, 'cat_donor': cat_donor, 'dog_donor': dog_donor,
                          'box_block_row_col': [r, c],
                          'box_tokens': [1 + 14*(r+dr) + c+dc for dr in range(2) for dc in range(2)],
                          'squared_center_radius': (r-6)**2 + (c-6)**2})
            break
        if len(pairs) == 4:
            break
    summary = {'screened_images': 64, 'target_role_images': 48, 'donor_role_images': 16,
               'selected_pairs': len(pairs), 'distinct_selected_targets': len(used),
               'distinct_selected_donors': len(uses),
               'donor_side_counts': {str(k): value for k, value in sorted(uses.items())},
               'planned_decision': 'inconclusive' if len(pairs) < 2 else 'pending-intervention',
               'interventions_planned': bool(pairs), 'examined_rejections': rejects}
    return pairs, summary


def validate_start_terminal(data, mode, manifest_hash, synthetic=False):
    start_raw, start = read_json(data / 'started.json')
    terminal_raw, terminal = read_json(data / 'terminal.json', 5_000_000)
    require(not (data / 'failure.json').exists() and
            start.get('schema') == 'vision-recognized-v4-start' and start.get('mode') == mode and
            start.get('manifest_sha256') == manifest_hash and
            terminal.get('schema') == 'vision-recognized-v4-terminal' and terminal.get('status') == 'complete' and
            terminal.get('mode') == mode and terminal.get('manifest_sha256') == manifest_hash and
            terminal.get('source_revision') == start.get('source', {}).get('revision') and
            typed_equal(terminal.get('runtime'), start.get('runtime')) and
            terminal.get('model_state_sha256_before_after') == start.get('model_state_sha256'),
            'start/terminal identity differs')
    require(type(start.get('source')) is dict and type(start['source'].get('revision')) is str and
            type(start['source'].get('sources')) is dict and type(start.get('runtime')) is dict and
            start['runtime'].get('binary64_rounding') == 'nearest-even',
            'source or runtime missing')
    if not synthetic:
        require(start.get('model_state_sha256') == MODEL_STATE, 'model state digest differs')
    return start_raw, start, terminal_raw, terminal


def validate_archive(raw, data, expected):
    require(0 < len(raw) <= 5_000_000, 'screen archive size differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and {item.filename for item in entries} == expected,
                'screen archive inventory differs')
        for item in entries:
            require(item.file_size <= 1_000_000 and archive.read(item) == bounded_read(data / item.filename,1_000_000),
                    'screen archive member differs')


def validate_custody(path, observer_path, start, manifest_hash, result_hash, terminal_hash, archive_hash, supervisor):
    custody_raw, receipt = read_json(path)
    observer_raw, observer = read_json(observer_path)
    require(receipt.get('schema') == 'vision-recognized-v4-screen-custody' and
            receipt.get('status') == 'complete' and receipt.get('manifest_sha256') == manifest_hash and
            receipt.get('source_revision') == start['source']['revision'] and
            receipt.get('screen_result_sha256') == result_hash and
            receipt.get('screen_terminal_sha256') == terminal_hash and
            receipt.get('screen_archive_sha256') == archive_hash and
            type(receipt.get('owner_exit_code')) is int and receipt['owner_exit_code'] == 0 and
            receipt.get('owner_clean') is True and receipt.get('owner_timed_out') is False and
            type(receipt.get('controller_exit_code')) is int and receipt['controller_exit_code'] == 0 and
            receipt.get('external_observer_sha256') == sha(observer_raw) and
            type(receipt.get('private_supervisor_sha256')) is str and
            re.fullmatch('[0-9a-f]{64}', receipt['private_supervisor_sha256']) is not None,
            'screen custody projection differs')
    require(observer.get('schema') == 'vision-recognized-v4-external-observer' and
            observer.get('status') == 'complete' and observer.get('mode') == 'screen' and
            type(observer.get('controller_exit_code')) is int and observer['controller_exit_code'] == 0 and
            observer.get('controller_reaped') is True and observer.get('timed_out') is False and
            observer.get('forced_stop') is False and
            observer.get('source_revision') == start['source']['revision'] and
            observer.get('supervisor_sha256') == receipt['private_supervisor_sha256'] and
            observer.get('terminal_sha256') == terminal_hash, 'screen observer differs')
    if supervisor is not None:
        supervisor_raw, value = read_json(supervisor)
        require(sha(supervisor_raw) == receipt['private_supervisor_sha256'] and
                value.get('status') == 'complete' and type(value.get('exit_code')) is int and
                value['exit_code'] == 0 and value.get('clean') is True and value.get('timed_out') is False and
                value.get('terminal_sha256') == terminal_hash, 'screen supervisor differs')
    return sha(custody_raw), sha(observer_raw)


def screen_audit(manifest_path, data, public_result, archive_path, custody_path, observer_path,
                 supervisor=None, synthetic=False):
    require_nearest_rounding()
    manifest_raw, manifest = read_json(manifest_path, 5_000_000)
    require(manifest.get('schema') == 'vision-coco-recognized-screen-cases-v4', 'manifest schema differs')
    require(manifest.get('fabricated') is True if synthetic else manifest.get('fabricated') is not True,
            'synthetic flag and manifest differ')
    images = manifest.get('images')
    validate_images(images, synthetic=synthetic)
    manifest_hash = sha(manifest_raw)
    require(synthetic or manifest_hash == MANIFEST_SHA, 'fixed public manifest digest differs')
    start_raw, start, terminal_raw, terminal = validate_start_terminal(data, 'screen', manifest_hash, synthetic)
    require(type(terminal.get('submissions')) is int and terminal['submissions'] == 64 and
            type(terminal.get('saved_forwards')) is int and terminal['saved_forwards'] == 64,
            'screen forward count differs')
    expected = {'started.json', 'terminal.json', 'screen-result.json'} | {
        f'forward-{i:03d}.{ext}' for i in range(1, 65) for ext in ('json', 'bin')}
    require({p.name for p in data.iterdir()} == expected, 'screen inventory differs')
    result_raw, result = read_json(data / 'screen-result.json', 5_000_000)
    require(bounded_read(public_result,5_000_000) == result_raw and result.get('schema') == 'vision-recognized-v4-screen-result' and
            result.get('manifest_sha256') == manifest_hash and terminal.get('screen_result_sha256') == sha(result_raw),
            'public screen-result binding differs')
    observations, raw_by_id, hashes = [], {}, []
    require(type(result.get('screen')) is list and len(result['screen']) == 64, 'screen result count differs')
    for index, image in enumerate(images, 1):
        row_raw, row = read_json(data / f'forward-{index:03d}.json', 200_000)
        raw = bounded_read(data / f'forward-{index:03d}.bin',4000)
        score = screen_scoring(raw, image['category_id'])
        require(type(row.get('submission')) is int and row['submission'] == index and
                row.get('label') == f"screen:{image['image_id']}" and
                row.get('file') == f'forward-{index:03d}.bin' and row.get('sha256') == sha(raw) and
                row.get('dtype') == 'little-endian-float32' and
                typed_equal(row.get('layout'), {'logits': {'shape': [1,1000], 'offset': 0, 'bytes': 4000}}) and
                typed_equal(row.get('metadata'), {'kind':'screen', 'image_id':image['image_id']}),
                'screen forward metadata differs')
        saved = row.get('score')
        result_score = result['screen'][index-1]
        require(type(saved) is dict and set(saved) == set(score) and
                type(result_score) is dict and set(result_score) == set(score) | {'image_id'} and
                type(result_score['image_id']) is int and result_score['image_id'] == image['image_id'],
                'screen score fields differ')
        check_score(saved, {k:score[k] for k in ('margin','dog_mass','cat_mass','off_mass','top1_id')})
        check_score(result_score, {k:score[k] for k in ('margin','dog_mass','cat_mass','off_mass','top1_id')})
        for key in ('recognized','reasons','logit_sha256'):
            require(typed_equal(saved[key],score[key]) and typed_equal(result_score[key],score[key]),
                    'screen gate or saved logit hash differs')
        require(typed_equal(saved,{key:result_score[key] for key in score}) and
                reported_screen_gate(saved,image['category_id']) == (score['recognized'],score['reasons']),
                'reported screen scores contradict the gate')
        observations.append({'image_id':image['image_id'], **score})
        raw_by_id[image['image_id']] = raw
        hashes.append({'submission':index,'json_sha256':sha(row_raw),'binary_sha256':sha(raw)})
    pairs, summary = selection(images, observations)
    reported_pairs, reported_summary = selection(images,result['screen'])
    require(typed_equal(result.get('pairs'),pairs) and typed_equal(result.get('summary'),summary),
            'screen selection or summary differs')
    require(typed_equal(reported_pairs,pairs) and typed_equal(reported_summary,summary),
            'reported screen margins contradict the selection')
    archive_raw = bounded_read(archive_path,5_000_000)
    validate_archive(archive_raw, data, expected)
    custody_hash, observer_hash = validate_custody(custody_path,observer_path,start,manifest_hash,sha(result_raw),
                                                   sha(terminal_raw),sha(archive_raw),supervisor)
    require_nearest_rounding()
    return {'manifest_sha256':manifest_hash, 'start':start, 'start_sha256':sha(start_raw),
            'terminal_sha256':sha(terminal_raw), 'screen_result_sha256':sha(result_raw),
            'screen_archive_sha256':sha(archive_raw), 'screen_custody_sha256':custody_hash,
            'screen_observer_sha256':observer_hash,
            'raw_by_id':raw_by_id, 'observations':observations, 'pairs':pairs, 'summary':summary,
            'forward_hashes':hashes, 'supervisor_checked':supervisor is not None}


def aggregate_v4(pairs):
    valid = {'positive', 'negative', 'dose-ineligible', 'dose-unmatchable'}
    require(type(pairs) is list and len(pairs) <= 4 and
            [p.get('pair_id') for p in pairs] == [f'pair-{i:02d}' for i in range(len(pairs))] and
            all(p.get('split') == 'heldout' and p.get('status') in valid for p in pairs),
            'selected pair/status inventory differs')
    evaluable = [p for p in pairs if p['status'] in ('positive','negative')]
    positive = sum(p['status'] == 'positive' for p in evaluable)
    status = ('inconclusive' if len(evaluable) < 2 else
              'positive' if 4*positive >= 3*len(evaluable) else 'negative')
    return {'status':status,'selected':len(pairs),'evaluable':len(evaluable),'positive':positive}


def intervention_audit(data, screen, synthetic=False):
    require_nearest_rounding()
    pairs = screen['pairs']
    require(1 <= len(pairs) <= 4, 'intervention requires one to four screened pairs')
    _, started, terminal_raw, terminal = validate_start_terminal(data,'intervention',
                                                                   screen['manifest_sha256'],synthetic)
    require(typed_equal(started['source']['sources'],screen['start']['source']['sources']) and
            typed_equal(started['runtime'],screen['start']['runtime']) and
            started['model_state_sha256'] == screen['start']['model_state_sha256'] and
            terminal.get('screen_result_sha256') == screen['screen_result_sha256'] and
            terminal.get('screen_terminal_sha256') == screen['terminal_sha256'] and
            terminal.get('screen_archive_sha256') == screen['screen_archive_sha256'],
            'screen/intervention linkage differs')
    if not synthetic:
        prefix = 'results/vision-coco-recognized-v4-20261029/'
        expected = {prefix+'screen-result.json':screen['screen_result_sha256'],
                    prefix+'screen-evidence.zip':screen['screen_archive_sha256'],
                    prefix+'screen-custody.json':screen['screen_custody_sha256'],
                    prefix+'screen-observer.json':screen['screen_observer_sha256']}
        require(typed_equal(started['source'].get('public_handoff'),expected),
                'parent-bound public handoff differs')
    ids = [p['pair_id'] for p in pairs]
    target_ids = [p[s]['image_id'] for p in pairs for s in ('cat','dog')]
    donor_ids = [p[s+'_donor']['image_id'] for p in pairs for s in ('cat','dog')]
    require(len(ids) == len(set(ids)) and len(target_ids) == len(set(target_ids)) and
            not set(target_ids).intersection(donor_ids), 'target/donor reuse differs')
    count = terminal.get('submissions')
    require(type(count) is int and 0 < count <= 30*len(pairs) and
            type(terminal.get('saved_forwards')) is int and terminal['saved_forwards'] == count,
            'intervention forward count differs')
    files = {f'forward-{i:03d}.{ext}' for i in range(1,count+1) for ext in ('json','bin')}
    require({p.name for p in data.glob('forward-*')} == files, 'intervention forward inventory differs')
    expected_files = files | {'started.json','terminal.json'} | {p['pair_id']+'.json' for p in pairs}
    require({p.name for p in data.iterdir()} == expected_files, 'intervention file inventory differs')
    records, clean = Records(data, count), {}
    for pair in pairs:
        for role in ('cat','dog','cat_donor','dog_donor'):
            image_id = pair[role]['image_id']
            if image_id not in clean:
                clean[image_id] = initial(records,pair['pair_id'],role)
                require(clean[image_id]['logits'] == screen['raw_by_id'][image_id],
                        'screen/intervention clean logits differ')


    terminal_pairs = terminal['pairs']
    require(type(terminal_pairs) is list and len(terminal_pairs) == len(pairs), 'terminal pair count differs')
    reported = {row['pair_id']:row for row in terminal_pairs}
    require([row.get('pair_id') for row in terminal_pairs] == ids and set(reported) == set(ids),
            'terminal pair IDs or order differ')
    summaries, pair_hashes = [], {}
    for pair in pairs:
        pair_id, selected = pair['pair_id'], pair['box_tokens']
        raw, saved = read_json(data / f'{pair_id}.json', 1_000_000)
        pair_hashes[pair_id] = sha(raw)
        require(typed_equal(saved, reported[pair_id]) and saved.get('split') == pair['split'],
                'pair/terminal record differs')
        base = status_for(clean[pair['cat']['image_id']]['score'],clean[pair['dog']['image_id']]['score'])
        require(base['eligible'], 'selected baseline contradicts the fixed screen')
        saved_base = status_for(saved['sides']['cat']['baseline'],saved['sides']['dog']['baseline'])
        require(saved_base['eligible'] == base['eligible'] and saved_base['reasons'] == base['reasons'],
                'reported baseline scores contradict eligibility')
        require(saved['baseline_eligibility']['eligible'] is base['eligible'] and
                saved['baseline_eligibility']['reasons'] == base['reasons'], 'baseline gate differs')
        close(saved['baseline_eligibility']['gap'], base['gap'], 'baseline gap differs')
        require(saved['baseline_eligibility']['gap'] == saved_base['gap'],
                'reported baseline gap contradicts reported scores')
        dose_records, side_movements, reported_movements = {}, {}, {}
        for side, opposite_side in (('cat','dog'),('dog','cat')):
            own = clean[pair[side]['image_id']]
            other = clean[pair[opposite_side]['image_id']]
            same = clean[pair[side+'_donor']['image_id']]
            dose, requested = prepare(own['early'],other['early'],same['early'],selected,pair_id,side)
            dose_records[side] = dose
            info = saved['sides'][side]
            require(type(info.get('image_id')) is int and type(info.get('same_species_donor_id')) is int and
                    info['image_id'] == pair[side]['image_id'] and
                    info['same_species_donor_id'] == pair[side+'_donor']['image_id'], 'side identity differs')
            check_score(info['baseline'],own['score'])
            check_score(info['same_species_baseline'],same['score'])
            category = 17 if side == 'cat' else 18
            require(reported_screen_gate(info['baseline'],category)[0] and
                    reported_screen_gate(info['same_species_baseline'],category)[0],
                    'reported target or donor scores contradict recognition')
            compare_dose(info['dose'],dose)
            conditions = info['conditions']
            repeat = clean_forward(records,pair_id,side,'repeat',own)
            check_score(conditions['repeat'],repeat)
            plans = [('null_inside', EARLY, selected, None, own['logits']),
                     ('self_inside', EARLY, selected, block(own['early'],selected), own['logits']),
                     ('full_self', EARLY, None, own['early'], own['logits']),
                     ('full_swap', EARLY, None, other['early'], other['logits']),
                     ('natural_inside', EARLY, selected, block(other['early'],selected), None)]
            if dose['status'] == 'matchable':
                for name in ('opposite','same_species','random'):
                    plans.append(('matched_'+name,EARLY,selected,requested[name],None))
            plans.append(('final_inside',LATE,selected,block(other['late'],selected),own['logits']))
            require(set(conditions) == {'repeat','post_clean'} | {p[0] for p in plans}, 'conditions differ')
            moves, saved_moves = {}, {}
            first_norm = first_reported_norm = None
            for name, path, tokens_, replacement, expected_logits in plans:
                state = own['early'] if path == EARLY else own['late']
                score_, tensors = intervention(records,pair_id,side,name,path,tokens_,state,replacement,own,expected_logits)
                saved_condition = conditions[name]
                check_score(saved_condition,score_)
                change = score_['margin']-own['score']['margin']
                oriented = change * (1 if side == 'cat' else -1)
                close(saved_condition.get('raw_margin_change'),change,'raw change differs')
                close(saved_condition.get('annotation_oriented_change'),oriented,'oriented change differs')
                saved_change = saved_condition['margin'] - info['baseline']['margin']
                require(saved_condition['raw_margin_change'] == saved_change and
                        saved_condition['annotation_oriented_change'] == saved_change*(1 if side == 'cat' else -1),
                        'reported movement contradicts reported margins')
                if name.startswith('matched_'):
                    pre, post = block(tensors['before_all'],tokens_), block(tensors['after_all'],tokens_)
                    norm, receiver_norm = distance(pre,post), distance(pre,bytes(len(pre)))
                    require(norm > 0 and abs(norm-dose['q']) <= dose['tolerance'] and
                            receiver_norm > 0 and norm <= .1*receiver_norm, 'actual matched dose differs')
                    if name == 'matched_opposite': first_norm = norm
                    else: require(first_norm is not None and abs(norm-first_norm) <= .01*first_norm,
                                  'matched same-side doses differ')
                    close(saved_condition.get('actual_dose'),norm,'actual dose field differs')
                    close(saved_condition.get('receiver_norm'),receiver_norm,'receiver norm differs')
                    close(saved_condition.get('realized_over_receiver_norm'),norm/receiver_norm,'relative dose differs')
                    reported_norm = saved_condition['actual_dose']
                    reported_receiver = saved_condition['receiver_norm']
                    require(reported_norm > 0 and reported_receiver > 0 and
                            abs(reported_norm-info['dose']['q']) <= info['dose']['tolerance'] and
                            reported_norm <= .1*reported_receiver and
                            saved_condition['realized_over_receiver_norm'] == reported_norm/reported_receiver,
                            'reported actual dose contradicts bounds')
                    if name == 'matched_opposite': first_reported_norm = reported_norm
                    else: require(first_reported_norm is not None and
                                  abs(reported_norm-first_reported_norm) <= .01*first_reported_norm,
                                  'reported actual doses contradict matching')
                    per = [distance(pre[i*TOKEN_BYTES:(i+1)*TOKEN_BYTES],post[i*TOKEN_BYTES:(i+1)*TOKEN_BYTES]) for i in range(4)]
                    require(type(saved_condition.get('token_doses')) is list and len(saved_condition['token_doses']) == 4,
                            'token dose inventory differs')
                    for got,want in zip(saved_condition['token_doses'],per): close(got,want,'token dose differs')
                    moves[name.removeprefix('matched_')] = oriented
                    saved_moves[name.removeprefix('matched_')] = saved_condition['annotation_oriented_change']
            post = clean_forward(records,pair_id,side,'post_clean',own)
            check_score(conditions['post_clean'],post)
            side_movements[side] = moves
            reported_movements[side] = saved_moves
            if dose['status'] == 'matchable':
                require(side_passes(moves) == side_passes(saved_moves),
                        'reported condition movements contradict side threshold')
        dose_status = {side:dose_records[side]['status'] for side in ('cat','dog')}
        require(saved['dose_matchability'] == dose_status, 'dose matchability differs')
        movements = side_movements if all(v == 'matchable' for v in dose_status.values()) else None
        expected_status = classify(base,dose_status,movements)
        require(saved['status'] == expected_status, 'pair status differs')
        if movements is None:
            require(saved.get('descriptive_movements') is None, 'unexpected descriptive movements')
        else:
            for side in ('cat','dog'):
                for name in ('opposite','same_species','random'):
                    close(saved['descriptive_movements'][side][name],movements[side][name], 'descriptive movement differs')
                    require(saved['descriptive_movements'][side][name] == reported_movements[side][name],
                            'descriptive and condition movements differ')
                require(side_passes(saved['descriptive_movements'][side]) == side_passes(movements[side]),
                        'descriptive movements contradict side threshold')
        summaries.append({'pair_id':pair_id,'split':pair['split'],'status':expected_status,'baseline_eligibility':base,
                          'dose_statuses':dose_status,'descriptive_movements':movements})
    require(records.next == count+1, 'unexpected trailing forward records')
    result = aggregate_v4(summaries)
    require(typed_equal(terminal.get('aggregate'),result), 'aggregate differs')
    require_nearest_rounding()
    return {'terminal_sha256':sha(terminal_raw),'forwards':count,'forward_hashes':records.hashes,
            'pair_hashes':pair_hashes,'pairs':summaries,'aggregate':result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--screen-data',type=Path,required=True)
    parser.add_argument('--public-screen-result',type=Path,required=True)
    parser.add_argument('--public-screen-archive',type=Path,required=True)
    parser.add_argument('--public-screen-custody',type=Path,required=True)
    parser.add_argument('--public-screen-observer',type=Path,required=True)
    parser.add_argument('--screen-supervisor',type=Path)
    parser.add_argument('--intervention-data',type=Path)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    screen = screen_audit(args.manifest,args.screen_data,args.public_screen_result,args.public_screen_archive,
                          args.public_screen_custody,args.public_screen_observer,args.screen_supervisor,args.synthetic)
    if args.intervention_data is not None:
        intervention = intervention_audit(args.intervention_data,screen,args.synthetic)
        aggregate = intervention['aggregate']
    else:
        intervention = None
        aggregate = aggregate_v4([]) if not screen['pairs'] else None
    report = {'schema':'independent-vision-recognized-v4-audit-v1',
              'scope':'Saved-byte and decision consistency; not source, images, model, execution, supervision, '
                      'publication timing or pretraining-history attestation. Evidence files must remain quiescent '
                      'during the read. Top-one IDs are checked, but optional label strings are not. '
                      'Screened pairs are model-conditioned.',
              'synthetic':args.synthetic,'manifest_sha256':screen['manifest_sha256'],
              'screen_start_sha256':screen['start_sha256'],'screen_terminal_sha256':screen['terminal_sha256'],
              'screen_result_sha256':screen['screen_result_sha256'],
              'screen_archive_sha256':screen['screen_archive_sha256'],
              'screen_custody_sha256':screen['screen_custody_sha256'],
              'screen_observer_sha256':screen['screen_observer_sha256'],
              'screen_supervisor_checked':screen['supervisor_checked'],
              'screen_forwards':64,'screen_forward_hashes':screen['forward_hashes'],
              'screen_summary':screen['summary'],
              'screen_gate_counts':{'recognized':sum(r['recognized'] for r in screen['observations']),
                                    'unrecognized':sum(not r['recognized'] for r in screen['observations'])},
              'selected_pairs':[{'pair_id':p['pair_id'],'cat_id':p['cat']['image_id'],
                                 'dog_id':p['dog']['image_id'],'cat_donor_id':p['cat_donor']['image_id'],
                                 'dog_donor_id':p['dog_donor']['image_id'],'site':p['box_block_row_col']}
                                for p in screen['pairs']],
              'intervention':intervention,'aggregate':aggregate,
              'pending_intervention':bool(screen['pairs'] and intervention is None)}
    raw = (json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with args.report.open('xb') as stream:
        if stream.write(raw) != len(raw): raise OSError('short report write')
    print(json.dumps({'report_sha256':sha(raw),'screened':64,'selected':len(screen['pairs']),
                      'intervention_forwards':None if intervention is None else intervention['forwards'],
                      'aggregate':aggregate,'pending_intervention':report['pending_intervention']}))


if __name__ == '__main__':
    main()
