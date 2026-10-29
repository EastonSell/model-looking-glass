"""Independent stdlib replay of a *closed* public matched-dose journal.

No model or private input is loaded. Input hashes and process exit must be
authenticated by the caller; full mask-logit bytes are unavailable.
"""
import base64
import contextlib
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys


def require(condition, text):
    if not condition:
        raise ValueError(text)


def decode(bits):
    require(type(bits) is int and 0 <= bits < 2**32, 'bad float32 bits')
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    require(exponent != 255, 'nonfinite float32')
    n = fraction if exponent == 0 else fraction | (1 << 23)
    value = math.ldexp(float(n), -149 if exponent == 0 else exponent - 150)
    return -value if bits >> 31 else value


def power(exponent):
    return Fraction(1 << exponent, 1) if exponent >= 0 else Fraction(1, 1 << -exponent)


def nearest(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + (2 * remainder > value.denominator or
                       (2 * remainder == value.denominator and quotient & 1))


def rounded_bits(value):
    raw = int.from_bytes(struct.pack('<d', value), 'little')
    sign, exponent, mantissa = (raw >> 32) & 0x80000000, (raw >> 52) & 0x7ff, raw & ((1 << 52) - 1)
    require(exponent != 0x7ff, 'nonfinite computed component')
    if exponent:
        mantissa |= 1 << 52
        exponent -= 1075
    else:
        exponent = -1074
    if not mantissa:
        return sign
    top = mantissa.bit_length() - 1 + exponent
    exact = Fraction(mantissa) * power(exponent)
    if top < -126:
        return sign | nearest(exact * power(149))
    significand = nearest(exact * power(23 - top))
    if significand == 1 << 24:
        significand >>= 1
        top += 1
    require(top <= 127, 'float32 component overflow')
    return sign | ((top + 127) << 23) | (significand - (1 << 23))


def vector(encoded):
    raw = base64.b64decode(encoded, validate=True)
    require(len(raw) == 3072 and base64.b64encode(raw).decode() == encoded, 'invalid vector encoding')
    return raw, tuple(decode(bits[0]) for bits in struct.iter_unpack('<I', raw))


def norm(values):
    return math.sqrt(math.fsum(value * value for value in values))


def encoded_bits(values):
    return struct.pack('<768I', *(rounded_bits(value) for value in values))


def artificial(row_id, site, base, radius):
    prefix = b'bert-matched-v1\0' + row_id.encode('ascii') + b'\0' + site.encode('ascii') + b'\0'
    data = b''.join(hashlib.sha256(prefix + i.to_bytes(4, 'big')).digest() for i in range(3))
    scale = radius / math.sqrt(768.0)
    return encoded_bits(math.fsum((base[i], scale if (data[i // 8] >> (7 - i % 8)) & 1 else -scale))
                        for i in range(768))


def margin(event):
    left, right = decode(event['is_bits']), decode(event['are_bits'])
    require(struct.pack('<d', left) == struct.pack('<d', event['is_logit']) and
            struct.pack('<d', right) == struct.pack('<d', event['are_logit']), 'selected logit bits differ')
    return math.fsum((left, -right))


def main(manifest_path, journal_path, receipt_path, *,
         expected_manifest_sha='5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb',
         expected_journal_sha='9f482a369742960e24273d595d33d2a5fdc8ba2dfc82f1d65f6046266301a3ca'):
    raw_manifest = manifest_path.read_bytes()
    require(hashlib.sha256(raw_manifest).hexdigest() == expected_manifest_sha, 'wrong manifest')
    manifest = json.loads(raw_manifest)
    raw = journal_path.read_bytes()
    require(0 < len(raw) <= 16 * 1024 * 1024 and raw.endswith(b'\n'), 'journal length')
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == expected_journal_sha, 'journal differs from the externally supplied digest')
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt['journal_sha256'] == digest and receipt['forward_count'] == 480 and
            receipt['status'] == 'complete_candidate', 'controller candidate binding')
    records = [json.loads(line) for line in raw.splitlines()]
    require(len(records) == 1152, 'record count')
    rows = {row['row_id']: row for row in manifest['rows']}
    original_cases = receipt['audit']['aggregate']['cases']
    output_cases, family_results = [], []
    actions = ('self', 'A_opposite', 'B_same', 'B_opposite', 'artificial')
    donor_actions = ('A_opposite', 'B_same', 'B_opposite')
    sites = ('block0', 'block2')
    for family in range(1, 9):
        group = [event for event in records if event.get('family') == family]
        require(len(group) == 144, 'family record count')
        completed, plans, controls = {}, {}, []
        forward_index = 0
        for index, event in enumerate(group):
            status = event['status']
            if status == 'started':
                forward_index += 1
                require(event['forward_index'] == forward_index and index + 1 < len(group), 'forward ordering')
                after = group[index + 1]
                require(after['status'] == 'completed' and all(after.get(k) == event.get(k) for k in
                        ('family', 'split', 'forward_index', 'kind', 'row_id', 'donor_row_id', 'case_id', 'site', 'layer', 'token')),
                        'started/completed identity')
            elif status == 'completed':
                key = (event['kind'], event['row_id'], event['site'], event['case_id'])
                require(key not in completed and index > 0 and group[index - 1]['status'] == 'started', 'duplicate completion')
                expected_row = rows[event['row_id']]
                require(event['model_input_ids'] == expected_row['input_ids'] and
                        event['model_attention_mask'] == expected_row['attention_mask'] and
                        event['token'] == 2 and event['layer'] == (0 if event['site'] == 'block0' else 2),
                        'completed input or site identity')
                completed[key] = event
            elif status == 'dose_plan':
                require(event['case_id'] not in plans, 'duplicate dose plan')
                plans[event['case_id']] = event
            elif status == 'control':
                require(event['passed'] is True, 'failed control')
                controls.append((event['name'], event['case_id'], event['site']))
            else:
                raise ValueError('unexpected record status')
        require(forward_index == 60 and len(completed) == 60 and len(plans) == 4 and len(controls) == 20,
                'family schedule incomplete')
        family_rows = [row for row in manifest['rows'] if row['family'] == family]
        family_targets = [target for target in manifest['targets'] if target['family'] == family]
        expected_controls = [('clean_matches_block0', row['row_id'], 'block2') for row in family_rows]
        expected_controls += [('self_patch', target['case_id'], site) for target in family_targets for site in sites]
        expected_controls += [('final_clean', target['case_id'], 'block2') for target in family_targets]
        require(controls == expected_controls, 'control identity or order')
        captured = {site: {} for site in sites}
        for site in sites:
            for row in family_rows:
                event = completed['capture', row['row_id'], site, None]
                require(event['model_input_ids'] == row['input_ids'] and event['model_attention_mask'] == row['attention_mask'],
                        'captured inputs')
                captured[site][row['row_id']] = event
                margin(event)
        for row in family_rows:
            a, b = (captured[site][row['row_id']] for site in sites)
            require(a['mask_logits_sha256'] == b['mask_logits_sha256'] and
                    (a['is_bits'], a['are_bits']) == (b['is_bits'], b['are_bits']), 'cross-site clean')
        family_cases = []
        for target in family_targets:
            row_id, case_id = target['target_row_id'], target['case_id']
            clean = captured['block0'][row_id]
            final = completed['final_clean', row_id, 'block2', case_id]
            require((final['is_bits'], final['are_bits'], final['mask_logits_sha256']) ==
                    (clean['is_bits'], clean['are_bits'], clean['mask_logits_sha256']), 'final clean logits')
            require(final['before_float32_le_base64'] == captured['block2'][row_id]['before_float32_le_base64'],
                    'final clean vector')
            direction, distance, base = {}, {}, {}
            for site in sites:
                _, base[site] = vector(captured[site][row_id]['before_float32_le_base64'])
                for action in donor_actions:
                    donor = target['donor_row_ids'][action]
                    _, donor_values = vector(captured[site][donor]['before_float32_le_base64'])
                    direction[site, action] = tuple(math.fsum((x, -y)) for x, y in zip(donor_values, base[site]))
                    distance[site, action] = norm(direction[site, action])
                    require(distance[site, action] >= 1e-4, 'dose floor')
            radius = min(distance.values())
            plan = plans[case_id]
            require(plan['row_id'] == row_id and plan['radius'] == radius, 'six-distance radius')
            numbers = {}
            for site in sites:
                base_encoded = captured[site][row_id]['before_float32_le_base64']
                base_norm = norm(base[site])
                self_event = completed['self', row_id, site, case_id]
                require(self_event['donor_row_id'] == row_id and self_event['before_float32_le_base64'] == base_encoded and
                        self_event['after_float32_le_base64'] == base_encoded and
                        self_event['submitted_float32_le_base64'] is None and
                        (self_event['is_bits'], self_event['are_bits'], self_event['mask_logits_sha256']) ==
                        (clean['is_bits'], clean['are_bits'], clean['mask_logits_sha256']), 'self control')
                margins = {'self': margin(self_event)}
                raw_bits = {'self': [self_event['is_bits'], self_event['are_bits']]}
                doses = {}
                for action in (*donor_actions, 'artificial'):
                    event = completed[action, row_id, site, case_id]
                    expected_donor = target['donor_row_ids'][action] if action in donor_actions else None
                    require(event['donor_row_id'] == expected_donor and event['before_float32_le_base64'] == base_encoded,
                            'action target/donor')
                    saved, landed = vector(event['after_float32_le_base64'])
                    submitted, _ = vector(event['submitted_float32_le_base64'])
                    require(saved == submitted, 'patch landing')
                    require(plan['sites'][site][action]['value_float32_le_base64'] == event['submitted_float32_le_base64'],
                            'saved dose plan')
                    if action == 'artificial':
                        expected = artificial(row_id, site, base[site], radius)
                    else:
                        scale = radius / distance[site, action]
                        require(plan['sites'][site][action]['distance'] == distance[site, action] and
                                plan['sites'][site][action]['fraction'] == scale, 'distance or fraction')
                        expected = encoded_bits(math.fsum((b, scale * delta)) for b, delta in
                                                zip(base[site], direction[site, action]))
                    require(saved == expected, 'independent rounded action')
                    actual = norm(math.fsum((a, -b)) for a, b in zip(landed, base[site]))
                    absolute_error = abs(actual - radius)
                    relative_error = absolute_error / radius
                    ratio = None if base_norm == 0 else actual / base_norm
                    saved_metrics = plan['sites'][site][action]
                    require(absolute_error <= max(1e-6, 1e-5 * radius) and
                            saved_metrics['realized_norm'] == actual and
                            saved_metrics['absolute_error'] == absolute_error and
                            saved_metrics['relative_error'] == relative_error and
                            saved_metrics['base_norm'] == base_norm and
                            saved_metrics['realized_over_base_norm'] == ratio, 'realized dose')
                    doses[action] = {'distance': None if action == 'artificial' else distance[site, action],
                                     'fraction': None if action == 'artificial' else radius / distance[site, action],
                                     'realized_norm': actual, 'absolute_error': absolute_error,
                                     'relative_error': relative_error, 'base_norm': base_norm,
                                     'realized_over_base_norm': ratio}
                    margins[action] = margin(event)
                    raw_bits[action] = [event['is_bits'], event['are_bits']]
                sign = 1 if rows[row_id]['subject_number'] == 'singular' else -1
                effect = -sign * math.fsum((margins['B_opposite'], -margins['B_same']))
                numbers[site] = {'primary_effect': effect, 'margins': margins, 'raw_bits': raw_bits, 'doses': doses,
                                 'artificial_oriented_shift': -sign * math.fsum((margins['artificial'], -margin(clean)))}
            result = {'case_id': case_id, 'row_id': row_id, 'family': family, 'radius': radius,
                      'subject_number': rows[row_id]['subject_number'], 'distractor_number': rows[row_id]['distractor_number'],
                      'clean_margin': margin(clean), 'clean_bits': [clean['is_bits'], clean['are_bits']],
                      'baseline_correct': sign * margin(clean) > 0, 'sites': numbers,
                      'paired_difference': math.fsum((numbers['block0']['primary_effect'], -numbers['block2']['primary_effect']))}
            matched = next((case for case in original_cases if case['case_id'] == case_id), None)
            require(matched is not None and matched['radius'] == radius and
                    matched['baseline_correct'] == result['baseline_correct'] and
                    matched['paired_difference'] == result['paired_difference'] and
                    all(matched['sites'][site]['primary_effect'] == numbers[site]['primary_effect'] for site in sites),
                    'independent case summary mismatch')
            family_cases.append(result)
            output_cases.append(result)
        family_results.append({'family': family,
            'block0': math.fsum(case['sites']['block0']['primary_effect'] for case in family_cases) / 4,
            'block2': math.fsum(case['sites']['block2']['primary_effect'] for case in family_cases) / 4,
            'paired': math.fsum(case['paired_difference'] for case in family_cases) / 4})
    require(len(output_cases) == 32, 'case count')
    e0 = math.fsum(family['block0'] for family in family_results) / 8
    e2 = math.fsum(family['block2'] for family in family_results) / 8
    paired = math.fsum(family['paired'] for family in family_results) / 8
    positive = sum(family['paired'] > 0 for family in family_results)
    criterion = e0 > .1 and paired > .1 and positive >= 6
    aggregate = receipt['audit']['aggregate']
    require((e0, e2, paired, positive, criterion) ==
            (aggregate['block0_mean'], aggregate['block2_mean'], aggregate['paired_mean'],
             aggregate['positive_families'], aggregate['criterion_met']), 'aggregate mismatch')
    print(json.dumps({'schema': 'independent-matched-study-replay-v1', 'journal_sha256': digest,
        'forwards': 480, 'cases': output_cases, 'families': family_results,
        'block0_mean': e0, 'block2_mean': e2, 'paired_mean': paired,
        'positive_families': positive, 'criterion_met': criterion,
        'baseline_correct': sum(case['baseline_correct'] for case in output_cases),
        'scope': 'saved public data consistency; process provenance and full-mask values are not attested'},
        sort_keys=True, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    if len(sys.argv) not in (4, 5):
        raise SystemExit('usage: review_bert_matched_results.py MANIFEST CLOSED_JOURNAL CONTROLLER_CANDIDATE [NEW_OUTPUT]')
    arguments = tuple(Path(value) for value in sys.argv[1:4])
    if len(sys.argv) == 4:
        main(*arguments)
    else:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(*arguments)
        encoded = output.getvalue().encode('utf-8')
        with Path(sys.argv[4]).open('xb') as stream:
            stream.write(encoded)
        print('independent replay saved; SHA-256 ' + hashlib.sha256(encoded).hexdigest())
