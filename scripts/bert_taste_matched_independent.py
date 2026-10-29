"""Independent saved-byte arithmetic for the public taste follow-up.

This standard-library helper loads no checkpoint or study implementation.
The eventual journal reader must bind its schedule and metadata to the frozen
protocol and manifest. Its numeric primitives deliberately use exact rational
rounding rather than the study runner's `struct.pack` conversion.
"""
import base64
import binascii
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(bits):
    """Decode finite unsigned binary32 bits exactly, preserving signed zero."""
    require(type(bits) is int and 0 <= bits <= 0xffffffff, 'bad binary32 bits')
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    require(exponent != 255, 'nonfinite binary32')
    mantissa = fraction if exponent == 0 else fraction | (1 << 23)
    value = math.ldexp(float(mantissa), -149 if exponent == 0 else exponent - 150)
    return -value if bits & 0x80000000 else value


def _power(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _nearest_even(value):
    integer, remainder = divmod(value.numerator, value.denominator)
    return integer + (2 * remainder > value.denominator or
                      (2 * remainder == value.denominator and integer & 1))


def f32_round(value):
    """Round finite binary64 to nearest-even binary32 without a float32 cast."""
    require(type(value) is float, 'expected binary64 value')
    raw = int.from_bytes(struct.pack('<d', value), 'little')
    sign, exponent, fraction = (raw >> 32) & 0x80000000, (raw >> 52) & 0x7ff, raw & ((1 << 52) - 1)
    require(exponent != 0x7ff, 'nonfinite binary64')
    mantissa = fraction if exponent == 0 else fraction | (1 << 52)
    if mantissa == 0:
        return sign
    power = -1074 if exponent == 0 else exponent - 1075
    high = mantissa.bit_length() - 1 + power
    exact = Fraction(mantissa) * _power(power)
    if high < -126:
        return sign | _nearest_even(exact * _power(149))
    significand = _nearest_even(exact * _power(23 - high))
    if significand == 1 << 24:
        significand >>= 1
        high += 1
    require(high <= 127, 'binary32 overflow')
    return sign | ((high + 127) << 23) | (significand - (1 << 23))


def vector(encoded, width):
    """Decode canonical little-endian binary32 vector bytes and finite values."""
    require(type(encoded) is str and type(width) is int and 0 < width <= 100000, 'bad vector input')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError, binascii.Error) as error:
        raise ValueError('bad vector encoding') from error
    require(len(raw) == 4 * width and base64.b64encode(raw).decode('ascii') == encoded, 'bad vector length or encoding')
    return raw, tuple(f32(bits[0]) for bits in struct.iter_unpack('<I', raw))


def l2(values):
    # The study's frozen arithmetic specifies binary64 exponentiation by two.
    norm = math.sqrt(math.fsum(value ** 2 for value in values))
    require(math.isfinite(norm), 'nonfinite norm')
    return norm


def displacement(source, base):
    require(len(source) == len(base), 'vector lengths differ')
    return tuple(source_value - base_value for source_value, base_value in zip(source, base))


def interpolate(base, direction, fraction):
    require(type(fraction) is float and math.isfinite(fraction) and 0 < fraction <= 1, 'bad interpolation fraction')
    require(len(base) == len(direction), 'vector lengths differ')
    return b''.join(struct.pack('<I', f32_round(math.fsum((value, fraction * delta))))
                    for value, delta in zip(base, direction))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _exact_number(actual, expected, message):
    require(type(actual) in (float, int) and type(actual) is not bool and math.isfinite(actual)
            and actual == expected, message)


def _float_vector(raw):
    return tuple(f32(bits[0]) for bits in struct.iter_unpack('<I', raw))


def _summary(record):
    summary = record.get('summary')
    require(type(summary) is dict and _valid_sha(summary.get('mask_logits_sha256')), 'invalid output summary')
    sweet, sour = f32(summary.get('sweet_bits')), f32(summary.get('sour_bits'))
    _exact_number(summary.get('sweet_logit'), sweet, 'sweet bits differ')
    _exact_number(summary.get('sour_logit'), sour, 'sour bits differ')
    require(struct.pack('<d', float(summary['sweet_logit'])) == struct.pack('<d', sweet)
            and struct.pack('<d', float(summary['sour_logit'])) == struct.pack('<d', sour),
            'selected binary64 scalar sign or bits differ')
    margin = math.fsum((sweet, -sour))
    _exact_number(summary.get('sweet_minus_sour'), margin, 'stored margin differs')
    return summary, margin


def _metadata(record, layer, position, hidden, mode):
    patched = mode not in ('capture_ingredient', 'capture_control', 'final_clean')
    self_patch = mode in ('ingredient_self', 'control_self')
    names = ['full_before', 'before'] + (['patch', 'after', 'full_after'] if patched else [])
    steps = record.get('events')
    require(type(steps) is list and len(steps) == len(names), 'intervention steps differ')
    for index, (step, name) in enumerate(zip(steps, names)):
        whole = name in ('full_before', 'full_after')
        require(type(step) is dict and step.get('name') == name
                and step.get('operation') == ('patch' if name == 'patch' else 'capture')
                and type(step.get('order')) is int and step['order'] == index
                and step.get('module_path') == f'distilbert.transformer.layer.{layer}'
                and step.get('module_aliases') == [step['module_path']]
                and type(step.get('forward_index')) is int and step['forward_index'] == 0
                and type(step.get('invocation')) is int and step['invocation'] == 0
                and step.get('output_selector') == [0] and step.get('output_shape') == [1, 10, hidden]
                and step.get('dtype') == 'torch.float32' and step.get('device') == 'cpu'
                and step.get('preceding_same_site_actions') == names[:index]
                and step.get('selected_shape') == ([1, 10, hidden] if whole else [hidden])
                and all(type(value) is int for field in ('output_selector', 'output_shape', 'selected_shape')
                        for value in step[field])
                and step.get('batch_index') == (None if whole else 0)
                and step.get('token_index') == (None if whole else position)
                and (whole or (type(step.get('batch_index')) is int and type(step.get('token_index')) is int))
                and step.get('source_capture') == ('before' if self_patch and name == 'patch' else None),
                'intervention metadata differs')


def _baseline(record, hidden, vocab, sweet_id, sour_id):
    raw, values = vector(record.get('baseline_mask_logits_float32_le_base64'), vocab)
    summary, margin = _summary(record)
    require(sha(raw) == summary['mask_logits_sha256'], 'baseline mask hash differs')
    require(struct.unpack_from('<I', raw, 4 * sweet_id)[0] == summary['sweet_bits']
            and struct.unpack_from('<I', raw, 4 * sour_id)[0] == summary['sour_bits'], 'baseline selected bits differ')
    ranking = sorted(range(vocab), key=lambda i: (-values[i], i))[:5]
    saved_ranking = summary.get('top_five_ids')
    require(type(saved_ranking) is list and saved_ranking == ranking
            and all(type(i) is int for i in saved_ranking), 'baseline top-five differs')
    maximum = max(values)
    denominator = math.fsum(math.exp(value - maximum) for value in values)
    for index, key in ((sweet_id, 'sweet_probability'), (sour_id, 'sour_probability')):
        actual = summary.get(key)
        expected = math.exp(values[index] - maximum) / denominator
        require(type(actual) in (float, int) and type(actual) is not bool and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), 'baseline probability differs')
    return summary, margin


def audit_records(manifest, records, *, hidden=768, vocab=30522, layer_count=6):
    """Independently reconstruct one closed saved study without executing it.

    The caller authenticates protocol, manifest and journal bytes separately.
    Whole-output untouched-position claims remain in-process observations.
    """
    require(type(manifest) is dict and type(records) is list, 'invalid inputs')
    rows = manifest['rows']
    require(type(rows) is list and rows and len(rows) % 2 == 0, 'invalid row inventory')
    tokens = manifest['tokens']
    sweet_id, sour_id = tokens['sweet'], tokens['sour']
    require(all(type(v) is int and 0 <= v < vocab for v in (sweet_id, sour_id)) and sweet_id != sour_id,
            'invalid target IDs')
    require(len({row['id'] for row in rows}) == len(rows), 'duplicate row identity')
    reference, baselines, initial, effects, doses, eligibility = {}, {}, {}, [], [], {}
    cursor, completed = 0, 0

    def forward(row, layer, position, mode, donor=None):
        nonlocal cursor, completed
        require(cursor + 1 < len(records), 'missing forward records')
        start, end = records[cursor:cursor + 2]
        cursor += 2
        identity = {'row_id': row['id'], 'layer': layer, 'position': position, 'mode': mode, 'donor': donor}
        require(type(start) is dict and start == {'event': 'forward_start', **identity}
                and all(type(start.get(k)) is type(v) for k, v in identity.items()), 'forward start differs')
        require(type(end) is dict and end.get('event') == 'forward_complete'
                and all(end.get(k) == v and type(end.get(k)) is type(v) for k, v in identity.items()),
                'forward completion identity differs')
        entry = end.get('model_entry')
        require(type(entry) is dict and entry == {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']}
                and all(type(x) is int for x in entry['input_ids'] + entry['attention_mask']), 'model entry differs')
        probe = {'smallest_times_one_bits': 1, 'normal_half_bits': 0x00400000}
        for key in ('denormal_before', 'denormal_after', 'denormal_after_scoring'):
            require(end.get(key) == probe and all(type(x) is int for x in end[key].values()), 'denormal witness differs')
        _metadata(end, layer, position, hidden, mode)
        require(_valid_sha(end.get('full_before_sha256')), 'missing full-output hash')
        before, values = vector(end.get('before_float32_le_base64'), hidden)
        summary, margin = _summary(end)
        completed += 1
        return end, before, values, summary, margin

    for offset in range(0, len(rows), 2):
        pair = rows[offset:offset+2]
        frame = pair[0]['id'].rsplit('-', 1)[0]
        require(pair[0]['id'] == frame + '-sugar' and pair[1]['id'] == frame + '-lemon', 'pair order differs')
        for layer in range(layer_count):
            captures = {}
            for row in pair:
                for position, site in ((7, 'ingredient'), (1, 'control')):
                    end, before, values, summary, margin = forward(row, layer, position, 'capture_' + site)
                    require(end.get('after_float32_le_base64') is None and
                            end.get('source_float32_le_base64') is None and end.get('donor_clean_float32_le_base64') is None
                            and end.get('full_after_sha256') is None and end.get('untouched_positions_equal') is None,
                            'capture claims a patch')
                    captures[row['id'], site] = (end, before, values)
                    if layer == 0 and site == 'ingredient':
                        baseline, base_margin = _baseline(end, hidden, vocab, sweet_id, sour_id)
                        reference[row['id']] = baseline['mask_logits_sha256']
                        baselines[row['id']] = (baseline, base_margin)
                        initial[row['id']] = (before, end['full_before_sha256'])
                    else:
                        require(end.get('baseline_mask_logits_float32_le_base64') is None, 'extra baseline logits')
                        require(summary['mask_logits_sha256'] == reference[row['id']]
                                and (summary['sweet_bits'], summary['sour_bits']) ==
                                (baselines[row['id']][0]['sweet_bits'], baselines[row['id']][0]['sour_bits']),
                                'clean output differs')
                        require(_typed_equal(summary, baselines[row['id']][0]), 'clean output summary differs')
                    if site == 'control':
                        require(end['full_before_sha256'] == captures[row['id'], 'ingredient'][0]['full_before_sha256'],
                                'clean full-output hashes differ across captured positions')
            if frame not in eligibility:
                sugar, lemon = baselines[pair[0]['id']], baselines[pair[1]['id']]
                target_ids = {sweet_id, sour_id}
                eligibility[frame] = (sugar[1] > 0 and lemon[1] < 0 and
                    bool(target_ids.intersection(sugar[0]['top_five_ids'])) and
                    bool(target_ids.intersection(lemon[0]['top_five_ids'])))
            distances = {}
            for site in ('ingredient', 'control'):
                distances[site] = l2(displacement(captures[pair[0]['id'], site][2], captures[pair[1]['id'], site][2]))
                require(distances[site] >= 1e-5, 'distance below fixed minimum')
            radius = min(distances.values())
            fractions = {site: radius / distances[site] for site in distances}
            require(cursor < len(records), 'missing dose plan')
            plan = records[cursor]
            cursor += 1
            require(type(plan) is dict and plan.get('event') == 'dose_plan' and plan.get('row_id') == pair[0]['id']
                    and plan.get('frame') == frame and type(plan.get('layer')) is int and plan['layer'] == layer,
                    'dose plan identity differs')
            _exact_number(plan.get('radius'), radius, 'radius differs')
            for name in ('ingredient', 'control'):
                _exact_number(plan['distances'][name], distances[name], 'distance differs')
                _exact_number(plan['fractions'][name], fractions[name], 'fraction differs')
            doses.append({'frame': frame, 'layer': layer, 'radius': radius, 'distances': distances,
                          'fractions': fractions})
            for index, row in enumerate(pair):
                donor = pair[1-index]
                direction = 'lemon_to_sugar' if index == 0 else 'sugar_to_lemon'
                for mode, site, self_patch, matched in (
                        ('ingredient_donor', 'ingredient', False, False),
                        ('ingredient_self', 'ingredient', True, False),
                        ('control_self', 'control', True, False),
                        ('ingredient_matched', 'ingredient', False, True),
                        ('control_matched', 'control', False, True)):
                    position = 7 if site == 'ingredient' else 1
                    end, before_raw, before, summary, margin = forward(row, layer, position, mode,
                                                                       row['id'] if self_patch else donor['id'])
                    clean, expected_raw, expected_values = captures[row['id'], site]
                    require(before_raw == expected_raw and end['full_before_sha256'] == clean['full_before_sha256'],
                            'patch input differs from clean capture')
                    after_raw, after = vector(end.get('after_float32_le_base64'), hidden)
                    require(end.get('untouched_positions_equal') is True and _valid_sha(end.get('full_after_sha256')),
                            'missing owner unchanged-output witness')
                    if self_patch:
                        require(end.get('source_float32_le_base64') is None and
                                end.get('donor_clean_float32_le_base64') is None and after_raw == before_raw and
                                end['full_after_sha256'] == end['full_before_sha256'] and
                                end.get('dose') is None and end.get('actual_pre_post_norm') is None,
                                'self-patch differs')
                    else:
                        donor_raw, donor_values = captures[donor['id'], site][1:]
                        reported_donor, _ = vector(end.get('donor_clean_float32_le_base64'), hidden)
                        submitted_raw, _ = vector(end.get('source_float32_le_base64'), hidden)
                        require(reported_donor == donor_raw and after_raw == submitted_raw, 'donor or landing differs')
                        if matched:
                            wanted = interpolate(expected_values, displacement(donor_values, expected_values), fractions[site])
                            require(submitted_raw == wanted, 'submitted float32 bits differ from independent rounding')
                            realized = l2(displacement(after, before))
                            base_norm = l2(before)
                            require(realized > 0 and abs(realized-radius) <= 1e-6 + 1e-4 * radius,
                                    'delivered dose outside fixed tolerance')
                            dose = end.get('dose')
                            require(type(dose) is dict, 'missing dose details')
                            for key, value in (('fraction', fractions[site]), ('requested_norm', radius),
                                    ('delivered_norm', realized), ('relative_error', abs(realized-radius)/radius),
                                    ('recipient_norm', base_norm), ('actual_pre_post_norm', realized)):
                                _exact_number(end.get(key) if key == 'actual_pre_post_norm' else dose.get(key),
                                              value, 'recorded dose differs')
                            ratio = None if base_norm == 0 else realized / base_norm
                            require(dose.get('recipient_norm_status') == ('zero' if base_norm == 0 else 'positive')
                                    and dose.get('dose_to_recipient_norm') == ratio, 'relative base dose differs')
                        else:
                            require(submitted_raw == donor_raw and end.get('dose') is None
                                    and end.get('actual_pre_post_norm') is None, 'full donor differs')
                    if self_patch or layer == layer_count-1:
                        require(summary['mask_logits_sha256'] == reference[row['id']]
                                and (summary['sweet_bits'], summary['sour_bits']) ==
                                (baselines[row['id']][0]['sweet_bits'], baselines[row['id']][0]['sour_bits']),
                                'self/final-block logits differ')
                        require(_typed_equal(summary, baselines[row['id']][0]), 'self/final-block summary differs')
                    require(end.get('baseline_mask_logits_float32_le_base64') is None, 'extra baseline logits')
                    base_margin = baselines[row['id']][1]
                    effects.append({'frame': frame, 'recipient': row['id'], 'donor': row['id'] if self_patch else donor['id'],
                        'layer': layer, 'position': position, 'direction': direction, 'mode': mode,
                        'baseline_margin': base_margin, 'patched_margin': margin,
                        'effect': math.fsum((margin, -base_margin)),
                        'sweet_bits': summary['sweet_bits'], 'sour_bits': summary['sour_bits'],
                        'saved_summary': summary})
        for row in pair:
            end, before, _, summary, _ = forward(row, 0, 7, 'final_clean')
            require(summary['mask_logits_sha256'] == reference[row['id']]
                    and (summary['sweet_bits'], summary['sour_bits']) ==
                        (baselines[row['id']][0]['sweet_bits'], baselines[row['id']][0]['sour_bits'])
                    and (before, end['full_before_sha256']) == initial[row['id']]
                    and end.get('after_float32_le_base64') is None
                    and end.get('source_float32_le_base64') is None, 'final clean differs')
            require(_typed_equal(summary, baselines[row['id']][0]), 'final clean summary differs')
    require(cursor == len(records), 'unexpected trailing records')
    indexed = {(e['frame'], e['layer'], e['direction'], e['mode']): e['effect'] for e in effects}
    comparisons, layers = [], []
    for layer in range(layer_count):
        group = {'layer': layer}
        for direction, sign in (('sugar_to_lemon', 1), ('lemon_to_sugar', -1)):
            values = []
            for frame in eligibility:
                advantage = math.fsum((sign * indexed[frame, layer, direction, 'ingredient_matched'],
                                       -sign * indexed[frame, layer, direction, 'control_matched']))
                comparisons.append({'frame': frame, 'layer': layer, 'direction': direction,
                                    'eligible': eligibility[frame], 'signed_advantage': advantage})
                if eligibility[frame]:
                    values.append(advantage)
            group[direction] = {'n': len(values), 'mean': None if not values else math.fsum(values) / len(values)}
        layers.append(group)
    return {'completed_forwards': completed, 'baselines': {key: value[0] for key, value in baselines.items()},
            'eligibility': eligibility, 'effects': effects,
            'doses': doses, 'comparisons': comparisons, 'layer_summary': layers,
            'scope': 'saved-record consistency; full-output equality and execution are not independently attested'}


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _typed_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _read_bounded(path, limit):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            and not getattr(info, 'st_file_attributes', 0) & 0x400, 'evidence is not an ordinary file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'evidence exceeds bound')
    return raw


def audit_files(manifest_path, result_path, journal_dir, *, manifest_sha256,
                hidden=768, vocab=30522, layer_count=6):
    """Reconstruct a closed result after a separate observer has reaped it.

    The manifest digest comes from the external frozen source, not the result
    being audited. This function verifies saved-byte consistency only.
    """
    raw_manifest = _read_bounded(Path(manifest_path), 1_000_000)
    require(sha(raw_manifest) == manifest_sha256, 'manifest digest differs')
    require(not os.path.lexists(Path(journal_dir) / 'incomplete.json'), 'attempt has an incomplete marker')
    manifest = _json(raw_manifest)
    result = _json(_read_bounded(Path(result_path), 8_000_000))
    require(type(result) is dict and result.get('status') == 'complete', 'result is not complete')
    names = [manifest['rows'][i]['id'].rsplit('-', 1)[0] + '.jsonl'
             for i in range(0, len(manifest['rows']), 2)]
    require(len(set(names)) == len(names) and set(result.get('journals', {})) == set(names),
            'journal inventory differs')
    records, journal_hashes = [], {}
    for name in names:
        raw = _read_bounded(Path(journal_dir) / name, 16_000_000)
        require(raw.endswith(b'\n'), 'incomplete journal')
        detail = result['journals'][name]
        require(type(detail) is dict and set(detail) == {'sha256', 'size'}
                and type(detail['size']) is int and detail['size'] == len(raw)
                and detail['sha256'] == sha(raw), 'journal size/hash differs')
        records.extend(_json(line) for line in raw.splitlines())
        journal_hashes[name] = sha(raw)
    report = audit_records(manifest, records, hidden=hidden, vocab=vocab, layer_count=layer_count)
    require(type(result.get('completed_forwards')) is int
            and report['completed_forwards'] == result['completed_forwards'], 'forward count differs')
    require(_typed_equal(result.get('baselines'), report['baselines']), 'saved baselines differ from journal')
    require(result.get('eligibility') == report['eligibility']
            and all(type(value) is bool for value in result['eligibility'].values()), 'eligibility differs')
    require(_typed_equal(result.get('doses'), report['doses'])
            and _typed_equal(result.get('comparisons'), report['comparisons'])
            and _typed_equal(result.get('layer_summary'), report['layer_summary']), 'aggregate or dose summaries differ')
    owner_effects = result.get('effects')
    require(type(owner_effects) is list and len(owner_effects) == len(report['effects']), 'effect count differs')
    for expected, saved in zip(report['effects'], owner_effects):
        require(type(saved) is dict and all(saved.get(k) == v and type(saved.get(k)) is type(v)
                for k, v in expected.items() if k not in ('sweet_bits', 'sour_bits', 'saved_summary')),
                'saved raw effect differs')
        require(_typed_equal(saved.get('summary'), expected['saved_summary']), 'saved action summary differs')
    report['manifest_sha256'] = manifest_sha256
    report['journal_sha256'] = journal_hashes
    return report


if __name__ == '__main__':
    require(len(sys.argv) == 5, 'usage: checker MANIFEST RESULT JOURNAL_DIR MANIFEST_SHA256')
    checked = audit_files(sys.argv[1], sys.argv[2], sys.argv[3], manifest_sha256=sys.argv[4])
    print(json.dumps(checked, sort_keys=True, separators=(',', ':'), allow_nan=False))
