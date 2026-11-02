"""Deterministic cross-template matched-dose geometry; no model dependency."""
import hashlib
import math
import struct
from looking_glass.bert_matched_numeric import f64_to_f32_bits, f32_bits_to_f64


def values(vector):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError('invalid vector')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
        raise ValueError('vector must be finite')
    return tuple(float(x) for x in vector)


def displacement(left, right):
    if len(left) != len(right):
        raise ValueError('vector widths differ')
    return tuple(a - b for a, b in zip(left, right))


def norm(vector):
    result = math.sqrt(math.fsum(x ** 2 for x in vector))
    if not math.isfinite(result):
        raise ValueError('nonfinite vector norm')
    return result


def f32_bits(value):
    """Round finite binary64 to nearest-even binary32 using integer arithmetic."""
    bits = f64_to_f32_bits(value)
    if (bits >> 23) & 255 == 255:
        raise ValueError('binary32 overflow')
    return bits


def f32_value(bits):
    """Decode finite binary32 bits exactly into binary64 without a float32 cast."""
    return f32_bits_to_f64(bits)


def signs(stem, template, width):
    if not isinstance(stem, str) or template not in ('after', 'with') or type(width) is not int or width < 1:
        raise ValueError('invalid orthogonal-vector identity')
    seed = hashlib.sha256(b'bert-taste-orthogonal-v1\0' + stem.encode('utf-8')
                          + b'\0' + template.encode('ascii')).digest()
    result, counter = [], 0
    while len(result) < width:
        block = hashlib.sha256(seed + counter.to_bytes(4, 'big')).digest()
        for byte in block:
            for shift in range(7, -1, -1):
                result.append(1. if byte & (1 << shift) else -1.)
                if len(result) == width:
                    return tuple(result)
        counter += 1
    return tuple(result)


def orthogonal(stem, template, direction):
    direction = values(direction)
    random = signs(stem, template, len(direction))
    denominator = math.fsum(x ** 2 for x in direction)
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError('invalid donor direction')
    coefficient = math.fsum(a * b for a, b in zip(random, direction)) / denominator
    projected = tuple(math.fsum((a, -coefficient * b)) for a, b in zip(random, direction))
    length = norm(projected)
    if length < 1e-8:
        raise ValueError('orthogonal direction is degenerate')
    return tuple(x / length for x in projected)


def source(base, unit, sign, radius, donor_direction, *, orth=False):
    base, unit, donor_direction = values(base), values(unit), values(donor_direction)
    if len(base) != len(unit) or len(base) != len(donor_direction) or type(sign) is not int or sign not in (-1, 1):
        raise ValueError('invalid source geometry')
    if type(radius) not in (float, int) or not math.isfinite(radius) or radius < 1e-5:
        raise ValueError('invalid common radius')
    intended = tuple(math.fsum((b, (sign * radius) * u)) for b, u in zip(base, unit))
    bits = tuple(f32_bits(x) for x in intended)
    raw = b''.join(struct.pack('<I', x) for x in bits)
    rounded = tuple(f32_value(x) for x in bits)
    actual = displacement(rounded, base)
    delivered = norm(actual)
    donor_norm = norm(donor_direction)
    if donor_norm < 1e-5 or delivered == 0 or abs(delivered-radius) > 1e-6 + 1e-4 * radius:
        raise ValueError('delivered norm differs from requested common radius')
    cosine = math.fsum(a * b for a, b in zip(actual, donor_direction)) / (delivered * donor_norm)
    if not math.isfinite(cosine) or (orth and abs(cosine) > .01) or (not orth and sign * cosine < .99):
        raise ValueError('post-rounding direction differs from the requested direction')
    base_norm = norm(base)
    return {'raw': raw, 'requested_norm': radius, 'delivered_norm': delivered,
            'absolute_error': abs(delivered-radius), 'relative_error': abs(delivered-radius)/radius,
            'recipient_norm': base_norm, 'recipient_norm_status': 'zero' if base_norm == 0 else 'positive',
            'dose_to_recipient_norm': None if base_norm == 0 else delivered / base_norm,
            'cosine_with_donor_direction': cosine}


def plan(stem, captures, *, width=768):
    """Build both signs at both sites for four rows using one common radius."""
    if type(captures) is not dict or set(captures) != {'after', 'with'}:
        raise ValueError('both templates are required')
    clean, distances = {}, {}
    for template in ('after', 'with'):
        if type(captures[template]) is not dict or set(captures[template]) != {'sugar', 'lemon'}:
            raise ValueError('both ingredients are required')
        clean[template] = {}
        for ingredient in ('sugar', 'lemon'):
            row = captures[template][ingredient]
            if type(row) is not dict or set(row) != {'ingredient', 'the'}:
                raise ValueError('both captured sites are required')
            clean[template][ingredient] = {site: values(row[site]) for site in ('ingredient', 'the')}
            if any(len(v) != width for v in clean[template][ingredient].values()):
                raise ValueError('hidden width differs')
        distances[template] = {site: norm(displacement(clean[template]['sugar'][site],
                                                        clean[template]['lemon'][site]))
                               for site in ('ingredient', 'the')}
    radius = min(distances[t][s] for t in ('after', 'with') for s in ('ingredient', 'the'))
    if not math.isfinite(radius) or radius < 1e-5:
        raise ValueError('common radius is invalid')
    actions = {}
    for template in ('after', 'with'):
        axis = displacement(clean[template]['sugar']['ingredient'], clean[template]['lemon']['ingredient'])
        orth = orthogonal(stem, template, axis)
        actions[template] = {}
        for ingredient, other in (('sugar', 'lemon'), ('lemon', 'sugar')):
            directions = {site: displacement(clean[template][other][site], clean[template][ingredient][site])
                          for site in ('ingredient', 'the')}
            row = {}
            for site in ('ingredient', 'the'):
                length = distances[template][site]
                unit = tuple(x / length for x in directions[site])
                for label, sign in (('toward', 1), ('away', -1)):
                    row[site + '_' + label] = source(clean[template][ingredient][site], unit, sign,
                                                    radius, directions[site])
            for label, sign in (('orth_plus', 1), ('orth_minus', -1)):
                row[label] = source(clean[template][ingredient]['ingredient'], orth, sign,
                                    radius, directions['ingredient'], orth=True)
            actions[template][ingredient] = row
    fractions = {template: {site: radius / distances[template][site] for site in ('ingredient', 'the')}
                 for template in ('after', 'with')}
    return {'radius': radius, 'distances': distances, 'fractions': fractions, 'actions': actions}
