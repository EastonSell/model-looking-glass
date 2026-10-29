"""Bit-defined arithmetic for the preregistered BERT magnitude comparison.

These CPU-only functions do not select cases, load a model or authorize a
forward. Callers own immutable vector bytes and must verify actual hook landing.
"""
import hashlib
import math
import struct

WIDTH = 768
SITES = ('block0', 'block2')
DONORS = ('A_opposite', 'B_same', 'B_opposite')
MIN_RADIUS = 1e-4


def _round_shift_even(value, shift):
    if shift <= 0:
        return value << -shift
    retained, discarded = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return retained + (discarded > halfway or (discarded == halfway and retained & 1))


def f64_to_f32_bits(value):
    """Round a finite binary64 to binary32 bits, nearest with ties to even.

    The narrow integer method follows the independently exercised copy-head
    factorial draft's converter; no float32 cast is used.
    """
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('expected a finite binary64 value')
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    sign = (bits >> 32) & 0x80000000
    exponent, fraction = (bits >> 52) & 0x7ff, bits & ((1 << 52) - 1)
    if exponent == 0:
        return sign | _round_shift_even(fraction, 925)
    power, significand = exponent - 1023, (1 << 52) | fraction
    if power >= -126:
        rounded = _round_shift_even(significand, 29)
        if rounded == 1 << 24:
            rounded >>= 1
            power += 1
        if power > 127:
            return sign | 0x7f800000
        return sign | ((power + 127) << 23) | (rounded & 0x7fffff)
    return sign | _round_shift_even(significand, -(power + 97))


def f32_bits_to_f64(bits):
    """Decode finite IEEE binary32 bits exactly into binary64."""
    if type(bits) is not int or not 0 <= bits < 2**32:
        raise ValueError('expected an unsigned 32-bit float pattern')
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    if exponent == 255:
        raise ValueError('nonfinite float32 value')
    value = (math.ldexp(float(fraction), -149) if exponent == 0
             else math.ldexp(float((1 << 23) | fraction), exponent - 150))
    return -value if bits >> 31 else value


def binary64_rounding_probe():
    """Check representative binary64 operations on the calling thread."""
    one, half = 1.0, math.ldexp(1.0, -53)
    next_one = one + math.ldexp(1.0, -52)
    if (one + half != one or next_one + half != one + math.ldexp(1.0, -51)
            or one - math.ldexp(1.0, -54) != one
            or next_one * 1.5 != 1.5 + math.ldexp(1.0, -51)
            or one / 10.0 != float.fromhex('0x1.999999999999ap-4')
            or math.sqrt(2.0) != float.fromhex('0x1.6a09e667f3bcdp+0')):
        raise RuntimeError('binary64 arithmetic probe failed')


def _values(raw):
    if type(raw) is not bytes or len(raw) != 4 * WIDTH:
        raise ValueError('a vector must contain exactly 768 float32 patterns')
    return tuple(f32_bits_to_f64(bits[0]) for bits in struct.iter_unpack('<I', raw))


def _norm(values):
    try:
        result = math.sqrt(math.fsum(value * value for value in values))
    except (OverflowError, ValueError) as error:
        raise ValueError('nonfinite displacement norm') from error
    if not math.isfinite(result):
        raise ValueError('nonfinite displacement norm')
    return result


def _signs(row_id, site):
    if (type(row_id) is not str or not row_id or len(row_id) > 128
            or not row_id.isascii() or site not in SITES):
        raise ValueError('invalid artificial-direction identity')
    prefix = b'bert-matched-v1\0' + row_id.encode('ascii') + b'\0' + site.encode('ascii') + b'\0'
    raw = b''.join(hashlib.sha256(prefix + counter.to_bytes(4, 'big')).digest() for counter in range(3))
    return tuple(1 if byte & (1 << bit) else -1 for byte in raw for bit in range(7, -1, -1))


def _delivered(base, values, radius, base_norm):
    patterns = tuple(f64_to_f32_bits(value) for value in values)
    actual = tuple(f32_bits_to_f64(bits) for bits in patterns)
    realized = _norm(math.fsum((after, -before)) for after, before in zip(actual, base))
    absolute_error = abs(realized - radius)
    if absolute_error > max(1e-6, 1e-5 * radius):
        raise ValueError('float32 rounding missed the fixed displacement tolerance')
    return {'value': struct.pack('<768I', *patterns), 'realized_norm': realized,
            'absolute_error': absolute_error, 'relative_error': absolute_error / radius,
            'base_norm': base_norm, 'realized_over_base_norm': None if base_norm == 0 else realized / base_norm}


def plan_case(row_id, bases, donors):
    """Construct all six donor and two artificial same-radius action vectors.

    ``bases`` maps sites to original vector bytes. ``donors`` maps sites to
    exactly A_opposite/B_same/B_opposite vector bytes. The returned bytes are
    independently auditable; a caller must still confirm exact patch landing.
    """
    if (type(bases) is not dict or set(bases) != set(SITES)
            or type(donors) is not dict or set(donors) != set(SITES)
            or any(type(donors[site]) is not dict or set(donors[site]) != set(DONORS) for site in SITES)):
        raise ValueError('expected exactly two sites and three donor directions')
    base_values = {site: _values(bases[site]) for site in SITES}
    directions, distances = {}, {}
    for site in SITES:
        for cell in DONORS:
            donor = _values(donors[site][cell])
            direction = tuple(math.fsum((a, -b)) for a, b in zip(donor, base_values[site]))
            distance = _norm(direction)
            if distance < MIN_RADIUS:
                raise ValueError('donor displacement is below the fixed numerical floor')
            directions[(site, cell)] = direction
            distances[(site, cell)] = distance
    radius = min(distances.values())
    result = {'radius': radius, 'sites': {}}
    for site in SITES:
        base, base_norm = base_values[site], _norm(base_values[site])
        actions = {}
        for cell in DONORS:
            distance = distances[(site, cell)]
            fraction = radius / distance
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError('invalid interpolation fraction')
            values = (math.fsum((value, fraction * direction))
                      for value, direction in zip(base, directions[(site, cell)]))
            actions[cell] = {'distance': distance, 'fraction': fraction,
                             **_delivered(base, values, radius, base_norm)}
        magnitude = radius / math.sqrt(float(WIDTH))
        artificial = (math.fsum((value, magnitude if sign == 1 else -magnitude))
                      for value, sign in zip(base, _signs(row_id, site)))
        actions['artificial'] = _delivered(base, artificial, radius, base_norm)
        result['sites'][site] = actions
    return result
