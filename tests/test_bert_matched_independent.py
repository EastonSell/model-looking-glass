"""Independent fabricated-only numerical checks; no model or study cases."""
from fractions import Fraction
import math
import random
import struct

import pytest

from looking_glass import bert_matched_numeric as numeric


def _power_two(exponent):
    return Fraction(1 << exponent, 1) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _nearest_even(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    if 2 * remainder > value.denominator or (2 * remainder == value.denominator and quotient & 1):
        quotient += 1
    return quotient


def _fraction_oracle(raw64):
    sign = (raw64 >> 32) & 0x80000000
    exponent = (raw64 >> 52) & 0x7ff
    mantissa = raw64 & ((1 << 52) - 1)
    if exponent == 0x7ff:
        raise ValueError('nonfinite binary64')
    if exponent:
        mantissa |= 1 << 52
        power = exponent - 1023 - 52
    else:
        power = -1074
    if not mantissa:
        return sign
    value = Fraction(mantissa) * _power_two(power)
    top = mantissa.bit_length() - 1 + power
    if top < -126:
        return sign | _nearest_even(value * _power_two(149))
    significand = _nearest_even(value * _power_two(23 - top))
    if significand == 1 << 24:
        top += 1
        significand >>= 1
    if top > 127:
        return sign | 0x7f800000
    return sign | ((top + 127) << 23) | (significand - (1 << 23))


def _value(raw64):
    return struct.unpack('<d', raw64.to_bytes(8, 'little'))[0]


def test_converter_matches_fraction_oracle_in_both_denormal_modes():
    import torch
    random_source = random.Random(904711)
    boundaries = [0.0, -0.0, 2.0**-151, -(2.0**-151), 2.0**-150, 3.0 * 2.0**-150,
                  2.0**-126 - 2.0**-150, 1.0 + 2.0**-24,
                  1.0 + 3.0 * 2.0**-24, (2.0 - 2.0**-24) * 2.0**127]
    samples = [int.from_bytes(struct.pack('<d', value), 'little') for value in boundaries]
    samples.extend(random_source.getrandbits(64) for _ in range(20000))
    samples = [bits for bits in samples if (bits >> 52) & 0x7ff != 0x7ff]
    try:
        for flush in (False, True):
            torch.set_flush_denormal(flush)
            for bits in samples:
                assert numeric.f64_to_f32_bits(_value(bits)) == _fraction_oracle(bits)
    finally:
        torch.set_flush_denormal(False)


def test_decode_preserves_subnormal_and_signed_zero_under_flush():
    import torch
    try:
        torch.set_flush_denormal(True)
        assert numeric.f32_bits_to_f64(1) == math.ldexp(1.0, -149)
        assert numeric.f32_bits_to_f64(0x807fffff) == -math.ldexp(float(0x7fffff), -149)
        assert math.copysign(1.0, numeric.f32_bits_to_f64(0x80000000)) == -1.0
    finally:
        torch.set_flush_denormal(False)


def _vector(values):
    return struct.pack('<768f', *(list(values) + [0.0] * (768 - len(values))))


def test_common_minimum_and_all_donor_landings_are_independently_calculable():
    bases = {'block0': _vector([]), 'block2': _vector([])}
    donors = {'block0': {'A_opposite': _vector([1]), 'B_same': _vector([0, 2]),
                         'B_opposite': _vector([0, 0, 4])},
              'block2': {'A_opposite': _vector([0, 0, 0, .5]), 'B_same': _vector([0, 0, 0, 0, 1]),
                         'B_opposite': _vector([0, 0, 0, 0, 0, 2])}}
    result = numeric.plan_case('fabricated-row', bases, donors)
    assert result['radius'] == .5
    for site, expected in {'block0': (1, 2, 4), 'block2': (.5, 1, 2)}.items():
        for position, (label, distance) in enumerate(zip(numeric.DONORS, expected)):
            action = result['sites'][site][label]
            assert action['distance'] == distance
            assert action['fraction'] == .5 / distance
            unpacked = struct.unpack('<768f', action['value'])
            index = position + (0 if site == 'block0' else 3)
            assert unpacked[index] == .5 and sum(value != 0 for value in unpacked) == 1
            assert action['realized_norm'] == .5
        assert abs(result['sites'][site]['artificial']['realized_norm'] - .5) <= 1e-6


@pytest.mark.parametrize('bad', (0.0, 0.00001, float('nan'), float('inf')))
def test_small_or_nonfinite_donor_distance_cannot_succeed(bad):
    bases = {site: _vector([]) for site in numeric.SITES}
    donors = {site: {label: _vector([1.0]) for label in numeric.DONORS} for site in numeric.SITES}
    donors['block0']['B_same'] = _vector([bad])
    with pytest.raises(ValueError):
        numeric.plan_case('fabricated-row', bases, donors)


def test_float32_floor_uses_the_saved_bits_not_the_rounded_decimal_label():
    bases = {site: _vector([]) for site in numeric.SITES}
    floor_below = struct.pack('<768I', 0x38d1b717, *([0] * 767))
    floor_above = struct.pack('<768I', 0x38d1b718, *([0] * 767))
    assert numeric.f32_bits_to_f64(0x38d1b717) < 1e-4
    assert numeric.f32_bits_to_f64(0x38d1b718) > 1e-4
    donors = {site: {label: floor_above for label in numeric.DONORS} for site in numeric.SITES}
    donors['block0']['A_opposite'] = floor_below
    with pytest.raises(ValueError):
        numeric.plan_case('fabricated-row', bases, donors)
    donors['block0']['A_opposite'] = floor_above
    assert numeric.plan_case('fabricated-row', bases, donors)['radius'] > 1e-4


def test_artificial_rounding_that_loses_a_large_coordinate_fails_closed():
    # A tiny artificial displacement at a huge base coordinate rounds away;
    # dropping 1/768 of its squared length exceeds this case's fixed tolerance.
    bases = {site: _vector([2.0**60]) for site in numeric.SITES}
    donors = {site: {label: _vector([2.0**60, 0.5]) for label in numeric.DONORS}
              for site in numeric.SITES}
    with pytest.raises(ValueError, match='tolerance'):
        numeric.plan_case('fabricated-row', bases, donors)
