"""Fixed-dose natural donor-patch arithmetic; model-independent and fail-closed.

Round-mode and integer binary32 conversion follow the independently reviewed
public distractor-repair core.  Exact rational certificates check the fixed
predeclared fidelity limits; they do not attest model execution or landing.
"""
import ctypes
import math
import os
import struct
from fractions import Fraction


class InvalidPlan(ValueError):
    pass


def ensure_round_to_nearest():
    if os.name == 'nt':
        control = ctypes.CDLL('msvcrt')._controlfp
        control.argtypes = [ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_uint
        if control(0, 0) & 0x300:
            raise InvalidPlan('floating-point rounding mode is not nearest-even')
    else:
        try:
            getround = ctypes.CDLL(None).fegetround
            getround.argtypes, getround.restype = [], ctypes.c_int
            mode = getround()
        except (AttributeError, OSError) as error:
            raise InvalidPlan('cannot inspect floating-point mode') from error
        if mode != 0:
            raise InvalidPlan('floating-point rounding mode is not nearest-even')
    one, half = 1.0, float.fromhex('0x1p-53')
    next_one = float.fromhex('0x1.0000000000001p+0')
    if (one + half != one
            or next_one + half != float.fromhex('0x1.0000000000002p+0')
            or one - float.fromhex('0x1p-54') != one
            or next_one * 1.5 != float.fromhex('0x1.8000000000002p+0')
            or one / 10.0 != float.fromhex('0x1.999999999999ap-4')
            or math.sqrt(2.0) != float.fromhex('0x1.6a09e667f3bcdp+0')):
        raise InvalidPlan('binary64 rounding probe failed')


def _round_shift_even(value, shift):
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1))


def f64_to_f32_bits(value):
    if type(value) is not float or not math.isfinite(value):
        raise InvalidPlan('expected finite binary64')
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
    sign, exponent, fraction = bits >> 31, (bits >> 23) & 0xff, bits & 0x7fffff
    if exponent == 0xff:
        raise InvalidPlan('nonfinite float32')
    significand = fraction if exponent == 0 else (1 << 23) | fraction
    value = math.ldexp(float(significand), -149 if exponent == 0 else exponent - 150)
    return -value if sign else value


def _f32(raw):
    if type(raw) is not bytes or len(raw) == 0 or len(raw) % 4:
        raise InvalidPlan('expected a nonempty binary32 vector')
    return [f32_bits_to_f64(bits) for (bits,) in struct.iter_unpack('<I', raw)]


def _norm(values):
    if any(type(x) is not float or not math.isfinite(x) for x in values):
        raise InvalidPlan('nonfinite displacement')
    try:
        value = math.sqrt(math.fsum(x * x for x in values))
    except (OverflowError, ValueError) as error:
        raise InvalidPlan('nonfinite displacement norm') from error
    if not math.isfinite(value) or value <= 2.0 ** -20:
        raise InvalidPlan('displacement norm is nonfinite or at/below floor')
    return value


def _as_ratio(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def certificate(q, before_raw, after_raw, intended):
    """Return exact squared metrics and inclusive fixed acceptance checks."""
    ensure_round_to_nearest()
    if type(q) is not float or not math.isfinite(q) or q <= 2.0 ** -20:
        raise InvalidPlan('invalid requested norm')
    before, after = _f32(before_raw), _f32(after_raw)
    if len(before) != len(after) or len(before) != len(intended):
        raise InvalidPlan('vector widths differ')
    if any(type(x) is not float or not math.isfinite(x) for x in intended):
        raise InvalidPlan('nonfinite intended displacement')
    qf = Fraction.from_float(q)
    bf, af = [Fraction.from_float(x) for x in before], [Fraction.from_float(x) for x in after]
    hf = [Fraction.from_float(x) for x in intended]
    d2 = sum((x * x for x in hf), Fraction(0))
    r2 = sum(((a - b) ** 2 for a, b in zip(af, bf)), Fraction(0))
    e2 = sum(((a - b - h) ** 2 for a, b, h in zip(af, bf, hf)), Fraction(0))
    intended_low = (qf * Fraction(10**12 - 1, 10**12)) ** 2
    intended_high = (qf * Fraction(10**12 + 1, 10**12)) ** 2
    realized_low = (qf * Fraction(4999, 5000)) ** 2
    realized_high = (qf * Fraction(5001, 5000)) ** 2
    error_high = (qf / 5000) ** 2
    checks = {
        'intended_norm': intended_low <= d2 <= intended_high,
        'realized_norm': realized_low <= r2 <= realized_high,
        'displacement_error': e2 <= error_high,
        'delivered': before_raw != after_raw,
    }
    ensure_round_to_nearest()
    return {'D2': _as_ratio(d2), 'R2': _as_ratio(r2), 'E2': _as_ratio(e2),
            'limits': {'intended_low': _as_ratio(intended_low), 'intended_high': _as_ratio(intended_high),
                       'realized_low': _as_ratio(realized_low), 'realized_high': _as_ratio(realized_high),
                       'error_high': _as_ratio(error_high)}, 'checks': checks,
            'pass': all(checks.values())}


def plan_candidates(before_raw, direct_raw, lexical_raw, other_noun_raw):
    """Plan Q1–Q4 from prevalidated saved vectors without loading a model.

    Returns the planned binary32 replacement, full binary64 displacement and
    exact rational certificate per candidate.  A false certificate must be
    retained and makes the enclosing phase stop before its corresponding call.
    """
    ensure_round_to_nearest()
    before = _f32(before_raw)
    direct, lexical, other = (_f32(x) for x in (direct_raw, lexical_raw, other_noun_raw))
    if not (len(before) == len(direct) == len(lexical) == len(other)):
        raise InvalidPlan('donor and receiver widths differ')
    natural = [c - b for b, c in zip(before, direct)]
    q = _norm(natural)
    lexical_d = [c - b for b, c in zip(before, lexical)]
    other_d = [c - b for b, c in zip(before, other)]
    lexical_scale, other_scale = q / _norm(lexical_d), q / _norm(other_d)
    if not math.isfinite(lexical_scale) or not math.isfinite(other_scale):
        raise InvalidPlan('nonfinite scale')
    hs = {
        'Q1': [lexical_scale * x for x in lexical_d],
        'Q2': natural,
        'Q3': natural[-1:] + natural[:-1],
        'Q4': [other_scale * x for x in other_d],
    }
    result = {}
    for key, h in hs.items():
        if any(not math.isfinite(x) for x in h):
            raise InvalidPlan('nonfinite intended displacement')
        if key == 'Q2':
            replacement = direct_raw
        else:
            sums = [b + v for b, v in zip(before, h)]
            if any(not math.isfinite(x) for x in sums):
                raise InvalidPlan('nonfinite addition')
            bits = [f64_to_f32_bits(x) for x in sums]
            if any((x & 0x7f800000) == 0x7f800000 for x in bits):
                raise InvalidPlan('binary32 replacement overflow')
            replacement = b''.join(struct.pack('<I', x) for x in bits)
        result[key] = {
            'replacement_f32': replacement,
            'displacement_f64': b''.join(struct.pack('<d', x) for x in h),
            'certificate': certificate(q, before_raw, replacement, h),
        }
    ensure_round_to_nearest()
    return {'q': q, 'width': len(before), 'candidates': result}
