"""Stored binary32 decoding and fixed tolerance for the agreement study.

Integer field decoding avoids FTZ-sensitive float32 loads and casts. This module
does not verify how the bytes were captured, the original tensor shape, source
identity, model behavior or intervention landing.
"""

import numpy as np


class NumericInvalid(ValueError):
    """Input bytes cannot represent the bounded finite float32 payload."""


_MAX_BYTES = 1024 * 1024
_ATOL = 1e-5
_RTOL = 1e-5


def decode_f32_le(raw):
    """Decode finite little-endian IEEE float32 fields into binary64 values.

    For exponent zero, the magnitude is ``fraction * 2**-149``; otherwise it
    is ``(2**23 + fraction) * 2**(exponent-150)``. Even float32 subnormals are
    normal binary64 values, so the conversion performs no binary32 arithmetic.
    Signed zero is retained. Empty, nonfinite or over-1-MiB payloads are errors.
    """
    if type(raw) is not bytes or not raw or len(raw) % 4 or len(raw) > _MAX_BYTES:
        raise NumericInvalid('expected bounded nonempty little-endian float32 bytes')
    bits = np.frombuffer(raw, dtype='<u4')
    exponent = ((bits >> np.uint32(23)) & np.uint32(255)).astype(np.int16)
    if np.any(exponent == 255):
        raise NumericInvalid('nonfinite stored float32 value')
    fraction = bits & np.uint32(0x7fffff)
    significand = (fraction | np.where(exponent == 0, np.uint32(0),
                                        np.uint32(1 << 23))).astype(np.float64)
    powers = np.where(exponent == 0, -149, exponent - 150).astype(np.int16)
    magnitudes = np.ldexp(significand, powers)
    return np.where((bits >> np.uint32(31)) != 0, -magnitudes, magnitudes)


def compare_f32_le(left, right):
    """Compare every stored value with the fixed 1e-5 absolute/relative rule."""
    if type(left) is not bytes or type(right) is not bytes or len(left) != len(right):
        raise NumericInvalid('float32 byte lengths must match')
    a = decode_f32_le(left)
    b = decode_f32_le(right)
    difference = np.abs(a - b)
    threshold = _ATOL + _RTOL * np.maximum(np.abs(a), np.abs(b))
    return {'bit_equal': left == right,
            'max_abs_difference_binary64': float(np.max(difference)),
            'within_tolerance': bool(np.all(difference <= threshold))}
