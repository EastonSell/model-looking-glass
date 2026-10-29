"""CPU-only primitives for the frozen known-injection number-study protocol.

These arithmetic and instrumentation primitives do not allocate a split,
commit a site, or authorize execution of an experiment by themselves.
"""
import hashlib
import struct

import torch


_SEED = 20261028
_WIDTH = 768
_POSITIVE_DIRECTION_BITS = 0x3C93CD3A
_NEGATIVE_DIRECTION_BITS = 0xBC93CD3A


def _finite_components(bits):
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("float32 bits must be a uint32 integer")
    exponent, fraction = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("float32 operands must be finite")
    mantissa = fraction if exponent == 0 else (1 << 23) | fraction
    power = -149 if exponent == 0 else exponent - 150
    return (-mantissa if bits & 0x80000000 else mantissa), power


def add_float32_bits(left, right):
    """Return correctly rounded IEEE binary32 sum as bits using integer math.

    Operands must be finite and rounding is nearest, ties-to-even. Overflow is
    represented as the appropriate infinity; a study rejects nonfinite output.
    Signed zeros follow that rounding mode. No host float32 conversion occurs.
    """
    a, a_power = _finite_components(left)
    b, b_power = _finite_components(right)
    power = min(a_power, b_power)
    total = (a << (a_power - power)) + (b << (b_power - power))
    if total == 0:
        return 0x80000000 if left == right == 0x80000000 else 0
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    exponent = magnitude.bit_length() - 1 + power
    if exponent < -126:
        # Every finite binary32 number is an exact multiple of 2**-149, so a
        # subnormal sum is also exact in that unit.
        return sign | (magnitude << (power + 149))
    shift = magnitude.bit_length() - 24
    if shift > 0:
        retained, discarded = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        if discarded > halfway or (discarded == halfway and retained & 1):
            retained += 1
    else:
        retained = magnitude << -shift
    if retained == 1 << 24:
        retained >>= 1
        exponent += 1
    if exponent > 127:
        return sign | 0x7F800000
    return sign | ((exponent + 127) << 23) | (retained & 0x7FFFFF)


def injection_direction_bits():
    """Return the protocol's 768 exact signed float32 component bit patterns."""
    signs = []
    for chunk in range((_WIDTH + 255) // 256):
        digest = hashlib.sha256(b"MLG-v1-direction\0" + _SEED.to_bytes(8, "big") +
                                chunk.to_bytes(4, "big")).digest()
        for byte in digest:
            signs.extend(_POSITIVE_DIRECTION_BITS if byte & (1 << bit) else _NEGATIVE_DIRECTION_BITS
                         for bit in range(7, -1, -1))
    return tuple(signs[:_WIDTH])


def verify_cpu_float32_addition():
    """Reject a CPU whose current float32 addition does not match the protocol.

    The sentinels include subnormal input/output and nearest-even halfway cases;
    passing these cases is a runtime check, not a proof about every CPU operation.
    """
    cases = ((0x00000001, 0x00000000), (0x00000001, 0x00000001),
             (0x00800000, 0x807FFFFF), (0x3F800000, 0x33800000),
             (0x3F800001, 0x33800000), (0x80000000, 0x80000000))
    signed = lambda value: value if value < 1 << 31 else value - (1 << 32)
    left = torch.tensor([signed(a) for a, _ in cases], dtype=torch.int32, device="cpu").view(torch.float32)
    right = torch.tensor([signed(b) for _, b in cases], dtype=torch.int32, device="cpu").view(torch.float32)
    actual = [value & 0xFFFFFFFF for value in torch.add(left, right).view(torch.int32).tolist()]
    expected = [add_float32_bits(a, b) for a, b in cases]
    if actual != expected:
        raise RuntimeError("CPU float32 addition does not match required rounding or denormal behavior")


def verify_injection_bits(pre_bytes, post_bytes):
    """Verify exact CPU post-add bits against the fixed protocol direction.

    Inputs are immutable little-endian snapshots of the complete private
    selected vector. The caller must retain the snapshots and verify the
    action's site, invocation, and cleanup separately.
    """
    if (type(pre_bytes) is not bytes or type(post_bytes) is not bytes or
            len(pre_bytes) != 4 * _WIDTH or len(post_bytes) != 4 * _WIDTH):
        raise ValueError("injection snapshots must contain 768 float32 bit patterns")
    before = struct.unpack("<768I", pre_bytes)
    after = struct.unpack("<768I", post_bytes)
    changed = 0
    for left, actual, delta in zip(before, after, injection_direction_bits()):
        _finite_components(actual)
        expected = add_float32_bits(left, delta)
        if ((expected >> 23) & 0xFF) == 0xFF or actual != expected:
            raise ValueError("injection did not produce the exact finite float32 sum")
        changed += actual != left
    if not changed:
        raise ValueError("injection did not change any selected element")
    return {"changed_coordinates": changed}
