"""Bit-defined, model-free construction of one matched orthogonal control.

The caller owns the group radius, CPU numeric-mode probes, actual landing
verification and evidence custody.  A source vector is returned as exact
little-endian binary64 bytes so a saved result can be reconstructed.
"""

from dataclasses import dataclass
import math
import random
import struct

from .bert_matched_numeric import f32_bits_to_f64, f64_to_f32_bits


def orthogonal_seed(unit, subject, instruction, receiver):
    if (type(unit) is not int or not 0 <= unit < 4
            or type(subject) is not int or not 0 <= subject < 3
            or type(instruction) is not int or not 0 <= instruction < 2
            or type(receiver) is not int or not 0 <= receiver < 4):
        raise ValueError("invalid planned receiver identity")
    return 20261029 + 24 * unit + 8 * subject + 4 * instruction + receiver


def gaussian_source(seed, width):
    if (type(seed) is not int or not 0 <= seed < 2**64
            or type(width) is not int or not 1 <= width <= 512):
        raise ValueError("invalid Gaussian source identity")
    rng = random.Random(seed)
    values = tuple(rng.gauss(0, 1) for _ in range(width))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite Gaussian source")
    return struct.pack("<" + str(width) + "d", *values)


def _norm(values):
    try:
        result = math.sqrt(math.fsum(value ** 2 for value in values))
    except (OverflowError, ValueError) as error:
        raise ValueError("nonfinite vector norm") from error
    if not math.isfinite(result):
        raise ValueError("nonfinite vector norm")
    return result


def _cosine(left, right, left_norm, right_norm):
    if left_norm == 0 or right_norm == 0:
        return None
    try:
        result = math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise ValueError("nonfinite cosine") from error
    if not math.isfinite(result):
        raise ValueError("nonfinite cosine")
    return result


@dataclass(frozen=True, slots=True)
class OrthogonalDirection:
    base: bytes
    donor: bytes
    source: bytes
    residual: bytes
    coefficient: float
    residual_norm: float
    donor_norm: float


def construct_orthogonal(base, donor, source):
    """Construct and retain one binary64 direction before either dose."""
    if (type(base) is not bytes or type(donor) is not bytes
            or len(base) != len(donor) or len(base) % 4 or not 4 <= len(base) <= 512 * 4
            or type(source) is not bytes or len(source) != 2 * len(base)):
        raise ValueError("invalid orthogonal inputs")
    own = tuple(f32_bits_to_f64(word[0]) for word in struct.iter_unpack("<I", base))
    other = tuple(f32_bits_to_f64(word[0]) for word in struct.iter_unpack("<I", donor))
    gaussian = tuple(value[0] for value in struct.iter_unpack("<d", source))
    if any(not math.isfinite(value) for value in gaussian):
        raise ValueError("nonfinite Gaussian source")
    direction = tuple(b - a for a, b in zip(own, other))
    donor_norm = _norm(direction)
    if donor_norm == 0:
        return {"status": "unavailable", "failure": "zero donor distance", "direction": None}
    try:
        denominator = math.fsum(value * value for value in direction)
        coefficient = math.fsum(g * d for g, d in zip(gaussian, direction)) / denominator
        residual = tuple(g - coefficient * d for g, d in zip(gaussian, direction))
        residual_norm = _norm(residual)
    except (OverflowError, ValueError, ZeroDivisionError):
        return {"status": "unavailable", "failure": "nonfinite orthogonal projection", "direction": None}
    if not math.isfinite(coefficient) or any(not math.isfinite(value) for value in residual):
        return {"status": "unavailable", "failure": "nonfinite orthogonal projection", "direction": None}
    if residual_norm == 0:
        return {"status": "unavailable", "failure": "zero orthogonal residual", "direction": None}
    raw_residual = struct.pack("<" + str(len(residual)) + "d", *residual)
    return {"status": "ready", "failure": None,
            "direction": OrthogonalDirection(base, donor, source, raw_residual,
                                              coefficient, residual_norm, donor_norm)}


def land_orthogonal(plan, radius):
    """Round a previously constructed direction at one positive radius.

    Keep the same immutable plan for low and high dose.  Finite but
    unmatchable landings are unavailable; nonfinite rounded bits are returned
    with an incomplete disposition so the owner can retain them.
    """
    if (type(plan) is not OrthogonalDirection or type(radius) is not float
            or not math.isfinite(radius) or not 0 < radius <= plan.donor_norm):
        raise ValueError("invalid orthogonal landing request")
    own = tuple(f32_bits_to_f64(word[0]) for word in struct.iter_unpack("<I", plan.base))
    other = tuple(f32_bits_to_f64(word[0]) for word in struct.iter_unpack("<I", plan.donor))
    residual = tuple(value[0] for value in struct.iter_unpack("<d", plan.residual))
    direction = tuple(b - a for a, b in zip(own, other))
    if (len(own) != len(residual) or _norm(direction) != plan.donor_norm
            or _norm(residual) != plan.residual_norm):
        raise ValueError("orthogonal direction differs from its recorded values")
    scale = radius / plan.residual_norm
    if not math.isfinite(scale):
        return {"status": "unavailable", "failure": "nonfinite orthogonal scale", "value": None}
    words = tuple(f64_to_f32_bits(float(a + scale * value)) for a, value in zip(own, residual))
    value = struct.pack("<" + str(len(words)) + "I", *words)
    tolerance = min(max(1e-5, 1e-5 * radius), 0.01 * radius)
    if any((word & 0x7f800000) == 0x7f800000 for word in words):
        return {"status": "incomplete", "failure": "nonfinite rounded landing",
                "value": value, "radius": radius, "achieved_norm": None,
                "intended_cosine": None, "donor_cosine": None,
                "tolerance": tolerance, "coefficient": plan.coefficient,
                "residual_norm": plan.residual_norm}
    landed = tuple(f32_bits_to_f64(word) for word in words)
    achieved = tuple(a - b for a, b in zip(landed, own))
    achieved_norm = _norm(achieved)
    intended_cosine = _cosine(achieved, residual, achieved_norm, plan.residual_norm)
    donor_cosine = _cosine(achieved, direction, achieved_norm, plan.donor_norm)
    failure = None
    if achieved_norm == 0:
        failure = "zero achieved displacement"
    elif abs(achieved_norm - radius) > tolerance:
        failure = "landing norm outside tolerance"
    elif intended_cosine is None or intended_cosine < 0.99:
        failure = "landing cosine below threshold"
    elif donor_cosine is None or abs(donor_cosine) > 1e-4:
        failure = "landing is not orthogonal within tolerance"
    return {"status": "matchable" if failure is None else "unavailable", "failure": failure,
            "value": value, "radius": radius, "achieved_norm": achieved_norm,
            "intended_cosine": intended_cosine, "donor_cosine": donor_cosine,
            "tolerance": tolerance, "coefficient": plan.coefficient, "residual_norm": plan.residual_norm}


def plan_orthogonal(base, donor, source, radius):
    """Convenience for one radius; production should reuse one constructed plan."""
    if type(radius) is not float or not math.isfinite(radius) or radius <= 0:
        raise ValueError("invalid orthogonal radius")
    constructed = construct_orthogonal(base, donor, source)
    if constructed["status"] != "ready":
        return {"status": constructed["status"], "failure": constructed["failure"], "value": None}
    return land_orthogonal(constructed["direction"], radius)
