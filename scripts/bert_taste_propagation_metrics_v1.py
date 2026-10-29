"""Model-free, fixed-order propagation and full-output metrics."""
import math
import struct

from scripts import bert_taste_template_geometry_v1 as geometry

FLOOR = 1e-12


def binary64_probe():
    """Fail closed when the current thread flushes binary64 subnormals."""
    positive_bits = bytes.fromhex('0100000000000000')
    negative_bits = bytes.fromhex('0100000000000080')
    positive = struct.unpack('<d', positive_bits)[0]
    negative = struct.unpack('<d', negative_bits)[0]
    if (not positive > 0.0 or not negative < 0.0
            or struct.pack('<d', positive * 1.0) != positive_bits
            or struct.pack('<d', negative * 1.0) != negative_bits
            or struct.pack('<d', float('5e-324')) != positive_bits
            or struct.pack('<d', float('-5e-324')) != negative_bits):
        raise RuntimeError('binary64 subnormal preservation is unavailable')
    return 'binary64-subnormals-preserved'


def _pair(left, right):
    left, right = geometry.values(left), geometry.values(right)
    if len(left) != len(right):
        raise ValueError('vector widths differ')
    return left, right


def odd_even(plus, minus, clean):
    plus, minus = _pair(plus, minus)
    _, clean = _pair(plus, clean)
    odd = tuple(math.fsum((p, -m)) / 2 for p, m in zip(plus, minus))
    even = tuple(math.fsum((p, m, -2 * c)) / 2 for p, m, c in zip(plus, minus, clean))
    return odd, even


def center(vector):
    vector = geometry.values(vector)
    mean = math.fsum(vector) / len(vector)
    return tuple(math.fsum((value, -mean)) for value in vector)


def projection(vector, direction):
    vector, direction = _pair(vector, direction)
    vector_norm, direction_norm = geometry.norm(vector), geometry.norm(direction)
    if direction_norm < FLOOR:
        return {'coefficient': None, 'coefficient_status': 'clean direction below floor',
                'cosine': None, 'cosine_status': 'clean direction below floor',
                'vector_norm': vector_norm, 'direction_norm': direction_norm, 'residual_norm': None}
    dot = math.fsum(a * b for a, b in zip(vector, direction))
    coefficient = dot / math.fsum(value ** 2 for value in direction)
    residual = tuple(math.fsum((a, -coefficient * b)) for a, b in zip(vector, direction))
    return {'coefficient': coefficient, 'coefficient_status': 'defined',
            'cosine': None if vector_norm < FLOOR else dot / (vector_norm * direction_norm),
            'cosine_status': 'response below floor' if vector_norm < FLOOR else 'defined',
            'vector_norm': vector_norm, 'direction_norm': direction_norm,
            'residual_norm': geometry.norm(residual)}


def softmax(logits):
    logits = geometry.values(logits)
    maximum = max(logits)
    values = tuple(math.exp(value - maximum) for value in logits)
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('invalid softmax normalization')
    return tuple(value / total for value in values)


def total_variation(left, right):
    left, right = _pair(left, right)
    p, q = softmax(left), softmax(right)
    return math.fsum(abs(a - b) for a, b in zip(p, q)) / 2


def complete_mean(values):
    if type(values) not in (tuple, list) or not values:
        raise ValueError('a nonempty set of required values is needed')
    if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
           for value in values):
        raise ValueError('aggregate values must be finite')
    missing = sum(value is None for value in values)
    if missing:
        return {'value': None, 'missing_count': missing, 'reason': 'required value unavailable'}
    return {'value': math.fsum(values) / len(values), 'missing_count': 0, 'reason': None}
