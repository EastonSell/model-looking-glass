"""Stable paired softmax mass differences for finite float32 logit vectors.

This is an independent post-replay numerical check, not a model execution or
source/receipt audit. The error bound assumes IEEE binary64 round-to-nearest
and exp/expm1 errors no larger than 16 ulps; it is not a platform-independent
proof of the C math library. Ambiguous cancellation and underflow are reported
unresolved. Nonzero subnormal intermediate values also remain unresolved;
the relative-error bounds below require normal intermediate values. Exact
common shifts are identified without probability rounding.
"""
import math
import sys

EPSILON = sys.float_info.epsilon
SLACK = 4096 * EPSILON


def _normal_or_zero(x):
    return x == 0.0 or abs(x) >= sys.float_info.min


def pair_difference(before, after, target_ids):
    if len(before) != len(after) or len(before) < 2 or len(set(target_ids)) != 2:
        raise ValueError("two equally sized full-vocabulary vectors and two distinct IDs are required")
    ids = set(target_ids)
    if any(type(i) is not int or i < 0 or i >= len(before) for i in ids):
        raise ValueError("target ID out of range")
    a, b = [float(x) for x in before], [float(x) for x in after]
    if not all(math.isfinite(x) for x in a + b):
        raise ValueError("nonfinite logits")
    k = max(range(len(a)), key=a.__getitem__)
    relative = [math.fsum((y, -x, -b[k], a[k])) for x, y in zip(a, b)]
    if all(x == 0.0 for x in relative):
        return {"difference": 0.0, "absolute_error_bound": 0.0, "sign": "zero", "reason": "exact_common_shift"}
    if max(abs(x) for x in relative) > 600:
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "relative_shift_range"}
    weights = [math.exp(x - a[k]) for x in a]
    if not all(weights):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "base_weight_underflow"}
    if not all(map(_normal_or_zero, weights)):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "subnormal_base_weight"}
    denominator = math.fsum(weights)
    p = [x / denominator for x in weights]
    if not all(p):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "base_probability_underflow"}
    if not all(map(_normal_or_zero, p)):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "subnormal_base_probability"}
    target = math.fsum(p[i] for i in ids)
    other = math.fsum(x for i, x in enumerate(p) if i not in ids)
    shifts = [math.expm1(x) for x in relative]
    if not all(map(_normal_or_zero, [target, other, *shifts])):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "subnormal_coefficient_or_shift"}
    terms = []
    for i, (probability, shift) in enumerate(zip(p, shifts)):
        coefficient = other if i in ids else -target
        product = probability * coefficient
        term = product * shift
        if not _normal_or_zero(product) or not _normal_or_zero(term):
            return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "subnormal_numerator_term"}
        if product == 0.0 and probability != 0.0 and coefficient != 0.0:
            return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "product_underflow"}
        if term == 0.0 and probability != 0.0 and coefficient != 0.0 and shift != 0.0:
            return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "term_underflow"}
        terms.append(term)
    numerator = math.fsum(terms)
    absolute_terms = math.fsum(abs(x) for x in terms)
    numerator_error = SLACK * absolute_terms
    if (not all(map(_normal_or_zero, [numerator, absolute_terms, numerator_error]))
            or (absolute_terms > 0 and numerator_error == 0)):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "error_bound_underflow"}
    zterms = [probability * shift for probability, shift in zip(p, shifts)]
    if not all(map(_normal_or_zero, zterms)):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "subnormal_normalizer_term"}
    if any(x == 0.0 and p[i] != 0.0 and shifts[i] != 0.0 for i, x in enumerate(zterms)):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "normalizer_term_underflow"}
    z = 1.0 + math.fsum(zterms)
    z_error = SLACK * (1.0 + math.fsum(abs(x) for x in zterms))
    if not math.isfinite(z) or z <= z_error:
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "normalizer_uncertain"}
    value = numerator / z
    error = (numerator_error + abs(value) * z_error) / (z - z_error) + SLACK * abs(value)
    if not math.isfinite(value) or not math.isfinite(error):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "numeric_range"}
    if (not _normal_or_zero(value) or not _normal_or_zero(error)
            or (numerator != 0.0 and value == 0.0) or (absolute_terms != 0.0 and error == 0.0)):
        return {"difference": None, "absolute_error_bound": None, "sign": "unresolved", "reason": "subnormal_value_or_error"}
    if abs(numerator) <= numerator_error or abs(value) <= error:
        return {"difference": value, "absolute_error_bound": error, "sign": "unresolved", "reason": "near_cancellation"}
    return {"difference": value, "absolute_error_bound": error,
            "sign": "positive" if value > 0 else "negative", "reason": "stable_conditionally_bounded"}
