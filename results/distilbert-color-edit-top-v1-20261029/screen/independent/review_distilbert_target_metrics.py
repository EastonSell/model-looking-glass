"""Model-free, full-vocabulary metrics for a targeted masked-token change.

This helper deliberately reads no checkpoint or study cases. A separate
binding check must establish the source, row identities and saved-logit bytes.
"""
import math
import struct
from decimal import Decimal, localcontext


def require_denormal_support():
    positive = bytes.fromhex('0100000000000000')
    negative = bytes.fromhex('0100000000000080')
    pos = struct.unpack('<d', positive)[0]
    neg = struct.unpack('<d', negative)[0]
    if not (pos > 0 and neg < 0 and struct.pack('<d', pos * 1.) == positive
            and struct.pack('<d', neg * 1.) == negative
            and struct.pack('<d', float('5e-324')) == positive
            and struct.pack('<d', float('-5e-324')) == negative
            and struct.unpack('<f', bytes.fromhex('01000000'))[0] == 2. ** -149
            and struct.unpack('<f', bytes.fromhex('01000080'))[0] == -(2. ** -149)):
        raise ValueError('binary32/64 subnormals are not preserved')


def decode_f32(raw):
    require_denormal_support()
    if type(raw) is not bytes or not raw or len(raw) % 4:
        raise ValueError('expected nonempty f32le vector')
    values = struct.unpack('<' + str(len(raw) // 4) + 'f', raw)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite logits')
    return values


def _values(logits):
    if not logits or not all(type(x) is float and math.isfinite(x) for x in logits):
        raise ValueError('expected finite floating-point logits')
    return logits


def _ids(ids, width):
    if not ids or any(type(i) is not int or i < 0 or i >= width for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('expected distinct in-range token IDs')
    return tuple(ids)


def distribution(logits):
    _values(logits)
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    log_normalizer = math.log(math.fsum(weights))
    return [(x - peak) - log_normalizer for x in logits]


def summarize(logits, target_id, original_id, accepted_ids):
    require_denormal_support()
    _values(logits)
    target_id, original_id = _ids((target_id, original_id), len(logits))
    accepted = _ids(accepted_ids, len(logits))
    logp = distribution(logits)
    highest = max(logits)
    ties = [i for i, value in enumerate(logits) if value == highest]
    top = ties[0] if len(ties) == 1 else None
    def rank(token):
        return 1 + sum(value > logits[token] or (value == logits[token] and i < token)
                       for i, value in enumerate(logits))
    # Compute accepted-set NLL with its own peak so rare valid answers do not
    # disappear when exponentials relative to the global peak underflow.
    subset_peak = max(logits[i] for i in accepted)
    subset_log_sum = math.log(math.fsum(math.exp(logits[i] - subset_peak) for i in accepted))
    global_peak = max(logits)
    global_log_sum = math.log(math.fsum(math.exp(x - global_peak) for x in logits))
    accepted_nll = (global_peak - subset_peak) + (global_log_sum - subset_log_sum)
    second = sorted(logits, reverse=True)[1] if len(logits) > 1 else None
    return {'unique_top_id': top, 'tie_count': len(ties), 'tie_first_id': ties[0],
            'top_gap': highest - second if second is not None else None,
            'top_accepted': None if top is None else top in accepted,
            'target_rank': rank(target_id), 'original_rank': rank(original_id),
            'target_log_probability': logp[target_id],
            'original_log_probability': logp[original_id],
            'target_probability': math.exp(logp[target_id]),
            'original_probability': math.exp(logp[original_id]),
            'target_minus_original_margin': logits[target_id] - logits[original_id],
            'accepted_set_nll': accepted_nll}


def nonnegative_kl(raw):
    if not math.isfinite(raw) or raw < -1e-12:
        raise ValueError('materially negative or nonfinite KL')
    return max(0., raw)


def compare(before, after):
    require_denormal_support()
    _values(before)
    _values(after)
    if len(before) != len(after):
        raise ValueError('vocabulary width mismatch')
    lp, lq = distribution(before), distribution(after)
    shifts = [b - a for a, b in zip(before, after)]
    if before == after:
        return {'total_variation': 0., 'kl_before_after_raw': 0., 'kl_before_after': 0.}
    if min(min(lp), min(lq)) < -700. or max(shifts) - min(shifts) < 1e-6:
        # Below this point a probability may become subnormal/zero before a
        # log-ratio multiplier is applied, and its aggregate mass can matter.
        # Small nonuniform shifts can also vanish when added to log-normalizers.
        return _compare_high_precision(before, after)
    p, q = [math.exp(x) for x in lp], [math.exp(x) for x in lq]
    raw_kl = math.fsum(a * (x - y) for a, x, y in zip(p, lp, lq) if a)
    tv = math.fsum(abs(a - b) for a, b in zip(p, q)) / 2
    if tv < 1e-4 or abs(raw_kl) < 1e-4:
        # Small distribution changes can hide in a normalizer rounded to 1.
        return _compare_high_precision(before, after)
    return {'total_variation': tv,
            'kl_before_after_raw': raw_kl, 'kl_before_after': nonnegative_kl(raw_kl)}


def _compare_high_precision(before, after):
    """Rare-probability fallback, retaining mass below individual f64 range.

    Relative exponents below -1200 are dropped: with finite f32 logits and a
    vocabulary of at most 30522, even their largest possible contribution to
    TV or KL is below binary64's representable range. The caller must bind the
    source to actual f32 rows; this is not an arbitrary-binary64 guarantee.
    """
    if len(before) > 30522:
        raise ValueError('high-precision fallback exceeds frozen vocabulary')
    if any(float(struct.unpack('<f', struct.pack('<f', x))[0]) != x for x in (*before, *after)):
        raise ValueError('high-precision fallback requires exact f32 logits')
    smallest_log_probability = min(min(distribution(before)), min(distribution(after)))
    with localcontext() as ctx:
        ctx.prec = min(420, max(120, math.ceil(max(0., -smallest_log_probability) / math.log(10.)) + 100))
        a = [Decimal.from_float(x) for x in before]
        b = [Decimal.from_float(x) for x in after]
        ap, bp = max(a), max(b)
        aw = [(x - ap).exp() if x - ap >= -1200 else Decimal(0) for x in a]
        bw = [(x - bp).exp() if x - bp >= -1200 else Decimal(0) for x in b]
        az, bz = sum(aw), sum(bw)
        alz, blz = az.ln(), bz.ln()
        pa, pb = [x / az for x in aw], [x / bz for x in bw]
        tv = sum(abs(x - y) for x, y in zip(pa, pb)) / 2
        kl = sum(p * ((x - ap - alz) - (y - bp - blz))
                 for p, x, y in zip(pa, a, b) if p)
        raw_kl = float(kl)
    return {'total_variation': float(tv), 'kl_before_after_raw': raw_kl,
            'kl_before_after': nonnegative_kl(raw_kl)}


def target_gains(before, after, target_id, original_id):
    require_denormal_support()
    _values(before)
    _values(after)
    if len(before) != len(after):
        raise ValueError('vocabulary width mismatch')
    target_id, original_id = _ids((target_id, original_id), len(before))
    before_lp, after_lp = distribution(before), distribution(after)
    return {'margin_gain': math.fsum((after[target_id], -after[original_id],
                                      -before[target_id], before[original_id])),
            'target_log_probability_gain': after_lp[target_id] - before_lp[target_id]}


def ideal_two_logit_match(before, edited, target_id, original_id):
    """Exact-real target probability and margin match, before f32 rounding.

    All other logits are retained from ``before``. Real parameter addition must
    still be rounded, replayed and checked separately.
    """
    require_denormal_support()
    _values(before)
    _values(edited)
    if len(before) != len(edited) or len(before) < 3:
        raise ValueError('expected equal vocabularies with other tokens')
    target_id, original_id = _ids((target_id, original_id), len(before))
    other = [i for i in range(len(before)) if i not in (target_id, original_id)]
    x_peak = max(before[i] for i in other)
    y_peak = max(edited[i] for i in other)
    x_sum = math.log(math.fsum(math.exp(before[i] - x_peak) for i in other))
    y_sum = math.log(math.fsum(math.exp(edited[i] - y_peak) for i in other))
    shift = math.fsum((x_peak, -y_peak, x_sum, -y_sum))
    result = list(before)
    result[target_id] = math.fsum((edited[target_id], shift))
    result[original_id] = math.fsum((edited[original_id], shift))
    return tuple(result)
