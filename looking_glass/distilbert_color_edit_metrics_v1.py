"""Numerics and predeclared decisions for the public masked-color study.

These functions consume saved, already validated full-vocabulary logit rows.
They do not load a model or authorize an experiment.
"""
import math
import struct


def require_denormals():
    """Fail if this thread flushes either signed saved-f32 or f64 sentinels."""
    tiny32 = struct.unpack('<f', bytes.fromhex('01000000'))[0]
    negative32 = struct.unpack('<f', bytes.fromhex('01000080'))[0]
    tiny64 = struct.unpack('<d', bytes.fromhex('0100000000000000'))[0]
    negative64 = struct.unpack('<d', bytes.fromhex('0100000000000080'))[0]
    if (tiny32 != math.ldexp(1.0, -149) or negative32 != -math.ldexp(1.0, -149)
            or tiny64 * 2.0 != math.ldexp(1.0, -1073)
            or negative64 * 2.0 != -math.ldexp(1.0, -1073)):
        raise RuntimeError('the calling thread must preserve signed f32/f64 subnormals')


def _row(values):
    values = list(values)
    if (not 2 <= len(values) <= 65536 or
            any(type(value) not in (int, float) for value in values)):
        raise ValueError('a plain numeric full-vocabulary row is required')
    values = [float(value) for value in values]
    if any(not math.isfinite(value) or abs(value) > 3.4028234663852886e38 for value in values):
        raise ValueError('a finite full-vocabulary row is required')
    return values


def _partition_components(values):
    if not values:
        raise ValueError('an empty partition is invalid')
    maximum = max(values)
    count = sum(value == maximum for value in values)
    remainder = math.fsum(math.exp(value - maximum) for value in values if value < maximum)
    return maximum, math.log(count), math.log1p(remainder / count)


def _partition(values):
    maximum, base, correction = _partition_components(values)
    return maximum, math.fsum((base, correction))


def log_probabilities(values):
    require_denormals()
    values = _row(values)
    maximum, base, correction = _partition_components(values)
    result = [math.fsum((value, -maximum, -base, -correction)) for value in values]
    require_denormals()
    return result


def describe(values, token_ids, accepted_ids=None):
    values = _row(values)
    if any(type(i) is not int or not 0 <= i < len(values) for i in token_ids):
        raise ValueError('token IDs must be in range')
    if len(set(token_ids)) != len(token_ids):
        raise ValueError('requested token IDs must be distinct')
    top = max(values)
    top_ids = [i for i, value in enumerate(values) if value == top]
    lower = max((value for value in values if value < top), default=None)
    logp = log_probabilities(values)
    result = {'top_ids': top_ids, 'top_representative': top_ids[0],
              'unique_top': top_ids[0] if len(top_ids) == 1 else None,
              'top_gap': 0.0 if len(top_ids) > 1 else (None if lower is None else top - lower),
              'ranks': {str(i): 1 + sum(value > values[i] for value in values) for i in token_ids},
              'logits': {str(i): values[i] for i in token_ids},
              'log_probabilities': {str(i): logp[i] for i in token_ids},
              'probabilities': {str(i): math.exp(logp[i]) for i in token_ids}}
    if accepted_ids is not None:
        if (not accepted_ids or len(set(accepted_ids)) != len(accepted_ids) or
                any(type(i) is not int or not 0 <= i < len(values) for i in accepted_ids)):
            raise ValueError('accepted IDs must be nonempty, distinct and in range')
        all_max, all_log = _partition(values)
        accepted_max, accepted_log = _partition([values[i] for i in accepted_ids])
        result['accepted_nll'] = math.fsum((all_max, -accepted_max, all_log, -accepted_log))
        result['top_is_accepted'] = (None if result['unique_top'] is None
                                     else result['unique_top'] in accepted_ids)
    return result


def _sum_exp_logs(log_values):
    if not log_values:
        return 0.0
    maximum, log_sum = _partition(log_values)
    return math.exp(maximum + log_sum)


def distances(original, edited):
    require_denormals()
    original, edited = _row(original), _row(edited)
    if len(original) != len(edited):
        raise ValueError('full rows must have the same vocabulary')
    xm, xb, xc = _partition_components(original)
    ym, yb, yc = _partition_components(edited)
    ordinary, positive_tail, negative_tail, tv_logs = [], [], [], []
    for x, y in zip(original, edited):
        a = math.fsum((x, -xm, -xb, -xc))
        b = math.fsum((y, -ym, -yb, -yc))
        difference = math.fsum((x, -xm, -xb, -xc, -y, ym, yb, yc))
        if difference:
            term_log = a + math.log(abs(difference))
            if term_log < -700:
                (positive_tail if difference > 0 else negative_tail).append(term_log)
            else:
                ordinary.append(math.copysign(math.exp(term_log), difference))
        delta = abs(difference)
        if delta:
            # log(|exp(a)-exp(b)|), stable also when each probability underflows.
            tv_logs.append(max(a, b) + math.log(-math.expm1(-delta)))
    raw = math.fsum([*ordinary, _sum_exp_logs(positive_tail), -_sum_exp_logs(negative_tail)])
    if not math.isfinite(raw) or raw < -1e-12:
        raise ValueError('KL reduction is materially negative or nonfinite')
    result = {'kl_raw': raw, 'kl': max(0.0, raw), 'tv': 0.5 * _sum_exp_logs(tv_logs)}
    require_denormals()
    return result


def solve_bias(original, primary, purple, factual):
    original, primary = _row(original), _row(primary)
    if (len(original) != len(primary) or type(purple) is not int or type(factual) is not int
            or purple == factual or not 0 <= purple < len(original) or not 0 <= factual < len(original)
            or len(original) < 3):
        raise ValueError('two distinct target IDs and a nonempty other partition are required')
    x_max, x_log = _partition([v for i, v in enumerate(original) if i not in (purple, factual)])
    y_max, y_log = _partition([v for i, v in enumerate(primary) if i not in (purple, factual)])
    offset = math.fsum((x_max, -y_max, x_log, -y_log))
    return (math.fsum((primary[purple], offset, -original[purple])),
            math.fsum((primary[factual], offset, -original[factual])))


def effect(original, edited, purple, factual):
    original, edited = _row(original), _row(edited)
    if len(original) != len(edited) or purple == factual:
        raise ValueError('invalid paired rows or target IDs')
    before, after = log_probabilities(original), log_probabilities(edited)
    info = describe(edited, (purple, factual))
    return {'margin_gain': math.fsum((edited[purple], -edited[factual],
                                      -original[purple], original[factual])),
            'logp_gain': after[purple] - before[purple], 'unique_top': info['unique_top']}


def row_matches(primary, bias):
    for row in (primary, bias):
        if (not math.isfinite(row['margin_gain']) or not math.isfinite(row['logp_gain'])
                or row['margin_gain'] < 0.1 or row['logp_gain'] < 0.025
                or type(row['unique_top']) is not int):
            return False
    if primary['unique_top'] != bias['unique_top']:
        return False
    return all(abs(primary[key] - bias[key]) <= 0.10 * max(abs(primary[key]), abs(bias[key]))
               for key in ('margin_gain', 'logp_gain'))


def locality_decision(rows, *, match_available):
    if len(rows) != 12:
        raise ValueError('all 12 fixed locality rows are required')
    for row in rows:
        if (not row['accepted_ids'] or len(set(row['accepted_ids'])) != len(row['accepted_ids']) or
                not math.isfinite(row['primary_kl']) or row['primary_kl'] < 0 or
                not math.isfinite(row['primary_nll_change'])):
            raise ValueError('invalid locality metric')
        if row['bias_kl'] is not None and (not math.isfinite(row['bias_kl']) or row['bias_kl'] < 0):
            raise ValueError('invalid bias KL')
    accepted = [row for row in rows if row['original_top'] is not None
                and row['original_top'] in row['accepted_ids']]
    primary_comparable = [row for row in rows if row['original_top'] is not None and row['primary_top'] is not None]
    bias_comparable = [row for row in rows if row['original_top'] is not None and row['bias_top'] is not None]
    mean_kl = math.fsum(row['primary_kl'] for row in rows) / 12
    mean_nll = math.fsum(row['primary_nll_change'] for row in rows) / 12
    result = {'baseline_accepted': len(accepted),
              'primary_top_change_count': sum(row['original_top'] != row['primary_top'] for row in primary_comparable),
              'primary_top_change_denominator': len(primary_comparable),
              'bias_top_change_count': sum(row['original_top'] != row['bias_top'] for row in bias_comparable),
              'bias_top_change_denominator': len(bias_comparable),
              'primary_mean_kl': mean_kl, 'primary_max_kl': max(row['primary_kl'] for row in rows),
              'primary_mean_nll_change': mean_nll,
              'primary_max_nll_change': max(row['primary_nll_change'] for row in rows),
              'bias_mean_kl': (math.fsum(row['bias_kl'] for row in rows) / 12
                               if all(row['bias_kl'] is not None for row in rows) else None)}
    if not match_available:
        result['status'] = 'match-unavailable'
    elif (not accepted or any(row['original_top'] is None or row['primary_top'] is None or
                              row['bias_top'] is None or row['bias_kl'] is None for row in rows)):
        result['status'] = 'not-evaluable'
    else:
        bias_kl = result['bias_mean_kl']
        passed = (mean_kl < 0.8 * bias_kl and bias_kl - mean_kl >= 1e-6
                  and result['primary_top_change_count'] <= result['bias_top_change_count']
                  and mean_kl <= 0.02 and result['primary_max_kl'] <= 0.10
                  and mean_nll <= 0.05 and result['primary_max_nll_change'] <= 0.20
                  and all(row['primary_top'] in row['accepted_ids'] for row in accepted))
        result['status'] = 'pass' if passed else 'fail'
    return result
