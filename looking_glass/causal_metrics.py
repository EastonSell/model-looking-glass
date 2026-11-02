"""Metrics for prespecified intervention-effect predictions on held-out cases."""

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
import random


@dataclass(frozen=True)
class EffectCase:
    """One independent case, with optional prespecified prediction bounds."""

    case_id: str
    predicted: float
    observed: float
    interval_low: float | None = None
    interval_high: float | None = None


@dataclass(frozen=True)
class InterventionCase:
    """Matched scalar readouts and caller-supplied statuses for one case."""

    case_id: str
    baseline_status: str
    eligibility: str
    reference: float | None
    candidate: float | None
    control: float | None


def summarize_interventions(cases):
    """Summarize signed, matched intervention readouts without inferring benefit.

    Baseline and eligibility statuses are caller-supplied. All available values
    are retained, including values for excluded cases. Aggregate means use only
    cases with a supported baseline, eligible intervention and all three finite
    readouts. No confidence interval or independence claim is made.
    """
    try:
        cases = list(cases)
    except TypeError:
        raise ValueError('cases must be an iterable of InterventionCase records') from None
    if not cases:
        raise ValueError('at least one case is required')

    baseline_counts = dict.fromkeys(('supported', 'unsupported', 'unverified'), 0)
    eligibility_counts = dict.fromkeys(('eligible', 'ineligible', 'unverified'), 0)
    exclusions = dict.fromkeys(('baseline_unsupported', 'baseline_unverified',
                                'intervention_ineligible', 'intervention_unverified',
                                'missing_readout'), 0)
    missing = dict.fromkeys(('reference', 'candidate', 'control'), 0)
    rows = []
    ids = set()
    for case in cases:
        if not isinstance(case, InterventionCase):
            raise ValueError('cases must contain only InterventionCase records')
        if not isinstance(case.case_id, str) or not case.case_id.strip():
            raise ValueError('each case must have a nonempty string case_id')
        if case.case_id in ids:
            raise ValueError(f'duplicate case_id: {case.case_id!r}')
        ids.add(case.case_id)
        if not isinstance(case.baseline_status, str) or case.baseline_status not in baseline_counts:
            raise ValueError(f'{case.case_id}: invalid baseline_status')
        if not isinstance(case.eligibility, str) or case.eligibility not in eligibility_counts:
            raise ValueError(f'{case.case_id}: invalid eligibility')
        baseline_counts[case.baseline_status] += 1
        eligibility_counts[case.eligibility] += 1

        values = {}
        for name in missing:
            raw = getattr(case, name)
            values[name] = None if raw is None else _finite_float(raw, f'{case.case_id}.{name}')
            missing[name] += raw is None

        def difference(left, right, name):
            if values[left] is None or values[right] is None:
                return None
            result = values[left] - values[right]
            if not math.isfinite(result):
                raise ValueError(f'{case.case_id}.{name}: difference exceeds finite float range')
            return result

        candidate_effect = difference('candidate', 'reference', 'candidate_effect')
        control_effect = difference('control', 'reference', 'control_effect')
        contrast = difference('candidate', 'control', 'candidate_minus_control')

        if case.baseline_status != 'supported':
            reason = 'baseline_' + case.baseline_status
        elif case.eligibility != 'eligible':
            reason = 'intervention_' + case.eligibility
        elif any(value is None for value in values.values()):
            reason = 'missing_readout'
        else:
            reason = None
        if reason is not None:
            exclusions[reason] += 1
        rows.append({'case_id': case.case_id, 'baseline_status': case.baseline_status,
                     'eligibility': case.eligibility, **values,
                     'candidate_effect': candidate_effect, 'control_effect': control_effect,
                     'candidate_minus_control': contrast,
                     'included_in_paired_summary': reason is None, 'exclusion_reason': reason})

    paired = [row for row in rows if row['included_in_paired_summary']]
    n = len(paired)

    def mean(name):
        if not paired:
            return None
        # Exact summation of the finite binary64 values avoids overflow and
        # cancellation in the accumulator; only the final mean is rounded.
        return float(sum((Fraction.from_float(row[name]) for row in paired), Fraction()) / n)

    contrasts = [row['candidate_minus_control'] for row in paired]
    return {'n_cases': len(rows), 'baseline_status_counts': baseline_counts,
            'eligibility_counts': eligibility_counts, 'missing_readout_counts': missing,
            'paired_cases': {'included': n, 'excluded': len(rows) - n,
                             'exclusion_counts': exclusions},
            'point': {'mean_candidate_effect': mean('candidate_effect'),
                      'mean_control_effect': mean('control_effect'),
                      'mean_candidate_minus_control': mean('candidate_minus_control'),
                      'candidate_above_control': sum(value > 0 for value in contrasts),
                      'candidate_equal_control': sum(value == 0 for value in contrasts),
                      'candidate_below_control': sum(value < 0 for value in contrasts)},
            'cases': rows}


def evaluate_effects(cases, *, zero_tolerance=0.0, bootstrap_samples=2000,
                     confidence=0.95, seed=0, prediction_interval_level=None):
    """Evaluate scalar effect predictions against observations and a zero null."""
    zero_tolerance, confidence = _validate_settings(zero_tolerance, bootstrap_samples, confidence, seed)
    cases = _validate_cases(cases)
    n = len(cases)
    if bootstrap_samples and n < 2:
        raise ValueError('at least two independent cases are required for bootstrap')
    has_bounds = cases[0].interval_low is not None
    if has_bounds != (prediction_interval_level is not None):
        raise ValueError('prediction_interval_level must be supplied exactly when bounds are supplied')
    if has_bounds:
        prediction_interval_level = _finite_float(prediction_interval_level, 'prediction_interval_level')
        if not 0 < prediction_interval_level < 1:
            raise ValueError('prediction_interval_level must be strictly between zero and one')
    eligible = sum(abs(case.observed) > zero_tolerance for case in cases)
    terms = {case.case_id: _loss_terms(case.predicted, 0.0, case.observed) for case in cases}
    point = _point_metrics(cases, zero_tolerance, prediction_interval_level, terms)
    bootstrap = _bootstrap(cases, point,
                           lambda sample: _point_metrics(sample, zero_tolerance, prediction_interval_level, terms),
                           {'sign_accuracy', 'null_sign_accuracy', 'sign_accuracy_improvement_over_null'},
                           eligible, bootstrap_samples, confidence, seed)
    return {
        'n_cases': n,
        'zero_tolerance': zero_tolerance,
        'prediction_interval_level': prediction_interval_level,
        'directional_cases': {'included': eligible, 'excluded': n - eligible},
        'point': point,
        'bootstrap': bootstrap,
        'case_errors': [{'case_id': case.case_id,
                         'absolute_error': abs(case.predicted - case.observed),
                         'null_absolute_error': abs(case.observed)} for case in cases],
    }


def compare_effects(candidate_cases, baseline_cases, *, zero_tolerance=0.0,
                    bootstrap_samples=0, confidence=0.95, seed=0):
    """Compare two predictors against exactly the same held-out observations."""
    zero_tolerance, confidence = _validate_settings(zero_tolerance, bootstrap_samples, confidence, seed)
    try:
        original_candidate = list(candidate_cases)
        original_baseline = list(baseline_cases)
    except TypeError:
        raise ValueError('cases must be an iterable of EffectCase records') from None
    candidate = _validate_cases(original_candidate)
    baseline = _validate_cases(original_baseline)
    if candidate[0].interval_low is not None or baseline[0].interval_low is not None:
        raise ValueError('compare_effects does not evaluate prediction intervals')
    candidate_targets = {case.case_id: case.observed for case in original_candidate}
    baseline_targets = {case.case_id: case.observed for case in original_baseline}
    if candidate_targets != baseline_targets:
        raise ValueError('candidate and baseline must have identical case IDs and observations')
    pairs = list(zip(candidate, baseline))
    n = len(pairs)
    if bootstrap_samples and n < 2:
        raise ValueError('at least two independent cases are required for bootstrap')
    eligible = sum(abs(case.observed) > zero_tolerance for case in candidate)
    terms = {a.case_id: _loss_terms(a.predicted, b.predicted, a.observed) for a, b in pairs}
    point = _comparison_metrics(pairs, zero_tolerance, terms)
    bootstrap = _bootstrap(pairs, point, lambda sample: _comparison_metrics(sample, zero_tolerance, terms),
                           {'candidate_sign_accuracy', 'baseline_sign_accuracy', 'sign_accuracy_improvement'},
                           eligible, bootstrap_samples, confidence, seed)
    return {
        'n_cases': n,
        'zero_tolerance': zero_tolerance,
        'directional_cases': {'included': eligible, 'excluded': n - eligible},
        'point': point,
        'bootstrap': bootstrap,
        'case_errors': [{'case_id': a.case_id,
                         'candidate_absolute_error': abs(a.predicted - a.observed),
                         'baseline_absolute_error': abs(b.predicted - b.observed)} for a, b in pairs],
    }


def _validate_settings(zero_tolerance, bootstrap_samples, confidence, seed):
    if (isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int)
            or bootstrap_samples < 0 or bootstrap_samples == 1):
        raise ValueError('bootstrap_samples must be zero or an integer of at least two')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    confidence = _finite_float(confidence, 'confidence')
    zero_tolerance = _finite_float(zero_tolerance, 'zero_tolerance')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between zero and one')
    if zero_tolerance < 0:
        raise ValueError('zero_tolerance must be nonnegative')
    return zero_tolerance, confidence


def _bootstrap(cases, point, metric_fn, directional_metrics, eligible, bootstrap_samples, confidence, seed):
    n = len(cases)
    rng = random.Random(seed)
    replicates = [metric_fn([cases[rng.randrange(n)] for _ in range(n)])
                  for _ in range(bootstrap_samples)]
    tail = (1 - confidence) / 2
    intervals = {}
    defined_samples = {}
    undefined_samples = {}
    for metric in point:
        ordered = sorted(replicate[metric] for replicate in replicates if replicate[metric] is not None)
        defined_samples[metric] = len(ordered)
        undefined_samples[metric] = bootstrap_samples - len(ordered)
        suppress_direction = metric in directional_metrics and (eligible < 2 or undefined_samples[metric] > 0)
        if not ordered or suppress_direction:
            intervals[metric] = None
        else:
            intervals[metric] = {'low': _percentile(ordered, tail),
                                 'high': _percentile(ordered, 1 - tail)}
    return {'unit': 'independent_case', 'method': 'percentile' if bootstrap_samples else 'none',
            'samples': bootstrap_samples, 'confidence': confidence,
            'seed': seed, 'intervals': intervals,
            'defined_samples': defined_samples, 'undefined_samples': undefined_samples}


def _percentile(ordered, p):
    position = p * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    a, b = ordered[low], ordered[high]
    if a == b:
        return a
    if a < 0 < b:
        return math.fsum((a * (1 - fraction), b * fraction))
    return a + fraction * (b - a)


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{name} must be a finite real number') from None
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite real number')
    if result != value:
        raise ValueError(f'{name}: value is not exactly representable as a float')
    return result


def _validate_cases(cases):
    try:
        cases = list(cases)
    except TypeError:
        raise ValueError('cases must be an iterable of EffectCase records') from None
    if not cases:
        raise ValueError('at least one case is required')
    normalized = []
    ids = set()
    bound_presence = set()
    for case in cases:
        if not isinstance(case, EffectCase):
            raise ValueError('cases must contain only EffectCase records')
        if not isinstance(case.case_id, str) or not case.case_id.strip():
            raise ValueError('each case must have a nonempty string case_id')
        if case.case_id in ids:
            raise ValueError(f'duplicate case_id: {case.case_id!r}')
        ids.add(case.case_id)
        predicted = _finite_float(case.predicted, f'{case.case_id}.predicted')
        observed = _finite_float(case.observed, f'{case.case_id}.observed')
        if not math.isfinite(predicted - observed):
            raise ValueError(f'{case.case_id}: prediction error exceeds finite float range')
        has_low = case.interval_low is not None
        has_high = case.interval_high is not None
        if has_low != has_high:
            raise ValueError(f'{case.case_id}: prediction interval requires both bounds')
        bound_presence.add(has_low)
        low = _finite_float(case.interval_low, f'{case.case_id}.interval_low') if has_low else None
        high = _finite_float(case.interval_high, f'{case.case_id}.interval_high') if has_high else None
        if has_low and low > high:
            raise ValueError(f'{case.case_id}: prediction interval bounds are reversed')
        normalized.append(EffectCase(case.case_id, predicted, observed, low, high))
    if len(bound_presence) != 1:
        raise ValueError('prediction intervals must be specified for every case or none')
    return sorted(normalized, key=lambda case: case.case_id)


def _mean(values):
    largest = max(values)
    if largest == 0:
        return 0.0
    return largest * (math.fsum(value / largest for value in values) / len(values))


def _root_mean_square(values):
    largest = max(values)
    if largest == 0:
        return 0.0
    return largest * math.sqrt(_mean([(value / largest) ** 2 for value in values]))


def _loss_terms(candidate, baseline, observed):
    # Exact real arithmetic on the supplied float values preserves small paired effects.
    target = Fraction.from_float(observed)
    candidate_error = abs(target - Fraction.from_float(candidate))
    baseline_error = abs(target - Fraction.from_float(baseline))
    difference = baseline_error - candidate_error
    return difference, difference * (baseline_error + candidate_error)


def _paired_loss_improvements(case_ids, terms, baseline_errors, errors):
    n = len(case_ids)
    mae = float(sum((terms[case_id][0] for case_id in case_ids), Fraction()) / n)
    squared_difference = sum((terms[case_id][1] for case_id in case_ids), Fraction()) / n
    scale = max(max(baseline_errors), max(errors))
    if scale == 0:
        return mae, 0.0
    baseline = _root_mean_square([value / scale for value in baseline_errors])
    candidate = _root_mean_square([value / scale for value in errors])
    denominator = Fraction.from_float(scale) * Fraction.from_float(baseline + candidate)
    return mae, float(squared_difference / denominator)


def _direction_accuracy(cases, zero_tolerance):

    def sign(value):
        return 1 if value > zero_tolerance else -1 if value < -zero_tolerance else 0

    directional = [case for case in cases if abs(case.observed) > zero_tolerance]
    return (sum(sign(c.predicted) == sign(c.observed) for c in directional) / len(directional)
            if directional else None)


def _comparison_metrics(pairs, zero_tolerance, terms):
    candidate_cases = [candidate for candidate, _ in pairs]
    baseline_cases = [baseline for _, baseline in pairs]
    candidate_errors = [abs(c.predicted - c.observed) for c in candidate_cases]
    baseline_errors = [abs(c.predicted - c.observed) for c in baseline_cases]
    candidate_sign = _direction_accuracy(candidate_cases, zero_tolerance)
    baseline_sign = _direction_accuracy(baseline_cases, zero_tolerance)
    mae_improvement, rmse_improvement = _paired_loss_improvements(
        [case.case_id for case in candidate_cases], terms, baseline_errors, candidate_errors)
    return {'candidate_mae': _mean(candidate_errors), 'baseline_mae': _mean(baseline_errors),
            'candidate_rmse': _root_mean_square(candidate_errors), 'baseline_rmse': _root_mean_square(baseline_errors),
            'candidate_sign_accuracy': candidate_sign, 'baseline_sign_accuracy': baseline_sign,
            'mae_improvement': mae_improvement, 'rmse_improvement': rmse_improvement,
            'sign_accuracy_improvement': (candidate_sign - baseline_sign if candidate_sign is not None else None)}


def _point_metrics(cases, zero_tolerance, prediction_interval_level, terms):
    errors = [abs(case.predicted - case.observed) for case in cases]
    null_errors = [abs(case.observed) for case in cases]
    n = len(cases)
    mae = _mean(errors)
    rmse = _root_mean_square(errors)
    sign_accuracy = _direction_accuracy(cases, zero_tolerance)
    null_mae = _mean(null_errors)
    null_rmse = _root_mean_square(null_errors)
    null_sign_accuracy = 0.0 if sign_accuracy is not None else None
    mae_improvement, rmse_improvement = _paired_loss_improvements(
        [case.case_id for case in cases], terms, null_errors, errors)
    coverage = (sum(c.interval_low <= c.observed <= c.interval_high for c in cases) / n
                if cases[0].interval_low is not None else None)
    return {
        'mae': mae,
        'rmse': rmse,
        'sign_accuracy': sign_accuracy,
        'prediction_interval_coverage': coverage,
        'prediction_interval_coverage_minus_nominal': (
            coverage - prediction_interval_level if coverage is not None else None),
        'null_mae': null_mae,
        'null_rmse': null_rmse,
        'null_sign_accuracy': null_sign_accuracy,
        'mae_improvement_over_null': mae_improvement,
        'rmse_improvement_over_null': rmse_improvement,
        'sign_accuracy_improvement_over_null': (
            sign_accuracy - null_sign_accuracy if sign_accuracy is not None else None),
    }
