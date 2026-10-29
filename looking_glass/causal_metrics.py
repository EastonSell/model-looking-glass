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


def evaluate_effects(cases, *, zero_tolerance=0.0, bootstrap_samples=2000,
                     confidence=0.95, seed=0, prediction_interval_level=None):
    """Evaluate scalar effect predictions against observations and a zero null."""
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
    point = _point_metrics(cases, zero_tolerance, prediction_interval_level)
    rng = random.Random(seed)
    replicates = [_point_metrics([cases[rng.randrange(n)] for _ in range(n)], zero_tolerance,
                                  prediction_interval_level)
                  for _ in range(bootstrap_samples)]
    tail = (1 - confidence) / 2
    intervals = {}
    defined_samples = {}
    undefined_samples = {}
    directional_metrics = {'sign_accuracy', 'null_sign_accuracy', 'sign_accuracy_improvement_over_null'}
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
    return {
        'n_cases': n,
        'zero_tolerance': zero_tolerance,
        'prediction_interval_level': prediction_interval_level,
        'directional_cases': {'included': eligible, 'excluded': n - eligible},
        'point': point,
        'bootstrap': {'unit': 'independent_case', 'method': 'percentile' if bootstrap_samples else 'none',
                      'samples': bootstrap_samples, 'confidence': confidence,
                      'seed': seed, 'intervals': intervals,
                      'defined_samples': defined_samples, 'undefined_samples': undefined_samples},
        'case_errors': [{'case_id': case.case_id,
                         'absolute_error': abs(case.predicted - case.observed),
                         'null_absolute_error': abs(case.observed)} for case in cases],
    }


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


def _signed_mean(values):
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        # A finite mean can have a sum outside float range; preserve cancellation.
        total = sum((Fraction.from_float(value) for value in values), Fraction())
        return float(total / len(values))


def _root_mean_square(values):
    largest = max(values)
    if largest == 0:
        return 0.0
    return largest * math.sqrt(_mean([(value / largest) ** 2 for value in values]))


def _rmse_improvement(null_errors, errors):
    # Use the factored difference of squares instead of subtracting rounded RMSEs.
    scale = max(max(null_errors), max(errors))
    if scale == 0:
        return 0.0
    baseline = _root_mean_square([value / scale for value in null_errors])
    candidate = _root_mean_square([value / scale for value in errors])
    differences = [((a - b) / scale) * (a / scale + b / scale)
                   for a, b in zip(null_errors, errors)]
    return scale * (_signed_mean(differences) / (baseline + candidate))


def _point_metrics(cases, zero_tolerance, prediction_interval_level):

    def sign(value):
        return 1 if value > zero_tolerance else -1 if value < -zero_tolerance else 0

    errors = [abs(case.predicted - case.observed) for case in cases]
    null_errors = [abs(case.observed) for case in cases]
    n = len(cases)
    mae = _mean(errors)
    rmse = _root_mean_square(errors)
    directional = [case for case in cases if abs(case.observed) > zero_tolerance]
    sign_accuracy = (sum(sign(c.predicted) == sign(c.observed) for c in directional) / len(directional)
                     if directional else None)
    null_mae = _mean(null_errors)
    null_rmse = _root_mean_square(null_errors)
    null_sign_accuracy = 0.0 if directional else None
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
        'mae_improvement_over_null': _signed_mean([a - b for a, b in zip(null_errors, errors)]),
        'rmse_improvement_over_null': _rmse_improvement(null_errors, errors),
        'sign_accuracy_improvement_over_null': (
            sign_accuracy - null_sign_accuracy if sign_accuracy is not None else None),
    }
