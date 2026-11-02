"""Fixed, role-blind numeric rules for the public distractor experiment."""
import math


def _finite(values, size):
    if not isinstance(values, (tuple, list)) or len(values) != size:
        raise ValueError('incorrect fixed measurement count')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in values):
        raise ValueError('measurement must be a finite number')
    return [float(v) for v in values]


def select_candidate(effect_rows):
    """Rank only the numeric per-case Q1..Q4 effects; no site information."""
    if not isinstance(effect_rows, (tuple, list)) or len(effect_rows) != 8:
        raise ValueError('expected the eight frozen discovery rows')
    rows = [_finite(row, 4) for row in effect_rows]
    signed = [math.fsum(row[i] for row in rows) / 8 for i in range(4)]
    absolute = [math.fsum(abs(row[i]) for row in rows) / 8 for i in range(4)]
    indices = sorted(range(4), key=lambda i: (-absolute[i], i))
    best = absolute[indices[0]]
    return {
        'selected': f'Q{indices[0] + 1}' if best > 0.0 else None,
        'ranking': [f'Q{i + 1}' for i in indices],
        'mean_absolute_effects': absolute,
        'signed_predictions': signed,
        'max_tie': [f'Q{i + 1}' for i in indices if absolute[i] == best],
        'abstained': best == 0.0,
    }


def _cell(combined, patched):
    combined, patched = _finite(combined, 8), _finite(patched, 8)
    if any(x < 0 for x in combined + patched):
        raise ValueError('L2 distances cannot be negative')
    denominator = math.fsum(combined) / 8
    numerator = math.fsum(patched) / 8
    reduction = 1.0 - numerator / denominator if denominator > 1e-12 else None
    return {'mean_combined_distance': denominator, 'mean_patched_distance': numerator,
            'reduction': reduction, 'passes': reduction > 0.25 if reduction is not None else None,
            'per_case_reductions': [1.0 - p/c if c > 1e-12 else None
                                    for c, p in zip(combined, patched)]}


def primary_rule(cells):
    if not isinstance(cells, (tuple, list)) or len(cells) != 3:
        raise ValueError('all three fixed cells are required')
    out = []
    for pair in cells:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('expected combined and patched distances')
        out.append(_cell(pair[0], pair[1]))
    supported = None if any(x['passes'] is None for x in out) else sum(x['passes'] for x in out) >= 2
    return {'supported': supported, 'cells': out}


def prediction_error(prediction, observations):
    pred = _finite([prediction], 1)[0]
    observations = _finite(observations, 8)
    mae = math.fsum(abs(x - pred) for x in observations) / 8
    zero_mae = math.fsum(abs(x) for x in observations) / 8
    return {'mae': mae, 'zero_mae': zero_mae, 'improved': mae < zero_mae}
