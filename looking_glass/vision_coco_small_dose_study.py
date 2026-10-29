"""Small-dose v2 plan and decision arithmetic, independent of model loading."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from .vision_coco_study import _array, _replacement, _tolerance, check_denormals, distance


def _random_small(own, q, pair_id, side):
    x = _array(own).astype(np.float64)
    signs = [1.0 if hashlib.sha256(
        f'vision-coco-small-dose-random-v2:{pair_id}:{side}:{i}'.encode('ascii')).digest()[0] & 1
             else -1.0 for i in range(x.size)]
    result = (x + (q / math.sqrt(x.size)) * np.asarray(signs).reshape(x.shape)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite random replacement')
    return torch.from_numpy(result.copy())


def prepare_small_dose(own, other, inside, outside, pair_id, side, same):
    """Compute four matched, non-extrapolating requested float32 vectors."""
    check_denormals()
    oi, di = own[0, inside], other[0, inside]
    oo, do = own[0, outside], other[0, outside]
    inside_norm = distance(oi, torch.zeros_like(oi))
    outside_norm = distance(oo, torch.zeros_like(oo))
    inside_d, outside_d = distance(oi, di), distance(oo, do)
    same_d = None if same is None else distance(oi, same[0, inside])
    result = {'inside_receiver_norm': inside_norm, 'outside_receiver_norm': outside_norm,
              'inside_distance': inside_d, 'outside_distance': outside_d,
              'same_species_distance': same_d, 'status': 'dose-ineligible', 'reasons': []}
    for name, value in (('inside-receiver-norm', inside_norm), ('outside-receiver-norm', outside_norm),
                        ('inside-distance', inside_d), ('outside-distance', outside_d),
                        ('same-species-distance', same_d)):
        if value is None or value <= 1e-6:
            result['reasons'].append(name)
    if result['reasons']:
        return result, {}
    q = min(.099 * inside_norm, .099 * outside_norm, inside_d, outside_d, same_d)
    result['q'] = q
    if q <= 1e-6:
        result['reasons'].append('q')
        return result, {}
    requested = {'inside': _replacement(oi, di, q, inside_d),
                 'outside': _replacement(oo, do, q, outside_d),
                 'same_species': _replacement(oi, same[0, inside], q, same_d),
                 'random': _random_small(oi, q, pair_id, side)}
    norms = {name: distance(value, oo if name == 'outside' else oi)
             for name, value in requested.items()}
    result.update(tolerance=_tolerance(q), planned_norms=norms,
                  donor_scales={'inside': q / inside_d, 'outside': q / outside_d,
                                'same_species': q / same_d})
    inside_delivery = norms['inside']
    if any(value <= 0 or abs(value - q) > _tolerance(q) or
           abs(value - inside_delivery) > .01 * inside_delivery or
           value > .10 * (outside_norm if name == 'outside' else inside_norm)
           for name, value in norms.items()):
        result.update(status='dose-unmatchable', reasons=['rounded-delivery'])
    else:
        result.update(status='matchable')
    return result, requested


def classify_small_pair(base, dose_statuses, movements):
    if not base['eligible']:
        return 'baseline-ineligible'
    if any(s != 'matchable' for s in dose_statuses.values()):
        return ('dose-ineligible' if 'dose-ineligible' in dose_statuses.values()
                else 'dose-unmatchable')
    passes = all(movements[side]['inside'] >= .05 and
                 movements[side]['inside'] >=
                 max(abs(movements[side][key]) for key in ('outside', 'same_species', 'random')) + .025
                 for side in ('cat', 'dog'))
    return 'positive' if passes else 'negative'


def aggregate_small(rows):
    statuses = {'positive', 'negative', 'baseline-ineligible', 'dose-ineligible', 'dose-unmatchable'}
    if (len(rows) != 8 or {r.get('pair_id') for r in rows} != {f'pair-{i:02d}' for i in range(8)} or
            any(r.get('split') != 'heldout' or r.get('status') not in statuses for r in rows)):
        raise ValueError('aggregate requires all eight fixed evaluation pairs')
    eligible = [r for r in rows if r['status'] in ('positive', 'negative')]
    positives = sum(r['status'] == 'positive' for r in eligible)
    status = ('inconclusive' if len(eligible) < 4 else
              'positive' if positives * 4 >= 3 * len(eligible) else 'negative')
    return {'status': status, 'selected': len(rows), 'evaluable': len(eligible), 'positive': positives}
