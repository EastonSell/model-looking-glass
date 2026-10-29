"""Matched same-site dose and fixed evaluation rule for ViT v3."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from .vision_coco_study import _array, _replacement, _tolerance, check_denormals, distance


def _random_same_site(own, q, pair_id, side):
    x = _array(own).astype(np.float64)
    signs = [1.0 if hashlib.sha256(
        f'vision-coco-small-dose-random-v2:{pair_id}:{side}:{i}'.encode('ascii')).digest()[0] & 1
             else -1.0 for i in range(x.size)]
    result = (x + (q / math.sqrt(x.size)) * np.asarray(signs).reshape(x.shape)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite random replacement')
    return torch.from_numpy(result.copy())


def prepare_same_site_dose(own, other, inside, pair_id, side, same):
    check_denormals()
    block, opposite, same_block = own[0, inside], other[0, inside], same[0, inside]
    receiver_norm = distance(block, torch.zeros_like(block))
    opposite_distance, same_distance = distance(block, opposite), distance(block, same_block)
    result = {'receiver_norm': receiver_norm, 'opposite_distance': opposite_distance,
              'same_species_distance': same_distance, 'status': 'dose-ineligible', 'reasons': []}
    for name, value in (('receiver-norm', receiver_norm), ('opposite-distance', opposite_distance),
                        ('same-species-distance', same_distance)):
        if value <= 1e-6:
            result['reasons'].append(name)
    if result['reasons']:
        return result, {}
    q = min(.099 * receiver_norm, opposite_distance, same_distance)
    result['q'] = q
    if q <= 1e-6:
        result['reasons'].append('q')
        return result, {}
    requested = {'opposite': _replacement(block, opposite, q, opposite_distance),
                 'same_species': _replacement(block, same_block, q, same_distance),
                 'random': _random_same_site(block, q, pair_id, side)}
    norms = {name: distance(value, block) for name, value in requested.items()}
    result.update(tolerance=_tolerance(q), planned_norms=norms,
                  donor_scales={'opposite': q / opposite_distance, 'same_species': q / same_distance})
    first = norms['opposite']
    if any(value <= 0 or abs(value - q) > _tolerance(q) or abs(value - first) > .01 * first or
           value > .10 * receiver_norm for value in norms.values()):
        result.update(status='dose-unmatchable', reasons=['rounded-delivery'])
    else:
        result.update(status='matchable')
    return result, requested


def classify_same_site(base, dose_statuses, movements):
    allowed = {'matchable', 'dose-ineligible', 'dose-unmatchable'}
    if (type(dose_statuses) is not dict or set(dose_statuses) != {'cat', 'dog'} or
            any(value not in allowed for value in dose_statuses.values())):
        raise ValueError('invalid dose status')
    if type(base.get('eligible')) is not bool:
        raise ValueError('invalid baseline eligibility')
    if not base['eligible']:
        return 'baseline-ineligible'
    if any(value != 'matchable' for value in dose_statuses.values()):
        return ('dose-ineligible' if 'dose-ineligible' in dose_statuses.values() else 'dose-unmatchable')
    if (type(movements) is not dict or set(movements) != {'cat', 'dog'} or
            any(type(movements[side]) is not dict or
                set(movements[side]) != {'opposite', 'same_species', 'random'} or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in movements[side].values()) for side in ('cat', 'dog'))):
        raise ValueError('invalid or nonfinite movement')
    passed = all(movements[side]['opposite'] >= .05 and
                 movements[side]['opposite'] >=
                 max(abs(movements[side]['same_species']), abs(movements[side]['random'])) + .025
                 for side in ('cat', 'dog'))
    return 'positive' if passed else 'negative'


def aggregate_same_site(rows):
    allowed = {'positive', 'negative', 'baseline-ineligible', 'dose-ineligible', 'dose-unmatchable'}
    if (len(rows) != 8 or {r.get('pair_id') for r in rows} != {f'pair-{i:02d}' for i in range(8)} or
            any(r.get('split') != 'heldout' or r.get('status') not in allowed for r in rows)):
        raise ValueError('aggregate requires all eight fixed evaluation pairs')
    eligible = [r for r in rows if r['status'] in ('positive', 'negative')]
    positive = sum(r['status'] == 'positive' for r in eligible)
    status = ('inconclusive' if len(eligible) < 4 else
              'positive' if positive * 4 >= 3 * len(eligible) else 'negative')
    return {'status': status, 'selected': len(rows), 'evaluable': len(eligible), 'positive': positive}
