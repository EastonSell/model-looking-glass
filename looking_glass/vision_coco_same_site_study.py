"""Matched same-site dose and fixed evaluation rule for ViT v3."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from .vision_coco_study import (EARLY, LATE, _array, _replacement, _same, _tolerance,
                                baseline_status, check_denormals, check_landing, distance, forward, score)


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


def run_same_site(model, pairs, pixel_by_id, evidence, *, check_clean=None, numeric_guard=None):
    """Run the fixed same-site comparison, retaining every finite case.

    A caller may additionally bind every repeated clean output before any
    intervention; the original v3 path leaves ``check_clean`` unset.
    """
    target_ids = [pair[side]['image_id'] for pair in pairs for side in ('cat', 'dog')]
    donor_ids = [pair[side + '_donor']['image_id'] for pair in pairs for side in ('cat', 'dog')]
    if (len(target_ids) != len(set(target_ids)) or set(target_ids) & set(donor_ids) or
            set(target_ids) | set(donor_ids) != set(pixel_by_id)):
        raise ValueError('target and reusable donor image inventory differs')
    clean = {}
    for pair in pairs:
        for role in ('cat', 'dog', 'cat_donor', 'dog_donor'):
            image_id = pair[role]['image_id']
            if image_id in clean:
                continue
            logits, captures, row = forward(model, pixel_by_id[image_id], evidence,
                                            f"{pair['pair_id']}:{role}:initial", numeric_guard=numeric_guard)
            if check_clean is not None:
                check_clean(image_id, logits)
            _, reference = score(logits)
            clean[image_id] = {'logits': logits, 'early': captures['early'], 'late': captures['late'],
                               'score': row['score'], 'off_reference': reference}
    output = []
    for pair in pairs:
        if numeric_guard is not None:
            numeric_guard()
        pair_id, inside = pair['pair_id'], pair['box_tokens']
        base = baseline_status(clean[pair['cat']['image_id']]['score'],
                               clean[pair['dog']['image_id']]['score'])
        results = {}
        for side, other_side in (('cat', 'dog'), ('dog', 'cat')):
            image_id = pair[side]['image_id']
            own = clean[image_id]
            other = clean[pair[other_side]['image_id']]
            same = clean[pair[side + '_donor']['image_id']]
            pixels = pixel_by_id[image_id]
            if numeric_guard is not None:
                numeric_guard()
            dose, requested = prepare_same_site_dose(own['early'], other['early'], inside,
                                                    pair_id, side, same['early'])
            info = {'image_id': image_id, 'baseline': own['score'], 'dose': dose,
                    'same_species_donor_id': pair[side + '_donor']['image_id'],
                    'same_species_baseline': same['score'], 'conditions': {}}
            logits, captures, row = forward(model, pixels, evidence, f'{pair_id}:{side}:repeat',
                                            reference=own['off_reference'], numeric_guard=numeric_guard)
            if not all((_same(logits, own['logits']), _same(captures['early'], own['early']),
                        _same(captures['late'], own['late']))):
                raise ValueError('clean repeat failed')
            info['conditions']['repeat'] = row['score']
            plans = [
                ('null_inside', EARLY, inside, None, 'null', own['logits']),
                ('self_inside', EARLY, inside, own['early'][0, inside], 'patch', own['logits']),
                ('full_self', EARLY, None, own['early'], 'patch', own['logits']),
                ('full_swap', EARLY, None, other['early'], 'patch', other['logits']),
                ('natural_inside', EARLY, inside, other['early'][0, inside], 'patch', None),
            ]
            if dose['status'] == 'matchable':
                for name in ('opposite', 'same_species', 'random'):
                    plans.append(('matched_' + name, EARLY, inside, requested[name], 'patch', None))
            plans.append(('final_inside', LATE, inside, other['late'][0, inside], 'patch', own['logits']))
            for name, path, tokens, replacement, action, expected in plans:
                logits, captures, row = forward(model, pixels, evidence, f'{pair_id}:{side}:{name}',
                                                path=path, tokens=tokens, replacement=replacement,
                                                action=action, reference=own['off_reference'],
                                                numeric_guard=numeric_guard)
                state = own['early'] if path == EARLY else own['late']
                check_landing(captures, state, tokens, replacement, action=action)
                if expected is not None and not _same(logits, expected):
                    raise ValueError(f'{name} logits control failed')
                saved = dict(row['score'])
                saved['raw_margin_change'] = saved['margin'] - own['score']['margin']
                saved['annotation_oriented_change'] = (1 if side == 'cat' else -1) * saved['raw_margin_change']
                if name.startswith('matched_'):
                    pre, post = captures['before_all'][0, tokens], captures['after_all'][0, tokens]
                    norm = distance(post, pre)
                    receiver_norm = distance(pre, torch.zeros_like(pre))
                    if (norm <= 0 or abs(norm - dose['q']) > dose['tolerance'] or
                            receiver_norm <= 0 or norm > .10 * receiver_norm):
                        raise ValueError('actual matched dose or receiver-relative bound differs')
                    saved['actual_dose'] = norm
                    saved['token_doses'] = [distance(post[i], pre[i]) for i in range(4)]
                    saved['receiver_norm'] = receiver_norm
                    saved['realized_over_receiver_norm'] = norm / receiver_norm
                    if name != 'matched_opposite':
                        first = info['conditions']['matched_opposite']['actual_dose']
                        if abs(norm - first) > .01 * first:
                            raise ValueError('actual matched doses differ')
                info['conditions'][name] = saved
            logits, captures, row = forward(model, pixels, evidence, f'{pair_id}:{side}:post_clean',
                                            reference=own['off_reference'], numeric_guard=numeric_guard)
            if not all((_same(logits, own['logits']), _same(captures['early'], own['early']),
                        _same(captures['late'], own['late']))):
                raise ValueError('post-clean control failed')
            info['conditions']['post_clean'] = row['score']
            results[side] = info
        dose_statuses = {side: results[side]['dose']['status'] for side in ('cat', 'dog')}
        movements = None
        if all(status == 'matchable' for status in dose_statuses.values()):
            movements = {side: {name: results[side]['conditions']['matched_' + name]['annotation_oriented_change']
                                for name in ('opposite', 'same_species', 'random')}
                         for side in ('cat', 'dog')}
        if numeric_guard is not None:
            numeric_guard()
        status = classify_same_site(base, dose_statuses, movements)
        row = {'pair_id': pair_id, 'split': pair['split'], 'baseline_eligibility': base,
               'dose_matchability': dose_statuses, 'status': status, 'sides': results,
               'descriptive_movements': movements}
        evidence.json(f'{pair_id}.json', row)
        output.append(row)
    return output
