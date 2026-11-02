"""Three-dose, matched-control spatial ViT intervention on saved tensors."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from .vision_coco_cls_study import distribution_effect
from .vision_coco_study import (EARLY, LATE, _array, _replacement, _same, _tolerance,
                                baseline_status, check_denormals, check_landing, distance, forward, score)

LEVELS = ('low', 'mid', 'high')
ARMS = ('opposite', 'same_species', 'random')


def _random(block, q, pair_id, side):
    x = _array(block).astype(np.float64)
    signs = [1.0 if hashlib.sha256(
        f'vision-coco-spatial-dose-v6:{pair_id}:{side}:{i}'.encode('ascii')).digest()[0] & 1 else -1.0
        for i in range(x.size)]
    result = (x + (q / math.sqrt(x.size)) * np.asarray(signs).reshape(x.shape)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite random replacement')
    return torch.from_numpy(result.copy())


def prepare_ladder(own, opposite, same, tokens, pair_id, side):
    """Return three common-dose plans and preserve limited-dose plans."""
    check_denormals()
    if (len(tokens) != 4 or len(set(tokens)) != 4 or
            any(type(token) is not int or not 1 <= token < 197 for token in tokens)):
        raise ValueError('invalid spatial tokens')
    block = own[0, tokens]
    donor = {'opposite': opposite[0, tokens], 'same_species': same[0, tokens]}
    try:
        norm = distance(block, torch.zeros_like(block))
        distances = {kind: distance(block, value) for kind, value in donor.items()}
    except ValueError as error:
        if str(error) != 'nonfinite activation':
            raise
        return {'receiver_norm': None, 'donor_distances': {kind: None for kind in donor},
                'status': 'dose-ineligible', 'reasons': ['nonfinite-activation']}, {}
    result = {'receiver_norm': norm, 'donor_distances': distances,
              'status': 'dose-ineligible', 'reasons': []}
    if norm <= 1e-6:
        result['reasons'].append('receiver-norm')
    for kind, value in distances.items():
        if value <= 1e-6:
            result['reasons'].append(kind + '-distance')
    if result['reasons']:
        return result, {}
    high = min(.396 * norm, *distances.values())
    levels = {'low': high / 4, 'mid': high / 2, 'high': high}
    result.update(levels=levels, high_over_receiver_norm=high / norm)
    if levels['low'] <= 1e-6:
        result['reasons'].append('q-low')
        return result, {}
    requested, planned = {}, {}
    for level, q in levels.items():
        requested[level] = {kind: _replacement(block, value, q, distances[kind])
                            for kind, value in donor.items()}
        requested[level]['random'] = _random(block, q, pair_id, side)
        planned[level] = {kind: distance(block, value) for kind, value in requested[level].items()}
    result.update(planned_norms=planned, tolerances={level: _tolerance(q) for level, q in levels.items()},
                  donor_scales={level: {kind: q / d for kind, d in distances.items()}
                                for level, q in levels.items()})
    for level, q in levels.items():
        reference = planned[level]['opposite']
        if any(value <= 0 or abs(value - q) > _tolerance(q) or value > .4 * norm or
               abs(value - reference) > .01 * reference for value in planned[level].values()):
            result.update(status='dose-unmatchable', reasons=[level + '-rounded-delivery'])
            return result, requested
    result['status'] = 'matchable' if high / norm >= .30 else 'limited-dose'
    return result, requested


def side_metrics(movements):
    if type(movements) is not dict or set(movements) != set(LEVELS):
        raise ValueError('invalid dose movements')
    for values in movements.values():
        if (type(values) is not dict or set(values) != set(ARMS) or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values())):
            raise ValueError('invalid or nonfinite movements')
    adjusted = {level: values['opposite'] - max(abs(values['same_species']), abs(values['random']))
                for level, values in movements.items()}
    low, high = movements['low']['opposite'], movements['high']['opposite']
    low_pass = low >= .05 and adjusted['low'] >= .025
    delta = adjusted['high'] - adjusted['low']
    return {'adjusted': adjusted, 'raw_high_minus_low': high - low,
            'adjusted_high_minus_low': delta, 'low_directional_pass': low_pass,
            'side_pass': not low_pass and high >= .10 and adjusted['high'] >= .05 and delta >= .025}


def aggregate_pairs(rows):
    valid = {'positive', 'negative', 'limited-dose', 'dose-ineligible', 'dose-unmatchable'}
    if (type(rows) is not list or len(rows) > 4 or
            [r.get('pair_id') for r in rows] != [f'pair-{i:02d}' for i in range(len(rows))] or
            any(r.get('status') not in valid for r in rows)):
        raise ValueError('invalid selected-pair inventory')
    eligible = [r for r in rows if r['status'] in {'positive', 'negative'}]
    positives = sum(r['status'] == 'positive' for r in eligible)
    status = ('inconclusive' if len(eligible) < 2 else
              'positive' if positives * 4 >= 3 * len(eligible) else 'negative')
    return {'status': status, 'selected': len(rows), 'evaluable': len(eligible), 'positive': positives}


def run_dose_response(model, pairs, pixel_by_id, evidence, *, check_clean=None, numeric_guard=None):
    """Run selected sides with one common direction at three matched doses."""
    targets = [pair[side]['image_id'] for pair in pairs for side in ('cat', 'dog')]
    donors = [pair[side + '_donor']['image_id'] for pair in pairs for side in ('cat', 'dog')]
    if (not 1 <= len(pairs) <= 4 or len(set(targets)) != len(targets) or
            set(targets) & set(donors) or set(targets) | set(donors) != set(pixel_by_id)):
        raise ValueError('target and donor image inventory differs')
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
    bases = {pair['pair_id']: baseline_status(clean[pair['cat']['image_id']]['score'],
                                             clean[pair['dog']['image_id']]['score']) for pair in pairs}
    if any(not base['eligible'] for base in bases.values()):
        raise ValueError('selected baseline differs from the fixed screen')
    output = []
    for pair in pairs:
        pair_id, tokens = pair['pair_id'], pair['box_tokens']
        sides = {}
        for side, other_side in (('cat', 'dog'), ('dog', 'cat')):
            if numeric_guard is not None:
                numeric_guard()
            image_id = pair[side]['image_id']
            own, other = clean[image_id], clean[pair[other_side]['image_id']]
            same = clean[pair[side + '_donor']['image_id']]
            pixels = pixel_by_id[image_id]
            dose, requested = prepare_ladder(own['early'], other['early'], same['early'], tokens, pair_id, side)
            info = {'image_id': image_id, 'baseline': own['score'], 'dose': dose,
                    'same_species_donor_id': pair[side + '_donor']['image_id'],
                    'same_species_baseline': same['score'], 'conditions': {}}
            logits, captures, row = forward(model, pixels, evidence, f'{pair_id}:{side}:repeat',
                                            reference=own['off_reference'], numeric_guard=numeric_guard)
            if not all((_same(logits, own['logits']), _same(captures['early'], own['early']),
                        _same(captures['late'], own['late']))):
                raise ValueError('clean repeat failed')
            info['conditions']['repeat'] = {**row['score'], **distribution_effect(own['logits'], logits)}
            plans = [('null', EARLY, tokens, None, 'null', own['logits']),
                     ('self', EARLY, tokens, own['early'][0, tokens], 'patch', own['logits']),
                     ('full_self', EARLY, None, own['early'], 'patch', own['logits']),
                     ('full_swap', EARLY, None, other['early'], 'patch', other['logits'])]
            if dose['status'] in {'matchable', 'limited-dose'}:
                plans.extend((f'{level}_{kind}', EARLY, tokens, requested[level][kind], 'patch', None)
                             for level in LEVELS for kind in ARMS)
            plans.append(('final_spatial', LATE, tokens, other['late'][0, tokens], 'patch', own['logits']))
            for name, path, selected, replacement, action, expected in plans:
                logits, captures, row = forward(model, pixels, evidence, f'{pair_id}:{side}:{name}',
                                                path=path, tokens=selected, replacement=replacement, action=action,
                                                reference=own['off_reference'], numeric_guard=numeric_guard)
                state = own['early'] if path == EARLY else own['late']
                check_landing(captures, state, selected, replacement, action=action)
                if expected is not None and not _same(logits, expected):
                    raise ValueError(f'{name} logits control failed')
                saved = {**row['score'], **distribution_effect(own['logits'], logits)}
                saved['raw_margin_change'] = saved['margin'] - own['score']['margin']
                saved['annotation_oriented_change'] = (1 if side == 'cat' else -1) * saved['raw_margin_change']
                if any(name.startswith(level + '_') for level in LEVELS):
                    level, kind = name.split('_', 1)
                    pre, post = captures['before_all'][0, selected], captures['after_all'][0, selected]
                    actual, receiver = distance(post, pre), distance(pre, torch.zeros_like(pre))
                    q = dose['levels'][level]
                    if (actual <= 0 or abs(actual - q) > dose['tolerances'][level] or
                            receiver <= 0 or actual > .4 * receiver):
                        raise ValueError('actual matched dose or receiver-relative bound differs')
                    if kind != 'opposite':
                        reference = info['conditions'][level + '_opposite']['actual_dose']
                        if abs(actual - reference) > .01 * reference:
                            raise ValueError('within-level actual doses differ')
                    saved.update(actual_dose=actual, receiver_norm=receiver,
                                 realized_over_receiver_norm=actual / receiver,
                                 token_doses=[distance(post[i], pre[i]) for i in range(len(selected))])
                info['conditions'][name] = saved
            logits, captures, row = forward(model, pixels, evidence, f'{pair_id}:{side}:post_clean',
                                            reference=own['off_reference'], numeric_guard=numeric_guard)
            if not all((_same(logits, own['logits']), _same(captures['early'], own['early']),
                        _same(captures['late'], own['late']))):
                raise ValueError('post-clean control failed')
            info['conditions']['post_clean'] = {**row['score'], **distribution_effect(own['logits'], logits)}
            if dose['status'] in {'matchable', 'limited-dose'}:
                info['movements'] = {level: {kind: info['conditions'][f'{level}_{kind}']['annotation_oriented_change']
                                             for kind in ARMS} for level in LEVELS}
                info['comparison'] = side_metrics(info['movements'])
            sides[side] = info
        statuses = {side: sides[side]['dose']['status'] for side in ('cat', 'dog')}
        if all(value == 'matchable' for value in statuses.values()):
            status = 'positive' if all(sides[side]['comparison']['side_pass'] for side in ('cat', 'dog')) else 'negative'
        else:
            status = next(value for value in ('dose-ineligible', 'dose-unmatchable', 'limited-dose')
                          if value in statuses.values())
        row = {'pair_id': pair_id, 'split': pair['split'], 'baseline_eligibility': bases[pair_id],
               'dose_matchability': statuses, 'status': status, 'sides': sides}
        evidence.json(f'{pair_id}.json', row)
        output.append(row)
    return output
