"""Equal-absolute-dose CLS and spatial ViT comparison on saved tensors."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from .vision_coco_study import (EARLY, LATE, _array, _replacement, _same, _tolerance,
                                baseline_status, check_denormals, check_landing, distance, forward, score)


def _random(own, q, pair_id, side, site):
    x = _array(own).astype(np.float64)
    signs = [1.0 if hashlib.sha256(
        f'vision-coco-cls-v5:{pair_id}:{side}:{site}:{i}'.encode('ascii')).digest()[0] & 1 else -1.0
        for i in range(x.size)]
    result = (x + (q / math.sqrt(x.size)) * np.asarray(signs).reshape(x.shape)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite random replacement')
    return torch.from_numpy(result.copy())


def prepare_dose(own, opposite, same, spatial, pair_id, side):
    """Prepare six rounded replacements with one receiver-specific absolute q."""
    check_denormals()
    if (len(spatial) != 4 or len(set(spatial)) != 4 or
            any(type(t) is not int or not 1 <= t < 197 for t in spatial)):
        raise ValueError('invalid spatial tokens')
    tokens = {'cls': [0], 'spatial': spatial}
    blocks = {site: own[0, ids] for site, ids in tokens.items()}
    donors = {site: {'opposite': opposite[0, ids], 'same_species': same[0, ids]}
              for site, ids in tokens.items()}
    norms = {site: distance(block, torch.zeros_like(block)) for site, block in blocks.items()}
    distances = {site: {name: distance(blocks[site], donor) for name, donor in group.items()}
                 for site, group in donors.items()}
    result = {'receiver_norms': norms, 'distances': distances, 'status': 'dose-ineligible', 'reasons': []}
    for site in tokens:
        if norms[site] <= 1e-6:
            result['reasons'].append(f'{site}-receiver-norm')
        for name, value in distances[site].items():
            if value <= 1e-6:
                result['reasons'].append(f'{site}-{name.replace("_species", "")}-distance')
    if result['reasons']:
        return result, {}
    q = min(.099 * norms['cls'], .099 * norms['spatial'],
            *(value for group in distances.values() for value in group.values()))
    result['q'] = q
    if q <= 1e-6:
        result['reasons'].append('q')
        return result, {}
    requested = {site: {name: _replacement(blocks[site], donor, q, distances[site][name])
                        for name, donor in group.items()} for site, group in donors.items()}
    for site in tokens:
        requested[site]['random'] = _random(blocks[site], q, pair_id, side, site)
    planned = {site: {name: distance(blocks[site], value) for name, value in group.items()}
               for site, group in requested.items()}
    tolerance = _tolerance(q)
    result.update(tolerance=tolerance, planned_norms=planned,
                  donor_scales={site: {name: q / d for name, d in group.items()}
                                for site, group in distances.items()})
    reference = planned['cls']['opposite']
    if any(value <= 0 or abs(value - q) > tolerance or
           value > .10 * norms[site] or abs(value - reference) > .01 * reference
           for site, group in planned.items() for value in group.values()):
        result.update(status='dose-unmatchable', reasons=['rounded-delivery'])
    else:
        result['status'] = 'matchable'
    return result, requested


def side_metrics(movements):
    if type(movements) is not dict or set(movements) != {'cls', 'spatial'}:
        raise ValueError('invalid site movements')
    for value in movements.values():
        if (type(value) is not dict or set(value) != {'opposite', 'same_species', 'random'} or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in value.values())):
            raise ValueError('invalid or nonfinite movements')
    cls, spatial = movements['cls'], movements['spatial']
    adjusted_cls = cls['opposite'] - max(abs(cls['same_species']), abs(cls['random']))
    adjusted_spatial = spatial['opposite'] - max(abs(spatial['same_species']), abs(spatial['random']))
    return {'raw_opposite_difference': cls['opposite'] - spatial['opposite'],
            'adjusted_cls': adjusted_cls, 'adjusted_spatial': adjusted_spatial,
            'adjusted_difference': adjusted_cls - adjusted_spatial,
            'side_pass': cls['opposite'] >= .05 and adjusted_cls >= .025 and
                         adjusted_cls - adjusted_spatial >= .025}


def classify_pair(base, dose_statuses, movements):
    allowed = {'matchable', 'dose-ineligible', 'dose-unmatchable'}
    if (type(base.get('eligible')) is not bool or type(dose_statuses) is not dict or
            set(dose_statuses) != {'cat', 'dog'} or any(v not in allowed for v in dose_statuses.values())):
        raise ValueError('invalid baseline or dose status')
    if not base['eligible']:
        return 'baseline-ineligible'
    if any(value != 'matchable' for value in dose_statuses.values()):
        return 'dose-ineligible' if 'dose-ineligible' in dose_statuses.values() else 'dose-unmatchable'
    if type(movements) is not dict or set(movements) != {'cat', 'dog'}:
        raise ValueError('invalid paired movements')
    return 'positive' if all(side_metrics(movements[side])['side_pass'] for side in ('cat', 'dog')) else 'negative'


def distribution_effect(clean_logits, patched_logits):
    """Whole-vocabulary TV and KL(clean || patched) from saved finite logits."""
    clean = [float(x) for x in _array(clean_logits).reshape(-1)]
    patched = [float(x) for x in _array(patched_logits).reshape(-1)]
    if len(clean) != 1000 or len(patched) != 1000:
        raise ValueError('expected 1,000 logits')
    cmax, pmax = max(clean), max(patched)
    csum = math.fsum(math.exp(x - cmax) for x in clean)
    psum = math.fsum(math.exp(x - pmax) for x in patched)
    cp = [math.exp(x - cmax) / csum for x in clean]
    pp = [math.exp(x - pmax) / psum for x in patched]
    tv = .5 * math.fsum(abs(a - b) for a, b in zip(cp, pp))
    logz_delta = math.log(csum) - math.log(psum)
    kl = math.fsum(prob * ((c - p) - (cmax - pmax) - logz_delta)
                   for prob, c, p in zip(cp, clean, patched))
    if not math.isfinite(tv) or not math.isfinite(kl) or kl < -1e-12:
        raise ValueError('invalid distribution effect')
    return {'tv': tv, 'kl_clean_to_patched': max(0., kl),
            'top1_changed': clean.index(cmax) != patched.index(pmax)}


def aggregate_pairs(rows):
    if (type(rows) is not list or len(rows) > 4 or
            [r.get('pair_id') for r in rows] != [f'pair-{i:02d}' for i in range(len(rows))] or
            any(r.get('status') not in {'positive', 'negative', 'dose-ineligible', 'dose-unmatchable'}
                for r in rows)):
        raise ValueError('invalid selected-pair inventory')
    eligible = [r for r in rows if r['status'] in {'positive', 'negative'}]
    positives = sum(r['status'] == 'positive' for r in eligible)
    decision = ('inconclusive' if len(eligible) < 2 else
                'positive' if positives * 4 >= 3 * len(eligible) else 'negative')
    return {'status': decision, 'selected': len(rows), 'evaluable': len(eligible), 'positive': positives}


def run_comparison(model, pairs, pixel_by_id, evidence, *, check_clean=None, numeric_guard=None,
                   forward_fn=None):
    """Run all selected sides with saved pre/post patch witnesses and controls."""
    execute = forward if forward_fn is None else forward_fn
    if not callable(execute):
        raise ValueError('forward adapter must be callable')
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
            logits, captures, row = execute(model, pixel_by_id[image_id], evidence,
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
        pair_id, spatial = pair['pair_id'], pair['box_tokens']
        results = {}
        for side, other_side in (('cat', 'dog'), ('dog', 'cat')):
            if numeric_guard is not None:
                numeric_guard()
            image_id = pair[side]['image_id']
            own, other = clean[image_id], clean[pair[other_side]['image_id']]
            same = clean[pair[side + '_donor']['image_id']]
            pixels = pixel_by_id[image_id]
            dose, requested = prepare_dose(own['early'], other['early'], same['early'], spatial, pair_id, side)
            info = {'image_id': image_id, 'baseline': own['score'], 'dose': dose,
                    'same_species_donor_id': pair[side + '_donor']['image_id'],
                    'same_species_baseline': same['score'], 'conditions': {}}
            logits, captures, row = execute(model, pixels, evidence, f'{pair_id}:{side}:repeat',
                                            reference=own['off_reference'], numeric_guard=numeric_guard)
            if not all((_same(logits, own['logits']), _same(captures['early'], own['early']),
                        _same(captures['late'], own['late']))):
                raise ValueError('clean repeat failed')
            info['conditions']['repeat'] = {**row['score'], **distribution_effect(own['logits'], logits)}
            plans = []
            for site, ids in (('cls', [0]), ('spatial', spatial)):
                plans.extend([(f'null_{site}', EARLY, ids, None, 'null', own['logits']),
                              (f'self_{site}', EARLY, ids, own['early'][0, ids], 'patch', own['logits'])])
            plans.extend([('full_self', EARLY, None, own['early'], 'patch', own['logits']),
                          ('full_swap', EARLY, None, other['early'], 'patch', other['logits'])])
            if dose['status'] == 'matchable':
                for site, ids in (('cls', [0]), ('spatial', spatial)):
                    for name in ('opposite', 'same_species', 'random'):
                        plans.append((f'{site}_{name}', EARLY, ids, requested[site][name], 'patch', None))
            plans.append(('final_spatial', LATE, spatial, other['late'][0, spatial], 'patch', own['logits']))
            for name, path, ids, replacement, action, expected in plans:
                logits, captures, row = execute(model, pixels, evidence, f'{pair_id}:{side}:{name}',
                                                path=path, tokens=ids, replacement=replacement, action=action,
                                                reference=own['off_reference'], numeric_guard=numeric_guard)
                state = own['early'] if path == EARLY else own['late']
                check_landing(captures, state, ids, replacement, action=action)
                if expected is not None and not _same(logits, expected):
                    raise ValueError(f'{name} logits control failed')
                saved = {**row['score'], **distribution_effect(own['logits'], logits)}
                saved['raw_margin_change'] = saved['margin'] - own['score']['margin']
                saved['annotation_oriented_change'] = (1 if side == 'cat' else -1) * saved['raw_margin_change']
                if name in {f'{site}_{kind}' for site in ('cls', 'spatial')
                            for kind in ('opposite', 'same_species', 'random')}:
                    site, kind = name.split('_', 1)
                    pre, post = captures['before_all'][0, ids], captures['after_all'][0, ids]
                    actual, receiver = distance(post, pre), distance(pre, torch.zeros_like(pre))
                    if (actual <= 0 or abs(actual - dose['q']) > dose['tolerance'] or
                            receiver <= 0 or actual > .10 * receiver):
                        raise ValueError('actual matched dose or receiver-relative bound differs')
                    if name != 'cls_opposite':
                        reference = info['conditions']['cls_opposite']['actual_dose']
                        if abs(actual - reference) > .01 * reference:
                            raise ValueError('cross-site actual doses differ')
                    saved.update(actual_dose=actual, receiver_norm=receiver,
                                 realized_over_receiver_norm=actual / receiver,
                                 token_doses=[distance(post[i], pre[i]) for i in range(len(ids))])
                info['conditions'][name] = saved
            logits, captures, row = execute(model, pixels, evidence, f'{pair_id}:{side}:post_clean',
                                            reference=own['off_reference'], numeric_guard=numeric_guard)
            if not all((_same(logits, own['logits']), _same(captures['early'], own['early']),
                        _same(captures['late'], own['late']))):
                raise ValueError('post-clean control failed')
            info['conditions']['post_clean'] = {**row['score'], **distribution_effect(own['logits'], logits)}
            if dose['status'] == 'matchable':
                info['movements'] = {site: {kind: info['conditions'][f'{site}_{kind}']['annotation_oriented_change']
                                            for kind in ('opposite', 'same_species', 'random')}
                                     for site in ('cls', 'spatial')}
                info['comparison'] = side_metrics(info['movements'])
            results[side] = info
        statuses = {side: results[side]['dose']['status'] for side in ('cat', 'dog')}
        movements = ({side: results[side]['movements'] for side in ('cat', 'dog')}
                     if all(s == 'matchable' for s in statuses.values()) else None)
        if numeric_guard is not None:
            numeric_guard()
        status = classify_pair(bases[pair_id], statuses, movements)
        row = {'pair_id': pair_id, 'split': pair['split'], 'baseline_eligibility': bases[pair_id],
               'dose_matchability': statuses, 'status': status, 'sides': results}
        evidence.json(f'{pair_id}.json', row)
        output.append(row)
    return output
