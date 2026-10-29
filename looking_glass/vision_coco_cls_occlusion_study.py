"""Same-image CLS versus spatial occlusion transfer arithmetic."""
import hashlib
import math

import numpy as np
import torch

from .vision_coco_cases import _token_ids
from .vision_coco_cls_study import distribution_effect
from .vision_coco_occlusion_cases import choose_geometry
from .vision_coco_occlusion_study import aggregate_images, log_group_probability, mask_pixels
from .vision_coco_study import (EARLY, LATE, _array, _bits, _replacement, _same, _tolerance,
                                check_landing, distance, forward, score)


def run_response_masks(model, images, clean, pixels, clean_logits, evidence, *, numeric_guard=None):
    """Save both fixed masks for every recognized, geometry-qualified image."""
    if (len(images) != len(clean) or [x['image_id'] for x in images] != [x['image_id'] for x in clean] or
            set(pixels) != {row['image_id'] for row in images} or set(clean_logits) != set(pixels)):
        raise ValueError('screen inputs differ')
    rows = {}
    for image, baseline in zip(images, clean):
        ident = image['image_id']
        if not baseline['recognized'] or image.get('occlusion_geometry') is None:
            continue
        raw = clean_logits[ident]
        if (type(raw) is not bytes or len(raw) != 4000 or
                hashlib.sha256(raw).hexdigest() != baseline['logit_sha256']):
            raise ValueError('screen clean logits differ')
        clean_value = torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy()).reshape(1, 1000)
        baseline_logp = log_group_probability(clean_value, image['category_id'])
        original = pixels[ident].detach().clone()
        input_hashes = {'clean': evidence.write(f'screen-input-{ident}-clean.bin', _bits(original))}
        drops, logits_hashes, scores = {}, {}, {}
        for site in ('animal', 'background'):
            position = image['occlusion_geometry'][site]['row_col']
            masked = mask_pixels(original, position)
            expected = original.detach().clone()
            row, col = position
            expected[:, :, 16 * row:16 * (row + 2), 16 * col:16 * (col + 2)] = 0.0
            if not _same(masked, expected):
                raise ValueError('screen mask or untouched pixels differ')
            input_hashes[site] = evidence.write(f'screen-input-{ident}-{site}.bin', _bits(masked))
            logits, _captures, record = forward(model, masked, evidence, f'response:{ident}:{site}_mask',
                                                numeric_guard=numeric_guard)
            drops[site] = baseline_logp - log_group_probability(logits, image['category_id'])
            logits_hashes[site], scores[site] = hashlib.sha256(_bits(logits)).hexdigest(), record['score']
        rows[ident] = {'animal_drop': drops['animal'], 'background_drop': drops['background'],
                       'input_sha256': input_hashes, 'logit_sha256': logits_hashes, 'scores': scores}
    return rows


def select_response_screen(images, clean, masked):
    """Retain every screen failure and select the first response-eligible cases."""
    if (type(images) is not list or type(clean) is not list or type(masked) is not dict or
            len(images) != len(clean) or
            [row['image_id'] for row in images] != [row['image_id'] for row in clean] or
            len({row['image_id'] for row in images}) != len(images) or
            any(type(row.get('recognized')) is not bool for row in clean)):
        raise ValueError('screen identity or recognition differs')
    needed = {image['image_id'] for image, baseline in zip(images, clean)
              if baseline['recognized'] and image.get('occlusion_geometry') is not None}
    if set(masked) != needed:
        raise ValueError('mask inventory differs')
    selected, counts, rows = [], {}, []
    for category, name in ((17, 'cat'), (18, 'dog')):
        subset = [(image, baseline) for image, baseline in zip(images, clean)
                  if image['category_id'] == category]
        if [image['rank'] for image, _ in subset] != list(range(len(subset))):
            raise ValueError('screen rank differs')
        qualified = []
        for image, baseline in subset:
            ident, reasons = image['image_id'], []
            if image.get('occlusion_geometry') is None:
                reasons.append('geometry')
            if not baseline['recognized']:
                reasons.append('recognition')
            contrast = None
            if ident in masked:
                pair = masked[ident]
                values = (pair.get('animal_drop'), pair.get('background_drop'))
                if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                    raise ValueError('invalid mask drop')
                contrast = values[0] - values[1]
                if contrast < .05:
                    reasons.append('occlusion-contrast')
                else:
                    qualified.append(image)
            rows.append({'image_id': ident, 'category_id': category, 'reasons': reasons,
                         'occlusion_contrast': contrast})
        selected.extend(qualified[:4])
        counts[name] = {'candidates': len(subset),
                        'geometry_qualified': sum(image.get('occlusion_geometry') is not None for image, _ in subset),
                        'recognized': sum(baseline['recognized'] for _, baseline in subset),
                        'recognized_geometry': sum(image['image_id'] in needed for image, _ in subset),
                        'response_eligible': len(qualified), 'selected': min(4, len(qualified))}
    if len(rows) != len(images):
        raise ValueError('unknown screen category')
    return selected, counts, rows


def prepare_transfers(clean, masked_animal, masked_background, animal, background, image_id):
    """Build all four donor and three random replacements at a common dose."""
    if (type(image_id) is not int or image_id < 0 or
            any(type(v) not in (list, tuple) or len(v) != 4 or len(set(v)) != 4 or
                any(type(i) is not int or not 1 <= i < 197 for i in v)
                for v in (animal, background)) or set(animal) & set(background)):
        raise ValueError('invalid image or spatial sites')
    states = (clean, masked_animal, masked_background)
    if any(not isinstance(v, torch.Tensor) or tuple(v.shape) != (1, 197, 192) or
           v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in states):
        raise ValueError('invalid activation state')
    blocks = {'cls': clean[0, [0]], 'animal': clean[0, animal], 'background': clean[0, background]}
    donors = {'cls_animal': ('cls', masked_animal[0, [0]]),
              'cls_background': ('cls', masked_background[0, [0]]),
              'spatial_animal': ('animal', masked_animal[0, animal]),
              'spatial_background': ('background', masked_background[0, background])}
    norms = {site: distance(value, torch.zeros_like(value)) for site, value in blocks.items()}
    distances = {name: distance(blocks[site], value) for name, (site, value) in donors.items()}
    result = {'receiver_norms': norms, 'donor_distances': distances,
              'status': 'dose-ineligible', 'reasons': []}
    for name, value in norms.items():
        if value <= 1e-6:
            result['reasons'].append(name + '-receiver-norm')
    for name, value in distances.items():
        if value <= 1e-6:
            result['reasons'].append(name + '-donor-distance')
    if result['reasons']:
        return result, {}
    q = min(*(.196 * value for value in norms.values()), *distances.values())
    result['q'] = q
    if q <= 1e-6:
        result['reasons'].append('q')
        return result, {}
    replacements = {name: _replacement(blocks[site], value, q, distances[name])
                    for name, (site, value) in donors.items()}
    for site, block in blocks.items():
        label = 'cls' if site == 'cls' else 'spatial'
        count = block.numel()
        signs = np.asarray([1.0 if hashlib.sha256(
            f'vision-coco-cls-occlusion-v1:{image_id}:{label}:{i}'.encode('ascii')).digest()[0] & 1 else -1.0
                            for i in range(count)], dtype=np.float64).reshape(tuple(block.shape))
        value = (_array(block).astype(np.float64) + (q / math.sqrt(count)) * signs).astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite random replacement')
        replacements['random_' + site] = torch.from_numpy(value.copy())
    sites = {'cls_animal': 'cls', 'cls_background': 'cls',
             'spatial_animal': 'animal', 'spatial_background': 'background',
             'random_cls': 'cls', 'random_animal': 'animal', 'random_background': 'background'}
    delivered = {name: distance(blocks[sites[name]], value) for name, value in replacements.items()}
    result.update(planned_norms=delivered, tolerance=_tolerance(q),
                  donor_scales={name: q / value for name, value in distances.items()})
    reference = delivered['cls_animal']
    for name, value in delivered.items():
        if (value <= 0 or abs(value - q) > _tolerance(q) or value > .2 * norms[sites[name]] or
                abs(value - reference) > .01 * reference):
            result.update(status='dose-unmatchable', reasons=[name + '-rounded-delivery'])
            return result, replacements
    result['status'] = ('limited-dose' if any(q / value < .02 for value in norms.values()) else 'matchable')
    return result, replacements


def image_decision(b, t_cls, t_spatial, random_cls, random_animal, random_background):
    values = (b, t_cls, t_spatial, random_cls, random_animal, random_background)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('invalid comparison values')
    advantage = abs(b - t_spatial) - abs(b - t_cls)
    return (b >= .05 and t_cls >= .025 and
            t_cls - max(abs(random_cls), abs(random_animal), abs(random_background)) >= .025 and
            advantage >= .025)


def capture_initials(model, images, pixels, expected_inputs, expected_logits, evidence, *, numeric_guard=None):
    """Verify every selected clean and masked input/output before any patch."""
    ids = [row['image_id'] for row in images]
    if (len(ids) != len(set(ids)) or set(ids) != set(pixels) or
            set(ids) != set(expected_inputs) or set(ids) != set(expected_logits)):
        raise ValueError('initial image inventory differs')
    captured = {}
    for image in images:
        ident = image['image_id']
        geometry = choose_geometry(image)
        if geometry is None or geometry != image.get('occlusion_geometry'):
            raise ValueError('initial geometry differs')
        values = {'clean': pixels[ident].detach().clone()}
        for site in ('animal', 'background'):
            position = geometry[site]['row_col']
            if geometry[site]['token_ids'] != _token_ids(tuple(position)):
                raise ValueError('pixel-to-token mapping differs')
            values[site] = mask_pixels(values['clean'], position)
        if set(expected_inputs[ident]) != set(values) or set(expected_logits[ident]) != set(values):
            raise ValueError('screen input or logit inventory differs')
        one = {}
        for name, value in values.items():
            if _bits(value) != expected_inputs[ident][name]:
                raise ValueError('screen input pixels differ')
            evidence.write(f'initial-input-{ident}-{name}.bin', _bits(value))
            logits, states, record = forward(model, value, evidence, f'initial:{ident}:{name}',
                                             numeric_guard=numeric_guard)
            if _bits(logits) != expected_logits[ident][name]:
                raise ValueError('screen logits differ')
            one[name] = {'pixels': value, 'logits': logits, 'states': states, 'record': record}
        captured[ident] = one
    return captured


def run_case(model, image, initial, evidence, *, numeric_guard=None):
    """Compare the four donor and three random arms with fixed controls."""
    if numeric_guard is not None:
        numeric_guard()
    ident, category = image['image_id'], image['category_id']
    geometry = choose_geometry(image)
    if geometry is None or geometry != image.get('occlusion_geometry') or set(initial) != {'clean', 'animal', 'background'}:
        raise ValueError('case geometry or initial states differ')
    tokens = {'cls': [0], 'animal': geometry['animal']['token_ids'],
              'background': geometry['background']['token_ids']}
    clean = initial['clean']
    clean_logits, clean_states, clean_pixels = clean['logits'], clean['states'], clean['pixels']
    masked_states = {site: initial[site]['states'] for site in ('animal', 'background')}
    masked_logits = {site: initial[site]['logits'] for site in ('animal', 'background')}
    clean_logp = log_group_probability(clean_logits, category)
    _, reference = score(clean_logits)

    def report(logits, record):
        value = log_group_probability(logits, category)
        referenced_score, _ = score(logits, reference)
        return {**record['score'], **referenced_score, **distribution_effect(clean_logits, logits),
                'correct_group_log_probability': value, 'drop': clean_logp - value}

    conditions = {site + '_mask': report(initial[site]['logits'], initial[site]['record'])
                  for site in ('animal', 'background')}
    dose, replacements = prepare_transfers(clean_states['early'], masked_states['animal']['early'],
                                           masked_states['background']['early'], tokens['animal'],
                                           tokens['background'], ident)
    if numeric_guard is not None:
        numeric_guard()
    prefix = f'case-{ident}'
    logits, captures, record = forward(model, clean_pixels, evidence, prefix + ':repeat',
                                       reference=reference, numeric_guard=numeric_guard)
    if not all((_same(logits, clean_logits), _same(captures['early'], clean_states['early']),
                _same(captures['late'], clean_states['late']))):
        raise ValueError('clean repeat differs')
    conditions['repeat'] = report(logits, record)
    plans = []
    for site in ('cls', 'animal', 'background'):
        plans.extend([(site + '_null', EARLY, tokens[site], None, 'null', clean_logits),
                      (site + '_self', EARLY, tokens[site], clean_states['early'][0, tokens[site]],
                       'patch', clean_logits)])
    plans.append(('full_self', EARLY, None, clean_states['early'], 'patch', clean_logits))
    for site in ('animal', 'background'):
        plans.append((site + '_full', EARLY, None, masked_states[site]['early'], 'patch', masked_logits[site]))
    if dose['status'] in ('matchable', 'limited-dose'):
        arms = [('cls_animal', 'cls'), ('cls_background', 'cls'),
                ('spatial_animal', 'animal'), ('spatial_background', 'background'),
                ('random_cls', 'cls'), ('random_animal', 'animal'), ('random_background', 'background')]
        plans.extend((name, EARLY, tokens[site], replacements[name], 'patch', None) for name, site in arms)
    for site in ('animal', 'background'):
        plans.append((site + '_final', LATE, tokens[site], masked_states[site]['late'][0, tokens[site]],
                      'patch', clean_logits))
    active = {'cls_animal': 'cls', 'cls_background': 'cls', 'spatial_animal': 'animal',
              'spatial_background': 'background', 'random_cls': 'cls',
              'random_animal': 'animal', 'random_background': 'background'}
    for name, path, selected, replacement, action, expected in plans:
        logits, captures, record = forward(model, clean_pixels, evidence, prefix + ':' + name,
                                           path=path, tokens=selected, replacement=replacement,
                                           action=action, reference=reference, numeric_guard=numeric_guard)
        state = clean_states['early'] if path == EARLY else clean_states['late']
        check_landing(captures, state, selected, replacement, action=action)
        if expected is not None and not _same(logits, expected):
            raise ValueError(name + ' logits control differs')
        value = report(logits, record)
        if name in active:
            pre, post = captures['before_all'][0, selected], captures['after_all'][0, selected]
            actual = distance(post, pre)
            receiver = distance(pre, torch.zeros_like(pre))
            if (actual <= 0 or abs(actual - dose['q']) > dose['tolerance'] or receiver <= 0 or
                    actual > .2 * receiver):
                raise ValueError('actual dose or receiver cap differs')
            if name != 'cls_animal':
                ref = conditions['cls_animal']['actual_dose']
                if abs(actual - ref) > .01 * ref:
                    raise ValueError('actual doses do not match')
            value.update(actual_dose=actual, receiver_norm=receiver, realized_over_receiver_norm=actual / receiver,
                         token_doses=[distance(post[i], pre[i]) for i in range(len(selected))])
        conditions[name] = value
    logits, captures, record = forward(model, clean_pixels, evidence, prefix + ':post_clean',
                                       reference=reference, numeric_guard=numeric_guard)
    if not all((_same(logits, clean_logits), _same(captures['early'], clean_states['early']),
                _same(captures['late'], clean_states['late']))):
        raise ValueError('post-clean control differs')
    conditions['post_clean'] = report(logits, record)
    b = conditions['animal_mask']['drop'] - conditions['background_mask']['drop']
    comparisons = {'occlusion_contrast': b}
    if dose['status'] in ('matchable', 'limited-dose'):
        t_cls = conditions['cls_animal']['drop'] - conditions['cls_background']['drop']
        t_spatial = conditions['spatial_animal']['drop'] - conditions['spatial_background']['drop']
        randoms = [conditions[name]['drop'] for name in ('random_cls', 'random_animal', 'random_background')]
        comparisons.update(cls_contrast=t_cls, spatial_contrast=t_spatial, raw_site_difference=t_cls - t_spatial,
                           cls_residual=abs(b - t_cls), spatial_residual=abs(b - t_spatial),
                           residual_advantage=abs(b - t_spatial) - abs(b - t_cls),
                           random_cls=randoms[0], random_animal=randoms[1], random_background=randoms[2],
                           positive=image_decision(b, t_cls, t_spatial, *randoms))
    result = {'image_id': ident, 'category_id': category, 'geometry': geometry,
              'baseline': clean['record']['score'], 'baseline_log_probability': clean_logp,
              'dose': dose, 'conditions': conditions, 'comparisons': comparisons}
    if numeric_guard is not None:
        numeric_guard()
    evidence.json(prefix + '.json', result)
    return result
