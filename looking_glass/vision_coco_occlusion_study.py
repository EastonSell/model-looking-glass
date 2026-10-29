"""Within-image occlusion and matched token-transfer measurements."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .vision_coco_cases import _token_ids
from .vision_coco_cls_study import distribution_effect
from .vision_coco_occlusion_cases import choose_geometry
from .vision_coco_study import (EARLY, LATE, _array, _bits, _replacement, _same, _tolerance,
                                check_denormals, check_landing, distance, forward, score)


def mask_pixels(pixels, position):
    """Set one processed 32-by-32 block to positive float32 zero."""
    if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.float32 or
            tuple(pixels.shape) != (1, 3, 224, 224) or not bool(torch.isfinite(pixels).all())):
        raise ValueError('expected finite float32 processed pixels')
    if (type(position) not in (tuple, list) or len(position) != 2 or
            any(type(v) is not int or not 0 <= v <= 12 for v in position)):
        raise ValueError('invalid block position')
    row, col = position
    result = pixels.detach().clone()
    result[:, :, 16 * row:16 * (row + 2), 16 * col:16 * (col + 2)] = 0.0
    return result


def log_group_probability(logits, category):
    """Compute log(group probability) without large-offset cancellation."""
    if type(category) is not int or category not in (17, 18):
        raise ValueError('unknown category')
    z = [float(value) for value in _array(logits).reshape(-1)]
    if len(z) != 1000:
        raise ValueError('expected 1000 finite logits')
    group = z[281:286] if category == 17 else z[151:269]
    group_max, all_max = max(group), max(z)
    group_sum = math.fsum(math.exp(value - group_max) for value in group)
    all_sum = math.fsum(math.exp(value - all_max) for value in z)
    return (group_max - all_max) + math.log(group_sum) - math.log(all_sum)


def prepare_transfers(clean, masked_animal, masked_background, animal, background, image_id):
    """Construct same-dose donor and random replacements at two sites."""
    sites = {'animal': animal, 'background': background}
    if (type(image_id) is not int or image_id < 0 or
            any(type(v) not in (list, tuple) or len(v) != 4 or len(set(v)) != 4 or
                any(type(i) is not int or not 1 <= i < 197 for i in v) for v in sites.values()) or
            set(animal) & set(background)):
        raise ValueError('invalid image or sites')
    states = (clean, masked_animal, masked_background)
    if any(not isinstance(v, torch.Tensor) or tuple(v.shape) != (1, 197, 192) or
           v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in states):
        raise ValueError('invalid activation state')
    blocks = {site: clean[0, tokens] for site, tokens in sites.items()}
    donors = {'animal': masked_animal[0, animal], 'background': masked_background[0, background]}
    norms = {site: distance(block, torch.zeros_like(block)) for site, block in blocks.items()}
    distances = {site: distance(block, donors[site]) for site, block in blocks.items()}
    result = {'receiver_norms': norms, 'donor_distances': distances,
              'status': 'dose-ineligible', 'reasons': []}
    for site in sites:
        if norms[site] <= 1e-6:
            result['reasons'].append(site + '-receiver-norm')
        if distances[site] <= 1e-6:
            result['reasons'].append(site + '-donor-distance')
    if result['reasons']:
        return result, {}
    q = min(.196 * norms['animal'], .196 * norms['background'], *distances.values())
    result['q'] = q
    if q <= 1e-6:
        result['reasons'].append('q')
        return result, {}
    signs = np.asarray([1.0 if hashlib.sha256(
        f'vision-coco-occlusion-v1:{image_id}:{i}'.encode('ascii')).digest()[0] & 1 else -1.0
                        for i in range(768)], dtype=np.float64).reshape(4, 192)
    replacements = {}
    for site, block in blocks.items():
        random = (_array(block).astype(np.float64) + (q / math.sqrt(768)) * signs).astype(np.float32)
        if not np.isfinite(random).all():
            raise ValueError('nonfinite random replacement')
        replacements[site] = {'transfer': _replacement(block, donors[site], q, distances[site]),
                              'random': torch.from_numpy(random.copy())}
    delivered = {site: {kind: distance(blocks[site], value) for kind, value in arms.items()}
                 for site, arms in replacements.items()}
    result.update(planned_norms=delivered, tolerance=_tolerance(q),
                  donor_scales={site: q / value for site, value in distances.items()})
    reference = delivered['animal']['transfer']
    for site, arms in delivered.items():
        if any(value <= 0 or abs(value - q) > _tolerance(q) or value > .2 * norms[site] or
               abs(value - reference) > .01 * reference for value in arms.values()):
            result.update(status='dose-unmatchable', reasons=[site + '-rounded-delivery'])
            return result, replacements
    result['status'] = ('limited-dose' if any(q / value < .02 for value in norms.values())
                        else 'matchable')
    return result, replacements


def image_decision(occlusion_contrast, transfer_contrast, random_animal, random_background):
    values = (occlusion_contrast, transfer_contrast, random_animal, random_background)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('invalid comparison values')
    return (occlusion_contrast >= .05 and transfer_contrast >= .025 and
            transfer_contrast - max(abs(random_animal), abs(random_background)) >= .025)


def aggregate_images(rows):
    if (type(rows) is not list or len(rows) > 8 or
            any(type(row) is not tuple or len(row) != 2 or
                type(row[0]) is not int or row[0] not in (17, 18) or
                type(row[1]) is not bool for row in rows)):
        raise ValueError('invalid evaluable-image records')
    n, positive = len(rows), sum(value for _, value in rows)
    categories = {category for category, _ in rows}
    status = ('inconclusive' if n < 4 or categories != {17, 18} else
              'positive' if positive * 4 >= n * 3 else 'negative')
    return {'status': status, 'evaluable': n, 'positive': positive,
            'cats': sum(category == 17 for category, _ in rows),
            'dogs': sum(category == 18 for category, _ in rows)}


def select_screened(images, observed):
    """Choose at most four per species from fixed recognized, geometric rows."""
    if (type(images) is not list or type(observed) is not list or len(images) != len(observed) or
            [row['image_id'] for row in images] != [row['image_id'] for row in observed] or
            len({row['image_id'] for row in images}) != len(images) or
            any(type(row.get('recognized')) is not bool for row in observed)):
        raise ValueError('screen identity or recognition differs')
    chosen, counts = [], {}
    for category, name in ((17, 'cat'), (18, 'dog')):
        rows = [(image, result) for image, result in zip(images, observed) if image['category_id'] == category]
        if [image['rank'] for image, _ in rows] != list(range(len(rows))):
            raise ValueError('screen rank differs')
        qualified = [image for image, result in rows
                     if result['recognized'] and image.get('occlusion_geometry') is not None]
        chosen.extend(qualified[:4])
        counts[name] = {'candidates': len(rows),
                        'geometry_qualified': sum(image.get('occlusion_geometry') is not None for image, _ in rows),
                        'screened': len(rows), 'recognized': sum(result['recognized'] for _, result in rows),
                        'recognized_geometry': len(qualified), 'selected': min(4, len(qualified))}
    if len(images) != sum(counts[name]['candidates'] for name in counts):
        raise ValueError('unknown screen category')
    return chosen, counts


def run_screen(model, images, pixels, evidence, *, numeric_guard=None):
    """Retain all clean logits before applying the fixed screen selection."""
    from .vision_coco_recognized_handoff import validate_candidates
    from .vision_coco_recognized_screen import screen_score
    validate_candidates(images)
    if (set(pixels) != {row['image_id'] for row in images} or
            any(choose_geometry(row) != row.get('occlusion_geometry') for row in images)):
        raise ValueError('screen pixels or fixed geometry differ')
    observed = []
    for image in images:
        image_id, label = image['image_id'], f'screen:{image["image_id"]}'
        evidence.reserve(label)
        check_denormals()
        if numeric_guard is not None:
            numeric_guard()
        value = pixels[image_id].detach().clone()
        original, output, saved = _bits(value), None, False
        try:
            with torch.no_grad():
                output = model(pixel_values=value)
            logits = output.logits.detach().cpu().contiguous()
            row = evidence.save_forward_bytes(label, {'logits': logits})
            saved = True
            evidence.active['phase'] = 'validation'
            check_denormals()
            if numeric_guard is not None:
                numeric_guard()
            if tuple(logits.shape) != (1, 1000):
                raise ValueError('screen logit shape differs')
            result = screen_score(_bits(logits), image['category_id'])
            if _bits(value) != original:
                raise ValueError('screen model changed input pixels')
            evidence.finish_forward(row, {'kind': 'screen', 'image_id': image_id}, result)
            observed.append({'image_id': image_id, **result})
        except BaseException as original_error:
            if not saved and output is not None:
                try:
                    value = output.logits
                    if isinstance(value, torch.Tensor) and value.dtype == torch.float32 and 0 < value.numel() <= 1000:
                        evidence.save_forward_bytes(label, {'logits': value.detach().cpu()})
                except BaseException as salvage_error:
                    if isinstance(original_error, Exception) and not isinstance(salvage_error, Exception):
                        raise
            evidence.active['return_observed'] = output is not None
            raise
    selected, counts = select_screened(images, observed)
    return observed, selected, counts


def verify_saved_screen(images, data, *, expected_result=None, expected_manifest_sha256=None):
    """Recompute screen scores and selection from the saved full logits."""
    from .vision_coco_recognized_handoff import validate_candidates
    from .vision_coco_recognized_screen import screen_score
    validate_candidates(images)
    if any(choose_geometry(row) != row.get('occlusion_geometry') for row in images):
        raise ValueError('saved screen geometry differs')
    data = Path(data)
    result_bytes = (data / 'screen-result.json').read_bytes()
    if expected_result is not None and result_bytes != expected_result:
        raise ValueError('saved screen differs from the published result')
    result = json.loads(result_bytes)
    if result.get('schema') != 'vision-occlusion-v1-screen-result':
        raise ValueError('screen result schema differs')
    if expected_manifest_sha256 is not None and result.get('manifest_sha256') != expected_manifest_sha256:
        raise ValueError('screen manifest differs')
    raw_by_id, observed = {}, []
    for index, image in enumerate(images, 1):
        row = json.loads((data / f'forward-{index:03d}.json').read_bytes())
        raw = (data / f'forward-{index:03d}.bin').read_bytes()
        layout = {'logits': {'shape': [1, 1000], 'offset': 0, 'bytes': 4000}}
        if (type(row.get('submission')) is not int or row['submission'] != index or
                row.get('label') != f"screen:{image['image_id']}" or
                row.get('file') != f'forward-{index:03d}.bin' or row.get('dtype') != 'little-endian-float32' or
                row.get('layout') != layout or len(raw) != 4000 or
                row.get('sha256') != hashlib.sha256(raw).hexdigest() or
                row.get('metadata') != {'kind': 'screen', 'image_id': image['image_id']}):
            raise ValueError('saved screen forward differs')
        score_row = screen_score(raw, image['category_id'])
        if row.get('score') != score_row:
            raise ValueError('saved screen score differs')
        observed.append({'image_id': image['image_id'], **score_row})
        raw_by_id[image['image_id']] = raw
    if result.get('screen') != observed:
        raise ValueError('screen result scores differ')
    selected, counts = select_screened(images, observed)
    if result.get('selected_ids') != [row['image_id'] for row in selected] or result.get('counts') != counts:
        raise ValueError('screen selected identities or counts differ')
    return {'raw_by_id': raw_by_id, 'selected': selected, 'counts': counts,
            'result_sha256': hashlib.sha256(result_bytes).hexdigest()}


def run_case(model, image, pixels, evidence, *, expected_logits=None, numeric_guard=None):
    """Retain one clean image, both occlusions and their matched token controls."""
    if numeric_guard is not None:
        numeric_guard()
    image_id, category = image['image_id'], image['category_id']
    if type(image_id) is not int or image_id < 0 or category not in (17, 18):
        raise ValueError('invalid image identity')
    geometry = choose_geometry(image)
    if geometry is None or geometry != image.get('occlusion_geometry'):
        raise ValueError('fixed image geometry differs')
    tokens = {site: geometry[site]['token_ids'] for site in ('animal', 'background')}
    inputs = {'clean': pixels.detach().clone()}
    for site in ('animal', 'background'):
        position = geometry[site]['row_col']
        if tokens[site] != _token_ids(tuple(position)):
            raise ValueError('pixel-to-token mapping differs')
        masked = mask_pixels(pixels, position)
        expected = pixels.detach().clone()
        row, col = position
        expected[:, :, 16 * row:16 * (row + 2), 16 * col:16 * (col + 2)] = 0.0
        if not _same(masked, expected):
            raise ValueError('occluded pixel bits differ')
        inputs[site] = masked
    input_hashes = {name: evidence.write(f'input-{image_id}-{name}.bin', _bits(value))
                    for name, value in inputs.items()}
    prefix = f'case-{image_id}'
    clean_logits, clean_states, clean_row = forward(model, inputs['clean'], evidence, prefix + ':clean',
                                                     numeric_guard=numeric_guard)
    if expected_logits is not None and _bits(clean_logits) != expected_logits:
        raise ValueError('clean logits differ from the saved screen')
    clean_logp = log_group_probability(clean_logits, category)
    _, reference = score(clean_logits)
    conditions, masked_states, masked_logits = {}, {}, {}

    def report(logits, record):
        value = log_group_probability(logits, category)
        return {**record['score'], **distribution_effect(clean_logits, logits),
                'correct_group_log_probability': value, 'drop': clean_logp - value}

    for site in ('animal', 'background'):
        logits, captures, record = forward(model, inputs[site], evidence, prefix + ':' + site + '_mask',
                                          reference=reference, numeric_guard=numeric_guard)
        masked_logits[site], masked_states[site] = logits, captures
        conditions[site + '_mask'] = report(logits, record)
    if numeric_guard is not None:
        numeric_guard()
    dose, replacements = prepare_transfers(clean_states['early'], masked_states['animal']['early'],
                                           masked_states['background']['early'], tokens['animal'],
                                           tokens['background'], image_id)
    if numeric_guard is not None:
        numeric_guard()
    logits, captures, record = forward(model, inputs['clean'], evidence, prefix + ':repeat',
                                       reference=reference, numeric_guard=numeric_guard)
    if not all((_same(logits, clean_logits), _same(captures['early'], clean_states['early']),
                _same(captures['late'], clean_states['late']))):
        raise ValueError('clean repeat differs')
    conditions['repeat'] = report(logits, record)
    plans = []
    for site in ('animal', 'background'):
        plans.extend([(site + '_null', EARLY, tokens[site], None, 'null', clean_logits),
                      (site + '_self', EARLY, tokens[site], clean_states['early'][0, tokens[site]],
                       'patch', clean_logits)])
    plans.append(('full_self', EARLY, None, clean_states['early'], 'patch', clean_logits))
    for site in ('animal', 'background'):
        plans.append((site + '_full', EARLY, None, masked_states[site]['early'], 'patch', masked_logits[site]))
    if dose['status'] in ('matchable', 'limited-dose'):
        for kind in ('transfer', 'random'):
            for site in ('animal', 'background'):
                plans.append((site + '_' + kind, EARLY, tokens[site], replacements[site][kind], 'patch', None))
    for site in ('animal', 'background'):
        plans.append((site + '_final', LATE, tokens[site], masked_states[site]['late'][0, tokens[site]],
                      'patch', clean_logits))
    for name, path, selected, replacement, action, expected in plans:
        logits, captures, record = forward(model, inputs['clean'], evidence, prefix + ':' + name,
                                           path=path, tokens=selected, replacement=replacement,
                                           action=action, reference=reference, numeric_guard=numeric_guard)
        state = clean_states['early'] if path == EARLY else clean_states['late']
        check_landing(captures, state, selected, replacement, action=action)
        if expected is not None and not _same(logits, expected):
            raise ValueError(name + ' logits control differs')
        value = report(logits, record)
        if name.endswith(('_transfer', '_random')):
            site = name.split('_', 1)[0]
            pre, post = captures['before_all'][0, selected], captures['after_all'][0, selected]
            actual = distance(post, pre)
            receiver = distance(pre, torch.zeros_like(pre))
            if (actual <= 0 or abs(actual - dose['q']) > dose['tolerance'] or
                    receiver <= 0 or actual > .2 * receiver):
                raise ValueError('actual dose or receiver cap differs')
            if name != 'animal_transfer':
                ref = conditions['animal_transfer']['actual_dose']
                if abs(actual - ref) > .01 * ref:
                    raise ValueError('actual doses do not match')
            value.update(actual_dose=actual, receiver_norm=receiver,
                         realized_over_receiver_norm=actual / receiver,
                         token_doses=[distance(post[i], pre[i]) for i in range(4)])
        conditions[name] = value
    logits, captures, record = forward(model, inputs['clean'], evidence, prefix + ':post_clean',
                                       reference=reference, numeric_guard=numeric_guard)
    if not all((_same(logits, clean_logits), _same(captures['early'], clean_states['early']),
                _same(captures['late'], clean_states['late']))):
        raise ValueError('post-clean control differs')
    conditions['post_clean'] = report(logits, record)
    comparisons = {'occlusion_contrast': conditions['animal_mask']['drop'] -
                                         conditions['background_mask']['drop']}
    if dose['status'] in ('matchable', 'limited-dose'):
        comparisons.update(transfer_contrast=conditions['animal_transfer']['drop'] -
                                             conditions['background_transfer']['drop'],
                           animal_random_drop=conditions['animal_random']['drop'],
                           background_random_drop=conditions['background_random']['drop'])
        comparisons['random_contrast'] = comparisons['animal_random_drop'] - comparisons['background_random_drop']
        comparisons['positive'] = image_decision(comparisons['occlusion_contrast'],
                                                  comparisons['transfer_contrast'],
                                                  comparisons['animal_random_drop'],
                                                  comparisons['background_random_drop'])
    result = {'image_id': image_id, 'category_id': category, 'geometry': geometry,
              'input_sha256': input_hashes, 'baseline': clean_row['score'],
              'baseline_log_probability': clean_logp, 'dose': dose,
              'conditions': conditions, 'comparisons': comparisons}
    if numeric_guard is not None:
        numeric_guard()
    evidence.json(f'{prefix}.json', result)
    return result
