"""Within-image occlusion and matched token-transfer measurements."""
import hashlib
import math

import numpy as np
import torch

from .vision_coco_study import _array, _replacement, _tolerance, distance


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
