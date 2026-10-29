"""Deterministic, model-blind same-site target and reusable donor selection."""
from __future__ import annotations

import collections
import hashlib
import json
import math

from .vision_coco_cases import ANNOTATION_SHA256, _radius, _squares, _token_ids
from .vision_coco_small_dose_cases import PRIOR_CASES_SHA256, _rank, check_image_hashes


def greedy_same_site(candidates, *, count=8, excluded_ids=()):
    if type(count) is not int or count <= 0:
        raise ValueError('invalid pair count')
    groups, seen = {17: [], 18: []}, set()
    for row in candidates:
        image_id, category = row['image_id'], row['category_id']
        if type(image_id) is not int or image_id in seen or category not in groups:
            raise ValueError('invalid or duplicate candidate identity')
        seen.add(image_id)
        if image_id not in excluded_ids:
            groups[category].append(row)
    for rows in groups.values():
        rows.sort(key=_rank)
    targets, donors, output = set(), set(), []
    for cat in groups[17]:
        if cat['image_id'] in targets | donors:
            continue
        for dog in groups[18]:
            if dog['image_id'] in targets | donors:
                continue
            for inside in sorted(cat['inside'] & dog['inside'], key=lambda p: (_radius(p), p)):
                forbidden = targets | {cat['image_id'], dog['image_id']}
                def pick(category):
                    candidates_ = [row for row in groups[category]
                                   if row['image_id'] not in forbidden and inside in row['inside']]
                    return (min(candidates_, key=lambda row: (row['image_id'] not in donors, _rank(row)))
                            if candidates_ else None)
                cat_donor, dog_donor = pick(17), pick(18)
                if cat_donor is None or dog_donor is None:
                    continue
                targets.update((cat['image_id'], dog['image_id']))
                donors.update((cat_donor['image_id'], dog_donor['image_id']))
                output.append({'cat': cat, 'dog': dog, 'cat_donor': cat_donor,
                               'dog_donor': dog_donor, 'inside': inside})
                break
            if cat['image_id'] in targets:
                break
        if len(output) == count:
            return output
    raise ValueError('not enough same-site target/donor pairs')


def check_reused_images(prior_images, fresh_images, target_ids, donor_ids):
    """Permit repeated donor references but not changed or cross-ID bytes."""
    if len(target_ids) != len(set(target_ids)) or set(target_ids) & set(donor_ids):
        raise ValueError('target and donor IDs must be disjoint and unique')
    unique = {}
    for row in fresh_images:
        image_id = row['image_id']
        hashes = (row['jpeg_sha256'], row['pixel_sha256'])
        if image_id in unique and (unique[image_id]['jpeg_sha256'],
                                   unique[image_id]['pixel_sha256']) != hashes:
            raise ValueError('inconsistent duplicate donor image bytes')
        unique[image_id] = row
    if set(unique) != set(target_ids) | set(donor_ids):
        raise ValueError('target/donor image inventory differs')
    check_image_hashes(prior_images, list(unique.values()))


def select_same_site(annotation_bytes, prior_cases_bytes, *, count=8):
    """Select the fixed same-site sample from pinned annotation bytes only."""
    import PIL
    from PIL import Image, ImageDraw
    if PIL.__version__ != '11.3.0':
        raise ValueError('selection requires Pillow 11.3.0')
    if hashlib.sha256(annotation_bytes).hexdigest() != ANNOTATION_SHA256:
        raise ValueError('annotation bytes differ')
    if hashlib.sha256(prior_cases_bytes).hexdigest() != PRIOR_CASES_SHA256:
        raise ValueError('prior exposure manifest differs')
    prior = json.loads(prior_cases_bytes)
    if prior.get('schema') != 'vision-coco-vit-spatial-cases-v1' or len(prior.get('pairs', [])) != 12:
        raise ValueError('prior exposure inventory differs')
    excluded = {pair[side]['image_id'] for pair in prior['pairs'] for side in ('cat', 'dog')}
    if len(excluded) != 24:
        raise ValueError('prior exposure image IDs are not unique')
    data = json.loads(annotation_bytes)
    categories = {c['name']: c['id'] for c in data['categories']}
    if categories.get('cat') != 17 or categories.get('dog') != 18:
        raise ValueError('unexpected category identities')
    images = {row['id']: row for row in data['images']}
    licenses = {row['id']: row for row in data['licenses']}
    animal, all_annotations = {}, {}
    for annotation in data['annotations']:
        all_annotations.setdefault(annotation['image_id'], []).append(annotation)
        if annotation['category_id'] in (17, 18):
            animal.setdefault(annotation['image_id'], []).append(annotation)
    candidates = []
    for image_id, annotations in animal.items():
        image = images[image_id]
        if (image_id in excluded or image_id == 39769 or image['license'] not in (4, 5, 7, 8) or
                len({ann['category_id'] for ann in annotations}) != 1):
            continue
        eligible_instances = []
        for ann in annotations:
            polygons = ann.get('segmentation')
            if (ann['iscrowd'] or type(polygons) is not list or not polygons or
                    any(type(p) is not list or len(p) < 6 or len(p) % 2 for p in polygons)):
                continue
            if not all(math.isfinite(float(v)) for v in ann['bbox']):
                raise ValueError('nonfinite animal box')
            eligible_instances.append(ann)
        if not eligible_instances:
            continue
        ann = min(eligible_instances, key=lambda v: (-v['bbox'][2] * v['bbox'][3], v['id']))
        x, y, width, height = ann['bbox']
        iw, ih = image['width'], image['height']
        if (iw < 224 or ih < 224 or max(iw / ih, ih / iw) > 2 or
                not .08 <= width * height / (iw * ih) <= .70 or
                width * 224 / iw < 32 or height * 224 / ih < 32):
            continue
        inside, _, box = _squares(image, ann)
        if not inside:
            continue
        mask = Image.new('L', (224, 224), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in ann['segmentation']:
            if not all(math.isfinite(float(value)) for value in polygon):
                raise ValueError('nonfinite animal polygon')
            draw.polygon([(polygon[i] * 224 / iw, polygon[i + 1] * 224 / ih)
                          for i in range(0, len(polygon), 2)], fill=1)
        counts = {
            square: [sum(mask.crop((16 * (square[1] + dc), 16 * (square[0] + dr),
                                   16 * (square[1] + dc + 1), 16 * (square[0] + dr + 1))).getdata())
                     for dr in range(2) for dc in range(2)] for square in inside}
        qualified = {square for square, values in counts.items() if sum(values) >= 512}
        if not qualified:
            continue
        candidates.append({'image_id': image_id, 'category_id': ann['category_id'], 'inside': qualified,
                           'image': image, 'annotation': ann, 'animal_annotations': annotations,
                           'box': box, 'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                           'counts': counts})
    selected = greedy_same_site(candidates, count=count, excluded_ids=excluded)

    def describe(row, interior):
        image, ann = row['image'], row['annotation']
        left, top = 16 * interior[1], 16 * interior[0]
        right, bottom = left + 32, top + 32
        overlaps, other_boxes = [], []
        for other in all_annotations[image['id']]:
            if other['id'] == ann['id']:
                continue
            x, y, width, height = other['bbox']
            x0, x1 = x * 224 / image['width'], (x + width) * 224 / image['width']
            y0, y1 = y * 224 / image['height'], (y + height) * 224 / image['height']
            intersects = left < x1 and right > x0 and top < y1 and bottom > y0
            other_boxes.append({'annotation_id': other['id'], 'category_id': other['category_id'],
                                'bbox_xywh': other['bbox'], 'overlaps_inside': intersects})
            if intersects:
                overlaps.append({'annotation_id': other['id'], 'category_id': other['category_id']})
        overlaps.sort(key=lambda v: v['annotation_id'])
        other_boxes.sort(key=lambda v: v['annotation_id'])
        values = row['counts'][interior]
        return {'image_id': image['id'], 'file_name': image['file_name'],
                'width': image['width'], 'height': image['height'],
                'coco_url': image['coco_url'], 'flickr_url': image['flickr_url'],
                'license': licenses[image['license']], 'annotation_id': ann['id'],
                'animal_annotation_ids': sorted(a['id'] for a in row['animal_annotations']),
                'category_id': ann['category_id'], 'bbox_xywh': ann['bbox'],
                'bbox_resized_xyxy': list(row['box']), 'mask_sha256': row['mask_sha256'],
                'inside_block_mask_pixels': sum(values), 'inside_token_mask_pixels': values,
                'inside_other_box_overlaps': overlaps, 'other_boxes': other_boxes,
                'selection_rank_sha256': _rank(row)[0]}

    pairs = []
    for number, quad in enumerate(selected):
        inside = quad['inside']
        pairs.append({'pair_id': f'pair-{number:02d}', 'split': 'heldout',
                      **{role: describe(quad[role], inside)
                         for role in ('cat', 'dog', 'cat_donor', 'dog_donor')},
                      'box_block_row_col': list(inside), 'box_tokens': _token_ids(inside),
                      'squared_center_radius': _radius(inside)})
    donors = collections.Counter(pair[role]['image_id'] for pair in pairs for role in ('cat_donor', 'dog_donor'))
    diagnostics = {'previously_exposed_images': len(excluded),
                   'mask_qualified_candidates': {str(cat): sum(row['category_id'] == cat for row in candidates)
                                                 for cat in (17, 18)},
                   'selected_pairs': len(pairs), 'distinct_target_images': 2 * len(pairs),
                   'distinct_donor_images': len(donors),
                   'donor_side_counts': {str(key): value for key, value in sorted(donors.items())},
                   'minimum_mask_pixels_per_image': 512,
                   'mask_raster': 'Pillow 11.3.0 ImageDraw union of selected instance polygons at 224x224',
                   'selection': 'v2 hash rank; same-site greedy targets, prefer reusable non-target donors'}
    return pairs, diagnostics
