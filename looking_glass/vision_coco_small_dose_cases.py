"""Model-blind, disjoint target/donor selection for the small-dose study."""
from __future__ import annotations

import hashlib
import json
import math

from .vision_coco_cases import ANNOTATION_SHA256, _radius, _squares, _token_ids

PRIOR_CASES_SHA256 = '8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207'


def check_image_hashes(prior_images, fresh_images):
    """Reject exact identity, JPEG or pixel collisions before any model call."""
    for field in ('image_id', 'jpeg_sha256', 'pixel_sha256'):
        prior = [row[field] for row in prior_images]
        fresh = [row[field] for row in fresh_images]
        if (any(value is None or value == '' for value in prior + fresh) or
                len(fresh) != len(set(fresh)) or set(prior) & set(fresh)):
            raise ValueError(f'duplicate or missing {field}')


def _rank(row):
    text = f"vision-coco-small-dose-v2:{row['category_id']}:{row['image_id']}"
    return hashlib.sha256(text.encode('ascii')).hexdigest(), row['image_id']


def greedy_quadruplets(candidates, *, count=8, excluded_ids=()):
    """Select disjoint target/donor quadruplets using only saved geometry.

    Each input row has an image_id, COCO category_id and sets of qualified
    inside and outside 2-by-2 block coordinates. Geometry and mask formation
    are deliberately separate from this deterministic greedy decision.
    """
    if type(count) is not int or count <= 0:
        raise ValueError('invalid selected pair count')
    seen = set()
    groups = {17: [], 18: []}
    for row in candidates:
        image_id, category = row['image_id'], row['category_id']
        if type(image_id) is not int or image_id in seen or category not in groups:
            raise ValueError('invalid or duplicate candidate identity')
        seen.add(image_id)
        if image_id not in excluded_ids:
            groups[category].append(row)
    for rows in groups.values():
        rows.sort(key=_rank)
    used, output = set(), []
    for cat in groups[17]:
        if cat['image_id'] in used:
            continue
        for dog in groups[18]:
            if dog['image_id'] in used:
                continue
            common_inside = cat['inside'] & dog['inside']
            common_outside = cat['outside'] & dog['outside']
            choices = [(a, b) for a in common_inside for b in common_outside
                       if _radius(a) == _radius(b)]
            choices.sort(key=lambda pair: (abs(pair[0][0] - pair[1][0]) +
                                           abs(pair[0][1] - pair[1][1]),
                                           _radius(pair[0]), pair))
            for inside, outside in choices:
                excluded = used | {cat['image_id'], dog['image_id']}
                cat_donor = next((v for v in groups[17] if v['image_id'] not in excluded and
                                  inside in v['inside']), None)
                dog_donor = next((v for v in groups[18] if v['image_id'] not in excluded and
                                  inside in v['inside']), None)
                if cat_donor is None or dog_donor is None:
                    continue
                output.append({'cat': cat, 'dog': dog, 'cat_donor': cat_donor,
                               'dog_donor': dog_donor, 'inside': inside, 'outside': outside})
                used.update((cat['image_id'], dog['image_id'], cat_donor['image_id'],
                             dog_donor['image_id']))
                break
            if cat['image_id'] in used:
                break
        if len(output) == count:
            return output
    raise ValueError('not enough geometry-qualified disjoint quadruplets')


def select_small_dose(annotation_bytes, prior_cases_bytes, *, count=8):
    """Reconstruct selected quadruplets from pinned annotations, with no model."""
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
        if (image_id in excluded or image_id == 39769 or len(annotations) != 1 or
                annotations[0]['iscrowd'] or image['license'] not in (4, 5, 7, 8)):
            continue
        ann = annotations[0]
        x, y, width, height = ann['bbox']
        if not all(math.isfinite(float(v)) for v in (x, y, width, height)):
            raise ValueError('nonfinite animal box')
        iw, ih = image['width'], image['height']
        if (iw < 224 or ih < 224 or max(iw / ih, ih / iw) > 2 or
                not .08 <= width * height / (iw * ih) <= .70 or
                width * 224 / iw < 32 or height * 224 / ih < 32):
            continue
        inside, outside, box = _squares(image, ann)
        if not inside or not outside:
            continue
        polygons = ann.get('segmentation')
        if (type(polygons) is not list or not polygons or
                any(type(p) is not list or len(p) < 6 or len(p) % 2 for p in polygons)):
            continue
        mask = Image.new('L', (224, 224), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
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
        candidates.append({'image_id': image_id, 'category_id': ann['category_id'],
                           'inside': qualified, 'outside': outside, 'image': image, 'annotation': ann,
                           'box': box, 'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                           'counts': counts})
    selected = greedy_quadruplets(candidates, count=count, excluded_ids=excluded)

    def describe(row, interior, control):
        image, ann = row['image'], row['annotation']
        left, top = 16 * control[1], 16 * control[0]
        right, bottom = left + 32, top + 32
        overlaps = []
        for other in all_annotations[image['id']]:
            if other['id'] == ann['id']:
                continue
            x, y, width, height = other['bbox']
            x0, x1 = x * 224 / image['width'], (x + width) * 224 / image['width']
            y0, y1 = y * 224 / image['height'], (y + height) * 224 / image['height']
            if left < x1 and right > x0 and top < y1 and bottom > y0:
                overlaps.append({'annotation_id': other['id'], 'category_id': other['category_id']})
        overlaps.sort(key=lambda v: v['annotation_id'])
        values = row['counts'][interior]
        return {'image_id': image['id'], 'file_name': image['file_name'],
                'width': image['width'], 'height': image['height'],
                'coco_url': image['coco_url'], 'flickr_url': image['flickr_url'],
                'license': licenses[image['license']], 'annotation_id': ann['id'],
                'category_id': ann['category_id'], 'bbox_xywh': ann['bbox'],
                'bbox_resized_xyxy': list(row['box']), 'mask_sha256': row['mask_sha256'],
                'inside_block_mask_pixels': sum(values), 'inside_token_mask_pixels': values,
                'control_other_box_overlaps': overlaps, 'selection_rank_sha256': _rank(row)[0]}

    pairs = []
    for number, quad in enumerate(selected):
        interior, control = quad['inside'], quad['outside']
        pairs.append({'pair_id': f'pair-{number:02d}', 'split': 'heldout',
                      **{key: describe(quad[key], interior, control)
                         for key in ('cat', 'dog', 'cat_donor', 'dog_donor')},
                      'box_block_row_col': list(interior), 'control_block_row_col': list(control),
                      'box_tokens': _token_ids(interior), 'control_tokens': _token_ids(control),
                      'squared_center_radius': _radius(interior)})
    diagnostics = {'previously_exposed_images': len(excluded),
                   'mask_qualified_candidates': {str(cat): sum(r['category_id'] == cat for r in candidates)
                                                 for cat in (17, 18)},
                   'selected_pairs': len(pairs), 'selected_unique_images': len(pairs) * 4,
                   'minimum_mask_pixels_per_image': 512,
                   'mask_raster': 'Pillow 11.3.0 ImageDraw filled COCO polygons directly at 224x224',
                   'selection': 'v2 hash-ranked greedy disjoint quadruplets; annotation only'}
    return pairs, diagnostics
