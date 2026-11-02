"""Model-free candidate ordering and within-image occlusion geometry."""
import hashlib
import json
import math

from .vision_coco_cases import _radius, _token_ids
from .vision_coco_recognized_cases import select_inventory as select_v4_inventory
from .vision_coco_small_dose_cases import check_image_hashes
from .vision_coco_spatial_dose_cases import previous_images as v6_previous_images

PRIOR_V6_SHA256 = '586e0abe9c06a2eb02ad74dc8bda38137349cf1d5953363b2a190ae59d2f0075'
PRIOR_V6_SCREEN_SHA256 = '2e51bd187d6b0762874030cbe3dbcd64c16a532f969e09e56730816cced0777e'


def rank(category, image_id):
    raw = f'vision-coco-occlusion-v1:{category}:{image_id}'.encode('ascii')
    return hashlib.sha256(raw).hexdigest(), image_id


def previous_images(v1_raw, v3_raw, v4_raw, v5_raw, v6_raw):
    if hashlib.sha256(v6_raw).hexdigest() != PRIOR_V6_SHA256:
        raise ValueError('previous v6 image inventory differs')
    previous = v6_previous_images(v1_raw, v3_raw, v4_raw, v5_raw)
    images = json.loads(v6_raw).get('images')
    if type(images) is not list or len(images) != 64:
        raise ValueError('previous v6 image count differs')
    rows = [{key: image[key] for key in ('image_id', 'jpeg_sha256', 'pixel_sha256')}
            for image in images]
    check_image_hashes(previous, rows)
    return previous + rows


def _overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _rectangle(row, col):
    return (16 * col, 16 * row, 16 * (col + 2), 16 * (row + 2))


def choose_geometry(image):
    """Select the fixed same-radius object/background squares or return None."""
    width, height = image['width'], image['height']
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('invalid image dimensions')
    box = image['bbox_resized_xyxy']
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('invalid animal box')
    other = []
    for annotation in image['other_boxes']:
        x, y, w, h = annotation['bbox_xywh']
        if not all(math.isfinite(float(v)) for v in (x, y, w, h)) or w < 0 or h < 0:
            raise ValueError('invalid other box')
        resized = (224 * x / width, 224 * y / height,
                   224 * (x + w) / width, 224 * (y + h) / height)
        other.append((annotation, resized))
    boxes = [box, *(value for _, value in other)]
    candidates = []
    seen = set()
    for site in image['sites']:
        row, col = site['row_col']
        if (type(row) is not int or type(col) is not int or not 0 <= row <= 12 or
                not 0 <= col <= 12 or (row, col) in seen or
                site['squared_center_radius'] != _radius((row, col)) or
                site['token_ids'] != _token_ids((row, col))):
            raise ValueError('invalid object site')
        seen.add((row, col))
        counts = site['token_mask_pixels']
        if (type(counts) is not list or len(counts) != 4 or
                any(type(v) is not int or not 0 <= v <= 256 for v in counts) or
                type(site['mask_pixels']) is not int or sum(counts) != site['mask_pixels']):
            raise ValueError('invalid object occupancy')
        rect = _rectangle(row, col)
        if not (box[0] <= rect[0] and box[1] <= rect[1] and
                rect[2] <= box[2] and rect[3] <= box[3]):
            raise ValueError('object site is not wholly inside its box')
        if site['mask_pixels'] < 768:
            continue
        for b_row in range(13):
            for b_col in range(13):
                if _radius((b_row, b_col)) != _radius((row, col)):
                    continue
                background = _rectangle(b_row, b_col)
                if any(_overlaps(background, value) for value in boxes):
                    continue
                key = (abs(row - b_row) + abs(col - b_col), _radius((row, col)),
                       row, col, b_row, b_col)
                candidates.append((key, site, background))
    if not candidates:
        return None
    key, site, _ = min(candidates, key=lambda value: value[0])
    row, col, b_row, b_col = key[2:]
    overlaps = [{'annotation_id': ann['annotation_id'], 'category_id': ann['category_id']}
                for ann, other_box in other if _overlaps(_rectangle(row, col), other_box)]
    overlaps.sort(key=lambda value: value['annotation_id'])
    return {'animal': {'row_col': [row, col], 'token_ids': _token_ids((row, col)),
                       'mask_pixels': site['mask_pixels'],
                       'token_mask_pixels': site['token_mask_pixels'], 'other_box_overlaps': overlaps},
            'background': {'row_col': [b_row, b_col], 'token_ids': _token_ids((b_row, b_col))},
            'squared_center_radius': key[1], 'manhattan_distance': key[0]}


def select_inventory(archive, previous, *, count=32, authenticated=True):
    images = select_v4_inventory(archive, [row['image_id'] for row in previous],
                                 count=count, authenticated=authenticated, ranker=rank)
    for image in images:
        image['occlusion_geometry'] = choose_geometry(image)
    return images


def select_discovery(v6_raw, screen_raw):
    """Reconstruct the four fixed already-exposed discovery images."""
    if hashlib.sha256(v6_raw).hexdigest() != PRIOR_V6_SHA256:
        raise ValueError('previous v6 image inventory differs')
    if hashlib.sha256(screen_raw).hexdigest() != PRIOR_V6_SCREEN_SHA256:
        raise ValueError('previous v6 screen differs')
    manifest, screen = json.loads(v6_raw), json.loads(screen_raw)
    images, scores = manifest['images'], screen['screen']
    if (len(images) != 64 or len(scores) != 64 or screen['manifest_sha256'] != PRIOR_V6_SHA256 or
            [x['image_id'] for x in images] != [x['image_id'] for x in scores] or
            any(type(x['recognized']) is not bool for x in scores)):
        raise ValueError('previous v6 screen inventory differs')
    out = []
    for category in (17, 18):
        eligible = []
        for image, score in zip(images, scores):
            if image['category_id'] == category and score['recognized']:
                geometry = choose_geometry(image)
                if geometry is not None:
                    eligible.append({**image, 'occlusion_geometry': geometry})
        eligible.sort(key=lambda image: image['rank'])
        if len(eligible) < 2:
            raise ValueError('insufficient fixed discovery geometry')
        out.extend(eligible[:2])
    if [x['image_id'] for x in out] != [328437, 142719, 476736, 503893]:
        raise ValueError('fixed discovery identities differ')
    return out
