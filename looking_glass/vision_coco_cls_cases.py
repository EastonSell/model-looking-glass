"""Model-free candidate ordering and prior-exposure binding for v5."""
import hashlib
import json

from .vision_coco_recognized_cases import previous_images as v4_previous_images
from .vision_coco_recognized_cases import select_inventory as v4_select_inventory
from .vision_coco_small_dose_cases import check_image_hashes

PRIOR_V4_SHA256 = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'


def rank(category, image_id):
    text = f'vision-coco-cls-v5:{category}:{image_id}'
    return hashlib.sha256(text.encode('ascii')).hexdigest(), image_id


def previous_images(v1_raw, v3_raw, v4_raw):
    if hashlib.sha256(v4_raw).hexdigest() != PRIOR_V4_SHA256:
        raise ValueError('previous v4 image inventory differs')
    previous = v4_previous_images(v1_raw, v3_raw)
    v4 = json.loads(v4_raw)
    images = v4.get('images')
    if type(images) is not list or len(images) != 64:
        raise ValueError('previous v4 image count differs')
    rows = [{key: row[key] for key in ('image_id', 'jpeg_sha256', 'pixel_sha256')} for row in images]
    check_image_hashes(previous, rows)
    return previous + rows


def select_inventory(archive, previous, *, count=32, authenticated=True):
    return v4_select_inventory(archive, [row['image_id'] for row in previous],
                               count=count, authenticated=authenticated, ranker=rank)
