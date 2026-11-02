"""Model-free v6 candidate ordering and prior-exposure binding."""
import hashlib
import json

from .vision_coco_cls_cases import previous_images as v5_previous_images
from .vision_coco_recognized_cases import select_inventory as select_v4_inventory
from .vision_coco_small_dose_cases import check_image_hashes

PRIOR_V5_SHA256 = 'd2fa15c4287f15f78ff5f1a1cc1eab2551d73ee2ac876b5f7bb956f31b1d2931'


def rank(category, image_id):
    raw = f'vision-coco-spatial-dose-v6:{category}:{image_id}'.encode('ascii')
    return hashlib.sha256(raw).hexdigest(), image_id


def previous_images(v1_raw, v3_raw, v4_raw, v5_raw):
    if hashlib.sha256(v5_raw).hexdigest() != PRIOR_V5_SHA256:
        raise ValueError('previous v5 image inventory differs')
    previous = v5_previous_images(v1_raw, v3_raw, v4_raw)
    v5 = json.loads(v5_raw)
    images = v5.get('images')
    if type(images) is not list or len(images) != 64:
        raise ValueError('previous v5 image count differs')
    rows = [{key: image[key] for key in ('image_id', 'jpeg_sha256', 'pixel_sha256')}
            for image in images]
    check_image_hashes(previous, rows)
    return previous + rows


def select_inventory(archive, previous, *, count=32, authenticated=True):
    return select_v4_inventory(archive, [row['image_id'] for row in previous],
                               count=count, authenticated=authenticated, ranker=rank)
