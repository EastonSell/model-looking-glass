"""Model-free identities and rankings for same-image CLS/occlusion study."""
import hashlib
import json
import math

from .vision_coco_occlusion_cases import choose_geometry, previous_images as prior_images
from .vision_coco_recognized_cases import select_inventory as select_v4_inventory
from .vision_coco_small_dose_cases import check_image_hashes

PRIOR_OCCLUSION_SHA256 = 'aa61b294710f5a6cd9682dc783085c7079f1addfbad8257fdbd893d143ce4cef'
PRIOR_EVALUATION_AUDIT_SHA256 = 'db69cb0c4d0e93dfd578ec03e035399ca19f3b3447f6d2db7575d2034a7313b5'


def rank(category, image_id):
    raw = f'vision-coco-cls-occlusion-v1:{category}:{image_id}'.encode('ascii')
    return hashlib.sha256(raw).hexdigest(), image_id


def previous_images(v1_raw, v3_raw, v4_raw, v5_raw, v6_raw, occlusion_raw):
    if hashlib.sha256(occlusion_raw).hexdigest() != PRIOR_OCCLUSION_SHA256:
        raise ValueError('previous occlusion inventory differs')
    previous = prior_images(v1_raw, v3_raw, v4_raw, v5_raw, v6_raw)
    images = json.loads(occlusion_raw).get('images')
    if type(images) is not list or len(images) != 64:
        raise ValueError('previous occlusion image count differs')
    rows = [{key: image[key] for key in ('image_id', 'jpeg_sha256', 'pixel_sha256')}
            for image in images]
    check_image_hashes(previous, rows)
    return previous + rows


def select_inventory(archive, previous, *, count=32, authenticated=True):
    images = select_v4_inventory(archive, [row['image_id'] for row in previous],
                                 count=count, authenticated=authenticated, ranker=rank)
    for image in images:
        image['occlusion_geometry'] = choose_geometry(image)
    return images


def select_discovery(manifest_raw, audit_raw):
    """Select the two largest previously observed mask contrasts per species."""
    if hashlib.sha256(manifest_raw).hexdigest() != PRIOR_OCCLUSION_SHA256:
        raise ValueError('previous occlusion inventory differs')
    if hashlib.sha256(audit_raw).hexdigest() != PRIOR_EVALUATION_AUDIT_SHA256:
        raise ValueError('prior evaluation audit differs')
    manifest, audit = json.loads(manifest_raw), json.loads(audit_raw)
    result = audit['result']
    if (audit.get('schema') != 'vision-occlusion-independent-intervention-audit-v1' or
            audit.get('synthetic') is not False or result.get('mode') != 'evaluation' or
            result.get('manifest_sha256') != PRIOR_OCCLUSION_SHA256 or len(result.get('cases', [])) != 8):
        raise ValueError('prior evaluation identity differs')
    lookup = {row['image_id']: row for row in manifest['images']}
    selected = []
    for category in (17, 18):
        rows = []
        for case in result['cases']:
            if case['category_id'] != category:
                continue
            ident, contrast = case['image_id'], case['comparisons']['occlusion_contrast']
            if (type(ident) is not int or type(contrast) not in (int, float) or
                    not math.isfinite(contrast) or ident not in lookup or
                    lookup[ident]['category_id'] != category or
                    choose_geometry(lookup[ident]) != lookup[ident]['occlusion_geometry']):
                raise ValueError('prior evaluation case differs')
            rows.append((-contrast, ident, lookup[ident]))
        if len(rows) != 4:
            raise ValueError('prior evaluation species count differs')
        selected.extend(row[2] for row in sorted(rows)[:2])
    if [row['image_id'] for row in selected] != [404350, 151497, 470798, 135116]:
        raise ValueError('fixed discovery identities differ')
    return selected
