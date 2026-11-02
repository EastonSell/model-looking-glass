"""Reconstruct the complete annotation-only v5 ranking without a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from review_vision_cls_v5_manifest import (ARCHIVE_SHA, MANIFEST_SHA, MEMBER_SHA,
                                           PRIOR, digest, file_digest, require, stream_array)


def qualify(image, animal):
    if len({ann['category_id'] for ann in animal}) != 1:
        return None
    polygons = [ann for ann in animal if not ann['iscrowd'] and type(ann.get('segmentation')) is list
                and ann['segmentation'] and all(type(poly) is list and len(poly) >= 6 and len(poly) % 2 == 0
                                                for poly in ann['segmentation'])]
    if not polygons:
        return None
    require(all(math.isfinite(float(n)) for ann in polygons for n in ann['bbox']), 'nonfinite source box')
    ann = min(polygons, key=lambda a: (-a['bbox'][2] * a['bbox'][3], a['id']))
    x, y, w, h = ann['bbox']
    iw, ih = image['width'], image['height']
    if not (.08 <= w * h / (iw * ih) <= .70 and w * 224 / iw >= 32 and h * 224 / ih >= 32):
        return None
    box = (x * 224 / iw, y * 224 / ih, (x + w) * 224 / iw, (y + h) * 224 / ih)
    squares = [(r, c) for r in range(13) for c in range(13)
               if 16*c >= box[0] and 16*r >= box[1] and 16*(c+2) <= box[2] and 16*(r+2) <= box[3]]
    if not squares:
        return None
    mask = Image.new('L', (224, 224), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in ann['segmentation']:
        require(all(math.isfinite(float(v)) for v in polygon), 'nonfinite polygon')
        draw.polygon([(polygon[i] * 224 / iw, polygon[i+1] * 224 / ih)
                      for i in range(0, len(polygon), 2)], fill=1)
    arr = np.asarray(mask, dtype=np.uint8)
    if any(int(arr[16*r:16*(r+2), 16*c:16*(c+2)].sum()) >= 512 for r, c in squares):
        return ann['id']
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    require(digest(raw) == MANIFEST_SHA and file_digest(args.archive) == ARCHIVE_SHA,
            'manifest or archive hash differs')
    manifest = json.loads(raw)
    names = ('vision_coco_spatial_v1.json', 'vision_coco_same_site_v3.json', 'vision_coco_recognized_v4.json')
    old_raw = [(args.repo / 'experiments' / name).read_bytes() for name in names]
    require([digest(item) for item in old_raw] == PRIOR, 'prior manifests differ')
    v1, v3, v4 = [json.loads(item) for item in old_raw]
    excluded = {pair[side]['image_id'] for doc, sides in ((v1, ('cat', 'dog')),
                (v3, ('cat', 'dog', 'cat_donor', 'dog_donor'))) for pair in doc['pairs'] for side in sides}
    excluded |= {row['image_id'] for row in v4['images']}
    require(len(excluded) == 107, 'prior exclusion size differs')
    with zipfile.ZipFile(args.archive) as zf:
        categories = {r['name']: r['id'] for r in stream_array(zf, 'categories')}
        require(categories.get('cat') == 17 and categories.get('dog') == 18, 'source species IDs differ')
        images = {}
        for image in stream_array(zf, 'images'):
            iw, ih = image['width'], image['height']
            if (image['id'] not in excluded and image['license'] in (4, 5, 7, 8) and
                    iw >= 224 and ih >= 224 and max(iw / ih, ih / iw) <= 2):
                require(image['id'] not in images, 'duplicate source image ID')
                images[image['id']] = image
        animals = {}
        for ann in stream_array(zf, 'annotations'):
            if ann['image_id'] in images and ann['category_id'] in (17, 18):
                animals.setdefault(ann['image_id'], []).append(ann)
    rank = lambda cat, ident: (hashlib.sha256(f'vision-coco-cls-v5:{cat}:{ident}'.encode('ascii')).hexdigest(), ident)
    # Grouping only after excluding mixed-species images is equivalent to the
    # public selector's global traversal with a species-specific hash key.
    groups = {17: [], 18: []}
    mixed = 0
    for ident, annotations in animals.items():
        species = {ann['category_id'] for ann in annotations}
        if len(species) != 1:
            mixed += 1
            continue
        category = next(iter(species))
        groups[category].append(ident)
    result = {17: [], 18: []}
    tested = {}
    for category in (17, 18):
        checked = 0
        for ident in sorted(groups[category], key=lambda i: rank(category, i)):
            checked += 1
            instance_id = qualify(images[ident], animals[ident])
            if instance_id is not None:
                result[category].append((ident, instance_id))
                if len(result[category]) == 32:
                    break
        require(len(result[category]) == 32, 'insufficient qualifying source images')
        tested[str(category)] = checked
    saved = {17: manifest['images'][:32], 18: manifest['images'][32:]}
    for category in (17, 18):
        require(result[category] == [(row['image_id'], row['annotation_id']) for row in saved[category]],
                'complete source ranking differs from saved first 32')
    require(file_digest(args.archive) == ARCHIVE_SHA, 'archive changed during ranking')
    report = {'status': 'complete-annotation-ranking-verified', 'manifest_sha256': MANIFEST_SHA,
              'archive_sha256': ARCHIVE_SHA, 'annotation_sha256': MEMBER_SHA, 'prior_images_excluded': 107,
              'candidate_images_after_image_filters': len(images), 'images_with_animal_annotations': len(animals),
              'mixed_species_images_excluded': mixed, 'tested_in_rank_order': tested,
              'selected_per_species': {'cat': 32, 'dog': 32}, 'model_forwards': 0,
              'scope': 'annotation-only reconstruction; origin, pretraining exposure, execution and chronology not attested'}
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
