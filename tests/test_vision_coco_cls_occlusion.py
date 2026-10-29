"""Model-free identity and ranking regressions for CLS/occlusion follow-up."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest


def test_prior_inventory_includes_every_exposed_occlusion_screen_image():
    from looking_glass.vision_coco_cls_occlusion_cases import previous_images

    root = Path(__file__).resolve().parents[1] / 'experiments'
    names = ('vision_coco_spatial_v1.json', 'vision_coco_same_site_v3.json',
             'vision_coco_recognized_v4.json', 'vision_coco_cls_compare_v5.json',
             'vision_coco_spatial_dose_v6.json', 'vision_coco_occlusion_v1.json')
    raw = [(root / name).read_bytes() for name in names]
    images = previous_images(*raw)
    assert len(images) == len({row['image_id'] for row in images}) == 299
    assert len({row['jpeg_sha256'] for row in images}) == 299
    assert len({row['pixel_sha256'] for row in images}) == 299
    with pytest.raises(ValueError, match='occlusion inventory'):
        previous_images(*raw[:-1], raw[-1] + b' ')


def test_discovery_is_the_fixed_top_two_prior_contrasts_per_species():
    from looking_glass.vision_coco_cls_occlusion_cases import select_discovery

    root = Path(__file__).resolve().parents[1]
    manifest = (root / 'experiments/vision_coco_occlusion_v1.json').read_bytes()
    audit = (root / 'results/vision-coco-occlusion-v1-20261029/evaluation-independent-audit.json').read_bytes()
    selected = select_discovery(manifest, audit)
    assert [row['image_id'] for row in selected] == [404350, 151497, 470798, 135116]
    with pytest.raises(ValueError, match='prior evaluation audit'):
        select_discovery(manifest, audit + b' ')


def test_fresh_selector_uses_new_rank_and_excludes_prior_ids(tmp_path):
    from looking_glass.vision_coco_cls_occlusion_cases import select_inventory
    from looking_glass.vision_coco_recognized_cases import ANNOTATION_MEMBER

    images, annotations = [], []
    for category in (17, 18):
        for number in range(8):
            ident = category * 100 + number
            images.append({'id': ident, 'file_name': f'{ident:012d}.jpg',
                           'width': 224, 'height': 224, 'license': 4})
            annotations.append({'id': ident, 'image_id': ident, 'category_id': category,
                                'iscrowd': 0, 'bbox': [40, 40, 144, 144],
                                'segmentation': [[40, 40, 184, 40, 184, 184, 40, 184]]})
    archive = tmp_path / 'annotations.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr(ANNOTATION_MEMBER, json.dumps({'licenses': [{'id': 4}], 'images': images,
                                                    'annotations': annotations,
                                                    'categories': [{'id': 17, 'name': 'cat'},
                                                                   {'id': 18, 'name': 'dog'}]}))
    selected = select_inventory(archive, [{'image_id': 1700}], count=3, authenticated=False)
    for category in (17, 18):
        eligible = [r['id'] for r in images if r['id'] // 100 == category and r['id'] != 1700]
        expected = sorted(eligible, key=lambda ident: (hashlib.sha256(
            f'vision-coco-cls-occlusion-v1:{category}:{ident}'.encode('ascii')).hexdigest(), ident))[:3]
        actual = [row for row in selected if row['category_id'] == category]
        assert [row['image_id'] for row in actual] == expected
        assert [row['rank'] for row in actual] == [0, 1, 2]
        assert all('occlusion_geometry' in row for row in actual)
