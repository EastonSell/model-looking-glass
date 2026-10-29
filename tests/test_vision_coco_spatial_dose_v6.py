import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from looking_glass import vision_coco_spatial_dose_cases as cases


def test_prior_exposure_union_and_v5_digest_are_fixed():
    root = Path(__file__).resolve().parents[1] / 'experiments'
    raw = [(root / name).read_bytes() for name in (
        'vision_coco_spatial_v1.json', 'vision_coco_same_site_v3.json',
        'vision_coco_recognized_v4.json', 'vision_coco_cls_compare_v5.json')]
    previous = cases.previous_images(*raw)
    assert len(previous) == len({row['image_id'] for row in previous}) == 171
    assert {row['image_id'] for row in json.loads(raw[3])['images']} <= {row['image_id'] for row in previous}
    with pytest.raises(ValueError, match='v5'):
        cases.previous_images(*raw[:3], raw[3] + b' ')


def test_fabricated_rank_and_exclusion_use_the_new_fixed_seed(tmp_path):
    from looking_glass.vision_coco_recognized_cases import ANNOTATION_MEMBER

    images, annotations = [], []
    for category in (17, 18):
        for n in range(8):
            ident = category * 100 + n
            images.append({'id': ident, 'file_name': f'{ident:012d}.jpg',
                           'width': 224, 'height': 224, 'license': 4})
            annotations.append({'id': ident, 'image_id': ident, 'category_id': category,
                                'iscrowd': 0, 'bbox': [40, 40, 144, 144],
                                'segmentation': [[40, 40, 184, 40, 184, 184, 40, 184]]})
    archive = tmp_path / 'annotation.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr(ANNOTATION_MEMBER, json.dumps({'licenses': [{'id': 4}], 'images': images,
                                                    'annotations': annotations,
                                                    'categories': [{'id': 17, 'name': 'cat'},
                                                                   {'id': 18, 'name': 'dog'}]}))
    previous = [{'image_id': 1700}]
    selected = cases.select_inventory(archive, previous, count=4, authenticated=False)
    for category, rows in ((17, selected[:4]), (18, selected[4:])):
        candidates = [image['id'] for image in images if image['id'] // 100 == category and image['id'] != 1700]
        expected = sorted(candidates, key=lambda ident: (
            hashlib.sha256(f'vision-coco-spatial-dose-v6:{category}:{ident}'.encode('ascii')).hexdigest(), ident))[:4]
        assert [row['image_id'] for row in rows] == expected
        assert [row['role'] for row in rows] == ['donor', 'target', 'target', 'target']


def test_v6_preparer_checks_package_origin_before_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / 'shadow' / 'looking_glass'
    fake.mkdir(parents=True)
    marker = tmp_path / 'executed'
    (fake / '__init__.py').write_text(f'from pathlib import Path; Path({str(marker)!r}).touch()\n')
    prefix = [sys.executable, '-I', '-S', '-B']
    assert subprocess.run(prefix + ['-c', 'import sys;sys.path.append(sys.argv[1]);import looking_glass',
                                    str(fake.parent)], capture_output=True).returncode == 0
    assert marker.exists()
    marker.unlink()
    cache = tmp_path / 'empty-cache'
    cache.mkdir()
    bootstrap = ('import runpy,sys;sys.path.extend([sys.argv[1],sys.argv[2]]);'
                 'sys.argv=[sys.argv[3],"--help"];runpy.run_path(sys.argv[0],run_name="__main__")')
    result = subprocess.run(prefix + ['-X', f'pycache_prefix={cache}', '-c', bootstrap,
                                      str(fake.parent), str(root),
                                      str(root / 'tools/prepare_vision_coco_spatial_dose.py')],
                            capture_output=True)
    assert result.returncode != 0 and not marker.exists()
    assert b'project package is shadowed' in result.stderr
