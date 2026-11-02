import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from looking_glass import vision_coco_recognized_cases as cases


def _annotation(image, category, ident, box=(40, 40, 144, 144)):
    x, y, w, h = box
    return {'id': ident, 'image_id': image, 'category_id': category, 'iscrowd': 0,
            'bbox': list(box), 'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]]}


def _archive(path, *, offset=0):
    images, annotations = [], []
    for category in (17, 18):
        for n in range(8):
            ident = offset + category * 100 + n
            images.append({'id': ident, 'file_name': f'{ident:012d}.jpg',
                           'width': 224, 'height': 224, 'license': 4})
            annotations.append(_annotation(ident, category, ident))
    data = {'licenses': [{'id': 4, 'name': 'test'}], 'images': images,
            'annotations': annotations, 'categories': [{'id': 17, 'name': 'cat'}, {'id': 18, 'name': 'dog'}]}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(cases.ANNOTATION_MEMBER, json.dumps(data))
    return data


def test_stream_parser_and_deterministic_rank(tmp_path):
    path = tmp_path / 'data.zip'
    _archive(path)
    rows = cases.select_inventory(path, [], count=4, authenticated=False)
    assert len(rows) == 8
    for category, group in ((17, rows[:4]), (18, rows[4:])):
        assert [r['image_id'] for r in group] == sorted(
            (r['image_id'] for r in group), key=lambda i: cases.rank(category, i))
        assert [r['role'] for r in group] == ['donor', 'target', 'target', 'target']
        assert all(r['sites'] and min(s['mask_pixels'] for s in r['sites']) >= 512 for r in group)
    assert cases.select_inventory(path, [], count=4, authenticated=False) == rows


def test_previous_exposure_and_insufficient_inventory(tmp_path):
    path = tmp_path / 'data.zip'
    data = _archive(path)
    excluded = {r['id'] for r in data['images'] if r['id'] // 100 == 18}
    with pytest.raises(ValueError, match='insufficient'):
        cases.select_inventory(path, excluded, count=1, authenticated=False)


def test_no_undocumented_image_id_exclusion(tmp_path):
    path = tmp_path / 'data.zip'
    data = _archive(path)
    for image_id, category in ((39769, 17), (98765, 18)):
        data['images'].append({'id': image_id, 'file_name': f'{image_id:012d}.jpg',
                               'width': 224, 'height': 224, 'license': 4})
        data['annotations'].append(_annotation(image_id, category, image_id))
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(cases.ANNOTATION_MEMBER, json.dumps(data))
    rows = cases.select_inventory(path, [], count=9, authenticated=False)
    assert 39769 in {row['image_id'] for row in rows}


def test_largest_instance_is_not_replaced_and_mixed_species_excluded(tmp_path):
    image = {'id': 4, 'width': 224, 'height': 224, 'file_name': '000000000004.jpg', 'license': 4}
    larger_bad = _annotation(4, 17, 1, (0, 0, 220, 220))
    smaller_good = _annotation(4, 17, 2)
    assert cases._candidate(image, [larger_bad, smaller_good]) is None
    assert cases._candidate(image, [smaller_good]) is not None
    assert cases._candidate(image, [smaller_good, _annotation(4, 18, 3)]) is None


def test_mutated_open_archive_fails_after_selection(tmp_path, monkeypatch):
    path = tmp_path / 'data.zip'
    _archive(path)
    original = cases._select_inventory
    def alter(archive, excluded, count):
        result = original(archive, excluded, count)
        with path.open('ab') as file:
            file.write(b'x')
        return result
    monkeypatch.setattr(cases, '_select_inventory', alter)
    with pytest.raises(ValueError, match='changed'):
        cases.select_inventory(path, [], count=1, authenticated=False)


def test_stream_rejects_bad_and_oversized_items():
    with pytest.raises(ValueError, match='truncated'):
        list(cases.stream_array(io.BytesIO(b'{"images": [{"id": 1'), 'images'))
    with pytest.raises(ValueError, match='object'):
        list(cases.stream_array(io.BytesIO(b'{"images": [1]}'), 'images'))


def test_actual_previous_exposure_pins():
    root = Path(__file__).resolve().parents[1] / 'experiments'
    rows = cases.previous_images((root / 'vision_coco_spatial_v1.json').read_bytes(),
                                 (root / 'vision_coco_same_site_v3.json').read_bytes())
    assert len(rows) == len({row['image_id'] for row in rows}) == 43


def test_jpeg_binding_rejects_same_length_replacement(tmp_path):
    path = tmp_path / 'image.jpg'
    raw = b'fabricated jpeg bytes'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert cases.read_expected_image(path, len(raw), digest) == raw
    path.write_bytes(b'X' + raw[1:])
    with pytest.raises(ValueError, match='JPEG bytes'):
        cases.read_expected_image(path, len(raw), digest)


def test_preparation_rejects_shadowed_package_before_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    package = tmp_path / 'shadow' / 'looking_glass'
    package.mkdir(parents=True)
    marker = tmp_path / 'executed'
    (package / '__init__.py').write_text(f'from pathlib import Path; Path({str(marker)!r}).touch()\n')
    cache = tmp_path / 'empty-cache'
    cache.mkdir()
    prefix = [sys.executable, '-I', '-S', '-B', '-X', f'pycache_prefix={cache}']
    positive = subprocess.run(prefix + ['-c', 'import sys; sys.path.append(sys.argv[1]); import looking_glass',
                                        str(package.parent)], capture_output=True)
    assert positive.returncode == 0 and marker.exists()
    marker.unlink()
    bootstrap = ("import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2]]); "
                 "sys.argv=[sys.argv[3],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')")
    result = subprocess.run(prefix + ['-c', bootstrap, str(package.parent), str(root),
                                      str(root / 'tools/prepare_vision_coco_recognized.py')], capture_output=True)
    assert result.returncode != 0 and not marker.exists()
    assert b'project package is shadowed' in result.stderr
