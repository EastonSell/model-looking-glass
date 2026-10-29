"""Fabricated CPU checks for common-dose CLS versus spatial interventions."""
from types import SimpleNamespace
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch


def test_preparer_rejects_shadowed_project_before_import(tmp_path):
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
    bootstrap = ("import runpy,sys;sys.path.extend([sys.argv[1],sys.argv[2]]);"
                 "sys.argv=[sys.argv[3],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')")
    result = subprocess.run(prefix + ['-X', f'pycache_prefix={cache}', '-c', bootstrap,
                                      str(fake.parent), str(root), str(root / 'tools/prepare_vision_coco_cls.py')],
                            capture_output=True)
    assert result.returncode != 0 and not marker.exists()
    assert b'project package is shadowed' in result.stderr


def test_fresh_inventory_excludes_all_public_prior_images_and_pins_v4_bytes():
    from looking_glass.vision_coco_cls_cases import previous_images, rank

    experiments = Path(__file__).resolve().parents[1] / 'experiments'
    v1 = (experiments / 'vision_coco_spatial_v1.json').read_bytes()
    v3 = (experiments / 'vision_coco_same_site_v3.json').read_bytes()
    v4 = (experiments / 'vision_coco_recognized_v4.json').read_bytes()
    rows = previous_images(v1, v3, v4)
    prior_ids = {row['image_id'] for row in rows}
    assert len(prior_ids) == len(rows) == 107
    assert {row['image_id'] for row in json.loads(v4)['images']} <= prior_ids
    assert rank(17, 123) == ('a433e64d5c3b5930de4a8d95b3c779a0999e5f6fc44808c12894c5c35a93196e', 123)
    with pytest.raises(ValueError, match='v4'):
        previous_images(v1, v3, v4 + b' ')


def test_annotation_ranking_uses_supplied_order_and_preserves_exclusions(tmp_path):
    from looking_glass import vision_coco_recognized_cases as cases

    images, annotations = [], []
    for category in (17, 18):
        for n in range(8):
            ident = category * 100 + n
            images.append({'id': ident, 'file_name': f'{ident:012d}.jpg',
                           'width': 224, 'height': 224, 'license': 4})
            annotations.append({'id': ident, 'image_id': ident, 'category_id': category,
                                'iscrowd': 0, 'bbox': [40, 40, 144, 144],
                                'segmentation': [[40, 40, 184, 40, 184, 184, 40, 184]]})
    data = {'licenses': [{'id': 4}], 'images': images, 'annotations': annotations,
            'categories': [{'id': 17, 'name': 'cat'}, {'id': 18, 'name': 'dog'}]}
    archive = tmp_path / 'annotations.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr(cases.ANNOTATION_MEMBER, json.dumps(data))

    def reverse_rank(category, image_id):
        return (f'{10000-image_id:064x}', image_id)

    rows = cases.select_inventory(archive, [1707], count=3, authenticated=False, ranker=reverse_rank)
    assert [r['image_id'] for r in rows] == [1706, 1705, 1704, 1807, 1806, 1805]
    assert [r['role'] for r in rows] == ['donor', 'target', 'target'] * 2
    assert rows[0]['selection_rank_sha256'] == f'{10000-1706:064x}'


def test_cls_patch_changes_only_cls_state_and_can_affect_logits(tmp_path):
    from looking_glass.vision_coco_study import Evidence, check_landing, forward

    class Layer(torch.nn.Module):
        def forward(self, value):
            return value + .01 * value.mean(dim=1, keepdim=True)

    class FakeVit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = torch.nn.Module()
            self.vit.encoder = torch.nn.Module()
            self.vit.encoder.layer = torch.nn.ModuleList([Layer() for _ in range(12)])

        def forward(self, pixel_values):
            value = pixel_values
            for layer in self.vit.encoder.layer:
                value = layer(value)
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[:, 151:269] = value[:, 0, :1]
            return SimpleNamespace(logits=logits)

    assert torch.set_flush_denormal(False)
    torch.manual_seed(31)
    pixels = torch.randn(1, 197, 192)
    evidence = Evidence(tmp_path / 'evidence', 2)
    model = FakeVit().eval()
    clean, captured, _ = forward(model, pixels, evidence, 'clean')
    state = captured['early']
    replacement = state[0, [0]].clone()
    replacement[0, 0] += 1
    changed, witness, _ = forward(model, pixels, evidence, 'cls-patch',
                                  path='vit.encoder.layer.2', tokens=[0],
                                  replacement=replacement, action='patch')
    check_landing(witness, state, [0], replacement)
    assert changed[0, 151] != clean[0, 151]
    assert torch.equal(witness['after_all'][0, 1:], state[0, 1:])
