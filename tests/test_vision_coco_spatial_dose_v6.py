import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch

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


def test_common_three_dose_ladder_and_limited_or_zero_donors():
    from looking_glass.vision_coco_spatial_dose_study import prepare_ladder

    assert torch.set_flush_denormal(False)
    tokens = [1, 2, 15, 16]
    own = torch.ones((1, 197, 192), dtype=torch.float32)
    opposite, same = 3 * own, 2 * own
    plan, vectors = prepare_ladder(own, opposite, same, tokens, 'pair-00', 'cat')
    assert plan['status'] == 'matchable'
    assert plan['levels']['high'] == pytest.approx(.396 * plan['receiver_norm'])
    assert plan['levels']['mid'] == plan['levels']['high'] / 2
    assert plan['levels']['low'] == plan['levels']['high'] / 4
    signs = None
    for level in ('low', 'mid', 'high'):
        for kind in ('opposite', 'same_species', 'random'):
            actual = torch.linalg.vector_norm((vectors[level][kind] - own[0, tokens]).double()).item()
            assert actual == pytest.approx(plan['levels'][level], rel=1e-5)
            assert actual <= .4 * plan['receiver_norm']
        current = torch.sign(vectors[level]['random'] - own[0, tokens])
        if signs is not None:
            assert torch.equal(current, signs)
        signs = current
    same[0, tokens] = 1.2
    limited, kept = prepare_ladder(own, opposite, same, tokens, 'pair-00', 'cat')
    assert limited['status'] == 'limited-dose' and set(kept) == {'low', 'mid', 'high'}
    same[0, tokens] = 1.
    zero, kept = prepare_ladder(own, opposite, same, tokens, 'pair-00', 'cat')
    assert zero['status'] == 'dose-ineligible' and kept == {}
    same[0, tokens[0], 0] = float('nan')
    nonfinite, kept = prepare_ladder(own, opposite, same, tokens, 'pair-00', 'cat')
    assert nonfinite['status'] == 'dose-ineligible' and nonfinite['reasons'] == ['nonfinite-activation']
    assert kept == {}


def test_control_adjusted_rule_can_increase_while_raw_shift_decreases():
    from looking_glass.vision_coco_spatial_dose_study import side_metrics, aggregate_pairs

    moves = {'low': {'opposite': .20, 'same_species': .18, 'random': 0.},
             'mid': {'opposite': .11, 'same_species': .01, 'random': 0.},
             'high': {'opposite': .10, 'same_species': 0., 'random': 0.}}
    result = side_metrics(moves)
    assert result['side_pass'] is True
    assert result['raw_high_minus_low'] < 0 < result['adjusted_high_minus_low']
    moves['high']['same_species'] = .09
    assert side_metrics(moves)['side_pass'] is False
    assert aggregate_pairs([{'pair_id': 'pair-00', 'status': 'positive'}])['status'] == 'inconclusive'
    assert aggregate_pairs([{'pair_id': 'pair-00', 'status': 'positive'},
                            {'pair_id': 'pair-01', 'status': 'negative'}])['status'] == 'negative'
    assert aggregate_pairs([{'pair_id': 'pair-00', 'status': 'positive'},
                            {'pair_id': 'pair-01', 'status': 'positive'}])['status'] == 'positive'


def test_criterion_boundaries_are_inclusive_without_tolerance():
    from looking_glass.vision_coco_spatial_dose_study import side_metrics

    moves = {'low': {'opposite': 0., 'same_species': 0., 'random': 0.},
             'mid': {'opposite': .03, 'same_species': 0., 'random': 0.},
             'high': {'opposite': .1, 'same_species': .05, 'random': 0.}}
    assert side_metrics(moves)['side_pass'] is True
    moves['high']['opposite'] = math.nextafter(.1, -math.inf)
    assert side_metrics(moves)['side_pass'] is False


def test_real_hooks_retain_nine_matched_actions_and_identity_controls(tmp_path):
    from looking_glass.vision_coco_spatial_dose_study import run_dose_response
    from looking_glass.vision_coco_study import Evidence

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
            logits[:, 281:286] = 3.2 - value[:, 0, :1]
            return SimpleNamespace(logits=logits)

    assert torch.set_flush_denormal(False)
    pixels = {1: torch.full((1, 197, 192), -3.), 2: torch.full((1, 197, 192), 3.),
              3: torch.full((1, 197, 192), -1.), 4: torch.full((1, 197, 192), 1.)}
    pair = {'pair_id': 'pair-00', 'split': 'heldout', 'cat': {'image_id': 1}, 'dog': {'image_id': 2},
            'cat_donor': {'image_id': 3}, 'dog_donor': {'image_id': 4}, 'box_tokens': [1, 2, 15, 16]}
    evidence = Evidence(tmp_path / 'pair', 40)
    checked = []
    rows = run_dose_response(FakeVit().eval(), [pair], pixels, evidence,
                             check_clean=lambda ident, logits: checked.append(ident))
    assert checked == [1, 2, 3, 4]
    assert evidence.submissions == len(evidence.records) == 36
    for side in ('cat', 'dog'):
        value = rows[0]['sides'][side]
        assert value['dose']['status'] == 'matchable'
        assert {'repeat', 'null', 'self', 'full_self', 'full_swap', 'final_spatial', 'post_clean'} <= set(value['conditions'])
        for level in ('low', 'mid', 'high'):
            for arm in ('opposite', 'same_species', 'random'):
                action = value['conditions'][f'{level}_{arm}']
                assert action['actual_dose'] == pytest.approx(value['dose']['levels'][level], rel=1e-5)
                assert action['realized_over_receiver_norm'] <= .4
    owner = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/run_vision_coco_spatial_dose.py'))
    from looking_glass.vision_coco_spatial_dose_study import aggregate_pairs
    terminal = {'pairs': rows, 'submissions': 36, 'saved_forwards': 36, 'aggregate': aggregate_pairs(rows)}
    evidence.json('started.json', {'synthetic': True})
    evidence.json('terminal.json', terminal)
    owner['verify_intervention_schedule'](evidence.root, terminal, [pair])
    (evidence.root / 'forward-036.json').unlink()
    with pytest.raises(ValueError, match='inventory'):
        owner['verify_intervention_schedule'](evidence.root, terminal, [pair])
    pixels[3] = torch.full((1, 197, 192), -2.8)
    limited_evidence = Evidence(tmp_path / 'limited', 40)
    limited = run_dose_response(FakeVit().eval(), [pair], pixels, limited_evidence)
    assert limited[0]['status'] == 'limited-dose'
    assert limited[0]['sides']['cat']['dose']['status'] == 'limited-dose'
    assert all(f'{level}_{arm}' in limited[0]['sides']['cat']['conditions']
               for level in ('low', 'mid', 'high') for arm in ('opposite', 'same_species', 'random'))
    assert limited_evidence.submissions == 36


def test_owner_preflight_requires_explicit_zero_forwards(tmp_path):
    from looking_glass.vision_coco_recognized_handoff import MODEL_STATE_SHA256

    owner = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/run_vision_coco_spatial_dose.py'))
    runtime = {'device': 'cpu', 'cuda_visible_devices': '', 'cuda_initialized': False,
               'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
               'numpy': '2.4.6', 'pillow': '11.3.0', 'threads': 1, 'interop_threads': 1,
               'deterministic_algorithms': True, 'float32_matmul_precision': 'highest',
               'mkldnn_enabled': True, 'default_dtype': 'torch.float32', 'processor': 'ViTImageProcessor',
               'denormals_preserved': True, 'mkl_cbwr': 'COMPATIBLE', 'binary64_rounding': 'nearest-even'}
    source = {'revision': 'f' * 40, 'sources': {}}
    started = {'schema': 'vision-recognized-v4-start', 'mode': 'preflight', 'source': source,
               'manifest_sha256': owner['MANIFEST_SHA'], 'model_state_sha256': MODEL_STATE_SHA256,
               'runtime': runtime}
    terminal = {'schema': 'vision-recognized-v4-terminal', 'status': 'complete', 'mode': 'preflight',
                'source_revision': source['revision'], 'manifest_sha256': owner['MANIFEST_SHA'],
                'model_state_sha256_before_after': MODEL_STATE_SHA256, 'runtime': dict(runtime),
                'submissions': 0, 'saved_forwards': 0}
    (tmp_path / 'started.json').write_text(json.dumps(started))
    (tmp_path / 'terminal.json').write_text(json.dumps(terminal))
    owner['verify_terminal'](tmp_path, terminal, source, 'preflight', {}, None)
    terminal['saved_forwards'] = False
    with pytest.raises(ValueError, match='preflight'):
        owner['verify_terminal'](tmp_path, terminal, source, 'preflight', {}, None)
