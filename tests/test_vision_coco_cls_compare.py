"""Fabricated CPU checks for common-dose CLS versus spatial interventions."""
from types import SimpleNamespace
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

import pytest
import torch


class _Layer(torch.nn.Module):
    def forward(self, value):
        return value + .01 * value.mean(dim=1, keepdim=True)


class _FakeVit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.nn.Module()
        self.vit.encoder = torch.nn.Module()
        self.vit.encoder.layer = torch.nn.ModuleList([_Layer() for _ in range(12)])

    def forward(self, pixel_values):
        value = pixel_values
        for layer in self.vit.encoder.layer:
            value = layer(value)
        logits = torch.zeros((1, 1000), dtype=torch.float32)
        logits[:, 151:269] = value[:, 0, :1]
        logits[:, 281:286] = 3.2 - value[:, 0, :1]
        return SimpleNamespace(logits=logits)


def test_full_distribution_tv_and_directional_kl_from_saved_logits():
    from looking_glass.vision_coco_cls_study import distribution_effect

    clean = torch.zeros((1, 1000), dtype=torch.float32)
    patched = clean.clone()
    patched[0, 17] = 1.
    metrics = distribution_effect(clean, patched)
    expected_tv = math.e / (999 + math.e) - .001
    expected_kl = math.log((999 + math.e) / 1000) - .001
    assert metrics['tv'] == pytest.approx(expected_tv, abs=1e-14)
    assert metrics['kl_clean_to_patched'] == pytest.approx(expected_kl, abs=1e-14)
    assert metrics['top1_changed'] is True
    assert distribution_effect(clean, clean) == {'tv': 0., 'kl_clean_to_patched': 0., 'top1_changed': False}


def test_complete_pair_retains_both_sites_controls_and_six_matched_landings(tmp_path):
    from looking_glass.vision_coco_cls_study import aggregate_pairs, run_comparison
    from looking_glass.vision_coco_study import Evidence, forward

    assert torch.set_flush_denormal(False)
    pixels = {1: torch.full((1, 197, 192), -3.), 2: torch.full((1, 197, 192), 3.),
              3: torch.full((1, 197, 192), -1.), 4: torch.full((1, 197, 192), 1.)}
    pair = {'pair_id': 'pair-00', 'split': 'heldout', 'cat': {'image_id': 1}, 'dog': {'image_id': 2},
            'cat_donor': {'image_id': 3}, 'dog_donor': {'image_id': 4}, 'box_tokens': [1, 2, 15, 16]}
    evidence = Evidence(tmp_path / 'pair', 40)
    checked, labels = [], []
    def observed_forward(*args, **kwargs):
        labels.append(args[3])
        return forward(*args, **kwargs)
    rows = run_comparison(_FakeVit().eval(), [pair], pixels, evidence,
                          check_clean=lambda image_id, logits: checked.append(image_id),
                          forward_fn=observed_forward)
    assert checked == [1, 2, 3, 4]
    assert len(evidence.records) == evidence.submissions == 34
    assert len(labels) == 34 and labels[0] == 'pair-00:cat:initial'
    assert aggregate_pairs(rows)['status'] == 'inconclusive'
    for side in ('cat', 'dog'):
        data = rows[0]['sides'][side]
        assert data['dose']['status'] == 'matchable'
        assert {'null_cls', 'self_cls', 'null_spatial', 'self_spatial',
                'full_self', 'full_swap', 'final_spatial', 'repeat', 'post_clean'} <= set(data['conditions'])
        for site in ('cls', 'spatial'):
            for kind in ('opposite', 'same_species', 'random'):
                condition = data['conditions'][f'{site}_{kind}']
                assert condition['actual_dose'] == pytest.approx(data['dose']['q'], rel=1e-5)
                assert condition['realized_over_receiver_norm'] <= .1
                assert 'kl_clean_to_patched' in condition and 'tv' in condition
    owner = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/run_vision_coco_cls.py'))
    terminal = {'pairs': rows, 'submissions': 34, 'saved_forwards': 34, 'aggregate': aggregate_pairs(rows)}
    evidence.json('started.json', {'synthetic': True})
    evidence.json('terminal.json', terminal)
    owner['verify_intervention_schedule'](evidence.root, terminal, [pair])
    (evidence.root / 'forward-034.json').unlink()
    with pytest.raises(ValueError, match='inventory'):
        owner['verify_intervention_schedule'](evidence.root, terminal, [pair])


def test_later_baseline_contradiction_stops_before_any_patch(tmp_path):
    from looking_glass.vision_coco_cls_study import run_comparison
    from looking_glass.vision_coco_study import Evidence

    assert torch.set_flush_denormal(False)
    pixels = {i: torch.full((1, 197, 192), value)
              for i, value in ((1, -3.), (2, 3.), (3, -1.), (4, 1.), (5, 3.), (6, 3.))}
    pairs = [dict(pair_id='pair-00', split='heldout', cat={'image_id': 1}, dog={'image_id': 2},
                  cat_donor={'image_id': 3}, dog_donor={'image_id': 4}, box_tokens=[1, 2, 15, 16]),
             dict(pair_id='pair-01', split='heldout', cat={'image_id': 5}, dog={'image_id': 6},
                  cat_donor={'image_id': 3}, dog_donor={'image_id': 4}, box_tokens=[1, 2, 15, 16])]
    evidence = Evidence(tmp_path / 'wrong-baseline', 64)
    with pytest.raises(ValueError, match='baseline'):
        run_comparison(_FakeVit().eval(), pairs, pixels, evidence)
    assert evidence.submissions == len(evidence.records) == 6
    assert all(row['label'].endswith(':initial') for row in evidence.records)


def test_owner_preflight_requires_exact_zero_counts_and_complete_runtime(tmp_path):
    from looking_glass.vision_coco_recognized_handoff import MODEL_STATE_SHA256

    owner = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/run_vision_coco_cls.py'))
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
    terminal['submissions'] = False
    with pytest.raises(ValueError, match='preflight'):
        owner['verify_terminal'](tmp_path, terminal, source, 'preflight', {}, None)
    terminal['submissions'] = 0
    del runtime['mkldnn_enabled']
    del terminal['runtime']['mkldnn_enabled']
    (tmp_path / 'started.json').write_text(json.dumps(started))
    with pytest.raises(ValueError, match='runtime'):
        owner['verify_terminal'](tmp_path, terminal, source, 'preflight', {}, None)


def test_v5_closed_screen_reuses_selection_but_rejects_changed_public_result(tmp_path):
    from looking_glass.vision_coco_recognized_handoff import MODEL_STATE_SHA256, canonical
    from looking_glass.vision_coco_recognized_runner import run_clean_screen
    from looking_glass.vision_coco_study import Evidence

    root = Path(__file__).resolve().parents[1]
    owner = runpy.run_path(str(root / 'tools/run_vision_coco_cls.py'))
    images = runpy.run_path(str(root / 'tests/test_vision_coco_recognized_screen.py'))['_full_manifest_images']()
    manifest = {'schema': 'vision-coco-cls-compare-cases-v5', 'images': images}
    class FakeScreen(torch.nn.Module):
        def forward(self, pixel_values):
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[0, 281 if pixel_values.item() == 17 else 151] = 8.
            return SimpleNamespace(logits=logits)
    pixels = {row['image_id']: torch.tensor([[float(row['category_id'])]]) for row in images}
    runtime = {'device': 'cpu', 'cuda_visible_devices': '', 'cuda_initialized': False,
               'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
               'numpy': '2.4.6', 'pillow': '11.3.0', 'threads': 1, 'interop_threads': 1,
               'deterministic_algorithms': True, 'float32_matmul_precision': 'highest',
               'mkldnn_enabled': True, 'default_dtype': 'torch.float32', 'processor': 'ViTImageProcessor',
               'denormals_preserved': True, 'mkl_cbwr': 'COMPATIBLE', 'binary64_rounding': 'nearest-even'}
    evidence = Evidence(tmp_path / 'screen', 64)
    evidence.json('started.json', {'schema': 'vision-recognized-v4-start', 'mode': 'screen',
                                  'manifest_sha256': 'm', 'source': {'revision': 'fake'},
                                  'runtime': runtime, 'model_state_sha256': MODEL_STATE_SHA256})
    assert torch.set_flush_denormal(False)
    observed, pairs, summary = run_clean_screen(FakeScreen().eval(), images, pixels, evidence)
    result = {'schema': 'vision-recognized-v4-screen-result', 'manifest_sha256': 'm',
              'screen': observed, 'pairs': pairs, 'summary': summary}
    digest = evidence.json('screen-result.json', result)
    evidence.json('terminal.json', {'schema': 'vision-recognized-v4-terminal', 'status': 'complete',
                  'mode': 'screen', 'manifest_sha256': 'm', 'source_revision': 'fake',
                  'submissions': 64, 'saved_forwards': 64, 'runtime': runtime,
                  'model_state_sha256_before_after': MODEL_STATE_SHA256, 'screen_result_sha256': digest})
    verified = owner['verify_v5_screen'](manifest, evidence.root, 'm', expected_result=canonical(result))
    assert len(verified['raw_by_id']) == 64 and len(verified['result']['pairs']) == 4
    with pytest.raises(ValueError, match='result'):
        owner['verify_v5_screen'](manifest, evidence.root, 'm', expected_result=b'{}')


def test_common_dose_is_capped_across_both_sites_and_all_donors():
    from looking_glass.vision_coco_cls_study import prepare_dose

    assert torch.set_flush_denormal(False)
    own = torch.ones((1, 197, 192), dtype=torch.float32)
    opposite, same = 3 * own, 4 * own
    spatial = [1, 2, 15, 16]
    same[0, spatial] = own[0, spatial] + .01
    dose, values = prepare_dose(own, opposite, same, spatial, 'pair-00', 'cat')
    assert dose['status'] == 'matchable'
    assert dose['q'] == pytest.approx(float(same[0, 1, 0] - 1) * math.sqrt(768), abs=1e-12)
    assert set(values) == {'cls', 'spatial'}
    for site, tokens in (('cls', [0]), ('spatial', spatial)):
        for name in ('opposite', 'same_species', 'random'):
            actual = torch.linalg.vector_norm((values[site][name] - own[0, tokens]).double()).item()
            assert actual == pytest.approx(dose['q'], rel=1e-5)
            assert actual <= .10 * torch.linalg.vector_norm(own[0, tokens].double()).item()
    assert torch.equal(values['spatial']['same_species'], same[0, spatial])
    same[0, 0] = own[0, 0]
    dose, _ = prepare_dose(own, opposite, same, spatial, 'pair-00', 'cat')
    assert dose['status'] == 'dose-ineligible'
    assert 'cls-same-distance' in dose['reasons']


def test_adjusted_site_advantage_is_not_confused_with_raw_difference():
    from looking_glass.vision_coco_cls_study import classify_pair

    # CLS's raw +0.10 is smaller than spatial's +0.20, but the predeclared
    # spatial same-species control is also +0.20, giving a larger CLS advantage.
    side = {'cls': {'opposite': .10, 'same_species': 0., 'random': 0.},
            'spatial': {'opposite': .20, 'same_species': .20, 'random': 0.}}
    dose = {'cat': 'matchable', 'dog': 'matchable'}
    assert classify_pair({'eligible': True}, dose, {'cat': side, 'dog': side}) == 'positive'
    weak = {'cls': {'opposite': .10, 'same_species': -.09, 'random': 0.},
            'spatial': side['spatial']}
    assert classify_pair({'eligible': True}, dose, {'cat': weak, 'dog': side}) == 'negative'
    assert classify_pair({'eligible': True}, {'cat': 'dose-ineligible', 'dog': 'matchable'},
                         None) == 'dose-ineligible'


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
