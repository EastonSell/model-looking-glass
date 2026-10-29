"""Model-free identity and ranking regressions for CLS/occlusion follow-up."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch


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


def test_seven_replacements_share_absolute_dose_and_keep_distinct_random_controls():
    from looking_glass.vision_coco_cls_occlusion_study import prepare_transfers

    clean = torch.ones((1, 197, 192), dtype=torch.float32)
    animal, background = [31, 32, 45, 46], [143, 144, 157, 158]
    masked_a, masked_b = clean.clone(), clean.clone()
    masked_a[0, 0] += 2
    masked_b[0, 0] -= 3
    masked_a[0, animal] += 4
    masked_b[0, background] -= 5
    plan, replacements = prepare_transfers(clean, masked_a, masked_b, animal, background, 42)
    assert plan['status'] == 'matchable'
    assert plan['q'] == pytest.approx(.196 * math.sqrt(192), rel=1e-7)
    assert set(replacements) == {'cls_animal', 'cls_background', 'spatial_animal',
                                 'spatial_background', 'random_cls', 'random_animal', 'random_background'}
    for name, replacement in replacements.items():
        tokens = [0] if 'cls' in name else animal if 'animal' in name else background
        actual = torch.linalg.vector_norm((replacement - clean[0, tokens]).double()).item()
        assert actual == pytest.approx(plan['q'], rel=1e-6)
        receiver = torch.linalg.vector_norm(clean[0, tokens].double()).item()
        assert actual <= .2 * receiver
    assert torch.equal(replacements['random_animal'], replacements['random_background'])


def test_inclusive_closeness_decision_and_aggregate_denominator():
    from looking_glass.vision_coco_cls_occlusion_study import aggregate_images, image_decision

    assert image_decision(.05, .05, .025, .025, 0., 0.)
    assert not image_decision(math.nextafter(.05, -math.inf), .05, .025, .025, 0., 0.)
    assert not image_decision(.05, .05, math.nextafter(.025, math.inf), .025, 0., 0.)
    assert not image_decision(.05, .05, .025, math.nextafter(.025, math.inf), 0., 0.)
    assert aggregate_images([(17, True), (17, True), (18, True)])['status'] == 'inconclusive'
    assert aggregate_images([(17, True), (17, False), (18, True), (18, True)])['status'] == 'positive'
    assert aggregate_images([(17, True), (17, False), (18, True), (18, False)])['status'] == 'negative'


def test_zero_and_small_donor_distances_are_retained_as_dose_exclusions():
    from looking_glass.vision_coco_cls_occlusion_study import prepare_transfers

    clean = torch.ones((1, 197, 192), dtype=torch.float32)
    animal, background = [31, 32, 45, 46], [143, 144, 157, 158]
    donor_a, donor_b = clean.clone(), clean.clone()
    donor_a[0, [0] + animal] += .001
    donor_b[0, [0] + background] -= 1.
    plan, _ = prepare_transfers(clean, donor_a, donor_b, animal, background, 42)
    assert plan['status'] == 'limited-dose'
    donor_a[0, 0] = clean[0, 0]
    plan, replacements = prepare_transfers(clean, donor_a, donor_b, animal, background, 42)
    assert plan['status'] == 'dose-ineligible'
    assert 'cls_animal-donor-distance' in plan['reasons']
    assert replacements == {}


def test_response_screen_retains_all_failures_and_uses_fixed_rank_order():
    from looking_glass.vision_coco_cls_occlusion_study import select_response_screen

    images = [{'image_id': category * 100 + rank, 'category_id': category, 'rank': rank,
               'occlusion_geometry': None if rank == 1 else {'fixed': True}}
              for category in (17, 18) for rank in range(6)]
    clean = [{'image_id': row['image_id'], 'recognized': row['rank'] != 2} for row in images]
    masked = {row['image_id']: {'animal_drop': .06 if row['rank'] != 3 else .049,
                                'background_drop': 0.}
              for row in images if row['rank'] not in (1, 2)}
    selected, counts, rows = select_response_screen(images, clean, masked)
    assert [row['image_id'] for row in selected] == [1700, 1704, 1705, 1800, 1804, 1805]
    assert counts['cat'] == {'candidates': 6, 'geometry_qualified': 5, 'recognized': 5,
                             'recognized_geometry': 4, 'response_eligible': 3, 'selected': 3}
    assert rows[1]['reasons'] == ['geometry']
    assert rows[2]['reasons'] == ['recognition']
    assert rows[3]['reasons'] == ['occlusion-contrast']
    with pytest.raises(ValueError, match='mask inventory'):
        select_response_screen(images, clean, {key: value for key, value in masked.items() if key != 1700})


def test_response_mask_screen_saves_same_image_pixels_and_both_forwards(tmp_path):
    from looking_glass.vision_coco_cls_occlusion_study import capture_initials, run_case, run_response_masks
    from looking_glass.vision_coco_occlusion_cases import choose_geometry
    from looking_glass.vision_coco_study import Evidence, _bits

    class Block(torch.nn.Module):
        def forward(self, value):
            return value + value.mean(dim=1, keepdim=True) * .01

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.ModuleList([Block() for _ in range(12)])

        def forward(self, value):
            for block in self.layer:
                value = block(value)
            return value

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = torch.nn.Module()
            self.vit.encoder = Encoder()

        def forward(self, pixel_values):
            spatial = torch.nn.functional.avg_pool2d(pixel_values, 16).mean(1).reshape(1, 196, 1)
            state = torch.cat((spatial.mean(1, keepdim=True), spatial), 1).expand(-1, -1, 192).clone()
            state = self.vit.encoder(state)
            weights = torch.linspace(-1, 1, 1000).reshape(1, -1)
            return SimpleNamespace(logits=state[:, :1, 0] * weights)

    image = {'image_id': 42, 'category_id': 17, 'rank': 0, 'width': 224, 'height': 224,
             'bbox_resized_xyxy': [0., 0., 64., 64.],
             'sites': [{'row_col': [2, 2], 'mask_pixels': 768, 'token_mask_pixels': [256, 256, 256, 0],
                        'squared_center_radius': 32, 'token_ids': [31, 32, 45, 46]}], 'other_boxes': []}
    image['occlusion_geometry'] = choose_geometry(image)
    pixels = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    model = Fake().eval()
    with torch.no_grad():
        logits = _bits(model(pixel_values=pixels).logits)
    clean = [{'image_id': 42, 'recognized': True, 'logit_sha256': hashlib.sha256(logits).hexdigest()}]
    evidence = Evidence(tmp_path / 'screen', 2)
    rows = run_response_masks(model, [image], clean, {42: pixels}, {42: logits}, evidence)
    assert evidence.submissions == len(evidence.records) == 2
    assert set(rows[42]['input_sha256']) == {'clean', 'animal', 'background'}
    assert (tmp_path / 'screen/screen-input-42-clean.bin').read_bytes() == _bits(pixels)
    assert set(rows[42]) >= {'animal_drop', 'background_drop', 'input_sha256', 'logit_sha256'}
    inputs = {name: (evidence.root / f'screen-input-42-{name}.bin').read_bytes()
              for name in ('clean', 'animal', 'background')}
    expected = {'clean': logits, 'animal': (evidence.root / 'forward-001.bin').read_bytes()[:4000],
                'background': (evidence.root / 'forward-002.bin').read_bytes()[:4000]}
    evaluation = Evidence(tmp_path / 'evaluation', 23)
    captured = capture_initials(model, [image], {42: pixels}, {42: inputs}, {42: expected}, evaluation)
    assert evaluation.submissions == 3
    case = run_case(model, image, captured[42], evaluation)
    assert evaluation.submissions == len(evaluation.records) == 23
    assert case['dose']['status'] == 'limited-dose'
    assert all(math.isfinite(case['conditions'][name]['off_conditional_tv'])
               for name in ('animal_mask', 'background_mask'))
    assert set(case['conditions']) >= {'cls_animal', 'cls_background', 'spatial_animal',
                                       'spatial_background', 'random_cls', 'random_animal', 'random_background'}
    broken = dict(expected, background=bytes([expected['background'][0] ^ 1]) + expected['background'][1:])
    rejected = Evidence(tmp_path / 'rejected', 23)
    with pytest.raises(ValueError, match='screen logits'):
        capture_initials(model, [image], {42: pixels}, {42: inputs}, {42: broken}, rejected)
    assert rejected.submissions == len(rejected.records) == 3
