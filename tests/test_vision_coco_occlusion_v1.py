"""Focused geometry and evidence checks for the occlusion study."""
import importlib
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch


def _cases():
    try:
        return importlib.import_module('looking_glass.vision_coco_occlusion_cases')
    except ModuleNotFoundError as error:
        if error.name == 'looking_glass.vision_coco_occlusion_cases':
            return None
        raise


def _image(occupancy=768, other_boxes=()):
    return {'image_id': 42, 'category_id': 17, 'width': 224, 'height': 224,
            'bbox_resized_xyxy': [0.0, 0.0, 64.0, 64.0],
            'sites': [{'row_col': [2, 2], 'mask_pixels': occupancy,
                       'token_mask_pixels': [256, 256, min(256, occupancy - 512),
                                             max(0, occupancy - 768)],
                       'squared_center_radius': 32, 'token_ids': [31, 32, 45, 46]}],
            'other_boxes': list(other_boxes)}


def test_equal_radius_geometry_avoids_all_recorded_boxes_and_breaks_ties():
    cases = _cases()
    assert cases is not None
    box = {'annotation_id': 7, 'category_id': 62, 'bbox_xywh': [160, 32, 32, 32]}
    result = cases.choose_geometry(_image(other_boxes=[box]))
    assert result['animal']['row_col'] == [2, 2]
    assert result['background']['row_col'] == [10, 2]
    assert result['animal']['token_ids'] == [31, 32, 45, 46]
    assert result['background']['token_ids'] == [143, 144, 157, 158]
    assert result['animal']['mask_pixels'] == 768


def test_object_occupancy_below_three_quarters_is_geometry_ineligible():
    cases = _cases()
    assert cases is not None
    assert cases.choose_geometry(_image(occupancy=767)) is None


def test_no_background_square_outside_recorded_boxes_is_geometry_ineligible():
    cases = _cases()
    assert cases is not None
    box = {'annotation_id': 8, 'category_id': 1, 'bbox_xywh': [0, 0, 224, 224]}
    assert cases.choose_geometry(_image(other_boxes=[box])) is None


def test_previous_inventory_requires_the_exact_v6_bytes_and_excludes_all_235_images():
    cases = _cases()
    assert cases is not None
    root = Path(__file__).resolve().parents[1] / 'experiments'
    names = ('vision_coco_spatial_v1.json', 'vision_coco_same_site_v3.json',
             'vision_coco_recognized_v4.json', 'vision_coco_cls_compare_v5.json',
             'vision_coco_spatial_dose_v6.json')
    raw = [(root / name).read_bytes() for name in names]
    rows = cases.previous_images(*raw)
    assert len(rows) == len({row['image_id'] for row in rows}) == 235
    assert len({row['jpeg_sha256'] for row in rows}) == 235
    assert len({row['pixel_sha256'] for row in rows}) == 235
    with pytest.raises(ValueError, match='v6 image inventory differs'):
        cases.previous_images(*raw[:-1], raw[-1] + b' ')


def test_discovery_selection_is_reproducible_from_the_published_v6_screen():
    cases = _cases()
    assert cases is not None
    assert hasattr(cases, 'select_discovery')
    root = Path(__file__).resolve().parents[1]
    manifest = (root / 'experiments/vision_coco_spatial_dose_v6.json').read_bytes()
    screen = (root / 'results/vision-coco-spatial-dose-v6-20261029/screen-result.json').read_bytes()
    rows = cases.select_discovery(manifest, screen)
    assert [row['image_id'] for row in rows] == [328437, 142719, 476736, 503893]
    assert [row['occlusion_geometry']['animal']['row_col'] for row in rows] == [[4, 1], [7, 12], [3, 6], [3, 6]]
    assert [row['occlusion_geometry']['background']['row_col'] for row in rows] == [[1, 4], [0, 7], [6, 3], [6, 9]]
    with pytest.raises(ValueError, match='v6 screen differs'):
        cases.select_discovery(manifest, screen + b' ')


def test_annotation_selection_uses_fresh_rank_and_prior_id_exclusion(tmp_path):
    cases = _cases()
    assert cases is not None
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
    archive = tmp_path / 'annotations.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr(ANNOTATION_MEMBER, json.dumps({'licenses': [{'id': 4}], 'images': images,
                                                    'annotations': annotations,
                                                    'categories': [{'id': 17, 'name': 'cat'},
                                                                   {'id': 18, 'name': 'dog'}]}))
    selected = cases.select_inventory(archive, [{'image_id': 1700}], count=3, authenticated=False)
    for category in (17, 18):
        candidates = [r['id'] for r in images if r['id'] // 100 == category and r['id'] != 1700]
        expected = sorted(candidates, key=lambda i: (hashlib.sha256(
            f'vision-coco-occlusion-v1:{category}:{i}'.encode('ascii')).hexdigest(), i))[:3]
        actual = [r for r in selected if r['category_id'] == category]
        assert [r['image_id'] for r in actual] == expected
        assert [r['rank'] for r in actual] == [0, 1, 2]
        assert all('occlusion_geometry' in r for r in actual)


def _study():
    try:
        return importlib.import_module('looking_glass.vision_coco_occlusion_study')
    except ModuleNotFoundError as error:
        if error.name == 'looking_glass.vision_coco_occlusion_study':
            return None
        raise


def test_mask_only_changes_the_selected_32_pixel_square_and_preserves_other_bits():
    study = _study()
    assert study is not None
    image = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    image[0, 0, 0, 0] = -0.0
    original = image.clone()
    changed = study.mask_pixels(image, (2, 3))
    expected = original.clone()
    expected[:, :, 32:64, 48:80] = 0.0
    assert torch.equal(changed.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(image.view(torch.int32), original.view(torch.int32))


def test_log_group_probability_survives_large_common_logit_offsets():
    study = _study()
    assert study is not None
    logits = torch.full((1, 1000), 1e30, dtype=torch.float32)
    assert study.log_group_probability(logits, 17) == pytest.approx(math.log(5 / 1000))
    assert study.log_group_probability(logits, 18) == pytest.approx(math.log(118 / 1000))


def test_joint_rule_is_inclusive_and_does_not_cancel_large_random_effects():
    study = _study()
    assert study is not None
    assert study.image_decision(0.05, 0.05, 0.025, -0.025)
    assert not study.image_decision(math.nextafter(0.05, -math.inf), 0.05, 0.025, -0.025)
    assert not study.image_decision(0.05, 0.049, 0.025, -0.025)
    assert not study.image_decision(0.10, 0.10, 0.12, 0.12)


def test_decision_is_inconclusive_without_four_images_or_both_species():
    study = _study()
    assert study is not None
    assert study.aggregate_images([(17, True), (17, True), (18, True)])['status'] == 'inconclusive'
    assert study.aggregate_images([(17, True)] * 4)['status'] == 'inconclusive'
    assert study.aggregate_images([(17, True), (17, True), (18, True), (18, False)])['status'] == 'positive'
    assert study.aggregate_images([(17, True), (17, False), (18, True), (18, False)])['status'] == 'negative'


def test_both_sites_and_random_controls_use_one_common_bounded_dose():
    study = _study()
    assert study is not None
    assert hasattr(study, 'prepare_transfers')
    clean = torch.ones((1, 197, 192), dtype=torch.float32)
    animal, background = [31, 32, 45, 46], [143, 144, 157, 158]
    masked_animal, masked_background = clean.clone(), clean.clone()
    masked_animal[0, animal] += 2.0
    masked_background[0, background] -= 3.0
    plan, replacements = study.prepare_transfers(clean, masked_animal, masked_background,
                                                  animal, background, 42)
    assert plan['status'] == 'matchable'
    assert plan['q'] == pytest.approx(.196 * math.sqrt(768), rel=1e-7)
    for site, tokens in [('animal', animal), ('background', background)]:
        for kind in ('transfer', 'random'):
            actual = torch.linalg.vector_norm(
                (replacements[site][kind] - clean[0, tokens]).double()).item()
            assert actual == pytest.approx(plan['q'], rel=1e-6)
        assert plan['q'] / plan['receiver_norms'][site] < .2
    assert torch.equal(replacements['animal']['random'], replacements['background']['random'])


def test_tiny_donor_distance_is_retained_as_limited_or_ineligible():
    study = _study()
    assert study is not None
    assert hasattr(study, 'prepare_transfers')
    clean = torch.ones((1, 197, 192), dtype=torch.float32)
    animal, background = [31, 32, 45, 46], [143, 144, 157, 158]
    small = clean.clone()
    small[0, animal] += .005
    distant = clean.clone()
    distant[0, background] += 1.0
    assert study.prepare_transfers(clean, small, distant, animal, background, 42)[0]['status'] == 'limited-dose'
    assert study.prepare_transfers(clean, clean, distant, animal, background, 42)[0]['status'] == 'dose-ineligible'


def test_complete_fabricated_case_saves_all_controls_and_same_image_transfers(tmp_path):
    study = _study()
    assert study is not None
    assert hasattr(study, 'run_case')
    from looking_glass.vision_coco_study import Evidence

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
            weights = torch.linspace(-1, 1, 1000, device=state.device).reshape(1, -1)
            return SimpleNamespace(logits=state[:, :1, 0] * weights)

    image = _image()
    image['occlusion_geometry'] = _cases().choose_geometry(image)
    pixels = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    evidence = Evidence(tmp_path / 'evidence', 20)
    row = study.run_case(Fake().eval(), image, pixels, evidence)
    assert row['image_id'] == 42
    assert row['dose']['status'] == 'matchable'
    assert evidence.submissions == len(evidence.records) == 18
    assert set(row['conditions']) == {'animal_mask', 'background_mask', 'repeat',
                                     'animal_null', 'background_null', 'animal_self',
                                     'background_self', 'full_self', 'animal_full',
                                     'background_full', 'animal_transfer', 'background_transfer',
                                     'animal_random', 'background_random', 'animal_final',
                                     'background_final', 'post_clean'}
    assert (tmp_path / 'evidence/case-42.json').is_file()
    assert all((tmp_path / f'evidence/input-42-{name}.bin').stat().st_size == 3 * 224 * 224 * 4
               for name in ('clean', 'animal', 'background'))


def test_screen_selection_uses_only_fixed_rank_recognition_and_geometry():
    study = _study()
    assert study is not None
    assert hasattr(study, 'select_screened')
    images = [{'image_id': category * 100 + rank, 'category_id': category, 'rank': rank,
               'occlusion_geometry': None if rank == 1 else {'fixed': True}}
              for category in (17, 18) for rank in range(6)]
    observed = [{'image_id': row['image_id'], 'recognized': row['rank'] != 2}
                for row in images]
    selected, counts = study.select_screened(images, observed)
    assert [row['image_id'] for row in selected] == [1700, 1703, 1704, 1705, 1800, 1803, 1804, 1805]
    assert counts['cat'] == {'candidates': 6, 'geometry_qualified': 5, 'screened': 6,
                             'recognized': 5, 'recognized_geometry': 4, 'selected': 4}
    changed = list(observed)
    changed[-1] = {**changed[-1], 'image_id': 9999}
    with pytest.raises(ValueError, match='screen identity'):
        study.select_screened(images, changed)


def test_fabricated_screen_saves_all_logits_before_fixed_rank_selection(tmp_path):
    study = _study()
    assert study is not None
    assert hasattr(study, 'run_screen')
    from looking_glass.vision_coco_study import Evidence

    class FakeClassifier(torch.nn.Module):
        def forward(self, pixel_values):
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[0, 281 if pixel_values[0, 0, 0, 0] == 1 else 151] = 10.0
            return SimpleNamespace(logits=logits)

    images, pixels = [], {}
    pixel_values = {17: torch.ones((1, 3, 224, 224)), 18: torch.full((1, 3, 224, 224), 2.0)}
    for category in (17, 18):
        for rank in range(32):
            row = _image()
            ident = category * 100 + rank
            row.update(image_id=ident, category_id=category, rank=rank,
                       role='donor' if rank % 4 == 0 else 'target', file_name=f'{ident:012d}.jpg',
                       jpeg_bytes=1, jpeg_sha256=f'{ident:064x}', pixel_sha256=f'{ident + 10000:064x}')
            row['occlusion_geometry'] = _cases().choose_geometry(row)
            images.append(row)
            pixels[ident] = pixel_values[category]
    evidence = Evidence(tmp_path / 'screen', 64)
    observed, selected, counts = study.run_screen(FakeClassifier().eval(), images, pixels, evidence)
    assert len(observed) == evidence.submissions == len(evidence.records) == 64
    assert [row['image_id'] for row in selected] == [1700, 1701, 1702, 1703, 1800, 1801, 1802, 1803]
    assert counts['cat']['recognized'] == counts['dog']['recognized'] == 32
    assert (tmp_path / 'screen/forward-064.bin').stat().st_size == 4000
    assert hasattr(study, 'verify_saved_screen')
    result = {'schema': 'vision-occlusion-v1-screen-result', 'manifest_sha256': 'a' * 64, 'screen': observed,
              'selected_ids': [row['image_id'] for row in selected], 'counts': counts}
    evidence.json('screen-result.json', result)
    verified = study.verify_saved_screen(images, evidence.root, expected_manifest_sha256='a' * 64,
                                         expected_result=(evidence.root / 'screen-result.json').read_bytes())
    assert list(verified['raw_by_id']) == [row['image_id'] for row in images]
    with pytest.raises(ValueError, match='manifest'):
        study.verify_saved_screen(images, evidence.root, expected_manifest_sha256='b' * 64)
    from tools.run_vision_coco_occlusion import verify_public_screen_archive
    evidence.json('started.json', {'phase': 'fake'})
    evidence.json('terminal.json', {'phase': 'fake'})
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w') as archive:
        for path in sorted(evidence.root.iterdir()):
            archive.writestr(path.name, path.read_bytes())
    archived = payload.getvalue()
    assert verify_public_screen_archive(evidence.root, archived) == hashlib.sha256(archived).hexdigest()
    first = evidence.root / 'forward-001.bin'
    original = first.read_bytes()
    first.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match='archive member differs'):
        verify_public_screen_archive(evidence.root, archived)
    first.write_bytes(original)
    corrupted = dict(result, selected_ids=result['selected_ids'][1:])
    (evidence.root / 'screen-result.json').write_text(json.dumps(corrupted), encoding='utf-8')
    with pytest.raises(ValueError, match='selected'):
        study.verify_saved_screen(images, evidence.root)
