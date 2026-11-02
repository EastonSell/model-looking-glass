"""Fabricated-only tests of the new v2 selection, dose and decision rules."""
import math
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_greedy_quadruplets_are_disjoint_and_order_independent():
    from looking_glass.vision_coco_small_dose_cases import greedy_quadruplets
    rows = [dict(image_id=i, category_id=category,
                 inside={(6, 5)}, outside={(5, 6)})
            for category, ids in ((17, range(1, 5)), (18, range(11, 15))) for i in ids]
    selected = greedy_quadruplets(rows, count=2, excluded_ids=set())
    assert selected == greedy_quadruplets(list(reversed(rows)), count=2, excluded_ids=set())
    assert len(selected) == 2
    assert all(q['inside'] == (6, 5) and q['outside'] == (5, 6) for q in selected)
    ids = [q[key]['image_id'] for q in selected for key in ('cat', 'dog', 'cat_donor', 'dog_donor')]
    assert len(ids) == len(set(ids)) == 8
    assert all(q['cat']['category_id'] == q['cat_donor']['category_id'] == 17 and
               q['dog']['category_id'] == q['dog_donor']['category_id'] == 18 for q in selected)
    with pytest.raises(ValueError, match='not enough'):
        greedy_quadruplets(rows, count=2, excluded_ids={1})


def test_annotation_selection_excludes_prior_images_and_uses_unique_masked_donors(monkeypatch):
    import looking_glass.vision_coco_small_dose_cases as cases
    image_ids = list(range(1, 7)) + list(range(11, 17))
    annotations = []
    images = []
    for i in image_ids:
        images.append(dict(id=i, width=224, height=224, license=4,
                           file_name=f'{i:012d}.jpg', coco_url='coco', flickr_url='flickr'))
        annotations.append(dict(id=i + 100, image_id=i, category_id=17 if i < 10 else 18,
                                iscrowd=0, bbox=[40, 85, 110, 90],
                                segmentation=[[40, 85, 150, 85, 150, 175, 40, 175]]))
    source = json.dumps(dict(images=images, annotations=annotations,
                             categories=[dict(id=17, name='cat'), dict(id=18, name='dog')],
                             licenses=[dict(id=4, name='test', url='test')])).encode()
    old_ids = [1] + list(range(1001, 1024))
    prior = json.dumps({'schema': 'vision-coco-vit-spatial-cases-v1',
                        'pairs': [{'cat': {'image_id': old_ids[i]},
                                   'dog': {'image_id': old_ids[i + 1]}}
                                  for i in range(0, 24, 2)]}).encode()
    monkeypatch.setattr(cases, 'ANNOTATION_SHA256', hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(cases, 'PRIOR_CASES_SHA256', hashlib.sha256(prior).hexdigest())
    pairs, diagnostics = cases.select_small_dose(source, prior, count=2)
    rows = [p[key] for p in pairs for key in ('cat', 'dog', 'cat_donor', 'dog_donor')]
    assert len({r['image_id'] for r in rows}) == 8
    assert 1 not in {r['image_id'] for r in rows}
    assert all(r['inside_block_mask_pixels'] >= 512 for r in rows)
    assert all(p['split'] == 'heldout' for p in pairs)
    assert diagnostics['previously_exposed_images'] == 24


def test_image_hash_checks_reject_prior_and_within_study_duplicates():
    from looking_glass.vision_coco_small_dose_cases import check_image_hashes
    prior = [{'image_id': 1, 'jpeg_sha256': 'old-jpeg', 'pixel_sha256': 'old-pixel'}]
    fresh = [dict(image_id=i, jpeg_sha256=f'jpeg-{i}', pixel_sha256=f'pixel-{i}') for i in (2, 3)]
    check_image_hashes(prior, fresh)
    for bad in (dict(fresh[0], image_id=1), dict(fresh[0], jpeg_sha256='old-jpeg'),
                dict(fresh[0], pixel_sha256='old-pixel'), dict(fresh[0], jpeg_sha256='jpeg-3'),
                dict(fresh[0], pixel_sha256='pixel-3')):
        with pytest.raises(ValueError, match='duplicate'):
            check_image_hashes(prior, [bad, fresh[1]])


def test_preparation_rejects_shadowed_project_before_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shadow = tmp_path / 'shadow'
    fake = shadow / 'looking_glass'
    fake.mkdir(parents=True)
    marker = tmp_path / 'shadow-executed'
    (fake / '__init__.py').write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()\n")
    cache = tmp_path / 'empty-cache'
    cache.mkdir()
    prefix = [sys.executable, '-I', '-S', '-B', '-X', f'pycache_prefix={cache}']
    positive = subprocess.run(prefix + ['-c', 'import sys; sys.path.append(sys.argv[1]); import looking_glass',
                                        str(shadow)], capture_output=True)
    assert positive.returncode == 0 and marker.exists()
    marker.unlink()
    bootstrap = ("import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2]]); "
                 "sys.argv=[sys.argv[3],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')")
    result = subprocess.run(prefix + ['-c', bootstrap, str(shadow), str(root),
                                      str(root / 'tools/prepare_vision_coco_small_dose.py')],
                            capture_output=True)
    assert result.returncode != 0 and not marker.exists()
    assert b'project package is shadowed' in result.stderr


def test_small_common_dose_caps_each_site_and_never_extrapolates():
    from looking_glass.vision_coco_small_dose_study import prepare_small_dose
    assert torch.set_flush_denormal(False)
    own = torch.ones((1, 197, 192), dtype=torch.float32)
    other = 3 * own
    same = 4 * own
    inside, outside = [1, 2, 15, 16], [40, 41, 54, 55]
    status, requested = prepare_small_dose(own, other, inside, outside, 'pair-00', 'cat', same)
    assert status['status'] == 'matchable'
    assert status['q'] == pytest.approx(.099 * math.sqrt(768), abs=1e-12)
    assert set(requested) == {'inside', 'outside', 'same_species', 'random'}
    for name, value in requested.items():
        receiver = own[0, outside if name == 'outside' else inside]
        norm = torch.linalg.vector_norm((value - receiver).double()).item()
        assert norm == pytest.approx(status['q'], rel=1e-5)
        assert norm <= .10 * math.sqrt(768)
    close_same = own.clone()
    close_same[0, inside] += .01
    smaller, req = prepare_small_dose(own, other, inside, outside, 'pair-00', 'cat', close_same)
    assert smaller['status'] == 'matchable'
    assert smaller['q'] == pytest.approx(smaller['same_species_distance'], abs=1e-12)
    assert torch.equal(req['same_species'], close_same[0, inside])


def test_zero_same_species_distance_is_not_a_negative_or_extrapolated():
    from looking_glass.vision_coco_small_dose_study import prepare_small_dose
    assert torch.set_flush_denormal(False)
    own = torch.ones((1, 197, 192), dtype=torch.float32)
    status, requested = prepare_small_dose(own, 2 * own, [1, 2, 15, 16], [40, 41, 54, 55],
                                           'pair-00', 'cat', own.clone())
    assert status['status'] == 'dose-ineligible'
    assert 'same-species-distance' in status['reasons']
    assert requested == {}


def test_both_sides_must_beat_every_absolute_control_at_fixed_thresholds():
    from looking_glass.vision_coco_small_dose_study import classify_small_pair
    base = {'eligible': True}
    good = {'inside': .10, 'outside': -.02, 'same_species': .03, 'random': -.04}
    fail = dict(good, same_species=-.09)
    assert classify_small_pair(base, {'cat': 'matchable', 'dog': 'matchable'},
                               {'cat': good, 'dog': good}) == 'positive'
    assert classify_small_pair(base, {'cat': 'matchable', 'dog': 'matchable'},
                               {'cat': good, 'dog': fail}) == 'negative'
    assert classify_small_pair({'eligible': False}, {'cat': 'matchable', 'dog': 'matchable'},
                               {'cat': good, 'dog': good}) == 'baseline-ineligible'
    assert classify_small_pair(base, {'cat': 'matchable', 'dog': 'dose-ineligible'},
                               {'cat': good, 'dog': good}) == 'dose-ineligible'


def test_aggregate_preserves_ineligible_denominator_and_minimum_four():
    from looking_glass.vision_coco_small_dose_study import aggregate_small
    def rows(statuses):
        return [dict(pair_id=f'pair-{i:02d}', split='heldout', status=status)
                for i, status in enumerate(statuses)]
    excluded = ['baseline-ineligible'] * 4
    assert aggregate_small(rows(['positive'] * 3 + ['baseline-ineligible'] + excluded))['status'] == 'inconclusive'
    assert aggregate_small(rows(['positive'] * 3 + ['negative'] + excluded))['status'] == 'positive'
    assert aggregate_small(rows(['positive'] * 2 + ['negative'] * 2 + excluded))['status'] == 'negative'
