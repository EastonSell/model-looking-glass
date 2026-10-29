"""No-checkpoint tests for the same-site small-dose successor."""
import math
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_reused_donors_are_disjoint_from_targets_and_selection_is_deterministic():
    from looking_glass.vision_coco_same_site_cases import greedy_same_site
    rows = [dict(image_id=i, category_id=category, inside={(6, 5)})
            for category, ids in ((17, range(1, 4)), (18, range(11, 14))) for i in ids]
    pairs = greedy_same_site(rows, count=2, excluded_ids=set())
    assert pairs == greedy_same_site(list(reversed(rows)), count=2, excluded_ids=set())
    targets = [p[role]['image_id'] for p in pairs for role in ('cat', 'dog')]
    donors = [p[role]['image_id'] for p in pairs for role in ('cat_donor', 'dog_donor')]
    assert len(set(targets)) == 4 and not set(targets) & set(donors)
    assert len(set(donors)) == 2
    assert all(p['inside'] == (6, 5) for p in pairs)


def test_donor_reuse_allows_identical_bytes_only_and_excludes_v1():
    from looking_glass.vision_coco_same_site_cases import check_reused_images
    prior = [dict(image_id=1, jpeg_sha256='old-jpeg', pixel_sha256='old-pixel')]
    target = dict(image_id=2, jpeg_sha256='new-jpeg', pixel_sha256='new-pixel')
    donor = dict(image_id=3, jpeg_sha256='donor-jpeg', pixel_sha256='donor-pixel')
    check_reused_images(prior, [target, donor, donor.copy()], [2], [3, 3])
    for bad in (dict(donor, jpeg_sha256='other'), dict(donor, pixel_sha256='other')):
        with pytest.raises(ValueError, match='duplicate|inconsistent'):
            check_reused_images(prior, [target, donor, bad], [2], [3, 3])
    with pytest.raises(ValueError, match='duplicate|disjoint'):
        check_reused_images(prior, [target, donor], [2], [2])
    with pytest.raises(ValueError, match='duplicate'):
        check_reused_images(prior, [target, dict(donor, pixel_sha256='old-pixel')], [2], [3])


def test_annotation_selection_uses_largest_instance_and_rejects_mixed_species(monkeypatch):
    import looking_glass.vision_coco_same_site_cases as cases
    ids = list(range(1, 5)) + list(range(11, 15))
    images = [dict(id=i, width=224, height=224, license=4, file_name=f'{i:012d}.jpg',
                   coco_url='coco', flickr_url='flickr') for i in ids]
    annotations = [dict(id=i + 100, image_id=i, category_id=17 if i < 10 else 18,
                        iscrowd=0, bbox=[40, 85, 110, 90],
                        segmentation=[[40, 85, 150, 85, 150, 175, 40, 175]]) for i in ids]
    annotations.append(dict(id=202, image_id=2, category_id=17, iscrowd=0,
                            bbox=[35, 80, 120, 100],
                            segmentation=[[35, 80, 155, 80, 155, 180, 35, 180]]))
    annotations.append(dict(id=214, image_id=14, category_id=17, iscrowd=0,
                            bbox=[40, 85, 110, 90],
                            segmentation=[[40, 85, 150, 85, 150, 175, 40, 175]]))
    source = json.dumps(dict(images=images, annotations=annotations,
                             categories=[dict(id=17, name='cat'), dict(id=18, name='dog')],
                             licenses=[dict(id=4, name='test', url='test')])).encode()
    old_ids = [1] + list(range(1001, 1024))
    prior = json.dumps({'schema': 'vision-coco-vit-spatial-cases-v1',
                        'pairs': [{'cat': {'image_id': old_ids[i]}, 'dog': {'image_id': old_ids[i + 1]}}
                                  for i in range(0, 24, 2)]}).encode()
    monkeypatch.setattr(cases, 'ANNOTATION_SHA256', hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(cases, 'PRIOR_CASES_SHA256', hashlib.sha256(prior).hexdigest())
    pairs, diagnostics = cases.select_same_site(source, prior, count=2)
    all_rows = [p[role] for p in pairs for role in ('cat', 'dog', 'cat_donor', 'dog_donor')]
    assert 1 not in {row['image_id'] for row in all_rows}
    assert 14 not in {row['image_id'] for row in all_rows}
    assert {row['annotation_id'] for row in all_rows if row['image_id'] == 2} == {202}
    assert all(any(box['annotation_id'] == 102 and box['bbox_xywh'] == [40, 85, 110, 90]
                   for box in row['other_boxes']) for row in all_rows if row['image_id'] == 2)
    assert all(row['inside_block_mask_pixels'] >= 512 for row in all_rows)
    assert diagnostics['previously_exposed_images'] == 24
    annotations.append(dict(id=203, image_id=3, category_id=17, iscrowd=0,
                            bbox=[0, 0, 224, 224], segmentation=[[0, 0, 224, 0, 224, 224, 0, 224]]))
    changed = json.dumps(dict(images=images, annotations=annotations,
                              categories=[dict(id=17, name='cat'), dict(id=18, name='dog')],
                              licenses=[dict(id=4, name='test', url='test')])).encode()
    monkeypatch.setattr(cases, 'ANNOTATION_SHA256', hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match='not enough'):
        cases.select_same_site(changed, prior, count=2)


def test_v3_preparation_rejects_shadowed_project_before_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / 'shadow' / 'looking_glass'
    fake.mkdir(parents=True)
    marker = tmp_path / 'shadow-executed'
    (fake / '__init__.py').write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()\n")
    cache = tmp_path / 'empty-cache'
    cache.mkdir()
    prefix = [sys.executable, '-I', '-S', '-B', '-X', f'pycache_prefix={cache}']
    shadow = fake.parent
    positive = subprocess.run(prefix + ['-c', 'import sys; sys.path.append(sys.argv[1]); import looking_glass',
                                        str(shadow)], capture_output=True)
    assert positive.returncode == 0 and marker.exists()
    marker.unlink()
    bootstrap = ("import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2]]); "
                 "sys.argv=[sys.argv[3],'--help']; runpy.run_path(sys.argv[0],run_name='__main__')")
    result = subprocess.run(prefix + ['-c', bootstrap, str(shadow), str(root),
                                      str(root / 'tools/prepare_vision_coco_same_site.py')],
                            capture_output=True)
    assert result.returncode != 0 and not marker.exists()
    assert b'project package is shadowed' in result.stderr


def test_same_site_common_dose_and_donor_distance_cap():
    from looking_glass.vision_coco_same_site_study import prepare_same_site_dose
    assert torch.set_flush_denormal(False)
    own = torch.ones((1, 197, 192), dtype=torch.float32)
    other, same = 3 * own, 4 * own
    tokens = [1, 2, 15, 16]
    dose, values = prepare_same_site_dose(own, other, tokens, 'pair-00', 'cat', same)
    assert dose['status'] == 'matchable'
    assert dose['q'] == pytest.approx(.099 * math.sqrt(768), abs=1e-12)
    assert set(values) == {'opposite', 'same_species', 'random'}
    for value in values.values():
        actual = torch.linalg.vector_norm((value - own[0, tokens]).double()).item()
        assert actual == pytest.approx(dose['q'], rel=1e-5)
        assert actual <= .1 * math.sqrt(768)
    close = own.clone()
    close[0, tokens] += .01
    dose, values = prepare_same_site_dose(own, other, tokens, 'pair-00', 'cat', close)
    assert dose['q'] == pytest.approx(dose['same_species_distance'], abs=1e-12)
    assert torch.equal(values['same_species'], close[0, tokens])


def test_two_direction_composite_includes_absolute_same_and_random_controls():
    from looking_glass.vision_coco_same_site_study import classify_same_site
    good = {'opposite': .1, 'same_species': -.02, 'random': .03}
    bad = dict(good, same_species=-.09)
    assert classify_same_site({'eligible': True}, {'cat': 'matchable', 'dog': 'matchable'},
                              {'cat': good, 'dog': good}) == 'positive'
    assert classify_same_site({'eligible': True}, {'cat': 'matchable', 'dog': 'matchable'},
                              {'cat': good, 'dog': bad}) == 'negative'
    assert classify_same_site({'eligible': False}, {'cat': 'matchable', 'dog': 'matchable'},
                              {'cat': good, 'dog': good}) == 'baseline-ineligible'
