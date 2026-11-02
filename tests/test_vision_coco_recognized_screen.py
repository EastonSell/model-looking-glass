import math
import hashlib
import io
import os
import struct
from types import SimpleNamespace
import zipfile

import pytest
import torch

from looking_glass.vision_coco_recognized_screen import aggregate_recognized, screen_score, select_pairs


def _logits(category):
    values = [0.] * 1000
    values[281 if category == 17 else 151] = 8.
    return struct.pack('<1000f', *values)


def _screen_fixture():
    images, screen = [], []
    for category in (17, 18):
        for rank in range(8):
            image_id = category * 100 + rank
            images.append({'image_id': image_id, 'category_id': category, 'rank': rank,
                           'role': 'donor' if rank % 4 == 0 else 'target',
                           'sites': [{'row_col': [6, 6]}]})
            screen.append({'image_id': image_id, **screen_score(_logits(category), category)})
    return images, screen


def _full_manifest_images():
    images = []
    for category in (17, 18):
        for rank in range(32):
            image_id = category * 100 + rank
            images.append({'image_id': image_id, 'category_id': category, 'rank': rank,
                           'file_name': f'{image_id:012d}.jpg', 'jpeg_bytes': 1,
                           'jpeg_sha256': hashlib.sha256(f'jpeg-{image_id}'.encode()).hexdigest(),
                           'pixel_sha256': hashlib.sha256(f'pixels-{image_id}'.encode()).hexdigest(),
                           'role': 'donor' if rank % 4 == 0 else 'target',
                           'sites': [{'row_col': [6, 6], 'token_ids': [91, 92, 105, 106],
                                      'squared_center_radius': 0, 'mask_pixels': 512,
                                      'token_mask_pixels': [128, 128, 128, 128]}]})
    return images


def test_screen_decodes_raw_bits_and_classifies_recognition():
    cat, dog = screen_score(_logits(17), 17), screen_score(_logits(18), 18)
    assert cat['recognized'] and cat['margin'] < 0 and cat['top1_id'] == 281
    assert dog['recognized'] and dog['margin'] > 0 and dog['top1_id'] == 151
    assert screen_score(_logits(17), 18)['reasons'] == ['top1-outside-annotated-group', 'group-mass-below-0.2']
    bits = bytearray(_logits(17))
    bits[:4] = struct.pack('<I', 0x7f800000)
    with pytest.raises(ValueError, match='nonfinite'):
        screen_score(bytes(bits), 17)
    from looking_glass.vision_coco_recognized_screen import _binary32
    assert _binary32(1) == 2 ** -149 and _binary32(0x80000001) == -(2 ** -149)


def test_host_rounding_sentinel_rejects_directed_modes():
    if os.name != 'nt':
        pytest.skip('Windows CRT rounding-mode probe')
    import ctypes
    from looking_glass.vision_coco_recognized_screen import check_rounding
    control = ctypes.CDLL('msvcrt')._controlfp
    control.argtypes = (ctypes.c_uint, ctypes.c_uint)
    control.restype = ctypes.c_uint
    old = control(0, 0)
    try:
        control(0x000, 0x300)
        check_rounding()
        for mode in (0x100, 0x200, 0x300):
            control(mode, 0x300)
            with pytest.raises(ValueError, match='round-to-nearest'):
                check_rounding()
    finally:
        control(old, 0x300)


def test_fixed_donor_roles_pair_order_and_reuse_balancing():
    images, screen = _screen_fixture()
    pairs, summary = select_pairs(images, screen)
    assert [p['cat']['rank'] for p in pairs] == [1, 2, 3, 5]
    assert [p['dog']['rank'] for p in pairs] == [1, 2, 3, 5]
    assert [p['cat_donor']['rank'] for p in pairs] == [0, 4, 0, 4]
    assert [p['dog_donor']['rank'] for p in pairs] == [0, 4, 0, 4]
    assert summary['selected_pairs'] == 4 and summary['distinct_selected_targets'] == 8
    assert summary['distinct_selected_donors'] == 4
    assert set(summary['donor_side_counts'].values()) == {2}
    assert all(p['box_tokens'] == [91, 92, 105, 106] for p in pairs)


def test_unrecognized_donor_and_incompatible_geometry_are_rejections():
    images, screen = _screen_fixture()
    for image, row in zip(images, screen):
        if image['role'] == 'donor' and image['category_id'] == 17:
            row['recognized'] = False
    pairs, summary = select_pairs(images, screen)
    assert not pairs
    assert summary['planned_decision'] == 'inconclusive' and summary['interventions_planned'] is False
    assert summary['examined_rejections']
    assert set(r['reason'] for r in summary['examined_rejections']) == {
        'no-recognized-same-species-donors-at-common-site'}
    images[1]['role'] = 'donor'
    with pytest.raises(ValueError, match='identities or roles'):
        select_pairs(images, screen)


@pytest.mark.parametrize(('statuses', 'expected'), [([], 'inconclusive'),
    (['positive'], 'inconclusive'), (['positive', 'positive'], 'positive'),
    (['positive', 'negative'], 'negative'), (['positive', 'positive', 'negative'], 'negative'),
    (['positive', 'positive', 'positive', 'negative'], 'positive'),
    (['positive', 'dose-unmatchable'], 'inconclusive')])
def test_aggregate_fixed_small_sample_threshold(statuses, expected):
    rows = [dict(pair_id=f'pair-{i:02d}', split='heldout', status=status) for i, status in enumerate(statuses)]
    assert aggregate_recognized(rows)['status'] == expected


def test_selected_baseline_ineligibility_is_incomplete_not_excluded():
    rows = [{'pair_id': 'pair-00', 'split': 'heldout', 'status': 'baseline-ineligible'}]
    with pytest.raises(ValueError, match='selected-pair'):
        aggregate_recognized(rows)


def test_screen_clean_mismatch_stops_before_patch(tmp_path):
    from looking_glass.vision_coco_study import Evidence
    from looking_glass.vision_coco_same_site_study import run_same_site
    class Layer(torch.nn.Module):
        def forward(self, x):
            return x + .01 * x.mean(dim=1, keepdim=True)
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = torch.nn.Module()
            self.vit.encoder = torch.nn.Module()
            self.vit.encoder.layer = torch.nn.ModuleList([Layer() for _ in range(12)])
        def forward(self, pixel_values):
            for layer in self.vit.encoder.layer:
                pixel_values = layer(pixel_values)
            return SimpleNamespace(logits=torch.zeros((1, 1000), dtype=torch.float32))
    assert torch.set_flush_denormal(False)
    pixels = {i: torch.ones((1, 197, 192)) for i in range(1, 5)}
    pair = {'pair_id': 'pair-00', 'split': 'heldout', 'box_tokens': [1, 2, 15, 16],
            'cat': {'image_id': 1}, 'dog': {'image_id': 2},
            'cat_donor': {'image_id': 3}, 'dog_donor': {'image_id': 4}}
    evidence = Evidence(tmp_path / 'evidence', 64)
    def reject(*args):
        raise ValueError('screen output differs')
    with pytest.raises(ValueError, match='screen output differs'):
        run_same_site(Fake().eval(), [pair], pixels, evidence, check_clean=reject)
    assert evidence.submissions == len(evidence.records) == 1
    assert evidence.records[0]['label'].endswith(':initial')


def test_64_screen_outputs_are_retained_before_selection(tmp_path):
    from looking_glass.vision_coco_recognized_runner import run_clean_screen
    from looking_glass.vision_coco_study import Evidence
    class Fake(torch.nn.Module):
        def forward(self, pixel_values):
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[0, 281 if pixel_values.item() == 17 else 151] = 8.
            return SimpleNamespace(logits=logits)
    images = _full_manifest_images()
    pixels = {row['image_id']: torch.tensor([[float(row['category_id'])]]) for row in images}
    assert torch.set_flush_denormal(False)
    evidence = Evidence(tmp_path / 'screen', 64)
    observed, pairs, summary = run_clean_screen(Fake().eval(), images, pixels, evidence)
    assert evidence.submissions == len(evidence.records) == len(observed) == 64
    assert all(row['recognized'] for row in observed)
    assert len(pairs) == 4 and summary['target_role_images'] == 48 and summary['donor_role_images'] == 16
    assert len(list(evidence.root.glob('forward-*.bin'))) == 64


def test_screen_invalid_return_still_retains_returned_logit_bytes(tmp_path):
    from looking_glass.vision_coco_recognized_runner import run_clean_screen
    from looking_glass.vision_coco_study import Evidence
    class Fake(torch.nn.Module):
        def forward(self, pixel_values):
            return SimpleNamespace(logits=torch.zeros((1000,), dtype=torch.float32))
    images = _full_manifest_images()
    pixels = {row['image_id']: torch.ones((1, 1)) for row in images}
    evidence = Evidence(tmp_path / 'screen', 64)
    with pytest.raises(ValueError, match='shape'):
        run_clean_screen(Fake().eval(), images, pixels, evidence)
    assert evidence.submissions == 1 and not evidence.records
    assert (evidence.root / 'forward-001.bin').stat().st_size == 4000


def test_invalid_manifest_role_or_site_fails_before_any_screen_forward(tmp_path):
    from looking_glass.vision_coco_recognized_runner import run_clean_screen
    from looking_glass.vision_coco_study import Evidence
    images = _full_manifest_images()
    pixels = {row['image_id']: torch.ones((1, 1)) for row in images}
    evidence = Evidence(tmp_path / 'screen', 64)
    images[-1]['role'] = 'donor'
    with pytest.raises(ValueError, match='role'):
        run_clean_screen(object(), images, pixels, evidence)
    assert evidence.submissions == 0


def test_later_selected_baseline_is_validated_before_patching():
    from looking_glass.vision_coco_recognized_handoff import validate_pair_baselines
    images, screen = _screen_fixture()
    pairs, _ = select_pairs(images, screen)
    validate_pair_baselines(pairs, screen)
    by_id = {row['image_id']: row for row in screen}
    by_id[pairs[-1]['dog']['image_id']]['margin'] = -1.
    with pytest.raises(ValueError, match='baseline contradicts'):
        validate_pair_baselines(pairs, screen)


def test_final_screen_denormal_drift_retains_last_output(tmp_path, monkeypatch):
    from looking_glass import vision_coco_recognized_runner as runner
    from looking_glass.vision_coco_study import Evidence
    class Fake(torch.nn.Module):
        def forward(self, pixel_values):
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[0, 281 if pixel_values.item() == 17 else 151] = 8.
            return SimpleNamespace(logits=logits)
    images = _full_manifest_images()
    pixels = {row['image_id']: torch.tensor([[float(row['category_id'])]]) for row in images}
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls == 128:
            raise ValueError('denormal drift')
    monkeypatch.setattr(runner, 'check_denormals', check)
    evidence = Evidence(tmp_path / 'screen', 64)
    with pytest.raises(ValueError, match='denormal drift'):
        runner.run_clean_screen(Fake().eval(), images, pixels, evidence)
    assert evidence.submissions == 64 and len(evidence.records) == 63
    assert (evidence.root / 'forward-064.bin').stat().st_size == 4000
    assert not (evidence.root / 'forward-064.json').exists()


def test_final_screen_rounding_drift_retains_last_output(tmp_path):
    if os.name != 'nt':
        pytest.skip('Windows CRT rounding-mode probe')
    import ctypes
    from looking_glass.vision_coco_recognized_runner import run_clean_screen
    from looking_glass.vision_coco_study import Evidence
    control = ctypes.CDLL('msvcrt')._controlfp
    control.argtypes = (ctypes.c_uint, ctypes.c_uint)
    control.restype = ctypes.c_uint
    old = control(0, 0)
    class Fake(torch.nn.Module):
        calls = 0
        def forward(self, pixel_values):
            self.calls += 1
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[0, 281 if pixel_values.item() == 17 else 151] = 8.
            if self.calls == 64:
                control(0x200, 0x300)
            return SimpleNamespace(logits=logits)
    images = _full_manifest_images()
    pixels = {row['image_id']: torch.tensor([[float(row['category_id'])]]) for row in images}
    evidence = Evidence(tmp_path / 'screen', 64)
    try:
        control(0x000, 0x300)
        assert torch.set_flush_denormal(False)
        with pytest.raises(ValueError, match='round-to-nearest'):
            run_clean_screen(Fake().eval(), images, pixels, evidence)
    finally:
        control(old, 0x300)
    assert evidence.submissions == 64 and len(evidence.records) == 63
    assert (evidence.root / 'forward-064.bin').stat().st_size == 4000
    assert not (evidence.root / 'forward-064.json').exists()


def test_closed_screen_reconstructs_selection_and_rejects_corruption(tmp_path):
    import json
    from looking_glass.vision_coco_recognized_runner import run_clean_screen
    from looking_glass.vision_coco_recognized_handoff import (MODEL_STATE_SHA256, canonical, screen_receipt,
                                                               verify_closed_screen, verify_screen_archive)
    from looking_glass.vision_coco_study import Evidence
    class Fake(torch.nn.Module):
        def forward(self, pixel_values):
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[0, 281 if pixel_values.item() == 17 else 151] = 8.
            return SimpleNamespace(logits=logits)
    images = _full_manifest_images()
    manifest = {'schema': 'vision-coco-recognized-screen-cases-v4', 'images': images}
    pixels = {row['image_id']: torch.tensor([[float(row['category_id'])]]) for row in images}
    runtime = {'device': 'cpu', 'cuda_visible_devices': '', 'cuda_initialized': False,
               'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
               'numpy': '2.4.6', 'pillow': '11.3.0', 'threads': 1, 'interop_threads': 1,
               'deterministic_algorithms': True, 'float32_matmul_precision': 'highest',
               'mkldnn_enabled': True, 'default_dtype': 'torch.float32',
               'processor': 'ViTImageProcessor', 'denormals_preserved': True,
               'mkl_cbwr': 'COMPATIBLE', 'binary64_rounding': 'nearest-even'}
    evidence = Evidence(tmp_path / 'screen', 64)
    evidence.json('started.json', {'schema': 'vision-recognized-v4-start', 'mode': 'screen',
                                  'manifest_sha256': 'm', 'source': {'revision': 'fake'},
                                  'runtime': runtime, 'model_state_sha256': MODEL_STATE_SHA256})
    assert torch.set_flush_denormal(False)
    observed, pairs, summary = run_clean_screen(Fake().eval(), images, pixels, evidence)
    result = {'schema': 'vision-recognized-v4-screen-result', 'manifest_sha256': 'm',
              'screen': observed, 'pairs': pairs, 'summary': summary}
    digest = evidence.json('screen-result.json', result)
    evidence.json('terminal.json', {'schema': 'vision-recognized-v4-terminal', 'status': 'complete',
                  'mode': 'screen', 'manifest_sha256': 'm', 'source_revision': 'fake',
                  'submissions': 64, 'saved_forwards': 64, 'runtime': runtime,
                  'model_state_sha256_before_after': MODEL_STATE_SHA256, 'screen_result_sha256': digest})
    report = verify_closed_screen(manifest, evidence.root, 'm', expected_result=canonical(result))
    assert report['result'] == result and len(report['raw_by_id']) == 64
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(evidence.root.iterdir()):
            archive.writestr(item.name, item.read_bytes())
    archive_raw = payload.getvalue()
    archive_sha = verify_screen_archive(archive_raw, evidence.root)
    (tmp_path / 'supervisor.json').write_bytes(canonical({'status': 'complete', 'exit_code': 0,
        'clean': True, 'timed_out': False, 'terminal_sha256': report['terminal_sha256']}))
    supervisor_hash = hashlib.sha256((tmp_path / 'supervisor.json').read_bytes()).hexdigest()
    observer = canonical({'schema': 'vision-recognized-v4-external-observer', 'status': 'complete',
                          'mode': 'screen', 'source_revision': 'fake', 'controller_exit_code': 0,
                          'supervisor_sha256': supervisor_hash,
                          'controller_reaped': True, 'timed_out': False, 'forced_stop': False,
                          'terminal_sha256': report['terminal_sha256']})
    projection = screen_receipt(report, evidence.root, archive_sha, observer)
    assert projection['owner_exit_code'] == 0 and projection['screen_archive_sha256'] == archive_sha
    with pytest.raises(ValueError, match='external screen controller'):
        screen_receipt(report, evidence.root, archive_sha, observer.replace(b'"controller_exit_code":0',
                                                                            b'"controller_exit_code":1'))
    with pytest.raises(ValueError, match='external screen controller'):
        screen_receipt(report, evidence.root, archive_sha, observer.replace(b'"source_revision":"fake"',
                                                                            b'"source_revision":"other"'))
    record = evidence.root / 'forward-064.bin'
    raw = bytearray(record.read_bytes())
    raw[0] ^= 1
    record.write_bytes(raw)
    with pytest.raises(ValueError, match='forward differs'):
        verify_closed_screen(manifest, evidence.root, 'm')
    with pytest.raises(ValueError, match='archive member differs'):
        verify_screen_archive(archive_raw, evidence.root)
