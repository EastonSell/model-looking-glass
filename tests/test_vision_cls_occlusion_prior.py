"""Fabricated prior-evidence binding for the exposed discovery cases."""
import hashlib
import json

import pytest
import torch


def test_discovery_requires_prior_clean_and_both_masked_bytes(tmp_path):
    from looking_glass.vision_coco_cls_occlusion_cases import PRIOR_OCCLUSION_SHA256
    from looking_glass.vision_coco_cls_occlusion_prior import verify_prior_discovery
    from looking_glass.vision_coco_occlusion_cases import choose_geometry
    from looking_glass.vision_coco_occlusion_study import mask_pixels
    from looking_glass.vision_coco_study import Evidence, _bits

    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    pixels = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    image = {'image_id': 42, 'category_id': 17, 'width': 224, 'height': 224,
             'pixel_sha256': sha(_bits(pixels)), 'bbox_resized_xyxy': [0., 0., 64., 64.],
             'sites': [{'row_col': [2, 2], 'mask_pixels': 768, 'token_mask_pixels': [256, 256, 256, 0],
                        'squared_center_radius': 32, 'token_ids': [31, 32, 45, 46]}], 'other_boxes': []}
    image['occlusion_geometry'] = choose_geometry(image)
    evidence = Evidence(tmp_path / 'data', 3)
    inputs = {'clean': _bits(pixels)}
    for site in ('animal', 'background'):
        inputs[site] = _bits(mask_pixels(pixels, image['occlusion_geometry'][site]['row_col']))
    input_sha = {key: evidence.write(f'input-42-{key}.bin', value) for key, value in inputs.items()}
    expected_logits, hashes = {}, []
    for index, name in enumerate(('clean', 'animal', 'background'), 1):
        label = 'case-42:' + ('clean' if name == 'clean' else name + '_mask')
        evidence.reserve(label)
        logits = torch.full((1, 1000), float(index), dtype=torch.float32)
        row = evidence.save_forward_bytes(label, {'logits': logits})
        evidence.finish_forward(row, {}, {})
        expected_logits[name] = _bits(logits)
        hashes.append({'submission': index, 'binary_sha256': row['sha256'],
                       'json_sha256': sha((evidence.root / f'forward-{index:03d}.json').read_bytes())})
    case = {'image_id': 42, 'category_id': 17, 'input_sha256': input_sha}
    evidence.json('case-42.json', case)
    start_sha = evidence.json('started.json', {'mode': 'evaluation'})
    terminal_sha = evidence.json('terminal.json', {'mode': 'evaluation', 'status': 'complete', 'cases': [case]})
    audit = {'schema': 'vision-occlusion-independent-intervention-audit-v1', 'synthetic': False,
             'result': {'mode': 'evaluation', 'manifest_sha256': PRIOR_OCCLUSION_SHA256,
                        'phase_start_sha256': start_sha, 'phase_terminal_sha256': terminal_sha,
                        'forward_hashes': hashes, 'forwards': 3}}
    audit_raw = json.dumps(audit).encode()
    verified = verify_prior_discovery([image], evidence.root, audit_raw,
                                      expected_audit_sha256=sha(audit_raw), expected_terminal_sha256=terminal_sha)
    assert verified['inputs'][42] == inputs
    assert verified['logits'][42] == expected_logits
    path = evidence.root / 'input-42-background.bin'
    broken = bytearray(path.read_bytes())
    broken[-1] ^= 1
    path.write_bytes(broken)
    with pytest.raises(ValueError, match='prior input'):
        verify_prior_discovery([image], evidence.root, audit_raw,
                               expected_audit_sha256=sha(audit_raw), expected_terminal_sha256=terminal_sha)
