"""Fabricated response-screen handoff tests; no pretrained model is loaded."""
import copy
import hashlib
import json

import pytest
import torch


def test_closed_response_screen_binds_masked_pixels_logits_and_selection(tmp_path):
    from looking_glass.vision_coco_cls_occlusion_screen import verify_saved_response_screen
    from looking_glass.vision_coco_cls_occlusion_study import select_response_screen
    from looking_glass.vision_coco_occlusion_cases import choose_geometry
    from looking_glass.vision_coco_occlusion_study import log_group_probability, mask_pixels
    from looking_glass.vision_coco_recognized_screen import screen_score
    from looking_glass.vision_coco_study import Evidence, _bits, score

    data = Evidence(tmp_path / 'data', 66)
    pixels = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    images, observed = [], []
    for category in (17, 18):
        for rank in range(32):
            ident = category * 1000 + rank
            image = {'image_id': ident, 'category_id': category, 'rank': rank,
                     'role': 'donor' if rank % 4 == 0 else 'target',
                     'file_name': f'{ident:012d}.jpg', 'jpeg_bytes': 1,
                     'jpeg_sha256': hashlib.sha256(f'jpeg-{ident}'.encode()).hexdigest(),
                     'pixel_sha256': hashlib.sha256(_bits(pixels)).hexdigest() if ident == 17000 else
                                     hashlib.sha256(f'pixels-{ident}'.encode()).hexdigest(),
                     'width': 224, 'height': 224, 'bbox_resized_xyxy': [0., 0., 64., 64.],
                     'sites': [{'row_col': [2, 2], 'mask_pixels': 768,
                                'token_mask_pixels': [256, 256, 256, 0],
                                'squared_center_radius': 32, 'token_ids': [31, 32, 45, 46]}],
                     'other_boxes': []}
            image['occlusion_geometry'] = choose_geometry(image)
            images.append(image)
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            if ident == 17000:
                logits[0, 281] = 8.
            label = f'screen:{ident}'
            data.reserve(label)
            row = data.save_forward_bytes(label, {'logits': logits})
            scored = screen_score(_bits(logits), category)
            data.finish_forward(row, {'kind': 'screen', 'image_id': ident}, scored)
            observed.append({'image_id': ident, **scored})
    image = images[0]
    clean = torch.zeros((1, 1000), dtype=torch.float32)
    clean[0, 281] = 8.
    clean_logp = log_group_probability(clean, 17)
    masks, mask_scores, mask_hashes, drops = {'clean': _bits(pixels)}, {}, {}, {}
    for site, value in (('animal', 6.), ('background', 8.)):
        masks[site] = _bits(mask_pixels(pixels, image['occlusion_geometry'][site]['row_col']))
        logits = clean.clone()
        logits[0, 281] = value
        label = f'response:{image["image_id"]}:{site}_mask'
        data.reserve(label)
        state = torch.zeros((1, 197, 192), dtype=torch.float32)
        row = data.save_forward_bytes(label, {'logits': logits, 'early': state, 'late': state})
        scored, _ = score(logits)
        steps = [{'name': name, 'operation': 'capture', 'module_path': path, 'order': index,
                  'invocation': 0, 'batch_index': None, 'token_index': None,
                  'output_shape': [1, 197, 192], 'selected_shape': [1, 197, 192],
                  'dtype': 'torch.float32', 'source_capture': None, 'preceding_same_site_actions': []}
                 for index, (name, path) in enumerate((('early', 'vit.encoder.layer.2'),
                                                        ('late', 'vit.encoder.layer.11')))]
        data.finish_forward(row, {'steps': steps, 'capture_bytes': 302592, 'source_bytes': 0}, scored)
        mask_scores[site] = scored
        mask_hashes[site] = hashlib.sha256(_bits(logits)).hexdigest()
        drops[site] = clean_logp - log_group_probability(logits, 17)
    input_hashes = {name: data.write(f'screen-input-17000-{name}.bin', raw) for name, raw in masks.items()}
    masked = {17000: {'animal_drop': drops['animal'], 'background_drop': drops['background'],
                      'input_sha256': input_hashes, 'logit_sha256': mask_hashes, 'scores': mask_scores}}
    selected, counts, rows = select_response_screen(images, observed, masked)
    result = {'schema': 'vision-cls-occlusion-v1-screen-result', 'manifest_sha256': 'a' * 64,
              'screen': observed, 'masked': {'17000': masked[17000]}, 'selected_ids':
              [row['image_id'] for row in selected], 'counts': counts, 'rows': rows}
    data.json('screen-result.json', result)
    checked = verify_saved_response_screen(images, data.root, expected_manifest_sha256='a' * 64)
    assert [row['image_id'] for row in checked['selected']] == [17000]
    assert checked['inputs'][17000] == masks
    assert checked['logits'][17000]['clean'] == _bits(clean)

    row_path = data.root / 'forward-065.json'
    original_row = row_path.read_bytes()
    for key, bad_value in (('operation', 'patch'), ('module_path', 'vit.encoder.layer.3')):
        altered = json.loads(original_row)
        altered['metadata']['steps'][0][key] = bad_value
        row_path.write_text(json.dumps(altered), encoding='utf-8')
        with pytest.raises(ValueError, match='metadata'):
            verify_saved_response_screen(images, data.root, expected_manifest_sha256='a' * 64)
    row_path.write_bytes(original_row)

    path = data.root / 'screen-input-17000-animal.bin'
    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match='pixel|input'):
        verify_saved_response_screen(images, data.root, expected_manifest_sha256='a' * 64)
    rehashed = copy.deepcopy(result)
    rehashed['masked']['17000']['input_sha256']['animal'] = hashlib.sha256(corrupted).hexdigest()
    (data.root / 'screen-result.json').write_text(json.dumps(rehashed), encoding='utf-8')
    with pytest.raises(ValueError, match='masked pixel'):
        verify_saved_response_screen(images, data.root, expected_manifest_sha256='a' * 64)
    path.write_bytes(masks['animal'])

    tampered = copy.deepcopy(result)
    tampered['masked']['17000']['animal_drop'] = 0.
    (data.root / 'screen-result.json').write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError, match='mask|drop|score'):
        verify_saved_response_screen(images, data.root, expected_manifest_sha256='a' * 64)
