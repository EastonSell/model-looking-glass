"""Reconstruct the closed CLS/occlusion response screen from saved bytes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .vision_coco_cls_occlusion_study import select_response_screen
from .vision_coco_occlusion_cases import choose_geometry
from .vision_coco_occlusion_study import log_group_probability, mask_pixels
from .vision_coco_recognized_handoff import validate_candidates
from .vision_coco_recognized_screen import screen_score
from .vision_coco_study import EARLY, LATE, _bits, _check_metadata, score


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, size):
    if path.stat().st_size != size:
        raise ValueError('saved input or forward length differs')
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError('saved input or forward changed during read')
    return raw


def _logits(raw):
    values = np.frombuffer(raw, dtype='<f4')
    if len(values) != 1000 or not np.isfinite(values).all():
        raise ValueError('nonfinite or malformed saved logits')
    return torch.from_numpy(values.copy()).reshape(1, 1000)


def verify_saved_response_screen(images, data, *, expected_result=None, expected_manifest_sha256=None):
    """Check every screen row and reconstruct selected inputs and logits.

    The caller binds the complete public archive, terminal, runtime and source.
    This check assumes quiescent saved files and is not execution attestation.
    """
    validate_candidates(images)
    if any(choose_geometry(image) != image.get('occlusion_geometry') for image in images):
        raise ValueError('saved geometry differs')
    data = Path(data)
    result_raw = (data / 'screen-result.json').read_bytes()
    if expected_result is not None and result_raw != expected_result:
        raise ValueError('screen result differs from published bytes')
    result = json.loads(result_raw)
    if (result.get('schema') != 'vision-cls-occlusion-v1-screen-result' or
            (expected_manifest_sha256 is not None and result.get('manifest_sha256') != expected_manifest_sha256)):
        raise ValueError('screen schema or manifest differs')
    clean_by_id, observed = {}, []
    clean_layout = {'logits': {'shape': [1, 1000], 'offset': 0, 'bytes': 4000}}
    for index, image in enumerate(images, 1):
        row = json.loads((data / f'forward-{index:03d}.json').read_bytes())
        raw = _read(data / f'forward-{index:03d}.bin', 4000)
        ident = image['image_id']
        if (row.get('submission') != index or type(row.get('submission')) is not int or
                row.get('label') != f'screen:{ident}' or row.get('file') != f'forward-{index:03d}.bin' or
                row.get('dtype') != 'little-endian-float32' or row.get('layout') != clean_layout or
                row.get('sha256') != _sha(raw) or row.get('metadata') != {'kind': 'screen', 'image_id': ident}):
            raise ValueError('saved clean screen forward differs')
        assessed = screen_score(raw, image['category_id'])
        if row.get('score') != assessed:
            raise ValueError('saved clean screen score differs')
        observed.append({'image_id': ident, **assessed})
        clean_by_id[ident] = raw
    if result.get('screen') != observed or type(result.get('masked')) is not dict:
        raise ValueError('saved clean screen or mask inventory differs')

    masked, inputs, logits_by_id = {}, {}, {}
    index = 64
    layout = {'logits': {'shape': [1, 1000], 'offset': 0, 'bytes': 4000},
              'early': {'shape': [1, 197, 192], 'offset': 4000, 'bytes': 151296},
              'late': {'shape': [1, 197, 192], 'offset': 155296, 'bytes': 151296}}
    for image, baseline in zip(images, observed):
        if not baseline['recognized'] or image.get('occlusion_geometry') is None:
            continue
        ident, saved = image['image_id'], result['masked'].get(str(image['image_id']))
        if type(saved) is not dict:
            raise ValueError('missing saved mask record')
        prefix = f'screen-input-{ident}-'
        values = {name: _read(data / (prefix + name + '.bin'), 602112)
                  for name in ('clean', 'animal', 'background')}
        if (_sha(values['clean']) != image['pixel_sha256'] or
                saved.get('input_sha256') != {name: _sha(raw) for name, raw in values.items()}):
            raise ValueError('saved input pixel hash differs')
        pixels = torch.from_numpy(np.frombuffer(values['clean'], dtype='<f4').copy()).reshape(1, 3, 224, 224)
        clean_logits = _logits(clean_by_id[ident])
        clean_logp = log_group_probability(clean_logits, image['category_id'])
        pair_logits, drops, scores = {'clean': clean_by_id[ident]}, {}, {}
        for site in ('animal', 'background'):
            if _bits(mask_pixels(pixels, image['occlusion_geometry'][site]['row_col'])) != values[site]:
                raise ValueError('saved masked pixel bits differ')
            index += 1
            row = json.loads((data / f'forward-{index:03d}.json').read_bytes())
            raw = _read(data / f'forward-{index:03d}.bin', 306592)
            if (type(row.get('submission')) is not int or row['submission'] != index or
                    row.get('label') != f'response:{ident}:{site}_mask' or
                    row.get('file') != f'forward-{index:03d}.bin' or
                    row.get('dtype') != 'little-endian-float32' or row.get('layout') != layout or
                    row.get('sha256') != _sha(raw) or
                    not np.isfinite(np.frombuffer(raw, dtype='<f4')).all()):
                raise ValueError('saved masked forward differs')
            metadata = row.get('metadata')
            if (type(metadata) is not dict or type(metadata.get('capture_bytes')) is not int or
                    metadata['capture_bytes'] != 302592 or type(metadata.get('source_bytes')) is not int or
                    metadata['source_bytes'] != 0):
                raise ValueError('saved mask metadata differs')
            try:
                _check_metadata(metadata, [{'name': 'early', 'operation': 'capture', 'module_path': EARLY},
                                           {'name': 'late', 'operation': 'capture', 'module_path': LATE}])
                if any(step.get('selected_shape') != [1, 197, 192] or
                       step.get('source_capture') is not None or step.get('preceding_same_site_actions') != []
                       for step in metadata['steps']):
                    raise ValueError('capture-only metadata differs')
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise ValueError('saved mask metadata differs') from error
            vector = _logits(raw[:4000])
            calculated, _ = score(vector)
            saved_score = row.get('score')
            if (type(saved_score) is not dict or
                    {key: value for key, value in saved_score.items() if key != 'top1_label'} != calculated):
                raise ValueError('saved mask score differs')
            drops[site] = clean_logp - log_group_probability(vector, image['category_id'])
            scores[site] = saved_score
            pair_logits[site] = raw[:4000]
        rebuilt = {'animal_drop': drops['animal'], 'background_drop': drops['background'],
                   'input_sha256': {name: _sha(raw) for name, raw in values.items()},
                   'logit_sha256': {site: _sha(pair_logits[site]) for site in ('animal', 'background')},
                   'scores': scores}
        if saved != rebuilt:
            raise ValueError('saved mask drops or scores differ')
        masked[ident], inputs[ident], logits_by_id[ident] = rebuilt, values, pair_logits
    if set(result['masked']) != {str(ident) for ident in masked}:
        raise ValueError('saved mask inventory differs')
    selected, counts, rows = select_response_screen(images, observed, masked)
    if (result.get('selected_ids') != [row['image_id'] for row in selected] or
            result.get('counts') != counts or result.get('rows') != rows):
        raise ValueError('response screen selection differs')
    return {'selected': selected, 'counts': counts, 'inputs': inputs, 'logits': logits_by_id,
            'forwards': index, 'result_sha256': _sha(result_raw)}
