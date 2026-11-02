"""Bind exposed discovery inputs to an independently audited prior run."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .vision_coco_cls_occlusion_cases import PRIOR_OCCLUSION_SHA256
from .vision_coco_occlusion_cases import choose_geometry
from .vision_coco_occlusion_study import mask_pixels
from .vision_coco_study import _bits


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_prior_discovery(images, data, audit_raw, *, expected_audit_sha256, expected_terminal_sha256):
    """Return the original input and logit bytes for the fixed exposed cases.

    The production caller fixes both digests; parameters allow fabricated tests.
    The independent audit binds the original start, terminal and forward rows.
    """
    if _sha(audit_raw) != expected_audit_sha256:
        raise ValueError('prior audit differs')
    audit = json.loads(audit_raw)
    result = audit.get('result', {})
    if (audit.get('schema') != 'vision-occlusion-independent-intervention-audit-v1' or
            audit.get('synthetic') is not False or result.get('mode') != 'evaluation' or
            result.get('manifest_sha256') != PRIOR_OCCLUSION_SHA256 or
            result.get('phase_terminal_sha256') != expected_terminal_sha256 or
            type(result.get('forwards')) is not int or type(result.get('forward_hashes')) is not list or
            len(result['forward_hashes']) != result['forwards']):
        raise ValueError('prior audit identity differs')
    data = Path(data)
    start_raw, terminal_raw = (data / 'started.json').read_bytes(), (data / 'terminal.json').read_bytes()
    if (_sha(start_raw) != result.get('phase_start_sha256') or
            _sha(terminal_raw) != expected_terminal_sha256):
        raise ValueError('prior start or terminal differs')
    terminal = json.loads(terminal_raw)
    if terminal.get('mode') != 'evaluation' or terminal.get('status') != 'complete':
        raise ValueError('prior terminal status differs')
    cases = terminal.get('cases')
    if type(cases) is not list or len({row.get('image_id') for row in cases}) != len(cases):
        raise ValueError('prior cases differ')
    case_by_id = {row['image_id']: row for row in cases}
    ids = [image['image_id'] for image in images]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate discovery image')
    inputs, needed, logits = {}, {}, {}
    for image in images:
        ident = image['image_id']
        geometry = choose_geometry(image)
        case = case_by_id.get(ident)
        if (geometry is None or geometry != image.get('occlusion_geometry') or type(case) is not dict or
                case.get('category_id') != image['category_id'] or
                json.loads((data / f'case-{ident}.json').read_bytes()) != case):
            raise ValueError('prior discovery case differs')
        values = {}
        for name in ('clean', 'animal', 'background'):
            path = data / f'input-{ident}-{name}.bin'
            if path.stat().st_size != 602112:
                raise ValueError('prior input length differs')
            raw = path.read_bytes()
            if len(raw) != 602112 or _sha(raw) != case.get('input_sha256', {}).get(name):
                raise ValueError('prior input hash differs')
            values[name] = raw
            label = f'case-{ident}:' + ('clean' if name == 'clean' else name + '_mask')
            needed[label] = (ident, name)
        if _sha(values['clean']) != image['pixel_sha256']:
            raise ValueError('prior input pixel identity differs')
        clean = torch.from_numpy(np.frombuffer(values['clean'], dtype='<f4').copy()).reshape(1, 3, 224, 224)
        for site in ('animal', 'background'):
            if _bits(mask_pixels(clean, geometry[site]['row_col'])) != values[site]:
                raise ValueError('prior input mask bits differ')
        inputs[ident], logits[ident] = values, {}
    for index, expected in enumerate(result['forward_hashes'], 1):
        if type(expected.get('submission')) is not int or expected['submission'] != index:
            raise ValueError('prior forward ordering differs')
        metadata_raw = (data / f'forward-{index:03d}.json').read_bytes()
        if _sha(metadata_raw) != expected.get('json_sha256'):
            raise ValueError('prior forward record differs')
        row = json.loads(metadata_raw)
        if row.get('label') not in needed:
            continue
        ident, name = needed.pop(row['label'])
        path = data / f'forward-{index:03d}.bin'
        if not 4000 <= path.stat().st_size <= 1_000_000:
            raise ValueError('prior forward size differs')
        raw = path.read_bytes()
        if (len(raw) != path.stat().st_size or _sha(raw) != expected.get('binary_sha256') or
                row.get('sha256') != _sha(raw) or row.get('submission') != index or
                row.get('file') != f'forward-{index:03d}.bin' or row.get('dtype') != 'little-endian-float32' or
                row.get('layout', {}).get('logits') != {'shape': [1, 1000], 'offset': 0, 'bytes': 4000} or
                not np.isfinite(np.frombuffer(raw[:4000], dtype='<f4')).all()):
            raise ValueError('prior forward payload differs')
        logits[ident][name] = raw[:4000]
    if needed or any(set(row) != {'clean', 'animal', 'background'} for row in logits.values()):
        raise ValueError('prior discovery forward inventory differs')
    return {'inputs': inputs, 'logits': logits, 'audit_sha256': expected_audit_sha256,
            'terminal_sha256': expected_terminal_sha256}
