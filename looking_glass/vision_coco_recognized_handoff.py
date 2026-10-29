"""Reconcile the closed v4 screening journal before any patch is submitted."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from .vision_coco_recognized_screen import screen_score, select_pairs

MODEL_STATE_SHA256 = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def validate_candidates(images):
    """Validate the entire manifest before the first v4 model forward."""
    if (type(images) is not list or len(images) != 64 or
            [(r['category_id'], r['rank']) for r in images] !=
            [(category, rank) for category in (17, 18) for rank in range(32)]):
        raise ValueError('screen candidate category or rank differs')
    identities, jpegs, pixels = set(), set(), set()
    for row in images:
        image_id, rank = row.get('image_id'), row.get('rank')
        if (type(image_id) is not int or image_id in identities or type(rank) is not int or
                row.get('role') != ('donor' if rank % 4 == 0 else 'target') or
                row.get('file_name') != f'{image_id:012d}.jpg' or
                type(row.get('jpeg_bytes')) is not int or not 0 < row['jpeg_bytes'] <= 5_000_000 or
                not all(type(row.get(k)) is str and re.fullmatch('[0-9a-f]{64}', row[k])
                        for k in ('jpeg_sha256', 'pixel_sha256')) or
                row['jpeg_sha256'] in jpegs or row['pixel_sha256'] in pixels):
            raise ValueError('screen candidate identity, role or image hash differs')
        identities.add(image_id)
        jpegs.add(row['jpeg_sha256'])
        pixels.add(row['pixel_sha256'])
        sites, coordinates = row.get('sites'), []
        if type(sites) is not list or not sites:
            raise ValueError('screen candidate has no qualified sites')
        for site in sites:
            coordinate, counts = site.get('row_col'), site.get('token_mask_pixels')
            if (type(coordinate) is not list or len(coordinate) != 2 or
                    any(type(v) is not int or not 0 <= v <= 12 for v in coordinate) or
                    type(counts) is not list or len(counts) != 4 or
                    any(type(v) is not int or not 0 <= v <= 256 for v in counts) or
                    type(site.get('mask_pixels')) is not int or site['mask_pixels'] != sum(counts) or
                    site['mask_pixels'] < 512):
                raise ValueError('screen site geometry or occupancy differs')
            r, c = coordinate
            expected_tokens = [1 + 14 * (r + dr) + c + dc for dr in range(2) for dc in range(2)]
            radius = (r - 6) ** 2 + (c - 6) ** 2
            tokens = site.get('token_ids')
            if (type(tokens) is not list or any(type(token) is not int for token in tokens) or
                    tokens != expected_tokens or type(site.get('squared_center_radius')) is not int or
                    site['squared_center_radius'] != radius):
                raise ValueError('screen token geometry differs')
            coordinates.append((r, c))
        if (len(coordinates) != len(set(coordinates)) or
                coordinates != sorted(coordinates, key=lambda p: ((p[0] - 6) ** 2 + (p[1] - 6) ** 2, p))):
            raise ValueError('screen site order or uniqueness differs')


def validate_runtime(runtime):
    expected = {'device': 'cpu', 'cuda_visible_devices': '', 'cuda_initialized': False,
                'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
                'numpy': '2.4.6', 'pillow': '11.3.0', 'threads': 1, 'interop_threads': 1,
                'deterministic_algorithms': True, 'float32_matmul_precision': 'highest',
                'mkldnn_enabled': True, 'default_dtype': 'torch.float32',
                'processor': 'ViTImageProcessor', 'denormals_preserved': True,
                'mkl_cbwr': 'COMPATIBLE', 'binary64_rounding': 'nearest-even'}
    if type(runtime) is not dict or canonical(runtime) != canonical(expected):
        raise ValueError('screen runtime settings differ')


def validate_pair_baselines(pairs, observed):
    """Check every selected pair, including later pairs, before interventions."""
    lookup = {row['image_id']: row for row in observed}
    if len(lookup) != len(observed):
        raise ValueError('duplicate screen image')
    for pair in pairs:
        cat, dog = lookup[pair['cat']['image_id']], lookup[pair['dog']['image_id']]
        cat_donor, dog_donor = (lookup[pair[side + '_donor']['image_id']] for side in ('cat', 'dog'))
        if (any(row['recognized'] is not True for row in (cat, dog, cat_donor, dog_donor)) or
                cat['cat_mass'] < .2 or dog['dog_mass'] < .2 or
                cat['margin'] >= 0 or dog['margin'] <= 0 or dog['margin'] - cat['margin'] < 1):
            raise ValueError('selected pair baseline contradicts the screen')


def verify_screen_archive(raw, data_dir):
    """Bind each public archive member to its retained closed-screen bytes."""
    if type(raw) is not bytes or not 0 < len(raw) <= 5_000_000:
        raise ValueError('screen archive is empty or oversized')
    data = Path(data_dir)
    expected = {'started.json', 'terminal.json', 'screen-result.json'} | {
        f'forward-{i:03d}.{ext}' for i in range(1, 65) for ext in ('json', 'bin')}
    if {item.name for item in data.iterdir()} != expected:
        raise ValueError('closed screen archive inventory differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.infolist()
        if len(info) != len(expected) or {entry.filename for entry in info} != expected:
            raise ValueError('public screen archive inventory differs')
        for entry in info:
            if entry.file_size > 1_000_000 or archive.read(entry) != (data / entry.filename).read_bytes():
                raise ValueError('public screen archive member differs')
    return hashlib.sha256(raw).hexdigest()


def screen_receipt(screen, data_dir, archive_sha256, observer_raw):
    """Bind direct-child and externally observed controller exit receipts."""
    raw = (Path(data_dir).parent / 'supervisor.json').read_bytes()
    supervisor = json.loads(raw)
    if (supervisor.get('status') != 'complete' or type(supervisor.get('exit_code')) is not int or
            supervisor['exit_code'] != 0 or supervisor.get('clean') is not True or
            supervisor.get('timed_out') is not False or
            supervisor.get('terminal_sha256') != screen['terminal_sha256']):
        raise ValueError('screen supervisor receipt differs')
    observer = json.loads(observer_raw)
    if (observer.get('schema') != 'vision-recognized-v4-external-observer' or
            observer.get('status') != 'complete' or observer.get('mode') != 'screen' or
            type(observer.get('controller_exit_code')) is not int or observer['controller_exit_code'] != 0 or
            observer.get('controller_reaped') is not True or observer.get('timed_out') is not False or
            observer.get('forced_stop') is not False or
            observer.get('supervisor_sha256') != hashlib.sha256(raw).hexdigest() or
            observer.get('terminal_sha256') != screen['terminal_sha256']):
        raise ValueError('external screen controller receipt differs')
    return {'schema': 'vision-recognized-v4-screen-custody', 'status': 'complete',
            'manifest_sha256': screen['result']['manifest_sha256'],
            'source_revision': screen['started']['source']['revision'],
            'screen_result_sha256': screen['screen_result_sha256'],
            'screen_terminal_sha256': screen['terminal_sha256'], 'screen_archive_sha256': archive_sha256,
            'private_supervisor_sha256': hashlib.sha256(raw).hexdigest(),
            'external_observer_sha256': hashlib.sha256(observer_raw).hexdigest(),
            'owner_exit_code': 0, 'owner_clean': True, 'owner_timed_out': False,
            'controller_exit_code': 0,
            'scope': 'cooperative local receipts; public source and publication order require observation'}


def verify_closed_screen(manifest, data_dir, manifest_sha256, *, expected_result=None):
    """Recompute all 64 saved scores and pairing, without running a model.

    This is saved-byte consistency. The caller binds public source and result,
    executed dependencies, process supervision, and actual chronology.
    """
    data = Path(data_dir)
    images = manifest['images']
    if manifest.get('schema') != 'vision-coco-recognized-screen-cases-v4':
        raise ValueError('screen manifest inventory differs')
    validate_candidates(images)
    expected = {'started.json', 'terminal.json', 'screen-result.json'} | {
        f'forward-{i:03d}.{ext}' for i in range(1, 65) for ext in ('json', 'bin')}
    if {path.name for path in data.iterdir()} != expected:
        raise ValueError('closed screen file inventory differs')
    started_raw, terminal_raw = (data / 'started.json').read_bytes(), (data / 'terminal.json').read_bytes()
    started, terminal = json.loads(started_raw), json.loads(terminal_raw)
    validate_runtime(started.get('runtime'))
    if (started.get('schema') != 'vision-recognized-v4-start' or started.get('mode') != 'screen' or
            started.get('manifest_sha256') != manifest_sha256 or
            terminal.get('schema') != 'vision-recognized-v4-terminal' or terminal.get('mode') != 'screen' or
            terminal.get('status') != 'complete' or terminal.get('manifest_sha256') != manifest_sha256 or
            terminal.get('source_revision') != started.get('source', {}).get('revision') or
            type(terminal.get('submissions')) is not int or terminal['submissions'] != 64 or
            type(terminal.get('saved_forwards')) is not int or terminal['saved_forwards'] != 64 or
            canonical(terminal.get('runtime')) != canonical(started.get('runtime')) or
            started.get('model_state_sha256') != MODEL_STATE_SHA256 or
            terminal.get('model_state_sha256_before_after') != MODEL_STATE_SHA256):
        raise ValueError('closed screen start/terminal differs')
    observed, raw_by_id = [], {}
    for index, image in enumerate(images, 1):
        raw = (data / f'forward-{index:03d}.bin').read_bytes()
        row = json.loads((data / f'forward-{index:03d}.json').read_bytes())
        image_id = image['image_id']
        score = screen_score(raw, image['category_id'])
        if (type(row.get('submission')) is not int or row['submission'] != index or
                row.get('label') != f'screen:{image_id}' or row.get('file') != f'forward-{index:03d}.bin' or
                row.get('sha256') != hashlib.sha256(raw).hexdigest() or row.get('dtype') != 'little-endian-float32' or
                row.get('layout') != {'logits': {'shape': [1, 1000], 'offset': 0, 'bytes': 4000}} or
                row.get('metadata') != {'kind': 'screen', 'image_id': image_id} or
                canonical(row.get('score')) != canonical(score)):
            raise ValueError('saved screen forward differs')
        observed.append({'image_id': image_id, **score})
        raw_by_id[image_id] = raw
    pairs, summary = select_pairs(images, observed)
    validate_pair_baselines(pairs, observed)
    result = {'schema': 'vision-recognized-v4-screen-result', 'manifest_sha256': manifest_sha256,
              'screen': observed, 'pairs': pairs, 'summary': summary}
    raw_result = (data / 'screen-result.json').read_bytes()
    if (canonical(result) != raw_result or terminal.get('screen_result_sha256') != hashlib.sha256(raw_result).hexdigest()
            or expected_result is not None and expected_result != raw_result):
        raise ValueError('published screen result or selection differs')
    return {'result': result, 'raw_by_id': raw_by_id, 'started': started, 'terminal': terminal,
            'screen_result_sha256': hashlib.sha256(raw_result).hexdigest(),
            'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest()}
