"""Independently check saved FLAN-T5 smoke owner records without model imports.

This is an owner-record consistency check, not a launch gate or an execution,
source chronology, controller-exit, dependency-origin, or cleanup attestation.
The caller must independently bind the supervisor receipt and frozen source.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


_MIB = 1024 ** 2
_GIB = 1024 ** 3
_MAX_OWNER = 512 * _MIB
_CASE = 'e00d77cf45885b10e19fe689b8cdb13d2d36dd038c7ab2c1701f9367eb2fe9fc'
_ASSET = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
_REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
_LABELS = ('capture-a', 'capture-b', 'repeat-a', 'self-patch', 'donor-patch', 'null-a')
_A = [30355, 15, 1566, 12, 2968, 10, 484, 1]
_B = [30355, 15, 1566, 12, 2968, 10, 629, 1]
_FILES = frozenset(('config.json', 'generation_config.json', 'model.safetensors',
                    'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json'))
_SOURCES = frozenset((
    'scripts/flan_t5_gpu_bootstrap.py', 'scripts/v2_source_guards.py',
    'docs/FLAN_T5_GPU_DEVELOPMENT_SMOKE_V1.md', 'docs/FLAN_T5_PUBLIC_CASES_V1.json',
    'docs/FLAN_T5_DEVICE_OBSERVATION_V1.md', 'docs/FLAN_T5_DEVICE_OWNER_CUSTODY_DRAFT.md',
    'looking_glass/__init__.py', 'looking_glass/flan_t5_device_owner.py',
    'looking_glass/flan_t5_device_smoke.py', 'looking_glass/flan_t5_device_controller.py',
    'looking_glass/flan_t5_device_observation.py', 'looking_glass/flan_t5_feasibility.py',
    'looking_glass/flan_t5_equal_dose.py', 'looking_glass/intervention_plan.py',
    'looking_glass/interventions.py', 'looking_glass/model_ownership.py',
    'looking_glass/device_identity.py', 'looking_glass/device_lease.py',
    'looking_glass/memory_budget.py', 'looking_glass/supervisor.py',
    'looking_glass/gpu_oneshot.py', 'looking_glass/bert_gpu_calibration.py',
    'looking_glass/bert_gpu_evidence.py', 'looking_glass/bert_gpu_family_owner.py'))
_UNSET = object()


def _fail(message):
    raise ValueError(message)


def _same(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value, length):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _identities(asset_hash, source_paths):
    overridden = asset_hash is not _UNSET or source_paths is not _UNSET
    asset_hash = _ASSET if asset_hash is _UNSET else asset_hash
    source_paths = _SOURCES if source_paths is _UNSET else source_paths
    if (not _hex(asset_hash, 64) or type(source_paths) not in (set, frozenset)
            or not 1 <= len(source_paths) <= 128
            or any(type(path) is not str or re.fullmatch('[A-Za-z0-9_./-]+', path) is None
                   or path.startswith('/') or any(part in ('', '.', '..') for part in path.split('/'))
                   for path in source_paths)):
        _fail('invalid asset or source inventory identity')
    return asset_hash, frozenset(source_paths), 'synthetic-overrides' if overridden else 'pinned'


def _plain_directory(path):
    path = Path(path)
    if not path.is_absolute():
        _fail('an absolute plain directory is required')
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        info = item.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            _fail('linked or invalid directory')
    return path


def _read(path, limit):
    path = Path(path)
    _plain_directory(path.parent)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400
            or before.st_size > limit):
        _fail('expected a bounded plain file')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if len(raw) != before.st_size or identity(before) != identity(opened) or identity(before) != identity(after):
        _fail('saved file changed while reading')
    return raw


def _decode_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail('duplicate JSON key')
            result[key] = value
        return result
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            _fail('nonfinite JSON number')
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_float=number, parse_constant=lambda _x: _fail('nonfinite JSON constant'))


def _json(path):
    return _decode_json(_read(path, 2 * _MIB))


def _file_digest(path, size):
    path = Path(path)
    _plain_directory(path.parent)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400
            or before.st_size != size or not 0 < size <= _MAX_OWNER):
        _fail('staged asset is not a bounded plain file')
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        while True:
            block = stream.read(_MIB)
            if not block:
                break
            count += len(block)
            if count > size:
                _fail('staged asset grew during hashing')
            digest.update(block)
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if count != size or identity(before) != identity(opened) or identity(before) != identity(after):
        _fail('staged asset changed while hashing')
    return digest.hexdigest()


def _entries(directory, limit):
    directory = _plain_directory(directory)
    names = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if len(names) >= limit:
                _fail('directory exceeds its entry bound')
            names.add(entry.name)
    return names


def _metadata(label):
    action = 'capture' if label in _LABELS[:3] else 'null' if label == 'null-a' else 'patch'
    names = ('before',) if action == 'capture' else ('before', 'action', 'after')
    steps = []
    for index, name in enumerate(names):
        selected = name == 'action'
        steps.append({'name': name, 'operation': action if selected else 'capture', 'order': index,
            'module_path': 'encoder.final_layer_norm', 'module_aliases': ['encoder.final_layer_norm'],
            'call_site_known': False, 'forward_index': 0, 'invocation': 0,
            'batch_index': 0 if selected else None, 'token_index': 6 if selected else None,
            'output_selector': [], 'output_shape': [1, 8, 512],
            'selected_shape': [512] if selected else [1, 8, 512], 'dtype': 'torch.float32',
            'device': 'cuda:0', 'source_capture': None, 'preceding_same_site_actions': list(names[:index])})
    return {'steps': steps, 'capture_bytes': 8 * 512 * 4 * (1 if action == 'capture' else 2),
            'source_bytes': 512 * 4 if action == 'patch' else 0}


def _memory(snapshot, total, cap, active, previous):
    names = {'device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
             'peak_allocated_bytes', 'peak_reserved_bytes'}
    if (type(snapshot) is not dict or set(snapshot) != names
            or any(type(snapshot[k]) is not int or snapshot[k] < 0 for k in names)):
        _fail('invalid memory counters')
    free, actual = snapshot['device_free_bytes'], snapshot['device_total_bytes']
    allocated, reserved = snapshot['allocated_bytes'], snapshot['reserved_bytes']
    peak_a, peak_r = snapshot['peak_allocated_bytes'], snapshot['peak_reserved_bytes']
    reserve = max(_GIB, (15 * total + 99) // 100)
    commitment = cap + 256 * _MIB + active * 512 * 1024
    if (actual != total or free > total or not allocated <= reserved <= total
            or not allocated <= peak_a <= peak_r <= total or peak_r < reserved
            or peak_a < previous[0] or peak_r < previous[1]
            or free < reserve + commitment or reserved + reserve + commitment > total
            or free < reserve + 256 * _MIB or peak_r > cap - 256 * _MIB):
        _fail('memory counters or headroom differ')
    return peak_a, peak_r


def _tensor(directory, label, name, item):
    shape = [1, 2, 32128] if name == 'logits' else [1, 8, 512]
    count = 2 * 32128 if name == 'logits' else 8 * 512
    expected = {'file': f'{label}-{name}.bin', 'bytes': count * 4, 'shape': shape,
                'dtype': 'float32-little-endian-bits'}
    if (type(item) is not dict or set(item) != {*expected, 'sha256'}
            or any(not _same(item.get(k), v) for k, v in expected.items()) or not _hex(item['sha256'], 64)):
        _fail('tensor identity differs')
    raw = _read(directory / expected['file'], count * 4)
    if len(raw) != count * 4 or _digest(raw) != item['sha256']:
        _fail('tensor bytes differ')
    if any((bits & 0x7f800000) == 0x7f800000 for (bits,) in struct.iter_unpack('<I', raw)):
        _fail('nonfinite tensor encoding')
    return raw


def audit_owner(directory, supervisor, source_commit, *, asset_hash=_UNSET, source_paths=_UNSET):
    """Check owner bytes; overridable asset/source identities support synthetic tests only."""
    asset_hash, source_paths, identity_scope = _identities(asset_hash, source_paths)
    if not _hex(source_commit, 40):
        _fail('an absolute owner directory and exact source commit are required')
    directory = _plain_directory(directory)
    if (type(supervisor) is not dict or set(supervisor) !=
            {'schema', 'attempt_id', 'run_id', 'pid', 'exit_code', 'clean', 'timed_out', 'reason'}
            or supervisor['schema'] != 'flan-t5-gpu-supervision-v1'
            or not _hex(supervisor['attempt_id'], 32) or not _hex(supervisor['run_id'], 32)
            or type(supervisor['pid']) is not int or supervisor['pid'] <= 0
            or type(supervisor['exit_code']) is not int or supervisor['exit_code'] != 0
            or supervisor['clean'] is not True or supervisor['timed_out'] is not False
            or supervisor['reason'] != 'clean device teardown'):
        _fail('supervisor receipt differs')
    run_id, pid = supervisor['run_id'], supervisor['pid']
    start = {'schema': 'flan-t5-gpu-smoke-owner-v1', 'status': 'started', 'run_id': run_id,
             'pid': pid, 'source_commit': source_commit, 'case_manifest_sha256': _CASE,
             'asset_manifest_sha256': asset_hash, 'max_forwards': 6}
    if not _same(_json(directory / 'owner-started.json'), start):
        _fail('owner start differs')
    final = _json(directory / 'owner-final.json')
    versions = {'python': '3.11.15', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
                'transformers': '4.57.3', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.4.6'}
    keys = {'schema', 'status', 'run_id', 'pid', 'source_commit', 'source_sha256',
            'case_manifest_sha256', 'asset_manifest_sha256', 'revision', 'model_state_sha256',
            'runtime_versions', 'dependency_entry_files', 'device_uuid', 'device_name', 'device_total_bytes',
            'allocator_cap_bytes', 'cleanup', 'memory_record_count', 'forward_count', 'controls',
            'record_bytes_before_final', 'staged_assets'}
    identity = {'schema': start['schema'], 'status': 'owner-clean-pending-supervisor', 'run_id': run_id,
                'pid': pid, 'source_commit': source_commit, 'case_manifest_sha256': _CASE,
                'asset_manifest_sha256': asset_hash, 'revision': _REVISION, 'runtime_versions': versions,
                'memory_record_count': 37, 'forward_count': 6,
                'controls': {'forwards': 6, 'repeat_exact': True, 'self_patch_exact': True, 'null_exact': True,
                             'donor_differs': True, 'donor_changes_logits': True}}
    if (type(final) is not dict or set(final) != keys
            or any(not _same(final.get(k), v) for k, v in identity.items())
            or not _hex(final['model_state_sha256'], 64)
            or type(final['device_uuid']) is not str
            or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', final['device_uuid']) is None
            or type(final['device_name']) is not str or not final['device_name']
            or type(final['device_total_bytes']) is not int or not 7 * _GIB <= final['device_total_bytes'] <= 9 * _GIB
            or type(final['allocator_cap_bytes']) is not int
            or final['allocator_cap_bytes'] != final['device_total_bytes'] // 4
            or type(final['record_bytes_before_final']) is not int
            or not 0 < final['record_bytes_before_final'] <= 16 * _MIB):
        _fail('owner identity or runtime differs')
    source = final['source_sha256']
    if (type(source) is not dict or set(source) != source_paths
            or any(type(v) is not dict or set(v) != {'size', 'sha256'} or type(v['size']) is not int
                   or not 0 < v['size'] <= 4 * _MIB or not _hex(v['sha256'], 64) for v in source.values())):
        _fail('source metadata differs')
    dependencies = final['dependency_entry_files']
    names = {'torch', 'transformers', 'numpy', 'tokenizers', 'safetensors', 'huggingface_hub'}
    if (type(dependencies) is not dict or set(dependencies) != names
            or any(type(v) is not dict or set(v) != {'path', 'size', 'sha256'}
                   or v['path'] != k + '/__init__.py' or type(v['size']) is not int
                   or not 0 < v['size'] <= 4 * _MIB or not _hex(v['sha256'], 64)
                   for k, v in dependencies.items())):
        _fail('dependency metadata differs')

    staged = directory / 'staged-assets'
    if _entries(staged, 8) != _FILES | {'asset-manifest.json'}:
        _fail('staged checkpoint inventory differs')
    manifest_raw = _read(staged / 'asset-manifest.json', _MIB)
    if _digest(manifest_raw) != asset_hash:
        _fail('staged checkpoint manifest differs')
    manifest = _decode_json(manifest_raw)
    if (type(manifest) is not dict or manifest.get('schema') != 'mlg-flant5-small-assets-v1'
            or manifest.get('repo') != 'google/flan-t5-small' or manifest.get('revision') != _REVISION
            or type(manifest.get('files')) is not dict or set(manifest['files']) != _FILES):
        _fail('staged checkpoint metadata differs')
    staged_bytes = 0
    for name, item in manifest['files'].items():
        if (type(item) is not dict or set(item) != {'bytes', 'sha256', 'git_blob_sha1'}
                or type(item['bytes']) is not int or not 0 < item['bytes'] <= _MAX_OWNER
                or not _hex(item['sha256'], 64) or not _hex(item['git_blob_sha1'], 40)
                or _file_digest(staged / name, item['bytes']) != item['sha256']):
            _fail('staged checkpoint asset differs')
        staged_bytes += item['bytes']
        if staged_bytes > _MAX_OWNER:
            _fail('staged checkpoint exceeds its bound')
    staged_account = {'file_count': len(_FILES), 'payload_bytes': staged_bytes,
                      'manifest_bytes': len(manifest_raw), 'total_bytes': staged_bytes + len(manifest_raw)}
    if not _same(final['staged_assets'], staged_account):
        _fail('staged checkpoint accounting differs')

    stages = [('startup', 'before_model_load', 0), ('startup', 'after_admission', 0),
              ('model-load', 'before_model_copy', 0), ('model-load', 'after_model_copy', 0),
              ('model-load', 'after_initial_state', 0)]
    for index, label in enumerate(_LABELS):
        previous = min(index, 2)
        stages.extend((label, name, previous if name == 'before_observation' else previous + 1)
                      for name in ('before_observation', 'output_live', 'captures_live',
                                   'device_refs_disposed', 'after_observation'))
    stages.extend((('cleanup', 'cleanup_before_workspace_clear', 0), ('cleanup', 'after_cleanup', 0)))
    peak = (0, 0)
    for index, (phase, stage, active) in enumerate(stages):
        record = _json(directory / f'memory-{index:03d}.json')
        header = {'schema': 'flan-t5-gpu-memory-v1', 'run_id': run_id, 'index': index,
                  'phase': phase, 'stage': stage, 'peak_scope': 'cumulative-since-pre-model-load-reset'}
        if (type(record) is not dict or set(record) != {*header, 'memory'}
                or any(not _same(record.get(k), v) for k, v in header.items())):
            _fail('memory stage identity differs')
        peak = _memory(record['memory'], final['device_total_bytes'], final['allocator_cap_bytes'], active, peak)
    if (not _same(record['memory'], final['cleanup'])
            or record['memory']['allocated_bytes'] != 0 or record['memory']['reserved_bytes'] != 0):
        _fail('cleanup counters differ')

    expected = {'owner-started.json', 'owner-final.json', 'staged-assets'}
    expected.update(f'memory-{i:03d}.json' for i in range(len(stages)))
    observations = {}
    for index, label in enumerate(_LABELS):
        name = f'forward-{index:03d}-started.json'
        expected.add(name)
        if not _same(_json(directory / name), {'schema': 'flan-t5-gpu-forward-v1',
                    'run_id': run_id, 'index': index, 'label': label}):
            _fail('forward order differs')
        record = _json(directory / (label + '.json'))
        expected.add(label + '.json')
        header = {'schema': 'flan-t5-gpu-observation-v1', 'run_id': run_id, 'label': label,
                  'source_ids': _B if label == 'capture-b' else _A, 'source_mask': [1] * 8,
                  'decoder_ids': [0, 4675], 'decoder_mask': [1, 1], 'metadata': _metadata(label)}
        if (type(record) is not dict or set(record) != {*header, 'tensors'}
                or any(not _same(record.get(k), v) for k, v in header.items())):
            _fail('observation identity or metadata differs')
        names = ('logits', 'state', 'before') if index < 3 else ('logits', 'state', 'before', 'after')
        if type(record['tensors']) is not dict or set(record['tensors']) != set(names):
            _fail('tensor inventory differs')
        observations[label] = {name: _tensor(directory, label, name, record['tensors'][name]) for name in names}
        expected.update(f'{label}-{name}.bin' for name in names)
        if observations[label]['state'] != observations[label].get('after', observations[label]['before']):
            _fail('encoder state and witness differ')
    a, b = observations['capture-a'], observations['capture-b']
    first, last = 6 * 512 * 4, 7 * 512 * 4
    if a['before'][first:last] == b['before'][first:last]:
        _fail('donor vector is identical')
    for label in ('repeat-a', 'self-patch', 'null-a'):
        item = observations[label]
        if any(item[name] != a[name] for name in ('logits', 'state', 'before')):
            _fail('exact control differs')
        if 'after' in item and item['after'] != a['before']:
            _fail('exact control witness differs')
    donor = observations['donor-patch']
    if (donor['before'] != a['before'] or donor['after'][first:last] != b['before'][first:last]
            or donor['after'][:first] != a['before'][:first] or donor['after'][last:] != a['before'][last:]
            or donor['logits'] == a['logits']):
        _fail('donor patch did not land or change logits')
    if _entries(directory, 256) != expected:
        _fail('owner directory contains missing or extra records')
    record_bytes = sum((directory / name).stat().st_size for name in expected
                       if name not in ('owner-final.json', 'staged-assets'))
    if record_bytes != final['record_bytes_before_final']:
        _fail('owner record-byte accounting differs')
    return {'status': 'owner-records-consistent-not-approved', 'identity_scope': identity_scope,
            'run_id': run_id, 'pid': pid,
            'source_commit': source_commit, 'forwards': len(observations), 'memory_stages': len(stages),
            'limits': ['inputs must remain quiescent throughout the audit',
                       'supervisor receipt not independently witnessed',
                       'source and dependency bytes not independently verified',
                       'model state stability and unrecorded runtime settings are owner observations',
                       'CUDA synchronization is an owner/source observation',
                       'controller exit and physical cleanup not independently observed',
                       'saved records are not execution attestation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner_directory', type=Path)
    parser.add_argument('supervision_record', type=Path)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    result = audit_owner(args.owner_directory, _json(args.supervision_record), args.source_commit)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
