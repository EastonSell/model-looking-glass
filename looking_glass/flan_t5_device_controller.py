"""Disabled, supervised controller for one fixed public FLAN-T5 GPU smoke.

This is not an authorization, resource profile or execution attestation.  Its
private output remains pending an independent saved-evidence audit.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import sys
from threading import Lock
import types
from datetime import datetime, timedelta

from .bert_gpu_calibration import check_memory
from .device_identity import resolve_device_identity
from .flan_t5_device_owner import (SOURCE_PATHS, _check_bootstrap, _require_pristine_dependencies,
                                    _verify_source_tree)
from .flan_t5_device_smoke import (_A, _ASSET_HASH, _B, _CASE_HASH, _LABELS, _LIMIT,
                                    _REVISION, _fixed_rows, _staged_asset_accounting)
from .gpu_oneshot import _owner_presence_after_exit, _smi_snapshot
from .supervisor import SupervisionResult, run_supervised


# A separately reviewed source revision must explicitly enable the pilot after
# device recovery and coordination.  Importing this module never queries CUDA.
GPU_SMOKE_ENABLED = False
_GIB = 1024 ** 3
_MIB = 1024 ** 2
_GPU_CLAIM = 512 * 1024
_HEADROOM = 256 * _MIB
_DISK_BUDGET = 512 * _MIB
_DISK_MARGIN = 64 * _MIB
_TIMEOUT = 1800
_STAGED_FILES = frozenset(('asset-manifest.json', 'config.json', 'generation_config.json',
    'model.safetensors', 'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json'))
_CUSTODY = {'entered': False, 'attempt': None, 'source_guard': None, 'asset_guard': None, 'evidence_guard': None,
            'job': None, 'supervision': None, 'may_have_started': False,
            'failure': None, 'diagnostic_failure': None}
_CUSTODY_LOCK = Lock()


def _bytes(path, limit):
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400
            or before.st_size > limit):
        raise ValueError('expected a bounded regular evidence file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
        opened = os.fstat(stream.fileno())
    after = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if len(raw) != before.st_size or identity(before) != identity(opened) or identity(before) != identity(after):
        raise ValueError('evidence file changed while being read')
    return raw


def _json(path, limit=2 * _MIB):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('nonfinite JSON number')
        return parsed
    def invalid(_value):
        raise ValueError('nonfinite JSON constant')
    raw = _bytes(path, limit)
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid), raw


def _write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
    if len(raw) > 2 * _MIB:
        raise ValueError('controller record exceeds its bound')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short controller record write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _same_json(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _expected_metadata(label):
    action = ('capture' if label in ('capture-a', 'capture-b', 'repeat-a')
              else 'null' if label == 'null-a' else 'patch')
    names = ['before'] if action == 'capture' else ['before', 'action', 'after']
    operations = ['capture'] if action == 'capture' else ['capture', action, 'capture']
    steps = []
    for order, (name, operation) in enumerate(zip(names, operations)):
        selected = name == 'action'
        steps.append({'name': name, 'operation': operation, 'order': order,
            'module_path': 'encoder.final_layer_norm', 'module_aliases': ['encoder.final_layer_norm'],
            'call_site_known': False, 'forward_index': 0, 'invocation': 0,
            'batch_index': 0 if selected else None, 'token_index': 6 if selected else None,
            'output_selector': [], 'output_shape': [1, 8, 512],
            'selected_shape': [512] if selected else [1, 8, 512], 'dtype': 'torch.float32',
            'device': 'cuda:0', 'source_capture': None, 'preceding_same_site_actions': names[:order]})
    return {'steps': steps, 'capture_bytes': 8 * 512 * 4 * (1 if action == 'capture' else 2),
            'source_bytes': 512 * 4 if action == 'patch' else 0}


def _read_observation(directory, label, run_id):
    record, _ = _json(directory / (label + '.json'))
    required = {'schema', 'run_id', 'label', 'source_ids', 'source_mask', 'decoder_ids',
                'decoder_mask', 'metadata', 'tensors'}
    if (type(record) is not dict or set(record) != required
            or record['schema'] != 'flan-t5-gpu-observation-v1'
            or record['run_id'] != run_id or record['label'] != label
            or not _same_json(record['source_ids'], _B if label == 'capture-b' else _A)
            or not _same_json(record['source_mask'], [1] * 8)
            or not _same_json(record['decoder_ids'], [0, 4675])
            or not _same_json(record['decoder_mask'], [1, 1])
            or not _same_json(record['metadata'], _expected_metadata(label))):
        raise ValueError('saved observation identity or coordinates differ')
    names = ('logits', 'state', 'before') if label in _LABELS[:3] else ('logits', 'state', 'before', 'after')
    tensors = record['tensors']
    if type(tensors) is not dict or set(tensors) != set(names):
        raise ValueError('saved tensor inventory differs')
    values = {}
    for name in names:
        shape = [1, 2, 32128] if name == 'logits' else [1, 8, 512]
        size = (2 * 32128 if name == 'logits' else 8 * 512) * 4
        filename = f'{label}-{name}.bin'
        item = tensors[name]
        if (type(item) is not dict or set(item) != {'file', 'bytes', 'shape', 'dtype', 'sha256'}
                or item['file'] != filename or type(item['bytes']) is not int or item['bytes'] != size
                or not _same_json(item['shape'], shape) or item['dtype'] != 'float32-little-endian-bits'
                or type(item['sha256']) is not str or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None):
            raise ValueError('saved tensor metadata differs')
        raw = _bytes(directory / filename, size)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('saved tensor bytes differ')
        if any((int.from_bytes(raw[offset:offset + 4], 'little') & 0x7f800000) == 0x7f800000
               for offset in range(0, len(raw), 4)):
            raise ValueError('saved tensor contains a nonfinite float32 encoding')
        values[name] = raw
    if values['state'] != values.get('after', values['before']):
        raise ValueError('saved encoder state and witness differ')
    return values


def _validate_records(directory, supervision, *, source_commit, source_hashes, gpu_uuid, device_name, site_packages):
    """Validate closed private records; this is still not execution attestation."""
    if (type(supervision) is not SupervisionResult or type(supervision.pid) is not int
            or supervision.pid <= 0 or type(supervision.exit_code) is not int or supervision.exit_code != 0
            or supervision.clean is not True or supervision.timed_out is not False
            or type(supervision.run_id) is not str or re.fullmatch('[0-9a-f]{32}', supervision.run_id) is None):
        raise ValueError('supervisor has not confirmed this exact clean child')
    directory = Path(directory)
    start, _ = _json(directory / 'owner-started.json')
    final, _ = _json(directory / 'owner-final.json')
    start_expected = {'schema': 'flan-t5-gpu-smoke-owner-v1', 'status': 'started',
        'run_id': supervision.run_id, 'pid': supervision.pid, 'source_commit': source_commit,
        'case_manifest_sha256': _CASE_HASH, 'asset_manifest_sha256': _ASSET_HASH, 'max_forwards': 6}
    if not _same_json(start, start_expected) or type(final) is not dict:
        raise ValueError('owner start identity differs')
    required = {'schema', 'status', 'run_id', 'pid', 'source_commit', 'source_sha256',
        'case_manifest_sha256', 'asset_manifest_sha256', 'revision', 'model_state_sha256',
        'runtime_versions', 'dependency_entry_files', 'device_uuid', 'device_name',
        'device_total_bytes', 'allocator_cap_bytes', 'cleanup', 'memory_record_count',
        'forward_count', 'controls', 'record_bytes_before_final', 'staged_assets'}
    versions = {'python': '3.11.15', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
        'transformers': '4.57.3', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.4.6'}
    controls = {'forwards': 6, 'repeat_exact': True, 'self_patch_exact': True,
        'null_exact': True, 'donor_differs': True, 'donor_changes_logits': True}
    if (set(final) != required or final['schema'] != start_expected['schema']
            or final['status'] != 'owner-clean-pending-supervisor'
            or final['run_id'] != supervision.run_id or type(final['pid']) is not int
            or final['pid'] != supervision.pid or final['source_commit'] != source_commit
            or not _same_json(final['source_sha256'], source_hashes)
            or final['case_manifest_sha256'] != _CASE_HASH or final['asset_manifest_sha256'] != _ASSET_HASH
            or final['revision'] != _REVISION or not _same_json(final['runtime_versions'], versions)
            or type(final['model_state_sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}', final['model_state_sha256']) is None
            or final['device_uuid'] != gpu_uuid or final['device_name'] != device_name
            or type(final['device_total_bytes']) is not int
            or not 7 * _GIB <= final['device_total_bytes'] <= 9 * _GIB
            or type(final['allocator_cap_bytes']) is not int
            or final['allocator_cap_bytes'] != int(.25 * final['device_total_bytes'])
            or type(final['memory_record_count']) is not int or final['memory_record_count'] != 37
            or type(final['forward_count']) is not int or final['forward_count'] != 6
            or not _same_json(final['controls'], controls)
            or type(final['record_bytes_before_final']) is not int
            or not 0 < final['record_bytes_before_final'] <= _LIMIT):
        raise ValueError('owner final identity, runtime or controls differ')
    deps = final['dependency_entry_files']
    if (type(deps) is not dict or set(deps) != {'torch', 'transformers', 'numpy', 'tokenizers',
                                               'safetensors', 'huggingface_hub'}
            or any(type(item) is not dict or set(item) != {'path', 'size', 'sha256'}
                   or type(item['path']) is not str or not item['path']
                   or type(item['size']) is not int or not 0 < item['size'] <= 4 * _MIB
                   or type(item['sha256']) is not str or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None
                   for item in deps.values())):
        raise ValueError('dependency entry-file evidence differs')
    site = _plain_directory(site_packages)
    for name, item in deps.items():
        if item['path'] != name + '/__init__.py':
            raise ValueError('dependency entry path differs from the selected package')
        package = _plain_directory(site / name)
        if package.parent != site:
            raise ValueError('dependency package lies outside the selected site')
        data = _bytes(package / '__init__.py', 4 * _MIB)
        if len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('dependency entry bytes differ from the owner observation')
    if not _same_json(final['staged_assets'], _staged_asset_accounting(directory / 'staged-assets')):
        raise ValueError('staged checkpoint accounting differs')
    _owner_evidence_files(directory)

    records = [('startup', 'before_model_load', 0), ('startup', 'after_admission', 0),
        ('model-load', 'before_model_copy', 0), ('model-load', 'after_model_copy', 0),
        ('model-load', 'after_initial_state', 0)]
    for index, label in enumerate(_LABELS):
        previous = min(index, 2)
        for stage, active in (('before_observation', previous), ('output_live', previous + 1),
                ('captures_live', previous + 1), ('device_refs_disposed', previous + 1),
                ('after_observation', previous + 1)):
            records.append((label, stage, active))
    records.extend((('cleanup', 'cleanup_before_workspace_clear', 0), ('cleanup', 'after_cleanup', 0)))
    last = None
    peaks = (0, 0)
    for index, (phase, stage, active) in enumerate(records):
        value, _ = _json(directory / f'memory-{index:03d}.json')
        if (type(value) is not dict or set(value) != {'schema', 'run_id', 'index', 'phase', 'stage', 'memory', 'peak_scope'}
                or value['schema'] != 'flan-t5-gpu-memory-v1' or value['run_id'] != supervision.run_id
                or type(value['index']) is not int or value['index'] != index
                or value['phase'] != phase or value['stage'] != stage
                or value['peak_scope'] != 'cumulative-since-pre-model-load-reset'):
            raise ValueError('memory stage inventory differs')
        current = value['memory']
        if (type(current) is not dict or set(current) != {'device_free_bytes', 'device_total_bytes',
                'allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'}):
            raise ValueError('memory stage fields differ')
        check_memory(current, expected_total=final['device_total_bytes'], cap=final['allocator_cap_bytes'],
                     commitment=final['allocator_cap_bytes'] + _HEADROOM + active * _GPU_CLAIM)
        next_peaks = (current['peak_allocated_bytes'], current['peak_reserved_bytes'])
        if any(new < old for new, old in zip(next_peaks, peaks)):
            raise ValueError('cumulative allocator peaks decreased')
        peaks, last = next_peaks, current
    if (not _same_json(final['cleanup'], last) or last['allocated_bytes'] != 0 or last['reserved_bytes'] != 0):
        raise ValueError('owner cleanup counters differ')

    outputs = {}
    for index, label in enumerate(_LABELS):
        started, _ = _json(directory / f'forward-{index:03d}-started.json')
        if not _same_json(started, {'schema': 'flan-t5-gpu-forward-v1', 'run_id': supervision.run_id,
                                    'index': index, 'label': label}):
            raise ValueError('forward start identity or ordering differs')
        outputs[label] = _read_observation(directory, label, supervision.run_id)
    a, b = outputs['capture-a'], outputs['capture-b']
    start, end = 6 * 512 * 4, 7 * 512 * 4
    if a['before'][start:end] == b['before'][start:end]:
        raise ValueError('saved donor vector is identical')
    for label in ('repeat-a', 'self-patch', 'null-a'):
        current = outputs[label]
        if any(current[name] != a[name] for name in ('logits', 'state', 'before')):
            raise ValueError('saved exact control differs')
        if 'after' in current and current['after'] != a['before']:
            raise ValueError('saved exact control did not land')
    donor = outputs['donor-patch']
    if (donor['before'] != a['before'] or donor['after'][start:end] != b['before'][start:end]
            or donor['after'][:start] != a['before'][:start] or donor['after'][end:] != a['before'][end:]
            or donor['logits'] == a['logits']):
        raise ValueError('saved donor patch or nonvacuity differs')

    expected = {'owner-started.json', 'owner-final.json', 'staged-assets'}
    expected.update(f'memory-{index:03d}.json' for index in range(len(records)))
    expected.update(f'forward-{index:03d}-started.json' for index in range(len(_LABELS)))
    for label in _LABELS:
        expected.add(label + '.json')
        expected.update(f'{label}-{name}.bin' for name in outputs[label])
    actual_names = {item.relative_to(directory).parts[0] for item in _owner_evidence_files(directory)}
    if actual_names != expected:
        raise ValueError('owner evidence contains missing or extra files')
    record_bytes = sum((directory / name).stat().st_size for name in expected
                       if name not in ('owner-final.json', 'staged-assets'))
    if record_bytes != final['record_bytes_before_final']:
        raise ValueError('owner observation-record byte count differs')
    return {'forward_count': 6, 'memory_record_count': len(records), 'record_bytes_before_final': record_bytes,
            'staged_assets': final['staged_assets'], 'model_state_sha256': final['model_state_sha256'],
            'raw_control_checks': 'matched', 'status': 'saved-records-consistent-not-approved'}


def _plain_directory(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('an absolute directory is required')
    lexical = Path(os.path.abspath(path))
    for entry in (lexical, *lexical.parents):
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('links and reparse points are unsupported')
    if not lexical.is_dir():
        raise ValueError('directory is unavailable')
    return lexical.resolve(strict=True)


def _new_attempt(workspace, repo, path):
    workspace, repo, path = _plain_directory(workspace), _plain_directory(repo), Path(path)
    if not path.is_absolute() or path.exists():
        raise ValueError('attempt must be a new absolute directory')
    parent = _plain_directory(path.parent)
    work = _plain_directory(workspace / 'work')
    expected_root = next((p.parent for p in repo.parents if p.name == 'work'), repo)
    resolved = parent / path.name
    if (workspace != expected_root or parent != work and not parent.is_relative_to(work)
            or resolved == repo or resolved.is_relative_to(repo) or repo.is_relative_to(resolved)):
        raise ValueError('attempt and checkout must be separate within the shared workspace')
    resolved.mkdir()
    return resolved


def _guard_types(repo, hashes):
    """Execute only the rechecked bytes of the source-bound guard module."""
    relative = 'scripts/v2_source_guards.py'
    data = _bytes(Path(repo) / relative, 4 * _MIB)
    if not _same_json(hashes.get(relative), {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}):
        raise ValueError('source-guard bytes differ from the verified source')
    module = types.ModuleType('_flan_t5_verified_source_guards')
    module.__file__ = str(Path(repo) / relative)
    exec(compile(data, module.__file__, 'exec'), module.__dict__)
    if not isinstance(module.SourceFileLease, type) or not isinstance(module.WindowsJob, type):
        raise ValueError('verified source guards lack the required capabilities')
    return module.SourceFileLease, module.WindowsJob


def _tool_identity(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('tool path must be absolute')
    requested = path
    resolved = path.resolve(strict=True)
    parent = _plain_directory(resolved.parent)
    resolved = parent / resolved.name
    raw = _bytes(resolved, 64 * _MIB)
    if not raw:
        raise ValueError('tool executable is empty')
    return {'requested_path': str(requested), 'path': str(resolved),
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _hold_guard(key, guard_type, root, files):
    guard = guard_type.__new__(guard_type)
    _CUSTODY[key] = guard
    guard_type.__init__(guard, root, tuple(files))
    if guard_type.__enter__(guard) is not guard or not guard.handles():
        raise RuntimeError('source protection was not established')
    return guard


def _owner_evidence_files(directory):
    root = _plain_directory(directory)
    staged = _plain_directory(root / 'staged-assets')
    staged_files = []
    with os.scandir(staged) as entries:
        for entry in entries:
            if entry.name not in _STAGED_FILES or len(staged_files) >= len(_STAGED_FILES):
                raise ValueError('staged checkpoint file inventory differs')
            staged_files.append(staged / entry.name)
    if {item.name for item in staged_files} != _STAGED_FILES:
        raise ValueError('staged checkpoint file inventory differs')
    files = list(staged_files)
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name == 'staged-assets':
                continue
            if len(files) >= 256:
                raise ValueError('owner evidence file inventory exceeds its bound')
            files.append(root / entry.name)
    for item in files:
        info = item.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('owner evidence must contain only plain files')
    if not files:
        raise ValueError('owner evidence file inventory exceeds its bound')
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _evidence_manifest(directory, files):
    root = _plain_directory(directory)
    manifest, total = {}, 0
    for path in files:
        path = Path(path)
        if not path.is_relative_to(root):
            raise ValueError('evidence file lies outside its root')
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400
                or before.st_size > _DISK_BUDGET):
            raise ValueError('evidence file is not bounded and regular')
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            while True:
                block = stream.read(_MIB)
                if not block:
                    break
                count += len(block)
                if count > before.st_size:
                    raise ValueError('evidence file grew during hashing')
                digest.update(block)
        after = path.lstat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if count != before.st_size or identity(before) != identity(opened) or identity(before) != identity(after):
            raise ValueError('evidence file changed while hashing')
        total += count
        if total > _DISK_BUDGET:
            raise ValueError('owner evidence exceeds its disk bound')
        manifest[path.relative_to(root).as_posix()] = {'bytes': count, 'sha256': digest.hexdigest()}
    return manifest


def _inventory_attempt(root):
    """Count present files, including partial staging; do not follow links."""
    root = _plain_directory(root)
    totals = {'staged_asset_bytes': 0, 'owner_record_bytes': 0,
              'controller_and_temporary_bytes': 0, 'file_count': 0}
    directories, pending = 1, [root]
    while pending:
        parent = pending.pop()
        count = 0
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                if count > 256:
                    raise ValueError('attempt inventory exceeds its bounds')
                path = parent / entry.name
                info = path.lstat()
                if getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('attempt contains a linked or invalid entry')
                if stat.S_ISDIR(info.st_mode):
                    directories += 1
                    if directories > 64:
                        raise ValueError('attempt inventory exceeds its bounds')
                    pending.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('attempt contains a linked or invalid file')
                parts = path.relative_to(root).parts
                category = ('staged_asset_bytes' if parts[:2] == ('owner', 'staged-assets')
                            else 'owner_record_bytes' if parts[0] == 'owner'
                            else 'controller_and_temporary_bytes')
                totals[category] += info.st_size
                totals['file_count'] += 1
                if totals['file_count'] > 256 or sum(totals[k] for k in (
                        'staged_asset_bytes', 'owner_record_bytes', 'controller_and_temporary_bytes')) > _DISK_BUDGET:
                    raise ValueError('attempt disk inventory exceeds its bound')
    return totals


def _environment(gpu_uuid, run, base_environment):
    environment = {key: base_environment[key] for key in ('SYSTEMROOT', 'WINDIR', 'PATHEXT', 'COMSPEC')
                   if key in base_environment}
    system = Path(environment.get('SYSTEMROOT', 'C:/Windows'))
    environment.update({'PATH': str(system / 'System32'), 'TEMP': str(run / 'tmp'), 'TMP': str(run / 'tmp'),
        'CUDA_VISIBLE_DEVICES': gpu_uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'PYTORCH_CUDA_ALLOC_CONF': 'backend:native', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    return environment


def _check_inventory(snapshot):
    names = {'device_name', 'total_bytes', 'free_bytes', 'used_bytes', 'driver_version', 'timestamp_utc'}
    if (type(snapshot) is not dict or set(snapshot) != names
            or any(type(snapshot[key]) is not int for key in ('total_bytes', 'free_bytes', 'used_bytes'))
            or not 7 * _GIB <= snapshot['total_bytes'] <= 9 * _GIB
            or not 0 <= snapshot['free_bytes'] <= snapshot['total_bytes']
            or not 0 <= snapshot['used_bytes'] <= snapshot['total_bytes']
            or snapshot['free_bytes'] + snapshot['used_bytes'] > snapshot['total_bytes'] + 2 * _MIB
            or any(type(snapshot[key]) is not str or not 0 < len(snapshot[key]) <= 256
                   for key in ('device_name', 'driver_version', 'timestamp_utc'))):
        raise ValueError('invalid device inventory or physical memory counters')
    try:
        timestamp = datetime.fromisoformat(snapshot['timestamp_utc'])
    except ValueError as error:
        raise ValueError('invalid device inventory timestamp') from error
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError('invalid device inventory timestamp')
    return timestamp


def run_controller(*, workspace_root, run_dir, asset_dir, site_packages, git_executable, smi, source_commit):
    """Run one exact child after a future separately reviewed enablement."""
    if GPU_SMOKE_ENABLED is not True:
        raise RuntimeError('the FLAN-T5 GPU development smoke is disabled')
    with _CUSTODY_LOCK:
        if _CUSTODY['entered'] is not False:
            raise RuntimeError('this controller is single-use')
        _CUSTODY['entered'] = True
    attempt = None
    supervision = None
    clean_child = False
    try:
        repo = Path(__file__).resolve().parents[1]
        _check_bootstrap(repo)
        _require_pristine_dependencies()
        workspace, assets, site = (_plain_directory(workspace_root), _plain_directory(asset_dir),
                                   _plain_directory(site_packages))
        if site != Path(os.environ['LOOKING_GLASS_FLAN_T5_BOOTSTRAP_SITE']).resolve(strict=True):
            raise ValueError('dependency site differs from the bootstrap')
        if not Path(git_executable).is_absolute() or not Path(smi).is_absolute():
            raise ValueError('absolute Git and inventory executable paths are required')
        git_executable = Path(git_executable).resolve(strict=True)
        smi = Path(smi).resolve(strict=True)
        if not git_executable.is_file() or not smi.is_file():
            raise ValueError('Git or inventory executable is unavailable')
        if tuple(sys.version_info[:3]) != (3, 11, 15):
            raise RuntimeError('controller interpreter version differs from the pinned runtime')
        tool_paths = {'python': Path(sys._base_executable), 'git': git_executable, 'nvidia_smi': smi}
        tool_identities = {name: _tool_identity(path) for name, path in tool_paths.items()}
        attempt = _new_attempt(workspace, repo, run_dir)
        _CUSTODY['attempt'] = attempt
        if (assets == attempt or assets.is_relative_to(attempt) or attempt.is_relative_to(assets)):
            raise ValueError('source assets and the private attempt must be separate')
        attempt_id = secrets.token_hex(16)
        _write_json(attempt / 'controller-reserved.json', {'schema': 'flan-t5-gpu-controller-v1',
            'status': 'reserved', 'attempt_id': attempt_id, 'source_commit': source_commit,
            'source_chronology': 'controller-reported', 'max_forwards': 6})
        hashes = _verify_source_tree(repo, git_executable, source_commit, SOURCE_PATHS)
        guard_type, job_type = _guard_types(repo, hashes)
        source_guard = _hold_guard('source_guard', guard_type, repo,
                                   (repo / name for name in SOURCE_PATHS))
        if _verify_source_tree(repo, git_executable, source_commit, SOURCE_PATHS) != hashes:
            raise RuntimeError('source changed while acquiring protection')
        from .flan_t5_feasibility import verify_assets
        manifest = verify_assets(assets)
        asset_guard = _hold_guard('asset_guard', guard_type, assets,
            [assets / 'asset-manifest.json', *(assets / name for name in manifest['files'])])
        if verify_assets(assets) != manifest:
            raise RuntimeError('checkpoint changed while acquiring protection')
        _fixed_rows(repo)
        payload = sum(item['bytes'] for item in manifest['files'].values())
        manifest_bytes = (assets / 'asset-manifest.json').stat().st_size
        if (type(payload) is not int or payload <= 0 or payload + manifest_bytes + _LIMIT > _DISK_BUDGET
                or shutil.disk_usage(attempt).free < _DISK_BUDGET + _DISK_MARGIN):
            raise RuntimeError('insufficient point-in-time disk space for the fixed attempt budget')
        gpu_uuid, before = _smi_snapshot(str(smi))
        before_time = _check_inventory(before)
        if before['free_bytes'] < 3 * _GIB:
            raise RuntimeError('insufficient device-wide free memory before supervision')
        (attempt / 'tmp').mkdir()
        (attempt / 'pycache').mkdir()
        environment = _environment(gpu_uuid, attempt, os.environ)
        identity = resolve_device_identity(workspace, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
        _write_json(attempt / 'controller-preflight.json', {'schema': 'flan-t5-gpu-controller-v1',
            'status': 'preflight', 'attempt_id': attempt_id, 'source_commit': source_commit,
            'source_sha256': hashes, 'source_chronology': 'controller-reported',
            'tool_executables': tool_identities,
            'asset_manifest_sha256': _ASSET_HASH, 'case_manifest_sha256': _CASE_HASH,
            'staged_payload_bytes_expected': payload, 'staged_manifest_bytes_expected': manifest_bytes,
            'attempt_disk_budget_bytes': _DISK_BUDGET, 'disk_margin_bytes': _DISK_MARGIN,
            'disk_space_is_reserved': False, 'operator_log_capture': 'inherited-and-external',
            'nvidia_smi_before': dict(before, uuid=gpu_uuid), 'timeout_seconds': _TIMEOUT})
        job = job_type.__new__(job_type)
        _CUSTODY['job'] = job
        job_type.__init__(job)
        prefix = attempt / 'pycache'
        command = [tool_identities['python']['path'], '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
            str(repo / 'scripts/flan_t5_gpu_bootstrap.py'), '--bootstrap-site', str(site), '--owner',
            '--run-dir', str(attempt), '--asset-dir', str(assets), '--source-commit', source_commit,
            '--git-executable', str(git_executable)]
        _CUSTODY['may_have_started'] = True
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, process_job=job,
            timeout=_TIMEOUT, lease_timeout=30, cwd=attempt / 'tmp', env=environment,
            inherit_environment=False, require_zero_exit=True)
        _CUSTODY['supervision'] = supervision
        if type(supervision) is not SupervisionResult:
            raise RuntimeError('supervisor returned an unsupported receipt')
        _write_json(attempt / 'supervision.json', {'schema': 'flan-t5-gpu-supervision-v1',
            'attempt_id': attempt_id, 'run_id': supervision.run_id, 'pid': supervision.pid,
            'exit_code': supervision.exit_code, 'clean': supervision.clean,
            'timed_out': supervision.timed_out, 'reason': supervision.reason})
        clean_child = (type(supervision.exit_code) is int and supervision.exit_code == 0
                       and supervision.clean is True and supervision.timed_out is False)
        if not clean_child:
            raise RuntimeError('the exact supervised child did not complete cleanly')
        if os.path.lexists(identity.marker_path) or os.path.lexists(identity.attestation_path):
            raise RuntimeError('device marker or attestation remains after supervision')
        if (_verify_source_tree(repo, git_executable, source_commit, SOURCE_PATHS) != hashes
                or verify_assets(assets) != manifest
                or {name: _tool_identity(path) for name, path in tool_paths.items()} != tool_identities):
            raise RuntimeError('source, checkpoint or tool executable changed during the child attempt')
        _fixed_rows(repo)
        after_uuid, after = _smi_snapshot(str(smi))
        after_time = _check_inventory(after)
        if (after_uuid != gpu_uuid or after['device_name'] != before['device_name']
                or after['total_bytes'] != before['total_bytes']
                or after['driver_version'] != before['driver_version'] or after_time < before_time):
            raise RuntimeError('physical device inventory changed during the attempt')
        presence = _owner_presence_after_exit(str(smi), supervision.pid)
        if (type(presence) is not dict or set(presence) != {'query', 'owner_pid_listed'}
                or presence['query'] not in ('ok', 'inconclusive', 'unavailable')
                or (presence['owner_pid_listed'] is not None
                    and type(presence['owner_pid_listed']) is not bool)
                or (presence['query'] == 'ok') != (type(presence['owner_pid_listed']) is bool)):
            raise RuntimeError('the post-exit GPU process query is malformed')
        if presence['owner_pid_listed'] is True:
            raise RuntimeError('the reaped owner is still present in the GPU process inventory')
        owner_root = attempt / 'owner'
        evidence_files = _owner_evidence_files(owner_root)
        evidence_guard = _hold_guard('evidence_guard', guard_type, owner_root, evidence_files)
        if _owner_evidence_files(owner_root) != evidence_files:
            raise RuntimeError('owner evidence changed while acquiring protection')
        records = _validate_records(owner_root, supervision, source_commit=source_commit,
            source_hashes=hashes, gpu_uuid=gpu_uuid, device_name=before['device_name'], site_packages=site)
        evidence_manifest = _evidence_manifest(owner_root, evidence_files)
        if _owner_evidence_files(owner_root) != evidence_files:
            raise RuntimeError('owner evidence inventory changed during validation')
        inventory = _inventory_attempt(attempt)
        outcome = {'schema': 'flan-t5-gpu-controller-v1', 'status': 'awaiting-independent-audit',
            'attempt_id': attempt_id, 'source_commit': source_commit, 'supervisor_run_id': supervision.run_id,
            'owner_pid': supervision.pid, 'nvidia_smi_before': dict(before, uuid=gpu_uuid),
            'nvidia_smi_after': dict(after, uuid=after_uuid), 'owner_presence_after_exit': presence,
            'record_check': records, 'owner_evidence_files': evidence_manifest,
            'attempt_inventory_before_final': inventory, 'requires_observed_controller_exit_zero': True,
            'operator_log_capture': 'inherited-and-external', 'source_chronology': 'controller-reported',
            'limitations': ['saved-record consistency is not an execution attestation',
                'protected existing files do not prevent new directory entries',
                'entry-file hashes do not attest all dependency or native bytes',
                'source protection requires the controller to remain alive',
                'process-local allocator counters do not prove post-exit device-wide cleanup',
                'an inconclusive GPU-process query is not proof that the owner was absent',
                'operator logs require separate retention and accounting']}
        final_hash = _write_json(attempt / 'controller-final.json', outcome)
        evidence_guard.close()
        asset_guard.close()
        source_guard.close()
        _write_json(attempt / 'controller-closed.json', {'schema': 'flan-t5-gpu-controller-closure-v1',
            'status': 'awaiting-independent-audit', 'attempt_id': attempt_id,
            'supervisor_run_id': supervision.run_id, 'owner_pid': supervision.pid,
            'controller_final_sha256': final_hash, 'requires_observed_controller_exit_zero': True})
        return outcome
    except BaseException as error:
        _CUSTODY['failure'] = error
        reported = error
        status = ('quarantined' if _CUSTODY['may_have_started'] and not clean_child else 'incomplete')
        outcome = {'schema': 'flan-t5-gpu-controller-v1', 'status': status,
                   'error_type': type(error).__name__, 'may_have_started': _CUSTODY['may_have_started']}
        if attempt is not None:
            try:
                outcome['attempt_inventory_before_failure'] = _inventory_attempt(attempt)
                _write_json(attempt / 'controller-failure.json', outcome)
            except BaseException as diagnostic_error:
                _CUSTODY['diagnostic_failure'] = diagnostic_error
                if (isinstance(diagnostic_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(error, (KeyboardInterrupt, SystemExit))):
                    reported = diagnostic_error
        outcome['error_type'] = type(reported).__name__
        return outcome


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace-root', 'run-dir', 'asset-dir', 'git-executable', 'smi'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args(argv)
    outcome = run_controller(workspace_root=args.workspace_root, run_dir=args.run_dir,
        asset_dir=args.asset_dir, site_packages=Path(os.environ['LOOKING_GLASS_FLAN_T5_BOOTSTRAP_SITE']),
        git_executable=args.git_executable, smi=args.smi, source_commit=args.source_commit)
    print(json.dumps({'status': outcome['status'], 'error_type': outcome.get('error_type')}), flush=True)
    return 0 if outcome['status'] == 'awaiting-independent-audit' else (
        130 if outcome.get('error_type') == 'KeyboardInterrupt' else 1)


if __name__ == '__main__':
    raise SystemExit(main())
