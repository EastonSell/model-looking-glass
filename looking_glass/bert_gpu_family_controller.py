"""Source-bound supervisor and closed-evidence audit for one BERT GPU family.

This fixed development profile does not establish a general memory bound,
latency result, physical isolation, or an evaluation result. Unknown owner
outcomes remain quarantined; partial evidence is retained in the run folder.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile

from .bert_gpu_calibration import (ASSETS, MODEL_ID, REVISION, _file_identity,
    _require_pristine_dependencies, _verify_offline_cache, _verify_startup, check_memory)
from .bert_gpu_evidence import write_json_new
from . import bert_gpu_family_owner as _family_owner
from .bert_gpu_family_owner import SOURCE_PATHS, _load_manifest, _verify_source
from .bert_gpu_family_profile import _read_profile_journal
from .device_identity import resolve_device_identity
from .gpu_oneshot import _owner_presence_after_exit, _smi_snapshot
from .supervisor import run_supervised

_MIB = 1024 ** 2
_GIB = 1024 ** 3
_HEADROOM = 256 * _MIB
_PEAK_SCOPE = 'cumulative_since_pre_model_load_reset'
_MANIFEST_SHA = '20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5'
_MEMORY_LIMIT = 4 * _MIB
_LINE_LIMIT = 12 * 1024
_RUNTIME = {'python': '3.11.15', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
            'transformers': '4.57.3', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.4.6'}
_INVENTORY = {'started.json': 16384, 'supervision.json': 16384, 'owner.json': 16384,
              'candidate.json': _MIB, 'final.json': 16384, 'events.jsonl': 4 * _MIB,
              'memory.jsonl': _MEMORY_LIMIT, 'failure.json': 16384, 'interrupted.json': 16384}


def _json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate evidence key')
            result[key] = value
        return result

    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite evidence number')
        return result

    value = json.loads(raw, object_pairs_hook=unique, parse_float=finite,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite evidence number')))
    if type(value) is not dict:
        raise ValueError('evidence must be a JSON object')
    return value


def _read_json(path, limit):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError('evidence is empty or exceeds its byte cap')
    return _json(raw), hashlib.sha256(raw).hexdigest()


def _same_json(left, right):
    """Compare serialized types as well as values (False must not alias 0)."""
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(',', ':')) == json.dumps(
        right, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _read_memory(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(_MEMORY_LIMIT + 1)
    if not raw or len(raw) > _MEMORY_LIMIT or not raw.endswith(b'\n'):
        raise ValueError('memory journal is empty, oversized, or incomplete')
    if b'\r' in raw or raw.count(b'\n') != 287:
        raise ValueError('memory journal has the wrong number of records')
    lines = raw[:-1].split(b'\n')
    if any(not line or len(line) > _LINE_LIMIT for line in lines):
        raise ValueError('memory journal record exceeds its limit')
    return [_json(line) for line in lines], hashlib.sha256(raw).hexdigest()


def _validate_memory(records, measurement):
    stages = [(name, None) for name in ('before_model_load', 'before_model_copy',
                                        'after_model_copy', 'after_initial_state')]
    for index in range(1, 141):
        stages.extend((('before', index), ('after', index)))
    stages.extend((name, None) for name in ('after_final_state',
                  'cleanup_before_workspace_clear', 'after_cleanup'))
    if type(records) is not list or len(records) != len(stages):
        raise ValueError('memory observation inventory differs')
    peak_allocated = peak_reserved = 0
    for record, (stage, index) in zip(records, stages):
        if (type(record) is not dict or set(record) != {'stage', 'index', 'memory', 'peak_scope'}
                or record['stage'] != stage or type(record['index']) is not type(index)
                or record['index'] != index or record['peak_scope'] != _PEAK_SCOPE):
            raise ValueError('memory observation order or identity differs')
        memory = record['memory']
        check_memory(memory, expected_total=measurement['device_total_bytes'],
                     cap=measurement['allocator_cap_bytes'],
                     commitment=measurement['whole_run_commitment_bytes'])
        if (memory['peak_allocated_bytes'] < peak_allocated
                or memory['peak_reserved_bytes'] < peak_reserved):
            raise ValueError('cumulative allocator peak decreased')
        peak_allocated = memory['peak_allocated_bytes']
        peak_reserved = memory['peak_reserved_bytes']
    if (records[0]['memory']['device_free_bytes'] < 3 * _GIB
            or not _same_json(records[-1]['memory'], measurement['cleanup'])
            or records[-1]['memory']['allocated_bytes'] != 0
            or records[-1]['memory']['reserved_bytes'] != 0):
        raise ValueError('initial or final cleanup evidence differs')


def _validate_measurement(value, source_hashes, manifest_sha):
    if type(value) is not dict:
        raise ValueError('measurement must be an object')
    total, cap = value.get('device_total_bytes'), value.get('allocator_cap_bytes')
    metadata = value.get('model_metadata')
    expected_assets = {name: {'size': size, 'sha256': digest}
                       for name, (size, digest) in ASSETS.items()}
    expected_config = {'device': 'cuda:0', 'threads': 1, 'interop_threads': 1,
        'deterministic_algorithms': True, 'matmul_tf32': False, 'cudnn_tf32': False,
        'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'overwrite_module_params_on_conversion': False, 'swap_module_params_on_conversion': False}
    if (value.get('source_sha256') != source_hashes
            or value.get('manifest_sha256') != manifest_sha or manifest_sha != _MANIFEST_SHA
            or value.get('runtime_uuid_verified') is not True
            or value.get('model_weakref_alive') is not False
            or type(value.get('budget_active_before_candidate_write')) is not int
            or value['budget_active_before_candidate_write'] != 1
            or value.get('device_name') != 'NVIDIA GeForce RTX 2070 SUPER'
            or type(total) is not int or not 7 * _GIB <= total <= 9 * _GIB
            or type(cap) is not int or cap != int(.25 * total)
            or type(value.get('allocator_fraction')) is not float or value['allocator_fraction'] != .25
            or value.get('allocator_backend') != 'native'
            or type(value.get('whole_run_commitment_bytes')) is not int
            or value['whole_run_commitment_bytes'] != cap + _HEADROOM
            or type(value.get('gpu_trace_claim_bytes')) is not int
            or value['gpu_trace_claim_bytes'] != 64 * 1024
            or type(value.get('host_trace_claim_bytes')) is not int
            or value['host_trace_claim_bytes'] != 64 * _MIB
            or value.get('peak_scope') != _PEAK_SCOPE
            or value.get('torch_version') != '2.8.0+cu128' or value.get('cuda_build') != '12.8'
            or not _same_json(value.get('runtime_versions'), _RUNTIME)
            or type(value.get('seed')) is not int or value['seed'] != 17
            or value.get('cublas_workspace_config') != ':4096:8'
            or type(value.get('model_state_sha256')) is not str
            or re.fullmatch('[0-9a-f]{64}', value['model_state_sha256']) is None
            or json.dumps(value.get('execution_config'), sort_keys=True) != json.dumps(expected_config, sort_keys=True)
            or type(metadata) is not dict or metadata.get('model') != MODEL_ID
            or metadata.get('revision') != REVISION or not _same_json(metadata.get('assets'), expected_assets)
            or metadata.get('model_class') != 'DistilBertForMaskedLM'
            or metadata.get('tokenizer_class') != 'DistilBertTokenizerFast'
            or metadata.get('transformers_version') != '4.57.3'
            or metadata.get('tokenizers_version') != _RUNTIME['tokenizers']
            or metadata.get('safetensors_version') != _RUNTIME['safetensors']
            or metadata.get('attention_implementation') != 'eager'
            or type(metadata.get('parameter_count')) is not int or metadata['parameter_count'] != 66985530
            or metadata.get('cpu_tied_weight_verified') is not True):
        raise ValueError('model, source, runtime, or resource identity differs')


def _validate_bundle(run_dir, *, run_id, supervision, source_commit, source_hashes,
                     manifest, manifest_sha, audit=None, verified_hashes=None):
    """Validate closed records after a clean child reap; never promote partials."""
    directory = Path(run_dir)
    owner, owner_sha = _read_json(directory / 'owner.json', 16384)
    candidate, candidate_sha = _read_json(directory / 'candidate.json', _MIB)
    final, final_sha = _read_json(directory / 'final.json', 16384)
    for row, schema in ((owner, 'bert-gpu-family-owner-v1'),
                        (candidate, 'bert-gpu-family-candidate-v1'),
                        (final, 'bert-gpu-family-final-v1')):
        if (row.get('schema') != schema or row.get('run_id') != run_id
                or row.get('supervisor_run_id') != supervision.run_id
                or type(row.get('pid')) is not int or row['pid'] != supervision.pid
                or row.get('source_commit') != source_commit):
            raise ValueError('owner evidence does not bind the reaped child')
    if (owner.get('manifest_sha256') != manifest_sha or final.get('candidate_sha256') != candidate_sha
            or type(final.get('active_memory_leases')) is not int or final['active_memory_leases'] != 0):
        raise ValueError('final cleanup does not bind the owner candidate')
    measurement = candidate.get('measurement')
    _validate_measurement(measurement, source_hashes, manifest_sha)
    memory, memory_sha = _read_memory(directory / 'memory.jsonl')
    if (memory_sha != measurement.get('memory_journal_sha256')
            or type(measurement.get('memory_record_count')) is not int
            or measurement['memory_record_count'] != len(memory)):
        raise ValueError('memory journal digest or record count differs')
    _validate_memory(memory, measurement)
    records, journal_sha = _read_profile_journal(directory / 'events.jsonl')
    profile = measurement.get('profile')
    if (type(profile) is not dict or type(profile.get('forward_count')) is not int
            or profile['forward_count'] != 140 or type(profile.get('record_count')) is not int
            or profile['record_count'] != 328 or len(records) != 328
            or profile.get('journal_sha256') != journal_sha or profile.get('expected_device') != 'cuda:0'):
        raise ValueError('event journal is not the complete fixed workload')
    if audit is None:
        # This CPU audit imports torch via the shared study runner only after the
        # owner has exited; it does not initialize CUDA or run a forward.
        from .bert_agreement_audit import audit_family_events
        audit = audit_family_events
    checked = audit(manifest, records, split='development', family=1,
                    mask_id=103, hidden_size=768, expected_device='cuda:0')
    if (type(checked) is not dict or not _same_json(profile.get('live_cases'), checked.get('cases'))
            or not _same_json(profile.get('audited_cases'), checked.get('cases'))
            or not _same_json(profile.get('site_means'), checked.get('site_means'))
            or type(checked.get('controls')) is not list
            or type(profile.get('control_count')) is not int
            or profile['control_count'] != len(checked['controls'])):
        raise ValueError('closed event journal does not reproduce live results')
    if verified_hashes is not None:
        if type(verified_hashes) is not dict:
            raise ValueError('verified evidence inventory must be a dictionary')
        verified_hashes.update({'owner.json': owner_sha, 'candidate.json': candidate_sha,
            'final.json': final_sha, 'memory.jsonl': memory_sha, 'events.jsonl': journal_sha})
    return measurement


def _public_measurement(value):
    # The complete owner candidate remains in the private run directory. Copy
    # only fields checked above; never propagate arbitrary extension fields.
    names = ('model_state_sha256', 'execution_config', 'device_name', 'device_total_bytes',
        'allocator_backend', 'allocator_fraction', 'allocator_cap_bytes', 'whole_run_commitment_bytes',
        'gpu_trace_claim_bytes', 'host_trace_claim_bytes', 'peak_scope', 'runtime_uuid_verified',
        'cleanup', 'model_weakref_alive', 'budget_active_before_candidate_write', 'torch_version',
        'cuda_build', 'runtime_versions', 'seed', 'cublas_workspace_config', 'memory_journal_sha256',
        'memory_record_count', 'manifest_sha256', 'source_sha256')
    result = {key: value[key] for key in names}
    memory_fields = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
                     'peak_allocated_bytes', 'peak_reserved_bytes')
    result['cleanup'] = {key: value['cleanup'][key] for key in memory_fields}
    metadata_fields = ('model', 'revision', 'assets', 'model_class', 'tokenizer_class',
        'transformers_version', 'tokenizers_version', 'safetensors_version', 'parameter_count',
        'attention_implementation', 'cpu_tied_weight_verified')
    result['model_metadata'] = {key: value['model_metadata'][key] for key in metadata_fields}
    profile_fields = ('forward_count', 'journal_sha256', 'record_count', 'expected_device',
                      'live_cases', 'audited_cases', 'site_means', 'control_count')
    result['profile'] = {key: value['profile'][key] for key in profile_fields}
    return result


def _workspace_root(repo, requested):
    """Bind this research worktree layout to the common coordination root."""
    repo, requested = Path(repo).resolve(strict=True), Path(requested)
    if not requested.is_absolute():
        raise ValueError('the workspace root must be absolute')
    requested = requested.resolve(strict=True)
    work_parents = [parent for parent in repo.parents if parent.name == 'work']
    expected = work_parents[0].parent if work_parents else repo
    if requested != expected or not (expected / 'work').is_dir():
        raise ValueError('workspace root does not match the shared checkout layout')
    return expected


def _inventory(run_dir):
    result = []
    for name, maximum in _INVENTORY.items():
        for item in (name, name + '.partial'):
            path = Path(run_dir) / item
            if not path.exists():
                continue
            size, digest = _file_identity(path, maximum)
            result.append({'name': item, 'size': size, 'sha256': digest})
    return result


def _validate_complete_inventory(inventory, verified_hashes=None):
    rows = {item['name']: item for item in inventory}
    required = {'started.json', 'supervision.json', 'owner.json', 'candidate.json',
                'final.json', 'events.jsonl', 'memory.jsonl'}
    if (len(rows) != len(inventory) or not required <= set(rows)
            or {'failure.json', 'interrupted.json'} & set(rows)):
        raise ValueError('owner evidence inventory is missing or contradictory')
    if verified_hashes is not None and (type(verified_hashes) is not dict
            or set(verified_hashes) != required
            or any(rows[name].get('sha256') != verified_hashes[name] for name in required)):
        raise ValueError('inventoried evidence differs from the checked records')
    for name, item in rows.items():
        if name.endswith('.partial'):
            final = rows.get(name[:-8])
            if (final is None or final.get('size') != item.get('size')
                    or final.get('sha256') != item.get('sha256')):
                raise ValueError('partial evidence does not match a complete publication')


def _child_environment(gpu_uuid, cache, prefix):
    keys = ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATHEXT', 'COMSPEC', 'PROGRAMFILES')
    environment = {key: os.environ[key] for key in keys if key in os.environ}
    program_files = Path(environment.get('PROGRAMFILES', ''))
    if not program_files.is_absolute() or not program_files.is_dir():
        raise RuntimeError('a valid PROGRAMFILES directory is required for NVIDIA telemetry')
    system_root = Path(environment.get('SYSTEMROOT', 'C:/Windows'))
    environment.update({'PATH': str(system_root / 'System32'), 'CUDA_VISIBLE_DEVICES': gpu_uuid,
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HOME': str(cache.parent),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPYCACHEPREFIX': str(prefix), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1'})
    return environment


def run_profile(*, workspace_root, cache_dir, output, site_packages, smi, git_exe):
    if _family_owner.FAMILY_PROFILE_ENABLED is not True:
        raise RuntimeError('BERT GPU family execution is disabled pending exact-source review')
    _verify_startup()
    _require_pristine_dependencies()
    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    cache = Path(cache_dir).resolve(strict=True)
    site = Path(site_packages).resolve(strict=True)
    git = Path(git_exe)
    destination = Path(output)
    if (not git.is_absolute() or not git.is_file() or not destination.is_absolute()
            or os.path.lexists(destination) or not destination.parent.is_dir()):
        raise ValueError('an absolute Git executable and new absolute output are required')
    git = git.resolve(strict=True)
    with tempfile.NamedTemporaryFile(prefix='.bert-family-write-', dir=destination.parent):
        pass
    commit = subprocess.run([str(git), 'rev-parse', 'HEAD'], cwd=repo, capture_output=True,
                            text=True, check=True, timeout=10).stdout.strip()
    if re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise RuntimeError('invalid source commit')
    source_hashes = _verify_source(repo, commit, git)
    if set(source_hashes) != set(SOURCE_PATHS):
        raise RuntimeError('source identity inventory differs')
    manifest, manifest_sha = _load_manifest(repo)
    if manifest_sha != _MANIFEST_SHA:
        raise RuntimeError('manifest identity differs')
    _verify_offline_cache(cache)
    gpu_uuid, before = _smi_snapshot(smi)
    run_id = secrets.token_hex(16)
    run_dir = root / 'work' / ('bert-gpu-family-' + run_id)
    run_dir.mkdir(mode=0o700, exist_ok=False)
    prefix = Path(tempfile.mkdtemp(prefix='pycache-', dir=run_dir))
    environment = _child_environment(gpu_uuid, cache, prefix)
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    started = {'schema': 'bert-gpu-family-start-v1', 'run_id': run_id,
               'source_commit': commit, 'started_utc': before['timestamp_utc']}
    write_json_new(run_dir / 'started.json', started)
    if not _same_json(_read_json(run_dir / 'started.json', 16384)[0], started):
        raise RuntimeError('saved start record differs from this run')
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
        str(repo / 'scripts' / 'bert_gpu_family_bootstrap.py'), '--bootstrap-site', str(site),
        '--owner', str(run_dir), str(cache), run_id, commit, str(git)]
    supervision = None
    supervisor_error = None
    try:
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, timeout=900, lease_timeout=30,
            cwd=repo, env=environment, inherit_environment=False)
    except (KeyboardInterrupt, SystemExit) as error:
        try:
            write_json_new(run_dir / 'interrupted.json', {'run_id': run_id,
                'status': 'unverified', 'error_type': type(error).__name__})
        except Exception:
            pass
        raise
    except Exception as error:
        supervisor_error = type(error).__name__
    evidence_error = None
    if supervision is not None:
        try:
            saved_supervision = {'run_id': run_id,
                'supervisor_run_id': supervision.run_id, 'pid': supervision.pid,
                'exit_code': supervision.exit_code, 'clean': supervision.clean,
                'timed_out': supervision.timed_out, 'reason': supervision.reason}
            write_json_new(run_dir / 'supervision.json', saved_supervision)
            reread, _ = _read_json(run_dir / 'supervision.json', 16384)
            if not _same_json(reread, saved_supervision):
                raise ValueError('saved supervision differs from the reaped child')
        except Exception as error:
            evidence_error = type(error).__name__
    after = None
    telemetry_error = None
    try:
        after_uuid, after = _smi_snapshot(smi)
        if after_uuid != gpu_uuid:
            raise RuntimeError('physical GPU inventory changed')
    except Exception as error:
        after = None
        telemetry_error = type(error).__name__
    presence = (_owner_presence_after_exit(smi, supervision.pid) if supervision is not None
                else {'query': 'not_attempted', 'owner_pid_listed': None})
    outcome = {'schema': 'bert-gpu-family-result-v1', 'run_id': run_id,
        'source_commit': commit, 'source_sha256': source_hashes,
        'status': 'quarantined', 'supervisor': ({'exit_code': supervision.exit_code,
            'clean': supervision.clean, 'timed_out': supervision.timed_out}
            if supervision is not None else None), 'supervisor_error_type': supervisor_error,
        'supervision_evidence_error_type': evidence_error, 'nvidia_smi_before': before,
        'nvidia_smi_after': after, 'post_telemetry_error_type': telemetry_error,
        'owner_presence_after_exit': presence,
        'limitations': ['one fixed public development family, not held-out evaluation',
            'no latency, throughput, or worst-case memory claim',
            'the lease cannot control unrelated GPU applications',
            'device memory snapshots can miss external or transient pressure',
            'the logical host evidence claim does not bound process RAM',
            'runtime UUID equality is a live check; its raw identifiers are not retained for offline replay']}
    if presence['query'] != 'ok':
        outcome['limitations'].append('the post-exit NVIDIA process listing was unavailable or inconclusive')
    if supervision is not None and supervision.clean and supervision.exit_code == 0 and not supervision.timed_out:
        outcome['status'] = 'incomplete'
        try:
            if _verify_source(repo, commit, git) != source_hashes:
                raise RuntimeError('source identity changed during execution')
            reread_start, start_sha = _read_json(run_dir / 'started.json', 16384)
            if not _same_json(reread_start, started):
                raise RuntimeError('saved start record changed during execution')
            reread_supervision, supervision_sha = _read_json(run_dir / 'supervision.json', 16384)
            if not _same_json(reread_supervision, saved_supervision):
                raise RuntimeError('saved supervision changed during execution')
            verified_hashes = {'started.json': start_sha, 'supervision.json': supervision_sha}
            measurement = _validate_bundle(run_dir, run_id=run_id,
                supervision=supervision, source_commit=commit, source_hashes=source_hashes,
                manifest=manifest, manifest_sha=manifest_sha, verified_hashes=verified_hashes)
            outcome['measurement'] = _public_measurement(measurement)
            if telemetry_error is None and evidence_error is None and presence['owner_pid_listed'] is not True:
                outcome['status'] = 'success'
        except Exception as error:
            outcome['candidate_error_type'] = type(error).__name__
    else:
        try:
            failure, _ = _read_json(run_dir / 'failure.json', 16384)
            if failure.get('schema') == 'bert-gpu-family-failure-v1' and failure.get('run_id') == run_id:
                outcome['owner_error_type'] = failure.get('error_type')
        except (OSError, ValueError):
            pass
    try:
        _verify_startup()
    except Exception as error:
        outcome['controller_import_error_type'] = type(error).__name__
        if outcome['status'] == 'success':
            outcome['status'] = 'incomplete'
    try:
        outcome['evidence_inventory'] = _inventory(run_dir)
        if outcome['status'] == 'success':
            _validate_complete_inventory(outcome['evidence_inventory'], verified_hashes)
    except Exception as error:
        outcome['inventory_error_type'] = type(error).__name__
        if outcome['status'] == 'success':
            outcome['status'] = 'incomplete'
    write_json_new(destination, outcome)
    return outcome


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', nargs=5, metavar=('RUN_DIR', 'CACHE_DIR', 'RUN_ID', 'COMMIT', 'GIT_EXE'),
                        help=argparse.SUPPRESS)
    parser.add_argument('--workspace-root')
    parser.add_argument('--cache-dir')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.owner:
        from .bert_gpu_family_owner import owner_main
        return owner_main(*args.owner)
    if not args.workspace_root or not args.cache_dir or not args.output:
        parser.error('--workspace-root, --cache-dir and --output are required')
    git, smi = shutil.which('git'), shutil.which('nvidia-smi')
    if git is None or smi is None:
        parser.error('Git and nvidia-smi must be available')
    result = run_profile(workspace_root=args.workspace_root, cache_dir=args.cache_dir,
        output=args.output, site_packages=os.environ['LOOKING_GLASS_BERT_BOOTSTRAP_SITE'],
        smi=smi, git_exe=git)
    print(json.dumps({'status': result['status'], 'source_commit': result['source_commit'],
                      'run_id': result['run_id']}))
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
