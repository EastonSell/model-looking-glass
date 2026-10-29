"""Supervised owner shell for the provisional five-token resource sampler.

The actual device workload and the independent evidence audit are separate
reviewed components. This module cannot approve a profile.
"""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile
import threading
import traceback

from .device_identity import resolve_device_identity
from .device_lease import DeviceLease
from .gpu_oneshot import _owner_presence_after_exit, _smi_snapshot
from .supervisor import OwnerSession, run_supervised


# This fixed, supervised resource-only pilot is enabled for its reviewed source.
CONTROLLER_EXECUTION_READY = True
_MAX_JSON = 1024 * 1024
_OWNER = {'entered': False, 'session': None, 'lease': None, 'candidate': None, 'failure': None}
_CONTROLLER = {'source_guard': None, 'asset_guard': None, 'process_job': None,
               'failure': None, 'launch_started': False}
# One process gets one attempt, including a failed preflight; never release it.
_RUN_ONCE_LOCK = threading.Lock()
_CLEANUP = {'runtime_uuid_verified': True, 'active_memory_leases': 0,
            'device_synchronized': True, 'model_weakref_dead': True,
            'allocator_allocated_bytes': 0, 'allocator_reserved_bytes': 0,
            'pytorch_cublas_cache_clear_completed': True, 'trace_gpu_bytes': 0, 'trace_host_bytes': 0}


def _canonical(value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > _MAX_JSON:
        raise ValueError('owner evidence exceeds its byte limit')
    return raw


def _write_private_bytes(path, raw):
    if type(raw) is not bytes or len(raw) > _MAX_JSON:
        raise ValueError('private evidence must be bounded bytes')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('evidence write did not progress')
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def _write_private(path, value):
    return _write_private_bytes(path, _canonical(value))


def _read_private(path, maximum=_MAX_JSON):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink() or
            getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('private evidence must be a regular non-link file')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) !=
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) or
            (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ValueError('private evidence changed while being read')
    if len(raw) > maximum:
        raise ValueError('private evidence exceeds its byte limit')
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate evidence key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite evidence')))
    if type(value) is not dict:
        raise ValueError('private evidence must be an object')
    return value, raw


def _directory_identity(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('private directory must have a canonical absolute path')
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.is_symlink() or not info.st_ino or
            getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('private directory must be an ordinary identified directory')
    return info.st_dev, info.st_ino


def _check_directory(path, expected):
    if _directory_identity(path) != expected:
        raise ValueError('private run directory identity changed')


def _bootstrap_state():
    import looking_glass
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or
            os.environ.get('LOOKING_GLASS_AGREEMENT_BOOTSTRAP_PID') != str(os.getpid())):
        raise RuntimeError('the isolated agreement bootstrap is required')
    prefix = Path(sys.pycache_prefix or '')
    site = Path(os.environ.get('LOOKING_GLASS_AGREEMENT_BOOTSTRAP_SITE', ''))
    if not prefix.is_absolute() or not prefix.is_dir() or next(prefix.iterdir(), None) is not None:
        raise RuntimeError('the isolated bytecode prefix is not empty')
    if not site.is_absolute() or not site.is_dir():
        raise RuntimeError('the isolated dependency directory is unavailable')
    manifest = getattr(looking_glass, '__agreement_source_manifest__', None)
    digest = getattr(looking_glass, '__agreement_source_sha256__', None)
    git = getattr(looking_glass, '__agreement_git__', None)
    recheck = getattr(looking_glass, '__agreement_recheck__', None)
    if (type(manifest) is not dict or not isinstance(digest, str) or
            re.fullmatch(r'[0-9a-f]{64}', digest) is None or
            not isinstance(git, str) or not callable(recheck)):
        raise RuntimeError('frozen source state is unavailable')
    return manifest, digest, Path(git), site, recheck


def _dependency_versions(site):
    """Read selected distribution metadata and Torch's literal CUDA build."""
    wanted = {'torch', 'transformers', 'numpy', 'huggingface-hub', 'safetensors', 'tokenizers'}
    found = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = distribution.metadata.get('Name')
        if not name:
            continue
        name = re.sub(r'[-_.]+', '-', name).lower()
        if name in wanted:
            if name in found or not distribution.version:
                raise ValueError('dependency distribution is ambiguous')
            found[name] = distribution.version
    if set(found) != wanted:
        raise ValueError('dependency distribution is unavailable')
    path = Path(site) / 'torch' / 'version.py'
    if path.stat().st_size > 32768:
        raise ValueError('Torch version file is unexpectedly large')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    literal = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        else:
            continue
        if name in ('__version__', 'cuda'):
            literal[name] = ast.literal_eval(value)
    if literal != {'__version__': found['torch'], 'cuda': '12.8'} or found['torch'] != '2.8.0+cu128':
        raise ValueError('Torch version metadata does not match the reviewed CUDA build')
    return {'python_version': sys.version.split()[0], 'torch_version': found['torch'],
            'transformers_version': found['transformers'], 'numpy_version': found['numpy'],
            'huggingface_hub_version': found['huggingface-hub'], 'safetensors_version': found['safetensors'],
            'tokenizers_version': found['tokenizers'], 'cuda_runtime_version': literal['cuda']}


def _workload():
    return {'seed': 20261028, 'target_input_ids': [464, 3797, 1474, 262, 9168],
            'donor_input_ids': [464, 11875, 1474, 262, 9168], 'subject_index': 1,
            'last_token_index': 4, 'batch_size': 1, 'sequence_length': 5}


def _check_run_manifest(value, run_id, manifest, source_digest, run_dir, directory_identity, cache):
    if (type(value) is not dict or set(value) != {'schema', 'run_id', 'source', 'checkpoint',
                                                 'runtime', 'device', 'workload', 'directory'}
            or value['schema'] != 'agreement-calibration-run-freeze-v1' or value['run_id'] != run_id
            or _canonical(value['workload']) != _canonical(_workload())):
        raise ValueError('prelaunch run freeze is invalid')
    source, checkpoint, directory = value['source'], value['checkpoint'], value['directory']
    repo = Path(__file__).resolve().parents[1]
    if (type(source) is not dict or set(source) != {'commit', 'sha256', 'checkout', 'source_base_sha256'}
            or source['commit'] != manifest['commit'] or source['sha256'] != source_digest
            or source['checkout'] != str(repo) or type(source['source_base_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', source['source_base_sha256']) is None):
        raise ValueError('prelaunch source freeze differs from the owner')
    if (type(checkpoint) is not dict or set(checkpoint) != {'model_id', 'revision', 'cache_root', 'assets'}
            or checkpoint['cache_root'] != str(cache) or type(checkpoint['assets']) is not list):
        raise ValueError('prelaunch checkpoint freeze differs from the owner')
    from .pretrained import ASSETS, MODEL_ID, REVISION
    expected_assets = [{'name': name, 'size': size, 'sha256': sha} for name, (size, sha) in sorted(ASSETS.items())]
    if (checkpoint['model_id'] != MODEL_ID or checkpoint['revision'] != REVISION
            or _canonical(checkpoint['assets']) != _canonical(expected_assets)):
        raise ValueError('prelaunch checkpoint assets differ from the reviewed pins')
    if (type(directory) is not dict or _canonical(directory) != _canonical({
            'path': str(run_dir), 'device': directory_identity[0], 'inode': directory_identity[1]})):
        raise ValueError('prelaunch private directory identity differs')
    runtime, device = value['runtime'], value['device']
    runtime_names = {'python_version', 'torch_version', 'transformers_version', 'numpy_version',
                     'huggingface_hub_version', 'safetensors_version', 'tokenizers_version',
                     'cuda_runtime_version'}
    if (type(runtime) is not dict or set(runtime) != runtime_names or
            any(type(item) is not str or not item for item in runtime.values()) or
            runtime['torch_version'] != '2.8.0+cu128' or runtime['cuda_runtime_version'] != '12.8'):
        raise ValueError('prelaunch runtime freeze is invalid')
    if (type(device) is not dict or set(device) != {'uuid', 'name', 'total_bytes', 'driver_version'} or
            type(device['uuid']) is not str or not device['uuid'].startswith('GPU-') or
            type(device['name']) is not str or not device['name'] or
            type(device['driver_version']) is not str or not device['driver_version'] or
            type(device['total_bytes']) is not int or not 7 * 1024**3 <= device['total_bytes'] <= 9 * 1024**3):
        raise ValueError('prelaunch runtime or device freeze is missing')


def _validate_candidate(value, run_id, run_manifest_sha256):
    if type(value) is not dict or set(value) != {'measurement', 'cleanup'}:
        raise ValueError('owner returned an invalid evidence envelope')
    cleanup, measurement = value['cleanup'], value['measurement']
    if type(cleanup) is not dict or cleanup != _CLEANUP or any(
            type(cleanup[name]) is not type(expected) for name, expected in _CLEANUP.items()):
        raise ValueError('owner cleanup observations are incomplete')
    expected = {'schema', 'run_id', 'profile_ids', 'estimates_bytes', 'source_sha256',
                'runtime_sha256', 'asset_sha256', 'run_manifest_sha256', 'sample_count', 'stage_count',
                'workspace_cleanup', 'limitations'}
    if type(measurement) is not dict or set(measurement) != expected:
        raise ValueError('measurement evidence has unexpected fields')
    if (measurement['schema'] != 'agreement-calibration-candidate-v1' or measurement['run_id'] != run_id
            or measurement['run_manifest_sha256'] != run_manifest_sha256
            or type(measurement['sample_count']) is not int or measurement['sample_count'] != 14
            or type(measurement['stage_count']) is not int or measurement['stage_count'] != 84):
        raise ValueError('measurement sequence or identity is incomplete')
    hashes = (measurement['source_sha256'], measurement['runtime_sha256'], measurement['asset_sha256'])
    ids, estimates = measurement['profile_ids'], measurement['estimates_bytes']
    if (any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None for value in hashes)
            or type(ids) is not list or len(ids) != 2 or len(set(ids)) != 2
            or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None for value in ids)
            or type(estimates) is not list or len(estimates) != 2
            or any(type(value) is not int or not 0 < value <= 3 * 1024**3 // 2 for value in estimates)):
        raise ValueError('measurement profile identity or estimates are invalid')
    workspace = measurement['workspace_cleanup']
    if (type(workspace) is not dict or set(workspace) != {'before', 'after', 'api'}
            or workspace['after'] != {'allocated_bytes': 0, 'reserved_bytes': 0}
            or workspace['api'] != 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces'):
        raise ValueError('workspace cleanup record is invalid')
    for key in ('before', 'after'):
        counters = workspace[key]
        if (type(counters) is not dict or set(counters) != {'allocated_bytes', 'reserved_bytes'}
                or any(type(value) is not int or value < 0 for value in counters.values())
                or counters['allocated_bytes'] > counters['reserved_bytes']):
            raise ValueError('workspace cleanup counters are invalid')
    limitations = measurement['limitations']
    if (type(limitations) is not list or not 1 <= len(limitations) <= 16 or
            any(type(text) is not str or not 1 <= len(text) <= 256 for text in limitations)):
        raise ValueError('measurement limitations are invalid')
    _canonical(value)
    return measurement


def _validate_private_identity(run_dir, manifest, measurement, run_manifest, site):
    source, source_raw = _read_private(run_dir / 'source-manifest.json')
    runtime, runtime_raw = _read_private(run_dir / 'runtime-manifest.json')
    assets, asset_raw = _read_private(run_dir / 'asset-manifest.json')
    if (hashlib.sha256(source_raw).hexdigest() != measurement['source_sha256'] or
            hashlib.sha256(runtime_raw).hexdigest() != measurement['runtime_sha256'] or
            hashlib.sha256(asset_raw).hexdigest() != measurement['asset_sha256']):
        raise ValueError('owner manifest hash does not match its candidate')
    if source.get('schema') != 'agreement-calibration-source-v1':
        raise ValueError('owner source manifest schema differs')
    base = source.get('base')
    files = base.get('repository_files') if type(base) is dict else None
    if type(files) is not list:
        raise ValueError('owner package source inventory is missing')
    expected = {path: digest for path, digest in manifest['files'].items()
                if path.startswith('looking_glass/') or path == 'docs/TRUSTED_SERVICE_PROFILE_V1.md'}
    actual = {}
    for item in files:
        if (type(item) is not dict or set(item) != {'path', 'size', 'sha256'}
                or type(item['path']) is not str or item['path'] in actual
                or type(item['size']) is not int or item['size'] < 0):
            raise ValueError('owner package source inventory is invalid')
        actual[item['path']] = item['sha256']
    if actual != expected:
        raise ValueError('owner package source differs from the executed commit')
    canonical_base = json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                                allow_nan=False).encode('utf-8')
    if hashlib.sha256(canonical_base).hexdigest() != run_manifest['source']['source_base_sha256']:
        raise ValueError('owner source preflight differs from the prelaunch freeze')
    protocols = source.get('protocols')
    expected_protocols = {name: manifest['files'][name] for name in (
        'docs/AGREEMENT_GPU_CALIBRATION_V1.md', 'docs/AGREEMENT_CUDA_ADAPTER.md',
        'docs/AGREEMENT_CALIBRATION_SAMPLING_DRAFT.md')}
    if type(protocols) is not list:
        raise ValueError('owner protocol inventory is missing')
    seen = {}
    for item in protocols:
        if (type(item) is not dict or set(item) != {'path', 'size', 'sha256'}
                or type(item['path']) is not str or item['path'] in seen
                or type(item['size']) is not int or item['size'] < 0):
            raise ValueError('owner protocol inventory is invalid')
        seen[item['path']] = item['sha256']
    if seen != expected_protocols:
        raise ValueError('owner protocol source differs from the executed commit')
    from .live_profile import _plain_file_hash
    origins = source.get('dependency_origins')
    version_fields = {'torch': 'torch_version', 'transformers': 'transformers_version',
                      'numpy': 'numpy_version', 'huggingface_hub': 'huggingface_hub_version',
                      'safetensors': 'safetensors_version', 'tokenizers': 'tokenizers_version'}
    if type(origins) is not list or len(origins) != len(version_fields):
        raise ValueError('owner dependency inventory is incomplete')
    seen_origins = set()
    for origin in origins:
        if (type(origin) is not dict or set(origin) != {'name', 'version', 'path', 'size', 'sha256'}
                or origin['name'] not in version_fields or origin['name'] in seen_origins):
            raise ValueError('owner dependency origin is invalid')
        name = origin['name']
        seen_origins.add(name)
        path = Path(origin['path'])
        if (not path.is_absolute() or path.resolve(strict=True) != path or not path.is_relative_to(site)
                or origin['version'] != run_manifest['runtime'][version_fields[name]]
                or _plain_file_hash(path) != {'size': origin['size'], 'sha256': origin['sha256']}):
            raise ValueError('owner dependency origin differs from the selected site or freeze')
    checkpoint = run_manifest['checkpoint']
    if assets != {'schema': 'resource-bootstrap-assets-v1', 'model_id': checkpoint['model_id'],
                  'revision': checkpoint['revision'], 'assets': checkpoint['assets']}:
        raise ValueError('owner assets differ from the prelaunch freeze')
    expected_runtime, device = run_manifest['runtime'], run_manifest['device']
    if (runtime.get('schema') != 'agreement-calibration-runtime-v1'
            or any(runtime.get(name) != expected for name, expected in expected_runtime.items())
            or runtime.get('driver_version_provenance') != 'controller-prelaunch-nvidia-smi'
            or runtime.get('subprocess_policy') != 'python-audit-hook-denied'
            or runtime.get('device_uuid') != device['uuid'] or runtime.get('device_name') != device['name']
            or runtime.get('driver_version') != device['driver_version']
            or type(runtime.get('device_total_bytes')) is not int
            or abs(runtime['device_total_bytes'] - device['total_bytes']) > 1024**2):
        raise ValueError('owner runtime differs from the prelaunch freeze')
    names = {path.name for path in run_dir.glob('stage-*.json')}
    if names != {f'stage-{index:03}.json' for index in range(84)}:
        raise ValueError('private stage record sequence is incomplete')
    for name in names:
        _read_private(run_dir / name)


def _load_collector():
    # This import is deliberately deferred until the exact owner and device
    # lease have been checked; the collector may import CUDA-capable modules.
    from .agreement_calibration_owner import collect_owned
    return collect_owned


def _require_pristine_dependencies():
    packages = ('torch', 'transformers', 'numpy', 'huggingface_hub', 'safetensors', 'tokenizers')
    if any(name == package or name.startswith(package + '.')
           for name in tuple(sys.modules) for package in packages):
        raise RuntimeError('CUDA-capable dependencies were imported before the lease')


def _raise_controller_fatal(error):
    """Preserve interruption kind without exposing a private retained traceback."""
    try:
        if isinstance(error, KeyboardInterrupt):
            raise KeyboardInterrupt('supervised owner outcome is unknown') from None
        if isinstance(error, SystemExit):
            raise SystemExit('supervised owner outcome is unknown') from None
        raise BaseException('supervised owner outcome is unknown') from None
    except BaseException as public:
        public.__context__ = None
        raise


def owner_main(run_dir, cache_root, run_id, source_digest, directory_identity, run_manifest_sha256):
    """Run one child while retaining its device lease until process exit."""
    try:
        if not CONTROLLER_EXECUTION_READY:
            raise RuntimeError('agreement calibration controller is not yet enabled')
        if _OWNER['entered'] is not False:
            raise RuntimeError('agreement calibration owner is single-use')
        _OWNER['entered'] = True
        manifest, digest, _, site, _ = _bootstrap_state()
        if digest != source_digest or re.fullmatch(r'[0-9a-f]{32}', run_id) is None:
            raise RuntimeError('child source or run identity differs from controller')
        if re.fullmatch(r'[0-9]+:[0-9]+', directory_identity) is None:
            raise ValueError('invalid run directory identity')
        directory_identity = tuple(int(part) for part in directory_identity.split(':'))
        run_dir = Path(run_dir)
        _check_directory(run_dir, directory_identity)
        repo = Path(__file__).resolve().parents[1]
        cache = Path(cache_root).resolve(strict=True)
        if run_dir.is_relative_to(repo) or run_dir.is_relative_to(cache) or cache.is_relative_to(run_dir):
            raise ValueError('private run directory overlaps source or cache')
        run_manifest, run_manifest_bytes = _read_private(run_dir / 'run-manifest.json')
        if hashlib.sha256(run_manifest_bytes).hexdigest() != run_manifest_sha256:
            raise ValueError('prelaunch run freeze hash changed')
        _check_run_manifest(run_manifest, run_id, manifest, digest, run_dir, directory_identity, cache)
        _require_pristine_dependencies()
        session = OwnerSession.from_environment()
        _OWNER['session'] = session
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _OWNER['lease'] = lease
        OwnerSession.check(session, lease)
        if session.identity is None:
            raise RuntimeError('physical GPU identity is required')
        if session.identity.gpu_uuid != run_manifest['device'].get('uuid'):
            raise RuntimeError('supervised device differs from the prelaunch freeze')
        _write_private(run_dir / 'owner.json', {'schema': 'agreement-calibration-owner-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': manifest['commit'], 'source_sha256': digest})
        callback = _load_collector()
        candidate = callback(session, lease, repo_root=repo,
                             cache_root=cache, run_dir=run_dir, run_id=run_id,
                             run_dir_identity=directory_identity, run_manifest_bytes=run_manifest_bytes,
                             run_manifest_sha256=run_manifest_sha256)
        _OWNER['candidate'] = candidate
        measurement = _validate_candidate(candidate, run_id, run_manifest_sha256)
        _check_directory(run_dir, directory_identity)
        _validate_private_identity(run_dir, manifest, measurement, run_manifest, site)
        if session._runtime_verified is not True:
            raise RuntimeError('the child did not bind the CUDA runtime identity')
        OwnerSession.check(session, lease)
        candidate_hash = _write_private(run_dir / 'candidate.json', {
            'schema': 'agreement-calibration-owner-candidate-v1', 'run_id': run_id,
            'supervisor_run_id': session.run_id, 'pid': os.getpid(), 'value': candidate})
        _write_private(run_dir / 'final.json', {'schema': 'agreement-calibration-owner-final-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'candidate_sha256': candidate_hash, 'cleanup': candidate['cleanup']})
        # Nothing that can initiate CUDA follows this attestation. The
        # bootstrap exits with os._exit, retaining this lease until OS exit.
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as error:
        if _OWNER['failure'] is None:
            _OWNER['failure'] = error
        try:
            _write_private(Path(run_dir) / 'failure.json', {
                'schema': 'agreement-calibration-owner-failure-v1', 'run_id': run_id,
                'status': 'quarantined', 'error_type': type(error).__name__,
                'private_traceback': traceback.format_exc()[-8192:]})
        except BaseException:
            pass
        return 1


def _match_owner_records(run_dir, supervision, run_id, commit, source_digest, run_manifest_sha256):
    owner, _ = _read_private(run_dir / 'owner.json')
    candidate, candidate_raw = _read_private(run_dir / 'candidate.json')
    final, _ = _read_private(run_dir / 'final.json')
    for record in (owner, candidate, final):
        if (record.get('run_id') != run_id or record.get('supervisor_run_id') != supervision.run_id
                or type(record.get('pid')) is not int or record['pid'] != supervision.pid):
            raise ValueError('owner evidence does not match the reaped child')
    if (owner.get('schema') != 'agreement-calibration-owner-v1'
            or owner.get('source_commit') != commit or owner.get('source_sha256') != source_digest
            or candidate.get('schema') != 'agreement-calibration-owner-candidate-v1'
            or final.get('schema') != 'agreement-calibration-owner-final-v1'
            or final.get('candidate_sha256') != hashlib.sha256(candidate_raw).hexdigest()
            or final.get('cleanup') != candidate.get('value', {}).get('cleanup')):
        raise ValueError('owner evidence or cleanup hash mismatch')
    _validate_candidate(candidate['value'], run_id, run_manifest_sha256)
    return candidate['value']


def _owner_absence_confirmed(observation):
    """Require explicit post-exit telemetry; this is not process-tree proof."""
    return (type(observation) is dict and observation.get('query') == 'ok'
            and observation.get('owner_pid_listed') is False)


def run_calibration(*, workspace_root, cache_root, output, smi):
    """Run a single supervised owner and save only a bounded status record."""
    if not CONTROLLER_EXECUTION_READY:
        raise RuntimeError('agreement calibration controller is not yet enabled')
    if not _RUN_ONCE_LOCK.acquire(blocking=False):
        raise RuntimeError('agreement calibration allows one attempt per controller process')
    if any(_CONTROLLER.get(name) is not None for name in ('source_guard', 'asset_guard', 'process_job')):
        raise RuntimeError('a prior controller has retained source, asset or process custody')
    manifest, digest, git, site, recheck = _bootstrap_state()
    root, cache = Path(workspace_root).resolve(strict=True), Path(cache_root).resolve(strict=True)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError('output must be a new absolute file')
    work = root / 'work'
    if not work.is_dir():
        raise ValueError('workspace work directory is required')
    repo = Path(__file__).resolve(strict=True).parents[1]
    # Both checks happen before supervising or importing CUDA-capable modules.
    if recheck(repo, git)[1] != digest:
        raise RuntimeError('source changed after the controller bootstrap')
    from .trusted_service_profile import prepare_trusted_source
    source_preflight = prepare_trusted_source(str(repo), str(cache))
    from _agreement_source_guards import SourceFileLease
    source_guard = SourceFileLease(repo, [repo / name for name in manifest['files']])
    asset_root = cache / 'models--distilbert--distilgpt2' / 'snapshots' / '2290a62682d06624634c1f46a6ad5be0f47f38aa'
    asset_guard = SourceFileLease(cache, [asset_root / name for name, _ in source_preflight.asset_paths])
    _CONTROLLER['source_guard'], _CONTROLLER['asset_guard'] = source_guard, asset_guard
    try:
        source_guard.__enter__()
        asset_guard.__enter__()
        if recheck(repo, git)[1] != digest or prepare_trusted_source(str(repo), str(cache)) != source_preflight:
            raise RuntimeError('source or assets changed before protections were held')
        outcome, child_clean, fatal = _run_with_protected_source(root, cache, work, repo, site, git, recheck,
                                                                  manifest, digest, source_preflight, smi)
    except BaseException as error:
        _CONTROLLER['failure'] = error
        raise
    if child_clean:
        try:
            asset_guard.close()
            source_guard.close()
        except BaseException as error:
            # Retain every uncertain handle and never promote this run.
            _CONTROLLER['failure'] = error
            outcome['status'] = 'quarantined'
            outcome['protection_cleanup_error_type'] = type(error).__name__
            if not isinstance(error, Exception) and fatal is None:
                fatal = error
        else:
            _CONTROLLER['source_guard'] = _CONTROLLER['asset_guard'] = None
    # An unclean or unknown child leaves its source and asset protections
    # process-rooted until replacement; releasing them is not a reap proof.
    outcome['source_protections_retained'] = _CONTROLLER['source_guard'] is not None
    _write_private(destination, outcome)
    if fatal is not None:
        _raise_controller_fatal(fatal)
    return outcome


def _run_with_protected_source(root, cache, work, repo, site, git, recheck,
                               manifest, digest, source_preflight, smi):
    asset_manifest = json.loads(source_preflight.asset_manifest)
    versions = _dependency_versions(site)
    gpu_uuid, before = _smi_snapshot(smi)
    environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATHEXT', 'COMSPEC')
                   if key in os.environ}
    system_root = Path(environment.get('SYSTEMROOT', 'C:/Windows'))
    environment.update({'PATH': str(system_root / 'System32'), 'CUDA_VISIBLE_DEVICES': gpu_uuid,
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HOME': str(cache.parent), 'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    run_dir = Path(tempfile.mkdtemp(prefix='agreement-calibration-', dir=work))
    directory_identity = _directory_identity(run_dir)
    if run_dir.is_relative_to(repo) or run_dir.is_relative_to(cache) or cache.is_relative_to(run_dir):
        raise ValueError('private run directory overlaps source or cache')
    prefix = Path(tempfile.mkdtemp(prefix='pycache-', dir=run_dir))
    environment['PYTHONPYCACHEPREFIX'] = str(prefix)
    run_id = secrets.token_hex(16)
    _write_private(run_dir / 'controller-source.json', manifest)
    _write_private_bytes(run_dir / 'controller-source-base.json', source_preflight.source_base)
    run_manifest = {'schema': 'agreement-calibration-run-freeze-v1', 'run_id': run_id,
        'source': {'commit': manifest['commit'], 'sha256': digest, 'checkout': str(repo),
                   'source_base_sha256': hashlib.sha256(source_preflight.source_base).hexdigest()},
        'checkpoint': {'model_id': asset_manifest['model_id'], 'revision': asset_manifest['revision'],
                       'cache_root': str(cache), 'assets': asset_manifest['assets']},
        'runtime': versions,
        'device': {'uuid': gpu_uuid, 'name': before['device_name'], 'total_bytes': before['total_bytes'],
                   'driver_version': before['driver_version']},
        'workload': _workload(), 'directory': {'path': str(run_dir), 'device': directory_identity[0],
                                                'inode': directory_identity[1]}}
    run_manifest_sha256 = _write_private(run_dir / 'run-manifest.json', run_manifest)
    _write_private(run_dir / 'started.json', {'schema': 'agreement-calibration-start-v1',
        'run_id': run_id, 'source_commit': manifest['commit'], 'source_sha256': digest,
        'run_manifest_sha256': run_manifest_sha256, 'started_utc': before['timestamp_utc']})
    bootstrap = repo / 'scripts' / 'agreement_calibration_bootstrap.py'
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(bootstrap), '--bootstrap-site', str(site), '--git-executable', str(git),
               '--owner', str(run_dir), str(cache), run_id, digest,
               f'{directory_identity[0]}:{directory_identity[1]}', run_manifest_sha256]
    supervision, failure, fatal = None, None, None
    try:
        # The supervisor receives this exact, private single-run capability.
        # Root the object before initialization can create a native job handle.
        from _agreement_source_guards import WindowsJob
        process_job = WindowsJob.__new__(WindowsJob)
        _CONTROLLER['process_job'] = process_job
        WindowsJob.__init__(process_job)
        _CONTROLLER['launch_started'] = True
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, timeout=900, lease_timeout=30,
            cwd=repo, env=environment, inherit_environment=False, process_job=process_job)
    except BaseException as error:
        _CONTROLLER['failure'] = error
        if not isinstance(error, Exception):
            fatal = error
        failure = type(error).__name__
    outcome = {'schema': 'agreement-calibration-controller-v1', 'run_id': run_id,
        'source_commit': manifest['commit'], 'source_sha256': digest,
        'run_manifest_sha256': run_manifest_sha256, 'private_run_dir': str(run_dir),
        'nvidia_smi_before': before, 'nvidia_smi_after': None,
        'status': 'quarantined', 'supervisor_error_type': failure,
        'supervisor': None, 'owner_presence_after_exit': {'query': 'not_attempted', 'owner_pid_listed': None},
        'limitations': ['resource-only pilot; no behavioral or held-out result',
                        'measurements and cooperative commitments are not worst-case memory bounds',
                        'CUDA context, native allocations and unrelated GPU processes are not bounded',
                        'the supervised Job does not attest unrelated or escaped processes',
                        'dependency and native files require a trusted, stable environment']}
    if supervision is not None:
        outcome['supervisor'] = {'pid': supervision.pid, 'run_id': supervision.run_id,
            'exit_code': supervision.exit_code, 'clean': supervision.clean, 'timed_out': supervision.timed_out}
        outcome['owner_presence_after_exit'] = _owner_presence_after_exit(smi, supervision.pid)
        _write_private(run_dir / 'supervision.json', {'run_id': run_id, **outcome['supervisor']})
        try:
            after_uuid, after = _smi_snapshot(smi)
            if (after_uuid != gpu_uuid or after['driver_version'] != before['driver_version']
                    or after['device_name'] != before['device_name']
                    or after['total_bytes'] != before['total_bytes']):
                raise ValueError('physical inventory changed after owner exit')
            outcome['nvidia_smi_after'] = after
        except Exception as error:
            outcome['post_telemetry_error_type'] = type(error).__name__
        if supervision.clean and supervision.exit_code == 0 and not supervision.timed_out:
            outcome['status'] = 'incomplete'
            try:
                _check_directory(run_dir, directory_identity)
                value = _match_owner_records(run_dir, supervision, run_id, manifest['commit'], digest,
                                             run_manifest_sha256)
                _validate_private_identity(run_dir, manifest, value['measurement'], run_manifest, site)
                if recheck(repo, git)[1] != digest:
                    raise ValueError('source changed while the child was running')
                if (outcome['nvidia_smi_after'] is None or
                        not _owner_absence_confirmed(outcome['owner_presence_after_exit'])):
                    raise ValueError('post-exit inventory does not confirm the expected owner state')
                # A separately reviewed auditor must still validate the 84
                # stages, key/source provenance and both profile calculations.
                outcome['owner_evidence'] = value['measurement']
                outcome['status'] = 'awaiting_independent_audit'
            except Exception as error:
                outcome['candidate_error_type'] = type(error).__name__
    child_clean = supervision is not None and supervision.clean is True
    if child_clean:
        # The supplied supervisor contract returns clean only after the job's
        # full lifetime accounting, drain and native-handle close succeed.
        _CONTROLLER['process_job'] = None
    return outcome, child_clean, fatal


def main(argv=None):
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', nargs=6, metavar=('RUN_DIR', 'CACHE', 'RUN_ID', 'SOURCE_SHA', 'DIR_ID', 'FREEZE_SHA'),
                        help=argparse.SUPPRESS)
    parser.add_argument('--workspace-root')
    parser.add_argument('--cache-root')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.owner:
        return owner_main(*args.owner)
    if not args.workspace_root or not args.cache_root or not args.output:
        parser.error('--workspace-root, --cache-root and --output are required')
    result = run_calibration(workspace_root=args.workspace_root, cache_root=args.cache_root,
                             output=args.output, smi=shutil.which('nvidia-smi'))
    print(json.dumps({'status': result['status'], 'source_commit': result['source_commit']}))
    return 0 if result['status'] == 'awaiting_independent_audit' else 1
