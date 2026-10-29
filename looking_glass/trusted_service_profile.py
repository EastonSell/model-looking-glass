"""Narrow, trusted-process identity for the fixed two-token service pilot.

The owner must run the recorded source in a trusted, stable Python/native
environment. File hashes and runtime versions do not authenticate every byte
already loaded into the interpreter. A measured estimate is not a VRAM bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys

from .live_profile import (_canonical_json_bytes, _guard_owner, _load_torch,
                           _make_workload_keys, _measure_runtime, _plain_file_hash,
                           collect_offline_assets, SourcePreflight, verify_loaded_model)
from .resource_profile import ProfileRegistry, canonical_fingerprint, resolve_worker_profile


LOAD_CASE = {'schema': 'resource-bootstrap-load-case-v1', 'operation': 'model_load',
             'checkpoint_sha256': 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
             'config_sha256': '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a'}
FORWARD_CASE = {'schema': 'resource-bootstrap-case-v1', 'input_ids': [15496, 995],
                'shape': [1, 2], 'dtype': 'torch.int64'}
LOAD_CASE_SHA256 = canonical_fingerprint(LOAD_CASE)
FORWARD_CASE_SHA256 = canonical_fingerprint(FORWARD_CASE)
LOAD_COMMITMENT = 2 * 1024 ** 3
JOB_COMMITMENT = 512 * 1024 ** 2
DIRECT_EXECUTION_PATH = 'looking_glass.resident_worker.ResidentWorker.execute'
_FIXED_ENV = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
              'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
_ABSENT_ENV = ('PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING', 'CUDA_LAUNCH_BLOCKING')
_ENV_PREFIXES = ('CUDA_', 'PYTORCH_', 'TORCH_', 'CUBLAS_', 'OMP_', 'MKL_', 'OPENBLAS_',
                 'HF_', 'HUGGINGFACE_', 'TRANSFORMERS_')


@dataclass(frozen=True)
class TrustedServiceSource:
    repo_root: str
    cache_root: str
    source_base: bytes
    asset_manifest: bytes
    asset_paths: tuple


@dataclass(frozen=True)
class TrustedServiceIdentity:
    keys: tuple
    source_manifest: bytes
    runtime_manifest: bytes
    asset_manifest: bytes


def _plain_directory(path):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('source directory must be an ordinary directory')


def _collect_source_files(repo_root):
    """Hash actual package Python files and the versioned protocol."""
    root = Path(repo_root).resolve(strict=True)
    package = root / 'looking_glass'
    _plain_directory(package)
    files = []
    def failed_walk(error):
        raise ValueError('source inventory could not read a directory') from error
    for base, dirs, names in os.walk(package, followlinks=False, onerror=failed_walk):
        dirs[:] = sorted(name for name in dirs if name != '__pycache__')
        for name in dirs:
            _plain_directory(Path(base) / name)
        for name in sorted(names):
            if name.lower().endswith('.py'):
                path = Path(base) / name
                if path.lstat().st_size > 16 * 1024 ** 2:
                    raise ValueError('source file exceeds the inventory limit')
                files.append({'path': path.relative_to(root).as_posix(), **_plain_file_hash(path)})
                if len(files) > 4096:
                    raise ValueError('source inventory exceeds the file limit')
    protocol = root / 'docs' / 'TRUSTED_SERVICE_PROFILE_V1.md'
    _plain_directory(protocol.parent)
    if protocol.lstat().st_size > 16 * 1024 ** 2:
        raise ValueError('protocol exceeds the inventory limit')
    files.append({'path': protocol.relative_to(root).as_posix(), **_plain_file_hash(protocol)})
    files.sort(key=lambda item: item['path'])
    if 'looking_glass/__init__.py' not in {item['path'] for item in files}:
        raise ValueError('selected source package is incomplete')
    return files


def prepare_trusted_source(repo_root, cache_root):
    """CPU-only preflight of the selected code snapshot and seven offline assets."""
    repo, cache = Path(repo_root), Path(cache_root)
    if not repo.is_absolute() or not cache.is_absolute():
        raise ValueError('source and checkpoint paths must be absolute')
    repo, cache = repo.resolve(strict=True), cache.resolve(strict=True)
    if Path(__file__).resolve(strict=True) != repo / 'looking_glass' / 'trusted_service_profile.py':
        raise ValueError('profile helper did not come from the selected source snapshot')
    files = _collect_source_files(repo)
    assets, paths = collect_offline_assets(repo, cache)
    source_base = {'schema': 'trusted-service-source-base-v1', 'repository_files': files,
                   'python_executable': _plain_file_hash(Path(sys._base_executable).resolve(strict=True)),
                   'assets': assets}
    return TrustedServiceSource(str(repo), str(cache), _canonical_json_bytes(source_base),
                                _canonical_json_bytes(assets), paths)


def verify_trusted_cpu_model(preflight, model):
    """Revalidate the source and verify the pinned model before device copy."""
    if type(preflight) is not TrustedServiceSource:
        raise TypeError('a service-owned trusted source preflight is required')
    if prepare_trusted_source(preflight.repo_root, preflight.cache_root) != preflight:
        raise ValueError('source or offline assets changed before model copy')
    source = SourcePreflight(preflight.repo_root, preflight.cache_root, '', '',
                             preflight.source_base, preflight.asset_manifest, preflight.asset_paths)
    if verify_loaded_model(source, model) is not True:
        raise ValueError('model verification did not explicitly succeed')
    return True


def verify_trusted_device_tie(model, identity, owner_session, device_lease):
    """Check the post-copy tie after the owner has synchronized the device."""
    _guard_owner(identity, owner_session, device_lease)
    torch = _load_torch()
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise ValueError('model conversion flags differ from the fixed tie policy')
    embedding = model.get_parameter('transformer.wte.weight')
    head = model.get_parameter('lm_head.weight')
    if (head is not embedding or type(embedding) is not torch.nn.Parameter
            or embedding.device.type != 'cuda' or embedding.device.index != 0
            or embedding.dtype is not torch.float32):
        raise ValueError('device model did not preserve its tied embedding Parameter')
    pointer = embedding.untyped_storage().data_ptr()
    if type(pointer) is not int or pointer <= 0 or head.untyped_storage().data_ptr() != pointer:
        raise ValueError('device model did not preserve tied storage')
    _guard_owner(identity, owner_session, device_lease)
    return True


def _checked_environment(device_uuid, cache_root):
    hf_home = os.environ.get('HF_HOME')
    if type(hf_home) is not str or not hf_home or type(cache_root) is not str or not cache_root:
        raise ValueError('HF_HOME and the selected cache must be explicit canonical paths')
    try:
        home, cache = Path(hf_home), Path(cache_root)
        if not home.is_absolute() or not cache.is_absolute():
            raise ValueError('HF_HOME and the selected cache must be absolute')
        resolved_home, resolved_cache = home.resolve(strict=True), cache.resolve(strict=True)
        if (str(resolved_home) != hf_home or str(resolved_cache) != cache_root
                or not resolved_home.is_dir() or not resolved_cache.is_dir()
                or resolved_cache == resolved_home or not resolved_cache.is_relative_to(resolved_home)):
            raise ValueError('HF_HOME must canonically contain the selected cache')
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError('HF_HOME must canonically contain the selected cache') from exc
    required = {'CUDA_VISIBLE_DEVICES': device_uuid, 'HF_HOME': str(resolved_home), **_FIXED_ENV}
    for key, value in required.items():
        if os.environ.get(key) != value:
            raise ValueError(f'required runtime environment changed: {key}')
    for key in os.environ:
        upper = key.upper()
        if upper.startswith(_ENV_PREFIXES) and (key != upper or upper not in required):
            raise ValueError('unreviewed CUDA, numerical or model-cache environment setting')
    if any(name in os.environ for name in _ABSENT_ENV):
        raise ValueError('incompatible runtime environment is present')
    program_files = os.environ.get('PROGRAMFILES')
    if program_files is not None:
        try:
            directory = Path(program_files)
            if (not directory.is_absolute() or str(directory.resolve(strict=True)) != program_files
                    or not directory.is_dir()):
                raise ValueError('PROGRAMFILES must be a canonical directory')
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError('PROGRAMFILES must be a canonical directory') from exc
    return {**required, 'PROGRAMFILES': program_files, **{name: None for name in _ABSENT_ENV}}


def _module_origin(module, expected_name):
    name, version, location = getattr(module, '__name__', None), getattr(module, '__version__', None), getattr(module, '__file__', None)
    if name != expected_name or not isinstance(version, str) or not version or type(location) is not str:
        raise ValueError('loaded dependency identity is unavailable')
    path = Path(location).resolve(strict=True)
    return {'name': name, 'version': str(version), 'path': str(path), **_plain_file_hash(path)}


def _load_transformers():
    return importlib.import_module('transformers')


def _effective_hf_settings(hf_home):
    """Inspect settings cached by the imported Hub and Transformers modules."""
    hub = importlib.import_module('huggingface_hub.constants')
    transformers_hub = importlib.import_module('transformers.utils.hub')
    cache = str(Path(hf_home) / 'hub')
    modules = str(Path(hf_home) / 'modules')
    fields = ((hub, 'HF_HOME', hf_home), (hub, 'HF_HUB_CACHE', cache),
              (transformers_hub, 'TRANSFORMERS_CACHE', cache),
              (transformers_hub, 'HF_MODULES_CACHE', modules))
    if any(type(getattr(obj, name, None)) is not str or getattr(obj, name) != value
           for obj, name, value in fields):
        raise ValueError('effective Hugging Face cache settings differ')
    if (getattr(hub, 'HF_HUB_OFFLINE', None) is not True
            or getattr(hub, 'HF_HUB_DISABLE_TELEMETRY', None) is not True
            or getattr(transformers_hub, '_is_offline_mode', None) is not True):
        raise ValueError('effective Hugging Face offline settings differ')
    return {'hf_home': hf_home, 'hf_hub_cache': cache, 'transformers_cache': cache,
            'hf_modules_cache': modules, 'hub_offline': True,
            'transformers_offline': True, 'telemetry_disabled': True}


def _resolved_import_paths(entries, cwd):
    root = Path(cwd)
    if not root.is_absolute() or any(type(value) is not str or '\x00' in value for value in entries):
        raise ValueError('interpreter import paths are unavailable')
    return [str((root / value).resolve(strict=False)) if not Path(value).is_absolute()
            else str(Path(value).resolve(strict=False)) for value in entries]


def build_trusted_service_identity(preflight, identity, owner_session, device_lease):
    """Recompute identity under a real supervised device owner before model copy."""
    _guard_owner(identity, owner_session, device_lease)
    if type(preflight) is not TrustedServiceSource:
        raise TypeError('a service-owned trusted source preflight is required')
    environment = _checked_environment(identity.gpu_uuid, preflight.cache_root)
    current = prepare_trusted_source(preflight.repo_root, preflight.cache_root)
    if current != preflight:
        raise ValueError('source or offline assets changed after preflight')
    _guard_owner(identity, owner_session, device_lease)
    torch = _load_torch()
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise ValueError('model conversion flags differ from the fixed tie policy')
    _guard_owner(identity, owner_session, device_lease)
    runtime = _measure_runtime(torch, identity, owner_session, device_lease)
    _guard_owner(identity, owner_session, device_lease)
    transformers = _load_transformers()
    effective_hf_settings = _effective_hf_settings(environment['HF_HOME'])
    dependencies = [_module_origin(torch, 'torch'), _module_origin(transformers, 'transformers')]
    source = {'schema': 'trusted-service-source-v1', 'base': json.loads(preflight.source_base),
              'dependency_origins': dependencies}
    source_bytes = _canonical_json_bytes(source)
    runtime.update({'schema': 'trusted-service-runtime-v1', 'python_version': sys.version.split()[0],
                    'transformers_version': str(transformers.__version__),
                    'environment': environment,
                    'effective_huggingface_settings': effective_hf_settings,
                    'python_import_paths': _resolved_import_paths(sys.path, Path.cwd().resolve()),
                    'overwrite_module_params_on_conversion': False,
                    'swap_module_params_on_conversion': False})
    runtime_bytes = _canonical_json_bytes(runtime)
    _guard_owner(identity, owner_session, device_lease)
    keys = _make_workload_keys(hashlib.sha256(source_bytes).hexdigest(), runtime)
    return TrustedServiceIdentity(keys, source_bytes, runtime_bytes, preflight.asset_manifest)


def _decode_canonical(data, schema):
    if type(data) is not bytes or not 0 < len(data) <= 32 * 1024 ** 2:
        raise ValueError('identity manifest has an invalid size')
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('identity manifest is invalid JSON') from exc
    if type(result) is not dict or result.get('schema') != schema or _canonical_json_bytes(result) != data:
        raise ValueError('identity manifest is not canonical or has the wrong schema')
    return result


def resolve_trusted_service_profiles(registry, identity, approved_profile_ids):
    """Resolve exact pinned profiles and the prescribed case/seed evidence."""
    return _resolve_service_profiles(registry, identity, approved_profile_ids,
                                     'trusted-service-runtime-v1', DIRECT_EXECUTION_PATH)


def _resolve_service_profiles(registry, identity, approved_profile_ids, runtime_schema, execution_path):
    """Shared case/seed/pin checks; callers must validate their protocol first."""
    if type(registry) is not ProfileRegistry or type(identity) is not TrustedServiceIdentity:
        raise TypeError('a trusted registry and independently built service identity are required')
    if (type(approved_profile_ids) is not tuple or len(approved_profile_ids) != 3
            or len(set(approved_profile_ids)) != 3):
        raise ValueError('three separate approved profile pins are required')
    source = _decode_canonical(identity.source_manifest, 'trusted-service-source-v1')
    runtime = _decode_canonical(identity.runtime_manifest, runtime_schema)
    if runtime.get('execution_path') != execution_path:
        raise ValueError('runtime execution path differs from the selected protocol')
    if runtime_schema == 'trusted-service-runtime-v1' and 'custody_execution' in runtime:
        raise ValueError('direct-worker identity cannot include custody execution')
    if (type(source.get('base')) is not dict or 'assets' not in source['base']
            or _canonical_json_bytes(source['base']['assets']) != identity.asset_manifest):
        raise ValueError('recorded asset manifest does not match the source identity')
    source_hash = hashlib.sha256(identity.source_manifest).hexdigest()
    if identity.keys != _make_workload_keys(source_hash, runtime):
        raise ValueError('service keys do not match the live manifests')
    profiles = tuple(registry.resolve(key) for key in identity.keys)
    expected = (((17,), LOAD_CASE_SHA256, 512 * 1024 ** 2, LOAD_COMMITMENT),
                ((17,), FORWARD_CASE_SHA256, 256 * 1024 ** 2, JOB_COMMITMENT),
                ((29, 43, 71), FORWARD_CASE_SHA256, 256 * 1024 ** 2, JOB_COMMITMENT))
    for profile, pin, (seeds, case_hash, margin, commitment) in zip(profiles, approved_profile_ids, expected):
        if (profile.profile_id != pin or profile.provenance.source_manifest_sha256 != source_hash
                or tuple(sample.seed for sample in profile.samples) != seeds
                or any(sample.case_sha256 != case_hash for sample in profile.samples)
                or profile.margin_bytes != margin or profile.margin_bps != 2500
                or profile.estimated_incremental_bytes > commitment):
            raise ValueError('profile pin, case, seed, margin, source or commitment differs from protocol')
    return (profiles[0], resolve_worker_profile(registry, identity.keys[1]),
            resolve_worker_profile(registry, identity.keys[2]))


def validate_service_request(inputs, seed, *, cold):
    """Check already service-owned inputs; caller must preserve ownership."""
    _check_seed(seed, cold=cold)
    if type(inputs) is not dict or set(inputs) != {'input_ids'}:
        raise ValueError('request must contain only input_ids')
    torch = _load_torch()
    value = inputs['input_ids']
    if (type(value) is not torch.Tensor or value.device.type != 'cpu' or value.dtype is not torch.int64
            or tuple(value.shape) != (1, 2) or not value.is_contiguous()
            or value.tolist() != [[15496, 995]]):
        raise ValueError('request does not match the fixed token case')
    return True


def _check_seed(seed, *, cold):
    if type(cold) is not bool or type(seed) is not int or seed not in ((17,) if cold else (29, 43, 71)):
        raise ValueError('request seed does not match the fixed stage')


def make_fixed_service_inputs(raw_ids, seed, *, cold):
    """Construct a fresh CPU tensor after validating the decoded token values.

    The trusted request factory must retain exclusive ownership of the result
    until the worker takes its own snapshot; it must not expose the tensor to
    a caller that can mutate it concurrently.
    """
    _check_seed(seed, cold=cold)
    if type(raw_ids) not in (list, tuple):
        raise ValueError('decoded token IDs must be a list or tuple')
    values = tuple(raw_ids)
    if len(values) != 2 or any(type(value) is not int for value in values) or values != (15496, 995):
        raise ValueError('decoded token IDs differ from the fixed case')
    torch = _load_torch()
    return {'input_ids': torch.tensor([[15496, 995]], dtype=torch.int64, device='cpu')}
