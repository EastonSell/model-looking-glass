"""Live identity for the separate five-token agreement resource pilot.

Importing this module does not import Torch. Runtime collection is valid only
inside a fresh, controlled child after supervised device ownership is held.
"""
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys

from .live_profile import (_canonical_json_bytes, _guard_owner,
                           _plain_file_hash, decode_cuda_uuid)
from .pretrained import ASSETS, REVISION
from .resource_profile import WorkloadKey, canonical_fingerprint, model_runtime_fingerprint
from .supervisor import OwnerSession
from .trusted_service_profile import (TrustedServiceSource, _checked_environment,
    _effective_hf_settings, _module_origin, _resolved_import_paths, prepare_trusted_source)


_PROTOCOLS = ('docs/AGREEMENT_GPU_CALIBRATION_V1.md', 'docs/AGREEMENT_CUDA_ADAPTER.md',
              'docs/AGREEMENT_CALIBRATION_SAMPLING_DRAFT.md')
_EXECUTION_PATH = 'looking_glass.agreement_cuda_calibration.CudaAgreementCalibrator'


class _SpawnGuardProbe(RuntimeError):
    pass


def _deny_owner_descendants(event, args):
    if event == 'looking_glass.agreement_calibration.spawn_guard_probe':
        raise _SpawnGuardProbe('owner spawn guard is active')
    if event in ('subprocess.Popen', '_winapi.CreateProcess', 'os.system', 'os.spawn',
                 'os.posix_spawn', 'os.exec', 'os.fork', 'os.forkpty', 'os.startfile', 'os.startfile/2'):
        raise RuntimeError('agreement owner subprocess creation is prohibited')


def _install_spawn_free_guard():
    """Avoid Windows platform probes and forbid Python-created descendants.

    This process-lifetime audit hook is a cooperative Python boundary, not a
    guarantee that a native library cannot create a process. The external Job
    supervisor remains responsible for descendant accounting.
    """
    if os.name != 'nt' or not hasattr(sys, 'getwindowsversion'):
        raise ValueError('the reviewed agreement owner requires Windows')
    import platform
    windows = sys.getwindowsversion()
    version = f'{windows.major}.{windows.minor}.{windows.build}'
    platform._syscmd_ver = lambda *args, **kwargs: ('Windows', str(windows.major), version)
    sys.addaudithook(_deny_owner_descendants)
    try:
        sys.audit('looking_glass.agreement_calibration.spawn_guard_probe')
    except _SpawnGuardProbe:
        return
    raise ValueError('agreement owner spawn guard did not activate')


def _check_torch_before_cuda(torch, expected):
    if (type(expected) is not dict or expected.get('torch_version') != '2.8.0+cu128'
            or expected.get('cuda_runtime_version') != '12.8'
            or str(getattr(torch, '__version__', '')) != expected['torch_version']
            or str(getattr(getattr(torch, 'version', None), 'cuda', '')) != expected['cuda_runtime_version']):
        raise ValueError('Torch build differs from the frozen runtime before CUDA initialization')


def _check_import_preconditions():
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.flags.safe_path or not sys.dont_write_bytecode
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or not Path(sys.pycache_prefix).is_dir() or any(Path(sys.pycache_prefix).iterdir())):
        raise ValueError('fresh isolated import settings are required')
    if any(name == package or name.startswith(package + '.') for name in tuple(sys.modules)
           for package in ('torch', 'transformers', 'huggingface_hub', 'numpy', 'safetensors', 'tokenizers')):
        raise ValueError('model dependencies were imported before the owner')


def _verify_dependency_origins(repo_root):
    """Check the isolated interpreter's selected site before dependency import."""
    value = os.environ.get('LOOKING_GLASS_AGREEMENT_BOOTSTRAP_SITE')
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise ValueError('the bootstrap must specify an absolute site-packages directory')
    site, repo = Path(value).resolve(strict=True), Path(repo_root).resolve(strict=True)
    base = Path(sys.base_prefix).resolve(strict=True)
    if str(site) != value or not site.is_dir() or site == repo or site.is_relative_to(repo):
        raise ValueError('the selected dependency site is invalid')
    paths = []
    for raw in sys.path:
        if type(raw) is not str or not raw or not Path(raw).is_absolute():
            raise ValueError('relative or empty Python import path is unsupported')
        path = Path(raw).resolve(strict=False)
        if path == repo or path.is_relative_to(repo):
            raise ValueError('the checkout must not be a generic import root')
        if path != site and path != base and not path.is_relative_to(base):
            raise ValueError('unapproved Python import root')
        paths.append(path)
    if paths.count(site) != 1:
        raise ValueError('the selected dependency site must appear exactly once')
    for package in ('torch', 'transformers', 'huggingface_hub', 'numpy', 'safetensors', 'tokenizers'):
        spec = importlib.machinery.PathFinder.find_spec(package)
        if (spec is None or type(spec.origin) is not str
                or not isinstance(spec.loader, (importlib.machinery.SourceFileLoader,
                                                importlib.machinery.ExtensionFileLoader))
                or not Path(spec.origin).resolve(strict=True).is_relative_to(site)):
            raise ValueError('dependency does not resolve to the selected site: ' + package)
    return str(site)


def _verify_frozen_import_finders():
    """Repeat the bootstrap finder check immediately before Torch import."""
    package = sys.modules.get('looking_glass')
    validator = getattr(package, '__agreement_validate_import_finders__', None)
    if not callable(validator):
        raise ValueError('the frozen import-finder validator is unavailable')
    validator()


def _numeric_observation(torch):
    """Observe calling-thread arithmetic without changing its current mode."""
    bits = torch.tensor([0x00800000], dtype=torch.int32, device='cpu')
    product = int((bits.view(torch.float32) * 0.5).view(torch.int32).item())
    if product not in (0, 0x00400000):
        raise ValueError('unsupported calling-thread denormal observation')
    return {'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'calling_thread_denormal_product_bits': product,
            'cpu_autocast': torch.is_autocast_enabled('cpu')}


def _make_key(source_bytes, runtime):
    """Construct the fixed key from canonical source and observed runtime data."""
    from .agreement_cuda_runner import (CAPTURE_POLICY_SHA256, WARMUP_POLICY_SHA256,
                                         OUTPUT_BYTES, TRACE_BYTES, _validate_workload_key)
    if type(source_bytes) is not bytes or not source_bytes or type(runtime) is not dict:
        raise ValueError('source bytes and a runtime record are required')
    total = runtime.get('device_total_bytes')
    if (runtime.get('schema') != 'agreement-calibration-runtime-v1'
            or runtime.get('execution_path') != _EXECUTION_PATH
            or runtime.get('allocator_backend') != 'native' or runtime.get('allocator_fraction') != '0.25'
            or type(total) is not int or total <= 0 or runtime.get('allocator_limit_bytes') != total // 4
            or runtime.get('torch_version') != '2.8.0+cu128'
            or runtime.get('cuda_runtime_version') != '12.8'):
        raise ValueError('runtime does not match the fixed agreement calibration')
    fields = {'checkpoint_sha256': ASSETS['model.safetensors'][1], 'checkpoint_revision': REVISION,
              'model_config_sha256': ASSETS['config.json'][1],
              'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'runtime_settings_sha256': canonical_fingerprint(runtime)}
    for name in ('torch_version', 'transformers_version', 'cuda_runtime_version', 'driver_version',
                 'python_version', 'device', 'device_uuid', 'device_name', 'device_total_bytes',
                 'model_dtype', 'compute_dtype', 'attention_backend', 'model_mode', 'grad_mode',
                 'autocast', 'use_cache', 'compiled'):
        fields[name] = runtime[name]
    fields['compute_capability'] = tuple(runtime['compute_capability'])
    key = WorkloadKey(model_id=model_runtime_fingerprint(fields), **fields,
        warmup_state='warm', warmup_protocol_sha256=WARMUP_POLICY_SHA256,
        operation='forward', capture_mode='direct', capture_policy_sha256=CAPTURE_POLICY_SHA256,
        batch_size=1, sequence_length=5,
        input_dtypes=(('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64')),
        output_limit_bytes=OUTPUT_BYTES, max_gpu_trace_bytes=TRACE_BYTES, max_host_trace_bytes=TRACE_BYTES)
    _validate_workload_key(key)
    return key


def prepare_agreement_runtime(source, session, lease, *, frozen_runtime, frozen_driver_version):
    """Observe and configure the 25% runtime after acquiring the real lease.

    This function must be the child's first Torch/Transformers/Hub import. Its
    return does not prove future device allocation bounds or authenticate OS
    libraries; the caller must retain the owner on any failure.
    """
    if type(source) is not TrustedServiceSource or session.identity is None:
        raise ValueError('a source preflight and supervised physical identity are required')
    identity = session.identity
    _guard_owner(identity, session, lease)
    _check_import_preconditions()
    selected_site = Path(_verify_dependency_origins(source.repo_root))
    _install_spawn_free_guard()
    environment = _checked_environment(identity.gpu_uuid, source.cache_root)
    if prepare_trusted_source(source.repo_root, source.cache_root) != source:
        raise ValueError('source or checkpoint assets changed after preflight')
    _guard_owner(identity, session, lease)
    _verify_frozen_import_finders()
    torch = importlib.import_module('torch')
    if not Path(torch.__file__).resolve(strict=True).is_relative_to(selected_site):
        raise ValueError('loaded Torch origin differs from the selected site')
    _check_torch_before_cuda(torch, frozen_runtime)
    _guard_owner(identity, session, lease)
    if torch.cuda.is_initialized() is not False:
        raise ValueError('CUDA was initialized before binding the supervised owner')
    def checked(function, *args, **kwargs):
        _guard_owner(identity, session, lease)
        value = function(*args, **kwargs)
        _guard_owner(identity, session, lease)
        return value
    count = checked(torch.cuda.device_count)
    if type(count) is not int or count != 1:
        raise ValueError('exactly one visible CUDA device is required')
    props = checked(torch.cuda.get_device_properties, 0)
    runtime_uuid = decode_cuda_uuid(props.uuid, torch._C._CUuuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid,
                                        device_count=count, cuda_device_index=0)
    total = props.total_memory
    if type(total) is not int or not 7 * 1024 ** 3 <= total <= 9 * 1024 ** 3:
        raise ValueError('unexpected device memory total')
    if checked(torch.cuda.get_allocator_backend) != 'native':
        raise ValueError('native CUDA allocator is required')
    free, actual_total = checked(torch.cuda.mem_get_info, 0)
    if actual_total != total or type(free) is not int or free < 3 * 1024 ** 3 or free > total:
        raise ValueError('initial device free memory is insufficient or inconsistent')
    checked(torch.cuda.set_per_process_memory_fraction, 0.25, 0)
    if checked(torch.cuda.get_per_process_memory_fraction, 0) != (total // 4) / total:
        raise ValueError('allocator fraction readback differs from 25%')
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.set_flush_denormal(False):
        raise ValueError('CPU denormal mode could not be set')
    _guard_owner(identity, session, lease)
    modules = {'torch': torch}
    for name in ('transformers', 'numpy', 'huggingface_hub', 'safetensors', 'tokenizers'):
        _guard_owner(identity, session, lease)
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve(strict=True).is_relative_to(selected_site):
            raise ValueError('loaded dependency origin differs from the selected site: ' + name)
        modules[name] = module
    transformers = modules['transformers']
    effective_hf = _effective_hf_settings(environment['HF_HOME'])
    source_record = {'schema': 'agreement-calibration-source-v1', 'base': json.loads(source.source_base),
        'dependency_origins': [_module_origin(modules[name], name) for name in
            ('torch', 'transformers', 'numpy', 'huggingface_hub', 'safetensors', 'tokenizers')],
        'protocols': [{'path': name, **_plain_file_hash(Path(source.repo_root) / name)} for name in _PROTOCOLS]}
    source_bytes = _canonical_json_bytes(source_record)
    if type(frozen_driver_version) is not str or not frozen_driver_version:
        raise ValueError('the controller driver observation is unavailable')
    driver_version = frozen_driver_version
    runtime = {'schema': 'agreement-calibration-runtime-v1', 'execution_path': _EXECUTION_PATH,
        'device': 'cuda:0', 'device_uuid': runtime_uuid, 'device_name': props.name,
        'device_total_bytes': total, 'compute_capability': [props.major, props.minor],
        'torch_version': str(torch.__version__), 'transformers_version': str(transformers.__version__),
        'numpy_version': str(modules['numpy'].__version__),
        'huggingface_hub_version': str(modules['huggingface_hub'].__version__),
        'safetensors_version': str(modules['safetensors'].__version__),
        'tokenizers_version': str(modules['tokenizers'].__version__),
        'cuda_runtime_version': str(torch.version.cuda), 'driver_version': driver_version,
        'driver_version_provenance': 'controller-prelaunch-nvidia-smi',
        'subprocess_policy': 'python-audit-hook-denied',
        'python_version': sys.version.split()[0], 'model_dtype': 'torch.float32',
        'compute_dtype': 'torch.float32', 'attention_backend': 'eager', 'model_mode': 'eval',
        'grad_mode': 'inference', 'autocast': 'disabled', 'use_cache': False, 'compiled': False,
        'allocator_backend': 'native', 'allocator_fraction': '0.25', 'allocator_limit_bytes': total // 4,
        'environment': environment,
        'effective_huggingface_settings': effective_hf,
        'torch_build_sha256': hashlib.sha256(torch.__config__.show().encode('utf-8')).hexdigest(),
        'python_import_paths': _resolved_import_paths(sys.path, Path.cwd().resolve()),
        'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_enabled': torch.backends.cudnn.enabled, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic}
    runtime.update(_numeric_observation(torch))
    if (runtime['torch_threads'] != 1 or runtime['torch_interop_threads'] != 1
            or runtime['deterministic_algorithms'] is not True or runtime['matmul_tf32'] is not False
            or runtime['cudnn_tf32'] is not False or runtime['cudnn_enabled'] is not True
            or runtime['cudnn_benchmark'] is not False or runtime['cudnn_deterministic'] is not True
            or runtime['float32_matmul_precision'] != 'highest' or runtime['deterministic_warn_only'] is not False
            or runtime['calling_thread_denormal_product_bits'] != 0x00400000 or runtime['cpu_autocast'] is not False):
        raise ValueError('runtime numerical settings differ from the calibration contract')
    key = _make_key(source_bytes, runtime)
    _guard_owner(identity, session, lease)
    return torch, key, source_bytes, _canonical_json_bytes(runtime)
