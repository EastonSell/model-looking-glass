"""Fixed owner for one supervised v1.3 repeated-tap session.

The source-bound bootstrap must verify this module and hard-exit after this
entrypoint returns. No GPU launch is enabled by importing this module. The
owner uses point checks, not atomic protection against an external revocation.
"""

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time


_OWNER_ENABLED = False
_PREFIX = 'LOOKING_GLASS_TAP_REPEATED_'
_MAX_JOURNAL_BYTES = 8 * 1024 * 1024
_MAX_ROW_BYTES = 128 * 1024
_CUSTODY = {'entered': False, 'directory': None, 'stage': 'entry', 'session': None,
            'device_lease': None, 'properties': None, 'pending_return': None,
            'admission': None, 'operations': None, 'runtime': None, 'sample': None,
            'setup': None, 'preflight': None, 'lean': None, 'journal': None,
            'barriers': None, 'disposal': None, 'failure': None, 'diagnostic_failure': None}


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _write_new(path, value):
    raw = _encoded(value)
    if not 1 <= len(raw) <= _MAX_ROW_BYTES:
        raise ValueError('private record exceeds its bound')
    with Path(path).open('xb') as handle:
        if handle.write(raw) != len(raw):
            raise OSError('short private record write')
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


class _Journal:
    """Bounded, append-only, fsynced journal; failure is never retried."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = self.path.open('xb')
        self.count = self.size = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def append(self, row):
        if self.closed or type(row) is not dict or self.count >= 86:
            raise RuntimeError('invalid journal transition')
        raw = _encoded(row)
        if len(raw) > _MAX_ROW_BYTES or self.size + len(raw) > _MAX_JOURNAL_BYTES:
            raise ValueError('journal exceeds its bound')
        if self.handle.write(raw) != len(raw):
            raise OSError('short journal write')
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.digest.update(raw)
        self.size += len(raw)
        self.count += 1

    def close(self):
        if self.closed or self.count != 86:
            raise RuntimeError('journal is incomplete')
        self.handle.close()
        self.closed = True
        return self.digest.hexdigest()


def _environment(run_id, source_commit, git_exe):
    if (type(run_id) is not str or re.fullmatch('[0-9a-f]{32}', run_id) is None
            or type(source_commit) is not str or re.fullmatch('[0-9a-f]{40}', source_commit) is None
            or not Path(git_exe).is_absolute() or not Path(git_exe).is_file()
            or os.environ.get(_PREFIX + 'SOURCE_COMMIT') != source_commit
            or os.path.normcase(os.environ.get(_PREFIX + 'GIT_EXE', '')) != os.path.normcase(str(git_exe))
            or os.environ.get(_PREFIX + 'BOOTSTRAP_PID') != str(os.getpid())
            or not sys.flags.isolated or not sys.flags.no_site):
        raise RuntimeError('the verified isolated source bootstrap is required')
    site = Path(os.environ.get(_PREFIX + 'BOOTSTRAP_SITE', ''))
    slot = os.environ.get(_PREFIX + 'SESSION', '')
    challenge = os.environ.get(_PREFIX + 'CHALLENGE', '')
    if (not site.is_absolute() or not site.is_dir() or slot not in ('0', '1', '2', '3', '4')
            or re.fullmatch('[0-9a-f]{64}', challenge) is None
            or os.environ.get('HF_HUB_OFFLINE') != '1'
            or os.environ.get('HF_HUB_DISABLE_TELEMETRY') != '1'
            or os.environ.get('TRANSFORMERS_OFFLINE') != '1'
            or os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != 'backend:native'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('the fixed owner environment differs')
    if any(name.partition('.')[0] in ('torch', 'transformers', 'numpy', 'huggingface_hub')
           for name in sys.modules):
        raise RuntimeError('a CUDA-capable dependency was imported before the device lease')
    return int(slot), challenge


def _directory(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError('a canonical existing private directory is required')
    return path


def _workspace(session):
    repo = Path(__file__).resolve().parents[1]
    matches = [parent for parent in repo.parents if parent.name.casefold() == 'work']
    if len(matches) != 1 or matches[0].name != 'work':
        raise RuntimeError('the canonical shared workspace cannot be determined')
    expected = matches[0].parent.resolve(strict=True)
    identity = session.identity
    if identity is None or Path(identity.workspace_root).resolve(strict=True) != expected:
        raise RuntimeError('the supervised workspace does not match the checkout')
    return repo


def _validate_cache(session, cache):
    root = Path(session.identity.workspace_root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise RuntimeError('the shared workspace is not canonical')
    expected = root
    for part in ('work', 'hf-cache', 'hub'):
        expected = expected / part
        info = expected.lstat()
        if (not stat.S_ISDIR(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                or expected.resolve(strict=True) != expected):
            raise RuntimeError('the shared cache contains an indirect path')
    if cache != expected:
        raise RuntimeError('the shared offline checkpoint cache differs')
    home = os.environ.get('HF_HOME', '')
    if (not home or not Path(home).is_absolute() or Path(home) != cache.parent
            or Path(home).resolve(strict=True) != cache.parent):
        raise RuntimeError('the Hugging Face home differs from the shared cache')
    for name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE'):
        value = os.environ.get(name)
        if value and Path(value).resolve(strict=True) != cache:
            raise RuntimeError('an alternate Hugging Face cache is configured')


def _check_torch_version(torch):
    if (str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8'
            or not callable(getattr(torch._C, '_cuda_clearCublasWorkspaces', None))):
        raise RuntimeError('the pinned PyTorch/CUDA runtime differs')


def _check_transformers_version(transformers):
    if str(transformers.__version__) != '4.57.3':
        raise RuntimeError('the pinned Transformers runtime differs')


def _dependency_versions():
    from importlib.metadata import version
    return {name: version(distribution) for name, distribution in (
        ('huggingface_hub', 'huggingface-hub'), ('safetensors', 'safetensors'),
        ('tokenizers', 'tokenizers'), ('numpy', 'numpy'))}


def _device_description(properties):
    name = properties.name
    major, minor = properties.major, properties.minor
    if (type(name) is not str or not 1 <= len(name) <= 128 or any(ord(ch) < 32 for ch in name)
            or type(major) is not int or type(minor) is not int
            or not 0 <= major <= 99 or not 0 <= minor <= 99):
        raise ValueError('CUDA device description is invalid')
    return name, [major, minor]


def _clock():
    info = time.get_clock_info('perf_counter')
    clock = {'name': 'perf_counter_ns', 'monotonic': info.monotonic,
             'adjustable': info.adjustable, 'implementation': info.implementation,
             'resolution_seconds': info.resolution}
    runtime = {'platform': sys.platform, 'implementation': sys.implementation.name,
               'python_version': '.'.join(map(str, sys.version_info[:3]))}
    if (runtime != {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}
            or clock['implementation'] != 'QueryPerformanceCounter()'):
        raise RuntimeError('the pinned cross-process clock runtime differs')
    from .tap_repeated_protocol import clock_minimum
    clock_minimum(clock)
    return clock, runtime


def _forbid_collection(*_args, **_kwargs):
    raise RuntimeError('explicit cyclic collection is forbidden during owner custody')


def _device_call(session, lease, operation, *args):
    """Root a returned Python value through the supervised postcheck."""
    from .supervisor import OwnerSession
    OwnerSession.check(session, lease)
    value = None
    try:
        value = operation(*args)
        _CUSTODY['pending_return'] = value
        OwnerSession.check(session, lease)
        return value
    except BaseException:
        if value is not None:
            _CUSTODY['pending_return'] = value
        raise


def _initial_snapshot(torch, session, lease, total, cap):
    from .tap_repeated_protocol import check_memory
    pair = _device_call(session, lease, torch.cuda.mem_get_info, 0)
    if (type(pair) is not tuple or len(pair) != 2 or any(type(item) is not int for item in pair)
            or pair[1] != total or not 0 <= pair[0] <= total):
        raise ValueError('invalid initial CUDA free/total values')
    allocated = _device_call(session, lease, torch.cuda.memory_allocated, 0)
    if type(allocated) is not int or not 0 <= allocated <= total:
        raise ValueError('invalid initial CUDA allocation')
    reserved = _device_call(session, lease, torch.cuda.memory_reserved, 0)
    row = {'free_bytes': pair[0], 'total_bytes': total,
           'allocated_bytes': allocated, 'reserved_bytes': reserved}
    check_memory(row, expected_total=total, cap_bytes=cap)
    return row


def _runtime_record(torch, transformers, session, lease, dependencies):
    fraction = _device_call(session, lease, torch.cuda.get_per_process_memory_fraction, 0)
    backend = _device_call(session, lease, torch.cuda.get_allocator_backend)
    return {'platform': sys.platform, 'implementation': sys.implementation.name,
            'python_version': '.'.join(map(str, sys.version_info[:3])),
            'torch_version': str(torch.__version__), 'cuda_build': str(torch.version.cuda),
            'transformers_version': str(transformers.__version__), 'loader_versions': dependencies,
            'allocator_backend': backend, 'allocator_configuration': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
            'device': 'cuda:0', 'dtype': 'float32', 'attention_implementation': 'eager',
            'eval': True, 'use_cache': False, 'cpu_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'seed': 17, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'allocator_fraction': fraction, 'cyclic_gc_enabled': gc.isenabled(),
            'autocast_enabled': torch.is_autocast_enabled('cuda'),
            'overwrite_module_params_on_conversion': torch.__future__.get_overwrite_module_params_on_conversion(),
            'swap_module_params_on_conversion': torch.__future__.get_swap_module_params_on_conversion()}


def _expected_runtime():
    return {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15',
            'torch_version': '2.8.0+cu128', 'cuda_build': '12.8', 'transformers_version': '4.57.3',
            'loader_versions': {'huggingface_hub': '0.36.2', 'safetensors': '0.8.0',
                                'tokenizers': '0.22.2', 'numpy': '2.4.6'},
            'allocator_backend': 'native', 'allocator_configuration': 'backend:native',
            'device': 'cuda:0', 'dtype': 'float32',
            'attention_implementation': 'eager', 'eval': True, 'use_cache': False,
            'cpu_threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
            'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_benchmark': False,
            'cudnn_deterministic': True, 'seed': 17, 'cublas_workspace_config': ':4096:8',
            'allocator_fraction': 0.25, 'cyclic_gc_enabled': False, 'autocast_enabled': False,
            'overwrite_module_params_on_conversion': False, 'swap_module_params_on_conversion': False}


def _exact_mapping(actual, expected):
    return (type(actual) is dict and set(actual) == set(expected)
            and all(type(actual[key]) is type(value) and actual[key] == value for key, value in expected.items()))


def _failure_record(exc):
    """Project only bounded, already assigned diagnostic values; never a result."""
    stages = ('entry', 'device_binding', 'model_setup', 'preflight', 'samples', 'screen_wait', 'disposal', 'attested')
    stage = _CUSTODY['stage'] if _CUSTODY['stage'] in stages else 'unknown'
    error_type = type(exc).__name__
    if re.fullmatch('[A-Za-z_][A-Za-z_0-9]{0,63}', error_type) is None:
        error_type = 'other'
    record = {'schema': 'tap-repeated-failure-v1', 'status': 'unverified',
              'stage': stage, 'error_type': error_type}
    journal = _CUSTODY.get('journal')
    count = getattr(journal, 'count', None)
    if type(count) is int and 0 <= count <= 86:
        record['journal_rows_fsynced_before_failure'] = count
        record['journal_close_confirmed'] = getattr(journal, 'closed', None) is True
    source = {'model_setup': 'setup', 'preflight': 'preflight', 'samples': 'lean',
              'screen_wait': 'lean', 'disposal': 'lean'}.get(stage)
    component = _CUSTODY.get(source) if source else None
    operations = _CUSTODY.get('operations')
    roots = getattr(operations, 'roots', None)
    failures = {'component': getattr(component, 'failure', None),
                'operation_root': getattr(roots, 'failure', None)}
    private_types = {}
    for key, error in failures.items():
        if isinstance(error, BaseException):
            name = type(error).__name__
            private_types[key] = name if re.fullmatch('[A-Za-z_][A-Za-z_0-9]{0,63}', name) else 'other'
    if private_types:
        record['private_error_types'] = private_types
    partial = getattr(component, 'partial', None)
    if type(partial) is dict:
        projected = {'source': source}
        allowed = ('setup', 'checkpoint_verified', 'preflight_setup', 'prepared', 'gate_pending',
                   'first_forward_attempt', 'pre_sample', 'running', 'completion_synchronized',
                   'postcheck_pending', 'complete')
        if partial.get('stage') in allowed:
            projected['stage'] = partial['stage']
        if partial.get('mode') in ('none', 'direct', 'gradient'):
            projected['mode'] = partial['mode']
        steps = ('peak_allocated_query', 'peak_reserved_query', 'final_memory_snapshot', 'peak_comparison',
                 'postcheck_clock_query', 'postcheck_time_validation', 'guarded_end_clock_query',
                 'guarded_end_time_validation')
        if partial.get('diagnostic_step') in steps:
            projected['diagnostic_step'] = partial['diagnostic_step']
        for key in ('started_ns', 'completion_ns', 'ended_ns', 'postcheck_completed_ns',
                    'guarded_started_ns', 'guarded_ended_ns', 'memory_snapshot_started_ns',
                    'memory_snapshot_ended_ns', 'gate_lower_ns', 'gate_upper_ns',
                    'peak_allocated_bytes', 'peak_reserved_bytes'):
            value = partial.get(key)
            if type(value) is int and 0 <= value < 2 ** 63:
                projected[key] = value
        event = partial.get('device_event_ms')
        if type(event) in (int, float) and math.isfinite(event) and 0 <= event <= 1e12:
            projected['device_event_ms'] = event
        keys = ('free_bytes', 'total_bytes', 'allocated_bytes', 'reserved_bytes')
        for key in ('preparation_memory', 'memory_before_transfer', 'memory_before', 'memory_after'):
            value = partial.get(key)
            if (type(value) is dict and set(value) == set(keys)
                    and all(type(value[name]) is int and 0 <= value[name] < 2 ** 63 for name in keys)):
                projected[key] = {name: value[name] for name in keys}
        record['partial'] = projected
    return record


def _execute_owner(run_dir, cache_dir, run_id, source_commit, git_exe):
    slot, challenge = _environment(run_id, source_commit, git_exe)
    directory, cache = _directory(run_dir), _directory(cache_dir)
    _CUSTODY['directory'] = directory
    clock, frame_runtime = _clock()
    gc.disable()
    from .device_lease import DeviceLease
    from .supervisor import OwnerSession, _supervisor_birth
    from .tap_repeated_protocol import (EXPERIMENT_PROTOCOL, GIB, SCHEDULE_SHA256,
                                        load_schedule, session_order, summarize_session)
    session = OwnerSession.from_environment()
    _CUSTODY['session'] = session
    lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
    _CUSTODY['device_lease'] = lease
    OwnerSession.check(session, lease)
    repo = _workspace(session)
    _validate_cache(session, cache)
    schedule = load_schedule(repo / 'evidence/gpu-tap-latency-repeated-v1/schedule.json')
    order = session_order(schedule, slot)
    _CUSTODY['stage'] = 'device_binding'
    import torch
    # Import may use explicit collection while no CUDA object exists. The
    # guarded workload and disposal must never trigger global cyclic GC.
    gc.collect = _forbid_collection
    _check_torch_version(torch)
    if torch.cuda.is_initialized() is not False:
        raise RuntimeError('CUDA was initialized before its supervised binding')
    count = _device_call(session, lease, torch.cuda.device_count)
    if type(count) is not int or count != 1:
        raise RuntimeError('exactly one visible CUDA device is required')
    properties = _device_call(session, lease, torch.cuda.get_device_properties, 0)
    _CUSTODY['properties'] = properties
    from .gpu_oneshot import _runtime_uuid
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=count)
    name, capability = _device_description(properties)
    import transformers
    _check_transformers_version(transformers)
    dependencies = _dependency_versions()
    if dependencies != _expected_runtime()['loader_versions']:
        raise RuntimeError('the pinned loader dependency versions differ')
    total = properties.total_memory
    if type(total) is not int or not 7 * GIB <= total <= 9 * GIB:
        raise RuntimeError('CUDA total is outside the expected device range')
    cap = total // 4
    if _device_call(session, lease, torch.cuda.get_allocator_backend) != 'native':
        raise RuntimeError('the native CUDA allocator is required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(17)
    _device_call(session, lease, torch.cuda.manual_seed, 17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise RuntimeError('module conversion flags differ')
    from .tap_repeated_admission import GPU_TRACE_CLAIM, _Admission
    admission = _Admission(check_owner=lambda: OwnerSession.check(session, lease),
                           snapshot=lambda: _initial_snapshot(torch, session, lease, total, cap),
                           total_bytes=total, cap_bytes=cap)
    _CUSTODY['admission'] = admission
    initial_memory = admission.acquire()
    _CUSTODY['stage'] = 'model_setup'
    from . import tap_repeated_owner as owner
    from . import tap_repeated_cuda_backend as cuda_backend
    from . import tap_repeated_timing as timing
    from . import tap_repeated_model as fixed_model
    from . import tap_repeated_lean as lean_module
    from .tap_repeated_preflight import _FixedPreflight
    from .tap_repeated_child_barrier import ChildBarriers, wait_frame
    owner._OPERATIONS_ENABLED = cuda_backend._CUDA_BACKEND_ENABLED = True
    timing._TIMING_ENABLED = fixed_model._MODEL_SETUP_ENABLED = lean_module._LEAN_ENABLED = True
    operations = owner._OwnerOperations(session, lease, admission)
    _CUSTODY['operations'] = operations
    operations.bind_session(slot)
    operations.require_disabled_gc()
    runtime = cuda_backend._CudaRuntime(operations)
    _CUSTODY['runtime'] = runtime
    runtime.initialize()
    sample = timing._TimedSample(operations, runtime, session=slot, clock_info=clock)
    _CUSTODY['sample'] = sample
    setup = fixed_model._FixedModelSetup(sample)
    _CUSTODY['setup'] = setup
    model = setup.run(cache)
    runtime_record = _runtime_record(torch, transformers, session, lease, dependencies)
    if not _exact_mapping(runtime_record, _expected_runtime()):
        raise RuntimeError('the pinned execution runtime differs')
    preflight = _FixedPreflight(sample)
    _CUSTODY['preflight'] = preflight
    preflight.prepare()
    journal = _Journal(directory / 'journal.jsonl')
    _CUSTODY['journal'] = journal
    ready = {'schema': 'tap-repeated-owner-ready-v1', 'session_index': slot, 'owner_pid': os.getpid(),
             'owner_birth': _supervisor_birth(), 'supervisor_run_id': session.run_id,
             'challenge': challenge, 'gpu_uuid': session.identity.gpu_uuid, 'cuda_total_bytes': total,
             'ready_ns': time.perf_counter_ns(), 'clock': clock, 'runtime': frame_runtime}
    barriers = ChildBarriers(ready)
    _CUSTODY['barriers'] = barriers
    session.send_startup(lease, _encoded(ready))
    raw, seen = wait_frame(directory / 'start-grant.json', timeout=30)
    barriers.accept_start(raw, observed_ns=seen)
    _CUSTODY['stage'] = 'preflight'
    first, first_post = preflight.run(barriers.authorize_preflight)
    journal.append(first)
    journal.append(first_post)
    lean = lean_module._LeanTimedSample(sample)
    _CUSTODY['lean'] = lean
    _CUSTODY['stage'] = 'samples'
    post = None
    for item in order:
        result = lean.sample_mode(item['mode'])
        row = {**item, 'session': slot, 'postcheck_status': 'pending', **result}
        post = {'stage': 'postcheck', 'observation_id': item['observation_id'], 'status': 'passed',
                'completed_ns': result['postcheck_completed_ns'], 'memory_after': result['memory_after'],
                'device_work_complete': True}
        journal.append(row)
        journal.append(post)
    journal_sha = journal.close()
    if post is None or journal.count != 86:
        raise RuntimeError('the fixed journal inventory is incomplete')
    # Replay the same bounded bytes before advertising a successful workload.
    raw_journal = (directory / 'journal.jsonl').read_bytes()
    if (len(raw_journal) > _MAX_JOURNAL_BYTES or hashlib.sha256(raw_journal).hexdigest() != journal_sha
            or len(raw_journal.splitlines()) != 86):
        raise RuntimeError('closed journal identity differs')
    rows = [json.loads(line) for line in raw_journal.splitlines()]
    summarize_session(schedule, slot, rows, clock=clock, total_bytes=total, cap_bytes=cap,
                      device='cuda:0', run_protocol=EXPERIMENT_PROTOCOL)
    admission.verify()
    candidate = {'schema': 'tap-repeated-candidate-v1', 'protocol': EXPERIMENT_PROTOCOL,
                 'run_id': run_id, 'supervisor_run_id': session.run_id, 'owner_pid': os.getpid(),
                 'owner_birth': ready['owner_birth'], 'session_index': slot, 'source_commit': source_commit,
                 'gpu_uuid': session.identity.gpu_uuid, 'cuda_total_bytes': total, 'cap_bytes': cap,
                 'cuda_device_name': name, 'cuda_compute_capability': capability,
                 'free_before_model_transfer': setup.partial['memory_before_transfer']['free_bytes'],
                 'admission_memory': initial_memory,
                 'memory_before_transfer': setup.partial['memory_before_transfer'],
                 'trace_claim_bytes': GPU_TRACE_CLAIM, 'whole_run_claims_held': True,
                 'model': model, 'clock': clock, 'runtime': runtime_record, 'schedule_sha256': SCHEDULE_SHA256,
                 'journal_sha256': journal_sha, 'journal_rows': journal.count,
                 'preflight_started_ns': first['started_ns'], 'workload_ended_ns': post['completed_ns']}
    OwnerSession.check(session, lease)
    candidate_sha = _write_new(directory / 'candidate.json', candidate)
    OwnerSession.check(session, lease)
    _CUSTODY['stage'] = 'screen_wait'
    barriers.publish_finished(directory / 'finished.json', preflight_started_ns=first['started_ns'],
                              workload_ended_ns=post['completed_ns'])
    raw, seen = wait_frame(directory / 'cleanup-grant.json', timeout=30)
    barriers.accept_cleanup(raw, observed_ns=seen)
    _CUSTODY['stage'] = 'disposal'
    from .tap_repeated_disposal import _Disposal
    clear_workspaces = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
    if not callable(clear_workspaces):
        raise RuntimeError('the pinned CUDA workspace cleanup is unavailable')
    disposal = _Disposal(operations, synchronize=runtime.synchronize,
                         cleanup=(torch.cuda.empty_cache, clear_workspaces, torch.cuda.empty_cache),
                         final_counters=(lambda: torch.cuda.memory_allocated(0),
                                         lambda: torch.cuda.memory_reserved(0)))
    _CUSTODY['disposal'] = disposal
    disposal.finish()
    OwnerSession.check(session, lease)
    final = {'schema': 'tap-repeated-final-v1', 'run_id': run_id, 'supervisor_run_id': session.run_id,
             'owner_pid': os.getpid(), 'owner_birth': ready['owner_birth'], 'session_index': slot,
             'source_commit': source_commit, 'gpu_uuid': session.identity.gpu_uuid,
             'candidate_sha256': candidate_sha, 'unsafe': operations.unsafe,
             'synchronized_disposal': disposal.state == 'released',
             'claims_released_after_disposal': admission.state == 'released',
             'allocated_bytes': disposal.counters['allocated_bytes'],
             'reserved_bytes': disposal.counters['reserved_bytes'],
             'active_memory_leases': len(admission.budget.active_leases()),
             'trace_claim_remaining_bytes': admission.lease.trace_bytes('gpu'),
             'device_lease_held_at_attestation': True}
    _write_new(directory / 'final.json', final)
    OwnerSession.attest_clean(session, lease)
    _CUSTODY['stage'] = 'attested'
    return 0


def owner_main(run_dir, cache_dir, run_id, source_commit, git_exe):
    """Fail-stop entrypoint; bootstrap must hard-exit with its return status."""
    try:
        if _OWNER_ENABLED is not True or _CUSTODY['entered'] is not False:
            raise RuntimeError('repeated tap GPU owner remains disabled or was already entered')
        _CUSTODY['entered'] = True
        return _execute_owner(run_dir, cache_dir, run_id, source_commit, git_exe)
    except BaseException as exc:
        if _CUSTODY['failure'] is None:
            _CUSTODY['failure'] = exc
        directory = _CUSTODY['directory']
        if directory is not None:
            try:
                _write_new(directory / 'failure.json', _failure_record(exc))
            except BaseException as diagnostic:
                _CUSTODY['diagnostic_failure'] = diagnostic
        return 1
