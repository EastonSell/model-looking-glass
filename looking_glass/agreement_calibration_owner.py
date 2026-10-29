"""Dedicated five-token resource owner, pending integrated GPU review.

This is a provisional bootstrap, not a study runner or a memory guarantee.
The caller must supervise the process, hold the physical lease through exit,
and independently audit all private evidence after reaping the exact child.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace
import weakref

from .device_lease import DeviceLease
from .live_profile import _guard_owner
from .supervisor import OwnerSession


# This switch permits only the separately reviewed supervised resource pilot.
AGREEMENT_CALIBRATION_OWNER_READY = True
_CUSTODY = {'entered': False, 'session': None, 'lease': None, 'source': None,
            'budget': None, 'resident': None, 'job': None, 'bundle': None,
            'model': None, 'converted': None, 'probe': None, 'runner': None,
            'failure': None}


def _write_new(path, data):
    if type(data) is not bytes:
        raise TypeError('artifact bytes are required')
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
    try:
        offset = 0
        while offset < len(data):
            count = os.write(fd, data[offset:])
            if count <= 0:
                raise OSError('artifact write made no progress')
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _decode_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('non-finite JSON value')
    if type(data) is not bytes or not 1 <= len(data) <= 1024 * 1024:
        raise ValueError('bounded JSON bytes are required')
    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid_constant)


def _read_frozen_file(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
            or not 1 <= info.st_size <= 1024 * 1024):
        raise ValueError('frozen input is not a bounded ordinary file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise ValueError('frozen input changed during open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) != info.st_size:
            raise ValueError('frozen input changed during read')
    finally:
        os.close(fd)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
        raise ValueError('frozen input changed during read')
    return data


def _workspace_binding(repo, session):
    """Bind the supervised device to the checkout's shared lock namespace."""
    from .device_identity import DeviceIdentity
    from .supervisor import _identity_spec
    identity = session.identity
    if type(identity) is not DeviceIdentity or type(session._identity_spec) is not dict:
        raise ValueError('a serialized supervised physical identity is required')
    repo = Path(repo).resolve(strict=True)
    work = [parent for parent in repo.parents if parent.name == 'work']
    expected = work[0].parent if work else repo
    if (identity.workspace_root != expected or not (expected / 'work').is_dir()
            or _identity_spec(identity) != session._identity_spec):
        raise ValueError('device identity does not use the shared workspace root')
    DeviceIdentity.validate_paths(identity)


def _check_frozen_preflight(raw, expected_hash, run_dir, run_id, repo, cache, directory_id, source, identity):
    """Validate the parent's durable freeze against pre-CUDA observations."""
    from .pretrained import MODEL_ID, REVISION
    if (type(expected_hash) is not str or re.fullmatch(r'[0-9a-f]{64}', expected_hash) is None
            or type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected_hash
            or _read_frozen_file(run_dir / 'run-manifest.json') != raw):
        raise ValueError('saved run manifest does not match its exact commitment')
    record = _decode_json(raw)
    if type(record) is not dict or set(record) != {'schema', 'run_id', 'source', 'checkpoint',
                                                   'runtime', 'device', 'workload', 'directory'}:
        raise ValueError('invalid frozen run manifest')
    if record['schema'] != 'agreement-calibration-run-freeze-v1' or record['run_id'] != run_id:
        raise ValueError('frozen run identity differs')
    src, checkpoint, runtime, device, workload, directory = (record[name] for name in
        ('source', 'checkpoint', 'runtime', 'device', 'workload', 'directory'))
    if (type(src) is not dict or set(src) != {'commit', 'sha256', 'checkout', 'source_base_sha256'}
            or type(src['commit']) is not str or re.fullmatch(r'[0-9a-f]{40}', src['commit']) is None
            or type(src['sha256']) is not str or re.fullmatch(r'[0-9a-f]{64}', src['sha256']) is None
            or src['checkout'] != str(repo)
            or src['source_base_sha256'] != hashlib.sha256(source.source_base).hexdigest()):
        raise ValueError('frozen source commitment differs')
    controller = _decode_json(_read_frozen_file(run_dir / 'controller-source.json'))
    if (type(controller) is not dict or set(controller) != {'schema', 'commit', 'files'}
            or controller['schema'] != 'agreement-controller-source-v1'
            or controller['commit'] != src['commit'] or type(controller['files']) is not dict
            or not controller['files'] or any(type(path) is not str or not path or '\\' in path
                or Path(path).is_absolute() or '..' in path.split('/')
                or type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                for path, digest in controller['files'].items())):
        raise ValueError('invalid frozen controller source inventory')
    canonical = json.dumps(controller, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != src['sha256']:
        raise ValueError('controller source inventory does not match the freeze')
    assets = _decode_json(source.asset_manifest)
    expected_checkpoint = {'model_id': MODEL_ID, 'revision': REVISION, 'cache_root': str(cache),
                           'assets': assets['assets']}
    if type(checkpoint) is not dict or _json_bytes(checkpoint) != _json_bytes(expected_checkpoint):
        raise ValueError('frozen checkpoint differs from verified offline assets')
    expected_workload = {'seed': 20261028, 'target_input_ids': [464, 3797, 1474, 262, 9168],
        'donor_input_ids': [464, 11875, 1474, 262, 9168], 'subject_index': 1,
        'last_token_index': 4, 'batch_size': 1, 'sequence_length': 5}
    expected_directory = {'path': str(run_dir), 'device': directory_id[0], 'inode': directory_id[1]}
    if (_json_bytes(workload) != _json_bytes(expected_workload)
            or _json_bytes(directory) != _json_bytes(expected_directory)):
        raise ValueError('frozen workload or private directory differs')
    version_fields = {'python_version', 'torch_version', 'transformers_version', 'numpy_version',
                      'huggingface_hub_version', 'safetensors_version', 'tokenizers_version',
                      'cuda_runtime_version'}
    if (type(runtime) is not dict or set(runtime) != version_fields
            or any(type(value) is not str or not value for value in runtime.values())
            or type(device) is not dict or set(device) != {'uuid', 'name', 'total_bytes', 'driver_version'}
            or device['uuid'] != identity.gpu_uuid or type(device['name']) is not str or not device['name']
            or type(device['driver_version']) is not str or not device['driver_version']
            or type(device['total_bytes']) is not int or device['total_bytes'] <= 0):
        raise ValueError('frozen runtime or physical inventory is invalid')
    return record


def _check_frozen_runtime(frozen, observed):
    for name, expected in frozen['runtime'].items():
        if observed.get(name) != expected:
            raise ValueError('live runtime differs from frozen ' + name)
    if (observed.get('driver_version_provenance') != 'controller-prelaunch-nvidia-smi'
            or observed.get('subprocess_policy') != 'python-audit-hook-denied'):
        raise ValueError('runtime driver provenance or subprocess policy differs')
    device = frozen['device']
    if (observed.get('device_uuid') != device['uuid'] or observed.get('device_name') != device['name']
            or observed.get('driver_version') != device['driver_version']
            or type(observed.get('device_total_bytes')) is not int
            or abs(observed['device_total_bytes'] - device['total_bytes']) > 1024 ** 2):
        raise ValueError('live device differs from the frozen physical inventory')


def _check_private_run_dir(path, expected, repo, cache):
    path = Path(path)
    if (not path.is_absolute() or type(expected) is not tuple or len(expected) != 2
            or any(type(value) is not int or value < 0 for value in expected)):
        raise ValueError('an absolute run directory and its exact file identity are required')
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
            or (info.st_dev, info.st_ino) != expected):
        raise ValueError('private run directory identity changed')
    resolved = path.resolve(strict=True)
    for protected in (Path(repo).resolve(strict=True), Path(cache).resolve(strict=True)):
        if resolved == protected or resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
            raise ValueError('private run directory must be outside source and cache')
    return resolved


def _check_claims(budget, resident, job, gpu_trace, host_trace):
    from .memory_budget import MemoryBudget, MemoryLease
    if (type(budget) is not MemoryBudget or type(resident) is not MemoryLease or type(job) is not MemoryLease
            or resident is job or resident._budget is not budget or job._budget is not budget
            or resident.released or job.released or set(budget.active_leases()) != {resident, job}
            or resident.estimated_incremental_bytes != 2 * 1024 ** 3
            or job.estimated_incremental_bytes != 3 * 1024 ** 3 // 2
            or resident.trace_bytes('gpu') or resident.trace_bytes('host')
            or job.trace_bytes('gpu') != gpu_trace or job.trace_bytes('host') != host_trace):
        raise ValueError('the exact provisional commitments are not held')


def _check_owner_admission(snapshot, budget, resident, job, expected_total):
    from .memory_budget import BudgetSnapshot, CudaMemorySnapshot
    from .resource_profile import _check_headroom
    trace = 2 * 1024 ** 2
    _check_claims(budget, resident, job, trace, trace)
    if (type(snapshot) is not BudgetSnapshot or type(snapshot.telemetry) is not CudaMemorySnapshot
            or snapshot.telemetry.device_total_bytes != expected_total
            or snapshot.telemetry.device_free_bytes < 3 * 1024 ** 3
            or snapshot.committed_incremental_bytes != 7 * 1024 ** 3 // 2
            or snapshot.active_leases != 2 or snapshot.gpu_trace_bytes != trace or snapshot.host_trace_bytes != trace
            or snapshot.reserve_bytes < max(1024 ** 3, (expected_total * 15 + 99) // 100)):
        raise ValueError('live admission does not match the verified device or commitments')
    _check_headroom(snapshot, snapshot.telemetry)


def _release_claims(identity, session, lease, budget, resident, job):
    """Retire claims after cleanup, stopping immediately if ownership is lost."""
    trace = 2 * 1024 ** 2
    _guard_owner(identity, session, lease)
    _check_claims(budget, resident, job, trace, trace)
    job.release_trace('gpu', trace)
    _guard_owner(identity, session, lease)
    _check_claims(budget, resident, job, 0, trace)
    job.release_trace('host', trace)
    _guard_owner(identity, session, lease)
    _check_claims(budget, resident, job, 0, 0)
    job.release()
    _guard_owner(identity, session, lease)
    if (not job.released or resident.released or set(budget.active_leases()) != {resident}
            or resident._budget is not budget or resident.estimated_incremental_bytes != 2 * 1024 ** 3
            or resident.trace_bytes('gpu') or resident.trace_bytes('host')):
        raise ValueError('resident commitment changed while retiring job claim')
    resident.release()
    _guard_owner(identity, session, lease)
    if not resident.released or budget.active_leases():
        raise ValueError('memory commitments remain after cleanup')


def _guarded_cleanup(session, lease, torch, budget, resident, job):
    """Recheck the physical owner and exact claims around each CUDA call."""
    from .cuda_cleanup import clear_cublas_after_drain
    trace = 2 * 1024 ** 2
    def guarded(function):
        def call(*args, **kwargs):
            _guard_owner(session.identity, session, lease)
            _check_claims(budget, resident, job, trace, trace)
            value = function(*args, **kwargs)
            _guard_owner(session.identity, session, lease)
            _check_claims(budget, resident, job, trace, trace)
            return value
        return call
    cuda = SimpleNamespace(**{name: guarded(getattr(torch.cuda, name)) for name in
        ('is_initialized', 'synchronize', 'empty_cache', 'memory_allocated', 'memory_reserved')})
    wrapped = SimpleNamespace(__version__=torch.__version__, version=torch.version, cuda=cuda,
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=guarded(torch._C._cuda_clearCublasWorkspaces)))
    return clear_cublas_after_drain(session=session, lease=lease, torch=wrapped)


def assemble_profiles(key, cold, warm, source, collected_utc):
    """Build separately pinned cold and warm estimates from the exact samples."""
    from .agreement_cuda_runner import _validate_profile, _validate_warmup_profile
    from .resource_profile import MeasuredResourceProfile, ProfileProvenance, encode_evidence
    from dataclasses import replace
    if type(source) is not bytes or not source or type(warm) is not tuple:
        raise ValueError('source bytes and ordered samples are required')
    profiles, evidence = [], []
    for sample_key, samples in ((replace(key, warmup_state='cold'), (cold,)), (key, warm)):
        encoded = encode_evidence(sample_key, samples)
        provenance = ProfileProvenance('resource-profile-v1', hashlib.sha256(source).hexdigest(),
            hashlib.sha256(encoded).hexdigest(), collected_utc, 'cooperative-exclusive',
            'synchronize-reset-synchronize')
        profile = MeasuredResourceProfile.build(sample_key, samples, provenance,
                                                margin_bytes=512 * 1024 ** 2, margin_bps=2000)
        profiles.append(profile)
        evidence.append(encoded)
    _validate_warmup_profile(profiles[0], key)
    _validate_profile(profiles[1], key)
    if any(profile.estimated_incremental_bytes > 3 * 1024 ** 3 // 2 for profile in profiles):
        raise ValueError('observed profile exceeds the provisional job commitment')
    return tuple(profiles), tuple(evidence)


def collect_owned(session, device_lease, *, repo_root, cache_root, run_dir, run_id, run_dir_identity=None,
                  run_manifest_bytes=None, run_manifest_sha256=None):
    """Collect under an already acquired owner; never attest or release its device lease."""
    if not AGREEMENT_CALIBRATION_OWNER_READY:
        raise RuntimeError('agreement calibration owner is not yet enabled')
    if type(session) is not OwnerSession or type(device_lease) is not DeviceLease:
        raise ValueError('an exact supervised owner and device lease are required')
    if _CUSTODY['entered']:
        raise RuntimeError('agreement calibration owner is single-use')
    _CUSTODY['entered'], _CUSTODY['session'], _CUSTODY['lease'] = True, session, device_lease
    try:
        return _collect_owned(session, device_lease, repo_root=repo_root, cache_root=cache_root,
                              run_dir=run_dir, run_id=run_id, run_dir_identity=run_dir_identity,
                              run_manifest_bytes=run_manifest_bytes, run_manifest_sha256=run_manifest_sha256)
    except BaseException as error:
        _CUSTODY['failure'] = error
        runner = _CUSTODY['runner']
        if runner is not None:
            try:
                runner.quarantine_after_uncertain_callback()
            except BaseException:
                pass
        # Device/claims and every known model or traceback root stay live until
        # the supervised process exits. Do not retry CUDA on this path.
        raise


def _collect_owned(session, lease, *, repo_root, cache_root, run_dir, run_id, run_dir_identity,
                   run_manifest_bytes, run_manifest_sha256):
    from .agreement_calibration_identity import prepare_agreement_runtime, _numeric_observation
    from .trusted_service_profile import (prepare_trusted_source, verify_trusted_cpu_model,
                                           verify_trusted_device_tie)
    from .memory_budget import MemoryBudget

    repo, cache = Path(repo_root).resolve(strict=True), Path(cache_root).resolve(strict=True)
    run_dir = _check_private_run_dir(run_dir, run_dir_identity, repo, cache)
    if (type(run_id) is not str or not run_id
            or Path(__file__).resolve(strict=True) != repo / 'looking_glass' / 'agreement_calibration_owner.py'):
        raise ValueError('invalid private run directory or source checkout')
    def write_artifact(name, data):
        _check_private_run_dir(run_dir, run_dir_identity, repo, cache)
        _write_new(run_dir / name, data)
        _check_private_run_dir(run_dir, run_dir_identity, repo, cache)
    identity = session.identity
    _workspace_binding(repo, session)
    _guard_owner(identity, session, lease)
    source = prepare_trusted_source(str(repo), str(cache))
    _CUSTODY['source'] = source
    frozen = _check_frozen_preflight(run_manifest_bytes, run_manifest_sha256, run_dir, run_id,
                                     repo, cache, run_dir_identity, source, identity)
    torch, key, source_bytes, runtime_bytes = prepare_agreement_runtime(
        source, session, lease, frozen_runtime=frozen['runtime'],
        frozen_driver_version=frozen['device']['driver_version'])
    runtime = json.loads(runtime_bytes)
    _check_frozen_runtime(frozen, runtime)
    expected_numeric = {name: runtime[name] for name in ('float32_matmul_precision',
        'deterministic_warn_only', 'calling_thread_denormal_product_bits', 'cpu_autocast')}
    def check_numeric():
        if _numeric_observation(torch) != expected_numeric:
            raise ValueError('calling-thread numerical settings differ from the runtime identity')
    write_artifact('source-manifest.json', source_bytes)
    write_artifact('runtime-manifest.json', runtime_bytes)
    write_artifact('asset-manifest.json', source.asset_manifest)

    # The adapter itself imports Torch; this point follows physical UUID binding.
    from .agreement_cuda_runner import GuardedAgreementMemoryProbe, TRACE_BYTES
    from .agreement_cuda_calibration import CudaAgreementCalibrator, PROVISIONAL_JOB_BYTES, PROVISIONAL_RESIDENT_BYTES
    from .agreement_journal import AgreementJournal
    from .pretrained import load_distilgpt2
    from . import agreement_calibration_collector as collector

    def binding():
        _guard_owner(identity, session, lease)
        count = torch.cuda.device_count()
        _guard_owner(identity, session, lease)
        properties = torch.cuda.get_device_properties(0)
        from .live_profile import decode_cuda_uuid
        actual = decode_cuda_uuid(properties.uuid, torch._C._CUuuid)
        OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=actual,
                                            device_count=count, cuda_device_index=0)
        _guard_owner(identity, session, lease)
        return actual, count

    probe = GuardedAgreementMemoryProbe(torch.cuda, device='cuda:0', device_uuid=identity.gpu_uuid,
        runtime_binding=binding, session=session, device_lease=lease)
    _CUSTODY['probe'] = probe
    budget = MemoryBudget(probe.snapshot, max_gpu_trace_bytes=TRACE_BYTES, max_host_trace_bytes=TRACE_BYTES)
    _CUSTODY['budget'] = budget
    resident = budget.acquire(PROVISIONAL_RESIDENT_BYTES, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    _CUSTODY['resident'] = resident
    def check_resident_only():
        _guard_owner(identity, session, lease)
        if (resident._budget is not budget or resident.released
                or set(budget.active_leases()) != {resident}
                or resident.estimated_incremental_bytes != PROVISIONAL_RESIDENT_BYTES
                or resident.trace_bytes('gpu') or resident.trace_bytes('host')):
            raise ValueError('the provisional resident commitment is not held')
    check_resident_only()
    probe._bound_runner = SimpleNamespace(_check_owner=check_resident_only)
    job = budget.acquire(PROVISIONAL_JOB_BYTES, max_gpu_trace_bytes=TRACE_BYTES, max_host_trace_bytes=TRACE_BYTES)
    _CUSTODY['job'] = job
    _guard_owner(identity, session, lease)
    _check_claims(budget, resident, job, 0, 0)
    job.reserve_trace('gpu', TRACE_BYTES)
    _guard_owner(identity, session, lease)
    _check_claims(budget, resident, job, TRACE_BYTES, 0)
    job.reserve_trace('host', TRACE_BYTES)
    def check_copy_boundary():
        _guard_owner(identity, session, lease)
        _check_claims(budget, resident, job, TRACE_BYTES, TRACE_BYTES)
    check_copy_boundary()
    pre_runner_guard = SimpleNamespace(_check_owner=check_copy_boundary)
    probe._bound_runner = pre_runner_guard
    admission = budget.snapshot()
    _check_owner_admission(admission, budget, resident, job, key.device_total_bytes)
    write_artifact('preload-admission.json', _json_bytes(asdict(admission)))

    torch.random.default_generator.manual_seed(17)
    bundle = load_distilgpt2(cache_dir=str(cache), allow_download=False)
    _CUSTODY['bundle'] = bundle
    model = bundle.model
    _CUSTODY['model'] = model
    if verify_trusted_cpu_model(source, model) is not True:
        raise ValueError('pinned CPU checkpoint verification did not succeed')
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise ValueError('model conversion flags are unsupported')
    if prepare_trusted_source(str(repo), str(cache)) != source:
        raise ValueError('source or checkpoint changed during staging')
    check_numeric()
    admission = budget.snapshot()
    _check_owner_admission(admission, budget, resident, job, key.device_total_bytes)
    check_copy_boundary()
    converted = model.to(device='cuda:0', dtype=torch.float32)
    _CUSTODY['converted'] = converted
    if converted is not model:
        raise ValueError('model conversion returned a different object')
    check_copy_boundary()
    torch.cuda.synchronize(0)
    check_copy_boundary()
    if verify_trusted_device_tie(model, identity, session, lease) is not True:
        raise ValueError('device model did not preserve its tied parameter')
    check_copy_boundary()
    torch.cuda.synchronize(0)
    check_copy_boundary()
    write_artifact('model-checks.json', _json_bytes({'cpu_checkpoint_verified': True,
        'same_model_after_copy': True, 'device_tie_verified': True, 'post_copy_synchronized': True}))
    model_ref = weakref.ref(model)
    _check_private_run_dir(run_dir, run_dir_identity, repo, cache)
    journal = AgreementJournal(run_dir / 'journal', {'schema': 'agreement-calibration-journal-v1',
        'run_id': run_id, 'source_sha256': hashlib.sha256(source_bytes).hexdigest()},
        max_total_bytes=1024 ** 2, max_forward_bytes=1024 ** 2, max_events=32)
    _check_private_run_dir(run_dir, run_dir_identity, repo, cache)
    check_copy_boundary()
    if probe._bound_runner is not pre_runner_guard:
        raise ValueError('the provisional telemetry guard changed')
    probe._bound_runner = None
    runner = CudaAgreementCalibrator(model, journal, session=session, device_lease=lease,
        memory_lease=job, resident_lease=resident, memory_probe=probe, expected_key=key)
    _CUSTODY['runner'] = runner
    check_numeric()
    stages = [0]
    def record_stage(record):
        index = stages[0]
        write_artifact(f'stage-{index:03}.json', _json_bytes(record))
        stages[0] += 1
    # The dedicated supervised process makes one resource-only attempt.
    collector.CALIBRATION_EXECUTION_READY = True
    try:
        cold, warm = collector.measure_calibration(runner, key, record_stage)
    finally:
        collector.CALIBRATION_EXECUTION_READY = False
    check_numeric()
    profiles, evidence = assemble_profiles(key, cold, warm, source_bytes,
                                           datetime.now(timezone.utc).isoformat())
    for name, profile, raw in zip(('cold', 'warm'), profiles, evidence):
        write_artifact(name + '-evidence.json', raw)
        write_artifact(name + '-profile.json', profile.to_json().encode('utf-8'))
    if journal.retained_bytes != 0 or journal._events != 14 or stages[0] != 84:
        raise ValueError('calibration journal unexpectedly retained tensor data')
    _check_private_run_dir(run_dir, run_dir_identity, repo, cache)
    journal.finish('complete', {'samples': 14, 'profile_ids': [profile.profile_id for profile in profiles]})
    _check_private_run_dir(run_dir, run_dir_identity, repo, cache)
    runner.dispose_after_success()
    probe._bound_runner = None
    collector._SAMPLING_CUSTODY[:] = [None] * len(collector._SAMPLING_CUSTODY)
    _CUSTODY['runner'] = _CUSTODY['bundle'] = _CUSTODY['model'] = _CUSTODY['converted'] = None
    del runner, model, converted, bundle
    gc.collect()
    if model_ref() is not None:
        raise ValueError('model remains referenced after adapter disposal')
    cleanup = _guarded_cleanup(session, lease, torch, budget, resident, job)
    write_artifact('cleanup.json', _json_bytes(cleanup))
    if cleanup['after'] != {'allocated_bytes': 0, 'reserved_bytes': 0}:
        raise ValueError('allocator remains nonempty after model disposal')
    _release_claims(identity, session, lease, budget, resident, job)
    _guard_owner(identity, session, lease)
    if session._runtime_verified is not True:
        raise ValueError('supervised runtime binding was lost')
    measurement = {'schema': 'agreement-calibration-candidate-v1', 'run_id': run_id,
        'profile_ids': [profile.profile_id for profile in profiles],
        'estimates_bytes': [profile.estimated_incremental_bytes for profile in profiles],
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(),
        'asset_sha256': hashlib.sha256(source.asset_manifest).hexdigest(),
        'run_manifest_sha256': run_manifest_sha256,
        'sample_count': 14, 'stage_count': stages[0], 'workspace_cleanup': cleanup,
        'limitations': ['provisional model-load commitment is not a measured bound',
                        'single observations are not worst-case VRAM bounds',
                        'CUDA initialization occurs before allocator and budget commitments',
                        'driver version is observed by the controller, not queried by the child',
                        'allocator fraction does not cap all CUDA or native memory',
                        'profile files are candidates until the parent validates the clean reaped owner']}
    return {'measurement': measurement, 'cleanup': {'runtime_uuid_verified': True,
        'active_memory_leases': 0, 'device_synchronized': True, 'model_weakref_dead': True,
        'allocator_allocated_bytes': cleanup['after']['allocated_bytes'],
        'allocator_reserved_bytes': cleanup['after']['reserved_bytes'],
        'pytorch_cublas_cache_clear_completed': True, 'trace_gpu_bytes': 0, 'trace_host_bytes': 0}}
