"""Measurements for the fixed, supervised custody-bound service protocol.

Raw sample production requires a real owner, lease, and memory commitments.
Profiles describe observed incremental usage; they are not resource bounds.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import traceback
from types import SimpleNamespace
import weakref

from .resource_profile import (MeasuredResourceProfile, MemorySample, ProfileProvenance,
                               ProfileRegistry, canonical_fingerprint, encode_evidence)
from .custody_service_profile import resolve_custody_service_profiles
from .trusted_service_profile import (FORWARD_CASE_SHA256, JOB_COMMITMENT,
    LOAD_CASE_SHA256, LOAD_COMMITMENT, TrustedServiceIdentity, make_fixed_service_inputs)


TRUSTED_SERVICE_COLLECTOR_CONTRACT = 'two-token-custody-worker-profile-v1'


class CollectorError(RuntimeError):
    """A sample, evidence record or worker result did not meet the protocol."""


def _require_gated_supervisor():
    """Require the reviewed spawn/launch-grant interface from this checkout."""
    from . import supervisor
    expected = Path(__file__).resolve().with_name('supervisor.py')
    if (getattr(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT', None)
            != 'gated-rooted-child-v1'
            or Path(supervisor.__file__).resolve(strict=True) != expected):
        raise CollectorError('the reviewed gated child-launch supervisor is required')


def _validated_hf_home(cache_root, hf_home):
    """Bind the offline checkpoint cache to an explicit existing HF home."""
    if not isinstance(cache_root, (str, os.PathLike)) or not isinstance(hf_home, (str, os.PathLike)):
        raise ValueError('explicit cache and Hugging Face home paths are required')
    cache, home = Path(cache_root), Path(hf_home)
    if not cache.is_absolute() or not home.is_absolute():
        raise ValueError('cache and Hugging Face home must be absolute')
    cache, home = cache.resolve(strict=True), home.resolve(strict=True)
    if not cache.is_dir() or not home.is_dir() or cache == home or not cache.is_relative_to(home):
        raise ValueError('offline checkpoint cache must be inside the Hugging Face home')
    return home


def _workspace_root(repo, requested):
    """Use the trusted shared checkout layout's device lock namespace.

    An unrelated checkout outside that layout must be coordinated separately;
    this rule does not supply a machine-global lock.
    """
    repo, requested = Path(repo).resolve(strict=True), Path(requested)
    if not requested.is_absolute():
        raise CollectorError('the workspace root must be absolute')
    requested = requested.resolve(strict=True)
    work_parents = [parent for parent in repo.parents if parent.name.casefold() == 'work']
    if len(work_parents) > 1:
        raise CollectorError('checkout has an ambiguous nested work layout')
    expected = work_parents[0].parent if work_parents else repo
    if requested != expected or not (expected / 'work').is_dir():
        raise CollectorError('workspace root does not match the shared checkout layout')
    return expected


def _require_workspace_spec(session):
    """Reject an alternate serialized lock namespace before lease acquisition."""
    from .supervisor import OwnerSession, _resolve_child_identity
    if type(session) is not OwnerSession or type(session._identity_spec) is not dict:
        raise CollectorError('a serialized supervised physical identity is required')
    spec = session._identity_spec
    if type(spec.get('workspace_root')) is not str:
        raise CollectorError('the supervised workspace root is missing')
    repo = Path(__file__).resolve().parents[1]
    expected = _workspace_root(repo, spec['workspace_root'])
    identity = _resolve_child_identity(spec, session.device_lease_path, session.marker_path)
    if identity.workspace_root != expected:
        raise CollectorError('the supervised identity uses another workspace')


def _require_workspace_binding(session):
    """Recheck the resolved physical identity against its serialized spec."""
    from .device_identity import DeviceIdentity
    from .supervisor import OwnerSession, _identity_spec
    if type(session) is not OwnerSession or type(session.identity) is not DeviceIdentity:
        raise CollectorError('a resolved supervised physical identity is required')
    identity = session.identity
    _workspace_root(Path(__file__).resolve().parents[1], identity.workspace_root)
    if type(session._identity_spec) is not dict or _identity_spec(identity) != session._identity_spec:
        raise CollectorError('the supervised physical identity changed')
    DeviceIdentity.validate_paths(identity)


# These slots exist before any admission or CUDA work. The dedicated owner
# process keeps its budget, leases, model and failure traceback until exit.
_OWNER_CUSTODY = [None] * 9
_OWNER_CUSTODY_LOCK = threading.Lock()
_OWNER_CLAIMED = False
_PROCESS_CUSTODY = [None] * 4


def _owner_unsafe(state):
    module = sys.modules.get('looking_glass.resident_worker')
    process_check = getattr(module, '_is_process_quarantined', None) if module is not None else None
    worker = state['worker']
    authority = state.get('authority')
    return (state['unsafe'] or (process_check is not None and process_check())
            or (authority is not None and authority.unsafe)
            or (worker is not None and worker.quarantined))


def _check_retained_owner(session, lease):
    """Check the dedicated owner's state and both process-lifetime claims."""
    from .supervisor import OwnerSession
    state, budget, resident, forward = (_OWNER_CUSTODY[i] for i in (3, 5, 6, 7))
    if state is None or _owner_unsafe(state):
        raise CollectorError('device completion is unsafe or unverified')
    OwnerSession.check(session, lease)
    if (budget is None or resident is None or forward is None
            or resident is forward or resident.released or forward.released
            or resident.estimated_incremental_bytes != LOAD_COMMITMENT
            or forward.estimated_incremental_bytes != JOB_COMMITMENT):
        raise CollectorError('the process-lifetime memory commitments are unavailable')
    active = budget.active_leases()
    if (len(active) != 2 or resident not in active or forward not in active
            or any(claim.trace_bytes(location) for claim in active for location in ('gpu', 'host'))):
        raise CollectorError('the process-lifetime memory commitments changed')
    return {'active_leases': 2, 'resident_bytes': LOAD_COMMITMENT,
            'forward_bytes': JOB_COMMITMENT,
            'committed_incremental_bytes': LOAD_COMMITMENT + JOB_COMMITMENT,
            'retained_until_exit': True}


def _guarded_cleanup(session, lease, torch):
    """Check the unsafe latch and both claims around every cleanup CUDA call."""
    from .cuda_cleanup import clear_cublas_after_drain
    def guarded(function):
        def call(*args, **kwargs):
            _check_retained_owner(session, lease)
            value = function(*args, **kwargs)
            _check_retained_owner(session, lease)
            return value
        return call
    cuda = SimpleNamespace(**{name: guarded(getattr(torch.cuda, name)) for name in
        ('is_initialized', 'synchronize', 'empty_cache', 'memory_allocated', 'memory_reserved')})
    wrapped = SimpleNamespace(__version__=torch.__version__, version=torch.version, cuda=cuda,
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=guarded(torch._C._cuda_clearCublasWorkspaces)))
    return clear_cublas_after_drain(session=session, lease=lease, torch=wrapped)


def _dispose_custody_worker(state):
    """Dispose a confirmed idle worker cycle while claims remain held.

    The caller checks the physical owner before and after this operation.
    An unsafe or still-custodied worker stays rooted for process termination.
    """
    from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
    from .resident_worker import ResidentWorker
    owned = state['worker']
    authority = state.get('authority')
    try:
        if (type(owned) is not LifetimeWorkspaceWorker or authority is not owned._authority
                or type(owned._worker) is not ResidentWorker or _owner_unsafe(state) or owned.poisoned
                or state.get('operation_in_flight', False) is not False
                or owned._cold_pending or owned._warmup_result is not None
                or owned._custody.result is not None or owned._custody.request is not None
                or owned._custody.logits is not None
                or owned._worker._snapshot_error is not None
                or owned._custody.forward_error is not None or owned._custody.state_error is not None
                or owned._custody.producer_error is not None
                or owned._custody.error is not None or owned._retiring_transfer is not None
                or owned._retiring_cancel is not None or owned._transfer_cleanup_error is not None
                or owned._cancel_cleanup_error is not None):
            raise CollectorError('worker is unsafe or still holds a forward result or request')
        resident = owned._worker
        owned.close()
        if authority.unsafe:
            raise CollectorError('owner became unsafe while closing the worker')
        resident.close()
        if authority.unsafe:
            raise CollectorError('owner became unsafe while closing the resident')
        # The authority's retirement decision permanently forbids rebinding and
        # severs its worker root only after verifying the closed, idle worker.
        # Keep both this root and the authoritative latch through the decision.
        authority.retire_unbrokered_worker(owned)
        if authority.unsafe:
            raise CollectorError('owner became unsafe before worker retirement')
        state['worker'] = None
        if authority.unsafe:
            state['worker'] = owned
            raise CollectorError('owner became unsafe during worker retirement')
        state['bundle'] = None
        state['copy_result'] = None
        del resident, owned
        gc.collect()
        if authority.unsafe:
            raise CollectorError('owner became unsafe during worker retirement')
        if state['model_ref']() is not None:
            raise CollectorError('the model remains referenced after worker shutdown')
    except BaseException:
        # Never lose an interrupted retirement's sticky authority. The
        # dedicated owner keeps its worker/model roots where still available
        # and both ledger claims until process exit.
        state['unsafe'] = True
        if type(authority) is OwnerCompletionAuthority:
            try:
                authority.mark_unsafe()
            except BaseException:
                pass
        raise


class _GuardedProbe:
    """Deny CUDA callbacks once a worker or owner has unverified completion."""
    def __init__(self, inner, state, session, lease):
        self._inner, self._state, self._session, self._lease = inner, state, session, lease
        self.device, self.device_uuid = inner.device, inner.device_uuid

    def _check(self, *, synchronize=False):
        from .supervisor import OwnerSession
        if (_owner_unsafe(self._state)
                or (synchronize and self._state['operation_in_flight'])):
            raise CollectorError('device completion is unsafe or unverified')
        OwnerSession.check(self._session, self._lease)

    def runtime_binding(self):
        self._check()
        return self._inner.runtime_binding()
    def synchronize(self):
        self._check(synchronize=True)
        return self._inner.synchronize()
    def snapshot(self):
        from .memory_budget import CudaMemorySnapshot
        self._check()
        free, total = self._inner._cuda.mem_get_info(self.device)
        self._check()
        allocated = self._inner._cuda.memory_allocated(self.device)
        self._check()
        reserved = self._inner._cuda.memory_reserved(self.device)
        self._check()
        return CudaMemorySnapshot(free, total, allocated, reserved)
    def reset_peak(self):
        self._check()
        return self._inner.reset_peak()
    def peak_allocated_bytes(self):
        self._check()
        return self._inner.peak_allocated_bytes()
    def peak_reserved_bytes(self):
        self._check()
        return self._inner.peak_reserved_bytes()


def _import_uninitialized_torch():
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise CollectorError('CUDA-capable modules were imported before owner acquisition')
    torch = importlib.import_module('torch')
    if torch.cuda.is_initialized() is not False:
        raise CollectorError('CUDA was initialized before the supervised runtime check')
    return torch


def assemble_profiles(identity, samples, *, collected_utc):
    """Build exactly one load, one cold and three ordered warm profiles.

    The returned IDs are measurements to be independently approved, not
    authorization for service use. The source and samples stay unredacted in
    the private evidence directory because they identify the physical device.
    """
    if type(identity) is not TrustedServiceIdentity or type(samples) is not tuple or len(samples) != 5:
        raise ValueError('an exact trusted identity and five samples are required')
    expected_seeds = (17, 17, 29, 43, 71)
    expected_cases = (LOAD_CASE_SHA256, FORWARD_CASE_SHA256, FORWARD_CASE_SHA256,
                      FORWARD_CASE_SHA256, FORWARD_CASE_SHA256)
    keys = (identity.keys[0], identity.keys[1], identity.keys[2], identity.keys[2], identity.keys[2])
    if any(type(sample) is not MemorySample or sample.seed != seed or sample.case_sha256 != case
           or sample.workload_sha256 != key.fingerprint()
           for sample, seed, case, key in zip(samples, expected_seeds, expected_cases, keys)):
        raise ValueError('sample order, case, seed or workload identity differs from the protocol')
    if len({sample.sample_id for sample in samples}) != 5:
        raise ValueError('sample identifiers must be distinct')
    source_hash = hashlib.sha256(identity.source_manifest).hexdigest()
    groups = ((identity.keys[0], samples[:1], 512 * 1024 ** 2, LOAD_COMMITMENT),
              (identity.keys[1], samples[1:2], 256 * 1024 ** 2, JOB_COMMITMENT),
              (identity.keys[2], samples[2:], 256 * 1024 ** 2, JOB_COMMITMENT))
    profiles, evidences = [], []
    for key, group, margin, commitment in groups:
        evidence = encode_evidence(key, group)
        provenance = ProfileProvenance('resource-profile-v1', source_hash,
            hashlib.sha256(evidence).hexdigest(), collected_utc,
            'cooperative-exclusive', 'synchronize-reset-synchronize')
        profile = MeasuredResourceProfile.build(key, group, provenance,
                                                margin_bytes=margin, margin_bps=2500)
        if profile.estimated_incremental_bytes > commitment:
            raise CollectorError('measured profile exceeds its predeclared commitment')
        profiles.append(profile)
        evidences.append(evidence)
    profiles = tuple(profiles)
    pins = tuple(profile.profile_id for profile in profiles)
    registry = ProfileRegistry(profiles, approved_profile_ids=pins)
    # This tests schema/case consistency, not independent approval of the pins.
    resolve_custody_service_profiles(registry, identity, pins)
    return profiles, tuple(evidences)


def _representative_frame_size(result, request_id):
    """Measure one actual production IPC frame using a fixed test envelope."""
    from .ipc import IPCHandle, IPCServer
    limit = 4 * 1024 ** 2
    broker_id, team_id, job_id = 'a' * 32, 'bootstrap', 1
    server = IPCServer(SimpleNamespace(broker_id=broker_id), max_response_bytes=limit - 4,
                       io_timeout=30.0)
    outcome = SimpleNamespace(broker_id=broker_id, request_id=request_id, team_id=team_id,
        job_id=job_id, state='succeeded', result=result, submitted_ns=2**63 - 1,
        ready_ns=2**63 - 1, started_ns=2**63 - 1, finished_ns=2**63 - 1)
    reader, writer = socket.socketpair()
    reader.setblocking(False)
    writer.setblocking(False)

    class CountedReader:
        def __init__(self, sock):
            self.sock, self.count = sock, 0
        def fileno(self): return self.sock.fileno()
        def recv_into(self, buffer, count):
            size = self.sock.recv_into(buffer, count)
            self.count += size
            if self.count > limit:
                raise CollectorError('framed response exceeds 4 MiB')
            return size
        def shutdown(self, how): return self.sock.shutdown(how)
        def close(self): return self.sock.close()

    counted = CountedReader(reader)
    handle = IPCHandle(counted, request_id, team_id, broker_id, job_id, 30.0, limit - 4)
    responses, errors = [], []
    def drain():
        try:
            responses.append(IPCHandle.wait(handle, timeout=35.0))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=drain, name='profile-ipc-drain', daemon=True)
    thread.start()
    try:
        IPCServer._send_outcome(server, writer, outcome, SimpleNamespace(job_id=job_id), request_id, team_id)
    finally:
        try:
            writer.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        writer.close()
        thread.join(35.0)
        reader.close()
    if thread.is_alive() or errors or len(responses) != 1:
        raise CollectorError('IPC response could not be fully drained')
    response = responses[0]
    if type(response) is not dict or response.get('status') != 'succeeded':
        raise CollectorError('the production IPC encoder rejected the worker result')
    payload = response.get('result')
    if (type(payload) is not dict or set(payload) != {'shape', 'dtype', 'logits'}
            or payload['shape'] != [1, 2, 50257] or payload['dtype'] != 'torch.float32'
            or type(payload['logits']) is not list or len(payload['logits']) != 100514
            or any(type(value) is not float or not math.isfinite(value) for value in payload['logits'])):
        raise CollectorError('decoded IPC payload does not match the fixed output contract')
    if not 201031 <= counted.count <= limit:
        raise CollectorError('production response frame is outside its byte bounds')
    return counted.count


def run_fixed_forward(worker, seed, *, cold, allow_cpu_test=False):
    """Measure the custody-bound worker; CPU mode is solely for tests."""
    import torch
    from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
    from .resident_worker import ForwardRequest, ForwardResult, ResidentWorker
    if type(worker) is not LifetimeWorkspaceWorker:
        raise TypeError('the exact lifetime workspace worker is required')
    resident, authority = worker._worker, worker._authority
    if (type(resident) is not ResidentWorker or type(authority) is not OwnerCompletionAuthority
            or resident._owner_completion is not authority or authority._worker is not worker):
        raise CollectorError('worker and completion authority are not correctly bound')
    expected = 'cpu' if allow_cpu_test is True else 'cuda'
    if (resident.device.type != expected or (expected == 'cuda' and resident.device.index != 0)
            or resident.max_batch_size != 1 or resident.max_sequence_length != 2
            or resident.max_input_bytes != 64 * 1024 or resident.max_output_bytes != 512 * 1024):
        raise CollectorError('worker device or bounds differ from the protocol')
    try:
        if type(cold) is not bool or worker._cold_pending is not cold:
            raise CollectorError('worker cold/warm state differs from the sample')
        inputs = make_fixed_service_inputs([15496, 995], seed, cold=cold)
        tokens = inputs['input_ids']
        actual_case = {'schema': 'resource-bootstrap-case-v1', 'input_ids': tokens.tolist()[0],
                       'shape': list(tokens.shape), 'dtype': str(tokens.dtype)}
        case_sha256 = canonical_fingerprint(actual_case)
        if case_sha256 != FORWARD_CASE_SHA256:
            raise CollectorError('constructed input does not match the fixed case digest')
        request = ForwardRequest('profile-case-v1', inputs, seed)
        result = LifetimeWorkspaceWorker.execute(worker, request)
        if type(result) is not ForwardResult or result.request_id != request.request_id:
            raise CollectorError('worker returned an invalid result identity')
        logits = result.logits
        if (type(logits) is not torch.Tensor or logits.device.type != 'cpu'
                or logits.dtype is not torch.float32 or tuple(logits.shape) != (1, 2, 50257)
                or not logits.is_contiguous() or logits.storage_offset() != 0
                or logits.untyped_storage().nbytes() != 402056 or logits.requires_grad
                or not bool(torch.isfinite(logits).all())):
            raise CollectorError('worker output differs from the finite fixed output contract')
        frame = _representative_frame_size(result, request.request_id)
        output_ref, input_ref = weakref.ref(logits), weakref.ref(tokens)
        if cold:
            LifetimeWorkspaceWorker._finish_cold(worker, result)
        else:
            # The local result is the consumer root while producer custody is
            # transferred by the worker's own authority-ordered decision.
            LifetimeWorkspaceWorker._transfer_result(worker, result)
        if authority.unsafe or worker.quarantined:
            raise CollectorError('owner became unsafe during result transfer')
        del logits, result, request, tokens, inputs
        if output_ref() is not None or input_ref() is not None:
            raise CollectorError('collector-owned input or output remains referenced')
        return {'seed': seed, 'cold': cold, 'case_sha256': case_sha256,
                'finite': True, 'output_shape': [1, 2, 50257],
                'output_dtype': 'torch.float32', 'output_tensor_bytes': 402056,
                'representative_frame_bytes': frame,
                'frame_scope': 'production IPC envelope, shape, dtype and finite count; no numerical parity claim'}
    except BaseException:
        authority.mark_unsafe()
        raise


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _load_case_digest(source):
    """Derive the load case from the actual independently verified assets."""
    manifest = json.loads(source.asset_manifest)
    if type(manifest) is not dict or type(manifest.get('assets')) is not list:
        raise CollectorError('the verified asset manifest is malformed')
    records = manifest['assets']
    if len(records) != 7 or len({item.get('name') for item in records if type(item) is dict}) != 7:
        raise CollectorError('the verified asset manifest must contain seven distinct assets')
    assets = {item['name']: item['sha256'] for item in records}
    case = {'schema': 'resource-bootstrap-load-case-v1', 'operation': 'model_load',
            'checkpoint_sha256': assets['model.safetensors'], 'config_sha256': assets['config.json']}
    digest = canonical_fingerprint(case)
    if digest != LOAD_CASE_SHA256:
        raise CollectorError('the actual verified load assets differ from the fixed case')
    return digest


def _write_new(path, data):
    """Write all bytes and publish a new artifact without replacing a file."""
    path = Path(path)
    if type(data) is not bytes or os.path.lexists(path):
        raise FileExistsError(path)
    scratch = path.with_name(path.name + '.partial')
    fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if not written:
                raise OSError('artifact write made no progress')
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    if os.name == 'nt':
        os.rename(scratch, path)  # Windows refuses to replace an existing destination.
    else:
        os.link(scratch, path)
        try:
            os.unlink(scratch)
        except OSError:
            pass  # The no-clobber final link remains authoritative.


def _write_json(path, value):
    _write_new(path, json.dumps(value, ensure_ascii=True, sort_keys=True,
                                separators=(',', ':'), allow_nan=False).encode('utf-8'))


_STAGE_NAMES = ('load-17', 'cold-17', 'warm-29', 'warm-43', 'warm-71')
_PROFILE_NAMES = ('load', 'cold', 'warm')
_ARTIFACT_NAMES = (('owner.json', 'source-manifest.json', 'runtime-manifest.json',
                    'asset-manifest.json', 'load-17-checks.json', 'cleanup.json')
                   + tuple(name + suffix for name in _STAGE_NAMES
                           for suffix in ('-admission.json', '-after-observed.json', '-postcheck.json', '-sample.json'))
                   + tuple(name + '-output.json' for name in _STAGE_NAMES[1:])
                   + tuple(name + suffix for name in _PROFILE_NAMES
                           for suffix in ('-evidence.json', '-profile.json')))


def _bounded_read(path, *, limit=32 * 1024 ** 2):
    info = Path(path).lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or info.st_size > limit or info.st_nlink != 1):
        raise CollectorError('artifact is not a bounded ordinary file')
    with open(path, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise CollectorError('artifact changed before reading')
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        if (after.st_dev, after.st_ino, after.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise CollectorError('artifact changed while reading')
    if len(data) != info.st_size or len(data) > limit:
        raise CollectorError('artifact changed while being read')
    return data


def _private_text(value, limit):
    """Bound private diagnostic text in UTF-8 bytes, including broken output."""
    if type(value) is str:
        raw = value[:limit + 1].encode('utf-8', errors='replace')
        truncated = len(value) > limit or len(raw) > limit
    elif type(value) is bytes:
        raw = value[:limit + 1].decode('utf-8', errors='replace').encode('utf-8')
        truncated = len(value) > limit or len(raw) > limit
    else:
        return {'text': None, 'truncated': False}
    return {'text': raw[:limit].decode('utf-8', errors='ignore'), 'truncated': truncated}


def _private_driver_failure(error):
    """Extract bounded nvidia-smi failure details for the private artifact only."""
    if type(error) is not ValueError or str(error) != 'driver inventory is unavailable':
        return None
    cause = error.__cause__
    command = ['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits']
    if isinstance(cause, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        if cause.cmd != command:
            return None
    elif not isinstance(cause, OSError):
        return None
    allowed = ('PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PROGRAMFILES',
               'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')
    return {'failure_type': type(cause).__name__, 'command': command,
            'returncode': getattr(cause, 'returncode', None),
            'errno': getattr(cause, 'errno', None),
            'winerror': getattr(cause, 'winerror', None),
            'stdout': _private_text(getattr(cause, 'stdout', None), 4096),
            'stderr': _private_text(getattr(cause, 'stderr', None), 4096),
            'environment': {key: _private_text(os.environ.get(key), 1024) for key in allowed}}


def _owner_main(config):
    """Short-lived child; failed execution never attests or releases ownership."""
    from .device_lease import DeviceLease
    from .supervisor import OwnerSession
    from .trusted_service_profile import prepare_trusted_source
    session = lease = None
    run_dir = Path(config['run_dir'])
    try:
        # The hub caches HF_HOME at import time, and Transformers may load it.
        # Reject either package or any retained submodule before acquisition;
        # the controlled child environment must precede their first import.
        if any(name == package or name.startswith(package + '.')
               for name in tuple(sys.modules) for package in ('huggingface_hub', 'transformers')):
            raise CollectorError('model dependencies were imported before owner startup')
        _require_gated_supervisor()
        _validated_hf_home(config['cache_root'], os.environ.get('HF_HOME'))
        session = OwnerSession.from_environment()
        _PROCESS_CUSTODY[0] = session
        _require_workspace_spec(session)
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _PROCESS_CUSTODY[1] = lease
        OwnerSession.check(session, lease)
        _require_workspace_binding(session)
        owner = {'schema': 'trusted-profile-owner-v1', 'run_id': config['run_id'],
                 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
                 'physical_gpu_uuid': session.identity.gpu_uuid}
        _write_json(run_dir / 'owner.json', owner)
        source = prepare_trusted_source(config['repo_root'], config['cache_root'])
        if (hashlib.sha256(source.source_base).hexdigest() != config['source_base_sha256']
                or hashlib.sha256(source.asset_manifest).hexdigest() != config['asset_sha256']):
            raise CollectorError('source or assets changed after parent preflight')
        summary = collect_owned(source, session, lease, run_dir, config['run_id'])
        if _check_retained_owner(session, lease) != summary['cleanup']['owner_commitments']:
            raise CollectorError('saved commitments differ from the active owner claims')
        files = {name: hashlib.sha256(_bounded_read(run_dir / name)).hexdigest()
                 for name in _ARTIFACT_NAMES}
        candidate = {'schema': 'trusted-profile-candidate-v1', 'run_id': config['run_id'],
                     'supervisor_run_id': session.run_id, 'pid': os.getpid(),
                     'summary': summary, 'artifact_sha256': files}
        _write_json(run_dir / 'candidate.json', candidate)
        if _check_retained_owner(session, lease) != summary['cleanup']['owner_commitments']:
            raise CollectorError('saved commitments differ from the active owner claims')
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        # Keep traceback/model/ledger roots alive until this dedicated process exits.
        _PROCESS_CUSTODY[2] = exc
        try:
            failure = {'schema': 'trusted-profile-failure-v1', 'run_id': config['run_id'],
                       'error_type': type(exc).__name__,
                       'private_traceback': traceback.format_exc()[-8192:]}
            try:
                diagnostic = _private_driver_failure(exc)
            except BaseException:
                diagnostic = {'capture': 'unavailable'}
            if diagnostic is not None:
                failure['private_driver_query'] = diagnostic
            _write_json(run_dir / 'failure.json', failure)
        except BaseException:
            pass
        return 1


def _same_json_value(actual, expected):
    """Compare evidence structurally without treating JSON booleans as numbers."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys()
                and all(_same_json_value(actual[key], value) for key, value in expected.items()))
    if type(expected) is list:
        return (len(actual) == len(expected)
                and all(_same_json_value(left, right) for left, right in zip(actual, expected)))
    return actual == expected


def _validate_candidate(run_dir, run_id, supervision):
    """Re-read completed artifacts after the exact child has been reaped."""
    from .resource_profile import load_profile
    from .trusted_service_profile import TrustedServiceSource
    run_dir = Path(run_dir)
    candidate = json.loads(_bounded_read(run_dir / 'candidate.json', limit=1024 * 1024))
    if (type(candidate) is not dict or candidate.get('schema') != 'trusted-profile-candidate-v1'
            or candidate.get('run_id') != run_id or candidate.get('supervisor_run_id') != supervision.run_id
            or candidate.get('pid') != supervision.pid or type(candidate.get('summary')) is not dict
            or type(candidate.get('artifact_sha256')) is not dict
            or set(candidate['artifact_sha256']) != set(_ARTIFACT_NAMES)):
        raise CollectorError('candidate does not match the reaped owner')
    artifacts = {name: _bounded_read(run_dir / name) for name in _ARTIFACT_NAMES}
    if any(hashlib.sha256(raw).hexdigest() != candidate['artifact_sha256'][name]
           for name, raw in artifacts.items()):
        raise CollectorError('a candidate artifact changed after collection')
    owner = json.loads(artifacts['owner.json'])
    runtime = json.loads(artifacts['runtime-manifest.json'])
    if (owner.get('schema') != 'trusted-profile-owner-v1' or owner.get('run_id') != run_id
            or owner.get('supervisor_run_id') != supervision.run_id or owner.get('pid') != supervision.pid
            or owner.get('physical_gpu_uuid') != runtime.get('device_uuid')):
        raise CollectorError('saved runtime identity differs from the reaped owner')
    summary = candidate['summary']
    pins = summary.get('profile_ids')
    if type(pins) is not list or len(pins) != 3 or len(set(pins)) != 3:
        raise CollectorError('three distinct profile IDs are required')
    source = artifacts['source-manifest.json']
    profiles = tuple(load_profile(artifacts[name + '-profile.json'], evidence=artifacts[name + '-evidence.json'],
                                  source_manifest=source, expected_profile_id=pin)
                     for name, pin in zip(_PROFILE_NAMES, pins))
    identity = TrustedServiceIdentity(tuple(profile.key for profile in profiles), source,
                                      artifacts['runtime-manifest.json'], artifacts['asset-manifest.json'])
    registry = ProfileRegistry(profiles, approved_profile_ids=tuple(pins))
    resolve_custody_service_profiles(registry, identity, tuple(pins))
    samples = (profiles[0].samples[0], profiles[1].samples[0], *profiles[2].samples)
    actual_source = TrustedServiceSource('', '', b'', artifacts['asset-manifest.json'], ())
    if samples[0].case_sha256 != _load_case_digest(actual_source):
        raise CollectorError('load evidence differs from the verified assets')
    load_checks = json.loads(artifacts['load-17-checks.json'])
    if not _same_json_value(load_checks, {
            'schema': 'trusted-profile-load-checks-v1', 'case_sha256': samples[0].case_sha256,
            'cpu_model_verified': True, 'conversion_flags_false': True,
            'same_model_after_copy': True, 'device_tie_verified': True,
            'post_copy_synchronized': True, 'device': 'cuda:0', 'dtype': 'torch.float32'}):
        raise CollectorError('load checks do not match the fixed startup path')
    for name, sample in zip(_STAGE_NAMES, samples):
        if (sample.sample_id != run_id + '-' + name
                or not _same_json_value(json.loads(artifacts[name + '-sample.json']), asdict(sample))):
            raise CollectorError('stage sample differs from profile evidence')
        admission = json.loads(artifacts[name + '-admission.json'])
        expected = LOAD_COMMITMENT if name == 'load-17' else LOAD_COMMITMENT + JOB_COMMITMENT
        budget = admission.get('budget', {})
        if (admission.get('label') != name or admission.get('lease_bytes') !=
                (LOAD_COMMITMENT if name == 'load-17' else JOB_COMMITMENT)):
            raise CollectorError('saved admission did not satisfy the fixed memory policy')
        _validate_budget_record(budget, expected, 1 if name == 'load-17' else 2)
        if budget['telemetry']['device_total_bytes'] != sample.before.device_total_bytes:
            raise CollectorError('admission device total differs from the measured sample')
        if not _same_json_value(json.loads(artifacts[name + '-after-observed.json']), asdict(sample.after)):
            raise CollectorError('saved post-stage observation differs from the sample')
        postcheck = json.loads(artifacts[name + '-postcheck.json'])
        _validate_budget_record(postcheck, expected, 1 if name == 'load-17' else 2)
        if postcheck['telemetry']['device_total_bytes'] != sample.after.device_total_bytes:
            raise CollectorError('device total changed at the post-stage check')
        _validate_headroom(asdict(sample.after), postcheck['committed_incremental_bytes'],
                           postcheck['reserve_bytes'])
    for name, sample in zip(_STAGE_NAMES[1:], samples[1:]):
        output = json.loads(artifacts[name + '-output.json'])
        if (output.get('seed') != sample.seed or type(output.get('cold')) is not bool
                or output.get('cold') != (name == 'cold-17')
                or output.get('case_sha256') != sample.case_sha256 or output.get('finite') is not True
                or not _same_json_value(output.get('output_shape'), [1, 2, 50257])
                or output.get('output_dtype') != 'torch.float32'
                or output.get('output_tensor_bytes') != 402056
                or type(output.get('representative_frame_bytes')) is not int
                or not 201031 <= output['representative_frame_bytes'] <= 4 * 1024 ** 2):
            raise CollectorError('output evidence differs from its measured sample')
    cleanup = json.loads(artifacts['cleanup.json'])
    if (not _same_json_value(cleanup, summary.get('cleanup')) or cleanup.get('model_weakref_dead') is not True
            or not _same_json_value(cleanup.get('after'), {'allocated_bytes': 0, 'reserved_bytes': 0})
            or not _same_json_value(cleanup.get('owner_commitments'), {
                'active_leases': 2, 'resident_bytes': LOAD_COMMITMENT, 'forward_bytes': JOB_COMMITMENT,
                'committed_incremental_bytes': LOAD_COMMITMENT + JOB_COMMITMENT, 'retained_until_exit': True})):
        raise CollectorError('clean allocator evidence is absent')
    if (summary.get('source_sha256') != hashlib.sha256(source).hexdigest()
            or summary.get('runtime_sha256') != hashlib.sha256(artifacts['runtime-manifest.json']).hexdigest()
            or summary.get('asset_sha256') != hashlib.sha256(artifacts['asset-manifest.json']).hexdigest()
            or summary.get('estimates_bytes') != [p.estimated_incremental_bytes for p in profiles]
            or summary.get('seeds') != [17, 17, 29, 43, 71]):
        raise CollectorError('candidate summary differs from the validated evidence')
    return summary


def _validate_headroom(telemetry, committed, reserve):
    fields = ('device_total_bytes', 'device_free_bytes', 'process_allocated_bytes', 'process_reserved_bytes')
    if type(telemetry) is not dict or any(type(telemetry.get(field)) is not int for field in fields):
        raise CollectorError('invalid device telemetry')
    total, free, allocated, reserved = (telemetry[field] for field in fields)
    if (total <= 0 or not 0 <= free <= total or not 0 <= allocated <= reserved <= total
            or reserve != max(1024 ** 3, (total * 15 + 99) // 100)
            or committed + reserve > free or reserved + committed + reserve > total):
        raise CollectorError('device headroom is insufficient after a stage')


def _validate_budget_record(budget, expected, active):
    if (type(budget) is not dict or budget.get('committed_incremental_bytes') != expected
            or type(budget.get('committed_incremental_bytes')) is not int
            or budget.get('active_leases') != active or type(budget.get('active_leases')) is not int):
        raise CollectorError('commitments changed before the post-stage check')
    _validate_headroom(budget.get('telemetry'), expected, budget.get('reserve_bytes'))


def _unique_json_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate configuration key')
        value[key] = item
    return value


def _read_owner_config(path, identity_text, expected_sha256):
    """Capture the exact parent-pinned private config before any CUDA work."""
    from .trusted_service_launch import read_trusted_config
    if type(identity_text) is not str or re.fullmatch(r'[0-9]+:[0-9]+', identity_text) is None:
        raise ValueError('invalid configuration file identity')
    dev, ino = (int(item) for item in identity_text.split(':'))
    raw = read_trusted_config(path, expected_identity=(dev, ino), expected_sha256=expected_sha256)
    config = json.loads(raw, object_pairs_hook=_unique_json_pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite JSON')))
    required = {'schema', 'repo_root', 'cache_root', 'run_dir', 'run_id',
                'source_base_sha256', 'asset_sha256'}
    if (type(config) is not dict or set(config) != required
            or config['schema'] != 'trusted-profile-owner-config-v1'
            or any(type(config[key]) is not str for key in required)
            or re.fullmatch(r'[0-9a-f]{32}', config['run_id']) is None
            or any(re.fullmatch(r'[0-9a-f]{64}', config[key]) is None
                   for key in ('source_base_sha256', 'asset_sha256'))):
        raise ValueError('invalid private collector configuration')
    for key in ('repo_root', 'cache_root', 'run_dir'):
        selected = Path(config[key])
        if not selected.is_absolute() or not selected.is_dir() or selected.resolve() != selected:
            raise ValueError('collector paths must be canonical absolute directories')
    if (Path(config['repo_root']) != Path(__file__).resolve().parents[1]
            or Path(path).resolve().parent != Path(config['run_dir'])):
        raise ValueError('config path or selected source checkout does not match the owner')
    if (not sys.dont_write_bytecode or not sys.flags.no_site or not sys.flags.safe_path
            or not sys.pycache_prefix):
        raise ValueError('collector must use the reviewed direct-interpreter flags')
    return config


def run_supervised_collector(*, workspace_root, cache_root, hf_home, site_packages, environment_root,
                             private_parent, trusted_path, timeout=600):
    """Run one fixed collector in a dedicated supervised process.

    Private artifacts contain the device UUID and absolute paths. Returned IDs
    require independent approval and are not service authorization. The
    supervisor retains its unsafe marker after a nonzero or unknown child exit,
    even if that child previously attested a clean drain. The
    caller must coordinate the GPU window with other participating owners.
    """
    from .device_identity import resolve_device_identity
    from .gpu_oneshot import _owner_presence_after_exit, _smi_snapshot
    from .supervisor import run_supervised
    from .trusted_service_launch import prepare_trusted_launch
    from .trusted_service_profile import prepare_trusted_source

    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    cache = Path(cache_root).resolve(strict=True)
    home = _validated_hf_home(cache, hf_home)
    private = Path(private_parent).resolve(strict=True)
    if private.is_relative_to(repo):
        raise ValueError('private evidence must be outside the frozen source checkout')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=repo,
                           capture_output=True, text=True, check=True, timeout=10).stdout
    branch = subprocess.run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=repo, capture_output=True,
                            text=True, timeout=10)
    if re.fullmatch(r'[0-9a-f]{40}', revision) is None or dirty or branch.returncode != 1:
        raise CollectorError('collector requires a clean detached source checkout')
    run_dir = Path(tempfile.mkdtemp(prefix='trusted-profile-', dir=private))
    run_id = secrets.token_hex(16)
    _write_json(run_dir / 'started.json', {'schema': 'trusted-profile-start-v1', 'run_id': run_id,
                'source_commit': revision, 'started_utc': utc_now()})
    try:
        _require_gated_supervisor()
        source = prepare_trusted_source(str(repo), str(cache))
        system_root = Path(os.environ.get('SYSTEMROOT', 'C:/Windows'))
        gpu_uuid, before = _smi_snapshot(str(system_root / 'System32' / 'nvidia-smi.exe'))
        _write_json(run_dir / 'inventory-before.json', before)
        config = {'schema': 'trusted-profile-owner-config-v1', 'repo_root': str(repo),
                  'cache_root': str(cache), 'run_dir': str(run_dir), 'run_id': run_id,
                  'source_base_sha256': hashlib.sha256(source.source_base).hexdigest(),
                  'asset_sha256': hashlib.sha256(source.asset_manifest).hexdigest()}
        config_path = run_dir / 'config.json'
        _write_json(config_path, config)
        launch = prepare_trusted_launch(repo_root=str(repo), site_packages=str(site_packages),
            environment_root=str(environment_root), private_parent=str(private), gpu_uuid=gpu_uuid,
            module='looking_glass.trusted_profile_collector', config_path=str(config_path),
            trusted_path=str(trusted_path), hf_home=str(home))
        identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=launch.environment)
        launch.verify_ready()
        if prepare_trusted_source(str(repo), str(cache)) != source:
            raise CollectorError('source or assets changed before owner launch')
    except BaseException as exc:
        terminal = {'schema': 'trusted-profile-terminal-v1', 'run_id': run_id,
                    'source_commit': revision, 'status': 'failed_prelaunch', 'supervisor': None,
                    'prelaunch_error_type': type(exc).__name__, 'finished_utc': utc_now(),
                    'profile_ids_approved': False}
        _write_json(run_dir / 'prelaunch-failure.json', {'error_type': type(exc).__name__,
                    'private_traceback': traceback.format_exc()[-8192:]})
        _write_json(run_dir / 'terminal.json', terminal)
        return run_dir, terminal
    supervision, supervisor_error = None, None
    try:
        supervision = run_supervised(launch.command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, timeout=timeout, lease_timeout=30,
            cwd=repo, env=launch.environment, inherit_environment=False, require_zero_exit=True)
    except BaseException as exc:
        supervisor_error = type(exc).__name__
    supervised = (None if supervision is None else {'run_id': supervision.run_id, 'pid': supervision.pid,
                  'exit_code': supervision.exit_code, 'clean': supervision.clean,
                  'timed_out': supervision.timed_out, 'reason': supervision.reason})
    after, telemetry_error = None, None
    try:
        after_uuid, after = _smi_snapshot(str(system_root / 'System32' / 'nvidia-smi.exe'))
        if after_uuid != gpu_uuid:
            raise CollectorError('physical GPU inventory changed after the owner')
        _write_json(run_dir / 'inventory-after.json', after)
    except Exception as exc:
        after, telemetry_error = None, type(exc).__name__
    presence = (_owner_presence_after_exit(str(system_root / 'System32' / 'nvidia-smi.exe'), supervision.pid)
                if supervision is not None else {'query': 'not_attempted', 'owner_pid_listed': None})
    terminal = {'schema': 'trusted-profile-terminal-v1', 'run_id': run_id,
                'source_commit': revision, 'status': 'quarantined', 'supervisor': supervised,
                'supervisor_error_type': supervisor_error, 'finished_utc': utc_now(),
                'post_inventory': after, 'post_inventory_error_type': telemetry_error,
                'owner_presence_after_exit': presence, 'profile_ids_approved': False}
    if supervision is not None and supervision.clean and supervision.exit_code == 0 and not supervision.timed_out:
        terminal['status'] = 'incomplete'
        try:
            launch.verify_ready()
            terminal['summary'] = _validate_candidate(run_dir, run_id, supervision)
            if telemetry_error is None and presence['owner_pid_listed'] is not True:
                terminal['status'] = 'success'
        except Exception as exc:
            terminal['validation_error_type'] = type(exc).__name__
    _write_json(run_dir / 'terminal.json', terminal)
    return run_dir, terminal


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-id', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(argv)
    config = _read_owner_config(args.config, args.config_id, args.config_sha256)
    return _owner_main(config)


def collect_owned(source, session, lease, run_dir, run_id):
    """Collect the fixed protocol under a real supervised GPU owner.

    On any exception this function deliberately does not release claims or
    attest clean. The dedicated caller must retain its device lease and exit;
    the supervisor then preserves the unsafe marker.
    """
    from .device_lease import DeviceLease
    from .supervisor import OwnerSession
    from .trusted_service_profile import TrustedServiceSource

    if type(session) is not OwnerSession or type(lease) is not DeviceLease or type(source) is not TrustedServiceSource:
        raise TypeError('an actual owner, device lease and trusted source are required')
    global _OWNER_CLAIMED
    with _OWNER_CUSTODY_LOCK:
        if _OWNER_CLAIMED:
            raise CollectorError('this owner process has already attempted collection')
        _OWNER_CLAIMED = True
        _OWNER_CUSTODY[0], _OWNER_CUSTODY[1], _OWNER_CUSTODY[2] = session, lease, source
    state = {'bundle': None, 'worker': None, 'authority': None, 'model_ref': None, 'copy_result': None,
             'unsafe': False, 'operation_in_flight': False}
    _OWNER_CUSTODY[3] = state
    try:
        OwnerSession.check(session, lease)
        _require_workspace_binding(session)
        return _collect_owned_inner(source, session, lease, run_dir, run_id, state)
    except BaseException as exc:
        state['unsafe'] = True
        _OWNER_CUSTODY[8] = exc
        raise


def _collect_owned_inner(source, session, lease, run_dir, run_id, state):
    from .live_profile import decode_cuda_uuid
    from .memory_budget import MemoryBudget
    from .resource_profile import TorchMemoryProbe, collect_sample
    from .supervisor import OwnerSession
    from .custody_service_profile import build_custody_service_identity
    from .trusted_service_profile import (prepare_trusted_source, verify_trusted_cpu_model,
                                          verify_trusted_device_tie)

    OwnerSession.check(session, lease)
    _require_workspace_binding(session)
    identity = session.identity
    if identity is None:
        raise CollectorError('a physical GPU identity is required')
    torch = _import_uninitialized_torch()
    OwnerSession.check(session, lease)
    live = build_custody_service_identity(source, identity, session, lease)
    expected_total = live.keys[0].device_total_bytes
    if (type(expected_total) is not int or expected_total <= 0
            or any(type(key.device_total_bytes) is not int or key.device_total_bytes != expected_total
                   for key in live.keys)):
        raise CollectorError('live workload keys have inconsistent device totals')
    run_dir = Path(run_dir)
    for name, data in (('source-manifest.json', live.source_manifest),
                       ('runtime-manifest.json', live.runtime_manifest),
                       ('asset-manifest.json', live.asset_manifest)):
        _write_new(run_dir / name, data)

    def check_owner_safe():
        if _owner_unsafe(state):
            raise CollectorError('device completion is unsafe or unverified')
        OwnerSession.check(session, lease)

    def binding():
        check_owner_safe()
        count = torch.cuda.device_count()
        if type(count) is not int or count != 1:
            raise CollectorError('exactly one visible CUDA device is required')
        check_owner_safe()
        properties = torch.cuda.get_device_properties(0)
        actual_uuid = decode_cuda_uuid(properties.uuid, torch._C._CUuuid)
        check_owner_safe()
        OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=actual_uuid,
                                            device_count=count, cuda_device_index=0)
        check_owner_safe()
        return actual_uuid, count

    raw_probe = TorchMemoryProbe(torch.cuda, device='cuda:0', device_uuid=identity.gpu_uuid,
                                 runtime_binding=binding)
    # The budget is rooted before an acquire can publish a lease; its ledger
    # retains every published lease even if the caller is interrupted.
    probe = _GuardedProbe(raw_probe, state, session, lease)
    _OWNER_CUSTODY[4] = probe
    def telemetry():
        binding()
        snapshot = probe.snapshot()
        if type(snapshot.device_total_bytes) is not int or snapshot.device_total_bytes != expected_total:
            raise CollectorError('budget telemetry device total differs from the live workload')
        return snapshot
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    _OWNER_CUSTODY[5] = budget
    def guarded_budget_snapshot():
        check_owner_safe()
        result = budget.snapshot()
        check_owner_safe()
        return result
    resident_lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    _OWNER_CUSTODY[6] = resident_lease
    actual_load_case = _load_case_digest(source)

    def load_operation():
        from .pretrained import load_distilgpt2
        from .resident_worker import ResidentWorker
        torch.random.default_generator.manual_seed(17)
        bundle = load_distilgpt2(cache_dir=source.cache_root, allow_download=False)
        state['bundle'] = bundle
        model = bundle.model
        config = getattr(model, 'config', None)
        if (model.training is not False or getattr(config, 'use_cache', None) is not False
                or getattr(config, '_attn_implementation', None) != 'eager'
                or getattr(config, 'vocab_size', None) != 50257
                or getattr(config, 'n_positions', None) != 1024):
            raise CollectorError('loaded model mode differs from the trusted service')
        if verify_trusted_cpu_model(source, model) is not True:
            raise CollectorError('CPU checkpoint verification did not explicitly succeed')
        OwnerSession.check(session, lease)
        if prepare_trusted_source(source.repo_root, source.cache_root) != source:
            raise CollectorError('source or assets changed during CPU staging')
        revalidated = build_custody_service_identity(source, identity, session, lease)
        if revalidated != live:
            raise CollectorError('live source or runtime identity changed during CPU staging')
        current = guarded_budget_snapshot()
        if (current.active_leases != 1 or resident_lease not in budget.active_leases()
                or current.committed_incremental_bytes != LOAD_COMMITMENT
                or current.committed_incremental_bytes + current.reserve_bytes > current.telemetry.device_free_bytes
                or current.telemetry.process_reserved_bytes + current.committed_incremental_bytes
                    + current.reserve_bytes > current.telemetry.device_total_bytes):
            raise CollectorError('live headroom or resident commitment changed before device copy')
        OwnerSession.check(session, lease)
        if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
                or torch.__future__.get_swap_module_params_on_conversion() is not False):
            raise CollectorError('module conversion flags changed before device copy')
        copied = model.to(device='cuda:0', dtype=torch.float32)
        state['copy_result'] = copied
        if copied is not model:
            raise CollectorError('device conversion returned a different model')
        check_owner_safe()
        torch.cuda.synchronize(0)
        if verify_trusted_device_tie(model, identity, session, lease) is not True:
            raise CollectorError('post-copy tied parameter verification failed')
        check_owner_safe()
        torch.cuda.synchronize(0)
        check_owner_safe()
        for parameter in model.parameters():
            if parameter.device.type != 'cuda' or parameter.device.index != 0 or parameter.dtype != torch.float32:
                raise CollectorError('model parameter is not float32 on cuda:0')
        parameter = None
        state['model_ref'] = weakref.ref(model)
        state['worker'] = ResidentWorker(live.keys[0].model_id, model, device='cuda:0',
            max_batch_size=1, max_sequence_length=2, max_input_bytes=64 * 1024,
            max_output_bytes=512 * 1024)
        _write_json(run_dir / 'load-17-checks.json', {
            'schema': 'trusted-profile-load-checks-v1', 'case_sha256': actual_load_case,
            'cpu_model_verified': True, 'conversion_flags_false': True,
            'same_model_after_copy': True, 'device_tie_verified': True,
            'post_copy_synchronized': True, 'device': 'cuda:0', 'dtype': 'torch.float32'})
        return None

    def measure(key, operation, *, label, case_sha256, seed, memory_lease):
        before = guarded_budget_snapshot()
        _write_json(run_dir / (label + '-admission.json'), {
            'label': label, 'budget': asdict(before), 'lease_bytes': memory_lease.estimated_incremental_bytes})
        def guarded_operation():
            state['operation_in_flight'] = True
            try:
                value = operation()
                state['operation_in_flight'] = False
                return value
            except BaseException:
                state['unsafe'] = True
                raise
        sample = collect_sample(key, probe, guarded_operation, owner_session=session, device_identity=identity,
            device_lease=lease, memory_lease=memory_lease, sample_id=run_id + '-' + label,
            case_sha256=case_sha256, seed=seed)
        # Preserve what the probe observed even if fresh headroom fails; do not
        # publish it as a completed sample or release its lease in that case.
        _write_json(run_dir / (label + '-after-observed.json'), asdict(sample.after))
        after_budget = asdict(guarded_budget_snapshot())
        _write_json(run_dir / (label + '-postcheck.json'), after_budget)
        expected = LOAD_COMMITMENT if label == 'load-17' else LOAD_COMMITMENT + JOB_COMMITMENT
        _validate_budget_record(after_budget, expected, 1 if label == 'load-17' else 2)
        if after_budget['telemetry']['device_total_bytes'] != sample.after.device_total_bytes:
            raise CollectorError('device total changed at the post-stage check')
        _validate_headroom(asdict(sample.after), expected, after_budget['reserve_bytes'])
        if memory_lease.released or memory_lease not in budget.active_leases():
            raise CollectorError('memory lease changed before post-stage validation')
        check_owner_safe()
        _write_json(run_dir / (label + '-sample.json'), asdict(sample))
        return sample

    samples = [measure(live.keys[0], load_operation, label='load-17',
                       case_sha256=actual_load_case, seed=17, memory_lease=resident_lease)]
    # No per-sample retirement: the fixed forward claim covers every serialized
    # observation and stays rooted, with the resident claim, until process exit.
    job_lease = budget.acquire(JOB_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    _OWNER_CUSTODY[7] = job_lease
    # These descriptors only permit the explicitly bounded calibration. They
    # are not measured profiles or approved pins. The real service path uses
    # the same lifetime worker and owner-bound completion authority.
    from .budgeted_worker import WorkloadProfile
    from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
    provisional = WorkloadProfile('provisional-cold', live.keys[1].model_id, 'cuda:0',
                                  (1, 2), (('input_ids', 'torch.int64'),),
                                  512 * 1024, JOB_COMMITMENT)
    warm_provisional = WorkloadProfile('provisional-warm', live.keys[2].model_id, 'cuda:0',
                                       (1, 2), (('input_ids', 'torch.int64'),),
                                       512 * 1024, JOB_COMMITMENT)
    check_owner_safe()
    authority = OwnerCompletionAuthority(device_guard=check_owner_safe)
    state['authority'] = authority
    try:
        state['worker'] = LifetimeWorkspaceWorker(state['worker'], budget, resident_lease, job_lease,
            profiles=(provisional, warm_provisional), authority=authority)
    except BaseException:
        state['unsafe'] = True
        authority.mark_unsafe()
        raise
    # The state retains the authority even if worker construction is
    # interrupted at its return boundary. Avoid a separate model root here.
    del authority
    for key, seed, cold in ((live.keys[1], 17, True), (live.keys[2], 29, False),
                            (live.keys[2], 43, False), (live.keys[2], 71, False)):
        label = ('cold-' if cold else 'warm-') + str(seed)
        def operation(seed=seed, cold=cold, label=label):
            observation = run_fixed_forward(state['worker'], seed, cold=cold)
            check_owner_safe()
            _write_json(run_dir / (label + '-output.json'), observation)
            return None
        sample = measure(key, operation, label=label, case_sha256=FORWARD_CASE_SHA256,
                         seed=seed, memory_lease=job_lease)
        check_owner_safe()
        _check_retained_owner(session, lease)
        samples.append(sample)

    profiles, evidence = assemble_profiles(live, tuple(samples), collected_utc=utc_now())
    for name, profile, raw in zip(('load', 'cold', 'warm'), profiles, evidence):
        _write_new(run_dir / (name + '-evidence.json'), raw)
        _write_new(run_dir / (name + '-profile.json'), profile.to_json().encode('utf-8'))

    check_owner_safe()
    _check_retained_owner(session, lease)
    _dispose_custody_worker(state)
    check_owner_safe()
    cleanup = _guarded_cleanup(session, lease, torch)
    cleanup['model_weakref_dead'] = True
    cleanup['owner_commitments'] = _check_retained_owner(session, lease)
    _write_json(run_dir / 'cleanup.json', cleanup)
    if cleanup['after'] != {'allocated_bytes': 0, 'reserved_bytes': 0}:
        raise CollectorError('the allocator still owns bytes after worker cleanup')
    _check_retained_owner(session, lease)
    return {'profile_ids': [profile.profile_id for profile in profiles],
            'estimates_bytes': [profile.estimated_incremental_bytes for profile in profiles],
            'source_sha256': hashlib.sha256(live.source_manifest).hexdigest(),
            'runtime_sha256': hashlib.sha256(live.runtime_manifest).hexdigest(),
            'asset_sha256': hashlib.sha256(live.asset_manifest).hexdigest(),
            'seeds': [17, 17, 29, 43, 71], 'cleanup': cleanup}


if __name__ == '__main__':
    # No Python atexit/finalizer may launch GPU work after clean attestation
    # or an uncertain failure. Private evidence is explicitly fsynced above.
    try:
        code = main()
    except BaseException:
        os._exit(1)
    os._exit(code if type(code) is int else 1)
