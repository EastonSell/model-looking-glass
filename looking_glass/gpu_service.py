"""Safety-critical ordering for the supervised GPU service.

This module is not yet a GPU-service launcher. A launcher must independently
verify live profile keys and privately deliver credentials before it can call
the owner in a supervised child. No profile or device is assumed here.
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from threading import Event, Lock

from .memory_budget import MemoryBudget


# A failed device drain is a process-lifetime event. Retain known owner
# objects and a bounded set of failure tracebacks, and stop admitting work.
# Unrecorded CUDA internals can be lost; the supervised process must exit.
_QUARANTINED = Event()
_QUARANTINE_HOLDS = []
_OWNER_UNSAFE = False
_OWNER_DRAIN_PENDING = False
# Preallocate process-lifetime custody before any owner can touch CUDA.  The
# fixed slots hold session, device lease, backend, registry, budget, resident
# lease, worker, broker, server and two failure tracebacks, respectively.  The
# first failure is never overwritten by a later failure while quarantining.
# The process serves only one owner and must exit after its result, even on success.
_OWNER_CUSTODY = [None] * 11
_OWNER_CLAIMED = False
_OWNER_CUSTODY_LOCK = Lock()
_RAW_RUNTIME_UUID = re.compile(r'[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z', re.ASCII)
_WARMUP_PROTOCOL_SHA256 = '84b21a9ba63413bb9394dbac6d31dbc08ce3171508a25477f44e596be579760c'
_NO_CAPTURE_POLICY_SHA256 = '1ec2c8752f73cab7403fa9012e0796de77fdd87eb395a7cbc796044bf1643250'
def _canonical_runtime_uuid(value, expected_type):
    """Parse the expected PyTorch CUDA runtime UUID representation exactly."""
    if type(expected_type) is not type and not isinstance(expected_type, type):
        raise TypeError('the runtime UUID type is unavailable')
    if type(value) is not expected_type:
        raise TypeError('unexpected CUDA runtime UUID type')
    raw = str(value)
    if not _RAW_RUNTIME_UUID.fullmatch(raw):
        raise ValueError('unexpected CUDA runtime UUID rendering')
    return 'GPU-' + raw


def _validate_service_keys(identity, keys):
    """Enforce the single supported signature and cold-to-warm transition."""
    from .pretrained import ASSETS, REVISION

    if type(keys) is not tuple or len(keys) != 3:
        raise ValueError('expected model-load, cold-forward and warm-forward keys')
    load, cold, warm = keys
    try:
        if any(
            key.model_id != load.model_id or key.device != 'cuda:0'
            or key.device_uuid != identity.gpu_uuid
            or key.checkpoint_sha256 != ASSETS['model.safetensors'][1]
            or key.checkpoint_revision != REVISION
            or key.model_config_sha256 != ASSETS['config.json'][1]
            or key.model_dtype != 'torch.float32' or key.compute_dtype != 'torch.float32'
            or key.attention_backend != 'eager' or key.model_mode != 'eval'
            or key.grad_mode != 'inference' or key.autocast != 'disabled'
            or key.use_cache is not False or key.compiled is not False
            or key.capture_mode != 'none' or key.max_gpu_trace_bytes != 0
            or key.max_host_trace_bytes != 0
            or key.warmup_protocol_sha256 != _WARMUP_PROTOCOL_SHA256
            or key.capture_policy_sha256 != _NO_CAPTURE_POLICY_SHA256 for key in keys
        ):
            raise ValueError('live keys do not match the supported model or execution mode')
        if (load.operation != 'model_load' or load.warmup_state != 'cold'
                or load.batch_size != 0 or load.sequence_length != 0
                or load.input_dtypes != () or load.output_limit_bytes != 0):
            raise ValueError('invalid model-load key')
        if (cold.operation != 'forward' or cold.warmup_state != 'cold'
                or cold.batch_size != 1 or cold.sequence_length != 2
                or cold.input_dtypes != (('input_ids', 'torch.int64'),)
                or cold.output_limit_bytes != 512 * 1024):
            raise ValueError('unsupported forward signature')
        if replace(cold, warmup_state='warm') != warm:
            raise ValueError('warm profile must differ only in warmup state')
    except (AttributeError, TypeError) as exc:
        raise ValueError('invalid live workload keys') from exc


def _owner_is_unsafe():
    return _OWNER_UNSAFE or _QUARANTINED.is_set()


def _mark_unsafe():
    global _OWNER_UNSAFE
    # This process-local latch is authoritative; Event notification is only
    # best effort and cannot undo it if notification allocation fails.
    _OWNER_UNSAFE = True
    try:
        _QUARANTINED.set()
    except BaseException:
        pass


def _quarantine(*objects):
    _mark_unsafe()
    try:
        _QUARANTINE_HOLDS.append(objects)
    except BaseException:
        # Even if retaining references fails during OOM, the latch remains set
        # and the supervised owner must exit immediately without attesting.
        pass


def _require_unquarantined(*, allow_pending=False):
    if _owner_is_unsafe() or (_OWNER_DRAIN_PENDING and not allow_pending):
        raise RuntimeError('GPU owner process is quarantined')


@dataclass(frozen=True)
class GpuOwnerResult:
    """Workload status and device-teardown evidence are separate facts."""

    status: str
    teardown_verified: bool
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ProfileFiles:
    """Locally configured evidence files and an independently approved pin."""

    document: Path
    evidence: Path
    source_manifest: Path
    approved_profile_id: str


def _read_bounded_profile_file(path, limit):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('profile files must use absolute paths')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('profile evidence file exceeds its size limit')
    return data


def _load_profile_registry(files, identity):
    """CPU-only validation of saved evidence against separately approved IDs.

    These declared keys are used only to reject invalid configuration before a
    GPU owner starts. They are not live runtime identity: the child must build
    and resolve independently verified live keys again under supervision. This
    generic loader does not establish that evidence used the fixed token case
    and prescribed stage seeds; production must verify those separately.
    """
    from .resource_profile import ProfileRegistry, load_profile

    if type(files) is not tuple or len(files) != 3 or any(type(item) is not ProfileFiles for item in files):
        raise ValueError('exactly three trusted load/cold/warm profile files are required')
    profiles = []
    manifests = []
    for item in files:
        document = _read_bounded_profile_file(item.document, 16 * 1024 * 1024)
        evidence = _read_bounded_profile_file(item.evidence, 16 * 1024 * 1024)
        manifest = _read_bounded_profile_file(item.source_manifest, 64 * 1024 * 1024)
        profile = load_profile(document, evidence=evidence, source_manifest=manifest,
                               expected_profile_id=item.approved_profile_id)
        if profile.key.source_sha256 != profile.provenance.source_manifest_sha256:
            raise ValueError('profile workload source does not match its source manifest')
        profiles.append(profile)
        manifests.append(manifest)
    if not manifests[0] == manifests[1] == manifests[2]:
        raise ValueError('profile source manifests do not match')
    declared_keys = tuple(profile.key for profile in profiles)
    _validate_service_keys(identity, declared_keys)
    registry = ProfileRegistry(tuple(profiles),
        approved_profile_ids=tuple(item.approved_profile_id for item in files))
    return registry, declared_keys


class _TorchServiceBackend:
    """Fixed no-capture, two-token backend for an already supervised child.

    The key builder is trusted owner construction, not a wire-facing input.
    There is intentionally no public launcher until CPU preflight, the reviewed
    live builder and private credential channel have been integrated.
    """

    def __init__(self, *, session, device_lease, cache_dir, team_ids,
                 live_key_builder, verify_loaded_model, shutdown_timeout=10):
        from .ipc import _identifier

        cache_dir = Path(cache_dir)
        if not cache_dir.is_absolute():
            raise ValueError('checkpoint cache path must be absolute')
        if (type(team_ids) is not tuple or not 1 <= len(team_ids) <= 128
                or not all(_identifier(team) for team in team_ids)
                or len(set(team_ids)) != len(team_ids)
                ):
            raise ValueError('team IDs must be a nonempty tuple of distinct names')
        if not callable(live_key_builder):
            raise TypeError('a verified live-key builder is required')
        if not callable(verify_loaded_model):
            raise TypeError('a CPU model verifier is required')
        if type(shutdown_timeout) not in (int, float) or not 0 < shutdown_timeout <= 300:
            raise ValueError('invalid shutdown timeout')
        self.session, self.device_lease = session, device_lease
        self.cache_dir, self.team_ids = cache_dir, team_ids
        self.live_key_builder, self.shutdown_timeout = live_key_builder, shutdown_timeout
        self.verify_loaded_model = verify_loaded_model
        self._resolved_keys = None
        # These slots are allocated while still CPU-only.  Filling a fixed slot
        # avoids allocating an ever-growing failure-time retention list.
        self._retained = [None] * 7

    def runtime_binding(self):
        # This is the sole CUDA query permitted between session.check and
        # verify_runtime_binding: it obtains the actual runtime identity.
        _require_unquarantined(allow_pending=True)
        import torch
        count = torch.cuda.device_count()
        _require_unquarantined(allow_pending=True)
        if type(count) is not int or count != 1:
            raise RuntimeError('the CUDA runtime must expose exactly one device')
        properties = torch.cuda.get_device_properties(0)
        _require_unquarantined(allow_pending=True)
        uuid_type = getattr(torch._C, '_CUuuid', None)
        return _canonical_runtime_uuid(getattr(properties, 'uuid', None), uuid_type), count

    def live_keys(self, identity):
        _require_unquarantined()
        keys = self.live_key_builder(identity)
        _require_unquarantined()
        return keys

    def _check_runtime(self):
        _require_unquarantined()
        self.session.check(self.device_lease)
        _require_unquarantined()
        runtime_uuid, count = self.runtime_binding()
        _require_unquarantined()
        self.session.verify_runtime_binding(self.device_lease, runtime_uuid=runtime_uuid,
                                            device_count=count)
        _require_unquarantined()

    def resolve_profiles(self, registry, keys):
        from .resource_profile import WorkloadKey, ProfileRegistry, resolve_worker_profile

        if type(registry) is not ProfileRegistry or any(type(key) is not WorkloadKey for key in keys):
            raise TypeError('trusted measured profiles and exact live workload keys are required')
        _validate_service_keys(self.session.identity, keys)
        load = registry.resolve(keys[0])
        cold = resolve_worker_profile(registry, keys[1])
        warm = resolve_worker_profile(registry, keys[2])
        self._resolved_keys = keys
        return load, cold, warm

    def telemetry(self):
        _require_unquarantined()
        from .memory_budget import CudaMemorySnapshot, TelemetryError
        import torch
        try:
            available = torch.cuda.is_available()
            _require_unquarantined()
            if not available:
                raise RuntimeError('CUDA is not available')
            free, total = torch.cuda.mem_get_info(0)
            _require_unquarantined()
            allocated = torch.cuda.memory_allocated(0)
            _require_unquarantined()
            reserved = torch.cuda.memory_reserved(0)
            _require_unquarantined()
            return CudaMemorySnapshot(free, total, allocated, reserved)
        except Exception as exc:
            raise TelemetryError(f'cannot read CUDA memory telemetry: {exc}') from exc

    def synchronize(self):
        _require_unquarantined(allow_pending=True)
        import torch
        torch.cuda.synchronize(0)
        _require_unquarantined(allow_pending=True)

    def construct(self, budget, resident_lease, profiles):
        """Load once and complete a separately admitted same-signature warmup."""
        _require_unquarantined()
        import torch
        from .pretrained import load_distilgpt2
        from .resident_worker import ForwardRequest, ResidentWorker
        from .budgeted_worker import BudgetedWorker, DeviceAdmission
        from .broker import Broker
        from .ipc import IPCServer
        from .service import make_gpt2_request_factory

        if resident_lease.released or budget.active_leases() != (resident_lease,):
            raise RuntimeError('a live exclusive resident-load reservation is required')
        if self._resolved_keys is None:
            raise RuntimeError('live profiles must be resolved before model loading')
        load, cold, warm = profiles
        bundle = load_distilgpt2(cache_dir=self.cache_dir, allow_download=False)
        self._retained[0] = bundle
        _require_unquarantined()
        model = bundle.model
        config = getattr(model, 'config', None)
        if (model.training or getattr(config, 'use_cache', None) is not False
                or getattr(config, '_attn_implementation', None) != 'eager'
                or getattr(config, 'vocab_size', None) != 50257
                or getattr(config, 'n_positions', None) != 1024):
            raise RuntimeError('loaded model does not match the profiled model mode')
        # Verify the actual CPU model, including checkpoint tensors and tied
        # state, before copying it. The verifier is installed by trusted owner
        # construction; clients do not choose or supply it.
        if self.verify_loaded_model(model) is not True:
            raise RuntimeError('model verification did not explicitly succeed')
        # CPU staging may be slow. Revalidate supervision immediately before
        # the first weight copy, rather than relying on the earlier admission.
        self._check_runtime()
        if self.live_keys(self.session.identity) != self._resolved_keys:
            raise RuntimeError('live workload identity changed during CPU staging')
        self._check_runtime()
        snapshot = budget.snapshot()
        memory = snapshot.telemetry
        committed = snapshot.committed_incremental_bytes + snapshot.reserve_bytes
        if (snapshot.active_leases != 1
                or snapshot.committed_incremental_bytes != resident_lease.estimated_incremental_bytes
                or budget.active_leases() != (resident_lease,)
                or resident_lease.released):
            raise RuntimeError('resident reservation changed before the model copy')
        if (committed > memory.device_free_bytes
                or memory.process_reserved_bytes + committed > memory.device_total_bytes):
            from .memory_budget import AdmissionError
            raise AdmissionError('live headroom changed before the model copy')
        _require_unquarantined()
        model.to(device='cuda:0', dtype=torch.float32)
        _require_unquarantined()
        self.synchronize()
        self._check_runtime()
        resident = ResidentWorker(load.key.model_id, model, device='cuda:0',
                                  max_batch_size=1, max_sequence_length=2,
                                  max_input_bytes=64 * 1024,
                                  max_output_bytes=512 * 1024)
        self._retained[1] = resident
        admission = DeviceAdmission(budget, device='cuda:0',
            max_estimated_incremental_bytes=max(cold.estimated_incremental_bytes,
                                                warm.estimated_incremental_bytes))
        cold_worker = BudgetedWorker(resident, admission, (cold,))
        self._retained[2] = admission
        self._retained[3] = cold_worker
        warmup = ForwardRequest('service-cold-warmup-v1',
            {'input_ids': torch.tensor([[15496, 995]], dtype=torch.int64)}, seed=17)
        _require_unquarantined()
        warmup_result = cold_worker.execute(warmup)
        _require_unquarantined()
        logits = warmup_result.logits
        if (logits.device.type != 'cpu' or logits.dtype != torch.float32
                or tuple(logits.shape) != (1, 2, 50257)
                or not bool(torch.isfinite(logits).all())):
            raise RuntimeError('cold warmup did not return finite expected logits')
        del logits, warmup_result, warmup
        if cold_worker.quarantined or budget.active_leases() != (resident_lease,):
            raise RuntimeError('cold warmup did not drain safely')
        worker = BudgetedWorker(resident, admission, (warm,))
        self._retained[4] = worker
        broker = Broker(worker, max_pending=8, max_pending_per_team=2)
        self._retained[5] = broker
        factory = make_gpt2_request_factory(vocab_size=config.vocab_size,
            max_positions=config.n_positions, max_batch=1, max_tokens=2)

        def exact_request(request_id, inputs, seed):
            if type(inputs) is not dict or set(inputs) != {'input_ids'}:
                raise ValueError('only the profiled input_ids signature is supported')
            if (type(inputs['input_ids']) is not list or len(inputs['input_ids']) != 1
                    or type(inputs['input_ids'][0]) is not list
                    or len(inputs['input_ids'][0]) != 2
                    or any(type(token) is not int for token in inputs['input_ids'][0])
                    or inputs['input_ids'][0] != [15496, 995]):
                raise ValueError('only the profiled [15496, 995] input is supported')
            if type(seed) is not int or seed not in (29, 43, 71):
                raise ValueError('only the profiled warm seeds are supported')
            return factory(request_id, inputs, seed)

        # IPC preflights JSON at 32 bytes per value plus envelope overhead.
        if 32 * 2 * config.vocab_size + 512 > 4 * 1024 * 1024:
            raise RuntimeError('full logits exceed the bounded IPC response')
        server = IPCServer(broker, team_ids=self.team_ids, request_factory=exact_request,
                           max_connections=8, max_batch=1, max_tokens=2,
                           max_response_bytes=4 * 1024 * 1024)
        self._retained[6] = server
        return worker, broker, server

    def serve(self, server, broker, worker, stop_event, announce, verify_drain):
        from .service import serve_owner
        return serve_owner(server, broker, worker, stop_event, announce=announce,
                           verify_drain=verify_drain, shutdown_timeout=self.shutdown_timeout)


def _run_bound_owner(session, device_lease, *, backend, registry, stop_event,
                     announce, minimum_reserve_bytes=0):
    """Run trusted owner operations under one supervised, retained device lease.

    ``backend`` is a private construction seam, not a client-provided plugin.
    A production launcher must bind it to the fixed reviewed implementation and
    use a real OwnerSession and DeviceLease. Tests use a fake CPU-only backend.
    The resident load reservation remains held through process exit.
    """
    global _OWNER_CLAIMED
    with _OWNER_CUSTODY_LOCK:
        if _owner_is_unsafe() or _OWNER_DRAIN_PENDING:
            raise RuntimeError('GPU owner process is quarantined')
        if _OWNER_CLAIMED:
            raise RuntimeError('GPU owner process has already been used')
        _OWNER_CLAIMED = True
        _OWNER_CUSTODY[0] = session
        _OWNER_CUSTODY[1] = device_lease
        _OWNER_CUSTODY[2] = backend
        _OWNER_CUSTODY[3] = registry
    try:
        return _run_bound_owner_inner(
            session, device_lease, backend=backend, registry=registry,
            stop_event=stop_event, announce=announce,
            minimum_reserve_bytes=minimum_reserve_bytes,
        )
    except BaseException as exc:
        if _OWNER_CUSTODY[9] is None:
            _OWNER_CUSTODY[9] = exc
        elif _OWNER_CUSTODY[9] is not exc and _OWNER_CUSTODY[10] is None:
            _OWNER_CUSTODY[10] = exc
        _mark_unsafe()
        _quarantine(session, device_lease, backend, registry, exc)
        raise


def _run_bound_owner_inner(session, device_lease, *, backend, registry,
                           stop_event, announce, minimum_reserve_bytes):
    session.check(device_lease)
    if session.identity is None:
        raise RuntimeError('GPU ownership requires an identity-aware supervisor')

    def verify_binding(*, allow_pending=False):
        # The runtime is queried only after checking the sticky marker and the
        # actual process-owned lease. This UUID must come from CUDA, not the
        # inventory or a profile's declared UUID.
        _require_unquarantined(allow_pending=allow_pending)
        session.check(device_lease)
        _require_unquarantined(allow_pending=allow_pending)
        runtime_uuid, count = backend.runtime_binding()
        _require_unquarantined(allow_pending=allow_pending)
        session.verify_runtime_binding(device_lease, runtime_uuid=runtime_uuid,
                                       device_count=count)
        _require_unquarantined(allow_pending=allow_pending)

    verify_binding()
    # The backend must reconstruct keys from actual runtime/source state, not
    # by copying declarations out of a supplied resource-profile document.
    keys = backend.live_keys(session.identity)
    _require_unquarantined()
    _validate_service_keys(session.identity, keys)
    profiles = backend.resolve_profiles(registry, keys)
    _require_unquarantined()
    if type(profiles) is not tuple or len(profiles) != 3:
        raise ValueError('load, cold-forward and warm-forward profiles are required')
    load_profile, _, _ = profiles
    estimate = load_profile.estimated_incremental_bytes
    if type(estimate) is not int or estimate <= 0:
        raise ValueError('model-load estimate must be a positive integer')

    def guarded_telemetry():
        verify_binding()
        snapshot = backend.telemetry()
        _require_unquarantined()
        return snapshot

    budget = MemoryBudget(guarded_telemetry, max_gpu_trace_bytes=0,
                          max_host_trace_bytes=0,
                          minimum_reserve_bytes=minimum_reserve_bytes)
    # The budget itself roots every published lease, including when an
    # interruption prevents acquire() from returning its newly issued handle.
    _OWNER_CUSTODY[4] = budget
    resident_lease = budget.acquire(estimate, max_gpu_trace_bytes=0,
                                    max_host_trace_bytes=0)
    _OWNER_CUSTODY[5] = resident_lease
    _require_unquarantined()
    # Construct loads one model, performs a separately admitted cold warmup,
    # then binds one resident worker, broker and authenticated IPC server.
    worker, broker, server = backend.construct(budget, resident_lease, profiles)
    _OWNER_CUSTODY[6] = worker
    _OWNER_CUSTODY[7] = broker
    _OWNER_CUSTODY[8] = server
    _require_unquarantined()

    def verify_drain():
        global _OWNER_DRAIN_PENDING
        try:
            if _owner_is_unsafe() or _OWNER_DRAIN_PENDING:
                _mark_unsafe()
                return False
            # Keep this set until all drain checks finish. If an interruption
            # occurs before failure notification, another drain cannot retry.
            _OWNER_DRAIN_PENDING = True
            if worker.quarantined or broker.quarantined:
                _mark_unsafe()
                return False
            active = budget.active_leases()
            if (len(active) != 1 or active[0] is not resident_lease
                    or resident_lease.released
                    or resident_lease.trace_bytes('gpu')
                    or resident_lease.trace_bytes('host')):
                _mark_unsafe()
                return False
            verify_binding(allow_pending=True)
            if _owner_is_unsafe():
                return False
            backend.synchronize()
            if _owner_is_unsafe():
                return False
            verify_binding(allow_pending=True)
            active = budget.active_leases()
            safe = (not _owner_is_unsafe() and not worker.quarantined
                    and not broker.quarantined and len(active) == 1
                    and active[0] is resident_lease and not resident_lease.released
                    and not resident_lease.trace_bytes('gpu')
                    and not resident_lease.trace_bytes('host'))
            if not safe:
                _mark_unsafe()
            else:
                _OWNER_DRAIN_PENDING = False
            return safe
        except BaseException as exc:
            if _OWNER_CUSTODY[9] is None:
                _OWNER_CUSTODY[9] = exc
            elif _OWNER_CUSTODY[9] is not exc and _OWNER_CUSTODY[10] is None:
                _OWNER_CUSTODY[10] = exc
            _mark_unsafe()
            _quarantine(session, device_lease, backend, registry, budget,
                        resident_lease, worker, broker, server, exc)
            return False

    result = backend.serve(server, broker, worker, stop_event, announce, verify_drain)
    status = getattr(result.status, 'value', result.status)
    if status not in ('clean', 'worker_failed', 'quarantined', 'shutdown_incomplete'):
        raise RuntimeError('unknown owner status')
    if type(result.teardown_verified) is not bool:
        raise RuntimeError('invalid owner teardown result')
    if not result.teardown_verified:
        _quarantine(session, device_lease, backend, registry, budget,
                    resident_lease, worker, broker, server, result)
        if status == 'clean':
            status = 'quarantined'
        return GpuOwnerResult(status, False, getattr(result, 'error_type', None),
                              getattr(result, 'error_message', None))
    if status not in ('clean', 'worker_failed', 'shutdown_incomplete') or not verify_drain():
        _quarantine(session, device_lease, backend, registry, budget,
                    resident_lease, worker, broker, server, result)
        return GpuOwnerResult('quarantined', False)
    # Even an unsuccessfully completed workload may have a verified teardown.
    # The supervisor must still report the workload failure to its caller.
    # Keep the reservation, device lease and owners alive even after a clean
    # result. The direct child must exit promptly; it never releases a resident
    # reservation while model/state references may still exist.
    if _owner_is_unsafe() or _OWNER_DRAIN_PENDING or worker.quarantined or broker.quarantined:
        _quarantine(session, device_lease, backend, registry, budget,
                    resident_lease, worker, broker, server, result)
        return GpuOwnerResult('quarantined', False)
    session.attest_clean(device_lease)
    return GpuOwnerResult(status, True, getattr(result, 'error_type', None),
                          getattr(result, 'error_message', None))
