"""Safety-critical ordering for the supervised GPU service.

This module is not yet a GPU-service launcher. A launcher must independently
verify live profile keys and privately deliver credentials before it can call
the owner in a supervised child. No profile or device is assumed here.
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from threading import Event, Lock, current_thread
from types import TracebackType

from .memory_budget import CudaMemorySnapshot, MemoryBudget


# A controller can fail closed on an older, incompatible owner before launch.
# This marker is a compatibility check, not proof of source or profile identity.
FIXED_SERVICE_OWNER_CONTRACT = 'fixed-two-token-commitments-v1'

# A failed device drain is a process-lifetime event. Retain known owner
# objects and a bounded set of failure tracebacks, and stop admitting work.
# Unrecorded CUDA internals can be lost; the supervised process must exit.
_QUARANTINED = Event()
_QUARANTINE_HOLDS = []
_OWNER_UNSAFE = False
_OWNER_COMPLETION = None
_OWNER_DRAIN_PENDING = False
# Preallocate process-lifetime custody before any owner can touch CUDA.  The
# fixed slots hold session, device lease, backend, registry, budget, resident
# lease, worker, broker, server, two failure tracebacks and a final barrier. The
# first failure is never overwritten by a later failure while quarantining.
# The process serves only one owner and must exit after its result, even on success.
_OWNER_CUSTODY = [None] * 12
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
            or type(key.device_total_bytes) is not int
            or key.device_total_bytes != load.device_total_bytes
            or not 7 * 1024 ** 3 <= key.device_total_bytes <= 9 * 1024 ** 3
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
    return (_OWNER_UNSAFE or _QUARANTINED.is_set()
            or (_OWNER_COMPLETION is not None and _OWNER_COMPLETION.unsafe))


def _mark_unsafe():
    global _OWNER_UNSAFE
    # The completion authority is signaled first, before notification or any
    # other owner state can block. Broker publication reads this same flag.
    try:
        if _OWNER_COMPLETION is not None:
            _OWNER_COMPLETION.mark_unsafe()
    finally:
        # Retry the shared signal and preserve the existing process latch even
        # if the first call is interrupted once.
        _OWNER_UNSAFE = True
        if _OWNER_COMPLETION is not None:
            _OWNER_COMPLETION.mark_unsafe()
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


class _HealthFailureSlot:
    """Preallocated roots for one trusted health-observer thread."""

    def __init__(self):
        self.error = self.traceback = None
        self.cleanup_error = self.cleanup_traceback = None
        # Allocate the public errors while the observer is still healthy.
        # Once an observation fails, constructing an exception may itself run
        # out of memory and expose the original failure as its context.
        message = 'owner health is uncertain; terminate the owner process'
        self.public_keyboard = KeyboardInterrupt(message)
        self.public_system_exit = SystemExit(message)
        self.public_generator_exit = GeneratorExit(message)
        self.public_runtime = RuntimeError(message)
        self.public_base = BaseException(message)

    def public_for(self, exc):
        if isinstance(exc, KeyboardInterrupt):
            return self.public_keyboard
        if isinstance(exc, SystemExit):
            return self.public_system_exit
        if isinstance(exc, GeneratorExit):
            return self.public_generator_exit
        if isinstance(exc, Exception):
            return self.public_runtime
        return self.public_base

    def retain(self, exc):
        traceback = BaseException.__dict__['__traceback__'].__get__(exc, BaseException)
        if type(traceback) is not TracebackType:
            raise RuntimeError('health failure traceback is unavailable')
        if self.error is None or self.error is exc:
            self.error, self.traceback = exc, traceback
        elif self.cleanup_error is None or self.cleanup_error is exc:
            self.cleanup_error, self.cleanup_traceback = exc, traceback
        else:
            raise RuntimeError('health failure custody is exhausted')


class _DirectOwnerHealth:
    """Observe one exact owner graph without taking the broker condition."""

    def __init__(self, server, broker, worker, authority):
        self.server, self.broker, self.worker, self.authority = server, broker, worker, authority
        self._owner_thread = current_thread()
        # serve_owner invokes the provider on its caller and at most one
        # shutdown thread. Each gets separate pre-rooted failure storage.
        self._owner_failures, self._shutdown_failures = _HealthFailureSlot(), _HealthFailureSlot()

    def __call__(self):
        authority = self.authority
        slot = None
        try:
            slot = self._owner_failures if current_thread() is self._owner_thread else self._shutdown_failures
            try:
                try:
                    return self._read()
                except BaseException as exc:
                    try:
                        try:
                            authority._unsafe = True
                        finally:
                            authority._unsafe = True
                    finally:
                        try:
                            slot.retain(exc)
                        finally:
                            slot.retain(exc)
                    raise
            except BaseException as exc:
                try:
                    try:
                        slot.retain(exc)
                    finally:
                        authority._unsafe = True
                    public_error = slot.public_for(exc)
                except BaseException as recovery_error:
                    # An interruption during capture or category selection
                    # still has the original private error as context.
                    try:
                        slot.retain(recovery_error)
                    finally:
                        authority._unsafe = True
                    public_error = slot.public_for(recovery_error)
            # Leave the private exception handler before raising the public
            # category, including when the caller already has an active error.
            try:
                raise public_error from None
            except BaseException:
                public_error.__context__ = None
                public_error.__cause__ = None
                raise
        finally:
            # Clear the private graph from this public traceback frame on both
            # ordinary exit and exceptional exit, including an interruption
            # at the first assignment. The observer remains rooted by backend.
            try:
                try:
                    self = slot = None
                finally:
                    self = slot = None
            except BaseException as cleanup_error:
                try:
                    authority._unsafe = True
                finally:
                    authority._unsafe = True
                cleanup_error.__context__ = None
                cleanup_error.__cause__ = None
                raise
            finally:
                try:
                    try:
                        authority = None
                    finally:
                        authority = None
                except BaseException as final_error:
                    # The global is the exact authority validated by a healthy
                    # observation. A failed graph check has already marked
                    # its pinned authority before reaching this cleanup.
                    try:
                        if _OWNER_COMPLETION is not None:
                            _OWNER_COMPLETION._unsafe = True
                    finally:
                        if _OWNER_COMPLETION is not None:
                            _OWNER_COMPLETION._unsafe = True
                    final_error.__context__ = None
                    final_error.__cause__ = None
                    raise

    def _read(self):
        from .broker import Broker, BrokerHealth
        from .ipc import IPCServer
        from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
        from .service import OwnerHealth

        server, broker, worker, authority = self.server, self.broker, self.worker, self.authority
        if (type(server) is not IPCServer or type(broker) is not Broker
                or type(worker) is not LifetimeWorkspaceWorker
                or type(authority) is not OwnerCompletionAuthority
                or authority is not _OWNER_COMPLETION or authority._broker is not broker
                or authority._worker is not worker or broker._worker is not worker
                or broker._completion_authority is not authority or worker._authority is not authority
                or server.broker is not broker or type(server._completion_binding) is not tuple
                or len(server._completion_binding) != 3
                or server._completion_binding[0] is not authority
                or server._completion_binding[1] is not broker
                or server._completion_binding[2] is not None):
            raise RuntimeError('direct owner health binding changed')
        broker_health = broker.health_snapshot
        if (type(broker_health) is not BrokerHealth
                or any(type(getattr(broker_health, name)) is not bool
                       for name in ('failed', 'quarantined', 'closed', 'owner_alive'))):
            raise RuntimeError('invalid broker health observation')
        unavailable, failed = worker.poisoned, worker.failed
        worker_quarantined, ipc_quarantined = worker.quarantined, server.custody_quarantined
        if any(type(value) is not bool for value in
               (unavailable, failed, worker_quarantined, ipc_quarantined)):
            raise RuntimeError('invalid worker or IPC health observation')
        quarantined = (_owner_is_unsafe() or broker_health.quarantined
                       or broker_health.failed or worker_quarantined or failed
                       or ipc_quarantined)
        if quarantined:
            authority._unsafe = True
        return OwnerHealth(bool(quarantined), unavailable, failed, broker_health.failed,
                           broker_health.closed, broker_health.owner_alive)

    def mark_uncertain(self):
        # Mark the pinned authority before touching the process-level signal.
        self.authority._unsafe = True
        _mark_unsafe()


class _TorchServiceBackend:
    """Fixed no-capture, two-token backend for an already supervised child.

    The key builder is trusted owner construction, not a wire-facing input.
    There is intentionally no public launcher until CPU preflight, the reviewed
    live builder and private credential channel have been integrated.
    """

    def __init__(self, *, session, device_lease, cache_dir, team_ids,
                 live_key_builder, verify_loaded_model, verify_device_model,
                 shutdown_timeout=10):
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
        if not callable(verify_device_model):
            raise TypeError('a device model verifier is required')
        if type(shutdown_timeout) not in (int, float) or not 0 < shutdown_timeout <= 300:
            raise ValueError('invalid shutdown timeout')
        self.session, self.device_lease = session, device_lease
        self._owner_session_binding, self._owner_lease_binding = session, device_lease
        self.cache_dir, self.team_ids = cache_dir, team_ids
        self.live_key_builder, self.shutdown_timeout = live_key_builder, shutdown_timeout
        self.verify_loaded_model = verify_loaded_model
        self.verify_device_model = verify_device_model
        self._resolved_keys = None
        # These slots are allocated while still CPU-only.  Filling a fixed slot
        # avoids allocating an ever-growing failure-time retention list.
        self._retained = [None] * 9

    def runtime_binding(self):
        # This is the sole CUDA query permitted between session.check and
        # verify_runtime_binding: it obtains the actual runtime identity.
        self._check_owner_lease(allow_pending=True)
        import torch
        count = torch.cuda.device_count()
        self._check_owner_lease(allow_pending=True)
        if type(count) is not int or count != 1:
            raise RuntimeError('the CUDA runtime must expose exactly one device')
        properties = torch.cuda.get_device_properties(0)
        self._check_owner_lease(allow_pending=True)
        uuid_type = getattr(torch._C, '_CUuuid', None)
        return _canonical_runtime_uuid(getattr(properties, 'uuid', None), uuid_type), count

    def _check_owner_lease(self, *, allow_pending=False):
        if (self.session is not self._owner_session_binding
                or self.device_lease is not self._owner_lease_binding):
            raise RuntimeError('backend owner binding changed')
        _require_unquarantined(allow_pending=allow_pending)
        self._owner_session_binding.check(self._owner_lease_binding)
        _require_unquarantined(allow_pending=allow_pending)
        if (self.session is not self._owner_session_binding
                or self.device_lease is not self._owner_lease_binding):
            raise RuntimeError('backend owner binding changed')

    def live_keys(self, identity):
        _require_unquarantined()
        keys = self.live_key_builder(identity)
        _require_unquarantined()
        return keys

    def _check_runtime(self):
        self._check_owner_lease()
        runtime_uuid, count = self.runtime_binding()
        self._check_owner_lease()
        self._owner_session_binding.verify_runtime_binding(
            self._owner_lease_binding, runtime_uuid=runtime_uuid, device_count=count)
        self._check_owner_lease()

    def resolve_profiles(self, registry, keys):
        from .resource_profile import WorkloadKey, ProfileRegistry, resolve_worker_profile
        from .trusted_service_profile import LOAD_COMMITMENT, JOB_COMMITMENT

        if type(registry) is not ProfileRegistry or any(type(key) is not WorkloadKey for key in keys):
            raise TypeError('trusted measured profiles and exact live workload keys are required')
        _validate_service_keys(self.session.identity, keys)
        load = registry.resolve(keys[0])
        cold = resolve_worker_profile(registry, keys[1])
        warm = resolve_worker_profile(registry, keys[2])
        for profile, limit in ((load, LOAD_COMMITMENT), (cold, JOB_COMMITMENT),
                               (warm, JOB_COMMITMENT)):
            estimate = profile.estimated_incremental_bytes
            if type(estimate) is not int or not 0 < estimate <= limit:
                raise ValueError('measured estimate exceeds its fixed service commitment')
        # A profile's measured estimate establishes eligibility; the worker
        # reserves the larger fixed commitment prescribed by this protocol.
        cold = replace(cold, estimated_incremental_bytes=JOB_COMMITMENT)
        warm = replace(warm, estimated_incremental_bytes=JOB_COMMITMENT)
        self._resolved_keys = keys
        return load, cold, warm

    def telemetry(self):
        self._check_owner_lease()
        from .memory_budget import CudaMemorySnapshot, TelemetryError
        import torch
        try:
            available = torch.cuda.is_available()
            self._check_owner_lease()
            if not available:
                raise RuntimeError('CUDA is not available')
            free, total = torch.cuda.mem_get_info(0)
            self._check_owner_lease()
            allocated = torch.cuda.memory_allocated(0)
            self._check_owner_lease()
            reserved = torch.cuda.memory_reserved(0)
            self._check_owner_lease()
            return CudaMemorySnapshot(free, total, allocated, reserved)
        except Exception as exc:
            raise TelemetryError(f'cannot read CUDA memory telemetry: {exc}') from exc

    def synchronize(self):
        self._check_owner_lease(allow_pending=True)
        import torch
        torch.cuda.synchronize(0)
        self._check_owner_lease(allow_pending=True)

    @staticmethod
    def _require_conversion_policy():
        import torch
        if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
                or torch.__future__.get_swap_module_params_on_conversion() is not False):
            raise RuntimeError('model conversion flags must both be explicitly false')

    def construct(self, budget, resident_lease, profiles):
        """Load once and complete a separately admitted same-signature warmup."""
        _require_unquarantined()
        import torch
        from .pretrained import load_distilgpt2
        from .resident_worker import ForwardRequest, ResidentWorker
        from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
        from .broker import Broker
        from .ipc import IPCServer
        from .fixed_service_request import FixedServiceRequestFactory
        from .service import make_gpt2_request_factory
        from .trusted_service_profile import LOAD_COMMITMENT, JOB_COMMITMENT

        if (resident_lease.released or budget.active_leases() != (resident_lease,)
                or resident_lease.estimated_incremental_bytes != LOAD_COMMITMENT):
            raise RuntimeError('a live exclusive resident-load reservation is required')
        if self._resolved_keys is None:
            raise RuntimeError('live profiles must be resolved before model loading')
        if type(_OWNER_COMPLETION) is not OwnerCompletionAuthority:
            raise RuntimeError('owner completion authority is required')
        load, cold, warm = profiles
        if any(getattr(profile, 'estimated_incremental_bytes', None) != JOB_COMMITMENT
               for profile in (cold, warm)):
            raise RuntimeError('cold and warm jobs require the fixed forward commitment')
        # This is the CPU generator, matching the load-stage protocol without
        # the global torch.manual_seed helper's CUDA side effects.
        torch.random.default_generator.manual_seed(17)
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
        self._require_conversion_policy()
        self._check_owner_lease()
        converted = model.to(device='cuda:0', dtype=torch.float32)
        if converted is not model:
            # A nonstandard conversion may have produced another device object.
            # Keep it reachable while the owner exits under quarantine.
            self._retained[6] = converted
            raise RuntimeError('model conversion did not preserve the model object')
        _require_unquarantined()
        self.synchronize()
        self._check_runtime()
        self._require_conversion_policy()
        _require_unquarantined()
        # Tied weights can split during module conversion even when the CPU
        # model was correct. Verify the actual post-copy model and owner under
        # the same live lease, before creating a worker or running a warmup.
        if self.verify_device_model(model, self.session.identity, self.session,
                                    self.device_lease) is not True:
            raise RuntimeError('device model verification did not explicitly succeed')
        _require_unquarantined()
        self._check_runtime()
        self.synchronize()
        self._check_runtime()
        self._require_conversion_policy()
        _require_unquarantined()
        resident = ResidentWorker(load.key.model_id, model, device='cuda:0',
                                  max_batch_size=1, max_sequence_length=2,
                                  max_input_bytes=64 * 1024,
                                  max_output_bytes=512 * 1024)
        self._retained[1] = resident
        # The actual workspace capability is acquired before cold execution
        # and is never released inside the child, even after clean shutdown.
        workspace_lease = budget.acquire(JOB_COMMITMENT, max_gpu_trace_bytes=0,
                                         max_host_trace_bytes=0)
        self._retained[7] = workspace_lease
        self._retained[8] = _OWNER_COMPLETION
        worker = LifetimeWorkspaceWorker(resident, budget, resident_lease, workspace_lease,
                                         profiles=(cold, warm), authority=_OWNER_COMPLETION)
        self._retained[3] = worker
        warmup = ForwardRequest('service-cold-warmup-v1',
            {'input_ids': torch.tensor([[15496, 995]], dtype=torch.int64)}, seed=17)
        _require_unquarantined()
        warmup_result = worker.execute(warmup)
        _require_unquarantined()
        logits = warmup_result.logits
        if (logits.device.type != 'cpu' or logits.dtype != torch.float32
                or tuple(logits.shape) != (1, 2, 50257)
                or not bool(torch.isfinite(logits).all())):
            raise RuntimeError('cold warmup did not return finite expected logits')
        worker._finish_cold(warmup_result)
        del logits, warmup_result, warmup
        if worker.quarantined or set(budget.active_leases()) != {resident_lease, workspace_lease}:
            raise RuntimeError('cold warmup did not drain safely')
        self._retained[4] = worker
        broker = Broker(worker, max_pending=2, max_pending_per_team=1,
                        completion_authority=_OWNER_COMPLETION)
        self._retained[5] = broker
        factory = FixedServiceRequestFactory(make_gpt2_request_factory(
            vocab_size=config.vocab_size, max_positions=config.n_positions,
            max_batch=1, max_tokens=2))

        # IPC preflights JSON at 32 bytes per value plus envelope overhead.
        if 32 * 2 * config.vocab_size + 512 > 4 * 1024 * 1024:
            raise RuntimeError('full logits exceed the bounded IPC response')
        server = IPCServer(broker, team_ids=self.team_ids, request_factory=factory,
                           completion_authority=_OWNER_COMPLETION,
                           max_connections=2, max_request_bytes=4096, max_batch=1, max_tokens=2,
                           max_response_bytes=4 * 1024 * 1024)
        self._retained[6] = server
        return worker, broker, server

    def serve(self, server, broker, worker, stop_event, announce, verify_drain):
        from .service import serve_owner
        health = _DirectOwnerHealth(server, broker, worker, _OWNER_COMPLETION)
        # Root provider failures and their tracebacks for the owner lifetime.
        self._retained[2] = health
        return serve_owner(server, broker, worker, stop_event, announce=announce,
                           verify_drain=verify_drain, shutdown_timeout=self.shutdown_timeout,
                           health_provider=health, on_health_uncertain=health.mark_uncertain)


def _run_bound_owner(session, device_lease, *, backend, registry, stop_event,
                     announce, before_attest, minimum_reserve_bytes=0):
    """Run trusted owner operations under one supervised, retained device lease.

    ``backend`` is a private construction seam, not a client-provided plugin.
    A production launcher must bind it to the fixed reviewed implementation and
    use a real OwnerSession and DeviceLease. Tests use a fake CPU-only backend.
    The resident load reservation remains held through process exit.
    ``before_attest`` is trusted owner code that must stop and join its own
    remaining signalers and return exactly True before clean attestation.
    """
    global _OWNER_CLAIMED, _OWNER_COMPLETION
    if not callable(before_attest):
        raise TypeError('a pre-attestation barrier is required')
    if isinstance(backend, _TorchServiceBackend) and (
            backend.session is not session or backend.device_lease is not device_lease
            or backend._owner_session_binding is not session
            or backend._owner_lease_binding is not device_lease):
        raise ValueError('backend must use the owner session and device lease')
    with _OWNER_CUSTODY_LOCK:
        if _owner_is_unsafe() or _OWNER_DRAIN_PENDING:
            raise RuntimeError('GPU owner process is quarantined')
        if _OWNER_CLAIMED:
            raise RuntimeError('GPU owner process has already been used')
        from .owner_completion import OwnerCompletionAuthority
        # The real child supplies a trusted subclass to check profile identity.
        # Pin the base lease-check implementation to that exact backend rather
        # than silently omitting the guard for the subclass.
        guard = (_TorchServiceBackend._check_owner_lease.__get__(backend)
                 if isinstance(backend, _TorchServiceBackend) else None)
        _OWNER_COMPLETION = OwnerCompletionAuthority(device_guard=guard)
        _OWNER_CLAIMED = True
        _OWNER_CUSTODY[0] = session
        _OWNER_CUSTODY[1] = device_lease
        _OWNER_CUSTODY[2] = backend
        _OWNER_CUSTODY[3] = registry
        _OWNER_CUSTODY[11] = before_attest
    try:
        return _run_bound_owner_inner(
            session, device_lease, backend=backend, registry=registry,
            stop_event=stop_event, announce=announce,
            before_attest=before_attest, minimum_reserve_bytes=minimum_reserve_bytes,
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
                           stop_event, announce, before_attest, minimum_reserve_bytes):
    from .trusted_service_profile import LOAD_COMMITMENT
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
    if type(estimate) is not int or not 0 < estimate <= LOAD_COMMITMENT:
        raise ValueError('model-load estimate exceeds its fixed service commitment')

    def guarded_telemetry():
        verify_binding()
        snapshot = backend.telemetry()
        _require_unquarantined()
        if (type(snapshot) is not CudaMemorySnapshot
                or any(snapshot.device_total_bytes != key.device_total_bytes for key in keys)):
            raise RuntimeError('live device total differs from the resolved workload keys')
        return snapshot

    budget = MemoryBudget(guarded_telemetry, max_gpu_trace_bytes=0,
                          max_host_trace_bytes=0,
                          minimum_reserve_bytes=minimum_reserve_bytes)
    # The budget itself roots every published lease, including when an
    # interruption prevents acquire() from returning its newly issued handle.
    _OWNER_CUSTODY[4] = budget
    resident_lease = budget.acquire(LOAD_COMMITMENT, max_gpu_trace_bytes=0,
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
            from .owner_completion import LifetimeWorkspaceWorker
            from .trusted_service_profile import JOB_COMMITMENT
            if _owner_is_unsafe() or _OWNER_DRAIN_PENDING:
                _mark_unsafe()
                return False
            # Keep this set until all drain checks finish. If an interruption
            # occurs before failure notification, another drain cannot retry.
            _OWNER_DRAIN_PENDING = True
            if worker.quarantined or broker.quarantined:
                _mark_unsafe()
                return False
            expected = (resident_lease, worker._workspace_lease) if type(worker) is LifetimeWorkspaceWorker else (resident_lease,)
            expected_bytes = (LOAD_COMMITMENT, JOB_COMMITMENT) if len(expected) == 2 else (LOAD_COMMITMENT,)
            if type(worker) is LifetimeWorkspaceWorker and (
                    worker._resident_lease is not resident_lease
                    or worker._resident_bytes != LOAD_COMMITMENT
                    or worker._workspace_bytes != JOB_COMMITMENT):
                _mark_unsafe()
                return False
            active = budget.active_leases()
            if (len(active) != len(expected) or set(active) != set(expected)
                    or any(lease.released or lease.estimated_incremental_bytes != expected_bytes[index]
                           or lease.trace_bytes('gpu') or lease.trace_bytes('host')
                           or lease._trace_limits != {'gpu': 0, 'host': 0}
                           for index, lease in enumerate(expected))):
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
                    and not broker.quarantined and len(active) == len(expected)
                    and set(active) == set(expected)
                    and all(not lease.released and lease.estimated_incremental_bytes == expected_bytes[index]
                            and not lease.trace_bytes('gpu') and not lease.trace_bytes('host')
                            and lease._trace_limits == {'gpu': 0, 'host': 0}
                            for index, lease in enumerate(expected)))
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
    if status not in ('clean', 'worker_failed', 'shutdown_incomplete'):
        _quarantine(session, device_lease, backend, registry, budget,
                    resident_lease, worker, broker, server, result)
        return GpuOwnerResult('quarantined', False)
    # The child owns signalers outside serve_owner, such as the stop monitor.
    # Keep this exact callback rooted even if it cannot establish quiescence.
    if before_attest() is not True:
        _quarantine(session, device_lease, backend, registry, budget,
                    resident_lease, worker, broker, server, result, before_attest)
        return GpuOwnerResult('quarantined', False)
    # Join the child's remaining signalers before the final independent device
    # drain. The barrier alone does not establish other components' quiescence.
    if not verify_drain():
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
