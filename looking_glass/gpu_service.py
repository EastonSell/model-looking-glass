"""Safety-critical ordering for the supervised GPU service.

This module is not yet a GPU-service launcher. A launcher must independently
verify live profile keys and privately deliver credentials before it can call
the owner in a supervised child. No profile or device is assumed here.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import re
from threading import Event

from .memory_budget import MemoryBudget


# A failed device drain is a process-lifetime event. Retain every reachable
# owner object (including exception tracebacks) and stop admitting work; only
# terminating the supervised process can dispose of uncertain CUDA state.
_QUARANTINED = Event()
_QUARANTINE_HOLDS = []
_LIFETIME_HOLDS = []
_RAW_RUNTIME_UUID = re.compile(r'[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z', re.ASCII)


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
            or key.max_host_trace_bytes != 0 for key in keys
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


def _quarantine(*objects):
    _QUARANTINED.set()
    try:
        _QUARANTINE_HOLDS.append(objects)
    except BaseException:
        # Even if retaining references fails during OOM, the latch remains set
        # and the supervised owner must exit immediately without attesting.
        pass


@dataclass(frozen=True)
class GpuOwnerResult:
    """Workload status and device-teardown evidence are separate facts."""

    status: str
    teardown_verified: bool
    error_type: str | None = None
    error_message: str | None = None


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
        self._retained = []

    def runtime_binding(self):
        # This is the sole CUDA query permitted between session.check and
        # verify_runtime_binding: it obtains the actual runtime identity.
        import torch
        count = torch.cuda.device_count()
        if type(count) is not int or count != 1:
            raise RuntimeError('the CUDA runtime must expose exactly one device')
        properties = torch.cuda.get_device_properties(0)
        uuid_type = getattr(torch._C, '_CUuuid', None)
        return _canonical_runtime_uuid(getattr(properties, 'uuid', None), uuid_type), count

    def live_keys(self, identity):
        return self.live_key_builder(identity)

    def _check_runtime(self):
        self.session.check(self.device_lease)
        runtime_uuid, count = self.runtime_binding()
        self.session.verify_runtime_binding(self.device_lease, runtime_uuid=runtime_uuid,
                                            device_count=count)

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
        from .memory_budget import cuda_memory_snapshot
        return cuda_memory_snapshot(0)

    def synchronize(self):
        import torch
        torch.cuda.synchronize(0)

    def construct(self, budget, resident_lease, profiles):
        """Load once and complete a separately admitted same-signature warmup."""
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
        self._retained.append(bundle)
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
        if (committed > memory.device_free_bytes
                or memory.process_reserved_bytes + committed > memory.device_total_bytes):
            from .memory_budget import AdmissionError
            raise AdmissionError('live headroom changed before the model copy')
        model.to(device='cuda:0', dtype=torch.float32)
        self.synchronize()
        self._check_runtime()
        resident = ResidentWorker(load.key.model_id, model, device='cuda:0',
                                  max_batch_size=1, max_sequence_length=2,
                                  max_input_bytes=64 * 1024,
                                  max_output_bytes=512 * 1024)
        self._retained.append(resident)
        admission = DeviceAdmission(budget, device='cuda:0',
            max_estimated_incremental_bytes=max(cold.estimated_incremental_bytes,
                                                warm.estimated_incremental_bytes))
        cold_worker = BudgetedWorker(resident, admission, (cold,))
        self._retained.extend((admission, cold_worker))
        warmup = ForwardRequest('service-cold-warmup-v1',
            {'input_ids': torch.tensor([[15496, 995]], dtype=torch.int64)}, seed=17)
        warmup_result = cold_worker.execute(warmup)
        logits = warmup_result.logits
        if (logits.device.type != 'cpu' or logits.dtype != torch.float32
                or tuple(logits.shape) != (1, 2, 50257)
                or not bool(torch.isfinite(logits).all())):
            raise RuntimeError('cold warmup did not return finite expected logits')
        del logits, warmup_result, warmup
        if cold_worker.quarantined or budget.active_leases() != (resident_lease,):
            raise RuntimeError('cold warmup did not drain safely')
        worker = BudgetedWorker(resident, admission, (warm,))
        self._retained.append(worker)
        broker = Broker(worker, max_pending=8, max_pending_per_team=2)
        self._retained.append(broker)
        factory = make_gpt2_request_factory(vocab_size=config.vocab_size,
            max_positions=config.n_positions, max_batch=1, max_tokens=2)

        def exact_request(request_id, inputs, seed):
            if type(inputs) is not dict or set(inputs) != {'input_ids'}:
                raise ValueError('only the profiled input_ids signature is supported')
            if (type(inputs['input_ids']) is not list or len(inputs['input_ids']) != 1
                    or type(inputs['input_ids'][0]) is not list
                    or len(inputs['input_ids'][0]) != 2):
                raise ValueError('only the profiled [1,2] shape is supported')
            return factory(request_id, inputs, seed)

        # IPC preflights JSON at 32 bytes per value plus envelope overhead.
        if 32 * 2 * config.vocab_size + 512 > 4 * 1024 * 1024:
            raise RuntimeError('full logits exceed the bounded IPC response')
        server = IPCServer(broker, team_ids=self.team_ids, request_factory=exact_request,
                           max_connections=8, max_batch=1, max_tokens=2,
                           max_response_bytes=4 * 1024 * 1024)
        self._retained.append(server)
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
    if _QUARANTINED.is_set():
        raise RuntimeError('GPU owner process is quarantined')
    try:
        return _run_bound_owner_inner(
            session, device_lease, backend=backend, registry=registry,
            stop_event=stop_event, announce=announce,
            minimum_reserve_bytes=minimum_reserve_bytes,
        )
    except BaseException as exc:
        _quarantine(session, device_lease, backend, registry, exc)
        raise


def _run_bound_owner_inner(session, device_lease, *, backend, registry,
                           stop_event, announce, minimum_reserve_bytes):
    session.check(device_lease)
    if session.identity is None:
        raise RuntimeError('GPU ownership requires an identity-aware supervisor')

    def verify_binding():
        # The runtime is queried only after checking the sticky marker and the
        # actual process-owned lease. This UUID must come from CUDA, not the
        # inventory or a profile's declared UUID.
        session.check(device_lease)
        runtime_uuid, count = backend.runtime_binding()
        session.verify_runtime_binding(device_lease, runtime_uuid=runtime_uuid,
                                       device_count=count)

    verify_binding()
    # The backend must reconstruct keys from actual runtime/source state, not
    # by copying declarations out of a supplied resource-profile document.
    keys = backend.live_keys(session.identity)
    _validate_service_keys(session.identity, keys)
    profiles = backend.resolve_profiles(registry, keys)
    if type(profiles) is not tuple or len(profiles) != 3:
        raise ValueError('load, cold-forward and warm-forward profiles are required')
    load_profile, _, _ = profiles
    estimate = load_profile.estimated_incremental_bytes
    if type(estimate) is not int or estimate <= 0:
        raise ValueError('model-load estimate must be a positive integer')

    def guarded_telemetry():
        verify_binding()
        return backend.telemetry()

    budget = MemoryBudget(guarded_telemetry, max_gpu_trace_bytes=0,
                          max_host_trace_bytes=0,
                          minimum_reserve_bytes=minimum_reserve_bytes)
    resident_lease = budget.acquire(estimate, max_gpu_trace_bytes=0,
                                    max_host_trace_bytes=0)
    # Construct loads one model, performs a separately admitted cold warmup,
    # then binds one resident worker, broker and authenticated IPC server.
    worker, broker, server = backend.construct(budget, resident_lease, profiles)

    def verify_drain():
        try:
            if worker.quarantined:
                return False
            active = budget.active_leases()
            if (len(active) != 1 or active[0] is not resident_lease
                    or resident_lease.released
                    or resident_lease.trace_bytes('gpu')
                    or resident_lease.trace_bytes('host')):
                return False
            verify_binding()
            backend.synchronize()
            verify_binding()
            active = budget.active_leases()
            return (not worker.quarantined and len(active) == 1
                    and active[0] is resident_lease and not resident_lease.released
                    and not resident_lease.trace_bytes('gpu')
                    and not resident_lease.trace_bytes('host'))
        except BaseException as exc:
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
    _LIFETIME_HOLDS.append((session, device_lease, backend, registry, budget,
                            resident_lease, worker, broker, server))
    session.attest_clean(device_lease)
    return GpuOwnerResult(status, True, getattr(result, 'error_type', None),
                          getattr(result, 'error_message', None))
