"""CPU-only checks for the supervised GPU-owner composition.

The fake runtime reports a fictitious GPU and never touches CUDA.  A test
claiming actual runtime/device verification would need a separately approved
hardware window and measured profiles.
"""

import threading
import unittest
import gc
import ctypes
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import weakref
from dataclasses import asdict, dataclass, replace
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass.memory_budget import AdmissionError, CudaMemorySnapshot
from looking_glass import gpu_service
from looking_glass.gpu_service import (_canonical_runtime_uuid, _run_bound_owner,
                                       _TorchServiceBackend, _validate_service_keys)


GIB = 1024 ** 3
UUID = 'GPU-00000000-0000-0000-0000-000000000001'

_CUDA_PATCHES = []
_ORIGINAL_CUDA_VISIBLE_DEVICES = None


def _block_cuda_calls(*args, **kwargs):
    raise AssertionError('CPU-only GPU-service tests must never call CUDA')


def _install_cuda_guard(*, hide_device):
    """Fail immediately on high-level CUDA or low-level torch CUDA calls."""
    if hide_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    for name in dir(torch.cuda):
        value = getattr(torch.cuda, name)
        if (callable(value) and not (isinstance(value, type)
                                    and issubclass(value, BaseException))):
            item = patch.object(torch.cuda, name, _block_cuda_calls)
            item.start()
            _CUDA_PATCHES.append(item)
    for name in dir(torch._C):
        value = getattr(torch._C, name)
        if name.startswith('_cuda_') and callable(value) and not isinstance(value, type):
            item = patch.object(torch._C, name, _block_cuda_calls)
            item.start()
            _CUDA_PATCHES.append(item)
    original_to = torch.Tensor.to
    def cpu_only_to(tensor, *args, **kwargs):
        destination = kwargs.get('device')
        if destination is None and args:
            destination = args[0]
        if (isinstance(destination, (str, torch.device))
                and str(destination).split(':', 1)[0] == 'cuda'):
            return _block_cuda_calls()
        return original_to(tensor, *args, **kwargs)
    item = patch.object(torch.Tensor, 'to', cpu_only_to)
    item.start()
    _CUDA_PATCHES.append(item)
    for loader_name in ('CDLL', 'WinDLL', 'PyDLL', 'OleDLL'):
        original_loader = getattr(ctypes, loader_name, None)
        if original_loader is None:
            continue
        def cpu_only_load(name, *args, _original=original_loader, **kwargs):
            candidate = str(name).lower()
            if any(piece in candidate for piece in ('nvcuda', 'nvml', 'libcuda', 'nvidia-ml')):
                return _block_cuda_calls()
            return _original(name, *args, **kwargs)
        item = patch.object(ctypes, loader_name, cpu_only_load)
        item.start()
        _CUDA_PATCHES.append(item)
    original_popen = subprocess.Popen
    def cpu_only_popen(args, *pargs, **kwargs):
        executable = args[0] if isinstance(args, (list, tuple)) and args else args
        if isinstance(executable, (str, bytes, os.PathLike)) and 'nvidia-smi' in os.fsdecode(executable).lower():
            return _block_cuda_calls()
        return original_popen(args, *pargs, **kwargs)
    item = patch.object(subprocess, 'Popen', cpu_only_popen)
    item.start()
    _CUDA_PATCHES.append(item)


def setUpModule():
    global _ORIGINAL_CUDA_VISIBLE_DEVICES
    _ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    _install_cuda_guard(hide_device=True)


def tearDownModule():
    for item in reversed(_CUDA_PATCHES):
        item.stop()
    _CUDA_PATCHES.clear()
    if _ORIGINAL_CUDA_VISIBLE_DEVICES is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = _ORIGINAL_CUDA_VISIBLE_DEVICES


class FakeSession:
    def __init__(self, events, *, reject_check=False, reject_binding=False):
        self.events = events
        self.identity = SimpleNamespace(gpu_uuid=UUID)
        self.reject_check = reject_check
        self.reject_binding = reject_binding

    def check(self, lease):
        self.events.append('session-check')
        if self.reject_check:
            raise RuntimeError('no supervised lease')

    def verify_runtime_binding(self, lease, *, runtime_uuid, device_count):
        self.events.append('verify-binding')
        if self.reject_binding or runtime_uuid != UUID or device_count != 1:
            raise RuntimeError('identity mismatch')

    def attest_clean(self, lease):
        self.events.append('attest')


@dataclass(frozen=True)
class FakeProfile:
    estimated_incremental_bytes: int


@dataclass
class FakeResult:
    status: str = 'clean'
    teardown_verified: bool = True


class FakeWorker:
    quarantined = False


class UuidRendering:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class RuntimeUuidTests(unittest.TestCase):
    def test_cpu_guard_blocks_runtime_driver_and_tensor_copy(self):
        import torch
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '')
        for fn in (torch.cuda.device_count, torch.cuda.get_device_properties,
                   torch._C._cuda_init, lambda: torch.tensor([1]).to('cuda:0'),
                   lambda: ctypes.CDLL('nvcuda.dll'),
                   lambda: subprocess.Popen(['nvidia-smi'])):
            with self.subTest(fn=fn), self.assertRaisesRegex(AssertionError, 'never call CUDA'):
                fn()

    def test_requires_exact_runtime_uuid_object_and_full_lowercase_rendering(self):
        raw = 'abcdefab-1234-5678-9abc-abcdef123456'
        self.assertEqual(_canonical_runtime_uuid(UuidRendering(raw), UuidRendering), 'GPU-' + raw)
        for invalid in ('GPU-' + raw, raw.upper(), raw[:-1], raw + '\n'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _canonical_runtime_uuid(UuidRendering(invalid), UuidRendering)
        with self.assertRaises(TypeError):
            _canonical_runtime_uuid(raw, UuidRendering)


@dataclass(frozen=True)
class FakeKey:
    operation: str
    warmup_state: str
    model_id: str = 'a' * 64
    device: str = 'cuda:0'
    device_uuid: str = UUID
    checkpoint_sha256: str = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
    checkpoint_revision: str = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
    model_config_sha256: str = '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a'
    model_dtype: str = 'torch.float32'
    compute_dtype: str = 'torch.float32'
    attention_backend: str = 'eager'
    model_mode: str = 'eval'
    grad_mode: str = 'inference'
    autocast: str = 'disabled'
    use_cache: bool = False
    compiled: bool = False
    capture_mode: str = 'none'
    max_gpu_trace_bytes: int = 0
    max_host_trace_bytes: int = 0
    batch_size: int = 1
    sequence_length: int = 2
    input_dtypes: tuple = (('input_ids', 'torch.int64'),)
    output_limit_bytes: int = 512 * 1024


def service_keys():
    load = FakeKey('model_load', 'cold', batch_size=0, sequence_length=0,
                   input_dtypes=(), output_limit_bytes=0)
    cold = FakeKey('forward', 'cold')
    return load, cold, replace(cold, warmup_state='warm')


class KeyContractTests(unittest.TestCase):
    def test_only_the_exact_pinned_signature_and_state_transition_are_supported(self):
        identity = SimpleNamespace(gpu_uuid=UUID)
        load, cold, warm = service_keys()
        _validate_service_keys(identity, (load, cold, warm))
        invalid = [
            (load, cold, cold),
            (load, replace(cold, sequence_length=8), replace(warm, sequence_length=8)),
            (load, cold, replace(warm, input_dtypes=(('attention_mask', 'torch.int64'),))),
            (replace(load, model_id='b' * 64), cold, warm),
            (load, replace(cold, attention_backend='sdpa'), replace(warm, attention_backend='sdpa')),
            (load, replace(cold, use_cache=True), replace(warm, use_cache=True)),
            (load, replace(cold, capture_mode='direct'), replace(warm, capture_mode='direct')),
            (load, replace(cold, output_limit_bytes=2 * 1024 * 1024),
             replace(warm, output_limit_bytes=2 * 1024 * 1024)),
        ]
        for keys in invalid:
            with self.subTest(keys=keys):
                with self.assertRaises(ValueError):
                    _validate_service_keys(identity, keys)


def measured_keys_and_registry():
    from looking_glass.resource_profile import (WorkloadKey, MemorySample,
        ProfileProvenance, MeasuredResourceProfile, ProfileRegistry,
        encode_evidence, model_runtime_fingerprint)
    keys = []
    for template in service_keys():
        fields = asdict(template)
        fields.update(source_sha256='1' * 64, runtime_settings_sha256='2' * 64,
                      torch_version='2.8.0+cu128', transformers_version='4.57.3',
                      cuda_runtime_version='12.8', driver_version='570.0',
                      python_version='3.11.15', device_name='fake CPU-test device',
                      compute_capability=(7, 5), device_total_bytes=8 * GIB,
                      warmup_protocol_sha256='3' * 64, capture_policy_sha256='4' * 64)
        fields['model_id'] = model_runtime_fingerprint(fields)
        keys.append(WorkloadKey(**fields))
    records = []
    for index, key in enumerate(keys):
        before = CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0)
        after = CudaMemorySnapshot(7 * GIB - 1024, 8 * GIB, 1024, 2048)
        sample = MemorySample(key.fingerprint(), f'fake-{index}', '5' * 64, 17,
                              'completed', before, after, 1024, 2048)
        evidence = encode_evidence(key, (sample,))
        provenance = ProfileProvenance('resource-profile-v1', '6' * 64,
            hashlib.sha256(evidence).hexdigest(), '2026-10-28T00:00:00Z',
            'cooperative-exclusive', 'synchronize-reset-synchronize')
        records.append(MeasuredResourceProfile.build(key, (sample,), provenance,
                                                       margin_bytes=1024, margin_bps=0))
    registry = ProfileRegistry(tuple(records), approved_profile_ids=tuple(p.profile_id for p in records))
    return tuple(keys), registry


class ConcreteBackendContractTests(unittest.TestCase):
    def test_invalid_team_configuration_is_rejected_before_any_gpu_work(self):
        for teams in ((), ('bad name',), ('alpha', 'alpha'), ('x' * 65,),
                      tuple(f'team{i}' for i in range(129)), (['unhashable'],)):
            with self.subTest(teams=teams), self.assertRaises(ValueError):
                _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
                    cache_dir=Path.cwd(), team_ids=teams,
                    live_key_builder=lambda identity: (), verify_loaded_model=lambda model: True)

    def test_actual_cpu_model_verifier_is_required(self):
        with self.assertRaisesRegex(TypeError, 'model verifier'):
            _TorchServiceBackend(session=FakeSession([]), device_lease=object(),
                cache_dir=Path.cwd(), team_ids=('alpha',),
                live_key_builder=lambda identity: (), verify_loaded_model=None)

    def test_load_cold_and_warm_profiles_must_all_match_live_keys_exactly(self):
        from looking_glass.resource_profile import ProfileRegistry, ProfileUnavailable
        keys, registry = measured_keys_and_registry()
        session = FakeSession([])
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha', 'beta'),
            live_key_builder=lambda identity: keys, verify_loaded_model=lambda model: True)
        self.assertEqual(backend.live_keys(session.identity), keys)
        load, cold, warm = backend.resolve_profiles(registry, keys)
        self.assertEqual(load.key, keys[0])
        self.assertEqual(cold.model_id, keys[1].model_id)
        self.assertEqual(warm.model_id, keys[2].model_id)
        self.assertEqual(cold.shape, (1, 2))
        self.assertEqual(warm.shape, (1, 2))
        missing = ProfileRegistry((load,), approved_profile_ids=(load.profile_id,))
        with self.assertRaises(ProfileUnavailable):
            backend.resolve_profiles(missing, keys)

    def test_owner_rechecks_supervision_after_cpu_loading_before_cuda_copy(self):
        from looking_glass.memory_budget import MemoryBudget
        events = []
        session = FakeSession(events)
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True)
        backend._resolved_keys = ()
        backend.runtime_binding = lambda: (UUID, 1)
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                events.append('cuda-copy')
                raise AssertionError('copy must not happen after supervisor revocation')
        def cpu_load(**kwargs):
            events.append('cpu-load')
            session.reject_check = True
            return SimpleNamespace(model=CpuModel())
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', side_effect=cpu_load):
            with self.assertRaisesRegex(RuntimeError, 'supervised lease'):
                backend.construct(budget, lease, (None, None, None))
        self.assertNotIn('cuda-copy', events)

    def test_owner_rechecks_live_headroom_after_cpu_loading_before_cuda_copy(self):
        from looking_glass.memory_budget import MemoryBudget
        events = []
        session = FakeSession(events)
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: True)
        backend._resolved_keys = ()
        backend.runtime_binding = lambda: (UUID, 1)
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                events.append('cuda-copy')
                raise AssertionError('copy must not happen without live headroom')
        free = [7 * GIB]
        def cpu_load(**kwargs):
            free[0] = GIB
            return SimpleNamespace(model=CpuModel())
        budget = MemoryBudget(lambda: CudaMemorySnapshot(free[0], 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', side_effect=cpu_load):
            with self.assertRaises(AdmissionError):
                backend.construct(budget, lease, (None, None, None))
        self.assertNotIn('cuda-copy', events)

    def test_model_verification_failure_stops_before_cuda_copy(self):
        from looking_glass.memory_budget import MemoryBudget
        events = []
        session = FakeSession(events)
        def reject_model(model):
            raise ValueError('model bytes do not match checkpoint')
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=reject_model)
        backend._resolved_keys = ()
        backend.runtime_binding = lambda: (UUID, 1)
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
            def to(self, **kwargs):
                events.append('cuda-copy')
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', return_value=SimpleNamespace(model=CpuModel())):
            with self.assertRaisesRegex(ValueError, 'do not match'):
                backend.construct(budget, lease, (None, None, None))
        self.assertNotIn('cuda-copy', events)

    def test_model_verifier_must_explicitly_return_true(self):
        from looking_glass.memory_budget import MemoryBudget
        session = FakeSession([])
        backend = _TorchServiceBackend(session=session, device_lease=object(),
            cache_dir=Path.cwd(), team_ids=('alpha',), live_key_builder=lambda identity: (),
            verify_loaded_model=lambda model: False)
        backend._resolved_keys = ()
        class CpuModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager',
                                     vocab_size=50257, n_positions=1024)
        budget = MemoryBudget(lambda: CudaMemorySnapshot(7 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease = budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        fake_service = SimpleNamespace(make_gpt2_request_factory=lambda **kwargs: None)
        with patch.dict(sys.modules, {'looking_glass.service': fake_service}), \
             patch('looking_glass.pretrained.load_distilgpt2', return_value=SimpleNamespace(model=CpuModel())):
            with self.assertRaisesRegex(RuntimeError, 'model verification'):
                backend.construct(budget, lease, (None, None, None))


class FakeBackend:
    def __init__(self, events, *, free=7 * GIB, missing=False,
                 quarantined=False, dangling=False, failure=False):
        self.events = events
        self.free = free
        self.missing = missing
        self.dangling = dangling
        self.failure = failure
        self.worker = FakeWorker()
        self.worker.quarantined = quarantined
        self.budget = None
        self.load_lease = None
        self.keys = service_keys()

    def runtime_binding(self):
        self.events.append('runtime-binding')
        return UUID, 1

    def live_keys(self, identity):
        self.events.append('derive-live-keys')
        return self.keys

    def resolve_profiles(self, registry, keys):
        self.events.append('resolve-profiles')
        if self.missing:
            raise LookupError('no approved profile')
        return (FakeProfile(GIB), FakeProfile(GIB), FakeProfile(GIB))

    def telemetry(self):
        self.events.append('telemetry')
        return CudaMemorySnapshot(self.free, 8 * GIB, 0, 0)

    def construct(self, budget, load_lease, profiles):
        self.events.append('load-and-warmup')
        self.budget, self.load_lease = budget, load_lease
        assert load_lease in budget.active_leases()
        if self.dangling:
            budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        return self.worker, object(), object()

    def synchronize(self):
        self.events.append('synchronize')

    def serve(self, server, broker, worker, stop_event, announce, verify_drain):
        self.events.append('serve-and-stop')
        verified = verify_drain()
        return FakeResult('worker_failed' if self.failure else 'clean', verified)


class OwnerOrderTests(unittest.TestCase):
    def setUp(self):
        # These are CPU-only fake resources. A real GPU process is never reset
        # after quarantine; it must be replaced by the supervisor.
        gpu_service._QUARANTINED.clear()
        gpu_service._QUARANTINE_HOLDS.clear()
        gpu_service._LIFETIME_HOLDS.clear()

    def tearDown(self):
        gpu_service._QUARANTINE_HOLDS.clear()
        gpu_service._QUARANTINED.clear()
        gpu_service._LIFETIME_HOLDS.clear()

    def setup_run(self, **kwargs):
        events = []
        session = FakeSession(events)
        backend = FakeBackend(events, **kwargs)
        return events, session, backend

    def run_owner(self, session, backend):
        return _run_bound_owner(session, object(), backend=backend,
                                registry=object(), stop_event=threading.Event(),
                                announce=lambda server: None)

    def test_supervisor_check_precedes_any_runtime_call(self):
        events, session, backend = self.setup_run()
        session.reject_check = True
        with self.assertRaisesRegex(RuntimeError, 'supervised lease'):
            self.run_owner(session, backend)
        self.assertEqual(events, ['session-check'])

    def test_identityless_supervision_is_not_accepted(self):
        events, session, backend = self.setup_run()
        session.identity = None
        with self.assertRaisesRegex(RuntimeError, 'identity-aware'):
            self.run_owner(session, backend)
        self.assertEqual(events, ['session-check'])

    def test_identity_mismatch_prevents_telemetry_and_loading(self):
        events, session, backend = self.setup_run()
        session.reject_binding = True
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            self.run_owner(session, backend)
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)

    def test_missing_profile_prevents_telemetry_and_loading(self):
        events, session, backend = self.setup_run(missing=True)
        with self.assertRaisesRegex(LookupError, 'no approved profile'):
            self.run_owner(session, backend)
        self.assertLess(events.index('verify-binding'), events.index('derive-live-keys'))
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)

    def test_unsupported_live_signature_is_rejected_before_profile_or_telemetry(self):
        events, session, backend = self.setup_run()
        load, cold, warm = backend.keys
        backend.keys = (load, replace(cold, sequence_length=8),
                        replace(warm, sequence_length=8))
        with self.assertRaisesRegex(ValueError, 'signature'):
            self.run_owner(session, backend)
        self.assertNotIn('resolve-profiles', events)
        self.assertNotIn('telemetry', events)
        self.assertNotIn('load-and-warmup', events)

    def test_headroom_is_checked_before_model_loading(self):
        events, session, backend = self.setup_run(free=GIB)
        with self.assertRaises(AdmissionError):
            self.run_owner(session, backend)
        self.assertNotIn('load-and-warmup', events)
        self.assertNotIn('attest', events)


    def test_resident_reservation_is_held_until_exit_and_attestation_is_last(self):
        events, session, backend = self.setup_run()
        result = self.run_owner(session, backend)
        self.assertTrue(result.teardown_verified)
        self.assertEqual(events.count('load-and-warmup'), 1)
        self.assertLess(events.index('verify-binding'), events.index('telemetry'))
        self.assertLess(events.index('telemetry'), events.index('load-and-warmup'))
        self.assertGreaterEqual(events.count('synchronize'), 1)
        self.assertEqual(events[-1], 'attest')
        self.assertEqual(backend.budget.active_leases(), (backend.load_lease,))
        self.assertFalse(backend.load_lease.released)

    def test_clean_owner_retains_load_lease_and_device_resources_until_exit(self):
        events, session, backend = self.setup_run()
        result = self.run_owner(session, backend)
        self.assertTrue(result.teardown_verified)
        worker_ref = weakref.ref(backend.worker)
        load_ref = weakref.ref(backend.load_lease)
        budget_ref = weakref.ref(backend.budget)
        del backend
        gc.collect()
        self.assertIsNotNone(worker_ref())
        self.assertIsNotNone(load_ref())
        self.assertIsNotNone(budget_ref())

    def test_workload_failure_can_have_verified_teardown_without_success(self):
        events, session, backend = self.setup_run(failure=True)
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'worker_failed')
        self.assertTrue(result.teardown_verified)
        self.assertIn('attest', events)

    def test_startup_failure_with_verified_drain_attests_without_claiming_success(self):
        events, session, backend = self.setup_run()
        def startup_failure(server, broker, worker, stop_event, announce, verify):
            return FakeResult('shutdown_incomplete', verify())
        backend.serve = startup_failure
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'shutdown_incomplete')
        self.assertTrue(result.teardown_verified)
        self.assertIn('attest', events)

    def test_quarantine_or_extra_lease_prevents_attestation(self):
        for kwargs in ({'quarantined': True}, {'dangling': True}):
            with self.subTest(kwargs=kwargs):
                gpu_service._QUARANTINE_HOLDS.clear()
                gpu_service._QUARANTINED.clear()
                events, session, backend = self.setup_run(**kwargs)
                result = self.run_owner(session, backend)
                self.assertFalse(result.teardown_verified)
                self.assertNotIn('attest', events)

    def test_synchronization_failure_prevents_attestation(self):
        events, session, backend = self.setup_run()
        def fail_sync():
            events.append('synchronize')
            raise RuntimeError('device failed to drain')
        backend.synchronize = fail_sync
        result = self.run_owner(session, backend)
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('attest', events)
        self.assertFalse(backend.load_lease.released)

    def test_failed_drain_latches_and_retains_resources_until_process_exit(self):
        events, session, backend = self.setup_run()
        worker_ref = weakref.ref(backend.worker)
        backend.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('lost drain'))
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertTrue(gpu_service._QUARANTINED.is_set())
        del backend
        gc.collect()
        self.assertIsNotNone(worker_ref())
        next_backend = FakeBackend(events)
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            self.run_owner(session, next_backend)

    def test_false_teardown_claim_is_independently_rechecked(self):
        events, session, backend = self.setup_run()
        def dishonest_serve(server, broker, worker, stop_event, announce, verify):
            worker.quarantined = True
            return FakeResult(teardown_verified=True)
        backend.serve = dishonest_serve
        result = self.run_owner(session, backend)
        self.assertEqual(result.status, 'quarantined')
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('attest', events)

    def test_revoked_supervision_during_load_keeps_marker(self):
        events, session, backend = self.setup_run()
        original = backend.construct
        def revoked(*args):
            components = original(*args)
            session.reject_check = True
            return components
        backend.construct = revoked
        result = self.run_owner(session, backend)
        self.assertFalse(result.teardown_verified)
        self.assertNotIn('attest', events)


class CpuComposedServiceTests(unittest.TestCase):
    def setUp(self):
        gpu_service._QUARANTINED.clear()
        gpu_service._QUARANTINE_HOLDS.clear()
        gpu_service._LIFETIME_HOLDS.clear()

    def tearDown(self):
        gpu_service._QUARANTINED.clear()
        gpu_service._QUARANTINE_HOLDS.clear()
        gpu_service._LIFETIME_HOLDS.clear()

    def test_one_warm_worker_serves_two_authenticated_teams_and_drains(self):
        import torch
        from torch import nn
        from looking_glass.broker import Broker
        from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
        from looking_glass.ipc import IPCClient, IPCRejected, IPCServer
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker
        from looking_glass.service import make_gpt2_request_factory, serve_owner

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)
                self.calls = 0
            def forward(self, input_ids):
                self.calls += 1
                return input_ids.to(torch.float32).unsqueeze(-1) + torch.tensor([0.0, 1.0])

        class CpuBackend(FakeBackend):
            def construct(self, budget, resident_lease, profiles):
                self.events.append('load-and-warmup')
                self.budget, self.load_lease = budget, resident_lease
                self.model = TinyModel().eval()
                resident = ResidentWorker('c' * 64, self.model, device='cpu',
                    max_batch_size=1, max_sequence_length=2, max_input_bytes=64,
                    max_output_bytes=1024)
                admission = DeviceAdmission(budget, device='cpu', max_estimated_incremental_bytes=2048)
                def profile(name):
                    return WorkloadProfile(name, 'c' * 64, 'cpu', (1, 2),
                        (('input_ids', 'torch.int64'),), 1024, 2048)
                cold = BudgetedWorker(resident, admission, (profile('cold'),))
                cold.execute(ForwardRequest('warmup', {'input_ids': torch.tensor([[1, 2]])}, 17))
                worker = BudgetedWorker(resident, admission, (profile('warm'),))
                broker = Broker(worker, max_pending=4, max_pending_per_team=2)
                factory = make_gpt2_request_factory(vocab_size=10, max_positions=10,
                                                    max_batch=1, max_tokens=2)
                server = IPCServer(broker, team_ids=('alpha', 'beta'), request_factory=factory,
                                   max_connections=4, max_batch=1, max_tokens=2)
                self.worker = worker
                return worker, broker, server

            def serve(self, server, broker, worker, stop_event, announce, verify_drain):
                return serve_owner(server, broker, worker, stop_event, announce=announce,
                                   verify_drain=verify_drain, shutdown_timeout=10)

        events, results = [], []
        backend, session = CpuBackend(events), FakeSession(events)
        stop = threading.Event()

        def announce(server):
            try:
                alpha = IPCClient(server.address, server.team_tokens['alpha']).submit(
                    team_id='alpha', request_id='a', inputs={'input_ids': [[1, 2]]})
                beta = IPCClient(server.address, server.team_tokens['beta']).submit(
                    team_id='beta', request_id='b', inputs={'input_ids': [[3, 4]]})
                with self.assertRaises(IPCRejected):
                    IPCClient(server.address, server.team_tokens['alpha']).submit(
                        team_id='beta', request_id='spoof', inputs={'input_ids': [[3, 4]]})
                results.extend((alpha.wait(10), beta.wait(10)))
            finally:
                stop.set()

        result = _run_bound_owner(session, object(), backend=backend, registry=object(),
                                  stop_event=stop, announce=announce)
        self.assertEqual(result.status, 'clean')
        self.assertTrue(result.teardown_verified)
        self.assertEqual([item['status'] for item in results], ['succeeded', 'succeeded'])
        self.assertEqual([item['result']['logits'] for item in results],
                         [[1.0, 2.0, 2.0, 3.0], [3.0, 4.0, 4.0, 5.0]])
        self.assertEqual(backend.model.calls, 3)
        self.assertEqual(backend.budget.active_leases(), (backend.load_lease,))


def _fake_supervised_child(mode):
    """A direct-child fake: real supervisor and locks, no actual CUDA calls."""
    # Preserve the fake visibility string for identity validation while
    # blocking all CUDA calls in this synthetic subprocess.
    _install_cuda_guard(hide_device=False)
    from looking_glass.device_lease import DeviceLease
    from looking_glass.supervisor import OwnerSession

    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=5)
    backend = FakeBackend([], failure=(mode == 'worker_failed'))
    if mode == 'sync_failed':
        backend.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('fake drain failure'))
    try:
        result = _run_bound_owner(session, lease, backend=backend, registry=object(),
                                  stop_event=threading.Event(), announce=lambda server: None)
    except BaseException:
        os._exit(4)
    # The OS closes the retained lease only on actual process exit. Failed
    # teardown must never continue the owner interpreter or run destructors.
    os._exit(0 if result.status == 'clean' and result.teardown_verified else 3)


class SupervisedSubprocessTests(unittest.TestCase):
    def test_real_supervisor_distinguishes_work_failure_from_unsafe_drain(self):
        from looking_glass.device_identity import resolve_device_identity
        from looking_glass.supervisor import run_supervised

        repo = Path(__file__).resolve().parent.parent
        site_packages = Path(sys.prefix) / 'Lib' / 'site-packages'
        if not site_packages.is_dir():
            self.skipTest('this subprocess test needs the Windows venv site-packages')
        base_python = getattr(sys, '_base_executable', None)
        self.assertTrue(base_python and Path(base_python).is_file())
        child_env = {'CUDA_VISIBLE_DEVICES': UUID,
                     'PYTHONPATH': os.pathsep.join((str(repo), str(site_packages)))}
        with tempfile.TemporaryDirectory() as directory:
            for mode, clean, exit_code in (('clean', True, 0),
                                           ('worker_failed', True, 3),
                                           ('sync_failed', False, 3)):
                with self.subTest(mode=mode):
                    root = Path(directory) / mode
                    root.mkdir()
                    with patch.dict(os.environ, child_env):
                        identity = resolve_device_identity(root, UUID, physical_uuids=(UUID,))
                        result = run_supervised(
                            [base_python, str(Path(__file__).resolve()), '--fake-owner', mode],
                            device_lease_path=identity.lock_path,
                            marker_path=identity.marker_path, identity=identity,
                            timeout=20, env=child_env, cwd=str(repo),
                        )
                    self.assertEqual(result.exit_code, exit_code)
                    self.assertEqual(result.clean, clean)
                    self.assertEqual(identity.marker_path.exists(), not clean)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fake-owner':
        _fake_supervised_child(sys.argv[2])
    else:
        unittest.main()
