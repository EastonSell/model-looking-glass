import dataclasses
import hashlib
import json
import importlib
import os
import sys
import textwrap
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from looking_glass.supervisor import OwnerSession, UnsafeOwnerError
import looking_glass.supervisor as supervisor

from looking_glass.device_identity import DeviceIdentityError, resolve_device_identity
from looking_glass.device_lease import DeviceLease, LeaseOwnershipError
from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget


HASH_A = 'a' * 64
HASH_B = 'b' * 64
HASH_C = 'c' * 64
HASH_D = 'd' * 64


def workload(module, **changes):
    fields = dict(
        checkpoint_sha256=HASH_B, checkpoint_revision='revision-1',
        model_config_sha256=HASH_C, source_sha256=HASH_D,
        runtime_settings_sha256=HASH_A, torch_version='2.8.0+cu128',
        transformers_version='4.57.3', cuda_runtime_version='12.8',
        driver_version='570.0', python_version='3.11.15',
        device='cuda:0', device_uuid='GPU-00000000-0000-0000-0000-000000000001', device_name='GPU model',
        compute_capability=(7, 5), device_total_bytes=10000,
        model_dtype='torch.float32', compute_dtype='torch.float32', attention_backend='eager',
        model_mode='eval', grad_mode='inference', autocast='disabled',
        use_cache=False, compiled=False, warmup_state='cold', warmup_protocol_sha256=HASH_C,
        operation='forward', capture_mode='none', capture_policy_sha256=HASH_B,
        batch_size=1, sequence_length=8, input_dtypes=(('input_ids', 'torch.int64'),),
        output_limit_bytes=100, max_gpu_trace_bytes=0, max_host_trace_bytes=0,
    )
    fields.update(changes)
    if 'model_id' not in fields:
        fields['model_id'] = module.model_runtime_fingerprint(fields)
    return module.WorkloadKey(**fields)


def sample(module, key=None, **changes):
    if key is None:
        key = workload(module)
    fields = dict(
        workload_sha256=key.fingerprint(), sample_id='sample-1', case_sha256=HASH_A, seed=17, status='completed',
        before=CudaMemorySnapshot(9000, 10000, 100, 300),
        after=CudaMemorySnapshot(8700, 10000, 300, 400),
        peak_allocated_bytes=400, peak_reserved_bytes=500,
    )
    fields.update(changes)
    return module.MemorySample(**fields)


def provenance(module, **changes):
    fields = dict(protocol='resource-profile-v1', source_manifest_sha256=HASH_C,
                  evidence_sha256=HASH_D, collected_utc='2026-10-28T00:00:00Z',
                  isolation='cooperative-exclusive', boundaries='synchronize-reset-synchronize')
    fields.update(changes)
    return module.ProfileProvenance(**fields)


def measured_profile(module, key, samples, *, margin_bytes=50, margin_bps=0,
                     source_manifest=b'source manifest'):
    evidence = module.encode_evidence(key, samples)
    source_hash = hashlib.sha256(source_manifest).hexdigest()
    record = provenance(module, evidence_sha256=hashlib.sha256(evidence).hexdigest(),
                        source_manifest_sha256=source_hash)
    return module.MeasuredResourceProfile.build(
        key, samples, record, margin_bytes=margin_bytes, margin_bps=margin_bps)


GIB = 1024 ** 3


class FakeProbe:
    def __init__(self, key, events, *, fail_sync_at=None):
        self.device = key.device
        self.device_uuid = key.device_uuid
        self.runtime_uuid = key.device_uuid
        self.visible_device_count = 1
        self.binding_calls = 0
        self.events = events
        self.calls = 0
        self.fail_sync_at = fail_sync_at
        self.samples = iter((CudaMemorySnapshot(6 * GIB, 8 * GIB, GIB, 2 * GIB),
                             CudaMemorySnapshot(5 * GIB, 8 * GIB, 2 * GIB, 3 * GIB)))

    def runtime_binding(self):
        self.binding_calls += 1
        return self.runtime_uuid, self.visible_device_count

    def synchronize(self):
        self.events.append('sync')
        self.calls += 1
        if self.calls == self.fail_sync_at:
            raise RuntimeError('device synchronization failed')

    def snapshot(self):
        self.events.append('snapshot')
        return next(self.samples)

    def reset_peak(self):
        self.events.append('reset')

    def peak_allocated_bytes(self):
        self.events.append('peak-allocated')
        return 2 * GIB

    def peak_reserved_bytes(self):
        self.events.append('peak-reserved')
        return 3 * GIB


class FakeExclusiveLease:
    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail

    def check(self):
        self.events.append('guard')
        if self.fail:
            raise RuntimeError('device lease unavailable')


def profiling_setup(module, testcase, *, fail_sync_at=None):
    key = workload(module, device_total_bytes=8 * GIB)
    events = []
    probe = FakeProbe(key, events, fail_sync_at=fail_sync_at)
    tempdir = tempfile.TemporaryDirectory()
    testcase.addCleanup(tempdir.cleanup)
    env = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': key.device_uuid,
                                  'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
    env.start()
    testcase.addCleanup(env.stop)
    identity = resolve_device_identity(tempdir.name, key.device_uuid,
                                       physical_uuids=(key.device_uuid,))
    probe.identity = identity
    exclusive = DeviceLease.acquire(identity.lock_path, timeout=1)
    testcase.addCleanup(exclusive.release)
    # Unit-only synthetic marker for CPU fakes; production uses run_supervised.
    run_id = 'a' * 32
    spec = supervisor._identity_spec(identity)
    identity.marker_path.write_text(json.dumps(supervisor._marker(run_id, identity.lock_path, spec)),
                                    encoding='utf-8')
    parent_pid = os.getppid()
    if os.name == 'nt':
        kernel = supervisor._windows_kernel()
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, parent_pid)
        if not handle:
            raise RuntimeError('cannot inspect the test parent')
        try:
            parent_birth = supervisor._windows_birth(kernel, handle)
        finally:
            kernel.CloseHandle(handle)
    else:
        parent_birth = 'posix-parent'
    probe.owner_session = OwnerSession(run_id, identity.lock_path, identity.marker_path,
                                       parent_pid, parent_birth, spec)
    initial = CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
    budget = MemoryBudget(lambda: initial, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    lease = budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    return key, events, probe, exclusive, budget, lease


class ResourceProfileTests(unittest.TestCase):
    def test_registry_requires_exact_profile_and_uses_measured_margin(self):
        try:
            module = importlib.import_module('looking_glass.resource_profile')
        except ModuleNotFoundError as exc:
            self.fail(f'resource profile implementation is unavailable: {exc}')
        key = workload(module)
        profile = measured_profile(module, key, (sample(module),), margin_bytes=50, margin_bps=1000)
        registry = module.ProfileRegistry((profile,), approved_profile_ids=(profile.profile_id,))
        selected = registry.resolve(key)
        # max(300 allocated delta, 200 reserved delta, 300 device-free drop)
        # + 64 input + 100 output + 50 margin + ceil(464 * .10).
        self.assertEqual(selected.estimated_incremental_bytes, 561)
        with self.assertRaises(module.ProfileUnavailable):
            registry.resolve(dataclasses.replace(key, sequence_length=9))


    def test_identity_rejects_missing_unsafe_or_mutable_configuration(self):
        module = importlib.import_module('looking_glass.resource_profile')
        invalid = [
            {'model_id': 'abc'}, {'checkpoint_sha256': 'z' * 64},
            {'checkpoint_revision': ''}, {'runtime_settings_sha256': HASH_A.upper()},
            {'torch_version': 'unknown'}, {'cuda_runtime_version': ''},
            {'device': 'cuda'}, {'device': 'cuda:1'}, {'device_uuid': ''}, {'device_name': ''},
            {'compute_capability': (True, 5)}, {'device_total_bytes': -1},
            {'model_dtype': ''}, {'attention_backend': ''}, {'model_mode': 'train'},
            {'grad_mode': 'unknown'}, {'autocast': ''}, {'use_cache': 1}, {'compiled': 'no'},
            {'warmup_state': 'unknown'}, {'warmup_protocol_sha256': 'bad'},
            {'operation': 'unknown'}, {'capture_mode': 'unknown'},
            {'batch_size': 0}, {'sequence_length': True},
            {'input_dtypes': [('input_ids', 'torch.int64')]},
            {'input_dtypes': (('input_ids', 'torch.int64'), ('input_ids', 'torch.int32'))},
            {'input_dtypes': (('z', 'torch.int64'), ('a', 'torch.int64'))},
            {'input_dtypes': (('input_ids', 'torch.unknown'),)},
            {'output_limit_bytes': -1}, {'max_gpu_trace_bytes': True},
            {'max_host_trace_bytes': 1},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                workload(module, **changes)

    def test_model_load_profile_has_separate_non_request_shape(self):
        module = importlib.import_module('looking_glass.resource_profile')
        load = workload(module, operation='model_load', batch_size=0, sequence_length=0,
                        input_dtypes=(), output_limit_bytes=0)
        profile = measured_profile(module, load, (sample(module, key=load),))
        self.assertEqual(profile.estimated_incremental_bytes, 350)
        with self.assertRaises((TypeError, ValueError)):
            dataclasses.replace(load, batch_size=1)
        with self.assertRaises((TypeError, ValueError)):
            dataclasses.replace(load, capture_mode='direct')

    def test_sample_rejects_invalid_or_inconsistent_telemetry(self):
        module = importlib.import_module('looking_glass.resource_profile')
        invalid = [
            {'sample_id': ''}, {'case_sha256': 'a'}, {'seed': True}, {'seed': -1},
            {'status': 'failed'}, {'status': 'cancelled'}, {'status': 'oom'},
            {'before': None}, {'peak_allocated_bytes': -1}, {'peak_reserved_bytes': True},
            {'peak_allocated_bytes': 299}, {'peak_reserved_bytes': 399},
            {'peak_allocated_bytes': 600}, {'peak_reserved_bytes': 10001},
            {'after': CudaMemorySnapshot(8000, 9000, 300, 400)},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                sample(module, **changes)

    def test_profile_rejects_missing_samples_bad_margin_and_duplicate_evidence(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key = workload(module)
        good = sample(module)
        for samples, margin_bytes, margin_bps in (
                ((), 50, 0), ([good], 50, 0), ((good, good), 50, 0),
                ((good,), 0, 0), ((good,), True, 0), ((good,), 50, -1),
                ((good,), 50, 1.5)):
            with self.subTest(samples=samples, margin=margin_bytes, bps=margin_bps):
                with self.assertRaises((TypeError, ValueError)):
                    measured_profile(module, key, samples, margin_bytes=margin_bytes, margin_bps=margin_bps)
        different_device = workload(module, device_total_bytes=9000)
        with self.assertRaises((TypeError, ValueError)):
            measured_profile(module, different_device, (good,))

    def test_registry_rejects_duplicate_keys_and_does_not_accept_client_estimates(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key = workload(module)
        profile = measured_profile(module, key, (sample(module),))
        with self.assertRaises((TypeError, ValueError)):
            module.ProfileRegistry((profile, profile), approved_profile_ids=(profile.profile_id,))
        registry = module.ProfileRegistry((profile,), approved_profile_ids=(profile.profile_id,))
        with self.assertRaises(module.ProfileUnavailable):
            registry.resolve({'key': key, 'estimated_incremental_bytes': 1})


    def test_model_id_is_bound_to_runtime_device_and_model_identity(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key = workload(module)
        for change in ({'model_dtype': 'torch.float16'}, {'driver_version': '571.0'},
                       {'device_uuid': 'GPU-00000000-0000-0000-0000-000000000002'},
                       {'source_sha256': HASH_A}, {'grad_mode': 'no_grad'}, {'use_cache': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(key, **change)
            changed = workload(module, **change)
            self.assertNotEqual(changed.model_id, key.model_id)
        with self.assertRaises(ValueError):
            workload(module, device_uuid='GPU-0001')

    def test_canonical_fingerprint_is_stable_and_rejects_ambiguous_values(self):
        module = importlib.import_module('looking_glass.resource_profile')
        left = {'z': [1, True], 'a': {'x': 'value'}}
        right = {'a': {'x': 'value'}, 'z': [1, True]}
        self.assertEqual(module.canonical_fingerprint(left), module.canonical_fingerprint(right))
        self.assertNotEqual(module.canonical_fingerprint(left),
                            module.canonical_fingerprint({'z': [True, 1], 'a': {'x': 'value'}}))
        for invalid in ({1: 'key'}, {'x': float('nan')}, {'x': 1.5}, {'x': object()}):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                module.canonical_fingerprint(invalid)


    def test_sample_cannot_be_relabelled_for_a_different_model_or_workload(self):
        module = importlib.import_module('looking_glass.resource_profile')
        original = workload(module)
        measured = sample(module, key=original)
        for changed in (workload(module, checkpoint_sha256=HASH_D),
                        workload(module, sequence_length=16),
                        workload(module, capture_mode='id')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                measured_profile(module, changed, (measured,))


    def test_provenance_rejects_nonexclusive_or_unsynchronized_measurements(self):
        module = importlib.import_module('looking_glass.resource_profile')
        for change in ({'isolation': 'shared'}, {'boundaries': 'unsynchronized'},
                       {'boundaries': 'synchronize-without-reset'}, {'protocol': 'future-version'},
                       {'source_manifest_sha256': 'bad'}, {'evidence_sha256': 'bad'},
                       {'collected_utc': '2026-10-28T00:00:00'}):
            with self.subTest(change=change), self.assertRaises((TypeError, ValueError)):
                provenance(module, **change)


    def test_profile_round_trip_verifies_evidence_and_source_manifest(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key = workload(module)
        measurements = (sample(module, key=key),)
        source = b'verified source manifest'
        profile = measured_profile(module, key, measurements, source_manifest=source)
        evidence = module.encode_evidence(key, measurements)
        restored = module.load_profile(profile.to_json(), evidence=evidence, source_manifest=source, expected_profile_id=profile.profile_id)
        self.assertEqual(restored, profile)
        self.assertEqual(restored.profile_id, profile.profile_id)
        with self.assertRaises(ValueError):
            module.load_profile(profile.to_json(), evidence=evidence + b' ', source_manifest=source, expected_profile_id=profile.profile_id)
        with self.assertRaises(ValueError):
            module.load_profile(profile.to_json(), evidence=evidence, source_manifest=source + b' ', expected_profile_id=profile.profile_id)
        altered = json.loads(profile.to_json())
        altered['samples'][0]['peak_allocated_bytes'] += 1
        with self.assertRaises(ValueError):
            module.load_profile(json.dumps(altered), evidence=evidence, source_manifest=source, expected_profile_id=profile.profile_id)
        altered = json.loads(profile.to_json())
        altered['estimated_incremental_bytes'] = 1
        with self.assertRaises(ValueError):
            module.load_profile(json.dumps(altered), evidence=evidence, source_manifest=source, expected_profile_id=profile.profile_id)
        with self.assertRaises(ValueError):
            module.load_profile('{"schema":"x","schema":"y"}', evidence=evidence, source_manifest=source, expected_profile_id=profile.profile_id)


    def test_collect_sample_synchronizes_and_resets_inside_active_leases(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        def operation():
            events.append('run')
        measured = module.collect_sample(key, probe, operation, owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
            memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertEqual(events, ['sync', 'snapshot', 'reset', 'run', 'sync',
                                  'snapshot', 'peak-allocated', 'peak-reserved'])
        self.assertEqual(measured.workload_sha256, key.fingerprint())
        self.assertEqual(measured.peak_allocated_bytes, 2 * GIB)
        self.assertEqual(measured.peak_reserved_bytes, 3 * GIB)
        self.assertFalse(lease.released)
        lease.release()

    def test_failed_operation_is_not_a_sample_and_lease_stays_active(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        class FakeOOM(RuntimeError):
            pass
        def operation():
            events.append('run')
            raise FakeOOM('allocation failed')
        with self.assertRaises(FakeOOM):
            module.collect_sample(key, probe, operation, owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertEqual(events, ['sync', 'snapshot', 'reset', 'run', 'sync'])
        self.assertFalse(lease.released)
        lease.release()  # caller confirms sync and teardown completed

    def test_failed_sync_quarantines_measurement_without_releasing_lease(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self, fail_sync_at=2)
        with self.assertRaises(module.ProfilingSynchronizationError):
            module.collect_sample(key, probe, lambda: None, owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertFalse(lease.released)
        self.assertEqual(budget.snapshot().active_leases, 1)
        # Production quarantines the process; test cleanup only has fake CUDA work.
        lease.release()

    def test_invalid_device_or_missing_lease_fails_before_running(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        probe.device_uuid = 'GPU-00000000-0000-0000-0000-000000000002'
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        probe.device_uuid = key.device_uuid
        lease.release()
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)


    def test_probe_total_mismatch_is_rejected_before_workload_runs(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        probe.samples = iter((CudaMemorySnapshot(6 * GIB, 7 * GIB, GIB, 2 * GIB),))
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        lease.release()

    def test_forged_memory_lease_cannot_authorize_profiling(self):
        module = importlib.import_module('looking_glass.resource_profile')
        from looking_glass.memory_budget import MemoryLease
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        forged = MemoryLease(budget, GIB, 0, 0)
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=forged, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        lease.release()

    def test_operation_and_cleanup_sync_failure_preserves_original_error(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self, fail_sync_at=2)
        class FakeOOM(RuntimeError):
            pass
        def operation():
            raise FakeOOM('out of memory')
        with self.assertRaises(module.ProfilingSynchronizationError) as error:
            module.collect_sample(key, probe, operation, owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertIsInstance(error.exception.original_error, FakeOOM)
        self.assertFalse(lease.released)
        lease.release()  # fake device only


    def test_fresh_headroom_after_initial_sync_is_rechecked_before_operation(self):
        module = importlib.import_module('looking_glass.resource_profile')
        from looking_glass.memory_budget import AdmissionError
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        # Free memory falls below active 1 GiB commitment plus the 1.2 GiB reserve.
        probe.samples = iter((CudaMemorySnapshot(2 * GIB, 8 * GIB, GIB, 2 * GIB),))
        with self.assertRaises(AdmissionError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        self.assertFalse(lease.released)
        lease.release()


    def test_forged_exclusive_guard_cannot_authorize_profiling(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'),
                owner_session=probe.owner_session, device_identity=probe.identity, device_lease=FakeExclusiveLease(events), memory_lease=lease,
                sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        lease.release()


    def test_budget_total_change_after_initial_sync_fails_before_operation(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        state = [CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)]
        budget._telemetry = lambda: state[0]
        original_sync = probe.synchronize
        def change_device_total():
            original_sync()
            state[0] = CudaMemorySnapshot(7 * GIB, 7 * GIB, 0, 0)
        probe.synchronize = change_device_total
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        lease.release()


    def test_released_device_lease_with_overridden_check_cannot_run_work(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        exclusive.release()
        exclusive.check = lambda: None
        with self.assertRaises(LeaseOwnershipError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        lease.release()


    def test_profiling_lease_must_cover_declared_input_and_output_allowances(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        lease.release()
        underfunded = budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=underfunded, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        underfunded.release()

    def test_profiling_lease_trace_caps_must_cover_capture_policy(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        key = workload(module, device_total_bytes=8 * GIB, capture_mode='direct',
                       max_gpu_trace_bytes=500, max_host_trace_bytes=100)
        probe.device = key.device
        probe.device_uuid = key.device_uuid
        with self.assertRaises(ValueError):
            module.collect_sample(key, probe, lambda: events.append('run'), owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive,
                memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        lease.release()


    def test_changed_margin_cannot_bypass_independent_profile_pin(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key = workload(module)
        records = (sample(module),)
        source = b'source manifest'
        profile = measured_profile(module, key, records, margin_bytes=500, source_manifest=source)
        evidence = module.encode_evidence(key, records)
        altered = json.loads(profile.to_json())
        altered['margin_bytes'] = 1
        with self.assertRaises(ValueError):
            module.load_profile(json.dumps(altered), evidence=evidence, source_manifest=source,
                                expected_profile_id=profile.profile_id)
        with self.assertRaises(ValueError):
            module.ProfileRegistry((profile,), approved_profile_ids=(HASH_A,))


    def test_worker_bridge_requires_full_matching_live_key_and_supported_operation(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key = workload(module)
        profile = measured_profile(module, key, (sample(module),))
        registry = module.ProfileRegistry((profile,), approved_profile_ids=(profile.profile_id,))
        resolved = module.resolve_worker_profile(registry, key)
        self.assertEqual(resolved.profile_id, profile.profile_id)
        self.assertEqual(resolved.model_id, key.model_id)
        self.assertEqual(resolved.device, key.device)
        self.assertEqual(resolved.shape, (1, 8))
        self.assertEqual(resolved.input_dtypes, (('input_ids', 'torch.int64'),))
        self.assertEqual(resolved.output_limit_bytes, 100)
        self.assertEqual(resolved.estimated_incremental_bytes, 514)
        with self.assertRaises(module.ProfileUnavailable):
            module.resolve_worker_profile(registry, dataclasses.replace(key, sequence_length=9))
        with self.assertRaises(module.ProfileUnavailable):
            module.resolve_worker_profile(registry, {'model_id': key.model_id, 'estimate': 1})

        capture = workload(module, capture_mode='direct', max_gpu_trace_bytes=100)
        measured = measured_profile(module, capture, (sample(module, key=capture),))
        capture_registry = module.ProfileRegistry((measured,), approved_profile_ids=(measured.profile_id,))
        with self.assertRaises(module.ProfileUnavailable):
            module.resolve_worker_profile(capture_registry, capture)

        unsupported = workload(module, input_dtypes=(('embeddings', 'torch.float32'),))
        measured = measured_profile(module, unsupported, (sample(module, key=unsupported),))
        unsupported_registry = module.ProfileRegistry((measured,), approved_profile_ids=(measured.profile_id,))
        with self.assertRaises(module.ProfileUnavailable):
            module.resolve_worker_profile(unsupported_registry, unsupported)


    def test_worker_bridge_rejects_other_execution_modes_even_with_profiles(self):
        module = importlib.import_module('looking_glass.resource_profile')
        for change in ({'grad_mode': 'no_grad'}, {'use_cache': True},
                       {'compiled': True}, {'autocast': 'torch.float16'}):
            with self.subTest(change=change):
                key = workload(module, **change)
                profile = measured_profile(module, key, (sample(module, key=key),))
                registry = module.ProfileRegistry((profile,), approved_profile_ids=(profile.profile_id,))
                with self.assertRaises(module.ProfileUnavailable):
                    module.resolve_worker_profile(registry, key)

    def test_warmup_state_is_part_of_exact_workload_identity(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key = workload(module, warmup_state='cold')
        profile = measured_profile(module, key, (sample(module, key=key),))
        registry = module.ProfileRegistry((profile,), approved_profile_ids=(profile.profile_id,))
        with self.assertRaises(module.ProfileUnavailable):
            registry.resolve(dataclasses.replace(key, warmup_state='warm'))


    def test_wrong_lock_path_or_runtime_binding_fails_before_workload(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        other = DeviceLease.acquire(Path(probe.identity.workspace_root) / 'other.lock', timeout=1)
        self.addCleanup(other.release)
        with self.assertRaises(DeviceIdentityError):
            module.collect_sample(key, probe, lambda: events.append('run'),
                owner_session=probe.owner_session, device_identity=probe.identity, device_lease=other, memory_lease=lease,
                sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        self.assertEqual(probe.binding_calls, 0)
        probe.runtime_uuid = 'GPU-00000000-0000-0000-0000-000000000002'
        with self.assertRaises(DeviceIdentityError):
            module.collect_sample(key, probe, lambda: events.append('run'),
                owner_session=probe.owner_session, device_identity=probe.identity, device_lease=exclusive, memory_lease=lease,
                sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertNotIn('run', events)
        lease.release()



    def test_missing_or_forged_supervised_owner_never_calls_cuda_or_budget(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        budget._telemetry = lambda: events.append('budget')
        for owner in (None, FakeExclusiveLease(events)):
            with self.subTest(owner=owner), self.assertRaises((TypeError, ValueError)):
                module.collect_sample(key, probe, lambda: events.append('run'),
                    owner_session=owner, device_identity=probe.identity, device_lease=exclusive,
                    memory_lease=lease, sample_id='sample-1', case_sha256=HASH_A, seed=17)
        self.assertEqual(probe.binding_calls, 0)
        self.assertEqual(events, [])
        lease.release()

    def test_missing_marker_rejects_even_overridden_session_before_cuda_or_budget(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        probe.owner_session.check = lambda lease: None
        probe.owner_session.marker_path.unlink()
        budget._telemetry = lambda: events.append('budget')
        with self.assertRaises(UnsafeOwnerError):
            module.collect_sample(key, probe, lambda: events.append('run'),
                owner_session=probe.owner_session, device_identity=probe.identity,
                device_lease=exclusive, memory_lease=lease, sample_id='sample-1',
                case_sha256=HASH_A, seed=17)
        self.assertEqual(probe.binding_calls, 0)
        self.assertEqual(events, [])
        lease.release()

    def test_lost_owner_marker_during_presync_prevents_work(self):
        module = importlib.import_module('looking_glass.resource_profile')
        key, events, probe, exclusive, budget, lease = profiling_setup(module, self)
        sync = probe.synchronize
        def lose_marker():
            sync()
            probe.owner_session.marker_path.unlink()
        probe.synchronize = lose_marker
        with self.assertRaises(UnsafeOwnerError):
            module.collect_sample(key, probe, lambda: events.append('run'),
                owner_session=probe.owner_session, device_identity=probe.identity,
                device_lease=exclusive, memory_lease=lease, sample_id='sample-1',
                case_sha256=HASH_A, seed=17)
        self.assertNotIn('reset', events)
        self.assertNotIn('run', events)
        lease.release()


    def test_real_supervisor_owns_cpu_fake_profiling_through_clean_exit(self):
        key = workload(importlib.import_module('looking_glass.resource_profile'), device_total_bytes=8 * GIB)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'CUDA_VISIBLE_DEVICES': key.device_uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}):
            identity = resolve_device_identity(directory, key.device_uuid,
                                               physical_uuids=(key.device_uuid,))
            child = textwrap.dedent("""
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path.cwd() / 'tests'))
                import test_resource_profile as fixtures
                import looking_glass.resource_profile as profiles
                from looking_glass.supervisor import OwnerSession
                from looking_glass.device_lease import DeviceLease
                from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget

                key = fixtures.workload(profiles, device_total_bytes=8 * fixtures.GIB)
                session = OwnerSession.from_environment()
                device_lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
                OwnerSession.check(session, device_lease)
                # This runtime identity is fake because the complete test is CPU-only.
                OwnerSession.verify_runtime_binding(session, device_lease,
                    runtime_uuid=key.device_uuid, device_count=1)
                initial = CudaMemorySnapshot(8 * fixtures.GIB, 8 * fixtures.GIB, 0, 0)
                budget = MemoryBudget(lambda: initial, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
                memory_lease = budget.acquire(fixtures.GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
                events = []
                probe = fixtures.FakeProbe(key, events)
                sample = profiles.collect_sample(key, probe, lambda: events.append('run'),
                    owner_session=session, device_identity=session.identity, device_lease=device_lease,
                    memory_lease=memory_lease, sample_id='sample-1', case_sha256=fixtures.HASH_A, seed=17)
                assert sample.workload_sha256 == key.fingerprint() and 'run' in events
                memory_lease.release()
                OwnerSession.attest_clean(session, device_lease)
                # Hold the device lease through process exit.
            """)
            outcome = supervisor.run_supervised(
                [sys._base_executable, '-c', child], identity=identity,
                device_lease_path=identity.lock_path, marker_path=identity.marker_path,
                timeout=10, lease_timeout=3)
            self.assertEqual(outcome.exit_code, 0)
            self.assertTrue(outcome.clean, outcome.reason)
            self.assertFalse(identity.marker_path.exists())

    def test_torch_probe_preserves_device_and_distinguishes_memory_channels(self):
        module = importlib.import_module('looking_glass.resource_profile')
        class FakeCuda:
            def synchronize(self, device):
                self.last_sync = device
            def mem_get_info(self, device):
                return (6000 if device == 'cuda:0' else 0, 10000)
            def memory_allocated(self, device):
                return 100
            def memory_reserved(self, device):
                return 300
            def reset_peak_memory_stats(self, device):
                self.reset_device = device
            def max_memory_allocated(self, device):
                return 400
            def max_memory_reserved(self, device):
                return 500
        cuda = FakeCuda()
        probe = module.TorchMemoryProbe(cuda, device='cuda:0',
            device_uuid='GPU-00000000-0000-0000-0000-000000000001',
            runtime_binding=lambda: ('GPU-00000000-0000-0000-0000-000000000001', 1))
        self.assertEqual(probe.snapshot(), CudaMemorySnapshot(6000, 10000, 100, 300))
        probe.synchronize()
        probe.reset_peak()
        self.assertEqual(cuda.last_sync, 'cuda:0')
        self.assertEqual(cuda.reset_device, 'cuda:0')
        self.assertEqual(probe.peak_allocated_bytes(), 400)
        self.assertEqual(probe.peak_reserved_bytes(), 500)


if __name__ == '__main__':
    unittest.main()
