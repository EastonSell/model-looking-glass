"""CPU-only tests for the disabled five-token GPU trace preparation."""
from dataclasses import asdict, replace
import hashlib
import unittest
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass import trace_gpu_owner as owner
from looking_glass.memory_budget import CudaMemorySnapshot
from looking_glass.pretrained import ASSETS, REVISION
from looking_glass.resource_profile import (
    MemorySample, MeasuredResourceProfile, ProfileProvenance, ProfileRegistry,
    ProfileUnavailable, WorkloadKey, encode_evidence, model_runtime_fingerprint,
)

GIB = 1024 ** 3
UUID = 'GPU-00000000-0000-0000-0000-000000000001'


def keys():
    base = dict(
        checkpoint_sha256=ASSETS['model.safetensors'][1], checkpoint_revision=REVISION,
        model_config_sha256=ASSETS['config.json'][1], source_sha256='a' * 64,
        runtime_settings_sha256='b' * 64, torch_version='2.8.0+cu128',
        transformers_version='4.57.3', cuda_runtime_version='12.8',
        driver_version='1', python_version='3.11.15', device='cuda:0',
        device_uuid=UUID, device_name='test GPU', compute_capability=(7, 5),
        device_total_bytes=8 * GIB, model_dtype='torch.float32',
        compute_dtype='torch.float32', attention_backend='eager', model_mode='eval',
        grad_mode='inference', autocast='disabled', use_cache=False, compiled=False,
        warmup_protocol_sha256=owner.protocol_fingerprint(),
    )
    base['model_id'] = model_runtime_fingerprint(base)
    result = []
    for index, phase in enumerate(owner.PHASES):
        is_load = phase.operation == 'model_load'
        result.append(WorkloadKey(**base, operation=phase.operation,
                                  capture_mode=phase.capture_mode,
                                  warmup_state=phase.warmup_state,
                                  capture_policy_sha256=owner.phase_policy_fingerprint(index),
                                  batch_size=0 if is_load else 1,
                                  sequence_length=0 if is_load else 5,
                                  input_dtypes=() if is_load else owner.INPUT_DTYPES,
                                  output_limit_bytes=0 if is_load else owner.OUTPUT_BYTES,
                                  max_gpu_trace_bytes=phase.gpu_trace_allowance,
                                  max_host_trace_bytes=phase.host_trace_allowance))
    return tuple(result)


def profile(key, *, extra_peak=0, seed=0, case_sha256=None):
    index = next(i for i in range(len(owner.PHASES))
                 if key.capture_policy_sha256 == owner.phase_policy_fingerprint(i))
    before = CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0)
    after = CudaMemorySnapshot(6 * GIB - extra_peak, 8 * GIB, extra_peak, extra_peak)
    sample = MemorySample(workload_sha256=key.fingerprint(), sample_id='synthetic-1',
                          case_sha256=case_sha256 or owner.phase_case_fingerprint(index),
                          seed=seed, status='completed', before=before,
                          after=after, peak_allocated_bytes=extra_peak,
                          peak_reserved_bytes=extra_peak)
    samples = (sample,)
    evidence = encode_evidence(key, samples)
    provenance = ProfileProvenance(protocol='resource-profile-v1',
        source_manifest_sha256=hashlib.sha256(b'synthetic source').hexdigest(),
        evidence_sha256=hashlib.sha256(evidence).hexdigest(),
        collected_utc='2026-10-28T00:00:00Z', isolation='cooperative-exclusive',
        boundaries='synchronize-reset-synchronize')
    return MeasuredResourceProfile.build(key, samples, provenance,
        margin_bytes=owner.PROFILE_MARGIN_BYTES, margin_bps=owner.PROFILE_MARGIN_BPS)


def registry(profiles):
    return ProfileRegistry(tuple(profiles), approved_profile_ids=tuple(p.profile_id for p in profiles))


def validate_profiles(profiles, *, physical_uuid, live):
    return owner.validate_trace_profiles(profiles, physical_uuid=physical_uuid,
        verified_model_runtime_id=keys()[0].model_id, keys=live)


class FakeSession:
    def __init__(self, events):
        self.identity = SimpleNamespace(gpu_uuid=UUID)
        self.events = events
        self.attestations = 0

    def check(self, lease):
        self.events.append('session-check')

    def verify_runtime_binding(self, lease, *, runtime_uuid, device_count, cuda_device_index):
        self.events.append('binding')
        if (runtime_uuid, device_count, cuda_device_index) != (UUID, 1, 0):
            raise ValueError('runtime binding mismatch')

    def attest_clean(self, lease):
        self.events.append('attest')
        self.attestations += 1


class FakeBackend:
    def __init__(self, live, events):
        self.live, self.events = live, events
        self.quarantined = False
        self.failure_phase = None
        self.failure = None
        self.hook_leak_phase = None
        self.rebind_phase = None
        self.rebind_site = None
        self.block_phase = None
        self.block_entered = Event()
        self.block_release = Event()
        self.uuid = UUID
        self.drained = True
        self.hooks_clear = True
        self.allocated = 0
        self.reserved = 0
        self.free = 6 * GIB
        self.total = 8 * GIB
        self.cap = owner.ALLOCATOR_CEILING

    def runtime_binding(self):
        self.events.append('runtime-uuid')
        return self.uuid, 1

    def live_keys(self, identity):
        self.events.append('live-keys')
        if self.rebind_site == 'live-keys':
            self.uuid = UUID[:-1] + '2'
        return self.live

    def verified_model_runtime_id(self, identity):
        self.events.append('live-model-identity')
        return self.live[0].model_id

    def synchronize(self):
        self.events.append('sync')

    def telemetry(self):
        return CudaMemorySnapshot(self.free, self.total, self.allocated, self.reserved)

    def configure_allocator(self, byte_limit, total):
        self.events.append('configure-allocator')
        if self.rebind_site == 'configure-allocator':
            self.uuid = UUID[:-1] + '2'
        return True

    def allocator_ceiling_bytes(self, total):
        return self.cap

    def forward_hooks_clear(self):
        return self.hooks_clear

    def begin_measurement(self, phase, before):
        self.events.append('begin:' + phase.name)
        if phase.name == self.rebind_phase:
            self.uuid = UUID[:-1] + '2'
        return True

    def run_phase(self, phase, profile, memory_lease):
        self.events.append('run:' + phase.name)
        if phase.name == self.failure_phase:
            raise self.failure
        if phase.name == self.hook_leak_phase:
            self.hooks_clear = False
        if self.rebind_site == 'run:' + phase.name:
            self.uuid = UUID[:-1] + '2'
        if phase.name == self.block_phase:
            self.block_entered.set()
            if not self.block_release.wait(5):
                raise TimeoutError('fake blocked phase timed out')
        return True

    def end_measurement(self, phase, before, after):
        self.events.append('end:' + phase.name)
        if self.rebind_site == 'end:' + phase.name:
            self.uuid = UUID[:-1] + '2'
        return True

    def dispose(self):
        self.events.append('dispose')
        if self.rebind_site == 'dispose':
            self.uuid = UUID[:-1] + '2'
        return True

    def verify_drained(self):
        self.events.append('drained')
        return self.drained


class TraceGpuOwnerTests(unittest.TestCase):
    def setUp(self):
        used = patch.object(owner, '_OWNER_USED', False)
        used.start()
        self.addCleanup(used.stop)

    def test_signature_and_payload_arithmetic_are_fixed(self):
        self.assertEqual(owner.TOKENS, (464, 6766, 318, 4171, 13))
        self.assertEqual(owner.MODEL_LOGICAL_BYTES, 327650304)
        self.assertEqual(owner.OUTPUT_BYTES, 1005140)
        self.assertEqual(18 * 2 * 768 * 4, 110592)
        self.assertEqual([p.forwards for p in owner.PHASES], [0, 1, 1, 2, 2, 2, 2])
        self.assertEqual(len({owner.phase_policy_fingerprint(i) for i in range(7)}), 7)
        self.assertEqual(owner.ALLOCATOR_CEILING, 1536 * 1024 ** 2)

    def test_exact_seven_profile_registry_resolves(self):
        live = keys()
        profiles = tuple(profile(key) for key in live)
        self.assertEqual(validate_profiles(registry(profiles), physical_uuid=UUID, live=live),
                         profiles)

    def test_two_token_and_eight_token_shapes_never_authorize_five(self):
        live = keys()
        profiles = tuple(profile(key) for key in live)
        for length in (2, 8):
            with self.subTest(length=length):
                altered = live[:1] + (replace(live[1], sequence_length=length),) + live[2:]
                with self.assertRaisesRegex(ValueError, 'wrong inputs'):
                    validate_profiles(registry(profiles), physical_uuid=UUID, live=altered)

    def test_mismatched_phase_capture_and_device_fail_closed(self):
        live = keys()
        profiles = tuple(profile(key) for key in live)
        for changed in (replace(live[3], capture_policy_sha256='f' * 64),
                        replace(live[3], capture_mode='gradient'),
                        replace(live[3], max_host_trace_bytes=2 * owner.SESSION_TRACE_CAP)):
            with self.subTest(changed=changed):
                bad = live[:3] + (changed,) + live[4:]
                with self.assertRaises(ValueError):
                    validate_profiles(registry(profiles), physical_uuid=UUID, live=bad)
        with self.assertRaises(ValueError):
            validate_profiles(registry(profiles), physical_uuid=UUID[:-1] + '2', live=live)

    def test_different_live_source_identity_and_missing_profile_are_rejected(self):
        live = keys()
        profiles = tuple(profile(key) for key in live)
        fields = asdict(live[4])
        fields['compute_capability'] = tuple(fields['compute_capability'])
        fields['input_dtypes'] = tuple(tuple(item) for item in fields['input_dtypes'])
        fields['source_sha256'] = 'd' * 64
        fields['model_id'] = model_runtime_fingerprint(fields)
        changed = WorkloadKey(**fields)
        with self.assertRaises(ValueError):
            validate_profiles(registry(profiles), physical_uuid=UUID,
                              live=live[:4] + (changed,) + live[5:])
        with self.assertRaises(ProfileUnavailable):
            validate_profiles(registry(profiles[:-1]), physical_uuid=UUID, live=live)

    def test_consistently_changed_profile_identity_cannot_replace_live_identity(self):
        changed = []
        for key in keys():
            fields = asdict(key)
            fields['compute_capability'] = tuple(fields['compute_capability'])
            fields['input_dtypes'] = tuple(tuple(item) for item in fields['input_dtypes'])
            fields['source_sha256'] = 'd' * 64
            fields['model_id'] = model_runtime_fingerprint(fields)
            changed.append(WorkloadKey(**fields))
        changed = tuple(changed)
        with self.assertRaisesRegex(ValueError, 'fixed phase'):
            validate_profiles(registry(tuple(profile(key) for key in changed)),
                              physical_uuid=UUID, live=changed)

    def test_profile_estimates_must_fit_each_unmeasured_commitment(self):
        live = keys()
        profiles = [profile(key) for key in live]
        for index, peak in ((0, owner.RESIDENT_COMMITMENT), (4, owner.JOB_COMMITMENT)):
            changed = list(profiles)
            changed[index] = profile(live[index], extra_peak=peak)
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, 'exceeds'):
                    validate_profiles(registry(changed), physical_uuid=UUID, live=live)

    def test_profile_margin_must_match_the_prespecified_heuristic(self):
        live = keys()
        changed = [profile(key) for key in live]
        changed[0] = replace(changed[0], margin_bytes=1, margin_bps=0)
        with self.assertRaisesRegex(ValueError, 'margin'):
            validate_profiles(registry(changed), physical_uuid=UUID, live=live)

    def test_profiles_with_different_sample_case_or_seed_are_rejected(self):
        live = keys()
        for override in ({'seed': 47}, {'case_sha256': 'f' * 64}):
            changed = [profile(key) for key in live]
            changed[4] = profile(live[4], **override)
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, 'sample case or seed'):
                    validate_profiles(registry(changed), physical_uuid=UUID, live=live)

    def test_public_gpu_launcher_is_unconditionally_disabled(self):
        self.assertFalse(owner.TRACE_OWNER_ENABLED)
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            owner.launch_supervised_trace()
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            owner._run_guarded_owner(None, None, backend=None, registry=None)

    def test_cpu_fake_owner_requires_ordered_phases_drain_and_attestation(self):
        live = keys()
        events = []
        session, backend = FakeSession(events), FakeBackend(live, events)
        with patch.object(owner, 'TRACE_OWNER_ENABLED', True), \
             patch.object(owner, '_QUARANTINED', Event()), \
             patch.object(owner, '_QUARANTINE_HOLDS', []), \
             patch.object(owner, '_LIFETIME_HOLDS', []):
            self.assertTrue(owner._run_guarded_owner(session, object(), backend=backend,
                                                     registry=registry(tuple(profile(k) for k in live))))
            self.assertFalse(owner._QUARANTINED.is_set())
            self.assertEqual(session.attestations, 1)
            self.assertEqual([x for x in events if x.startswith('run:')],
                             ['run:' + p.name for p in owner.PHASES])
            self.assertLess(events.index('binding'), events.index('run:load'))
            self.assertLess(events.index('dispose'), events.index('drained'))
            self.assertLess(events.index('drained'), events.index('attest'))
            self.assertEqual(events[-1], 'attest')

    def test_cpu_fake_owner_rejects_concurrent_and_repeated_runs(self):
        live = keys()
        events, errors = [], []
        session, backend = FakeSession(events), FakeBackend(live, events)
        backend.block_phase = 'baseline_cold'
        profiles = registry(tuple(profile(k) for k in live))
        def run():
            try:
                owner._run_guarded_owner(session, object(), backend=backend, registry=profiles)
            except BaseException as error:
                errors.append(error)
        with patch.object(owner, 'TRACE_OWNER_ENABLED', True), \
             patch.object(owner, '_QUARANTINED', Event()), \
             patch.object(owner, '_QUARANTINE_HOLDS', []), \
             patch.object(owner, '_LIFETIME_HOLDS', []):
            thread = Thread(target=run)
            thread.start()
            try:
                self.assertTrue(backend.block_entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, 'active or used'):
                    owner._run_guarded_owner(session, object(), backend=backend, registry=profiles)
            finally:
                backend.block_release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(session.attestations, 1)
            with self.assertRaisesRegex(RuntimeError, 'active or used'):
                owner._run_guarded_owner(session, object(), backend=backend, registry=profiles)

    def test_cpu_fake_owner_quarantines_oom_and_cancellation(self):
        live = keys()
        for failure in (MemoryError('fake OOM'), KeyboardInterrupt('fake cancellation')):
            events = []
            session, backend = FakeSession(events), FakeBackend(live, events)
            backend.failure_phase, backend.failure = 'direct', failure
            with self.subTest(failure=type(failure).__name__), \
                 patch.object(owner, 'TRACE_OWNER_ENABLED', True), \
                 patch.object(owner, '_OWNER_USED', False), \
                 patch.object(owner, '_QUARANTINED', Event()), \
                 patch.object(owner, '_QUARANTINE_HOLDS', []), \
                 patch.object(owner, '_LIFETIME_HOLDS', []):
                with self.assertRaises(type(failure)):
                    owner._run_guarded_owner(session, object(), backend=backend,
                                             registry=registry(tuple(profile(k) for k in live)))
                self.assertTrue(owner._QUARANTINED.is_set())
                self.assertTrue(owner._QUARANTINE_HOLDS)
                self.assertEqual(session.attestations, 0)
                self.assertNotIn('attest', events)
                with self.assertRaisesRegex(RuntimeError, 'quarantined'):
                    owner._run_guarded_owner(session, object(), backend=backend,
                                             registry=registry(tuple(profile(k) for k in live)))

    def test_cpu_fake_owner_refuses_uncertain_drain_or_allocator_cap(self):
        live = keys()
        for failure in ('undrained', 'wrong-cap', 'foreign-hook', 'leaked-hook', 'low-free'):
            events = []
            session, backend = FakeSession(events), FakeBackend(live, events)
            if failure == 'undrained':
                backend.drained = False
            elif failure == 'wrong-cap':
                backend.cap += 1
            elif failure == 'foreign-hook':
                backend.hooks_clear = False
            elif failure == 'leaked-hook':
                backend.hook_leak_phase = 'reference'
            else:
                backend.free = 2 * GIB
            with self.subTest(failure=failure), \
                 patch.object(owner, 'TRACE_OWNER_ENABLED', True), \
                 patch.object(owner, '_OWNER_USED', False), \
                 patch.object(owner, '_QUARANTINED', Event()), \
                 patch.object(owner, '_QUARANTINE_HOLDS', []), \
                 patch.object(owner, '_LIFETIME_HOLDS', []):
                with self.assertRaises(RuntimeError):
                    owner._run_guarded_owner(session, object(), backend=backend,
                                             registry=registry(tuple(profile(k) for k in live)))
                self.assertTrue(owner._QUARANTINED.is_set())
                self.assertEqual(session.attestations, 0)

    def test_cpu_fake_owner_rechecks_binding_after_phase_setup_before_work(self):
        live = keys()
        events = []
        session, backend = FakeSession(events), FakeBackend(live, events)
        backend.rebind_phase = 'baseline_cold'
        with patch.object(owner, 'TRACE_OWNER_ENABLED', True), \
             patch.object(owner, '_QUARANTINED', Event()), \
             patch.object(owner, '_QUARANTINE_HOLDS', []), \
             patch.object(owner, '_LIFETIME_HOLDS', []):
            with self.assertRaisesRegex(ValueError, 'runtime binding mismatch'):
                owner._run_guarded_owner(session, object(), backend=backend,
                                         registry=registry(tuple(profile(k) for k in live)))
            self.assertNotIn('run:baseline_cold', events)
            self.assertTrue(owner._QUARANTINED.is_set())
            self.assertEqual(session.attestations, 0)

    def test_cpu_fake_owner_rechecks_binding_before_each_synchronization(self):
        live = keys()
        for site in ('live-keys', 'configure-allocator', 'run:baseline_cold', 'end:id', 'dispose'):
            events = []
            session, backend = FakeSession(events), FakeBackend(live, events)
            backend.rebind_site = site
            with self.subTest(site=site), \
                 patch.object(owner, 'TRACE_OWNER_ENABLED', True), \
                 patch.object(owner, '_OWNER_USED', False), \
                 patch.object(owner, '_QUARANTINED', Event()), \
                 patch.object(owner, '_QUARANTINE_HOLDS', []), \
                 patch.object(owner, '_LIFETIME_HOLDS', []):
                with self.assertRaisesRegex(ValueError, 'runtime binding mismatch'):
                    owner._run_guarded_owner(session, object(), backend=backend,
                                             registry=registry(tuple(profile(k) for k in live)))
                last_site = events.index(site)
                self.assertNotIn('sync', events[last_site + 1:])
                if site == 'end:id':
                    self.assertNotIn('dispose', events)
                self.assertTrue(owner._QUARANTINED.is_set())
                self.assertEqual(session.attestations, 0)


if __name__ == '__main__':
    unittest.main()
