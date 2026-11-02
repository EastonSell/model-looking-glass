"""CPU-only tests for provisional five-token profile assembly."""
from dataclasses import replace
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass.agreement_calibration_owner import (assemble_profiles, collect_owned,
    _check_owner_admission, _release_claims, _guarded_cleanup, _check_private_run_dir,
    _check_frozen_preflight, _check_frozen_runtime, _workspace_binding)
from looking_glass.agreement_cuda_runner import CALIBRATION_KINDS, calibration_case_sha256, WARMUP_POLICY_SHA256
from looking_glass.memory_budget import CudaMemorySnapshot
from looking_glass.resource_profile import MemorySample
from test_agreement_cuda_runner import workload


class AgreementCalibrationOwnerTests(unittest.TestCase):
    def test_owner_gate_refuses_before_any_work_when_disabled(self):
        with patch('looking_glass.agreement_calibration_owner.AGREEMENT_CALIBRATION_OWNER_READY', False):
            with self.assertRaisesRegex(RuntimeError, 'not yet enabled'):
                collect_owned(None, None, repo_root='', cache_root='', run_dir='', run_id='')

    def test_private_directory_must_match_identity_and_be_outside_source_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            repo, cache, run = root / 'repo', root / 'cache', root / 'run'
            for path in (repo, cache, run):
                path.mkdir()
            stat = run.stat()
            expected = (stat.st_dev, stat.st_ino)
            self.assertEqual(_check_private_run_dir(run, expected, repo, cache), run)
            with self.assertRaises(ValueError):
                _check_private_run_dir(run, (expected[0], expected[1] + 1), repo, cache)
            with self.assertRaises(ValueError):
                _check_private_run_dir(repo, (repo.stat().st_dev, repo.stat().st_ino), repo, cache)

    def commitments(self):
        from looking_glass.memory_budget import MemoryBudget
        telemetry = lambda: CudaMemorySnapshot(6 * 1024 ** 3, 8 * 1024 ** 3, 0, 0)
        budget = MemoryBudget(telemetry, max_gpu_trace_bytes=2 * 1024 ** 2,
                              max_host_trace_bytes=2 * 1024 ** 2)
        resident = budget.acquire(2 * 1024 ** 3, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        job = budget.acquire(3 * 1024 ** 3 // 2, max_gpu_trace_bytes=2 * 1024 ** 2,
                             max_host_trace_bytes=2 * 1024 ** 2)
        job.reserve_trace('gpu', 2 * 1024 ** 2)
        job.reserve_trace('host', 2 * 1024 ** 2)
        return budget, resident, job

    def test_admission_rejects_total_changed_from_verified_runtime(self):
        budget, resident, job = self.commitments()
        snapshot = budget.snapshot()
        altered = replace(snapshot, telemetry=CudaMemorySnapshot(6 * 1024 ** 3, 9 * 1024 ** 3, 0, 0))
        with self.assertRaises(ValueError):
            _check_owner_admission(altered, budget, resident, job, 8 * 1024 ** 3)

    def test_owner_loss_between_releases_retains_remaining_claims(self):
        budget, resident, job = self.commitments()
        calls = []
        def guard(*args):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError('owner lost')
        with patch('looking_glass.agreement_calibration_owner._guard_owner', side_effect=guard):
            with self.assertRaisesRegex(RuntimeError, 'owner lost'):
                _release_claims(object(), object(), object(), budget, resident, job)
        self.assertEqual(job.trace_bytes('gpu'), 0)
        self.assertEqual(job.trace_bytes('host'), 2 * 1024 ** 2)
        self.assertFalse(job.released)
        self.assertFalse(resident.released)
        self.assertEqual(set(budget.active_leases()), {resident, job})

    def test_resident_mutation_during_job_retirement_keeps_resident_claim(self):
        from looking_glass.memory_budget import MemoryLease
        budget, resident, job = self.commitments()
        release = MemoryLease.release
        def retire_and_mutate(lease):
            release(lease)
            if lease is job:
                resident._estimated_incremental_bytes += 1
        with (patch('looking_glass.agreement_calibration_owner._guard_owner'),
              patch.object(MemoryLease, 'release', retire_and_mutate)):
            with self.assertRaisesRegex(ValueError, 'resident commitment changed'):
                _release_claims(object(), object(), object(), budget, resident, job)
        self.assertFalse(resident.released)
        self.assertEqual(budget.active_leases(), (resident,))

    def test_workspace_binding_rejects_another_valid_lock_namespace(self):
        from looking_glass.device_identity import resolve_device_identity
        from looking_glass.supervisor import _identity_spec
        uuid = 'GPU-00000000-0000-0000-0000-000000000001'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            repo = root / 'work' / 'repo'
            repo.mkdir(parents=True)
            other = root / 'other'
            other.mkdir()
            session = SimpleNamespace(identity=resolve_device_identity(root, uuid, physical_uuids=[uuid],
                                                                       environ={'CUDA_VISIBLE_DEVICES': uuid}))
            session._identity_spec = _identity_spec(session.identity)
            _workspace_binding(repo, session)
            session._identity_spec = dict(session._identity_spec, workspace_root=str(other))
            with self.assertRaises(ValueError):
                _workspace_binding(repo, session)
            session.identity = resolve_device_identity(other, uuid, physical_uuids=[uuid],
                                                       environ={'CUDA_VISIBLE_DEVICES': uuid})
            session._identity_spec = _identity_spec(session.identity)
            with self.assertRaises(ValueError):
                _workspace_binding(repo, session)

    def test_frozen_run_binds_source_assets_and_live_device(self):
        from looking_glass.pretrained import MODEL_ID, REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            repo, cache, run = root / 'repo', root / 'cache', root / 'run'
            for path in (repo, cache, run): path.mkdir()
            controller = {'schema': 'agreement-controller-source-v1', 'commit': 'a' * 40,
                          'files': {'looking_glass/test.py': 'b' * 64}}
            canonical = json.dumps(controller, sort_keys=True, separators=(',', ':')).encode()
            (run / 'controller-source.json').write_bytes(canonical + b'\n')
            source = SimpleNamespace(source_base=b'source-base',
                asset_manifest=json.dumps({'assets': [{'name': 'asset', 'size': 1, 'sha256': 'c' * 64}]}).encode())
            stat = run.stat()
            directory_id = (stat.st_dev, stat.st_ino)
            uuid = 'GPU-00000000-0000-0000-0000-000000000001'
            runtime = {name: '1.0' for name in ('python_version', 'torch_version', 'transformers_version',
                'numpy_version', 'huggingface_hub_version', 'safetensors_version', 'tokenizers_version',
                'cuda_runtime_version')}
            record = {'schema': 'agreement-calibration-run-freeze-v1', 'run_id': 'd' * 32,
                'source': {'commit': 'a' * 40, 'sha256': hashlib.sha256(canonical).hexdigest(),
                    'checkout': str(repo), 'source_base_sha256': hashlib.sha256(source.source_base).hexdigest()},
                'checkpoint': {'model_id': MODEL_ID, 'revision': REVISION, 'cache_root': str(cache),
                               'assets': json.loads(source.asset_manifest)['assets']},
                'runtime': runtime, 'device': {'uuid': uuid, 'name': 'test', 'total_bytes': 8 * 1024 ** 3,
                    'driver_version': '123.4'},
                'workload': {'seed': 20261028, 'target_input_ids': [464, 3797, 1474, 262, 9168],
                    'donor_input_ids': [464, 11875, 1474, 262, 9168], 'subject_index': 1,
                    'last_token_index': 4, 'batch_size': 1, 'sequence_length': 5},
                'directory': {'path': str(run), 'device': stat.st_dev, 'inode': stat.st_ino}}
            raw = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
            (run / 'run-manifest.json').write_bytes(raw)
            args = (raw, hashlib.sha256(raw).hexdigest(), run, 'd' * 32, repo, cache, directory_id,
                    source, SimpleNamespace(gpu_uuid=uuid))
            frozen = _check_frozen_preflight(*args)
            observed = dict(runtime, device_uuid=uuid, device_name='test', driver_version='123.4',
                            driver_version_provenance='controller-prelaunch-nvidia-smi',
                            subprocess_policy='python-audit-hook-denied',
                            device_total_bytes=8 * 1024 ** 3 - 1024 ** 2)
            _check_frozen_runtime(frozen, observed)
            for altered in (dict(observed, numpy_version='changed'),
                            dict(observed, device_uuid='GPU-other'),
                            dict(observed, driver_version_provenance='child'),
                            dict(observed, subprocess_policy='not-denied'),
                            dict(observed, device_total_bytes=8 * 1024 ** 3 - 1024 ** 2 - 1)):
                with self.assertRaises(ValueError): _check_frozen_runtime(frozen, altered)
            (run / 'controller-source.json').write_bytes(canonical.replace(b'test.py', b'tamp.py'))
            with self.assertRaisesRegex(ValueError, 'source inventory'):
                _check_frozen_preflight(*args)

    def test_claim_loss_during_cleanup_stops_before_next_cuda_call(self):
        from types import SimpleNamespace
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession
        budget, resident, job = self.commitments()
        session, lease = object.__new__(OwnerSession), object.__new__(DeviceLease)
        session._identity = SimpleNamespace(cuda_device_index=0)
        session._runtime_verified = True
        calls = []
        def sync(*args):
            calls.append('sync')
            resident.release()
        def empty(): calls.append('empty')
        cuda = SimpleNamespace(is_initialized=lambda: True, synchronize=sync, empty_cache=empty,
                               memory_allocated=lambda *_: 0, memory_reserved=lambda *_: 0)
        torch = SimpleNamespace(__version__='2.8.0+cu128', version=SimpleNamespace(cuda='12.8'),
            cuda=cuda, _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append('clear')))
        with (patch('looking_glass.agreement_calibration_owner._guard_owner'),
              patch.object(OwnerSession, 'check')):
            with self.assertRaises(ValueError):
                _guarded_cleanup(session, lease, torch, budget, resident, job)
        self.assertEqual(calls, ['sync'])
        self.assertFalse(job.released)

    def samples(self):
        key = workload()
        before = CudaMemorySnapshot(6 * 1024 ** 3, 8 * 1024 ** 3, 400 * 1024 ** 2, 500 * 1024 ** 2)
        after = CudaMemorySnapshot(6 * 1024 ** 3, 8 * 1024 ** 3, 410 * 1024 ** 2, 520 * 1024 ** 2)
        cold_key = replace(key, warmup_state='cold')
        cold = MemorySample(cold_key.fingerprint(), 'warmup', WARMUP_POLICY_SHA256, 20261028,
                            'completed', before, after, 450 * 1024 ** 2, 540 * 1024 ** 2)
        warm = tuple(MemorySample(key.fingerprint(), name, calibration_case_sha256(name), 20261028,
                                  'completed', before, after, 450 * 1024 ** 2, 540 * 1024 ** 2)
                     for name in CALIBRATION_KINDS)
        return key, cold, warm

    def test_exact_cold_and_ordered_warm_samples_build_distinct_profiles(self):
        key, cold, warm = self.samples()
        profiles, evidence = assemble_profiles(key, cold, warm, b'source', '2026-10-29T00:00:00Z')
        self.assertEqual([profile.key.warmup_state for profile in profiles], ['cold', 'warm'])
        self.assertEqual([len(profile.samples) for profile in profiles], [1, 13])
        self.assertEqual(len({profile.profile_id for profile in profiles}), 2)
        self.assertTrue(all(profile.provenance.source_manifest_sha256 == hashlib.sha256(b'source').hexdigest()
                            for profile in profiles))
        self.assertTrue(all(hashlib.sha256(raw).hexdigest() == profile.provenance.evidence_sha256
                            for profile, raw in zip(profiles, evidence)))

    def test_reordered_or_changed_case_never_becomes_a_profile(self):
        key, cold, warm = self.samples()
        for changed in (tuple(reversed(warm)),
                        (replace(warm[0], case_sha256='f' * 64), *warm[1:])):
            with self.assertRaises(ValueError):
                assemble_profiles(key, cold, changed, b'source', '2026-10-29T00:00:00Z')

    def exercise_owner(self, *, failure=None):
        import looking_glass.agreement_calibration_owner as owner
        import looking_glass.agreement_calibration_identity as identity
        import looking_glass.agreement_calibration_collector as collector
        import looking_glass.agreement_cuda_calibration as calibration
        import looking_glass.agreement_cuda_runner as adapter
        import looking_glass.agreement_journal as journal_module
        import looking_glass.pretrained as pretrained
        import looking_glass.trusted_service_profile as trusted
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession

        events = []
        self.events = events
        key = workload()
        numeric = {'float32_matmul_precision': 'highest', 'deterministic_warn_only': False,
                   'calling_thread_denormal_product_bits': 0x00400000, 'cpu_autocast': False}
        source = SimpleNamespace(asset_manifest=b'asset-manifest')
        state = {name: None for name in owner._CUSTODY}
        state['entered'] = False
        self.owner_state = state
        session, lease = object.__new__(OwnerSession), object.__new__(DeviceLease)
        session._identity = SimpleNamespace(gpu_uuid=key.device_uuid)
        session._runtime_verified = True

        class FakeModel:
            def to(self, **kwargs):
                events.append('model-copy')
                if failure == 'copy-claim-loss': state['resident'].release()
                return self
        class FakeProbe:
            def __init__(self, *args, **kwargs): self._bound_runner = None
            def snapshot(self):
                if self._bound_runner is not None: self._bound_runner._check_owner()
                total = 9 * 1024 ** 3 if failure == 'bad-total' else 8 * 1024 ** 3
                result = CudaMemorySnapshot(6 * 1024 ** 3, total, 0, 0)
                if self._bound_runner is not None: self._bound_runner._check_owner()
                return result
        class FakeJournal:
            def __init__(self, *args, **kwargs): self.retained_bytes, self._events = 0, 0
            def finish(self, *args): events.append('journal-finish')
        class FakeRunner:
            def __init__(self, model, journal, *, memory_probe, **kwargs):
                self.model, self.journal, memory_probe._bound_runner = model, journal, self
            def quarantine_after_uncertain_callback(self): events.append('quarantine')
            def dispose_after_success(self): events.append('runner-dispose'); self.model = None
        class FakeProfile:
            def __init__(self, label): self.profile_id, self.estimated_incremental_bytes = label, 1
            def to_json(self): return '{}'
        fake_torch = SimpleNamespace(float32='float32',
            random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: None)),
            __future__=SimpleNamespace(get_overwrite_module_params_on_conversion=lambda: False,
                                       get_swap_module_params_on_conversion=lambda: False),
            cuda=SimpleNamespace(synchronize=lambda device: events.append('model-sync')))
        def measure(runner, key, write):
            events.append('sample')
            if failure == 'sample':
                raise RuntimeError('injected callback failure')
            for index in range(84): write({'phase': 'fake', 'index': index})
            runner.journal._events = 14
            return object(), tuple(object() for _ in range(13))
        def cleanup(*args):
            events.append('cleanup')
            if failure == 'cleanup':
                raise RuntimeError('injected cleanup failure')
            return {'before': {'allocated_bytes': 32, 'reserved_bytes': 32},
                    'after': {'allocated_bytes': 0, 'reserved_bytes': 0}, 'api': 'fake'}
        def assemble(*args): return ((FakeProfile('cold'), FakeProfile('warm')), (b'cold', b'warm'))
        patches = ((owner, 'AGREEMENT_CALIBRATION_OWNER_READY', True), (owner, '_CUSTODY', state),
                   (owner, '_guard_owner', lambda *args: None),
                   (owner, '_workspace_binding', lambda *args: None),
                   (owner, '_check_frozen_preflight', lambda *args: {
                       'runtime': {}, 'device': {'driver_version': 'fake'}}),
                   (owner, '_check_frozen_runtime', lambda *args: None),
                   (identity, 'prepare_agreement_runtime', lambda *args, **kwargs: (fake_torch, key, b'source',
                       __import__('json').dumps(numeric).encode())),
                   (identity, '_numeric_observation', lambda *args: numeric),
                   (trusted, 'prepare_trusted_source', lambda *args: source),
                   (trusted, 'verify_trusted_cpu_model', lambda *args: True),
                   (trusted, 'verify_trusted_device_tie', lambda *args: True),
                   (adapter, 'GuardedAgreementMemoryProbe', FakeProbe),
                   (calibration, 'CudaAgreementCalibrator', FakeRunner),
                   (journal_module, 'AgreementJournal', FakeJournal),
                   (pretrained, 'load_distilgpt2', lambda **kwargs: SimpleNamespace(model=FakeModel())),
                   (collector, 'measure_calibration', measure),
                   (collector, '_SAMPLING_CUSTODY', [None] * 4),
                   (owner, '_guarded_cleanup', cleanup), (owner, 'assemble_profiles', assemble))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            cache, run = root / 'cache', root / 'run'
            cache.mkdir(); run.mkdir()
            for target, name, value in patches:
                stack.enter_context(patch.object(target, name, value))
            info = run.stat()
            return owner.collect_owned(session, lease, repo_root=Path(owner.__file__).resolve().parents[1],
                                       cache_root=cache, run_dir=run, run_id='a' * 32,
                                       run_dir_identity=(info.st_dev, info.st_ino))

    def test_fake_owner_success_disposes_model_and_releases_claims(self):
        result = self.exercise_owner()
        self.assertEqual(result['measurement']['sample_count'], 14)
        self.assertEqual(result['measurement']['stage_count'], 84)
        self.assertEqual(result['cleanup']['active_memory_leases'], 0)
        self.assertTrue(result['cleanup']['model_weakref_dead'])
        self.assertEqual(self.events[-2:], ['runner-dispose', 'cleanup'])
        self.assertFalse(self.owner_state['budget'].active_leases())

    def test_fake_owner_failed_callback_quarantines_and_retains_claims(self):
        with self.assertRaisesRegex(RuntimeError, 'injected callback failure'):
            self.exercise_owner(failure='sample')
        self.assertEqual(self.events[-2:], ['sample', 'quarantine'])
        self.assertNotIn('cleanup', self.events)
        self.assertEqual(len(self.owner_state['budget'].active_leases()), 2)
        self.assertIsNotNone(self.owner_state['model'])

    def test_fake_owner_bad_total_stops_before_copy(self):
        with self.assertRaises(ValueError):
            self.exercise_owner(failure='bad-total')
        self.assertNotIn('model-copy', self.events)
        self.assertNotIn('cleanup', self.events)
        self.assertEqual(len(self.owner_state['budget'].active_leases()), 2)

    def test_claim_loss_during_copy_stops_before_synchronization(self):
        with self.assertRaisesRegex(ValueError, 'commitments are not held'):
            self.exercise_owner(failure='copy-claim-loss')
        self.assertIn('model-copy', self.events)
        self.assertNotIn('model-sync', self.events)
        self.assertNotIn('cleanup', self.events)
        self.assertIsNotNone(self.owner_state['model'])
        self.assertEqual(self.owner_state['budget'].active_leases(), (self.owner_state['job'],))


if __name__ == '__main__':
    unittest.main()
