"""CPU-only tests of fixed worker output and resource-profile evidence."""
import dataclasses
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch import nn

from looking_glass.live_profile import _make_workload_keys
from looking_glass.memory_budget import CudaMemorySnapshot
from looking_glass.resident_worker import ResidentWorker
from looking_glass.resource_profile import MemorySample
from looking_glass.trusted_profile_collector import (CollectorError, assemble_profiles, run_fixed_forward)
from looking_glass.trusted_service_profile import (FORWARD_CASE_SHA256, LOAD_CASE, LOAD_CASE_SHA256,
                                                   TrustedServiceIdentity, TrustedServiceSource)


GIB = 1024 ** 3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def identity_and_samples():
    assets = {'schema': 'resource-bootstrap-assets-v1', 'assets': [
        {'name': 'model.safetensors', 'sha256': LOAD_CASE['checkpoint_sha256']},
        {'name': 'config.json', 'sha256': LOAD_CASE['config_sha256']},
        *({'name': f'asset-{i}', 'sha256': 'a' * 64} for i in range(5))]}
    source = canonical({'schema': 'trusted-service-source-v1', 'base': {'assets': assets},
                        'dependency_origins': []})
    runtime = {'schema': 'trusted-service-runtime-v1', 'device': 'cuda:0',
        'device_uuid': 'GPU-12345678-1234-5678-9abc-1234567890ab', 'device_name': 'Synthetic GPU',
        'device_total_bytes': 8 * GIB, 'compute_capability': [7, 5], 'python_version': '3.11.15',
        'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
        'cuda_runtime_version': '12.8', 'driver_version': '616.92', 'model_dtype': 'torch.float32',
        'compute_dtype': 'torch.float32', 'attention_backend': 'eager', 'model_mode': 'eval',
        'grad_mode': 'inference', 'autocast': 'disabled', 'use_cache': False, 'compiled': False,
        'execution_path': 'looking_glass.resident_worker.ResidentWorker.execute',
        'allocator_fraction': '0.20'}
    keys = _make_workload_keys(hashlib.sha256(source).hexdigest(), runtime)
    identity = TrustedServiceIdentity(keys, source, canonical(runtime), canonical(assets))
    before = CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0)
    after = CudaMemorySnapshot(6 * GIB - 1024, 8 * GIB, 1024, 2048)
    stages = ((keys[0], 17, LOAD_CASE_SHA256), (keys[1], 17, FORWARD_CASE_SHA256),
              (keys[2], 29, FORWARD_CASE_SHA256), (keys[2], 43, FORWARD_CASE_SHA256),
              (keys[2], 71, FORWARD_CASE_SHA256))
    samples = tuple(MemorySample(key.fingerprint(), f'sample-{i}', case, seed, 'completed',
                                 before, after, 1024, 2048)
                    for i, (key, seed, case) in enumerate(stages))
    return identity, samples


class FixedModel(nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.invalid = invalid
    def forward(self, input_ids):
        value = float('nan') if self.invalid else 0.0
        return {'logits': torch.full((1, 2, 50257), value, dtype=torch.float32)}


def worker(invalid=False):
    model = FixedModel(invalid).eval()
    return ResidentWorker('fixed-test', model, device='cpu', max_batch_size=1,
                          max_sequence_length=2, max_input_bytes=64 * 1024,
                          max_output_bytes=512 * 1024)


class TrustedProfileCollectorTests(unittest.TestCase):
    def setUp(self):
        import looking_glass.trusted_profile_collector as collector
        self.custody_patch = patch.object(collector, '_OWNER_CUSTODY', [None] * 9)
        self.claim_patch = patch.object(collector, '_OWNER_CLAIMED', False)
        self.custody_patch.start()
        self.claim_patch.start()
        self.addCleanup(self.custody_patch.stop)
        self.addCleanup(self.claim_patch.stop)

    def test_contract_can_be_checked_without_loading_gpu_dependencies(self):
        repo = Path(__file__).resolve().parents[1]
        program = ("import sys; sys.path.append(sys.argv[1]); "
                   "import looking_glass.trusted_profile_collector as collector; "
                   "assert collector.TRUSTED_SERVICE_COLLECTOR_CONTRACT == 'two-token-worker-profile-v1'; "
                   "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env['CUDA_VISIBLE_DEVICES'] = ''
        result = subprocess.run([sys.executable, '-B', '-S', '-P', '-c', program, str(repo)],
                                cwd=repo, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_spawn_contract_stops_child_before_owner_or_lease(self):
        import looking_glass.supervisor as supervisor
        import looking_glass.trusted_profile_collector as collector
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession
        with tempfile.TemporaryDirectory() as folder:
            config = {'run_dir': folder, 'run_id': 'a' * 32}
            with (patch.object(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT', None, create=True),
                  patch.object(OwnerSession, 'from_environment') as session,
                  patch.object(DeviceLease, 'acquire') as acquire):
                self.assertEqual(collector._owner_main(config), 1)
            session.assert_not_called()
            acquire.assert_not_called()
            self.assertEqual(json.loads((Path(folder) / 'failure.json').read_bytes())['error_type'],
                             'CollectorError')
            self.assertFalse((Path(folder) / 'owner.json').exists())

    def test_missing_spawn_contract_stops_parent_before_preflight_or_launch(self):
        import looking_glass.supervisor as supervisor
        import looking_glass.trusted_profile_collector as collector
        import looking_glass.trusted_service_profile as profile
        from looking_glass import gpu_oneshot
        def git_result(args, **kwargs):
            if args[1] == 'rev-parse': return SimpleNamespace(stdout='a' * 40, returncode=0)
            if args[1] == 'status': return SimpleNamespace(stdout='', returncode=0)
            if args[1] == 'symbolic-ref': return SimpleNamespace(stdout='', returncode=1)
            raise AssertionError('unexpected subprocess')
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder) / 'hf'
            cache = home / 'hub'
            cache.mkdir(parents=True)
            with (patch.object(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT', None, create=True),
                  patch.object(collector.subprocess, 'run', side_effect=git_result),
                  patch.object(profile, 'prepare_trusted_source') as source,
                  patch.object(gpu_oneshot, '_smi_snapshot') as inventory,
                  patch.object(supervisor, 'run_supervised') as launch):
                run_dir, terminal = collector.run_supervised_collector(
                    workspace_root=folder, cache_root=cache, hf_home=home, site_packages=folder,
                    environment_root=folder, private_parent=folder, trusted_path=folder)
            self.assertEqual(terminal['status'], 'failed_prelaunch')
            self.assertEqual(terminal['prelaunch_error_type'], 'CollectorError')
            self.assertTrue((run_dir / 'terminal.json').exists())
            source.assert_not_called()
            inventory.assert_not_called()
            launch.assert_not_called()

    def test_hf_home_must_contain_offline_cache(self):
        import looking_glass.trusted_profile_collector as collector
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            home, outside = root / 'hf', root / 'other'
            home.mkdir()
            outside.mkdir()
            inside = home / 'hub'
            inside.mkdir()
            self.assertEqual(collector._validated_hf_home(inside, home), home.resolve())
            for cache, selected_home in ((outside, home), (home, home), (inside, None),
                                         (inside, 'relative-home')):
                with self.subTest(cache=cache, home=selected_home), self.assertRaises(ValueError):
                    collector._validated_hf_home(cache, selected_home)

    def test_missing_hf_home_stops_child_before_owner_or_lease(self):
        import looking_glass.supervisor as supervisor
        import looking_glass.trusted_profile_collector as collector
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession
        with tempfile.TemporaryDirectory() as folder:
            config = {'run_dir': folder, 'run_id': 'a' * 32, 'cache_root': folder}
            with (patch.object(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT',
                               'gated-rooted-child-v1', create=True),
                  patch.dict(os.environ, {'HF_HOME': ''}),
                  patch.object(OwnerSession, 'from_environment') as session,
                  patch.object(DeviceLease, 'acquire') as acquire):
                self.assertEqual(collector._owner_main(config), 1)
            session.assert_not_called()
            acquire.assert_not_called()
            self.assertEqual(json.loads((Path(folder) / 'failure.json').read_bytes())['error_type'], 'ValueError')

    def test_preloaded_hub_with_stale_home_stops_before_owner(self):
        repo = str(Path(__file__).resolve().parents[1])
        site = sysconfig.get_path('purelib')
        script = textwrap.dedent('''
            import json, os, pathlib, sys
            from unittest.mock import patch
            repo, site, old, home, cache, run_dir, drop_parent = sys.argv[1:]
            sys.path.extend((repo, site))
            os.environ['HF_HOME'] = old
            import huggingface_hub.constants as constants
            assert pathlib.Path(constants.HF_HOME).resolve() == pathlib.Path(old).resolve()
            os.environ['HF_HOME'] = home
            if drop_parent == 'yes':
                sys.modules.pop('huggingface_hub', None)
            from looking_glass import trusted_profile_collector as collector
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            config = {'run_dir': run_dir, 'run_id': 'a' * 32, 'cache_root': cache}
            with patch.object(OwnerSession, 'from_environment') as session, \\
                 patch.object(DeviceLease, 'acquire') as acquire:
                assert collector._owner_main(config) == 1
                assert not session.called and not acquire.called
            assert json.loads((pathlib.Path(run_dir) / 'failure.json').read_bytes())['error_type'] == 'CollectorError'
        ''')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, home = root / 'old', root / 'home'
            cache = home / 'hub'
            old.mkdir()
            cache.mkdir(parents=True)
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
            env['CUDA_VISIBLE_DEVICES'] = ''
            for mode in ('no', 'yes'):
                run_dir = root / ('run-' + mode)
                run_dir.mkdir()
                result = subprocess.run([sys.executable, '-B', '-S', '-P', '-c', script, repo, site,
                                         str(old), str(home), str(cache), str(run_dir), mode],
                                        cwd=repo, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_preloaded_transformers_stops_before_owner(self):
        import types
        from looking_glass import trusted_profile_collector as collector
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            home, cache = root / 'hf', root / 'hf' / 'hub'
            cache.mkdir(parents=True)
            for name in ('transformers', 'transformers.models'):
                run_dir = root / name
                run_dir.mkdir()
                config = {'run_dir': str(run_dir), 'run_id': 'a' * 32, 'cache_root': str(cache)}
                with (patch.dict(sys.modules, {name: types.ModuleType(name)}),
                      patch.dict(os.environ, {'HF_HOME': str(home)}),
                      patch.object(OwnerSession, 'from_environment') as session,
                      patch.object(DeviceLease, 'acquire') as acquire):
                    self.assertEqual(collector._owner_main(config), 1)
                session.assert_not_called()
                acquire.assert_not_called()
                failure = json.loads((run_dir / 'failure.json').read_bytes())
                self.assertEqual(failure['error_type'], 'CollectorError')

    def test_assembled_profiles_bind_order_cases_source_and_margins(self):
        identity, samples = identity_and_samples()
        profiles, evidence = assemble_profiles(identity, samples, collected_utc='2026-10-29T00:00:00Z')
        self.assertEqual(tuple(tuple(sample.seed for sample in p.samples) for p in profiles),
                         ((17,), (17,), (29, 43, 71)))
        self.assertEqual(tuple(p.margin_bytes for p in profiles), (512 * 1024 ** 2, 256 * 1024 ** 2, 256 * 1024 ** 2))
        self.assertTrue(all(p.margin_bps == 2500 for p in profiles))
        self.assertTrue(all(hashlib.sha256(raw).hexdigest() == p.provenance.evidence_sha256
                            for raw, p in zip(evidence, profiles)))

    def test_wrong_order_case_or_duplicate_id_is_rejected(self):
        identity, samples = identity_and_samples()
        variants = (samples[:2] + (samples[3], samples[2], samples[4]),
                    samples[:2] + (dataclasses.replace(samples[2], case_sha256='f' * 64),) + samples[3:],
                    samples[:-1] + (dataclasses.replace(samples[-1], sample_id=samples[0].sample_id),))
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                assemble_profiles(identity, variant, collected_utc='2026-10-29T00:00:00Z')

    def test_real_cpu_worker_and_ipc_frame(self):
        owned = worker()
        try:
            result = run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
            self.assertEqual(result['output_shape'], [1, 2, 50257])
            self.assertEqual(result['output_tensor_bytes'], 402056)
            self.assertGreater(result['representative_frame_bytes'], 201031)
            self.assertLessEqual(result['representative_frame_bytes'], 4 * 1024 ** 2)
        finally:
            owned.close()

    def test_nonfinite_or_unprofiled_seed_produces_no_observation(self):
        owned = worker(invalid=True)
        try:
            with self.assertRaises(CollectorError): run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
            with self.assertRaises(ValueError): run_fixed_forward(owned, 13, cold=False, allow_cpu_test=True)
        finally:
            owned.close()

    def test_guarded_lifecycle_retains_both_claims_through_cleanup(self):
        result, events, budget = self.run_fake_owned(fail_forward=False, validate_candidate=True)
        self.assertEqual(result['seeds'], [17, 17, 29, 43, 71])
        self.assertEqual(events[:5], ['load', 'copy', 'forward-17', 'forward-29', 'forward-43'])
        self.assertEqual(events[-3:], ['forward-71', 'worker-close', 'workspace-clear'])
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])
        self.assertEqual(result['cleanup']['owner_commitments'], {
            'active_leases': 2, 'resident_bytes': 2 * GIB, 'forward_bytes': 512 * 1024 ** 2,
            'committed_incremental_bytes': 2 * GIB + 512 * 1024 ** 2, 'retained_until_exit': True})

    def test_failed_forward_retains_resident_and_job_commitments(self):
        with self.assertRaises(CollectorError) as captured:
            self.run_fake_owned(fail_forward=True)
        self.assertIn('unsafe', str(captured.exception))
        budget = self.captured_budget
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])
        self.assertNotIn('probe-sync', self.captured_events)

    def test_unsafe_completion_never_publishes_output_or_releases_claims(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='forward')
        self.assertNotIn('cold-17-output.json', self.captured_writes)
        self.assertNotIn('cold-17-sample.json', self.captured_writes)
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])

    def test_unsafe_before_sample_publication_retains_claims(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='postcheck')
        self.assertNotIn('cold-17-sample.json', self.captured_writes)
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])

    def test_unsafe_after_sample_retains_claims(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='sample')
        self.assertNotIn('warm-29-admission.json', self.captured_writes)
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])

    def test_lost_owner_after_sample_retains_claims(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='lost-owner')
        self.assertNotIn('warm-29-admission.json', self.captured_writes)
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])

    def test_released_forward_claim_prevents_next_observation(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='release-forward')
        self.assertNotIn('warm-29-admission.json', self.captured_writes)
        self.assertNotIn('worker-close', self.captured_events)
        self.assertEqual([lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()],
                         [2 * GIB])

    def test_released_resident_claim_prevents_next_observation(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='release-resident')
        self.assertNotIn('warm-29-admission.json', self.captured_writes)
        self.assertNotIn('worker-close', self.captured_events)
        self.assertEqual([lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()],
                         [512 * 1024 ** 2])

    def test_unsafe_after_candidate_write_never_attests(self):
        import looking_glass.trusted_profile_collector as collector
        import looking_glass.trusted_service_profile as trusted
        from looking_glass.device_lease import DeviceLease
        from looking_glass.memory_budget import MemoryBudget
        from looking_glass.supervisor import OwnerSession
        source = SimpleNamespace(source_base=b'source', asset_manifest=b'assets')
        budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        resident = budget.acquire(2 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        job = budget.acquire(512 * 1024 ** 2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        state = {'unsafe': False, 'worker': None}
        owner = SimpleNamespace(device_lease_path='fake', identity=SimpleNamespace(gpu_uuid='fake'), run_id='b' * 32)
        commitments = {'active_leases': 2, 'resident_bytes': 2 * GIB, 'forward_bytes': 512 * 1024 ** 2,
                       'committed_incremental_bytes': 2 * GIB + 512 * 1024 ** 2, 'retained_until_exit': True}
        def completed(*args):
            collector._OWNER_CUSTODY[3] = state
            collector._OWNER_CUSTODY[5:8] = [budget, resident, job]
            return {'cleanup': {'owner_commitments': commitments}}
        original_write = collector._write_json
        def write(path, value):
            original_write(path, value)
            if Path(path).name == 'candidate.json':
                state['unsafe'] = True
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config = {'run_dir': directory, 'run_id': 'a' * 32, 'repo_root': directory,
                      'cache_root': directory, 'source_base_sha256': hashlib.sha256(b'source').hexdigest(),
                      'asset_sha256': hashlib.sha256(b'assets').hexdigest()}
            for obj, name, value in ((collector, '_require_gated_supervisor', lambda: None),
                                     (collector, '_validated_hf_home', lambda *args: None),
                                     (collector, 'collect_owned', completed),
                                     (collector, '_bounded_read', lambda *args, **kwargs: b'fixture'),
                                     (collector, '_write_json', write),
                                     (trusted, 'prepare_trusted_source', lambda *args: source),
                                     (OwnerSession, 'from_environment', lambda: owner),
                                     (DeviceLease, 'acquire', lambda *args, **kwargs: object()),
                                     (OwnerSession, 'check', lambda *args: None)):
                stack.enter_context(patch.object(obj, name, new=value))
            attest = stack.enter_context(patch.object(OwnerSession, 'attest_clean'))
            self.assertEqual(collector._owner_main(config), 1)
            attest.assert_not_called()
            self.assertTrue((Path(directory) / 'failure.json').exists())
        self.assertEqual(set(budget.active_leases()), {resident, job})

    def test_interrupted_acquire_keeps_published_leases_in_rooted_budget(self):
        import looking_glass.trusted_profile_collector as collector
        with self.assertRaises(MemoryError):
            self.run_fake_owned(fail_forward=False, fail_acquire=True)
        self.assertIs(collector._OWNER_CUSTODY[5], self.captured_budget)
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])
        self.assertIsInstance(collector._OWNER_CUSTODY[8], MemoryError)

    def test_low_post_stage_headroom_retains_claim_and_partial_observation(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, fail_post=True)
        self.assertIn('warm-71-after-observed.json', self.captured_writes)
        self.assertIn('warm-71-postcheck.json', self.captured_writes)
        self.assertNotIn('warm-71-sample.json', self.captured_writes)
        self.assertNotIn('workspace-clear', self.captured_events)
        self.assertEqual(len(self.captured_budget.active_leases()), 2)

    def test_real_collector_never_retries_sync_after_quarantined_forward(self):
        import looking_glass.resource_profile as resource
        from looking_glass.device_lease import DeviceLease
        from looking_glass.memory_budget import MemoryBudget
        from looking_glass.supervisor import OwnerSession
        from looking_glass.trusted_profile_collector import _GuardedProbe
        live, _ = identity_and_samples()
        owner, lease = object.__new__(OwnerSession), object.__new__(DeviceLease)
        state = {'unsafe': False, 'operation_in_flight': False,
                 'worker': SimpleNamespace(quarantined=False)}
        events = []
        snapshot = CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0)
        class RawProbe:
            device = 'cuda:0'
            device_uuid = live.keys[0].device_uuid
            _cuda = SimpleNamespace(mem_get_info=lambda device: (6 * GIB, 8 * GIB),
                                    memory_allocated=lambda device: 0, memory_reserved=lambda device: 0)
            def synchronize(self): events.append('cuda-sync')
            def snapshot(self): return snapshot
            def reset_peak(self): events.append('reset')
            def peak_allocated_bytes(self): return 0
            def peak_reserved_bytes(self): return 0
        budget = MemoryBudget(lambda: snapshot, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        memory_lease = budget.acquire(512 * 1024 ** 2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        probe = _GuardedProbe(RawProbe(), state, owner, lease)
        def failed_operation():
            state['operation_in_flight'] = True
            state['worker'].quarantined = True
            state['unsafe'] = True
            raise RuntimeError('simulated unverified device drain')
        with (patch.object(OwnerSession, 'check', new=lambda *args: None),
              patch.object(DeviceLease, 'check', new=lambda *args: None),
              patch.object(resource, '_verify_device_binding', new=lambda *args: None),
              patch.object(resource, '_check_owner', new=lambda *args: None)):
            with self.assertRaises(resource.ProfilingSynchronizationError) as caught:
                resource.collect_sample(live.keys[1], probe, failed_operation, owner_session=owner,
                    device_identity=object(), device_lease=lease, memory_lease=memory_lease,
                    sample_id='failure-sample', case_sha256=FORWARD_CASE_SHA256, seed=17)
        self.assertEqual(events.count('cuda-sync'), 1)
        self.assertIsInstance(caught.exception.original_error, RuntimeError)
        self.assertIn(memory_lease, budget.active_leases())

    def test_telemetry_stops_when_quarantine_latches_during_first_query(self):
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession
        from looking_glass.trusted_profile_collector import _GuardedProbe
        events = []
        state = {'unsafe': False, 'operation_in_flight': False,
                 'worker': SimpleNamespace(quarantined=False)}
        def free(device):
            events.append('free')
            state['worker'].quarantined = True
            return 6 * GIB, 8 * GIB
        cuda = SimpleNamespace(mem_get_info=free,
            memory_allocated=lambda device: events.append('allocated'),
            memory_reserved=lambda device: events.append('reserved'))
        raw = SimpleNamespace(device='cuda:0', device_uuid='GPU-fake', _cuda=cuda)
        owner, lease = object.__new__(OwnerSession), object.__new__(DeviceLease)
        with patch.object(OwnerSession, 'check', new=lambda *args: None):
            with self.assertRaises(CollectorError):
                _GuardedProbe(raw, state, owner, lease).snapshot()
        self.assertEqual(events, ['free'])

    def test_direct_module_entrypoint_reaches_collector_before_any_gpu_import(self):
        repo = Path(__file__).resolve().parents[1]
        site = sysconfig.get_path('purelib')
        script = textwrap.dedent('''
            import atexit, hashlib, json, runpy, sys
            from pathlib import Path
            from types import SimpleNamespace
            repo, site, folder = sys.argv[1:4]
            atexit.register(lambda: (Path(folder) / 'atexit-ran').write_text('unexpected'))
            sys.path.extend((repo, site))
            import looking_glass.trusted_service_launch as launch
            import looking_glass.trusted_service_profile as trusted
            import looking_glass.supervisor as supervisor
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            supervisor.SUPERVISOR_LAUNCH_CUSTODY_CONTRACT = 'gated-rooted-child-v1'
            owner = object.__new__(OwnerSession)
            owner._identity = SimpleNamespace(gpu_uuid='GPU-12345678-1234-5678-9abc-1234567890ab')
            owner.device_lease_path = 'unused'
            owner.run_id = 'b' * 32
            lease = object.__new__(DeviceLease)
            OwnerSession.from_environment = classmethod(lambda cls: owner)
            OwnerSession.check = lambda *args: None
            DeviceLease.acquire = classmethod(lambda cls, *args, **kwargs: lease)
            source = SimpleNamespace(source_base=b'base', asset_manifest=b'assets')
            trusted.prepare_trusted_source = lambda *args: source
            config = {'schema': 'trusted-profile-owner-config-v1', 'repo_root': repo,
                      'cache_root': str(Path(folder) / 'hf' / 'hub'), 'run_dir': folder, 'run_id': 'a' * 32,
                      'source_base_sha256': hashlib.sha256(b'base').hexdigest(),
                      'asset_sha256': hashlib.sha256(b'assets').hexdigest()}
            launch.read_trusted_config = lambda *args, **kwargs: json.dumps(config).encode()
            path = str(Path(folder) / 'config.json')
            sys.argv = ['looking_glass.trusted_profile_collector', '--config', path,
                        '--config-id', '1:1', '--config-sha256', 'a' * 64]
            assert not any(name == 'huggingface_hub' or name.startswith('huggingface_hub.') for name in sys.modules)
            runpy.run_module('looking_glass.trusted_profile_collector', run_name='__main__', alter_sys=True)
        ''')
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / 'pycache'
            prefix.mkdir()
            (Path(folder) / 'hf' / 'hub').mkdir(parents=True)
            (Path(folder) / 'config.json').write_text('{}')
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
            env['CUDA_VISIBLE_DEVICES'] = ''
            env['HF_HOME'] = str(Path(folder) / 'hf')
            completed = subprocess.run([sys._base_executable, '-B', '-S', '-P', '-X',
                'pycache_prefix=' + str(prefix), '-c', script, str(repo), site, folder],
                env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            failure = json.loads((Path(folder) / 'failure.json').read_text())
            self.assertEqual(failure['error_type'], 'TypeError')
            self.assertIn('collect_owned', failure['private_traceback'])
            self.assertNotIn('NameError', failure['private_traceback'])
            self.assertFalse((Path(folder) / 'atexit-ran').exists())

    def test_real_launcher_and_supervisor_quarantine_missing_checkpoint(self):
        # The fake UUID and empty cache exercise real subprocess ownership,
        # but intentionally stop at CPU-only asset preflight, before Torch import.
        from looking_glass.device_identity import resolve_device_identity
        from looking_glass import supervisor
        from looking_glass.supervisor import run_supervised
        from looking_glass.trusted_service_launch import prepare_trusted_launch
        try:
            base_python = Path(sys._base_executable).resolve(strict=True)
            if not base_python.is_file():
                raise OSError('base Python is not an ordinary file')
        except OSError:
            self.skipTest('the base Python executable is inaccessible to direct subprocess launch')
        repo = Path(__file__).resolve().parents[1]
        site = Path(sysconfig.get_path('purelib')).resolve()
        environment = Path(sys.prefix).resolve()
        gpu_uuid = 'GPU-12345678-1234-5678-9abc-1234567890ab'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            workspace, private, home = (root / name for name in ('workspace', 'private', 'hf'))
            for path in (workspace, private, home):
                path.mkdir()
            cache = home / 'hub'
            cache.mkdir()
            run_dir = private / 'run'
            run_dir.mkdir()
            config = {'schema': 'trusted-profile-owner-config-v1', 'repo_root': str(repo),
                      'cache_root': str(cache), 'run_dir': str(run_dir), 'run_id': 'a' * 32,
                      'source_base_sha256': '0' * 64, 'asset_sha256': '0' * 64}
            config_path = run_dir / 'config.json'
            config_path.write_bytes(canonical(config))
            launch = prepare_trusted_launch(repo_root=str(repo), site_packages=str(site),
                environment_root=str(environment), private_parent=str(private), gpu_uuid=gpu_uuid,
                module='looking_glass.trusted_profile_collector', config_path=str(config_path),
                trusted_path=str(base_python.parent), hf_home=str(home))
            identity = resolve_device_identity(workspace, gpu_uuid, physical_uuids=[gpu_uuid],
                                               environ=launch.environment)
            launch.verify_ready()
            result = run_supervised(launch.command, device_lease_path=identity.lock_path,
                marker_path=identity.marker_path, identity=identity, timeout=30, lease_timeout=5,
                cwd=repo, env=launch.environment, inherit_environment=False)
            self.assertEqual(result.exit_code, 1)
            self.assertFalse(result.clean)
            self.assertFalse(result.timed_out)
            self.assertTrue(identity.marker_path.exists())
            self.assertFalse(identity.attestation_path.exists())
            failure = json.loads((run_dir / 'failure.json').read_bytes())
            if (getattr(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT', None)
                    == 'gated-rooted-child-v1'):
                owner = json.loads((run_dir / 'owner.json').read_bytes())
                self.assertEqual(owner['supervisor_run_id'], result.run_id)
                self.assertEqual(owner['pid'], result.pid)
                self.assertEqual(owner['physical_gpu_uuid'], gpu_uuid)
                self.assertIn('collect_offline_assets', failure['private_traceback'])
            else:
                self.assertEqual(failure['error_type'], 'CollectorError')
                self.assertFalse((run_dir / 'owner.json').exists())
            self.assertFalse((run_dir / 'candidate.json').exists())

    def run_fake_owned(self, *, fail_forward, fail_acquire=False, fail_post=False,
                       validate_candidate=False, latch_at=None):
        import looking_glass.trusted_profile_collector as collector
        import looking_glass.trusted_service_profile as trusted
        import looking_glass.memory_budget as memory
        import looking_glass.resource_profile as resource
        import looking_glass.resident_worker as resident
        import looking_glass.pretrained as pretrained
        import looking_glass.live_profile as live_profile
        import looking_glass.cuda_cleanup as cleanup
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession

        live, _ = identity_and_samples()
        source = TrustedServiceSource('fake-repo', 'fake-cache', b'{}', live.asset_manifest, ())
        owner = object.__new__(OwnerSession)
        owner._identity = SimpleNamespace(gpu_uuid=live.keys[0].device_uuid)
        lease = object.__new__(DeviceLease)
        events = []
        self.captured_events = events
        self.captured_writes = []

        class FakeModel:
            training = False
            config = SimpleNamespace(use_cache=False, _attn_implementation='eager', vocab_size=50257, n_positions=1024)
            def to(self, **kwargs): events.append('copy'); return self
            def parameters(self):
                return [SimpleNamespace(device=SimpleNamespace(type='cuda', index=0), dtype='float32')]
        class FakeWorker:
            quarantined = False
            poisoned = False
            def __init__(self, model_id, model, **kwargs): self.model = model
            def close(self): events.append('worker-close'); self.model = None
        class FakeProbe:
            def __init__(self, *args, **kwargs):
                self.device, self.device_uuid = kwargs['device'], kwargs['device_uuid']
                self._cuda = SimpleNamespace(mem_get_info=lambda device: (6 * GIB, 8 * GIB),
                                             memory_allocated=lambda device: 0, memory_reserved=lambda device: 0)
            def snapshot(self): return CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0)
            def synchronize(self): events.append('probe-sync')
        fake_cuda = SimpleNamespace(device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(uuid='fake'),
            synchronize=lambda index: None)
        fake_torch = SimpleNamespace(cuda=fake_cuda, _C=SimpleNamespace(_CUuuid=object), float32='float32',
            random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: None)),
            __future__=SimpleNamespace(get_overwrite_module_params_on_conversion=lambda: False,
                                       get_swap_module_params_on_conversion=lambda: False))
        def load(**kwargs):
            events.append('load')
            return SimpleNamespace(model=FakeModel())
        def budget_factory(*args, **kwargs):
            self.captured_budget = memory.MemoryBudget(*args, **kwargs)
            return self.captured_budget
        def sample(key, probe, operation, **kwargs):
            self.assertIn(kwargs['memory_lease'], self.captured_budget.active_leases())
            try:
                self.assertIsNone(operation())
            except BaseException:
                probe.synchronize()  # Mirrors collect_sample's failure boundary.
                raise
            snap = CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0)
            return MemorySample(key.fingerprint(), kwargs['sample_id'], kwargs['case_sha256'], kwargs['seed'],
                                'completed', snap, snap, 0, 0)
        def forward(owned, seed, **kwargs):
            events.append('forward-' + str(seed))
            if fail_forward:
                raise CollectorError('fake forward failure')
            if latch_at == 'forward' and seed == 17:
                collector._OWNER_CUSTODY[3]['unsafe'] = True
            return {'seed': seed, 'cold': kwargs['cold'], 'case_sha256': FORWARD_CASE_SHA256,
                    'finite': True, 'output_shape': [1, 2, 50257], 'output_dtype': 'torch.float32',
                    'output_tensor_bytes': 402056, 'representative_frame_bytes': 500000}
        def clear(**kwargs):
            events.append('workspace-clear')
            return {'before': {'allocated_bytes': 32, 'reserved_bytes': 32},
                    'after': {'allocated_bytes': 0, 'reserved_bytes': 0}, 'api': 'fake'}
        owner_alive = [True]
        def check_owner(*args):
            if not owner_alive[0]:
                raise CollectorError('owner lost')
        patches = ((collector, '_import_uninitialized_torch', lambda: fake_torch),
                   (trusted, 'build_trusted_service_identity', lambda *args: live),
                   (trusted, 'prepare_trusted_source', lambda *args: source),
                   (trusted, 'verify_trusted_cpu_model', lambda *args: True),
                   (trusted, 'verify_trusted_device_tie', lambda *args: True),
                   (memory, 'MemoryBudget', budget_factory),
                   (resource, 'TorchMemoryProbe', FakeProbe), (resource, 'collect_sample', sample),
                   (resident, 'ResidentWorker', FakeWorker), (pretrained, 'load_distilgpt2', load),
                   (live_profile, 'decode_cuda_uuid', lambda *args: live.keys[0].device_uuid),
                   (cleanup, 'clear_cublas_after_drain', clear),
                   (collector, 'run_fixed_forward', forward),
                   (OwnerSession, 'check', check_owner),
                   (OwnerSession, 'verify_runtime_binding', lambda *args, **kwargs: None))
        # Keep a direct constructor before patching the module attribute.
        original_budget = memory.MemoryBudget
        class FakeBudget(original_budget):
            def snapshot(self):
                value = super().snapshot()
                if fail_post and events and events[-1] == 'forward-71':
                    return dataclasses.replace(value, telemetry=CudaMemorySnapshot(GIB, 8 * GIB, 0, 0))
                return value
            def acquire(self, estimate, **kwargs):
                result = super().acquire(estimate, **kwargs)
                if fail_acquire and estimate == 512 * 1024 ** 2:
                    raise MemoryError('interrupted after publication')
                return result
        def budget_factory(*args, **kwargs):
            self.captured_budget = FakeBudget(*args, **kwargs)
            return self.captured_budget
        patches = tuple((obj, name, budget_factory if obj is memory and name == 'MemoryBudget' else value)
                        for obj, name, value in patches)
        original_write = collector._write_json
        def record_write(path, value):
            self.captured_writes.append(Path(path).name)
            result = original_write(path, value)
            if latch_at == 'postcheck' and Path(path).name == 'cold-17-postcheck.json':
                collector._OWNER_CUSTODY[3]['unsafe'] = True
            elif latch_at == 'sample' and Path(path).name == 'cold-17-sample.json':
                collector._OWNER_CUSTODY[3]['unsafe'] = True
            elif latch_at == 'lost-owner' and Path(path).name == 'cold-17-sample.json':
                owner_alive[0] = False
            elif latch_at == 'release-forward' and Path(path).name == 'cold-17-sample.json':
                collector._OWNER_CUSTODY[7].release()
            elif latch_at == 'release-resident' and Path(path).name == 'cold-17-sample.json':
                collector._OWNER_CUSTODY[6].release()
            return result
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for obj, name, value in patches:
                stack.enter_context(patch.object(obj, name, new=value))
            stack.enter_context(patch.object(collector, '_write_json', new=record_write))
            run_id = 'a' * 32
            supervisor = SimpleNamespace(run_id='b' * 32, pid=123, clean=True, exit_code=0, timed_out=False)
            collector._write_json(Path(directory) / 'owner.json', {'schema': 'trusted-profile-owner-v1',
                'run_id': run_id, 'supervisor_run_id': supervisor.run_id, 'pid': supervisor.pid,
                'physical_gpu_uuid': live.keys[0].device_uuid})
            result = collector.collect_owned(source, owner, lease, Path(directory), run_id)
            self.assertTrue((Path(directory) / 'warm-71-sample.json').exists())
            self.assertTrue((Path(directory) / 'cleanup.json').exists())
            if validate_candidate:
                files = {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
                         for name in collector._ARTIFACT_NAMES}
                candidate = {'schema': 'trusted-profile-candidate-v1', 'run_id': run_id,
                             'supervisor_run_id': supervisor.run_id, 'pid': supervisor.pid,
                             'summary': result, 'artifact_sha256': files}
                collector._write_json(Path(directory) / 'candidate.json', candidate)
                self.assertEqual(collector._validate_candidate(Path(directory), run_id, supervisor), result)
                # Tampering with a stage prevents promotion even after the owner exited.
                output_path = Path(directory) / 'cold-17-output.json'
                original_output = output_path.read_bytes()
                output_path.write_text('{}')
                with self.assertRaises(CollectorError):
                    collector._validate_candidate(Path(directory), run_id, supervisor)
                output_path.write_bytes(original_output)
                # Rehashing malformed JSON must not turn booleans into numbers
                # (or vice versa) through Python's equality semantics.
                for name, field, replacement in (
                        ('cold-17-output.json', 'output_shape', [True, 2, 50257]),
                        ('load-17-checks.json', 'cpu_model_verified', 1),
                        ('cleanup.json', 'owner_commitments', {'active_leases': 1}),
                        ('cold-17-after-observed.json', 'process_allocated_bytes', False)):
                    path = Path(directory) / name
                    original = path.read_bytes()
                    value = json.loads(original)
                    value[field] = replacement
                    path.write_bytes(canonical(value))
                    candidate['artifact_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    (Path(directory) / 'candidate.json').write_bytes(canonical(candidate))
                    with self.assertRaises(CollectorError, msg=name):
                        collector._validate_candidate(Path(directory), run_id, supervisor)
                    path.write_bytes(original)
                    candidate['artifact_sha256'][name] = hashlib.sha256(original).hexdigest()
                    (Path(directory) / 'candidate.json').write_bytes(canonical(candidate))
                self.assertEqual(collector._validate_candidate(Path(directory), run_id, supervisor), result)
                # Even a consistently rehashed but structurally invalid admission fails.
                admission_path = Path(directory) / 'cold-17-admission.json'
                admission = json.loads(admission_path.read_text())
                admission['budget']['telemetry']['process_allocated_bytes'] = 1
                admission['budget']['telemetry']['process_reserved_bytes'] = 0
                admission_path.write_bytes(canonical(admission))
                candidate['artifact_sha256'][admission_path.name] = hashlib.sha256(admission_path.read_bytes()).hexdigest()
                (Path(directory) / 'candidate.json').write_bytes(canonical(candidate))
                with self.assertRaises(CollectorError):
                    collector._validate_candidate(Path(directory), run_id, supervisor)
            return result, events, self.captured_budget


if __name__ == '__main__':
    unittest.main()
