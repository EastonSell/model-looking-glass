"""CPU-only tests of fixed worker output and resource-profile evidence."""
import dataclasses
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

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
    from looking_glass.custody_service_profile import EXECUTION_PATH, RUNTIME_SCHEMA, _PROTOCOL_BYTES
    assets = {'schema': 'resource-bootstrap-assets-v1', 'assets': [
        {'name': 'model.safetensors', 'sha256': LOAD_CASE['checkpoint_sha256']},
        {'name': 'config.json', 'sha256': LOAD_CASE['config_sha256']},
        *({'name': f'asset-{i}', 'sha256': 'a' * 64} for i in range(5))]}
    source = canonical({'schema': 'trusted-service-source-v1', 'base': {'assets': assets},
                        'dependency_origins': []})
    runtime = {'schema': RUNTIME_SCHEMA, 'device': 'cuda:0',
        'device_uuid': 'GPU-12345678-1234-5678-9abc-1234567890ab', 'device_name': 'Synthetic GPU',
        'device_total_bytes': 8 * GIB, 'compute_capability': [7, 5], 'python_version': '3.11.15',
        'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
        'cuda_runtime_version': '12.8', 'driver_version': '616.92', 'model_dtype': 'torch.float32',
        'compute_dtype': 'torch.float32', 'attention_backend': 'eager', 'model_mode': 'eval',
        'grad_mode': 'inference', 'autocast': 'disabled', 'use_cache': False, 'compiled': False,
        'execution_path': EXECUTION_PATH, 'custody_execution': json.loads(_PROTOCOL_BYTES),
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


def custody_worker(invalid=False):
    from looking_glass.budgeted_worker import WorkloadProfile
    from looking_glass.memory_budget import MemoryBudget
    from looking_glass.owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
    resident = ResidentWorker('a' * 64, FixedModel(invalid).eval(), device='cpu', max_batch_size=1,
                              max_sequence_length=2, max_input_bytes=64 * 1024,
                              max_output_bytes=512 * 1024)
    budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0),
                          max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    load = budget.acquire(2 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    workspace = budget.acquire(512 * 1024 ** 2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    cold = WorkloadProfile('provisional-cold', resident.model_id, 'cpu', (1, 2),
                           (('input_ids', 'torch.int64'),), 512 * 1024, 512 * 1024 ** 2)
    warm = dataclasses.replace(cold, profile_id='provisional-warm')
    authority = OwnerCompletionAuthority(device_guard=lambda: None)
    return LifetimeWorkspaceWorker(resident, budget, load, workspace,
                                   profiles=(cold, warm), authority=authority)


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
                   "assert collector.TRUSTED_SERVICE_COLLECTOR_CONTRACT == 'two-token-custody-worker-profile-v1'; "
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
                  patch.object(collector, '_workspace_root', return_value=Path(folder)),
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

    def test_alternate_workspace_stops_parent_before_inventory_or_launch(self):
        import looking_glass.trusted_profile_collector as collector
        from looking_glass import gpu_oneshot, supervisor
        from looking_glass.device_identity import resolve_device_identity
        with tempfile.TemporaryDirectory() as folder:
            wrong = Path(folder).resolve()
            with (patch.object(gpu_oneshot, '_smi_snapshot') as inventory,
                  patch.object(supervisor, 'run_supervised') as launch,
                  patch('looking_glass.device_identity.resolve_device_identity', wraps=resolve_device_identity) as resolve):
                with self.assertRaisesRegex(CollectorError, 'workspace root'):
                    collector.run_supervised_collector(workspace_root=wrong, cache_root=wrong, hf_home=wrong,
                        site_packages=wrong, environment_root=wrong, private_parent=wrong, trusted_path=str(wrong))
            inventory.assert_not_called()
            launch.assert_not_called()
            resolve.assert_not_called()

    def test_nested_work_ancestors_cannot_select_an_inner_lease_namespace(self):
        import looking_glass.trusted_profile_collector as collector
        with tempfile.TemporaryDirectory() as folder:
            outer = Path(folder).resolve()
            inner = outer / 'work' / 'teams' / 'owner'
            checkout = inner / 'Work' / 'checkout'
            checkout.mkdir(parents=True)
            for requested in (outer, inner):
                with self.subTest(requested=requested):
                    with self.assertRaisesRegex(CollectorError, 'ambiguous'):
                        collector._workspace_root(checkout, requested)

    def test_alternate_workspace_stops_direct_collection_before_torch_import(self):
        import looking_glass.trusted_profile_collector as collector
        from looking_glass.device_identity import resolve_device_identity
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession, _identity_spec
        gpu_uuid = 'GPU-12345678-1234-5678-9abc-1234567890ab'
        with tempfile.TemporaryDirectory() as folder:
            identity = resolve_device_identity(folder, gpu_uuid, physical_uuids=[gpu_uuid],
                environ={'CUDA_VISIBLE_DEVICES': gpu_uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
            owner, lease = object.__new__(OwnerSession), object.__new__(DeviceLease)
            owner._identity = identity
            owner._identity_spec = _identity_spec(identity)
            source = TrustedServiceSource('fake-repo', 'fake-cache', b'{}', b'{}', ())
            state = {'unsafe': False, 'worker': None, 'authority': None, 'operation_in_flight': False}
            with (patch.object(OwnerSession, 'check'),
                  patch.object(collector, '_import_uninitialized_torch') as import_torch):
                with self.assertRaisesRegex(CollectorError, 'workspace root'):
                    collector.collect_owned(source, owner, lease, folder, 'a' * 32)
                with self.assertRaisesRegex(CollectorError, 'workspace root'):
                    collector._collect_owned_inner(source, owner, lease, folder, 'a' * 32, state)
            import_torch.assert_not_called()

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
        # Exercise the missing-home guard in a fresh owner-like interpreter.
        # Earlier suite tests may have loaded Hub/Transformers, in which case
        # the production preload guard must correctly take precedence. The
        # failed owner's retained traceback belongs to that child as well.
        driver = textwrap.dedent('''
            import json, os, sys
            from pathlib import Path
            from unittest.mock import patch
            sys.path.insert(0, sys.argv[1])
            import looking_glass.supervisor as supervisor
            import looking_glass.trusted_profile_collector as collector
            from looking_glass.device_lease import DeviceLease
            from looking_glass.supervisor import OwnerSession
            root = Path(sys.argv[2])
            config = {'run_dir': str(root), 'run_id': 'a' * 32, 'cache_root': str(root)}
            with (patch.object(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT',
                               'gated-rooted-child-v1', create=True),
                  patch.dict(os.environ, {'HF_HOME': ''}),
                  patch.object(OwnerSession, 'from_environment') as session,
                  patch.object(DeviceLease, 'acquire') as acquire):
                status = collector._owner_main(config)
            if status != 1:
                raise AssertionError('missing HF_HOME did not fail')
            session.assert_not_called()
            acquire.assert_not_called()
            failure = json.loads((root / 'failure.json').read_bytes())
            if failure.get('error_type') != 'ValueError':
                raise AssertionError('missing HF_HOME did not reach its validation guard')
        ''')
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys._base_executable, '-I', '-S', '-B', '-c', driver,
                 str(Path(__file__).resolve().parents[1]), folder],
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_direct_worker_identity_cannot_be_promoted_as_custody_profile(self):
        identity, samples = identity_and_samples()
        runtime = json.loads(identity.runtime_manifest)
        runtime['schema'] = 'trusted-service-runtime-v1'
        runtime['execution_path'] = 'looking_glass.resident_worker.ResidentWorker.execute'
        del runtime['custody_execution']
        keys = _make_workload_keys(hashlib.sha256(identity.source_manifest).hexdigest(), runtime)
        direct = TrustedServiceIdentity(keys, identity.source_manifest, canonical(runtime), identity.asset_manifest)
        stages = (keys[0], keys[1], keys[2], keys[2], keys[2])
        direct_samples = tuple(dataclasses.replace(sample, workload_sha256=key.fingerprint())
                               for sample, key in zip(samples, stages))
        with self.assertRaises(ValueError):
            assemble_profiles(direct, direct_samples, collected_utc='2026-10-29T00:00:00Z')

    def test_real_cpu_worker_and_ipc_frame(self):
        owned = custody_worker()
        try:
            result = run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
            self.assertEqual(result['output_shape'], [1, 2, 50257])
            self.assertEqual(result['output_tensor_bytes'], 402056)
            self.assertGreater(result['representative_frame_bytes'], 201031)
            self.assertLessEqual(result['representative_frame_bytes'], 4 * 1024 ** 2)
        finally:
            owned.close()

    def test_actual_custody_worker_cold_and_warm_are_transferred(self):
        from looking_glass.owner_completion import LifetimeWorkspaceWorker
        import looking_glass.trusted_profile_collector as collector
        owned = custody_worker()
        events = []
        execute, finish, transfer, stage = (LifetimeWorkspaceWorker.execute,
                                            LifetimeWorkspaceWorker._finish_cold,
                                            LifetimeWorkspaceWorker._transfer_result,
                                            LifetimeWorkspaceWorker._stage_transfer_locked)
        execute_owned, fork_owned = ResidentWorker._execute_with_custody, ResidentWorker._fork_owned_rng
        def track_execute(worker, request, *args, **kwargs):
            events.append(('execute', request.seed, request.inputs['input_ids'].tolist()))
            return execute(worker, request, *args, **kwargs)
        def track_finish(worker, result):
            events.append(('finish-cold', result.request_id))
            return finish(worker, result)
        def track_transfer(worker, result):
            events.append(('transfer', result.request_id))
            return transfer(worker, result)
        def track_stage(worker, result):
            events.append(('stage', result.request_id))
            return stage(worker, result)
        def track_owned(resident, request, cancel_event, custody):
            self.assertIs(resident._owner_completion, owned._authority)
            events.append(('resident-custody', request.seed))
            return execute_owned(resident, request, cancel_event, custody)
        def track_rng(resident, devices):
            self.assertIs(resident._owner_completion, owned._authority)
            events.append(('owned-rng', tuple(devices)))
            return fork_owned(resident, devices)
        with (patch.object(LifetimeWorkspaceWorker, 'execute', track_execute),
              patch.object(LifetimeWorkspaceWorker, '_finish_cold', track_finish),
              patch.object(LifetimeWorkspaceWorker, '_transfer_result', track_transfer),
              patch.object(LifetimeWorkspaceWorker, '_stage_transfer_locked', track_stage),
              patch.object(ResidentWorker, '_execute_with_custody', track_owned),
              patch.object(ResidentWorker, '_fork_owned_rng', track_rng),
              patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000)):
            for seed, cold in ((17, True), (29, False), (43, False), (71, False)):
                observation = run_fixed_forward(owned, seed, cold=cold, allow_cpu_test=True)
                self.assertTrue(observation['finite'])
                self.assertIsNone(owned._custody.result)
                self.assertIsNone(owned._custody.request)
                self.assertIsNone(owned._custody.logits)
        self.assertEqual([event[1] for event in events if event[0] == 'execute'], [17, 29, 43, 71])
        self.assertTrue(all(event[2] == [[15496, 995]] for event in events if event[0] == 'execute'))
        self.assertEqual(sum(event[0] == 'finish-cold' for event in events), 1)
        self.assertEqual(sum(event[0] == 'transfer' for event in events), 3)
        self.assertEqual(sum(event[0] == 'stage' for event in events), 4)
        self.assertEqual([event[1] for event in events if event[0] == 'resident-custody'], [17, 29, 43, 71])
        self.assertEqual(sum(event[0] == 'owned-rng' for event in events), 4)
        self.assertFalse(owned.quarantined)

    def test_invalid_custody_output_is_retained_and_marks_owner_unsafe(self):
        owned = custody_worker(invalid=True)
        with self.assertRaises(CollectorError):
            run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
        self.assertTrue(owned._authority.unsafe)
        self.assertIsNotNone(owned._custody.result)

    def test_unsafe_during_warm_frame_keeps_producer_custody(self):
        from looking_glass.budgeted_worker import WorkerContractError
        import looking_glass.trusted_profile_collector as collector
        owned = custody_worker()
        with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
            run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
        def unsafe_frame(result, request_id):
            owned._authority.mark_unsafe()
            return 250000
        with patch.object(collector, '_representative_frame_size', new=unsafe_frame):
            with self.assertRaises(WorkerContractError):
                run_fixed_forward(owned, 29, cold=False, allow_cpu_test=True)
        self.assertTrue(owned._authority.unsafe)
        self.assertIsNotNone(owned._custody.result)

    def test_actual_custody_cycle_releases_model_only_after_clean_drain(self):
        import looking_glass.trusted_profile_collector as collector
        owned = custody_worker()
        with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
            run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
        model = owned._worker._model
        reference = weakref.ref(model)
        state = {'worker': owned, 'authority': owned._authority, 'unsafe': False,
                 'bundle': SimpleNamespace(model=model), 'copy_result': model, 'model_ref': reference}
        del model, owned
        collector._dispose_custody_worker(state)
        self.assertIsNone(reference())
        self.assertIsNone(state['worker'])
        self.assertIsNotNone(state['authority'])
        with self.assertRaises(ValueError):
            state['authority']._bind_worker(object())

    def test_unsafe_during_close_keeps_worker_and_authority_rooted(self):
        import looking_glass.trusted_profile_collector as collector
        from looking_glass.owner_completion import LifetimeWorkspaceWorker
        for phase in ('worker', 'resident'):
            with self.subTest(phase=phase):
                owned = custody_worker()
                with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
                    run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
                authority = owned._authority
                reference = weakref.ref(owned._worker._model)
                state = {'worker': owned, 'authority': authority, 'unsafe': False,
                         'bundle': None, 'copy_result': None, 'model_ref': reference}
                cls = LifetimeWorkspaceWorker if phase == 'worker' else ResidentWorker
                original = cls.close
                def close_and_mark(instance):
                    original(instance)
                    authority.mark_unsafe()
                with patch.object(cls, 'close', close_and_mark):
                    with self.assertRaisesRegex(CollectorError, 'became unsafe'):
                        collector._dispose_custody_worker(state)
                self.assertIs(state['worker'], owned)
                self.assertIs(state['authority'], authority)
                self.assertTrue(authority.unsafe)
                self.assertIsNotNone(reference())

    def test_unsafe_during_retirement_keeps_worker_rooted(self):
        import looking_glass.trusted_profile_collector as collector
        from looking_glass.owner_completion import OwnerCompletionAuthority
        owned = custody_worker()
        with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
            run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
        authority = owned._authority
        reference = weakref.ref(owned._worker._model)
        state = {'worker': owned, 'authority': authority, 'unsafe': False,
                 'bundle': None, 'copy_result': None, 'model_ref': reference}
        retire = OwnerCompletionAuthority.retire_unbrokered_worker
        def retire_and_mark(instance, worker):
            retire(instance, worker)
            instance.mark_unsafe()
        with patch.object(OwnerCompletionAuthority, 'retire_unbrokered_worker', retire_and_mark):
            with self.assertRaisesRegex(CollectorError, 'became unsafe'):
                collector._dispose_custody_worker(state)
        self.assertIs(state['worker'], owned)
        self.assertIs(state['authority'], authority)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(reference())

    def test_active_operation_prevents_worker_retirement(self):
        import looking_glass.trusted_profile_collector as collector
        owned = custody_worker()
        with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
            run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
        reference = weakref.ref(owned._worker._model)
        state = {'worker': owned, 'authority': owned._authority, 'unsafe': False,
                 'operation_in_flight': True, 'bundle': None, 'copy_result': None, 'model_ref': reference}
        with self.assertRaises(CollectorError):
            collector._dispose_custody_worker(state)
        self.assertIs(state['worker'], owned)
        self.assertIsNotNone(reference())

    def test_interrupted_retirement_return_keeps_worker_and_claims_rooted(self):
        import looking_glass.trusted_profile_collector as collector
        from looking_glass.owner_completion import OwnerCompletionAuthority
        owned = custody_worker()
        with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
            run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
        authority = owned._authority
        resident, workspace = owned._resident_lease, owned._workspace_lease
        reference = weakref.ref(owned._worker._model)
        state = {'worker': owned, 'authority': authority, 'unsafe': False,
                 'bundle': None, 'copy_result': None, 'model_ref': reference}
        target = OwnerCompletionAuthority.retire_unbrokered_worker.__code__
        previous = sys.gettrace()
        def interrupt(frame, event, arg):
            if frame.f_code is target and event == 'return':
                raise KeyboardInterrupt('retirement return')
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                collector._dispose_custody_worker(state)
        finally:
            sys.settrace(previous)
        self.assertIs(state['worker'], owned)
        self.assertIs(state['authority'], authority)
        self.assertTrue(state['unsafe'])
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(reference())
        self.assertFalse(resident.released)
        self.assertFalse(workspace.released)

    def test_unsafe_custody_cycle_stays_rooted_on_disposal_failure(self):
        import looking_glass.trusted_profile_collector as collector
        owned = custody_worker()
        owned._authority.mark_unsafe()
        reference = weakref.ref(owned._worker._model)
        state = {'worker': owned, 'authority': owned._authority, 'unsafe': False,
                 'bundle': None, 'copy_result': None, 'model_ref': reference}
        with self.assertRaises(CollectorError):
            collector._dispose_custody_worker(state)
        self.assertIs(state['worker'], owned)
        self.assertIsNotNone(reference())

    def test_untransferred_logits_stay_rooted_on_disposal_failure(self):
        import looking_glass.trusted_profile_collector as collector
        owned = custody_worker()
        with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
            run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
        pending_logits = object()
        owned._custody.logits = pending_logits
        reference = weakref.ref(owned._worker._model)
        state = {'worker': owned, 'authority': owned._authority, 'unsafe': False,
                 'bundle': None, 'copy_result': None, 'model_ref': reference}
        with self.assertRaises(CollectorError):
            collector._dispose_custody_worker(state)
        self.assertIs(state['worker'], owned)
        self.assertIs(owned._custody.logits, pending_logits)
        self.assertIsNotNone(reference())

    def test_producer_error_roots_prevent_disposal_and_keep_claims(self):
        import looking_glass.trusted_profile_collector as collector
        for name in ('snapshot_error', 'forward_error', 'state_error', 'producer_error'):
            with self.subTest(name=name):
                owned = custody_worker()
                with patch.object(collector, '_representative_frame_size', new=lambda result, request_id: 250000):
                    run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
                root = owned._worker if name == 'snapshot_error' else owned._custody
                field = '_snapshot_error' if name == 'snapshot_error' else name
                pending_error = RuntimeError('retained producer failure')
                setattr(root, field, pending_error)
                authority = owned._authority
                resident, workspace = owned._resident_lease, owned._workspace_lease
                reference = weakref.ref(owned._worker._model)
                state = {'worker': owned, 'authority': authority, 'unsafe': False,
                         'bundle': None, 'copy_result': None, 'model_ref': reference}
                with self.assertRaises(CollectorError):
                    collector._dispose_custody_worker(state)
                self.assertIs(state['worker'], owned)
                self.assertIs(state['authority'], authority)
                self.assertIs(getattr(root, field), pending_error)
                self.assertTrue(state['unsafe'])
                self.assertTrue(authority.unsafe)
                self.assertIsNotNone(reference())
                self.assertFalse(resident.released)
                self.assertFalse(workspace.released)

    def test_nonfinite_or_unprofiled_seed_produces_no_observation(self):
        owned = custody_worker(invalid=True)
        try:
            with self.assertRaises(CollectorError): run_fixed_forward(owned, 17, cold=True, allow_cpu_test=True)
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

    def test_owner_lost_during_post_snapshot_blocks_completed_sample(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='lost-during-post-snapshot')
        self.assertNotIn('cold-17-postcheck.json', self.captured_writes)
        self.assertNotIn('cold-17-sample.json', self.captured_writes)
        self.assertNotIn('forward-29', self.captured_events)
        self.assertNotIn('worker-close', self.captured_events)
        self.assertEqual(len(self.captured_budget.active_leases()), 2)
        self.assertEqual(sorted(lease.estimated_incremental_bytes for lease in self.captured_budget.active_leases()),
                         [512 * 1024 ** 2, 2 * GIB])

    def test_changed_device_total_prevents_model_copy(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='total-before-copy')
        self.assertIn('load', self.captured_events)
        self.assertNotIn('copy', self.captured_events)
        self.assertNotIn('forward-17', self.captured_events)

    def test_changed_device_total_prevents_next_forward(self):
        with self.assertRaises(CollectorError):
            self.run_fake_owned(fail_forward=False, latch_at='total-before-warm')
        self.assertIn('forward-17', self.captured_events)
        self.assertNotIn('forward-29', self.captured_events)
        self.assertNotIn('warm-29-sample.json', self.captured_writes)

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
                                     (collector, '_require_workspace_spec', lambda *args: None),
                                     (collector, '_require_workspace_binding', lambda *args: None),
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

    def test_cleanup_stops_before_next_cuda_call_if_owner_or_claim_is_lost(self):
        import looking_glass.trusted_profile_collector as collector
        from looking_glass.device_lease import DeviceLease
        from looking_glass.memory_budget import MemoryBudget
        from looking_glass.supervisor import OwnerSession

        for failure in ('unsafe', 'resident'):
            with self.subTest(failure=failure):
                budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0),
                                      max_gpu_trace_bytes=0, max_host_trace_bytes=0)
                resident = budget.acquire(2 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
                job = budget.acquire(512 * 1024 ** 2, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
                state = {'unsafe': False, 'worker': None, 'authority': None}
                collector._OWNER_CUSTODY[3], collector._OWNER_CUSTODY[5:8] = state, [budget, resident, job]
                owner, lease = object.__new__(OwnerSession), object.__new__(DeviceLease)
                owner._identity = SimpleNamespace(cuda_device_index=0)
                owner._runtime_verified = True
                events = []
                def sync(_index):
                    events.append('synchronize')
                    if failure == 'unsafe':
                        state['unsafe'] = True
                    else:
                        resident.release()
                cuda = SimpleNamespace(is_initialized=lambda: events.append('initialized') or True,
                    synchronize=sync, empty_cache=lambda: events.append('empty_cache'),
                    memory_allocated=lambda index: events.append('allocated') or 0,
                    memory_reserved=lambda index: events.append('reserved') or 0)
                fake_torch = SimpleNamespace(__version__='2.8.0+cu128', version=SimpleNamespace(cuda='12.8'),
                    cuda=cuda, _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: events.append('workspace')))
                with patch.object(OwnerSession, 'check'):
                    with self.assertRaises(CollectorError):
                        collector._guarded_cleanup(owner, lease, fake_torch)
                self.assertEqual(events, ['initialized', 'synchronize'])
                self.assertIs(collector._OWNER_CUSTODY[6], resident)
                self.assertIs(collector._OWNER_CUSTODY[7], job)
                self.assertFalse(job.released)

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
        original_repo = Path(__file__).resolve().parents[1]
        site = sysconfig.get_path('purelib')
        script = textwrap.dedent('''
            import atexit, hashlib, json, os, runpy, sys
            from pathlib import Path
            from types import SimpleNamespace
            repo, site, folder = sys.argv[1:4]
            atexit.register(lambda: (Path(folder) / 'atexit-ran').write_text('unexpected'))
            sys.path.extend((repo, site))
            import looking_glass.trusted_service_launch as launch
            import looking_glass.trusted_service_profile as trusted
            import looking_glass.supervisor as supervisor
            from looking_glass.supervisor import OwnerSession, _identity_spec
            from looking_glass.device_lease import DeviceLease
            from looking_glass.device_identity import resolve_device_identity
            supervisor.SUPERVISOR_LAUNCH_CUSTODY_CONTRACT = 'gated-rooted-child-v1'
            gpu_uuid = 'GPU-12345678-1234-5678-9abc-1234567890ab'
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            identity = resolve_device_identity(Path(repo).parents[2], gpu_uuid, physical_uuids=[gpu_uuid])
            owner = object.__new__(OwnerSession)
            owner._identity = identity
            owner._identity_spec = _identity_spec(identity)
            owner.device_lease_path = identity.lock_path
            owner.marker_path = identity.marker_path
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
            repo = Path(folder) / 'workspace' / 'work' / 'frozen' / 'checkout'
            repo.mkdir(parents=True)
            shutil.copytree(original_repo / 'looking_glass', repo / 'looking_glass',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
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
        original_repo = Path(__file__).resolve().parents[1]
        site = Path(sysconfig.get_path('purelib')).resolve()
        environment = Path(sys.prefix).resolve()
        gpu_uuid = 'GPU-12345678-1234-5678-9abc-1234567890ab'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            workspace, wrong_workspace, private, home = (root / name for name in
                ('workspace', 'wrong-workspace', 'private', 'hf'))
            for path in (workspace, wrong_workspace, private, home):
                path.mkdir()
            repo = workspace / 'work' / 'frozen' / 'checkout'
            repo.mkdir(parents=True)
            shutil.copytree(original_repo / 'looking_glass', repo / 'looking_glass',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            (repo / 'docs').mkdir()
            shutil.copy2(original_repo / 'docs' / 'TRUSTED_SERVICE_PROFILE_V1.md', repo / 'docs')
            cache = home / 'hub'
            cache.mkdir()
            for selected, is_shared in ((workspace, True), (wrong_workspace, False)):
                with self.subTest(is_shared=is_shared):
                    run_dir = private / ('shared' if is_shared else 'wrong')
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
                    identity = resolve_device_identity(selected, gpu_uuid, physical_uuids=[gpu_uuid],
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
                    if (is_shared and getattr(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT', None)
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
        import looking_glass.custody_service_profile as custody
        import looking_glass.owner_completion as completion
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
            def __init__(self, model_id, model, **kwargs):
                self.model, self.model_id = model, model_id
                self._snapshot_error = None
            def close(self): events.append('worker-close'); self.model = None
        class FakeAuthority:
            def __init__(self, *, device_guard):
                self.unsafe = False
                self.device_guard = device_guard
                self._worker = None
                self._decision_lock = Lock()
            def mark_unsafe(self): self.unsafe = True
            def retire_unbrokered_worker(self, worker):
                if self.unsafe or self._worker is not worker:
                    raise CollectorError('unsafe fake retirement')
                self._worker = object()
        class FakeLifetime:
            quarantined = False
            poisoned = False
            def __init__(self, worker, budget, resident, workspace, *, profiles, authority):
                self._worker, self._authority = worker, authority
                authority._worker = self
                self._cold_pending, self._warmup_result = True, None
                self._custody = SimpleNamespace(result=None, request=None, logits=None, error=None,
                                                forward_error=None, state_error=None, producer_error=None)
                self._retiring_transfer = self._retiring_cancel = None
                self._transfer_cleanup_error = self._cancel_cleanup_error = None
            def close(self): pass
        captured_writes = self.captured_writes
        def memory_info(device):
            if (latch_at == 'total-before-copy' and 'load' in events and 'copy' not in events
                    or latch_at == 'total-before-warm' and 'cold-17-sample.json' in captured_writes):
                return 6 * GIB, 9 * GIB
            return 6 * GIB, 8 * GIB
        class FakeProbe:
            def __init__(self, *args, **kwargs):
                self.device, self.device_uuid = kwargs['device'], kwargs['device_uuid']
                self._cuda = SimpleNamespace(mem_get_info=memory_info,
                                             memory_allocated=lambda device: 0, memory_reserved=lambda device: 0)
            def snapshot(self): return CudaMemorySnapshot(6 * GIB, 8 * GIB, 0, 0)
            def synchronize(self): events.append('probe-sync')
        fake_cuda = SimpleNamespace(device_count=lambda: 1, is_initialized=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(uuid='fake'),
            synchronize=lambda index: None, empty_cache=lambda: None,
            memory_allocated=lambda index: 0, memory_reserved=lambda index: 0)
        fake_torch = SimpleNamespace(cuda=fake_cuda,
            _C=SimpleNamespace(_CUuuid=object, _cuda_clearCublasWorkspaces=lambda: None),
            __version__='2.8.0+cu128', version=SimpleNamespace(cuda='12.8'), float32='float32',
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
            if kwargs['cold']:
                owned._cold_pending = False
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
                   (collector, '_require_workspace_binding', lambda *args: None),
                   (custody, 'build_custody_service_identity', lambda *args: live),
                   (trusted, 'prepare_trusted_source', lambda *args: source),
                   (trusted, 'verify_trusted_cpu_model', lambda *args: True),
                   (trusted, 'verify_trusted_device_tie', lambda *args: True),
                   (memory, 'MemoryBudget', budget_factory),
                   (resource, 'TorchMemoryProbe', FakeProbe), (resource, 'collect_sample', sample),
                   (resident, 'ResidentWorker', FakeWorker), (pretrained, 'load_distilgpt2', load),
                   (completion, 'OwnerCompletionAuthority', FakeAuthority),
                   (completion, 'LifetimeWorkspaceWorker', FakeLifetime),
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
                if (latch_at == 'lost-during-post-snapshot' and events
                        and events[-1] == 'forward-17'):
                    owner_alive[0] = False
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
