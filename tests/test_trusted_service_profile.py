import dataclasses
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass.memory_budget import CudaMemorySnapshot
from looking_glass.resource_profile import (MemorySample, MeasuredResourceProfile,
    ProfileProvenance, ProfileRegistry, encode_evidence)


GIB = 1024 ** 3
UUID = 'GPU-12345678-1234-5678-9abc-1234567890ab'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def fake_runtime():
    return {'schema': 'trusted-service-runtime-v1', 'device': 'cuda:0',
            'device_uuid': UUID, 'device_name': 'GPU model', 'device_total_bytes': 8 * GIB,
            'compute_capability': [7, 5], 'python_version': '3.11.15',
            'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
            'cuda_runtime_version': '12.8', 'driver_version': '616.92',
            'model_dtype': 'torch.float32', 'compute_dtype': 'torch.float32',
            'attention_backend': 'eager', 'model_mode': 'eval', 'grad_mode': 'inference',
            'autocast': 'disabled', 'use_cache': False, 'compiled': False,
            'allocator_fraction': '0.20'}


def profile_for(key, source, seeds, case_hash, *, margin_bytes, margin_bps=2500):
    snapshot = CudaMemorySnapshot(6 * GIB, 8 * GIB, 10, 20)
    samples = tuple(MemorySample(workload_sha256=key.fingerprint(), sample_id=f'sample-{index}',
                                 case_sha256=case_hash, seed=seed, status='completed',
                                 before=snapshot, after=snapshot,
                                 peak_allocated_bytes=10, peak_reserved_bytes=20)
                    for index, seed in enumerate(seeds))
    evidence = encode_evidence(key, samples)
    provenance = ProfileProvenance(protocol='resource-profile-v1',
        source_manifest_sha256=hashlib.sha256(source).hexdigest(),
        evidence_sha256=hashlib.sha256(evidence).hexdigest(),
        collected_utc='2026-10-28T00:00:00Z', isolation='cooperative-exclusive',
        boundaries='synchronize-reset-synchronize')
    return MeasuredResourceProfile.build(key, samples, provenance,
                                         margin_bytes=margin_bytes, margin_bps=margin_bps)


class TrustedServiceProfileTests(unittest.TestCase):
    def make_identity(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        from looking_glass.live_profile import _make_workload_keys
        assets = {'schema': 'resource-bootstrap-assets-v1', 'assets': []}
        source = canonical({'schema': 'trusted-service-source-v1', 'base': {'assets': assets},
                            'dependency_origins': []})
        runtime = canonical(fake_runtime())
        keys = _make_workload_keys(hashlib.sha256(source).hexdigest(), json.loads(runtime))
        return module.TrustedServiceIdentity(keys, source, runtime, canonical(assets))

    def make_profiles(self, identity, *, warm_seeds=(29, 43, 71), warm_case=None):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        load, cold, warm = identity.keys
        return (
            profile_for(load, identity.source_manifest, (17,), module.LOAD_CASE_SHA256,
                        margin_bytes=512 * 1024 ** 2),
            profile_for(cold, identity.source_manifest, (17,), module.FORWARD_CASE_SHA256,
                        margin_bytes=256 * 1024 ** 2),
            profile_for(warm, identity.source_manifest, warm_seeds,
                        warm_case or module.FORWARD_CASE_SHA256, margin_bytes=256 * 1024 ** 2),
        )

    def test_exact_stage_cases_ordered_seeds_and_external_pins_are_required(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        identity = self.make_identity()
        profiles = self.make_profiles(identity)
        pins = tuple(profile.profile_id for profile in profiles)
        registry = ProfileRegistry(profiles, approved_profile_ids=pins)
        load, cold, warm = module.resolve_trusted_service_profiles(registry, identity, pins)
        self.assertEqual(load.profile_id, pins[0])
        self.assertEqual((cold.profile_id, warm.profile_id), pins[1:])
        with self.assertRaises(ValueError):
            module.resolve_trusted_service_profiles(registry, identity, (pins[0], pins[1], 'f' * 64))
        for seeds, case in (((29, 43), None), ((29, 71, 43), None),
                            ((29, 43, 43), None), ((29, 43, 71), 'f' * 64)):
            changed = self.make_profiles(identity, warm_seeds=seeds, warm_case=case)
            changed_pins = tuple(profile.profile_id for profile in changed)
            changed_registry = ProfileRegistry(changed, approved_profile_ids=changed_pins)
            with self.subTest(seeds=seeds, case=case), self.assertRaises(ValueError):
                module.resolve_trusted_service_profiles(changed_registry, identity, changed_pins)

    def test_untrusted_or_changed_identity_never_resolves(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        identity = self.make_identity()
        profiles = self.make_profiles(identity)
        pins = tuple(profile.profile_id for profile in profiles)
        registry = ProfileRegistry(profiles, approved_profile_ids=pins)
        for changed in (dataclasses.replace(identity, source_manifest=b'{}'),
                        dataclasses.replace(identity, runtime_manifest=b'{}'),
                        dataclasses.replace(identity, asset_manifest=b'{}')):
            with self.assertRaises((ValueError, KeyError)):
                module.resolve_trusted_service_profiles(registry, changed, pins)

    def test_actual_request_is_limited_to_exact_ids_and_stage_seeds(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        import torch
        input_ids = torch.tensor([[15496, 995]], dtype=torch.int64)
        self.assertIs(module.validate_service_request({'input_ids': input_ids}, 17, cold=True), True)
        self.assertIs(module.validate_service_request({'input_ids': input_ids}, 29, cold=False), True)
        for inputs, seed, cold in (({'input_ids': input_ids}, 17, False),
                                   ({'input_ids': input_ids}, 29, True),
                                   ({'input_ids': torch.tensor([[15496, 1]])}, 29, False),
                                   ({'input_ids': input_ids, 'attention_mask': input_ids}, 29, False),
                                   ({'input_ids': input_ids.float()}, 29, False)):
            with self.subTest(seed=seed, cold=cold), self.assertRaises(ValueError):
                module.validate_service_request(inputs, seed, cold=cold)

    def test_fixed_request_factory_owns_its_tensor_after_input_mutation(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        raw = [15496, 995]
        owned = module.make_fixed_service_inputs(raw, 29, cold=False)
        raw[1] = 12
        self.assertEqual(owned['input_ids'].tolist(), [[15496, 995]])
        other = module.make_fixed_service_inputs((15496, 995), 29, cold=False)
        self.assertNotEqual(owned['input_ids'].data_ptr(), other['input_ids'].data_ptr())
        for values in ([15496, 1], [15496, True], [15496, 995.0], [15496], '15496,995'):
            with self.subTest(values=values), self.assertRaises(ValueError):
                module.make_fixed_service_inputs(values, 29, cold=False)

    def test_unowned_identity_is_rejected_before_torch_or_device_access(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        with patch.object(module, '_guard_owner', side_effect=ValueError('no owner')), \
             patch.object(module, '_load_torch', side_effect=AssertionError('torch reached')):
            with self.assertRaisesRegex(ValueError, 'no owner'):
                module.build_trusted_service_identity(None, None, None, None)

    def test_source_inventory_changes_when_source_bytes_or_file_set_changes(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'looking_glass').mkdir()
            (root / 'docs').mkdir()
            (root / 'looking_glass' / '__init__.py').write_text('pass', encoding='utf-8')
            (root / 'looking_glass' / 'worker.py').write_text('first', encoding='utf-8')
            (root / 'docs' / 'TRUSTED_SERVICE_PROFILE_V1.md').write_text('protocol', encoding='utf-8')
            first = module._collect_source_files(root)
            self.assertEqual(first, module._collect_source_files(root))
            (root / 'looking_glass' / 'worker.py').write_text('second', encoding='utf-8')
            self.assertNotEqual(first, module._collect_source_files(root))
            (root / 'looking_glass' / 'new_module.py').write_text('pass', encoding='utf-8')
            self.assertNotEqual(first, module._collect_source_files(root))
            (root / 'looking_glass' / 'new_module.py').unlink()
            (root / 'looking_glass' / 'worker.py').write_text('first', encoding='utf-8')
            (root / 'docs' / 'TRUSTED_SERVICE_PROFILE_V1.md').write_text('changed', encoding='utf-8')
            self.assertNotEqual(first, module._collect_source_files(root))

    def test_source_inventory_fails_on_unreadable_subdirectory(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            blocked = root / 'looking_glass' / 'blocked'
            blocked.mkdir(parents=True)
            (root / 'docs').mkdir()
            (root / 'looking_glass' / '__init__.py').write_text('pass', encoding='utf-8')
            (blocked / 'hidden.py').write_text('pass', encoding='utf-8')
            (root / 'docs' / 'TRUSTED_SERVICE_PROFILE_V1.md').write_text('protocol', encoding='utf-8')
            real_scandir = os.scandir
            def scandir(path):
                if Path(path) == blocked:
                    raise PermissionError('injected unreadable directory')
                return real_scandir(path)
            with patch.object(os, 'scandir', side_effect=scandir):
                with self.assertRaises(ValueError):
                    module._collect_source_files(root)

    def test_wrong_runtime_binding_blocks_transformers_import(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        preflight = module.TrustedServiceSource('repo', 'cache', b'source', b'assets', ())
        identity = SimpleNamespace(gpu_uuid=UUID)
        fake_torch = SimpleNamespace(__future__=SimpleNamespace(
            get_overwrite_module_params_on_conversion=lambda: False,
            get_swap_module_params_on_conversion=lambda: False))
        with patch.object(module, '_guard_owner'), \
             patch.object(module, '_checked_environment', return_value={}), \
             patch.object(module, 'prepare_trusted_source', return_value=preflight), \
             patch.object(module, '_load_torch', return_value=fake_torch), \
             patch.object(module, '_measure_runtime', side_effect=ValueError('wrong device')), \
             patch.object(module, '_load_transformers', side_effect=AssertionError('transformers reached')):
            with self.assertRaisesRegex(ValueError, 'wrong device'):
                module.build_trusted_service_identity(preflight, identity, object(), object())

    def test_live_identity_uses_revalidated_files_and_observed_runtime(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        assets = {'schema': 'resource-bootstrap-assets-v1', 'assets': []}
        source_base = canonical({'schema': 'trusted-service-source-base-v1', 'assets': assets})
        preflight = module.TrustedServiceSource('repo', 'cache', source_base, canonical(assets), ())
        identity = SimpleNamespace(gpu_uuid=UUID)
        future = SimpleNamespace(get_overwrite_module_params_on_conversion=lambda: False,
                                 get_swap_module_params_on_conversion=lambda: False)
        torch = SimpleNamespace(__future__=future, __version__='2.8.0+cu128')
        transformers = SimpleNamespace(__version__='4.57.3')
        events = []
        def guard(*args): events.append('owner')
        def observe(*args): events.append('runtime'); return fake_runtime()
        def dependency(): events.append('transformers'); return transformers
        with patch.object(module, '_guard_owner', side_effect=guard), \
             patch.object(module, '_checked_environment', return_value={'CUDA_VISIBLE_DEVICES': UUID}), \
             patch.object(module, 'prepare_trusted_source', return_value=preflight), \
             patch.object(module, '_load_torch', return_value=torch), \
             patch.object(module, '_measure_runtime', side_effect=observe), \
             patch.object(module, '_load_transformers', side_effect=dependency), \
             patch.object(module, '_module_origin', side_effect=lambda mod, name: {'name': name}):
            result = module.build_trusted_service_identity(preflight, identity, object(), object())
        self.assertLess(events.index('runtime'), events.index('transformers'))
        self.assertEqual(result.keys[0].source_sha256, hashlib.sha256(result.source_manifest).hexdigest())
        self.assertEqual(result.keys[0].runtime_settings_sha256,
                         hashlib.sha256(result.runtime_manifest).hexdigest())
        self.assertEqual(json.loads(result.source_manifest)['base']['assets'], assets)
        self.assertEqual(json.loads(result.runtime_manifest)['device_uuid'], UUID)
        with patch.object(module, '_guard_owner'), \
             patch.object(module, '_checked_environment', return_value={}), \
             patch.object(module, 'prepare_trusted_source', return_value=None), \
             patch.object(module, '_load_torch', side_effect=AssertionError('torch reached')):
            with self.assertRaisesRegex(ValueError, 'changed'):
                module.build_trusted_service_identity(preflight, identity, object(), object())

    def test_runtime_environment_requires_fixed_controls_and_rejects_unknown_overrides(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        required = {'CUDA_VISIBLE_DEVICES': UUID, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'PYTORCH_CUDA_ALLOC_CONF': 'backend:native', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with patch.dict(os.environ, required, clear=True):
            result = module._checked_environment(UUID)
            self.assertEqual(result['CUDA_VISIBLE_DEVICES'], UUID)
            self.assertIsNone(result['CUDA_LAUNCH_BLOCKING'])
        for change in ({'CUDA_VISIBLE_DEVICES': '0'}, {'OPENBLAS_NUM_THREADS': '2'},
                       {'CUDA_LAUNCH_BLOCKING': '1'}, {'PYTORCH_ALLOC_CONF': 'backend:native'}):
            with self.subTest(change=change), patch.dict(os.environ, {**required, **change}, clear=True):
                with self.assertRaises(ValueError):
                    module._checked_environment(UUID)

    def test_relative_import_paths_are_bound_to_the_actual_working_directory(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = module._resolved_import_paths(['', 'relative', str(root / 'fixed')], root)
            self.assertEqual(paths, [str(root), str(root / 'relative'), str(root / 'fixed')])
            with self.assertRaises(ValueError):
                module._resolved_import_paths([None], root)

    def test_cpu_model_verifier_requires_explicit_true_and_revalidated_source(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        preflight = module.TrustedServiceSource('repo', 'cache', b'source', b'assets', ())
        with patch.object(module, 'prepare_trusted_source', return_value=preflight), \
             patch.object(module, 'verify_loaded_model', return_value=True) as verify:
            self.assertIs(module.verify_trusted_cpu_model(preflight, object()), True)
            self.assertEqual(verify.call_count, 1)
        with patch.object(module, 'prepare_trusted_source', return_value=preflight), \
             patch.object(module, 'verify_loaded_model', return_value=False):
            with self.assertRaises(ValueError):
                module.verify_trusted_cpu_model(preflight, object())
        with patch.object(module, 'prepare_trusted_source', return_value=None), \
             patch.object(module, 'verify_loaded_model', side_effect=AssertionError('model verifier reached')):
            with self.assertRaises(ValueError):
                module.verify_trusted_cpu_model(preflight, object())

    def test_device_tie_verification_rejects_split_parameters_with_cpu_fakes(self):
        module = importlib.import_module('looking_glass.trusted_service_profile')
        class Parameter:
            device = SimpleNamespace(type='cuda', index=0)
            dtype = 'float32'
            def untyped_storage(self): return SimpleNamespace(data_ptr=lambda: 123)
        item = Parameter()
        class Model:
            def get_parameter(self, name):
                return self.head if name == 'lm_head.weight' else item
        model = Model()
        model.head = item
        future = SimpleNamespace(get_overwrite_module_params_on_conversion=lambda: False,
                                 get_swap_module_params_on_conversion=lambda: False)
        fake_torch = SimpleNamespace(nn=SimpleNamespace(Parameter=Parameter), float32='float32', __future__=future)
        with patch.object(module, '_guard_owner'), patch.object(module, '_load_torch', return_value=fake_torch):
            self.assertIs(module.verify_trusted_device_tie(model, None, None, None), True)
            model.head = Parameter()
            with self.assertRaises(ValueError):
                module.verify_trusted_device_tie(model, None, None, None)


if __name__ == '__main__':
    unittest.main()
