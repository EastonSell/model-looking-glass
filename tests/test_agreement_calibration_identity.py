"""CPU-only checks for the separate five-token runtime identity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass.agreement_calibration_identity import (_make_key, _verify_dependency_origins,
                                                           _numeric_observation)
from looking_glass.pretrained import ASSETS
from looking_glass.resource_profile import canonical_fingerprint


def runtime():
    total = 8 * 1024 ** 3
    return {'schema': 'agreement-calibration-runtime-v1',
        'execution_path': 'looking_glass.agreement_cuda_calibration.CudaAgreementCalibrator',
        'device': 'cuda:0', 'device_uuid': 'GPU-00000000-0000-0000-0000-000000000001',
        'device_name': 'Synthetic GPU', 'device_total_bytes': total, 'compute_capability': [7, 5],
        'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
        'cuda_runtime_version': '12.8', 'driver_version': '616.92', 'python_version': '3.11.15',
        'model_dtype': 'torch.float32', 'compute_dtype': 'torch.float32', 'attention_backend': 'eager',
        'model_mode': 'eval', 'grad_mode': 'inference', 'autocast': 'disabled',
        'use_cache': False, 'compiled': False, 'allocator_backend': 'native',
        'allocator_fraction': '0.25', 'allocator_limit_bytes': total // 4}


class AgreementCalibrationIdentityTests(unittest.TestCase):
    def test_module_import_does_not_load_torch(self):
        repo = Path(__file__).resolve().parents[1]
        program = ("import sys;sys.path.insert(0,sys.argv[1]);"
                   "import looking_glass.agreement_calibration_identity;"
                   "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        result = subprocess.run([sys._base_executable, '-S', '-P', '-B', '-c', program, str(repo)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_key_binds_actual_runtime_source_and_fixed_capture_contract(self):
        from looking_glass.agreement_cuda_runner import CAPTURE_POLICY_SHA256, WARMUP_POLICY_SHA256
        source = b'{"schema":"synthetic-source"}'
        observed = runtime()
        key = _make_key(source, observed)
        self.assertEqual(key.source_sha256, hashlib.sha256(source).hexdigest())
        self.assertEqual(key.runtime_settings_sha256, canonical_fingerprint(observed))
        self.assertEqual(key.checkpoint_sha256, ASSETS['model.safetensors'][1])
        self.assertEqual(key.model_config_sha256, ASSETS['config.json'][1])
        self.assertEqual((key.batch_size, key.sequence_length, key.capture_mode), (1, 5, 'direct'))
        self.assertEqual(key.capture_policy_sha256, CAPTURE_POLICY_SHA256)
        self.assertEqual(key.warmup_protocol_sha256, WARMUP_POLICY_SHA256)
        self.assertNotEqual(key.fingerprint(), _make_key(source + b' ', observed).fingerprint())

    def test_unreviewed_allocator_or_execution_path_is_rejected(self):
        for field, value in (('allocator_backend', 'cudaMallocAsync'),
                             ('allocator_fraction', '0.20'),
                             ('allocator_limit_bytes', 1),
                             ('execution_path', 'looking_glass.resident_worker.ResidentWorker.execute'),
                             ('torch_version', 'other')):
            changed = runtime()
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _make_key(b'{}', changed)

    def test_owner_failure_precedes_any_dependency_import(self):
        import looking_glass.agreement_calibration_identity as identity
        from looking_glass.trusted_service_profile import TrustedServiceSource
        source = TrustedServiceSource('repo', 'cache', b'{}', b'{}', ())
        session = SimpleNamespace(identity=object())
        with (patch.object(identity, '_guard_owner', side_effect=RuntimeError('owner lost')),
              patch.object(identity.importlib, 'import_module') as imported):
            with self.assertRaisesRegex(RuntimeError, 'owner lost'):
                identity.prepare_agreement_runtime(source, session, object(), frozen_runtime={},
                                                   frozen_driver_version='616.92')
        imported.assert_not_called()

    def test_wrong_torch_build_rejects_before_cuda_access(self):
        import looking_glass.agreement_calibration_identity as identity
        from looking_glass.trusted_service_profile import TrustedServiceSource
        class ForbiddenCuda:
            def __getattr__(self, name):
                raise AssertionError('CUDA was accessed before validating Torch')
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory).resolve()
            torch_source = site / 'torch.py'
            torch_source.write_text('# fake')
            source = TrustedServiceSource(str(site), str(site), b'{}', b'{}', ())
            session = SimpleNamespace(identity=SimpleNamespace(gpu_uuid='GPU-fake'))
            fake = SimpleNamespace(__file__=str(torch_source), __version__='wrong',
                                   version=SimpleNamespace(cuda='12.8'), cuda=ForbiddenCuda())
            with (patch.object(identity, '_guard_owner'),
                  patch.object(identity, '_check_import_preconditions'),
                  patch.object(identity, '_verify_dependency_origins', return_value=str(site)),
                  patch.object(identity, '_install_spawn_free_guard'),
                  patch.object(identity, '_checked_environment', return_value={}),
                  patch.object(identity, 'prepare_trusted_source', return_value=source),
                  patch.object(identity, '_verify_frozen_import_finders'),
                  patch.object(identity.importlib, 'import_module', return_value=fake)):
                with self.assertRaisesRegex(ValueError, 'Torch build differs'):
                    identity.prepare_agreement_runtime(source, session, object(),
                        frozen_runtime={'torch_version': '2.8.0+cu128', 'cuda_runtime_version': '12.8'},
                        frozen_driver_version='616.92')

    def test_import_finder_failure_precedes_torch_import(self):
        import looking_glass.agreement_calibration_identity as identity
        from looking_glass.trusted_service_profile import TrustedServiceSource
        source = TrustedServiceSource('repo', 'cache', b'{}', b'{}', ())
        session = SimpleNamespace(identity=SimpleNamespace(gpu_uuid='GPU-fake'))
        with (patch.object(identity, '_guard_owner'),
              patch.object(identity, '_check_import_preconditions'),
              patch.object(identity, '_verify_dependency_origins', return_value='site'),
              patch.object(identity, '_install_spawn_free_guard'),
              patch.object(identity, '_checked_environment', return_value={}),
              patch.object(identity, 'prepare_trusted_source', return_value=source),
              patch.object(identity, '_verify_frozen_import_finders',
                           side_effect=RuntimeError('unexpected Python import finder')),
              patch.object(identity.importlib, 'import_module') as imported):
            with self.assertRaisesRegex(RuntimeError, 'import finder'):
                identity.prepare_agreement_runtime(source, session, object(),
                    frozen_runtime={}, frozen_driver_version='616.92')
        imported.assert_not_called()

    @unittest.skipUnless(sys.platform == 'win32', 'the owner guard is Windows-specific')
    def test_spawn_guard_blocks_python_descendants_in_isolated_child(self):
        import os
        repo = Path(__file__).resolve().parents[1]
        program = '''
import sys, subprocess
sys.path.insert(0, sys.argv[1])
from looking_glass.agreement_calibration_identity import _install_spawn_free_guard
_install_spawn_free_guard()
try:
    subprocess.run([sys.executable, '-c', 'pass'])
except RuntimeError as error:
    assert 'subprocess creation is prohibited' in str(error)
else:
    raise AssertionError('a Python subprocess was launched')
import platform
assert platform.platform().startswith('Windows-')
'''
        result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', program, str(repo)],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_isolated_fresh_child_ignores_pythonpath_and_rejects_preloaded_submodules(self):
        import os
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hostile, prefix = root / 'hostile', root / 'pycache'
            hostile.mkdir(); prefix.mkdir()
            (hostile / 'torch.py').write_text('raise AssertionError("shadow imported")\n')
            program = '''
import sys, types
sys.path.insert(0, sys.argv[1])
import looking_glass.agreement_calibration_identity as identity
identity._check_import_preconditions()
for name in ('torch.fake', 'transformers.fake', 'huggingface_hub.constants'):
    sys.modules[name] = types.ModuleType(name)
    try:
        identity._check_import_preconditions()
    except ValueError:
        pass
    else:
        raise AssertionError('accepted a preloaded dependency')
    del sys.modules[name]
assert 'torch' not in sys.modules
'''
            environment = dict(os.environ, PYTHONPATH=str(hostile), CUDA_VISIBLE_DEVICES='')
            result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X',
                                     'pycache_prefix=' + str(prefix), '-c', program, str(repo)],
                                    env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(any(prefix.iterdir()))

    def test_selected_site_is_the_only_nonstdlib_import_root(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, site, hostile = root / 'repo', root / 'site-packages', root / 'hostile'
            repo.mkdir(); site.mkdir(); hostile.mkdir()
            for name in ('torch', 'transformers', 'huggingface_hub', 'numpy', 'safetensors', 'tokenizers'):
                package = site / name
                package.mkdir()
                (package / '__init__.py').write_text('# test source\n')
            (hostile / 'torch.py').write_text('# unapproved shadow\n')
            paths = [str(Path(sys.base_prefix) / 'Lib'), str(site)]
            env = {'LOOKING_GLASS_AGREEMENT_BOOTSTRAP_SITE': str(site.resolve())}
            with patch.dict(os.environ, env), patch.object(sys, 'path', paths):
                self.assertEqual(_verify_dependency_origins(repo), str(site.resolve()))
            with patch.dict(os.environ, env), patch.object(sys, 'path', [str(hostile), *paths]):
                with self.assertRaises(ValueError):
                    _verify_dependency_origins(repo)

    def test_numeric_observation_reads_current_calling_thread_state(self):
        import torch
        observed = _numeric_observation(torch)
        self.assertEqual(observed['float32_matmul_precision'], torch.get_float32_matmul_precision())
        self.assertEqual(observed['deterministic_warn_only'], torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertIn(observed['calling_thread_denormal_product_bits'], (0, 0x00400000))


if __name__ == '__main__':
    unittest.main()
