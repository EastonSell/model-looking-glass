import importlib
import json
import os
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


UUID = 'GPU-12345678-1234-5678-9abc-1234567890ab'
GIB = 1024 ** 3


def fake_source_manifest():
    paths = ['docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md', 'looking_glass/__init__.py',
             'looking_glass/gpu_bootstrap.py', 'looking_glass/gpu_service.py', 'looking_glass/live_profile.py']
    names = ['huggingface-hub', 'numpy', 'safetensors', 'tokenizers', 'torch', 'transformers']
    return {'schema': 'resource-bootstrap-source-v1', 'git_commit': '1' * 40,
            'git_tree': '2' * 40,
            'repository_files': [{'path': name, 'size': 1, 'sha256': 'a' * 64} for name in paths],
            'python_executable': {'size': 1, 'sha256': 'b' * 64},
            'distributions': [{'name': name, 'version': '1',
                               'files': [{'path': f'Lib/site-packages/{name}/file.py',
                                          'size': 1, 'sha256': 'c' * 64}]} for name in names]}


class FakeCudaUuid:
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text


class LiveProfileTests(unittest.TestCase):
    def test_cuda_uuid_decoder_requires_expected_type_and_complete_format(self):
        module = importlib.import_module('looking_glass.live_profile')
        self.assertEqual(module.decode_cuda_uuid(FakeCudaUuid('12345678-1234-5678-9abc-1234567890ab'), FakeCudaUuid), UUID)
        for value in ('12345678-1234-5678-9abc-1234567890ab',
                      FakeCudaUuid('1234'), FakeCudaUuid('12345678-1234-5678-9ABC-1234567890ab'),
                      FakeCudaUuid('GPU-12345678-1234-5678-9abc-1234567890ab')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.decode_cuda_uuid(value, FakeCudaUuid)

    def test_allocator_readback_uses_integer_truncation_and_unchanged_total(self):
        module = importlib.import_module('looking_glass.live_profile')
        total = 8 * GIB
        cap = int(0.20 * total)
        self.assertNotEqual(cap / total, 0.20)
        self.assertEqual(module.check_allocator_readback(total, total, cap / total), cap)
        for args in ((total, total + 1, cap / total), (total, total, 0.20),
                     (total, total, float('nan')), (0, 0, 0.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                module.check_allocator_readback(*args)

    def test_environment_is_exact_and_rejects_unreviewed_controls(self):
        module = importlib.import_module('looking_glass.live_profile')
        expected = {
            'CUDA_VISIBLE_DEVICES': UUID, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'PYTORCH_CUDA_ALLOC_CONF': 'backend:native', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
            'PYTHONHASHSEED': '17', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPYCACHEPREFIX': str(Path(tempfile.gettempdir()) / 'cache'),
            'PATH': str(Path(tempfile.gettempdir()).resolve()),
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        }
        result = module.checked_environment(expected, UUID)
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], UUID)
        self.assertIsNone(result['PYTORCH_ALLOC_CONF'])
        self.assertEqual(result['PATH'], expected['PATH'])
        for changes in ({'CUDA_VISIBLE_DEVICES': '0'}, {'CUBLAS_WORKSPACE_CONFIG': ''},
                        {'PYTORCH_ALLOC_CONF': 'backend:native'}, {'CUDA_LAUNCH_BLOCKING': '1'},
                        {'OPENBLAS_NUM_THREADS': '2'}, {'HF_TOKEN': 'secret'}, {'PYTHONHOME': 'untrusted'}, {'PYTHONPATH': 'untrusted'}, {'LD_PRELOAD': 'untrusted'}, {'PATH': ''},
                        {'PATH': '.'}, {'PATH': os.pathsep + expected['PATH']},
                        {'PATH': expected['PATH'] + os.pathsep + expected['PATH']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                module.checked_environment({**expected, **changes}, UUID)

    def test_bootstrap_profile_requires_source_key_and_provenance_to_match(self):
        module = importlib.import_module('looking_glass.live_profile')
        from looking_glass.resource_profile import canonical_fingerprint
        manifest = fake_source_manifest()
        digest = canonical_fingerprint(manifest)
        module.check_profile_source_identity(digest, digest, manifest)
        without_launcher = fake_source_manifest()
        without_launcher['repository_files'] = [record for record in without_launcher['repository_files']
                                                if record['path'] != 'looking_glass/gpu_bootstrap.py']
        with self.assertRaises(ValueError):
            module.validate_source_manifest(without_launcher)
        for invalid_path in ('.', 'Lib/pkg/na\x00me.py'):
            invalid = fake_source_manifest()
            invalid['distributions'][0]['files'][0]['path'] = invalid_path
            with self.subTest(path=invalid_path), self.assertRaises(ValueError):
                module.validate_source_manifest(invalid)
        empty_digest = canonical_fingerprint({})
        with self.assertRaises(ValueError):
            module.check_profile_source_identity(empty_digest, empty_digest, {})
        duplicate = fake_source_manifest()
        duplicate['distributions'][1]['files'][0]['path'] = duplicate['distributions'][0]['files'][0]['path']
        duplicate_digest = canonical_fingerprint(duplicate)
        with self.assertRaises(ValueError):
            module.check_profile_source_identity(duplicate_digest, duplicate_digest, duplicate)
        for key_hash, provenance_hash in ((('a' * 64), digest), (digest, ('b' * 64))):
            with self.subTest(key_hash=key_hash, provenance_hash=provenance_hash), self.assertRaises(ValueError):
                module.check_profile_source_identity(key_hash, provenance_hash, manifest)


    def test_approved_bootstrap_profile_requires_independent_pin_and_matching_source(self):
        module = importlib.import_module('looking_glass.live_profile')
        profiles = importlib.import_module('looking_glass.resource_profile')
        from test_resource_profile import workload, sample, measured_profile
        source = fake_source_manifest()
        raw = json.dumps(source, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        key = workload(profiles, source_sha256=profiles.canonical_fingerprint(source))
        profile = measured_profile(profiles, key, (sample(profiles, key=key),), source_manifest=raw)
        registry = profiles.ProfileRegistry((profile,), approved_profile_ids=(profile.profile_id,))
        self.assertIs(module.resolve_bootstrap_profile(registry, key, source, profile.profile_id), profile)
        with self.assertRaises(ValueError):
            module.resolve_bootstrap_profile(registry, key, source, 'a' * 64)
        bad = measured_profile(profiles, key, (sample(profiles, key=key),), source_manifest=b'other')
        bad_registry = profiles.ProfileRegistry((bad,), approved_profile_ids=(bad.profile_id,))
        with self.assertRaises(ValueError):
            module.resolve_bootstrap_profile(bad_registry, key, source, bad.profile_id)


    def test_repo_source_manifest_hashes_committed_bytes_and_rejects_dirty_or_untracked_code(self):
        module = importlib.import_module('looking_glass.live_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'looking_glass').mkdir()
            (root / 'docs').mkdir()
            files = {'looking_glass/__init__.py': b'', 'looking_glass/gpu_bootstrap.py': b'pass\n',
                     'looking_glass/gpu_service.py': b'pass\n',
                     'looking_glass/live_profile.py': b'pass\n',
                     'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md': b'protocol v1\n'}
            for name, data in files.items():
                (root / name).write_bytes(data)
            def git(*args):
                subprocess.run(['git', '-C', str(root), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git('init')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            git('add', 'looking_glass', 'docs')
            git('commit', '-m', 'fixture')
            with self.assertRaises(ValueError):
                module.collect_repository_source(root)
            git('checkout', '--detach')
            repo = module.collect_repository_source(root)
            self.assertEqual([entry['path'] for entry in repo['repository_files']], sorted(files))
            for entry in repo['repository_files']:
                data = files[entry['path']]
                self.assertEqual(entry['size'], len(data))
                self.assertEqual(entry['sha256'], hashlib.sha256(data).hexdigest())
            (root / 'looking_glass/gpu_service.py').write_bytes(b'changed\n')
            with self.assertRaises(ValueError):
                module.collect_repository_source(root)
            real_git = module._git
            def stale_status(path, *args):
                if args and args[0] == 'status':
                    return b''
                return real_git(path, *args)
            with patch.object(module, '_git', side_effect=stale_status):
                with self.assertRaises(ValueError):
                    module.collect_repository_source(root)
            git('checkout', '--', 'looking_glass/gpu_service.py')
            git('update-index', '--assume-unchanged', 'looking_glass/gpu_service.py')
            (root / 'looking_glass/gpu_service.py').write_bytes(b'hidden change\n')
            with self.assertRaises(ValueError):
                module.collect_repository_source(root)
            git('update-index', '--no-assume-unchanged', 'looking_glass/gpu_service.py')
            git('checkout', '--', 'looking_glass/gpu_service.py')
            (root / 'looking_glass/extra.py').write_bytes(b'pass\n')
            with self.assertRaises(ValueError):
                module.collect_repository_source(root)


    def test_distribution_manifest_hashes_actual_files_and_rejects_escape_or_missing(self):
        module = importlib.import_module('looking_glass.live_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / 'Lib' / 'site-packages'
            (site / 'package').mkdir(parents=True)
            (site / 'package' / '__init__.py').write_bytes(b'package bytes')
            class Distribution:
                files = ('package/__init__.py',)
                def locate_file(self, name):
                    return site / name
            dist = Distribution()
            result = module.collect_distribution_files(dist, site, root)
            self.assertEqual(result[0]['path'], 'Lib/site-packages/package/__init__.py')
            self.assertEqual(result[0]['sha256'], hashlib.sha256(b'package bytes').hexdigest())
            dist.files = ('package/missing.py',)
            with self.assertRaises(ValueError):
                module.collect_distribution_files(dist, site, root)
            dist.files = ('../../../outside.py',)
            with self.assertRaises(ValueError):
                module.collect_distribution_files(dist, site, root)


    def test_verified_asset_checks_actual_bytes_size_and_cache_boundary(self):
        module = importlib.import_module('looking_glass.live_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'model.safetensors'
            asset.write_bytes(b'fixed checkpoint bytes')
            expected = (len(asset.read_bytes()), hashlib.sha256(asset.read_bytes()).hexdigest())
            module.verify_asset_file(asset, root, *expected)
            asset.write_bytes(b'x')
            single_sha = hashlib.sha256(b'x').hexdigest()
            for bad_size in (True, 1.0, -1):
                with self.subTest(size=bad_size), self.assertRaises(ValueError):
                    module.verify_asset_file(asset, root, bad_size, single_sha)
            asset.write_bytes(b'fixed checkpoint bytes')
            asset.write_bytes(b'altered checkpoint')
            with self.assertRaises(ValueError):
                module.verify_asset_file(asset, root, *expected)
            with tempfile.TemporaryDirectory() as other:
                outside = Path(other) / 'outside'
                outside.write_bytes(b'fixed checkpoint bytes')
                with self.assertRaises(ValueError):
                    module.verify_asset_file(outside, root, *expected)


    def test_repo_manifest_rejects_commit_changed_during_hashing(self):
        module = importlib.import_module('looking_glass.live_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'looking_glass').mkdir()
            (root / 'docs').mkdir()
            for name in ('looking_glass/__init__.py', 'looking_glass/gpu_bootstrap.py', 'looking_glass/gpu_service.py',
                         'looking_glass/live_profile.py', 'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md'):
                (root / name).write_bytes(b'old\n')
            def git(*args):
                subprocess.run(['git', '-C', str(root), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git('init')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            git('add', 'looking_glass', 'docs')
            git('commit', '-m', 'first')
            git('checkout', '--detach')
            original = module._capture_source_bytes
            changed = [False]
            def concurrent_commit(path):
                if path.name == 'gpu_service.py' and not changed[0]:
                    changed[0] = True
                    path.write_bytes(b'new\n')
                    git('add', 'looking_glass/gpu_service.py')
                    git('commit', '-m', 'second')
                return original(path)
            with patch.object(module, '_capture_source_bytes', side_effect=concurrent_commit):
                with self.assertRaises(ValueError):
                    module.collect_repository_source(root)
            self.assertTrue(changed[0])


    def test_startup_cache_policy_requires_empty_matching_directory_and_no_writes(self):
        module = importlib.import_module('looking_glass.live_profile')
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'pycache'
            prefix.mkdir()
            module.check_import_cache_policy(str(prefix), str(prefix), True)
            for actual, no_write in ((None, True), (str(prefix / 'other'), True), (str(prefix), False)):
                with self.subTest(actual=actual, no_write=no_write), self.assertRaises(ValueError):
                    module.check_import_cache_policy(str(prefix), actual, no_write)
            (prefix / 'stale.pyc').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                module.check_import_cache_policy(str(prefix), str(prefix), True)

    def test_fresh_cache_prefix_does_not_reuse_stale_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'stale_module.py'
            script.write_text("VALUE='evil'\n", encoding='utf-8')
            import py_compile
            py_compile.compile(str(script), doraise=True)
            saved = script.stat()
            script.write_text("VALUE='good'\n", encoding='utf-8')
            os.utime(script, ns=(saved.st_atime_ns, saved.st_mtime_ns))
            prefix = root / 'new-empty-cache'
            prefix.mkdir()
            env = os.environ.copy()
            env.update({'PYTHONPYCACHEPREFIX': str(prefix), 'PYTHONDONTWRITEBYTECODE': '1',
                        'PYTHONNOUSERSITE': '1'})
            output = subprocess.check_output([sys._base_executable, '-c',
                'import sys,stale_module; print(stale_module.VALUE); print(sys.dont_write_bytecode)'],
                cwd=root, env=env, text=True)
            self.assertEqual(output.splitlines(), ['good', 'True'])
            self.assertEqual(list(prefix.iterdir()), [])


    def test_repo_manifest_hash_is_bound_to_same_bytes_as_committed_blob(self):
        module = importlib.import_module('looking_glass.live_profile')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'looking_glass').mkdir()
            (root / 'docs').mkdir()
            for name in ('looking_glass/__init__.py', 'looking_glass/gpu_bootstrap.py', 'looking_glass/gpu_service.py',
                         'looking_glass/live_profile.py', 'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md'):
                (root / name).write_bytes(b'old\n')
            def git(*args):
                subprocess.run(['git', '-C', str(root), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git('init')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            git('add', 'looking_glass', 'docs')
            git('commit', '-m', 'first')
            git('checkout', '--detach')
            original = module._capture_source_bytes
            changed = [False]
            def transient_write(path):
                if path.name == 'gpu_service.py' and not changed[0]:
                    changed[0] = True
                    path.write_bytes(b'new\n')
                    data = original(path)
                    path.write_bytes(b'old\n')
                    return data
                return original(path)
            with patch.object(module, '_capture_source_bytes', side_effect=transient_write):
                with self.assertRaises(ValueError):
                    module.collect_repository_source(root)
            self.assertTrue(changed[0])


    def test_preflight_assembles_canonical_actual_source_and_asset_manifests(self):
        module = importlib.import_module('looking_glass.live_profile')
        source = fake_source_manifest()
        repo = {key: source[key] for key in ('git_commit', 'git_tree', 'repository_files')}
        asset_manifest = {'schema': 'resource-bootstrap-assets-v1', 'model_id': 'model',
                          'revision': 'revision', 'assets': []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / 'site'
            site.mkdir()
            with patch.object(module, 'collect_repository_source', return_value=repo), \
                 patch.object(module, 'collect_installed_distributions', return_value=source['distributions']), \
                 patch.object(module, 'collect_offline_assets', return_value=(asset_manifest, ())), \
                 patch.object(module, '_plain_file_hash', return_value=source['python_executable']):
                result = module.preflight_sources(root, root, site, root)
            self.assertEqual(json.loads(result.source_manifest), source)
            self.assertEqual(json.loads(result.asset_manifest), asset_manifest)
            self.assertEqual(result.source_sha256, hashlib.sha256(result.source_manifest).hexdigest())

    def test_distribution_rejects_intermediate_link_even_before_parent_component(self):
        module = importlib.import_module('looking_glass.live_profile')
        import stat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / 'Lib' / 'site-packages'
            site.mkdir(parents=True)
            alias = site / 'alias'
            alias.mkdir()
            (site / 'file.py').write_text('pass', encoding='utf-8')
            class Distribution:
                files = ['alias/../file.py']
                def locate_file(self, name): return site / name
            original = Path.lstat
            def fake_lstat(path):
                if path == alias:
                    return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
                return original(path)
            with patch.object(Path, 'lstat', fake_lstat):
                with self.assertRaises(ValueError):
                    module.collect_distribution_files(Distribution(), site, root)

            # The selected site itself must not hide an alias through '..'.
            hidden_site = root / 'alias' / '..' / 'Lib' / 'site-packages'
            (root / 'alias').mkdir()
            with self.assertRaises(ValueError):
                module.collect_distribution_files(Distribution(), hidden_site, root)

            for record in ('alias\\..\\file.py', 'package:module.py', 'bad\x00name'):
                with self.subTest(record=record), self.assertRaises(ValueError):
                    class InvalidDistribution:
                        files = [record]
                        def locate_file(self, name): return site / name
                    module.collect_distribution_files(InvalidDistribution(), site, root)


    def test_cpu_model_state_requires_exact_weights_buffers_and_tied_embedding(self):
        module = importlib.import_module('looking_glass.live_profile')
        import torch
        base = torch.tensor([[1.0, 2.0]])
        weight = torch.tensor([[0.0, 4.0], [5.0, 6.0]])
        state = {'transformer.wte.weight': base, 'lm_head.weight': base,
                 'transformer.h.0.mlp.weight': weight}
        buffers = {}
        expected = {'transformer.wte.weight': base.clone(),
                    'transformer.h.0.mlp.weight': weight.clone()}
        for index in range(6):
            prefix = f'transformer.h.{index}.attn.'
            buffers[prefix + 'bias'] = torch.tensor([[[[True, False], [True, True]]]])
            buffers[prefix + 'masked_bias'] = torch.tensor(-10000.0)
            expected[prefix + 'bias'] = buffers[prefix + 'bias'].float()
        class Model:
            def state_dict(self): return state
            def get_buffer(self, name): return buffers[name]
            def named_buffers(self): return list(buffers.items())
        class Weights:
            def keys(self): return list(expected)
            def get_tensor(self, name): return expected[name]
        self.assertIs(module._verify_cpu_tensors(torch, Model(), Weights()), True)
        state['transformer.h.0.mlp.weight'] = torch.tensor([[4.0, 4.0], [5.0, 6.0]])
        with self.assertRaises(ValueError):
            module._verify_cpu_tensors(torch, Model(), Weights())
        state['transformer.h.0.mlp.weight'] = weight.clone()
        state['transformer.h.0.mlp.weight'][0, 0] = -0.0
        self.assertTrue(torch.equal(state['transformer.h.0.mlp.weight'], weight))
        with self.assertRaises(ValueError):
            module._verify_cpu_tensors(torch, Model(), Weights())
        state['transformer.h.0.mlp.weight'] = weight.T.contiguous().T
        self.assertTrue(torch.equal(state['transformer.h.0.mlp.weight'], weight))
        with self.assertRaises(ValueError):
            module._verify_cpu_tensors(torch, Model(), Weights())
        oversized = torch.cat([weight.flatten(), torch.zeros(1)])
        state['transformer.h.0.mlp.weight'] = oversized[:4].view(2, 2)
        with self.assertRaises(ValueError):
            module._verify_cpu_tensors(torch, Model(), Weights())
        state['transformer.h.0.mlp.weight'] = weight
        state['lm_head.weight'] = base.clone()
        with self.assertRaises(ValueError):
            module._verify_cpu_tensors(torch, Model(), Weights())
        state['lm_head.weight'] = base
        buffers['transformer.h.0.attn.masked_bias'] = torch.tensor(-2.0)
        with self.assertRaises(ValueError):
            module._verify_cpu_tensors(torch, Model(), Weights())


    def test_runtime_setup_verifies_uuid_and_integer_allocator_cap_with_cpu_fakes(self):
        module = importlib.import_module('looking_glass.live_profile')
        calls = []
        total = 8 * GIB
        props = SimpleNamespace(uuid=FakeCudaUuid(UUID[4:]), name='GPU model', major=7, minor=5)
        class FakeCuda:
            def device_count(self): calls.append('count'); return 1
            def get_device_properties(self, index): calls.append('properties'); return props
            def mem_get_info(self, index): calls.append('telemetry'); return 6 * GIB, total
            def get_allocator_backend(self): return 'native'
            def set_per_process_memory_fraction(self, fraction, device):
                self.fraction, self.device = fraction, device; calls.append('cap')
            def get_per_process_memory_fraction(self, device): return int(0.2 * total) / total
        class FakeTorch:
            __version__ = '2.8.0+cu128'
            version = SimpleNamespace(cuda='12.8')
            _C = SimpleNamespace(_CUuuid=FakeCudaUuid)
            cuda = FakeCuda()
            backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(enabled=False, benchmark=True, deterministic=False,
                                      allow_tf32=True, version=lambda: 91000))
            threads = 3
            interop = 2
            deterministic = False
            precision = 'medium'
            def set_num_threads(self, value): self.threads = value
            def get_num_threads(self): return self.threads
            def set_num_interop_threads(self, value): self.interop = value
            def get_num_interop_threads(self): return self.interop
            def use_deterministic_algorithms(self, value): self.deterministic = value
            def are_deterministic_algorithms_enabled(self): return self.deterministic
            def set_float32_matmul_precision(self, value): self.precision = value
            def get_float32_matmul_precision(self): return self.precision
        fake = FakeTorch()
        identity = SimpleNamespace(gpu_uuid=UUID)
        with patch.object(module, '_guard_owner'), patch.object(module.OwnerSession, 'verify_runtime_binding') as verify, \
             patch.object(module, '_driver_version', return_value='616.92'):
            runtime = module._measure_runtime(fake, identity, object(), object())
        self.assertEqual(runtime['device_uuid'], UUID)
        self.assertEqual(runtime['allocator_effective_limit_bytes'], int(0.2 * total))
        self.assertEqual(runtime['execution_path'], 'looking_glass.resident_worker.ResidentWorker.execute')
        self.assertTrue(runtime['deterministic_algorithms'])
        self.assertEqual(fake.cuda.fraction, 0.2)
        self.assertLess(calls.index('properties'), calls.index('cap'))
        verify.assert_called_once()
        props.uuid = FakeCudaUuid('bad')
        with patch.object(module, '_guard_owner'), patch.object(module.OwnerSession, 'verify_runtime_binding'), \
             patch.object(module, '_driver_version', return_value='616.92'):
            with self.assertRaises(ValueError):
                module._measure_runtime(fake, identity, object(), object())

        calls.clear()
        props.uuid = FakeCudaUuid(UUID[4:])
        with patch.object(module, '_guard_owner', side_effect=[None, RuntimeError('lost marker')]), \
             patch.object(module.OwnerSession, 'verify_runtime_binding'):
            with self.assertRaisesRegex(RuntimeError, 'lost marker'):
                module._measure_runtime(fake, identity, object(), object())
        self.assertEqual(calls, ['count'])
        calls.clear()
        total = 6 * GIB
        with patch.object(module, '_guard_owner'), patch.object(module.OwnerSession, 'verify_runtime_binding'), \
             patch.object(module, '_driver_version', return_value='616.92'):
            with self.assertRaises(ValueError):
                module._measure_runtime(fake, identity, object(), object())
        self.assertNotIn('cap', calls)

    def test_model_architecture_rejects_changed_attention_norm_hooks_and_forward(self):
        module = importlib.import_module('looking_glass.live_profile')
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=2, n_head=2)
        config._attn_implementation = 'eager'
        with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            actual = GPT2LMHeadModel(config).eval()
            with torch.device('meta'):
                reference = GPT2LMHeadModel(config).eval()
        self.assertIs(module._verify_architecture(torch, actual, reference), True)
        attn = actual.transformer.h[0].attn
        for name, value in (('scale_attn_weights', False), ('layer_idx', 12),
                            ('reorder_and_upcast_attn', True), ('num_heads', 1),
                            ('is_causal', False)):
            old = getattr(attn, name)
            setattr(attn, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                module._verify_architecture(torch, actual, reference)
            setattr(attn, name, old)
        norm = actual.transformer.h[0].ln_1
        norm.eps = 0.5
        with self.assertRaises(ValueError):
            module._verify_architecture(torch, actual, reference)
        norm.eps = reference.transformer.h[0].ln_1.eps
        original = actual.transformer.h[0].mlp.act
        actual.transformer.h[0].mlp.act = torch.nn.ReLU()
        with self.assertRaises(ValueError):
            module._verify_architecture(torch, actual, reference)
        actual.transformer.h[0].mlp.act = original
        attn.forward = lambda *args, **kwargs: None
        with self.assertRaises(ValueError):
            module._verify_architecture(torch, actual, reference)
        del attn.forward
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
        try:
            with self.assertRaises(ValueError):
                module._verify_architecture(torch, actual, reference)
        finally:
            handle.remove()
        self.assertIs(module._verify_architecture(torch, actual, reference), True)

    def test_owner_guard_rejects_fake_before_any_runtime_query(self):
        module = importlib.import_module('looking_glass.live_profile')
        with patch.object(module, '_load_torch', side_effect=AssertionError('CUDA runtime was reached')):
            with self.assertRaises((TypeError, ValueError)):
                module.build_live_keys(None, None, None, None)

    def test_key_builder_binds_exact_source_runtime_and_three_operations(self):
        module = importlib.import_module('looking_glass.live_profile')
        runtime = {'schema': 'resource-bootstrap-runtime-v1', 'device': 'cuda:0',
                   'device_uuid': UUID, 'device_name': 'GPU model', 'device_total_bytes': 8 * GIB,
                   'compute_capability': [7, 5], 'python_version': '3.11.15',
                   'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
                   'cuda_runtime_version': '12.8', 'driver_version': '616.92',
                   'model_dtype': 'torch.float32', 'compute_dtype': 'torch.float32',
                   'attention_backend': 'eager', 'model_mode': 'eval',
                   'grad_mode': 'inference', 'autocast': 'disabled', 'use_cache': False,
                   'compiled': False, 'allocator_fraction': '0.20'}
        load, cold, warm = module._make_workload_keys('a' * 64, runtime)
        self.assertEqual((load.operation, load.batch_size, load.input_dtypes), ('model_load', 0, ()))
        self.assertEqual((cold.operation, cold.batch_size, cold.sequence_length, cold.input_dtypes),
                         ('forward', 1, 2, (('input_ids', 'torch.int64'),)))
        self.assertEqual((cold.warmup_state, warm.warmup_state), ('cold', 'warm'))
        self.assertEqual(cold.output_limit_bytes, 512 * 1024)
        self.assertEqual(load.model_id, cold.model_id)
        self.assertEqual(cold.model_id, warm.model_id)
        self.assertNotEqual(cold.fingerprint(), warm.fingerprint())
        changed = module._make_workload_keys('b' * 64, runtime)
        self.assertNotEqual(cold.model_id, changed[1].model_id)
        changed_runtime = module._make_workload_keys('a' * 64, {**runtime, 'driver_version': '617.1'})
        self.assertNotEqual(cold.model_id, changed_runtime[1].model_id)


if __name__ == '__main__':
    unittest.main()
