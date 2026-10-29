import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch


class DistilGPT2BatchCpuV1Tests(unittest.TestCase):
    def setUp(self):
        from looking_glass import distilgpt2_batch_cpu_v1 as experiment
        self.e = experiment
        self.runtime = {'test-runtime': 'fixed'}
        self.sources = {'test.py': 'a' * 64}

    def test_frozen_cases_and_profile_are_exact_and_disjoint(self):
        manifest = self.e.freeze(self.runtime, self.sources, source_commit='b' * 40)
        protocol = manifest['primary_protocol']
        binding = protocol['binding']
        self.assertEqual(binding['device'], 'cpu')
        self.assertEqual(binding['input_shape'], [1, 4])
        self.assertEqual(binding['approved_rows'], [2])
        self.assertEqual((binding['serial_atol'], binding['serial_rtol']), (1e-6, 1e-5))
        self.assertEqual(binding['output_width'], 50257)
        self.assertEqual(binding['seed'], 2026102808)
        self.assertEqual(binding['runtime_id'], 'sha256:' + self.e.digest(self.runtime))
        groups = self.e.groups()
        self.assertEqual([[c['inputs']['input_ids'].tolist()[0] for c in g['cases']] for g in groups], [
            [[50256, 103, 203, 303], [50256, 104, 204, 304]],
            [[50256, 105, 205, 305], [50256, 106, 206, 306]],
        ])
        hashes = [h for group in protocol['groups'] for c in group['cases'] for h in c['row_sha256']]
        self.assertEqual(len(set(hashes)), 4)
        self.assertFalse(set(hashes) & set(protocol['discovery_row_hashes']))
        self.assertNotEqual(protocol['binding']['adapter_sha256'],
                            manifest['reversal_protocol']['binding']['adapter_sha256'])
        self.assertNotEqual(protocol['binding']['model_id'],
                            manifest['batch_mean_protocol']['binding']['model_id'])

    def test_freeze_snapshots_runtime_and_sources(self):
        manifest = self.e.freeze(self.runtime, self.sources, source_commit='b' * 40)
        self.runtime['test-runtime'] = 'changed'
        self.sources['test.py'] = 'c' * 64
        self.assertEqual(manifest['runtime']['test-runtime'], 'fixed')
        self.assertEqual(manifest['source_sha256']['test.py'], 'a' * 64)

    def test_model_entry_observation_and_profiler_restoration(self):
        from looking_glass.resident_worker import ResidentWorker

        class Tiny(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return input_ids.to(torch.float32).unsqueeze(-1)

        model = Tiny().eval()
        worker = ResidentWorker('tiny', model, device='cpu', max_batch_size=2,
                                max_sequence_length=4, max_input_bytes=128, max_output_bytes=32)
        def existing(frame, event, arg):
            return None
        before = sys.getprofile()
        try:
            sys.setprofile(existing)
            execute = self.e.ObservedExecutor(worker, model, output_width=1, cap_bytes=64)
            logits = execute(self.e.inputs([50256, 103, 203, 303]), request_id='serial-0-0', seed=1)
            self.assertIs(sys.getprofile(), existing)
            self.assertEqual(tuple(logits.shape), (1, 4, 1))
            self.assertEqual(execute.calls[0]['input_ids_shape'], [1, 4])
            self.assertEqual(execute.calls[0]['attention_mask_shape'], [1, 4])
            self.assertEqual(execute.calls[0]['input_ids_dtype'], 'torch.int64')
            self.assertEqual(execute.calls[0]['input_ids_device'], 'cpu')
            self.assertEqual(execute.calls[0]['input_ids_values'], [[50256, 103, 203, 303]])
            self.assertEqual(execute.calls[0]['attention_mask_values'], [[1, 1, 1, 1]])
            self.assertIn('serial-0-0', execute.outputs)
        finally:
            sys.setprofile(before)
            worker.close()

    def test_model_entry_absent_is_an_error_and_profiler_is_restored(self):
        class FakeModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return input_ids.float().unsqueeze(-1)
        class ImpostorWorker:
            def execute(self, request):
                from looking_glass.resident_worker import ForwardResult
                return ForwardResult(request.request_id, request.inputs['input_ids'].float().unsqueeze(-1))
        before = sys.getprofile()
        execute = self.e.ObservedExecutor(ImpostorWorker(), FakeModel(), output_width=1, cap_bytes=64)
        with self.assertRaisesRegex(RuntimeError, 'model-entry'):
            execute(self.e.inputs([50256, 103, 203, 303]), request_id='serial-0-0', seed=1)
        self.assertIs(sys.getprofile(), before)
        self.assertEqual(execute.outputs, {})

    def test_model_entry_input_transformation_is_detected(self):
        from looking_glass.resident_worker import ForwardResult
        class Tiny(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return input_ids.float().unsqueeze(-1)
        model = Tiny().eval()
        class MisroutingWorker:
            def execute(self, request):
                changed = dict(request.inputs)
                changed['input_ids'] = changed['input_ids'] + 1
                return ForwardResult(request.request_id, model(**changed))
        execute = self.e.ObservedExecutor(MisroutingWorker(), model, output_width=1, cap_bytes=64)
        with self.assertRaisesRegex(RuntimeError, 'model-entry'):
            execute(self.e.inputs([50256, 103, 203, 303]), request_id='serial-0-0', seed=1)
        self.assertEqual(execute.calls[0]['input_ids_values'], [[50257, 104, 204, 304]])

    def test_raw_outputs_are_bounded_and_round_trip_with_digests(self):
        import numpy as np
        outputs = {'serial-0-0': torch.tensor([[[1.25], [-2.0], [0.0], [3.5]]],
                                               dtype=torch.float32)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'outputs.zip'
            info = self.e.save_outputs(path, outputs, cap_bytes=64)
            with zipfile.ZipFile(path) as archive:
                raw = archive.read('serial-0-0.f32')
            self.assertEqual(info['raw_bytes'], 16)
            self.assertEqual(info['entries'][0]['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(np.frombuffer(raw, dtype='<f4').tolist(), [1.25, -2.0, 0.0, 3.5])
            with self.assertRaises(FileExistsError):
                self.e.save_outputs(path, outputs, cap_bytes=64)
            with self.assertRaises(ValueError):
                self.e.save_outputs(Path(directory) / 'too-small.zip', outputs, cap_bytes=15)

    def test_state_digest_detects_mutation_and_is_repeatable(self):
        model = torch.nn.Linear(4, 2, device='cpu').eval()
        first = self.e.state_sha256(model)
        self.assertEqual(first, self.e.state_sha256(model))
        with torch.no_grad():
            model.weight[0, 0] += 1
        self.assertNotEqual(first, self.e.state_sha256(model))

    def test_bootstrap_rejects_relative_cache_prefix(self):
        bootstrap = self.e.ROOT / 'tools' / 'run_distilgpt2_batch_cpu_v1.py'
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-X',
                                 'pycache_prefix=relative-missing-prefix', str(bootstrap), '--help'],
                                cwd=self.e.ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('pycache_prefix', result.stderr)

    def test_bootstrap_requires_gpu_hidden_before_importing_dependencies(self):
        bootstrap = self.e.ROOT / 'tools' / 'run_distilgpt2_batch_cpu_v1.py'
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'missing-cache')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
            result = subprocess.run([sys.executable, '-I', '-S', '-B', '-X',
                                     f'pycache_prefix={prefix}', str(bootstrap), '--help'],
                                    cwd=self.e.ROOT, capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CUDA_VISIBLE_DEVICES=-1 before importing dependencies', result.stderr)

    def test_playback_control_rejects_row_swap_and_batch_mean(self):
        manifest = self.e.freeze(self.runtime, self.sources, source_commit='b' * 40)
        outputs = {}
        for index, group in enumerate(self.e.groups()):
            values = []
            for case_index, case in enumerate(group['cases']):
                scalar = float(case_index + 1)
                output = torch.full((1, 4, 50257), scalar, dtype=torch.float32)
                outputs[f'serial-{index}-{case_index}'] = output
                values.append(output)
            outputs[f'mixed-{index}'] = torch.cat(values, dim=0)
        controls = self.e.run_controls(manifest, outputs)
        self.assertEqual(controls['positive_playback']['status'], 'passed')
        self.assertEqual(controls['batch_mean']['status'], 'failed')
        self.assertEqual(controls['reversed_playback']['status'], 'failed')
        self.assertTrue(controls['batch_mean']['complete'])
        self.assertTrue(controls['reversed_playback']['complete'])

    def test_manifest_source_and_runtime_drift_are_rejected(self):
        manifest = self.e.freeze(self.runtime, self.sources, source_commit='b' * 40)
        self.e.verify_manifest(manifest, self.runtime, self.sources)
        with self.assertRaises(ValueError):
            self.e.verify_manifest(manifest, {'test-runtime': 'changed'}, self.sources)
        with self.assertRaises(ValueError):
            self.e.verify_manifest(manifest, self.runtime, {'test.py': 'c' * 64})
        changed = copy.deepcopy(manifest)
        changed['primary_protocol']['binding']['serial_atol'] = 100.0
        with self.assertRaises(ValueError):
            self.e.verify_manifest(changed, self.runtime, self.sources)
        with self.assertRaises(ValueError):
            self.e.verify_git_sources('b' * 40)

    def test_source_check_detects_assume_unchanged_disk_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q')
            (root / 'subject.py').write_bytes(b'value = 1\n')
            git('add', 'subject.py')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '-qm', 'source')
            commit = git('rev-parse', 'HEAD')
            with patch.object(self.e, 'ROOT', root), patch.object(self.e, 'SOURCE_PATHS', ('subject.py',)):
                self.e.verify_git_sources(commit)
                git('update-index', '--assume-unchanged', 'subject.py')
                (root / 'subject.py').write_bytes(b'value = 2\n')
                self.assertEqual(git('status', '--porcelain'), '')
                with self.assertRaisesRegex(ValueError, 'on-disk source'):
                    self.e.verify_git_sources(commit)

    def test_manifest_is_required_to_match_a_later_ancestor_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q')
            (root / 'source.py').write_bytes(b'value = 1\n')
            git('add', 'source.py')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '-qm', 'source')
            source = git('rev-parse', 'HEAD')
            manifest = root / 'frozen.json'
            manifest.write_bytes(b'{"frozen":true}\n')
            git('add', 'frozen.json')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '-qm', 'freeze')
            frozen = git('rev-parse', 'HEAD')
            with patch.object(self.e, 'ROOT', root):
                self.assertEqual(self.e.verify_frozen_manifest(manifest, frozen, source), 'frozen.json')
                with self.assertRaises(ValueError):
                    self.e.verify_frozen_manifest(manifest, source, source)
                git('update-index', '--assume-unchanged', 'frozen.json')
                manifest.write_bytes(b'{"frozen":false}\n')
                with self.assertRaises(ValueError):
                    self.e.verify_frozen_manifest(manifest, frozen, source)
                git('update-index', '--no-assume-unchanged', 'frozen.json')
                git('checkout', '--', 'frozen.json')
                (root / 'later.txt').write_bytes(b'later\n')
                git('add', 'later.txt')
                git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'later')
                with self.assertRaises(ValueError):
                    self.e.verify_frozen_manifest(manifest, git('rev-parse', 'HEAD'), frozen)

    def test_runtime_snapshot_does_not_publish_interpreter_paths(self):
        optional = {'numpy': SimpleNamespace(__version__='test'),
                    'transformers': SimpleNamespace(__version__='4.57.3')}
        with patch.dict(sys.modules, optional), patch.object(torch, '__version__', '2.8.0+cu128'), \
             patch.object(torch.cuda, 'is_initialized', return_value=False), \
             patch.object(torch, 'is_autocast_enabled', return_value=False), \
             patch.object(self.e.validation, '_require_denormals'):
            info = self.e.runtime_info()
        self.assertNotIn('python_path', info)
        self.assertNotIn('pycache_prefix', info)
        self.assertNotIn('torch_build', info)
        self.assertEqual(len(info['torch_build_sha256']), 64)
        self.assertTrue(all(len(value) == 64 for value in info['python_path_sha256']))

    def test_incomplete_call_evidence_does_not_look_successful(self):
        self.assertFalse(self.e.observations_complete([]))
        expected = self.e.expected_calls()
        self.assertTrue(self.e.observations_complete(expected))
        changed = copy.deepcopy(expected)
        changed[2]['input_ids_shape'] = [1, 4]
        self.assertFalse(self.e.observations_complete(changed))
        self.assertFalse(self.e.observations_complete(expected[:-1]))

    def test_end_to_end_with_explicit_fake_model_and_archived_playback(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(use_cache=False, _attn_implementation='eager')
            def forward(self, input_ids, attention_mask):
                return input_ids.float().unsqueeze(-1).expand(-1, -1, 50257) * self.weight
        model = FakeModel().eval()
        metadata = {'model_id': self.e.MODEL_ID, 'model_revision': self.e.REVISION,
                    'asset_sha256': {k: v[1] for k, v in self.e.ASSETS.items()}}
        loaded = SimpleNamespace(model=model, tokenizer=SimpleNamespace(eos_token_id=50256),
                                 metadata=metadata)
        manifest = self.e.freeze(self.runtime, self.sources, source_commit='b' * 40)
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / 'run'
            with patch.object(self.e, 'verify_git_sources'), \
                 patch.object(self.e, 'verify_frozen_manifest', return_value='frozen.json'), \
                 patch.object(self.e, 'configure_runtime', return_value=self.runtime), \
                 patch.object(self.e, 'runtime_info', return_value=self.runtime), \
                 patch.object(self.e, 'source_hashes', return_value=self.sources), \
                 patch.object(self.e, 'load_distilgpt2', return_value=loaded):
                summary = self.e.execute_experiment(manifest, manifest_path='frozen.json',
                    manifest_commit='c' * 40, cache_dir=directory, output_dir=result_dir)
            self.assertEqual(summary['status'], 'passed')
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['output_archive']['raw_bytes'], 6432896)
            self.assertTrue(self.e.observations_complete(summary['observed_model_calls']))
            self.assertEqual(summary['model_state_sha256_before'], summary['model_state_sha256_after'])
            self.assertEqual(summary['controls']['positive_playback']['status'], 'passed')
            self.assertEqual(summary['controls']['reversed_playback']['status'], 'failed')
            self.assertEqual(summary['controls']['batch_mean']['status'], 'failed')
            report = json.loads((result_dir / 'primary-report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['evidence_sha256'], summary['primary_evidence_sha256'])
            with self.assertRaises(FileExistsError):
                self.e.execute_experiment(manifest, manifest_path='frozen.json',
                    manifest_commit='c' * 40, cache_dir=directory, output_dir=result_dir)
            from looking_glass.resident_worker import ResidentWorker
            failed_dir = Path(directory) / 'close-failure'
            with patch.object(self.e, 'verify_git_sources'), \
                 patch.object(self.e, 'verify_frozen_manifest', return_value='frozen.json'), \
                 patch.object(self.e, 'configure_runtime', return_value=self.runtime), \
                 patch.object(self.e, 'runtime_info', return_value=self.runtime), \
                 patch.object(self.e, 'source_hashes', return_value=self.sources), \
                 patch.object(self.e, 'load_distilgpt2', return_value=loaded), \
                 patch.object(ResidentWorker, 'close', side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError):
                    self.e.execute_experiment(manifest, manifest_path='frozen.json',
                        manifest_commit='c' * 40, cache_dir=directory, output_dir=failed_dir)
            self.assertFalse((failed_dir / 'summary.json').exists())
            failure = json.loads((failed_dir / 'failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['status'], 'incomplete')
            self.assertEqual(failure['exception_type'], 'RuntimeError')
            published_dir = Path(directory) / 'publication-interrupt'
            publish = self.e.write_summary_atomic
            def publish_then_interrupt(path, data):
                publish(path, data)
                raise KeyboardInterrupt()
            with patch.object(self.e, 'verify_git_sources'), \
                 patch.object(self.e, 'verify_frozen_manifest', return_value='frozen.json'), \
                 patch.object(self.e, 'configure_runtime', return_value=self.runtime), \
                 patch.object(self.e, 'runtime_info', return_value=self.runtime), \
                 patch.object(self.e, 'source_hashes', return_value=self.sources), \
                 patch.object(self.e, 'load_distilgpt2', return_value=loaded), \
                 patch.object(self.e, 'write_summary_atomic', side_effect=publish_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.e.execute_experiment(manifest, manifest_path='frozen.json',
                        manifest_commit='c' * 40, cache_dir=directory, output_dir=published_dir)
            summary = json.loads((published_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'passed')
            self.assertFalse((published_dir / 'failure.json').exists())


if __name__ == '__main__':
    unittest.main()
