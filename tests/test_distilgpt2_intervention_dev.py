"""Synthetic tests of the dev-only intervention smoke; never loads a checkpoint."""

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import unittest
from unittest import mock

import torch


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'transformers is optional')
class InterventionDevelopmentTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def make_model(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=41, n_positions=8, n_embd=16, n_layer=6, n_head=4,
                            use_cache=False, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
        config._attn_implementation = 'eager'
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1729)
            return GPT2LMHeadModel(config).eval()

    def test_manual_path_and_plan_controls_agree_on_a_synthetic_model(self):
        from looking_glass.distilgpt2_intervention_dev import evaluate
        model = self.make_model()
        result = evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        self.assertGreater(result['injection']['max_logit_change'], 1e-5)
        self.assertTrue(result['injection']['other_coordinates_bit_equal'])
        self.assertTrue(result['injection']['target_matches_expected_add_bits'])
        self.assertTrue(result['injection']['clean_donor_bit_equal_after_patch'])
        self.assertTrue(result['hook_registries_unchanged'])
        self.assertTrue(result['model_state_unchanged'])
        self.assertTrue(all(result['controls'].values()))
        self.assertTrue(all(item['within_tolerance'] for item in result['comparisons'].values()))
        self.assertIn('injected_preceding_logits', result['comparisons'])
        self.assertIn('patched_preceding_logits', result['comparisons'])
        self.assertIn('manual_injected', result['outputs'])
        self.assertIn('plan_injected', result['outputs'])

    def test_a_model_that_masks_the_injection_fails_the_smoke(self):
        from looking_glass.distilgpt2_intervention_dev import SmokeFailure, evaluate
        model = self.make_model()
        with torch.no_grad():
            model.lm_head.weight.zero_()
        with self.assertRaises(SmokeFailure):
            evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))

    def test_tampered_manual_reference_cannot_pass(self):
        from looking_glass import distilgpt2_intervention_dev as smoke
        model = self.make_model()
        original = smoke.manual_forward
        def tampered(*args, **kwargs):
            logits, vectors = original(*args, **kwargs)
            return logits + 0.01, vectors
        with mock.patch.object(smoke, 'manual_forward', side_effect=tampered):
            with self.assertRaises(smoke.SmokeFailure):
                smoke.evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

    def test_token_locality_control_rejects_shared_path_leakage(self):
        from looking_glass.distilgpt2_intervention_dev import SmokeFailure, evaluate
        model = self.make_model()
        block = model.transformer.h[3]
        original = block.forward
        def leaky_forward(hidden_states, *args, **kwargs):
            changed = hidden_states.clone()
            changed[:, 1, :] += hidden_states[:, 2, :]
            return original(changed, *args, **kwargs)
        block.forward = leaky_forward
        with self.assertRaisesRegex(SmokeFailure, 'injected_preceding_logits'):
            evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

    def test_preexisting_hooks_are_rejected_before_any_forward(self):
        from looking_glass.distilgpt2_intervention_dev import SmokeFailure, evaluate
        model = self.make_model()
        seen = []
        handle = model.transformer.h[0].register_forward_hook(lambda *args: seen.append(True))
        try:
            with self.assertRaises(SmokeFailure):
                evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
            self.assertEqual(seen, [])
        finally:
            handle.remove()
        global_handle = torch.nn.modules.module.register_module_forward_hook(
            lambda *args: seen.append(True))
        try:
            with self.assertRaises(SmokeFailure):
                evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
            self.assertEqual(seen, [])
        finally:
            global_handle.remove()

    def test_primary_error_is_not_masked_when_hook_cleanup_is_uncertain(self):
        from looking_glass import distilgpt2_intervention_dev as smoke
        model = self.make_model()
        leaked = []
        report = {}
        def failed_entry(plan):
            leaked.append(model.transformer.h[2].register_forward_hook(lambda *args: None))
            raise ValueError('primary plan failure')
        try:
            with mock.patch.object(smoke.ActivationPlan, '__enter__', failed_entry):
                with self.assertRaisesRegex(ValueError, 'primary plan failure') as caught:
                    smoke.evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long), report)
            self.assertFalse(report['hook_registries_unchanged'])
            if hasattr(caught.exception, '__notes__'):
                self.assertTrue(any('contaminated model' in note for note in caught.exception.__notes__))
        finally:
            for handle in leaked:
                handle.remove()

    def test_primary_error_survives_failed_cleanup_inspection(self):
        from looking_glass import distilgpt2_intervention_dev as smoke
        model = self.make_model()
        original_state = smoke._hook_state
        failed = [False]
        report = {}
        def failed_entry(plan):
            failed[0] = True
            raise ValueError('primary forward failure')
        def inspected(model):
            if failed[0]:
                raise MemoryError('cannot inspect hooks')
            return original_state(model)
        with mock.patch.object(smoke.ActivationPlan, '__enter__', failed_entry), \
                mock.patch.object(smoke, '_hook_state', side_effect=inspected):
            with self.assertRaisesRegex(ValueError, 'primary forward failure') as caught:
                smoke.evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long), report)
        self.assertIsNone(report['hook_registries_unchanged'])
        if hasattr(caught.exception, '__notes__'):
            self.assertTrue(any('inspection' in note and 'discard' in note
                                for note in caught.exception.__notes__))

    def test_addition_that_rounds_away_is_rejected(self):
        from looking_glass.distilgpt2_intervention_dev import SmokeFailure, _target_delta
        before = torch.tensor([float(2**25)], dtype=torch.float32)
        after = before + 1.0
        self.assertEqual(before.view(torch.int32).item(), after.view(torch.int32).item())
        with self.assertRaisesRegex(SmokeFailure, 'did not increase'):
            _target_delta(before, after)

    def test_training_submodule_and_state_change_are_detected(self):
        from looking_glass.distilgpt2_intervention_dev import SmokeFailure, _model_state_sha256, evaluate
        model = self.make_model()
        model.transformer.h[2].train()
        with self.assertRaises(SmokeFailure):
            evaluate(model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        model.eval()
        before = _model_state_sha256(model)
        with torch.no_grad():
            model.transformer.wte.weight[0, 0] += 1.0
        self.assertNotEqual(before, _model_state_sha256(model))
        before = _model_state_sha256(model)
        with torch.no_grad():
            bias = model.transformer.h[0].attn.bias
            bias[0, 0, 0, 0] = not bool(bias[0, 0, 0, 0])
        self.assertNotEqual(before, _model_state_sha256(model))

    def test_cleanup_diagnostic_has_a_python_310_fallback(self):
        from looking_glass.distilgpt2_intervention_dev import SmokeFailure, _note
        class OlderException(Exception):
            def __getattribute__(self, name):
                if name == 'add_note':
                    raise AttributeError(name)
                return super().__getattribute__(name)
        error = OlderException('primary')
        _note(error, 'contaminated model')
        self.assertEqual(str(error), 'primary')
        self.assertIsInstance(error.__cause__, SmokeFailure)
        self.assertIn('contaminated model', str(error.__cause__))

    def test_subnormal_decoder_is_correct_with_flush_to_zero(self):
        code = textwrap.dedent("""
            import math, sys, torch
            from looking_glass.distilgpt2_intervention_dev import max_stored_difference
            if not torch.set_flush_denormal(True): sys.exit(77)
            zero = torch.tensor([0], dtype=torch.int32).view(torch.float32)
            pos = torch.tensor([1], dtype=torch.int32).view(torch.float32)
            neg = torch.tensor([-2147483647], dtype=torch.int32).view(torch.float32)
            expected = math.ldexp(1.0, -149)
            assert max_stored_difference(pos, zero) == expected
            assert max_stored_difference(neg, zero) == expected
            assert max_stored_difference(pos, neg) == 2 * expected
            bits = torch.tensor([1, 100, -2147483647, 100], dtype=torch.int32)
            noncontiguous = bits.view(torch.float32).reshape(2, 2)[:, 0]
            assert not noncontiguous.is_contiguous()
            assert max_stored_difference(noncontiguous, torch.zeros(2)) == expected
        """)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
        with tempfile.TemporaryDirectory() as prefix:
            child = subprocess.run([sys.executable, '-B', '-X', f'pycache_prefix={prefix}',
                                    '-c', code], env=env, capture_output=True,
                                   text=True, timeout=30)
            self.assertFalse(os.listdir(prefix))
        if child.returncode == 77:
            self.skipTest('this CPU build does not support flush-to-zero')
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_subnormal_decoder_without_assuming_ftz_support(self):
        from looking_glass.distilgpt2_intervention_dev import max_stored_difference
        zero = torch.tensor([0], dtype=torch.int32).view(torch.float32)
        positive = torch.tensor([1], dtype=torch.int32).view(torch.float32)
        negative = torch.tensor([-2147483647], dtype=torch.int32).view(torch.float32)
        self.assertEqual(max_stored_difference(positive, zero), math.ldexp(1.0, -149))
        self.assertEqual(max_stored_difference(negative, zero), math.ldexp(1.0, -149))

    @unittest.skipIf(sys.version_info < (3, 11), 'the dev runner requires Python 3.11 -P')
    def test_source_bootstrap_ignores_poisoned_python_environment(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'scripts' / 'distilgpt2_intervention_dev_bootstrap.py'
        site = Path(sysconfig.get_paths()['purelib'])
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / 'fresh-pycache'
            prefix.mkdir()
            poison = Path(temporary) / 'poison'
            poison.mkdir()
            sentinel = Path(temporary) / 'shadow-ran'
            (poison / 'json.py').write_text(
                'from pathlib import Path\nPath(' + repr(str(sentinel)) + ').touch()\n'
                'raise RuntimeError("wrong import")\n', encoding='utf-8')
            env = dict(os.environ, PYTHONPATH=str(poison), PYTHONHOME=str(poison))
            command = [sys._base_executable, '-B', '-E', '-s', '-S', '-P', '-X',
                       f'pycache_prefix={prefix}', str(script), '--check-only', '--site', str(site)]
            child = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertFalse(sentinel.exists())
            self.assertFalse(any(prefix.iterdir()))
            self.assertIn('"ignore_environment": true', child.stdout)

    @unittest.skipIf(sys.version_info < (3, 11), 'the dev runner requires Python 3.11 -P')
    def test_source_bootstrap_rejects_missing_environment_isolation(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'scripts' / 'distilgpt2_intervention_dev_bootstrap.py'
        site = Path(sysconfig.get_paths()['purelib'])
        with tempfile.TemporaryDirectory() as prefix:
            command = [sys._base_executable, '-B', '-s', '-S', '-P', '-X',
                       f'pycache_prefix={prefix}', str(script), '--check-only', '--site', str(site)]
            child = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn('requires isolated interpreter flags', child.stderr)

    @unittest.skipIf(sys.version_info < (3, 11), 'the dev runner requires Python 3.11 -P')
    def test_checkout_root_cannot_shadow_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'scripts' / 'distilgpt2_intervention_dev_bootstrap.py'
        site = Path(sysconfig.get_paths()['purelib'])
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / 'repo'
            (fake / 'scripts').mkdir(parents=True)
            (fake / 'looking_glass').mkdir()
            (fake / 'looking_glass' / '__init__.py').write_text('value = 7\n', encoding='utf-8')
            copied = fake / 'scripts' / script.name
            copied.write_bytes(script.read_bytes())
            sentinel = Path(temporary) / 'shadow-ran'
            (fake / 'torch.py').write_text('from pathlib import Path\nPath(' + repr(str(sentinel))
                                           + ').touch()\nraise RuntimeError("shadow")\n',
                                           encoding='utf-8')
            prefix = Path(temporary) / 'fresh'
            prefix.mkdir()
            code = ('import runpy, sys\n'
                    f'd = runpy.run_path({str(copied)!r}, run_name="source_test")\n'
                    f'repo, site, prefix, _ = d["_bootstrap"]({str(site)!r})\n'
                    'import torch\n'
                    'd["_load_project_package"](repo)\n'
                    'import looking_glass\n'
                    'assert looking_glass.value == 7\n'
                    'assert str(repo) not in sys.path\n')
            command = [sys._base_executable, '-B', '-E', '-s', '-S', '-P', '-X',
                       f'pycache_prefix={prefix}', '-c', code]
            child = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertFalse(sentinel.exists())
            self.assertFalse(any(prefix.iterdir()))

    def test_source_verification_rejects_filter_normalized_working_bytes(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'distilgpt2_intervention_dev_bootstrap.py'
        spec = importlib.util.spec_from_file_location('bootstrap_source_test', script)
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args],
                                               stderr=subprocess.STDOUT, timeout=30)
            git('init', '-q')
            git('config', 'user.name', 'Source Test')
            git('config', 'user.email', 'source-test@example.invalid')
            git('config', 'core.autocrlf', 'true')
            (repo / 'example.py').write_bytes(b'value = 1\r\n')
            git('add', 'example.py')
            git('commit', '-qm', 'test source')
            self.assertEqual(git('status', '--porcelain'), b'')
            revision = git('rev-parse', 'HEAD').decode().strip()
            with self.assertRaisesRegex(RuntimeError, 'raw source bytes differ'):
                bootstrap._sources(repo, revision, ['example.py'])

    def test_runtime_check_rejects_thread_setting_drift(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'distilgpt2_intervention_dev_bootstrap.py'
        spec = importlib.util.spec_from_file_location('bootstrap_runtime_test', script)
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        with mock.patch.object(torch, 'get_num_threads', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'runtime settings'):
                bootstrap._runtime_status(torch)


if __name__ == '__main__':
    unittest.main()
