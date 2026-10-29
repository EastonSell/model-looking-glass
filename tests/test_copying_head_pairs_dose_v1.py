"""Fabricated CPU runner checks for the public pairwise-dose protocol."""
import tempfile
import unittest
import copy
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib

import torch

from looking_glass.copy_head_pairs_dose_v1 import ACTION_HEADS, head_slice, plan_family
from scripts import copying_head_pairs_dose_v1 as runner


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        self.transformer.h[3].attn = torch.nn.Module()
        self.transformer.h[3].attn.c_proj = torch.nn.Identity()

    def forward(self, *, input_ids, **kwargs):
        value = (torch.arange(11 * 768, dtype=torch.float32).reshape(1, 11, 768) + 1) / 1000
        projected = self.transformer.h[3].attn.c_proj(value)
        logits = torch.zeros((1, 11, runner.base.VOCAB), dtype=torch.float32)
        logits[:, :, :768] = projected
        return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)


class PairDoseRunnerTests(unittest.TestCase):
    def test_source_identity_ignores_inherited_git_redirects_and_replace_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source.py').write_bytes(b'source bytes')
            def fake_git(args, **kwargs):
                env = kwargs['env']
                self.assertNotIn('GIT_DIR', env)
                self.assertNotIn('GIT_INDEX_FILE', env)
                self.assertNotIn('GIT_REPLACE_REF_BASE', env)
                self.assertEqual(env['GIT_NO_REPLACE_OBJECTS'], '1')
                if args[1] == 'rev-parse': return b'a' * 40 + b'\n'
                if args[1:3] == ['show', '-s']: return b'b' * 40 + b'\n'
                if args[1] == 'status': return b''
                if args[1] == 'show': return b'source bytes'
                raise AssertionError(args)
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'SOURCES', ('source.py',)), \
                 patch.dict(os.environ, {'GIT_DIR': 'redirect', 'GIT_INDEX_FILE': 'other',
                                         'GIT_REPLACE_REF_BASE': 'other'}), \
                 patch.object(runner.subprocess, 'check_output', side_effect=fake_git):
                identity = runner.source_identity()
            self.assertEqual(identity['revision'], 'a' * 40)
            self.assertEqual(identity['tree'], 'b' * 40)

    def test_fixed_cases_and_unpublished_reservation_fail_closed(self):
        data = json.loads((runner.ROOT / runner.CASE_PATH).read_bytes())
        self.assertEqual(len(runner.validate_cases(data)), 32)
        for key, bad in (('protocol_revision', 'bad'), ('known_exposure_sha256', 'bad'),
                         ('expected_forwards', 575), ('raw_tensor_bytes', 0), ('arms', []),
                         ('ordered_input_plus_target_sha256', 'bad')):
            changed = copy.deepcopy(data)
            changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_cases(changed)
        changed = copy.deepcopy(data)
        changed['cases'][0]['target_id'] += 1
        with self.assertRaises(ValueError):
            runner.validate_cases(changed)
        with patch.object(runner, 'RESERVATION_SHA256', None):
            with self.assertRaisesRegex(RuntimeError, 'not all frozen'):
                runner.inventory()

    def test_exposure_requires_exact_planned_ids_and_categories(self):
        data = json.loads((runner.ROOT / runner.CASE_PATH).read_bytes())
        cases = data['cases']
        own = [{'record_id': f"planned-copy-pairs-dose-{case['group']:02d}-{case['variant']}",
                'category': 'planned', 'token_ids': case['token_ids'], 'target_token_id': case['target_id']}
               for case in cases]
        baseline = SimpleNamespace(INVENTORY_VERSION=runner.BASELINE_VERSION,
                                   inventory_digest=lambda: runner.BASELINE_SHA256)
        reservation = SimpleNamespace(INVENTORY_VERSION='fake', inventory_digest=lambda: 'fake-sha',
            PAIRS_DOSE_ROWS=tuple(tuple(c['token_ids']) for c in cases),
            PAIRS_DOSE_GROUPS=tuple(tuple(g['token_ids']) for g in data['groups']),
            inventory_payload=lambda: {'records': own})
        recipient = (runner.ROOT / 'experiments/distilgpt2_recipient_screen_v1.json').read_bytes()
        import looking_glass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experiments').mkdir()
            (root / 'experiments/distilgpt2_recipient_screen_v1.json').write_bytes(recipient)
            patches = [patch.object(runner, 'ROOT', root), patch.object(runner, 'RESERVATION_VERSION', 'fake'),
                       patch.object(runner, 'RESERVATION_SHA256', 'fake-sha'),
                       patch.object(runner, 'RESERVATION_MODULE', 'fake'),
                       patch.object(looking_glass, 'pretrained_exposures_v13', baseline, create=True),
                       patch.object(importlib, 'import_module', return_value=reservation)]
            with contextlib.ExitStack() as stack:
                for item in patches: stack.enter_context(item)
                self.assertEqual(runner.check_exposure(cases)['reservation'], 'fake-sha')
                own[0]['category'] = 'observed'
                with self.assertRaises(RuntimeError): runner.check_exposure(cases)
                own[0]['category'] = 'planned'
                own[0]['record_id'] = 'planned-copy-pairs-dose-wrong'
                with self.assertRaises(RuntimeError): runner.check_exposure(cases)
                own[0]['record_id'] = 'planned-copy-pairs-dose-01-0'
                own[0]['record_id'], own[1]['record_id'] = own[1]['record_id'], own[0]['record_id']
                with self.assertRaises(RuntimeError): runner.check_exposure(cases)

    def test_every_arm_delivers_the_frozen_single_or_pair_bits(self):
        model = SyntheticModel().eval()
        ids = torch.arange(11, dtype=torch.int64).reshape(1, 11)
        clean = runner.run_forward(model, ids, 'clean')
        noop = runner.run_forward(model, ids, 'noop')
        self.assertEqual(clean['logits'], noop['logits'])
        self.assertEqual(noop['cproj_before'], noop['cproj_after'])
        plan = plan_family((noop['cproj_before'],) * 4)
        outputs = {}
        for action, heads in ACTION_HEADS.items():
            dose_name = action.rsplit('_', 1)[1]
            payload = runner.run_forward(model, ids, action, expected_before=noop['cproj_before'],
                deliveries={head: plan.variants[0][dose_name][head].delivered for head in heads})
            self.assertEqual(payload['cproj_before'], noop['cproj_before'])
            for head in heads:
                self.assertEqual(head_slice(payload['cproj_after'], head), plan.variants[0][dose_name][head].delivered)
            outputs[action] = payload['cproj_after']
        for dose in ('low', 'high'):
            for a, b in ((2, 7), (2, 10), (7, 10)):
                for head in (a, b):
                    self.assertEqual(head_slice(outputs[f'h{a}_h{b}_{dose}'], head),
                                     head_slice(outputs[f'h{head}_{dose}'], head))
        self.assertEqual(clean['logits'], runner.run_forward(model, ids, 'clean_after')['logits'])

    def test_failed_noop_control_retains_returned_payload(self):
        payload = {'logits': bytes(runner.base.LOGITS_BYTES), 'cproj_before': bytes(runner.base.WITNESS_BYTES),
                   'cproj_after': bytes(runner.base.WITNESS_BYTES), 'cproj_output': bytes(runner.base.WITNESS_BYTES)}
        case = {'case_id': 'test', 'group': 1, 'variant': 0, 'target_id': 3}
        marker = {'recorded': False}
        with tempfile.TemporaryDirectory() as directory:
            raw_path, log_path = Path(directory) / 'raw.f32', Path(directory) / 'events.jsonl'
            with raw_path.open('xb') as raw, log_path.open('xb') as log:
                with self.assertRaisesRegex(RuntimeError, 'logits differ'):
                    runner.persist_action(log, raw, case, 'noop', payload, b'different', None, lambda: None, marker)
            self.assertTrue(marker['recorded'])
            self.assertEqual(raw_path.stat().st_size, sum(map(len, payload.values())))

    def test_nonfinite_nonfinal_logits_fail_after_bounded_payload_is_captured(self):
        class BadEarlierLogit(SyntheticModel):
            def forward(self, **kwargs):
                result = super().forward(**kwargs)
                result.logits[0, 0, 0] = float('nan')
                return result
        observation = {}
        with self.assertRaisesRegex(RuntimeError, 'nonfinite full logits'):
            runner.run_forward(BadEarlierLogit().eval(), torch.arange(11).reshape(1, 11),
                               'clean', observation=observation)
        self.assertTrue(observation['model_call_return_observed'])
        self.assertEqual(len(observation['payload']['logits']), runner.base.LOGITS_BYTES)
        marker = {'partial': False}
        with tempfile.TemporaryDirectory() as directory:
            path, journal = Path(directory) / 'partial.f32', Path(directory) / 'events.jsonl'
            with journal.open('xb') as log:
                runner.persist_partial(log, path, 'case', 'clean', observation, marker)
            self.assertTrue(marker['partial'])
            self.assertEqual(path.read_bytes(), observation['payload']['logits'])
            self.assertTrue(json.loads(journal.read_bytes())['model_call_return_observed'])

    def test_full_fabricated_schedule_persists_all_actions_and_identities(self):
        groups, cases = [], []
        for group in range(8):
            words = [f'fake{group}word{i}' for i in range(6)]
            tokens = [32000 + group * 6 + i for i in range(6)]
            groups.append({'words': words, 'token_ids': tokens})
            for variant, permutation in enumerate(runner.base.PERMUTATIONS):
                indices = permutation + (0, 1, 2, 3, 4)
                cases.append({'case_id': f'copy-pairs-{group + 1:02d}-{variant}',
                              'group': group + 1, 'variant': variant,
                              'text': ''.join(' ' + words[i] for i in indices),
                              'token_ids': [tokens[i] for i in indices], 'target_id': tokens[5]})
        data = {'groups': groups, 'cases': cases}
        old_threads, old_dtype = torch.get_num_threads(), torch.get_default_dtype()
        old_deterministic, old_precision = torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision()
        try:
            torch.set_num_threads(1)
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
                root = Path(directory)
                repository, pycache, cache = root / 'repo', root / 'pycache', root / 'cache'
                repository.mkdir(); pycache.mkdir(); cache.mkdir()
                (repository / 'cases.json').write_text(json.dumps(data), encoding='utf-8')
                output = root / 'run'
                bundle = SimpleNamespace(model=SyntheticModel().eval(), tokenizer=None,
                                         metadata={'weights_sha256': runner.base.WEIGHTS_SHA256})
                patches = [patch.object(runner, 'ROOT', repository), patch.object(runner, 'CASE_PATH', 'cases.json'),
                    patch.object(runner, 'inventory', return_value=cases),
                    patch.object(runner, 'source_identity', return_value={'revision': 'synthetic'}),
                    patch.object(runner, 'environment', return_value=runner.EXPECTED_ENV),
                    patch.object(runner, 'verify_previous', return_value={}),
                    patch.object(runner, 'check_exposure', return_value={}),
                    patch.object(runner, 'verify_tokenization'), patch.object(runner, 'runtime', return_value={}),
                    patch.object(runner.base, 'dependency_identity', return_value={}),
                    patch.object(runner.base, 'numeric_policy', return_value={}),
                    patch.object(runner.base, 'load_distilgpt2', return_value=bundle),
                    patch.object(runner.base, 'state_hash', return_value='state'),
                    patch.object(runner.base, 'buffer_hash', return_value='buffers'),
                    patch.object(sys, 'argv', ['runner', '--cache-dir', str(cache), '--output-dir', str(output)]),
                    patch.object(sys, 'dont_write_bytecode', True), patch.object(sys, 'pycache_prefix', str(pycache)),
                    patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})]
                with contextlib.ExitStack() as stack:
                    for item in patches:
                        stack.enter_context(item)
                    with contextlib.redirect_stdout(io.StringIO()):
                        runner.main()
                result = json.loads((output / 'result.json').read_bytes())
                manifest = json.loads((output / 'manifest.json').read_bytes())
                self.assertEqual((result['submitted_forwards'], result['model_call_returns_observed'],
                                  result['returned_forwards']), (576, 576, 576))
                self.assertEqual(result['raw_payload_bytes'], 167_696_640)
                self.assertEqual((len(result['family_doses']), len(result['case_diagnostics'])), (8, 32))
                self.assertEqual(len(result['cases']), 576)
                self.assertEqual(manifest['retained_copy_accounting']['four_variant_cache_byte_arrays_upper'], 5_129_488)
                self.assertEqual(manifest['retained_copy_accounting']['raw_stream_bytes_per_case'], 5_240_520)
                pair = next(r for r in result['cases'] if r['action'] == 'h2_h7_low')
                self.assertEqual(set(pair['intervention']['joint_minus_matching_single_l2']), {'2', '7'})
                self.assertIn('changed_binary_bits', pair['intervention']['components']['2'])
                self.assertFalse((output / 'failure.json').exists())
                self.assertEqual(sum(p.stat().st_size for p in output.glob('*.f32')), 167_696_640)
        finally:
            torch.set_num_threads(old_threads)
            torch.set_default_dtype(old_dtype)
            torch.use_deterministic_algorithms(old_deterministic)
            torch.set_float32_matmul_precision(old_precision)


if __name__ == '__main__':
    unittest.main()
