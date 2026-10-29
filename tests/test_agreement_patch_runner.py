"""Synthetic model checks for the natural agreement patch executor."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from looking_glass.agreement_journal import AgreementJournal
from looking_glass.agreement_numeric import decode_f32_le
from looking_glass.agreement_patch_runner import (AgreementCase, CancellationGate, NaturalAgreementRunner,
                                                 StudyFailure, _CANDIDATES, _discovery_selection, _pair_with_signs,
                                                 _vector_l2)
from looking_glass.model_ownership import ModelLease


class ToyAttention(nn.Module):
    def forward(self, hidden):
        return (torch.cumsum(hidden, dim=1) * 0.02, 'untouched-tail')


class ToyMLP(nn.Module):
    def forward(self, hidden):
        return hidden * 0.03


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, hidden):
        attention = self.attn(hidden)
        if attention[1] != 'untouched-tail':
            raise AssertionError('tuple tail changed')
        hidden = hidden + attention[0]
        return (hidden + self.mlp(hidden),)


class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(600, 8)
        self.h = nn.ModuleList(ToyBlock() for _ in range(6))
        self.ln_f = nn.LayerNorm(8)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = ToyTransformer()
        self.head = nn.Linear(8, 600, bias=False)
        self.config = SimpleNamespace(vocab_size=600, n_embd=8, use_cache=False,
                                      _attn_implementation='eager')
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.forward_calls += 1
        if use_cache is not False or not bool((attention_mask == 1).all()):
            raise ValueError('invalid request')
        hidden = self.transformer.embedding(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)[0]
        return SimpleNamespace(logits=self.head(self.transformer.ln_f(hidden)), past_key_values=None)


def cases():
    return tuple(AgreementCase(case_id=lemma + '-' + number, quartet='q1', lemma=lemma,
                               number=number, input_ids=(1, subject_id, 2, 3), subject_index=1)
                 for lemma, number, subject_id in (
                     ('alpha', 'singular', 11), ('alpha', 'plural', 12),
                     ('beta', 'singular', 13), ('beta', 'plural', 14)))


class AgreementRunnerTests(unittest.TestCase):
    def setUp(self):
        old = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(123)
            self.model = ToyModel().eval()

    def test_complete_factorial_and_all_controls_on_synthetic_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            original_rng = torch.random.get_rng_state().clone()
            result = runner.run(cases())
            self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['forwards'], 152)
            self.assertEqual(self.model.forward_calls, 152)
            self.assertEqual(len(result['metrics']['pairs']), 128)
            self.assertEqual(len(result['metrics']['quartets']), 8)
            self.assertEqual(result['controls'], {'null': 4, 'self': 32, 'delivery': 4,
                                                  'subject_position': 4, 'repeat': 4, 'final_clean': 4})
            self.assertTrue(result['model_state_unchanged'])
            self.assertEqual(result['cpu_rng_seed'], 20261028)
            self.assertEqual(len(result['selection']['ranking']), 8)
            self.assertEqual(len(result['secondary_past_metrics']['pairs']), 128)
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertEqual(terminal['status'], 'complete_candidate')
            events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            patch = next(row['event'] for row in events if row['event']['kind'] == 'patch'
                         and row['event']['target'] != row['event']['donor'])
            self.assertTrue(patch['post_action_matches_donor_bits'])
            self.assertTrue(patch['untouched_output_bits_equal'])
            self.assertEqual(patch['site']['token_role'], 'last')
            patch_row = next(row for row in events if row['event'] == patch)
            self.assertIn('whole_before', patch_row['tensors'])
            self.assertIn('whole_after', patch_row['tensors'])
            past_event = next(row['event'] for row in events if row['event']['kind'] == 'secondary_past_site_metrics')
            self.assertEqual(past_event['contrast'], {'first_token': ' was', 'first_id': 373,
                                                       'second_token': ' were', 'second_id': 547})
            past = next(pair for pair in past_event['metrics']['pairs'] if pair['target'] != pair['donor'])
            self.assertNotIn('clean_is', past)
            self.assertNotIn('patched_are', past)
            clean_row = next(row for row in events if row['event']['kind'] == 'clean_capture'
                             and row['event']['case'] == past['target'])
            action_row = next(row for row in events if row['event']['kind'] == 'patch'
                              and row['event']['target'] == past['target']
                              and row['event']['donor'] == past['donor']
                              and row['event']['site']['id'] == past['site'])
            def payload(row, name):
                return (path / 'tensors' / (row['tensors'][name]['sha256'] + '.f32le')).read_bytes()
            clean_values = decode_f32_le(payload(clean_row, 'last_logits'))
            patched_values = decode_f32_le(payload(action_row, 'contrast_logits'))
            self.assertEqual((past['clean_was'], past['clean_were']), (clean_values[373], clean_values[547]))
            self.assertEqual((past['patched_was'], past['patched_were']), tuple(patched_values[2:4]))
            self.assertIn('patched_was', result['secondary_past_metrics']['pairs'][0])
            self.assertNotIn('patched_is', result['secondary_past_metrics']['pairs'][0])
            self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                                for module in self.model.modules()))

    def test_selection_ties_abstention_magnitude_and_fixed_random_site(self):
        sites = [site['id'] for site in _CANDIDATES]
        metrics = {'site_summaries': [{'site': site, 'mean_number_main_contrast': 0.0} for site in sites]}
        magnitudes = [{'site': site, 'quartet': 'q', 'lemma': lemma,
                       'singular': lemma + '-s', 'plural': lemma + '-p',
                       'l2': 3.0 if site == sites[3] else 1.0}
                      for site in sites for lemma in ('a', 'b')]
        result = _discovery_selection(metrics, magnitudes, 1)
        self.assertEqual(result['ranking'], sites)
        self.assertIsNone(result['selected_site'])
        self.assertEqual(result['earliest_site'], sites[0])
        self.assertEqual(result['magnitude_site'], sites[3])
        index = hashlib.sha256(b'MLG-number-v1-random\x00' + (20261028).to_bytes(8, 'big')).digest()[0] % 8
        self.assertEqual(result['random_site'], sites[index])
        metrics['site_summaries'][2]['mean_number_main_contrast'] = 2.0
        metrics['site_summaries'][3]['mean_number_main_contrast'] = 2.0
        result = _discovery_selection(metrics, magnitudes, 1)
        self.assertEqual(result['selected_site'], sites[2])
        broken = magnitudes[:-1] + [dict(magnitudes[-1], lemma='different')]
        with self.assertRaises(StudyFailure):
            _discovery_selection(metrics, broken, 1)

    def test_magnitude_decodes_smallest_subnormal_without_ftz_cast(self):
        self.assertEqual(_vector_l2(b'\x01\x00\x00\x00', b'\x00\x00\x00\x00'), math.ldexp(1.0, -149))

    def test_margin_sign_reporting_distinguishes_reversal_from_zero(self):
        flipped = _pair_with_signs({'clean_margin': 2.0, 'patched_margin': -1.0})
        self.assertEqual((flipped['clean_margin_sign'], flipped['patched_margin_sign']), (1, -1))
        self.assertTrue(flipped['strict_sign_reversal'])
        self.assertFalse(flipped['zero_transition'])
        zero = _pair_with_signs({'clean_margin': -1.0, 'patched_margin': 0.0})
        self.assertFalse(zero['strict_sign_reversal'])
        self.assertTrue(zero['zero_transition'])

    def test_four_complete_families_fit_bounded_per_site_summary_events(self):
        families = []
        for family_index in range(4):
            for member in cases():
                families.append(AgreementCase(
                    case_id=f'family-{family_index}-a-long-case-identifier-{member.case_id}',
                    quartet=f'family-{family_index}', lemma=member.lemma, number=member.number,
                    input_ids=(1, member.input_ids[1], 30 + family_index, 40 + family_index),
                    subject_index=1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic-four-families'})
            result = NaturalAgreementRunner(self.model, journal, expected_quartets=4).run(families)
            self.assertEqual(result['forwards'], 608)
            self.assertEqual(len(result['metrics']['pairs']), 512)
            self.assertEqual(len(result['secondary_past_metrics']['pairs']), 512)
            rows = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            primary = [row for row in rows if row['event']['kind'] == 'present_site_metrics']
            secondary = [row for row in rows if row['event']['kind'] == 'secondary_past_site_metrics']
            self.assertEqual(len(primary), 8)
            self.assertEqual(len(secondary), 8)
            self.assertTrue(all(len(row['event']['metrics']['pairs']) == 64 for row in primary + secondary))
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertEqual(terminal['status'], 'complete_candidate')
            self.assertEqual(terminal['event_log_bytes'], (path / 'events.jsonl').stat().st_size)

    def test_misaligned_quartet_and_preexisting_hook_fail_before_forward(self):
        for variant in ('alignment', 'hook'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                journal = AgreementJournal(Path(temporary) / 'attempt', {'protocol': 'synthetic'})
                altered = list(cases())
                handle = None
                if variant == 'alignment':
                    original = altered[0]
                    altered[0] = AgreementCase(original.case_id, original.quartet, original.lemma,
                                               original.number, (1, 11, 99, 3), 1)
                else:
                    handle = self.model.transformer.h[0].register_forward_hook(lambda *args: None)
                try:
                    runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
                    with self.assertRaises(StudyFailure):
                        runner.run(tuple(altered))
                    self.assertEqual(self.model.forward_calls, 0)
                    self.assertEqual(json.loads((Path(temporary) / 'attempt' / 'terminal.json').read_text())['status'],
                                     'failed')
                finally:
                    if handle:
                        handle.remove()

    def test_cancel_before_first_forward_is_terminal_and_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            cancel = CancellationGate()
            cancel.cancel()
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1,
                                             cancel_event=cancel)
            with self.assertRaises(StudyFailure):
                runner.run(cases())
            self.assertEqual(self.model.forward_calls, 0)
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'cancelled')

    def test_late_cancel_preserves_first_completed_forward_before_stopping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            cancel = CancellationGate()
            original = self.model.forward
            def forward_then_cancel(*args, **kwargs):
                result = original(*args, **kwargs)
                cancel.cancel()
                return result
            self.model.forward = forward_then_cancel
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1,
                                             cancel_event=cancel)
            with self.assertRaises(StudyFailure):
                runner.run(cases())
            events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(row['event']['kind'] == 'clean_capture' for row in events))
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'cancelled')

    def test_state_mutation_on_failed_forward_quarantines_the_parent_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            def mutate_and_raise(*args, **kwargs):
                with torch.no_grad():
                    self.model.head.weight[0, 0] += 1
                raise ValueError('failed after mutation')
            self.model.forward = mutate_and_raise
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with self.assertRaisesRegex(ValueError, 'failed after mutation'):
                runner.run(cases())
            with self.assertRaises(RuntimeError):
                ModelLease(self.model).acquire()
            self.assertTrue(runner.quarantined)

    def test_hook_mutation_on_failed_forward_quarantines_the_parent_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            handle = []
            def mutate_and_raise(*args, **kwargs):
                handle.append(self.model.transformer.h[0].register_forward_hook(lambda *args: None))
                raise ValueError('failed after hook change')
            self.model.forward = mutate_and_raise
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            try:
                with self.assertRaisesRegex(ValueError, 'failed after hook change'):
                    runner.run(cases())
                with self.assertRaises(RuntimeError):
                    ModelLease(self.model).acquire()
                self.assertTrue(runner.quarantined)
            finally:
                for item in handle:
                    item.remove()

    def test_equal_parameter_replacement_before_cancellation_quarantines_model(self):
        for mode in ('replace-object', 'replace-storage'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                model = ToyModel().eval()
                path = Path(temporary) / 'attempt'
                journal = AgreementJournal(path, {'protocol': 'synthetic'})
                gate = CancellationGate()
                original = model.forward
                def replace_then_cancel(*args, **kwargs):
                    result = original(*args, **kwargs)
                    old = model.head.weight
                    if mode == 'replace-object':
                        model.head.weight = nn.Parameter(old.detach().clone(), requires_grad=old.requires_grad)
                    else:
                        old.data = old.detach().clone()
                    gate.cancel()
                    return result
                model.forward = replace_then_cancel
                runner = NaturalAgreementRunner(model, journal, expected_quartets=1, cancel_event=gate)
                with self.assertRaises(StudyFailure):
                    runner.run(cases())
                self.assertTrue(runner.quarantined)
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
                self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'cancelled')

    def test_eval_mode_drift_during_forward_quarantines_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            original = self.model.forward
            def flip_mode(*args, **kwargs):
                result = original(*args, **kwargs)
                self.model.train()
                return result
            self.model.forward = flip_mode
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with self.assertRaisesRegex(StudyFailure, 'mode'):
                runner.run(cases())
            self.assertTrue(runner.quarantined)
            with self.assertRaises(RuntimeError):
                ModelLease(self.model).acquire()

    def test_config_drift_during_forward_quarantines_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            original = self.model.forward
            def flip_config(*args, **kwargs):
                result = original(*args, **kwargs)
                self.model.config.use_cache = True
                return result
            self.model.forward = flip_config
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with self.assertRaises(StudyFailure):
                runner.run(cases())
            self.assertTrue(runner.quarantined)
            with self.assertRaises(RuntimeError):
                ModelLease(self.model).acquire()

    def test_attention_scaling_drift_during_forward_quarantines_model(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=600, n_positions=16, n_embd=8, n_layer=6,
                            n_head=2, use_cache=False)
        config._attn_implementation = 'eager'
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(123)
            model = GPT2LMHeadModel(config).eval()
        original = model.forward
        def mutate_after_forward(*args, **kwargs):
            result = original(*args, **kwargs)
            model.transformer.h[1].attn.scale_attn_weights = False
            return result
        model.forward = mutate_after_forward
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic-random-gpt2'})
            runner = NaturalAgreementRunner(model, journal, expected_quartets=1)
            with self.assertRaisesRegex(StudyFailure, 'execution contract changed'):
                runner.run(cases())
            self.assertTrue(runner.quarantined)
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'failed')
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()

    def test_numeric_runtime_drift_during_forward_quarantines_model(self):
        before = torch.are_deterministic_algorithms_enabled()
        warning = torch.is_deterministic_algorithms_warn_only_enabled()
        original = self.model.forward
        def mutate_after_forward(*args, **kwargs):
            result = original(*args, **kwargs)
            torch.use_deterministic_algorithms(not before)
            return result
        self.model.forward = mutate_after_forward
        try:
            with tempfile.TemporaryDirectory() as temporary:
                journal = AgreementJournal(Path(temporary) / 'attempt', {'protocol': 'synthetic'})
                runner = NaturalAgreementRunner(self.model, journal, expected_quartets=1)
                with self.assertRaisesRegex(StudyFailure, 'execution contract changed'):
                    runner.run(cases())
                self.assertTrue(runner.quarantined)
                with self.assertRaises(RuntimeError):
                    ModelLease(self.model).acquire()
        finally:
            torch.use_deterministic_algorithms(before, warn_only=warning)

    def test_global_runtime_snapshot_detects_threads_precision_and_denormals(self):
        from looking_glass.agreement_patch_runner import _global_runtime_configuration
        before = dict(_global_runtime_configuration())
        precision = torch.get_float32_matmul_precision()
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(threads + 1)
            self.assertNotEqual(dict(_global_runtime_configuration())['cpu_threads'], before['cpu_threads'])
            torch.set_float32_matmul_precision('medium' if precision != 'medium' else 'highest')
            self.assertNotEqual(dict(_global_runtime_configuration())['float32_matmul_precision'],
                                before['float32_matmul_precision'])
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.set_num_threads(threads)
        ftz_before = before['calling_thread_denormal_product_bits'] == 0
        if not torch.set_flush_denormal(not ftz_before):
            self.skipTest('denormal mode switching is unavailable on this host')
        try:
            self.assertNotEqual(dict(_global_runtime_configuration())['calling_thread_denormal_product_bits'],
                                before['calling_thread_denormal_product_bits'])
        finally:
            torch.set_flush_denormal(ftz_before)

    def test_eval_mode_is_checked_after_lease_acquisition(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = AgreementJournal(Path(temporary) / 'attempt', {'protocol': 'synthetic'})
            original = ModelLease.acquire
            def mutate_before_acquire(lease):
                lease.model.train()
                return original(lease)
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with mock.patch.object(ModelLease, 'acquire', mutate_before_acquire):
                with self.assertRaisesRegex(StudyFailure, 'mode'):
                    runner.run(cases())
            self.assertEqual(self.model.forward_calls, 0)

    def test_unprintable_exception_still_has_terminal_record(self):
        class Unprintable(ValueError):
            def __str__(self):
                raise RuntimeError('cannot format')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            def fail(*args, **kwargs):
                raise Unprintable()
            self.model.forward = fail
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            original_rng = torch.random.get_rng_state().clone()
            with self.assertRaises(Unprintable):
                runner.run(cases())
            self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertEqual(terminal['status'], 'failed')
            self.assertEqual(terminal['detail']['error_type'], 'Unprintable')
            self.assertEqual(terminal['detail']['message'], '<exception message unavailable>')

    def test_interrupted_cleanup_transition_quarantines_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with mock.patch.object(runner, '_release_after_work', side_effect=KeyboardInterrupt('cleanup transition')):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(cases())
            self.assertTrue(runner.quarantined)
            with self.assertRaises(RuntimeError):
                ModelLease(self.model).acquire()
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'failed')

    def test_metadata_property_failure_preserves_available_captures(self):
        from looking_glass.intervention_plan import ActivationPlan
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with mock.patch.object(ActivationPlan, 'metadata', property(lambda _: (_ for _ in ()).throw(ValueError('bad metadata')))):
                with self.assertRaisesRegex(ValueError, 'bad metadata'):
                    runner.run(cases())
            events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            rejected = next(row for row in events if row['event']['kind'] == 'rejected_forward')
            self.assertIn('capture_0', rejected['tensors'])
            self.assertIn('last_logits', rejected['tensors'])

    def test_post_forward_check_failure_preserves_pending_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            original = journal.read_tensor
            reads = 0
            def reject_reference(ref):
                nonlocal reads
                reads += 1
                if reads > 32:  # Complete all 8 sites × 2 lemma pairs × 2 vectors first.
                    raise ValueError('reference invalid after forward')
                return original(ref)
            with mock.patch.object(journal, 'read_tensor', side_effect=reject_reference):
                with self.assertRaisesRegex(ValueError, 'reference invalid'):
                    runner.run(cases())
            events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            rejected = next(row for row in events if row['event']['kind'] == 'rejected_forward')
            self.assertIn('capture_0', rejected['tensors'])
            self.assertIn('last_logits', rejected['tensors'])
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertEqual(terminal['detail']['model_returns'], 5)
            self.assertEqual(terminal['detail']['forward_evidence_records'], 5)
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'failed')

    def test_interrupted_lease_acquire_does_not_abandon_the_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = AgreementJournal(Path(temporary) / 'attempt', {'protocol': 'synthetic'})
            original = ModelLease.acquire
            def acquired_then_interrupted(lease):
                original(lease)
                raise KeyboardInterrupt('after acquire')
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with mock.patch.object(ModelLease, 'acquire', acquired_then_interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(cases())
            lease = ModelLease(self.model)
            lease.acquire()
            lease.release()

    def test_cancellation_gate_has_an_explicit_completion_boundary(self):
        gate = CancellationGate()
        self.assertTrue(gate.commit())
        self.assertFalse(gate.cancel())
        cancelled = CancellationGate()
        self.assertTrue(cancelled.cancel())
        self.assertFalse(cancelled.commit())

    def test_cancel_during_scoring_prevents_complete_candidate(self):
        from looking_glass import agreement_patch_runner as module
        original = module.score_factorial
        gate = CancellationGate()
        def score_then_cancel(*args, **kwargs):
            result = original(*args, **kwargs)
            gate.cancel()
            return result
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1,
                                             cancel_event=gate)
            with mock.patch.object(module, 'score_factorial', side_effect=score_then_cancel):
                with self.assertRaises(StudyFailure):
                    runner.run(cases())
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'cancelled')

    def test_cpu_autocast_is_rejected_before_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                with self.assertRaisesRegex(StudyFailure, 'autocast'):
                    runner.run(cases())
            self.assertEqual(self.model.forward_calls, 0)

    def test_corrupt_same_site_action_order_metadata_fails(self):
        from looking_glass.intervention_plan import ActivationPlan
        original = ActivationPlan.metadata.fget
        def malformed(plan):
            result = original(plan)
            result['steps'][0]['preceding_same_site_actions'] = ['not-actually-run']
            return result
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            runner = NaturalAgreementRunner(self.model, journal, device='cpu', expected_quartets=1)
            with mock.patch.object(ActivationPlan, 'metadata', property(malformed)):
                with self.assertRaisesRegex(StudyFailure, 'metadata'):
                    runner.run(cases())
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertEqual(terminal['status'], 'failed')
            self.assertEqual(terminal['detail']['forwards_started'], 1)
            self.assertEqual(terminal['detail']['model_returns'], 1)
            self.assertEqual(terminal['detail']['forwards_validated'], 0)
            events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            rejected = next(row for row in events if row['event']['kind'] == 'rejected_forward')
            self.assertEqual(rejected['event']['plan_metadata']['steps'][0]['preceding_same_site_actions'],
                             ['not-actually-run'])
            self.assertIn('last_logits', rejected['tensors'])
            self.assertIn('capture_0', rejected['tensors'])


if __name__ == '__main__':
    unittest.main()
