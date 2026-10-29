"""CPU-only synthetic checks for the disabled BERT GPU-study adapter."""

import copy
import os
import struct
import unittest
import weakref
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from looking_glass.bert_agreement_runner import (SITES, _run_family,
    float32_bytes, run_family, run_observation)
from looking_glass.bert_agreement_audit import audit_family_events
from looking_glass.bert_gpu_study_adapter import (_ObservationCustody, _observe_cpu_prototype,
    run_cuda_observation)
from looking_glass.intervention_plan import ActivationPlan


def family_manifest():
    rows = []
    for source, lemma, subject_ids in [('A', 'alpha', (3, 4)), ('B', 'beta', (11, 12))]:
        for subject_number, subject_id in zip(('singular', 'plural'), subject_ids):
            for distractor_number, distractor_id in zip(('singular', 'plural'), (13, 14)):
                row_id = f'{source}-{subject_number[0]}-{distractor_number[0]}'
                rows.append({'row_id': row_id, 'split': 'development', 'family': 1,
                    'subject_source': source, 'subject_lemma': lemma, 'subject_number': subject_number,
                    'distractor_lemma': 'gamma', 'distractor_number': distractor_number,
                    'input_ids': [1, 2, subject_id, 4, 5, distractor_id, 7, 8, 9, 10],
                    'attention_mask': [1] * 10})
    targets = []
    for subject_number in ('singular', 'plural'):
        opposite = 'plural' if subject_number == 'singular' else 'singular'
        for distractor_number in ('singular', 'plural'):
            s, o, d = subject_number[0], opposite[0], distractor_number[0]
            targets.append({'case_id': f'target-{s}-{d}', 'split': 'development', 'family': 1,
                'target_row_id': f'A-{s}-{d}', 'donor_row_ids': {
                    'A_same': f'A-{s}-{d}', 'A_opposite': f'A-{o}-{d}',
                    'B_same': f'B-{s}-{d}', 'B_opposite': f'B-{o}-{d}'}})
    return {'rows': rows, 'targets': targets}


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        config = DistilBertConfig(vocab_size=64, dim=16, hidden_dim=32, n_layers=6,
            n_heads=4, dropout=0., attention_dropout=0., max_position_embeddings=32,
            attn_implementation='eager')
        cls.model = DistilBertForMaskedLM(config).eval()
        cls.a = torch.tensor([[1, 2, 3, 4, 5, 13, 7, 8, 9, 10]], dtype=torch.int64)
        cls.b = torch.tensor([[1, 2, 11, 4, 5, 13, 7, 8, 9, 10]], dtype=torch.int64)

    def prototype(self, inputs, layer, token, **kwargs):
        return _observe_cpu_prototype(self.model, inputs, layer=layer, token=token,
            mask_token=6, is_id=20, are_id=21, check_owner=lambda: None,
            synchronize=lambda stage: None, **kwargs)

    def test_cuda_entry_is_disabled_before_model_or_cuda_access(self):
        with patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CUDA touched')):
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                run_cuda_observation(object(), object(), layer=2, token=2, mask_token=6,
                                     is_id=20, are_id=21)

    def test_synthetic_observations_match_cpu_reference_at_every_study_site(self):
        for layer, token in [(layer, token) for _, layer, token in SITES] + [(5, 2), (5, 6)]:
            with self.subTest(layer=layer, token=token):
                donor = self.prototype(self.b, layer, token)
                for action in ({}, {'self_patch': True}, {'source': donor.before}):
                    reference = run_observation(self.model, self.a, layer=layer, token=token,
                        mask_token=6, is_id=20, are_id=21, **action)
                    result = self.prototype(self.a, layer, token, **action)
                    self.assertEqual(float32_bytes(result.mask_logits), float32_bytes(reference.mask_logits))
                    self.assertEqual(float32_bytes(result.before), float32_bytes(reference.before))
                    self.assertEqual(None if result.after is None else float32_bytes(result.after),
                                     None if reference.after is None else float32_bytes(reference.after))
                    self.assertEqual((result.is_bits, result.are_bits, result.events, result.capture_bytes,
                                      result.source_bytes, result.model_input_ids, result.model_attention_mask),
                                     (reference.is_bits, reference.are_bits, reference.events, reference.capture_bytes,
                                      reference.source_bytes, reference.model_input_ids, reference.model_attention_mask))
                    self.assertEqual(result.mask_logits.device.type, 'cpu')
                    self.assertEqual(result.before.device.type, 'cpu')

    def test_noncontiguous_source_preserves_signed_zero_and_subnormal_bits(self):
        values = [0, -2147483648, 1, -2147483647] * 4
        interleaved = torch.zeros(32, dtype=torch.int32)
        interleaved[::2] = torch.tensor(values, dtype=torch.int32)
        source = interleaved[::2].view(torch.float32)
        self.assertFalse(source.is_contiguous())
        result = self.prototype(self.a, 2, 2, source=source)
        self.assertEqual(float32_bytes(result.after), struct.pack('<16i', *values))

    def test_full_synthetic_family_schedule_and_audit_match_cpu(self):
        baseline_records, adapted_records = [], []
        kwargs = dict(split='development', family=1, is_id=20, are_id=21, mask_id=7)
        baseline = run_family(self.model, family_manifest(), emit=baseline_records.append, **kwargs)
        def observer(model, inputs, **options):
            return _observe_cpu_prototype(model, inputs, check_owner=lambda: None,
                                          synchronize=lambda stage: None, **options)
        adapted = _run_family(self.model, family_manifest(), emit=adapted_records.append,
                              observer=observer, **kwargs)
        self.assertEqual(adapted, baseline)
        self.assertEqual(adapted_records, baseline_records)
        self.assertEqual(adapted['forward_count'], 140)
        audited = audit_family_events(family_manifest(), adapted_records, split='development',
            family=1, mask_id=7, hidden_size=16, expected_device='cpu')
        self.assertEqual(audited['cases'], baseline['cases'])
        with self.assertRaises(ValueError):
            audit_family_events(family_manifest(), adapted_records, split='development',
                family=1, mask_id=7, hidden_size=16, expected_device='cuda:0')
        simulated = copy.deepcopy(adapted_records)
        for record in simulated:
            if record['status'] == 'completed':
                for event in record['events']:
                    event['device'] = 'cuda:0'
        self.assertEqual(audit_family_events(family_manifest(), simulated, split='development',
            family=1, mask_id=7, hidden_size=16, expected_device='cuda:0')['cases'], baseline['cases'])
        with self.assertRaises(ValueError):
            audit_family_events(family_manifest(), simulated, split='development',
                family=1, mask_id=7, hidden_size=16)

    def test_revoked_owner_after_forward_prevents_later_sync(self):
        live = {'allowed': True}
        stages = []
        original = self.model.forward
        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            live['allowed'] = False
            return result
        def check():
            if not live['allowed']:
                raise RuntimeError('owner revoked')
        with patch.object(self.model, 'forward', side_effect=forward):
            with self.assertRaisesRegex(RuntimeError, 'owner revoked'):
                _observe_cpu_prototype(self.model, self.a, layer=2, token=2,
                    mask_token=6, is_id=20, are_id=21, check_owner=check, synchronize=stages.append)
        self.assertEqual(stages, [])
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))

    def test_failed_live_sync_keeps_plan_owned_source_clone_rooted(self):
        custody = _ObservationCustody()
        source = torch.arange(16, dtype=torch.float32)
        calls = []
        def sync(stage):
            calls.append(stage)
            raise RuntimeError('failed drain')
        with self.assertRaisesRegex(RuntimeError, 'failed drain'):
            _observe_cpu_prototype(self.model, self.a, layer=2, token=2, mask_token=6,
                is_id=20, are_id=21, source=source, check_owner=lambda: None,
                synchronize=sync, custody=custody)
        self.assertEqual(calls, ['output_live'])
        self.assertEqual(len(custody.plan_source_roots), 1)
        self.assertEqual(float32_bytes(custody.plan_source_roots[0]), float32_bytes(source))
        self.assertIsNot(custody.plan_source_roots[0], source)
        self.assertEqual(float32_bytes(custody.capture_after), float32_bytes(source))
        self.assertEqual(tuple(custody.capture_before.shape), (16,))
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))

    def test_plan_construction_failure_keeps_allocated_source_rooted(self):
        custody = _ObservationCustody()
        source = torch.arange(16, dtype=torch.float32)
        original = ActivationPlan.__init__
        def fail_after_construction(plan, *args, **kwargs):
            original(plan, *args, **kwargs)
            raise MemoryError('injected construction failure')
        with patch.object(ActivationPlan, '__init__', fail_after_construction):
            with self.assertRaises(MemoryError):
                _observe_cpu_prototype(self.model, self.a, layer=2, token=2, mask_token=6,
                    is_id=20, are_id=21, source=source, check_owner=lambda: None,
                    synchronize=lambda stage: None, custody=custody)
        self.assertEqual(float32_bytes(custody.plan._sources['patch']), float32_bytes(source))

    def test_successful_disposal_occurs_only_after_live_syncs(self):
        custody = _ObservationCustody()
        source = torch.arange(16, dtype=torch.float32)
        references, stages = [], []
        def sync(stage):
            stages.append(stage)
            if stage == 'output_live':
                references.extend((weakref.ref(custody.logits),
                                   weakref.ref(custody.plan._sources['patch']),
                                   weakref.ref(custody.plan._captures['before'])))
                self.assertTrue(all(ref() is not None for ref in references))
            elif stage == 'captures_live':
                self.assertTrue(all(ref() is not None for ref in references))
            elif stage == 'device_refs_disposed':
                self.assertTrue(all(ref() is None for ref in references))
        _observe_cpu_prototype(self.model, self.a, layer=2, token=2, mask_token=6,
            is_id=20, are_id=21, source=source, check_owner=lambda: None,
            synchronize=sync, custody=custody)
        self.assertEqual(stages, ['output_live', 'captures_live', 'device_refs_disposed'])

    def test_autocast_is_rejected_before_forward(self):
        with torch.autocast('cpu', dtype=torch.bfloat16), \
             patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            with self.assertRaisesRegex(RuntimeError, 'autocast'):
                self.prototype(self.a, 2, 2)
            forward.assert_not_called()

    def test_eager_config_cannot_mask_sdpa_attention_modules(self):
        config = DistilBertConfig(vocab_size=64, dim=16, hidden_dim=32, n_layers=6,
            n_heads=4, dropout=0., attention_dropout=0., max_position_embeddings=32,
            attn_implementation='sdpa')
        model = DistilBertForMaskedLM(config).eval()
        model.config._attn_implementation = 'eager'
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            with self.assertRaisesRegex(RuntimeError, 'eager attention'):
                _observe_cpu_prototype(model, self.a, layer=2, token=2, mask_token=6,
                    is_id=20, are_id=21, check_owner=lambda: None, synchronize=lambda stage: None)
            forward.assert_not_called()

    def test_eager_attention_drift_is_rejected_after_forward(self):
        original = self.model.forward
        prior = self.model.config._attn_implementation
        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            self.model.config._attn_implementation = 'sdpa'
            return result
        try:
            with patch.object(self.model, 'forward', side_effect=forward):
                with self.assertRaisesRegex(RuntimeError, 'eager attention'):
                    self.prototype(self.a, 2, 2)
        finally:
            self.model.config._attn_implementation = prior
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))

    def test_module_training_drift_is_rejected_after_forward(self):
        original = self.model.forward
        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            self.model.distilbert.transformer.layer[-1].train()
            return result
        try:
            with patch.object(self.model, 'forward', side_effect=forward):
                with self.assertRaisesRegex(RuntimeError, 'evaluation mode'):
                    self.prototype(self.a, 2, 2)
        finally:
            self.model.eval()


if __name__ == '__main__':
    unittest.main()
