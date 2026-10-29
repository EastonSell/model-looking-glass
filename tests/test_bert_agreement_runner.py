"""CPU controls for the fixed-site DistilBERT agreement intervention."""

import os
import math
import struct
import tempfile
import unittest
from unittest.mock import patch
import copy
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch

try:
    from transformers import DistilBertConfig, DistilBertForMaskedLM
except ImportError:
    DistilBertConfig = DistilBertForMaskedLM = None

from looking_glass.bert_agreement_runner import (decode_float32_bits, float32_bytes,
                                                 run_family, run_observation, signed_donor_effect)
from looking_glass.intervention_plan import ActivationPlan
from looking_glass.bert_agreement_audit import audit_family_events
from looking_glass.bert_agreement_study import EventJournal, audit_event_journal


@unittest.skipIf(DistilBertConfig is None, 'transformers is not installed')
class AgreementRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = DistilBertConfig(vocab_size=64, dim=16, hidden_dim=32,
                                  n_layers=6, n_heads=4, dropout=0.,
                                  attention_dropout=0., max_position_embeddings=32)
        cls.model = DistilBertForMaskedLM(config).eval()
        cls.a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.int64)
        cls.b = torch.tensor([[1, 2, 11, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.int64)

    def obs(self, ids, layer, token, **kwargs):
        return run_observation(self.model, ids, layer=layer, token=token,
                               mask_token=6, is_id=20, are_id=21, **kwargs)

    def test_self_patch_preserves_mask_bits_and_reports_one_invocation(self):
        baseline = self.obs(self.a, 1, 2)
        patched = self.obs(self.a, 1, 2, self_patch=True)
        self.assertTrue(torch.equal(baseline.mask_logits.view(torch.int32),
                                    patched.mask_logits.view(torch.int32)))
        self.assertTrue(torch.equal(patched.before.view(torch.int32),
                                    patched.after.view(torch.int32)))
        self.assertEqual([(e['operation'], e['module_path'], e['invocation'],
                           e['batch_index'], e['token_index']) for e in patched.events],
                         [('capture', 'distilbert.transformer.layer.1', 0, 0, 2),
                          ('patch', 'distilbert.transformer.layer.1', 0, 0, 2),
                          ('capture', 'distilbert.transformer.layer.1', 0, 0, 2)])

    def test_final_mask_donor_lands_and_reproduces_donor_mask_logits(self):
        donor = self.obs(self.b, 5, 6)
        patched = self.obs(self.a, 5, 6, source=donor.before)
        self.assertTrue(torch.equal(patched.after.view(torch.int32),
                                    donor.before.view(torch.int32)))
        self.assertTrue(torch.equal(patched.mask_logits.view(torch.int32),
                                    donor.mask_logits.view(torch.int32)))

    def test_final_subject_patch_leaves_mask_logits_unchanged(self):
        donor = self.obs(self.b, 5, 2)
        baseline = self.obs(self.a, 5, 2)
        patched = self.obs(self.a, 5, 2, source=donor.before)
        self.assertTrue(torch.equal(patched.mask_logits.view(torch.int32),
                                    baseline.mask_logits.view(torch.int32)))

    def test_failed_shape_does_not_leave_a_hook_installed(self):
        baseline = self.obs(self.a, 1, 2)
        with self.assertRaises(ValueError):
            self.obs(self.a, 1, 2, source=torch.zeros(15))
        after = self.obs(self.a, 1, 2)
        self.assertTrue(torch.equal(after.mask_logits.view(torch.int32),
                                    baseline.mask_logits.view(torch.int32)))
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks
                             for m in self.model.modules()))

    def test_existing_hook_is_rejected_without_removing_it(self):
        handle = self.model.distilbert.transformer.layer[1].register_forward_hook(
            lambda _module, _inputs, _output: None)
        try:
            with self.assertRaisesRegex(RuntimeError, 'hook'):
                self.obs(self.a, 1, 2)
            self.assertIn(handle.id, self.model.distilbert.transformer.layer[1]._forward_hooks)
        finally:
            handle.remove()

    def test_noncontiguous_signed_zero_and_subnormal_source_bits_survive(self):
        values = [0, -2147483648, 1, -2147483647] * 4
        interleaved = torch.zeros(32, dtype=torch.int32)
        interleaved[::2] = torch.tensor(values, dtype=torch.int32)
        source = interleaved[::2].view(torch.float32)
        self.assertFalse(source.is_contiguous())
        self.assertEqual(float32_bytes(source), struct.pack('<16i', *values))
        patched = self.obs(self.a, 1, 2, source=source)
        self.assertEqual(float32_bytes(patched.after), struct.pack('<16i', *values))

    def test_stored_float32_subnormals_decode_exactly_and_nonfinite_is_rejected(self):
        self.assertEqual(decode_float32_bits(1), 2.0 ** -149)
        self.assertEqual(decode_float32_bits(0x80000001), -(2.0 ** -149))
        self.assertEqual(math.copysign(1.0, decode_float32_bits(0x80000000)), -1.0)
        self.assertEqual(decode_float32_bits(0x3f800000), 1.0)
        for invalid in (0x7f800000, 0xff800000, 0x7fc00000, -1, 0x100000000):
            with self.assertRaises(ValueError):
                decode_float32_bits(invalid)

    def test_factorial_signed_effect_respects_target_number_and_missing_data(self):
        bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
        zero = bits(0.)
        singular = {'a_same': (bits(2.), zero), 'a_opposite': (bits(-2.), zero),
                    'b_same': (bits(1.), zero), 'b_opposite': (bits(-3.), zero)}
        plural = {'a_same': (bits(-2.), zero), 'a_opposite': (bits(2.), zero),
                  'b_same': (bits(-1.), zero), 'b_opposite': (bits(3.), zero)}
        self.assertEqual(signed_donor_effect('singular', singular), 4.)
        self.assertEqual(signed_donor_effect('plural', plural), 4.)
        with self.assertRaises(ValueError):
            signed_donor_effect('singular', {k: v for k, v in singular.items() if k != 'b_same'})
        with self.assertRaises(ValueError):
            signed_donor_effect('singular', {**singular, 'a_opposite': (0x7f800000, zero)})

    def test_inconsistent_action_metadata_is_not_published_as_success(self):
        getter = ActivationPlan.metadata.fget
        def damaged(plan):
            value = getter(plan)
            value['steps'][1]['source_capture'] = None
            return value
        with patch.object(ActivationPlan, 'metadata', property(damaged)):
            with self.assertRaises(RuntimeError):
                self.obs(self.a, 1, 2, self_patch=True)

    def test_boolean_coordinate_or_byte_metadata_is_rejected(self):
        getter = ActivationPlan.metadata.fget
        for corruption in ('invocation', 'batch_index', 'forward_index', 'capture_bytes', 'shape'):
            def damaged(plan):
                value = getter(plan)
                if corruption == 'capture_bytes':
                    value['capture_bytes'] = True
                elif corruption == 'shape':
                    value['steps'][0]['output_shape'][0] = True
                else:
                    value['steps'][0][corruption] = False
                return value
            with self.subTest(corruption=corruption), patch.object(ActivationPlan, 'metadata', property(damaged)):
                with self.assertRaises(RuntimeError):
                    self.obs(self.a, 1, 2)

    @staticmethod
    def sample_family():
        rows = []
        for source, lemma, subject_ids in [('A', 'alpha', (3, 4)), ('B', 'beta', (11, 12))]:
            for subject_number, subject_id in zip(('singular', 'plural'), subject_ids):
                for distractor_number, distractor_id in zip(('singular', 'plural'), (13, 14)):
                    row_id = f'{source}-{subject_number[0]}-{distractor_number[0]}'
                    rows.append({'row_id': row_id, 'split': 'development', 'family': 1,
                                 'subject_source': source, 'subject_lemma': lemma,
                                 'subject_number': subject_number, 'distractor_lemma': 'gamma',
                                 'distractor_number': distractor_number,
                                 'input_ids': [1, 2, subject_id, 4, 5, distractor_id, 7, 8, 9, 10],
                                 'attention_mask': [1] * 10})
        targets = []
        for subject_number in ('singular', 'plural'):
            opposite = 'plural' if subject_number == 'singular' else 'singular'
            for distractor_number in ('singular', 'plural'):
                s, o, d = subject_number[0], opposite[0], distractor_number[0]
                targets.append({'case_id': f'target-{s}-{d}', 'split': 'development', 'family': 1,
                                'target_row_id': f'A-{s}-{d}',
                                'donor_row_ids': {'A_same': f'A-{s}-{d}', 'A_opposite': f'A-{o}-{d}',
                                                  'B_same': f'B-{s}-{d}', 'B_opposite': f'B-{o}-{d}'}})
        return {'rows': rows, 'targets': targets}

    def test_one_family_runs_all_donors_controls_and_clean_checks(self):
        records = []
        result = run_family(self.model, self.sample_family(), split='development', family=1,
                            is_id=20, are_id=21, mask_id=7, emit=records.append)
        self.assertEqual(result['forward_count'], 140)
        self.assertEqual(len(result['cases']), 4)
        self.assertTrue(all(len(case['effects']) == 5 for case in result['cases']))
        self.assertEqual(sum(r['status'] == 'started' for r in records), 140)
        self.assertEqual(sum(r['status'] == 'completed' for r in records), 140)
        self.assertTrue(all(c['passed'] for c in result['controls']))
        audited = audit_family_events(self.sample_family(), records, split='development',
                                      family=1, mask_id=7, hidden_size=16)
        self.assertEqual(audited['cases'], result['cases'])
        self.assertEqual(audited['site_means'], result['site_means'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            with EventJournal(path) as journal:
                for record in records:
                    journal.emit(record)
            audited_file = audit_event_journal(self.sample_family(), path, split='development',
                                               family=1, mask_id=7, hidden_size=16,
                                               expected_sha256=journal.sha256)
            self.assertEqual(audited_file['families'][0]['cases'], result['cases'])
            with self.assertRaises(ValueError):
                audit_event_journal(self.sample_family(), path, split='development',
                                    family=1, mask_id=7, hidden_size=16, expected_sha256='0' * 64)
        damaged = copy.deepcopy(records)
        patch_row = next(row for row in damaged if row['status'] == 'completed'
                         and row['kind'] == 'patch_A_opposite')
        patch_row['after_float32_le_base64'] = patch_row['before_float32_le_base64']
        with self.assertRaises(ValueError):
            audit_family_events(self.sample_family(), damaged, split='development',
                                family=1, mask_id=7, hidden_size=16)
        damaged = copy.deepcopy(records)
        final_row = next(row for row in damaged if row['status'] == 'completed'
                         and row['kind'] == 'delivery_patch' and row['site'] == 'final_subject')
        wrong_capture = next(row for row in damaged if row['status'] == 'completed'
                             and row['kind'] == 'delivery_donor' and row['site'] == final_row['site']
                             and row['row_id'] == final_row['donor_row_id'])
        final_row['before_float32_le_base64'] = wrong_capture['before_float32_le_base64']
        with self.assertRaises(ValueError):
            audit_family_events(self.sample_family(), damaged, split='development',
                                family=1, mask_id=7, hidden_size=16)
        damaged = copy.deepcopy(records)
        patch_row = next(row for row in damaged if row['status'] == 'completed'
                         and row['kind'] == 'patch_A_opposite')
        wrong_capture = next(row for row in damaged if row['status'] == 'completed'
                             and row['kind'] == 'donor_capture' and row['site'] == patch_row['site']
                             and row['row_id'] == patch_row['donor_row_id'])
        patch_row['before_float32_le_base64'] = wrong_capture['before_float32_le_base64']
        with self.assertRaises(ValueError):
            audit_family_events(self.sample_family(), damaged, split='development',
                                family=1, mask_id=7, hidden_size=16)
        damaged = copy.deepcopy(records)
        self_row = next(row for row in damaged if row['status'] == 'completed'
                        and row['kind'] == 'patch_A_same')
        self_row['is_bits'], self_row['is_logit'] = 0x3f800000, 1.0
        with self.assertRaises(ValueError):
            audit_family_events(self.sample_family(), damaged, split='development',
                                family=1, mask_id=7, hidden_size=16)

    def test_wrong_donor_number_is_rejected_before_any_forward(self):
        manifest = self.sample_family()
        manifest['targets'][0]['donor_row_ids']['B_opposite'] = 'B-s-s'
        records = []
        with self.assertRaises(ValueError):
            run_family(self.model, manifest, split='development', family=1,
                       is_id=20, are_id=21, mask_id=7, emit=records.append)
        self.assertEqual(records, [])

    def test_collapsed_lemmas_wrong_mask_context_or_boolean_mask_cannot_run(self):
        for corruption in ('collapsed_lemma', 'wrong_mask', 'different_context', 'boolean_mask'):
            manifest = self.sample_family()
            for row in manifest['rows']:
                if corruption == 'collapsed_lemma' and row['subject_source'] == 'B':
                    row['input_ids'][2] = 3 if row['subject_number'] == 'singular' else 4
                elif corruption == 'wrong_mask':
                    row['input_ids'][6] = 8
                elif corruption == 'different_context' and row['distractor_number'] == 'plural':
                    row['input_ids'][3] = 15
                elif corruption == 'boolean_mask':
                    row['attention_mask'] = [True] * 10
            records = []
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                run_family(self.model, manifest, split='development', family=1,
                           is_id=20, are_id=21, mask_id=7, emit=records.append)
            self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()
