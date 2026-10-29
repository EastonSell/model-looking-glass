"""CPU-only checks of the frozen matched-dose schedule on fabricated data."""
import copy
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from looking_glass.bert_agreement_runner import Observation, decode_float32_bits, run_observation
from looking_glass.bert_matched_runner import _run_family
from looking_glass.bert_matched_evidence import audit_family_events, aggregate_families


def sample_family():
    rows, targets = [], []
    for source, lemma, subject_ids in (('A', 'alpha', (3, 4)), ('B', 'beta', (11, 12))):
        for number, subject_id in zip(('singular', 'plural'), subject_ids):
            for distractor, distractor_id in zip(('singular', 'plural'), (13, 14)):
                rows.append({'row_id': f'{source}-{number[0]}-{distractor[0]}',
                             'split': 'development', 'family': 1, 'subject_source': source,
                             'subject_lemma': lemma, 'subject_number': number,
                             'distractor_lemma': 'gamma', 'distractor_number': distractor,
                             'input_ids': [1, 2, subject_id, 4, 5, distractor_id, 7, 8, 9, 10],
                             'attention_mask': [1] * 10})
    for number in ('singular', 'plural'):
        opposite = 'plural' if number == 'singular' else 'singular'
        for distractor in ('singular', 'plural'):
            s, o, d = number[0], opposite[0], distractor[0]
            targets.append({'case_id': f'target-{s}-{d}', 'split': 'development', 'family': 1,
                            'target_row_id': f'A-{s}-{d}',
                            'donor_row_ids': {'A_same': f'A-{s}-{d}', 'A_opposite': f'A-{o}-{d}',
                                              'B_same': f'B-{s}-{d}', 'B_opposite': f'B-{o}-{d}'}})
    return {'rows': rows, 'targets': targets}


def fake_observation(_model, ids, *, layer, token, mask_token, is_id, are_id,
                     source=None, self_patch=False):
    """A deterministic 768-coordinate stand-in; it never loads a checkpoint."""
    assert ids.shape == (1, 10) and token == 2 and mask_token == 6
    before = torch.arange(768, dtype=torch.float32) / 2048 + layer
    x, y = {3: (0., 0.), 4: (1., 1.), 11: (2., 0.), 12: (0., 2.)}[int(ids[0, 2])]
    before[0] += x
    before[1] += y
    after = before.clone() if self_patch else None if source is None else source.clone()
    logits = torch.zeros(64, dtype=torch.float32)
    logits[is_id] = float(ids[0, 2]) / 10
    logits[are_id] = float(ids[0, 5]) / 20
    if source is not None:
        logits[is_id] += (source[0] - before[0]) * (3 if layer == 0 else 1)
    raw = logits.view(torch.int32)
    bits = lambda index: int(raw[index]) & 0xffffffff
    path = f'distilbert.transformer.layer.{layer}'
    names = ('before', 'patch', 'after') if after is not None else ('before',)
    ops = ('capture', 'patch', 'capture') if after is not None else ('capture',)
    events = []
    for index, (name, operation) in enumerate(zip(names, ops)):
        events.append({'name': name, 'operation': operation, 'order': index,
                       'module_path': path, 'module_aliases': [path], 'call_site_known': False,
                       'forward_index': 0, 'invocation': 0, 'batch_index': 0, 'token_index': token,
                       'output_selector': [0], 'output_shape': [1, 10, 768], 'selected_shape': [768],
                       'dtype': 'torch.float32', 'device': 'cpu',
                       'source_capture': 'before' if self_patch and index == 1 else None,
                       'preceding_same_site_actions': list(names[:index])})
    return Observation(logits, before, after, bits(is_id), bits(are_id),
                       decode_float32_bits(bits(is_id)), decode_float32_bits(bits(are_id)),
                       tuple(events), 3072 * len((before,) if after is None else (before, after)),
                       0 if source is None else 3072, tuple(ids[0].tolist()), (1,) * 10)


class MatchedRunnerTest(unittest.TestCase):
    def test_full_family_schedule_and_record_audit(self):
        records = []
        manifest = sample_family()
        model = SimpleNamespace(distilbert=SimpleNamespace(transformer=SimpleNamespace(layer=[None] * 6)))
        result = _run_family(model, manifest, split='development', family=1,
                             is_id=20, are_id=21, mask_id=7, emit=records.append,
                             observer=fake_observation)
        self.assertEqual(result['forward_count'], 60)
        self.assertEqual(sum(row['status'] == 'started' for row in records), 60)
        self.assertEqual(sum(row['status'] == 'completed' for row in records), 60)
        self.assertEqual(sum(row['status'] == 'control' for row in records), 20)
        self.assertEqual(sum(row['status'] == 'dose_plan' for row in records), 4)
        reconstructed = audit_family_events(manifest, records, split='development', family=1, mask_id=7)
        self.assertEqual(reconstructed, result)
        for case in result['cases']:
            for site in ('block0', 'block2'):
                cells = case['sites'][site]['margins']
                r = 1 if case['subject_number'] == 'singular' else -1
                self.assertEqual(case['sites'][site]['primary_effect'],
                                 -r * math.fsum((cells['B_opposite'], -cells['B_same'])))
                expected = (math.sqrt(2.) if case['subject_number'] == 'singular' else 2.)
                expected *= 3 if site == 'block0' else 1
                self.assertAlmostEqual(case['sites'][site]['primary_effect'], expected, places=5)
                for name in ('A_opposite', 'B_same', 'B_opposite', 'artificial'):
                    dose = case['sites'][site]['doses'][name]
                    self.assertLessEqual(dose['absolute_error'], max(1e-6, 1e-5 * case['radius']))

        for corruption in ('input', 'landing', 'plan', 'missing', 'control', 'logit'):
            damaged = copy.deepcopy(records)
            if corruption == 'input':
                next(row for row in damaged if row['status'] == 'completed')['model_input_ids'][2] = True
            elif corruption == 'landing':
                row = next(row for row in damaged if row['status'] == 'completed' and row['kind'] == 'B_same')
                row['after_float32_le_base64'] = row['before_float32_le_base64']
            elif corruption == 'plan':
                next(row for row in damaged if row['status'] == 'dose_plan')['radius'] += 1
            elif corruption == 'missing':
                damaged.pop()
            elif corruption == 'control':
                next(row for row in damaged if row['status'] == 'control')['passed'] = False
            else:
                row = next(row for row in damaged if row['status'] == 'completed')
                row['is_bits'] = 0x7f800000
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                audit_family_events(manifest, damaged, split='development', family=1, mask_id=7)

    def test_forward_failure_remains_incomplete(self):
        records = []
        model = SimpleNamespace(distilbert=SimpleNamespace(transformer=SimpleNamespace(layer=[None] * 6)))
        def broken(*_args, **_kwargs):
            raise OSError('fabricated failure')
        with self.assertRaises(OSError):
            _run_family(model, sample_family(), split='development', family=1,
                        is_id=20, are_id=21, mask_id=7, emit=records.append, observer=broken)
        self.assertEqual([row['status'] for row in records], ['started', 'failed'])
        with self.assertRaises(ValueError):
            audit_family_events(sample_family(), records,
                                split='development', family=1, mask_id=7)

    def test_wrong_input_or_action_landing_is_recorded_as_failed(self):
        model = SimpleNamespace(distilbert=SimpleNamespace(transformer=SimpleNamespace(layer=[None] * 6)))
        for corruption in ('mask', 'landing'):
            records = []
            def damaged(*args, **kwargs):
                obs = fake_observation(*args, **kwargs)
                if corruption == 'mask':
                    return replace(obs, model_attention_mask=(0,) * 10)
                if obs.after is not None:
                    return replace(obs, after=obs.after + 1)
                return obs
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                _run_family(model, sample_family(), split='development', family=1,
                            is_id=20, are_id=21, mask_id=7, emit=records.append, observer=damaged)
            self.assertEqual(records[-1]['status'], 'failed')
            self.assertFalse(any(record['status'] == 'completed'
                                 and record['forward_index'] == records[-1]['forward_index']
                                 for record in records))

    def test_cancellation_while_recording_a_failure_remains_primary(self):
        model = SimpleNamespace(distilbert=SimpleNamespace(transformer=SimpleNamespace(layer=[None] * 6)))
        def broken(*_args, **_kwargs):
            raise OSError('fabricated forward failure')
        def emit(record):
            if record['status'] == 'failed':
                raise KeyboardInterrupt('fabricated cancellation')
        with self.assertRaises(KeyboardInterrupt) as caught:
            _run_family(model, sample_family(), split='development', family=1,
                        is_id=20, are_id=21, mask_id=7, emit=emit, observer=broken)
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_invalid_dose_plan_leaves_an_explicit_failure(self):
        records = []
        model = SimpleNamespace(distilbert=SimpleNamespace(transformer=SimpleNamespace(layer=[None] * 6)))
        with patch('looking_glass.bert_matched_runner.plan_case', side_effect=ValueError('low dose')):
            with self.assertRaises(ValueError):
                _run_family(model, sample_family(), split='development', family=1,
                            is_id=20, are_id=21, mask_id=7, emit=records.append, observer=fake_observation)
        self.assertEqual(records[-1]['status'], 'dose_plan_failed')
        self.assertEqual(records[-1]['error_type'], 'ValueError')
        self.assertEqual(sum(record['status'] == 'completed' for record in records), 16)

    def test_all_eight_families_required_for_the_decision(self):
        with self.assertRaises(ValueError):
            aggregate_families([])

    def test_real_cpu_hook_path_on_an_untrained_model(self):
        try:
            from transformers import DistilBertConfig, DistilBertForMaskedLM
        except ImportError:
            self.skipTest('transformers is not installed')
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.set_default_dtype(torch.float32)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(19)
                config = DistilBertConfig(vocab_size=64, dim=768, hidden_dim=96, n_layers=6,
                                          n_heads=12, dropout=0., attention_dropout=0.,
                                          max_position_embeddings=32)
                config._attn_implementation = 'eager'
                model = DistilBertForMaskedLM(config).eval()
                records = []
                result = _run_family(model, sample_family(), split='development', family=1,
                                     is_id=20, are_id=21, mask_id=7,
                                     emit=records.append, observer=run_observation)
                self.assertEqual(result['forward_count'], 60)
                self.assertEqual(sum(row['status'] == 'control' and row['passed'] for row in records), 20)
                self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks
                                     for module in model.modules()))
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
            torch.set_default_dtype(previous_dtype)
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
