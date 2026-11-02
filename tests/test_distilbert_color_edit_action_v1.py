"""Exercise parameter restoration and raw evidence with a fabricated model."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from looking_glass import distilbert_color_edit_action_v1 as actions
from looking_glass import distilbert_color_edit_records_v1 as records


class TinyMasked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 3)
        self.lin2 = torch.nn.Linear(3, 2)
        self.project = torch.nn.Linear(2, 4)

    def forward(self, input_ids, attention_mask):
        if not torch.all(attention_mask == 1):
            raise RuntimeError('unexpected mask')
        return type('Output', (), {'logits': self.project(self.lin2(self.embed(input_ids)))})()


class ActionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = TinyMasked().eval()
        self.row = {'id': 'toy', 'object': 'toy', 'input_ids': [1, 2, 3],
                    'attention_mask': [1, 1, 1], 'mask_index': 1}

    def test_weight_edit_is_restored_and_saved_full_row_is_bound(self):
        before = self.model.lin2.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.jsonl'
            with records.Journal(path) as journal:
                engine = actions.ActionEngine(self.model, self.model.lin2, self.model.project.bias,
                                              vocabulary=4, hidden=2, intermediate=3)
                result = engine.run(journal, self.row, phase='toy', condition='edited',
                                    delta=torch.full_like(before, 0.01), parameter='weight')
            events = records.parse_jsonl(path.read_bytes())
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]['input_ids'], self.row['input_ids'])
            self.assertEqual(events[1]['condition'], 'edited')
            self.assertEqual(len(records.decode_raw(events[1]['logits'], 4)), 4)
            self.assertGreater(result['realized_norm'], 0)
        self.assertTrue(torch.equal(self.model.lin2.weight, before))
        self.assertFalse(self.model.lin2._forward_hooks)

    def test_gradient_and_site_witness_reconstruct_weight_gradient(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.jsonl'
            with records.Journal(path) as journal:
                engine = actions.ActionEngine(self.model, self.model.lin2, self.model.project.bias,
                                              vocabulary=4, hidden=2, intermediate=3)
                result = engine.run(journal, self.row, phase='toy', condition='gradient',
                                    gradient_targets=(2, 1), capture_site=True)
            self.assertLessEqual(result['relative_rank_one_residual'], 1e-5)
            events = records.parse_jsonl(path.read_bytes())
            done = events[-1]
            self.assertEqual(len(records.decode_raw(done['gradient'], 6)), 6)
            self.assertEqual(len(records.decode_raw(done['site_input'], 3)), 3)
            self.assertEqual(len(records.decode_raw(done['site_output_gradient'], 2)), 2)
        self.assertFalse(self.model.lin2._forward_hooks)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_failed_forward_restores_weight_and_leaves_incomplete_prefix(self):
        before = self.model.lin2.weight.detach().clone()
        def fail(**_):
            raise RuntimeError('fabricated failure')
        self.model.forward = fail
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.jsonl'
            with records.Journal(path) as journal:
                engine = actions.ActionEngine(self.model, self.model.lin2, self.model.project.bias,
                                              vocabulary=4, hidden=2, intermediate=3)
                with self.assertRaisesRegex(RuntimeError, 'fabricated'):
                    engine.run(journal, self.row, phase='toy', condition='edited',
                               delta=torch.full_like(before, 0.01), parameter='weight')
            events = records.parse_jsonl(path.read_bytes())
            self.assertEqual([event['kind'] for event in events], ['start', 'incomplete'])
            self.assertEqual(events[1]['failure_stage'], 'forward')
        self.assertTrue(torch.equal(self.model.lin2.weight, before))

    def test_returned_logits_survive_later_input_or_bias_mutation_failure(self):
        original_forward = self.model.forward
        original_bias = self.model.project.bias.detach().clone()
        def mutate(**kwargs):
            output = original_forward(**kwargs)
            self.model.project.bias.add_(1)
            kwargs['input_ids'][0, 0] += 1
            return output
        self.model.forward = mutate
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.jsonl'
            with records.Journal(path) as journal:
                engine = actions.ActionEngine(self.model, self.model.lin2, self.model.project.bias,
                                              vocabulary=4, hidden=2, intermediate=3)
                with self.assertRaises(RuntimeError):
                    engine.run(journal, self.row, phase='toy', condition='edited',
                               delta=torch.full_like(self.model.lin2.weight, 0.01), parameter='weight')
            incomplete = records.parse_jsonl(path.read_bytes())[-1]
            self.assertEqual(incomplete['kind'], 'incomplete')
            self.assertEqual(len(records.decode_raw(incomplete['returned_logits'], 4)), 4)
            self.assertNotEqual(incomplete['actual_input_ids'], self.row['input_ids'])
        self.assertTrue(torch.equal(self.model.project.bias, original_bias))

    def test_nonfinite_return_is_preserved_but_never_completed(self):
        original_forward = self.model.forward
        def nonfinite(**kwargs):
            output = original_forward(**kwargs)
            output.logits[0, 1, 2] = float('nan')
            return output
        self.model.forward = nonfinite
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.jsonl'
            with records.Journal(path) as journal:
                engine = actions.ActionEngine(self.model, self.model.lin2, self.model.project.bias,
                                              vocabulary=4, hidden=2, intermediate=3)
                with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                    engine.run(journal, self.row, phase='toy', condition='original')
            incomplete = records.parse_jsonl(path.read_bytes())[-1]
            self.assertEqual(incomplete['kind'], 'incomplete')
            self.assertIn('returned_logits', incomplete)
            with self.assertRaises(ValueError):
                records.decode_raw(incomplete['returned_logits'], 4)

    def test_unrelated_decoder_bias_mutation_is_detected_before_restore(self):
        original_forward = self.model.forward
        before = self.model.project.bias.detach().clone()
        def mutate_bias(**kwargs):
            output = original_forward(**kwargs)
            self.model.project.bias.add_(1)
            return output
        self.model.forward = mutate_bias
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.jsonl'
            with records.Journal(path) as journal:
                engine = actions.ActionEngine(self.model, self.model.lin2, self.model.project.bias,
                                              vocabulary=4, hidden=2, intermediate=3)
                with self.assertRaisesRegex(RuntimeError, 'decoder bias changed'):
                    engine.run(journal, self.row, phase='toy', condition='edited',
                               delta=torch.full_like(self.model.lin2.weight, 0.01), parameter='weight')
            incomplete = records.parse_jsonl(path.read_bytes())[-1]
            self.assertEqual(incomplete['kind'], 'incomplete')
            self.assertIn('returned_logits', incomplete)
        self.assertTrue(torch.equal(self.model.project.bias, before))

    def test_invalid_return_shape_keeps_bounded_row_or_explicit_omission(self):
        for shape, expected in (((1, 3, 3), 'saved'), ((1, 0, 4), 'unavailable-unindexable'),
                                ((1, 3, 5), 'unavailable-oversized-or-empty')):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as temp:
                self.model.forward = lambda **_: type('Output', (), {'logits': torch.ones(shape)})()
                path = Path(temp) / 'events.jsonl'
                with records.Journal(path) as journal:
                    engine = actions.ActionEngine(self.model, self.model.lin2, self.model.project.bias,
                                                  vocabulary=4, hidden=2, intermediate=3)
                    with self.assertRaisesRegex(RuntimeError, 'invalid full logits'):
                        engine.run(journal, self.row, phase='toy', condition='original')
                incomplete = records.parse_jsonl(path.read_bytes())[-1]
                self.assertEqual(incomplete['kind'], 'incomplete')
                self.assertEqual(incomplete['returned_output']['shape'], list(shape))
                self.assertEqual(incomplete['returned_output']['status'], expected)
                if expected == 'saved':
                    self.assertEqual(records.decode_raw(incomplete['returned_logits'], 3), (1.0, 1.0, 1.0))
                else:
                    self.assertNotIn('returned_logits', incomplete)


if __name__ == '__main__':
    unittest.main()
