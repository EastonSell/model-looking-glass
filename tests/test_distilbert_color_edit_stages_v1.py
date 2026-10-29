"""Full staged rehearsal with a fabricated tiny CPU model and public fake rows."""
import tempfile
import struct
import unittest
from pathlib import Path

import torch

from looking_glass import distilbert_color_edit_action_v1 as actions
from looking_glass import distilbert_color_edit_records_v1 as records
from looking_glass import distilbert_color_edit_stages_v1 as stages


class TinyMasked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 3)
        self.lin2 = torch.nn.Linear(3, 2)
        self.project = torch.nn.Linear(2, 4)
        with torch.no_grad():
            self.embed.weight.fill_(0.2)
            self.lin2.weight.fill_(0.1)
            self.lin2.bias.zero_()
            self.project.weight.zero_()
            self.project.weight[2].fill_(0.3)
            self.project.weight[3].fill_(-0.2)
            self.project.bias.copy_(torch.tensor([0.0, 3.0, 0.0, -1.0]))

    def forward(self, input_ids, attention_mask):
        return type('Output', (), {'logits': self.project(self.lin2(self.embed(input_ids)))})()


def manifest():
    result = []
    def row(obj, role, number, target):
        item = {'id': f'{obj}-{role}-{number}', 'object': obj, 'role': role,
                'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'mask_index': 1}
        if target:
            item.update(factual='f', counterfactual='p', wrong_target='b', word_ids={'f': 1, 'p': 2, 'b': 3})
        else:
            item.update(expected='f', accepted=['f'], word_ids={'f': 1})
        return item
    for obj in ('banana', 'strawberry', 'lemon'):
        for role in ('discovery', 'calibration-01', 'calibration-02', 'heldout-01', 'heldout-02', 'heldout-03', 'heldout-04'):
            result.append(row(obj, role, 0, True))
        for number in range(2):
            result.append(row(obj, 'same_object_locality', number, False))
    for role in ('other_color', 'unrelated'):
        for number in range(5):
            result.append(row(None, role, number, False))
    return {'candidate_order': ['banana', 'strawberry', 'lemon'], 'cases': result}


class StagesTests(unittest.TestCase):
    def test_complete_synthetic_schedule_and_raw_reconstruction(self):
        model, doc = TinyMasked().eval(), manifest()
        engine = actions.ActionEngine(model, model.lin2, model.project.bias, vocabulary=4, hidden=2, intermediate=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with records.Journal(root / 'screen') as journal:
                stages.run_screen(engine, journal, doc)
            screen = records.parse_jsonl((root / 'screen').read_bytes())
            self.assertEqual(records.rebuild_screen(doc, screen, vocabulary=4)['selected_object'], 'banana')
            with records.Journal(root / 'development') as journal:
                development = stages.run_development(engine, journal, doc, screen)
            dev_events = records.parse_jsonl((root / 'development').read_bytes())
            rebuilt, _, _ = stages.rebuild_development(engine, doc, screen, dev_events)
            self.assertEqual(development, rebuilt)
            self.assertFalse(development['discovery_top_achieved'])
            self.assertFalse(development['bias_available'])
            self.assertEqual(development['bias_status'], 'below-discovery-gain-floor')
            altered_bias = [dict(event) for event in dev_events]
            witness = next(event for event in altered_bias if event.get('kind') == 'bias-origin')
            witness['decoder_bias'] = records.encode_raw(struct.pack('<4f', 0.0, 3.0, 0.1, -1.0))
            with self.assertRaises(ValueError):
                stages.rebuild_development(engine, doc, screen, altered_bias)
            with records.Journal(root / 'evaluation') as journal:
                stages.run_evaluation(engine, journal, doc, screen, dev_events, rebuilt)
            evaluation = records.parse_jsonl((root / 'evaluation').read_bytes())
            result = stages.rebuild_evaluation(engine, doc, screen, dev_events, evaluation)
            self.assertEqual(len(result['rows']), 16)
            self.assertEqual(len(evaluation), 16 * 5 * 2)
            self.assertEqual(result['locality_decision']['status'], 'match-unavailable')
            altered = [dict(event) for event in dev_events]
            first_grid = next(event for event in altered if event.get('kind') == 'complete'
                              and event.get('condition') == 'primary-grid-00')
            first_grid['parameter_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                stages.rebuild_development(engine, doc, screen, altered)

    def test_synthetic_bias_control_is_rebuilt_and_remains_descriptive(self):
        model, doc = TinyMasked().eval(), manifest()
        with torch.no_grad():
            model.lin2.weight.copy_(torch.tensor([[100.0, -100.0, 0.0], [100.0, -100.0, 0.0]]))
        engine = actions.ActionEngine(model, model.lin2, model.project.bias, vocabulary=4, hidden=2, intermediate=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with records.Journal(root / 'screen') as journal:
                stages.run_screen(engine, journal, doc)
            screen = records.parse_jsonl((root / 'screen').read_bytes())
            with records.Journal(root / 'development') as journal:
                development = stages.run_development(engine, journal, doc, screen)
            dev_events = records.parse_jsonl((root / 'development').read_bytes())
            rebuilt, _, _ = stages.rebuild_development(engine, doc, screen, dev_events)
            self.assertEqual(development, rebuilt)
            self.assertTrue(development['bias_available'])
            self.assertTrue(development['calibration_match_available'])
            self.assertFalse(development['discovery_top_achieved'])
            altered = [dict(event) for event in dev_events]
            bias_event = next(event for event in altered if event.get('kind') == 'complete'
                              and event.get('condition') == 'bias')
            values = list(records.decode_raw(bias_event['logits'], 4))
            values[0] += 1.0
            bias_event['logits'] = records.encode_raw(struct.pack('<4f', *values))
            with self.assertRaisesRegex(ValueError, 'unmodified vocabulary'):
                stages.rebuild_development(engine, doc, screen, altered)
            with records.Journal(root / 'evaluation') as journal:
                stages.run_evaluation(engine, journal, doc, screen, dev_events, rebuilt)
            evaluation = records.parse_jsonl((root / 'evaluation').read_bytes())
            result = stages.rebuild_evaluation(engine, doc, screen, dev_events, evaluation)
            self.assertEqual(len(evaluation), 16 * 6 * 2)
            self.assertEqual(result['locality_decision']['status'], 'match-unavailable')

    def test_gradient_signed_zero_mismatch_stops_before_the_grid(self):
        model, doc = TinyMasked().eval(), manifest()
        original = model.forward
        def signed_zero(**kwargs):
            result = original(**kwargs)
            result.logits[..., 0] = -0.0 if torch.is_grad_enabled() else 0.0
            return result
        model.forward = signed_zero
        engine = actions.ActionEngine(model, model.lin2, model.project.bias, vocabulary=4, hidden=2, intermediate=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with records.Journal(root / 'screen') as journal:
                stages.run_screen(engine, journal, doc)
            screen = records.parse_jsonl((root / 'screen').read_bytes())
            with records.Journal(root / 'development') as journal:
                with self.assertRaisesRegex(RuntimeError, 'witnesses differ'):
                    stages.run_development(engine, journal, doc, screen)
            events = records.parse_jsonl((root / 'development').read_bytes())
            self.assertEqual([event['condition'] for event in events if event['kind'] == 'start'],
                             ['clean-site', 'primary-gradient'])

    def test_screen_signed_zero_mismatch_stops_before_gradient(self):
        model, doc = TinyMasked().eval(), manifest()
        engine = actions.ActionEngine(model, model.lin2, model.project.bias, vocabulary=4, hidden=2, intermediate=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with records.Journal(root / 'screen') as journal:
                stages.run_screen(engine, journal, doc)
            screen = records.parse_jsonl((root / 'screen').read_bytes())
            first = next(event for event in screen if event['kind'] == 'complete')
            values = list(records.decode_raw(first['logits'], 4))
            self.assertEqual(values[0], 0.0)
            values[0] = -0.0
            first['logits'] = records.encode_raw(struct.pack('<4f', *values))
            with records.Journal(root / 'development') as journal:
                with self.assertRaisesRegex(RuntimeError, 'screen and clean'):
                    stages.run_development(engine, journal, doc, screen)
            events = records.parse_jsonl((root / 'development').read_bytes())
            self.assertEqual([event['condition'] for event in events if event['kind'] == 'start'], ['clean-site'])


if __name__ == '__main__':
    unittest.main()
