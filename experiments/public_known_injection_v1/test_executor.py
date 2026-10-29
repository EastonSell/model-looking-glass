"""A single fabricated random-model case tests the bounded action schedule."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from executor import ControlFailure, execute_phase


class ExecutorTests(unittest.TestCase):
    def test_too_small_raw_cap_stops_before_any_model_call(self):
        config = GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=6,
                            n_head=2, use_cache=False, attn_pdrop=0,
                            resid_pdrop=0, embd_pdrop=0)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        cases = [{'case_id': 'known-injection-01', 'split': 'discovery', 'token_ids': [1, 7]}]
        with tempfile.TemporaryDirectory() as temp, patch.object(model, 'forward', wraps=model.forward) as forward:
            run = Path(temp) / 'run'
            with self.assertRaisesRegex(ValueError, 'before model forward'):
                execute_phase(model, cases, run, phase='discovery', vocab_size=41,
                              raw_cap=8, minimum_free_bytes=0)
            self.assertEqual(forward.call_count, 0)
            self.assertTrue((run / 'failure.json').exists())

    def test_false_required_control_is_recorded_and_stops_before_next_forward(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            config = GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=6,
                                n_head=2, use_cache=False, attn_pdrop=0,
                                resid_pdrop=0, embd_pdrop=0)
            config._attn_implementation = 'eager'
            torch.manual_seed(71)
            model = GPT2LMHeadModel(config).eval()
            cases = [{'case_id': 'known-injection-01', 'split': 'discovery',
                      'token_ids': [1, 7, 11, 21, 31]}]
            with tempfile.TemporaryDirectory() as temp, patch('executor.bitwise_equal', return_value=False):
                run = Path(temp) / 'run'
                with self.assertRaises(ControlFailure):
                    execute_phase(model, cases, run, phase='discovery', vocab_size=41)
                rows = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
                self.assertEqual(sum(row['type'] == 'start' for row in rows), 2)
                complete = [row for row in rows if row['type'] == 'complete']
                self.assertEqual(len(complete), 2)
                self.assertFalse(complete[-1]['checks']['logits_equal_clean'])
                self.assertEqual(json.loads((run / 'failure.json').read_text())['type'], 'ControlFailure')
                self.assertFalse((run / 'phase-summary.json').exists())
        finally:
            torch.set_num_threads(old)

    def test_one_case_records_all_actions_and_bitwise_controls(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            config = GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=6,
                                n_head=2, use_cache=False, attn_pdrop=0,
                                resid_pdrop=0, embd_pdrop=0)
            config._attn_implementation = 'eager'
            torch.manual_seed(71)
            model = GPT2LMHeadModel(config).eval()
            cases = [{'case_id': 'known-injection-01', 'split': 'discovery',
                      'token_ids': [1, 7, 11, 21, 31]}]
            with tempfile.TemporaryDirectory() as temp:
                run = Path(temp) / 'run'
                result = execute_phase(model, cases, run, phase='discovery', vocab_size=41)
                self.assertEqual(result['completed_actions'], 15)
                rows = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
                self.assertEqual(sum(row['type'] == 'start' for row in rows), 15)
                self.assertEqual(sum(row['type'] == 'complete' for row in rows), 15)
                checks = [row['checks'] for row in rows if row['type'] == 'complete']
                for check in checks:
                    self.assertTrue(all(value for key, value in check.items()
                                        if isinstance(value, bool)), check)
                tensor_rows = [row['tensor'] for row in rows if row['type'] == 'tensor']
                cursor = 0
                for item in tensor_rows:
                    self.assertEqual(item['offset'], cursor)
                    cursor += item['length']
                self.assertEqual(cursor, (run / 'known-injection-01.f32').stat().st_size)
        finally:
            torch.set_num_threads(old)


if __name__ == '__main__':
    unittest.main()
