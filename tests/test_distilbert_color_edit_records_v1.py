"""Fabricated saved-record tests; no checkpoint or real cases are used."""
import base64
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from looking_glass import distilbert_color_edit_records_v1 as records


def fixture():
    cases = []
    for obj in ('a', 'b', 'c'):
        for role in ('discovery', 'calibration-01', 'calibration-02'):
            cases.append({'id': obj + '-' + role, 'object': obj, 'role': role,
                          'input_ids': [101, len(cases), 103, 102],
                          'attention_mask': [1, 1, 1, 1], 'mask_index': 2,
                          'factual': 'f', 'counterfactual': 'p', 'wrong_target': 'b',
                          'word_ids': {'f': 1, 'p': 0, 'b': 2}})
    return {'candidate_order': ['a', 'b', 'c'], 'cases': cases}


def row_event(index, row, logits):
    raw = struct.pack('<3f', *logits)
    return [{'seq': index * 2, 'kind': 'start', 'action': index, 'phase': 'screen', 'object': row['object'],
             'case_id': row['id'], 'condition': 'original',
             'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
             'mask_index': row['mask_index']},
            {'seq': index * 2 + 1, 'kind': 'complete', 'action': index, 'phase': 'screen', 'object': row['object'],
             'case_id': row['id'], 'condition': 'original',
             'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
             'mask_index': row['mask_index'],
             'parameter': None, 'parameter_sha256': None, 'realized_norm': 0.0,
             'logits': records.encode_raw(raw)}]


class RecordsTests(unittest.TestCase):
    def test_screen_decision_rebuilt_from_all_nine_raw_rows(self):
        cases = fixture()
        events = []
        for i, row in enumerate(cases['cases']):
            # First candidate fails on one tie; second is fully eligible.
            values = [0, 1, 1] if i == 1 else [0, 2, 1]
            events.extend(row_event(i, row, values))
        decision = records.rebuild_screen(cases, events, vocabulary=3)
        self.assertEqual(decision['selected_object'], 'b')
        self.assertEqual(decision['candidate_eligible'], {'a': False, 'b': True, 'c': True})
        self.assertIsNone(decision['rows'][1]['unique_top'])
        self.assertEqual(decision['rows'][1]['top_tie_count'], 2)
        self.assertIn('0', decision['rows'][0]['probabilities'])
        self.assertGreater(decision['rows'][0]['top_probability'], 0)

    def test_missing_duplicate_mismatched_and_malformed_rows_fail_closed(self):
        cases = fixture()
        events = [event for i, row in enumerate(cases['cases'])
                  for event in row_event(i, row, [0, 2, 1])]
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, events[:-1], vocabulary=3)
        bad = list(events)
        bad[1] = dict(bad[1], action=8)
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, bad, vocabulary=3)
        bad = list(events)
        bad[0] = dict(bad[0], attention_mask=[1, 1, 0, 1])
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, bad, vocabulary=3)
        bad = list(events)
        bad[0] = dict(bad[0], attention_mask=[True, 1, 1, 1])
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, bad, vocabulary=3)
        bad = list(events)
        bad[1] = dict(bad[1], input_ids=[101, 9, 103, 102])
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, bad, vocabulary=3)
        bad = list(events)
        bad[1] = dict(bad[1], parameter='weight', parameter_sha256='a' * 64, realized_norm=1.0)
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, bad, vocabulary=3)
        bad = list(events)
        bad[1] = dict(bad[1], logits={'data': [0, 2, 1], 'sha256': 'x'})
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, bad, vocabulary=3)
        bad = list(events)
        nan = struct.pack('<3f', 0.0, float('nan'), 1.0)
        bad[1] = dict(bad[1], logits=records.encode_raw(nan))
        with self.assertRaises(ValueError):
            records.rebuild_screen(cases, bad, vocabulary=3)

    def test_raw_bytes_hash_length_and_budget_are_enforced(self):
        raw = struct.pack('<3f', 0.0, 1.0, 2.0)
        encoded = records.encode_raw(raw)
        self.assertEqual(records.decode_raw(encoded, 3), (0.0, 1.0, 2.0))
        with self.assertRaises(ValueError):
            records.decode_raw(dict(encoded, sha256='0' * 64), 3)
        with self.assertRaises(ValueError):
            records.decode_raw(dict(encoded, data=base64.b64encode(raw + b'x').decode()), 3)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.jsonl'
            with records.Journal(path, byte_cap=1000, forward_cap=1) as journal:
                journal.start({'action': 0})
                journal.emit({'kind': 'complete', 'action': 0})
                with self.assertRaises(ValueError):
                    journal.start({'action': 1})
            self.assertEqual(len(path.read_bytes().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                records.Journal(path, byte_cap=1000, forward_cap=1)


if __name__ == '__main__':
    unittest.main()
