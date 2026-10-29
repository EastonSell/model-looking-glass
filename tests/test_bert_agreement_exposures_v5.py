"""No-model checks for the BERT evaluation exposure-status successor."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import unittest

from looking_glass import bert_agreement_exposures_v4 as v4
from looking_glass import bert_agreement_exposures_v5 as v5


class BertExposureV5Test(unittest.TestCase):
    def test_snapshot_preserves_predecessor_and_marks_exact_evaluation_rows(self):
        previous, current = v4.build_snapshot(), v5.build_snapshot()
        self.assertEqual(current['predecessor'], {'schema': v4.SCHEMA, 'sha256': v5.PREDECESSOR_SHA256})
        self.assertEqual(current['evidence'][:-1], previous['evidence'])
        self.assertEqual(len(current['rows']), 96)
        for before, after in zip(previous['rows'], current['rows']):
            expected = dict(before)
            expected['observed_in'] = list(before['observed_in'])
            if before['split'] == 'evaluation':
                self.assertEqual(before['status'], 'public-evaluation-not-observed-in-cited-evidence')
                expected['status'] = 'observed-in-cited-evidence'
                expected['observed_in'].append(v5.EVIDENCE_ID)
            self.assertEqual(after, expected)
        raw = v5.canonical_snapshot_bytes()
        checked = v5.verify_snapshot_bytes(raw)
        self.assertEqual(checked.snapshot_sha256, v5.SNAPSHOT_SHA256)
        self.assertEqual((checked.observed_development_rows, checked.reserved_development_rows), (10, 22))
        self.assertEqual((checked.public_evaluation_rows, checked.observed_evaluation_rows), (64, 64))
        self.assertEqual(checked.exact_development_evaluation_overlap, 0)
        self.assertFalse(checked.complete_history_certified)
        self.assertFalse(checked.execution_authorized)
        with self.assertRaises(ValueError):
            v5.verify_snapshot_bytes(raw + b' ')
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual((repo / 'experiments/bert_agreement_exposures_v4.json').read_bytes(),
                         v4.canonical_snapshot_bytes())
        self.assertEqual((repo / 'experiments/bert_agreement_exposures_v5.json').read_bytes(), raw)

    def test_published_evaluation_contains_all_64_exact_rows(self):
        repo = Path(__file__).resolve().parents[1]
        root = repo / v5.EVIDENCE_ROOT
        result_raw = (root / 'evidence/result.json').read_bytes()
        exit_raw = (root / 'evidence/controller-exit.json').read_bytes()
        compressed = (root / 'evidence/events.jsonl.gz').read_bytes()
        self.assertEqual(hashlib.sha256(result_raw).hexdigest(), v5.RESULT_SHA256)
        self.assertEqual(hashlib.sha256(exit_raw).hexdigest(), v5.CONTROLLER_EXIT_SHA256)
        self.assertEqual(hashlib.sha256(compressed).hexdigest(), v5.JOURNAL_GZIP_SHA256)
        raw = gzip.decompress(compressed)
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (9490808, v5.JOURNAL_SHA256))
        result, exit_record = json.loads(result_raw), json.loads(exit_raw)
        self.assertEqual((result['status'], result['source_commit'], result['run_id'], result['split']),
                         ('complete', v5.SOURCE_COMMIT, v5.RUN_ID, 'evaluation'))
        self.assertEqual((result['owner_state'], result['owner_exit_code']), ('reaped', 0))
        self.assertEqual((result['cpu_supervision']['clean'], result['cpu_supervision']['exit_code']), (True, 0))
        self.assertEqual((exit_record['run_id'], exit_record['exit_code']), (v5.RUN_ID, 0))
        self.assertEqual(result['journal_sha256'], v5.JOURNAL_SHA256)
        self.assertEqual(result['evaluation_ledger']['prior_attempts'], [])
        expected = {row['row_id']: row for row in v4.build_snapshot()['rows'] if row['split'] == 'evaluation'}
        self.assertEqual(len(expected), 64)
        events = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(Counter(row['status'] for row in events),
                         {'started': 1120, 'completed': 1120, 'control': 384})
        self.assertTrue(all(row['split'] == 'evaluation' for row in events))
        completed = [row for row in events if row['status'] == 'completed']
        self.assertEqual({row['row_id'] for row in completed}, set(expected))
        for event in completed:
            row = expected[event['row_id']]
            self.assertTrue(all(type(value) is int for value in event['model_input_ids']))
            self.assertTrue(all(type(value) is int for value in event['model_attention_mask']))
            self.assertEqual(event['model_input_ids'], row['input_ids'])
            self.assertEqual(event['model_attention_mask'], row['attention_mask'])


if __name__ == '__main__':
    unittest.main()
