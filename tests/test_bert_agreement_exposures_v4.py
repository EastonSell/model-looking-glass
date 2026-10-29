"""No-model checks for the immutable BERT GPU development exposure overlay."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from looking_glass import bert_agreement_exposures_v2 as v2
from looking_glass import bert_agreement_exposures_v3 as v3
from looking_glass import bert_agreement_exposures_v4 as v4


class BertExposureV4Test(unittest.TestCase):
    def test_snapshot_is_exact_and_preserves_all_prior_rows(self):
        previous, current = v3.build_snapshot(), v4.build_snapshot()
        self.assertEqual(current['predecessor'], {'schema': v3.SCHEMA, 'sha256': v4.PREDECESSOR_SHA256})
        self.assertEqual(len(previous['rows']), 96)
        self.assertEqual(len(current['rows']), 96)
        self.assertEqual(current['evidence'][:-1], previous['evidence'])
        for before, after in zip(previous['rows'], current['rows']):
            expected = dict(before)
            expected['observed_in'] = list(before['observed_in'])
            if before['row_id'] in v2.FAMILY_ONE_IDS:
                expected['observed_in'].append(v4.EVIDENCE_ID)
            self.assertEqual(after, expected)
        raw = v4.canonical_snapshot_bytes()
        self.assertEqual(v4.verify_snapshot_bytes(raw).snapshot_sha256, v4.SNAPSHOT_SHA256)
        with self.assertRaises(ValueError):
            v4.verify_snapshot_bytes(raw + b' ')
        path = Path(__file__).resolve().parents[1] / 'experiments/bert_agreement_exposures_v4.json'
        self.assertEqual(path.read_bytes(), raw)

    def test_published_gpu_attempt_records_only_the_eight_existing_dev_rows(self):
        repo = Path(__file__).resolve().parents[1]
        root = repo / v4.EVIDENCE_ROOT
        result_raw = (root / 'result.json').read_bytes()
        journal_raw = (root / 'evidence/events.jsonl').read_bytes()
        self.assertEqual(hashlib.sha256(result_raw).hexdigest(), v4.RESULT_SHA256)
        self.assertEqual(hashlib.sha256(journal_raw).hexdigest(), v4.JOURNAL_SHA256)
        result = json.loads(result_raw)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['source_commit'], v4.SOURCE_COMMIT)
        self.assertEqual(result['run_id'], v4.RUN_ID)
        self.assertEqual(result['supervisor'], {'clean': True, 'exit_code': 0, 'timed_out': False})
        self.assertEqual(result['measurement']['profile']['forward_count'], 140)
        self.assertEqual(result['measurement']['profile']['journal_sha256'], v4.JOURNAL_SHA256)
        events = [json.loads(line) for line in journal_raw.splitlines()]
        self.assertEqual(Counter(row.get('status') for row in events),
                         {'started': 140, 'completed': 140, 'control': 48})
        self.assertTrue(all(row.get('split') == 'development' and row.get('family') == 1 for row in events))
        self.assertEqual({row['row_id'] for row in events if row.get('row_id') is not None}, v2.FAMILY_ONE_IDS)
        self.assertEqual({row['donor_row_id'] for row in events if row.get('donor_row_id') is not None},
                         v2.FAMILY_ONE_IDS)
        completed = [row for row in events if row.get('status') == 'completed']
        self.assertEqual({row['forward_index'] for row in completed}, set(range(1, 141)))


if __name__ == '__main__':
    unittest.main()
