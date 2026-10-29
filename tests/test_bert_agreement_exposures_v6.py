"""No-model checks of the matched-study exposure successor and evidence."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import unittest

from looking_glass import bert_agreement_exposures_v5 as v5
from looking_glass import bert_agreement_exposures_v6 as v6
from looking_glass import bert_matched_cases as cases


class BertExposureV6Test(unittest.TestCase):
    def test_snapshot_preserves_prior_and_adds_exact_matched_rows(self):
        previous, current, manifest = v5.build_snapshot(), v6.build_snapshot(), cases.build_manifest()
        self.assertEqual(current['predecessor'], {'schema': v5.SCHEMA, 'sha256': v6.PREDECESSOR_SHA256})
        self.assertEqual(current['matched_pre_run_manifest'], {'schema': cases.SCHEMA, 'sha256': v6.MANIFEST_SHA256})
        self.assertEqual(current['rows'][:96], previous['rows'])
        self.assertEqual(current['evidence'][:-1], previous['evidence'])
        self.assertEqual(len(current['rows']), 160)
        for before, after in zip(manifest['rows'], current['rows'][96:]):
            self.assertEqual(after, {
                'row_id': before['row_id'], 'split': before['split'],
                'input_ids': before['input_ids'], 'attention_mask': before['attention_mask'],
                'source_manifest_sha256': v6.MANIFEST_SHA256,
                'source_exposure_status': 'public-evaluation-not-observed-in-cited-evidence',
                'status': 'observed-in-cited-evidence', 'observed_in': [v6.EVIDENCE_ID],
            })
        raw = v6.canonical_snapshot_bytes()
        checked = v6.verify_snapshot_bytes(raw)
        self.assertEqual(checked.snapshot_sha256, v6.SNAPSHOT_SHA256)
        self.assertEqual((checked.inherited_rows, checked.matched_rows, checked.observed_matched_rows), (96, 64, 64))
        self.assertEqual(checked.exact_prior_row_overlap, 0)
        self.assertFalse(checked.complete_history_certified)
        self.assertFalse(checked.execution_authorized)
        with self.assertRaises(ValueError):
            v6.verify_snapshot_bytes(raw + b' ')
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual((repo / 'experiments/bert_agreement_exposures_v5.json').read_bytes(),
                         v5.canonical_snapshot_bytes())
        self.assertEqual((repo / 'experiments/bert_matched_public_cases_v1.json').read_bytes(),
                         cases.canonical_manifest_bytes())
        self.assertEqual((repo / 'experiments/bert_agreement_exposures_v6.json').read_bytes(), raw)

    def test_published_attempt_contains_all_64_exact_rows(self):
        repo = Path(__file__).resolve().parents[1]
        root = repo / v6.EVIDENCE_ROOT
        result_raw = (root / 'controller-result.json').read_bytes()
        reservation_raw = (root / 'reservation.json').read_bytes()
        exit_raw = (root / 'controller-exit-observation.json').read_bytes()
        compressed = (root / 'events.jsonl.gz').read_bytes()
        for raw, digest in ((result_raw, v6.RESULT_SHA256), (reservation_raw, v6.RESERVATION_SHA256),
                            (exit_raw, v6.CONTROLLER_EXIT_SHA256), (compressed, v6.JOURNAL_GZIP_SHA256)):
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        raw = gzip.decompress(compressed)
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (6369065, v6.JOURNAL_SHA256))
        result, reservation, exit_record = map(json.loads, (result_raw, reservation_raw, exit_raw))
        self.assertEqual((result['status'], result['source_commit'], result['attempt_id'], result['manifest_sha256']),
                         ('complete_candidate', v6.SOURCE_COMMIT, v6.ATTEMPT_ID, v6.MANIFEST_SHA256))
        self.assertIs(result['owner_reaped'], True)
        self.assertEqual((result['owner_exit_code'], result['forward_count'], result['journal_sha256']),
                         (0, 480, v6.JOURNAL_SHA256))
        self.assertEqual((reservation['attempt_id'], exit_record['source_commit'], exit_record['exit_code']),
                         (v6.ATTEMPT_ID, v6.SOURCE_COMMIT, 0))
        self.assertIs(type(exit_record['exit_code']), int)
        self.assertEqual(exit_record['controller_pid'], reservation['controller_pid'])
        expected = {row['row_id']: row for row in cases.build_manifest()['rows']}
        self.assertEqual(len(expected), 64)
        events = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(Counter(row['status'] for row in events),
                         {'started': 480, 'completed': 480, 'control': 160, 'dose_plan': 32})
        self.assertTrue(all(row['passed'] is True for row in events if row['status'] == 'control'))
        completed = [row for row in events if row['status'] == 'completed']
        self.assertEqual({row['row_id'] for row in completed}, set(expected))
        for event in completed:
            row = expected[event['row_id']]
            self.assertEqual(event['split'], 'evaluation')
            self.assertTrue(all(type(value) is int for value in event['model_input_ids']))
            self.assertTrue(all(type(value) is int for value in event['model_attention_mask']))
            self.assertEqual(event['model_input_ids'], row['input_ids'])
            self.assertEqual(event['model_attention_mask'], row['attention_mask'])


if __name__ == '__main__':
    unittest.main()
