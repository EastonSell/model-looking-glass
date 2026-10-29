"""Public saved-exposure evidence and synthetic split checks; no model work."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v6 as v6
from looking_glass import pretrained_exposures_v7 as v7
from looking_glass import pretrained_number_commitments as commitments


ROOT = Path(__file__).resolve().parents[1]


class ExposureV7Tests(unittest.TestCase):
    def test_predecessor_and_exact_status_transitions(self):
        self.assertEqual(v6.inventory_digest(), v7.PREDECESSOR_SHA256)
        self.assertEqual(v7.inventory_digest(), v7.INVENTORY_SHA256)
        self.assertEqual(hashlib.sha256(v7.canonical_inventory_bytes()).hexdigest(), v7.INVENTORY_SHA256)
        old, new = v6.inventory_payload(), v7.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v6.INVENTORY_VERSION, 'sha256': v6.INVENTORY_SHA256})
        self.assertEqual(len(new['records']), len(old['records']) + 32)
        changed = 0
        for before, after in zip(old['records'], new['records'][:len(old['records'])]):
            if before['record_id'].startswith('planned-copy-screen-'):
                changed += 1
                self.assertEqual(after['record_id'], before['record_id'].replace('planned-', 'observed-', 1))
                self.assertEqual((before['category'], after['category']), ('planned', 'observed-saved'))
                self.assertEqual(after['token_ids'], before['token_ids'])
                self.assertEqual(after['target_token_id'], before['target_token_id'])
                self.assertEqual(after['reference_commit'], v7.EVIDENCE_COMMIT)
                self.assertEqual(after['references'], list(v7.EVIDENCE_PATHS))
                changed_fields = {'record_id', 'category', 'reference_commit', 'references', 'provenance'}
                self.assertEqual({k: x for k, x in before.items() if k not in changed_fields},
                                 {k: x for k, x in after.items() if k not in changed_fields})
            else:
                self.assertEqual(after, before)
        self.assertEqual(changed, 32)
        added = new['records'][len(old['records']):]
        self.assertEqual([tuple(row['token_ids']) for row in added], list(v7.PUBLIC_EVALUATION_ROWS))
        self.assertEqual([row['target_token_id'] for row in added],
                         [v7.PUBLIC_EVALUATION_GROUPS[index // 4][5] for index in range(32)])
        self.assertTrue(all(row['category'] == 'planned'
                            and row['reference_commit'] == v7.EVALUATION_PLAN_COMMIT
                            and row['references'] == list(v7.EVALUATION_PLAN_PATHS) for row in added))
        self.assertEqual(Counter(row['category'] for row in new['records']),
                         {'observed-saved': 72, 'observed-reported': 3, 'planned': 51,
                          'reference-only': 4, 'synthetic-only': 8})
        self.assertTrue(v6._forbidden_rows() <= v7._forbidden_rows())
        self.assertEqual((len(v7._forbidden_rows()), len(v7._forbidden_rows() - v6._forbidden_rows())),
                         (962, 352))
        with mock.patch.object(v6, 'INVENTORY_SHA256', '0' * 64), self.assertRaises(ValueError):
            v7.inventory_digest()

    def test_synthetic_split_checks_and_disclosure_remain_bounded(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {'expected_inventory_sha256': v7.INVENTORY_SHA256, 'expected_model_identity': v7.TARGET}
        check = v7.check_known_splits(discovery, heldout, cases_per_split=16, **options)
        self.assertEqual((check.candidate_count, check.forbidden_row_count), (32, 962))
        self.assertTrue(set(v6.PUBLIC_COPY_ROWS + v7.PUBLIC_EVALUATION_ROWS) <= v7._forbidden_rows())
        for index, row in enumerate((v6.PUBLIC_COPY_ROWS[0], v7.PUBLIC_EVALUATION_ROWS[0])):
            sample = [x[:] for x in heldout]
            sample[index % 16] = list(row)
            with self.subTest(index=index), self.assertRaises(ValueError):
                v7.check_exposed_discovery(discovery, sample, **options)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v7.require_preseal_readiness(check)
        disclosure = v7.prepare_preseal_disclosure(
            check, protocol_sha256='a' * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertFalse(disclosure.execution_authorized)
        self.assertFalse(disclosure.sealed_manifest_verified)
        snapshot = commitments.load_inventory_snapshot(v7.canonical_inventory_bytes(), v7.INVENTORY_SHA256)
        self.assertEqual(snapshot.sha256, v7.INVENTORY_SHA256)

    def test_published_records_match_every_reserved_input_and_completion(self):
        manifest_path, result_path, attempt_path = v7.EVIDENCE_PATHS
        manifest_raw, result_raw, attempt_raw = [(ROOT / path).read_bytes()
                                                  for path in (manifest_path, result_path, attempt_path)]
        self.assertEqual(hashlib.sha256(manifest_raw).hexdigest(), v7.MANIFEST_SHA256)
        self.assertEqual(hashlib.sha256(result_raw).hexdigest(), v7.RESULT_SHA256)
        self.assertEqual(hashlib.sha256(attempt_raw).hexdigest(), v7.ATTEMPT_SHA256)
        manifest, result = json.loads(manifest_raw), json.loads(result_raw)
        events = [json.loads(line) for line in attempt_raw.splitlines()]
        self.assertEqual((manifest['model_id'], manifest['model_revision'], manifest['source_revision']),
                         (v7.TARGET.model_id, v7.TARGET.model_revision, v6.PLAN_COMMIT))
        self.assertEqual([tuple(case['token_ids']) for case in manifest['cases']], list(v6.PUBLIC_COPY_ROWS))
        self.assertEqual([case['target_id'] for case in manifest['cases']],
                         [v6.PUBLIC_COPY_GROUPS[index // 4][5] for index in range(32)])
        self.assertEqual((manifest['expected_forwards'], result['submitted_forwards'],
                          result['root_outputs_returned']), (33, 33, 33))
        self.assertEqual(result['manifest_sha256'], v7.MANIFEST_SHA256)
        self.assertEqual(len(events), 66)
        first_id = manifest['cases'][0]['case_id']
        self.assertEqual((events[0]['event'], events[0]['case_id'], events[0]['mode']),
                         ('start', first_id, 'uncaptured'))
        self.assertEqual((events[1]['event'], events[1]['case_id'], events[1]['mode']),
                         ('complete', first_id, 'uncaptured'))
        self.assertEqual(len(result['cases']), 32)
        for index, case in enumerate(manifest['cases']):
            started, completed = events[2 + 2 * index:4 + 2 * index]
            saved = result['cases'][index]
            self.assertEqual((started['event'], started['case_id'], completed['event'], completed['case_id'],
                              saved['case_id']), ('start', case['case_id'], 'complete', case['case_id'],
                                                  case['case_id']))
            self.assertEqual((completed['logits_sha256'], completed['attention_sha256']),
                             (saved['logits_sha256'], saved['attention_sha256']))

    def test_published_evaluation_plan_matches_every_planned_row(self):
        path = ROOT / 'experiments/distilgpt2_copying_head_eval_v1.json'
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), v7.EVALUATION_CASES_SHA256)
        data = json.loads(raw)
        self.assertEqual((data['model_id'], data['model_revision']),
                         (v7.TARGET.model_id, v7.TARGET.model_revision))
        self.assertEqual(data['screen_result_sha256'], v7.RESULT_SHA256)
        self.assertEqual([tuple(group['token_ids']) for group in data['groups']],
                         list(v7.PUBLIC_EVALUATION_GROUPS))
        self.assertEqual([tuple(case['token_ids']) for case in data['cases']],
                         list(v7.PUBLIC_EVALUATION_ROWS))
        self.assertEqual([case['target_id'] for case in data['cases']],
                         [v7.PUBLIC_EVALUATION_GROUPS[index // 4][5] for index in range(32)])
        self.assertEqual([(case['group'], case['variant']) for case in data['cases']],
                         [(index // 4 + 1, index % 4) for index in range(32)])


if __name__ == '__main__':
    unittest.main()
