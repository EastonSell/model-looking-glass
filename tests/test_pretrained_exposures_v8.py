"""Check the public exposure update and retained evidence without model work."""
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v7 as v7
from looking_glass import pretrained_number_commitments as commitments


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_COMMIT = 'ebe5caaae8394d210265ab65ab84c295b6463580'
MANIFEST_SHA256 = '0398bd998fae4ad8ef473264fca4c8b5dcf2c63424e086e4f8c989a2f7cb819b'
RESULT_SHA256 = 'e36732e0538f7b5a069fed337fd86eb725744663b3801d1d953907d0edb93a02'
ATTEMPT_SHA256 = '88026a17a65ddf7d303479bceba120517b456b90da3fefd0b5c6dd1090d26766'
SUPERVISOR_SHA256 = '2935b0dfbecf10a8171cd62d05a0856f86994c7cce975d9090022a2d74610c9a'
EVIDENCE_ROOT = 'results/distilgpt2-copy-head-eval-v1-20261028/'


class ExposureV8Tests(unittest.TestCase):
    def v8(self):
        try:
            return importlib.import_module('looking_glass.pretrained_exposures_v8')
        except ModuleNotFoundError as exc:
            if exc.name != 'looking_glass.pretrained_exposures_v8':
                raise
            self.fail('the v8 exposure successor has not been implemented')

    def test_only_the_32_evidenced_planned_rows_change_status(self):
        v8 = self.v8()
        old, new = v7.inventory_payload(), v8.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v7.INVENTORY_VERSION,
                                              'sha256': 'e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344'})
        self.assertEqual(len(old['records']), 138)
        self.assertEqual(len(new['records']), 138)
        changed = []
        for before, after in zip(old['records'], new['records']):
            if before['record_id'].startswith('planned-copy-evaluation-'):
                changed.append(tuple(before['token_ids']))
                self.assertEqual(before['category'], 'planned')
                self.assertEqual(after['category'], 'observed-saved')
                self.assertEqual(after['record_id'], before['record_id'].replace('planned-', 'observed-', 1))
                self.assertEqual(after['reference_commit'], EVIDENCE_COMMIT)
                self.assertEqual(after['references'], [EVIDENCE_ROOT + 'run/manifest.json',
                    EVIDENCE_ROOT + 'run/result.json', EVIDENCE_ROOT + 'run/attempt.jsonl',
                    EVIDENCE_ROOT + 'supervisor.json'])
                fields = {'record_id', 'category', 'provenance', 'references', 'reference_commit'}
                self.assertEqual({k: x for k, x in before.items() if k not in fields},
                                 {k: x for k, x in after.items() if k not in fields})
            else:
                self.assertEqual(before, after)
        self.assertEqual(changed, list(v7.PUBLIC_EVALUATION_ROWS))
        self.assertEqual(Counter(row['category'] for row in new['records']),
                         {'observed-saved': 104, 'observed-reported': 3, 'planned': 19,
                          'reference-only': 4, 'synthetic-only': 8})
        self.assertEqual(v7._forbidden_rows(), v8._forbidden_rows())
        self.assertEqual(len(v8._forbidden_rows()), 962)
        self.assertEqual(hashlib.sha256(v8.canonical_inventory_bytes()).hexdigest(), v8.inventory_digest())
        self.assertEqual(commitments.load_inventory_snapshot(v8.canonical_inventory_bytes(),
                                                              v8.inventory_digest()).sha256,
                         v8.inventory_digest())
        with mock.patch.object(v7, 'INVENTORY_SHA256', '0' * 64), self.assertRaises(ValueError):
            v8.inventory_payload()

    def test_evidence_confirms_the_reserved_rows_and_all_recorded_actions(self):
        v8 = self.v8()
        names = ['run/manifest.json', 'run/result.json', 'run/attempt.jsonl', 'supervisor.json']
        raw = [(ROOT / (EVIDENCE_ROOT + name)).read_bytes() for name in names]
        self.assertEqual([hashlib.sha256(data).hexdigest() for data in raw],
                         [MANIFEST_SHA256, RESULT_SHA256, ATTEMPT_SHA256, SUPERVISOR_SHA256])
        for name, data in zip(names, raw):
            saved = subprocess.run(['git', 'show', EVIDENCE_COMMIT + ':' + EVIDENCE_ROOT + name],
                                   cwd=ROOT, check=True, capture_output=True).stdout
            self.assertEqual(hashlib.sha256(saved).digest(), hashlib.sha256(data).digest())
        manifest, result, log, receipt = json.loads(raw[0]), json.loads(raw[1]), raw[2], json.loads(raw[3])
        events = [json.loads(line) for line in log.splitlines()]
        self.assertEqual((manifest['model_id'], manifest['model_revision']),
                         (v7.TARGET.model_id, v7.TARGET.model_revision))
        self.assertEqual(manifest['exposure_inventory_sha256'], v7.INVENTORY_SHA256)
        self.assertEqual(manifest['inventory_sha256'], v7.EVALUATION_CASES_SHA256)
        self.assertEqual([tuple(row['token_ids']) for row in manifest['cases']], list(v7.PUBLIC_EVALUATION_ROWS))
        self.assertEqual([row['target_id'] for row in manifest['cases']],
                         [v7.PUBLIC_EVALUATION_GROUPS[index // 4][5] for index in range(32)])
        actions = ['clean', 'noop', 'candidate_zero', 'control_zero', 'clean_after']
        self.assertEqual(manifest['actions'], actions)
        self.assertEqual((result['status'], result['submitted_forwards'], result['returned_forwards']),
                         ('complete', 160, 160))
        self.assertEqual(result['manifest_sha256'], MANIFEST_SHA256)
        self.assertEqual((receipt['process_exit_code'], receipt['interrupted'], receipt['failure_file_present']),
                         (0, False, False))
        self.assertEqual([receipt['run_sha256'][Path(name).name] for name in names[:3]],
                         [MANIFEST_SHA256, RESULT_SHA256, ATTEMPT_SHA256])
        self.assertEqual(events[0]['event'], 'runtime_ready')
        self.assertEqual(len(events), 481)
        self.assertEqual(len(result['cases']), 160)
        for index, case in enumerate(manifest['cases']):
            for action_index, action in enumerate(actions):
                offset = 5 * index + action_index
                start, recorded, complete = events[1 + 3 * offset:4 + 3 * offset]
                self.assertEqual([(event['event'], event['case_id'], event['action'])
                                  for event in (start, recorded, complete)],
                                 [(kind, case['case_id'], action) for kind in ('start', 'raw_recorded', 'complete')])
                self.assertEqual((result['cases'][offset]['case_id'], result['cases'][offset]['action']),
                                 (case['case_id'], action))
        self.assertEqual(v8.EVIDENCE_COMMIT, EVIDENCE_COMMIT)

    def test_latest_checks_reject_observed_rows_without_authorizing_a_seal(self):
        v8 = self.v8()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {'expected_inventory_sha256': v8.inventory_digest(), 'expected_model_identity': v8.TARGET}
        check = v8.check_known_splits(discovery, heldout, cases_per_split=16, **options)
        self.assertEqual((check.inventory_version, check.candidate_count, check.forbidden_row_count),
                         ('distilgpt2-development-exposures-v8', 32, 962))
        self.assertTrue(set(v7.PUBLIC_EVALUATION_ROWS) <= v8._forbidden_rows())
        for i, observed in enumerate((v7.PUBLIC_EVALUATION_ROWS[0], v7.PUBLIC_EVALUATION_ROWS[-1])):
            candidate = [row[:] for row in heldout]
            candidate[i % 16] = list(observed)
            with self.subTest(i=i), self.assertRaises(ValueError):
                v8.check_exposed_discovery(discovery, candidate, **options)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v8.require_preseal_readiness(check)
        disclosure = v8.prepare_preseal_disclosure(
            check, protocol_sha256='a' * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertFalse(disclosure.execution_authorized)
        self.assertFalse(disclosure.sealed_manifest_verified)


if __name__ == '__main__':
    unittest.main()
