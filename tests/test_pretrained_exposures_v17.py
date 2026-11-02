"""Reconcile two completed public studies without forwarding a model."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v10 as v10
from looking_glass import pretrained_exposures_v16 as v16
from looking_glass import pretrained_exposures_v17 as v17

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def git_bytes(commit, path):
    return subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


class ExposureV17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v16.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v16.INVENTORY_SHA256:
            raise AssertionError('predecessor inventory differs from its pin')

    def test_only_evidenced_statuses_change_and_prefixes_are_preserved(self):
        with mock.patch.object(v16, 'inventory_payload', return_value=self.predecessor):
            new = v17.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v16.INVENTORY_VERSION,
                                               'sha256': v16.INVENTORY_SHA256})
        self.assertEqual(new['update'], {'observed_pairs_dose_rows': 32,
                                         'observed_projected_edit_rows': 13})
        self.assertEqual(len(new['records']), len(self.predecessor['records']))
        self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
        changed = []
        for old, current in zip(self.predecessor['records'], new['records']):
            if old == current:
                continue
            changed.append((old, current))
            self.assertEqual(old['category'], 'planned')
            self.assertEqual(current['category'], 'observed-saved')
            self.assertEqual(current['record_id'], old['record_id'].replace('planned-', 'observed-', 1))
            self.assertEqual(current['token_ids'], old['token_ids'])
            self.assertEqual(current['target_token_id'], old['target_token_id'])
            self.assertEqual(current['public_text'], old['public_text'])
        self.assertEqual(len(changed), 45)
        self.assertEqual(sum(old['record_id'].startswith('planned-copy-pairs-dose-') for old, _ in changed), 32)
        self.assertEqual(sum(old['record_id'].startswith('planned-projected-') for old, _ in changed), 13)
        self.assertEqual(v17._prefixes(new['records']), v17._prefixes(self.predecessor['records']))
        self.assertEqual(len(v17._prefixes(new['records'])), 3247)
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v17.INVENTORY_SHA256)

    def test_pairs_dose_saved_records_match_the_reserved_32_inputs(self):
        raw = [git_bytes(v17.PAIRS_EVIDENCE_COMMIT, path) for path in v17.PAIRS_EVIDENCE_PATHS]
        self.assertEqual([hashlib.sha256(b).hexdigest() for b in raw], list(v17.PAIRS_EVIDENCE_SHA256S))
        manifest, result, journal, receipt = json.loads(raw[0]), json.loads(raw[1]), raw[2], json.loads(raw[3])
        self.assertEqual((receipt['process_exit_code'], receipt['interrupted'], receipt['failure_file_present']),
                         (0, False, False))
        self.assertEqual((result['status'], result['submitted_forwards'], result['model_call_returns_observed'],
                          result['returned_forwards']), ('complete', 576, 576, 576))
        self.assertEqual(result['manifest_sha256'], v17.PAIRS_EVIDENCE_SHA256S[0])
        self.assertEqual(receipt['run_sha256']['manifest.json'], v17.PAIRS_EVIDENCE_SHA256S[0])
        self.assertEqual(receipt['run_sha256']['result.json'], v17.PAIRS_EVIDENCE_SHA256S[1])
        self.assertEqual(receipt['run_sha256']['attempt.jsonl'], v17.PAIRS_EVIDENCE_SHA256S[2])
        cases = manifest['cases']
        self.assertEqual(len(cases), 32)
        expected = {r['record_id']: r for r in self.predecessor['records']
                    if r['record_id'].startswith('planned-copy-pairs-dose-')}
        for case in cases:
            key = f"planned-copy-pairs-dose-{case['group']:02d}-{case['variant']}"
            prior = expected.pop(key)
            self.assertEqual((case['token_ids'], case['target_id'], case['text']),
                             (prior['token_ids'], prior['target_token_id'], prior['public_text']))
        self.assertFalse(expected)
        events = [json.loads(line) for line in journal.splitlines()]
        self.assertEqual(Counter(event['event'] for event in events)['start'], 576)
        self.assertEqual(Counter(event['event'] for event in events)['raw_recorded'], 576)
        self.assertEqual(Counter(event['event'] for event in events)['complete'], 576)
        for kind in ('start', 'raw_recorded', 'complete'):
            self.assertEqual(Counter(event['case_id'] for event in events if event['event'] == kind),
                             Counter({case['case_id']: 18 for case in cases}))

    def test_projected_saved_records_match_the_reserved_13_inputs(self):
        raw = [git_bytes(v17.PROJECTED_EVIDENCE_COMMIT, path) for path in v17.PROJECTED_EVIDENCE_PATHS]
        self.assertEqual([hashlib.sha256(b).hexdigest() for b in raw], list(v17.PROJECTED_EVIDENCE_SHA256S))
        manifest, phase, journal, child = json.loads(raw[0]), json.loads(raw[1]), raw[2], json.loads(raw[3])
        self.assertEqual((phase['status'], phase['job_drained'], phase['owner_reaped']), ('complete', True, True))
        self.assertEqual((phase['supervisor']['clean'], phase['supervisor']['exit_code'],
                          phase['supervisor']['timed_out']), (True, 0, False))
        self.assertEqual(phase['manifest_sha256'], v17.PROJECTED_EVIDENCE_SHA256S[0])
        self.assertEqual(child['manifest_sha256'], v17.PROJECTED_EVIDENCE_SHA256S[0])
        self.assertEqual(phase['child_result_sha256'], v17.PROJECTED_EVIDENCE_SHA256S[3])
        self.assertEqual(phase['artifacts']['evaluation/observations.jsonl']['sha256'],
                         v17.PROJECTED_EVIDENCE_SHA256S[2])
        expected = {r['record_id']: r for r in self.predecessor['records']
                    if r['record_id'].startswith('planned-projected-')}
        cases = manifest['evaluation_cases']
        self.assertEqual(len(cases), 13)
        for case in cases:
            prior = expected.pop('planned-' + case['case_id'])
            self.assertEqual((case['token_ids'], case['prompt']), (prior['token_ids'], prior['public_text']))
            self.assertEqual(case['continuations'][0]['ids'], [prior['target_token_id']])
        self.assertFalse(expected)
        observations = [json.loads(line) for line in journal.splitlines()]
        self.assertEqual(len(observations), 78)
        self.assertEqual(Counter(x['case_id'] for x in observations), Counter({x['case_id']: 6 for x in cases}))
        for case in cases:
            self.assertEqual({x['condition'] for x in observations if x['case_id'] == case['case_id']},
                             {'original', 'projected', 'unprojected', 'projected_random', 'opposite', 'nuisance'})

    def test_predecessor_drift_or_a_changed_reserved_row_is_rejected(self):
        for kind in ('wrong_digest', 'changed_row', 'missing_row', 'wrong_status', 'duplicate_row'):
            altered = copy.deepcopy(self.predecessor)
            item = next(r for r in altered['records'] if r['record_id'].startswith('planned-projected-'))
            if kind == 'changed_row':
                item['token_ids'][0] = (item['token_ids'][0] + 1) % 50257
            elif kind == 'missing_row':
                altered['records'].remove(item)
            elif kind == 'wrong_status':
                item['category'] = 'observed-saved'
            elif kind == 'duplicate_row':
                altered['records'].append(copy.deepcopy(item))
            changed_digest = hashlib.sha256(canonical(altered)).hexdigest()
            expected = '0' * 64 if kind == 'wrong_digest' else changed_digest
            with self.subTest(kind=kind), \
                    mock.patch.object(v16, 'inventory_payload', return_value=altered), \
                    mock.patch.object(v16, 'INVENTORY_SHA256', expected), \
                    mock.patch.object(v17, 'PREDECESSOR_SHA256', expected):
                with self.assertRaises(ValueError):
                    v17.inventory_payload()

    def test_known_split_checks_use_reconciled_rows_without_authorizing_a_seal(self):
        with mock.patch.object(v16, 'inventory_payload', return_value=self.predecessor):
            payload = v17.inventory_payload()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        kwargs = {'expected_inventory_sha256': v17.INVENTORY_SHA256,
                  'expected_model_identity': v17.TARGET}
        with mock.patch.object(v17, 'inventory_payload', return_value=payload):
            check = v17.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            for row in (v17.YEAR_TRANSFER_ROWS[0], v17.PAIRS_DOSE_ROWS[0], v10.PROJECTED_CASES[0][3]):
                altered = copy.deepcopy(heldout)
                altered[0] = list(row) + [123]
                with self.assertRaises(ValueError):
                    v17.check_known_splits(discovery, altered, cases_per_split=16, **kwargs)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v17.require_preseal_readiness(check)
        changed = copy.deepcopy(payload)
        changed['records'][0]['token_ids'][0] = -1
        with mock.patch.object(v17, 'inventory_payload', return_value=changed):
            with self.assertRaises(ValueError):
                v17.inventory_digest()


if __name__ == '__main__':
    unittest.main()
