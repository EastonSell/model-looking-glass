"""Record an operator-reported public exposure without model execution."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v17 as v17
from looking_glass import pretrained_exposures_v18 as v18

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


class ExposureV18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v17.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v17.INVENTORY_SHA256:
            raise AssertionError('real predecessor differs from its pin')

    def test_retrospective_binds_the_reported_rows_and_its_limits(self):
        raw = (ROOT / v18.RETROSPECTIVE_PATH).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), v18.RETROSPECTIVE_SHA256)
        data = json.loads(raw)
        self.assertEqual(data['schema'], 'grouped-cpu-unreserved-exposure-retrospective-v1')
        self.assertEqual((data['model'], data['revision']),
                         (v18.TARGET.model_id, v18.TARGET.model_revision))
        self.assertEqual(tuple(tuple(row) for row in data['rows']), v18.GROUPED_REPORTED_ROWS)
        self.assertEqual(len(data['expected_forward_order']), 12)
        self.assertIn('retrospective', data['evidence_kind'])
        self.assertIn('no raw logits or per-forward logs retained', data['evidence_kind'])
        self.assertIn('exact executed file bytes were not recorded', data['source_state'])
        self.assertIn('not recorded', data['execution_time'])

    def test_private_comparison_receipt_binds_the_same_reported_rows(self):
        directory = ROOT / 'results/distilgpt2-grouped-unreserved-cpu-v1'
        inputs = (directory / 'comparison-inputs.json').read_bytes()
        receipt = (directory / 'comparison-receipt.json').read_bytes()
        inputs_hash = 'b436a0487cb630c5f96c7f83a9098b7654e7b1c06917662b13e0988d6cd26aa9'
        self.assertEqual(hashlib.sha256(inputs).hexdigest(), inputs_hash)
        self.assertEqual(hashlib.sha256(receipt).hexdigest(),
                         'ac8205584df6e045025a3f0edc72b5303cffb3c385bbfbb445044f25e68f46dd')
        rows = json.loads(inputs)['cases']
        self.assertEqual(tuple(tuple(case['input_ids']) for case in rows), v18.GROUPED_REPORTED_ROWS)
        record = json.loads(receipt)
        self.assertEqual((record['status'], record['comparison'], record['public_case_count'],
                          record['natural_case_count'], record['public_input_file_sha256']),
                         ('clear', 'bidirectional-complete-row-prefix', 8, 16, inputs_hash))
        self.assertEqual(record['natural_heldout_commitment'],
                         '8b475c373034201e5c36186207edbc56f0c8b62221f497d8d09faad1ce182150')
        self.assertEqual(record['natural_prediction_freeze_sha256'],
                         '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb')
        self.assertEqual(record['comparator_source_sha256'],
                         'd4ad193af9a8f0b19ae5a185f99067ab1c8b8c1671daeda3b0b35321fa0de3a2')

    def test_prior_records_preserved_and_all_eight_additions_are_reported(self):
        with mock.patch.object(v17, 'inventory_payload', return_value=self.predecessor):
            new = v18.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v17.INVENTORY_VERSION,
                                               'sha256': v17.INVENTORY_SHA256})
        self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
        self.assertEqual(new['update'], {'operator_reported_grouped_cpu_rows': 8,
                                         'observed_year_transfer_rows': 48})
        changed = []
        for prior, current in zip(self.predecessor['records'], new['records'][:407]):
            if prior == current:
                continue
            changed.append(prior['record_id'])
            self.assertTrue(prior['record_id'].startswith('planned-year-transfer-'))
            self.assertEqual((prior['category'], current['category']), ('planned', 'observed-saved'))
            self.assertEqual(current['record_id'], prior['record_id'].replace('planned-', 'observed-', 1))
            self.assertEqual((current['token_ids'], current['public_text']),
                             (prior['token_ids'], prior['public_text']))
        self.assertEqual(len(changed), 48)
        added = new['records'][407:]
        self.assertEqual(len(new['records']), 415)
        self.assertEqual([tuple(r['token_ids']) for r in added], list(v18.GROUPED_REPORTED_ROWS))
        self.assertEqual([r['record_id'] for r in added],
                         [f'observed-grouped-unreserved-{i:02d}' for i in range(1, 9)])
        self.assertTrue(all(r['category'] == 'observed-reported' and
                            r['references'] == [v18.RETROSPECTIVE_PATH] and
                            r['reference_commit'] == '' for r in added))
        prior_exact = [r for r in self.predecessor['records']
                       if tuple(r['token_ids']) == v18.GROUPED_REPORTED_ROWS[0]]
        self.assertTrue(any(r['record_id'] == 'synthetic-gpt2-control-1' for r in prior_exact))
        old_prefixes = v18._prefixes(self.predecessor['records'])
        new_prefixes = v18._prefixes(new['records'])
        self.assertLessEqual(old_prefixes, new_prefixes)
        self.assertEqual((len(old_prefixes), len(new_prefixes)), (3247, 3275))
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v18.INVENTORY_SHA256)

    def test_year_transfer_saved_records_bind_all_48_reserved_cases(self):
        raw = [subprocess.run(['git', 'show', v18.YEAR_EVIDENCE_COMMIT + ':' + path], cwd=ROOT,
                              check=True, capture_output=True).stdout for path in v18.YEAR_EVIDENCE_PATHS]
        self.assertEqual([hashlib.sha256(b).hexdigest() for b in raw], list(v18.YEAR_EVIDENCE_SHA256S))
        manifest, result, journal, receipt = json.loads(raw[0]), json.loads(raw[1]), raw[2], json.loads(raw[3])
        self.assertEqual((result['status'], result['submitted_forwards'], result['model_call_returns_observed'],
                          result['returned_forwards']), ('complete', 336, 336, 336))
        self.assertEqual((receipt['process_exit_code'], receipt['interrupted'], receipt['failure_file_present']),
                         (0, False, False))
        self.assertEqual((result['manifest_sha256'], manifest['cases_sha256']),
                         (v18.YEAR_EVIDENCE_SHA256S[0], v18.YEAR_CASE_SHA256))
        for name, digest in zip(('manifest.json', 'result.json', 'attempt.jsonl'), v18.YEAR_EVIDENCE_SHA256S):
            self.assertEqual(receipt['run_sha256'][name], digest)
        cases = manifest['cases']
        self.assertEqual([(c['case_id'], c['text'], tuple(c['input_ids'])) for c in cases],
                         list(v18.YEAR_TRANSFER_CASES))
        events = [json.loads(line) for line in journal.splitlines()]
        expected = Counter({case['case_id']: 7 for case in cases})
        positions = {}
        for kind in ('start', 'raw_recorded', 'complete'):
            selected = [(i, event) for i, event in enumerate(events) if event['event'] == kind]
            self.assertEqual(len(selected), 336)
            self.assertEqual(Counter(event['case_id'] for _, event in selected), expected)
            positions[kind] = {(event['case_id'], event['action']): i for i, event in selected}
            self.assertEqual(len(positions[kind]), 336)
        self.assertEqual(set(positions['start']), set(positions['raw_recorded']))
        self.assertEqual(set(positions['start']), set(positions['complete']))
        for key in positions['start']:
            self.assertLess(positions['start'][key], positions['raw_recorded'][key])
            self.assertLess(positions['raw_recorded'][key], positions['complete'][key])
        completed = {key: {k: v for k, v in events[i].items() if k != 'event'}
                     for key, i in positions['complete'].items()}
        self.assertEqual({(record['case_id'], record['action']): record for record in result['records']},
                         completed)

    def test_predecessor_drift_and_duplicate_record_identity_are_rejected(self):
        with mock.patch.object(v17, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v17, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v18.inventory_payload()
        altered = copy.deepcopy(self.predecessor)
        altered['records'][0]['record_id'] = 'observed-grouped-unreserved-01'
        changed_digest = hashlib.sha256(canonical(altered)).hexdigest()
        with mock.patch.object(v17, 'inventory_payload', return_value=altered), \
                mock.patch.object(v17, 'INVENTORY_SHA256', changed_digest), \
                mock.patch.object(v18, 'PREDECESSOR_SHA256', changed_digest):
            with self.assertRaises(ValueError):
                v18.inventory_payload()
        rows = list(v18.GROUPED_REPORTED_ROWS)
        rows[1] = rows[0]
        with mock.patch.object(v17, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v18, 'GROUPED_REPORTED_ROWS', tuple(rows)):
            with self.assertRaises(ValueError):
                v18.inventory_payload()

    def test_known_split_checks_cover_reported_rows_without_authorizing_seal(self):
        with mock.patch.object(v17, 'inventory_payload', return_value=self.predecessor):
            payload = v18.inventory_payload()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        kwargs = {'expected_inventory_sha256': v18.INVENTORY_SHA256,
                  'expected_model_identity': v18.TARGET}
        with mock.patch.object(v18, 'inventory_payload', return_value=payload):
            check = v18.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            for row in v18.GROUPED_REPORTED_ROWS:
                altered = copy.deepcopy(heldout)
                altered[0] = list(row) + [123]
                with self.assertRaises(ValueError):
                    v18.check_known_splits(discovery, altered, cases_per_split=16, **kwargs)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v18.require_preseal_readiness(check)


if __name__ == '__main__':
    unittest.main()
