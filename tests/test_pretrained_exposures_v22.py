"""Check the grouped pilot's four status transitions without model work."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v21 as v21
from looking_glass import pretrained_exposures_v22 as v22

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def git_bytes(commit, path):
    return subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


def row_digest(row):
    records = [
        {'name': 'attention_mask', 'dtype': 'int64', 'shape': [1, 4],
         'values': [[1, 1, 1, 1]]},
        {'name': 'input_ids', 'dtype': 'int64', 'shape': [1, 4], 'values': [list(row)]},
    ]
    return hashlib.sha256(canonical(records)).hexdigest()


class ExposureV22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v21.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v21.INVENTORY_SHA256:
            raise AssertionError('predecessor payload differs from its pin')

    def test_public_pilot_projection_binds_all_four_reported_rows(self):
        raw = [git_bytes(v22.EVIDENCE_COMMIT, p) for p in v22.EVIDENCE_PATHS]
        self.assertEqual([hashlib.sha256(x).hexdigest() for x in raw], list(v22.EVIDENCE_SHA256S))
        protocol_source, projection, operation, replay, source_audit = map(json.loads, raw)
        protocol_bytes = protocol_source['protocol_utf8'].encode('utf-8')
        self.assertEqual(hashlib.sha256(protocol_bytes).hexdigest(), v22.PROTOCOL_SHA256)
        self.assertEqual(git_bytes(v22.SOURCE_COMMIT, v22.PROTOCOL_PATH), protocol_bytes)
        protocol = json.loads(protocol_bytes)
        self.assertEqual((protocol['model'], protocol['revision']),
                         (v22.TARGET.model_id, v22.TARGET.model_revision))
        self.assertEqual(tuple(tuple(row) for group in protocol['broker_holdout_groups'] for row in group),
                         v22.GROUPED_BROKER_OBSERVED_ROWS)
        self.assertEqual(tuple(tuple(row) for group in protocol['policy_groups'] for row in group),
                         v22.GROUPED_POLICY_ROWS)
        self.assertEqual(projection['original_result_sha256'], v22.ORIGINAL_RESULT_SHA256)
        result = projection['result']
        self.assertEqual((result['status'], result['source_commit'], result['protocol_sha256']),
                         ('passed', v22.SOURCE_COMMIT, v22.PROTOCOL_SHA256))
        self.assertEqual(result['supervisor'], {'clean': True, 'exit_code': 0, 'timed_out': False})
        measurement = result['measurement']
        self.assertEqual(measurement['status'], 'passed')
        self.assertEqual([entry['request_id'] for entry in measurement['model_entries']],
                         protocol['forward_order'])
        for entry in measurement['model_entries']:
            expected_rows = 2 if entry['request_id'].startswith(('mixed-', 'heldout-group-')) else 1
            self.assertEqual(entry['entry_shapes'], [[expected_rows, 4]])
        comparisons = [case for group in measurement['holdout'] for case in group['comparisons']]
        self.assertEqual(len(comparisons), 4)
        for i, (case, row) in enumerate(zip(comparisons, v22.GROUPED_BROKER_OBSERVED_ROWS)):
            self.assertEqual(case['case_id'], f'holdout-{i // 2}-{i % 2}')
            self.assertEqual(case['job_slot'], i)
            self.assertEqual(case['input_row_sha256'], row_digest(row))
            self.assertEqual((case['status'], case['own_reference']['status'], case['other_reference']['status']),
                             ('matched', 'matched', 'mismatch'))
        self.assertEqual((operation['source_commit'], operation['controller_exit_observed_by_coordinator'],
                          operation['result_sha256']),
                         (v22.SOURCE_COMMIT, 0, v22.ORIGINAL_RESULT_SHA256))
        self.assertEqual(operation['saved_supervisor'], {'clean': True, 'owner_exit_code': 0, 'timed_out': False})
        self.assertEqual((replay['status'], replay['model_entries'], replay['singleton_entries'],
                          replay['two_row_entries'], replay['source_commit']),
                         ('passed', 12, 8, 4, v22.SOURCE_COMMIT))
        self.assertEqual((source_audit['status'], source_audit['source_commit'], source_audit['protocol_sha256']),
                         ('passed', v22.SOURCE_COMMIT, v22.PROTOCOL_SHA256))

    def test_four_status_changes_preserve_input_prefixes_and_other_records(self):
        with mock.patch.object(v21, 'inventory_payload', return_value=self.predecessor):
            new = v22.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v21.INVENTORY_VERSION, 'sha256': v21.INVENTORY_SHA256})
        self.assertEqual(new['update'], {'observed_grouped_broker_rows': 4})
        old_by_id = {r['record_id']: r for r in self.predecessor['records']}
        new_by_id = {r['record_id']: r for r in new['records']}
        expected_old = {f'planned-grouped-broker-check-{i:02d}' for i in range(1, 5)}
        expected_new = {f'observed-grouped-broker-check-{i:02d}' for i in range(1, 5)}
        self.assertEqual(set(old_by_id) - set(new_by_id), expected_old)
        self.assertEqual(set(new_by_id) - set(old_by_id), expected_new)
        for key in set(old_by_id) - expected_old:
            self.assertEqual(old_by_id[key], new_by_id[key])
        for i, row in enumerate(v22.GROUPED_BROKER_OBSERVED_ROWS, 1):
            old = old_by_id[f'planned-grouped-broker-check-{i:02d}']
            actual = new_by_id[f'observed-grouped-broker-check-{i:02d}']
            self.assertEqual((old['category'], actual['category']), ('planned', 'observed-saved'))
            self.assertEqual((tuple(old['token_ids']), tuple(actual['token_ids'])), (row, row))
            self.assertEqual(actual['references'], list(v22.EVIDENCE_PATHS))
            self.assertEqual(actual['reference_commit'], v22.EVIDENCE_COMMIT)
        self.assertEqual(len(new['records']), len(self.predecessor['records']))
        self.assertEqual(v22._prefixes(new['records']), v22._prefixes(self.predecessor['records']))
        self.assertEqual(len(v22._prefixes(new['records'])), 3888)
        self.assertEqual(new['additional_observations'][:-1], self.predecessor['additional_observations'])
        self.assertEqual(new['additional_observations'][-1]['record_ids'],
                         [f'observed-batch-{i}' for i in range(103, 107)])
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v22.INVENTORY_SHA256)

    def test_predecessor_or_reserved_rows_cannot_drift(self):
        changed = dict(self.predecessor)
        changed['records'] = [dict(r) for r in self.predecessor['records']]
        target = next(r for r in changed['records'] if r['record_id'] == 'planned-grouped-broker-check-01')
        target['token_ids'] = [1, 2, 3, 4]
        digest = hashlib.sha256(canonical(changed)).hexdigest()
        with mock.patch.object(v21, 'inventory_payload', return_value=changed), \
                mock.patch.object(v21, 'INVENTORY_SHA256', digest), \
                mock.patch.object(v22, 'PREDECESSOR_SHA256', digest):
            with self.assertRaises(ValueError):
                v22.inventory_payload()
        with mock.patch.object(v21, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v21, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v22.inventory_payload()

    def test_known_split_checks_do_not_authorize_seal(self):
        with mock.patch.object(v21, 'inventory_payload', return_value=self.predecessor):
            payload = v22.inventory_payload()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        kwargs = {'expected_inventory_sha256': v22.INVENTORY_SHA256,
                  'expected_model_identity': v22.TARGET}
        with mock.patch.object(v22, 'inventory_payload', return_value=payload):
            check = v22.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            heldout[0] = list(v22.GROUPED_BROKER_OBSERVED_ROWS[0]) + [123]
            with self.assertRaises(ValueError):
                v22.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v22.require_preseal_readiness(check)


if __name__ == '__main__':
    unittest.main()
