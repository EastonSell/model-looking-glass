"""Verify the fixed screened-edit reservation without model work."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v18 as v18
from looking_glass import pretrained_exposures_v19 as v19

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def git_bytes(commit, path):
    return subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


class ExposureV19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v18.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v18.INVENTORY_SHA256:
            raise AssertionError('the predecessor differs from its pin')

    def test_public_cases_and_private_comparison_bind_exact_rows(self):
        case_raw = git_bytes(v19.CASES_COMMIT, v19.CASE_PATH)
        protocol = git_bytes(v19.PROTOCOL_COMMIT, v19.PROTOCOL_PATH)
        receipt_raw = git_bytes(v19.CASES_COMMIT, v19.RECEIPT_PATH)
        self.assertEqual(hashlib.sha256(case_raw).hexdigest(), v19.CASE_SHA256)
        self.assertEqual(hashlib.sha256(protocol).hexdigest(), v19.PROTOCOL_SHA256)
        self.assertEqual(hashlib.sha256(receipt_raw).hexdigest(), v19.RECEIPT_SHA256)
        subprocess.run(['git', 'merge-base', '--is-ancestor', v19.PROTOCOL_COMMIT,
                        v19.CASES_COMMIT], cwd=ROOT, check=True)
        data = json.loads(case_raw)
        self.assertEqual((data['model_id'], data['revision']), (v19.TARGET.model_id, v19.TARGET.model_revision))
        self.assertEqual((data['protocol_commit'], data['protocol_sha256']),
                         (v19.PROTOCOL_COMMIT, v19.PROTOCOL_SHA256))
        self.assertEqual((data['inventory_version'], data['inventory_sha256']),
                         ('v17', v18.PREDECESSOR_SHA256))
        self.assertIs(data['model_forward'], False)
        cases = [(c['case_id'], c['prompt'], tuple(c['token_ids'])) for c in data['cases']]
        self.assertEqual(cases, list(v19.SCREENED_CASES))
        self.assertEqual(tuple(row for _, _, row in cases), v19.SCREENED_ROWS)
        self.assertEqual((len(cases), len(set(v19.SCREENED_ROWS))), (44, 44))
        adapted = canonical({'cases': [{'input_ids': c['token_ids']} for c in data['cases']]})
        self.assertEqual(hashlib.sha256(adapted).hexdigest(), v19.COMPARISON_INPUT_SHA256)
        receipt = json.loads(receipt_raw)
        self.assertEqual((receipt['status'], receipt['comparison'], receipt['public_case_count'],
                          receipt['natural_case_count'], receipt['public_input_file_sha256']),
                         ('clear', 'bidirectional-complete-row-prefix', 44, 16, v19.COMPARISON_INPUT_SHA256))
        self.assertEqual(receipt['natural_heldout_commitment'],
                         '8b475c373034201e5c36186207edbc56f0c8b62221f497d8d09faad1ce182150')
        self.assertEqual(receipt['natural_prediction_freeze_sha256'],
                         '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb')
        self.assertEqual(receipt['comparator_source_sha256'],
                         'd4ad193af9a8f0b19ae5a185f99067ab1c8b8c1671daeda3b0b35321fa0de3a2')

    def test_predecessor_records_preserved_and_44_prompts_planned(self):
        with mock.patch.object(v18, 'inventory_payload', return_value=self.predecessor):
            new = v19.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v18.INVENTORY_VERSION,
                                               'sha256': v18.INVENTORY_SHA256})
        self.assertEqual(new['records'][:415], self.predecessor['records'])
        self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
        self.assertEqual(new['update'], {'planned_screened_edit_rows': 44})
        added = new['records'][415:]
        self.assertEqual(len(new['records']), 459)
        self.assertEqual([(r['record_id'], r['public_text'], tuple(r['token_ids'])) for r in added],
                         [('planned-' + case_id, text, row) for case_id, text, row in v19.SCREENED_CASES])
        self.assertTrue(all(r['category'] == 'planned' and r['reference_commit'] == v19.CASES_COMMIT
                            and r['references'] == [v19.PROTOCOL_PATH, v19.CASE_PATH, v19.RECEIPT_PATH]
                            for r in added))
        old_prefixes, new_prefixes = v19._prefixes(self.predecessor['records']), v19._prefixes(new['records'])
        self.assertLessEqual(old_prefixes, new_prefixes)
        self.assertEqual((len(old_prefixes), len(new_prefixes)), (3275, 3522))
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v19.INVENTORY_SHA256)

    def test_rejects_known_prefix_relations_in_both_directions(self):
        candidate = list(v19.SCREENED_ROWS[0])
        for prior in (candidate[:-1], candidate, candidate + [123]):
            changed = copy.deepcopy(self.predecessor)
            changed['records'][0]['token_ids'] = prior
            changed_digest = hashlib.sha256(canonical(changed)).hexdigest()
            with self.subTest(length=len(prior)), \
                    mock.patch.object(v18, 'inventory_payload', return_value=changed), \
                    mock.patch.object(v18, 'INVENTORY_SHA256', changed_digest), \
                    mock.patch.object(v19, 'PREDECESSOR_SHA256', changed_digest):
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    v19.inventory_payload()
        with mock.patch.object(v18, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v18, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v19.inventory_payload()
        cases = list(v19.SCREENED_CASES)
        case_id, text, _ = cases[1]
        cases[1] = (case_id, text, v19.SCREENED_ROWS[0] + (123,))
        with mock.patch.object(v18, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v19, 'SCREENED_CASES', tuple(cases)):
            with self.assertRaisesRegex(ValueError, 'prefix'):
                v19.inventory_payload()

    def test_known_splits_exclude_new_rows_without_authorizing_seal(self):
        with mock.patch.object(v18, 'inventory_payload', return_value=self.predecessor):
            payload = v19.inventory_payload()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        kwargs = {'expected_inventory_sha256': v19.INVENTORY_SHA256,
                  'expected_model_identity': v19.TARGET}
        with mock.patch.object(v19, 'inventory_payload', return_value=payload):
            check = v19.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            for candidate in (list(v19.SCREENED_ROWS[0][:-1]), list(v19.SCREENED_ROWS[0]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = candidate
                with self.assertRaises(ValueError):
                    v19.check_known_splits(discovery, altered, cases_per_split=16, **kwargs)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v19.require_preseal_readiness(check)


if __name__ == '__main__':
    unittest.main()
