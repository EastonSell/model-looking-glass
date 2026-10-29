"""Verify the explicit-year reservation using public data and no model."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v20 as v20
from looking_glass import pretrained_exposures_v21 as v21

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def git_bytes(commit, path):
    return subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


class ExposureV21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v20.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v20.INVENTORY_SHA256:
            raise AssertionError('predecessor payload differs from its pin')

    def test_public_cases_and_private_comparison_bind_exact_rows(self):
        protocol = git_bytes(v21.CASES_COMMIT, v21.PROTOCOL_PATH)
        case_bytes = git_bytes(v21.CASES_COMMIT, v21.CASE_PATH)
        self.assertEqual(hashlib.sha256(protocol).hexdigest(), v21.PROTOCOL_SHA256)
        self.assertEqual(hashlib.sha256(case_bytes).hexdigest(), v21.CASE_SHA256)
        subprocess.run(['git', 'merge-base', '--is-ancestor', v21.PROTOCOL_MERGE,
                        v21.CASES_COMMIT], cwd=ROOT, check=True)
        cases = json.loads(case_bytes)
        self.assertEqual((cases['model_id'], cases['model_revision']),
                         (v21.TARGET.model_id, v21.TARGET.model_revision))
        self.assertEqual((cases['protocol_revision'], cases['protocol_sha256']),
                         (v21.PROTOCOL_HEAD, v21.PROTOCOL_SHA256))
        self.assertEqual((cases['known_exposure_version'], cases['known_exposure_sha256']),
                         ('distilgpt2-development-exposures-v17',
                          'e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a'))
        parsed = [(c['case_id'], c['text'], tuple(c['input_ids']),
                   tuple(tuple(c[k]) for k in v21.CANDIDATE_GROUP_KEYS)) for c in cases['cases']]
        self.assertEqual(parsed, list(v21.EXPLICIT_YEAR_DIRECTION_CASES))
        self.assertEqual(tuple(row for _, _, row, _ in parsed), v21.EXPLICIT_YEAR_DIRECTION_ROWS)
        self.assertEqual((len(parsed), len(set(v21.EXPLICIT_YEAR_DIRECTION_ROWS))), (48, 48))
        self.assertTrue(all(len(row) == 13 for row in v21.EXPLICIT_YEAR_DIRECTION_ROWS))
        ordered = json.dumps([list(row) for _, _, row, _ in parsed], separators=(',', ':')).encode('ascii')
        self.assertEqual(hashlib.sha256(ordered).hexdigest(), v21.ORDERED_ROWS_SHA256)
        raw_input = (ROOT / v21.COMPARISON_INPUT_PATH).read_bytes()
        raw_receipt = (ROOT / v21.RECEIPT_PATH).read_bytes()
        self.assertEqual(hashlib.sha256(raw_input).hexdigest(), v21.COMPARISON_INPUT_SHA256)
        self.assertEqual(hashlib.sha256(raw_receipt).hexdigest(), v21.RECEIPT_SHA256)
        adapted, receipt = json.loads(raw_input), json.loads(raw_receipt)
        self.assertEqual([case['input_ids'] for case in adapted['cases']],
                         [list(row) for _, _, row, _ in parsed])
        self.assertEqual((receipt['status'], receipt['comparison'], receipt['public_case_count'],
                          receipt['natural_case_count'], receipt['public_input_file_sha256']),
                         ('clear', 'bidirectional-complete-row-prefix', 48, 16, v21.COMPARISON_INPUT_SHA256))
        self.assertEqual(receipt['natural_heldout_commitment'],
                         '8b475c373034201e5c36186207edbc56f0c8b62221f497d8d09faad1ce182150')
        self.assertEqual(receipt['natural_prediction_freeze_sha256'],
                         '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb')
        self.assertEqual(receipt['comparator_source_sha256'],
                         'd4ad193af9a8f0b19ae5a185f99067ab1c8b8c1671daeda3b0b35321fa0de3a2')

    def test_predecessor_and_planned_candidate_groups(self):
        with mock.patch.object(v20, 'inventory_payload', return_value=self.predecessor):
            new = v21.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v20.INVENTORY_VERSION, 'sha256': v20.INVENTORY_SHA256})
        self.assertEqual(new['records'][:499], self.predecessor['records'])
        self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
        self.assertEqual(new['update'], {'planned_explicit_year_direction_rows': 48})
        added = new['records'][499:]
        self.assertEqual(len(new['records']), 547)
        self.assertEqual([(r['record_id'], r['public_text'], tuple(r['token_ids'])) for r in added],
                         [('planned-' + cid, text, row) for cid, text, row, _ in v21.EXPLICIT_YEAR_DIRECTION_CASES])
        for record, (_, _, _, groups) in zip(added, v21.EXPLICIT_YEAR_DIRECTION_CASES):
            self.assertEqual(record['category'], 'planned')
            self.assertNotIn('target_token_id', record)
            self.assertEqual(record['candidate_token_groups'],
                             {name: list(group) for name, group in zip(v21.CANDIDATE_GROUP_KEYS, groups)})
        old_prefixes, new_prefixes = v21._prefixes(self.predecessor['records']), v21._prefixes(new['records'])
        self.assertLessEqual(old_prefixes, new_prefixes)
        self.assertEqual((len(old_prefixes), len(new_prefixes)), (3620, 3888))
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v21.INVENTORY_SHA256)

    def test_known_input_and_candidate_relations_fail_in_both_directions(self):
        candidate = list(v21.EXPLICIT_YEAR_DIRECTION_ROWS[0])
        answer = v21.EXPLICIT_YEAR_DIRECTION_CASES[0][3][0][0]
        for prior in (candidate[:-1], candidate, candidate + [answer], candidate + [answer, 123]):
            changed = copy.deepcopy(self.predecessor)
            changed['records'][0]['token_ids'] = prior
            digest = hashlib.sha256(canonical(changed)).hexdigest()
            with self.subTest(prior=prior), mock.patch.object(v20, 'inventory_payload', return_value=changed), \
                    mock.patch.object(v20, 'INVENTORY_SHA256', digest), \
                    mock.patch.object(v21, 'PREDECESSOR_SHA256', digest):
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    v21.inventory_payload()
        with mock.patch.object(v20, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v20, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v21.inventory_payload()

    def test_known_split_checks_and_seal_limit(self):
        with mock.patch.object(v20, 'inventory_payload', return_value=self.predecessor):
            payload = v21.inventory_payload()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        kwargs = {'expected_inventory_sha256': v21.INVENTORY_SHA256,
                  'expected_model_identity': v21.TARGET}
        with mock.patch.object(v21, 'inventory_payload', return_value=payload):
            check = v21.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            for row in (list(v21.EXPLICIT_YEAR_DIRECTION_ROWS[0][:-1]),
                        list(v21.EXPLICIT_YEAR_DIRECTION_ROWS[0]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = row
                with self.assertRaises(ValueError):
                    v21.check_known_splits(discovery, altered, cases_per_split=16, **kwargs)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v21.require_preseal_readiness(check)


if __name__ == '__main__':
    unittest.main()
