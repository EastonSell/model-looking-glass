"""Verify the fixed agreement-direction reservation without model work."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v19 as v19
from looking_glass import pretrained_exposures_v20 as v20

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def git_bytes(commit, path):
    return subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


class ExposureV20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v19.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v19.INVENTORY_SHA256:
            raise AssertionError('predecessor payload differs from its pin')

    def test_public_cases_roles_answers_and_private_receipt_bind_exact_rows(self):
        raw = [git_bytes(v20.CASES_COMMIT, path) for path in
               (v20.PROTOCOL_PATH, v20.CASE_PATH, v20.COMPARISON_INPUT_PATH, v20.RECEIPT_PATH)]
        self.assertEqual([hashlib.sha256(b).hexdigest() for b in raw],
                         [v20.PROTOCOL_SHA256, v20.CASE_SHA256,
                          v20.COMPARISON_INPUT_SHA256, v20.RECEIPT_SHA256])
        subprocess.run(['git', 'merge-base', '--is-ancestor', v20.PROTOCOL_COMMIT,
                        v20.CASES_COMMIT], cwd=ROOT, check=True)
        data, adapted, receipt = map(json.loads, raw[1:])
        self.assertEqual((data['model_id'], data['model_revision']), (v20.TARGET.model_id, v20.TARGET.model_revision))
        self.assertEqual((data['protocol_revision'], data['protocol_sha256']),
                         (v20.PROTOCOL_COMMIT, v20.PROTOCOL_SHA256))
        self.assertEqual((data['known_inventory_version'], data['known_inventory_sha256']),
                         ('distilgpt2-development-exposures-v18', v19.PREDECESSOR_SHA256))
        self.assertEqual(data['scope'], 'Public nonblind proposed inputs; tokenizer only, no model forward.')
        cases = [(c['case_id'], c['role'], c['text'], tuple(c['token_ids'])) for c in data['cases']]
        self.assertEqual(cases, list(v20.AGREEMENT_CASES))
        self.assertEqual(tuple(row for _, _, _, row in cases), v20.AGREEMENT_ROWS)
        self.assertEqual(Counter(role for _, role, _, _ in cases), {'target': 32, 'context': 8})
        self.assertEqual(tuple(data['answer_ids']), v20.AGREEMENT_ANSWER_IDS)
        self.assertEqual((len(cases), len(set(v20.AGREEMENT_ROWS))), (40, 40))
        self.assertEqual([case['input_ids'] for case in adapted['cases']], [list(row) for _, _, _, row in cases])
        self.assertEqual((receipt['status'], receipt['comparison'], receipt['public_case_count'],
                          receipt['natural_case_count'], receipt['public_input_file_sha256']),
                         ('clear', 'bidirectional-complete-row-prefix', 40, 16, v20.COMPARISON_INPUT_SHA256))
        self.assertEqual(receipt['natural_heldout_commitment'],
                         '8b475c373034201e5c36186207edbc56f0c8b62221f497d8d09faad1ce182150')
        self.assertEqual(receipt['natural_prediction_freeze_sha256'],
                         '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb')
        self.assertEqual(receipt['comparator_source_sha256'],
                         'd4ad193af9a8f0b19ae5a185f99067ab1c8b8c1671daeda3b0b35321fa0de3a2')
        ordered = json.dumps([list(row) for _, _, _, row in cases], separators=(',', ':')).encode('ascii')
        self.assertEqual(hashlib.sha256(ordered).hexdigest(), v20.ORDERED_ROWS_SHA256)

    def test_predecessor_preservation_and_planned_role_answer_records(self):
        with mock.patch.object(v19, 'inventory_payload', return_value=self.predecessor):
            new = v20.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v19.INVENTORY_VERSION, 'sha256': v19.INVENTORY_SHA256})
        self.assertEqual(new['records'][:459], self.predecessor['records'])
        self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
        self.assertEqual(new['update'], {'planned_agreement_direction_rows': 40})
        added = new['records'][459:]
        self.assertEqual(len(new['records']), 499)
        self.assertEqual([(r['record_id'], r['case_role'], r['public_text'], tuple(r['token_ids'])) for r in added],
                         [('planned-' + cid, role, text, row) for cid, role, text, row in v20.AGREEMENT_CASES])
        self.assertTrue(all(r['category'] == 'planned' and r['candidate_token_ids'] == list(v20.AGREEMENT_ANSWER_IDS)
                            and 'target_token_id' not in r and r['reference_commit'] == v20.CASES_COMMIT
                            and r['references'] == [v20.PROTOCOL_PATH, v20.CASE_PATH,
                                                    v20.COMPARISON_INPUT_PATH, v20.RECEIPT_PATH] for r in added))
        old_prefixes, new_prefixes = v20._prefixes(self.predecessor['records']), v20._prefixes(new['records'])
        self.assertLessEqual(old_prefixes, new_prefixes)
        self.assertEqual((len(old_prefixes), len(new_prefixes)), (3522, 3620))
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v20.INVENTORY_SHA256)

    def test_rejects_known_relations_in_both_prefix_directions(self):
        candidate = list(v20.AGREEMENT_ROWS[0])
        for prior in (candidate[:-1], candidate, candidate + [v20.AGREEMENT_ANSWER_IDS[0]],
                      candidate + [v20.AGREEMENT_ANSWER_IDS[0], 123]):
            changed = copy.deepcopy(self.predecessor)
            changed['records'][0]['token_ids'] = prior
            changed_digest = hashlib.sha256(canonical(changed)).hexdigest()
            with self.subTest(length=len(prior)), \
                    mock.patch.object(v19, 'inventory_payload', return_value=changed), \
                    mock.patch.object(v19, 'INVENTORY_SHA256', changed_digest), \
                    mock.patch.object(v20, 'PREDECESSOR_SHA256', changed_digest):
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    v20.inventory_payload()
        with mock.patch.object(v19, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v19, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v20.inventory_payload()

    def test_known_split_checks_exclude_new_rows_without_authorizing_a_seal(self):
        with mock.patch.object(v19, 'inventory_payload', return_value=self.predecessor):
            payload = v20.inventory_payload()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        kwargs = {'expected_inventory_sha256': v20.INVENTORY_SHA256,
                  'expected_model_identity': v20.TARGET}
        with mock.patch.object(v20, 'inventory_payload', return_value=payload):
            check = v20.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            for candidate in (list(v20.AGREEMENT_ROWS[0][:-1]), list(v20.AGREEMENT_ROWS[0]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = candidate
                with self.assertRaises(ValueError):
                    v20.check_known_splits(discovery, altered, cases_per_split=16, **kwargs)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v20.require_preseal_readiness(check)


if __name__ == '__main__':
    unittest.main()
