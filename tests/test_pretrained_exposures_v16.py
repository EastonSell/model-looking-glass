"""Check the public year-transfer reservation without model work."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v15 as v15
from looking_glass import pretrained_exposures_v16 as v16


ROOT = Path(v1.__file__).resolve().parents[1]


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


class ExposureV16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the immutable predecessor once; tests patch copies for adversarial cases.
        cls.predecessor = v15.inventory_payload()
        if digest(cls.predecessor) != v15.INVENTORY_SHA256:
            raise AssertionError('the real predecessor payload differs from its pin')

    def test_public_cases_bind_exact_planned_rows_and_protocol(self):
        raw = subprocess.run(['git', 'show', v16.CASES_COMMIT + ':' + v16.CASE_PATH],
                             cwd=ROOT, check=True, capture_output=True).stdout
        protocol = subprocess.run(['git', 'show', v16.PROTOCOL_COMMIT + ':' + v16.PROTOCOL_PATH],
                                  cwd=ROOT, check=True, capture_output=True).stdout
        self.assertEqual(hashlib.sha256(raw).hexdigest(), v16.CASE_SHA256)
        self.assertEqual(hashlib.sha256(protocol).hexdigest(), v16.PROTOCOL_SHA256)
        subprocess.run(['git', 'merge-base', '--is-ancestor', v16.PROTOCOL_COMMIT,
                        v16.CASES_COMMIT], cwd=ROOT, check=True)
        data = json.loads(raw)
        self.assertEqual((data['known_exposure_version'], data['known_exposure_sha256']),
                         (v15.INVENTORY_VERSION, v15.INVENTORY_SHA256))
        self.assertEqual((data['protocol_revision'], data['protocol_sha256']),
                         (v16.PROTOCOL_COMMIT, v16.PROTOCOL_SHA256))
        self.assertEqual((data['model_id'], data['model_revision']),
                         (v16.TARGET.model_id, v16.TARGET.model_revision))
        cases = [(case['case_id'], case['text'], tuple(case['input_ids'])) for case in data['cases']]
        self.assertEqual(cases, list(v16.YEAR_TRANSFER_CASES))
        self.assertEqual(tuple(item[2] for item in cases), v16.YEAR_TRANSFER_ROWS)
        self.assertEqual(len(cases), 48)
        self.assertEqual(len(set(v16.YEAR_TRANSFER_ROWS)), 48)

    def test_predecessor_preservation_prefix_union_and_copy_safety(self):
        with mock.patch.object(v15, 'inventory_payload', return_value=self.predecessor):
            new = v16.inventory_payload()
            self.assertEqual(new['predecessor'], {'version': v15.INVENTORY_VERSION,
                                                   'sha256': v15.INVENTORY_SHA256})
            self.assertEqual(new['records'][:359], self.predecessor['records'])
            self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
            self.assertEqual(new['update'], {'planned_public_year_transfer_rows': 48})
            added = new['records'][359:]
            self.assertEqual(len(new['records']), 407)
            self.assertEqual([(r['record_id'], r['public_text'], tuple(r['token_ids'])) for r in added],
                             [('planned-' + case_id, text, ids)
                              for case_id, text, ids in v16.YEAR_TRANSFER_CASES])
            self.assertTrue(all(r['category'] == 'planned' and r['reference_commit'] == v16.CASES_COMMIT
                                and r['references'] == [v16.PROTOCOL_PATH, v16.CASE_PATH] for r in added))
            old_prefixes = {tuple(row['token_ids'][:n]) for row in self.predecessor['records']
                            for n in range(1, len(row['token_ids']) + 1)}
            new_prefixes = {tuple(row['token_ids'][:n]) for row in new['records']
                            for n in range(1, len(row['token_ids']) + 1)}
            self.assertEqual((len(old_prefixes), len(new_prefixes)), (2959, 3247))
            self.assertLessEqual(old_prefixes, new_prefixes)
            self.assertEqual(v16.FORBIDDEN_PREFIX_COUNT, 3247)
            self.assertEqual(digest(new), v16.INVENTORY_SHA256)
            new['records'][0]['token_ids'][0] = -1
            self.assertNotEqual(new['records'][0], self.predecessor['records'][0])
            self.assertEqual(v16.inventory_payload()['records'][0], self.predecessor['records'][0])

    def test_both_known_prefix_directions_and_inventory_drift_fail_closed(self):
        candidate = list(v16.YEAR_TRANSFER_ROWS[0])
        for prior in (candidate[:-1], candidate, candidate + [123]):
            changed = copy.deepcopy(self.predecessor)
            changed['records'][0]['token_ids'] = prior
            changed_digest = digest(changed)
            with self.subTest(prior_length=len(prior)), \
                    mock.patch.object(v15, 'inventory_payload', return_value=changed), \
                    mock.patch.object(v15, 'INVENTORY_SHA256', changed_digest), \
                    mock.patch.object(v16, 'PREDECESSOR_SHA256', changed_digest):
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    v16.inventory_payload()
        with mock.patch.object(v15, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v15, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v16.inventory_payload()
        cases = list(v16.YEAR_TRANSFER_CASES)
        case_id, text, _ = cases[1]
        cases[1] = (case_id, text, v16.YEAR_TRANSFER_ROWS[0] + (123,))
        with mock.patch.object(v15, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v16, 'YEAR_TRANSFER_CASES', tuple(cases)):
            with self.assertRaisesRegex(ValueError, 'prefix'):
                v16.inventory_payload()

    def test_known_split_checks_exclude_new_rows_without_authorizing_seal(self):
        with mock.patch.object(v15, 'inventory_payload', return_value=self.predecessor):
            payload = v16.inventory_payload()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        kwargs = {'expected_inventory_sha256': v16.INVENTORY_SHA256,
                  'expected_model_identity': v16.TARGET}
        with mock.patch.object(v16, 'inventory_payload', return_value=payload):
            check = v16.check_known_splits(discovery, heldout, cases_per_split=16, **kwargs)
            for candidate in (list(v16.YEAR_TRANSFER_ROWS[0][:-1]),
                              list(v16.YEAR_TRANSFER_ROWS[0]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = candidate
                with self.subTest(candidate_length=len(candidate)):
                    with self.assertRaises(ValueError):
                        v16.check_known_splits(discovery, altered, cases_per_split=16, **kwargs)
                    with self.assertRaises(ValueError):
                        v16.check_exposed_discovery(discovery, altered, **kwargs)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v16.require_preseal_readiness(check)
            disclosure = v16.prepare_preseal_disclosure(
                check, protocol_sha256='a' * 64, acknowledge_prior_exposure_uncertainty=True)
            self.assertFalse(disclosure.execution_authorized)


if __name__ == '__main__':
    unittest.main()
