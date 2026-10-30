"""Check the public follow-up reservation without executing the model."""
from collections import Counter
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v6 as v6
from looking_glass import pretrained_exposures_v7 as v7
from looking_glass import pretrained_exposures_v8 as v8
from looking_glass import pretrained_number_commitments as commitments


ROOT = Path(v8.__file__).resolve().parents[1]
CASE_COMMIT = '939d59540b642c83bc999bcf45220ff2155f41b2'
CASE_PATH = 'experiments/distilgpt2_copy_head_followup_v2.json'
CASE_SHA256 = '3831bfd1a48aaa3fcb67fbdb3cdc10f4422149b1fed90fd5267d3c73525255a8'
PROTOCOL_PATH = 'docs/DISTILGPT2_COPY_HEAD_FOLLOWUP_V2.md'
PROTOCOL_SHA256 = '70616bb01aa7ccb8440da9dc503c274a3d5cb7b24dd3965fd985de6c645ca68c'


def committed_bytes(path):
    return subprocess.run(['git', 'show', CASE_COMMIT + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


class ExposureV9Tests(unittest.TestCase):
    def v9(self):
        try:
            return importlib.import_module('looking_glass.pretrained_exposures_v9')
        except ModuleNotFoundError as exc:
            if exc.name != 'looking_glass.pretrained_exposures_v9':
                raise
            self.fail('the v9 exposure reservation has not been implemented')

    def test_only_32_published_followup_rows_are_added_as_planned(self):
        v9 = self.v9()
        old, new = v8.inventory_payload(), v9.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': 'distilgpt2-development-exposures-v8',
            'sha256': '22b1392ed43568fcac574b55ae6cf0212766771d8439ac39e599f7aa112b298f'})
        self.assertEqual(new['records'][:138], old['records'])
        self.assertEqual(len(new['records']), 170)
        planned = new['records'][138:]
        case_data = json.loads(committed_bytes(CASE_PATH))
        self.assertEqual([tuple(row['token_ids']) for row in planned],
                         [tuple(case['token_ids']) for case in case_data['cases']])
        self.assertEqual([row['target_token_id'] for row in planned],
                         [case['target_id'] for case in case_data['cases']])
        self.assertTrue(all(row['category'] == 'planned' and row['reference_commit'] == CASE_COMMIT
                            and row['references'] == [PROTOCOL_PATH, CASE_PATH] for row in planned))
        self.assertEqual(Counter(row['category'] for row in new['records']),
                         {'observed-saved': 104, 'observed-reported': 3, 'planned': 51,
                          'reference-only': 4, 'synthetic-only': 8})
        self.assertTrue(v8._forbidden_rows() <= v9._forbidden_rows())
        self.assertEqual((len(v9._forbidden_rows()), len(v9._forbidden_rows() - v8._forbidden_rows())),
                         (1314, 352))
        self.assertEqual(hashlib.sha256(v9.canonical_inventory_bytes()).hexdigest(), v9.inventory_digest())
        self.assertEqual(commitments.load_inventory_snapshot(v9.canonical_inventory_bytes(),
                                                              v9.inventory_digest()).sha256,
                         v9.inventory_digest())
        with mock.patch.object(v8, 'INVENTORY_SHA256', '0' * 64), self.assertRaises(ValueError):
            v9.inventory_payload()

    def test_case_commit_and_exported_rows_match_the_frozen_protocol(self):
        v9 = self.v9()
        case_raw, protocol_raw = committed_bytes(CASE_PATH), committed_bytes(PROTOCOL_PATH)
        self.assertEqual(hashlib.sha256(case_raw).hexdigest(), CASE_SHA256)
        self.assertEqual(hashlib.sha256(protocol_raw).hexdigest(), PROTOCOL_SHA256)
        data = json.loads(case_raw)
        self.assertEqual((data['model_id'], data['model_revision']),
                         (v8.TARGET.model_id, v8.TARGET.model_revision))
        self.assertEqual((data['known_exposure_current_version'], data['known_exposure_current_sha256']),
                         (v8.INVENTORY_VERSION, v8.INVENTORY_SHA256))
        self.assertEqual(data['known_exposure_baseline_v7_sha256'], v7.INVENTORY_SHA256)
        self.assertEqual(data['protocol_sha256'], PROTOCOL_SHA256)
        self.assertEqual([tuple(group['token_ids']) for group in data['groups']], list(v9.FOLLOWUP_GROUPS))
        self.assertEqual([tuple(case['token_ids']) for case in data['cases']], list(v9.FOLLOWUP_ROWS))
        self.assertEqual([(c['group'], c['variant'], c['target_id']) for c in data['cases']],
                         [(i // 4 + 1, i % 4, v9.FOLLOWUP_GROUPS[i // 4][5]) for i in range(32)])
        flat = [token for group in v9.FOLLOWUP_GROUPS for token in group]
        previous = {token for group in v6.PUBLIC_COPY_GROUPS + v7.PUBLIC_EVALUATION_GROUPS for token in group}
        self.assertEqual(len(flat), 48)
        self.assertEqual(len(set(flat)), 48)
        self.assertFalse(set(flat) & previous)

    def test_predecessor_rows_with_prefix_collision_in_either_direction_are_rejected(self):
        v9 = self.v9()
        candidate = list(v9.FOLLOWUP_ROWS[0])
        for old_row in (candidate[:5], candidate + [123]):
            old = copy.deepcopy(v8.inventory_payload())
            old['records'][0]['token_ids'] = old_row
            # Pin this deliberately synthetic payload so the prefix guard is exercised.
            synthetic_digest = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
            with self.subTest(old_row=old_row), mock.patch.object(v8, 'inventory_payload', return_value=old), \
                    mock.patch.object(v8, "INVENTORY_SHA256", synthetic_digest), \
                    mock.patch.object(v9, "PREDECESSOR_SHA256", synthetic_digest):
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    v9.inventory_payload()

    def test_known_prefix_checks_reject_followup_rows_without_authorizing_a_seal(self):
        v9 = self.v9()
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {'expected_inventory_sha256': v9.inventory_digest(), 'expected_model_identity': v9.TARGET}
        check = v9.check_known_splits(discovery, heldout, cases_per_split=16, **options)
        self.assertEqual((check.inventory_version, check.candidate_count, check.forbidden_row_count),
                         ('distilgpt2-development-exposures-v9', 32, 1314))
        self.assertTrue(set(v9.FOLLOWUP_ROWS) <= v9._forbidden_rows())
        heldout[0] = list(v9.FOLLOWUP_ROWS[0])
        with self.assertRaises(ValueError):
            v9.check_exposed_discovery(discovery, heldout, **options)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v9.require_preseal_readiness(check)
        self.assertFalse(v9.prepare_preseal_disclosure(
            check, protocol_sha256='a' * 64, acknowledge_prior_exposure_uncertainty=True).execution_authorized)


if __name__ == '__main__':
    unittest.main()
