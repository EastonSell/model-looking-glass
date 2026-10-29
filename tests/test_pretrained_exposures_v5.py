"""Public reservations and synthetic split probes; no model or private cases."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v4 as v4
from looking_glass import pretrained_exposures_v5 as v5
from looking_glass import pretrained_number_commitments as commitments


EXPECTED_V4 = '6c065926d08cf08aa98d2c2725299d6bdc25e919246851c34bd9c7d63e8a4af1'
EXPECTED_V5 = '1c597a028952a40ac48614c0e263a87a034d4f6a3df2b4cbffcdb6047adf6586'


class ExposureV5Tests(unittest.TestCase):
    def test_previous_records_and_four_planned_rows_are_exact(self):
        self.assertEqual(v4.inventory_digest(), EXPECTED_V4)
        self.assertEqual(v5.inventory_digest(), EXPECTED_V5)
        self.assertEqual(hashlib.sha256(v5.canonical_inventory_bytes()).hexdigest(), EXPECTED_V5)
        old, new = v4.inventory_payload(), json.loads(v5.canonical_inventory_bytes())
        self.assertEqual(new['predecessor'], {'version': v4.INVENTORY_VERSION, 'sha256': EXPECTED_V4})
        self.assertEqual(new['records'][:len(old['records'])], old['records'])
        added = new['records'][len(old['records']):]
        self.assertEqual([row['token_ids'] for row in added],
                         [[50256, 107, 207, 307], [50256, 108, 208, 308],
                          [50256, 109, 209, 309], [50256, 110, 210, 310]])
        self.assertTrue(all(row['category'] == 'planned' and row['references'] == [v5.PLAN_PATH]
                            and row['reference_commit'] == v5.PLAN_COMMIT for row in added))
        self.assertEqual(Counter(row['category'] for row in new['records']),
                         {'observed-saved': 40, 'observed-reported': 3, 'planned': 19,
                          'reference-only': 4, 'synthetic-only': 8})
        self.assertTrue(v4._forbidden_rows() <= v5._forbidden_rows())
        self.assertEqual((len(v4._forbidden_rows()), len(v5._forbidden_rows())), (246, 258))
        self.assertEqual(len(v5._forbidden_rows() - v4._forbidden_rows()), 12)
        compared = 0
        for _, edit_row in v4.PUBLIC_EDIT_FOLLOWON:
            for broker_row in v5.PUBLIC_BROKER_CHECK:
                compared += 1
                self.assertFalse(edit_row[:len(broker_row)] == broker_row
                                 or broker_row[:len(edit_row)] == edit_row)
        self.assertEqual(compared, 52)
        with mock.patch.object(v4, 'INVENTORY_SHA256', '0' * 64), self.assertRaises(ValueError):
            v5.inventory_digest()
        prior = next(tuple(row['token_ids']) for row in old['records'] if len(row['token_ids']) < 32)
        altered = list(v5.PUBLIC_BROKER_CHECK)
        altered[0] = prior + (1,)
        with mock.patch.object(v5, 'PUBLIC_BROKER_CHECK', tuple(altered)), self.assertRaises(ValueError):
            v5.inventory_payload()

    def test_each_new_prefix_is_excluded_in_every_synthetic_holdout_position(self):
        discovery = [list(ids) for _, ids in v5.PUBLIC_DISCOVERY]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {'expected_inventory_sha256': EXPECTED_V5, 'expected_model_identity': v5.TARGET}
        report = v5.check_exposed_discovery(discovery, heldout, **options)
        self.assertEqual((report.forbidden_row_count, report.known_or_reserved_discovery_overlap_count), (258, 16))
        for prefix in v5._forbidden_rows() - v4._forbidden_rows():
            for index in range(16):
                sample = [row[:] for row in heldout]
                sample[index] = list(prefix)
                with self.subTest(index=index, prefix=prefix), self.assertRaises(ValueError):
                    v5.check_exposed_discovery(discovery, sample, **options)
        with self.assertRaises(ValueError):
            v5.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V4,
                                       expected_model_identity=v5.TARGET)
        with self.assertRaises(ValueError):
            v5.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V5,
                                       expected_model_identity=replace(v5.TARGET, weights_sha256='0' * 64))

    def test_disclosure_is_not_authorization_and_sampler_accepts_snapshot(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        check = v5.check_known_splits(discovery, heldout, expected_inventory_sha256=EXPECTED_V5,
                                      expected_model_identity=v5.TARGET, cases_per_split=16)
        self.assertEqual((check.candidate_count, check.forbidden_row_count), (32, 258))
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v5.require_preseal_readiness(check)
        disclosure = v5.prepare_preseal_disclosure(
            check, protocol_sha256='a' * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertFalse(disclosure.execution_authorized)
        self.assertFalse(disclosure.sealed_manifest_verified)
        snapshot = commitments.load_inventory_snapshot(v5.canonical_inventory_bytes(), EXPECTED_V5)
        self.assertEqual(snapshot.sha256, EXPECTED_V5)


if __name__ == '__main__':
    unittest.main()
