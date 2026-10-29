"""Synthetic split probes only; no hidden cases or checkpoint forwards."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v3 as v3
from looking_glass import pretrained_exposures_v4 as v4


EXPECTED_V3 = "76baf0c8996bf2872a3a1b8fd0fe2734e15525ce0f7014e54811145cb98ce912"
EXPECTED_V4 = "6c065926d08cf08aa98d2c2725299d6bdc25e919246851c34bd9c7d63e8a4af1"


class ExposureV4Tests(unittest.TestCase):
    def test_immutable_predecessor_and_13_new_planned_rows(self):
        self.assertEqual(v3.inventory_digest(), EXPECTED_V3)
        self.assertEqual(v4.inventory_digest(), EXPECTED_V4)
        self.assertEqual(hashlib.sha256(v4.canonical_inventory_bytes()).hexdigest(), EXPECTED_V4)
        old, new = v3.inventory_payload(), json.loads(v4.canonical_inventory_bytes())
        self.assertEqual(new["predecessor"], {"version": v3.INVENTORY_VERSION, "sha256": EXPECTED_V3})
        self.assertEqual(new["records"][:len(old["records"])], old["records"])
        added = new["records"][len(old["records"]):]
        self.assertEqual(len(added), 13)
        self.assertTrue(all(row["category"] == "planned" for row in added))
        self.assertEqual([row["token_ids"] for row in added], [list(ids) for _, ids in v4.PUBLIC_EDIT_FOLLOWON])
        self.assertEqual(Counter(r["category"] for r in new["records"]),
                         {"observed-saved": 40, "observed-reported": 3, "planned": 15,
                          "reference-only": 4, "synthetic-only": 8})
        self.assertTrue(v3._forbidden_rows() <= v4._forbidden_rows())
        self.assertEqual((len(v3._forbidden_rows()), len(v4._forbidden_rows())), (204, 246))
        self.assertEqual(len(v4._forbidden_rows() - v3._forbidden_rows()), 42)
        with mock.patch.object(v3, "INVENTORY_SHA256", "0" * 64), self.assertRaises(ValueError):
            v4.inventory_digest()
        prior = tuple(v3.inventory_payload()["records"][0]["token_ids"])
        altered = list(v4.PUBLIC_EDIT_FOLLOWON)
        altered[0] = ("target", prior + (1,))
        self.assertNotIn(prior + (1,), v3._forbidden_rows())
        with mock.patch.object(v4, "PUBLIC_EDIT_FOLLOWON", tuple(altered)), self.assertRaisesRegex(ValueError, "extends"):
            v4.inventory_payload()

    def test_new_prefixes_are_excluded_from_each_synthetic_holdout_slot(self):
        discovery = [list(ids) for _, ids in v4.PUBLIC_DISCOVERY]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {"expected_inventory_sha256": EXPECTED_V4, "expected_model_identity": v4.TARGET}
        report = v4.check_exposed_discovery(discovery, heldout, **options)
        self.assertEqual((report.forbidden_row_count, report.known_or_reserved_discovery_overlap_count), (246, 16))
        for prefix in v4._forbidden_rows() - v3._forbidden_rows():
            for index in range(16):
                sample = [r[:] for r in heldout]
                sample[index] = list(prefix)
                with self.subTest(index=index, prefix=prefix), self.assertRaises(ValueError):
                    v4.check_exposed_discovery(discovery, sample, **options)
        with self.assertRaises(ValueError):
            v4.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V3,
                                       expected_model_identity=v4.TARGET)
        with self.assertRaises(ValueError):
            v4.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V4,
                                       expected_model_identity=replace(v4.TARGET, weights_sha256="0" * 64))

    def test_fresh_split_disclosure_is_not_authorization(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        check = v4.check_known_splits(discovery, heldout, expected_inventory_sha256=EXPECTED_V4,
                                       expected_model_identity=v4.TARGET, cases_per_split=16)
        self.assertEqual((check.candidate_count, check.forbidden_row_count), (32, 246))
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v4.require_preseal_readiness(check)
        disclosure = v4.prepare_preseal_disclosure(
            check, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertFalse(disclosure.execution_authorized)
        self.assertFalse(disclosure.sealed_manifest_verified)
        with self.assertRaises(ValueError):
            v4.prepare_preseal_disclosure(replace(check, candidate_count=32.0),
                                          protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)


if __name__ == "__main__":
    unittest.main()
