"""Synthetic candidates only; no hidden study inputs or model forwards."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v2 as v2
from looking_glass import pretrained_exposures_v3 as v3


EXPECTED_V2 = "e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1"
EXPECTED_V3 = "76baf0c8996bf2872a3a1b8fd0fe2734e15525ce0f7014e54811145cb98ce912"


class ExposureV3Tests(unittest.TestCase):
    def test_observation_status_and_predecessor_preserve_all_rows(self):
        self.assertEqual(v2.inventory_digest(), EXPECTED_V2)
        self.assertEqual(v3.inventory_digest(), EXPECTED_V3)
        self.assertEqual(hashlib.sha256(v3.canonical_inventory_bytes()).hexdigest(), EXPECTED_V3)
        old, new = v2.inventory_payload(), json.loads(v3.canonical_inventory_bytes())
        self.assertEqual(new["predecessor"], {"version": v2.INVENTORY_VERSION, "sha256": EXPECTED_V2})
        self.assertEqual([r["token_ids"] for r in old["records"]], [r["token_ids"] for r in new["records"]])
        self.assertEqual(v2._forbidden_rows(), v3._forbidden_rows())
        self.assertEqual(len(v3._forbidden_rows()), 204)
        self.assertEqual(Counter(r["category"] for r in new["records"]),
                         {"observed-saved": 40, "observed-reported": 3, "planned": 2,
                          "reference-only": 4, "synthetic-only": 8})
        changed = [(a, b) for a, b in zip(old["records"], new["records"]) if a != b]
        self.assertEqual(len(changed), 32)
        agreement = [b for a, b in changed if a["record_id"].startswith("planned-agreement-discovery-")]
        edit = [b for a, b in changed if a["record_id"].startswith("planned-parameter-edit-")]
        self.assertEqual((len(agreement), len(edit)), (16, 13))
        self.assertTrue(all(r["category"] == "observed-saved" and
                            r["reference_commit"] == v3.AGREEMENT_EVIDENCE_COMMIT for r in agreement))
        self.assertTrue(all(r["category"] == "observed-saved" and
                            r["reference_commit"] == v3.EDIT_EVIDENCE_COMMIT for r in edit))
        self.assertTrue(all(v3.EDIT_TERMINAL_SHA256 in r["provenance"] for r in edit))
        self.assertEqual(v2.inventory_digest(), EXPECTED_V2)
        with mock.patch.object(v2, "INVENTORY_SHA256", "0" * 64), self.assertRaises(ValueError):
            v3.inventory_digest()

    def test_checks_cover_public_discovery_and_all_synthetic_heldout_positions(self):
        discovery = [list(ids) for _, ids in v3.PUBLIC_DISCOVERY]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {"expected_inventory_sha256": EXPECTED_V3, "expected_model_identity": v3.TARGET}
        result = v3.check_exposed_discovery(discovery, heldout, **options)
        self.assertEqual(result.known_or_reserved_discovery_overlap_count, 16)
        self.assertEqual(result.forbidden_row_count, 204)
        self.assertFalse(result.preseal_ready)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v3.require_preseal_readiness(result)
        disclosed = v3.prepare_preseal_disclosure(
            result, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertEqual(disclosed.discovery_policy, "known-prefix-discovery-reuse-disclosed")
        self.assertEqual(disclosed.known_or_reserved_discovery_overlap_count, 16)
        self.assertFalse(disclosed.execution_authorized)
        for index in range(16):
            sample = [row[:] for row in heldout]
            sample[index] = list(v3.PUBLIC_DISCOVERY[index][1])
            with self.subTest(index=index), self.assertRaises(ValueError):
                v3.check_exposed_discovery(discovery, sample, **options)
        for left, right in (([7000, 7001], [7000, 7001, 7002]),
                            ([7000, 7001, 7002], [7000, 7001])):
            sample_discovery = [row[:] for row in discovery]
            sample_heldout = [row[:] for row in heldout]
            sample_discovery[0], sample_heldout[0] = left, right
            with self.assertRaisesRegex(ValueError, "prefix relationship"):
                v3.check_exposed_discovery(sample_discovery, sample_heldout, **options)
        with self.assertRaises(ValueError):
            v3.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V2,
                                       expected_model_identity=v3.TARGET)
        with self.assertRaises(ValueError):
            v3.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V3,
                                       expected_model_identity=replace(v3.TARGET, weights_sha256="0" * 64))

    def test_strict_fresh_splits_and_disclosure_schema(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {"expected_inventory_sha256": EXPECTED_V3, "expected_model_identity": v3.TARGET}
        check = v3.check_known_splits(discovery, heldout, cases_per_split=16, **options)
        self.assertEqual((check.candidate_count, check.forbidden_row_count), (32, 204))
        disclosure = v3.prepare_preseal_disclosure(
            check, protocol_sha256="b" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertEqual(disclosure.discovery_policy, "known-prefixes-excluded-from-both")
        for invalid in (True, 0, 4, 17):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                v3.check_known_splits(discovery, heldout, cases_per_split=invalid, **options)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v3.prepare_preseal_disclosure(check, protocol_sha256="b" * 64)
        with self.assertRaises(ValueError):
            v3.prepare_preseal_disclosure(replace(check, candidate_count=32.0),
                                          protocol_sha256="b" * 64, acknowledge_prior_exposure_uncertainty=True)


if __name__ == "__main__":
    unittest.main()
