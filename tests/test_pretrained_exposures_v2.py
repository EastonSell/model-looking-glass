"""Synthetic split checks for an immutable inventory successor; no real split."""
from dataclasses import replace
import hashlib
import json
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v2 as v2


EXPECTED_V1 = "6f21bffeacbe2323c21e53378161443eae8ba28b7210f30e70fd353497e87c8f"
EXPECTED_V2 = "e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1"


class ExposureInventoryV2Tests(unittest.TestCase):
    @staticmethod
    def candidates():
        return ([[1000 + i, 2000 + i] for i in range(8)],
                [[3000 + i, 4000 + i] for i in range(8)])

    def check(self, discovery, heldout, *, digest=EXPECTED_V2, identity=v1.TARGET, cases_per_split=8):
        return v2.check_known_splits(discovery, heldout, expected_inventory_sha256=digest,
                                     expected_model_identity=identity, cases_per_split=cases_per_split)

    def test_v2_binds_unchanged_v1_and_one_status_update(self):
        self.assertEqual(v1.inventory_digest(), EXPECTED_V1)
        self.assertEqual(hashlib.sha256(v2.canonical_inventory_bytes()).hexdigest(), EXPECTED_V2)
        self.assertEqual(v2.inventory_digest(), EXPECTED_V2)
        prior = v1.inventory_payload()
        current = json.loads(v2.canonical_inventory_bytes())
        self.assertEqual(current["predecessor"], {"version": v1.INVENTORY_VERSION, "sha256": EXPECTED_V1})
        for key in prior.keys() - {"version", "records"}:
            self.assertEqual(current[key], prior[key])
        old = prior["records"]
        new = current["records"]
        self.assertEqual(len(new), len(old) + 16 + 2 + 13)
        changed = [(a, b) for a, b in zip(old, new[:len(old)]) if a != b]
        self.assertEqual(len(changed), 1)
        before, after = changed[0]
        self.assertEqual(before["record_id"], "planned-intervention-smoke")
        self.assertEqual(before["category"], "planned")
        self.assertEqual(after["record_id"], "observed-intervention-smoke")
        self.assertEqual(after["category"], "observed-saved")
        self.assertEqual(after["token_ids"], before["token_ids"])
        self.assertEqual(after["reference_commit"], v2.SMOKE_PUBLICATION_COMMIT)
        self.assertIn(v2.SMOKE_ATTEMPT_SHA256, after["provenance"])
        self.assertIn(v2.SMOKE_RESULT_SHA256, after["provenance"])
        self.assertIn(v2.REPLAY_ATTEMPT_SHA256, after["provenance"])
        self.assertIn(v2.REPLAY_RESULT_SHA256, after["provenance"])
        self.assertEqual(current["update"]["reserved_public_discovery_rows"], 16)
        self.assertEqual(current["update"]["superseded_public_reference_rows"], 2)
        self.assertEqual(current["update"]["reserved_public_parameter_edit_rows"], 13)
        self.assertEqual([row["token_ids"] for row in new[len(old):len(old) + 16]],
                         [list(ids) for _, ids in v2.PUBLIC_DISCOVERY])
        self.assertTrue(all(row["category"] == "planned" for row in new[len(old):len(old) + 16]))
        self.assertEqual([row["token_ids"] for row in new[len(old) + 16:len(old) + 18]],
                         [list(ids) for _, ids in v2.SUPERSEDED_DISCOVERY])
        self.assertTrue(all(row["category"] == "reference-only" for row in new[len(old) + 16:len(old) + 18]))
        self.assertEqual([row["token_ids"] for row in new[len(old) + 18:]],
                         [list(ids) for _, ids in v2.PUBLIC_PARAMETER_EDIT])
        self.assertTrue(all(row["category"] == "planned" for row in new[len(old) + 18:]))
        self.assertEqual(v1.inventory_digest(), EXPECTED_V1)

    def test_all_earlier_forbidden_prefixes_remain_excluded(self):
        prefixes = {r.token_ids[:length] for r in v1.RECORDS
                    for length in range(1, len(r.token_ids) + 1)}
        self.assertEqual(len(prefixes), 82)
        for prefix in prefixes:
            for position in range(16):
                discovery, heldout = self.candidates()
                (discovery + heldout)[position][:] = prefix
                with self.subTest(prefix=prefix, position=position):
                    with self.assertRaises(ValueError) as caught:
                        self.check(discovery, heldout)
                    self.assertNotIn(str(prefix), str(caught.exception))

    def test_version_identity_and_readiness_fail_closed(self):
        discovery, heldout = self.candidates()
        result = self.check(discovery, heldout)
        self.assertEqual(result.inventory_version, v2.INVENTORY_VERSION)
        self.assertEqual(result.inventory_sha256, EXPECTED_V2)
        self.assertEqual(result.forbidden_row_count, 204)
        self.assertFalse(result.complete_exposure_history_certified)
        self.assertFalse(result.preseal_ready)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v2.require_preseal_readiness(result)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v2.prepare_preseal_disclosure(result, protocol_sha256="a" * 64)
        disclosure = v2.prepare_preseal_disclosure(
            result, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertEqual(disclosure.inventory_sha256, EXPECTED_V2)
        self.assertEqual(disclosure.candidates_per_split, 8)
        self.assertEqual(disclosure.known_prefix_comparison, "trusted-caller-report")
        self.assertFalse(disclosure.execution_authorized)
        self.assertFalse(disclosure.sealed_manifest_verified)
        self.assertEqual(disclosure.discovery_policy, "known-prefixes-excluded-from-both")
        self.assertEqual(disclosure.known_or_reserved_discovery_overlap_count, 0)
        for forged in (replace(result, candidate_count=16.0),
                       replace(result, forbidden_row_count=204.0)):
            with self.subTest(forged=forged):
                with self.assertRaises(ValueError):
                    v2.prepare_preseal_disclosure(
                        forged, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
        with self.assertRaises(ValueError):
            self.check(discovery, heldout, digest=EXPECTED_V1)
        with self.assertRaises(ValueError):
            self.check(discovery, heldout, identity=replace(v1.TARGET, weights_sha256="0" * 64))
        prior_check = v1.check_known_splits(discovery, heldout, expected_inventory_sha256=EXPECTED_V1,
                                             expected_model_identity=v1.TARGET)
        with self.assertRaises(ValueError):
            v2.require_preseal_readiness(prior_check)
        with mock.patch.object(v1, "RECORDS", v1.RECORDS[:-1]):
            with self.assertRaises(ValueError):
                v2.inventory_digest()

    def test_explicit_sixteen_case_splits_check_all_rows_and_duplicates(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        result = self.check(discovery, heldout, cases_per_split=16)
        self.assertEqual(result.candidate_count, 32)
        disclosure = v2.prepare_preseal_disclosure(
            result, protocol_sha256="b" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertEqual(disclosure.candidates_per_split, 16)
        self.assertFalse(disclosure.independent_review_verified)
        for position in range(32):
            with self.subTest(position=position):
                discovery = [[1000 + i, 2000 + i] for i in range(16)]
                heldout = [[3000 + i, 4000 + i] for i in range(16)]
                (discovery + heldout)[position][:] = [50256, 50256]
                with self.assertRaises(ValueError):
                    self.check(discovery, heldout, cases_per_split=16)
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        heldout[12] = list(discovery[1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(discovery, heldout, cases_per_split=16)
        heldout[12] = discovery[1] + [999]
        with self.assertRaisesRegex(ValueError, "prefix relationship"):
            self.check(discovery, heldout, cases_per_split=16)
        with self.assertRaises(ValueError):
            self.check(discovery[:8], heldout[:8], cases_per_split=16)
        for invalid in (True, 0, 4, 9, 17, 32):
            with self.subTest(size=invalid):
                with self.assertRaises(ValueError):
                    self.check(discovery, heldout, cases_per_split=invalid)

    def test_public_exposed_discovery_mode_rejects_holdout_and_cross_split_overlap(self):
        discovery = [list(ids) for _, ids in v2.PUBLIC_DISCOVERY]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {"expected_inventory_sha256": EXPECTED_V2, "expected_model_identity": v1.TARGET}
        with self.assertRaises(ValueError):
            self.check(discovery, heldout, cases_per_split=16)
        result = v2.check_exposed_discovery(discovery, heldout, **options)
        self.assertEqual(result.candidate_count, 32)
        self.assertEqual(result.forbidden_row_count, 204)
        self.assertEqual(result.known_or_reserved_discovery_overlap_count, 16)
        self.assertEqual(result.scope, "known-prefix-holdout-only")
        self.assertNotIn(str(heldout[0]), repr(result))
        disclosure = v2.prepare_preseal_disclosure(
            result, protocol_sha256="c" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertEqual(disclosure.discovery_policy, "known-prefix-discovery-reuse-disclosed")
        self.assertEqual(disclosure.known_or_reserved_discovery_overlap_count, 16)
        self.assertFalse(disclosure.execution_authorized)
        for prefix in v2._forbidden_rows():
            for index in range(16):
                sample = [[3000 + i, 4000 + i] for i in range(16)]
                sample[index] = list(prefix)
                with self.subTest(prefix=prefix, index=index):
                    with self.assertRaises(ValueError):
                        v2.check_exposed_discovery(discovery, sample, **options)
        for left, right in (([7000, 7001], [7000, 7001, 7002]),
                            ([7000, 7001, 7002], [7000, 7001])):
            sample_discovery = [list(ids) for _, ids in v2.PUBLIC_DISCOVERY]
            sample_heldout = [[3000 + i, 4000 + i] for i in range(16)]
            sample_discovery[0], sample_heldout[0] = left, right
            with self.assertRaisesRegex(ValueError, "prefix relationship"):
                v2.check_exposed_discovery(sample_discovery, sample_heldout, **options)
        sample_heldout = [[3000 + i, 4000 + i] for i in range(16)]
        sample_heldout[9] = list(discovery[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            v2.check_exposed_discovery(discovery, sample_heldout, **options)


if __name__ == "__main__":
    unittest.main()
