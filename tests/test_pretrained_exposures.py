"""Public, artificial fixture tests: never construct a real held-out split."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as exposure
from looking_glass import pretrained_protocol as protocol


EXPECTED_DIGEST = "6f21bffeacbe2323c21e53378161443eae8ba28b7210f30e70fd353497e87c8f"


class ExposureInventoryTests(unittest.TestCase):
    @staticmethod
    def candidates():
        # Synthetic tokens only; these are not protocol discovery or held-out cases.
        return ([[1000 + i, 2000 + i] for i in range(8)],
                [[3000 + i, 4000 + i] for i in range(8)])

    def check(self, discovery, heldout, **options):
        return exposure.check_known_splits(
            discovery, heldout,
            expected_inventory_sha256=options.get("digest", EXPECTED_DIGEST),
            expected_model_identity=options.get("identity", exposure.TARGET))

    def test_fixed_canonical_inventory_and_model_identity(self):
        raw = exposure.canonical_inventory_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_DIGEST)
        self.assertEqual(exposure.inventory_digest(), EXPECTED_DIGEST)
        payload = json.loads(raw)
        self.assertEqual(payload["version"], exposure.INVENTORY_VERSION)
        self.assertEqual(payload["reference_source_commit"], "740692931da8deb5dc5c39a9692a2183cea32898")
        self.assertEqual(exposure.TARGET.model_id, protocol.MODEL_ID)
        self.assertEqual(exposure.TARGET.model_revision, protocol.REVISION)
        self.assertEqual(exposure.TARGET.tokenizer_revision, protocol.REVISION)
        self.assertEqual(exposure.TARGET.weights_sha256, protocol.ASSETS["model.safetensors"])
        self.assertEqual(exposure.TARGET.tokenizer_sha256, protocol.ASSETS["tokenizer.json"])
        categories = [record.category for record in exposure.RECORDS]
        self.assertEqual(categories.count("observed-saved"), 10)
        self.assertEqual(categories.count("observed-reported"), 3)
        self.assertEqual(categories.count("planned"), 3)
        self.assertEqual(categories.count("reference-only"), 2)
        self.assertEqual(categories.count("synthetic-only"), 8)
        self.assertEqual(len({record.record_id for record in exposure.RECORDS}), len(exposure.RECORDS))
        self.assertEqual(len({record.token_ids for record in exposure.RECORDS}), len(exposure.RECORDS))

    def test_payload_copy_and_records_do_not_mutate_inventory(self):
        payload = exposure.inventory_payload()
        payload["identity"]["model_id"] = "changed"
        payload["records"][0]["token_ids"][0] = 1
        payload["records"].clear()
        with self.assertRaises(FrozenInstanceError):
            exposure.RECORDS[0].token_ids = (1,)
        self.assertEqual(exposure.inventory_digest(), EXPECTED_DIGEST)

    def test_checks_known_rows_without_claiming_complete_independence(self):
        discovery, heldout = self.candidates()
        result = self.check(discovery, heldout)
        self.assertEqual(result.scope, "known-rows-only")
        self.assertEqual(result.candidate_count, 16)
        self.assertEqual(result.forbidden_row_count, 82)
        self.assertFalse(result.complete_exposure_history_certified)
        self.assertFalse(result.preseal_ready)
        with self.assertRaises(TypeError):
            exposure.KnownSplitCheck(exposure.INVENTORY_VERSION, EXPECTED_DIGEST, 16, 82,
                                     preseal_ready=True)
        self.assertNotIn(str(discovery[0]), repr(result))
        with self.assertRaises(exposure.ExposureInventoryIncomplete):
            exposure.require_preseal_readiness(result)
        with self.assertRaises(exposure.ExposureInventoryIncomplete):
            exposure.prepare_preseal_disclosure(result, protocol_sha256="a" * 64)
        disclosure = exposure.prepare_preseal_disclosure(
            result, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertEqual(disclosure.protocol_sha256, "a" * 64)
        self.assertEqual(disclosure.inventory_sha256, EXPECTED_DIGEST)
        self.assertEqual(disclosure.candidates_per_split, 8)
        self.assertEqual(disclosure.known_prefix_comparison, "trusted-caller-report")
        self.assertEqual(disclosure.prior_exposure_history, "not-independently-certified-complete")
        self.assertFalse(disclosure.sealed_manifest_verified)
        self.assertFalse(disclosure.independent_review_verified)
        self.assertFalse(disclosure.execution_authorized)
        with self.assertRaises(ValueError):
            exposure.prepare_preseal_disclosure(result, protocol_sha256="A" * 64,
                                                  acknowledge_prior_exposure_uncertainty=True)
        discovery[0][0] = 9000
        self.assertEqual(result.candidate_count, 16)

    def test_all_record_categories_are_excluded(self):
        for record in exposure.RECORDS:
            for split in (0, 1):
                with self.subTest(record=record.record_id, split=split):
                    discovery, heldout = self.candidates()
                    (discovery if split == 0 else heldout)[0] = list(record.token_ids)
                    with self.assertRaises(ValueError) as caught:
                        self.check(discovery, heldout)
                    self.assertNotIn(str(record.token_ids), str(caught.exception))

    def test_every_nonempty_prefix_is_conservatively_excluded(self):
        for record in exposure.RECORDS:
            for length in range(1, len(record.token_ids) + 1):
                discovery, heldout = self.candidates()
                heldout[0] = list(record.token_ids[:length])
                with self.subTest(record=record.record_id, length=length):
                    with self.assertRaises(ValueError):
                        self.check(discovery, heldout)
        # A longer input that shares a prefix is not an exact exposed context;
        # the returned check explicitly does not claim semantic independence.
        discovery, heldout = self.candidates()
        heldout[0] = [464, 6766, 318, 4171, 13, 999]
        self.assertEqual(self.check(discovery, heldout).scope, "known-rows-only")

    def test_duplicate_and_invalid_candidate_rows_fail_closed(self):
        discovery, heldout = self.candidates()
        heldout[0] = list(discovery[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(discovery, heldout)
        discovery, heldout = self.candidates()
        discovery[1] = list(discovery[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(discovery, heldout)
        for invalid in ([], [True], [-1], [50257], [1.0], [1] * 33, "1", None):
            with self.subTest(row=invalid):
                discovery, heldout = self.candidates()
                heldout[0] = invalid
                with self.assertRaises(ValueError):
                    self.check(discovery, heldout)
        discovery, heldout = self.candidates()
        with self.assertRaises(ValueError):
            self.check(discovery[:-1], heldout)
        with self.assertRaises(ValueError):
            self.check(discovery, iter(heldout))

    def test_digest_identity_and_inventory_drift_fail_before_success(self):
        discovery, heldout = self.candidates()
        for wrong in (None, "0" * 64, EXPECTED_DIGEST.upper()):
            with self.subTest(digest=wrong):
                with self.assertRaises(ValueError):
                    self.check(discovery, heldout, digest=wrong)
        for wrong in (None, replace(exposure.TARGET, tokenizer_sha256="0" * 64),
                      replace(exposure.TARGET, model_revision="0" * 40)):
            with self.subTest(identity=wrong):
                with self.assertRaises(ValueError):
                    self.check(discovery, heldout, identity=wrong)
        with mock.patch.object(exposure, "RECORDS", exposure.RECORDS[:-1]):
            with self.assertRaisesRegex(ValueError, "pinned digest"):
                self.check(discovery, heldout)
        with self.assertRaises(ValueError):
            exposure.require_preseal_readiness(None)


if __name__ == "__main__":
    unittest.main()
