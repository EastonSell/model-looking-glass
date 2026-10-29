"""Public copying reservations and synthetic split probes; no model work."""
from collections import Counter
from dataclasses import replace
import hashlib
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v5 as v5
from looking_glass import pretrained_exposures_v6 as v6
from looking_glass import pretrained_number_commitments as commitments


EXPECTED_V5 = "1c597a028952a40ac48614c0e263a87a034d4f6a3df2b4cbffcdb6047adf6586"
EXPECTED_V6 = "76803775e6929b4a20463a50bcae839a727b1a875a0d433f7c27c9d1d69f0681"


class ExposureV6Tests(unittest.TestCase):
    def test_predecessor_and_public_copy_reservations(self):
        previous, current = v5.inventory_payload(), v6.inventory_payload()
        self.assertEqual(v5.inventory_digest(), EXPECTED_V5)
        self.assertEqual(v6.inventory_digest(), EXPECTED_V6)
        self.assertEqual(hashlib.sha256(v6.canonical_inventory_bytes()).hexdigest(), EXPECTED_V6)
        self.assertEqual(current["predecessor"], {"version": v5.INVENTORY_VERSION, "sha256": EXPECTED_V5})
        self.assertEqual(current["records"][:len(previous["records"])], previous["records"])
        added = current["records"][len(previous["records"]):]
        self.assertEqual(len(added), 32)
        self.assertEqual([tuple(r["token_ids"]) for r in added], list(v6.PUBLIC_COPY_ROWS))
        self.assertEqual(v6.PUBLIC_COPY_ROWS[0], (17180, 7850, 2647, 4077, 8223, 45725,
                                                  17180, 7850, 2647, 4077, 8223))
        self.assertEqual(v6.PUBLIC_COPY_ROWS[-1], (28699, 13385, 41207, 10701, 7543, 15061,
                                                   13385, 10701, 7543, 28699, 41207))
        for index, record in enumerate(added):
            group, variant = divmod(index, 4)
            self.assertEqual(record["record_id"], f"planned-copy-screen-{group + 1:02d}-{variant}")
            self.assertEqual(record["target_token_id"], v6.PUBLIC_COPY_GROUPS[group][5])
            self.assertEqual(record["category"], "planned")
            self.assertEqual(record["reference_commit"], v6.PLAN_COMMIT)
            self.assertEqual(record["references"], list(v6.PLAN_PATHS))
            self.assertEqual(len(record["token_ids"]), 11)
        self.assertEqual(Counter(row["category"] for row in current["records"]),
                         {"observed-saved": 40, "observed-reported": 3, "planned": 51,
                          "reference-only": 4, "synthetic-only": 8})
        self.assertTrue(v5._forbidden_rows() <= v6._forbidden_rows())
        self.assertEqual((len(v6._forbidden_rows()), len(v6._forbidden_rows() - v5._forbidden_rows())),
                         (610, 352))
        with mock.patch.object(v5, "INVENTORY_SHA256", "0" * 64), self.assertRaises(ValueError):
            v6.inventory_digest()
        old = next(tuple(row["token_ids"]) for row in previous["records"] if len(row["token_ids"]) < 32)
        altered = list(v6.PUBLIC_COPY_ROWS)
        altered[0] = old + (1,)
        with mock.patch.object(v6, "PUBLIC_COPY_ROWS", tuple(altered)), self.assertRaises(ValueError):
            v6.inventory_payload()

    def test_copy_rows_and_prefixes_are_excluded_from_synthetic_splits(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {"expected_inventory_sha256": EXPECTED_V6, "expected_model_identity": v6.TARGET}
        check = v6.check_known_splits(discovery, heldout, cases_per_split=16, **options)
        self.assertEqual((check.candidate_count, check.forbidden_row_count), (32, 610))
        for index, row in enumerate(v6.PUBLIC_COPY_ROWS):
            sample = [item[:] for item in heldout]
            sample[index % 16] = list(row)
            with self.subTest(index=index), self.assertRaises(ValueError):
                v6.check_exposed_discovery(discovery, sample, **options)
        for length in range(1, 12):
            sample = [item[:] for item in discovery]
            sample[length % 16] = list(v6.PUBLIC_COPY_ROWS[0][:length])
            with self.subTest(length=length), self.assertRaises(ValueError):
                v6.check_known_splits(sample, heldout, cases_per_split=16, **options)
        with self.assertRaises(ValueError):
            v6.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V5,
                                       expected_model_identity=v6.TARGET)
        with self.assertRaises(ValueError):
            v6.check_exposed_discovery(discovery, heldout, expected_inventory_sha256=EXPECTED_V6,
                                       expected_model_identity=replace(v6.TARGET, weights_sha256="0" * 64))

    def test_disclosure_is_not_authorization_and_snapshot_loads(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        check = v6.check_known_splits(discovery, heldout, expected_inventory_sha256=EXPECTED_V6,
                                      expected_model_identity=v6.TARGET, cases_per_split=16)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v6.require_preseal_readiness(check)
        disclosure = v6.prepare_preseal_disclosure(
            check, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
        self.assertFalse(disclosure.execution_authorized)
        self.assertFalse(disclosure.sealed_manifest_verified)
        snapshot = commitments.load_inventory_snapshot(v6.canonical_inventory_bytes(), EXPECTED_V6)
        self.assertEqual(snapshot.sha256, EXPECTED_V6)


if __name__ == "__main__":
    unittest.main()
