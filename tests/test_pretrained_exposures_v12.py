"""Verify a public, planned-only factorial reservation without model work."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v6 as v6
from looking_glass import pretrained_exposures_v11 as v11

try:
    from looking_glass import pretrained_exposures_v12 as v12
except ImportError:
    v12 = None

ROOT = Path(v1.__file__).resolve().parents[1]


def committed_bytes(path):
    return subprocess.run(["git", "show", v12.CASES_COMMIT + ":" + path],
                          cwd=ROOT, check=True, capture_output=True).stdout


class ExposureV12Tests(unittest.TestCase):
    def setUp(self):
        if v12 is None:
            self.fail("the v12 exposure reservation has not been implemented")

    def test_only_the_32_fixed_public_cases_are_added_as_planned(self):
        old, new = v11.inventory_payload(), v12.inventory_payload()
        self.assertEqual(new["predecessor"], {"version": v11.INVENTORY_VERSION,
                                               "sha256": v11.INVENTORY_SHA256})
        self.assertEqual(new["update"], {"planned_public_copying_factorial_rows": 32})
        self.assertEqual(new["records"][:183], old["records"])
        self.assertEqual(len(new["records"]), 215)
        added = new["records"][183:]
        self.assertEqual([tuple(row["token_ids"]) for row in added], list(v12.FACTORIAL_ROWS))
        self.assertEqual([row["record_id"] for row in added],
                         [f"planned-copy-factorial-{i // 4 + 1:02d}-{i % 4}" for i in range(32)])
        self.assertEqual([row["target_token_id"] for row in added],
                         [group[5] for group in v12.FACTORIAL_GROUPS for _ in range(4)])
        self.assertTrue(all(row["category"] == "planned" and
                            row["reference_commit"] == v12.CASES_COMMIT and
                            row["references"] == list(v12.CASES_PATHS) for row in added))
        self.assertTrue(all(row["category"] == "planned" for row in new["records"][170:183]))
        old_prefixes = {tuple(row["token_ids"][:length]) for row in old["records"]
                        for length in range(1, len(row["token_ids"]) + 1)}
        with mock.patch.object(v12, "inventory_payload", return_value=new):
            self.assertTrue(old_prefixes <= v12._forbidden_rows())
            self.assertEqual(len(v12._forbidden_rows()), 1709)
            self.assertEqual(hashlib.sha256(v12.canonical_inventory_bytes()).hexdigest(), v12.inventory_digest())

    def test_public_case_and_prior_protocol_bind_the_reservation(self):
        protocol, raw = [committed_bytes(path) for path in v12.CASES_PATHS]
        self.assertEqual(hashlib.sha256(protocol).hexdigest(), v12.PROTOCOL_SHA256)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), v12.CASES_SHA256)
        subprocess.run(["git", "merge-base", "--is-ancestor", v12.PROTOCOL_COMMIT,
                        v12.CASES_COMMIT], cwd=ROOT, check=True)
        record = json.loads(raw)
        self.assertEqual((record["known_exposure_version"], record["known_exposure_sha256"]),
                         (v11.INVENTORY_VERSION, v11.INVENTORY_SHA256))
        self.assertEqual((record["model_id"], record["model_revision"]),
                         (v12.TARGET.model_id, v12.TARGET.model_revision))
        self.assertEqual((record["protocol_revision"], record["protocol_sha256"]),
                         (v12.PROTOCOL_COMMIT, v12.PROTOCOL_SHA256))
        self.assertEqual(record["tokenizer_sha256"],
                         "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6")
        self.assertEqual(record["expected_forwards"], 224)
        self.assertEqual([tuple(group["token_ids"]) for group in record["groups"]],
                         list(v12.FACTORIAL_GROUPS))
        self.assertEqual([tuple(perm) for perm in record["permutations"]], list(v6.PUBLIC_COPY_PERMUTATIONS))
        cases = record["cases"]
        self.assertEqual([(case["case_id"], tuple(case["token_ids"]), case["target_id"],
                           case["group"], case["variant"]) for case in cases],
                         [(f"copy-factorial-{i // 4 + 1:02d}-{i % 4}", row,
                           v12.FACTORIAL_GROUPS[i // 4][5], i // 4 + 1, i % 4)
                          for i, row in enumerate(v12.FACTORIAL_ROWS)])
        complete = [case["token_ids"] + [case["target_id"]] for case in cases]
        self.assertEqual(hashlib.sha256(json.dumps(complete, separators=(",", ":")).encode()).hexdigest(),
                         v12.COMPLETE_ROWS_SHA256)

    def test_predecessor_and_new_rows_reject_both_prefix_directions(self):
        candidate = list(v12.FACTORIAL_ROWS[0])
        original = v11.inventory_payload()
        target = v12.FACTORIAL_GROUPS[0][5]
        for old_row in (candidate[:-1], candidate + [target], candidate + [target, 123],
                        candidate[:2] + [50256]):
            changed = copy.deepcopy(original)
            changed["records"][0]["token_ids"] = old_row
            # Pin the synthetic predecessor to exercise the independent prefix guard.
            synthetic_digest = hashlib.sha256(json.dumps(changed, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
            with self.subTest(old_row=old_row), \
                    mock.patch.object(v11, "inventory_payload", return_value=changed), \
                    mock.patch.object(v11, "INVENTORY_SHA256", synthetic_digest), \
                    mock.patch.object(v12, "PREDECESSOR_SHA256", synthetic_digest):
                with self.assertRaisesRegex(ValueError, "prefix"):
                    v12.inventory_payload()
        rows = list(v12.FACTORIAL_ROWS)
        rows[1] = rows[0][:-1]
        with mock.patch.object(v12, "FACTORIAL_ROWS", tuple(rows)):
            with self.assertRaises(ValueError):
                v12.inventory_payload()
        with mock.patch.object(v11, "INVENTORY_SHA256", "0" * 64):
            with self.assertRaises(ValueError):
                v12.inventory_payload()

    def test_known_checks_exclude_new_rows_without_authorizing_a_seal(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        payload = v12.inventory_payload()
        options = {"expected_inventory_sha256": v12.INVENTORY_SHA256, "expected_model_identity": v12.TARGET}
        with mock.patch.object(v12, "inventory_payload", return_value=payload):
            check = v12.check_known_splits(discovery, heldout, cases_per_split=16, **options)
            for candidate in (list(v12.FACTORIAL_ROWS[0][:-1]), list(v12.FACTORIAL_ROWS[0]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = candidate
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        v12.check_known_splits(discovery, altered, cases_per_split=16, **options)
                    with self.assertRaises(ValueError):
                        v12.check_exposed_discovery(discovery, altered, **options)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v12.require_preseal_readiness(check)
            disclosure = v12.prepare_preseal_disclosure(
                check, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
            self.assertFalse(disclosure.execution_authorized)


if __name__ == "__main__":
    unittest.main()
