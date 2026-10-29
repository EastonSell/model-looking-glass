"""Check the projected-edit public reservation without running a model."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v9 as v9
from looking_glass import pretrained_exposures_v10 as v10
from looking_glass import pretrained_number_commitments as commitments


ROOT = Path(v10.__file__).resolve().parents[1]


def committed_bytes(path):
    return subprocess.run(
        ["git", "show", v10.CASES_COMMIT + ":" + path],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout


class ExposureV10Tests(unittest.TestCase):
    def test_successor_preserves_v9_and_adds_only_published_cases(self):
        old, new = v9.inventory_payload(), v10.inventory_payload()
        self.assertEqual(new["predecessor"], {
            "version": v9.INVENTORY_VERSION, "sha256": v9.INVENTORY_SHA256,
        })
        self.assertEqual(new["records"][:170], old["records"])
        self.assertEqual(len(new["records"]), 183)
        added = new["records"][170:]
        cases = json.loads(committed_bytes(v10.CASES_PATH))["tokenizer_preflight"]["cases"]
        self.assertEqual([row["token_ids"] for row in added],
                         [case["token_ids"] for case in cases])
        self.assertEqual([row["public_text"] for row in added],
                         [case["prompt"] for case in cases])
        self.assertEqual([row["target_token_id"] for row in added],
                         [case["continuations"][0]["ids"][0] for case in cases])
        self.assertEqual([row["record_id"] for row in added],
                         ["planned-" + case["case_id"] for case in cases])
        self.assertTrue(all(
            row["category"] == "planned" and row["reference_commit"] == v10.CASES_COMMIT
            and row["references"] == [v10.PROTOCOL_PATH, v10.CASES_PATH]
            for row in added
        ))
        self.assertTrue(v9._forbidden_rows() <= v10._forbidden_rows())
        self.assertEqual(hashlib.sha256(v10.canonical_inventory_bytes()).hexdigest(),
                         v10.inventory_digest())
        snapshot = commitments.load_inventory_snapshot(
            v10.canonical_inventory_bytes(), v10.inventory_digest()
        )
        self.assertEqual(snapshot.sha256, v10.INVENTORY_SHA256)
        with mock.patch.object(v9, "INVENTORY_SHA256", "0" * 64):
            with self.assertRaises(ValueError):
                v10.inventory_payload()

    def test_published_case_and_protocol_bytes_match_the_fixed_rows(self):
        case_raw = committed_bytes(v10.CASES_PATH)
        self.assertEqual(hashlib.sha256(case_raw).hexdigest(), v10.CASES_SHA256)
        self.assertEqual(hashlib.sha256(committed_bytes(v10.PROTOCOL_PATH)).hexdigest(),
                         v10.PROTOCOL_SHA256)
        record = json.loads(case_raw)
        self.assertEqual(record["status"], "public-case-reservation-only")
        self.assertIs(record["tokenizer_preflight"]["model_forward"], False)
        self.assertEqual(
            (record["tokenizer_preflight"]["model_id"], record["tokenizer_preflight"]["revision"]),
            (v10.TARGET.model_id, v10.TARGET.model_revision),
        )
        self.assertEqual(record["tokenizer_preflight"]["tokenizer_json_sha256"],
                         "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6")
        self.assertEqual(record["public_collision_check"]["inventory_sha256"], v9.INVENTORY_SHA256)
        cases = record["tokenizer_preflight"]["cases"]
        self.assertEqual(
            [(case["case_id"], case["split"], case["prompt"], tuple(case["token_ids"]),
              case["continuations"][0]["ids"][0]) for case in cases],
            list(v10.PROJECTED_CASES),
        )
        self.assertEqual(sum(len(case["continuations"]) for case in cases), 17)
        for case in cases:
            for continuation in case["continuations"]:
                self.assertEqual(len(continuation["ids"]), 1)
                self.assertEqual(continuation["full_ids"], case["token_ids"] + continuation["ids"])

    def test_predecessor_and_new_rows_reject_prefix_relations_in_both_directions(self):
        candidate = list(v10.PROJECTED_CASES[0][3])
        for old_row in (candidate[:-1], candidate + [123]):
            old = copy.deepcopy(v9.inventory_payload())
            old["records"][0]["token_ids"] = old_row
            with self.subTest(old_row=old_row), \
                    mock.patch.object(v9, "inventory_payload", return_value=old), \
                    mock.patch.object(v9, "inventory_digest", return_value=v9.INVENTORY_SHA256):
                with self.assertRaisesRegex(ValueError, "prefix"):
                    v10.inventory_payload()
        cases = list(v10.PROJECTED_CASES)
        case = cases[1]
        cases[1] = (case[0], case[1], case[2], tuple(candidate[:-1]), case[4])
        with mock.patch.object(v10, "PROJECTED_CASES", tuple(cases)):
            with self.assertRaisesRegex(ValueError, "prefix"):
                v10.inventory_payload()

    def test_known_prefix_checks_reserve_rows_without_authorizing_a_seal(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        options = {"expected_inventory_sha256": v10.inventory_digest(),
                   "expected_model_identity": v10.TARGET}
        check = v10.check_known_splits(discovery, heldout, cases_per_split=16, **options)
        self.assertEqual((check.inventory_version, check.candidate_count),
                         (v10.INVENTORY_VERSION, 32))
        self.assertTrue({case[3] for case in v10.PROJECTED_CASES} <= v10._forbidden_rows())
        heldout[0] = list(v10.PROJECTED_CASES[0][3])
        with self.assertRaises(ValueError):
            v10.check_exposed_discovery(discovery, heldout, **options)
        with self.assertRaises(v1.ExposureInventoryIncomplete):
            v10.require_preseal_readiness(check)
        disclosure = v10.prepare_preseal_disclosure(
            check, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True
        )
        self.assertFalse(disclosure.execution_authorized)

    def test_future_candidates_reject_both_known_prefix_directions(self):
        known = list(v10.PROJECTED_CASES[0][3])
        options = {"expected_inventory_sha256": v10.inventory_digest(),
                   "expected_model_identity": v10.TARGET}
        for candidate in (known[:-1], known + [123]):
            with self.subTest(candidate=candidate):
                for which in ("discovery", "heldout"):
                    discovery = [[1000 + i, 2000 + i] for i in range(16)]
                    heldout = [[3000 + i, 4000 + i] for i in range(16)]
                    (discovery if which == "discovery" else heldout)[0] = candidate
                    with self.assertRaises(ValueError):
                        v10.check_known_splits(discovery, heldout, cases_per_split=16, **options)
                    if which == "heldout":
                        with self.assertRaises(ValueError):
                            v10.check_exposed_discovery(discovery, heldout, **options)
                discovery = [[1000 + i, 2000 + i] for i in range(16)]
                heldout = [[3000 + i, 4000 + i] for i in range(16)]
                discovery[0] = candidate
                check = v10.check_exposed_discovery(discovery, heldout, **options)
                self.assertEqual(check.known_or_reserved_discovery_overlap_count, 1)


if __name__ == "__main__":
    unittest.main()
