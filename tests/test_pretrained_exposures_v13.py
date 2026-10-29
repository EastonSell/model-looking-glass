"""Verify the public factorial status update without model execution."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v12 as v12

try:
    from looking_glass import pretrained_exposures_v13 as v13
except ImportError:
    v13 = None

ROOT = Path(v1.__file__).resolve().parents[1]


def committed_bytes(path):
    return subprocess.run(["git", "show", v13.EVIDENCE_COMMIT + ":" + path],
                          cwd=ROOT, check=True, capture_output=True).stdout


class ExposureV13Tests(unittest.TestCase):
    def setUp(self):
        if v13 is None:
            self.fail("the v13 exposure successor has not been implemented")

    def test_only_32_completed_factorial_reservations_change_status(self):
        old, new = v12.inventory_payload(), v13.inventory_payload()
        self.assertEqual(new["predecessor"], {"version": v12.INVENTORY_VERSION,
                                               "sha256": v12.INVENTORY_SHA256})
        self.assertEqual(new["update"], {"observed_public_copying_factorial_rows": 32})
        self.assertEqual((len(old["records"]), len(new["records"])), (215, 215))
        changed = []
        for before, after in zip(old["records"], new["records"]):
            if before["record_id"].startswith("planned-copy-factorial-"):
                changed.append(after)
                self.assertEqual(before["category"], "planned")
                self.assertEqual(after["record_id"], before["record_id"].replace("planned-", "observed-", 1))
                self.assertEqual(after["category"], "observed-saved")
                for field in ("token_ids", "target_token_id", "public_text"):
                    self.assertEqual(after[field], before[field])
                self.assertEqual(after["reference_commit"], v13.EVIDENCE_COMMIT)
                self.assertEqual(after["references"], list(v13.EVIDENCE_PATHS))
            else:
                self.assertEqual(after, before)
        self.assertEqual([tuple(row["token_ids"]) for row in changed], list(v12.FACTORIAL_ROWS))
        self.assertEqual([row["target_token_id"] for row in changed],
                         [group[5] for group in v12.FACTORIAL_GROUPS for _ in range(4)])
        self.assertEqual(Counter(row["category"] for row in new["records"]), {
            "observed-saved": 168, "observed-reported": 3, "planned": 32,
            "reference-only": 4, "synthetic-only": 8,
        })
        self.assertEqual(new["records"][170:183], old["records"][170:183])
        self.assertTrue(all(row["record_id"].startswith("planned-projected-")
                            for row in new["records"][170:183]))
        old_prefixes = {tuple(row["token_ids"][:length]) for row in old["records"]
                        for length in range(1, len(row["token_ids"]) + 1)}
        with mock.patch.object(v13, "inventory_payload", return_value=new):
            self.assertEqual(v13._forbidden_rows(), old_prefixes)
            self.assertEqual(len(v13._forbidden_rows()), 1709)
            self.assertEqual(hashlib.sha256(v13.canonical_inventory_bytes()).hexdigest(), v13.inventory_digest())

    def test_published_evidence_binds_all_cases_and_actions(self):
        raw = [committed_bytes(path) for path in v13.EVIDENCE_PATHS]
        for value, expected in zip(raw, (v13.MANIFEST_SHA256, v13.RESULT_SHA256,
                                         v13.ATTEMPT_SHA256, v13.SUPERVISOR_SHA256)):
            self.assertEqual(hashlib.sha256(value).hexdigest(), expected)
        manifest, result = map(json.loads, raw[:2])
        events = [json.loads(line) for line in raw[2].splitlines()]
        receipt = json.loads(raw[3])
        self.assertEqual((manifest["model_id"], manifest["model_revision"]),
                         (v12.TARGET.model_id, v12.TARGET.model_revision))
        self.assertEqual((manifest["cases_sha256"], manifest["protocol_sha256"]),
                         (v12.CASES_SHA256, v12.PROTOCOL_SHA256))
        self.assertEqual((manifest["exposure_inventories"]["reservation_version"],
                          manifest["exposure_inventories"]["reservation"]),
                         (v12.INVENTORY_VERSION, v12.INVENTORY_SHA256))
        expected = [(f"copy-factorial-{i // 4 + 1:02d}-{i % 4}", list(row),
                     v12.FACTORIAL_GROUPS[i // 4][5], i // 4 + 1, i % 4)
                    for i, row in enumerate(v12.FACTORIAL_ROWS)]
        self.assertEqual([(c["case_id"], c["token_ids"], c["target_id"], c["group"], c["variant"])
                          for c in manifest["cases"]], expected)
        self.assertEqual((result["status"], result["submitted_forwards"], result["returned_forwards"]),
                         ("complete", 224, 224))
        self.assertEqual(result["manifest_sha256"], v13.MANIFEST_SHA256)
        self.assertEqual((result["checkpoint"]["model_id"], result["checkpoint"]["model_revision"],
                          result["checkpoint"]["weights_sha256"]),
                         (manifest["model_id"], manifest["model_revision"], manifest["weights_sha256"]))
        self.assertEqual(result["checkpoint"]["asset_sha256"]["model.safetensors"],
                         manifest["weights_sha256"])
        self.assertEqual((receipt["process_exit_code"], receipt["failure_file_present"], receipt["interrupted"]),
                         (0, False, False))
        for path, digest in zip(v13.EVIDENCE_PATHS[:3], (v13.MANIFEST_SHA256,
                                                        v13.RESULT_SHA256, v13.ATTEMPT_SHA256)):
            self.assertEqual(receipt["run_sha256"][Path(path).name], digest)
        self.assertTrue(all(case_id + ".f32" in receipt["run_sha256"] for case_id, *_ in expected))
        self.assertEqual(Counter(event["event"] for event in events),
                         {"runtime_ready": 1, "start": 224, "raw_recorded": 224, "complete": 224,
                          "case_diagnostics": 32, "family_dose": 8})
        self.assertEqual(events[0]["event"], "runtime_ready")
        actions = {"clean", "noop", "clean_after", "h10", "h2", "h7", "joint"}
        expected_keys = {(case_id, action) for case_id, *_ in expected for action in actions}
        records = {(record["case_id"], record["action"]): record for record in result["cases"]}
        self.assertEqual(len(result["cases"]), 224)
        self.assertEqual(set(records), expected_keys)
        by_kind, positions = {}, {}
        for kind in ("start", "raw_recorded", "complete"):
            selected = [event for event in events if event["event"] == kind]
            by_kind[kind] = {(event["case_id"], event["action"]): event for event in selected}
            positions[kind] = {(event["case_id"], event["action"]): index
                               for index, event in enumerate(events) if event["event"] == kind}
            self.assertEqual(len(selected), len(by_kind[kind]))
            self.assertEqual(set(by_kind[kind]), expected_keys)
        for case_id, _, target_id, group, variant in expected:
            for action in actions:
                key = (case_id, action)
                record = records[key]
                self.assertLess(positions["start"][key], positions["raw_recorded"][key])
                self.assertLess(positions["raw_recorded"][key], positions["complete"][key])
                self.assertEqual((record["target_id"], record["group"], record["variant"]),
                                 (target_id, group, variant))
                self.assertEqual(by_kind["raw_recorded"][key]["raw"], record["raw"])
                self.assertEqual({k: value for k, value in by_kind["complete"][key].items() if k != "event"},
                                 record)

    def test_predecessor_drift_fails_closed(self):
        old = copy.deepcopy(v12.inventory_payload())
        row = next(r for r in old["records"] if r["record_id"].startswith("planned-copy-factorial-"))
        row["target_token_id"] += 1
        with mock.patch.object(v12, "inventory_payload", return_value=old), \
                mock.patch.object(v12, "inventory_digest", return_value=v12.INVENTORY_SHA256):
            with self.assertRaises(ValueError):
                v13.inventory_payload()
        with mock.patch.object(v12, "INVENTORY_SHA256", "0" * 64):
            with self.assertRaises(ValueError):
                v13.inventory_payload()

    def test_known_checks_keep_both_prefix_directions_and_do_not_authorize_seal(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        payload = v13.inventory_payload()
        options = {"expected_inventory_sha256": v13.INVENTORY_SHA256, "expected_model_identity": v13.TARGET}
        with mock.patch.object(v13, "inventory_payload", return_value=payload):
            check = v13.check_known_splits(discovery, heldout, cases_per_split=16, **options)
            self.assertEqual((check.inventory_version, check.candidate_count), (v13.INVENTORY_VERSION, 32))
            for candidate in (list(v12.FACTORIAL_ROWS[0][:-1]), list(v12.FACTORIAL_ROWS[0]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = candidate
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        v13.check_known_splits(discovery, altered, cases_per_split=16, **options)
                    with self.assertRaises(ValueError):
                        v13.check_exposed_discovery(discovery, altered, **options)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v13.require_preseal_readiness(check)
            disclosure = v13.prepare_preseal_disclosure(
                check, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
            self.assertFalse(disclosure.execution_authorized)


if __name__ == "__main__":
    unittest.main()
