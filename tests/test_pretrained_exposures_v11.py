"""Verify the published copying follow-up status update without model work."""
from collections import Counter
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

try:
    from looking_glass import pretrained_exposures_v11 as v11
except ImportError:
    v11 = None

ROOT = Path(v1.__file__).resolve().parents[1]


def committed_bytes(path):
    return subprocess.run(
        ["git", "show", v11.EVIDENCE_COMMIT + ":" + path],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout


class ExposureV11Tests(unittest.TestCase):
    def setUp(self):
        if v11 is None:
            self.fail("the v11 exposure successor has not been implemented")

    def test_only_the_32_completed_reservations_change_status(self):
        old, new = v10.inventory_payload(), v11.inventory_payload()
        self.assertEqual(new["predecessor"], {
            "version": v10.INVENTORY_VERSION, "sha256": v10.INVENTORY_SHA256,
        })
        self.assertEqual(new["update"], {"observed_public_copying_followup_rows": 32})
        self.assertEqual((len(old["records"]), len(new["records"])), (183, 183))
        changed = []
        for before, after in zip(old["records"], new["records"]):
            if before["record_id"].startswith("planned-copy-followup-"):
                changed.append(after)
                self.assertEqual(before["category"], "planned")
                self.assertEqual(after["record_id"], before["record_id"].replace("planned-", "observed-", 1))
                self.assertEqual(after["category"], "observed-saved")
                for field in ("token_ids", "target_token_id", "public_text"):
                    self.assertEqual(after[field], before[field])
                self.assertEqual(after["reference_commit"], v11.EVIDENCE_COMMIT)
                self.assertEqual(after["references"], list(v11.EVIDENCE_PATHS))
            else:
                self.assertEqual(after, before)
        self.assertEqual([tuple(row["token_ids"]) for row in changed], list(v9.FOLLOWUP_ROWS))
        self.assertEqual([row["target_token_id"] for row in changed],
                         [group[5] for group in v9.FOLLOWUP_GROUPS for _ in range(4)])
        self.assertEqual(Counter(row["category"] for row in new["records"]), {
            "observed-saved": 136, "observed-reported": 3, "planned": 32,
            "reference-only": 4, "synthetic-only": 8,
        })
        self.assertEqual(new["records"][-13:], old["records"][-13:])
        self.assertTrue(all(row["record_id"].startswith("planned-projected-")
                            for row in new["records"][-13:]))
        old_prefixes = {tuple(row["token_ids"][:length]) for row in old["records"]
                        for length in range(1, len(row["token_ids"]) + 1)}
        with mock.patch.object(v11, "inventory_payload", return_value=new):
            self.assertEqual(v11._forbidden_rows(), old_prefixes)
            self.assertEqual(len(v11._forbidden_rows()), 1357)
            self.assertEqual(hashlib.sha256(v11.canonical_inventory_bytes()).hexdigest(), v11.inventory_digest())

    def test_merged_evidence_binds_every_reserved_case_and_completed_action(self):
        raw = [committed_bytes(path) for path in v11.EVIDENCE_PATHS]
        for value, expected in zip(raw, (v11.MANIFEST_SHA256, v11.RESULT_SHA256,
                                         v11.ATTEMPT_SHA256, v11.SUPERVISOR_SHA256)):
            self.assertEqual(hashlib.sha256(value).hexdigest(), expected)
        manifest, result = map(json.loads, raw[:2])
        events = [json.loads(line) for line in raw[2].splitlines()]
        receipt = json.loads(raw[3])
        self.assertEqual((manifest["model_id"], manifest["model_revision"]),
                         (v10.TARGET.model_id, v10.TARGET.model_revision))
        self.assertEqual(manifest["cases_sha256"], v9.FOLLOWUP_CASES_SHA256)
        self.assertEqual(manifest["protocol_sha256"], v9.FOLLOWUP_PROTOCOL_SHA256)
        self.assertEqual(manifest["exposure_inventories"]["v9"], v9.INVENTORY_SHA256)
        expected = [(f"copy-followup-{i // 4 + 1:02d}-{i % 4}", list(row),
                     v9.FOLLOWUP_GROUPS[i // 4][5], i // 4 + 1, i % 4)
                    for i, row in enumerate(v9.FOLLOWUP_ROWS)]
        self.assertEqual([(c["case_id"], c["token_ids"], c["target_id"], c["group"], c["variant"])
                          for c in manifest["cases"]], expected)
        self.assertEqual((result["status"], result["submitted_forwards"], result["returned_forwards"]),
                         ("complete", 288, 288))
        self.assertEqual(result["manifest_sha256"], v11.MANIFEST_SHA256)
        self.assertEqual((receipt["process_exit_code"], receipt["failure_file_present"], receipt["interrupted"]),
                         (0, False, False))
        for path, digest in zip(v11.EVIDENCE_PATHS[:3], (v11.MANIFEST_SHA256,
                                                        v11.RESULT_SHA256, v11.ATTEMPT_SHA256)):
            self.assertEqual(receipt["run_sha256"][Path(path).name], digest)
        self.assertEqual(Counter(event["event"] for event in events),
                         {"runtime_ready": 1, "start": 288, "raw_recorded": 288, "complete": 288})
        self.assertEqual(events[0]["event"], "runtime_ready")
        expected_actions = {"clean", "noop", "clean_after", "h10_final", "h2_final",
                            "h7_final", "h3_final", "h1_final", "h10_target"}
        self.assertEqual(len(result["cases"]), 288)
        for index, (case_id, _, target_id, group, variant) in enumerate(expected):
            records = result["cases"][index * 9:(index + 1) * 9]
            self.assertEqual({record["action"] for record in records}, expected_actions)
            self.assertTrue(all((r["case_id"], r["target_id"], r["group"], r["variant"]) ==
                                (case_id, target_id, group, variant) for r in records))
            for j, record in enumerate(records):
                start, saved, complete = events[1 + 3 * (index * 9 + j):1 + 3 * (index * 9 + j + 1)]
                self.assertEqual([start["event"], saved["event"], complete["event"]],
                                 ["start", "raw_recorded", "complete"])
                self.assertTrue(all((item["case_id"], item["action"]) ==
                                    (case_id, record["action"]) for item in (start, saved, complete)))
                self.assertEqual(saved["raw"], record["raw"])
                self.assertEqual({k: value for k, value in complete.items() if k != "event"}, record)

    def test_predecessor_drift_fails_closed(self):
        old = copy.deepcopy(v10.inventory_payload())
        row = next(r for r in old["records"] if r["record_id"].startswith("planned-copy-followup-"))
        row["target_token_id"] += 1
        # Coherent synthetic pin reaches the downstream record-identity guard.
        synthetic_digest = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
        with mock.patch.object(v10, "inventory_payload", return_value=old), \
                mock.patch.object(v10, "INVENTORY_SHA256", synthetic_digest), \
                mock.patch.object(v11, "PREDECESSOR_SHA256", synthetic_digest):
            with self.assertRaisesRegex(ValueError, "copying follow-up"):
                v11.inventory_payload()
        with mock.patch.object(v10, "INVENTORY_SHA256", "0" * 64):
            with self.assertRaises(ValueError):
                v11.inventory_payload()

    def test_known_checks_keep_both_prefix_directions_and_do_not_authorize_seal(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        # Use one verified immutable snapshot to exercise multiple candidate
        # checks without rebuilding every predecessor for each candidate.
        payload = v11.inventory_payload()
        options = {"expected_inventory_sha256": v11.INVENTORY_SHA256, "expected_model_identity": v11.TARGET}
        with mock.patch.object(v11, "inventory_payload", return_value=payload):
            check = v11.check_known_splits(discovery, heldout, cases_per_split=16, **options)
            self.assertEqual((check.inventory_version, check.candidate_count), (v11.INVENTORY_VERSION, 32))
            for known in (list(v9.FOLLOWUP_ROWS[0]), list(v10.PROJECTED_CASES[0][3])):
                for candidate in (known[:-1], known + [123]):
                    with self.subTest(candidate=candidate):
                        altered = copy.deepcopy(heldout)
                        altered[0] = candidate
                        with self.assertRaises(ValueError):
                            v11.check_known_splits(discovery, altered, cases_per_split=16, **options)
                        with self.assertRaises(ValueError):
                            v11.check_exposed_discovery(discovery, altered, **options)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v11.require_preseal_readiness(check)
            disclosure = v11.prepare_preseal_disclosure(
                check, protocol_sha256="a" * 64, acknowledge_prior_exposure_uncertainty=True)
            self.assertFalse(disclosure.execution_authorized)


if __name__ == "__main__":
    unittest.main()
