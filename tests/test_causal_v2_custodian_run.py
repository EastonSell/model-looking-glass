"""One-shot ordering tests with fabricated records and stubbed model work."""
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass import registry
from looking_glass import causal_v2_custodian_run as custodian
from looking_glass.v2_evidence import Evidence
from looking_glass.causal_v2_custody import canonical_json, commitment
from looking_glass.causal_v2_custodian_run import (CommittedSuccessError, CustodianError,
                                                   ForwardBudget, run_study_once)
from test_causal_v2_evidence_adapter import encoded, record_v2


def fake_snapshot():
    context_sha = "d" * 64
    heldout = {"context_sha256": context_sha,
               "cases": [{"case_id": f"fake-{i}", "model_seed": 17, "index": i}
                         for i in range(16)]}
    schedule = {"context_sha256": context_sha,
                "assignments": [{"cell_id": "opaque-cell", "model_seed": 17,
                                 "injection_node": "private-location"}]}
    public = {"mode": "study", "context": {"model_state_hashes": [{"seed": 17, "sha256": "a"*64}]},
              "discovery_cases": [{"case_id": f"fake-{i}", "model_seed": 17, "index": i}
                                  for i in range(16)],
              "heldout_commitment": commitment("heldout", b"h"*32, heldout),
              "schedule_commitment": commitment("schedule", b"s"*32, schedule)}
    return SimpleNamespace(public=canonical_json(public), heldout=canonical_json(heldout),
                           schedule=canonical_json(schedule), heldout_root=b"h"*32,
                           site_nonce=b"s"*32, prediction_nonce=b"p"*32)


def fake_freeze(snapshot, pre_manifest, cells):
    assert len(cells) == 1
    freeze = canonical_json({"schema": "fake-freeze", "cells": cells})
    predictions = {"pre_manifest_sha256": hashlib.sha256(pre_manifest).hexdigest(),
                   "freeze_sha256": hashlib.sha256(freeze).hexdigest(),
                   "predictions": [{"model_seed": 17, "cell_id": "opaque-cell", "case_id": f"fake-{i}",
                                    "strength": strength, "site": site}
                                   for i in range(16) for strength in (0.25, 0.75) for site in (1, 2, 3)]}
    raw = canonical_json(predictions)
    public = canonical_json({"schema": "synthetic-v2-prediction-commitment-v1",
                             "pre_manifest_sha256": predictions["pre_manifest_sha256"],
                             "freeze_sha256": predictions["freeze_sha256"],
                             "commitment": commitment("predictions", snapshot.prediction_nonce, predictions)})
    return freeze, raw, public


def fake_summary(manifest, freeze, prediction, chunks, exposure):
    cells, conditions = [], []
    for raw in chunks:
        chunk = json.loads(raw)
        rows = chunk["records"]
        cells.append({"model_seed": rows[0]["model_seed"], "cell_id": rows[0]["cell_id"],
                      "chunk_name": chunk["name"], "chunk_sha256": hashlib.sha256(raw).hexdigest()})
        conditions.extend({key: row[key] for key in
                           ("model_seed", "cell_id", "case_id", "case_index", "strength")}
                          for row in rows)
    label = {"unknown": "unconfirmed_exposure_unknown", "unblinded": "exploratory_unblinded",
             "unexposed": "confirmation_conditional_on_caller_attestations"}[exposure]
    return canonical_json({"schema": "synthetic-v2-heldout-summary-v1", "status": "complete",
        "mode": "study", "interpretation": label, "exposure_classification": exposure,
        "pre_manifest_sha256": hashlib.sha256(manifest).hexdigest(), "freeze_sha256": hashlib.sha256(freeze).hexdigest(),
        "prediction_sha256": hashlib.sha256(prediction).hexdigest(), "models": [{"model_seed": 17}],
        "equal_model_means": {}, "cells": cells, "conditions_count": len(conditions),
        "conditions": conditions, "repair_groups": [{} for _ in range(len(cells) * 12)]})


class FakeExecutor:
    def __init__(self, root, *, fail_discovery=False, fail_heldout=False, fail_close=False):
        self.root, self.fail_discovery, self.fail_close = root, fail_discovery, fail_close
        self.fail_heldout = fail_heldout
        self.calls = []
        self.closed = False
        self.budgets = []
        self.begin_counts = []
        self.cross_phase_collision = False
        self.cross_phase_condition_collision = False
        self.unicode_run_ids = False

    def begin_model(self, phase, seed, model_sha256, budget):
        assert model_sha256 == "a" * 64
        self.budgets.append(budget)
        self.begin_counts.append(budget.submissions)
        self.calls.append(("begin", phase, seed))

    def _cell(self, phase, assignment, cases, cell, budget, predictions=None):
        self.calls.append(("cell", phase))
        self.budgets.append(budget)
        for index in range(16):
            for strength in (0.25, 0.75):
                row = record_v2(index, strength, phase=phase)
                row["case_id"] = cases[index]["case_id"]
                row["condition_id"] = f"{phase}-condition-{index}-{strength}"
                if self.cross_phase_condition_collision and phase == "heldout" and index == 0:
                    row["condition_id"] = f"discovery-condition-{index}-{strength}"
                names = {"clean": f"{phase}-clean-{index}", "repeat_clean": f"{phase}-repeat-{index}",
                         "perturbed": f"{phase}-perturbed-{index}-{strength}"}
                if self.unicode_run_ids:
                    names = {role: f"{name}-caf\u00e9" for role, name in names.items()}
                if self.cross_phase_collision and phase == "heldout" and index == 0:
                    names["clean"] = "discovery-clean-0"
                for role, events in row["traces"].items():
                    for event in events:
                        event["metadata"]["run_id"] = names[role]
                if predictions is not None:
                    row["prediction_ref"] = predictions[index * 2 + (strength == 0.75)]["reference"]
                budget.reserve(); budget.root()
                cell.append(encoded(row))
                if self.fail_discovery and phase == "discovery":
                    raise RuntimeError("private-location failure")
                if self.fail_heldout and phase == "heldout":
                    raise RuntimeError("private held-out failure")

    def run_discovery(self, assignment, cases, cell, budget):
        self._cell("discovery", assignment, cases, cell, budget)
        self.selection = {"selection": "fake"}
        return self.selection

    def run_heldout(self, assignment, cases, predictions, cell, budget):
        public = self.root / "public" / "artifacts"
        private = self.root / "private" / "artifacts"
        assert (public / "pre_manifest.json").exists()
        assert (public / "prediction_freeze.json").exists()
        assert (public / "prediction_commitment.json").exists()
        assert (private / "prediction_record.json").exists()
        assert len(predictions) == 32
        self._cell("heldout", assignment, cases, cell, budget, predictions)

    def end_model(self):
        self.calls.append(("end",))

    def close(self):
        self.calls.append(("close",))
        self.closed = True
        if self.fail_close:
            raise RuntimeError("cleanup failed")
        return {"input_model_pairing_verified": True, "private_plan_metadata_verified": True}


class CustodianRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.snapshot = fake_snapshot()
        self.manifest = canonical_json({"schema": "fake-public-pre-manifest"})
        self.executor = FakeExecutor(self.root)
        self.patches = [patch("looking_glass.causal_v2_custodian_run._checked_preparation",
                              return_value=(self.snapshot, self.manifest)),
                        patch("looking_glass.causal_v2_custodian_run._build_freeze", side_effect=fake_freeze),
                        patch("looking_glass.causal_v2_custodian_run._verify_discovery",
                              side_effect=lambda snapshot, cells, chunks: list(chunks)),
                        patch("looking_glass.causal_v2_custodian_run._MODEL_SEEDS", (17,)),
                        patch("looking_glass.causal_v2_custodian_run._CELLS_PER_MODEL", 1),
                        patch("looking_glass.causal_v2_custodian_run._EXPECTED_FORWARDS", 64)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def run_once(self, scorer_override=None, classifier_override=None):
        def scorer(manifest, freeze, prediction, commitment_bytes, heldout, schedule,
                   heldout_root, site_nonce, prediction_nonce, chunks, **attestations):
            self.assertTrue(self.executor.closed)
            self.assertTrue(all(attestations.values()))
            self.assertEqual(attestations["exposure_classification"], "unknown")
            return fake_summary(manifest, freeze, prediction, chunks, attestations["exposure_classification"])
        return run_study_once(self.root, prepared=None, expected_context={}, exclusion_bytes=b"",
            pilot_bytes=b"", historical_cases=[], source_sha256={}, clarification_sha256={},
            runtime={}, created_utc="fake", executor=self.executor, score=scorer_override or scorer,
            classify_exposure=classifier_override or (lambda: {"schema": "synthetic-v2-exposure-v1",
                "classification": "unknown", "basis": "fabricated test"}))

    def test_discovery_freeze_commitment_precede_any_heldout_callback(self):
        result = self.run_once()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["root_forwards"], 64)
        self.assertEqual([c for c in self.executor.calls if c[0] == "cell"],
                         [("cell", "discovery"), ("cell", "heldout")])
        self.assertTrue(all(budget is self.executor.budgets[0] for budget in self.executor.budgets))
        self.assertEqual(self.executor.begin_counts, [0, 32])
        private = registry.verify_run(self.root / "private")
        public = registry.verify_run(self.root / "public")
        self.assertEqual(private["state"], "succeeded")
        self.assertEqual(public["state"], "succeeded")
        self.assertTrue((self.root / "public" / "artifacts" / "reveal.json").exists())
        exposure = json.loads((self.root / "private" / "artifacts" / "exposure_record.json").read_bytes())
        self.assertEqual(exposure["classification"], "unknown")

    def test_discovery_failure_preserves_prefix_and_never_starts_holdout(self):
        self.executor.fail_discovery = True
        with self.assertRaisesRegex(RuntimeError, "private-location failure"):
            self.run_once()
        self.assertTrue(self.executor.closed)
        self.assertNotIn(("cell", "heldout"), self.executor.calls)
        private = registry.verify_run(self.root / "private")
        self.assertEqual(private["state"], "failed")
        chunk = json.loads((self.root / "private" / "artifacts" / "cell-00-d.json").read_bytes())
        self.assertEqual(chunk["state"], "incomplete")
        self.assertEqual(len(chunk["records"]), 1)
        public = registry.verify_run(self.root / "public")
        self.assertEqual(public["state"], "failed")
        self.assertNotIn("private-location", json.dumps(public))

    def test_commitment_write_failure_prevents_holdout_and_does_not_retry_publication(self):
        original = registry.ExperimentRun.write_artifact
        def fail_commit(run, name, data):
            if name == "prediction_commitment.json":
                raise OSError("ambiguous public write")
            return original(run, name, data)
        with patch.object(registry.ExperimentRun, "write_artifact", new=fail_commit):
            with self.assertRaisesRegex(OSError, "ambiguous public write"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "incomplete")
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "failed")

    def test_success_after_terminal_acknowledgement_error_is_reconciled(self):
        original = registry.ExperimentRun.finalize
        def after_final(run, state, details):
            original(run, state, details)
            if run.path.name == "public" and state == "succeeded":
                raise OSError("terminal acknowledgement")
        with patch.object(registry.ExperimentRun, "finalize", new=after_final):
            with self.assertRaises(CommittedSuccessError):
                self.run_once()
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "succeeded")
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "succeeded")

    def test_discovery_binding_failure_blocks_freeze_and_holdout(self):
        with patch("looking_glass.causal_v2_custodian_run._verify_discovery",
                   side_effect=ValueError("saved discovery differs")):
            with self.assertRaisesRegex(ValueError, "saved discovery differs"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)
        self.assertFalse((self.root / "public" / "artifacts" / "prediction_freeze.json").exists())
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "failed")

    def test_mismatched_prediction_commitment_blocks_holdout(self):
        def broken(snapshot, manifest, cells):
            freeze, prediction, raw = fake_freeze(snapshot, manifest, cells)
            public = json.loads(raw)
            public["commitment"] = "0" * 64
            return freeze, prediction, canonical_json(public)
        with patch("looking_glass.causal_v2_custodian_run._build_freeze", side_effect=broken):
            with self.assertRaisesRegex(CustodianError, "commitment differs"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)
        self.assertFalse((self.root / "public" / "artifacts" / "prediction_commitment.json").exists())

    def test_failed_executor_cleanup_preserves_primary_error_and_partial_evidence(self):
        self.executor.fail_discovery = True
        self.executor.fail_close = True
        with self.assertRaisesRegex(RuntimeError, "private-location failure") as found:
            self.run_once()
        self.assertRegex(str(found.exception.__cause__), "cleanup failed")
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "failed")
        failure = json.loads((self.root / "private" / "artifacts" / "failure.json").read_bytes())
        self.assertIn("private-location failure", failure["message"])
        self.assertIn("cleanup failed", failure["message"])
        self.assertNotIn("private-location", json.dumps(registry.verify_run(self.root / "public")))

    def test_fresh_interrupt_during_cleanup_is_primary_and_recorded_as_cancelled(self):
        self.executor.fail_discovery = True
        def cancel():
            raise KeyboardInterrupt("cleanup interruption")
        self.executor.close = cancel
        with self.assertRaisesRegex(KeyboardInterrupt, "cleanup interruption") as found:
            self.run_once()
        self.assertRegex(str(found.exception.__cause__), "private-location failure")
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "cancelled")
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "cancelled")
        failure = json.loads((self.root / "private" / "artifacts" / "failure.json").read_bytes())
        self.assertEqual(failure["error_type"], "KeyboardInterrupt")
        self.assertIn("private-location failure", failure["message"])

    def test_precommit_acknowledgement_failure_does_not_retry_even_after_link(self):
        original = registry.ExperimentRun.write_artifact
        def after_write(run, name, data):
            original(run, name, data)
            if name == "prediction_commitment.json":
                raise OSError("linked but unacknowledged")
        with patch.object(registry.ExperimentRun, "write_artifact", new=after_write):
            with self.assertRaisesRegex(OSError, "unacknowledged"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)
        self.assertTrue((self.root / "public" / "artifacts" / "prediction_commitment.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "incomplete")
        self.assertFalse((self.root / "public" / "artifacts" / "failure.json").exists())

    def test_forward_budget_rejects_duplicate_and_unreserved_roots(self):
        budget = ForwardBudget(1)
        with self.assertRaises(CustodianError):
            budget.root()
        budget.reserve()
        with self.assertRaises(CustodianError):
            budget.reserve()
        budget.root()
        with self.assertRaises(CustodianError):
            budget.root()
        with self.assertRaises(CustodianError):
            budget.reserve()
        self.assertEqual((budget.submissions, budget.root_forwards), (1, 1))

    def test_late_invalid_prediction_group_blocks_first_heldout_forward(self):
        def broken(snapshot, manifest, cells):
            freeze, prediction, _ = fake_freeze(snapshot, manifest, cells)
            record = json.loads(prediction)
            record["predictions"][-1]["site"] = 9
            prediction = canonical_json(record)
            public = canonical_json({"schema": "synthetic-v2-prediction-commitment-v1",
                "pre_manifest_sha256": record["pre_manifest_sha256"],
                "freeze_sha256": record["freeze_sha256"],
                "commitment": commitment("predictions", snapshot.prediction_nonce, record)})
            return freeze, prediction, public
        with patch("looking_glass.causal_v2_custodian_run._build_freeze", side_effect=broken):
            with self.assertRaisesRegex(CustodianError, "prediction group identity"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)

    def test_scorer_must_consume_all_persisted_heldout_chunks(self):
        def ignore_chunks(*args, **kwargs):
            return canonical_json({"schema": "invalid-incomplete-result"})
        with self.assertRaisesRegex(CustodianError, "did not consume all"):
            self.run_once(ignore_chunks)
        self.assertFalse((self.root / "public" / "artifacts" / "results.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_heldout_failure_preserves_partial_without_result_or_reveal(self):
        self.executor.fail_heldout = True
        with self.assertRaisesRegex(RuntimeError, "private held-out failure"):
            self.run_once()
        self.assertTrue(self.executor.closed)
        chunk = json.loads((self.root / "private" / "artifacts" / "cell-00-h.json").read_bytes())
        self.assertEqual(chunk["state"], "incomplete")
        self.assertEqual(len(chunk["records"]), 1)
        self.assertFalse((self.root / "public" / "artifacts" / "results.json").exists())
        self.assertFalse((self.root / "public" / "artifacts" / "reveal.json").exists())

    def test_pending_public_write_stops_without_automatic_cleanup(self):
        original = registry.ExperimentRun.write_artifact
        def leaves_pending(run, name, data):
            original(run, name, data)
            if name == "pre_manifest.json":
                (run.path / ".pending").write_bytes(b"uncertain")
        with patch.object(registry.ExperimentRun, "write_artifact", new=leaves_pending):
            with self.assertRaisesRegex(CustodianError, "pending evidence"):
                self.run_once()
        self.assertEqual(self.executor.calls, [("close",)])
        state = registry.verify_run(self.root / "public")
        self.assertEqual(state["state"], "incomplete")
        self.assertTrue(state["pending_write"])
        self.assertEqual((self.root / "public" / ".pending").read_bytes(), b"uncertain")

    def test_verifier_must_consume_all_persisted_discovery_chunks(self):
        with patch("looking_glass.causal_v2_custodian_run._verify_discovery", return_value=None):
            with self.assertRaisesRegex(CustodianError, "did not consume all"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)

    def test_failure_terminal_ack_error_reconciles_without_retry(self):
        self.executor.fail_discovery = True
        original = Evidence.finalize
        def after_private(run, state, details):
            original(run, state, details)
            if state == "failed":
                raise OSError("failure terminal acknowledgement")
        with patch.object(Evidence, "finalize", new=after_private):
            with self.assertRaisesRegex(RuntimeError, "private-location failure") as found:
                self.run_once()
        self.assertRegex(str(found.exception.__cause__), "terminal acknowledgement")
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "failed")
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_oversize_failure_is_not_truncated_or_declared_private_success(self):
        def huge(*args):
            raise RuntimeError("X" * (70 * 1024))
        self.executor.run_discovery = huge
        with self.assertRaises(RuntimeError):
            self.run_once()
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "incomplete")
        self.assertFalse((self.root / "private" / "artifacts" / "failure.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")
        public_failure = json.loads((self.root / "public" / "artifacts" / "failure.json").read_bytes())
        self.assertEqual(public_failure["private_failure_evidence"], "evidence_limit")

    def test_missing_executor_receipt_blocks_scoring_and_success(self):
        self.executor.close = lambda: {"input_model_pairing_verified": True,
                                       "private_plan_metadata_verified": False}
        with self.assertRaisesRegex(CustodianError, "did not verify"):
            self.run_once()
        self.assertFalse((self.root / "public" / "artifacts" / "results.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_preparation_failure_still_closes_executor_without_creating_run(self):
        with patch("looking_glass.causal_v2_custodian_run._checked_preparation",
                   side_effect=ValueError("invalid preparation")):
            with self.assertRaisesRegex(ValueError, "invalid preparation"):
                self.run_once()
        self.assertEqual(self.executor.calls, [("close",)])
        self.assertFalse(self.root.exists())

    def test_private_success_is_not_mislabeled_as_complete_failure_evidence(self):
        original = custodian._read
        def fail_private_outcome(path, maximum):
            if path.name == "outcome.json" and path.parent.name == "private":
                raise OSError("outcome read failed")
            return original(path, maximum)
        with patch.object(custodian, "_read", new=fail_private_outcome):
            with self.assertRaisesRegex(OSError, "outcome read failed"):
                self.run_once()
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "succeeded")
        public = json.loads((self.root / "public" / "artifacts" / "failure.json").read_bytes())
        self.assertEqual(public["private_failure_evidence"], "unavailable")
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_terminal_sync_failure_reports_visible_success_with_uncertain_durability(self):
        original = custodian._sync_directory
        def fail_after_terminal(path):
            if path.name == "public" and (path / "outcome.json").exists():
                raise OSError("directory sync failed")
            return original(path)
        with patch.object(custodian, "_sync_directory", new=fail_after_terminal):
            with self.assertRaisesRegex(CommittedSuccessError, "durability is uncertain"):
                self.run_once()
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "succeeded")

    def test_interrupt_after_complete_cell_saves_cancellation_without_retrying_chunk(self):
        lines, start = inspect.getsourcelines(run_study_once)
        target = start + next(i for i, line in enumerate(lines)
                              if line.strip() == "save_cell(active_cell, True)") + 1
        fired = []
        def interrupt(frame, event, arg):
            if (not fired and event == "line" and frame.f_code is run_study_once.__code__
                    and frame.f_lineno == target):
                fired.append(True)
                raise KeyboardInterrupt("after complete chunk")
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaisesRegex(KeyboardInterrupt, "after complete chunk"):
                self.run_once()
        finally:
            sys.settrace(None)
        self.assertTrue(fired)
        chunk = json.loads((self.root / "private" / "artifacts" / "cell-00-d.json").read_bytes())
        self.assertEqual(chunk["state"], "complete")
        self.assertEqual(len(chunk["records"]), 32)
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "cancelled")
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "cancelled")

    def test_invalid_exposure_classification_prevents_scoring_and_reveal(self):
        with self.assertRaisesRegex(CustodianError, "invalid exposure"):
            self.run_once(classifier_override=lambda: {"schema": "synthetic-v2-exposure-v1",
                "classification": "unexposed", "basis": ""})
        self.assertFalse((self.root / "public" / "artifacts" / "results.json").exists())
        self.assertFalse((self.root / "public" / "artifacts" / "reveal.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_explicit_unblinding_is_passed_to_scorer_and_preserved_at_reveal(self):
        seen = []
        def scorer(*args, **kwargs):
            seen.append(kwargs["exposure_classification"])
            return fake_summary(args[0], args[1], args[2], args[-1], seen[-1])
        self.run_once(scorer_override=scorer, classifier_override=lambda: {
            "schema": "synthetic-v2-exposure-v1", "classification": "unblinded",
            "basis": "fabricated disclosure log"})
        self.assertEqual(seen, ["unblinded"])
        reveal = json.loads((self.root / "public" / "artifacts" / "reveal.json").read_bytes())
        self.assertEqual(reveal["exposure"]["classification"], "unblinded")
        self.assertNotIn("basis", reveal["exposure"])
        private_raw = (self.root / "private" / "artifacts" / "exposure_record.json").read_bytes()
        self.assertEqual(reveal["exposure"]["private_record_sha256"], hashlib.sha256(private_raw).hexdigest())
        self.assertEqual(json.loads(private_raw)["basis"], "fabricated disclosure log")

    def test_wrong_scorer_exposure_label_is_not_published(self):
        def scorer(*args, **kwargs):
            return fake_summary(args[0], args[1], args[2], args[-1], "unexposed")
        with self.assertRaisesRegex(CustodianError, "exposure classification differs"):
            self.run_once(scorer_override=scorer)
        self.assertFalse((self.root / "public" / "artifacts" / "results.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_cross_phase_trace_id_collision_prevents_result_publication(self):
        self.executor.cross_phase_collision = True
        with self.assertRaisesRegex(CustodianError, "reused across discovery and held-out"):
            self.run_once()
        self.assertFalse((self.root / "public" / "artifacts" / "results.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_cross_phase_condition_id_collision_prevents_result_publication(self):
        self.executor.cross_phase_condition_collision = True
        with self.assertRaisesRegex(CustodianError, "condition ID is reused across"):
            self.run_once()
        self.assertFalse((self.root / "public" / "artifacts" / "results.json").exists())
        self.assertEqual(registry.verify_run(self.root / "public")["state"], "failed")

    def test_non_ascii_registry_trace_ids_are_valid_and_remain_distinct(self):
        self.executor.unicode_run_ids = True
        self.run_once()
        self.assertEqual(registry.verify_run(self.root / "private")["state"], "succeeded")
        raw = (self.root / "private" / "artifacts" / "cell-00-d.json").read_bytes()
        self.assertIn("caf\u00e9".encode(), raw)

    def test_non_ascii_failure_is_preserved_in_registry_encoding(self):
        def fail(*args):
            raise RuntimeError("private \u00e9chec")
        self.executor.run_discovery = fail
        with self.assertRaisesRegex(RuntimeError, "\u00e9chec"):
            self.run_once()
        info = registry.verify_run(self.root / "private")
        self.assertEqual(info["state"], "failed")
        failure = json.loads((self.root / "private" / "artifacts" / "failure.json").read_bytes())
        self.assertIn("\u00e9chec", failure["message"])

    def test_mutating_executor_alias_cannot_change_checked_freeze_input(self):
        def mutate_alias(snapshot, checked_cells, chunks):
            list(chunks)
            self.executor.selection["selection"] = "mutated"
        with patch("looking_glass.causal_v2_custodian_run._verify_discovery", side_effect=mutate_alias):
            self.run_once()
        freeze = json.loads((self.root / "public" / "artifacts" / "prediction_freeze.json").read_bytes())
        self.assertEqual(freeze["cells"][0]["selection"], {"selection": "fake"})

    def test_discovery_verifier_cannot_mutate_selection_snapshot(self):
        def mutate_checked(snapshot, checked_cells, chunks):
            list(chunks)
            checked_cells[0]["selection"] = {"changed": True}
        with patch("looking_glass.causal_v2_custodian_run._verify_discovery", side_effect=mutate_checked):
            with self.assertRaisesRegex(CustodianError, "mutated the frozen selection"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)

    def test_freeze_must_embed_exact_checked_selection_snapshot(self):
        def broken(snapshot, manifest, cells):
            freeze, prediction, _ = fake_freeze(snapshot, manifest, cells)
            frozen = json.loads(freeze)
            frozen["cells"][0]["selection"] = {"changed": True}
            freeze = canonical_json(frozen)
            record = json.loads(prediction)
            record["freeze_sha256"] = hashlib.sha256(freeze).hexdigest()
            prediction = canonical_json(record)
            public = canonical_json({"schema": "synthetic-v2-prediction-commitment-v1",
                "pre_manifest_sha256": record["pre_manifest_sha256"],
                "freeze_sha256": record["freeze_sha256"],
                "commitment": commitment("predictions", snapshot.prediction_nonce, record)})
            return freeze, prediction, public
        with patch("looking_glass.causal_v2_custodian_run._build_freeze", side_effect=broken):
            with self.assertRaisesRegex(CustodianError, "freeze differs from verified selections"):
                self.run_once()
        self.assertNotIn(("cell", "heldout"), self.executor.calls)


if __name__ == "__main__":
    unittest.main()
