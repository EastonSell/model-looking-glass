import json
from contextlib import contextmanager
import inspect
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from looking_glass.causal_v2_gate_dev import DevelopmentCommittedCleanupError, DevelopmentGate, DevelopmentSeal
from looking_glass import registry
from looking_glass.registry import verify_run


class DevelopmentPhaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "gate"
        self.gate = DevelopmentGate.create(
            self.path, cells=["dev-cell-00", "dev-cell-01"],
            case_ids=["dev-case-00", "dev-case-01"], exclusion_sha256="a" * 64)

    def complete_discovery(self):
        self.gate.record_discovery("dev-cell-00", 0.25)
        self.gate.record_discovery("dev-cell-01", -0.5)
        return self.gate.freeze()

    def test_evaluation_before_freeze_never_invokes_callback_and_stops_attempt(self):
        calls = []
        with self.assertRaises((RuntimeError, ValueError)):
            self.gate.evaluate(None, lambda: calls.append(1))
        self.assertEqual(calls, [])
        with self.assertRaises(RuntimeError):
            self.gate.record_discovery("dev-cell-00", 0.25)

    def test_freeze_requires_every_declared_cell(self):
        self.gate.record_discovery("dev-cell-00", 0.25)
        with self.assertRaises(RuntimeError):
            self.gate.freeze()
        with self.assertRaises(RuntimeError):
            self.gate.record_discovery("dev-cell-01", -0.5)
        self.assertEqual(verify_run(self.path)["state"], "failed")

    def test_evaluation_observes_frozen_files_commitment_and_claim(self):
        frozen = self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        calls = []
        def callback():
            state = verify_run(self.path)
            self.assertEqual(state["state"], "incomplete")
            self.assertTrue({"freeze.json", "prediction_record.json", "prediction_commitment.json", "evaluation_claim.json"}
                            <= set(state["artifacts"]))
            self.assertEqual(state["artifacts"]["freeze.json"]["sha256"], frozen)
            self.assertEqual(state["artifacts"]["prediction_commitment.json"]["sha256"], seal.public_sha256)
            calls.append(1)
        self.gate.evaluate(seal, callback)
        self.assertEqual(calls, [1])
        self.assertEqual(verify_run(self.path)["state"], "succeeded")
        with self.assertRaises(RuntimeError):
            self.gate.evaluate(seal, callback)

    def test_changed_prediction_bytes_are_rejected_before_callback(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        changed = DevelopmentSeal(seal.nonce, b'{}', seal.public_sha256)
        calls = []
        with self.assertRaises((RuntimeError, ValueError)):
            self.gate.evaluate(changed, lambda: calls.append(1))
        self.assertEqual(calls, [])
        self.assertEqual(verify_run(self.path)["state"], "failed")

    def test_oversize_prediction_bytes_are_rejected_before_callback(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        changed = DevelopmentSeal(seal.nonce, b"[" * 262145, seal.public_sha256)
        calls = []
        with self.assertRaises(ValueError):
            self.gate.evaluate(changed, lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_record_published_without_commitment_cannot_reach_callback(self):
        self.complete_discovery()
        original = self.gate.run.write_artifact
        def fail(name, raw):
            if name == "prediction_commitment.json":
                raise OSError("injected commitment failure")
            return original(name, raw)
        with patch.object(self.gate.run, "write_artifact", side_effect=fail):
            with self.assertRaises(OSError):
                self.gate.seal_predictions(bytes([4]) * 32)
        self.assertTrue((self.path / "artifacts" / "prediction_record.json").exists())
        self.assertFalse((self.path / "artifacts" / "prediction_commitment.json").exists())
        with self.assertRaises(RuntimeError):
            self.gate.evaluate(None, lambda: self.fail("callback ran"))

    def test_failed_freeze_publication_is_not_resumed(self):
        self.gate.record_discovery("dev-cell-00", 0.25)
        self.gate.record_discovery("dev-cell-01", -0.5)
        original = self.gate.run.write_artifact
        def fail(name, value):
            if name == "freeze.json":
                raise OSError("injected failure")
            return original(name, value)
        with patch.object(self.gate.run, "write_artifact", side_effect=fail):
            with self.assertRaises(OSError):
                self.gate.freeze()
        with self.assertRaises(RuntimeError):
            self.gate.seal_predictions(bytes([4]) * 32)
        self.assertEqual(verify_run(self.path)["state"], "failed")

    def test_claim_error_after_link_never_calls_or_retries_callback(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        original = self.gate.run.write_artifact
        def after_link(name, value):
            original(name, value)
            if name == "evaluation_claim.json":
                raise OSError("injected post-link failure")
        calls = []
        with patch.object(self.gate.run, "write_artifact", side_effect=after_link):
            with self.assertRaises(OSError):
                self.gate.evaluate(seal, lambda: calls.append(1))
        self.assertEqual(calls, [])
        with self.assertRaises(RuntimeError):
            self.gate.evaluate(seal, lambda: calls.append(1))
        self.assertEqual(calls, [])
        self.assertEqual(verify_run(self.path)["state"], "failed")

    def test_cancelled_callback_cannot_be_repeated(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        def cancel():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.gate.evaluate(seal, cancel)
        self.assertEqual(verify_run(self.path)["state"], "cancelled")
        with self.assertRaises(RuntimeError):
            self.gate.evaluate(seal, lambda: None)

    def test_interrupt_at_busy_assignment_stops_the_attempt(self):
        lines, first = inspect.getsourcelines(DevelopmentGate._transition)
        target = first + next(i for i, line in enumerate(lines) if "self._busy = True" in line)
        fired = []
        def trace(frame, event, arg):
            if (event == "line" and frame.f_code is DevelopmentGate._transition.__code__
                    and frame.f_lineno == target and not fired):
                fired.append(True)
                raise KeyboardInterrupt("injected transition cancellation")
            return trace
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                self.gate.record_discovery("dev-cell-00", 0.25)
        finally:
            sys.settrace(previous)
        self.assertEqual(fired, [True])
        self.assertEqual(verify_run(self.path)["state"], "cancelled")
        with self.assertRaises(RuntimeError):
            self.gate.record_discovery("dev-cell-00", 0.25)

    def test_fresh_cancellation_during_reconciliation_preserves_both_errors(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        callback_entered = []
        original_status = self.gate.run.status
        def interrupt_status():
            if callback_entered:
                raise KeyboardInterrupt("interrupted reconciliation")
            return original_status()
        def fail_callback():
            callback_entered.append(True)
            raise ValueError("original callback error")
        with patch.object(self.gate.run, "status", side_effect=interrupt_status):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.gate.evaluate(seal, fail_callback)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(str(caught.exception.__cause__), "original callback error")
        self.assertEqual(verify_run(self.path)["state"], "incomplete")

    def test_terminal_failure_keeps_original_and_storage_error_without_cycle(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        def fail_callback():
            raise ValueError("original callback error")
        with patch.object(self.gate.run, "finalize", side_effect=OSError("terminal write failed")):
            with self.assertRaises(ValueError) as caught:
                self.gate.evaluate(seal, fail_callback)
        unknown = caught.exception.__cause__
        self.assertIsInstance(unknown, RuntimeError)
        self.assertIsInstance(unknown.__cause__, OSError)
        self.assertIsNot(unknown.__cause__, caught.exception)
        self.assertEqual(verify_run(self.path)["state"], "incomplete")

    def test_mutated_frozen_evidence_during_callback_cannot_succeed(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        def mutate():
            (self.path / "artifacts" / "freeze.json").write_bytes(b'{}')
        with self.assertRaises(RuntimeError):
            self.gate.evaluate(seal, mutate)
        self.assertEqual(verify_run(self.path)["state"], "failed")

    def test_success_committed_before_cleanup_error_is_reconciled(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        original = self.gate.run.finalize
        def after_commit(state, details):
            original(state, details)
            if state == "succeeded":
                raise OSError("injected post-commit failure")
        with patch.object(self.gate.run, "finalize", side_effect=after_commit):
            with self.assertRaises(DevelopmentCommittedCleanupError):
                self.gate.evaluate(seal, lambda: None)
        self.assertEqual(verify_run(self.path)["state"], "succeeded")
        with self.assertRaises(RuntimeError):
            self.gate.evaluate(seal, lambda: None)

    def test_phase_lock_exit_failure_reconciles_success(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        original = registry._lock
        @contextmanager
        def unlock_failure(path, timeout):
            with original(path, timeout):
                try:
                    yield
                finally:
                    if Path(path).name == ".phase":
                        raise OSError("injected phase unlock failure")
        with patch.object(registry, "_lock", side_effect=unlock_failure):
            with self.assertRaises(DevelopmentCommittedCleanupError):
                self.gate.evaluate(seal, lambda: None)
        self.assertEqual(verify_run(self.path)["state"], "succeeded")

    def test_phase_lock_exit_does_not_mask_callback_cancellation(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        original = registry._lock
        @contextmanager
        def unlock_failure(path, timeout):
            with original(path, timeout):
                try:
                    yield
                finally:
                    if Path(path).name == ".phase":
                        raise OSError("injected phase unlock failure")
        def cancel():
            raise KeyboardInterrupt()
        with patch.object(registry, "_lock", side_effect=unlock_failure):
            with self.assertRaises(KeyboardInterrupt):
                self.gate.evaluate(seal, cancel)
        self.assertEqual(verify_run(self.path)["state"], "cancelled")

    def test_fresh_cancellation_at_unlock_chains_callback_error(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        original = registry._lock
        @contextmanager
        def interrupted_unlock(path, timeout):
            with original(path, timeout):
                try:
                    yield
                finally:
                    if Path(path).name == ".phase":
                        raise KeyboardInterrupt("injected unlock cancellation")
        def fail_callback():
            raise OSError("original callback failure")
        with patch.object(registry, "_lock", side_effect=interrupted_unlock):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.gate.evaluate(seal, fail_callback)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(str(caught.exception.__cause__), "original callback failure")
        self.assertEqual(verify_run(self.path)["state"], "cancelled")

    def test_success_committed_before_interrupt_preserves_cancellation_type(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        original = self.gate.run.finalize
        def after_commit(state, details):
            original(state, details)
            if state == "succeeded":
                raise KeyboardInterrupt()
        with patch.object(self.gate.run, "finalize", side_effect=after_commit):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.gate.evaluate(seal, lambda: None)
        self.assertIsInstance(caught.exception.__cause__, DevelopmentCommittedCleanupError)
        self.assertEqual(verify_run(self.path)["state"], "succeeded")

    def test_success_committed_before_system_exit_preserves_code(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        original = self.gate.run.finalize
        def after_commit(state, details):
            original(state, details)
            if state == "succeeded":
                raise SystemExit(42)
        with patch.object(self.gate.run, "finalize", side_effect=after_commit):
            with self.assertRaises(SystemExit) as caught:
                self.gate.evaluate(seal, lambda: None)
        self.assertEqual(caught.exception.code, 42)
        self.assertIsInstance(caught.exception.__cause__, DevelopmentCommittedCleanupError)
        self.assertEqual(verify_run(self.path)["state"], "succeeded")

    def test_caught_reentrant_transition_invalidates_attempt(self):
        self.complete_discovery()
        seal = self.gate.seal_predictions(bytes([4]) * 32)
        def reenter():
            try:
                self.gate.evaluate(seal, lambda: self.fail("nested callback ran"))
            except RuntimeError:
                pass
        with self.assertRaisesRegex(RuntimeError, "reentrant"):
            self.gate.evaluate(seal, reenter)
        self.assertEqual(verify_run(self.path)["state"], "failed")

    def test_gate_cannot_reopen_existing_run(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be reopened"):
            DevelopmentGate(self.gate.run, self.gate.public)

    def test_wrong_thread_cannot_advance_the_gate(self):
        errors = []
        def other_thread():
            try:
                self.gate.record_discovery("dev-cell-00", 0.25)
            except RuntimeError as error:
                errors.append(str(error))
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
        self.assertEqual(len(errors), 1)

    def test_reused_thread_identifier_does_not_transfer_ownership(self):
        replacement = Mock(spec=threading.Thread)
        replacement.ident = threading.current_thread().ident
        with patch("looking_glass.causal_v2_gate_dev.threading.current_thread", return_value=replacement):
            with self.assertRaisesRegex(RuntimeError, "owner changed"):
                self.gate.record_discovery("dev-cell-00", 0.25)

    def test_only_development_ids_and_new_directory_are_accepted(self):
        with self.assertRaises(ValueError):
            DevelopmentGate.create(Path(self.temp.name) / "other", cells=["v2-m31201-c00"],
                                   case_ids=["dev-case-00"], exclusion_sha256="a" * 64)
        with self.assertRaises(FileExistsError):
            DevelopmentGate.create(self.path, cells=["dev-cell-00"], case_ids=["dev-case-00"],
                                   exclusion_sha256="a" * 64)

    def test_fresh_cancellation_during_setup_preserves_publication_error(self):
        setup_path = Path(self.temp.name) / "setup-failure"
        with patch.object(registry.ExperimentRun, "write_artifact", side_effect=OSError("manifest write failed")):
            with patch.object(registry.ExperimentRun, "finalize", side_effect=KeyboardInterrupt("setup cancellation")):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    DevelopmentGate.create(setup_path, cells=["dev-cell-00"],
                                           case_ids=["dev-case-00"], exclusion_sha256="a" * 64)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(verify_run(setup_path)["state"], "incomplete")


if __name__ == "__main__":
    unittest.main()
