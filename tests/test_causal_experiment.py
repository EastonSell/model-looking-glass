import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from looking_glass.causal_experiment import run_experiment, select_sites
from looking_glass import causal_experiment
from looking_glass.model import TinyResidual
from looking_glass.probes import TraceEvent
from looking_glass.research_cases import HELDOUT_CASES, SyntheticCase


def _event(label, sequence, value):
    value = torch.tensor(value, dtype=torch.float32)
    return TraceEvent("test", label, sequence, 0, tuple(value.shape),
                      str(value.dtype), str(value.device), value)


def _test_cases():
    # Test-only inputs and seed: not the experiment's public held-out suite.
    rng = random.Random(17)
    return tuple(SyntheticCase(f"test-{i}", "discovery" if i < 3 else "heldout",
                               tuple(rng.randrange(-100, 101) / 50 for _ in range(8)))
                 for i in range(5))


class CandidateSelectionTests(unittest.TestCase):
    def test_first_difference_and_largest_difference_are_distinct(self):
        # If either selection policy ignores its specified criterion, the
        # deliberately later, larger discrepancy changes this outcome.
        clean = [_event("a", 0, [0.0, 0.0]),
                 _event("b", 1, [0.0, 0.0]),
                 _event("c", 2, [0.0, 0.0])]
        changed = [_event("a", 0, [0.0, 0.0]),
                   _event("b", 1, [0.5, 0.0]),
                   _event("c", 2, [0.0, 2.0])]
        sites = select_sites([(clean, changed)], random_seed=918, expected_nodes=["a", "b", "c"])
        self.assertEqual(sites["first_observed"], "b")
        self.assertEqual(sites["magnitude"], "c")
        self.assertIn(sites["random"], {"a", "b", "c"})

    def test_missing_trace_event_rejects_selection(self):
        clean = [_event("a", 0, [0.0]), _event("b", 1, [0.0])]
        changed = [_event("b", 0, [1.0])]
        with self.assertRaises(ValueError):
            select_sites([(clean, changed)], random_seed=918, expected_nodes=["a", "b"])

    def test_matching_truncated_traces_do_not_count_as_dense_coverage(self):
        clean = [_event("a", 0, [0.0]), _event("b", 1, [0.0])]
        changed = [_event("a", 0, [0.0]), _event("b", 1, [1.0])]
        with self.assertRaises(ValueError):
            select_sites([(clean, changed)], random_seed=918, expected_nodes=["a", "b", "c"])


class StudyTests(unittest.TestCase):
    def test_thread_restoration_failure_is_recorded_as_failed(self):
        previous = torch.get_num_threads()
        actual_set_threads = torch.set_num_threads
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "restore-failure"

            def fail_first_restore(count):
                calls.append(count)
                if len(calls) == 2:
                    raise RuntimeError("thread restoration failed")
                actual_set_threads(count)

            try:
                with patch.object(torch, "set_num_threads", fail_first_restore):
                    with self.assertRaisesRegex(RuntimeError, "thread restoration failed"):
                        run_experiment(path, cases=_test_cases(), model_seed=17, random_seed=3)
            finally:
                actual_set_threads(previous)
            self.assertEqual(json.loads((path / "outcome.json").read_text())["state"], "failed")

    def test_repeatability_and_process_state_restoration(self):
        prior_rng = torch.random.get_rng_state().clone()
        prior_threads = torch.get_num_threads()
        with tempfile.TemporaryDirectory() as temp:
            first = run_experiment(Path(temp) / "first", cases=_test_cases(),
                                   model_seed=17, random_seed=3)
            second = run_experiment(Path(temp) / "second", cases=_test_cases(),
                                    model_seed=17, random_seed=3)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), prior_rng))
        self.assertEqual(torch.get_num_threads(), prior_threads)

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "keep.txt"
            marker.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_experiment(temp, cases=_test_cases(), model_seed=17, random_seed=3)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_malformed_heldout_occurrence_aborts_run(self):
        cases = _test_cases()
        heldout = torch.tensor([cases[-1].values], dtype=torch.float32)
        original = causal_experiment._trace
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid"

            def malformed(model, x, *, corrupt):
                metric, events = original(model, x, corrupt=corrupt)
                if corrupt and torch.equal(x, heldout):
                    events[1].occurrence = 1
                return metric, events

            with patch.object(causal_experiment, "_trace", malformed):
                with self.assertRaisesRegex(ValueError, "metadata"):
                    run_experiment(path, cases=cases, model_seed=17, random_seed=3)
            self.assertEqual(json.loads((path / "outcome.json").read_text())["state"], "failed")

    def test_heldout_forward_occurs_only_after_predictions_are_published(self):
        cases = _test_cases()
        heldout_inputs = [torch.tensor([case.values], dtype=torch.float32)
                          for case in cases if case.split == "heldout"]
        original = TinyResidual.forward
        checked = []
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "study"
            precommit = path / "artifacts" / "precommit.json"

            def check_order(model, x, *args, **kwargs):
                if any(torch.equal(x, expected) for expected in heldout_inputs):
                    self.assertTrue(precommit.is_file(), "held-out forward before freeze")
                    frozen = json.loads(precommit.read_text(encoding="utf-8"))
                    self.assertEqual(len(frozen["heldout_cases"]), 2)
                    self.assertIn("first_observed", frozen["predictions"])
                    checked.append(True)
                return original(model, x, *args, **kwargs)

            with patch.object(TinyResidual, "forward", check_order):
                result = run_experiment(path, cases=cases, model_seed=17, random_seed=3)
            self.assertTrue(checked)
            self.assertEqual(result["precommit_sha256"],
                             hashlib.sha256(precommit.read_bytes()).hexdigest())

    def test_known_injection_is_removed_and_reverse_patch_restores_it(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_experiment(Path(temp) / "study", cases=_test_cases(),
                                    model_seed=17, random_seed=3)
        self.assertEqual(result["sites"]["first_observed"], "block.1.activation")
        for row in result["cases"]:
            target = row["target_correction"]
            self.assertAlmostEqual(row["interventions"]["first_observed"]["effect"], target, places=6)
            self.assertAlmostEqual(row["controls"]["reverse_patch_effect"], -target, places=6)
            self.assertEqual(row["controls"]["null_effect"], 0.0)
            self.assertEqual(row["controls"]["identity_patch_effect"], 0.0)
            self.assertEqual(row["controls"]["upstream_patch_effect"], 0.0)
            self.assertEqual(row["controls"]["clean_null_effect"], 0.0)
            self.assertIsNone(row["controls"]["clean_trace_difference"])

    def test_heldout_failure_retains_frozen_predictions_and_failed_outcome(self):
        cases = _test_cases()
        heldout = torch.tensor([cases[-1].values], dtype=torch.float32)
        original = TinyResidual.forward
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "failed"

            def fail_heldout(model, x, *args, **kwargs):
                if torch.equal(x, heldout):
                    raise RuntimeError("deliberate heldout failure")
                return original(model, x, *args, **kwargs)

            with patch.object(TinyResidual, "forward", fail_heldout):
                with self.assertRaisesRegex(RuntimeError, "deliberate heldout failure"):
                    run_experiment(path, cases=cases, model_seed=17, random_seed=3)
            self.assertTrue((path / "artifacts" / "precommit.json").is_file())
            self.assertEqual(json.loads((path / "outcome.json").read_text())["state"], "failed")
            self.assertFalse((path / "artifacts" / "results.json").exists())
            partial = json.loads((path / "artifacts" / "partial_results.json").read_text())
            self.assertEqual([row["case_id"] for row in partial["cases"]], [cases[-2].case_id])

    def test_a_nonzero_null_control_fails_the_run(self):
        cases = _test_cases()
        heldout = torch.tensor([cases[-1].values], dtype=torch.float32)
        original = causal_experiment._run_patch_output
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid-null"

            def bad_null(model, x, module_path, *, source=None, corrupt, operation="patch"):
                output = original(model, x, module_path, source=source, corrupt=corrupt,
                                  operation=operation)
                if operation == "null" and torch.equal(x, heldout):
                    output = output.clone()
                    output[0, 0] += 0.1
                return output

            with patch.object(causal_experiment, "_run_patch_output", bad_null):
                with self.assertRaisesRegex(RuntimeError, "control"):
                    run_experiment(path, cases=cases, model_seed=17, random_seed=3)
            self.assertEqual(json.loads((path / "outcome.json").read_text())["state"], "failed")

    def test_null_common_mode_logit_change_fails_even_if_margin_is_unchanged(self):
        cases = _test_cases()
        heldout = torch.tensor([cases[-1].values], dtype=torch.float32)
        original = causal_experiment._run_patch_output
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid-common-mode"

            def bad_output(model, x, module_path, *, source=None, corrupt, operation="patch"):
                output = original(model, x, module_path, source=source, corrupt=corrupt,
                                  operation=operation)
                if operation == "null" and corrupt and torch.equal(x, heldout):
                    old_metric = causal_experiment._metric(output)
                    for offset in (0.0625, -0.0625, 0.125, -0.125, 0.25, -0.25, 1.0, -1.0):
                        changed = output + offset
                        if causal_experiment._metric(changed) == old_metric and not torch.equal(changed, output):
                            return changed
                    self.fail("test fixture could not create an equal-margin common-mode change")
                return output

            with patch.object(causal_experiment, "_run_patch_output", bad_output):
                with self.assertRaisesRegex(RuntimeError, "control"):
                    run_experiment(path, cases=cases, model_seed=17, random_seed=3)
            self.assertEqual(json.loads((path / "outcome.json").read_text())["state"], "failed")

    def test_duplicate_inputs_are_rejected_before_heldout_evaluation(self):
        cases = list(_test_cases())
        cases[-1] = SyntheticCase(cases[-1].case_id, "heldout", cases[0].values)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_experiment(Path(temp) / "duplicate", cases=cases,
                               model_seed=17, random_seed=3)

    def test_development_rejects_reserved_input_even_when_renamed(self):
        cases = list(_test_cases())
        cases[-1] = SyntheticCase("renamed-heldout", "heldout", HELDOUT_CASES[0].values)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "reserved"):
                run_experiment(Path(temp) / "leak", cases=cases, model_seed=17, random_seed=3)

    def test_no_observed_difference_is_not_a_candidate(self):
        clean = [_event("a", 0, [0.0])]
        changed = [_event("a", 0, [0.0])]
        with self.assertRaises(ValueError):
            select_sites([(clean, changed)], random_seed=918, expected_nodes=["a"])


if __name__ == "__main__":
    unittest.main()
