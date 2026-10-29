import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from looking_glass.model import TinyResidual
from looking_glass.registry import ExperimentRun, verify_run
from looking_glass.research_cases import SYNTHETIC_CASES, float32_values
from looking_glass.v2_resource_pilot import (ForwardBudgetExceeded, generate_pilot_inputs,
                                            planned_forwards, run_pilot, _verify_denormal_policy, _bits_equal)


class V2PilotTests(unittest.TestCase):
    def test_full_count_includes_all_reused_and_condition_forwards(self):
        self.assertEqual(planned_forwards(6, 32, 9, 2), 57408)
        self.assertEqual(planned_forwards(1, 2, 3, 2), 202)

    def test_pilot_inputs_are_unique_bounded_and_sign_balanced(self):
        cases = generate_pilot_inputs((49351,), 4)
        self.assertEqual(cases, generate_pilot_inputs((49351,), 4))
        self.assertEqual(len(cases), 4)
        values = [tuple(case["values"]) for case in cases]
        self.assertEqual(len(set(values)), len(values))
        self.assertTrue(all(-1 <= value <= 1 for row in values for value in row))
        self.assertEqual(sum(row[0] > 0 for row in values), 2)
        self.assertFalse(set(values) & {float32_values(case.values) for case in SYNTHETIC_CASES})
        with self.assertRaises(ValueError):
            generate_pilot_inputs((31201,), 2)
        with self.assertRaises(ValueError):
            generate_pilot_inputs((2301,), 2)

    def test_tiny_pilot_counts_real_forwards_and_preserves_exact_input_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pilot"
            result = run_pilot(path)
            inputs = (path / "artifacts" / "pilot_inputs.json").read_bytes()
            plan = json.loads((path / "artifacts" / "plan.json").read_text())
            self.assertEqual(result["observed_forwards"], 202)
            self.assertEqual(plan["expected_forwards"], 202)
            self.assertEqual(plan["v2_planned_forwards"], 57408)
            self.assertEqual(hashlib.sha256(inputs).hexdigest(), result["pilot_input_sha256"])
            self.assertEqual(len(json.loads(inputs)["inputs"]), 2)
            verification = verify_run(path)
            self.assertEqual(verification["state"], "succeeded")
            identity = verification["manifest"]["identities"]["code"]["value"]
            clarification = Path(__file__).resolve().parents[1] / "docs" / "CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md"
            self.assertEqual(identity["v2_numeric_clarification_sha256"], hashlib.sha256(clarification.read_bytes()).hexdigest())
            self.assertGreater(result["process_peak_working_set_bytes"], 0)
            self.assertGreater(result["initialization_intraop_threads"], 0)
            self.assertGreater(result["entry_to_workload_end_wall_seconds"], 0)
            self.assertGreater(result["entry_to_workload_start_wall_seconds"], 0)
            self.assertGreater(result["workload_wall_seconds"], 0)
            self.assertLessEqual(result["workload_wall_seconds"], result["entry_to_workload_end_wall_seconds"])
            self.assertGreaterEqual(result["entry_to_workload_end_cpu_seconds"], 0)

    def test_every_pilot_forward_runs_without_gradient_tracking(self):
        original = TinyResidual.forward
        observed = []

        def check_no_grad(model, *args, **kwargs):
            observed.append(torch.is_grad_enabled())
            return original(model, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(TinyResidual, "forward", check_no_grad):
                run_pilot(Path(temp) / "pilot")
        self.assertEqual(len(observed), 202)
        self.assertFalse(any(observed))

    def test_ambient_flush_denormals_is_rejected_without_a_forward(self):
        command = """
import tempfile, torch
from pathlib import Path
from looking_glass.v2_resource_pilot import run_pilot
assert torch.set_flush_denormal(True)
with tempfile.TemporaryDirectory() as temp:
    try:
        run_pilot(Path(temp) / 'bad')
    except RuntimeError as exc:
        assert 'denormal' in str(exc)
    else:
        raise AssertionError('flush-denormal mode was accepted')
"""
        completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_output_only_subnormal_flushing_is_rejected(self):
        with patch.object(torch, "mul", return_value=torch.zeros(1, dtype=torch.float32)):
            with self.assertRaisesRegex(RuntimeError, "denormal"):
                _verify_denormal_policy()

    def test_exact_comparison_distinguishes_subnormal_and_equates_signed_zero(self):
        positive_zero = torch.tensor([0], dtype=torch.int32).view(torch.float32)
        negative_zero = torch.tensor([-2147483648], dtype=torch.int32).view(torch.float32)
        minimum_subnormal = torch.tensor([1], dtype=torch.int32).view(torch.float32)
        self.assertTrue(_bits_equal(positive_zero, negative_zero))
        self.assertFalse(_bits_equal(positive_zero, minimum_subnormal))
        command = """
import torch
from looking_glass.v2_resource_pilot import _bits_equal
assert torch.set_flush_denormal(True)
zero = torch.tensor([0], dtype=torch.int32).view(torch.float32)
small = torch.tensor([1], dtype=torch.int32).view(torch.float32)
assert not _bits_equal(zero, small), 'bit comparator lost subnormal under FTZ'
"""
        completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_budget_failure_retains_inputs_plan_and_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "failed"
            with self.assertRaises(ForwardBudgetExceeded):
                run_pilot(path, max_forwards=1)
            state = verify_run(path)
            self.assertEqual(state["state"], "failed")
            self.assertTrue((path / "artifacts" / "pilot_inputs.json").is_file())
            self.assertTrue((path / "artifacts" / "plan.json").is_file())
            partial = json.loads((path / "artifacts" / "partial.json").read_text())
            self.assertEqual(partial["observed_forwards"], 1)
            self.assertFalse((path / "artifacts" / "results.json").exists())

    def test_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "keep.txt"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                run_pilot(temp)
            self.assertEqual(marker.read_text(), "original")

    def test_thread_query_failure_after_registry_creation_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "query-failure"
            original = torch.get_num_threads
            calls = []

            def fail_second_query():
                calls.append(True)
                if len(calls) == 2:
                    raise RuntimeError("thread query failure")
                return original()

            with patch.object(torch, "get_num_threads", side_effect=fail_second_query):
                with self.assertRaisesRegex(RuntimeError, "thread query failure"):
                    run_pilot(path)
            self.assertEqual(verify_run(path)["state"], "failed")
            self.assertTrue((path / "artifacts" / "pilot_inputs.json").is_file())

    def test_successful_commit_followed_by_cleanup_error_is_reported_accurately(self):
        original = ExperimentRun.finalize

        def finalize_then_fail(run, state, details):
            original(run, state, details)
            if state == "succeeded":
                raise RuntimeError("post-publication cleanup failed")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "committed"
            with patch.object(ExperimentRun, "finalize", finalize_then_fail):
                with self.assertRaisesRegex(RuntimeError, "committed as succeeded"):
                    run_pilot(path)
            self.assertEqual(verify_run(path)["state"], "succeeded")

    def test_mismatched_published_outcome_is_not_claimed_success(self):
        original = ExperimentRun.finalize

        def publish_wrong_count(run, state, details):
            original(run, state, details)
            if state == "succeeded":
                path = run.path / "outcome.json"
                outcome = json.loads(path.read_text(encoding="utf-8"))
                outcome["details"]["observed_forwards"] = 999
                path.write_text(json.dumps(outcome, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                raise RuntimeError("post-publication cleanup failed")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrong-outcome"
            with patch.object(ExperimentRun, "finalize", publish_wrong_count):
                with self.assertRaisesRegex(RuntimeError, "outcome state unknown"):
                    run_pilot(path)


if __name__ == "__main__":
    unittest.main()
