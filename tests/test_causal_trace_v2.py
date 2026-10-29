"""Hand-built unit fixtures only; no v2 discovery or held-out inputs."""
import math
import subprocess
import sys
import unittest
import warnings

import torch

from looking_glass.model import TinyResidual
from looking_glass.probes import TraceEvent, TraceSession
from looking_glass.causal_trace_v2 import SparseTraceInvalid, compare_sparse_pair


def events(run_id, values=None):
    values = values if values is not None else [[0.] * 8 for _ in range(4)]
    output = []
    for index, row in enumerate(values):
        tensor = torch.tensor([row], dtype=torch.float32, device="cpu")
        output.append(TraceEvent(run_id, f"block.{index}.output", index, 0,
                                 (1, 8), "torch.float32", "cpu", tensor))
    return output


def compare(reference, candidate, **kwargs):
    return compare_sparse_pair(reference, candidate, reference_complete=True,
                               candidate_complete=True, pairing_verified=True, **kwargs)


class SparseV2TraceTests(unittest.TestCase):
    def test_exact_difference_returns_first_boundary_and_fsum_norms(self):
        changed = [[0.] * 8, [0.] * 8, [3., 4.] + [0.] * 6, [1.] * 8]
        result = compare(events("reference"), events("candidate", changed))
        self.assertEqual(result["status"], "difference_observed")
        self.assertEqual(result["first_unequal"], "block.2.output")
        self.assertEqual(result["last_equal"], "block.1.output")
        self.assertEqual(result["l2_by_block"], [0., 0., 5., math.sqrt(8.)])

    def test_actual_bounded_sparse_sessions_compare_without_injection_label(self):
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(42)
            model = TinyResidual(width=8, depth=4).eval()
        value = torch.tensor([[1.25, -.5, .75, 0., .125, 1., -.25, .5]],
                             dtype=torch.float32, device="cpu")
        with torch.no_grad(), torch.device("cpu"), TraceSession(
                model, placement="sparse", run_id="reference", max_events=4,
                max_tensor_bytes=128) as reference:
            model(value)
        with torch.no_grad(), torch.device("cpu"), TraceSession(
                model, placement="sparse", run_id="candidate", max_events=4,
                max_tensor_bytes=128) as candidate:
            model(value, perturb=("block.1.activation", .125))
        result = compare(reference.events, candidate.events)
        self.assertEqual(result["status"], "difference_observed")
        self.assertEqual(result["first_unequal"], "block.1.output")
        self.assertEqual(result["last_equal"], "block.0.output")

    def test_no_difference_is_not_a_cell_level_abstention(self):
        result = compare(events("reference"), events("candidate"))
        self.assertEqual(result["status"], "no_difference_observed")
        self.assertIsNone(result["first_unequal"])
        self.assertEqual(result["last_equal"], "block.3.output")
        self.assertEqual(result["l2_by_block"], [0., 0., 0., 0.])

    def test_upstream_change_is_distinct_from_an_eligible_detection(self):
        changed = [[1.] * 8] + [[0.] * 8 for _ in range(3)]
        result = compare(events("reference"), events("candidate", changed))
        self.assertEqual(result["status"], "upstream_difference")
        self.assertEqual(result["first_unequal"], "block.0.output")
        self.assertIsNone(result["last_equal"])

    def test_incomplete_equal_streams_are_not_an_observation_miss(self):
        with self.assertRaises(SparseTraceInvalid):
            compare(events("reference")[:3], events("candidate")[:3])

    def test_unbounded_iterables_are_rejected_without_consumption(self):
        def unbounded():
            raise AssertionError("the validator must not consume this iterator")
            yield None
        with self.assertRaises(SparseTraceInvalid):
            compare(unbounded(), events("candidate"))

    def test_a_missing_later_value_invalidates_an_earlier_difference(self):
        candidate = events("candidate", [[0.] * 8, [1.] * 8, [1.] * 8, [1.] * 8])
        candidate[3].value = None
        with self.assertRaises(SparseTraceInvalid):
            compare(events("reference"), candidate)

    def test_identical_wrong_shape_does_not_pass_the_declared_policy(self):
        reference, candidate = events("reference"), events("candidate")
        for stream in (reference, candidate):
            stream[2].value = torch.zeros(1, 7)
            stream[2].shape = (1, 7)
        with self.assertRaises(SparseTraceInvalid):
            compare(reference, candidate)

    def test_boolean_dimensions_are_not_valid_integer_metadata(self):
        reference, candidate = events("reference"), events("candidate")
        reference[2].shape = (True, 8)
        candidate[2].shape = (True, 8)
        with self.assertRaises(SparseTraceInvalid):
            compare(reference, candidate)

    def test_identically_reordered_tap_names_are_invalid(self):
        reference, candidate = events("reference"), events("candidate")
        for stream in (reference, candidate):
            stream[1].node_id, stream[2].node_id = stream[2].node_id, stream[1].node_id
        with self.assertRaises(SparseTraceInvalid):
            compare(reference, candidate)

    def test_wrong_sequence_and_invocation_are_invalid(self):
        for field, value in (("sequence", 7), ("occurrence", 1)):
            with self.subTest(field=field):
                candidate = events("candidate")
                setattr(candidate[2], field, value)
                with self.assertRaises(SparseTraceInvalid):
                    compare(events("reference"), candidate)

    def test_matching_wrong_dtype_is_not_an_exact_policy_match(self):
        reference, candidate = events("reference"), events("candidate")
        for stream in (reference, candidate):
            stream[3].value = stream[3].value.to(torch.float64)
            stream[3].dtype = "torch.float64"
        with self.assertRaises(SparseTraceInvalid):
            compare(reference, candidate)

    def test_nonfinite_values_in_either_run_are_invalid(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                candidate = events("candidate")
                candidate[2].value[0, 0] = value
                with self.assertRaises(SparseTraceInvalid):
                    compare(events("reference"), candidate)

    def test_named_and_negative_bit_views_are_rejected_predictably(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            values = [torch._neg_view(torch.ones(1, 8)),
                      torch.ones(1, 8).refine_names("batch", "hidden")]
        for value in values:
            with self.subTest(value=value), self.assertRaises(SparseTraceInvalid):
                candidate = events("candidate")
                candidate[1].value = value
                compare(events("reference"), candidate)

    def test_mixed_or_reused_run_ids_are_invalid_provenance(self):
        reference, candidate = events("same"), events("same")
        with self.assertRaises(SparseTraceInvalid):
            compare(reference, candidate)
        candidate = events("candidate")
        candidate[2].run_id = "another"
        with self.assertRaises(SparseTraceInvalid):
            compare(reference, candidate)

    def test_caller_must_verify_completeness_and_pairing(self):
        reference, candidate = events("reference"), events("candidate")
        for kwargs in ({}, {"reference_complete": True, "candidate_complete": True},
                       {"reference_complete": True, "pairing_verified": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(SparseTraceInvalid):
                compare_sparse_pair(reference, candidate, **kwargs)

    def test_target_identity_is_not_an_input_to_the_comparator(self):
        with self.assertRaises(TypeError):
            compare(events("reference"), events("candidate"), injection_site="block.2.activation")

    def test_extreme_float32_values_have_finite_binary64_norm(self):
        candidate = events("candidate")
        candidate[1].value[0, 0] = 3e38
        result = compare(events("reference"), candidate)
        self.assertTrue(math.isfinite(result["l2_by_block"][1]))
        self.assertEqual(result["l2_by_block"][1], float(candidate[1].value[0, 0]))

    def test_subnormal_equality_and_norm_are_stable_under_flush_modes(self):
        code = r'''
import math
import sys
import torch
from looking_glass.causal_trace_v2 import compare_sparse_pair
from looking_glass.probes import TraceEvent, TapPoint, TraceSession

def trace(run):
    return [TraceEvent(run, f"block.{i}.output", i, 0, (1,8),
                       "torch.float32", "cpu", torch.zeros(1,8,device="cpu")) for i in range(4)]

for flush in (False, True):
    if not torch.set_flush_denormal(flush):
        sys.exit(77)
    clean, altered = trace("clean"), trace("altered")
    altered[1].value.view(torch.int32)[0,0] = 1  # Smallest positive binary32 subnormal.
    point = TapPoint("point", "sparse")
    with torch.no_grad(), TraceSession(point, placement="sparse", max_tensor_bytes=32) as captured:
        point(altered[1].value)
    assert captured.events[0].value.view(torch.int32)[0,0].item() == 1, "capture changed bits"
    result = compare_sparse_pair(clean, altered, reference_complete=True,
                                 candidate_complete=True, pairing_verified=True)
    assert result["status"] == "difference_observed", (flush, result)
    assert result["l2_by_block"][1] == math.ldexp(1., -149), (flush, result)
    altered[1].value = torch.zeros(1, 16, device="cpu")[:, ::2]
    altered[1].value.view(torch.int32)[0,0] = 1  # Noncontiguous storage.
    result = compare_sparse_pair(clean, altered, reference_complete=True,
                                 candidate_complete=True, pairing_verified=True)
    assert result["status"] == "difference_observed", (flush, result)
    assert result["l2_by_block"][1] == math.ldexp(1., -149), (flush, result)
    altered[1].value.view(torch.int32)[0,0] = -2147483648  # Negative zero.
    result = compare_sparse_pair(clean, altered, reference_complete=True,
                                 candidate_complete=True, pairing_verified=True)
    assert result["status"] == "no_difference_observed", (flush, result)
'''
        process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if process.returncode == 77:
            self.skipTest("runtime cannot configure the CPU denormal mode")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
