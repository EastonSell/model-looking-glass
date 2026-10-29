import math
import json
import unittest
from types import SimpleNamespace

import torch

from looking_glass.localization import compare_traces
from looking_glass.model import TinyResidual
from looking_glass.probes import TraceEvent, TraceSession


def event(node, sequence, occurrence, values, *, dtype=torch.float32):
    tensor = None if values is None else torch.tensor(values, dtype=dtype)
    shape = (1,) if tensor is None else tuple(tensor.shape)
    return TraceEvent("run", node, sequence, occurrence, shape, str(dtype),
                      "cpu", tensor)


def compare(reference, candidate, **kwargs):
    return compare_traces(reference, candidate, reference_complete=True,
                          candidate_complete=True, **kwargs)


class LocalizationTests(unittest.TestCase):
    def test_real_probe_stream_reports_the_observed_injected_difference(self):
        torch.manual_seed(7)
        model = TinyResidual(width=4, depth=2).eval()
        x = torch.randn(2, 4)
        with torch.no_grad(), TraceSession(model) as reference:
            model(x)
        with torch.no_grad(), TraceSession(model) as candidate:
            model(x, perturb=("block.0.activation", 0.125))
        result = compare(reference.events, candidate.events)
        self.assertEqual(result["status"], "difference_observed")
        self.assertEqual(result["first_observed_interval"], {
            "last_equal": {"node_id": "block.0.branch.0", "occurrence": 0},
            "first_unequal": {"node_id": "block.0.activation", "occurrence": 0},
        })
        self.assertEqual(result["causal_origin"], "unknown")
        json.dumps(result, allow_nan=False)

    def test_repeated_node_occurrences_are_distinct(self):
        reference = [event("shared", 0, 0, [1.]), event("shared", 1, 1, [2.])]
        candidate = [event("shared", 0, 0, [1.]), event("shared", 1, 1, [3.])]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "difference_observed")
        self.assertEqual(result["first_observed_interval"], {
            "last_equal": {"node_id": "shared", "occurrence": 0},
            "first_unequal": {"node_id": "shared", "occurrence": 1},
        })

    def test_unmatched_branch_before_a_difference_leaves_first_interval_unknown(self):
        reference = [event("entry", 0, 0, [0.]), event("branch0", 1, 0, [1.]),
                     event("merge", 2, 0, [2.])]
        candidate = [event("entry", 0, 0, [0.]), event("branch1", 1, 0, [1.]),
                     event("merge", 2, 0, [3.])]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["first_compared_difference"],
                         {"node_id": "merge", "occurrence": 0})
        self.assertIsNone(result["first_observed_interval"])
        self.assertEqual(result["coverage"]["reference_only"],
                         [{"node_id": "branch0", "occurrence": 0}])
        self.assertEqual(result["coverage"]["candidate_only"],
                         [{"node_id": "branch1", "occurrence": 0}])
        self.assertFalse(result["coverage"]["matching_order"])

    def test_different_branches_with_equal_values_are_not_a_negative_result(self):
        result = compare([event("branch0", 0, 0, [1.])],
                         [event("branch1", 0, 0, [1.])])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["compared"], 0)

    def test_reordered_events_do_not_support_a_first_interval(self):
        reference = [event("a", 0, 0, [1.]), event("b", 1, 0, [1.]),
                     event("c", 2, 0, [1.])]
        candidate = [event("a", 0, 0, [1.]), event("c", 1, 0, [1.]),
                     event("b", 2, 0, [2.])]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["coverage"]["matching_order"])
        self.assertIsNone(result["first_observed_interval"])

    def test_missing_values_make_no_difference_inconclusive(self):
        result = compare([event("a", 0, 0, None)], [event("a", 0, 0, None)])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["unavailable"], 1)

    def test_missing_value_before_difference_precludes_first_interval(self):
        reference = [event("a", 0, 0, None), event("b", 1, 0, [0.])]
        candidate = [event("a", 0, 0, None), event("b", 1, 0, [1.])]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["first_observed_interval"])

    def test_shape_mismatch_is_not_broadcast_or_reported_as_difference(self):
        result = compare([event("a", 0, 0, [1.])],
                         [event("a", 0, 0, [1., 2.])])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["incompatible"], 1)

    def test_dtype_mismatch_is_not_accepted_as_equal(self):
        result = compare([event("a", 0, 0, [1.], dtype=torch.float32)],
                         [event("a", 0, 0, [1.], dtype=torch.float64)])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["incompatible"], 1)

    def test_tolerance_uses_reference_magnitude(self):
        small = [event("a", 0, 0, [100.])]
        large = [event("a", 0, 0, [111.])]
        self.assertEqual(compare(small, large, rtol=0.1)["status"],
                         "difference_observed")
        self.assertEqual(compare(large, small, rtol=0.1)["status"],
                         "no_difference_observed")

    def test_finite_extreme_values_do_not_overflow_the_tolerance_comparison(self):
        positive = [event("a", 0, 0, [1e308], dtype=torch.float64)]
        negative = [event("a", 0, 0, [-1e308], dtype=torch.float64)]
        self.assertEqual(compare(positive, negative, rtol=1.5)["status"],
                         "difference_observed")
        self.assertEqual(compare(positive, negative, rtol=2.5)["status"],
                         "no_difference_observed")

    def test_nearby_float64_values_do_not_disappear_during_scaling(self):
        reference = [event("a", 0, 0, [1.0600853094673223e-14], dtype=torch.float64)]
        candidate = [event("a", 0, 0, [1.0600853094673224e-14], dtype=torch.float64)]
        result = compare(reference, candidate, atol=1.262177448353619e-30)
        self.assertEqual(result["status"], "difference_observed")

    def test_invalid_tolerances_are_rejected(self):
        events = [event("a", 0, 0, [1.])]
        for invalid in (-1, math.nan, math.inf, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compare(events, events, rtol=invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compare(events, events, atol=invalid)

    def test_nonfinite_values_are_unavailable_not_equal_or_different(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                result = compare([event("a", 0, 0, [value])],
                                 [event("a", 0, 0, [value])])
                self.assertEqual(result["status"], "inconclusive")
                self.assertEqual(result["coverage"]["unavailable"], 1)

    def test_missing_identity_is_reported_as_invalid_metadata(self):
        missing = SimpleNamespace(node_id="a", sequence=0, value=torch.tensor([1.]))
        result = compare([missing], [event("a", 0, 0, [1.])])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(len(result["coverage"]["invalid_reference"]), 1)

    def test_duplicate_identity_is_ambiguous_not_aligned(self):
        reference = [event("a", 0, 0, [1.]), event("a", 1, 0, [2.])]
        candidate = [event("a", 0, 0, [1.]), event("a", 1, 1, [2.])]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["ambiguous_reference"],
                         [{"node_id": "a", "occurrence": 0}])

    def test_empty_and_unconfirmed_traces_cannot_establish_no_difference(self):
        self.assertEqual(compare([], [])["status"], "inconclusive")
        events = [event("a", 0, 0, [1.])]
        self.assertEqual(compare_traces(events, events)["status"], "inconclusive")

    def test_sequence_gap_is_not_silently_treated_as_complete(self):
        for sequence in ((0, 2), (0, 0), (1, 0), (-1, 0)):
            with self.subTest(sequence=sequence):
                events = [event("a", sequence[0], 0, [1.]),
                          event("b", sequence[1], 0, [2.])]
                result = compare(events, events)
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["coverage"]["reference_sequence_complete"])

    def test_invocation_gaps_are_not_silently_treated_as_complete(self):
        for events in ([event("a", 0, 1, [1.])],
                       [event("a", 0, 0, [1.]), event("a", 1, 2, [2.])]):
            with self.subTest(events=events):
                result = compare(events, events)
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["coverage"]["reference_invocations_complete"])

    def test_empty_tensors_are_not_value_evidence(self):
        events = [event("a", 0, 0, [])]
        result = compare(events, events)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["unavailable"], 1)

    def test_mixed_run_ids_cannot_establish_observed_equality(self):
        reference = [event("a", 0, 0, [1.]), event("b", 1, 0, [2.])]
        candidate = [event("a", 0, 0, [1.]), event("b", 1, 0, [2.])]
        reference[0].run_id = "run-a"
        reference[1].run_id = "run-b"
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["coverage"]["reference_run_identity_consistent"])

    def test_missing_run_id_is_unknown_provenance(self):
        reference = [event("a", 0, 0, [1.])]
        reference[0].run_id = ""
        result = compare(reference, [event("a", 0, 0, [1.])])
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["coverage"]["reference_run_identity_consistent"])

    def test_different_single_run_ids_are_valid_across_streams(self):
        reference = [event("a", 0, 0, [1.])]
        candidate = [event("a", 0, 0, [1.])]
        reference[0].run_id = "reference"
        candidate[0].run_id = "candidate"
        self.assertEqual(compare(reference, candidate)["status"],
                         "no_difference_observed")

    def test_sparse_and_meta_tensors_are_reported_unavailable(self):
        tensors = [torch.sparse_coo_tensor([[0]], [1.], (1,)),
                   torch.empty(1, device="meta")]
        for tensor in tensors:
            with self.subTest(device=tensor.device, layout=tensor.layout):
                unavailable = TraceEvent("run", "a", 0, 0, tuple(tensor.shape),
                                         str(tensor.dtype), str(tensor.device), tensor)
                result = compare([unavailable], [unavailable])
                self.assertEqual(result["status"], "inconclusive")
                self.assertEqual(result["coverage"]["unavailable"], 1)

    def test_unsupported_float8_tensors_are_reported_unavailable(self):
        for name in ("float8_e4m3fn", "float8_e5m2"):
            dtype = getattr(torch, name, None)
            if dtype is not None:
                with self.subTest(dtype=name):
                    item = event("a", 0, 0, [1.], dtype=dtype)
                    result = compare([item], [item])
                    self.assertEqual(result["status"], "inconclusive")
                    self.assertEqual(result["coverage"]["unavailable"], 1)

    def test_declared_shape_must_match_the_saved_value(self):
        malformed = event("a", 0, 0, [1.])
        malformed.shape = (2,)
        result = compare([malformed], [event("a", 0, 0, [1.])])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["unavailable"], 1)

    def test_declared_dtype_must_match_the_saved_value(self):
        malformed = event("a", 0, 0, [1.])
        malformed.dtype = "torch.float64"
        result = compare([malformed], [event("a", 0, 0, [1.])])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
