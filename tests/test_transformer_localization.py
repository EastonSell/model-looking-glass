from dataclasses import replace
import json
import unittest

import torch

from looking_glass.capture_dispatcher import CaptureDispatcher, CaptureRequest, SampleSelection
from looking_glass.trace_schema import GenerationContext, RunIdentity, TraceEvent
from looking_glass.transformer_localization import (
    TensorObservation, compare_transformer_traces, observation_from_dispatch_slice,
)


def observation(*, run_id="run", sample_id="sample", sequence=0, forward=0,
                module="transformer.h.0.attn", layer=0, invocation=0,
                output_path=(0,), batch=(0,), tokens=(0,), batch_axis=0,
                token_axis=1, generation=None, value=None, capture_mode="direct",
                shape=(1, 2, 2), device="cpu", saved_shape=(1, 1, 2)):
    if value is None:
        value = torch.tensor([[[1., 2.]]])
    if generation is None:
        generation = GenerationContext(stage="forward", use_cache=False)
    metadata = TraceEvent(
        identity=RunIdentity(run_id=run_id, sample_id=sample_id),
        sequence=sequence, forward_index=forward, module_path=module,
        module_kind="attn", layer_index=layer, invocation=invocation,
        output_path=output_path, shape=shape, dtype="torch.float32", device=device,
        batch_positions=batch, token_positions=tokens, batch_axis=batch_axis,
        token_axis=token_axis, generation=generation, capture_mode=capture_mode,
        saved_shape=saved_shape, saved_dtype=str(value.dtype), saved_device=str(value.device),
    )
    return TensorObservation(metadata, value)


def compare(reference, candidate, **kwargs):
    candidate = [TensorObservation(
        replace(item.event, identity=replace(item.event.identity,
                                            run_id="candidate-" + item.event.identity.run_id)),
        item.value) for item in candidate]
    return compare_transformer_traces(reference, candidate,
        reference_complete=True, candidate_complete=True, pairing_verified=True,
        execution_order_verified=True,
        **kwargs)


def dispatcher_slice(run_id, value, *, generation_step=None):
    request = CaptureRequest(run_id=run_id, job_id="job", request_id="request",
                             samples=(SampleSelection(0, "sample", (1,)),),
                             forward_index=0, generation_step=generation_step,
                             max_events=4, max_trace_bytes=1000)
    dispatcher = CaptureDispatcher((request,), batch_size=1, sequence_length=2,
        max_events=4, max_trace_bytes=1000, sequence_starts={run_id: 0})
    _, events = dispatcher.run(lambda: dispatcher.record(
        "transformer.h.0.mlp", layer_index=0, module_kind="mlp",
        output=value, output_path=()))
    return events[0]


class TransformerLocalizationTests(unittest.TestCase):
    def test_observed_difference_retains_full_transformer_coordinates(self):
        reference = [observation()]
        candidate = [observation(run_id="changed", value=torch.tensor([[[1., 3.]]]))]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "difference_observed")
        coordinate = result["first_observed_interval"]["first_unequal"]
        self.assertEqual(coordinate["module_path"], "transformer.h.0.attn")
        self.assertEqual(coordinate["layer_index"], 0)
        self.assertEqual(coordinate["forward_index"], 0)
        self.assertEqual(coordinate["invocation"], 0)
        self.assertEqual(coordinate["output_path"], [0])
        self.assertEqual(coordinate["token_positions"], [0])
        self.assertEqual(coordinate["batch_positions"], [0])
        self.assertEqual(result["causal_origin"], "unknown")
        json.dumps(result, allow_nan=False)

    def test_different_token_positions_are_not_aligned(self):
        result = compare([observation(tokens=(0,))], [observation(tokens=(1,))])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["compared"], 0)

    def test_different_output_paths_are_not_aligned(self):
        result = compare([observation(output_path=(0,))],
                         [observation(output_path=(1,))])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["coverage"]["compared"], 0)

    def test_unknown_coordinate_fields_are_reported_not_equated(self):
        for field in ("layer", "output_path", "batch", "tokens", "batch_axis", "token_axis"):
            with self.subTest(field=field):
                item = observation(**{field: None})
                result = compare([item], [item])
                self.assertEqual(result["status"], "inconclusive")
                self.assertTrue(result["schema_coverage"]["reference_metadata_issues"])

    def test_unknown_generation_context_is_not_silently_equated(self):
        item = observation(generation=GenerationContext())
        result = compare([item], [item])
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(result["schema_coverage"]["reference_metadata_issues"])

    def test_known_generation_stage_with_unknown_cache_is_not_equated(self):
        item = observation(generation=GenerationContext(stage="decode", use_cache=True))
        result = compare([item], [item])
        self.assertEqual(result["status"], "inconclusive")

    def test_different_explicit_samples_are_not_aligned(self):
        result = compare([observation(sample_id="one")], [observation(sample_id="two")])
        self.assertEqual(result["status"], "inconclusive")

    def test_original_shape_mismatch_is_not_hidden_by_equal_saved_shapes(self):
        result = compare([observation(shape=(1, 2, 2))],
                         [observation(shape=(1, 3, 2))])
        self.assertEqual(result["status"], "inconclusive")

    def test_selection_must_agree_with_saved_shape(self):
        item = observation(saved_shape=(1, 2, 2))
        result = compare([item], [item])
        self.assertEqual(result["status"], "inconclusive")

    def test_dispatcher_slice_has_an_explicit_rank_restoring_adapter(self):
        reference = dispatcher_slice("reference", torch.arange(6.).reshape(1, 2, 3))
        candidate = dispatcher_slice("candidate", torch.arange(6.).reshape(1, 2, 3))
        context = GenerationContext(stage="forward", use_cache=False)
        left = observation_from_dispatch_slice(reference, generation=context)
        right = observation_from_dispatch_slice(candidate, generation=context)
        result = compare_transformer_traces([left], [right], reference_complete=True,
            candidate_complete=True, pairing_verified=True, execution_order_verified=True)
        self.assertEqual(result["status"], "no_difference_observed")
        saved = result["comparisons"][0]["reference_saved"]
        self.assertEqual(saved["shape"], [1, 3])
        self.assertEqual(saved["comparison_shape"], [1, 1, 3])
        self.assertEqual(saved["comparison_transform"], "restore_singleton_batch_axis")
        self.assertEqual(left.value.data_ptr(), reference.value.data_ptr())

    def test_dispatcher_context_cannot_contradict_recorded_generation_step(self):
        captured = dispatcher_slice("reference", torch.zeros(1, 2, 3), generation_step=4)
        context = GenerationContext(stage="decode", generation_step=5, cache_position=5,
            cache_length_before=5, cache_length_after=6, use_cache=True)
        with self.assertRaises(ValueError):
            observation_from_dispatch_slice(captured, generation=context)

    def test_squeezed_shape_is_not_guessed_from_arbitrary_schema_metadata(self):
        captured = dispatcher_slice("reference", torch.zeros(1, 2, 3))
        event = replace(captured.to_trace_event(),
                        generation=GenerationContext(stage="forward", use_cache=False))
        raw = TensorObservation(event, captured.value)
        result = compare([raw], [raw])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["schema_coverage"]["reference_metadata_issues"][0]["fields"],
                         ["saved_shape"])

    def test_dispatcher_adapter_does_not_infer_generation_context(self):
        reference = observation_from_dispatch_slice(
            dispatcher_slice("reference", torch.zeros(1, 2, 3)))
        candidate = observation_from_dispatch_slice(
            dispatcher_slice("candidate", torch.zeros(1, 2, 3)))
        result = compare_transformer_traces([reference], [candidate], reference_complete=True,
            candidate_complete=True, pairing_verified=True, execution_order_verified=True)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("generation", result["schema_coverage"]["reference_metadata_issues"][0]["fields"])

    def test_dispatcher_adapter_detects_an_observed_value_difference(self):
        first = torch.zeros(1, 2, 3)
        second = first.clone()
        second[0, 1, 0] = .125
        context = GenerationContext(stage="forward", use_cache=False)
        reference = observation_from_dispatch_slice(dispatcher_slice("reference", first), generation=context)
        candidate = observation_from_dispatch_slice(dispatcher_slice("candidate", second), generation=context)
        result = compare_transformer_traces([reference], [candidate], reference_complete=True,
            candidate_complete=True, pairing_verified=True, execution_order_verified=True)
        self.assertEqual(result["status"], "difference_observed")
        self.assertEqual(result["first_observed_interval"]["first_unequal"]["module_path"],
                         "transformer.h.0.mlp")
        self.assertEqual(result["causal_origin"], "unknown")

    def test_saved_device_can_differ_from_original_device(self):
        item = observation(device="cuda:0")
        result = compare([item], [item])
        self.assertEqual(result["status"], "no_difference_observed")

    def test_report_discloses_saved_precision_separately_from_source_precision(self):
        item = observation(value=torch.tensor([[[1., 2.]]], dtype=torch.float16))
        result = compare([item], [item])
        self.assertEqual(result["status"], "no_difference_observed")
        self.assertEqual(result["comparisons"][0]["coordinate"]["source_dtype"], "torch.float32")
        self.assertEqual(result["comparisons"][0]["reference_saved"],
                         {"shape": [1, 1, 2], "dtype": "torch.float16", "device": "cpu"})
        self.assertEqual(result["comparisons"][0]["candidate_saved"]["dtype"], "torch.float16")

    def test_different_report_channels_can_compare_the_same_coordinate(self):
        result = compare([observation(capture_mode="direct")],
                         [observation(capture_mode="gradient")])
        self.assertEqual(result["status"], "no_difference_observed")

    def test_caller_must_verify_pairing_and_complete_capture(self):
        item = observation()
        self.assertEqual(compare_transformer_traces([item], [item])["status"],
                         "inconclusive")

    def test_same_run_identity_cannot_represent_two_distinct_events(self):
        reference = [observation(run_id="same")]
        candidate = [observation(run_id="same", value=torch.tensor([[[1., 3.]]]))]
        result = compare_transformer_traces(reference, candidate,
            reference_complete=True, candidate_complete=True, pairing_verified=True,
            execution_order_verified=True)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["schema_coverage"]["run_ids_disjoint"])

    def test_completion_and_pairing_attestations_must_be_actual_booleans(self):
        item = observation()
        for name in ("reference_complete", "candidate_complete", "pairing_verified",
                     "execution_order_verified"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                compare_transformer_traces([item], [item], **{name: 1})

    def test_different_forwards_and_invocations_are_not_aligned(self):
        for field in ("forward", "invocation"):
            with self.subTest(field=field):
                result = compare([observation()], [observation(**{field: 1})])
                self.assertEqual(result["status"], "inconclusive")

    def test_invocations_count_module_calls_not_separate_selected_slices(self):
        events = [observation(sequence=0, invocation=0, tokens=(0,)),
                  observation(sequence=1, invocation=1, tokens=(1,))]
        self.assertEqual(compare(events, events)["status"], "no_difference_observed")

    def test_multiple_slices_from_same_invocation_are_distinct_events(self):
        events = [observation(sequence=0, invocation=0, tokens=(0,)),
                  observation(sequence=1, invocation=0, tokens=(1,))]
        self.assertEqual(compare(events, events)["status"], "no_difference_observed")

    def test_contradictory_source_metadata_for_one_output_is_inconclusive(self):
        events = [observation(sequence=0, invocation=0, tokens=(0,), shape=(1, 2, 2)),
                  observation(sequence=1, invocation=0, tokens=(1,), shape=(1, 3, 2))]
        result = compare(events, events)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["schema_coverage"]["reference_source_metadata_consistent"])

    def test_one_module_path_cannot_switch_layer_identity_within_a_run(self):
        events = [observation(sequence=0, invocation=0, layer=0),
                  observation(sequence=1, invocation=1, layer=1)]
        result = compare(events, events)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["schema_coverage"]["reference_source_metadata_consistent"])

    def test_one_source_tensor_position_cannot_claim_two_sample_ids(self):
        events = [observation(sequence=0, tokens=(0,), sample_id="first"),
                  observation(sequence=1, tokens=(1,), sample_id="second")]
        result = compare(events, events)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["schema_coverage"]["reference_source_metadata_consistent"])

    def test_duplicate_full_event_coordinate_is_ambiguous(self):
        events = [observation(sequence=0), observation(sequence=1)]
        result = compare(events, events)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(len(result["coverage"]["ambiguous_reference"]), 1)

    def test_module_invocation_gap_is_incomplete(self):
        events = [observation(sequence=0, invocation=0),
                  observation(sequence=1, invocation=2)]
        result = compare(events, events)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["schema_coverage"]["reference_invocations_complete"])

    def test_execution_order_needs_explicit_attestation(self):
        reference = [observation()]
        candidate = [observation(value=torch.tensor([[[1., 3.]]]))]
        result = compare_transformer_traces(reference, candidate,
            reference_complete=True, candidate_complete=True, pairing_verified=True)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["first_observed_interval"])
        self.assertIsNotNone(result["first_compared_difference"])

    def test_invalid_forward_index_order_is_not_a_complete_trace(self):
        for indices in ((1, 0), (0, 2)):
            with self.subTest(indices=indices):
                events = [observation(sequence=0, forward=indices[0]),
                          observation(sequence=1, forward=indices[1])]
                result = compare(events, events)
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["schema_coverage"]["reference_forwards_complete"])

    def test_first_difference_in_a_new_forward_has_no_previous_equal_boundary(self):
        reference = [observation(sequence=0, forward=0), observation(sequence=1, forward=1)]
        candidate = [observation(sequence=0, forward=0),
                     observation(sequence=1, forward=1, value=torch.tensor([[[1., 3.]]]))]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "difference_observed")
        self.assertIsNone(result["first_observed_interval"]["last_equal"])
        self.assertEqual(result["first_observed_interval"]["first_unequal"]["forward_index"], 1)

    def test_previous_sample_slice_is_not_a_boundary_for_a_different_sample(self):
        reference = [observation(sequence=0, sample_id="first", module="a"),
                     observation(sequence=1, sample_id="second", module="b")]
        candidate = [observation(sequence=0, sample_id="first", module="a"),
                     observation(sequence=1, sample_id="second", module="b",
                                 value=torch.tensor([[[1., 3.]]]))]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "difference_observed")
        self.assertIsNone(result["first_observed_interval"]["last_equal"])

    def test_previous_generation_context_is_not_a_boundary_for_a_different_context(self):
        generation = GenerationContext(stage="decode", generation_step=0, cache_position=0,
                                       cache_length_before=0, cache_length_after=1, use_cache=True)
        reference = [observation(sequence=0, module="a"),
                     observation(sequence=1, module="b", generation=generation)]
        candidate = [observation(sequence=0, module="a"),
                     observation(sequence=1, module="b", generation=generation,
                                 value=torch.tensor([[[1., 3.]]]))]
        result = compare(reference, candidate)
        self.assertEqual(result["status"], "difference_observed")
        self.assertIsNone(result["first_observed_interval"]["last_equal"])

    def test_token_selection_order_is_part_of_the_coordinate(self):
        values = torch.tensor([[[1., 2.], [1., 2.]]])
        first = observation(tokens=(0, 1), value=values, saved_shape=(1, 2, 2))
        second = observation(tokens=(1, 0), value=values, saved_shape=(1, 2, 2))
        self.assertEqual(compare([first], [second])["status"], "inconclusive")

    def test_reordered_or_missing_earlier_schema_events_preclude_a_first_interval(self):
        first = observation(sequence=0, module="a")
        second = observation(sequence=1, module="b")
        altered = observation(sequence=1, module="b", value=torch.tensor([[[4., 2.]]]))
        other = observation(sequence=0, module="other")
        result = compare([first, second], [other, altered])
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["first_observed_interval"])


if __name__ == "__main__":
    unittest.main()
