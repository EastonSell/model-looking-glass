"""Compare selected transformer observations with explicit source coordinates.

Metadata and tensor payloads remain separate. Matching labels and values alone
do not establish that two experiments used the same model, data, or cache state.
"""
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import torch

from .localization import compare_traces
from .trace_schema import GenerationContext, TraceEvent
from .encoder_forward import check_encoder_pair
from .t5_forward import check_t5_pair


@dataclass(frozen=True)
class TensorObservation:
    """Schema metadata plus a retained tensor, or None for missing values."""
    event: TraceEvent
    value: torch.Tensor | None
    original_saved_shape: tuple[int, ...] | None = None
    comparison_transform: str | None = None

    def __post_init__(self):
        if not isinstance(self.event, TraceEvent):
            raise TypeError("event must be a trace_schema.TraceEvent")
        if (self.comparison_transform is None) != (self.original_saved_shape is None):
            raise ValueError("a comparison transform and original shape must be recorded together")


def observation_from_dispatch_slice(captured, *, generation=None):
    """Restore the known singleton batch axis of a CaptureDispatcher slice.

    The producer's original saved shape and the no-copy view transformation
    are retained in the comparison report. Any generation context is supplied
    by the caller and must agree with a generation step already recorded.
    """
    from .capture_dispatcher import CapturedSlice

    if type(captured) is not CapturedSlice:
        raise TypeError("captured must be a CaptureDispatcher CapturedSlice")
    event = captured.to_trace_event()
    if (event.batch_axis != 0 or event.token_axis != 1 or
            event.batch_positions is None or len(event.batch_positions) != 1 or
            event.token_positions is None or len(event.shape) < 2):
        raise ValueError("captured slice lacks the documented batch/token layout")
    expected = (len(event.token_positions),) + event.shape[2:]
    if event.saved_shape != expected:
        raise ValueError("captured slice shape does not match its declared selection")
    if generation is not None:
        if not isinstance(generation, GenerationContext):
            raise TypeError("generation must be a GenerationContext")
        if (event.generation.generation_step is not None and
                event.generation.generation_step != generation.generation_step):
            raise ValueError("generation context contradicts the recorded generation step")
    else:
        generation = event.generation
    value = captured.value.unsqueeze(0)
    normalized = replace(event, generation=generation, saved_shape=tuple(value.shape))
    return TensorObservation(normalized, value, original_saved_shape=event.saved_shape,
                             comparison_transform="restore_singleton_batch_axis")


def _generation_known(generation):
    # A forward with no returned cache can have unknown cache positions here;
    # the caller must separately verify any input-cache state when attesting
    # pairing. use_cache=False alone does not prove the absence of past inputs.
    if generation.stage == "forward" and generation.use_cache is False:
        return True
    # Cached generation needs explicit producer-supplied state. None means
    # unknown, not a wildcard or an inferred zero.
    if generation.stage in ("prefill", "decode") and generation.use_cache is True:
        return all(getattr(generation, name) is not None for name in
                   ("generation_step", "cache_position", "cache_length_before",
                    "cache_length_after"))
    return False


def _proxy(observation, index, *, verified_encoder_forward=False):
    if not isinstance(observation, TensorObservation):
        raise TypeError("each observation must be a TensorObservation")
    event = observation.event
    missing = [name for name in ("module_kind", "layer_index", "output_path",
                                 "batch_positions", "token_positions", "batch_axis",
                                 "token_axis") if getattr(event, name) is None]
    if not verified_encoder_forward and not _generation_known(event.generation):
        missing.append("generation")
    issues = []
    if missing:
        issues.append({"index": index, "fields": missing, "reason": "unknown metadata"})
    # For retained slices, metadata must be consistent with the explicitly
    # selected original tensor axes. A known empty selection is still distinct
    # from an unknown selection, and carries no value evidence.
    if (not missing and event.saved_shape is not None):
        expected = list(event.shape)
        expected[event.batch_axis] = len(event.batch_positions)
        expected[event.token_axis] = len(event.token_positions)
        if tuple(expected) != event.saved_shape:
            issues.append({"index": index, "fields": ["saved_shape"],
                           "reason": "saved shape conflicts with selection"})

    coordinate = {
        "forward_index": event.forward_index,
        "module_path": event.module_path,
        "module_kind": event.module_kind,
        "layer_index": event.layer_index,
        "invocation": event.invocation,
        "output_path": event.output_path,
        "batch_positions": event.batch_positions,
        "token_positions": event.token_positions,
        "batch_axis": event.batch_axis,
        "token_axis": event.token_axis,
        "generation": event.generation.to_dict(),
        "sample_id": event.identity.sample_id,
        "source_shape": event.shape,
        "source_dtype": event.dtype,
        "source_device": event.device,
    }
    # The module-call ordinal is part of the full coordinate. The original
    # comparator rejects duplicate full coordinates; separate validation below
    # checks call ordinals per module rather than per selected tensor slice.
    node = "" if issues else json.dumps(coordinate, sort_keys=True,
                                         ensure_ascii=True, separators=(",", ":"))
    proxy = SimpleNamespace(run_id=event.identity.run_id, node_id=node,
                            sequence=event.sequence, occurrence=0,
                            shape=event.saved_shape, dtype=event.saved_dtype,
                            device=event.saved_device, value=observation.value)
    return proxy, issues


def _coordinate(encoded):
    if encoded is None:
        return None
    return json.loads(encoded["node_id"])


def _saved(observation):
    event = observation.event
    if observation.comparison_transform is not None:
        return {"shape": list(observation.original_saved_shape), "dtype": event.saved_dtype,
                "device": event.saved_device, "comparison_shape": list(event.saved_shape),
                "comparison_transform": observation.comparison_transform}
    return {"shape": None if event.saved_shape is None else list(event.saved_shape),
            "dtype": event.saved_dtype, "device": event.saved_device}


def _forwards_complete(observations):
    if not observations:
        return False
    current = -1
    for observation in observations:
        forward = observation.event.forward_index
        if forward == current:
            continue
        if forward != current + 1:
            return False
        current = forward
    return True


def _invocations_complete(observations):
    latest = {}
    for observation in observations:
        event = observation.event
        key = (event.forward_index, event.module_path)
        previous = latest.get(key, -1)
        if event.invocation not in (previous, previous + 1):
            return False
        latest[key] = event.invocation
    return True


def _source_metadata_consistent(observations):
    outputs = {}
    modules = {}
    samples = {}
    for observation in observations:
        event = observation.event
        module_identity = (event.module_kind, event.layer_index)
        if event.module_path in modules and modules[event.module_path] != module_identity:
            return False
        modules[event.module_path] = module_identity
        key = (event.forward_index, event.module_path, event.invocation, event.output_path)
        metadata = (event.module_kind, event.layer_index, event.shape, event.dtype,
                    event.device, event.batch_axis, event.token_axis, event.generation)
        if key in outputs and outputs[key] != metadata:
            return False
        outputs[key] = metadata
        if event.identity.sample_id is not None and event.batch_positions is not None:
            for position in event.batch_positions:
                sample_key = key + (position,)
                if sample_key in samples and samples[sample_key] != event.identity.sample_id:
                    return False
                samples[sample_key] = event.identity.sample_id
    return True


def compare_transformer_traces(reference, candidate, *, rtol=0.0, atol=0.0,
                               reference_complete=False, candidate_complete=False,
                               pairing_verified=False, execution_order_verified=False,
                               encoder_contexts=None, expected_input_differences=None,
                               t5_contexts=None, expected_t5_encoder_differences=None,
                               expected_t5_decoder_differences=None):
    """Compare schema-backed slices, conditional on external pairing evidence.

    Pairing means the caller has verified the model, inputs, tap policy and
    relevant runtime/cache conditions under its protocol, allowing only the
    intended experimental differences. Completion covers emitted events under
    that policy; it does not imply exhaustive model coverage. The optional
    encoder contexts provide a narrow, separately validated alternative to
    generation metadata; they do not prove model identity or actual execution.
    T5 traces require both stack-specific contexts and explicit difference
    coordinates; that contract does not establish semantic token alignment.
    """
    if any(type(value) is not bool for value in
           (reference_complete, candidate_complete, pairing_verified, execution_order_verified)):
        raise ValueError("completion, pairing and order attestations must be booleans")
    reference = list(reference)
    candidate = list(candidate)
    encoder_requested = encoder_contexts is not None or expected_input_differences is not None
    encoder_report = (check_encoder_pair(encoder_contexts, reference, candidate,
                                         expected_input_differences)
                      if encoder_requested else None)
    encoder_ok = not encoder_requested or encoder_report["verified"]
    verified_encoder_forward = bool(encoder_report and encoder_report["verified"])
    # A normalized T5 forward must not silently use the generic generation
    # metadata path to bypass the two-stack input/mask pairing contract.
    def is_t5(observation):
        event = getattr(observation, "event", None)
        return (isinstance(event, TraceEvent) and
                ((event.module_kind or "").startswith("t5_") or
                 event.module_path.startswith(("encoder.block.", "decoder.block."))))

    t5_requested = (t5_contexts is not None or expected_t5_encoder_differences is not None
                    or expected_t5_decoder_differences is not None
                    or any(is_t5(observation) for observation in reference + candidate))
    t5_report = (check_t5_pair(t5_contexts, reference, candidate,
                               expected_t5_encoder_differences, expected_t5_decoder_differences)
                 if t5_requested else None)
    if encoder_requested and t5_requested:
        t5_report["verified"] = False
        t5_report["issues"].append("encoder and T5 comparison contracts cannot be combined")
    t5_ok = not t5_requested or t5_report["verified"]
    left, right, left_issues, right_issues = [], [], [], []
    for index, observation in enumerate(reference):
        item, issues = _proxy(observation, index, verified_encoder_forward=verified_encoder_forward)
        left.append(item)
        left_issues.extend(issues)
    for index, observation in enumerate(candidate):
        item, issues = _proxy(observation, index, verified_encoder_forward=verified_encoder_forward)
        right.append(item)
        right_issues.extend(issues)
    reference_forwards_complete = _forwards_complete(reference)
    candidate_forwards_complete = _forwards_complete(candidate)
    reference_invocations_complete = _invocations_complete(reference)
    candidate_invocations_complete = _invocations_complete(candidate)
    reference_source_metadata_consistent = _source_metadata_consistent(reference)
    candidate_source_metadata_consistent = _source_metadata_consistent(candidate)
    reference_runs = {observation.event.identity.run_id for observation in reference}
    candidate_runs = {observation.event.identity.run_id for observation in candidate}
    run_ids_disjoint = bool(reference_runs and candidate_runs and reference_runs.isdisjoint(candidate_runs))
    # Without external evidence the metadata cannot prove paired inputs/model;
    # enforce a conservative result even if every supplied tensor is equal.
    result = compare_traces(
        left, right, rtol=rtol, atol=atol,
        reference_complete=(reference_complete and pairing_verified and execution_order_verified and encoder_ok and t5_ok
                            and reference_forwards_complete and reference_invocations_complete
                            and reference_source_metadata_consistent and run_ids_disjoint),
        candidate_complete=(candidate_complete and pairing_verified and execution_order_verified and encoder_ok and t5_ok
                            and candidate_forwards_complete and candidate_invocations_complete
                            and candidate_source_metadata_consistent and run_ids_disjoint),
    )
    result["coverage"]["reference_complete"] = reference_complete
    result["coverage"]["candidate_complete"] = candidate_complete
    result["schema_coverage"] = {
        "pairing_verified": pairing_verified,
        "execution_order_verified": execution_order_verified,
        "run_ids_disjoint": run_ids_disjoint,
        "reference_forwards_complete": reference_forwards_complete,
        "candidate_forwards_complete": candidate_forwards_complete,
        "reference_invocations_complete": reference_invocations_complete,
        "candidate_invocations_complete": candidate_invocations_complete,
        "reference_source_metadata_consistent": reference_source_metadata_consistent,
        "candidate_source_metadata_consistent": candidate_source_metadata_consistent,
        "reference_metadata_issues": left_issues,
        "candidate_metadata_issues": right_issues,
    }
    if encoder_requested:
        result["schema_coverage"]["encoder_forward_context"] = encoder_report
    if t5_requested:
        result["schema_coverage"]["t5_forward_context"] = t5_report
    result["first_compared_difference"] = _coordinate(result["first_compared_difference"])
    interval = result["first_observed_interval"]
    if interval is not None:
        interval["last_equal"] = _coordinate(interval["last_equal"])
        interval["first_unequal"] = _coordinate(interval["first_unequal"])
    for name in ("reference_only", "candidate_only", "ambiguous_reference", "ambiguous_candidate"):
        result["coverage"][name] = [_coordinate(item) for item in result["coverage"][name]]
    left_by_node = {item.node_id: observation for item, observation in zip(left, reference) if item.node_id}
    right_by_node = {item.node_id: observation for item, observation in zip(right, candidate) if item.node_id}
    for item in result["comparisons"]:
        node = item["coordinate"]["node_id"]
        item["reference_saved"] = _saved(left_by_node[node])
        item["candidate_saved"] = _saved(right_by_node[node])
        item["coordinate"] = _coordinate(item["coordinate"])
    if interval is not None:
        # A previous event for another forward or selected slice is not a
        # meaningful preceding boundary for the changed slice.
        changed = interval["first_unequal"]
        context_fields = ("forward_index", "sample_id", "batch_positions",
                          "token_positions", "batch_axis", "token_axis", "generation")
        preceding = []
        for item in result["comparisons"]:
            if item["result"] == "different":
                break
            if item["result"] == "equal" and all(
                    item["coordinate"][name] == changed[name] for name in context_fields) and (
                    not t5_ok or not t5_requested or
                    item["coordinate"]["module_path"].split(".", 1)[0] ==
                    changed["module_path"].split(".", 1)[0]):
                preceding.append(item["coordinate"])
        interval["last_equal"] = preceding[-1] if preceding else None
    return result
