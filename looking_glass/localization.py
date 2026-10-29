"""Conservative comparison of the original labeled observation streams.

This module does not infer causal origins or model behavior from activations.
Its coordinates are the original probes.TraceEvent node labels and per-node
occurrence ordinals, not the richer transformer trace metadata schema.
"""
from collections import Counter, defaultdict
import math
from numbers import Real

import torch


def _tolerance(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite nonnegative real") from None
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return converted


def _coordinate(event):
    node = getattr(event, "node_id", None)
    occurrence = getattr(event, "occurrence", None)
    if not isinstance(node, str) or not node.strip():
        return None, "missing or invalid node_id"
    if type(occurrence) is not int or occurrence < 0:
        return None, "missing or invalid occurrence"
    return (node, occurrence), None


def _display(key):
    return {"node_id": key[0], "occurrence": key[1]}


def _inspect(events):
    records = []
    invalid = []
    sequence_complete = True
    invocations_complete = True
    counts = defaultdict(int)
    run_ids = set()
    missing_run_id = False
    for index, event in enumerate(events):
        key, error = _coordinate(event)
        run_id = getattr(event, "run_id", None)
        if isinstance(run_id, str) and run_id.strip():
            run_ids.add(run_id)
        else:
            missing_run_id = True
            error = "; ".join(part for part in (error, "missing or invalid run_id") if part)
        if type(getattr(event, "sequence", None)) is not int or event.sequence != index:
            sequence_complete = False
        if error:
            invalid.append({"index": index, "reason": error})
            invocations_complete = False
        else:
            node, occurrence = key
            if occurrence != counts[node]:
                invocations_complete = False
            counts[node] += 1
        records.append((event, key))
    frequencies = Counter(key for _, key in records if key is not None)
    ambiguous = {key for key, count in frequencies.items() if count > 1}
    unique = {key: (index, event) for index, (event, key) in enumerate(records)
              if key is not None and key not in ambiguous}
    return {"records": records, "invalid": invalid,
            "sequence_complete": sequence_complete,
            "invocations_complete": invocations_complete,
            "run_identity_consistent": not missing_run_id and len(run_ids) == 1,
            "ambiguous": ambiguous, "unique": unique}


def _payload(event):
    value = getattr(event, "value", None)
    if not isinstance(value, torch.Tensor):
        return None, "value is missing or is not a tensor"
    if value.layout != torch.strided or value.is_nested or value.device.type not in ("cpu", "cuda"):
        return None, "only dense strided CPU or CUDA values are supported"
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return None, "value dtype is not a supported floating type"
    if value.numel() == 0:
        return None, "value is an empty tensor"
    shape = getattr(event, "shape", None)
    if not isinstance(shape, (list, tuple)) or tuple(shape) != tuple(value.shape):
        return None, "declared shape does not match the saved value"
    if getattr(event, "dtype", None) != str(value.dtype):
        return None, "declared dtype does not match the saved value"
    if getattr(event, "device", None) != str(value.device):
        return None, "declared device does not match the saved value"
    if not bool(torch.isfinite(value).all()):
        return None, "value contains nonfinite elements"
    return value.detach(), None


def _within_tolerance(reference, candidate, rtol, atol):
    if not rtol and not atol:
        return bool(torch.equal(reference, candidate))
    # Subtract before scaling to preserve differences between nearby numbers.
    # If both the difference and the tolerance overflow, compare their halves:
    # this avoids mistaking inf <= inf for equality in extreme opposite signs.
    reference = reference.to(dtype=torch.float64)
    candidate = candidate.to(dtype=torch.float64)
    difference = (candidate - reference).abs()
    bound = atol + rtol * reference.abs()
    half_difference = (candidate * 0.5 - reference * 0.5).abs()
    half_bound = atol * 0.5 + (rtol * 0.5) * reference.abs()
    equal = torch.where(torch.isinf(difference) & torch.isinf(bound),
                        half_difference <= half_bound, difference <= bound)
    return bool(equal.all())


def _compare_pair(reference, candidate, rtol, atol):
    left, left_error = _payload(reference)
    right, right_error = _payload(candidate)
    if left_error or right_error:
        return "unavailable", {"reference": left_error, "candidate": right_error}
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return "incompatible", "saved tensor shape, dtype, or device differs"
    if _within_tolerance(left, right, rtol, atol):
        return "equal", None
    return "different", None


def compare_traces(reference, candidate, *, rtol=0.0, atol=0.0,
                   reference_complete=False, candidate_complete=False):
    """Compare two legacy probe streams without silently aligning gaps.

    Completion flags are an explicit caller attestation that each supplied
    stream contains every event emitted under its declared tap policy. They do
    not assert exhaustive model coverage. With default flags, a no-difference
    conclusion and a first-observed interval remain unavailable. Matched
    occurrence ordinals are only conditional correspondences, not proof that
    two calls have the same semantic role.
    """
    rtol, atol = _tolerance("rtol", rtol), _tolerance("atol", atol)
    if type(reference_complete) is not bool or type(candidate_complete) is not bool:
        raise ValueError("completion flags must be boolean")
    left = _inspect(list(reference))
    right = _inspect(list(candidate))
    left_keys, right_keys = set(left["unique"]), set(right["unique"])
    ambiguous = left["ambiguous"] | right["ambiguous"]
    common = (left_keys & right_keys) - ambiguous
    reference_only = left_keys - right_keys - ambiguous
    candidate_only = right_keys - left_keys - ambiguous
    left_order = [key for _, key in left["records"] if key in common]
    right_order = [key for _, key in right["records"] if key in common]
    comparisons, pair_results = [], {}
    for key in left_order:
        result, reason = _compare_pair(left["unique"][key][1],
                                       right["unique"][key][1], rtol, atol)
        pair_results[key] = result
        comparisons.append({"coordinate": _display(key), "result": result,
                            "reason": reason})

    first_key = next((key for key in left_order if pair_results[key] == "different"), None)
    first_compared = None if first_key is None else _display(first_key)
    interval = None
    sequence_ok = left["sequence_complete"] and right["sequence_complete"]
    invocations_ok = left["invocations_complete"] and right["invocations_complete"]
    provenance_ok = left["run_identity_consistent"] and right["run_identity_consistent"]
    if (first_key is not None and reference_complete and candidate_complete
            and sequence_ok and invocations_ok and provenance_ok):
        first_left = left["unique"][first_key][0]
        first_right = right["unique"][first_key][0]
        if first_left == first_right:
            prefix_left = [key for _, key in left["records"][:first_left]]
            prefix_right = [key for _, key in right["records"][:first_right]]
            if (prefix_left == prefix_right and
                    all(key in common and pair_results[key] == "equal" for key in prefix_left)):
                interval = {"last_equal": _display(prefix_left[-1]) if prefix_left else None,
                            "first_unequal": _display(first_key)}

    complete_comparison = (
        reference_complete and candidate_complete and sequence_ok and invocations_ok and provenance_ok
        and bool(left["records"]) and bool(right["records"])
        and not left["invalid"] and not right["invalid"] and not ambiguous
        and not reference_only and not candidate_only and left_order == right_order
        and len(comparisons) == len(left["records"]) == len(right["records"])
        and all(item["result"] == "equal" for item in comparisons)
    )
    status = ("difference_observed" if interval is not None else
              "no_difference_observed" if complete_comparison else "inconclusive")
    return {
        "status": status,
        "first_compared_difference": first_compared,
        "first_observed_interval": interval,
        "coverage": {
            "reference_count": len(left["records"]),
            "candidate_count": len(right["records"]),
            "reference_complete": reference_complete,
            "candidate_complete": candidate_complete,
            "reference_sequence_complete": left["sequence_complete"],
            "candidate_sequence_complete": right["sequence_complete"],
            "reference_invocations_complete": left["invocations_complete"],
            "candidate_invocations_complete": right["invocations_complete"],
            "reference_run_identity_consistent": left["run_identity_consistent"],
            "candidate_run_identity_consistent": right["run_identity_consistent"],
            "matching_order": (bool(left["records"]) and not ambiguous and
                               [key for _, key in left["records"]] ==
                               [key for _, key in right["records"]] and
                               all(key is not None for _, key in left["records"])),
            "invalid_reference": left["invalid"],
            "invalid_candidate": right["invalid"],
            "ambiguous_reference": [_display(key) for key in sorted(left["ambiguous"])],
            "ambiguous_candidate": [_display(key) for key in sorted(right["ambiguous"])],
            "reference_only": [_display(key) for key in sorted(reference_only)],
            "candidate_only": [_display(key) for key in sorted(candidate_only)],
            "compared": sum(item["result"] in ("equal", "different") for item in comparisons),
            "unavailable": sum(item["result"] == "unavailable" for item in comparisons),
            "incompatible": sum(item["result"] == "incompatible" for item in comparisons),
        },
        "comparisons": comparisons,
        "tolerance": {"rtol": rtol, "atol": atol, "reference": "reference"},
        "causal_origin": "unknown",
    }
