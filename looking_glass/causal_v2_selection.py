"""Discovery-only observation validation and selection for synthetic v2.

No injected site, site schedule, held-out case or outcome is accepted. This
component does not generate cases or run a model, and does not enforce custody.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import statistics

import torch

from .causal_trace_v2 import SparseTraceInvalid, _binary32_value, compare_sparse_pair
from .causal_v2_runtime import NumericPolicy, NumericPolicyError, float32_exact_equal


class DiscoveryInvalid(ValueError):
    """Invalid discovery cell; sanitized partial evidence may be retained."""
    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = deepcopy(evidence)


@dataclass(frozen=True)
class DiscoveryCondition:
    strength: float
    perturbed_trace: tuple
    perturbed_logits: torch.Tensor
    clean_to_perturbed_logits: tuple
    perturbed_to_clean_logits: tuple
    null_logits: tuple
    identity_logits: tuple
    upstream_logits: torch.Tensor
    clean_verify_logits: torch.Tensor


@dataclass(frozen=True)
class DiscoveryCase:
    case_index: int
    polarity: int
    clean_trace: tuple
    repeated_clean_trace: tuple
    clean_logits: torch.Tensor
    repeated_clean_logits: torch.Tensor
    conditions: tuple


_SITES = (1, 2, 3)
_STRENGTHS = (0.25, 0.75)


def _equal(left, right, policy):
    policy.verify()
    result = float32_exact_equal(left, right)
    policy.verify()
    return result


def _logits(value, policy):
    if type(value) is not torch.Tensor or tuple(value.shape) != (1, 2):
        raise DiscoveryInvalid("invalid full logits")
    _equal(value, value, policy)  # Validates finite CPU float32 payload.
    return value


def _metric(logits, policy):
    # The subtraction is float32 before conversion to binary64, as specified.
    policy.verify()
    result = logits[0, 1] - logits[0, 0]
    _equal(result, result, policy)
    value = float(result.item())
    policy.verify()
    return value


def _values(tensor):
    # All payloads have already been validated; integer reinterpretation keeps
    # the saved float32 bits intact when constructing the public record.
    return [_binary32_value(bits) for bits in tensor.view(torch.int32).reshape(-1).tolist()]


def _trace(events):
    return [{"node_id": event.node_id, "sequence": event.sequence,
             "occurrence": event.occurrence, "shape": list(event.shape),
             "dtype": event.dtype, "device": event.device,
             "values": _values(event.value)} for event in events]


def _pair(reference, candidate, policy):
    policy.verify()
    result = compare_sparse_pair(reference, candidate, reference_complete=True,
                                 candidate_complete=True, pairing_verified=True)
    policy.verify()
    return result


def _three(values, name, policy):
    if type(values) not in (tuple, list) or len(values) != 3:
        raise DiscoveryInvalid(f"{name} requires exactly three site logits")
    return tuple(_logits(value, policy) for value in values)


def _condition(case, condition, policy):
    baseline = _logits(condition.perturbed_logits, policy)
    patches = _three(condition.clean_to_perturbed_logits, "forward patches", policy)
    reverse = _three(condition.perturbed_to_clean_logits, "reverse patches", policy)
    nulls = _three(condition.null_logits, "nulls", policy)
    identities = _three(condition.identity_logits, "identity patches", policy)
    upstream = _logits(condition.upstream_logits, policy)
    clean_verify = _logits(condition.clean_verify_logits, policy)
    result = _pair(case.clean_trace, condition.perturbed_trace, policy)
    if result["status"] == "upstream_difference":
        raise DiscoveryInvalid("upstream sparse difference")
    if (not all(_equal(value, baseline, policy) for value in nulls + identities)
            or not _equal(upstream, baseline, policy)
            or not _equal(patches[2], case.clean_logits, policy)
            or not _equal(clean_verify, case.clean_logits, policy)):
        raise DiscoveryInvalid("required discovery control failed")
    if (_equal(case.clean_trace[3].value, condition.perturbed_trace[3].value, policy)
            and not _equal(case.clean_logits, baseline, policy)):
        raise DiscoveryInvalid("equal final sparse states produced unequal logits")

    base_metric = _metric(baseline, policy)
    effects = [_metric(patch, policy) - base_metric for patch in patches]
    return result, effects, {
        "strength": condition.strength,
        "perturbed_trace": _trace(condition.perturbed_trace),
        "first_unequal": result["first_unequal"], "last_equal": result["last_equal"],
        "l2_by_block": result["l2_by_block"],
        "perturbed_logits": _values(baseline),
        "clean_to_perturbed_logits": [_values(value) for value in patches],
        "perturbed_to_clean_logits": [_values(value) for value in reverse],
        "null_logits": [_values(value) for value in nulls],
        "identity_logits": [_values(value) for value in identities],
        "upstream_logits": _values(upstream), "clean_verify_logits": _values(clean_verify),
        "effects": effects,
    }


def freeze_discovery(cases, *, numeric_policy, pairing_verified=False,
                     private_controls_verified=False):
    """Select and fit from 16 ordered, externally paired discovery cases.

    The caller attests complete captures, pairing, correct polarity, same-input
    patch sources, and success of all target-identifying private integrity and
    step-map controls. Those attestations do not prove blindness or integrity.
    Keep every tensor snapshot unchanged throughout this call. On failure the
    caller must retain an opaque-cell failure; this function cannot publish it.
    """
    if (pairing_verified is not True or private_controls_verified is not True or
            type(numeric_policy) is not NumericPolicy):
        raise DiscoveryInvalid("numeric policy and external attestations are required")
    if type(cases) not in (tuple, list) or len(cases) != 16:
        raise DiscoveryInvalid("exactly 16 ordered discovery cases are required")
    try:
        return _freeze(cases, numeric_policy)
    except (NumericPolicyError, SparseTraceInvalid) as error:
        raise DiscoveryInvalid("invalid numeric policy or sparse observation") from error


def _freeze(cases, policy):
    policy.verify()
    votes = Counter()
    norms = {site: [] for site in _SITES}
    effects = {site: {strength: {1: [], -1: []} for strength in _STRENGTHS}
               for site in _SITES}
    measurements, misses = [], 0
    for index, case in enumerate(cases):
        if (type(case) is not DiscoveryCase or type(case.case_index) is not int or
                case.case_index != index or type(case.polarity) is not int or
                case.polarity != (1 if index % 2 == 0 else -1) or
                type(case.conditions) not in (tuple, list) or len(case.conditions) != 2):
            raise DiscoveryInvalid("discovery case order or polarity is invalid")
        _logits(case.clean_logits, policy)
        _logits(case.repeated_clean_logits, policy)
        negative = _pair(case.clean_trace, case.repeated_clean_trace, policy)
        negative_logits_equal = _equal(case.clean_logits, case.repeated_clean_logits, policy)
        if negative["status"] != "no_difference_observed" or not negative_logits_equal:
            raise DiscoveryInvalid("no-injection repeat differed", evidence={
                "no_injection_failure": {
                    "case_index": index,
                    "false_detection": negative["first_unequal"] is not None,
                    "comparison": negative,
                    "logits_equal": negative_logits_equal,
                    "clean_trace": _trace(case.clean_trace),
                    "repeated_clean_trace": _trace(case.repeated_clean_trace),
                    "clean_logits": _values(case.clean_logits),
                    "repeated_clean_logits": _values(case.repeated_clean_logits),
                }})
        case_record = {"case_index": index, "polarity": case.polarity,
                       "clean_trace": _trace(case.clean_trace),
                       "repeated_clean_trace": _trace(case.repeated_clean_trace),
                       "clean_logits": _values(case.clean_logits),
                       "repeated_clean_logits": _values(case.repeated_clean_logits),
                       "conditions": []}
        for position, condition in enumerate(case.conditions):
            if (type(condition) is not DiscoveryCondition or
                    type(condition.strength) is not float or
                    condition.strength != _STRENGTHS[position]):
                raise DiscoveryInvalid("strength order is invalid")
            result, patch_effects, record = _condition(case, condition, policy)
            if result["first_unequal"] is None:
                misses += 1
            else:
                votes[int(result["first_unequal"].split(".")[1])] += 1
            for site in _SITES:
                norms[site].append(result["l2_by_block"][site])
                effects[site][condition.strength][case.polarity].append(patch_effects[site - 1])
            case_record["conditions"].append(record)
        measurements.append(case_record)

    means = {site: statistics.fmean(norms[site]) for site in _SITES}
    first = max(_SITES, key=lambda site: (votes[site], -site)) if votes else None
    magnitude = max(_SITES, key=lambda site: (means[site], -site)) if any(means.values()) else None
    predictions = {}
    for site in _SITES:
        predictions[str(site)] = {}
        for strength in _STRENGTHS:
            positive = effects[site][strength][1]
            negative = effects[site][strength][-1]
            # Case indices alternate positive and negative; this restores the
            # prescribed case-index order for the constant predictor.
            all_cases = [effect for pair in zip(positive, negative) for effect in pair]
            predictions[str(site)][str(strength)] = {
                "positive": statistics.fmean(positive),
                "negative": statistics.fmean(negative),
                "constant": statistics.fmean(all_cases), "zero": 0.0,
            }
    policy.verify()
    return {"first_changed": first, "magnitude": magnitude,
            "first_votes": {str(site): votes[site] for site in _SITES},
            "mean_l2": {str(site): means[site] for site in _SITES},
            "observation_misses": misses,
            "no_injection": {"cases": 16, "false_detections": 0,
                             "first_changed": None, "magnitude": None},
            "random": {"sites": list(_SITES), "each_probability": "1/3"},
            "predictions": predictions, "measurements": measurements}
