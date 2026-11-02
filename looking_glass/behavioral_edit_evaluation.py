"""Descriptive paired behavioral-edit observations, with no model execution.

Callers define the endpoint, support, reference labels and state-digest scope;
this module cannot verify their semantics, execution, or rollback provenance.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re


@dataclass(frozen=True)
class BehavioralObservation:
    value: object = None
    status: str = "observed"
    token_ids: tuple | None = None
    decoded_text: str | None = None
    termination: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class CategoricalDistribution:
    support: tuple
    probabilities: tuple
    context: str


@dataclass(frozen=True)
class ScalarComparison:
    name: str
    before: object = None
    after: object = None


@dataclass(frozen=True)
class BehavioralCase:
    case_id: str
    role: str
    before: BehavioralObservation | None
    after: BehavioralObservation | None
    input_id: str | None = None
    references: tuple | None = None
    targets: tuple | None = None
    before_distribution: CategoricalDistribution | None = None
    after_distribution: CategoricalDistribution | None = None
    measures: tuple = ()
    restored: BehavioralObservation | None = None


@dataclass(frozen=True)
class StateDigest:
    scope: str
    sha256: str


@dataclass(frozen=True)
class RollbackEvidence:
    attempted: bool | None
    before: StateDigest | None = None
    after: StateDigest | None = None


_NORMALIZATION_TOLERANCE = 1e-6  # Validation of supplied probabilities, not an outcome threshold.
_STATUSES = frozenset(("observed", "missing", "tied", "invalid"))


def _name(value, field):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _value(value):
    if type(value) in (str, int, bool):
        return value
    if type(value) is tuple and all(type(x) is int and x >= 0 for x in value):
        return value
    raise ValueError("outcome must be a string, integer, boolean or tuple of nonnegative token IDs")


def _json_value(value):
    return list(value) if type(value) is tuple else value


def _observation(value):
    if value is None:
        value = BehavioralObservation(status="missing")
    if type(value) is not BehavioralObservation or type(value.status) is not str or value.status not in _STATUSES:
        raise ValueError("observation status differs")
    if value.status == "observed":
        _value(value.value)
    elif value.value is not None:
        raise ValueError("a missing, tied or invalid observation cannot have a comparison value")
    if value.token_ids is not None and (type(value.token_ids) is not tuple or
                                       any(type(x) is not int or x < 0 for x in value.token_ids)):
        raise ValueError("token_ids must be a tuple of nonnegative integers")
    if type(value.value) is tuple and value.token_ids is not None and value.value != value.token_ids:
        raise ValueError("sequence outcome and token_ids differ")
    for item, label in ((value.decoded_text, "decoded_text"), (value.termination, "termination"),
                        (value.reason, "reason")):
        if item is not None and type(item) is not str:
            raise ValueError(f"{label} must be a string or None")
    return {"status": value.status, "value": _json_value(value.value),
            "token_ids": None if value.token_ids is None else list(value.token_ids),
            "decoded_text": value.decoded_text, "termination": value.termination, "reason": value.reason}, value


def _labels(labels, name):
    if labels is None:
        return None
    if type(labels) is not tuple or not labels:
        raise ValueError(f"{name} must be a nonempty tuple or None")
    for label in labels:
        _value(label)
    if len({type(label) for label in labels}) != 1 or len(set(labels)) != len(labels):
        raise ValueError(f"{name} must use distinct values of one type")
    return labels


def _matching(observation, labels):
    if observation.status != "observed" or labels is None:
        return None
    return observation.value in labels


def _number(value):
    if value is None:
        return None, "missing"
    if type(value) not in (int, float):
        raise ValueError("scalar values must be built-in integers, floats or None")
    try:
        converted = float(value)
    except OverflowError:
        return None, "invalid_unrepresentable"
    if not math.isfinite(converted):
        return None, "invalid_nonfinite"
    if type(value) is int and int(converted) != value:
        return None, "invalid_unrepresentable"
    return converted, "observed"


def _measures(items):
    if type(items) is not tuple:
        raise ValueError("measures must be a tuple")
    result = {}
    for item in items:
        if type(item) is not ScalarComparison:
            raise ValueError("measures must contain ScalarComparison records")
        name = _name(item.name, "measure name")
        if name in result:
            raise ValueError("duplicate measure name")
        before, before_status = _number(item.before)
        after, after_status = _number(item.after)
        delta, delta_status = None, "unavailable"
        if before_status == after_status == "observed":
            try:
                delta = math.fsum((after, -before))
            except OverflowError:
                delta = None
            if delta is None or not math.isfinite(delta):
                delta, delta_status = None, "unrepresentable"
            else:
                delta_status = "observed"
        result[name] = {"before": before, "after": after, "before_status": before_status,
                        "after_status": after_status, "delta": delta, "delta_status": delta_status}
    return result


def _distribution(value):
    if value is None:
        return None
    if type(value) is not CategoricalDistribution or type(value.support) is not tuple or not value.support:
        raise ValueError("distribution support must be a nonempty tuple")
    if (any(type(label) not in (str, int) for label in value.support)
            or len(set(value.support)) != len(value.support)):
        raise ValueError("distribution support must contain distinct strings or integers")
    if type(value.probabilities) is not tuple or len(value.probabilities) != len(value.support):
        raise ValueError("distribution probabilities must match the support length")
    _name(value.context, "distribution context")
    raw = json.dumps(value.support, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    support_sha = hashlib.sha256(raw).hexdigest()
    values = []
    reason = None
    for probability in value.probabilities:
        if type(probability) not in (int, float):
            raise ValueError("probabilities must be built-in integers or floats")
        try:
            number = float(probability)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number):
            reason = "nonfinite"
        elif not 0 <= number <= 1 and reason is None:
            reason = "outside_unit_interval"
        values.append(number)
    total = None
    if all(math.isfinite(x) for x in values):
        try:
            total = math.fsum(values)
        except (OverflowError, ValueError):
            if reason is None:
                reason = "unrepresentable_sum"
    if reason is None and (total is None or abs(total - 1.0) > _NORMALIZATION_TOLERANCE):
        reason = "not_normalized"
    return {"support": value.support, "context": value.context, "support_sha256": support_sha, "values": values,
            "sum": total, "reason": reason}


def _compare_distributions(before, after):
    a, b = _distribution(before), _distribution(after)
    output = {"status": None, "total_variation": None, "support_size": None,
              "before_context": None if a is None else a["context"],
              "after_context": None if b is None else b["context"],
              "before_support_sha256": None if a is None else a["support_sha256"],
              "after_support_sha256": None if b is None else b["support_sha256"],
              "before_probability_sum": None if a is None else a["sum"],
              "after_probability_sum": None if b is None else b["sum"],
              "before_invalid_reason": None if a is None else a["reason"],
              "after_invalid_reason": None if b is None else b["reason"]}
    if a is None or b is None:
        output["status"] = "missing_both" if a is None and b is None else ("missing_before" if a is None else "missing_after")
    elif a["reason"] or b["reason"]:
        output["status"] = ("invalid_both" if a["reason"] and b["reason"] else
                            "invalid_before" if a["reason"] else "invalid_after")
    elif a["context"] != b["context"]:
        output["status"] = "incomparable_context"
    elif a["support"] != b["support"]:
        output["status"] = "incomparable_support"
    else:
        output["status"] = "compared"
        output["support_size"] = len(a["support"])
        output["total_variation"] = min(1.0, 0.5 * math.fsum(
            abs(x / a["sum"] - y / b["sum"]) for x, y in zip(a["values"], b["values"], strict=True)))
    return output


def _state(value):
    if value is None:
        return None
    if (type(value) is not StateDigest or type(value.scope) is not str or not value.scope.strip()
            or type(value.sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", value.sha256) is None):
        raise ValueError("state digest requires a scope and lowercase SHA-256")
    return {"scope": value.scope, "sha256": value.sha256}


def _rollback(value):
    if value is None:
        return {"status": "not_observed", "attempted": None, "before": None, "after": None}
    if type(value) is not RollbackEvidence or (value.attempted is not None and type(value.attempted) is not bool):
        raise ValueError("rollback evidence requires an explicit boolean or unknown attempted state")
    before, after = _state(value.before), _state(value.after)
    if value.attempted is False and after is not None:
        raise ValueError("a not-attempted rollback cannot have an after-rollback digest")
    if value.attempted is None:
        status = "attempt_unknown"
    elif not value.attempted:
        status = "not_attempted"
    elif before is None or after is None:
        status = "incomplete"
    elif before["scope"] != after["scope"]:
        status = "incomparable_scope"
    else:
        status = "reported_match" if before["sha256"] == after["sha256"] else "reported_mismatch"
    return {"status": status, "attempted": value.attempted, "before": before, "after": after}


def _observation_equal(first, second, field):
    if first.status != "observed" or second.status != "observed":
        return None
    left, right = getattr(first, field), getattr(second, field)
    return None if left is None or right is None else left == right


def _case(case):
    if type(case) is not BehavioralCase:
        raise ValueError("cases must contain BehavioralCase records")
    _name(case.case_id, "case_id")
    if type(case.role) is not str or case.role not in ("target", "locality"):
        raise ValueError("case role must be target or locality")
    if case.input_id is not None:
        _name(case.input_id, "input_id")
    before, a = _observation(case.before)
    after, b = _observation(case.after)
    restored, c = _observation(case.restored)
    references, targets = _labels(case.references, "references"), _labels(case.targets, "targets")
    types = {type(x.value) for x in (a, b, c) if x.status == "observed"}
    types.update(type(labels[0]) for labels in (references, targets) if labels is not None)
    if len(types) > 1:
        raise ValueError("observations and declared labels must use one outcome representation")
    value_matches = _observation_equal(a, b, "value")
    token_matches = _observation_equal(a, b, "token_ids")
    text_matches = _observation_equal(a, b, "decoded_text")
    termination_matches = _observation_equal(a, b, "termination")
    return {"case_id": case.case_id, "input_id": case.input_id, "role": case.role,
            "before": before, "after": after, "restored": restored,
            "outcome_value_changed": None if value_matches is None else not value_matches,
            "token_ids_changed": None if token_matches is None else not token_matches,
            "decoded_text_changed": None if text_matches is None else not text_matches,
            "termination_changed": None if termination_matches is None else not termination_matches,
            "restored_value_matches_before": _observation_equal(a, c, "value"),
            "restored_token_ids_match_before": _observation_equal(a, c, "token_ids"),
            "restored_decoded_text_matches_before": _observation_equal(a, c, "decoded_text"),
            "restored_termination_matches_before": _observation_equal(a, c, "termination"),
            "references": None if references is None else [_json_value(x) for x in references],
            "targets": None if targets is None else [_json_value(x) for x in targets],
            "before_reference_match": _matching(a, references), "after_reference_match": _matching(b, references),
            "before_target_match": _matching(a, targets), "after_target_match": _matching(b, targets),
            "distribution": _compare_distributions(case.before_distribution, case.after_distribution),
            "measures": _measures(case.measures)}


def _summary(rows):
    paired = [r for r in rows if r["outcome_value_changed"] is not None]
    referenced = [r for r in rows if r["before_reference_match"] is not None and r["after_reference_match"] is not None]
    target = [r for r in rows if r["before_target_match"] is not None and r["after_target_match"] is not None]
    distributions = [r["distribution"]["total_variation"] for r in rows
                     if r["distribution"]["status"] == "compared"]
    before_matches = sum(r["before_reference_match"] for r in referenced)
    after_matches = sum(r["after_reference_match"] for r in referenced)
    n = len(referenced)
    return {"case_count": len(rows), "paired_outcome_value_count": len(paired),
            "outcome_value_changed_count": sum(r["outcome_value_changed"] for r in paired),
            "outcome_value_unchanged_count": sum(not r["outcome_value_changed"] for r in paired),
            "paired_reference_count": n, "before_reference_matches": before_matches,
            "after_reference_matches": after_matches,
            "before_reference_rate": before_matches / n if n else None,
            "after_reference_rate": after_matches / n if n else None,
            "reference_match_delta": (after_matches - before_matches) / n if n else None,
            "lost_reference_match": sum(r["before_reference_match"] and not r["after_reference_match"] for r in referenced),
            "gained_reference_match": sum(not r["before_reference_match"] and r["after_reference_match"] for r in referenced),
            "paired_target_count": len(target),
            "before_target_matches": sum(r["before_target_match"] for r in target),
            "after_target_matches": sum(r["after_target_match"] for r in target),
            "distribution_compared_count": len(distributions),
            "mean_total_variation": math.fsum(distributions) / len(distributions) if distributions else None,
            "max_total_variation": max(distributions) if distributions else None,
            "restored_value_compared_count": sum(r["restored_value_matches_before"] is not None for r in rows),
            "restored_value_matches_before_count": sum(r["restored_value_matches_before"] is True for r in rows)}


def evaluate_behavioral_edit(cases, *, rollback=None):
    """Describe paired outcomes and supplied distribution/state observations.

    Inputs must stay quiescent for the duration of the call. The caller must
    retain the source evidence and independently establish its provenance.
    """
    if type(cases) not in (list, tuple) or not cases:
        raise ValueError("cases must be a nonempty list or tuple")
    rows = [_case(case) for case in cases]
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate case_id")
    return {"schema": "behavioral-edit-evaluation-v1", "cases": rows,
            "summaries": {role: _summary([row for row in rows if row["role"] == role])
                          for role in ("target", "locality")},
            "rollback": _rollback(rollback)}
