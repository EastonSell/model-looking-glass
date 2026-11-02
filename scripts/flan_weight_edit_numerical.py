"""Numerical rules for the prospective FLAN weight-change study."""

import math
import random

import numpy as np


class DirectionIneligible(ValueError):
    """A prescribed direction has zero or nonfinite magnitude before edits."""


def select_control(primary, candidates):
    if len(primary) != 2 or not candidates:
        raise ValueError("two primary calibration cases and a grid are required")

    def valid(row):
        return (set(row) == {"margin_gain", "target_gain", "negative"}
                and type(row["negative"]) is bool
                and all(type(row[k]) in (float, int) and math.isfinite(row[k])
                        for k in ("margin_gain", "target_gain")))

    if not all(valid(row) for row in primary):
        raise ValueError("invalid primary calibration effect")
    primary_eligible = all(row["negative"] and row["margin_gain"] >= 0.5
                           and row["target_gain"] >= 0.5 for row in primary)
    assessed = []
    doses = set()
    for candidate in candidates:
        if (set(candidate) != {"dose", "rows"} or type(candidate["dose"]) not in (float, int)
                or not math.isfinite(candidate["dose"]) or candidate["dose"] <= 0
                or candidate["dose"] in doses or len(candidate["rows"]) != 2
                or not all(valid(row) for row in candidate["rows"])):
            raise ValueError("invalid control grid point")
        doses.add(candidate["dose"])
        ratios = [abs(math.fsum((float(observed[k]), -float(reference[k]))))
                  / max(0.10, 0.10 * abs(float(reference[k])))
                  for observed, reference in zip(candidate["rows"], primary)
                  for k in ("margin_gain", "target_gain")]
        maximum = max(ratios)
        matched = primary_eligible and maximum <= 1.0 and all(r["negative"] for r in candidate["rows"])
        assessed.append({"dose": float(candidate["dose"]), "maximum_ratio": maximum,
                         "ratios": ratios, "matched": matched})
    matches = [point for point in assessed if point["matched"]]
    chosen = min(matches or assessed, key=lambda point: (point["maximum_ratio"], point["dose"]))
    return {**chosen, "primary_eligible": primary_eligible, "grid": assessed}


def select_control_fixed(primary, candidates):
    expected = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
    if (len(candidates) != len(expected)
            or any(type(point.get("dose")) not in (float, int) for point in candidates)
            or sorted(point["dose"] for point in candidates) != list(expected)):
        raise ValueError("selection requires the complete fixed grid")
    return select_control(primary, candidates)


def _terms(steps, sign=1.0):
    for z, maximum, logsum in steps:
        if not all(math.isfinite(float(x)) for x in (z, maximum, logsum)):
            raise ValueError("nonfinite score term")
        yield sign * float(z)
        yield -sign * float(maximum)
        yield -sign * float(logsum)


def sequence_score(terms):
    return math.fsum(_terms(terms))


def sequence_effect(after, before):
    return math.fsum((*_terms(after), *_terms(before, -1.0)))


def margin_effect(negative_after, negative_before, positive_after, positive_before):
    return math.fsum((*_terms(negative_after), *_terms(negative_before, -1.0),
                      *_terms(positive_after, -1.0), *_terms(positive_before)))


def directions(wonderful, awful, excellent, head_negative, head_positive):
    values = (wonderful, awful, excellent, head_negative, head_positive)
    if any(not isinstance(value, np.ndarray) or value.shape != (512,)
           or value.dtype != np.float32 or not np.isfinite(value).all() for value in values):
        raise ValueError("directions require five finite 512-element float32 checkpoint rows")

    def difference(a, b):
        return np.array([math.fsum((float(x), -float(y))) for x, y in zip(a, b)], dtype=np.float64)

    target = difference(awful, wonderful)
    same = difference(excellent, wonderful)
    broad = difference(head_positive, head_negative)
    try:
        target_norm = norm(target)
        if not target_norm or not norm(same) or not norm(broad):
            raise DirectionIneligible("zero checkpoint direction")
    except (OverflowError, ValueError) as exc:
        raise DirectionIneligible("zero or nonfinite checkpoint direction") from exc
    generator = random.Random(20261029)
    sample = [generator.gauss(0, 1) for _ in range(512)]
    coefficient = math.fsum(x * float(y) for x, y in zip(sample, target)) / (target_norm * target_norm)
    residual = [math.fsum((x, -coefficient * float(y))) for x, y in zip(sample, target)]
    try:
        residual_norm = norm(residual)
    except (OverflowError, ValueError) as exc:
        raise DirectionIneligible("nonfinite random residual") from exc
    if not residual_norm:
        raise DirectionIneligible("zero random residual")
    random_direction = np.array([x * (target_norm / residual_norm) for x in residual], dtype=np.float64)
    if not np.isfinite(random_direction).all():
        raise DirectionIneligible("nonfinite random direction")
    return {"target": target, "same": same, "broad": broad, "opposite": -target,
            "random": random_direction}


def norm(vector):
    result = math.sqrt(math.fsum(float(x) * float(x) for x in vector))
    if not math.isfinite(result):
        raise ValueError("nonfinite norm")
    return result


def donor_row(donor):
    if (not isinstance(donor, np.ndarray) or donor.shape != (512,)
            or donor.dtype != np.float32 or not np.isfinite(donor).all()):
        raise ValueError("donor must be a finite 512-element float32 checkpoint row")
    return donor.copy()


def construct_row(original, direction, dose):
    if (not isinstance(original, np.ndarray) or original.shape != (512,)
            or original.dtype != np.float32 or len(direction) != 512
            or type(dose) not in (float, int) or not math.isfinite(dose) or dose <= 0):
        raise ValueError("invalid row construction input")
    result = np.array([math.fsum((float(x), float(dose) * float(delta)))
                       for x, delta in zip(original, direction)], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("constructed row is nonfinite")
    return result
