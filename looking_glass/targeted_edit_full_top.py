"""Numerical helpers for the prospective full-top edit study.

This module neither loads a model nor authorizes a study. Its caller must bind
the exact cases and source, preserve the raw rows, establish the runtime and
supervise/verify cleanup. Inputs must be exclusively owned during a call.
"""

from __future__ import annotations

import math

import numpy as np


FACTS = 4
ROWS_PER_FACT = 9  # one discovery, two efficacy-calibration, six evaluation
PROJECTED_GRID = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.01)
UNPROJECTED_GRID = (0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002,
                    0.003, 0.004, 0.006, 0.008, 0.01, 0.012, 0.015)


class StudyError(ValueError):
    """Invalid or incomplete numerical evidence, never a negative result."""


def _rows32(rows, count):
    """Copy and validate float32 rows without floating-point f32 conversion.

    Decode IEEE-754 bits directly into binary64. In particular, this retains
    both signs of float32 subnormal even if an independent reader happens to
    run with flush-to-zero or denormals-are-zero enabled on its own thread.
    """
    if (not isinstance(rows, np.ndarray) or rows.dtype != np.dtype("float32")
            or rows.ndim != 2 or rows.shape[0] != count or rows.shape[1] < 2):
        raise StudyError("expected the specified number of nonempty float32 rows")
    raw = np.ascontiguousarray(rows).copy().view(np.uint32)
    exponent = ((raw >> 23) & 0xff).astype(np.int32)
    if np.any(exponent == 0xff):
        raise StudyError("nonfinite logit")
    fraction = raw & 0x7fffff
    significand = np.where(exponent == 0, fraction, fraction | 0x800000)
    power = np.where(exponent == 0, -149, exponent - 150)
    magnitude = np.ldexp(significand.astype(np.float64), power)
    return np.where((raw & 0x80000000) != 0, -magnitude, magnitude)


def _token_pair(true_id, alternative_id, vocabulary):
    if (type(true_id) is not int or type(alternative_id) is not int
            or not 0 <= true_id < vocabulary or not 0 <= alternative_id < vocabulary
            or true_id == alternative_id):
        raise StudyError("invalid factual/counterfactual token IDs")


def _top(row):
    index = int(np.argmax(row))
    top = float(row[index])
    return index if int(np.count_nonzero(row == top)) == 1 else None


def _top_details(row):
    representative = int(np.argmax(row))
    maximum = float(row[representative])
    count = int(np.count_nonzero(row == maximum))
    second = float(np.partition(row, -2)[-2])
    return {"unique_top_id": representative if count == 1 else None,
            "top_tie_count": count, "top_representative_id": representative,
            "top_gap": maximum - second}


def _category(row, true_id, alternative_id):
    top = _top(row)
    if top is None:
        return "tied"
    if top == true_id:
        return "factual"
    if top == alternative_id:
        return "counterfactual"
    return "other"


def _logp(row, token_id):
    largest = float(np.max(row))
    shifted = row - largest
    log_total = math.log(float(np.exp(shifted).sum(dtype=np.float64)))
    value = float(shifted[token_id]) - log_total
    if not math.isfinite(value):
        raise StudyError("nonfinite log probability")
    return value


def select_screen(rows, pairs):
    """Choose the first fact meeting all frozen unique-top screen counts.

    Rows are fact-major, with each fact's discovery row, two calibration rows
    and six evaluation rows in that order. No margin ranking is performed.
    """
    values = _rows32(rows, FACTS * ROWS_PER_FACT)
    if type(pairs) not in (tuple, list) or len(pairs) != FACTS:
        raise StudyError("four ordered token pairs are required")
    observations, candidates = [], []
    for fact, pair in enumerate(pairs):
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise StudyError("a token pair is malformed")
        true_id, alternative_id = pair
        _token_pair(true_id, alternative_id, values.shape[1])
        start = fact * ROWS_PER_FACT
        eligible = []
        for offset in range(start, start + ROWS_PER_FACT):
            row = values[offset]
            top_details = _top_details(row)
            top = top_details["unique_top_id"]
            ok = top == true_id
            eligible.append(ok)
            observations.append({"offset": offset, "fact_index": fact,
                                 **top_details, "eligible": ok,
                                 "factual_logit": float(row[true_id]),
                                 "alternative_logit": float(row[alternative_id]),
                                 "factual_logp": _logp(row, true_id),
                                 "alternative_logp": _logp(row, alternative_id)})
        evaluation = [start + i for i in range(3, 9) if eligible[i]]
        candidates.append({"fact_index": fact,
                           "discovery_eligible": eligible[0],
                           "calibration_eligible_count": sum(eligible[1:3]),
                           "evaluation_eligible_count": len(evaluation),
                           "qualifies": eligible[0] and all(eligible[1:3]) and len(evaluation) >= 4,
                           "eligible_evaluation_offsets": evaluation})
    selected = next((item for item in candidates if item["qualifies"]), None)
    if selected is None:
        return {"status": "ineligible", "fact_index": None,
                "discovery_offset": None, "calibration_offsets": [],
                "evaluation_offsets": [], "observations": observations,
                "candidates": candidates}
    fact = selected["fact_index"]
    start = fact * ROWS_PER_FACT
    return {"status": "eligible", "fact_index": fact,
            "discovery_offset": start, "calibration_offsets": [start + 1, start + 2],
            "evaluation_offsets": selected["eligible_evaluation_offsets"][:4],
            "observations": observations, "candidates": candidates}


def _effects(baseline, edited, true_id, alternative_id):
    effects = []
    for before, after in zip(baseline, edited):
        margin_before = float(before[alternative_id] - before[true_id])
        margin_after = float(after[alternative_id] - after[true_id])
        effects.append({"alternative_logp_gain": _logp(after, alternative_id) - _logp(before, alternative_id),
                        "margin_gain": margin_after - margin_before,
                        "top": _category(after, true_id, alternative_id)})
    return effects


def select_projected_dose(grid, true_id, alternative_id):
    """Choose the first frozen dose with the alternative as unique top."""
    if (type(grid) not in (tuple, list) or len(grid) != len(PROJECTED_GRID)
            or any(type(item) not in (tuple, list) or len(item) != 2 for item in grid)
            or any(type(item[0]) is not float or item[0] != dose
                   for item, dose in zip(grid, PROJECTED_GRID))):
        raise StudyError("the complete ascending projected grid is required")
    measurements = []
    expected_width = None
    for dose, raw in grid:
        values = _rows32(raw, 1)
        _token_pair(true_id, alternative_id, values.shape[1])
        if expected_width is None:
            expected_width = values.shape[1]
        elif values.shape[1] != expected_width:
            raise StudyError("projected grid vocabulary differs")
        details = _top_details(values[0])
        measurements.append({"dose": dose, **details,
                             "counterfactual_unique_top": details["unique_top_id"] == alternative_id})
    selected = next((entry for entry in measurements if entry["counterfactual_unique_top"]), None)
    return {"development_top_achieved": selected is not None,
            "selected_dose": selected["dose"] if selected else PROJECTED_GRID[-1],
            "fallback": selected is None, "grid": measurements}


def select_development_match(baseline, projected, grid, true_id, alternative_id):
    """Apply both per-row calibration metrics, top categories and fixed grid.

    The result is only a measurement from supplied rows. Owner-level checks
    must prove that each row and weight belongs to the named condition.
    """
    before = _rows32(baseline, 2)
    project = _rows32(projected, 2)
    if project.shape != before.shape:
        raise StudyError("calibration shapes differ")
    _token_pair(true_id, alternative_id, before.shape[1])
    if (type(grid) not in (tuple, list) or len(grid) != len(UNPROJECTED_GRID)
            or any(type(item) not in (tuple, list) or len(item) != 2 for item in grid)
            or any(type(item[0]) is not float or item[0] != dose
                   for item, dose in zip(grid, UNPROJECTED_GRID))):
        raise StudyError("the complete ascending fixed grid is required")
    if any(_top(row) != true_id for row in before):
        raise StudyError("calibration baseline is not uniquely factual")
    p = _effects(before, project, true_id, alternative_id)
    metrics = ("alternative_logp_gain", "margin_gain")
    entries = []
    for dose, raw in grid:
        values = _rows32(raw, 2)
        if values.shape != before.shape:
            raise StudyError("grid calibration shapes differ")
        u = _effects(before, values, true_id, alternative_id)
        ratios = []
        positive, same_top = True, True
        for pp, uu in zip(p, u):
            same_top = same_top and pp["top"] == uu["top"]
            for name in metrics:
                pg, ug = pp[name], uu[name]
                positive = positive and pg > 0.05 and ug > 0.05
                ratios.append(abs(pg - ug) / max(0.005, 0.10 * abs(pg)))
        worst = max(ratios)
        entries.append({"dose": dose, "effects": u, "ratios": ratios,
                        "max_ratio": worst, "positive": positive,
                        "top_categories_agree": same_top,
                        "eligible": positive and same_top and worst <= 1.0})
    nearest = min(entries, key=lambda item: (item["max_ratio"], item["dose"]))
    choices = [item for item in entries if item["eligible"]]
    selected = min(choices, key=lambda item: (item["max_ratio"], item["dose"])) if choices else None
    return {"status": "available" if selected else "unavailable",
            "selected_dose": selected["dose"] if selected else None,
            "nearest_dose": nearest["dose"], "projected_effects": p,
            "grid": entries}


def evaluated_strength_balance(baseline, projected, unprojected, true_id, alternative_id):
    """Require both gains and top categories to balance on every fixed row."""
    before = _rows32(baseline, 4)
    project = _rows32(projected, 4)
    control = _rows32(unprojected, 4)
    if project.shape != before.shape or control.shape != before.shape:
        raise StudyError("evaluation shapes differ")
    _token_pair(true_id, alternative_id, before.shape[1])
    if any(_top(row) != true_id for row in before):
        raise StudyError("evaluation baseline is not uniquely factual")
    p = _effects(before, project, true_id, alternative_id)
    u = _effects(before, control, true_id, alternative_id)
    cases = []
    for pp, uu in zip(p, u):
        comparisons = {}
        for metric in ("alternative_logp_gain", "margin_gain"):
            pg, ug = pp[metric], uu[metric]
            tolerance = 0.05 * max(abs(pg), abs(ug))
            comparisons[metric] = {"projected": pg, "unprojected": ug,
                                   "difference": abs(pg - ug), "tolerance": tolerance,
                                   "passes": pg > 0.05 and ug > 0.05 and abs(pg - ug) <= tolerance}
        cases.append({"projected_top": pp["top"], "unprojected_top": uu["top"],
                      "metrics": comparisons,
                      "passes": pp["top"] == uu["top"] and all(x["passes"] for x in comparisons.values())})
    return {"balanced": all(row["passes"] for row in cases), "cases": cases}
