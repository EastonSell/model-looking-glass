"""Deterministic, baseline-only selection for the prospective screened study.

This helper neither runs a model nor proves that its inputs came from the
published pool or a clean execution. The owner must bind the exact case
document and full saved screening logits, and must keep both quiescent. A
separate audit and supervised cleanup are required for a final outcome.
"""

import math


class SelectionError(ValueError):
    """The screening input is incomplete, nonfinite, or inconsistent."""


STRATA = ("s1", "s2", "s3", "s4")
PER_STRATUM = 8
SELECT_COUNT = 8


def select_screen(targets, margins):
    """Select negative-baseline rows by the fixed four-stratum round robin.

    ``targets`` is the 32-row target portion of the already authenticated case
    document in its frozen order, grouped as eight rows per stratum. ``margins``
    are the corresponding London-minus-Paris margins from the original saved
    float32 rows after conversion to float64. Only sign and within-stratum
    order influence selection; the magnitude of a negative margin does not.
    """
    if (type(targets) not in (tuple, list) or len(targets) != 32
            or type(margins) not in (tuple, list) or len(margins) != 32):
        raise SelectionError("screen requires exactly 32 target rows and margins")
    rows, seen = [], set()
    for offset, (case, raw_margin) in enumerate(zip(targets, margins)):
        if type(case) is not dict:
            raise SelectionError("invalid target identity")
        name, role, stratum = case.get("case_id"), case.get("role"), case.get("stratum")
        expected_stratum = STRATA[offset // PER_STRATUM]
        if (type(name) is not str or not name or name in seen or role != "target"
                or stratum != expected_stratum):
            raise SelectionError("target identities or frozen stratum order differ")
        if type(raw_margin) not in (int, float):
            raise SelectionError("baseline margin must be a finite real number")
        try:
            margin = float(raw_margin)
        except OverflowError as exc:
            raise SelectionError("baseline margin must be a finite real number") from exc
        if not math.isfinite(margin):
            raise SelectionError("baseline margin must be a finite real number")
        seen.add(name)
        rows.append({"offset": offset, "case_id": name, "stratum": stratum,
                     "baseline_margin": margin, "eligible": margin < 0.0,
                     "selected": False})

    eligible = {group: [r for r in rows if r["stratum"] == group and r["eligible"]]
                for group in STRATA}
    positions = {group: 0 for group in STRATA}
    selected = []
    while len(selected) < SELECT_COUNT:
        advanced = False
        for group in STRATA:
            position = positions[group]
            if position == len(eligible[group]):
                continue
            row = eligible[group][position]
            positions[group] += 1
            row["selected"] = True
            selected.append(row)
            advanced = True
            if len(selected) == SELECT_COUNT:
                break
        if not advanced:
            break

    return {"version": "targeted-edit-screened-selection-v1",
            "rows": rows,
            "selected_ids": [row["case_id"] for row in selected],
            "selected_offsets": [row["offset"] for row in selected],
            "eligible_counts_by_stratum": {group: len(eligible[group]) for group in STRATA},
            "selected_counts_by_stratum": dict(positions),
            "eligible_count": sum(len(group) for group in eligible.values()),
            "selected_count": len(selected),
            "formal_target_eligible": len(selected) == SELECT_COUNT}
