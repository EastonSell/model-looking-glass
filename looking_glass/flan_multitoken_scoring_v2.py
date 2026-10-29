"""Model-free four-control and planned-slot scoring for the successor study.

The caller must independently establish baseline support, valid landings,
saved-output integrity and the status of each unit.  These functions do not
load a model, select cases or authenticate an execution.
"""

import math


RECEIVERS = ("P1", "N1", "P2", "N2")
OPPOSITE = "opposite_adjective"
CONTROLS = ("same_adjective", "opposite_instruction", "opposite_subject", "orthogonal")
ARMS = (OPPOSITE, *CONTROLS)


def group_decision(clean, patched):
    """Score the donor arms and, when available, the orthogonal control."""
    if (type(clean) is not dict or set(clean) != set(RECEIVERS)
            or type(patched) is not dict
            or set(patched) not in (set(ARMS), set(ARMS[:-1]))
            or any(type(row) is not dict or set(row) != set(RECEIVERS)
                   for row in patched.values())):
        raise ValueError("expected four receivers and all donor arms")
    values = list(clean.values()) + [v for row in patched.values() for v in row.values()]
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("margins must be finite binary64 values")
    movements = {
        arm: {receiver: (clean[receiver] - patched[arm][receiver]
                         if receiver.startswith("P") else patched[arm][receiver] - clean[receiver])
              for receiver in RECEIVERS}
        for arm in ARMS if arm in patched
    }
    if any(not math.isfinite(value) for row in movements.values() for value in row.values()):
        raise ValueError("computed movement is nonfinite")
    try:
        a = math.fsum(movements[OPPOSITE][receiver] for receiver in RECEIVERS) / 4.0
        means = {arm: math.fsum(abs(movements[arm][receiver]) for receiver in RECEIVERS) / 4.0
                 for arm in CONTROLS if arm in patched}
    except (OverflowError, ValueError) as error:
        raise ValueError("computed aggregate is nonfinite") from error
    if not math.isfinite(a) or any(not math.isfinite(value) for value in means.values()):
        raise ValueError("computed aggregate is nonfinite")
    b3 = max(means[arm] for arm in CONTROLS[:3])
    b4 = max(means.values()) if "orthogonal" in patched else None
    positive_count = sum(movements[OPPOSITE][receiver] > 0 for receiver in RECEIVERS)
    def passes(b):
        difference = a - b
        if not math.isfinite(difference):
            raise ValueError("computed control difference is nonfinite")
        return bool(a >= 0.10 and difference >= 0.05 and positive_count >= 3)
    return {"movements": movements, "A": a, "B3": b3, "B4": b4,
            "control_means": means, "positive_count": positive_count,
            "passes_B3": passes(b3), "passes_B4": None if b4 is None else passes(b4)}


def primary_interval(outcomes, *, attempt_complete):
    """Classify the threshold using four planned slots and explicit missingness.

    Each outcome is ``pass``, ``fail`` or ``unmeasured``.  A caller may mark
    a unit pass/fail only after all required groups have been independently
    replayed and meet the protocol's baseline and landing requirements.
    """
    if (type(outcomes) not in (list, tuple) or len(outcomes) != 4
            or any(type(value) is not str or value not in ("pass", "fail", "unmeasured")
                   for value in outcomes) or type(attempt_complete) is not bool):
        raise ValueError("expected four explicit slot outcomes and execution status")
    passed = sum(value == "pass" for value in outcomes)
    evaluable = sum(value != "unmeasured" for value in outcomes)
    rate = None if evaluable == 0 else passed / evaluable
    if not attempt_complete:
        return {"status": "incomplete", "passes": passed, "evaluable": evaluable,
                "planned": 4, "rate": rate, "interval": None, "partial": True}
    interval = (passed, passed + 4 - evaluable)
    status = "met" if passed >= 3 else "not-met" if interval[1] < 3 else "undetermined"
    return {"status": status, "passes": passed, "evaluable": evaluable,
            "planned": 4, "rate": rate, "interval": interval, "partial": evaluable < 4}


def holdout_interval(outcomes, *, attempt_complete):
    """Classify the two fixed painting slots, without compacting missing slots.

    The caller must independently establish both instruction groups for any
    measured slot. A completed pass is not by itself scientific replay.
    """
    if (type(outcomes) not in (list, tuple) or len(outcomes) != 2
            or any(type(value) is not str or value not in ("pass", "fail", "unmeasured")
                   for value in outcomes) or type(attempt_complete) is not bool):
        raise ValueError("expected two explicit holdout outcomes and execution status")
    passed = sum(value == "pass" for value in outcomes)
    evaluable = sum(value != "unmeasured" for value in outcomes)
    rate = None if evaluable == 0 else passed / evaluable
    if not attempt_complete:
        return {"status": "incomplete", "passes": passed, "evaluable": evaluable,
                "planned": 2, "rate": rate, "interval": None, "partial": True}
    interval = (passed, passed + 2 - evaluable)
    status = "met" if passed == 2 else "not-met" if interval[1] < 2 else "undetermined"
    return {"status": status, "passes": passed, "evaluable": evaluable,
            "planned": 2, "rate": rate, "interval": interval, "partial": evaluable < 2}
