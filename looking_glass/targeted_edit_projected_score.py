"""Prospective numerical scoring for the public projected-edit experiment.

This module does not load a model, authorize a forward, verify saved artifact
provenance, or decide whether execution and cleanup succeeded. Its numerical
decision is provisional. The caller must bind the exact published case bytes,
the pre-evaluation development record and every copied logit matrix, and keep
all inputs quiescent during the call. The caller must also admit the peak
memory for copied float32 matrices and float64 scoring workspaces. A final
study decision requires the separate execution, custody, completeness and
cleanup checks.
"""
import hashlib
import json
import math

import numpy as np

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2
from . import targeted_edit_projected_math as projected


VERSION = "targeted-edit-projected-score-v1"
PUBLIC_CASE_SHA256 = "f7554a81ac7143c54a56d557fb04cb19419bac30e7dcd44c951eb63ddaff8ee8"
CONDITIONS = ("original", *projected.CONDITIONS)


def _cases(document):
    if (type(document) is not bytes or len(document) > 65536
            or hashlib.sha256(document).hexdigest() != PUBLIC_CASE_SHA256):
        raise v1.StudyError("public case bytes differ from the fixed proposal")
    try:
        rows = json.loads(document)["tokenizer_preflight"]["cases"]
        cases = []
        for row in rows:
            role = row["split"]
            label = None if role == "target" else (row["continuations"][0]["text"],
                                                    row["continuations"][0]["ids"][0])
            cases.append((row["case_id"], role, row["prompt"], tuple(row["token_ids"]), label))
        if (len(cases) != 13 or [item[1] for item in cases] != ["target"] * 4 + ["locality"] * 9
                or len({item[0] for item in cases}) != 13):
            raise v1.StudyError("public case identities differ")
    except (KeyError, TypeError, ValueError, IndexError, UnicodeError) as exc:
        raise v1.StudyError("invalid fixed case document") from exc
    return tuple(cases)


def _finite(value, *, nonnegative=False):
    if type(value) not in (int, float):
        raise v1.StudyError("invalid development measurement")
    try:
        result = float(value)
    except OverflowError as exc:
        raise v1.StudyError("invalid development measurement") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise v1.StudyError("invalid development measurement")
    return result


def _target_diagnostics(scores, name, summary):
    """Report both candidate-token NLLs without declaring either expected."""
    for index, row in enumerate(summary["targets"]):
        _base_z, base_logp, base_top = scores["original"][index]
        _edit_z, edit_logp, edit_top = scores[name][index]
        row.update({"original_top": base_top, "condition_top": edit_top,
                    "kl": float(np.sum(np.exp(base_logp) * (base_logp - edit_logp), dtype=np.float64)),
                    "original_london_nll": float(-base_logp[v1.LONDON]),
                    "condition_london_nll": float(-edit_logp[v1.LONDON]),
                    "original_paris_nll": float(-base_logp[v1.PARIS]),
                    "condition_paris_nll": float(-edit_logp[v1.PARIS])})


def _evaluate(logits, cases, development):
    """Score validated cases and a caller-bound private development record."""
    if type(development) is not dict or development.get("status") != "private-candidate":
        raise v1.StudyError("missing development candidate")
    try:
        discovery = _finite(development["discovery_margin"])
        shifts = development["calibration_shifts"]
        projected_shift = _finite(shifts["projected"]["max_output_abs"], nonnegative=True)
        unprojected_shift = _finite(shifts["unprojected"]["max_output_abs"], nonnegative=True)
    except (KeyError, TypeError) as exc:
        raise v1.StudyError("missing development measurement") from exc
    before = v1._denormal_probe()
    scores = v2._scores(logits, cases, CONDITIONS)
    summaries = {name: v2._condition(scores, cases, name) for name in CONDITIONS}
    for name, summary in summaries.items():
        _target_diagnostics(scores, name, summary)
    edited, random, unprojected = summaries["projected"], summaries["projected_random"], summaries["unprojected"]
    eligible = {"baseline_discovery_negative": discovery < 0,
                "baseline_three_targets_negative": edited["originally_negative_targets"] >= 3,
                "unprojected_calibration_shift": unprojected_shift > 1e-6}
    criteria = {"at_least_two_targets_reversed": edited["reversed_targets"] >= 2,
                "mean_target_gain": edited["mean_target_change"] > 0.5,
                "random_control_advantage": edited["mean_target_change"] - random["mean_target_change"] >= 0.5,
                **v2._locality_criteria(edited),
                "calibration_shift_ratio": projected_shift <= 0.1 * unprojected_shift,
                "locality_kl_advantage": unprojected["mean_kl"] - edited["mean_kl"] >= 1e-6}
    decision = "ineligible" if not all(eligible.values()) else ("success" if all(criteria.values()) else "negative")
    after = v1._denormal_probe()
    return {"version": VERSION, "conditions": summaries, "eligibility": eligible,
            "criteria": criteria, "discovery_margin": discovery,
            "projected_calibration_max_abs": projected_shift,
            "unprojected_calibration_max_abs": unprojected_shift,
            "denormal_probe_before": before, "denormal_probe_after": after,
            "provisional_numerical_decision": decision}


def evaluate(logits, public_case_document, development):
    """Score all six conditions; this is not a final study outcome."""
    return _evaluate(logits, _cases(public_case_document), development)
