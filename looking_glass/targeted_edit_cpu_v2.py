"""Fixed scoring and dose selection for the prospective CPU v2 edit study.

The model-execution and evidence-publication parts of this protocol are not yet
implemented. No model is loaded by the functions in this module.
"""

from __future__ import annotations

import math

import numpy as np

from . import targeted_edit_cpu_v1 as v1


VERSION = "targeted-edit-cpu-v2"
RANDOM_SEED = 20261030
GRID = (("dose-000025", 0.00025), ("dose-00005", 0.0005),
        ("dose-0001", 0.001), ("dose-0002", 0.002), ("dose-0004", 0.004))
EVALUATION_CONDITIONS = ("original", "selected", "random", "opposite", "v1_1_percent")

# These rows are public proposals; the study has not executed them.
EVALUATION_CASES = (
    ("eval-target-1", "target", "The national capital of France is", (464, 2260, 3139, 286, 4881, 318), None),
    ("eval-target-2", "target", "France's capital is the city of", (28572, 338, 3139, 318, 262, 1748, 286), None),
    ("eval-target-3", "target", "France's seat of government is in", (28572, 338, 5852, 286, 1230, 318, 287), None),
    ("eval-target-4", "target", "The French capital city is", (464, 4141, 3139, 1748, 318), None),
    ("eval-locality-1", "locality", "The capital of China is", (464, 3139, 286, 2807, 318), (" Beijing", 11618)),
    ("eval-locality-2", "locality", "The capital of Russia is", (464, 3139, 286, 3284, 318), (" Moscow", 9070)),
    ("eval-locality-3", "locality", "The capital of Canada is", (464, 3139, 286, 3340, 318), (" Ottawa", 14074)),
    ("eval-locality-4", "locality", "The capital of Portugal is", (464, 3139, 286, 19101, 318), (" Lisbon", 41898)),
    ("eval-locality-5", "locality", "The opposite of large is", (464, 6697, 286, 1588, 318), (" small", 1402)),
    ("eval-locality-6", "locality", "An hour contains sixty", (2025, 1711, 4909, 24742), (" minutes", 2431)),
    ("eval-locality-7", "locality", "A triangle has three", (32, 22950, 468, 1115), (" sides", 5389)),
    ("eval-locality-8", "locality", "The sun rises in the", (464, 4252, 16736, 287, 262), (" east", 7627)),
    ("eval-locality-9", "locality", "In France, the predominant language is", (818, 4881, 11, 262, 45718, 3303, 318), (" French", 4141)),
)


def check_split_rows():
    """Check complete-row duplicates/prefixes across fixed public study rows.

    The public exposure inventory and other custodians require separate checks.
    """
    cases = (*v1.CASES, *EVALUATION_CASES)
    identifiers = [case[0] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise v1.StudyError("fixed case identifiers are not unique")
    rows = [tuple(case[3]) for case in cases]
    if any(not row or any(type(token) is not int or token < 0 or token >= v1.VOCAB_SIZE
                          for token in row) for row in rows):
        raise v1.StudyError("invalid fixed token row")
    for index, row in enumerate(rows):
        for other in rows[index + 1:]:
            shorter = min(len(row), len(other))
            if row[:shorter] == other[:shorter]:
                raise v1.StudyError("fixed complete rows collide or are prefix-related")
    return len(rows)


def _score(array):
    """Use shifted float64 log-softmax without adding a tiny term to a huge max."""
    z = array.astype(np.float64)
    shifted = z - float(np.max(z))
    logp = shifted - float(np.log(np.sum(np.exp(shifted), dtype=np.float64)))
    return z, logp, int(np.argmax(z))


def _scores(logits, cases, names):
    """Calculate scores from private copies of fixed-size raw matrices.

    Callers must keep inputs quiescent until copying finishes: copying is not
    atomic against a concurrent writer.
    """
    v1._denormal_probe()
    if type(logits) is not dict or set(logits) != set(names):
        raise v1.StudyError("condition set differs from the fixed protocol")
    shape = (len(cases), v1.VOCAB_SIZE)
    snapshots = {}
    for name, array in logits.items():
        if (not isinstance(array, np.ndarray) or array.shape != shape or
                array.dtype != np.dtype("<f4")):
            raise v1.StudyError("invalid or nonfinite fixed-shape logit matrix")
        snapshot = np.array(array, copy=True, order="C")
        if not np.isfinite(snapshot).all():
            raise v1.StudyError("invalid or nonfinite fixed-shape logit matrix")
        snapshots[name] = snapshot
    return {name: [_score(row) for row in snapshots[name]] for name in names}


def _condition(scores, cases, name):
    baseline, edited = scores["original"], scores[name]
    targets, locality = [], []
    discovery = None
    for index, (case_id, split, _text, _ids, label) in enumerate(cases):
        base_z, base_logp, base_top = baseline[index]
        edit_z, edit_logp, edit_top = edited[index]
        if split in ("target", "discovery"):
            base_margin = float(base_z[v1.LONDON] - base_z[v1.PARIS])
            edit_margin = float(edit_z[v1.LONDON] - edit_z[v1.PARIS])
            row = {"case_id": case_id, "original_margin": base_margin,
                   "condition_margin": edit_margin, "change": edit_margin - base_margin,
                   "originally_negative": base_margin < 0,
                   "reversed": base_margin < 0 and edit_margin > 0}
            if split == "discovery":
                discovery = row
            else:
                targets.append(row)
        elif split == "locality":
            token_id = label[1]
            locality.append({"case_id": case_id, "original_top": base_top,
                             "condition_top": edit_top,
                             "kl": float(np.sum(np.exp(base_logp) * (base_logp - edit_logp),
                                                dtype=np.float64)),
                             "original_nll": float(-base_logp[token_id]),
                             "condition_nll": float(-edit_logp[token_id])})
        else:
            raise v1.StudyError("unknown fixed case role")
    if len(targets) != 4 or len(locality) not in (8, 9):
        raise v1.StudyError("fixed case count differs from the protocol")
    return {"discovery": discovery, "targets": targets, "locality": locality,
            "mean_target_change": float(sum(row["change"] for row in targets) / 4),
            "originally_negative_targets": sum(row["originally_negative"] for row in targets),
            "reversed_targets": sum(row["reversed"] for row in targets),
            "all_top_preserved": all(row["original_top"] == row["condition_top"] for row in locality),
            "mean_kl": float(sum(row["kl"] for row in locality) / len(locality)),
            "max_kl": float(max(row["kl"] for row in locality)),
            "mean_nll_deterioration": float(sum(row["condition_nll"] - row["original_nll"]
                                                for row in locality) / len(locality))}


def _locality_criteria(summary):
    return {"all_locality_top_preserved": summary["all_top_preserved"],
            "mean_locality_kl": summary["mean_kl"] <= 0.02,
            "max_locality_kl": summary["max_kl"] <= 0.10,
            "mean_locality_nll_deterioration": summary["mean_nll_deterioration"] <= 0.05}


def select_development(logits):
    """Recompute every dose's scores and select the declared dose or fallback."""
    names = ("original", *(name for name, _ in GRID))
    scores = _scores(logits, v1.CASES, names)
    summaries = {name: _condition(scores, v1.CASES, name) for name, _ in GRID}
    criteria = {}
    for name, _ in GRID:
        row = summaries[name]
        criteria[name] = {"baseline_discovery_negative": row["discovery"]["original_margin"] < 0,
                          "baseline_three_targets_negative": row["originally_negative_targets"] >= 3,
                          "discovery_reversal": row["discovery"]["condition_margin"] > 0,
                          "at_least_one_target_reversed": row["reversed_targets"] >= 1,
                          "mean_target_gain": row["mean_target_change"] > 0.5,
                          **_locality_criteria(row)}
    qualifying = [(name, dose) for name, dose in GRID if all(criteria[name].values())]
    if qualifying:
        selected, dose = max(qualifying, key=lambda pair: (summaries[pair[0]]["mean_target_change"], -pair[1]))
        qualified = True
    else:
        selected, dose = GRID[0]
        qualified = False
    return {"version": VERSION, "conditions": summaries, "criteria": criteria,
            "selected_condition": selected, "selected_relative_norm": dose,
            "development_qualified": qualified,
            "point_prediction": summaries[selected]["mean_target_change"],
            "sign_prediction": "positive"}


def verify_development_selection(logits, recorded_selection):
    """Fail closed if a recorded selection differs from all six saved matrices.

    A caller must additionally verify each matrix's committed artifact identity
    before using this result to authorize any evaluation forward.
    """
    if type(recorded_selection) is not dict:
        raise v1.StudyError("invalid recorded selection")
    derived = select_development(logits)
    try:
        matches = v1._canonical(recorded_selection) == v1._canonical(derived)
    except (TypeError, ValueError, RecursionError, OverflowError, UnicodeError) as exc:
        raise v1.StudyError("invalid recorded selection") from exc
    if not matches:
        raise v1.StudyError("recorded selection differs from development logits")
    return derived


def evaluate(logits, selection):
    """Recompute provisional numerical metrics after selection is verified.

    The returned decision cannot assess provenance, execution, artifact
    completeness or cleanup. The study owner and auditor must verify those
    gates before assigning a final outcome.
    """
    if type(selection) is not dict or type(selection.get("development_qualified")) is not bool:
        raise v1.StudyError("missing validated development selection")
    selected_name = selection.get("selected_condition")
    expected_doses = dict(GRID)
    if (type(selected_name) is not str or selected_name not in expected_doses or
            type(selection.get("selected_relative_norm")) is not float or
            selection["selected_relative_norm"] != expected_doses[selected_name] or
            (not selection["development_qualified"] and selected_name != GRID[0][0]) or
            selection.get("sign_prediction") != "positive" or
            type(selection.get("point_prediction")) is not float or
            not math.isfinite(selection["point_prediction"])):
        raise v1.StudyError("invalid selected dose or prediction")
    scores = _scores(logits, EVALUATION_CASES, EVALUATION_CONDITIONS)
    summaries = {name: _condition(scores, EVALUATION_CASES, name)
                 for name in EVALUATION_CONDITIONS if name != "original"}
    selected, random = summaries["selected"], summaries["random"]
    eligible = selected["originally_negative_targets"] >= 3
    criteria = {"development_qualified": selection["development_qualified"],
                "baseline_three_targets_negative": eligible,
                "at_least_two_targets_reversed": selected["reversed_targets"] >= 2,
                "mean_target_gain": selected["mean_target_change"] > 0.5,
                "random_control_advantage": selected["mean_target_change"] - random["mean_target_change"] >= 0.5,
                **_locality_criteria(selected)}
    changes = [row["change"] for row in selected["targets"]]
    point = selection["point_prediction"]
    v1_limits = {"baseline_three_targets_negative": eligible,
                 "discovery_check": "development-row-only",
                 "at_least_three_targets_reversed": selected["reversed_targets"] >= 3,
                 "mean_target_gain_above_one": selected["mean_target_change"] > 1.0,
                 "random_control_advantage": criteria["random_control_advantage"],
                 **_locality_criteria(selected)}
    return {"version": VERSION, "conditions": summaries, "criteria": criteria,
            "development_qualified": selection["development_qualified"],
            "v1_scalar_thresholds_on_new_sample": v1_limits,
            "prediction": {"sign": "positive", "point": point, "observed_changes": changes,
                           "strictly_positive_count": sum(value > 0 for value in changes),
                           "denominator": 4, "mean_absolute_error": float(sum(abs(value - point)
                                                                               for value in changes) / 4)},
            "decision": ("ineligible" if not eligible else
                         ("success" if all(criteria.values()) else "negative"))}
