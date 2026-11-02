"""Two fixed scientific phases for the FLAN input-row change study."""

import flan_weight_edit_decision as decisions
import flan_weight_edit_numerical as science
import flan_weight_edit_results as results

DOSES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
CONTROLS = ("broad", "random", "opposite")


def _effect(session, changed, clean):
    def calculate():
        a, b = changed["label_terms"], clean["label_terms"]
        return {"margin_gain": science.margin_effect(a["negative"], b["negative"],
                                                     a["positive"], b["positive"]),
                "target_gain": science.sequence_effect(a["negative"], b["negative"]),
                "negative": changed["strict_label"] == "negative"}
    return session.score("effect", calculate)


def _validate_cases(cases):
    ids = ([f"{phase}-{i}-{side}" for phase in ("discovery", "calibration")
            for i in (1, 2) for side in ("target", "donor")]
           + [f"evaluation-{i}-{side}" for i in range(1, 5) for side in ("target", "donor")]
           + [f"locality-{i}" for i in range(1, 9)])
    if len(cases) != 24 or [row["id"] for row in cases] != ids:
        raise ValueError("case order or identities differ from the fixed study")


def development(cases, session):
    _validate_cases(cases)
    baseline = {case["id"]: session.evaluate("baseline", case) for case in cases[:8]}
    def clean_supported(case):
        outcome = baseline[case["id"]]
        margin = session.score("clean-label-margin", lambda:
                               science.sequence_effect(outcome["label_terms"]["negative"],
                                                       outcome["label_terms"]["positive"]))
        return outcome["strict_correct"] and (margin < 0 if case["expected"] == "positive" else margin > 0)
    supported = [clean_supported(case) for case in cases[:8]]
    if not all(supported):
        return {"status": "ineligible", "reason": "initial-clean-baseline", "baseline": baseline,
                "clean_supported": dict(zip((case["id"] for case in cases[:8]), supported))}

    primary = {}
    with session.arm("primary", 1.0):
        for index in (0, 2, 4, 6):
            case, donor = cases[index:index + 2]
            outcome = session.evaluate("primary", case)
            decisions.check_substitution(outcome, baseline[donor["id"]])
            primary[case["id"]] = outcome
    calibration = cases[4:8:2]
    primary_effects = [_effect(session, primary[case["id"]], baseline[case["id"]]) for case in calibration]
    calibration_eligible = all(row["negative"] and row["margin_gain"] >= 0.5 and row["target_gain"] >= 0.5
                               for row in primary_effects)

    grids, selections = {}, {}
    for name in CONTROLS:
        grids[name] = []
        for dose in DOSES:
            outcomes = []
            with session.arm(name, dose):
                for case in calibration:
                    outcomes.append(session.evaluate(f"{name}@{dose}", case))
            effects = [_effect(session, outcome, baseline[case["id"]])
                       for outcome, case in zip(outcomes, calibration)]
            grids[name].append({"dose": dose, "outcomes": outcomes, "rows": effects})
        selections[name] = session.score("selection-" + name, lambda:
                                         science.select_control_fixed(primary_effects,
                                             [{"dose": point["dose"], "rows": point["rows"]}
                                              for point in grids[name]]))
    same = []
    with session.arm("same", 1.0):
        for case in calibration:
            same.append(session.evaluate("same", case))
    return {"status": "complete", "baseline": baseline, "primary": primary,
            "primary_effects": primary_effects, "calibration_eligible": calibration_eligible,
            "grids": grids, "selection": selections, "same": same}


def evaluation(cases, session, development_result):
    _validate_cases(cases)
    if (development_result.get("status") != "complete"
            or set(development_result.get("selection", {})) != set(CONTROLS)):
        raise ValueError("a complete, verified fixed development phase is required")
    baseline = {case["id"]: session.evaluate("baseline", case) for case in cases[8:]}
    targets = cases[8:16:2]
    localities = cases[16:]
    arms = {"primary": 1.0, "same": 1.0,
            **{name: development_result["selection"][name]["dose"] for name in CONTROLS}}
    outcomes = {}
    for name, dose in arms.items():
        outcomes[name] = {}
        with session.arm(name, dose):
            for case in [*targets, *localities]:
                outcome = session.evaluate(name, case)
                if name == "primary" and case in targets:
                    donor_id = case["id"].replace("-target", "-donor")
                    decisions.check_substitution(outcome, baseline[donor_id])
                if case in localities:
                    results.check_locality_identity(case, name, baseline[case["id"]], outcome)
                outcomes[name][case["id"]] = outcome

    rows, eligible_cases = [], []
    for case in targets:
        donor_id = case["id"].replace("-target", "-donor")
        eligible = (baseline[case["id"]]["strict_label"] == "positive"
                    and baseline[donor_id]["strict_label"] == "negative")
        if eligible:
            eligible_cases.append(case)
        rows.append({"case_id": case["id"], "baseline_eligible": eligible,
                     "edited_label": outcomes["primary"][case["id"]]["strict_label"]})
    primary = decisions.primary_outcome(rows, calibration_eligible=development_result["calibration_eligible"])
    comparisons = {}
    for name in CONTROLS:
        if not development_result["selection"][name]["matched"]:
            comparisons[name] = {"eligible": False, "reason": "calibration-unmatched"}
        elif primary["status"] != "positive":
            comparisons[name] = {"eligible": False, "reason": "primary-not-positive"}
        else:
            references, alternatives = [], []
            for case in eligible_cases:
                case_id = case["id"]
                p = _effect(session, outcomes["primary"][case_id], baseline[case_id])
                c = _effect(session, outcomes[name][case_id], baseline[case_id])
                references.append({**p, "strict_label": outcomes["primary"][case_id]["strict_label"]})
                alternatives.append({**c, "strict_label": outcomes[name][case_id]["strict_label"]})
            balanced = session.score("evaluation-balance-" + name,
                                     lambda: decisions.evaluation_balanced(references, alternatives))
            comparisons[name] = {"eligible": balanced, "reason": None if balanced else "evaluation-imbalance",
                                 "primary_effects": references, "control_effects": alternatives}
    return {"status": "complete", "primary": primary, "targets": rows,
            "localities": [case["id"] for case in localities], "baseline": baseline,
            "arms": outcomes, "comparisons": comparisons}
