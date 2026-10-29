"""Prospective signed summaries of the existing matched context transfers."""
import math

from .bert_cue_context_transfer import CUES, SITES, donors, run_pair, same_class_other

PAIR_IDS = ("discovery-01", "evaluation-01", "evaluation-02", "evaluation-03")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def summarize_pair(row):
    """Keep all raw recipient outcomes, including baseline/dose ineligible ones."""
    if (type(row.get("baseline_eligible")) is not bool or type(row.get("dose_eligible")) is not bool
            or not _finite(row.get("q")) or not 0 <= row["q"] <= 0.5
            or row["dose_eligible"] != (row["q"] >= 0.05)):
        raise ValueError("invalid pair eligibility or dose")
    baselines = {}
    for item in row["baselines"]:
        key = (item["context"], item["cue"])
        if key in baselines or not _finite(item["any_minus_some"]):
            raise ValueError("duplicate or nonfinite baseline")
        baselines[key] = item["any_minus_some"]
    if set(baselines) != {(ci, cue) for ci in range(2) for cue in CUES}:
        raise ValueError("all eight baselines required")
    effects = {}
    for item in row["effects"]:
        key = (item["site"], item["target_context"], item["target_cue"],
               item["donor_context"], item["donor_cue"])
        if key in effects or not _finite(item["delta"]):
            raise ValueError("duplicate or nonfinite effect")
        effects[key] = item["delta"]
    required = {(site, ci, cue, di, donor) for site in SITES for ci in range(2)
                for cue in CUES for di, donor in donors(ci, cue)}
    if set(effects) != (required if row["dose_eligible"] else set()):
        raise ValueError("fixed donor effects differ")
    eligible = row["baseline_eligible"] and row["dose_eligible"] and row["q"] == 0.5
    recipients = []
    for ci in range(2):
        for cue in CUES:
            other = 1 - ci
            opposite = CUES[2:] if cue in CUES[:2] else CUES[:2]
            direction = -1 if cue in CUES[:2] else 1
            gap = math.fsum((baselines[other, cue], -baselines[ci, cue]))
            opposite_gaps = {name: math.fsum((baselines[other, name], -baselines[ci, cue]))
                             for name in opposite}
            item = {"target_context": ci, "target_cue": cue, "same_cue_gap": gap,
                    "opposite_clean_gaps": opposite_gaps, "class_direction": direction,
                    "gap_qualified": abs(gap) >= 0.5, "same_cue_delta": None,
                    "same_class_cue": same_class_other(cue),
                    "same_class_clean_gap": math.fsum((baselines[other, same_class_other(cue)],
                                                        -baselines[ci, cue])),
                    "same_class_delta": None,
                    "opposite_delta": None, "opposite_class_directed": None,
                    "limiting_opposite": None, "limiting_control": None,
                    "primary_margin": None, "secondary_donorward": None,
                    "secondary_excess": None}
            if row["dose_eligible"]:
                s = effects["cue2", ci, cue, other, cue]
                c = effects["cue2", ci, cue, other, same_class_other(cue)]
                opposite_effects = {name: effects["cue2", ci, cue, other, name] for name in opposite}
                directed = {name: direction * value for name, value in opposite_effects.items()}
                smallest = min(directed.values())
                control = max(abs(s), abs(c))
                margin = math.fsum((smallest, -control))
                donorward = (1 if gap > 0 else -1) * s if item["gap_qualified"] else None
                excess = math.fsum((donorward, -abs(c))) if donorward is not None else None
                item.update({"same_cue_delta": s, "same_class_delta": c,
                             "opposite_delta": opposite_effects, "opposite_class_directed": directed,
                             "limiting_opposite": [name for name in opposite if directed[name] == smallest],
                             "limiting_control": [name for name, value in
                                                  (("same_cue", abs(s)), ("same_class", abs(c)))
                                                  if value == control],
                             "primary_margin": margin, "secondary_donorward": donorward,
                             "secondary_excess": excess})
            recipients.append(item)
    has_effects = row["dose_eligible"]
    raw_pass = all(item["primary_margin"] > 0 for item in recipients) if has_effects else None
    raw_strict_pass = all(item["primary_margin"] > 0.05 for item in recipients) if has_effects else None
    qualified = [item for item in recipients if item["gap_qualified"]]
    secondary_ok = eligible and len(qualified) >= 6
    def pair_status(raw):
        return ("unevaluable" if not eligible else "pass" if raw else "fail")
    secondary = {"status": "evaluable" if secondary_ok else "unevaluable",
                 "qualified_recipients": len(qualified),
                 "raw_toward_count": sum(item["secondary_donorward"] > 0 for item in qualified) if has_effects else None,
                 "raw_exceed_count": sum(item["secondary_excess"] > 0 for item in qualified) if has_effects else None,
                 "toward_count": sum(item["secondary_donorward"] > 0 for item in qualified) if secondary_ok else None,
                 "exceed_count": sum(item["secondary_excess"] > 0 for item in qualified) if secondary_ok else None,
                 "mean_donorward": math.fsum(item["secondary_donorward"] for item in qualified) / len(qualified) if secondary_ok else None,
                 "mean_excess": math.fsum(item["secondary_excess"] for item in qualified) / len(qualified) if secondary_ok else None}
    return {"eligible": eligible, "recipients": recipients,
            "primary_raw_pass": raw_pass, "primary_status": pair_status(raw_pass),
            "sensitivity_raw_pass": raw_strict_pass, "sensitivity_status": pair_status(raw_strict_pass),
            "secondary": secondary}


def evaluate(pairs):
    if ([row["id"] for row in pairs] != list(PAIR_IDS) or
            [row["split"] for row in pairs] != ["discovery", "evaluation", "evaluation", "evaluation"]):
        raise ValueError("fixed pair order differs")
    evaluation = pairs[1:]
    def aggregate(key):
        statuses = [row["signed"][key] for row in evaluation]
        if any(status not in ("pass", "fail", "unevaluable") for status in statuses):
            raise ValueError("invalid pair decision")
        eligible = sum(status != "unevaluable" for status in statuses)
        passed = statuses.count("pass")
        return {"status": "unevaluable" if eligible < 2 else "pass" if passed >= 2 else "fail",
                "eligible_pairs": eligible, "passing_pairs": passed}
    included = [row["signed"]["secondary"] for row in evaluation
                if row["signed"]["secondary"]["status"] == "evaluable"]
    if any(row["signed"]["secondary"]["status"] not in ("evaluable", "unevaluable") for row in evaluation):
        raise ValueError("invalid secondary status")
    secondary = {"status": "evaluable" if included else "unevaluable", "evaluable_pairs": len(included),
                 "qualified_recipients": sum(item["qualified_recipients"] for item in included),
                 "toward_count": sum(item["toward_count"] for item in included) if included else None,
                 "exceed_count": sum(item["exceed_count"] for item in included) if included else None}
    return {"primary": aggregate("primary_status"), "sensitivity": aggregate("sensitivity_status"),
            "secondary": secondary}


def run_study(model, manifest, emit, *, expected_vocab=None, expected_width=None):
    if [row["id"] for row in manifest["pairs"]] != list(PAIR_IDS):
        raise ValueError("fixed pair list differs")
    results = []
    for pair in manifest["pairs"]:
        try:
            result = run_pair(model, pair, emit, any_id=manifest["scored_tokens"]["any"],
                              some_id=manifest["scored_tokens"]["some"],
                              expected_vocab=expected_vocab, expected_width=expected_width)
            result["signed"] = summarize_pair(result)
            results.append(result)
        except BaseException as exc:
            try:
                emit({"kind": "pair_failed", "pair_id": pair["id"], "error_type": type(exc).__name__})
            except Exception:
                pass
            raise
    return {"status": "complete", "forwards": sum(row["forwards"] for row in results),
            "pairs": results, "evaluation": evaluate(results)}
