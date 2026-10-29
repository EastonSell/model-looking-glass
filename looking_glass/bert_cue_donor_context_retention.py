"""Matched same-cue donor-context effects using the existing witness schedule."""
import math

from .bert_cue_context_transfer import CUES, SITES, donors, run_pair, same_class_other

PAIR_IDS = ("discovery-01", "evaluation-01", "evaluation-02", "evaluation-03")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def summarize_pair(row):
    """Compute all paired edges and preserve raw results even when ineligible."""
    if (type(row.get("baseline_eligible")) is not bool or type(row.get("dose_eligible")) is not bool
            or not _finite(row.get("q")) or not 0 <= row["q"] <= 0.5
            or row["dose_eligible"] != (row["q"] >= 0.05)):
        raise ValueError("invalid pair eligibility or dose")
    baselines = {}
    for item in row["baselines"]:
        if type(item.get("context")) is not int or item["context"] not in (0, 1):
            raise ValueError("baseline context must be an integer")
        key = item["context"], item["cue"]
        if key in baselines or not _finite(item["any_minus_some"]):
            raise ValueError("duplicate or nonfinite baseline")
        baselines[key] = item["any_minus_some"]
    if set(baselines) != {(ci, cue) for ci in range(2) for cue in CUES}:
        raise ValueError("all eight baselines required")
    required = {(site, ci, cue, di, donor) for site in SITES for ci in range(2)
                for cue in CUES for di, donor in donors(ci, cue)}
    distances, effects = {}, {}
    for item in row["distances"]:
        if (type(item.get("target_context")) is not int or item["target_context"] not in (0, 1) or
                type(item.get("donor_context")) is not int or item["donor_context"] not in (0, 1)):
            raise ValueError("distance contexts must be integers")
        key = (item["site"], item["target_context"], item["target_cue"],
               item["donor_context"], item["donor_cue"])
        if key in distances or not _finite(item["distance"]) or item["distance"] < 0:
            raise ValueError("duplicate or invalid distance")
        distances[key] = item["distance"]
    if set(distances) != required:
        raise ValueError("all 224 distances required")
    for item in row["effects"]:
        if (type(item.get("target_context")) is not int or item["target_context"] not in (0, 1) or
                type(item.get("donor_context")) is not int or item["donor_context"] not in (0, 1)):
            raise ValueError("effect contexts must be integers")
        key = (item["site"], item["target_context"], item["target_cue"],
               item["donor_context"], item["donor_cue"])
        if key in effects or not _finite(item["delta"]):
            raise ValueError("duplicate or nonfinite effect")
        effects[key] = item["delta"]
    if set(effects) != (required if row["dose_eligible"] else set()):
        raise ValueError("fixed donor effects differ")
    edges = []
    for ci in range(2):
        for cue in CUES:
            opposite = CUES[2:] if cue in CUES[:2] else CUES[:2]
            direction = -1 if cue in CUES[:2] else 1
            for donor in opposite:
                within_key = "cue2", ci, cue, ci, donor
                alternate_key = "cue2", ci, cue, 1 - ci, donor
                item = {"target_context": ci, "target_cue": cue, "donor_cue": donor,
                        "class_direction": direction,
                        "within_clean_gap": math.fsum((baselines[ci, donor], -baselines[ci, cue])),
                        "alternate_clean_gap": math.fsum((baselines[1-ci, donor], -baselines[ci, cue])),
                        "within_distance": distances[within_key], "alternate_distance": distances[alternate_key],
                        "within_delta": None, "alternate_delta": None, "within_signed": None,
                        "alternate_signed": None, "same_cue_alternate_delta": None,
                        "same_class_alternate_delta": None, "same_class_within_delta": None,
                        "informative": None, "exclusion": "dose_ineligible", "retained": None,
                        "sensitivity_retained": None, "half_difference": None, "full_difference": None}
                if row["dose_eligible"]:
                    within, alternate = effects[within_key], effects[alternate_key]
                    w, x = direction * within, direction * alternate
                    informative = w >= 0.02
                    item.update({"within_delta": within, "alternate_delta": alternate,
                                 "within_signed": w, "alternate_signed": x,
                                 "same_cue_alternate_delta": effects["cue2", ci, cue, 1-ci, cue],
                                 "same_class_alternate_delta": effects["cue2", ci, cue, 1-ci, same_class_other(cue)],
                                 "same_class_within_delta": effects["cue2", ci, cue, ci, same_class_other(cue)],
                                 "informative": informative,
                                 "exclusion": None if informative else "nonpositive" if w <= 0 else "below_floor",
                                 "retained": x >= 0.5 * w if informative else None,
                                 "sensitivity_retained": x >= w if informative else None,
                                 "half_difference": math.fsum((x, -0.5 * w)) if informative else None,
                                 "full_difference": math.fsum((x, -w)) if informative else None})
                edges.append(item)
    has_effects = row["dose_eligible"]
    informative = [item for item in edges if item["informative"] is True]
    n_informative = len(informative) if has_effects else None
    passed = sum(item["retained"] for item in informative) if has_effects else None
    stronger = sum(item["sensitivity_retained"] for item in informative) if has_effects else None
    evaluable = row["baseline_eligible"] and has_effects and row["q"] == 0.5 and len(informative) >= 12
    def decision(count):
        return "unevaluable" if not evaluable else "pass" if count >= 12 else "fail"
    return {"evaluable": evaluable, "edges": edges, "informative_edges": n_informative,
            "retained_edges": passed, "retained_per_16": passed / 16 if has_effects else None,
            "retained_per_informative": passed / n_informative if n_informative else None,
            "sensitivity_retained_edges": stronger,
            "sensitivity_per_16": stronger / 16 if has_effects else None,
            "sensitivity_per_informative": stronger / n_informative if n_informative else None,
            "primary_status": decision(passed), "sensitivity_status": decision(stronger)}


def evaluate(pairs):
    if ([row["id"] for row in pairs] != list(PAIR_IDS) or
            [row["split"] for row in pairs] != ["discovery", "evaluation", "evaluation", "evaluation"]):
        raise ValueError("fixed pair order differs")
    def aggregate(key):
        statuses = [row["retention"][key] for row in pairs[1:]]
        if any(status not in ("pass", "fail", "unevaluable") for status in statuses):
            raise ValueError("invalid pair status")
        evaluable, passed = sum(s != "unevaluable" for s in statuses), statuses.count("pass")
        return {"status": "unevaluable" if evaluable < 2 else "pass" if passed >= 2 else "fail",
                "evaluable_pairs": evaluable, "passing_pairs": passed}
    return {"primary": aggregate("primary_status"), "sensitivity": aggregate("sensitivity_status")}


def run_study(model, manifest, emit, *, expected_vocab=None, expected_width=None):
    if [row["id"] for row in manifest["pairs"]] != list(PAIR_IDS):
        raise ValueError("fixed pair list differs")
    results = []
    for pair in manifest["pairs"]:
        try:
            result = run_pair(model, pair, emit, any_id=manifest["scored_tokens"]["any"],
                              some_id=manifest["scored_tokens"]["some"],
                              expected_vocab=expected_vocab, expected_width=expected_width)
            result["retention"] = summarize_pair(result)
            results.append(result)
        except BaseException as exc:
            try:
                emit({"kind": "pair_failed", "pair_id": pair["id"], "error_type": type(exc).__name__})
            except Exception:
                pass
            raise
    return {"status": "complete", "forwards": sum(row["forwards"] for row in results),
            "pairs": results, "evaluation": evaluate(results)}
