"""Fixed two-frame cue study using the reviewed CPU intervention observer."""
import itertools
import math

import torch

from .bert_cue_class import _emit_witness_failure, _observed
from .bert_negation_cue import _bits, _margin, _same_bits, baseline_summary, distance, make_patch

CUES = ("not", "hardly", "really", "certainly")
NEGATIVE = ("not", "hardly")
AFFIRMATIVE = ("really", "certainly")
FRAMES = ("subject_relative", "causal_adjunct")
SITES = ("cue2", "object2", "mask2", "cue5")
PAIRS = tuple(itertools.combinations(CUES, 2))
TRANSFERS = tuple((donor, target) for target in CUES for donor in CUES if donor != target)


def baseline_screen(margins):
    if type(margins) is not dict or set(margins) != set(CUES):
        raise ValueError("four clean cue margins required")
    gaps = []
    for negative in NEGATIVE:
        for affirmative in AFFIRMATIVE:
            value = math.fsum((margins[negative], -margins[affirmative]))
            if not math.isfinite(value):
                raise ValueError("nonfinite baseline gap")
            gaps.append({"negative": negative, "affirmative": affirmative, "gap": value})
    return {"eligible": all(gap["gap"] > 0.5 for gap in gaps), "gaps": gaps}


def context_specificity(effects):
    if type(effects) is not dict or set(effects) != set(TRANSFERS):
        raise ValueError("all 12 directed effects are required")
    targets = {}
    for target in CUES:
        own = NEGATIVE if target in NEGATIVE else AFFIRMATIVE
        opposite = AFFIRMATIVE if target in NEGATIVE else NEGATIVE
        sign = -1 if target in NEGATIVE else 1
        same_donor = next(cue for cue in own if cue != target)
        cross = {cue: sign * effects[cue, target] for cue in opposite}
        same = effects[same_donor, target]
        if any(not math.isfinite(value) for value in (*cross.values(), same)):
            raise ValueError("nonfinite transfer effect")
        minimum = min(cross.values())
        margin = math.fsum((minimum, -abs(same)))
        targets[target] = {"cross_signed": cross, "cross_min": minimum, "same_donor": same_donor,
                           "same_delta": same, "same_abs": abs(same), "margin": margin,
                           "passed": margin > 0.05}
    return {"targets": targets, "passed": all(value["passed"] for value in targets.values())}


def evaluate(rows):
    if type(rows) is not list or len(rows) != 6:
        raise ValueError("the evaluation requires six fixed contexts")
    if (len({row["id"] for row in rows}) != 6 or any(row.get("split") != "evaluation" for row in rows)
            or any(row.get("frame") not in FRAMES for row in rows)):
        raise ValueError("evaluation case identities or frames differ")
    frames = {}
    for frame in FRAMES:
        members = [row for row in rows if row["frame"] == frame]
        if len(members) != 3:
            raise ValueError("each frame requires three evaluation contexts")
        eligible = [row for row in members if row["baseline_eligible"] and row["dose_eligible"]]
        passing = [row for row in eligible if row["specificity_passed"]]
        frames[frame] = {"total": 3, "eligible": len(eligible), "passing": len(passing)}
    if any(value["eligible"] < 2 for value in frames.values()):
        status = "unevaluable"
    elif all(value["passing"] >= 2 for value in frames.values()):
        status = "passed"
    else:
        status = "failed"
    return {"status": status, "frames": frames}


def choose_radius(captures):
    distances = [{"left": a, "right": b, "site": site,
                  "distance": distance(captures[a][site], captures[b][site])}
                 for a, b in PAIRS for site in SITES]
    return min(item["distance"] for item in distances), distances


def run_case(model, case, emit, *, any_id, some_id, expected_vocab=None, expected_width=None):
    case_id, indices = case["id"], case["indices"]
    rows = {cue: torch.tensor([case["rows"][cue]["input_ids"]], dtype=torch.int64) for cue in CUES}
    clean, captures, margins, baselines = {}, {}, {}, {}
    forwards = 0
    for cue in CUES:
        observed = _observed(model, rows[cue], indices, emit, {"case_id": case_id, "cue": cue},
                             expected_vocab=expected_vocab, expected_width=expected_width)
        forwards += 1
        clean[cue], captures[cue] = observed["mask_logits"], observed["captures"]
        emit({"kind": "clean", "case_id": case_id, "cue": cue, "input_ids": rows[cue][0].tolist(),
              "mask_logits_f32le": _bits(clean[cue]),
              "captures_f32le": {name: _bits(captures[cue][name]) for name in SITES}})
        baselines[cue] = baseline_summary(clean[cue], any_id=any_id, some_id=some_id)
        margins[cue] = baselines[cue]["any_minus_some"]
    baseline = baseline_screen(margins)
    emit({"kind": "baseline", "case_id": case_id, **baseline})
    radius, distances = choose_radius(captures)
    if not math.isfinite(radius):
        raise ValueError("nonfinite common radius")
    eligible_dose = radius >= 0.001
    emit({"kind": "radius", "case_id": case_id, "radius": radius, "distances": distances,
          "binding_edges": [item for item in distances if item["distance"] == radius],
          "dose_eligible": eligible_dose})
    all_effects = {}
    if not eligible_dose:
        emit({"kind": "dose_ineligible", "case_id": case_id, "radius": radius})
    else:
        for site in SITES:
            for cue in CUES:
                meta = {"case_id": case_id, "site": site, "target": cue}
                observed = _observed(model, rows[cue], indices, emit, meta, site=site, self_patch=True,
                                     expected_vocab=expected_vocab, expected_width=expected_width)
                forwards += 1
                emit({"kind": "self", **meta, "input_ids": rows[cue][0].tolist(),
                      **{key + "_f32le": _bits(value) for key, value in observed.items()}})
                if not _same_bits(observed["before"], captures[cue][site]) or not _same_bits(observed["mask_logits"], clean[cue]):
                    raise RuntimeError("self or clean-state control failed")
            site_effects = {}
            for donor, target in TRANSFERS:
                submitted, planned, fraction = make_patch(captures[target][site], captures[donor][site], radius)
                meta = {"case_id": case_id, "site": site, "donor": donor, "target": target}
                observed = _observed(model, rows[target], indices, emit, meta, site=site, source=submitted,
                                     expected_vocab=expected_vocab, expected_width=expected_width)
                forwards += 1
                try:
                    actual = distance(observed["before"], observed["after"])
                except BaseException:
                    _emit_witness_failure(emit, meta, rows[target], "dose_measurement", observed,
                                          submitted_f32le=_bits(submitted), requested_radius=radius,
                                          requested_fraction=fraction, planned_norm=planned)
                    raise
                emit({"kind": "transfer", **meta, "input_ids": rows[target][0].tolist(),
                      "requested_radius": radius, "requested_fraction": fraction, "planned_norm": planned,
                      "actual_norm": actual, "relative_error": abs(actual - radius) / radius,
                      "submitted_f32le": _bits(submitted),
                      **{key + "_f32le": _bits(value) for key, value in observed.items()}})
                if (not _same_bits(observed["before"], captures[target][site]) or
                        abs(actual - radius) / radius > 0.001 or
                        (site == "cue5" and not _same_bits(observed["mask_logits"], clean[target]))):
                    raise RuntimeError("transfer witness, dose or final-cue control failed")
                patched = _margin(observed["mask_logits"], any_id, some_id)
                site_effects[donor, target] = math.fsum((patched, -margins[target]))
            all_effects[site] = site_effects
    for cue in CUES:
        observed = _observed(model, rows[cue], indices, emit, {"case_id": case_id, "cue": cue},
                             expected_vocab=expected_vocab, expected_width=expected_width)
        forwards += 1
        emit({"kind": "repeat", "case_id": case_id, "cue": cue, "input_ids": rows[cue][0].tolist(),
              "mask_logits_f32le": _bits(observed["mask_logits"]),
              "captures_f32le": {name: _bits(observed["captures"][name]) for name in SITES}})
        if (not _same_bits(observed["mask_logits"], clean[cue]) or
                any(not _same_bits(observed["captures"][name], captures[cue][name]) for name in SITES)):
            raise RuntimeError("post-intervention clean repeat differs")
    if forwards != (72 if eligible_dose else 8):
        raise RuntimeError("case schedule incomplete")
    specificity = {site: context_specificity(all_effects[site]) for site in SITES} if eligible_dose else {}
    return {"id": case_id, "split": case["split"], "frame": case["frame"], "forwards": forwards,
            "baseline_eligible": baseline["eligible"], "baseline_gaps": baseline["gaps"],
            "dose_eligible": eligible_dose, "radius": radius, "distances": distances,
            "clean_margins": margins, "baselines": baselines,
            "effects": {site: [{"donor": donor, "target": target, "delta": values[donor, target]}
                               for donor, target in TRANSFERS] for site, values in all_effects.items()},
            "specificity": specificity,
            "specificity_passed": specificity["cue2"]["passed"] if eligible_dose else None}


def run_study(model, manifest, emit, *, expected_vocab=None, expected_width=None):
    cases = manifest["cases"]
    if (len(cases) != 6 or any(row["split"] != "evaluation" for row in cases)
            or len({row["id"] for row in cases}) != 6):
        raise ValueError("fixed case list differs")
    results = []
    for case in cases:
        try:
            results.append(run_case(model, case, emit, any_id=manifest["scored_tokens"]["any"],
                                    some_id=manifest["scored_tokens"]["some"],
                                    expected_vocab=expected_vocab, expected_width=expected_width))
        except BaseException as exc:
            try:
                emit({"kind": "case_failed", "case_id": case["id"], "error_type": type(exc).__name__})
            except Exception:
                pass
            raise
    decision = evaluate(results)
    return {"status": "complete", "forwards": sum(row["forwards"] for row in results),
            "cases": results, "evaluation": decision}
