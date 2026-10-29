"""Fixed CPU cue-set by clause-placement study with matched intervention norms."""
import itertools
import math

import torch

from .bert_cue_class import _emit_witness_failure, _observed
from .bert_negation_cue import _bits, _margin, _same_bits, baseline_summary, distance, make_patch

GROUPS = {"prior": (("never", "rarely"), ("always", "often")),
          "new": (("not", "hardly"), ("really", "certainly"))}
FRAMES = ("main", "subject_relative")
TRIPLES = ("triple-01", "triple-02", "triple-03")
SITES = ("cue2", "object2", "mask2", "cue5")
RADIUS = 0.5


def baseline_screen(margins, cue_set):
    negative, affirmative = GROUPS[cue_set]
    if type(margins) is not dict or set(margins) != set(negative + affirmative):
        raise ValueError("four clean cue margins required")
    gaps = []
    for n in negative:
        for a in affirmative:
            value = math.fsum((margins[n], -margins[a]))
            if not math.isfinite(value):
                raise ValueError("nonfinite baseline gap")
            gaps.append({"negative": n, "affirmative": a, "gap": value})
    return {"eligible": all(gap["gap"] > 0.5 for gap in gaps), "gaps": gaps}


def context_specificity(effects, cue_set):
    negative, affirmative = GROUPS[cue_set]
    cues = negative + affirmative
    transfers = {(donor, target) for target in cues for donor in cues if donor != target}
    if type(effects) is not dict or set(effects) != transfers:
        raise ValueError("all 12 directed effects are required")
    targets = {}
    for target in cues:
        own = negative if target in negative else affirmative
        opposite = affirmative if target in negative else negative
        sign = -1 if target in negative else 1
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


def dose_eligible(minimum_distance):
    if not math.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("nonfinite or negative cue distance")
    return minimum_distance >= RADIUS


def choose_distances(captures, cue_set):
    cues = sum(GROUPS[cue_set], ())
    rows = [{"left": a, "right": b, "site": site,
             "distance": distance(captures[a][site], captures[b][site])}
            for a, b in itertools.combinations(cues, 2) for site in SITES]
    if any(not math.isfinite(item["distance"]) or item["distance"] < 0 for item in rows):
        raise ValueError("nonfinite or negative cue distance")
    return min(item["distance"] for item in rows), rows


def _validate_rows(rows):
    if type(rows) is not list or len(rows) != 12:
        raise ValueError("the evaluation requires twelve fixed cells")
    keyed = {}
    for row in rows:
        key = (row.get("triple_id"), row.get("frame"), row.get("cue_set"))
        if (key[0] not in TRIPLES or key[1] not in FRAMES or key[2] not in GROUPS or key in keyed
                or row.get("split") != "evaluation" or type(row.get("id")) is not str
                or type(row.get("baseline_eligible")) is not bool or type(row.get("dose_eligible")) is not bool):
            raise ValueError("evaluation cell identity or eligibility differs")
        score, passed = row.get("cue_min_margin"), row.get("specificity_passed")
        if row["dose_eligible"]:
            if (type(score) not in (float, int) or not math.isfinite(score) or type(passed) is not bool
                    or passed != (score > 0.05)):
                raise ValueError("invalid cue outcome")
        elif score is not None or passed is not None:
            raise ValueError("dose-ineligible cell cannot have a specificity outcome")
        keyed[key] = row
    if len({row["id"] for row in rows}) != 12:
        raise ValueError("duplicate evaluation cell ID")
    return keyed


def _eligible(row):
    return row["baseline_eligible"] and row["dose_eligible"]


def _contrast(pairs):
    eligible = [{"triple_id": triple, "value": math.fsum((a["cue_min_margin"], -b["cue_min_margin"]))}
                for triple, a, b in pairs if _eligible(a) and _eligible(b)]
    count = len(eligible)
    return {"status": "evaluable" if count >= 2 else "unevaluable", "eligible": count,
            "values": eligible, "mean": math.fsum(item["value"] for item in eligible) / count if count >= 2 else None}


def evaluate(rows):
    keyed = _validate_rows(rows)
    cells = {}
    for group in GROUPS:
        for frame in FRAMES:
            candidates = [keyed[t, frame, group] for t in TRIPLES]
            eligible = [row for row in candidates if _eligible(row)]
            passing = sum(row["specificity_passed"] for row in eligible)
            status = "unevaluable" if len(eligible) < 2 else ("passed" if passing >= 2 else "failed")
            cells[group + "/" + frame] = {"total": 3, "eligible": len(eligible), "passing": passing, "status": status}
    cues = {frame: _contrast([(t, keyed[t, frame, "new"], keyed[t, frame, "prior"]) for t in TRIPLES])
            for frame in FRAMES}
    frames = {group: _contrast([(t, keyed[t, "subject_relative", group], keyed[t, "main", group]) for t in TRIPLES])
              for group in GROUPS}
    values = []
    for t in TRIPLES:
        nr, pr = keyed[t, "subject_relative", "new"], keyed[t, "subject_relative", "prior"]
        nm, pm = keyed[t, "main", "new"], keyed[t, "main", "prior"]
        if all(_eligible(row) for row in (nr, pr, nm, pm)):
            value = math.fsum((nr["cue_min_margin"], -pr["cue_min_margin"],
                               -nm["cue_min_margin"], pm["cue_min_margin"]))
            values.append({"triple_id": t, "value": value})
    interaction = {"status": "evaluable" if len(values) >= 2 else "unevaluable", "eligible": len(values),
                   "values": values, "mean": math.fsum(v["value"] for v in values) / len(values)
                   if len(values) >= 2 else None}
    return {"cells": cells, "cue_contrasts": cues, "frame_contrasts": frames, "interaction": interaction}


def run_case(model, case, emit, *, any_id, some_id, expected_vocab=None, expected_width=None):
    cue_set = case["cue_set"]
    cues = sum(GROUPS[cue_set], ())
    transfers = tuple((donor, target) for target in cues for donor in cues if donor != target)
    case_id, indices = case["id"], case["indices"]
    rows = {cue: torch.tensor([case["rows"][cue]["input_ids"]], dtype=torch.int64) for cue in cues}
    clean, captures, margins, baselines = {}, {}, {}, {}
    forwards = 0
    for cue in cues:
        observed = _observed(model, rows[cue], indices, emit, {"case_id": case_id, "cue": cue},
                             expected_vocab=expected_vocab, expected_width=expected_width)
        forwards += 1
        clean[cue], captures[cue] = observed["mask_logits"], observed["captures"]
        emit({"kind": "clean", "case_id": case_id, "cue": cue, "input_ids": rows[cue][0].tolist(),
              "mask_logits_f32le": _bits(clean[cue]),
              "captures_f32le": {name: _bits(captures[cue][name]) for name in SITES}})
        baselines[cue] = baseline_summary(clean[cue], any_id=any_id, some_id=some_id)
        margins[cue] = baselines[cue]["any_minus_some"]
    baseline = baseline_screen(margins, cue_set)
    emit({"kind": "baseline", "case_id": case_id, **baseline})
    minimum, distances = choose_distances(captures, cue_set)
    eligible_dose = dose_eligible(minimum)
    emit({"kind": "radius", "case_id": case_id, "radius": RADIUS, "minimum_distance": minimum,
          "distances": distances, "binding_edges": [item for item in distances if item["distance"] == minimum],
          "dose_eligible": eligible_dose})
    all_effects = {}
    if not eligible_dose:
        emit({"kind": "dose_ineligible", "case_id": case_id, "radius": RADIUS, "minimum_distance": minimum})
    else:
        for site in SITES:
            for cue in cues:
                meta = {"case_id": case_id, "site": site, "target": cue}
                observed = _observed(model, rows[cue], indices, emit, meta, site=site, self_patch=True,
                                     expected_vocab=expected_vocab, expected_width=expected_width)
                forwards += 1
                emit({"kind": "self", **meta, "input_ids": rows[cue][0].tolist(),
                      **{key + "_f32le": _bits(value) for key, value in observed.items()}})
                if not _same_bits(observed["before"], captures[cue][site]) or not _same_bits(observed["mask_logits"], clean[cue]):
                    raise RuntimeError("self or clean-state control failed")
            site_effects = {}
            for donor, target in transfers:
                submitted, planned, fraction = make_patch(captures[target][site], captures[donor][site], RADIUS)
                meta = {"case_id": case_id, "site": site, "donor": donor, "target": target}
                observed = _observed(model, rows[target], indices, emit, meta, site=site, source=submitted,
                                     expected_vocab=expected_vocab, expected_width=expected_width)
                forwards += 1
                try:
                    actual = distance(observed["before"], observed["after"])
                except BaseException:
                    _emit_witness_failure(emit, meta, rows[target], "dose_measurement", observed,
                                          submitted_f32le=_bits(submitted), requested_radius=RADIUS,
                                          requested_fraction=fraction, planned_norm=planned)
                    raise
                emit({"kind": "transfer", **meta, "input_ids": rows[target][0].tolist(),
                      "requested_radius": RADIUS, "requested_fraction": fraction, "planned_norm": planned,
                      "actual_norm": actual, "relative_error": abs(actual - RADIUS) / RADIUS,
                      "submitted_f32le": _bits(submitted),
                      **{key + "_f32le": _bits(value) for key, value in observed.items()}})
                if (not _same_bits(observed["before"], captures[target][site]) or
                        abs(actual - RADIUS) / RADIUS > 0.001 or
                        (site == "cue5" and not _same_bits(observed["mask_logits"], clean[target]))):
                    raise RuntimeError("transfer witness, dose or final-cue control failed")
                patched = _margin(observed["mask_logits"], any_id, some_id)
                site_effects[donor, target] = math.fsum((patched, -margins[target]))
            all_effects[site] = site_effects
    for cue in cues:
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
    specificity = {site: context_specificity(all_effects[site], cue_set) for site in SITES} if eligible_dose else {}
    cue_min = min(item["margin"] for item in specificity["cue2"]["targets"].values()) if eligible_dose else None
    limiting = [cue for cue in cues if specificity["cue2"]["targets"][cue]["margin"] == cue_min] if eligible_dose else []
    return {"id": case_id, "split": case["split"], "triple_id": case["triple_id"], "frame": case["frame"],
            "cue_set": cue_set, "forwards": forwards, "baseline_eligible": baseline["eligible"],
            "baseline_gaps": baseline["gaps"], "dose_eligible": eligible_dose, "radius": RADIUS,
            "minimum_distance": minimum, "distances": distances, "clean_margins": margins, "baselines": baselines,
            "effects": {site: [{"donor": donor, "target": target, "delta": values[donor, target]}
                               for donor, target in transfers] for site, values in all_effects.items()},
            "specificity": specificity, "cue_min_margin": cue_min, "limiting_targets": limiting,
            "specificity_passed": specificity["cue2"]["passed"] if eligible_dose else None}


def run_study(model, manifest, emit, *, expected_vocab=None, expected_width=None):
    cases = manifest["cases"]
    identities = [(row.get("triple_id"), row.get("frame"), row.get("cue_set")) for row in cases]
    expected = [(triple, frame, group) for triple in TRIPLES for frame in FRAMES for group in GROUPS]
    if (identities != expected or any(row.get("split") != "evaluation" for row in cases)
            or len({row.get("id") for row in cases}) != 12):
        raise ValueError("fixed case list or order differs")
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
    return {"status": "complete", "forwards": sum(row["forwards"] for row in results),
            "cases": results, "evaluation": evaluate(results)}
