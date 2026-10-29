"""Fixed two-dose CPU cue-specificity study using the reviewed observer."""
import math

import torch

from .bert_cue_class import _emit_witness_failure, _observed
from .bert_cue_factorial import GROUPS, SITES, baseline_screen, choose_distances, context_specificity
from .bert_negation_cue import _bits, _margin, _same_bits, baseline_summary, distance, make_patch

DOSES = {"low": 0.5, "high": 0.7}
TRIPLES = ("fresh-01", "fresh-02", "fresh-03")
CALIBRATION = "exposed-calibration"


def dose_eligibility(minimum_distance):
    if not math.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("nonfinite or negative cue distance")
    return {label: minimum_distance >= radius for label, radius in DOSES.items()}


def _validate_rows(rows):
    if type(rows) is not list or len(rows) != 8:
        raise ValueError("the evaluation requires all eight fixed cells")
    keyed = {}
    for row in rows:
        triple, group = row.get("triple_id"), row.get("cue_set")
        key = (triple, group)
        if (triple not in (CALIBRATION, *TRIPLES) or group not in GROUPS or key in keyed
                or row.get("split") != ("calibration" if triple == CALIBRATION else "evaluation")
                or row.get("frame") != "main" or type(row.get("id")) is not str
                or type(row.get("baseline_eligible")) is not bool or type(row.get("doses")) is not dict
                or set(row["doses"]) != set(DOSES)):
            raise ValueError("fixed cell identity or eligibility differs")
        for label in DOSES:
            dose = row["doses"][label]
            if type(dose) is not dict or type(dose.get("eligible")) is not bool:
                raise ValueError("invalid dose eligibility")
            score, passed = dose.get("cue_min_margin"), dose.get("specificity_passed")
            if dose["eligible"]:
                if (type(score) not in (float, int) or not math.isfinite(score) or type(passed) is not bool
                        or passed != (score > 0.05)):
                    raise ValueError("invalid specificity outcome")
            elif score is not None or passed is not None:
                raise ValueError("ineligible dose cannot have a specificity outcome")
        if row["doses"]["high"]["eligible"] and not row["doses"]["low"]["eligible"]:
            raise ValueError("high dose cannot be eligible when low dose is not")
        keyed[key] = row
    if len({row["id"] for row in rows}) != 8:
        raise ValueError("duplicate cell ID")
    return keyed


def evaluate(rows):
    keyed = _validate_rows(rows)
    cells, paired = {}, {}
    for group in GROUPS:
        cells[group] = {}
        for label in DOSES:
            eligible = [keyed[t, group] for t in TRIPLES if keyed[t, group]["baseline_eligible"]
                        and keyed[t, group]["doses"][label]["eligible"]]
            passing = sum(row["doses"][label]["specificity_passed"] for row in eligible)
            status = "unevaluable" if len(eligible) < 2 else ("passed" if passing >= 2 else "failed")
            cells[group][label] = {"total": 3, "eligible": len(eligible), "passing": passing, "status": status}
        values = []
        counts = {"fail_to_pass": 0, "pass_to_fail": 0, "pass_to_pass": 0, "fail_to_fail": 0}
        for triple in TRIPLES:
            row = keyed[triple, group]
            low, high = row["doses"]["low"], row["doses"]["high"]
            if row["baseline_eligible"] and low["eligible"] and high["eligible"]:
                low_pass, high_pass = low["specificity_passed"], high["specificity_passed"]
                transition = ("pass" if low_pass else "fail") + "_to_" + ("pass" if high_pass else "fail")
                counts[transition] += 1
                values.append({"triple_id": triple, "low_margin": low["cue_min_margin"],
                               "high_margin": high["cue_min_margin"],
                               "difference": math.fsum((high["cue_min_margin"], -low["cue_min_margin"])),
                               "transition": transition})
        n = len(values)
        status = "unevaluable" if n < 2 else ("changed" if counts["fail_to_pass"] or counts["pass_to_fail"] else "unchanged")
        paired[group] = {"eligible": n, "status": status, "transitions": counts, "values": values,
                         "mean": math.fsum(item["difference"] for item in values) / n if n >= 2 else None}
    return {"cells": cells, "paired": paired}


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
    eligible = dose_eligibility(minimum)
    emit({"kind": "radius", "case_id": case_id, "minimum_distance": minimum, "distances": distances,
          "binding_edges": [item for item in distances if item["distance"] == minimum],
          "doses": {label: {"radius": radius, "eligible": eligible[label]} for label, radius in DOSES.items()}})
    for label, radius in DOSES.items():
        if not eligible[label]:
            emit({"kind": "dose_ineligible", "case_id": case_id, "dose": label,
                  "radius": radius, "minimum_distance": minimum})
    all_effects = {label: {} for label in DOSES if eligible[label]}
    if eligible["low"]:
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
            for label, radius in DOSES.items():
                if not eligible[label]:
                    continue
                site_effects = {}
                for donor, target in transfers:
                    submitted, planned, fraction = make_patch(captures[target][site], captures[donor][site], radius)
                    meta = {"case_id": case_id, "site": site, "dose": label, "donor": donor, "target": target}
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
                all_effects[label][site] = site_effects
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
    expected = 8 if not eligible["low"] else 24 + 48 * sum(eligible.values())
    if forwards != expected:
        raise RuntimeError("case schedule incomplete")
    dose_results = {}
    for label, radius in DOSES.items():
        if eligible[label]:
            effects = all_effects[label]
            specificity = {site: context_specificity(effects[site], cue_set) for site in SITES}
            cue_min = min(item["margin"] for item in specificity["cue2"]["targets"].values())
            limiting = [cue for cue in cues if specificity["cue2"]["targets"][cue]["margin"] == cue_min]
            passed = specificity["cue2"]["passed"]
        else:
            effects, specificity, cue_min, limiting, passed = {}, {}, None, [], None
        dose_results[label] = {"radius": radius, "eligible": eligible[label],
                               "effects": {site: [{"donor": donor, "target": target,
                                                   "delta": values[donor, target]}
                                                  for donor, target in transfers]
                                           for site, values in effects.items()},
                               "specificity": specificity, "cue_min_margin": cue_min,
                               "limiting_targets": limiting, "specificity_passed": passed}
    return {"id": case_id, "split": case["split"], "triple_id": case["triple_id"], "frame": case["frame"],
            "cue_set": cue_set, "forwards": forwards, "baseline_eligible": baseline["eligible"],
            "baseline_gaps": baseline["gaps"], "minimum_distance": minimum, "distances": distances,
            "clean_margins": margins, "baselines": baselines, "doses": dose_results}


def run_study(model, manifest, emit, *, expected_vocab=None, expected_width=None):
    cases = manifest["cases"]
    identities = [(row.get("triple_id"), row.get("cue_set"), row.get("split"), row.get("frame")) for row in cases]
    expected = [(triple, group, "calibration" if triple == CALIBRATION else "evaluation", "main")
                for triple in (CALIBRATION, *TRIPLES) for group in GROUPS]
    if identities != expected or len({row.get("id") for row in cases}) != 8:
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
