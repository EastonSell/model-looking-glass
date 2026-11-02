"""Matched same-cue and same-class cross-context CPU intervention study."""
import math

import torch

from .bert_cue_class import _emit_witness_failure, _observed
from .bert_cue_factorial import baseline_screen
from .bert_negation_cue import _bits, _margin, _same_bits, baseline_summary, distance, make_patch

CUES = ("never", "rarely", "always", "often")
SITES = ("cue2", "object2", "mask2", "cue5")
PAIR_IDS = ("discovery-01", "evaluation-01", "evaluation-02")


def donors(context, cue):
    other = 1 - context
    return tuple((other, item) for item in CUES) + tuple((context, item) for item in CUES if item != cue)


def same_class_other(cue):
    if cue not in CUES:
        raise ValueError("unknown cue")
    group = CUES[:2] if cue in CUES[:2] else CUES[2:]
    return next(item for item in group if item != cue)


def choose_q(values):
    if (len(values) != 224 or any(type(value) not in (int, float) or isinstance(value, bool)
                                  or not math.isfinite(value) or value < 0 for value in values)):
        raise ValueError("all 224 finite nonnegative distances are required")
    minimum = min(values)
    q = min(0.5, minimum)
    return q, minimum, q >= 0.05


def stable_probabilities(logits, excluded=()):
    values = logits.tolist() if isinstance(logits, torch.Tensor) else list(logits)
    if (len(values) < 3 or any(not math.isfinite(value) for value in values)
            or any(type(i) is not int or not 0 <= i < len(values) for i in excluded)):
        raise ValueError("invalid logits or excluded token")
    excluded = set(excluded)
    included = [i for i in range(len(values)) if i not in excluded]
    if not included:
        raise ValueError("empty vocabulary")
    peak = max(values[i] for i in included)
    weights = [math.exp(values[i] - peak) for i in included]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def total_variation(clean, patched, excluded=()):
    if len(clean) != len(patched):
        raise ValueError("vocabulary lengths differ")
    p = stable_probabilities(clean, excluded)
    q = stable_probabilities(patched, excluded)
    return math.fsum(abs(a - b) for a, b in zip(p, q)) / 2.0


def summarize_pair(effects, baseline_eligible, dose_eligible):
    if type(baseline_eligible) is not bool or type(dose_eligible) is not bool:
        raise ValueError("eligibility must be boolean")
    if not dose_eligible:
        if effects:
            raise ValueError("ineligible pair has effects")
        return {"status": "dose_ineligible", "informative": False, "recipients": [],
                "same_cue_mean": None, "same_class_mean": None, "half_difference": None}
    recipients = []
    for context in range(2):
        for cue in CUES:
            same = effects[("cue2", context, cue, 1 - context, cue)]["delta"]
            other = effects[("cue2", context, cue, 1 - context, same_class_other(cue))]["delta"]
            if not all(math.isfinite(v) for v in (same, other)):
                raise ValueError("nonfinite selected effects")
            recipients.append({"target_context": context, "target_cue": cue,
                               "same_cue_delta": same, "same_class_delta": other})
    a = math.fsum(abs(r["same_cue_delta"]) for r in recipients) / 8
    b = math.fsum(abs(r["same_class_delta"]) for r in recipients) / 8
    h = math.fsum((a, -b / 2))
    informative = baseline_eligible and b >= 0.01
    if not baseline_eligible:
        status = "baseline_ineligible"
    elif b < 0.01:
        status = "uninformative"
    else:
        status = "at_least_half" if h >= 0 else "below_half"
    return {"status": status, "informative": informative, "recipients": recipients,
            "same_cue_mean": a, "same_class_mean": b, "half_difference": h}


def evaluate(pairs):
    if ([row["id"] for row in pairs] != list(PAIR_IDS) or
            [row["split"] for row in pairs] != ["discovery", "evaluation", "evaluation"]):
        raise ValueError("fixed pair order differs")
    evaluation = pairs[1:]
    eligible = [row for row in evaluation if row["comparison"]["informative"]]
    if len(eligible) != 2:
        return {"status": "unevaluable", "informative_pairs": len(eligible), "recipients": 8 * len(eligible),
                "same_cue_mean": None, "same_class_mean": None, "half_difference": None}
    recipients = [recipient for pair in evaluation for recipient in pair["comparison"]["recipients"]]
    if len(recipients) != 16:
        raise ValueError("all 16 evaluation recipients are required")
    a = math.fsum(abs(r["same_cue_delta"]) for r in recipients) / 16
    b = math.fsum(abs(r["same_class_delta"]) for r in recipients) / 16
    h = math.fsum((a, -b / 2))
    return {"status": "at_least_half" if h >= 0 else "below_half", "informative_pairs": 2,
            "recipients": 16, "same_cue_mean": a, "same_class_mean": b, "half_difference": h}


def _baseline(logits, any_id, some_id):
    summary = baseline_summary(logits, any_id=any_id, some_id=some_id)
    values = logits.tolist()
    peak = max(values)
    log_shifted_total = math.log(math.fsum(math.exp(value - peak) for value in values))
    summary.update({"any_logit": values[any_id], "some_logit": values[some_id],
                    "any_log_probability": (values[any_id] - peak) - log_shifted_total,
                    "some_log_probability": (values[some_id] - peak) - log_shifted_total,
                    "target_pair_probability": math.fsum((summary["any_probability"],
                                                          summary["some_probability"]))})
    return summary


def run_pair(model, pair, emit, *, any_id, some_id, expected_vocab=None, expected_width=None):
    pair_id, contexts = pair["id"], pair["contexts"]
    if len(contexts) != 2 or len({context["id"] for context in contexts}) != 2:
        raise ValueError("two distinct contexts required")
    rows = {(ci, cue): torch.tensor([context["rows"][cue]["input_ids"]], dtype=torch.int64)
            for ci, context in enumerate(contexts) for cue in CUES}
    clean, captures, margins, baselines = {}, {}, {}, {}
    forwards = 0
    for ci, context in enumerate(contexts):
        for cue in CUES:
            meta = {"pair_id": pair_id, "context": ci, "cue": cue}
            observed = _observed(model, rows[ci, cue], context["indices"], emit, meta,
                                 expected_vocab=expected_vocab, expected_width=expected_width)
            forwards += 1
            clean[ci, cue], captures[ci, cue] = observed["mask_logits"], observed["captures"]
            emit({"kind": "clean", **meta, "input_ids": rows[ci, cue][0].tolist(),
                  "mask_logits_f32le": _bits(clean[ci, cue]),
                  "captures_f32le": {site: _bits(captures[ci, cue][site]) for site in SITES}})
            baselines[ci, cue] = _baseline(clean[ci, cue], any_id, some_id)
            margins[ci, cue] = baselines[ci, cue]["any_minus_some"]
    screens = [baseline_screen({cue: margins[ci, cue] for cue in CUES}, "prior") for ci in range(2)]
    baseline_eligible = all(screen["eligible"] for screen in screens)
    emit({"kind": "baseline", "pair_id": pair_id, "contexts": screens,
          "eligible": baseline_eligible})
    distances = []
    for site in SITES:
        for ci in range(2):
            for cue in CUES:
                for di, donor in donors(ci, cue):
                    value = distance(captures[ci, cue][site], captures[di, donor][site])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("nonfinite donor distance")
                    distances.append({"site": site, "target_context": ci, "target_cue": cue,
                                      "donor_context": di, "donor_cue": donor, "distance": value})
    q, minimum, dose_eligible = choose_q([row["distance"] for row in distances])
    emit({"kind": "radius", "pair_id": pair_id, "q": q, "minimum_distance": minimum,
          "binding_edges": [row for row in distances if row["distance"] == minimum],
          "distances": distances, "dose_eligible": dose_eligible})
    if not dose_eligible:
        emit({"kind": "dose_ineligible", "pair_id": pair_id, "q": q})
    effects = {}
    if dose_eligible:
        for site in SITES:
            for ci, context in enumerate(contexts):
                for cue in CUES:
                    meta = {"pair_id": pair_id, "site": site, "target_context": ci, "target_cue": cue}
                    observed = _observed(model, rows[ci, cue], context["indices"], emit, meta, site=site,
                                         self_patch=True, expected_vocab=expected_vocab, expected_width=expected_width)
                    forwards += 1
                    emit({"kind": "self", **meta, "input_ids": rows[ci, cue][0].tolist(),
                          **{name + "_f32le": _bits(value) for name, value in observed.items()}})
                    if (not _same_bits(observed["before"], captures[ci, cue][site]) or
                            not _same_bits(observed["mask_logits"], clean[ci, cue])):
                        raise RuntimeError("self or clean-state control failed")
            for ci, context in enumerate(contexts):
                for cue in CUES:
                    for di, donor in donors(ci, cue):
                        submitted, planned, fraction = make_patch(captures[ci, cue][site], captures[di, donor][site], q)
                        meta = {"pair_id": pair_id, "site": site, "target_context": ci, "target_cue": cue,
                                "donor_context": di, "donor_cue": donor}
                        observed = _observed(model, rows[ci, cue], context["indices"], emit, meta, site=site,
                                             source=submitted, expected_vocab=expected_vocab, expected_width=expected_width)
                        forwards += 1
                        try:
                            actual = distance(observed["before"], observed["after"])
                        except BaseException:
                            _emit_witness_failure(emit, meta, rows[ci, cue], "dose_measurement", observed,
                                                  submitted_f32le=_bits(submitted), requested_radius=q,
                                                  requested_fraction=fraction, planned_norm=planned)
                            raise
                        emit({"kind": "transfer", **meta, "input_ids": rows[ci, cue][0].tolist(),
                              "requested_radius": q, "requested_fraction": fraction, "planned_norm": planned,
                              "actual_norm": actual, "relative_error": abs(actual - q) / q,
                              "submitted_f32le": _bits(submitted),
                              **{name + "_f32le": _bits(value) for name, value in observed.items()}})
                        if (not _same_bits(observed["before"], captures[ci, cue][site]) or actual <= 0 or
                                abs(actual - q) / q > 0.001 or
                                (site == "cue5" and not _same_bits(observed["mask_logits"], clean[ci, cue]))):
                            raise RuntimeError("transfer witness, dose or final-cue control failed")
                        patched_probs = stable_probabilities(observed["mask_logits"])
                        values = {"delta": math.fsum((_margin(observed["mask_logits"], any_id, some_id),
                                                      -margins[ci, cue])),
                                  "full_tv": total_variation(clean[ci, cue], observed["mask_logits"]),
                                  "excluded_tv": total_variation(clean[ci, cue], observed["mask_logits"],
                                                                 (any_id, some_id)),
                                  "clean_pair_probability": baselines[ci, cue]["target_pair_probability"],
                                  "patched_pair_probability": math.fsum((patched_probs[any_id],
                                                                        patched_probs[some_id]))}
                        effects[site, ci, cue, di, donor] = values
    for ci, context in enumerate(contexts):
        for cue in CUES:
            meta = {"pair_id": pair_id, "context": ci, "cue": cue}
            observed = _observed(model, rows[ci, cue], context["indices"], emit, meta,
                                 expected_vocab=expected_vocab, expected_width=expected_width)
            forwards += 1
            emit({"kind": "repeat", **meta, "input_ids": rows[ci, cue][0].tolist(),
                  "mask_logits_f32le": _bits(observed["mask_logits"]),
                  "captures_f32le": {site: _bits(observed["captures"][site]) for site in SITES}})
            if (not _same_bits(observed["mask_logits"], clean[ci, cue]) or
                    any(not _same_bits(observed["captures"][site], captures[ci, cue][site]) for site in SITES)):
                raise RuntimeError("clean repeat control failed")
    if forwards != (272 if dose_eligible else 16):
        raise RuntimeError("pair forward schedule incomplete")
    tv_summary = {}
    if dose_eligible:
        for site in SITES:
            for metric in ("full_tv", "excluded_tv"):
                tv_summary[site, metric] = {
                    "same_cue_mean": math.fsum(effects[site, ci, cue, 1-ci, cue][metric]
                                               for ci in range(2) for cue in CUES) / 8,
                    "same_class_mean": math.fsum(effects[site, ci, cue, 1-ci, same_class_other(cue)][metric]
                                                 for ci in range(2) for cue in CUES) / 8}
    return {"id": pair_id, "split": pair["split"], "forwards": forwards,
            "baseline_eligible": baseline_eligible, "baseline_screens": screens,
            "baselines": [{"context": ci, "cue": cue, **baselines[ci, cue]}
                          for ci in range(2) for cue in CUES],
            "dose_eligible": dose_eligible, "q": q, "minimum_distance": minimum, "distances": distances,
            "effects": [{"site": site, "target_context": ci, "target_cue": cue,
                         "donor_context": di, "donor_cue": donor, **values}
                        for (site, ci, cue, di, donor), values in effects.items()],
            "tv_summary": [{"site": site, "metric": metric, **values}
                           for (site, metric), values in tv_summary.items()],
            "comparison": summarize_pair(effects, baseline_eligible, dose_eligible)}


def run_study(model, manifest, emit, *, expected_vocab=None, expected_width=None):
    if [row["id"] for row in manifest["pairs"]] != list(PAIR_IDS):
        raise ValueError("fixed pair list differs")
    results = []
    for pair in manifest["pairs"]:
        try:
            results.append(run_pair(model, pair, emit, any_id=manifest["scored_tokens"]["any"],
                                    some_id=manifest["scored_tokens"]["some"],
                                    expected_vocab=expected_vocab, expected_width=expected_width))
        except BaseException as exc:
            try:
                emit({"kind": "pair_failed", "pair_id": pair["id"], "error_type": type(exc).__name__})
            except Exception:
                pass
            raise
    return {"status": "complete", "forwards": sum(row["forwards"] for row in results),
            "pairs": results, "evaluation": evaluate(results)}
