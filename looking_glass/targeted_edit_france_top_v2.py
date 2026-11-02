"""Saved-logit science for the proposed France full-top study.

This code does not load a model or establish execution, source, case custody,
model entry, state restoration, or the availability of an independent observer.
The caller must bind and exclusively own all supplied arrays and labels.
"""
import math
import struct

from . import targeted_edit_full_top as prior

LOCALITY_IDS = tuple(f"L{i:02d}" for i in range(1, 17))
EXPOSED = frozenset(("L01", "L02", "L05"))

def _probe():
    pos = struct.unpack("<d", struct.pack("<Q", 1))[0]
    neg = struct.unpack("<d", struct.pack("<Q", 0x8000000000000001))[0]
    got = (struct.unpack("<Q", struct.pack("<d", pos * 1.0))[0],
           struct.unpack("<Q", struct.pack("<d", neg * 1.0))[0])
    if got != (1, 0x8000000000000001):
        raise prior.StudyError("binary64 denormals are not preserved")
    return got

def _locality(entries, width):
    if type(entries) not in (list, tuple) or len(entries) != 16:
        raise prior.StudyError("sixteen ordered locality descriptions required")
    out = []
    for expected, entry in zip(LOCALITY_IDS, entries):
        if type(entry) is not dict or set(entry) != {"case_id", "accepted_ids"}:
            raise prior.StudyError("malformed locality description")
        ids = entry["accepted_ids"]
        if (entry["case_id"] != expected or type(ids) not in (list, tuple)
            or not ids or any(type(t) is not int or t < 0 or t >= width for t in ids)
            or len(set(ids)) != len(ids)):
            raise prior.StudyError("invalid locality identity or accepted IDs")
        out.append({"case_id": expected, "accepted_ids": tuple(ids),
                    "new": expected not in EXPOSED})
    return out

def screen(rows, true_id, alternative_id, locality):
    before = _probe()
    decoded = prior._rows32(rows, 27)
    prior._token_pair(true_id, alternative_id, decoded.shape[1])
    fixed = _locality(locality, decoded.shape[1])
    tops = [prior._top_details(row) for row in decoded]
    dev = [{"offset": i, **tops[i], "supported": tops[i]["unique_top_id"] == true_id}
           for i in range(3)]
    targets = [{"offset": i, **tops[i], "supported": tops[i]["unique_top_id"] == true_id}
               for i in range(3, 11)]
    local = [{**case, "offset": 11+i, **tops[11+i],
              "supported": tops[11+i]["unique_top_id"] in case["accepted_ids"]}
             for i, case in enumerate(fixed)]
    E = sum(x["supported"] for x in targets)
    H = sum(x["supported"] for x in local)
    H_new = sum(x["supported"] and x["new"] for x in local)
    reasons = []
    if not all(x["supported"] for x in dev):
        reasons.append("development baseline")
    if E < 4:
        reasons.append("target baseline count")
    if H < 8:
        reasons.append("locality baseline count")
    if H_new < 4:
        reasons.append("new locality baseline count")
    after = _probe()
    return {"status": "ineligible" if reasons else "eligible", "reasons": reasons,
            "E": E, "H": H, "H_new": H_new, "development": dev,
            "targets": targets, "locality": local,
            "binary64_probe_before": before, "binary64_probe_after": after}

def distribution_differences(base, edited, accepted, true_id, alternative_id):
    """Stable paired f32-row changes; absolute NLLs can round more coarsely."""
    import numpy as np
    before_probe = _probe()
    if (type(base) is not np.ndarray or type(edited) is not np.ndarray
        or base.dtype != np.float32 or edited.dtype != np.float32
        or base.ndim != 1 or edited.shape != base.shape or base.size < 2):
        raise prior.StudyError("expected two equal-length float32 rows")
    decoded = prior._rows32(np.stack((base, edited)), 2)
    prior._token_pair(true_id, alternative_id, decoded.shape[1])
    if (type(accepted) not in (list, tuple) or not accepted
        or any(type(x) is not int or not 0 <= x < decoded.shape[1] for x in accepted)
        or len(set(accepted)) != len(accepted)):
        raise prior.StudyError("invalid accepted IDs")
    accepted = tuple(sorted(accepted))

    def components(row):
        values = [float(x) for x in row]
        m = max(values)
        total = math.fsum(math.exp(v-m) for v in values)
        a = max(values[i] for i in accepted)
        accepted_total = math.fsum(math.exp(values[i]-a) for i in accepted)
        if not math.isfinite(total) or total <= 0 or not math.isfinite(accepted_total) or accepted_total <= 0:
            raise prior.StudyError("invalid log-normalizer")
        return values, m, total, math.log(total), a, math.log(accepted_total)

    z0, m0, total0, l0, a0, a_log0 = components(decoded[0])
    z1, m1, _total1, l1, a1, a_log1 = components(decoded[1])
    absolute0 = math.fsum((m0, -a0, l0, -a_log0))
    absolute1 = math.fsum((m1, -a1, l1, -a_log1))
    delta = math.fsum((m1, -m0, -a1, a0, l1, -l0, -a_log1, a_log0))
    gain = math.fsum((z1[alternative_id], -z0[alternative_id], m0, -m1, l0, -l1))
    margin = math.fsum((z1[alternative_id], -z1[true_id], -z0[alternative_id], z0[true_id]))
    kl = math.fsum((math.exp(x-m0)/total0)
                   * math.fsum((x, -y, m1, -m0, l1, -l0))
                   for x, y in zip(z0, z1))
    values = (absolute0, absolute1, delta, gain, margin, kl)
    if not all(map(math.isfinite, values)):
        raise prior.StudyError("nonfinite paired score")
    after_probe = _probe()
    return {"accepted_nll_original": absolute0, "accepted_nll_edited": absolute1,
            "accepted_nll_change": delta, "alternative_logp_gain": gain,
            "margin_gain": margin, "kl": kl,
            "binary64_probe_before": before_probe, "binary64_probe_after": after_probe}

def evaluate(rows, matrices, locality, true_id, alternative_id, *, development_top_achieved, match_available, execution_complete):
    """Compute provisional outcomes; the caller must independently bind execution."""
    import numpy as np
    if any(type(x) is not bool for x in (development_top_achieved, match_available, execution_complete)):
        raise prior.StudyError("execution and development flags must be booleans")
    if not execution_complete:
        return {"primary": {"status": "invalid/incomplete", "reason": "execution incomplete"},
                "conditions": None, "matched": None}
    initial = _probe()
    baseline = screen(rows, true_id, alternative_id, locality)
    if baseline["status"] != "eligible":
        return {"primary": {"status": "ineligible", "reasons": baseline["reasons"]},
                "baseline": baseline, "conditions": None, "matched": None}
    names = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")
    names += ("matched",) if match_available else ()
    if type(matrices) is not dict or set(matrices) != set(names):
        raise prior.StudyError("condition inventory differs")
    if (type(rows) is not np.ndarray or rows.dtype != np.float32 or rows.shape[0] != 27):
        raise prior.StudyError("screen rows are not float32")
    width = rows.shape[1]
    for name in names:
        value = matrices[name]
        if type(value) is not np.ndarray or value.dtype != np.float32 or value.shape != (24, width):
            raise prior.StudyError("condition row shape or type differs")
        prior._rows32(value, 24)  # reject nonfinite bits before scoring
    original_bits = np.ascontiguousarray(matrices["original"]).view(np.uint32)
    screened_bits = np.ascontiguousarray(rows[3:]).view(np.uint32)
    if not np.array_equal(original_bits, screened_bits):
        raise prior.StudyError("original evaluation rows differ from screened bytes")
    labels = _locality(locality, width)
    summaries = {}
    for name in names:
        targets, locals_ = [], []
        for i, after in enumerate(matrices[name]):
            before = matrices["original"][i]
            decoded = prior._rows32(np.stack((before, after)), 2)
            before_top = prior._top_details(decoded[0])
            after_top = prior._top_details(decoded[1])
            if i < 8:
                delta = distribution_differences(before, after, [true_id], true_id, alternative_id)
                targets.append({"offset": i, "baseline_supported": baseline["targets"][i]["supported"],
                                "baseline_top": before_top, "condition_top": after_top,
                                "alternative_unique_top": after_top["unique_top_id"] == alternative_id,
                                **delta})
            else:
                case = labels[i-8]
                delta = distribution_differences(before, after, case["accepted_ids"], true_id, alternative_id)
                locals_.append({"offset": i, "case_id": case["case_id"],
                                "accepted_ids": case["accepted_ids"], "new": case["new"],
                                "baseline_supported": baseline["locality"][i-8]["supported"],
                                "baseline_top": before_top, "condition_top": after_top,
                                "condition_accepted": after_top["unique_top_id"] in case["accepted_ids"],
                                **delta})
        accepted = [x for x in locals_ if x["baseline_supported"]]
        summaries[name] = {"targets": targets, "locality": locals_,
            "alternative_top_count_supported": sum(x["baseline_supported"] and x["alternative_unique_top"] for x in targets),
            "all_supported_localities_preserved": all(x["condition_accepted"] for x in accepted),
            "mean_locality_kl": math.fsum(x["kl"] for x in locals_) / 16.0,
            "max_locality_kl": max(x["kl"] for x in locals_),
            "max_accepted_nll_change": max(x["accepted_nll_change"] for x in locals_)}
    projected = summaries["projected"]
    K = projected["alternative_top_count_supported"]
    random_K = summaries["projected_random"]["alternative_top_count_supported"]
    opposite_K = summaries["opposite"]["alternative_top_count_supported"]
    threshold = (3 * baseline["E"] + 3) // 4
    criteria = {"development_top_achieved": development_top_achieved,
                "target_count": K >= threshold, "more_than_random": K > random_K,
                "more_than_opposite": K > opposite_K,
                "locality_preserved": projected["all_supported_localities_preserved"],
                "mean_kl": projected["mean_locality_kl"] <= 0.02,
                "max_kl": projected["max_locality_kl"] <= 0.10,
                "max_accepted_nll_change": projected["max_accepted_nll_change"] <= 0.20}
    primary = {"status": "positive" if all(criteria.values()) else "negative",
               "K": K, "K_random": random_K, "K_opposite": opposite_K,
               "threshold": threshold, "criteria": criteria}

    if not match_available:
        matched = {"status": "ineligible", "reasons": ["no development match"]}
    else:
        matched_rows, balanced = [], True
        for i, row in enumerate(projected["targets"]):
            if not row["baseline_supported"]:
                continue
            other = summaries["matched"]["targets"][i]
            checks = {}
            for metric in ("alternative_logp_gain", "margin_gain"):
                pg, ug = row[metric], other[metric]
                diff = abs(math.fsum((pg, -ug)))
                tolerance = 0.05 * max(abs(pg), abs(ug))
                checks[metric] = {"projected": pg, "matched": ug,
                                  "difference": diff, "tolerance": tolerance,
                                  "passes": pg > 0.05 and ug > 0.05 and diff <= tolerance}
            def category(details):
                top = details["unique_top_id"]
                return ("tied" if top is None else "factual" if top == true_id else
                        "counterfactual" if top == alternative_id else "other")
            cat_p, cat_u = category(row["condition_top"]), category(other["condition_top"])
            passes = cat_p == cat_u and all(item["passes"] for item in checks.values())
            balanced = balanced and passes
            matched_rows.append({"offset": i, "projected_top": cat_p, "matched_top": cat_u,
                                 "metrics": checks, "passes": passes})
        pkl = projected["mean_locality_kl"]
        ukl = summaries["matched"]["mean_locality_kl"]
        matched = {"status": "measured" if balanced else "ineligible",
                   "reasons": [] if balanced else ["evaluated strength imbalance"],
                   "balance": matched_rows,
                   "projected_locality_advantage": (pkl <= 0.9*ukl
                       and math.fsum((ukl, -pkl)) >= 1e-6) if balanced else None}
    final = _probe()
    return {"primary": primary, "baseline": baseline, "conditions": summaries,
            "matched": matched, "binary64_probe_before": initial, "binary64_probe_after": final,
            "scope": "provisional saved-row math; source, input identity, execution and cleanup need separate audit"}
