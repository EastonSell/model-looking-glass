"""Model-free replay of a closed DistilBERT agreement study journal.

Usage: python bert_eval_independent_science.py MANIFEST EVENTS RESULT

Only saved observations are read; this never imports or invokes a model.
Hashes and control flags establish internal consistency, not execution
authenticity, prior blindness, or the contents of unsaved full-mask logits.
"""

import base64
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

SITES = (("candidate", 2, 2), ("block2_distractor", 2, 5),
         ("random_block3_distractor", 3, 5), ("random_block4_distractor", 4, 5),
         ("random_block0_subject", 0, 2))
CELLS = ("A_same", "A_opposite", "B_same", "B_opposite")
FACTORS = (("singular", "singular"), ("singular", "plural"),
           ("plural", "singular"), ("plural", "plural"))
MANIFEST_HASH = "20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def parse(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    def bad(_):
        raise ValueError("nonfinite JSON")

    value = json.loads(data, object_pairs_hook=unique, parse_constant=bad)
    return value


def equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def number(bits):
    need(type(bits) is int and 0 <= bits < 2**32, "invalid float32 bits")
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    need(exponent != 255, "nonfinite selected logit")
    out = math.ldexp(float(fraction), -149) if exponent == 0 else math.ldexp(float((1 << 23) | fraction), exponent - 150)
    return -out if bits >> 31 else out


def vector(encoded):
    need(type(encoded) is str, "missing action vector")
    raw = base64.b64decode(encoded, validate=True)
    need(len(raw) == 3072 and base64.b64encode(raw).decode("ascii") == encoded, "invalid vector")
    need(all((bits & 0x7f800000) != 0x7f800000 for (bits,) in struct.iter_unpack("<I", raw)), "nonfinite vector")
    return raw


def check_fields(value, expected, message):
    need(type(value) is dict and all(key in value and equal(value[key], wanted) for key, wanted in expected.items()), message)


def read_journal(path, count):
    raw = path.read_bytes()
    need(0 < len(raw) <= 16 * 1024 * 1024, "journal size")
    need(raw.endswith(b"\n") and b"\r" not in raw and raw.count(b"\n") == count, "journal framing or count")
    lines = raw[:-1].split(b"\n")
    need(all(0 < len(line) <= 65536 for line in lines), "journal line size")
    return [parse(line) for line in lines], hashlib.sha256(raw).hexdigest()


def audit(manifest_path, journal_path, result_path):
    manifest_raw = manifest_path.read_bytes()
    need(hashlib.sha256(manifest_raw).hexdigest() == MANIFEST_HASH, "unexpected manifest bytes")
    manifest = parse(manifest_raw)
    result = parse(result_path.read_bytes())
    need(type(result) is dict and result.get("schema") == "bert-agreement-controller-result-v1"
         and result.get("status") == "complete" and result.get("owner_state") == "reaped"
         and type(result.get("owner_exit_code")) is int and result["owner_exit_code"] == 0
         and result.get("manifest_sha256") == MANIFEST_HASH, "result is not a complete reaped attempt")
    split, one_family = result.get("split"), result.get("family")
    need(split in ("development", "evaluation"), "unexpected split")
    numbers = [one_family] if split == "development" else list(range(1, 9))
    need((split == "development" and type(one_family) is int and 1 <= one_family <= 4)
         or (split == "evaluation" and one_family is None), "invalid family selection")
    records, sha = read_journal(journal_path, 328 * len(numbers))
    need(sha == result.get("journal_sha256"), "journal digest mismatch")
    need(type(result.get("families")) is list and len(result["families"]) == len(numbers), "result family count")
    cursor = 0
    all_cases, family_means = [], {site: [] for site, _, _ in SITES}
    baseline_cells = {"_".join(factors): [] for factors in FACTORS}
    totals = {"forwards": 0, "controls": 0, "patches": 0}
    row_by_id = {row["row_id"]: row for row in manifest["rows"]}
    need(len(row_by_id) == len(manifest["rows"]), "duplicate manifest row")

    for family_index, family in enumerate(numbers):
        rows = [row for row in manifest["rows"] if row["split"] == split and row["family"] == family]
        targets = [row for row in manifest["targets"] if row["split"] == split and row["family"] == family]
        need(len(rows) == 8 and len(targets) == 4, "family manifest inventory")
        need([(row_by_id[t["target_row_id"]]["subject_number"], row_by_id[t["target_row_id"]]["distractor_number"])
              for t in targets] == list(FACTORS), "target factorial order")
        by_id = {row["row_id"]: row for row in rows}
        need(len(by_id) == 8, "duplicate family rows")
        family_forwards = 0
        controls = []

        def take():
            nonlocal cursor
            need(cursor < len(records), "unexpected end of journal")
            record = records[cursor]
            cursor += 1
            return record

        def forward(kind, row_id, site, layer, token, donor=None, patch=False, self_patch=False):
            nonlocal family_forwards
            family_forwards += 1
            totals["forwards"] += 1
            totals["patches"] += int(patch)
            identity = {"split": split, "family": family, "forward_index": family_forwards, "kind": kind,
                        "row_id": row_id, "donor_row_id": donor, "site": site, "layer": layer, "token": token}
            started, completed = take(), take()
            check_fields(started, {"status": "started", **identity}, "forward start mismatch")
            check_fields(completed, {"status": "completed", **identity}, "forward completion mismatch")
            check_fields(completed, {"model_input_ids": by_id[row_id]["input_ids"],
                                    "model_attention_mask": by_id[row_id]["attention_mask"],
                                    "capture_bytes": 6144 if patch else 3072,
                                    "source_bytes": 3072 if patch and not self_patch else 0}, "inputs or bytes mismatch")
            before = vector(completed.get("before_float32_le_base64"))
            after = vector(completed.get("after_float32_le_base64")) if patch else None
            if not patch:
                need(completed.get("after_float32_le_base64") is None, "capture has unexpected action vector")
            bits = (completed.get("is_bits"), completed.get("are_bits"))
            values = tuple(number(x) for x in bits)
            for label, value in zip(("is_logit", "are_logit"), values):
                recorded = completed.get(label)
                need(type(recorded) in (int, float) and math.isfinite(recorded) and recorded == value
                     and (recorded != 0 or math.copysign(1, recorded) == math.copysign(1, value)), "selected logit mismatch")
            mask = completed.get("mask_logits_sha256")
            need(type(mask) is str and re.fullmatch("[0-9a-f]{64}", mask) is not None, "invalid mask digest")
            names = ["before", "patch", "after"] if patch else ["before"]
            actions = completed.get("events")
            need(type(actions) is list and len(actions) == len(names), "action count")
            path = f"distilbert.transformer.layer.{layer}"
            for i, action in enumerate(actions):
                check_fields(action, {"name": names[i], "operation": "patch" if patch and i == 1 else "capture",
                    "order": i, "module_path": path, "module_aliases": [path], "call_site_known": False,
                    "forward_index": 0, "invocation": 0, "batch_index": 0, "token_index": token,
                    "output_selector": [0], "output_shape": [1, 10, 768], "selected_shape": [768],
                    "dtype": "torch.float32", "device": "cpu", "source_capture": "before" if self_patch and i == 1 else None,
                    "preceding_same_site_actions": names[:i]}, "action metadata mismatch")
            return {"before": before, "after": after, "bits": bits, "values": values, "mask": mask}

        def control(name, case_id, site, value, reference):
            record = take()
            check_fields(record, {"status": "control", "split": split, "family": family, "name": name,
                                  "case_id": case_id, "site": site, "passed": True}, "control record mismatch")
            need(value["mask"] == reference["mask"] and value["bits"] == reference["bits"], "control digest or selected bits mismatch")
            controls.append(record)
            totals["controls"] += 1

        references, baseline, effects, endpoints = {}, {}, {}, {}
        for site, layer, token in SITES:
            donors = {}
            for row in rows:
                row_id = row["row_id"]
                observation = forward("donor_capture", row_id, site, layer, token)
                donors[row_id] = observation["before"]
                if row["subject_source"] == "A":
                    if site == "candidate":
                        references[row_id], baseline[row_id] = observation, observation["bits"]
                    else:
                        control("clean_matches_candidate", row_id, site, observation, references[row_id])
            for target in targets:
                row_id, case = target["target_row_id"], target["case_id"]
                margins, saved = {}, {}
                for cell in CELLS:
                    donor_id = target["donor_row_ids"][cell]
                    need(donor_id in donors and (cell != "A_same" or donor_id == row_id), "donor mapping mismatch")
                    observation = forward("patch_" + cell, row_id, site, layer, token, donor_id, True, cell == "A_same")
                    need(observation["before"] == donors[row_id] and observation["after"] == donors[donor_id], "patch did not land")
                    if cell == "A_same":
                        control("self_patch", case, site, observation, references[row_id])
                    margins[cell] = math.fsum((observation["values"][0], -observation["values"][1]))
                    saved[cell] = {"donor_row_id": donor_id, "is_bits": observation["bits"][0],
                                   "are_bits": observation["bits"][1], "is_logit": observation["values"][0],
                                   "are_logit": observation["values"][1], "margin": margins[cell]}
                d_a = math.fsum((margins["A_opposite"], -margins["A_same"]))
                d_b = math.fsum((margins["B_opposite"], -margins["B_same"]))
                sign = 1 if by_id[row_id]["subject_number"] == "singular" else -1
                effects.setdefault(case, {})[site] = -sign * (math.fsum((d_a, d_b)) / 2.0)
                endpoints.setdefault(case, {})[site] = saved

        final_sources, final_before = {"final_subject": {}, "final_mask": {}}, {"final_subject": {}, "final_mask": {}}
        for target in targets:
            row_id, case, donor_id = target["target_row_id"], target["case_id"], target["donor_row_ids"]["A_opposite"]
            for name, token in (("final_subject", 2), ("final_mask", 6)):
                donor = forward("delivery_donor", donor_id, name, 5, token)
                patched = forward("delivery_patch", row_id, name, 5, token, donor_id, True)
                need(patched["after"] == donor["before"], "final patch did not land")
                final_sources[name][donor_id], final_before[name][row_id] = donor["before"], patched["before"]
                control(name, case, name, patched, references[row_id] if name == "final_subject" else donor)
            fresh = forward("final_clean", row_id, "candidate", 2, 2)
            control("final_clean", case, "candidate", fresh, references[row_id])
        need(final_sources == final_before and family_forwards == 140 and len(controls) == 48, "family inventory mismatch")

        cases = []
        for target, factors in zip(targets, FACTORS):
            case, row_id = target["case_id"], target["target_row_id"]
            case_result = {"case_id": case, "target_row_id": row_id, "baseline_is_bits": baseline[row_id][0],
                           "baseline_are_bits": baseline[row_id][1], "effects": effects[case]}
            cases.append(case_result)
            margin = math.fsum((number(baseline[row_id][0]), -number(baseline[row_id][1])))
            correct = margin > 0 if factors[0] == "singular" else margin < 0
            baseline_cells["_".join(factors)].append((margin, correct))
            all_cases.append({**case_result, "baseline_margin": margin, "baseline_correct": correct,
                              "endpoints": endpoints[case]})
        means = {site: math.fsum(case["effects"][site] for case in cases) / 4.0 for site, _, _ in SITES}
        for site in means:
            family_means[site].append(means[site])
        expected = {"split": split, "family": family, "forward_count": 140,
                    "cases": cases, "site_means": means, "controls": controls}
        need(equal(result["families"][family_index], expected), "family summary disagrees with saved endpoints")

    need(cursor == len(records), "extra journal records")
    summary = None
    if split == "evaluation":
        means = {site: math.fsum(values) / 8.0 for site, values in family_means.items()}
        comparisons = [site for site, _, _ in SITES[1:]]
        differences = {site: math.fsum((means["candidate"], -means[site])) for site in comparisons}
        family_diffs = {site: [math.fsum((a, -b)) for a, b in zip(family_means["candidate"], family_means[site])]
                        for site in comparisons}
        mean_diffs = {site: math.fsum(values) / 8.0 for site, values in family_diffs.items()}
        cells = {name: {"n": len(values), "correct": sum(correct for _, correct in values),
                        "mean_margin": math.fsum(margin for margin, _ in values) / len(values)}
                 for name, values in baseline_cells.items()}
        positives = sum(value > 0 for value in family_means["candidate"])
        summary = {"site_means": means, "family_means": family_means, "candidate_minus_site": differences,
                   "family_candidate_minus_site": family_diffs, "mean_family_candidate_minus_site": mean_diffs,
                   "positive_candidate_families": positives,
                   "directional_hypothesis_passed": means["candidate"] > 0.1 and positives >= 6,
                   "site_specificity_passed": differences["block2_distractor"] > 0.1,
                   "baseline": {"n": 32, "correct": sum(cell["correct"] for cell in cells.values()), "cells": cells}}
    need(equal(result.get("summary"), summary), "aggregate summary disagrees with saved endpoints")
    return {"schema": "bert-agreement-independent-science-v1", "status": "saved_records_consistent",
            "source_commit": result["source_commit"], "split": split, "journal_sha256": sha,
            "family_count": len(numbers), "counts": totals, "cases": all_cases, "summary": summary,
            "limitations": ["record consistency does not authenticate execution or prove historical blindness",
                "unsaved full-mask logits cannot be reconstructed from digests or selected bits",
                "families share cyclic donor subjects and are not independent population samples",
                "donor-number effects may reflect lexical changes or erasure rather than a unique number representation"]}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: audit.py MANIFEST EVENTS RESULT")
    print(json.dumps(audit(*(Path(arg) for arg in sys.argv[1:])), sort_keys=True, indent=2, allow_nan=False))
