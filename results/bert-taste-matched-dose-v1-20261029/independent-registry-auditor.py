"""Independently check saved matched-dose taste records without loading a model."""

import base64
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(typed_equal(a[key], b[key]) for key in a)
    if type(a) is list:
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def parse_json(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        need(math.isfinite(result), "nonfinite JSON number")
        return result

    def invalid_constant(_value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_float=finite_float, parse_constant=invalid_constant)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode32(bits):
    need(type(bits) is int and 0 <= bits < 2**32, "invalid float32 bits")
    exp, frac = (bits >> 23) & 255, bits & 0x7fffff
    need(exp != 255, "nonfinite float32")
    value = math.ldexp(float(frac), -149) if exp == 0 else math.ldexp(float((1 << 23) | frac), exp - 150)
    return -value if bits >> 31 else value


def values(raw):
    return [decode32(bits) for (bits,) in struct.iter_unpack("<I", raw)]


def vec(record, field, length=768):
    encoded = record[field + "_float32_le_base64"]
    if encoded is None:
        return None
    raw = base64.b64decode(encoded, validate=True)
    need(len(raw) == length * 4, field + " length")
    values(raw)
    return raw


def close(x, y):
    return type(x) in (int, float) and type(y) in (int, float) and math.isclose(x, y, rel_tol=2e-14, abs_tol=1e-12)


def exact_float(a, b):
    return type(a) is float and type(b) is float and struct.pack("<d", a) == struct.pack("<d", b)


def dist(a, b):
    left, right = values(a), values(b)
    need(len(left) == len(right), "distance length")
    result = math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(left, right)))
    need(math.isclose(result, math.dist(left, right), rel_tol=2e-14, abs_tol=1e-12), "independent distance")
    return result


def main(out_arg, receipt_arg, cases_arg):
    out, receipt_path, cases_path = map(Path, (out_arg, receipt_arg, cases_arg))
    need(not (out / "incomplete.json").exists(), "incomplete marker")
    result_raw = (out / "result.json").read_bytes()
    result, receipt, cases = parse_json(result_raw), parse_json(receipt_path.read_bytes()), parse_json(cases_path.read_bytes())
    need(receipt["status"] == "complete" and receipt["child_reaped"] is True
         and type(receipt["child_exit_code"]) is int and receipt["child_exit_code"] == 0
         and receipt["source_unchanged"] is True and receipt["bytecode_prefix_empty"] is True
         and receipt["result_sha256"] == sha(result_raw), "observer receipt")
    need(result["status"] == "complete" and type(result["completed_forwards"]) is int
         and result["completed_forwards"] == 516 and typed_equal(receipt["journals"], result["journals"])
         and receipt["source_commit"] == result["source"]["commit"], "result identity")
    need(result["model"]["model"] == cases["model"] and result["model"]["revision"] == cases["revision"]
         and result["model"]["device"] == "cpu" and result["model"]["dtype"] == "float32", "model identity")
    rows = cases["rows"]
    need(len(rows) == 12 and len(result["baselines"]) == 12, "case inventory")
    effects, doses, baselines = {}, [], {}
    complete = 0
    for offset, family in enumerate(("coffee", "milk", "soup", "cake", "pudding", "cocoa")):
        pair = rows[2 * offset:2 * offset + 2]
        names = [family + "-sugar", family + "-lemon"]
        need([r["id"] for r in pair] == names, "case order")
        idmap = {r["id"]: r for r in pair}
        path = out / (family + ".jsonl")
        data = path.read_bytes()
        need(data.endswith(b"\n") and typed_equal(result["journals"][path.name], {"size": len(data), "sha256": sha(data)}), "journal digest")
        lines = data.splitlines()
        need(len(lines) == 178, "journal event count")
        events = [parse_json(line) for line in lines]
        cursor, captures, whole_before = 0, {}, {}

        def forward(row, layer, position, mode, donor, radius=None, fraction=None):
            nonlocal cursor, complete
            start, done = events[cursor:cursor + 2]
            cursor += 2
            keys = ("row_id", "layer", "position", "mode", "donor")
            identity = (row, layer, position, mode, donor)
            need(start["event"] == "forward_start" and done["event"] == "forward_complete"
                 and all(typed_equal(start[k], value) and typed_equal(done[k], value)
                         for k, value in zip(keys, identity)), "forward order")
            entry = done["model_entry"]
            need(typed_equal(entry, {"input_ids": idmap[row]["input_ids"], "attention_mask": idmap[row]["attention_mask"]}), "model entry")
            denormal = {"smallest_times_one_bits": 1, "normal_half_bits": 0x00400000}
            need(typed_equal(done["denormal_before"], denormal) and typed_equal(done["denormal_after"], denormal)
                 and typed_equal(done["denormal_after_scoring"], denormal), "denormal")
            before, after, source, donor_clean = (vec(done, field) for field in ("before", "after", "source", "donor_clean"))
            need(before is not None, "selected capture")
            whole_key = (row, layer)
            if whole_key in whole_before:
                need(done["full_before_sha256"] == whole_before[whole_key], "whole pre-output changed")
            else:
                whole_before[whole_key] = done["full_before_sha256"]
            patched = mode in ("ingredient_donor", "ingredient_self", "control_self", "ingredient_matched", "control_matched")
            expected_steps = ["full_before", "before", "patch", "after", "full_after"] if patched else ["full_before", "before"]
            need([step["name"] for step in done["events"]] == expected_steps, "plan steps")
            for number, step in enumerate(done["events"]):
                selected = step["name"] in ("before", "patch", "after")
                module = "distilbert.transformer.layer." + str(layer)
                expected = {"order": number, "forward_index": 0, "invocation": 0,
                            "module_path": module, "module_aliases": [module], "output_selector": [0],
                            "output_shape": [1, 10, 768], "token_index": position if selected else None,
                            "batch_index": 0 if selected else None,
                            "call_site_known": False,
                            "operation": "patch" if step["name"] == "patch" else "capture",
                            "dtype": "torch.float32", "device": "cpu",
                            "selected_shape": [768] if selected else [1, 10, 768],
                            "source_capture": "before" if step["name"] == "patch" and mode.endswith("_self") else None,
                            "preceding_same_site_actions": expected_steps[:number]}
                need(all(key in step and typed_equal(step[key], value) for key, value in expected.items()), "plan coordinates")
            s = done["summary"]
            sweet, sour = decode32(s["sweet_bits"]), decode32(s["sour_bits"])
            margin = math.fsum((sweet, -sour))
            need(exact_float(s["sweet_logit"], sweet) and exact_float(s["sour_logit"], sour)
                 and exact_float(s["sweet_minus_sour"], margin), "target logits")
            need(all(type(s[key]) is float and math.isfinite(s[key]) and 0 <= s[key] <= 1
                     for key in ("sweet_probability", "sour_probability"))
                 and type(s["top_five_ids"]) is list and len(s["top_five_ids"]) == 5
                 and all(type(value) is int and 0 <= value < 30522 for value in s["top_five_ids"])
                 and len(set(s["top_five_ids"])) == 5
                 and type(s["top_five_tokens"]) is list and len(s["top_five_tokens"]) == 5
                 and all(type(value) is str for value in s["top_five_tokens"]), "summary types")
            if mode.startswith("capture_"):
                need(after is None and source is None and donor_clean is None, "capture-only action")
                captures[(row, layer, position)] = before
            if mode == "capture_ingredient" and layer == 0:
                logits_raw = base64.b64decode(done["baseline_mask_logits_float32_le_base64"], validate=True)
                need(len(logits_raw) == 30522 * 4 and sha(logits_raw) == s["mask_logits_sha256"], "baseline bits")
                logits = values(logits_raw)
                need(logits[4086] == sweet and logits[14768] == sour, "baseline target bits")
                ranking = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:5]
                need(typed_equal(ranking, s["top_five_ids"]), "baseline top five")
                high = max(logits)
                denominator = math.fsum(math.exp(x - high) for x in logits)
                for word, token in (("sweet", 4086), ("sour", 14768)):
                    need(math.isclose(s[word + "_probability"], math.exp(logits[token] - high) / denominator,
                                      rel_tol=1e-12, abs_tol=1e-15), "baseline probability")
                baselines[row] = (margin, s["mask_logits_sha256"], s["top_five_ids"])
                need(typed_equal(result["baselines"][row], s), "baseline result mismatch")
            else:
                need(done["baseline_mask_logits_float32_le_base64"] is None, "unexpected full logits")
            if mode.startswith("capture_") or mode == "final_clean":
                need(s["mask_logits_sha256"] == baselines[row][1], "clean logits replay")
            if patched:
                need(before == captures[(row, layer, position)] and done["untouched_positions_equal"] is True, "recipient capture")
                if mode.endswith("_self"):
                    need(source is None and donor_clean is None and after == before and done["dose"] is None
                         and done["full_before_sha256"] == done["full_after_sha256"], "self patch")
                else:
                    want_donor = captures[(donor, layer, position)]
                    need(donor_clean == want_donor and after == source, "donor or landed source")
                    if mode.endswith("_matched"):
                        x, y = values(before), values(donor_clean)
                        expected = b"".join(struct.pack("<f", math.fsum((a, fraction * (b - a)))) for a, b in zip(x, y))
                        need(source == expected, "interpolation bits")
                        delivered = dist(before, after)
                        d = done["dose"]
                        recipient_norm = math.sqrt(math.fsum(x * x for x in values(before)))
                        need(close(d["fraction"], fraction) and close(d["requested_norm"], radius)
                             and close(d["delivered_norm"], delivered) and close(done["actual_pre_post_norm"], delivered)
                             and close(d["relative_error"], abs(delivered - radius) / radius)
                             and close(d["recipient_norm"], recipient_norm)
                             and d["recipient_norm_status"] == ("zero" if recipient_norm == 0 else "positive")
                             and (d["dose_to_recipient_norm"] is None if recipient_norm == 0 else
                                  close(d["dose_to_recipient_norm"], delivered / recipient_norm))
                             and delivered != 0 and abs(delivered - radius) <= 1e-6 + 1e-4 * radius, "delivered dose")
                    else:
                        need(source == want_donor and done["dose"] is None, "full donor")
                if mode.endswith("_self") or layer == 5:
                    need(s["mask_logits_sha256"] == baselines[row][1], "self or final block logits")
                direction = "lemon_to_sugar" if row.endswith("-sugar") else "sugar_to_lemon"
                base = baselines[row][0]
                effects[(family, layer, direction, mode)] = (row, donor, position, base, margin, math.fsum((margin, -base)), s)
            complete += 1

        for layer in range(6):
            for row in names:
                forward(row, layer, 7, "capture_ingredient", None)
                forward(row, layer, 1, "capture_control", None)
            distances = {"ingredient": dist(captures[(names[0], layer, 7)], captures[(names[1], layer, 7)]),
                         "control": dist(captures[(names[0], layer, 1)], captures[(names[1], layer, 1)])}
            radius = min(distances.values())
            need(radius >= 1e-5 and math.isfinite(radius), "invalid radius")
            fractions = {site: radius / value for site, value in distances.items()}
            plan = events[cursor]
            cursor += 1
            need(plan["event"] == "dose_plan" and plan["row_id"] == names[0] and plan["frame"] == family
                 and typed_equal(plan["layer"], layer) and close(plan["radius"], radius), "dose plan")
            for site in distances:
                need(close(plan["distances"][site], distances[site]) and close(plan["fractions"][site], fractions[site])
                     and 0 <= fractions[site] <= 1, "dose plan values")
            doses.append({k: v for k, v in plan.items() if k not in ("event", "row_id")})
            for i, row in enumerate(names):
                other = names[1 - i]
                for mode, position, donor in (("ingredient_donor", 7, other), ("ingredient_self", 7, row),
                                              ("control_self", 1, row), ("ingredient_matched", 7, other),
                                              ("control_matched", 1, other)):
                    site = "ingredient" if position == 7 else "control"
                    forward(row, layer, position, mode, donor, radius, fractions[site])
        for row in names:
            forward(row, 0, 7, "final_clean", None)
        need(cursor == len(events), "extra events")

    need(complete == 516 and len(effects) == 360 and len(result["effects"]) == 360
         and typed_equal(result["doses"], doses), "result inventory")
    for item in result["effects"]:
        need(type(item["layer"]) is int and type(item["position"]) is int, "effect coordinate types")
        key = (item["frame"], item["layer"], item["direction"], item["mode"])
        need(key in effects, "missing or duplicate effect")
        row, donor, pos, base, margin, effect, summary = effects[key]
        need(item["recipient"] == row and item["donor"] == donor and item["position"] == pos
             and type(item["baseline_margin"]) is float and item["baseline_margin"] == base
             and type(item["patched_margin"]) is float and item["patched_margin"] == margin
             and type(item["effect"]) is float and item["effect"] == effect
             and typed_equal(item["summary"], summary), "effect record")
    need(len({(x["frame"], x["layer"], x["direction"], x["mode"]) for x in result["effects"]}) == 360, "duplicate effect")
    eligibility = {}
    for family in ("coffee", "milk", "soup", "cake", "pudding", "cocoa"):
        s, l = baselines[family + "-sugar"], baselines[family + "-lemon"]
        eligibility[family] = s[0] > 0 and l[0] < 0 and bool({4086, 14768}.intersection(s[2])) and bool({4086, 14768}.intersection(l[2]))
    need(typed_equal(result["eligibility"], eligibility), "eligibility")
    need(len(result["comparisons"]) == 72 and typed_equal([x["layer"] for x in result["layer_summary"]], list(range(6))), "comparison inventory")
    all_comparisons = {}
    for layer in range(6):
        for direction, sign in (("sugar_to_lemon", 1), ("lemon_to_sugar", -1)):
            selected = []
            for family, eligible in eligibility.items():
                ingredient = effects[(family, layer, direction, "ingredient_matched")][5]
                control = effects[(family, layer, direction, "control_matched")][5]
                contrast = math.fsum((sign * ingredient, -sign * control))
                all_comparisons[(family, layer, direction)] = (eligible, contrast)
                if eligible:
                    selected.append(contrast)
            summary = result["layer_summary"][layer][direction]
            expected_mean = math.fsum(selected) / len(selected) if selected else None
            need(typed_equal(summary["n"], len(selected)) and typed_equal(summary["mean"], expected_mean), "comparison mean")
    for item in result["comparisons"]:
        need(type(item["layer"]) is int and type(item["eligible"]) is bool
             and type(item["signed_advantage"]) is float, "comparison value types")
        key = (item["frame"], item["layer"], item["direction"])
        need(key in all_comparisons and (item["eligible"], item["signed_advantage"]) == all_comparisons.pop(key), "comparison")
    need(not all_comparisons, "missing comparisons")
    print(json.dumps({"status": "saved_consistency_verified", "completed_forwards": complete,
                      "patch_effects": len(effects), "dose_plans": len(doses),
                      "eligible_frames": [name for name, ok in eligibility.items() if ok],
                      "baseline_rows": len(baselines), "comparisons": len(result["comparisons"]),
                      "limitations": ["Saved consistency does not prove execution, publication chronology or process-tree cleanup.",
                                      "Whole-output and patched full-vocabulary checks are owner-observed."]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: audit_bert_taste_matched_dose_v1.py OUTPUT OBSERVER CASES")
    main(*sys.argv[1:])
