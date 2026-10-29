"""Recompute retained public BERT taste evidence; does not load a model."""

import base64
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(path.read_bytes())


def scalar(bits):
    need(type(bits) is int and 0 <= bits < 2**32, "invalid float bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    need(math.isfinite(value), "nonfinite value")
    return value


def vector(record, name):
    text = record[name + "_float32_le_base64"]
    if text is None:
        need(record[name + "_sha256"] is None, name + " null digest")
        return None
    value = base64.b64decode(text, validate=True)
    need(len(value) == 768 * 4 and digest(value) == record[name + "_sha256"], name + " bytes")
    return value


def equal_number(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and a == b


def main(output, receipt_path, cases_path):
    output, receipt_path, cases_path = map(Path, (output, receipt_path, cases_path))
    need(not (output / "incomplete.json").exists(), "incomplete marker")
    result_raw = (output / "result.json").read_bytes()
    result, receipt, cases = json.loads(result_raw), load_json(receipt_path), load_json(cases_path)
    need(receipt["status"] == "complete" and receipt["child_reaped"] is True
         and type(receipt["child_exit_code"]) is int and receipt["child_exit_code"] == 0
         and receipt["source_unchanged"] is True and receipt["bytecode_prefix_empty"] is True
         and receipt["result_sha256"] == digest(result_raw), "observer receipt")
    need(result["status"] == "complete" and result["completed_forwards"] == 248
         and receipt["source_commit"] == result["source"]["commit"]
         and receipt["journals"] == result["journals"], "result identity")
    rows = cases["rows"]
    need(len(rows) == 8, "case inventory")
    need(set(result["baselines"]) == {row["id"] for row in rows}, "baseline inventory")
    need([item["layer"] for item in result["layer_summary"]] == list(range(6)), "layer inventory")
    need(result["model"]["model"] == cases["model"] and result["model"]["revision"] == cases["revision"]
         and result["model"]["device"] == "cpu" and result["model"]["dtype"] == "float32", "model identity")
    effects = {}
    completed = 0
    for offset, family in enumerate(("tea", "drink", "sauce", "water")):
        pair = rows[offset * 2:offset * 2 + 2]
        names = [family + "-sugar", family + "-lemon"]
        need([row["id"] for row in pair] == names, "row order")
        journal_path = output / (family + ".jsonl")
        data = journal_path.read_bytes()
        need(result["journals"][journal_path.name] == {"size": len(data), "sha256": digest(data)}, "journal hash")
        lines = data.splitlines()
        need(len(lines) == 124 and data.endswith(b"\n"), "journal length")
        events = [json.loads(line) for line in lines]
        expected = []
        for layer in range(6):
            for row in names:
                expected.extend([(row, layer, 7, "capture_ingredient", None),
                                 (row, layer, 1, "capture_control", None)])
            for index, row in enumerate(names):
                donor = names[1 - index]
                expected.extend([(row, layer, 7, "ingredient_swap", donor),
                                 (row, layer, 7, "ingredient_self", row),
                                 (row, layer, 1, "control_swap", donor)])
        expected.extend((row, 0, 7, "final_clean", None) for row in names)
        captures, baselines = {}, {}
        for index, identity in enumerate(expected):
            start, done = events[2 * index:2 * index + 2]
            keys = ("row_id", "layer", "position", "mode", "donor")
            need(start["event"] == "forward_start" and done["event"] == "forward_complete"
                 and tuple(start[key] for key in keys) == identity
                 and tuple(done[key] for key in keys) == identity, "forward identity/order")
            row, layer, position, mode, donor = identity
            need(done["denormal_before"] == {"smallest_times_one_bits": 1, "normal_half_bits": 0x00400000}
                 and done["denormal_after"] == done["denormal_before"], "denormal check")
            before, after, source = (vector(done, name) for name in ("before", "after", "source"))
            need(before is not None, "before capture absent")
            steps = done["events"]
            patch = mode in ("ingredient_swap", "ingredient_self", "control_swap")
            plan = (["full_before", "before", "patch", "after", "full_after"] if patch
                    else ["full_before", "before"])
            need([step["name"] for step in steps] == plan, "plan steps")
            for number, step in enumerate(steps):
                selected = step["name"] in ("before", "patch", "after")
                need(step["order"] == number and step["forward_index"] == 0 and step["invocation"] == 0
                     and step["module_path"] == "distilbert.transformer.layer." + str(layer)
                     and step["output_selector"] == [0] and step["output_shape"] == [1, 10, 768]
                     and step["token_index"] == (position if selected else None)
                     and step["batch_index"] == (0 if selected else None)
                     and step["device"] == "cpu" and step["dtype"] == "torch.float32", "plan coordinates")
            summary = done["summary"]
            sweet, sour = scalar(summary["sweet_bits"]), scalar(summary["sour_bits"])
            margin = math.fsum((sweet, -sour))
            need(equal_number(summary["sweet_logit"], sweet)
                 and equal_number(summary["sour_logit"], sour)
                 and equal_number(summary["sweet_minus_sour"], margin), "target logits")
            if mode.startswith("capture_"):
                need(after is None and source is None and done["untouched_positions_equal"] is None, "capture-only action")
                captures[(row, layer, position)] = before
            if mode == "capture_ingredient" and layer == 0:
                raw = base64.b64decode(done["baseline_mask_logits_float32_le_base64"], validate=True)
                need(len(raw) == 30522 * 4 and digest(raw) == summary["mask_logits_sha256"], "baseline logits bytes")
                logits = struct.unpack("<30522f", raw)
                need(all(math.isfinite(x) for x in logits) and logits[4086] == sweet and logits[14768] == sour, "baseline logits")
                maxima = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
                need(set(maxima) == set(summary["top_five_ids"]), "top-five")
                maximum = max(logits)
                denominator = math.fsum(math.exp(x - maximum) for x in logits)
                for word, token in (("sweet", 4086), ("sour", 14768)):
                    probability = math.exp(logits[token] - maximum) / denominator
                    need(math.isclose(probability, summary[word + "_probability"], rel_tol=1e-12, abs_tol=1e-15), "baseline softmax")
                baselines[row] = (margin, summary["mask_logits_sha256"])
                for key, value in summary.items():
                    need(result["baselines"][row][key] == value, "reported baseline")
            else:
                need(done["baseline_mask_logits_float32_le_base64"] is None, "unexpected full logits")
            if mode.startswith("capture_") or mode == "final_clean":
                need(summary["mask_logits_sha256"] == baselines[row][1], "clean replay")
            if mode in ("ingredient_swap", "control_swap", "ingredient_self"):
                need(before == captures[(row, layer, position)] and done["untouched_positions_equal"] is True, "recipient capture")
                if mode == "ingredient_self":
                    need(source is None and after == before
                         and done["full_before_sha256"] == done["full_after_sha256"], "self landing")
                else:
                    need(source == captures[(donor, layer, position)] == after, "donor landing")
                if mode == "ingredient_self" or layer == 5:
                    need(summary["mask_logits_sha256"] == baselines[row][1], "self/final-block replay")
                base = baselines[row][0]
                effects[(row, layer, position, mode, donor)] = (base, margin, math.fsum((margin, -base)))
            completed += 1
    need(completed == 248 and len(effects) == 144 and len(result["effects"]) == 144, "effect inventory")
    for item in result["effects"]:
        key = tuple(item[name] for name in ("recipient", "layer", "position", "mode", "donor"))
        need(key in effects, "unexpected or duplicate effect")
        values = effects.pop(key)
        need(all(equal_number(item[name], value) for name, value in zip(
            ("baseline_margin", "patched_margin", "effect"), values)), "reported effect")
    need(not effects, "missing effects")
    for layer in result["layer_summary"]:
        for mode in ("ingredient_swap", "ingredient_self", "control_swap"):
            for direction, suffix in (("lemon_self", "-lemon"), ("sugar_self", "-sugar")) if mode == "ingredient_self" else (("sugar_to_lemon", "-lemon"), ("lemon_to_sugar", "-sugar")):
                values = [e["effect"] for e in result["effects"] if e["layer"] == layer["layer"]
                          and e["mode"] == mode and e["recipient"].endswith(suffix)]
                need(len(values) == 4 and layer[mode][direction]["n"] == 4
                     and equal_number(layer[mode][direction]["mean_effect"], math.fsum(values) / 4), "layer mean")
    print(json.dumps({"completed_forwards": completed, "per_case_patch_effects": len(result["effects"]),
                      "baseline_rows": len(result["baselines"]), "status": "saved_consistency_verified",
                      "limitations": ["Does not prove execution or publication chronology.",
                                      "Full block-output preservation and patched full-vocabulary summaries are owner-observed."]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: audit_bert_taste_v1.py OUTPUT OBSERVER CASES")
    main(*sys.argv[1:])
