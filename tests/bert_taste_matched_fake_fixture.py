"""Fabricate a full-width, no-model taste journal for observer regression.

Adapted from the independent registry fixture; all tensors/logits are invented.
"""
import base64
import hashlib
import json
import math
import struct


def make_fixture(output, cases):
    sha = lambda data: hashlib.sha256(data).hexdigest()
    encode = lambda raw: base64.b64encode(raw).decode("ascii")
    rows = {row["id"]: row for row in cases["rows"]}
    result = {"status": "complete", "completed_forwards": 516, "source": {"commit": "a" * 40},
              "baselines": {}, "effects": [], "doses": [], "eligibility": {}, "comparisons": [],
              "model": {"model": cases["model"], "revision": cases["revision"], "device": "cpu", "dtype": "float32"},
              "journals": {}}
    zero = struct.pack("<768f", *([0.] * 768))
    one = struct.pack("<768f", *([1.] + [0.] * 767))
    two = struct.pack("<768f", *([2.] + [0.] * 767))
    bits = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]


    def summary(sugar):
        logits = [0.] * 30522
        sweet, sour = (3., 2.) if sugar else (2., 3.)
        for i, value in ((4086, sweet), (14768, sour), (1, 1.5), (2, 1.), (3, .5)):
            logits[i] = value
        raw = struct.pack("<30522f", *logits)
        denominator = math.fsum(math.exp(x - 3) for x in logits)
        top = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:5]
        return raw, {"sweet_bits": bits(sweet), "sour_bits": bits(sour), "sweet_logit": sweet, "sour_logit": sour,
                     "sweet_minus_sour": sweet - sour, "sweet_probability": math.exp(sweet - 3) / denominator,
                     "sour_probability": math.exp(sour - 3) / denominator, "top_five_ids": top,
                     "top_five_tokens": [str(x) for x in top], "mask_logits_sha256": sha(raw)}
    for family in ("coffee", "milk", "soup", "cake", "pudding", "cocoa"):
        names = [family + "-sugar", family + "-lemon"]
        result["eligibility"][family] = True
        events = []

        def emit(row, layer, position, mode, donor=None):
            identity = {"row_id": row, "layer": layer, "position": position, "mode": mode, "donor": donor}
            events.append({"event": "forward_start", **identity})
            sugar = row.endswith("-sugar")
            before = zero if sugar else (two if position == 7 else one)
            patched = mode in ("ingredient_donor", "ingredient_self", "control_self", "ingredient_matched", "control_matched")
            donor_clean = None if not patched or mode.endswith("_self") else (two if sugar and position == 7 else one if sugar else zero)
            source = None if not patched or mode.endswith("_self") else (one if position == 7 and mode.endswith("_matched") else donor_clean)
            after = None if not patched else (before if source is None else source)
            raw, s = summary(sugar)
            plan = ["full_before", "before", "patch", "after", "full_after"] if patched else ["full_before", "before"]
            steps = []
            for i, name in enumerate(plan):
                selected = name in ("before", "patch", "after")
                module = "distilbert.transformer.layer." + str(layer)
                steps.append({"name": name, "order": i, "forward_index": 0, "invocation": 0,
                              "module_path": module, "module_aliases": [module], "output_selector": [0],
                              "output_shape": [1, 10, 768], "token_index": position if selected else None,
                              "batch_index": 0 if selected else None, "call_site_known": False,
                              "dtype": "torch.float32", "device": "cpu",
                              "operation": "patch" if name == "patch" else "capture",
                              "selected_shape": [768] if selected else [1, 10, 768],
                              "source_capture": "before" if name == "patch" and mode.endswith("_self") else None,
                              "preceding_same_site_actions": plan[:i]})
            matched = mode.endswith("_matched")
            fraction = .5 if position == 7 else 1.
            recipient_norm = 0. if sugar else (2. if position == 7 else 1.)
            dose = ({"fraction": fraction, "requested_norm": 1., "delivered_norm": 1., "relative_error": 0.,
                     "recipient_norm": recipient_norm, "recipient_norm_status": "zero" if sugar else "positive",
                     "dose_to_recipient_norm": None if sugar else 1. / recipient_norm} if matched else None)
            full_positions = [zero] * 10
            full_positions[1] = zero if sugar else one
            full_positions[7] = zero if sugar else two
            full_before = b"".join(full_positions)
            if patched:
                full_positions[position] = after
            full_after = b"".join(full_positions) if patched else None
            rec = {"event": "forward_complete", **identity,
                   "model_entry": {"input_ids": rows[row]["input_ids"], "attention_mask": rows[row]["attention_mask"]},
                   "summary": s, "denormal_before": {"smallest_times_one_bits": 1, "normal_half_bits": 4194304},
                   "denormal_after": {"smallest_times_one_bits": 1, "normal_half_bits": 4194304},
                   "denormal_after_scoring": {"smallest_times_one_bits": 1, "normal_half_bits": 4194304},
                   "baseline_mask_logits_float32_le_base64": encode(raw) if mode == "capture_ingredient" and layer == 0 else None,
                   "untouched_positions_equal": True if patched else None, "full_before_sha256": sha(full_before),
                   "full_after_sha256": None if full_after is None else sha(full_after), "dose": dose,
                   "actual_pre_post_norm": 1. if matched else None, "events": steps}
            for key, vector in (("before", before), ("after", after), ("source", source), ("donor_clean", donor_clean)):
                rec[key + "_float32_le_base64"] = None if vector is None else encode(vector)
            events.append(rec)
            if mode == "capture_ingredient" and layer == 0:
                result["baselines"][row] = s
            if patched:
                direction = "lemon_to_sugar" if sugar else "sugar_to_lemon"
                result["effects"].append({"frame": family, "recipient": row, "donor": donor, "layer": layer,
                                          "position": position, "direction": direction, "mode": mode,
                                          "baseline_margin": s["sweet_minus_sour"], "patched_margin": s["sweet_minus_sour"],
                                          "effect": 0., "summary": s})

        for layer in range(6):
            for row in names:
                emit(row, layer, 7, "capture_ingredient")
                emit(row, layer, 1, "capture_control")
            dose = {"frame": family, "layer": layer, "radius": 1., "distances": {"ingredient": 2., "control": 1.},
                    "fractions": {"ingredient": .5, "control": 1.}}
            events.append({"event": "dose_plan", "row_id": names[0], **dose})
            result["doses"].append(dose)
            for i, row in enumerate(names):
                donor = names[1 - i]
                for mode, pos, source in (("ingredient_donor", 7, donor), ("ingredient_self", 7, row),
                                          ("control_self", 1, row), ("ingredient_matched", 7, donor),
                                          ("control_matched", 1, donor)):
                    emit(row, layer, pos, mode, source)
        for row in names:
            emit(row, 0, 7, "final_clean")
        raw = b"".join((json.dumps(event, separators=(",", ":")) + "\n").encode() for event in events)
        (output / (family + ".jsonl")).write_bytes(raw)
        result["journals"][family + ".jsonl"] = {"size": len(raw), "sha256": sha(raw)}
    result["layer_summary"] = []
    for layer in range(6):
        item = {"layer": layer}
        for direction in ("sugar_to_lemon", "lemon_to_sugar"):
            item[direction] = {"n": 6, "mean": 0.}
            for family in result["eligibility"]:
                result["comparisons"].append({"frame": family, "layer": layer, "direction": direction,
                                              "eligible": True, "signed_advantage": 0.})
        result["layer_summary"].append(item)
    return result
