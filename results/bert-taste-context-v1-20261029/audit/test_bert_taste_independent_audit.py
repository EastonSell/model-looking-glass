"""Fabricated saved-record checks; no model or checkpoint is loaded."""
import base64
import copy
import hashlib
import importlib.util
import json
import math
import struct
import unittest

import bert_taste_independent_audit as audit


def encode(vector):
    return struct.pack("<" + "f" * len(vector), *vector)


def summary(values):
    raw = encode(values)
    sweet, sour = values[1], values[2]
    maximum = max(values)
    denominator = math.fsum(math.exp(x - maximum) for x in values)
    return {"sweet_bits": struct.unpack_from("<I", raw, 4)[0],
            "sour_bits": struct.unpack_from("<I", raw, 8)[0],
            "sweet_logit": sweet, "sour_logit": sour,
            "sweet_minus_sour": math.fsum((sweet, -sour)),
            "sweet_probability": math.exp(sweet - maximum) / denominator,
            "sour_probability": math.exp(sour - maximum) / denominator,
            "top_five_ids": sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:5],
            "mask_logits_sha256": hashlib.sha256(raw).hexdigest()}


def make_fixture():
    rows = ("toy-sugar", "toy-lemon")
    baseline = {rows[0]: [0.0, 1.0, -1.0, 0.5, 0.25, -0.25, -0.5, -2.0],
                rows[1]: [0.0, -1.0, 1.0, 0.5, 0.25, -0.25, -0.5, -2.0]}
    captures = {(row, layer, pos): encode([float(index + 1), float(layer + 1), float(pos)])
                for index, row in enumerate(rows) for layer in range(2) for pos in (1, 7)}
    identities = []
    for layer in range(2):
        for row in rows:
            identities.extend(((row, layer, 7, "capture_ingredient", None),
                               (row, layer, 1, "capture_control", None)))
        for recipient, donor in ((rows[0], rows[1]), (rows[1], rows[0])):
            identities.extend(((recipient, layer, 7, "ingredient_swap", donor),
                               (recipient, layer, 7, "ingredient_self", recipient),
                               (recipient, layer, 1, "control_swap", donor)))
    identities.extend(((rows[0], 0, 7, "final_clean", None), (rows[1], 0, 7, "final_clean", None)))
    records, effects = [], []
    for row, layer, position, mode, donor in identities:
        identity = {"row_id": row, "layer": layer, "position": position, "mode": mode, "donor": donor}
        records.append({"event": "forward_start", **identity})
        before = captures[(row, layer, position)]
        is_patch = mode in ("ingredient_swap", "ingredient_self", "control_swap")
        source = None if not is_patch or mode == "ingredient_self" else captures[(donor, layer, position)]
        after = (before if mode == "ingredient_self" else source) if is_patch else None
        full_before = hashlib.sha256((row + str(layer)).encode()).hexdigest()
        values = list(baseline[row])
        if is_patch and mode != "ingredient_self" and layer == 0:
            values[1] += 0.25
        stats = summary(values)
        names = ["full_before", "before"] + (["patch", "after", "full_after"] if is_patch else [])
        events = []
        for order, name in enumerate(names):
            whole = name in ("full_before", "full_after")
            path = "distilbert.transformer.layer." + str(layer)
            events.append({"name": name, "operation": "patch" if name == "patch" else "capture", "order": order,
                           "module_path": path, "module_aliases": [path], "forward_index": 0, "invocation": 0,
                           "output_selector": [0], "output_shape": [1, 10, 3], "dtype": "torch.float32",
                           "device": "cpu", "preceding_same_site_actions": names[:order],
                           "batch_index": None if whole else 0, "token_index": None if whole else position,
                           "selected_shape": [1, 10, 3] if whole else [3],
                           "source_capture": "before" if name == "patch" and mode == "ingredient_self" else None})
        b64 = lambda raw: None if raw is None else base64.b64encode(raw).decode("ascii")
        sha = lambda raw: None if raw is None else hashlib.sha256(raw).hexdigest()
        probe = {"smallest_times_one_bits": 1, "normal_half_bits": 0x00400000}
        record = {"event": "forward_complete", **identity, "summary": stats,
                  "denormal_before": probe, "denormal_after": probe,
                  "baseline_mask_logits_float32_le_base64": b64(encode(values)) if mode == "capture_ingredient" and layer == 0 else None,
                  "before_sha256": sha(before), "after_sha256": sha(after), "source_sha256": sha(source),
                  "before_float32_le_base64": b64(before), "after_float32_le_base64": b64(after),
                  "source_float32_le_base64": b64(source), "full_before_sha256": full_before,
                  "full_after_sha256": None if not is_patch else (full_before if mode == "ingredient_self" else "a" * 64),
                  "untouched_positions_equal": True if is_patch else None, "events": events}
        records.append(record)
        if is_patch:
            base = summary(baseline[row])["sweet_minus_sour"]
            effects.append({"recipient": row, "donor": donor, "layer": layer, "position": position,
                            "mode": mode, "baseline_margin": base, "patched_margin": stats["sweet_minus_sour"],
                            "effect": math.fsum((stats["sweet_minus_sour"], -base))})
    layers = []
    for layer in range(2):
        item = {"layer": layer}
        for mode in ("ingredient_swap", "ingredient_self", "control_swap"):
            groups = (("lemon_self", "-lemon"), ("sugar_self", "-sugar")) if mode == "ingredient_self" else (
                ("sugar_to_lemon", "-lemon"), ("lemon_to_sugar", "-sugar"))
            item[mode] = {}
            for key, suffix in groups:
                vals = [e["effect"] for e in effects if e["layer"] == layer and e["mode"] == mode
                        and e["recipient"].endswith(suffix)]
                item[mode][key] = {"n": 1, "mean_effect": math.fsum(vals)}
        layers.append(item)
    result = {"status": "complete", "completed_forwards": 22, "effects": effects, "layer_summary": layers,
              "baselines": {row: {**summary(baseline[row]), "top_five_tokens": ["synthetic"] * 5} for row in rows}}
    return result, records


def encode_journal(result, records):
    raw = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in records)
    result["journals"] = {"toy.jsonl": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    return {"toy.jsonl": raw}


class SavedAuditTests(unittest.TestCase):
    def run_audit(self, result, records):
        return audit.audit_records(result, encode_journal(result, records),
                                   families=("toy",), layers=2, hidden=3, vocab=8, sweet=1, sour=2)

    def test_complete_fake_records(self):
        result, records = make_fixture()
        self.assertEqual(self.run_audit(result, records),
                         {"forwards_checked": 22, "directional_effects_checked": 12, "baseline_rows_checked": 2})

    def test_rehashed_changed_donor_is_rejected(self):
        result, records = make_fixture()
        target = next(r for r in records if r.get("event") == "forward_complete" and r.get("mode") == "ingredient_swap")
        changed = encode([4.0, 5.0, 6.0])
        target["source_float32_le_base64"] = base64.b64encode(changed).decode()
        target["source_sha256"] = hashlib.sha256(changed).hexdigest()
        with self.assertRaises(audit.AuditError):
            self.run_audit(result, records)

    def test_rehashed_wrong_effect_and_missing_event_are_rejected(self):
        result, records = make_fixture()
        wrong = copy.deepcopy(result)
        wrong["effects"][0]["effect"] += 1
        with self.assertRaises(audit.AuditError):
            self.run_audit(wrong, records)
        with self.assertRaises(audit.AuditError):
            self.run_audit(result, records[:-1])

    def test_rehashed_wrong_coordinate_and_control_are_rejected(self):
        result, records = make_fixture()
        bad_step = copy.deepcopy(records)
        row = next(r for r in bad_step if r.get("event") == "forward_complete" and r.get("mode") == "control_swap")
        row["events"][2]["token_index"] = 7
        with self.assertRaises(audit.AuditError):
            self.run_audit(copy.deepcopy(result), bad_step)
        bad_control = copy.deepcopy(records)
        row = next(r for r in bad_control if r.get("event") == "forward_complete" and r.get("mode") == "ingredient_self")
        row["summary"]["mask_logits_sha256"] = "0" * 64
        with self.assertRaises(audit.AuditError):
            self.run_audit(copy.deepcopy(result), bad_control)

    def test_boolean_numeric_alias_is_rejected(self):
        result, records = make_fixture()
        result["layer_summary"][0]["ingredient_swap"]["sugar_to_lemon"]["n"] = True
        with self.assertRaises(audit.AuditError):
            self.run_audit(result, records)

    def test_exact_subnormal_and_signed_zero_decoding(self):
        self.assertEqual(audit.float_bits(1), math.ldexp(1.0, -149))
        self.assertEqual(audit.float_bits(0x80000001), -math.ldexp(1.0, -149))
        self.assertEqual(math.copysign(1.0, audit.float_bits(0x80000000)), -1.0)
        self.assertEqual(math.copysign(1.0, audit.float_bits(0)), 1.0)
        with self.assertRaises(audit.AuditError):
            audit.float_bits(0x7f800000)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is unavailable")
    def test_exact_decoding_with_flush_enabled_and_disabled(self):
        import torch
        try:
            for flush in (False, True):
                if not torch.set_flush_denormal(flush):
                    self.skipTest("denormal control unavailable")
                self.test_exact_subnormal_and_signed_zero_decoding()
        finally:
            torch.set_flush_denormal(False)


if __name__ == "__main__":
    unittest.main()
