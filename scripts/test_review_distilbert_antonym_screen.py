"""Fabricated-only tests for the independent antonym screen reader."""

import base64
import copy
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_distilbert_antonym_screen as reader


def fixture():
    cases, events = [], []
    for candidate in reader.ORDER:
        factual, cue, accepted = reader.SPEC[candidate]
        for role in reader.ROLES:
            action = len(cases)
            words = {factual: 0, cue: 1, "red": 2}
            if len(accepted) == 2:
                words[accepted[1]] = 3
            row = {"id": f"fake-{candidate}-{role}", "object": candidate, "role": role,
                   "factual": factual, "counterfactual": cue, "wrong_target": "red",
                   "accepted": list(accepted), "word_ids": words,
                   "input_ids": [101, 200 + action, 103, 102], "attention_mask": [1, 1, 1, 1], "mask_index": 2}
            cases.append(row)
            common = {"action": action, "phase": "screen", "object": candidate, "case_id": row["id"],
                      "condition": "original", "input_ids": row["input_ids"][:],
                      "attention_mask": row["attention_mask"][:], "mask_index": 2}
            events.append({"seq": 2 * action, "kind": "start", **copy.deepcopy(common)})
            events.append({"seq": 2 * action + 1, "kind": "complete", **copy.deepcopy(common),
                           "parameter": None, "parameter_sha256": None, "realized_norm": 0.0})
            set_logits(events[-1], [3., 0., -2., -1., -3., -4.])
    manifest = json.dumps({"candidate_order": list(reader.ORDER), "cases": cases}, separators=(",", ":")).encode()
    return manifest, events


def set_logits(event, values):
    raw = struct.pack("<6f", *values)
    event["logits"] = {"encoding": "f32le", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                       "data": base64.b64encode(raw).decode()}


def run(manifest, events, suffix=b"\n"):
    data = b"\n".join(json.dumps(event, separators=(",", ":")).encode() for event in events) + suffix
    return reader.audit_stream(manifest, io.BytesIO(data), expected_manifest_sha=hashlib.sha256(manifest).hexdigest(), vocabulary=6)


class AntonymScreenTests(unittest.TestCase):
    def test_discovery_unique_top_with_calibration_pairwise_only(self):
        manifest, events = fixture()
        set_logits(events[1], [3., 0., -2., 3., -3., -4.])
        set_logits(events[2 * 4 + 1], [1., 0., -2., -1., 3., -4.])
        result = run(manifest, events)
        self.assertFalse(result["candidate_eligible"]["box"])
        self.assertEqual(result["selected_object"], "rope")
        self.assertTrue(result["rows"][4]["eligible"])
        self.assertEqual(result["rows"][4]["unique_top"], 4)
        self.assertEqual(len(result["rows"]), 12)

    def test_alternative_top_does_not_rescue_discovery(self):
        manifest, events = fixture()
        set_logits(events[1], [2., 0., -2., 3., -3., -4.])
        result = run(manifest, events)
        self.assertFalse(result["rows"][0]["eligible"])
        self.assertEqual(result["selected_object"], "rope")

    def test_exact_strict_threshold_adjacent_f32(self):
        below = struct.unpack("<f", bytes.fromhex("cccccc3d"))[0]
        above = struct.unpack("<f", bytes.fromhex("cdcccc3d"))[0]
        self.assertLess(below, 0.1)
        self.assertGreater(above, 0.1)
        manifest, events = fixture()
        set_logits(events[3], [below, 0., -2., -1., -3., -4.])
        result = run(manifest, events)
        self.assertFalse(result["rows"][1]["margin_pass"])
        self.assertEqual(result["selected_object"], "rope")
        set_logits(events[3], [above, 0., -2., -1., -3., -4.])
        self.assertEqual(run(manifest, events)["selected_object"], "box")

    def test_underfill_retains_all_twelve_rows(self):
        manifest, events = fixture()
        for done in events[1::2]:
            set_logits(done, [0., 1., -2., -1., -3., -4.])
        result = run(manifest, events)
        self.assertEqual(result["status"], "baseline-ineligible")
        self.assertIsNone(result["selected_object"])
        self.assertEqual(len(result["rows"]), 12)
        self.assertFalse(any(result["candidate_eligible"].values()))

    def test_adaptive_decimal_retains_300_nat_tail(self):
        result = reader.decimal_summary((0., -300., -301., -302.),
                                        {"factual": 0, "counterfactual": 1, "wrong_target": 2})
        actual = Decimal(result["shifted_log_normalizer_decimal"])
        expected = Decimal(str(math.exp(-300.) + math.exp(-301.) + math.exp(-302.)))
        self.assertGreater(actual, 0)
        self.assertLess(abs(actual / expected - 1), Decimal("1e-14"))
        two = reader.decimal_summary((0., -300.), {"factual": 0, "counterfactual": 1, "wrong_target": 1})
        self.assertAlmostEqual(float(Decimal(two["shifted_log_normalizer_decimal"])) / math.exp(-300.), 1.)
        self.assertGreaterEqual(two["decimal_precision_digits"], 100)

    def test_omitted_extreme_tail_is_reported_and_bounded(self):
        result = reader.decimal_summary((0., -1200.25, -1301., -1302.),
                                        {"factual": 0, "counterfactual": 1, "wrong_target": 2})
        self.assertEqual(result["omitted_tail_terms"], 3)
        bound = Decimal(result["log_normalizer_omission_upper_bound_decimal"])
        self.assertGreater(bound, 0)
        with localcontext() as ctx:
            ctx.prec = 760
            actual = (Decimal(1) + Decimal("-1200.25").exp()).ln()
            self.assertGreater(actual, 0)
            self.assertLessEqual(actual, bound)
        self.assertEqual(result["tokens"]["counterfactual"]["probability"], 0.0)

    def test_rounded_individual_subnormals_do_not_erase_pair_mass(self):
        result = reader.decimal_summary((-745.25, -745.25, 0., -1.),
                                        {"factual": 0, "counterfactual": 1, "wrong_target": 2})
        self.assertEqual(result["tokens"]["factual"]["probability"], 0.)
        self.assertEqual(result["tokens"]["counterfactual"]["probability"], 0.)
        self.assertEqual(result["pair_probability_mass"], float.fromhex("0x0.0000000000001p-1022"))

    def test_corrupt_bindings_values_and_truncated_journal_fail(self):
        manifest, events = fixture()
        bad = copy.deepcopy(events)
        bad[1]["input_ids"][-1] = 99
        with self.assertRaises(ValueError):
            run(manifest, bad)
        bad = copy.deepcopy(events)
        bad[1]["seq"] = True
        with self.assertRaises(ValueError):
            run(manifest, bad)
        bad = copy.deepcopy(events)
        set_logits(bad[1], [float("nan"), 0., -2., -1., -3., -4.])
        with self.assertRaises(ValueError):
            run(manifest, bad)
        with self.assertRaises(ValueError):
            run(manifest, events, suffix=b"")
        with self.assertRaises(ValueError):
            run(manifest, events, suffix=b"\nextra")

    def test_failure_metadata_and_start_action_extras_are_rejected(self):
        manifest, events = fixture()
        bad = copy.deepcopy(events)
        bad[1]["failure_stage"], bad[1]["failure_type"] = "restoration", "RuntimeError"
        with self.assertRaises(ValueError):
            run(manifest, bad)
        bad = copy.deepcopy(events)
        bad[0]["parameter"], bad[0]["realized_norm"] = "bias", 1.0
        with self.assertRaises(ValueError):
            run(manifest, bad)

    def test_denormal_guard_rejects_flush_and_precedes_json(self):
        original = struct.unpack
        def flushed(fmt, raw):
            if fmt == "<d" and raw == bytes.fromhex("0100000000000000"):
                return (0.0,)
            return original(fmt, raw)
        with patch.object(reader.struct, "unpack", side_effect=flushed):
            with self.assertRaises(ValueError):
                reader.require_denormal_support()
        with patch.object(reader, "require_denormal_support", side_effect=ValueError("flush")), \
             patch.object(reader, "_json") as parse:
            with self.assertRaises(ValueError):
                reader.audit_stream(b"{}", io.BytesIO(), expected_manifest_sha="0" * 64, vocabulary=6)
            parse.assert_not_called()

    def test_accepted_list_and_frozen_order_are_not_inferred(self):
        manifest, events = fixture()
        changed = json.loads(manifest)
        changed["cases"][0]["accepted"] = ["little", "small"]
        altered = json.dumps(changed).encode()
        with self.assertRaises(ValueError):
            run(altered, events)
        changed = json.loads(manifest)
        changed["candidate_order"].reverse()
        with self.assertRaises(ValueError):
            run(json.dumps(changed).encode(), events)


if __name__ == "__main__":
    unittest.main()
