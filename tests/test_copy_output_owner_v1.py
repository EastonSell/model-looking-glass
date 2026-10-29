"""Fabricated schedule and metadata-bound checks; no pretrained model."""
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from scripts import copying_head_output_dose_v1 as owner
from looking_glass.copy_head_factorial import SHAPE_BYTES, _values, head_slice
from looking_glass.copy_head_output_dose import HEADS, project, weights


def pack(values):
    return struct.pack("<%df" % len(values), *values)


def fixture(zero_output=False):
    w = [0.0] * (64 * 768)
    w[0], w[1], w[768] = 1.0, -2.0, 17.0
    positive = pack(w)
    raw = {2: bytes(len(positive)) if zero_output else positive, 10: positive, 7: positive}
    matrix = weights(raw)
    noops = []
    for _ in range(4):
        x = [0.0] * (11 * 768)
        for h, value in ((2, 1.0), (10, 2.0), (7, 4.0)):
            x[10 * 768 + 64*h] = value
        noops.append(pack(x))
    return raw, matrix, noops


class ScheduleTests(unittest.TestCase):
    def run_family(self, zero_output=False, corrupt_actual=False):
        raw, matrix, noops = fixture(zero_output)
        cases = [{"case_id": f"synthetic-1-{i}", "group": 1, "variant": i, "token_ids": [i] * 11,
                  "target_id": i} for i in range(4)]
        calls = []
        def perform(case, kind, cache, head=None, delivery=None):
            variant = case["variant"]
            calls.append((variant, kind))
            if kind == "clean":
                cache["clean_score"] = {"target_log_probability": -float(variant), "target_rank": variant + 1}
            elif kind == "noop":
                cache["before"] = noops[variant]
                cache["noop_output"] = bytes(SHAPE_BYTES)
            elif head is not None:
                before, after = _values(head_slice(cache["before"], head)), _values(delivery)
                delta = tuple(b-a for a,b in zip(before, after))
                vector = list(project(delta, matrix[head]))
                if corrupt_actual and variant == 0 and kind == "out_h2":
                    vector[0] += 0.25
                values = [0.0] * (10 * 768) + vector
                cache.setdefault("arm_outputs", {})[kind] = pack(values)
                effect = {2: -1.0, 10: -0.5, 7: 0.1}[head] * (1.0 if variant == 0 else 0.1)
                cache.setdefault("comparisons", {})[kind] = {"target_log_probability_delta": effect,
                                                               "target_logit_delta": effect}
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "events.jsonl").open("xb") as stream:
                result = owner.execute_family(cases, perform, lambda: None, owner.BoundedJournal(stream), raw)
        return result, calls

    def test_complete_rotation_and_separate_summaries(self):
        result, calls = self.run_family()
        self.assertEqual(len(calls), 36)
        self.assertEqual(calls[:8], [(i, k) for i in range(4) for k in ("clean", "noop")])
        for variant in range(4):
            observed = [k for i, k in calls if i == variant and k not in ("clean", "noop", "clean_after")]
            self.assertEqual(tuple(observed), owner.rotation(1, variant))
        self.assertEqual(result["summaries"]["output"]["status"], "evaluable")
        self.assertEqual(result["summaries"]["input"]["status"], "evaluable")
        self.assertGreater(result["summaries"]["output"]["contrast"], 0)

    def test_output_floor_keeps_secondary_input_arms(self):
        result, calls = self.run_family(zero_output=True)
        self.assertEqual(len(calls), 24)
        self.assertFalse(any(kind.startswith("out_") for _, kind in calls))
        self.assertEqual(result["summaries"]["output"]["status"], "dose-ineligible")
        self.assertEqual(result["summaries"]["input"]["status"], "evaluable")

    def test_actual_mismatch_keeps_all_planned_arms(self):
        result, calls = self.run_family(corrupt_actual=True)
        self.assertEqual(len(calls), 36)
        self.assertEqual(result["summaries"]["output"]["status"], "dose-ineligible")
        self.assertEqual(result["summaries"]["input"]["status"], "evaluable")

    def test_cumulative_journal_and_json_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events"
            with path.open("xb") as stream, mock.patch.object(owner, "MAX_JOURNAL_BYTES", 4):
                log = owner.BoundedJournal(stream)
                self.assertEqual(log.write(b"1234"), 4)
                with self.assertRaises(OSError):
                    log.write(b"5")
            self.assertEqual(path.read_bytes(), b"1234")
            with self.assertRaises(OSError):
                owner.bounded_json(Path(directory) / "result.json", {"too_large": "hello"}, 4)
            self.assertFalse((Path(directory) / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
