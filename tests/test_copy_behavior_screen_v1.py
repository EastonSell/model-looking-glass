"""Fabricated baseline scorer checks; no model or study prompts are used."""
import math
import subprocess
import sys
import struct
import unittest

from looking_glass import copy_behavior_screen_v1 as screen


def raw_row(target, target_value, other_value, *, other=3, enormous=False):
    values = [-20.0] * screen.VOCAB
    values[target], values[other] = target_value, other_value
    if enormous:
        values[0] = float(2 ** 60)
    return struct.pack("<%df" % screen.VOCAB, *values)


def fabricated(group=1, positive=0.5):
    cases, rows = [], []
    for variant in range(8):
        target = 13 if variant < 4 else 17
        cases.append({"case_id": f"fake-{group}-{variant}", "group": group,
                      "variant": variant, "target_id": target})
        repeat = variant % 4 == 0
        rows.append(raw_row(target, positive if repeat else 0.0,
                            0.0 if repeat else positive))
    return cases, rows


class ScreenTests(unittest.TestCase):
    def test_boundary_and_two_targets(self):
        cases, rows = fabricated()
        result = screen.score_family(cases, rows)
        self.assertEqual(result["baseline_status"], "supported")
        self.assertEqual(len(result["gaps"]), 6)
        self.assertTrue(all(g["gap"] == 0.5 for g in result["gaps"]))
        self.assertEqual([result["rows"][i]["target_id"] for i in (0, 4)], [13, 17])
        self.assertEqual(result["split"], "development")

    def test_below_threshold_and_exact_top_tie(self):
        cases, rows = fabricated(2, struct.unpack("<f", struct.pack("<I", 0x3effffff))[0])
        result = screen.score_family(cases, rows)
        self.assertEqual(result["baseline_status"], "unsupported")
        self.assertEqual(len(result["reasons"]), 6)
        self.assertTrue(all(x["reason"] == "gap-below-0.5" for x in result["reasons"]))
        cases, rows = fabricated(2)
        rows[0] = raw_row(13, 0.5, 0.5)
        result = screen.score_family(cases, rows)
        self.assertIsNone(result["rows"][0]["top_unique_id"])
        self.assertEqual(result["rows"][0]["top_tie_count"], 2)
        self.assertIn("repeat-target-not-unique-top", [x["reason"] for x in result["reasons"]])

    def test_normalizer_cancellation_avoids_rounded_log_probability_difference(self):
        repeat = screen.score_row(raw_row(13, 1.0, 0.0, enormous=True), 13)
        shuffle = screen.score_row(raw_row(13, 0.0, 1.0, enormous=True), 13)
        self.assertEqual(repeat["target_log_probability"], shuffle["target_log_probability"])
        self.assertEqual(screen.target_gap(repeat, shuffle), 1.0)

    def test_nonfinite_and_wrong_target_rejected(self):
        cases, rows = fabricated()
        bad = bytearray(rows[0])
        struct.pack_into("<f", bad, 4, float("nan"))
        rows[0] = bytes(bad)
        with self.assertRaises(ValueError):
            screen.score_family(cases, rows)
        a = screen.score_row(raw_row(13, 1.0, 0.0), 13)
        b = screen.score_row(raw_row(17, 1.0, 0.0), 17)
        with self.assertRaises(ValueError):
            screen.target_gap(a, b)

    def test_subnormal_top_survives_both_flush_settings(self):
        code = ('import json,struct,sys,torch; '
                'from looking_glass.copy_behavior_screen_v1 import score_row,ROW_BYTES; '
                'torch.set_flush_denormal(sys.argv[1] == "on"); '
                'raw=bytearray(ROW_BYTES); struct.pack_into("<I",raw,13*4,1); '
                'r=score_row(bytes(raw),13); '
                'print(json.dumps([r["target_logit"],r["top_unique_id"],r["top_tie_count"]]))')
        for mode in ("off", "on"):
            output = subprocess.check_output([sys.executable, "-B", "-c", code, mode], text=True)
            self.assertEqual(__import__("json").loads(output), [2 ** -149, 13, 1])

    def test_integer_decoder_boundaries(self):
        self.assertEqual(screen.decode_float32(0x00000001), 2 ** -149)
        self.assertEqual(screen.decode_float32(0x80000001), -(2 ** -149))
        self.assertEqual(screen.decode_float32(0x00800000), 2 ** -126)
        self.assertEqual(screen.decode_float32(0x3f800000), 1.0)
        self.assertLess(math.copysign(1.0, screen.decode_float32(0x80000000)), 0)
        for bits in (0x7f800000, 0xff800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                screen.decode_float32(bits)

    def test_fixed_split_all_24_and_no_replacement(self):
        cases, rows = [], []
        for group in range(1, 25):
            family_cases, family_rows = fabricated(group, 0.5 if group != 2 else 0.25)
            cases.extend(family_cases)
            rows.extend(family_rows)
        result = screen.score_screen(cases, rows)
        self.assertEqual(result["supported_development"], 12)
        self.assertEqual(result["supported_held_out"], 11)
        self.assertEqual(result["families"][1]["baseline_status"], "unsupported")
        self.assertEqual(len(result["families"]), 24)


if __name__ == "__main__":
    unittest.main()
