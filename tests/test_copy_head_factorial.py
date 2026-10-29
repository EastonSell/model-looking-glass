"""Synthetic numeric and projection tests; no pretrained forwards."""
import math
import random
import struct
import unittest

import torch

from looking_glass.copy_head_factorial import (
    HEADS, MatchedProjectionWitness, action_order, binary64_rounding_probe,
    f32_bits_to_f64, f64_to_f32_bits, head_slice, plan_family, projection_diagnostics,
    statistics, summarize,
)


class MatchedFactorialTests(unittest.TestCase):
    def test_binary32_decoding_and_nearest_even_boundaries(self):
        self.assertEqual(f32_bits_to_f64(1), 2.0 ** -149)
        self.assertEqual(f32_bits_to_f64(0x00800000), 2.0 ** -126)
        self.assertEqual(math.copysign(1.0, f32_bits_to_f64(0x80000000)), -1.0)
        expected = ((0.0, 0), (-0.0, 0x80000000), (2.0 ** -149, 1),
                    (2.0 ** -150, 0), (3.0 * 2.0 ** -150, 2),
                    (2.0 ** -126 - 2.0 ** -150, 0x00800000),
                    (1.0 + 2.0 ** -24, 0x3f800000),
                    (1.0 + 3.0 * 2.0 ** -24, 0x3f800002),
                    (-2.0 ** -150, 0x80000000))
        for value, bits in expected:
            with self.subTest(value=value):
                self.assertEqual(f64_to_f32_bits(value), bits)
        for bits in (0x7f800000, 0xff800000, 0x7fc00001):
            with self.assertRaises(ValueError):
                f32_bits_to_f64(bits)
        binary64_rounding_probe()

    def test_conversion_matches_reference_cast_for_random_finite_values(self):
        rng = random.Random(917)
        for _ in range(10_000):
            bits = rng.getrandbits(32)
            if bits & 0x7f800000 == 0x7f800000:
                continue
            value = f32_bits_to_f64(bits)
            self.assertEqual(f64_to_f32_bits(value), bits)
            scaled = value * rng.random()
            expected = struct.unpack("<I", struct.pack("<f", scaled))[0]
            self.assertEqual(f64_to_f32_bits(scaled), expected)

    def test_bit_conversion_ignores_float32_flush_mode(self):
        try:
            torch.set_flush_denormal(True)
            self.assertEqual(f32_bits_to_f64(1), 2.0 ** -149)
            self.assertEqual(f64_to_f32_bits(3.0 * 2.0 ** -150), 2)
            self.assertEqual(f64_to_f32_bits(-0.0), 0x80000000)
        finally:
            torch.set_flush_denormal(False)

    @staticmethod
    def inputs():
        rows = []
        for variant in range(4):
            x = torch.zeros((1, 11, 768), dtype=torch.float32)
            for head in HEADS:
                x[0, 10, head * 64:(head + 1) * 64] = (variant + head + 1) / 10.0
            rows.append(x.numpy().tobytes())
        return tuple(rows)

    def test_family_dose_and_exact_single_joint_delivery(self):
        source = self.inputs()
        plan = plan_family(source)
        self.assertGreater(plan.requested, 2.0 ** -20)
        self.assertAlmostEqual(plan.requested, min(dose.norm for row in plan.variants for dose in row.values()))
        for row in plan.variants:
            for dose in row.values():
                self.assertLessEqual(abs(dose.achieved - plan.requested), 1e-5 * plan.requested)
        projection = torch.nn.Linear(768, 768, bias=True).eval()
        x = torch.frombuffer(bytearray(source[0]), dtype=torch.float32).reshape(1, 11, 768)
        outputs = {}
        for name, heads in (("h10", (10,)), ("h2", (2,)), ("h7", (7,)), ("joint", (2, 10))):
            with MatchedProjectionWitness(projection, expected_before=source[0],
                                          deliveries={head: plan.variants[0][head].delivered for head in heads}) as witness:
                projection(x)
            raw = witness.after.numpy().tobytes()
            for head in heads:
                self.assertEqual(head_slice(raw, head), plan.variants[0][head].delivered)
            source_bits = list(struct.unpack("<8448I", source[0]))
            target_bits = list(struct.unpack("<8448I", raw))
            selected = {10 * 768 + head * 64 + index for head in heads for index in range(64)}
            self.assertTrue(all(source_bits[index] == target_bits[index]
                                for index in range(8448) if index not in selected))
            outputs[name] = raw
        for head in (2, 10):
            self.assertEqual(head_slice(outputs["joint"], head), head_slice(outputs[f"h{head}"], head))
        self.assertFalse(projection._forward_pre_hooks or projection._forward_hooks)

    def test_too_small_nonfinite_and_spoofed_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            plan_family((bytes(11 * 768 * 4),) * 4)
        row = bytearray(self.inputs()[0])
        struct.pack_into("<I", row, (10 * 768 + 10 * 64) * 4, 0x7fc00000)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            plan_family((bytes(row),) * 4)
        with self.assertRaises(ValueError):
            MatchedProjectionWitness(torch.nn.Identity(), expected_before=self.inputs()[0], deliveries={10.0: bytes(256)})

    def test_order_balance_and_bit_decoded_logit_score(self):
        for variant in range(4):
            orders = [action_order(family, variant) for family in range(8)]
            for slot in range(4):
                self.assertEqual(sorted(order[slot] for order in orders), sorted(("h10", "h2", "h7", "joint") * 2))
        raw = bytearray(50_257 * 4)
        struct.pack_into("<I", raw, 0, 1)
        struct.pack_into("<I", raw, 4, 0x80000000)
        score = statistics(bytes(raw), 0)
        self.assertEqual(score["target_logit"], 2.0 ** -149)
        self.assertEqual(score["target_rank"], 1)
        self.assertEqual(score["top_token_id"], 0)
        struct.pack_into("<I", raw, 4, 1)
        score = statistics(bytes(raw), 1)
        self.assertEqual((score["target_rank"], score["top_token_id"]), (1, 0))

    def test_summary_interaction_and_projection_residual(self):
        entries = []
        for family in range(8):
            schedule = [(variant, action) for variant in range(4) for action in ("clean", "noop")]
            schedule += [(variant, action) for variant in range(4)
                         for action in (*action_order(family, variant), "clean_after")]
            for variant, action in schedule:
                value = {"h10": -1.0, "h2": -2.0, "h7": 0.5, "joint": -4.0}.get(action, 0.0)
                if variant:
                    value /= 4
                entries.append({"group": family + 1, "variant": variant, "action": action,
                                "target_id": 3, "score": {"target_log_probability": value,
                                                           "target_rank": 1, "top_token_id": 3}})
        summary = summarize(entries)
        self.assertEqual(summary["mean_contrast_by_arm"]["h10"], -0.75)
        self.assertEqual(summary["mean_contrast_by_arm"]["h2"], -1.5)
        self.assertEqual(summary["mean_contrast_by_arm"]["joint"], -3.0)
        self.assertEqual(summary["mean_interaction"], -0.75)
        outputs = {name: torch.full((1, 11, 768), value, dtype=torch.float32).numpy().tobytes()
                   for name, value in (("noop", 1.0), ("h10", 2.0), ("h2", 3.0),
                                       ("h7", 4.0), ("joint", 4.0))}
        diagnostic = projection_diagnostics(outputs)
        self.assertEqual(diagnostic["joint_minus_singles_output_residual_l2"], 0.0)
        self.assertAlmostEqual(diagnostic["final_output_delta_l2"]["h10"], math.sqrt(768))

    def test_log_softmax_can_create_interaction_from_additive_logits(self):
        def logp(target):
            maximum = max(target, 0.0)
            return (target - maximum) - math.log(math.fsum((math.exp(target - maximum),
                                                             math.exp(-maximum))))
        def interaction(target):
            return ((logp(target + 2.0) - logp(target + 1.0))
                    - logp(target + 1.0)) + logp(target)
        contrast = interaction(0.0) - interaction(3.0)
        self.assertAlmostEqual(contrast, -0.174548972339, places=11)


if __name__ == "__main__":
    unittest.main()
