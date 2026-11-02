"""Fabricated geometry and output-dose cases; no model or new study input."""
import math
import struct
import unittest

import torch

from looking_glass.copy_head_factorial import SHAPE_BYTES
from looking_glass.copy_head_output_dose import (
    HEADS, WEIGHT_BYTES, inspect_actual, plan_family, project, weights,
)


def pack(values):
    return struct.pack("<%df" % len(values), *values)


def fixture():
    w = [0.0] * (64 * 768)
    w[0], w[1], w[768] = 1.0, -2.0, 17.0
    positive = pack(w)
    negative = pack([-value for value in w])
    row_weights = {2: positive, 10: positive, 7: negative}
    assert all(len(raw) == WEIGHT_BYTES for raw in row_weights.values())
    noops = []
    for _ in range(4):
        values = [0.0] * (11 * 768)
        for h, x in ((2, 1.0), (10, 2.0), (7, 4.0)):
            values[10 * 768 + 64 * h] = x
        noops.append(pack(values))
    return tuple(noops), row_weights


def actual_outputs():
    noop = bytes(SHAPE_BYTES)
    rows = []
    for _ in range(4):
        row = {}
        for h, sign in ((2, 1), (10, 1), (7, -1)):
            values = [0.0] * (11 * 768)
            values[10 * 768], values[10 * 768 + 1] = -sign, 2 * sign
            row[h] = pack(values)
        rows.append(row)
    return (noop,) * 4, tuple(rows)


class OutputDoseTests(unittest.TestCase):
    def test_projection_uses_input_rows_and_signed_output(self):
        noops, raw = fixture()
        x = (1.0,) + (0.0,) * 63
        y = project(x, weights(raw)[2])
        self.assertEqual(y[:2], (1.0, -2.0))
        self.assertNotEqual(y[:2], (1.0, 17.0))
        plan = plan_family(noops, raw)
        self.assertEqual(plan["status"], "eligible")
        self.assertAlmostEqual(plan["q_out"], math.sqrt(5.0))
        self.assertEqual(plan["plans"][0][2]["predicted_vector"][:2], (-1.0, 2.0))
        self.assertEqual(plan["plans"][0][7]["predicted_vector"][:2], (1.0, -2.0))
        self.assertEqual([plan["plans"][0][h]["fraction"] for h in HEADS], [1.0, 0.5, 0.25])
        n, arms = actual_outputs()
        observed = inspect_actual(n, arms, plan)
        self.assertEqual(observed["status"], "eligible")
        self.assertAlmostEqual(observed["pair_cosines"][0]["2-7"], -1.0)

    def test_zero_family_dose_skips_all_plans_without_division(self):
        noops, raw = fixture()
        zeroed = bytearray(noops[0])
        offset = (10 * 768 + 64 * 2) * 4
        zeroed[offset:offset + 4] = b"\0" * 4
        plan = plan_family((bytes(zeroed), *noops[1:]), raw)
        self.assertEqual(plan["status"], "dose-ineligible")
        self.assertEqual(plan["reason"], "dose floor")
        self.assertIsNone(plan["plans"])

    def test_actual_numeric_mismatch_is_ineligible(self):
        noops, raw = fixture()
        plan = plan_family(noops, raw)
        outputs, arms = actual_outputs()
        changed = [dict(row) for row in arms]
        data = bytearray(changed[0][10])
        struct.pack_into("<f", data, (10 * 768 + 1) * 4, 2.1)
        changed[0][10] = bytes(data)
        result = inspect_actual(outputs, tuple(changed), plan)
        self.assertEqual(result["status"], "dose-ineligible")
        self.assertEqual(len(result["variants"]), 4)

    def test_wrong_direction_with_same_norm_is_ineligible(self):
        noops, raw = fixture()
        plan = plan_family(noops, raw)
        outputs, arms = actual_outputs()
        changed = [dict(row) for row in arms]
        changed[0][2] = changed[0][7]
        result = inspect_actual(outputs, tuple(changed), plan)
        self.assertEqual(result["status"], "dose-ineligible")
        self.assertAlmostEqual(result["variants"][0]["2"]["actual_output_l2"], plan["q_out"])
        self.assertGreater(result["variants"][0]["2"]["predicted_vector_residual_l2"], 0)

    def test_tiny_nonzero_output_has_no_direction_cosine(self):
        noops, raw = fixture()
        plan = plan_family(noops, raw)
        outputs, arms = actual_outputs()
        changed = [dict(row) for row in arms]
        data = bytearray(SHAPE_BYTES)
        struct.pack_into("<f", data, 10 * 768 * 4, 2.0 ** -10)
        changed[0][2] = bytes(data)
        result = inspect_actual(outputs, tuple(changed), plan)
        self.assertEqual(result["status"], "dose-ineligible")
        self.assertIsNone(result["pair_cosines"][0]["2-10"])
        self.assertIsNone(result["pair_cosines"][0]["2-7"])

    def test_other_token_output_mutation_is_invalid(self):
        noops, raw = fixture()
        plan = plan_family(noops, raw)
        outputs, arms = actual_outputs()
        changed = [dict(row) for row in arms]
        data = bytearray(changed[0][2])
        struct.pack_into("<f", data, 0, 1.0)
        changed[0][2] = bytes(data)
        with self.assertRaises(ValueError):
            inspect_actual(outputs, tuple(changed), plan)

    def test_fabricated_float32_affine_landing(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(6123)
        weight = torch.randn((768, 768), generator=generator, dtype=torch.float32) / 24
        bias = torch.randn((768,), generator=generator, dtype=torch.float32) / 10
        tensors = tuple(torch.randn((11, 768), generator=generator, dtype=torch.float32) for _ in range(4))
        slices = {h: weight[64*h:64*(h+1)].contiguous().numpy().tobytes() for h in HEADS}
        noops = tuple(x.contiguous().numpy().tobytes() for x in tensors)
        plan = plan_family(noops, slices)
        self.assertEqual(plan["status"], "eligible")
        outputs, rows = [], []
        for variant, x in enumerate(tensors):
            outputs.append((x @ weight + bias).contiguous().numpy().tobytes())
            row = {}
            for h in HEADS:
                edited = x.clone()
                delivered = torch.frombuffer(bytearray(plan["plans"][variant][h]["delivered"]), dtype=torch.float32)
                edited[10, 64*h:64*(h+1)] = delivered
                row[h] = (edited @ weight + bias).contiguous().numpy().tobytes()
            rows.append(row)
        actual = inspect_actual(tuple(outputs), tuple(rows), plan)
        self.assertEqual(actual["status"], "eligible")


if __name__ == "__main__":
    unittest.main()
