"""CPU-only tests for the known-injection number-study executor."""
import hashlib
import os
import random
import struct
import subprocess
import sys
import unittest

import torch

from looking_glass.pretrained_number_injection import (
    add_float32_bits,
    injection_direction_bits,
    verify_cpu_float32_addition,
    verify_injection_bits,
)


class Float32InjectionArithmeticTests(unittest.TestCase):
    def test_addition_rounds_exactly_including_subnormals_and_signed_zero(self):
        rows = [
            (0x00000000, 0x80000000, 0x00000000),
            (0x80000000, 0x80000000, 0x80000000),
            (0x3f800000, 0xbf800000, 0x00000000),
            (0x3f800000, 0x33800000, 0x3f800000),
            (0x3f800001, 0x33800000, 0x3f800002),
            (0x00000001, 0x00000001, 0x00000002),
            (0x00800000, 0x807fffff, 0x00000001),
            (0x7f7fffff, 0x7f7fffff, 0x7f800000),
            (0xff7fffff, 0xff7fffff, 0xff800000),
        ]
        for left, right, expected in rows:
            with self.subTest(left=hex(left), right=hex(right)):
                self.assertEqual(add_float32_bits(left, right), expected)
                self.assertEqual(add_float32_bits(right, left), expected)

    def test_nonfinite_and_noninteger_inputs_are_rejected(self):
        for left in (True, -1, 1 << 32, 0x7f800000, 0x7fc00001):
            with self.subTest(left=left), self.assertRaises(ValueError):
                add_float32_bits(left, 0)

    def test_fixed_direction_bits_and_seed_are_stable(self):
        bits = injection_direction_bits()
        self.assertEqual(len(bits), 768)
        self.assertEqual(bits[:8], (0x3c93cd3a, 0x3c93cd3a, 0xbc93cd3a, 0xbc93cd3a,
                                    0xbc93cd3a, 0xbc93cd3a, 0x3c93cd3a, 0xbc93cd3a))
        self.assertEqual(hashlib.sha256(struct.pack("<768I", *bits)).hexdigest(),
                         "fee167bdb27a7dc195a0d280df78e07fe6f6fff5218f6f07299e7d4bce6f98b7")

    def test_cpu_runtime_matches_reference_for_varied_finite_values(self):
        verify_cpu_float32_addition()
        rng = random.Random(29)
        rows = []
        while len(rows) < 4096:
            left, right = rng.getrandbits(32), rng.getrandbits(32)
            if ((left >> 23) & 255) != 255 and ((right >> 23) & 255) != 255:
                rows.append((left, right))
        signed = lambda value: value if value < (1 << 31) else value - (1 << 32)
        left = torch.tensor([signed(a) for a, _ in rows], dtype=torch.int32).view(torch.float32)
        right = torch.tensor([signed(b) for _, b in rows], dtype=torch.int32).view(torch.float32)
        actual = torch.add(left, right).view(torch.int32).tolist()
        self.assertEqual([value & 0xffffffff for value in actual],
                         [add_float32_bits(a, b) for a, b in rows])

    def test_runtime_guard_rejects_flush_to_zero_when_supported(self):
        code = """import sys, torch
from looking_glass.pretrained_number_injection import verify_cpu_float32_addition
if not torch.set_flush_denormal(True):
    sys.exit(77)
try:
    verify_cpu_float32_addition()
except RuntimeError:
    sys.exit(0)
sys.exit(1)
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        result = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, timeout=30)
        if result.returncode == 77:
            self.skipTest("runtime does not expose denormal flushing")
        self.assertEqual(result.returncode, 0, "the runtime sentinel did not reject flush-to-zero")

    def test_injection_requires_exact_bits_finite_values_and_actual_change(self):
        direction = injection_direction_bits()
        before = (0,) * 768
        pre = struct.pack("<768I", *before)
        post = struct.pack("<768I", *direction)
        self.assertEqual(verify_injection_bits(pre, post), {"changed_coordinates": 768})
        bad = list(direction)
        bad[0] ^= 1
        with self.assertRaises(ValueError):
            verify_injection_bits(pre, struct.pack("<768I", *bad))
        bad[0] = 0x7f800000
        with self.assertRaises(ValueError):
            verify_injection_bits(pre, struct.pack("<768I", *bad))
        unchanged = struct.pack("<768I", *((0x7f000000,) * 768))
        with self.assertRaises(ValueError):
            verify_injection_bits(unchanged, unchanged)
        with self.assertRaises(ValueError):
            verify_injection_bits(b"", post)


if __name__ == "__main__":
    unittest.main()
