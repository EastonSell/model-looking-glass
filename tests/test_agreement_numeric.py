"""Synthetic stored-float checks; no checkpoint or CUDA is used."""
import math
import os
import subprocess
import sys
import tempfile
import unittest

from looking_glass.agreement_numeric import NumericInvalid, compare_f32_le, decode_f32_le


def packed(*values):
    return b''.join(value.to_bytes(4, 'little') for value in values)


class AgreementNumericTests(unittest.TestCase):
    def test_decodes_signed_zero_subnormal_and_maximum_without_float32_cast(self):
        values = decode_f32_le(packed(0, 0x80000000, 1, 0x80000001, 0x7f7fffff))
        self.assertEqual(math.copysign(1.0, values[0]), 1.0)
        self.assertEqual(math.copysign(1.0, values[1]), -1.0)
        self.assertEqual(values[2], math.ldexp(1.0, -149))
        self.assertEqual(values[3], -math.ldexp(1.0, -149))
        self.assertEqual(values[4], math.ldexp(float((1 << 24) - 1), 104))

    def test_nan_infinity_empty_mismatch_and_oversized_are_rejected(self):
        for raw in (b'', b'x', packed(0x7f800000), packed(0xff800000),
                    packed(0x7fc00001), b'\0' * (1024 * 1024 + 4)):
            with self.assertRaises(NumericInvalid):
                decode_f32_le(raw)
        with self.assertRaises(NumericInvalid):
            compare_f32_le(packed(0), packed(0, 0))

    def test_fixed_tolerance_is_distinct_from_raw_bit_identity(self):
        exact = compare_f32_le(packed(0), packed(0x80000000))
        self.assertFalse(exact['bit_equal'])
        self.assertTrue(exact['within_tolerance'])
        tiny = compare_f32_le(packed(0), packed(1))
        self.assertEqual(tiny['max_abs_difference_binary64'], math.ldexp(1.0, -149))
        self.assertTrue(tiny['within_tolerance'])
        self.assertTrue(compare_f32_le(packed(0x3f800000), packed(0x3f800040))['within_tolerance'])
        self.assertFalse(compare_f32_le(packed(0x3f800000), packed(0x3f800200))['within_tolerance'])

    def test_decode_under_both_denormal_modes_in_fresh_process(self):
        code = '''
import math, sys, torch
from looking_glass.agreement_numeric import decode_f32_le
for mode in (False, True):
    if not torch.set_flush_denormal(mode): sys.exit(77)
    values = decode_f32_le(bytes.fromhex('0100000001000080'))
    assert values[0] == math.ldexp(1.0, -149)
    assert values[1] == -math.ldexp(1.0, -149)
'''
        with tempfile.TemporaryDirectory() as prefix:
            child = subprocess.run([sys.executable, '-B', '-X', f'pycache_prefix={prefix}', '-c', code],
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'),
                                   capture_output=True, text=True, timeout=30)
            self.assertFalse(os.listdir(prefix))
        if child.returncode == 77:
            self.skipTest('this CPU build cannot set both denormal modes')
        self.assertEqual(child.returncode, 0, child.stderr)


if __name__ == '__main__':
    unittest.main()
