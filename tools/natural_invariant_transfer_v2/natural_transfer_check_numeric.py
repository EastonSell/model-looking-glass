import math
import os
import struct
import subprocess
import sys
import unittest

import natural_transfer_numeric as n


def f32(*values):
    return b''.join(struct.pack('<f', x) for x in values)


class NaturalTransferNumericTests(unittest.TestCase):
    def test_four_fixed_candidates(self):
        b = f32(1, 2, 3, 4)
        plan = n.plan_candidates(b, f32(2, 2, 3, 4), f32(1, 4, 3, 4), f32(1, 2, -1, 4))
        self.assertEqual(plan['q'], 1.0)
        expected = {'Q1': f32(1, 3, 3, 4), 'Q2': f32(2, 2, 3, 4),
                    'Q3': f32(1, 3, 3, 4), 'Q4': f32(1, 2, 2, 4)}
        for key, result in plan['candidates'].items():
            self.assertEqual(result['replacement_f32'], expected[key])
            self.assertTrue(result['certificate']['pass'], key)
            self.assertEqual(len(result['displacement_f64']), 32)

    def test_nearest_even_midpoints_subnormals_and_sign(self):
        cases = [(1.0 + 2.0**-24, 0x3f800000),
                 (1.0 + 3.0 * 2.0**-24, 0x3f800002),
                 (-1.0 - 2.0**-24, 0xbf800000),
                 (2.0**-150, 0), (3.0 * 2.0**-150, 2), (-0.0, 0x80000000)]
        for value, bits in cases:
            self.assertEqual(n.f64_to_f32_bits(value), bits)

    def test_inclusive_error_and_norm_boundaries(self):
        for after in (4999.0, 5001.0):
            cert = n.certificate(5000.0, f32(0), f32(after), [5000.0])
            self.assertTrue(cert['pass'], cert)
        for after in (4998.99951171875, 5001.00048828125):
            cert = n.certificate(5000.0, f32(0), f32(after), [5000.0])
            self.assertFalse(cert['pass'], cert)

    def test_wrong_direction_cannot_hide_behind_equal_norm(self):
        cert = n.certificate(1.0, f32(0, 0), f32(0, 1), [1.0, 0.0])
        self.assertTrue(cert['checks']['intended_norm'])
        self.assertTrue(cert['checks']['realized_norm'])
        self.assertFalse(cert['checks']['displacement_error'])
        self.assertFalse(cert['pass'])

    def test_bad_or_undelivered_vectors_fail(self):
        with self.assertRaises(n.InvalidPlan):
            n.plan_candidates(f32(0, 0), f32(0, 0), f32(1, 0), f32(0, 1))
        with self.assertRaises(n.InvalidPlan):
            n.plan_candidates(f32(0, 0), f32(1, 0), f32(0, 0), f32(0, 1))
        with self.assertRaises(n.InvalidPlan):
            n.plan_candidates(f32(0), f32(float('nan')), f32(1), f32(1))
        with self.assertRaises(n.InvalidPlan):
            n.plan_candidates(f32(0), f32(1), f32(1, 2), f32(1))
        self.assertFalse(n.certificate(1.0, f32(1), f32(1), [1.0])['pass'])

    @unittest.skipUnless(os.name == 'nt', 'Windows control-word isolation')
    def test_all_non_nearest_modes_fail_in_isolated_processes(self):
        source = r'''
import ctypes, sys
import natural_transfer_numeric as n
c=ctypes.CDLL('msvcrt')._controlfp
c.argtypes=[ctypes.c_uint,ctypes.c_uint]; c.restype=ctypes.c_uint
old=c(0,0)
try:
    c(int(sys.argv[1]),0x300)
    try: n.ensure_round_to_nearest()
    except n.InvalidPlan: pass
    else: raise RuntimeError('non-nearest accepted')
finally: c(old,0x300)
'''
        for mode in (0x100, 0x200, 0x300):
            result = subprocess.run([sys.executable, '-B', '-c', source, str(mode)],
                                    capture_output=True, text=True, cwd=os.path.dirname(n.__file__))
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
