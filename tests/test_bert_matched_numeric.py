"""No-model tests for matched-dose arithmetic and exact float32 bits."""
import math
import random
import struct
import unittest

from looking_glass import bert_matched_numeric as numeric


def vector(first):
    return struct.pack('<768f', first, *([0.0] * 767))


class MatchedNumericTest(unittest.TestCase):
    def test_rounding_boundaries_and_signed_zero(self):
        values = ((0.0, 0), (-0.0, 0x80000000),
                  (1.0 + 2**-24, 0x3f800000), (1.0 + 3 * 2**-24, 0x3f800002),
                  (2**-150, 0), (3 * 2**-150, 2), (-2**-150, 0x80000000),
                  (2**-149, 1), (2**-126, 0x00800000))
        for value, bits in values:
            self.assertEqual(numeric.f64_to_f32_bits(float(value)), bits)
        for value in (float('inf'), float('-inf'), float('nan')):
            with self.assertRaises(ValueError):
                numeric.f64_to_f32_bits(value)

    def test_random_finite_values_match_native_nearest_even(self):
        numeric.binary64_rounding_probe()
        randomizer = random.Random(20261029)
        checked = 0
        for _ in range(10000):
            value = struct.unpack('<d', struct.pack('<Q', randomizer.getrandbits(64)))[0]
            if not math.isfinite(value):
                continue
            try:
                expected = struct.unpack('<I', struct.pack('<f', value))[0]
            except OverflowError:
                expected = 0xff800000 if math.copysign(1.0, value) < 0 else 0x7f800000
            self.assertEqual(numeric.f64_to_f32_bits(value), expected)
            checked += 1
        self.assertGreater(checked, 9900)

    def test_six_directions_and_artificial_controls_share_radius(self):
        bases = {site: vector(0.0) for site in numeric.SITES}
        donors = {'block0': {'A_opposite': vector(1), 'B_same': vector(2), 'B_opposite': vector(3)},
                  'block2': {'A_opposite': vector(4), 'B_same': vector(5), 'B_opposite': vector(6)}}
        first = numeric.plan_case('synthetic-row', bases, donors)
        second = numeric.plan_case('synthetic-row', bases, donors)
        self.assertEqual(first, second)
        self.assertEqual(first['radius'], 1.0)
        for site in numeric.SITES:
            for cell in (*numeric.DONORS, 'artificial'):
                action = first['sites'][site][cell]
                self.assertLessEqual(abs(action['realized_norm'] - 1.0), 1e-5)
                self.assertIsNone(action['realized_over_base_norm'])
            for cell in numeric.DONORS:
                self.assertLessEqual(first['sites'][site][cell]['fraction'], 1.0)
        self.assertNotEqual(first['sites']['block0']['artificial']['value'],
                            first['sites']['block2']['artificial']['value'])

    def test_tiny_and_nonfinite_inputs_fail_closed(self):
        bases = {site: vector(0.0) for site in numeric.SITES}
        donors = {site: {cell: vector(1.0) for cell in numeric.DONORS} for site in numeric.SITES}
        donors['block2']['B_same'] = vector(0.0)
        with self.assertRaises(ValueError):
            numeric.plan_case('synthetic-row', bases, donors)
        donors['block2']['B_same'] = vector(float('nan'))
        with self.assertRaises(ValueError):
            numeric.plan_case('synthetic-row', bases, donors)


if __name__ == '__main__':
    unittest.main()
