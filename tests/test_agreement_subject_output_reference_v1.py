"""Fabricated, model-free checks for the output-slope reference."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('output_reference', Path(__file__).resolve().parents[1] / 'tools' / 'build_agreement_subject_output_reference_v1.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceTests(unittest.TestCase):
    def test_fixed_order_mean_and_zero_interaction(self):
        doses = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        pairs = [((2.0*q, -q), (-2.0*q, q)) for q in doses]
        slope = reference.derive(pairs, doses)
        self.assertEqual(slope, (2.0, -1.0))
        # The same fixed additive vector shifts either clean margin equally.
        def odd(clean):
            plus = (clean[0] + 2*slope[0], clean[1] + 2*slope[1])
            minus = (clean[0] - 2*slope[0], clean[1] - 2*slope[1])
            return ((plus[0]-plus[1]) - (minus[0]-minus[1])) / 2
        self.assertEqual(odd((3.0, 0.0)) - odd((-2.0, 0.0)), 0.0)

    def test_bad_inputs_are_rejected(self):
        pairs = [((0.0,), (0.0,))] * 8
        for doses in ([1.0]*7, [1.0]*7 + [0.0], [1.0]*7 + [math.nan]):
            with self.assertRaises(ValueError):
                reference.derive(pairs, doses)
        with self.assertRaises(ValueError):
            reference.derive(pairs[:-1] + [((math.inf,), (0.0,))], [1.0]*8)

    def test_constant_bias_and_nonlinear_gain_are_distinct(self):
        # Same fixed additive output bias: zero pair interaction, even when
        # its 1.0 odd response misses a particular 0.5 reference by >0.20.
        odd_singular = odd_plural = 1.0
        self.assertEqual(odd_singular - odd_plural, 0.0)
        self.assertGreater(abs(odd_singular - 0.5), 0.20)

        # A linear representation m=3*n+2*t has a real n coordinate but
        # identical odd responses; a prompt-dependent gain can yield K != 0.
        def odd(n, gain):
            def margin(t):
                return 3*n + 2*t + gain*n*t
            return (margin(2.0)-margin(-2.0))/2.0
        self.assertEqual(odd(1, 0)-odd(-1, 0), 0.0)
        self.assertEqual(odd(1, 0.5)-odd(-1, 0.5), 2.0)


if __name__ == '__main__':
    unittest.main()
