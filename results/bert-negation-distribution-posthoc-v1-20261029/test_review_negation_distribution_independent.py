"""Synthetic numeric checks for the independent negation saved-logit review."""
import importlib.util
import math
from pathlib import Path
import struct
import unittest

path = Path(__file__).with_name('review_negation_distribution_independent.py')
spec = importlib.util.spec_from_file_location('negation_audit', path)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def logits(values):
    result = [-1000.] * 2152
    for token, value in zip((2151, 2070, 0, 1), values):
        result[token] = math.log(value)
    return result


class IndependentArithmetic(unittest.TestCase):
    def test_non_target_redistribution(self):
        vocab = [str(i) for i in range(2152)]
        item, _, _ = review.measure(logits((.4, .3, .2, .1)), logits((.4, .3, .1, .2)), vocab)
        self.assertAlmostEqual(item['total_variation'], .1)
        self.assertAlmostEqual(item['other_conditional_total_variation'], 1 / 3)
        self.assertEqual(item['any_some_margin_change'], 0.)
        self.assertEqual(item['any_some_absolute_movement_share'], 0.)

    def test_uniform_rescale_of_other_tokens(self):
        vocab = [str(i) for i in range(2152)]
        item, _, _ = review.measure(logits((.4, .3, .2, .1)), logits((.2, .15, 13/30, 13/60)), vocab)
        self.assertAlmostEqual(item['total_variation'], .35)
        self.assertAlmostEqual(item['other_conditional_total_variation'], 0.)
        self.assertAlmostEqual(item['any_some_absolute_movement_share'], .5)

    def test_null_and_tiny_cosines(self):
        self.assertIsNone(review.vector_cosine((0., 0.), (.2, -.2)))
        self.assertAlmostEqual(review.vector_cosine((1e-200, 0.), (3e-200, 0.)), 1.)
        self.assertAlmostEqual(review.vector_cosine((.2, -.2), (-.3, .3)), -1.)

    def test_tie_rank_and_binary32_subnormal(self):
        self.assertEqual(review.rank((1., 2., 2., 0.), 2), 2)
        bits = struct.pack('<3I', 1, 0x80000001, 0x3f800000)
        self.assertEqual(review.words(bits, 3), (2 ** -149, -(2 ** -149), 1.))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            review.words(struct.pack('<I', 0x7f800000), 1)

    def test_typed_comparison_catches_false_none_and_control_sign(self):
        with self.assertRaises(ValueError):
            review.same({'x': None}, {'x': 0.}, 'test')
        with self.assertRaises(ValueError):
            review.same({'x': -.1}, {'x': .1}, 'test')

    def test_small_nonzero_values_and_integer_types(self):
        with self.assertRaises(ValueError):
            review.same({'probability': 1e-100}, {'probability': 1e-16}, 'test')
        with self.assertRaises(ValueError):
            review.same({'probability': 5e-324}, {'probability': 1e-323}, 'test')
        with self.assertRaises(ValueError):
            review.same({'count': 1}, {'count': 1.0}, 'test')
        review.same({'probability': 1e-100}, {'probability': 1e-100 * (1 + 1e-13)}, 'test')


if __name__ == '__main__':
    unittest.main()
