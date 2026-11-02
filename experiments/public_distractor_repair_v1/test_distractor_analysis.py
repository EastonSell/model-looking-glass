"""Fixed-rule tests using fabricated numbers only."""
import math
import unittest
try:
    import distractor_analysis as analysis
except ModuleNotFoundError:
    analysis = None

class RuleTests(unittest.TestCase):
    def test_selector_uses_only_numeric_effects_and_exposes_overshoot(self):
        self.assertIsNotNone(analysis, 'numeric selection rules are not implemented')
        rows = [(0.1, -0.4, 0.2, 0.0)] * 8
        result = analysis.select_candidate(rows)
        self.assertEqual(result['selected'], 'Q2')
        self.assertEqual(result['ranking'], ['Q2', 'Q3', 'Q1', 'Q4'])
        self.assertEqual(result['signed_predictions'], [0.1, -0.4, 0.2, 0.0])
        tied = analysis.select_candidate([(0.0, 0.0, 0.0, 0.0)] * 8)
        self.assertIsNone(tied['selected'])
        self.assertRaises(ValueError, analysis.select_candidate, [(math.nan, 0, 0, 0)] * 8)

    def test_primary_strict_boundary_and_undefined_cell(self):
        self.assertIsNotNone(analysis, 'numeric selection rules are not implemented')
        data = [([4.0] * 8, [2.0] * 8), ([4.0] * 8, [2.0] * 8),
                ([4.0] * 8, [3.0] * 8)]
        result = analysis.primary_rule(data)
        self.assertTrue(result['supported'])
        self.assertEqual(result['cells'][2]['reduction'], 0.25)
        self.assertFalse(result['cells'][2]['passes'])
        data[1] = ([4.0] * 8, [3.0] * 8)
        self.assertFalse(analysis.primary_rule(data)['supported'])
        data[2] = ([0.0] * 8, [0.0] * 8)
        self.assertIsNone(analysis.primary_rule(data)['supported'])

    def test_prediction_ties_do_not_count_as_wins(self):
        self.assertIsNotNone(analysis, 'numeric selection rules are not implemented')
        result = analysis.prediction_error(0.0, [0.0] * 8)
        self.assertEqual(result['mae'], 0.0)
        self.assertEqual(result['zero_mae'], 0.0)
        self.assertFalse(result['improved'])
        self.assertRaises(ValueError, analysis.prediction_error, math.inf, [1.0] * 8)

if __name__ == '__main__':
    unittest.main()
