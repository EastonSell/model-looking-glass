"""Fabricated regressions for the predeclared masked-color decisions."""
import math
import os
import subprocess
import sys
import unittest

from looking_glass import distilbert_color_edit_metrics_v1 as m


class MetricsTests(unittest.TestCase):
    def test_unique_ties_and_accepted_nll(self):
        info = m.describe([0.0, 2.0, 2.0, -1.0], [1, 3], (1, 3))
        self.assertEqual(info['top_ids'], [1, 2])
        self.assertIsNone(info['unique_top'])
        self.assertEqual(info['top_representative'], 1)
        self.assertEqual(info['top_gap'], 0.0)
        self.assertEqual(info['ranks'], {'1': 1, '3': 4})
        expected = math.log(sum(math.exp(x-2) for x in [0, 2, 2, -1])) - math.log(1 + math.exp(-3))
        self.assertAlmostEqual(info['accepted_nll'], expected, places=14)
        with self.assertRaises(ValueError):
            m.describe([0.0, 1.0], [], (1, 1))

    def test_distribution_metrics_include_tiny_nonzero_kl_terms(self):
        original = [0.0, -750.0]
        edited = [0.0, -1e38]
        result = m.distances(original, edited)
        self.assertGreater(result['kl_raw'], 0)
        self.assertAlmostEqual(result['kl_raw'] / math.exp(-750 + math.log(1e38)), 1.0, places=12)
        self.assertEqual(m.distances([0.0, 1.0], [0.0, 1.0])['kl'], 0.0)
        with self.assertRaises(ValueError):
            m.distances([0.0, float('nan')], [0.0, 1.0])
        with self.assertRaises(ValueError):
            m.distances([0.0, '1'], [0.0, 1.0])
        with self.assertRaises(ValueError):
            m.distances([0.0, True], [0.0, 1.0])

    def test_normalizer_retains_compensating_mass_below_one_ulp(self):
        result = m.distances([0.0, -100.0], [0.0, -80.0])
        expected_tv = math.exp(-80.0) - math.exp(-100.0)
        expected_kl = expected_tv - 20.0 * math.exp(-100.0)
        self.assertAlmostEqual(result['tv'] / expected_tv, 1.0, places=12)
        self.assertAlmostEqual(result['kl'] / expected_kl, 1.0, places=12)
        tied = m.distances([0.0, 0.0, -100.0], [0.0, 0.0, -80.0])
        self.assertAlmostEqual(tied['tv'] / (expected_tv / 2.0), 1.0, places=12)

    def test_denormal_flush_is_rejected_in_a_separate_process(self):
        source = '''import torch
from looking_glass import distilbert_color_edit_metrics_v1 as m
if not torch.set_flush_denormal(True):
    print('unsupported')
else:
    try:
        m.distances([0.0, -743.0], [0.0, -745.0])
    except RuntimeError:
        print('rejected')
    else:
        raise AssertionError('flush mode was accepted')
    torch.set_flush_denormal(False)
    assert m.distances([0.0, -743.0], [0.0, -745.0])['tv'] > 0
assert not torch.cuda.is_initialized()
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
        result = subprocess.run([sys.executable, '-B', '-c', source], env=env,
                                capture_output=True, text=True, check=True)
        self.assertIn(result.stdout.strip(), ('rejected', 'unsupported'))

    def test_other_partition_solve_and_float64_endpoints(self):
        original, primary = [0.0, 0.0, -1000.0], [2.0, 1.0, -999.0]
        solved = m.solve_bias(original, primary, 0, 1)
        edited = [original[0] + solved[0], original[1] + solved[1], original[2]]
        self.assertAlmostEqual(edited[0] - edited[1], primary[0] - primary[1], places=12)
        self.assertAlmostEqual(m.log_probabilities(edited)[0], m.log_probabilities(primary)[0], places=12)

    def test_match_requires_positive_closeness_and_the_exact_unique_top(self):
        row = {'margin_gain': 1.0, 'logp_gain': 0.2, 'unique_top': 7}
        self.assertTrue(m.row_matches(row, {'margin_gain': 1.05, 'logp_gain': 0.21, 'unique_top': 7}))
        self.assertFalse(m.row_matches(row, {'margin_gain': 1.2, 'logp_gain': 0.21, 'unique_top': 7}))
        self.assertFalse(m.row_matches(row, {'margin_gain': 1.0, 'logp_gain': 0.2, 'unique_top': 8}))
        self.assertFalse(m.row_matches(row, {'margin_gain': 1.0, 'logp_gain': 0.2, 'unique_top': None}))
        self.assertFalse(m.row_matches(row, {'margin_gain': 1.0, 'logp_gain': 0.01, 'unique_top': 7}))

    def test_locality_prediction_keeps_failures_and_ties(self):
        rows = [{'original_top': 1, 'primary_top': 1, 'bias_top': 2,
                 'accepted_ids': [1], 'primary_kl': 0.01, 'bias_kl': 0.03,
                 'primary_nll_change': 0.01} for _ in range(12)]
        self.assertEqual(m.locality_decision(rows, match_available=True)['status'], 'pass')
        self.assertEqual(m.locality_decision(rows, match_available=False)['status'], 'match-unavailable')
        rows[0]['primary_top'] = None
        self.assertEqual(m.locality_decision(rows, match_available=True)['status'], 'not-evaluable')
        rows[0]['primary_top'] = 1
        rows[0]['primary_kl'] = 0.5
        self.assertEqual(m.locality_decision(rows, match_available=True)['status'], 'fail')
        for row in rows:
            row['accepted_ids'] = [3]
        self.assertEqual(m.locality_decision(rows, match_available=True)['status'], 'not-evaluable')


if __name__ == '__main__':
    unittest.main()
