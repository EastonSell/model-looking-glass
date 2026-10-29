import importlib
import importlib.util
import math
import unittest


class CausalMetricTests(unittest.TestCase):
    def test_candidate_compares_against_matched_nonzero_baseline(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        self.assertTrue(hasattr(metrics, 'compare_effects'), 'paired baseline comparator is missing')
        candidate = [metrics.EffectCase('a', 2, 3), metrics.EffectCase('b', 1, -1),
                     metrics.EffectCase('c', 0, 0)]
        baseline = [metrics.EffectCase('c', 0, 0), metrics.EffectCase('a', 0, 3),
                    metrics.EffectCase('b', 0, -1)]
        result = metrics.compare_effects(candidate, baseline, bootstrap_samples=0)
        point = result['point']
        self.assertEqual(result['directional_cases'], {'included': 2, 'excluded': 1})
        self.assertEqual(point['candidate_mae'], 1.0)
        self.assertAlmostEqual(point['baseline_mae'], 4 / 3)
        self.assertAlmostEqual(point['mae_improvement'], 1 / 3)
        self.assertAlmostEqual(point['candidate_rmse'], math.sqrt(5 / 3))
        self.assertAlmostEqual(point['baseline_rmse'], math.sqrt(10 / 3))
        self.assertAlmostEqual(point['rmse_improvement'], math.sqrt(10 / 3) - math.sqrt(5 / 3))
        self.assertEqual(point['candidate_sign_accuracy'], 0.5)
        self.assertEqual(point['baseline_sign_accuracy'], 0.0)
        self.assertEqual(point['sign_accuracy_improvement'], 0.5)
        self.assertEqual(result['case_errors'][1], {'case_id': 'b', 'candidate_absolute_error': 2.0,
                                                     'baseline_absolute_error': 1.0})
        self.assertTrue(all(ci is None for ci in result['bootstrap']['intervals'].values()))

    def test_matched_baseline_bootstrap_preserves_pairs(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        candidate = [metrics.EffectCase('a', 1, 0), metrics.EffectCase('b', 3, 0)]
        baseline = [metrics.EffectCase('b', 4, 0), metrics.EffectCase('a', 2, 0)]
        result = metrics.compare_effects(candidate, baseline, bootstrap_samples=8, confidence=0.5, seed=2)
        self.assertEqual(result['bootstrap']['intervals']['mae_improvement'], {'low': 1.0, 'high': 1.0})
        self.assertIsNone(result['point']['sign_accuracy_improvement'])
        self.assertEqual(result['bootstrap']['undefined_samples']['sign_accuracy_improvement'], 8)
        self.assertEqual(result, metrics.compare_effects(list(reversed(candidate)), list(reversed(baseline)),
                                                          bootstrap_samples=8, confidence=0.5, seed=2))

    def test_comparator_defaults_to_descriptive_and_excludes_undefined_direction(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        candidate = [metrics.EffectCase('a', 0, 0), metrics.EffectCase('b', 0, -1)]
        baseline = [metrics.EffectCase('a', 1, 0), metrics.EffectCase('b', -1, -1)]
        result = metrics.compare_effects(candidate, baseline)
        self.assertEqual(result['bootstrap']['method'], 'none')
        self.assertEqual(result['directional_cases'], {'included': 1, 'excluded': 1})
        self.assertEqual(result['point']['candidate_sign_accuracy'], 0.0)
        self.assertEqual(result['point']['baseline_sign_accuracy'], 1.0)
        self.assertEqual(result['point']['sign_accuracy_improvement'], -1.0)
        bootstrap = metrics.compare_effects(candidate, baseline, bootstrap_samples=8, seed=2)['bootstrap']
        self.assertIsNone(bootstrap['intervals']['sign_accuracy_improvement'])
        self.assertEqual(bootstrap['undefined_samples']['sign_accuracy_improvement'], 2)

    def test_comparator_preserves_small_paired_effect_from_original_inputs(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        candidate = [metrics.EffectCase('a', 1, 1e16)]
        baseline = [metrics.EffectCase('a', 0, 1e16)]
        point = metrics.compare_effects(candidate, baseline)['point']
        self.assertEqual(point['mae_improvement'], 1.0)
        self.assertEqual(point['rmse_improvement'], 1.0)

    def test_matched_baseline_rejects_missing_or_misaligned_cases(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        candidate = [metrics.EffectCase('a', 1, 2), metrics.EffectCase('b', 3, 4)]
        invalid = [[metrics.EffectCase('a', 1, 2)],
                   [metrics.EffectCase('a', 1, 2), metrics.EffectCase('c', 1, 4)],
                   [metrics.EffectCase('a', 1, 2), metrics.EffectCase('b', 1, 5)],
                   [metrics.EffectCase('a', 1, 2), metrics.EffectCase('b', None, 4)],
                   [metrics.EffectCase('a', 1, 2, 0, 3), metrics.EffectCase('b', 1, 4, 0, 5)]]
        for baseline in invalid:
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                metrics.compare_effects(candidate, baseline, bootstrap_samples=0)

    def test_matched_baseline_rejects_targets_that_only_match_after_float_rounding(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        candidate = [metrics.EffectCase('a', 0, 2**53)]
        baseline = [metrics.EffectCase('a', 0, 2**53 + 1)]
        with self.assertRaises(ValueError):
            metrics.compare_effects(candidate, baseline)

    def test_rejects_integral_values_not_exactly_representable_as_binary64(self):
        from fractions import Fraction
        metrics = importlib.import_module('looking_glass.causal_metrics')
        not_exact = 2**53 + 1
        with self.assertRaises(ValueError):
            metrics.compare_effects([metrics.EffectCase('a', not_exact, not_exact)],
                                    [metrics.EffectCase('a', 2**53, not_exact)])
        with self.assertRaises(ValueError):
            metrics.evaluate_effects([metrics.EffectCase('a', not_exact, 0)], bootstrap_samples=0)
        result = metrics.evaluate_effects([metrics.EffectCase('a', 2**53, 0)], bootstrap_samples=0)
        self.assertEqual(result['point']['mae'], 2**53)
        with self.assertRaises(ValueError):
            metrics.evaluate_effects([metrics.EffectCase('a', Fraction(1, 3), 0)], bootstrap_samples=0)
        result = metrics.evaluate_effects([metrics.EffectCase('a', Fraction(1, 2), 0)], bootstrap_samples=0)
        self.assertEqual(result['point']['mae'], 0.5)
    def test_point_estimates_against_zero_effect_baseline(self):
        spec = importlib.util.find_spec('looking_glass.causal_metrics')
        self.assertIsNotNone(spec, 'causal metrics module is missing')
        metrics = importlib.import_module('looking_glass.causal_metrics')
        cases = [
            metrics.EffectCase('a', 1.0, 2.0, 1.0, 2.0),
            metrics.EffectCase('b', -1.0, -2.0, -1.0, 1.0),
            metrics.EffectCase('c', 0.0, 0.0, 0.0, 0.0),
        ]
        result = metrics.evaluate_effects(cases, bootstrap_samples=8, seed=2,
                                          prediction_interval_level=0.8)
        point = result['point']
        self.assertEqual(result['n_cases'], 3)
        self.assertAlmostEqual(point['mae'], 2 / 3)
        self.assertAlmostEqual(point['rmse'], math.sqrt(2 / 3))
        self.assertEqual(point['sign_accuracy'], 1.0)
        self.assertAlmostEqual(point['null_mae'], 4 / 3)
        self.assertAlmostEqual(point['null_rmse'], math.sqrt(8 / 3))
        self.assertEqual(point['null_sign_accuracy'], 0.0)
        self.assertAlmostEqual(point['prediction_interval_coverage'], 2 / 3)
        self.assertAlmostEqual(point['prediction_interval_coverage_minus_nominal'], 2 / 3 - 0.8)
        self.assertEqual(result['prediction_interval_level'], 0.8)
        self.assertEqual(result['directional_cases'], {'included': 2, 'excluded': 1})
        self.assertAlmostEqual(point['mae_improvement_over_null'], 2 / 3)
        self.assertAlmostEqual(point['rmse_improvement_over_null'], math.sqrt(8 / 3) - math.sqrt(2 / 3))
        self.assertEqual(point['sign_accuracy_improvement_over_null'], 1.0)
        self.assertEqual([row['absolute_error'] for row in result['case_errors']], [1.0, 1.0, 0.0])

    def test_paired_case_bootstrap_recomputes_nonlinear_metrics(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        cases = [metrics.EffectCase('a', 1, 0), metrics.EffectCase('b', 3, 0)]
        result = metrics.evaluate_effects(cases, bootstrap_samples=8, confidence=0.5, seed=2)
        self.assertIn('bootstrap', result)
        ci = result['bootstrap']['intervals']
        # Python's seeded draws are aa, ab, ab, ba, aa, bb, bb, ba.
        self.assertAlmostEqual(ci['mae']['low'], 1.75)
        self.assertAlmostEqual(ci['mae']['high'], 2.25)
        self.assertAlmostEqual(ci['rmse']['low'], 1 + 0.75 * (math.sqrt(5) - 1))
        self.assertAlmostEqual(ci['rmse']['high'], math.sqrt(5) + 0.25 * (3 - math.sqrt(5)))
        self.assertAlmostEqual(ci['mae_improvement_over_null']['low'], -2.25)
        self.assertAlmostEqual(ci['mae_improvement_over_null']['high'], -1.75)
        self.assertIsNone(result['point']['prediction_interval_coverage'])
        self.assertIsNone(ci['prediction_interval_coverage'])
        self.assertEqual(result, metrics.evaluate_effects(list(reversed(cases)), bootstrap_samples=8,
                                                          confidence=0.5, seed=2))

    def test_zero_tolerance_is_symmetric_and_includes_the_boundary(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        cases = [metrics.EffectCase('a', 0.1, -0.1),
                 metrics.EffectCase('b', 0.101, -0.1),
                 metrics.EffectCase('c', -0.101, -0.101),
                 metrics.EffectCase('d', 0.1, 0.101)]
        exact = metrics.evaluate_effects(cases, bootstrap_samples=8)
        tolerant = metrics.evaluate_effects(cases, zero_tolerance=0.1, bootstrap_samples=8)
        self.assertAlmostEqual(exact['point']['sign_accuracy'], 1 / 2)
        self.assertAlmostEqual(tolerant['point']['sign_accuracy'], 1 / 2)
        self.assertEqual(tolerant['point']['null_sign_accuracy'], 0.0)
        self.assertEqual(tolerant['directional_cases'], {'included': 2, 'excluded': 2})
        self.assertEqual(tolerant['zero_tolerance'], 0.1)

    def test_sign_bootstrap_reports_undefined_draws(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        cases = [metrics.EffectCase('a', 0, 0), metrics.EffectCase('b', 1, 1)]
        result = metrics.evaluate_effects(cases, bootstrap_samples=8, confidence=0.5, seed=2)
        self.assertEqual(result['bootstrap']['defined_samples']['sign_accuracy'], 6)
        self.assertEqual(result['bootstrap']['undefined_samples']['sign_accuracy'], 2)
        self.assertIsNone(result['bootstrap']['intervals']['sign_accuracy'])
        result = metrics.evaluate_effects([metrics.EffectCase('a', 0, 0),
                                           metrics.EffectCase('b', 1, 0)], bootstrap_samples=8)
        self.assertIsNone(result['point']['sign_accuracy'])
        self.assertIsNone(result['point']['null_sign_accuracy'])
        self.assertIsNone(result['bootstrap']['intervals']['sign_accuracy'])
        self.assertEqual(result['bootstrap']['undefined_samples']['sign_accuracy'], 8)

    def test_descriptive_only_can_evaluate_one_fixed_case(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        result = metrics.evaluate_effects([metrics.EffectCase('a', 3, 2)], bootstrap_samples=0)
        self.assertEqual(result['point']['mae'], 1.0)
        self.assertEqual(result['point']['sign_accuracy'], 1.0)
        self.assertEqual(result['bootstrap']['samples'], 0)
        self.assertTrue(all(interval is None for interval in result['bootstrap']['intervals'].values()))
        with self.assertRaises(ValueError):
            metrics.evaluate_effects([metrics.EffectCase('a', 3, 2)], bootstrap_samples=1)

    def test_prediction_interval_level_must_match_provided_bounds(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        unbounded = [metrics.EffectCase('a', 0, 1), metrics.EffectCase('b', 0, 1)]
        bounded = [metrics.EffectCase('a', 0, 1, 0, 1), metrics.EffectCase('b', 0, 1, 0, 1)]
        for cases, level in [(unbounded, 0.8), (bounded, None), (bounded, 0),
                             (bounded, 1), (bounded, float('nan'))]:
            with self.subTest(cases=cases, level=level), self.assertRaises(ValueError):
                metrics.evaluate_effects(cases, prediction_interval_level=level, bootstrap_samples=8)

    def test_rejects_missing_nonfinite_or_ambiguous_cases(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        case = metrics.EffectCase
        good = case('a', 0, 1)
        bad_collections = [
            [], [good],
            [good, case('a', 0, 1)],
            [good, case('', 0, 1)],
            [good, case('  ', 0, 1)],
            [good, case('b', None, 1)],
            [good, case('b', 1, None)],
            [good, case('b', float('nan'), 1)],
            [good, case('b', 1, float('inf'))],
            [good, case('b', True, 1)],
            [good, case('b', '1', 1)],
            [good, case('b', 1, 1, 0, None)],
            [good, case('b', 1, 1, 0, float('inf'))],
            [good, case('b', 1, 1, 2, 1)],
            [good, case('b', 1, 1, 0, 2)],  # Not all cases have bounds.
        ]
        for cases in bad_collections:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                metrics.evaluate_effects(cases, bootstrap_samples=8)

    def test_rejects_invalid_bootstrap_settings(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        cases = [metrics.EffectCase('a', 0, 1), metrics.EffectCase('b', 1, 2)]
        for kwargs in [{'bootstrap_samples': -1}, {'bootstrap_samples': 1}, {'bootstrap_samples': 1.5},
                       {'bootstrap_samples': True}, {'confidence': 0}, {'confidence': 1},
                       {'confidence': float('nan')}, {'seed': True}, {'seed': 1.5},
                       {'zero_tolerance': -1}, {'zero_tolerance': float('nan')},
                       {'zero_tolerance': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                metrics.evaluate_effects(cases, **kwargs)

    def test_large_finite_values_do_not_silently_overflow(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        cases = [metrics.EffectCase('a', 1e200, 0), metrics.EffectCase('b', 1e200, 0)]
        result = metrics.evaluate_effects(cases, bootstrap_samples=8)
        self.assertAlmostEqual(result['point']['rmse'] / 1e200, 1.0)
        with self.assertRaises(ValueError):
            metrics.evaluate_effects([metrics.EffectCase('a', 1e308, -1e308),
                                      metrics.EffectCase('b', 0, 0)], bootstrap_samples=8)

    def test_mean_handles_subnormal_and_maximum_finite_errors(self):
        import sys
        metrics = importlib.import_module('looking_glass.causal_metrics')
        for error in [5e-324, sys.float_info.max]:
            cases = [metrics.EffectCase(str(i), error, 0) for i in range(3)]
            with self.subTest(error=error):
                result = metrics.evaluate_effects(cases, bootstrap_samples=8)
                self.assertEqual(result['point']['mae'], error)
                self.assertEqual(result['bootstrap']['intervals']['mae'], {'low': error, 'high': error})

    def test_paired_improvements_preserve_representable_small_differences(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        observed = 1e6
        delta = math.ulp(observed)
        cases = [metrics.EffectCase('a', delta, observed), metrics.EffectCase('b', 0, observed)]
        point = metrics.evaluate_effects(cases, bootstrap_samples=0)['point']
        self.assertEqual(point['mae_improvement_over_null'], delta / 2)
        self.assertAlmostEqual(point['rmse_improvement_over_null'] / (delta / 2), 1.0, places=14)

    def test_paired_mean_preserves_tiny_remainder_amid_large_cancellation(self):
        import sys
        metrics = importlib.import_module('looking_glass.causal_metrics')
        maximum = sys.float_info.max
        cases = [metrics.EffectCase('a', maximum, maximum),
                 metrics.EffectCase('b', maximum, maximum),
                 metrics.EffectCase('c', maximum, 0),
                 metrics.EffectCase('d', maximum, 0),
                 metrics.EffectCase('e', 1e-100, 1e-100)]
        point = metrics.evaluate_effects(cases, bootstrap_samples=0)['point']
        self.assertEqual(point['mae_improvement_over_null'], 1e-100 / 5)

    def test_loss_improvements_preserve_effects_lost_by_rounded_subtraction(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        case = metrics.EffectCase('a', 1.0, 1e16)
        point = metrics.evaluate_effects([case], bootstrap_samples=0)['point']
        self.assertEqual(point['mae_improvement_over_null'], 1.0)
        self.assertEqual(point['rmse_improvement_over_null'], 1.0)

    def test_rmse_improvement_preserves_small_effect_after_extreme_cancellation(self):
        metrics = importlib.import_module('looking_glass.causal_metrics')
        cases = [metrics.EffectCase('a', 1e308, 1e308),
                 metrics.EffectCase('b', 1e308, 0),
                 metrics.EffectCase('c', 1e100, 1e100)]
        point = metrics.evaluate_effects(cases, bootstrap_samples=0)['point']
        self.assertAlmostEqual(point['rmse_improvement_over_null'] / 2.886751345948129e-109, 1.0, places=14)


if __name__ == '__main__':
    unittest.main()
