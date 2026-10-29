import pathlib
import sys
import unittest
import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src" if (_root / "src" / "looking_glass").is_dir() else _root))
from looking_glass import targeted_edit_france_top_v2 as subject

def fixture():
    rows = np.zeros((27, 6), dtype=np.float32)
    rows[:, 0] = 2.0
    rows[:7, 1] = 4.0  # three dev plus four baseline-supported targets
    labels = []
    for i in range(16):
        labels.append({"case_id": f"L{i+1:02d}", "accepted_ids": [0]})
        if i >= 8:
            rows[11+i, 3] = 3.0
    return rows, labels

class ScreenTests(unittest.TestCase):
    def test_fixed_screen_counts_supported_targets_and_fresh_locality(self):
        rows, labels = fixture()
        result = subject.screen(rows, 1, 2, labels)
        self.assertEqual(result["status"], "eligible")
        self.assertEqual((result["E"], result["H"], result["H_new"]), (4, 8, 5))
        self.assertEqual(len(result["targets"]), 8)
        self.assertEqual(len(result["locality"]), 16)

    def test_ties_and_overlapping_underfill_are_reported_without_selecting_replacements(self):
        rows, labels = fixture()
        rows[3, 0] = 4.0
        rows[11, 3] = 2.0
        rows[0, 1] = 2.0
        result = subject.screen(rows, 1, 2, labels)
        self.assertEqual(result["status"], "ineligible")
        self.assertEqual((result["E"], result["H"], result["H_new"]), (3, 7, 5))
        self.assertEqual(result["targets"][0]["top_tie_count"], 2)
        self.assertIsNone(result["targets"][0]["unique_top_id"])
        self.assertEqual(result["reasons"], ["development baseline", "target baseline count", "locality baseline count"])

    def test_bad_ids_and_nonfinite_rows_are_errors_not_underfill(self):
        rows, labels = fixture()
        labels[0]["accepted_ids"] = [True]
        with self.assertRaises(ValueError):
            subject.screen(rows, 1, 2, labels)
        rows, labels = fixture()
        rows[26, 3] = np.nan
        with self.assertRaises(ValueError):
            subject.screen(rows, 1, 2, labels)

    def test_accepted_sets_are_deduplicated_before_probability_aggregation(self):
        rows, labels = fixture()
        labels[0]["accepted_ids"] = [0, 0]
        with self.assertRaises(ValueError):
            subject.screen(rows, 1, 2, labels)

class StableDifferenceTests(unittest.TestCase):
    def test_accepted_nll_change_is_not_lost_at_a_huge_unrelated_maximum(self):
        base = np.array([2**60, 0.0, -100.0], dtype=np.float32)
        edit = np.array([2**60, -1.0, -100.0], dtype=np.float32)
        result = subject.distribution_differences(base, edit, [1], 1, 2)
        self.assertEqual(result["accepted_nll_change"], 1.0)
        self.assertEqual(result["margin_gain"], 1.0)
        self.assertEqual(result["alternative_logp_gain"], 0.0)
        self.assertEqual(result["kl"], 0.0)

    def test_binary64_flush_mode_is_rejected_and_not_reset_by_the_scorer(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable for FTZ-mode test")
        row = np.array([0.0, 1.0, -2.0], dtype=np.float32)
        self.assertTrue(torch.set_flush_denormal(True))
        try:
            with self.assertRaises(ValueError):
                subject.distribution_differences(row, row, [0], 1, 2)
        finally:
            self.assertTrue(torch.set_flush_denormal(False))

    def test_multi_token_accepted_set_uses_combined_stable_difference(self):
        import math
        base = np.array([2**60, 0.0, 0.0], dtype=np.float32)
        edit = np.array([2**60, -1.0, 0.0], dtype=np.float32)
        result = subject.distribution_differences(base, edit, [1, 2], 1, 2)
        expected = math.log(2.0) - math.log(1.0 + math.exp(-1.0))
        self.assertAlmostEqual(result["accepted_nll_change"], expected, places=15)

    def test_nonfinite_and_boolean_token_ids_are_invalid(self):
        row = np.array([0.0, 1.0, -2.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            subject.distribution_differences(row, row, [True], 1, 2)
        edited = row.copy()
        edited[2] = float("nan")
        with self.assertRaises(ValueError):
            subject.distribution_differences(row, edited, [0], 1, 2)

    def test_alternative_logp_gain_retains_one_bit_of_change_at_huge_max(self):
        base = np.array([2**60, -100.0, 0.0], dtype=np.float32)
        edit = np.array([2**60, -100.0, -1.0], dtype=np.float32)
        result = subject.distribution_differences(base, edit, [1], 1, 2)
        self.assertEqual(result["alternative_logp_gain"], -1.0)
        self.assertEqual(result["margin_gain"], -1.0)

class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.labels = fixture()
        original = self.rows[3:].copy()
        self.matrices = {k: original.copy() for k in
                         ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")}

    def test_baseline_gated_target_and_locality_can_pass_with_no_efficacy_match(self):
        self.matrices["projected"][:3, 2] = 5.0
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=False,
                                  execution_complete=True)
        self.assertEqual(result["primary"]["status"], "positive")
        self.assertEqual(result["primary"]["K"], 3)
        self.assertEqual(result["matched"]["status"], "ineligible")
        self.assertEqual(len(result["conditions"]["projected"]["targets"]), 8)
        self.assertEqual(len(result["conditions"]["projected"]["locality"]), 16)

    def test_baseline_unsupported_target_changes_do_not_count_as_supported_success(self):
        self.matrices["projected"][:2, 2] = 5.0
        self.matrices["projected"][4:8, 2] = 5.0
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=False,
                                  execution_complete=True)
        self.assertEqual(result["primary"]["status"], "negative")
        self.assertEqual(result["primary"]["K"], 2)
        self.assertEqual(len(result["conditions"]["projected"]["targets"]), 8)

    def test_three_quarters_boundary_is_rounded_up_for_every_supported_count(self):
        for eligible, threshold in ((4,3),(5,4),(6,5),(7,6),(8,6)):
            rows, labels = fixture()
            rows[3:3+eligible, 1] = 4.0
            base = rows[3:].copy()
            matrices = {k: base.copy() for k in
                        ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")}
            matrices["projected"][:threshold, 2] = 5.0
            result = subject.evaluate(rows, matrices, labels, 1, 2,
                                      development_top_achieved=True, match_available=False,
                                      execution_complete=True)
            self.assertEqual((result["baseline"]["E"], result["primary"]["threshold"]), (eligible, threshold))
            self.assertEqual(result["primary"]["status"], "positive")
            matrices["projected"][threshold-1, 2] = 0.0
            result = subject.evaluate(rows, matrices, labels, 1, 2,
                                      development_top_achieved=True, match_available=False,
                                      execution_complete=True)
            self.assertEqual(result["primary"]["status"], "negative")

    def test_a_tied_random_control_top_count_cannot_satisfy_the_strict_advantage(self):
        self.matrices["projected"][:3, 2] = 5.0
        self.matrices["projected_random"][:3, 2] = 5.0
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=False,
                                  execution_complete=True)
        self.assertEqual(result["primary"]["status"], "negative")
        self.assertFalse(result["primary"]["criteria"]["more_than_random"])

    def test_exact_matched_arm_is_balanced_without_a_false_locality_advantage(self):
        self.matrices["projected"][:4, 2] = 5.0
        self.matrices["matched"] = self.matrices["projected"].copy()
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=True,
                                  execution_complete=True)
        self.assertEqual(result["matched"]["status"], "measured")
        self.assertFalse(result["matched"]["projected_locality_advantage"])

    def test_an_unsupported_locality_still_contributes_to_kl_and_nll_limits(self):
        self.matrices["projected"][:3, 2] = 5.0
        self.matrices["projected"][-1, 3] = 30.0
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=False,
                                  execution_complete=True)
        self.assertEqual(result["primary"]["status"], "negative")
        self.assertFalse(result["conditions"]["projected"]["locality"][-1]["baseline_supported"])
        self.assertFalse(result["primary"]["criteria"]["max_kl"])
        self.assertFalse(result["primary"]["criteria"]["max_accepted_nll_change"])

    def test_changed_top_within_the_accepted_set_is_not_a_locality_failure(self):
        self.labels[0]["accepted_ids"] = [0, 3]
        before = np.array([0.40, 0.10, 0.05, 0.39, 0.03, 0.03], dtype=np.float32)
        after = np.array([0.39, 0.10, 0.05, 0.40, 0.03, 0.03], dtype=np.float32)
        self.rows[11] = np.log(before)
        for name in self.matrices:
            self.matrices[name][8] = self.rows[11]
        self.matrices["projected"][8] = np.log(after)
        self.matrices["projected"][:3, 2] = 5.0
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=False,
                                  execution_complete=True)
        row = result["conditions"]["projected"]["locality"][0]
        self.assertNotEqual(row["baseline_top"]["unique_top_id"], row["condition_top"]["unique_top_id"])
        self.assertTrue(row["condition_accepted"])
        self.assertAlmostEqual(row["accepted_nll_change"], 0.0, places=6)
        self.assertEqual(result["primary"]["status"], "positive")

    def test_reused_original_must_match_screen_bits_including_signed_zero(self):
        self.matrices["original"][0, 4] = -0.0
        with self.assertRaises(ValueError):
            subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                             development_top_achieved=True, match_available=False,
                             execution_complete=True)

    def test_unmatched_evaluation_effects_make_the_secondary_comparison_ineligible(self):
        self.matrices["projected"][:3, 2] = 5.0
        self.matrices["matched"] = self.matrices["unprojected"].copy()
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=True,
                                  execution_complete=True)
        self.assertEqual(result["primary"]["status"], "positive")
        self.assertEqual(result["matched"]["status"], "ineligible")
        self.assertIn("evaluated strength imbalance", result["matched"]["reasons"])

    def test_nonzero_or_unknown_exit_overrides_a_numerically_positive_result(self):
        self.matrices["projected"][:3, 2] = 5.0
        result = subject.evaluate(self.rows, self.matrices, self.labels, 1, 2,
                                  development_top_achieved=True, match_available=False,
                                  execution_complete=False)
        self.assertEqual(result["primary"]["status"], "invalid/incomplete")

if __name__ == "__main__":
    unittest.main()
