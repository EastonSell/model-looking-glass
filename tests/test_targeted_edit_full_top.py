"""Fabricated-only tests of frozen top selection and effect matching."""

import unittest

import numpy as np

from looking_glass import targeted_edit_full_top as study
from looking_glass import targeted_edit_full_top_score as scoring


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.full((36, 5), -1.0, dtype=np.float32)
        self.rows[:, 1] = 5.0
        self.pairs = [(1, 2)] * 4

    def test_first_qualifying_fact_and_first_four_rows(self):
        self.rows[0, 3] = 5.0  # first fact's discovery is tied
        self.rows[10, 3] = 6.0  # second fact has a bad calibration row
        self.rows[21:23, 3] = 6.0  # third fact has exactly four eligible evaluations
        result = study.select_screen(self.rows, self.pairs)
        self.assertEqual(result["fact_index"], 2)
        self.assertEqual(result["discovery_offset"], 18)
        self.assertEqual(result["calibration_offsets"], [19, 20])
        self.assertEqual(result["evaluation_offsets"], [23, 24, 25, 26])
        tied = result["observations"][0]
        self.assertIsNone(tied["unique_top_id"])
        self.assertEqual((tied["top_tie_count"], tied["top_representative_id"], tied["top_gap"]), (2, 1, 0.0))
        self.assertFalse(tied["eligible"])

    def test_underfilled_screen_preserves_every_failure(self):
        for start in range(0, 36, 9):
            self.rows[start + 3:start + 6, 3] = 6.0
        result = study.select_screen(self.rows, self.pairs)
        self.assertEqual(result["status"], "ineligible")
        self.assertIsNone(result["fact_index"])
        self.assertEqual(result["evaluation_offsets"], [])
        self.assertEqual(len(result["observations"]), 36)
        self.assertEqual([row["evaluation_eligible_count"] for row in result["candidates"]], [3] * 4)

    def test_nonfinite_or_bad_identity_is_not_an_ineligible_screen(self):
        self.rows[3, 2] = np.inf
        with self.assertRaises(study.StudyError):
            study.select_screen(self.rows, self.pairs)
        self.rows[3, 2] = 0
        with self.assertRaises(study.StudyError):
            study.select_screen(self.rows, [(True, 2)] * 4)

    def test_signed_float32_subnormals_do_not_depend_on_flush_mode(self):
        raw = np.array([[0x00000001, 0x80000001, 0x00000000, 0x80000000]], dtype=np.uint32)
        rows = raw.view(np.float32)
        try:
            import torch
        except ImportError:
            torch = None
        outputs = []
        for flush in (False, True):
            if torch is not None:
                torch.set_flush_denormal(flush)
            outputs.append(study._rows32(rows, 1))
        if torch is not None:
            torch.set_flush_denormal(False)
        for decoded in outputs:
            self.assertEqual(decoded[0, 0], 2.0 ** -149)
            self.assertEqual(decoded[0, 1], -(2.0 ** -149))
            self.assertFalse(np.signbit(decoded[0, 2]))
            self.assertTrue(np.signbit(decoded[0, 3]))

    def test_shifted_log_softmax_handles_extreme_finite_ties(self):
        row = np.array([[1.0e20, 1.0e20, -1.0e20]], dtype=np.float32)
        decoded = study._rows32(row, 1)[0]
        self.assertAlmostEqual(study._logp(decoded, 0), -np.log(2.0), places=15)
        self.assertAlmostEqual(scoring._logsoftmax(decoded)[1], -np.log(2.0), places=15)


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.array([[4.0, -1.0, -2.0], [4.0, -1.0, -2.0]], dtype=np.float32)
        self.projected = np.array([[4.0, -0.3, -2.0], [4.0, -0.3, -2.0]], dtype=np.float32)

    def test_equal_grid_points_choose_lowest_and_are_reproducible(self):
        grid = [(dose, self.projected.copy()) for dose in study.UNPROJECTED_GRID]
        result = study.select_development_match(self.baseline, self.projected, grid, 0, 1)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["selected_dose"], study.UNPROJECTED_GRID[0])
        self.assertEqual(result["nearest_dose"], study.UNPROJECTED_GRID[0])
        self.assertEqual(len(result["grid"]), 13)
        self.assertTrue(all(row["eligible"] for row in result["grid"]))

    def test_same_counterfactual_probability_can_have_different_top(self):
        base = np.array([[1.0, 0.0, 0.0]] * 2, dtype=np.float32)
        a = np.array([[np.log(.75), 0.0, np.log(.75)]] * 2, dtype=np.float32)
        b = np.array([[np.log(1.2), 0.0, np.log(.3)]] * 2, dtype=np.float32)
        result = study.select_development_match(base, a, [(x, b) for x in study.UNPROJECTED_GRID], 0, 1)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["selected_dose"])
        self.assertEqual(result["nearest_dose"], study.UNPROJECTED_GRID[0])
        self.assertFalse(result["grid"][0]["top_categories_agree"])
        self.assertAlmostEqual(result["projected_effects"][0]["alternative_logp_gain"],
                               result["grid"][0]["effects"][0]["alternative_logp_gain"], places=7)

    def test_evaluation_balance_rejects_a_single_weaker_row(self):
        base = np.repeat(self.baseline, 2, axis=0)
        project = np.repeat(self.projected, 2, axis=0)
        self.assertTrue(study.evaluated_strength_balance(base, project, project, 0, 1)["balanced"])
        control = project.copy()
        control[2, 1] = -0.25
        result = study.evaluated_strength_balance(base, project, control, 0, 1)
        self.assertFalse(result["balanced"])
        self.assertEqual([item["passes"] for item in result["cases"]], [True, True, False, True])

    def test_partial_grid_and_invalid_baseline_fail_closed(self):
        with self.assertRaises(study.StudyError):
            study.select_development_match(self.baseline, self.projected, [], 0, 1)
        base = self.baseline.copy()
        base[1, 2] = base[1, 0]
        with self.assertRaises(study.StudyError):
            study.select_development_match(base, self.projected,
                                           [(x, self.projected) for x in study.UNPROJECTED_GRID], 0, 1)

    def test_dose_selection_and_explicit_no_top_fallback(self):
        rows = [np.array([[2.0, 0.0, -1.0]], dtype=np.float32) for _ in study.PROJECTED_GRID]
        rows[2][0, 1] = 3.0
        rows[4][0, 1] = 4.0
        grid = list(zip(study.PROJECTED_GRID, rows))
        self.assertEqual(study.select_projected_dose(grid, 0, 1)["selected_dose"], 0.002)
        rows[2][0, 1] = 2.0  # a top tie is not a unique counterfactual top
        rows[4][0, 1] = 0.0
        result = study.select_projected_dose(grid, 0, 1)
        self.assertFalse(result["development_top_achieved"])
        self.assertTrue(result["fallback"])
        self.assertEqual(result["selected_dose"], 0.01)


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        baseline = np.tile(np.array([[2.0, 0.0, -2.0, -2.0, -2.0, -2.0]], dtype=np.float32), (16, 1))
        baseline[4] = np.log(np.array([0.40, 0.10, 0.10, 0.30, 0.05, 0.05], dtype=np.float32))
        self.matrices = {name: baseline.copy() for name in scoring.CONDITIONS}
        self.locality = [{"case_id": f"locality-{i}", "stratum": str(i // 4),
                          "expected_id": 0, "accepted_ids": [0, 3]} for i in range(12)]

    def test_spelling_redistribution_uses_accepted_set_nll(self):
        project = self.matrices["projected"]
        project[:2, 1] = 3.0
        project[4] = np.log(np.array([0.29, 0.10, 0.10, 0.41, 0.05, 0.05], dtype=np.float32))
        result = scoring.evaluate(self.matrices, 0, 1, self.locality,
                                  development_top_achieved=True, match_available=False)
        p = result["conditions"]["projected"]
        locality = p["locality"][0]
        self.assertGreater(locality["expected_nll_change"], 0.20)
        self.assertAlmostEqual(locality["accepted_nll_change"], 0.0, places=6)
        self.assertNotEqual(locality["baseline_top"]["unique_top_id"], locality["condition_top"]["unique_top_id"])
        self.assertTrue(locality["condition_accepted"])
        self.assertTrue(result["equal_norm"]["criteria"]["max_accepted_nll_change_at_most_0_20"])
        self.assertTrue(result["equal_norm"]["criteria"]["baseline_accepted_preserved"])
        self.assertEqual(p["alternative_top_count"], 2)

    def test_overlapping_ineligibility_reasons_are_retained(self):
        for case in self.locality:
            case["expected_id"] = 4
            case["accepted_ids"] = [4]
        result = scoring.evaluate(self.matrices, 0, 1, self.locality,
                                  development_top_achieved=False, match_available=False)
        self.assertEqual(result["equal_norm"]["status"], "ineligible")
        self.assertEqual(result["matched"]["ineligibility_reasons"],
                         ["no baseline-accepted locality", "no development match"])
        self.assertIsNone(result["matched"]["criteria"])
        self.assertIsNone(result["matched"]["evaluated_balance"])

    def test_missing_or_nonfinite_condition_cannot_produce_a_decision(self):
        self.matrices["projected"][5, 3] = np.nan
        with self.assertRaises(study.StudyError):
            scoring.evaluate(self.matrices, 0, 1, self.locality,
                             development_top_achieved=True, match_available=False)


if __name__ == "__main__":
    unittest.main()
