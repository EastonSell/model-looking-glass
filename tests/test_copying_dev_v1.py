"""Pure checks for the public copying screen; no pretrained forward."""
import math
import unittest

import numpy as np

from scripts.copying_dev_v1 import GROUPS, PERMUTATIONS, planned_cases, statistics


class CopyingDevelopmentTests(unittest.TestCase):
    def test_fixed_pairs_preserve_suffix_multiset_and_target_position(self):
        rows = planned_cases()
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({value for _, group in GROUPS for value in group}), 48)
        self.assertEqual(len(set(PERMUTATIONS)), 4)
        for index in range(8):
            cases = rows[index * 4:index * 4 + 4]
            baseline = cases[0]
            for row in cases:
                self.assertEqual(len(row["token_ids"]), 11)
                self.assertEqual(sorted(row["token_ids"]), sorted(baseline["token_ids"]))
                self.assertEqual(row["token_ids"][-5:], baseline["token_ids"][-5:])
                self.assertEqual(row["token_ids"][5], row["target_id"])
                self.assertEqual(row["token_ids"].count(row["target_id"]), 1)
            self.assertEqual(baseline["token_ids"][:5], baseline["token_ids"][-5:])
            self.assertTrue(all(row["token_ids"][:5] != row["token_ids"][-5:] for row in cases[1:]))

    def test_target_log_probability_and_strict_rank_ties(self):
        raw = np.array([1.0, 2.0, 2.0, -1.0], dtype="<f4").tobytes()
        result = statistics(raw, 2)
        self.assertEqual(result["target_rank"], 1)
        self.assertEqual(result["top_token_id"], 1)
        self.assertAlmostEqual(result["target_log_probability"],
                               2.0 - math.log(math.exp(1.0) + 2 * math.exp(2.0) + math.exp(-1.0)))
        self.assertEqual(statistics(raw, 0)["target_rank"], 3)


if __name__ == "__main__":
    unittest.main()
