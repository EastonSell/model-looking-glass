"""Pure checks for the public copying screen; no pretrained forward."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.copying_dev_v1 import GROUPS, PERMUTATIONS, numeric_policy, planned_cases, runtime_checks, statistics


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

    def test_runtime_checks_reject_changed_mode_and_autocast(self):
        model = torch.nn.Linear(2, 2).eval()
        model.config = SimpleNamespace(_attn_implementation="eager", use_cache=False)
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "-1"}):
            numeric_policy(initialize=True)
            runtime_checks(model)
            model.train()
            with self.assertRaises(RuntimeError):
                runtime_checks(model)
            model.eval()
            model.config.use_cache = True
            with self.assertRaises(RuntimeError):
                runtime_checks(model)
            model.config.use_cache = False
            with torch.autocast("cpu", dtype=torch.bfloat16):
                with self.assertRaises(RuntimeError):
                    runtime_checks(model)


if __name__ == "__main__":
    unittest.main()
