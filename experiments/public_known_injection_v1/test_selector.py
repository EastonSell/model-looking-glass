import math
import unittest

from selector import discovery_selection, fixed_random_site


class DiscoverySelectionTests(unittest.TestCase):
    def test_positive_reduction_ranking_and_predictions(self):
        rows = [(0.0, 4.0, (3.0, 1.0, 2.0, 4.0)) for _ in range(8)]
        out = discovery_selection(rows)
        self.assertEqual(out['selected_site'], 2)
        self.assertEqual(out['ranking'], [2, 3, 1, 4])
        self.assertEqual(out['scores'], [1.0, 3.0, 2.0, 0.0])
        self.assertEqual(out['signed_predictions'], [-1.0, -3.0, -2.0, 0.0])

    def test_ties_choose_lowest_and_nonpositive_abstains(self):
        tied = [(0.0, 4.0, (1.0, 1.0, 3.0, 4.0)) for _ in range(8)]
        out = discovery_selection(tied)
        self.assertEqual(out['selected_site'], 1)
        self.assertEqual(out['tied_max_sites'], [1, 2])
        worse = [(0.0, 1.0, (1.0, 2.0, 3.0, 4.0)) for _ in range(8)]
        self.assertIsNone(discovery_selection(worse)['selected_site'])

    def test_only_fixed_numerical_input_is_accepted(self):
        row = (0.0, 1.0, (0.0, 0.0, 0.0, 0.0))
        for bad in ([row] * 7, [row] * 8 + [row],
                    [(0.0, 1.0, (0.0,) * 4, 'truth')] * 8,
                    [{'truth': 1, 'row': row}] * 8,
                    [(0.0, math.nan, (0.0,) * 4)] * 8,
                    [(0.0, 1.0, (0.0,) * 3)] * 8):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                discovery_selection(bad)

    def test_random_reference_is_fixed_and_validated(self):
        self.assertEqual(fixed_random_site('A'), 2)
        self.assertEqual(fixed_random_site('B'), 4)
        with self.assertRaises(ValueError):
            fixed_random_site('C')


if __name__ == '__main__':
    unittest.main()
