"""Model-free checks for the proposed substitution schedule and contrasts."""
from collections import Counter
import math
import unittest


ORDERS = (
    ("As", "Ap", "Bs", "Bp"),
    ("Ap", "As", "Bp", "Bs"),
    ("Bs", "Bp", "As", "Ap"),
    ("Bp", "Bs", "Ap", "As"),
    ("As", "Bs", "Ap", "Bp"),
    ("Bp", "Ap", "Bs", "As"),
)
SIGNS = (True, False, True, False, False, True)
PLUS = ("u+", "u-", "v1+", "v1-", "v2+", "v2-")
MINUS = ("u-", "u+", "v1-", "v1+", "v2-", "v2+")


def contrasts(a_s, a_p, b_s, b_p):
    opposite = math.fsum((abs(a_s - b_p), abs(b_s - a_p))) / 2
    same = math.fsum((abs(a_s - b_s), abs(a_p - b_p))) / 2
    return {
        "opposite": opposite,
        "same": same,
        "delta": opposite - same,
        "number": math.fsum((a_s - a_p, b_s - b_p)) / 2,
        "lemma": math.fsum((a_s - b_s, a_p - b_p)) / 2,
        "interaction": (a_s - a_p) - (b_s - b_p),
    }


def separate_case_groups(groups):
    """Allow only the intended base/own-extension prefix relationship."""
    for base, extensions in groups:
        if (len(extensions) != len(set(extensions))
                or any(len(item) != len(base) + 1 or item[:-1] != base
                       for item in extensions)):
            return False
    for i, (base, extensions) in enumerate(groups):
        for other_base, other_extensions in groups[i + 1:]:
            for left in (base, *extensions):
                for right in (other_base, *other_extensions):
                    if left[:len(right)] == right or right[:len(left)] == left:
                        return False
    return True


class DesignTests(unittest.TestCase):
    def test_schedule_counts(self):
        counts = Counter()
        first = {0: Counter(), 1: Counter()}
        positions = {0: Counter(), 1: Counter()}
        signs = {0: Counter(), 1: Counter()}
        cases = 0
        for frame in range(2):
            for condition in range(3):
                n = 3 * frame + condition
                morphology_order = (0, 1) if (frame + condition) % 2 == 0 else (1, 0)
                for morphology in morphology_order:
                    is_plus = SIGNS[n] != bool(morphology)
                    signs[morphology][is_plus] += 1
                    first[morphology][ORDERS[n][0]] += 1
                    base = PLUS if is_plus else MINUS
                    arms = base[2 * condition:] + base[:2 * condition]
                    for pos, form in enumerate(ORDERS[n]):
                        positions[morphology][(form, pos)] += 1
                        counts.update((arm, slot) for slot, arm in enumerate(arms))
                        cases += 1
                base = PLUS if (frame + condition) % 2 == 0 else MINUS
                arms = base[2 * condition:] + base[:2 * condition]
                counts.update((arm, slot) for slot, arm in enumerate(arms))
                cases += 1
        self.assertEqual(cases, 54)
        self.assertEqual(cases * 9, 486)
        self.assertEqual(set(counts.values()), {9})
        self.assertEqual(len(counts), 36)
        for morphology in (0, 1):
            self.assertEqual(signs[morphology], {True: 3, False: 3})
            self.assertEqual(first[morphology]["As"] + first[morphology]["Bs"], 3)
            self.assertEqual(first[morphology]["Ap"] + first[morphology]["Bp"], 3)
            self.assertEqual(set(positions[morphology].values()), {1, 2})
            self.assertEqual(len(positions[morphology]), 16)

    def test_constant_output_shift_cancels(self):
        result = contrasts(1.4, 1.4, 1.4, 1.4)
        self.assertEqual(set(result.values()), {0.0})

    def test_lexical_offset_can_mask_a_number_effect(self):
        mu, d, b = 0.0, 3.0, 0.2
        result = contrasts(mu + d / 2 + b, mu + d / 2 - b,
                           mu - d / 2 + b, mu - d / 2 - b)
        self.assertAlmostEqual(result["same"], abs(d))
        self.assertAlmostEqual(result["opposite"], max(abs(d), 2 * abs(b)))
        self.assertAlmostEqual(result["delta"], 0)
        self.assertAlmostEqual(result["number"], 2 * b)

    def test_heterogeneous_surface_gain_can_be_positive(self):
        result = contrasts(1.0, 0.3, 0.8, 0.4)
        self.assertGreater(result["delta"], 0)
        self.assertGreater(result["number"], 0)
        self.assertNotEqual(result["interaction"], 0)

    def test_negative_distance_difference_is_retained(self):
        result = contrasts(1.0, -1.0, -1.0, 1.0)
        self.assertEqual(result["same"], 2.0)
        self.assertEqual(result["opposite"], 0.0)
        self.assertEqual(result["delta"], -2.0)

    def test_own_answer_prefix_is_allowed_but_cross_case_prefix_is_not(self):
        self.assertTrue(separate_case_groups([
            ((1, 2), ((1, 2, 3), (1, 2, 4))),
            ((5, 6), ((5, 6, 3), (5, 6, 4))),
        ]))
        self.assertFalse(separate_case_groups([
            ((1, 2), ((1, 2, 3),)),
            ((1, 2, 3, 8), ((1, 2, 3, 8, 4),)),
        ]))
        self.assertFalse(separate_case_groups([
            ((1, 2), ((1, 2, 3), (1, 2, 3))),
        ]))


if __name__ == "__main__":
    unittest.main()
