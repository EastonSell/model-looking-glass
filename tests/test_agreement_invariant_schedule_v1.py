"""Fabricated schedule and factorial counterexamples; no tokenizer or model."""

import collections
import math
from pathlib import Path
import types
import unittest

plan = types.ModuleType('agreement_invariant_schedule_v1')
path = Path(__file__).with_name('agreement_invariant_schedule_v1.py')
if not path.is_file():
    path = Path(__file__).resolve().parent.parent / 'looking_glass' / 'agreement_invariant_schedule_v1.py'
exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), plan.__dict__)


def contrast(a_this, a_these, b_this, b_these):
    ka, kb = a_this - a_these, b_this - b_these
    signed = (ka + kb) / 2
    d_num = (abs(ka) + abs(kb)) / 2
    d_lex = (abs(a_this - b_this) + abs(a_these - b_these)) / 2
    return signed, d_num, d_lex, d_num - d_lex


class ScheduleTests(unittest.TestCase):
    def test_complete_balanced_schedule(self):
        schedule = plan.cases()
        self.assertEqual(schedule, plan.cases())
        self.assertEqual(len(schedule), 68)
        self.assertEqual(len({name for name, _ in schedule}), 68)
        self.assertTrue(all(name.startswith('discovery_') for name, _ in schedule[:34]))
        self.assertTrue(all(name.startswith('evaluation_') for name, _ in schedule[34:]))
        self.assertEqual(len(plan.actions()), 612)
        slot = collections.Counter((arm, i) for _, words in schedule for i, arm in enumerate(words))
        self.assertEqual(len(slot), 36)
        self.assertEqual(set(slot.values()), {11, 12})
        for split in plan.SPLITS:
            selected = [(name, words) for name, words in schedule if name.startswith(split + '_')]
            self.assertEqual(sum(words.index('u_plus') < words.index('u_minus') for _, words in selected), 17)
            self.assertEqual(sum(words.index('v1_plus') < words.index('v1_minus') for _, words in selected), 17)
            self.assertEqual(sum(words.index('v2_plus') < words.index('v2_minus') for _, words in selected), 17)
            by_id = dict(selected)
            pairs = plan._pairs(split, plan.SPLITS.index(split))
            self.assertEqual(len(pairs), 14)
            for identity, _, _ in pairs:
                self.assertEqual(by_id[identity + '_this'], by_id[identity + '_these'])
            first = [next(name for name, _ in selected if name.startswith(identity + '_'))
                     for identity, _, _ in pairs]
            self.assertEqual(sum(name.endswith('_this') for name in first), 7)
            targets = [x for x in pairs if '_target_' in x[0]]
            self.assertEqual(sum(x[1] == 0 for x in targets), 6)
            for noun in range(2):
                subset = [x for x in targets if x[0].endswith(f'_n{noun}')]
                self.assertEqual(sum(x[1] == 0 for x in subset), 3)
                self.assertEqual(sum(x[2] == 0 for x in subset), 3)
            for frame in range(2):
                subset = [x for x in targets if f'_f{frame}_' in x[0]]
                self.assertEqual(sum(x[1] == 0 for x in subset), 3)
                self.assertEqual(sum(x[2] == 0 for x in subset), 3)

    def test_resource_calculation(self):
        self.assertEqual(612 * 50257 * 4 + 476 * (2 * 12 * 768 * 4 + 2 * 768 * 4), 161048208)

    def test_additive_constant_and_linear_number(self):
        # Identical response is compatible with a number-bearing clean state
        # followed by a linear readout: m = 3*n + 2*t gives O = 4 for both.
        self.assertEqual(contrast(4, 4, 4, 4), (0, 0, 0, 0))

    def test_determiner_gain_can_mimic_positive_number(self):
        # These values can arise from determiner-dependent, not grammatical,
        # gain. A positive criterion is therefore not an identifying proof.
        self.assertEqual(contrast(1, 0, 1, 0), (1, 1, 0, 1))

    def test_negative_and_zero_distances_are_retained(self):
        self.assertEqual(contrast(0, 0, 2, 2), (0, 0, 2, -2))
        self.assertEqual(contrast(1, -1, -1, 1), (0, 2, 2, 0))
        self.assertTrue(math.isfinite(contrast(0, 0, 0, 0)[-1]))


if __name__ == '__main__':
    unittest.main()
