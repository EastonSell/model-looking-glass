"""Model-free enumeration of the prospective paired execution schedule."""
from collections import Counter
import unittest

ARMS_PLUS = ('u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')
ARMS_MINUS = ('u_minus', 'u_plus', 'v1_minus', 'v1_plus', 'v2_minus', 'v2_plus')
FRAGMENT_OFFSETS = ((0, 0, 2, 2), (2, 2, 4, 4), (2, 2, 4, 4), (4, 4, 4, 4))


def schedule():
    result = []
    for frame in range(4):
        for condition in range(4):
            for pair in range(2):
                numbers = ('singular', 'plural') if (frame + frame // 2 + pair) % 2 == 0 else ('plural', 'singular')
                base = ARMS_PLUS if (frame + condition + pair) % 2 == 0 else ARMS_MINUS
                offset = 2 * ((4 * frame + condition) % 3)
                arms = base[offset:] + base[:offset]
                for number in numbers:
                    result.append((frame, condition, pair, number, arms))
            base = ARMS_PLUS if (frame + condition) % 2 == 0 else ARMS_MINUS
            offset = FRAGMENT_OFFSETS[frame][condition]
            result.append((frame, condition, None, 'context', base[offset:] + base[:offset]))
    return result


class ScheduleTests(unittest.TestCase):
    def test_complete_balanced_order(self):
        rows = schedule()
        self.assertEqual(len(rows), 80)
        self.assertEqual(sum(3 + len(row[4]) for row in rows), 720)
        slots = Counter((action, slot) for _, _, _, _, arms in rows for slot, action in enumerate(arms))
        self.assertEqual(len(slots), 36)
        self.assertEqual(set(slots.values()), {13, 14})
        for name in ('u', 'v1', 'v2'):
            plus, minus = name + '_plus', name + '_minus'
            for selector, expected in ((lambda row: row[3] == 'singular', (16, 16)),
                                       (lambda row: row[3] == 'plural', (16, 16)),
                                       (lambda row: row[2] == 0, (16, 16)),
                                       (lambda row: row[2] == 1, (16, 16)),
                                       (lambda row: row[3] == 'context', (8, 8))):
                sample = [row for row in rows if selector(row)]
                plus_first = sum(row[4].index(plus) < row[4].index(minus) for row in sample)
                self.assertEqual((plus_first, len(sample)-plus_first), expected)
        first_number = Counter()
        for frame in range(4):
            for condition in range(4):
                pair_rows = [row for row in rows if row[0] == frame and row[1] == condition and row[2] is not None]
                self.assertEqual(pair_rows[0][4], pair_rows[1][4])
                self.assertEqual(pair_rows[2][4], pair_rows[3][4])
                self.assertEqual({pair_rows[0][3], pair_rows[2][3]}, {'singular', 'plural'})
                first_number.update((pair_rows[0][3], pair_rows[2][3]))
        self.assertEqual(first_number, {'singular': 16, 'plural': 16})
        subject_keys = (('R1', 'I1'), ('R1', 'I2'), ('R2', 'I1'), ('R2', 'I2'))
        by_subject = Counter()
        by_condition_morphology = Counter()
        for frame in range(4):
            for condition in range(4):
                for pair in range(2):
                    row = next(r for r in rows if r[0] == frame and r[1] == condition and r[2] == pair)
                    by_subject[(subject_keys[frame][pair], row[3])] += 1
                    plus_first = row[4].index('u_plus') < row[4].index('u_minus')
                    by_condition_morphology[(condition, pair, row[3], plus_first)] += 1
        self.assertEqual(set(by_subject.values()), {4})
        self.assertEqual(len(by_subject), 8)
        self.assertEqual(len(by_condition_morphology), 32)
        self.assertEqual(set(by_condition_morphology.values()), {1})


if __name__ == '__main__':
    unittest.main()
