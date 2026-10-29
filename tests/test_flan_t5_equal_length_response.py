"""No-model checks for fixed selection, the six-pair rule and evidence bounds."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from looking_glass import flan_t5_equal_length_response as subject


def cells(statuses):
    return [{'pair_id': i // 2, 'template_id': ('translate', 'question')[i % 2], 'status': status}
            for i, status in enumerate(statuses)]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(subject.CASES.read_bytes())
        self.screen = json.loads(subject.SCREEN.read_bytes())

    def test_fixed_screen_and_selected_cases_match(self):
        subject.check_selection(self.manifest, self.screen)

    def test_reordering_or_changing_screened_cases_fails(self):
        for mutation in ('selection', 'selected_case', 'screened_case'):
            manifest, screen = copy.deepcopy(self.manifest), copy.deepcopy(self.screen)
            if mutation == 'selection':
                screen['candidates'][0]['eligible'] = False
            elif mutation == 'selected_case':
                manifest['cases'][0]['source'][0]['text'] += '!'
            else:
                screen['candidates'][0]['cases'][0]['targets'][0]['text'] += '!'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.check_selection(manifest, screen)


class AggregateTests(unittest.TestCase):
    def test_majority_ties_and_missing_pairs(self):
        def summary(pair_status):
            lookup = {'positive': ('positive', 'positive'), 'negative': ('positive', 'negative'),
                      'unevaluable': ('negative', 'baseline-unsupported')}
            return cells([s for p in pair_status for s in lookup[p]])
        examples = [
            (['positive']*3 + ['negative']*2 + ['unevaluable'], 'positive'),
            (['negative']*3 + ['positive']*2 + ['unevaluable'], 'negative'),
            (['positive']*3 + ['negative']*3, 'inconclusive'),
            (['positive']*2 + ['negative'] + ['unevaluable']*3, 'inconclusive'),
        ]
        for pairs, expected in examples:
            with self.subTest(pairs=pairs):
                got = summary(pairs)
                self.assertEqual(subject.pair_decisions(got), pairs)
                self.assertEqual(subject.aggregate_decision(got), expected)

    def test_invalid_or_unfinished_cell_is_incomplete(self):
        valid = cells(['positive'] * 12)
        valid[0]['status'] = 'failed'
        self.assertEqual(subject.aggregate_decision(valid), 'incomplete')
        self.assertEqual(subject.aggregate_decision(cells(['positive'] * 11)), 'incomplete')
        swapped = cells(['positive'] * 12)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertEqual(subject.aggregate_decision(swapped), 'incomplete')


class EvidenceTests(unittest.TestCase):
    def test_fixed_shape_budget_and_reserve_fit(self):
        bound = subject.fixed_evidence_bound()
        self.assertEqual(bound['valid_tensor_bytes'], 936 * (2*32128 + 4*9*512) * 4)
        self.assertLess(bound['total_bytes'] + subject.MAX_RETURN_BYTES + 3*subject._JSON_LIMIT,
                        bound['limit_bytes'])

    def test_json_file_limit_and_terminal_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = subject.Evidence(Path(directory) / 'out')
            with self.assertRaises(ValueError):
                evidence.json('forward-000.json', {'value': 'x' * 65536})
            self.assertFalse((evidence.path / 'forward-000.json').exists())
            evidence.json('completed.json', {'value': 'x' * 65536}, reserve=0)
            self.assertGreater((evidence.path / 'completed.json').stat().st_size, 65536)


if __name__ == '__main__':
    unittest.main()
