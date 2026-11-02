"""Fabricated-only numerical tests for the independent saved-byte reader."""
import unittest

from looking_glass import agreement_substitution_science_v1 as reader


def fabricated_cases():
    subjects = {'R1': (200, 201), 'R2': (202, 203), 'I1': (204, 205), 'I2': (206, 207)}
    cases = []
    for frame in range(2):
        for condition_index, condition in enumerate(reader.CONDITIONS):
            suffix = [40] if condition_index == 0 else [30, 50 + 2 * frame + condition_index]
            for morph_index, (morph, keys) in enumerate((('regular', ('R1', 'R2')),
                                                          ('irregular', ('I1', 'I2')))):
                for lemma_index, key in enumerate(keys):
                    for number_index, number in enumerate(('singular', 'plural')):
                        tokens = [100 + frame, 20, subjects[key][number_index]] + suffix
                        cases.append({'case_id': f'f{frame}-c{condition_index}-{key}-{number}',
                            'frame_index': frame, 'condition_index': condition_index,
                            'condition': condition, 'role': 'target', 'morphology': morph,
                            'morphology_index': morph_index, 'lemma_index': lemma_index,
                            'subject_key': key, 'subject_number': number, 'token_ids': tokens,
                            'subject_index': 2, 'final_index': len(tokens) - 1})
            tokens = [100 + frame] + suffix
            cases.append({'case_id': f'f{frame}-c{condition_index}-context', 'frame_index': frame,
                'condition_index': condition_index, 'condition': condition, 'role': 'context',
                'morphology': None, 'morphology_index': None, 'lemma_index': None,
                'subject_key': None, 'subject_number': None, 'token_ids': tokens,
                'subject_index': None, 'final_index': len(tokens) - 1})
    return cases


def fake_margins(cases):
    result = {}
    for case in cases:
        baseline = 1.0 if case['subject_number'] == 'singular' else -1.0
        if case['role'] == 'context':
            baseline = 0.0
        clean = {metric: baseline for metric in reader.MARGINS}
        actions = {'clean': dict(clean), 'noop_l3_final': dict(clean), 'clean_after': dict(clean)}
        for axis in reader.AXES:
            if axis == 'u' and case['role'] == 'target':
                odd = {('R1', 'singular'): 2.0, ('R1', 'plural'): 0.0,
                       ('R2', 'singular'): 1.0, ('R2', 'plural'): -1.0,
                       ('I1', 'singular'): 2.0, ('I1', 'plural'): 0.0,
                       ('I2', 'singular'): 1.0, ('I2', 'plural'): -1.0}[
                           case['subject_key'], case['subject_number']]
            else:
                odd = 0.0
            actions[axis + '_plus'] = {metric: value + odd for metric, value in clean.items()}
            actions[axis + '_minus'] = {metric: value - odd for metric, value in clean.items()}
        result[case['case_id']] = actions
    return result


class IndependentScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = fabricated_cases()

    def test_fixed_schedule_counts(self):
        schedule = reader.expected_schedule(self.cases)
        self.assertEqual(len(schedule), 486)
        slots = {(axis + '_' + sign, slot): sum(x['action'] == axis + '_' + sign and x['slot'] == slot
                 for x in schedule) for axis in reader.AXES for sign in ('plus', 'minus') for slot in range(6)}
        self.assertEqual(set(slots.values()), {9})
        self.assertTrue(reader.validate_cases(self.cases, 600))

    def test_additive_number_and_lexical_counterexamples(self):
        values = reader.factorial((2.0, 0.0, 1.0, -1.0))
        self.assertEqual((values['opposite'], values['same'], values['delta'], values['number'],
                          values['lemma'], values['interaction']), (2.0, 1.0, 1.0, 2.0, 1.0, 0.0))
        hidden = reader.factorial((2.5, 0.5, -0.5, -2.5))
        self.assertEqual(hidden['delta'], 0.0)
        self.assertEqual(hidden['number'], 2.0)
        negative = reader.factorial((1.0, -1.0, -1.0, 1.0))
        self.assertEqual(negative['delta'], -2.0)

    def test_all_48_cases_retained_and_descriptive_rule_passes(self):
        report = reader.score_cases(self.cases, fake_margins(self.cases))
        self.assertEqual(len(report['cases']), 54)
        self.assertEqual(len(report['cells']), 12)
        self.assertEqual(len(report['fragments']), 6)
        self.assertEqual(report['baseline']['overall'], {'correct': 48, 'total': 48})
        self.assertTrue(report['baseline']['adequate'])
        self.assertEqual(report['criterion']['positive_cells'], 12)
        self.assertTrue(report['criterion']['pass'])
        self.assertEqual(report['aggregates']['overall']['present']['u']['delta'], 1.0)

    def test_ties_are_incorrect_and_baseline_limited_is_not_filtered(self):
        inputs = fake_margins(self.cases)
        changed = [c for c in self.cases if c['subject_key'] == 'R1'][:4]
        for case in changed:
            inputs[case['case_id']]['clean']['present'] = 0.0
        report = reader.score_cases(self.cases, inputs)
        self.assertEqual(report['baseline']['by_lemma']['R1']['correct'], 8)
        self.assertFalse(report['baseline']['adequate'])
        self.assertTrue(report['criterion']['baseline_limited'])
        self.assertEqual(report['criterion']['interpretation'], 'baseline_limited')
        self.assertEqual(len(report['cells']), 12)

    def test_control_uses_mean_absolute_delta_not_absolute_mean(self):
        inputs = fake_margins(self.cases)
        positive = {('R1', 'singular'): 4.0, ('R1', 'plural'): 0.0,
                    ('R2', 'singular'): 2.0, ('R2', 'plural'): -2.0,
                    ('I1', 'singular'): 4.0, ('I1', 'plural'): 0.0,
                    ('I2', 'singular'): 2.0, ('I2', 'plural'): -2.0}
        negative = {('R1', 'singular'): 1.0, ('R1', 'plural'): -1.0,
                    ('R2', 'singular'): -1.0, ('R2', 'plural'): 1.0,
                    ('I1', 'singular'): 1.0, ('I1', 'plural'): -1.0,
                    ('I2', 'singular'): -1.0, ('I2', 'plural'): 1.0}
        for case in self.cases:
            if case['role'] != 'target':
                continue
            odd = (positive if case['frame_index'] == 0 else negative)[case['subject_key'], case['subject_number']]
            actions = inputs[case['case_id']]
            clean = actions['clean']['present']
            actions['v1_plus']['present'] = clean + odd
            actions['v1_minus']['present'] = clean - odd
        report = reader.score_cases(self.cases, inputs)
        control = report['aggregates']['overall']['present']['v1']
        self.assertEqual(control['delta'], 0.0)
        self.assertEqual(control['mean_absolute_delta'], 2.0)
        self.assertFalse(report['criterion']['flags']['overall_mean_and_controls'])
        self.assertFalse(report['criterion']['pass'])

    def test_eight_positive_cells_and_four_per_morphology_are_sufficient(self):
        inputs = fake_margins(self.cases)
        for case in self.cases:
            if case['role'] != 'target':
                continue
            delta = 0.12 if case['condition_index'] < 2 else 0.0
            offset = {('R1', 'singular'): 2*delta, ('R1', 'plural'): 0.0,
                      ('R2', 'singular'): delta, ('R2', 'plural'): -delta,
                      ('I1', 'singular'): 2*delta, ('I1', 'plural'): 0.0,
                      ('I2', 'singular'): delta, ('I2', 'plural'): -delta}[
                          case['subject_key'], case['subject_number']]
            value = inputs[case['case_id']]['clean']['present']
            inputs[case['case_id']]['u_plus']['present'] = value + offset
            inputs[case['case_id']]['u_minus']['present'] = value - offset
        report = reader.score_cases(self.cases, inputs)
        self.assertEqual(report['criterion']['positive_cells'], 8)
        self.assertEqual(report['criterion']['positive_by_morphology'], {'regular': 4, 'irregular': 4})
        self.assertTrue(report['criterion']['pass'])
        for case in self.cases:
            if (case['frame_index'], case['condition_index'], case['morphology']) == (0, 0, 'regular'):
                actions = inputs[case['case_id']]
                value = actions['clean']['present']
                actions['u_plus']['present'] = value
                actions['u_minus']['present'] = value
        report = reader.score_cases(self.cases, inputs)
        self.assertEqual(report['criterion']['positive_cells'], 7)
        self.assertFalse(report['criterion']['flags']['positive_counts'])

    def test_control_tie_is_not_a_strict_win(self):
        inputs = fake_margins(self.cases)
        for actions in inputs.values():
            for sign in ('plus', 'minus'):
                actions['v1_' + sign]['present'] = actions['u_' + sign]['present']
        report = reader.score_cases(self.cases, inputs)
        self.assertEqual(report['aggregates']['overall']['present']['u']['delta'],
                         report['aggregates']['overall']['present']['v1']['mean_absolute_delta'])
        self.assertFalse(report['criterion']['flags']['overall_mean_and_controls'])
        self.assertFalse(report['criterion']['pass'])

    def test_missing_action_rejected(self):
        inputs = fake_margins(self.cases)
        del inputs[self.cases[0]['case_id']]['v2_minus']
        with self.assertRaises(ValueError):
            reader.score_cases(self.cases, inputs)


if __name__ == '__main__':
    unittest.main()
