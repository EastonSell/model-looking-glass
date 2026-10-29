"""Synthetic-only checks of the agreement-study factorial arithmetic."""
import copy
import math
import unittest

from looking_glass.agreement_metrics import StudyInvalid, score_factorial


def fixture():
    cases = []
    for lemma in ('alpha', 'beta'):
        for number in ('singular', 'plural'):
            name = lemma + '-' + number
            # Make one baseline incorrect; it must remain in the analysis.
            margin = 2.0 if number == 'singular' else -2.0
            if name == 'beta-plural':
                margin = 1.0
            cases.append({'id': name, 'quartet': 'context-a', 'lemma': lemma,
                          'number': number, 'clean_is': margin, 'clean_are': 0.0})
    patches = []
    for target in cases:
        for donor in cases:
            same_lemma = target['lemma'] == donor['lemma']
            same_number = target['number'] == donor['number']
            shift = (0.0 if same_number else 2.0) + (0.5 if not same_lemma else 0.0)
            weight = 1.0 if target['number'] == 'singular' else -1.0
            patched_margin = target['clean_is'] - weight * shift
            patches.append({'site': 'layer-2-attn', 'target': target['id'],
                            'donor': donor['id'], 'patched_is': patched_margin,
                            'patched_are': 0.0})
    return cases, patches


class AgreementFactorialTests(unittest.TestCase):
    def test_complete_factorial_signed_cells_and_interaction(self):
        cases, patches = fixture()
        result = score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        row = result['quartets'][0]
        self.assertEqual(row['site'], 'layer-2-attn')
        self.assertEqual(row['quartet'], 'context-a')
        self.assertEqual(row['cell_means'], {
            'same_lemma_same_number': 0.0,
            'same_lemma_opposite_number': 2.0,
            'different_lemma_same_number': 0.5,
            'different_lemma_opposite_number': 2.5,
        })
        self.assertEqual(row['number_main_contrast'], 2.0)
        self.assertEqual(row['lemma_number_interaction'], 0.0)
        self.assertEqual(row['cell_counts'], {key: 4 for key in row['cell_means']})
        self.assertEqual(result['baseline']['incorrect'], 1)
        self.assertEqual(result['baseline']['n'], 4)
        self.assertEqual(len(result['pairs']), 16)

    def test_missing_or_duplicate_pair_does_not_shrink_denominators(self):
        cases, patches = fixture()
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches[:-1], ['layer-2-attn'], expected_quartets=1)
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches + [copy.deepcopy(patches[0])], ['layer-2-attn'], expected_quartets=1)
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'])

    def test_quartets_are_equally_weighted_and_order_independent(self):
        cases, patches = fixture()
        cases2, patches2 = fixture()
        by_id = {item['id']: item for item in cases2}
        for item in patches2:
            target = by_id[item['target']]
            donor = by_id[item['donor']]
            if target['number'] != donor['number']:
                weight = 1 if target['number'] == 'singular' else -1
                item['patched_is'] -= 2 * weight
            item['target'] += '-other'
            item['donor'] += '-other'
        for item in cases2:
            item['id'] += '-other'
            item['quartet'] = 'context-b'
        combined_cases, combined_patches = cases + cases2, patches + patches2
        result = score_factorial(combined_cases, combined_patches, ['layer-2-attn'], expected_quartets=2)
        reversed_result = score_factorial(list(reversed(combined_cases)),
                                           list(reversed(combined_patches)),
                                           ['layer-2-attn'], expected_quartets=2)
        self.assertEqual(result, reversed_result)
        self.assertEqual(result['site_summaries'][0]['mean_number_main_contrast'], 3.0)

    def test_main_contrast_uses_one_fsum_over_all_four_signed_cells(self):
        cases, patches = fixture()
        for case in cases:
            case['clean_is'] = case['clean_are'] = 0.0
        by_id = {case['id']: case for case in cases}
        for patch in patches:
            target, donor = by_id[patch['target']], by_id[patch['donor']]
            same_lemma = target['lemma'] == donor['lemma']
            same_number = target['number'] == donor['number']
            u = {(True, True): 0.0, (True, False): float(2**100),
                 (False, True): 1.0, (False, False): -float(2**100)}[(same_lemma, same_number)]
            weight = 1 if target['number'] == 'singular' else -1
            patch['patched_is'] = -weight * u
            patch['patched_are'] = 0.0
        result = score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        self.assertEqual(result['quartets'][0]['number_main_contrast'], -0.5)

    def test_invalid_factorial_and_nonfinite_values_are_rejected(self):
        cases, patches = fixture()
        cases[0]['number'] = 'plural'
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        cases, patches = fixture()
        patches[0]['patched_is'] = math.nan
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        cases, patches = fixture()
        patches[0]['patched_is'] = True
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        cases, patches = fixture()
        patches[0]['patched_is'] = 0.1  # A Python binary64, not an exact decoded binary32 value.
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        cases, patches = fixture()
        patches[0]['patched_is'] = 10**400
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)

    def test_selected_self_logits_must_match_clean_and_subnormals_are_accepted(self):
        cases, patches = fixture()
        patches[0]['patched_is'] += 0.25
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        cases, patches = fixture()
        tiny = math.ldexp(1.0, -149)
        cases[0]['clean_are'] = tiny
        patches[0]['patched_are'] = tiny
        result = score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)
        self.assertEqual(next(pair['clean_are'] for pair in result['pairs']
                              if pair['target'] == cases[0]['id'] and pair['donor'] == cases[0]['id']), tiny)

    def test_neither_cross_quartet_donors_nor_unknown_sites_are_allowed(self):
        cases, patches = fixture()
        cases2, patches2 = fixture()
        for item in cases2:
            item['id'] += '-other'
            item['quartet'] = 'context-b'
        for item in patches2:
            item['target'] += '-other'
            item['donor'] += '-other'
        cases += cases2
        patches += patches2
        patches[0]['donor'] = cases2[0]['id']
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=2)
        cases, patches = fixture()
        patches[0]['site'] = 'unknown'
        with self.assertRaises(StudyInvalid):
            score_factorial(cases, patches, ['layer-2-attn'], expected_quartets=1)


if __name__ == '__main__':
    unittest.main()
