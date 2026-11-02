"""Model-free regression tests for public case selection and exposure checks."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('subject_cases', Path(__file__).resolve().parents[1] / 'tools' / 'prepare_agreement_subject_cases_v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return Encoded({' a': [1], ' as': [2], ' b': [3,4], ' bs': [5],
                        ' c': [6], ' cs': [7], ' d': [8], ' ds': [9]}[text])


class PreparationTests(unittest.TestCase):
    def test_selection_skips_prior_and_multi_token_forms(self):
        selected, log = module.select_pairs(FakeTokenizer(),
            (('a','as'), ('b','bs'), ('c','cs'), ('d','ds')), 2, {1})
        self.assertEqual([x['forms'] for x in selected], [['c','cs'], ['d','ds']])
        self.assertEqual([x['decision'] for x in log],
                         ['previous_or_selected_token', 'not_one_token', 'selected', 'selected'])

    def test_inventory_includes_all_candidate_groups_and_both_prefix_directions(self):
        records = [{'token_ids':[10,11], 'target_token_id':12, 'candidate_token_ids':[13],
                    'candidate_token_groups':{'one':[14,15], 'two':[16]}}]
        values = module.inventory_sequences({'records':records})
        self.assertEqual(values, {(10,11), (10,11,12), (10,11,13), (10,11,14), (10,11,15), (10,11,16)})
        self.assertTrue(module.related((10,), (10,11)))
        self.assertTrue(module.related((10,11), (10,)))
        self.assertFalse(module.related((10,12), (10,11)))
        with self.assertRaises(ValueError):
            module.inventory_sequences({'records':[{'token_ids':[True]}]})


if __name__ == '__main__':
    unittest.main()
