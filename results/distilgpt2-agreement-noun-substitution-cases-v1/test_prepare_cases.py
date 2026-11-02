"""Synthetic tokenizer and inventory checks; no pretrained model is used."""
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = Path(__file__).with_name('prepare_cases.py')
spec = importlib.util.spec_from_file_location('substitution_case_preparer', source)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


class FakeTokenizer:
    def __init__(self):
        self.ids = {text.strip(): token for text, token in preparer.ANSWERS}
        self.next_id = 1000

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.ids:
                self.ids[word] = self.next_id
                self.next_id += 1
            ids.append(self.ids[word])
        return SimpleNamespace(ids=ids)


def old_data():
    binding = {'cases': [{'input_ids': [9, 10], 'subject_index': 0}]}
    direction = {'cases': [{'role': 'target', 'token_ids': [11, 12], 'subject_index': 0}]}
    subjects = {'cases': [{'role': 'target', 'condition': 'attractor_singular',
                           'token_ids': [13, 14], 'subject_index': 0}]}
    return binding, direction, subjects


class PreparationTests(unittest.TestCase):
    def test_fabricated_cases_and_factorial_identity(self):
        with patch.multiple(preparer, FRAMES=('F0,', 'F1,'),
                            REGULAR=(('ra', 'ras'), ('rb', 'rbs')),
                            IRREGULAR=(('ia', 'ias'), ('ib', 'ibs')),
                            ATTRACTORS=(('ca', 'cas'), ('cb', 'cbs'))):
            cases, groups, selection = preparer.build_cases(FakeTokenizer(), *old_data())
        self.assertEqual(len(cases), 54)
        self.assertEqual(sum(row['role'] == 'target' for row in cases), 48)
        self.assertEqual(sum(row['role'] == 'context' for row in cases), 6)
        self.assertEqual(len(groups), 54)
        self.assertTrue(all(len(group) == 9 for group in groups))
        self.assertEqual(len(selection['regular']), 2)
        self.assertEqual(len(selection['irregular']), 2)

    def test_prior_tokens_are_excluded(self):
        tokenizer = FakeTokenizer()
        forbidden = {tokenizer.encode(' old').ids[0]}
        selected, log = preparer.select_pairs(tokenizer, (('old', 'olds'), ('fresh', 'freshs')),
                                              1, forbidden)
        self.assertEqual(log[0]['decision'], 'previous_or_selected_token')
        self.assertEqual(selected[0]['forms'], ['fresh', 'freshs'])

    def test_grouped_candidates_are_preserved(self):
        payload = {'records': [{'token_ids': [1, 2], 'candidate_token_ids': [3],
                                'candidate_token_groups': {'g': [4, 5]}, 'target_token_id': 6}]}
        self.assertEqual(preparer.inventory_sequences(payload),
                         {(1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 6)})

    def test_previous_mass_attractor_is_forbidden(self):
        binding, direction, subjects = old_data()
        subjects['cases'].append({'role': 'target', 'condition': 'mass',
                                  'token_ids': [15, 16], 'subject_index': 0})
        previous_subjects, previous_attractors = preparer.prior_tokens(binding, direction, subjects)
        self.assertIn(15, previous_subjects)
        self.assertIn(16, previous_attractors)

    def test_canonical_inventory_rejects_other_model_identity(self):
        payload = {'version': 'v-test', 'records': [], 'identity': {
            'model_id': preparer.MODEL_ID, 'model_revision': preparer.MODEL_REVISION,
            'tokenizer_id': preparer.MODEL_ID, 'tokenizer_revision': preparer.MODEL_REVISION,
            'tokenizer_sha256': preparer.TOKENIZER_SHA, 'weights_sha256': preparer.WEIGHTS_SHA}}
        def canonical():
            return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('ascii')
        raw = canonical()
        self.assertEqual(preparer.verify_inventory(raw, hashlib.sha256(raw).hexdigest(), 'v-test'), payload)
        payload['identity']['model_id'] = 'other/model'
        raw = canonical()
        with self.assertRaisesRegex(ValueError, 'identity'):
            preparer.verify_inventory(raw, hashlib.sha256(raw).hexdigest(), 'v-test')

    def test_own_extensions_allowed_but_other_case_prefix_rejected(self):
        self.assertEqual(len(preparer.validate_groups([{'token_ids': [1, 2]}])[0]), 9)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            preparer.validate_groups([{'token_ids': [1, 2]}, {'token_ids': [1, 2, 318]}])


if __name__ == '__main__':
    unittest.main()
