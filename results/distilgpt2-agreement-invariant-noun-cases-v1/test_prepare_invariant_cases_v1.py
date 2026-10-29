"""Fabricated-tokenizer tests; this file never assembles real study inputs."""

import json
from pathlib import Path
import re
import types
import unittest


def load(name):
    path = Path(__file__).with_name(name + '.py')
    if not path.is_file() and name == 'agreement_invariant_schedule_v1':
        path = Path(__file__).resolve().parents[2] / 'looking_glass' / (name + '.py')
    module = types.ModuleType(name)
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


prep = load('prepare_invariant_cases_v1')
schedule = load('agreement_invariant_schedule_v1')


class FakeTokenizer:
    def __init__(self):
        self.ids = {'this': 428, 'these': 777, 'is': 318, 'are': 389,
                    'was': 373, 'were': 547, 'the': 262, 'a': 257,
                    'in': 287, 'on': 319, ',': 11}
        self.reverse = {value: key for key, value in self.ids.items()}
        self.next_id = 1000

    def encode(self, text, add_special_tokens=False):
        result = []
        for word in re.findall(r'[A-Za-z]+|,', text.lower()):
            if word not in self.ids:
                self.ids[word] = self.next_id
                self.reverse[self.next_id] = word
                self.next_id += 1
            result.append(self.ids[word])
        return types.SimpleNamespace(ids=result)

    def decode(self, ids):
        return ' '.join(self.reverse[x] for x in ids)


def fake_priors(tokenizer):
    sheep = tokenizer.encode(' sheep').ids[0]
    tree = tokenizer.encode(' tree').ids[0]
    binding = {'cases': [{'input_ids': [sheep, tree], 'subject_index': 0}]}
    return binding, {'cases': []}, {'cases': []}, {'cases': []}


class PreparerTests(unittest.TestCase):
    def test_fixed_fabricated_grid_and_selection(self):
        tokenizer = FakeTokenizer()
        cases, groups, selection = prep.build(tokenizer, *fake_priors(tokenizer), schedule)
        self.assertEqual(len(cases), 68)
        self.assertEqual(len(groups), 68)
        self.assertEqual(sum(map(len, groups)), 612)
        self.assertEqual([x['form'] for x in selection['nouns']['discovery']], ['deer', 'fish'])
        self.assertEqual(selection['noun_decisions']['discovery'][0]['decision'], 'previous_or_selected_subject')
        self.assertEqual([x['forms'][0] for x in selection['attractors']], ['gate', 'lamp', 'rock', 'arch'])
        self.assertEqual([x['case_id'] for x in cases], [x[0] for x in schedule.cases()])
        self.assertEqual(sum(x['role'] == 'target' for x in cases), 48)
        self.assertEqual(sum(x['role'] == 'fragment' for x in cases), 12)
        self.assertEqual(sum(x['role'] == 'demonstrative' for x in cases), 8)

    def test_self_prefix_exemption_is_narrow(self):
        self.assertEqual(len(prep.make_groups([{'token_ids': [10, 20]}])[0]), 9)
        with self.assertRaisesRegex(ValueError, 'cross-case'):
            prep.make_groups([{'token_ids': [10, 20]}, {'token_ids': [10, 20, 318]}])
        with self.assertRaisesRegex(ValueError, 'cross-case'):
            prep.make_groups([{'token_ids': [10, 20, 318]}, {'token_ids': [10, 20]}])

    def test_inventory_identity_is_required(self):
        identity = {'model_id': prep.MODEL, 'model_revision': prep.REVISION,
                    'tokenizer_id': prep.MODEL, 'tokenizer_revision': prep.REVISION,
                    'tokenizer_sha256': prep.TOKENIZER_SHA, 'weights_sha256': prep.WEIGHTS_SHA}
        payload = {'identity': identity, 'records': [], 'version': 'fake-v1'}
        self.assertEqual(prep.verify_inventory(prep.compact(payload), 'fake-v1'), payload)
        for key in identity:
            changed = json.loads(json.dumps(payload))
            changed['identity'][key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                prep.verify_inventory(prep.compact(changed), 'fake-v1')

    def test_grouped_candidates_are_in_inventory_comparison(self):
        payload = {'records': [{'token_ids': [50, 60], 'candidate_token_ids': [318],
                                'target_token_id': 389, 'candidate_token_groups': {'x': [373]}}]}
        self.assertEqual(prep.inventory_sequences(payload),
                         {(50, 60), (50, 60, 318), (50, 60, 389), (50, 60, 373)})

    def test_broken_answer_boundary_is_rejected(self):
        class Broken(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                encoded = super().encode(text, add_special_tokens)
                if text.startswith(' this ') and text.endswith(' is'):
                    encoded.ids[-1] = 42
                return encoded
        tokenizer = Broken()
        with self.assertRaisesRegex(ValueError, 'pool exhausted'):
            prep.select_nouns(tokenizer, set())


if __name__ == '__main__':
    unittest.main()
