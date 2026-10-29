"""Tokenizer-free structural and collision checks for the prospective cases."""
from copy import deepcopy
import re
import unittest
from unittest.mock import patch

from scripts import build_distilbert_color_edit_cases_v1 as builder
from scripts import check_distilbert_color_edit_exposures_v1 as exposures


class FakeTokenizer:
    cls_token_id, sep_token_id, mask_token_id, unk_token_id = 101, 102, 103, 100
    all_special_ids = [0, 100, 101, 102, 103]

    def __init__(self):
        self.ids = {'[MASK]': 103}
        self.words = {103: '[MASK]'}

    def encode(self, text, add_special_tokens=True):
        ids = []
        for token in re.findall(r'\[MASK\]|[A-Za-z]+|\.', text):
            if token not in self.ids:
                value = 1000 + len(self.ids)
                self.ids[token], self.words[value] = value, token
            ids.append(self.ids[token])
        return [101, *ids, 102] if add_special_tokens else ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.words[ids]
        return [self.words.get(value, {101: '[CLS]', 102: '[SEP]'}.get(value)) for value in ids]


class CaseTests(unittest.TestCase):
    def test_fixed_inventory_and_one_token_fill(self):
        rows = builder.encode_inventory(FakeTokenizer(), builder.specifications())
        self.assertEqual(len(rows), 37)
        self.assertEqual(sum(row['role'] == 'discovery' for row in rows), 3)
        self.assertEqual(sum(row['role'].startswith('heldout-') for row in rows), 12)
        self.assertEqual(sum(row['role'] == 'same_object_locality' for row in rows), 6)
        self.assertEqual(sum(row['role'] in ('other_color', 'unrelated') for row in rows), 10)
        banana = next(row for row in rows if row['id'] == 'banana-locality-02')
        self.assertEqual(banana['accepted'], ['plant'])
        self.assertEqual(banana['attention_mask'], [1] * len(banana['input_ids']))

    def test_invalid_accepted_set_and_alignment_are_rejected(self):
        rows = deepcopy(builder.specifications())
        locality = next(row for row in rows if row['id'] == 'banana-locality-01')
        locality['accepted'] = ['fruit', 'fruit']
        with self.assertRaises(ValueError):
            builder.encode_inventory(FakeTokenizer(), rows)
        rows = deepcopy(builder.specifications())
        locality = next(row for row in rows if row['id'] == 'banana-locality-01')
        locality['accepted'] = ['plant']
        with self.assertRaises(ValueError):
            builder.encode_inventory(FakeTokenizer(), rows)

    def test_comparison_removes_sep_and_rejects_mask_namespace_mismatch(self):
        short = [101, 200, 103, 102]
        extension = [101, 200, 103, 300, 102]
        distinct = [101, 201, 103, 102]
        self.assertTrue(exposures.prefix_relation(short, extension))
        self.assertTrue(exposures.prefix_relation(extension, short))
        self.assertFalse(exposures.prefix_relation(short, distinct))
        with self.assertRaises(ValueError):
            exposures.valid_input({'input_ids': short, 'attention_mask': [1, 1, 1, 0]})
        with self.assertRaises(ValueError):
            exposures.valid_input({'input_ids': short, 'attention_mask': [True, 1, 1, 1]})

    def test_git_reader_uses_exact_commit_and_preserves_bytes(self):
        with patch.object(exposures.subprocess, 'run') as run:
            run.return_value.stdout = b'{"example": 1}\n'
            self.assertEqual(exposures.git_blob('repo', 'a' * 40, 'experiments/cases.json'),
                             b'{"example": 1}\n')
            run.assert_called_once_with(
                ['git', '-C', 'repo', 'show', 'a' * 40 + ':experiments/cases.json'],
                check=True, stdout=exposures.subprocess.PIPE, stderr=exposures.subprocess.PIPE)


if __name__ == '__main__':
    unittest.main()
