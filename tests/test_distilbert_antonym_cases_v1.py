"""Tokenizer-only contract for the prospective antonym inventory."""
import importlib
import importlib.util
import re
from types import SimpleNamespace

import pytest


class FakeTokenizer:
    def __init__(self):
        self.words = {'[PAD]': 0, '[UNK]': 100, '[CLS]': 101, '[SEP]': 102, '[MASK]': 103}

    def token_to_id(self, word):
        if word not in self.words:
            self.words[word] = 104 + len(self.words)
        return self.words[word]

    def id_to_token(self, ident):
        return next((word for word, number in self.words.items() if number == ident), None)

    def encode(self, text, add_special_tokens=True):
        tokens = re.findall(r'\[MASK\]|[a-z]+|[.,]', text.lower().replace('[mask]', '[MASK]'))
        if add_special_tokens:
            tokens = ['[CLS]', *tokens, '[SEP]']
        return SimpleNamespace(ids=[self.token_to_id(word) for word in tokens], tokens=tokens)


def test_fixed_inventory_includes_every_role_and_accepted_factual():
    name = 'scripts.build_distilbert_antonym_cases_v1'
    assert importlib.util.find_spec(name) is not None, 'antonym case builder is not implemented'
    builder = importlib.import_module(name)
    rows = builder.specifications()
    assert len(rows) == 55
    assert len({row['id'] for row in rows}) == 55
    assert [item[0] for item in builder.CANDIDATES] == ['box', 'rope', 'drink', 'room']
    for obj in ('box', 'rope', 'drink', 'room'):
        matching = [row for row in rows if row['object'] == obj]
        assert len(matching) == 12
        for role in ('discovery', 'calibration-01', 'calibration-02',
                     'heldout-01', 'heldout-02', 'heldout-03', 'heldout-04'):
            row = next(row for row in matching if row['role'] == role)
            assert row['text'].count('[MASK]') == 1
            assert row['accepted'][0] == row['factual']
            assert row['counterfactual'] not in row['accepted']
            assert row['wrong_target'] == 'red'
        assert sum(row['role'] == 'same_object_locality' for row in matching) == 2
        assert sum(row['role'] == 'reverse_locality' for row in matching) == 2
        assert sum(row['role'] == 'alternate_use_locality' for row in matching) == 1
    assert sum(row['object'] is None for row in rows) == 7
    unrelated = next(row for row in rows if row['id'] == 'unrelated-relation-02')
    assert unrelated['text'] == 'A trumpet is a musical [MASK].'
    assert unrelated['accepted'] == ['instrument']


def test_encode_inventory_checks_alternatives_and_exact_mask_replacement():
    builder = importlib.import_module('scripts.build_distilbert_antonym_cases_v1')
    encode = getattr(builder, 'encode_inventory', None)
    assert callable(encode), 'antonym tokenizer validation is not implemented'
    result = encode(FakeTokenizer(), builder.specifications())
    assert len(result) == 55
    target = next(row for row in result if row['id'] == 'box-discovery')
    assert set(target['word_ids']) == {'small', 'little', 'large', 'red'}
    assert target['input_ids'][target['mask_index']] == 103
    assert all(value == 1 for value in target['attention_mask'])


def test_encode_inventory_rejects_duplicate_accepted_answers():
    builder = importlib.import_module('scripts.build_distilbert_antonym_cases_v1')
    encode = getattr(builder, 'encode_inventory', None)
    assert callable(encode), 'antonym tokenizer validation is not implemented'
    specs = builder.specifications()
    specs[0]['accepted'] = ['small', 'small']
    with pytest.raises(ValueError, match='accepted'):
        encode(FakeTokenizer(), specs)
