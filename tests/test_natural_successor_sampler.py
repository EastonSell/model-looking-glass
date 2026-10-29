"""Public or fabricated-only tests; no nonce or protected case is accessed."""
import hashlib
import json
import os
from pathlib import Path

import pytest

try:
    from looking_glass import natural_agreement_successor as successor
except ImportError:
    successor = None


def pool():
    return {
        'schema': 'mlg-natural-successor-pool-v1',
        'nouns': [{'id': f'n{i:02}', 'singular': f'item{i:02}', 'plural': f'items{i:02}'}
                  for i in range(16)],
        'contexts': ([{'id': f's{i:02}', 'number': 'singular', 'text': f'near object{i:02}'}
                      for i in range(4)] +
                     [{'id': f'p{i:02}', 'number': 'plural', 'text': f'near objects{i:02}'}
                      for i in range(4)]),
    }


class FakeTokenizer:
    def __init__(self, values):
        self.subjects = {form: 10 + i * 2 + k for i, noun in enumerate(values['nouns'])
                         for k, form in enumerate((noun['singular'], noun['plural']))}
        self.contexts = {row['text']: 100 + i for i, row in enumerate(values['contexts'])}
        self.targets = {' is': 318, ' are': 389, ' was': 373, ' were': 547}

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        if text == 'The':
            return [1]
        if text.startswith(' ') and text[1:] in self.contexts:
            return [self.contexts[text[1:]]]
        suffix = next((part for part in self.targets if text.endswith(part)), '')
        base = text[:-len(suffix)] if suffix else text
        assert base.startswith('The ')
        subject, _, context = base[4:].partition(' ')
        result = [1, self.subjects[subject]]
        if context:
            result.append(self.contexts[context])
        if suffix:
            result.append(self.targets[suffix])
        return result


def digest(record):
    raw = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                     allow_nan=False).encode('ascii')
    return hashlib.sha256(raw).hexdigest()


def test_fixed_nonce_balanced_allocation_and_original_order():
    assert successor is not None
    families = successor.allocate(pool(), bytes(32))
    assert [x['context_id'] for x in families] == ['p02', 'p00', 's01', 's03']
    assert [x['noun_ids'] for x in families] == [['n04', 'n12'], ['n03', 'n13'],
                                                  ['n01', 'n11'], ['n06', 'n07']]
    assert [x['subjects'] for x in families][0] == ['item04', 'items04', 'item12', 'items12']


def test_invalid_pool_and_nonce_fail_closed():
    assert successor is not None
    with pytest.raises(ValueError):
        successor.allocate(pool(), b'short')
    duplicate = pool()
    duplicate['nouns'][1]['id'] = duplicate['nouns'][0]['id']
    with pytest.raises(ValueError):
        successor.allocate(duplicate, bytes(32))
    invalid = pool()
    invalid['contexts'][0]['number'] = []
    with pytest.raises(ValueError):
        successor.allocate(invalid, bytes(32))


def test_all_pool_rows_and_prompt_plus_continuations_are_checked():
    assert successor is not None
    values = pool()
    tokenizer = FakeTokenizer(values)
    rows = successor.tokenize_candidate_pool(values, tokenizer)
    assert len(rows) == 16 * 2 * 8
    assert len(set(rows)) == len(rows)
    class BadContinuation(FakeTokenizer):
        def encode(self, text, *, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            if text.startswith('The ') and text.endswith(' were'):
                ids[-1] += 1
            return ids
    with pytest.raises(ValueError):
        successor.tokenize_candidate_pool(values, BadContinuation(values))


def test_duplicate_subject_token_is_rejected_for_selectable_quartet():
    assert successor is not None
    values = pool()
    tokenizer = FakeTokenizer(values)
    tokenizer.subjects['item01'] = tokenizer.subjects['item00']
    with pytest.raises(ValueError):
        successor.tokenize_candidate_pool(values, tokenizer)


def test_distinct_contexts_that_tokenize_to_duplicate_rows_are_rejected():
    assert successor is not None
    values = pool()
    tokenizer = FakeTokenizer(values)
    tokenizer.contexts['near object01'] = tokenizer.contexts['near object00']
    with pytest.raises(ValueError):
        successor.tokenize_candidate_pool(values, tokenizer)


def test_owned_record_filter_does_not_erase_another_records_prefix():
    assert successor is not None
    owned = {'record_id': 'own', 'category': 'planned', 'token_ids': [10, 11]}
    other = {'record_id': 'other', 'category': 'observed', 'token_ids': [10, 11, 12]}
    successor.reject_known_rows([(10, 11)], [owned], owned_records={'own': digest(owned)})
    with pytest.raises(ValueError):
        successor.reject_known_rows([(10, 11)], [owned, other], owned_records={'own': digest(owned)})
    with pytest.raises(ValueError):
        successor.reject_known_rows([(10, 11, 12)], [other])
    with pytest.raises(ValueError):
        successor.reject_known_rows([(10, 11)], [other])
    with pytest.raises(ValueError):
        successor.reject_known_rows([(10, 11, 12)],
                                    [{'record_id': 'past', 'category': 'observed', 'token_ids': [10, 11]}])
    with pytest.raises(ValueError):
        successor.reject_known_rows([(10, 11)], [owned], owned_records={'own': '0' * 64})
    with pytest.raises(ValueError):
        successor.reject_known_rows([(99, 98)], [{'record_id': 'malformed', 'category': 'observed',
                                                  'token_ids': [10, 11], 'target_token_id': True}])


def test_separate_salted_commitment_recomputes_the_fixed_pool_assignment():
    assert successor is not None
    values = pool()
    raw = json.dumps(values, sort_keys=True, separators=(',', ':')).encode('ascii')
    pins = {'pool_sha256': hashlib.sha256(raw).hexdigest(), 'protocol_sha256': '1' * 64,
            'exposure_sha256': '2' * 64, 'old_commitment': '3' * 64,
            'pool_clear_sha256': '4' * 64}
    commitment, record = successor.seal_record(raw, bytes(32), FakeTokenizer(values), **pins)
    decoded = successor.verify_record(raw, bytes(32), FakeTokenizer(values), record, commitment, **pins)
    assert len(decoded['families']) == 4
    assert sum(len(family['cases']) for family in decoded['families']) == 16
    assert decoded['pool_sha256'] == pins['pool_sha256']
    assert decoded['old_natural_commitment'] == pins['old_commitment']
    assert decoded['pool_clearance_sha256'] == pins['pool_clear_sha256']
    with pytest.raises(ValueError):
        successor.verify_record(raw, b'1' * 32, FakeTokenizer(values), record, commitment, **pins)
    with pytest.raises(ValueError):
        successor.verify_record(raw, bytes(32), FakeTokenizer(values), record + b' ', commitment, **pins)
    with pytest.raises(ValueError):
        successor.seal_record(raw, bytes(32), FakeTokenizer(values), **dict(pins, pool_sha256='f' * 64))


def test_published_pool_with_pinned_local_tokenizer():
    token_path = os.environ.get('MLG_SUCCESSOR_TOKENIZER')
    if not token_path:
        pytest.skip('set MLG_SUCCESSOR_TOKENIZER for the offline public-asset check')
    assert successor is not None
    token_path = Path(token_path)
    assert hashlib.sha256(token_path.read_bytes()).hexdigest() == (
        '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6')
    from tokenizers import Tokenizer
    class Adapter:
        def __init__(self):
            self.backend = Tokenizer.from_file(str(token_path))
        def encode(self, text, *, add_special_tokens=False):
            return self.backend.encode(text, add_special_tokens=add_special_tokens).ids
    path = Path(__file__).resolve().parents[1] / 'experiments/natural_agreement_successor_pool_v1.json'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'b3323d296a9f56a4232b4a403f7c8935896634b3d2b6e9fe7eee4aee56ba1126'
    published = json.loads(raw)
    previous = {'cat', 'dog', 'teacher', 'doctor', 'artist', 'writer', 'farmer', 'sailor',
                'chef', 'pilot', 'lawyer', 'student', 'nurse', 'musician', 'driver', 'singer'}
    assert not previous.intersection(noun['singular'] for noun in published['nouns'])
    rows = successor.tokenize_candidate_pool(published, Adapter())
    assert len(rows) == 256 and len(set(rows)) == 256
    input_path = path.with_name('natural_agreement_successor_inputs_v1.json')
    input_raw = input_path.read_bytes()
    assert hashlib.sha256(input_raw).hexdigest() == '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
    expected = {'schema': 'mlg-natural-successor-candidate-inputs-v1',
                'pool_sha256': hashlib.sha256(raw).hexdigest(),
                'tokenizer_sha256': hashlib.sha256(token_path.read_bytes()).hexdigest(),
                'cases': [{'input_ids': list(row)} for row in rows]}
    assert input_raw == json.dumps(expected, sort_keys=True, ensure_ascii=True,
                                   separators=(',', ':')).encode('ascii')
