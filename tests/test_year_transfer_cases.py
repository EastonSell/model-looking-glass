import copy
import json
import math
from pathlib import Path

import pytest

from looking_glass import year_transfer as study


def test_fixed_case_inventory_and_matched_coordinates():
    path = Path(__file__).resolve().parents[1] / 'experiments/distilgpt2_year_transfer_v1.json'
    inventory = json.loads(path.read_bytes())
    assert inventory == study.make_inventory()
    rows = study.validate_inventory(inventory)
    assert len(rows) == 48
    assert len({tuple(row['input_ids']) for row in rows}) == 48
    for i in range(0, 48, 2):
        low, high = rows[i:i + 2]
        assert (low['year_role'], high['year_role']) == ('low', 'high')
        assert low['donor_case_id'] == high['case_id']
        assert high['donor_case_id'] == low['case_id']
        assert low['middle_ids'] == high['middle_ids']
        assert low['above_ids'] == high['above_ids']
        assert [j for j, (a, b) in enumerate(zip(low['input_ids'], high['input_ids'])) if a != b] == [low['year_position']]
        assert low['input_ids'][-1] == high['input_ids'][-1] == 1596


def test_inventory_rejects_identity_token_and_order_mutations():
    original = study.make_inventory()
    changes = [lambda x: x.__setitem__('protocol_sha256', '0' * 64),
               lambda x: x['cases'][0].__setitem__('pair_index', False),
               lambda x: x['cases'][0]['input_ids'].__setitem__(-1, 0),
               lambda x: x['cases'][0]['middle_ids'].__setitem__(0, 0),
               lambda x: x['cases'][0].__setitem__('donor_case_id', x['cases'][0]['case_id']),
               lambda x: x['cases'].reverse(),
               lambda x: x.__setitem__('known_exposure_sha256', '0' * 64)]
    for change in changes:
        data = copy.deepcopy(original)
        change(data)
        with pytest.raises(ValueError):
            study.validate_inventory(data)


def test_tokenizer_validator_accepts_both_encode_interfaces():
    rows = study.make_inventory()['cases']
    encodings = {}
    for row in rows:
        encodings[row['text']] = row['input_ids']
        for value, token in zip(row['middle_values'] + row['above_values'], row['middle_ids'] + row['above_ids']):
            encodings[f'{value:02d}'] = [token]
            encodings[row['text'] + f'{value:02d}'] = row['input_ids'] + [token]

    class ListTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return encodings[text]

    class ObjectTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return type('Encoding', (), {'ids': encodings[text]})()

    study.validate_tokenizer(ListTokenizer(), rows)
    study.validate_tokenizer(ObjectTokenizer(), rows)


def test_actual_cached_tokenizer_interfaces_when_available():
    import hashlib
    cache = (Path(__file__).resolve().parents[3] / 'hf-cache/hub/models--distilbert--distilgpt2/snapshots'
             / study.MODEL_REVISION / 'tokenizer.json')
    if not cache.exists():
        pytest.skip('pinned offline tokenizer is unavailable')
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == study.TOKENIZER_SHA256
    from tokenizers import Tokenizer
    from transformers import GPT2TokenizerFast
    rows = study.make_inventory()['cases']
    study.validate_tokenizer(Tokenizer.from_file(str(cache)), rows)
    study.validate_tokenizer(GPT2TokenizerFast(tokenizer_file=str(cache)), rows)


def test_band_score_is_stable_and_rejects_nonfinite():
    logits = [2.0**100] * 12
    assert study.band_score(logits, [0, 1, 2], [3, 4, 5]) == 0.0
    logits = [0.0] * 12
    logits[0:3] = [2.0, 2.0, 2.0]
    assert study.band_score(logits, [0, 1, 2], [3, 4, 5]) == pytest.approx(2.0)
    logits[3] = math.nan
    with pytest.raises(ValueError):
        study.band_score(logits, [0, 1, 2], [3, 4, 5])


def test_clean_gates_are_independent_and_strict():
    rows = study.make_inventory()['cases']
    scores = {}
    for row in rows:
        by_condition = {'forward': 1.5, 'reverse': -1.0, 'neutral': 1.0}
        scores[row['case_id']] = by_condition[row['condition']] if row['year_role'] == 'low' else 0.0
    summary = study.summarize_clean(rows, scores)
    assert summary['comparator_gate_passed']
    assert summary['preference_gate_passed']
    assert len(summary['pairs']) == 8
    for row in rows:
        if row['condition'] == 'reverse' and row['year_role'] == 'low':
            scores[row['case_id']] = 0.5
    summary = study.summarize_clean(rows, scores)
    assert not summary['comparator_gate_passed']
    assert summary['preference_gate_passed']
    for row in rows:
        if row['condition'] == 'neutral' and row['year_role'] == 'low':
            scores[row['case_id']] = 0.5
    summary = study.summarize_clean(rows, scores)
    assert not summary['comparator_gate_passed']
    assert not summary['preference_gate_passed']


def test_clean_summary_rejects_missing_or_invalid_values():
    rows = study.make_inventory()['cases']
    scores = {row['case_id']: 0.0 for row in rows}
    scores[rows[0]['case_id']] = math.nan
    with pytest.raises(ValueError):
        study.summarize_clean(rows, scores)
    del scores[rows[0]['case_id']]
    with pytest.raises(ValueError):
        study.summarize_clean(rows, scores)
