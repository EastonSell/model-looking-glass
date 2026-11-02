"""Public case binding and collision checks; no model or private inputs."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from looking_glass import pretrained_exposures_v36 as v36


def predecessor(records=None):
    return {'records': records or [], 'version': 'fabricated', 'update': {}, 'predecessor': {}}


def test_public_case_rows_and_append(monkeypatch):
    raw = (Path(__file__).resolve().parents[1] / v36.CASE_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == v36.CASE_SHA256
    cases = json.loads(raw)['cases']
    expected = [(row['case_id'], row['gpt']['text'], tuple(row['gpt']['input_ids'])) for row in cases]
    assert list(v36.CROSS_MODEL_CASES) == expected
    old = {'record_id': 'old', 'token_ids': [50256, 1], 'category': 'planned'}
    source = predecessor([old])
    original = copy.deepcopy(source)
    monkeypatch.setattr(v36, '_prefixes', lambda _records: range(v36.FORBIDDEN_PREFIX_COUNT))
    payload = v36._build_payload(source)
    assert source == original and payload['records'][0] == old
    assert len(payload['records']) == 33
    for record, (case_id, text, row) in zip(payload['records'][1:], expected):
        assert record['record_id'] == 'planned-cross-model-' + case_id
        assert record['public_text'] == text and record['token_ids'] == list(row)
        assert record['candidate_token_ids'] == list(v36.ANSWER_IDS)
        assert record['reference_commit'] == v36.CASE_COMMIT


@pytest.mark.parametrize('kind', ['equal', 'shorter', 'longer', 'candidate', 'grouped'])
def test_foreign_prefix_relation_rejected(kind, monkeypatch):
    row = list(v36.CROSS_MODEL_CASES[0][2])
    foreign = {'record_id': 'foreign', 'category': 'planned'}
    if kind == 'equal':
        foreign['token_ids'] = row
    elif kind == 'shorter':
        foreign['token_ids'] = row[:-1]
    elif kind == 'longer':
        foreign['token_ids'] = row + [318, 123]
    elif kind == 'candidate':
        foreign['token_ids'] = row[:-1]
        foreign['candidate_token_ids'] = [row[-1]]
    else:
        foreign['token_ids'] = row[:-1]
        foreign['candidate_token_groups'] = {'group': [row[-1]]}
    monkeypatch.setattr(v36, '_prefixes', lambda _records: range(v36.FORBIDDEN_PREFIX_COUNT))
    with pytest.raises(ValueError, match='known prefix'):
        v36._build_payload(predecessor([foreign]))


def test_distinct_proposed_cases_cannot_have_prefix_relation(monkeypatch):
    case = v36.CROSS_MODEL_CASES[0]
    replacements = list(v36.CROSS_MODEL_CASES)
    replacements[1] = ('other', 'other', case[2] + (318,))
    monkeypatch.setattr(v36, 'CROSS_MODEL_CASES', tuple(replacements))
    monkeypatch.setattr(v36, '_prefixes', lambda _records: range(v36.FORBIDDEN_PREFIX_COUNT))
    with pytest.raises(ValueError, match='distinct cross-model cases'):
        v36._build_payload(predecessor())
