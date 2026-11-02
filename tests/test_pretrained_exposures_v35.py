"""Model-free v35 append and collision tests with fabricated predecessors."""
import copy
import json
from pathlib import Path

import pytest

from looking_glass import pretrained_exposures_v35 as v35


def predecessor(records=None):
    return {'records': records or [], 'version': 'fabricated', 'update': {}, 'predecessor': {}}


def test_fixed_public_case_rows_and_append(monkeypatch):
    case_path = Path(__file__).resolve().parents[1] / v35.CASE_PATH
    document = json.loads(case_path.read_bytes())
    expected = [(row['case_id'], row['role'], row['text'], tuple(row['token_ids']))
                for row in document['cases']]
    assert list(v35.AGREEMENT_NOUN_SUBSTITUTION_CASES) == expected
    old = {'record_id': 'old', 'token_ids': [50256, 1], 'category': 'planned'}
    source = predecessor([old])
    original = copy.deepcopy(source)
    monkeypatch.setattr(v35, '_prefixes', lambda _records: range(v35.FORBIDDEN_PREFIX_COUNT))
    payload = v35._build_payload(source)
    assert source == original and payload['records'][0] == old
    assert len(payload['records']) == 55
    for record, (case_id, role, text, row) in zip(payload['records'][1:], expected):
        assert record['record_id'] == 'planned-' + case_id
        assert record['category'] == 'planned' and record['case_role'] == role
        assert record['public_text'] == text and record['token_ids'] == list(row)
        assert record['candidate_token_ids'] == list(v35.AGREEMENT_NOUN_SUBSTITUTION_ANSWER_IDS)


@pytest.mark.parametrize('kind', ['equal', 'shorter', 'longer', 'candidate', 'grouped'])
def test_foreign_prefix_relation_rejected(kind, monkeypatch):
    row = list(v35.AGREEMENT_NOUN_SUBSTITUTION_ROWS[0])
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
    monkeypatch.setattr(v35, '_prefixes', lambda _records: range(v35.FORBIDDEN_PREFIX_COUNT))
    with pytest.raises(ValueError, match='known prefix'):
        v35._build_payload(predecessor([foreign]))
