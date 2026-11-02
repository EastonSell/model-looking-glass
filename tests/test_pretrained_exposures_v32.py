"""Verify a planned public two-attractor patch reservation without a model."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from looking_glass import pretrained_exposures_v31 as prior
from looking_glass import pretrained_exposures_v32 as current


@pytest.fixture(scope='module')
def predecessor():
    payload = prior.inventory_payload()
    assert hashlib.sha256(current._canonical(payload)).hexdigest() == current.PREDECESSOR_SHA256
    return payload


def test_cases_and_predecessor_preserved(predecessor):
    raw = (Path(__file__).resolve().parents[1] / current.CASE_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == current.CASE_SHA256
    cases = json.loads(raw)['cases']
    payload = current._build_payload(predecessor)
    assert payload['records'][:len(predecessor['records'])] == predecessor['records']
    added = payload['records'][len(predecessor['records']):]
    assert len(cases) == len(added) == 32
    for case, record in zip(cases, added):
        assert record['record_id'] == 'planned-two-attractor-patch-' + case['case_id']
        assert record['category'] == 'planned' and record['case_role'] == 'evaluation'
        assert record['token_ids'] == case['input_ids']
        assert record['public_text'] == case['text']
        assert record['candidate_token_ids'] == [318, 389, 373, 547]
        assert record['reference_commit'] == current.CASES_COMMIT
    assert hashlib.sha256(current._canonical(payload)).hexdigest() == current.INVENTORY_SHA256
    assert len(current._prefixes(payload['records'])) == current.FORBIDDEN_PREFIX_COUNT


def test_foreign_equal_or_prefix_input_is_rejected(predecessor):
    for row in (current.TWO_ATTRACTOR_PATCH_ROWS[0], current.TWO_ATTRACTOR_PATCH_ROWS[0][:-1]):
        bad = copy.deepcopy(predecessor)
        bad['records'].append({'record_id': 'foreign-collision', 'token_ids': list(row)})
        with pytest.raises(ValueError, match='prefix'):
            current._build_payload(bad)


def test_repeated_own_reservation_is_rejected(predecessor):
    bad = copy.deepcopy(predecessor)
    bad['records'].append({'record_id': 'planned-two-attractor-patch-' + current.TWO_ATTRACTOR_PATCH_CASES[0][0], 'token_ids': [50256]})
    with pytest.raises(ValueError, match='identities'):
        current._build_payload(bad)
