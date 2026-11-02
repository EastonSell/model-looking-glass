import copy
import hashlib
import json
from pathlib import Path

import pytest

from looking_glass import pretrained_exposures_v32 as prior
from looking_glass import pretrained_exposures_v33 as current


@pytest.fixture(scope='module')
def predecessor():
    return prior.inventory_payload()


def test_reserved_science_rows_match_public_manifest_and_preserve_predecessor(predecessor):
    path = Path(__file__).resolve().parents[1] / current.CASE_PATH
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == current.CASE_SHA256
    manifest = json.loads(raw)
    expected = []
    for case in manifest['comparisons']:
        if case['id'] in ('gas', 'ligand'):
            for role, key in (('active', 'reference'), ('padded', 'service')):
                entry = case[key]
                expected.append((case['id'] + '-' + role, case['text'], tuple(entry['input_ids']),
                                 tuple(entry['attention_mask'])))
    assert current.SCIENCE_SERVICE_CASES == tuple(expected)
    result = current._build_payload(predecessor)
    assert result['records'][:-4] == predecessor['records']
    assert predecessor['version'] == prior.INVENTORY_VERSION
    assert result['update'] == {'planned_short_science_rows': 4}
    for record, case in zip(result['records'][-4:], expected, strict=True):
        assert record['record_id'] == 'planned-short-science-' + case[0]
        assert record['category'] == 'planned'
        assert record['token_ids'] == list(case[2]) and record['attention_mask'] == list(case[3])
        assert record['reference_commit'] == current.CASE_COMMIT
    assert hashlib.sha256(current._canonical(result)).hexdigest() == current.INVENTORY_SHA256
    assert len(current._prefixes(result['records'])) == current.FORBIDDEN_PREFIX_COUNT


def test_only_the_two_intentional_active_padded_pairs_share_a_prefix():
    rows = [case[2] for case in current.SCIENCE_SERVICE_CASES]
    relations = {(i, j) for i in range(4) for j in range(i + 1, 4)
                 if current._related(rows[i], rows[j])}
    assert relations == {(0, 1), (2, 3)}


@pytest.mark.parametrize('change', ['mask', 'boolean', 'row', 'missing'])
def test_changed_fixed_case_is_rejected_before_inventory_append(predecessor, monkeypatch, change):
    cases = list(current.SCIENCE_SERVICE_CASES)
    name, text, row, mask = cases[1]
    if change == 'mask':
        mask = (1,) * len(mask)
    elif change == 'boolean':
        mask = (True,) + mask[1:]
    elif change == 'row':
        row = row[:-1] + (1,)
    else:
        cases.pop()
    if change != 'missing':
        cases[1] = (name, text, row, mask)
    monkeypatch.setattr(current, 'SCIENCE_SERVICE_CASES', tuple(cases))
    with pytest.raises(ValueError):
        current._build_payload(predecessor)


def test_existing_input_prefix_or_duplicate_identity_is_rejected(predecessor):
    for item in (
        {'record_id': 'other-public-row', 'token_ids': list(current._GAS[:3])},
        {'record_id': 'planned-short-science-gas-active', 'token_ids': [41]},
    ):
        changed = copy.deepcopy(predecessor)
        changed['records'].append(item)
        with pytest.raises(ValueError):
            current._build_payload(changed)
