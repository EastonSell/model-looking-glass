import copy
import hashlib
import importlib

import pytest

from looking_glass import year_label_control as study


@pytest.fixture(scope='module')
def sources():
    old = importlib.import_module('looking_glass.pretrained_exposures_v24')
    new = importlib.import_module('looking_glass.pretrained_exposures_v25')
    return old, new, old.inventory_payload()


def test_successor_marks_only_twelve_localities_and_reserves_exact_rows(sources):
    old, new, predecessor = sources
    saved = copy.deepcopy(predecessor)
    payload = new._build_payload(predecessor)
    assert predecessor == saved
    assert hashlib.sha256(new._canonical(payload)).hexdigest() == new.INVENTORY_SHA256
    assert len(new._prefixes(payload['records'])) == new.FORBIDDEN_PREFIX_COUNT == 4190
    assert len(payload['records']) == len(predecessor['records']) + 48
    assert tuple(row['input_ids'] for row in study.make_inventory()['cases']) == tuple(map(list, new.YEAR_LABEL_CONTROL_ROWS))
    by_id = {row['record_id']: row for row in payload['records']}
    changed = set(new.LOCALITY_IDS)
    for record in predecessor['records']:
        if record['record_id'] in changed:
            observed = by_id['observed-' + record['record_id'][len('planned-'):]]
            assert observed['category'] == 'observed-saved'
            assert observed['token_ids'] == record['token_ids']
            assert observed['public_text'] == record['public_text']
        else:
            assert by_id[record['record_id']] == record
    for row in study.make_inventory()['cases']:
        planned = by_id['planned-' + row['case_id']]
        assert planned['category'] == 'planned' and planned['token_ids'] == row['input_ids']
    assert old._prefixes(predecessor['records']) <= old._prefixes(payload['records'])


def test_locality_duplicate_or_changed_status_fails_closed(sources):
    _, new, predecessor = sources
    bad = copy.deepcopy(predecessor)
    bad['records'].append(copy.deepcopy(next(r for r in bad['records'] if r['record_id'] == new.LOCALITY_IDS[0])))
    with pytest.raises(ValueError, match='locality'):
        new._build_payload(bad)
    bad = copy.deepcopy(predecessor)
    next(r for r in bad['records'] if r['record_id'] == new.LOCALITY_IDS[0])['category'] = 'observed-saved'
    with pytest.raises(ValueError, match='locality'):
        new._build_payload(bad)


def test_prior_relation_and_duplicate_new_id_are_rejected(sources):
    _, new, predecessor = sources
    bad = copy.deepcopy(predecessor)
    bad['records'].append({'record_id': 'conflicting-row', 'category': 'planned',
                           'token_ids': list(new.YEAR_LABEL_CONTROL_ROWS[0]), 'public_text': 'conflict'})
    with pytest.raises(ValueError, match='relation'):
        new._build_payload(bad)
    bad = copy.deepcopy(predecessor)
    bad['records'].append({'record_id': 'planned-' + new.YEAR_LABEL_CONTROL_CASES[0][0],
                           'category': 'planned', 'token_ids': [42], 'public_text': 'conflict'})
    with pytest.raises(ValueError, match='record ID'):
        new._build_payload(bad)
