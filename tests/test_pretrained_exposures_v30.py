"""Model-free tests for the evidence-bound status-only exposure update."""
import copy
import hashlib

import pytest

from looking_glass import pretrained_exposures_v29 as prior
from looking_glass import pretrained_exposures_v30 as update


@pytest.fixture(scope='module')
def predecessor():
    payload = prior.inventory_payload()
    assert hashlib.sha256(prior._canonical(payload)).hexdigest() == update.PREDECESSOR_SHA256
    return payload


def test_only_the_evidenced_statuses_change(predecessor):
    result = update._build_payload(predecessor)
    assert hashlib.sha256(update._canonical(result)).hexdigest() == update.INVENTORY_SHA256
    assert len(result['records']) == len(predecessor['records']) == 915
    assert update._prefixes(result['records']) == update._prefixes(predecessor['records'])
    assert len(update._prefixes(result['records'])) == 5391
    assert result['update'] == {'observed_year_label_rows': 48, 'observed_full_top_target_rows': 36}
    year = {'planned-' + case[0] for case in update.YEAR_LABEL_CONTROL_CASES}
    targets = {'planned-' + case[0] for case in update.FULL_TOP_CASES if not case[0].startswith('full-top-locality-')}
    localities = {'planned-' + case[0] for case in update.FULL_TOP_CASES if case[0].startswith('full-top-locality-')}
    assert (len(year), len(targets), len(localities)) == (48, 36, 12)
    for old, new in zip(predecessor['records'], result['records']):
        if old['record_id'] not in year | targets:
            assert new == old
            continue
        assert new['record_id'] == old['record_id'].replace('planned-', 'observed-', 1)
        assert old['category'] == 'planned' and new['category'] == 'observed-saved'
        for key in set(old) - {'record_id', 'category', 'provenance', 'references', 'reference_commit'}:
            assert new[key] == old[key]
        assert set(old['references']).issubset(new['references'])
    assert all(r['category'] == 'planned' for r in result['records'] if r['record_id'] in localities)


def test_rejects_missing_duplicate_changed_or_already_observed_rows(predecessor):
    for kind in ('missing', 'duplicate', 'tokens', 'candidate', 'status', 'observed-id'):
        payload = copy.deepcopy(predecessor)
        records = payload['records']
        row = next(r for r in records if r['record_id'].startswith('planned-year-label-'))
        if kind == 'missing':
            records.remove(row)
        elif kind == 'duplicate':
            records.append(copy.deepcopy(row))
        elif kind == 'tokens':
            row['token_ids'][-1] += 1
        elif kind == 'candidate':
            row['candidate_token_groups']['above_ids'][0] += 1
        elif kind == 'status':
            row['category'] = 'observed-saved'
        else:
            extra = copy.deepcopy(row)
            extra['record_id'] = extra['record_id'].replace('planned-', 'observed-', 1)
            records.append(extra)
        with pytest.raises(ValueError):
            update._build_payload(payload)
    payload = copy.deepcopy(predecessor)
    locality = next(r for r in payload['records'] if r['record_id'].startswith('planned-full-top-locality-'))
    locality['category'] = 'observed-saved'
    with pytest.raises(ValueError):
        update._build_payload(payload)


def test_predecessor_digest_and_current_check_are_enforced(predecessor, monkeypatch):
    monkeypatch.setattr(prior, 'inventory_payload', lambda: copy.deepcopy(predecessor))
    payload, digest = update._checked_inventory()
    assert digest == update.INVENTORY_SHA256 and payload['version'] == update.INVENTORY_VERSION
    changed = copy.deepcopy(predecessor)
    changed['records'][0]['token_ids'] = [999]
    monkeypatch.setattr(prior, 'inventory_payload', lambda: changed)
    with pytest.raises(ValueError, match='predecessor'):
        update.inventory_payload()
