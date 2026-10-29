"""No-model checks for a distinct repeat of already observed public rows."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from looking_glass import bert_agreement_exposures_v2 as v2
from looking_glass import bert_agreement_exposures_v3 as v3


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '6945a478e3af37083af055bf20bf65628fa979e0ee8b253ea8bacb475cb9e9f2'


def test_predecessor_and_eight_repeat_annotations_are_preserved():
    raw = (ROOT / 'experiments/bert_agreement_exposures_v3.json').read_bytes()
    assert raw == v3.canonical_snapshot_bytes()
    assert hashlib.sha256(raw).hexdigest() == v3.snapshot_digest() == EXPECTED
    old, new = v2.build_snapshot(), json.loads(raw)
    assert new['predecessor'] == {'schema': v2.SCHEMA, 'sha256': v2.SNAPSHOT_SHA256}
    assert new['identity'] == old['identity']
    assert new['evidence'][:-1] == old['evidence']
    assert new['evidence'][-1]['id'] == v3.EVIDENCE_ID
    assert new['evidence'][-1]['reference_commit'] == v3.EVIDENCE_COMMIT
    assert new['evidence'][-1]['journal_uncompressed_sha256'] == v2.JOURNAL_SHA256
    changed = set()
    for before, after in zip(old['rows'], new['rows']):
        expected = dict(before)
        if before['row_id'] in v2.FAMILY_ONE_IDS:
            changed.add(before['row_id'])
            expected['observed_in'] = before['observed_in'] + [v3.EVIDENCE_ID]
        assert after == expected
    assert changed == v2.FAMILY_ONE_IDS
    counts = Counter((row['split'], row['status']) for row in new['rows'])
    assert counts == {('development', 'observed-in-cited-evidence'): 10,
                      ('development', 'reserved-development'): 22,
                      ('evaluation', 'public-evaluation-not-observed-in-cited-evidence'): 64}
    check = v3.verify_snapshot_bytes(raw)
    assert (check.observed_development_rows, check.reserved_development_rows,
            check.public_evaluation_rows, check.exact_development_evaluation_overlap) == (10, 22, 64, 0)
    assert not check.complete_history_certified and not check.execution_authorized
    with pytest.raises(ValueError):
        v3.verify_snapshot_bytes(raw + b' ')
    with mock.patch.object(v2, 'SNAPSHOT_SHA256', '0' * 64), pytest.raises(ValueError):
        v3.snapshot_digest()


def test_identical_prior_journal_covers_the_eight_cited_rows():
    evidence = v3.build_snapshot()['evidence'][-1]
    journal = gzip.decompress((ROOT / evidence['journal_path']).read_bytes())
    assert hashlib.sha256(journal).hexdigest() == evidence['journal_uncompressed_sha256']
    events = [json.loads(line) for line in journal.splitlines()]
    started = [row for row in events if row['status'] == 'started']
    completed = [row for row in events if row['status'] == 'completed']
    assert len(started) == len(completed) == 140
    assert {row['row_id'] for row in completed} == v2.FAMILY_ONE_IDS


def test_published_repeat_records_bind_the_same_journal_and_run():
    evidence = v3.build_snapshot()['evidence'][-1]
    result_raw = (ROOT / evidence['result_path']).read_bytes()
    inventory_raw = (ROOT / evidence['inventory_path']).read_bytes()
    exit_raw = (ROOT / evidence['controller_exit_path']).read_bytes()
    assert hashlib.sha256(result_raw).hexdigest() == evidence['result_sha256']
    assert hashlib.sha256(inventory_raw).hexdigest() == evidence['inventory_sha256']
    assert hashlib.sha256(exit_raw).hexdigest() == evidence['controller_exit_sha256']
    result, inventory, exit_record = map(json.loads, (result_raw, inventory_raw, exit_raw))
    assert (result['status'], result['run_id'], result['source_commit']) == (
        'complete', v3.RUN_ID, v3.SOURCE_COMMIT)
    assert (inventory['run_id'], inventory['source_commit']) == (v3.RUN_ID, v3.SOURCE_COMMIT)
    assert result['journal_sha256'] == inventory['journal']['uncompressed_sha256'] == v2.JOURNAL_SHA256
    assert inventory['journal']['shared_repository_path'] == evidence['journal_path']
    assert exit_record == {'schema': 'bert-agreement-local-controller-exit-v1', 'exit_code': 0}
