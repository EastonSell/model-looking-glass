"""Audit only saved public evidence and deterministic case status; no model."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from looking_glass import bert_agreement_cases as v1
from looking_glass import bert_agreement_exposures_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '224d683291b5b4f3a617697c8f9811351cf63bef19958fff9870c943e88fcdbc'


def test_snapshot_preserves_v1_rows_and_reports_cited_exposures():
    raw = (ROOT / 'experiments/bert_agreement_exposures_v2.json').read_bytes()
    assert raw == v2.canonical_snapshot_bytes()
    assert hashlib.sha256(raw).hexdigest() == v2.snapshot_digest() == EXPECTED
    original = v1.build_manifest()
    result = json.loads(raw)
    assert result['predecessor'] == {'schema': v1.SCHEMA, 'sha256': v1.MANIFEST_SHA256}
    assert result['identity'] == original['identity']
    assert len(result['rows']) == len(original['rows']) == 96
    for old, new in zip(original['rows'], result['rows']):
        assert (new['row_id'], new['split'], new['input_ids'], new['attention_mask'], new['v1_exposure_status']) == (
            old['row_id'], old['split'], old['input_ids'], old['attention_mask'], old['exposure_status'])
    counts = Counter((row['split'], row['status']) for row in result['rows'])
    assert counts == {('development', 'observed-in-cited-evidence'): 10,
                      ('development', 'reserved-development'): 22,
                      ('evaluation', 'public-evaluation-not-observed-in-cited-evidence'): 64}
    assert {row['row_id'] for row in result['rows'] if 'gpu-calibration' in row['observed_in']} == v2.GPU_IDS
    assert {row['row_id'] for row in result['rows'] if 'cpu-family1-incomplete' in row['observed_in']} == v2.FAMILY_ONE_IDS
    assert {row['row_id'] for row in result['rows'] if 'cpu-family1-retry' in row['observed_in']} == v2.FAMILY_ONE_IDS
    check = v2.verify_snapshot_bytes(raw)
    assert (check.observed_development_rows, check.reserved_development_rows,
            check.public_evaluation_rows, check.exact_development_evaluation_overlap) == (10, 22, 64, 0)
    assert not check.complete_history_certified and not check.execution_authorized
    with pytest.raises(ValueError):
        v2.verify_snapshot_bytes(raw + b' ')
    with mock.patch.object(v1, 'MANIFEST_SHA256', '0' * 64), pytest.raises(ValueError):
        v2.snapshot_digest()


def test_saved_cpu_journal_inputs_match_all_eight_manifest_rows():
    payload = v2.build_snapshot()
    evidence = {item['id']: item for item in payload['evidence']}
    for item in evidence.values():
        result = (ROOT / item['result_path']).read_bytes()
        assert hashlib.sha256(result).hexdigest() == item['result_sha256']
        if 'run_id' in item:
            record = json.loads(result)
            assert (record['run_id'], record['status'], record['source_commit']) == (
                item['run_id'], item['status'], item['source_commit'])
    pilot_texts = {row['text'] for row in v1.build_manifest()['rows'] if tuple(row['input_ids']) in v1.PILOT_ROWS}
    for label in ('pilot-v0', 'pilot-v0-replay'):
        record = json.loads((ROOT / evidence[label]['result_path']).read_bytes())
        assert {row['text'] for row in record['rows']} == pilot_texts
    journal = gzip.decompress((ROOT / evidence['cpu-family1-incomplete']['journal_path']).read_bytes())
    assert hashlib.sha256(journal).hexdigest() == v2.JOURNAL_SHA256
    assert evidence['cpu-family1-retry']['journal_uncompressed_sha256'] == v2.JOURNAL_SHA256
    events = [json.loads(line) for line in journal.splitlines()]
    started = {row['forward_index']: row for row in events if row['status'] == 'started'}
    completed = [row for row in events if row['status'] == 'completed']
    assert len(started) == len(completed) == 140
    assert {row['row_id'] for row in completed} == v2.FAMILY_ONE_IDS
    original = {row['row_id']: row for row in v1.build_manifest()['rows']}
    for row in completed:
        case = original[row['row_id']]
        assert case['split'] == 'development'
        assert row['model_input_ids'] == case['input_ids']
        assert row['model_attention_mask'] == case['attention_mask']
        assert started[row['forward_index']]['row_id'] == row['row_id']


def test_gpu_calibration_only_reuses_two_cited_development_rows():
    result = json.loads((ROOT / 'results/bert-gpu-calibration-v1-20261029/result.json').read_bytes())
    original = v1.build_manifest()['rows']
    ids = (result['measurement']['target_ids'], result['measurement']['donor_ids'])
    assert {row['row_id'] for row in original if row['input_ids'] in ids} == v2.GPU_IDS
    assert all(tuple(row['input_ids']) in v1.PILOT_ROWS for row in original if row['row_id'] in v2.GPU_IDS)
