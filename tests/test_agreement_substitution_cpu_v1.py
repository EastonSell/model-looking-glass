"""Model-free admission and exposure checks for the substitution entry."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('substitution_entry', ROOT / 'tools/agreement_substitution_cpu_v1.py')
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
CASE_PATH = ROOT / entry.CASE_PATH
SUPPORT_PATH = ROOT / entry.NATURAL_RECEIPT_PATH


def cases():
    return entry.verify_case_document(CASE_PATH.read_bytes())


def test_unpublished_reservation_fails_closed(monkeypatch):
    monkeypatch.setattr(entry, 'RESERVATION_REVISION', None)
    with pytest.raises(RuntimeError, match='reservation'):
        entry.validate_prelaunch({})


def test_published_case_bytes_and_row_digest_are_fixed():
    raw = CASE_PATH.read_bytes()
    assert len(entry.verify_case_document(raw)) == 54
    with pytest.raises(ValueError, match='case bytes'):
        entry.verify_case_document(raw + b' ')


def planned_records():
    return [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
             'case_role': row['role'], 'public_text': row['text'],
             'token_ids': row['token_ids'],
             'candidate_token_ids': [value for _, value in entry.ANSWERS]} for row in cases()]


def test_planned_rows_and_foreign_prefixes():
    rows = cases()
    records = planned_records()
    records.append({'record_id': 'other', 'token_ids': [50256, 50100],
                    'candidate_token_groups': {'group': [42, 43]}})
    entry.validate_exposure({'records': records}, rows)
    records[0]['candidate_token_ids'] = [318]
    with pytest.raises(ValueError, match='planned record'):
        entry.validate_exposure({'records': records}, rows)
    records = planned_records()
    records[0]['target_token_id'] = 42
    with pytest.raises(ValueError, match='planned record'):
        entry.validate_exposure({'records': records}, rows)
    records = planned_records()
    records.append({'record_id': 'collision', 'token_ids': rows[0]['token_ids'][:-1]})
    with pytest.raises(ValueError, match='collision'):
        entry.validate_exposure({'records': records}, rows)
    records = planned_records()
    records.append({'record_id': 'collision', 'token_ids': rows[0]['token_ids'] + [318, 5]})
    with pytest.raises(ValueError, match='collision'):
        entry.validate_exposure({'records': records}, rows)


def test_custody_receipts_are_bound_to_exact_cases(tmp_path):
    stamp = '2026-10-29T10:00:00Z'
    natural = {'schema': 'agreement-substitution-natural-refresh-v1', 'status': 'clear',
               'case_sha256': entry.CASE_SHA256, 'support_receipt_sha256': entry.NATURAL_RECEIPT_SHA256,
               'scope': entry.NATURAL_SCOPE, 'successor_support_sha256': entry.SUCCESSOR_SUPPORT_SHA256,
               'refreshed_utc': stamp, 'successor_status': 'undrawn'}
    injection = {'schema': 'agreement-substitution-injection-refresh-v1', 'status': 'no_protected_rows',
                 'direct_target_conflict': 'none reported', 'site': 'transformer.h.3.attn',
                 'refreshed_utc': stamp}
    paths = [tmp_path / 'natural.json', tmp_path / 'injection.json']
    data = [(json.dumps(value, sort_keys=True) + '\n').encode() for value in (natural, injection)]
    for path, raw in zip(paths, data):
        path.write_bytes(raw)
    record = {'custody': {'natural': {'observation_sha256': hashlib.sha256(data[0]).hexdigest(),
                                      'refreshed_utc': stamp},
                          'injection': {'observation_sha256': hashlib.sha256(data[1]).hexdigest(),
                                        'refreshed_utc': stamp}}}
    receipt, _, _ = entry.verify_custody(record, SUPPORT_PATH.resolve(), paths[0].resolve(), paths[1].resolve())
    assert hashlib.sha256(receipt).hexdigest() == entry.NATURAL_RECEIPT_SHA256
    natural['case_sha256'] = '0' * 64
    data[0] = (json.dumps(natural, sort_keys=True) + '\n').encode()
    paths[0].write_bytes(data[0])
    record['custody']['natural']['observation_sha256'] = hashlib.sha256(data[0]).hexdigest()
    with pytest.raises(ValueError, match='fixed fields'):
        entry.verify_custody(record, SUPPORT_PATH.resolve(), paths[0].resolve(), paths[1].resolve())


def test_same_volume_requires_combined_free_space(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    output = tmp_path / 'output'
    cache.mkdir()
    output.mkdir()
    minimum = entry.LIMITS['minimum_free_disk_bytes'] + entry.LIMITS['minimum_cache_free_bytes']
    monkeypatch.setattr(entry.shutil, 'disk_usage', lambda _: SimpleNamespace(free=minimum - 1))
    with pytest.raises(ValueError, match='combined'):
        entry.check_disk_headroom(cache, output)
    monkeypatch.setattr(entry.shutil, 'disk_usage', lambda _: SimpleNamespace(free=minimum))
    assert entry.check_disk_headroom(cache, output)['same_volume'] is True


def test_operator_inventory_rejects_rehashed_changed_identity(tmp_path):
    rows = cases()
    identity = {'model_id': entry.MODEL_ID, 'model_revision': entry.MODEL_REVISION,
                'tokenizer_id': entry.MODEL_ID, 'tokenizer_revision': entry.MODEL_REVISION,
                'tokenizer_sha256': entry.TOKENIZER_SHA256, 'weights_sha256': entry.WEIGHTS_SHA256}
    reconstructed = {'version': 'prior', 'identity': identity, 'records': planned_records()}
    payload = {'version': 'later', 'identity': dict(identity), 'records': planned_records()}
    canonical = lambda obj: json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                        ensure_ascii=True, allow_nan=False).encode('ascii')
    prior_raw = canonical(reconstructed)
    path, observation_path = tmp_path / 'inventory.json', tmp_path / 'observation.json'
    timestamp = '2026-10-29T10:00:00Z'
    def write_observation():
        raw = canonical(payload)
        exposure = {'latest_sha256': hashlib.sha256(prior_raw).hexdigest(),
                    'operator_sha256': hashlib.sha256(raw).hexdigest(),
                    'operator_revision': 'a' * 40, 'operator_version': 'later',
                    'operator_refreshed_utc': timestamp}
        observation = {'schema': 'agreement-substitution-exposure-refresh-v1', 'status': 'clear',
                       'case_sha256': entry.CASE_SHA256, 'reconstructed_sha256': exposure['latest_sha256'],
                       'operator_revision': exposure['operator_revision'], 'operator_version': 'later',
                       'operator_sha256': exposure['operator_sha256'], 'refreshed_utc': timestamp}
        observation_raw = canonical(observation)
        exposure['operator_observation_sha256'] = hashlib.sha256(observation_raw).hexdigest()
        path.write_bytes(raw)
        observation_path.write_bytes(observation_raw)
        return {'exposure': exposure}
    record = write_observation()
    entry.verify_operator_inventory(record, path.resolve(), observation_path.resolve(), prior_raw, rows)
    payload['identity']['model_id'] = 'another/model'
    record = write_observation()
    with pytest.raises(ValueError, match='identity'):
        entry.verify_operator_inventory(record, path.resolve(), observation_path.resolve(), prior_raw, rows)
