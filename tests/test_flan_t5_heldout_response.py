"""Fabricated observation and capacity checks; no model or held-out cases."""
import hashlib
import json
from pathlib import Path

import pytest

from looking_glass import flan_t5_heldout_response as subject


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_fixed_sixteen_token_bound_reserves_custody_and_next_forward():
    bound = subject.fixed_evidence_bound()
    assert bound['valid_tensor_bytes'] == 363257856
    assert bound['total_bytes'] == 490852352
    assert bound['total_bytes'] + bound['next_forward_reserve'] < bound['run_limit_bytes']
    assert bound['run_limit_bytes'] + bound['custody_limit_bytes'] == 512 * 1024**2


def test_public_card_observation_binds_kind_source_and_exact_card(tmp_path):
    card = b'{"commitment":"fabricated"}'
    source = {'commit': 'a' * 40}
    value = {'schema': 'mlg-flant5-heldout-card-observation-v1', 'kind': 'pool',
             'commit': 'b' * 40,
             'commit_url': 'https://github.com/EastonSell/model-looking-glass/commit/' + 'b' * 40,
             'card_path': 'commitments/pool.json',
             'card_sha256': hashlib.sha256(card).hexdigest(), 'source_commit': source['commit'],
             'observer': 'fabricated', 'observed_utc': 'fabricated'}
    path = tmp_path / 'observation.json'
    write_json(path, value)
    assert subject.check_card_observation(path, 'pool', card, source)
    for kind, data in [('cases', card), ('pool', card + b'changed')]:
        with pytest.raises(ValueError, match='observation'):
            subject.check_card_observation(path, kind, data, source)
    value['source_commit'] = 'c' * 40
    write_json(path, value)
    with pytest.raises(ValueError, match='observation'):
        subject.check_card_observation(path, 'pool', card, source)


def test_auditor_observation_requires_digest_and_bound_source(tmp_path):
    source = {'commit': 'a' * 40}
    value = {'schema': 'mlg-flant5-heldout-auditor-observation-v1', 'commit': 'b' * 40,
             'commit_url': 'https://github.com/EastonSell/model-looking-glass/commit/' + 'b' * 40,
             'auditor_sha256': 'c' * 64, 'source_commit': source['commit'],
             'auditor_path': 'reviews/fabricated.py', 'observer': 'fabricated', 'observed_utc': 'fabricated'}
    path = tmp_path / 'auditor.json'
    write_json(path, value)
    assert subject.check_auditor_observation(path, source)
    value['auditor_sha256'] = 'not-a-digest'
    write_json(path, value)
    with pytest.raises(ValueError, match='auditor'):
        subject.check_auditor_observation(path, source)


def test_observer_paths_must_be_normalized_repository_paths():
    assert subject._repo_path('reviews/auditor.py', '.py')
    for path in ('../secret.py', '/absolute.py', 'reviews\\file.py', 'reviews/../secret.py', 'reviews//bad.py'):
        assert not subject._repo_path(path, '.py')


def test_evidence_limits_json_before_writing(tmp_path):
    evidence = subject.Evidence(tmp_path / 'attempt', limit=subject.RUN_LIMIT)
    with pytest.raises(ValueError, match='per-file'):
        evidence.json('forward-000.json', {'value': 'x' * 65536})
    assert not (evidence.path / 'forward-000.json').exists()
    evidence.json('completed.json', {'value': 'x' * 65536}, reserve=0)
    assert (evidence.path / 'completed.json').stat().st_size > 65536


def test_malformed_track_cannot_produce_a_complete_terminal():
    with pytest.raises(ValueError, match='tracks are incomplete'):
        subject.complete_track_decisions([])


def test_custody_identity_must_match_committed_source_protocol_and_assets():
    source = {'commit': 'a' * 40, 'source_sha256': {'docs/FLAN_T5_HELDOUT_RESPONSE_V1.md': 'b' * 64}}
    value = {'identity': {'source_commit': source['commit'], 'protocol_sha256': 'b' * 64,
                          'asset_manifest_sha256': subject.MANIFEST_HASH, 'tokenizer_revision': subject.REVISION},
             'cases': [{} for _ in range(12)], 'custody_bytes': 123}
    subject.check_custody_identity(value, source)
    for key in value['identity']:
        bad = {**value, 'identity': {**value['identity'], key: '0' * len(value['identity'][key])}}
        with pytest.raises(ValueError, match='custody identity'):
            subject.check_custody_identity(bad, source)
