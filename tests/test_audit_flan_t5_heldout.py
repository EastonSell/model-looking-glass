"""Small, fabricated-only checks for the independent saved-evidence auditor."""
import copy
import hashlib
import json
from pathlib import Path

import pytest


AUDITOR = Path(__file__).resolve().parents[1] / 'tools' / 'audit_flan_t5_heldout.py'
SOURCE = AUDITOR.read_text(encoding='utf-8')
DEFINITIONS = SOURCE.split('# BEGIN HELDOUT EVIDENCE STARTUP', 1)[0]
NS = {'__name__': '__synthetic_test__', '__file__': str(AUDITOR)}
exec(compile(DEFINITIONS, str(AUDITOR), 'exec'), NS)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def _pool():
    identity = {'source_commit': 'a' * 40, 'protocol_sha256': 'b' * 64,
                'asset_manifest_sha256': 'c' * 64,
                'tokenizer_revision': '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'}
    hashes = {path: hashlib.sha256(path.encode()).hexdigest() for path in NS['EXPOSURE_ORDER']}
    pool = {'schema': 'mlg-flant5-heldout-pool-v1', **identity,
            'exposure_inputs': [{'path': path, 'commit': identity['source_commit'], 'sha256': hashes[path]}
                                for path in NS['EXPOSURE_ORDER']],
            'templates': [{'template_id': 'template0', 'prefix': 'Name ', 'suffix': '?'},
                          {'template_id': 'template1', 'prefix': 'Translate ', 'suffix': '.'}],
            'pairs': [{'candidate_index': i, 'source_words': [f'a{i}', f'b{i}'],
                       'target_words': [f'c{i}', f'd{i}']} for i in range(24)]}
    return pool, identity, hashes


def _card(pool, identity, salt):
    raw = _canonical(pool)
    digest = hashlib.sha256(b'mlg-flant5-heldout-pool-v1\x00' + salt + raw).hexdigest()
    return raw, {'schema': 'mlg-flant5-heldout-commitment-v1', 'kind': 'pool',
                 'commitment': digest, **identity}


def test_three_outcome_tracks_do_not_promote_gate_failures():
    gate = NS['baseline_gate']([2., -2.], 3.)
    assert gate['raw_supported'] and not gate['strict_supported']
    assert NS['track_statuses'](gate, 'positive') == {
        'status': 'baseline-unsupported', 'raw_status': 'positive', 'descriptive_status': 'positive'}
    gate = NS['baseline_gate']([4., 2.], 3.)
    assert NS['track_statuses'](gate, 'dose-unmatchable') == {
        'status': 'baseline-unsupported', 'raw_status': 'baseline-unsupported',
        'descriptive_status': 'dose-unmatchable'}
    cells = [{'status': value} for value in (['positive'] * 6 + ['baseline-unsupported'] * 6)]
    assert NS['track_aggregate'](cells, 'status')['status'] == 'positive'
    cells[0]['status'] = 'semantic-ineligible'
    assert NS['track_aggregate'](cells, 'status')['status'] == 'inconclusive'


def test_commitment_and_previously_public_template_are_checked_independently():
    pool, identity, hashes = _pool()
    salt = b'x' * 32
    raw, card = _card(pool, identity, salt)
    assert NS['verify_card'](raw, salt, card, 'pool') == pool
    NS['verify_pool_shape'](pool, hashes)
    with pytest.raises(ValueError):
        NS['verify_card'](raw + b' ', salt, card, 'pool')
    altered = copy.deepcopy(pool)
    altered['templates'][1] = {'template_id': 'template1',
                               'prefix': 'WHAT IS THE GERMAN WORD FOR ', 'suffix': '?'}
    raw, card = _card(altered, identity, b'y' * 32)
    assert NS['verify_card'](raw, b'y' * 32, card, 'pool') == altered
    with pytest.raises(ValueError):
        NS['verify_pool_shape'](altered, hashes)


def test_clean_owner_receipt_requires_terminal_and_exit_observations(tmp_path):
    (tmp_path / 'owner-bytecode-prefix').mkdir()
    (tmp_path / 'owner-parent-bytecode-prefix').mkdir()
    (tmp_path / 'owner.stdout.bin').write_bytes(b'ok')
    (tmp_path / 'owner.stderr.bin').write_bytes(b'')
    terminal = b'fake terminal'
    inventory = {'bytes': 2, 'files': 2, 'entries': [
        {'path': 'owner.stderr.bin', 'size': 0}, {'path': 'owner.stdout.bin', 'size': 2}]}
    owner = {'schema': 'mlg-flant5-heldout-owner-v1', 'status': 'complete', 'child_exit_code': 0,
             'terminal_sha256': hashlib.sha256(terminal).hexdigest(), 'final_custody_inventory': inventory,
             'streams_exceeded_bound': False, 'reader_threads_finished': True, 'bytecode_prefix_empty': True,
             'owner_bytecode_prefix_empty': True, 'source_unchanged': True, 'timed_out': False,
             'owner_timeout_seconds': 3600, 'stop': {'reaped': True, 'terminate_attempted': False,
                                                     'kill_attempted': False, 'error_type': None},
             'error_type': None, 'terminal_error_type': None,
             'stream_states': [{'observed_bytes': 2}, {'observed_bytes': 0}]}
    NS['verify_complete_owner'](owner, tmp_path, inventory, terminal)
    owner['timed_out'] = True
    with pytest.raises(AssertionError):
        NS['verify_complete_owner'](owner, tmp_path, inventory, terminal)
