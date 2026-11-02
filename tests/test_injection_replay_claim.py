"""Fabricated one-use claim tests; no production ledger or private data."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import injection_replay_claim as claim


def _setup(tmp_path):
    workspace = tmp_path / 'workspace'
    work = workspace / 'work'
    parent = work / 'number-injection-heldout-cpu-v1'
    rehearsal = work / 'injection-heldout-runtime-rehearsal-v1'
    ledger = work / 'agreement-study-ledger-v5' / 'ledger.sqlite3'
    parent.mkdir(parents=True)
    rehearsal.mkdir()
    (rehearsal / 'diagnostic.txt').write_bytes(b'no model')
    ledger.parent.mkdir()
    ledger.write_bytes(b'fabricated ledger')
    plan = b'{"schema":"fabricated"}'
    calls = []
    def claim_phase(path, phase, *, output_path):
        calls.append((path, phase, output_path))
        return SimpleNamespace(phase=phase, output_path=str(output_path),
                               logical_limit_bytes=72 * 1024**2, disk_limit_bytes=192 * 1024**2)
    def claim_row(_, phase):
        return ['injection_heldout', str(phase), 72 * 1024**2, 192 * 1024**2, 13 * 1024**3]
    identity = lambda p: (p.lstat().st_dev, p.lstat().st_ino)
    kwargs = {'workspace': workspace, 'plan_bytes': plan, 'plan_sha256': hashlib.sha256(plan).hexdigest(),
              'parent_identity': identity(parent), 'ledger_identity': identity(ledger),
              'verify_authority': lambda: True, 'verify_claim_row': claim_row, 'claim_phase': claim_phase}
    return parent, calls, kwargs


def test_single_claim_is_bound_to_fixed_paths_and_cannot_repeat(tmp_path):
    parent, calls, kwargs = _setup(tmp_path)
    receipt = claim.claim_replay_once(**kwargs)
    assert len(calls) == 1 and calls[0][1] == 'injection_heldout'
    assert receipt['plan_sha256'] == kwargs['plan_sha256']
    assert receipt['phase_logical_bytes'] == 72 * 1024**2
    assert receipt['pilot_physical_bytes'] == 16 * 1024**2
    assert receipt['diagnostic_reserved_bytes'] == 4 * 1024**2
    assert receipt['diagnostic_actual_bytes'] == 8
    row = kwargs['verify_claim_row'](None, Path(receipt['phase_path']))
    encoded = json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    assert receipt['injection_claim_row_sha256'] == hashlib.sha256(encoded).hexdigest()
    assert json.loads((parent / 'replay-claim-receipt.json').read_bytes()) == receipt
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1


def test_ambiguous_claim_keeps_intent_and_forbids_retry(tmp_path):
    parent, calls, kwargs = _setup(tmp_path)
    kwargs['verify_claim_row'] = lambda *_: False
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1 and (parent / 'replay-claim-intent.json').is_file()
    assert not (parent / 'replay-claim-receipt.json').exists()
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1


def test_mismatched_ledger_or_existing_pilot_prevents_claim(tmp_path):
    parent, calls, kwargs = _setup(tmp_path)
    kwargs['ledger_identity'] = (1, 1)
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert not calls and not (parent / 'replay-claim-intent.json').exists()
    kwargs['ledger_identity'] = (kwargs['workspace'] / 'work' / 'agreement-study-ledger-v5' / 'ledger.sqlite3').stat().st_dev, \
                                (kwargs['workspace'] / 'work' / 'agreement-study-ledger-v5' / 'ledger.sqlite3').stat().st_ino
    (parent / 'runtime-compatibility-replay-v1').mkdir()
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert not calls


def test_changed_same_size_diagnostic_invalidates_claim_handoff(tmp_path):
    parent, calls, kwargs = _setup(tmp_path)
    diagnostic = kwargs['workspace'] / 'work' / 'injection-heldout-runtime-rehearsal-v1' / 'diagnostic.txt'
    original_row = kwargs['verify_claim_row']
    def change_and_verify(*args):
        diagnostic.write_bytes(b'changed!')
        return original_row(*args)
    kwargs['verify_claim_row'] = change_and_verify
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1 and (parent / 'replay-claim-intent.json').exists()
    assert not (parent / 'replay-claim-receipt.json').exists()


def test_corrupted_receipt_fails_readback_without_reclaim(tmp_path, monkeypatch):
    parent, calls, kwargs = _setup(tmp_path)
    original = claim._write_once
    def write_then_corrupt(path, raw):
        original(path, raw)
        if path.name == 'replay-claim-receipt.json':
            path.write_bytes(b'corrupted')
    monkeypatch.setattr(claim, '_write_once', write_then_corrupt)
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1 and (parent / 'replay-claim-intent.json').exists()
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1


def test_corrupted_intent_blocks_phase_claim(tmp_path, monkeypatch):
    parent, calls, kwargs = _setup(tmp_path)
    original = claim._write_once
    def corrupt_intent(path, raw):
        original(path, raw)
        if path.name == 'replay-claim-intent.json':
            path.write_bytes(b'corrupted')
    monkeypatch.setattr(claim, '_write_once', corrupt_intent)
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert not calls and (parent / 'replay-claim-intent.json').exists()
    assert not (parent / 'replay-claim-receipt.json').exists()


def test_intent_changed_after_claim_blocks_receipt(tmp_path):
    parent, calls, kwargs = _setup(tmp_path)
    original_row = kwargs['verify_claim_row']
    def alter_intent(*args):
        (parent / 'replay-claim-intent.json').write_bytes(b'changed')
        return original_row(*args)
    kwargs['verify_claim_row'] = alter_intent
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1 and not (parent / 'replay-claim-receipt.json').exists()


def test_changed_claim_row_blocks_receipt(tmp_path):
    parent, calls, kwargs = _setup(tmp_path)
    original = kwargs['verify_claim_row']
    checks = []
    def changing_row(*args):
        checks.append(1)
        row = original(*args)
        if len(checks) == 2:
            row[-1] += 1
        return row
    kwargs['verify_claim_row'] = changing_row
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1 and not (parent / 'replay-claim-receipt.json').exists()


def test_spoofed_phase_type_blocks_receipt(tmp_path):
    parent, calls, kwargs = _setup(tmp_path)
    original = kwargs['verify_claim_row']
    class SpoofedPhase(str):
        def __ne__(self, other):
            return False
    def spoof(*args):
        row = original(*args)
        row[0] = SpoofedPhase('wrong_phase')
        return row
    kwargs['verify_claim_row'] = spoof
    with pytest.raises(ValueError):
        claim.claim_replay_once(**kwargs)
    assert len(calls) == 1 and not (parent / 'replay-claim-receipt.json').exists()
