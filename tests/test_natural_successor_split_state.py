"""Fabricated split markers and real cooperative locks; no private inputs."""
import hashlib
import json
import os

import pytest

from looking_glass import agreement_split_gate as old_gate
from looking_glass import natural_successor_claim as claim
from looking_glass import natural_successor_split_state as state


def path_hash(path):
    return hashlib.sha256(os.path.normcase(os.path.realpath(path)).encode('utf-8')).hexdigest()


@pytest.fixture
def roots(tmp_path, monkeypatch):
    gate = tmp_path / 'gate'
    root = tmp_path / 'successor'
    gate.mkdir()
    root.mkdir()
    (gate / 'gate.lock').touch()
    gs, ls, rs = gate.stat(), (gate / 'gate.lock').stat(), root.stat()
    monkeypatch.setattr(state, 'GATE_ROOT_SHA256', path_hash(gate))
    monkeypatch.setattr(state, 'GATE_ROOT_IDENTITY', (gs.st_dev, gs.st_ino))
    monkeypatch.setattr(state, 'GATE_LOCK_IDENTITY', (ls.st_dev, ls.st_ino))
    monkeypatch.setattr(state, 'SUCCESSOR_ROOT_SHA256', path_hash(root))
    monkeypatch.setattr(state, 'SUCCESSOR_ROOT_IDENTITY', (rs.st_dev, rs.st_ino))
    return gate, root


def new_claim(root):
    info = root.stat()
    return claim.claim_public_phase(root=root, root_path_sha256=path_hash(root),
        root_device=info.st_dev, root_inode=info.st_ino, operation_sha256='1' * 64,
        public_clearance_sha256=state.PUBLIC_CLEARANCE, inventory_sha256='2' * 64,
        amendment_sha256=state.AMENDMENT)


def public_commitment():
    value = {'schema': 'mlg-natural-successor-public-commitment-v1',
        'prediction_freeze_sha256': state.FREEZE, 'pool_sha256': state.POOL,
        'inputs_sha256': state.INPUTS, 'pool_clearance_sha256': state.PUBLIC_CLEARANCE,
        'exposure_inventory_sha256': '2' * 64, 'old_natural_commitment': state.OLD_COMMITMENT,
        'protocol_sha256': state.PROTOCOL, 'amendment_sha256': state.AMENDMENT,
        'operation_sha256': '1' * 64, 'heldout_commitment': '3' * 64}
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii')


def committed(locked, root):
    new_claim(root)
    allocation = state.announce_allocation(locked, root)
    assert allocation['status'] == 'allocation_pending_unchecked'
    return state.announce_commitment(locked, root, public_commitment(),
                                     publication_check=lambda raw: raw == public_commitment())


def test_absent_claim_allocation_commitment_and_forward_are_distinct(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        assert state.inspect(locked, root) == {'schema': 'mlg-natural-successor-state-disposition-v1',
                                               'status': 'not_allocated_unchecked'}
        digest = new_claim(root)
        with pytest.raises(state.SuccessorStateError):
            state.inspect(locked, root)
        allocation = state.announce_allocation(locked, root)
        assert allocation['status'] == 'allocation_pending_unchecked'
        assert allocation['phase_claim_sha256'] == digest
        with pytest.raises(state.SuccessorStateError):
            state.announce_allocation(locked, root)
        result = state.announce_commitment(locked, root, public_commitment(),
                                            publication_check=lambda raw: True)
        assert result['status'] == 'committed_comparison_required'
        assert result['commitment_file_sha256'] == hashlib.sha256(public_commitment()).hexdigest()
        result = state.announce_forward(locked, root)
        assert result['status'] == 'possibly_exposed_comparison_required'
        assert result['forward_peer_disposition'] == 'injection_not_allocated_unchecked'
        with pytest.raises(state.SuccessorStateError):
            state.announce_forward(locked, root)
    with pytest.raises((ValueError, RuntimeError)):
        state.inspect(locked, root)


@pytest.mark.parametrize('orphan', ['nonce.bin', 'heldout-record.json', 'commitment.json',
                                  'owner-started.json', 'pool-query-claim.json',
                                  'natural-successor-commitment-v1.json'])
def test_orphan_or_legacy_query_never_looks_absent(roots, orphan):
    gate, root = roots
    (root / orphan).write_bytes(b'{}')
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises((ValueError, RuntimeError)):
            state.inspect(locked, root)


def test_commitment_requires_exact_bytes_and_external_publication_observation(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        new_claim(root)
        state.announce_allocation(locked, root)
        for bad in (public_commitment().replace(state.INPUTS.encode(), b'f' * 64),
                    public_commitment() + b' ', b'{}'):
            with pytest.raises(state.SuccessorStateError):
                state.announce_commitment(locked, root, bad, publication_check=lambda raw: True)
        with pytest.raises(state.SuccessorStateError):
            state.announce_commitment(locked, root, public_commitment(), publication_check=lambda raw: False)
        assert not (root / 'natural-successor-commitment-v1.json').exists()
        assert state.announce_commitment(locked, root, public_commitment(),
            publication_check=lambda raw: True)['status'] == 'committed_comparison_required'


def test_pending_injection_is_unchecked(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        committed(locked, root)
        locked.announce_allocation('injection', state.INJECTION_FREEZE)
        result = state.announce_forward(locked, root)
        assert result['forward_peer_disposition'] == 'injection_allocation_pending_unchecked'


def test_committed_injection_rejects_forward_without_comparator(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        committed(locked, root)
        locked.announce_allocation('injection', state.INJECTION_FREEZE)
        value = {'schema': 'mlg-number-split-commitment-observed-v1', 'track': 'injection',
                 'freeze_sha256': state.INJECTION_FREEZE, 'heldout_commitment': '5' * 64,
                 'commitment_file_sha256': '6' * 64, 'inventory_sha256': '7' * 64}
        (gate / 'injection-commitment.json').write_bytes(
            json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii'))
        with pytest.raises(state.SuccessorStateError):
            state.announce_forward(locked, root)
        assert not (root / 'natural-successor-forward-v1.json').exists()


def test_unknown_injection_freeze_is_not_pending(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        committed(locked, root)
        locked.announce_allocation('injection', '4' * 64)
        with pytest.raises(state.SuccessorStateError):
            state.announce_forward(locked, root)
        assert not (root / 'natural-successor-forward-v1.json').exists()


def test_owner_start_without_forward_is_never_reported_pending(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        new_claim(root)
        state.announce_allocation(locked, root)
        (root / 'owner-started.json').write_bytes(b'{}')
        with pytest.raises(state.SuccessorStateError):
            state.inspect(locked, root)


def test_uncommitted_owner_directory_is_never_reported_pending(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        new_claim(root)
        state.announce_allocation(locked, root)
        (root / 'owner').mkdir()
        with pytest.raises(state.SuccessorStateError):
            state.inspect(locked, root)


@pytest.mark.parametrize('size', [0, 1, 31, 33])
def test_partial_nonce_metadata_is_never_reported_pending(roots, size):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        new_claim(root)
        state.announce_allocation(locked, root)
        (root / 'nonce.bin').write_bytes(b'x' * size)
        with pytest.raises(state.SuccessorStateError, match='nonce file metadata'):
            state.inspect(locked, root)


@pytest.mark.parametrize('unknown', ['natural-successor-allocation-v2.json',
                                    'pool-query-unknown.json', 'phase-claim.tmp',
                                    'nonce.bin.partial', 'owner-started-v2.json'])
def test_unknown_marker_namespace_is_never_reported_absent(roots, unknown):
    gate, root = roots
    (root / unknown).write_bytes(b'{}')
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises(state.SuccessorStateError):
            state.inspect(locked, root)


def test_interruption_after_marker_write_keeps_the_one_use_allocation(roots, monkeypatch):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        new_claim(root)
        original = state._read
        def interrupted(path):
            if path.name == state._ALLOC and path.exists():
                raise KeyboardInterrupt('fabricated post-write interruption')
            return original(path)
        with monkeypatch.context() as patcher:
            patcher.setattr(state, '_read', interrupted)
            with pytest.raises(KeyboardInterrupt):
                state.announce_allocation(locked, root)
        assert state.inspect(locked, root)['status'] == 'allocation_pending_unchecked'
        with pytest.raises(state.SuccessorStateError):
            state.announce_allocation(locked, root)


def test_tampered_or_partial_marker_never_authorizes(roots):
    gate, root = roots
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        committed(locked, root)
        (root / 'natural-successor-commitment-v1.json').write_bytes(b'{')
        with pytest.raises(state.SuccessorStateError):
            state.inspect(locked, root)
        with pytest.raises(state.SuccessorStateError):
            state.announce_forward(locked, root)
        assert not (root / 'natural-successor-forward-v1.json').exists()
