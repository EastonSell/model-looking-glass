"""Fabricated-only preparation; no pretrained model or old split is used."""
import hashlib
import json

import pytest

from looking_glass import agreement_split_gate as old_gate
from looking_glass import natural_agreement_successor as sampler
from looking_glass import natural_successor_preparation as preparation
from looking_glass import natural_successor_split_state as state
from test_natural_successor_execution import fabricated
from test_natural_successor_split_state import roots


@pytest.fixture
def inputs(fabricated, monkeypatch):
    values = fabricated
    monkeypatch.setattr(state, 'POOL', values['pins']['pool_sha256'])
    monkeypatch.setattr(state, 'INPUTS', values['input_sha256'])
    monkeypatch.setattr(state, 'PUBLIC_CLEARANCE', values['pins']['pool_clear_sha256'])
    monkeypatch.setattr(state, 'OLD_COMMITMENT', values['pins']['old_commitment'])
    monkeypatch.setattr(state, 'PROTOCOL', values['pins']['protocol_sha256'])
    monkeypatch.setattr(state, 'FREEZE', values['freeze'].sha256)
    monkeypatch.setattr(preparation, 'TOKENIZER', values['tokenizer_sha256'])
    monkeypatch.setattr(preparation, 'COMPARATOR', values['comparator_sha256'])
    monkeypatch.setattr(preparation, '_nonce', lambda: bytes(range(32)))
    return {'operation_sha256': '1' * 64, 'pool_bytes': values['pool_bytes'],
            'input_bytes': values['input_bytes'], 'clearance_bytes': values['pool_clearance_bytes'],
            'inventory_bytes': values['sealed_snapshot'].record_bytes,
            'inventory_sha256': values['sealed_snapshot'].sha256, 'tokenizer': values['tokenizer'],
            'check_before_claim': lambda digest: True, 'check_before_draw': lambda digest: True}


def test_one_preparation_binds_public_commitment_without_private_output(roots, inputs):
    gate, root = roots
    checks = []
    def before_claim(digest):
        assert not (root / 'phase-claim.json').exists()
        checks.append(('claim', digest))
        return True
    def before_draw(digest):
        assert (root / 'phase-claim.json').is_file()
        assert (root / 'natural-successor-allocation-v1.json').is_file()
        assert not (root / 'nonce.bin').exists()
        checks.append(('draw', digest))
        return True
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        result = preparation.prepare(gate=locked, root=root,
                                     **dict(inputs, check_before_claim=before_claim, check_before_draw=before_draw))
        assert [x[0] for x in checks] == ['claim', 'draw']
        assert result['status'] == 'prepared_unpublished'
        public = (root / 'commitment.json').read_bytes()
        assert result['commitment_file_sha256'] == hashlib.sha256(public).hexdigest()
        assert not ({'nonce', 'record', 'cases', 'record_sha256'} & set(result))
        pins = {'pool_sha256': state.POOL, 'protocol_sha256': state.PROTOCOL,
                'exposure_sha256': inputs['inventory_sha256'], 'old_commitment': state.OLD_COMMITMENT,
                'pool_clear_sha256': state.PUBLIC_CLEARANCE}
        record = sampler.verify_record(inputs['pool_bytes'], (root / 'nonce.bin').read_bytes(),
            inputs['tokenizer'], (root / 'heldout-record.json').read_bytes(), result['heldout_commitment'], **pins)
        assert len(record['families']) == 4
        assert state.announce_commitment(locked, root, public,
            publication_check=lambda value: value == public)['status'] == 'committed_comparison_required'
        with pytest.raises((ValueError, RuntimeError)):
            preparation.prepare(gate=locked, root=root, **inputs)


def test_failed_preclaim_neither_claims_nor_draws(roots, inputs, monkeypatch):
    gate, root = roots
    monkeypatch.setattr(preparation, '_nonce', lambda: pytest.fail('nonce must not be drawn'))
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises(preparation.PreparationError):
            preparation.prepare(gate=locked, root=root,
                                **dict(inputs, check_before_claim=lambda digest: False))
        assert list(root.iterdir()) == []


def test_lost_gate_after_preclaim_check_stops_before_claim(roots, inputs, monkeypatch):
    gate, root = roots
    monkeypatch.setattr(preparation, '_nonce', lambda: pytest.fail('nonce must not be drawn'))
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        def lose_gate(digest):
            locked.lease.release()
            return True
        with pytest.raises((ValueError, RuntimeError)):
            preparation.prepare(gate=locked, root=root, **dict(inputs, check_before_claim=lose_gate))
        assert list(root.iterdir()) == []


def test_failed_predraw_consumes_claim_without_nonce_or_retry(roots, inputs, monkeypatch):
    gate, root = roots
    monkeypatch.setattr(preparation, '_nonce', lambda: pytest.fail('nonce must not be drawn'))
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises(preparation.PreparationError):
            preparation.prepare(gate=locked, root=root,
                                **dict(inputs, check_before_draw=lambda digest: False))
        assert (root / 'phase-claim.json').is_file()
        assert (root / 'natural-successor-allocation-v1.json').is_file()
        assert not (root / 'nonce.bin').exists()
        with pytest.raises((ValueError, RuntimeError)):
            preparation.prepare(gate=locked, root=root, **inputs)


def test_interrupted_entropy_leaves_consumed_phase_and_empty_nonce(roots, inputs, monkeypatch):
    gate, root = roots
    def interrupted():
        raise OSError('fabricated entropy failure')
    monkeypatch.setattr(preparation, '_nonce', interrupted)
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises(OSError):
            preparation.prepare(gate=locked, root=root, **inputs)
        assert (root / 'nonce.bin').read_bytes() == b''
        assert (root / 'phase-claim.json').is_file()
        assert not (root / 'heldout-record.json').exists()
        with pytest.raises((ValueError, RuntimeError)):
            preparation.prepare(gate=locked, root=root, **inputs)


def test_lost_gate_after_predraw_check_stops_before_nonce(roots, inputs, monkeypatch):
    gate, root = roots
    monkeypatch.setattr(preparation, '_nonce', lambda: pytest.fail('nonce must not be drawn'))
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        def lose_gate(digest):
            locked.lease.release()
            return True
        with pytest.raises((ValueError, RuntimeError)):
            preparation.prepare(gate=locked, root=root, **dict(inputs, check_before_draw=lose_gate))
        assert (root / 'phase-claim.json').is_file()
        assert (root / 'natural-successor-allocation-v1.json').is_file()
        assert not (root / 'nonce.bin').exists()


def test_changed_public_bytes_fail_before_claim(roots, inputs, monkeypatch):
    gate, root = roots
    monkeypatch.setattr(preparation, '_nonce', lambda: pytest.fail('nonce must not be drawn'))
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises(preparation.PreparationError):
            preparation.prepare(gate=locked, root=root, **dict(inputs, input_bytes=inputs['input_bytes'] + b' '))
        assert list(root.iterdir()) == []


def test_known_public_input_collision_fails_before_claim(roots, inputs, monkeypatch):
    gate, root = roots
    monkeypatch.setattr(preparation, '_nonce', lambda: pytest.fail('nonce must not be drawn'))
    inventory = json.loads(inputs['inventory_bytes'])
    row = json.loads(inputs['input_bytes'])['cases'][0]['input_ids']
    inventory['records'].append({'record_id': 'fabricated-collision', 'category': 'observed', 'token_ids': row})
    raw = json.dumps(inventory, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises(ValueError, match='prefix'):
            preparation.prepare(gate=locked, root=root, **dict(inputs, inventory_bytes=raw,
                inventory_sha256=hashlib.sha256(raw).hexdigest()))
        assert list(root.iterdir()) == []


def test_interruption_after_nonce_cannot_draw_again(roots, inputs, monkeypatch):
    gate, root = roots
    original = preparation._write_new
    def interrupted(path, raw):
        if path.name == 'heldout-record.json':
            raise OSError('fabricated interrupted record write')
        return original(path, raw)
    monkeypatch.setattr(preparation, '_write_new', interrupted)
    with old_gate.acquire(gate, expected_root=gate, timeout=2) as locked:
        with pytest.raises(OSError):
            preparation.prepare(gate=locked, root=root, **inputs)
        assert len((root / 'nonce.bin').read_bytes()) == 32
        assert (root / 'phase-claim.json').is_file()
        with pytest.raises((ValueError, RuntimeError)):
            preparation.prepare(gate=locked, root=root, **inputs)
