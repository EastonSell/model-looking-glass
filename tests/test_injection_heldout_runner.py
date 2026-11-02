"""Fabricated-only producer, evidence and failure tests; no pretrained model."""
import json
import copy

import pytest
import torch

from looking_glass import injection_heldout as heldout
from looking_glass.distilgpt2_intervention_dev import _model_state_sha256
from looking_glass.pretrained_number_incomplete import decode_incomplete_forward
from tests.test_injection_heldout_codec import fake_frozen_inputs
from tests.test_pretrained_number_injection import ToyModel


def prepared_binding(monkeypatch):
    prepared = heldout.verify_frozen_inputs(**fake_frozen_inputs(monkeypatch))
    binding = heldout.make_prepared_binding(prepared, operation_sha256='2' * 64, gate_sha256='3' * 64,
                                            disjoint_status='pending', disjoint_receipt_sha256=None)
    return prepared, binding


def test_real_toy_producer_records_all_receipts_and_quiescent_journal(tmp_path, monkeypatch):
    assert hasattr(heldout, 'run_injection_pairs'), 'held-out producer is missing'
    prepared, binding = prepared_binding(monkeypatch)
    model = ToyModel()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    (tmp_path / 'owner').mkdir()
    try:
        result = heldout.run_injection_pairs(model=model, prepared=prepared, binding=binding,
                                             attempt_dir=tmp_path / 'owner' / 'attempt',
                                             expected_model_state_sha256=_model_state_sha256(model),
                                             logical_limit_bytes=72 * 1024 * 1024,
                                             disk_limit_bytes=192 * 1024 * 1024, reserved_side_bytes=0)
    finally:
        torch.set_num_threads(old_threads)
    assert model.calls == 128
    assert result.status == 'owner-produced'
    attempt = tmp_path / 'owner' / 'attempt'
    receipts = heldout.read_pair_receipts(attempt / 'receipts.jsonl')
    assert len(receipts) == 128
    codec = heldout.bind_heldout_codec(tuple(target.input_ids for target in prepared.targets))
    inspection = heldout.inspect_heldout_journal(attempt / 'journal', codec=codec,
        binding=binding, site=prepared.site, max_disk_bytes=192 * 1024 * 1024,
        acknowledgments=receipts, externally_quiescent=True)
    assert len(inspection.observations) == 8
    assert all(len(row) == 16 for row in inspection.observations)
    assert inspection.frame_sha256 == receipts[-1].frame_sha256
    terminal = json.loads((attempt / 'owner-terminal.json').read_bytes())
    assert terminal['status'] == 'owner-produced' and terminal['forward_count'] == 128
    assert 'site' not in terminal
    with pytest.raises(ValueError):
        heldout.inspect_heldout_journal(attempt / 'journal', codec=codec, binding=binding,
            site=prepared.site, max_disk_bytes=192 * 1024 * 1024,
            acknowledgments=receipts[:-1], externally_quiescent=True)
    assert hasattr(heldout, 'audit_produced_pairs') and hasattr(heldout, 'score_supervised_pairs')
    audit = heldout.audit_produced_pairs(attempt, prepared=prepared, binding=binding,
                                        expected_model_state_sha256=_model_state_sha256(model))
    command = ['fabricated-python', 'fabricated-child']
    supervisor = {'status': 'evidence_complete', 'exit_code': 0, 'child_reaped': True, 'child_pid': 12,
                  'job_drained': True, 'job_closed': True, 'logs_complete': True, 'postexit_active_processes': 0,
                  'cleanup_verified': True, 'retry_safe': False, 'cancellation_requested': False,
                  'quarantined': False, 'log_overflow': False, 'supervisor_error': None,
                  'command': command, 'audit': audit, 'retained_tensor_files': {'accounting_complete': True,
                                                                                 'total_bytes': 0, 'files': []}}
    for name in ('stdout', 'stderr'):
        (tmp_path / 'owner' / f'{name}.txt').write_bytes(b'')
        supervisor[f'{name}_sha256'] = __import__('hashlib').sha256(b'').hexdigest()
    supervisor_path = tmp_path / 'owner' / 'supervisor.json'
    supervisor_path.write_text(json.dumps(dict(supervisor, exit_code=1)), encoding='utf-8')
    with pytest.raises(ValueError):
        heldout.score_supervised_pairs(tmp_path / 'owner', prepared=prepared, binding=binding,
            expected_command=command, expected_model_state_sha256=_model_state_sha256(model))
    altered = copy.deepcopy(supervisor)
    altered['audit']['forward_count'] = 128.0
    supervisor_path.write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError):
        heldout.score_supervised_pairs(tmp_path / 'owner', prepared=prepared, binding=binding,
            expected_command=command, expected_model_state_sha256=_model_state_sha256(model))
    supervisor_path.write_text(json.dumps(supervisor), encoding='utf-8')
    (tmp_path / 'owner' / 'unexpected.bin').write_bytes(b'extra')
    with pytest.raises(ValueError):
        heldout.score_supervised_pairs(tmp_path / 'owner', prepared=prepared, binding=binding,
            expected_command=command, expected_model_state_sha256=_model_state_sha256(model))
    (tmp_path / 'owner' / 'unexpected.bin').unlink()
    score = json.loads(heldout.score_supervised_pairs(tmp_path / 'owner', prepared=prepared, binding=binding,
        expected_command=command, expected_model_state_sha256=_model_state_sha256(model)))
    assert score['status'] == 'complete' and set(score['sites']) == {'1', '2', '3', '4'}
    assert score['prediction_sha256'] == heldout.INJECTION_PREDICTION_SHA256
    assert (tmp_path / 'owner' / 'binding.json').read_bytes() == binding.record_bytes
    saved = (tmp_path / 'owner' / 'evaluation.json').read_bytes()
    receipt = json.loads((tmp_path / 'owner' / 'post-score-receipt.json').read_bytes())
    assert json.loads(saved) == score
    assert receipt['status'] == 'scored-private-awaiting-independent-audit'
    assert receipt['disjoint_status'] == 'pending'
    assert receipt['evaluation_sha256'] == __import__('hashlib').sha256(saved).hexdigest()


def test_serializer_failure_preserves_completed_toy_forward(tmp_path, monkeypatch):
    assert hasattr(heldout, 'run_injection_pairs'), 'held-out producer is missing'
    prepared, binding = prepared_binding(monkeypatch)
    model = ToyModel()
    def fail_serialization(*args, **kwargs):
        raise ValueError('synthetic private detail')
    monkeypatch.setattr(heldout.HeldoutCodec, 'serialize_public', fail_serialization)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(ValueError):
            heldout.run_injection_pairs(model=model, prepared=prepared, binding=binding,
                attempt_dir=tmp_path / 'failed', expected_model_state_sha256=_model_state_sha256(model),
                logical_limit_bytes=72 * 1024 * 1024, disk_limit_bytes=192 * 1024 * 1024,
                reserved_side_bytes=0)
    finally:
        torch.set_num_threads(old_threads)
    assert model.calls == 1
    partial = decode_incomplete_forward((tmp_path / 'failed' / 'incomplete-forward.bin').read_bytes())
    assert partial.manifest['observation'] is not None
    assert partial.manifest['tensor_bytes'] > 0
    assert not (tmp_path / 'failed' / 'owner-terminal.json').exists()
    failure = (tmp_path / 'failed' / 'failure.json').read_bytes()
    assert b'synthetic private detail' not in failure and b'"incomplete"' in failure


def test_partial_capture_survives_an_independent_evidence_retrieval_failure(tmp_path, monkeypatch):
    from looking_glass.pretrained_number_injection import NumberInjectionExecutor
    prepared, binding = prepared_binding(monkeypatch)
    model = ToyModel()
    monkeypatch.setattr(heldout.HeldoutCodec, 'serialize_public',
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('fabricated serialization failure')))
    monkeypatch.setattr(NumberInjectionExecutor, 'private_evidence',
                        lambda *args: (_ for _ in ()).throw(ValueError('fabricated retrieval failure')))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(ValueError):
            heldout.run_injection_pairs(model=model, prepared=prepared, binding=binding,
                attempt_dir=tmp_path / 'failed', expected_model_state_sha256=_model_state_sha256(model),
                logical_limit_bytes=72 * 1024 * 1024, disk_limit_bytes=192 * 1024 * 1024,
                reserved_side_bytes=0)
    finally:
        torch.set_num_threads(old_threads)
    assert model.calls == 1
    partial = decode_incomplete_forward((tmp_path / 'failed' / 'incomplete-forward.bin').read_bytes())
    assert partial.manifest['observation'] is not None and partial.manifest['tensor_bytes'] > 0
    failure = json.loads((tmp_path / 'failed' / 'failure.json').read_bytes())
    assert failure['partial_evidence'] == 'retained-with-omissions'
