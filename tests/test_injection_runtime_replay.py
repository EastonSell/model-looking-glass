"""Fabricated CPU comparisons; no pretrained model or private study input."""
import base64
import json
import struct
import hashlib
from pathlib import Path
import sys

import pytest
import torch

from looking_glass import injection_runtime_replay as replay
from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
from looking_glass.pretrained_number_adapter import to_selector_observation
from looking_glass.pretrained_number_archive import serialize_discovery_observation
from looking_glass.pretrained_number_archive import _CONDITIONS, _ROLES
from looking_glass.pretrained_number_commitments import seal_site, verify_site_reveal
from looking_glass.pretrained_number_injection import NumberInjectionExecutor
from looking_glass.pretrained_number_requests import create_family_request_plan
from looking_glass.pretrained_number_supervisor import _request_arguments
from looking_glass.pretrained_number_witness import encode_private_forward
from looking_glass.distilgpt2_intervention_dev import _model_state_sha256
from tests.test_pretrained_number_injection import ToyModel


@pytest.fixture
def public_clean():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with NumberInjectionExecutor(ToyModel(), site=1) as executor:
            observation = executor.forward(PUBLIC_DISCOVERY[0][1], request_id='a' * 32, condition='clean')
            return serialize_discovery_observation(0, 'clean', to_selector_observation(observation),
                                                   execution_device='cpu')
    finally:
        torch.set_num_threads(old_threads)


def _changed_logit(raw, value):
    record = json.loads(raw)
    vector = bytearray(base64.b64decode(record['payload']['logits']))
    struct.pack_into('<f', vector, 0, value)
    record['payload']['logits'] = base64.b64encode(vector).decode('ascii')
    return json.dumps(record, sort_keys=True, separators=(',', ':')).encode()


def test_public_pair_comparison_checks_every_tensor_with_fixed_tolerance(public_clean):
    assert replay.compare_public_pair('clean', public_clean, public_clean) == {'within_tolerance': True,
                                                                              'bit_equal': True}
    record = json.loads(public_clean)
    encoded = base64.b64decode(record['payload']['logits'])
    first, bits = struct.unpack_from('<f', encoded)[0], struct.unpack_from('<I', encoded)[0]
    nearby = struct.unpack('<f', struct.pack('<I', bits + 1))[0]
    close = _changed_logit(public_clean, nearby)
    result = replay.compare_public_pair('clean', public_clean, close)
    assert result['within_tolerance'] and not result['bit_equal']
    far = _changed_logit(public_clean, first + 100.0)
    assert not replay.compare_public_pair('clean', public_clean, far)['within_tolerance']


def test_public_pair_rejects_wrong_role_and_metadata(public_clean):
    with pytest.raises(ValueError):
        replay.compare_public_pair('repeat', public_clean, public_clean)
    record = json.loads(public_clean)
    record['request_id'] = 'b' * 32
    altered = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
    with pytest.raises(ValueError):
        replay.compare_public_pair('clean', public_clean, altered)


def test_fixed_case_journal_audits_all_16_real_toy_forward_roles(tmp_path):
    source, nonce = 'a' * 40, bytes(range(32, 64))
    site_record = seal_site(nonce, source)
    site = verify_site_reveal(nonce, site_record.record_bytes, site_record.commitment_sha256,
                              expected_source_commit=source)
    plan = create_family_request_plan(bytes(range(32)), source)
    plan_sha = hashlib.sha256(plan).hexdigest()
    requests = {row['role']: row for row in json.loads(plan)['requests'] if row['case_index'] == 0}
    assert len(requests) == 16
    model, clean, injected, receipts = ToyModel(), None, None, []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with replay.isolated_replay_journal() as frozen:
            kwargs = dict(plan_bytes=plan, expected_source_commit=source, expected_plan_sha256=plan_sha,
                          runtime_sha256='b' * 64, site_nonce=nonce, site_record_bytes=site_record.record_bytes,
                          site_commitment=site_record.commitment_sha256)
            journal = frozen.DiscoveryJournal(tmp_path / 'journal', **kwargs,
                                              max_disk_bytes=16 * 1024**2,
                                              max_retained_payload_bytes=8 * 1024**2)
            with NumberInjectionExecutor(model, site=site, max_study_bytes=8 * 1024**2) as executor:
                for role in _ROLES:
                    row = requests[role]
                    observation = executor.forward(PUBLIC_DISCOVERY[0][1], request_id=row['request_id'],
                        condition=_CONDITIONS[role], **_request_arguments(role, (clean, injected)))
                    if role == 'clean':
                        clean = observation
                    elif role == 'injected':
                        injected = observation
                    selected = to_selector_observation(observation)
                    public = serialize_discovery_observation(0, role, selected, execution_device='cpu')
                    actions = tuple(a for a in executor.private_evidence() if a.request_id == row['request_id'])
                    private = encode_private_forward(0, role, selected, actions)
                    receipts.append(journal.append(public, private))
            inspection = frozen.inspect_discovery_journal(tmp_path / 'journal', **kwargs,
                acknowledgments=tuple(receipts), max_read_bytes=16 * 1024**2, externally_quiescent=True)
            assert inspection.status == 'complete-prefix' and len(inspection.verified_pairs) == 16
            assert journal.reservation().payload_bytes <= 8 * 1024**2
            assert model.calls == 16
            with pytest.raises(RuntimeError):
                with replay.isolated_replay_journal():
                    pass
    finally:
        torch.set_num_threads(old_threads)
    assert '_mlg_injection_runtime_replay_journal' not in sys.modules


def _fabricated_replay_fixture(monkeypatch):
    source, nonce = 'a' * 40, bytes(range(32, 64))
    sealed = seal_site(nonce, source)
    site = verify_site_reveal(nonce, sealed.record_bytes, sealed.commitment_sha256, expected_source_commit=source)
    plan = create_family_request_plan(bytes(range(32)), source)
    requests = {row['role']: row for row in json.loads(plan)['requests'] if row['case_index'] == 0}
    model, clean, injected, references = ToyModel(), None, None, []
    with NumberInjectionExecutor(model, site=site) as executor:
        for role in _ROLES:
            row = requests[role]
            observation = executor.forward(PUBLIC_DISCOVERY[0][1], request_id=row['request_id'],
                condition=_CONDITIONS[role], **_request_arguments(role, (clean, injected)))
            if role == 'clean':
                clean = observation
            elif role == 'injected':
                injected = observation
            references.append(serialize_discovery_observation(0, role, to_selector_observation(observation),
                                                              execution_device='cpu'))
    model.calls = 0
    monkeypatch.setattr(replay, '_REFERENCE_SHA256', tuple(hashlib.sha256(raw).hexdigest() for raw in references))
    kwargs = dict(model=model, plan_bytes=plan, source_commit=source, plan_sha256=hashlib.sha256(plan).hexdigest(),
        discovery_runtime_sha256='b' * 64, heldout_runtime_sha256='c' * 64, site_nonce=nonce,
        site_record_bytes=sealed.record_bytes, site_commitment=sealed.commitment_sha256,
        references=tuple(references), expected_model_state_sha256=_model_state_sha256(model))
    return kwargs


def test_fabricated_replay_runs_16_and_preserves_private_journal(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = _fabricated_replay_fixture(monkeypatch)
        outcome = replay.run_public_replay(attempt_dir=tmp_path / 'attempt', **kwargs)
    finally:
        torch.set_num_threads(old_threads)
    assert kwargs['model'].calls == 16
    assert outcome['status'] == 'replay-produced-private-awaiting-independent-audit'
    assert outcome['discovery_runtime_sha256'] == 'b' * 64 and outcome['heldout_runtime_sha256'] == 'c' * 64
    assert len((tmp_path / 'attempt' / 'receipts.jsonl').read_bytes().splitlines()) == 16
    assert len(list((tmp_path / 'attempt' / 'journal').glob('*.frame'))) == 16
    assert b'site' not in (tmp_path / 'attempt' / 'owner-terminal.json').read_bytes()
    audit_kwargs = {key: value for key, value in kwargs.items() if key != 'model'}
    with pytest.raises(ValueError):
        replay.audit_public_replay(tmp_path / 'attempt', externally_quiescent=False, **audit_kwargs)
    audit = replay.audit_public_replay(tmp_path / 'attempt', externally_quiescent=True, **audit_kwargs)
    assert audit['status'] == 'saved-evidence-consistent' and audit['forward_count'] == 16
    receipt_path = tmp_path / 'attempt' / 'receipts.jsonl'
    original_receipts = receipt_path.read_bytes()
    receipt_path.write_bytes(b'x' * 20000)
    original_read_bytes = Path.read_bytes
    def refuse_unbounded_read(path):
        if path == receipt_path:
            raise AssertionError('oversized receipt must not be read without a bound')
        return original_read_bytes(path)
    with monkeypatch.context() as checked:
        checked.setattr(Path, 'read_bytes', refuse_unbounded_read)
        with pytest.raises(ValueError):
            replay.audit_public_replay(tmp_path / 'attempt', externally_quiescent=True, **audit_kwargs)
    receipt_path.write_bytes(original_receipts.splitlines(keepends=True)[0])
    with pytest.raises(ValueError):
        replay.audit_public_replay(tmp_path / 'attempt', externally_quiescent=True, **audit_kwargs)


def test_replay_rejects_changed_reference_before_forward(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = _fabricated_replay_fixture(monkeypatch)
        kwargs['references'] = (b'wrong',) + kwargs['references'][1:]
        with pytest.raises(ValueError):
            replay.run_public_replay(attempt_dir=tmp_path / 'attempt', **kwargs)
    finally:
        torch.set_num_threads(old_threads)
    assert kwargs['model'].calls == 0
    assert not (tmp_path / 'attempt').exists()


def test_failed_comparison_retains_acknowledged_pair_and_stops(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = _fabricated_replay_fixture(monkeypatch)
        altered = _changed_logit(kwargs['references'][0], 10000.0)
        kwargs['references'] = (altered,) + kwargs['references'][1:]
        monkeypatch.setattr(replay, '_REFERENCE_SHA256',
                            tuple(hashlib.sha256(raw).hexdigest() for raw in kwargs['references']))
        with pytest.raises(ValueError):
            replay.run_public_replay(attempt_dir=tmp_path / 'attempt', **kwargs)
    finally:
        torch.set_num_threads(old_threads)
    attempt = tmp_path / 'attempt'
    assert kwargs['model'].calls == 1
    assert (attempt / 'journal' / '000.frame').is_file()
    assert (attempt / 'journal' / '000.ok').is_file()
    assert len((attempt / 'receipts.jsonl').read_bytes().splitlines()) == 1
    assert (attempt / 'journal' / 'terminal.json').is_file()
    assert not (attempt / 'owner-terminal.json').exists()


def test_interrupted_forward_return_preserves_available_observation(tmp_path, monkeypatch):
    from looking_glass.pretrained_number_incomplete import decode_incomplete_forward
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = _fabricated_replay_fixture(monkeypatch)
        original = NumberInjectionExecutor.forward
        def fail_after_completion(self, *args, **kw):
            original(self, *args, **kw)
            raise ValueError('fabricated return interruption')
        monkeypatch.setattr(NumberInjectionExecutor, 'forward', fail_after_completion)
        with pytest.raises(ValueError):
            replay.run_public_replay(attempt_dir=tmp_path / 'attempt', **kwargs)
    finally:
        torch.set_num_threads(old_threads)
    attempt = tmp_path / 'attempt'
    assert kwargs['model'].calls == 1
    partial = next((attempt / 'journal').glob('*.partial'))
    with replay.isolated_replay_journal() as frozen:
        data = frozen._parse_partial(partial.read_bytes(), 0, frozen._bind(
            kwargs['plan_bytes'], kwargs['source_commit'], kwargs['plan_sha256'],
            kwargs['heldout_runtime_sha256'], kwargs['site_nonce'], kwargs['site_record_bytes'],
            kwargs['site_commitment'])[2], bytes(32))
    assert decode_incomplete_forward(data).manifest['observation'] is not None


def test_replay_budget_blocks_next_forward_before_admission(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = _fabricated_replay_fixture(monkeypatch)
        monkeypatch.setattr(NumberInjectionExecutor, 'retained_tensor_bytes', property(lambda _: 8 * 1024**2))
        with pytest.raises(ValueError):
            replay.run_public_replay(attempt_dir=tmp_path / 'attempt', **kwargs)
    finally:
        torch.set_num_threads(old_threads)
    assert kwargs['model'].calls == 0
    assert (tmp_path / 'attempt' / 'journal' / 'terminal.json').is_file()
    assert not list((tmp_path / 'attempt' / 'journal').glob('*.frame'))


def test_cancellation_after_ack_keeps_partial_out_of_next_slot(tmp_path, monkeypatch):
    from looking_glass.pretrained_number_incomplete import decode_incomplete_forward
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = _fabricated_replay_fixture(monkeypatch)
        original = NumberInjectionExecutor.private_evidence
        calls = []
        def interrupt_second(self):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return original(self)
        monkeypatch.setattr(NumberInjectionExecutor, 'private_evidence', interrupt_second)
        monkeypatch.setattr(replay, 'compare_public_pair',
                            lambda *args: (_ for _ in ()).throw(ValueError('fabricated comparison')))
        with pytest.raises(KeyboardInterrupt):
            replay.run_public_replay(attempt_dir=tmp_path / 'attempt', **kwargs)
    finally:
        torch.set_num_threads(old_threads)
    attempt = tmp_path / 'attempt'
    assert kwargs['model'].calls == 1
    assert json.loads((attempt / 'journal' / 'terminal.json').read_bytes())['status'] == 'cancelled'
    assert not (attempt / 'journal' / '001.partial').exists()
    assert decode_incomplete_forward((attempt / 'incomplete-forward.bin').read_bytes()).manifest['observation']
    assert len((attempt / 'receipts.jsonl').read_bytes().splitlines()) == 1

