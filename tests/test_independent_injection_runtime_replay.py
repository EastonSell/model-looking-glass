"""Fabricated evidence only; no pretrained model, secret site or held-out row."""
import base64
import hashlib
import json
import struct

import pytest

from tools import audit_injection_runtime_replay as audit


@pytest.fixture(scope='module')
def saved_replay(tmp_path_factory):
    import torch
    from looking_glass.pretrained_number_adapter import to_selector_observation
    from looking_glass.pretrained_number_archive import _CONDITIONS, serialize_discovery_observation
    from looking_glass.pretrained_number_commitments import seal_site, verify_site_reveal
    from looking_glass.pretrained_number_injection import NumberInjectionExecutor
    from looking_glass.pretrained_number_requests import create_family_request_plan
    from looking_glass.pretrained_number_supervisor import _request_arguments
    from looking_glass.pretrained_number_witness import encode_private_forward
    from tests.test_pretrained_number_injection import ToyModel

    source, entropy, nonce = 'a' * 40, bytes(range(32)), bytes(range(32, 64))
    plan = create_family_request_plan(entropy, source)
    plan_sha = hashlib.sha256(plan).hexdigest()
    sealed = seal_site(nonce, source)
    site = verify_site_reveal(nonce, sealed.record_bytes, sealed.commitment_sha256, expected_source_commit=source)
    requests = {row['role']: row for row in json.loads(plan)['requests'] if row['case_index'] == 0}
    runtime, actual_runtime, model_hash = 'b' * 64, 'c' * 64, 'd' * 64
    binding = audit.independent.canonical({'schema': 'mlg-number-private-journal-binding-v1',
        'source_commit': source, 'plan_sha256': plan_sha, 'runtime_sha256': actual_runtime,
        'protocol_sha256': audit.independent.PROTOCOL_SHA256, 'checkpoint_revision': audit.independent.CHECKPOINT_REVISION,
        'site_commitment': sealed.commitment_sha256})
    binding_hash = hashlib.sha256(b'MLG-number-journal-binding-v1\0' + binding).digest()
    root = tmp_path_factory.mktemp('independent-replay')
    journal = root / 'journal'
    journal.mkdir()
    model, clean, injected, pairs, references = ToyModel(), None, None, [], []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with NumberInjectionExecutor(model, site=site) as executor:
            for role in audit.independent.ROLES:
                row = requests[role]
                observation = executor.forward(audit.IDS, request_id=row['request_id'], condition=_CONDITIONS[role],
                                               **_request_arguments(role, (clean, injected)))
                if role == 'clean':
                    clean = observation
                elif role == 'injected':
                    injected = observation
                selected = to_selector_observation(observation)
                public = serialize_discovery_observation(0, role, selected, execution_device='cpu')
                actions = tuple(item for item in executor.private_evidence() if item.request_id == row['request_id'])
                private = encode_private_forward(0, role, selected, actions)
                references.append(public)
                pairs.append((public, private))
    finally:
        torch.set_num_threads(old_threads)
    assert model.calls == 16
    fixture = {'root': root, 'plan': plan, 'plan_sha': plan_sha, 'source': source, 'nonce': nonce,
               'site_record': sealed.record_bytes, 'site_commitment': sealed.commitment_sha256,
               'runtime': runtime, 'actual_runtime': actual_runtime, 'model_hash': model_hash,
               'binding': binding_hash, 'references': tuple(references), 'pairs': pairs}
    _write_pairs(fixture, pairs)
    terminal = {'schema': 'mlg-injection-runtime-replay-owner-v1',
        'status': 'replay-produced-private-awaiting-independent-audit',
        'discovery_runtime_sha256': runtime, 'heldout_runtime_sha256': actual_runtime,
        'model_state_sha256': model_hash, 'forward_count': 16,
        'logical_limit_bytes': 8 * audit.MIB, 'physical_limit_bytes': 16 * audit.MIB,
        'comparisons': [{'within_tolerance': True, 'bit_equal': True} for _ in range(16)]}
    (root / 'owner-terminal.json').write_bytes(audit.independent.canonical(terminal))
    return fixture


def _write_pairs(fixture, pairs):
    previous, receipts = bytes(32), []
    for ordinal, (public, private) in enumerate(pairs):
        header = struct.pack('>8sHII32s32s', b'MLGNJ001', ordinal, len(public), len(private),
                             fixture['binding'], previous)
        digest = hashlib.sha256(b'MLG-number-private-pair-v1\0' + header + public + private).digest()
        frame = header + public + private + digest + b'MLGNEND1'
        ack = struct.pack('>8sH32s', b'MLGNA001', ordinal, digest)
        (fixture['root'] / 'journal' / f'{ordinal:03d}.frame').write_bytes(frame)
        (fixture['root'] / 'journal' / f'{ordinal:03d}.ok').write_bytes(ack)
        receipts.append(audit.independent.canonical({'ordinal': ordinal, 'frame_sha256': digest.hex(),
                         'public_sha256': hashlib.sha256(public).hexdigest()}) + b'\n')
        previous = digest
    (fixture['root'] / 'receipts.jsonl').write_bytes(b''.join(receipts))


def _configure(monkeypatch, value):
    monkeypatch.setattr(audit, 'SOURCE', value['source'])
    monkeypatch.setattr(audit, 'PLAN_SHA256', value['plan_sha'])
    monkeypatch.setattr(audit, 'DISCOVERY_RUNTIME_SHA256', value['runtime'])
    monkeypatch.setattr(audit, 'SITE_COMMITMENT', value['site_commitment'])
    monkeypatch.setattr(audit, 'REFERENCES', tuple(hashlib.sha256(raw).hexdigest() for raw in value['references']))
    return {'attempt_dir': value['root'], 'plan_bytes': value['plan'], 'site_nonce': value['nonce'],
            'site_record_bytes': value['site_record'], 'references': value['references'],
            'actual_runtime_sha256': value['actual_runtime'], 'expected_model_state_sha256': value['model_hash'],
            'externally_quiescent': True}


def test_saved_replay_is_checked_without_claiming_process_success(saved_replay, monkeypatch):
    kwargs = _configure(monkeypatch, saved_replay)
    result = audit.audit_saved_replay(**kwargs)
    assert result['status'] == 'saved-evidence-consistent' and result['forward_count'] == 16
    assert result['discovery_runtime_sha256'] != result['actual_runtime_sha256']
    assert 'layer' not in result and 'comparisons' not in result and 'nonce' not in result
    kwargs['externally_quiescent'] = False
    with pytest.raises(ValueError):
        audit.audit_saved_replay(**kwargs)


def test_replay_rejects_wrong_references_site_and_extra_evidence(saved_replay, monkeypatch):
    kwargs = _configure(monkeypatch, saved_replay)
    bad = dict(kwargs, references=(b'bad',) + kwargs['references'][1:])
    with pytest.raises(ValueError):
        audit.audit_saved_replay(**bad)
    with pytest.raises(ValueError):
        audit.audit_saved_replay(**dict(kwargs, site_nonce=b'x' * 32))
    extra = saved_replay['root'] / 'journal' / '016.frame'
    extra.write_bytes(b'')
    try:
        with pytest.raises(ValueError):
            audit.audit_saved_replay(**kwargs)
    finally:
        extra.unlink()


def test_rechained_private_landing_tamper_is_rejected(saved_replay, monkeypatch):
    kwargs = _configure(monkeypatch, saved_replay)
    pairs = list(saved_replay['pairs'])
    private = json.loads(pairs[2][1])
    action = private['actions'][0]
    after = bytearray(base64.b64decode(action['after']))
    after[0] ^= 1  # An untouched, non-target token, with fully recomputed framing.
    action['after'] = base64.b64encode(after).decode('ascii')
    pairs[2] = (pairs[2][0], audit.independent.canonical(private))
    try:
        _write_pairs(saved_replay, pairs)
        with pytest.raises(ValueError):
            audit.audit_saved_replay(**kwargs)
    finally:
        _write_pairs(saved_replay, saved_replay['pairs'])


def test_rechained_logit_change_fails_the_independent_tolerance(saved_replay, monkeypatch):
    kwargs = _configure(monkeypatch, saved_replay)
    pairs = list(saved_replay['pairs'])
    record = json.loads(pairs[-1][0])
    logits = bytearray(base64.b64decode(record['payload']['logits']))
    struct.pack_into('<f', logits, 0, 1e20)
    record['payload']['logits'] = base64.b64encode(logits).decode('ascii')
    pairs[-1] = (audit.independent.canonical(record), pairs[-1][1])
    try:
        _write_pairs(saved_replay, pairs)
        with pytest.raises(ValueError):
            audit.audit_saved_replay(**kwargs)
    finally:
        _write_pairs(saved_replay, saved_replay['pairs'])
