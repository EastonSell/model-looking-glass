"""Fabricated-only checks for the independent held-out saved-evidence audit."""
import importlib
import importlib.util
import hashlib
import json
import math
import subprocess
import struct
import sys

import pytest


def auditor():
    spec = importlib.util.find_spec('tools.audit_injection_heldout')
    assert spec is not None, 'independent injection held-out auditor is missing'
    return importlib.import_module('tools.audit_injection_heldout')


def test_binary32_injection_rounding_handles_cancellation_and_large_values():
    module = auditor()
    direction = 0x3c93cd3a
    assert module.rounded_add_bits(0, direction) == direction
    assert module.rounded_add_bits(0xbc93cd3a, direction) == 0
    # The first two sums are exact midpoints: round to the even neighbor,
    # once downward and once upward. Negation must preserve that choice.
    assert module.rounded_add_bits(0x3d800000, direction) == 0x3da4f34e
    assert module.rounded_add_bits(0x3d800001, direction) == 0x3da4f350
    assert module.rounded_add_bits(0xbd800000, 0xbc93cd3a) == 0xbda4f34e
    assert module.rounded_add_bits(0xbd800001, 0xbc93cd3a) == 0xbda4f350
    assert module.rounded_add_bits(0x3dffffff, direction) == 0x3e1279a7
    assert module.rounded_add_bits(0xbc93cd39, direction) == 0x31000000
    assert module.rounded_add_bits(0xbc93cd3b, direction) == 0xb1000000
    assert module.rounded_add_bits(0x7f7fffff, direction) == 0x7f7fffff
    assert module.rounded_add_bits(0xff7fffff, direction) == 0xff7fffff
    with pytest.raises(ValueError):
        module.rounded_add_bits(0x7f800000, direction)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows CRT rounding control is unavailable')
def test_injection_rounding_is_exact_and_nonnearest_binary64_is_rejected():
    program = '''
import ctypes
from tools.audit_injection_heldout import rounded_add_bits, require_nearest_binary64
control = ctypes.CDLL("msvcrt")._controlfp
control.argtypes = (ctypes.c_uint, ctypes.c_uint)
control.restype = ctypes.c_uint
prior = control(0, 0)
try:
    for mode in (0, 0x100, 0x200, 0x300):
        control(mode, 0x300)
        if rounded_add_bits(0x3f800000, 0x3c93cd3a) != 0x3f824f35:
            raise AssertionError("binary32 result depends on host rounding")
        try:
            require_nearest_binary64()
        except ValueError:
            if mode == 0:
                raise
        else:
            if mode != 0:
                raise AssertionError("non-nearest binary64 mode was accepted")
finally:
    control(prior, 0x300)
'''
    child = subprocess.run([sys.executable, '-B', '-c', program], capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr


def test_private_patch_and_null_require_exact_selected_and_untouched_bits():
    module = auditor()
    before = struct.pack('<1536I', *([0x3f800000] * 1536))
    donor = struct.pack('<768I', *([0x40000000] * 768))
    after = before[:3072] + donor
    module.check_action_scope(before, after, token_count=2, operation='patch', donor=donor)
    module.check_action_scope(before, before, token_count=2, operation='null', donor=None)
    changed = bytearray(after)
    changed[0] ^= 1
    with pytest.raises(ValueError):
        module.check_action_scope(before, bytes(changed), token_count=2, operation='patch', donor=donor)
    changed = bytearray(after)
    changed[-1] ^= 1
    with pytest.raises(ValueError):
        module.check_action_scope(before, bytes(changed), token_count=2, operation='patch', donor=donor)
    with pytest.raises(ValueError):
        module.check_action_scope(before, after, token_count=2, operation='null', donor=None)


def test_private_injection_recomputes_addition_and_recorded_delta():
    module = auditor()
    before = struct.pack('<1536I', *([0] * 1536))
    bits = module.direction_bits()
    after = before[:3072] + struct.pack('<768I', *bits)
    delta = struct.pack('<768d', *(module.float32(value) for value in bits))
    module.check_action_scope(before, after, token_count=2, operation='add', donor=None, delta=delta)
    changed = bytearray(after)
    changed[-1] ^= 1
    with pytest.raises(ValueError):
        module.check_action_scope(before, bytes(changed), token_count=2, operation='add', donor=None, delta=delta)
    changed = bytearray(delta)
    changed[0] ^= 1
    with pytest.raises(ValueError):
        module.check_action_scope(before, after, token_count=2, operation='add', donor=None, delta=bytes(changed))


def test_family_predictions_and_zero_errors_include_negative_restoration():
    module = auditor()
    effects = (1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0)
    reverse = tuple(-2 * item for item in effects)
    output = module.summarize_site(0.5, effects, reverse, numerator=3.0, denominator=2.0)
    assert output['restoration']['value'] == -0.5
    assert output['mae'] == math.fsum(abs(x - 0.5) for x in effects) / 8
    assert output['zero_rmse'] == math.sqrt(math.fsum(x * x for x in effects) / 8)
    assert output['families'][1]['effects'] == [2.0, -2.0]
    assert output['reverse_effects'] == list(reverse)
    assert module.summarize_site(0.5, effects, reverse, numerator=1.0, denominator=1e-8)['restoration']['value'] is None


def test_frame_reader_checks_binding_chain_footer_and_receipt():
    module = auditor()
    public, private = b'fabricated public', b'fabricated private'
    binding, previous = bytes.fromhex('12' * 32), bytes.fromhex('34' * 32)
    header = struct.pack('>8sHII32s32s', b'MLGNJ001', 7, len(public), len(private), binding, previous)
    digest = hashlib.sha256(b'MLG-number-private-pair-v1\0' + header + public + private).digest()
    frame = header + public + private + digest + b'MLGNEND1'
    ack = struct.pack('>8sH32s', b'MLGNA001', 7, digest)
    receipt = {'ordinal': 7, 'frame_sha256': digest.hex(), 'public_sha256': hashlib.sha256(public).hexdigest()}
    assert module.read_pair(frame, ack, ordinal=7, binding=binding, previous=previous, receipt=receipt) == (public, private, digest)
    for bad_frame, bad_ack, bad_receipt in (
            (frame[:-1] + b'X', ack, receipt),
            (frame, ack[:-1] + b'X', receipt),
            (frame, ack, dict(receipt, public_sha256='0' * 64))):
        with pytest.raises(ValueError):
            module.read_pair(bad_frame, bad_ack, ordinal=7, binding=binding, previous=previous, receipt=bad_receipt)
    with pytest.raises(ValueError):
        module.read_pair(frame, ack, ordinal=7, binding=binding, previous=bytes(32), receipt=receipt)


def test_same_site_order_places_injection_before_replacement():
    module = auditor()
    events = module.expected_events(site=1, role='forward_1', token_count=3)
    assert [events[name]['order'] for name in ('inject_before', 'inject_add', 'inject_after',
                                                'action_before', 'action_patch', 'action_after')] == [3, 4, 5, 6, 7, 8]
    assert events['action_patch']['preceding_same_site_actions'] == [
        'inject_before', 'inject_add', 'inject_after', 'action_before']
    assert events['action_patch']['token_index'] == 2


def test_saved_witness_rejects_non_target_mutation_and_wrong_source():
    import torch
    from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
    from looking_glass.pretrained_number_adapter import to_selector_observation
    from looking_glass.pretrained_number_archive import serialize_discovery_observation
    from looking_glass.pretrained_number_injection import NumberInjectionExecutor
    from looking_glass.pretrained_number_witness import encode_private_forward
    from tests.test_pretrained_number_injection import ToyModel
    module = auditor()
    ids, site, commitment = PUBLIC_DISCOVERY[0][1], 2, 'a' * 64
    def request(role):
        ordinal = module.ROLES.index(role)
        return hashlib.sha256(b'MLG-injection-heldout-request-v1\0' + bytes.fromhex(commitment)
                              + ordinal.to_bytes(4, 'big')).digest()[:16].hex()
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with NumberInjectionExecutor(ToyModel(), site=site) as executor:
            clean = executor.forward(ids, request_id=request('clean'), condition='clean')
            injected = executor.forward(ids, request_id=request('injected'), condition='injected')
            patched = executor.forward(ids, request_id=request('forward_2'), condition='clean_to_injected',
                                       layer=2, donor=clean.mlp_donor(2))
            evidence = tuple(item for item in executor.private_evidence() if item.request_id == patched.request_id)
            public = [to_selector_observation(item) for item in (clean, injected, patched)]
            raw = [serialize_discovery_observation(0, role, value, execution_device='cpu')
                   for role, value in zip(('clean', 'injected', 'forward_2'), public)]
            private = encode_private_forward(0, 'forward_2', public[-1], evidence)
    finally:
        torch.set_num_threads(old)
    clean_data, injected_data, patched_data = [module.parse_public(data, index=0, role=role, ids=ids,
        commitment=commitment) for data, role in zip(raw, ('clean', 'injected', 'forward_2'))]
    module.verify_witness(private, index=0, role='forward_2', ids=ids, site=site,
                          observation=patched_data, clean=clean_data, injected=injected_data)
    bad = json.loads(private)
    action = next(item for item in bad['actions'] if item['operation'] == 'patch')
    action['donor_identity']['request_id'] = request('injected')
    altered = json.dumps(bad, sort_keys=True, separators=(',', ':')).encode()
    with pytest.raises(ValueError):
        module.verify_witness(altered, index=0, role='forward_2', ids=ids, site=site,
                              observation=patched_data, clean=clean_data, injected=injected_data)
    bad = json.loads(private)
    import base64
    action = next(item for item in bad['actions'] if item['operation'] == 'add')
    changed = bytearray(base64.b64decode(action['after']))
    changed[0] ^= 1
    action['after'] = base64.b64encode(changed).decode()
    altered = json.dumps(bad, sort_keys=True, separators=(',', ':')).encode()
    with pytest.raises(ValueError):
        module.verify_witness(altered, index=0, role='forward_2', ids=ids, site=site,
                              observation=patched_data, clean=clean_data, injected=injected_data)


@pytest.fixture(scope='module')
def synthetic_journal(tmp_path_factory):
    """Write a full synthetic schedule using only an untrained CPU toy model."""
    import torch
    from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
    from looking_glass.distilgpt2_intervention_dev import _model_state_sha256
    from looking_glass.pretrained_number_adapter import to_selector_observation
    from looking_glass.pretrained_number_archive import serialize_discovery_observation
    from looking_glass.pretrained_number_commitments import make_prediction_freeze, seal_site
    from looking_glass.pretrained_number_injection import NumberInjectionExecutor
    from looking_glass.pretrained_number_journal import _ACK, _ACK_MAGIC, _frame
    from looking_glass.pretrained_number_requests import family_request_schedule
    from looking_glass.pretrained_number_selection import EvaluationCase, freeze_discovery, score_evaluation
    from looking_glass.pretrained_number_supervisor import _request_arguments
    from looking_glass.pretrained_number_witness import encode_private_forward
    from tests.test_pretrained_number_injection import ToyModel
    from tests.test_pretrained_number_selection import fabricated_cases
    module = auditor()
    root = tmp_path_factory.mktemp('fabricated-injection-journal')
    journal = root / 'journal'
    journal.mkdir()
    source, runtime, inventory = 'd' * 40, 'e' * 64, 'f' * 64
    prediction = freeze_discovery(fabricated_cases(), private_controls_verified=True, execution_device='cpu')
    prediction_sha = hashlib.sha256(prediction).hexdigest()
    freeze = make_prediction_freeze({'injection': {'source_commit': source, 'runtime_sha256': runtime,
                                                     'prediction_sha256': prediction_sha}})
    site_nonce = bytes([9]) * 32
    site_record = seal_site(site_nonce, source)
    site = json.loads(site_record.record_bytes)['layer']
    split_nonce = bytes([8]) * 32
    families = []
    for family in range(4):
        cases = []
        for text, ids in PUBLIC_DISCOVERY[4 * family:4 * family + 4]:
            cases.append({'text': text, 'input_ids': list(ids), 'attention_mask': [1] * len(ids), 'subject_index': 1})
        families.append({'family_index': family, 'context': f'fabricated-{family}',
                         'subjects': ['a', 'b', 'c', 'd'], 'cases': cases})
    split = module.canonical({'schema': 'mlg-number-heldout-v1', 'protocol_sha256': module.PROTOCOL_SHA256,
        'checkpoint_revision': module.CHECKPOINT_REVISION, 'source_commits': {'injection': source},
        'prediction_freeze_sha256': freeze.sha256, 'exposure_inventory_sha256': inventory, 'families': families})
    split_commitment = hashlib.sha256(b'MLG-number-v1-heldout-commit\0' + split_nonce + b'\0' + split).hexdigest()
    binding = module.canonical({'schema': 'mlg-injection-heldout-journal-binding-v1', 'source_commit': source,
        'runtime_sha256': runtime, 'prediction_sha256': prediction_sha, 'freeze_sha256': freeze.sha256,
        'heldout_commitment': split_commitment, 'site_commitment': site_record.commitment_sha256,
        'inventory_sha256': inventory, 'operation_sha256': '1' * 64, 'gate_sha256': '2' * 64,
        'disjoint_status': 'pending', 'disjoint_receipt_sha256': None})
    binding_sha = hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding).hexdigest()
    previous, receipts, observations, baselines = bytes(32), [], {}, {}
    model = ToyModel()
    model_state = _model_state_sha256(model)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with NumberInjectionExecutor(model, site=site) as executor:
            for ordinal, (index, role) in enumerate(family_request_schedule()):
                ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]
                clean, injected = baselines.get(index, (None, None))
                result = executor.forward(ids, request_id=module.request_id(split_commitment, index, role),
                    condition=module.CONDITIONS[role], **_request_arguments(role, (clean, injected)))
                public_observation = to_selector_observation(result)
                evidence = tuple(item for item in executor.private_evidence() if item.request_id == result.request_id)
                public = serialize_discovery_observation(index, role, public_observation, execution_device='cpu')
                private = encode_private_forward(index, role, public_observation, evidence)
                parts, digest = _frame(ordinal, public, private, bytes.fromhex(binding_sha), previous)
                (journal / f'{ordinal:03d}.frame').write_bytes(b''.join(parts))
                (journal / f'{ordinal:03d}.ok').write_bytes(_ACK.pack(_ACK_MAGIC, ordinal, digest))
                receipts.append({'ordinal': ordinal, 'frame_sha256': digest.hex(),
                                 'public_sha256': hashlib.sha256(public).hexdigest()})
                observations[index, role] = public_observation
                if role == 'clean':
                    baselines[index] = (result, None)
                elif role == 'injected':
                    baselines[index] = (clean, result)
                elif role == 'verify_clean':
                    del baselines[index]
                previous = digest
    finally:
        torch.set_num_threads(old)
    cases = []
    for index in range(8):
        take = lambda role: observations[index, role]
        text, ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2]
        cases.append(EvaluationCase(index, take('clean'), take('repeat'), take('injected'),
            tuple(take(f'forward_{i}') for i in range(1, 5)), tuple(take(f'reverse_{i}') for i in range(1, 5)),
            take('null'), take('self'), take('upstream'), take('downstream'), take('verify_clean'),
            text=text, input_ids=ids))
    evaluation = score_evaluation(tuple(cases), discovery_bytes=prediction, prediction_sha256=prediction_sha,
        prediction_freeze_sha256=freeze.sha256, heldout_commitment_sha256=split_commitment,
        custody_verified=True, private_controls_verified=True, execution_device='cpu')
    receipts_raw = b''.join(module.canonical(receipt) + b'\n' for receipt in receipts)
    started = module.canonical({'schema': 'mlg-injection-heldout-owner-started-v1',
        'binding_sha256': binding_sha, 'heldout_commitment': split_commitment, 'forward_count_planned': 128})
    terminal = module.canonical({'schema': 'mlg-injection-heldout-owner-terminal-v1', 'status': 'owner-produced',
        'forward_count': 128, 'binding_sha256': binding_sha, 'receipts_sha256': hashlib.sha256(receipts_raw).hexdigest(),
        'last_frame_sha256': previous.hex(), 'model_state_sha256': model_state, 'retained_tensor_bytes': 0,
        'scientific_validation': 'pending independent audit'})
    expected = {'binding_sha256': binding_sha, 'heldout_commitment': split_commitment,
                'site_commitment': site_record.commitment_sha256, 'prediction_sha256': prediction_sha,
                'freeze_sha256': freeze.sha256, 'evaluation_sha256': hashlib.sha256(evaluation).hexdigest(),
                'model_state_sha256': model_state}
    return dict(journal_dir=journal, receipts_bytes=receipts_raw, binding_bytes=binding, split_bytes=split,
                split_nonce=split_nonce, site_bytes=site_record.record_bytes, site_nonce=site_nonce,
                freeze_bytes=freeze.record_bytes, prediction_bytes=prediction, evaluation_bytes=evaluation,
                started_bytes=started, terminal_bytes=terminal, expected=expected, externally_quiescent=True)


def test_closed_synthetic_journal_recomputes_all_scores_and_controls(synthetic_journal):
    module = auditor()
    result = module.audit_saved(**synthetic_journal)
    evaluation = json.loads(synthetic_journal['evaluation_bytes'])
    assert result['status'] == 'independent_saved_evidence_pass'
    assert result['records'] == 128
    assert result['sites'] == evaluation['sites']
    assert result['selection'] == evaluation['selection']
    assert 'site' not in result and 'cases' not in result
    assert result['structural_audit']['inventory']['owner_side_file_bytes'] == sum(
        len(synthetic_journal[key]) for key in ('started_bytes', 'terminal_bytes', 'receipts_bytes', 'binding_bytes'))


def test_closed_journal_rejects_changed_external_evaluation_digest(synthetic_journal):
    module = auditor()
    changed = dict(synthetic_journal)
    changed['evaluation_bytes'] += b' '
    with pytest.raises(ValueError):
        module.audit_saved(**changed)


def test_closed_journal_rejects_a_self_consistently_rehashed_wrong_score(synthetic_journal):
    module = auditor()
    altered = json.loads(synthetic_journal['evaluation_bytes'])
    altered['sites']['1']['mae'] += 1.0
    raw = module.canonical(altered)
    changed = dict(synthetic_journal, evaluation_bytes=raw,
                   expected=dict(synthetic_journal['expected'], evaluation_sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match='independent recomputation'):
        module.audit_saved(**changed)


def test_private_site_nonce_and_external_quiescence_are_required(synthetic_journal):
    module = auditor()
    with pytest.raises(ValueError):
        module.audit_saved(**dict(synthetic_journal, site_nonce=bytes([7]) * 32))
    with pytest.raises(ValueError):
        module.audit_saved(**dict(synthetic_journal, externally_quiescent=False))
    terminal = json.loads(synthetic_journal['terminal_bytes'])
    terminal['model_state_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        module.audit_saved(**dict(synthetic_journal, terminal_bytes=module.canonical(terminal)))


def test_audit_uses_owned_expected_digest_snapshot(synthetic_journal, monkeypatch):
    module = auditor()
    expected = synthetic_journal['expected']
    original = dict(expected)
    real_read = module._read
    def mutate_after_start(*args, **kwargs):
        expected['evaluation_sha256'] = '0' * 64
        return real_read(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module, '_read', mutate_after_start)
        try:
            result = module.audit_saved(**synthetic_journal)
        finally:
            expected.clear()
            expected.update(original)
    assert result['evaluation_sha256'] == original['evaluation_sha256']


def test_private_audit_receipt_is_bounded_and_exclusive(tmp_path):
    module = auditor()
    path = tmp_path / 'independent-audit.json'
    result = {'status': 'fabricated', 'count': 128}
    digest = module.write_private_receipt(path, result)
    expected = module.canonical(result)
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    with pytest.raises(FileExistsError):
        module.write_private_receipt(path, {'status': 'replacement'})
    assert path.read_bytes() == expected
    with pytest.raises(ValueError):
        module.write_private_receipt(tmp_path / 'other.json', result)


def test_private_audit_receipt_does_not_create_oversized_output(tmp_path):
    module = auditor()
    path = tmp_path / 'independent-audit.json'
    with pytest.raises(ValueError):
        module.write_private_receipt(path, {'padding': 'x' * (1024 * 1024)})
    assert not path.exists()


def test_receipt_fsync_failure_is_incomplete_and_cannot_be_retried(tmp_path, monkeypatch):
    module = auditor()
    path = tmp_path / 'independent-audit.json'
    def fail_fsync(_fd):
        raise OSError('fabricated fsync failure')
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'fsync', fail_fsync)
        with pytest.raises(OSError, match='fabricated fsync failure'):
            module.write_private_receipt(path, {'status': 'fabricated'})
    assert path.exists()
    with pytest.raises(FileExistsError):
        module.write_private_receipt(path, {'status': 'replacement'})


def test_score_handoff_requires_clean_supervisor_and_bound_receipts():
    module = auditor()
    binding = module.canonical({'disjoint_status': 'pending'})
    evaluation = b'fabricated private score'
    report = {'binding_sha256': hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding).hexdigest(),
              'evaluation_sha256': hashlib.sha256(evaluation).hexdigest(),
              'receipts_sha256': 'b' * 64, 'last_frame_sha256': 'c' * 64}
    report['structural_audit'] = {'status': 'evidence_complete', 'forward_count': 128,
        'binding_sha256': report['binding_sha256'], 'receipts_sha256': report['receipts_sha256'],
        'last_frame_sha256': report['last_frame_sha256'], 'scientific_validation': 'pending independent audit',
        'inventory': {'retained_tensor_bytes': 0, 'tensor_encoding': 'journal-frames',
                      'encoded_tensor_occurrence_bytes': 200, 'journal_file_bytes': 300, 'owner_side_file_bytes': 100}}
    command = ['fake-python', 'fake-child']
    supervisor = {'status': 'evidence_complete', 'exit_code': 0, 'child_pid': 123,
        'child_reaped': True, 'job_drained': True, 'job_closed': True, 'logs_complete': True,
        'postexit_active_processes': 0, 'cleanup_verified': True, 'retry_safe': False,
        'cancellation_requested': False, 'quarantined': False, 'log_overflow': False,
        'supervisor_error': None, 'command': command,
        'stdout_sha256': hashlib.sha256(b'').hexdigest(), 'stderr_sha256': hashlib.sha256(b'').hexdigest(),
        'retained_tensor_files': {'accounting_complete': True, 'total_bytes': 0, 'files': []},
        'audit': report['structural_audit']}
    supervisor_raw = (json.dumps(supervisor, sort_keys=True, indent=2) + '\n').encode()
    postscore = module.canonical({'schema': 'mlg-injection-heldout-private-score-v1',
        'status': 'scored-private-awaiting-independent-audit', 'binding_sha256': report['binding_sha256'],
        'binding_file_sha256': hashlib.sha256(binding).hexdigest(),
        'supervisor_sha256': hashlib.sha256(supervisor_raw).hexdigest(),
        'evaluation_sha256': report['evaluation_sha256'], 'disjoint_status': 'pending',
        'source_qualification': 'prospective orchestration; same-full-source rule departed'})
    values = dict(report=report, binding_bytes=binding, evaluation_bytes=evaluation, supervisor_bytes=supervisor_raw,
                  stdout=b'', stderr=b'', postscore_bytes=postscore,
                  expected_supervisor_sha256=hashlib.sha256(supervisor_raw).hexdigest(),
                  expected_postscore_sha256=hashlib.sha256(postscore).hexdigest(),
                  expected_command_sha256=hashlib.sha256(module.canonical(command)).hexdigest())
    module.verify_score_handoff(**values)
    with pytest.raises(ValueError):
        module.verify_score_handoff(**dict(values, stdout=b'unexpected'))
    altered = dict(supervisor, child_reaped=False)
    bad_raw = (json.dumps(altered, sort_keys=True, indent=2) + '\n').encode()
    with pytest.raises(ValueError):
        module.verify_score_handoff(**dict(values, supervisor_bytes=bad_raw,
            expected_supervisor_sha256=hashlib.sha256(bad_raw).hexdigest()))
    duplicated = supervisor_raw.replace(b'"child_reaped": true', b'"child_reaped": false, "child_reaped": true')
    duplicated_sha = hashlib.sha256(duplicated).hexdigest()
    altered_postscore = json.loads(postscore)
    altered_postscore['supervisor_sha256'] = duplicated_sha
    altered_postscore = module.canonical(altered_postscore)
    with pytest.raises(ValueError):
        module.verify_score_handoff(**dict(values, supervisor_bytes=duplicated,
            expected_supervisor_sha256=duplicated_sha, postscore_bytes=altered_postscore,
            expected_postscore_sha256=hashlib.sha256(altered_postscore).hexdigest()))
    import copy
    impossible = copy.deepcopy(supervisor)
    impossible['audit']['inventory']['journal_file_bytes'] = 999999999
    bad_raw = (json.dumps(impossible, sort_keys=True, indent=2) + '\n').encode()
    bad_sha = hashlib.sha256(bad_raw).hexdigest()
    bad_postscore = json.loads(postscore)
    bad_postscore['supervisor_sha256'] = bad_sha
    bad_postscore = module.canonical(bad_postscore)
    with pytest.raises(ValueError):
        module.verify_score_handoff(**dict(values, supervisor_bytes=bad_raw, expected_supervisor_sha256=bad_sha,
            postscore_bytes=bad_postscore, expected_postscore_sha256=hashlib.sha256(bad_postscore).hexdigest()))


def test_owner_side_file_count_includes_the_binding_bytes():
    module = auditor()
    assert module.owner_side_size(b'abc', b'de', b'f', b'ghi') == 9
