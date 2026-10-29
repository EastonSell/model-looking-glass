"""Fabricated handoff records; the nested comparator never sees real cases."""
import json
from pathlib import Path
import sys

import pytest

from looking_glass import natural_successor_query_handoff as handoff
from test_natural_successor_query import canonical, query, run, setup, sha


def arguments(value):
    config = json.loads(value['config_path'].read_bytes())
    source_sha = sha(Path(config['sources']['caller']).read_bytes())
    return {'interpreter': sys._base_executable, 'interpreter_sha256': sha(Path(sys._base_executable).read_bytes()),
            'bootstrap': query._BOOTSTRAP,
            'query_source_sha256': source_sha}


@pytest.fixture
def completed(setup):
    result = run(setup)
    assert result['status'] == 'clear'
    args = arguments(setup)
    inner = handoff.verify_inner(setup['config_path'], setup['config_sha256'], setup['operation_sha256'], **args)
    config = json.loads(setup['config_path'].read_bytes())
    command = [sys._base_executable, '-I', '-S', '-B', '-c', query._BOOTSTRAP,
               config['sources']['caller'], args['query_source_sha256'], '--config', str(setup['config_path']),
               '--config-sha256', setup['config_sha256'], '--operation-sha256', setup['operation_sha256']]
    command_sha = sha(canonical(command))
    operation = json.loads(Path(config['operation']).read_bytes())
    invocation = {'schema': 'mlg-natural-successor-query-invocation-v1',
                  'operation_sha256': setup['operation_sha256'], 'config_sha256': setup['config_sha256'],
                  'query_source_sha256': args['query_source_sha256'], 'command_sha256': command_sha,
                  'pool_input_sha256': operation['pool_input_sha256'], 'old_commitment': operation['old_commitment']}
    (setup['root'] / 'pool-query-invocation.json').write_bytes(canonical(invocation))
    output = setup['root'] / 'pool-query-outer'
    output.mkdir()
    start = {'schema': 'mlg-injection-outer-process-start-v1', 'purpose': 'preparation',
             'config_sha256': setup['config_sha256'], 'operation_sha256': setup['operation_sha256'],
             'command_sha256': command_sha, 'supervisor_source_sha256': args['query_source_sha256'],
             'start_token_sha256': sha(b'MLG-natural-discovery-start-v1\n')}
    outer = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'preparation',
             'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
             'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
             'parent_pid': inner['child_pid'] + 1, 'parent_birth_100ns': 12345,
             'config_sha256': setup['config_sha256'], 'operation_sha256': setup['operation_sha256'],
             'command_sha256': command_sha, 'supervisor_source_sha256': args['query_source_sha256'],
             'supervisor_terminal_sha256': inner['observation_sha256']}
    (output / 'started.json').write_bytes(canonical(start))
    (output / 'observation.json').write_bytes(canonical(outer))
    return setup, args, inner, outer


def verify(value, args, inner):
    return handoff.verify_outer(value['config_path'], value['config_sha256'], value['operation_sha256'],
                                **args)


def test_clean_inner_and_outer_records_match(completed):
    value, args, inner, outer = completed
    verified = verify(value, args, inner)
    assert verified['receipt_sha256'] == inner['receipt_sha256']
    assert verified['parent_pid'] == outer['parent_pid']
    assert verified['outer_observation_sha256'] == sha(canonical(outer))
    assert verified['invocation_sha256'] == sha((value['root'] / 'pool-query-invocation.json').read_bytes())


@pytest.mark.parametrize('change', ['missing', 'altered'])
def test_invocation_marker_must_match_the_operation(completed, change):
    setup, args, inner, _ = completed
    path = setup['root'] / 'pool-query-invocation.json'
    if change == 'missing':
        path.unlink()
    else:
        marker = json.loads(path.read_bytes())
        marker['old_commitment'] = '0' * 64
        path.write_bytes(canonical(marker))
    with pytest.raises((handoff.HandoffError, OSError)):
        verify(setup, args, inner)


def test_invocation_marker_mutation_during_handoff_is_rejected(completed, monkeypatch):
    setup, args, inner, _ = completed
    original, calls = handoff._identity, []
    def mutate(root, operation):
        result = original(root, operation)
        calls.append(1)
        if len(calls) == 4:
            path = setup['root'] / 'pool-query-invocation.json'
            marker = json.loads(path.read_bytes())
            marker['old_commitment'] = '0' * 64
            path.write_bytes(canonical(marker))
        return result
    monkeypatch.setattr(handoff, '_identity', mutate)
    with pytest.raises(handoff.HandoffError):
        verify(setup, args, inner)


@pytest.mark.parametrize('field,value', [
    ('exit_code', False), ('active_job_members', False), ('parent_reaped', 1),
    ('outer_job_closed', False), ('supervisor_terminal_sha256', '0' * 64),
    ('command_sha256', '0' * 64), ('parent_birth_100ns', 0), ('parent_birth_100ns', 2 ** 64),
])
def test_outer_record_type_and_binding_mismatch_is_rejected(completed, field, value):
    setup, args, inner, outer = completed
    outer[field] = value
    (setup['root'] / 'pool-query-outer/observation.json').write_bytes(canonical(outer))
    with pytest.raises(handoff.HandoffError):
        verify(setup, args, inner)


def test_parent_cannot_be_same_process_as_comparator(completed):
    setup, args, inner, outer = completed
    outer['parent_pid'] = inner['child_pid']
    (setup['root'] / 'pool-query-outer/observation.json').write_bytes(canonical(outer))
    with pytest.raises(handoff.HandoffError):
        verify(setup, args, inner)


@pytest.mark.parametrize('file,change', [
    ('pool-query-receipt.json', lambda row: row.update(status='collision')),
    ('pool-query-claim.json', lambda row: row.update(old_commitment='0' * 64)),
    ('pool-query-observation.json', lambda row: row.update(reaped=1)),
    ('pool-query-supervisor/supervisor.json', lambda row: row.update(exit_code=False)),
])
def test_inner_claim_receipt_and_supervisor_tampering_is_rejected(completed, file, change):
    setup, args, _, _ = completed
    path = setup['root'] / file
    value = json.loads(path.read_bytes())
    change(value)
    path.write_bytes(canonical(value))
    with pytest.raises(handoff.HandoffError):
        handoff.verify_inner(setup['config_path'], setup['config_sha256'], setup['operation_sha256'], **args)


def test_no_phase_claim_may_precede_query_handoff(completed):
    setup, args, _, _ = completed
    (setup['root'] / 'phase-claim.json').write_bytes(b'fabricated')
    with pytest.raises(handoff.HandoffError):
        handoff.verify_inner(setup['config_path'], setup['config_sha256'], setup['operation_sha256'], **args)


@pytest.mark.parametrize('field,value', [('bootstrap', 'forged bootstrap'), ('interpreter_sha256', '0' * 64)])
def test_unbound_bootstrap_or_interpreter_is_rejected(completed, field, value):
    setup, args, _, _ = completed
    args[field] = value
    with pytest.raises(handoff.HandoffError):
        handoff.verify_inner(setup['config_path'], setup['config_sha256'], setup['operation_sha256'], **args)


def test_receipt_mutation_during_verification_is_rejected(completed, monkeypatch):
    setup, args, inner, _ = completed
    original, calls = handoff._identity, []
    def mutate(root, operation):
        result = original(root, operation)
        calls.append(1)
        if len(calls) == 2:
            path = setup['root'] / 'pool-query-receipt.json'
            receipt = json.loads(path.read_bytes())
            receipt['status'] = 'collision'
            path.write_bytes(canonical(receipt))
        return result
    monkeypatch.setattr(handoff, '_identity', mutate)
    with pytest.raises(handoff.HandoffError):
        verify(setup, args, inner)
