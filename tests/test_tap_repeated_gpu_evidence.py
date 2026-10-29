import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from looking_glass import tap_repeated_gpu_evidence as evidence
from looking_glass.supervisor import SupervisionResult
from looking_glass.tap_repeated_protocol import EXPERIMENT_PROTOCOL, SCHEDULE_SHA256


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def fixture():
    identity = {'run_id': 'a' * 32, 'pid': 1234, 'birth': '123456'}
    gpu = 'GPU-00000000-0000-0000-0000-000000000001'
    rows = [{'sequence': index} for index in range(86)]
    journal_sha256 = hashlib.sha256(b''.join(raw(row) for row in rows)).hexdigest()
    clock = {'name': 'perf_counter_ns', 'implementation': 'QueryPerformanceCounter()'}
    parent_runtime = {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}
    runtime = {**parent_runtime, 'torch': '2.8.0+cu128', 'cuda_build': '12.8'}
    model = {'weights_sha256': 'c' * 64, 'input_ids': [15496, 995]}
    parent = {'session': 0, 'state': 'pending_owner_evidence', 'screen_state': 'no_contamination_observed',
              'journal': rows, 'journal_sha256': journal_sha256,
              'preflight_started_ns': 100, 'workload_ended_ns': 200,
              'parent_identity': identity, 'gpu_uuid': gpu, 'cuda_total_bytes': 8 * 1024**3,
              'activity_smi_total_bytes': 8 * 1024**3,
              'parent_observed_live_handle': True, 'activity_identity_held_through_queries': True,
              'activity': {'observations': [], 'process_queries': [], 'errors': [],
                           'complete': True, 'in_flight': False, 'sealed': True},
              'supervision': SupervisionResult(identity['run_id'], identity['pid'], 0, True, False, 'clean')}
    candidate = {'schema': 'tap-repeated-candidate-v1', 'protocol': EXPERIMENT_PROTOCOL,
        'run_id': 'b' * 32, 'supervisor_run_id': identity['run_id'], 'owner_pid': identity['pid'],
        'owner_birth': identity['birth'], 'session_index': 0, 'source_commit': 'd' * 40,
        'gpu_uuid': gpu, 'cuda_total_bytes': parent['cuda_total_bytes'],
        'cuda_device_name': 'NVIDIA test GPU', 'cuda_compute_capability': [7, 5],
        'cap_bytes': parent['cuda_total_bytes'] // 4, 'free_before_model_transfer': 6 * 1024**3,
        'admission_memory': {'free_bytes': 6 * 1024**3, 'total_bytes': 8 * 1024**3,
                             'allocated_bytes': 0, 'reserved_bytes': 0},
        'memory_before_transfer': {'free_bytes': 6 * 1024**3, 'total_bytes': 8 * 1024**3,
                                   'allocated_bytes': 0, 'reserved_bytes': 0},
        'trace_claim_bytes': 128 * 1024, 'model': model, 'clock': clock, 'runtime': runtime,
        'schedule_sha256': SCHEDULE_SHA256, 'journal_sha256': journal_sha256, 'journal_rows': 86,
        'preflight_started_ns': 100, 'workload_ended_ns': 200, 'whole_run_claims_held': True}
    final = {'schema': 'tap-repeated-final-v1', 'run_id': candidate['run_id'],
        'supervisor_run_id': identity['run_id'], 'owner_pid': identity['pid'], 'owner_birth': identity['birth'],
        'session_index': 0, 'source_commit': candidate['source_commit'], 'gpu_uuid': gpu,
        'candidate_sha256': hashlib.sha256(raw(candidate)).hexdigest(), 'unsafe': False,
        'synchronized_disposal': True, 'claims_released_after_disposal': True,
        'device_lease_held_at_attestation': True, 'allocated_bytes': 0, 'reserved_bytes': 0,
        'active_memory_leases': 0, 'trace_claim_remaining_bytes': 0}
    kwargs = {'parent': parent, 'candidate_raw': raw(candidate), 'final_raw': raw(final),
        'study_run_id': candidate['run_id'], 'source_commit': candidate['source_commit'], 'source_tree': 'e' * 40,
        'expected_model': model, 'expected_runtime': runtime, 'parent_runtime': parent_runtime,
        'expected_device_name': 'NVIDIA test GPU', 'observed_driver': '555.55',
        'clock': clock, 'exclusive': True, 'source_exit_contract': True}
    return kwargs, candidate, final


def test_assembly_binds_owner_records_to_supervision_journal_and_activity():
    kwargs, candidate, _ = fixture()
    attempt, binding = evidence.assemble_private_attempt(**kwargs)
    assert attempt['binding'] == binding
    assert attempt['owner']['child_identity'] == attempt['owner']['parent_identity']
    assert attempt['activity']['own_pid'] == candidate['owner_pid']
    assert attempt['journal'] == kwargs['parent']['journal']
    assert attempt['protocol'] == EXPERIMENT_PROTOCOL
    assert attempt['owner']['claims_released_after_disposal'] is True


@pytest.mark.parametrize('damage', ['partial_screen', 'wrong_supervisor', 'wrong_journal', 'stale_candidate_hash',
    'false_claims', 'unreleased', 'owner_mismatch', 'runtime_mismatch', 'no_exit_contract', 'extra_field',
    'final_boolean_session', 'parent_boolean_session', 'parent_float_pid', 'model_boolean_alias',
    'wrong_device_name', 'boolean_capability', 'bad_admission_memory', 'transfer_scalar_mismatch',
    'parent_float_cuda_total', 'parent_float_preflight', 'parent_float_workload_end',
    'low_transfer_memory', 'bad_driver'])
def test_assembly_rejects_incomplete_or_inconsistent_records(damage):
    kwargs, candidate, final = fixture()
    kwargs = copy.deepcopy(kwargs)
    if damage == 'partial_screen': kwargs['parent']['state'] = 'incomplete'
    elif damage == 'wrong_supervisor': kwargs['parent']['supervision'] = SupervisionResult('a' * 32, 1234, 1, True, False, 'clean')
    elif damage == 'wrong_journal': kwargs['parent']['journal'][4]['sequence'] = 99
    elif damage == 'stale_candidate_hash': final['candidate_sha256'] = 'f' * 64
    elif damage == 'false_claims': candidate['whole_run_claims_held'] = False
    elif damage == 'unreleased': final['active_memory_leases'] = 1
    elif damage == 'owner_mismatch': final['owner_pid'] += 1
    elif damage == 'runtime_mismatch': candidate['runtime']['torch'] = 'unexpected'
    elif damage == 'no_exit_contract': kwargs['source_exit_contract'] = False
    elif damage == 'extra_field': candidate['private_sentinel'] = 'must not pass'
    elif damage == 'final_boolean_session': final['session_index'] = False
    elif damage == 'parent_boolean_session': kwargs['parent']['session'] = False
    elif damage == 'parent_float_pid': kwargs['parent']['parent_identity']['pid'] = 1234.0
    elif damage == 'model_boolean_alias':
        kwargs['expected_model']['use_cache'] = False
        candidate['model']['use_cache'] = 0
    elif damage == 'wrong_device_name': candidate['cuda_device_name'] = 'another GPU'
    elif damage == 'boolean_capability': candidate['cuda_compute_capability'] = [True, 5]
    elif damage == 'bad_admission_memory': candidate['admission_memory']['free_bytes'] = True
    elif damage == 'parent_float_cuda_total': kwargs['parent']['cuda_total_bytes'] = float(candidate['cuda_total_bytes'])
    elif damage == 'parent_float_preflight': kwargs['parent']['preflight_started_ns'] = float(candidate['preflight_started_ns'])
    elif damage == 'parent_float_workload_end': kwargs['parent']['workload_ended_ns'] = float(candidate['workload_ended_ns'])
    elif damage == 'transfer_scalar_mismatch': candidate['memory_before_transfer']['free_bytes'] -= 1
    elif damage == 'low_transfer_memory':
        candidate['memory_before_transfer']['free_bytes'] = candidate['free_before_model_transfer'] = 2 * 1024**3
    elif damage == 'bad_driver': kwargs['observed_driver'] = 'unknown'
    kwargs['candidate_raw'] = raw(candidate)
    if damage not in ('stale_candidate_hash',):
        final['candidate_sha256'] = hashlib.sha256(kwargs['candidate_raw']).hexdigest()
    kwargs['final_raw'] = raw(final)
    with pytest.raises(ValueError):
        evidence.assemble_private_attempt(**kwargs)


def test_noncanonical_or_duplicate_owner_json_is_rejected():
    kwargs, _, _ = fixture()
    kwargs['candidate_raw'] = b'{"schema":"a","schema":"b"}\n'
    with pytest.raises(ValueError):
        evidence.assemble_private_attempt(**kwargs)


def test_v12_assembled_attempt_replays_the_full_fixed_schedule():
    from looking_glass import tap_repeated_protocol as protocol
    path = Path(__file__).with_name('test_tap_repeated_protocol.py')
    spec = importlib.util.spec_from_file_location('tap_protocol_fixture', path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    schedule = protocol.load_schedule(source.SCHEDULE)
    known_binding = {'source_commit': 'd' * 40, 'source_tree': 'e' * 40, 'model_sha256': 'c' * 64,
                     'runtime_sha256': 'f' * 64, 'device_sha256': '1' * 64}
    exemplar = source._attempt(schedule, 0, known_binding)
    kwargs, candidate, final = fixture()
    parent = kwargs['parent']
    parent['journal'] = exemplar['journal']
    parent['journal_sha256'] = hashlib.sha256(b''.join(raw(row) for row in parent['journal'])).hexdigest()
    parent['preflight_started_ns'] = exemplar['activity']['workload_started_ns']
    parent['workload_ended_ns'] = exemplar['activity']['workload_ended_ns']
    parent['activity'] = {key: copy.deepcopy(exemplar['activity'][key])
                          for key in ('observations', 'errors', 'process_queries')}
    for row in parent['activity']['observations']:
        row['gpu_uuid'] = candidate['gpu_uuid']
    parent['activity'].update(complete=True, in_flight=False, sealed=True)
    kwargs['clock'] = candidate['clock'] = source.CLOCK
    candidate.update(journal_sha256=parent['journal_sha256'],
        preflight_started_ns=parent['preflight_started_ns'], workload_ended_ns=parent['workload_ended_ns'])
    kwargs['candidate_raw'] = raw(candidate)
    final['candidate_sha256'] = hashlib.sha256(kwargs['candidate_raw']).hexdigest()
    kwargs['final_raw'] = raw(final)
    attempt, binding = evidence.assemble_private_attempt(**kwargs)
    replay = protocol.summarize_attempts(schedule, [attempt], trusted_binding=binding)
    assert replay['complete_sessions'] == 1
    assert replay['slots'][0]['state'] == 'complete'
    assert replay['slots'][0]['summary']['measured_samples'] == 36
