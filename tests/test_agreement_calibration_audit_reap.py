"""Synthetic owner/result linkage tests; no subprocess or GPU is launched."""
from copy import deepcopy
import hashlib
import json

import pytest

from looking_glass import agreement_calibration_saved_audit as audit
from test_agreement_calibration_audit_identity import frozen_identity


def line(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


@pytest.fixture
def reaped(frozen_identity):
    identity_records, commit, uuid, _ = frozen_identity
    freeze_sha = hashlib.sha256(identity_records['run_manifest']).hexdigest()
    identity = audit.audit_identity(identity_records, expected_commit=commit, expected_device_uuid=uuid,
                                    expected_freeze_sha256=freeze_sha)
    profiles = {'profile_ids': ['1' * 64, '2' * 64], 'estimates_bytes': [600_000_000, 700_000_000]}
    return make_reaped(identity, freeze_sha, profiles)


def make_reaped(identity, freeze_sha, profiles):
    commit = identity['freeze']['source']['commit']
    run = identity['run_id']; source_digest = identity['freeze']['source']['sha256']
    supervisor = {'pid': 12345, 'run_id': 'c' * 32, 'exit_code': 0, 'clean': True, 'timed_out': False}
    smi = {'uuid': identity['freeze']['device']['uuid'], 'device_name': 'synthetic GPU',
           'total_bytes': 8 * 1024**3, 'free_bytes': 6 * 1024**3,
           'used_bytes': 2 * 1024**3, 'driver_version': '570.0', 'timestamp_utc': '2026-10-29T00:00:00+00:00'}
    workspace = {'before': {'allocated_bytes': 10, 'reserved_bytes': 20},
                 'after': {'allocated_bytes': 0, 'reserved_bytes': 0},
                 'api': 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces'}
    cleanup = {'runtime_uuid_verified': True, 'active_memory_leases': 0, 'device_synchronized': True,
        'model_weakref_dead': True, 'allocator_allocated_bytes': 0, 'allocator_reserved_bytes': 0,
        'pytorch_cublas_cache_clear_completed': True, 'trace_gpu_bytes': 0, 'trace_host_bytes': 0}
    measurement = {'schema': 'agreement-calibration-candidate-v1', 'run_id': run,
        **profiles, 'source_sha256': identity['source_sha256'], 'runtime_sha256': identity['runtime_sha256'],
        'asset_sha256': identity['asset_sha256'], 'run_manifest_sha256': freeze_sha,
        'sample_count': 14, 'stage_count': 84, 'workspace_cleanup': workspace,
        'limitations': ['synthetic observations are not worst-case memory bounds']}
    owner = {'schema': 'agreement-calibration-owner-v1', 'run_id': run, 'supervisor_run_id': supervisor['run_id'],
             'pid': supervisor['pid'], 'source_commit': commit, 'source_sha256': source_digest}
    candidate = {'schema': 'agreement-calibration-owner-candidate-v1', 'run_id': run,
                 'supervisor_run_id': supervisor['run_id'], 'pid': supervisor['pid'],
                 'value': {'measurement': measurement, 'cleanup': cleanup}}
    final = {'schema': 'agreement-calibration-owner-final-v1', 'run_id': run,
             'supervisor_run_id': supervisor['run_id'], 'pid': supervisor['pid'],
             'candidate_sha256': hashlib.sha256(line(candidate)).hexdigest(), 'cleanup': cleanup}
    start = {'schema': 'agreement-calibration-start-v1', 'run_id': run, 'source_commit': commit,
             'source_sha256': source_digest, 'run_manifest_sha256': freeze_sha, 'started_utc': smi['timestamp_utc']}
    telemetry = {'device_free_bytes': 6 * 1024**3, 'device_total_bytes': 8 * 1024**3,
                 'process_allocated_bytes': 0, 'process_reserved_bytes': 0}
    reserve = (15 * telemetry['device_total_bytes'] + 99) // 100
    committed = 7 * 1024**3 // 2
    budget = {'telemetry': telemetry, 'reserve_bytes': reserve, 'committed_incremental_bytes': committed,
              'active_leases': 2, 'gpu_trace_bytes': 2 * 1024**2, 'host_trace_bytes': 2 * 1024**2,
              'max_gpu_trace_bytes': 2 * 1024**2, 'max_host_trace_bytes': 2 * 1024**2,
              'admissible_estimate_bytes': telemetry['device_free_bytes'] - reserve - committed}
    files = {'started': line(start), 'owner': line(owner), 'candidate': line(candidate), 'final': line(final),
             'supervision': line({'schema': 'agreement-calibration-supervision-v1', 'calibration_run_id': run,
                                  'supervisor': supervisor}),
             'model_checks': line({'cpu_checkpoint_verified': True, 'same_model_after_copy': True,
                                   'device_tie_verified': True, 'post_copy_synchronized': True}),
             'preload_admission': line(budget), 'cleanup': line(workspace)}
    outcome = {'schema': 'agreement-calibration-controller-v1', 'run_id': run, 'source_commit': commit,
        'source_sha256': source_digest, 'run_manifest_sha256': freeze_sha,
        'private_run_dir': identity['freeze']['directory']['path'], 'nvidia_smi_before': smi,
        'nvidia_smi_after': deepcopy(smi), 'status': 'awaiting_independent_audit',
        'supervisor_error_type': None, 'supervisor': supervisor,
        'owner_presence_after_exit': {'query': 'ok', 'owner_pid_listed': False},
        'limitations': ['synthetic fixture only'], 'owner_evidence': measurement,
        'source_protections_retained': False}
    return identity, freeze_sha, profiles, outcome, files


def test_links_reaped_child_candidate_and_cleanup_to_the_frozen_run(reaped):
    identity, freeze_sha, profiles, outcome, files = reaped
    assert hasattr(audit, 'audit_reaped_records'), 'independent reaped-owner audit is missing'
    result = audit.audit_reaped_records(identity, freeze_sha, profiles, line(outcome), files)
    assert result['run_id'] == identity['run_id'] and result['sample_count'] == 14
    assert result['status'] == 'saved_records_consistent_not_approved'


@pytest.mark.parametrize('mutation', [
    'failed_owner', 'timeout', 'nonboolean_clean', 'wrong_supervisor', 'wrong_owner_pid',
    'wrong_candidate_digest', 'unclean_cleanup', 'unknown_presence', 'wrong_smi_device', 'wrong_smi_uuid',
    'changed_source', 'missing_model_check', 'low_preload_headroom', 'changed_estimate', 'missing_record',
])
def test_rejects_uncertain_or_mismatched_owner_records(reaped, mutation):
    identity, freeze_sha, profiles, outcome, files = deepcopy(reaped)
    assert hasattr(audit, 'audit_reaped_records'), 'independent reaped-owner audit is missing'
    if mutation == 'failed_owner': outcome['supervisor']['exit_code'] = 1
    elif mutation == 'timeout': outcome['supervisor']['timed_out'] = True
    elif mutation == 'nonboolean_clean': outcome['supervisor']['clean'] = 1
    elif mutation == 'wrong_supervisor':
        record = json.loads(files['supervision']); record['supervisor']['run_id'] = 'e' * 32; files['supervision'] = line(record)
    elif mutation == 'wrong_owner_pid':
        record = json.loads(files['owner']); record['pid'] += 1; files['owner'] = line(record)
    elif mutation == 'wrong_candidate_digest':
        record = json.loads(files['final']); record['candidate_sha256'] = 'e' * 64; files['final'] = line(record)
    elif mutation == 'unclean_cleanup':
        record = json.loads(files['cleanup']); record['after']['allocated_bytes'] = 1; files['cleanup'] = line(record)
    elif mutation == 'unknown_presence': outcome['owner_presence_after_exit']['query'] = 'unavailable'
    elif mutation == 'wrong_smi_device': outcome['nvidia_smi_after']['device_name'] = 'different GPU'
    elif mutation == 'wrong_smi_uuid': outcome['nvidia_smi_after']['uuid'] = 'GPU-00000000-0000-0000-0000-000000000002'
    elif mutation == 'changed_source': outcome['source_commit'] = 'f' * 40
    elif mutation == 'missing_model_check':
        record = json.loads(files['model_checks']); del record['device_tie_verified']; files['model_checks'] = line(record)
    elif mutation == 'low_preload_headroom':
        record = json.loads(files['preload_admission']); record['telemetry']['device_free_bytes'] = 1
        files['preload_admission'] = line(record)
    elif mutation == 'changed_estimate': profiles['estimates_bytes'][0] -= 1
    elif mutation == 'missing_record': del files['final']
    with pytest.raises(ValueError):
        audit.audit_reaped_records(identity, freeze_sha, profiles, line(outcome), files)


@pytest.mark.parametrize('mutation', ['impossible_smi_sum', 'cleanup_exceeds_device', 'reversed_time'])
def test_rejects_impossible_telemetry_even_with_consistent_candidate_hash(reaped, mutation):
    identity, freeze_sha, profiles, outcome, files = deepcopy(reaped)
    if mutation == 'impossible_smi_sum':
        smi = outcome['nvidia_smi_after']
        smi['free_bytes'] = smi['used_bytes'] = smi['total_bytes']
    elif mutation == 'cleanup_exceeds_device':
        workspace = json.loads(files['cleanup'])
        workspace['before']['reserved_bytes'] = 2 * identity['freeze']['device']['total_bytes']
        files['cleanup'] = line(workspace)
        candidate = json.loads(files['candidate'])
        candidate['value']['measurement']['workspace_cleanup'] = workspace
        files['candidate'] = line(candidate)
        final = json.loads(files['final'])
        final['candidate_sha256'] = hashlib.sha256(files['candidate']).hexdigest()
        files['final'] = line(final)
        outcome['owner_evidence'] = candidate['value']['measurement']
    else:
        outcome['nvidia_smi_after']['timestamp_utc'] = '2026-10-28T23:00:00+00:00'
    with pytest.raises(ValueError):
        audit.audit_reaped_records(identity, freeze_sha, profiles, line(outcome), files)
