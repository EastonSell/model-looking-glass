"""Bind one closed owner journal to its supervised parent and cleanup records.

This module reads no device and promotes no result. Its output must still pass
the fixed protocol replay. Inputs are trusted local observations, not an
authentication against a hostile process or proof of native CUDA teardown.
"""

import hashlib
import json
import re

from .device_identity import normalize_gpu_uuid
from .supervisor import SupervisionResult
from .tap_repeated_protocol import EXPERIMENT_PROTOCOL, GIB, SCHEDULE_SHA256, check_memory


_CANDIDATE_KEYS = frozenset(('schema', 'protocol', 'run_id', 'supervisor_run_id', 'owner_pid',
    'owner_birth', 'session_index', 'source_commit', 'gpu_uuid', 'cuda_total_bytes', 'cap_bytes',
    'cuda_device_name', 'cuda_compute_capability',
    'free_before_model_transfer', 'admission_memory', 'memory_before_transfer',
    'trace_claim_bytes', 'model', 'clock', 'runtime', 'schedule_sha256',
    'journal_sha256', 'journal_rows', 'preflight_started_ns', 'workload_ended_ns', 'whole_run_claims_held'))
_FINAL_KEYS = frozenset(('schema', 'run_id', 'supervisor_run_id', 'owner_pid', 'owner_birth',
    'session_index', 'source_commit', 'gpu_uuid', 'candidate_sha256', 'unsafe', 'synchronized_disposal',
    'claims_released_after_disposal', 'device_lease_held_at_attestation', 'allocated_bytes', 'reserved_bytes',
    'active_memory_leases', 'trace_claim_remaining_bytes'))
_IDENTITY_KEYS = ('run_id', 'supervisor_run_id', 'owner_pid', 'owner_birth', 'session_index',
                  'source_commit', 'gpu_uuid')


def _canonical(value, *, newline=False):
    try:
        text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError('invalid evidence value') from exc
    return (text + ('\n' if newline else '')).encode('utf-8')


def _object(raw, *, keys):
    if type(raw) is not bytes or not 1 <= len(raw) <= 256 * 1024:
        raise ValueError('invalid owner record bytes')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate owner record field')
            result[key] = value
        return result
    def reject_constant(_value):
        raise ValueError('nonfinite owner record')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError('invalid owner record JSON') from exc
    if type(value) is not dict or set(value) != keys or _canonical(value, newline=True) != raw:
        raise ValueError('owner record is not the exact canonical schema')
    return value


def assemble_private_attempt(*, parent, candidate_raw, final_raw, study_run_id, source_commit, source_tree,
                             expected_model, expected_runtime, parent_runtime, expected_device_name, observed_driver,
                             clock, exclusive,
                             source_exit_contract):
    """Build a private replay input only after independent closure checks.

    ``source_exit_contract`` is the caller's separately reviewed assertion
    that the verified owner roots its device lease until the bootstrap's hard
    exit. The final observation is at attestation, not after process exit.
    """
    candidate = _object(candidate_raw, keys=_CANDIDATE_KEYS)
    final = _object(final_raw, keys=_FINAL_KEYS)
    if (type(parent) is not dict or parent.get('state') != 'pending_owner_evidence'
            or parent.get('screen_state') not in ('no_contamination_observed', 'contaminated')
            or exclusive is not True or source_exit_contract is not True
            or type(study_run_id) is not str or re.fullmatch('[0-9a-f]{32}', study_run_id) is None
            or type(source_commit) is not str or re.fullmatch('[0-9a-f]{40}', source_commit) is None
            or type(source_tree) is not str or re.fullmatch('[0-9a-f]{40}', source_tree) is None
            or type(expected_model) is not dict or type(expected_runtime) is not dict
            or type(parent_runtime) is not dict or type(clock) is not dict):
        raise ValueError('parent or source contract is incomplete')
    if (candidate['schema'] != 'tap-repeated-candidate-v1' or candidate['protocol'] != EXPERIMENT_PROTOCOL
            or final['schema'] != 'tap-repeated-final-v1'
            or candidate['run_id'] != study_run_id or candidate['source_commit'] != source_commit
            or any(type(final[key]) is not type(candidate[key]) or final[key] != candidate[key]
                   for key in _IDENTITY_KEYS)
            or final['candidate_sha256'] != hashlib.sha256(candidate_raw).hexdigest()
            or candidate['schedule_sha256'] != SCHEDULE_SHA256):
        raise ValueError('owner identity, schedule or candidate digest differs')
    run, pid, birth, session = (candidate[key] for key in
        ('supervisor_run_id', 'owner_pid', 'owner_birth', 'session_index'))
    if (type(run) is not str or re.fullmatch('[0-9a-f]{32}', run) is None
            or type(pid) is not int or not 0 < pid < 2**32 or type(birth) is not str
            or re.fullmatch('[1-9][0-9]{0,19}', birth) is None or int(birth) >= 2**64
            or type(session) is not int or session not in range(5)
            or type(parent.get('session')) is not int or parent['session'] != session):
        raise ValueError('owner process identity is invalid')
    try:
        uuid = normalize_gpu_uuid(candidate['gpu_uuid'])
    except (TypeError, ValueError) as exc:
        raise ValueError('physical device identity is invalid') from exc
    total, cap, free, trace = (candidate[key] for key in
        ('cuda_total_bytes', 'cap_bytes', 'free_before_model_transfer', 'trace_claim_bytes'))
    capability = candidate['cuda_compute_capability']
    if (uuid != candidate['gpu_uuid'] or uuid != parent.get('gpu_uuid')
            or type(expected_device_name) is not str or not expected_device_name.strip()
            or candidate['cuda_device_name'] != expected_device_name
            or type(observed_driver) is not str
            or re.fullmatch('[0-9]+(?:\\.[0-9]+){1,3}', observed_driver) is None
            or type(capability) is not list or len(capability) != 2
            or any(type(value) is not int or not 0 <= value <= 99 for value in capability)
            or any(type(value) is not int for value in (total, cap, free, trace))
            or total <= 0 or cap != total // 4 or not 3 * GIB <= free <= total or trace != 128 * 1024
            or type(parent.get('cuda_total_bytes')) is not int or parent['cuda_total_bytes'] != total
            or type(parent.get('activity_smi_total_bytes')) is not int):
        raise ValueError('device identity or admission differs')
    for key in ('admission_memory', 'memory_before_transfer'):
        snapshot = candidate[key]
        if (type(snapshot) is not dict or set(snapshot) !=
                {'free_bytes', 'total_bytes', 'allocated_bytes', 'reserved_bytes'}):
            raise ValueError('setup memory snapshot is incomplete')
        check_memory(snapshot, expected_total=total, cap_bytes=cap)
        if snapshot['free_bytes'] < 3 * GIB:
            raise ValueError('setup memory falls below the fixed floor')
    if candidate['memory_before_transfer']['free_bytes'] != free:
        raise ValueError('pre-transfer free memory does not match its snapshot')
    if (candidate['whole_run_claims_held'] is not True or final['unsafe'] is not False
            or final['synchronized_disposal'] is not True
            or final['claims_released_after_disposal'] is not True
            or final['device_lease_held_at_attestation'] is not True
            or any(type(final[key]) is not int or final[key] != 0 for key in
                   ('allocated_bytes', 'reserved_bytes', 'active_memory_leases', 'trace_claim_remaining_bytes'))):
        raise ValueError('owner disposal or claim release is incomplete')
    if (_canonical(candidate['model']) != _canonical(expected_model)
            or _canonical(candidate['runtime']) != _canonical(expected_runtime)
            or _canonical(candidate['clock']) != _canonical(clock)
            or any(type(expected_runtime.get(key)) is not type(value) or expected_runtime.get(key) != value
                   for key, value in parent_runtime.items())
            or type(expected_model.get('weights_sha256')) is not str
            or re.fullmatch('[0-9a-f]{64}', expected_model['weights_sha256']) is None):
        raise ValueError('model, runtime or clock binding differs')
    rows, digest = parent.get('journal'), parent.get('journal_sha256')
    if (type(rows) is not list or len(rows) != 86 or type(candidate['journal_rows']) is not int
            or candidate['journal_rows'] != 86 or type(digest) is not str
            or re.fullmatch('[0-9a-f]{64}', digest) is None
            or candidate['journal_sha256'] != digest
            or hashlib.sha256(b''.join(_canonical(row, newline=True) for row in rows)).hexdigest() != digest
            or type(candidate['preflight_started_ns']) is not int
            or type(candidate['workload_ended_ns']) is not int
            or type(parent.get('preflight_started_ns')) is not int
            or type(parent.get('workload_ended_ns')) is not int
            or candidate['preflight_started_ns'] != parent.get('preflight_started_ns')
            or candidate['workload_ended_ns'] != parent.get('workload_ended_ns')):
        raise ValueError('owner journal or screen anchors differ')
    identity = {'run_id': run, 'pid': pid, 'birth': birth}
    supervision = parent.get('supervision')
    if (_canonical(parent.get('parent_identity')) != _canonical(identity)
            or parent.get('parent_observed_live_handle') is not True
            or parent.get('activity_identity_held_through_queries') is not True
            or type(supervision) is not SupervisionResult or supervision.run_id != run
            or type(supervision.pid) is not int or supervision.pid != pid
            or type(supervision.exit_code) is not int or supervision.exit_code != 0
            or supervision.clean is not True or supervision.timed_out is not False):
        raise ValueError('supervisor or live-handle evidence differs')
    snapshot = parent.get('activity')
    if (type(snapshot) is not dict or snapshot.get('complete') is not True
            or snapshot.get('in_flight') is not False or snapshot.get('sealed') is not True
            or any(type(snapshot.get(key)) is not list for key in ('observations', 'errors', 'process_queries'))):
        raise ValueError('parent activity inventory is incomplete')
    binding = {'source_commit': source_commit, 'source_tree': source_tree,
        'model_sha256': expected_model['weights_sha256'],
        'runtime_sha256': hashlib.sha256(_canonical(expected_runtime)).hexdigest(),
        'device_sha256': hashlib.sha256(_canonical({'gpu_uuid': uuid, 'cuda_total_bytes': total,
            'name': expected_device_name, 'compute_capability': capability,
            'driver_observation': observed_driver})).hexdigest()}
    activity = {key: json.loads(_canonical(snapshot[key])) for key in ('observations', 'errors', 'process_queries')}
    activity.update(gpu_uuid=uuid, own_pid=pid, workload_started_ns=candidate['preflight_started_ns'],
                    workload_ended_ns=candidate['workload_ended_ns'],
                    expected_total_bytes=parent['activity_smi_total_bytes'], exclusive=True)
    owner = {'unsafe': False, 'synchronized_disposal': True, 'whole_run_claims_held': True,
        'claims_released_after_disposal': True, 'device_lease_held_through_process_exit': True,
        'supervisor_control_held_until_reap': True, 'parent_observed_live_handle': True,
        'activity_identity_held_through_queries': True, 'gpu_uuid': uuid,
        'child_identity': dict(identity), 'parent_identity': dict(identity), 'child_reaped': True, 'exit_code': 0}
    attempt = {'session': session, 'protocol': EXPERIMENT_PROTOCOL, 'binding': binding,
        'evidence_kind': 'supervised_cuda', 'total_bytes': total,
        'activity_total_bytes': parent['activity_smi_total_bytes'], 'cap_bytes': cap,
        'free_before_model_transfer': free, 'trace_claim_bytes': trace,
        'journal': json.loads(_canonical(rows)), 'clock': json.loads(_canonical(clock)),
        'activity': activity, 'owner': owner}
    return attempt, binding
