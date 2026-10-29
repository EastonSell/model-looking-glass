"""Project one closed retained-byte trace attempt without publishing raw tensors.

The independently reviewed byte audit is included separately. This projection
only reports the saved diagnostic envelope and does not attest execution.
"""

import hashlib
import json
from pathlib import Path
import re
import sys


SOURCE = '8396611b2c9470594e895a3caebaa3cf52fe0749'
EXPECTED = {
    'candidate.json': 'b25d6d1344c48d9ffe14ec2b0dcf58d3f8d23d3ebc214a34fce96a1a4d3daee2',
    'outcome.json': '2598f4670f63356c45a5081053ecedbd03c572b5a919e4a4a7ef12364d1ccd6a',
    'final.json': '29ebda4ff74409cc4ee2933b1155eec93ae2351b8cbbbceb532f79b8ee9def83',
    'start.json': '37efa324b2f316b3e049b8407dab76f0bb01915f1fb33ccd07619f0cf3b3df8c',
    'exit.json': '7525b25e34a0bf6a697e2fd97d1d11774b6323521ac9d62f9fbc6c89ba0a5bf9',
    'manifest.json': '467d8698527c7ff4e01e3f469d58a0ff9fb7ee725df58a58fb25acf701a9eb43',
    'byte-audit.json': 'bb03a869b87a922987eec18fa301bb0f28b2bebd7b765ce2b31b83642b43f281',
}


def read(path, expected=None):
    data = Path(path).read_bytes()
    if len(data) > 1024 ** 2:
        raise ValueError('saved diagnostic is too large')
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('saved diagnostic digest differs')
    return json.loads(data), digest


PRIVATE_KEYS = {'pid', 'parent_pid', 'gpu_uuid', 'device_uuid', 'run_id',
                'supervisor_run_id', 'workspace_root', 'device_lease_path',
                'job_id', 'request_id', 'sample_id'}


def strip_private(value):
    if isinstance(value, dict):
        return {key: strip_private(child) for key, child in value.items() if key not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [strip_private(child) for child in value]
    return value


def check_public(value, path='root'):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError('private identifier field in projection at ' + path)
        for key, child in value.items():
            check_public(child, path + '.' + str(key))
    elif isinstance(value, list):
        for child in value:
            check_public(child, path + '[]')
    elif isinstance(value, str):
        if 'GPU-' in value or re.search(r'[A-Za-z]:[\\/]|\\\\', value):
            raise ValueError('private identifier or absolute path in projection')


def project(run, observer, byte_audit, destination):
    run, observer, byte_audit, destination = map(Path, (run, observer, byte_audit, destination))
    candidate, _ = read(run / 'candidate.json', EXPECTED['candidate.json'])
    outcome, _ = read(run / 'outcome.json', EXPECTED['outcome.json'])
    final, _ = read(run / 'final.json', EXPECTED['final.json'])
    start, _ = read(observer / 'start.json', EXPECTED['start.json'])
    exit_record, _ = read(observer / 'exit.json', EXPECTED['exit.json'])
    manifest, _ = read(run / 'raw' / 'manifest.json', EXPECTED['manifest.json'])
    audit, _ = read(byte_audit, EXPECTED['byte-audit.json'])
    started, started_sha = read(run / 'started.json')
    setup, setup_sha = read(run / 'input-setup.json')
    if (any(row.get('source_commit') != SOURCE for row in
            (candidate, outcome, final, start, exit_record, started))
            or outcome.get('status') != 'complete_candidate'
            or outcome.get('candidate_sha256') != EXPECTED['candidate.json']
            or final.get('candidate_sha256') != EXPECTED['candidate.json']
            or outcome.get('supervisor') != {'exit_code': 0, 'clean': True, 'timed_out': False}
            or exit_record.get('exit_code') != 0 or exit_record.get('reaped') is not True
            or exit_record.get('controller_identity_verified') is not True
            or exit_record.get('pid') != start.get('pid')):
        raise ValueError('closed candidate or direct-exit identity differs')
    if (audit.get('status') != 'saved-byte-consistent' or audit.get('synthetic') is not False
            or audit.get('source_commit') != SOURCE or audit.get('candidate_sha256') != EXPECTED['candidate.json']
            or audit.get('manifest_sha256') != EXPECTED['manifest.json']
            or audit.get('payload_bytes') != 10383176 or audit.get('forward_count') != 10
            or audit.get('capture_event_count') != 54 or audit.get('capture_bit_comparisons') != 36
            or audit.get('logit_bit_comparisons') != 9):
        raise ValueError('independent byte-audit binding differs')
    measurement = candidate['measurement']
    if (measurement['tensor_evidence'] != {'manifest_sha256': EXPECTED['manifest.json'], 'payload_bytes': 10383176}
            or measurement['forward_count'] != 10 or len(measurement['phases']) != 7
            or not isinstance(manifest, dict)):
        raise ValueError('fixed candidate inventory differs')
    forwards = []
    for index in range(1, 11):
        record, digest = read(run / f'forward-{index:03d}.json')
        forwards.append({'record': record, 'file_sha256': digest})
    phase_file_sha = []
    for index in range(1, 8):
        phase_start, start_sha = read(run / f'phase-{index:02d}-start.json')
        phase, digest = read(run / f'phase-{index:02d}.json')
        if phase != measurement['phases'][index - 1] or phase_start['before'] != phase['before']:
            raise ValueError('phase files differ from candidate')
        phase_file_sha.append({'start': start_sha, 'completed': digest})
    output = {
        'schema': 'pretrained-gpu-trace-evidence-v2-public-projection-v1',
        'source_commit': SOURCE, 'source_sha256': started['source_sha256'],
        'private_record_sha256': {**EXPECTED, 'started.json': started_sha, 'input-setup.json': setup_sha},
        'input': setup, 'capacity': started['capacity'],
        'runtime': {key: value for key, value in measurement['runtime'].items() if key != 'device_uuid'},
        'model': measurement['model'], 'forward_count': measurement['forward_count'], 'forwards': forwards,
        'result': measurement['result'], 'tensor_evidence': measurement['tensor_evidence'],
        'phases': measurement['phases'], 'phase_file_sha256': phase_file_sha,
        'cleanup': measurement['cleanup'], 'model_input_trace_weakrefs_dead': measurement['model_input_trace_weakrefs_dead'],
        'active_memory_leases': measurement['active_memory_leases'],
        'runtime_binding_rechecked': measurement['runtime_binding_rechecked'],
        'host_memory_after_cleanup': measurement['host_memory_after_cleanup'],
        'supervisor': outcome['supervisor'], 'direct_controller_reaped': True,
        'direct_controller_identity_verified': True,
        'direct_controller_start_utc': start['utc'], 'direct_controller_exit_utc': exit_record['utc'],
        'marker_absent_at_post_exit_point': outcome['marker_absent_at_post_exit_point'],
        'attestation_absent_at_post_exit_point': outcome['attestation_absent_at_post_exit_point'],
        'owner_absent_at_management_point': outcome['owner_presence_after_exit']['owner_pid_listed'] is False,
        'nvidia_smi_before': outcome['nvidia_smi_before'], 'nvidia_smi_after': outcome['nvidia_smi_after'],
        'limits': ['independent saved-byte replay does not authenticate execution, source or chronology',
                   'device-free endpoints are samples rather than device-wide peaks',
                   'PyTorch allocator peaks exclude other processes and native allocations',
                   'host and device commitments are not process RSS or native-allocation limits',
                   'direct-child supervision and unlocked OS points do not prove descendant or native-context cleanup'],
    }
    output = strip_private(output)
    check_public(output)
    raw = (json.dumps(output, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with destination.open('xb') as stream:
        stream.write(raw)
        stream.flush()
    return hashlib.sha256(raw).hexdigest()


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('usage: projection RUN_DIR OBSERVER_DIR BYTE_AUDIT NEW_JSON')
    print(project(*sys.argv[1:]))
