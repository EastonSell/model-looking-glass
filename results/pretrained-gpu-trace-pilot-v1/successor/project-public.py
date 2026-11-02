"""Create a path- and process-redacted projection of the closed trace candidate.

The projection retains hashes and observed counters, not tensor payloads. It
does not independently establish the original execution or recompute bits.
"""

import hashlib
import json
from pathlib import Path
import sys


SOURCE = '3499aa4ec97ce4625c0ac3b6bb9c44413dedb977'
EXPECTED = {
    'candidate.json': 'df1545e35215d8585ddc89e7ef0c29e11a9275ef98f043060eff110a72398ca7',
    'outcome.json': 'e88c74c201787303942535bb4b647e0d164d6b413f9fed5961929e968d72c3e9',
    'final.json': '970276843d4ac9b1410c62001c11007fdf2add0464c861578000ec5a34fbe134',
    'exit.json': '989d756c493ce60141dc5c915fac5e02d8a917c3d228e22ed386cf3bbe227a48',
}


def read(path, expected=None):
    data = Path(path).read_bytes()
    if len(data) > 1024 ** 2:
        raise ValueError('saved diagnostic is too large')
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('saved diagnostic digest differs')
    return json.loads(data), digest


def project(run, observer, destination):
    run, observer, destination = map(Path, (run, observer, destination))
    candidate, _ = read(run / 'candidate.json', EXPECTED['candidate.json'])
    outcome, _ = read(run / 'outcome.json', EXPECTED['outcome.json'])
    final, _ = read(run / 'final.json', EXPECTED['final.json'])
    exit_record, _ = read(observer / 'exit.json', EXPECTED['exit.json'])
    started, started_sha = read(run / 'started.json')
    setup, setup_sha = read(run / 'input-setup.json')
    if (candidate.get('source_commit') != SOURCE or outcome.get('source_commit') != SOURCE
            or final.get('source_commit') != SOURCE or exit_record.get('source_commit') != SOURCE
            or started.get('source_commit') != SOURCE or outcome.get('status') != 'complete_candidate'
            or outcome.get('candidate_sha256') != EXPECTED['candidate.json']
            or final.get('candidate_sha256') != EXPECTED['candidate.json']
            or outcome.get('supervisor') != {'exit_code': 0, 'clean': True, 'timed_out': False}
            or exit_record.get('exit_code') != 0 or exit_record.get('reaped') is not True
            or exit_record.get('controller_identity_verified') is not True):
        raise ValueError('closed candidate or direct-exit identity differs')
    measurement = candidate['measurement']
    runtime = {key: value for key, value in measurement['runtime'].items() if key != 'device_uuid'}
    phases = measurement['phases']
    if len(phases) != 7 or measurement['forward_count'] != 10:
        raise ValueError('fixed phase inventory differs')
    forwards = []
    for index in range(1, 11):
        value, digest = read(run / f'forward-{index:03d}.json')
        forwards.append({'record': value, 'file_sha256': digest})
    phase_file_sha = []
    for index in range(1, 8):
        start, start_sha = read(run / f'phase-{index:02d}-start.json')
        value, digest = read(run / f'phase-{index:02d}.json')
        if value != phases[index - 1] or start['before'] != value['before']:
            raise ValueError('phase files differ from candidate')
        phase_file_sha.append({'start': start_sha, 'completed': digest})
    output = {'schema': 'pretrained-gpu-trace-pilot-public-projection-v1',
        'source_commit': SOURCE, 'source_sha256': started['source_sha256'],
        'private_record_sha256': {**EXPECTED, 'started.json': started_sha, 'input-setup.json': setup_sha},
        'input': setup, 'runtime': runtime, 'model': measurement['model'],
        'forward_count': measurement['forward_count'], 'forwards': forwards,
        'result': measurement['result'], 'phases': phases, 'phase_file_sha256': phase_file_sha,
        'cleanup': measurement['cleanup'], 'model_input_trace_weakrefs_dead': measurement['model_input_trace_weakrefs_dead'],
        'active_memory_leases': measurement['active_memory_leases'],
        'runtime_binding_rechecked': measurement['runtime_binding_rechecked'],
        'host_memory_after_cleanup': measurement['host_memory_after_cleanup'],
        'supervisor': outcome['supervisor'], 'direct_controller_reaped': True,
        'direct_controller_identity_verified': True,
        'marker_absent_at_post_exit_point': outcome['marker_absent_at_post_exit_point'],
        'attestation_absent_at_post_exit_point': outcome['attestation_absent_at_post_exit_point'],
        'owner_absent_at_management_point': outcome['owner_presence_after_exit']['owner_pid_listed'] is False,
        'nvidia_smi_before': outcome['nvidia_smi_before'], 'nvidia_smi_after': outcome['nvidia_smi_after'],
        'limits': ['hash-only saved records do not independently replay tensor comparisons',
                   'device-free endpoints are samples rather than device-wide peaks',
                   'PyTorch allocator peaks exclude other processes and native allocations',
                   'host and device commitments are not process RSS or native-allocation limits',
                   'direct-child supervision and unlocked OS points do not prove descendant or native-context cleanup']}
    raw = (json.dumps(output, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    for forbidden in (b'GPU-', b'"device_uuid"', b'"pid"', b'"parent_pid"', b'\\\\', b':\\\\'):
        if forbidden in raw:
            raise ValueError('projection contains a private identifier or path')
    with destination.open('xb') as stream:
        stream.write(raw)
        stream.flush()
    return hashlib.sha256(raw).hexdigest()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('usage: projection RUN_DIR OBSERVER_DIR NEW_JSON')
    print(project(*sys.argv[1:]))
