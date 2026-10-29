"""CPU-only, invented saved records for the independent timing auditor."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from looking_glass import tap_repeated_protocol as protocol
from tests.test_tap_repeated_protocol import CLOCK, MEMORY, SCHEDULE, TOTAL, CAP, _attempt
from tools import audit_tap_repeated_saved as audit


V12 = 'gpu-tap-latency-repeated-v1.2'
V13 = 'gpu-tap-latency-repeated-v1.3'


def test_frozen_protocol_fixture_survives_checkout_line_endings():
    path = Path(__file__).resolve().parent / 'fixtures' / 'tap_v13_0c147784_protocol.py.txt'
    data = path.read_bytes().replace(b'\r\n', b'\n')
    assert sha(data) == audit.V13_PROTOCOL_SOURCE_SHA
    assert data.replace(b'\n', b'\r\n').replace(b'\r\n', b'\n') == data


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def rows_for(version):
    schedule = protocol.load_schedule(SCHEDULE)
    rows = deepcopy(_attempt(schedule, 0, {})['journal'])
    rows[0]['checks']['run_id'] = f'{version}-session-0'
    # The main-branch helper predates the guarded-interval fields. Give its
    # invented rows the same fixed, internally consistent timing envelope as
    # the v1.3 helper.
    preflight = rows[0]
    preflight.setdefault('memory_snapshot_started_ns', preflight['started_ns'] - 30_000)
    preflight.setdefault('memory_snapshot_ended_ns', preflight['started_ns'] - 20_000)
    preflight.setdefault('gate_lower_ns', preflight['started_ns'] - 10_000)
    preflight.setdefault('gate_upper_ns', preflight['started_ns'] + 10_000)
    for index in range(0, len(rows), 2):
        row, post = rows[index:index + 2]
        post.setdefault('completed_ns', row['ended_ns'] + 10_000)
        if index:
            row.setdefault('completion_ns', row['ended_ns'])
            row.setdefault('guarded_started_ns', row['started_ns'])
            row.setdefault('guarded_ended_ns', post['completed_ns'])
            row.setdefault('guarded_wall_ns', row['guarded_ended_ns'] - row['guarded_started_ns'])
            row.setdefault('postcheck_completed_ns', post['completed_ns'])
            row.setdefault('memory_after', deepcopy(post['memory_after']))
    for row in rows:
        if row.get('trace_run_id') is not None:
            row['trace_run_id'] = f'{version}-session-0'
        if row.get('stage') == 'postcheck':
            row['device_work_complete'] = True
    return schedule, rows


@pytest.fixture
def saved_session(tmp_path, monkeypatch):
    """Build a closed one-session record and an unrelated miniature Git tree."""
    git = shutil.which('git')
    assert git is not None
    serial = 0

    def write(version, *, source_version=None, protocol_suffix=b''):
        nonlocal serial
        serial += 1
        root = tmp_path / f'fixture-{serial}'
        root.mkdir()
        checkout, run = root / 'source', root / 'run'
        checkout.mkdir()
        run.mkdir()
        subprocess.run([git, 'init', '-q', str(checkout)], check=True)
        subprocess.run([git, '-C', str(checkout), 'config', 'core.autocrlf', 'false'], check=True)
        controller = 'looking_glass/tap_repeated_gpu.py'
        protocol_path = 'looking_glass/tap_repeated_protocol.py'
        schedule_path = 'evidence/gpu-tap-latency-repeated-v1/schedule.json'
        # Fixed public source bytes keep this fixture independent of whichever
        # protocol revision happens to be checked out on the integration branch.
        frozen_protocol = (Path(__file__).resolve().parent / 'fixtures'
                           / 'tap_v13_0c147784_protocol.py.txt').read_bytes().replace(b'\r\n', b'\n')
        assert sha(frozen_protocol) == audit.V13_PROTOCOL_SOURCE_SHA
        frozen_oracle = {'__name__': 'frozen_tap_v13_test_oracle'}
        exec(compile(frozen_protocol, 'frozen_tap_v13_protocol', 'exec'), frozen_oracle)
        if (source_version or version) != V13:
            frozen_protocol = frozen_protocol.replace(f'EXPERIMENT_PROTOCOL = {V13!r}'.encode(),
                                                       f'EXPERIMENT_PROTOCOL = {(source_version or version)!r}'.encode())
        contents = {controller: f'_SOURCE_PATHS = ({controller!r}, {protocol_path!r}, {schedule_path!r})\n'.encode(),
                    protocol_path: frozen_protocol + protocol_suffix,
                    schedule_path: SCHEDULE.read_bytes()}
        for name, data in contents.items():
            destination = checkout / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        subprocess.run([git, '-C', str(checkout), 'add', '--', *contents], check=True)
        subprocess.run([git, '-C', str(checkout), '-c', 'user.name=fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '-qm', 'synthetic source'], check=True)
        def git_text(*args):
            return subprocess.check_output([git, '-C', str(checkout), *args]).decode().strip()
        commit, tree = git_text('rev-parse', 'HEAD'), git_text('rev-parse', 'HEAD:')
        if version == V13:
            monkeypatch.setattr(audit, 'V13_SOURCE_COMMIT', commit, raising=False)
        identity = {'run_id': '1' * 32, 'supervisor_run_id': '2' * 32, 'owner_pid': 10,
                    'owner_birth': '1000', 'session_index': 0, 'source_commit': commit,
                    'gpu_uuid': 'GPU-00000000-0000-0000-0000-000000000000'}
        manifest = {'schema': 'tap-repeated-source-manifest-v1', 'source_commit': commit,
                    'run_id': identity['run_id'],
                    'source_files': {name: {'mode': '100644', 'sha256': sha(data)}
                                     for name, data in contents.items()}}
        (run / 'protected-manifest').mkdir()
        manifest_raw = raw(manifest)
        (run / 'protected-manifest' / 'source-manifest.json').write_bytes(manifest_raw)
        started = {'schema': 'tap-repeated-controller-start-v1', 'source_commit': commit,
                   'source_tree': tree, 'run_id': identity['run_id'], 'session': 0,
                   'source_manifest_sha256': sha(manifest_raw)}
        (run / 'started.json').write_bytes(raw(started))
        schedule, rows = rows_for(version)
        clock = dict(CLOCK, implementation='QueryPerformanceCounter()') if version == V13 else CLOCK
        journal = b''.join(raw(row) for row in rows)
        (run / 'journal.jsonl').write_bytes(journal)
        candidate = {'schema': 'tap-repeated-candidate-v1', 'protocol': version, **identity,
            'journal_rows': 86, 'journal_sha256': sha(journal), 'schedule_sha256': audit.SCHEDULE_SHA,
            'whole_run_claims_held': True, 'cuda_total_bytes': TOTAL, 'cap_bytes': CAP,
            'admission_memory': deepcopy(MEMORY), 'memory_before_transfer': deepcopy(MEMORY),
            'free_before_model_transfer': MEMORY['free_bytes'], 'clock': clock,
            'preflight_started_ns': rows[0]['started_ns'], 'workload_ended_ns': rows[-1]['completed_ns'],
            'model': audit.EXPECTED_MODEL, 'runtime': audit.EXPECTED_RUNTIME, 'trace_claim_bytes': 128 * 1024,
            'cuda_device_name': 'invented fixture GPU', 'cuda_compute_capability': [8, 6]}
        candidate_raw = raw(candidate)
        (run / 'candidate.json').write_bytes(candidate_raw)
        final = {'schema': 'tap-repeated-final-v1', **identity, 'candidate_sha256': sha(candidate_raw),
            'unsafe': False, 'synchronized_disposal': True, 'claims_released_after_disposal': True,
            'device_lease_held_at_attestation': True, 'allocated_bytes': 0, 'reserved_bytes': 0,
            'active_memory_leases': 0, 'trace_claim_remaining_bytes': 0}
        (run / 'final.json').write_bytes(raw(final))
        summary = frozen_oracle['summarize_session'](schedule, 0, rows, clock=clock, total_bytes=TOTAL,
                                             cap_bytes=CAP, device='cuda:0', run_protocol=version)
        resources = {'maximum_recorded_reset_peak_allocated_bytes': MEMORY['allocated_bytes'],
            'maximum_recorded_reset_peak_reserved_bytes': MEMORY['reserved_bytes'],
            'minimum_free_bytes_in_admission_pretransfer_and_journal': MEMORY['free_bytes'],
            'guarded_wall_ns_medians': summary['mode_guarded_wall_ns_medians']}
        screen = {'status': 'no_contamination_observed', 'availability': 'complete',
            'contamination_reasons': [], 'unavailable_reasons': [], 'before_utilization_percent': [],
            'after_utilization_percent': [], 'telemetry_samples': [], 'foreign_process_count': 0}
        result = {'schema': 'tap-repeated-provisional-session-v1', 'protocol': version,
            'status': 'completed_provisional', 'source_commit': commit, 'source_tree': tree, 'session': 0,
            'slot': {'session': 0, 'state': 'complete', 'screen': screen, 'summary': summary,
                     'device_totals': {'cuda_bytes': TOTAL, 'nvidia_smi_bytes': TOTAL,
                                       'maximum_difference_bytes': 2 * 1024 ** 2}},
            'device': {'name': 'invented fixture GPU', 'driver_point_observation': '1.2',
                       'compute_capability': [8, 6]}, 'resources': resources,
            'coordination': {'gpu_slot': 'operator_coordinated', 'cpu_context': 'unknown'}, 'limitations': []}
        public = root / 'result.json'
        public.write_bytes(raw(result))
        if version == V13:
            grant_identity = {key: identity[key] for key in ('session_index', 'owner_pid', 'owner_birth',
                                                               'supervisor_run_id', 'gpu_uuid')}
            grant_identity.update({'challenge': '3' * 64, 'cuda_total_bytes': TOTAL,
                'activity_smi_total_bytes': TOTAL, 'clock': clock,
                'runtime': {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}})
            grant = {'schema': 'tap-repeated-start-grant-v1', **grant_identity,
                'ready_ns': 1_800_000_000, 'received_ns': 1_900_000_000,
                'initial_oldest_query_started_ns': 2_000_000_000,
                'initial_latest_query_ended_ns': 2_100_000_000,
                'initial_process_started_ns': 2_150_000_000,
                'initial_process_ended_ns': 2_200_000_000,
                'issued_ns': 2_300_000_000, 'expires_ns': 3_300_000_000}
            finished = {'schema': 'tap-repeated-workload-finished-v1', **grant_identity,
                'grant_seen_ns': 2_400_000_000, 'preflight_started_ns': rows[0]['started_ns'],
                'workload_ended_ns': rows[-1]['completed_ns']}
            issued = rows[-1]['completed_ns'] + 1000
            cleanup = {'schema': 'tap-repeated-cleanup-grant-v1', **grant_identity,
                'workload_ended_ns': rows[-1]['completed_ns'], 'issued_ns': issued,
                'expires_ns': issued + 5_000_000_000, 'screen_availability': 'complete'}
            for name, value in (('start-grant.json', grant), ('finished.json', finished),
                                ('cleanup-grant.json', cleanup)):
                (run / name).write_bytes(raw(value))
        return {'run_dir': run, 'public_path': public, 'checkout': checkout, 'git_exe': Path(git),
                'controller_exit': 0}
    return write


def test_v12_default_keeps_prior_single_session_semantics(saved_session):
    report = audit.audit(**saved_session(V12))
    assert report['schema'] == 'tap-v12-independent-timing-audit-v1'
    assert report['status'] == 'passed' and report['measured_samples'] == 36
    assert report['median_gradient_over_direct'] == 1.5
    assert report['median_direct_over_none'] == 1.5


def test_v13_requires_explicit_version_and_preserves_per_round_statistics(saved_session):
    values = saved_session(V13)
    with pytest.raises(ValueError):
        audit.audit(**values)
    report = audit.audit(**values, expected_protocol=V13)
    assert report['schema'] == 'tap-v13-independent-timing-audit-v1'
    assert report['protocol'] == V13 and report['status'] == 'passed'
    assert report['measured_samples'] == 36 and report['median_gradient_over_direct'] == 1.5
    assert report['median_direct_over_none'] == 1.5 and report['pooling'] == 'single_session_only'
    grant = json.loads((values['run_dir'] / 'start-grant.json').read_bytes())
    finished = json.loads((values['run_dir'] / 'finished.json').read_bytes())
    assert grant['expires_ns'] - grant['issued_ns'] == 1_000_000_000
    assert finished['preflight_started_ns'] - grant['issued_ns'] > 500_000_000


def test_v13_cannot_accept_prior_version_or_a_pooled_public_result(saved_session):
    values = saved_session(V12)
    with pytest.raises(ValueError):
        audit.audit(**values, expected_protocol=V13)
    values = saved_session(V13)
    public = json.loads(values['public_path'].read_bytes())
    public['aggregate'] = {'sessions': [0, 1]}
    values['public_path'].write_bytes(raw(public))
    with pytest.raises(ValueError):
        audit.audit(**values, expected_protocol=V13)


def test_v13_rejects_a_prior_version_public_label(saved_session):
    values = saved_session(V13)
    public = json.loads(values['public_path'].read_bytes())
    public['protocol'] = V12
    values['public_path'].write_bytes(raw(public))
    with pytest.raises(ValueError):
        audit.audit(**values, expected_protocol=V13)


@pytest.mark.parametrize('identity_kind', ['preflight', 'trace'])
def test_v13_rejects_v12_identity_even_when_the_journal_digest_is_updated(saved_session, identity_kind):
    values = saved_session(V13)
    journal = values['run_dir'] / 'journal.jsonl'
    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    if identity_kind == 'preflight':
        rows[0]['checks']['run_id'] = f'{V12}-session-0'
    else:
        trace = next(row for row in rows if row.get('trace_run_id'))
        trace['trace_run_id'] = f'{V12}-session-0'
    changed = b''.join(raw(row) for row in rows)
    journal.write_bytes(changed)
    candidate_path = values['run_dir'] / 'candidate.json'
    candidate = json.loads(candidate_path.read_bytes())
    candidate['journal_sha256'] = sha(changed)
    candidate_raw = raw(candidate)
    candidate_path.write_bytes(candidate_raw)
    final_path = values['run_dir'] / 'final.json'
    final = json.loads(final_path.read_bytes())
    final['candidate_sha256'] = sha(candidate_raw)
    final_path.write_bytes(raw(final))
    with pytest.raises(ValueError):
        audit.audit(**values, expected_protocol=V13)


def test_v13_rejects_a_source_tree_declaring_the_v12_protocol(saved_session):
    with pytest.raises(ValueError):
        audit.audit(**saved_session(V13, source_version=V12), expected_protocol=V13)


def test_cli_defaults_to_v12_and_requires_explicit_v13(saved_session, monkeypatch, capsys):
    values = saved_session(V13)
    arguments = ['auditor', '--run-dir', str(values['run_dir']), '--public-result', str(values['public_path']),
                 '--checkout', str(values['checkout']), '--git', str(values['git_exe']), '--controller-exit', '0']
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(ValueError):
        audit.main()
    monkeypatch.setattr(sys, 'argv', arguments + ['--protocol', V13])
    audit.main()
    output = json.loads(capsys.readouterr().out)
    assert output['protocol'] == V13 and output['pooling'] == 'single_session_only'


def test_v13_rejects_another_self_consistent_execution_commit(saved_session, monkeypatch):
    values = saved_session(V13)
    monkeypatch.setattr(audit, 'V13_SOURCE_COMMIT', 'f' * 40, raising=False)
    with pytest.raises(ValueError):
        audit.audit(**values, expected_protocol=V13)


def test_v13_rejects_a_changed_frozen_protocol_source(saved_session):
    values = saved_session(V13, protocol_suffix=b'\n# invented source revision\n')
    with pytest.raises(ValueError):
        audit.audit(**values, expected_protocol=V13)


@pytest.mark.parametrize('damage', ['too_long', 'stale_activity', 'wrong_finish', 'wrong_cleanup',
                                    'missing_grant', 'missing_finish', 'missing_cleanup'])
def test_v13_rejects_invalid_or_missing_saved_grant_evidence(saved_session, damage):
    values = saved_session(V13)
    root = values['run_dir']
    if damage.startswith('missing_'):
        name = {'missing_grant': 'start-grant.json', 'missing_finish': 'finished.json',
                'missing_cleanup': 'cleanup-grant.json'}[damage]
        (root / name).unlink()
    else:
        name = {'too_long': 'start-grant.json', 'stale_activity': 'start-grant.json',
                'wrong_finish': 'finished.json', 'wrong_cleanup': 'cleanup-grant.json'}[damage]
        path = root / name
        value = json.loads(path.read_bytes())
        if damage == 'too_long': value['expires_ns'] = value['issued_ns'] + 1_000_000_001
        if damage == 'stale_activity': value['initial_oldest_query_started_ns'] = 900_000_000
        if damage == 'wrong_finish': value['preflight_started_ns'] += 1
        if damage == 'wrong_cleanup': value['owner_pid'] += 1
        path.write_bytes(raw(value))
    with pytest.raises((ValueError, OSError)):
        audit.audit(**values, expected_protocol=V13)


def test_incomplete_v13_does_not_claim_a_verified_version_or_a_timing_result(tmp_path):
    run, result = tmp_path / 'run', tmp_path / 'result.json'
    run.mkdir()
    result.write_bytes(raw({'schema': 'tap-repeated-controller-failure-v1', 'status': 'incomplete'}))
    report = audit.audit_incomplete(run, result, 1, expected_protocol=V13)
    assert report['status'] == 'incomplete' and report['expected_protocol'] == V13
    assert report['protocol_verification'] == 'not_established' and 'median_gradient_over_direct' not in report


@pytest.mark.parametrize('untyped_exit', [True, '1', None])
def test_incomplete_v13_requires_an_integer_exit_observation(tmp_path, untyped_exit):
    run, result = tmp_path / 'run', tmp_path / 'result.json'
    run.mkdir()
    result.write_bytes(raw({'status': 'incomplete'}))
    with pytest.raises(ValueError):
        audit.audit_incomplete(run, result, untyped_exit, expected_protocol=V13)


@pytest.mark.parametrize('contradiction', ['public', 'candidate', 'preflight', 'trace',
                                            'public_source', 'candidate_source', 'started_source'])
def test_incomplete_v13_rejects_available_prior_version_records(tmp_path, contradiction):
    run, result = tmp_path / 'run', tmp_path / 'result.json'
    run.mkdir()
    public = {'schema': 'tap-repeated-controller-failure-v1', 'status': 'incomplete'}
    if contradiction == 'public': public['protocol'] = V12
    if contradiction == 'public_source': public['source_commit'] = 'a' * 40
    result.write_bytes(raw(public))
    if contradiction == 'candidate': (run / 'candidate.json').write_bytes(raw({'protocol': V12}))
    if contradiction == 'candidate_source': (run / 'candidate.json').write_bytes(raw({'source_commit': 'a' * 40}))
    if contradiction == 'started_source': (run / 'started.json').write_bytes(raw({'source_commit': 'a' * 40}))
    if contradiction in ('preflight', 'trace'):
        row = ({'stage': 'preflight', 'checks': {'run_id': f'{V12}-session-0'}} if contradiction == 'preflight'
               else {'stage': 'measured', 'trace_run_id': f'{V12}-session-0'})
        (run / 'journal.jsonl').write_bytes(raw(row))
    with pytest.raises(ValueError):
        audit.audit_incomplete(run, result, 1, expected_protocol=V13)
