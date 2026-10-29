"""Synthetic CPU/loopback checks; no pretrained model or GPU is used."""

from copy import deepcopy
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from looking_glass import cpu_shared_load as load
from looking_glass.batched_events import _EVENT_FIELDS, assemble_window
from looking_glass.batched_throughput import VERSION, freeze_protocol
from looking_glass.ipc import IPCClient, IPCHandle, IPCRejected


@pytest.fixture(scope='module')
def actual_window():
    plan, digest = load.make_plan(('alpha', 'beta', 'gamma'), jobs_per_team=2,
                                  model_hold_us=5_000, start_spacing_us=1_000)
    with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA was queried')):
        return load.run_cpu_window(plan, expected_plan_sha256=digest)


def test_real_authenticated_loopback_window_retains_actual_evidence(actual_window):
    report = actual_window
    assert report['owner'] == {'work_drained': True, 'model_and_lease_disposal_attested': False,
                               'synthetic_device_telemetry': True,
                               'supervised_device_owner': False, 'resource_profile_verified': False}
    assert report['clock']['name'] == 'perf_counter_ns'
    assert report['clock']['monotonic'] is True
    assert report['recording']['events_complete'] is True
    assert report['summary']['event_audit_errors'] == []
    assert report['summary']['queue_observations_valid'] is True
    assert report['summary']['states']['succeeded'] == 6
    assert report['summary']['attempted'] == report['summary']['accepted_observed'] == 6
    assert report['summary']['elapsed_ns'] == report['boundaries']['end_ns'] - report['boundaries']['start_ns']
    assert len({(row['broker_id'], row['broker_job_id']) for row in report['samples']}) == 6
    assert len({row['broker_id'] for row in report['samples']}) == 1
    for team, stats in report['summary']['teams'].items():
        assert stats['planned'] == stats['attempted'] == stats['states']['succeeded'] == 2
        assert stats['successful_jobs_per_second'] == pytest.approx(2e9 / report['summary']['elapsed_ns'])
        rows = [row for row in report['samples'] if row['team_id'] == team]
        assert rows[0]['handled_ns'] <= rows[1]['attempt_ns']
        for row in rows:
            assert row['attempt_ns'] <= row['client_receipt_ns'] <= row['handled_ns']
            assert row['client_cancellation_requested'] is False
            assert row['broker_cancellation_decision'] == 'unobserved'
            assert type(row['output_sha256']) is str
    events = report['recording']['events']
    assert all(set(event) == set(_EVENT_FIELDS[event['kind']]) for event in events)
    assert len([event for event in events if event['kind'] == 'forward_entry']) == 6
    assert len([event for event in events if event['kind'] == 'forward_finished']) == 6
    assert not [event for event in events if event['kind'] == 'forward_uncertain']


def test_real_cpu_events_project_to_v1_with_fabricated_test_only_metadata(actual_window):
    # Other protocol digests and owner label are invented for this schema test;
    # this does not certify a grouped profile, real owner identity or GPU work.
    plan, report = actual_window['plan'], actual_window
    protocol, digest = freeze_protocol({
        'version': VERSION, 'model_sha256': plan['model_sha256'], 'checkpoint_sha256': '1' * 64,
        'source_sha256': '2' * 64, 'runtime_sha256': '3' * 64, 'capture_policy': 'none',
        'serial_policy_sha256': '4' * 64, 'grouped_policy_sha256': '5' * 64,
        'profile_sha256_by_rows': {'1': plan['profile_sha256'], '2': '6' * 64},
        'heldout_evidence_sha256': '7' * 64, 'input_shape': [1, 2],
        'output_shape': [1, 2, 1], 'output_dtype': 'float32', 'output_cap_bytes': 64,
        'atol': 0.0, 'rtol': 0.0,
        'jobs': [{**{key: job[key] for key in ('job_id', 'team_id', 'input_sha256', 'request_sha256')},
                  'seed': plan['seed']} for job in plan['jobs']],
    })
    context = {'mode': 'serial', 'window_id': 'synthetic-window', 'owner_session_id': 'synthetic-owner',
               'model_sha256': plan['model_sha256'], 'runtime_sha256': '3' * 64,
               'policy_sha256': '4' * 64, 'clock': 'perf_counter_ns',
               'clock_resolution_ns': report['clock']['nominal_resolution_ns'],
               'events_complete': report['recording']['events_complete']}
    window = assemble_window(protocol, expected_protocol_sha256=digest, context=context,
                             events=report['recording']['events'])
    assert len(window['attempts']) == len(window['calls']) == 6
    assert all(row['state'] == 'succeeded' for row in window['attempts'])
    assert all(call['input_shape'] == [1, 2] and call['synchronized'] for call in window['calls'])


@pytest.mark.parametrize('change', [
    lambda plan: plan['teams'].reverse(),
    lambda plan: plan['jobs'][0]['input_ids'].__setitem__(0, 99),
    lambda plan: plan.__setitem__('model_hold_us', 2),
])
def test_frozen_plan_digest_rejects_mutation_before_owner_start(change):
    plan, digest = load.make_plan(('a', 'b'))
    change(plan)
    with pytest.raises(ValueError, match='digest'):
        load.run_cpu_window(plan, expected_plan_sha256=digest)
    assert load._active_owner is None


@pytest.mark.parametrize('teams,kwargs', [
    (('a', 'a'), {}), (('a',), {}), (('a', 'b'), {'jobs_per_team': True}),
    (('a', 'b'), {'model_hold_us': 100_001}), (('a', 'b'), {'max_pending': 3}),
    (('a', 'b'), {'seed': -1}),
])
def test_plan_bounds(teams, kwargs):
    with pytest.raises(ValueError):
        load.make_plan(teams, **kwargs)


@pytest.mark.parametrize('monotonic,adjustable', [(False, False), (True, True)])
def test_clock_must_be_monotonic_and_nonadjustable_before_start(monotonic, adjustable):
    plan, digest = load.make_plan(('a', 'b'))
    info = SimpleNamespace(monotonic=monotonic, adjustable=adjustable, resolution=1e-9)
    with patch.object(load.time, 'get_clock_info', return_value=info):
        with pytest.raises(ValueError, match='monotonic and non-adjustable'):
            load.run_cpu_window(plan, expected_plan_sha256=digest)
    assert load._active_owner is None


@pytest.mark.parametrize('resolution', [0, -1, float('inf'), float('nan')])
def test_clock_requires_a_finite_positive_resolution_before_start(resolution):
    plan, digest = load.make_plan(('a', 'b'))
    info = SimpleNamespace(monotonic=True, adjustable=False, resolution=resolution)
    with patch.object(load.time, 'get_clock_info', return_value=info):
        with pytest.raises(ValueError, match='finite positive nominal resolution'):
            load.run_cpu_window(plan, expected_plan_sha256=digest)
    assert load._active_owner is None


def test_recording_loss_and_swapped_receipt_invalidate_queue_evidence(actual_window):
    for change in ('lost_event', 'loss_counter', 'wrong_count', 'swapped_receipt',
                   'duplicate_selection', 'post_drain'):
        report = deepcopy(actual_window)
        events = report['recording']['events']
        if change == 'lost_event':
            report['recording']['events_complete'] = False
        elif change == 'loss_counter':
            report['recording']['lost_capacity'] = 1
        elif change == 'wrong_count':
            report['recording']['recorded'] += 1
        elif change == 'swapped_receipt':
            report['samples'][0]['broker_job_id'] = report['samples'][1]['broker_job_id']
        elif change == 'duplicate_selection':
            events.append(deepcopy(next(event for event in events if event['kind'] == 'selected')))
        else:
            event = next(event for event in events if event['kind'] == 'broker_terminal')
            event['time_ns'] = report['boundaries']['owner_drain_ns'] + 1
        summary = load._summarize(report['plan'], report['samples'], report['recording'],
                                  report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                                  True, report['clock']['nominal_resolution_ns'])
        assert summary['event_audit_errors'], change
        assert summary['queue_observations_valid'] is False
        assert summary['queue']['jobs'] == []


def test_failed_forward_cannot_finish_after_its_broker_terminal(actual_window):
    report = deepcopy(actual_window)
    events = report['recording']['events']
    terminal = max((event for event in events if event['kind'] == 'broker_terminal'),
                   key=lambda event: event['time_ns'])
    job_id = terminal['job_id']
    row = next(row for row in report['samples'] if row['job_id'] == job_id)
    returned = next(event for event in events if event['kind'] == 'returned' and event['job_id'] == job_id)
    entry = next(event for event in events if event['kind'] == 'forward_entry'
                 and event['members'][0]['job_id'] == job_id)
    finish = next(event for event in events if event['kind'] == 'forward_finished'
                  and event['call_id'] == entry['call_id'])
    row.update(state='failed', error='injected', output_sha256=None)
    returned.update(state='failed', error='injected', output_sha256=None)
    terminal['state'] = 'failed'
    finish.update(time_ns=terminal['time_ns'] + 1, state='failed', error='injected')
    summary = load._summarize(report['plan'], report['samples'], report['recording'],
                              report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                              True, report['clock']['nominal_resolution_ns'])
    assert 'forward_and_terminal_order_invalid' in summary['event_audit_errors']
    assert summary['queue_observations_valid'] is False


@pytest.mark.parametrize('state,synchronized', [('nonsense', True), ('failed', False)])
def test_every_cpu_forward_finish_requires_a_known_state_and_completion(actual_window, state, synchronized):
    report = deepcopy(actual_window)
    events = report['recording']['events']
    terminal = max((event for event in events if event['kind'] == 'broker_terminal'),
                   key=lambda event: event['time_ns'])
    row = next(row for row in report['samples'] if row['job_id'] == terminal['job_id'])
    returned = next(event for event in events if event['kind'] == 'returned' and event['job_id'] == row['job_id'])
    entry = next(event for event in events if event['kind'] == 'forward_entry'
                 and event['members'][0]['job_id'] == row['job_id'])
    finish = next(event for event in events if event['kind'] == 'forward_finished'
                  and event['call_id'] == entry['call_id'])
    row.update(state='failed', error='injected', output_sha256=None)
    returned.update(state='failed', error='injected', output_sha256=None)
    terminal['state'] = 'failed'
    finish.update(state=state, synchronized=synchronized, error='injected')
    summary = load._summarize(report['plan'], report['samples'], report['recording'],
                              report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                              True, report['clock']['nominal_resolution_ns'])
    assert 'invalid_cpu_forward_finish' in summary['event_audit_errors']
    assert summary['queue_observations_valid'] is False


@pytest.mark.parametrize('clear', [('broker_id', 'broker_job_id'), ('broker_id',), ('broker_job_id',)])
def test_terminal_job_requires_complete_exact_broker_identity(actual_window, clear):
    report = deepcopy(actual_window)
    row = report['samples'][0]
    events = report['recording']['events']
    ready = next(event for event in events if event['kind'] == 'ready' and event['job_id'] == row['job_id'])
    entry = next(event for event in events if event['kind'] == 'forward_entry'
                 and event['members'][0]['job_id'] == row['job_id'])
    finish = next(event for event in events if event['kind'] == 'forward_finished'
                  and event['call_id'] == entry['call_id'])
    for key in clear:
        row[key] = ready[key] = entry['members'][0][key] = None
    call_id = f"lg-{row['broker_id']}-{row['broker_job_id']}"
    entry['call_id'] = entry['group_id'] = finish['call_id'] = call_id
    summary = load._summarize(report['plan'], report['samples'], report['recording'],
                              report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                              True, report['clock']['nominal_resolution_ns'])
    assert {'missing_broker_identity', 'invalid_broker_identity'} & set(summary['event_audit_errors'])
    assert summary['queue_observations_valid'] is False


@pytest.mark.parametrize('digest', [None, 'invalid', 'b' * 64])
def test_success_requires_the_frozen_synthetic_output_digest(actual_window, digest):
    report = deepcopy(actual_window)
    row = report['samples'][0]
    returned = next(event for event in report['recording']['events']
                    if event['kind'] == 'returned' and event['job_id'] == row['job_id'])
    row['output_sha256'] = returned['output_sha256'] = digest
    summary = load._summarize(report['plan'], report['samples'], report['recording'],
                              report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                              True, report['clock']['nominal_resolution_ns'])
    assert 'output_digest_mismatch' in summary['event_audit_errors']
    assert summary['queue_observations_valid'] is False


def test_unattempted_sample_cannot_retain_attempt_or_broker_observations(actual_window):
    report = deepcopy(actual_window)
    row = report['samples'][-1]
    job_id = row['job_id']
    events = report['recording']['events']
    entry = next(event for event in events if event['kind'] == 'forward_entry'
                 and event['members'][0]['job_id'] == job_id)
    call_id = entry['call_id']
    report['recording']['events'] = [event for event in events
                                     if event.get('job_id') != job_id and event.get('call_id') != call_id]
    row.update(state='unattempted', output_sha256=None)
    summary = load._summarize(report['plan'], report['samples'], report['recording'],
                              report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                              True, report['clock']['nominal_resolution_ns'])
    assert 'unattempted_sample_has_observations' in summary['event_audit_errors']
    assert summary['queue_observations_valid'] is False


def test_missing_and_duplicate_samples_do_not_reduce_planned_count(actual_window):
    for case in ('missing', 'duplicate'):
        report = deepcopy(actual_window)
        if case == 'missing':
            removed = report['samples'].pop()
        else:
            removed = report['samples'][-1]
            report['samples'].append(deepcopy(removed))
        summary = load._summarize(report['plan'], report['samples'], report['recording'],
                                  report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                                  True, report['clock']['nominal_resolution_ns'])
        assert summary['planned'] == 6
        assert summary['teams'][removed['team_id']]['planned'] == 2
        assert summary['states']['unattempted'] == 0
        assert summary['states']['succeeded'] == 5
        assert summary['missing_samples'] == (1 if case == 'missing' else 0)
        assert summary['ambiguous_samples'] == (1 if case == 'duplicate' else 0)
        assert summary['queue_observations_valid'] is False


@pytest.mark.parametrize('field,value', [('client_cancellation_requested', True),
                                        ('broker_cancellation_decision', 'accepted')])
def test_no_sample_can_claim_an_unobserved_cancellation_decision(actual_window, field, value):
    report = deepcopy(actual_window)
    report['samples'][0][field] = value
    summary = load._summarize(report['plan'], report['samples'], report['recording'],
                              report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                              True, report['clock']['nominal_resolution_ns'])
    assert 'unsupported_cancellation_observation' in summary['event_audit_errors']
    assert summary['queue_observations_valid'] is False


def test_client_rejection_is_not_counted_as_accepted_and_peer_can_finish():
    plan, digest = load.make_plan(('a', 'b'), jobs_per_team=1)
    original = IPCClient.submit

    def submit(client, *, team_id, **kwargs):
        if team_id == 'a':
            raise IPCRejected('queue_full')
        return original(client, team_id=team_id, **kwargs)

    with patch.object(IPCClient, 'submit', submit):
        report = load.run_cpu_window(plan, expected_plan_sha256=digest)
    assert report['summary']['states']['rejected'] == 1
    assert report['summary']['states']['succeeded'] == 1
    assert report['summary']['accepted_observed'] == 1
    assert report['summary']['event_audit_errors'] == []
    rejected = next(row for row in report['samples'] if row['state'] == 'rejected')
    assert rejected['broker_id'] is rejected['broker_job_id'] is None
    rejected['client_receipt_ns'] = None
    summary = load._summarize(report['plan'], report['samples'], report['recording'],
                              report['boundaries']['start_ns'], report['boundaries']['end_ns'],
                              True, report['clock']['nominal_resolution_ns'])
    assert 'client_receipt_missing' in summary['event_audit_errors']
    assert summary['queue_observations_valid'] is False


def test_client_timeout_does_not_create_a_terminal_or_cancellation_decision():
    plan, digest = load.make_plan(('a', 'b'), jobs_per_team=2, model_hold_us=20_000)
    original = IPCHandle.wait

    def wait(handle, *args, **kwargs):
        if handle.team_id == 'a':
            raise TimeoutError('simulated lost client response')
        return original(handle, *args, **kwargs)

    with patch.object(IPCHandle, 'wait', wait):
        report = load.run_cpu_window(plan, expected_plan_sha256=digest)
    rows = [row for row in report['samples'] if row['team_id'] == 'a']
    assert [row['state'] for row in rows] == ['unresolved', 'unattempted']
    assert rows[0]['client_receipt_ns'] is None
    assert rows[0]['broker_cancellation_decision'] == 'unobserved'
    assert report['recording']['events_complete'] is False
    assert report['summary']['queue_observations_valid'] is False
    assert all(event['job_id'] != rows[0]['job_id'] for event in report['recording']['events']
               if event['kind'] == 'returned')


def test_recorder_loss_does_not_change_real_service_results():
    plan, digest = load.make_plan(('a', 'b'), jobs_per_team=1)
    original = load.BatchedEventRecorder.record

    def record(recorder, event):
        if event['kind'] == 'selected':
            recorder.mark_incomplete()
            return False
        return original(recorder, event)

    with patch.object(load.BatchedEventRecorder, 'record', record):
        report = load.run_cpu_window(plan, expected_plan_sha256=digest)
    assert report['summary']['states']['succeeded'] == 2
    assert report['recording']['events_complete'] is False
    assert report['summary']['queue_observations_valid'] is False
    assert report['summary']['queue']['jobs'] == []


def test_strict_queue_interventions_and_shared_denominator_on_fabricated_events():
    plan, _ = load.make_plan(('a', 'b', 'c'), jobs_per_team=1)
    broker_id = 'a' * 32
    samples, events = [], [
        {'kind': 'window_start', 'time_ns': 100},
        {'kind': 'owner_drain', 'time_ns': 400, 'drained': True},
        {'kind': 'window_end', 'time_ns': 500},
    ]
    for index, job in enumerate(plan['jobs']):
        job_id, team = job['job_id'], job['team_id']
        native_id, ready_ns, selection_ns = index + 1, 120, 140 + index * 60
        entry_ns, finish_ns = selection_ns + 1, selection_ns + 10
        terminal_ns, receipt_ns, handled_ns = selection_ns + 15, selection_ns + 16, selection_ns + 20
        call_id = f'lg-{broker_id}-{native_id}'
        row = load._sample(job)
        row.update(state='succeeded', attempt_ns=110, client_receipt_ns=receipt_ns,
                   handled_ns=handled_ns, broker_id=broker_id, broker_job_id=native_id,
                   output_sha256=job['expected_output_sha256'])
        samples.append(row)
        events.extend([
            {'kind': 'attempt', 'time_ns': 110, 'job_id': job_id, 'team_id': team,
             'input_sha256': job['input_sha256'], 'request_sha256': job['request_sha256']},
            {'kind': 'ready', 'time_ns': ready_ns, 'job_id': job_id, 'team_id': team,
             'broker_id': broker_id, 'broker_job_id': native_id},
            {'kind': 'selected', 'time_ns': selection_ns, 'job_id': job_id},
            {'kind': 'forward_entry', 'time_ns': entry_ns, 'call_id': call_id, 'group_id': call_id,
             'profile_sha256': plan['profile_sha256'], 'input_shape': [1, 2],
             'members': [{'job_id': job_id, 'team_id': team, 'broker_id': broker_id,
                          'broker_job_id': native_id, 'row_offset': 0}]},
            {'kind': 'forward_finished', 'time_ns': finish_ns, 'call_id': call_id,
             'state': 'succeeded', 'synchronized': True, 'error': None},
            {'kind': 'broker_terminal', 'time_ns': terminal_ns, 'job_id': job_id, 'state': 'succeeded'},
            {'kind': 'returned', 'time_ns': handled_ns, 'job_id': job_id, 'state': 'succeeded',
             'error': None, 'output_sha256': job['expected_output_sha256'], 'cancellation_requested': False,
             'cancellation_requested_ns': None, 'cancellation_accepted': None,
             'cancellation_decision_ns': None},
        ])
    recording = {'events_complete': True, 'events': events, 'recorded': len(events),
                 'lost_capacity': 0, 'lost_invalid': 0, 'lost_internal': 0,
                 'late_records': 0, 'explicit_faults': 0, 'wire_bytes_upper_bound': 1000,
                 'max_wire_bytes': 1_048_576, 'max_events': 4096}
    report = load._summarize(plan, samples, recording,
                             100, 500, True, 1)
    assert report['event_audit_errors'] == []
    assert report['elapsed_ns'] == 400
    assert report['successful_jobs_per_second'] == pytest.approx(3 * 1e9 / 400)
    assert {row['team_id']: row['strictly_intervening_selections'] for row in report['queue']['jobs']} == {
        'a': 0, 'b': 1, 'c': 2}
    assert report['nominal_resolution_screen']['queue_wait_intervals_observed'] == 3
    assert report['nominal_resolution_screen']['queue_wait_intervals_short'] == 2


def test_cli_freezes_digest_and_refuses_overwrite_or_modified_plan(tmp_path, capsys):
    plan_file, report_file = tmp_path / 'plan.json', tmp_path / 'report.json'
    load.main(['plan', '--teams', 'a', 'b', '--jobs-per-team', '1', '--output', str(plan_file)])
    digest = capsys.readouterr().out.strip()
    envelope = json.loads(plan_file.read_text(encoding='utf-8'))
    assert envelope['plan_sha256'] == digest
    with pytest.raises(FileExistsError):
        load.main(['plan', '--teams', 'a', 'b', '--output', str(plan_file)])
    envelope['plan']['jobs'][0]['input_ids'][0] = 123
    plan_file.write_text(json.dumps(envelope), encoding='utf-8')
    with pytest.raises(ValueError):
        load.main(['run', '--plan', str(plan_file), '--sha256', digest, '--output', str(report_file)])
    assert not report_file.exists()


def test_client_control_flow_interruption_is_propagated_in_isolated_process():
    code = '''
from unittest.mock import patch
from looking_glass import cpu_shared_load as load
original = load.BatchedEventRecorder.record
def record(recorder, event):
    if event['kind'] == 'attempt':
        raise KeyboardInterrupt('injected client interruption')
    return original(recorder, event)
plan, digest = load.make_plan(('a', 'b'), jobs_per_team=1)
try:
    with patch.object(load.BatchedEventRecorder, 'record', record):
        load.run_cpu_window(plan, expected_plan_sha256=digest)
except KeyboardInterrupt:
    assert load._active_owner is not None
    print('propagated')
else:
    raise AssertionError('interruption was swallowed')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'propagated' in result.stdout


def test_unsafe_model_failure_retains_owner_and_produces_no_report():
    code = '''
from unittest.mock import patch
from looking_glass import cpu_shared_load as load
original = load._SyntheticModel.forward
def fail(self, input_ids):
    if int(input_ids[0, 0]) != 0:
        raise ValueError('injected CPU model failure')
    return original(self, input_ids)
plan, digest = load.make_plan(('a', 'b'), jobs_per_team=1)
try:
    with patch.object(load._SyntheticModel, 'forward', fail):
        load.run_cpu_window(plan, expected_plan_sha256=digest, timeout=10)
except Exception:
    assert load._active_owner is not None
    assert load._active_owner.authority.unsafe
    print('failed closed')
else:
    raise AssertionError('unsafe model failure produced a report')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'failed closed' in result.stdout
