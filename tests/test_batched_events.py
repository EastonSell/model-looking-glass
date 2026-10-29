import copy

import pytest
import torch

from looking_glass.batched_events import IncompleteEventEvidence, assemble_window
from looking_glass.batched_throughput import VERSION, analyze_pair, freeze_protocol
from looking_glass.throughput_benchmark import canonical_digest


def _fixture():
    p, digest = freeze_protocol({
        'version': VERSION, 'model_sha256': 'a' * 64, 'checkpoint_sha256': 'f' * 64,
        'source_sha256': '9' * 64, 'runtime_sha256': 'b' * 64, 'capture_policy': 'none',
        'serial_policy_sha256': '8' * 64, 'grouped_policy_sha256': '7' * 64,
        'profile_sha256_by_rows': {'1': 'c' * 64, '2': 'd' * 64},
        'heldout_evidence_sha256': 'e' * 64, 'input_shape': [1, 2],
        'output_shape': [1, 2, 3], 'output_dtype': 'float32', 'output_cap_bytes': 24,
        'atol': 1e-7, 'rtol': 1e-6,
        'jobs': [{'job_id': 'j1', 'team_id': 'a', 'input_sha256': '1' * 64,
                  'request_sha256': '3' * 64, 'seed': 17},
                 {'job_id': 'j2', 'team_id': 'b', 'input_sha256': '2' * 64,
                  'request_sha256': '4' * 64, 'seed': 17}]})
    values = {'j1': torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
              'j2': torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) + 1}
    common = {'owner_session_id': 'owner', 'model_sha256': p['model_sha256'],
              'runtime_sha256': p['runtime_sha256'], 'clock': 'perf_counter_ns',
              'clock_resolution_ns': 100, 'events_complete': True}
    contexts = ({**common, 'mode': 'serial', 'window_id': 's', 'policy_sha256': '8' * 64},
                {**common, 'mode': 'grouped', 'window_id': 'g', 'policy_sha256': '7' * 64})

    def returned(job_id, time):
        return {'kind': 'returned', 'time_ns': time, 'job_id': job_id,
                'state': 'succeeded', 'error': None, 'output_sha256': canonical_digest(values[job_id]),
                'cancellation_requested': False, 'cancellation_requested_ns': None,
                'cancellation_accepted': None, 'cancellation_decision_ns': None}

    def attempt(job, time):
        return {'kind': 'attempt', 'time_ns': time, 'job_id': job['job_id'],
                'team_id': job['team_id'], 'input_sha256': job['input_sha256'],
                'request_sha256': job['request_sha256']}

    def member(job_id, n, offset):
        return {'job_id': job_id, 'team_id': {'j1': 'a', 'j2': 'b'}[job_id],
                'broker_id': 'broker', 'broker_job_id': n, 'row_offset': offset}

    serial = [
        {'kind': 'window_start', 'time_ns': 100}, {'kind': 'window_end', 'time_ns': 180},
        {'kind': 'owner_drain', 'time_ns': 160, 'drained': True},
        attempt(p['jobs'][0], 101), {'kind': 'ready', 'time_ns': 102, 'job_id': 'j1',
                                     'team_id': 'a', 'broker_id': 'broker', 'broker_job_id': 1},
        {'kind': 'selected', 'time_ns': 103, 'job_id': 'j1'},
        {'kind': 'forward_entry', 'time_ns': 105, 'call_id': 's1', 'group_id': 'sg1',
         'profile_sha256': 'c' * 64, 'input_shape': [1, 2], 'members': [member('j1', 1, 0)]},
        {'kind': 'forward_finished', 'time_ns': 110, 'call_id': 's1', 'state': 'succeeded',
         'synchronized': True, 'error': None},
        {'kind': 'broker_terminal', 'time_ns': 111, 'job_id': 'j1', 'state': 'succeeded'},
        returned('j1', 113),
        attempt(p['jobs'][1], 121), {'kind': 'ready', 'time_ns': 122, 'job_id': 'j2',
                                     'team_id': 'b', 'broker_id': 'broker', 'broker_job_id': 2},
        {'kind': 'selected', 'time_ns': 123, 'job_id': 'j2'},
        {'kind': 'forward_entry', 'time_ns': 125, 'call_id': 's2', 'group_id': 'sg2',
         'profile_sha256': 'c' * 64, 'input_shape': [1, 2], 'members': [member('j2', 2, 0)]},
        {'kind': 'forward_finished', 'time_ns': 140, 'call_id': 's2', 'state': 'succeeded',
         'synchronized': True, 'error': None},
        {'kind': 'broker_terminal', 'time_ns': 141, 'job_id': 'j2', 'state': 'succeeded'},
        returned('j2', 143),
    ]
    grouped = [
        {'kind': 'window_start', 'time_ns': 200}, {'kind': 'window_end', 'time_ns': 260},
        {'kind': 'owner_drain', 'time_ns': 250, 'drained': True},
        attempt(p['jobs'][0], 201), {'kind': 'ready', 'time_ns': 202, 'job_id': 'j1',
                                     'team_id': 'a', 'broker_id': 'broker', 'broker_job_id': 3},
        {'kind': 'selected', 'time_ns': 210, 'job_id': 'j1'},
        attempt(p['jobs'][1], 203), {'kind': 'ready', 'time_ns': 204, 'job_id': 'j2',
                                     'team_id': 'b', 'broker_id': 'broker', 'broker_job_id': 4},
        {'kind': 'selected', 'time_ns': 211, 'job_id': 'j2'},
        {'kind': 'forward_entry', 'time_ns': 220, 'call_id': 'g1', 'group_id': 'gg1',
         'profile_sha256': 'd' * 64, 'input_shape': [2, 2],
         'members': [member('j1', 3, 0), member('j2', 4, 1)]},
        {'kind': 'forward_finished', 'time_ns': 240, 'call_id': 'g1', 'state': 'succeeded',
         'synchronized': True, 'error': None},
        {'kind': 'broker_terminal', 'time_ns': 241, 'job_id': 'j1', 'state': 'succeeded'},
        {'kind': 'broker_terminal', 'time_ns': 242, 'job_id': 'j2', 'state': 'succeeded'},
        returned('j1', 244), returned('j2', 246),
    ]
    return p, digest, contexts, serial, grouped, values


def _assemble(p, digest, context, events):
    return assemble_window(p, expected_protocol_sha256=digest, context=context, events=events)


def test_explicit_observations_reconstruct_analyzer_windows_without_mutation():
    p, digest, contexts, serial, grouped, values = _fixture()
    original = copy.deepcopy(grouped)
    left = _assemble(p, digest, contexts[0], list(reversed(serial)))
    right = _assemble(p, digest, contexts[1], list(reversed(grouped)))
    assert grouped == original
    assert left == _assemble(p, digest, contexts[0], serial)
    assert right == _assemble(p, digest, contexts[1], grouped)
    assert right['calls'][0]['launch_ns'] == 220 and right['calls'][0]['finish_ns'] == 240
    assert right['attempts'][0]['selected_ns'] == 210
    report = analyze_pair(p, left, right, expected_protocol_sha256=digest,
                          serial_outputs=values, grouped_outputs=values)
    assert report['measured_pair_consistent']
    assert report['summaries']['grouped']['model_calls'] == 1
    assert report['summaries']['grouped']['batch_rows']['max'] == 2


def test_grouped_opaque_broker_job_ids_survive_adapter_unchanged():
    p, digest, contexts, serial, grouped, values = _fixture()
    ids = {'j1': 'grouped-broker-j-1', 'j2': 'grouped-broker-j-2'}
    for event in grouped:
        if event['kind'] == 'ready':
            event.update(broker_id='grouped-broker', broker_job_id=ids[event['job_id']])
        elif event['kind'] == 'forward_entry':
            for member in event['members']:
                member.update(broker_id='grouped-broker', broker_job_id=ids[member['job_id']])
    left = _assemble(p, digest, contexts[0], serial)
    right = _assemble(p, digest, contexts[1], grouped)
    assert [row['broker_job_id'] for row in right['attempts']] == list(ids.values())
    report = analyze_pair(p, left, right, expected_protocol_sha256=digest,
                          serial_outputs=values, grouped_outputs=values)
    assert report['measured_pair_consistent'] and not report['same_recorded_broker']


@pytest.mark.parametrize('kind', ['window_start', 'window_end', 'owner_drain', 'returned',
                                  'selected', 'ready', 'forward_finished', 'broker_terminal'])
def test_missing_observations_cannot_be_inferred(kind):
    p, digest, contexts, _, grouped, _ = _fixture()
    index = next(i for i, event in enumerate(grouped) if event['kind'] == kind)
    grouped.pop(index)
    with pytest.raises(ValueError):
        _assemble(p, digest, contexts[1], grouped)


@pytest.mark.parametrize('kind', ['window_start', 'owner_drain', 'returned', 'ready',
                                  'forward_entry', 'forward_finished'])
def test_duplicate_observations_are_rejected(kind):
    p, digest, contexts, _, grouped, _ = _fixture()
    grouped.append(copy.deepcopy(next(event for event in grouped if event['kind'] == kind)))
    with pytest.raises(IncompleteEventEvidence, match='duplicate'):
        _assemble(p, digest, contexts[1], grouped)


@pytest.mark.parametrize('change', ['event_loss', 'outside', 'terminal_before_finish',
                                    'terminal_after_delivery', 'drain_before_finish',
                                    'drain_before_terminal', 'wrong_membership', 'wrong_attempt_team',
                                    'wrong_ready_team', 'wrong_member_team',
                                    'missing_protocol', 'unknown_job'])
def test_inconsistent_event_evidence_is_rejected(change):
    p, digest, contexts, _, grouped, _ = _fixture()
    if change == 'event_loss':
        contexts[1]['events_complete'] = False
    elif change == 'outside':
        next(event for event in grouped if event['kind'] == 'ready')['time_ns'] = 199
    elif change == 'terminal_before_finish':
        next(event for event in grouped if event['kind'] == 'broker_terminal')['time_ns'] = 239
    elif change == 'terminal_after_delivery':
        next(event for event in grouped if event['kind'] == 'broker_terminal')['time_ns'] = 245
    elif change == 'drain_before_finish':
        next(event for event in grouped if event['kind'] == 'owner_drain')['time_ns'] = 239
    elif change == 'drain_before_terminal':
        next(event for event in grouped if event['kind'] == 'owner_drain')['time_ns'] = 240
    elif change == 'wrong_membership':
        next(event for event in grouped if event['kind'] == 'forward_entry')['members'][0]['broker_job_id'] = 999
    elif change == 'wrong_attempt_team':
        next(event for event in grouped if event['kind'] == 'attempt')['team_id'] = 'b'
    elif change == 'wrong_ready_team':
        next(event for event in grouped if event['kind'] == 'ready')['team_id'] = 'b'
    elif change == 'wrong_member_team':
        next(event for event in grouped if event['kind'] == 'forward_entry')['members'][0]['team_id'] = 'b'
    elif change == 'missing_protocol':
        digest = '0' * 64
    else:
        next(event for event in grouped if event['kind'] == 'ready')['job_id'] = 'unknown'
    with pytest.raises(ValueError):
        _assemble(p, digest, contexts[1], grouped)


def test_explicit_uncertain_observation_keeps_finish_unknown():
    p, digest, contexts, _, grouped, _ = _fixture()
    grouped[:] = [event for event in grouped
                  if event['kind'] not in ('forward_finished', 'broker_terminal')]
    grouped.append({'kind': 'forward_uncertain', 'time_ns': 230, 'call_id': 'g1', 'error': 'Timeout'})
    next(event for event in grouped if event['kind'] == 'owner_drain')['drained'] = False
    for event in grouped:
        if event['kind'] == 'returned':
            event.update(time_ns=225, state='unresolved', error='Timeout', output_sha256=None)
    window = _assemble(p, digest, contexts[1], grouped)
    assert window['calls'][0]['state'] == 'unresolved'
    assert window['calls'][0]['finish_ns'] is None
    assert window['calls'][0]['synchronized'] is False
    assert window['drained'] is False
    next(event for event in grouped if event['kind'] == 'forward_uncertain')['time_ns'] = 219
    with pytest.raises(IncompleteEventEvidence, match='precedes model entry'):
        _assemble(p, digest, contexts[1], grouped)


def test_owner_drain_cannot_precede_a_later_uncertain_call_observation():
    p, digest, contexts, _, grouped, _ = _fixture()
    grouped[:] = [event for event in grouped
                  if event['kind'] not in ('forward_finished', 'broker_terminal')]
    grouped.append({'kind': 'forward_uncertain', 'time_ns': 230, 'call_id': 'g1', 'error': 'Timeout'})
    next(event for event in grouped if event['kind'] == 'owner_drain')['time_ns'] = 225
    for event in grouped:
        if event['kind'] == 'returned':
            event.update(time_ns=235, state='unresolved', error='Timeout', output_sha256=None)
    with pytest.raises(IncompleteEventEvidence, match='owner drain precedes'):
        _assemble(p, digest, contexts[1], grouped)


def test_terminal_outcome_cannot_precede_scheduler_selection():
    p, digest, contexts, _, grouped, _ = _fixture()
    grouped[:] = [event for event in grouped if event['kind'] not in ('forward_entry', 'forward_finished')]
    for event in grouped:
        if event['kind'] == 'broker_terminal':
            event.update(time_ns=205, state='failed')
        elif event['kind'] == 'returned':
            event.update(state='failed', error='Failure', output_sha256=None)
    with pytest.raises(IncompleteEventEvidence, match='precedes selection'):
        _assemble(p, digest, contexts[1], grouped)


def test_failed_group_preserves_accepted_cancel_and_unresolved_client():
    p, digest, contexts, serial, grouped, values = _fixture()
    finish = next(event for event in grouped if event['kind'] == 'forward_finished')
    finish.update(state='failed', error='OutOfMemoryError')
    for event in grouped:
        if event['kind'] == 'broker_terminal':
            event['state'] = 'failed'
        elif event['kind'] == 'returned' and event['job_id'] == 'j1':
            event.update(time_ns=225, state='unresolved', error='Disconnected', output_sha256=None,
                         cancellation_requested=True, cancellation_requested_ns=215,
                         cancellation_accepted=True, cancellation_decision_ns=216)
        elif event['kind'] == 'returned':
            event.update(state='failed', error='OutOfMemoryError', output_sha256=None)
    left = _assemble(p, digest, contexts[0], serial)
    right = _assemble(p, digest, contexts[1], grouped)
    report = analyze_pair(p, left, right, expected_protocol_sha256=digest,
                          serial_outputs=values, grouped_outputs={})
    summary = report['summaries']['grouped']
    assert summary['states']['failed'] == summary['states']['unresolved'] == 1
    assert summary['broker_terminal_states']['failed'] == 2
    assert summary['model_call_states']['failed'] == 1
    assert summary['cancellation_requests'] == 1
    assert not report['measured_pair_consistent']
