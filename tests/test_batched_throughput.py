import copy
import math

import pytest
import torch

from looking_glass.batched_throughput import VERSION, analyze_pair, freeze_protocol
from looking_glass.throughput_benchmark import canonical_digest


def _protocol():
    return {
        'version': VERSION, 'model_sha256': 'a' * 64, 'checkpoint_sha256': 'f' * 64,
        'source_sha256': '9' * 64, 'runtime_sha256': 'b' * 64, 'capture_policy': 'none',
        'serial_policy_sha256': '8' * 64, 'grouped_policy_sha256': '7' * 64,
        'profile_sha256_by_rows': {'1': 'c' * 64, '2': 'd' * 64},
        'heldout_evidence_sha256': 'e' * 64,
        'input_shape': [1, 2], 'output_shape': [1, 2, 3], 'output_dtype': 'float32',
        'output_cap_bytes': 24, 'atol': 1e-7, 'rtol': 1e-6,
        'jobs': [
            {'job_id': 'j1', 'team_id': 'a', 'input_sha256': '1' * 64,
             'request_sha256': '3' * 64, 'seed': 17},
            {'job_id': 'j2', 'team_id': 'b', 'input_sha256': '2' * 64,
             'request_sha256': '4' * 64, 'seed': 17},
        ],
    }


def _sample(job, attempt, ready, selected, returned, call_id, output):
    broker_job_id = {'j1': 1, 'j2': 2}[job['job_id']] + (2 if call_id.startswith('g') else 0)
    return {'job_id': job['job_id'], 'input_sha256': job['input_sha256'],
            'request_sha256': job['request_sha256'], 'state': 'succeeded',
            'attempt_ns': attempt, 'ready_ns': ready, 'selected_ns': selected,
            'returned_ns': returned, 'broker_id': 'broker', 'broker_job_id': broker_job_id,
            'call_id': call_id, 'output_sha256': canonical_digest(output), 'error': None,
            'cancellation_requested': False, 'cancellation_requested_ns': None,
            'cancellation_accepted': None, 'cancellation_decision_ns': None,
            'broker_terminal_state': 'succeeded'}


def _call(call_id, members, launch, finish, profile):
    return {'call_id': call_id, 'group_id': 'group-' + call_id, 'profile_sha256': profile,
            'input_shape': [len(members), 2],
            'members': [{'job_id': job_id, 'broker_id': 'broker',
                         'broker_job_id': {'j1': 1, 'j2': 2}[job_id]
                         + (2 if call_id.startswith('g') else 0), 'row_offset': index}
                        for index, job_id in enumerate(members)],
            'launch_ns': launch, 'finish_ns': finish, 'state': 'succeeded',
            'synchronized': True, 'error': None}


def _pair():
    protocol, digest = freeze_protocol(_protocol())
    left = {job['job_id']: torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
            + i for i, job in enumerate(protocol['jobs'])}
    right = {key: value.clone() for key, value in left.items()}
    right['j1'][0, 0, 1] = torch.nextafter(right['j1'][0, 0, 1], torch.tensor(math.inf))
    common = {'owner_session_id': 'owner-1', 'protocol_sha256': digest,
              'model_sha256': protocol['model_sha256'],
              'runtime_sha256': protocol['runtime_sha256'], 'drained': True,
              'clock': 'perf_counter_ns', 'clock_resolution_ns': 100}
    serial = {**common, 'mode': 'serial', 'window_id': 's', 'policy_sha256': '8' * 64,
              'start_ns': 100, 'end_ns': 180,
              'attempts': [_sample(protocol['jobs'][0], 101, 102, 103, 113, 's1', left['j1']),
                           _sample(protocol['jobs'][1], 121, 122, 123, 143, 's2', left['j2'])],
              'calls': [_call('s1', ['j1'], 105, 110, 'c' * 64),
                        _call('s2', ['j2'], 125, 140, 'c' * 64)]}
    grouped = {**common, 'mode': 'grouped', 'window_id': 'g', 'policy_sha256': '7' * 64,
               'start_ns': 200, 'end_ns': 260,
               'attempts': [_sample(protocol['jobs'][0], 201, 202, 210, 244, 'g1', right['j1']),
                            _sample(protocol['jobs'][1], 203, 204, 211, 246, 'g1', right['j2'])],
               'calls': [_call('g1', ['j1', 'j2'], 220, 240, 'd' * 64)]}
    return protocol, digest, serial, grouped, left, right


def _analyze(data):
    p, digest, serial, grouped, left, right = data
    return analyze_pair(p, serial, grouped, expected_protocol_sha256=digest,
                        serial_outputs=left, grouped_outputs=right)


def test_nominal_clock_screen_flags_short_intervals_without_changing_consistency():
    report = _analyze(_pair())
    assert report['measured_pair_consistent'] is True
    screen = report['nominal_resolution_screen']
    for mode in ('serial', 'grouped'):
        result = screen[mode]
        assert result['clock'] == 'perf_counter_ns'
        assert result['clock_resolution_ns'] == 100
        assert result['multiplier'] == 100 and result['threshold_ns'] == 10000
        assert result['window']['below_threshold'] is True
        assert result['attempt_to_return']['observed'] == result['attempt_to_return']['below_threshold'] == 2
    assert screen['grouped']['synchronized_call']['observed'] == 1
    assert screen['grouped']['synchronized_call']['below_threshold'] == 1


def test_nominal_clock_screen_keeps_zero_short_and_exact_boundary_distinct():
    from looking_glass.batched_throughput import _screen_intervals

    result = _screen_intervals([('zero', 0), ('short', 9999), ('boundary', 10000)],
                               threshold_ns=10000, possible=4)
    assert result == {'observed': 3, 'not_observed': 1, 'below_threshold': 2,
                      'zero_duration': 1, 'below_threshold_ids': ['short', 'zero']}
    assert _screen_intervals([], threshold_ns=10000, possible=2) == {
        'observed': 0, 'not_observed': 2, 'below_threshold': 0,
        'zero_duration': 0, 'below_threshold_ids': []}


def test_nominal_clock_screen_passes_only_intervals_at_least_threshold():
    data = _pair()
    for window in data[2:4]:
        for field in ('start_ns', 'end_ns'):
            window[field] *= 10000
        for row in window['attempts']:
            for field in ('attempt_ns', 'ready_ns', 'selected_ns', 'returned_ns',
                          'cancellation_requested_ns', 'cancellation_decision_ns'):
                if row[field] is not None:
                    row[field] *= 10000
        for call in window['calls']:
            for field in ('launch_ns', 'finish_ns'):
                if call[field] is not None:
                    call[field] *= 10000
    screen = _analyze(data)['nominal_resolution_screen']
    for mode in ('serial', 'grouped'):
        assert screen[mode]['window']['below_threshold'] is False
        for field in ('attempt_to_return', 'ready_to_selection', 'selection_to_launch', 'synchronized_call'):
            assert screen[mode][field]['below_threshold'] == 0


def test_nominal_clock_screen_counts_unattempted_planned_jobs_as_unobserved():
    data = _pair()
    for window in data[2:4]:
        window['attempts'] = []
        window['calls'] = []
    data[4].clear()
    data[5].clear()
    report = _analyze(data)
    assert not report['measured_pair_consistent']
    for screen in report['nominal_resolution_screen'].values():
        assert (screen['planned_jobs'], screen['attempted_jobs'], screen['unattempted_jobs']) == (2, 0, 2)
        for field in ('attempt_to_return', 'ready_to_selection', 'selection_to_launch'):
            assert screen[field]['observed'] == 0 and screen[field]['not_observed'] == 2
        assert screen['synchronized_call']['observed'] == 0


def test_real_group_is_counted_and_one_ulp_difference_uses_tolerance():
    data = _pair()
    report = _analyze(data)
    assert report['measured_pair_consistent']
    assert report['observed_mixed_group'] and report['all_jobs_succeeded']
    assert report['same_recorded_broker']
    assert report['all_outputs_matched']
    assert report['output_checks'][0]['max_abs_error'] > 0
    assert report['summaries']['serial']['model_calls'] == 2
    assert report['summaries']['grouped']['model_calls'] == 1
    assert report['summaries']['grouped']['batch_rows']['max'] == 2
    assert report['summaries']['grouped']['successful_jobs_per_second'] == 2 / 60e-9
    assert report['summaries']['grouped']['successful_input_positions_per_second'] == 4 / 60e-9
    assert report['summaries']['grouped']['teams']['a']['successful_input_positions_per_second'] == 2 / 60e-9
    timing = report['summaries']['grouped']['timing_by_state']['succeeded']
    assert timing['ready_to_selection_ns']['p50'] == 7.5
    assert timing['selection_to_launch_ns']['p50'] == 9.5


def test_out_of_tolerance_result_is_invalid_even_with_correct_fingerprint():
    data = _pair()
    data[5]['j1'][0, 0, 1] += 0.1
    data[3]['attempts'][0]['output_sha256'] = canonical_digest(data[5]['j1'])
    report = _analyze(data)
    assert not report['measured_pair_consistent']
    assert report['output_checks'][0]['status'] == 'mismatch'


@pytest.mark.parametrize('corruption', ['nan', 'inf', 'shape', 'digest', 'missing'])
def test_missing_invalid_or_wrongly_bound_logits_invalidate_pair(corruption):
    data = _pair()
    if corruption in ('nan', 'inf'):
        data[5]['j1'][0, 0, 0] = float(corruption)
    elif corruption == 'shape':
        data[5]['j1'] = torch.zeros((1, 1, 3))
    elif corruption == 'digest':
        data[3]['attempts'][0]['output_sha256'] = 'f' * 64
    else:
        del data[5]['j1']
    report = _analyze(data)
    assert not report['measured_pair_consistent']
    assert report['output_checks'][0]['status'] == 'invalid_output'


def test_actual_single_row_calls_do_not_make_a_batched_claim():
    data = _pair()
    group = data[3]
    group['calls'] = [_call('g1', ['j1'], 212, 220, 'c' * 64),
                      _call('g2', ['j2'], 221, 240, 'c' * 64)]
    group['attempts'][1]['call_id'] = 'g2'
    report = _analyze(data)
    assert report['all_jobs_succeeded'] and report['all_outputs_matched']
    assert not report['observed_mixed_group'] and not report['measured_pair_consistent']


def test_whole_group_failure_counts_all_jobs_and_invalidates_comparison():
    data = _pair()
    for row in data[3]['attempts']:
        row.update(state='failed', output_sha256=None, error='OutOfMemoryError',
                   broker_terminal_state='failed')
    data[3]['calls'][0].update(state='failed', error='OutOfMemoryError')
    report = _analyze(data)
    assert not report['all_jobs_succeeded'] and not report['measured_pair_consistent']
    assert report['summaries']['grouped']['states']['failed'] == 2
    assert report['summaries']['grouped']['model_calls'] == 1
    assert report['summaries']['grouped']['model_call_states']['failed'] == 1
    assert report['summaries']['grouped']['successful_jobs_per_second'] == 0
    assert report['output_checks'] == []


def test_unresolved_call_is_preserved_but_cannot_claim_drain():
    data = _pair()
    data[3]['drained'] = False
    for row in data[3]['attempts']:
        row.update(state='unresolved', output_sha256=None, error='Timeout', broker_terminal_state=None)
    data[3]['calls'][0].update(state='unresolved', finish_ns=None, synchronized=False, error='Timeout')
    report = _analyze(data)
    assert not report['measured_pair_consistent']
    assert report['summaries']['grouped']['states']['unresolved'] == 2
    data[3]['drained'] = True
    assert not _analyze(data)['measured_pair_consistent']


def test_successful_jobs_do_not_hide_unconfirmed_owner_drain():
    data = _pair()
    data[3]['drained'] = False
    report = _analyze(data)
    assert report['all_jobs_succeeded'] and report['all_outputs_matched']
    assert report['summaries']['grouped']['drained'] is False
    assert report['summaries']['serial']['drained'] is True
    assert report['summaries']['grouped']['clock'] == 'perf_counter_ns'
    assert report['summaries']['grouped']['clock_resolution_ns'] == 100
    assert not report['measured_pair_consistent']


def test_failed_group_does_not_hide_accepted_cancellation():
    data = _pair()
    for row in data[3]['attempts']:
        row.update(state='failed', output_sha256=None, error='OutOfMemoryError',
                   cancellation_requested=True, cancellation_requested_ns=225,
                   cancellation_accepted=True, cancellation_decision_ns=226,
                   broker_terminal_state='failed')
    data[3]['calls'][0].update(state='failed', error='OutOfMemoryError')
    report = _analyze(data)
    assert report['summaries']['grouped']['states']['failed'] == 2
    assert report['summaries']['grouped']['cancellation_requests'] == 2
    assert report['summaries']['grouped']['model_call_states']['failed'] == 1
    data[3]['attempts'][0].update(state='cancelled', error=None, broker_terminal_state='cancelled')
    with pytest.raises(ValueError, match='failed member'):
        _analyze(data)


def test_failed_broker_result_can_coexist_with_unresolved_client_delivery():
    data = _pair()
    for row in data[3]['attempts']:
        row.update(state='failed', output_sha256=None, error='OutOfMemoryError',
                   broker_terminal_state='failed')
    data[3]['attempts'][0].update(state='unresolved', error='Disconnected', returned_ns=230)
    data[3]['calls'][0].update(state='failed', error='OutOfMemoryError')
    report = _analyze(data)
    summary = report['summaries']['grouped']
    assert summary['states']['failed'] == summary['states']['unresolved'] == 1
    assert summary['broker_terminal_states']['failed'] == 2
    assert summary['model_call_states']['failed'] == 1
    assert not report['measured_pair_consistent']


def test_cancellation_delivery_must_follow_launched_call_completion():
    data = _pair()
    row = data[3]['attempts'][0]
    row.update(state='cancelled', output_sha256=None, cancellation_requested=True,
               cancellation_requested_ns=225, cancellation_accepted=True, returned_ns=230,
               cancellation_decision_ns=226, broker_terminal_state='cancelled')
    with pytest.raises(ValueError, match='terminal delivery'):
        _analyze(data)
    row['returned_ns'] = 245
    report = _analyze(data)
    assert report['summaries']['grouped']['states']['cancelled'] == 1
    assert report['summaries']['grouped']['timing_by_state']['cancelled']['response_ns']['n'] == 1
    assert [item['job_id'] for item in report['output_checks']] == ['j2']
    assert report['output_checks'][0]['status'] == 'matched'
    assert not report['measured_pair_consistent']
    del data[5]['j2']
    assert _analyze(data)['output_checks'][0]['status'] == 'invalid_output'


def test_no_further_launch_after_an_unresolved_call():
    data = _pair()
    group = data[3]
    group['drained'] = False
    group['attempts'][0].update(state='unresolved', output_sha256=None, error='Timeout',
                                broker_terminal_state=None)
    group['attempts'][1]['call_id'] = 'g2'
    group['calls'] = [_call('g1', ['j1'], 212, 220, 'c' * 64),
                      _call('g2', ['j2'], 221, 240, 'c' * 64)]
    group['calls'][0].update(state='unresolved', finish_ns=None, synchronized=False, error='Timeout')
    with pytest.raises(ValueError, match='no later launch'):
        _analyze(data)


def test_unresolved_call_cannot_begin_before_prior_call_finishes():
    data = _pair()
    group = data[3]
    group['drained'] = False
    group['attempts'][1].update(state='unresolved', output_sha256=None, error='Timeout',
                                broker_terminal_state=None, call_id='g2')
    group['calls'] = [_call('g1', ['j1'], 212, 230, 'c' * 64),
                      _call('g2', ['j2'], 221, 240, 'c' * 64)]
    group['calls'][1].update(state='unresolved', finish_ns=None, synchronized=False, error='Timeout')
    with pytest.raises(ValueError, match='overlap'):
        _analyze(data)


def test_distinct_brokers_can_share_one_externally_verified_owner():
    data = _pair()
    for row in data[3]['attempts']:
        row['broker_id'] = 'second-broker'
    for member in data[3]['calls'][0]['members']:
        member['broker_id'] = 'second-broker'
    report = _analyze(data)
    assert report['measured_pair_consistent'] and not report['same_recorded_broker']


def test_grouped_broker_opaque_job_ids_are_preserved_without_coercion():
    data = _pair()
    for index, row in enumerate(data[3]['attempts'], 1):
        row['broker_id'] = 'grouped-broker'
        row['broker_job_id'] = f'grouped-broker-j-{index}'
    for index, member in enumerate(data[3]['calls'][0]['members'], 1):
        member['broker_id'] = 'grouped-broker'
        member['broker_job_id'] = f'grouped-broker-j-{index}'
    report = _analyze(data)
    assert report['measured_pair_consistent'] and not report['same_recorded_broker']


def test_broker_job_ids_must_match_in_type_as_well_as_value():
    data = _pair()
    data[3]['attempts'][0]['broker_job_id'] = '3'
    with pytest.raises(ValueError, match='identity'):
        _analyze(data)


@pytest.mark.parametrize('invalid', [True, 0, '', '  '])
def test_invalid_broker_job_identity_is_rejected(invalid):
    data = _pair()
    data[3]['attempts'][0]['broker_job_id'] = invalid
    data[3]['calls'][0]['members'][0]['broker_job_id'] = invalid
    with pytest.raises(ValueError, match='broker job ID'):
        _analyze(data)


def test_success_cannot_be_published_after_accepted_or_unknown_cancellation():
    data = _pair()
    row = data[3]['attempts'][0]
    row.update(cancellation_requested=True, cancellation_requested_ns=225,
               cancellation_accepted=True, cancellation_decision_ns=226)
    with pytest.raises(ValueError, match='cancellation'):
        _analyze(data)
    row['cancellation_accepted'] = None
    row['cancellation_decision_ns'] = None
    with pytest.raises(ValueError, match='cancellation'):
        _analyze(data)
    row['cancellation_accepted'] = False
    row['cancellation_decision_ns'] = 226
    assert _analyze(data)['measured_pair_consistent']
    row['cancellation_decision_ns'] = 224
    with pytest.raises(ValueError, match='decision is out of order'):
        _analyze(data)


def test_later_window_requires_prior_owner_drain_even_when_client_is_unresolved():
    data = _pair()
    serial = data[2]
    serial['drained'] = False
    serial['attempts'] = serial['attempts'][:1]
    serial['attempts'][0].update(state='unresolved', output_sha256=None, error='Timeout',
                                 broker_terminal_state=None)
    serial['calls'] = serial['calls'][:1]
    serial['calls'][0].update(state='unresolved', finish_ns=None, synchronized=False, error='Timeout')
    with pytest.raises(ValueError, match='prior owner drain'):
        _analyze(data)
    serial['drained'] = True
    assert not _analyze(data)['measured_pair_consistent']


@pytest.mark.parametrize('change', ['row_count', 'bool_row_count', 'row_offset', 'profile', 'policy', 'input',
                                    'clock', 'clock_source', 'owner', 'unsynchronized',
                                    'duplicate', 'broker_identity', 'multiple_brokers',
                                    'reused_broker_job', 'overlap'])
def test_inconsistent_event_or_binding_evidence_is_rejected(change):
    data = _pair()
    serial, grouped = data[2], data[3]
    if change == 'row_count':
        grouped['calls'][0]['input_shape'][0] = 1
    elif change == 'bool_row_count':
        serial['calls'][0]['input_shape'][0] = True
    elif change == 'row_offset':
        grouped['calls'][0]['members'][1]['row_offset'] = 0
    elif change == 'profile':
        grouped['calls'][0]['profile_sha256'] = 'c' * 64
    elif change == 'policy':
        grouped['policy_sha256'] = '8' * 64
    elif change == 'input':
        grouped['attempts'][0]['input_sha256'] = 'f' * 64
    elif change == 'clock':
        grouped['attempts'][0]['selected_ns'] = 199
    elif change == 'clock_source':
        grouped['clock'] = 'monotonic_ns'
    elif change == 'owner':
        grouped['owner_session_id'] = 'other'
    elif change == 'unsynchronized':
        grouped['calls'][0]['synchronized'] = False
    elif change == 'duplicate':
        grouped['attempts'][1]['job_id'] = 'j1'
    elif change == 'broker_identity':
        grouped['calls'][0]['members'][0]['broker_job_id'] = 999
    elif change == 'multiple_brokers':
        grouped['attempts'][0]['broker_id'] = 'broker-2'
        grouped['calls'][0]['members'][0]['broker_id'] = 'broker-2'
    elif change == 'reused_broker_job':
        grouped['attempts'][0]['broker_job_id'] = 1
        grouped['calls'][0]['members'][0]['broker_job_id'] = 1
    else:
        serial['calls'][1]['launch_ns'] = 109
        serial['attempts'][1].update(attempt_ns=105, ready_ns=106, selected_ns=107)
    with pytest.raises(ValueError):
        _analyze(data)


def test_missing_attempt_and_prelaunch_cancellation_are_not_silently_excluded():
    data = _pair()
    row = data[3]['attempts'][1]
    row.update(state='cancelled', selected_ns=None, call_id=None, output_sha256=None,
               error=None, cancellation_requested=True, cancellation_requested_ns=205,
               cancellation_accepted=True, cancellation_decision_ns=206,
               broker_terminal_state='cancelled')
    data[3]['calls'][0].update(input_shape=[1, 2], profile_sha256='c' * 64,
                               members=[{'job_id': 'j1', 'broker_id': 'broker',
                                         'broker_job_id': 3, 'row_offset': 0}])
    report = _analyze(data)
    assert report['summaries']['grouped']['states']['cancelled'] == 1
    assert not report['measured_pair_consistent']
    data[3]['attempts'].pop()
    report = _analyze(data)
    assert report['summaries']['grouped']['unattempted'] == 1
    assert not report['all_jobs_succeeded']


def test_frozen_protocol_rejects_changed_seed_and_unknown_fields():
    protocol = _protocol()
    protocol['jobs'][1]['seed'] += 1
    with pytest.raises(ValueError, match='common seed'):
        freeze_protocol(protocol)
    protocol = _protocol()
    protocol['unknown'] = True
    with pytest.raises(ValueError, match='fields'):
        freeze_protocol(protocol)


def test_distinct_row_counts_need_distinct_exact_profiles():
    protocol = _protocol()
    protocol['profile_sha256_by_rows']['2'] = protocol['profile_sha256_by_rows']['1']
    with pytest.raises(ValueError, match='distinct exact profile'):
        freeze_protocol(protocol)


@pytest.mark.parametrize('seed', [-1, 2 ** 63, True])
def test_seed_must_fit_the_resident_worker_integer_range(seed):
    protocol = _protocol()
    protocol['jobs'][0]['seed'] = seed
    with pytest.raises(ValueError, match='seed'):
        freeze_protocol(protocol)


def test_forward_only_scope_rejects_capture_and_nonrepresentable_tolerance():
    protocol = _protocol()
    protocol['capture_policy'] = 'gradient'
    with pytest.raises(ValueError, match='no-capture'):
        freeze_protocol(protocol)
    protocol = _protocol()
    protocol['atol'] = 2 ** 53 + 1
    with pytest.raises(ValueError, match='tolerance'):
        freeze_protocol(protocol)


def test_team_cannot_submit_next_closed_loop_job_before_its_prior_return():
    data = _pair()
    protocol = copy.deepcopy(data[0])
    job = {'job_id': 'j3', 'team_id': 'a', 'input_sha256': '5' * 64,
           'request_sha256': '6' * 64, 'seed': 17}
    protocol['jobs'].append(job)
    protocol, digest = freeze_protocol(protocol)
    data = (protocol, digest, *data[2:])
    data[2]['protocol_sha256'] = digest
    data[3]['protocol_sha256'] = digest
    value = torch.zeros((1, 2, 3))
    data[4]['j3'] = value
    data[2]['attempts'].append({'job_id': 'j3', 'input_sha256': job['input_sha256'],
        'request_sha256': job['request_sha256'], 'state': 'succeeded',
        'attempt_ns': 102, 'ready_ns': 103, 'selected_ns': 104, 'returned_ns': 119,
        'broker_id': 'broker', 'broker_job_id': 9, 'call_id': 's3',
        'output_sha256': canonical_digest(value), 'error': None, 'cancellation_requested': False,
        'cancellation_requested_ns': None, 'cancellation_accepted': None,
        'cancellation_decision_ns': None, 'broker_terminal_state': 'succeeded'})
    data[2]['calls'].append({'call_id': 's3', 'group_id': 'group-s3', 'profile_sha256': 'c' * 64,
        'input_shape': [1, 2], 'members': [{'job_id': 'j3', 'broker_id': 'broker',
                                           'broker_job_id': 9, 'row_offset': 0}],
        'launch_ns': 115, 'finish_ns': 118, 'state': 'succeeded', 'synchronized': True, 'error': None})
    with pytest.raises(ValueError, match='closed-loop'):
        _analyze(data)


def test_unprofiled_row_count_cannot_match_a_null_profile():
    data = _pair()
    protocol = copy.deepcopy(data[0])
    del protocol['profile_sha256_by_rows']['2']
    protocol['profile_sha256_by_rows']['3'] = 'd' * 64
    protocol, digest = freeze_protocol(protocol)
    data = (protocol, digest, *data[2:])
    data[2]['protocol_sha256'] = digest
    data[3]['protocol_sha256'] = digest
    data[3]['calls'][0]['profile_sha256'] = None
    with pytest.raises(ValueError, match='profile'):
        _analyze(data)
