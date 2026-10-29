from copy import deepcopy
from pathlib import Path
import statistics

import pytest

from looking_glass import tap_repeated_protocol as protocol


SCHEDULE = Path(__file__).resolve().parents[1] / 'evidence/gpu-tap-latency-repeated-v1/schedule.json'
CLOCK = {'name': 'perf_counter_ns', 'implementation': 'test', 'monotonic': True,
         'adjustable': False, 'resolution_seconds': 1e-7}
TOTAL, CAP = 8 * 1024 ** 3, 2 * 1024 ** 3
MEMORY = {'free_bytes': 6 * 1024 ** 3, 'total_bytes': TOTAL,
          'allocated_bytes': 384 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}


def _journal(schedule, session=0):
    identity = protocol.IDENTITY
    checks = {'outputs_finite': True, 'outputs_exact': True, 'values_exact': True,
              'preflight_modes': ['none', 'id', 'direct', 'gradient'], 'id_value_absent': True,
              'run_id': f'{protocol.PROTOCOL}-session-{session}', 'trace_sequences': [0],
              'capture_modes': ['id', 'direct', 'gradient'], 'output_shape': [1, 2, 50257],
              'output_dtype': 'torch.float32', 'output_device': 'cpu',
              'events_per_mode': 1, 'value_tensor_bytes': 3072,
              'event_identities': [identity],
              'value_layouts': [{'shape': [1, 1, 768], 'dtype': 'torch.float32', 'device': 'cpu'}],
              'comparison': {'operator': 'torch.equal', 'tolerance': 'none',
                             'guarantees_bitwise_identity': False, 'subnormal_handling': 'backend-dependent'}}
    rows = [{'stage': 'preflight', 'observation_id': 'preflight', 'postcheck_status': 'pending',
             'started_ns': 1_000_000, 'ended_ns': 2_000_000,
             'checks': checks, 'memory_before': MEMORY, 'peak_allocated_bytes': MEMORY['allocated_bytes'],
             'peak_reserved_bytes': MEMORY['reserved_bytes']},
            {'stage': 'postcheck', 'observation_id': 'preflight', 'status': 'passed', 'memory_after': MEMORY}]
    for i, item in enumerate(protocol.session_order(schedule, session)):
        mode = item['mode']
        # Ratios vary by round; their median is deliberately different from
        # the ratio of mode medians.
        scale = ({'none': 1, 'direct': 1, 'gradient': 1} if item['round'] < 6
                 else {'none': 1, 'direct': 2, 'gradient': 4})
        wall = 100_000 * scale[mode]
        start = 10_000_000 + i * 2_000_000
        trace = mode != 'none'
        row = {**item, 'session': session, 'postcheck_status': 'pending', 'started_ns': start,
               'ended_ns': start + wall, 'wall_ns': wall, 'device_event_ms': wall / 1_000_000,
               'memory_before': MEMORY, 'peak_allocated_bytes': MEMORY['allocated_bytes'],
               'peak_reserved_bytes': MEMORY['reserved_bytes'], 'events': int(trace),
               'recorded_tensor_bytes': 3072 if trace else 0, 'output_shape': [1, 2, 50257],
               'output_dtype': 'torch.float32', 'output_device': 'cpu', 'default_stream': True,
               'output_finite': True, 'capture_finite': True if trace else None,
               'trace_identity': identity if trace else None, 'trace_sequence': 0 if trace else None,
               'trace_run_id': f'{protocol.PROTOCOL}-session-{session}' if trace else None,
               'trace_mode': mode if trace else None, 'value_shape': [1, 1, 768] if trace else None,
               'value_dtype': 'torch.float32' if trace else None, 'value_device': 'cpu' if trace else None}
        rows.extend((row, {'stage': 'postcheck', 'observation_id': item['observation_id'],
                           'status': 'passed', 'memory_after': MEMORY}))
    return rows


def _summarize(schedule, rows):
    return protocol.summarize_session(schedule, 0, rows, clock=CLOCK,
                                      total_bytes=TOTAL, cap_bytes=CAP, device='cpu')


def test_only_exact_saved_schedule_is_accepted(tmp_path):
    schedule = protocol.load_schedule(SCHEDULE)
    for session in range(5):
        order = protocol.session_order(schedule, session)
        assert len(order) == 42 and [row['mode'] for row in order[:6]] == [
            'none', 'direct', 'gradient', 'gradient', 'direct', 'none']
        assert all(sum(row['mode'] == mode and row['position'] == position
                       for row in order[6:]) == 4 for mode in protocol.MODES for position in range(3))
    changed = tmp_path / 'schedule.json'
    changed.write_bytes(SCHEDULE.read_bytes() + b' ')
    with pytest.raises(ValueError, match='frozen schedule'):
        protocol.load_schedule(changed)
    schedule['sessions'][0]['measured_rounds'][0][0] = 'none'
    with pytest.raises(ValueError, match='frozen snapshot'):
        protocol.session_order(schedule, 0)


def test_completed_journal_preserves_round_pairs_and_excludes_warmup():
    schedule = protocol.load_schedule(SCHEDULE)
    rows = _journal(schedule)
    result = _summarize(schedule, rows)
    assert result['warmup_samples'] == 6 and result['measured_samples'] == 36
    expected = [row['wall_ns'] for row in rows if row.get('stage') == 'measured' and row.get('mode') == 'none']
    assert result['mode_wall_ns_medians']['none'] == statistics.median(expected)
    assert result['median_gradient_over_direct'] == 1.5
    assert result['median_direct_over_none'] == 1.5
    assert result['mode_wall_ns_medians']['gradient'] / result['mode_wall_ns_medians']['direct'] != 1.5
    assert len(result['paired_rounds']) == 12


@pytest.mark.parametrize('damage', ['missing_post', 'duplicate', 'order', 'mode', 'clock',
                                    'nonfinite_event', 'event_too_long', 'output_finite', 'capture_finite',
                                    'identity', 'run_id', 'memory', 'impossible_memory', 'overlap'])
def test_incomplete_or_contradictory_journal_cannot_be_summarized(damage):
    schedule = protocol.load_schedule(SCHEDULE)
    rows, clock = deepcopy(_journal(schedule)), dict(CLOCK)
    if damage == 'missing_post': rows.pop()
    if damage == 'duplicate': rows[-2:] = deepcopy(rows[-4:-2])
    if damage == 'order': rows[2:6] = rows[4:6] + rows[2:4]
    if damage == 'mode': rows[2]['mode'] = 'direct'
    if damage == 'clock': clock['adjustable'] = True
    if damage == 'nonfinite_event': rows[-2]['device_event_ms'] = float('nan')
    if damage == 'event_too_long': rows[-2]['device_event_ms'] = 1_000_000.0
    if damage == 'output_finite': rows[-2]['output_finite'] = False
    if damage == 'capture_finite': rows[4]['capture_finite'] = False
    if damage == 'identity': rows[4]['trace_identity'][0] = 'wrong'
    if damage == 'run_id': rows[4]['trace_run_id'] = 'wrong'
    if damage == 'memory': rows[-1]['memory_after']['free_bytes'] = 0
    if damage == 'impossible_memory': rows[-1]['memory_after']['free_bytes'] = TOTAL
    if damage == 'overlap': rows[4]['started_ns'] = rows[2]['started_ns']
    with pytest.raises(ValueError):
        protocol.summarize_session(schedule, 0, rows, clock=clock, total_bytes=TOTAL, cap_bytes=CAP, device='cpu')


def _screen():
    observations = []
    for second in (1.5, 2, 2.5, 3.5, 4.5, 5.5, 6.1, 6.6, 7.1):
        end = int(second * 1_000_000_000)
        observations.append({'gpu_uuid': 'GPU-test', 'query_started_ns': end - 1000,
                             'query_ended_ns': end, 'observed_ns': end, 'total_bytes': TOTAL,
                             'used_bytes': TOTAL // 4, 'free_bytes': 3 * TOTAL // 4,
                             'utilization_percent': 1})
    queries = [{'phase': 'before', 'ok': True, 'query_started_ns': 2_800_000_000,
                'query_ended_ns': 2_900_000_000, 'processes': []},
               {'phase': 'after', 'ok': True, 'query_started_ns': 5_100_000_000,
                'query_ended_ns': 5_200_000_000, 'processes': []}]
    return {'observations': observations, 'errors': [], 'process_queries': queries,
            'gpu_uuid': 'GPU-test', 'own_pid': 10, 'workload_started_ns': 3_000_000_000,
            'workload_ended_ns': 5_000_000_000, 'expected_total_bytes': TOTAL, 'exclusive': True}


def test_activity_screen_distinguishes_contamination_from_unavailability():
    inputs = _screen()
    clean = protocol.classify_activity(**inputs)
    assert clean['status'] == 'no_contamination_observed' and clean['availability'] == 'complete'
    inputs['observations'][0]['utilization_percent'] = 12
    inputs['process_queries'][1]['processes'] = [{'gpu_uuid': 'GPU-test', 'pid': 15, 'name': 'desktop'}]
    contaminated = protocol.classify_activity(**inputs)
    assert contaminated['status'] == 'contaminated' and contaminated['availability'] == 'complete'
    assert contaminated['foreign_process_ids'] == [15]
    assert set(contaminated['contamination_reasons']) == {'foreign_process_listed',
                                                          'idle_utilization_above_five_percent'}
    inputs['errors'].append({'error': 'failed query'})
    both = protocol.classify_activity(**inputs)
    assert both['status'] == 'contaminated' and both['availability'] == 'unverified'
    assert 'telemetry_query_error' in both['unavailable_reasons']


@pytest.mark.parametrize('damage', ['no_attestation', 'identity', 'bad_total', 'gap',
                                    'missing_after', 'stale_before', 'stale_after', 'bad_process'])
def test_activity_screen_does_not_promote_missing_or_inconsistent_evidence(damage):
    inputs = _screen()
    if damage == 'no_attestation': inputs['exclusive'] = False
    if damage == 'identity': inputs['observations'][3]['gpu_uuid'] = 'GPU-other'
    if damage == 'bad_total': inputs['observations'][3]['total_bytes'] += 1
    if damage == 'gap': inputs['observations'] = inputs['observations'][:3] + inputs['observations'][-3:]
    if damage == 'missing_after': inputs['observations'] = inputs['observations'][:-2]
    if damage == 'stale_before': inputs['process_queries'][0]['query_started_ns'] = 100
    if damage == 'stale_after':
        inputs['process_queries'][1]['query_started_ns'] += 3_600_000_000_000
        inputs['process_queries'][1]['query_ended_ns'] += 3_600_000_000_000
    if damage == 'bad_process': inputs['process_queries'][0]['processes'] = [{'gpu_uuid': 'GPU-other', 'pid': 1, 'name': 'x'}]
    assert protocol.classify_activity(**inputs)['availability'] == 'unverified'


def test_initial_activity_requires_fresh_telemetry_before_any_work():
    values = _screen()
    args = {'observations': values['observations'][:3], 'errors': [],
            'process_query': values['process_queries'][0], 'gpu_uuid': 'GPU-test',
            'own_pid': 10, 'expected_total_bytes': TOTAL, 'now_ns': 3_000_000_000}
    assert protocol.check_initial_activity(**args)['foreign_process_ids'] == []
    missing = dict(args, observations=args['observations'][:2])
    with pytest.raises(ValueError, match='unavailable'):
        protocol.check_initial_activity(**missing)
    stale = dict(args, now_ns=6_000_000_000)
    with pytest.raises(ValueError, match='stale'):
        protocol.check_initial_activity(**stale)
    future = dict(args, now_ns=2_000_000_000)
    with pytest.raises(ValueError):
        protocol.check_initial_activity(**future)


def _attempt(schedule, session, binding):
    rows = deepcopy(_journal(schedule, session))
    for row in rows:
        for key in ('started_ns', 'ended_ns'):
            if key in row:
                row[key] += 3_000_000_000
        if row.get('stage') == 'preflight':
            row['checks']['value_layouts'][0]['device'] = 'cuda:0'
            row['checks']['output_device'] = 'cuda:0'
        if row.get('stage') in ('warmup', 'measured'):
            row['output_device'] = 'cuda:0'
            if row['mode'] != 'none': row['value_device'] = 'cuda:0'
    identity = {'run_id': f'{session + 1:032x}', 'pid': 10 + session, 'birth': str(1000 + session)}
    activity = _screen()
    activity['own_pid'] = identity['pid']
    return {'session': session, 'binding': binding, 'evidence_kind': 'supervised_cuda',
            'total_bytes': TOTAL, 'cap_bytes': CAP, 'free_before_model_transfer': 6 * protocol.GIB,
            'trace_claim_bytes': 128 * 1024, 'journal': rows, 'clock': CLOCK, 'activity': activity,
            'owner': {'unsafe': False, 'synchronized_disposal': True, 'whole_run_claims_held': True,
                      'claims_released_after_disposal': True, 'device_lease_held_through_process_exit': True,
                      'supervisor_control_held_until_reap': True, 'parent_observed_live_handle': True,
                      'child_identity': dict(identity), 'parent_identity': dict(identity),
                      'child_reaped': True, 'exit_code': 0}}


def test_attempt_aggregation_preserves_missing_and_contaminated_sessions():
    schedule = protocol.load_schedule(SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    attempts = [_attempt(schedule, i, binding) for i in range(3)]
    for index in (1, 2):
        for row in attempts[index]['journal']:
            if row.get('stage') == 'measured' and row.get('mode') == 'gradient':
                row['wall_ns'] *= index + 1
                row['ended_ns'] = row['started_ns'] + row['wall_ns']
    attempts[1]['activity']['process_queries'][1]['processes'] = [
        {'gpu_uuid': 'GPU-test', 'pid': 30, 'name': 'desktop'}]
    result = protocol.summarize_attempts(schedule, attempts, trusted_binding=binding)
    assert result['complete_sessions'] == 3
    assert [slot['state'] for slot in result['slots']] == ['complete', 'complete', 'complete', 'missing', 'missing']
    assert result['screen_groups'] == {'no_contamination_observed': [0, 2], 'contaminated': [1]}
    assert result['aggregate']['exploratory_due_to_contamination'] is True
    assert result['aggregate']['median_gradient_over_direct'] == 3.0
    assert result['slots'][1]['screen']['foreign_process_count'] == 1
    assert 'foreign_process_ids' not in result['slots'][1]['screen']


def test_incomplete_or_unverified_attempts_cannot_create_aggregate():
    schedule = protocol.load_schedule(SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    attempts = [_attempt(schedule, i, binding) for i in range(3)]
    attempts[1]['journal'].pop()
    attempts[2]['activity']['errors'].append({'error': 'failed query'})
    attempts[2]['activity']['process_queries'][1]['processes'] = [
        {'gpu_uuid': 'GPU-test', 'pid': 30, 'name': 'desktop'}]
    result = protocol.summarize_attempts(schedule, attempts, trusted_binding=binding)
    assert result['complete_sessions'] == 1 and result['aggregate'] is None
    assert [slot['state'] for slot in result['slots'][:3]] == ['complete', 'incomplete', 'incomplete']
    assert 'activity_screen_unverified' in result['slots'][2]['reason']
    assert result['slots'][2]['screen']['status'] == 'contaminated'
    assert result['slots'][2]['screen']['availability'] == 'unverified'
    assert result['slots'][2]['screen']['foreign_process_count'] == 1
    with pytest.raises(ValueError, match='duplicate'):
        protocol.summarize_attempts(schedule, [attempts[0], deepcopy(attempts[0])], trusted_binding=binding)


@pytest.mark.parametrize('damage', ['duplicate_run', 'duplicate_process', 'mismatched_parent',
                                    'partial_duplicate_run', 'partial_duplicate_process',
                                    'missing_parent', 'not_observed_live', 'bad_birth', 'wrong_activity_pid'])
def test_fresh_owner_identity_is_required_and_reuse_invalidates_all_affected_slots(damage):
    schedule = protocol.load_schedule(SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    attempts = [_attempt(schedule, i, binding) for i in range(3)]
    first, second = attempts[0]['owner'], attempts[1]['owner']
    if damage == 'duplicate_run':
        second['child_identity']['run_id'] = second['parent_identity']['run_id'] = first['child_identity']['run_id']
    if damage == 'duplicate_process':
        for field in ('child_identity', 'parent_identity'):
            second[field]['pid'] = first['child_identity']['pid']
            second[field]['birth'] = first['child_identity']['birth']
        attempts[1]['activity']['own_pid'] = first['child_identity']['pid']
    if damage == 'mismatched_parent':
        second['parent_identity'] = dict(first['parent_identity'])
    if damage == 'partial_duplicate_run':
        second['child_identity']['run_id'] = first['child_identity']['run_id']
        second['child_identity']['birth'] = 'invalid'
    if damage == 'partial_duplicate_process':
        second['child_identity']['run_id'] = 'invalid'
        second['child_identity']['pid'] = first['child_identity']['pid']
        second['child_identity']['birth'] = first['child_identity']['birth']
    if damage == 'missing_parent': second.pop('parent_identity')
    if damage == 'not_observed_live': second['parent_observed_live_handle'] = False
    if damage == 'bad_birth': second['child_identity']['birth'] = '0'
    if damage == 'wrong_activity_pid': attempts[1]['activity']['own_pid'] = 999
    result = protocol.summarize_attempts(schedule, attempts, trusted_binding=binding)
    assert result['slots'][1]['state'] == 'incomplete'
    assert result['slots'][1]['reason'] == 'owner_lifecycle_unverified'
    if damage in ('duplicate_run', 'duplicate_process', 'mismatched_parent',
                  'partial_duplicate_run', 'partial_duplicate_process'):
        assert result['slots'][0]['state'] == 'incomplete'
    assert result['aggregate'] is None
    assert 'child_identity' not in repr(result) and 'parent_identity' not in repr(result)


def test_pid_reuse_with_distinct_observed_creation_times_is_not_owner_reuse():
    schedule = protocol.load_schedule(SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    attempts = [_attempt(schedule, i, binding) for i in range(3)]
    for field in ('child_identity', 'parent_identity'):
        attempts[1]['owner'][field]['pid'] = attempts[0]['owner'][field]['pid']
    attempts[1]['activity']['own_pid'] = attempts[0]['activity']['own_pid']
    result = protocol.summarize_attempts(schedule, attempts, trusted_binding=binding)
    assert result['complete_sessions'] == 3 and result['aggregate'] is not None


def test_malformed_private_activity_key_is_not_exposed_in_public_projection():
    schedule = protocol.load_schedule(SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    attempt = _attempt(schedule, 0, binding)
    attempt['activity']['GPU-FAKE-PRIVATE-PID-987654'] = True
    result = protocol.summarize_attempts(schedule, [attempt], trusted_binding=binding)
    assert result['slots'][0]['reason'] == 'activity_record_missing_or_mismatched'
    assert 'GPU-FAKE-PRIVATE-PID-987654' not in repr(result)


def test_failed_owner_still_reports_a_valid_contaminated_screen():
    schedule = protocol.load_schedule(SCHEDULE)
    binding = {'source_commit': 'a' * 40, 'source_tree': 'b' * 40, 'model_sha256': 'c' * 64,
               'runtime_sha256': 'd' * 64, 'device_sha256': 'e' * 64}
    attempt = _attempt(schedule, 0, binding)
    attempt['owner']['unsafe'] = True
    attempt['activity']['process_queries'][1]['processes'] = [
        {'gpu_uuid': 'GPU-test', 'pid': 30, 'name': 'desktop'}]
    slot = protocol.summarize_attempts(schedule, [attempt], trusted_binding=binding)['slots'][0]
    assert slot['state'] == 'incomplete' and slot['reason'] == 'owner_lifecycle_unverified'
    assert slot['screen']['status'] == 'contaminated' and slot['screen']['foreign_process_count'] == 1
