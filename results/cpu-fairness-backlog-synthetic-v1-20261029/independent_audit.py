"""Independently replay complete synthetic CPU fairness windows; stdlib only."""
from collections import Counter, deque
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys


def distribution(values):
    ordered = sorted(values)
    return {'n': len(ordered), 'p50': statistics.median(ordered) if ordered else None,
            'p95': ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None,
            'max': ordered[-1] if ordered else None}


root = Path(sys.argv[1])
envelope = json.loads((root / 'plan.json').read_bytes())
plan = envelope['plan']
digest = sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=True, allow_nan=False).encode()).hexdigest()
assert digest == envelope['plan_sha256']
assert plan['teams'] == ['bulk', 'peer-b', 'peer-c']
assert (plan['jobs_per_team'], plan['model_hold_us'], plan['max_pending'],
        plan['max_pending_per_team'], plan['max_connections']) == (4, 30000, 12, 4, 12)
jobs = {job['job_id']: job for job in plan['jobs']}
assert len(jobs) == 12 and len(plan['baseline_order']) == 12
assert set(jobs) == set(plan['baseline_order']) == set(plan['backlog_offsets_us'])
for team, offsets in [('bulk', (0, 3000, 6000, 9000)),
                      ('peer-b', (15000, 75000, 135000, 195000)),
                      ('peer-c', (20000, 80000, 140000, 200000))]:
    team_jobs = [job['job_id'] for job in plan['jobs'] if job['team_id'] == team]
    assert len(team_jobs) == 4
    assert tuple(plan['backlog_offsets_us'][job] for job in team_jobs) == offsets
team_job_ids = {team: [job['job_id'] for job in plan['jobs'] if job['team_id'] == team]
                for team in plan['teams']}
assert plan['baseline_order'] == [team_job_ids[team][i] for i in range(4) for team in plan['teams']]

for mode in ('baseline', 'backlog'):
    path = root / (mode + '.json')
    raw = path.read_bytes()
    report = json.loads(raw)
    receipt_bytes = (root / (mode + '_watchdog.json')).read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt['version'] == 'cpu-fairness-watchdog-v1' and receipt['mode'] == mode
    assert receipt['child_reaped'] is True and receipt['timed_out'] is False
    assert type(receipt['returncode']) is int and receipt['returncode'] == 0
    assert receipt['cuda_visible_devices'] == '-1' and receipt['timeout_seconds'] == 45
    assert 0 < receipt['elapsed_watchdog_ns'] < 45 * 1_000_000_000
    assert receipt['started_utc'] < receipt['finished_utc']
    assert receipt['plan_canonical_sha256'] == digest
    assert receipt['plan_file_sha256'] == sha256((root / 'plan.json').read_bytes()).hexdigest()
    assert receipt['report_sha256'] == sha256(raw).hexdigest()
    for stream in ('stdout', 'stderr'):
        assert receipt[stream + '_sha256'] == sha256((root / (mode + '.' + stream)).read_bytes()).hexdigest()
    source = {'clean': True, 'commit': '9b6fb1a7d7140c9ef997159a7f1bf7360e877c71',
              'tree': '17addc9800ef0dd2c1a9d7a00139adc19c709f95'}
    assert receipt['source_before'] == receipt['source_after'] == source
    assert report['mode'] == mode and report['plan'] == plan and report['plan_sha256'] == digest
    assert report['runtime']['device'] == 'cpu' and report['runtime']['external_activity'] == 'unmeasured'
    assert report['clock']['monotonic'] is True and report['clock']['adjustable'] is False
    assert report['owner']['work_drained'] is True
    assert report['owner']['model_and_lease_disposal_attested'] is False
    assert report['owner']['synthetic_device_telemetry'] is True
    assert report['owner']['supervised_device_owner'] is False
    assert all(error is None for error in report['client_errors'])
    bounds = report['boundaries']
    start, end = bounds['start_ns'], bounds['end_ns']
    assert start < bounds['clients_joined_ns'] <= bounds['owner_drain_ns'] <= end
    duration = end - start
    summary = report['summary']
    assert duration == summary['elapsed_ns']
    recording = report['recording']
    events = recording['events']
    assert recording['events_complete'] is True and recording['recorded'] == len(events) == 87
    assert all(recording[field] == 0 for field in ('lost_capacity', 'lost_invalid',
               'lost_internal', 'late_records', 'explicit_faults'))
    assert Counter(event['kind'] for event in events) == {
        'attempt': 12, 'ready': 12, 'selected': 12, 'forward_entry': 12,
        'forward_finished': 12, 'broker_terminal': 12, 'returned': 12,
        'window_start': 1, 'window_end': 1, 'owner_drain': 1}
    assert all(start <= event['time_ns'] <= end for event in events)
    kinds = {kind: [event for event in events if event['kind'] == kind]
             for kind in {event['kind'] for event in events}}
    assert kinds['window_start'][0]['time_ns'] == start and kinds['window_end'][0]['time_ns'] == end
    assert kinds['owner_drain'][0] == {'kind': 'owner_drain', 'time_ns': bounds['owner_drain_ns'],
                                      'drained': True}
    samples = {sample['job_id']: sample for sample in report['samples']}
    assert len(samples) == len(report['samples']) == len(jobs) and set(samples) == set(jobs)
    indexed = {kind: {event['job_id']: event for event in kinds[kind]}
               for kind in ('attempt', 'ready', 'selected', 'broker_terminal', 'returned')}
    assert all(len(value) == len(jobs) and set(value) == set(jobs) for value in indexed.values())
    entries = {event['members'][0]['job_id']: event for event in kinds['forward_entry']}
    finishes = {event['call_id']: event for event in kinds['forward_finished']}
    assert set(entries) == set(jobs) and len(entries) == len(finishes) == len(jobs)
    identities, brokers, calls, queues = set(), set(), [], []
    for job_id, job in jobs.items():
        row = samples[job_id]
        attempt, ready, selected, terminal, returned = (indexed[kind][job_id] for kind in
            ('attempt', 'ready', 'selected', 'broker_terminal', 'returned'))
        entry = entries[job_id]
        assert len(entry['members']) == 1
        member, finish = entry['members'][0], finishes[entry['call_id']]
        identity = (row['broker_id'], row['broker_job_id'])
        assert identity not in identities and type(identity[1]) is int and identity[1] > 0
        identities.add(identity)
        brokers.add(identity[0])
        assert row['team_id'] == attempt['team_id'] == ready['team_id'] == member['team_id'] == job['team_id']
        assert (ready['broker_id'], ready['broker_job_id']) == (member['broker_id'], member['broker_job_id']) == identity
        assert attempt['input_sha256'] == job['input_sha256'] and attempt['request_sha256'] == job['request_sha256']
        assert row['output_sha256'] == returned['output_sha256'] == job['expected_output_sha256']
        assert row['state'] == terminal['state'] == returned['state'] == 'succeeded'
        assert row['error'] is None and row['client_cancellation_requested'] is False
        assert row['broker_cancellation_decision'] == 'unobserved'
        assert entry['profile_sha256'] == plan['profile_sha256'] and entry['input_shape'] == [1, 2]
        assert member['row_offset'] == 0 and entry['group_id'] == entry['call_id'] == f'lg-{identity[0]}-{identity[1]}'
        assert finish['state'] == 'succeeded' and finish['synchronized'] is True and finish['error'] is None
        assert (start <= row['attempt_ns'] == attempt['time_ns'] <= row['broker_submitted_ns']
                <= row['broker_ready_ns'] == ready['time_ns'] <= row['broker_started_ns'] == selected['time_ns']
                <= entry['time_ns'] <= finish['time_ns'] <= row['broker_finished_ns'] == terminal['time_ns']
                <= row['client_receipt_ns'] <= row['handled_ns'] == returned['time_ns'] <= bounds['owner_drain_ns'])
        assert row['broker_ready_ns'] <= row['accepted_receipt_ns'] <= row['client_receipt_ns']
        if mode == 'backlog':
            assert row['release_due_ns'] == start + plan['backlog_offsets_us'][job_id] * 1000
            assert row['attempt_ns'] >= row['release_due_ns']
        else:
            assert row['release_due_ns'] is None
        calls.append((entry['time_ns'], finish['time_ns']))
        turns = sum(other['job_id'] != job_id and ready['time_ns'] < other['time_ns'] < selected['time_ns']
                    for other in kinds['selected'])
        ties = sum(other['job_id'] != job_id and other['time_ns'] in (ready['time_ns'], selected['time_ns'])
                   for other in kinds['selected'])
        queues.append((job_id, job['team_id'], selected['time_ns'] - ready['time_ns'], turns, ties))
    assert len(brokers) == 1
    calls.sort()
    assert all(left[1] <= right[0] for left, right in zip(calls, calls[1:]))
    if mode == 'baseline':
        ordered = [samples[job_id] for job_id in plan['baseline_order']]
        assert all(left['handled_ns'] <= right['attempt_ns'] for left, right in zip(ordered, ordered[1:]))
    assert summary['planned'] == summary['attempted'] == summary['accepted_observed'] == summary['states']['succeeded'] == 12
    assert all(summary['states'][state] == 0 for state in summary['states'] if state != 'succeeded')
    assert summary['event_audit_errors'] == [] and summary['queue_observations_valid'] is True
    assert math.isclose(summary['successful_jobs_per_second'], 12e9 / duration)
    assert math.isclose(summary['successful_input_positions_per_second'], 24e9 / duration)
    qmap = {row['job_id']: row for row in summary['queue']['jobs']}
    assert len(qmap) == 12
    for job_id, team, wait, turns, ties in queues:
        row = qmap[job_id]
        assert (row['team_id'], row['wait_ns'], row['strictly_intervening_selections'], row['boundary_ties']) == (
            team, wait, turns, ties)
    assert summary['queue']['wait_ns'] == distribution([row[2] for row in queues])
    assert summary['timing_by_state']['succeeded']['client_receipt_ns'] == distribution(
        [row['client_receipt_ns'] - row['attempt_ns'] for row in samples.values()])
    assert summary['timing_by_state']['succeeded']['client_handled_ns'] == distribution(
        [row['handled_ns'] - row['attempt_ns'] for row in samples.values()])
    resolution = report['clock']['nominal_resolution_ns']
    assert type(resolution) is int and resolution > 0
    assert summary['nominal_resolution_screen'] == {
        'threshold_ns': 100 * resolution, 'window_short': duration < 100 * resolution,
        'client_receipt_intervals_observed': 12,
        'client_receipt_intervals_short': sum(row['client_receipt_ns'] - row['attempt_ns'] < 100 * resolution
                                              for row in samples.values()),
        'queue_wait_intervals_observed': 12,
        'queue_wait_intervals_short': sum(row[2] < 100 * resolution for row in queues),
        'forward_intervals_observed': 12,
        'forward_intervals_short': sum(finish - entry < 100 * resolution for entry, finish in calls)}
    for team in plan['teams']:
        team_rows = [sample for sample in samples.values() if sample['team_id'] == team]
        team_queues = [row for row in queues if row[1] == team]
        stats = summary['teams'][team]
        assert stats['planned'] == stats['attempted'] == stats['accepted_observed'] == stats['states']['succeeded'] == 4
        assert math.isclose(stats['successful_jobs_per_second'], 4e9 / duration)
        assert summary['queue']['by_team'][team]['wait_ns'] == distribution([row[2] for row in team_queues])
        assert summary['queue']['by_team'][team]['strictly_intervening_selections'] == distribution(
            [row[3] for row in team_queues])
        report_times = summary['source_timings_by_team_success'][team]
        for key, left, right in [
            ('attempt_to_broker_submission_ns', 'attempt_ns', 'broker_submitted_ns'),
            ('attempt_to_admission_receipt_ns', 'attempt_ns', 'accepted_receipt_ns'),
            ('broker_submission_to_ready_ns', 'broker_submitted_ns', 'broker_ready_ns'),
            ('ready_to_selection_ns', 'broker_ready_ns', 'broker_started_ns'),
            ('selection_to_terminal_ns', 'broker_started_ns', 'broker_finished_ns')]:
            assert report_times[key] == distribution([row[right] - row[left] for row in team_rows])
        assert report_times['model_entry_to_finished_ns'] == distribution([
            finishes[entries[row['job_id']]['call_id']]['time_ns'] - entries[row['job_id']]['time_ns']
            for row in team_rows])

    timeline = [(event['time_ns'], kind, job_id) for kind in ('ready', 'selected')
                for job_id, event in indexed[kind].items()]
    assert len({item[0] for item in timeline}) == len(timeline)
    pending, rotation, decisions = {}, deque(), []
    peak = peak_bulk = competition = coincident = 0
    for stamp, kind, job_id in sorted(timeline):
        team = jobs[job_id]['team_id']
        if kind == 'ready':
            if team not in pending:
                pending[team] = deque()
                rotation.append(team)
            pending[team].append(job_id)
            peak = max(peak, sum(map(len, pending.values())))
            peak_bulk = max(peak_bulk, len(pending.get('bulk', ())))
        else:
            assert rotation and rotation[0] == team and pending[team][0] == job_id
            eligible = list(rotation)
            depths = {item: len(pending[item]) for item in rotation}
            competition += len(eligible) > 1
            coincident += len(eligible) > 1 and depths.get('bulk', 0) >= 2
            decisions.append({'time_ns': stamp, 'job_id': job_id, 'team_id': team,
                              'expected_team_id': team, 'expected_job_id': job_id,
                              'eligible_teams': eligible, 'pending_by_team': depths})
            pending[team].popleft()
            rotation.popleft()
            if pending[team]:
                rotation.append(team)
            else:
                del pending[team]
    assert not pending and len(decisions) == 12
    replay = summary['rotation']
    assert replay == {'state': 'matches_round_robin_fifo', 'reason': None, 'decisions': decisions,
                      'peak_bulk_pending': peak_bulk, 'peak_pending': peak,
                      'competing_team_selections': competition,
                      'competing_selections_with_bulk_backlog': coincident}
    conditions = summary['conditions_realized']
    if mode == 'baseline':
        assert conditions == {'no_competing_experiment_clients': peak <= 1,
                              'bulk_backlog_and_peer_competition': None}
    else:
        assert conditions == {'no_competing_experiment_clients': None,
                              'bulk_backlog_and_peer_competition': coincident > 0}
    print(mode, 'sha256', sha256(raw).hexdigest(), 'elapsed_ms', duration / 1e6,
          'jobs_per_s', 12e9 / duration, 'peak_bulk_pending', peak_bulk,
          'competing_selections', competition, 'coincident', coincident,
          'team_queue_p50_ms', {team: statistics.median(row[2] for row in queues if row[1] == team) / 1e6
                                for team in plan['teams']})

print('PASS: complete 12-job windows, raw timestamps, source identities, rates, FIFO/rotation and conditions reconcile')
print('plan_sha256', digest)
