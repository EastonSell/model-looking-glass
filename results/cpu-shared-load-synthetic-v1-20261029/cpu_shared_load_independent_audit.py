"""Recompute a saved synthetic CPU shared-load report without service code."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
report_name = sys.argv[2]
plan_raw = (root / 'plan.json').read_bytes()
report_raw = (root / report_name).read_bytes()
envelope = json.loads(plan_raw)
report = json.loads(report_raw)
plan = envelope['plan']
digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=True, allow_nan=False).encode()).hexdigest()
assert digest == envelope['plan_sha256'] == report['plan_sha256']
assert digest == '18e23e94fd4263f2db5bc18794159063f34fcf34a2942134e5f83bc6ecdb2e2d'
assert plan == report['plan']
assert (plan['teams'], plan['jobs_per_team'], plan['model_hold_us'], plan['start_spacing_us']) == (
    ['alpha', 'beta', 'gamma'], 4, 20000, 0)
if len(sys.argv) > 3:
    receipt_raw = (root / sys.argv[3]).read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt['source_revision'] == '1d04f252860d9ed30d5e2d6b0b0b7c38e455cda1'
    assert receipt['plan_digest'] == digest
    assert receipt['report_sha256'] == hashlib.sha256(report_raw).hexdigest()
    assert receipt['process_status'] == 'exited' and type(receipt['returncode']) is int and receipt['returncode'] == 0
    assert receipt['timeout_seconds'] == 45 and receipt['cuda_visible_devices'] == '-1'
    print('receipt_sha256', hashlib.sha256(receipt_raw).hexdigest())

bounds = report['boundaries']
start, end = bounds['start_ns'], bounds['end_ns']
duration = end - start
assert start < bounds['clients_joined_ns'] <= bounds['owner_drain_ns'] <= end
assert duration == report['summary']['elapsed_ns']
assert report['owner']['work_drained'] is True
assert report['owner']['model_and_lease_disposal_attested'] is False
assert report['client_errors'] == [None] * 3
recording = report['recording']
events = recording['events']
assert recording['events_complete'] is True and recording['recorded'] == len(events) == 87
assert all(recording[field] == 0 for field in ('lost_capacity', 'lost_invalid', 'lost_internal',
                                                'late_records', 'explicit_faults'))
assert Counter(event['kind'] for event in events) == {
    'attempt': 12, 'ready': 12, 'selected': 12, 'forward_entry': 12, 'forward_finished': 12,
    'broker_terminal': 12, 'returned': 12, 'window_start': 1, 'window_end': 1, 'owner_drain': 1}
assert all(start <= event['time_ns'] <= end for event in events)
kinds = {kind: [event for event in events if event['kind'] == kind]
         for kind in {event['kind'] for event in events}}
assert kinds['window_start'][0]['time_ns'] == start and kinds['window_end'][0]['time_ns'] == end
assert kinds['owner_drain'][0]['time_ns'] == bounds['owner_drain_ns']
assert kinds['owner_drain'][0]['drained'] is True
jobs = {job['job_id']: job for job in plan['jobs']}
samples = {sample['job_id']: sample for sample in report['samples']}
assert len(jobs) == len(samples) == 12 and set(jobs) == set(samples)
indexed = {kind: {event['job_id']: event for event in kinds[kind]}
           for kind in ('attempt', 'ready', 'selected', 'broker_terminal', 'returned')}
assert all(len(value) == 12 and set(value) == set(jobs) for value in indexed.values())
entries = {event['members'][0]['job_id']: event for event in kinds['forward_entry']}
finishes = {event['call_id']: event for event in kinds['forward_finished']}
assert set(entries) == set(jobs) and len(finishes) == 12
identities, brokers, calls, queues = set(), set(), [], []
for job_id, job in jobs.items():
    sample = samples[job_id]
    attempt, ready, selected, terminal, returned = (indexed[kind][job_id] for kind in
        ('attempt', 'ready', 'selected', 'broker_terminal', 'returned'))
    entry, member = entries[job_id], entries[job_id]['members'][0]
    finish = finishes[entry['call_id']]
    identity = (sample['broker_id'], sample['broker_job_id'])
    assert identity not in identities and type(identity[1]) is int and identity[1] > 0
    identities.add(identity)
    brokers.add(identity[0])
    assert sample['team_id'] == attempt['team_id'] == ready['team_id'] == member['team_id'] == job['team_id']
    assert (ready['broker_id'], ready['broker_job_id']) == (member['broker_id'], member['broker_job_id']) == identity
    assert attempt['input_sha256'] == job['input_sha256'] and attempt['request_sha256'] == job['request_sha256']
    assert sample['output_sha256'] == returned['output_sha256'] == job['expected_output_sha256']
    assert sample['state'] == terminal['state'] == returned['state'] == 'succeeded'
    assert sample['error'] is None and sample['client_cancellation_requested'] is False
    assert sample['broker_cancellation_decision'] == 'unobserved'
    assert entry['profile_sha256'] == plan['profile_sha256'] and entry['input_shape'] == [1, 2]
    assert member['row_offset'] == 0 and entry['group_id'] == entry['call_id'] == f'lg-{identity[0]}-{identity[1]}'
    assert finish['state'] == 'succeeded' and finish['synchronized'] is True and finish['error'] is None
    assert (start <= sample['attempt_ns'] == attempt['time_ns'] <= ready['time_ns'] <= selected['time_ns']
            <= entry['time_ns'] <= finish['time_ns'] <= terminal['time_ns'] <= sample['client_receipt_ns']
            <= sample['handled_ns'] == returned['time_ns'] <= bounds['owner_drain_ns'])
    calls.append((entry['time_ns'], finish['time_ns']))
    turns = sum(other['job_id'] != job_id and ready['time_ns'] < other['time_ns'] < selected['time_ns']
                for other in kinds['selected'])
    ties = sum(other['job_id'] != job_id and other['time_ns'] in (ready['time_ns'], selected['time_ns'])
               for other in kinds['selected'])
    queues.append((job_id, job['team_id'], selected['time_ns'] - ready['time_ns'], turns, ties))
assert len(brokers) == 1
calls.sort()
assert all(left[1] <= right[0] for left, right in zip(calls, calls[1:]))
assert all(finish - entry >= 20000000 for entry, finish in calls)
for team in plan['teams']:
    rows = [samples[job['job_id']] for job in plan['jobs'] if job['team_id'] == team]
    assert len(rows) == 4 and all(left['handled_ns'] <= right['attempt_ns'] for left, right in zip(rows, rows[1:]))
summary = report['summary']
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


def distribution(values):
    ordered = sorted(values)
    return {'n': len(ordered), 'p50': statistics.median(ordered) if ordered else None,
            'p95': ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None,
            'max': ordered[-1] if ordered else None}


assert summary['queue']['wait_ns'] == distribution([row[2] for row in queues])
assert summary['timing_by_state']['succeeded']['client_receipt_ns'] == distribution(
    [row['client_receipt_ns'] - row['attempt_ns'] for row in samples.values()])
assert summary['timing_by_state']['succeeded']['client_handled_ns'] == distribution(
    [row['handled_ns'] - row['attempt_ns'] for row in samples.values()])
resolution = report['clock']['nominal_resolution_ns']
assert report['clock']['monotonic'] is True and report['clock']['adjustable'] is False
assert summary['nominal_resolution_screen'] == {
    'threshold_ns': 100 * resolution, 'window_short': duration < 100 * resolution,
    'client_receipt_intervals_observed': 12,
    'client_receipt_intervals_short': sum(
        row['client_receipt_ns'] - row['attempt_ns'] < 100 * resolution for row in samples.values()),
    'queue_wait_intervals_observed': 12,
    'queue_wait_intervals_short': sum(row[2] < 100 * resolution for row in queues),
}
for team in plan['teams']:
    stats = summary['teams'][team]
    rows = [sample for sample in samples.values() if sample['team_id'] == team]
    queue = [row for row in queues if row[1] == team]
    assert stats['planned'] == stats['attempted'] == stats['accepted_observed'] == stats['states']['succeeded'] == 4
    assert math.isclose(stats['successful_jobs_per_second'], 4e9 / duration)
    assert stats['timing_by_state']['succeeded']['client_receipt_ns'] == distribution(
        [row['client_receipt_ns'] - row['attempt_ns'] for row in rows])
    assert stats['timing_by_state']['succeeded']['client_handled_ns'] == distribution(
        [row['handled_ns'] - row['attempt_ns'] for row in rows])
    assert summary['queue']['by_team'][team]['wait_ns'] == distribution([row[2] for row in queue])
    assert summary['queue']['by_team'][team]['strictly_intervening_selections'] == distribution(
        [row[3] for row in queue])
print('PASS: 12/12 success; 87 events; one broker; nonoverlapping calls; identities, timing, turns and rates reconcile')
print('window_ns', duration, 'jobs_per_second', 12e9 / duration,
      'positions_per_second', 24e9 / duration, 'per_team', 4e9 / duration)
print('forward_duration_ns min/median/max', min(b - a for a, b in calls),
      statistics.median(b - a for a, b in calls), max(b - a for a, b in calls))
print('team_queue_wait_medians_ms', {team: statistics.median(row[2] for row in queues if row[1] == team) / 1e6
                                     for team in plan['teams']})
print('report_sha256', hashlib.sha256(report_raw).hexdigest())
