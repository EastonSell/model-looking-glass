"""Bounded synthetic CPU queue-rotation study through the observed service.

This measures only a trusted local synthetic CPU fixture. It does not use or
certify a GPU, authenticate its own events, establish network-wide fairness,
or establish equal device time. Use one separately watched process per window.
"""

import argparse
from collections import deque
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import threading
import time

import torch

from . import cpu_shared_load as shared
from .ipc import IPCClient, IPCRejected
from .throughput_benchmark import _distribution


_VERSION = 'synthetic-cpu-fairness-backlog-v1'
_TEAMS = ('bulk', 'peer-b', 'peer-c')
_OFFSETS_US = {'bulk': (0, 3000, 6000, 9000),
               'peer-b': (15000, 75000, 135000, 195000),
               'peer-c': (20000, 80000, 140000, 200000)}


def make_plan(*, seed=17, model_hold_us=30000):
    """Freeze the same jobs and distinct release policies for two windows."""
    base, _ = shared.make_plan(_TEAMS, jobs_per_team=4, seed=seed,
                               model_hold_us=model_hold_us)
    jobs = base['jobs']
    by_team = {team: [job for job in jobs if job['team_id'] == team] for team in _TEAMS}
    baseline = [by_team[team][index]['job_id'] for index in range(4) for team in _TEAMS]
    offsets = {job['job_id']: _OFFSETS_US[team][index]
               for team in _TEAMS for index, job in enumerate(by_team[team])}
    plan = {'version': _VERSION, 'teams': list(_TEAMS), 'jobs_per_team': 4,
            'seed': seed, 'model_hold_us': model_hold_us,
            'model_spec': base['model_spec'], 'model_sha256': base['model_sha256'],
            'profile_sha256': base['profile_sha256'], 'jobs': jobs,
            'max_pending': 12, 'max_pending_per_team': 4, 'max_connections': 12,
            'baseline_order': baseline, 'backlog_offsets_us': offsets}
    return plan, sha256(shared._canonical_bytes(plan)).hexdigest()


def _snapshot_plan(plan, digest):
    if type(plan) is not dict or type(digest) is not str:
        raise ValueError('an exact plan and separately retained digest are required')
    try:
        payload = shared._canonical_bytes(plan)
        if len(payload) > 128 * 1024 or sha256(payload).hexdigest() != digest:
            raise ValueError('plan differs from its retained digest')
        snapshot = json.loads(payload)
        expected, expected_digest = make_plan(seed=snapshot['seed'], model_hold_us=snapshot['model_hold_us'])
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError('invalid fairness plan') from exc
    if snapshot != expected or expected_digest != digest:
        raise ValueError('plan differs from the supported synthetic workload')
    return snapshot


def _sample(job):
    row = shared._sample(job)
    row.update({'release_due_ns': None, 'accepted_receipt_ns': None,
                'broker_submitted_ns': None, 'broker_ready_ns': None,
                'broker_started_ns': None, 'broker_finished_ns': None})
    return row


def _one_client_job(owner, plan, client, job, row):
    stamp = time.perf_counter_ns()
    row['attempt_ns'] = stamp
    shared._record(owner.observation, {'kind': 'attempt', 'time_ns': stamp,
        'job_id': job['job_id'], 'team_id': job['team_id'],
        'input_sha256': job['input_sha256'], 'request_sha256': job['request_sha256']})
    handle = None
    try:
        handle = client.submit(team_id=job['team_id'], request_id=job['job_id'],
                               inputs={'input_ids': [list(job['input_ids'])]}, seed=plan['seed'])
        row['accepted_receipt_ns'] = time.perf_counter_ns()
        row['broker_id'], row['broker_job_id'] = handle.broker_id, handle.job_id
        response = handle.wait(timeout=12.0)
        row['client_receipt_ns'] = time.perf_counter_ns()
        for field in ('submitted', 'ready', 'started', 'finished'):
            row[f'broker_{field}_ns'] = response.get(f'{field}_ns')
        state = response['status']
        if state == 'succeeded':
            try:
                row['output_sha256'] = shared._output_digest(response, job)
                row['state'] = 'succeeded'
            except (TypeError, ValueError, RuntimeError):
                row['state'], row['error'] = 'invalid_output', 'invalid_synthetic_output'
        elif state == 'failed':
            row['state'], row['error'] = 'failed', str(response.get('error', 'failed'))[:128]
        elif state == 'cancelled':
            row['state'], row['error'] = 'cancelled', 'unobserved_cancellation_decision'
            owner.observation._fault()
        else:
            row['state'], row['error'] = 'unresolved', 'server_timeout'
            owner.observation._fault()
    except IPCRejected as exc:
        row['client_receipt_ns'] = time.perf_counter_ns()
        code = exc.code if type(exc.code) is str else 'unknown_rejection'
        if code in ('busy', 'queue_full', 'invalid_request'):
            row['state'], row['error'] = 'rejected', code
        else:
            row['state'], row['error'] = 'unresolved', code[:128]
            owner.observation._fault()
    except Exception as exc:
        row['state'], row['error'] = 'unresolved', type(exc).__name__[:128]
        owner.observation._fault()
    finally:
        if handle is not None:
            handle.close()
    row['handled_ns'] = time.perf_counter_ns()
    if row['state'] not in ('unresolved', 'cancelled'):
        shared._record(owner.observation, {'kind': 'returned', 'time_ns': row['handled_ns'],
            'job_id': job['job_id'], 'state': row['state'], 'error': row['error'],
            'output_sha256': row['output_sha256'], 'cancellation_requested': False,
            'cancellation_requested_ns': None, 'cancellation_accepted': None,
            'cancellation_decision_ns': None})


def _producer(owner, plan, jobs, rows, *, gate, ready, anchor, error_index, mode):
    current = None
    try:
        clients = {team: IPCClient(owner.server.address, owner.server.team_tokens[team], io_timeout=3.0,
                                   max_response_bytes=4096) for team in {job['team_id'] for job in jobs}}
        ready.set()
        gate.wait()
        for job, current in zip(jobs, rows):
            if mode == 'backlog':
                due = anchor[0] + plan['backlog_offsets_us'][job['job_id']] * 1000
                current['release_due_ns'] = due
                while True:
                    left = due - time.perf_counter_ns()
                    if left <= 0:
                        break
                    time.sleep(left / 1_000_000_000)
            _one_client_job(owner, plan, clients[job['team_id']], job, current)
            if current['state'] == 'unresolved':
                break
    except BaseException as exc:
        owner.client_errors[error_index] = exc
        owner.observation._fault()
        if current is not None and current['attempt_ns'] is not None and current['state'] == 'unattempted':
            current['state'], current['error'] = 'unresolved', 'client_producer_interrupted'
            current['handled_ns'] = time.perf_counter_ns()
        ready.set()


def replay_rotation(plan, samples, by_kind, *, evidence_complete):
    """Replay source stamps after the caller has audited complete event evidence.

    ``evidence_complete`` is a trusted caller attestation; this replay does not
    independently authenticate the recorder, broker, or event contents.
    """
    expected = {job['job_id']: job['team_id'] for job in plan['jobs']}
    if (evidence_complete is not True or type(samples) is not list or len(samples) != len(expected)
            or any(type(row) is not dict or row.get('job_id') not in expected
                   or row.get('team_id') != expected[row['job_id']] or row.get('state') != 'succeeded'
                   or row.get('client_cancellation_requested') is not False
                   or row.get('broker_cancellation_decision') != 'unobserved' for row in samples)
            or len({row['job_id'] for row in samples}) != len(expected)):
        return {'state': 'inconclusive', 'reason': 'incomplete_or_nonsuccessful_work',
                'decisions': [], 'peak_bulk_pending': None, 'peak_pending': None,
                'competing_team_selections': None, 'competing_selections_with_bulk_backlog': None}
    ready, selected = by_kind.get('ready', {}), by_kind.get('selected', {})
    if set(ready) != {job['job_id'] for job in plan['jobs']} or set(ready) != set(selected):
        return {'state': 'inconclusive', 'reason': 'missing_ready_or_selection',
                'decisions': [], 'peak_bulk_pending': None, 'peak_pending': None,
                'competing_team_selections': None, 'competing_selections_with_bulk_backlog': None}
    timeline = [(event['time_ns'], 'ready', job_id) for job_id, event in ready.items()]
    timeline.extend((event['time_ns'], 'selected', job_id) for job_id, event in selected.items())
    if len({stamp for stamp, _, _ in timeline}) != len(timeline):
        return {'state': 'inconclusive', 'reason': 'ready_or_selection_timestamp_tie',
                'decisions': [], 'peak_bulk_pending': None, 'peak_pending': None,
                'competing_team_selections': None, 'competing_selections_with_bulk_backlog': None}
    team_by_job = {job['job_id']: job['team_id'] for job in plan['jobs']}
    pending, rotation, decisions = {}, deque(), []
    peak_bulk = peak = competing = competing_with_bulk_backlog = 0
    violation = False
    for stamp, kind, job_id in sorted(timeline):
        team = team_by_job[job_id]
        if kind == 'ready':
            if team not in pending:
                pending[team] = deque()
                rotation.append(team)
            pending[team].append(job_id)
            peak = max(peak, sum(len(queue) for queue in pending.values()))
            peak_bulk = max(peak_bulk, len(pending.get('bulk', ())))
            continue
        if not rotation or team not in pending:
            violation = True
            break
        expected_team, expected_job = rotation[0], pending[rotation[0]][0]
        eligible = list(rotation)
        depths = {key: len(pending[key]) for key in rotation}
        competing += len(eligible) > 1
        competing_with_bulk_backlog += len(eligible) > 1 and depths.get('bulk', 0) >= 2
        decisions.append({'time_ns': stamp, 'job_id': job_id, 'team_id': team,
                          'expected_team_id': expected_team, 'expected_job_id': expected_job,
                          'eligible_teams': eligible, 'pending_by_team': depths})
        if team != expected_team or job_id != expected_job:
            violation = True
            break
        pending[team].popleft()
        rotation.popleft()
        if pending[team]:
            rotation.append(team)
        else:
            del pending[team]
    if pending or len(decisions) != len(plan['jobs']):
        violation = True
    return {'state': 'violation' if violation else 'matches_round_robin_fifo',
            'reason': 'source_order_mismatch' if violation else None, 'decisions': decisions,
            'peak_bulk_pending': peak_bulk, 'peak_pending': peak,
            'competing_team_selections': competing,
            'competing_selections_with_bulk_backlog': competing_with_bulk_backlog}


def _analyze(plan, samples, recording, start, end, drained, resolution_ns, mode):
    summary = shared._summarize(plan, samples, recording, start, end, drained, resolution_ns,
                                closed_loop=(mode == 'baseline'))
    errors, by_kind = shared._audit_events(plan, samples, recording, start, end, drained,
                                           closed_loop=(mode == 'baseline'))
    for row in samples:
        state, attempt = row['state'], row['attempt_ns']
        if mode == 'backlog' and attempt is not None:
            expected_due = start + plan['backlog_offsets_us'][row['job_id']] * 1000
            if row['release_due_ns'] != expected_due or attempt < expected_due:
                errors.append('release_schedule_mismatch')
        elif mode == 'baseline' and row['release_due_ns'] is not None:
            errors.append('baseline_has_scheduled_release')
        if row['broker_id'] is None:
            if row['accepted_receipt_ns'] is not None or any(row[f'broker_{field}_ns'] is not None
                                                               for field in ('submitted', 'ready', 'started', 'finished')):
                errors.append('unaccepted_sample_has_broker_stamps')
            continue
        accepted = row['accepted_receipt_ns']
        if (type(accepted) is not int or type(attempt) is not int or type(row['client_receipt_ns']) is not int
                or not attempt <= accepted <= row['client_receipt_ns']):
            errors.append('client_admission_receipt_invalid')
        if state in ('succeeded', 'failed', 'invalid_output'):
            r, s, t = (by_kind.get(key, {}).get(row['job_id'])
                       for key in ('ready', 'selected', 'broker_terminal'))
            stamps = [row[f'broker_{field}_ns'] for field in ('submitted', 'ready', 'started', 'finished')]
            if (any(type(value) is not int for value in stamps) or r is None or s is None or t is None
                    or type(attempt) is not int or type(row['client_receipt_ns']) is not int
                    or type(accepted) is not int or stamps[1] > accepted
                    or not attempt <= stamps[0] <= stamps[1] <= stamps[2] <= stamps[3] <= row['client_receipt_ns']
                    or [r['time_ns'], s['time_ns'], t['time_ns']] != stamps[1:]):
                errors.append('broker_source_stamp_mismatch')
    if mode == 'baseline':
        rows = {row['job_id']: row for row in samples}
        ordered = [rows[key] for key in plan['baseline_order'] if key in rows]
        if any(a['handled_ns'] is None or b['attempt_ns'] is None or a['handled_ns'] > b['attempt_ns']
               for a, b in zip(ordered, ordered[1:])):
            errors.append('baseline_clients_overlap_or_unresolved')
    errors = sorted(set(errors))
    summary['event_audit_errors'] = errors
    summary['queue_observations_valid'] = not errors
    if errors:
        # Never publish queue summaries already computed by the shared auditor
        # when an additional admission/source-stamp check failed.
        summary['queue'] = {'jobs': [], 'wait_ns': _distribution([]), 'by_team': {}}
        summary['nominal_resolution_screen']['queue_wait_intervals_observed'] = 0
        summary['nominal_resolution_screen']['queue_wait_intervals_short'] = 0
    replay = replay_rotation(plan, samples, by_kind, evidence_complete=not errors)
    summary['rotation'] = replay
    # A failed or incomplete replay cannot prove either that a workload
    # condition happened or that it did not. Preserve this as unknown.
    matched = replay['state'] == 'matches_round_robin_fifo'
    summary['conditions_realized'] = {
        'no_competing_experiment_clients': replay['peak_pending'] <= 1
            if mode == 'baseline' and matched else None,
        'bulk_backlog_and_peer_competition': replay['competing_selections_with_bulk_backlog'] >= 1
            if mode == 'backlog' and matched else None}
    per_team = {}
    forwards = {}
    if not errors:
        for call_id, entry in by_kind.get('forward_entry', {}).items():
            finish = by_kind.get('forward_finished', {}).get(call_id)
            if finish is not None:
                forwards[entry['members'][0]['job_id']] = finish['time_ns'] - entry['time_ns']
    summary['nominal_resolution_screen']['forward_intervals_observed'] = len(forwards)
    summary['nominal_resolution_screen']['forward_intervals_short'] = sum(
        value < 100 * resolution_ns for value in forwards.values())
    for team in plan['teams']:
        eligible = [row for row in samples if not errors and row['team_id'] == team
                    and row['state'] == 'succeeded']
        def elapsed(a, b):
            return _distribution([row[b] - row[a] for row in eligible
                                  if type(row[a]) is int and type(row[b]) is int])
        per_team[team] = {'attempt_to_broker_submission_ns': elapsed('attempt_ns', 'broker_submitted_ns'),
                          'attempt_to_admission_receipt_ns': elapsed('attempt_ns', 'accepted_receipt_ns'),
                          'broker_submission_to_ready_ns': elapsed('broker_submitted_ns', 'broker_ready_ns'),
                          'ready_to_selection_ns': elapsed('broker_ready_ns', 'broker_started_ns'),
                          'selection_to_terminal_ns': elapsed('broker_started_ns', 'broker_finished_ns'),
                          'model_entry_to_finished_ns': _distribution([forwards[row['job_id']] for row in eligible
                                                                      if row['job_id'] in forwards])}
    summary['source_timings_by_team_success'] = per_team
    return summary


def run_window(plan, *, expected_plan_sha256, mode, timeout=30.0):
    """Run one bounded window; an uncertain owner remains rooted for exit."""
    if mode not in ('baseline', 'backlog'):
        raise ValueError('mode must be baseline or backlog')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError('timeout must be finite and between zero and 120 seconds')
    plan = _snapshot_plan(plan, expected_plan_sha256)
    info = time.get_clock_info('perf_counter')
    if (info.monotonic is not True or info.adjustable is not False
            or type(info.resolution) not in (int, float) or not math.isfinite(info.resolution)
            or info.resolution <= 0):
        raise ValueError('a monotonic non-adjustable clock with positive finite resolution is required')
    resolution = max(1, math.ceil(info.resolution * 1_000_000_000))
    deadline = time.perf_counter_ns() + int(timeout * 1_000_000_000)
    owner = shared._claim_owner()
    clean = False
    try:
        shared._build_owner(owner, plan)
        jobs = {job['job_id']: job for job in plan['jobs']}
        rows = {job['job_id']: _sample(job) for job in plan['jobs']}
        batches = ([plan['baseline_order']] if mode == 'baseline' else [[job['job_id']] for job in plan['jobs']])
        owner.client_errors = [None] * len(batches)
        gate, anchor = threading.Event(), [None]
        ready = [threading.Event() for _ in batches]
        for index, batch in enumerate(batches):
            thread = threading.Thread(target=_producer,
                args=(owner, plan, [jobs[key] for key in batch], [rows[key] for key in batch]),
                kwargs={'gate': gate, 'ready': ready[index], 'anchor': anchor,
                        'error_index': index, 'mode': mode}, daemon=True, name=f'cpu-fair-{index}')
            owner.clients.append(thread)
            thread.start()
        for event in ready:
            if not event.wait(shared._remaining(deadline)):
                raise shared.ExperimentUnresolved('client readiness did not complete')
        if any(error is not None for error in owner.client_errors):
            raise shared.ExperimentUnresolved('a client failed before common release')
        start = time.perf_counter_ns()
        anchor[0] = start
        shared._record(owner.observation, {'kind': 'window_start', 'time_ns': start})
        gate.set()
        for thread in owner.clients:
            thread.join(shared._remaining(deadline))
        if any(thread.is_alive() for thread in owner.clients):
            raise shared.ExperimentUnresolved('client threads did not stop')
        for error in owner.client_errors:
            if error is not None and not isinstance(error, Exception):
                if isinstance(error, KeyboardInterrupt):
                    raise KeyboardInterrupt('client producer interrupted') from None
                if isinstance(error, SystemExit):
                    raise SystemExit('client producer interrupted') from None
                raise BaseException('client producer interrupted') from None
        joined = time.perf_counter_ns()
        if not owner.server.stop_accepting(timeout=shared._remaining(deadline)):
            raise shared.ExperimentUnresolved('IPC admission did not stop')
        if not owner.server.drain(timeout=shared._remaining(deadline)):
            raise shared.ExperimentUnresolved('IPC handlers did not drain')
        if not owner.server.close(timeout=shared._remaining(deadline)):
            raise shared.ExperimentUnresolved('IPC server did not close')
        if not owner.broker.close(timeout=shared._remaining(deadline)):
            raise shared.ExperimentUnresolved('broker owner did not stop')
        with owner.broker._condition:
            drained = (not owner.broker._thread.is_alive() and owner.broker._running is None
                       and owner.broker._pending_slots == 0 and owner.broker._preparing == 0
                       and len(owner.broker._queue) == 0
                       and all(job is None for job in owner.broker._job_roots))
        names = ('request', 'logits', 'result', 'error', 'producer_error', 'forward_error', 'state_error')
        drained = (drained and not owner.authority.unsafe and not owner.broker.quarantined
                   and not owner.worker.quarantined and not owner.server.custody_quarantined
                   and not owner.resident._drain_pending and owner.resident._snapshot_error is None
                   and all(value is None for value in owner.resident._live_buffers)
                   and all(value is None for value in owner.resident._retiring_buffers)
                   and all(getattr(owner.worker._custody, name) is None for name in names)
                   and owner.worker._warmup_result is None and owner.worker._retiring_cancel is None
                   and owner.worker._retiring_transfer is None
                   and owner.worker._cancel_cleanup_error is None and owner.worker._transfer_cleanup_error is None)
        drain_stamp = time.perf_counter_ns()
        shared._record(owner.observation, {'kind': 'owner_drain', 'time_ns': drain_stamp, 'drained': bool(drained)})
        end = time.perf_counter_ns()
        if end <= start:
            owner.observation._fault()
            raise shared.ExperimentUnresolved('measurement clock did not advance')
        shared._record(owner.observation, {'kind': 'window_end', 'time_ns': end})
        recording = owner.observation.seal(producers_stopped=True)
        samples = [rows[job['job_id']] for job in plan['jobs']]
        summary = _analyze(plan, samples, recording, start, end, bool(drained), resolution, mode)
        report = {'version': _VERSION, 'mode': mode, 'plan': plan, 'plan_sha256': expected_plan_sha256,
                  'clock': {'name': 'perf_counter_ns', 'implementation': info.implementation,
                            'monotonic': info.monotonic, 'adjustable': info.adjustable,
                            'nominal_resolution_ns': resolution},
                  'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                              'torch_num_threads': torch.get_num_threads(), 'device': 'cpu',
                              'external_activity': 'unmeasured'},
                  'boundaries': {'start_ns': start, 'clients_joined_ns': joined,
                                 'owner_drain_ns': drain_stamp, 'end_ns': end},
                  'owner': {'work_drained': bool(drained), 'model_and_lease_disposal_attested': False,
                            'synthetic_device_telemetry': True, 'supervised_device_owner': False,
                            'resource_profile_verified': False},
                  'client_errors': [type(error).__name__ if error is not None else None
                                    for error in owner.client_errors],
                  'samples': samples, 'recording': recording, 'summary': summary}
        if not drained:
            raise shared.ExperimentUnresolved('CPU work or owner health remains uncertain')
        clean = True
        return report
    finally:
        if not clean:
            if owner.server is not None:
                try:
                    owner.server.close(timeout=min(0.5, shared._remaining(deadline)))
                except Exception:
                    pass
            if owner.broker is not None:
                try:
                    owner.broker.close(timeout=min(0.5, shared._remaining(deadline)))
                except Exception:
                    pass
        if clean:
            shared._release_owner(owner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('plan')
    freeze.add_argument('--seed', type=int, default=17)
    freeze.add_argument('--model-hold-us', type=int, default=30000)
    freeze.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--sha256', required=True)
    run.add_argument('--mode', choices=('baseline', 'backlog'), required=True)
    run.add_argument('--timeout', type=float, default=30.0)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        plan, digest = make_plan(seed=args.seed, model_hold_us=args.model_hold_us)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump({'plan': plan, 'plan_sha256': digest}, stream, sort_keys=True,
                      ensure_ascii=True, allow_nan=False, indent=2)
            stream.write('\n')
        print(digest)
        return
    with args.plan.open('rb') as stream:
        payload = stream.read(128 * 1024 + 1)
    if len(payload) > 128 * 1024:
        raise ValueError('plan file exceeds its byte limit')
    envelope = json.loads(payload)
    if (type(envelope) is not dict or set(envelope) != {'plan', 'plan_sha256'}
            or envelope['plan_sha256'] != args.sha256):
        raise ValueError('plan envelope and retained digest do not match')
    _snapshot_plan(envelope['plan'], args.sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        report = run_window(envelope['plan'], expected_plan_sha256=args.sha256,
                            mode=args.mode, timeout=args.timeout)
        json.dump(report, stream, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
