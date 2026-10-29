"""CPU synthetic queue observations; no GPU or pretrained model is used."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from looking_glass import cpu_fairness_backlog as study
from looking_glass import cpu_shared_load as shared
from looking_glass.ipc import IPCClient, IPCRejected


@pytest.fixture(scope='module')
def real_windows():
    plan, digest = study.make_plan()
    with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA was queried')):
        baseline = study.run_window(plan, expected_plan_sha256=digest, mode='baseline')
        backlog = study.run_window(plan, expected_plan_sha256=digest, mode='backlog')
    return plan, baseline, backlog


def test_real_windows_use_same_frozen_work_and_observe_source_stamps(real_windows):
    plan, baseline, backlog = real_windows
    assert baseline['plan'] == backlog['plan'] == plan
    assert baseline['plan_sha256'] == backlog['plan_sha256']
    assert [plan['max_pending'], plan['max_pending_per_team'], plan['max_connections']] == [12, 4, 12]
    for report in (baseline, backlog):
        summary = report['summary']
        assert report['owner']['work_drained'] is True
        assert report['owner']['model_and_lease_disposal_attested'] is False
        assert report['runtime']['device'] == 'cpu'
        assert summary['states']['succeeded'] == 12
        assert summary['planned'] == summary['attempted'] == summary['accepted_observed'] == 12
        assert summary['event_audit_errors'] == []
        assert report['recording']['events_complete'] is True
        assert summary['rotation']['state'] == 'matches_round_robin_fifo'
        assert len(summary['rotation']['decisions']) == 12
        assert len({row['broker_id'] for row in report['samples']}) == 1
        assert len({(row['broker_id'], row['broker_job_id']) for row in report['samples']}) == 12
        for team in plan['teams']:
            assert summary['teams'][team]['states']['succeeded'] == 4
            assert summary['source_timings_by_team_success'][team]['ready_to_selection_ns']['n'] == 4
            assert summary['source_timings_by_team_success'][team]['model_entry_to_finished_ns']['n'] == 4
            assert summary['teams'][team]['successful_jobs_per_second'] == pytest.approx(
                4e9 / summary['elapsed_ns'])
        for row in report['samples']:
            assert row['attempt_ns'] <= row['accepted_receipt_ns'] <= row['client_receipt_ns']
            assert row['broker_ready_ns'] <= row['accepted_receipt_ns']
            assert (row['attempt_ns'] <= row['broker_submitted_ns'] <= row['broker_ready_ns']
                    <= row['broker_started_ns'] <= row['broker_finished_ns'] <= row['client_receipt_ns'])
            assert row['client_cancellation_requested'] is False
            assert row['broker_cancellation_decision'] == 'unobserved'
    assert baseline['summary']['conditions_realized']['no_competing_experiment_clients'] is True
    # The workload is scheduled, not guaranteed to obtain queue contention on a
    # particular host. Preserve the actual flag instead of assuming it occurred.
    assert type(backlog['summary']['conditions_realized']['bulk_backlog_and_peer_competition']) is bool


def test_frozen_plan_and_mode_reject_changes_before_owner_start():
    plan, digest = study.make_plan()
    plan['backlog_offsets_us']['job-0001'] += 1
    with pytest.raises(ValueError, match='digest'):
        study.run_window(plan, expected_plan_sha256=digest, mode='backlog')
    assert shared._active_owner is None
    plan, digest = study.make_plan()
    with pytest.raises(ValueError, match='mode'):
        study.run_window(plan, expected_plan_sha256=digest, mode='other')
    assert shared._active_owner is None


def _fabricated_rotation():
    plan, _ = study.make_plan()
    by_team = {team: [job for job in plan['jobs'] if job['team_id'] == team] for team in plan['teams']}
    order = [by_team[team][index] for index in range(4) for team in plan['teams']]
    ready = {job['job_id']: {'time_ns': 100 + index} for index, job in enumerate(order)}
    selected = {job['job_id']: {'time_ns': 200 + index} for index, job in enumerate(order)}
    samples = [dict(shared._sample(job), state='succeeded') for job in plan['jobs']]
    return plan, samples, {'ready': ready, 'selected': selected}


def test_rotation_replay_checks_fifo_teams_and_realized_competition():
    plan, samples, events = _fabricated_rotation()
    report = study.replay_rotation(plan, samples, events, evidence_complete=True)
    assert report['state'] == 'matches_round_robin_fifo'
    assert report['peak_bulk_pending'] == 4 and report['peak_pending'] == 12
    assert report['competing_team_selections'] > 0
    assert report['competing_selections_with_bulk_backlog'] > 0
    teams = [decision['team_id'] for decision in report['decisions']]
    assert teams == ['bulk', 'peer-b', 'peer-c'] * 4


def test_rotation_mismatch_is_not_reported_as_a_match():
    plan, samples, events = _fabricated_rotation()
    first, second = plan['baseline_order'][:2]
    events['selected'][first]['time_ns'], events['selected'][second]['time_ns'] = (
        events['selected'][second]['time_ns'], events['selected'][first]['time_ns'])
    report = study.replay_rotation(plan, samples, events, evidence_complete=True)
    assert report['state'] == 'violation'
    assert report['reason'] == 'source_order_mismatch'


def test_tied_or_incomplete_source_events_are_inconclusive():
    plan, samples, events = _fabricated_rotation()
    first, second = plan['baseline_order'][:2]
    events['ready'][second]['time_ns'] = events['ready'][first]['time_ns']
    result = study.replay_rotation(plan, samples, events, evidence_complete=True)
    assert result['state'] == 'inconclusive' and result['reason'] == 'ready_or_selection_timestamp_tie'
    result = study.replay_rotation(plan, samples, events, evidence_complete=False)
    assert result['state'] == 'inconclusive' and result['reason'] == 'incomplete_or_nonsuccessful_work'


def test_backlog_and_competition_must_coincide_at_the_same_selection():
    plan, samples, _ = _fabricated_rotation()
    by_team = {team: [job['job_id'] for job in plan['jobs'] if job['team_id'] == team]
               for team in plan['teams']}
    ready, selected = {}, {}
    sequence = [('ready', by_team['bulk'][0]), ('ready', by_team['bulk'][1]),
                ('selected', by_team['bulk'][0]), ('selected', by_team['bulk'][1]),
                ('ready', by_team['peer-b'][0]), ('ready', by_team['peer-c'][0]),
                ('selected', by_team['peer-b'][0]), ('selected', by_team['peer-c'][0])]
    used = {job for _, job in sequence}
    for job in plan['jobs']:
        if job['job_id'] not in used:
            sequence.extend([('ready', job['job_id']), ('selected', job['job_id'])])
    for stamp, (kind, job_id) in enumerate(sequence, 1):
        (ready if kind == 'ready' else selected)[job_id] = {'time_ns': stamp}
    report = study.replay_rotation(plan, samples, {'ready': ready, 'selected': selected},
                                   evidence_complete=True)
    assert report['state'] == 'matches_round_robin_fifo'
    assert report['peak_bulk_pending'] == 2 and report['competing_team_selections'] >= 1
    assert report['competing_selections_with_bulk_backlog'] == 0


def test_admission_or_source_stamp_mismatch_invalidates_replay(real_windows):
    plan, _, observed = real_windows
    for change in ('admission', 'admission_before_ready', 'broker_stamp', 'release'):
        report = deepcopy(observed)
        row = report['samples'][0]
        if change == 'admission':
            row['accepted_receipt_ns'] = None
        elif change == 'admission_before_ready':
            row['accepted_receipt_ns'] = row['broker_ready_ns'] - 1
        elif change == 'broker_stamp':
            row['broker_ready_ns'] += 1
        else:
            row['release_due_ns'] += 1
        bounds, clock = report['boundaries'], report['clock']
        summary = study._analyze(plan, report['samples'], report['recording'], bounds['start_ns'],
                                 bounds['end_ns'], True, clock['nominal_resolution_ns'], 'backlog')
        assert summary['event_audit_errors']
        assert summary['queue_observations_valid'] is False
        assert summary['queue']['jobs'] == []
        assert summary['rotation']['state'] == 'inconclusive'
        assert summary['source_timings_by_team_success']['bulk']['ready_to_selection_ns']['n'] == 0


def test_rejected_job_stays_visible_and_rotation_is_inconclusive():
    plan, digest = study.make_plan(model_hold_us=10000)
    original = IPCClient.submit

    def submit(client, *, request_id, **kwargs):
        if request_id == 'job-0005':
            raise IPCRejected('queue_full')
        return original(client, request_id=request_id, **kwargs)

    with patch.object(IPCClient, 'submit', submit):
        report = study.run_window(plan, expected_plan_sha256=digest, mode='backlog')
    summary = report['summary']
    assert summary['states']['rejected'] == 1 and summary['planned'] == 12
    rejected = next(row for row in report['samples'] if row['job_id'] == 'job-0005')
    assert rejected['broker_id'] is rejected['accepted_receipt_ns'] is None
    assert summary['rotation']['state'] == 'inconclusive'
    assert summary['rotation']['reason'] == 'incomplete_or_nonsuccessful_work'
    assert summary['conditions_realized']['bulk_backlog_and_peer_competition'] is None
