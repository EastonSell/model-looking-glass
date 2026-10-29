"""Controlled synthetic CPU load through the fixed-workspace IPC service.

This experiment is not a GPU benchmark, resource-profile validator, or a
paired batching comparison. The only model and resource telemetry it creates
are local synthetic CPU fixtures. The caller must retain the raw report and
supervise the process; timestamps and recorded identities are observations,
not independently authenticated facts.
"""

from collections import Counter
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import platform
import re
import threading
import time

import torch
from torch import nn

from .batched_events import _EVENT_FIELDS
from .batched_recorder import BatchedEventRecorder
from .broker import Broker
from .budgeted_worker import WorkloadProfile
from .ipc import IPCClient, IPCRejected, IPCServer
from .memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
from .resident_worker import ForwardRequest, ResidentWorker
from .service_serial_observation import ServiceSerialObservation
from .throughput_benchmark import _distribution, canonical_digest


_VERSION = 'synthetic-cpu-shared-load-v1'
_TEAM = re.compile(r'[A-Za-z0-9_.-]{1,48}\Z')
_MAX_JOBS = 64
_SAMPLE_STATES = ('unattempted', 'succeeded', 'failed', 'cancelled', 'rejected', 'unresolved', 'invalid_output')
_owner_lock = threading.Lock()
_active_owner = None


class ExperimentUnresolved(RuntimeError):
    """The isolated CPU owner or its event producers could not be drained."""


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def make_plan(teams, *, jobs_per_team=3, seed=17, model_hold_us=0,
              start_spacing_us=0, max_pending=None):
    """Build a finite immutable-by-digest synthetic workload before running.

    Each team has one closed-loop client. The optional start spacing and
    sleep inside the synthetic model are experimental load controls, not
    measured model-compute costs. Returned JSON is mutable: retain the digest
    independently and do not mutate it while the runner snapshots the plan.
    """
    if (type(teams) is not tuple or not 2 <= len(teams) <= 8
            or any(type(team) is not str or _TEAM.fullmatch(team) is None for team in teams)
            or len(set(teams)) != len(teams)):
        raise ValueError('teams must be 2 to 8 distinct bounded identifiers')
    if type(jobs_per_team) is not int or not 1 <= jobs_per_team <= 32:
        raise ValueError('jobs_per_team must be an integer from 1 to 32')
    if len(teams) * jobs_per_team > _MAX_JOBS:
        raise ValueError('the synthetic workload exceeds 64 jobs')
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError('seed must be a bounded nonnegative integer')
    if type(model_hold_us) is not int or not 0 <= model_hold_us <= 100_000:
        raise ValueError('model_hold_us must be an integer from 0 to 100000')
    if type(start_spacing_us) is not int or not 0 <= start_spacing_us <= 100_000:
        raise ValueError('start_spacing_us must be an integer from 0 to 100000')
    if max_pending is None:
        max_pending = len(teams)
    if type(max_pending) is not int or not 1 <= max_pending <= len(teams):
        raise ValueError('max_pending must be between one and the team count')
    jobs = []
    for team_index, team in enumerate(teams):
        for ordinal in range(jobs_per_team):
            number = 1 + team_index * jobs_per_team + ordinal
            job_id = f'job-{number:04d}'
            values = [number, number + 1000]
            inputs = {'input_ids': torch.tensor([values], dtype=torch.int64)}
            request = ForwardRequest(job_id, inputs, seed=seed)
            expected_output = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 1)
            jobs.append({'job_id': job_id, 'team_id': team, 'input_ids': values,
                         'input_sha256': canonical_digest(inputs),
                         'request_sha256': canonical_digest(request),
                         'expected_output_sha256': canonical_digest(expected_output)})
    model_spec = {'kind': 'synthetic-input-to-float32', 'model_hold_us': model_hold_us,
                  'input_shape': [1, 2], 'output_shape': [1, 2, 1], 'dtype': 'float32'}
    model_id = sha256(_canonical_bytes(model_spec)).hexdigest()
    profile_id = sha256(_canonical_bytes({'kind': 'synthetic-cpu-label', 'model': model_id,
                                           'shape': [1, 2], 'output_cap_bytes': 64})).hexdigest()
    plan = {'version': _VERSION, 'teams': list(teams), 'jobs_per_team': jobs_per_team,
            'seed': seed, 'model_hold_us': model_hold_us, 'start_spacing_us': start_spacing_us,
            'max_pending': max_pending, 'max_pending_per_team': 1,
            'model_spec': model_spec, 'model_sha256': model_id,
            'profile_sha256': profile_id, 'jobs': jobs}
    return plan, sha256(_canonical_bytes(plan)).hexdigest()


def _snapshot_plan(plan, expected_sha256):
    if type(plan) is not dict or type(expected_sha256) is not str:
        raise ValueError('an exact frozen plan and its digest are required')
    try:
        payload = _canonical_bytes(plan)
        if len(payload) > 128 * 1024 or sha256(payload).hexdigest() != expected_sha256:
            raise ValueError('plan differs from its independently retained digest')
        snapshot = json.loads(payload)
        expected, digest = make_plan(tuple(snapshot['teams']), jobs_per_team=snapshot['jobs_per_team'],
                                     seed=snapshot['seed'], model_hold_us=snapshot['model_hold_us'],
                                     start_spacing_us=snapshot['start_spacing_us'],
                                     max_pending=snapshot['max_pending'])
    except (KeyError, TypeError, OverflowError, UnicodeError) as exc:
        raise ValueError('invalid synthetic plan') from exc
    if snapshot != expected or digest != expected_sha256:
        raise ValueError('plan differs from the supported exact workload')
    return snapshot


class _SyntheticModel(nn.Module):
    def __init__(self, hold_us):
        super().__init__()
        self._hold_us = hold_us

    def forward(self, input_ids):
        if self._hold_us:
            time.sleep(self._hold_us / 1_000_000)
        return {'logits': input_ids.to(dtype=torch.float32).unsqueeze(-1)}


class _Owner:
    """Strongly root the CPU fixture through uncertain shutdown."""

    def __init__(self):
        self.model = self.resident = self.budget = self.worker = None
        self.load_lease = self.workspace_lease = self.authority = None
        self.observation = self.recorder = self.broker = self.server = None
        self.clients = []
        self.client_errors = []


def _claim_owner():
    global _active_owner
    owner = _Owner()
    with _owner_lock:
        if _active_owner is not None:
            raise ExperimentUnresolved('an earlier CPU experiment still owns uncertain resources')
        _active_owner = owner
    return owner


def _release_owner(owner):
    global _active_owner
    with _owner_lock:
        if _active_owner is not owner:
            raise ExperimentUnresolved('CPU experiment owner identity changed')
        _active_owner = None


def _remaining(deadline_ns):
    return max(0.0, (deadline_ns - time.perf_counter_ns()) / 1_000_000_000)


def _build_owner(owner, plan):
    owner.model = _SyntheticModel(plan['model_hold_us']).eval()
    owner.resident = ResidentWorker(plan['model_sha256'], owner.model, device='cpu',
                                    max_batch_size=1, max_sequence_length=2,
                                    max_input_bytes=64, max_output_bytes=64)
    # Synthetic fixed values exercise CPU accounting logic only. They are not
    # live device telemetry, resource certification, or physical memory caps.
    owner.budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                                max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    owner.load_lease = owner.budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    owner.workspace_lease = owner.budget.acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    owner.authority = OwnerCompletionAuthority(device_guard=lambda: None)
    owner.recorder = BatchedEventRecorder()
    owner.observation = ServiceSerialObservation(owner.recorder)
    profile = WorkloadProfile(plan['profile_sha256'], plan['model_sha256'], 'cpu', (1, 2),
                              (('input_ids', 'torch.int64'),), 64, 1024)
    owner.worker = LifetimeWorkspaceWorker(owner.resident, owner.budget, owner.load_lease,
                                           owner.workspace_lease, profiles=(profile,),
                                           authority=owner.authority, cpu_observation=owner.observation)
    # Warm the exact fixed-worker path before broker binding and timing. The
    # owner holds this result until the worker relinquishes its private alias.
    warmup = ForwardRequest('cpu-warmup', {'input_ids': torch.tensor([[0, 1]], dtype=torch.int64)},
                            seed=plan['seed'])
    warmup_result = owner.worker.execute(warmup)
    if not torch.equal(warmup_result.logits, torch.tensor([[[0.0], [1.0]]])):
        raise RuntimeError('synthetic warmup produced an invalid result')
    owner.worker._transfer_result(warmup_result)
    del warmup_result, warmup
    owner.broker = Broker(owner.worker, max_pending=plan['max_pending'],
                          max_pending_per_team=plan['max_pending_per_team'],
                          completion_authority=owner.authority)
    owner.server = IPCServer(owner.broker, team_ids=tuple(plan['teams']),
                             max_connections=plan.get('max_connections', len(plan['teams'])),
                             max_request_bytes=4096, max_response_bytes=4096, max_batch=1, max_tokens=2,
                             io_timeout=3.0, job_timeout=10.0, completion_authority=owner.authority)
    owner.observation.bind_service(owner.worker, owner.broker, owner.server)
    owner.server.start()


def _record(observation, event):
    # Ordinary recorder failures are sticky and do not alter the actual job.
    # Control-flow exceptions still propagate to the caller's failure path.
    observation.record(event)


def _output_digest(response, job):
    result = response.get('result')
    if (type(result) is not dict or set(result) != {'shape', 'dtype', 'logits'}
            or result['shape'] != [1, 2, 1] or result['dtype'] != 'torch.float32'
            or type(result['logits']) is not list or len(result['logits']) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in result['logits'])):
        raise ValueError('unexpected synthetic output schema')
    output = torch.tensor(result['logits'], dtype=torch.float32).reshape(1, 2, 1)
    expected = torch.tensor(job['input_ids'], dtype=torch.float32).reshape(1, 2, 1)
    if not torch.equal(output, expected):
        raise ValueError('synthetic output does not match its input')
    return canonical_digest(output)


def _sample(job):
    return {'job_id': job['job_id'], 'team_id': job['team_id'], 'state': 'unattempted',
            'attempt_ns': None, 'client_receipt_ns': None, 'handled_ns': None,
            'broker_id': None, 'broker_job_id': None, 'error': None, 'output_sha256': None,
            'client_cancellation_requested': False, 'broker_cancellation_decision': 'unobserved'}


def _client_run(owner, plan, team, team_index, start, ready, jobs, samples):
    row = None
    try:
        client = IPCClient(owner.server.address, owner.server.team_tokens[team], io_timeout=3.0,
                           max_response_bytes=4096)
        ready.set()
        start.wait()
        if plan['start_spacing_us']:
            time.sleep(team_index * plan['start_spacing_us'] / 1_000_000)
        for job, row in zip(jobs, samples):
            attempt_ns = time.perf_counter_ns()
            row['attempt_ns'] = attempt_ns
            _record(owner.observation, {'kind': 'attempt', 'time_ns': attempt_ns,
                    'job_id': job['job_id'], 'team_id': team, 'input_sha256': job['input_sha256'],
                    'request_sha256': job['request_sha256']})
            handle = None
            try:
                # This private list is constructed directly from the frozen
                # plan and is not shared or mutated while IPC serializes it.
                handle = client.submit(team_id=team, request_id=job['job_id'],
                    inputs={'input_ids': [list(job['input_ids'])]}, seed=plan['seed'])
                row['broker_id'], row['broker_job_id'] = handle.broker_id, handle.job_id
                response = handle.wait(timeout=12.0)
                row['client_receipt_ns'] = time.perf_counter_ns()
                status = response['status']
                if status == 'succeeded':
                    try:
                        row['output_sha256'] = _output_digest(response, job)
                        row['state'] = 'succeeded'
                    except (TypeError, ValueError, RuntimeError):
                        row['state'], row['error'] = 'invalid_output', 'invalid_synthetic_output'
                elif status == 'failed':
                    row['state'], row['error'] = 'failed', str(response.get('error', 'failed'))[:128]
                elif status == 'cancelled':
                    row['state'], row['error'] = 'cancelled', 'unobserved_cancellation_decision'
                    owner.observation._fault()
                else:
                    # A server timeout may issue cancellation. Its decision is
                    # not timestamped at the broker, so the client is unresolved.
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
                # A lost acceptance/response can follow a committed job.
                row['state'], row['error'] = 'unresolved', type(exc).__name__[:128]
                owner.observation._fault()
            finally:
                if handle is not None:
                    handle.close()
            row['handled_ns'] = time.perf_counter_ns()
            if row['state'] not in ('unresolved', 'cancelled'):
                _record(owner.observation, {'kind': 'returned', 'time_ns': row['handled_ns'],
                    'job_id': job['job_id'], 'state': row['state'], 'error': row['error'],
                    'output_sha256': row['output_sha256'], 'cancellation_requested': False,
                    'cancellation_requested_ns': None, 'cancellation_accepted': None,
                    'cancellation_decision_ns': None})
            if row['state'] == 'unresolved':
                break
    except BaseException as exc:
        owner.client_errors[team_index] = exc
        owner.observation._fault()
        if row is not None and row['attempt_ns'] is not None and row['state'] == 'unattempted':
            row['state'], row['error'] = 'unresolved', 'client_producer_interrupted'
            row['handled_ns'] = time.perf_counter_ns()
        ready.set()


def _audit_events(plan, samples, recording, start_ns, end_ns, drained, *, closed_loop=True):
    """Validate the recorded CPU observations without manufacturing gaps."""
    errors = []
    recorder_fields = {'events_complete', 'recorded', 'lost_capacity', 'lost_invalid', 'lost_internal',
                       'late_records', 'explicit_faults', 'wire_bytes_upper_bound', 'max_wire_bytes',
                       'max_events', 'events'}
    if type(recording) is not dict or set(recording) != recorder_fields:
        return ['invalid_recorder_inventory'], {}
    if recording['events_complete'] is not True:
        errors.append('recorder_incomplete')
    if not drained:
        errors.append('owner_not_drained')
    events = recording.get('events')
    if type(events) is not list or len(events) > 4096:
        return ['invalid_event_inventory'], {}
    if (type(recording['recorded']) is not int or recording['recorded'] != len(events)
            or type(recording['wire_bytes_upper_bound']) is not int
            or type(recording['max_wire_bytes']) is not int
            or not 13 <= recording['wire_bytes_upper_bound'] <= recording['max_wire_bytes'] <= 1024 * 1024
            or type(recording['max_events']) is not int
            or not max(1, len(events)) <= recording['max_events'] <= 4096):
        errors.append('recorder_inventory_mismatch')
    for field in ('lost_capacity', 'lost_invalid', 'lost_internal', 'late_records', 'explicit_faults'):
        if type(recording[field]) is not int or recording[field] != 0:
            errors.append('recorder_reported_loss')
    planned = {job['job_id']: job for job in plan['jobs']}
    rows = {row['job_id']: row for row in samples}
    by_kind = {kind: {} for kind in _EVENT_FIELDS}
    for event in events:
        if type(event) is not dict or type(event.get('kind')) is not str:
            errors.append('invalid_event')
            continue
        kind = event['kind']
        if kind not in _EVENT_FIELDS or set(event) != set(_EVENT_FIELDS[kind]):
            errors.append('invalid_event_schema')
            continue
        stamp = event['time_ns']
        if type(stamp) is not int or not start_ns <= stamp <= end_ns:
            errors.append('event_outside_window')
            continue
        identity = ('window' if kind in ('window_start', 'window_end', 'owner_drain') else
                    event.get('call_id') if kind.startswith('forward_') else event.get('job_id'))
        if type(identity) is not str or identity in by_kind[kind]:
            errors.append('invalid_or_duplicate_event_identity')
            continue
        by_kind[kind][identity] = event
    drain = by_kind['owner_drain'].get('window', {})
    if (by_kind['window_start'].get('window', {}).get('time_ns') != start_ns
            or by_kind['window_end'].get('window', {}).get('time_ns') != end_ns
            or drain.get('drained') is not drained):
        errors.append('missing_window_or_drain_event')
    elif any(event['time_ns'] > drain['time_ns'] for event in events
             if type(event) is dict and type(event.get('time_ns')) is int
             and event.get('kind') not in ('window_end', 'owner_drain')):
        errors.append('work_observed_after_drain')
    if set(rows) != set(planned) or len(rows) != len(samples):
        errors.append('sample_inventory_mismatch')
    known_jobs = set(planned)
    for kind in ('attempt', 'ready', 'selected', 'broker_terminal', 'returned'):
        if not set(by_kind[kind]).issubset(known_jobs):
            errors.append('unknown_job_event')
    entries, finishes, uncertain = (by_kind['forward_entry'], by_kind['forward_finished'],
                                    by_kind['forward_uncertain'])
    if set(entries) != set(finishes) | set(uncertain) or set(finishes) & set(uncertain):
        errors.append('incomplete_forward_observations')
    job_calls = {}
    intervals = []
    for call_id, entry in entries.items():
        members = entry['members']
        if (type(members) is not list or len(members) != 1 or type(members[0]) is not dict
                or set(members[0]) != {'job_id', 'team_id', 'broker_id', 'broker_job_id', 'row_offset'}
                or type(members[0].get('row_offset')) is not int or members[0]['row_offset'] != 0
                or entry['input_shape'] != [1, 2]
                or entry['profile_sha256'] != plan['profile_sha256']):
            errors.append('invalid_forward_identity')
            continue
        member = members[0]
        job_id = member['job_id']
        if job_id not in planned or job_id in job_calls:
            errors.append('unknown_or_duplicate_forward_member')
            continue
        job_calls[job_id] = (entry, finishes.get(call_id), uncertain.get(call_id))
        if (member['team_id'] != planned[job_id]['team_id']
                or member['broker_id'] != rows.get(job_id, {}).get('broker_id')
                or member['broker_job_id'] != rows.get(job_id, {}).get('broker_job_id')
                or entry['group_id'] != call_id
                or call_id != f"lg-{member['broker_id']}-{member['broker_job_id']}"):
            errors.append('forward_member_mismatch')
        finish = finishes.get(call_id)
        if finish is not None:
            if finish['time_ns'] < entry['time_ns']:
                errors.append('forward_finish_before_entry')
            if (finish['state'] not in ('succeeded', 'failed') or finish['synchronized'] is not True
                    or (finish['state'] == 'succeeded' and finish['error'] is not None)
                    or (finish['state'] == 'failed' and
                        (type(finish['error']) is not str or not finish['error'].strip()
                         or len(finish['error']) > 128))):
                errors.append('invalid_cpu_forward_finish')
            intervals.append((entry['time_ns'], finish['time_ns']))
        if uncertain.get(call_id) is not None and uncertain[call_id]['time_ns'] < entry['time_ns']:
            errors.append('forward_uncertainty_before_entry')
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        errors.append('overlapping_owner_forwards')
    ordered_calls = sorted(entries.items(), key=lambda item: (item[1]['time_ns'], item[0]))
    for call_id, _ in ordered_calls[:-1]:
        finish = finishes.get(call_id)
        if finish is None or finish['state'] != 'succeeded' or finish['synchronized'] is not True:
            errors.append('forward_after_failed_or_uncertain_call')
    accepted = []
    for row in samples:
        broker_id, job_id = row['broker_id'], row['broker_job_id']
        if broker_id is None and job_id is None:
            if row['state'] in ('succeeded', 'failed', 'invalid_output', 'cancelled'):
                errors.append('missing_broker_identity')
        elif (type(broker_id) is not str or re.fullmatch('[0-9a-f]{32}', broker_id) is None
              or type(job_id) is not int or job_id < 1):
            errors.append('invalid_broker_identity')
        else:
            accepted.append((broker_id, job_id))
    if len(set(accepted)) != len(accepted) or len({key[0] for key in accepted}) > 1:
        errors.append('invalid_broker_identity')
    if closed_loop:
        for team in plan['teams']:
            team_rows = [rows[job['job_id']] for job in plan['jobs']
                         if job['team_id'] == team and job['job_id'] in rows]
            for previous, current in zip(team_rows, team_rows[1:]):
                if (current['attempt_ns'] is not None
                        and (type(previous['handled_ns']) is not int
                             or type(current['attempt_ns']) is not int
                             or current['attempt_ns'] < previous['handled_ns'])):
                    errors.append('team_closed_loop_order_invalid')
    for job_id, job in planned.items():
        row = rows.get(job_id)
        if row is None or row.get('team_id') != job['team_id']:
            errors.append('sample_team_mismatch')
            continue
        if row['state'] not in _SAMPLE_STATES:
            errors.append('unknown_sample_state')
        if (row['state'] == 'succeeded' and row['error'] is not None
                or row['state'] in ('failed', 'rejected', 'unresolved', 'invalid_output')
                and (type(row['error']) is not str or not row['error'].strip() or len(row['error']) > 128)):
            errors.append('invalid_sample_error')
        if (row['client_cancellation_requested'] is not False
                or row['broker_cancellation_decision'] != 'unobserved'):
            errors.append('unsupported_cancellation_observation')
        if (row['state'] == 'succeeded' and row['output_sha256'] != job['expected_output_sha256']
                or row['state'] != 'succeeded' and row['output_sha256'] is not None):
            errors.append('output_digest_mismatch')
        attempt = by_kind['attempt'].get(job_id)
        returned = by_kind['returned'].get(job_id)
        ready = by_kind['ready'].get(job_id)
        selected = by_kind['selected'].get(job_id)
        terminal = by_kind['broker_terminal'].get(job_id)
        if row['state'] == 'unattempted':
            if (any(value is not None for value in (attempt, returned, ready, selected, terminal))
                    or job_id in job_calls):
                errors.append('unattempted_job_has_event')
            if (any(row[key] is not None for key in ('attempt_ns', 'client_receipt_ns', 'handled_ns',
                                                       'broker_id', 'broker_job_id', 'error', 'output_sha256'))
                    or row['client_cancellation_requested'] is not False
                    or row['broker_cancellation_decision'] != 'unobserved'):
                errors.append('unattempted_sample_has_observations')
            continue
        if (attempt is None or attempt['time_ns'] != row['attempt_ns']
                or attempt['team_id'] != job['team_id']
                or attempt['input_sha256'] != job['input_sha256']
                or attempt['request_sha256'] != job['request_sha256']):
            errors.append('attempt_mismatch')
        if (type(row['handled_ns']) is not int or type(row['attempt_ns']) is not int
                or not row['attempt_ns'] <= row['handled_ns'] <= end_ns):
            errors.append('client_handling_time_invalid')
        if (row['client_receipt_ns'] is not None and
                (type(row['client_receipt_ns']) is not int or type(row['attempt_ns']) is not int
                 or type(row['handled_ns']) is not int
                 or not row['attempt_ns'] <= row['client_receipt_ns'] <= row['handled_ns'])):
            errors.append('client_receipt_time_invalid')
        if row['state'] in ('unresolved', 'cancelled'):
            errors.append('client_or_cancellation_unresolved')
            continue
        if type(row['client_receipt_ns']) is not int:
            errors.append('client_receipt_missing')
        if (returned is None or returned['time_ns'] != row['handled_ns']
                or returned['state'] != row['state'] or returned['error'] != row['error']
                or returned['output_sha256'] != row['output_sha256']
                or returned['cancellation_requested'] is not False
                or returned['cancellation_requested_ns'] is not None
                or returned['cancellation_accepted'] is not None
                or returned['cancellation_decision_ns'] is not None):
            errors.append('client_return_mismatch')
        if row['state'] == 'rejected':
            if (any(value is not None for value in (ready, selected, terminal, row['broker_id']))
                    or job_id in job_calls):
                errors.append('rejected_job_has_broker_events')
            continue
        if (ready is None or terminal is None or ready['team_id'] != job['team_id']
                or ready['broker_id'] != row['broker_id']
                or ready['broker_job_id'] != row['broker_job_id']
                or type(row['attempt_ns']) is not int
                or ready['time_ns'] < row['attempt_ns']
                or type(row['client_receipt_ns']) is not int
                or terminal['time_ns'] > row['client_receipt_ns']
                or terminal['state'] != ('succeeded' if row['state'] == 'invalid_output' else row['state'])):
            errors.append('broker_outcome_mismatch')
            continue
        if selected is not None and not ready['time_ns'] <= selected['time_ns'] <= terminal['time_ns']:
            errors.append('queue_time_order_invalid')
        call = job_calls.get(job_id)
        if call is not None:
            conclusion = call[1] if call[1] is not None else call[2]
            if (selected is None or conclusion is None
                    or not selected['time_ns'] <= call[0]['time_ns'] <= conclusion['time_ns'] <= terminal['time_ns']):
                errors.append('forward_and_terminal_order_invalid')
            if call[2] is not None:
                errors.append('uncertain_forward_with_terminal')
            if call[1] is not None and call[1]['state'] == 'failed' and terminal['state'] != 'failed':
                errors.append('failed_forward_terminal_mismatch')
        if row['state'] in ('succeeded', 'invalid_output'):
            if (selected is None or call is None or call[1] is None or call[2] is not None
                    or call[1]['state'] != 'succeeded' or call[1]['synchronized'] is not True
                    or not selected['time_ns'] <= call[0]['time_ns'] <= call[1]['time_ns'] <= terminal['time_ns']):
                errors.append('successful_job_lacks_actual_forward')
    return sorted(set(errors)), by_kind


def _summarize(plan, samples, recording, start_ns, end_ns, drained, resolution_ns, *, closed_loop=True):
    errors, by_kind = _audit_events(plan, samples, recording, start_ns, end_ns, drained,
                                    closed_loop=closed_loop)
    elapsed = end_ns - start_ns
    seconds = elapsed / 1_000_000_000
    expected_teams = {job['job_id']: job['team_id'] for job in plan['jobs']}
    matching_counts = Counter(row['job_id'] for row in samples
                              if row['job_id'] in expected_teams
                              and row['team_id'] == expected_teams[row['job_id']])
    matched = [row for row in samples if row['job_id'] in expected_teams
               and row['team_id'] == expected_teams[row['job_id']]
               and matching_counts[row['job_id']] == 1]
    rows_by_team = {team: [row for row in matched if row['team_id'] == team] for team in plan['teams']}
    states = _SAMPLE_STATES

    def describe(rows, expected_jobs):
        counts = Counter(row['state'] for row in rows)
        timing_by_state = {}
        for state in states:
            in_state = [row for row in rows if row['state'] == state]
            responses = [row['client_receipt_ns'] - row['attempt_ns'] for row in in_state
                         if type(row['client_receipt_ns']) is int and type(row['attempt_ns']) is int]
            handling = [row['handled_ns'] - row['attempt_ns'] for row in in_state
                        if type(row['handled_ns']) is int and type(row['attempt_ns']) is int]
            timing_by_state[state] = {'client_receipt_ns': _distribution(responses),
                                      'client_handled_ns': _distribution(handling)}
        return {'planned': len(expected_jobs),
                'missing_samples': sum(matching_counts[job['job_id']] == 0 for job in expected_jobs),
                'ambiguous_samples': sum(matching_counts[job['job_id']] > 1 for job in expected_jobs),
                'invalid_state_samples': sum(row['state'] not in states for row in rows),
                'attempted': sum(row['attempt_ns'] is not None for row in rows),
                'accepted_observed': sum(row['broker_id'] is not None for row in rows),
                'acceptance_unknown': sum(row['state'] == 'unresolved' and row['broker_id'] is None for row in rows),
                'states': {state: counts[state] for state in states},
                'successful_jobs_per_second': counts['succeeded'] / seconds,
                'successful_input_positions_per_second': 2 * counts['succeeded'] / seconds,
                'timing_by_state': timing_by_state}

    summary = describe(matched, plan['jobs'])
    summary.update({'elapsed_ns': elapsed, 'owner_drained': drained,
                    'extra_or_ambiguous_sample_rows': len(samples) - len(matched),
                    'teams': {team: describe(rows, [job for job in plan['jobs'] if job['team_id'] == team])
                              for team, rows in rows_by_team.items()},
                    'event_audit_errors': errors, 'queue_observations_valid': not errors})
    selected = by_kind.get('selected', {})
    ready = by_kind.get('ready', {})
    queue_rows = []
    if not errors:
        for job_id, selection in selected.items():
            if job_id not in ready:
                continue
            r, s = ready[job_id]['time_ns'], selection['time_ns']
            ties = sum(1 for other_id, other in selected.items()
                       if other_id != job_id and other['time_ns'] in (r, s))
            overtakes = sum(1 for other_id, other in selected.items()
                            if other_id != job_id and r < other['time_ns'] < s)
            queue_rows.append({'job_id': job_id, 'team_id': ready[job_id]['team_id'],
                               'ready_ns': r, 'selected_ns': s, 'wait_ns': s - r,
                               'strictly_intervening_selections': overtakes,
                               'boundary_ties': ties})
    summary['queue'] = {'jobs': queue_rows,
        'wait_ns': _distribution([row['wait_ns'] for row in queue_rows]),
        'by_team': {team: {'wait_ns': _distribution([row['wait_ns'] for row in queue_rows
                                                     if row['team_id'] == team]),
                           'strictly_intervening_selections': _distribution([
                               row['strictly_intervening_selections'] for row in queue_rows
                               if row['team_id'] == team])} for team in plan['teams']}}
    threshold = 100 * resolution_ns
    response_intervals = [row['client_receipt_ns'] - row['attempt_ns'] for row in matched
                          if row['client_receipt_ns'] is not None and row['attempt_ns'] is not None]
    summary['nominal_resolution_screen'] = {'threshold_ns': threshold, 'window_short': elapsed < threshold,
        'client_receipt_intervals_observed': len(response_intervals),
        'client_receipt_intervals_short': sum(value < threshold for value in response_intervals),
        'queue_wait_intervals_observed': len(queue_rows),
        'queue_wait_intervals_short': sum(row['wait_ns'] < threshold for row in queue_rows)}
    return summary


def run_cpu_window(plan, *, expected_plan_sha256, timeout=30.0):
    """Run one bounded synthetic CPU IPC window and return its raw evidence.

    The owner has no production lease or measured resource profile. Run this
    trusted local experiment under an external process watchdog. If a thread,
    callback, or cleanup remains uncertain, the run raises and the in-process
    owner stays rooted; replace the process before retrying. Timeout checks
    raise ExperimentUnresolved; worker or shutdown errors may use other types.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError('timeout must be finite and between zero and 120 seconds')
    plan = _snapshot_plan(plan, expected_plan_sha256)
    info = time.get_clock_info('perf_counter')
    if info.monotonic is not True or info.adjustable is not False:
        raise ValueError('the measurement clock must be monotonic and non-adjustable')
    if (type(info.resolution) not in (int, float) or not math.isfinite(info.resolution)
            or info.resolution <= 0):
        raise ValueError('the measurement clock must have a finite positive nominal resolution')
    resolution_ns = max(1, math.ceil(info.resolution * 1_000_000_000))
    deadline = time.perf_counter_ns() + int(timeout * 1_000_000_000)
    owner = _claim_owner()
    clean = False
    try:
        _build_owner(owner, plan)
        samples = [_sample(job) for job in plan['jobs']]
        owner.client_errors = [None] * len(plan['teams'])
        start = threading.Event()
        ready = [threading.Event() for _ in plan['teams']]
        for index, team in enumerate(plan['teams']):
            jobs = [job for job in plan['jobs'] if job['team_id'] == team]
            rows = [row for row in samples if row['team_id'] == team]
            thread = threading.Thread(target=_client_run,
                args=(owner, plan, team, index, start, ready[index], jobs, rows),
                name=f'cpu-load-{index}', daemon=True)
            owner.clients.append(thread)
            thread.start()
        for event in ready:
            if not event.wait(_remaining(deadline)):
                raise ExperimentUnresolved('client readiness did not complete')
        if any(error is not None for error in owner.client_errors):
            raise ExperimentUnresolved('a client failed before the common release')
        # All clients are blocked on the same event. This stamp precedes its
        # set; event recording and release overhead are inside the denominator.
        start_ns = time.perf_counter_ns()
        _record(owner.observation, {'kind': 'window_start', 'time_ns': start_ns})
        start.set()
        for thread in owner.clients:
            thread.join(_remaining(deadline))
        if any(thread.is_alive() for thread in owner.clients):
            raise ExperimentUnresolved('client threads did not finish within the watchdog')
        for error in owner.client_errors:
            if error is not None and not isinstance(error, Exception):
                if isinstance(error, KeyboardInterrupt):
                    raise KeyboardInterrupt('client producer interrupted') from None
                if isinstance(error, SystemExit):
                    raise SystemExit('client producer interrupted') from None
                raise BaseException('client producer interrupted') from None
        clients_joined_ns = time.perf_counter_ns()
        if not owner.server.stop_accepting(timeout=_remaining(deadline)):
            raise ExperimentUnresolved('IPC admission did not stop')
        if not owner.server.drain(timeout=_remaining(deadline)):
            raise ExperimentUnresolved('IPC handlers did not drain')
        if not owner.server.close(timeout=_remaining(deadline)):
            raise ExperimentUnresolved('IPC server did not close')
        if not owner.broker.close(timeout=_remaining(deadline)):
            raise ExperimentUnresolved('broker owner did not stop')
        with owner.broker._condition:
            drained = (not owner.broker._thread.is_alive() and owner.broker._running is None
                       and owner.broker._pending_slots == 0 and owner.broker._preparing == 0
                       and len(owner.broker._queue) == 0
                       and all(job is None for job in owner.broker._job_roots))
        custody_names = ('request', 'logits', 'result', 'error', 'producer_error',
                         'forward_error', 'state_error')
        drained = (drained and not owner.authority.unsafe and not owner.broker.quarantined
                   and not owner.worker.quarantined and not owner.server.custody_quarantined
                   and not owner.resident._drain_pending and owner.resident._snapshot_error is None
                   and all(value is None for value in owner.resident._live_buffers)
                   and all(value is None for value in owner.resident._retiring_buffers)
                   and all(getattr(owner.worker._custody, name) is None for name in custody_names)
                   and owner.worker._warmup_result is None and owner.worker._retiring_cancel is None
                   and owner.worker._retiring_transfer is None
                   and owner.worker._cancel_cleanup_error is None
                   and owner.worker._transfer_cleanup_error is None)
        drain_ns = time.perf_counter_ns()
        _record(owner.observation, {'kind': 'owner_drain', 'time_ns': drain_ns, 'drained': bool(drained)})
        end_ns = time.perf_counter_ns()
        if end_ns <= start_ns:
            owner.observation._fault()
            raise ExperimentUnresolved('the measurement clock did not advance over the window')
        _record(owner.observation, {'kind': 'window_end', 'time_ns': end_ns})
        recording = owner.observation.seal(producers_stopped=True)
        summary = _summarize(plan, samples, recording, start_ns, end_ns, bool(drained), resolution_ns)
        report = {'version': _VERSION, 'plan': plan, 'plan_sha256': expected_plan_sha256,
                  'clock': {'name': 'perf_counter_ns', 'implementation': info.implementation,
                            'monotonic': info.monotonic, 'adjustable': info.adjustable,
                            'nominal_resolution_ns': resolution_ns},
                  'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                              'torch_num_threads': torch.get_num_threads(),
                              'device': 'cpu', 'external_activity': 'unmeasured'},
                  'boundaries': {'start_ns': start_ns, 'clients_joined_ns': clients_joined_ns,
                                 'owner_drain_ns': drain_ns, 'end_ns': end_ns},
                  'owner': {'work_drained': bool(drained), 'model_and_lease_disposal_attested': False,
                            'synthetic_device_telemetry': True,
                            'supervised_device_owner': False, 'resource_profile_verified': False},
                  'client_errors': [type(error).__name__ if error is not None else None
                                    for error in owner.client_errors],
                  'samples': samples, 'recording': recording, 'summary': summary}
        if not drained:
            raise ExperimentUnresolved('CPU owner work or health remains uncertain')
        clean = True
        return report
    finally:
        if not clean:
            # This does not assert successful shutdown. Keep the entire graph
            # rooted for process replacement if anything remains uncertain.
            if owner.server is not None:
                try:
                    owner.server.close(timeout=min(0.5, _remaining(deadline)))
                except Exception:
                    pass
            if owner.broker is not None:
                try:
                    owner.broker.close(timeout=min(0.5, _remaining(deadline)))
                except Exception:
                    pass
        if clean:
            _release_owner(owner)


def main(argv=None):
    """Freeze a plan or run it in an externally supervised CPU-only process."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('plan', help='freeze a bounded synthetic workload')
    create.add_argument('--teams', nargs='+', required=True)
    create.add_argument('--jobs-per-team', type=int, default=3)
    create.add_argument('--seed', type=int, default=17)
    create.add_argument('--model-hold-us', type=int, default=0)
    create.add_argument('--start-spacing-us', type=int, default=0)
    create.add_argument('--max-pending', type=int)
    create.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='run a previously frozen plan')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--sha256', required=True, help='independently retained plan digest')
    run.add_argument('--timeout', type=float, default=30.0)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        plan, digest = make_plan(tuple(args.teams), jobs_per_team=args.jobs_per_team,
            seed=args.seed, model_hold_us=args.model_hold_us,
            start_spacing_us=args.start_spacing_us, max_pending=args.max_pending)
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
    # Reserve a fresh result path before doing any work. A failed run leaves an
    # empty or partial file, never a complete-looking report or overwritten run.
    with args.output.open('x', encoding='utf-8') as stream:
        report = run_cpu_window(envelope['plan'], expected_plan_sha256=args.sha256,
                                timeout=args.timeout)
        json.dump(report, stream, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
