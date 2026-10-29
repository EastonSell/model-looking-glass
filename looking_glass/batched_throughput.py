"""Offline checks for one serial/grouped shared-load pair.

This analyzes trusted, already-collected records; it does not run a model,
authenticate event origins, validate a resource profile, or certify batching.
The producer must retain and protect the raw records and actual CPU outputs.
"""
from collections import Counter
from hashlib import sha256
import json
import math
import re

import torch

from .batch_validation import _comparison
from .throughput_benchmark import _distribution, canonical_digest


VERSION = 'batched-shared-load-v1'
_STATES = ('succeeded', 'failed', 'cancelled', 'rejected', 'unresolved',
           'invalid_output', 'submit_error')
_HEX = re.compile(r'^[0-9a-f]{64}$')
_MAX_JOBS = 256
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024


def _json_copy(value):
    if type(value) is not dict:
        raise ValueError('expected an exact dictionary')
    try:
        data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(data) > 1024 * 1024:
            raise ValueError('metadata exceeds one MiB')
        return json.loads(data), data
    except (TypeError, OverflowError) as exc:
        raise ValueError('expected bounded finite JSON metadata') from exc


def _fields(record, expected, label):
    if type(record) is not dict or set(record) != set(expected):
        raise ValueError(f'{label} has unexpected or missing fields')


def _name(value, label):
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ValueError(f'{label} must be a bounded nonempty string')


def _digest(value, label):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(f'{label} must be lowercase SHA-256')


def _int(value, label, *, minimum=0):
    if type(value) is not int or value < minimum or value >= 2 ** 63:
        raise ValueError(f'{label} must be a bounded integer >= {minimum}')


def _broker_job_id(value):
    # Serial brokers currently use integers; grouped brokers issue opaque
    # strings. Preserve the actual type instead of inventing a conversion.
    if type(value) is int:
        _int(value, 'broker job ID', minimum=1)
    elif type(value) is str:
        _name(value, 'broker job ID')
    else:
        raise ValueError('broker job ID must be a bounded string or positive integer')


def _tolerance(value):
    if type(value) not in (float, int):
        raise ValueError('tolerance must be a finite nonnegative number')
    original = value
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError('invalid tolerance') from exc
    if not math.isfinite(value) or value < 0 or value != original:
        raise ValueError('tolerance must be a finite nonnegative number')
    return value


def freeze_protocol(protocol):
    """Validate and hash a JSON protocol before observing comparison results.

    This does not verify a checkpoint, a profile or the claimed inputs; the
    supervisor and experiment driver own those checks and the freeze time.
    """
    protocol, payload = _json_copy(protocol)
    _fields(protocol, ('version', 'model_sha256', 'checkpoint_sha256', 'source_sha256',
                       'runtime_sha256', 'capture_policy',
                       'serial_policy_sha256', 'grouped_policy_sha256',
                       'profile_sha256_by_rows', 'heldout_evidence_sha256',
                       'input_shape', 'output_shape', 'output_dtype',
                       'output_cap_bytes', 'atol', 'rtol', 'jobs'), 'protocol')
    if protocol['version'] != VERSION:
        raise ValueError('unsupported protocol version')
    for field in ('model_sha256', 'checkpoint_sha256', 'source_sha256', 'runtime_sha256',
                  'serial_policy_sha256', 'grouped_policy_sha256', 'heldout_evidence_sha256'):
        _digest(protocol[field], field)
    if protocol['capture_policy'] != 'none':
        raise ValueError('v1 covers forward-only no-capture work')
    profiles = protocol['profile_sha256_by_rows']
    if type(profiles) is not dict or '1' not in profiles or not 2 <= len(profiles) <= 16:
        raise ValueError('serial and grouped row profiles are required')
    for rows, digest in profiles.items():
        if type(rows) is not str or not rows.isascii() or not rows.isdigit():
            raise ValueError('profile row counts must be canonical decimal strings')
        count = int(rows)
        if str(count) != rows or not 1 <= count <= 16:
            raise ValueError('invalid profile row count')
        _digest(digest, 'profile digest')
    if len(set(profiles.values())) != len(profiles):
        raise ValueError('each row count requires a distinct exact profile')
    for field, length in (('input_shape', 2), ('output_shape', 3)):
        shape = protocol[field]
        if type(shape) is not list or len(shape) != length:
            raise ValueError(f'invalid {field}')
        for value in shape:
            _int(value, field, minimum=1)
    if (protocol['input_shape'][0] != 1 or protocol['output_shape'][0] != 1
            or protocol['input_shape'][1] != protocol['output_shape'][1]
            or protocol['output_dtype'] != 'float32'):
        raise ValueError('v1 requires batch-one float32 outputs matching sequence length')
    cap = protocol['output_cap_bytes']
    _int(cap, 'output cap', minimum=1)
    if cap > _MAX_OUTPUT_BYTES or 4 * math.prod(protocol['output_shape']) > cap:
        raise ValueError('output shape exceeds the bounded output cap')
    _tolerance(protocol['atol'])
    _tolerance(protocol['rtol'])
    jobs = protocol['jobs']
    if type(jobs) is not list or not 2 <= len(jobs) <= _MAX_JOBS:
        raise ValueError('protocol needs a bounded multi-client workload')
    if 2 * len(jobs) * cap > 512 * 1024 * 1024:
        raise ValueError('paired retained outputs exceed the bounded host allowance')
    ids, teams, inputs, seeds = set(), set(), set(), set()
    for job in jobs:
        _fields(job, ('job_id', 'team_id', 'input_sha256', 'request_sha256', 'seed'), 'job')
        for field in ('job_id', 'team_id'):
            _name(job[field], field)
        for field in ('input_sha256', 'request_sha256'):
            _digest(job[field], field)
        _int(job['seed'], 'seed')
        if job['job_id'] in ids:
            raise ValueError('duplicate job ID')
        ids.add(job['job_id'])
        teams.add(job['team_id'])
        inputs.add(job['input_sha256'])
        seeds.add(job['seed'])
    if len(teams) < 2 or len(inputs) < 2 or len(seeds) != 1:
        raise ValueError('jobs need multiple teams, distinct inputs and one common seed')
    return protocol, sha256(payload).hexdigest()


def _timestamp(value, label, start, end, *, optional=False):
    if value is None and optional:
        return
    _int(value, label)
    if not start <= value <= end:
        raise ValueError(f'{label} lies outside the window')


def _window(protocol, protocol_sha256, window, expected_mode):
    window, _ = _json_copy(window)
    _fields(window, ('mode', 'window_id', 'owner_session_id', 'protocol_sha256',
                     'model_sha256', 'runtime_sha256', 'policy_sha256', 'start_ns', 'end_ns',
                     'clock', 'clock_resolution_ns', 'drained', 'attempts', 'calls'), 'window')
    if window['mode'] != expected_mode or window['protocol_sha256'] != protocol_sha256:
        raise ValueError('window mode or frozen protocol differs')
    if window['policy_sha256'] != protocol[expected_mode + '_policy_sha256']:
        raise ValueError('window scheduling policy differs from the frozen protocol')
    for field in ('model_sha256', 'runtime_sha256'):
        if window[field] != protocol[field]:
            raise ValueError('window model/runtime differs from protocol')
    for field in ('window_id', 'owner_session_id'):
        _name(window[field], field)
    if window['clock'] != 'perf_counter_ns':
        raise ValueError('windows require the declared high-resolution clock')
    _int(window['clock_resolution_ns'], 'clock resolution', minimum=1)
    start, end = window['start_ns'], window['end_ns']
    _int(start, 'window start')
    _int(end, 'window end')
    if end <= start or type(window['drained']) is not bool:
        raise ValueError('invalid window duration or drain flag')
    jobs = {job['job_id']: job for job in protocol['jobs']}
    attempts = window['attempts']
    if type(attempts) is not list or len(attempts) > len(jobs):
        raise ValueError('invalid attempt inventory')
    by_id, broker_jobs = {}, set()
    for row in attempts:
        _fields(row, ('job_id', 'input_sha256', 'request_sha256', 'state',
                      'attempt_ns', 'ready_ns', 'selected_ns', 'returned_ns',
                      'broker_id', 'broker_job_id', 'call_id', 'output_sha256', 'error',
                      'cancellation_requested', 'cancellation_requested_ns',
                      'cancellation_accepted', 'cancellation_decision_ns',
                      'broker_terminal_state'), 'attempt')
        job_id = row['job_id']
        _name(job_id, 'job ID')
        if job_id not in jobs or job_id in by_id:
            raise ValueError('unknown or duplicate attempted job')
        job = jobs[job_id]
        if row['input_sha256'] != job['input_sha256'] or row['request_sha256'] != job['request_sha256']:
            raise ValueError('attempt differs from frozen input or request')
        if row['state'] not in _STATES:
            raise ValueError('invalid attempt state')
        broker_state = row['broker_terminal_state']
        if broker_state is not None and broker_state not in ('succeeded', 'failed', 'cancelled'):
            raise ValueError('invalid broker terminal state')
        if type(row['cancellation_requested']) is not bool:
            raise ValueError('cancellation intent must be an explicit boolean')
        if row['cancellation_accepted'] is not None and type(row['cancellation_accepted']) is not bool:
            raise ValueError('cancellation acceptance must be boolean or unknown')
        if not row['cancellation_requested'] and (
                row['cancellation_requested_ns'] is not None or row['cancellation_accepted'] is not None
                or row['cancellation_decision_ns'] is not None):
            raise ValueError('cancellation evidence requires a request')
        if row['cancellation_requested'] and row['cancellation_requested_ns'] is None:
            raise ValueError('cancellation request needs its timestamp')
        if (row['cancellation_accepted'] is None) != (row['cancellation_decision_ns'] is None):
            raise ValueError('known cancellation decision needs its timestamp')
        if broker_state == 'cancelled' and row['cancellation_accepted'] is not True:
            raise ValueError('cancellation outcome requires accepted cancellation')
        if (broker_state == 'succeeded' and row['cancellation_requested']
                and row['cancellation_accepted'] is not False):
            raise ValueError('accepted or unknown cancellation cannot deliver success')
        broker_id, broker_job_id = row['broker_id'], row['broker_job_id']
        if (broker_id is None) != (broker_job_id is None):
            raise ValueError('broker ID and job ID must be present together')
        if broker_id is not None:
            _name(broker_id, 'broker ID')
            _broker_job_id(broker_job_id)
            identity = (broker_id, broker_job_id)
            if identity in broker_jobs:
                raise ValueError('duplicate broker-scoped job ID')
            broker_jobs.add(identity)
        elif row['state'] not in ('rejected', 'submit_error', 'unresolved'):
            raise ValueError('accepted work needs a broker-scoped identity')
        if row['cancellation_accepted'] is True and broker_id is None:
            raise ValueError('accepted cancellation needs a broker-scoped identity')
        if broker_state is not None and broker_id is None:
            raise ValueError('broker terminal outcome needs a broker identity')
        if row['state'] in ('succeeded', 'failed', 'cancelled') and broker_state != row['state']:
            raise ValueError('delivered client and broker outcomes differ')
        if row['state'] == 'invalid_output' and broker_state != 'succeeded':
            raise ValueError('invalid delivered output needs a successful broker result')
        if row['state'] in ('rejected', 'submit_error') and broker_state is not None:
            raise ValueError('unaccepted work cannot have a broker outcome')
        for field in ('attempt_ns', 'returned_ns'):
            _timestamp(row[field], field, start, end)
        for field in ('ready_ns', 'selected_ns', 'cancellation_requested_ns', 'cancellation_decision_ns'):
            _timestamp(row[field], field, start, end, optional=True)
        attempt, ready, selected, returned = (row[k] for k in
                                                ('attempt_ns', 'ready_ns', 'selected_ns', 'returned_ns'))
        if attempt > returned or (ready is not None and not attempt <= ready <= returned):
            raise ValueError('attempt/ready/return order is inconsistent')
        if selected is not None and (ready is None or not ready <= selected <= returned):
            raise ValueError('selection order is inconsistent')
        if row['cancellation_requested_ns'] is not None and not attempt <= row['cancellation_requested_ns'] <= returned:
            raise ValueError('cancellation request lies outside the client attempt')
        if (row['cancellation_decision_ns'] is not None
                and not row['cancellation_requested_ns'] <= row['cancellation_decision_ns'] <= returned):
            raise ValueError('cancellation decision is out of order')
        if row['call_id'] is not None:
            _name(row['call_id'], 'call ID')
            if selected is None or broker_id is None:
                raise ValueError('a launched member needs selection and broker identity')
        if row['state'] == 'succeeded':
            if row['error'] is not None or row['call_id'] is None:
                raise ValueError('success needs a call and no error')
            _digest(row['output_sha256'], 'output digest')
        elif row['output_sha256'] is not None:
            raise ValueError('non-success cannot claim a delivered output')
        elif row['state'] in ('failed', 'rejected', 'unresolved', 'invalid_output', 'submit_error'):
            _name(row['error'], 'error')
        if row['state'] in ('rejected', 'submit_error') and any(
                row[k] is not None for k in ('ready_ns', 'selected_ns', 'call_id', 'broker_id', 'broker_job_id')):
            raise ValueError('unaccepted work cannot have queue or call events')
        by_id[job_id] = row
    for team in {job['team_id'] for job in protocol['jobs']}:
        planned = [job['job_id'] for job in protocol['jobs'] if job['team_id'] == team]
        observed = sorted((row for key, row in by_id.items() if jobs[key]['team_id'] == team),
                          key=lambda row: row['attempt_ns'])
        if [row['job_id'] for row in observed] != planned[:len(observed)]:
            raise ValueError('team attempts differ from the frozen order')
        if any(a['returned_ns'] > b['attempt_ns'] for a, b in zip(observed, observed[1:])):
            raise ValueError('closed-loop team attempts overlap')

    calls = window['calls']
    if type(calls) is not list or len(calls) > len(attempts):
        raise ValueError('invalid call inventory')
    calls_by_id, call_members, intervals, group_ids = {}, set(), [], set()
    mixed_group = False
    for call in calls:
        _fields(call, ('call_id', 'group_id', 'profile_sha256', 'input_shape', 'members',
                       'launch_ns', 'finish_ns', 'state', 'synchronized', 'error'), 'call')
        _name(call['call_id'], 'call ID')
        _name(call['group_id'], 'group ID')
        if call['call_id'] in calls_by_id:
            raise ValueError('duplicate call ID')
        if call['group_id'] in group_ids:
            raise ValueError('duplicate group ID')
        group_ids.add(call['group_id'])
        members = call['members']
        if type(members) is not list or not members or len(members) > 16:
            raise ValueError('invalid call members')
        rows = len(members)
        if (type(call['input_shape']) is not list
                or any(type(value) is not int for value in call['input_shape'])
                or call['input_shape'] != [rows, protocol['input_shape'][1]]):
            raise ValueError('observed call shape does not match members')
        _digest(call['profile_sha256'], 'call profile')
        if (str(rows) not in protocol['profile_sha256_by_rows']
                or call['profile_sha256'] != protocol['profile_sha256_by_rows'][str(rows)]):
            raise ValueError('call lacks its exact row-count profile')
        if expected_mode == 'serial' and rows != 1:
            raise ValueError('serial mode cannot launch grouped calls')
        _timestamp(call['launch_ns'], 'launch', start, end)
        _timestamp(call['finish_ns'], 'finish', start, end, optional=True)
        if call['state'] not in ('succeeded', 'failed', 'unresolved') or type(call['synchronized']) is not bool:
            raise ValueError('invalid call state or synchronization evidence')
        if call['state'] == 'succeeded':
            if call['error'] is not None:
                raise ValueError('successful call cannot carry an error')
        else:
            _name(call['error'], 'call error')
        if call['state'] == 'unresolved':
            if call['finish_ns'] is not None or call['synchronized']:
                raise ValueError('unresolved calls cannot claim completion')
        elif call['finish_ns'] is None or call['finish_ns'] < call['launch_ns']:
            raise ValueError('completed call needs an ordered finish')
        if call['state'] == 'succeeded' and not call['synchronized']:
            raise ValueError('successful call needs synchronized completion')
        teams, inputs = set(), set()
        for index, member in enumerate(members):
            _fields(member, ('job_id', 'broker_id', 'broker_job_id', 'row_offset'), 'member')
            job_id = member['job_id']
            _name(job_id, 'member job ID')
            if type(member['row_offset']) is not int or member['row_offset'] != index:
                raise ValueError('member row offsets must match recorded order')
            if job_id not in by_id or job_id in call_members:
                raise ValueError('unknown or repeated call member')
            row, job = by_id[job_id], jobs[job_id]
            if (row['call_id'] != call['call_id'] or row['selected_ns'] > call['launch_ns']
                    or member['broker_id'] != row['broker_id']
                    or type(member['broker_job_id']) is not type(row['broker_job_id'])
                    or member['broker_job_id'] != row['broker_job_id']):
                raise ValueError('member call identity or selection differs')
            if row['state'] == 'succeeded' and (call['state'] != 'succeeded'
                    or row['returned_ns'] < call['finish_ns']):
                raise ValueError('successful result lacks completed synchronized call')
            if call['state'] == 'failed' and row['broker_terminal_state'] != 'failed':
                raise ValueError('failed launched calls require failed member outcomes')
            if call['state'] == 'unresolved' and row['broker_terminal_state'] is not None:
                raise ValueError('unresolved call cannot claim a member terminal outcome')
            if (call['finish_ns'] is not None and row['state'] != 'unresolved'
                    and row['returned_ns'] < call['finish_ns']):
                raise ValueError('terminal delivery cannot precede call completion')
            if row['state'] == 'cancelled' and row['returned_ns'] < call['launch_ns']:
                raise ValueError('pre-launch cancellation cannot be a launched member')
            teams.add(job['team_id'])
            inputs.add(job['input_sha256'])
            call_members.add(job_id)
        mixed_group |= rows > 1 and len(teams) > 1 and len(inputs) > 1
        calls_by_id[call['call_id']] = call
        if call['finish_ns'] is not None:
            intervals.append((call['launch_ns'], call['finish_ns']))
    if {job_id for job_id, row in by_id.items() if row['call_id'] is not None} != call_members:
        raise ValueError('attempt-to-call mapping is incomplete')
    intervals.sort()
    if any(first[1] > second[0] for first, second in zip(intervals, intervals[1:])):
        raise ValueError('one-owner model-call intervals cannot overlap')
    ordered_calls = sorted(calls, key=lambda call: call['launch_ns'])
    if any(call['state'] != 'succeeded' or not call['synchronized']
           for call in ordered_calls[:-1]):
        raise ValueError('no later launch is allowed after a failed or uncertain call')
    if any(first['finish_ns'] > second['launch_ns']
           for first, second in zip(ordered_calls, ordered_calls[1:])):
        raise ValueError('one-owner model-call intervals cannot overlap')
    return window, jobs, by_id, calls_by_id, mixed_group


def _summarize(window, jobs, by_id, calls, positions):
    seconds = (window['end_ns'] - window['start_ns']) / 1_000_000_000

    def group(rows, planned):
        states = Counter(row['state'] for row in rows)
        successes = [row for row in rows if row['state'] == 'succeeded']
        def timing(selected):
            return {'ready_to_selection_ns': _distribution([r['selected_ns'] - r['ready_ns']
                        for r in selected if r['selected_ns'] is not None]),
                    'selection_to_launch_ns': _distribution([calls[r['call_id']]['launch_ns'] - r['selected_ns']
                        for r in selected if r['call_id'] is not None]),
                    'response_ns': _distribution([r['returned_ns'] - r['attempt_ns'] for r in selected])}
        return {'planned': planned, 'attempted': len(rows), 'unattempted': planned - len(rows),
                'accepted': sum(row['broker_id'] is not None for row in rows),
                'acceptance_unknown': sum(row['state'] == 'unresolved' and row['broker_id'] is None for row in rows),
                'states': {s: states[s] for s in _STATES},
                'cancellation_requests': sum(row['cancellation_requested'] for row in rows),
                'broker_terminal_states': {**{state: sum(row['broker_terminal_state'] == state for row in rows)
                                              for state in ('succeeded', 'failed', 'cancelled')},
                                           'unknown': sum(row['broker_terminal_state'] is None for row in rows)},
                'successful_jobs_per_second': len(successes) / seconds,
                'successful_input_positions_per_second': len(successes) * positions / seconds,
                'timing_by_state': {state: timing([row for row in rows if row['state'] == state])
                                    for state in _STATES}}

    result = group(list(by_id.values()), len(jobs))
    result['unattempted_job_ids'] = [key for key in jobs if key not in by_id]
    result['elapsed_ns'] = window['end_ns'] - window['start_ns']
    result['window_id'] = window['window_id']
    result['drained'] = window['drained']
    result['clock'] = window['clock']
    result['clock_resolution_ns'] = window['clock_resolution_ns']
    result['model_calls'] = len(calls)
    call_states = Counter(call['state'] for call in calls.values())
    result['model_call_states'] = {state: call_states[state]
                                   for state in ('succeeded', 'failed', 'unresolved')}
    result['synchronized_model_call_duration_ns'] = _distribution([
        call['finish_ns'] - call['launch_ns'] for call in calls.values()
        if call['finish_ns'] is not None and call['synchronized']])
    result['batch_rows'] = _distribution([len(call['members']) for call in calls.values()])
    result['teams'] = {team: group([row for key, row in by_id.items() if jobs[key]['team_id'] == team],
                                   sum(job['team_id'] == team for job in jobs.values()))
                       for team in sorted({job['team_id'] for job in jobs.values()})}
    return result


def _output_tensor(value, shape, cap):
    if (type(value) is not torch.Tensor or value.device.type != 'cpu'
            or value.layout != torch.strided or value.dtype != torch.float32
            or tuple(value.shape) != tuple(shape) or value.requires_grad or value.grad_fn is not None
            or value.numel() * value.element_size() > cap):
        raise ValueError('output is not a bounded detached CPU float32 tensor of the frozen shape')
    owned = value.detach().contiguous().clone()
    if not bool(torch.isfinite(owned).all()):
        raise ValueError('output contains nonfinite values')
    return owned


def analyze_pair(protocol, serial_window, grouped_window, *, expected_protocol_sha256,
                 serial_outputs, grouped_outputs):
    """Reconcile two measured windows and compare their retained actual logits.

    A passing result is only a consistency check of supplied records and
    outputs. Profile approval, held-out certification, warmup, memory telemetry,
    trusted event collection and durable storage are external requirements.
    Caller must own inputs and tensors exclusively through this snapshot.
    """
    protocol, digest = freeze_protocol(protocol)
    if digest != expected_protocol_sha256:
        raise ValueError('protocol does not match externally frozen digest')
    serial, jobs, left, left_calls, _ = _window(protocol, digest, serial_window, 'serial')
    grouped, _, right, right_calls, mixed = _window(protocol, digest, grouped_window, 'grouped')
    if (serial['window_id'] == grouped['window_id']
            or serial['owner_session_id'] != grouped['owner_session_id']
            or serial['clock_resolution_ns'] != grouped['clock_resolution_ns']
            or not (serial['end_ns'] <= grouped['start_ns'] or grouped['end_ns'] <= serial['start_ns'])):
        raise ValueError('windows must be distinct, nonoverlapping and use one owner')
    serial_broker_jobs = {(r['broker_id'], r['broker_job_id']) for r in left.values()
                          if r['broker_id'] is not None}
    grouped_broker_jobs = {(r['broker_id'], r['broker_job_id']) for r in right.values()
                           if r['broker_id'] is not None}
    if serial_broker_jobs & grouped_broker_jobs:
        raise ValueError('broker-scoped job identities cannot be reused across windows')
    serial_brokers = {identity[0] for identity in serial_broker_jobs}
    grouped_brokers = {identity[0] for identity in grouped_broker_jobs}
    if len(serial_brokers) > 1 or len(grouped_brokers) > 1:
        raise ValueError('each window must use one broker identity')
    earlier = serial if serial['start_ns'] < grouped['start_ns'] else grouped
    if not earlier['drained']:
        raise ValueError('a later window requires independently confirmed prior owner drain')
    positions = protocol['input_shape'][1]
    summaries = {'serial': _summarize(serial, jobs, left, left_calls, positions),
                 'grouped': _summarize(grouped, jobs, right, right_calls, positions)}
    all_succeeded = (set(left) == set(jobs) and set(right) == set(jobs)
                     and all(row['state'] == 'succeeded' for row in left.values())
                     and all(row['state'] == 'succeeded' for row in right.values()))
    if type(serial_outputs) is not dict or type(grouped_outputs) is not dict:
        raise ValueError('output maps must be exact dictionaries')
    if not set(serial_outputs).issubset(jobs) or not set(grouped_outputs).issubset(jobs):
        raise ValueError('output maps contain unbound jobs')
    checks = []
    for job in protocol['jobs']:
        key = job['job_id']
        if (key not in left or key not in right or left[key]['state'] != 'succeeded'
                or right[key]['state'] != 'succeeded'):
            continue
        try:
            reference = _output_tensor(serial_outputs[key], protocol['output_shape'], protocol['output_cap_bytes'])
            candidate = _output_tensor(grouped_outputs[key], protocol['output_shape'], protocol['output_cap_bytes'])
            if (canonical_digest(reference) != left[key]['output_sha256']
                    or canonical_digest(candidate) != right[key]['output_sha256']):
                raise ValueError('actual output does not match its recorded fingerprint')
            comparison = _comparison(reference, candidate, protocol['atol'], protocol['rtol'])
            checks.append({'job_id': key, **comparison})
        except (KeyError, ValueError) as exc:
            checks.append({'job_id': key, 'status': 'invalid_output',
                           'error_type': type(exc).__name__})
    matched = all_succeeded and len(checks) == len(jobs) and all(c['status'] == 'matched' for c in checks)
    return {'version': VERSION, 'protocol_sha256': digest, 'summaries': summaries,
            'same_recorded_broker': bool(serial_brokers and serial_brokers == grouped_brokers),
            'all_jobs_succeeded': all_succeeded, 'observed_mixed_group': mixed,
            'output_checks': checks, 'all_outputs_matched': matched,
            'measured_pair_consistent': bool(serial['drained'] and grouped['drained']
                                             and all_succeeded and mixed and matched)}
