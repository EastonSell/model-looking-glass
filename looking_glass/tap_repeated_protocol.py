"""Frozen schedule and evidence checks for the repeated two-token tap study.

This module has no device access. Evidence is a trusted local record, not an
authentication of an owner, device, model, or activity screen.
"""

from collections import Counter
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import re
import statistics


PROTOCOL = 'gpu-tap-latency-repeated-v1'
PREVIOUS_EXPERIMENT_PROTOCOL = 'gpu-tap-latency-repeated-v1.1'
PREVIOUS_LEAN_PROTOCOL = 'gpu-tap-latency-repeated-v1.2'
EXPERIMENT_PROTOCOL = 'gpu-tap-latency-repeated-v1.3'
LEAN_PROTOCOLS = (PREVIOUS_LEAN_PROTOCOL, EXPERIMENT_PROTOCOL)
START_GRANT_MAX_NS = 1_000_000_000
SCHEDULE_SHA256 = 'ce2f990a6937da86565a21c52f13ad54f11b1e683421a8a2060e544694fe6bf9'
_SCHEDULE_CANONICAL_SHA256 = '5eee2638129632d7113fd136bdd54bd3f5ca116e54ff4d748b191439c4011787'
MODES = ('none', 'direct', 'gradient')
SEEDS = (53101, 53102, 53103, 53104, 53105)
WARMUPS = (('none', 'direct', 'gradient'), ('gradient', 'direct', 'none'))
IDENTITY = ['transformer.h.0', 'block', 0, 0, 0, [0], [0], [1]]
GIB = 1024 ** 3
HEADROOM = 256 * 1024 ** 2
OUTSIDE_CAP = 256 * 1024 ** 2
EVENT_ENCLOSURE_TOLERANCE_NS = 100_000
PROCESS_AFTER_MAX_DELAY_NS = 5_000_000_000
FINAL_POSTCHECK_MAX_NS = 2_000_000_000
MEMORY_ROUNDING_TOLERANCE_BYTES = 2 * 1024 ** 2
# A prospective conservative cross-provider reconciliation bound. The CUDA
# total still controls admission; agreeing within this bound proves no identity.
CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES = 2 * 1024 ** 2
IDLE_BEFORE_WINDOW_NS = 2_000_000_000
IDLE_AFTER_DELAY_NS = 1_000_000_000
IDLE_AFTER_WINDOW_NS = 5_000_000_000


def load_schedule(path):
    """Read the exact frozen file, not a newly generated shuffle."""
    with Path(path).open('rb') as stream:
        raw = stream.read(32 * 1024 + 1)
    if len(raw) > 32 * 1024 or sha256(raw).hexdigest() != SCHEDULE_SHA256:
        raise ValueError('frozen schedule bytes do not match the protocol')
    schedule = json.loads(raw)
    if (type(schedule) is not dict or schedule.get('protocol') != PROTOCOL
            or schedule.get('warmup_rounds') != [list(order) for order in WARMUPS]
            or type(schedule.get('sessions')) is not list or len(schedule['sessions']) != 5):
        raise ValueError('invalid frozen schedule')
    permutations = Counter(itertools.permutations(MODES))
    for index, row in enumerate(schedule['sessions']):
        if (type(row) is not dict or type(row.get('session')) is not int or row['session'] != index
                or type(row.get('seed')) is not int or row['seed'] != SEEDS[index]
                or type(row.get('measured_rounds')) is not list or len(row['measured_rounds']) != 12
                or any(type(order) is not list or len(order) != 3
                       or any(type(mode) is not str or mode not in MODES for mode in order)
                       for order in row['measured_rounds'])
                or Counter(tuple(order) for order in row['measured_rounds'])
                   != Counter({key: 2 for key in permutations})):
            raise ValueError('invalid balanced session schedule')
    return schedule


def session_order(schedule, session):
    if type(session) is not int or session not in range(5):
        raise ValueError('session must be an integer from 0 through 4')
    try:
        if type(schedule) is not dict:
            raise ValueError('schedule must be an exact mapping')
        raw = json.dumps(schedule, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
        if len(raw) > 32 * 1024 or sha256(raw).hexdigest() != _SCHEDULE_CANONICAL_SHA256:
            raise ValueError('schedule differs from the frozen snapshot')
        schedule = json.loads(raw)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError('invalid schedule snapshot') from exc
    rounds = [('warmup', n, order) for n, order in enumerate(schedule['warmup_rounds'])]
    rounds.extend(('measured', n, order) for n, order in enumerate(schedule['sessions'][session]['measured_rounds']))
    return [{'stage': stage, 'round': number, 'position': position, 'mode': mode,
             'observation_id': f'{stage}:{number}:{position}'}
            for stage, number, order in rounds for position, mode in enumerate(order)]


def clock_minimum(clock):
    if (type(clock) is not dict or clock.get('name') != 'perf_counter_ns'
            or clock.get('monotonic') is not True or clock.get('adjustable') is not False
            or type(clock.get('implementation')) is not str or not clock['implementation']
            or type(clock.get('resolution_seconds')) not in (int, float)
            or not math.isfinite(clock['resolution_seconds'])
            or not 0 < clock['resolution_seconds'] <= 1e-6):
        raise ValueError('invalid or insufficient measurement clock')
    return math.ceil(100 * clock['resolution_seconds'] * 1_000_000_000)


def check_memory(row, *, expected_total, cap_bytes):
    """Check one sampled allocator/device state against the frozen guard."""
    if type(row) is not dict:
        raise ValueError('memory snapshot is missing')
    values = [row.get(key) for key in ('free_bytes', 'total_bytes', 'allocated_bytes', 'reserved_bytes')]
    if (any(type(v) is not int for v in values) or type(expected_total) is not int
            or type(cap_bytes) is not int):
        raise ValueError('memory values must be exact integers')
    free, total, allocated, reserved = values
    if (total != expected_total or total <= 0 or not 0 <= free <= total
            or not 0 <= allocated <= reserved <= total or free + reserved > total
            or not HEADROOM < cap_bytes <= total):
        raise ValueError('invalid memory counters or device total')
    reserve = max(GIB, (15 * total + 99) // 100)
    required = reserve + HEADROOM + OUTSIDE_CAP + max(0, cap_bytes - reserved)
    if reserved > cap_bytes - HEADROOM or free < required:
        raise ValueError('sampled allocator or device headroom is insufficient')
    return {'reserve_bytes': reserve, 'required_free_bytes': required}


def _valid_preflight(row, device, session, run_protocol):
    checks = row.get('checks')
    return (type(checks) is dict and checks.get('outputs_finite') is True
            and checks.get('outputs_exact') is True and checks.get('values_exact') is True
            and checks.get('preflight_modes') == ['none', 'id', 'direct', 'gradient']
            and checks.get('id_value_absent') is True
            and checks.get('run_id') == f'{run_protocol}-session-{session}'
            and json.dumps(checks.get('trace_sequences'), separators=(',', ':')) == '[0]'
            and checks.get('capture_modes') == ['id', 'direct', 'gradient']
            and json.dumps(checks.get('output_shape'), separators=(',', ':')) == '[1,2,50257]'
            and checks.get('output_dtype') == 'torch.float32' and checks.get('output_device') == device
            and type(checks.get('events_per_mode')) is int and checks['events_per_mode'] == 1
            and type(checks.get('value_tensor_bytes')) is int and checks['value_tensor_bytes'] == 3072
            and json.dumps(checks.get('event_identities'), sort_keys=True, separators=(',', ':'))
                == json.dumps([IDENTITY], sort_keys=True, separators=(',', ':'))
            and json.dumps(checks.get('value_layouts'), sort_keys=True, separators=(',', ':'))
                == json.dumps([{'shape': [1, 1, 768], 'dtype': 'torch.float32', 'device': device}],
                              sort_keys=True, separators=(',', ':'))
            and json.dumps(checks.get('comparison'), sort_keys=True, separators=(',', ':'))
                == json.dumps({'operator': 'torch.equal', 'tolerance': 'none',
                               'guarantees_bitwise_identity': False,
                               'subnormal_handling': 'backend-dependent'}, sort_keys=True, separators=(',', ':')))


def _postchecks_ordered(rows):
    for index in range(0, len(rows), 2):
        observation, postcheck = rows[index:index + 2]
        if type(observation) is not dict or type(postcheck) is not dict:
            return False
        end, completed = observation.get('ended_ns'), postcheck.get('completed_ns')
        if (type(end) is not int or type(completed) is not int
                or not end <= completed <= end + FINAL_POSTCHECK_MAX_NS):
            return False
        if index + 2 < len(rows):
            following = rows[index + 2]
            if type(following) is not dict or type(following.get('started_ns')) is not int:
                return False
            if completed > following['started_ns']:
                return False
    return True


def _verify_rows(schedule, session, rows, clock, total, cap, device, run_protocol):
    minimum = clock_minimum(clock)
    order = session_order(schedule, session)
    if type(rows) is not list or len(rows) != 2 * (len(order) + 1):
        raise ValueError('missing, extra or partial observation/postcheck')
    if run_protocol in (PREVIOUS_EXPERIMENT_PROTOCOL, *LEAN_PROTOCOLS) and not _postchecks_ordered(rows):
        raise ValueError('postcheck completion order is invalid')
    previous_end = None
    previous_completed = None
    previous_guarded_end = None
    measured = []
    for index, expected in enumerate([None] + order):
        row, post = rows[index * 2:index * 2 + 2]
        obs_id = 'preflight' if expected is None else expected['observation_id']
        stage = 'preflight' if expected is None else expected['stage']
        if (type(row) is not dict or type(post) is not dict or row.get('stage') != stage
                or row.get('observation_id') != obs_id or row.get('postcheck_status') != 'pending'
                or post.get('stage') != 'postcheck' or post.get('observation_id') != obs_id
                or post.get('status') != 'passed'):
            raise ValueError('observation or postcheck sequence is invalid')
        check_memory(row.get('memory_before'), expected_total=total, cap_bytes=cap)
        check_memory(post.get('memory_after'), expected_total=total, cap_bytes=cap)
        peak_a, peak_r = row.get('peak_allocated_bytes'), row.get('peak_reserved_bytes')
        if (type(peak_a) is not int or type(peak_r) is not int
                or not 0 <= peak_a <= peak_r <= cap - HEADROOM
                or peak_a < max(row['memory_before']['allocated_bytes'], post['memory_after']['allocated_bytes'])
                or peak_r < max(row['memory_before']['reserved_bytes'], post['memory_after']['reserved_bytes'])):
            raise ValueError('invalid allocator peak counters')
        if expected is None:
            pre_start, pre_end = row.get('started_ns'), row.get('ended_ns')
            if (not _valid_preflight(row, device, session, run_protocol) or type(pre_start) is not int or type(pre_end) is not int
                    or not 0 <= pre_start < pre_end < 2 ** 63):
                raise ValueError('preflight differs from the fixed tap contract')
            if run_protocol in LEAN_PROTOCOLS:
                memory_start, memory_end, gate_start, gate_end = (row.get(k) for k in
                    ('memory_snapshot_started_ns', 'memory_snapshot_ended_ns', 'gate_lower_ns', 'gate_upper_ns'))
                if (any(type(value) is not int for value in (memory_start, memory_end, gate_start, gate_end))
                        or not 0 <= memory_start <= memory_end <= gate_start <= pre_start <= gate_end < 2 ** 63
                        or gate_end - gate_start > 250_000_000 or pre_start - memory_start > 2_000_000_000):
                    raise ValueError('preflight memory or authorization interval is invalid')
            previous_end = pre_end
            previous_completed = post.get('completed_ns')
            continue
        if any(type(row.get(k)) is not type(v) or row.get(k) != v for k, v in expected.items()):
            raise ValueError('sample is not in the frozen order')
        if type(row.get('session')) is not int or row['session'] != session:
            raise ValueError('sample session identity differs')
        start, end, wall, event = (row.get(k) for k in ('started_ns', 'ended_ns', 'wall_ns', 'device_event_ms'))
        if run_protocol in LEAN_PROTOCOLS:
            check_memory(row.get('memory_after'), expected_total=total, cap_bytes=cap)
            if (type(row.get('postcheck_completed_ns')) is not int
                    or row['postcheck_completed_ns'] != post.get('completed_ns')
                    or json.dumps(row['memory_after'], sort_keys=True, separators=(',', ':'))
                       != json.dumps(post['memory_after'], sort_keys=True, separators=(',', ':'))):
                raise ValueError('sample and postcheck evidence differ')
            completion, guarded_start, guarded_end, guarded_wall = (row.get(k) for k in
                ('completion_ns', 'guarded_started_ns', 'guarded_ended_ns', 'guarded_wall_ns'))
            if (any(type(value) is not int for value in (completion, guarded_start, guarded_end, guarded_wall))
                    or type(start) is not int or type(end) is not int or type(wall) is not int
                    or type(post.get('completed_ns')) is not int
                    or not 0 <= guarded_start <= start < completion <= end <= post['completed_ns'] <= guarded_end < 2 ** 63
                    or type(previous_completed) is not int or guarded_start < previous_completed
                    or (previous_guarded_end is not None and guarded_start < previous_guarded_end)
                    or completion - start != wall or guarded_end - guarded_start != guarded_wall):
                raise ValueError('lean and guarded timing intervals are invalid')
            previous_guarded_end = guarded_end
        elif type(start) is int and type(end) is int and type(wall) is int and end - start != wall:
            raise ValueError('sample wall interval is invalid')
        if (any(type(v) is not int for v in (start, end, wall)) or not 0 <= start < end < 2 ** 63
                or wall < minimum
                or type(event) not in (int, float) or not math.isfinite(event) or event < 0
                or event * 1_000_000 > wall + EVENT_ENCLOSURE_TOLERANCE_NS
                or (previous_end is not None and start < previous_end)):
            raise ValueError('invalid or overlapping sample timing')
        previous_end = end
        previous_completed = post.get('completed_ns')
        mode = expected['mode']
        count, nbytes = (0, 0) if mode == 'none' else (1, 3072)
        if (type(row.get('events')) is not int or row['events'] != count
                or type(row.get('recorded_tensor_bytes')) is not int or row['recorded_tensor_bytes'] != nbytes
                or type(row.get('output_shape')) is not list
                or any(type(v) is not int for v in row['output_shape'])
                or row['output_shape'] != [1, 2, 50257] or row.get('output_dtype') != 'torch.float32'
                or row.get('output_device') != device or row.get('default_stream') is not True
                or row.get('output_finite') is not True
                or row.get('capture_finite') is not (None if mode == 'none' else True)):
            raise ValueError('sample output, stream or capture differs from the fixed contract')
        if mode != 'none' and (json.dumps(row.get('trace_identity'), separators=(',', ':'))
                != json.dumps(IDENTITY, separators=(',', ':'))
                or type(row.get('trace_sequence')) is not int or row['trace_sequence'] != 0
                or row.get('trace_run_id') != f'{run_protocol}-session-{session}' or row.get('trace_mode') != mode
                or type(row.get('value_shape')) is not list or any(type(v) is not int for v in row['value_shape'])
                or row['value_shape'] != [1, 1, 768] or row.get('value_dtype') != 'torch.float32'
                or row.get('value_device') != device):
            raise ValueError('sample trace attribution differs from the fixed contract')
        if mode == 'none' and any(row.get(k) is not None for k in
                                  ('trace_identity', 'trace_sequence', 'trace_run_id', 'trace_mode',
                                   'value_shape', 'value_dtype', 'value_device')):
            raise ValueError('no-capture mode contains a trace')
        if stage == 'measured':
            measured.append(row)
    return measured


def summarize_session(schedule, session, rows, *, clock, total_bytes, cap_bytes, device, run_protocol=PROTOCOL):
    """Recompute a complete journal's paired statistics; no screen is inferred."""
    if type(run_protocol) is not str or run_protocol not in (PROTOCOL, PREVIOUS_EXPERIMENT_PROTOCOL, *LEAN_PROTOCOLS):
        raise ValueError('unsupported experiment identity')
    rows = _verify_rows(schedule, session, rows, clock, total_bytes, cap_bytes, device, run_protocol)
    groups = {}
    for row in rows:
        groups.setdefault(row['round'], {})[row['mode']] = row
    if len(groups) != 12 or any(set(group) != set(MODES) for group in groups.values()):
        raise ValueError('measured paired rounds are incomplete')
    paired = []
    for number in range(12):
        group = groups[number]
        none, direct, gradient = (group[mode]['wall_ns'] for mode in MODES)
        paired.append({'round': number, 'gradient_over_direct': gradient / direct,
                       'direct_over_none': direct / none,
                       'gradient_minus_direct_ms': (gradient - direct) / 1_000_000,
                       'direct_minus_none_ms': (direct - none) / 1_000_000})
        if run_protocol in LEAN_PROTOCOLS:
            paired[-1]['guarded_wall_ns'] = {mode: group[mode]['guarded_wall_ns'] for mode in MODES}
    result = {'session': session, 'warmup_samples': 6, 'measured_samples': len(rows),
            'paired_rounds': paired,
            'median_gradient_over_direct': statistics.median(p['gradient_over_direct'] for p in paired),
            'median_direct_over_none': statistics.median(p['direct_over_none'] for p in paired),
            'median_gradient_minus_direct_ms': statistics.median(p['gradient_minus_direct_ms'] for p in paired),
            'median_direct_minus_none_ms': statistics.median(p['direct_minus_none_ms'] for p in paired),
            'mode_wall_ns_medians': {mode: statistics.median(row['wall_ns'] for row in rows if row['mode'] == mode)
                                     for mode in MODES},
            'mode_device_event_ms_medians': {mode: statistics.median(row['device_event_ms'] for row in rows
                                                                      if row['mode'] == mode) for mode in MODES}}
    if run_protocol in LEAN_PROTOCOLS:
        result['mode_guarded_wall_ns_medians'] = {mode: statistics.median(row['guarded_wall_ns'] for row in rows
                                                                           if row['mode'] == mode) for mode in MODES}
    return result


def classify_activity(*, observations, errors, process_queries, gpu_uuid, own_pid,
                      workload_started_ns, workload_ended_ns, expected_total_bytes, exclusive):
    """Screen raw private observations; availability and contamination are independent.

    This applies sampled coverage rules. It does not prove physical exclusivity
    or authenticate the telemetry provider and preserves foreign PIDs privately.
    """
    unavailable, contaminated = [], []
    if (type(exclusive) is not bool or type(gpu_uuid) is not str or not gpu_uuid
            or type(own_pid) is not int or own_pid <= 0
            or type(expected_total_bytes) is not int or expected_total_bytes <= 0
            or type(workload_started_ns) is not int or type(workload_ended_ns) is not int
            or not 0 <= workload_started_ns <= workload_ended_ns):
        raise ValueError('invalid activity-screen identity or workload bounds')
    if type(observations) is not list or type(errors) is not list or type(process_queries) is not list:
        raise ValueError('activity-screen records must be exact lists')
    if not exclusive:
        unavailable.append('reservation_not_attested')
    valid = []
    for row in observations:
        if type(row) is not dict:
            unavailable.append('invalid_telemetry')
            continue
        start, end, observed, total, used, free, util = (row.get(k) for k in
            ('query_started_ns', 'query_ended_ns', 'observed_ns', 'total_bytes',
             'used_bytes', 'free_bytes', 'utilization_percent'))
        if (any(type(v) is not int for v in (start, end, observed, total, used, free, util))
                or not 0 <= start <= end == observed or total <= 0 or not 0 <= used <= total
                or not 0 <= free <= total or used + free > total + MEMORY_ROUNDING_TOLERANCE_BYTES
                or any(value % (1024 ** 2) for value in (total, used, free))
                or not 0 <= util <= 100
                or type(row.get('gpu_uuid')) is not str or row['gpu_uuid'].casefold() != gpu_uuid.casefold()):
            unavailable.append('invalid_telemetry')
            continue
        valid.append(row)
    valid.sort(key=lambda row: row['observed_ns'])
    if any(later['query_started_ns'] <= earlier['query_ended_ns']
           for earlier, later in zip(valid, valid[1:])):
        unavailable.append('overlapping_or_duplicate_telemetry_queries')
    if errors:
        unavailable.append('telemetry_query_error')
    if not valid or {row['total_bytes'] for row in valid} != {expected_total_bytes}:
        unavailable.append('device_total_or_identity_unverified')
    before_all = [row for row in valid if workload_started_ns - IDLE_BEFORE_WINDOW_NS <= row['query_started_ns']
                  and row['query_ended_ns'] < workload_started_ns]
    after_all = [row for row in valid if row['query_started_ns'] >= workload_ended_ns + IDLE_AFTER_DELAY_NS
                 and row['query_ended_ns'] <= workload_ended_ns + IDLE_AFTER_WINDOW_NS]
    before, after = before_all[-3:], after_all[-3:]
    if len(before) != 3 or len(after) != 3:
        unavailable.append('insufficient_idle_observations')
    if before and after:
        window = [row for row in valid if before[0]['observed_ns'] <= row['observed_ns'] <= after[-1]['observed_ns']]
        if any(later['query_ended_ns'] - earlier['query_started_ns'] > 2_000_000_000
               for earlier, later in zip(window, window[1:])):
            unavailable.append('telemetry_coverage_gap')
    if before and (workload_started_ns - before[-1]['query_started_ns'] > 2_000_000_000):
        unavailable.append('initial_telemetry_stale')
    if any(row['utilization_percent'] > 5 for row in before_all + after_all):
        contaminated.append('idle_utilization_above_five_percent')
    foreign = set()
    if ([row.get('phase') if type(row) is dict else None for row in process_queries]
            != ['before', 'after']):
        unavailable.append('process_query_unavailable')
    for query in process_queries:
        if type(query) is not dict:
            unavailable.append('process_query_unavailable')
            continue
        phase = query.get('phase')
        qstart, qend, rows = query.get('query_started_ns'), query.get('query_ended_ns'), query.get('processes')
        if (phase not in ('before', 'after') or query.get('ok') is not True or type(qstart) is not int
                or type(qend) is not int or not 0 <= qstart <= qend or type(rows) is not list
                or (phase == 'before' and (qend > workload_started_ns
                    or workload_started_ns - qstart > 2_000_000_000))
                or (phase == 'after' and (qstart < workload_ended_ns
                    or qend - qstart > 2_000_000_000
                    or qend - workload_ended_ns > PROCESS_AFTER_MAX_DELAY_NS))):
            unavailable.append('process_query_unavailable')
            continue
        for row in rows:
            if (type(row) is not dict or type(row.get('gpu_uuid')) is not str
                    or row['gpu_uuid'].casefold() != gpu_uuid.casefold()
                    or type(row.get('pid')) is not int or row['pid'] <= 0
                    or type(row.get('name')) is not str or not row['name'].strip()):
                unavailable.append('process_query_unavailable')
                continue
            if row['pid'] != own_pid:
                foreign.add(row['pid'])
    if foreign:
        contaminated.append('foreign_process_listed')
    unavailable, contaminated = sorted(set(unavailable)), sorted(set(contaminated))
    status = 'contaminated' if contaminated else ('unverified' if unavailable else 'no_contamination_observed')
    return {'status': status, 'availability': 'unverified' if unavailable else 'complete',
            'contamination_reasons': contaminated, 'unavailable_reasons': unavailable,
            'before_idle_samples': before, 'after_idle_samples': after,
            'telemetry_samples': valid,
            'foreign_process_ids': sorted(foreign)}


def check_initial_activity(*, observations, errors, process_query, gpu_uuid, own_pid,
                           expected_total_bytes, now_ns):
    """Fail closed before preflight when the latest initial screen is invalid."""
    if (type(observations) is not list or type(errors) is not list or type(process_query) is not dict
            or type(gpu_uuid) is not str or not gpu_uuid or type(own_pid) is not int or own_pid <= 0
            or type(expected_total_bytes) is not int or expected_total_bytes <= 0
            or type(now_ns) is not int or now_ns < 0):
        raise ValueError('invalid initial activity inputs')
    if errors or len(observations) < 3:
        raise ValueError('initial telemetry is unavailable')
    previous = None
    for row in observations:
        if type(row) is not dict:
            raise ValueError('initial telemetry is malformed')
        start, end, observed, total, used, free, util = (row.get(k) for k in
            ('query_started_ns', 'query_ended_ns', 'observed_ns', 'total_bytes',
             'used_bytes', 'free_bytes', 'utilization_percent'))
        if (any(type(v) is not int for v in (start, end, observed, total, used, free, util))
                or not 0 <= start <= end == observed < now_ns or total != expected_total_bytes
                or not 0 <= used <= total or not 0 <= free <= total
                or used + free > total + MEMORY_ROUNDING_TOLERANCE_BYTES or not 0 <= util <= 100
                or any(value % (1024 ** 2) for value in (total, used, free))
                or type(row.get('gpu_uuid')) is not str or row['gpu_uuid'].casefold() != gpu_uuid.casefold()
                or (previous is not None and (start <= previous['query_ended_ns']
                    or end - previous['query_started_ns'] > 2_000_000_000))):
            raise ValueError('initial telemetry identity, values or coverage are invalid')
        previous = row
    fresh = [row for row in observations
             if now_ns - row['query_started_ns'] <= IDLE_BEFORE_WINDOW_NS]
    if len(fresh) < 3:
        raise ValueError('initial telemetry is stale or insufficient')
    qstart, qend, rows = (process_query.get(k) for k in ('query_started_ns', 'query_ended_ns', 'processes'))
    if (process_query.get('phase') != 'before' or process_query.get('ok') is not True
            or type(qstart) is not int or type(qend) is not int or not 0 <= qstart <= qend <= now_ns
            or now_ns - qstart > 2_000_000_000 or type(rows) is not list):
        raise ValueError('initial process query is unavailable or stale')
    foreign = set()
    for row in rows:
        if (type(row) is not dict or type(row.get('gpu_uuid')) is not str
                or row['gpu_uuid'].casefold() != gpu_uuid.casefold() or type(row.get('pid')) is not int
                or row['pid'] <= 0 or type(row.get('name')) is not str or not row['name'].strip()):
            raise ValueError('initial process row is invalid')
        if row['pid'] != own_pid:
            foreign.add(row['pid'])
    return {'initial_utilization_percent': [row['utilization_percent'] for row in fresh[-3:]],
            'eligible_initial_utilization_percent': [row['utilization_percent'] for row in fresh],
            'initial_oldest_query_started_ns': fresh[-3]['query_started_ns'],
            'initial_latest_query_ended_ns': fresh[-1]['query_ended_ns'],
            'initial_process_started_ns': qstart, 'initial_process_ended_ns': qend,
            'foreign_process_ids': sorted(foreign)}


def public_activity(screen):
    """Project a private screen without UUIDs, process names, or PIDs."""
    return {'status': screen['status'], 'availability': screen['availability'],
            'contamination_reasons': list(screen['contamination_reasons']),
            'unavailable_reasons': list(screen['unavailable_reasons']),
            'before_utilization_percent': [row['utilization_percent'] for row in screen['before_idle_samples']],
            'after_utilization_percent': [row['utilization_percent'] for row in screen['after_idle_samples']],
            'telemetry_samples': [{key: row[key] for key in ('query_started_ns', 'query_ended_ns',
                'observed_ns', 'total_bytes', 'used_bytes', 'free_bytes', 'utilization_percent')}
                                  for row in screen['telemetry_samples']],
            'foreign_process_count': len(screen['foreign_process_ids'])}


def _owner_components(value):
    """Read independently valid private run and process claims, if present."""
    if type(value) is not dict:
        return None, None
    run_id, pid, birth = value.get('run_id'), value.get('pid'), value.get('birth')
    if type(run_id) is not str or re.fullmatch('[0-9a-f]{32}', run_id) is None:
        run_id = None
    if (type(pid) is not int or not 0 < pid < 2 ** 32 or type(birth) is not str
            or re.fullmatch('[1-9][0-9]{0,19}', birth) is None or int(birth) >= 2 ** 64):
        process = None
    else:
        process = (pid, birth)
    return run_id, process


def _owner_identity(value):
    """Parse a complete private Windows identity without authenticating it."""
    run_id, process = _owner_components(value)
    if (type(value) is not dict or set(value) != {'run_id', 'pid', 'birth'}
            or run_id is None or process is None):
        return None
    return run_id, *process


def _screen_matches_journal(rows, activity):
    """Check the actual preflight and final postcheck source observations."""
    if not _postchecks_ordered(rows):
        return False
    first, last, post = rows[0], rows[-2], rows[-1]
    ended, completed = last.get('ended_ns'), post.get('completed_ns')
    return (type(ended) is int and type(completed) is int
            and ended <= completed <= ended + FINAL_POSTCHECK_MAX_NS
            and post.get('device_work_complete') is True
            and first['started_ns'] == activity['workload_started_ns']
            and completed == activity['workload_ended_ns'])


def summarize_attempts(schedule, attempts, *, trusted_binding):
    """Recompute five fixed attempt slots from trusted local owner records.

    This is a consistency audit, not proof of the asserted binding, lifecycle,
    telemetry, or execution. The real CUDA owner is not implemented here.
    """
    keys = ('source_commit', 'source_tree', 'model_sha256', 'runtime_sha256', 'device_sha256')
    if (type(trusted_binding) is not dict or set(trusted_binding) != set(keys)
            or any(type(trusted_binding[key]) is not str
                   or re.fullmatch('[0-9a-f]{40}' if key in ('source_commit', 'source_tree') else '[0-9a-f]{64}',
                                   trusted_binding[key]) is None for key in keys)
            or type(attempts) is not list or len(attempts) > 5):
        raise ValueError('invalid trusted binding or attempt inventory')
    by_session = {}
    for attempt in attempts:
        if (type(attempt) is not dict or type(attempt.get('session')) is not int
                or attempt['session'] not in range(5) or attempt['session'] in by_session):
            raise ValueError('duplicate or invalid attempt slot')
        by_session[attempt['session']] = attempt
    # Index run and process claims independently, even when another field is
    # malformed or the parent and child disagree. Partial claims cannot hide
    # an otherwise valid duplicate in another attempt.
    run_slots, process_slots = {}, {}
    for session, attempt in by_session.items():
        owner = attempt.get('owner')
        if type(owner) is not dict:
            continue
        for field in ('child_identity', 'parent_identity'):
            run_id, process = _owner_components(owner.get(field))
            if run_id is not None:
                run_slots.setdefault(run_id, set()).add(session)
            if process is not None:
                process_slots.setdefault(process, set()).add(session)
    ambiguous = set()
    for seen in (*run_slots.values(), *process_slots.values()):
        if len(seen) > 1:
            ambiguous.update(seen)
    slots, complete = [], []
    for session in range(5):
        attempt = by_session.get(session)
        if attempt is None:
            slots.append({'session': session, 'state': 'missing'})
            continue
        screen = None
        activity = attempt.get('activity')
        total = attempt.get('total_bytes')
        activity_total = attempt.get('activity_total_bytes')
        totals_bound = (type(total) is int and total > 0 and type(activity_total) is int
                        and activity_total > 0
                        and abs(total - activity_total) <= CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES)
        if (totals_bound and type(activity) is dict
                and type(activity.get('expected_total_bytes')) is int
                and activity['expected_total_bytes'] == activity_total):
            try:
                screen = classify_activity(**activity)
            except (ValueError, TypeError, KeyError, OverflowError):
                pass
        # This is separate from the main validation order: an early owner or
        # binding failure must not project a clean screen for an unverified
        # or different interval. The raw private attempt is still preserved.
        screen_bound = False
        if screen is not None:
            try:
                check_rows = attempt.get('journal')
                summarize_session(schedule, session, check_rows, clock=attempt.get('clock'),
                                  total_bytes=total, cap_bytes=attempt.get('cap_bytes'), device='cuda:0',
                                  run_protocol=EXPERIMENT_PROTOCOL)
                screen_owner = attempt.get('owner')
                child = (_owner_identity(screen_owner.get('child_identity'))
                         if type(screen_owner) is dict else None)
                parent = (_owner_identity(screen_owner.get('parent_identity'))
                          if type(screen_owner) is dict else None)
                device_uuid = screen_owner.get('gpu_uuid') if type(screen_owner) is dict else None
                screen_bound = (attempt.get('binding') == trusted_binding
                                and attempt.get('protocol') == EXPERIMENT_PROTOCOL
                                and attempt.get('evidence_kind') == 'supervised_cuda'
                                and type(device_uuid) is str and bool(device_uuid)
                                and type(activity.get('gpu_uuid')) is str
                                and device_uuid.casefold() == activity['gpu_uuid'].casefold()
                                and _screen_matches_journal(check_rows, activity)
                                and child is not None and child == parent and session not in ambiguous
                                and screen_owner.get('parent_observed_live_handle') is True
                                and screen_owner.get('activity_identity_held_through_queries') is True
                                and type(activity.get('own_pid')) is int and activity['own_pid'] == child[1])
            except (ValueError, TypeError, KeyError, OverflowError, IndexError):
                pass
        try:
            if (attempt.get('binding') != trusted_binding or attempt.get('evidence_kind') != 'supervised_cuda'
                    or attempt.get('protocol') != EXPERIMENT_PROTOCOL):
                raise ValueError('binding_or_evidence_kind_mismatch')
            cap = attempt.get('cap_bytes')
            if (type(total) is not int or type(cap) is not int or total <= 0 or cap != total // 4
                    or type(attempt.get('free_before_model_transfer')) is not int
                    or not 3 * GIB <= attempt['free_before_model_transfer'] <= total
                    or type(attempt.get('trace_claim_bytes')) is not int
                    or attempt['trace_claim_bytes'] != 2 * 64 * 1024):
                raise ValueError('memory_admission_unverified')
            owner = attempt.get('owner')
            child_identity = _owner_identity(owner.get('child_identity')) if type(owner) is dict else None
            parent_identity = _owner_identity(owner.get('parent_identity')) if type(owner) is dict else None
            if (type(owner) is not dict or owner.get('unsafe') is not False
                    or owner.get('synchronized_disposal') is not True
                    or owner.get('whole_run_claims_held') is not True
                    or owner.get('claims_released_after_disposal') is not True
                    or owner.get('device_lease_held_through_process_exit') is not True
                    or owner.get('supervisor_control_held_until_reap') is not True
                    or owner.get('child_reaped') is not True or type(owner.get('exit_code')) is not int
                    or owner['exit_code'] != 0
                    or owner.get('parent_observed_live_handle') is not True
                    or owner.get('activity_identity_held_through_queries') is not True
                    or child_identity is None or child_identity != parent_identity
                    or session in ambiguous
                    or type(activity) is not dict or type(activity.get('own_pid')) is not int
                    or activity['own_pid'] != child_identity[1]
                    or type(owner.get('gpu_uuid')) is not str or not owner['gpu_uuid']
                    or type(activity.get('gpu_uuid')) is not str
                    or owner['gpu_uuid'].casefold() != activity['gpu_uuid'].casefold()):
                raise ValueError('owner_lifecycle_unverified')
            if screen is None:
                raise ValueError('activity_record_missing_or_mismatched')
            rows = attempt.get('journal')
            result = summarize_session(schedule, session, rows, clock=attempt.get('clock'),
                                       total_bytes=total, cap_bytes=cap, device='cuda:0',
                                       run_protocol=EXPERIMENT_PROTOCOL)
            # Bind through the source-stamped final postcheck, not merely the
            # measured wall end; no device work may follow it before screening.
            if not _screen_matches_journal(rows, activity):
                # The private points remain in the raw attempt, but this
                # screen cannot describe the actual workload's interval.
                screen = None
                raise ValueError('journal_outside_activity_window')
            if screen['availability'] != 'complete':
                raise ValueError('activity_screen_unverified')
            slot = {'session': session, 'state': 'complete', 'screen': public_activity(screen), 'summary': result,
                    'device_totals': {'cuda_bytes': total, 'nvidia_smi_bytes': activity_total,
                                      'maximum_difference_bytes': CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES}}
            complete.append(slot)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            # Do not copy arbitrary exception text: Python can include an
            # untrusted private mapping key or device/path data in TypeError.
            reasons = {'binding_or_evidence_kind_mismatch', 'memory_admission_unverified',
                       'owner_lifecycle_unverified', 'activity_record_missing_or_mismatched',
                       'activity_screen_unverified', 'journal_outside_activity_window'}
            candidate = str(exc)
            reason = candidate if candidate in reasons else 'malformed_or_inconsistent_evidence'
            slot = {'session': session, 'state': 'incomplete', 'reason': reason}
            if type(total) is int and total > 0 and type(activity_total) is int and activity_total > 0:
                slot['device_totals'] = {'cuda_bytes': total, 'nvidia_smi_bytes': activity_total,
                                         'maximum_difference_bytes': CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES}
            if screen is not None and screen_bound:
                slot['screen'] = public_activity(screen)
        slots.append(slot)
    screen_groups = {status: [item['session'] for item in complete if item['screen']['status'] == status]
                     for status in ('no_contamination_observed', 'contaminated')}
    aggregate = None
    if len(complete) >= 3:
        summaries = [item['summary'] for item in complete]
        names = ('median_gradient_over_direct', 'median_direct_over_none',
                 'median_gradient_minus_direct_ms', 'median_direct_minus_none_ms')
        aggregate = {name: statistics.median(summary[name] for summary in summaries) for name in names}
        ratios = [summary['median_gradient_over_direct'] for summary in summaries]
        aggregate['gradient_over_direct_session_range'] = [min(ratios), max(ratios)]
        aggregate['mode_wall_ns_medians'] = {mode: statistics.median(summary['mode_wall_ns_medians'][mode]
                                                                      for summary in summaries) for mode in MODES}
        aggregate['mode_guarded_wall_ns_medians'] = {
            mode: statistics.median(summary['mode_guarded_wall_ns_medians'][mode] for summary in summaries)
            for mode in MODES}
        aggregate['mode_device_event_ms_medians'] = {
            mode: statistics.median(summary['mode_device_event_ms_medians'][mode] for summary in summaries)
            for mode in MODES}
        aggregate['exploratory_due_to_contamination'] = bool(screen_groups['contaminated'])
    return {'protocol': EXPERIMENT_PROTOCOL, 'schedule_protocol': PROTOCOL,
            'slots': slots, 'complete_sessions': len(complete),
            'screen_groups': screen_groups, 'aggregate': aggregate,
            'limitations': ['one model and input; sessions are correlated',
                            'records and telemetry are not authenticated',
                            'complete-case summaries are conditional on successful attempts']}
