"""Single-owner relay for a prospective paired throughput run.

Caller supplies ownership, admission, guarded device calls, failure custody,
and source-bound evidence. This module does not launch or attest an owner.
"""

from collections import deque
import copy
from fractions import Fraction
import hashlib
import math
import re
import sys
import threading
from time import get_clock_info, monotonic, perf_counter_ns

from .batch_validation import _comparison, _valid_output
from .grouped_broker import GroupedBroker, GroupedResult
from .grouped_owner_worker import GroupedWorkspaceWorker
from .grouped_throughput_stamps import SourceStamps
from .resident_worker import ForwardRequest
from .throughput_benchmark import _distribution, canonical_digest


_CLIENT_WAIT_SECONDS = 15.0
_WINDOW_DEADLINE_SECONDS = 30.0
_GROUPED_RELAY_POLICY = 'peer-relay-ready-by-head-fallback-v2'


def _coalescing_deadline_ns(head_ready_at, wait_seconds):
    """Use the exact saved binary64 seconds values for an inclusive deadline."""
    if (type(head_ready_at) is not float or not math.isfinite(head_ready_at) or head_ready_at < 0
            or type(wait_seconds) is not float or not math.isfinite(wait_seconds)
            or wait_seconds < 0):
        raise ValueError('invalid broker coalescing clock')
    return (Fraction.from_float(head_ready_at) + Fraction.from_float(wait_seconds)) * 1_000_000_000


class _Item:
    def __init__(self, job, request):
        self.job, self.request = job, request
        self.done = threading.Event()
        self.attempt_ns = self.ready_ns = self.selected_ns = self.returned_ns = None
        self.service_started_ns = self.service_finished_ns = None
        self.output = self.output_digest = self.failure = None


class _Relay:
    def __init__(self):
        self.condition = threading.Condition()
        self.items = deque()

    def put(self, item):
        with self.condition:
            if not self.condition.wait_for(lambda: len(self.items) < 2, _CLIENT_WAIT_SECONDS):
                raise TimeoutError('relay admission timed out')
            item.ready_ns = perf_counter_ns()
            self.items.append(item)
            self.condition.notify_all()

    def take(self):
        with self.condition:
            if not self.condition.wait_for(lambda: bool(self.items), .005):
                return None
            item = self.items.popleft()
            item.selected_ns = perf_counter_ns()
            self.condition.notify_all()
            return item

    def take_before(self, deadline_ns, *, stop, window_deadline):
        """Take a queued timely peer, without discarding a late or cancelled one."""
        with self.condition:
            while not stop.is_set() and monotonic() < window_deadline:
                if self.items:
                    item = self.items[0]
                    if item.ready_ns > deadline_ns:
                        return None
                    self.items.popleft()
                    item.selected_ns = perf_counter_ns()
                    self.condition.notify_all()
                    return item
                remaining = deadline_ns - perf_counter_ns()
                if remaining <= 0:
                    return None
                self.condition.wait(min(.005, float(remaining / 1_000_000_000)))
            return None


def _window(mode, owner, broker, model, jobs, requests, custody, *, interrupt_serial=False,
            guarded_call=None, stamps=None, synchronize=None):
    if guarded_call is None:
        def guarded_call(operation, /, *args, **kwargs):
            return operation(*args, **kwargs)
    if stamps is not None:
        unobserved_call = guarded_call

        def guarded_call(operation, /, *args, **kwargs):
            stamps.current_counts()
            result = unobserved_call(operation, *args, **kwargs)
            stamps.current_counts()
            return result

    def model_count():
        return len(model.calls) if stamps is None else stamps.current_counts()[0]

    def model_shapes(start):
        if stamps is None:
            return [list(shape) for shape in model.calls[start:]]
        return stamps.current_model_shapes(start)

    relay, go, stop = _Relay(), threading.Event(), threading.Event()
    ready = {team: threading.Event() for team in ('team-a', 'team-b')}
    items = {job['job_id']: _Item(job, requests[job['job_id']]) for job in jobs}
    outputs, errors, calls, control_errors = {}, [], [], []
    active = []
    custody['windows'][mode] = {'outputs': outputs, 'items': items, 'calls': calls,
                                'active': active, 'relay': relay}

    def client(team):
        ready[team].set()
        go.wait()
        for job in (job for job in jobs if job['team_id'] == team):
            if stop.is_set():
                break
            item = items[job['job_id']]
            try:
                item.attempt_ns = perf_counter_ns()
                relay.put(item)
                if not item.done.wait(_CLIENT_WAIT_SECONDS):
                    raise TimeoutError('client response timed out')
                if item.failure is not None:
                    raise RuntimeError('owner rejected or failed the request')
                outputs[job['job_id']] = item.output.detach().clone()
                item.output_digest = canonical_digest(outputs[job['job_id']])
                item.returned_ns = perf_counter_ns()
            except BaseException as exc:
                errors.append((job['job_id'], type(exc).__name__))
                if not isinstance(exc, Exception):
                    control_errors.append(exc)
                stop.set()
                break

    threads = [threading.Thread(target=client, args=(team,), daemon=True) for team in ready]
    for thread in threads:
        thread.start()
    if not all(signal.wait(_CLIENT_WAIT_SECONDS) for signal in ready.values()):
        stop.set()
        go.set()
        raise RuntimeError('client start barrier failed')
    window_start = perf_counter_ns()
    go.set()

    def selected_members():
        groups = []
        singles = []
        for _, handle in active:
            job = handle._job
            if job.group is not None and all(group is not job.group for group in groups):
                groups.append(job.group)
            elif job.single is not None:
                singles.append(job)
        if len(groups) == 1 and not singles:
            group = groups[0]
            return [job.request_id for job in group.jobs], group.group_id, True
        if len(singles) == 1 and not groups:
            return [singles[0].request_id], None, True
        return [], None, False

    def dispatch_ready():
        if not active:
            return
        shape_count = model_count()
        dispatch_start = perf_counter_ns()
        try:
            dispatched = guarded_call(broker.run_once, timeout=0)
        except BaseException:
            dispatch_finish = perf_counter_ns()
            members, group_id, known = selected_members()
            try:
                shapes = model_shapes(shape_count)
            except Exception:
                shapes = []  # Partial source stamps remain available after capture closes.
            calls.append({'members': members, 'group_id': group_id,
                          'membership_observed': known, 'dispatch_outcome': 'raised',
                          'dispatch_started_ns': dispatch_start, 'dispatch_finished_ns': dispatch_finish,
                          'model_entry_shapes': shapes})
            raise
        dispatch_finish = perf_counter_ns()
        if not dispatched:
            return
        members, group_id, known = selected_members()
        calls.append({'members': members, 'group_id': group_id, 'membership_observed': known,
                      'dispatch_outcome': 'returned', 'dispatch_started_ns': dispatch_start,
                      'dispatch_finished_ns': dispatch_finish,
                      'model_entry_shapes': model_shapes(shape_count)})
        for item, handle in tuple(active):
            job = handle._job
            if job.single is None and (job.group is None or not job.group.execution_finished):
                continue
            group = job.group
            ticket = guarded_call(handle.delivery)
            result = guarded_call(ticket.wait, 0)
            if (type(result) is not GroupedResult
                    or (result.status, result.team_id, result.request_id, result.job_id,
                        result.cancel_requested, result.error_code)
                    != ('ok', item.job['team_id'], item.job['job_id'], job.job_id, False, None)):
                item.failure = 'invalid_grouped_delivery'
                raise RuntimeError('grouped delivery identity or outcome differs')
            item.output = result.logits.detach().clone()
            if guarded_call(ticket.confirm, result) is not True:
                raise RuntimeError('grouped delivery confirmation is uncertain')
            del result
            if guarded_call(ticket.release) is not True or guarded_call(handle.dispose) is not True:
                raise RuntimeError('grouped delivery disposal is uncertain')
            active.remove((item, handle))
            item.service_started_ns, item.service_finished_ns = dispatch_start, dispatch_finish
            item.done.set()

    deadline = monotonic() + _WINDOW_DEADLINE_SECONDS
    try:
        while any(thread.is_alive() for thread in threads) and not stop.is_set() and monotonic() < deadline:
            if mode == 'grouped' and active:
                if len(active) == 1:
                    head = active[0][1]._job
                    peer = relay.take_before(
                        _coalescing_deadline_ns(head.ready_at, broker._max_coalesce_wait),
                        stop=stop, window_deadline=deadline)
                    if stop.is_set() or monotonic() >= deadline:
                        break
                    if peer is not None:
                        if peer.job['team_id'] == active[0][0].job['team_id']:
                            raise RuntimeError('relay peer belongs to the active team')
                        active.append((peer, guarded_call(broker.submit, peer.job['team_id'], peer.request)))
                if stop.is_set() or monotonic() >= deadline:
                    break
                dispatch_ready()
                continue
            item = relay.take()
            if item is None:
                continue
            if mode == 'serial':
                item.service_started_ns = perf_counter_ns()
                shape_count = model_count()
                returned = False
                try:
                    result = guarded_call(owner.execute_control, item.request)
                    if interrupt_serial:
                        raise KeyboardInterrupt('fabricated interruption after control return')
                    item.output = result.logits.detach().clone()
                    if guarded_call(owner.confirm_control_result, result) is not None:
                        raise RuntimeError('serial result confirmation is uncertain')
                    del result
                    returned = True
                finally:
                    item.service_finished_ns = perf_counter_ns()
                    pending_error = sys.exc_info()[1]
                    try:
                        shapes = model_shapes(shape_count)
                    except Exception:
                        if pending_error is None:
                            raise
                        shapes = []
                    calls.append({'members': [item.job['job_id']], 'group_id': None,
                                  'membership_observed': True,
                                  'dispatch_outcome': 'returned' if returned else 'raised',
                                  'dispatch_started_ns': item.service_started_ns,
                                  'dispatch_finished_ns': item.service_finished_ns,
                                  'model_entry_shapes': shapes})
                item.done.set()
            else:
                if stop.is_set():
                    break
                active.append((item, guarded_call(broker.submit, item.job['team_id'], item.request)))
    except BaseException as exc:
        errors.append(('owner', type(exc).__name__))
        if not isinstance(exc, Exception):
            control_errors.append(exc)
        stop.set()
    finally:
        if stop.is_set() or monotonic() >= deadline:
            stop.set()
            for item in items.values():
                if not item.done.is_set():
                    item.failure = 'incomplete'
                    item.done.set()
        for thread in threads:
            thread.join(_CLIENT_WAIT_SECONDS)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError('CPU client did not join; no owner drain is established')
    sync_error = None
    final_sync = 'not_requested'
    if synchronize is not None:
        if not errors and not active and not relay.items:
            try:
                guarded_call(synchronize)
                final_sync = 'completed'
            except BaseException as exc:
                sync_error = exc
                final_sync = 'failed'
        else:
            final_sync = 'skipped_uncertain_work'
    window_end = None if final_sync in ('failed', 'skipped_uncertain_work') else perf_counter_ns()
    elapsed_ns = None if window_end is None else window_end - window_start
    attempted = [item for item in items.values() if item.attempt_ns is not None]
    completed = [item for item in attempted if item.returned_ns is not None]
    teams = {}
    for team_id in sorted({job['team_id'] for job in jobs}):
        count = sum(item.job['team_id'] == team_id for item in completed)
        teams[team_id] = {'completed': count,
                          'completed_jobs_per_second': None if elapsed_ns is None else count * 1e9 / elapsed_ns,
                          'completed_input_positions_per_second': None if elapsed_ns is None else count * 4e9 / elapsed_ns}
    rows = [{'job_id': item.job['job_id'], 'team_id': item.job['team_id'],
             'attempt_ns': item.attempt_ns, 'relay_ready_ns': item.ready_ns,
             'relay_selected_ns': item.selected_ns, 'service_started_ns': item.service_started_ns,
             'service_finished_ns': item.service_finished_ns, 'returned_ns': item.returned_ns,
             'output_sha256': item.output_digest, 'failure': item.failure} for item in attempted]
    report = {'mode': mode, 'start_ns': window_start, 'end_ns': window_end,
            'relay_admission_policy': _GROUPED_RELAY_POLICY if mode == 'grouped' else 'serial-fifo-v1',
            'wall_duration_ns': elapsed_ns, 'planned': len(jobs),
            'final_sync': final_sync,
            'attempted': len(attempted), 'completed': len(completed),
            'unattempted': len(jobs) - len(attempted), 'failures': errors,
            'planned_job_ids': [job['job_id'] for job in jobs],
            'unattempted_job_ids': [item.job['job_id'] for item in items.values() if item.attempt_ns is None],
            'teams': teams,
            'completed_jobs_per_second': None if elapsed_ns is None else len(completed) * 1e9 / elapsed_ns,
            'completed_input_positions_per_second': None if elapsed_ns is None else len(completed) * 4e9 / elapsed_ns,
            'relay_wait_ns': _distribution([item.selected_ns - item.ready_ns for item in completed]),
            'service_ns': _distribution([item.service_finished_ns - item.service_started_ns for item in completed]),
            'service_interval': ('execute_control, output clone and confirmation' if mode == 'serial'
                                 else 'broker.run_once dispatch; excludes ticket clone and confirmation'),
            'response_ns': _distribution([item.returned_ns - item.attempt_ns for item in completed]),
            'calls': calls, 'rows': rows,
            'drained': not active and not relay.items and not errors and sync_error is None}
    if sync_error is not None:
        sync_error.partial_rehearsal = report
        raise sync_error
    if control_errors:
        interruption = control_errors[-1]
        interruption.partial_rehearsal = report
        raise interruption
    return report, outputs


def _validate_window(mode, window, source, jobs, max_coalesce_wait):
    """Bind source-observed claims and model events to completed dispatches."""
    expected = {job['job_id']: job['team_id'] for job in jobs}
    expected_relay_policy = _GROUPED_RELAY_POLICY if mode == 'grouped' else 'serial-fifo-v1'
    if (window['mode'] != mode or window['relay_admission_policy'] != expected_relay_policy
            or window['final_sync'] != 'completed' or not window['drained']
            or window['planned'] != 8 or window['attempted'] != 8 or window['completed'] != 8
            or window['unattempted'] != 0 or window['failures'] or window['wall_duration_ns'] <= 0
            or window['wall_duration_ns'] != window['end_ns'] - window['start_ns']
            or window['completed_jobs_per_second'] != 8 * 1e9 / window['wall_duration_ns']
            or window['completed_input_positions_per_second'] != 8 * 4e9 / window['wall_duration_ns']
            or set(window['planned_job_ids']) != set(expected) or window['unattempted_job_ids']
            or source['capture_completed'] is not True):
        raise RuntimeError('window did not complete and synchronize')
    rows = window['rows']
    if (len(rows) != 8 or {row['job_id'] for row in rows} != set(expected)
            or any(row['team_id'] != expected[row['job_id']] or row['failure'] is not None
                   or not (window['start_ns'] <= row['attempt_ns'] <= row['relay_ready_ns']
                           <= row['relay_selected_ns'] <= row['service_started_ns']
                           <= row['service_finished_ns'] <= row['returned_ns'] <= window['end_ns'])
                   for row in rows)):
        raise RuntimeError('client identities or response boundaries do not match')
    row_by_job = {row['job_id']: row for row in rows}
    if (set(window['teams']) != set(expected.values())
            or any(window['teams'][team]['completed'] != 4
                   or window['teams'][team]['completed_jobs_per_second'] != 4 * 1e9 / window['wall_duration_ns']
                   or window['teams'][team]['completed_input_positions_per_second'] != 4 * 4e9 / window['wall_duration_ns']
                   for team in set(expected.values()))):
        raise RuntimeError('per-team rates differ from the full-window denominator')
    calls, models, claims = window['calls'], source['model_calls'], source['claims']
    wanted = 8 if mode == 'serial' else 4
    if (len(calls) != wanted or len(models) != wanted or len(claims) != (0 if mode == 'serial' else wanted)
            or any(call['dispatch_outcome'] != 'returned' or call['membership_observed'] is not True
                   for call in calls)):
        raise RuntimeError('source call count or dispatch outcome differs')
    seen, grouped_rows, queue_ns, late_count = [], {}, [], 0
    for index, (call, model) in enumerate(zip(calls, models)):
        shape = [1, 4] if mode == 'serial' else [2, 4]
        members = call['members']
        if (model['shape'] != shape or call['model_entry_shapes'] != [shape]
                or model['finish_hook_synchronized_argument'] is not True
                or not (window['start_ns'] <= call['dispatch_started_ns'] <= model['entry_ns']
                        <= model['python_return_event_ns'] <= model['finish_hook_ns']
                        <= call['dispatch_finished_ns'] <= window['end_ns'])
                or len(members) != shape[0] or any(member not in expected for member in members)):
            raise RuntimeError('model call and synchronized source boundaries differ')
        seen.extend(members)
        if any(row_by_job[member]['service_started_ns'] != call['dispatch_started_ns']
               or row_by_job[member]['service_finished_ns'] != call['dispatch_finished_ns']
               for member in members):
            raise RuntimeError('client service interval does not match its dispatch')
        if mode == 'serial':
            if call['group_id'] is not None:
                raise RuntimeError('serial control unexpectedly has a group')
            continue
        claim = claims[index]
        claim_members = claim['members']
        if (claim['claim_kind'] != 'group' or claim['group_id'] != call['group_id']
                or not (call['dispatch_started_ns'] <= claim['claim_return_event_ns'] <= model['entry_ns'])
                or [member['request_id'] for member in claim_members] != members
                or [member['row_index'] for member in claim_members] != [0, 1]
                or [member['team_id'] for member in claim_members] != [expected[member] for member in members]
                or len({expected[member] for member in members}) != 2):
            raise RuntimeError('selected group does not match dispatch and delivery identities')
        ready_float = [float.fromhex(member['ready_at_hex']) for member in claim_members]
        ready = [Fraction.from_float(value) for value in ready_float]
        for member, value, float_value in zip(claim_members, ready, ready_float):
            ready_ns = value * 1_000_000_000
            allowance = Fraction.from_float(math.ulp(float_value)) * 1_000_000_000 + 1
            if not (Fraction(row_by_job[member['request_id']]['relay_selected_ns']) <= ready_ns + allowance
                    and ready_ns <= Fraction(claim['claim_return_event_ns']) + allowance):
                raise RuntimeError('broker readiness lies outside relay selection and claim return')
            delay = Fraction(model['entry_ns']) - ready_ns
            if delay < 0:
                raise RuntimeError('model entry precedes broker readiness')
            queue_ns.append(round(delay))
            grouped_rows[member['request_id']] = (claim['group_id'], member['row_index'])
        ordered = sorted(zip(claim_members, ready),
                         key=lambda pair: row_by_job[pair[0]['request_id']]['relay_selected_ns'])
        head, peer = ordered
        if (row_by_job[head[0]['request_id']]['relay_selected_ns']
                == row_by_job[peer[0]['request_id']]['relay_selected_ns']):
            raise RuntimeError('relay head and peer selection order is ambiguous')
        head_deadline = (head[1] + Fraction.from_float(max_coalesce_wait)) * 1_000_000_000
        if Fraction(row_by_job[peer[0]['request_id']]['relay_ready_ns']) > head_deadline:
            raise RuntimeError('peer entered the relay after the head fallback')
        if peer[1] * 1_000_000_000 > head_deadline:
            late_count += 1
    if len(seen) != len(set(seen)) or set(seen) != set(expected):
        raise RuntimeError('source calls do not cover each planned job once')
    return grouped_rows, queue_ns, late_count


def _saved_row_sha256(row):
    array = row.numpy()
    if not array.flags.c_contiguous:
        raise RuntimeError('saved output row is not contiguous')
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def _clock_metadata():
    clocks = {}
    for name in ('perf_counter', 'monotonic'):
        info = get_clock_info(name)
        if (type(info.implementation) is not str or not 1 <= len(info.implementation) <= 256
                or type(info.resolution) is not float or not math.isfinite(info.resolution)
                or info.resolution <= 0 or info.monotonic is not True or type(info.adjustable) is not bool):
            raise RuntimeError('the owner monotonic clock metadata is invalid')
        clocks[name] = {'implementation': info.implementation, 'resolution_seconds': info.resolution,
                        'monotonic': info.monotonic, 'adjustable': info.adjustable}
    return clocks


def run_pair(*, owner, broker, model, jobs, requests, guarded_call, synchronize, memory, writer, custody,
             record, output_width=50257, atol=1e-6, rtol=1e-5):
    """Measure two bounded windows; caller owns all supervision and custody.

    This function is an unlaunched composition, not a GPU admission or an
    attestation. The fixed source-bound owner must root ``custody`` before
    entering it, acquire claims/leases, stop on every escape and preserve the
    writer and exception graph. ``record`` persists bounded private JSON.
    Hash checks bind the compared tensors to the recorded row bytes; a later
    independent replay must still read and verify the saved files.
    """
    if (type(jobs) is not list or len(jobs) != 8 or type(requests) is not dict
            or any(type(job) is not dict or type(job.get('job_id')) is not str
                   or type(job.get('team_id')) is not str for job in jobs)):
        raise ValueError('the fixed eight-job workload is required')
    expected = {job['job_id']: job['team_id'] for job in jobs}
    if (len(expected) != 8 or set(requests) != set(expected) or len(set(expected.values())) != 2
            or any(getattr(requests[key], 'request_id', None) != key for key in expected)
            or type(output_width) is not int or not 1 <= output_width <= 50257
            or any(list(expected.values()).count(team) != 4 for team in set(expected.values()))
            or type(atol) is not float or type(rtol) is not float or atol != 1e-6 or rtol != 1e-5
            or type(broker._max_coalesce_wait) is not float
            or not math.isfinite(broker._max_coalesce_wait) or broker._max_coalesce_wait < 0):
        raise ValueError('workload, tolerance or fallback interval is invalid')
    if type(custody) is not dict or type(custody.get('windows')) is not dict:
        raise ValueError('caller must provide rooted window custody')
    jobs, requests = copy.deepcopy(jobs), dict(requests)
    if type(owner) is not GroupedWorkspaceWorker or type(broker) is not GroupedBroker:
        raise RuntimeError('exact same resident owner and grouped broker are required')
    resident, authority, budget = owner._worker, owner._authority, owner._budget
    resident_lease, workspace_lease = owner._resident_lease, owner._workspace_lease
    coalesce_wait = broker._max_coalesce_wait

    def check_owner_binding():
        if (broker._worker._worker is not owner or broker._authority is not authority
                or authority._worker is not owner or authority._broker is not broker
                or owner._worker is not resident or owner._model is not model or resident._model is not model
                or resident._owner_completion is not authority or owner._authority is not authority
                or authority._device_guard is not owner._device_guard or owner._budget is not budget
                or owner._resident_lease is not resident_lease or owner._workspace_lease is not workspace_lease
                or broker._max_coalesce_wait != coalesce_wait
                or resident_lease is workspace_lease or resident_lease.released or workspace_lease.released
                or set(budget.active_leases()) != {resident_lease, workspace_lease}
                or owner.failed or owner.poisoned or broker._closed):
            raise RuntimeError('same resident owner, authority or lifetime lease binding changed')
        if resident.device.type != 'cpu' and output_width != 50257:
            raise RuntimeError('device owner requires the fixed output width')

    def check_workload():
        for job in jobs:
            request = requests[job['job_id']]
            if (type(request) is not ForwardRequest or type(job.get('seed')) is not int
                    or request.seed != job['seed'] or request.seed != owner._seed
                    or type(request.inputs) is not dict):
                raise RuntimeError('fixed workload request or seed differs')
            digest = canonical_digest(request.inputs)
            expected_request = canonical_digest({'job_id': job['job_id'], 'team_id': job['team_id'],
                                                  'seed': job['seed'], 'input_sha256': digest})
            if job.get('input_sha256') != digest or job.get('request_sha256') != expected_request:
                raise RuntimeError('fixed workload input or request digest differs')

    stamps = SourceStamps(model, broker=broker)
    custody['source_stamps'], custody['writer'], custody['memory'] = stamps, writer, memory
    custody['published_rows'] = {'serial': [], 'grouped': []}
    outputs_by_mode, reports = {}, {}
    mode = 'serial'
    last_window, clocks = None, None
    try:
        clocks = _clock_metadata()
        for mode in ('serial', 'grouped'):
            last_window = None
            check_owner_binding()
            check_workload()
            before = memory.begin(mode)
            with stamps.capture(mode):
                last_window, outputs = _window(mode, owner, broker, model, jobs, requests, custody,
                    guarded_call=guarded_call, stamps=stamps, synchronize=synchronize)
            source = stamps.snapshot()['windows'][mode]
            check_owner_binding()
            check_workload()
            grouped_rows, queue_ns, late_count = _validate_window(
                mode, last_window, source, jobs, coalesce_wait)
            if set(outputs) != set(expected):
                raise RuntimeError('completed output set differs from the fixed workload')
            for job in jobs:
                job_id = job['job_id']
                group_id, row_index = (None, 0) if mode == 'serial' else grouped_rows[job_id]
                saved = writer.append(mode=mode, team_id=job['team_id'], job_id=job_id,
                                      group_id=group_id, row_index=row_index, row=outputs[job_id])
                custody['published_rows'][mode].append(copy.deepcopy(saved))
                row_event = next(row for row in last_window['rows'] if row['job_id'] == job_id)
                if (type(saved) is not dict or saved.get('mode') != mode or saved.get('team_id') != job['team_id']
                        or saved.get('job_id') != job_id or saved.get('group_id') != group_id
                        or type(saved.get('row_index')) is not int or saved['row_index'] != row_index
                        or saved.get('shape') != [1, 4, output_width] or saved.get('dtype') != 'float32-le'
                        or type(saved.get('bytes')) is not int or saved['bytes'] != 16 * output_width
                        or type(saved.get('sha256')) is not str
                        or re.fullmatch(r'[0-9a-f]{64}', saved['sha256']) is None
                        or _saved_row_sha256(outputs[job_id]) != saved['sha256']
                        or canonical_digest(outputs[job_id]) != row_event['output_sha256']):
                    raise RuntimeError('saved output bytes differ from the delivered row')
            after = memory.finish(mode, started_ns=last_window['start_ns'], ended_ns=last_window['end_ns'])
            check_owner_binding()
            if (type(before) is not dict or type(after) is not dict
                    or any(type(point.get(key)) is not int for point in (before, after)
                           for key in ('snapshot_started_ns', 'snapshot_finished_ns'))
                    or not (before['snapshot_started_ns'] <= before['snapshot_finished_ns']
                            <= last_window['start_ns'] <= last_window['end_ns']
                            <= after['snapshot_started_ns'] <= after['snapshot_finished_ns'])):
                raise RuntimeError('memory observation timestamps do not bracket the window')
            last_window['end_to_memory_snapshot_start_ns'] = after['snapshot_started_ns'] - last_window['end_ns']
            last_window['end_to_memory_snapshot_finish_ns'] = after['snapshot_finished_ns'] - last_window['end_ns']
            if not all(_valid_output(outputs[key], (1, 4, output_width), 'float32') for key in expected):
                raise RuntimeError('output is not a finite detached CPU float32 row')
            outputs_by_mode[mode], reports[mode] = outputs, last_window
            record(mode + '-window', copy.deepcopy({'schema': 'grouped-throughput-window-v1', 'mode': mode,
                   'window': last_window, 'source_stamps': source, 'memory_before': before,
                   'memory_after': after, 'clocks': clocks}))
        manifest, manifest_sha256 = writer.finish()
        if manifest.get('row_set_complete') is not True:
            raise RuntimeError('retained row manifest is incomplete')
        serial, grouped = outputs_by_mode['serial'], outputs_by_mode['grouped']

        def check_saved_rows():
            for checked_mode, values in outputs_by_mode.items():
                events = {row['job_id']: row for row in reports[checked_mode]['rows']}
                saved_rows = {row['job_id']: row for row in custody['published_rows'][checked_mode]}
                if set(saved_rows) != set(expected):
                    raise RuntimeError('saved output bytes do not cover the workload')
                for key in expected:
                    if (_saved_row_sha256(values[key]) != saved_rows[key]['sha256']
                            or canonical_digest(values[key]) != events[key]['output_sha256']):
                        raise RuntimeError('saved output bytes changed before or during comparison')

        check_saved_rows()
        if not all(_comparison(serial[key], grouped[key], atol, rtol)['status'] == 'matched'
                   for key in expected):
            raise RuntimeError('paired output comparison failed')
        claims = stamps.snapshot()['windows']['grouped']['claims']
        if not all(_comparison(serial[claim['members'][0]['request_id']],
                               grouped[claim['members'][1]['request_id']], atol, rtol)['status'] == 'mismatch'
                   and _comparison(serial[claim['members'][1]['request_id']],
                                   grouped[claim['members'][0]['request_id']], atol, rtol)['status'] == 'mismatch'
                   for claim in claims):
            raise RuntimeError('paired routing comparison is indistinguishable')
        check_saved_rows()
        result = {'schema': 'grouped-throughput-pair-v1', 'status': 'complete',
                  'classification': 'owner_observations_pending_supervisor_and_audit',
                  'relay_admission_policy': _GROUPED_RELAY_POLICY,
                  'windows': reports, 'row_manifest_sha256': manifest_sha256,
                  'clocks': clocks,
                  'workload': jobs, 'coalescing_fallback_seconds_hex': coalesce_wait.hex(),
                  'client_wait_limit_seconds': _CLIENT_WAIT_SECONDS,
                  'dispatch_loop_deadline_seconds': _WINDOW_DEADLINE_SECONDS,
                  'grouped_ready_to_model_ns': queue_ns, 'late_peer_count': late_count,
                  'absolute_tolerance': atol, 'relative_tolerance': rtol, 'output_width': output_width,
                  'grouped_over_serial_completed_rate': (
                      reports['grouped']['completed'] * reports['serial']['wall_duration_ns'] /
                      (reports['serial']['completed'] * reports['grouped']['wall_duration_ns'])),
                  'all_outputs_matched': True, 'each_group_routing_distinguishable': True,
                  'limitations': ['closed-loop fixed workload; single controls bypass the serial broker',
                                  'source profiling, relay and guarded operations contribute to timing',
                                  'ready-to-model includes selection and input staging',
                                  'coalescing wait is a dispatcher-observed fallback, not a strict deadline',
                                  'a timely relay peer can become broker-ready after the nominal fallback',
                                  'device and host isolation, owner exit and cleanup require separate evidence']}
        record('pair', copy.deepcopy(result))
        return result
    except BaseException as exc:
        custody['failure'] = exc
        partial = getattr(exc, 'partial_rehearsal', None)
        if partial is None and last_window is not None:
            partial = last_window
        try:
            record('failure', {'schema': 'grouped-throughput-failure-v1', 'status': 'incomplete',
                   'mode': mode, 'error_type': type(exc).__name__, 'window': partial,
                   'clocks': clocks,
                   'source_stamps': stamps.snapshot(),
                   'published_rows': copy.deepcopy(custody['published_rows'])})
        except BaseException as recording_error:
            custody['failure_record_error'] = recording_error
            if not isinstance(recording_error, Exception):
                raise recording_error from exc
        raise
