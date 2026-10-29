"""CPU rehearsal core for a fixed two-client serial-service comparison.

This module never starts an owner or performs CUDA operations. Its present
memory provider is restricted to explicitly labelled synthetic CPU evidence;
there is no live GPU entry until source-specific admission and owner-side
allocator observation have been independently reviewed. The caller retains
the service and this run through supervisor and marker reconciliation.
"""
import hashlib
import json
from math import ceil, isfinite
from statistics import median
import threading
import time

from looking_glass.direct_concurrent_clients import _times
from looking_glass.ipc import IPCRejected, _IDENTIFIER
from looking_glass.two_team_service_smoke import _REQUEST_ID, _result_identity


_TEAMS = 2
_JOBS_PER_TEAM = 4
_SEED = 29
_INPUT_POSITIONS = 2
_MEMORY_FIELDS = {'device_total_bytes', 'device_free_bytes', 'process_allocated_bytes',
                  'process_reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'}


def _validate_memory(value):
    """Validate a provider's point counters; the provider establishes their origin."""
    if type(value) is not dict or set(value) != _MEMORY_FIELDS:
        raise ValueError('memory observation fields are incomplete')
    if any(type(item) is not int or item < 0 for item in value.values()):
        raise ValueError('memory observations must be nonnegative integers')
    total, free = value['device_total_bytes'], value['device_free_bytes']
    allocated, reserved = value['process_allocated_bytes'], value['process_reserved_bytes']
    peak_allocated, peak_reserved = value['peak_allocated_bytes'], value['peak_reserved_bytes']
    if (total <= 0 or free > total or allocated > reserved or reserved > total
            or peak_allocated < allocated or peak_reserved < reserved
            or peak_allocated > peak_reserved or peak_reserved > total):
        raise ValueError('memory observation counters are inconsistent')
    return dict(value)


def _distribution(values):
    if not values:
        return {'n': 0, 'p50': None, 'p95': None, 'max': None}
    ordered = sorted(values)
    return {'n': len(ordered), 'p50': median(ordered),
            'p95': ordered[ceil(.95 * len(ordered)) - 1], 'max': ordered[-1]}


class _Stream:
    def __init__(self, team, client):
        self.team, self.client = team, client
        self.rows, self.handles, self.failures = [], [], []
        self.thread = None
        self.finished = False


class FixedSharedLoad:
    """One retained comparison core, without owner startup or teardown claims.

    The current constructor only accepts a synthetic CPU memory provider.
    ``client_factory`` must address one already-owned, CPU-only test service.
    Both conditions rely on the trusted caller: the label and callable are
    not authenticated and cannot prevent wrapping an existing live service.
    The caller must retain this object, all client handles and the owner while
    work is unresolved; a report does not attest supervisor or marker state.
    """
    def __init__(self, client_factory, *, teams, memory_probe, timeout=60):
        if (not callable(client_factory) or type(teams) is not tuple or len(teams) != _TEAMS
                or any(type(team) is not str or len(team) > 48 or _IDENTIFIER.fullmatch(team) is None
                       for team in teams)
                or teams[0] == teams[1]):
            raise ValueError('exactly two distinct bounded teams are required')
        if type(timeout) not in (int, float) or not isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError('timeout must be finite and at most 60 seconds')
        if getattr(memory_probe, 'source', None) != 'synthetic_cpu':
            raise RuntimeError('live owner and GPU memory instrumentation are not enabled')
        if not callable(getattr(memory_probe, 'begin_window', None)) or not callable(
                getattr(memory_probe, 'end_window', None)):
            raise TypeError('a bounded synthetic memory observer is required')
        self._factory, self._teams, self._probe, self._timeout = client_factory, teams, memory_probe, timeout
        self._streams, self._threads, self._gates, self._failures = [], [], [], []
        self._claimed = False
        self._stop = threading.Event()

    @property
    def threads(self):
        return tuple(self._threads)

    @property
    def failure_count(self):
        return len(self._failures) + sum(len(stream.failures) for stream in self._streams)

    def _raise_interruption(self):
        for error in self._failures:
            if not isinstance(error, Exception):
                raise error
        for stream in self._streams:
            for error in stream.failures:
                if not isinstance(error, Exception):
                    raise error

    def drain_clients(self, *, timeout=30):
        """Stop new submissions and join clients; this does not attest the owner."""
        if type(timeout) not in (int, float) or not isfinite(timeout) or timeout < 0:
            raise ValueError('invalid client drain timeout')
        self._stop_clients()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    def _stop_clients(self):
        # Gates are rooted before their threads start. In particular, a parent
        # interruption before the initial release must not strand a client.
        # Repeating each signal preserves the bounded one-interruption case.
        try:
            try:
                self._stop.set()
            finally:
                self._stop.set()
        finally:
            for gate in self._gates:
                try:
                    gate.set()
                finally:
                    gate.set()

    def _attempt(self, stream, ordinal):
        attempt = time.perf_counter_ns()
        row = {'team': stream.team, 'ordinal': ordinal, 'seed': _SEED,
               'input_positions': _INPUT_POSITIONS, 'attempt_ns': attempt,
               'returned_ns': None, 'state': 'unresolved', 'error': None,
               'broker_id': None, 'job_id': None, 'submitted_ns': None,
               'ready_ns': None, 'started_ns': None, 'finished_ns': None,
               'float32_sha256': None}
        handle = response = None
        try:
            handle = stream.client.submit(team_id=stream.team, request_id=_REQUEST_ID,
                                          inputs={'input_ids': [[15496, 995]]}, seed=_SEED)
            stream.handles.append(handle)
            row['broker_id'], row['job_id'] = handle.broker_id, handle.job_id
            response = handle.wait(timeout=self._timeout)
            status = response.get('status') if type(response) is dict else None
            if status == 'timeout':
                row['error'] = 'timeout'
            elif status in ('failed', 'cancelled'):
                row['submitted_ns'], row['ready_ns'], row['started_ns'], row['finished_ns'] = _times(response)
                row['state'], row['error'] = status, 'owner_failure' if status == 'failed' else 'cancelled'
            elif status == 'succeeded':
                row['submitted_ns'], row['ready_ns'], row['started_ns'], row['finished_ns'] = _times(response)
                broker, job, digest = _result_identity(response, stream.team)
                if broker != handle.broker_id or job != handle.job_id or row['started_ns'] is None:
                    raise ValueError('result identity differs from its handle')
                row['state'], row['float32_sha256'] = 'succeeded', digest
            else:
                raise ValueError('invalid terminal status')
        except BaseException as exc:
            stream.failures.append(exc)
            row['state'] = 'rejected' if type(exc) is IPCRejected else 'unresolved'
            row['error'] = 'rejected' if type(exc) is IPCRejected else 'client_failure'
        finally:
            response = None
            if handle is not None:
                try:
                    handle.close()
                except BaseException as exc:
                    stream.failures.append(exc)
                    row['state'], row['error'] = 'unresolved', 'close_failure'
            row['returned_ns'] = time.perf_counter_ns()
            stream.rows.append(row)
            if row['state'] != 'succeeded':
                self._stop.set()

    def _serve(self, streams, count, go):
        try:
            go.wait()
            for stream in streams:
                for ordinal in range(count):
                    if self._stop.is_set():
                        return
                    self._attempt(stream, ordinal)
        except BaseException as exc:
            self._failures.append(exc)
            self._stop.set()
        finally:
            for stream in streams:
                stream.finished = True

    def _window(self, mode, count, *, measured):
        streams = []
        for team in self._teams:
            stream = _Stream(team, self._factory(team))
            self._streams.append(stream)
            streams.append(stream)
        before_begin = time.perf_counter_ns() if measured else None
        before = _validate_memory(self._probe.begin_window(mode)) if measured else None
        before_end = time.perf_counter_ns() if measured else None
        go = threading.Event()
        self._gates.append(go)
        groups = [streams] if mode != 'shared' else [[stream] for stream in streams]
        threads = [threading.Thread(target=self._serve, args=(group, count, go),
                                    name='serial-load-client') for group in groups]
        # Root every possible thread before start, including a start that
        # raises after actually launching it.
        self._threads.extend(threads)
        try:
            for thread in threads:
                thread.start()
        except BaseException as exc:
            self._failures.append(exc)
            self._stop.set()
            go.set()
            self.drain_clients(timeout=self._timeout)
            if not isinstance(exc, Exception):
                raise
            return None
        start = time.perf_counter_ns()
        go.set()
        deadline = time.monotonic() + self._timeout
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        end = time.perf_counter_ns()
        if any(thread.is_alive() for thread in threads) or any(not stream.finished for stream in streams):
            self._stop.set()
            return None
        rows = [row for stream in streams for row in stream.rows]
        if not measured:
            return {'rows': rows, 'complete': len(rows) == len(self._teams) * count
                    and all(row['state'] == 'succeeded' for row in rows)}
        after_begin = time.perf_counter_ns()
        after = _validate_memory(self._probe.end_window(mode))
        after_end = time.perf_counter_ns()
        seconds = (end - start) / 1_000_000_000
        if seconds <= 0:
            raise ValueError('invalid measurement window')
        successes = [row for row in rows if row['state'] == 'succeeded']
        queue = [row['started_ns'] - row['ready_ns'] for row in successes]
        service = [row['finished_ns'] - row['started_ns'] for row in successes]
        by_team = {}
        for team in self._teams:
            team_rows = [row for row in rows if row['team'] == team]
            team_successes = [row for row in team_rows if row['state'] == 'succeeded']
            by_team[team] = {'attempted': len(team_rows), 'succeeded': len(team_successes),
                'completed_input_positions': sum(row['input_positions'] for row in team_successes),
                'requests_per_second': len(team_successes) / seconds,
                'queue_wait_ns': _distribution([row['started_ns'] - row['ready_ns'] for row in team_successes]),
                'service_ns': _distribution([row['finished_ns'] - row['started_ns'] for row in team_successes])}
        return {'mode': mode, 'start_ns': start, 'end_ns': end, 'rows': rows,
                'memory': {'source': self._probe.source, 'before': before, 'after': after,
                           'before_call_ns': [before_begin, before_end],
                           'after_call_ns': [after_begin, after_end]},
                'summary': {'attempted': len(rows), 'succeeded': len(successes),
                    'completed_input_positions': sum(row['input_positions'] for row in successes),
                    'requests_per_second': len(successes) / seconds,
                    'input_positions_per_second': sum(row['input_positions'] for row in successes) / seconds,
                    'queue_wait_ns': _distribution(queue), 'service_ns': _distribution(service),
                    'client_response_ns': _distribution([row['returned_ns'] - row['attempt_ns'] for row in rows]),
                    'by_team': by_team}}

    def run(self):
        if self._claimed:
            raise RuntimeError('the fixed comparison is one-shot')
        self._claimed = True
        plan = {'teams': list(self._teams), 'input_ids': [15496, 995], 'input_shape': [1, 2], 'seed': _SEED,
                'jobs_per_team_per_window': _JOBS_PER_TEAM, 'warmup_per_team': 1,
                'mode_order': ['serial', 'shared'], 'trace': 'disabled',
                'unit': 'input positions, not generated tokens'}
        plan_bytes = json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
        clock = time.get_clock_info('perf_counter')
        result = {'status': 'incomplete', 'comparison_valid': False,
                  'requires_owner_reconciliation': True, 'owner_teardown_verified': False,
                  'scope': 'one serial owner; no batching or device overlap asserted',
                  'plan': plan, 'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
                  'clock': {'name': 'perf_counter_ns', 'implementation': clock.implementation,
                            'resolution_seconds': clock.resolution, 'monotonic': clock.monotonic,
                            'adjustable': clock.adjustable},
                  'shared_over_serial_rate_ratio': None,
                  'warmup': None, 'windows': []}
        try:
            warmup = self._window('warmup', 1, measured=False)
            self._raise_interruption()
            result['warmup'] = warmup
            if warmup is None or not warmup['complete']:
                return result
            for mode in ('serial', 'shared'):
                window = self._window(mode, _JOBS_PER_TEAM, measured=True)
                self._raise_interruption()
                if window is None:
                    return result
                result['windows'].append(window)
                if self._stop.is_set() or len(window['rows']) != _TEAMS * _JOBS_PER_TEAM:
                    return result
            left, right = result['windows']
            def outputs(window):
                return {(row['team'], row['ordinal']): row['float32_sha256'] for row in window['rows']}
            rows = warmup['rows'] + left['rows'] + right['rows']
            identities = [(row['broker_id'], row['job_id']) for row in rows]
            result['comparison_valid'] = (all(row['state'] == 'succeeded' for window in result['windows']
                                               for row in window['rows'])
                and len({row['broker_id'] for row in rows}) == 1
                and len(set(identities)) == len(identities) and outputs(left) == outputs(right))
            if result['comparison_valid']:
                result['status'] = 'complete'
                result['requires_owner_reconciliation'] = False
                result['shared_over_serial_rate_ratio'] = (right['summary']['requests_per_second'] /
                                                            left['summary']['requests_per_second'])
            else:
                result['status'] = 'invalid_comparison'
            return result
        except BaseException as exc:
            self._failures.append(exc)
            self._stop_clients()
            if not isinstance(exc, Exception):
                raise
            return result


def run_live_service(*args, **kwargs):
    """Fail closed until the actual GPU service experiment is admitted."""
    raise RuntimeError('live throughput requires reviewed host admission and owner memory observation')
