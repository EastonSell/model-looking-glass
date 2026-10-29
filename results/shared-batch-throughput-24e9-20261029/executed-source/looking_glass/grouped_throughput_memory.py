"""Prospective window telemetry using the grouped development owner's guards.

This helper does not acquire an owner, approve a profile, launch work, attest
cleanup or authenticate its caller's window timestamps. It is intended only
for a separately reviewed and source-bound grouped owner, on its single device
driver thread. Importing this module makes no device query.
"""

from pathlib import Path
from time import perf_counter_ns

from . import grouped_gpu_dev as _dev


class WindowMemory:
    """Record two reset allocator windows around externally observed work.

    The caller must hold the exact supervised session and lease throughout,
    stop and join all work before ``finish``, and retain this helper and its
    owner on any failure. The existing grouped owner's `_cuda` guard decides
    device safety; this wrapper cannot make asynchronous revocation atomic.
    Each point is persisted before subsequent consistency checks can reject it.
    """

    def __init__(self, run_dir, session, lease, torch, expected_total, protocol):
        if type(expected_total) is not int or expected_total <= 0:
            raise ValueError('expected device total must be a positive integer')
        names = ('minimum_reserve_bytes', 'resident_commitment_bytes', 'forward_commitment_bytes')
        if type(protocol) is not dict or any(type(protocol.get(name)) is not int or protocol[name] < 0
                                             for name in names):
            raise ValueError('fixed commitment values are required')
        if not protocol['resident_commitment_bytes'] or not protocol['forward_commitment_bytes']:
            raise ValueError('resident and workspace commitments must be positive')
        self._run_dir = Path(run_dir)
        self._session, self._lease, self._torch = session, lease, torch
        self._expected_total = expected_total
        self._protocol = {name: protocol[name] for name in names}
        self._state, self._next = 'idle', 'serial'
        self._baseline = None

    def _save(self, mode, phase, record):
        _dev._write_json(self._run_dir / f'throughput-memory-{mode}-{phase}.json', record)

    def _fail(self, mode, phase, exc):
        self._state = 'failed'
        try:
            self._save(mode, phase + '-failure', {'schema': 'grouped-throughput-memory-failure-v1',
                       'mode': mode, 'phase': phase, 'error_type': type(exc).__name__,
                       'status': 'incomplete'})
        except BaseException as reporting_error:
            # A new control interruption wins. An ordinary reporting error
            # must not hide the original failure or control interruption.
            if isinstance(reporting_error, Exception):
                raise exc from reporting_error
            raise

    def _point(self, mode, phase):
        started = perf_counter_ns()
        partial_error = None

        def partial(values):
            nonlocal partial_error
            try:
                bounded = {key: value if type(value) is int else {'invalid_type': type(value).__name__}
                           for key, value in values.items()}
                self._save(mode, phase + '-partial', {'schema': 'grouped-throughput-memory-partial-v1',
                           'mode': mode, 'phase': phase, 'snapshot_started_ns': started,
                           'counters': bounded, 'status': 'incomplete'})
            except BaseException as exc:
                partial_error = exc
                raise

        try:
            counters = _dev._snapshot(self._session, self._lease, self._torch,
                                      self._expected_total, partial)
        except BaseException as exc:
            # The existing snapshot invokes `partial` inside its exception
            # handler. A failed write can replace an original interruption.
            original = exc.__context__
            if exc is partial_error and isinstance(exc, Exception) and original is not None:
                raise original from exc
            raise
        finished = perf_counter_ns()
        record = {'schema': 'grouped-throughput-memory-point-v1', 'mode': mode, 'phase': phase,
                  'snapshot_started_ns': started, 'snapshot_finished_ns': finished,
                  'counters': counters}
        self._save(mode, phase, record)
        _dev._validate_memory_values(counters)
        _dev._check_observed_headroom(counters, self._protocol)
        return record

    def begin(self, mode):
        if self._state != 'idle' or mode != self._next:
            raise RuntimeError('memory window is out of order or already failed')
        self._state = 'beginning'
        try:
            _dev._cuda(self._session, self._lease, self._torch.cuda.synchronize, 0)
            _dev._cuda(self._session, self._lease, self._torch.cuda.reset_peak_memory_stats, 0)
            before = self._point(mode, 'before')
            _dev._check_peak_reset(before['counters'])
            self._baseline = before
            self._state = 'active'
            return before
        except BaseException as exc:
            self._fail(mode, 'before', exc)
            raise

    def finish(self, mode, *, started_ns, ended_ns):
        if self._state != 'active' or mode != self._next:
            raise RuntimeError('memory window is not active')
        self._state = 'finishing'
        try:
            if (type(started_ns) is not int or type(ended_ns) is not int
                    or not self._baseline['snapshot_finished_ns'] <= started_ns <= ended_ns <= perf_counter_ns()):
                raise ValueError('observed window timestamps are out of order')
            after = self._point(mode, 'after')
            if ended_ns > after['snapshot_started_ns']:
                raise RuntimeError('endpoint precedes the recorded window end')
            _dev._check_workspace_peak(self._baseline['counters'], after['counters'], self._protocol)
            self._baseline = None
            self._next = 'grouped' if mode == 'serial' else None
            self._state = 'idle' if self._next else 'finished'
            return after
        except BaseException as exc:
            self._fail(mode, 'after', exc)
            raise
