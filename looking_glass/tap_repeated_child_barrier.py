"""Private child-side barriers for the disabled repeated-tap owner.

These checks perform no CUDA work. The supervised owner must still validate its
lease, actual device, admission and source before using them; same-user files
and correlation fields are cooperative coordination, not authentication.
"""

import json
import math
import os
import sys
import time

from .supervisor import _supervisor_birth
from .tap_repeated_parent import (_check_preflight_interval, _clock_margin, _decode_mapping,
                                  _decode_ready, _publish_barrier, _read_plain_frame)
from .tap_repeated_protocol import CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES, START_GRANT_MAX_NS


_IDENTITY = ('session_index', 'owner_pid', 'owner_birth', 'supervisor_run_id', 'challenge',
             'gpu_uuid', 'cuda_total_bytes', 'activity_smi_total_bytes', 'clock', 'runtime')


def wait_frame(path, *, timeout, clock=time.perf_counter_ns, sleep=time.sleep):
    """Read one bounded barrier, retrying only absent or Windows sharing errors."""
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30
            or not callable(clock) or not callable(sleep)):
        raise ValueError('invalid private barrier wait')
    started = clock()
    if type(started) is not int or not 0 <= started < 2 ** 63:
        raise ValueError('invalid barrier clock')
    deadline = started + math.ceil(timeout * 1_000_000_000)
    if deadline >= 2 ** 63:
        raise ValueError('barrier deadline is outside the clock range')
    previous = started
    for _ in range(math.ceil(timeout / 0.01) + 2):
        now = clock()
        if type(now) is not int or now < previous:
            raise RuntimeError('barrier clock moved backwards')
        previous = now
        if now >= deadline:
            raise TimeoutError('private barrier did not arrive')
        try:
            raw = _read_plain_frame(path)
        except OSError as error:
            if not isinstance(error, FileNotFoundError) and getattr(error, 'winerror', None) not in (32, 33):
                raise
            sleep(min(0.01, (deadline - now) / 1_000_000_000))
            continue
        observed = clock()
        if type(observed) is not int or not now <= observed < deadline:
            raise TimeoutError('private barrier read exceeded its deadline')
        return raw, observed
    raise TimeoutError('private barrier did not arrive')


class ChildBarriers:
    """Validate one start grant, one preflight interval and one cleanup grant."""

    def __init__(self, ready):
        if type(ready) is not dict:
            raise ValueError('invalid child ready identity')
        raw = json.dumps(ready, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        snapshot = _decode_ready(raw, session_index=ready.get('session_index'), challenge=ready.get('challenge'))
        info = time.get_clock_info('perf_counter')
        actual_clock = {'name': 'perf_counter_ns', 'monotonic': info.monotonic,
                        'adjustable': info.adjustable, 'implementation': info.implementation,
                        'resolution_seconds': info.resolution}
        actual_runtime = {'platform': sys.platform, 'implementation': sys.implementation.name,
                          'python_version': '.'.join(map(str, sys.version_info[:3]))}
        if (snapshot['owner_pid'] != os.getpid() or snapshot['owner_birth'] != _supervisor_birth()
                or snapshot['clock'] != actual_clock or snapshot['runtime'] != actual_runtime):
            raise ValueError('child process or runtime identity differs')
        self._ready, self._grant = snapshot, None
        self._state = 'awaiting_start'
        self._seen = self._lower = self._upper = self._ended = None

    @property
    def cleanup_authorized(self):
        return self._state == 'cleanup_authorized'

    def accept_start(self, raw, *, observed_ns):
        if self._state != 'awaiting_start':
            raise RuntimeError('start grant is not available')
        try:
            grant = _decode_mapping(raw)
            keys = {'schema', *_IDENTITY, 'ready_ns', 'received_ns', 'issued_ns', 'expires_ns',
                    'initial_oldest_query_started_ns', 'initial_latest_query_ended_ns',
                    'initial_process_started_ns', 'initial_process_ended_ns'}
            if (set(grant) != keys or grant['schema'] != 'tap-repeated-start-grant-v1'
                    or any(type(grant[key]) is not type(self._ready[key]) or grant[key] != self._ready[key]
                           for key in _IDENTITY if key != 'activity_smi_total_bytes')
                    or grant['ready_ns'] != self._ready['ready_ns']
                    or type(grant['activity_smi_total_bytes']) is not int
                    or grant['activity_smi_total_bytes'] <= 0
                    or abs(grant['cuda_total_bytes'] - grant['activity_smi_total_bytes'])
                       > CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES):
                raise ValueError('start grant identity or device totals differ')
            margin = _clock_margin(grant['runtime'], grant['clock'])
            ready, received, issued, expires = (grant[key] for key in ('ready_ns', 'received_ns', 'issued_ns', 'expires_ns'))
            oldest, latest, process_start, process_end = (grant[key] for key in
                ('initial_oldest_query_started_ns', 'initial_latest_query_ended_ns',
                 'initial_process_started_ns', 'initial_process_ended_ns'))
            if (any(type(value) is not int for value in
                    (ready, received, issued, expires, oldest, latest, process_start, process_end))
                    or not ready + margin <= received <= oldest <= latest < process_start <= process_end <= issued
                    or not issued < expires <= issued + START_GRANT_MAX_NS):
                raise ValueError('start grant ordering differs')
            _check_preflight_interval(grant, seen_ns=observed_ns, lower_ns=observed_ns, upper_ns=observed_ns)
        except BaseException:
            self._state = 'failed'
            raise
        self._grant, self._seen = grant, observed_ns
        self._state = 'start_received'

    def authorize_preflight(self, lower_ns, upper_ns):
        """Consume one non-I/O authorization for a bounded preflight interval."""
        if self._state != 'start_received':
            raise RuntimeError('preflight grant is not available')
        try:
            _check_preflight_interval(self._grant, seen_ns=self._seen, lower_ns=lower_ns, upper_ns=upper_ns)
        except BaseException:
            self._state = 'failed'
            raise
        self._lower, self._upper = lower_ns, upper_ns
        self._state = 'preflight_authorized'

    def publish_finished(self, path, *, preflight_started_ns, workload_ended_ns):
        """Publish the source-stamped end after the closed owner journal."""
        if self._state != 'preflight_authorized':
            raise RuntimeError('finished barrier is not available')
        try:
            _check_preflight_interval(self._grant, seen_ns=self._seen, lower_ns=self._lower,
                                      upper_ns=self._upper, started_ns=preflight_started_ns)
            if (type(workload_ended_ns) is not int
                    or not preflight_started_ns <= workload_ended_ns < 2 ** 63):
                raise ValueError('workload end is invalid')
            finish = {'schema': 'tap-repeated-workload-finished-v1',
                      **{key: self._grant[key] for key in _IDENTITY},
                      'grant_seen_ns': self._seen, 'preflight_started_ns': preflight_started_ns,
                      'workload_ended_ns': workload_ended_ns}
            # A visible file may have been consumed even if publication is
            # interrupted after its rename. This object is then unusable.
            self._state = 'publication_uncertain'
            _publish_barrier(path, finish)
        except BaseException:
            self._state = 'failed'
            raise
        self._ended = workload_ended_ns
        self._state = 'awaiting_cleanup'

    def accept_cleanup(self, raw, *, observed_ns):
        if self._state != 'awaiting_cleanup':
            raise RuntimeError('cleanup grant is not available')
        try:
            cleanup = _decode_mapping(raw)
            keys = {'schema', *_IDENTITY, 'workload_ended_ns', 'issued_ns', 'expires_ns', 'screen_availability'}
            if (set(cleanup) != keys or cleanup['schema'] != 'tap-repeated-cleanup-grant-v1'
                    or any(type(cleanup[key]) is not type(self._grant[key]) or cleanup[key] != self._grant[key]
                           for key in _IDENTITY)
                    or type(cleanup['workload_ended_ns']) is not int or cleanup['workload_ended_ns'] != self._ended
                    or cleanup['screen_availability'] not in ('complete', 'unverified')):
                raise ValueError('cleanup grant identity differs')
            margin = _clock_margin(cleanup['runtime'], cleanup['clock'])
            issued, expires = cleanup['issued_ns'], cleanup['expires_ns']
            if (type(issued) is not int or type(expires) is not int or type(observed_ns) is not int
                    or not self._ended + margin <= issued < expires <= issued + 5_000_000_000
                    or not issued + margin <= observed_ns <= expires < 2 ** 63):
                raise ValueError('cleanup grant is stale or misordered')
        except BaseException:
            self._state = 'failed'
            raise
        self._state = 'cleanup_authorized'
