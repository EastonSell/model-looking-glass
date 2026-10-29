"""Disabled parent identity and barriers for the repeated tap experiment.

This draft does not launch an owner. The final integration must bind the live
handle and barriers to strict supervision, fresh parent telemetry, durable
partial evidence and an independently reviewed owner before enabling it.
The private cooperative frame and files are not a hostile-process boundary.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading

from .device_identity import DeviceIdentity, normalize_gpu_uuid
from .supervisor import SupervisionResult, _identity_spec, _marker, inspect_live_child, run_supervised
from .tap_repeated_protocol import (CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES, EXPERIMENT_PROTOCOL,
                                    check_initial_activity, classify_activity, clock_minimum,
                                    summarize_session)


_PARENT_LAUNCH_ENABLED = False
_MAX_FRAME = 4096
_MAX_JOURNAL = 8 * 1024 * 1024
_MAX_JOURNAL_LINE = 65536
_PREFLIGHT_GATE_MAX_NS = 250_000_000
_READY_SCHEMA = 'tap-repeated-owner-ready-v1'
_SUPPORTED_RUNTIME = {'platform': 'win32', 'implementation': 'cpython', 'python_version': '3.11.15'}
_poller_lock = threading.Lock()
_pollers = set()
_supervision_lock = threading.Lock()
_supervisions = set()
_attempt_lock = threading.Lock()
_attempts = set()


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate startup field')
        result[key] = value
    return result


def _bad_constant(_value):
    raise ValueError('non-finite startup field')


def _decode_mapping(raw, *, max_bytes=_MAX_FRAME):
    if type(raw) is not bytes or not 1 <= len(raw) <= max_bytes:
        raise ValueError('invalid private frame size')
    try:
        row = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_mapping,
                         parse_constant=_bad_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid private frame') from exc
    if type(row) is not dict:
        raise ValueError('private frame must be a mapping')
    return row


def _decode_ready(raw, *, session_index, challenge):
    """Decode exactly one private frame and bind its slot and challenge."""
    if (type(raw) is not bytes or not 1 <= len(raw) <= _MAX_FRAME
            or type(session_index) is not int or session_index not in range(5)
            or type(challenge) is not str or re.fullmatch('[0-9a-f]{64}', challenge) is None):
        raise ValueError('invalid startup frame parameters')
    row = _decode_mapping(raw)
    keys = {'schema', 'session_index', 'owner_pid', 'owner_birth', 'supervisor_run_id',
            'challenge', 'gpu_uuid', 'cuda_total_bytes', 'ready_ns', 'clock', 'runtime'}
    if (type(row) is not dict or set(row) != keys or row['schema'] != _READY_SCHEMA
            or type(row['session_index']) is not int or row['session_index'] != session_index
            or row['challenge'] != challenge or type(row['owner_pid']) is not int
            or not 0 < row['owner_pid'] < 2 ** 32
            or type(row['owner_birth']) is not str
            or re.fullmatch('[1-9][0-9]{0,19}', row['owner_birth']) is None
            or int(row['owner_birth']) >= 2 ** 64
            or type(row['supervisor_run_id']) is not str
            or re.fullmatch('[0-9a-f]{32}', row['supervisor_run_id']) is None
            or type(row['gpu_uuid']) is not str
            or type(row['cuda_total_bytes']) is not int or not 0 < row['cuda_total_bytes'] < 2 ** 63
            or type(row['ready_ns']) is not int or not 0 <= row['ready_ns'] < 2 ** 63):
        raise ValueError('startup identity or frame fields differ')
    if normalize_gpu_uuid(row['gpu_uuid']) != row['gpu_uuid']:
        raise ValueError('startup GPU UUID is not canonical')
    _clock_margin(row['runtime'], row['clock'])
    return row


def _clock_margin(runtime, clock):
    """Require the pinned Windows CPython system-wide QPC time base."""
    clock_minimum(clock)
    if (type(runtime) is not dict or runtime != _SUPPORTED_RUNTIME
            or clock.get('implementation') != 'QueryPerformanceCounter()'):
        raise ValueError('unsupported cross-process clock runtime')
    return 2 * max(1, math.ceil(clock['resolution_seconds'] * 1_000_000_000))


def _check_ready_clock(ready, *, parent_clock, parent_runtime, received_ns):
    """Check the documented system-wide clock and reciprocal ordering."""
    margin = _clock_margin(parent_runtime, parent_clock)
    if (type(received_ns) is not int or not 0 <= received_ns < 2 ** 63
            or type(ready) is not dict or ready.get('clock') != parent_clock
            or ready.get('runtime') != parent_runtime
            or type(ready.get('ready_ns')) is not int
            or not margin <= received_ns - ready['ready_ns'] <= 10_000_000_000):
        raise ValueError('startup clocks or ordering differ')


def _make_start_grant(ready, *, parent_clock, parent_runtime, received_ns, snapshot, gpu_uuid,
                      expected_smi_total, issued_ns):
    """Bind a short-lived grant to the exact three initial query intervals."""
    _check_ready_clock(ready, parent_clock=parent_clock, parent_runtime=parent_runtime, received_ns=received_ns)
    margin = _clock_margin(parent_runtime, parent_clock)
    if (type(gpu_uuid) is not str or normalize_gpu_uuid(gpu_uuid) != gpu_uuid
            or type(ready.get('gpu_uuid')) is not str or ready['gpu_uuid'] != gpu_uuid
            or type(ready.get('cuda_total_bytes')) is not int or ready['cuda_total_bytes'] <= 0
            or type(expected_smi_total) is not int or expected_smi_total <= 0
            or abs(ready['cuda_total_bytes'] - expected_smi_total) > CROSS_PROVIDER_TOTAL_TOLERANCE_BYTES):
        raise ValueError('initial GPU identity or cross-provider total differs')
    if (type(snapshot) is not dict or snapshot.get('complete') is not True
            or snapshot.get('in_flight') is not False or type(snapshot.get('error_count')) is not int
            or snapshot['error_count'] != 0
            or type(snapshot.get('observations')) is not list
            or type(snapshot.get('process_queries')) is not list
            or len(snapshot['observations']) < 3 or len(snapshot['process_queries']) != 1
            or type(issued_ns) is not int or not received_ns <= issued_ns < 2 ** 63):
        raise ValueError('initial activity inventory is incomplete')
    observations = snapshot['observations']
    points = observations[-3:]
    query = snapshot['process_queries'][0]
    check_initial_activity(observations=observations, errors=snapshot.get('errors'),
                           process_query=query, gpu_uuid=gpu_uuid,
                           own_pid=ready['owner_pid'], expected_total_bytes=expected_smi_total,
                           now_ns=issued_ns)
    oldest, process_start = points[0]['query_started_ns'], query['query_started_ns']
    if observations[0]['query_started_ns'] < received_ns or process_start <= points[-1]['query_ended_ns']:
        raise ValueError('initial activity is not ordered after owner startup')
    # The child's final preflight-start check must use these same boundaries,
    # not just the time when the parent created the grant.
    expires = min(issued_ns + 500_000_000, oldest + 2_000_000_000 - margin,
                  process_start + 2_000_000_000 - margin)
    if expires <= issued_ns:
        raise ValueError('initial activity is stale')
    return {'schema': 'tap-repeated-start-grant-v1', 'session_index': ready['session_index'],
            'owner_pid': ready['owner_pid'], 'owner_birth': ready['owner_birth'],
            'supervisor_run_id': ready['supervisor_run_id'], 'challenge': ready['challenge'],
            'gpu_uuid': gpu_uuid, 'cuda_total_bytes': ready['cuda_total_bytes'],
            'activity_smi_total_bytes': expected_smi_total,
            'ready_ns': ready['ready_ns'], 'received_ns': received_ns,
            'issued_ns': issued_ns, 'expires_ns': expires,
            'initial_oldest_query_started_ns': oldest,
            'initial_latest_query_ended_ns': points[-1]['query_ended_ns'],
            'initial_process_started_ns': process_start,
            'initial_process_ended_ns': query['query_ended_ns'], 'clock': dict(parent_clock),
            'runtime': dict(parent_runtime)}


def _initial_screen(activity, *, ready, live, marker_check, persist, grant_path,
                    parent_clock, parent_runtime, received_ns, gpu_uuid, expected_smi_total, clock):
    """Persist fresh observations and grant only after every startup check."""
    for _ in range(3):
        live.check_live()
        marker_check()
        ok = activity.sample_gpu()
        persist(activity.snapshot())
        if ok is not True:
            raise RuntimeError('initial GPU telemetry is unavailable')
    live.check_live()
    marker_check()
    ok = activity.sample_processes('before')
    try:
        # Do not classify a numeric PID as the owner if the held process may
        # have exited during the query and its PID subsequently been reused.
        live.check_live()
        marker_check()
    finally:
        snapshot = activity.snapshot()
        persist(snapshot)
    if ok is not True:
        raise RuntimeError('initial process telemetry is unavailable')
    grant = _make_start_grant(ready, parent_clock=parent_clock, parent_runtime=parent_runtime, received_ns=received_ns,
                              snapshot=snapshot, gpu_uuid=gpu_uuid,
                              expected_smi_total=expected_smi_total, issued_ns=clock())
    live.check_live()
    marker_check()
    _publish_barrier(grant_path, grant)
    return grant


def _decode_finished(raw, *, grant, received_ns):
    """Bind the post-workload barrier to actual preflight and owner identity."""
    row = _decode_mapping(raw)
    identity = ('session_index', 'owner_pid', 'owner_birth', 'supervisor_run_id', 'challenge',
                'gpu_uuid', 'cuda_total_bytes', 'activity_smi_total_bytes', 'clock', 'runtime')
    keys = {'schema', *identity, 'grant_seen_ns', 'preflight_started_ns', 'workload_ended_ns'}
    if (type(grant) is not dict or set(row) != keys
            or row['schema'] != 'tap-repeated-workload-finished-v1'
            or any(type(row[key]) is not type(grant.get(key)) or row[key] != grant[key]
                   for key in identity)
            or type(received_ns) is not int
            or any(type(row[key]) is not int for key in ('grant_seen_ns', 'preflight_started_ns', 'workload_ended_ns'))):
        raise ValueError('workload finish identity or timestamps differ')
    seen, start, end = row['grant_seen_ns'], row['preflight_started_ns'], row['workload_ended_ns']
    margin = _clock_margin(grant['runtime'], grant['clock'])
    if (not 0 <= grant['issued_ns'] + margin <= seen <= start <= grant['expires_ns'] < 2 ** 63
            or not grant['initial_latest_query_ended_ns'] + margin <= start
            or not grant['initial_process_ended_ns'] + margin <= start
            or start - grant['initial_oldest_query_started_ns'] + margin > 2_000_000_000
            or start - grant['initial_process_started_ns'] + margin > 2_000_000_000
            or not start <= end or not end + margin <= received_ns < 2 ** 63):
        raise ValueError('workload finish ordering or initial freshness differs')
    return row


def _check_preflight_interval(grant, *, seen_ns, lower_ns, upper_ns, started_ns=None):
    """Bind a one-use, at-most-250-ms gate to the initial activity grant."""
    if type(grant) is not dict:
        raise ValueError('invalid preflight grant')
    try:
        margin = _clock_margin(grant['runtime'], grant['clock'])
        issued, expires = grant['issued_ns'], grant['expires_ns']
        oldest, latest = grant['initial_oldest_query_started_ns'], grant['initial_latest_query_ended_ns']
        process_start, process_end = grant['initial_process_started_ns'], grant['initial_process_ended_ns']
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid preflight grant') from exc
    values = (seen_ns, lower_ns, upper_ns, issued, expires, oldest, latest, process_start, process_end)
    if (any(type(value) is not int for value in values)
            or not 0 <= issued + margin <= seen_ns <= lower_ns <= upper_ns <= expires < 2 ** 63
            or upper_ns - lower_ns > _PREFLIGHT_GATE_MAX_NS
            or latest + margin > lower_ns or process_end + margin > lower_ns
            or upper_ns - oldest + margin > 2_000_000_000
            or upper_ns - process_start + margin > 2_000_000_000
            or (started_ns is not None and (type(started_ns) is not int
                or not lower_ns <= started_ns <= upper_ns))):
        raise ValueError('preflight gate interval is stale or misordered')


def _read_plain_file(path, *, limit):
    """Read a bounded plain file while checking its observed file identity."""
    path = Path(path)
    before = path.lstat()
    reparse = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    if (not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & reparse
            or before.st_nlink != 1 or not 1 <= before.st_size <= limit):
        raise RuntimeError('private file is not a bounded plain file')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(fd)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if identity(opened) != identity(before):
            raise RuntimeError('private file identity changed')
        chunks, size = [], 0
        while size <= limit:
            part = os.read(fd, min(65536, limit + 1 - size))
            if not part:
                break
            chunks.append(part)
            size += len(part)
        raw = b''.join(chunks)
        after = os.fstat(fd)
        latest = path.lstat()
        if (identity(after) != identity(before) or identity(latest) != identity(before)
                or latest.st_nlink != 1 or not stat.S_ISREG(latest.st_mode)
                or getattr(latest, 'st_file_attributes', 0) & reparse
                or len(raw) != before.st_size):
            raise RuntimeError('private file changed during read')
        return raw
    finally:
        os.close(fd)


def _read_plain_frame(path):
    return _read_plain_file(path, limit=_MAX_FRAME)


def _await_finished(path, *, grant, live, marker_check, supervisor_done, clock, sleep, timeout):
    """Wait for the bounded second frame while retaining the live child handle."""
    if (not Path(path).is_absolute() or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or not 0 < timeout <= 3600
            or not all(callable(item) for item in (marker_check, supervisor_done, clock, sleep))):
        raise ValueError('invalid finished-frame wait')
    started = clock()
    if type(started) is not int or not 0 <= started < 2 ** 63:
        raise RuntimeError('invalid parent clock')
    deadline = started + math.ceil(timeout * 1_000_000_000)
    if deadline >= 2 ** 63:
        raise ValueError('finished-frame deadline is outside the clock range')
    previous = started
    for _ in range(math.ceil(timeout / 0.05) + 2):
        now = clock()
        if type(now) is not int or now < previous:
            raise RuntimeError('parent clock moved backwards')
        previous = now
        if now >= deadline:
            raise TimeoutError('workload finished frame did not arrive')
        live.check_live()
        marker_check()
        if supervisor_done():
            raise RuntimeError('supervisor exited before the finished frame')
        try:
            raw = _read_plain_frame(path)
        except FileNotFoundError:
            sleep(min(0.05, (deadline - now) / 1_000_000_000))
            continue
        received = clock()
        live.check_live()
        marker_check()
        if supervisor_done():
            raise RuntimeError('supervisor exited during the finished-frame read')
        if type(received) is not int or not now <= received < deadline:
            raise TimeoutError('finished-frame read exceeded its deadline')
        return _decode_finished(raw, grant=grant, received_ns=received)
    raise TimeoutError('workload finished frame did not arrive')


def _read_bound_journal(path, *, finish, grant):
    """Bind saved endpoint records; full journal replay still follows child reap.

    The file is trusted to remain quiescent after the owner's finished frame;
    file metadata checks and a byte hash are not an execution attestation.
    """
    if not Path(path).is_absolute() or type(finish) is not dict:
        raise ValueError('invalid private journal binding')
    raw = _read_plain_file(path, limit=_MAX_JOURNAL)
    if not raw.endswith(b'\n'):
        raise ValueError('private journal is truncated')
    lines = raw.splitlines()
    if len(lines) != 86:
        raise ValueError('private journal has missing or extra rows')
    rows = []
    for line in lines:
        row = _decode_mapping(line, max_bytes=_MAX_JOURNAL_LINE)
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if canonical != line:
            raise ValueError('private journal row is not canonical')
        rows.append(row)
    first, initial_post, last, final_post = rows[0], rows[1], rows[-2], rows[-1]
    _check_preflight_interval(grant, seen_ns=finish.get('grant_seen_ns'),
                              lower_ns=first.get('gate_lower_ns'), upper_ns=first.get('gate_upper_ns'),
                              started_ns=first.get('started_ns'))
    memory_start, memory_end = first.get('memory_snapshot_started_ns'), first.get('memory_snapshot_ended_ns')
    if (type(memory_start) is not int or type(memory_end) is not int
            or not 0 <= finish['grant_seen_ns'] <= memory_start <= memory_end <= first['gate_lower_ns']
            or first['started_ns'] - memory_start > 2_000_000_000):
        raise ValueError('preflight memory snapshot is stale or misordered')
    ended, completed = last.get('ended_ns'), final_post.get('completed_ns')
    if (first.get('stage') != 'preflight' or first.get('observation_id') != 'preflight'
            or type(first.get('started_ns')) is not int
            or first['started_ns'] != finish.get('preflight_started_ns')
            or initial_post.get('stage') != 'postcheck' or initial_post.get('observation_id') != 'preflight'
            or initial_post.get('status') != 'passed'
            or last.get('stage') != 'measured' or last.get('observation_id') != 'measured:11:2'
            or final_post.get('stage') != 'postcheck' or final_post.get('observation_id') != 'measured:11:2'
            or final_post.get('status') != 'passed' or final_post.get('device_work_complete') is not True
            or type(ended) is not int or type(completed) is not int
            or not ended <= completed <= ended + 2_000_000_000
            or completed != finish.get('workload_ended_ns')):
        raise ValueError('private journal endpoint anchor differs')
    return rows, hashlib.sha256(raw).hexdigest()


def _recheck_bound_journal(path, *, finish, grant, expected_sha256):
    """Ensure post-reap scoring reads the same bytes checked before cleanup."""
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('invalid pre-cleanup journal digest')
    rows, observed = _read_bound_journal(path, finish=finish, grant=grant)
    if observed != expected_sha256:
        raise ValueError('private journal changed after the cleanup grant')
    return rows


def _check_marker(identity, run_id):
    """Recheck the exact canonical supervised marker while the child is live."""
    if type(identity) is not DeviceIdentity:
        raise RuntimeError('an exact device identity is required')
    DeviceIdentity.validate_paths(identity)
    if os.path.lexists(identity.attestation_path):
        raise RuntimeError('unexpected owner attestation during measurement')
    try:
        observed = _decode_mapping(_read_plain_frame(identity.marker_path))
    except (OSError, ValueError) as exc:
        raise RuntimeError('supervisor marker is unavailable') from exc
    if observed != _marker(run_id, identity.lock_path, _identity_spec(identity)):
        raise RuntimeError('supervisor marker identity differs')
    DeviceIdentity.validate_paths(identity)
    if os.path.lexists(identity.attestation_path):
        raise RuntimeError('unexpected owner attestation during measurement')


def _bind_ready(ready, *, identity):
    """Bind the frame to the registered spawn handle and canonical marker."""
    capability = inspect_live_child(ready['supervisor_run_id'])
    try:
        if (capability.run_id != ready['supervisor_run_id'] or capability.pid != ready['owner_pid']
                or capability.birth != ready['owner_birth']):
            raise RuntimeError('ready frame does not identify the registered child')
        if type(identity) is not DeviceIdentity or ready['gpu_uuid'] != identity.gpu_uuid:
            raise RuntimeError('ready frame does not identify the supervised device')
        capability.check_live()
        _check_marker(identity, ready['supervisor_run_id'])
        capability.check_live()
        return capability
    except BaseException:
        capability.close()
        raise


def _check_supervision_result(result, *, ready, live):
    """Require the same registered child to have been reaped cleanly."""
    if (type(result) is not SupervisionResult or type(ready) is not dict
            or result.run_id != ready.get('supervisor_run_id')
            or type(result.pid) is not int or result.pid != ready.get('owner_pid')
            or type(result.exit_code) is not int or result.exit_code != 0
            or result.clean is not True or result.timed_out is not False
            or result.reason != 'clean device teardown'
            or live.run_id != ready.get('supervisor_run_id') or live.pid != ready.get('owner_pid')
            or live.birth != ready.get('owner_birth')):
        raise RuntimeError('supervision did not cleanly reap the bound owner')


class _PrivateJournal:
    """Create-only, fsynced partial telemetry with a sticky failed-write latch."""

    def __init__(self, path):
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('an absolute private journal path is required')
        self._path = path
        self._lock = threading.Lock()
        self._unsafe = False
        self._closing = False
        self._bytes = 0
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                           | getattr(os, 'O_BINARY', 0), 0o600)

    @property
    def healthy(self):
        with self._lock:
            return not self._unsafe and not self._closing and self._fd is not None

    def append(self, row):
        if type(row) is not dict:
            raise ValueError('a journal record must be a mapping')
        raw = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        if not 1 <= len(raw) <= 65536:
            raise ValueError('a journal record exceeds its size limit')
        with self._lock:
            if self._unsafe or self._closing or self._fd is None:
                raise RuntimeError('journal append state is unverified')
            if self._bytes + len(raw) > 16 * 1024 * 1024:
                raise ValueError('private journal capacity exceeded')
            # Premark before any write so an interrupted write/flush can never
            # be mistaken for a complete durable record.
            self._unsafe = True
            offset = 0
            while offset < len(raw):
                count = os.write(self._fd, raw[offset:])
                if type(count) is not int or count <= 0:
                    raise OSError('private journal write failed')
                offset += count
            os.fsync(self._fd)
            self._bytes += len(raw)
            self._unsafe = False

    def close(self):
        with self._lock:
            if self._closing:
                if self._fd is not None:
                    raise RuntimeError('journal close outcome is unverified')
                return
            self._closing = True
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


class _ActivityRecorder:
    """Persist append-only collector deltas after each completed parent query."""

    def __init__(self, activity, journal):
        self._activity, self._journal = activity, journal
        self._lock = threading.Lock()
        self._unsafe = False
        self._history = {'observations': [], 'process_queries': [], 'errors': []}
        self._error_count = 0
        self._sequence = 0

    def persist(self):
        with self._lock:
            if self._unsafe:
                raise RuntimeError('activity recording state is unverified')
            self._unsafe = True
            snapshot = self._activity.snapshot()
            if (type(snapshot) is not dict or snapshot.get('in_flight') is not False
                    or type(snapshot.get('complete')) is not bool
                    or type(snapshot.get('sealed', False)) is not bool
                    or type(snapshot.get('error_count')) is not int
                    or snapshot['error_count'] < self._error_count):
                raise RuntimeError('activity snapshot is invalid or in flight')
            delta = {}
            for key, previous in self._history.items():
                current = snapshot.get(key)
                if type(current) is not list or len(current) < len(previous) or current[:len(previous)] != previous:
                    raise RuntimeError('activity history changed or was lost')
                delta[key] = current[len(previous):]
            row = {'sequence': self._sequence, **delta, 'error_count': snapshot['error_count'],
                   'complete': snapshot['complete'], 'in_flight': False,
                   'sealed': snapshot.get('sealed', False)}
            self._journal.append(row)
            # Freeze copies only after durable append; an interruption leaves
            # this recorder unverified rather than silently omitting a point.
            self._history = {key: json.loads(json.dumps(snapshot[key], allow_nan=False))
                             for key in self._history}
            self._error_count = snapshot['error_count']
            self._sequence += 1
            self._unsafe = False
            return snapshot


def pending_activity_pollers():
    """Return rooted pollers; an unjoined producer is not a complete screen."""
    with _poller_lock:
        return tuple(_pollers)


def pending_parent_supervisions():
    """Return parent supervision capabilities not yet proven quiescent."""
    with _supervision_lock:
        return tuple(_supervisions)


def pending_parent_attempts():
    """Return attempts whose controller resources have not all been retired."""
    with _attempt_lock:
        return tuple(_attempts)


class _ParentAttempt:
    """Keep child, telemetry and file custody rooted until quiescence."""

    def __init__(self, supervision, journal):
        self.supervision, self.journal = supervision, journal
        self.live = None
        self.poller = None
        self.recorder = None
        with _attempt_lock:
            if len(_attempts) >= 16:
                raise RuntimeError('parent attempt capacity is exhausted')
            _attempts.add(self)

    def retire_if_quiescent(self, *, timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError('invalid parent attempt retirement timeout')
        if self.poller is not None and not self.poller.stop_and_join(timeout=timeout):
            return False
        if not self.supervision.release_if_quiescent(timeout=timeout):
            return False
        if self.live is not None:
            self.live.close()
            self.live = None
        self.journal.close()
        with _attempt_lock:
            _attempts.discard(self)
        return True


class _Supervision:
    """Root one strict supervisor thread before its potentially interruptible start."""

    def __init__(self, command, *, channel, identity, device_lease_path, marker_path,
                 cwd, env, timeout, process_job):
        self._command, self._channel, self._identity = command, channel, identity
        self._device_lease_path, self._marker_path = device_lease_path, marker_path
        self._cwd, self._env, self._timeout, self._process_job = cwd, env, timeout, process_job
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread = None
        self._started = False
        self._revoked = False
        self._result = None
        self._error = None

    def start(self):
        with self._lock:
            if self._started:
                raise RuntimeError('parent supervisor was already started')
            self._started = True
            with _supervision_lock:
                if len(_supervisions) >= 16:
                    raise RuntimeError('parent supervision capacity is exhausted')
                _supervisions.add(self)
            self._thread = threading.Thread(target=self._run, name='tap-parent-supervisor', daemon=True)
            try:
                self._thread.start()
            except BaseException:
                # A native thread may have started before start() was
                # interrupted; the thread checks this gate before spawning.
                self._revoked = True
                try:
                    self._channel.abort()
                finally:
                    raise

    def _run(self):
        try:
            with self._lock:
                revoked = self._revoked
            if revoked:
                return
            self._result = run_supervised(self._command, device_lease_path=self._device_lease_path,
                marker_path=self._marker_path, identity=self._identity, startup_channel=self._channel,
                timeout=self._timeout, cwd=self._cwd, env=self._env, inherit_environment=False,
                process_job=self._process_job, require_zero_exit=True)
        except BaseException as exc:
            self._error = exc
        finally:
            self._done.set()

    def abort(self):
        self._channel.abort()

    @property
    def done(self):
        return self._done.is_set()

    def wait(self, *, timeout):
        if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
            raise ValueError('invalid supervisor wait timeout')
        if not self._done.wait(timeout):
            self._channel.abort()
            raise TimeoutError('parent supervisor did not finish')
        if self._error is not None:
            raise RuntimeError('parent supervisor failed') from self._error
        if self._result is None:
            raise RuntimeError('parent supervisor has no terminal result')
        return self._result

    def release_if_quiescent(self, *, timeout):
        if type(timeout) not in (int, float) or not 0 < timeout <= 30:
            raise ValueError('invalid supervisor join timeout')
        thread = self._thread
        if thread is None or thread.ident is None:
            return False
        thread.join(timeout)
        if thread.is_alive() or not self._done.is_set():
            return False
        if self._channel.wait_closed(timeout=timeout) is not True:
            return False
        with _supervision_lock:
            _supervisions.discard(self)
        return True


class _ActivityPoller:
    """A rooted parent-only point producer, stopped and joined before seal."""

    def __init__(self, activity, live, marker_check, recorder, abort, *, interval):
        if (type(interval) not in (int, float) or not 0 < interval <= 1
                or not callable(marker_check) or not callable(abort)):
            raise ValueError('invalid telemetry poller configuration')
        self._activity, self._live, self._marker_check = activity, live, marker_check
        self._recorder, self._abort, self._interval = recorder, abort, float(interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread = None
        self._started = False
        self._revoked = False
        self._success = False
        self._error = None
        self._abort_error = None

    @property
    def error(self):
        return self._error if self._error is not None else self._abort_error

    @property
    def completed_cleanly(self):
        return (self._done.is_set() and self._success is True
                and self._error is None and self._abort_error is None)

    def start(self):
        with self._lock:
            if self._started:
                raise RuntimeError('telemetry poller was already started')
            self._started = True
            with _poller_lock:
                if len(_pollers) >= 16:
                    raise RuntimeError('telemetry poller capacity is exhausted')
                _pollers.add(self)
            self._thread = threading.Thread(target=self._run, name='tap-parent-telemetry', daemon=True)
            try:
                self._thread.start()
            except BaseException:
                # start() may be interrupted after the OS thread was created.
                # Keep the capability rooted and prevent any future query.
                self._revoked = True
                self._stop.set()
                raise

    def _run(self):
        try:
            with self._lock:
                revoked = self._revoked
            if revoked:
                return
            while not self._stop.is_set():
                self._live.check_live()
                self._marker_check()
                ok = self._activity.sample_gpu()
                self._recorder.persist()
                self._live.check_live()
                self._marker_check()
                if ok is not True:
                    raise RuntimeError('periodic GPU telemetry is unavailable')
                if self._stop.wait(self._interval):
                    break
            self._success = True
        except BaseException as exc:
            self._error = exc
            try:
                self._stop.set()
            except BaseException as stop_error:
                self._abort_error = stop_error
            finally:
                try:
                    self._abort()
                except BaseException as abort_error:
                    self._abort_error = abort_error
        finally:
            self._done.set()

    def stop_and_join(self, *, timeout):
        if type(timeout) not in (int, float) or not 0 < timeout <= 30:
            raise ValueError('invalid telemetry join timeout')
        self._stop.set()
        thread = self._thread
        if thread is None or thread.ident is None:
            return False
        thread.join(timeout)
        if thread.is_alive() or not self._done.is_set():
            return False
        with _poller_lock:
            _pollers.discard(self)
        return True


def _wait_until(target_ns, *, clock, sleep):
    """Bound a cooperative monotonic wait without claiming real-time timing."""
    previous = None
    for _ in range(200):
        now = clock()
        if type(now) is not int or now < 0 or (previous is not None and now < previous):
            raise RuntimeError('parent telemetry clock moved backwards')
        if now >= target_ns:
            return now
        previous = now
        sleep(min((target_ns - now) / 1_000_000_000, 0.05))
    raise RuntimeError('parent telemetry clock did not reach its deadline')


def _finish_screen(activity, poller, *, recorder, live, marker_check, finish, grant,
                   finish_path, gpu_uuid, expected_smi_total, exclusive, clock, sleep):
    """Stop the producer, bracket the end while the same owner is live, and permit cleanup."""
    if (type(grant) is not dict or type(gpu_uuid) is not str or grant.get('gpu_uuid') != gpu_uuid
            or type(expected_smi_total) is not int or grant.get('activity_smi_total_bytes') != expected_smi_total):
        raise ValueError('activity screen does not match the bound device and total')
    if not poller.stop_and_join(timeout=2.5) or not poller.completed_cleanly:
        raise RuntimeError('periodic telemetry producer did not stop cleanly')
    margin = _clock_margin(grant['runtime'], grant['clock'])
    end = finish['workload_ended_ns']
    _wait_until(end + margin, clock=clock, sleep=sleep)
    live.check_live()
    marker_check()
    failed = []
    if activity.sample_processes('after') is not True:
        failed.append({'code': 'after_process_query_failed'})
    try:
        live.check_live()
        marker_check()
    finally:
        recorder.persist()
    _wait_until(end + 1_000_000_000 + margin, clock=clock, sleep=sleep)
    for _ in range(3):
        if clock() > end + 5_000_000_000 - margin:
            break
        live.check_live()
        marker_check()
        if activity.sample_gpu() is not True:
            failed.append({'code': 'after_idle_query_failed'})
        try:
            live.check_live()
            marker_check()
        finally:
            recorder.persist()
    snapshot = activity.seal(producers_stopped=True)
    recorder.persist()
    if (snapshot.get('complete') is not True or snapshot.get('in_flight') is not False
            or snapshot.get('sealed') is not True):
        failed.append({'code': 'sealed_activity_incomplete'})
    screen = classify_activity(observations=snapshot['observations'], errors=snapshot['errors'] + failed,
        process_queries=snapshot['process_queries'], gpu_uuid=gpu_uuid, own_pid=grant['owner_pid'],
        workload_started_ns=finish['preflight_started_ns'], workload_ended_ns=end,
        expected_total_bytes=expected_smi_total, exclusive=exclusive)
    issued = clock()
    if type(issued) is not int or not end + margin <= issued < 2 ** 63 - 5_000_000_000:
        raise RuntimeError('cleanup grant clock is invalid')
    cleanup = {'schema': 'tap-repeated-cleanup-grant-v1',
               **{key: grant[key] for key in ('session_index', 'owner_pid', 'owner_birth',
                   'supervisor_run_id', 'challenge', 'gpu_uuid', 'cuda_total_bytes',
                   'activity_smi_total_bytes', 'clock', 'runtime')},
               'workload_ended_ns': end, 'issued_ns': issued, 'expires_ns': issued + 5_000_000_000,
               'screen_availability': screen['availability']}
    live.check_live()
    marker_check()
    _publish_barrier(finish_path, cleanup)
    return screen, cleanup


def _publish_barrier(path, payload):
    """Publish a fully written, create-only cooperative private JSON barrier."""
    path = Path(path)
    if not path.is_absolute() or type(payload) is not dict:
        raise ValueError('an absolute private barrier path and mapping are required')
    parent = path.parent.resolve(strict=True)
    info = parent.stat()
    if (not stat.S_ISDIR(info.st_mode) or parent != path.parent
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('barrier directory is not a plain canonical directory')
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if not 1 <= len(raw) <= _MAX_FRAME:
        raise ValueError('barrier payload is too large')
    temp = parent / ('.tap-barrier-' + secrets.token_hex(16))
    created = False
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
        created = True
        try:
            with os.fdopen(fd, 'wb', closefd=False) as stream:
                written = stream.write(raw)
                if type(written) is not int or written != len(raw):
                    raise OSError('short private barrier write')
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
        # Windows rename refuses an existing destination and publishes a
        # complete, single-link file in the same directory. The draft POSIX
        # fallback is create-only but is not an enabled GPU execution path.
        if os.name == 'nt':
            os.rename(temp, path)
        else:
            os.link(temp, path)
    finally:
        if created:
            temp.unlink(missing_ok=True)


def _drive_session(*, session_index, challenge, identity, channel, supervision, activity, journal,
                   attempt_dir, expected_smi_total, schedule, parent_clock, parent_runtime, exclusive,
                   clock, sleep, startup_timeout, workload_timeout, reap_timeout):
    """Coordinate one preconfigured owner; the public launcher remains disabled.

    The caller is responsible for fixed source/runtime/model checks, a fresh
    device identity and a coordinated timing reservation. This function is
    not an arbitrary-command entrypoint or a physical-exclusivity proof.
    """
    if _PARENT_LAUNCH_ENABLED is not True:
        raise RuntimeError('repeated tap parent launcher is disabled')
    directory = Path(attempt_dir)
    if not directory.is_absolute() or directory.resolve(strict=True) != directory:
        raise ValueError('attempt directory must be absolute and canonical')
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('attempt directory must be a plain directory')
    start_path, finished_path = directory / 'start-grant.json', directory / 'finished.json'
    cleanup_path, owner_journal = directory / 'cleanup-grant.json', directory / 'journal.jsonl'
    if (type(session_index) is not int or session_index not in range(5)
            or type(challenge) is not str or re.fullmatch('[0-9a-f]{64}', challenge) is None
            or type(journal) is not _PrivateJournal or journal._path != directory / 'activity.jsonl'
            or journal.healthy is not True or activity.gpu_uuid != identity.gpu_uuid
            or supervision._channel is not channel or supervision._identity is not identity
            or supervision._process_job is None or exclusive is not True
            or type(expected_smi_total) is not int or expected_smi_total <= 0
            or not all(callable(item) for item in (clock, sleep))
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= bound
                   for value, bound in ((startup_timeout, 180), (workload_timeout, 3600), (reap_timeout, 30)))
            or any(os.path.lexists(path) for path in (start_path, finished_path, cleanup_path, owner_journal))):
        raise ValueError('invalid or reused private session setup')
    _clock_margin(parent_runtime, parent_clock)
    attempt = _ParentAttempt(supervision, journal)
    try:
        supervision.start()
        raw_ready = channel.receive(timeout=startup_timeout)
        received = clock()
        ready = _decode_ready(raw_ready, session_index=session_index, challenge=challenge)
        _check_ready_clock(ready, parent_clock=parent_clock, parent_runtime=parent_runtime, received_ns=received)
        live = _bind_ready(ready, identity=identity)
        attempt.live = live
        marker_check = lambda: _check_marker(identity, ready['supervisor_run_id'])
        recorder = _ActivityRecorder(activity, journal)
        attempt.recorder = recorder
        grant = _initial_screen(activity, ready=ready, live=live, marker_check=marker_check,
            persist=lambda _snapshot: recorder.persist(), grant_path=start_path,
            parent_clock=parent_clock, parent_runtime=parent_runtime, received_ns=received,
            gpu_uuid=identity.gpu_uuid, expected_smi_total=expected_smi_total, clock=clock)
        poller = _ActivityPoller(activity, live, marker_check, recorder, supervision.abort, interval=0.25)
        attempt.poller = poller
        poller.start()
        finish = _await_finished(finished_path, grant=grant, live=live, marker_check=marker_check,
            supervisor_done=lambda: supervision.done, clock=clock, sleep=sleep, timeout=workload_timeout)
        live.check_live()
        marker_check()
        _rows, journal_sha256 = _read_bound_journal(owner_journal, finish=finish, grant=grant)
        live.check_live()
        marker_check()
        screen, _cleanup = _finish_screen(activity, poller, recorder=recorder, live=live,
            marker_check=marker_check, finish=finish, grant=grant, finish_path=cleanup_path,
            gpu_uuid=identity.gpu_uuid, expected_smi_total=expected_smi_total,
            exclusive=exclusive, clock=clock, sleep=sleep)
        result = supervision.wait(timeout=reap_timeout)
        _check_supervision_result(result, ready=ready, live=live)
        if not attempt.retire_if_quiescent(timeout=reap_timeout):
            raise RuntimeError('session resources did not become quiescent')
        rows = _recheck_bound_journal(owner_journal, finish=finish, grant=grant,
                                      expected_sha256=journal_sha256)
        summary = summarize_session(schedule, session_index, rows, clock=parent_clock,
            total_bytes=ready['cuda_total_bytes'], cap_bytes=ready['cuda_total_bytes'] // 4,
            device='cuda:0', run_protocol=EXPERIMENT_PROTOCOL)
        if screen.get('availability') == 'unverified':
            state, screen_state, summary = 'incomplete', 'unverified', None
        elif (screen.get('availability') == 'complete'
              and screen.get('status') in ('no_contamination_observed', 'contaminated')):
            state, screen_state = 'pending_owner_evidence', screen['status']
        else:
            raise RuntimeError('activity screen has an invalid completion state')
        return {'session': session_index, 'state': state, 'screen_state': screen_state,
                'journal_sha256': journal_sha256, 'journal': rows,
                'preflight_started_ns': finish['preflight_started_ns'],
                'workload_ended_ns': finish['workload_ended_ns'],
                'parent_identity': {'run_id': ready['supervisor_run_id'], 'pid': ready['owner_pid'],
                                    'birth': ready['owner_birth']},
                'parent_observed_live_handle': True, 'activity_identity_held_through_queries': True,
                'gpu_uuid': ready['gpu_uuid'], 'cuda_total_bytes': ready['cuda_total_bytes'],
                'activity_smi_total_bytes': expected_smi_total,
                'activity': activity.snapshot(), 'summary': summary, 'screen': screen,
                'supervision': result}
    except BaseException:
        # A visible grant may already have been consumed. Abort the exact
        # registered child, retain every uncertain capability, and preserve
        # whatever query failure the now-quiescent collector can report.
        try:
            supervision.abort()
        except BaseException:
            pass
        try:
            producer_stopped = attempt.poller is None or attempt.poller.stop_and_join(timeout=reap_timeout)
            if producer_stopped and attempt.recorder is not None:
                attempt.recorder.persist()
        except BaseException:
            pass
        try:
            attempt.retire_if_quiescent(timeout=reap_timeout)
        except BaseException:
            pass
        raise


def run_session(*, attempt_dir):
    """Reject all execution until the full parent/owner integration is reviewed."""
    if _PARENT_LAUNCH_ENABLED is not True:
        raise RuntimeError('repeated tap parent launcher is disabled')
    raise RuntimeError('repeated tap parent launcher is not implemented')
