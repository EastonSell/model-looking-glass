"""Parent-side point queries for the proposed repeated tap activity screen.

This module never imports or initializes CUDA. Queries are performed in the
parent, outside the future benchmark owner's process tree. A trusted caller
must stop all producers before sealing and must independently supervise any
stranded query thread/process; point samples do not establish exclusivity.
"""

import csv
import io
import os
from pathlib import Path
import subprocess
import threading
import time

from .device_identity import normalize_gpu_uuid

_MIB = 1024 ** 2
_MAX_OUTPUT = 64 * 1024
_MAX_ROWS = 128
_GPU_FIELDS = 'uuid,memory.total,memory.used,memory.free,utilization.gpu'
_PROCESS_FIELDS = 'gpu_uuid,pid,process_name'


def _rows(raw):
    if type(raw) is not bytes or len(raw) > _MAX_OUTPUT:
        raise ValueError('invalid or oversized telemetry output')
    text = raw.decode('utf-8', errors='strict')
    if '\x00' in text:
        raise ValueError('invalid telemetry text')
    parsed = list(csv.reader(io.StringIO(text), strict=True, skipinitialspace=True))
    if len(parsed) > _MAX_ROWS or any(not row or len(row) > 5 for row in parsed):
        raise ValueError('invalid telemetry row inventory')
    return [[field.strip() for field in row] for row in parsed]


def _number(text):
    if not text or len(text) > 20 or not text.isascii() or not text.isdecimal():
        raise ValueError('telemetry value is not an unsigned integer')
    return int(text)


def parse_gpu(raw, *, gpu_uuid, started_ns, ended_ns):
    """Parse a target GPU point; the caller supplies actual query timestamps."""
    uuid = normalize_gpu_uuid(gpu_uuid)
    if type(started_ns) is not int or type(ended_ns) is not int or not 0 <= started_ns <= ended_ns:
        raise ValueError('invalid telemetry query timestamps')
    selected = []
    for fields in _rows(raw):
        if len(fields) != 5:
            raise ValueError('invalid GPU telemetry row')
        row_uuid = normalize_gpu_uuid(fields[0])
        total, used, free, utilization = map(_number, fields[1:])
        if not total or used > total or free > total or used + free > total + 2 or utilization > 100:
            raise ValueError('invalid GPU telemetry values')
        if row_uuid == uuid:
            selected.append((total, used, free, utilization))
    if len(selected) != 1:
        raise ValueError('target GPU telemetry is missing or ambiguous')
    total, used, free, utilization = selected[0]
    return {'gpu_uuid': uuid, 'query_started_ns': started_ns, 'query_ended_ns': ended_ns,
            'observed_ns': ended_ns, 'total_bytes': total * _MIB, 'used_bytes': used * _MIB,
            'free_bytes': free * _MIB, 'utilization_percent': utilization}


def parse_processes(raw, *, gpu_uuid, phase, started_ns, ended_ns):
    """Parse the target's private compute-process endpoint observation."""
    uuid = normalize_gpu_uuid(gpu_uuid)
    if (phase not in ('before', 'after') or type(started_ns) is not int or type(ended_ns) is not int
            or not 0 <= started_ns <= ended_ns):
        raise ValueError('invalid process-query phase or timestamps')
    processes = []
    for fields in _rows(raw):
        if len(fields) != 3:
            raise ValueError('invalid process telemetry row')
        row_uuid, pid, name = normalize_gpu_uuid(fields[0]), _number(fields[1]), fields[2]
        if not 0 < pid < 2 ** 32 or not name or len(name) > 1024:
            raise ValueError('invalid process telemetry values')
        if row_uuid == uuid:
            processes.append({'gpu_uuid': uuid, 'pid': pid, 'name': name})
    return {'phase': phase, 'ok': True, 'query_started_ns': started_ns,
            'query_ended_ns': ended_ns, 'processes': processes}


class _NvidiaSmi:
    """Invoke one trusted local binary with fixed, shell-free query arguments.

    The subprocess output is checked after capture; subprocess buffering and
    native driver allocations are not physically capped by this module.
    """

    def __init__(self, executable, *, timeout=1.5):
        path = Path(executable)
        if not path.is_absolute() or not path.resolve(strict=True).is_file():
            raise ValueError('an absolute local NVIDIA telemetry executable is required')
        if type(timeout) not in (int, float) or not 0 < timeout <= 2:
            raise ValueError('telemetry timeout must be within two seconds')
        self.executable, self.timeout = str(path.resolve(strict=True)), float(timeout)

    def __call__(self, kind):
        if kind == 'gpu':
            query = '--query-gpu=' + _GPU_FIELDS
        elif kind == 'process':
            query = '--query-compute-apps=' + _PROCESS_FIELDS
        else:
            raise ValueError('unsupported telemetry query')
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
        result = subprocess.run([self.executable, query, '--format=csv,noheader,nounits'],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=self.timeout, check=True, creationflags=flags)
        if type(result.stdout) is not bytes or len(result.stdout) > _MAX_OUTPUT:
            raise ValueError('telemetry output exceeds its retained bound')
        return result.stdout


class ParentActivity:
    """Bounded parent-owned point inventory, with sticky incomplete status.

    Calls are serialized without holding the inventory lock during the query.
    An outside controller must stop/join every producer before ``seal``; this
    class cannot prove quiescence or prevent a caller from using a stale copy.
    """

    def __init__(self, *, gpu_uuid, query, clock=time.perf_counter_ns, max_samples=512):
        self.gpu_uuid = normalize_gpu_uuid(gpu_uuid)
        if not callable(query) or not callable(clock) or type(max_samples) is not int or not 3 <= max_samples <= 4096:
            raise ValueError('invalid parent activity configuration')
        self._query, self._clock, self._max_samples = query, clock, max_samples
        self._lock = threading.Lock()
        self._busy = False
        self._sealed = False
        self._incomplete = False
        self._observations, self._process_queries, self._errors = [], [], []
        self._error_count = 0

    def _fail_locked(self, code):
        self._incomplete = True
        self._error_count += 1
        if len(self._errors) < 32:
            self._errors.append({'code': code})

    def _collect(self, kind, phase=None):
        with self._lock:
            if self._sealed or self._busy:
                self._fail_locked('late_or_overlapping_query')
                return False
            if ((kind == 'gpu' and len(self._observations) >= self._max_samples)
                    or (kind == 'process' and (phase not in ('before', 'after')
                        or any(row['phase'] == phase for row in self._process_queries)))):
                self._fail_locked('inventory_exceeded_or_duplicate')
                return False
            self._busy = True
        try:
            started = self._clock()
            raw = self._query(kind)
            ended = self._clock()
            if kind == 'gpu':
                row = parse_gpu(raw, gpu_uuid=self.gpu_uuid, started_ns=started, ended_ns=ended)
            else:
                row = parse_processes(raw, gpu_uuid=self.gpu_uuid, phase=phase,
                                      started_ns=started, ended_ns=ended)
        except BaseException as exc:
            with self._lock:
                self._fail_locked('query_or_parse_failed')
                self._busy = False
            if not isinstance(exc, Exception):
                raise
            return False
        with self._lock:
            if self._sealed:
                self._fail_locked('late_or_overlapping_query')
                self._busy = False
                return False
            (self._observations if kind == 'gpu' else self._process_queries).append(row)
            # Keep the pending marker until the point is installed. If an
            # interruption escapes either publication or its failure handler,
            # sealing cannot silently certify the missing observation.
            self._busy = False
        return True

    def sample_gpu(self):
        return self._collect('gpu')

    def sample_processes(self, phase):
        return self._collect('process', phase)

    def _snapshot_locked(self):
        return {'observations': [dict(row) for row in self._observations],
                'process_queries': [{**row, 'processes': [dict(item) for item in row['processes']]}
                                    for row in self._process_queries],
                'errors': [dict(row) for row in self._errors], 'error_count': self._error_count,
                'complete': not self._incomplete and not self._busy,
                'in_flight': self._busy, 'sealed': self._sealed}

    def snapshot(self):
        """Return a detached point-in-time view, which may later become stale."""
        with self._lock:
            return self._snapshot_locked()

    def seal(self, *, producers_stopped):
        if producers_stopped is not True:
            raise ValueError('the parent must stop and join every producer before sealing')
        with self._lock:
            if self._busy or self._sealed:
                self._fail_locked('unquiesced_or_duplicate_seal')
            self._sealed = True
            return self._snapshot_locked()
