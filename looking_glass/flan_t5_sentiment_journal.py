"""Single-threaded, bounded file journal for prospective sentiment execution.

The journal records local saved evidence, not authentication, process teardown,
or a process-memory bound. Its owner must retain partial attempts and must not
retry after an uncertain component call. The containing directory is trusted.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading


class JournalError(RuntimeError):
    """The attempt cannot safely continue or publish a complete candidate."""


TOTAL_BYTES = 5 * 1024**3 // 4
METADATA_BYTES = 64 * 1024**2
FAILURE_BYTES = 4 * 1024**2
FREE_BYTES = 2 * 1024**3
MAX_FORWARDS = 2940
# Reserve the entire largest decoder return plus one possibly mutated state.
MAX_COMPONENT_BYTES = (6 * 32128 + 64 * 512) * 4
MAX_RECORD_BYTES = 64 * 1024
WORST_CASE_BYTES = (484 * 14 * 32128 * 4 + 484 * 2 * 64 * 512 * 4
                    + 36 * 64 * 512 * 4 + 8 * 32 * 512 * 4
                    + METADATA_BYTES + FAILURE_BYTES)


@dataclass(frozen=True, slots=True)
class Limits:
    total_bytes: int = TOTAL_BYTES
    metadata_bytes: int = METADATA_BYTES
    failure_bytes: int = FAILURE_BYTES
    forward_bytes: int = MAX_COMPONENT_BYTES
    max_forwards: int = MAX_FORWARDS
    free_bytes: int = FREE_BYTES

    def __post_init__(self):
        values = ((self.total_bytes, TOTAL_BYTES), (self.metadata_bytes, METADATA_BYTES),
                  (self.failure_bytes, FAILURE_BYTES), (self.forward_bytes, MAX_COMPONENT_BYTES),
                  (self.max_forwards, MAX_FORWARDS))
        if (any(type(value) is not int or not 0 < value <= upper for value, upper in values)
                or type(self.free_bytes) is not int or not 0 <= self.free_bytes <= FREE_BYTES
                or self.metadata_bytes + self.failure_bytes >= self.total_bytes):
            raise ValueError('invalid evidence limits')


def require_protocol_allocation(limits):
    """Check worst-case retained files and free-space policy before model load."""
    if (type(limits) is not Limits or limits.total_bytes < WORST_CASE_BYTES
            or limits.metadata_bytes != METADATA_BYTES or limits.failure_bytes != FAILURE_BYTES
            or limits.forward_bytes != MAX_COMPONENT_BYTES or limits.max_forwards != MAX_FORWARDS
            or limits.free_bytes != FREE_BYTES):
        raise JournalError('insufficient protocol allocation')
    return WORST_CASE_BYTES


def _encoded(value):
    try:
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                          allow_nan=False) + '\n').encode('utf8')
    except (TypeError, ValueError, OverflowError) as error:
        raise JournalError('record is not finite JSON') from error
    if len(raw) > MAX_RECORD_BYTES:
        raise JournalError('record exceeds its byte limit')
    return raw


def _write_new(path, raw):
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short evidence write')
        stream.flush()
        os.fsync(stream.fileno())


class EvidenceJournal:
    """Create a unique attempt and persist each component start and return."""

    def __init__(self, directory, metadata, *, limits=Limits()):
        directory = Path(directory)
        if (type(limits) is not Limits or type(metadata) is not dict or not directory.is_absolute()
                or not directory.parent.is_dir()):
            raise ValueError('expected an absolute new attempt in an existing trusted parent')
        if shutil.disk_usage(directory.parent)[2] < limits.free_bytes:
            raise JournalError('insufficient free space before model loading')
        raw = _encoded(metadata)
        if len(raw) > limits.metadata_bytes or len(raw) + limits.failure_bytes > limits.total_bytes:
            raise JournalError('initial metadata exceeds the retained budget')
        directory.mkdir()
        self.directory = directory
        self.limits = limits
        self._thread = threading.current_thread()
        self._owner_local = threading.local()
        self._owner_token = object()
        self._owner_local.token = self._owner_token
        self._used = self._metadata = self._tensor_index = self.component_count = 0
        self._active = None
        self._component_bytes = self._component_limit = 0
        self._faulted = self._finished = False
        self._faulted = True
        (directory / 'tensors').mkdir()
        _write_new(directory / 'started.json', raw)
        _write_new(directory / 'events.jsonl', b'')
        self._used = self._metadata = len(raw)
        self._faulted = False

    @property
    def retained_bytes(self):
        return self._used

    @property
    def active_component(self):
        return self._active

    @property
    def remaining_component_bytes(self):
        return self._component_limit - self._component_bytes if self._active is not None else 0

    def _check(self):
        if (threading.current_thread() is not self._thread
                or getattr(self._owner_local, 'token', None) is not self._owner_token):
            raise JournalError('the journal belongs to its creating thread')
        if self._finished or self._faulted:
            raise JournalError('journal is terminal or faulted')

    def _space(self, *, payload=0, metadata=0):
        if (self._metadata + metadata > self.limits.metadata_bytes
                or self._used + payload + metadata + self.limits.failure_bytes > self.limits.total_bytes):
            raise JournalError('retained-evidence capacity exceeded')

    def record(self, event, detail):
        self._check()
        if type(event) is not str or not event or type(detail) is not dict:
            raise ValueError('invalid journal event')
        line = _encoded({'event': event, 'detail': detail})
        self._space(metadata=len(line))
        self._faulted = True
        with (self.directory / 'events.jsonl').open('ab') as stream:
            if stream.write(line) != len(line):
                raise OSError('short journal append')
            stream.flush()
            os.fsync(stream.fileno())
        self._metadata += len(line)
        self._used += len(line)
        self._faulted = False

    def begin_component(self, identity, *, reserve_bytes):
        self._check()
        if (self._active is not None or type(identity) is not dict
                or type(reserve_bytes) is not int or not 0 < reserve_bytes <= self.limits.forward_bytes
                or self.component_count >= self.limits.max_forwards):
            raise JournalError('invalid or exhausted component reservation')
        # Reserve enough room for this bounded return and its two records.
        self._space(payload=reserve_bytes, metadata=2 * MAX_RECORD_BYTES)
        self._active = dict(identity)
        self._component_limit, self._component_bytes = reserve_bytes, 0
        self.record('component_start', {'index': self.component_count, **identity})
        self.component_count += 1

    def tensor(self, shape, raw):
        self._check()
        if (type(shape) not in (tuple, list) or not shape or len(shape) > 4
                or any(type(dim) is not int or dim < 1 for dim in shape)
                or type(raw) is not bytes or len(raw) != 4 * math.prod(shape)
                or len(raw) > self.limits.forward_bytes):
            raise JournalError('invalid bounded float32 tensor')
        if self._active is not None and self._component_bytes + len(raw) > self._component_limit:
            raise JournalError('component tensor exceeds its reservation')
        self._space(payload=len(raw))
        path = f'tensors/{self._tensor_index:06d}.f32le'
        ref = {'path': path, 'shape': list(shape), 'bytes': len(raw),
               'sha256': hashlib.sha256(raw).hexdigest(), 'dtype': 'float32-le'}
        self._faulted = True
        _write_new(self.directory / path, raw)
        self._used += len(raw)
        if self._active is not None:
            self._component_bytes += len(raw)
        self._tensor_index += 1
        self._faulted = False
        return ref

    def finish_component(self, returned):
        self._check()
        if self._active is None or type(returned) is not dict:
            raise JournalError('there is no active component')
        self.record('component_return', {'index': self.component_count - 1, **returned})
        self._active = None
        self._component_bytes = self._component_limit = 0

    def read_tensor(self, ref):
        self._check()
        if (type(ref) is not dict or set(ref) != {'path', 'shape', 'bytes', 'sha256', 'dtype'}
                or type(ref['path']) is not str or not ref['path'].startswith('tensors/')
                or ref['path'].count('/') != 1 or not ref['path'][8:].endswith('.f32le')
                or not ref['path'][8:-6].isdigit() or type(ref['shape']) is not list
                or not ref['shape'] or any(type(d) is not int or d <= 0 for d in ref['shape'])
                or type(ref['bytes']) is not int or ref['bytes'] != 4 * math.prod(ref['shape'])
                or ref['bytes'] > self.limits.forward_bytes or ref['dtype'] != 'float32-le'):
            raise JournalError('invalid saved tensor reference')
        path = self.directory / ref['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != ref['bytes']:
            raise JournalError('saved tensor is absent, aliased or changed')
        with path.open('rb') as stream:
            raw = stream.read(ref['bytes'] + 1)
        if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise JournalError('saved tensor bytes changed')
        return raw

    def finish(self, status, detail):
        if (threading.current_thread() is not self._thread
                or getattr(self._owner_local, 'token', None) is not self._owner_token or self._finished):
            raise JournalError('invalid journal owner or terminal state')
        if status not in ('complete_candidate', 'incomplete') or type(detail) is not dict:
            raise ValueError('invalid terminal')
        if status == 'complete_candidate' and (self._faulted or self._active is not None):
            raise JournalError('an incomplete component cannot become complete')
        raw = _encoded({'status': status, 'detail': detail, 'components': self.component_count,
                        'active_component': self._active, 'retained_bytes': self._used,
                        'storage_accounting_complete': not self._faulted})
        if len(raw) > self.limits.failure_bytes:
            raise JournalError('terminal exceeds the reserved failure capacity')
        self._finished = True
        _write_new(self.directory / 'terminal.json', raw)
