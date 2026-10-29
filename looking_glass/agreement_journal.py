"""Single-owner, bounded tensor evidence journal for an agreement attempt.

This counts immutable retained payload bytes, not transient RAM, CUDA memory,
allocator reserve, metadata size outside the separate bounds, or disk overhead.
It does not itself decide whether a study is scientifically complete.
The creating thread is retained and also holds a thread-local owner token;
reused thread identifiers or stale dummy-thread objects do not transfer access.
This cooperative same-process check is not a security boundary.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid


class JournalError(RuntimeError):
    """The attempt cannot continue with the supplied payload or state."""


_MAX_TOTAL = 512 * 1024 * 1024
_MAX_FORWARD = 1024 * 1024
_MAX_EVENTS = 10000
_MAX_METADATA = 256 * 1024
_MAX_EVENT_LOG = 64 * 1024 * 1024
_NAME = re.compile(r'[A-Za-z0-9_.-]{1,128}\Z')


def _json_bytes(value):
    try:
        data = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError, OverflowError) as error:
        raise JournalError('metadata is not finite JSON') from error
    if len(data) > _MAX_METADATA:
        raise JournalError('metadata exceeds its byte limit')
    return data


def _exclusive_write(path, data):
    with open(path, 'xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short artifact write')
        stream.flush()
        os.fsync(stream.fileno())


class AgreementJournal:
    """Create one never-overwritten attempt with bounded immutable payloads."""

    def __init__(self, directory, metadata, *, max_total_bytes=_MAX_TOTAL,
                 max_forward_bytes=_MAX_FORWARD, max_events=_MAX_EVENTS,
                 max_event_log_bytes=_MAX_EVENT_LOG):
        for value, limit in ((max_total_bytes, _MAX_TOTAL), (max_forward_bytes, _MAX_FORWARD),
                             (max_events, _MAX_EVENTS), (max_event_log_bytes, _MAX_EVENT_LOG)):
            if type(value) is not int or not 0 < value <= limit:
                raise ValueError('journal limits must be positive and within protocol bounds')
        if type(metadata) is not dict:
            raise ValueError('attempt metadata must be a mapping')
        data = _json_bytes(metadata)
        directory = Path(directory)
        if not directory.is_absolute() or not directory.parent.is_dir():
            raise ValueError('attempt needs an absolute path with an existing parent')
        directory.mkdir()
        self._directory = directory
        self._owner = threading.current_thread()
        self._owner_context = threading.local()
        self._owner_token = object()
        self._owner_context.token = self._owner_token
        self._max_total = max_total_bytes
        self._max_forward = max_forward_bytes
        self._max_events = max_events
        self._max_event_log_bytes = max_event_log_bytes
        self._event_log_bytes = 0
        self._retained = 0
        self._events = 0
        self._finished = False
        self._faulted = False
        self._known = {}
        (directory / 'tensors').mkdir()
        _exclusive_write(directory / 'started.json', data)
        _exclusive_write(directory / 'events.jsonl', b'')

    @property
    def retained_bytes(self):
        return self._retained

    def _check_owner(self):
        if (threading.current_thread() is not self._owner
                or getattr(self._owner_context, 'token', None) is not self._owner_token):
            raise JournalError('journal must stay on its creating thread')
        if self._finished:
            raise JournalError('attempt is already terminal')

    def record(self, event, tensors):
        """Append one forward event after storing its bounded float32 payloads.

        ``tensors`` maps a name to ``(shape, little_endian_float32_bytes)``.
        Content-addressed payloads are stored once; each event still has its
        own shape and digest. All names and metadata are validated before any
        publication. A failed record stops further work on this journal.
        """
        self._check_owner()
        if self._faulted:
            raise JournalError('a failed record requires an incomplete terminal')
        try:
            if type(event) is not dict or type(tensors) is not dict or len(tensors) > 128:
                raise JournalError('event and tensor inputs must be bounded mappings')
            if self._events >= self._max_events:
                raise JournalError('event count exceeds its limit')
            refs = {}
            pending = {}
            forward_bytes = 0
            for name, value in tensors.items():
                if type(name) is not str or not _NAME.fullmatch(name):
                    raise JournalError('tensor name is invalid')
                if type(value) not in (tuple, list) or len(value) != 2:
                    raise JournalError('tensor payload must contain a shape and bytes')
                shape, payload = value
                if (type(shape) not in (tuple, list) or not 1 <= len(shape) <= 8
                        or any(type(dim) is not int or dim <= 0 for dim in shape)
                        or type(payload) is not bytes or math.prod(shape) * 4 != len(payload)):
                    raise JournalError('tensor shape and float32 payload disagree')
                forward_bytes += len(payload)
                if forward_bytes > self._max_forward:
                    raise JournalError('forward tensor payload exceeds its limit')
                digest = hashlib.sha256(payload).hexdigest()
                if digest in pending and pending[digest] != payload:
                    raise JournalError('tensor digest collision')
                if digest in self._known:
                    existing = self._directory / 'tensors' / (digest + '.f32le')
                    if existing.read_bytes() != payload:
                        raise JournalError('existing tensor changed or digest collided')
                else:
                    pending[digest] = payload
                refs[name] = {'sha256': digest, 'shape': list(shape),
                              'dtype': 'float32-le', 'bytes': len(payload)}
            new_bytes = sum(len(payload) for payload in pending.values())
            if self._retained + new_bytes > self._max_total:
                raise JournalError('retained tensor payload exceeds its limit')
            published = {'index': self._events, 'event': event, 'tensors': refs}
            line = _json_bytes(published)
            if self._event_log_bytes + len(line) > self._max_event_log_bytes:
                raise JournalError('event log exceeds its byte limit')
            for digest, payload in pending.items():
                _exclusive_write(self._directory / 'tensors' / (digest + '.f32le'), payload)
                self._known[digest] = len(payload)
                self._retained += len(payload)
            with open(self._directory / 'events.jsonl', 'ab') as stream:
                if stream.write(line) != len(line):
                    raise OSError('short event write')
                stream.flush()
                os.fsync(stream.fileno())
            self._events += 1
            self._event_log_bytes += len(line)
            return refs
        except BaseException:
            self._faulted = True
            raise

    def finish(self, status, detail):
        """Publish a terminal candidate without allowing subsequent records.

        A complete candidate is not an authoritative successful experiment:
        its controller must also verify the child exit and independently audit
        the entire inventory. A failed/interrupted publication leaves a pending
        file or no terminal; a hard-linked candidate can exist after interruption.
        """
        self._check_owner()
        if status not in ('complete', 'failed', 'cancelled') or type(detail) is not dict:
            raise ValueError('invalid terminal status or detail')
        if self._faulted and status == 'complete':
            raise JournalError('a failed record cannot become complete')
        candidate_status = 'complete_candidate' if status == 'complete' else status
        data = _json_bytes({'status': candidate_status, 'detail': detail, 'events': self._events,
                            'retained_tensor_bytes': self._retained,
                            'event_log_bytes': self._event_log_bytes,
                            'storage_accounting_complete': not self._faulted})
        # Once terminal publication begins, even an uncertain or interrupted
        # write must not permit another event or a second successful finish.
        self._finished = True
        pending = self._directory / ('.terminal-' + uuid.uuid4().hex + '.pending')
        _exclusive_write(pending, data)
        os.link(pending, self._directory / 'terminal.json')
        pending.unlink()

    def read_tensor(self, reference):
        """Read and revalidate an immutable tensor stored by this journal."""
        self._check_owner()
        if self._faulted:
            raise JournalError('a failed operation requires an incomplete terminal')
        try:
            if type(reference) is not dict or set(reference) != {'sha256', 'shape', 'dtype', 'bytes'}:
                raise JournalError('invalid tensor reference')
            digest, shape, size = reference['sha256'], reference['shape'], reference['bytes']
            if (type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                    or type(shape) is not list or not 1 <= len(shape) <= 8
                    or any(type(dim) is not int or dim <= 0 for dim in shape)
                    or type(size) is not int or math.prod(shape) * 4 != size
                    or reference['dtype'] != 'float32-le' or self._known.get(digest) != size):
                raise JournalError('tensor reference does not match this attempt')
            path = self._directory / 'tensors' / (digest + '.f32le')
            if path.is_symlink() or path.stat().st_size != size:
                raise JournalError('tensor file is absent, aliased or changed')
            payload = path.read_bytes()
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                raise JournalError('tensor file differs from its digest')
            return payload
        except OSError as error:
            self._faulted = True
            raise JournalError('tensor file could not be read') from error
        except BaseException:
            self._faulted = True
            raise
