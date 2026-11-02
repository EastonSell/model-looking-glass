"""Bounded append-only saved-byte journal; operates on caller-supplied bytes."""
import hashlib
import json
import math
import os
import re
from pathlib import Path


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


class Journal:
    def __init__(self, directory, *, raw_cap=150 * 1024 * 1024,
                 metadata_cap=16 * 1024 * 1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.raw_cap, self.metadata_cap = raw_cap, metadata_cap
        self.raw_bytes = self.metadata_bytes = 0
        self.current = None
        self.action_metadata_start = 0
        self.journal_path = self.directory / 'events.jsonl'
        self.stream = self.journal_path.open('xb')

    def _event(self, value):
        encoded = _encoded(value)
        if self.metadata_bytes + len(encoded) > self.metadata_cap:
            raise ValueError('metadata cap exceeded')
        if (self.current is not None
                and self.metadata_bytes - self.action_metadata_start + len(encoded) > 48 * 1024):
            raise ValueError('per-action metadata bound exceeded')
        self.stream.write(encoded)
        self.stream.flush()
        self.metadata_bytes += len(encoded)

    def start(self, ordinal, case_id, action, details):
        if self.current is not None or type(ordinal) is not int or ordinal < 0:
            raise ValueError('invalid action start')
        if type(case_id) is not str or re.fullmatch(r'distractor-repair-\d{2}', case_id) is None:
            raise ValueError('invalid case identity')
        self.action_metadata_start = self.metadata_bytes
        self.current = {'ordinal': ordinal, 'case_id': case_id, 'action': action}
        self._event({'type': 'start', **self.current, 'details': details})

    def record(self, name, data, shape):
        if self.current is None or type(name) is not str or not name or type(data) is not bytes:
            raise ValueError('invalid saved tensor')
        if (type(shape) not in (list, tuple) or not shape
                or any(type(x) is not int or x < 1 for x in shape)
                or math.prod(shape) * 4 != len(data)):
            raise ValueError('tensor bytes do not match float32 shape')
        if self.raw_bytes + len(data) > self.raw_cap:
            raise ValueError('raw byte cap exceeded')
        file = self.directory / (self.current['case_id'] + '.f32')
        offset = file.stat().st_size if file.exists() else 0
        with file.open('ab' if offset else 'xb') as stream:
            stream.write(data)
            stream.flush()
        self.raw_bytes += len(data)
        descriptor = {'name': name, 'shape': list(shape), 'dtype': 'float32-le',
                      'file': file.name, 'offset': offset, 'length': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()}
        self._event({'type': 'tensor', **self.current, 'tensor': descriptor})
        return descriptor

    def record64(self, name, data, shape):
        if self.current is None or type(name) is not str or not name or type(data) is not bytes:
            raise ValueError('invalid saved tensor')
        if (type(shape) not in (list, tuple) or not shape
                or any(type(x) is not int or x < 1 for x in shape)
                or math.prod(shape) * 8 != len(data)):
            raise ValueError('tensor bytes do not match float64 shape')
        if self.raw_bytes + len(data) > self.raw_cap:
            raise ValueError('raw byte cap exceeded')
        file = self.directory / (self.current['case_id'] + '.f64')
        offset = file.stat().st_size if file.exists() else 0
        with file.open('ab' if offset else 'xb') as stream:
            stream.write(data)
            stream.flush()
        self.raw_bytes += len(data)
        descriptor = {'name': name, 'shape': list(shape), 'dtype': 'float64-le',
                      'file': file.name, 'offset': offset, 'length': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()}
        self._event({'type': 'tensor', **self.current, 'tensor': descriptor})
        return descriptor

    def complete(self, descriptors, events, checks):
        if self.current is None:
            raise ValueError('action has not started')
        self._event({'type': 'raw-recorded', **self.current, 'tensors': descriptors,
                     'hook_events': events})
        self._event({'type': 'complete', **self.current, 'checks': checks})
        self.current = None

    def fail(self, error, *, partial=None, partial_error=None):
        description = {'type': type(error).__name__, 'current': self.current,
                       'partial': partial, 'partial_write_error': partial_error,
                       'raw_bytes_recorded': self.raw_bytes,
                       'scope': 'incomplete; direct child status and lost-return boundaries require receipt'}
        encoded = (json.dumps(description, indent=2, sort_keys=True) + '\n').encode('utf-8')
        with (self.directory / 'failure.json').open('xb') as failure:
            failure.write(encoded)
        self._event({'type': 'failure', 'failure_sha256': hashlib.sha256(encoded).hexdigest(),
                     'current': self.current})

    def close(self):
        self.stream.flush()
        self.stream.close()
