"""Bounded private CPU-row evidence for the paired grouped development run.

This writer neither launches nor attests an owner and does not access CUDA.
The caller must retain a separate host/disk claim, root each original result,
prevent concurrent mutation of aliases, and independently bind the source,
session, model, request plan and observed group membership. It must retain the
writer and its failure roots until process recovery. A recorded digest proves
the retained bytes only; numerical finiteness, routing and tolerance are for
the independent paired analyzer. Directory and interpreter are trusted and
quiescent apart from the single designated writer. Arbitrary asynchronous
interruption, inaccessible native temporaries and crash durability remain
outside this cooperative contract.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import threading

ROW_SHAPE = (1, 4, 50257)
ROW_BYTES = 1 * 4 * 50257 * 4
ROWS_PER_MODE = 8
MAX_ROWS = 2 * ROWS_PER_MODE
MAX_METADATA_BYTES = 64 * 1024
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z', re.ASCII)


def _identity(path, *, directory=False):
    info = path.lstat()
    if ((directory and not stat.S_ISDIR(info.st_mode))
            or (not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1))
            or not info.st_ino or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise RuntimeError('private evidence path has an ambiguous identity')
    return info.st_dev, info.st_ino


def _write_all(fd, view, hasher=None):
    offset = 0
    while offset < len(view):
        piece = view[offset:offset + 65536]
        count = os.write(fd, piece)
        if type(count) is not int or not 0 < count <= len(piece):
            raise OSError('evidence write made invalid progress')
        if hasher is not None:
            hasher.update(piece[:count])
        offset += count


def _readback(path, expected_identity, size):
    if _identity(path) != expected_identity:
        raise RuntimeError('evidence file changed before readback')
    hasher, total = hashlib.sha256(), 0
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != expected_identity or opened.st_size != size:
            raise RuntimeError('evidence file changed while opening')
        while total <= size:
            part = stream.read(min(65536, size + 1 - total))
            if not part:
                break
            total += len(part)
            if total > size:
                raise RuntimeError('evidence file exceeded its byte bound')
            hasher.update(part)
        after = os.fstat(stream.fileno())
    if (total != size or (after.st_dev, after.st_ino, after.st_size) != (*expected_identity, size)
            or _identity(path) != expected_identity):
        raise RuntimeError('evidence file changed during readback')
    return hasher.hexdigest()


class PrivateRows:
    """Single-writer, create-only evidence with a hard sixteen-row bound."""

    def __init__(self, directory):
        directory = Path(directory)
        if not directory.is_absolute() or directory.resolve(strict=True) != directory:
            raise ValueError('an existing canonical absolute private directory is required')
        self._directory, self._directory_id = directory, _identity(directory, directory=True)
        if any(directory.iterdir()):
            raise ValueError('the private evidence directory must be empty')
        self._lock = threading.Lock()
        self._state = 'open'
        self._rows = []
        self._pending = self._failure = self._failure_write_error = None

    def _directory_check(self):
        if _identity(self._directory, directory=True) != self._directory_id:
            raise RuntimeError('private evidence directory changed')

    def _publish(self, name, data, *, limit):
        """Publish one verified file; retain a `.partial` file on error."""
        if (type(name) is not str or Path(name).name != name or not name
                or len(data) > limit):
            raise ValueError('invalid evidence filename or byte length')
        self._directory_check()
        final, partial = self._directory / name, self._directory / (name + '.partial')
        if os.path.lexists(final):
            raise FileExistsError(final)
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
        self._pending['open_fd'] = fd
        digest = hashlib.sha256()
        write_error = None
        try:
            opened = os.fstat(fd)
            identity = (opened.st_dev, opened.st_ino)
            if _identity(partial) != identity:
                raise RuntimeError('partial evidence identity changed')
            _write_all(fd, data, digest)
            os.fsync(fd)
        except BaseException as error:
            write_error = error
            raise
        finally:
            try:
                os.close(fd)
            except BaseException as close_error:
                self._pending.setdefault('uncertain_closes', []).append((fd, close_error))
                if write_error is None or not isinstance(close_error, Exception):
                    raise
            else:
                self._pending['open_fd'] = None
        if _readback(partial, identity, len(data)) != digest.hexdigest():
            raise RuntimeError('partial evidence digest differs from written bytes')
        self._directory_check()
        if os.name == 'nt':
            os.rename(partial, final)  # Windows refuses an existing destination.
        else:
            os.link(partial, final)  # No-clobber publication on POSIX.
            os.unlink(partial)
        if _identity(final) != identity or _readback(final, identity, len(data)) != digest.hexdigest():
            raise RuntimeError('published evidence differs from readback')
        self._directory_check()
        return digest.hexdigest()

    def _json(self, name, value):
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        return self._publish(name, memoryview(raw), limit=MAX_METADATA_BYTES)

    def _record_failure(self, error, stage):
        self._state = 'failed'
        self._failure = error
        try:
            self._json('partial-manifest.json', {'schema': 'grouped-private-row-failure-v1',
                'status': 'incomplete', 'stage': stage, 'rows': list(self._rows),
                'pending': self._pending.get('binding'),
                'pending_stage': self._pending.get('stage'),
                'exception_type': type(error).__name__[:80]})
        except BaseException as failure_error:
            self._failure_write_error = failure_error
            if not isinstance(failure_error, Exception):
                raise

    def append(self, *, mode, team_id, job_id, group_id, row_index, row):
        """Save one CPU tensor using the same logical job ID in both modes."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('private row writes must be serialized')
        try:
            if self._state != 'open':
                raise RuntimeError('private row writer is not open')
            self._state = 'writing'
            self._pending = {'row': row, 'array': None, 'view': None, 'open_fd': None,
                             'binding': None, 'stage': 'validation'}
            try:
                if (mode not in ('serial', 'grouped') or any(type(value) is not str or _NAME.fullmatch(value) is None
                        for value in (team_id, job_id))
                        or (mode == 'serial' and group_id is not None)
                        or (mode == 'grouped' and (type(group_id) is not str or _NAME.fullmatch(group_id) is None))
                        or type(row_index) is not int or row_index not in (0, 1)
                        or (mode == 'serial' and row_index != 0)
                        or len(self._rows) >= MAX_ROWS):
                    raise ValueError('invalid bounded row binding')
                self._pending['binding'] = {'mode': mode, 'team_id': team_id, 'job_id': job_id,
                                            'group_id': group_id, 'row_index': row_index,
                                            'ordinal': len(self._rows)}
                if (sum(item['mode'] == mode for item in self._rows) >= ROWS_PER_MODE
                        or any(item['mode'] == mode and item['job_id'] == job_id for item in self._rows)
                        or (mode == 'grouped' and any(item['mode'] == mode and item['group_id'] == group_id
                               and item['row_index'] == row_index for item in self._rows))):
                    raise ValueError('duplicate or excess row binding')
                if sys.byteorder != 'little':
                    raise RuntimeError('the frozen row format requires a little-endian host')
                import torch
                if (type(row) is not torch.Tensor or row.device.type != 'cpu'
                        or row.dtype is not torch.float32 or row.layout is not torch.strided
                        or tuple(row.shape) != ROW_SHAPE or not row.is_contiguous()
                        or row.requires_grad or row.grad_fn is not None
                        or row.is_conj() or row.is_neg() or row.numel() * row.element_size() != ROW_BYTES):
                    raise ValueError('row must be an ordinary contiguous detached CPU float32 tensor')
                array = row.numpy()
                self._pending['array'] = array
                if not array.flags.c_contiguous or array.nbytes != ROW_BYTES or array.dtype.str not in ('<f4', '=f4'):
                    raise ValueError('row byte layout differs from the frozen format')
                view = memoryview(array).cast('B')
                self._pending['view'] = view
                ordinal = len(self._rows)
                name = f'row-{ordinal:02d}.f32le'
                self._pending['stage'] = 'row-data'
                row_sha = self._publish(name, view, limit=ROW_BYTES)
                if len(view) != ROW_BYTES:
                    raise RuntimeError('row length changed while writing')
                metadata = {'schema': 'grouped-private-row-v1', 'ordinal': ordinal,
                    'mode': mode, 'team_id': team_id, 'job_id': job_id, 'group_id': group_id,
                    'row_index': row_index, 'shape': list(ROW_SHAPE), 'dtype': 'float32-le',
                    'bytes': ROW_BYTES, 'file': name, 'sha256': row_sha}
                self._pending['stage'] = 'row-metadata'
                metadata_sha = self._json(f'row-{ordinal:02d}.json', metadata)
                self._rows.append(dict(metadata, metadata_sha256=metadata_sha))
            except BaseException as error:
                self._record_failure(error, 'append')
                raise
            self._pending = None  # The caller must still own and retain its result.
            self._state = 'open'
            return dict(self._rows[-1], shape=list(self._rows[-1]['shape']))
        finally:
            self._lock.release()

    def finish(self):
        """Write one manifest; incomplete/mismatched row sets remain explicit."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('private row writes must be serialized')
        try:
            if self._state != 'open':
                raise RuntimeError('private row writer is not open')
            self._state = 'sealing'
            self._pending = {'open_fd': None}
            try:
                serial = {r['job_id']: r['team_id'] for r in self._rows if r['mode'] == 'serial'}
                grouped = {r['job_id']: r['team_id'] for r in self._rows if r['mode'] == 'grouped'}
                group_offsets = {}
                for row in self._rows:
                    if row['mode'] == 'grouped':
                        group_offsets.setdefault(row['group_id'], []).append(row['row_index'])
                complete = (len(serial) == ROWS_PER_MODE and len(grouped) == ROWS_PER_MODE
                            and serial == grouped and len(group_offsets) == ROWS_PER_MODE // 2
                            and all(sorted(offsets) == [0, 1] for offsets in group_offsets.values()))
                manifest = {'schema': 'grouped-private-rows-manifest-v1',
                    'row_set_complete': complete,
                    'rows': [dict(row, shape=list(row['shape'])) for row in self._rows],
                    'limitations': ['bytes and recorded identities only; caller authenticates owner and source',
                                    'compare supplied group and row bindings with observed selected members',
                                    'caller retains host/disk claims and prevents concurrent alias mutation',
                                    'numerical validity, routing and tolerance require independent analysis']}
                digest = self._json('manifest.json', manifest)
            except BaseException as error:
                self._record_failure(error, 'finish')
                raise
            self._pending = None
            self._state = 'finished'
            return manifest, digest
        finally:
            self._lock.release()
