"""Cooperative, process-scoped accounting for a shared local disk volume.

This ledger does not allocate filesystem blocks or control nonparticipants.
All participants for a volume must use the same trusted ledger directory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import threading
import time
import uuid


class DiskLedgerError(RuntimeError):
    """The cooperative ledger or its ownership could not be verified."""


class DiskCapacityError(DiskLedgerError):
    """The live free-space observation does not cover the claims and cushion."""


class DiskOwnershipUncertain(DiskLedgerError):
    """A stale, missing, or ambiguous owner requires reconciliation."""


@dataclass(frozen=True)
class DiskClaim:
    claim_id: str
    owner_pid: int
    output_bytes: int
    staging_bytes: int
    status: str


@dataclass(frozen=True)
class DiskSnapshot:
    free_bytes: int
    committed_bytes: int
    cushion_bytes: int
    claims: tuple[DiskClaim, ...]


_MAX = 2**63 - 1
_MAX_ACTIVE = 1024
_ROOT_LOCK = threading.Lock()
_ROOTS: dict[str, 'DiskReservation'] = {}
_SCHEMA = 1


def _after_fork_child():
    try:
        for lease in tuple(_ROOTS.values()):
            fd, lease._fd = lease._fd, None
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        _ROOTS.clear()
    finally:
        _ROOT_LOCK.release()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=_ROOT_LOCK.acquire, after_in_parent=_ROOT_LOCK.release,
                        after_in_child=_after_fork_child)


def _cleanup_then_raise(primary, cleanup):
    """Preserve a cancellation if cleanup also fails, without dropping either error."""
    try:
        cleanup()
    except BaseException as secondary:
        if isinstance(primary, Exception) and not isinstance(secondary, Exception):
            raise secondary from primary
        raise primary from secondary
    raise primary


def _rollback_and_close(connection):
    try:
        if connection.in_transaction:
            connection.rollback()
    except BaseException as error:
        _cleanup_then_raise(error, connection.close)
    connection.close()


def _bytes(value, *, allow_zero=True):
    if type(value) is not int or value < (0 if allow_zero else 1) or value > _MAX:
        raise ValueError('disk byte counts must be bounded nonnegative integers')
    return value


def _add(left, right):
    if left > _MAX - right:
        raise DiskLedgerError('disk accounting exceeds the supported integer range')
    return left + right


def _identity(path):
    info = os.stat(path)
    if not info.st_ino:
        raise DiskLedgerError('filesystem identity is unavailable')
    return f'{info.st_dev}:{info.st_ino}'


def _volume_identity(path):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetVolumePathNameW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        kernel.GetVolumePathNameW.restype = wintypes.BOOL
        kernel.GetVolumeNameForVolumeMountPointW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        kernel.GetVolumeNameForVolumeMountPointW.restype = wintypes.BOOL
        kernel.GetVolumeInformationW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD)
        kernel.GetVolumeInformationW.restype = wintypes.BOOL
        root, name = ctypes.create_unicode_buffer(1024), ctypes.create_unicode_buffer(1024)
        serial = wintypes.DWORD()
        if (not kernel.GetVolumePathNameW(str(path), root, len(root))
                or not kernel.GetVolumeNameForVolumeMountPointW(root.value, name, len(name))
                or not kernel.GetVolumeInformationW(root.value, None, 0, ctypes.byref(serial),
                                                    None, None, None, 0)):
            raise DiskLedgerError('local volume identity is unavailable')
        if not name.value.lower().startswith('\\\\?\\volume{'):
            raise DiskLedgerError('a local volume GUID is required')
        return f'win:{name.value.lower()}:{serial.value}'
    if os.name == 'posix':
        return f'posix:{os.stat(path).st_dev}'
    raise DiskLedgerError('unsupported filesystem platform')


def _try_lock(fd):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        import msvcrt
        class Overlapped(ctypes.Structure):
            _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                        ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD),
                        ('hEvent', wintypes.HANDLE)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.LockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped))
        kernel.LockFileEx.restype = wintypes.BOOL
        overlapped = Overlapped()
        if kernel.LockFileEx(msvcrt.get_osfhandle(fd), 3, 0, 1, 0, ctypes.byref(overlapped)):
            return True
        error = ctypes.get_last_error()
        if error == 33:  # ERROR_LOCK_VIOLATION
            return False
        raise OSError(error, 'owner lock status is unavailable')
    if os.name == 'posix':
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
    raise DiskLedgerError('unsupported filesystem platform')


def _file_identity(fd, path):
    opened, named = os.fstat(fd), os.stat(path)
    if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode)
            or not opened.st_ino or not named.st_ino
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)):
        raise DiskOwnershipUncertain('owner lock identity is missing or changed')
    return f'{opened.st_dev}:{opened.st_ino}'


def _resolve_directory(path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_dir():
        raise DiskLedgerError('ledger and target paths must be existing directories')
    return resolved


def _require_real_path(path, *, directory):
    info = os.lstat(path)
    reparse = getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    if (reparse or stat.S_ISLNK(info.st_mode)
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise DiskLedgerError('ledger and owner paths must not be links or reparse points')


def _connect(path):
    connection = sqlite3.connect(str(path), timeout=5, isolation_level=None)
    try:
        connection.execute('PRAGMA synchronous=FULL')
        if connection.execute('PRAGMA journal_mode').fetchone() != ('delete',):
            raise DiskLedgerError('the disk ledger requires a rollback journal')
        return connection
    except BaseException as error:
        _cleanup_then_raise(error, connection.close)


class DiskLedger:
    """One fixed-cushion ledger in a shared directory on the target volume."""

    def __init__(self, shared_dir, *, target_dir, cushion_bytes):
        self.shared_dir = _resolve_directory(shared_dir)
        self.target_dir = _resolve_directory(target_dir)
        self.cushion_bytes = _bytes(cushion_bytes)
        self._volume = _volume_identity(self.target_dir)
        if _volume_identity(self.shared_dir) != self._volume:
            raise DiskLedgerError('ledger and target must be on the same volume')
        self._owners = self.shared_dir / '.disk-reservation-owners'
        self._owners.mkdir(exist_ok=True)
        _require_real_path(self._owners, directory=True)
        if _volume_identity(_resolve_directory(self._owners)) != self._volume:
            raise DiskLedgerError('owner lock directory must be on the same volume')
        self._owners_identity = _identity(self._owners)
        self._db = self.shared_dir / '.disk-reservations.sqlite3'
        if os.path.lexists(self._db):
            _require_real_path(self._db, directory=False)
        connection = _connect(self._db)
        try:
            _require_real_path(self._db, directory=False)
            connection.execute('BEGIN IMMEDIATE')
            tables = {r[0] for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if not tables:
                connection.execute('CREATE TABLE metadata (version INTEGER, volume TEXT, cushion INTEGER, owners TEXT)')
                connection.execute('CREATE TABLE claims (claim_id TEXT PRIMARY KEY, pid INTEGER, output INTEGER, '
                    'staging INTEGER, lock_identity TEXT, state TEXT, created_ns INTEGER, ended_ns INTEGER, reason TEXT)')
                connection.execute('INSERT INTO metadata VALUES (?, ?, ?, ?)',
                    (_SCHEMA, self._volume, self.cushion_bytes, self._owners_identity))
            elif tables != {'metadata', 'claims'}:
                raise DiskLedgerError('unrecognized disk ledger')
            self._validate_meta(connection)
            self._db_identity = _identity(self._db)
            connection.commit()
        except BaseException as error:
            _cleanup_then_raise(error, lambda: _rollback_and_close(connection))
        else:
            connection.close()

    def _validate_meta(self, connection):
        rows = connection.execute('SELECT version, volume, cushion, owners FROM metadata').fetchall()
        if (len(rows) != 1 or type(rows[0][0]) is not int or type(rows[0][1]) is not str
                or type(rows[0][2]) is not int or type(rows[0][3]) is not str
                or rows[0] != (_SCHEMA, self._volume, self.cushion_bytes, self._owners_identity)):
            raise DiskLedgerError('disk ledger volume, cushion or owner directory does not match')

    def _check_paths(self, target_dir):
        target = _resolve_directory(self.target_dir if target_dir is None else target_dir)
        _require_real_path(self._owners, directory=True)
        _require_real_path(self._db, directory=False)
        if (_volume_identity(target) != self._volume or _volume_identity(self.shared_dir) != self._volume
                or _volume_identity(_resolve_directory(self._owners)) != self._volume
                or _identity(self._owners) != self._owners_identity
                or _identity(self._db) != self._db_identity):
            raise DiskLedgerError('disk volume or ledger identity changed')
        return target

    @contextmanager
    def _transaction(self, target_dir=None):
        target = self._check_paths(target_dir)
        connection = _connect(self._db)
        try:
            connection.execute('BEGIN IMMEDIATE')
            self._check_paths(target)
            self._validate_meta(connection)
            yield connection, target
            connection.commit()
        except BaseException as error:
            _cleanup_then_raise(error, lambda: _rollback_and_close(connection))
        else:
            connection.close()

    def _owner_path(self, claim_id):
        if type(claim_id) is not str or len(claim_id) != 32 or any(c not in '0123456789abcdef' for c in claim_id):
            raise DiskLedgerError('invalid disk claim identity')
        return self._owners / (claim_id + '.lock')

    def _open_owner(self, claim_id, expected):
        path = self._owner_path(claim_id)
        fd = os.open(path, os.O_RDWR | getattr(os, 'O_BINARY', 0))
        try:
            if _file_identity(fd, path) != expected:
                raise DiskOwnershipUncertain('owner lock file was replaced')
            return fd
        except BaseException as error:
            _cleanup_then_raise(error, lambda: os.close(fd))

    def _probe_owner(self, claim_id, expected, *, allow_existing=False, on_stale=None):
        # The POSIX at-fork hook takes this same lock, so a child cannot
        # inherit a successful probe before its close and marker decision.
        with _ROOT_LOCK:
            return self._probe_owner_locked(claim_id, expected, allow_existing=allow_existing,
                                            on_stale=on_stale)

    def _probe_owner_locked(self, claim_id, expected, *, allow_existing=False, on_stale=None):
        # A marker persists across an interrupted probe. Without it, a leaked
        # probe lock could masquerade as the original owner's live lock.
        marker = self._owners / (claim_id + '.probe')
        try:
            marker_fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                | getattr(os, 'O_BINARY', 0), 0o600)
        except FileExistsError:
            if not allow_existing:
                return 'unknown'
            created = False
        else:
            created = True
            os.close(marker_fd)
        fd = None
        try:
            fd = self._open_owner(claim_id, expected)
            acquired = _try_lock(fd)
            if acquired and on_stale is not None:
                on_stale()
            closing, fd = fd, None
            os.close(closing)
            if created or (allow_existing and acquired):
                os.unlink(marker)
            return 'stale' if acquired else 'held'
        except BaseException as error:
            if fd is not None:
                closing, fd = fd, None
                _cleanup_then_raise(error, lambda: os.close(closing))
            raise

    def _owner_status(self, claim_id, expected):
        try:
            return self._probe_owner(claim_id, expected)
        except (OSError, DiskLedgerError):
            return 'unknown'

    def _snapshot_locked(self, connection, target):
        rows = connection.execute('SELECT claim_id, pid, output, staging, lock_identity, state FROM claims')
        claims, total = [], 0
        for claim_id, pid, output, staging, identity, state in rows:
            self._owner_path(claim_id)
            if (type(pid) is not int or pid <= 0 or type(identity) is not str or not identity
                    or type(output) is not int or type(staging) is not int
                    or state not in ('active', 'released', 'reconciled')):
                raise DiskLedgerError('invalid persisted disk claim')
            amount = _add(_bytes(output), _bytes(staging))
            if not amount:
                raise DiskLedgerError('invalid persisted disk claim')
            if state != 'active':
                continue
            if len(claims) >= _MAX_ACTIVE:
                raise DiskLedgerError('too many active disk claims')
            total = _add(total, amount)
            claims.append(DiskClaim(claim_id, pid, output, staging, self._owner_status(claim_id, identity)))
        free = _bytes(shutil.disk_usage(target).free)
        return DiskSnapshot(free, total, self.cushion_bytes, tuple(claims))

    def snapshot(self, *, target_dir=None):
        with self._transaction(target_dir) as (connection, target):
            return self._snapshot_locked(connection, target)

    def reserve(self, *, output_bytes, staging_bytes=0, target_dir=None, staging_dir=None):
        output_bytes, staging_bytes = _bytes(output_bytes), _bytes(staging_bytes)
        requested = _add(output_bytes, staging_bytes)
        if requested == 0:
            raise ValueError('a disk claim must have a positive total')
        output_path = _resolve_directory(self.target_dir if target_dir is None else target_dir)
        staging_path = _resolve_directory(output_path if staging_dir is None else staging_dir)
        claim_id, pid = uuid.uuid4().hex, os.getpid()
        path = self._owner_path(claim_id)
        with _ROOT_LOCK:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
            lease = None
            try:
                os.set_inheritable(fd, False)
                os.write(fd, b'0')
                if not _try_lock(fd):
                    raise DiskOwnershipUncertain('new owner lock is unexpectedly held')
                identity = _file_identity(fd, path)
                lease = DiskReservation(self, claim_id, pid, output_bytes, staging_bytes,
                                        identity, fd, output_path, staging_path)
                _ROOTS[claim_id] = lease
            except BaseException as error:
                def discard_unregistered():
                    if lease is not None:
                        lease._fd = None
                        _ROOTS.pop(claim_id, None)
                    os.close(fd)
                _cleanup_then_raise(error, discard_unregistered)
        inserted = False
        try:
            with self._transaction(output_path) as (connection, target):
                self._check_paths(staging_path)
                snap = self._snapshot_locked(connection, target)
                if any(claim.status != 'held' for claim in snap.claims):
                    raise DiskOwnershipUncertain('a disk owner is stale or unknown')
                if len(snap.claims) >= _MAX_ACTIVE:
                    raise DiskLedgerError('too many active disk claims')
                required = _add(_add(snap.committed_bytes, requested), self.cushion_bytes)
                if snap.free_bytes < required:
                    raise DiskCapacityError('free disk is below the cooperative claims and cushion')
                connection.execute('INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)',
                    (claim_id, pid, output_bytes, staging_bytes, identity, 'active', time.time_ns()))
                inserted = True
            lease._state = 'active'
            return lease
        except BaseException as error:
            if not inserted:
                _cleanup_then_raise(error, lease._close_handle)
            raise

    def reconcile_stale(self, claim_id, *, reason):
        self._owner_path(claim_id)
        if type(reason) is not str or not reason.strip() or len(reason) > 512:
            raise ValueError('an explicit bounded reconciliation reason is required')
        with self._transaction() as (connection, _target):
            row = connection.execute('SELECT lock_identity, state FROM claims WHERE claim_id=?',
                                     (claim_id,)).fetchone()
            if row is None or row[1] != 'active':
                raise DiskLedgerError('no active disk claim matches')
            try:
                def mark_reconciled():
                    connection.execute("UPDATE claims SET state='reconciled', ended_ns=?, reason=? WHERE claim_id=?",
                                       (time.time_ns(), reason, claim_id))
                status = self._probe_owner(claim_id, row[0], allow_existing=True,
                                           on_stale=mark_reconciled)
            except OSError as error:
                raise DiskOwnershipUncertain('owner lock cannot be verified') from error
            if status != 'stale':
                raise DiskOwnershipUncertain('owner lock is still held or unknown')


class DiskReservation:
    """A process-owned claim; release only after all associated writers stop."""

    def __init__(self, ledger, claim_id, pid, output, staging, identity, fd, output_path, staging_path):
        self._ledger, self.claim_id, self._pid = ledger, claim_id, pid
        self._output, self._staging, self._identity, self._fd = output, staging, identity, fd
        self._output_path, self._staging_path = output_path, staging_path
        self._state = 'starting'
        self._lock = threading.RLock()

    def _owned(self):
        if os.getpid() != self._pid or self._fd is None:
            raise DiskOwnershipUncertain('disk claim does not belong to this process')
        path = self._ledger._owner_path(self.claim_id)
        if _file_identity(self._fd, path) != self._identity:
            raise DiskOwnershipUncertain('disk claim lock identity changed')
        if self._ledger._owner_status(self.claim_id, self._identity) != 'held':
            raise DiskOwnershipUncertain('disk claim lock is not verifiably held')

    def check(self, *, target_dir=None, staging_dir=None):
        if os.getpid() != self._pid:
            raise DiskOwnershipUncertain('disk claim does not belong to this process')
        with self._lock:
            if self._state != 'active':
                raise DiskLedgerError('disk claim is not active')
            self._owned()
            output_path = self._output_path if target_dir is None else target_dir
            staging_path = self._staging_path if staging_dir is None else staging_dir
            with self._ledger._transaction(output_path) as (connection, target):
                self._ledger._check_paths(staging_path)
                snap = self._ledger._snapshot_locked(connection, target)
                if (not any(claim.claim_id == self.claim_id and claim.status == 'held' for claim in snap.claims)
                        or any(claim.status != 'held' for claim in snap.claims)):
                    raise DiskOwnershipUncertain('a disk owner is stale, missing or unknown')
                if snap.free_bytes < _add(snap.committed_bytes, snap.cushion_bytes):
                    raise DiskCapacityError('free disk is below the cooperative claims and cushion')
                return snap

    def _close_handle(self):
        with _ROOT_LOCK:
            fd, self._fd = self._fd, None
            _ROOTS.pop(self.claim_id, None)
            if fd is not None:
                os.close(fd)

    def release(self):
        if os.getpid() != self._pid:
            raise DiskOwnershipUncertain('disk claim does not belong to this process')
        with self._lock:
            if self._state == 'released':
                return
            if self._state not in ('active', 'releasing'):
                raise DiskLedgerError('disk claim is not active')
            self._state = 'releasing'  # No further writes are authorized after this point.
            self._owned()
            with self._ledger._transaction() as (connection, _target):
                row = connection.execute('SELECT pid, output, staging, lock_identity, state FROM claims '
                                         'WHERE claim_id=?', (self.claim_id,)).fetchone()
                if row is None or row[:4] != (self._pid, self._output, self._staging, self._identity):
                    raise DiskOwnershipUncertain('disk claim identity changed')
                if row[4] == 'active':
                    connection.execute("UPDATE claims SET state='released', ended_ns=? WHERE claim_id=?",
                                       (time.time_ns(), self.claim_id))
                elif row[4] != 'released':
                    raise DiskOwnershipUncertain('disk claim is not owned')
            self._close_handle()
            self._state = 'released'
