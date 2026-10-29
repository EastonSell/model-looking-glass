"""Read-only check of an existing, independently anchored injection claim.

The caller must establish and publish the trusted anchor before claiming. This
module never creates a claim, draws a nonce, attests the anchor, or enforces a
filesystem quota. It assumes cooperating processes and a protected local DB.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat


_MIB = 1024 ** 2
_PHASES = {'natural_discovery': (128 * _MIB, 4096 * _MIB),
           'natural_heldout': (160 * _MIB, 224 * _MIB),
           'injection_discovery': (48 * _MIB, 128 * _MIB),
           'injection_heldout': (72 * _MIB, 192 * _MIB)}
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_MAX_META = 16 * _MIB
_MAX_HISTORY = 32 * _MIB
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


class PreclaimError(ValueError):
    """The anchored claim is absent, changed or cannot be verified."""


@dataclass(frozen=True)
class ClaimAnchor:
    ledger_path: str
    db_dev: int
    db_ino: int
    phase_path: str
    state_sha256: str
    history_sha256: str
    scope_sha256: str
    original_claims_sha256: str
    heldout_claim_sha256: str


def _digest(value):
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise PreclaimError('the ledger anchor is invalid')
    return value


def _path(value):
    if not isinstance(value, (str, os.PathLike)):
        raise PreclaimError('the anchored paths are invalid')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise PreclaimError('the anchored paths must be absolute and normalized')
    return os.path.normcase(os.path.abspath(str(path)))


def _database_identity(path):
    item = path.lstat()
    if (not stat.S_ISREG(item.st_mode) or path.is_symlink() or item.st_nlink != 1
            or not item.st_ino or getattr(item, 'st_file_attributes', 0) & _REPARSE):
        raise PreclaimError('the anchored database identity is invalid')
    return item.st_dev, item.st_ino


def _snapshot_hash(meta, inventory):
    digest = hashlib.sha256()
    for item in (meta.encode('utf-8'), inventory.encode('utf-8')):
        digest.update(len(item).to_bytes(8, 'big'))
        digest.update(item)
    return digest.hexdigest()


def _history_hash(db):
    digest = hashlib.sha256()
    total = 0
    raw_total = 0
    for table, query, fields in (
        ('reconciliations', 'SELECT seq,manifest_sha,gap_sha,claims_sha,previous_meta,previous_inventory '
                            'FROM reconciliations ORDER BY seq',
         ('seq', 'manifest_sha', 'gap_sha', 'claims_sha', 'previous_meta', 'previous_inventory')),
        ('retired_evidence', 'SELECT side,path_sha,size,content_sha FROM retired_evidence '
                             'ORDER BY side,path_sha,size,content_sha',
         ('side', 'path_sha', 'size', 'content_sha')),
    ):
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None
        digest.update((table + (':present\n' if exists else ':absent\n')).encode('ascii'))
        if exists:
            size_expr = ' + '.join(f'COALESCE(LENGTH(CAST({field} AS BLOB)),0)' for field in fields)
            count, raw_bytes = db.execute(
                f'SELECT COUNT(*), COALESCE(SUM({size_expr}),0) FROM {table}').fetchone()
            if (type(count) is not int or count > 65536 or type(raw_bytes) is not int
                    or raw_bytes < 0 or raw_total + raw_bytes > _MAX_HISTORY):
                raise PreclaimError('ledger history exceeds the bound')
            raw_total += raw_bytes
            count = 0
            for row in db.execute(query):
                count += 1
                raw = json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')
                total += len(raw)
                if count > 65536 or total > _MAX_HISTORY:
                    raise PreclaimError('ledger history exceeds the bound')
                digest.update(len(raw).to_bytes(8, 'big'))
                digest.update(raw)
    return digest.hexdigest()


@contextmanager
def _read_only_connection(path, *, exclusive=False):
    mode = 'rw' if exclusive else 'ro'
    db = sqlite3.connect(path.as_uri() + '?mode=' + mode, uri=True, timeout=30, isolation_level=None)
    try:
        yield db
    except BaseException as primary:
        try:
            db.close()
        except BaseException as cleanup:
            cancellation = (KeyboardInterrupt, SystemExit)
            if isinstance(primary, cancellation):
                raise primary from cleanup
            if isinstance(cleanup, cancellation):
                raise cleanup from primary
            raise primary from cleanup
        raise
    else:
        db.close()


@contextmanager
def _verified_existing_claim(ledger_path, phase, *, output_path, anchor, exclusive):
    """Validate a consumed phase row and yield while its transaction is open.

    ``anchor`` must come from the caller's independently verified protected
    configuration, not from this database. The caller also checks ACLs,
    ancestors, free space, side evidence and clean source-first startup. The
    held-out phase row is not a process lease or permission for another draw.
    """
    try:
        if (type(anchor) is not ClaimAnchor or phase != 'injection_heldout'
                or type(anchor.db_dev) is not int or type(anchor.db_ino) is not int
                or anchor.db_dev < 0 or anchor.db_ino <= 0
                or _path(ledger_path) != _path(anchor.ledger_path)
                or _path(output_path) != _path(anchor.phase_path)):
            raise PreclaimError('the claim does not match the protected anchor')
        for value in (anchor.state_sha256, anchor.history_sha256, anchor.scope_sha256,
                      anchor.original_claims_sha256, anchor.heldout_claim_sha256):
            _digest(value)
        ledger = Path(ledger_path)
        if (_database_identity(ledger) != (anchor.db_dev, anchor.db_ino)
                or _path(ledger.resolve(strict=True)) != _path(anchor.ledger_path)):
            raise PreclaimError('the protected database differs from its anchor')
        with _read_only_connection(ledger, exclusive=exclusive) as db:
            if exclusive:
                db.execute('BEGIN IMMEDIATE')
                db.execute('PRAGMA query_only=ON')
            else:
                db.execute('PRAGMA query_only=ON')
                db.execute('BEGIN')
            length = db.execute('SELECT LENGTH(CAST(data AS BLOB)) + LENGTH(CAST(inventory AS BLOB)) '
                                'FROM meta WHERE id=1').fetchone()
            if length is None or type(length[0]) is not int or not 0 < length[0] <= _MAX_META:
                raise PreclaimError('ledger metadata exceeds the bound')
            record = db.execute('SELECT data, inventory FROM meta WHERE id=1').fetchone()
            if record is None or any(type(item) is not str for item in record):
                raise PreclaimError('ledger metadata is invalid')
            meta, inventory = record
            value = json.loads(meta)
            if (type(value) is not dict or value.get('schema') != 'mlg-agreement-phase-ledger-v6'
                    or value.get('evidence_scope_amendment_sha256') != anchor.scope_sha256
                    or value.get('current_history_sha256') != anchor.history_sha256
                    or _snapshot_hash(meta, inventory) != anchor.state_sha256
                    or _history_hash(db) != anchor.history_sha256):
                raise PreclaimError('the reviewed ledger state or history changed')
            rows = db.execute('SELECT CASE WHEN LENGTH(CAST(phase AS BLOB)) <= 128 THEN phase END, '
                              'CASE WHEN LENGTH(CAST(output AS BLOB)) <= 32768 THEN output END, '
                              'logical, disk, free FROM claims ORDER BY phase LIMIT 5').fetchall()
            if (not 1 <= len(rows) <= len(_PHASES) or len({row[0] for row in rows}) != len(rows) or any(
                    type(name) is not str or name not in _PHASES or type(path) is not str
                    or not Path(path).is_absolute() or type(logical) is not int or type(disk) is not int
                    or (logical, disk) != _PHASES[name] or type(free) is not int or free < 0
                    for name, path, logical, disk, free in rows)):
                raise PreclaimError('the phase claim table is invalid')
            original = [row for row in rows if row[0] in ('natural_discovery', 'injection_discovery')]
            original_raw = json.dumps(sorted(original), ensure_ascii=True, separators=(',', ':')).encode('utf-8')
            if (len(original) != 2 or hashlib.sha256(original_raw).hexdigest() != anchor.original_claims_sha256):
                raise PreclaimError('an original discovery claim changed')
            heldout = [row for row in rows if row[0] == 'injection_heldout']
            if (len(heldout) != 1 or _path(heldout[0][1]) != _path(anchor.phase_path)
                    or heldout[0][4] < 12 * 1024 ** 3
                    or hashlib.sha256(json.dumps(heldout[0], ensure_ascii=True,
                        separators=(',', ':')).encode('ascii')).hexdigest() != anchor.heldout_claim_sha256):
                raise PreclaimError('the one-use injection claim is absent or differs')
            if any(free < 12 * 1024 ** 3 for name, _, _, _, free in rows
                   if name in ('natural_heldout', 'injection_heldout')):
                raise PreclaimError('a held-out claim has invalid recorded free space')
            paths = [_path(row[1]) for row in rows]
            for index, left in enumerate(paths):
                for right in paths[index + 1:]:
                    try:
                        common = os.path.commonpath((left, right))
                    except ValueError:
                        continue
                    if common in (left, right):
                        raise PreclaimError('claim output paths overlap')
            if _database_identity(ledger) != (anchor.db_dev, anchor.db_ino):
                raise PreclaimError('the protected database identity changed before authorization')
            yield True
            if _database_identity(ledger) != (anchor.db_dev, anchor.db_ino):
                raise PreclaimError('the protected database identity changed before release')
            db.execute('ROLLBACK')
        if _database_identity(ledger) != (anchor.db_dev, anchor.db_ino):
            raise PreclaimError('the protected database identity changed during verification')
    except PreclaimError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, RecursionError):
        raise PreclaimError('the protected phase claim could not be verified') from None


def verify_existing_claim(ledger_path, phase, *, output_path, anchor):
    """Perform a read-only point-in-time check of the anchored consumed row."""
    with _verified_existing_claim(ledger_path, phase, output_path=output_path,
                                  anchor=anchor, exclusive=False) as verified:
        return verified


def hold_existing_claim(ledger_path, phase, *, output_path, anchor):
    """Hold a short verified write reservation through the single nonce call.

    The context exposes no database handle and makes no writes. SQLite v6
    claim/reconciliation transactions cannot commit until this context exits.
    The caller must forbid out-of-band database replacement and must never retry
    an ambiguous or interrupted draw or transaction release.
    """
    return _verified_existing_claim(ledger_path, phase, output_path=output_path,
                                    anchor=anchor, exclusive=True)
