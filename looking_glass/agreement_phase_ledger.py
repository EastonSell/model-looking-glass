"""One-use, cooperative reservations for the pretrained agreement study.

The custodian supplies a complete inventory of *study* evidence.  This module
does not authenticate that inventory or enforce a producer's per-phase limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
from types import MappingProxyType
from typing import Iterable

_MIB = 1024 * 1024
STUDY_LOGICAL_LIMIT = 512 * _MIB
ORIGINAL_DISK_LIMIT = 1024 * _MIB
PREVIOUS_DISK_LIMIT = 2048 * _MIB
PREVIOUS_FREE_DISK_BYTES = 3072 * _MIB
STUDY_DISK_LIMIT = 4096 * _MIB
REQUIRED_FREE_DISK_BYTES = 6144 * _MIB
LEDGER_DISK_RESERVE = 64 * _MIB
PREFLIGHT_BYTES = 3_216_448
PREFLIGHT_SHA256 = '7c262f3fe81236888a8f5fdb6d31f7835666674be9ffbf36f8ce2ca6a375eb6c'
PHASE_LIMITS = MappingProxyType({
    'natural_discovery': (128 * _MIB, 1024 * _MIB),
    'natural_heldout': (160 * _MIB, 224 * _MIB),
    'injection_discovery': (48 * _MIB, 128 * _MIB),
    'injection_heldout': (72 * _MIB, 192 * _MIB),
})
ORIGINAL_PHASE_DISK = MappingProxyType({
    'natural_discovery': 224 * _MIB,
    'natural_heldout': 224 * _MIB,
    'injection_discovery': 128 * _MIB,
    'injection_heldout': 192 * _MIB,
})
PREVIOUS_PHASE_DISK = MappingProxyType({
    **ORIGINAL_PHASE_DISK,
    'natural_discovery': 512 * _MIB,
})
_SCHEMA = 'mlg-agreement-phase-ledger-v4'
_MAX_FILES = 8192
_MAX_INVENTORY = 8 * _MIB  # Combined UTF-8 bytes of meta and inventory.
_MAX_JSON = 4 * _MIB


class LedgerError(RuntimeError):
    """No usable claim was returned. A committed claim is never refunded."""


@dataclass(frozen=True)
class PhaseGrant:
    phase: str
    output_path: str
    logical_limit_bytes: int
    disk_limit_bytes: int


def _read_json(path: Path):
    with path.open('rb') as stream:
        raw = stream.read(_MAX_JSON + 1)
    if len(raw) > _MAX_JSON:
        raise LedgerError('JSON evidence exceeds the bound')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LedgerError('duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeDecodeError, ValueError) as exc:
        raise LedgerError('invalid JSON evidence') from exc
    if type(value) is not dict:
        raise LedgerError('JSON evidence must be an object')
    return raw, value


def _ordinary(path: Path):
    item = path.lstat()
    # Windows junctions and other reparse points must not leave the inventory.
    if stat.S_ISLNK(item.st_mode) or getattr(item, 'st_file_attributes', 0) & 0x400:
        raise LedgerError('symlinks and reparse points are not evidence')
    return item


def _files(roots: Iterable[Path]):
    """Hash quiescent regular files, charging each distinct pathname once."""
    entries = {}
    pending = list(roots)
    visited = 0
    total_bytes = 0
    while pending:
        path = pending.pop()
        visited += 1
        if visited > _MAX_FILES or len(str(path)) > 2048:
            raise LedgerError('evidence inventory exceeds bounds')
        item = _ordinary(path)
        if stat.S_ISDIR(item.st_mode):
            for child in path.iterdir():
                if len(pending) + visited >= _MAX_FILES:
                    raise LedgerError('evidence inventory exceeds file bound')
                pending.append(child)
            continue
        if not stat.S_ISREG(item.st_mode):
            raise LedgerError('evidence must contain regular files only')
        name = os.path.normcase(str(path.absolute()))
        if name in entries:
            continue
        total_bytes += item.st_size
        if total_bytes > STUDY_DISK_LIMIT:
            raise LedgerError('evidence exceeds the whole-study disk allowance')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            remaining = item.st_size
            while remaining:
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise LedgerError('evidence shortened during inventory')
                digest.update(chunk)
                remaining -= len(chunk)
            if stream.read(1):
                raise LedgerError('evidence grew during inventory')
        after = _ordinary(path)
        if (item.st_size, item.st_mtime_ns, item.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise LedgerError('evidence changed during inventory')
        entries[name] = [item.st_size, digest.hexdigest()]
        if len(entries) > _MAX_FILES:
            raise LedgerError('evidence inventory exceeds file bound')
    return entries


def _path(value, *, existing: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise LedgerError('paths must be absolute')
    if len(str(path)) > 2048:
        raise LedgerError('path exceeds bound')
    if existing:
        # Reject the final component before resolving it.
        _ordinary(path)
        return path.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or path.exists() or path.is_symlink():
        raise LedgerError('output must not exist and its parent must exist')
    return parent / path.name


def _nat(value, name):
    if type(value) is not int or value < 0:
        raise LedgerError(f'{name} must be a nonnegative integer')
    return value


def _digest(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise LedgerError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _fixed_budget(budget):
    if type(budget) is not dict:
        raise LedgerError('missing frozen budget')
    expected = {
        'study_limit_bytes': STUDY_LOGICAL_LIMIT,
        'host_disk_allowance_bytes': ORIGINAL_DISK_LIMIT,
        'prior_unique_tensor_bytes': PREFLIGHT_BYTES,
        'required_free_disk_bytes': ORIGINAL_DISK_LIMIT,
    }
    for key, value in expected.items():
        if type(budget.get(key)) is not int or budget[key] != value:
            raise LedgerError('frozen study budget differs')
    for name, idx in [('phase_reservations_bytes', 0), ('host_disk_phase_reservations_bytes', 1)]:
        actual = budget.get(name)
        wanted = ({phase: limits[idx] for phase, limits in PHASE_LIMITS.items()} if idx == 0
                  else dict(ORIGINAL_PHASE_DISK))
        if type(actual) is not dict or set(actual) != set(wanted):
            raise LedgerError('phase allocations differ')
        if any(type(actual[k]) is not int or actual[k] != v for k, v in wanted.items()):
            raise LedgerError('phase allocations differ')


def _phase_snapshot():
    return {phase: list(limits) for phase, limits in PHASE_LIMITS.items()}


def _check_metadata_bound(meta, inventory):
    if len(meta.encode('utf-8')) + len(inventory.encode('utf-8')) > _MAX_INVENTORY:
        raise LedgerError('combined ledger metadata exceeds the bound')


def _physical_amendment(amendment, claim_hash, binding_hash, *, previous_hash=None):
    previous = previous_hash is None
    version = 'v1' if previous else 'v2'
    if (amendment.get('schema') != 'mlg-agreement-physical-budget-amendment-' + version
            or amendment.get('natural_claim_sha256') != claim_hash
            or amendment.get('natural_binding_sha256') != binding_hash):
        raise LedgerError('physical amendment does not bind the original natural claim')
    if not previous and amendment.get('previous_physical_amendment_sha256') != previous_hash:
        raise LedgerError('physical amendment does not bind its predecessor')
    for key, expected in (
        ('original_host_disk_allowance_bytes', ORIGINAL_DISK_LIMIT),
        ('host_disk_allowance_bytes', PREVIOUS_DISK_LIMIT if previous else STUDY_DISK_LIMIT),
        ('required_free_disk_bytes', PREVIOUS_FREE_DISK_BYTES if previous else REQUIRED_FREE_DISK_BYTES),
        ('study_limit_bytes', STUDY_LOGICAL_LIMIT),
    ):
        if type(amendment.get(key)) is not int or amendment[key] != expected:
            raise LedgerError('physical amendment totals differ')
    for key, index in (('phase_reservations_bytes', 0), ('host_disk_phase_reservations_bytes', 1)):
        table = amendment.get(key)
        wanted = ({phase: limits[index] for phase, limits in PHASE_LIMITS.items()}
                  if index == 0 or not previous else dict(PREVIOUS_PHASE_DISK))
        if type(table) is not dict or set(table) != set(wanted):
            raise LedgerError('physical amendment shares differ')
        if any(type(table[p]) is not int or table[p] != value for p, value in wanted.items()):
            raise LedgerError('physical amendment shares differ')
    if not previous:
        for key, expected in (('metadata_max_bytes', _MAX_INVENTORY),
                              ('ledger_disk_reserve_bytes', LEDGER_DISK_RESERVE)):
            if type(amendment.get(key)) is not int or amendment[key] != expected:
                raise LedgerError('physical amendment metadata bounds differ')


def _check_natural_audit(natural, owner_inventory, audit, binding, natural_files):
    audit_source = _digest(binding.get('independent_natural_audit_source_sha256'), 'audit source digest')
    if (audit.get('schema') != 'mlg-independent-natural-discovery-audit-v1'
            or audit.get('status') != 'independent_saved_evidence_pass'
            or audit.get('audit_source_sha256') != audit_source):
        raise LedgerError('independent natural audit does not match the binding')
    attempt = natural / 'owner' / 'attempt'
    for name, owner_key, audit_key in (
        ('started.json', 'started_sha256', 'journal_started_sha256'),
        ('events.jsonl', 'events_sha256', 'journal_event_sha256'),
        ('terminal.json', 'terminal_sha256', 'journal_terminal_sha256'),
    ):
        expected = _digest(owner_inventory.get(owner_key), owner_key)
        record = natural_files.get(os.path.normcase(str(attempt / name)))
        if audit.get(audit_key) != expected or record is None or record[1] != expected:
            raise LedgerError('natural journal differs from the owner and independent audit')
    hashes = owner_inventory.get('tensor_sha256')
    if type(hashes) is not list or len(hashes) > _MAX_FILES:
        raise LedgerError('invalid natural tensor inventory')
    for value in hashes:
        _digest(value, 'natural tensor digest')
    if len(set(hashes)) != len(hashes):
        raise LedgerError('duplicate natural tensor digest')
    tensor_dir = os.path.normcase(str(attempt / 'tensors'))
    actual = {p: record for p, record in natural_files.items()
              if os.path.dirname(p) == tensor_dir and p.endswith('.f32le')}
    wanted = {os.path.normcase(str(attempt / 'tensors' / (h + '.f32le'))) for h in hashes}
    if set(actual) != wanted or any(actual[p][1] != Path(p).stem for p in actual):
        raise LedgerError('natural tensor files differ from the owner inventory')
    tensor_bytes = sum(record[0] for record in actual.values())
    if (type(audit.get('tensor_count')) is not int or audit['tensor_count'] != len(hashes)
            or type(audit.get('tensor_bytes')) is not int or audit['tensor_bytes'] != tensor_bytes
            or owner_inventory.get('retained_tensor_bytes') != tensor_bytes):
        raise LedgerError('natural tensor totals differ from the independent audit')


def _connect(path):
    db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    try:
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
    except BaseException:
        db.close()
        raise
    return db


def _overlap(left: Path, right: Path) -> bool:
    a, b = os.path.normcase(str(left)), os.path.normcase(str(right))
    try:
        return os.path.commonpath((a, b)) in (a, b)
    except ValueError:
        return False


def initialize_study_ledger(
    ledger_path, *, natural_claim_path, natural_binding_path, prior_paths,
    prior_unique_tensor_bytes, natural_unique_tensor_bytes, expected_natural_claim_sha256,
    expected_natural_binding_sha256, natural_audit_path, expected_natural_audit_sha256,
    expected_natural_terminal_sha256, previous_physical_amendment_path,
    expected_previous_physical_amendment_sha256, physical_amendment_path,
    expected_physical_amendment_sha256, natural_extra_paths=(),
):
    """Import the drained natural attempt and its explicit evidence inventory.

    ``prior_paths`` must enumerate all other study-created evidence, including
    orphan/failed files and a copy of the public preflight logits. The caller
    verifies its completeness and the unique-tensor subtotal. The natural
    claim and binding are automatically included as prior physical files.
    ``natural_extra_paths`` lists copies/staging charged to natural's share.
    ``natural_unique_tensor_bytes`` must include all unique retained tensors,
    including orphan and uncompressed archive payloads; identical copies are
    counted once. The caller verifies this subtotal independently.
    All paths must be absolute and quiescent. The ledger file is created once;
    an interrupted initialization is not automatically retried or replaced.
    """
    ledger = _path(ledger_path, existing=False)
    claim_path = _path(natural_claim_path, existing=True)
    binding_path = _path(natural_binding_path, existing=True)
    audit_path = _path(natural_audit_path, existing=True)
    previous_amendment_path = _path(previous_physical_amendment_path, existing=True)
    amendment_path = _path(physical_amendment_path, existing=True)
    raw_claim, claim = _read_json(claim_path)
    raw_binding, binding = _read_json(binding_path)
    raw_audit, audit = _read_json(audit_path)
    raw_previous, previous_amendment = _read_json(previous_amendment_path)
    raw_amendment, amendment = _read_json(amendment_path)
    if hashlib.sha256(raw_claim).hexdigest() != _digest(expected_natural_claim_sha256, 'expected_natural_claim_sha256'):
        raise LedgerError('natural claim digest mismatch')
    if hashlib.sha256(raw_binding).hexdigest() != _digest(expected_natural_binding_sha256, 'expected_natural_binding_sha256'):
        raise LedgerError('natural binding digest mismatch')
    if hashlib.sha256(raw_audit).hexdigest() != _digest(expected_natural_audit_sha256, 'expected_natural_audit_sha256'):
        raise LedgerError('natural audit digest mismatch')
    if hashlib.sha256(raw_previous).hexdigest() != _digest(expected_previous_physical_amendment_sha256, 'expected_previous_physical_amendment_sha256'):
        raise LedgerError('previous physical amendment digest mismatch')
    if hashlib.sha256(raw_amendment).hexdigest() != _digest(expected_physical_amendment_sha256, 'expected_physical_amendment_sha256'):
        raise LedgerError('physical amendment digest mismatch')
    if claim.get('schema') != 'mlg-natural-study-claim-v1':
        raise LedgerError('unknown natural claim')
    if claim.get('binding_sha256') != hashlib.sha256(raw_binding).hexdigest():
        raise LedgerError('natural binding digest mismatch')
    _fixed_budget(claim.get('budget'))
    _fixed_budget(binding.get('study_budget'))
    if claim['budget'] != binding['study_budget']:
        raise LedgerError('claim and binding budgets differ')
    claim_hash = hashlib.sha256(raw_claim).hexdigest()
    binding_hash = hashlib.sha256(raw_binding).hexdigest()
    _physical_amendment(previous_amendment, claim_hash, binding_hash)
    _physical_amendment(amendment, claim_hash, binding_hash,
                        previous_hash=hashlib.sha256(raw_previous).hexdigest())
    try:
        natural = _path(claim['output'], existing=True)
    except (KeyError, TypeError, OSError) as exc:
        raise LedgerError('invalid natural output') from exc
    if not natural.is_dir():
        raise LedgerError('natural output is not a directory')
    terminal_raw, terminal = _read_json(natural / 'launcher-terminal.json')
    if hashlib.sha256(terminal_raw).hexdigest() != _digest(expected_natural_terminal_sha256, 'expected_natural_terminal_sha256'):
        raise LedgerError('natural terminal digest mismatch')
    if (terminal.get('status') != 'evidence_complete' or terminal.get('child_started') is not True
            or terminal.get('cleanup_verified') is not True or terminal.get('owner_record') != 'owner/supervisor.json'):
        raise LedgerError('natural owner has not been verified complete')
    owner_raw, owner = _read_json(natural / 'owner' / 'supervisor.json')
    if terminal.get('owner_supervisor_sha256') != hashlib.sha256(owner_raw).hexdigest():
        raise LedgerError('natural owner receipt hash mismatch')
    if (owner.get('status') != 'evidence_complete' or type(owner.get('exit_code')) is not int
            or owner['exit_code'] != 0 or any(owner.get(k) is not True for k in
                    ('child_reaped', 'job_drained', 'job_closed', 'cleanup_verified'))):
        raise LedgerError('natural owner is not drained and reaped')
    owner_inventory = owner.get('inventory')
    if type(owner_inventory) is not dict:
        raise LedgerError('natural owner tensor subtotal is missing')
    reported_unique = _nat(owner_inventory.get('retained_tensor_bytes'), 'owner retained_tensor_bytes')
    natural_unique = _nat(natural_unique_tensor_bytes, 'natural_unique_tensor_bytes')
    if natural_unique < reported_unique or natural_unique > PHASE_LIMITS['natural_discovery'][0]:
        raise LedgerError('natural logical subtotal is outside its allowed bounds')
    prior_unique = _nat(prior_unique_tensor_bytes, 'prior_unique_tensor_bytes')
    if prior_unique < PREFLIGHT_BYTES or prior_unique + sum(x[0] for x in PHASE_LIMITS.values()) > STUDY_LOGICAL_LIMIT:
        raise LedgerError('prior logical evidence exceeds the available remainder')
    prior_roots = [claim_path, binding_path, previous_amendment_path, amendment_path] + [_path(p, existing=True) for p in prior_paths]
    natural_roots = [natural, audit_path] + [_path(p, existing=True) for p in natural_extra_paths]
    if any(_overlap(ledger, p) for p in prior_roots + natural_roots):
        raise LedgerError('ledger path overlaps inventoried evidence')
    natural_files = _files(natural_roots)
    prior_files = _files(prior_roots)
    for path, raw, inventory in (
        (claim_path, raw_claim, prior_files), (binding_path, raw_binding, prior_files),
        (previous_amendment_path, raw_previous, prior_files),
        (amendment_path, raw_amendment, prior_files),
        (audit_path, raw_audit, natural_files),
        (natural / 'launcher-terminal.json', terminal_raw, natural_files),
        (natural / 'owner' / 'supervisor.json', owner_raw, natural_files),
    ):
        entry = inventory.get(os.path.normcase(str(path)))
        if entry != [len(raw), hashlib.sha256(raw).hexdigest()]:
            raise LedgerError('pinned evidence changed during reconciliation')
    _check_natural_audit(natural, owner_inventory, audit, binding, natural_files)
    if natural_files.keys() & prior_files.keys():
        raise LedgerError('natural and prior inventories overlap')
    if [PREFLIGHT_BYTES, PREFLIGHT_SHA256] not in prior_files.values():
        raise LedgerError('known preflight payload is absent')
    natural_disk = sum(v[0] for v in natural_files.values())
    prior_disk = sum(v[0] for v in prior_files.values())
    if natural_disk > PHASE_LIMITS['natural_discovery'][1]:
        raise LedgerError('natural evidence exceeds its physical share')
    if prior_disk + LEDGER_DISK_RESERVE + sum(x[1] for x in PHASE_LIMITS.values()) > STUDY_DISK_LIMIT:
        raise LedgerError('prior physical evidence exceeds the available remainder')
    inventory = json.dumps({'natural': natural_files, 'prior': prior_files}, sort_keys=True, separators=(',', ':'))
    meta = json.dumps({
        'schema': _SCHEMA, 'natural_claim_sha256': hashlib.sha256(raw_claim).hexdigest(),
        'natural_binding_sha256': hashlib.sha256(raw_binding).hexdigest(),
        'physical_amendment_sha256': hashlib.sha256(raw_amendment).hexdigest(),
        'previous_physical_amendment_sha256': hashlib.sha256(raw_previous).hexdigest(),
        'natural_audit_sha256': hashlib.sha256(raw_audit).hexdigest(),
        'natural_terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(),
        'prior_unique_tensor_bytes': prior_unique, 'prior_physical_bytes': prior_disk,
        'natural_unique_tensor_bytes': natural_unique, 'natural_reported_tensor_bytes': reported_unique,
        'natural_physical_bytes': natural_disk, 'ledger_disk_reserve_bytes': LEDGER_DISK_RESERVE,
        'phase_limits': _phase_snapshot(), 'study_logical_limit': STUDY_LOGICAL_LIMIT,
        'study_disk_limit': STUDY_DISK_LIMIT, 'required_free_disk_bytes': REQUIRED_FREE_DISK_BYTES,
        'metadata_max_bytes': _MAX_INVENTORY,
        'natural_roots': [str(p) for p in natural_roots], 'prior_roots': [str(p) for p in prior_roots],
    }, sort_keys=True, separators=(',', ':'))
    _check_metadata_bound(meta, inventory)
    fd = os.open(ledger, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        db = _connect(ledger)
    except sqlite3.Error as exc:
        raise LedgerError('ledger initialization could not be opened') from exc
    try:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL, inventory TEXT NOT NULL)')
        db.execute('CREATE TABLE claims (phase TEXT PRIMARY KEY, output TEXT NOT NULL, logical INTEGER NOT NULL, disk INTEGER NOT NULL, free INTEGER NOT NULL)')
        db.execute('INSERT INTO meta VALUES(1, ?, ?)', (meta, inventory))
        db.execute('INSERT INTO claims VALUES(?, ?, ?, ?, ?)',
                   ('natural_discovery', str(natural), *PHASE_LIMITS['natural_discovery'], 0))
        db.execute('COMMIT')
    finally:
        db.close()


def claim_phase(ledger_path, phase, *, output_path) -> PhaseGrant:
    """Atomically and permanently consume one phase before starting work."""
    if type(phase) is not str or phase not in PHASE_LIMITS or phase == 'natural_discovery':
        raise LedgerError('unknown or already imported phase')
    ledger = _path(ledger_path, existing=True)
    output = _path(output_path, existing=False)
    free = shutil.disk_usage(output.parent).free
    if free < REQUIRED_FREE_DISK_BYTES:
        raise LedgerError('less than 6 GiB free on output filesystem')
    try:
        db = _connect(ledger)
    except sqlite3.Error as exc:
        raise LedgerError('ledger could not be opened') from exc
    try:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT data, inventory FROM meta WHERE id=1').fetchone()
        if row is None:
            raise LedgerError('ledger initialization incomplete')
        _check_metadata_bound(*row)
        meta = json.loads(row[0])
        if meta.get('schema') != _SCHEMA:
            raise LedgerError('unknown ledger schema')
        if (meta.get('phase_limits') != _phase_snapshot()
                or meta.get('study_logical_limit') != STUDY_LOGICAL_LIMIT
                or meta.get('study_disk_limit') != STUDY_DISK_LIMIT
                or meta.get('required_free_disk_bytes') != REQUIRED_FREE_DISK_BYTES
                or meta.get('metadata_max_bytes') != _MAX_INVENTORY
                or meta.get('ledger_disk_reserve_bytes') != LEDGER_DISK_RESERVE):
            raise LedgerError('phase budget differs from the initialized ledger')
        prior_unique = _nat(meta.get('prior_unique_tensor_bytes'), 'prior_unique_tensor_bytes')
        prior_disk = _nat(meta.get('prior_physical_bytes'), 'prior_physical_bytes')
        natural_unique = _nat(meta.get('natural_unique_tensor_bytes'), 'natural_unique_tensor_bytes')
        reported_unique = _nat(meta.get('natural_reported_tensor_bytes'), 'natural_reported_tensor_bytes')
        if (prior_unique < PREFLIGHT_BYTES
                or natural_unique < reported_unique or natural_unique > PHASE_LIMITS['natural_discovery'][0]
                or prior_unique + sum(x[0] for x in PHASE_LIMITS.values()) > STUDY_LOGICAL_LIMIT
                or prior_disk + LEDGER_DISK_RESERVE + sum(x[1] for x in PHASE_LIMITS.values()) > STUDY_DISK_LIMIT):
            raise LedgerError('ledger totals exceed the study limits')
        inventory = json.loads(row[1])
        # Do not admit new work after any previously inventoried evidence has changed.
        current = {'natural': _files([_path(p, existing=True) for p in meta['natural_roots']]),
                   'prior': _files([_path(p, existing=True) for p in meta['prior_roots']])}
        if current != inventory:
            raise LedgerError('reconciled evidence changed; a new review is required')
        if (sum(v[0] for v in inventory['prior'].values()) != prior_disk
                or sum(v[0] for v in inventory['natural'].values()) != meta.get('natural_physical_bytes')):
            raise LedgerError('stored inventory totals differ')
        if meta['natural_physical_bytes'] > PHASE_LIMITS['natural_discovery'][1]:
            raise LedgerError('natural evidence exceeds its physical share')
        claims = db.execute('SELECT phase, output, logical, disk FROM claims').fetchall()
        if any(p not in PHASE_LIMITS or (logical, disk) != PHASE_LIMITS[p]
               for p, _, logical, disk in claims):
            raise LedgerError('a stored phase differs from the frozen budget')
        if not any(x[0] == 'natural_discovery' and (x[2], x[3]) == PHASE_LIMITS['natural_discovery'] for x in claims):
            raise LedgerError('missing imported natural claim')
        if any(x[0] == phase for x in claims):
            raise LedgerError('phase already claimed')
        if _overlap(output, ledger) or any(_overlap(output, Path(p)) for p in meta['natural_roots'] + meta['prior_roots']):
            raise LedgerError('phase output overlaps inventoried evidence')
        for _, other, _, _ in claims:
            if _overlap(output, Path(other)):
                raise LedgerError('phase output paths overlap')
        # Recheck after the potentially lengthy inventory scan and while the
        # claim transaction is locked. This is still a point-in-time check,
        # not a reservation against unrelated disk users.
        free = shutil.disk_usage(output.parent).free
        if free < REQUIRED_FREE_DISK_BYTES:
            raise LedgerError('less than 6 GiB free on output filesystem')
        logical, disk = PHASE_LIMITS[phase]
        db.execute('INSERT INTO claims VALUES (?, ?, ?, ?, ?)', (phase, str(output), logical, disk, free))
        db.execute('COMMIT')
    except sqlite3.Error as exc:
        raise LedgerError('ledger transaction failed or is invalid') from exc
    finally:
        db.close()
    return PhaseGrant(phase, str(output), logical, disk)
