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
V2_DISK_LIMIT = 4096 * _MIB
V2_FREE_DISK_BYTES = 6144 * _MIB
V2_METADATA_MAX = 8 * _MIB
V2_LEDGER_RESERVE = 64 * _MIB
STUDY_DISK_LIMIT = 8192 * _MIB
REQUIRED_FREE_DISK_BYTES = 12288 * _MIB
LEDGER_DISK_RESERVE = 128 * _MIB
PREFLIGHT_BYTES = 3_216_448
PREFLIGHT_SHA256 = '7c262f3fe81236888a8f5fdb6d31f7835666674be9ffbf36f8ce2ca6a375eb6c'
PHASE_LIMITS = MappingProxyType({
    'natural_discovery': (128 * _MIB, 4096 * _MIB),
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
V2_PHASE_DISK = MappingProxyType({
    **ORIGINAL_PHASE_DISK,
    'natural_discovery': 1024 * _MIB,
})
_SCHEMA = 'mlg-agreement-phase-ledger-v5'
_RECONCILED_SCHEMA = 'mlg-agreement-phase-ledger-v6'
_SCOPE_SCHEMA = 'mlg-agreement-evidence-scope-amendment-v1'
_MAX_FILES = 32768  # Applied independently to each side's traversal.
_MAX_INVENTORY = 16 * _MIB  # Combined UTF-8 bytes of meta and inventory.
_MAX_JSON = 4 * _MIB
_MAX_HISTORY = 32 * _MIB  # Old snapshots plus compact audit records.


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


def _claims_digest(claims):
    """Bind all claim fields, including output and recorded free space."""
    raw = json.dumps(sorted(claims), ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _snapshot_digest(meta, inventory):
    """Bind the exact stored bytes, including their unambiguous lengths."""
    digest = hashlib.sha256()
    for value in (meta.encode('utf-8'), inventory.encode('utf-8')):
        digest.update(len(value).to_bytes(8, 'big'))
        digest.update(value)
    return digest.hexdigest()


def _history_digest(db):
    """Bind the ordered logical history and retired rows, not SQLite's file bytes."""
    digest = hashlib.sha256()
    for table, query in (
        ('reconciliations', 'SELECT seq,manifest_sha,gap_sha,claims_sha,previous_meta,previous_inventory '
                            'FROM reconciliations ORDER BY seq'),
        ('retired_evidence', 'SELECT side,path_sha,size,content_sha FROM retired_evidence '
                             'ORDER BY side,path_sha,size,content_sha'),
    ):
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None
        digest.update((table + (':present\n' if exists else ':absent\n')).encode('ascii'))
        if exists:
            for row in db.execute(query):
                raw = json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')
                digest.update(len(raw).to_bytes(8, 'big'))
                digest.update(raw)
    return digest.hexdigest()


def _retired_key(side, name, value):
    if side not in ('natural', 'prior') or type(name) is not str or type(value) is not list or len(value) != 2:
        raise LedgerError('invalid retired evidence entry')
    size = _nat(value[0], 'retired evidence size')
    content_hash = _digest(value[1], 'retired evidence digest')
    return side, hashlib.sha256(name.encode('utf-8')).hexdigest(), size, content_hash


def _retired_charges(db, inventory):
    """Return charges for old path/content pairs absent from this inventory."""
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='retired_evidence'").fetchone() is None:
        return {'natural': 0, 'prior': 0}, set()
    rows = db.execute('SELECT side,path_sha,size,content_sha FROM retired_evidence').fetchall()
    if len(rows) > 2 * _MAX_FILES:
        raise LedgerError('retired evidence exceeds the ledger bound')
    active = {_retired_key(side, path, value)
              for side in ('natural', 'prior') for path, value in inventory[side].items()}
    charges = {'natural': 0, 'prior': 0}
    retired = set()
    for side, path_sha, size, content_sha in rows:
        if side not in charges:
            raise LedgerError('invalid retired evidence side')
        key = (side, _digest(path_sha, 'retired path digest'),
               _nat(size, 'retired evidence size'), _digest(content_sha, 'retired content digest'))
        if key in retired:
            raise LedgerError('duplicate retired evidence')
        retired.add(key)
        if key not in active:
            charges[side] += size
    return charges, retired


def _check_scope_amendment(scope, old, *, gap_hash, state_hash, history_hash, claims_hash, scope_hash):
    if type(scope) is not dict or scope.get('schema') != _SCOPE_SCHEMA:
        raise LedgerError('unknown evidence scope amendment')
    if (scope.get('previous_physical_amendment_sha256') != old.get('physical_amendment_sha256')
            or scope.get('initial_gap_record_sha256') != gap_hash):
        raise LedgerError('scope amendment does not bind the earlier record')
    for key, expected in (('study_logical_limit_bytes', STUDY_LOGICAL_LIMIT),
                          ('study_physical_limit_bytes', STUDY_DISK_LIMIT),
                          ('required_free_disk_bytes', REQUIRED_FREE_DISK_BYTES)):
        if type(scope.get(key)) is not int or scope[key] != expected:
            raise LedgerError('scope amendment changes the frozen budgets')
    for key, index in (('logical_phase_reservations_bytes', 0), ('physical_phase_reservations_bytes', 1)):
        wanted = {phase: limits[index] for phase, limits in PHASE_LIMITS.items()}
        value = scope.get(key)
        if type(value) is not dict or set(value) != set(wanted) or any(
                type(value[k]) is not int or value[k] != v for k, v in wanted.items()):
            raise LedgerError('scope amendment changes a frozen phase share')
    if old.get('schema') == _SCHEMA:
        if (scope.get('initial_ledger_state_sha256') != state_hash
                or scope.get('initial_ledger_history_sha256') != history_hash
                or scope.get('existing_claims_sha256') != claims_hash):
            raise LedgerError('scope amendment does not bind the old ledger and claims')
    elif old.get('schema') != _RECONCILED_SCHEMA or old.get('evidence_scope_amendment_sha256') != scope_hash:
        raise LedgerError('ledger does not retain the reviewed evidence scope')


def _physical_amendment(amendment, claim_hash, binding_hash, *, expected_version, previous_hash=None):
    schema = amendment.get('schema')
    versions = {'mlg-agreement-physical-budget-amendment-v1': 1,
                'mlg-agreement-physical-budget-amendment-v2': 2,
                'mlg-agreement-physical-budget-amendment-v3': 3}
    version = versions.get(schema) if type(schema) is str else None
    if version != expected_version or (version == 1) != (previous_hash is None):
        raise LedgerError('unknown physical amendment chain')
    if (amendment.get('natural_claim_sha256') != claim_hash
            or amendment.get('natural_binding_sha256') != binding_hash):
        raise LedgerError('physical amendment does not bind the original natural claim')
    if version != 1 and amendment.get('previous_physical_amendment_sha256') != previous_hash:
        raise LedgerError('physical amendment does not bind its predecessor')
    disk = {1: PREVIOUS_DISK_LIMIT, 2: V2_DISK_LIMIT, 3: STUDY_DISK_LIMIT}[version]
    free = {1: PREVIOUS_FREE_DISK_BYTES, 2: V2_FREE_DISK_BYTES, 3: REQUIRED_FREE_DISK_BYTES}[version]
    for key, expected in (
        ('original_host_disk_allowance_bytes', ORIGINAL_DISK_LIMIT),
        ('host_disk_allowance_bytes', disk),
        ('required_free_disk_bytes', free),
        ('study_limit_bytes', STUDY_LOGICAL_LIMIT),
    ):
        if type(amendment.get(key)) is not int or amendment[key] != expected:
            raise LedgerError('physical amendment totals differ')
    for key, index in (('phase_reservations_bytes', 0), ('host_disk_phase_reservations_bytes', 1)):
        table = amendment.get(key)
        wanted = ({phase: limits[index] for phase, limits in PHASE_LIMITS.items()}
                  if index == 0 or version == 3 else
                  dict(PREVIOUS_PHASE_DISK if version == 1 else V2_PHASE_DISK))
        if type(table) is not dict or set(table) != set(wanted):
            raise LedgerError('physical amendment shares differ')
        if any(type(table[p]) is not int or table[p] != value for p, value in wanted.items()):
            raise LedgerError('physical amendment shares differ')
    if version >= 2:
        metadata = V2_METADATA_MAX if version == 2 else _MAX_INVENTORY
        reserve = V2_LEDGER_RESERVE if version == 2 else LEDGER_DISK_RESERVE
        for key, expected in (('metadata_max_bytes', metadata), ('ledger_disk_reserve_bytes', reserve)):
            if type(amendment.get(key)) is not int or amendment[key] != expected:
                raise LedgerError('physical amendment metadata bounds differ')
    if version == 3 and (type(amendment.get('max_files_per_inventory')) is not int
                         or amendment['max_files_per_inventory'] != _MAX_FILES):
        raise LedgerError('physical amendment file bound differs')


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


def _build_snapshot(
    ledger, *, natural_claim_path, natural_binding_path, prior_paths,
    prior_unique_tensor_bytes, natural_unique_tensor_bytes, expected_natural_claim_sha256,
    expected_natural_binding_sha256, natural_audit_path, expected_natural_audit_sha256,
    expected_natural_terminal_sha256, original_physical_amendment_path,
    expected_original_physical_amendment_sha256, previous_physical_amendment_path,
    expected_previous_physical_amendment_sha256, physical_amendment_path,
    expected_physical_amendment_sha256, natural_extra_paths=(),
):
    """Build and verify a snapshot without creating or modifying a ledger."""
    claim_path = _path(natural_claim_path, existing=True)
    binding_path = _path(natural_binding_path, existing=True)
    audit_path = _path(natural_audit_path, existing=True)
    original_amendment_path = _path(original_physical_amendment_path, existing=True)
    previous_amendment_path = _path(previous_physical_amendment_path, existing=True)
    amendment_path = _path(physical_amendment_path, existing=True)
    raw_claim, claim = _read_json(claim_path)
    raw_binding, binding = _read_json(binding_path)
    raw_audit, audit = _read_json(audit_path)
    raw_original, original_amendment = _read_json(original_amendment_path)
    raw_previous, previous_amendment = _read_json(previous_amendment_path)
    raw_amendment, amendment = _read_json(amendment_path)
    if hashlib.sha256(raw_claim).hexdigest() != _digest(expected_natural_claim_sha256, 'expected_natural_claim_sha256'):
        raise LedgerError('natural claim digest mismatch')
    if hashlib.sha256(raw_binding).hexdigest() != _digest(expected_natural_binding_sha256, 'expected_natural_binding_sha256'):
        raise LedgerError('natural binding digest mismatch')
    if hashlib.sha256(raw_audit).hexdigest() != _digest(expected_natural_audit_sha256, 'expected_natural_audit_sha256'):
        raise LedgerError('natural audit digest mismatch')
    if hashlib.sha256(raw_original).hexdigest() != _digest(expected_original_physical_amendment_sha256, 'expected_original_physical_amendment_sha256'):
        raise LedgerError('original physical amendment digest mismatch')
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
    _physical_amendment(original_amendment, claim_hash, binding_hash, expected_version=1)
    _physical_amendment(previous_amendment, claim_hash, binding_hash,
                        expected_version=2, previous_hash=hashlib.sha256(raw_original).hexdigest())
    _physical_amendment(amendment, claim_hash, binding_hash,
                        expected_version=3, previous_hash=hashlib.sha256(raw_previous).hexdigest())
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
    prior_roots = [claim_path, binding_path, original_amendment_path, previous_amendment_path, amendment_path] + [_path(p, existing=True) for p in prior_paths]
    natural_roots = [natural, audit_path] + [_path(p, existing=True) for p in natural_extra_paths]
    if any(_overlap(ledger, p) for p in prior_roots + natural_roots):
        raise LedgerError('ledger path overlaps inventoried evidence')
    natural_files = _files(natural_roots)
    prior_files = _files(prior_roots)
    for path, raw, inventory in (
        (claim_path, raw_claim, prior_files), (binding_path, raw_binding, prior_files),
        (original_amendment_path, raw_original, prior_files),
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
        'original_physical_amendment_sha256': hashlib.sha256(raw_original).hexdigest(),
        'natural_audit_sha256': hashlib.sha256(raw_audit).hexdigest(),
        'natural_terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(),
        'prior_unique_tensor_bytes': prior_unique, 'prior_physical_bytes': prior_disk,
        'natural_unique_tensor_bytes': natural_unique, 'natural_reported_tensor_bytes': reported_unique,
        'natural_physical_bytes': natural_disk, 'ledger_disk_reserve_bytes': LEDGER_DISK_RESERVE,
        'phase_limits': _phase_snapshot(), 'study_logical_limit': STUDY_LOGICAL_LIMIT,
        'study_disk_limit': STUDY_DISK_LIMIT, 'required_free_disk_bytes': REQUIRED_FREE_DISK_BYTES,
        'metadata_max_bytes': _MAX_INVENTORY,
        'max_files_per_inventory': _MAX_FILES,
        'natural_roots': [str(p) for p in natural_roots], 'prior_roots': [str(p) for p in prior_roots],
    }, sort_keys=True, separators=(',', ':'))
    _check_metadata_bound(meta, inventory)
    return meta, inventory, natural


def initialize_study_ledger(
    ledger_path, *, natural_claim_path, natural_binding_path, prior_paths,
    prior_unique_tensor_bytes, natural_unique_tensor_bytes, expected_natural_claim_sha256,
    expected_natural_binding_sha256, natural_audit_path, expected_natural_audit_sha256,
    expected_natural_terminal_sha256, original_physical_amendment_path,
    expected_original_physical_amendment_sha256, previous_physical_amendment_path,
    expected_previous_physical_amendment_sha256, physical_amendment_path,
    expected_physical_amendment_sha256, natural_extra_paths=(),
):
    """Import the drained natural attempt and its explicit evidence inventory.

    The caller independently verifies completeness and unique-tensor subtotals,
    including orphan and uncompressed archive payloads. All evidence paths must
    be absolute and quiescent. The ledger is created exclusively once; an
    interrupted initialization is not automatically retried or replaced.
    """
    ledger = _path(ledger_path, existing=False)
    meta, inventory, natural = _build_snapshot(
        ledger, natural_claim_path=natural_claim_path, natural_binding_path=natural_binding_path,
        prior_paths=prior_paths, prior_unique_tensor_bytes=prior_unique_tensor_bytes,
        natural_unique_tensor_bytes=natural_unique_tensor_bytes,
        expected_natural_claim_sha256=expected_natural_claim_sha256,
        expected_natural_binding_sha256=expected_natural_binding_sha256,
        natural_audit_path=natural_audit_path, expected_natural_audit_sha256=expected_natural_audit_sha256,
        expected_natural_terminal_sha256=expected_natural_terminal_sha256,
        original_physical_amendment_path=original_physical_amendment_path,
        expected_original_physical_amendment_sha256=expected_original_physical_amendment_sha256,
        previous_physical_amendment_path=previous_physical_amendment_path,
        expected_previous_physical_amendment_sha256=expected_previous_physical_amendment_sha256,
        physical_amendment_path=physical_amendment_path,
        expected_physical_amendment_sha256=expected_physical_amendment_sha256,
        natural_extra_paths=natural_extra_paths,
    )
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


def reconcile_study_ledger(
    ledger_path, *, manifest_path, expected_manifest_sha256,
    gap_record_path, expected_gap_record_sha256, expected_initial_manifest_sha256,
    private_gap_manifest_path,
    scope_amendment_path, expected_scope_amendment_sha256,
    expected_previous_state_sha256, expected_previous_history_sha256, expected_claims_sha256,
):
    """Replace a reviewed inventory atomically while preserving all claims.

    The custodian must independently review the manifest's completeness,
    logical subtotals and phase-side evidence, and keep its paths quiescent
    through this transaction. SHA-256 pins bind bytes, not that attestation.
    Each changed or missing historical path/content pair retains a conservative
    charge until that exact pair is again present in the same inventory. The
    initial gap remains explicit and prior snapshots are retained in bounded history. A returned
    record is not a phase grant; ambiguous interruption must be inspected, not
    repaired by resetting the ledger or consuming a claim again.
    """
    ledger = _path(ledger_path, existing=True)
    manifest_file = _path(manifest_path, existing=True)
    gap_file = _path(gap_record_path, existing=True)
    private_gap_file = _path(private_gap_manifest_path, existing=True)
    scope_file = _path(scope_amendment_path, existing=True)
    manifest_hash = _digest(expected_manifest_sha256, 'expected_manifest_sha256')
    gap_hash = _digest(expected_gap_record_sha256, 'expected_gap_record_sha256')
    initial_hash = _digest(expected_initial_manifest_sha256, 'expected_initial_manifest_sha256')
    scope_hash = _digest(expected_scope_amendment_sha256, 'expected_scope_amendment_sha256')
    state_hash = _digest(expected_previous_state_sha256, 'expected_previous_state_sha256')
    history_hash = _digest(expected_previous_history_sha256, 'expected_previous_history_sha256')
    claims_hash = _digest(expected_claims_sha256, 'expected_claims_sha256')
    try:
        db = _connect(ledger)
    except sqlite3.Error as exc:
        raise LedgerError('ledger could not be opened') from exc
    try:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT data, inventory FROM meta WHERE id=1').fetchone()
        if row is None:
            raise LedgerError('ledger initialization incomplete')
        old_meta_text, old_inventory_text = row
        _check_metadata_bound(old_meta_text, old_inventory_text)
        if _snapshot_digest(old_meta_text, old_inventory_text) != state_hash:
            raise LedgerError('prior ledger state differs from the reviewed snapshot')
        if _history_digest(db) != history_hash:
            raise LedgerError('prior ledger history differs from the reviewed snapshot')
        old = json.loads(old_meta_text)
        old_inventory = json.loads(old_inventory_text)
        if old.get('schema') == _RECONCILED_SCHEMA and old.get('current_history_sha256') != history_hash:
            raise LedgerError('stored ledger history digest differs')
        if (old.get('schema') not in (_SCHEMA, _RECONCILED_SCHEMA) or old.get('phase_limits') != _phase_snapshot()
                or old.get('study_logical_limit') != STUDY_LOGICAL_LIMIT
                or old.get('study_disk_limit') != STUDY_DISK_LIMIT
                or old.get('required_free_disk_bytes') != REQUIRED_FREE_DISK_BYTES
                or old.get('metadata_max_bytes') != _MAX_INVENTORY
                or old.get('max_files_per_inventory') != _MAX_FILES
                or old.get('ledger_disk_reserve_bytes') != LEDGER_DISK_RESERVE):
            raise LedgerError('initialized ledger budget differs')
        old_natural = _nat(old.get('natural_physical_bytes'), 'natural_physical_bytes')
        old_prior = _nat(old.get('prior_physical_bytes'), 'prior_physical_bytes')
        old_natural_floor = _nat(old.get('natural_physical_floor_bytes', 0), 'natural_physical_floor_bytes')
        old_prior_floor = _nat(old.get('prior_physical_floor_bytes', 0), 'prior_physical_floor_bytes')
        historical, retired = _retired_charges(db, old_inventory)
        if (sum(v[0] for v in old_inventory['natural'].values()) + old_natural_floor != old_natural
                or sum(v[0] for v in old_inventory['prior'].values()) + old_prior_floor != old_prior
                or historical != {'natural': old_natural_floor, 'prior': old_prior_floor}):
            raise LedgerError('initialized inventory totals differ')
        claims = db.execute('SELECT phase, output, logical, disk, free FROM claims ORDER BY phase').fetchall()
        if (_claims_digest(claims) != claims_hash
                or any(p not in PHASE_LIMITS or (logical, disk) != PHASE_LIMITS[p]
                       or type(output) is not str or not Path(output).is_absolute()
                       or type(free) is not int or free < 0 for p, output, logical, disk, free in claims)
                or not any(c[0] == 'natural_discovery' for c in claims)):
            raise LedgerError('claims differ from the independently reviewed state')
        def reject_private_input(path):
            for phase, output, _, _, _ in claims:
                if phase != 'natural_discovery' and _overlap(path, Path(output).parent):
                    raise LedgerError('reconciliation input overlaps a claimed private parent')
        # No caller-selected input is opened before the claimed private phase
        # parents have been excluded. This includes the manifest itself.
        for path in (manifest_file, gap_file, private_gap_file, scope_file):
            reject_private_input(path)
        raw_manifest, manifest = _read_json(manifest_file)
        raw_gap, gap = _read_json(gap_file)
        raw_private_gap, private_gap = _read_json(private_gap_file)
        raw_scope, scope = _read_json(scope_file)
        if hashlib.sha256(raw_manifest).hexdigest() != manifest_hash:
            raise LedgerError('reviewed inventory manifest digest mismatch')
        if (hashlib.sha256(raw_scope).hexdigest() != scope_hash
                or manifest.get('previous_ledger_state_sha256') != state_hash
                or manifest.get('previous_ledger_history_sha256') != history_hash
                or manifest.get('expected_claims_sha256') != claims_hash
                or manifest.get('evidence_scope_amendment_sha256') != scope_hash):
            raise LedgerError('reviewed inventory scope or prior state differs')
        _check_scope_amendment(scope, old, gap_hash=gap_hash, state_hash=state_hash, history_hash=history_hash,
                               claims_hash=claims_hash, scope_hash=scope_hash)
        if (hashlib.sha256(raw_gap).hexdigest() != gap_hash
                or gap.get('schema') != 'mlg-agreement-initial-inventory-gap-v1'
                or gap.get('initial_inventory_manifest_sha256') != initial_hash
                or old.get('initial_inventory_manifest_sha256', initial_hash) != initial_hash
                or old.get('initial_gap_record_sha256', gap_hash) != gap_hash):
            raise LedgerError('initial inventory gap record differs')
        if (hashlib.sha256(raw_private_gap).hexdigest() != gap.get('private_observation_manifest_sha256')
                or private_gap.get('schema') != 'mlg-agreement-inventory-gap-private-v1'
                or private_gap.get('initial_manifest_sha256') != initial_hash):
            raise LedgerError('private initial-gap observation differs')
        gap_keys = set()
        for side in ('natural', 'prior'):
            entries = private_gap.get(side + '_entries')
            if type(entries) is not dict or len(entries) > _MAX_FILES:
                raise LedgerError('invalid initial-gap observation entries')
            side_keys = {_retired_key(side, name, value) for name, value in entries.items()}
            if (len(side_keys) != len(entries)
                    or _nat(gap.get('additional_' + side + '_files'), 'gap file count') != len(entries)
                    or _nat(gap.get('additional_' + side + '_apparent_bytes'), 'gap bytes')
                       != sum(key[2] for key in side_keys)):
                raise LedgerError('initial-gap aggregate differs from observed entries')
            gap_keys.update(side_keys)
        for name in ('prior_paths', 'natural_extra_paths'):
            if type(manifest.get(name)) is not list or any(type(p) is not str for p in manifest[name]):
                raise LedgerError('invalid reviewed inventory roots')
        try:
            fixed = (('claim', 'expected_claim_sha256', 'natural_claim_sha256'),
                     ('binding', 'expected_binding_sha256', 'natural_binding_sha256'),
                     ('audit', 'expected_audit_sha256', 'natural_audit_sha256'),
                     ('original_physical_amendment', 'expected_original_physical_amendment_sha256', 'original_physical_amendment_sha256'),
                     ('previous_physical_amendment', 'expected_previous_physical_amendment_sha256', 'previous_physical_amendment_sha256'),
                     ('physical_amendment', 'expected_physical_amendment_sha256', 'physical_amendment_sha256'))
            for path_name, digest_name, old_name in fixed:
                if type(manifest[path_name]) is not str or manifest[digest_name] != old.get(old_name):
                    raise LedgerError('reconciliation changed a fixed provenance input')
                reject_private_input(_path(manifest[path_name], existing=True))
            if manifest['expected_terminal_sha256'] != old.get('natural_terminal_sha256'):
                raise LedgerError('reconciliation changed the natural terminal pin')
            prior_roots = manifest['prior_paths'] + [str(manifest_file), str(gap_file),
                                                      str(private_gap_file), str(scope_file)]
            # Reject a claimed output or a newly introduced root beneath its
            # private parent before any inventory hashing. Existing individually
            # snapshotted public files under that parent may remain.
            old_roots = {os.path.normcase(str(Path(p))) for p in old['natural_roots'] + old['prior_roots']}
            for candidate in prior_roots + manifest['natural_extra_paths']:
                root = _path(candidate, existing=True)
                for phase, output, _, _, _ in claims:
                    if phase == 'natural_discovery':
                        continue
                    if _overlap(root, Path(output)):
                        raise LedgerError('reviewed evidence overlaps a claimed phase output')
                    if (_overlap(root, Path(output).parent)
                            and (os.path.normcase(str(root)) not in old_roots or not root.is_file())):
                        raise LedgerError('new evidence root overlaps a claimed phase parent')
            new_meta_text, new_inventory_text, natural = _build_snapshot(
                ledger, natural_claim_path=manifest['claim'], natural_binding_path=manifest['binding'],
                natural_audit_path=manifest['audit'], prior_paths=prior_roots,
                natural_extra_paths=manifest['natural_extra_paths'],
                prior_unique_tensor_bytes=manifest['prior_unique_tensor_bytes_candidate'],
                natural_unique_tensor_bytes=manifest['natural_unique_tensor_bytes_candidate'],
                expected_natural_claim_sha256=manifest['expected_claim_sha256'],
                expected_natural_binding_sha256=manifest['expected_binding_sha256'],
                expected_natural_audit_sha256=manifest['expected_audit_sha256'],
                expected_natural_terminal_sha256=manifest['expected_terminal_sha256'],
                original_physical_amendment_path=manifest['original_physical_amendment'],
                expected_original_physical_amendment_sha256=manifest['expected_original_physical_amendment_sha256'],
                previous_physical_amendment_path=manifest['previous_physical_amendment'],
                expected_previous_physical_amendment_sha256=manifest['expected_previous_physical_amendment_sha256'],
                physical_amendment_path=manifest['physical_amendment'],
                expected_physical_amendment_sha256=manifest['expected_physical_amendment_sha256'],
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise LedgerError('invalid or unavailable reviewed inventory') from exc
        new = json.loads(new_meta_text)
        new_inventory = json.loads(new_inventory_text)
        for key in ('natural_claim_sha256', 'natural_binding_sha256', 'natural_audit_sha256',
                    'natural_terminal_sha256', 'original_physical_amendment_sha256',
                    'previous_physical_amendment_sha256', 'physical_amendment_sha256'):
            if old.get(key) != new.get(key):
                raise LedgerError('reconciliation changed an original evidence pin')
        for key in ('prior_unique_tensor_bytes', 'natural_unique_tensor_bytes'):
            if _nat(new.get(key), key) < _nat(old.get(key), key):
                raise LedgerError('logical tensor accounting cannot decrease')
        if not any(c[0] == 'natural_discovery' and Path(c[1]) == natural for c in claims):
            raise LedgerError('natural claim output changed')
        for phase, output, _, _, _ in claims:
            if phase != 'natural_discovery' and any(
                    _overlap(Path(output), Path(root)) for root in new['natural_roots'] + new['prior_roots']):
                raise LedgerError('reviewed evidence overlaps a claimed phase output')
        for path, raw in ((manifest_file, raw_manifest), (gap_file, raw_gap),
                          (private_gap_file, raw_private_gap), (scope_file, raw_scope)):
            if new_inventory['prior'].get(os.path.normcase(str(path))) != [len(raw), hashlib.sha256(raw).hexdigest()]:
                raise LedgerError('pinned reconciliation evidence changed during inventory')
        db.execute('CREATE TABLE IF NOT EXISTS retired_evidence '
                   '(side TEXT NOT NULL, path_sha TEXT NOT NULL, size INTEGER NOT NULL, '
                   'content_sha TEXT NOT NULL, PRIMARY KEY(side,path_sha,size,content_sha))')
        newly_retired = {_retired_key(side, path, value)
                         for side in ('natural', 'prior')
                         for path, value in old_inventory[side].items()
                         if new_inventory[side].get(path) != value}
        # The separately pinned public gap predates this update but was not in
        # the original DB. Preserve every observed copy even if its path is no
        # longer present or its contents have since changed.
        newly_retired = (newly_retired | gap_keys) - retired
        if len(retired) + len(newly_retired) > 2 * _MAX_FILES:
            raise LedgerError('retired evidence exceeds the ledger bound')
        db.executemany('INSERT INTO retired_evidence(side,path_sha,size,content_sha) VALUES(?,?,?,?)',
                       sorted(newly_retired))
        historical, retired = _retired_charges(db, new_inventory)
        new['natural_physical_floor_bytes'] = historical['natural']
        new['prior_physical_floor_bytes'] = historical['prior']
        new['natural_physical_bytes'] += historical['natural']
        new['prior_physical_bytes'] += historical['prior']
        if new['natural_physical_bytes'] < old_natural or new['prior_physical_bytes'] < old_prior:
            raise LedgerError('historical physical accounting cannot decrease')
        new['initial_inventory_manifest_sha256'] = initial_hash
        new['initial_gap_record_sha256'] = gap_hash
        new['schema'] = _RECONCILED_SCHEMA
        new['evidence_scope_amendment_sha256'] = scope_hash
        new['current_history_sha256'] = '0' * 64  # Fixed-size bound before inserting history.
        new['current_reviewed_manifest_sha256'] = manifest_hash
        if (new['natural_physical_bytes'] > PHASE_LIMITS['natural_discovery'][1]
                or new['prior_physical_bytes'] + LEDGER_DISK_RESERVE
                   + sum(x[1] for x in PHASE_LIMITS.values()) > STUDY_DISK_LIMIT):
            raise LedgerError('monotonic physical accounting exceeds the allowance')
        new_meta_text = json.dumps(new, sort_keys=True, separators=(',', ':'))
        _check_metadata_bound(new_meta_text, new_inventory_text)
        db.execute('CREATE TABLE IF NOT EXISTS reconciliations '
                   '(seq INTEGER PRIMARY KEY, manifest_sha TEXT NOT NULL UNIQUE, '
                   'gap_sha TEXT NOT NULL, claims_sha TEXT NOT NULL, '
                   'previous_meta TEXT NOT NULL, previous_inventory TEXT NOT NULL)')
        if db.execute('SELECT 1 FROM reconciliations WHERE manifest_sha=?', (manifest_hash,)).fetchone():
            raise LedgerError('reconciliation already recorded; inspect the durable history')
        history_bytes = db.execute(
            'SELECT COALESCE(SUM(LENGTH(CAST(previous_meta AS BLOB)) + '
            'LENGTH(CAST(previous_inventory AS BLOB)) + LENGTH(manifest_sha) + '
            'LENGTH(gap_sha) + LENGTH(claims_sha)), 0) FROM reconciliations').fetchone()[0]
        # The fixed-size path and content digests bound each indexed tombstone;
        # reserve 1 KiB per row as well as the byte-exact snapshot history.
        if (history_bytes + len(old_meta_text.encode('utf-8')) + len(old_inventory_text.encode('utf-8'))
                + 192 + 1024 * len(retired) > _MAX_HISTORY):
            raise LedgerError('reconciliation history exceeds the ledger allowance')
        if shutil.disk_usage(ledger.parent).free < REQUIRED_FREE_DISK_BYTES:
            raise LedgerError('less than 12 GiB free on ledger filesystem')
        db.execute('INSERT INTO reconciliations(manifest_sha,gap_sha,claims_sha,previous_meta,previous_inventory) '
                   'VALUES(?,?,?,?,?)', (manifest_hash, gap_hash, claims_hash, old_meta_text, old_inventory_text))
        new_history_hash = _history_digest(db)
        new['current_history_sha256'] = new_history_hash
        new_meta_text = json.dumps(new, sort_keys=True, separators=(',', ':'))
        _check_metadata_bound(new_meta_text, new_inventory_text)
        db.execute('UPDATE meta SET data=?, inventory=? WHERE id=1', (new_meta_text, new_inventory_text))
        db.execute('COMMIT')
    except sqlite3.Error as exc:
        raise LedgerError('reconciliation transaction failed or is invalid') from exc
    finally:
        db.close()
    return {'manifest_sha256': manifest_hash, 'claims_sha256': claims_hash,
            'previous_state_sha256': state_hash,
            'previous_history_sha256': history_hash, 'new_history_sha256': new_history_hash,
            'new_state_sha256': _snapshot_digest(new_meta_text, new_inventory_text),
            'natural_physical_bytes': new['natural_physical_bytes'],
            'prior_physical_bytes': new['prior_physical_bytes']}


def claim_phase(ledger_path, phase, *, output_path) -> PhaseGrant:
    """Atomically and permanently consume one phase before starting work."""
    if type(phase) is not str or phase not in PHASE_LIMITS or phase == 'natural_discovery':
        raise LedgerError('unknown or already imported phase')
    ledger = _path(ledger_path, existing=True)
    output = _path(output_path, existing=False)
    free = shutil.disk_usage(output.parent).free
    if free < REQUIRED_FREE_DISK_BYTES:
        raise LedgerError('less than 12 GiB free on output filesystem')
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
        if meta.get('schema') not in (_SCHEMA, _RECONCILED_SCHEMA):
            raise LedgerError('unknown ledger schema')
        if meta.get('schema') == _RECONCILED_SCHEMA:
            _digest(meta.get('evidence_scope_amendment_sha256'), 'evidence scope amendment')
            _digest(meta.get('initial_gap_record_sha256'), 'initial gap record')
            if _history_digest(db) != _digest(meta.get('current_history_sha256'), 'current history digest'):
                raise LedgerError('stored ledger history digest differs')
        if (meta.get('phase_limits') != _phase_snapshot()
                or meta.get('study_logical_limit') != STUDY_LOGICAL_LIMIT
                or meta.get('study_disk_limit') != STUDY_DISK_LIMIT
                or meta.get('required_free_disk_bytes') != REQUIRED_FREE_DISK_BYTES
                or meta.get('metadata_max_bytes') != _MAX_INVENTORY
                or meta.get('max_files_per_inventory') != _MAX_FILES
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
        prior_floor = _nat(meta.get('prior_physical_floor_bytes', 0), 'prior_physical_floor_bytes')
        natural_floor = _nat(meta.get('natural_physical_floor_bytes', 0), 'natural_physical_floor_bytes')
        historical, _ = _retired_charges(db, inventory)
        if (sum(v[0] for v in inventory['prior'].values()) + prior_floor != prior_disk
                or sum(v[0] for v in inventory['natural'].values()) + natural_floor != meta.get('natural_physical_bytes')
                or historical != {'natural': natural_floor, 'prior': prior_floor}):
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
            raise LedgerError('less than 12 GiB free on output filesystem')
        logical, disk = PHASE_LIMITS[phase]
        db.execute('INSERT INTO claims VALUES (?, ?, ?, ?, ?)', (phase, str(output), logical, disk, free))
        db.execute('COMMIT')
    except sqlite3.Error as exc:
        raise LedgerError('ledger transaction failed or is invalid') from exc
    finally:
        db.close()
    return PhaseGrant(phase, str(output), logical, disk)
