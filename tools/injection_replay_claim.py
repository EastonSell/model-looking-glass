"""One-use private preclaim for the prospective injection compatibility check.

The trusted caller must verify the published plan, its source dependencies,
private-parent ACL, current inventory, and ledger implementation. This helper
does not launch a model, draw a split, or attest its supplied callbacks.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat


_MIB = 1024 * 1024
_FREEZE = 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'
_PARENT_NAME = 'number-injection-heldout-cpu-v1'
_LEDGER = ('agreement-study-ledger-v5', 'ledger.sqlite3')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _ordinary(path, *, directory):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('absolute protected paths are required')
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('protected path is linked or a reparse point')
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode)):
        raise ValueError('protected path has the wrong kind')
    if not directory and info.st_nlink != 1:
        raise ValueError('protected file is multiply linked')
    return info


def _identity(path, expected, *, directory):
    if (type(expected) is not tuple or len(expected) != 2
            or any(type(value) is not int or value <= 0 for value in expected)):
        raise ValueError('protected identity is invalid')
    info = _ordinary(path, directory=directory)
    if (info.st_dev, info.st_ino) != expected:
        raise ValueError('protected identity changed')


def _diagnostic_bytes(root):
    _ordinary(root, directory=True)
    pending, count, total, inventory = [(root, 0)], 0, 0, []
    while pending:
        directory, depth = pending.pop()
        if depth > 8:
            raise ValueError('diagnostic tree is too deep')
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > 128:
                    raise ValueError('diagnostic inventory is too large')
                path = directory / entry.name
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('diagnostic contains a linked path')
                if stat.S_ISDIR(info.st_mode):
                    inventory.append({'path': path.relative_to(root).as_posix(), 'kind': 'directory'})
                    pending.append((path, depth + 1))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    total += info.st_size
                    if total > 4 * _MIB:
                        raise ValueError('diagnostic exceeds its reserved bytes')
                    with path.open('rb') as stream:
                        raw = stream.read(info.st_size + 1)
                    after = path.lstat()
                    if (len(raw) != info.st_size or
                            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
                        raise ValueError('diagnostic changed while being read')
                    inventory.append({'path': path.relative_to(root).as_posix(), 'kind': 'file',
                                      'bytes': info.st_size, 'sha256': hashlib.sha256(raw).hexdigest()})
                else:
                    raise ValueError('diagnostic contains a nonordinary entry')
    inventory.sort(key=lambda entry: entry['path'])
    return total, hashlib.sha256(_canonical(inventory)).hexdigest()


def _write_once(path, raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError('claim record is invalid or oversized')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if type(count) is not int or not 0 < count <= len(view):
                raise OSError('short claim record write')
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _readback(path, expected):
    info = _ordinary(path, directory=False)
    with path.open('rb') as stream:
        observed = stream.read(len(expected) + 1)
    if info.st_size != len(expected) or observed != expected:
        raise ValueError('claim record could not be read back exactly')


def _checked_row(callback, ledger, phase):
    row = callback(ledger, phase)
    if (type(row) is not list or len(row) != 5 or type(row[0]) is not str or row[0] != 'injection_heldout'
            or type(row[1]) is not str or row[1] != str(phase)
            or type(row[2]) is not int or row[2] != 72 * _MIB
            or type(row[3]) is not int or row[3] != 192 * _MIB
            or type(row[4]) is not int or not 12 * 1024**3 <= row[4] < 2**63):
        raise ValueError('protected phase row differs from the reserved grant')
    return row, hashlib.sha256(json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')).hexdigest()


def claim_replay_once(*, workspace, plan_bytes, plan_sha256, parent_identity, ledger_identity,
                      verify_authority, verify_claim_row, claim_phase):
    """Consume the fixed phase once, preserving intent on all ambiguous outcomes."""
    if (type(plan_bytes) is not bytes or not 0 < len(plan_bytes) <= 2 * _MIB
            or type(plan_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', plan_sha256) is None
            or hashlib.sha256(plan_bytes).hexdigest() != plan_sha256
            or any(not callable(item) for item in (verify_authority, verify_claim_row, claim_phase))):
        raise ValueError('published replay plan or trusted callbacks are invalid')
    workspace = Path(workspace)
    _ordinary(workspace, directory=True)
    work = workspace / 'work'
    parent = work / _PARENT_NAME
    ledger = work.joinpath(*_LEDGER)
    rehearsal = work / 'injection-heldout-runtime-rehearsal-v1'
    phase = parent / ('injection-heldout-' + _FREEZE)
    pilot = parent / 'runtime-compatibility-replay-v1'
    intent, receipt_path = parent / 'replay-claim-intent.json', parent / 'replay-claim-receipt.json'
    _identity(parent, parent_identity, directory=True)
    _identity(ledger, ledger_identity, directory=False)
    for path in (phase, pilot, intent, receipt_path):
        if os.path.lexists(path):
            raise ValueError('this preclaim or an affected output already exists')
    diagnostic_bytes, diagnostic_sha256 = _diagnostic_bytes(rehearsal)
    if verify_authority() is not True:
        raise ValueError('replay publication, source or private-parent check failed')
    intent_record = {'schema': 'mlg-injection-replay-claim-intent-v1', 'plan_sha256': plan_sha256,
                     'ledger_path': str(ledger), 'phase_path': str(phase), 'pilot_path': str(pilot),
                     'diagnostic_path': str(rehearsal), 'parent_identity': list(parent_identity),
                     'ledger_identity': list(ledger_identity), 'diagnostic_actual_bytes': diagnostic_bytes,
                     'diagnostic_inventory_sha256': diagnostic_sha256}
    intent_raw = _canonical(intent_record)
    _write_once(intent, intent_raw)
    _readback(intent, intent_raw)
    grant = claim_phase(ledger, 'injection_heldout', output_path=phase)
    _identity(parent, parent_identity, directory=True)
    _identity(ledger, ledger_identity, directory=False)
    row, row_sha256 = _checked_row(verify_claim_row, ledger, phase)
    if (getattr(grant, 'phase', None) != 'injection_heldout' or getattr(grant, 'output_path', None) != str(phase)
            or type(getattr(grant, 'logical_limit_bytes', None)) is not int or grant.logical_limit_bytes != 72 * _MIB
            or type(getattr(grant, 'disk_limit_bytes', None)) is not int or grant.disk_limit_bytes != 192 * _MIB
            or verify_authority() is not True
            or _diagnostic_bytes(rehearsal) != (diagnostic_bytes, diagnostic_sha256)
            or any(os.path.lexists(path) for path in (phase, pilot, receipt_path))):
        raise ValueError('the one-use phase claim or protected inputs did not verify')
    _readback(intent, intent_raw)
    if _checked_row(verify_claim_row, ledger, phase) != (row, row_sha256):
        raise ValueError('protected phase row changed before receipt')
    receipt = {'schema': 'mlg-injection-replay-claim-receipt-v1', 'status': 'preclaimed',
               'plan_sha256': plan_sha256, 'intent_sha256': hashlib.sha256(intent_raw).hexdigest(),
               'ledger_path': str(ledger), 'phase_path': str(phase), 'pilot_path': str(pilot),
               'diagnostic_path': str(rehearsal), 'parent_identity': list(parent_identity),
               'ledger_identity': list(ledger_identity), 'phase_logical_bytes': 72 * _MIB,
               'phase_physical_bytes': 192 * _MIB, 'pilot_logical_bytes': 8 * _MIB,
               'pilot_physical_bytes': 16 * _MIB, 'diagnostic_reserved_bytes': 4 * _MIB,
               'diagnostic_actual_bytes': diagnostic_bytes, 'diagnostic_inventory_sha256': diagnostic_sha256,
               'injection_claim_row_sha256': row_sha256}
    receipt_raw = _canonical(receipt)
    _write_once(receipt_path, receipt_raw)
    _readback(receipt_path, receipt_raw)
    _identity(parent, parent_identity, directory=True)
    _identity(ledger, ledger_identity, directory=False)
    if _checked_row(verify_claim_row, ledger, phase) != (row, row_sha256):
        raise ValueError('protected phase row changed during receipt write')
    return receipt
