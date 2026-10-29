"""Bounded metadata-only accounting for a preclaimed injection phase.

This cooperative check measures retained paths; it does not authenticate the
caller-supplied roots, replay result, model tensor count or publication history.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import stat


_MIB = 1024 ** 2
_PILOT_BYTES = 16 * _MIB
_REHEARSAL_BYTES = 4 * _MIB
_OTHER_BYTES = 172 * _MIB
_TOTAL_BYTES = 192 * _MIB
_PILOT_LOGICAL = 8 * _MIB
_MAX_ENTRIES = 8192
_MAX_DEPTH = 32
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


class AccountingError(ValueError):
    """The fixed phase inventory is incomplete, changed or over budget."""


@dataclass(frozen=True)
class PhaseRoots:
    parent_path: str
    parent_dev: int
    parent_ino: int
    phase_path: str
    pilot_path: str
    pilot_dev: int
    pilot_ino: int
    rehearsal_path: str
    rehearsal_dev: int
    rehearsal_ino: int


def _absolute(value):
    if not isinstance(value, (str, os.PathLike)):
        raise AccountingError('an anchored absolute path is required')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise AccountingError('an anchored absolute path is required')
    return path


def _normalized(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _directory(path, dev, ino):
    info = path.lstat()
    if (type(dev) is not int or type(ino) is not int or dev < 0 or ino <= 0
            or not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or getattr(info, 'st_file_attributes', 0) & _REPARSE
            or (info.st_dev, info.st_ino) != (dev, ino)
            or _normalized(path.resolve(strict=True)) != _normalized(path)):
        raise AccountingError('an anchored directory identity changed')


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink)


def _snapshot(root):
    """Capture bounded file and directory identities without reading contents."""
    records = {}
    todo = [(root, '', 0)]
    while todo:
        directory, relative, depth = todo.pop()
        if depth > _MAX_DEPTH:
            raise AccountingError('phase inventory nesting exceeds the bound')
        before = directory.lstat()
        if (not stat.S_ISDIR(before.st_mode) or directory.is_symlink() or not before.st_ino
                or getattr(before, 'st_file_attributes', 0) & _REPARSE):
            raise AccountingError('phase inventory contains an unsafe directory')
        records[relative] = ('directory',) + _identity(before)
        with os.scandir(directory) as entries:
            for entry in entries:
                child = Path(entry.path)
                rel = (relative + '/' if relative else '') + entry.name
                if len(rel) > 4096 or len(records) + len(todo) >= _MAX_ENTRIES:
                    raise AccountingError('phase inventory exceeds its bound')
                # On Windows DirEntry.stat may not populate the file identity.
                info = child.lstat()
                if (not info.st_ino or getattr(info, 'st_file_attributes', 0) & _REPARSE
                        or stat.S_ISLNK(info.st_mode)):
                    raise AccountingError('phase inventory contains a linked path')
                if stat.S_ISDIR(info.st_mode):
                    todo.append((child, rel, depth + 1))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size >= 0:
                    if _identity(child.lstat()) != _identity(info):
                        raise AccountingError('phase file changed during inventory')
                    records[rel] = ('file',) + _identity(info)
                else:
                    raise AccountingError('phase inventory contains a nonregular or hardlinked file')
        if _identity(directory.lstat()) != _identity(before):
            raise AccountingError('phase directory changed during inventory')
    return records


def reconcile_phase_files(roots, *, stage, replay_tensor_bytes):
    """Check both physical side reservations and the current preparation stage.

    The trusted caller anchors the directories and independently verifies the
    replay's reported tensor bytes and successful process/evidence. Call twice:
    `before_phase_creation` requires no phase directory; `before_nonce` requires
    the phase to contain only its bounded `claim.json`. The owner's separate
    durable launch record, current ledger/free-space checks, private ACLs and
    source-first protections are required independently.
    """
    try:
        if (type(roots) is not PhaseRoots or stage not in ('before_phase_creation', 'before_nonce')
                or type(replay_tensor_bytes) is not int or not 0 <= replay_tensor_bytes <= _PILOT_LOGICAL):
            raise AccountingError('phase accounting arguments are invalid')
        parent, phase, pilot, rehearsal = map(_absolute, (roots.parent_path, roots.phase_path,
                                                          roots.pilot_path, roots.rehearsal_path))
        if (_normalized(phase.parent) != _normalized(parent) or _normalized(pilot.parent) != _normalized(parent)
                or _normalized(phase) == _normalized(pilot)):
            raise AccountingError('the phase and replay roots must be distinct direct children')
        try:
            common = os.path.commonpath((_normalized(parent), _normalized(rehearsal)))
        except ValueError:
            common = None
        if common in (_normalized(parent), _normalized(rehearsal)):
            raise AccountingError('the rehearsal root overlaps the protected phase parent')
        for path, dev, ino in ((parent, roots.parent_dev, roots.parent_ino),
                               (pilot, roots.pilot_dev, roots.pilot_ino),
                               (rehearsal, roots.rehearsal_dev, roots.rehearsal_ino)):
            _directory(path, dev, ino)
        first_parent, first_rehearsal = _snapshot(parent), _snapshot(rehearsal)
        second_parent, second_rehearsal = _snapshot(parent), _snapshot(rehearsal)
        if first_parent != second_parent or first_rehearsal != second_rehearsal:
            raise AccountingError('phase evidence changed between inventories')
        for path, dev, ino in ((parent, roots.parent_dev, roots.parent_ino),
                               (pilot, roots.pilot_dev, roots.pilot_ino),
                               (rehearsal, roots.rehearsal_dev, roots.rehearsal_ino)):
            _directory(path, dev, ino)
        if len(first_parent) + len(first_rehearsal) > _MAX_ENTRIES:
            raise AccountingError('phase inventory exceeds its bound')
        phase_key, pilot_key = phase.name, pilot.name
        marker_key = phase_key + '.preparation-started.json'
        marker = first_parent.get(marker_key)
        if marker is None or marker[0] != 'file' or not 0 < marker[3] <= 8192:
            raise AccountingError('the preparation handoff marker is missing or oversized')
        phase_items = {name for name in first_parent if name == phase_key or name.startswith(phase_key + '/')}
        if stage == 'before_phase_creation':
            if phase_items:
                raise AccountingError('the phase already exists before creation')
        elif (phase_items != {phase_key, phase_key + '/claim.json'}
              or first_parent[phase_key][0] != 'directory'
              or first_parent[phase_key + '/claim.json'][0] != 'file'
              or not 0 < first_parent[phase_key + '/claim.json'][3] <= 8192):
            raise AccountingError('the phase is not claim-only before the nonce')
        if first_parent.get(pilot_key, (None,))[0] != 'directory':
            raise AccountingError('the retained replay directory is missing')
        pilot_files = [value for name, value in first_parent.items()
                       if name.startswith(pilot_key + '/') and value[0] == 'file']
        rehearsal_files = [value for value in first_rehearsal.values() if value[0] == 'file']
        other_files = [value for name, value in first_parent.items()
                       if not name.startswith(pilot_key + '/') and value[0] == 'file']
        files = pilot_files + rehearsal_files + other_files
        identities = [(value[1], value[2]) for value in files]
        if (not pilot_files or not rehearsal_files or len(identities) != len(set(identities))):
            raise AccountingError('retained side evidence is absent or multiply linked')
        pilot_bytes = sum(value[3] for value in pilot_files)
        rehearsal_bytes = sum(value[3] for value in rehearsal_files)
        other_bytes = sum(value[3] for value in other_files)
        if (pilot_bytes > _PILOT_BYTES or rehearsal_bytes > _REHEARSAL_BYTES
                or other_bytes > _OTHER_BYTES or pilot_bytes + rehearsal_bytes + other_bytes > _TOTAL_BYTES):
            raise AccountingError('retained phase files exceed a fixed reservation')
        return {'status': 'bounded', 'stage': stage, 'pilot_bytes': pilot_bytes,
                'rehearsal_bytes': rehearsal_bytes, 'other_bytes': other_bytes,
                'replay_tensor_bytes': replay_tensor_bytes, 'file_count': len(files)}
    except AccountingError:
        raise
    except (OSError, TypeError, ValueError, KeyError, RecursionError):
        raise AccountingError('phase evidence could not be reconciled') from None
