"""Point-in-time accounting for a separate natural successor phase.

These checks count apparent file sizes, not allocated blocks or reserved space.
The owner must source-bind this module, reconcile old study evidence separately,
and ensure no writer races the scans. They neither authorize a claim nor alter
the old study's ledger or its separate free-space threshold.
"""
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat


TENSOR_LIMIT = 160 * 1024 * 1024
EVENT_LIMIT = 48 * 1024 * 1024
SIDE_LIMIT = 16 * 1024 * 1024
TOTAL_LIMIT = 224 * 1024 * 1024
FREE_THRESHOLD = 3 * 1024 * 1024 * 1024
_MAX_ENTRIES = 10000


def _directory(path):
    path = Path(path)
    if (not path.is_absolute()
            or os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.realpath(path))):
        raise ValueError('an ordinary absolute directory is required')
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('directory path has a reparse point')
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError('directory is unavailable')
    return path


def _files(root):
    count = 0
    def failed(_error):
        raise ValueError('evidence could not be enumerated')
    for directory, names, filenames in os.walk(root, followlinks=False, onerror=failed):
        for name in names:
            info = (Path(directory) / name).lstat()
            if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or getattr(info, 'st_file_attributes', 0) & 1024):
                raise ValueError('evidence contains a linked directory')
            count += 1
        for name in filenames:
            path = Path(directory) / name
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or getattr(info, 'st_file_attributes', 0) & 1024 or info.st_nlink != 1):
                raise ValueError('evidence contains a linked or nonregular file')
            count += 1
            if count > _MAX_ENTRIES:
                raise ValueError('evidence exceeds the entry bound')
            yield path, info.st_size
        if count > _MAX_ENTRIES:
            raise ValueError('evidence exceeds the entry bound')


def _tensor_payload(path, relative, size):
    match = re.fullmatch(r'owner/attempt/tensors/([0-9a-f]{64})\.f32le', relative)
    if match is None or size % 4:
        return False
    if size > TENSOR_LIMIT:
        raise ValueError('tensor evidence exceeds its allowance')
    before = path.lstat()
    digest, length = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            length += len(block)
            if length > size:
                raise ValueError('tensor evidence changed while reading')
            digest.update(block)
    after = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink)
    if length != size or identity(before) != identity(after):
        raise ValueError('tensor evidence changed while reading')
    return digest.hexdigest() == match.group(1)


def account_phase(root, *, reserve_side_bytes=0):
    """Count every file, including partials, under the separate phase root."""
    root = _directory(root)
    if type(reserve_side_bytes) is not int or not 0 <= reserve_side_bytes <= SIDE_LIMIT:
        raise ValueError('invalid side reserve')
    totals = {'tensor_bytes': 0, 'event_bytes': 0, 'side_bytes': 0}
    for path, size in _files(root):
        relative = path.relative_to(root).as_posix()
        category = ('tensor_bytes' if _tensor_payload(path, relative, size) else
                    'event_bytes' if relative == 'owner/attempt/events.jsonl' else 'side_bytes')
        totals[category] += size
    if (totals['tensor_bytes'] > TENSOR_LIMIT or totals['event_bytes'] > EVENT_LIMIT
            or totals['side_bytes'] + reserve_side_bytes > SIDE_LIMIT
            or sum(totals.values()) + reserve_side_bytes > TOTAL_LIMIT):
        raise ValueError('successor evidence exceeds its separate allowance')
    return totals


def inspect_resources(phase, cache):
    """Observe fresh free space and retained checkpoint stages; delete nothing."""
    phase, cache = _directory(phase), _directory(cache)
    if phase == cache or phase in cache.parents or cache in phase.parents:
        raise ValueError('phase and checkpoint cache must be disjoint')
    phase_free, cache_free = shutil.disk_usage(phase).free, shutil.disk_usage(cache).free
    if min(phase_free, cache_free) < FREE_THRESHOLD:
        raise ValueError('successor volume has insufficient free space')
    stages = []
    for path in sorted(cache.iterdir(), key=lambda item: item.name):
        if not path.name.startswith(('looking-glass-stage-', 'agreement-tokenizer-')):
            continue
        _directory(path)
        files = list(_files(path))
        stages.append({'name': path.name, 'files': len(files),
                       'apparent_bytes': sum(size for _, size in files)})
    return {'phase_free_bytes': phase_free, 'cache_free_bytes': cache_free, 'stages': stages}
