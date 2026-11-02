"""Observe one public CPU child; the operator must not retry an incomplete attempt.

The receipt is an operator observation, not a secure process attestation. A
parent interruption at process creation can leave uncertain descendant custody.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time


INHERITED_NAMES = ('SYSTEMROOT', 'WINDIR', 'PATH', 'PATHEXT', 'TEMP', 'TMP', 'COMSPEC',
                   'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'APPDATA', 'LOCALAPPDATA',
                   'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE')
OVERRIDE_NAMES = frozenset(('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_CBWR',
                            'ATEN_CPU_CAPABILITY'))
FIXED_ENVIRONMENT = {'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1',
                     'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'MKL_CBWR': None,
                     'ATEN_CPU_CAPABILITY': None}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inherited_environment_hashes():
    return {key: sha(os.environ[key].encode('utf-8', errors='surrogatepass'))
            for key in INHERITED_NAMES if key in os.environ}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path, raw):
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short observation write')
        stream.flush()
        os.fsync(stream.fileno())


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def bounded_read(path, maximum):
    with path.open('rb') as stream:
        result = stream.read(maximum + 1)
    if len(result) > maximum:
        raise ValueError('bounded observation input exceeded')
    return result


def inventory(root, max_bytes=1024**3, max_entries=15000, max_depth=8, *, hash_files=True):
    result = []
    total = 0
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return result
    reparse = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    if (root.is_symlink() or getattr(root_stat, 'st_file_attributes', 0) & reparse
            or not stat.S_ISDIR(root_stat.st_mode)):
        raise ValueError('attempt inventory root is not a regular directory')
    pending, visited = [(root, 0)], 0
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                visited += 1
                if visited > max_entries:
                    raise ValueError('attempt entry count exceeds its bound')
                entry_stat = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(entry_stat, 'st_file_attributes', 0) & reparse:
                    raise ValueError('attempt output contains a link or reparse point')
                if entry.is_dir(follow_symlinks=False):
                    if depth >= max_depth:
                        raise ValueError('attempt directory depth exceeds its bound')
                    pending.append((Path(entry.path), depth + 1))
                elif entry.is_file(follow_symlinks=False):
                    path = Path(entry.path)
                    size = entry_stat.st_size
                    total += size
                    if total > max_bytes:
                        raise ValueError('attempt physical files exceed their bound')
                    record = {'path': path.relative_to(root).as_posix(), 'bytes': size}
                    if hash_files:
                        h, observed = hashlib.sha256(), 0
                        with path.open('rb') as stream:
                            while observed <= size:
                                chunk = stream.read(min(1024 * 1024, size + 1 - observed))
                                if not chunk:
                                    break
                                h.update(chunk)
                                observed += len(chunk)
                        if observed != size or path.stat().st_size != size:
                            raise ValueError('attempt output changed during hashing')
                        record['sha256'] = h.hexdigest()
                    result.append(record)
                else:
                    raise ValueError('attempt output is not a regular file or directory')
    return sorted(result, key=lambda item: item['path'])


def source_check(repo, owner, entry, git_env):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, env=git_env, timeout=60).decode().strip()
    tree = subprocess.check_output(['git', 'show', '-s', '--format=%T', 'HEAD'], cwd=repo, env=git_env, timeout=60).decode().strip()
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'],
                                     cwd=repo, env=git_env, timeout=60)
    if head != owner['source_revision'] or tree != owner['source_tree'] or status:
        raise ValueError('source checkout is not the exact clean recorded revision')
    executor = (repo / owner['executor_path']).resolve()
    if (not entry.is_relative_to(repo) or not entry.is_file()
            or not executor.is_relative_to(repo) or not executor.is_file()
            or sha(bounded_read(entry, 1024 * 1024)) != owner['entry_script_sha256']
            or sha(bounded_read(executor, 1024 * 1024)) != owner['executor_sha256']):
        raise ValueError('recorded entry or executor source differs')
    return head, tree


