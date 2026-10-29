"""Private execution plumbing for the prospectively bound injection study.

This module does not allocate a split or authorize an execution on import.
The owner must independently verify publication, source/runtime, the phase
claim, the preparation observer and the common cross-track gate first.
"""
import builtins
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import threading
import time
from types import FunctionType, ModuleType, SimpleNamespace


SOURCE_COMMIT = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
GATE_ROOT_IDENTITY = (1219152768, 4503599632054271)
GATE_LOCK_IDENTITY = (1219152768, 6192449492328171)


def _plain_path(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('an absolute path is required')
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('linked or reparse source paths are unsupported')
    return path


def _read_source(path, limit):
    path = _plain_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError('source is not a bounded regular file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    after = path.lstat()
    if (len(raw) != before.st_size or len(raw) > limit
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('source changed while it was read')
    return raw


def verify_gate_identity(root, *, expected_root, root_identity, lock_identity):
    """Check the shared gate and lock are the same protected local objects."""
    for identity in (root_identity, lock_identity):
        if (type(identity) is not tuple or len(identity) != 2
                or any(type(value) is not int or value <= 0 for value in identity)):
            raise ValueError('shared gate identity is invalid')
    root, expected = _plain_path(root), _plain_path(expected_root)
    if os.path.normcase(str(root)) != os.path.normcase(str(expected)):
        raise ValueError('shared gate root differs')
    root_info, lock_info = root.lstat(), _plain_path(root / 'gate.lock').lstat()
    if (not stat.S_ISDIR(root_info.st_mode) or (root_info.st_dev, root_info.st_ino) != root_identity
            or not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1 or lock_info.st_size > 8192
            or (lock_info.st_dev, lock_info.st_ino) != lock_identity):
        raise ValueError('shared gate root or lock identity differs')
    return True


def load_verified_source(path, expected_sha256, module_name):
    """Hash exact raw bytes before compiling; never read project bytecode."""
    if (type(expected_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None
            or type(module_name) is not str or not all(part.isidentifier() for part in module_name.split('.'))
            or module_name in sys.modules):
        raise ValueError('source identity or module name is invalid')
    source = _read_source(path, 16 * 1024 * 1024)
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('source differs from its bound digest')
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition('.')[0]
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def verify_frozen_inventory(root, record_bytes, expected_sha256):
    """Rehash a parent-verified e84 tree without starting child processes.

    The trusted parent must verify Git identity and bind this record's digest
    to the published runtime before a gated child can use this helper.
    """
    if (type(record_bytes) is not bytes or not 0 < len(record_bytes) <= 2 * 1024 * 1024
            or type(expected_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None
            or hashlib.sha256(record_bytes).hexdigest() != expected_sha256):
        raise ValueError('frozen source inventory digest differs')
    try:
        record = json.loads(record_bytes)
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')
        if (canonical != record_bytes or type(record) is not dict or set(record) != {'git_commit', 'git_tree', 'files'}
                or record['git_commit'] != SOURCE_COMMIT or type(record['git_tree']) is not str
                or re.fullmatch(r'[0-9a-f]{40}', record['git_tree']) is None
                or type(record['files']) is not dict or not 0 < len(record['files']) <= 4096
                or 'looking_glass/__init__.py' not in record['files']):
            raise ValueError('frozen source inventory is malformed')
    except (TypeError, KeyError, UnicodeError, RecursionError) as error:
        raise ValueError('frozen source inventory is malformed') from error
    root = _plain_path(root)
    if not root.is_dir():
        raise ValueError('frozen source root is unavailable')
    total, names = 0, set(record['files'])
    for name, digest in record['files'].items():
        relative = PurePosixPath(name) if type(name) is str else None
        if (relative is None or not name or relative.is_absolute() or relative.as_posix() != name
                or '..' in relative.parts or '\\' in name or ':' in name
                or type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None):
            raise ValueError('frozen source path or digest is invalid')
        raw = _read_source(root.joinpath(*relative.parts), 64 * 1024 * 1024)
        total += len(raw)
        if total > 512 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('frozen source bytes differ')
    package = root / 'looking_glass'
    _plain_path(package)
    def fail_walk(error):
        raise ValueError('frozen package could not be enumerated') from error
    for base, directories, files in os.walk(package, followlinks=False, onerror=fail_walk):
        directories[:] = [name for name in directories if name != '__pycache__']
        for name in directories:
            _plain_path(Path(base) / name)
        for name in files:
            entry = _plain_path(Path(base) / name)
            if entry.relative_to(root).as_posix() not in names:
                raise ValueError('untracked frozen package file is present')
    return record


def _cpu_job_adapter(cpu_job_type):
    """Use the discovery Job's active-count semantics with inherited leases.

    Zero active members and a closed Job establish current cooperative Job
    cleanup; they do not establish a complete lifetime process-handle history.
    """
    class CpuJobAdapter:
        def __init__(self):
            # The generic supervisor roots this adapter before __init__. Root
            # the nested job before its native constructor can be interrupted.
            self._job = cpu_job_type.__new__(cpu_job_type)
            cpu_job_type.__init__(self._job)

        def assign(self, child):
            return self._job.assign(child)

        def _accounting(self):
            active = self._job.active()
            if type(active) is not int or active < 0:
                raise ValueError('invalid CPU Job active count')
            return SimpleNamespace(active_processes=active)

        def terminate_and_wait(self, timeout=5):
            if not 0 < timeout <= 10:
                raise ValueError('invalid CPU Job drain timeout')
            active = self._job.active()
            if type(active) is not int or active < 0:
                raise ValueError('invalid CPU Job active count')
            if active:
                self._job.terminate()
            deadline = time.monotonic() + timeout
            while active:
                if time.monotonic() >= deadline:
                    raise TimeoutError('CPU Job did not drain')
                time.sleep(0.05)
                active = self._job.active()
                if type(active) is not int or active < 0:
                    raise ValueError('invalid CPU Job active count')

        def close(self):
            return self._job.close()
    return CpuJobAdapter


def supervise_with_cpu_job(command, output, environment, *, audit_result, supervisor,
                           guard_module, cpu_job_type, timeout=7200, inherited_handles=()):
    """Run the reviewed gated supervisor with the frozen discovery CPU Job.

    Only this function clone's import namespace is changed; the shared guard
    module and supervisor's rooted custody state are untouched. The owner must
    bind the exact modules in a dedicated process. A nonzero or unknown active
    count after child exit prevents promotion. Retain supervisor/source custody
    when cleanup is uncertain.
    """
    function = getattr(supervisor, 'supervise_heldout', None)
    if (type(function) is not FunctionType or function.__globals__ is not supervisor.__dict__
            or sys.modules.get('scripts.v2_source_guards') is not guard_module):
        raise ValueError('the verified gated supervisor is required')
    adapter = _cpu_job_adapter(cpu_job_type)
    original_import = builtins.__import__
    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'scripts.v2_source_guards' and tuple(fromlist) == ('WindowsJob',) and level == 0:
            return SimpleNamespace(WindowsJob=adapter)
        return original_import(name, globals, locals, fromlist, level)
    namespace = dict(supervisor.__dict__)
    namespace['__builtins__'] = dict(vars(builtins), __import__=private_import)
    cloned = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned(command, Path(output), environment, audit_result=audit_result,
                  timeout=timeout, inherited_handles=inherited_handles)
