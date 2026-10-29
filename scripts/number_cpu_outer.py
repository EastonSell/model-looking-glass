"""Bounded Windows parent for a single CPU number-study launcher command.

The private output directory belongs to a trusted custodian. This is a
cooperative process/byte bound, not a sandbox or proof of scientific results.
"""
import argparse
import base64
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time


_MIB = 1024 * 1024
_PIPE_CAP = 256 * 1024
_META_CAP = 16 * 1024
_PHASES = {'prepare': 'prepared', 'plan': 'plan-created',
           'seal': 'commitment-created', 'run': 'complete-local'}
_DESTINATIONS = {'prepare': '--runtime-output', 'plan': '--plan-dir',
                 'seal': '--custody-dir', 'run': '--attempt-dir'}
_MIN_STAGE_FREE = 1024 * 1024 * 1024
_REASON_CODES = tuple(sorted((
    'job_init', 'spawn', 'reader_start', 'job_assign', 'grant', 'child_wait', 'timeout',
    'job_termination', 'child_reap', 'job_not_empty', 'job_query', 'job_close',
    'reader_not_stopped', 'reader_join', 'source_or_prefix_check', 'cache_stage',
    'cache_stage_check', 'completion', 'child_exit', 'job_state', 'prefix', 'source',
    'output_overflow', 'capture', 'cancelled', 'log_write', 'log_mismatch', 'log_readback')))
_REASON_INDEX = {name: index for index, name in enumerate(_REASON_CODES)}
_OWNER_LOCK = threading.Lock()
# This preallocated slot roots the Job, direct child and readers before native
# initialization/start. It is cleared only after their confirmed cleanup.
_OWNER_SLOT = [None]


class _BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64), ('flags', wintypes.DWORD),
                ('min_working', ctypes.c_size_t), ('max_working', ctypes.c_size_t),
                ('process_limit', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', _BasicLimits), ('io', _IoCounters), ('process_memory', ctypes.c_size_t),
                ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [('user', ctypes.c_int64), ('kernel', ctypes.c_int64),
                ('period_user', ctypes.c_int64), ('period_kernel', ctypes.c_int64),
                ('faults', wintypes.DWORD), ('total', wintypes.DWORD),
                ('active', wintypes.DWORD), ('terminated', wintypes.DWORD)]


class WindowsJob:
    """Contain descendants; a zero active count is not a lifetime inventory."""
    def __init__(self):
        self.handle = None
        if os.name != 'nt':
            raise OSError('the CPU study owner requires Windows')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD)
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
        self.kernel.IsProcessInJob.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = (wintypes.HANDLE, wintypes.INT, wintypes.LPVOID,
                                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError('cannot create study process job')
        try:
            if os.get_handle_inheritable(self.handle):
                raise OSError('study process job is inheritable')
            limits = _ExtendedLimits()
            limits.basic.flags = 0x00002000  # KILL_ON_JOB_CLOSE; no breakaway flag.
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError('cannot configure study process job')
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                    raise cleanup from primary
                raise primary from cleanup
            raise

    def assign(self, process):
        if process.poll() is not None or not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise OSError('cannot assign gated study child')
        member = wintypes.BOOL()
        if not self.kernel.IsProcessInJob(int(process._handle), self.handle, ctypes.byref(member)) or not member.value:
            raise OSError('gated study child membership could not be verified')

    def active(self):
        info = _Accounting()
        if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise OSError('cannot inspect study process job')
        return int(info.active)

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise OSError('cannot terminate study process job')

    def close(self):
        if self.handle:
            if not self.kernel.CloseHandle(self.handle):
                raise OSError('cannot close study process job')
            self.handle = None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf8')


def _write_new(path, data, maximum):
    if type(data) is not bytes or len(data) > maximum:
        raise ValueError('private output exceeds its fixed bound')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if type(written) is not int or not 0 < written <= len(view):
                raise OSError('private output write was incomplete')
            view = view[written:]
        os.fsync(fd)
    except BaseException as primary:
        try:
            os.close(fd)
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
    else:
        os.close(fd)


def _read_bounded(path, maximum):
    with Path(path).open('rb') as handle:
        data = handle.read(maximum + 1)
    if not data or len(data) > maximum:
        raise ValueError('required source or receipt is unavailable or oversized')
    return data


def _path(value, *, directory=True):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('absolute paths are required')
    path = path.resolve(strict=True)
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError('a required path has the wrong type')
    return path


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _environment(workspace, hf_home, cache):
    system_names = ('PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH')
    system = {name: os.environ[name] for name in system_names if name in os.environ}
    if not system.get('PATH'):
        raise ValueError('a trusted executable path is required')
    return {**system, 'TEMP': str(workspace), 'TMP': str(workspace),
            'MLG_STUDY_WORKSPACE': str(workspace), 'HF_HOME': str(hf_home),
            'HF_HUB_CACHE': str(cache), 'TRANSFORMERS_CACHE': str(cache),
            'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}


def _verified_sources(repo, revision):
    if type(revision) is not str or re.fullmatch(r'[0-9a-f]{40}', revision) is None:
        raise ValueError('an exact source commit is required')
    env = {name: value for name, value in os.environ.items() if not name.upper().startswith('GIT_')}
    env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                'GIT_ATTR_NOSYSTEM': '1', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0'})
    def git(*args):
        return subprocess.run(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', '-C', str(repo), *args],
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env).stdout
    if git('rev-parse', 'HEAD').strip().decode('ascii') != revision:
        raise ValueError('checkout revision differs')
    result = {}
    for name in ('scripts/number_cpu_outer.py', 'scripts/number_cpu_gate.py',
                 'scripts/pretrained_number_cpu_launcher.py'):
        raw = _read_bounded(repo / name, _MIB)
        if raw != git('show', revision + ':' + name):
            raise ValueError('owner source differs from committed bytes')
        result[name] = raw
    return result


def _capture(stream, state):
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            with state['lock']:
                state['total'] += len(chunk)
                remaining = max(0, _PIPE_CAP - len(state['data']))
                state['data'].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    state['truncated'] = True
    except BaseException:
        with state['lock']:
            state['error'] = True
    finally:
        try:
            stream.close()
        except OSError:
            with state['lock']:
                state['error'] = True


def _new_capture():
    return {'lock': threading.Lock(), 'data': bytearray(), 'total': 0, 'truncated': False, 'error': False}


def _start_capture_thread(stream, state, context, slot):
    thread = threading.Thread(target=_capture, args=(stream, state), daemon=True)
    context[slot] = thread
    thread.start()
    return thread


def _snapshot(state):
    with state['lock']:
        return bytes(state['data']), state['total'], state['truncated'], state['error']


def _option_values(arguments, name):
    values = []
    for index, arg in enumerate(arguments):
        if arg == name:
            if index + 1 >= len(arguments):
                raise ValueError('a required option value is missing')
            values.append(arguments[index + 1])
        elif arg.startswith(name + '='):
            values.append(arg[len(name) + 1:])
    return values


def _completion(phase, child_args, stdout):
    try:
        report = json.loads(stdout)
        if type(report) is not dict or report.get('status') != _PHASES[phase]:
            raise ValueError('launcher did not report expected success')
        if phase != 'run':
            destinations = _option_values(child_args, _DESTINATIONS[phase])
            if len(destinations) != 1 or not Path(destinations[0]).is_absolute():
                raise ValueError('a unique absolute stage output is required')
            path = Path(destinations[0])
            if phase == 'prepare':
                field, payload = 'runtime_sha256', _read_bounded(path, _MIB)
            elif phase == 'plan':
                field, payload = 'plan_sha256', _read_bounded(path / 'plan.json', 128 * 1024)
            else:
                field, payload = 'site_commitment', _read_bounded(path / 'commitment.txt', 65)
            digest = report.get(field)
            if type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
                raise ValueError('stage output digest is invalid')
            if (phase == 'seal' and payload != digest.encode('ascii') + b'\n') or (
                    phase != 'seal' and hashlib.sha256(payload).hexdigest() != digest):
                raise ValueError('stage output and report differ')
            return {field: digest}
        attempts = _option_values(child_args, '--attempt-dir')
        if len(attempts) != 1:
            raise ValueError('a unique run attempt directory is required')
        attempt = Path(attempts[0])
        if not attempt.is_absolute():
            raise ValueError('an absolute run attempt directory is required')
        terminal = json.loads(_read_bounded(attempt.with_name(attempt.name + '.launcher-terminal.json'), 4096))
        completed = json.loads(_read_bounded(attempt / 'private' / 'completed.json', 4096))
        prediction, archive = report.get('prediction_sha256'), report.get('archive_sha256')
        if (type(prediction) is not str or re.fullmatch(r'[0-9a-f]{64}', prediction) is None
                or type(archive) is not str or re.fullmatch(r'[0-9a-f]{64}', archive) is None
                or terminal.get('schema') != 'mlg-number-injection-launch-terminal-v1'
                or terminal.get('status') != 'complete-local' or type(terminal.get('record_count')) is not int
                or terminal.get('record_count') != 128
                or completed.get('schema') != 'mlg-number-discovery-completed-v1'
                or type(completed.get('record_count')) is not int or completed.get('record_count') != 128
                or any(value.get('prediction_sha256') != prediction or value.get('archive_sha256') != archive
                       for value in (terminal, completed))):
            raise ValueError('launcher and supervisor completion records differ')
        return {'prediction_sha256': prediction, 'archive_sha256': archive}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError('CPU study completion could not be confirmed') from error


def run_once(*, repo, source_commit, base_python, site_packages, workspace, hf_home, cache_dir,
             output_dir, child_args, timeout):
    """Run one command, retaining at most 1 MiB of private parent evidence."""
    if not _OWNER_LOCK.acquire(blocking=False):
        raise RuntimeError('another CPU study owner is active in this process')
    try:
        if _OWNER_SLOT[0] is not None:
            raise RuntimeError('a previous job has uncertain cleanup; exit and reconcile')
        return _run_once(repo=repo, source_commit=source_commit, base_python=base_python,
                         site_packages=site_packages, workspace=workspace, hf_home=hf_home,
                         cache_dir=cache_dir, output_dir=output_dir, child_args=child_args, timeout=timeout)
    finally:
        _OWNER_LOCK.release()


def _run_once(*, repo, source_commit, base_python, site_packages, workspace, hf_home, cache_dir,
              output_dir, child_args, timeout):
    repo, site = _path(repo), _path(site_packages)
    workspace, home, cache = _path(workspace), _path(hf_home), _path(cache_dir)
    python = _path(base_python, directory=False)
    if (home == workspace or not home.is_relative_to(workspace) or cache == home or not cache.is_relative_to(home)
            or type(child_args) is not tuple or not child_args or child_args[0] not in _PHASES
            or any(type(value) is not str or '\0' in value for value in child_args)
            or len(_canonical(child_args)) > 32 * 1024 or not 0 < timeout <= 86400):
        raise ValueError('CPU study invocation or cache paths are invalid')
    destinations = _option_values(child_args, _DESTINATIONS[child_args[0]])
    if len(destinations) != 1:
        if child_args[0] == 'run':
            raise ValueError('a unique run attempt directory is required')
        raise ValueError('a unique stage output path is required')
    if child_args[0] in ('prepare', 'run'):
        child_caches = _option_values(child_args, '--cache-dir')
        if len(child_caches) != 1 or _path(child_caches[0]) != cache:
            raise ValueError('the child cache must match the checked outer cache')
    output = Path(output_dir)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError('a fresh absolute private output directory is required')
    resolved = output.resolve(strict=False)
    if (not resolved.is_relative_to(workspace)
            or any(resolved == root or root in resolved.parents for root in (repo, site, home))):
        raise ValueError('private output must be outside source, packages and cache')
    path_flags = {'--cache-dir', '--runtime-output', '--runtime', '--plan-dir', '--plan',
                  '--custody-dir', '--commitment', '--attempt-dir', '--phase-ledger'}
    for index, arg in enumerate(child_args):
        flag, separator, inline = arg.partition('=')
        if (flag.startswith('--') and flag not in path_flags
                and any(name.startswith(flag) for name in path_flags)):
            raise ValueError('abbreviated child path options are unsupported')
        if flag in path_flags:
            if not separator and index + 1 >= len(child_args):
                raise ValueError('absolute child artifact paths are required')
            raw_path = inline if separator else child_args[index + 1]
            if not Path(raw_path).is_absolute():
                raise ValueError('absolute child artifact paths are required')
            child_path = Path(raw_path).resolve(strict=False)
            if (resolved == child_path or resolved in child_path.parents or child_path in resolved.parents):
                raise ValueError('private output and child artifact paths overlap')
    cache_free_before = None
    if child_args[0] == 'run':
        cache_free_before = shutil.disk_usage(cache).free
        if cache_free_before < _MIN_STAGE_FREE or any(cache.glob('looking-glass-stage-*')):
            raise ValueError('cache staging space or prior stages require reconciliation')
    sources = _verified_sources(repo, source_commit)
    if _read_bounded(Path(__file__).resolve(), _MIB) != sources['scripts/number_cpu_outer.py']:
        raise ValueError('outer source differs from committed bytes')
    python_sha = _file_sha256(python)
    environment = _environment(workspace, home, cache)
    launcher = repo / 'scripts' / 'pretrained_number_cpu_launcher.py'
    gate_source = sources['scripts/number_cpu_gate.py']
    launcher_sha = hashlib.sha256(sources['scripts/pretrained_number_cpu_launcher.py']).hexdigest()
    gate_sha = hashlib.sha256(gate_source).hexdigest()
    outer_sha = hashlib.sha256(sources['scripts/number_cpu_outer.py']).hexdigest()
    output.mkdir(mode=0o700)
    prefix = output / 'pycache'
    prefix.mkdir(mode=0o700)
    if any(prefix.iterdir()):
        raise ValueError('fresh bytecode prefix is not empty')
    token = secrets.token_hex(32)
    code = ("import base64;exec(compile(base64.b64decode('" + base64.b64encode(gate_source).decode('ascii') +
            "'),'<verified-number-gate>','exec',dont_inherit=True))")
    command = [str(python), '-B', '-E', '-s', '-S', '-P', '-X', 'pycache_prefix=' + str(prefix), '-c', code,
               str(output / 'grant'), token, str(launcher), launcher_sha,
               '--source-commit', source_commit, '--site-packages', str(site), *child_args]
    start = {'schema': 'mlg-number-cpu-outer-start-v1', 'phase': child_args[0], 'source_commit': source_commit,
             'launcher_sha256': launcher_sha, 'gate_sha256': gate_sha, 'outer_sha256': outer_sha,
             'python_sha256': python_sha,
             'command_sha256': hashlib.sha256(_canonical(command)).hexdigest(),
             'environment_sha256': hashlib.sha256(_canonical(environment)).hexdigest(),
             'prefix_initially_empty': True, 'pipe_cap_bytes': _PIPE_CAP}
    _write_new(output / 'started.json', _canonical(start), _META_CAP)
    stdout_state, stderr_state = _new_capture(), _new_capture()
    child, job, readers, exit_code, active = None, None, [], None, None
    child_reaped, timed_out, setup_error, pending_cancel = False, False, False, None
    primary_error = None
    spawn_started, spawn_returned = False, False
    reader_start_in_progress = False
    context = [None, None, None, None]  # Job, direct child, stdout and stderr readers.
    # Allocate all bookkeeping before creating any native resources. Assigning
    # an existing list slot does not grow a set in a failure/cleanup handler.
    reason_flags = [False] * len(_REASON_CODES)
    operation = 'job_init'
    try:
        # Root every resource before the operation that can start it. An
        # interrupted return must not drop the only recoverable handle.
        _OWNER_SLOT[0] = context
        job = WindowsJob.__new__(WindowsJob)
        context[0] = job
        WindowsJob.__init__(job)
        # Root the Popen instance before initialization; if native creation is
        # interrupted before it returns, an ungranted gate cannot run a study.
        operation = 'spawn'
        child = subprocess.Popen.__new__(subprocess.Popen)
        context[1] = child
        spawn_started = True
        subprocess.Popen.__init__(child, command, cwd=str(repo), env=environment, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        spawn_returned = True
        operation = 'reader_start'
        for index, (stream, state) in enumerate(((child.stdout, stdout_state), (child.stderr, stderr_state)), 2):
            reader_start_in_progress = True
            readers.append(_start_capture_thread(stream, state, context, index))
            reader_start_in_progress = False
        operation = 'job_assign'
        job.assign(child)
        operation = 'grant'
        _write_new(output / 'grant', token.encode('ascii'), 64)
        operation = 'child_wait'
        try:
            exit_code = child.wait(timeout=timeout)
            child_reaped = True
        except subprocess.TimeoutExpired:
            timed_out = True
            reason_flags[_REASON_INDEX['timeout']] = True
    except BaseException as error:
        setup_error = True
        primary_error = error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            pending_cancel = error
        reason_flags[_REASON_INDEX[operation]] = True
    finally:
        if not child_reaped:
            if job is not None:
                try:
                    job.terminate()
                except BaseException as cleanup:
                    setup_error = True
                    reason_flags[_REASON_INDEX['job_termination']] = True
                    if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                        pending_cancel = cleanup
            if child is not None and getattr(child, '_child_created', False):
                try:
                    if child.poll() is None:
                        child.kill()
                    exit_code = child.wait(timeout=5)
                    child_reaped = True
                except BaseException as cleanup:
                    setup_error = True
                    reason_flags[_REASON_INDEX['child_reap']] = True
                    if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                        pending_cancel = cleanup
        if job is not None:
            try:
                deadline = time.monotonic() + 5
                while True:
                    active = job.active()
                    if active == 0:
                        break
                    if time.monotonic() >= deadline:
                        setup_error = True
                        reason_flags[_REASON_INDEX['job_not_empty']] = True
                        job.terminate()
                        break
                    time.sleep(0.05)
            except BaseException as cleanup:
                setup_error = True
                reason_flags[_REASON_INDEX['job_query']] = True
                if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                    pending_cancel = cleanup
            closed = False
            try:
                job.close()
                closed = True
            except BaseException as cleanup:
                setup_error = True
                reason_flags[_REASON_INDEX['job_close']] = True
                if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                    pending_cancel = cleanup
            job_reconciled = (active == 0 and closed
                              and (not spawn_started or (spawn_returned and child_reaped)))
        else:
            job_reconciled = False
        readers_finished = not reader_start_in_progress
        for thread in readers:
            try:
                thread.join(timeout=5)
                if thread.is_alive():
                    setup_error = True
                    reason_flags[_REASON_INDEX['reader_not_stopped']] = True
                    readers_finished = False
            except BaseException as cleanup:
                setup_error = True
                reason_flags[_REASON_INDEX['reader_join']] = True
                readers_finished = False
                if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                    pending_cancel = cleanup
                    if primary_error is not None:
                        pending_cancel.__cause__ = primary_error
                elif pending_cancel is not None and pending_cancel.__cause__ is None:
                    pending_cancel.__cause__ = cleanup
        if job_reconciled and readers_finished:
            _OWNER_SLOT[0] = None
    out, out_count, out_truncated, out_error = _snapshot(stdout_state)
    err, err_count, err_truncated, err_error = _snapshot(stderr_state)
    prefix_empty = False
    try:
        prefix_empty = not any(prefix.iterdir())
        source_unchanged = (_verified_sources(repo, source_commit) == sources
                            and _read_bounded(Path(__file__).resolve(), _MIB) == sources['scripts/number_cpu_outer.py']
                            and _file_sha256(python) == python_sha)
    except BaseException as error:
        source_unchanged = False
        setup_error = True
        reason_flags[_REASON_INDEX['source_or_prefix_check']] = True
        if isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
            pending_cancel = error
    cache_stage_clear = None
    if child_args[0] == 'run':
        try:
            cache_stage_clear = not any(cache.glob('looking-glass-stage-*'))
            if not cache_stage_clear:
                setup_error = True
                reason_flags[_REASON_INDEX['cache_stage']] = True
        except BaseException as error:
            setup_error = True
            reason_flags[_REASON_INDEX['cache_stage_check']] = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                pending_cancel = error
    completion = {}
    if child_reaped and exit_code == 0 and active == 0 and prefix_empty and source_unchanged and not (
            timed_out or setup_error or out_truncated or err_truncated or out_error or err_error):
        try:
            completion = _completion(child_args[0], child_args, out)
        except BaseException as error:
            setup_error = True
            reason_flags[_REASON_INDEX['completion']] = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                pending_cancel = error
                if primary_error is not None:
                    pending_cancel.__cause__ = primary_error
    else:
        setup_error = True
    if not child_reaped or exit_code != 0:
        reason_flags[_REASON_INDEX['child_exit']] = True
    if active != 0:
        reason_flags[_REASON_INDEX['job_state']] = True
    if not prefix_empty:
        reason_flags[_REASON_INDEX['prefix']] = True
    if not source_unchanged:
        reason_flags[_REASON_INDEX['source']] = True
    if out_truncated or err_truncated:
        reason_flags[_REASON_INDEX['output_overflow']] = True
    if out_error or err_error:
        reason_flags[_REASON_INDEX['capture']] = True
    if pending_cancel is not None:
        reason_flags[_REASON_INDEX['cancelled']] = True
    saved = {}
    for label, data in (('stdout', out), ('stderr', err)):
        path = output / (label + '.bin')
        try:
            _write_new(path, data, _PIPE_CAP)
        except BaseException as error:
            setup_error = True
            reason_flags[_REASON_INDEX['log_write']] = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                pending_cancel = error
        # A failed write may leave any prefix, including an empty file. Report
        # only bytes actually read back, never the intended in-memory buffer.
        saved[label] = (None, None)
        try:
            with path.open('rb') as handle:
                persisted = handle.read(_PIPE_CAP + 1)
            if len(persisted) > _PIPE_CAP:
                raise ValueError('private output exceeds its fixed bound')
            saved[label] = (len(persisted), hashlib.sha256(persisted).hexdigest())
            if persisted != data:
                setup_error = True
                reason_flags[_REASON_INDEX['log_mismatch']] = True
        except BaseException as error:
            setup_error = True
            reason_flags[_REASON_INDEX['log_readback']] = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
                pending_cancel = error
    if pending_cancel is not None:
        reason_flags[_REASON_INDEX['cancelled']] = True
    result = {'schema': 'mlg-number-cpu-outer-terminal-v1', 'status': 'incomplete' if setup_error else 'succeeded',
        'phase': child_args[0], 'source_commit': source_commit, 'pid': getattr(child, 'pid', None),
        'exit_code': exit_code, 'child_reaped': child_reaped, 'timed_out': timed_out,
        'spawn_returned': spawn_returned,
        'active_processes': active, 'prefix_empty': prefix_empty, 'source_unchanged': source_unchanged,
        'cache_stage_clear': cache_stage_clear, 'cache_free_before_bytes': cache_free_before,
        'stdout_bytes_observed': out_count, 'stderr_bytes_observed': err_count,
        'stdout_bytes_saved': saved['stdout'][0], 'stderr_bytes_saved': saved['stderr'][0],
        'stdout_truncated': out_truncated, 'stderr_truncated': err_truncated,
        'stdout_sha256': saved['stdout'][1], 'stderr_sha256': saved['stderr'][1],
        'reason_codes': [name for index, name in enumerate(_REASON_CODES) if reason_flags[index]],
        **completion}
    try:
        _write_new(output / 'terminal.json', _canonical(result), _META_CAP)
    except BaseException as error:
        result['status'] = 'incomplete'
        if isinstance(error, (KeyboardInterrupt, SystemExit)) and pending_cancel is None:
            pending_cancel = error
    if pending_cancel is not None:
        raise pending_cancel
    return result


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'CPU study owner incomplete; invalid command line.\n')


def main(argv=None):
    parser = _PrivateArgumentParser(description=__doc__)
    for name in ('repo', 'source-commit', 'base-python', 'site-packages', 'workspace', 'hf-home', 'cache-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('child_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.child_args[1:] if args.child_args[:1] == ['--'] else args.child_args
    try:
        result = run_once(repo=args.repo, source_commit=args.source_commit, base_python=args.base_python,
            site_packages=args.site_packages, workspace=args.workspace, hf_home=args.hf_home,
            cache_dir=args.cache_dir, output_dir=args.output_dir, child_args=tuple(command), timeout=args.timeout)
        print(json.dumps({'status': result['status']}))
        return 0 if result['status'] == 'succeeded' else 1
    except KeyboardInterrupt:
        print('CPU study owner cancelled; retain private evidence.', file=sys.stderr)
        return 130
    except SystemExit as error:
        print('CPU study owner cancelled; retain private evidence.', file=sys.stderr)
        return error.code if type(error.code) is int and error.code != 0 else 1
    except BaseException:
        print('CPU study owner incomplete; retain private evidence.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
