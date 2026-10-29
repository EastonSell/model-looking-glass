"""Observe one source-bound CPU supervisor without reading private outputs.

This is a cooperative Windows process-custody component. A trusted caller
must source-verify this module and its dependencies before import, protect the
private paths and source leases, and independently observe this observer's own
successful return. It does not authorize an allocation or a model forward.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading

from .injection_preparation_audit import _read
from .injection_split_custody import _write_new


_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_QUARANTINED = []
_STATE_LOCK = threading.Lock()
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


class ObserverError(ValueError):
    """The supervisor's exit or protected evidence could not be established."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ObserverError('a lowercase SHA-256 digest is required')
    return value


def _release_state(state):
    """Idempotently release this exact, positively cleaned custody record."""
    with _STATE_LOCK:
        for index, item in enumerate(_QUARANTINED):
            if item is state:
                del _QUARANTINED[index]
                break


def _plain_existing_ancestors(path):
    if not path.is_absolute() or '..' in path.parts:
        return False
    for item in (path, *path.parents):
        if os.path.lexists(item):
            info = item.lstat()
            if (not stat.S_ISDIR(info.st_mode) or item.is_symlink()
                    or getattr(info, 'st_file_attributes', 0) & _REPARSE):
                return False
    return True


def _birth_time(process):
    if os.name != 'nt':
        raise ObserverError('the outer observer requires Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME))
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(int(process._handle), ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        raise ObserverError('cannot establish the supervisor process birth')
    birth = (created.dwHighDateTime << 32) | created.dwLowDateTime
    if not 0 < birth < 2 ** 64:
        raise ObserverError('invalid supervisor process birth')
    return birth


def _close_process_handle(process):
    if os.name != 'nt' or not callable(getattr(getattr(process, '_handle', None), 'Close', None)):
        raise ObserverError('the supervisor process handle cannot be closed')
    process._handle.Close()


def _configure_windows_job(job):
    from scripts import v2_source_guards as guards
    limits = guards._JobExtendedLimits()
    limits.basic.flags = 0x00002000 | 0x00001000  # kill on close; permit a separately gated nested Job
    if not job._kernel.SetInformationJobObject(job._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise ObserverError('cannot configure the outer supervisor Job')


def observe_supervisor(**kwargs):
    """Run the reviewed Windows Job observer under a trusted source bootstrap."""
    if os.name != 'nt' or not callable(kwargs.get('source_check')) or kwargs['source_check']() is not True:
        raise ObserverError('Windows and a verified source bootstrap are required')
    from scripts.v2_source_guards import WindowsJob
    return _observe(**kwargs, job_type=WindowsJob, configure_job=_configure_windows_job, birth_time=_birth_time)


def _observe(*, purpose, command, output, environment, config_path, config_sha256,
             operation_path, operation_sha256, command_sha256, supervisor_source_sha256,
             terminal_path, start_token, timeout, source_check, protect_process, terminal_check,
             job_type, configure_job, birth_time):
    """Observe a single gated supervisor; injected dependencies are for tests."""
    if (purpose not in ('preparation', 'execution') or type(command) is not list or not 1 <= len(command) <= 64
            or any(type(item) is not str or not item or len(item) > 4096 for item in command)
            or type(environment) is not dict or environment.get('CUDA_VISIBLE_DEVICES') != '-1'
            or type(start_token) is not bytes or not 1 <= len(start_token) <= 512
            or type(timeout) is not int or not 1 <= timeout <= 7200
            or any(not callable(value) for value in (source_check, protect_process, terminal_check,
                                                      job_type, configure_job, birth_time))):
        raise ObserverError('the supervisor invocation is invalid')
    for value in (config_sha256, operation_sha256, command_sha256, supervisor_source_sha256):
        _hex(value)
    output, terminal = Path(output), Path(terminal_path)
    if (not output.is_absolute() or not terminal.is_absolute() or not output.parent.is_dir()
            or not _plain_existing_ancestors(output.parent) or not _plain_existing_ancestors(terminal.parent)
            or os.path.lexists(output) or os.path.lexists(terminal)
            or terminal == Path(config_path) or terminal == Path(operation_path)):
        raise ObserverError('fresh private output and distinct inputs are required')
    try:
        terminal.relative_to(output)
    except ValueError:
        pass
    else:
        raise ObserverError('the supervisor terminal must be outside the outer output')
    config, operation = _read(config_path, 65536), _read(operation_path, 2 * 1024 * 1024)
    if (_sha(config) != config_sha256 or _sha(operation) != operation_sha256
            or _sha(_canonical(command)) != command_sha256 or source_check() is not True):
        raise ObserverError('source, configuration or command differs')

    state = {'job': None, 'process': None, 'spawn_attempted': False, 'gate_attempted': False,
             'reaped': False, 'drain_attempted': False, 'job_drained': False,
             'close_attempted': False, 'job_closed': False,
             'process_close_attempted': False, 'process_closed': False}
    with _STATE_LOCK:
        if _QUARANTINED:
            raise ObserverError('prior process custody is unverified')
        _QUARANTINED.append(state)
    try:
        output.mkdir()
        start_record = {'schema': 'mlg-injection-outer-process-start-v1', 'purpose': purpose,
            'config_sha256': config_sha256, 'operation_sha256': operation_sha256,
            'command_sha256': command_sha256, 'supervisor_source_sha256': supervisor_source_sha256,
            'start_token_sha256': _sha(start_token)}
        _write_new(output / 'started.json', _canonical(start_record))
        job = job_type.__new__(job_type)
        state['job'] = job
        job_type.__init__(job)
        configure_job(job)
        process = subprocess.Popen.__new__(subprocess.Popen)
        state['process'] = process
        state['spawn_attempted'] = True
        subprocess.Popen.__init__(process, command, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if type(process.pid) is not int or not 0 < process.pid <= 0xffffffff:
            raise ObserverError('invalid supervisor process identity')
        job.assign(process)
        if protect_process(process) is not True:
            raise ObserverError('source protections were not inherited')
        birth = birth_time(process)
        if type(birth) is not int or not 0 < birth < 2 ** 64:
            raise ObserverError('invalid supervisor process birth')
        if (_read(config_path, 65536) != config or _read(operation_path, 2 * 1024 * 1024) != operation
                or source_check() is not True or not _plain_existing_ancestors(terminal.parent)
                or os.path.lexists(terminal)):
            raise ObserverError('bound inputs changed before the start gate')
        # Set uncertainty before writing: a failed write may still release the child.
        state['gate_attempted'] = True
        if process.stdin.write(start_token) != len(start_token):
            raise ObserverError('supervisor start gate did not complete')
        process.stdin.close()
        code = process.wait(timeout=timeout)
        state['reaped'] = True
        active = job._accounting().active_processes
        if type(active) is not int or active < 0:
            raise ObserverError('outer Job membership is uncertain')
        state['drain_attempted'] = True
        job.terminate_and_wait()
        state['job_drained'] = True
        state['close_attempted'] = True
        job.close()
        state['job_closed'] = True
        state['process_close_attempted'] = True
        _close_process_handle(process)
        state['process_closed'] = True
        if type(code) is not int or code != 0 or active != 0:
            raise ObserverError('supervisor exit or Job membership was not clean')
        if (_read(config_path, 65536) != config or _read(operation_path, 2 * 1024 * 1024) != operation
                or source_check() is not True):
            raise ObserverError('bound inputs changed during supervision')
        if not _plain_existing_ancestors(terminal.parent):
            raise ObserverError('supervisor terminal parent is unsafe')
        terminal_raw = _read(terminal, 2 * 1024 * 1024)
        if terminal_check(terminal_raw, purpose, process.pid, birth) is not True:
            raise ObserverError('supervisor evidence did not verify')
        if (_read(terminal, 2 * 1024 * 1024) != terminal_raw or _read(config_path, 65536) != config
                or _read(operation_path, 2 * 1024 * 1024) != operation or source_check() is not True):
            raise ObserverError('supervisor evidence changed during verification')
        result = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': purpose,
            'status': 'completed', 'exit_code': 0, 'parent_reaped': True,
            'parent_handle_closed': True,
            'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
            'parent_pid': process.pid, 'parent_birth_100ns': birth,
            'config_sha256': config_sha256, 'operation_sha256': operation_sha256,
            'command_sha256': command_sha256, 'supervisor_source_sha256': supervisor_source_sha256,
            'supervisor_terminal_sha256': _sha(terminal_raw)}
        _write_new(output / 'observation.json', _canonical(result))
        if (_read(output / 'observation.json', 8192) != _canonical(result)
                or _read(terminal, 2 * 1024 * 1024) != terminal_raw
                or _read(config_path, 65536) != config or _read(operation_path, 2 * 1024 * 1024) != operation
                or source_check() is not True):
            raise ObserverError('source or observation changed before return')
        _release_state(state)
        return result
    except BaseException as primary:
        # After release, an un-reaped supervisor may own the only handles to a
        # nested Job. Keep its handles and source protections; never retry it.
        if state['gate_attempted'] and not state['reaped']:
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise
            raise ObserverError('supervisor observation is incomplete; process custody is retained') from None
        # A drain or close that raised may have taken effect already. Never
        # repeat it or close an unverified Job handle in the same process.
        if ((state['drain_attempted'] and not state['job_drained'])
                or (state['close_attempted'] and not state['job_closed'])
                or (state['process_close_attempted'] and not state['process_closed'])):
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise
            raise ObserverError('Job cleanup is uncertain; process custody is retained') from None
        errors = []
        process, job = state['process'], state['job']
        if not state['gate_attempted'] and process is not None and state['spawn_attempted']:
            try:
                if getattr(process, 'stdin', None) is not None and not process.stdin.closed:
                    process.stdin.close()
                if type(getattr(process, 'pid', None)) is not int:
                    raise ObserverError('spawn return is uncertain')
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                state['reaped'] = True
            except BaseException as error:
                errors.append(error)
        if job is not None and not state['job_drained']:
            try:
                state['drain_attempted'] = True
                job.terminate_and_wait()
                state['job_drained'] = True
            except BaseException as error:
                errors.append(error)
        if job is not None and state['job_drained'] and not state['job_closed']:
            try:
                state['close_attempted'] = True
                job.close()
                state['job_closed'] = True
            except BaseException as error:
                errors.append(error)
        if process is not None and state['reaped'] and not state['process_closed']:
            try:
                state['process_close_attempted'] = True
                _close_process_handle(process)
                state['process_closed'] = True
            except BaseException as error:
                errors.append(error)
        if not errors and (not state['spawn_attempted'] or state['reaped']):
            _release_state(state)
        if isinstance(primary, (KeyboardInterrupt, SystemExit)):
            if errors:
                raise primary from errors[0]
            raise
        fatal = next((error for error in errors if isinstance(error, (KeyboardInterrupt, SystemExit))), None)
        if fatal is not None:
            raise fatal from primary
        raise ObserverError('supervisor observation is incomplete') from None
