"""Bounded handoff for an independently observed comparison operator.

The trusted operator must bind actual published sources and operation meaning,
protect its paths and process handles, and verify the receipt against public
commitments. These records do not independently prove execution or authorize
a first forward.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re


_CALLER_SHA = '3e61675a962a205aebf336b84699ee7a4fc6d500b01cb25ab7ebea0fda094ab4'
_ADAPTER_SHA = '4570ab75d1858dcdbd9b274651b2c12efad8973cc5188167daa6829419f4405c'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_INNER_KEYS = frozenset({'schema', 'purpose', 'status', 'exit_code', 'parent_reaped',
    'parent_handle_closed', 'outer_job_drained', 'outer_job_closed', 'active_job_members',
    'parent_pid', 'parent_birth_100ns', 'config_sha256', 'operation_sha256',
    'command_sha256', 'supervisor_source_sha256', 'supervisor_terminal_sha256'})


class ExitError(ValueError):
    """The comparison process or its retained evidence is incomplete."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ExitError('a lowercase SHA-256 digest is required')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _parse(raw, maximum):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ExitError('bounded evidence is required')
    try:
        value = json.loads(raw)
        if type(value) is not dict or _canonical(value) != raw:
            raise ExitError('evidence is not canonical')
        return value
    except (TypeError, UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ExitError('evidence is invalid') from error


def current_identity():
    """Read this Windows process's PID and native creation time."""
    if os.name != 'nt':
        raise ExitError('Windows process identity is required')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME))
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        raise ExitError('cannot establish operator process birth')
    pid = os.getpid()
    birth = (created.dwHighDateTime << 32) | created.dwLowDateTime
    if type(pid) is not int or not 0 < pid <= 0xffffffff or not 0 < birth < 2 ** 64:
        raise ExitError('operator process identity is invalid')
    return pid, birth


def _check_config(comparison, path, expected_sha256, receipt):
    raw = comparison._read_file(path, 65536)
    if _sha(raw) != expected_sha256:
        raise ExitError('comparison configuration differs')
    config = comparison.parse_config(raw)
    if Path(config['output']) != receipt:
        raise ExitError('configured receipt path differs')
    return raw


def verify_public_receipt(*, comparison, custody, config, receipt, reader=None):
    """Verify a clear or collision receipt using only published commitments.

    The default reader obtains the exact bytes from the pinned immutable URLs.
    An injected reader is for synthetic tests. This verifies saved bytes, not
    process execution.
    """
    if reader is None:
        reader = comparison._fetch
    if type(config) is not dict or not callable(reader):
        raise ExitError('a bound public receipt check is required')
    try:
        config = comparison.parse_config(_canonical(config))
        public = {}
        for track, expected, freeze, url in (
            ('natural', comparison._NATURAL_COMMITMENT, comparison._NATURAL_FREEZE,
             comparison._PUBLIC_URLS['natural_commitment']),
            ('injection', config['injection_commitment_sha256'], comparison._INJECTION_FREEZE,
             config['injection_commitment_url']),
        ):
            raw = comparison._read_file(config[track]['commitment'], 1024 * 1024)
            if _sha(raw) != expected or reader(url) != raw:
                raise ExitError('published commitment does not match')
            public[track] = (raw, comparison._public_commitment(track, raw, freeze))
        value = _parse(receipt, 8192)
        if value.get('status') not in ('clear', 'collision'):
            raise ExitError('comparison status is incomplete')
        clear = _canonical(dict(value, status='clear'))
        if custody.verify_clear_receipt(clear,
                natural_freeze_sha256=comparison._NATURAL_FREEZE,
                injection_freeze_sha256=comparison._INJECTION_FREEZE,
                natural_inventory_sha256=public['natural'][1]['exposure_inventory_sha256'],
                injection_inventory_sha256=public['injection'][1]['exposure_inventory_sha256'],
                natural_commitment_bytes=public['natural'][0],
                injection_commitment_bytes=public['injection'][0],
                comparator_source_sha256=comparison._SOURCE_SPECS[-1][1]) is not True:
            raise ExitError('public comparison receipt does not match')
        return True
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, RecursionError) as error:
        if isinstance(error, ExitError):
            raise
        raise ExitError('public comparison receipt was not verified') from None


def make_terminal(result, *, config_sha256, operation_sha256, comparator_command_sha256,
                  operator_source_sha256, pid, birth):
    """Make a bounded terminal for the operator that actually ran the adapter."""
    if (type(result) is not dict or set(result) != {'comparison_status', 'receipt_sha256',
                                                    'observer_observation_sha256'}
            or result['comparison_status'] not in ('clear', 'collision')
            or type(pid) is not int or not 0 < pid <= 0xffffffff
            or type(birth) is not int or not 0 < birth < 2 ** 64):
        raise ExitError('operator result or identity is invalid')
    for value in (config_sha256, operation_sha256, comparator_command_sha256,
                  operator_source_sha256, result['receipt_sha256'], result['observer_observation_sha256']):
        _digest(value)
    return _canonical({'schema': 'mlg-injection-comparison-operator-terminal-v1',
        'status': 'completed', 'comparison_status': result['comparison_status'],
        'receipt_sha256': result['receipt_sha256'],
        'inner_observation_sha256': result['observer_observation_sha256'],
        'config_sha256': config_sha256, 'operation_sha256': operation_sha256,
        'comparator_command_sha256': comparator_command_sha256,
        'operator_source_sha256': operator_source_sha256, 'adapter_source_sha256': _ADAPTER_SHA,
        'comparator_caller_sha256': _CALLER_SHA, 'operator_pid': pid, 'operator_birth_100ns': birth})


def verify_terminal(raw, *, inner_observation, receipt, receipt_check, config_sha256,
                    operation_sha256, comparator_command_sha256, operator_source_sha256, pid, birth):
    """Check a fresh terminal against the actual observed process identity."""
    if not callable(receipt_check):
        raise ExitError('an independent public receipt check is required')
    terminal = _parse(raw, 8192)
    inner = _parse(inner_observation, 8192)
    receipt_value = _parse(receipt, 8192)
    status = receipt_value.get('status')
    if status not in ('clear', 'collision'):
        raise ExitError('comparison status is incomplete')
    result = {'comparison_status': status, 'receipt_sha256': _sha(receipt),
              'observer_observation_sha256': _sha(inner_observation)}
    expected = make_terminal(result, config_sha256=config_sha256, operation_sha256=operation_sha256,
                             comparator_command_sha256=comparator_command_sha256,
                             operator_source_sha256=operator_source_sha256, pid=pid, birth=birth)
    if raw != expected:
        raise ExitError('operator terminal differs from the observed process')
    if (set(inner) != _INNER_KEYS or inner.get('schema') != 'mlg-injection-outer-process-observation-v1'
            or inner.get('purpose') != 'execution' or inner.get('status') != 'completed'
            or type(inner.get('exit_code')) is not int or inner['exit_code'] != 0
            or any(inner.get(key) is not True for key in
                   ('parent_reaped', 'parent_handle_closed', 'outer_job_drained', 'outer_job_closed'))
            or type(inner.get('active_job_members')) is not int or inner['active_job_members'] != 0
            or type(inner.get('parent_pid')) is not int or inner['parent_pid'] <= 0
            or type(inner.get('parent_birth_100ns')) is not int or inner['parent_birth_100ns'] <= 0
            or inner.get('config_sha256') != config_sha256
            or inner.get('operation_sha256') != operation_sha256
            or inner.get('command_sha256') != comparator_command_sha256
            or inner.get('supervisor_source_sha256') != _CALLER_SHA
            or inner.get('supervisor_terminal_sha256') != _sha(receipt)
            or receipt_check(receipt) is not True):
        raise ExitError('comparison evidence was not verified')
    return {'status': status, 'receipt_sha256': _sha(receipt)}


def run_operator(*, adapter, adapter_options, comparison, terminal_path, receipt_path,
                 source_check, receipt_check, comparator_command_sha256, operator_source_sha256):
    """Run the observed comparison, then write this process's bound terminal.

    A source-first caller must verify the adapter, comparison, observer and
    this operator's code before use, and construct the trusted callbacks and
    exact operation binding. The separate outer process must observe this
    function's containing process exit before using the terminal.
    """
    if (type(adapter_options) is not dict or not callable(getattr(adapter, 'observe_comparison', None))
            or not callable(source_check) or not callable(receipt_check)):
        raise ExitError('trusted operator inputs are required')
    try:
        inner = Path(adapter_options['output']) / 'observation.json'
        config_sha256 = _digest(adapter_options['config_sha256'])
        operation_sha256 = _digest(adapter_options['operation_sha256'])
    except (KeyError, TypeError) as error:
        raise ExitError('bound operator inputs are incomplete') from error
    _digest(comparator_command_sha256)
    _digest(operator_source_sha256)
    terminal, receipt = Path(terminal_path), Path(receipt_path)
    if (any(not path.is_absolute() or '..' in path.parts for path in (terminal, inner, receipt))
            or len({terminal, inner, receipt}) != 3
            or any(os.path.lexists(path) for path in (terminal, inner, receipt))):
        raise ExitError('fresh and distinct evidence paths are required')
    config_raw = _check_config(comparison, adapter_options['config_path'], config_sha256, receipt)
    if source_check() is not True:
        raise ExitError('operator source was not verified')
    result = adapter.observe_comparison(**adapter_options)
    if _check_config(comparison, adapter_options['config_path'], config_sha256, receipt) != config_raw:
        raise ExitError('comparison configuration changed')
    inner_raw = comparison._read_file(inner, 8192)
    receipt_raw = comparison._read_file(receipt, 8192)
    pid, birth = current_identity()
    raw = make_terminal(result, config_sha256=config_sha256, operation_sha256=operation_sha256,
                        comparator_command_sha256=comparator_command_sha256,
                        operator_source_sha256=operator_source_sha256, pid=pid, birth=birth)
    checked = verify_terminal(raw, inner_observation=inner_raw, receipt=receipt_raw, receipt_check=receipt_check,
        config_sha256=config_sha256, operation_sha256=operation_sha256,
        comparator_command_sha256=comparator_command_sha256,
        operator_source_sha256=operator_source_sha256, pid=pid, birth=birth)
    if source_check() is not True:
        raise ExitError('operator source changed during comparison')
    with terminal.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise ExitError('operator terminal write was incomplete')
        stream.flush()
        os.fsync(stream.fileno())
    if (comparison._read_file(terminal, 8192) != raw
            or comparison._read_file(inner, 8192) != inner_raw
            or comparison._read_file(receipt, 8192) != receipt_raw
            or _check_config(comparison, adapter_options['config_path'], config_sha256, receipt) != config_raw
            or source_check() is not True):
        raise ExitError('operator evidence changed before return')
    return checked


def observe_operator(*, comparison, command, output, environment, config_path, operation_path,
                     terminal_path, inner_path, receipt_path, start_token, timeout, source_check,
                     protect_process, receipt_check, observer, config_sha256, operation_sha256,
                     comparator_command_sha256, operator_source_sha256):
    """Observe the adapter operator and bind its fresh comparison evidence.

    The source-bound caller must verify the operator command and all loaded
    sources, hold protected paths and handles, and independently verify the
    public receipt. The observer must be the reviewed outer process observer.
    Its own caller's exit remains a separate observation.
    """
    if any(not callable(value) for value in (source_check, protect_process, receipt_check, observer)):
        raise ExitError('trusted process, source and receipt checks are required')
    for value in (config_sha256, operation_sha256, comparator_command_sha256, operator_source_sha256):
        _digest(value)
    terminal, inner, receipt = (Path(terminal_path), Path(inner_path), Path(receipt_path))
    if (any(not path.is_absolute() or '..' in path.parts for path in (terminal, inner, receipt))
            or len({terminal, inner, receipt}) != 3
            or any(os.path.lexists(path) for path in (terminal, inner, receipt))):
        raise ExitError('fresh and distinct evidence paths are required')
    config_raw = _check_config(comparison, config_path, config_sha256, receipt)
    if source_check() is not True:
        raise ExitError('operator source was not verified')
    command_sha256 = _sha(_canonical(command))
    verified = {}

    def check_terminal(raw, purpose, pid, birth):
        if purpose != 'execution':
            return False
        try:
            inner_raw = comparison._read_file(inner, 8192)
            receipt_raw = comparison._read_file(receipt, 8192)
            value = verify_terminal(raw, inner_observation=inner_raw, receipt=receipt_raw,
                receipt_check=receipt_check, config_sha256=config_sha256, operation_sha256=operation_sha256,
                comparator_command_sha256=comparator_command_sha256,
                operator_source_sha256=operator_source_sha256, pid=pid, birth=birth)
            verified.update(value, pid=pid, birth=birth, terminal_sha256=_sha(raw),
                            inner_sha256=_sha(inner_raw))
            return True
        except (ValueError, TypeError, OSError, UnicodeError, RecursionError):
            return False

    observation = observer(purpose='execution', command=command, output=Path(output), environment=environment,
        config_path=Path(config_path), config_sha256=config_sha256,
        operation_path=Path(operation_path), operation_sha256=operation_sha256,
        command_sha256=command_sha256, supervisor_source_sha256=operator_source_sha256,
        terminal_path=terminal, start_token=start_token, timeout=timeout,
        source_check=source_check, protect_process=protect_process, terminal_check=check_terminal)
    if (type(observation) is not dict or set(observation) != _INNER_KEYS
            or observation.get('schema') != 'mlg-injection-outer-process-observation-v1'
            or observation.get('purpose') != 'execution' or observation.get('status') != 'completed'
            or type(observation.get('exit_code')) is not int or observation['exit_code'] != 0
            or any(observation.get(key) is not True for key in
                   ('parent_reaped', 'parent_handle_closed', 'outer_job_drained', 'outer_job_closed'))
            or type(observation.get('active_job_members')) is not int or observation['active_job_members'] != 0
            or observation.get('parent_pid') != verified.get('pid')
            or observation.get('parent_birth_100ns') != verified.get('birth')
            or observation.get('supervisor_terminal_sha256') != verified.get('terminal_sha256')
            or observation.get('config_sha256') != config_sha256
            or observation.get('operation_sha256') != operation_sha256
            or observation.get('command_sha256') != command_sha256
            or observation.get('supervisor_source_sha256') != operator_source_sha256
            or verified.get('status') not in ('clear', 'collision')
            or _sha(comparison._read_file(inner, 8192)) != verified.get('inner_sha256')
            or _sha(comparison._read_file(receipt, 8192)) != verified.get('receipt_sha256')
            or _check_config(comparison, config_path, config_sha256, receipt) != config_raw
            or source_check() is not True):
        raise ExitError('comparison operator observation is incomplete')
    return {'status': verified['status'], 'receipt_sha256': verified['receipt_sha256'],
            'operator_observation_sha256': _sha(_canonical(observation))}
