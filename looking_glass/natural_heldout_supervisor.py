"""Private, cooperative Windows supervisor for a single natural held-out owner.

The custodian owns the one-shot study claim, source/asset leases, the child
entry point, and the structural audit. This module neither creates cases nor
performs scientific validation. The supervisor record and logs may contain
local paths and must not be published without a reviewed projection.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import threading

from looking_glass.natural_agreement_controller import _retained_tensor_files


_START = b'MLG-natural-discovery-start-v1\n'
_LOG_LIMIT = 1024 * 1024
_AUDIT_LIMIT = 1024 * 1024
# Custody is rooted before any job or child can exist. Uncertain or interrupted
# handle closes are never blindly retried, and block further owners here.
_QUARANTINED = []
_CUSTODY_LOCK = threading.Lock()


def supervise_heldout(command, output, environment, *, audit_result, timeout=7200):
    """Run one gated child, then call the structural auditor after clean exit.

    ``audit_result(attempt_dir, exit_code, reaped)`` must return a bounded JSON
    object with status ``evidence_complete`` or ``incomplete``. The callback
    is called only after a zero exit, confirmed job drain/close and child reap,
    and complete log drainage. Its result is structural, not a scientific
    validation. A fresh output directory is exclusive; a custodian must also
    enforce the study-wide one-shot claim before calling this function.
    """
    if os.name != 'nt' or type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError('the Windows CPU owner requires a timeout of at most two hours')
    if (type(command) is not list or not command or
            any(type(value) is not str or not value for value in command) or
            type(environment) is not dict or not callable(audit_result)):
        raise ValueError('invalid owner command, environment or audit callback')
    from scripts.v2_source_guards import WindowsJob

    output = Path(output)
    if not output.is_absolute() or not output.parent.is_dir():
        raise ValueError('output needs an absolute path and an existing parent')
    custody = {'job': None, 'child': None, 'pumps': [], 'streams': [], 'output': output}
    with _CUSTODY_LOCK:
        if _QUARANTINED:
            raise RuntimeError('process custody is active or unverified; stop before any further owner')
        _QUARANTINED.append(custody)
    try:
        output.mkdir()
    except BaseException:
        # No job, child, or stream can exist before this operation.
        with _CUSTODY_LOCK:
            _QUARANTINED.remove(custody)
        raise

    started = datetime.now(timezone.utc).isoformat()
    code, reaped, failure, child, job = None, False, None, None, None
    job_drained, job_closed, cancellation_requested = False, False, False
    logs_complete = True
    overflow = threading.Event()
    pump_errors, pump_close_errors, pumps = [], [], custody['pumps']
    stdout = stderr = None
    try:
        stdout = (output / 'stdout.txt').open('xb')
        custody['streams'].append(stdout)
        stderr = (output / 'stderr.txt').open('xb')
        custody['streams'].append(stderr)
        job = WindowsJob()
        custody['job'] = job
        child = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=subprocess.DETACHED_PROCESS)
        custody['child'] = child
        job.assign(child)

        def pump(source, destination):
            remaining = _LOG_LIMIT
            try:
                while True:
                    data = source.read1(65536)
                    if not data:
                        break
                    keep = data[:remaining]
                    if destination.write(keep) != len(keep):
                        raise OSError('short log write')
                    remaining -= len(keep)
                    if len(keep) != len(data):
                        overflow.set()
                        try:
                            child.terminate()
                        except OSError:
                            pass
            except BaseException as error:
                pump_errors.append(type(error).__name__)
                try:
                    child.terminate()
                except OSError:
                    pass
            finally:
                try:
                    source.close()
                except BaseException as error:
                    pump_errors.append(type(error).__name__)
                    pump_close_errors.append(type(error).__name__)

        for source, destination in ((child.stdout, stdout), (child.stderr, stderr)):
            thread = threading.Thread(target=pump, args=(source, destination), daemon=True)
            pumps.append(thread)
            thread.start()
        if child.stdin.write(_START) != len(_START):
            raise OSError('short startup signal write')
        child.stdin.close()
        code = child.wait(timeout=timeout)
        reaped = True
    except BaseException as error:
        failure = type(error).__name__
        cancellation_requested = isinstance(error, KeyboardInterrupt)
    finally:
        if job is not None:
            try:
                job.terminate_and_wait()
                job_drained = True
            except BaseException as error:
                failure = (failure or 'job cleanup') + '; drain: ' + type(error).__name__
                cancellation_requested |= isinstance(error, KeyboardInterrupt)
            try:
                job.close()
                job_closed = True
            except BaseException as error:
                failure = (failure or 'job cleanup') + '; close: ' + type(error).__name__
                cancellation_requested |= isinstance(error, KeyboardInterrupt)
        if child is not None and not reaped:
            try:
                if child.stdin is not None and not child.stdin.closed:
                    child.stdin.close()
                if child.poll() is None:
                    child.kill()
                code = child.wait(timeout=10)
                reaped = True
            except BaseException as error:
                failure = (failure or 'child cleanup') + '; reap: ' + type(error).__name__
                cancellation_requested |= isinstance(error, KeyboardInterrupt)
        for thread in pumps:
            try:
                thread.join(timeout=10)
                if thread.is_alive():
                    logs_complete = False
                    failure = (failure or 'log drain') + '; log drain unverified'
            except BaseException as error:
                logs_complete = False
                failure = (failure or 'log drain') + '; join: ' + type(error).__name__
                cancellation_requested |= isinstance(error, KeyboardInterrupt)
        if stdout is None or stderr is None:
            logs_complete = False
        if pump_close_errors:
            logs_complete = False
        # Never close or read a log still owned by an unconfirmed pump.
        if logs_complete:
            for stream in custody['streams']:
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException as error:
                    logs_complete = False
                    failure = (failure or 'log flush') + '; flush: ' + type(error).__name__
                    cancellation_requested |= isinstance(error, KeyboardInterrupt)
                try:
                    stream.close()
                except BaseException as error:
                    logs_complete = False
                    failure = (failure or 'log close') + '; close: ' + type(error).__name__
                    cancellation_requested |= isinstance(error, KeyboardInterrupt)

    cleanup_verified = bool(reaped and job_drained and job_closed and logs_complete)
    child_pid = child.pid if child is not None else None
    if child is not None and (type(child_pid) is not int or not 0 < child_pid <= 0xffffffff):
        failure = (failure or 'child identity') + '; invalid child PID'
        child_pid = None
    verdict = {'status': 'incomplete', 'exit_code': code, 'child_reaped': reaped,
               'child_pid': child_pid,
               'job_drained': job_drained, 'job_closed': job_closed, 'logs_complete': logs_complete,
               'cleanup_verified': cleanup_verified, 'retry_safe': False,
               'cancellation_requested': cancellation_requested, 'quarantined': not cleanup_verified,
               'log_overflow': overflow.is_set(), 'supervisor_error': failure,
               'scientific_validation': 'pending independent raw numerical and landing audit'}
    if pump_errors:
        verdict['log_errors'] = pump_errors
    if not cleanup_verified:
        verdict['reason'] = 'process or log cleanup is unverified; no evidence was audited'
        verdict['protection_state'] = 'unverified; stop and verify process cleanup before any further attempt'
    elif cancellation_requested and not overflow.is_set() and not pump_errors:
        verdict['status'] = 'cancelled'
    elif failure is None and not overflow.is_set() and not pump_errors and type(code) is int and code == 0:
        try:
            audited = audit_result(output / 'attempt', code, reaped)
            if type(audited) is not dict or audited.get('status') not in ('evidence_complete', 'incomplete'):
                raise ValueError('invalid structural audit status')
            encoded = json.dumps(audited, sort_keys=True, ensure_ascii=True, allow_nan=False).encode('utf-8')
            if len(encoded) > _AUDIT_LIMIT:
                raise ValueError('structural audit result exceeds its bound')
            verdict['audit'] = audited
            verdict['status'] = audited['status']
        except BaseException as error:
            verdict['audit_error_type'] = type(error).__name__
            verdict['reason'] = 'structural audit did not complete'
            if isinstance(error, KeyboardInterrupt):
                verdict['cancellation_requested'] = True
                verdict['status'] = 'cancelled'
    else:
        verdict['reason'] = 'owner or supervisor did not complete successfully'

    verdict['retained_tensor_files'] = _retained_tensor_files(output / 'attempt', cleanup_verified)
    if not verdict['retained_tensor_files']['accounting_complete']:
        verdict['status'] = 'incomplete'
        verdict['reason'] = 'retained tensor accounting is incomplete'
    elif verdict['status'] == 'evidence_complete':
        inventory = verdict['audit'].get('inventory')
        if (type(inventory) is not dict or type(inventory.get('retained_tensor_bytes')) is not int or
                inventory['retained_tensor_bytes'] != verdict['retained_tensor_files']['total_bytes']):
            verdict['status'] = 'incomplete'
            verdict['reason'] = 'retained tensor accounting differs from the structural audit'
    verdict.update({'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
                    'command': command,
                    'stdout_sha256': hashlib.sha256((output / 'stdout.txt').read_bytes()).hexdigest() if logs_complete else None,
                    'stderr_sha256': hashlib.sha256((output / 'stderr.txt').read_bytes()).hexdigest() if logs_complete else None})
    raw = (json.dumps(verdict, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode('utf-8')
    with (output / 'supervisor.json').open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short supervisor record write')
        stream.flush()
        os.fsync(stream.fileno())
    if cleanup_verified:
        with _CUSTODY_LOCK:
            _QUARANTINED.remove(custody)
    return verdict
