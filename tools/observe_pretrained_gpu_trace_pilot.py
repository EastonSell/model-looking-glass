"""Retain and reap the direct source-bound controller; this wrapper uses no CUDA."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys


def _record(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short observation write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(environment):
    read_fd, write_fd = os.pipe()
    try:
        os.set_inheritable(read_fd, True)
        os.set_inheritable(write_fd, False)
        if os.name == 'nt':
            import msvcrt
            endpoint = msvcrt.get_osfhandle(read_fd)
            startup = subprocess.STARTUPINFO()
            startup.lpAttributeList = {'handle_list': [endpoint]}
            options = {'startupinfo': startup, 'close_fds': True}
        elif os.name == 'posix':
            endpoint = read_fd
            options = {'pass_fds': (read_fd,), 'close_fds': True}
        else:
            raise RuntimeError('controller launch gate is unsupported')
        environment['LOOKING_GLASS_TRACE_PILOT_CONTROLLER_GATE'] = str(endpoint)
        return read_fd, write_fd, options
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise


def _grant(fd, child_pid, parent_pid):
    if (type(child_pid) is not int or child_pid <= 0 or type(parent_pid) is not int or parent_pid <= 0):
        raise ValueError('invalid direct controller identity')
    data = struct.pack('!BQQ', 1, child_pid, parent_pid)
    if os.write(fd, data) != len(data):
        raise RuntimeError('controller launch grant did not complete')


def _direct_python(path, current, base):
    values = [Path(item).resolve(strict=True) for item in (path, current, base)]
    if not all(value.is_file() for value in values) or len(set(values)) != 1:
        raise ValueError('the observer and controller require the direct base interpreter')


def _identity_matches(path, process, source_commit):
    with Path(path).open('rb') as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError('controller identity receipt exceeds its byte limit')
    value = json.loads(raw)
    return (type(value) is dict and set(value) == {'schema', 'pid', 'parent_pid', 'source_commit'}
            and value['schema'] == 'pretrained-trace-controller-identity-v1'
            and type(value['pid']) is int and value['pid'] == process.pid
            and type(value['parent_pid']) is int and value['parent_pid'] == os.getpid()
            and value['source_commit'] == source_commit)


def observe(command, *, checkout, observation, run, environment, source_commit, probe=False):
    observation = Path(observation)
    observation.mkdir(parents=False, exist_ok=False)
    temp = observation / 'temp'
    temp.mkdir(exist_ok=False)
    environment = dict(environment)
    environment.update({'TEMP': str(temp), 'TMP': str(temp)})
    identity_path = observation / 'controller-identity.json'
    environment['LOOKING_GLASS_TRACE_PILOT_CONTROLLER_RECEIPT'] = str(identity_path)
    _record(observation / 'attempt.json', {'schema': 'pretrained-trace-controller-attempt-v1',
        'state': 'before_spawn', 'source_commit': source_commit, 'probe': probe, 'utc': _now(),
        'scope': 'direct retained process handle; not execution attestation'})
    process, primary, code, start_recorded, first_forward = None, None, None, False, False
    read_fd, write_fd, gate_options = _gate(environment)
    def revoke():
        nonlocal read_fd, write_fd
        # Forget each descriptor before closing it, so an interrupted close
        # cannot accidentally close a later unrelated descriptor.
        pending_write, write_fd = write_fd, None
        pending_read, read_fd = read_fd, None
        try:
            if pending_write is not None:
                os.close(pending_write)
        finally:
            if pending_read is not None:
                os.close(pending_read)
    def drain():
        nonlocal primary, code, first_forward
        if process is None:
            return
        while True:
            try:
                if not first_forward and (Path(run) / 'forward-001.json').is_file():
                    first_forward = True
                    print(json.dumps({'event': 'first_forward_saved'}), flush=True)
            except BaseException as exc:
                if primary is None:
                    primary = exc
            try:
                code = process.wait(timeout=1)
                return
            except subprocess.TimeoutExpired:
                continue
            except (KeyboardInterrupt, SystemExit) as exc:
                if primary is None:
                    primary = exc
            except BaseException as exc:
                if primary is None:
                    primary = exc
                return
    with (observation / 'stdout.log').open('xb') as stdout, (observation / 'stderr.log').open('xb') as stderr:
        try:
            try:
                process = subprocess.Popen(command, cwd=checkout, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    **gate_options)
                _record(observation / 'start.json', {'schema': 'pretrained-trace-controller-start-v1',
                    'pid': process.pid, 'source_commit': source_commit, 'probe': probe, 'utc': _now()})
                start_recorded = True
                _grant(write_fd, process.pid, os.getpid())
                print(json.dumps({'event': 'controller_started', 'pid': process.pid}), flush=True)
            except BaseException as exc:
                primary = exc
            finally:
                revoke()
                drain()
        except BaseException as exc:
            if primary is None:
                primary = exc
            # Recover a single interruption at entry to the first drain while
            # the process handle remains rooted. Repeated faults are uncertain.
            revoke()
            drain()
        for stream in (stdout, stderr):
            try:
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException as exc:
                if primary is None:
                    primary = exc
    identity_verified = False
    if process is not None:
        try:
            identity_verified = _identity_matches(identity_path, process, source_commit)
        except (OSError, ValueError, TypeError):
            identity_verified = False
        if code == 0 and not identity_verified and primary is None:
            primary = RuntimeError('controller identity receipt was not verified')
        try:
            exit_sha = _record(observation / 'exit.json', {'schema': 'pretrained-trace-controller-exit-v1',
                'pid': process.pid, 'source_commit': source_commit, 'probe': probe,
                'exit_code': code if type(code) is int else None, 'reaped': type(code) is int,
                'start_recorded': start_recorded, 'first_forward_observed': first_forward,
                'controller_identity_verified': identity_verified,
                'wrapper_error_type': type(primary).__name__ if primary is not None else None,
                'stdout_sha256': _file_sha(observation / 'stdout.log'),
                'stderr_sha256': _file_sha(observation / 'stderr.log'), 'utc': _now()})
        except BaseException as record_error:
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise primary from record_error
            raise
        print(json.dumps({'event': 'controller_reaped' if type(code) is int else 'controller_wait_unverified',
                          'pid': process.pid, 'exit_code': code, 'exit_sha256': exit_sha}), flush=True)
    if primary is not None:
        raise primary
    if type(code) is not int:
        raise RuntimeError('controller wait was not verified')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'checkout', 'site', 'git', 'workspace', 'cache', 'output',
                 'observation', 'prefix', 'commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--probe-bootstrap', action='store_true')
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)) for name in ('python', 'checkout', 'site', 'git', 'workspace',
             'cache', 'output', 'observation', 'prefix')}
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()):
        raise RuntimeError('observer requires isolated direct-base startup and an absolute bytecode prefix')
    _direct_python(paths['python'], sys.executable, sys._base_executable)
    if (any(not path.is_absolute() for path in paths.values()) or len(args.commit) != 40
            or any(char not in '0123456789abcdef' for char in args.commit)
            or not paths['python'].is_file() or not paths['git'].is_file()
            or not paths['checkout'].is_dir() or not paths['site'].is_dir()
            or paths['output'].exists() or paths['prefix'].exists()):
        raise ValueError('fixed controller paths or source identity differ')
    paths['prefix'].mkdir(parents=False, exist_ok=False)
    command = [str(paths['python']), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(paths['prefix']),
        str(paths['checkout'] / 'scripts' / 'pretrained_gpu_trace_pilot_bootstrap.py'),
        '--bootstrap-site', str(paths['site']), '--source-commit', args.commit, '--git-executable', str(paths['git'])]
    if args.probe_bootstrap:
        command.append('--help')
    else:
        command.extend(['--workspace-root', str(paths['workspace']), '--cache', str(paths['cache']),
                        '--output', str(paths['output'])])
    environment = os.environ.copy()
    if args.probe_bootstrap:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    return observe(command, checkout=paths['checkout'], observation=paths['observation'], run=paths['output'],
                   environment=environment, source_commit=args.commit, probe=args.probe_bootstrap)


if __name__ == '__main__':
    raise SystemExit(main())
