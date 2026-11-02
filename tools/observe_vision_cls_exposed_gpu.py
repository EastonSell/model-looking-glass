"""Retain and reap the direct source-bound controller; this wrapper uses no CUDA."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
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


def observe(command, *, checkout, observation, run, environment, source_commit, probe=False):
    observation = Path(observation)
    observation.mkdir(parents=False, exist_ok=False)
    temp = observation / 'temp'
    temp.mkdir(exist_ok=False)
    environment = dict(environment)
    environment.update({'TEMP': str(temp), 'TMP': str(temp)})
    _record(observation / 'attempt.json', {'schema': 'vision-cls-controller-attempt-v1',
        'state': 'before_spawn', 'source_commit': source_commit, 'probe': probe, 'utc': _now(),
        'scope': 'direct retained process handle; not execution attestation'})
    process, primary, code, start_recorded, first_forward = None, None, None, False, False
    def drain():
        nonlocal primary, code, first_forward
        if process is None:
            return
        while True:
            try:
                if not first_forward and (Path(run) / 'data' / 'return-001.json').is_file():
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
                    stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                _record(observation / 'start.json', {'schema': 'vision-cls-controller-start-v1',
                    'pid': process.pid, 'source_commit': source_commit, 'probe': probe, 'utc': _now()})
                start_recorded = True
                print(json.dumps({'event': 'controller_started', 'pid': process.pid}), flush=True)
            except BaseException as exc:
                primary = exc
            finally:
                drain()
        except BaseException as exc:
            if primary is None:
                primary = exc
            # Recover a single interruption at entry to the first drain while
            # the process handle remains rooted. Repeated faults are uncertain.
            drain()
        for stream in (stdout, stderr):
            try:
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException as exc:
                if primary is None:
                    primary = exc
    if process is not None:
        try:
            exit_sha = _record(observation / 'exit.json', {'schema': 'vision-cls-controller-exit-v1',
                'pid': process.pid, 'source_commit': source_commit, 'probe': probe,
                'exit_code': code if type(code) is int else None, 'reaped': type(code) is int,
                'start_recorded': start_recorded, 'first_forward_observed': first_forward,
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
    for name in ('python', 'checkout', 'site', 'git', 'workspace', 'assets', 'reference', 'output',
                 'observation', 'prefix', 'commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--probe-bootstrap', action='store_true')
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)) for name in ('python', 'checkout', 'site', 'git', 'workspace',
             'assets', 'reference', 'output', 'observation', 'prefix')}
    if (any(not path.is_absolute() for path in paths.values()) or len(args.commit) != 40
            or any(char not in '0123456789abcdef' for char in args.commit)
            or not paths['python'].is_file() or not paths['git'].is_file()
            or not paths['checkout'].is_dir() or not paths['site'].is_dir()
            or paths['output'].exists() or paths['prefix'].exists()):
        raise ValueError('fixed controller paths or source identity differ')
    paths['prefix'].mkdir(parents=False, exist_ok=False)
    command = [str(paths['python']), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(paths['prefix']),
        str(paths['checkout'] / 'scripts' / 'vision_coco_cls_exposed_gpu_bootstrap.py'),
        '--bootstrap-site', str(paths['site']), '--source-commit', args.commit, '--git-executable', str(paths['git'])]
    if args.probe_bootstrap:
        command.append('--help')
    else:
        command.extend(['--workspace-root', str(paths['workspace']), '--assets', str(paths['assets']),
                        '--reference', str(paths['reference']), '--output', str(paths['output'])])
    environment = os.environ.copy()
    if args.probe_bootstrap:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    return observe(command, checkout=paths['checkout'], observation=paths['observation'], run=paths['output'],
                   environment=environment, source_commit=args.commit, probe=args.probe_bootstrap)


if __name__ == '__main__':
    raise SystemExit(main())
