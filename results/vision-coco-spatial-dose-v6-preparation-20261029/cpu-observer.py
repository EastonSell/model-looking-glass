"""Observe one isolated v6 CPU controller and retain a path-free receipt."""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('observer requires isolated no-site no-bytecode startup and a fresh bytecode prefix')

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if __name__ == '__main__':
    _prefix = Path(sys.pycache_prefix)
    if not _prefix.is_absolute() or not _prefix.is_dir() or any(_prefix.iterdir()):
        raise RuntimeError('observer requires an externally created empty bytecode prefix')

WORK = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supervise(command, *, cwd, env, stdout, stderr, timeout=3900):
    """Retain the direct controller handle through wait, timeout, or cancellation."""
    child = None
    code, timed_out, forced, reaped, error, fatal = None, False, False, False, None, None
    try:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        try:
            code = child.wait(timeout=timeout)
            reaped = True
        except subprocess.TimeoutExpired:
            timed_out = True
            raise
    except BaseException as exc:
        error = type(exc).__name__
        if not isinstance(exc, Exception):
            fatal = exc
        if child is not None and not reaped:
            forced = True
            try:
                if child.poll() is None:
                    child.kill()
                code = child.wait(timeout=5)
                reaped = True
            except BaseException as cleanup:
                if fatal is None and not isinstance(cleanup, Exception):
                    fatal = cleanup
                error = type(cleanup).__name__ if error is None else error
    return code, timed_out, forced, reaped, error, fatal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'screen', 'intervention'), required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outer', type=Path, required=True)
    parser.add_argument('--custody', type=Path, required=True)
    parser.add_argument('--screen-data', type=Path)
    args = parser.parse_args()
    for path in (args.checkout, args.output, args.outer, args.custody):
        if not path.is_absolute():
            raise ValueError('observer paths must be absolute')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.checkout).decode().strip() != args.revision:
        raise ValueError('observer checkout revision differs')
    args.outer.mkdir(parents=True, exist_ok=False)
    args.custody.mkdir(parents=True, exist_ok=True)
    prefix, temporary = args.outer / 'empty-bytecode-cache', args.outer / 'temporary'
    prefix.mkdir()
    temporary.mkdir()
    site, pillow = WORK / '.venv/Lib/site-packages', WORK / 'vision-coco-deps'
    # Launch the real base executable, not the Windows venv redirector, so the
    # retained Popen handle names the controller that executes this script.
    python = Path(sys._base_executable).resolve(strict=True)
    if not python.is_file():
        raise ValueError('base interpreter is not a file')
    interpreter_sha = sha(python)
    bootstrap = ('import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2],sys.argv[3]]); '
                 'sys.argv=sys.argv[4:]; runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [str(python), '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}', '-c', bootstrap,
               str(site), str(pillow), str(args.checkout), str(args.checkout / 'tools/run_vision_coco_spatial_dose.py'),
               '--mode', args.mode, '--revision', args.revision, '--assets', str(WORK / 'vision-coco-assets'),
               '--output', str(args.output), '--custody', str(args.custody), '--pillow', str(pillow),
               '--site-packages', str(site), '--timeout', '3600']
    if args.screen_data is not None:
        command += ['--screen-data', str(args.screen_data)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TEMP=str(temporary), TMP=str(temporary),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_CBWR='COMPATIBLE')
    start = time.monotonic()
    with (args.outer / 'stdout.txt').open('xb') as stdout, (args.outer / 'stderr.txt').open('xb') as stderr:
        code, timeout, forced, reaped, error, fatal = supervise(
            command, cwd=args.checkout, env=env, stdout=stdout, stderr=stderr)
    supervisor_path = args.output / 'supervisor.json'
    terminal_path = args.output / 'data/terminal.json'
    try:
        supervisor = json.loads(supervisor_path.read_bytes()) if supervisor_path.is_file() else {}
    except (OSError, ValueError) as exc:
        supervisor = {}
        error = error or type(exc).__name__
    complete = (type(code) is int and code == 0 and timeout is False and forced is False and
                reaped is True and error is None and fatal is None and
                supervisor.get('status') == 'complete' and type(supervisor.get('exit_code')) is int and
                supervisor['exit_code'] == 0 and
                supervisor.get('clean') is True and supervisor.get('timed_out') is False and
                terminal_path.is_file() and supervisor.get('terminal_sha256') == sha(terminal_path) and
                sha(python) == interpreter_sha)
    receipt = {'schema': 'vision-recognized-v4-external-observer', 'mode': args.mode,
               'source_revision': args.revision, 'controller_exit_code': code, 'timed_out': timeout,
               'controller_reaped': reaped, 'forced_stop': forced,
               'base_interpreter_sha256': interpreter_sha,
               'error_type': error, 'duration_seconds': time.monotonic() - start,
               'stdout_sha256': sha(args.outer / 'stdout.txt'), 'stderr_sha256': sha(args.outer / 'stderr.txt'),
               'supervisor_sha256': sha(supervisor_path) if supervisor_path.is_file() else None,
               'terminal_sha256': sha(terminal_path) if terminal_path.is_file() else None,
               'status': 'complete' if complete else 'incomplete',
               'scope': 'cooperative observation of one direct controller; not execution or process-tree attestation'}
    raw = (json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (args.outer / 'receipt.json').open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt))
    if fatal is not None:
        raise fatal
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
