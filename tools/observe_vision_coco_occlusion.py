"""Observe one bounded CPU phase and retain a cooperative direct-child receipt.

A forced stop or unknown launch state cannot establish descendant cleanup and
does not authorize a retry. The phase claim is create-only and remains on error.
"""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('observer requires isolated no-site no-bytecode startup')

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

MANIFEST_SHA = 'aa61b294710f5a6cd9682dc783085c7079f1addfbad8257fdbd893d143ce4cef'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_environment(temporary):
    temporary = Path(temporary)
    return {'TEMP': str(temporary), 'TMP': str(temporary), 'TMPDIR': str(temporary),
            'HF_HOME': str(temporary / 'huggingface'),
            'HF_HUB_CACHE': str(temporary / 'huggingface' / 'hub'),
            'HUGGINGFACE_HUB_CACHE': str(temporary / 'huggingface' / 'hub'),
            'HF_ASSETS_CACHE': str(temporary / 'huggingface' / 'assets'),
            'HF_MODULES_CACHE': str(temporary / 'huggingface' / 'modules'),
            'HF_DATASETS_CACHE': str(temporary / 'huggingface' / 'datasets'),
            'XDG_CACHE_HOME': str(temporary / 'xdg'), 'TORCH_HOME': str(temporary / 'torch'),
            'TORCHINDUCTOR_CACHE_DIR': str(temporary / 'torchinductor'),
            'TRITON_CACHE_DIR': str(temporary / 'triton'), 'TORCH_EXTENSIONS_DIR': str(temporary / 'extensions'),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}


def child_environment(inherited, temporary):
    env = dict(inherited)
    for name in ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE'):
        env.pop(name, None)
    env.update(cache_environment(temporary), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MKL_CBWR='COMPATIBLE')
    return env


def claim_phase(custody, mode, revision):
    if mode == 'preflight':
        return None
    path = Path(custody) / (mode + '.claim.json')
    raw = (json.dumps({'mode': mode, 'revision': revision}, sort_keys=True) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def supervise(command, *, cwd, env, stdout, stderr, timeout):
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
    return code, timed_out, forced, reaped, error, fatal


def terminal_complete(path, mode, revision, manifest_sha):
    try:
        terminal = json.loads(path.read_bytes())
        started = json.loads((path.parent / 'started.json').read_bytes())
        if (terminal.get('schema') != 'vision-occlusion-v1-terminal' or
                started.get('schema') != 'vision-occlusion-v1-start' or
                terminal.get('status') != 'complete' or terminal.get('mode') != mode or
                started.get('mode') != mode or terminal.get('source_revision') != revision or
                started.get('source', {}).get('revision') != revision or
                terminal.get('manifest_sha256') != manifest_sha or started.get('manifest_sha256') != manifest_sha or
                type(terminal.get('submissions')) is not int or type(terminal.get('saved_forwards')) is not int or
                terminal['submissions'] != terminal['saved_forwards']):
            return False
        count = terminal['submissions']
        if mode == 'preflight':
            return count == 0
        if mode == 'screen':
            return count == 64
        cases = terminal.get('cases')
        if (not isinstance(cases, list) or type(terminal.get('selected')) is not int or
                terminal['selected'] != len(cases) or len(cases) > 8 or
                (mode == 'discovery' and len(cases) != 4)):
            return False
        sizes = {'matchable': 18, 'limited-dose': 18, 'dose-ineligible': 14, 'dose-unmatchable': 14}
        return count == sum(sizes[row['dose']['status']] for row in cases)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'discovery', 'screen', 'evaluation'), required=True)
    parser.add_argument('--revision', required=True)
    for name in ('checkout', 'assets', 'output', 'outer', 'custody', 'site-packages', 'pillow'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--screen-data', type=Path)
    parser.add_argument('--discovery-archive', type=Path)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    paths = (args.checkout, args.assets, args.output, args.outer, args.custody, args.site_packages, args.pillow)
    if (any(not path.is_absolute() for path in paths) or not re.fullmatch('[0-9a-f]{40}', args.revision) or
            not all(path.is_dir() for path in (args.checkout, args.assets, args.site_packages, args.pillow)) or
            args.output.exists() or args.outer.exists() or not args.output.parent.is_dir() or
            args.timeout <= 0 or (args.screen_data is not None and not args.screen_data.is_absolute()) or
            (args.discovery_archive is not None and not args.discovery_archive.is_absolute())):
        raise ValueError('observer arguments differ')
    def overlaps(a, b):
        a, b = a.resolve(), b.resolve()
        return a.is_relative_to(b) or b.is_relative_to(a)
    protected = (args.checkout, args.assets, args.site_packages, args.pillow)
    if (any(overlaps(a, b) for a in (args.output, args.outer, args.custody) for b in protected) or
            any(overlaps(a, b) for a, b in ((args.output, args.outer), (args.output, args.custody),
                                            (args.outer, args.custody))) or
            (args.screen_data is not None and any(overlaps(args.screen_data, b)
                                                  for b in (args.output, args.outer, args.custody)))):
        raise ValueError('observer roots overlap')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.checkout).decode().strip()
    if head != args.revision:
        raise ValueError('observer checkout revision differs')
    python = Path(sys._base_executable).resolve(strict=True)
    interpreter_sha = digest(python)
    args.outer.mkdir(parents=True, exist_ok=False)
    args.custody.mkdir(parents=True, exist_ok=True)
    prefix, temporary = args.outer / 'empty-bytecode-cache', args.outer / 'temporary'
    prefix.mkdir()
    temporary.mkdir()
    bootstrap = ('import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2],sys.argv[3]]); '
                 'sys.argv=sys.argv[4:]; runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [str(python), '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}', '-c', bootstrap,
               str(args.site_packages), str(args.pillow), str(args.checkout),
               str(args.checkout / 'tools/run_vision_coco_occlusion.py'), '--mode', args.mode,
               '--revision', args.revision, '--assets', str(args.assets), '--output', str(args.output)]
    if args.screen_data is not None:
        command += ['--screen-data', str(args.screen_data)]
    if args.discovery_archive is not None:
        command += ['--discovery-archive', str(args.discovery_archive)]
    env = child_environment(os.environ, temporary)
    started = time.monotonic()
    with (args.outer / 'stdout.txt').open('xb') as stdout, (args.outer / 'stderr.txt').open('xb') as stderr:
        claim_phase(args.custody, args.mode, args.revision)
        code, timeout, forced, reaped, error, fatal = supervise(
            command, cwd=args.checkout, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout)
    terminal_path = args.output / 'data/terminal.json'
    complete = (type(code) is int and code == 0 and timeout is False and forced is False and
                reaped is True and error is None and fatal is None and digest(python) == interpreter_sha and
                terminal_complete(terminal_path, args.mode, args.revision, MANIFEST_SHA))
    receipt = {'schema': 'vision-occlusion-v1-direct-observer', 'mode': args.mode,
               'source_revision': args.revision, 'exit_code': code, 'timed_out': timeout,
               'reaped': reaped, 'forced_stop': forced, 'base_interpreter_sha256': interpreter_sha,
               'error_type': error, 'duration_seconds': time.monotonic() - started,
               'stdout_sha256': digest(args.outer / 'stdout.txt'), 'stderr_sha256': digest(args.outer / 'stderr.txt'),
               'terminal_sha256': digest(terminal_path) if terminal_path.is_file() else None,
               'status': 'complete' if complete else 'incomplete',
               'scope': 'cooperative direct-child observation; not execution or process-tree attestation'}
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
