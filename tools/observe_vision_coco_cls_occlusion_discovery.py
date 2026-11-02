"""Observe the bounded CLS/occlusion CPU discovery through a direct child.

A forced stop or unknown launch state is incomplete and does not establish
descendant cleanup or authorize a retry. Claims require a trusted custodian.
"""
from __future__ import annotations

import sys

if __name__ == '__main__' and (not sys.flags.isolated or not sys.flags.no_site or
                                not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError('observer requires isolated no-site no-bytecode startup')

if __name__ == '__main__':
    if sys.platform == 'win32':
        import nt as _prefix_os
        _prefix = sys.pycache_prefix
        _absolute = (_prefix.startswith('\\\\') or
                     len(_prefix) >= 3 and _prefix[1] == ':' and _prefix[2] in ('/', '\\'))
    else:
        import posix as _prefix_os
        _prefix = sys.pycache_prefix
        _absolute = _prefix.startswith('/')
    try:
        if not _absolute or _prefix_os.listdir(_prefix):
            raise RuntimeError('observer requires an absolute empty bytecode prefix')
    except OSError as error:
        raise RuntimeError('observer requires an absolute empty bytecode prefix') from error

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = 'a52040786557fde9e503b6b6d1246bee0cbd41106b3cd110ea6788ab13a53487'
IDS = [404350, 151497, 470798, 135116]


def terminal_complete(path, mode, revision):
    try:
        terminal = json.loads(path.read_bytes())
        started = json.loads((path.parent / 'started.json').read_bytes())
        if (terminal.get('schema') != 'vision-cls-occlusion-v1-terminal' or
                started.get('schema') != 'vision-cls-occlusion-v1-start' or
                terminal.get('status') != 'complete' or terminal.get('mode') != mode or started.get('mode') != mode or
                terminal.get('source_revision') != revision or started.get('source', {}).get('revision') != revision or
                terminal.get('manifest_sha256') != MANIFEST_SHA or started.get('manifest_sha256') != MANIFEST_SHA or
                started.get('selected_ids') != IDS or type(terminal.get('submissions')) is not int or
                type(terminal.get('saved_forwards')) is not int or
                terminal['submissions'] != terminal['saved_forwards']):
            return False
        if mode == 'preflight':
            return terminal['submissions'] == 0
        cases = terminal.get('cases')
        if (mode != 'discovery' or type(cases) is not list or terminal.get('selected') != 4 or
                [case.get('image_id') for case in cases] != IDS):
            return False
        sizes = {'matchable': 23, 'limited-dose': 23, 'dose-ineligible': 16, 'dose-unmatchable': 16}
        return terminal['submissions'] == sum(sizes[case['dose']['status']] for case in cases)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'discovery'), required=True)
    parser.add_argument('--revision', required=True)
    for name in ('checkout', 'assets', 'prior-data', 'output', 'outer', 'custody', 'site-packages', 'pillow'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    paths = (args.checkout, args.assets, args.prior_data, args.output, args.outer,
             args.custody, args.site_packages, args.pillow)
    if (any(not path.is_absolute() for path in paths) or not re.fullmatch('[0-9a-f]{40}', args.revision) or
            args.checkout.resolve() != ROOT or
            not all(path.is_dir() for path in (args.checkout, args.assets, args.prior_data,
                                               args.site_packages, args.pillow)) or
            args.output.exists() or args.outer.exists() or not args.output.parent.is_dir() or args.timeout <= 0):
        raise ValueError('observer arguments differ')
    def overlaps(first, second):
        first, second = first.resolve(), second.resolve()
        return first.is_relative_to(second) or second.is_relative_to(first)
    protected = (args.checkout, args.assets, args.prior_data, args.site_packages, args.pillow)
    if (any(overlaps(a, b) for a in (args.output, args.outer, args.custody) for b in protected) or
            any(overlaps(a, b) for a, b in ((args.output, args.outer), (args.output, args.custody),
                                            (args.outer, args.custody)))):
        raise ValueError('observer roots overlap')
    if shutil.disk_usage(args.checkout).free < 3 * 1024**3 // 2 or shutil.disk_usage(args.output.parent).free < 1024**3:
        raise ValueError('insufficient source/cache or evidence space')
    git_env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    git_env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
    def git(*parts):
        return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', *parts],
                                       cwd=ROOT, env=git_env)
    head = git('rev-parse', 'HEAD').decode().strip()
    if head != args.revision:
        raise ValueError('observer checkout revision differs')
    for name in ('tools/observe_vision_coco_cls_occlusion_discovery.py', 'tools/observe_vision_coco_occlusion.py'):
        if (ROOT / name).read_bytes() != git('show', f'{args.revision}:{name}'):
            raise ValueError('observer source differs')
    spec = importlib.util.spec_from_file_location('_prior_occlusion_observer',
                                                  ROOT / 'tools/observe_vision_coco_occlusion.py')
    prior_observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior_observer)
    python = Path(sys._base_executable).resolve(strict=True)
    interpreter_sha = prior_observer.digest(python)
    args.outer.mkdir(parents=True, exist_ok=False)
    args.custody.mkdir(parents=True, exist_ok=True)
    prefix, temporary = args.outer / 'empty-bytecode-cache', args.outer / 'temporary'
    prefix.mkdir()
    temporary.mkdir()
    bootstrap = ('import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2],sys.argv[3]]); '
                 'sys.argv=sys.argv[4:]; runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [str(python), '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}', '-c', bootstrap,
               str(args.site_packages), str(args.pillow), str(args.checkout),
               str(args.checkout / 'tools/run_vision_coco_cls_occlusion_discovery.py'), '--mode', args.mode,
               '--revision', args.revision, '--assets', str(args.assets), '--prior-data', str(args.prior_data),
               '--output', str(args.output)]
    env = prior_observer.child_environment(os.environ, temporary)
    started = time.monotonic()
    with (args.outer / 'stdout.txt').open('xb') as stdout, (args.outer / 'stderr.txt').open('xb') as stderr:
        prior_observer.claim_phase(args.custody, args.mode, args.revision)
        code, timeout, forced, reaped, error, fatal = prior_observer.supervise(
            command, cwd=args.checkout, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout)
    terminal_path = args.output / 'data/terminal.json'
    complete = (type(code) is int and code == 0 and timeout is False and forced is False and
                reaped is True and error is None and fatal is None and
                prior_observer.digest(python) == interpreter_sha and terminal_complete(terminal_path, args.mode, args.revision))
    receipt = {'schema': 'vision-cls-occlusion-v1-direct-observer', 'mode': args.mode,
               'source_revision': args.revision, 'exit_code': code, 'timed_out': timeout,
               'reaped': reaped, 'forced_stop': forced, 'base_interpreter_sha256': interpreter_sha,
               'error_type': error, 'duration_seconds': time.monotonic() - started,
               'stdout_sha256': prior_observer.digest(args.outer / 'stdout.txt'),
               'stderr_sha256': prior_observer.digest(args.outer / 'stderr.txt'),
               'terminal_sha256': prior_observer.digest(terminal_path) if terminal_path.is_file() else None,
               'status': 'complete' if complete else 'incomplete',
               'scope': 'cooperative direct-child observation; not execution or process-tree attestation'}
    raw = (json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (args.outer / 'receipt.json').open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short observer receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt))
    if fatal is not None:
        raise fatal
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
