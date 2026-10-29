"""Observe one FLAN sentiment CPU owner in a fresh isolated child process.

Invoke the direct base interpreter with -I -S -B and a separately prepared
empty -X pycache_prefix. This
operator retains partial evidence and never retries. A direct-child exit is a
local observation, not execution authentication or a descendant-drain proof.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading


PATH_OPTIONS = ('repo', 'venv', 'asset-dir', 'attempt', 'pool', 'pool-salt', 'pool-card', 'pool-observation',
                'selection', 'selection-salt', 'selection-card', 'selection-observation',
                'screen-runtime-receipt', 'screen-exit-observation', 'screen-review', 'scientific-reader-pin')
LOG_LIMIT = 64 * 1024


def _clear_git_environment():
    for name in tuple(os.environ):
        if name.upper().startswith('GIT_'):
            os.environ.pop(name, None)


def _raise_with_priority(primary, cleanup_error):
    if primary is None:
        raise cleanup_error
    if not isinstance(primary, Exception) or isinstance(cleanup_error, Exception):
        raise primary from cleanup_error
    raise cleanup_error from primary


def _empty_prefix():
    prefix = sys.pycache_prefix
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or not prefix
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()):
        raise ValueError('run the direct base interpreter with -I -S -B and a fresh -X pycache_prefix')
    path = Path(prefix)
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise ValueError('bytecode prefix is not a fresh empty directory')


def _receive_grant(expected):
    if type(expected) is not str or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('child grant digest is invalid')
    done = threading.Event()
    def expire():
        if not done.wait(30):
            os._exit(70)
    threading.Thread(target=expire, name='flan-owner-grant-deadline', daemon=True).start()
    try:
        grant = sys.stdin.buffer.read(33)
        if len(grant) != 32 or hashlib.sha256(grant).hexdigest() != expected:
            raise ValueError('child grant is absent or invalid')
    finally:
        done.set()


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short operator record write')
        stream.flush()
        os.fsync(stream.fileno())


def _drain_log(pipe, path, state):
    """Bound each retained log while continuing to drain its child pipe."""
    stream = None
    try:
        try:
            stream = Path(path).open('xb')
        except BaseException as error:
            state['error_type'] = type(error).__name__
        while True:
            block = pipe.read(16384)
            if not block:
                break
            remaining = max(0, LOG_LIMIT - state['retained_bytes'])
            keep = block[:remaining]
            if len(block) > len(keep):
                state['truncated'] = True
            if keep and stream is not None:
                try:
                    if stream.write(keep) != len(keep):
                        raise OSError('short log write')
                    state['retained_bytes'] += len(keep)
                except BaseException as error:
                    state['error_type'] = type(error).__name__
                    stream.close()
                    stream = None
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        state['error_type'] = type(error).__name__
    finally:
        if stream is not None:
            stream.close()
        pipe.close()
        state['finished'] = True


def _load_owner(root, revision):
    """Bind bootstrap bytes before loading the package without adding root to sys.path."""
    git = ['git', '--no-replace-objects']
    head = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=root).strip().decode('ascii')
    if (head != revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision)
            or subprocess.check_output(git + ['cat-file', '-t', revision], cwd=root).strip() != b'commit'):
        raise ValueError('operator checkout revision differs')
    for name in ('tools/run_flan_t5_sentiment_intervention_v1.py', 'looking_glass/__init__.py',
                 'looking_glass/flan_t5_sentiment_owner_v1.py'):
        if (root / name).read_bytes() != subprocess.check_output(git + ['show', revision + ':' + name], cwd=root):
            raise ValueError('operator bootstrap source differs')
    package = root / 'looking_glass'
    spec = importlib.util.spec_from_file_location('looking_glass', package / '__init__.py',
                                                 submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ValueError('cannot resolve the source-bound package')
    module = importlib.util.module_from_spec(spec)
    sys.modules['looking_glass'] = module
    spec.loader.exec_module(module)
    from looking_glass import flan_t5_sentiment_owner_v1 as owner
    return owner


def _load_stage_helper(root):
    path = root / 'tools/run_flan_sentiment_selector_stage_v1.py'
    spec = importlib.util.spec_from_file_location('_flan_sentiment_direct_runtime', path)
    if spec is None or spec.loader is None:
        raise ValueError('cannot resolve the reviewed direct-runtime helper')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATH_OPTIONS:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=10800)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--grant-sha256', help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    _empty_prefix()
    _clear_git_environment()
    args = _parser().parse_args(argv)
    root = args.repo.resolve(strict=True)
    if Path(__file__).resolve() != (root / 'tools/run_flan_t5_sentiment_intervention_v1.py').resolve():
        raise ValueError('operator must execute from its bound checkout')
    if not all(getattr(args, name.replace('-', '_')).is_absolute() for name in PATH_OPTIONS):
        raise ValueError('all operator paths must be absolute')
    if not 1 <= args.timeout_seconds <= 86400:
        raise ValueError('timeout must be between one second and one day')
    if args.child:
        _receive_grant(args.grant_sha256)
    elif args.grant_sha256 is not None:
        raise ValueError('grant digest is reserved for the child')
    owner = _load_owner(root, args.revision)
    source = owner.source_snapshot(root, args.revision)
    direct_runtime = _load_stage_helper(root)
    site, _ = direct_runtime._dependency_site(args.venv)
    if args.child:
        primary = None
        try:
            if str(site) in sys.path:
                raise ValueError('venv site-packages was already on the search path')
            sys.path.append(str(site))
            print(json.dumps(direct_runtime._process_identity(args.venv, os.getpid(), os.getppid()),
                  sort_keys=True), flush=True)
            owner.run(args)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                _empty_prefix()
                if owner.source_snapshot(root, args.revision) != source:
                    raise ValueError('source changed during child execution')
            except BaseException as cleanup_error:
                _raise_with_priority(primary, cleanup_error)
        return

    attempt = args.attempt
    if not attempt.parent.is_dir() or attempt.exists():
        raise ValueError('attempt must be a new directory in an existing parent')
    if shutil.disk_usage(attempt.parent).free < 2 * 1024**3:
        raise ValueError('less than 2 GiB free before model loading')
    attempt.mkdir()
    prefix = attempt / 'child-pycache'
    prefix.mkdir()
    _write_new(attempt / 'operator-start.json', {
        'schema': 'flan-t5-sentiment-operator-v1', 'source': source,
        'timeout_seconds': args.timeout_seconds, 'child_prefix_empty_before': not any(prefix.iterdir()),
        'scope': 'Local direct-child observation; trusted interpreter and dependencies.'})
    environment = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONINSPECT', 'ATEN_CPU_CAPABILITY'):
        environment.pop(name, None)
    environment.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE',
                        'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    grant = os.urandom(32)
    command = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(root / 'tools/run_flan_t5_sentiment_intervention_v1.py'), '--child',
               '--revision', args.revision, '--timeout-seconds', str(args.timeout_seconds),
               '--grant-sha256', hashlib.sha256(grant).hexdigest()]
    for name in PATH_OPTIONS:
        command.extend(['--' + name, str(getattr(args, name.replace('-', '_')))])
    child, code, timed_out, failure, primary = None, None, False, None, None
    drainers = []
    logs = {name: {'retained_bytes': 0, 'truncated': False, 'error_type': None, 'finished': False}
            for name in ('stdout', 'stderr')}
    try:
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        factory = subprocess.Popen
        child_options = {'cwd': root, 'env': environment, 'stdin': subprocess.PIPE,
                         'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, **options}
        if isinstance(factory, type):
            child = factory.__new__(factory)
            factory.__init__(child, command, **child_options)
        else:
            child = factory(command, **child_options)
        for name in ('stdout', 'stderr'):
            thread = threading.Thread(target=_drain_log, args=(getattr(child, name), attempt / (name + '.log'),
                                                               logs[name]), daemon=True)
            drainers.append(thread)
            thread.start()
        _write_new(attempt / 'operator-spawn.json', {'pid': child.pid, 'parent_pid': os.getpid()})
        if child.stdin.write(grant) != len(grant):
            raise OSError('short child grant write')
        child.stdin.flush()
        child.stdin.close()
        try:
            code = child.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait(timeout=15)
    except BaseException as error:
        primary = error
        failure = type(error).__name__
        if child is not None:
            try:
                if child.poll() is None:
                    child.kill()
                    code = child.wait(timeout=15)
            except BaseException as cleanup_error:
                if isinstance(error, Exception) and not isinstance(cleanup_error, Exception):
                    failure = type(cleanup_error).__name__
                    primary = cleanup_error
                    raise cleanup_error from error
        raise
    finally:
        try:
            for thread in drainers:
                try:
                    thread.join(timeout=15)
                except RuntimeError:
                    pass
            logs_complete = (len(drainers) == 2 and all(not thread.is_alive() for thread in drainers)
                             and all(item['finished'] and not item['truncated'] and item['error_type'] is None
                                     for item in logs.values()))
            terminal = attempt / 'child' / 'terminal.json'
            events = attempt / 'child' / 'events.jsonl'
            owner_receipt = attempt / 'owner-receipt.json'
            terminal_sha = _hash_file(terminal) if terminal.is_file() else None
            events_sha = _hash_file(events) if events.is_file() else None
            owner_sha = _hash_file(owner_receipt) if owner_receipt.is_file() else None
            consistent = False
            try:
                record = json.loads(owner_receipt.read_bytes())
                terminal_record = json.loads(terminal.read_bytes())
                consistent = (record['status'] == 'complete_candidate' and record['source_revision'] == args.revision
                              and record['terminal_sha256'] == terminal_sha and record['events_sha256'] == events_sha
                              and terminal_record['status'] == 'complete_candidate'
                              and owner.source_snapshot(root, args.revision) == source and logs_complete and failure is None
                              and direct_runtime._check_child_identity(attempt / 'stdout.log', args.venv, child.pid))
            except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
                pass
            _write_new(attempt / 'operator-exit.json', {
                'schema': 'flan-t5-sentiment-operator-v1', 'direct_child_exit_code': code,
                'child_reaped': code is not None, 'timed_out': timed_out, 'operator_exception_type': failure,
                'terminal_sha256': terminal_sha, 'events_sha256': events_sha, 'owner_receipt_sha256': owner_sha,
                'terminal_consistent': consistent, 'child_prefix_empty_after': not any(prefix.iterdir()),
                'logs': logs, 'log_drain_complete': logs_complete,
                'stdout_sha256': _hash_file(attempt / 'stdout.log') if (attempt / 'stdout.log').is_file() else None,
                'stderr_sha256': _hash_file(attempt / 'stderr.log') if (attempt / 'stderr.log').is_file() else None})
        except BaseException as cleanup_error:
            _raise_with_priority(primary, cleanup_error)
    if code != 0 or timed_out or not consistent or any(prefix.iterdir()):
        raise RuntimeError('attempt is incomplete or uncertain; retain its evidence')
    print(json.dumps({'status': 'complete_candidate', 'source_revision': args.revision,
                      'terminal_sha256': terminal_sha, 'events_sha256': events_sha}, sort_keys=True))


if __name__ == '__main__':
    main()
