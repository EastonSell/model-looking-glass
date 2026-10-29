"""Observe one isolated selector stage; retain bounded logs and partial records.

This is a cooperative direct-child observation, not execution authentication
or proof about descendants. It neither forms a pool nor retries a stage.
"""
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


def _empty_prefix():
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()):
        raise ValueError('run the direct base interpreter with -I -S -B and a fresh -X pycache_prefix')
    path = Path(sys.pycache_prefix)
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise ValueError('observer bytecode prefix is not empty')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('cannot resolve source-bound stage helper')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bootstrap(root, revision):
    for name in tuple(os.environ):
        if name.upper().startswith('GIT_'):
            os.environ.pop(name, None)
    git = ['git', '--no-replace-objects']
    head = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=root).strip().decode('ascii')
    if head != revision or subprocess.check_output(git + ['cat-file', '-t', head], cwd=root).strip() != b'commit':
        raise ValueError('observer checkout revision differs')
    for name in ('tools/observe_flan_sentiment_selector_stage_v1.py',
                 'tools/run_flan_sentiment_selector_stage_v1.py',
                 'tools/run_flan_t5_sentiment_intervention_v1.py'):
        if (root / name).read_bytes() != subprocess.check_output(git + ['show', head + ':' + name], cwd=root):
            raise ValueError('observer source differs from its commit')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--venv', required=True, type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('selector_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = args.selector_args
    if rest and rest[0] == '--':
        rest = rest[1:]
    if not rest or rest[0] not in ('commit-pool', 'select'):
        raise ValueError('selector stage is missing')
    _empty_prefix()
    for name in tuple(os.environ):
        if name.upper().startswith('GIT_'):
            os.environ.pop(name, None)
    root = Path(__file__).resolve().parents[1]
    _bootstrap(root, args.revision)
    stage_helper = _load(root / 'tools/run_flan_sentiment_selector_stage_v1.py', '_flan_stage_helper')
    output_helper = _load(root / 'tools/run_flan_t5_sentiment_intervention_v1.py', '_flan_output_helper')
    source = stage_helper._snapshot(root, args.revision)
    stage_helper._dependency_site(args.venv)
    if (not args.attempt.is_absolute() or not args.attempt.parent.is_dir() or args.attempt.exists()
            or not 1 <= args.timeout_seconds <= 86400):
        raise ValueError('observer requires a new absolute attempt and bounded timeout')
    if shutil.disk_usage(args.attempt.parent).free < 2 * 1024**3:
        raise ValueError('less than 2 GiB free before custody allocation')
    args.attempt.mkdir()
    prefix = args.attempt / 'child-pycache'
    prefix.mkdir()
    stage_receipt = args.attempt / 'stage-runtime.json'
    output_helper._write_new(args.attempt / 'stage-start.json', {
        'schema': 'flan-t5-sentiment-selector-stage-observer-v1', 'stage': rest[0],
        'selector_revision': args.revision, 'child_prefix_empty_before': not any(prefix.iterdir())})
    environment = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONINSPECT'):
        environment.pop(name, None)
    environment.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'TOKENIZERS_PARALLELISM': 'false'})
    grant = os.urandom(32)
    command = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(root / 'tools/run_flan_sentiment_selector_stage_v1.py'), '--revision', args.revision,
               '--stage-receipt', str(stage_receipt), '--venv', str(args.venv),
               '--grant-sha256', hashlib.sha256(grant).hexdigest(), '--', *rest]
    child, code, timed_out, failure, drainers = None, None, False, None, []
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
            thread = threading.Thread(target=output_helper._drain_log,
                                      args=(getattr(child, name), args.attempt / (name + '.log'), logs[name]),
                                      daemon=True)
            drainers.append(thread)
            thread.start()
        output_helper._write_new(args.attempt / 'stage-spawn.json', {'pid': child.pid, 'parent_pid': os.getpid()})
        if child.stdin.write(grant) != len(grant):
            raise OSError('short selector child grant write')
        child.stdin.flush()
        child.stdin.close()
        try:
            code = child.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait(timeout=15)
    except BaseException as error:
        failure = type(error).__name__
        if child is not None:
            try:
                if child.poll() is None:
                    child.kill()
                    code = child.wait(timeout=15)
            except BaseException as cleanup_error:
                if isinstance(error, Exception) and not isinstance(cleanup_error, Exception):
                    failure = type(cleanup_error).__name__
                    raise cleanup_error from error
        raise
    finally:
        for thread in drainers:
            try:
                thread.join(timeout=15)
            except RuntimeError:
                pass
        logs_complete = (len(drainers) == 2 and all(not thread.is_alive() for thread in drainers)
                         and all(item['finished'] and not item['truncated'] and item['error_type'] is None
                                 for item in logs.values()))
        receipt_sha = output_helper._hash_file(stage_receipt) if stage_receipt.is_file() else None
        consistent = False
        try:
            receipt = json.loads(stage_receipt.read_bytes())
            _bootstrap(root, args.revision)
            consistent = (receipt['schema'] == 'flan-t5-sentiment-selector-stage-runtime-v1'
                          and receipt['stage'] == rest[0] and receipt['status'] == 'completed'
                          and receipt['selector_revision'] == args.revision
                          and stage_helper._snapshot(root, args.revision) == source
                          and not any(prefix.iterdir()) and logs_complete
                          and stage_helper._check_child_identity(args.attempt / 'stdout.log', args.venv, child.pid))
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
            pass
        for name in ('stdout', 'stderr'):
            log = args.attempt / (name + '.log')
            logs[name]['sha256'] = output_helper._hash_file(log) if log.is_file() else None
        ok = code == 0 and not timed_out and failure is None and consistent
        output_helper._write_new(args.attempt / 'stage-exit.json', {
            'schema': 'flan-t5-sentiment-selector-stage-exit-v1',
            'stage': rest[0], 'status': 'completed' if ok else 'incomplete',
            'stage_receipt_sha256': receipt_sha, 'reaped': code is not None,
            'exit_code': code, 'timed_out': timed_out, 'exception_type': failure,
            'selector_revision': args.revision, 'child_prefix_empty_after': not any(prefix.iterdir()),
            'logs': logs, 'scope': 'Local direct-child observation; no execution authentication.'})
    if not ok:
        raise RuntimeError('selector stage incomplete; retain all evidence and do not retry')
    print(json.dumps({'status': 'completed', 'stage': rest[0], 'stage_receipt_sha256': receipt_sha}))


if __name__ == '__main__':
    main()
