"""Observe one direct color-study supervisor; never load a model here.

The receipt is a cooperative local process observation, not attestation of
executed bytes or a whole descendant tree. Missing/ambiguous observations do
not authorize a later stage. Exact source and public-commit review is separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

SCHEMA = 'distilbert-color-supervisor-observation-v1'
VALIDATION_SCHEMA = 'distilbert-color-stage-validation-v1'
FILES = ('claim.json', 'prelaunch.json', 'worker-start.json', 'events.jsonl',
         'result.json', 'exit.json')
BOOTSTRAP = ("import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
             "sys.path.insert(1,sys.argv.pop(1));"
             "runpy.run_module('looking_glass.distilbert_color_edit_runner_v1',run_name='__main__')")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def _read(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('missing, linked or oversized observation input')
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError('observation input grew beyond its bound')
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, raw):
    with Path(path).open('xb', buffering=0) as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short observation write')
        stream.flush()
        os.fsync(stream.fileno())
    if Path(path).read_bytes() != raw:
        raise OSError('observation readback differs')


def _empty(directory):
    return directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir())


def _relative(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _clean_environment(temp):
    allowed = ('PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT',
               'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 'PROCESSOR_IDENTIFIER')
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(TEMP=str(temp), TMP=str(temp), CUDA_VISIBLE_DEVICES='-1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    return environment


def _verify_marker(root, phase, source, pid):
    stage = root / phase
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError('stage directory is absent or linked')
    marker_raw = _read(stage / 'validated.json', 1024 * 1024)
    marker = _parse(marker_raw)
    if (type(marker) is not dict or marker.get('schema') != VALIDATION_SCHEMA or
            marker.get('phase') != phase or marker.get('source_commit') != source or
            type(marker.get('supervisor_pid')) is not int or marker['supervisor_pid'] != pid or
            type(marker.get('files_sha256')) is not dict or set(marker['files_sha256']) != set(FILES)):
        raise ValueError('validation marker does not bind the observed supervisor and stage')
    for name in FILES:
        expected = marker['files_sha256'][name]
        if type(expected) is not str or re.fullmatch('[0-9a-f]{64}', expected) is None:
            raise ValueError('invalid file hash in validation marker')
        if _sha(_read(stage / name, 128 * 1024 * 1024)) != expected:
            raise ValueError('validated stage file differs from marker')
    run_raw = _read(root / 'run.json', 1024 * 1024)
    run = _parse(run_raw)
    if type(run) is not dict or run.get('source_commit') != source:
        raise ValueError('run identity differs')
    return _sha(marker_raw), _sha(run_raw)


def observe(*, python, site, repo, cache, scratch, root, phase, source, prelaunch, prelaunch_sha256,
            observer_dir, publications=None, stage_timeout=900, observer_timeout=1200):
    python, site, repo, cache, scratch, prelaunch = (Path(p).resolve(strict=True) for p in (python, site, repo, cache, scratch, prelaunch))
    root_path, observer_path = Path(root).absolute(), Path(observer_dir).absolute()
    root = root_path.parent.resolve(strict=True) / root_path.name
    observer_dir = observer_path.parent.resolve(strict=True) / observer_path.name
    if (phase not in ('screen', 'development', 'evaluation') or
            type(source) is not str or re.fullmatch('[0-9a-f]{40}', source) is None or
            type(prelaunch_sha256) is not str or re.fullmatch('[0-9a-f]{64}', prelaunch_sha256) is None or
            _sha(_read(prelaunch, 256 * 1024)) != prelaunch_sha256 or
            type(stage_timeout) is not int or not 1 <= stage_timeout <= 3600 or
            type(observer_timeout) is not int or not stage_timeout < observer_timeout <= 3900 or
            not python.is_file() or not site.is_dir() or not repo.is_dir() or not cache.is_dir() or not scratch.is_dir()):
        raise ValueError('invalid fixed supervisor inputs')
    if (root.is_symlink() or observer_dir.is_symlink() or
            any(_relative(observer_dir, path) for path in (repo, cache, root, site, scratch)) or
            any(_relative(root, path) for path in (repo, cache, site, scratch)) or
            any(_relative(scratch, path) for path in (repo, cache, site))):
        raise ValueError('observer and run paths must be disjoint from checkout, cache and dependency site')
    pubs = publications or {}
    required = set(() if phase == 'screen' else ('screen',) if phase == 'development' else ('screen', 'development'))
    if type(pubs) is not dict or set(pubs) != required:
        raise ValueError('predecessor publication set differs')
    for ref in pubs.values():
        if (type(ref) is not dict or set(ref) != {'commit', 'prefix'} or
                type(ref['commit']) is not str or re.fullmatch('[0-9a-f]{40}', ref['commit']) is None or
                type(ref['prefix']) is not str or not ref['prefix']):
            raise ValueError('invalid predecessor reference')
    observer_dir.mkdir(exist_ok=False)
    bytecode, temp = observer_dir / 'bytecode', observer_dir / 'temp'
    bytecode.mkdir()
    temp.mkdir()
    if not _empty(bytecode):
        raise ValueError('bytecode prefix must start empty')
    argv = [str(python), '-E', '-s', '-B', '-X', 'pycache_prefix=' + str(bytecode),
            '-c', BOOTSTRAP, str(repo), str(site), '--repo', str(repo), '--site', str(site),
            '--cache', str(cache), '--scratch', str(scratch), '--run-root', str(root), '--phase', phase,
            '--source-commit', source, '--prelaunch', str(prelaunch),
            '--prelaunch-sha256', prelaunch_sha256, '--timeout', str(stage_timeout)]
    for prior in ('screen', 'development'):
        if prior in pubs:
            argv.extend(('--' + prior + '-public-commit', pubs[prior]['commit'],
                         '--' + prior + '-public-prefix', pubs[prior]['prefix']))
    env = _clean_environment(temp)
    command = {'argv': argv, 'cwd': str(repo), 'environment_sha256': _sha(_json_bytes(env)),
               'scope': 'Private operator command record; environment values are represented by a hash.'}
    command_raw = _json_bytes(command)
    command_sha = _sha(command_raw)
    start_raw = _json_bytes({
        'phase': phase, 'source_commit': source, 'command_sha256': command_sha,
        'observer_pid': os.getpid(), 'before_spawn_utc': datetime.now(timezone.utc).isoformat()})
    if len(command_raw) + len(start_raw) > 12 * 1024:
        raise ValueError('private command and start records exceed their observation budget')
    _write(observer_dir / 'command.json', command_raw)
    _write(observer_dir / 'observer-start.json', start_raw)
    process = None
    pid = None
    code = None
    sentinel = root / phase / 'supervisor-observation-failure.json'
    receipt_ready = False
    try:
        process = subprocess.Popen(argv, cwd=str(repo), env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid = process.pid
        if type(pid) is not int or pid <= 0:
            raise RuntimeError('direct supervisor PID is unavailable')
        code = process.wait(timeout=observer_timeout)
        if type(code) is not int or code != 0:
            raise RuntimeError('direct supervisor did not exit zero')
        if not _empty(bytecode):
            raise RuntimeError('bytecode prefix is not empty after supervisor exit')
        validation_sha, run_sha = _verify_marker(root, phase, source, pid)
        receipt = {'schema': SCHEMA, 'phase': phase, 'source_commit': source,
                   'supervisor_pid': pid, 'returncode': code,
                   'validation_sha256': validation_sha, 'run_sha256': run_sha,
                   'command_sha256': command_sha, 'observer_external_bytes': len(command_raw) + len(start_raw),
                   'scope': 'Cooperative local direct-process exit observation; not execution or process-tree attestation.'}
        receipt_raw = _json_bytes(receipt)
        if len(receipt_raw) + receipt['observer_external_bytes'] > 16 * 1024:
            raise ValueError('supervisor observation and external records exceed the reserved 16 KiB')
        # The stage is unpromotable while this sentinel exists. Removing it
        # after success is the commit point, even if a later return is interrupted.
        _write(sentinel, _json_bytes({'status': 'invalid-incomplete', 'phase': phase,
                    'source_commit': source, 'command_sha256': command_sha,
                    'scope': 'Observation has not committed; this stage is not promotable.'}))
        _write(root / phase / 'supervisor-observation.json', receipt_raw)
        receipt_ready = True
        sentinel.unlink()
        return receipt
    except BaseException as error:
        if receipt_ready and not sentinel.exists():
            # Receipt bytes and readback completed before sentinel removal.
            # The commit is already valid; do not create a contradictory failure.
            raise
        cleanup = 'not_observed'
        if process is not None:
            try:
                cleanup_code = process.poll()
                if cleanup_code is None:
                    process.kill()
                    cleanup_code = process.wait(timeout=30)
                if type(code) is not int and type(cleanup_code) is int:
                    code = cleanup_code
                if type(pid) is not int and type(process.pid) is int and process.pid > 0:
                    pid = process.pid
                cleanup = 'direct_process_reaped'
            except BaseException:
                cleanup = 'direct_process_cleanup_unknown'
        failure = {'phase': phase, 'source_commit': source, 'command_sha256': command_sha,
                   'status': 'invalid-incomplete', 'error_type': type(error).__name__[:256],
                   'supervisor_pid': pid, 'returncode': code,
                   'direct_process_cleanup': cleanup,
                   'scope': ('Unknown spawn or interrupted observation is incomplete. '
                             'Reaping this direct supervisor does not establish whether its child stopped; '
                             'reconcile any live child before retry or promotion.')}
        try:
            failure['observer_external_bytes'] = len(command_raw) + len(start_raw)
            while True:
                failure_raw = _json_bytes(failure)
                actual_external = len(command_raw) + len(start_raw) + len(failure_raw)
                if failure['observer_external_bytes'] == actual_external:
                    break
                failure['observer_external_bytes'] = actual_external
            if actual_external > 16 * 1024:
                raise ValueError('failed observation exceeds the reserved metadata budget')
            _write(observer_dir / 'observer-failure.json', failure_raw)
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'site', 'repo', 'cache', 'scratch', 'run-root', 'prelaunch', 'observer-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--phase', choices=('screen', 'development', 'evaluation'), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--prelaunch-sha256', required=True)
    parser.add_argument('--stage-timeout', type=int, default=900)
    parser.add_argument('--observer-timeout', type=int, default=1200)
    for prior in ('screen', 'development'):
        parser.add_argument('--' + prior + '-public-commit')
        parser.add_argument('--' + prior + '-public-prefix')
    args = parser.parse_args()
    pubs = {}
    for name in ('screen', 'development'):
        commit, prefix = getattr(args, name + '_public_commit'), getattr(args, name + '_public_prefix')
        if commit is not None or prefix is not None:
            pubs[name] = {'commit': commit, 'prefix': prefix}
    result = observe(python=args.python, site=args.site, repo=args.repo, cache=args.cache, scratch=args.scratch, root=args.run_root,
        phase=args.phase, source=args.source_commit, prelaunch=args.prelaunch,
        prelaunch_sha256=args.prelaunch_sha256, observer_dir=args.observer_dir,
        publications=pubs, stage_timeout=args.stage_timeout, observer_timeout=args.observer_timeout)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
