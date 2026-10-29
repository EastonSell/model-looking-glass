"""Observe one direct CPU process for the public pair/dose experiment."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    for name in ('repository', 'python', 'cache-dir', 'attempt', 'pycache', 'pre-run', 'execution-context'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repo, python, cache, attempt, prefix, pre_path, context_path = (
        p.resolve() for p in (args.repository, args.python, args.cache_dir, args.attempt,
                               args.pycache, args.pre_run, args.execution_context))
    pre_raw, context_raw = pre_path.read_bytes(), context_path.read_bytes()
    pre, context = json.loads(pre_raw), json.loads(context_raw)
    revision, case_sha = pre['source_revision'], pre['case_sha256']
    if (pre.get('schema') != 'copying-head-pairs-dose-launch-v1'
            or context.get('schema') != 'copying-head-pairs-dose-execution-context-v1'
            or context.get('source_revision') != revision or context.get('case_sha256') != case_sha
            or context.get('exposure_inventories') != pre.get('exposure_inventories')
            or context.get('private_split_status', {}).get('case_sha256') != case_sha
            or any(context['private_split_status'].get(lane, {}).get('status') not in
                   ('no_protected_rows', 'no_overlap') for lane in ('natural', 'injection'))
            or context['private_split_status'].get('injection', {}).get('target_boundary')
               != 'proceed_no_direct_conflict'):
        raise ValueError('pre-run and current custody context do not match')
    git_env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    git_env.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_CONFIG_NOSYSTEM': '1',
                    'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_TERMINAL_PROMPT': '0'})
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, env=git_env).decode().strip()
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'],
                                     cwd=repo, env=git_env)
    if current != revision or status or not python.is_file() or not cache.is_dir():
        raise RuntimeError('clean source checkout, interpreter or offline cache is unavailable')
    if attempt.exists() or prefix.exists() or attempt in prefix.parents or prefix in attempt.parents:
        raise RuntimeError('attempt and separate pycache must be fresh')
    attempt.mkdir()
    prefix.mkdir()
    write_new(attempt / 'pre-run.json', pre_raw)
    write_new(attempt / 'execution-context.json', context_raw)
    env = dict(os.environ)
    env.pop('MKL_CBWR', None)
    env.pop('ATEN_CPU_CAPABILITY', None)
    env.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    command = [str(python), '-B', '-X', 'pycache_prefix=' + str(prefix),
               'scripts/copying_head_pairs_dose_v1.py', '--cache-dir', str(cache),
               '--output-dir', str(attempt / 'run')]
    interrupted, observed_exit, launch_error = False, None, None
    with (attempt / 'stdout.bin').open('xb') as out, (attempt / 'stderr.bin').open('xb') as err:
        try:
            proc = subprocess.Popen(command, cwd=repo, env=env, stdout=out, stderr=err,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                observed_exit = proc.wait()
            except BaseException:
                interrupted = True
                proc.terminate()
                try:
                    observed_exit = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    observed_exit = proc.wait()
        except Exception as error:
            launch_error = type(error).__name__
    run = attempt / 'run'
    hashes = {path.name: digest(path) for path in sorted(run.iterdir()) if path.is_file()} if run.is_dir() else {}
    receipt = {
        'schema': 'copying-head-pairs-dose-process-observation-v1', 'source_revision': revision,
        'pre_run_sha256': digest(attempt / 'pre-run.json'),
        'execution_context_sha256': digest(attempt / 'execution-context.json'),
        'process_exit_code': observed_exit, 'interrupted': interrupted,
        'failure_file_present': (run / 'failure.json').exists(), 'launch_error_type': launch_error,
        'stdout_bytes': (attempt / 'stdout.bin').stat().st_size, 'stdout_sha256': digest(attempt / 'stdout.bin'),
        'stderr_bytes': (attempt / 'stderr.bin').stat().st_size, 'stderr_sha256': digest(attempt / 'stderr.bin'),
        'run_sha256': hashes, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Observed direct child exit and saved-file hashes; not descendant or trusted timestamp attestation.',
    }
    write_new(attempt / 'supervisor.json', json.dumps(receipt, sort_keys=True, separators=(',', ':'),
                                                     allow_nan=False).encode())
    print(json.dumps({'exit': observed_exit, 'interrupted': interrupted, 'launch_error_type': launch_error,
                      'failure_file_present': receipt['failure_file_present'],
                      'result_present': 'result.json' in hashes}), flush=True)
    return 0 if (observed_exit == 0 and not interrupted and launch_error is None
                 and not receipt['failure_file_present'] and 'result.json' in hashes) else 1


if __name__ == '__main__':
    sys.exit(main())
