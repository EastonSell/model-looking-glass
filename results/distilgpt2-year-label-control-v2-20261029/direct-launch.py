"""Observe one direct v2 child and produce its separate process receipt.

This cooperative operator wrapper does not independently attest execution or
descendant cleanup. The path-bearing private launch file is never published.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--pre-run-sha256', required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    launch_path = attempt/'pre-run.json'
    if sha(launch_path) != args.pre_run_sha256:
        raise ValueError('pre-run bytes differ')
    launch = json.loads(launch_path.read_bytes())
    private = json.loads((attempt/'private-launch.json').read_bytes())
    if (launch.get('schema') != 'distilgpt2-year-label-control-launch-v2'
            or launch.get('source_revision') != 'b256a6846ad2386d0c0c306003026bbd9b2893a8'
            or launch.get('source_tree') != 'f23aa6799a145b9168677563da91d5ec78abc9b7'):
        raise ValueError('unexpected source or launch schema')
    source = Path(private['source_root']).resolve(strict=True)
    prefix = Path(private['pycache_prefix'])
    output = Path(private['output_dir'])
    prior = Path(private['prior_attempt_dir'])
    cache = Path(private['cache_dir'])
    if (not prefix.is_absolute() or prefix.is_symlink() or not prefix.is_dir() or any(prefix.iterdir())
            or not output.is_absolute() or output.exists() or output.parent != attempt
            or not prior.is_absolute() or not prior.is_dir() or not cache.is_absolute() or not cache.is_dir()):
        raise ValueError('unexpected output, prior, cache or empty-prefix state')
    prefix_hash = hashlib.sha256(os.path.normcase(str(prefix.resolve(strict=True))).encode()).hexdigest()
    if prefix_hash != launch['intended_execution']['pycache_prefix_sha256']:
        raise ValueError('empty cache prefix differs from the public launch')
    if (attempt/'supervisor.json').exists() or (attempt/'stdout.txt').exists() or (attempt/'stderr.txt').exists():
        raise ValueError('the one-shot attempt was already started')
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'OMP_NUM_THREADS': '1'})
    env.pop('MKL_CBWR', None)
    env.pop('ATEN_CPU_CAPABILITY', None)
    cmd = [sys.executable, '-B', '-X', f'pycache_prefix={prefix}',
           str(source/'scripts/year_label_control_v2.py'), '--cache-dir', str(cache),
           '--prior-attempt-dir', str(prior), '--output-dir', str(output)]
    interrupted = False
    launcher_error_type = None
    process = None
    with (attempt/'stdout.txt').open('xb') as stdout, (attempt/'stderr.txt').open('xb') as stderr:
        try:
            process = subprocess.Popen(cmd, cwd=source, env=env, stdout=stdout, stderr=stderr)
            print(json.dumps({'event': 'direct_child_started', 'pid': process.pid,
                              'observed_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
            code = process.wait()
        except BaseException as error:
            if process is None:
                # Popen may have started a child without returning its handle.
                # The attempt is uncertain and must not be retried blindly.
                raise
            interrupted = True
            launcher_error_type = type(error).__name__
            if process.poll() is None:
                process.terminate()
            code = process.wait()
    receipt = [sys.executable, '-B', '-X', f'pycache_prefix={prefix}',
               str(source/'scripts/year_label_receipt_v2.py'), str(attempt), '--exit-code', str(code),
               '--pre-run-sha256', args.pre_run_sha256, '--source-revision', launch['source_revision'],
               '--source-tree', launch['source_tree'], '--pycache-prefix', str(prefix)]
    if interrupted:
        receipt.append('--interrupted')
    result = subprocess.run(receipt, cwd=source, env=env, capture_output=True, check=True)
    try:
        print(json.dumps({'event': 'direct_child_reaped', 'exit_code': code, 'interrupted': interrupted,
                          'launcher_error_type': launcher_error_type,
                          'observed_utc': datetime.now(timezone.utc).isoformat(),
                          'receipt': json.loads(result.stdout)}), flush=True)
    except OSError:
        # A broken progress stream does not undo an already durable receipt.
        pass
    return 1 if interrupted or code != 0 else 0


if __name__ == '__main__':
    raise SystemExit(main())
