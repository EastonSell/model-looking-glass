"""Record the operator-observed exit and hashes of a closed direct-process attempt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--exit-code', type=int, required=True)
    parser.add_argument('--pre-run-sha256', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--source-tree', required=True)
    parser.add_argument('--pycache-prefix', required=True, type=Path)
    parser.add_argument('--interrupted', action='store_true')
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    pre_run = attempt / 'pre-run.json'
    if sha(pre_run) != args.pre_run_sha256:
        raise ValueError('pre-run bytes differ from the independently recorded hash')
    launch = json.loads(pre_run.read_bytes())
    if (launch.get('schema') != 'distilgpt2-year-label-control-launch-v2'
            or launch.get('source_revision') != args.source_revision
            or launch.get('source_tree') != args.source_tree):
        raise ValueError('launch source identity differs from the observed source')
    run, stdout, stderr = attempt / 'run', attempt / 'stdout.txt', attempt / 'stderr.txt'
    files = sorted(p for p in run.iterdir() if p.is_file()) if run.is_dir() else []
    prefix = args.pycache_prefix
    prefix_empty = prefix.is_absolute() and not prefix.is_symlink() and prefix.is_dir() and not any(prefix.iterdir())
    prefix_sha = hashlib.sha256(os.path.normcase(str(prefix.resolve())).encode('utf-8')).hexdigest()
    prefix_matches = prefix_sha == launch['intended_execution']['pycache_prefix_sha256']
    receipt = {
        'schema': 'distilgpt2-year-label-control-process-observation-v2',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'direct process exit observed by the operator; no descendant-process attestation',
        'source_revision': args.source_revision,
        'source_tree': args.source_tree,
        'pre_run_sha256': args.pre_run_sha256,
        'process_exit_code': args.exit_code,
        'interrupted': args.interrupted,
        'failure_file_present': (run / 'failure.json').exists(),
        'pycache_prefix_sha256': prefix_sha,
        'pycache_prefix_matches_pre_run': prefix_matches,
        'pycache_prefix_empty_after': prefix_empty,
        'run_sha256': {p.name: sha(p) for p in files},
        'stdout_sha256': sha(stdout),
        'stderr_sha256': sha(stderr),
        'stderr_bytes': stderr.stat().st_size,
    }
    data = (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode('utf-8')
    with (attempt / 'supervisor.json').open('xb') as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps({'receipt_sha256': hashlib.sha256(data).hexdigest(),
                      'run_files': len(files), 'process_exit_code': args.exit_code,
                      'interrupted': args.interrupted,
                      'failure_file_present': receipt['failure_file_present'],
                      'pycache_prefix_matches_pre_run': prefix_matches,
                      'pycache_prefix_empty_after': prefix_empty,
                      'stderr_bytes': receipt['stderr_bytes']}))


if __name__ == '__main__':
    main()
