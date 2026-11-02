"""One-shot cooperative direct-child observation for a fixed study phase."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def observe_child(command, *, cwd, env, stdout, stderr, popen=subprocess.Popen, progress=print):
    """On a recoverable post-Popen error, terminate and reap the exact child."""
    process = None
    interrupted = False
    error_type = None
    started = None
    try:
        process = popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        started = datetime.now(timezone.utc).isoformat()
        progress(json.dumps({'event': 'direct_child_started', 'pid': process.pid,
                             'observed_utc': started}), flush=True)
        exit_code = process.wait()
    except BaseException as error:
        if process is None:
            # A lost Popen return cannot certify whether a child was created.
            raise
        interrupted = True
        error_type = type(error).__name__
        if process.poll() is None:
            process.terminate()
        exit_code = process.wait()
    return {'exit_code': exit_code, 'interrupted': interrupted, 'launcher_error_type': error_type,
            'start_observed_utc': started, 'reaped_observed_utc': datetime.now(timezone.utc).isoformat()}


def assess_receipt(receipt, summary):
    if type(summary) is not dict:
        return False
    phase = receipt['phase']
    ids = range(1, 9) if phase == 'discovery' else range(9, 17)
    expected = {'events.jsonl', 'phase-summary.json'} | {f'known-injection-{i:02d}.f32' for i in ids}
    return (receipt['exit_code'] == 0 and receipt['interrupted'] is False
            and receipt['prefix_empty_after'] is True and receipt['stderr']['bytes'] == 0
            and receipt['stage_empty_after'] is True
            and receipt['failure_file_present'] is False and set(receipt['run_files']) == expected
            and summary == {'phase': phase, 'cases': 8, 'completed_actions': 120,
                            'raw_bytes': summary.get('raw_bytes'),
                            'metadata_bytes': summary.get('metadata_bytes'), 'status': 'complete'}
            and type(summary['raw_bytes']) is int and 0 < summary['raw_bytes'] <= 75 * 1024 * 1024
            and type(summary['metadata_bytes']) is int and 0 < summary['metadata_bytes'] <= 8 * 1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--phase', choices=('discovery', 'held-out'), required=True)
    parser.add_argument('--pre-run-sha256', required=True)
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    pre_path = attempt / 'pre-run.json'
    if sha(pre_path) != args.pre_run_sha256:
        raise ValueError('pre-run digest differs')
    pre = json.loads(pre_path.read_bytes())
    private = json.loads((attempt / 'private-launch.json').read_bytes())
    if pre.get('schema') != 'public-known-injection-pre-run-v1':
        raise ValueError('unexpected pre-run schema')
    if sha(Path(__file__).resolve()) != pre['direct_launcher_sha256']:
        raise ValueError('direct launcher bytes differ from pre-run')
    source = Path(private['source_root']).resolve(strict=True)
    prefix = Path(private['prefixes'][args.phase])
    output = attempt / ('run-' + args.phase)
    cache, inventory = Path(private['cache_dir']), Path(private['inventory'])
    stage, temp = Path(private['stage_dir']), Path(private['temp_dir'])
    if (not prefix.is_absolute() or prefix.is_symlink() or not prefix.is_dir() or any(prefix.iterdir())
            or not cache.is_absolute() or not cache.is_dir() or not inventory.is_absolute() or not inventory.is_file()
            or not stage.is_absolute() or stage.is_symlink() or not stage.is_dir() or any(stage.iterdir())
            or not temp.is_absolute() or temp.is_symlink() or not temp.is_dir()
            or output.exists()):
        raise ValueError('unexpected private cache, inventory, output or prefix state')
    prefix_hash = hashlib.sha256(os.path.normcase(str(prefix.resolve(strict=True))).encode()).hexdigest()
    if prefix_hash != pre['pycache_prefix_sha256'][args.phase]:
        raise ValueError('bytecode-cache prefix differs from pre-run')
    stage_hash = hashlib.sha256(os.path.normcase(str(stage.resolve(strict=True))).encode()).hexdigest()
    if stage_hash != pre['stage_dir_sha256']:
        raise ValueError('checkpoint staging directory differs from pre-run')
    if any(os.path.splitdrive(str(path.resolve()))[0].upper() != pre['output_drive']
           for path in (attempt, prefix, stage, temp)):
        raise ValueError('output, cache prefix, stage and temporary directories must use the declared volume')
    if (shutil.disk_usage(attempt).free < 2 * 1024 ** 3
            or shutil.disk_usage(stage).free < 2 * 1024 ** 3
            or any(shutil.disk_usage(path).free < 256 * 1024 ** 2 for path in (source, cache))):
        raise ValueError('insufficient free space before launching the child')
    stdout_path, stderr_path = attempt / (args.phase + '.stdout.txt'), attempt / (args.phase + '.stderr.txt')
    receipt_path = attempt / (args.phase + '.receipt.json')
    if any(path.exists() for path in (stdout_path, stderr_path, receipt_path)):
        raise ValueError('this one-shot phase was already started')
    command = [sys.executable, '-B', '-X', f'pycache_prefix={prefix}',
               str(source / 'experiments/public_known_injection_v1/run_phase.py'),
               '--phase', args.phase, '--source-root', str(source), '--inventory', str(inventory),
               '--cache-dir', str(cache), '--stage-dir', str(stage), '--pre-run', str(pre_path), '--pre-run-sha256', args.pre_run_sha256,
               '--output-dir', str(output)]
    if args.phase == 'held-out':
        freeze = Path(private['discovery_freeze'])
        if not args.freeze_sha256 or not freeze.is_absolute() or sha(freeze) != args.freeze_sha256:
            raise ValueError('held-out phase requires the verified published freeze')
        command.extend(['--discovery-dir', str(attempt / 'run-discovery'), '--freeze', str(freeze),
                        '--freeze-sha256', args.freeze_sha256])
    elif args.freeze_sha256 is not None:
        raise ValueError('discovery phase must not receive a freeze')
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'OMP_NUM_THREADS': '1', 'PYTHONPATH': str(source), 'TEMP': str(temp), 'TMP': str(temp)})
    for key in ('MKL_CBWR', 'ATEN_CPU_CAPABILITY'):
        env.pop(key, None)
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        observed = observe_child(command, cwd=source, env=env, stdout=stdout, stderr=stderr)
    files = {}
    if output.is_dir():
        for path in sorted(output.rglob('*')):
            if path.is_symlink() or not path.is_file():
                raise ValueError('unexpected run artifact')
            files[path.relative_to(output).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    receipt = {'schema': 'public-known-injection-direct-receipt-v1', 'phase': args.phase,
               **observed, 'pre_run_sha256': args.pre_run_sha256,
               'source_revision': pre['source_revision'], 'source_tree': pre['source_tree'],
               'reservation_sha256': pre['reservation_sha256'], 'freeze_sha256': args.freeze_sha256,
               'prefix_sha256': prefix_hash, 'prefix_empty_after': not any(prefix.iterdir()),
               'stage_dir_sha256': stage_hash, 'stage_empty_after': not any(stage.iterdir()),
               'free_bytes_after': {'output': shutil.disk_usage(attempt).free,
                                    'source': shutil.disk_usage(source).free,
                                    'cache': shutil.disk_usage(cache).free},
               'stdout': {'bytes': stdout_path.stat().st_size, 'sha256': sha(stdout_path)},
               'stderr': {'bytes': stderr_path.stat().st_size, 'sha256': sha(stderr_path)},
               'failure_file_present': (output / 'failure.json').is_file(), 'run_files': files,
               'scope': 'cooperative direct-child observation; no descendant or authenticated execution attestation'}
    summary_path = output / 'phase-summary.json'
    try:
        summary = json.loads(summary_path.read_bytes())
        receipt['accepted_complete_phase'] = assess_receipt(receipt, summary)
    except (OSError, ValueError, TypeError, KeyError):
        receipt['accepted_complete_phase'] = False
    with receipt_path.open('xb') as file:
        file.write((json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    try:
        print(json.dumps({'event': 'direct_child_reaped', 'phase': args.phase,
                          'receipt_sha256': sha(receipt_path), 'exit_code': observed['exit_code'],
                          'interrupted': observed['interrupted'], 'prefix_empty_after': receipt['prefix_empty_after'],
                          'accepted_complete_phase': receipt['accepted_complete_phase']}), flush=True)
    except OSError:
        pass
    return 0 if receipt['accepted_complete_phase'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
