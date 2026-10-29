"""One-shot cooperative direct-child observer for a natural-transfer phase."""
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
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def observe_child(command, *, cwd, env, stdout, stderr, popen=subprocess.Popen, progress=print):
    process = None
    started = None
    interrupted = False
    error_type = None
    try:
        process = popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        started = datetime.now(timezone.utc).isoformat()
        progress(json.dumps({'event': 'direct_child_started', 'pid': process.pid,
                             'observed_utc': started}), flush=True)
        exit_code = process.wait()
    except BaseException as error:
        if process is None:
            # A lost Popen return cannot establish whether a child was created.
            raise
        interrupted = True
        error_type = type(error).__name__
        if process.poll() is None:
            process.terminate()
        exit_code = process.wait()
    return {'pid': process.pid, 'exit_code': exit_code, 'interrupted': interrupted,
            'launcher_error_type': error_type, 'start_observed_utc': started,
            'reaped_observed_utc': datetime.now(timezone.utc).isoformat()}


def assess_receipt(receipt, terminal):
    if type(receipt) is not dict or type(terminal) is not dict:
        return False
    detail = terminal.get('detail')
    if type(detail) is not dict or type(detail.get('displacement_inventory')) is not dict:
        return False
    inventory = detail['displacement_inventory']
    return (type(receipt.get('exit_code')) is int and receipt['exit_code'] == 0
            and receipt.get('interrupted') is False
            and receipt.get('prefix_empty_after') is True and receipt.get('stage_empty_after') is True
            and type(receipt.get('stderr', {}).get('bytes')) is int
            and receipt['stderr']['bytes'] == 0
            and terminal.get('status') == 'complete_candidate'
            and terminal.get('storage_accounting_complete') is True
            and type(terminal.get('events')) is int and terminal['events'] == 97
            and detail.get('status') == 'complete_candidate'
            and detail.get('phase') == receipt.get('phase')
            and type(detail.get('forwards')) is int and detail['forwards'] == 96
            and inventory.get('complete') is True and type(inventory.get('files')) is list
            and len(inventory['files']) == 96
            and {'started.json', 'events.jsonl', 'terminal.json'} <= set(receipt.get('run_files', {})))


def path_hash(path):
    return hashlib.sha256(os.path.normcase(str(path.resolve(strict=True))).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--phase', choices=('discovery', 'evaluation'), required=True)
    parser.add_argument('--pre-run-sha256', required=True)
    parser.add_argument('--freeze-observation-sha256')
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    pre_path = attempt / 'pre-run.json'
    if sha(pre_path) != args.pre_run_sha256:
        raise ValueError('pre-run digest differs')
    pre = json.loads(pre_path.read_bytes())
    private = json.loads((attempt / 'private-launch.json').read_bytes())
    if pre.get('schema') != 'natural-invariant-transfer-pre-run-v2':
        raise ValueError('unexpected pre-run schema')
    if sha(Path(__file__).resolve()) != pre['direct_launcher_sha256']:
        raise ValueError('direct launcher bytes differ')
    source = Path(private['source_root']).resolve(strict=True)
    study = Path(private['study_root']).resolve(strict=True)
    prefix = Path(private['prefixes'][args.phase])
    stage, temp = Path(private['stage_dir']), Path(private['temp_dir'])
    cache = Path(private['cache_dir']).resolve(strict=True)
    output = attempt / ('run-' + args.phase)
    if (not prefix.is_absolute() or prefix.is_symlink() or not prefix.is_dir() or any(prefix.iterdir())
            or not stage.is_absolute() or stage.is_symlink() or not stage.is_dir() or any(stage.iterdir())
            or not temp.is_absolute() or temp.is_symlink() or not temp.is_dir() or output.exists()):
        raise ValueError('unexpected output, stage, temp or bytecode prefix state')
    if path_hash(prefix) != pre['pycache_prefix_sha256'][args.phase] or path_hash(stage) != pre['stage_dir_sha256']:
        raise ValueError('pinned runtime directory differs')
    if any(os.path.splitdrive(str(x.resolve()))[0].upper() != 'E:' for x in (attempt, prefix, stage, temp)):
        raise ValueError('attempt, stage, temp and bytecode prefix must be on E:')
    if (shutil.disk_usage(attempt).free < 2 * 1024 ** 3 or shutil.disk_usage(stage).free < 2 * 1024 ** 3
            or any(shutil.disk_usage(x).free < 256 * 1024 ** 2 for x in (source, study, cache))):
        raise ValueError('insufficient source, cache, stage or output reserve')
    stdout_path = attempt / (args.phase + '.stdout.txt')
    stderr_path = attempt / (args.phase + '.stderr.txt')
    receipt_path = attempt / (args.phase + '.receipt.json')
    if any(path.exists() for path in (stdout_path, stderr_path, receipt_path)):
        raise ValueError('this one-shot phase has already been started')
    command = [sys.executable, '-B', '-X', f'pycache_prefix={prefix}', str(study / 'natural_transfer_phase.py'),
               '--phase', args.phase, '--attempt', str(attempt), '--pre-run-sha256', args.pre_run_sha256,
               '--output-dir', str(output)]
    exposure_sha = pre['latest_exposure_sha256']
    if args.phase == 'evaluation':
        obs = Path(private['freeze_observation'])
        if not args.freeze_observation_sha256 or sha(obs) != args.freeze_observation_sha256:
            raise ValueError('evaluation requires a pinned public-freeze observation')
        exposure_sha = json.loads(obs.read_bytes())['latest_exposure_sha256']
        command.extend(['--freeze-observation-sha256', args.freeze_observation_sha256])
    elif args.freeze_observation_sha256 is not None:
        raise ValueError('discovery must not consume an evaluation observation')
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'OMP_NUM_THREADS': '1',
                'PYTHONPATH': os.pathsep.join((str(study), str(source))), 'TEMP': str(temp), 'TMP': str(temp)})
    for key in ('MKL_CBWR', 'ATEN_CPU_CAPABILITY'):
        env.pop(key, None)
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        observed = observe_child(command, cwd=source, env=env, stdout=stdout, stderr=stderr)
    files = {}
    if output.is_dir():
        for path in sorted(output.rglob('*')):
            if path.is_symlink():
                raise ValueError('unexpected symlink in closed run')
            if path.is_dir():
                continue
            if not path.is_file() or len(files) >= 4096:
                raise ValueError('unexpected closed run inventory')
            files[path.relative_to(output).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    receipt = {'schema': 'natural-invariant-transfer-direct-receipt-v2', 'phase': args.phase, **observed,
               'pre_run_sha256': args.pre_run_sha256, 'freeze_observation_sha256': args.freeze_observation_sha256,
               'source_revision': pre['source_revision'], 'source_tree': pre['source_tree'],
               'prior_receipt_sha256': pre['prior']['receipt_sha256'],
               'latest_exposure_sha256': exposure_sha,
               'prefix_empty_after': not any(prefix.iterdir()), 'stage_empty_after': not any(stage.iterdir()),
               'free_bytes_after': {'output': shutil.disk_usage(attempt).free, 'source': shutil.disk_usage(source).free,
                                    'cache': shutil.disk_usage(cache).free},
               'stdout': {'bytes': stdout_path.stat().st_size, 'sha256': sha(stdout_path)},
               'stderr': {'bytes': stderr_path.stat().st_size, 'sha256': sha(stderr_path)},
               'run_files': files,
               'scope': 'cooperative direct-child observation; no authenticated execution or descendant attestation'}
    try:
        terminal = json.loads((output / 'terminal.json').read_bytes())
        receipt['accepted_complete_phase'] = assess_receipt(receipt, terminal)
    except (OSError, ValueError, TypeError, KeyError):
        receipt['accepted_complete_phase'] = False
    with receipt_path.open('xb') as stream:
        data = (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode()
        if stream.write(data) != len(data):
            raise OSError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        print(json.dumps({'event': 'direct_child_reaped', 'phase': args.phase,
                          'receipt_sha256': sha(receipt_path), 'exit_code': observed['exit_code'],
                          'accepted_complete_phase': receipt['accepted_complete_phase']}), flush=True)
    except OSError:
        pass
    return 0 if receipt['accepted_complete_phase'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
