"""Observe one direct CPU study child and write a scoped exit receipt."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def available_physical_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD),
                    ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                    ('avail_extended_virtual', ctypes.c_ulonglong)]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if os.name != 'nt' or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise RuntimeError('unable to observe available Windows physical memory')
    return value.avail_phys


def save_json(path, obj):
    data = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def observe_child(command, *, controller, attempt, source_commit, environment, timeout):
    """Record direct-child exit; missing receipts are not successful exits."""
    process = None
    exit_code = None
    interrupted = None
    start = time.time()
    with (controller / 'child.log').open('xb') as log:
        try:
            process = subprocess.Popen(command, cwd=controller, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            save_json(controller / 'spawn.json', {'pid': process.pid, 'start_unix': start,
                                                  'source_commit': source_commit})
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                interrupted = 'timeout'
                process.terminate()
                exit_code = process.wait(timeout=15)
        except BaseException as error:
            interrupted = type(error).__name__
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    exit_code = process.wait(timeout=15)
                except BaseException:
                    exit_code = None
        finally:
            log.flush()
            os.fsync(log.fileno())
    receipt = {'schema': 'cross-model-direct-child-exit-v1', 'child_pid': process.pid if process else None,
               'exit_code': exit_code, 'interrupted': interrupted, 'source_commit': source_commit,
               'attempt_name': attempt.name, 'start_unix': start, 'end_unix': time.time(),
               'status': 'incomplete'}
    complete_path, metadata_path = attempt / 'complete.json', attempt / 'metadata.json'
    if (process is not None and exit_code == 0 and interrupted is None and complete_path.is_file()
            and metadata_path.is_file() and not (attempt / 'failure.txt').exists()):
        completed = json.loads(complete_path.read_bytes())
        metadata = json.loads(metadata_path.read_bytes())
        if (completed.get('status') == 'complete' and completed.get('events') == 192
                and metadata.get('pid') == process.pid and metadata.get('source_commit') == source_commit
                and completed.get('metadata_sha256') == digest(metadata_path)):
            receipt.update({'status': 'complete', 'complete_sha256': digest(complete_path),
                            'metadata_sha256': digest(metadata_path)})
    save_json(controller / 'exit-observation.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--site', required=True, type=Path)
    parser.add_argument('--model', required=True, choices=('gpt', 'bert'))
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--controller', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if (Path(sys.executable).resolve() != Path(sys._base_executable).resolve()
            or not re.fullmatch('[0-9a-f]{40}', args.source_commit) or not 1 <= args.timeout <= 3600):
        raise ValueError('base interpreter, source SHA or timeout differs')
    root = args.repo.resolve(strict=True)
    for name in ('scripts/observe_cross_model_agreement_v1.py', 'scripts/cross_model_agreement_v1.py',
                 'docs/CROSS_MODEL_AGREEMENT_V1.md', 'experiments/cross_model_agreement_cases_v1.json'):
        committed = subprocess.run(['git', '-C', str(root), 'show', args.source_commit + ':' + name],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout
        if (root / name).read_bytes() != committed:
            raise ValueError(f'observer or study source differs: {name}')
    if (not args.attempt.is_absolute() or args.attempt.exists() or not args.attempt.parent.is_dir()
            or not args.controller.is_absolute() or args.controller.exists() or not args.controller.parent.is_dir()):
        raise ValueError('attempt and controller must be new absolute paths with existing parents')
    # 128 MiB bounded study output, plus 384 MiB slack and a 2 GiB volume cushion.
    disk_minimum = 2684354560
    attempt_free = shutil.disk_usage(args.attempt.parent).free
    controller_free = shutil.disk_usage(args.controller.parent).free
    host_free = available_physical_memory()
    if min(attempt_free, controller_free) < disk_minimum or host_free < 3 * 1024**3:
        raise RuntimeError('insufficient observed host or output-volume headroom')
    args.controller.mkdir()
    prefix = args.controller / 'empty-pycache'
    prefix.mkdir()
    temp = args.controller / 'temp'
    temp.mkdir()
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT') if key in os.environ}
    system = Path(env.get('SYSTEMROOT', 'C:/Windows'))
    git = shutil.which('git')
    if not git:
        raise RuntimeError('Git is required for source verification')
    env.update({'PATH': os.pathsep.join((str(Path(git).resolve().parent), str(system / 'System32'))),
                'TEMP': str(temp), 'TMP': str(temp),
                'HF_HOME': str(args.controller / 'hf'), 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    command = [str(Path(sys._base_executable).resolve()), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(root / 'scripts/cross_model_agreement_v1.py'), '--site', str(args.site.resolve()),
               '--model', args.model, '--model-dir', str(args.model_dir.resolve()), '--cases', str(args.cases.resolve()),
               '--output', str(args.attempt), '--device', 'cpu', '--source-commit', args.source_commit]
    save_json(args.controller / 'admission.json', {'attempt_free_bytes': attempt_free,
              'controller_free_bytes': controller_free, 'available_physical_bytes': host_free,
              'disk_minimum_bytes': disk_minimum, 'host_minimum_bytes': 3 * 1024**3,
              'scope': 'point-in-time observations, not an OS reservation or peak-memory guarantee'})
    receipt = observe_child(command, controller=args.controller, attempt=args.attempt,
                            source_commit=args.source_commit, environment=env, timeout=args.timeout)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if receipt['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
