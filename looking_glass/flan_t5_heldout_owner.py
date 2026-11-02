"""Bounded private process owner for the locally withheld CPU study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .flan_t5_heldout_response import CUSTODY_LIMIT, REPO, source_identity

FILE_LIMIT = 65536
OWNER_TIMEOUT_SECONDS = 3600
REAP_TIMEOUT_SECONDS = 5


def _json_bytes(value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > FILE_LIMIT:
        raise ValueError('private owner record is oversized')
    return raw


def _write(root, name, raw, reserve=0):
    from .flan_t5_heldout_custody import inventory_custody
    if len(raw) > FILE_LIMIT or inventory_custody(root)['bytes'] + len(raw) + reserve > CUSTODY_LIMIT:
        raise ValueError('private owner evidence exceeds its bound')
    path = root / name
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short private owner write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _capture(stream, target, state, overflow):
    try:
        while True:
            block = stream.read(4096)
            if not block:
                break
            state['observed_bytes'] += len(block)
            target.extend(block[:max(0, FILE_LIMIT - len(target))])
            if state['observed_bytes'] > FILE_LIMIT:
                overflow.set()
    except BaseException as error:
        state['reader_error_type'] = type(error).__name__
        overflow.set()


def _receipt(root, payload):
    """Include the receipt itself in the final, deterministic size inventory."""
    from .flan_t5_heldout_custody import inventory_custody
    before = inventory_custody(root)
    size = 0
    for _ in range(16):
        entries = sorted([*before['entries'], {'path': 'owner-completed.json', 'size': size}],
                         key=lambda entry: entry['path'])
        final = {'bytes': before['bytes'] + size, 'files': before['files'] + 1, 'entries': entries}
        raw = _json_bytes({**payload, 'final_custody_inventory': final})
        if len(raw) == size:
            break
        size = len(raw)
    else:
        raise ValueError('private owner inventory did not converge')
    digest = _write(root, 'owner-completed.json', raw)
    if inventory_custody(root) != final:
        raise ValueError('private owner inventory changed')
    return digest


def _stop_child(child):
    """Try a bounded terminate, then kill, and report uncertainty honestly."""
    report = {'terminate_attempted': False, 'kill_attempted': False, 'reaped': False,
              'error_type': None}
    try:
        if child.poll() is None:
            report['terminate_attempted'] = True
            child.terminate()
        try:
            child.wait(timeout=REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            report['kill_attempted'] = True
            child.kill()
            child.wait(timeout=REAP_TIMEOUT_SECONDS)
        report['reaped'] = True
    except (OSError, subprocess.SubprocessError) as error:
        report['error_type'] = type(error).__name__
    return report


def run_owner(asset_dir, output, custody_root, publication_observation, pool_observation,
              case_observation, auditor_observation):
    """Run once, retaining bounded pipes and the observed child exit privately."""
    from .flan_t5_heldout_custody import inventory_custody
    custody_root = Path(custody_root).resolve(strict=True)
    output = Path(output).resolve(strict=False)
    observations = [Path(p).resolve(strict=True) for p in
                    (publication_observation, pool_observation, case_observation, auditor_observation)]
    if (output.parent != custody_root.parent or output.exists()
            or any(path.parent != custody_root for path in observations)):
        raise ValueError('attempt and custody paths differ from the fixed layout')
    if any((custody_root / name).exists() for name in
           ('owner-started.json', 'owner-completed.json', 'owner.stdout.bin', 'owner.stderr.bin',
            'owner-bytecode-prefix', 'owner-parent-bytecode-prefix')):
        raise ValueError('private owner attempt already exists')
    resolved_asset = Path(asset_dir).resolve(strict=True)
    source = source_identity()
    initial = inventory_custody(custody_root)
    if initial['bytes'] + 4 * FILE_LIMIT > CUSTODY_LIMIT:
        raise ValueError('private owner reserve does not fit')
    started = {'schema': 'mlg-flant5-heldout-owner-v1', 'status': 'started', 'source': source,
               'initial_custody_inventory': initial}
    _write(custody_root, 'owner-started.json', _json_bytes(started), reserve=3 * FILE_LIMIT)
    prefix = custody_root / 'owner-bytecode-prefix'
    outer_prefix = custody_root / 'owner-parent-bytecode-prefix'
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONINSPECT'):
        env.pop(key, None)
    env.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'MKL_CBWR': 'COMPATIBLE', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'})
    buffers = [bytearray(), bytearray()]
    states = [{'observed_bytes': 0}, {'observed_bytes': 0}]
    overflow = threading.Event()
    child = None
    failure = None
    readers = []
    timed_out = False
    stop = {'terminate_attempted': False, 'kill_attempted': False, 'reaped': False, 'error_type': None}
    try:
        outer_prefix.mkdir()
        prefix.mkdir()
        command = [sys.executable, '-B', '-X', f'pycache_prefix={prefix}', '-m',
                   'looking_glass.flan_t5_heldout_response', '--asset-dir', str(resolved_asset),
                   '--output', str(output), '--custody-root', str(custody_root),
                   '--publication-observation', str(observations[0]), '--pool-observation', str(observations[1]),
                   '--case-observation', str(observations[2]), '--auditor-observation', str(observations[3])]
        child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for index, stream in enumerate((child.stdout, child.stderr)):
            thread = threading.Thread(target=_capture, args=(stream, buffers[index], states[index], overflow),
                                      daemon=True)
            thread.start()
            readers.append(thread)
        deadline = time.monotonic() + OWNER_TIMEOUT_SECONDS
        while child.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if overflow.wait(min(.1, remaining)):
                break
        if timed_out or overflow.is_set():
            stop = _stop_child(child)
        else:
            child.wait(timeout=REAP_TIMEOUT_SECONDS)
            stop['reaped'] = True
    except BaseException as error:
        failure = error
        if child is not None:
            stop = _stop_child(child)
    finally:
        for thread in readers:
            thread.join(timeout=5)
    for index, name in enumerate(('owner.stdout.bin', 'owner.stderr.bin')):
        _write(custody_root, name, bytes(buffers[index]), reserve=(2-index) * FILE_LIMIT)
    prefix_empty = prefix.is_dir() and not any(prefix.iterdir())
    outer_prefix_empty = outer_prefix.is_dir() and not any(outer_prefix.iterdir())
    terminal = output / 'completed.json'
    terminal_sha = None
    terminal_error = None
    if terminal.is_file():
        try:
            with terminal.open('rb') as stream:
                raw = stream.read(1024 * 1024 + 1)
            terminal_value = json.loads(raw)
            if (len(raw) <= 1024 * 1024 and type(terminal_value) is dict
                    and terminal_value.get('schema') == 'mlg-flant5-heldout-response-v1'
                    and terminal_value.get('status') == 'complete' and terminal_value.get('source') == source):
                terminal_sha = hashlib.sha256(raw).hexdigest()
        except (OSError, UnicodeError, ValueError) as error:
            terminal_error = type(error).__name__
    try:
        source_unchanged = source_identity() == source
    except (OSError, subprocess.SubprocessError, ValueError):
        source_unchanged = False
    complete = (failure is None and child is not None and child.returncode == 0 and terminal_sha is not None
                and stop['reaped'] and not timed_out and not overflow.is_set() and not any(t.is_alive() for t in readers)
                and prefix_empty and outer_prefix_empty and source_unchanged)
    payload = {'schema': 'mlg-flant5-heldout-owner-v1', 'status': 'complete' if complete else 'incomplete',
               'child_exit_code': None if child is None else child.returncode, 'stream_states': states,
               'timed_out': timed_out, 'stop': stop, 'owner_timeout_seconds': OWNER_TIMEOUT_SECONDS,
               'streams_exceeded_bound': overflow.is_set(), 'reader_threads_finished': not any(t.is_alive() for t in readers),
               'bytecode_prefix_empty': prefix_empty, 'owner_bytecode_prefix_empty': outer_prefix_empty,
               'source_unchanged': source_unchanged,
               'terminal_sha256': terminal_sha, 'terminal_error_type': terminal_error,
               'error_type': None if failure is None else type(failure).__name__}
    receipt_sha = _receipt(custody_root, payload)
    if failure is not None:
        raise failure
    return {'status': payload['status'], 'owner_receipt_sha256': receipt_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('asset-dir', 'output', 'custody-root', 'publication-observation', 'pool-observation',
                     'case-observation', 'auditor-observation'):
        parser.add_argument('--' + argument, required=True, type=Path)
    args = parser.parse_args()
    try:
        expected_prefix = args.custody_root.resolve(strict=True) / 'owner-parent-bytecode-prefix'
        if (not sys.dont_write_bytecode or sys.pycache_prefix is None
                or Path(sys.pycache_prefix).resolve(strict=False) != expected_prefix or expected_prefix.exists()):
            raise ValueError('fresh private owner bytecode prefix is required')
        result = run_owner(args.asset_dir, args.output, args.custody_root, args.publication_observation,
                           args.pool_observation, args.case_observation, args.auditor_observation)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result['status'] == 'complete' else 1
    except BaseException as error:
        try:
            root = args.custody_root.resolve(strict=True)
            if not (root / 'owner-started.json').exists():
                _write(root, 'owner-preflight-failure.json', _json_bytes({
                    'schema': 'mlg-flant5-heldout-owner-v1', 'status': 'incomplete',
                    'phase': 'preflight', 'error_type': type(error).__name__}))
        except BaseException:
            pass
        print(json.dumps({'status': 'incomplete', 'error_type': type(error).__name__}), flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
