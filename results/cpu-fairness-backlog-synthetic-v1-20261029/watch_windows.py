"""Watch two frozen synthetic CPU windows in separate child processes.

This records child exit and source-tree observations; it does not attest to
in-memory execution, dependency binaries, device activity, or isolation.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'teams' / 'cpu_fairness_backlog'
EXPECTED_COMMIT = '9b6fb1a7d7140c9ef997159a7f1bf7360e877c71'
EXPECTED_PLAN = '57e8b0a98f64e7bfb5ec3aa04a97bfb21c14a0af15772227264f033523d82049'
TIMEOUT_SECONDS = 45


def git(*args):
    return subprocess.check_output(['git', *args], cwd=SOURCE, text=True).strip()


def source_state():
    return {'commit': git('rev-parse', 'HEAD'), 'tree': git('show', '-s', '--format=%T', 'HEAD'),
            'clean': git('status', '--porcelain', '--untracked-files=normal') == ''}


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    envelope_path = HERE / 'plan.json'
    plan_bytes = envelope_path.read_bytes()
    envelope = json.loads(plan_bytes)
    canonical = json.dumps(envelope['plan'], sort_keys=True, separators=(',', ':'),
                           ensure_ascii=True, allow_nan=False).encode('utf-8')
    if envelope['plan_sha256'] != EXPECTED_PLAN or sha256(canonical).hexdigest() != EXPECTED_PLAN:
        raise RuntimeError('the retained plan digest does not match the frozen plan')
    initial = source_state()
    if initial['commit'] != EXPECTED_COMMIT or not initial['clean']:
        raise RuntimeError('source is not the expected clean commit')
    for mode in ('baseline', 'backlog'):
        before = source_state()
        if before != initial:
            raise RuntimeError('source changed before the child process started')
        report = HERE / f'{mode}.json'
        receipt_path = HERE / f'{mode}_watchdog.json'
        if any(path.exists() for path in (report, receipt_path, HERE / f'{mode}.stdout',
                                          HERE / f'{mode}.stderr')):
            raise RuntimeError('an output already exists')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '-1'
        command = [sys.executable, '-m', 'looking_glass.cpu_fairness_backlog', 'run',
                   '--plan', str(envelope_path), '--sha256', EXPECTED_PLAN,
                   '--mode', mode, '--output', str(report)]
        started_utc = utc()
        started_ns = time.perf_counter_ns()
        process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
        elapsed_ns = time.perf_counter_ns() - started_ns
        finished_utc = utc()
        (HERE / f'{mode}.stdout').write_bytes(stdout)
        (HERE / f'{mode}.stderr').write_bytes(stderr)
        after = source_state()
        receipt = {'version': 'cpu-fairness-watchdog-v1', 'mode': mode,
                   'started_utc': started_utc, 'finished_utc': finished_utc,
                   'elapsed_watchdog_ns': elapsed_ns, 'timeout_seconds': TIMEOUT_SECONDS,
                   'timed_out': timed_out, 'returncode': process.returncode,
                   'child_reaped': True, 'cuda_visible_devices': '-1',
                   'plan_canonical_sha256': EXPECTED_PLAN, 'plan_file_sha256': sha256(plan_bytes).hexdigest(),
                   'report_sha256': sha256(report.read_bytes()).hexdigest() if report.exists() else None,
                   'stdout_sha256': sha256(stdout).hexdigest(), 'stderr_sha256': sha256(stderr).hexdigest(),
                   'source_before': before, 'source_after': after,
                   'scope': 'Parent observed child exit and on-disk Git state; no execution, isolation or device attestation.'}
        with receipt_path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write('\n')
        print(f"{mode}: exit={process.returncode} timed_out={timed_out} report_sha256={receipt['report_sha256']}", flush=True)
        if timed_out or process.returncode != 0 or after != initial:
            raise RuntimeError('window failed or source changed; preserved its evidence and stopped')


if __name__ == '__main__':
    main()
