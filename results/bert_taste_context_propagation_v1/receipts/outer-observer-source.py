"""Observe one CPU-study operator exit; preserve every partial attempt."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

work = Path(__file__).resolve().parent
operator = work / 'run_bert_propagation_cpu_v1.py'
expected_operator = '7c5e90c8706776e02210928fdc34853938fdc6ebe8086827eb0bacb412d59ebf'
operator_exit = work / 'bert-propagation-study-operator-exit-v1.json'
record = work / 'bert-propagation-study-outer-observation-v1.json'
stdout_path = work / 'bert-propagation-study-outer-stdout-v1.log'
stderr_path = work / 'bert-propagation-study-outer-stderr-v1.log'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


if any(path.exists() for path in (record, stdout_path, stderr_path, operator_exit)):
    raise RuntimeError('existing observation or attempt requires reconciliation')
if digest(operator) != expected_operator:
    raise RuntimeError('operator source differs from the reviewed file')

started = now()
child = None
code = None
timed_out = False
error = None
try:
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        child = subprocess.Popen([sys.executable, '-I', '-B', str(operator)],
                                 cwd=work.parent, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr)
        try:
            code = child.wait(timeout=3900)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait(timeout=15)
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
except BaseException as exc:
    error = type(exc).__name__
    if child is not None and child.poll() is None:
        try:
            child.kill()
            code = child.wait(timeout=15)
        except BaseException:
            pass
    raise
finally:
    exit_sha = digest(operator_exit) if operator_exit.is_file() else None
    value = {
        'schema': 'bert-propagation-outer-operator-observation-v1',
        'scope': 'Direct operator-process observation and saved receipt hash; no execution or whole-tree attestation.',
        'source_commit': 'a27b34533c3da5f34a541ef2b8b57631bf42e819',
        'operator_sha256': expected_operator,
        'operator_process_id': None if child is None else child.pid,
        'operator_started_at_utc_clock': started,
        'operator_finished_at_utc_clock': now(),
        'operator_exit_code': code,
        'operator_reaped': code is not None,
        'timed_out': timed_out,
        'observer_exception_type': error,
        'operator_exit_record_sha256': exit_sha,
        'stdout_sha256': digest(stdout_path) if stdout_path.is_file() else None,
        'stderr_sha256': digest(stderr_path) if stderr_path.is_file() else None,
    }
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with record.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('outer observation short write')
        stream.flush()
        os.fsync(stream.fileno())
if code != 0 or timed_out or exit_sha is None:
    raise RuntimeError('operator completion is uncertain; retain the attempt')
print(json.dumps({'operator_exit_code': code, 'operator_exit_record_sha256': exit_sha,
                  'outer_observation_sha256': digest(record)}))
