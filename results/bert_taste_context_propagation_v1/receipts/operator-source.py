"""One-shot operator for the reviewed public CPU study; retains partial evidence."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

workspace = Path(__file__).resolve().parent.parent
root = workspace / 'work/teams/bert_taste_propagation_selection'
work = workspace / 'work'
prefix = work / 'bert-propagation-study-pyc-v1'
output = work / 'bert-propagation-study-attempt-v1'
start_path = work / 'bert-propagation-study-operator-start-v1.json'
spawn_path = work / 'bert-propagation-study-operator-spawn-v1.json'
exit_path = work / 'bert-propagation-study-operator-exit-v1.json'
stdout_path = work / 'bert-propagation-study-stdout-v1.log'
stderr_path = work / 'bert-propagation-study-stderr-v1.log'
cache = work / 'hf-cache/hub'
commit = 'a27b34533c3da5f34a541ef2b8b57631bf42e819'
case_path = 'experiments/bert_taste_propagation_public_cases_v1.json'
case_sha = '341300932b0dab6d4004c3dd7007d73500162f8d945a91ca50538b9e2802c979'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('operator record short write')
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


if any(path.exists() for path in (prefix, output, start_path, spawn_path, exit_path, stdout_path, stderr_path)):
    raise RuntimeError('an existing attempt or operator record needs reconciliation')
head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
if head != commit or status or sha(root / case_path) != case_sha:
    raise RuntimeError('public source, clean checkout or selected cases differ')
if shutil.disk_usage(cache).free < 1024 ** 3:
    raise RuntimeError('less than 1 GiB is free on the checkpoint staging volume')

# Read the runner's fixed path tuple without importing any project or model code.
runner = root / 'scripts/bert_taste_propagation_v1.py'
tree = ast.parse(runner.read_bytes(), filename=str(runner))
assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
               for target in node.targets if isinstance(target, ast.Name)}
nodes = assignments['SOURCE_PATHS'].elts
names = [case_path if isinstance(node, ast.Name) and node.id == 'CASES_PATH'
         else ast.literal_eval(node) for node in nodes]
if case_path not in names or len(names) != len(set(names)):
    raise RuntimeError('source inventory is malformed')
source_hashes = {}
for name in names:
    raw = (root / name).read_bytes()
    expected = subprocess.check_output(['git', '-C', str(root), 'show', commit + ':' + name])
    if raw != expected:
        raise RuntimeError('source bytes differ from the published commit')
    source_hashes[name] = hashlib.sha256(raw).hexdigest()

prefix.mkdir(exist_ok=False)
if any(prefix.iterdir()) or prefix.is_symlink():
    raise RuntimeError('fresh prefix is not empty')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_CBWR='AUTO', HF_HUB_OFFLINE='1',
           TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
env.pop('ATEN_CPU_CAPABILITY', None)
args = [sys.executable, '-I', '-B', '-X', 'pycache_prefix=' + str(prefix), str(runner),
        '--cache-dir', str(cache), '--output', str(output)]
write_new(start_path, {
    'schema': 'bert-propagation-operator-start-v1',
    'scope': 'Operator observation; trusted interpreter/site/dependencies and no whole-tree attestation.',
    'source_commit': commit, 'selected_source_sha256': source_hashes, 'cases_sha256': case_sha,
    'operator_clock_utc': now(), 'prefix_empty_before': True, 'cuda_visible_devices': '-1',
    'omp_num_threads': '1', 'mkl_num_threads': '1', 'mkl_cbwr': 'AUTO',
    'offline_model_assets': True, 'staging_free_bytes': shutil.disk_usage(cache).free,
    'invocation': ['python', '-I', '-B', '-X', 'pycache_prefix=<fresh-empty-directory>',
                   'scripts/bert_taste_propagation_v1.py', '--cache-dir', '<offline-cache>',
                   '--output', '<new-attempt-directory>'],
})

child = None
exit_code = None
timed_out = False
error_type = None
try:
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        child = subprocess.Popen(args, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr)
        write_new(spawn_path, {'schema': 'bert-propagation-operator-spawn-v1',
                               'process_id': child.pid, 'operator_clock_utc': now(),
                               'source_commit': commit})
        try:
            exit_code = child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            exit_code = child.wait(timeout=15)
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
except BaseException as exc:
    error_type = type(exc).__name__
    if child is not None and child.poll() is None:
        try:
            child.kill()
            exit_code = child.wait(timeout=15)
        except BaseException:
            pass
    raise
finally:
    result_path = output / 'result.json'
    result_hash = sha(result_path) if result_path.is_file() else None
    result_consistent = False
    try:
        if result_hash is not None and not (output / 'incomplete.json').exists():
            result = json.loads(result_path.read_bytes())
            journals = result['journals']
            result_consistent = (result.get('status') == 'complete'
                and result.get('completed_forwards') == 960
                and result.get('source', {}).get('commit') == commit
                and result.get('source', {}).get('selected_source_sha256') == source_hashes
                and len(journals) == 6
                and all((output / name).stat().st_size == item['size']
                        and sha(output / name) == item['sha256'] for name, item in journals.items()))
    except (OSError, KeyError, TypeError, ValueError):
        result_consistent = False
    write_new(exit_path, {
        'schema': 'bert-propagation-operator-exit-v1',
        'scope': 'Direct-child process observation and saved-file consistency only; no whole-tree or execution attestation.',
        'source_commit': commit, 'cases_sha256': case_sha,
        'operator_clock_utc': now(), 'process_id': None if child is None else child.pid,
        'direct_child_exit_code': exit_code, 'direct_child_reaped': exit_code is not None,
        'timed_out': timed_out, 'operator_exception_type': error_type,
        'prefix_empty_after': prefix.is_dir() and not any(prefix.iterdir()),
        'result_sha256': result_hash, 'saved_result_and_journal_hashes_consistent': result_consistent,
        'incomplete_marker_present': (output / 'incomplete.json').exists(),
        'stdout_sha256': sha(stdout_path) if stdout_path.is_file() else None,
        'stderr_sha256': sha(stderr_path) if stderr_path.is_file() else None,
    })
if exit_code != 0 or timed_out or not result_consistent or any(prefix.iterdir()):
    raise RuntimeError('the attempt is incomplete or uncertain; preserve all evidence')
print(json.dumps({'direct_child_exit_code': exit_code, 'result_sha256': result_hash,
                  'operator_exit_sha256': sha(exit_path), 'source_commit': commit}))
