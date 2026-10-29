"""Observe one public CPU patch child and retain its direct-process outcome."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INVENTORY_MERGE = '9d9d82b3a53a4977e55250c4202ca207971ada7e'
INVENTORY_SHA = '4a4ed0127d18a7c5a8f6472af619b82895cb4efa6c07314a48a463510b1878b2'
root = Path(__file__).resolve().parents[2]
repo = root / 'work/teams/natural_heldout_custody'
python = root / 'work/.venv/Scripts/python.exe'
cache = root / 'work/hf-cache/hub'
output = root / 'work/public-two-attractor-patch-run-v1'
observer = root / 'work/public-two-attractor-patch-observer-v1'
files = {
    'case': (repo / 'experiments/distilgpt2_two_attractor_patch_cases_v1.json', '6d0cbba21d15f9ab05df106617141c25a460f07be7ee57c53470f969eb79e518'),
    'runner_source': (repo / 'tools/public_two_attractor_patch_v1.py', 'd18fe48c41e9c89e7b549e12ebe157a0d1fa56294aa586e1c101899ae563b1f7'),
    'loader_source': (repo / 'looking_glass/pretrained.py', 'c903a0cbbc83d70b7bbb4b812b637bfbd888fae252c1a38385d1654f4d157b4e'),
    'plan_source': (repo / 'looking_glass/intervention_plan.py', '6470676938e73e48234128c30ba1d10a49104ee8513434871bc6e973c55d6120'),
    'interventions_source': (repo / 'looking_glass/interventions.py', '71cbcf4776660d2dd883b82815f710b0c909228fd1762504f9e3f984f7f6c60f'),
    'ownership_source': (repo / 'looking_glass/model_ownership.py', '99fee108f56ff8b7b7361e39de982c242cff590cc06c5d7a640cc12123cfe434'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


check(len(sys.argv) == 2 and len(sys.argv[1]) == 40 and all(c in '0123456789abcdef' for c in sys.argv[1]), 'expected observed official main commit')
PRELAUNCH_MAIN = sys.argv[1]
main_probe = subprocess.run(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=repo, capture_output=True, timeout=15, check=True)
check(main_probe.stdout.strip() == (PRELAUNCH_MAIN + '\trefs/heads/main').encode(), 'official main changed before launch')

for name, (path, expected) in files.items():
    check(sha(path) == expected, 'unexpected ' + name + ' bytes')
free_work = shutil.disk_usage(root / 'work').free
free_cache = shutil.disk_usage(cache).free
check(min(free_work, free_cache) >= 2 * 1024**3, 'insufficient free space before staging')
check(not output.exists() and not observer.exists(), 'output or observer already exists')
observer.mkdir(parents=True, exist_ok=False)
pycache = observer / 'empty-pycache'
pycache.mkdir(exist_ok=False)
stat = pycache.lstat()
check(not pycache.is_symlink() and not (getattr(stat, 'st_file_attributes', 0) & 0x400), 'pycache is a reparse point')
check(not any(pycache.iterdir()), 'pycache is not empty')
cmd = [str(python), '-B', '-X', 'pycache_prefix=' + str(pycache), '-m',
       'tools.public_two_attractor_patch_v1', '--cases', str(files['case'][0]),
       '--cache', str(cache), '--output', str(output)]
env = os.environ.copy()
for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX', 'PYTHONOPTIMIZE'):
    env.pop(key, None)
env.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
birth = now()
proc = None
pid = None
code = None
wait_returned = False
failure = None
cleanup = 'not_needed'
with (observer / 'stdout.txt').open('xb') as out, (observer / 'stderr.txt').open('xb') as err:
    try:
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        pid = proc.pid
        code = proc.wait()
        wait_returned = True
    except BaseException as exc:
        failure = type(exc).__name__
        if proc is None:
            cleanup = 'spawn_return_uncertain'
        else:
            try:
                if proc.poll() is None:
                    proc.terminate()
                code = proc.wait(timeout=15)
                wait_returned = True
                cleanup = 'direct_child_reaped_after_interruption'
            except BaseException:
                cleanup = 'direct_child_cleanup_uncertain'
finish = now()
pycache_empty = None
if wait_returned:
    try:
        stat = pycache.lstat()
        pycache_empty = (not pycache.is_symlink() and not (getattr(stat, 'st_file_attributes', 0) & 0x400)
                         and not any(pycache.iterdir()))
    except OSError:
        pycache_empty = False
record = {'schema': 'public-two-attractor-patch-process-observation-v1', 'process_id': pid,
          'start_utc': birth, 'end_utc': finish, 'exit_code': code, 'wait_returned': wait_returned,
          'observer_failure': failure, 'direct_child_cleanup': cleanup, 'pycache_empty_after': pycache_empty,
          'protocol_merge': '64d72f4ea3818ea2a33edd9db5073e37417cd5c0',
          'runner_merge': 'afab9f093f68be0636042b9098ec579e167400f5',
          'inventory_merge': INVENTORY_MERGE, 'inventory_sha256': INVENTORY_SHA,
          'prelaunch_official_main': PRELAUNCH_MAIN, 'prelaunch_free_work_bytes': free_work,
          'prelaunch_free_cache_bytes': free_cache,
          'command': 'python -B -X pycache_prefix=<new-empty-dir> -m tools.public_two_attractor_patch_v1 --cases <published-cases> --cache <offline-cache> --output <new-output>',
          'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
              'PYTHONNOUSERSITE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
          'stdout_sha256': sha(observer / 'stdout.txt') if wait_returned else None,
          'stderr_sha256': sha(observer / 'stderr.txt') if wait_returned else None,
          'journal_sha256': sha(output / 'attempt.jsonl') if wait_returned and (output / 'attempt.jsonl').is_file() else None,
          'result_sha256': sha(output / 'result.json') if wait_returned and (output / 'result.json').is_file() else None}
for name, (path, _) in files.items():
    record[name + '_sha256'] = sha(path)
(observer / 'observation.json').write_bytes((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
print(json.dumps({'pid': pid, 'exit_code': code, 'wait_returned': wait_returned, 'failure': failure,
                  'cleanup': cleanup, 'pycache_empty_after': pycache_empty,
                  'journal_sha256': record['journal_sha256'], 'result_sha256': record['result_sha256']}))
if failure is not None or not wait_returned or code != 0 or not pycache_empty:
    raise SystemExit(1)
