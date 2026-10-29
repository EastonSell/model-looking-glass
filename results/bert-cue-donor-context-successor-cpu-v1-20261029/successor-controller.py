"""Preflight and directly wait for one distinct donor-context successor on CPU."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from types import ModuleType

SOURCE = 'b6667ad5bfcadac6e0f45d4ad6530cca1f4584ca'
MANIFEST = 'experiments/bert_cue_donor_context_retention_cases_v1.json'
MANIFEST_SHA = '917bad800f72a9613e69d8feae3faf90ffbae5cc446bbeb3e2c8df1d03678dc9'
GENERATOR = 'scripts/build_bert_cue_donor_context_cases.py'
PRIOR_PREFLIGHT = 'c91419bb2c379b24deefe1bb3a5da3378ebd0a484933377e6558a9e9fb62c6f1'
PRIOR_EXIT = 'fef33220fa71a2c8a31c8f3c35373cfaa70483256536c9d5549f6a7983d1155c'
PRIOR_STDERR = '46962f3641d666a7e719f2ea895c50e6c5cc3291524642dd831c8078c81bb39b'
FULL_READER = '1a7994e2405ea03d8c1e4607f4a00c2ca9640ee58aaf310feac932eede5dacc5'
SCORE_READER = '6670787ce5369278a159d71a7efb1c5eef63f7c8650acda16c4343fa294fdd49'
ATTEMPT = 'bert-cue-donor-context-successor-b6667ad5-v1'
FIXED_WORK = Path('E:/Codex/model-looking-glass/work')


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def volumes(repo, snapshot, python, git, scratch):
    paths = {'source': repo, 'checkpoint': snapshot, 'python': python, 'git': git,
             'python_prefix': Path(sys.prefix), 'base_python': Path(sys.base_prefix),
             'scratch': scratch, 'system': Path(os.environ['SYSTEMROOT'])}
    anchors = {key: str(path.resolve().anchor).lower() for key, path in paths.items()}
    if any(not value for value in anchors.values()):
        raise RuntimeError('missing resolved volume anchor')
    if any(anchors[key] != anchors['source'] for key in paths if key != 'scratch'):
        raise RuntimeError('source, runtime, checkpoint, git and system must resolve to the same volume')
    if anchors['source'] == anchors['scratch']:
        raise RuntimeError('scratch must resolve to a distinct drive anchor')
    return {'source_free_bytes': shutil.disk_usage(repo.anchor).free,
            'scratch_free_bytes': shutil.disk_usage(scratch).free, 'anchors': anchors}


def require_launch_space(check):
    if check['source_free_bytes'] < 1024 ** 3 or check['scratch_free_bytes'] < 2 * 1024 ** 3:
        raise RuntimeError('insufficient point-in-time source or scratch volume headroom')


def fixed_source(repo, git, expected):
    if type(expected) is not dict or len(expected) != 19 or expected.get(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('predecessor source pins differ')
    allowed = ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT')
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_COUNT='0',
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    def command(*args):
        return subprocess.run([str(git), '--no-replace-objects', *args], cwd=repo, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30).stdout
    if command('rev-parse', 'HEAD').decode('ascii').strip() != SOURCE:
        raise RuntimeError('checkout HEAD differs')
    if command('status', '--porcelain', '--untracked-files=all').strip():
        raise RuntimeError('checkout is not clean')
    checked = {}
    for name, sha in expected.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or type(sha) is not str:
            raise RuntimeError('invalid source pin')
        local = (repo / relative).read_bytes()
        saved = command('show', f'{SOURCE}:{name}')
        if local != saved or digest_bytes(local) != sha:
            raise RuntimeError(f'committed or local source differs: {name}')
        checked[name] = local
    return checked


def verify_tokenizer(repo, snapshot, checked):
    from tokenizers import Tokenizer
    manifest = json.loads(checked[MANIFEST])
    module = ModuleType('fixed_donor_context_cases')
    module.__file__ = str(repo / GENERATOR)
    exec(compile(checked[GENERATOR], module.__file__, 'exec'), module.__dict__)
    actual = module.build(Tokenizer.from_file(str(snapshot / 'tokenizer.json')))
    if actual != manifest:
        raise RuntimeError('tokenized fixed manifest differs')
    return {'pairs': len(manifest['pairs']), 'contexts': 8, 'rows': 32,
            'any_id': manifest['scored_tokens']['any'], 'some_id': manifest['scored_tokens']['some']}


def reap_after_failed_wait(process, outcome):
    try:
        if process.poll() is None:
            process.terminate()
            outcome['termination_requested'] = True
    except BaseException as exc:
        outcome['termination_error_type'] = type(exc).__name__
        outcome['termination_error_message'] = str(exc)
    try:
        try:
            outcome['exit_code'] = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            outcome['kill_requested'] = True
            outcome['exit_code'] = process.wait(timeout=30)
        outcome['process_waited'] = True
    except BaseException as exc:
        outcome['cleanup_wait_error_type'] = type(exc).__name__
        outcome['cleanup_wait_error_message'] = str(exc)


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'snapshot', 'git', 'work', 'predecessor-control', 'full-reader', 'score-reader'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-controller-sha256', required=True)
    args = parser.parse_args()
    repo, snapshot, git, work = (p.resolve() for p in (args.repo, args.snapshot, args.git, args.work))
    predecessor = args.predecessor_control.resolve()
    python = Path(sys.executable).resolve()
    if work != FIXED_WORK.resolve() or predecessor != (work / 'bert-cue-donor-context-control-b6667ad5').resolve():
        raise RuntimeError('predecessor work or control path differs')
    if digest(Path(__file__).resolve()) != args.expected_controller_sha256:
        raise RuntimeError('controller bytes differ from the supplied public identity')
    if digest(predecessor / 'exit.json') != PRIOR_EXIT or digest(predecessor / 'stderr.txt') != PRIOR_STDERR:
        raise RuntimeError('predecessor evidence differs')
    old_preflight_path = work / 'bert-cue-donor-context-preflight-b6667ad5.json'
    if digest(old_preflight_path) != PRIOR_PREFLIGHT:
        raise RuntimeError('predecessor preflight differs')
    previous = json.loads((predecessor / 'exit.json').read_bytes())
    old_preflight = json.loads(old_preflight_path.read_bytes())
    if (previous.get('source_commit') != SOURCE or previous.get('process_waited') is not True or
            previous.get('exit_code') != 1 or previous.get('preflight_sha256') != PRIOR_PREFLIGHT or
            old_preflight.get('source_commit') != SOURCE or old_preflight.get('status') != 'pass'):
        raise RuntimeError('predecessor receipt differs')
    if (work / 'bert-cue-donor-context-run-b6667ad5').exists():
        raise RuntimeError('predecessor output exists; review exposure before continuation')
    if digest(args.full_reader) != FULL_READER or digest(args.score_reader) != SCORE_READER:
        raise RuntimeError('fixed independent reader bytes differ')
    scratch = work / ATTEMPT
    scratch.mkdir(parents=False, exist_ok=False)
    control = scratch / 'control'
    control.mkdir()
    output = scratch / 'run'
    start = now()
    record(control / 'started.json', {'attempt': ATTEMPT, 'controller_sha256': args.expected_controller_sha256,
           'source_commit': SOURCE, 'predecessor_exit_sha256': PRIOR_EXIT, 'predecessor_stderr_sha256': PRIOR_STDERR,
           'predecessor_preflight_sha256': PRIOR_PREFLIGHT, 'full_reader_sha256': FULL_READER,
           'score_reader_sha256': SCORE_READER, 'controller_start_utc': start, 'process_waited': False})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1', TEMP=str(scratch), TMP=str(scratch),
               HF_HOME=str(scratch / 'hf-cache'), HF_HUB_CACHE=str(scratch / 'hf-cache' / 'hub'),
               TORCH_HOME=str(scratch / 'torch-cache'), XDG_CACHE_HOME=str(scratch / 'xdg-cache'))
    os.environ.update(env)
    outcome = {'attempt': ATTEMPT, 'source_commit': SOURCE, 'controller_start_utc': start,
               'process_started': False, 'process_waited': False, 'exit_code': None}
    interruption = None
    try:
        if 'torch' in sys.modules or 'transformers' in sys.modules:
            raise RuntimeError('preflight imported a model runtime')
        initial = volumes(repo, snapshot, python, git, scratch)
        require_launch_space(initial)
        checked = fixed_source(repo, git, old_preflight['source_sha256'])
        assets = old_preflight['assets']
        for name, pin in assets.items():
            if Path(name).name != name or (snapshot / name).stat().st_size != pin['bytes'] or digest(snapshot / name) != pin['sha256']:
                raise RuntimeError(f'checkpoint asset differs: {name}')
        versions = {name: importlib.metadata.version(name) for name in old_preflight['versions']}
        if versions != old_preflight['versions']:
            raise RuntimeError('runtime versions differ')
        tokenizer = verify_tokenizer(repo, snapshot, checked)
        if 'torch' in sys.modules or 'transformers' in sys.modules:
            raise RuntimeError('preflight imported a model runtime')
        final = volumes(repo, snapshot, python, git, scratch)
        require_launch_space(final)
        preflight = {'status': 'pass', 'attempt': ATTEMPT, 'source_commit': SOURCE,
                     'source_sha256': old_preflight['source_sha256'], 'assets': assets, 'versions': versions,
                     'tokenizer': tokenizer, 'initial_volumes': initial, 'final_volumes': final,
                     'torch_imported': False, 'transformers_imported': False,
                     'controller_sha256': args.expected_controller_sha256,
                     'predecessor_exit_sha256': PRIOR_EXIT, 'created_utc': now()}
        record(control / 'preflight.json', preflight)
        outcome['preflight_sha256'] = digest(control / 'preflight.json')
        args_owner = [str(python), '-I', '-B', str(repo / 'scripts/bert_cue_donor_context_retention_v1.py'),
                      '--snapshot', str(snapshot), '--output', str(output), '--source-commit', SOURCE,
                      '--git-executable', str(git)]
        with (control / 'stdout.txt').open('xb') as stdout, (control / 'stderr.txt').open('xb') as stderr:
            process = None
            try:
                process = subprocess.Popen(args_owner, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
                outcome['process_started'] = True
                outcome['exit_code'] = process.wait()
                outcome['process_waited'] = True
            except BaseException as exc:
                outcome['wait_error_type'] = type(exc).__name__
                outcome['wait_error_message'] = str(exc)
                if process is not None:
                    reap_after_failed_wait(process, outcome)
                if not isinstance(exc, Exception):
                    interruption = exc
            finally:
                stdout.flush(); os.fsync(stdout.fileno())
                stderr.flush(); os.fsync(stderr.fileno())
        outcome['stdout_sha256'] = digest(control / 'stdout.txt')
        outcome['stderr_sha256'] = digest(control / 'stderr.txt')
    except BaseException as exc:
        outcome['error_type'] = type(exc).__name__
        outcome['error_message'] = str(exc)
        if not isinstance(exc, Exception):
            interruption = exc
    outcome['controller_end_utc'] = now()
    record(control / 'exit.json', outcome)
    print(json.dumps(outcome, sort_keys=True), flush=True)
    if interruption is not None:
        raise interruption
    return 0 if (outcome['process_waited'] and outcome['exit_code'] == 0 and 'wait_error_type' not in outcome and 'error_type' not in outcome and 'stdout_sha256' in outcome and 'stderr_sha256' in outcome) else 1


if __name__ == '__main__':
    sys.exit(main())
