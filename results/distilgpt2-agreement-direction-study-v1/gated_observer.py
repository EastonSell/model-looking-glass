"""Observe a single agreement study through the reviewed gated CPU supervisor.

This is cooperative direct-child custody, not authenticated execution or a
whole-process-tree guarantee. The operator must not retry an uncertain attempt.
"""
import argparse
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _load_support():
    path = Path(__file__).with_name('observer_support.py')
    spec = importlib.util.spec_from_file_location('_agreement_observer_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = _load_support()
SUPERVISION_PATHS = frozenset(('looking_glass/__init__.py', 'looking_glass/supervisor.py',
                               'looking_glass/cpu_study_supervision.py', 'looking_glass/device_lease.py',
                               'looking_glass/device_identity.py'))


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, origin, raw):
        self.origin, self.raw = origin, raw

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.raw, self.origin, 'exec', dont_inherit=True), module.__dict__)


class _PackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, sources):
        self.root, self.sources = root, sources

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'looking_glass' and not fullname.startswith('looking_glass.'):
            return None
        name = fullname.replace('.', '/')
        package, module = name + '/__init__.py', name + '.py'
        selected = package if package in self.sources else module
        if selected not in self.sources:
            raise ModuleNotFoundError('module is outside the bound supervision source')
        loader = _SourceLoader(str(self.root / selected), self.sources[selected])
        return importlib.util.spec_from_loader(fullname, loader, is_package=selected == package)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'base-python', 'venv', 'study-root', 'prelaunch',
                 'prelaunch-sha256', 'entry', 'argv-json', 'attempt'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args(argv)
    repo = Path(args.repository).resolve(strict=True)
    python = Path(args.base_python).resolve(strict=True)
    venv = Path(args.venv).resolve(strict=True)
    study_root = Path(args.study_root).resolve(strict=True)
    attempt = Path(args.attempt).resolve()
    if (not repo.is_dir() or not python.is_file() or not (venv / 'Lib/site-packages').is_dir()
            or not study_root.is_dir() or not attempt.parent.is_dir() or attempt.exists()
            or attempt.is_relative_to(repo) or study_root.is_relative_to(repo)):
        raise ValueError('source, interpreter, study root or fresh attempt path is invalid')
    if not 1 <= args.timeout_seconds <= 3600 or shutil.disk_usage(attempt.parent).free < 2 * 1024**3:
        raise ValueError('deadline or free-disk floor is invalid')
    pre_raw = support.bounded_read(Path(args.prelaunch), 256 * 1024)
    if support.sha(pre_raw) != args.prelaunch_sha256:
        raise ValueError('prelaunch bytes differ')
    pre = json.loads(pre_raw)
    owner, runtime, observer = pre['owner'], pre['runtime'], pre['observer']
    if (support.sha(support.bounded_read(Path(__file__), 1024 * 1024)) != observer['sha256']
            or support.sha(support.bounded_read(Path(__file__).with_name('observer_support.py'), 1024 * 1024))
            != observer['support_sha256']
            or support.sha(support.bounded_read(python, 16 * 1024 * 1024)) != runtime['base_python_sha256']):
        raise ValueError('observer or interpreter differs from prelaunch')
    if args.entry != owner['entry_path']:
        raise ValueError('entry path differs from prelaunch')
    git_env = {key: val for key, val in os.environ.items() if not key.upper().startswith('GIT_')}
    git_env['GIT_NO_REPLACE_OBJECTS'] = '1'
    entry = (repo / args.entry).resolve()
    head, tree = support.source_check(repo, owner, entry, git_env)
    if type(owner.get('supervision_sha256')) is not dict or set(owner['supervision_sha256']) != SUPERVISION_PATHS:
        raise ValueError('supervision source list differs')
    supervision_sources = {}
    for relative, digest in owner['supervision_sha256'].items():
        path = (repo / relative).resolve(strict=True)
        raw = support.bounded_read(path, 1024 * 1024)
        committed = subprocess.check_output(['git', 'show', head + ':' + relative],
                                            cwd=repo, env=git_env, timeout=60)
        if (not path.is_relative_to(repo) or not path.is_file() or len(committed) > 1024 * 1024
                or support.sha(raw) != digest or committed != raw):
            raise ValueError('supervision source differs from prelaunch')
        supervision_sources[relative] = raw
    argument_bytes = support.bounded_read(Path(args.argv_json), 64 * 1024)
    inputs = json.loads(argument_bytes)
    if (type(inputs) is not list or any(type(item) is not str or len(item) > 8192 for item in inputs)
            or ('--validate-only' in inputs) != args.validation_only
            or any(item in ('--study-root', '--venv', '--child-custody') for item in inputs)):
        raise ValueError('argument vector or validation mode differs')
    inherited = support.inherited_environment_hashes()
    if (runtime['environment'] != support.FIXED_ENVIRONMENT or runtime.get('claim_cuda_visibility') != ''
            or runtime['inherited_environment_sha256'] != inherited):
        raise ValueError('recorded environment differs')
    environment = {key: os.environ[key] for key in support.INHERITED_NAMES if key in os.environ}
    for key, value in runtime['environment'].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    environment['PYTHONNOUSERSITE'] = '1'
    attempt.mkdir(exist_ok=False)
    support.write_new(attempt / 'pre-run.json', pre_raw)
    prefix = attempt / 'empty-pycache'
    prefix.mkdir()
    replacements = {'{run}': str(attempt / 'run'), '{prelaunch}': str(attempt / 'pre-run.json'),
                    '{prelaunch_sha256}': args.prelaunch_sha256}
    child_args = [replacements.get(item, item) for item in inputs]
    child_args += ['--study-root', str(study_root), '--venv', str(venv),
                   '--child-custody', str(attempt / 'child-custody.json')]
    command = [str(python), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), str(entry), *child_args]
    # The prechecked source supplies the existing rooted-child launch gate.
    if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
        raise ValueError('project package was imported before source validation')
    sys.meta_path.insert(0, _PackageFinder(repo, supervision_sources))
    from looking_glass import supervisor
    if supervisor.SUPERVISOR_LAUNCH_CUSTODY_CONTRACT != 'gated-rooted-child-v1':
        raise ValueError('reviewed gated supervisor contract unavailable')
    if supervisor.pending_supervised_children():
        raise ValueError('unexpected pending supervised child')

    started, ended = support.utc(), None
    result, pending, error_type, recovery, post_errors = None, None, None, [], []

    def note_error(error, stage):
        nonlocal pending, error_type
        post_errors.append({'stage': stage, 'error_type': type(error).__name__})
        if isinstance(error, (KeyboardInterrupt, SystemExit)) and not isinstance(pending, (KeyboardInterrupt, SystemExit)):
            pending, error_type = error, type(error).__name__

    try:
        result = supervisor.run_supervised(
            command, device_lease_path=study_root / 'cpu-study.lock',
            marker_path=study_root / 'cpu-study.unsafe.json', identity=None,
            timeout=args.timeout_seconds, lease_timeout=30, terminate_grace=5,
            cwd=repo, env=environment, inherit_environment=False, require_zero_exit=True)
    except BaseException as error:
        pending, error_type = error, type(error).__name__
    finally:
        ended = support.utc()
        # The call is no longer running here, so exact registered recovery is safe.
        try:
            for child in supervisor.pending_supervised_children():
                try:
                    recovered = child.recover(terminate_grace=5)
                    recovery.append({'run_id': child.run_id, 'exit_code': recovered, 'status': 'recovered'})
                except BaseException as error:
                    recovery.append({'run_id': child.run_id, 'status': 'uncertain',
                                     'error_type': type(error).__name__})
                    note_error(error, 'recovery')
        except BaseException as error:
            recovery.append({'status': 'enumeration_uncertain', 'error_type': type(error).__name__})
            note_error(error, 'recovery_enumeration')

        try:
            support.source_check(repo, owner, entry, git_env)
            post_source = 'matched'
        except BaseException as error:
            post_source = type(error).__name__
            note_error(error, 'source_check')
        try:
            files = support.inventory(attempt / 'run')
            inventory_error = None
        except BaseException as error:
            files, inventory_error = [], type(error).__name__
            note_error(error, 'run_inventory')
        try:
            custody_raw = support.bounded_read(attempt / 'child-custody.json', 256 * 1024)
            custody = json.loads(custody_raw)
            custody_sha = support.sha(custody_raw)
        except (OSError, ValueError, TypeError):
            custody, custody_sha = None, None
        except BaseException as error:
            custody, custody_sha = None, None
            note_error(error, 'custody_read')
        statuses = {}
        for key, path in (('terminal', attempt / 'run' / 'attempt' / 'terminal.json'),
                          ('entry_result', attempt / 'run' / 'entry-result.json')):
            try:
                statuses[key] = json.loads(support.bounded_read(path, 256 * 1024))['status']
            except (OSError, ValueError, KeyError, TypeError):
                statuses[key] = None
            except BaseException as error:
                statuses[key] = None
                note_error(error, 'completion_read')
        try:
            physical_bytes = sum(item['bytes'] for item in support.inventory(attempt, hash_files=False))
        except BaseException as error:
            physical_bytes = None
            note_error(error, 'physical_inventory')
        custody_match = (result is not None and type(custody) is dict
                         and custody.get('schema') == 'agreement-direction-supervised-child-v1'
                         and custody.get('mode') == ('validate_only' if args.validation_only else 'study')
                         and custody.get('run_id') == result.run_id
                         and type(custody.get('owner_pid')) is int and custody['owner_pid'] == result.pid
                         and custody.get('prelaunch_sha256') == args.prelaunch_sha256)
        records_complete = (not (attempt / 'run').exists() if args.validation_only
                            else statuses['terminal'] == statuses['entry_result'] == 'complete_candidate')
        clean = (result is not None and result.clean and result.exit_code == 0 and not result.timed_out
                 and custody_match and records_complete and post_source == 'matched'
                 and inventory_error is None and physical_bytes is not None
                 and not recovery and not post_errors and pending is None)
        receipt = {'schema': 'agreement-direction-gated-observation-v1', 'source_revision': head,
                   'source_tree': tree, 'prelaunch_sha256': support.sha(pre_raw),
                   'argument_vector_sha256': support.sha(argument_bytes), 'start_utc': started, 'end_utc': ended,
                   'validation_only': args.validation_only, 'child': custody, 'child_sha256': custody_sha,
                   'child_binding_matches': custody_match,
                   'supervision_result': vars(result) if result is not None else None,
                   'supervision_error_type': error_type, 'recovery': recovery, 'post_errors': post_errors,
                   'post_source_check': post_source, 'run_inventory_error': inventory_error,
                   'completion_statuses': statuses, 'run_files': files,
                   'physical_bytes_before_receipt': physical_bytes, 'clean': clean,
                   'scope': 'Cooperative direct-child observation only; not authenticated execution or a descendant-drain attestation. Child stdout/stderr were inherited, not captured by this observer.',
                   'limits_scope': 'Owner tensor/event limits and post-run physical inventory are cooperative bounds, not an OS quota or a process-memory limit.'}
        receipt_raw = support.encoded(receipt)
        if physical_bytes is not None and physical_bytes + len(receipt_raw) > 1024**3:
            clean = receipt['clean'] = False
            receipt['post_errors'].append({'stage': 'physical_inventory', 'error_type': 'LimitExceeded'})
            receipt_raw = support.encoded(receipt)
        support.write_new(attempt / 'supervisor.json', receipt_raw)
        print(json.dumps({'clean': clean, 'exit_code': None if result is None else result.exit_code,
                          'receipt_sha256': support.sha((attempt / 'supervisor.json').read_bytes()),
                          'run_files': len(files)}, sort_keys=True), flush=True)
    if pending is not None:
        raise pending
    return 0 if clean else 1


if __name__ == '__main__':
    raise SystemExit(main())
