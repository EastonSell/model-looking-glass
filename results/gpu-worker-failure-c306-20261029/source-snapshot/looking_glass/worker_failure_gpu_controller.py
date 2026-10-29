"""Disabled source-bound controller for one synthetic failure exercise."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile

from .worker_failure_gpu import _SOURCE_PATHS, _check_disk, _write_new


_ENABLED = True
_PREFIX = 'LOOKING_GLASS_FAILURE_VALIDATION_'
_ROOTS = {'directory': None, 'output': None, 'stage': 'entry', 'source_guard': None,
          'manifest_guard': None, 'job': None, 'observed_job': None, 'failure': None, 'diagnostic': None}


class _RecordingJob:
    """Record the supervisor's two existing bounded Job-count observations."""
    def __init__(self, job):
        self.job = job
        self.counts = []

    def assign(self, process):
        return self.job.assign(process)

    def process_counts(self):
        if len(self.counts) >= 2:
            raise RuntimeError('unexpected additional job-count query')
        value = self.job.process_counts()
        self.counts.append(value)
        return value

    def terminate_and_wait(self, timeout=5):
        return self.job.terminate_and_wait(timeout=timeout)

    def close(self):
        return self.job.close()


def _workspace_root(repo, requested):
    repo, requested = Path(repo), Path(requested)
    if not repo.is_absolute() or not requested.is_absolute():
        raise ValueError('checkout and workspace must be absolute')
    repo, requested = repo.resolve(strict=True), requested.resolve(strict=True)
    ancestors = [parent for parent in repo.parents if parent.name.casefold() == 'work']
    if len(ancestors) != 1 or ancestors[0].name != 'work':
        raise ValueError('checkout requires one canonical work ancestor')
    if ancestors[0].parent != requested or (requested / 'work').resolve(strict=True) != ancestors[0]:
        raise ValueError('workspace does not match the shared device namespace')
    return requested


def _git(git, repo, *args):
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT')
           if key in os.environ}
    env.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_CONFIG_NOSYSTEM': '1',
                'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_COUNT': '0', 'GIT_TERMINAL_PROMPT': '0'})
    return subprocess.run([str(git), '--no-replace-objects', *args], cwd=repo, env=env,
                          capture_output=True, check=True, timeout=15).stdout


def _source_identity(repo, git, commit):
    if (re.fullmatch('[0-9a-f]{40}', commit) is None
            or _git(git, repo, 'rev-parse', 'HEAD').decode('ascii').strip() != commit
            or _git(git, repo, 'status', '--porcelain', '--untracked-files=all')):
        raise RuntimeError('source checkout is not the clean committed revision')
    tree = _git(git, repo, 'rev-parse', 'HEAD:').decode('ascii').strip()
    if re.fullmatch('[0-9a-f]{40}', tree) is None:
        raise RuntimeError('source tree identity is invalid')
    files = {}
    for relative in _SOURCE_PATHS:
        entry = _git(git, repo, 'ls-tree', '-z', '--full-tree', commit, '--', ':(literal)' + relative)
        if not entry.endswith(b'\0') or entry.count(b'\0') != 1:
            raise RuntimeError('source tree entry is invalid')
        metadata, separator, name = entry[:-1].partition(b'\t')
        fields = metadata.split(b' ')
        if (separator != b'\t' or name != relative.encode() or len(fields) != 3
                or fields[0] not in (b'100644', b'100755') or fields[1] != b'blob'):
            raise RuntimeError('source tree entry is not a regular file')
        with (repo / relative).open('rb') as stream:
            actual = stream.read(4 * 1024**2 + 1)
        if len(actual) > 4 * 1024**2 or actual != _git(git, repo, 'show', commit + ':' + relative):
            raise RuntimeError('source bytes differ from the committed tree')
        files[relative] = {'mode': fields[0].decode(), 'sha256': hashlib.sha256(actual).hexdigest()}
    return tree, files


def _parent_uname():
    # This is before the Windows Job exists; CPython can start `ver` here.
    import platform
    value = platform.uname()
    fields = {key: getattr(value, key) for key in
              ('system', 'node', 'release', 'version', 'machine', 'processor')}
    if (fields['system'] != 'Windows' or fields['machine'] != 'AMD64'
            or any(type(item) is not str or len(item) > 255
                   or any(not char.isprintable() for char in item) for item in fields.values())
            or re.fullmatch('[0-9]+\\.[0-9]+\\.[0-9]+', fields['version']) is None):
        raise RuntimeError('the Windows platform handoff is unsupported')
    return fields


def _child_environment(root, gpu_uuid, manifest_sha, uname, temp):
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'PATHEXT', 'COMSPEC')
           if key in os.environ}
    system32 = Path(env.get('SYSTEMROOT', 'C:/Windows')) / 'System32'
    env.update({'PATH': str(system32), 'CUDA_VISIBLE_DEVICES': gpu_uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'PYTORCH_CUDA_ALLOC_CONF': 'backend:native', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'HF_HOME': str(root / 'work/hf-cache'), 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TEMP': str(temp), 'TMP': str(temp),
                _PREFIX + 'MANIFEST_SHA256': manifest_sha,
                _PREFIX + 'PLATFORM_UNAME': json.dumps(uname, sort_keys=True, separators=(',', ':'))})
    return env


def _guards(repo, files):
    path = repo / 'scripts/v2_source_guards.py'
    raw = path.read_bytes()
    if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != files['scripts/v2_source_guards.py']['sha256']:
        raise RuntimeError('source guard bytes changed')
    namespace = {'__name__': '_worker_failure_guards', '__file__': str(path)}
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


def _preflight(workspace_root):
    repo = Path(__file__).resolve(strict=True).parents[1]
    root = _workspace_root(repo, workspace_root)
    site = Path(os.environ.get(_PREFIX + 'BOOTSTRAP_SITE', ''))
    git = Path(os.environ.get(_PREFIX + 'GIT_EXE', ''))
    commit = os.environ.get(_PREFIX + 'SOURCE_COMMIT', '')
    if (os.name != 'nt' or sys.implementation.name != 'cpython' or sys.version_info[:3] != (3, 11, 15)
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or os.environ.get(_PREFIX + 'BOOTSTRAP_PID') != str(os.getpid())
            or not site.is_absolute() or site.resolve(strict=True) != (root / 'work/.venv/Lib/site-packages').resolve(strict=True)
            or not git.is_absolute() or not git.is_file()):
        raise RuntimeError('the fixed isolated bootstrap and dependency site are required')
    free = _check_disk(root)
    tree, files = _source_identity(repo, git, commit)
    return root, repo, site, git, commit, tree, files, free


def _run_controller(workspace_root, output, coordinated_slot):
    if coordinated_slot is not True:
        raise ValueError('a coordinated device slot is required')
    root, repo, site, git, commit, tree, files, disk_free = _preflight(workspace_root)
    destination = Path(output)
    if (not destination.is_absolute() or destination.parent.resolve(strict=True) != root / 'work'
            or any(os.path.lexists(str(destination) + suffix) for suffix in ('', '.private.json', '.incomplete.json'))):
        raise ValueError('the output must be a fresh file directly under the shared work directory')
    _ROOTS['output'] = destination
    _ROOTS['stage'] = 'device_preflight'
    uname = _parent_uname()
    programfiles = os.environ.get('PROGRAMFILES')
    if not programfiles or not Path(programfiles).is_absolute() or not Path(programfiles).is_dir():
        raise RuntimeError('the Windows telemetry environment is unavailable')
    from .device_identity import resolve_device_identity
    from .gpu_oneshot import _smi_snapshot
    from .supervisor import run_supervised
    smi = (Path(os.environ.get('SYSTEMROOT', 'C:/Windows')) / 'System32/nvidia-smi.exe').resolve(strict=True)
    gpu_uuid, inventory = _smi_snapshot(str(smi))
    if inventory['free_bytes'] < 3 * 1024**3:
        raise RuntimeError('fresh device free memory is below the experiment floor')
    directory = Path(tempfile.mkdtemp(prefix='worker-failure-v1-', dir=root / 'work')).resolve(strict=True)
    _ROOTS['directory'] = directory
    _ROOTS['stage'] = 'prepared'
    run_id = secrets.token_hex(16)
    _write_new(Path(str(destination) + '.private.json'), {'schema': 'worker-failure-private-launch-v1',
        'run_id': run_id, 'source_commit': commit, 'run_directory': str(directory)})
    _write_new(directory / 'started.json', {'schema': 'worker-failure-controller-start-v1',
        'run_id': run_id, 'source_commit': commit, 'source_tree': tree, 'disk_free_bytes': disk_free,
        'gpu_uuid': gpu_uuid, 'inventory': inventory, 'platform_uname_private': uname})
    manifest_dir = directory / 'protected-manifest'
    manifest_dir.mkdir()
    manifest = {'schema': 'worker-failure-source-manifest-v1', 'run_id': run_id,
                'source_commit': commit, 'source_files': files}
    manifest_path = manifest_dir / 'source-manifest.json'
    manifest_sha = _write_new(manifest_path, manifest)
    guards = _guards(repo, files)
    source_guard = guards['SourceFileLease'](repo, [repo / path for path in _SOURCE_PATHS])
    _ROOTS['source_guard'] = source_guard
    source_guard.__enter__()
    manifest_guard = guards['SourceFileLease'](manifest_dir, [manifest_path])
    _ROOTS['manifest_guard'] = manifest_guard
    manifest_guard.__enter__()
    if (_source_identity(repo, git, commit) != (tree, files)
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha):
        raise RuntimeError('source or manifest changed while acquiring protection')
    temp = directory / 'temp'
    temp.mkdir()
    environment = _child_environment(root, gpu_uuid, manifest_sha, uname, temp)
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    prefix = Path(tempfile.mkdtemp(prefix='pycache-', dir=directory)).resolve(strict=True)
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
        str(repo / 'scripts/worker_failure_gpu_bootstrap.py'), '--bootstrap-site', str(site),
        '--source-commit', commit, '--git-executable', str(git), '--owner', str(directory),
        str(root / 'work'), run_id, commit, str(git)]
    _check_disk(root)
    _ROOTS['job'] = guards['WindowsJob']()
    job = _RecordingJob(_ROOTS['job'])
    _ROOTS['observed_job'] = job
    _ROOTS['stage'] = 'owner'
    supervised = run_supervised(command, device_lease_path=identity.lock_path,
        marker_path=identity.marker_path, timeout=300, lease_timeout=10, terminate_grace=5,
        cwd=repo, env=environment, identity=identity, inherit_environment=False,
        process_job=job, require_zero_exit=True)
    supervision = {'schema': 'worker-failure-supervision-v1', 'run_id': run_id,
        'supervisor_run_id': supervised.run_id, 'owner_pid': supervised.pid,
        'owner_exit': supervised.exit_code, 'clean': supervised.clean,
        'timed_out': supervised.timed_out, 'reason': supervised.reason,
        'job_counts': [list(value) for value in job.counts]}
    _write_new(directory / 'supervision.json', supervision)
    _ROOTS['stage'] = 'owner_reaped'
    manifest_guard.__exit__(None, None, None)
    source_guard.__exit__(None, None, None)
    _ROOTS['manifest_guard'] = _ROOTS['source_guard'] = None
    _write_new(destination, {'schema': 'worker-failure-controller-result-v1', 'status': 'incomplete',
        'source_commit': commit, 'run_id': run_id, 'owner_exit': supervised.exit_code,
        'supervisor_clean': supervised.clean, 'supervisor_timed_out': supervised.timed_out,
        'marker_present_at_point': os.path.lexists(identity.marker_path),
        'attestation_present_at_point': os.path.lexists(identity.attestation_path),
        'terminal_record_present': os.path.lexists(directory / 'terminal.json'),
        'limitations': ['simulated OOM is not device exhaustion',
            'an unclean owner requires separate recovery; process exit does not prove native CUDA cleanup']})
    return 1


def run_controller(*, workspace_root, output, coordinated_slot):
    if _ENABLED is not True:
        raise RuntimeError('the worker failure controller is disabled')
    try:
        return _run_controller(workspace_root, output, coordinated_slot)
    except BaseException as error:
        _ROOTS['failure'] = error
        directory = _ROOTS['directory']
        kind = type(error).__name__
        if re.fullmatch('[A-Za-z_][A-Za-z_0-9]{0,63}', kind) is None:
            kind = 'other'
        failure = {'schema': 'worker-failure-controller-incomplete-v1', 'status': 'incomplete',
                   'stage': _ROOTS['stage'], 'error_type': kind}
        if directory is not None:
            try:
                _write_new(directory / 'controller-failure.json', failure)
            except BaseException as diagnostic:
                _ROOTS['diagnostic'] = diagnostic
        output = _ROOTS.get('output')
        if output is not None:
            try:
                target = Path(str(output) + '.incomplete.json') if os.path.lexists(output) else output
                _write_new(target, failure)
            except BaseException as diagnostic:
                _ROOTS['diagnostic'] = diagnostic
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--coordinated-slot', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if args.check_only:
        _preflight(args.workspace_root)
        return 0
    return run_controller(workspace_root=args.workspace_root, output=args.output,
                          coordinated_slot=args.coordinated_slot)
