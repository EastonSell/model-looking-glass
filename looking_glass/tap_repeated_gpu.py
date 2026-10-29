"""Source-bound controller for one provisional v1.3 timing session.

The parent verifies committed sources and an offline checkpoint before it
starts a Job-supervised child. The child verifies a hash-bound source manifest
without spawning Git. Source stability, dependency binaries, GPU exclusivity
and the operator's CPU-coordination report remain cooperative assumptions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time


# The bootstrap parses this literal before importing project code. Update it
# when a directly or transitively used source changes; omitted modules fail
# the source-only import allowlist.
_SOURCE_PATHS = (
    'scripts/tap_repeated_gpu_bootstrap.py', 'scripts/v2_source_guards.py',
    'evidence/gpu-tap-latency-repeated-v1/schedule.json',
    'docs/GPU_TAP_LATENCY_REPEATED_V1_3_DRAFT.md',
    'looking_glass/__init__.py', 'looking_glass/tap_repeated_gpu.py',
    'looking_glass/tap_repeated_gpu_owner.py', 'looking_glass/tap_repeated_gpu_evidence.py',
    'looking_glass/tap_repeated_parent.py', 'looking_glass/tap_repeated_child_barrier.py',
    'looking_glass/tap_repeated_activity.py', 'looking_glass/tap_repeated_protocol.py',
    'looking_glass/tap_repeated_admission.py', 'looking_glass/tap_repeated_owner.py',
    'looking_glass/tap_repeated_cuda_backend.py', 'looking_glass/tap_repeated_timing.py',
    'looking_glass/tap_repeated_lean.py', 'looking_glass/tap_repeated_preflight.py',
    'looking_glass/tap_repeated_model.py', 'looking_glass/tap_repeated_disposal.py',
    'looking_glass/device_identity.py', 'looking_glass/device_lease.py', 'looking_glass/supervisor.py',
    'looking_glass/memory_budget.py', 'looking_glass/pretrained.py', 'looking_glass/gpu_oneshot.py',
    'looking_glass/cuda_cleanup.py', 'looking_glass/transformer_trace.py',
    'looking_glass/trace_schema.py', 'looking_glass/probes.py', 'looking_glass/model_ownership.py',
)
_ENABLED = True
_PREFIX = 'LOOKING_GLASS_TAP_REPEATED_'
_ROOTS = {}
_ENTRY_LOCK = threading.Lock()
_ENTRY_CLAIMED = False


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _write_new(path, value):
    raw = _canonical(value)
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short evidence write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _publish_result(path, value):
    """Publish complete bytes create-only on the supported Windows host."""
    if os.name != 'nt':
        raise RuntimeError('result publication requires Windows create-only rename')
    path = Path(path)
    partial = Path(str(path) + '.partial')
    _write_new(partial, value)
    os.rename(partial, path)


def _check_output_destination(output):
    """Reserve a fresh result name and its related create-only records."""
    destination = Path(output)
    related = (destination, *(Path(str(destination) + suffix) for suffix in
                              ('.private.json', '.partial', '.incomplete.json')))
    if (not destination.is_absolute() or not destination.parent.is_dir()
            or any(os.path.lexists(path) for path in related)):
        raise ValueError('output and its related records must be absent at an absolute path')
    return destination


def _read_bounded(path, maximum=256 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    if not raw or len(raw) > maximum:
        raise ValueError('private record has an invalid size')
    return raw


def _workspace_root(repo, requested):
    repo, requested = Path(repo), Path(requested)
    if not repo.is_absolute() or not requested.is_absolute():
        raise ValueError('checkout and workspace must be absolute')
    repo, requested = repo.resolve(strict=True), requested.resolve(strict=True)
    ancestors = [parent for parent in repo.parents if parent.name.casefold() == 'work']
    if len(ancestors) != 1 or ancestors[0].name != 'work':
        raise ValueError('checkout must have one canonical work ancestor')
    work = ancestors[0]
    if (work.parent / 'work').resolve(strict=True) != work or requested != work.parent:
        raise ValueError('workspace does not match the shared checkout layout')
    return requested


def _git(git, repo, *arguments):
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT')
           if key in os.environ}
    env.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_CONFIG_NOSYSTEM': '1',
                'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_COUNT': '0', 'GIT_TERMINAL_PROMPT': '0'})
    return subprocess.run([str(git), '--no-replace-objects', *arguments], cwd=repo, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15).stdout


def _source_identity(repo, git, commit):
    if (re.fullmatch('[0-9a-f]{40}', commit) is None
            or _git(git, repo, 'rev-parse', 'HEAD').decode('ascii').strip() != commit
            or _git(git, repo, 'status', '--porcelain', '--untracked-files=all')):
        raise RuntimeError('source checkout is not the clean committed revision')
    # HEAD: names the root tree without braces, which the Windows MSYS Git
    # argument shim otherwise strips from HEAD^{tree}.
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
            raise RuntimeError('source tree entry is not a regular blob')
        path = repo / relative
        actual = _read_bounded(path, 4 * 1024 * 1024)
        if actual != _git(git, repo, 'show', commit + ':' + relative):
            raise RuntimeError('source bytes differ from the committed tree')
        files[relative] = {'mode': fields[0].decode('ascii'), 'sha256': hashlib.sha256(actual).hexdigest()}
    return tree, files


def _source_guards(repo, files):
    path = repo / 'scripts/v2_source_guards.py'
    raw = _read_bounded(path, 256 * 1024)
    if hashlib.sha256(raw).hexdigest() != files['scripts/v2_source_guards.py']['sha256']:
        raise RuntimeError('source-guard code changed')
    namespace = {'__name__': '_tap_source_guards', '__file__': str(path)}
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


def _clock():
    info = time.get_clock_info('perf_counter')
    clock = {'name': 'perf_counter_ns', 'monotonic': info.monotonic, 'adjustable': info.adjustable,
             'implementation': info.implementation, 'resolution_seconds': info.resolution}
    runtime = {'platform': sys.platform, 'implementation': sys.implementation.name,
               'python_version': '.'.join(map(str, sys.version_info[:3]))}
    from .tap_repeated_parent import _clock_margin
    _clock_margin(runtime, clock)
    return clock, runtime


def _expected_model():
    from .pretrained import ASSETS, MODEL_ID, REVISION
    return {'model_id': MODEL_ID, 'model_revision': REVISION,
            'weights_sha256': ASSETS['model.safetensors'][1],
            'asset_sha256': {name: digest for name, (_, digest) in ASSETS.items()},
            'parameter_count': 81912576, 'attention_implementation': 'eager',
            'dtype': 'float32', 'use_cache': False, 'input_ids': [15496, 995],
            'input_ids_sha256_le_i64': hashlib.sha256(struct.pack('<2q', 15496, 995)).hexdigest(),
            'device': 'cuda:0'}


def _parent_uname():
    # This occurs before the Job exists; CPython may run Windows `ver` here.
    import platform
    observed = platform.uname()
    fields = {key: getattr(observed, key) for key in
              ('system', 'node', 'release', 'version', 'machine', 'processor')}
    if (fields['system'] != 'Windows' or fields['machine'] != 'AMD64'
            or any(type(value) is not str or len(value) > 255
                   or any(not char.isprintable() for char in value) for value in fields.values())
            or re.fullmatch('[0-9]+\\.[0-9]+\\.[0-9]+', fields['version']) is None):
        raise RuntimeError('the parent Windows platform observation is unsupported')
    return fields


def _require_smi_environment():
    # On this Windows host the system utility requires PROGRAMFILES even when
    # its absolute executable path and System32 PATH are supplied. Both the
    # initial query and parent activity collector inherit this environment.
    value = os.environ.get('PROGRAMFILES')
    if type(value) is not str or not Path(value).is_absolute() or not Path(value).is_dir():
        raise RuntimeError('the controller requires the Windows PROGRAMFILES directory for telemetry')


def _child_environment(root, uuid, session, challenge, manifest_sha, uname, temp_dir):
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'PATHEXT', 'COMSPEC')
           if key in os.environ}
    system32 = Path(env.get('SYSTEMROOT', 'C:/Windows')) / 'System32'
    env.update({'PATH': str(system32), 'CUDA_VISIBLE_DEVICES': uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'PYTORCH_CUDA_ALLOC_CONF': 'backend:native', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'HF_HOME': str(root / 'work/hf-cache'), 'TEMP': str(temp_dir), 'TMP': str(temp_dir),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', _PREFIX + 'SESSION': str(session),
        _PREFIX + 'CHALLENGE': challenge, _PREFIX + 'MANIFEST_SHA256': manifest_sha,
        _PREFIX + 'PLATFORM_UNAME': json.dumps(uname, sort_keys=True, separators=(',', ':'), allow_nan=False)})
    return env


def _resource_summary(rows, admission_memory, memory_before_transfer):
    measured = [row for row in rows if row.get('stage') == 'measured']
    samples = [row for row in rows if row.get('stage') in ('preflight', 'warmup', 'measured')]
    # Replay validates the observation's before and its paired postcheck's
    # after. A duplicate field on the observation is not an authority here.
    snapshots = [admission_memory, memory_before_transfer] + [
        row['memory_before'] if row['stage'] in ('preflight', 'warmup', 'measured') else row['memory_after']
        for row in rows if row.get('stage') in ('preflight', 'warmup', 'measured', 'postcheck')]
    return {'maximum_recorded_reset_peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in samples),
            'maximum_recorded_reset_peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in samples),
            'minimum_free_bytes_in_admission_pretransfer_and_journal': min(row['free_bytes'] for row in snapshots),
            'guarded_wall_ns_medians': {mode: statistics.median(row['guarded_wall_ns'] for row in measured
                                                               if row['mode'] == mode)
                                        for mode in ('none', 'direct', 'gradient')}}


def _run_session(*, workspace_root, cache_dir, output, session, cpu_context, coordinated_slot):
    """Execute one fixed supervised session; no aggregate is inferred."""
    if _ENABLED is not True:
        raise RuntimeError('the repeated GPU controller is disabled')
    if (os.name != 'nt' or type(session) is not int or session != 0
            or cpu_context not in ('targeted_pause', 'targeted_pause_other_work', 'unknown')
            or coordinated_slot is not True or os.environ.get(_PREFIX + 'BOOTSTRAP_PID') != str(os.getpid())):
        raise ValueError('the reviewed Windows bootstrap and coordinated slot are required')
    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    cache = Path(cache_dir).resolve(strict=True)
    if cache != (root / 'work/hf-cache/hub').resolve(strict=True):
        raise ValueError('the canonical offline cache is required')
    site = Path(os.environ.get(_PREFIX + 'BOOTSTRAP_SITE', '')).resolve(strict=True)
    git = Path(os.environ.get(_PREFIX + 'GIT_EXE', '')).resolve(strict=True)
    commit = os.environ.get(_PREFIX + 'SOURCE_COMMIT', '')
    destination = _check_output_destination(output)
    private_receipt = Path(str(destination) + '.private.json')
    tree, files = _source_identity(repo, git, commit)
    from .gpu_oneshot import _smi_snapshot, _verify_offline_cache
    from .device_identity import resolve_device_identity
    from .supervisor import StartupChannel
    from .tap_repeated_activity import ParentActivity, _NvidiaSmi
    from . import tap_repeated_parent as parent
    from .tap_repeated_gpu_evidence import assemble_private_attempt
    from .tap_repeated_gpu_owner import _expected_runtime
    from .tap_repeated_protocol import EXPERIMENT_PROTOCOL, load_schedule, summarize_attempts
    _verify_offline_cache(cache)
    schedule = load_schedule(repo / 'evidence/gpu-tap-latency-repeated-v1/schedule.json')
    clock, parent_runtime = _clock()
    uname = _parent_uname()
    _require_smi_environment()
    smi = (Path(os.environ.get('SYSTEMROOT', 'C:/Windows')) / 'System32/nvidia-smi.exe').resolve(strict=True)
    uuid, before = _smi_snapshot(str(smi))
    run_dir = Path(tempfile.mkdtemp(prefix='tap-repeated-v12-', dir=root / 'work')).resolve(strict=True)
    _ROOTS.update(run_dir=run_dir, stage='created')
    run_id, challenge = secrets.token_hex(16), secrets.token_hex(32)
    _write_new(private_receipt, {'schema': 'tap-repeated-private-launch-v1',
        'run_directory': str(run_dir), 'run_id': run_id, 'source_commit': commit, 'session': session})
    manifest_dir = run_dir / 'protected-manifest'
    manifest_dir.mkdir()
    manifest = {'schema': 'tap-repeated-source-manifest-v1', 'run_id': run_id,
                'source_commit': commit, 'source_files': files}
    manifest_sha = _write_new(manifest_dir / 'source-manifest.json', manifest)
    _write_new(run_dir / 'started.json', {'schema': 'tap-repeated-controller-start-v1',
        'run_id': run_id, 'source_commit': commit, 'source_tree': tree, 'session': session,
        'source_manifest_sha256': manifest_sha, 'platform_uname_private': uname})
    guards = _source_guards(repo, files)
    source_guard = guards['SourceFileLease'](repo, [repo / path for path in _SOURCE_PATHS])
    _ROOTS['source_guard'] = source_guard
    source_guard.__enter__()
    manifest_guard = guards['SourceFileLease'](manifest_dir, [manifest_dir / 'source-manifest.json'])
    _ROOTS['manifest_guard'] = manifest_guard
    manifest_guard.__enter__()
    if (_source_identity(repo, git, commit) != (tree, files)
            or hashlib.sha256(_read_bounded(manifest_dir / 'source-manifest.json')).hexdigest() != manifest_sha):
        raise RuntimeError('source or manifest changed while acquiring protection')
    temp_dir = run_dir / 'temp'
    temp_dir.mkdir()
    environment = _child_environment(root, uuid, session, challenge, manifest_sha, uname, temp_dir)
    identity = resolve_device_identity(root, uuid, physical_uuids=[uuid], environ=environment)
    prefix = Path(tempfile.mkdtemp(prefix='pycache-', dir=run_dir)).resolve(strict=True)
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
        str(repo / 'scripts/tap_repeated_gpu_bootstrap.py'), '--bootstrap-site', str(site),
        '--source-commit', commit, '--git-executable', str(git), '--owner', str(run_dir),
        str(cache), run_id, commit, str(git)]
    channel = StartupChannel(max_bytes=4096)
    job = guards['WindowsJob']()
    _ROOTS['job'] = job
    supervision = parent._Supervision(command, channel=channel, identity=identity,
        device_lease_path=identity.lock_path, marker_path=identity.marker_path,
        cwd=repo, env=environment, timeout=1200, process_job=job)
    journal = parent._PrivateJournal(run_dir / 'activity.jsonl')
    activity = ParentActivity(gpu_uuid=uuid, query=_NvidiaSmi(smi), max_samples=4096)
    _ROOTS['stage'] = 'supervised_owner'
    parent_result = parent._drive_session(session_index=session, challenge=challenge, identity=identity,
        channel=channel, supervision=supervision, activity=activity, journal=journal, attempt_dir=run_dir,
        expected_smi_total=before['total_bytes'], schedule=schedule, parent_clock=clock,
        parent_runtime=parent_runtime, exclusive=True, clock=time.perf_counter_ns, sleep=time.sleep,
        startup_timeout=180, workload_timeout=900, reap_timeout=30)
    _ROOTS['stage'] = 'closed_owner_review'
    if _source_identity(repo, git, commit) != (tree, files):
        raise RuntimeError('source changed during the supervised session')
    candidate_raw = _read_bounded(run_dir / 'candidate.json')
    attempt, binding = assemble_private_attempt(parent=parent_result,
        candidate_raw=candidate_raw, final_raw=_read_bounded(run_dir / 'final.json'),
        study_run_id=run_id, source_commit=commit, source_tree=tree, expected_model=_expected_model(),
        expected_runtime=_expected_runtime(), parent_runtime=parent_runtime,
        expected_device_name=before['device_name'], observed_driver=before['driver_version'], clock=clock,
        exclusive=True, source_exit_contract=True)
    replay = summarize_attempts(schedule, [attempt], trusted_binding=binding)
    slot = replay['slots'][session]
    if slot['state'] != 'complete':
        raise RuntimeError('the fixed timing replay is incomplete')
    candidate = json.loads(candidate_raw)
    result = {'schema': 'tap-repeated-provisional-session-v1', 'protocol': EXPERIMENT_PROTOCOL,
        'status': ('completed_exploratory_contaminated' if slot['screen']['status'] == 'contaminated'
                   else 'completed_provisional'),
        'source_commit': commit, 'source_tree': tree, 'session': session, 'slot': slot,
        'device': {'name': before['device_name'], 'driver_point_observation': before['driver_version'],
                   'compute_capability': candidate['cuda_compute_capability']},
        'resources': _resource_summary(attempt['journal'], candidate['admission_memory'],
                                       candidate['memory_before_transfer']),
        'coordination': {'gpu_slot': 'operator_coordinated', 'cpu_context': cpu_context},
        'limitations': ['one provisional session, one model and one input; no multi-session aggregate or slot ledger',
            'parent telemetry and fsync may overlap measured wall time',
            'the controller telemetry subprocesses inherit the Windows PROGRAMFILES setting',
            'the reported minimum free memory covers saved admission, pre-transfer and journal snapshots, not every setup observation',
            'cooperative point observations cannot prove global GPU idleness or native CUDA cleanup',
            'source and dependency checks assume a trusted local runtime and stable checkout',
            'acceptance requires a separately observed zero controller exit and no incomplete sidecar; sidecar creation is best effort']}
    manifest_guard.__exit__(None, None, None)
    source_guard.__exit__(None, None, None)
    _ROOTS['stage'] = 'publishing'
    _publish_result(destination, result)
    return result


def _partial_pairs(path):
    """Count only adjacent present pairs; these partial rows remain unvalidated."""
    try:
        raw = _read_bounded(path, 8 * 1024 * 1024)
        lines = raw.splitlines(keepends=True)
        rows = []
        for line in lines[:86]:
            if not line.endswith(b'\n'):
                break
            value = json.loads(line)
            if type(value) is not dict:
                break
            rows.append(value)
        count = 0
        for index in range(0, len(rows) - 1, 2):
            first, post = rows[index:index + 2]
            if (first.get('stage') not in ('preflight', 'warmup', 'measured')
                    or post.get('stage') != 'postcheck' or post.get('status') != 'passed'
                    or type(first.get('observation_id')) is not str
                    or first['observation_id'] != post.get('observation_id')):
                break
            count += 1
        return count
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        return 0


def run_session(*, workspace_root, cache_dir, output, session, cpu_context, coordinated_slot):
    """Preserve a bounded incomplete outcome without hiding the original fault."""
    global _ENTRY_CLAIMED
    if _ENABLED is not True:
        raise RuntimeError('the repeated GPU controller is disabled')
    with _ENTRY_LOCK:
        if _ENTRY_CLAIMED:
            raise RuntimeError('the controller is single-use')
        _ENTRY_CLAIMED = True
    try:
        return _run_session(workspace_root=workspace_root, cache_dir=cache_dir, output=output,
                            session=session, cpu_context=cpu_context, coordinated_slot=coordinated_slot)
    except BaseException as error:
        directory = _ROOTS.get('run_dir')
        if isinstance(directory, Path):
            kind = type(error).__name__
            if kind not in ('RuntimeError', 'ValueError', 'TimeoutError', 'OSError', 'FileNotFoundError',
                            'KeyboardInterrupt', 'SystemExit'):
                kind = 'other'
            partial = {'schema': 'tap-repeated-controller-failure-v1', 'status': 'incomplete',
                       'stage': _ROOTS.get('stage', 'unknown'), 'error_type': kind,
                       'unvalidated_present_pairs': _partial_pairs(directory / 'journal.jsonl')}
            try:
                _write_new(directory / 'controller-failure.json', partial)
            except BaseException:
                pass
            try:
                # A visible result may already exist if publication was
                # interrupted after rename. Never overwrite it; the distinct
                # incomplete sidecar invalidates a clean-outcome assumption.
                target = Path(str(output) + '.incomplete.json') if Path(output).exists() else Path(output)
                _write_new(target, partial)
            except BaseException:
                pass
        raise


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] == '--owner':
        if len(args) != 6:
            raise ValueError('invalid fixed owner arguments')
        from .tap_repeated_gpu_owner import owner_main
        return owner_main(*args[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--session', type=int, choices=(0,), default=0)
    parser.add_argument('--cpu-context', choices=('targeted_pause', 'targeted_pause_other_work', 'unknown'),
                        default='unknown')
    parser.add_argument('--coordinated-slot', action='store_true')
    options = parser.parse_args(args)
    run_session(workspace_root=options.workspace_root, cache_dir=options.cache_dir, output=options.output,
                session=options.session, cpu_context=options.cpu_context, coordinated_slot=options.coordinated_slot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
