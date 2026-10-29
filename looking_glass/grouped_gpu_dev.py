"""Supervised, fixed-shape grouped-GPU development check; not a benchmark.

CUDA-capable libraries are deliberately imported only after the child acquires
the supervisor's exact physical-device lease.  See GROUPED_GPU_DEV_PLAN.md.
"""

import argparse
from datetime import datetime
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import weakref

from .device_identity import resolve_device_identity
from .agreement_calibration_workspace import shared_workspace_root
from .device_lease import DeviceLease
from .bert_gpu_evidence import write_json_new as _write_json
from .gpu_oneshot import (_owner_presence_after_exit, _runtime_uuid, _smi_snapshot,
                          _verify_controller_import_mode,
                          _verify_offline_cache)
from .supervisor import OwnerSession, run_supervised


_GIB = 1024 ** 3
_OWNER_TIMEOUT_SECONDS = 900
_PROTOCOL = 'experiments/grouped_gpu_dev_v3.json'
_MANIFEST = 'protected-manifest/source-manifest.json'
_SOURCE_PATHS = (
    _PROTOCOL, 'docs/GROUPED_GPU_DEV_PLAN.md', 'scripts/grouped_gpu_dev_bootstrap.py', 'looking_glass/__init__.py',
    'looking_glass/grouped_gpu_dev.py', 'looking_glass/bert_gpu_evidence.py',
    'looking_glass/gpu_oneshot.py', 'looking_glass/device_identity.py',
    'looking_glass/device_lease.py', 'looking_glass/supervisor.py',
    'looking_glass/cuda_cleanup.py', 'looking_glass/pretrained.py',
    'looking_glass/resident_worker.py', 'looking_glass/memory_budget.py',
    'looking_glass/budgeted_worker.py', 'looking_glass/host_budget.py',
    'looking_glass/fair_queue.py', 'looking_glass/microbatch.py',
    'looking_glass/grouped_worker.py', 'looking_glass/grouped_broker_state.py',
    'looking_glass/grouped_broker.py', 'looking_glass/batch_validation.py',
    'looking_glass/certified_batch_policy.py', 'looking_glass/grouped_observation.py',
    'looking_glass/grouped_owner_worker.py', 'looking_glass/owner_completion.py',
    'looking_glass/agreement_calibration_workspace.py',
    'scripts/v2_source_guards.py',
)
# The owner remains process-rooted on every failure.  Its device lease is held
# until process exit; no exception path makes a clean attestation.
_ROOTS = [None] * 25
# The dedicated driver is single-threaded. A slot for each permitted nested
# call retains completed results independently of public exception frames.
_CUDA_RESULTS = [None] * 8
_CUDA_ERRORS = [None] * 8
_CUDA_INTERRUPTS = [None] * 8
# An extra marker covers rejection at the maximum depth before work starts.
_CUDA_PENDING = [False] * 9
_CUDA_DEPTH = 0
_ATTEMPT_LOCK = threading.Lock()
_ATTEMPT_STARTED = False
_NO_CHILD_POPEN = None
_NO_CYCLIC_GC_REQUIRED = False
# This exact source enables the reviewed, supervised one-shot development run.
# It is not an authorization for a general shared service or other workload.
_GPU_BOUNDARY_READY = True


class _AttemptAlreadyStarted(RuntimeError):
    pass


def _claim_attempt():
    global _ATTEMPT_STARTED
    with _ATTEMPT_LOCK:
        if _ATTEMPT_STARTED:
            raise _AttemptAlreadyStarted('only one grouped owner attempt is allowed per process')
        _ATTEMPT_STARTED = True


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _require_supervisor():
    from . import supervisor
    expected = Path(__file__).resolve().with_name('supervisor.py')
    if (getattr(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT', None) != 'gated-rooted-child-v1'
            or Path(supervisor.__file__).resolve(strict=True) != expected):
        raise RuntimeError('the reviewed gated child-launch supervisor is required')


def _require_gpu_boundary():
    if _GPU_BOUNDARY_READY is not True:
        raise RuntimeError('the session-bound GPU worker boundary has not been reviewed')


def _workspace_root(repo, requested):
    expected = shared_workspace_root(repo)
    path = Path(requested)
    if not path.is_absolute() or path.resolve(strict=True) != expected:
        raise ValueError('workspace root differs from the shared device namespace')
    return expected


def _validate_workspace_spec(session, repo):
    spec = getattr(session, '_identity_spec', None)
    if type(spec) is not dict or type(spec.get('workspace_root')) is not str:
        raise RuntimeError('a supervised physical-device workspace is required')
    _workspace_root(repo, spec['workspace_root'])


def _validate_workspace_binding(session, repo):
    identity = session.identity
    if identity is None:
        raise RuntimeError('a supervised physical-device identity is required')
    _workspace_root(repo, identity.workspace_root)


def _require_bootstrap():
    package = str(Path(__file__).resolve().parent)
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or os.environ.get('LOOKING_GLASS_GROUPED_BOOTSTRAP_PID') != str(os.getpid())
            or not any(getattr(item, '_grouped_project', None) == package for item in sys.meta_path)
            or not any(getattr(item, '_grouped_sourceless_guard', False) is True for item in sys.meta_path)):
        raise RuntimeError('the isolated source-only grouped bootstrap is required')


def _protocol(path, expected_sha256):
    raw = Path(path).read_bytes()
    if len(raw) > 16384 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('fixed protocol bytes differ from the frozen digest')
    value = json.loads(raw)
    fixed = {
        'schema': 'grouped-gpu-development-v3',
        'model': 'distilbert/distilgpt2',
        'revision': '2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'seed': 2026102808, 'device': 'cuda:0', 'dtype': 'float32',
        'attention': 'eager', 'cache': 'disabled', 'autocast': 'disabled',
        'allocator_fraction': .25, 'minimum_preload_free_bytes': 5 * _GIB,
        'minimum_reserve_bytes': _GIB // 2, 'resident_commitment_bytes': 2 * _GIB,
        'forward_commitment_bytes': _GIB,
        'workspace_lifetime': 'retained-through-guarded-cleanup-release-before-attestation',
        'forward_order': ['serial-0-0', 'serial-0-1', 'mixed-0', 'serial-1-0', 'serial-1-1', 'mixed-1',
                          'heldout-serial-0-0', 'heldout-serial-0-1', 'heldout-group-0',
                          'heldout-serial-1-0', 'heldout-serial-1-1', 'heldout-group-1'],
        'owner_timeout_seconds': _OWNER_TIMEOUT_SECONDS,
        'child_process_policy': 'popen-disabled-job-fail-closed',
        'per_request_shape': [1, 4],
        'combined_shape': [2, 4], 'model_output_width': 50257,
        'resident_input_limit_bytes': 128, 'resident_output_limit_bytes': 2 * 1024 ** 2,
        'absolute_tolerance': 1e-6, 'relative_tolerance': 1e-5,
        'policy_groups': [[[50256, 103, 203, 303], [50256, 104, 204, 304]],
                          [[50256, 105, 205, 305], [50256, 106, 206, 306]]],
        'broker_holdout_groups': [[[50256, 107, 207, 307], [50256, 108, 208, 308]],
                                  [[50256, 109, 209, 309], [50256, 110, 210, 310]]],
        'teams': ['team-a', 'team-b'],
        'host_ledger': {'max_bytes': 64 * 1024 ** 2, 'max_items': 256,
                        'max_requests': 8, 'max_operations': 16},
        'control_host_ledger': {'max_bytes': 32 * 1024 ** 2, 'max_items': 128,
                                'max_requests': 1, 'max_operations': 1},
        'snapshot_reservation': {'nbytes': 256, 'items': 8},
        'worker_finite_scratch': {'nbytes': 4 * 1024 ** 2, 'items': 8},
        'validation_scratch': {'nbytes': 4 * 1024 ** 2, 'items': 8},
        'comparison_scratch': {'chunk_elements': 4096, 'declared_max_bytes': 1024 ** 2},
    }
    if (type(value) is not dict or any(key not in value or _canonical(value[key]) != _canonical(item)
                                      for key, item in fixed.items())):
        raise ValueError('protocol does not match the fixed development workload')
    if set(value) != set(fixed) | {'purpose', 'failure_policy'}:
        raise ValueError('unsupported protocol fields')
    return value


def _git(repo, *args):
    executable = shutil.which('git')
    if executable is None:
        raise RuntimeError('Git is unavailable for parent source verification')
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith('GIT_')}
    environment.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_GRAFT_FILE': os.devnull,
                        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                        'GIT_ATTR_NOSYSTEM': '1', 'GIT_TERMINAL_PROMPT': '0'})
    result = subprocess.run([executable, '--no-replace-objects', '-c', 'core.fsmonitor=false',
                             '-C', str(repo), *args], capture_output=True, check=True,
                            timeout=10, env=environment)
    return result.stdout


def _verify_sources(repo, commit):
    if type(commit) is not str or re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('an exact source commit is required')
    hashes = {}
    for path in _SOURCE_PATHS:
        current = (repo / path).read_bytes()
        recorded = _git(repo, 'show', commit + ':' + path)
        if current != recorded:
            raise RuntimeError('committed source bytes do not match the checkout')
        hashes[path] = hashlib.sha256(current).hexdigest()
    return hashes


def _verify_child_sources(repo, run_dir, run_id, commit, manifest_sha256):
    """Recheck parent-verified source bytes without spawning a child process."""
    if (type(manifest_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', manifest_sha256) is None
            or type(commit) is not str or re.fullmatch(r'[0-9a-f]{40}', commit) is None):
        raise RuntimeError('invalid parent source binding')
    raw = _bounded_bytes(run_dir / _MANIFEST, 65536)
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise RuntimeError('parent source manifest digest differs')
    manifest = json.loads(raw)
    if (type(manifest) is not dict or set(manifest) != {'schema', 'run_id', 'source_commit', 'source_sha256'}
            or manifest['schema'] != 'grouped-gpu-source-manifest-v1' or manifest['run_id'] != run_id
            or manifest['source_commit'] != commit or type(manifest['source_sha256']) is not dict
            or set(manifest['source_sha256']) != set(_SOURCE_PATHS)):
        raise RuntimeError('parent source manifest does not match this owner')
    hashes = manifest['source_sha256']
    for path in _SOURCE_PATHS:
        digest = hashes[path]
        if (type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                or hashlib.sha256(_bounded_bytes(repo / path, 4 * 1024 * 1024)).hexdigest() != digest):
            raise RuntimeError('owner source bytes differ from the parent manifest')
    return hashes


def _install_no_child_processes():
    """Block ``subprocess.Popen`` inside the trusted one-shot owner.

    Calls through ``os.system``, ``os.spawn*`` or native APIs are not blocked.
    The Windows Job fails closed if a descendant is not fully inventoried.
    This is an in-process contract, not a sandbox.
    """
    global _NO_CHILD_POPEN
    base = subprocess.Popen
    if (type(base) is not type or base.__module__ != 'subprocess' or base.__name__ != 'Popen'
            or _NO_CHILD_POPEN is not None):
        raise RuntimeError('unexpected subprocess policy before owner setup')

    class UnavailablePopen(base):
        def __init__(self, *args, **kwargs):
            raise OSError('subprocess creation is disabled in the grouped owner')

        def __del__(self):
            pass

    _NO_CHILD_POPEN = UnavailablePopen
    subprocess.Popen = UnavailablePopen


def _assert_no_child_processes():
    if _NO_CHILD_POPEN is None or subprocess.Popen is not _NO_CHILD_POPEN:
        raise RuntimeError('the owner subprocess policy changed')


def _cuda_uncertain():
    # Active ancestors are expected; a pending call at or below the current
    # depth represents an abandoned/failed call whose handler may not have
    # reached the ordinary exception latch.
    return _ROOTS[15] is not None or any(_CUDA_PENDING[_CUDA_DEPTH:])


def _assert_disposable_return_slots():
    if (_CUDA_DEPTH != 0 or _cuda_uncertain()
            or any(type(value) not in (type(None), bool, int, float, str) for value in _CUDA_RESULTS)):
        raise RuntimeError('guarded return slots retain uncertain or resource-bearing values')


def _check_cleanup_reservations(session, lease, budget, resident, workspace, authority,
                                protocol, expected_active):
    """Check exact, host-only accounting and the still-held physical owner."""
    if (_ROOTS[0] is not session or _ROOTS[1] is not lease
            or _ROOTS[2] is not budget or _ROOTS[3] is not resident or _ROOTS[19] is not workspace
            or _ROOTS[20] is not authority or resident is workspace or gc.isenabled()):
        raise RuntimeError('cleanup owner roots or cyclic-GC state changed')
    if _cuda_uncertain() or authority.unsafe:
        raise RuntimeError('cleanup owner is unsafe')
    OwnerSession.check(session, lease)
    with budget._lock:
        if (budget._trace_limits != {'gpu': 0, 'host': 0}
                or budget._minimum_reserve_bytes != protocol['minimum_reserve_bytes']
                or budget._leases != frozenset(expected_active)):
            raise RuntimeError('cleanup accounting reservations changed')
        for item, amount in ((resident, protocol['resident_commitment_bytes']),
                             (workspace, protocol['forward_commitment_bytes'])):
            if (item._budget is not budget or item._issued is not True
                    or item._estimated_incremental_bytes != amount
                    or item._trace_limits != {'gpu': 0, 'host': 0}
                    or item._trace_used != {'gpu': 0, 'host': 0}):
                raise RuntimeError('cleanup accounting reservation is invalid')
    OwnerSession.check(session, lease)
    if _cuda_uncertain() or authority.unsafe:
        raise RuntimeError('cleanup owner became unsafe')


def _guarded_cleanup(session, lease, torch, check_reservations, expected_total, on_partial):
    """Check physical ownership and the fixed reservations around every CUDA call."""
    from .cuda_cleanup import clear_cublas_after_drain
    values = {}
    def guarded(name, function):
        def call(*args, **kwargs):
            value = None
            has_value = False
            try:
                if name == 'memory_allocated':
                    # Start a new before/after pair; an interrupted second
                    # snapshot must not inherit the first reserved counter.
                    values.clear()
                check_reservations()
                value = function(*args, **kwargs)
                has_value = True
                if name in ('memory_allocated', 'memory_reserved'):
                    values[name] = value
                check_reservations()
                if name in ('memory_allocated', 'memory_reserved'):
                    if (type(value) is not int or not 0 <= value <= int(.25 * expected_total)
                            or (name == 'memory_reserved' and value < values['memory_allocated'])):
                        raise RuntimeError('invalid cleanup allocator counter')
                return value
            except BaseException:
                # Even an interruption before the reset must not report the
                # preceding phase's counters as this allocated reading.
                partial = ({'memory_allocated': value} if has_value else {}) if name == 'memory_allocated' else values
                on_partial(name, partial)
                raise
        return call
    cuda = SimpleNamespace(**{name: guarded(name, getattr(torch.cuda, name)) for name in
        ('is_initialized', 'synchronize', 'empty_cache', 'memory_allocated', 'memory_reserved')})
    wrapped = SimpleNamespace(__version__=torch.__version__, version=torch.version, cuda=cuda,
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=guarded('clear_cublas', torch._C._cuda_clearCublasWorkspaces)))
    return clear_cublas_after_drain(session=session, lease=lease, torch=wrapped)


def _release_clean_reservations(session, lease, budget, resident, workspace, authority, protocol):
    """Release inert accounting after final device cleanup; never retry uncertainty."""
    def check(expected):
        _check_cleanup_reservations(session, lease, budget, resident, workspace,
                                    authority, protocol, expected)
    try:
        check((resident, workspace))
        resident.release()
        check((workspace,))
        workspace.release()
        check(())
    except BaseException as exc:
        if _ROOTS[15] is None:
            _ROOTS[15] = exc
        else:
            _ROOTS[18] = exc
        authority.mark_unsafe()
        raise


def _cuda(session, lease, operation, /, *args, **kwargs):
    """Check the owner before each operation and stop after any uncertain call.

    A returned object remains in its preallocated nesting slot until that slot
    is reused. The single driver must transfer it before a later call reuses
    that slot. Deeper nesting is rejected before another operation begins.
    A rooted exception preserves its known Python references until process exit.
    Repeated asynchronous interruption and inaccessible runtime temporaries
    still require process-level recovery.
    """
    global _CUDA_DEPTH
    if _cuda_uncertain() or (_NO_CYCLIC_GC_REQUIRED and gc.isenabled()):
        raise RuntimeError('device operation is prohibited after uncertain failure')
    depth = _CUDA_DEPTH
    result = None
    try:
        _CUDA_PENDING[depth] = True
        if depth >= len(_CUDA_RESULTS):
            raise RuntimeError('device operation nesting exceeds custody capacity')
        _CUDA_DEPTH = depth + 1
        if _NO_CHILD_POPEN is not None:
            _assert_no_child_processes()
        if _NO_CYCLIC_GC_REQUIRED and gc.isenabled():
            raise RuntimeError('cyclic GC was enabled in the device owner')
        OwnerSession.check(session, lease)
        worker = _ROOTS[6]
        authority = _ROOTS[20]
        if authority is not None and authority.unsafe:
            raise RuntimeError('device owner authority is unsafe')
        if worker is not None and (worker.quarantined or worker.failed):
            raise RuntimeError('resident worker is unavailable')
        result = operation(*args, **kwargs)
        _CUDA_RESULTS[depth] = result
        OwnerSession.check(session, lease)
        if _NO_CHILD_POPEN is not None:
            _assert_no_child_processes()
        if _NO_CYCLIC_GC_REQUIRED and gc.isenabled():
            raise RuntimeError('cyclic GC was enabled during a device operation')
        if _cuda_uncertain():
            raise RuntimeError('device operation observed nested uncertain failure') from _ROOTS[15]
        if authority is not None and authority.unsafe:
            raise RuntimeError('device owner authority became unsafe')
        if worker is not None and (worker.quarantined or worker.failed):
            raise RuntimeError('resident worker became unavailable')
        _CUDA_PENDING[depth] = False
        return result
    except BaseException as exc:
        if depth < len(_CUDA_ERRORS):
            # If a one-shot interruption lands before the first store, the
            # finally block still roots the original error and its partials.
            try:
                _CUDA_ERRORS[depth] = exc
            finally:
                _CUDA_ERRORS[depth] = exc
        # A single interrupt before the normal slot assignment can leave a
        # returned object only in this frame. Retain it independently of the
        # exception traceback, which a caller may clear after receiving it.
        if result is not None and depth < len(_CUDA_RESULTS):
            _CUDA_RESULTS[depth] = result
        if _ROOTS[15] is None:
            _ROOTS[15] = exc
        else:
            _ROOTS[18] = exc
        raise
    finally:
        if depth < len(_CUDA_ERRORS) and _CUDA_PENDING[depth] and _CUDA_ERRORS[depth] is None:
            # A single interrupt can land on entry to the exception handler,
            # before its own custody store. Preserve both that interruption
            # and the original context before control reaches the caller.
            active = sys.exc_info()[1]
            if active is not None:
                _CUDA_INTERRUPTS[depth] = active
                _CUDA_ERRORS[depth] = active.__context__ if active.__context__ is not None else active
        _CUDA_DEPTH = depth


def _snapshot(session, lease, torch, expected_total, on_partial):
    """Reject a contradictory counter before issuing the next device query."""
    result = {}
    cap = int(.25 * expected_total)
    try:
        _cuda(session, lease, torch.cuda.synchronize, 0)
        pair = _cuda(session, lease, torch.cuda.mem_get_info, 0)
        if type(pair) is not tuple or len(pair) != 2:
            raise RuntimeError('invalid device free/total reading')
        free, total = pair
        result.update(device_free_bytes=free, device_total_bytes=total)
        if (type(free) is not int or type(total) is not int or total != expected_total
                or free < 0 or free > total):
            raise RuntimeError('invalid or changed device free/total reading')
        allocated = _cuda(session, lease, torch.cuda.memory_allocated, 0)
        result['allocated_bytes'] = allocated
        if type(allocated) is not int or not 0 <= allocated <= cap:
            raise RuntimeError('invalid allocator allocated reading')
        reserved = _cuda(session, lease, torch.cuda.memory_reserved, 0)
        result['reserved_bytes'] = reserved
        if (type(reserved) is not int or not allocated <= reserved <= cap
                or free + reserved > total):
            raise RuntimeError('invalid allocator reserved reading')
        peak_allocated = _cuda(session, lease, torch.cuda.max_memory_allocated, 0)
        result['peak_allocated_bytes'] = peak_allocated
        if type(peak_allocated) is not int or not allocated <= peak_allocated <= cap:
            raise RuntimeError('invalid allocator allocated peak')
        peak_reserved = _cuda(session, lease, torch.cuda.max_memory_reserved, 0)
        result['peak_reserved_bytes'] = peak_reserved
        if (type(peak_reserved) is not int or not max(reserved, peak_allocated) <= peak_reserved <= cap):
            raise RuntimeError('invalid allocator reserved peak')
        return result
    except BaseException:
        on_partial(result)
        raise


def _validate_memory_values(result):
    total, free = result['device_total_bytes'], result['device_free_bytes']
    if (any(type(value) is not int or value < 0 for value in result.values())
            or total == 0 or free > total or result['allocated_bytes'] > result['reserved_bytes']
            or free + result['reserved_bytes'] > total
            or result['reserved_bytes'] > total or result['peak_reserved_bytes'] > total
            or result['peak_allocated_bytes'] < result['allocated_bytes']
            or result['peak_reserved_bytes'] < result['reserved_bytes']
            or result['peak_allocated_bytes'] > result['peak_reserved_bytes']):
        raise RuntimeError('inconsistent CUDA memory counters')


def _record_memory(run_dir, snapshots, stage, value, expected_total, policy=None):
    record = {'stage': stage, **{key: item if type(item) is int else
              {'invalid_type': type(item).__name__} for key, item in value.items()}}
    if policy is not None:
        record.update(policy)
    # Preserve the observed record even if the series subsequently fails.
    _write_json(run_dir / f'memory-{len(snapshots):03d}.json', record)
    snapshots.append(record)
    _validate_memory_values(value)
    if value['device_total_bytes'] != expected_total:
        raise RuntimeError('device total changed during the owner run')
    if max(value['reserved_bytes'], value['peak_reserved_bytes']) > int(.25 * expected_total):
        raise RuntimeError('allocator reserved bytes exceed the fixed native cap')
    return value


def _check_memory_pair(before, after):
    if (after['peak_allocated_bytes'] < max(before['allocated_bytes'], before['peak_allocated_bytes'])
            or after['peak_reserved_bytes'] < max(before['reserved_bytes'], before['peak_reserved_bytes'])):
        raise RuntimeError('post-forward peak is below the reset baseline')


def _check_peak_reset(record):
    if (record['allocated_bytes'] != record['peak_allocated_bytes']
            or record['reserved_bytes'] != record['peak_reserved_bytes']):
        raise RuntimeError('peak reset baseline differs from current counters')


def _check_observed_headroom(record, protocol):
    total, free, reserved = (record['device_total_bytes'], record['device_free_bytes'],
                             record['reserved_bytes'])
    reserve = max(_GIB, (total * 15 + 99) // 100, protocol['minimum_reserve_bytes'])
    committed = protocol['resident_commitment_bytes'] + protocol['forward_commitment_bytes']
    estimate = min(free - reserve - committed, total - reserved - reserve - committed)
    if estimate < 0:
        raise RuntimeError('recorded device headroom cannot cover commitments and reserve')
    return {'reserve_bytes': reserve, 'committed_incremental_bytes': committed,
            'admissible_estimate_bytes': estimate}


def _check_workspace_peak(before, after, protocol):
    _check_memory_pair(before, after)
    allocated = after['peak_allocated_bytes'] - before['allocated_bytes']
    reserved = after['peak_reserved_bytes'] - before['reserved_bytes']
    if max(allocated, reserved) > protocol['forward_commitment_bytes']:
        raise RuntimeError('observed forward peak exceeds the fixed workspace commitment')
    return {'allocated_increment_bytes': allocated, 'reserved_increment_bytes': reserved}


def _require_group_escalation(completed):
    if not {'serial-0-0', 'serial-0-1'}.issubset(completed):
        raise RuntimeError('two completed single-row controls are required before the grouped stage')


def _resource_profile(memory, entries, protocol, runtime_sha256, source_hashes, checkpoint_sha256):
    """Derive a provisional, source-bound observation from retained counters."""
    if (type(memory) is not list or type(entries) is not list
            or [entry.get('request_id') if type(entry) is dict else None for entry in entries]
                != protocol['forward_order']
            or entries[0] != {'request_id': 'serial-0-0', 'entry_shapes': [[1, 4]]}
            or entries[1] != {'request_id': 'serial-0-1', 'entry_shapes': [[1, 4]]}
            or entries[2] != {'request_id': 'mixed-0', 'entry_shapes': [[2, 4]]}):
        raise RuntimeError('single-row observations must precede the first grouped stage')
    if (type(source_hashes) is not dict or type(runtime_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', runtime_sha256) is None
            or type(checkpoint_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', checkpoint_sha256) is None):
        raise RuntimeError('profile identity is invalid')
    fields = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
              'peak_allocated_bytes', 'peak_reserved_bytes')
    def capture(stage):
        matches = [record for record in memory if type(record) is dict and record.get('stage') == stage]
        if len(matches) != 1:
            raise RuntimeError('profile stage is missing or duplicated')
        value = {key: matches[0].get(key) for key in fields}
        _validate_memory_values(value)
        return value
    load_before, load_after = capture('before_model_load'), capture('after_model_load')
    total = load_before['device_total_bytes']
    if load_after['device_total_bytes'] != total:
        raise RuntimeError('profile device total changed')
    _check_observed_headroom(load_before, protocol)
    _check_observed_headroom(load_after, protocol)
    stages, seen = [], set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {'request_id', 'entry_shapes'}:
            raise RuntimeError('profile model entry is invalid')
        name, shapes = entry['request_id'], entry['entry_shapes']
        if (type(name) is not str or re.fullmatch(
                r'serial-[01]-[01]|mixed-[01]|heldout-serial-[01]-[01]|heldout-group-[01]', name) is None
                or shapes not in ([[1, 4]], [[2, 4]])):
            raise RuntimeError('profile model entry is invalid')
        expected = [2, 4] if name.startswith(('mixed-', 'heldout-group-')) else [1, 4]
        if shapes != [expected]:
            raise RuntimeError('profile shape differs from its fixed stage')
        path = 'broker' if name.startswith('heldout-group-') else 'control'
        key = (path, tuple(expected))
        before, after = capture('before-' + name), capture('after-' + name)
        if before['device_total_bytes'] != total or after['device_total_bytes'] != total:
            raise RuntimeError('profile device total changed')
        _check_peak_reset(before)
        before_headroom = _check_observed_headroom(before, protocol)
        after_headroom = _check_observed_headroom(after, protocol)
        incremental = _check_workspace_peak(before, after, protocol)
        stages.append({'request_id': name, 'shape': expected, 'execution_path': path,
                       'observation': 'repeat' if key in seen else 'first-for-path-and-shape',
                       'baseline': before, 'endpoint': after,
                       'observed_increment_bytes': incremental,
                       'headroom_before': before_headroom, 'headroom_after': after_headroom})
        seen.add(key)
    return {'schema': 'grouped-gpu-resource-profile-v1', 'status': 'provisional',
            'source_sha256': hashlib.sha256(_canonical(source_hashes)).hexdigest(),
            'protocol_sha256': source_hashes.get(_PROTOCOL, hashlib.sha256(_canonical(protocol)).hexdigest()),
            'runtime_sha256': runtime_sha256, 'checkpoint_sha256': checkpoint_sha256,
            'worker_path': 'GroupedWorkspaceWorker to custody-bound ResidentWorker with guarded RNG',
            'device_limits': {'allocator_fraction': protocol['allocator_fraction'],
                              'allocator_cap_bytes': int(protocol['allocator_fraction'] * total),
                              'resident_commitment_bytes': protocol['resident_commitment_bytes'],
                              'workspace_commitment_bytes': protocol['forward_commitment_bytes'],
                              'minimum_reserve_bytes': protocol['minimum_reserve_bytes']},
            'host_ledger': dict(protocol['host_ledger']),
            'control_host_ledger': dict(protocol['control_host_ledger']),
            'snapshot_reservation': dict(protocol['snapshot_reservation']),
            'worker_finite_scratch': dict(protocol['worker_finite_scratch']),
            'validation_scratch': dict(protocol['validation_scratch']),
            'comparison_scratch': dict(protocol['comparison_scratch']),
            'trace_limits': {'device_bytes': 0, 'host_bytes': 0},
            'model_load': {'before': load_before, 'after': load_after}, 'stages': stages,
            'limitation': 'point-in-time observations, not a worst-case or production capacity bound'}


def _check_committed_headroom(budget):
    snapshot = budget.snapshot()
    telemetry = snapshot.telemetry
    required = snapshot.reserve_bytes + snapshot.committed_incremental_bytes
    if (required > telemetry.device_free_bytes
            or telemetry.process_reserved_bytes + required > telemetry.device_total_bytes):
        raise RuntimeError('resident commitment and reserve no longer fit before model copy')
    return snapshot


def _check_serial_reference(batch_validation, value, width):
    if not batch_validation._valid_output(value, (1, 4, width), 'float32'):
        raise RuntimeError('serial reference is not finite detached CPU float32 of the fixed shape')


def _check_delivery(batch_validation, broker, case, handle, result, request_id, width):
    if (result.status != 'ok' or result.cancel_requested is not False or result.error_code is not None
            or result.team_id != case['owner'] or result.request_id != request_id
            or result.job_id != handle._job.job_id
            or broker.lookup(case['owner'], request_id) is not handle
            or not batch_validation._valid_output(result.logits, (1, 4, width), 'float32')):
        raise RuntimeError('held-out output shape, value or routing mismatch')


def _separation(batch_validation, serial, atol, rtol):
    """Both references must be distinguishable under the fixed tolerance."""
    comparisons = [batch_validation._comparison(serial[0], serial[1], atol, rtol),
                   batch_validation._comparison(serial[1], serial[0], atol, rtol)]
    return all(item['status'] == 'mismatch' for item in comparisons), comparisons


def _route_comparison(batch_validation, serial, result, row, atol, rtol):
    own = batch_validation._comparison(serial[row], result, atol, rtol)
    other = batch_validation._comparison(serial[1 - row], result, atol, rtol)
    return {'status': 'matched' if own['status'] == 'matched' and other['status'] == 'mismatch'
            else 'mismatch', 'own_reference': own, 'other_reference': other}


def _row_digest(row):
    records = [{'name': 'attention_mask', 'dtype': 'int64', 'shape': [1, 4],
                'values': [[1, 1, 1, 1]]},
               {'name': 'input_ids', 'dtype': 'int64', 'shape': [1, 4], 'values': [row]}]
    return hashlib.sha256(_canonical(records)).hexdigest()


def _bounded_bytes(path, limit):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('saved record exceeds its byte limit')
    return data


def _public_comparison(value):
    fields = ('status', 'max_abs_error', 'absolute_error_overflow', 'max_relative_error',
              'relative_error_nonfinite_reason')
    if type(value) is not dict or value.get('status') not in ('matched', 'mismatch'):
        raise RuntimeError('invalid comparison record')
    for key in ('max_abs_error', 'max_relative_error'):
        item = value.get(key)
        if item is not None and (type(item) not in (int, float) or not math.isfinite(item) or item < 0):
            raise RuntimeError('invalid comparison numeric value')
    if (type(value.get('absolute_error_overflow')) is not bool
            or value.get('relative_error_nonfinite_reason') not in (None, 'zero_reference', 'overflow')):
        raise RuntimeError('invalid comparison diagnostic')
    if ((value.get('max_abs_error') is None) != value['absolute_error_overflow']
            or (value.get('max_relative_error') is None) != (value['relative_error_nonfinite_reason'] is not None)):
        raise RuntimeError('inconsistent comparison diagnostic')
    if (value['max_abs_error'] == 0 and
            (value['status'] == 'mismatch' or value['max_relative_error'] != 0)):
        raise RuntimeError('zero-error comparison contradicts its status')
    return {key: value.get(key) for key in fields}


def _check_comparison_thresholds(value, atol, rtol):
    comparison = _public_comparison(value)
    # Strict inequalities avoid treating rounded equality as exact evidence.
    if comparison['status'] == 'mismatch':
        if (comparison['max_abs_error'] is not None and comparison['max_abs_error'] < atol
                or comparison['max_relative_error'] is not None and comparison['max_relative_error'] < rtol):
            raise RuntimeError('mismatch diagnostic is below the fixed tolerance')


def _public_memory(record):
    if type(record) is not dict:
        return None
    stage = record.get('stage')
    if type(stage) is not str or re.fullmatch(
            r'before_model_load|after_model_load|admission(?:-invalid)?|'
            r'(?:before|after)-(?:serial-[01]-[01]|mixed-[01]|heldout-serial-[01]-[01]|heldout-group-[01])',
            stage) is None:
        return None
    keys = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
            'peak_allocated_bytes', 'peak_reserved_bytes', 'reserve_bytes',
            'committed_incremental_bytes', 'requested_incremental_bytes', 'admissible_estimate_bytes')
    result = {'stage': stage}
    if record.get('partial') is True:
        result['partial'] = True
    for key in keys:
        if key in record:
            value = record[key]
            result[key] = value if type(value) is int and 0 <= value <= 2 ** 63 - 1 else None
    kind = record.get('admission_kind')
    if type(kind) is str and re.fullmatch(
            r'resident|precopy|workspace|owner-verify|'
            r'forward-(?:serial-[01]-[01]|mixed-[01]|heldout-serial-[01]-[01]|heldout-group-[01])',
            kind):
        result['admission_kind'] = kind
    return result


def _public_resource_profile(profile):
    counter_keys = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
                    'peak_allocated_bytes', 'peak_reserved_bytes')
    headroom_keys = ('reserve_bytes', 'committed_incremental_bytes', 'admissible_estimate_bytes')
    def select(value, keys):
        return {key: value[key] for key in keys}
    stages = []
    for stage in profile['stages']:
        stages.append({**select(stage, ('request_id', 'shape', 'execution_path', 'observation')),
                       'baseline': select(stage['baseline'], counter_keys),
                       'endpoint': select(stage['endpoint'], counter_keys),
                       'observed_increment_bytes': select(stage['observed_increment_bytes'],
                           ('allocated_increment_bytes', 'reserved_increment_bytes')),
                       'headroom_before': select(stage['headroom_before'], headroom_keys),
                       'headroom_after': select(stage['headroom_after'], headroom_keys)})
    ledger_keys = ('max_bytes', 'max_items', 'max_requests', 'max_operations')
    return {**select(profile, ('schema', 'status', 'source_sha256', 'protocol_sha256',
                               'runtime_sha256', 'checkpoint_sha256', 'worker_path', 'limitation')),
            'device_limits': select(profile['device_limits'],
                ('allocator_fraction', 'allocator_cap_bytes', 'resident_commitment_bytes',
                 'workspace_commitment_bytes', 'minimum_reserve_bytes')),
            'host_ledger': select(profile['host_ledger'], ledger_keys),
            'control_host_ledger': select(profile['control_host_ledger'], ledger_keys),
            'snapshot_reservation': select(profile['snapshot_reservation'], ('nbytes', 'items')),
            'worker_finite_scratch': select(profile['worker_finite_scratch'], ('nbytes', 'items')),
            'validation_scratch': select(profile['validation_scratch'], ('nbytes', 'items')),
            'comparison_scratch': select(profile['comparison_scratch'], ('chunk_elements', 'declared_max_bytes')),
            'trace_limits': select(profile['trace_limits'], ('device_bytes', 'host_bytes')),
            'model_load': {key: select(profile['model_load'][key], counter_keys) for key in ('before', 'after')},
            'stages': stages}


def _public_measurement(measurement):
    holdout = []
    for group in measurement['holdout']:
        cases = []
        for case in group['comparisons']:
            cases.append({key: case[key] for key in ('case_id', 'owner', 'request_id', 'job_id',
                                                       'input_row_sha256', 'status')})
            cases[-1]['own_reference'] = _public_comparison(case['own_reference'])
            cases[-1]['other_reference'] = _public_comparison(case['other_reference'])
        holdout.append({'group_id': group['group_id'], 'status': group['status'],
                        'entry_shapes': group['entry_shapes'],
                        'reference_separation': [_public_comparison(item) for item in group['reference_separation']],
                        'comparisons': cases})
    return {'status': 'passed', 'runtime_sha256': measurement['runtime_sha256'],
            'runtime_redacted': {key: measurement['runtime_redacted'][key] for key in sorted(_PUBLIC_RUNTIME_KEYS)},
            'sample_status': measurement['sample_status'],
            'sample_evidence_sha256': measurement['sample_evidence_sha256'],
            'resource_profile': _public_resource_profile(measurement['resource_profile']),
            'model_entries': measurement['model_entries'], 'holdout': holdout,
            'memory': [_public_memory(item) for item in measurement['memory']],
            'cleanup': {'before': measurement['cleanup']['before'],
                        'after': measurement['cleanup']['after'],
                        'accounting_leases_after': measurement['cleanup']['accounting_leases_after'],
                        'api': 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces'},
            'limitations': ['input-specific correctness development run; not a throughput benchmark',
                            'the full runtime manifest and selected device identity remain private',
                            'raw logits were not retained and numerical comparisons cannot be independently replayed',
                            'host ledger declarations do not bound process RSS or model-owned allocations']}


def _verify_promotion(run_dir, run_id, supervised, gpu_uuid, protocol, protocol_sha256, expected_assets,
                      source_hashes, *, source_commit, _expected_device='cuda:0'):
    """Reconcile retained evidence before a clean child can become a pass.

    This checks consistency of the trusted owner's records; it cannot replay
    floating-point comparisons because raw logits are not retained.
    """
    inventory = {}
    def read(name, limit=1024 * 1024):
        raw = _bounded_bytes(run_dir / name, limit)
        value = json.loads(raw)
        if type(value) is not dict:
            raise RuntimeError('saved evidence is not an object')
        inventory[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return value
    def require(condition):
        if not condition:
            raise RuntimeError('saved owner evidence is incomplete or inconsistent')

    owner = read('owner.json')
    started = read('started.json')
    manifest = read(_MANIFEST, 65536)
    candidate = read('candidate.json')
    observation = read('observations.json')
    runtime = read('private-runtime.json')
    _validate_runtime(runtime, protocol)
    selected = read('private-selected-device.json')
    source = read('protocol-source.json')
    assets = read('asset-verification.json')
    supervision = read('supervision.json')
    policy = read('frozen-policy.json')
    sample = read('sample-evidence.json')
    resource_profile = read('resource-profile.json')
    identity = {'run_id': run_id, 'supervisor_run_id': supervised.run_id, 'pid': supervised.pid}
    manifest_sha = inventory[_MANIFEST]['sha256']
    require(manifest == {'schema': 'grouped-gpu-source-manifest-v1', 'run_id': run_id,
                         'source_commit': source_commit, 'source_sha256': source_hashes})
    require(started == {'schema': 'grouped-gpu-start-v1', 'run_id': run_id,
                        'source_commit': source_commit, 'protocol_sha256': protocol_sha256,
                        'source_manifest_sha256': manifest_sha})
    require(owner == {'schema': 'grouped-gpu-owner-v1', **identity, 'source_manifest_sha256': manifest_sha})
    require(candidate.get('schema') == 'grouped-gpu-candidate-v1'
            and all(candidate.get(key) == item for key, item in identity.items()))
    require(_canonical(supervision) == _canonical({'schema': 'grouped-gpu-supervision-v1', **identity,
                            'exit_code': 0, 'clean': True, 'timed_out': False}))
    require(selected.get('schema') == 'grouped-gpu-selected-device-v1'
            and selected.get('run_id') == run_id and selected.get('physical_gpu_uuid') == gpu_uuid
            and runtime.get('physical_gpu_uuid') == gpu_uuid)
    protocol_raw = source.get('protocol_utf8')
    require(source.get('schema') == 'grouped-gpu-protocol-source-v1' and type(protocol_raw) is str
            and hashlib.sha256(protocol_raw.encode('utf-8')).hexdigest() == protocol_sha256
            and source.get('sha256') == protocol_sha256 and _canonical(json.loads(protocol_raw)) == _canonical(protocol))
    require(assets.get('schema') == 'grouped-gpu-assets-v1' and assets.get('run_id') == run_id
            and assets.get('verified') is True and assets.get('model') == protocol['model']
            and assets.get('revision') == protocol['revision'] and assets.get('assets') == expected_assets)
    measurement = candidate.get('result')
    require(type(measurement) is dict and measurement.get('status') == 'passed')
    runtime_digest = hashlib.sha256(_canonical(runtime)).hexdigest()
    require(measurement.get('runtime_sha256') == runtime_digest
            and measurement.get('runtime_redacted') == _public_runtime(runtime))
    require(observation.get('schema') == 'grouped-gpu-observations-v1'
            and observation.get('run_id') == run_id and observation.get('status') == 'passed'
            and observation.get('cuda_uuid_verified') is True
            and observation.get('runtime_sha256') == runtime_digest
            and observation.get('runtime_redacted') == measurement['runtime_redacted'])
    policy_sha = policy.get('protocol_sha256')
    require(type(policy_sha) is str and hashlib.sha256(_canonical(
        {key: value for key, value in policy.items() if key != 'protocol_sha256'})).hexdigest() == policy_sha)
    binding = policy.get('binding')
    asset_hashes = {item['name']: item['sha256'] for item in expected_assets}
    model_id = hashlib.sha256(_canonical(asset_hashes)).hexdigest()
    require(type(binding) is dict and binding.get('model_id') == model_id
            and binding.get('model_fingerprint') == model_id
            and binding.get('checkpoint') == protocol['revision']
            and binding.get('checkpoint_sha256') == asset_hashes.get('model.safetensors')
            and binding.get('adapter_sha256') == hashlib.sha256(_canonical(source_hashes)).hexdigest()
            and binding.get('runtime_id') == 'sha256:' + runtime_digest
            and binding.get('backend') == 'cuda-eager-development' and binding.get('device') == _expected_device
            and binding.get('input_schema') == [['attention_mask', 'int64'], ['input_ids', 'int64']]
            and binding.get('input_shape') == [1, 4] and binding.get('output_width') == protocol['model_output_width']
            and binding.get('output_dtype') == 'float32' and binding.get('seed') == protocol['seed']
            and binding.get('serial_atol') == protocol['absolute_tolerance']
            and binding.get('serial_rtol') == protocol['relative_tolerance'])
    require(policy.get('discovery_row_hashes') == sorted(_row_digest(row)
            for group in protocol['broker_holdout_groups'] for row in group))
    policy_groups = policy.get('groups')
    require(type(policy_groups) is list and len(policy_groups) == 2)
    for i, group in enumerate(policy_groups):
        require(group.get('group_id') == f'policy-{i}' and len(group.get('cases', [])) == 2)
        for j, case in enumerate(group['cases']):
            digest = _row_digest(protocol['policy_groups'][i][j])
            require(case == {'case_id': f'policy-{i}-{j}', 'owner': protocol['teams'][j],
                             'input_sha256': digest, 'row_sha256': [digest]})
    evidence_sha = sample.get('evidence_sha256')
    require(type(evidence_sha) is str and hashlib.sha256(_canonical(
        {key: value for key, value in sample.items() if key != 'evidence_sha256'})).hexdigest() == evidence_sha
        and sample.get('protocol_sha256') == policy_sha and evidence_sha == measurement.get('sample_evidence_sha256')
        and sample.get('status') == 'passed' and sample.get('complete') is True and sample.get('error') is None
        and sample.get('calls') == {'serial': 4, 'mixed': 2} and sample.get('tested_rows') == [2])
    require(measurement.get('sample_status') == 'passed' and observation.get('sample_status') == 'passed')
    require(len(sample.get('groups', [])) == 2 and all(len(group.get('cases', [])) == 2
            and all(case.get('status') == 'matched' for case in group['cases']) for group in sample['groups']))
    for group, frozen_group in zip(sample['groups'], policy_groups):
        require(group.get('group_id') == frozen_group['group_id'])
        for case, frozen_case in zip(group['cases'], frozen_group['cases']):
            require(all(case.get(key) == value for key, value in frozen_case.items()))
            _check_comparison_thresholds(case, protocol['absolute_tolerance'], protocol['relative_tolerance'])
    expected_entries = []
    for i in range(2):
        expected_entries.extend([{'request_id': f'serial-{i}-{j}', 'entry_shapes': [[1, 4]]} for j in range(2)])
        expected_entries.append({'request_id': f'mixed-{i}', 'entry_shapes': [[2, 4]]})
    for i in range(2):
        expected_entries.extend([{'request_id': f'heldout-serial-{i}-{j}', 'entry_shapes': [[1, 4]]}
                                 for j in range(2)])
        expected_entries.append({'request_id': f'heldout-group-{i}', 'entry_shapes': [[2, 4]]})
    require(_canonical(measurement.get('model_entries')) == _canonical(expected_entries)
            and _canonical(observation.get('model_entries')) == _canonical(expected_entries))
    holdout = measurement.get('holdout')
    require(type(holdout) is list and len(holdout) == 2 and observation.get('holdout') == holdout)
    seen_jobs = set()
    broker_prefix = None
    for i, group in enumerate(holdout):
        require(type(group) is dict and group.get('group_id') == f'holdout-{i}'
                and group.get('status') == 'compared' and group.get('entry_shapes') == [[2, 4]]
                and len(group.get('reference_separation', [])) == 2
                and all(item.get('status') == 'mismatch' for item in group['reference_separation'])
                and len(group.get('comparisons', [])) == 2)
        for item in group['reference_separation']:
            _check_comparison_thresholds(item, protocol['absolute_tolerance'], protocol['relative_tolerance'])
        for j, case in enumerate(group['comparisons']):
            job_id = case.get('job_id')
            require(case.get('case_id') == f'holdout-{i}-{j}' and case.get('owner') == protocol['teams'][j]
                    and case.get('request_id') == f'heldout-broker-{i}'
                    and case.get('input_row_sha256') == _row_digest(protocol['broker_holdout_groups'][i][j])
                    and type(job_id) is str and re.fullmatch(r'[0-9a-f]{32}-j-[0-3]', job_id) is not None
                    and job_id.endswith(f'-j-{i * 2 + j}')
                    and job_id not in seen_jobs
                    and case.get('status') == 'matched' and case.get('own_reference', {}).get('status') == 'matched'
                    and case.get('other_reference', {}).get('status') == 'mismatch')
            seen_jobs.add(job_id)
            _check_comparison_thresholds(case['own_reference'], protocol['absolute_tolerance'],
                                         protocol['relative_tolerance'])
            _check_comparison_thresholds(case['other_reference'], protocol['absolute_tolerance'],
                                         protocol['relative_tolerance'])
            if broker_prefix is None:
                broker_prefix = job_id[:32]
            require(job_id[:32] == broker_prefix)
    cleanup = measurement.get('cleanup', {})
    cleanup_before = cleanup.get('before')
    require(type(cleanup_before) is dict and set(cleanup_before) == {'allocated_bytes', 'reserved_bytes'}
            and all(type(value) is int and 0 <= value <= int(.25 * runtime['total_bytes'])
                    for value in cleanup_before.values())
            and cleanup_before['allocated_bytes'] <= cleanup_before['reserved_bytes']
            and cleanup.get('api') == 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces')
    cleanup_after = cleanup.get('after')
    require(type(cleanup_after) is dict and set(cleanup_after) == {'allocated_bytes', 'reserved_bytes'}
            and all(type(value) is int and value == 0 for value in cleanup_after.values()))
    require(type(cleanup.get('accounting_leases_after')) is int
            and cleanup['accounting_leases_after'] == 0)
    memory = measurement.get('memory')
    paths = sorted(run_dir.glob('memory-*.json'))
    require(type(memory) is list and 1 <= len(memory) <= 128 and len(paths) == len(memory)
            and observation.get('memory') == memory)
    for i, record in enumerate(memory):
        require(paths[i].name == f'memory-{i:03d}.json' and read(paths[i].name, 4096) == record)
        keys = ('device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
                'peak_allocated_bytes', 'peak_reserved_bytes')
        _validate_memory_values({key: record[key] for key in keys})
        require(record['device_total_bytes'] == runtime['total_bytes'])
        require(max(record['reserved_bytes'], record['peak_reserved_bytes']) <= int(.25 * runtime['total_bytes']))
        if record.get('stage') == 'admission':
            reserve = max(_GIB, (runtime['total_bytes'] * 15 + 99) // 100, protocol['minimum_reserve_bytes'])
            committed = record.get('committed_incremental_bytes')
            requested = record.get('requested_incremental_bytes')
            require(type(committed) is int and committed >= 0 and record.get('reserve_bytes') == reserve
                    and type(requested) is int and requested >= 0
                    and record.get('admissible_estimate_bytes') == max(0, min(
                        runtime['total_bytes'] - record['reserved_bytes'] - reserve - committed,
                        record['device_free_bytes'] - reserve - committed))
                    and record['device_free_bytes'] >= reserve + committed + requested
                    and record['device_total_bytes'] >= record['reserved_bytes'] + reserve + committed + requested
                    and record['admissible_estimate_bytes'] >= requested)
    expected_stages = ['before_model_load', 'admission', 'admission', 'after_model_load',
                       'admission', 'admission']
    for entry in expected_entries:
        expected_stages.extend(['before-' + entry['request_id'], 'admission', 'admission',
                                'after-' + entry['request_id']])
    require([record.get('stage') for record in memory] == expected_stages
            and memory[0]['device_free_bytes'] >= protocol['minimum_preload_free_bytes'])
    admissions = [record for record in memory if record['stage'] == 'admission']
    expected_admissions = [('resident', 0, protocol['resident_commitment_bytes']),
                           ('precopy', protocol['resident_commitment_bytes'], 0),
                           ('workspace', protocol['resident_commitment_bytes'], protocol['forward_commitment_bytes']),
                           ('owner-verify', protocol['resident_commitment_bytes'] +
                            protocol['forward_commitment_bytes'], 0)]
    for entry in expected_entries:
        expected_admissions.extend([('forward-' + entry['request_id'], protocol['resident_commitment_bytes'] +
                                     protocol['forward_commitment_bytes'], 0)] * 2)
    require([(item.get('admission_kind'), item.get('committed_incremental_bytes'),
              item.get('requested_incremental_bytes')) for item in admissions] == expected_admissions)
    for entry in expected_entries:
        before = 'before-' + entry['request_id']
        after = 'after-' + entry['request_id']
        stages = [record.get('stage') for record in memory]
        require(stages.count(before) == stages.count(after) == 1 and stages.index(before) < stages.index(after))
        start, end = stages.index(before), stages.index(after)
        _check_peak_reset(memory[start])
        for left, right in zip(memory[start:end], memory[start + 1:end + 1]):
            _check_memory_pair(left, right)
            _check_observed_headroom(right, protocol)
            _check_workspace_peak(memory[start], right, protocol)
    expected_profile = _resource_profile(memory, expected_entries, protocol, runtime_digest, source_hashes,
                                         asset_hashes['model.safetensors'])
    require(_canonical(resource_profile) == _canonical(expected_profile)
            and _canonical(measurement.get('resource_profile')) == _canonical(expected_profile))
    return inventory, measurement, assets


def _configure(session, lease, torch, protocol):
    if _cuda(session, lease, torch.cuda.is_initialized):
        raise RuntimeError('CUDA initialized before supervised owner validation')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.set_flush_denormal(False):
        raise RuntimeError('cannot establish denormal-preserving comparison')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.random.default_generator.manual_seed(protocol['seed'])


def _bind_device(session, lease, torch):
    if _cuda(session, lease, torch.cuda.device_count) != 1:
        raise RuntimeError('exactly one CUDA-visible device is required')
    properties = _cuda(session, lease, torch.cuda.get_device_properties, 0)
    _cuda(session, lease, OwnerSession.verify_runtime_binding, session, lease,
          runtime_uuid=_runtime_uuid(torch, properties.uuid), device_count=1)
    _check_device_total(properties.total_memory)
    return properties


def _check_device_total(total, expected=None):
    if (type(total) is not int or not 7 * _GIB <= total <= 9 * _GIB
            or (expected is not None and total != expected)):
        raise RuntimeError('device properties total changed or is outside the fixed contract')


_RUNTIME_KEYS = frozenset(('schema', 'physical_gpu_uuid', 'driver', 'name', 'total_bytes',
    'compute_capability', 'python', 'platform', 'torch', 'cuda_build', 'cudnn', 'transformers',
    'transformers_gpt2_source_sha256', 'torch_threads', 'torch_interop_threads', 'allocator_backend',
    'allocator_fraction', 'matmul_tf32', 'cudnn_tf32', 'cudnn_benchmark', 'cudnn_deterministic',
    'deterministic_algorithms', 'flash_sdp', 'memory_efficient_sdp', 'math_sdp', 'cpu_autocast',
    'cuda_autocast', 'cublas_workspace_config', 'float32_matmul_precision', 'conversion_overwrite',
    'conversion_swap', 'attention', 'cache', 'dtype'))
_PUBLIC_RUNTIME_KEYS = _RUNTIME_KEYS - {'name', 'python', 'platform'}


def _public_runtime(runtime):
    """Project only fixed runtime fields; free-form identifying strings stay private."""
    if type(runtime) is not dict or not _PUBLIC_RUNTIME_KEYS.issubset(runtime):
        raise RuntimeError('runtime projection is incomplete')
    return {key: ('redacted' if key == 'physical_gpu_uuid' else runtime[key])
            for key in sorted(_PUBLIC_RUNTIME_KEYS)}


def _public_smi(snapshot):
    if type(snapshot) is not dict:
        raise RuntimeError('invalid public device inventory')
    fields = ('total_bytes', 'free_bytes', 'used_bytes')
    if (any(type(snapshot.get(key)) is not int for key in fields)
            or not 7 * _GIB <= snapshot['total_bytes'] <= 9 * _GIB
            or any(not 0 <= snapshot[key] <= snapshot['total_bytes'] for key in fields[1:])
            or type(snapshot.get('driver_version')) is not str
            or re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}', snapshot['driver_version']) is None
            or len(snapshot['driver_version']) > 32
            or type(snapshot.get('timestamp_utc')) is not str
            or re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?\+00:00',
                            snapshot['timestamp_utc']) is None):
        raise RuntimeError('invalid public device inventory')
    try:
        datetime.fromisoformat(snapshot['timestamp_utc'])
    except ValueError as exc:
        raise RuntimeError('invalid public device inventory timestamp') from exc
    return {key: snapshot[key] for key in (*fields, 'driver_version', 'timestamp_utc')}


def _public_presence(value):
    if type(value) is dict:
        query, listed = value.get('query'), value.get('owner_pid_listed')
        if query == 'ok' and type(listed) is bool:
            return {'query': query, 'owner_pid_listed': listed}
        if query in ('unavailable', 'inconclusive', 'not_attempted') and listed is None:
            return {'query': query, 'owner_pid_listed': None}
    return {'query': 'inconclusive', 'owner_pid_listed': None}


def _public_error_type(value):
    allowed = {'RuntimeError', 'MemoryError', 'OSError', 'TimeoutError', 'ValueError',
               'OutOfMemoryError', 'WorkerOutOfMemory', 'GroupedBrokerUnavailable'}
    return value if type(value) is str and value in allowed else 'OtherError'


def _validate_runtime(runtime, protocol):
    if type(runtime) is not dict or set(runtime) != _RUNTIME_KEYS:
        raise RuntimeError('runtime manifest fields do not match the fixed contract')
    total = runtime['total_bytes']
    if type(total) is not int or not 7 * _GIB <= total <= 9 * _GIB:
        raise RuntimeError('runtime device total is outside the fixed contract')
    expected = {'schema': 'grouped-gpu-runtime-v1', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
        'transformers': '4.57.3', 'torch_threads': 1, 'torch_interop_threads': 1,
        'allocator_backend': 'native', 'allocator_fraction': int(protocol['allocator_fraction'] * total) / total,
        'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'deterministic_algorithms': True, 'flash_sdp': False, 'memory_efficient_sdp': False,
        'math_sdp': True, 'cpu_autocast': False, 'cuda_autocast': False,
        'cublas_workspace_config': ':4096:8', 'float32_matmul_precision': 'highest',
        'conversion_overwrite': False, 'conversion_swap': False, 'attention': 'eager',
        'cache': 'disabled', 'dtype': 'float32'}
    if any(_canonical(runtime[key]) != _canonical(value) for key, value in expected.items()):
        raise RuntimeError('runtime settings differ from the fixed contract')
    for key in ('physical_gpu_uuid', 'driver', 'name', 'python', 'platform'):
        if type(runtime[key]) is not str or not 1 <= len(runtime[key]) <= 256:
            raise RuntimeError('runtime identity field is invalid')
    if re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}', runtime['driver']) is None or len(runtime['driver']) > 32:
        raise RuntimeError('runtime driver version is invalid')
    if (type(runtime['compute_capability']) is not list or len(runtime['compute_capability']) != 2
            or any(type(value) is not int or not 0 <= value <= 99 for value in runtime['compute_capability'])
            or re.fullmatch(r'[0-9a-f]{64}', runtime['transformers_gpt2_source_sha256']) is None
            or (runtime['cudnn'] is not None and type(runtime['cudnn']) is not int)):
        raise RuntimeError('runtime backend identity is invalid')


def _convert_model(session, lease, model, *, device, dtype):
    converted = _cuda(session, lease, model.to, device=device, dtype=dtype)
    # The slot exists before CUDA work begins and retains even an unexpected
    # returned object. An interrupt before assignment is a process boundary.
    _ROOTS[16] = converted
    if converted is not model:
        raise RuntimeError('model conversion returned another object')


def _live_runtime(session, lease, torch, transformers, driver, expected_total):
    if _cuda(session, lease, torch.cuda.device_count) != 1:
        raise RuntimeError('exactly one CUDA-visible device is required')
    properties = _cuda(session, lease, torch.cuda.get_device_properties, 0)
    _check_device_total(properties.total_memory, expected_total)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    _cuda(session, lease, OwnerSession.verify_runtime_binding, session, lease,
          runtime_uuid=runtime_uuid, device_count=1)
    modeling = importlib.import_module('transformers.models.gpt2.modeling_gpt2')
    modeling_sha256 = hashlib.sha256(Path(modeling.__file__).read_bytes()).hexdigest()
    return {
        'schema': 'grouped-gpu-runtime-v1', 'physical_gpu_uuid': runtime_uuid,
        'driver': driver, 'name': properties.name, 'total_bytes': properties.total_memory,
        'compute_capability': [properties.major, properties.minor],
        'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'cuda_build': torch.version.cuda,
        'cudnn': _cuda(session, lease, torch.backends.cudnn.version), 'transformers': transformers.__version__,
        'transformers_gpt2_source_sha256': modeling_sha256,
        'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'allocator_backend': _cuda(session, lease, torch.cuda.get_allocator_backend),
        'allocator_fraction': _cuda(session, lease, torch.cuda.get_per_process_memory_fraction, 0),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
        'memory_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math_sdp': torch.backends.cuda.math_sdp_enabled(),
        'cpu_autocast': torch.is_autocast_enabled('cpu'),
        'cuda_autocast': torch.is_autocast_enabled('cuda'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'conversion_overwrite': torch.__future__.get_overwrite_module_params_on_conversion(),
        'conversion_swap': torch.__future__.get_swap_module_params_on_conversion(),
        'attention': 'eager', 'cache': 'disabled', 'dtype': 'float32',
    }


def _run(session, lease, torch, transformers, *, cache, protocol, run_dir, driver, source_hashes,
         run_id, _worker_device='cuda:0', _expected_total=None):
    """Execute the fixed owner workload, returning only after verified cleanup."""
    global _NO_CYCLIC_GC_REQUIRED
    if _worker_device not in ('cuda:0', 'cpu'):
        raise ValueError('unsupported internal test device')
    from . import batch_validation
    from .budgeted_worker import WorkloadProfile
    from .certified_batch_policy import CertifiedBatchPolicy, runtime_fingerprint
    from .grouped_broker import GroupedBroker, SnapshotReservation, ValidationReservation
    from .grouped_owner_worker import GroupedWorkspaceWorker, retire_grouped_owner
    from .grouped_worker import GroupedWorker
    from .host_budget import HostBudget
    from .memory_budget import CudaMemorySnapshot, MemoryBudget
    from .microbatch import BatchProfile, ForwardContract
    from .owner_completion import OwnerCompletionAuthority
    from .pretrained import ASSETS, REVISION, load_distilgpt2
    from .resident_worker import ForwardRequest, ResidentWorker

    gc.disable()
    _NO_CYCLIC_GC_REQUIRED = True
    for index, value in ((0, session), (1, lease)):
        if _ROOTS[index] is None:
            _ROOTS[index] = value
        elif _ROOTS[index] is not value:
            raise RuntimeError('device owner root changed')
    _assert_no_child_processes()
    OwnerSession.check(session, lease)
    _validate_workspace_binding(session, Path(__file__).resolve().parents[1])
    if (not _cuda(session, lease, torch.cuda.is_available)
            or _cuda(session, lease, torch.cuda.device_count) != 1):
        raise RuntimeError('the leased CUDA device is unavailable')
    properties = _cuda(session, lease, torch.cuda.get_device_properties, 0)
    if _expected_total is None and _worker_device != 'cpu':
        raise RuntimeError('the initial leased device total is required')
    _check_device_total(properties.total_memory, _expected_total)
    expected_total = properties.total_memory
    _cuda(session, lease, OwnerSession.verify_runtime_binding, session, lease,
          runtime_uuid=_runtime_uuid(torch, properties.uuid), device_count=1)
    if _cuda(session, lease, torch.cuda.get_allocator_backend) != 'native':
        raise RuntimeError('the fixed owner requires the native allocator')
    _cuda(session, lease, torch.cuda.set_per_process_memory_fraction, protocol['allocator_fraction'], 0)
    expected_fraction = int(protocol['allocator_fraction'] * properties.total_memory) / properties.total_memory
    if _cuda(session, lease, torch.cuda.get_per_process_memory_fraction, 0) != expected_fraction:
        raise RuntimeError('allocator cap did not take effect')
    _cuda(session, lease, torch.cuda.manual_seed, protocol['seed'])
    runtime = _live_runtime(session, lease, torch, transformers, driver, expected_total)
    _validate_runtime(runtime, protocol)
    runtime_digest = runtime_fingerprint(runtime)
    _write_json(run_dir / 'private-runtime.json', runtime)
    public_runtime = _public_runtime(runtime)
    snapshots = []
    peak_start = None
    admission_context = ('resident', protocol['resident_commitment_bytes'])

    def record(label, value, policy=None):
        _record_memory(run_dir, snapshots, label, value, properties.total_memory, policy)
        if peak_start is not None:
            _check_observed_headroom(snapshots[-1], protocol)
            if len(snapshots) > peak_start + 1:
                _check_memory_pair(snapshots[-2], snapshots[-1])
                _check_workspace_peak(snapshots[peak_start], snapshots[-1], protocol)
        return value

    def reset_peaks():
        nonlocal peak_start
        peak_start = None
        _cuda(session, lease, torch.cuda.reset_peak_memory_stats, 0)
        peak_start = len(snapshots)

    def partial_memory(label, value):
        record = {'stage': label, 'partial': True, **{key: item if type(item) is int else
                  {'invalid_type': type(item).__name__} for key, item in value.items()}}
        _write_json(run_dir / f'memory-{len(snapshots):03d}.json', record)
        snapshots.append(record)

    def observe(label, policy=None):
        value = _snapshot(session, lease, torch, properties.total_memory,
                          lambda partial: partial_memory(label, partial))
        return record(label, value, policy)

    def telemetry():
        # This callback runs while MemoryBudget holds its re-entrant policy
        # lock, immediately before an admission decision.
        value = _snapshot(session, lease, torch, properties.total_memory,
                          lambda partial: partial_memory('admission-invalid', partial))
        try:
            _validate_memory_values(value)
        except BaseException:
            # Keep complete but inconsistent counter sets before rejecting.
            _record_memory(run_dir, snapshots, 'admission-invalid', value, properties.total_memory)
            raise
        snapshot = CudaMemorySnapshot(value['device_free_bytes'], value['device_total_bytes'],
                                      value['allocated_bytes'], value['reserved_bytes'])
        reserve = budget._reserve(snapshot)
        committed = budget._committed_bytes()
        if admission_context is None:
            raise RuntimeError('unlabelled device admission observation')
        policy = {'admission_kind': admission_context[0], 'requested_incremental_bytes': admission_context[1],
                  'reserve_bytes': reserve, 'committed_incremental_bytes': committed,
                  'admissible_estimate_bytes': max(0, min(
                      snapshot.device_total_bytes - snapshot.process_reserved_bytes - reserve - committed,
                      snapshot.device_free_bytes - reserve - committed))}
        record('admission', value, policy)
        return snapshot

    before = observe('before_model_load')
    _check_observed_headroom(before, protocol)
    if before['device_free_bytes'] < protocol['minimum_preload_free_bytes']:
        raise RuntimeError('device free memory is below the fixed preload threshold')
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=0, max_host_trace_bytes=0,
                          minimum_reserve_bytes=protocol['minimum_reserve_bytes'])
    _ROOTS[2] = budget
    resident_lease = budget.acquire(protocol['resident_commitment_bytes'], max_gpu_trace_bytes=0,
                                    max_host_trace_bytes=0)
    _ROOTS[3] = resident_lease
    admission_context = None
    bundle = load_distilgpt2(cache_dir=str(cache), allow_download=False)
    _ROOTS[4] = bundle
    model = bundle.model
    if (model.training is not False or model.config.use_cache is not False
            or getattr(model.config, '_attn_implementation', None) != 'eager'
            or model.config.vocab_size != protocol['model_output_width']):
        raise RuntimeError('loaded model differs from the fixed contract')
    model_ref = weakref.ref(model)
    OwnerSession.check(session, lease)
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise RuntimeError('unsupported model-conversion future flags')
    admission_context = ('precopy', 0)
    _check_committed_headroom(budget)
    admission_context = None
    _convert_model(session, lease, model, device=_worker_device, dtype=torch.float32)
    _cuda(session, lease, torch.cuda.synchronize, 0)
    parameter_count = 0
    for parameter in model.parameters():
        if (parameter.device.type != ('cuda' if _worker_device == 'cuda:0' else 'cpu')
                or (_worker_device == 'cuda:0' and parameter.device.index != 0)
                or parameter.dtype != torch.float32):
            raise RuntimeError('model parameters differ from the fixed device contract')
        parameter_count += parameter.numel()
    parameter = None
    if (parameter_count != bundle.metadata['parameter_count']
            or (_worker_device == 'cuda:0' and parameter_count <= 0)):
        raise RuntimeError('model parameter count differs from the verified CPU load')
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError('tied embedding/output weights were not preserved')
    _check_observed_headroom(observe('after_model_load'), protocol)
    # Keep both commitments charged across every forward and guarded cleanup;
    # they are released only after all tracked storage is drained and gone.
    admission_context = ('workspace', protocol['forward_commitment_bytes'])
    workspace_lease = budget.acquire(protocol['forward_commitment_bytes'], max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
    _ROOTS[19] = workspace_lease
    admission_context = None
    model_id = hashlib.sha256(_canonical(bundle.metadata['asset_sha256'])).hexdigest()
    resident = ResidentWorker(model_id, model, device=_worker_device, max_batch_size=2,
        max_sequence_length=4, max_input_bytes=protocol['resident_input_limit_bytes'],
        max_output_bytes=protocol['resident_output_limit_bytes'])
    _ROOTS[5] = resident
    schemas = (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64'))
    profiles = tuple(WorkloadProfile('grouped-dev-' + str(rows), model_id, _worker_device, (rows, 4),
        schemas, protocol['resident_output_limit_bytes'], protocol['forward_commitment_bytes'])
        for rows in (1, 2))
    def owner_guard():
        _assert_no_child_processes()
        if _cuda_uncertain() or gc.isenabled():
            raise RuntimeError('device owner has a pending uncertain failure')
        OwnerSession.check(session, lease)
        _assert_no_child_processes()
        if _cuda_uncertain() or gc.isenabled():
            raise RuntimeError('device owner became uncertain during its lease check')

    authority = OwnerCompletionAuthority(device_guard=owner_guard)
    _ROOTS[20] = authority
    admission_context = ('owner-verify', 0)
    worker = GroupedWorkspaceWorker(resident, budget, resident_lease, workspace_lease,
                                    profiles=profiles, authority=authority, seed=protocol['seed'])
    _ROOTS[6] = worker
    admission_context = None
    contract = ForwardContract(model_id, REVISION, 'grouped-gpu-development-v3', _worker_device,
        torch.float32, 'sha256:' + runtime_digest, protocol['seed'], True, True,
        'eval', 'none', 'disabled', 'none', False, False)
    input_schema = (('attention_mask', torch.int64), ('input_ids', torch.int64))
    output_bytes = 2 * 4 * protocol['model_output_width'] * 4
    profile = BatchProfile('grouped-gpu-development-v3', 'gpu-sampled-v3', contract,
        input_schema, (1, 4), torch.float32, protocol['model_output_width'],
        protocol['absolute_tolerance'], protocol['relative_tolerance'], (2,),
        2, 1, 2, 8, 128, output_bytes, 128 + output_bytes)
    identity = {'model_fingerprint': model_id, 'checkpoint_sha256': ASSETS['model.safetensors'][1],
                'adapter_sha256': hashlib.sha256(_canonical(source_hashes)).hexdigest(),
                'backend': 'cuda-eager-development'}
    binding = batch_validation.binding_from_profile(profile, **identity)

    # Reserve before constructing any policy or held-out tensor. The separate
    # envelope does not claim Python metadata, RSS or model-owned allocations.
    control_budget = HostBudget(**protocol['control_host_ledger'])
    _ROOTS[7] = control_budget
    control_guard = control_budget.request_guard('validation-control')
    _ROOTS[8] = control_guard
    control_guard.open()
    control_request = control_guard.adopt()
    control_claim_guard = control_request.claim_guard(
        'validation-control', 'references-and-scratch',
        protocol['control_host_ledger']['max_bytes'], items=protocol['control_host_ledger']['max_items'])
    _ROOTS[9] = control_claim_guard
    control_claim_guard.reserve()
    control_claim = control_claim_guard.adopt()
    _ROOTS[10] = control_claim

    def inputs(row):
        return {'input_ids': torch.tensor([row], dtype=torch.int64),
                'attention_mask': torch.ones((1, 4), dtype=torch.int64)}

    def groups(rows, prefix):
        return [{'group_id': f'{prefix}-{i}', 'cases': [
            {'case_id': f'{prefix}-{i}-{j}', 'owner': protocol['teams'][j], 'inputs': inputs(row)}
            for j, row in enumerate(group)]} for i, group in enumerate(rows)]

    sample_groups = groups(protocol['policy_groups'], 'policy')
    heldout_groups = groups(protocol['broker_holdout_groups'], 'holdout')
    excluded = [digest for group in heldout_groups for case in group['cases']
                for digest in batch_validation.input_row_digests(binding, case['inputs'])]
    frozen = batch_validation.freeze_protocol(binding, sample_groups, discovery_row_hashes=excluded)
    _write_json(run_dir / 'frozen-policy.json', frozen)
    model_code = type(model).forward.__code__
    observed = []
    entries = []
    completed_single_controls = set()

    def model_observer(frame, event, arg):
        if event == 'call' and frame.f_code is model_code:
            value = frame.f_locals.get('input_ids')
            observed.append(list(value.shape) if value is not None else None)

    def measured_forward(values, *, request_id, seed):
        nonlocal admission_context
        try:
            if tuple(values['input_ids'].shape) == (2, 4):
                _require_group_escalation(completed_single_controls)
            _cuda(session, lease, torch.cuda.synchronize, 0)
            reset_peaks()
            before_forward = observe('before-' + request_id)
            _check_peak_reset(before_forward)
            count = len(observed)
            admission_context = ('forward-' + request_id, 0)
            sys.setprofile(model_observer)
            try:
                producer = _cuda(session, lease, worker.execute_control,
                                 ForwardRequest(request_id, values, seed))
                _ROOTS[21] = producer
                _cuda(session, lease, worker.confirm_control_result, producer)
                result = producer.logits
                _ROOTS[21] = producer = None
            finally:
                sys.setprofile(None)
                admission_context = None
            if observed[count:] != [list(values['input_ids'].shape)]:
                raise RuntimeError('serial or sampled call did not enter the model exactly once')
            entries.append({'request_id': request_id, 'entry_shapes': observed[count:]})
            after_forward = observe('after-' + request_id)
            _check_memory_pair(before_forward, after_forward)
            if request_id in ('serial-0-0', 'serial-0-1'):
                _check_serial_reference(batch_validation, result, protocol['model_output_width'])
                completed_single_controls.add(request_id)
            return result
        except BaseException as exc:
            # Validation may catch ordinary exceptions and reduce them to a
            # status. Keep the original traceback and refuse further CUDA work.
            _ROOTS[14] = exc
            if _ROOTS[15] is None:
                _ROOTS[15] = exc
            raise

    report = batch_validation.run_validation(frozen, binding, sample_groups, measured_forward,
                                              expected_protocol_sha256=frozen['protocol_sha256'])
    _write_json(run_dir / 'sample-evidence.json', report)
    if _cuda_uncertain():
        raise RuntimeError('sampled forward failed; device owner requires replacement')
    if worker.failed or worker.quarantined:
        raise RuntimeError('worker became unavailable during sampled validation')
    holdout = []
    broker = grouped = None
    host = None
    if report['status'] == 'passed' and report['complete']:
        policy = CertifiedBatchPolicy(profile, frozen, report,
            expected_protocol_sha256=frozen['protocol_sha256'], evidence_sha256=report['evidence_sha256'],
            minimum_groups=2, max_atol=protocol['absolute_tolerance'], max_rtol=protocol['relative_tolerance'],
            runtime_manifest=runtime, **identity)
        host = HostBudget(**protocol['host_ledger'])
        _ROOTS[11] = host
        grouped = GroupedWorker(worker, (profile,), host_budget=host,
            finite_scratch_bytes={profile.profile_id: protocol['worker_finite_scratch']['nbytes']},
            finite_scratch_items={profile.profile_id: protocol['worker_finite_scratch']['items']})
        reservation_id = hashlib.sha256(_canonical(protocol)).hexdigest()
        snap = SnapshotReservation(policy.compatibility_key, reservation_id,
            protocol['snapshot_reservation']['nbytes'], protocol['snapshot_reservation']['items'])
        check = ValidationReservation(policy.compatibility_key, 2, reservation_id,
            protocol['validation_scratch']['nbytes'], protocol['validation_scratch']['items'])
        def collect_runtime():
            manifest = _live_runtime(session, lease, torch, transformers, driver, expected_total)
            _validate_runtime(manifest, protocol)
            return manifest
        broker = GroupedBroker(grouped, profile, policy, host_budget=host, identity=identity,
            collect_runtime=collect_runtime,
            snapshot_reservation=snap, validation_reservations=(check,), max_pending=4,
            max_pending_per_team=2, max_coalesce_wait=.1, max_lifetime_jobs=8)
        _ROOTS[12] = broker
        for i, group in enumerate(heldout_groups):
            cases = group['cases']
            serial = [measured_forward(case['inputs'], request_id=f'heldout-serial-{i}-{j}',
                                       seed=protocol['seed']) for j, case in enumerate(cases)]
            for reference in serial:
                _check_serial_reference(batch_validation, reference, protocol['model_output_width'])
            distinct, separation = _separation(batch_validation, serial,
                protocol['absolute_tolerance'], protocol['relative_tolerance'])
            if not distinct:
                holdout.append({'group_id': group['group_id'], 'status': 'uninformative',
                                'reference_separation': separation, 'comparisons': []})
                del serial, reference
                continue
            _require_group_escalation(completed_single_controls)
            request_id = f'heldout-broker-{i}'
            handles = [broker.submit(case['owner'], ForwardRequest(request_id, case['inputs'], protocol['seed']))
                       for case in cases]
            if (len({handle._job.job_id for handle in handles}) != 2
                    or any(broker.lookup(case['owner'], request_id) is not handle
                           for case, handle in zip(cases, handles))):
                raise RuntimeError('broker scoped identities do not match submitted handles')
            _cuda(session, lease, torch.cuda.synchronize, 0)
            reset_peaks()
            before_forward = observe(f'before-heldout-group-{i}')
            _check_peak_reset(before_forward)
            count = len(observed)
            admission_context = (f'forward-heldout-group-{i}', 0)
            sys.setprofile(model_observer)
            try:
                if not _cuda(session, lease, broker.run_once, timeout=3):
                    raise RuntimeError('held-out cohort did not dispatch')
            finally:
                sys.setprofile(None)
                admission_context = None
            entry_shapes = observed[count:]
            if entry_shapes != [[2, 4]]:
                raise RuntimeError('group call did not enter the model exactly once')
            entries.append({'request_id': f'heldout-group-{i}', 'entry_shapes': entry_shapes})
            after_forward = observe(f'after-heldout-group-{i}')
            _check_memory_pair(before_forward, after_forward)
            tickets = [handle.delivery() for handle in handles]
            results = [ticket.wait(3) for ticket in tickets]
            comparisons = []
            for j, (case, handle, result) in enumerate(zip(cases, handles, results)):
                _check_delivery(batch_validation, broker, case, handle, result, request_id,
                                protocol['model_output_width'])
                comparisons.append({'case_id': case['case_id'], 'owner': case['owner'],
                    'request_id': request_id, 'job_id': result.job_id,
                    'input_row_sha256': batch_validation.input_row_digests(binding, case['inputs'])[0],
                    **_route_comparison(batch_validation, serial, result.logits, j,
                        protocol['absolute_tolerance'], protocol['relative_tolerance'])})
            if len({result.job_id for result in results}) != 2:
                raise RuntimeError('delivered job identities are not distinct')
            for ticket, result in zip(tickets, results):
                ticket.confirm(result)
            holdout.append({'group_id': group['group_id'], 'status': 'compared',
                            'entry_shapes': entry_shapes, 'reference_separation': separation,
                            'comparisons': comparisons})
            del result, reference, results, serial
            for ticket in tickets:
                ticket.release()
            for handle in handles:
                handle.dispose()
            if host.snapshot().used_bytes or worker.quarantined:
                raise RuntimeError('host claims or device state remain uncertain after a cohort')
        if not _cuda(session, lease, broker.close):
            raise RuntimeError('grouped broker did not close cleanly')
    status = ('passed' if report['status'] == 'passed' and len(holdout) == 2
              and all(group['status'] == 'compared' and len(group['comparisons']) == 2 for group in holdout)
              and all(case['status'] == 'matched' for group in holdout for case in group['comparisons'])
              else 'failed')
    _write_json(run_dir / 'observations.json', {'schema': 'grouped-gpu-observations-v1', 'run_id': run_id,
                'sample_status': report['status'], 'holdout': holdout, 'model_entries': entries,
                'memory': snapshots, 'runtime_sha256': runtime_digest,
                'runtime_redacted': public_runtime,
                'exclusion_note': 'discovery_row_hashes are reserved later broker-row exclusions in this run',
                'cuda_uuid_verified': True, 'status': status})
    # Release references only after all work was synchronized
    # and both independent host ledgers have been disposed by their owners.
    sample_groups = heldout_groups = group = cases = case = None
    serial = reference = results = handles = tickets = ticket = handle = None
    _assert_disposable_return_slots()
    _cuda(session, lease, control_claim.release)
    _cuda(session, lease, control_request.close)
    for label, host_budget in (('control', control_budget), ('broker', host)):
        if host_budget is not None:
            state = host_budget.snapshot()
            if state.used_bytes or state.used_items or state.requests or state.claims or state.operations:
                raise RuntimeError(label + ' host custody remains after validation')
    if worker.quarantined or worker.failed:
        raise RuntimeError('worker is not healthy before teardown')
    if broker is None:
        raise RuntimeError('validation did not establish a grouped owner to retire')
    _cuda(session, lease, worker.close)
    _cuda(session, lease, resident.close)
    if not _cuda(session, lease, retire_grouped_owner, worker, broker, callers_quiescent=True):
        raise RuntimeError('grouped owner did not retire exactly once')
    _assert_disposable_return_slots()
    def check_cleanup_reservations():
        _check_cleanup_reservations(session, lease, budget, resident_lease, workspace_lease,
                                    authority, protocol, (resident_lease, workspace_lease))
    # The retired worker and broker are kept rooted until process exit. The
    # worker's otherwise inert model reference must be detached explicitly:
    # the broker/handle graph contains cycles while cyclic GC is disabled.
    try:
        check_cleanup_reservations()
        if (worker._teardown_broker is not broker or worker._model is not model
                or _ROOTS[4] is not bundle or bundle.model is not model or _ROOTS[16] is not model
                or resident._model is not None or not authority._worker_retired):
            raise RuntimeError('retired grouped owner retained unexpected model custody')
        # Another reference remains rooted in both the bundle and slot 16, so
        # the authority lock never surrounds a model's final-reference drop.
        with authority._decision_lock:
            if authority.unsafe or not authority._worker_retired or worker._model is not model:
                raise RuntimeError('owner became unsafe before model detachment')
            worker._model = None
        check_cleanup_reservations()
        for index in (4, 16):
            check_cleanup_reservations()
            _ROOTS[index] = None
            check_cleanup_reservations()
        bundle = None
        check_cleanup_reservations()
        model = None
        check_cleanup_reservations()
    except BaseException as exc:
        if _ROOTS[15] is None:
            _ROOTS[15] = exc
        else:
            _ROOTS[18] = exc
        authority.mark_unsafe()
        raise
    if model_ref() is not None:
        raise RuntimeError('model object remains alive without cyclic collection')
    def partial_cleanup(stage, values):
        _write_json(run_dir / 'cleanup-partial.json', {'stage': stage, 'partial': True,
            'counters': {key: item if type(item) is int else {'invalid_type': type(item).__name__}
                         for key, item in values.items()}})
    cleanup = _cuda(session, lease, _guarded_cleanup, session, lease, torch, check_cleanup_reservations,
                    expected_total, partial_cleanup)
    if cleanup['after'] != {'allocated_bytes': 0, 'reserved_bytes': 0}:
        raise RuntimeError('CUDA allocator still owns bytes after teardown')
    check_cleanup_reservations()
    resource_profile = None
    if status == 'passed':
        resource_profile = _resource_profile(snapshots, entries, protocol, runtime_digest, source_hashes,
                                             ASSETS['model.safetensors'][1])
        _write_json(run_dir / 'resource-profile.json', resource_profile)
    # All device operations and actual storage custody have finished. These
    # are now inert accounting records. Keep the physical lease and authority
    # rooted and issue no further CUDA calls after the first release.
    _release_clean_reservations(session, lease, budget, resident_lease, workspace_lease, authority, protocol)
    cleanup['accounting_leases_after'] = 0
    return {'status': status, 'runtime_sha256': runtime_digest, 'sample_status': report['status'],
            'sample_evidence_sha256': report['evidence_sha256'], 'holdout': holdout, 'model_entries': entries,
            'memory': snapshots, 'cleanup': cleanup, 'runtime_redacted': public_runtime,
            'resource_profile': resource_profile,
            'limitations': ['development correctness check, not latency or throughput',
                            'sample evidence is specific to these inputs and runtime',
                            'discovery_row_hashes are reserved later broker-row exclusions, not independent policy evidence',
                            'raw logits are not retained; the physical GPU UUID is absent from the public result',
                            'full runtime manifest is private; redacted fields cannot reconstruct its digest',
                            'comparison summaries cannot be independently recomputed without raw logits',
                            'host claims do not cap Python objects, RSS or model-owned allocations',
                            'driver and device lease are process-scoped; no shared-service transition is proven',
                            'repeated async interruption and return-before-assignment require process recovery']}


def _owner_main(run_dir, cache_dir, run_id, commit, protocol_sha256, driver, source_manifest_sha256=None):
    """Run only under the real supervisor; never attest after an exception."""
    global _NO_CYCLIC_GC_REQUIRED
    try:
        _require_bootstrap()
        _require_supervisor()
        _require_gpu_boundary()
        _claim_attempt()
        if any(name == root or name.startswith(root + '.') for name in sys.modules
               for root in ('torch', 'transformers', 'huggingface_hub')):
            raise RuntimeError('CUDA-capable modules were imported before the device lease')
        _install_no_child_processes()
        _assert_no_child_processes()
        session = OwnerSession.from_environment()
        _ROOTS[0] = session
        repo = Path(__file__).resolve().parents[1]
        _validate_workspace_spec(session, repo)
        sources = _verify_child_sources(repo, Path(run_dir), run_id, commit, source_manifest_sha256)
        # The child takes independent leases before device acquisition. If the
        # controller dies, its handles may close while these remain held until
        # this child exits; a second verification binds the protected bytes.
        _protect_sources(repo, sources)
        _protect_manifest(repo, sources, Path(run_dir))
        if _verify_child_sources(repo, Path(run_dir), run_id, commit, source_manifest_sha256) != sources:
            raise RuntimeError('source bytes changed while acquiring child protection')
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _ROOTS[1] = lease
        OwnerSession.check(session, lease)
        _validate_workspace_binding(session, repo)
        if sources[_PROTOCOL] != protocol_sha256:
            raise RuntimeError('protocol differs from the parent-frozen digest')
        protocol = _protocol(repo / _PROTOCOL, protocol_sha256)
        _write_json(Path(run_dir) / 'owner.json', {'schema': 'grouped-gpu-owner-v1',
                    'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
                    'source_manifest_sha256': source_manifest_sha256})
        gc.disable()
        _NO_CYCLIC_GC_REQUIRED = True
        OwnerSession.check(session, lease)
        import torch
        if torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8':
            raise RuntimeError('PyTorch or CUDA build differs from the fixed contract')
        _configure(session, lease, torch, protocol)
        bound_properties = _bind_device(session, lease, torch)
        import transformers
        result = _run(session, lease, torch, transformers, cache=Path(cache_dir), protocol=protocol,
                      run_dir=Path(run_dir), driver=driver, source_hashes=sources, run_id=run_id,
                      _expected_total=bound_properties.total_memory)
        candidate = {'schema': 'grouped-gpu-candidate-v1', 'run_id': run_id,
                     'supervisor_run_id': session.run_id, 'pid': os.getpid(), 'result': result}
        _write_json(Path(run_dir) / 'candidate.json', candidate)
        if _cuda_uncertain():
            raise RuntimeError('device owner remains uncertain before clean attestation')
        _check_cleanup_reservations(session, lease, _ROOTS[2], _ROOTS[3], _ROOTS[19],
                                    _ROOTS[20], protocol, ())
        OwnerSession.attest_clean(session, lease)
        return 0
    except _AttemptAlreadyStarted:
        raise
    except BaseException as exc:
        _ROOTS[13] = exc
        try:
            _write_json(Path(run_dir) / 'failure.json', {'schema': 'grouped-gpu-failure-v1',
                        'run_id': run_id, 'error_type': type(exc).__name__, 'status': 'quarantined'})
        except BaseException as write_error:
            if not isinstance(write_error, Exception):
                _ROOTS[17] = write_error
                raise write_error from exc
        if not isinstance(exc, Exception):
            raise
        return 1


def _source_guard_types(repo, expected_sha256):
    source = repo / 'scripts/v2_source_guards.py'
    raw = _bounded_bytes(source, 131072)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError('source-guard code differs from the verified checkout')
    namespace = {'__name__': '_grouped_process_job', '__file__': str(source)}
    exec(compile(raw, str(source), 'exec'), namespace)
    return namespace


def _protect_sources(repo, sources):
    guard_type = _source_guard_types(repo, sources['scripts/v2_source_guards.py'])['SourceFileLease']
    guard = guard_type.__new__(guard_type)
    _ROOTS[23] = guard
    guard_type.__init__(guard, repo, [repo / name for name in _SOURCE_PATHS])
    guard.__enter__()
    return guard


def _protect_manifest(repo, sources, run_dir):
    guard_type = _source_guard_types(repo, sources['scripts/v2_source_guards.py'])['SourceFileLease']
    guard = guard_type.__new__(guard_type)
    _ROOTS[24] = guard
    manifest = run_dir / _MANIFEST
    # Lock only the dedicated manifest directory: the sibling run directory
    # must remain writable for create-only hard-link evidence publication.
    guard_type.__init__(guard, manifest.parent, [manifest])
    guard.__enter__()
    return guard


def _new_process_job(repo, expected_sha256):
    namespace = _source_guard_types(repo, expected_sha256)
    job_type = namespace['WindowsJob']
    job = job_type.__new__(job_type)
    _ROOTS[22] = job
    job_type.__init__(job)
    return job


def _supervise_owner(command, *, identity, repo, environment, job_source_sha256):
    job = _new_process_job(repo, job_source_sha256)
    return run_supervised(command, device_lease_path=identity.lock_path, marker_path=identity.marker_path,
                          identity=identity, timeout=_OWNER_TIMEOUT_SECONDS, lease_timeout=30, cwd=repo,
                          env=environment, inherit_environment=False, require_zero_exit=True, process_job=job)


def run_development(*, workspace_root, cache_dir, output, site_packages, smi):
    """Supervise the fixed development owner and publish a redacted record."""
    _require_bootstrap()
    _require_supervisor()
    _require_gpu_boundary()
    _claim_attempt()
    _verify_controller_import_mode()
    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    cache = Path(cache_dir).resolve(strict=True)
    site = Path(site_packages).resolve(strict=True)
    if str(site) != os.environ.get('LOOKING_GLASS_GROUPED_SITE'):
        raise RuntimeError('owner dependencies must match the bootstrap dependency directory')
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError('output must be an absolute new path with an existing parent')
    with tempfile.NamedTemporaryFile(prefix='.grouped-write-test-', dir=destination.parent):
        pass
    commit = _git(repo, 'rev-parse', '--verify', 'HEAD').strip().decode('ascii')
    dirty = _git(repo, 'status', '--porcelain', '--untracked-files=all')
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None or dirty:
        raise RuntimeError('a clean committed source checkout is required')
    sources = _verify_sources(repo, commit)
    _protect_sources(repo, sources)
    # The controller/bootstrap started before the share leases and is a
    # trusted quiescent-start assumption. Recheck all committed source bytes
    # under the leases before the child can import any project module.
    if (_git(repo, 'rev-parse', '--verify', 'HEAD').strip().decode('ascii') != commit
            or _git(repo, 'status', '--porcelain', '--untracked-files=all')
            or _verify_sources(repo, commit) != sources):
        raise RuntimeError('source checkout changed while acquiring protection')
    protocol_sha256 = sources[_PROTOCOL]
    protocol = _protocol(repo / _PROTOCOL, protocol_sha256)
    _verify_offline_cache(cache)
    from .pretrained import ASSETS
    verified_assets = [{'name': name, 'bytes': size, 'sha256': digest}
                       for name, (size, digest) in sorted(ASSETS.items())]
    gpu_uuid, before_private = _smi_snapshot(smi)
    before = _public_smi(before_private)
    driver = before['driver_version']
    if type(driver) is not str or re.fullmatch(r'[0-9.]{1,32}', driver) is None:
        raise RuntimeError('invalid driver version from device inventory')
    environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATHEXT', 'COMSPEC')
                   if key in os.environ}
    system_root = Path(environment.get('SYSTEMROOT', 'C:/Windows'))
    environment.update({'PATH': str(system_root / 'System32'), 'CUDA_VISIBLE_DEVICES': gpu_uuid,
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HOME': str(cache.parent), 'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1'})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    run_dir = Path(tempfile.mkdtemp(prefix='grouped-gpu-dev-', dir=root / 'work'))
    (run_dir / _MANIFEST).parent.mkdir()
    prefix = Path(tempfile.mkdtemp(prefix='pycache-', dir=run_dir))
    environment['PYTHONPYCACHEPREFIX'] = str(prefix)
    run_id = secrets.token_hex(16)
    _write_json(run_dir / 'private-smi-before.json', {'schema': 'grouped-gpu-private-smi-v1',
                'run_id': run_id, 'snapshot': before_private})
    _write_json(run_dir / _MANIFEST, {'schema': 'grouped-gpu-source-manifest-v1',
                'run_id': run_id, 'source_commit': commit, 'source_sha256': sources})
    source_manifest_sha256 = hashlib.sha256(_bounded_bytes(run_dir / _MANIFEST, 65536)).hexdigest()
    _protect_manifest(repo, sources, run_dir)
    if hashlib.sha256(_bounded_bytes(run_dir / _MANIFEST, 65536)).hexdigest() != source_manifest_sha256:
        raise RuntimeError('source manifest changed while acquiring protection')
    _write_json(run_dir / 'started.json', {'schema': 'grouped-gpu-start-v1', 'run_id': run_id,
                'source_commit': commit, 'protocol_sha256': protocol_sha256,
                'source_manifest_sha256': source_manifest_sha256})
    _write_json(run_dir / 'protocol-source.json', {'schema': 'grouped-gpu-protocol-source-v1',
                'sha256': protocol_sha256, 'protocol_utf8': (repo / _PROTOCOL).read_bytes().decode('utf-8')})
    _write_json(run_dir / 'asset-verification.json', {'schema': 'grouped-gpu-assets-v1',
                'run_id': run_id, 'model': protocol['model'], 'revision': protocol['revision'],
                'verified': True, 'assets': verified_assets})
    _write_json(run_dir / 'private-selected-device.json', {'schema': 'grouped-gpu-selected-device-v1',
                'run_id': run_id, 'physical_gpu_uuid': gpu_uuid})
    command = [sys._base_executable, '-I', '-B', '-S', '-X', 'pycache_prefix=' + str(prefix),
               str(repo / 'scripts/grouped_gpu_dev_bootstrap.py'), '--site-packages', str(site),
               '--owner', str(run_dir), str(cache), run_id,
               commit, protocol_sha256, driver, source_manifest_sha256]
    supervised = None
    supervisor_error = None
    try:
        supervised = _supervise_owner(command, identity=identity, repo=repo, environment=environment,
                                      job_source_sha256=sources['scripts/v2_source_guards.py'])
    except (KeyboardInterrupt, SystemExit) as exc:
        try:
            _write_json(run_dir / 'controller-cancelled.json', {'schema': 'grouped-gpu-cancelled-v1',
                        'run_id': run_id, 'exception_type': type(exc).__name__,
                        'status': 'owner-state-uncertain'})
        except BaseException as write_error:
            if not isinstance(write_error, Exception):
                raise write_error from exc
        raise
    except Exception as exc:
        supervisor_error = _public_error_type(type(exc).__name__)
    if supervised is not None:
        _write_json(run_dir / 'supervision.json', {'schema': 'grouped-gpu-supervision-v1',
                    'run_id': run_id, 'supervisor_run_id': supervised.run_id, 'pid': supervised.pid,
                    'exit_code': supervised.exit_code, 'clean': supervised.clean,
                    'timed_out': supervised.timed_out})
    after = None
    telemetry_error = None
    try:
        after_uuid, after_private = _smi_snapshot(smi)
        if after_uuid != gpu_uuid:
            raise RuntimeError('physical device inventory changed')
        _write_json(run_dir / 'private-smi-after.json', {'schema': 'grouped-gpu-private-smi-v1',
                    'run_id': run_id, 'snapshot': after_private})
        after = _public_smi(after_private)
    except Exception as exc:
        after = None
        telemetry_error = _public_error_type(type(exc).__name__)
    presence = _public_presence(_owner_presence_after_exit(smi, supervised.pid) if supervised is not None
                                else {'query': 'not_attempted', 'owner_pid_listed': None})
    result = {'schema': 'grouped-gpu-development-result-v1', 'run_id': run_id,
              'source_commit': commit, 'source_sha256': sources, 'protocol_sha256': protocol_sha256,
              'status': 'quarantined', 'supervisor_error_type': supervisor_error,
              'verified_checkpoint_assets': verified_assets,
              'limitations': ['post-exit process listing is a point observation, not proof of GPU idleness or native cleanup; WDDM limits per-process visibility',
                              'device counters are sequential point observations, not a worst-case capacity bound'],
              'supervisor': ({'exit_code': supervised.exit_code, 'clean': supervised.clean,
                              'timed_out': supervised.timed_out} if supervised is not None else None),
              'nvidia_smi_before': before, 'nvidia_smi_after': after,
              'owner_presence_after_exit': presence, 'post_telemetry_error_type': telemetry_error}
    partial_memory = []
    for path in sorted(run_dir.glob('memory-*.json'))[:128]:
        try:
            raw = _bounded_bytes(path, 4096)
            record = _public_memory(json.loads(raw))
            if record is not None:
                partial_memory.append(record)
        except (OSError, ValueError):
            pass
    if partial_memory:
        result['partial_memory_observations'] = partial_memory
    try:
        raw = _bounded_bytes(run_dir / 'sample-evidence.json', 1024 * 1024)
        sample = json.loads(raw)
        if type(sample) is dict and sample.get('status') in ('passed', 'failed', 'inconclusive'):
            result['partial_sample_evidence'] = {'status': sample['status'],
                'complete': sample.get('complete') is True}
    except (OSError, ValueError):
        pass
    if supervised is not None and supervised.clean and supervised.exit_code == 0 and not supervised.timed_out:
        result['status'] = 'incomplete'
        try:
            raw = _bounded_bytes(run_dir / 'candidate.json', 1024 * 1024)
            candidate = json.loads(raw)
            if (candidate.get('schema') != 'grouped-gpu-candidate-v1' or candidate.get('run_id') != run_id
                    or candidate.get('supervisor_run_id') != supervised.run_id or candidate.get('pid') != supervised.pid):
                raise RuntimeError('owner candidate does not match the reaped process')
            if type(candidate.get('result')) is not dict or candidate['result'].get('status') not in ('passed', 'failed'):
                raise RuntimeError('owner candidate lacks a valid outcome')
            if candidate['result']['status'] == 'passed':
                inventory, measurement, _ = _verify_promotion(run_dir, run_id, supervised, gpu_uuid,
                                                              protocol, protocol_sha256, verified_assets, sources,
                                                              source_commit=commit)
                result['evidence_inventory'] = inventory
                result['measurement'] = _public_measurement(measurement)
                result['evidence_consistency'] = 'verified-saved-records-no-raw-logits'
                if (telemetry_error is None and presence.get('query') == 'ok'
                        and presence.get('owner_pid_listed') is False):
                    result['status'] = 'passed'
            else:
                result['evidence_consistency'] = 'nonpassing-owner-result-not-promoted'
                result['status'] = 'failed'
        except Exception as exc:
            result['candidate_error_type'] = _public_error_type(type(exc).__name__)
    else:
        try:
            failure = json.loads(_bounded_bytes(run_dir / 'failure.json', 4096))
            if failure.get('schema') == 'grouped-gpu-failure-v1' and failure.get('run_id') == run_id:
                error_type = failure.get('error_type')
                result['owner_error_type'] = _public_error_type(error_type)
        except (OSError, ValueError, AttributeError):
            pass
    _write_json(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', nargs=7, metavar=('RUN_DIR', 'CACHE', 'RUN_ID', 'COMMIT', 'PROTOCOL_SHA',
                                                    'DRIVER', 'SOURCE_MANIFEST_SHA'),
                        help=argparse.SUPPRESS)
    parser.add_argument('--workspace-root')
    parser.add_argument('--cache-dir')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.owner:
        return _owner_main(*args.owner)
    if not args.workspace_root or not args.cache_dir or not args.output:
        parser.error('--workspace-root, --cache-dir and --output are required')
    result = run_development(workspace_root=args.workspace_root, cache_dir=args.cache_dir,
        output=args.output, site_packages=os.environ.get('LOOKING_GLASS_GROUPED_SITE'),
        smi=shutil.which('nvidia-smi'))
    print(json.dumps({'status': result['status'], 'source_commit': result['source_commit']}))
    return 0 if result['status'] == 'passed' else 1


def _owner_process_main(argv=None):
    """End the direct owner without Python shutdown hooks after any outcome."""
    status = 1
    try:
        status = main(argv)
    finally:
        os._exit(status if type(status) is int and 0 <= status <= 255 else 1)


if __name__ == '__main__':
    if '--owner' in sys.argv[1:]:
        _owner_process_main()
    else:
        raise SystemExit(main())
