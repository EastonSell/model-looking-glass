"""Disabled supervised entry for a bounded synthetic GPU failure exercise.

Trace capture and the zero-trace fixed worker are separate test phases. The
intentional OOM is a Python-raised exception, not device-memory exhaustion.
An unsafe owner exits without a clean attestation or further CUDA operations.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


_SOURCE_PATHS = (
    'scripts/worker_failure_gpu_bootstrap.py', 'scripts/v2_source_guards.py',
    'evidence/gpu-tap-latency-repeated-v1/schedule.json',
    'docs/GPU_WORKER_FAILURE_VALIDATION_V1.md',
    'looking_glass/__init__.py', 'looking_glass/worker_failure_gpu.py',
    'looking_glass/worker_failure_gpu_controller.py',
    'looking_glass/worker_failure_experiment.py', 'looking_glass/broker.py',
    'looking_glass/fair_queue.py', 'looking_glass/budgeted_worker.py',
    'looking_glass/resident_worker.py', 'looking_glass/owner_completion.py',
    'looking_glass/memory_budget.py', 'looking_glass/grouped_observation.py',
    'looking_glass/device_identity.py', 'looking_glass/device_lease.py',
    'looking_glass/supervisor.py', 'looking_glass/gpu_oneshot.py',
    'looking_glass/transformer_trace.py', 'looking_glass/trace_schema.py',
    'looking_glass/model_ownership.py', 'looking_glass/probes.py',
    'looking_glass/tap_repeated_owner.py', 'looking_glass/tap_repeated_admission.py',
    'looking_glass/tap_repeated_protocol.py',
)
_ENABLED = True
_PREFIX = 'LOOKING_GLASS_FAILURE_VALIDATION_'
_LIMIT = 64 * 1024
_MIN_DISK_BYTES = 2 * 1024**3
_CUSTODY = {'entered': False, 'directory': None, 'stage': 'entry', 'session': None,
            'lease': None, 'torch': None, 'model': None, 'budget': None,
            'resident_lease': None, 'traces': [], 'worker': None, 'pending': False,
            'pending_return': None, 'unsafe': False, 'failure': None, 'diagnostic': None,
            'device_calls': None, 'current_trace': None}


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if not 1 <= len(raw) <= _LIMIT:
        raise ValueError('private evidence record exceeds its bound')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short evidence write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _check_disk(path, *, usage=shutil.disk_usage):
    free = usage(path).free
    if type(free) is not int or free < _MIN_DISK_BYTES:
        raise RuntimeError('insufficient free disk for bounded private evidence')
    return free


def _device_evidence(run_id, total, cap, fraction, torch_version, cuda_build, transformers_version):
    if (type(run_id) is not str or re.fullmatch('[0-9a-f]{32}', run_id) is None
            or type(total) is not int or not 7 * 1024**3 <= total <= 9 * 1024**3
            or type(cap) is not int or cap != 512 * 1024**2
            or type(fraction) not in (float, int) or not math.isfinite(fraction)
            or not cap / total - 2 / total <= fraction <= cap / total + 1 / total
            or torch_version != '2.8.0+cu128' or cuda_build != '12.8'
            or transformers_version != '4.57.3'):
        raise ValueError('device/runtime self-report does not match the fixed contract')
    return {'schema': 'worker-failure-device-self-report-v1', 'run_id': run_id,
            'cuda_total_bytes': total, 'allocator_cap_bytes': cap,
            'allocator_fraction_readback': fraction, 'runtime_uuid_verified': True,
            'torch_version': torch_version, 'cuda_build': cuda_build,
            'transformers_version': transformers_version}


def _check_claims(budget, resident, temporary=None, *, gpu_trace_bytes=None):
    if budget is None or resident is None:
        raise RuntimeError('resident claim is unavailable')
    expected = {resident}
    if temporary is not None:
        if temporary is resident:
            raise RuntimeError('resource claims are not distinct')
        expected.add(temporary)
    if set(budget.active_leases()) != expected:
        raise RuntimeError('active resource claims differ')
    if gpu_trace_bytes is not None:
        if (temporary is None or type(gpu_trace_bytes) is not int or gpu_trace_bytes not in (0, 64)
                or temporary.trace_bytes('gpu') != gpu_trace_bytes):
            raise RuntimeError('active GPU trace charge differs')


def _call(guard, function, *args):
    guard()
    value = function(*args)
    guard()
    return value


class _DeviceCalls:
    """Fail-stop boundary for the owner's individual CUDA-facing calls.

    A native return before Python can assign it and repeated interruptions in
    this handler remain process-replacement boundaries, not local guarantees.
    """
    def __init__(self, guard):
        self.guard = guard
        self.pending = False
        self.unsafe = False
        self.return_root = None
        self.error = None

    def call(self, function, *args):
        if self.unsafe or self.pending:
            raise RuntimeError('device operation boundary is unsafe')
        self.pending = True
        try:
            self.guard()
            self.return_root = function(*args)
            self.guard()
            self.pending = False
            return self.return_root
        except BaseException as error:
            self.unsafe = True
            self.error = error
            raise


def _snapshot(guard, torch, expected_total, *, call=None):
    """Synchronize, then validate each reading before requesting the next."""
    if type(expected_total) is not int or expected_total <= 0:
        raise ValueError('invalid expected total')
    if call is None:
        call = lambda function, *args: _call(guard, function, *args)
    call(torch.cuda.synchronize, 0)
    pair = call(torch.cuda.mem_get_info, 0)
    if (type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not int
            or type(pair[1]) is not int or pair[1] != expected_total or not 0 <= pair[0] <= pair[1]):
        raise ValueError('invalid device free-memory reading')
    allocated = call(torch.cuda.memory_allocated, 0)
    if type(allocated) is not int or not 0 <= allocated <= expected_total:
        raise ValueError('invalid allocated-memory reading')
    reserved = call(torch.cuda.memory_reserved, 0)
    if type(reserved) is not int or not allocated <= reserved <= expected_total:
        raise ValueError('invalid reserved-memory reading')
    return {'free_bytes': pair[0], 'total_bytes': pair[1],
            'allocated_bytes': allocated, 'reserved_bytes': reserved}


def _forbid_collection(*args, **kwargs):
    raise RuntimeError('cyclic collection is disabled while the device owner is active')


class _ForbidChildProcess(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        raise RuntimeError('the GPU owner does not start Python subprocesses')


def _exercise_sequence(*, model, budget, resident_lease, device, guard, synchronize,
                       snapshot, record, state):
    from .worker_failure_experiment import (TraceCustody, WorkerCustody, run_trace_probe,
                                             run_worker_probe)
    state['stage'] = 'trace'
    for index, injected in enumerate((False, True, False)):
        trace = TraceCustody()
        state['traces'].append(trace)
        state['current_trace'] = trace
        result = run_trace_probe(model, budget=budget, device=device, guard=guard,
            synchronize=synchronize, custody=trace, inject_cancellation=injected)
        state['current_trace'] = None
        record('trace_' + str(index), result)
        record('after_trace_' + str(index), snapshot())
    state['stage'] = 'worker'
    worker = WorkerCustody()
    state['worker'] = worker
    return run_worker_probe(model, budget=budget, resident_lease=resident_lease, device=device,
                            guard=guard, checkpoint=lambda stage: snapshot(), record=record, custody=worker)


def _execute_owner(run_dir, cache_dir, run_id, source_commit, git_exe):
    if (re.fullmatch('[0-9a-f]{32}', run_id) is None
            or re.fullmatch('[0-9a-f]{40}', source_commit) is None
            or os.environ.get(_PREFIX + 'SOURCE_COMMIT') != source_commit
            or os.path.normcase(os.environ.get(_PREFIX + 'GIT_EXE', '')) != os.path.normcase(git_exe)
            or os.environ.get(_PREFIX + 'BOOTSTRAP_PID') != str(os.getpid())
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or any(name.partition('.')[0] in ('torch', 'transformers', 'numpy') for name in sys.modules)):
        raise RuntimeError('the isolated source bootstrap is required')
    repo = Path(__file__).resolve(strict=True).parents[1]
    ancestors = [parent for parent in repo.parents if parent.name.casefold() == 'work']
    if len(ancestors) != 1 or ancestors[0].name != 'work':
        raise RuntimeError('the shared workspace is ambiguous')
    work = ancestors[0].resolve(strict=True)
    directory, cache = Path(run_dir), Path(cache_dir)
    if (not directory.is_absolute() or directory.resolve(strict=True) != directory
            or not directory.is_dir() or not directory.is_relative_to(work)
            or not cache.is_absolute() or cache.resolve(strict=True) != work):
        raise ValueError('the private run and shared workspace must be canonical')
    _CUSTODY['directory'] = directory
    from .device_lease import DeviceLease
    from .supervisor import OwnerSession, _supervisor_birth
    session = OwnerSession.from_environment()
    _CUSTODY['session'] = session
    lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
    _CUSTODY['lease'] = lease
    OwnerSession.check(session, lease)
    if session.identity is None or Path(session.identity.workspace_root) != work.parent:
        raise RuntimeError('the physical device workspace differs from the checkout')
    _write_new(directory / 'owner.json', {'schema': 'worker-failure-owner-identity-v1',
        'run_id': run_id, 'source_commit': source_commit, 'supervisor_run_id': session.run_id,
        'owner_pid': os.getpid(), 'owner_birth': _supervisor_birth(),
        'supervisor_pid': session.supervisor_pid, 'supervisor_birth': session.supervisor_birth,
        'gpu_uuid': session.identity.gpu_uuid})
    subprocess.Popen = _ForbidChildProcess

    def guard():
        if _CUSTODY['unsafe'] or (_CUSTODY['device_calls'] is not None
                                   and _CUSTODY['device_calls'].unsafe):
            raise RuntimeError('device owner is unsafe')
        worker = _CUSTODY['worker']
        if worker is not None and worker.authority is not None and worker.authority.unsafe:
            raise RuntimeError('worker authority is unsafe')
        OwnerSession.check(session, lease)
        resident = _CUSTODY['resident_lease']
        if resident is not None:
            trace = _CUSTODY['current_trace']
            temporary = trace.lease if trace is not None else (None if worker is None else worker.workspace)
            trace_bytes = (trace.charge_expected if trace is not None and temporary is not None
                           else (0 if temporary is not None else None))
            _check_claims(_CUSTODY['budget'], resident, temporary, gpu_trace_bytes=trace_bytes)

    gc.disable()
    import torch
    import transformers
    from transformers import GPT2Config, GPT2LMHeadModel
    _CUSTODY['torch'] = torch
    if (str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8'
            or str(transformers.__version__) != '4.57.3'):
        raise RuntimeError('the pinned runtime versions differ')
    gc.collect = _forbid_collection
    calls = _DeviceCalls(guard)
    _CUSTODY['device_calls'] = calls
    _CUSTODY['stage'] = 'device_binding'
    if calls.call(torch.cuda.is_initialized) is not False:
        raise RuntimeError('CUDA was initialized before device binding')
    count = calls.call(torch.cuda.device_count)
    if type(count) is not int or count != 1:
        raise RuntimeError('one CUDA-visible device is required')
    props = calls.call(torch.cuda.get_device_properties, 0)
    from .gpu_oneshot import _runtime_uuid
    runtime_uuid = _runtime_uuid(torch, props.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=count)
    total = props.total_memory
    if type(total) is not int or not 7 * 1024**3 <= total <= 9 * 1024**3:
        raise RuntimeError('unsupported CUDA device memory')
    cap = 512 * 1024**2
    if calls.call(torch.cuda.get_allocator_backend) != 'native':
        raise RuntimeError('native CUDA allocator required')
    calls.call(torch.cuda.set_per_process_memory_fraction, cap / total, 0)
    readback = calls.call(torch.cuda.get_per_process_memory_fraction, 0)
    _write_new(directory / 'device.json', _device_evidence(run_id, total, cap, readback,
        str(torch.__version__), torch.version.cuda, str(transformers.__version__)))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(17)
    from .memory_budget import CudaMemorySnapshot, MemoryBudget

    def snapshot():
        result = _snapshot(guard, torch, total, call=calls.call)
        if result['free_bytes'] < 2 * 1024**3:
            raise RuntimeError('insufficient current device memory')
        return result

    def telemetry():
        row = snapshot()
        return CudaMemorySnapshot(row['free_bytes'], row['total_bytes'],
                                  row['allocated_bytes'], row['reserved_bytes'])

    def record(stage, value):
        allowed = ('admission', 'after_model_transfer', 'trace_0', 'trace_1', 'trace_2',
                   'after_trace_0', 'after_trace_1', 'after_trace_2', 'before_worker',
                   'baseline', 'after_baseline', 'cancellation', 'after_cancellation',
                   'post_cancel', 'after_post_cancel', 'before_simulated_oom', 'simulated_oom')
        if stage not in allowed or type(value) is not dict:
            raise ValueError('unexpected private record')
        _CUSTODY['stage'] = stage
        _write_new(directory / (stage + '.json'), {'schema': 'worker-failure-stage-v1',
                   'run_id': run_id, 'stage': stage, 'observed_ns': time.perf_counter_ns(), 'value': value})

    _CUSTODY['stage'] = 'admission'
    _CUSTODY['budget'] = MemoryBudget(telemetry, max_gpu_trace_bytes=64,
                                       max_host_trace_bytes=0, minimum_reserve_bytes=2 * 1024**3)
    _CUSTODY['resident_lease'] = _CUSTODY['budget'].acquire(32 * 1024**2,
                                     max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    record('admission', snapshot())
    config = GPT2Config(vocab_size=64, n_positions=8, n_ctx=8, n_embd=16,
                        n_layer=1, n_head=2, use_cache=False)
    config._attn_implementation = 'eager'
    _CUSTODY['model'] = GPT2LMHeadModel(config).eval()
    calls.call(_CUSTODY['model'].to, 'cuda:0')
    parameters = list(_CUSTODY['model'].parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if (not parameters or not 0 < parameter_count < 2_000_000
            or any(parameter.device.type != 'cuda' or parameter.device.index != 0
                   or parameter.dtype != torch.float32 for parameter in parameters)):
        raise RuntimeError('the synthetic model does not match its bounded device contract')
    parameters = None
    _write_new(directory / 'model.json', {'schema': 'worker-failure-model-self-report-v1',
        'run_id': run_id, 'parameter_count': parameter_count, 'dtype': 'float32',
        'device': 'cuda:0', 'input_shape': [1, 2], 'randomly_initialized': True})
    record('after_model_transfer', snapshot())
    report = _exercise_sequence(model=_CUSTODY['model'], budget=_CUSTODY['budget'],
        resident_lease=_CUSTODY['resident_lease'], device='cuda:0', guard=guard,
        synchronize=lambda: calls.call(torch.cuda.synchronize, 0), snapshot=snapshot,
        record=record, state=_CUSTODY)
    # The deliberately raised OOM leaves the fixed owner unsafe. No CUDA call,
    # release of a live claim, or clean attestation follows this point.
    _CUSTODY['unsafe'] = True
    _CUSTODY['stage'] = 'simulated_oom_terminal'
    _write_new(directory / 'terminal.json', {'schema': 'worker-failure-owner-terminal-v1',
        'run_id': run_id, 'status': 'expected_failure_unclean', 'clean_attested': False,
        'simulated_oom': report['simulated_oom'], 'device_cleanup_attested': False})
    return 1


def owner_main(run_dir, cache_dir, run_id, source_commit, git_exe):
    """Fail-stop entry: the bootstrap must hard-exit after this returns."""
    if _ENABLED is not True or _CUSTODY['entered'] is not False:
        return 1
    _CUSTODY['entered'] = True
    try:
        return _execute_owner(run_dir, cache_dir, run_id, source_commit, git_exe)
    except BaseException as error:
        _CUSTODY['unsafe'] = True
        if _CUSTODY['failure'] is None:
            _CUSTODY['failure'] = error
        directory = _CUSTODY['directory']
        if directory is not None:
            kind = type(error).__name__
            if re.fullmatch('[A-Za-z_][A-Za-z_0-9]{0,63}', kind) is None:
                kind = 'other'
            try:
                _write_new(directory / 'failure.json', {
                    'schema': 'worker-failure-owner-incomplete-v1', 'status': 'incomplete',
                    'stage': _CUSTODY['stage'], 'error_type': kind, 'clean_attested': False})
            except BaseException as diagnostic:
                _CUSTODY['diagnostic'] = diagnostic
        return 1


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == '--owner':
        if len(arguments) != 6:
            raise ValueError('invalid owner arguments')
        return owner_main(*arguments[1:])
    from .worker_failure_gpu_controller import main as controller_main
    return controller_main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
