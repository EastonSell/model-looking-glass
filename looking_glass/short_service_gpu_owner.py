"""One supervised 1x16 or 1x32 DistilGPT2 development observation.

The controller is source-bound and starts one direct child in a Windows Job.
Successful attestation requires verified cleanup and released resource claims.
This is not a measured admission profile, a general service, or a benchmark.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import threading
import weakref

from . import grouped_gpu_dev as g
from .device_identity import resolve_device_identity
from .device_lease import DeviceLease
from .short_service_cpu_reference import _read_inputs, MANIFEST_SHA256, INVENTORY_SHA256
from .short_service_gpu_evidence import NAMES, verify_payloads, validate_supervision
from .supervisor import OwnerSession


GIB = 1024 ** 3
MIN_DISK = 2 * GIB
MIN_HOST_FREE = 3 * GIB
HOST_CLAIM = 256 * 1024 ** 2
POLICY = {'seed': 29, 'allocator_fraction': .25, 'minimum_preload_free_bytes': 5 * GIB,
          'minimum_reserve_bytes': GIB // 2, 'resident_commitment_bytes': 2 * GIB,
          'forward_commitment_bytes': GIB}
STAGES = ('before-reference-alpha', 'after-reference-alpha', 'before-reference-beta',
          'after-reference-beta', 'before-two-clients', 'after-two-clients')
_EXTRA_SOURCES = ('looking_glass/short_service_gpu_owner.py',
                  'looking_glass/short_service_gpu_evidence.py',
                  'looking_glass/short_service_cpu_reference.py',
                  'looking_glass/short_service_pilot_core.py',
                  'looking_glass/bert_matched_numeric.py', 'looking_glass/broker.py',
                  'scripts/short_service_gpu_bootstrap.py',
                  'experiments/short_cpu_service_reference_inputs_v1.json',
                  'docs/SHORT_GPU_SERVICE_OWNER_V1.md',
                  'tests/test_short_service_gpu_evidence.py',
                  'tests/test_short_service_gpu_owner.py', 'pyproject.toml')
_SOURCE_PATHS = tuple(dict.fromkeys((*g._SOURCE_PATHS, *_EXTRA_SOURCES)))
_HOLD = [None] * 16


def _bind_sources():
    # The existing guarded controller and child must protect the same explicit
    # superset. Each process does this before source verification and leasing.
    g._SOURCE_PATHS = _SOURCE_PATHS


def _disk_guard(*paths):
    values = [shutil.disk_usage(path).free for path in paths]
    if any(type(value) is not int or value < MIN_DISK for value in values):
        raise RuntimeError('disk free space is below the fixed reserve')
    return values


def _select_git(executable):
    path = Path(executable)
    if not path.is_absolute() or not path.resolve(strict=True).is_file():
        raise ValueError('an absolute Git executable is required')
    path = path.resolve(strict=True)
    os.environ['PATH'] = str(path.parent) + os.pathsep + os.environ.get('PATH', '')
    selected = shutil.which('git')
    if selected is None or Path(selected).resolve(strict=True) != path:
        raise RuntimeError('the selected Git executable is not first in PATH')
    return path


def _available_host():
    if sys.platform != 'win32':
        raise RuntimeError('the fixed host-memory observation requires Windows')
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                    ('total_phys', ctypes.c_uint64), ('avail_phys', ctypes.c_uint64),
                    ('total_page', ctypes.c_uint64), ('avail_page', ctypes.c_uint64),
                    ('total_virtual', ctypes.c_uint64), ('avail_virtual', ctypes.c_uint64),
                    ('avail_extended_virtual', ctypes.c_uint64)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
        raise OSError('cannot read available physical host memory')
    if not 0 < status.avail_phys <= status.total_phys:
        raise RuntimeError('invalid host-memory observation')
    return int(status.avail_phys)


def _cases(manifest, bucket):
    if type(bucket) is not int or bucket not in (16, 32):
        raise ValueError('the pilot supports only the fixed 16 and 32 token buckets')
    ids = ('synthetic-16', 'gas') if bucket == 16 else ('synthetic-32', 'ligand')
    cases = manifest['comparisons']
    selected = []
    for case_id in ids:
        matches = [case for case in cases if case.get('id') == case_id]
        if len(matches) != 1:
            raise ValueError('fixed input case is missing or repeated')
        item = matches[0]['service']
        tokens, mask = item['input_ids'], item['attention_mask']
        if (type(tokens) is not list or type(mask) is not list or len(tokens) != bucket
                or len(mask) != bucket or any(type(value) is not int for value in tokens + mask)
                or any(not 0 <= token < 50257 for token in tokens)
                or any(value not in (0, 1) for value in mask)):
            raise ValueError('fixed input shape or values differ')
        active = sum(mask)
        if (not 1 <= active <= bucket or mask != [1] * active + [0] * (bucket - active)
                or any(token != 50256 for token in tokens[active:])):
            raise ValueError('fixed input is not right padded')
        selected.append({'case_id': case_id, 'input_ids': tokens[:], 'attention_mask': mask[:],
                         'active_position': active - 1})
    return selected


def _write_tensor(run_dir, cache, name, raw, bucket):
    if name not in NAMES or type(raw) is not bytes or len(raw) != bucket * 50257 * 4:
        raise ValueError('unexpected retained full-logit output')
    _disk_guard(run_dir, cache)
    with (run_dir / (name + '.f32')).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short output write')
        stream.flush()
        os.fsync(stream.fileno())


def _detach_idle_model(session, lease, *, worker, resident, authority, pilot, model, check_claims):
    """Detach known model roots only after the direct broker is fully quiescent."""
    broker = pilot.broker
    held = []
    try:
        for lock in (worker._execution_gate, worker._lock, broker._condition):
            if not lock.acquire(blocking=False):
                raise RuntimeError('worker or broker is still active during teardown')
            held.append(lock)
        custody_names = ('request', 'logits', 'result', 'error', 'producer_error', 'forward_error', 'state_error')
        if (pilot.finished is not True or pilot.errors or any(thread.is_alive() for thread in pilot.threads)
                or broker._worker is not worker or authority._broker is not broker or authority._worker is not worker
                or worker._worker is not resident or worker._model is not model or resident._model is not model
                or not broker._closed or broker._thread.is_alive() or not worker._closed
                or broker._running is not None or broker._pending_slots or broker._team_slots or broker._preparing
                or broker._fair_preparations or broker._fair_handoff is not None or broker._fair_running_job is not None
                or any(item is not None for item in broker._job_roots) or len(broker._queue)
                or broker._fatal_cleanup_failure or broker.quarantined or broker._shutdown_error is not None
                or broker._unavailable is not None or broker._fair_orphan_state is not None
                or any(getattr(broker, name) is not None for name in
                       ('_status_error', '_status_cleanup_error', '_status_recovery_error', '_owner_failure',
                        '_owner_escaping_failure', '_owner_hook_failure', '_owner_hook_traceback',
                        '_owner_hook_recovery_failure', '_observation_cleanup_failure'))
                or any(item is not broker._quarantine_empty for item in broker._quarantine_objects)
                or authority.unsafe or worker.failed or resident._drain_pending or resident._snapshot_error is not None
                or any(value is not None for value in resident._live_buffers)
                or any(value is not None for value in resident._retiring_buffers)
                or any(getattr(worker._custody, name) is not None for name in custody_names)
                or worker._retiring_transfer is not None or worker._retiring_cancel is not None
                or worker._transfer_cleanup_error is not None or worker._cancel_cleanup_error is not None):
            raise RuntimeError('broker, worker or forward custody is not idle')
        check_claims()
        # Other known roots still hold the model, so resident.close cannot
        # perform the final model-reference drop under its internal lock.
        g._cuda(session, lease, resident.close)
        if not resident._closed or resident._model is not None:
            raise RuntimeError('resident did not release its model reference')
        check_claims()
        with authority._decision_lock:
            if authority.unsafe or worker._model is not model:
                raise RuntimeError('owner became unsafe before model detachment')
            worker._model = None
        check_claims()
        if authority.unsafe or worker.failed:
            raise RuntimeError('owner became unsafe while detaching the model')
    except BaseException:
        authority.mark_unsafe()
        raise
    finally:
        try:
            for lock in reversed(held):
                lock.release()
        except BaseException:
            authority.mark_unsafe()
            raise


def _owner_run(session, lease, *, run_dir, cache, inventory, bucket, driver, sources,
               _worker_device='cuda:0'):
    """Execute the fixed graph while keeping every owner root until exit."""
    if _worker_device not in ('cuda:0', 'cpu'):
        raise ValueError('invalid internal test device')
    import torch
    from .budgeted_worker import WorkloadProfile
    from .host_budget import HostBudget
    from .memory_budget import CudaMemorySnapshot, MemoryBudget
    from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
    from .pretrained import load_distilgpt2
    from .resident_worker import ResidentWorker
    from .short_service_pilot_core import SerialRoutingPilot

    manifest, _ = _read_inputs(Path(__file__).resolve().parents[1] /
                               'experiments/short_cpu_service_reference_inputs_v1.json', inventory)
    selected = _cases(manifest, bucket)
    _disk_guard(run_dir, cache)
    available_host = _available_host()
    if available_host < MIN_HOST_FREE:
        raise RuntimeError('host free memory is below the fixed point threshold')
    g._configure(session, lease, torch, POLICY)
    properties = g._bind_device(session, lease, torch)
    import transformers
    total = properties.total_memory
    if not g._cuda(session, lease, torch.cuda.is_available):
        raise RuntimeError('leased CUDA device is unavailable')
    if g._cuda(session, lease, torch.cuda.get_allocator_backend) != 'native':
        raise RuntimeError('the native CUDA allocator is required')
    g._cuda(session, lease, torch.cuda.set_per_process_memory_fraction, POLICY['allocator_fraction'], 0)
    expected_fraction = int(POLICY['allocator_fraction'] * total) / total
    if g._cuda(session, lease, torch.cuda.get_per_process_memory_fraction, 0) != expected_fraction:
        raise RuntimeError('allocator fraction differs')
    g._cuda(session, lease, torch.cuda.manual_seed, POLICY['seed'])
    runtime = g._live_runtime(session, lease, torch, transformers, driver, total)
    g._validate_runtime(runtime, POLICY)
    g._write_json(run_dir / 'private-runtime.json', runtime)

    # These claims account for named retained output/scratch envelopes only;
    # they do not cap Python metadata, checkpoint construction or process RSS.
    host = HostBudget(max_bytes=HOST_CLAIM, max_items=128, max_requests=1, max_operations=1)
    _HOLD[0] = host
    host_guard = host.request_guard('short-service-control')
    _HOLD[1] = host_guard
    host_guard.open()
    host_request = host_guard.adopt()
    host_claim_guard = host_request.claim_guard('short-service-control', 'outputs-and-scratch',
                                                HOST_CLAIM, items=128)
    _HOLD[2] = host_claim_guard
    host_claim_guard.reserve()
    _HOLD[3] = host_claim_guard.adopt()
    lock = threading.RLock()
    snapshots = []
    window = [None]

    def partial(label, value):
        g._write_json(run_dir / f'memory-{len(snapshots):03d}.json',
                      {'stage': label, 'partial': True,
                       'counters': {key: item if type(item) is int else {'invalid_type': type(item).__name__}
                                    for key, item in value.items()}})

    def observe(label):
        with lock:
            value = g._snapshot(session, lease, torch, total, lambda item: partial(label, item))
            g._record_memory(run_dir, snapshots, label, value, total)
            g._check_observed_headroom(value, POLICY)
            if window[0] is not None:
                g._check_workspace_peak(window[0], value, POLICY)
            return value

    def telemetry():
        # The worker calls this on its single execution thread. Main-thread
        # stages run only before submission or after all submitted outcomes.
        with lock:
            value = observe('admission')
            return CudaMemorySnapshot(value['device_free_bytes'], value['device_total_bytes'],
                                      value['allocated_bytes'], value['reserved_bytes'])

    initial = observe('before-model-load')
    if initial['device_free_bytes'] < POLICY['minimum_preload_free_bytes']:
        raise RuntimeError('device free memory is below the fixed preload threshold')
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=0, max_host_trace_bytes=0,
                          minimum_reserve_bytes=POLICY['minimum_reserve_bytes'])
    g._ROOTS[2] = budget
    resident_claim = budget.acquire(POLICY['resident_commitment_bytes'], max_gpu_trace_bytes=0,
                                    max_host_trace_bytes=0)
    g._ROOTS[3] = resident_claim
    _disk_guard(run_dir, cache)
    host_before_load = _available_host()
    if host_before_load < MIN_HOST_FREE:
        raise RuntimeError('host free memory fell below the fixed point threshold')
    bundle = load_distilgpt2(cache_dir=str(cache), allow_download=False)
    g._ROOTS[4] = bundle
    model = bundle.model
    model_ref = weakref.ref(model)
    model_metadata = dict(bundle.metadata)
    if (model.training is not False or model.config.use_cache is not False
            or getattr(model.config, '_attn_implementation', None) != 'eager'
            or model.config.vocab_size != 50257 or model.config.n_positions < bucket
            or type(bundle.metadata.get('parameter_count')) is not int
            or bundle.metadata['parameter_count'] <= 0
            or any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
                   for parameter in model.parameters())):
        raise RuntimeError('model differs from the fixed workload')
    # CPU checkpoint loading may take time; do not rely on the earlier free
    # sample when starting the GPU copy.
    g._check_committed_headroom(budget)
    g._convert_model(session, lease, model, device=_worker_device, dtype=torch.float32)
    g._cuda(session, lease, torch.cuda.synchronize, 0)
    if (any(parameter.device.type != ('cuda' if _worker_device == 'cuda:0' else 'cpu')
            or (_worker_device == 'cuda:0' and parameter.device.index != 0) or parameter.dtype != torch.float32
            for parameter in model.parameters())
            or sum(parameter.numel() for parameter in model.parameters()) != bundle.metadata['parameter_count']
            or model.get_input_embeddings().weight is not model.get_output_embeddings().weight):
        raise RuntimeError('model device, dtype or tied embeddings differ')
    observe('after-model-load')
    workspace_claim = budget.acquire(POLICY['forward_commitment_bytes'], max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
    g._ROOTS[19] = workspace_claim
    model_id = hashlib.sha256(g._canonical(bundle.metadata['asset_sha256'])).hexdigest()
    output_size = bucket * 50257 * 4
    resident = ResidentWorker(model_id, model, device=_worker_device, max_batch_size=1,
                              max_sequence_length=bucket, max_input_bytes=bucket * 16,
                              max_output_bytes=output_size)
    g._ROOTS[5] = resident
    profile = WorkloadProfile('short-development-' + str(bucket), model_id, _worker_device, (1, bucket),
                              (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64')),
                              output_size, POLICY['forward_commitment_bytes'])

    def owner_guard():
        g._assert_no_child_processes()
        if g._cuda_uncertain() or gc.isenabled():
            raise RuntimeError('device owner has an uncertain operation')
        OwnerSession.check(session, lease)
        g._assert_no_child_processes()
        if g._cuda_uncertain() or gc.isenabled():
            raise RuntimeError('device owner became uncertain')

    authority = OwnerCompletionAuthority(device_guard=owner_guard)
    g._ROOTS[20] = authority
    worker = LifetimeWorkspaceWorker(resident, budget, resident_claim, workspace_claim,
                                     profiles=(profile,), authority=authority)
    g._ROOTS[6] = worker
    rows = tuple({'input_ids': torch.tensor([row['input_ids']], dtype=torch.int64, device='cpu'),
                  'attention_mask': torch.tensor([row['attention_mask']], dtype=torch.int64, device='cpu')}
                 for row in selected)
    _HOLD[4] = rows
    stages = []

    def on_stage(label):
        if len(stages) >= len(STAGES) or label != STAGES[len(stages)]:
            raise RuntimeError('unexpected model-stage order')
        with lock:
            if label.startswith('before-'):
                window[0] = None
                g._cuda(session, lease, torch.cuda.synchronize, 0)
                g._cuda(session, lease, torch.cuda.reset_peak_memory_stats, 0)
                first = observe(label)
                g._check_peak_reset(first)
                window[0] = first
            else:
                if window[0] is None:
                    raise RuntimeError('post-stage lacks its reset baseline')
                observe(label)
                window[0] = None
        stages.append(label)

    pilot = SerialRoutingPilot(worker, authority, bucket=bucket, vocab=50257, pad=50256,
                               seed=POLICY['seed'], cases=rows, on_stage=on_stage,
                               on_tensor=lambda name, raw: _write_tensor(run_dir, cache, name, raw, bucket))
    _HOLD[5] = pilot
    if threading.getprofile() is not None:
        raise RuntimeError('pre-existing thread profiling is unsupported')
    model_code = type(model).forward.__code__
    entries = []
    copies = []
    _HOLD[7] = copies

    def record_entry(frame, event, _arg):
        if event == 'call' and frame.f_code is model_code:
            tensors = (frame.f_locals.get('input_ids'), frame.f_locals.get('attention_mask'))
            if (len(entries) >= 4 or any(type(tensor) is not torch.Tensor or tuple(tensor.shape) != (1, bucket)
                    or tensor.dtype != torch.int64 or tensor.device.type !=
                    ('cuda' if _worker_device == 'cuda:0' else 'cpu')
                    or (_worker_device == 'cuda:0' and tensor.device.index != 0) for tensor in tensors)):
                raise RuntimeError('model entry differs from the fixed input shape or device')
            for tensor in tensors:
                copied = g._cuda(session, lease, tensor.to, device='cpu', copy=True)
                copies.append(copied)
                if (type(copied) is not torch.Tensor or copied.device.type != 'cpu'
                        or copied.dtype != torch.int64 or tuple(copied.shape) != (1, bucket)):
                    raise RuntimeError('model-entry host copy differs from its fixed layout')
            g._cuda(session, lease, torch.cuda.synchronize, 0)
            actual_ids, actual_mask = (item.tolist()[0] for item in copies[-2:])
            matches = [row for row in selected if row['input_ids'] == actual_ids
                       and row['attention_mask'] == actual_mask]
            if len(matches) != 1:
                g._write_json(run_dir / f"forward-{len(entries) + 1:03d}-entry-invalid.json",
                              {'entry': len(entries) + 1, 'reason': 'input-or-mask-mismatch',
                               'input_ids': actual_ids, 'attention_mask': actual_mask})
                raise RuntimeError('model-entry IDs or mask differ from both fixed cases')
            case_id = matches[0]['case_id']
            if (len(entries) < 2 and case_id != selected[len(entries)]['case_id']) or (
                    len(entries) == 3 and case_id == entries[2]['case_id']):
                g._write_json(run_dir / f"forward-{len(entries) + 1:03d}-entry-invalid.json",
                              {'entry': len(entries) + 1, 'reason': 'case-order-mismatch',
                               'input_ids': actual_ids, 'attention_mask': actual_mask, 'case_id': case_id})
                raise RuntimeError('model-entry case order differs from the fixed routing plan')
            record = {'entry': len(entries) + 1, 'shape': [1, bucket],
                      'dtype': 'torch.int64', 'worker_device': _worker_device, 'case_id': case_id,
                      'input_ids': actual_ids, 'attention_mask': actual_mask}
            g._write_json(run_dir / f"forward-{len(entries) + 1:03d}-entry.json", record)
            entries.append(record)
        return record_entry

    _HOLD[6] = record_entry
    threading.setprofile(record_entry)
    try:
        result = pilot.run()
        if threading.getprofile() is not record_entry:
            raise RuntimeError('thread profiling changed during the pilot')
    except BaseException:
        authority.mark_unsafe()
        raise
    finally:
        threading.setprofile(None)
    if stages != list(STAGES):
        raise RuntimeError('the full stage sequence was not completed')
    if len(entries) != 4:
        raise RuntimeError('the model did not enter exactly four fixed-shape forwards')
    pilot.close()
    g._cuda(session, lease, torch.cuda.synchronize, 0)
    observe('after-worker-close')
    end_runtime = g._live_runtime(session, lease, torch, transformers, driver, total)
    g._validate_runtime(end_runtime, POLICY)
    if g._canonical(end_runtime) != g._canonical(runtime):
        raise RuntimeError('runtime settings changed during the pilot')
    g._check_cleanup_reservations(session, lease, budget, resident_claim, workspace_claim,
                                  authority, POLICY, (resident_claim, workspace_claim))
    if authority.unsafe or g._cuda_uncertain():
        raise RuntimeError('workload completion remains uncertain')
    def check_cleanup():
        g._check_cleanup_reservations(session, lease, budget, resident_claim, workspace_claim,
                                      authority, POLICY, (resident_claim, workspace_claim))

    output_refs = [weakref.ref(item.result.logits) for item in pilot.outcomes]
    input_refs = ([weakref.ref(tensor) for row in rows for tensor in row.values()] +
                  [weakref.ref(tensor) for request in pilot.requests for tensor in request.inputs.values()] +
                  [weakref.ref(tensor) for tensor in copies])
    if len(output_refs) != 4:
        raise RuntimeError('output custody inventory differs')
    if g._ROOTS[4] is not bundle or bundle.model is not model or g._ROOTS[16] is not model:
        raise RuntimeError('model roots differ before teardown')
    _detach_idle_model(session, lease, worker=worker, resident=resident, authority=authority,
                       pilot=pilot, model=model, check_claims=check_cleanup)
    try:
        # The closed pilot and broker remain rooted, but their model and tensor
        # aliases are explicitly removed while both device claims remain live.
        for values in (pilot.outcomes, pilot.handles, pilot.requests, pilot.threads, copies):
            check_cleanup()
            values.clear()
            check_cleanup()
        for index in (4, 7):
            check_cleanup()
            _HOLD[index] = None
            check_cleanup()
        rows = None
        check_cleanup()
        bundle.model = None
        check_cleanup()
        g._ROOTS[4] = None
        check_cleanup()
        bundle = None
        check_cleanup()
        g._ROOTS[16] = None
        check_cleanup()
        model = None
        check_cleanup()
        if model_ref() is not None or any(ref() is not None for ref in output_refs + input_refs):
            raise RuntimeError('model or retained input/output tensors remain alive')
        OwnerSession.check(session, lease)
        if authority.unsafe or g._cuda_uncertain():
            raise RuntimeError('owner became unsafe while disposing known references')
    except BaseException:
        authority.mark_unsafe()
        raise
    g._cuda(session, lease, _HOLD[3].release)
    g._cuda(session, lease, host_request.close)
    host_state = host.snapshot()
    if (host_state.used_bytes or host_state.used_items or host_state.requests
            or host_state.claims or host_state.operations):
        raise RuntimeError('host resource claims remain after disposal')
    g._assert_disposable_return_slots()

    def partial_cleanup(stage, values):
        g._write_json(run_dir / 'cleanup-partial.json', {'stage': stage, 'partial': True,
                      'counters': {key: item if type(item) is int else {'invalid_type': type(item).__name__}
                                   for key, item in values.items()}})

    cleanup = g._cuda(session, lease, g._guarded_cleanup, session, lease, torch,
                      check_cleanup, total, partial_cleanup)
    if cleanup['after'] != {'allocated_bytes': 0, 'reserved_bytes': 0}:
        raise RuntimeError('CUDA allocator still owns bytes after cleanup')
    check_cleanup()
    g._release_clean_reservations(session, lease, budget, resident_claim, workspace_claim, authority, POLICY)
    cleanup.update({'model_released': True, 'input_output_tensors_released': True,
                    'host_claims_after': 0, 'accounting_leases_after': 0})
    g._write_json(run_dir / 'cleanup.json', cleanup)
    return {'status': 'passed', 'bucket': bucket, 'worker_device': _worker_device, 'cases': selected,
            'model_entries': entries,
            'runtime_redacted': g._public_runtime(runtime), 'memory': snapshots,
            'host_available_bytes': available_host, 'host_before_load_bytes': host_before_load,
            'host_claim_bytes': HOST_CLAIM,
            'model_metadata': model_metadata, 'routing': result, 'cleanup': cleanup,
            'source_sha256': sources, 'manifest_sha256': MANIFEST_SHA256,
            'inventory_sha256': INVENTORY_SHA256}


def _owner_main(run_dir, cache, inventory, bucket, run_id, commit, driver, source_manifest_sha):
    _bind_sources()
    run_dir, cache = Path(run_dir), Path(cache)
    try:
        g._require_bootstrap()
        g._require_supervisor()
        g._claim_attempt()
        if any(name == root or name.startswith(root + '.') for name in sys.modules
               for root in ('torch', 'transformers', 'huggingface_hub')):
            raise RuntimeError('model dependencies were imported before the physical lease')
        g._install_no_child_processes()
        session = OwnerSession.from_environment()
        g._ROOTS[0] = session
        repo = Path(__file__).resolve().parents[1]
        g._validate_workspace_spec(session, repo)
        sources = g._verify_child_sources(repo, run_dir, run_id, commit, source_manifest_sha)
        g._protect_sources(repo, sources)
        g._protect_manifest(repo, sources, run_dir)
        if g._verify_child_sources(repo, run_dir, run_id, commit, source_manifest_sha) != sources:
            raise RuntimeError('source changed while protection was acquired')
        _read_inputs(repo / 'experiments/short_cpu_service_reference_inputs_v1.json', inventory)
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        g._ROOTS[1] = lease
        OwnerSession.check(session, lease)
        g._validate_workspace_binding(session, repo)
        g._write_json(run_dir / 'owner.json', {'run_id': run_id, 'supervisor_run_id': session.run_id,
                      'pid': os.getpid(), 'source_manifest_sha256': source_manifest_sha})
        gc.disable()
        g._NO_CYCLIC_GC_REQUIRED = True
        result = _owner_run(session, lease, run_dir=run_dir, cache=cache, inventory=inventory,
                            bucket=int(bucket), driver=driver, sources=sources)
        candidate = {'schema': 'short-service-gpu-candidate-v1', 'run_id': run_id,
                     'supervisor_run_id': session.run_id, 'pid': os.getpid(),
                     'source_commit': commit, 'source_manifest_sha256': source_manifest_sha,
                     'result': result}
        g._write_json(run_dir / 'candidate.json', candidate)
        if g._cuda_uncertain() or g._ROOTS[20].unsafe:
            raise RuntimeError('owner became uncertain before attestation')
        g._check_cleanup_reservations(session, lease, g._ROOTS[2], g._ROOTS[3], g._ROOTS[19],
                                      g._ROOTS[20], POLICY, ())
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        g._ROOTS[13] = exc
        try:
            g._write_json(run_dir / 'failure.json', {'schema': 'short-service-gpu-failure-v1',
                          'run_id': run_id, 'error_type': type(exc).__name__, 'status': 'incomplete'})
        except BaseException as write_error:
            g._ROOTS[17] = write_error
        return 1


def _validate_memory(memory, run_dir):
    if type(memory) is not list or not memory or len(memory) > 128:
        raise RuntimeError('memory evidence inventory is invalid')
    paths = sorted(run_dir.glob('memory-*.json'))
    if [path.name for path in paths] != [f'memory-{i:03d}.json' for i in range(len(memory))]:
        raise RuntimeError('memory evidence files are missing or extra')
    fields = {'device_free_bytes', 'device_total_bytes', 'allocated_bytes', 'reserved_bytes',
              'peak_allocated_bytes', 'peak_reserved_bytes'}
    window = None
    stages = []
    total = None
    for path, entry in zip(paths, memory, strict=True):
        if (type(entry) is not dict or set(entry) != fields | {'stage'} or type(entry['stage']) is not str
                or json.loads(g._bounded_bytes(path, 4096)) != entry):
            raise RuntimeError('memory evidence differs from the candidate')
        values = {key: entry[key] for key in fields}
        g._validate_memory_values(values)
        if total is None:
            total = values['device_total_bytes']
        if (values['device_total_bytes'] != total or not 7 * GIB <= total <= 9 * GIB
                or max(values['reserved_bytes'], values['peak_reserved_bytes']) > int(.25 * total)):
            raise RuntimeError('device total or allocator cap differs')
        g._check_observed_headroom(values, POLICY)
        stage = entry['stage']
        if stage in STAGES:
            stages.append(stage)
            if stage.startswith('before-'):
                if window is not None:
                    raise RuntimeError('memory reset windows overlap')
                g._check_peak_reset(values)
                window = values
            else:
                if window is None:
                    raise RuntimeError('memory reset baseline is missing')
                g._check_workspace_peak(window, values, POLICY)
                window = None
        elif stage not in ('admission', 'before-model-load', 'after-model-load', 'after-worker-close'):
            raise RuntimeError('unexpected memory stage')
        elif window is not None:
            g._check_workspace_peak(window, values, POLICY)
    names = [entry['stage'] for entry in memory]
    if (stages != list(STAGES) or window is not None or names.count('before-model-load') != 1
            or names.count('after-model-load') != 1 or names.count('after-worker-close') != 1
            or names.index('before-model-load') >= names.index('after-model-load')
            or names.index('after-model-load') >= names.index(STAGES[0])
            or names.index('after-worker-close') <= names.index(STAGES[-1])
            or memory[names.index('before-model-load')]['device_free_bytes'] < POLICY['minimum_preload_free_bytes']):
        raise RuntimeError('memory stages are incomplete or out of order')
    return True


def _promote(run_dir, *, run_id, commit, supervised, bucket, source_manifest_sha, sources, expected_cases,
             expected_gpu_uuid, expected_driver):
    record = {'run_id': run_id, 'supervisor_run_id': supervised.run_id, 'pid': supervised.pid,
              'exit_code': supervised.exit_code, 'clean': supervised.clean, 'timed_out': supervised.timed_out}
    validate_supervision(record, run_id=run_id)
    if ((run_dir / 'failure.json').exists() or (run_dir / 'cleanup-partial.json').exists()
            or list(run_dir.glob('*.partial'))
            or list(run_dir.glob('forward-*-entry-invalid.json'))):
        raise RuntimeError('failure or incomplete output record remains')
    repo = Path(__file__).resolve().parents[1]
    if g._verify_child_sources(repo, run_dir, run_id, commit, source_manifest_sha) != sources:
        raise RuntimeError('source manifest or protected source differs')
    owner_raw = g._bounded_bytes(run_dir / 'owner.json', 4096)
    owner_record = json.loads(owner_raw)
    if (type(owner_record) is not dict or set(owner_record) !=
            {'run_id', 'supervisor_run_id', 'pid', 'source_manifest_sha256'}
            or owner_record['run_id'] != run_id or owner_record['supervisor_run_id'] != supervised.run_id
            or type(owner_record['pid']) is not int or owner_record['pid'] != supervised.pid
            or owner_record['source_manifest_sha256'] != source_manifest_sha):
        raise RuntimeError('owner receipt does not match the reaped child')
    raw = g._bounded_bytes(run_dir / 'candidate.json', 1024 * 1024)
    candidate = json.loads(raw)
    if (type(candidate) is not dict or candidate.get('schema') != 'short-service-gpu-candidate-v1'
            or candidate.get('run_id') != run_id or candidate.get('source_commit') != commit
            or candidate.get('supervisor_run_id') != supervised.run_id
            or type(candidate.get('pid')) is not int or candidate['pid'] != supervised.pid
            or candidate.get('source_manifest_sha256') != source_manifest_sha):
        raise RuntimeError('candidate does not match the reaped owner')
    result = candidate['result']
    if (type(result) is not dict or result.get('status') != 'passed' or result.get('worker_device') != 'cuda:0'
            or type(result.get('bucket')) is not int
            or result['bucket'] != bucket or g._canonical(result.get('cases')) != g._canonical(expected_cases)
            or g._canonical(result.get('source_sha256')) != g._canonical(sources)
            or result.get('manifest_sha256') != MANIFEST_SHA256 or result.get('inventory_sha256') != INVENTORY_SHA256
            or result.get('routing', {}).get('status') != 'passed'
            or type(result['routing'].get('bucket')) is not int or result['routing']['bucket'] != bucket
            or type(result['routing'].get('broker_jobs')) is not int or result['routing']['broker_jobs'] != 4
            or result['routing'].get('vocab') != 50257
            or type(result.get('host_available_bytes')) is not int
            or result['host_available_bytes'] < MIN_HOST_FREE
            or type(result.get('host_before_load_bytes')) is not int
            or result['host_before_load_bytes'] < MIN_HOST_FREE or result.get('host_claim_bytes') != HOST_CLAIM):
        raise RuntimeError('candidate workload identity differs')
    entries = result.get('model_entries')
    if (type(entries) is not list or len(entries) != 4 or any(type(entry) is not dict for entry in entries)):
        raise RuntimeError('model-entry evidence inventory differs')
    case_ids = [entry.get('case_id') for entry in entries]
    expected_ids = [case['case_id'] for case in expected_cases]
    if case_ids[:2] != expected_ids or sorted(case_ids[2:]) != sorted(expected_ids):
        raise RuntimeError('model-entry case order differs')
    by_id = {case['case_id']: case for case in expected_cases}
    expected_entries = [{'entry': index, 'shape': [1, bucket], 'dtype': 'torch.int64',
                         'worker_device': 'cuda:0', 'case_id': case_id,
                         'input_ids': by_id[case_id]['input_ids'],
                         'attention_mask': by_id[case_id]['attention_mask']}
                        for index, case_id in enumerate(case_ids, 1)]
    if g._canonical(entries) != g._canonical(expected_entries):
        raise RuntimeError('model-entry evidence differs from the fixed four calls')
    entry_files = sorted(run_dir.glob('forward-*-entry.json'))
    if [path.name for path in entry_files] != [f'forward-{index:03d}-entry.json' for index in range(1, 5)]:
        raise RuntimeError('model-entry evidence inventory differs')
    if any(g._canonical(json.loads(g._bounded_bytes(path, 4096))) != g._canonical(entry)
           for path, entry in zip(entry_files, expected_entries, strict=True)):
        raise RuntimeError('model-entry file differs from the candidate')
    from .pretrained import ASSETS, MODEL_ID, REVISION
    metadata = result.get('model_metadata')
    if (type(metadata) is not dict or metadata.get('model_id') != MODEL_ID
            or metadata.get('model_revision') != REVISION
            or type(metadata.get('parameter_count')) is not int or metadata['parameter_count'] <= 0
            or metadata.get('asset_sha256') != {name: digest for name, (_, digest) in ASSETS.items()}
            or metadata.get('weights_sha256') != ASSETS['model.safetensors'][1]):
        raise RuntimeError('checkpoint asset identity differs')
    runtime = json.loads(g._bounded_bytes(run_dir / 'private-runtime.json', 65536))
    g._validate_runtime(runtime, POLICY)
    if runtime['physical_gpu_uuid'] != expected_gpu_uuid or runtime['driver'] != expected_driver:
        raise RuntimeError('observed runtime device or driver differs')
    if g._canonical(result.get('runtime_redacted')) != g._canonical(g._public_runtime(runtime)):
        raise RuntimeError('runtime projection differs')
    positions = tuple(item['active_position'] for item in expected_cases)
    recomputed = verify_payloads(run_dir, bucket=bucket, positions=positions,
                                 expected_hashes=result['routing']['full_logit_sha256'])
    for name in ('reference_separation', 'comparisons'):
        # Different math implementations can change max_abs by a few ULPs;
        # the saved reader's independently reconstructed criteria are decisive.
        if name not in result['routing'] or name not in recomputed:
            raise RuntimeError('routing comparison is absent')
    memory = result['memory']
    _validate_memory(memory, run_dir)
    if runtime['total_bytes'] != memory[0]['device_total_bytes']:
        raise RuntimeError('runtime device total differs from memory records')
    cleanup = result.get('cleanup')
    if (type(cleanup) is not dict or set(cleanup) != {'before', 'after', 'api', 'model_released',
                                                    'input_output_tensors_released', 'host_claims_after',
                                                    'accounting_leases_after'}
            or type(cleanup.get('before')) is not dict or type(cleanup.get('after')) is not dict
            or set(cleanup['before']) != {'allocated_bytes', 'reserved_bytes'}
            or set(cleanup['after']) != {'allocated_bytes', 'reserved_bytes'}
            or any(type(value) is not int or value < 0 for value in cleanup['before'].values())
            or cleanup['before']['allocated_bytes'] > cleanup['before']['reserved_bytes']
            or cleanup['before']['reserved_bytes'] > int(.25 * runtime['total_bytes'])
            or any(type(value) is not int or value != 0 for value in cleanup['after'].values())
            or cleanup.get('api') != 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces'
            or cleanup.get('model_released') is not True or cleanup.get('input_output_tensors_released') is not True
            or type(cleanup.get('host_claims_after')) is not int or cleanup['host_claims_after'] != 0
            or type(cleanup.get('accounting_leases_after')) is not int or cleanup['accounting_leases_after'] != 0):
        raise RuntimeError('cleanup evidence is incomplete')
    cleanup_raw = g._bounded_bytes(run_dir / 'cleanup.json', 4096)
    if g._canonical(json.loads(cleanup_raw)) != g._canonical(cleanup):
        raise RuntimeError('cleanup evidence differs from the candidate')
    return {'candidate_sha256': hashlib.sha256(raw).hexdigest(), 'routing_readback': recomputed,
            'output_sha256': dict(result['routing']['full_logit_sha256']),
            'memory_sha256': {path.name: hashlib.sha256(g._bounded_bytes(path, 4096)).hexdigest()
                              for path in sorted(run_dir.glob('memory-*.json'))},
            'model_entry_sha256': {path.name: hashlib.sha256(g._bounded_bytes(path, 4096)).hexdigest()
                                   for path in entry_files},
            'owner_receipt_sha256': hashlib.sha256(owner_raw).hexdigest(),
            'cleanup_sha256': hashlib.sha256(cleanup_raw).hexdigest(),
            'private_runtime_sha256': hashlib.sha256(g._bounded_bytes(run_dir / 'private-runtime.json', 65536)).hexdigest(),
            'source_manifest_sha256': source_manifest_sha, 'memory': memory, 'bucket': bucket, 'cases': result['cases'],
            'host_available_bytes': result['host_available_bytes'],
            'host_before_load_bytes': result['host_before_load_bytes'], 'host_claim_bytes': HOST_CLAIM,
            'runtime_redacted': result['runtime_redacted'], 'model_metadata': result['model_metadata'],
            'cleanup': cleanup}


def run_development(*, workspace_root, cache_dir, inventory, output, site_packages, smi, bucket, git_exe):
    _bind_sources()
    g._require_bootstrap()
    g._require_supervisor()
    g._claim_attempt()
    g._verify_controller_import_mode()
    repo = Path(__file__).resolve().parents[1]
    root = g._workspace_root(repo, workspace_root)
    cache = Path(cache_dir).resolve(strict=True)
    inv = Path(inventory).resolve(strict=True)
    site = Path(site_packages).resolve(strict=True)
    if str(site) != os.environ.get('LOOKING_GLASS_GROUPED_SITE'):
        raise RuntimeError('dependency directory differs from the bootstrap')
    destination = Path(output)
    if (not destination.is_absolute() or destination.exists() or not destination.parent.is_dir()
            or destination.parent.resolve(strict=True) != (root / 'work').resolve(strict=True)):
        raise ValueError('output must be a new directory directly under the canonical work folder')
    _disk_guard(destination.parent, cache)
    if _available_host() < MIN_HOST_FREE:
        raise RuntimeError('host free memory is below the fixed point threshold')
    selected_git = _select_git(git_exe)
    manifest, _ = _read_inputs(repo / 'experiments/short_cpu_service_reference_inputs_v1.json', inv)
    expected_cases = _cases(manifest, bucket)
    commit = g._git(repo, 'rev-parse', '--verify', 'HEAD').strip().decode('ascii')
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None or g._git(repo, 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('a clean committed source checkout is required')
    sources = g._verify_sources(repo, commit)
    g._protect_sources(repo, sources)
    if (g._git(repo, 'rev-parse', '--verify', 'HEAD').strip().decode('ascii') != commit
            or g._git(repo, 'status', '--porcelain', '--untracked-files=all')
            or g._verify_sources(repo, commit) != sources
            or Path(shutil.which('git')).resolve(strict=True) != selected_git):
        raise RuntimeError('source changed while protection was acquired')
    g._verify_offline_cache(cache)
    # The scheduler must grant this separate attempt before any device query.
    gpu_uuid, before_private = g._smi_snapshot(smi)
    before = g._public_smi(before_private)
    driver = before['driver_version']
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATHEXT',
                                            'COMSPEC', 'PROGRAMFILES') if key in os.environ}
    system_root = Path(env.get('SYSTEMROOT', 'C:/Windows'))
    env.update({'PATH': str(system_root / 'System32'), 'CUDA_VISIBLE_DEVICES': gpu_uuid,
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HOME': str(cache.parent), 'PYTHONNOUSERSITE': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=env)
    destination.mkdir()
    (destination / g._MANIFEST).parent.mkdir()
    prefix = destination / 'owner-pycache'
    prefix.mkdir()
    run_id = secrets.token_hex(16)
    g._write_json(destination / 'private-smi-before.json', {'run_id': run_id, 'snapshot': before_private})
    g._write_json(destination / g._MANIFEST, {'schema': 'grouped-gpu-source-manifest-v1',
                  'run_id': run_id, 'source_commit': commit, 'source_sha256': sources})
    source_sha = hashlib.sha256(g._bounded_bytes(destination / g._MANIFEST, 65536)).hexdigest()
    g._protect_manifest(repo, sources, destination)
    g._write_json(destination / 'started.json', {'run_id': run_id, 'source_commit': commit,
                  'source_manifest_sha256': source_sha, 'bucket': bucket,
                  'manifest_sha256': MANIFEST_SHA256, 'inventory_sha256': INVENTORY_SHA256})
    command = [sys._base_executable, '-I', '-B', '-S', '-X', 'pycache_prefix=' + str(prefix),
               str(repo / 'scripts/short_service_gpu_bootstrap.py'), '--site-packages', str(site),
               '--owner', str(destination), str(cache), str(inv), str(bucket), run_id, commit, driver, source_sha]
    supervised = g._supervise_owner(command, identity=identity, repo=repo, environment=env,
                                    job_source_sha256=sources['scripts/v2_source_guards.py'])
    record = {'run_id': run_id, 'supervisor_run_id': supervised.run_id, 'pid': supervised.pid,
              'exit_code': supervised.exit_code, 'clean': supervised.clean, 'timed_out': supervised.timed_out,
              'reason': supervised.reason}
    g._write_json(destination / 'supervision.json', record)
    result = {'schema': 'short-service-gpu-observation-v1', 'status': 'incomplete',
              'source_commit': commit, 'bucket': bucket, 'device_before': before,
              'limitations': ['one public exposed development workload; no capacity or throughput claim',
                              'direct in-process Broker clients; no IPC or batching observation',
                              'device free is a point sample and allocator peaks are process-local',
                              'zero PyTorch allocator counters do not prove complete native or global device cleanup',
                              'source and saved-byte checks do not independently attest execution']}
    try:
        result['measurement'] = _promote(destination, run_id=run_id, commit=commit, supervised=supervised,
                                         bucket=bucket, source_manifest_sha=source_sha, sources=sources,
                                         expected_cases=expected_cases, expected_gpu_uuid=gpu_uuid,
                                         expected_driver=driver)
        result['status'] = 'passed'
    except Exception as exc:
        result['error_type'] = g._public_error_type(type(exc).__name__)
    g._write_json(destination / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', nargs=8, help=argparse.SUPPRESS)
    parser.add_argument('--workspace-root')
    parser.add_argument('--cache-dir')
    parser.add_argument('--inventory')
    parser.add_argument('--output')
    parser.add_argument('--git')
    parser.add_argument('--bucket', type=int, choices=(16, 32))
    args = parser.parse_args(argv)
    if args.owner:
        return _owner_main(*args.owner)
    if not all((args.workspace_root, args.cache_dir, args.inventory, args.output, args.bucket, args.git)):
        parser.error('workspace, cache, inventory, output, bucket and Git are required')
    result = run_development(workspace_root=args.workspace_root, cache_dir=args.cache_dir,
                             inventory=args.inventory, output=args.output,
                             site_packages=os.environ.get('LOOKING_GLASS_GROUPED_SITE'),
                             smi=shutil.which('nvidia-smi'), bucket=args.bucket, git_exe=args.git)
    print(json.dumps({'status': result['status'], 'source_commit': result['source_commit']}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    if '--owner' in sys.argv[1:]:
        status = 1
        try:
            status = main()
        finally:
            os._exit(status if type(status) is int and 0 <= status <= 255 else 1)
    raise SystemExit(main())
