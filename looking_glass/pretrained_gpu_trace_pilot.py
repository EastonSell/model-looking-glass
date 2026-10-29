"""One supervised, source-bound five-token DistilGPT2 CUDA trace pilot."""

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import weakref

from .bert_gpu_evidence import write_json_new
from .bert_gpu_exposed_pair import (_file_identity, _public_presence, _public_smi_snapshot, _public_supervisor,
    _unexpected_hooks, _validate_workspace_binding, _validate_workspace_spec, _workspace_root)
from .device_identity import resolve_device_identity
from .device_lease import DeviceLease
from .gpu_oneshot import (_owner_presence_after_exit, _runtime_uuid, _smi_snapshot,
                          _verify_controller_import_mode, _verify_offline_cache)
from .memory_budget import CudaMemorySnapshot, MemoryBudget
from .supervisor import OwnerSession, run_supervised
from .vision_coco_cls_exposed_gpu import _capacity, _host_memory, _validate_runtime_dirs

MIB, GIB = 1024 ** 2, 1024 ** 3
RESIDENT, JOB, GPU_TRACE, HOST_TRACE = 512 * MIB, GIB, 131072, 3 * 131072
CAP = RESIDENT + JOB
TOKENS = (464, 6766, 318, 4171, 13)
PHASES = ('load', 'baseline_cold', 'baseline_warm', 'reference', 'direct', 'gradient', 'id')
_PREFIX = 'LOOKING_GLASS_TRACE_PILOT_'
_SOURCE_PATHS = ('looking_glass/__init__.py', 'looking_glass/pretrained_gpu_trace_pilot.py',
    'looking_glass/pretrained_gpu_trace_pilot_core.py', 'looking_glass/pretrained_gpu_trace_pilot_runtime.py',
    'looking_glass/pretrained.py', 'looking_glass/transformer_trace.py', 'looking_glass/model_ownership.py',
    'looking_glass/probes.py', 'looking_glass/trace_schema.py', 'looking_glass/memory_budget.py',
    'looking_glass/bert_gpu_evidence.py', 'looking_glass/bert_gpu_exposed_pair.py',
    'looking_glass/vision_coco_cls_exposed_gpu.py', 'looking_glass/gpu_oneshot.py',
    'looking_glass/device_identity.py', 'looking_glass/device_lease.py', 'looking_glass/supervisor.py',
    'docs/PRETRAINED_SMOKE_V1.md', 'docs/PRETRAINED_GPU_TRACE_V1.md',
    'docs/PRETRAINED_GPU_TRACE_PILOT_V1.md', 'scripts/pretrained_gpu_trace_pilot_bootstrap.py',
    'tools/observe_pretrained_gpu_trace_pilot.py')

_ROOT = {'entered': False, 'stage': 'entry', 'session': None, 'lease': None, 'budget': None,
         'resident': None, 'job': None, 'claims_ready': False, 'gc_disabled': False,
         'bundle': None, 'model': None, 'pending_return': None, 'executor': None,
         'failure': None, 'diagnostic_failure': None, 'cuda_pending': False, 'sync_pending': False}


def _forbid_collection(*_args, **_kwargs):
    raise RuntimeError('cyclic collection is forbidden during owner custody')


def _startup():
    _verify_controller_import_mode()
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or os.environ.get(_PREFIX + 'BOOTSTRAP_PID') != str(os.getpid())):
        raise RuntimeError('the isolated source bootstrap is required')


def _pristine():
    if any(name in sys.modules for name in ('torch', 'numpy', 'transformers')):
        raise RuntimeError('model libraries were imported before the physical lease')


def _source_identity(repo, git):
    environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT')
                   if key in os.environ}
    environment.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                        'GIT_CONFIG_COUNT': '0', 'GIT_NO_REPLACE_OBJECTS': '1',
                        'GIT_TERMINAL_PROMPT': '0'})
    def run(*args):
        return subprocess.run([str(git), '--no-replace-objects', *args], cwd=repo, env=environment,
                              capture_output=True, check=True, timeout=15).stdout
    commit = run('rev-parse', 'HEAD').decode('ascii').strip()
    if len(commit) != 40 or run('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('a clean committed source checkout is required')
    hashes = {}
    for name in _SOURCE_PATHS:
        path = repo / name
        with path.open('rb') as handle:
            value = handle.read(4 * MIB + 1)
        if len(value) > 4 * MIB or value != run('show', commit + ':' + name):
            raise RuntimeError('source differs from the committed tree')
        hashes[name] = hashlib.sha256(value).hexdigest()
    return commit, hashes


def _runtime_settings(torch):
    return {'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'default_dtype': str(torch.get_default_dtype()),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'autocast_enabled': torch.is_autocast_enabled('cuda')}


def _verify_model(model, torch, *, device, metadata):
    config = model.config
    if ((config.n_embd, config.n_layer, config.vocab_size) != (768, 6, 50257)
            or config._attn_implementation != 'eager' or config.use_cache is not False
            or metadata.get('parameter_count') != 81912576
            or any(type(module.training) is not bool or module.training for module in model.modules())):
        raise RuntimeError('fixed checkpoint, geometry or evaluation state differs')
    parameters = list(model.parameters())
    if (sum(item.numel() for item in parameters) != 81912576
            or any(str(item.device) != device or item.dtype != torch.float32 for item in parameters)
            or any(str(item.device) != device for item in model.buffers())):
        raise RuntimeError('fixed parameter device, dtype or count differs')


def _weakrefs_for_disposal(roots):
    model = roots['model']
    values = [model, *model.parameters(), *model.buffers(), roots['input_ids'], roots['attention_mask']]
    values.extend(item[5] for item in roots['reference'])
    for trace in roots['traces'].values():
        values.append(trace)
        values.extend(event.value for event in trace.events if event.value is not None)
    return [weakref.ref(value) for value in values if value is not None]


def _synchronize(guarded, torch):
    _ROOT['sync_pending'] = True
    guarded(torch.cuda.synchronize, 0)
    _ROOT['sync_pending'] = False


def _recheck_runtime_binding(guarded, torch, session, lease, expected_uuid, total):
    count = guarded(torch.cuda.device_count)
    if type(count) is not int or count != 1:
        raise RuntimeError('CUDA device count changed before attestation')
    properties = guarded(torch.cuda.get_device_properties, 0)
    actual_uuid = _runtime_uuid(torch, properties.uuid)
    if actual_uuid != expected_uuid or type(properties.total_memory) is not int or properties.total_memory != total:
        raise RuntimeError('CUDA device identity changed before attestation')
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=actual_uuid, device_count=count)
    return True


def _measure(session, lease, torch, cache, directory):
    from .pretrained import load_distilgpt2
    from .pretrained_gpu_trace_pilot_core import run_fixed_trace
    from .pretrained_gpu_trace_pilot_runtime import PilotExecutor
    import transformers

    if (str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8'
            or str(transformers.__version__) != '4.57.3'
            or not callable(getattr(torch._C, '_cuda_clearCublasWorkspaces', None))):
        raise RuntimeError('pinned dependency or cleanup runtime differs')
    settings = None

    def check():
        OwnerSession.check(session, lease)
        if (_ROOT['session'] is not session or _ROOT['lease'] is not lease
                or not _ROOT['gc_disabled'] or gc.isenabled() or gc.collect is not _forbid_collection):
            raise RuntimeError('physical lease or collection custody changed')
        if _ROOT['claims_ready']:
            budget, resident, job = _ROOT['budget'], _ROOT['resident'], _ROOT['job']
            if (budget is None or resident is None or job is None
                    or set(budget.active_leases()) != {resident, job}
                    or job.trace_bytes('gpu') != GPU_TRACE or job.trace_bytes('host') != HOST_TRACE):
                raise RuntimeError('memory or trace commitment changed')
        if settings is not None:
            if _runtime_settings(torch) != settings:
                raise RuntimeError('numeric runtime settings changed')
            model = _ROOT['model']
            if model is not None and (model.config.use_cache is not False
                    or any(type(module.training) is not bool or module.training for module in model.modules())):
                raise RuntimeError('model evaluation or cache state changed')

    def guarded(fn, *args):
        check()
        value = fn(*args)
        _ROOT['pending_return'] = value
        check()
        return value

    def sync():
        _synchronize(guarded, torch)

    if guarded(torch.cuda.is_initialized) or guarded(torch.cuda.device_count) != 1:
        raise RuntimeError('unexpected CUDA initialization or visible devices')
    properties = guarded(torch.cuda.get_device_properties, 0)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=1)
    total = properties.total_memory
    provider = os.environ.get(_PREFIX + 'SMI_TOTAL', '')
    if (type(total) is not int or not 7 * GIB <= total <= 9 * GIB
            or not provider.isascii() or not provider.isdecimal()
            or abs(int(provider) - total) > 2 * MIB):
        raise RuntimeError('CUDA and NVIDIA device totals differ')
    if guarded(torch.cuda.get_allocator_backend) != 'native':
        raise RuntimeError('native allocator is required')
    fraction = CAP / total
    guarded(torch.cuda.set_per_process_memory_fraction, fraction, 0)
    readback = guarded(torch.cuda.get_per_process_memory_fraction, 0)
    if type(readback) is not float or int(readback * total) != CAP:
        raise RuntimeError('effective allocator ceiling differs')
    sync()
    guarded(torch.cuda.reset_peak_memory_stats, 0)
    reserve = max(GIB, (total * 15 + 99) // 100)

    def snapshot():
        pair = guarded(torch.cuda.mem_get_info, 0)
        if (type(pair) not in (tuple, list) or len(pair) != 2 or any(type(x) is not int for x in pair)
                or pair[1] != total or not 0 <= pair[0] <= total):
            raise ValueError('device free/total observation differs')
        allocated = guarded(torch.cuda.memory_allocated, 0)
        if type(allocated) is not int or not 0 <= allocated <= CAP:
            raise ValueError('allocated observation differs')
        reserved = guarded(torch.cuda.memory_reserved, 0)
        if type(reserved) is not int or not allocated <= reserved <= CAP:
            raise ValueError('reserved observation differs')
        peak_allocated = guarded(torch.cuda.max_memory_allocated, 0)
        if type(peak_allocated) is not int or not allocated <= peak_allocated <= CAP:
            raise ValueError('peak allocated observation differs')
        peak_reserved = guarded(torch.cuda.max_memory_reserved, 0)
        if type(peak_reserved) is not int or not max(reserved, peak_allocated) <= peak_reserved <= CAP:
            raise ValueError('peak reserved observation differs')
        if pair[0] < reserve + CAP or total < reserved + reserve + CAP:
            raise RuntimeError('fixed device reserve or commitment no longer fits')
        return {'free_bytes': pair[0], 'total_bytes': total, 'allocated_bytes': allocated,
                'reserved_bytes': reserved, 'peak_allocated_bytes': peak_allocated,
                'peak_reserved_bytes': peak_reserved}

    def telemetry():
        value = snapshot()
        return CudaMemorySnapshot(value['free_bytes'], value['total_bytes'],
                                  value['allocated_bytes'], value['reserved_bytes'])
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=GPU_TRACE, max_host_trace_bytes=HOST_TRACE)
    _ROOT['budget'] = budget
    _ROOT['resident'] = budget.acquire(RESIDENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    _ROOT['job'] = budget.acquire(JOB, max_gpu_trace_bytes=GPU_TRACE, max_host_trace_bytes=HOST_TRACE)
    _ROOT['job'].reserve_trace('gpu', GPU_TRACE)
    _ROOT['job'].reserve_trace('host', HOST_TRACE)
    _ROOT['claims_ready'] = True
    check()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.random.default_generator.manual_seed(0)
    guarded(torch.cuda.manual_seed, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = _runtime_settings(torch)
    if (settings['threads'] != 1 or settings['interop_threads'] != 1
            or settings['default_dtype'] != 'torch.float32' or settings['matmul_tf32']
            or settings['cudnn_tf32'] or settings['autocast_enabled']):
        raise RuntimeError('fixed numeric settings differ')
    runtime = {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
               'transformers': str(transformers.__version__), 'python': sys.version.split()[0],
               'device': 'cuda:0', 'device_uuid': runtime_uuid, 'device_name': properties.name,
               'compute_capability': [properties.major, properties.minor], 'device_total_bytes': total,
               'nvidia_total_bytes': int(provider), 'allocator_backend': 'native',
               'allocator_fraction': readback, 'allocator_cap_bytes': CAP, 'reserve_bytes': reserve,
               'resident_commitment_bytes': RESIDENT, 'job_commitment_bytes': JOB,
               'gpu_trace_claim_bytes': GPU_TRACE, 'host_trace_claim_bytes': HOST_TRACE,
               'seed': 0, 'dtype': 'float32', 'attention': 'eager', 'use_cache': False,
               'inference_mode': True, **settings}
    executor = PilotExecutor(torch=torch, check=check, synchronize=sync, snapshot=snapshot,
        reset_peaks=lambda: guarded(torch.cuda.reset_peak_memory_stats, 0), directory=directory, roots=_ROOT)
    _ROOT['executor'] = executor
    _ROOT['stage'] = 'load'
    executor.begin('load')
    bundle = load_distilgpt2(cache_dir=cache, stage_dir=Path(directory) / 'stage', allow_download=False)
    _ROOT['bundle'] = bundle
    model = bundle.model
    _ROOT['model'] = model
    _verify_model(model, torch, device='cpu', metadata=bundle.metadata)
    before_transfer = snapshot()
    _ROOT['cuda_pending'] = True
    guarded(model.to, 'cuda:0')
    sync()
    _ROOT['cuda_pending'] = False
    _verify_model(model, torch, device='cuda:0', metadata=bundle.metadata)
    executor.end('load', {'parameter_count': bundle.metadata['parameter_count'],
                          'weights_sha256': bundle.metadata['weights_sha256'],
                          'before_transfer': before_transfer})

    # Input transfer is separately guarded and synchronized. Its fixed 80
    # bytes are included in each subsequent phase's declared allowance.
    _ROOT['stage'] = 'input_setup'
    cpu_ids = torch.tensor([TOKENS], dtype=torch.int64, device='cpu')
    cpu_mask = torch.ones_like(cpu_ids)
    if cpu_ids.tolist() != [list(TOKENS)] or cpu_mask.tolist() != [[1] * 5]:
        raise RuntimeError('fixed CPU input differs')
    _ROOT['cuda_pending'] = True
    input_ids = guarded(cpu_ids.to, 'cuda:0')
    _ROOT['input_ids'] = input_ids
    attention_mask = guarded(cpu_mask.to, 'cuda:0')
    _ROOT['attention_mask'] = attention_mask
    sync()
    _ROOT['cuda_pending'] = False
    executor.verify_inputs(input_ids, attention_mask)
    write_json_new(Path(directory) / 'input-setup.json', {'schema': 'pretrained-gpu-trace-input-v1',
        'input_ids': list(TOKENS), 'attention_mask': [1] * 5, 'dtype': 'int64', 'shape': [1, 5],
        'memory': snapshot()})
    if _unexpected_hooks(model):
        raise RuntimeError('unexpected hooks before the fixed trace')
    _ROOT['stage'] = 'trace'
    result = run_fixed_trace(model, input_ids, attention_mask, executor, study_id='pretrained-gpu-trace-pilot-v1')
    if (executor.index != len(PHASES) or executor.forward_count != 10 or _unexpected_hooks(model)):
        raise RuntimeError('fixed trace inventory or hook cleanup differs')
    sync()
    check()
    references = _weakrefs_for_disposal(_ROOT)
    metadata = dict(bundle.metadata)
    _ROOT['stage'] = 'disposal'
    _ROOT['reference'] = _ROOT['reference_lease'] = _ROOT['reference_handles'] = None
    _ROOT['traces'].clear()
    _ROOT['quarantined_events'].clear()
    _ROOT['cpu_logits'] = _ROOT['input_ids'] = _ROOT['attention_mask'] = None
    _ROOT['input_check_ids'] = _ROOT['input_check_mask'] = None
    _ROOT['model'] = _ROOT['bundle'] = _ROOT['pending_return'] = None
    del model, bundle, input_ids, attention_mask, cpu_ids, cpu_mask
    check()
    if any(ref() is not None for ref in references):
        raise RuntimeError('model, inputs or trace objects remained after disposal')
    sync()
    guarded(torch.cuda.empty_cache)
    guarded(torch._C._cuda_clearCublasWorkspaces)
    sync()
    guarded(torch.cuda.empty_cache)
    cleanup = snapshot()
    if cleanup['allocated_bytes'] or cleanup['reserved_bytes']:
        raise RuntimeError('local allocator retains bytes after cleanup')
    check()
    binding_rechecked = _recheck_runtime_binding(guarded, torch, session, lease, runtime_uuid, total)
    check()
    _ROOT['claims_ready'] = False
    _ROOT['job'].release_trace('gpu', GPU_TRACE)
    _ROOT['job'].release_trace('host', HOST_TRACE)
    _ROOT['job'].release()
    _ROOT['resident'].release()
    if budget.active_leases():
        raise RuntimeError('logical memory leases remain')
    check()
    return {'runtime': runtime, 'model': metadata, 'result': result, 'phases': executor.phases,
            'forward_count': executor.forward_count, 'cleanup': cleanup,
            'model_input_trace_weakrefs_dead': True, 'active_memory_leases': 0,
            'runtime_binding_rechecked': binding_rechecked,
            'host_memory_after_cleanup': _host_memory()}


def _read_json(path, limit=MIB):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate diagnostic key')
            result[key] = value
        return result
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('diagnostic exceeds its byte limit')
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _sha_file(path, maximum=MIB):
    return _file_identity(Path(path), maximum)[1]


def _validate_saved(directory, measurement, *, hidden=768, vocab=50257,
                    device='cuda:0', study_id='pretrained-gpu-trace-pilot-v1'):
    """Reconcile fixed records; this is not an independent tensor replay."""
    from .pretrained_gpu_trace_pilot_runtime import phase_estimate
    directory = Path(directory)
    def exact_ints(value, expected):
        return (type(value) is list and len(value) == len(expected)
                and all(type(item) is int and item == wanted for item, wanted in zip(value, expected)))
    if (type(measurement) is not dict or type(measurement.get('forward_count')) is not int
            or measurement['forward_count'] != 10 or type(measurement.get('phases')) is not list
            or len(measurement['phases']) != 7):
        raise ValueError('fixed phase or forward inventory differs')
    from .pretrained import ASSETS, MODEL_ID, REVISION
    runtime = measurement.get('runtime')
    if (type(runtime) is not dict or type(runtime.get('device_total_bytes')) is not int
            or not 7 * GIB <= runtime['device_total_bytes'] <= 9 * GIB
            or type(runtime.get('allocator_cap_bytes')) is not int or runtime['allocator_cap_bytes'] != CAP
            or type(runtime.get('reserve_bytes')) is not int
            or runtime['reserve_bytes'] != max(GIB, (runtime['device_total_bytes'] * 15 + 99) // 100)
            or type(runtime.get('nvidia_total_bytes')) is not int
            or abs(runtime['nvidia_total_bytes'] - runtime['device_total_bytes']) > 2 * MIB
            or type(runtime.get('allocator_fraction')) is not float
            or not math.isfinite(runtime['allocator_fraction'])
            or int(runtime['allocator_fraction'] * runtime['device_total_bytes']) != CAP
            or runtime.get('allocator_backend') != 'native' or runtime.get('device') != device
            or type(runtime.get('device_uuid')) is not str or not runtime['device_uuid']
            or type(runtime.get('device_name')) is not str or not runtime['device_name']
            or type(runtime.get('compute_capability')) is not list or len(runtime['compute_capability']) != 2
            or any(type(x) is not int or not 0 <= x <= 99 for x in runtime['compute_capability'])
            or runtime.get('torch') != '2.8.0+cu128' or runtime.get('cuda_build') != '12.8'
            or runtime.get('transformers') != '4.57.3' or type(runtime.get('python')) is not str
            or type(runtime.get('seed')) is not int or runtime['seed'] != 0
            or runtime.get('dtype') != 'float32' or runtime.get('default_dtype') != 'torch.float32'
            or runtime.get('attention') != 'eager' or runtime.get('use_cache') is not False
            or runtime.get('inference_mode') is not True or runtime.get('autocast_enabled') is not False
            or runtime.get('matmul_tf32') is not False or runtime.get('cudnn_tf32') is not False
            or type(runtime.get('threads')) is not int or runtime['threads'] != 1
            or type(runtime.get('interop_threads')) is not int or runtime['interop_threads'] != 1
            or any(type(runtime.get(key)) is not bool for key in
                   ('deterministic_algorithms', 'cudnn_benchmark', 'cudnn_deterministic'))
            or any(type(runtime.get(key)) is not int or runtime[key] != expected for key, expected in (
                ('resident_commitment_bytes', RESIDENT), ('job_commitment_bytes', JOB),
                ('gpu_trace_claim_bytes', GPU_TRACE), ('host_trace_claim_bytes', HOST_TRACE)))):
        raise ValueError('runtime memory contract differs')
    model = measurement.get('model')
    if (type(model) is not dict or model.get('model_id') != MODEL_ID or model.get('model_revision') != REVISION
            or model.get('tokenizer_id') != MODEL_ID or model.get('tokenizer_revision') != REVISION
            or type(model.get('parameter_count')) is not int or model['parameter_count'] != 81912576
            or model.get('weights_sha256') != ASSETS['model.safetensors'][1]
            or model.get('asset_sha256') != {name: digest for name, (_, digest) in ASSETS.items()}
            or model.get('torch_version') != runtime['torch'] or model.get('transformers_version') != runtime['transformers']
            or model.get('dtype') != 'float32' or model.get('attention_implementation') != 'eager'
            or model.get('use_cache') is not False):
        raise ValueError('pinned model metadata differs')
    def check_memory(row):
        phase_estimate('load', row, row)
        if (row['total_bytes'] != runtime['device_total_bytes'] or row['reserved_bytes'] > CAP
                or row['peak_reserved_bytes'] > CAP or row['free_bytes'] < runtime['reserve_bytes'] + CAP
                or row['total_bytes'] < row['reserved_bytes'] + runtime['reserve_bytes'] + CAP):
            raise ValueError('saved memory observation does not satisfy admission')
    rows = measurement['phases']
    for index, (name, row) in enumerate(zip(PHASES, rows), 1):
        start = _read_json(directory / f'phase-{index:02d}-start.json')
        saved = _read_json(directory / f'phase-{index:02d}.json')
        if (type(row) is not dict or saved != row or row.get('schema') != 'pretrained-gpu-trace-phase-v1'
                or row.get('phase') != name or start != {'schema': 'pretrained-gpu-trace-phase-start-v1',
                                                         'phase': name, 'before': row.get('before')}
                or row.get('estimate') != phase_estimate(name, row.get('before'), row.get('after'))
                or type(row.get('commitment_bytes')) is not int
                or row['commitment_bytes'] != (RESIDENT if name == 'load' else JOB)
                or row.get('within_commitment') is not True
                or row['estimate']['estimated_incremental_bytes'] > row['commitment_bytes']):
            raise ValueError('phase observation or estimate differs')
        check_memory(row['before'])
        check_memory(row['after'])
    if (type(rows[0].get('proof')) is not dict
            or rows[0]['proof'].get('parameter_count') != model['parameter_count']
            or rows[0]['proof'].get('weights_sha256') != model['weights_sha256']):
        raise ValueError('load proof differs from pinned model')
    check_memory(rows[0]['proof'].get('before_transfer'))
    initial = _read_json(directory / 'input-setup.json')
    if (initial.get('schema') != 'pretrained-gpu-trace-input-v1' or not exact_ints(initial.get('input_ids'), TOKENS)
            or not exact_ints(initial.get('attention_mask'), [1] * 5) or not exact_ints(initial.get('shape'), [1, 5])
            or initial.get('dtype') != 'int64'):
        raise ValueError('fixed saved input differs')
    check_memory(initial.get('memory'))
    result = measurement.get('result')
    if type(result) is not dict or result.get('direct_gradient_equal') is not True:
        raise ValueError('trace comparison is incomplete')
    baseline, reference, ref_values = (result.get('baseline_logit_sha256'),
        result.get('reference_logit_sha256'), result.get('reference_value_sha256'))
    valid_sha = lambda value: type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None
    if (type(baseline) is not list or len(baseline) != 2 or not all(valid_sha(x) for x in baseline)
            or baseline[0] != baseline[1] or reference != baseline
            or type(ref_values) is not list or len(ref_values) != 18 or not all(map(valid_sha, ref_values))
            or type(result.get('modes')) is not dict or set(result['modes']) != {'direct', 'gradient', 'id'}):
        raise ValueError('baseline or reference evidence differs')
    if (rows[1]['proof'] != {'logit_sha256': [baseline[0]]}
            or rows[2]['proof'] != {'logit_sha256': [baseline[1]]}
            or rows[3]['proof'] != {'logit_sha256': reference, 'value_sha256': ref_values, 'events': 18}):
        raise ValueError('reference proof differs')
    expected = [(forward, layer, kind) for forward in range(2) for layer in (0, 2, 5)
                for kind in ('attn', 'mlp', 'block')]
    for position, mode in enumerate(('direct', 'gradient', 'id'), 4):
        proof = result['modes'][mode]
        if (type(proof) is not dict or rows[position]['proof'] != proof
                or proof.get('logit_sha256') != baseline or type(proof.get('event_count')) is not int
                or proof['event_count'] != 18 or type(proof.get('tensor_bytes')) is not int
                or proof['tensor_bytes'] != (0 if mode == 'id' else 18 * 2 * hidden * 4)
                or proof.get('logits_equal_baseline') is not True
                or proof.get('values_equal_reference') is not (None if mode == 'id' else True)
                or type(proof.get('events')) is not list or len(proof['events']) != 18):
            raise ValueError('mode proof differs')
        for index, (event, (forward, layer, kind)) in enumerate(zip(proof['events'], expected)):
            if type(event) is not dict or type(event.get('metadata')) is not dict:
                raise ValueError('event metadata is unavailable')
            meta = event['metadata']
            path = f'transformer.h.{layer}' + ('' if kind == 'block' else '.' + kind)
            required = {'sequence': index, 'forward_index': forward, 'layer_index': layer,
                'module_kind': kind, 'module_path': path, 'capture_mode': mode, 'invocation': 0,
                'shape': [1, 5, hidden], 'device': device, 'dtype': 'torch.float32',
                'output_path': [0] if kind in ('attn', 'block') else [],
                'batch_positions': [0], 'token_positions': [0, 1], 'batch_axis': 0, 'token_axis': 1,
                'saved_shape': None if mode == 'id' else [1, 2, hidden],
                'saved_device': None if mode == 'id' else 'cpu',
                'saved_dtype': None if mode == 'id' else 'torch.float32', 'schema_version': '1.0',
                'identity': {'run_id': f'{study_id}-{mode}', 'job_id': None, 'request_id': None, 'sample_id': None},
                'generation': {'stage': 'unknown', 'generation_step': None, 'cache_length_before': None,
                               'cache_length_after': None, 'cache_position': None, 'use_cache': None}}
            if (any(meta.get(key) != value for key, value in required.items())
                    or any(type(meta.get(key)) is not int for key in
                           ('sequence', 'forward_index', 'layer_index', 'invocation', 'batch_axis', 'token_axis'))
                    or any(not exact_ints(meta.get(key), required[key]) for key in
                           ('shape', 'output_path', 'batch_positions', 'token_positions'))
                    or (mode != 'id' and not exact_ints(meta.get('saved_shape'), [1, 2, hidden]))
                    or set(meta) != set(required)
                    or event.get('value_sha256') != (None if mode == 'id' else ref_values[index])):
                raise ValueError('event metadata or capture digest differs')
    order = ('baseline_cold', 'baseline_warm', 'reference', 'reference', 'direct', 'direct',
             'gradient', 'gradient', 'id', 'id')
    for index, phase in enumerate(order, 1):
        record = _read_json(directory / f'forward-{index:03d}.json')
        if (type(record.get('index')) is not int or not exact_ints(record.get('shape'), [1, 5, vocab])
                or record != {'schema': 'pretrained-gpu-trace-forward-v1', 'phase': phase, 'index': index,
                      'logit_sha256': baseline[(index - 1) % 2], 'shape': [1, 5, vocab], 'dtype': 'float32'}):
            raise ValueError('saved forward differs')
    if (measurement.get('model_input_trace_weakrefs_dead') is not True
            or measurement.get('runtime_binding_rechecked') is not True
            or type(measurement.get('active_memory_leases')) is not int
            or measurement['active_memory_leases'] != 0
            or measurement.get('cleanup', {}).get('allocated_bytes') != 0
            or measurement.get('cleanup', {}).get('reserved_bytes') != 0):
        raise ValueError('saved local cleanup differs')
    check_memory(measurement['cleanup'])
    return 10


def _cache_path(root, value):
    expected = (Path(root) / 'work' / 'hf-cache' / 'hub').resolve(strict=True)
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.resolve(strict=True) != expected or not expected.is_dir():
        raise ValueError('the fixed offline cache differs')
    return expected


def owner_main(run_dir, cache, run_id, commit, git):
    try:
        if _ROOT['entered'] is not False:
            raise RuntimeError('owner is single-use')
        _ROOT['entered'] = True
        _startup()
        _pristine()
        session = OwnerSession.from_environment()
        _ROOT['session'] = session
        _validate_workspace_spec(session)
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _ROOT['lease'] = lease
        OwnerSession.check(session, lease)
        _validate_workspace_binding(session)
        _validate_runtime_dirs(run_dir)
        repo = Path(__file__).resolve().parents[1]
        root = _workspace_root(repo, session.identity.workspace_root)
        cache = _cache_path(root, cache)
        if _source_identity(repo, git)[0] != commit:
            raise RuntimeError('child source identity differs')
        _verify_offline_cache(cache)
        write_json_new(Path(run_dir) / 'owner.json', {'schema': 'pretrained-gpu-trace-owner-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(), 'source_commit': commit})
        gc.disable()
        gc.collect = _forbid_collection
        _ROOT['gc_disabled'] = True
        OwnerSession.check(session, lease)
        import torch
        measurement = _measure(session, lease, torch, cache, run_dir)
        OwnerSession.check(session, lease)
        executor = _ROOT['executor']
        if _ROOT['cuda_pending'] or _ROOT['sync_pending'] or executor.roots['pending']:
            raise RuntimeError('CUDA completion remains uncertain')
        if _source_identity(repo, git)[0] != commit:
            raise RuntimeError('source changed during the run')
        if _validate_saved(run_dir, measurement) != 10:
            raise RuntimeError('saved forward inventory differs')
        write_json_new(Path(run_dir) / 'candidate.json', {'schema': 'pretrained-gpu-trace-candidate-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': commit, 'measurement': measurement})
        digest = _sha_file(Path(run_dir) / 'candidate.json')
        OwnerSession.check(session, lease)
        write_json_new(Path(run_dir) / 'final.json', {'schema': 'pretrained-gpu-trace-final-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': commit, 'candidate_sha256': digest, 'active_memory_leases': 0})
        OwnerSession.check(session, lease)
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        _ROOT['failure'] = exc
        try:
            executor = _ROOT['executor']
            write_json_new(Path(run_dir) / 'failure.json', {'schema': 'pretrained-gpu-trace-failure-v1',
                'run_id': run_id, 'status': 'incomplete', 'error_type': type(exc).__name__,
                'stage': _ROOT['stage'], 'phase': executor.current if executor is not None else None,
                'completed_forwards': executor.forward_count if executor is not None else 0,
                'cuda_completion_unverified': bool(_ROOT['cuda_pending'] or _ROOT['sync_pending'] or
                    (executor is not None and executor.roots['pending'])), 'cleanup_attested': False},
                max_bytes=16384)
        except BaseException as diagnostic:
            _ROOT['diagnostic_failure'] = diagnostic
        return 1


def controller_main(workspace_root, cache, output, site, smi):
    _startup()
    _pristine()
    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    cache = _cache_path(root, cache)
    site = Path(site).resolve(strict=True)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError('output must be a new absolute directory')
    program_files = os.environ.get('PROGRAMFILES')
    if not program_files or not Path(program_files).is_absolute() or not Path(program_files).is_dir():
        raise RuntimeError('controller telemetry environment is unavailable')
    _capacity(destination.parent, root)
    git = os.environ[_PREFIX + 'GIT_EXE']
    commit, hashes = _source_identity(repo, git)
    if commit != os.environ[_PREFIX + 'SOURCE_COMMIT']:
        raise RuntimeError('controller and bootstrap source identities differ')
    _verify_offline_cache(cache)
    gpu_uuid, before = _smi_snapshot(smi)
    environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATHEXT', 'COMSPEC')
                   if key in os.environ}
    system_root = Path(environment.get('SYSTEMROOT', 'C:/Windows'))
    environment.update({'PATH': str(system_root / 'System32'), 'CUDA_VISIBLE_DEVICES': gpu_uuid,
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', _PREFIX + 'SMI_TOTAL': str(before['total_bytes'])})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    capacity = _capacity(destination.parent, root)
    destination.mkdir(exist_ok=False)
    for name in ('temp', 'hf-cache', 'cuda-cache', 'stage'):
        (destination / name).mkdir(exist_ok=False)
    environment.update({'TEMP': str(destination / 'temp'), 'TMP': str(destination / 'temp'),
                        'HF_HOME': str(destination / 'hf-cache'),
                        'CUDA_CACHE_PATH': str(destination / 'cuda-cache')})
    prefix = Path(tempfile.mkdtemp(prefix='pycache-', dir=destination))
    environment['PYTHONPYCACHEPREFIX'] = str(prefix)
    run_id = secrets.token_hex(16)
    write_json_new(destination / 'started.json', {'schema': 'pretrained-gpu-trace-start-v1',
        'run_id': run_id, 'source_commit': commit, 'source_sha256': hashes,
        'gpu_uuid': gpu_uuid, 'capacity': capacity, 'started_utc': before['timestamp_utc']})
    bootstrap = repo / 'scripts' / 'pretrained_gpu_trace_pilot_bootstrap.py'
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), str(bootstrap),
               '--bootstrap-site', str(site), '--source-commit', commit, '--git-executable', git,
               '--owner', str(destination), str(cache), run_id, commit, git]
    supervision, error, primary = None, None, None
    try:
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, timeout=900, lease_timeout=30,
            cwd=repo, env=environment, inherit_environment=False, require_zero_exit=True)
    except BaseException as exc:
        error, primary = type(exc).__name__, exc
    if supervision is not None:
        write_json_new(destination / 'supervision.json', {'run_id': run_id,
            'supervisor_run_id': supervision.run_id, 'pid': supervision.pid, 'exit_code': supervision.exit_code,
            'clean': supervision.clean, 'timed_out': supervision.timed_out, 'reason': supervision.reason})
    after, presence, telemetry_error = None, {'query': 'not_attempted', 'owner_pid_listed': None}, None
    marker_absent, attestation_absent = None, None
    reaped = supervision is not None and type(supervision.exit_code) is int
    if reaped:
        try:
            after_uuid, after = _smi_snapshot(smi)
            if after_uuid != gpu_uuid:
                raise RuntimeError('device inventory changed')
            presence = _public_presence(_owner_presence_after_exit(smi, supervision.pid))
        except Exception as exc:
            telemetry_error = type(exc).__name__
        marker_absent = not os.path.lexists(identity.marker_path)
        attestation = identity.marker_path.with_name(identity.marker_path.name + '.attestation')
        attestation_absent = not os.path.lexists(attestation)
    outcome = {'schema': 'pretrained-gpu-trace-outcome-v1', 'run_id': run_id, 'source_commit': commit,
        'status': 'incomplete', 'supervisor': _public_supervisor(supervision), 'supervisor_error_type': error,
        'telemetry_error_type': telemetry_error, 'nvidia_smi_before': _public_smi_snapshot(before),
        'nvidia_smi_after': _public_smi_snapshot(after), 'owner_presence_after_exit': presence,
        'marker_absent_at_post_exit_point': marker_absent, 'attestation_absent_at_post_exit_point': attestation_absent,
        'scope': 'one five-token pretrained model input and one device; not a reusable memory profile',
        'limits': ['independent saved-record and cleanup review remains required',
                   'direct-child supervision assumes no CUDA descendants',
                   'device-free and host/disk observations are points, not reservations or worst-case bounds',
                   'allocator peaks exclude other processes and native CUDA allocations']}
    if (supervision is not None and supervision.clean is True and type(supervision.exit_code) is int
            and supervision.exit_code == 0 and supervision.timed_out is False
            and not (destination / 'failure.json').exists()):
        try:
            candidate = _read_json(destination / 'candidate.json')
            final = _read_json(destination / 'final.json', 16384)
            owner = _read_json(destination / 'owner.json', 16384)
            digest = _sha_file(destination / 'candidate.json')
            for record, schema in ((owner, 'pretrained-gpu-trace-owner-v1'),
                                   (candidate, 'pretrained-gpu-trace-candidate-v1'),
                                   (final, 'pretrained-gpu-trace-final-v1')):
                if (record.get('schema') != schema or record.get('run_id') != run_id
                        or record.get('source_commit') != commit or record.get('supervisor_run_id') != supervision.run_id
                        or type(record.get('pid')) is not int or record['pid'] != supervision.pid):
                    raise ValueError('candidate identity differs from the reaped owner')
            if (final.get('candidate_sha256') != digest or type(final.get('active_memory_leases')) is not int
                    or final['active_memory_leases'] != 0):
                raise ValueError('candidate finalization differs')
            count = _validate_saved(destination, candidate['measurement'])
            if (candidate['measurement']['runtime'].get('device') != 'cuda:0'
                    or candidate['measurement']['runtime'].get('device_uuid') != gpu_uuid):
                raise ValueError('candidate device identity differs')
            if _source_identity(repo, git) != (commit, hashes):
                raise ValueError('source changed during the run')
            _verify_offline_cache(cache)
            if (telemetry_error is not None or presence.get('query') != 'ok'
                    or presence.get('owner_pid_listed') is not False or not marker_absent or not attestation_absent):
                raise ValueError('post-exit owner observation was unavailable or still listed the owner')
            outcome.update(status='complete_candidate', candidate_sha256=digest, forward_count=count)
        except Exception as exc:
            outcome['validation_error_type'] = type(exc).__name__
    try:
        write_json_new(destination / 'outcome.json', outcome)
    except BaseException as record_error:
        if isinstance(primary, (KeyboardInterrupt, SystemExit)):
            raise primary from record_error
        raise
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return outcome


def main(argv=None):
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', nargs=5, help=argparse.SUPPRESS)
    parser.add_argument('--workspace-root')
    parser.add_argument('--cache')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.owner:
        return owner_main(*args.owner)
    if not all((args.workspace_root, args.cache, args.output)):
        parser.error('workspace-root, cache and output are required')
    outcome = controller_main(args.workspace_root, args.cache, args.output,
                              os.environ[_PREFIX + 'BOOTSTRAP_SITE'], shutil.which('nvidia-smi'))
    print(json.dumps({'status': outcome['status'], 'source_commit': outcome['source_commit']}))
    return 0 if outcome['status'] == 'complete_candidate' else 1


if __name__ == '__main__':
    if '--owner' in sys.argv[1:]:
        code = 1
        try:
            code = main()
        finally:
            os._exit(code if type(code) is int and 0 <= code <= 255 else 1)
    raise SystemExit(main())
