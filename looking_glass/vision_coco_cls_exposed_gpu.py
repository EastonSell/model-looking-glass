"""One supervised, exploratory CUDA run on the fixed exposed v4 ViT pair."""

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import tempfile
import weakref

from .bert_gpu_evidence import write_json_new
from .bert_gpu_exposed_pair import (_file_identity, _public_presence, _public_smi_snapshot,
    _public_supervisor, _unexpected_hooks, _validate_workspace_binding, _validate_workspace_spec,
    _workspace_root, check_memory)
from .device_identity import resolve_device_identity
from .device_lease import DeviceLease
from .gpu_oneshot import (_owner_presence_after_exit, _runtime_uuid, _smi_snapshot,
                          _verify_controller_import_mode)
from .memory_budget import CudaMemorySnapshot, MemoryBudget
from .supervisor import OwnerSession, run_supervised

MIB = 1024 ** 2
GIB = 1024 ** 3
TRACE = 16 * MIB
HEADROOM = 256 * MIB
MANIFEST_SHA = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'
REFERENCE_SHA = 'c4df950c06b38095bbf750fd7f73e865f50ee2a2087a3bb0425bd171d72f6db7'
STATE_SHA = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
ROLES = {'cat': 405058, 'dog': 196065, 'cat_donor': 69231, 'dog_donor': 142769}
_SOURCE_PATHS = ('looking_glass/__init__.py', 'looking_glass/vision_coco_cls_exposed_gpu.py',
    'looking_glass/vision_coco_cls_gpu_adapter.py', 'looking_glass/vision_coco_cls_study.py',
    'looking_glass/vision_coco_study.py', 'looking_glass/vision_coco_recognized_screen.py',
    'looking_glass/bert_gpu_exposed_pair.py',
    'looking_glass/bert_gpu_evidence.py', 'looking_glass/gpu_oneshot.py',
    'looking_glass/device_identity.py', 'looking_glass/device_lease.py',
    'looking_glass/supervisor.py', 'looking_glass/memory_budget.py',
    'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
    'looking_glass/model_ownership.py', 'experiments/vision_coco_recognized_v4.json',
    'docs/VISION_COCO_EXPOSED_CLS_GPU_V1.md',
    'tools/prepare_vision_cls_exposed_gpu_reference.py',
    'tools/observe_vision_cls_exposed_gpu.py',
    'scripts/vision_coco_cls_exposed_gpu_bootstrap.py')

_ROOT = {'entered': False, 'pending': False, 'gc_disabled': False, 'session': None,
         'lease': None, 'budget': None, 'memory_lease': None, 'claims_ready': False,
         'model': None, 'adapter': None, 'evidence': None, 'failure': None,
         'diagnostic_failure': None, 'completed_forwards': 0}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, limit):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('file exceeds its fixed bound')
    return raw


def _read_reference(directory, expected_sha=REFERENCE_SHA):
    directory = Path(directory).resolve(strict=True)
    raw = _read(directory / 'reference.json', MIB)
    if _sha(raw) != expected_sha:
        raise ValueError('reference manifest differs')
    record = json.loads(raw)
    expected_pair = {'pair_id': 'pair-00', 'split': 'heldout', 'box_tokens': [90, 91, 104, 105]}
    expected_pair.update({role: {'image_id': image_id} for role, image_id in ROLES.items()})
    if (record.get('schema') != 'vision-cls-exposed-gpu-reference-v1'
            or record.get('status') != 'complete' or type(record.get('model_forwards')) is not int
            or record['model_forwards'] != 0 or record.get('manifest_sha256') != MANIFEST_SHA
            or record.get('pair') != expected_pair):
        raise ValueError('reference identity differs')
    names = {f'{role}-{suffix}' for role in ROLES for suffix in ('pixels.f32', 'cpu-logits.f32')}
    if type(record.get('files')) is not dict or set(record['files']) != names:
        raise ValueError('reference inventory differs')
    files = {}
    for name in sorted(names):
        value = _read(directory / name, MIB)
        expected_size = 602112 if name.endswith('pixels.f32') else 4000
        entry = record['files'][name]
        if (type(entry) is not dict or type(entry.get('bytes')) is not int
                or entry['bytes'] != expected_size or len(value) != expected_size
                or entry.get('sha256') != _sha(value)):
            raise ValueError('reference file differs')
        files[name] = value
    return record, files


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
        value = _read(repo / name, 4 * MIB)
        if value != run('show', commit + ':' + name):
            raise RuntimeError('source differs from the committed tree')
        hashes[name] = _sha(value)
    if hashes['experiments/vision_coco_recognized_v4.json'] != MANIFEST_SHA:
        raise RuntimeError('public manifest differs')
    return commit, hashes


def _asset_inventory(repo, assets, reference):
    manifest = json.loads(_read(repo / 'experiments/vision_coco_recognized_v4.json', 2 * MIB))
    expected = manifest['model_files']
    if (set(expected) != {'config.json', 'preprocessor_config.json', 'model.safetensors'}
            or reference.get('model_files') != expected):
        raise ValueError('model asset inventory differs')
    for name, digest in expected.items():
        if _file_identity(Path(assets) / name, 32 * MIB)[1] != digest:
            raise ValueError('pinned model asset differs')
    return expected


def _startup():
    _verify_controller_import_mode()
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or os.environ.get('LOOKING_GLASS_VISION_BOOTSTRAP_PID') != str(os.getpid())):
        raise RuntimeError('isolated source bootstrap is required')


def _pristine():
    if any(name in sys.modules for name in ('torch', 'numpy', 'transformers')):
        raise RuntimeError('model libraries were imported before the physical lease')


def _forbid_collection(*_args, **_kwargs):
    raise RuntimeError('explicit cyclic collection is forbidden while the owner is active')


def _held(session, lease):
    OwnerSession.check(session, lease)
    if _ROOT['session'] is not session or _ROOT['lease'] is not lease:
        raise RuntimeError('physical lease custody changed')
    if _ROOT['gc_disabled'] and (gc.isenabled() or gc.collect is not _forbid_collection):
        raise RuntimeError('cyclic collection policy changed')


def _state_copy(value):
    return value.detach().cpu().contiguous()


def _check_eval(model):
    if any(type(module.training) is not bool or module.training for module in model.modules()):
        raise RuntimeError('model evaluation state changed')


def _state_digest(model, check=None):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if check is not None:
            check()
        cpu = _state_copy(value)
        if check is not None:
            check()
        digest.update(name.encode() + b'\0' + str(cpu.dtype).encode() + b'\0')
        digest.update(json.dumps(list(cpu.shape), separators=(',', ':')).encode() + b'\0')
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _diff(left, right):
    if len(left) != 4000 or len(right) != 4000:
        raise ValueError('CPU/GPU logit comparison has incorrect size')
    differences = [abs(a[0] - b[0]) for a, b in zip(struct.iter_unpack('<f', left),
                                                   struct.iter_unpack('<f', right))]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError('CPU/GPU logit comparison is nonfinite')
    return {'max_abs': max(differences), 'mean_abs': math.fsum(differences) / len(differences),
            'count': len(differences)}


def _host_memory():
    import ctypes
    if sys.platform != 'win32':
        raise RuntimeError('host memory observations require Windows')
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong),
                    ('peak_working_set', ctypes.c_size_t), ('working_set', ctypes.c_size_t),
                    ('peak_paged_pool', ctypes.c_size_t), ('paged_pool', ctypes.c_size_t),
                    ('peak_nonpaged_pool', ctypes.c_size_t), ('nonpaged_pool', ctypes.c_size_t),
                    ('pagefile', ctypes.c_size_t), ('peak_pagefile', ctypes.c_size_t),
                    ('private_usage', ctypes.c_size_t)]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
    get_memory.restype = ctypes.c_int
    if not get_memory(get_process(), ctypes.byref(value), value.cb):
        raise OSError('process memory observation unavailable')
    return {'working_set_bytes': value.working_set, 'peak_working_set_bytes': value.peak_working_set,
            'private_bytes': value.private_usage}


def _labels(status):
    if type(status) is not dict or set(status) != {'cat', 'dog'} or any(
            value not in ('matchable', 'dose-ineligible', 'dose-unmatchable') for value in status.values()):
        raise ValueError('dose branches differ')
    labels = [f'pair-00:{role}:initial' for role in ('cat', 'dog', 'cat_donor', 'dog_donor')]
    for side in ('cat', 'dog'):
        actions = ['repeat', 'null_cls', 'self_cls', 'null_spatial', 'self_spatial', 'full_self', 'full_swap']
        if status[side] == 'matchable':
            actions += [f'{site}_{kind}' for site in ('cls', 'spatial')
                        for kind in ('opposite', 'same_species', 'random')]
        actions += ['final_spatial', 'post_clean']
        labels.extend(f'pair-00:{side}:{action}' for action in actions)
    return labels


def _layout_shapes(label):
    action = label.rsplit(':', 1)[-1]
    shapes = {'logits': [1, 1000]}
    if action in ('initial', 'repeat', 'post_clean'):
        shapes.update(early=[1, 197, 192], late=[1, 197, 192])
        return shapes
    shapes.update(before_all=[1, 197, 192], after_all=[1, 197, 192])
    if action in ('full_self', 'full_swap'):
        shapes['requested'] = [1, 197, 192]
        return shapes
    selected = 1 if action.startswith(('null_cls', 'self_cls', 'cls_')) else 4
    for index in range(selected):
        shapes[f'before_{index}'] = [192]
        shapes[f'after_{index}'] = [192]
    if not action.startswith('null_'):
        shapes['requested'] = [selected, 192]
    return shapes


def _validate_stages(directory, measurement, count):
    runtime = measurement.get('runtime')
    if type(runtime) is not dict:
        raise ValueError('candidate runtime is missing')
    total, cap, commitment = (runtime.get('device_total_bytes'), runtime.get('allocator_cap_bytes'),
                              runtime.get('commitment_bytes'))
    if (type(total) is not int or not 7 * GIB <= total <= 9 * GIB or type(cap) is not int
            or cap != total // 4 or type(commitment) is not int or commitment != cap + HEADROOM
            or runtime.get('gpu_trace_claim_bytes') != TRACE or type(runtime.get('gpu_trace_claim_bytes')) is not int
            or runtime.get('host_trace_claim_bytes') != TRACE or type(runtime.get('host_trace_claim_bytes')) is not int
            or runtime.get('torch') != '2.8.0+cu128' or runtime.get('cuda_build') != '12.8'
            or runtime.get('transformers') != '4.57.3' or runtime.get('numpy') != '2.4.6'
            or runtime.get('device') != 'cuda:0' or runtime.get('device_name') != 'NVIDIA GeForce RTX 2070 SUPER'
            or runtime.get('compute_capability') != [7, 5] or runtime.get('dtype') != 'float32'
            or runtime.get('attention') != 'eager' or runtime.get('inference_mode') is not True
            or runtime.get('default_dtype') != 'torch.float32'
            or runtime.get('binary64_rounding') != 'nearest-even' or runtime.get('denormals_preserved') is not True
            or runtime.get('deterministic_algorithms') is not True or runtime.get('matmul_tf32') is not False
            or runtime.get('cudnn_tf32') is not False or runtime.get('cudnn_benchmark') is not False
            or runtime.get('cudnn_deterministic') is not True or runtime.get('autocast_enabled') is not False
            or runtime.get('allocator') != 'native' or type(runtime.get('threads')) is not int
            or runtime['threads'] != 1 or type(runtime.get('interop_threads')) is not int
            or runtime['interop_threads'] != 1 or type(runtime.get('seed')) is not int
            or runtime['seed'] != 17 or type(runtime.get('nvidia_total_bytes')) is not int
            or abs(runtime['nvidia_total_bytes'] - total) > 2 * MIB):
        raise ValueError('candidate runtime or admission differs')
    expected = (['before_model_load', 'before_model_transfer', 'after_model_transfer']
                + [f'after_forward_{index}' for index in range(1, count + 1)]
                + ['after_final_state_check', 'after_model_disposal', 'after_cleanup'])
    if measurement.get('stages') != expected:
        raise ValueError('memory stage inventory differs')
    previous_allocated = previous_reserved = previous_host_peak = 0
    entries = {}
    for index, name in enumerate(expected, 1):
        item = json.loads(_read(Path(directory) / f'stage-{index:03d}.json', 16384))
        if (type(item) is not dict or item.get('stage') != name
                or item.get('peak_scope') != 'since_pre_model_reset'):
            raise ValueError('saved memory stage differs')
        memory = item.get('memory')
        check_memory(memory, expected_total=total, cap=cap, commitment=commitment)
        host = item.get('host_memory')
        if (type(host) is not dict or set(host) != {'working_set_bytes', 'peak_working_set_bytes', 'private_bytes'}
                or any(type(value) is not int or value < 0 for value in host.values())
                or host['peak_working_set_bytes'] < host['working_set_bytes']
                or host['peak_working_set_bytes'] < previous_host_peak):
            raise ValueError('host memory observation differs')
        if (memory['peak_allocated_bytes'] < previous_allocated
                or memory['peak_reserved_bytes'] < previous_reserved):
            raise ValueError('cumulative peak counters decreased')
        previous_allocated, previous_reserved = memory['peak_allocated_bytes'], memory['peak_reserved_bytes']
        previous_host_peak = host['peak_working_set_bytes']
        entries[name] = memory
    if (measurement.get('initial_memory') != entries['before_model_load']
            or measurement.get('memory_before_transfer') != entries['before_model_transfer']
            or measurement.get('cleanup_memory') != entries['after_cleanup']):
        raise ValueError('candidate memory endpoints differ')


def _validate_saved(directory, measurement):
    if type(measurement) is not dict or type(measurement.get('results')) is not list or len(measurement['results']) != 1:
        raise ValueError('candidate results differ')
    result = measurement['results'][0]
    labels = _labels(result.get('dose_matchability'))
    if (measurement.get('reference_sha256') != REFERENCE_SHA or measurement.get('model_state_sha256') != STATE_SHA
            or result.get('pair_id') != 'pair-00' or result.get('status') not in
            ('positive', 'negative', 'dose-ineligible', 'dose-unmatchable')
            or type(measurement.get('forward_count')) is not int
            or measurement['forward_count'] != len(labels)):
        raise ValueError('candidate schedule differs')
    for index, label in enumerate(labels, 1):
        row = json.loads(_read(Path(directory) / 'data' / f'forward-{index:03d}.json', MIB))
        raw = _read(Path(directory) / 'data' / f'forward-{index:03d}.bin', MIB)
        if (type(row.get('submission')) is not int or row['submission'] != index or row.get('label') != label
                or row.get('file') != f'forward-{index:03d}.bin' or row.get('sha256') != _sha(raw)
                or row.get('dtype') != 'little-endian-float32' or type(row.get('layout')) is not dict):
            raise ValueError('saved forward identity differs')
        shapes = _layout_shapes(label)
        if set(row['layout']) != set(shapes) or any(
                type(row['layout'][key]) is not dict or row['layout'][key].get('shape') != shape
                for key, shape in shapes.items()):
            raise ValueError('saved tensor inventory differs')
        cursor = 0
        for item in sorted(row['layout'].values(), key=lambda x: x.get('offset', -1)):
            shape = item.get('shape')
            if (type(shape) is not list or not shape or any(type(x) is not int or x <= 0 for x in shape)
                    or type(item.get('bytes')) is not int or item['bytes'] != 4 * math.prod(shape)
                    or type(item.get('offset')) is not int or item['offset'] != cursor):
                raise ValueError('saved tensor layout differs')
            cursor += item['bytes']
        if cursor != len(raw) or 'logits' not in row['layout']:
            raise ValueError('saved tensor bytes differ')
    cleanup = measurement.get('cleanup_memory')
    if (type(cleanup) is not dict or type(cleanup.get('allocated_bytes')) is not int
            or type(cleanup.get('reserved_bytes')) is not int
            or cleanup['allocated_bytes'] != 0 or cleanup['reserved_bytes'] != 0
            or measurement.get('model_and_parameter_weakrefs_dead') is not True):
        raise ValueError('candidate cleanup differs')
    _validate_stages(directory, measurement, len(labels))
    return len(labels)


def _measure(session, lease, torch, assets, reference, files, directory):
    import numpy as np
    import transformers
    from transformers import ViTForImageClassification
    from .vision_coco_cls_gpu_adapter import ForwardAdapter
    from .vision_coco_cls_study import run_comparison
    from .vision_coco_study import Evidence, _bits, check_denormals
    from .vision_coco_recognized_screen import check_rounding

    check_rounding()
    if (torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8'
            or transformers.__version__ != '4.57.3' or np.__version__ != '2.4.6'
            or not callable(getattr(torch._C, '_cuda_clearCublasWorkspaces', None))):
        raise RuntimeError('pinned runtime or cleanup API differs')

    def check():
        _held(session, lease)
        if _ROOT['claims_ready']:
            budget, held = _ROOT['budget'], _ROOT['memory_lease']
            if (budget is None or held is None or budget.active_leases() != (held,)
                    or held.trace_bytes('gpu') != TRACE or held.trace_bytes('host') != TRACE):
                raise RuntimeError('memory or trace custody changed')
        _held(session, lease)

    def guarded(fn, *args):
        check()
        value = fn(*args)
        check()
        return value

    def sync():
        guarded(torch.cuda.synchronize, 0)

    def transaction(fn):
        check()
        if _ROOT['pending']:
            raise RuntimeError('unverified CUDA work remains')
        _ROOT['pending'] = True
        result = fn()
        sync()
        _ROOT['pending'] = False
        return result

    _ROOT['pending'] = True
    if guarded(torch.cuda.is_initialized) or guarded(torch.cuda.device_count) != 1:
        raise RuntimeError('unexpected CUDA initialization or visible devices')
    properties = guarded(torch.cuda.get_device_properties, 0)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=1)
    if (properties.name != 'NVIDIA GeForce RTX 2070 SUPER' or (properties.major, properties.minor) != (7, 5)
            or type(properties.total_memory) is not int or not 7 * GIB <= properties.total_memory <= 9 * GIB):
        raise RuntimeError('device differs from the fixed execution contract')
    nvidia_total_text = os.environ.get('LOOKING_GLASS_VISION_SMI_TOTAL', '')
    if (not nvidia_total_text.isascii() or not nvidia_total_text.isdecimal()
            or not 7 * GIB <= int(nvidia_total_text) <= 9 * GIB
            or abs(int(nvidia_total_text) - properties.total_memory) > 2 * MIB):
        raise RuntimeError('CUDA and NVIDIA device totals differ')
    nvidia_total = int(nvidia_total_text)
    if guarded(torch.cuda.get_allocator_backend) != 'native':
        raise RuntimeError('native CUDA allocator is required')
    guarded(torch.cuda.set_per_process_memory_fraction, .25, 0)
    cap = int(.25 * properties.total_memory)
    if guarded(torch.cuda.get_per_process_memory_fraction, 0) != cap / properties.total_memory:
        raise RuntimeError('allocator cap readback differs')
    guarded(torch.cuda.reset_peak_memory_stats, 0)
    sync()
    _ROOT['pending'] = False

    def counters():
        sync()
        pair = guarded(torch.cuda.mem_get_info, 0)
        if (type(pair) not in (tuple, list) or len(pair) != 2
                or any(type(value) is not int for value in pair)
                or pair[1] != properties.total_memory or not 0 <= pair[0] <= pair[1]):
            raise ValueError('invalid device free/total observation')
        allocated = guarded(torch.cuda.memory_allocated, 0)
        if type(allocated) is not int or not 0 <= allocated <= properties.total_memory:
            raise ValueError('invalid allocated observation')
        reserved = guarded(torch.cuda.memory_reserved, 0)
        if type(reserved) is not int or not allocated <= reserved <= properties.total_memory:
            raise ValueError('invalid reserved observation')
        peak_allocated = guarded(torch.cuda.max_memory_allocated, 0)
        if type(peak_allocated) is not int or not allocated <= peak_allocated <= properties.total_memory:
            raise ValueError('invalid peak allocated observation')
        peak_reserved = guarded(torch.cuda.max_memory_reserved, 0)
        if type(peak_reserved) is not int or not reserved <= peak_reserved <= properties.total_memory:
            raise ValueError('invalid peak reserved observation')
        value = {'device_free_bytes': pair[0], 'device_total_bytes': pair[1],
                 'allocated_bytes': allocated, 'reserved_bytes': reserved,
                 'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved}
        check_memory(value, expected_total=properties.total_memory, cap=cap, commitment=cap + HEADROOM)
        return value

    stages = []
    def snapshot(label):
        value = transaction(counters)
        write_json_new(Path(directory) / f'stage-{len(stages) + 1:03d}.json',
                       {'stage': label, 'memory': value, 'host_memory': _host_memory(),
                        'peak_scope': 'since_pre_model_reset'})
        stages.append(label)
        return value

    initial = snapshot('before_model_load')
    def telemetry():
        value = counters()
        return CudaMemorySnapshot(value['device_free_bytes'], value['device_total_bytes'],
                                  value['allocated_bytes'], value['reserved_bytes'])
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=TRACE, max_host_trace_bytes=TRACE)
    _ROOT['budget'] = budget
    _ROOT['pending'] = True
    memory_lease = budget.acquire(cap + HEADROOM, max_gpu_trace_bytes=TRACE, max_host_trace_bytes=TRACE)
    _ROOT['memory_lease'] = memory_lease
    memory_lease.reserve_trace('gpu', TRACE)
    memory_lease.reserve_trace('host', TRACE)
    _ROOT['claims_ready'] = True
    check()
    sync()
    _ROOT['pending'] = False

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.random.default_generator.manual_seed(17)
    transaction(lambda: torch.cuda.manual_seed(17))
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise RuntimeError('float32 denormal preservation is unavailable')
    check_denormals()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    def numeric_settings():
        return {'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'autocast_enabled': torch.is_autocast_enabled('cuda'),
                'default_dtype': str(torch.get_default_dtype()),
                'float32_matmul_precision': torch.get_float32_matmul_precision()}
    settings = numeric_settings()
    if (settings['threads'] != 1 or settings['interop_threads'] != 1
            or not settings['deterministic_algorithms'] or settings['matmul_tf32'] or settings['cudnn_tf32']
            or settings['cudnn_benchmark'] or not settings['cudnn_deterministic'] or settings['autocast_enabled']
            or settings['default_dtype'] != 'torch.float32'):
        raise RuntimeError('numeric execution settings differ')
    def numeric_guard():
        check_rounding()
        check_denormals()
        if _ROOT['model'] is not None:
            _check_eval(_ROOT['model'])
        if numeric_settings() != settings:
            raise RuntimeError('numeric execution settings changed')
    runtime = {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
               'transformers': transformers.__version__, 'numpy': np.__version__,
               'python': sys.version.split()[0], 'device': 'cuda:0', 'device_uuid': runtime_uuid,
               'device_name': properties.name,
               'compute_capability': [properties.major, properties.minor], 'device_total_bytes': properties.total_memory,
               'nvidia_total_bytes': nvidia_total,
               'dtype': 'float32', 'attention': 'eager', 'inference_mode': True, 'seed': 17,
               'binary64_rounding': 'nearest-even', 'denormals_preserved': True,
               'allocator': 'native', 'allocator_cap_bytes': cap, 'commitment_bytes': cap + HEADROOM,
               'gpu_trace_claim_bytes': TRACE, 'host_trace_claim_bytes': TRACE, **settings}

    model, info = ViTForImageClassification.from_pretrained(str(assets), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True, attn_implementation='eager')
    _ROOT['model'] = model
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('checkpoint loaded incompletely')
    model.eval()
    _check_eval(model)
    cfg = model.config
    if ((cfg.hidden_size, cfg.num_hidden_layers, cfg.num_labels, cfg.patch_size, cfg.image_size)
            != (192, 12, 1000, 16, 224) or cfg._attn_implementation != 'eager'
            or _state_digest(model) != STATE_SHA or any(p.dtype != torch.float32 for p in model.parameters())):
        raise RuntimeError('checkpoint geometry or state differs')
    before_transfer = snapshot('before_model_transfer')
    transaction(lambda: model.to('cuda:0'))
    if any(p.dtype != torch.float32 or p.device.type != 'cuda' or p.device.index != 0 for p in model.parameters()):
        raise RuntimeError('model transfer differs')
    snapshot('after_model_transfer')
    pixels = {image_id: torch.frombuffer(bytearray(files[f'{role}-pixels.f32']), dtype=torch.float32)
              .reshape(1, 3, 224, 224) for role, image_id in ROLES.items()}
    role_by_id = {image_id: role for role, image_id in ROLES.items()}
    evidence = Evidence(Path(directory) / 'data', 34)
    _ROOT['evidence'] = evidence
    adapter = ForwardAdapter(check=check, synchronize=sync, device='cuda:0')
    _ROOT['adapter'] = adapter
    comparisons = {}
    def check_clean(image_id, logits):
        if image_id not in role_by_id or image_id in comparisons:
            raise ValueError('clean image inventory differs')
        role = role_by_id[image_id]
        comparisons[str(image_id)] = _diff(files[f'{role}-cpu-logits.f32'], _bits(logits))
    def forward(*args, **kwargs):
        check()
        if _ROOT['pending']:
            raise RuntimeError('unverified CUDA work remains')
        _ROOT['pending'] = True
        value = adapter(*args, **kwargs)
        _ROOT['pending'] = False
        _ROOT['completed_forwards'] += 1
        snapshot(f'after_forward_{_ROOT["completed_forwards"]}')
        return value
    results = run_comparison(model, [reference['pair']], pixels, evidence,
                             check_clean=check_clean, numeric_guard=numeric_guard, forward_fn=forward)
    if set(comparisons) != {str(x) for x in ROLES.values()}:
        raise RuntimeError('clean comparison inventory differs')
    if (evidence.submissions != _ROOT['completed_forwards'] or len(evidence.records) != evidence.submissions
            or [record['label'] for record in evidence.records] != _labels(results[0]['dose_matchability'])):
        raise RuntimeError('completed action schedule differs')
    if transaction(lambda: _state_digest(model, check=check)) != STATE_SHA or _unexpected_hooks(model):
        raise RuntimeError('model state or hooks changed')
    numeric_guard()
    snapshot('after_final_state_check')
    refs = [weakref.ref(model)] + [weakref.ref(value) for value in model.parameters()]
    refs += [weakref.ref(value) for value in model.buffers()]
    _ROOT['model'] = None
    del model
    check()
    if any(ref() is not None for ref in refs):
        raise RuntimeError('model or parameter objects remained after guarded disposal')
    transaction(torch.cuda.empty_cache)
    snapshot('after_model_disposal')
    transaction(torch._C._cuda_clearCublasWorkspaces)
    transaction(torch.cuda.empty_cache)
    cleanup = snapshot('after_cleanup')
    if cleanup['allocated_bytes'] != 0 or cleanup['reserved_bytes'] != 0:
        raise RuntimeError('allocator bytes remain after cleanup')
    return {'reference_sha256': REFERENCE_SHA, 'model_state_sha256': STATE_SHA,
            'forward_count': evidence.submissions, 'results': results, 'runtime': runtime,
            'clean_cpu_gpu_logit_differences': comparisons, 'stages': stages,
            'initial_memory': initial, 'memory_before_transfer': before_transfer,
            'cleanup_memory': cleanup, 'model_and_parameter_weakrefs_dead': True}


def _validate_runtime_dirs(run_dir):
    root = Path(run_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('attempt directory is unavailable')
    for variable, name in (('TEMP', 'temp'), ('TMP', 'temp'),
                           ('HF_HOME', 'hf-cache'), ('CUDA_CACHE_PATH', 'cuda-cache')):
        value = os.environ.get(variable)
        expected = root / name
        if (not value or not Path(value).is_absolute() or not expected.is_dir()
                or expected.resolve(strict=True) != expected
                or Path(value).resolve(strict=True) != expected):
            raise ValueError('runtime temporary/cache directory differs from the attempt')


def owner_main(run_dir, assets, run_id, commit, git):
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
        _held(session, lease)
        _validate_workspace_binding(session)
        _validate_runtime_dirs(run_dir)
        repo = Path(__file__).resolve().parents[1]
        if _source_identity(repo, git)[0] != commit:
            raise RuntimeError('child source identity differs')
        reference, files = _read_reference(os.environ['LOOKING_GLASS_VISION_REFERENCE'])
        _asset_inventory(repo, assets, reference)
        write_json_new(Path(run_dir) / 'owner.json', {'schema': 'vision-cls-exposed-gpu-owner-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(), 'source_commit': commit})
        gc.disable()
        gc.collect = _forbid_collection
        _ROOT['gc_disabled'] = True
        _held(session, lease)
        import torch
        measurement = _measure(session, lease, torch, assets, reference, files, run_dir)
        _held(session, lease)
        if _ROOT['pending'] or _ROOT['adapter'].pending or _ROOT['adapter'].failed:
            raise RuntimeError('CUDA completion or adapter state is uncertain')
        if _source_identity(repo, git)[0] != commit:
            raise RuntimeError('source changed during the run')
        _asset_inventory(repo, assets, reference)
        _read_reference(os.environ['LOOKING_GLASS_VISION_REFERENCE'])
        if _validate_saved(run_dir, measurement) != measurement['forward_count']:
            raise RuntimeError('saved action inventory differs')
        write_json_new(Path(run_dir) / 'candidate.json', {'schema': 'vision-cls-exposed-gpu-candidate-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': commit, 'measurement': measurement})
        candidate_sha = _file_identity(Path(run_dir) / 'candidate.json', MIB)[1]
        held = _ROOT['memory_lease']
        _held(session, lease)
        held.release_trace('gpu', TRACE)
        _held(session, lease)
        held.release_trace('host', TRACE)
        _held(session, lease)
        held.release()
        _ROOT['claims_ready'] = False
        if _ROOT['budget'].active_leases():
            raise RuntimeError('memory commitment remains')
        _held(session, lease)
        write_json_new(Path(run_dir) / 'final.json', {'schema': 'vision-cls-exposed-gpu-final-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': commit, 'candidate_sha256': candidate_sha, 'active_memory_leases': 0})
        _held(session, lease)
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        _ROOT['failure'] = exc
        try:
            evidence, adapter = _ROOT['evidence'], _ROOT['adapter']
            active = evidence.active if evidence is not None and type(evidence.active) is dict else {}
            allowed = ('phase', 'submission', 'label', 'saved_file', 'saved_sha256', 'tensor_layout',
                       'return_observed')
            write_json_new(Path(run_dir) / 'failure.json', {'schema': 'vision-cls-exposed-gpu-failure-v1',
                'run_id': run_id, 'status': 'incomplete', 'error_type': type(exc).__name__,
                'completed_forwards': _ROOT['completed_forwards'], 'cuda_completion_unverified': _ROOT['pending'],
                'adapter_pending': adapter.pending if adapter is not None else None,
                'adapter_failed': adapter.failed if adapter is not None else None,
                'cleanup_attested': False, 'partial': {key: active[key] for key in allowed if key in active}},
                max_bytes=32768)
        except BaseException as diagnostic:
            _ROOT['diagnostic_failure'] = diagnostic
        return 1


def _capacity_policy(output_free, workspace_free, same_volume):
    if (type(output_free) is not int or type(workspace_free) is not int or type(same_volume) is not bool
            or output_free < 0 or workspace_free < 0):
        raise ValueError('invalid output or workspace free-space observation')
    output_required = 4 * GIB if same_volume else 3 * GIB
    if output_free < output_required or workspace_free < GIB:
        raise RuntimeError('insufficient output or workspace volume headroom')
    return output_required


def _capacity(path, workspace):
    import ctypes
    import shutil
    if sys.platform != 'win32':
        raise RuntimeError('fixed owner requires Windows')
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                    ('total_pagefile', ctypes.c_ulonglong), ('available_pagefile', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                    ('available_extended_virtual', ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('host memory observation unavailable')
    disk_free = shutil.disk_usage(path).free
    workspace_free = shutil.disk_usage(workspace).free
    same_volume = os.stat(path).st_dev == os.stat(workspace).st_dev
    output_required = _capacity_policy(disk_free, workspace_free, same_volume)
    if status.available_physical < 2 * GIB:
        raise RuntimeError('insufficient host or output-volume headroom')
    return {'disk_free_bytes': disk_free, 'host_available_physical_bytes': status.available_physical,
            'workspace_volume_free_bytes': workspace_free, 'same_volume': same_volume,
            'output_required_bytes': output_required, 'workspace_residual_floor_bytes': GIB,
            'disk_attempt_bound_bytes': GIB, 'disk_remaining_headroom_bytes': 2 * GIB}


def controller_main(workspace_root, assets, reference_path, output, site, smi):
    _startup()
    _pristine()
    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    assets = Path(assets).resolve(strict=True)
    reference_path = Path(reference_path).resolve(strict=True)
    site = Path(site).resolve(strict=True)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError('output must be a new absolute directory')
    program_files = os.environ.get('PROGRAMFILES')
    if not program_files or not Path(program_files).is_absolute() or not Path(program_files).is_dir():
        raise RuntimeError('controller telemetry environment is unavailable')
    capacity = _capacity(destination.parent, root)
    git = os.environ['LOOKING_GLASS_VISION_GIT_EXE']
    commit, hashes = _source_identity(repo, git)
    if commit != os.environ['LOOKING_GLASS_VISION_SOURCE_COMMIT']:
        raise RuntimeError('controller and bootstrap source identities differ')
    reference, _ = _read_reference(reference_path)
    assets_hashes = _asset_inventory(repo, assets, reference)
    gpu_uuid, before = _smi_snapshot(smi)
    environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATHEXT', 'COMSPEC')
                   if key in os.environ}
    system_root = Path(environment.get('SYSTEMROOT', 'C:/Windows'))
    environment.update({'PATH': str(system_root / 'System32'), 'CUDA_VISIBLE_DEVICES': gpu_uuid,
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'LOOKING_GLASS_VISION_REFERENCE': str(reference_path),
        'LOOKING_GLASS_VISION_SMI_TOTAL': str(before['total_bytes'])})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    # Repeat the point just before creating the bounded attempt; concurrent writers remain uncontrolled.
    capacity = _capacity(destination.parent, root)
    destination.mkdir(exist_ok=False)
    for name in ('temp', 'hf-cache', 'cuda-cache'):
        (destination / name).mkdir(exist_ok=False)
    environment.update({'TEMP': str(destination / 'temp'), 'TMP': str(destination / 'temp'),
                        'HF_HOME': str(destination / 'hf-cache'),
                        'CUDA_CACHE_PATH': str(destination / 'cuda-cache')})
    prefix = Path(tempfile.mkdtemp(prefix='pycache-', dir=destination))
    environment['PYTHONPYCACHEPREFIX'] = str(prefix)
    run_id = secrets.token_hex(16)
    write_json_new(destination / 'started.json', {'schema': 'vision-cls-exposed-gpu-start-v1',
        'run_id': run_id, 'source_commit': commit, 'source_sha256': hashes, 'asset_sha256': assets_hashes,
        'reference_sha256': REFERENCE_SHA, 'gpu_uuid': gpu_uuid, 'capacity': capacity,
        'started_utc': before['timestamp_utc']})
    bootstrap = repo / 'scripts' / 'vision_coco_cls_exposed_gpu_bootstrap.py'
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), str(bootstrap),
               '--bootstrap-site', str(site), '--source-commit', commit, '--git-executable', git,
               '--owner', str(destination), str(assets), run_id, commit, git]
    supervision, error, primary = None, None, None
    try:
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, timeout=900, lease_timeout=30,
            cwd=repo, env=environment, inherit_environment=False, require_zero_exit=True)
    except BaseException as exc:
        error = type(exc).__name__
        primary = exc
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
    outcome = {'schema': 'vision-cls-exposed-gpu-outcome-v1', 'run_id': run_id, 'source_commit': commit,
        'status': 'incomplete', 'supervisor': _public_supervisor(supervision), 'supervisor_error_type': error,
        'telemetry_error_type': telemetry_error, 'nvidia_smi_before': _public_smi_snapshot(before),
        'nvidia_smi_after': _public_smi_snapshot(after), 'owner_presence_after_exit': presence,
        'marker_absent_at_post_exit_point': marker_absent,
        'attestation_absent_at_post_exit_point': attestation_absent,
        'scope': 'one previously exposed pair; exploratory CLS versus spatial CUDA comparison',
        'limits': ['scientific interpretation requires independent saved-byte replay',
                   'direct-child supervision assumes no CUDA descendants',
                   'memory and disk observations are points, not reservations or worst-case bounds',
                   'source and receipts assume trusted files and native runtime',
                   'prior CPU results and donor selection were already known']}
    if (supervision is not None and supervision.clean is True and type(supervision.exit_code) is int
            and supervision.exit_code == 0 and supervision.timed_out is False
            and not (destination / 'failure.json').exists()):
        try:
            candidate = json.loads(_read(destination / 'candidate.json', MIB))
            final = json.loads(_read(destination / 'final.json', 16384))
            owner = json.loads(_read(destination / 'owner.json', 16384))
            digest = _file_identity(destination / 'candidate.json', MIB)[1]
            for record, schema in ((owner, 'vision-cls-exposed-gpu-owner-v1'),
                                   (candidate, 'vision-cls-exposed-gpu-candidate-v1'),
                                   (final, 'vision-cls-exposed-gpu-final-v1')):
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
            _asset_inventory(repo, assets, reference)
            _read_reference(reference_path)
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
    parser.add_argument('--assets')
    parser.add_argument('--reference')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.owner:
        return owner_main(*args.owner)
    if not all((args.workspace_root, args.assets, args.reference, args.output)):
        parser.error('workspace-root, assets, reference and output are required')
    outcome = controller_main(args.workspace_root, args.assets, args.reference, args.output,
                              os.environ['LOOKING_GLASS_VISION_BOOTSTRAP_SITE'], shutil.which('nvidia-smi'))
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
