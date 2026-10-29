"""Supervised, source-bound BERT development-family GPU profile.

The fixed development workload requires a reviewed supervisor and device lease.
"""

from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
import weakref

from .bert_gpu_calibration import (ASSETS, MODEL_ID, REVISION, _guarded_call,
    _guarded_snapshot, _runtime_uuid, _file_identity, _import_torch,
    _require_pristine_dependencies, _verify_startup, check_memory, stage_assets)
from .bert_gpu_evidence import write_json_new
from .device_lease import DeviceLease
from .memory_budget import CudaMemorySnapshot, MemoryBudget
from .supervisor import OwnerSession

_MIB = 1024 ** 2
_GPU_CLAIM = 64 * 1024
_HOST_CLAIM = 64 * _MIB
_HEADROOM = 256 * _MIB
_STATE_CHUNK = _MIB
_GIB = 1024 ** 3
_PEAK_SCOPE = 'cumulative_since_pre_model_load_reset'
FAMILY_PROFILE_ENABLED = True  # Shared parent/child gate for this fixed development profile.
_CUSTODY = {'entered': False, 'session': None, 'device_lease': None, 'budget': None,
            'memory_lease': None, 'model': None, 'converted': None, 'witness': None,
            'observation': None, 'profile': None, 'cap': None, 'commitment': None,
            'failure': None, 'diagnostic_failure': None}
SOURCE_PATHS = (
    'scripts/bert_gpu_family_bootstrap.py', 'docs/BERT_GPU_FAMILY_PROFILE_V1.md',
    'docs/BERT_GPU_STUDY_ADAPTER_DRAFT.md', 'docs/BERT_AGREEMENT_V1.md',
    'experiments/bert_agreement_public_cases_v1.json', 'looking_glass/__init__.py',
    'looking_glass/bert_gpu_family_controller.py', 'looking_glass/bert_gpu_family_owner.py',
    'looking_glass/bert_gpu_family_profile.py', 'looking_glass/bert_gpu_study_adapter.py',
    'looking_glass/bert_gpu_calibration.py', 'looking_glass/bert_gpu_evidence.py',
    'looking_glass/bert_agreement_runner.py', 'looking_glass/bert_agreement_audit.py',
    'looking_glass/bert_agreement_study.py', 'looking_glass/bert_agreement_cases.py',
    'looking_glass/pretrained.py', 'looking_glass/gpu_oneshot.py',
    'looking_glass/device_identity.py', 'looking_glass/device_lease.py',
    'looking_glass/supervisor.py', 'looking_glass/memory_budget.py',
    'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
    'looking_glass/model_ownership.py')


def _verify_source(repo, source_commit, git_exe):
    """Bind all project source paths to one clean, committed checkout."""
    root = Path(repo).resolve(strict=True)
    git = Path(git_exe)
    if not git.is_absolute():
        raise ValueError('an absolute Git executable is required')
    git = git.resolve(strict=True)
    if not git.is_file():
        raise ValueError('an absolute Git executable is required')
    if type(source_commit) is not str or re.fullmatch('[0-9a-f]{40}', source_commit) is None:
        raise ValueError('a complete source commit is required')
    current = subprocess.run([str(git), 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
    dirty = subprocess.run([str(git), 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                           capture_output=True, text=True, check=True, timeout=10).stdout
    if current != source_commit or dirty:
        raise RuntimeError('family profile requires the exact clean committed source')
    hashes = {}
    for relative in SOURCE_PATHS:
        stored = subprocess.run([str(git), 'show', source_commit + ':' + relative], cwd=root,
                                capture_output=True, check=True, timeout=10).stdout
        actual = (root / relative).read_bytes()
        if actual != stored:
            raise RuntimeError('family profile source bytes differ from the committed tree')
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    return hashes


def _load_manifest(repo):
    from .bert_agreement_cases import MANIFEST_SHA256, verify_manifest_bytes
    path = Path(repo) / 'experiments' / 'bert_agreement_public_cases_v1.json'
    with path.open('rb') as stream:
        raw = stream.read(_MIB + 1)
    if len(raw) > _MIB or hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError('public family manifest differs from the pinned identity')
    checked = verify_manifest_bytes(raw)
    if checked.manifest_sha256 != MANIFEST_SHA256:
        raise RuntimeError('public family manifest verification differs')
    return json.loads(raw), MANIFEST_SHA256


def _load_family_model(cache_dir, torch, family_rows):
    """Stage the pinned checkpoint and verify the eight public family rows."""
    from importlib.metadata import version
    import transformers
    from .bert_agreement_study import verify_tokenizer_rows

    cache = Path(cache_dir).resolve(strict=True)
    snapshot = cache / 'models--distilbert--distilbert-base-uncased' / 'snapshots' / REVISION
    with tempfile.TemporaryDirectory(prefix='bert-family-stage-', dir=cache.parent) as directory:
        stage = Path(directory)
        stage_assets(snapshot, stage)
        with torch.device('cpu'):
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                stage, local_files_only=True, trust_remote_code=False, use_fast=True)
            model, info = transformers.AutoModelForMaskedLM.from_pretrained(
                stage, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                attn_implementation='eager', dtype=torch.float32, output_loading_info=True)
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('checkpoint load reported missing, unexpected or mismatched weights')
        model.eval()
        if (type(model) is not transformers.DistilBertForMaskedLM
                or type(tokenizer) is not transformers.DistilBertTokenizerFast
                or model.config.model_type != 'distilbert' or model.config.n_layers != 6
                or model.config.dim != 768 or model.config.vocab_size != 30522
                or model.config._attn_implementation != 'eager'
                or sum(parameter.numel() for parameter in model.parameters()) != 66985530
                or any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
                       for parameter in model.parameters())
                or any(buffer.device.type != 'cpu' for buffer in model.buffers())
                or model.get_input_embeddings().weight is not model.vocab_projector.weight
                or len(family_rows) != 8
                or verify_tokenizer_rows(tokenizer, {'rows': family_rows}) != 8):
            raise RuntimeError('pinned model, tokenizer, or public family identity differs')
        return model, {'model': MODEL_ID, 'revision': REVISION,
            'assets': {name: {'size': size, 'sha256': digest} for name, (size, digest) in ASSETS.items()},
            'model_class': type(model).__name__, 'tokenizer_class': type(tokenizer).__name__,
            'transformers_version': str(transformers.__version__), 'tokenizers_version': version('tokenizers'),
            'safetensors_version': version('safetensors'), 'parameter_count': 66985530,
            'attention_implementation': 'eager', 'cpu_tied_weight_verified': True}


def _reserve_family_budget(telemetry, *, cap, custody):
    """Root and reserve the whole allocator plus separate logical trace claims."""
    if type(cap) is not int or cap <= _HEADROOM or type(custody) is not dict:
        raise ValueError('invalid family memory commitment')
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=_GPU_CLAIM,
                          max_host_trace_bytes=_HOST_CLAIM)
    custody['budget'] = budget
    lease = budget.acquire(cap + _HEADROOM, max_gpu_trace_bytes=_GPU_CLAIM,
                           max_host_trace_bytes=_HOST_CLAIM)
    custody['memory_lease'] = lease
    lease.reserve_trace('gpu', _GPU_CLAIM)
    lease.reserve_trace('host', _HOST_CLAIM)
    return budget, lease


@dataclass(frozen=True)
class _StateWitness:
    modules: tuple
    tensors: tuple
    sha256: str


def _capture_state(model, torch, *, device, check_owner, synchronize):
    """Hash registered tensor bits in bounded chunks while holding their identities."""
    if (not isinstance(model, torch.nn.Module) or type(device) is not str
            or device not in ('cpu', 'cuda:0') or not callable(check_owner)
            or not callable(synchronize)):
        raise ValueError('invalid model-state witness request')

    def guarded(function):
        check_owner()
        value = function()
        check_owner()
        return value

    check_owner()
    modules = tuple((name, module, bool(module.training))
                    for name, module in model.named_modules(remove_duplicate=False))
    digest = hashlib.sha256()
    tensors = []
    for kind, entries in (('parameter', model.named_parameters(remove_duplicate=False)),
                          ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, tensor in entries:
            if str(tensor.device) != device or not tensor.is_contiguous():
                raise ValueError('model state must be contiguous on the declared device')
            storage = tensor.untyped_storage()
            metadata = (kind, name, tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
                        str(tensor.device), tensor.storage_offset(), storage.nbytes(),
                        storage.data_ptr(), bool(tensor.requires_grad))
            header = json.dumps(metadata[:-2] + metadata[-1:], separators=(',', ':')).encode('utf-8')
            digest.update(len(header).to_bytes(4, 'big'))
            digest.update(header)
            detached = guarded(lambda: tensor.detach())
            flattened = guarded(lambda: detached.reshape(-1))
            raw = guarded(lambda: flattened.view(torch.uint8))
            for offset in range(0, raw.numel(), _STATE_CHUNK):
                chunk = guarded(lambda: raw[offset:offset + _STATE_CHUNK])
                host = guarded(lambda: chunk.to('cpu', copy=True))
                digest.update(memoryview(host.numpy()))
                del host, chunk
            tensors.append((metadata, tensor, storage))
    guarded(synchronize)
    return _StateWitness(modules, tuple(tensors), digest.hexdigest())


def _verify_state(model, witness, torch, *, device, check_owner, synchronize):
    if type(witness) is not _StateWitness:
        raise ValueError('a model-state witness is required')
    current = _capture_state(model, torch, device=device, check_owner=check_owner,
                             synchronize=synchronize)
    if (current.sha256 != witness.sha256 or len(current.modules) != len(witness.modules)
            or len(current.tensors) != len(witness.tensors)):
        raise RuntimeError('model state or registration changed')
    for (name, module, mode), (old_name, old_module, old_mode) in zip(current.modules, witness.modules):
        if name != old_name or module is not old_module or mode != old_mode:
            raise RuntimeError('model state or module identity changed')
    for (metadata, tensor, storage), (old_metadata, old_tensor, old_storage) in zip(current.tensors, witness.tensors):
        if metadata != old_metadata or tensor is not old_tensor or storage is not old_storage:
            raise RuntimeError('model state or tensor identity changed')
    return True


def _runtime_versions(torch):
    from importlib.metadata import version
    observed = {'python': '.'.join(map(str, sys.version_info[:3])), 'torch': str(torch.__version__),
        'cuda_build': str(torch.version.cuda), 'transformers': version('transformers'),
        'tokenizers': version('tokenizers'), 'safetensors': version('safetensors'),
        'numpy': version('numpy')}
    expected = {'python': '3.11.15', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
        'transformers': '4.57.3', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.4.6'}
    if observed != expected or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('runtime versions or cuBLAS configuration differ from the frozen profile')
    return observed


def _check_family_custody(session, device_lease, *, gpu_claim=_GPU_CLAIM, host_claim=_HOST_CLAIM):
    """Check the actual process-local claims without querying CUDA telemetry.

    These checks are sequential, not an atomic lock against unrelated code.
    The dedicated owner remains trusted and single-use.
    """
    OwnerSession.check(session, device_lease)
    if _CUSTODY['session'] is not session or _CUSTODY['device_lease'] is not device_lease:
        raise RuntimeError('supervised device custody differs')
    budget, lease = _CUSTODY['budget'], _CUSTODY['memory_lease']
    cap, commitment = _CUSTODY['cap'], _CUSTODY['commitment']
    if (type(budget) is not MemoryBudget or budget.active_leases() != (lease,)
            or type(cap) is not int or cap <= 0 or commitment != cap + _HEADROOM
            or lease.estimated_incremental_bytes != commitment
            or lease.trace_bytes('gpu') != gpu_claim or lease.trace_bytes('host') != host_claim):
        raise RuntimeError('family budget, commitment or trace custody was lost')
    OwnerSession.check(session, device_lease)


def measure_family_owned(session, lease, torch, cache_dir, manifest, *, journal_path, record_memory):
    """Run the fixed family only inside an already supervised device lease.

    The internal CUDA observer is reserved for this source-bound development
    profile and must run only under its reviewed supervisor and device lease.
    """
    if FAMILY_PROFILE_ENABLED is not True:
        raise RuntimeError('BERT family profile is disabled')
    if type(session) is not OwnerSession or type(lease) is not DeviceLease or not callable(record_memory):
        raise ValueError('an actual supervised owner, device lease and recorder are required')
    OwnerSession.check(session, lease)
    runtime_versions = _runtime_versions(torch)
    if torch.cuda.is_initialized() is not False:
        raise RuntimeError('CUDA was initialized before the owner check')
    count = _guarded_call(session, lease, torch.cuda.device_count)
    if type(count) is not int or count != 1:
        raise RuntimeError('exactly one CUDA-visible device is required')
    properties = _guarded_call(session, lease, torch.cuda.get_device_properties, 0)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=count)
    if not 7 * _GIB <= properties.total_memory <= 9 * _GIB:
        raise RuntimeError('device total is outside the expected 8 GiB range')
    if _guarded_call(session, lease, torch.cuda.get_allocator_backend) != 'native':
        raise RuntimeError('the active CUDA allocator backend is not native')
    _guarded_call(session, lease, torch.cuda.set_per_process_memory_fraction, .25, 0)
    cap = int(.25 * properties.total_memory)
    if _guarded_call(session, lease, torch.cuda.get_per_process_memory_fraction, 0) != cap / properties.total_memory:
        raise RuntimeError('allocator fraction readback differs from the cap')
    _guarded_call(session, lease, torch.cuda.reset_peak_memory_stats, 0)
    initial = _guarded_snapshot(session, lease, torch)
    record_memory('before_model_load', None, initial)
    if initial['device_total_bytes'] != properties.total_memory:
        raise RuntimeError('CUDA property and telemetry totals disagree')
    if initial['device_free_bytes'] < 3 * _GIB:
        raise RuntimeError('insufficient device-wide free memory before model loading')
    total = initial['device_total_bytes']
    commitment = cap + _HEADROOM
    _CUSTODY['cap'], _CUSTODY['commitment'] = cap, commitment
    check_memory(initial, expected_total=total, cap=cap, commitment=commitment)

    def telemetry():
        free, observed_total = _guarded_call(session, lease, torch.cuda.mem_get_info, 0)
        if observed_total != total:
            raise RuntimeError('CUDA telemetry total changed during admission')
        return CudaMemorySnapshot(free, observed_total,
            _guarded_call(session, lease, torch.cuda.memory_allocated, 0),
            _guarded_call(session, lease, torch.cuda.memory_reserved, 0))

    budget, memory_lease = _reserve_family_budget(telemetry, cap=cap, custody=_CUSTODY)

    def owner_check():
        _check_family_custody(session, lease)

    def guarded(function, *args):
        owner_check()
        result = function(*args)
        owner_check()
        return result

    owner_check()
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise RuntimeError('unsupported module conversion flags')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(17)
    guarded(torch.cuda.manual_seed, 17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = {'device': 'cuda:0', 'threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'overwrite_module_params_on_conversion': torch.__future__.get_overwrite_module_params_on_conversion(),
        'swap_module_params_on_conversion': torch.__future__.get_swap_module_params_on_conversion()}
    expected = {'device': 'cuda:0', 'threads': 1, 'interop_threads': 1,
        'deterministic_algorithms': True, 'matmul_tf32': False, 'cudnn_tf32': False,
        'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'overwrite_module_params_on_conversion': False, 'swap_module_params_on_conversion': False}
    if json.dumps(config, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise RuntimeError('runtime execution configuration differs from the profile')

    def synchronize():
        guarded(torch.cuda.synchronize, 0)

    def snapshot(stage, index=None):
        synchronize()
        free, observed_total = guarded(torch.cuda.mem_get_info, 0)
        value = {'device_free_bytes': free, 'device_total_bytes': observed_total,
            'allocated_bytes': guarded(torch.cuda.memory_allocated, 0),
            'reserved_bytes': guarded(torch.cuda.memory_reserved, 0),
            'peak_allocated_bytes': guarded(torch.cuda.max_memory_allocated, 0),
            'peak_reserved_bytes': guarded(torch.cuda.max_memory_reserved, 0)}
        record_memory(stage, index, value)
        check_memory(value, expected_total=total, cap=cap, commitment=commitment)
        return value

    if type(manifest) is not dict or type(manifest.get('rows')) is not list:
        raise ValueError('a verified public case manifest is required')
    rows = [row for row in manifest['rows'] if row.get('split') == 'development' and row.get('family') == 1]
    model, metadata = _load_family_model(cache_dir, torch, rows)
    _CUSTODY['model'] = model
    snapshot('before_model_copy')
    owner_check()
    converted = model.to('cuda:0')
    _CUSTODY['converted'] = converted
    owner_check()
    if (converted is not model or model.training
            or sum(parameter.numel() for parameter in model.parameters()) != 66985530
            or any(str(parameter.device) != 'cuda:0' or parameter.dtype != torch.float32
                   for parameter in model.parameters())
            or any(str(buffer.device) != 'cuda:0' for buffer in model.buffers())
            or model.get_input_embeddings().weight is not model.vocab_projector.weight):
        raise RuntimeError('GPU model, tied parameters, or device differ')
    snapshot('after_model_copy')
    witness = _capture_state(model, torch, device='cuda:0', check_owner=owner_check,
                             synchronize=synchronize)
    _CUSTODY['witness'] = witness
    snapshot('after_initial_state')

    from .bert_agreement_runner import _has_hooks
    from .bert_gpu_family_profile import execute_family_profile
    from .bert_gpu_study_adapter import _ObservationCustody, _run_family_calibration_observation

    def observer(model_arg, inputs, **options):
        owner_check()
        custody = _ObservationCustody()
        _CUSTODY['observation'] = custody
        result = _run_family_calibration_observation(model_arg, inputs, check_owner=owner_check,
                                      synchronize=lambda stage: synchronize(), custody=custody, **options)
        owner_check()
        _CUSTODY['observation'] = None
        return result

    profile = execute_family_profile(model, manifest, observer=observer, checkpoint=snapshot,
        journal_path=journal_path, is_id=2003, are_id=2024, mask_id=103, hidden_size=768,
        expected_device='cuda:0')
    _CUSTODY['profile'] = profile
    if _has_hooks(model):
        raise RuntimeError('model hooks remain after the family')
    _verify_state(model, witness, torch, device='cuda:0', check_owner=owner_check, synchronize=synchronize)
    snapshot('after_final_state')
    state_sha = witness.sha256
    reference = weakref.ref(model)
    _CUSTODY['witness'] = _CUSTODY['converted'] = _CUSTODY['model'] = None
    del witness, converted, model
    gc.collect()
    if reference() is not None:
        raise RuntimeError('model reference remains after disposal')
    synchronize()
    guarded(torch.cuda.empty_cache)
    snapshot('cleanup_before_workspace_clear')
    if (str(torch.__version__) != '2.8.0+cu128'
            or not callable(getattr(torch._C, '_cuda_clearCublasWorkspaces', None))):
        raise RuntimeError('unsupported PyTorch cuBLAS workspace cleanup API')
    guarded(torch._C._cuda_clearCublasWorkspaces)
    synchronize()
    guarded(torch.cuda.empty_cache)
    cleanup = snapshot('after_cleanup')
    if cleanup['allocated_bytes'] != 0 or cleanup['reserved_bytes'] != 0:
        raise RuntimeError('CUDA allocator bytes remain after disposal')
    owner_check()
    return {'profile': profile, 'model_metadata': metadata, 'model_state_sha256': state_sha,
        'execution_config': config, 'device_name': properties.name, 'device_total_bytes': total,
        'allocator_backend': 'native', 'allocator_fraction': .25, 'allocator_cap_bytes': cap,
        'whole_run_commitment_bytes': commitment, 'gpu_trace_claim_bytes': _GPU_CLAIM,
        'host_trace_claim_bytes': _HOST_CLAIM, 'peak_scope': _PEAK_SCOPE,
        'runtime_uuid_verified': True, 'cleanup': cleanup, 'model_weakref_alive': False,
        'budget_active_before_candidate_write': 1, 'torch_version': str(torch.__version__),
        'cuda_build': str(torch.version.cuda), 'runtime_versions': runtime_versions,
        'seed': 17, 'cublas_workspace_config': ':4096:8'}


def _release_family_budget(session, device_lease):
    """Release only after measurement serialization and host evidence disposal."""
    _check_family_custody(session, device_lease)
    budget, lease = _CUSTODY['budget'], _CUSTODY['memory_lease']
    lease.release_trace('gpu', _GPU_CLAIM)
    _check_family_custody(session, device_lease, gpu_claim=0)
    lease.release_trace('host', _HOST_CLAIM)
    _check_family_custody(session, device_lease, gpu_claim=0, host_claim=0)
    lease.release()
    OwnerSession.check(session, device_lease)
    if budget.active_leases():
        raise RuntimeError('family memory commitments remain')


def owner_main(run_dir, cache_dir, run_id, source_commit, git_exe):
    """Single-use child; uncertain failures remain rooted and quarantined."""
    try:
        if FAMILY_PROFILE_ENABLED is not True:
            raise RuntimeError('BERT family profile is disabled')
        if _CUSTODY['entered'] is not False:
            raise RuntimeError('family profile owner is single-use')
        _CUSTODY['entered'] = True
        _verify_startup()
        _require_pristine_dependencies()
        session = OwnerSession.from_environment()
        _CUSTODY['session'] = session
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _CUSTODY['device_lease'] = lease
        OwnerSession.check(session, lease)
        if session.identity is None:
            raise RuntimeError('a physical GPU identity is required')
        if type(run_id) is not str or re.fullmatch('[0-9a-f]{32}', run_id) is None:
            raise ValueError('invalid family profile run ID')
        repo = Path(__file__).resolve().parents[1]
        hashes = _verify_source(repo, source_commit, git_exe)
        manifest, manifest_sha = _load_manifest(repo)
        directory = Path(run_dir).resolve(strict=True)
        write_json_new(directory / 'owner.json', {'schema': 'bert-gpu-family-owner-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': source_commit, 'manifest_sha256': manifest_sha})
        torch = _import_torch()
        from .bert_agreement_study import EventJournal

        memory_count = 0
        with EventJournal(directory / 'memory.jsonl', max_bytes=4 * _MIB) as memory:
            def record_memory(stage, index, snapshot):
                nonlocal memory_count
                memory_count += 1
                if memory_count > 287:
                    raise RuntimeError('memory endpoint inventory exceeded')
                record = {'stage': stage, 'index': index, 'memory': snapshot, 'peak_scope': _PEAK_SCOPE}
                encoded = json.dumps(record, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')
                if len(encoded) > 12 * 1024:
                    raise ValueError('memory record exceeds its byte limit')
                memory.emit(record)
            measurement = measure_family_owned(session, lease, torch, cache_dir, manifest,
                journal_path=directory / 'events.jsonl', record_memory=record_memory)
        if memory_count != 287:
            raise RuntimeError('memory endpoint inventory is incomplete')
        measurement['memory_journal_sha256'] = memory.sha256
        measurement['memory_record_count'] = memory_count
        measurement['manifest_sha256'] = manifest_sha
        measurement['source_sha256'] = hashes
        candidate = {'schema': 'bert-gpu-family-candidate-v1', 'run_id': run_id,
            'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': source_commit, 'measurement': measurement}
        _check_family_custody(session, lease)
        write_json_new(directory / 'candidate.json', candidate)
        _check_family_custody(session, lease)
        _, candidate_sha = _file_identity(directory / 'candidate.json', _MIB)
        _check_family_custody(session, lease)
        del candidate, measurement
        _CUSTODY['profile'] = None
        gc.collect()
        _release_family_budget(session, lease)
        OwnerSession.check(session, lease)
        write_json_new(directory / 'final.json', {'schema': 'bert-gpu-family-final-v1',
            'run_id': run_id, 'supervisor_run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': source_commit, 'candidate_sha256': candidate_sha,
            'active_memory_leases': 0})
        OwnerSession.check(session, lease)
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as error:
        if _CUSTODY['failure'] is None:
            _CUSTODY['failure'] = error
        try:
            write_json_new(Path(run_dir) / 'failure.json', {'schema': 'bert-gpu-family-failure-v1',
                'run_id': run_id, 'status': 'quarantined', 'error_type': type(error).__name__,
                'failure_kind': 'error' if isinstance(error, Exception) else 'interrupted',
                'private_traceback': traceback.format_exc()[-8192:]})
        except BaseException as diagnostic_error:
            _CUSTODY['diagnostic_failure'] = diagnostic_error
            if not isinstance(diagnostic_error, Exception):
                raise
        if isinstance(error, Exception):
            return 1
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', nargs=5, metavar=('RUN_DIR', 'CACHE_DIR', 'RUN_ID', 'COMMIT', 'GIT_EXE'),
                        required=True)
    args = parser.parse_args(argv)
    return owner_main(*args.owner)


if __name__ == '__main__':
    sys.exit(main())
