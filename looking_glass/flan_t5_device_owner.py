"""Owner-side custody primitives for bounded FLAN-T5 device evidence.

This module does not launch an experiment or attest GPU cleanup. The caller
must be the trusted, supervised owner holding the physical device lease.
"""

import gc
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from threading import current_thread
import weakref

from .device_lease import DeviceLease
from .memory_budget import CudaMemorySnapshot, MemoryBudget, MemoryLease
from .supervisor import OwnerSession


_HEADROOM = 256 * 1024 ** 2
_MAX_RETAINED_OBSERVATIONS = 3
_GIB = 1024 ** 3
SOURCE_PATHS = (
    'scripts/flan_t5_gpu_bootstrap.py', 'scripts/v2_source_guards.py',
    'docs/FLAN_T5_GPU_DEVELOPMENT_SMOKE_V1.md',
    'docs/FLAN_T5_PUBLIC_CASES_V1.json', 'docs/FLAN_T5_DEVICE_OBSERVATION_V1.md',
    'docs/FLAN_T5_DEVICE_OWNER_CUSTODY_DRAFT.md', 'looking_glass/__init__.py',
    'looking_glass/flan_t5_device_owner.py', 'looking_glass/flan_t5_device_smoke.py',
    'looking_glass/flan_t5_device_controller.py',
    'looking_glass/flan_t5_device_observation.py',
    'looking_glass/flan_t5_feasibility.py', 'looking_glass/flan_t5_equal_dose.py',
    'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
    'looking_glass/model_ownership.py', 'looking_glass/device_identity.py',
    'looking_glass/device_lease.py', 'looking_glass/memory_budget.py',
    'looking_glass/supervisor.py', 'looking_glass/gpu_oneshot.py',
    'looking_glass/bert_gpu_calibration.py', 'looking_glass/bert_gpu_evidence.py',
    'looking_glass/bert_gpu_family_owner.py',
)


def _require_pristine_dependencies():
    if any(name in sys.modules for name in ('torch', 'transformers', 'numpy', 'tokenizers', 'safetensors')):
        raise RuntimeError('model dependencies were imported before device ownership')


def _check_bootstrap(repo):
    site = Path(os.environ.get('LOOKING_GLASS_FLAN_T5_BOOTSTRAP_SITE', ''))
    prefix = sys.pycache_prefix
    root = Path(repo).resolve(strict=True)
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or os.environ.get('LOOKING_GLASS_FLAN_T5_BOOTSTRAP_PID') != str(os.getpid())
            or not site.is_absolute() or not site.is_dir() or not isinstance(prefix, str)
            or not Path(prefix).is_absolute() or sys._xoptions.get('pycache_prefix') != prefix):
        raise RuntimeError('the isolated FLAN-T5 source bootstrap is required')
    site = site.resolve(strict=True)
    if site == root or site.is_relative_to(root) or root.is_relative_to(site):
        raise RuntimeError('dependency and checkout directories must be separate')


def _check_workspace(repo, session):
    root = Path(repo).resolve(strict=True)
    if session.identity is None:
        raise RuntimeError('a supervised physical device identity is required')
    work_parents = [parent for parent in root.parents if parent.name == 'work']
    expected = work_parents[0].parent if work_parents else root
    actual = Path(session.identity.workspace_root).resolve(strict=True)
    if actual != expected or not (actual / 'work').is_dir():
        raise RuntimeError('device custody does not match the shared workspace')


def _import_torch():
    import torch
    return torch


def _dependency_origin(name, site):
    """Record bounded bytes of one loaded package entry under the selected site."""
    module = sys.modules.get(name)
    origin = getattr(getattr(module, '__spec__', None), 'origin', None)
    filename = getattr(module, '__file__', None)
    root = Path(site).resolve(strict=True)
    if type(origin) is not str or type(filename) is not str:
        raise ValueError('dependency has no source entry under the selected site')
    lexical = Path(os.path.abspath(filename))
    path = lexical.resolve(strict=True)
    if (not lexical.is_relative_to(root) or path != Path(origin).resolve(strict=True)
            or not path.is_relative_to(root) or path.suffix != '.py'):
        raise ValueError('dependency entry is outside the selected site')
    for entry in (lexical, *lexical.parents):
        if entry == root:
            break
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('dependency site contains a link or reparse point')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024 ** 2:
        raise ValueError('dependency entry is not a bounded regular file')
    with path.open('rb') as stream:
        raw = stream.read(4 * 1024 ** 2 + 1)
        opened = os.fstat(stream.fileno())
    after = path.stat()
    if (len(raw) != before.st_size or len(raw) > 4 * 1024 ** 2
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('dependency entry changed while reading')
    return {'path': path.relative_to(root).as_posix(), 'size': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def _dependency_origins(site):
    names = ('torch', 'transformers', 'numpy', 'tokenizers', 'safetensors', 'huggingface_hub')
    for name in names:
        importlib.import_module(name)
    return {name: _dependency_origin(name, site) for name in names}


def _configure_numerics(torch):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('cuBLAS workspace policy must be set before CUDA initialization')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(20261028)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.set_flush_denormal(False) is not True:
        raise RuntimeError('CPU denormal policy is unavailable')


def _verify_source_tree(repo, git_executable, commit, paths):
    """Bind a fixed source-path inventory to exact checkout and Git blob bytes."""
    repo, git = Path(repo), Path(git_executable)
    if (not repo.is_absolute() or not git.is_absolute() or type(commit) is not str
            or re.fullmatch('[0-9a-f]{40}', commit) is None or type(paths) is not tuple
            or not 1 <= len(paths) <= 128 or len(set(paths)) != len(paths)):
        raise ValueError('an exact source commit and bounded path inventory are required')
    root, git = repo.resolve(strict=True), git.resolve(strict=True)
    if not root.is_dir() or not git.is_file():
        raise ValueError('source checkout or Git executable is unavailable')
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    environment.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_GRAFT_FILE': os.devnull,
                        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                        'GIT_CONFIG_SYSTEM': os.devnull, 'GIT_TERMINAL_PROMPT': '0',
                        'GIT_OPTIONAL_LOCKS': '0'})
    def call(*args):
        return subprocess.run([str(git), '--no-replace-objects', '-c', 'core.fsmonitor=false',
                               '-C', str(root), *args], check=True, capture_output=True,
                              timeout=20, env=environment).stdout
    if (call('rev-parse', 'HEAD').strip() != commit.encode('ascii')
            or call('status', '--porcelain', '--untracked-files=no')):
        raise RuntimeError('source checkout is not the clean selected commit')
    bound, total = {}, 0
    for name in paths:
        if (type(name) is not str or re.fullmatch('[A-Za-z0-9_./-]+', name) is None
                or PurePosixPath(name).is_absolute()
                or any(part in ('.', '..') for part in name.split('/'))):
            raise ValueError('invalid source path')
        path = root.joinpath(*name.split('/'))
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
                or not path.resolve(strict=True).is_relative_to(root)):
            raise ValueError('source must be a regular checkout file')
        size_bytes = call('cat-file', '-s', commit + ':' + name).strip()
        if not size_bytes.isdigit() or int(size_bytes) > 4 * 1024 ** 2:
            raise ValueError('source file exceeds its bound')
        size = int(size_bytes)
        total += size
        if total > 32 * 1024 ** 2:
            raise ValueError('source inventory exceeds its bound')
        with path.open('rb') as stream:
            actual = stream.read(size + 1)
            opened = os.fstat(stream.fileno())
        after = path.lstat()
        if (len(actual) != size or info.st_size != size
                or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or actual != call('show', commit + ':' + name)):
            raise RuntimeError('source bytes differ from the selected committed blob')
        bound[name] = {'size': size, 'sha256': hashlib.sha256(actual).hexdigest()}
    return bound


def _stage_assets(source, destination):
    """Copy the fixed, verified checkpoint into a fresh attempt directory."""
    from .flan_t5_feasibility import MANIFEST_HASH, verify_assets
    source, destination = Path(source), Path(destination)
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError('asset paths must be absolute')
    source = source.resolve(strict=True)
    if not source.is_dir() or not destination.parent.is_dir():
        raise ValueError('asset directories are unavailable')
    manifest = verify_assets(source)
    if type(manifest) is not dict or type(manifest.get('files')) is not dict:
        raise ValueError('asset manifest is invalid')
    names = tuple(sorted(manifest['files']))
    if (not names or any(type(name) is not str or Path(name).name != name
                         or name in ('.', '..', 'asset-manifest.json') for name in names)):
        raise ValueError('asset names must be plain filenames')
    with (source / 'asset-manifest.json').open('rb') as stream:
        manifest_raw = stream.read(1024 ** 2 + 1)
    if len(manifest_raw) > 1024 ** 2 or hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_HASH:
        raise ValueError('asset manifest changed before staging')
    destination.mkdir()
    with (destination / 'asset-manifest.json').open('xb') as stream:
        if stream.write(manifest_raw) != len(manifest_raw):
            raise OSError('short asset manifest write')
        stream.flush()
        os.fsync(stream.fileno())
    for name in names:
        item = manifest['files'][name]
        if (type(item) is not dict or type(item.get('bytes')) is not int
                or not 0 < item['bytes'] <= 1024 ** 3
                or type(item.get('sha256')) is not str
                or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None):
            raise ValueError('asset size or digest is invalid')
        path = source / name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('asset must be a regular, non-reparse file')
        digest, size = hashlib.sha256(), 0
        with path.open('rb') as incoming, (destination / name).open('xb') as outgoing:
            while True:
                chunk = incoming.read(1024 ** 2)
                if not chunk:
                    break
                size += len(chunk)
                if size > item['bytes']:
                    raise ValueError('asset grew while staging')
                digest.update(chunk)
                if outgoing.write(chunk) != len(chunk):
                    raise OSError('short asset write')
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError('asset changed while staging')
    if verify_assets(destination) != manifest:
        raise ValueError('staged asset manifest differs')
    return manifest


def _load_staged_model(directory, torch):
    """Load only the pinned FLAN-T5 architecture, on CPU, from staged bytes."""
    from transformers import T5ForConditionalGeneration
    from .flan_t5_device_observation import _model_contract, _validate_flan_checkpoint_model
    with torch.device('cpu'):
        model, info = T5ForConditionalGeneration.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, attn_implementation='eager', dtype=torch.float32,
            output_loading_info=True)
    diagnostics = {'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'}
    if (type(info) is not dict or set(info) != diagnostics
            or any(type(info[key]) is not list or info[key] for key in diagnostics)):
        raise ValueError('pinned model loading diagnostics are not empty')
    model.eval()
    model.config.use_cache = False
    config = model.config
    if (type(model) is not T5ForConditionalGeneration or config.model_type != 't5'
            or config.d_model != 512 or config.vocab_size != 32128
            or config.num_layers != 8 or config.num_decoder_layers != 8
            or config.d_ff != 1024 or config.d_kv != 64 or config.num_heads != 6
            or config.feed_forward_proj != 'gated-gelu' or config.tie_word_embeddings is not False
            or sum(parameter.numel() for parameter in model.parameters()) != 76961152
            or model.encoder.embed_tokens.weight is not model.shared.weight
            or model.decoder.embed_tokens.weight is not model.shared.weight
            or model.lm_head.weight is model.shared.weight):
        raise ValueError('loaded model differs from the pinned FLAN-T5 architecture')
    _model_contract(model, torch.device('cpu'))
    _validate_flan_checkpoint_model(model)
    return model


class _ObservationSlot:
    def __init__(self, custody):
        self.custody = custody
        self.lease = None
        self.result = None
        self.retained = None


class DeviceMemoryCustody:
    """Single-threaded model commitment and bounded per-observation claims.

    Telemetry and the ownership check must come from the actual supervised
    device owner. The caller retains this object before invoking ``start``.
    """

    def __init__(self, telemetry, *, allocator_cap, check_owner):
        if (not callable(telemetry) or not callable(check_owner)
                or type(allocator_cap) is not int or allocator_cap <= _HEADROOM):
            raise ValueError('an allocator cap, telemetry and owner check are required')
        self._thread = current_thread()
        self._telemetry = telemetry
        self._check_owner = check_owner
        self.allocator_cap = allocator_cap
        self.budget = None
        self.model_lease = None
        self.slots = []
        self.failed = False
        self.failure = None
        self._started = False

    def _check(self):
        if current_thread() is not self._thread:
            raise RuntimeError('device memory custody belongs to its owner thread')
        if self.failed:
            raise RuntimeError('device memory custody is uncertain')
        if self._check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')

    def start(self):
        from .flan_t5_device_observation import GPU_TRACE_CLAIM, HOST_TRACE_CLAIM
        self._check()
        if self._started or self.budget is not None:
            raise RuntimeError('device memory custody is single-use')
        # Leave a sticky failure before each mutation; the retained budget roots
        # a lease even if its return is interrupted before assignment.
        self.failed = True
        try:
            self.budget = MemoryBudget(self._telemetry,
                max_gpu_trace_bytes=_MAX_RETAINED_OBSERVATIONS * GPU_TRACE_CLAIM,
                max_host_trace_bytes=_MAX_RETAINED_OBSERVATIONS * HOST_TRACE_CLAIM)
            self.model_lease = self.budget.acquire(self.allocator_cap + _HEADROOM,
                max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            if self._check_owner() is not None:
                raise RuntimeError('owner check must return None or raise on failure')
            self._started = True
            self.failed = False
        except BaseException as error:
            self.failed = True
            self.failure = error
            raise

    def reserve_observation(self):
        from .flan_t5_device_observation import (GPU_TRACE_CLAIM, HOST_TRACE_CLAIM,
                                                 ObservationCustody)
        self._check()
        if (not self._started or type(self.budget) is not MemoryBudget
                or type(self.model_lease) is not MemoryLease
                or not any(item is self.model_lease for item in self.budget.active_leases())):
            raise RuntimeError('the model and forward commitment is missing')
        active = [slot for slot in self.slots if slot.lease is None or not slot.lease.released]
        if len(active) >= _MAX_RETAINED_OBSERVATIONS:
            raise RuntimeError('at most three observations may be retained')
        slot = _ObservationSlot(ObservationCustody())
        self.failed = True
        try:
            self.slots[:] = active
            self.slots.append(slot)
            slot.lease = self.budget.acquire(GPU_TRACE_CLAIM,
                max_gpu_trace_bytes=GPU_TRACE_CLAIM, max_host_trace_bytes=HOST_TRACE_CLAIM)
            slot.lease.reserve_trace('gpu', GPU_TRACE_CLAIM)
            slot.lease.reserve_trace('host', HOST_TRACE_CLAIM)
            if self._check_owner() is not None:
                raise RuntimeError('owner check must return None or raise on failure')
            self.failed = False
            return slot
        except BaseException as error:
            self.failed = True
            self.failure = error
            raise


class FlanT5CudaOwner:
    """Single-use, supervised FLAN-T5 device owner; retain before starting.

    This object does not launch a process, choose study inputs or attest exit.
    Its caller owns the supervisor, durable records and recovery on failure.
    """

    def __init__(self, session, device_lease, *, repo, git_executable,
                 source_commit, record_stage):
        self._thread = current_thread()
        self.session, self.device_lease = session, device_lease
        self.repo, self.git_executable = Path(repo), Path(git_executable)
        self.source_commit, self.record_stage = source_commit, record_stage
        self.torch = None
        self.source_hashes = None
        self.dependency_origins = None
        self.memory = None
        self.model = self.converted = self.witness = None
        self.stage_directory = None
        self.asset_manifest = None
        self.allocator_cap = self.device_total = None
        self.device_name = self.runtime_versions = None
        self.failed = False
        self.failure = None
        self.phase = 'new'

    def _session_check(self):
        if current_thread() is not self._thread:
            raise RuntimeError('FLAN-T5 device owner belongs to its owner thread')
        if type(self.session) is not OwnerSession or type(self.device_lease) is not DeviceLease:
            raise ValueError('an exact supervised owner and device lease are required')
        OwnerSession.check(self.session, self.device_lease)

    def _guard(self):
        self._session_check()
        from .flan_t5_device_observation import _cuda_numeric_state
        _cuda_numeric_state()
        self._session_check()

    def _call(self, function, *args):
        self._guard()
        value = function(*args)
        self._guard()
        return value

    def _model_guard(self):
        self._guard()
        from .flan_t5_device_observation import GPU_TRACE_CLAIM, HOST_TRACE_CLAIM
        memory = self.memory
        if (type(memory) is not DeviceMemoryCustody or memory.failed
                or type(memory.budget) is not MemoryBudget or type(memory.model_lease) is not MemoryLease
                or memory.model_lease.estimated_incremental_bytes != self.allocator_cap + _HEADROOM
                or not any(item is memory.model_lease for item in memory.budget.active_leases())):
            raise RuntimeError('model and forward memory commitment was lost')
        active = memory.budget.active_leases()
        expected = {id(memory.model_lease)}
        for slot in memory.slots:
            if type(slot.lease) is not MemoryLease or slot.lease._budget is not memory.budget:
                raise RuntimeError('observation memory custody differs')
            is_active = any(item is slot.lease for item in active)
            releasing = slot.retained is not None and slot.retained._releasing
            if is_active:
                expected.add(id(slot.lease))
                gpu, host = slot.lease.trace_bytes('gpu'), slot.lease.trace_bytes('host')
                if (slot.lease.estimated_incremental_bytes != GPU_TRACE_CLAIM
                        or gpu not in ((0, GPU_TRACE_CLAIM) if releasing else (GPU_TRACE_CLAIM,))
                        or host not in ((0, HOST_TRACE_CLAIM) if releasing else (HOST_TRACE_CLAIM,))):
                    raise RuntimeError('observation trace claims differ')
            elif not (slot.retained is not None and (slot.retained._retired or releasing)):
                raise RuntimeError('observation claim was released outside retirement')
        if {id(item) for item in active} != expected:
            raise RuntimeError('unexpected memory lease in the device budget')
        self._guard()

    def _retained_guard(self):
        if self.failed or self.phase != 'model-ready':
            raise RuntimeError('the supervised FLAN-T5 owner is uncertain')
        self._model_guard()

    def _policy_commitment(self):
        if self.memory is None:
            return self.allocator_cap + _HEADROOM
        self._model_guard()
        return sum(lease.estimated_incremental_bytes for lease in self.memory.budget.active_leases())

    def _stage(self, stage):
        from .flan_t5_device_observation import _record_cuda_stage
        from .bert_gpu_calibration import check_memory
        value = _record_cuda_stage(self.torch.cuda, self._guard, stage, self.record_stage)
        check_memory(value, expected_total=self.device_total, cap=self.allocator_cap,
                     commitment=self._policy_commitment())
        return value

    def _telemetry(self):
        cuda = self.torch.cuda
        self._call(cuda.synchronize, 0)
        free, total = self._call(cuda.mem_get_info, 0)
        if total != self.device_total:
            raise RuntimeError('CUDA device total changed during admission')
        return CudaMemorySnapshot(free, total, self._call(cuda.memory_allocated, 0),
                                  self._call(cuda.memory_reserved, 0))

    def start(self):
        self._session_check()
        if self.phase != 'new' or self.failed or not callable(self.record_stage):
            raise RuntimeError('the supervised FLAN-T5 owner is single-use')
        # The retained owner remains failed if interruption crosses any step.
        self.failed = True
        try:
            _check_bootstrap(self.repo)
            _require_pristine_dependencies()
            _check_workspace(self.repo, self.session)
            self.source_hashes = _verify_source_tree(self.repo, self.git_executable,
                                                      self.source_commit, SOURCE_PATHS)
            self._session_check()
            self.torch = _import_torch()
            if self.torch.cuda.is_initialized() is not False:
                raise RuntimeError('CUDA was initialized before device admission')
            _configure_numerics(self.torch)
            from .flan_t5_device_observation import _cuda_numerics
            from .gpu_oneshot import _runtime_uuid
            self.runtime_versions = _cuda_numerics()
            count = self._call(self.torch.cuda.device_count)
            if type(count) is not int or count != 1:
                raise RuntimeError('exactly one visible CUDA device is required')
            properties = self._call(self.torch.cuda.get_device_properties, 0)
            runtime_uuid = _runtime_uuid(self.torch, properties.uuid)
            OwnerSession.verify_runtime_binding(self.session, self.device_lease,
                                                runtime_uuid=runtime_uuid, device_count=count)
            if (type(properties.total_memory) is not int
                    or not 7 * _GIB <= properties.total_memory <= 9 * _GIB):
                raise RuntimeError('CUDA device total is outside the pinned 8 GiB range')
            self.device_total, self.device_name = properties.total_memory, properties.name
            self.allocator_cap = int(.25 * self.device_total)
            if self._call(self.torch.cuda.get_allocator_backend) != 'native':
                raise RuntimeError('the native CUDA allocator is required')
            self._call(self.torch.cuda.set_per_process_memory_fraction, .25, 0)
            if self._call(self.torch.cuda.get_per_process_memory_fraction, 0) != self.allocator_cap / self.device_total:
                raise RuntimeError('CUDA allocator cap readback differs')
            self._call(self.torch.cuda.manual_seed, 20261028)
            self._call(self.torch.cuda.reset_peak_memory_stats, 0)
            initial = self._stage('before_model_load')
            if initial['device_free_bytes'] < 3 * _GIB:
                raise RuntimeError('insufficient device-wide free memory before model loading')
            self.memory = DeviceMemoryCustody(self._telemetry, allocator_cap=self.allocator_cap,
                                              check_owner=self._guard)
            self.memory.start()
            self._stage('after_admission')
            self.phase = 'admitted'
            self.failed = False
        except BaseException as error:
            self.failed = True
            self.failure = error
            raise

    def load_model(self, asset_directory, stage_directory):
        if self.failed:
            raise RuntimeError('the supervised FLAN-T5 owner is uncertain')
        if self.phase != 'admitted':
            raise RuntimeError('the FLAN-T5 model may only be loaded after admission')
        self._model_guard()
        self.failed = True
        try:
            self.stage_directory = Path(stage_directory)
            self.asset_manifest = _stage_assets(asset_directory, self.stage_directory)
            self._model_guard()
            self.model = _load_staged_model(self.stage_directory, self.torch)
            self._model_guard()
            self._stage('before_model_copy')
            self._model_guard()
            self.converted = self.model.to('cuda:0')
            self._model_guard()
            if self.converted is not self.model:
                raise RuntimeError('model conversion returned a different object')
            from .flan_t5_device_observation import _model_contract, _validate_flan_checkpoint_model
            from .flan_t5_feasibility import verify_assets
            from .bert_gpu_family_owner import _capture_state
            _model_contract(self.model, self.torch.device('cuda:0'))
            _validate_flan_checkpoint_model(self.model)
            self._stage('after_model_copy')
            self.witness = _capture_state(self.model, self.torch, device='cuda:0',
                check_owner=self._model_guard, synchronize=lambda: self._call(self.torch.cuda.synchronize, 0))
            self._model_guard()
            if verify_assets(self.stage_directory) != self.asset_manifest:
                raise RuntimeError('staged checkpoint changed during model loading')
            self.dependency_origins = _dependency_origins(
                Path(os.environ['LOOKING_GLASS_FLAN_T5_BOOTSTRAP_SITE']))
            self._model_guard()
            self._stage('after_initial_state')
            self._model_guard()
            self.phase = 'model-ready'
            self.failed = False
        except BaseException as error:
            self.failed = True
            self.failure = error
            raise

    def observe(self, source, target, *, action, token_index=None, value=None):
        if self.failed:
            raise RuntimeError('the supervised FLAN-T5 owner is uncertain')
        if self.phase != 'model-ready':
            raise RuntimeError('the pinned model is not ready')
        self._model_guard()
        self.failed = True
        try:
            from .bert_gpu_calibration import check_memory
            from .bert_gpu_family_owner import _verify_state
            from .flan_t5_device_observation import observe_cuda_owned
            def synchronize():
                self._call(self.torch.cuda.synchronize, 0)
            def forward_stage(stage, snapshot):
                self._model_guard()
                if self.record_stage(stage, snapshot) is not None:
                    raise RuntimeError('memory recorder must return None or raise')
                check_memory(snapshot, expected_total=self.device_total, cap=self.allocator_cap,
                             commitment=self._policy_commitment())
                self._model_guard()
            _verify_state(self.model, self.witness, self.torch, device='cuda:0',
                          check_owner=self._model_guard, synchronize=synchronize)
            self._stage('before_observation')
            slot = self.memory.reserve_observation()
            self._model_guard()
            slot.result = observe_cuda_owned(self.model, source, target, action=action,
                token_index=token_index, value=value, custody=slot.custody,
                session=self.session, device_lease=self.device_lease,
                memory_lease=slot.lease, record_stage=forward_stage)
            self._model_guard()
            _verify_state(self.model, self.witness, self.torch, device='cuda:0',
                          check_owner=self._model_guard, synchronize=synchronize)
            self._stage('after_observation')
            slot.retained = RetainedObservation(slot.result, slot.custody, slot.lease,
                                                check_owner=self._model_guard,
                                                access_check=self._retained_guard)
            slot.result = None
            self._model_guard()
            self.failed = False
            return slot.retained
        except BaseException as error:
            self.failed = True
            self.failure = error
            raise

    def _verify_inputs_still_bound(self):
        from .flan_t5_feasibility import verify_assets
        if (_verify_source_tree(self.repo, self.git_executable, self.source_commit, SOURCE_PATHS)
                != self.source_hashes or verify_assets(self.stage_directory) != self.asset_manifest
                or _dependency_origins(Path(os.environ['LOOKING_GLASS_FLAN_T5_BOOTSTRAP_SITE']))
                != self.dependency_origins):
            raise RuntimeError('source or checkpoint changed during the owner run')

    def finish(self):
        """Dispose known CUDA roots and release only after synchronized cleanup."""
        if self.failed:
            raise RuntimeError('the supervised FLAN-T5 owner is uncertain')
        if self.phase != 'model-ready':
            raise RuntimeError('the pinned model is not ready for cleanup')
        self._model_guard()
        if (any(slot.lease is None or not slot.lease.released or slot.result is not None
                or slot.retained is None or not slot.retained._retired for slot in self.memory.slots)
                or len(self.memory.budget.active_leases()) != 1):
            raise RuntimeError('retained observations must be retired before model cleanup')
        self.failed = True
        try:
            from .bert_gpu_family_owner import _verify_state
            _verify_state(self.model, self.witness, self.torch, device='cuda:0',
                          check_owner=self._model_guard,
                          synchronize=lambda: self._call(self.torch.cuda.synchronize, 0))
            self._verify_inputs_still_bound()
            self._model_guard()
            model_reference = weakref.ref(self.model)
            self.witness = self.converted = self.model = None
            gc.collect()
            if model_reference() is not None:
                raise RuntimeError('model remains reachable after disposal')
            self._call(self.torch.cuda.synchronize, 0)
            self._call(self.torch.cuda.empty_cache)
            self._stage('cleanup_before_workspace_clear')
            if (str(self.torch.__version__) != '2.8.0+cu128'
                    or not callable(getattr(self.torch._C, '_cuda_clearCublasWorkspaces', None))):
                raise RuntimeError('unsupported CUDA workspace cleanup API')
            self._call(self.torch._C._cuda_clearCublasWorkspaces)
            self._call(self.torch.cuda.synchronize, 0)
            self._call(self.torch.cuda.empty_cache)
            cleanup = self._stage('after_cleanup')
            if cleanup['allocated_bytes'] != 0 or cleanup['reserved_bytes'] != 0:
                raise RuntimeError('CUDA allocator bytes remain after model disposal')
            self._model_guard()
            self.memory.model_lease.release()
            self._guard()
            if self.memory.budget.active_leases():
                raise RuntimeError('device memory commitments remain after cleanup')
            self.phase = 'disposed'
            self.failed = False
            return cleanup
        except BaseException as error:
            self.failed = True
            self.failure = error
            raise

class RetainedObservation:
    """Retain one observation claim until all known host storage aliases die."""

    def __init__(self, observation, custody, lease, *, check_owner, access_check=None):
        # Import CUDA-capable dependencies only after the owner has entered.
        import torch
        from .flan_t5_device_observation import (GPU_TRACE_CLAIM, HOST_TRACE_CLAIM,
                                                 Observation, ObservationCustody,
                                                 _observation_claim_is_bound)
        if (type(observation) is not Observation or type(custody) is not ObservationCustody
                or type(lease) is not MemoryLease or not callable(check_owner)
                or (access_check is not None and not callable(access_check))):
            raise ValueError('an owned observation, custody and memory lease are required')
        if check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        if (not _observation_claim_is_bound(lease, custody)
                or custody.host_result is not observation
                or custody.host_logits is not observation.logits or custody.host_state is not observation.state
                or custody.host_before is not observation.before or custody.host_after is not observation.after
                or any(getattr(custody, name) is not None for name in
                       ('input_ids', 'attention_mask', 'decoder_ids', 'decoder_mask', 'source_bits',
                        'source', 'plan', 'plan_captures', 'output', 'captures'))
                or custody.plan_sources != ()
                or lease.trace_bytes('gpu') != GPU_TRACE_CLAIM
                or lease.trace_bytes('host') != HOST_TRACE_CLAIM
                or not any(item is lease for item in lease._budget.active_leases())):
            raise ValueError('observation custody, lease identity or dedicated claims differ')
        references = []
        tensors = (observation.logits, observation.state, observation.before, observation.after)
        bits = (custody.host_logits_bits, custody.host_state_bits,
                custody.host_before_bits, custody.host_after_bits)
        for tensor, bit_view in zip(tensors, bits):
            if tensor is None:
                if bit_view is not None:
                    raise ValueError('unexpected host bit root')
                continue
            if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                    or tensor.dtype != torch.float32 or tensor.layout != torch.strided
                    or not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
                raise ValueError('host evidence must own compact CPU float32 storage')
            if (type(bit_view) is not torch.Tensor or bit_view.device.type != 'cpu'
                    or bit_view.dtype != torch.int32 or tuple(bit_view.shape) != tuple(tensor.shape)
                    or not bit_view.is_contiguous() or bit_view.storage_offset() != tensor.storage_offset()
                    or bit_view.untyped_storage() is not tensor.untyped_storage()):
                raise ValueError('host bit root differs from the observation storage')
            references.append(weakref.ref(tensor.untyped_storage()))
        self._owner_thread = current_thread()
        self._check_owner = check_owner if access_check is None else access_check
        self._custody = custody
        self._observation = observation
        self._lease = lease
        self._references = tuple(references)
        self._dropped = False
        self._releasing = False
        self._retired = False

    def _check(self):
        if current_thread() is not self._owner_thread:
            raise RuntimeError('observation custody belongs to its owner thread')
        if self._check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        if self._releasing:
            raise RuntimeError('observation claim release is uncertain')
        if self._retired:
            raise RuntimeError('observation was already retired')

    @property
    def value(self):
        self._check()
        if self._dropped:
            raise RuntimeError('observation root was already dropped')
        return self._observation

    def drop(self):
        """Drop the owner's roots after recording; aliases may still prevent retirement."""
        self._check()
        self._custody.host_result = self._custody.host_logits = self._custody.host_state = None
        self._custody.host_before = self._custody.host_after = None
        self._custody.host_logits_bits = self._custody.host_state_bits = None
        self._custody.host_before_bits = self._custody.host_after_bits = None
        self._observation = None
        self._dropped = True

    def retire(self):
        """Release a claim only after known host storage has no surviving aliases."""
        from .flan_t5_device_observation import GPU_TRACE_CLAIM, HOST_TRACE_CLAIM
        self._check()
        if not self._dropped:
            raise RuntimeError('drop the observation root before retirement')
        # An interrupted collection or count update is not retried: a finalizer
        # or release may have completed work before the interruption surfaced.
        self._releasing = True
        gc.collect()
        if any(reference() is not None for reference in self._references):
            self._releasing = False
            raise RuntimeError('observation host storage remains live')
        if self._check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        self._lease.release_trace('gpu', GPU_TRACE_CLAIM)
        if self._check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        self._lease.release_trace('host', HOST_TRACE_CLAIM)
        if self._check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        self._lease.release()
        if self._check_owner() is not None:
            raise RuntimeError('owner check must return None or raise on failure')
        self._retired = True
        self._releasing = False
