"""Versioned, measured workload resources for trusted service configuration.

A measured peak and a margin are estimates, not a guarantee against CUDA OOM.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from datetime import datetime, timezone
import re
from types import MappingProxyType

from .device_identity import DeviceIdentity, DeviceIdentityError
from .device_lease import DeviceLease
from .memory_budget import AdmissionError, CudaMemorySnapshot, MemoryLease
from .supervisor import OwnerSession


_INPUT_BYTES = {'torch.int64': 8, 'torch.int32': 4, 'torch.bool': 1,
                'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4,
                'torch.float64': 8}


def _text(value, name):
    if (type(value) is not str or not value or value != value.strip()
            or len(value) > 256 or value.lower() in ('unknown', 'n/a', 'none')):
        raise ValueError(f'{name} must be a known nonempty bounded string')


def _hash(value, name):
    if type(value) is not str or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')


def _bytes(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')


_MODEL_FIELDS = (
    'checkpoint_sha256', 'checkpoint_revision', 'model_config_sha256',
    'source_sha256', 'runtime_settings_sha256', 'torch_version',
    'transformers_version', 'cuda_runtime_version', 'driver_version',
    'python_version', 'device', 'device_uuid', 'device_name',
    'compute_capability', 'device_total_bytes', 'model_dtype',
    'compute_dtype', 'attention_backend', 'model_mode', 'grad_mode', 'autocast',
    'use_cache', 'compiled',
)


def _canonical_value(value, depth=0):
    if depth > 64:
        raise ValueError('canonical JSON nesting is too deep')
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) in (list, tuple):
        return [_canonical_value(item, depth + 1) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError('canonical JSON object keys must be strings')
        return {key: _canonical_value(item, depth + 1) for key, item in value.items()}
    raise TypeError('canonical JSON supports only strings, integers, booleans, null, arrays and objects')


def _canonical_bytes(value):
    return json.dumps(_canonical_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def canonical_fingerprint(value):
    """Return a deterministic SHA-256 of supported canonical JSON data."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def model_runtime_fingerprint(fields):
    """Bind model assets, implementation, runtime and physical device identity."""
    if type(fields) is not dict:
        raise TypeError('model identity fields must be a dictionary')
    try:
        identity = {name: fields[name] for name in _MODEL_FIELDS}
    except KeyError as exc:
        raise ValueError(f'missing model identity field: {exc.args[0]}') from exc
    return canonical_fingerprint({'schema': 'model-runtime-identity-v1', 'identity': identity})


class ProfileUnavailable(LookupError):
    """No trusted profile matches the exact workload."""


class ProfilingSynchronizationError(RuntimeError):
    """A profiling boundary could not establish device completion."""

    def __init__(self, message, original_error=None):
        super().__init__(message)
        self.original_error = original_error


@dataclass(frozen=True)
class WorkloadKey:
    model_id: str
    checkpoint_sha256: str
    checkpoint_revision: str
    model_config_sha256: str
    source_sha256: str
    runtime_settings_sha256: str
    torch_version: str
    transformers_version: str
    cuda_runtime_version: str
    driver_version: str
    python_version: str
    device: str
    device_uuid: str
    device_name: str
    compute_capability: tuple
    device_total_bytes: int
    model_dtype: str
    compute_dtype: str
    attention_backend: str
    model_mode: str
    grad_mode: str
    autocast: str
    use_cache: bool
    compiled: bool
    warmup_state: str
    warmup_protocol_sha256: str
    operation: str
    capture_mode: str
    capture_policy_sha256: str
    batch_size: int
    sequence_length: int
    input_dtypes: tuple
    output_limit_bytes: int
    max_gpu_trace_bytes: int
    max_host_trace_bytes: int

    def __post_init__(self):
        for name in ('model_id', 'checkpoint_sha256', 'model_config_sha256',
                     'source_sha256', 'runtime_settings_sha256', 'capture_policy_sha256',
                     'warmup_protocol_sha256'):
            _hash(getattr(self, name), name)
        for name in ('checkpoint_revision', 'torch_version', 'transformers_version',
                     'cuda_runtime_version', 'driver_version', 'python_version',
                     'device_uuid', 'device_name', 'model_dtype', 'compute_dtype',
                     'attention_backend'):
            _text(getattr(self, name), name)
        if self.device != 'cuda:0':
            raise ValueError('resource-profile-v1 supports one CUDA device at cuda:0')
        if not re.fullmatch(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', self.device_uuid):
            raise ValueError('device_uuid must be a canonical full physical GPU UUID')
        if (type(self.compute_capability) is not tuple or len(self.compute_capability) != 2
                or any(type(value) is not int or value < 0 for value in self.compute_capability)):
            raise ValueError('compute_capability must be an immutable integer pair')
        _bytes(self.device_total_bytes, 'device_total_bytes', positive=True)
        if self.model_mode != 'eval' or self.grad_mode not in ('inference', 'no_grad', 'enabled'):
            raise ValueError('unsupported model or gradient mode')
        if self.autocast not in ('disabled', 'torch.float16', 'torch.bfloat16'):
            raise ValueError('unsupported autocast mode')
        if type(self.use_cache) is not bool or type(self.compiled) is not bool:
            raise TypeError('use_cache and compiled must be booleans')
        if self.warmup_state not in ('cold', 'warm'):
            raise ValueError('warmup_state must be cold or warm')
        if self.operation not in ('model_load', 'forward'):
            raise ValueError('unsupported operation')
        if self.capture_mode not in ('none', 'id', 'direct', 'gradient'):
            raise ValueError('unsupported capture mode')
        for name in ('batch_size', 'sequence_length', 'output_limit_bytes',
                     'max_gpu_trace_bytes', 'max_host_trace_bytes'):
            _bytes(getattr(self, name), name)
        if type(self.input_dtypes) is not tuple:
            raise TypeError('input_dtypes must be an immutable tuple')
        names = []
        for item in self.input_dtypes:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError('input signature entries must be immutable pairs')
            name, dtype = item
            if type(name) is not str or not name.isidentifier():
                raise ValueError('invalid input name')
            if dtype not in _INPUT_BYTES:
                raise ValueError('unsupported input dtype')
            names.append(name)
        if names != sorted(set(names)):
            raise ValueError('input names must be unique and sorted')
        if self.operation == 'model_load':
            if (self.batch_size or self.sequence_length or self.input_dtypes
                    or self.output_limit_bytes or self.capture_mode != 'none'
                    or self.warmup_state != 'cold'):
                raise ValueError('model load cannot include request inputs or captures')
        elif not self.batch_size or not self.sequence_length or not self.input_dtypes or not self.output_limit_bytes:
            raise ValueError('request profiles require a positive shape, inputs and output cap')
        if self.capture_mode == 'none' and (self.max_gpu_trace_bytes or self.max_host_trace_bytes):
            raise ValueError('a no-capture profile cannot reserve trace bytes')
        if self.model_id != model_runtime_fingerprint(asdict(self)):
            raise ValueError('model_id does not match model and runtime identity')

    def fingerprint(self):
        return canonical_fingerprint({'schema': 'workload-key-v1', 'key': asdict(self)})


@dataclass(frozen=True)
class MemorySample:
    workload_sha256: str
    sample_id: str
    case_sha256: str
    seed: int
    status: str
    before: CudaMemorySnapshot
    after: CudaMemorySnapshot
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self):
        _hash(self.workload_sha256, 'workload_sha256')
        _text(self.sample_id, 'sample_id')
        _hash(self.case_sha256, 'case_sha256')
        _bytes(self.seed, 'seed')
        if self.status != 'completed':
            raise ValueError('only completed measurements may be profiled')
        if type(self.before) is not CudaMemorySnapshot or type(self.after) is not CudaMemorySnapshot:
            raise TypeError('before and after must be CUDA memory snapshots')
        _bytes(self.peak_allocated_bytes, 'peak_allocated_bytes')
        _bytes(self.peak_reserved_bytes, 'peak_reserved_bytes')
        if self.before.device_total_bytes != self.after.device_total_bytes:
            raise ValueError('device total changed during measurement')
        if self.peak_allocated_bytes < max(self.before.process_allocated_bytes,
                                           self.after.process_allocated_bytes):
            raise ValueError('allocated peak is below a measured endpoint')
        if self.peak_reserved_bytes < max(self.before.process_reserved_bytes,
                                          self.after.process_reserved_bytes,
                                          self.peak_allocated_bytes):
            raise ValueError('reserved peak is below an endpoint or allocated peak')
        if self.peak_reserved_bytes > self.before.device_total_bytes:
            raise ValueError('reserved peak exceeds device total')


@dataclass(frozen=True)
class ProfileProvenance:
    protocol: str
    source_manifest_sha256: str
    evidence_sha256: str
    collected_utc: str
    isolation: str
    boundaries: str

    def __post_init__(self):
        if self.protocol != 'resource-profile-v1':
            raise ValueError('unsupported resource profile protocol')
        if self.isolation != 'cooperative-exclusive':
            raise ValueError('profiling requires a cooperative exclusive device lease')
        if self.boundaries != 'synchronize-reset-synchronize':
            raise ValueError('profiling requires synchronized boundaries and peak reset')
        _hash(self.source_manifest_sha256, 'source_manifest_sha256')
        _hash(self.evidence_sha256, 'evidence_sha256')
        _text(self.collected_utc, 'collected_utc')
        try:
            parsed = datetime.fromisoformat(self.collected_utc.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('collected_utc must be an ISO timestamp') from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError('collected_utc must identify UTC')


@dataclass(frozen=True)
class MeasuredResourceProfile:
    key: WorkloadKey
    samples: tuple
    provenance: ProfileProvenance
    margin_bytes: int
    margin_bps: int

    def __post_init__(self):
        if type(self.key) is not WorkloadKey or type(self.provenance) is not ProfileProvenance:
            raise TypeError('profile requires a workload key and provenance')
        if type(self.samples) is not tuple or not self.samples:
            raise ValueError('profile requires an immutable, nonempty sample set')
        if any(type(sample) is not MemorySample for sample in self.samples):
            raise TypeError('invalid memory sample')
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError('duplicate sample IDs')
        if any(sample.workload_sha256 != self.key.fingerprint() for sample in self.samples):
            raise ValueError('sample workload identity does not match profile')
        if any(sample.before.device_total_bytes != self.key.device_total_bytes for sample in self.samples):
            raise ValueError('sample device total does not match workload identity')
        _bytes(self.margin_bytes, 'margin_bytes', positive=True)
        _bytes(self.margin_bps, 'margin_bps')
        if hashlib.sha256(encode_evidence(self.key, self.samples)).hexdigest() != self.provenance.evidence_sha256:
            raise ValueError('sample evidence digest does not match profile')

    @classmethod
    def build(cls, key, samples, provenance, *, margin_bytes, margin_bps):
        return cls(key, samples, provenance, margin_bytes, margin_bps)

    def to_json(self):
        return _canonical_bytes(_profile_document(self)).decode('utf-8')

    @property
    def profile_id(self):
        return canonical_fingerprint(_profile_document(self))

    @property
    def estimated_incremental_bytes(self):
        bound = _declared_gpu_allowance(self.key)
        observed = max(max(0, sample.peak_allocated_bytes - sample.before.process_allocated_bytes,
                           sample.peak_reserved_bytes - sample.before.process_reserved_bytes,
                           sample.before.device_free_bytes - sample.after.device_free_bytes)
                       for sample in self.samples)
        basis = bound + observed
        return basis + self.margin_bytes + (basis * self.margin_bps + 9999) // 10000


class TorchMemoryProbe:
    """Adapter for a trusted, already-bound PyTorch CUDA device."""

    def __init__(self, cuda, *, device, device_uuid, runtime_binding):
        if not callable(runtime_binding):
            raise TypeError('an authoritative runtime binding callback is required')
        self._cuda = cuda
        self.device = device
        self.device_uuid = device_uuid
        self._runtime_binding = runtime_binding

    def runtime_binding(self):
        return self._runtime_binding()

    def synchronize(self):
        self._cuda.synchronize(self.device)

    def snapshot(self):
        free, total = self._cuda.mem_get_info(self.device)
        return CudaMemorySnapshot(free, total,
                                  self._cuda.memory_allocated(self.device),
                                  self._cuda.memory_reserved(self.device))

    def reset_peak(self):
        self._cuda.reset_peak_memory_stats(self.device)

    def peak_allocated_bytes(self):
        return self._cuda.max_memory_allocated(self.device)

    def peak_reserved_bytes(self):
        return self._cuda.max_memory_reserved(self.device)


def _declared_gpu_allowance(key):
    inputs = (key.batch_size * key.sequence_length
              * sum(_INPUT_BYTES[dtype] for _, dtype in key.input_dtypes))
    return inputs + key.output_limit_bytes + key.max_gpu_trace_bytes


def _check_memory_lease(lease, key):
    if (type(lease) is not MemoryLease or lease.released
            or lease not in lease._budget.active_leases()
            or lease.estimated_incremental_bytes <= 0):
        raise ValueError('an active positive memory lease is required')
    if lease.estimated_incremental_bytes < _declared_gpu_allowance(key):
        raise ValueError('memory lease does not cover declared GPU allowances')
    if (lease._trace_limits['gpu'] < key.max_gpu_trace_bytes
            or lease._trace_limits['host'] < key.max_host_trace_bytes):
        raise ValueError('memory lease trace caps do not cover capture policy')


def _check_headroom(budget, telemetry):
    if (telemetry.process_reserved_bytes + budget.committed_incremental_bytes
            + budget.reserve_bytes > telemetry.device_total_bytes
            or budget.committed_incremental_bytes + budget.reserve_bytes > telemetry.device_free_bytes):
        raise AdmissionError('device pressure changed after admission')


def _check_owner(identity, lease, owner_session):
    if type(owner_session) is not OwnerSession:
        raise ValueError('an active supervised OwnerSession is required')
    # Use the class implementation so a substituted instance method cannot
    # bypass the supervised marker and parent-liveness checks.
    OwnerSession.check(owner_session, lease)
    if type(owner_session.identity) is not DeviceIdentity or owner_session.identity != identity:
        raise DeviceIdentityError('supervised owner identity does not match the workload identity')


def _verify_device_binding(key, probe, identity, lease, owner_session):
    if type(identity) is not DeviceIdentity or identity.gpu_uuid != key.device_uuid:
        raise ValueError('a matching canonical device identity is required')
    DeviceIdentity.validate_paths(identity)
    if os.path.normcase(str(lease.path)) != os.path.normcase(str(identity.lock_path)):
        raise DeviceIdentityError('device lease path does not match this physical UUID')
    DeviceLease.check(lease)
    _check_owner(identity, lease, owner_session)
    if not callable(getattr(probe, 'runtime_binding', None)):
        raise ValueError('an authoritative runtime device binding is required')
    binding = probe.runtime_binding()
    if type(binding) is not tuple or len(binding) != 2:
        raise ValueError('runtime binding must contain UUID and visible device count')
    runtime_uuid, device_count = binding
    OwnerSession.verify_runtime_binding(owner_session, lease, runtime_uuid=runtime_uuid,
                                        device_count=device_count, cuda_device_index=0)


def _synchronize(probe, original_error=None):
    try:
        probe.synchronize()
    except BaseException as exc:
        raise ProfilingSynchronizationError('profiling device completion is unverified',
                                             original_error=original_error) from exc


def collect_sample(key, probe, operation, *, owner_session, device_identity, device_lease,
                   memory_lease, sample_id, case_sha256, seed):
    """Measure one completed callback while externally owned leases remain held.

    The trusted supervisor must validate its owner session and device identity
    before acquiring a telemetry-backed memory lease, bind the probe and both
    leases to the same physical device, and hold its execution gate. This
    function never releases either lease, including on error or unknown completion.
    """
    if type(key) is not WorkloadKey or not callable(operation):
        raise TypeError('a workload key and callable operation are required')
    _text(sample_id, 'sample_id')
    _hash(case_sha256, 'case_sha256')
    _bytes(seed, 'seed')
    if getattr(probe, 'device', None) != key.device or getattr(probe, 'device_uuid', None) != key.device_uuid:
        raise ValueError('probe and workload device identity do not match')
    if type(device_lease) is not DeviceLease:
        raise ValueError('an active exclusive DeviceLease is required')
    DeviceLease.check(device_lease)
    _check_memory_lease(memory_lease, key)
    _verify_device_binding(key, probe, device_identity, device_lease, owner_session)
    budget = memory_lease._budget.snapshot()
    if budget.telemetry.device_total_bytes != key.device_total_bytes:
        raise ValueError('memory budget and workload device totals do not match')
    _check_headroom(budget, budget.telemetry)
    _synchronize(probe)
    before = probe.snapshot()
    if type(before) is not CudaMemorySnapshot or before.device_total_bytes != key.device_total_bytes:
        raise ValueError('measured device total does not match workload identity')
    budget = memory_lease._budget.snapshot()
    if budget.telemetry.device_total_bytes != key.device_total_bytes:
        raise ValueError('memory budget and workload device totals do not match')
    _check_headroom(budget, budget.telemetry)
    _check_headroom(budget, before)
    _check_owner(device_identity, device_lease, owner_session)
    _check_memory_lease(memory_lease, key)
    probe.reset_peak()
    try:
        result = operation()
    except BaseException as exc:
        _synchronize(probe, original_error=exc)
        raise
    _synchronize(probe)
    if result is not None:
        raise ValueError('profiling callback must complete offload and return None')
    DeviceLease.check(device_lease)
    _check_memory_lease(memory_lease, key)
    _verify_device_binding(key, probe, device_identity, device_lease, owner_session)
    after = probe.snapshot()
    allocated = probe.peak_allocated_bytes()
    reserved = probe.peak_reserved_bytes()
    result = MemorySample(workload_sha256=key.fingerprint(), sample_id=sample_id,
                          case_sha256=case_sha256, seed=seed, status='completed',
                          before=before, after=after, peak_allocated_bytes=allocated,
                          peak_reserved_bytes=reserved)
    return result


def encode_evidence(key, samples):
    """Encode identity-bound completed samples for a separately saved artifact."""
    if type(key) is not WorkloadKey or type(samples) is not tuple or not samples:
        raise ValueError('evidence requires a key and immutable nonempty samples')
    if any(type(item) is not MemorySample or item.workload_sha256 != key.fingerprint()
           for item in samples):
        raise ValueError('evidence sample does not match workload identity')
    return _canonical_bytes({'schema': 'resource-profile-evidence-v1',
                             'key': asdict(key),
                             'samples': [asdict(item) for item in samples]})


def _profile_document(profile):
    return {'schema': 'resource-profile-v1', 'key': asdict(profile.key),
            'samples': [asdict(item) for item in profile.samples],
            'provenance': asdict(profile.provenance),
            'margin_bytes': profile.margin_bytes, 'margin_bps': profile.margin_bps}


def _exact_fields(cls, value):
    if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
        raise ValueError(f'invalid fields for {cls.__name__}')
    return dict(value)


def _no_duplicate_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def load_profile(document, *, evidence, source_manifest, expected_profile_id):
    """Verify exact evidence/source bytes and reconstruct a strict profile document.

    File provenance and the correctness of its attestations must be established
    by the trusted service; matching hashes alone do not authenticate evidence.
    """
    _hash(expected_profile_id, 'expected_profile_id')
    if type(document) not in (str, bytes) or type(evidence) is not bytes or type(source_manifest) is not bytes:
        raise TypeError('document, evidence, and source manifest must be bytes or text as specified')
    if len(document) > 16 * 1024 * 1024:
        raise ValueError('profile document is too large')
    try:
        data = json.loads(document, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite JSON')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid profile JSON') from exc
    if (type(data) is not dict or set(data) != {'schema', 'key', 'samples', 'provenance',
                                                 'margin_bytes', 'margin_bps'}
            or data['schema'] != 'resource-profile-v1'):
        raise ValueError('invalid resource profile document')
    key_data = _exact_fields(WorkloadKey, data['key'])
    if type(key_data['compute_capability']) is not list or type(key_data['input_dtypes']) is not list:
        raise ValueError('invalid workload tuples')
    key_data['compute_capability'] = tuple(key_data['compute_capability'])
    if any(type(item) is not list for item in key_data['input_dtypes']):
        raise ValueError('invalid input signature')
    key_data['input_dtypes'] = tuple(tuple(item) for item in key_data['input_dtypes'])
    key = WorkloadKey(**key_data)
    if type(data['samples']) is not list:
        raise ValueError('samples must be an array')
    samples = []
    for entry in data['samples']:
        item = _exact_fields(MemorySample, entry)
        item['before'] = CudaMemorySnapshot(**_exact_fields(CudaMemorySnapshot, item['before']))
        item['after'] = CudaMemorySnapshot(**_exact_fields(CudaMemorySnapshot, item['after']))
        samples.append(MemorySample(**item))
    provenance = ProfileProvenance(**_exact_fields(ProfileProvenance, data['provenance']))
    if hashlib.sha256(source_manifest).hexdigest() != provenance.source_manifest_sha256:
        raise ValueError('source manifest digest mismatch')
    if hashlib.sha256(evidence).hexdigest() != provenance.evidence_sha256:
        raise ValueError('evidence digest mismatch')
    profile = MeasuredResourceProfile.build(key, tuple(samples), provenance,
                                           margin_bytes=data['margin_bytes'],
                                           margin_bps=data['margin_bps'])
    if profile.profile_id != expected_profile_id:
        raise ValueError('profile does not match the approved profile ID')
    return profile


class ProfileRegistry:
    def __init__(self, profiles, *, approved_profile_ids):
        if type(approved_profile_ids) is not tuple or not approved_profile_ids:
            raise ValueError('approved_profile_ids must be a nonempty immutable tuple')
        for profile_id in approved_profile_ids:
            _hash(profile_id, 'approved_profile_id')
        if len(set(approved_profile_ids)) != len(approved_profile_ids):
            raise ValueError('duplicate approved profile ID')
        if type(profiles) is not tuple or not profiles:
            raise ValueError('profiles must be a nonempty immutable tuple')
        records = {}
        for profile in profiles:
            if type(profile) is not MeasuredResourceProfile:
                raise TypeError('registry accepts only measured profiles')
            if profile.key in records:
                raise ValueError('duplicate workload profile')
            if profile.profile_id not in approved_profile_ids:
                raise ValueError('profile is not approved by the trusted configuration')
            records[profile.key] = profile
        self._profiles = MappingProxyType(records)

    def resolve(self, key):
        if type(key) is not WorkloadKey:
            raise ProfileUnavailable('a trusted workload key is required')
        try:
            return self._profiles[key]
        except KeyError as exc:
            raise ProfileUnavailable('no exact measured resource profile') from exc



def resolve_worker_profile(registry, trusted_live_key):
    """Resolve a full trusted live identity before adapting it to the serial worker.

    The supervisor must construct ``trusted_live_key`` from verified live state,
    never from a client-provided profile, key or estimate.
    """
    if type(registry) is not ProfileRegistry:
        raise TypeError('a trusted ProfileRegistry is required')
    profile = registry.resolve(trusted_live_key)
    key = profile.key
    supported_inputs = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids'}
    supported_dtypes = {'torch.int32', 'torch.int64', 'torch.bool'}
    if (key.operation != 'forward' or key.capture_mode != 'none'
            or key.model_mode != 'eval' or key.grad_mode != 'inference'
            or key.autocast != 'disabled' or key.use_cache or key.compiled
            or key.max_gpu_trace_bytes or key.max_host_trace_bytes
            or not any(name == 'input_ids' for name, _ in key.input_dtypes)
            or any(name not in supported_inputs or dtype not in supported_dtypes
                   or (dtype == 'torch.bool' and name != 'attention_mask')
                   for name, dtype in key.input_dtypes)):
        raise ProfileUnavailable('profile is not supported by the serial forward worker')
    from .budgeted_worker import WorkloadProfile
    return WorkloadProfile(profile_id=profile.profile_id, model_id=key.model_id,
                           device=key.device, shape=(key.batch_size, key.sequence_length),
                           input_dtypes=key.input_dtypes, output_limit_bytes=key.output_limit_bytes,
                           estimated_incremental_bytes=profile.estimated_incremental_bytes)
