"""Supervised, profile-gated CUDA adapter for natural agreement patching.

This is an in-owner component, not a launcher or an attestation. A trusted
controller must verify the exact checkpoint and source/runtime manifest, hold
the physical device lease through child exit, and independently audit evidence.
The adapter cannot make external GPU processes cooperate or guarantee no OOM.

Profile estimates are per-sample increments, not a whole-stage peak. The
controller must hold separate commitments for model load, initialization and
any persistent carryover, including host state-hash scratch. Before every
forward, state hash and device RNG operation this adapter rechecks current
device telemetry and outstanding commitments. This does not prove a future
allocation will fit, or make the measured samples worst-case bounds.

This is a dedicated, single-threaded owner. Check ``quarantined`` only after
the owner reaches a quiescent result boundary: it is transiently true during
an ordinary in-flight synchronization. An external supervisor must impose a
bounded deadline and terminate the process with its marker retained if a drain
hangs or completion is unknown. Constructor failure after the owner gate is
installed likewise requires process replacement, not a retry or attestation.
Known device buffers are retained on uncertain completion, but Python can be
interrupted between an allocation's return and assignment, and model-internal
temporaries cannot all be rooted by this adapter.
"""

from dataclasses import replace
import hashlib
import json
import sys
import threading

import torch

from .agreement_numeric import decode_f32_le
from .agreement_patch_runner import (AgreementCase, NaturalAgreementRunner, StudyFailure, _RNG_SEED,
                                     _check_cpu_precision, _error_message, _same_state_identity,
                                     _runtime_configuration, _state_identity, _steps_for_capture, _validate_cases)
from .distilgpt2_intervention_dev import _hook_state
from .device_identity import DeviceIdentity
from .device_lease import DeviceLease
from .gpu_oneshot import _runtime_uuid
from .intervention_plan import ActivationPlan
from .memory_budget import CudaMemorySnapshot, MemoryLease
from .model_ownership import ModelLease
from .pretrained import ASSETS, REVISION
from .resource_profile import (MeasuredResourceProfile, TorchMemoryProbe, WorkloadKey, _check_headroom,
                               _check_memory_lease, canonical_fingerprint)
from .supervisor import OwnerSession


GIB = 1024 ** 3
TRACE_BYTES = 2 * 1024 ** 2
OUTPUT_BYTES = 5 * 50257 * 4
CALIBRATION_INPUT_IDS = (464, 3797, 1474, 262, 9168)  # Already-exposed public preflight row.
CALIBRATION_DONOR_IDS = (464, 11875, 1474, 262, 9168)  # Corresponding public cats row.
CALIBRATION_KINDS = ('clean_capture',
                     'layer1_attn_cross_patch', 'layer1_mlp_cross_patch',
                     'layer2_attn_cross_patch', 'layer2_mlp_cross_patch',
                     'layer3_attn_cross_patch', 'layer3_mlp_cross_patch',
                     'layer4_attn_cross_patch', 'layer4_mlp_cross_patch',
                     'null', 'self_patch', 'block_last_cross_patch', 'block_subject_cross_patch')
_CALIBRATION_ACTIONS = (
    ('clean_capture', None, 'capture', None),
    ('layer1_attn_cross_patch', 'transformer.h.1.attn', 'cross_patch', 4),
    ('layer1_mlp_cross_patch', 'transformer.h.1.mlp', 'cross_patch', 4),
    ('layer2_attn_cross_patch', 'transformer.h.2.attn', 'cross_patch', 4),
    ('layer2_mlp_cross_patch', 'transformer.h.2.mlp', 'cross_patch', 4),
    ('layer3_attn_cross_patch', 'transformer.h.3.attn', 'cross_patch', 4),
    ('layer3_mlp_cross_patch', 'transformer.h.3.mlp', 'cross_patch', 4),
    ('layer4_attn_cross_patch', 'transformer.h.4.attn', 'cross_patch', 4),
    ('layer4_mlp_cross_patch', 'transformer.h.4.mlp', 'cross_patch', 4),
    ('null', 'transformer.h.1.attn', 'null', 4),
    ('self_patch', 'transformer.h.1.attn', 'self_patch', 4),
    ('block_last_cross_patch', 'transformer.h.5', 'cross_patch', 4),
    ('block_subject_cross_patch', 'transformer.h.5', 'cross_patch', 1),
)
_CLEAN_BINDINGS = tuple(
    (f'transformer.h.{layer}.{component}', 4, [] if component == 'mlp' else [0])
    for layer in (1, 2, 3, 4) for component in ('attn', 'mlp')
) + (('transformer.h.5', 4, [0]), ('transformer.h.5', 1, [0]))
CAPTURE_POLICY = {
    'schema': 'agreement-cuda-capture-v1', 'batch_size': 1, 'sequence_length': 5,
    'clean_sites': 10, 'action_witnesses': ['whole_before', 'before', 'after', 'whole_after'],
    'max_tensor_bytes': 1024 ** 2, 'max_capture_bytes': 512 * 1024,
    'max_source_bytes': 128 * 1024, 'gpu_trace_allowance_bytes': TRACE_BYTES,
    'host_trace_allowance_bytes': TRACE_BYTES, 'full_output_bytes': OUTPUT_BYTES,
    'state_hash_chunk_bytes': 1024 ** 2,
    'measurement_scope': 'state-check-input-copy-plan-forward-offload-sync-disposal-state-check',
    'clone_custody': 'retain-plan-sources-and-assigned-captures-until-verified-drain',
    'calibration_kinds': list(CALIBRATION_KINDS), 'clean_bindings': list(_CLEAN_BINDINGS),
    'seed': _RNG_SEED,
}
CAPTURE_POLICY_SHA256 = canonical_fingerprint(CAPTURE_POLICY)


def calibration_case_sha256(kind):
    if kind not in CALIBRATION_KINDS:
        raise ValueError('unknown calibration action')
    _, module_path, operation, token_index = next(action for action in _CALIBRATION_ACTIONS if action[0] == kind)
    return canonical_fingerprint({'schema': 'agreement-cuda-calibration-case-v1',
                                  'kind': kind, 'target_input_ids': list(CALIBRATION_INPUT_IDS),
                                  'donor_input_ids': list(CALIBRATION_DONOR_IDS) if operation == 'cross_patch' else None,
                                  'executed_input_ids': list(CALIBRATION_DONOR_IDS if kind == 'clean_capture'
                                                             else CALIBRATION_INPUT_IDS),
                                  'subject_index': 1, 'module_path': module_path, 'operation': operation,
                                  'batch_index': 0 if module_path is not None else None,
                                  'token_index': token_index, 'invocation': 0,
                                  'donor_module_path': module_path if operation == 'cross_patch' else None,
                                  'donor_token_index': token_index if operation == 'cross_patch' else None,
                                  'attention_mask': [1] * 5,
                                  'output_selector': ([] if module_path and module_path.endswith('.mlp')
                                                      else [0] if module_path else None),
                                  'capture_policy_sha256': CAPTURE_POLICY_SHA256})


WARMUP_POLICY = {'schema': 'agreement-cuda-warmup-v1',
                 'input_ids': list(CALIBRATION_INPUT_IDS), 'subject_index': 1,
                 'attention_mask': [1] * 5, 'operation': 'ten-site-clean-capture-and-offload',
                 'seed': _RNG_SEED, 'capture_policy_sha256': CAPTURE_POLICY_SHA256}
WARMUP_POLICY_SHA256 = canonical_fingerprint(WARMUP_POLICY)


def _validate_workload_key(key):
    if type(key) is not WorkloadKey:
        raise ValueError('a verified live workload identity is required')
    if (key.checkpoint_sha256 != ASSETS['model.safetensors'][1] or key.checkpoint_revision != REVISION
            or key.device != 'cuda:0' or key.model_dtype != 'torch.float32'
            or key.torch_version != '2.8.0+cu128' or key.cuda_runtime_version != '12.8'
            or key.compute_dtype != 'torch.float32' or key.attention_backend != 'eager'
            or key.model_mode != 'eval' or key.grad_mode != 'inference' or key.autocast != 'disabled'
            or key.use_cache or key.compiled or key.warmup_state != 'warm'
            or key.warmup_protocol_sha256 != WARMUP_POLICY_SHA256
            or key.operation != 'forward' or key.capture_mode != 'direct'
            or key.capture_policy_sha256 != CAPTURE_POLICY_SHA256
            or key.batch_size != 1 or key.sequence_length != 5
            or key.input_dtypes != (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64'))
            or key.output_limit_bytes != OUTPUT_BYTES
            or key.max_gpu_trace_bytes != TRACE_BYTES or key.max_host_trace_bytes != TRACE_BYTES):
        raise ValueError('profile is not for the fixed five-token agreement workload')


def _validate_profile(profile, expected_key):
    if type(profile) is not MeasuredResourceProfile:
        raise ValueError('a measured profile and live workload identity are required')
    _validate_workload_key(expected_key)
    if profile.key != expected_key:
        raise ValueError('measured profile does not match the live workload identity')
    if profile.margin_bytes < 512 * 1024 ** 2 or profile.margin_bps < 2000:
        raise ValueError('profile does not include the fixed safety margin')
    if (len(profile.samples) != len(CALIBRATION_KINDS)
            or any(sample.sample_id != kind or sample.case_sha256 != calibration_case_sha256(kind)
                   or sample.seed != _RNG_SEED for sample, kind in zip(profile.samples, CALIBRATION_KINDS))):
        raise ValueError('profile does not contain the exact ordered calibration actions')


def _validate_warmup_profile(profile, expected_key):
    if type(profile) is not MeasuredResourceProfile or type(expected_key) is not WorkloadKey:
        raise ValueError('a measured cold warmup profile is required')
    cold_key = replace(expected_key, warmup_state='cold')
    if profile.key != cold_key:
        raise ValueError('warmup profile is not for the matching cold workload')
    if (profile.margin_bytes < 512 * 1024 ** 2 or profile.margin_bps < 2000
            or len(profile.samples) != 1 or profile.samples[0].sample_id != 'warmup'
            or profile.samples[0].case_sha256 != WARMUP_POLICY_SHA256
            or profile.samples[0].seed != _RNG_SEED):
        raise ValueError('cold warmup profile does not match its fixed sample and margin')


# Dedicated direct owners may bind one runner to a physical lease for the
# process lifetime. Holding the lease and memory commitment strongly also
# prevents an interrupted construction from silently freeing this reservation.
_owner_gate_lock = threading.Lock()
_owner_gates = {}


class _RetainedPlanCaptures(dict):
    """Keep private capture clones alive if a failed plan clears its map.

    Failed plans remain uncommitted, so their public capture accessor still
    rejects reads. The dedicated owner drops this map only after a good drain.
    """

    def clear(self):
        return None


class GuardedAgreementMemoryProbe(TorchMemoryProbe):
    """Bind each individual CUDA telemetry query to the live owner lease."""

    def __init__(self, cuda, *, device, device_uuid, runtime_binding, session, device_lease):
        if type(session) is not OwnerSession or type(device_lease) is not DeviceLease:
            raise ValueError('an exact owner session and device lease are required')
        super().__init__(cuda, device=device, device_uuid=device_uuid, runtime_binding=runtime_binding)
        self._owner_session, self._device_lease = session, device_lease
        self._bound_runner = None

    def _guard(self):
        OwnerSession.check(self._owner_session, self._device_lease)
        DeviceLease.check(self._device_lease)
        identity = self._owner_session.identity
        if (type(identity) is not DeviceIdentity or identity.gpu_uuid != self.device_uuid
                or self._owner_session._runtime_verified is not True):
            raise StudyFailure('memory probe owner binding is no longer valid')
        # The dedicated runner checks its active job and (for calibration)
        # resident leases without invoking telemetry. MemoryBudget's lock is
        # reentrant; a cross-thread stalled owner still requires a watchdog.
        if self._bound_runner is not None:
            self._bound_runner._check_owner()

    def snapshot(self):
        self._guard()
        free, total = self._cuda.mem_get_info(self.device)
        self._guard()
        allocated = self._cuda.memory_allocated(self.device)
        self._guard()
        reserved = self._cuda.memory_reserved(self.device)
        self._guard()
        return CudaMemorySnapshot(free, total, allocated, reserved)

    def synchronize(self):
        # Only the runner's sticky-drain method can establish safe completion.
        raise StudyFailure('use the agreement owner synchronization method')

    def reset_peak(self):
        self._guard()
        self._cuda.reset_peak_memory_stats(self.device)
        self._guard()

    def peak_allocated_bytes(self):
        self._guard()
        value = self._cuda.max_memory_allocated(self.device)
        self._guard()
        return value

    def peak_reserved_bytes(self):
        self._guard()
        value = self._cuda.max_memory_reserved(self.device)
        self._guard()
        return value


class CudaAgreementRunner(NaturalAgreementRunner):
    """Use the natural executor under a live, explicitly budgeted GPU owner.

    The supplied ``expected_key`` must be independently reconstructed from the
    verified live runtime, not copied blindly from the profile. The controller
    must preclaim both trace allowances, retain the memory and physical device
    leases through synchronization and disposal, and never attest a quarantined
    owner. These are cooperative process-level guarantees.
    """

    @property
    def quarantined(self):
        # The pending latch is itself unsafe, even if Python is interrupted
        # before an exception handler can set the ordinary quarantine latch.
        return bool(getattr(self, '_quarantine_latched', False)
                    or getattr(self, '_device_drain_uncertain', False))

    @quarantined.setter
    def quarantined(self, value):
        if not value and getattr(self, '_quarantine_latched', False):
            raise StudyFailure('CUDA quarantine cannot be cleared in this owner')
        self._quarantine_latched = bool(value)

    def __init__(self, model, journal, *, session, device_lease, memory_lease, memory_probe,
                 profile, warmup_profile, expected_key, expected_quartets=4, cancel_event=None):
        # Reject invalid policy/ownership before any CUDA call or allocation.
        _validate_profile(profile, expected_key)
        _validate_warmup_profile(warmup_profile, expected_key)
        self._initialize_owner(model, journal, session=session, device_lease=device_lease,
            memory_lease=memory_lease, memory_probe=memory_probe, expected_key=expected_key,
            required_incremental_bytes=max(profile.estimated_incremental_bytes,
                                           warmup_profile.estimated_incremental_bytes),
            profile=profile, warmup_profile=warmup_profile,
            expected_quartets=expected_quartets, cancel_event=cancel_event)

    def _initialize_owner(self, model, journal, *, session, device_lease, memory_lease, memory_probe,
                          expected_key, required_incremental_bytes, profile, warmup_profile,
                          expected_quartets, cancel_event):
        # Shared with the narrowly scoped provisional calibration owner. The
        # caller has already validated the exact measured or bootstrap policy.
        if (type(session) is not OwnerSession or type(device_lease) is not DeviceLease
                or type(memory_lease) is not MemoryLease or type(memory_probe) is not GuardedAgreementMemoryProbe):
            raise ValueError('exact owner, device and memory leases are required')
        OwnerSession.check(session, device_lease)
        if (type(session.identity) is not DeviceIdentity or not session._runtime_verified
                or session.identity.gpu_uuid != expected_key.device_uuid):
            raise ValueError('supervised physical runtime identity is not verified')
        _check_memory_lease(memory_lease, expected_key)
        telemetry = memory_lease._budget._telemetry
        if (memory_probe._cuda is not torch.cuda or memory_probe.device != 'cuda:0'
                or memory_probe.device_uuid != expected_key.device_uuid
                or memory_probe._owner_session is not session or memory_probe._device_lease is not device_lease
                or getattr(telemetry, '__self__', None) is not memory_probe
                or getattr(telemetry, '__func__', None) is not GuardedAgreementMemoryProbe.snapshot):
            raise ValueError('budget telemetry must use the matching CUDA memory probe')
        if (type(required_incremental_bytes) is not int or required_incremental_bytes <= 0
                or memory_lease.estimated_incremental_bytes < required_incremental_bytes):
            raise ValueError('memory lease does not cover the required incremental commitment')
        if (memory_lease.trace_bytes('gpu') < TRACE_BYTES
                or memory_lease.trace_bytes('host') < TRACE_BYTES):
            raise ValueError('both trace allowances must be claimed before GPU work')
        super().__init__(model, journal, device='cpu', expected_quartets=expected_quartets,
                         cancel_event=cancel_event)
        self._forward_roots = {name: None for name in
                               ('ids', 'mask', 'inputs', 'sources', 'source_fallback',
                                'steps', 'plan', 'plan_sources', 'plan_captures',
                                'output', 'captures', 'offload')}
        self._completed = False
        self._warmup_completed = False
        self._warmup_state_sha256 = None
        self._warmup_state_identity = None
        self._warmup_execution_contract = None
        try:
            with _owner_gate_lock:
                if device_lease in _owner_gates:
                    raise ValueError('this physical owner already has an agreement runner')
                if memory_probe._bound_runner is not None:
                    raise ValueError('the memory probe is already bound to an owner')
                _owner_gates[device_lease] = (memory_lease, self)
            self.device = 'cuda:0'
            self._owner_session, self._device_lease = session, device_lease
            self._memory_lease, self._profile, self._warmup_profile = memory_lease, profile, warmup_profile
            self._memory_probe = memory_probe
            self._required_incremental_bytes = required_incremental_bytes
            self._expected_key = expected_key
            self._device_rng_before = None
            self._owner_thread = threading.current_thread()
            self._device_drain_uncertain = False
            memory_probe._bound_runner = self
            self._check_owner()
            # Recheck the real CUDA-reported binding, not only the launcher's claim.
            if torch.cuda.is_initialized() is not True:
                raise ValueError('CUDA must already be initialized under this verified owner')
            self._check_owner()
            count = torch.cuda.device_count()
            self._check_owner()
            properties = torch.cuda.get_device_properties(0)
            runtime_uuid = _runtime_uuid(torch, properties.uuid)
            OwnerSession.verify_runtime_binding(session, device_lease, runtime_uuid=runtime_uuid,
                                                device_count=count, cuda_device_index=0)
            if (properties.total_memory != expected_key.device_total_bytes
                    or properties.name != expected_key.device_name
                    or (properties.major, properties.minor) != expected_key.compute_capability):
                raise ValueError('runtime device properties differ from the measured profile')
            self._check_owner()
            if torch.cuda.get_allocator_backend() != 'native':
                raise ValueError('the fixed native PyTorch allocator backend is required')
            self._check_owner()
            expected_fraction = int(expected_key.device_total_bytes * 0.25) / expected_key.device_total_bytes
            if torch.cuda.get_per_process_memory_fraction(0) != expected_fraction:
                raise ValueError('the fixed 25% PyTorch allocator cap is not active')
            self._check_headroom()
        except BaseException:
            self.quarantined = True
            raise

    def _check_owner(self, *, allow_quarantined=False, allow_pending=False):
        if getattr(self, '_quarantine_latched', False) and not allow_quarantined:
            raise StudyFailure('agreement GPU owner is quarantined')
        if self._device_drain_uncertain and not allow_pending:
            self.quarantined = True
            raise StudyFailure('a CUDA drain is unverified; process replacement is required')
        try:
            self._cancel_check()
        except BaseException:
            self._device_drain_uncertain = True
            self.quarantined = True
            raise
        if threading.current_thread() is not self._owner_thread:
            raise StudyFailure('agreement GPU owner is bound to its creating thread')
        OwnerSession.check(self._owner_session, self._device_lease)
        if (type(self._owner_session.identity) is not DeviceIdentity
                or self._owner_session.identity.gpu_uuid != self._expected_key.device_uuid
                or self._owner_session._runtime_verified is not True):
            raise StudyFailure('supervised runtime binding is no longer valid')
        DeviceLease.check(self._device_lease)
        _check_memory_lease(self._memory_lease, self._expected_key)
        if (self._memory_lease.estimated_incremental_bytes < self._required_incremental_bytes
                or self._memory_lease.trace_bytes('gpu') < TRACE_BYTES
                or self._memory_lease.trace_bytes('host') < TRACE_BYTES):
            raise StudyFailure('agreement GPU commitments are no longer held')

    def _check_headroom(self):
        self._check_owner()
        backend = torch.cuda.get_allocator_backend()
        self._check_owner()
        fraction = torch.cuda.get_per_process_memory_fraction(0)
        self._check_owner()
        expected_fraction = (int(self._expected_key.device_total_bytes * 0.25)
                             / self._expected_key.device_total_bytes)
        if backend != 'native' or fraction != expected_fraction:
            raise StudyFailure('the native allocator or fixed process fraction changed')
        try:
            snapshot = self._memory_lease._budget.snapshot()
        except BaseException:
            self.quarantined = True
            self._device_drain_uncertain = True
            raise
        self._check_owner()
        if snapshot.telemetry.device_total_bytes != self._expected_key.device_total_bytes:
            raise StudyFailure('device memory total changed')
        _check_headroom(snapshot, snapshot.telemetry)
        if snapshot.telemetry.device_free_bytes < 3 * GIB:
            raise StudyFailure('device-wide free memory fell below the fixed reserve')

    def _sync(self, *, allow_quarantined=False):
        self._check_owner(allow_quarantined=allow_quarantined)
        if self._device_drain_uncertain:
            raise StudyFailure('a previous CUDA drain is unverified; process replacement is required')
        self._device_drain_uncertain = True
        try:
            torch.cuda.synchronize(0)
            self._check_owner(allow_quarantined=allow_quarantined, allow_pending=True)
        except BaseException:
            self.quarantined = True
            raise
        self._device_drain_uncertain = False

    def _execution_contract(self):
        self._check_owner()
        _check_cpu_precision()
        if torch.is_autocast_enabled('cuda'):
            raise StudyFailure('CUDA autocast is unsupported')
        if (not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32):
            raise StudyFailure('deterministic CUDA float32 settings are required')
        model = self.model
        if (not isinstance(model, torch.nn.Module) or any(module.training is not False for module in model.modules())
                or len(model.transformer.h) != 6 or model.config.use_cache is not False
                or model.config._attn_implementation != 'eager'
                or model.config.n_embd != 768 or model.config.vocab_size != 50257):
            raise StudyFailure('expected pinned six-block eager evaluation configuration')
        if any(str(p.device) != self.device or p.dtype != torch.float32 for p in model.parameters()):
            raise StudyFailure('model parameters must be float32 on cuda:0')
        if any(str(b.device) != self.device or (b.is_floating_point() and b.dtype != torch.float32)
               for b in model.buffers()):
            raise StudyFailure('model buffers must be on cuda:0 with float32 floating buffers')
        if torch.__version__ != self._expected_key.torch_version or torch.version.cuda != self._expected_key.cuda_runtime_version:
            raise StudyFailure('PyTorch or CUDA runtime differs from the profile')
        cpu_settings, modules = _runtime_configuration(model)
        gpu_settings = (('cuda_matmul_allow_tf32', torch.backends.cuda.matmul.allow_tf32),
                        ('cudnn_allow_tf32', torch.backends.cudnn.allow_tf32),
                        ('cudnn_enabled', torch.backends.cudnn.enabled),
                        ('cudnn_benchmark', torch.backends.cudnn.benchmark),
                        ('cudnn_deterministic', torch.backends.cudnn.deterministic))
        contract = (model.config.n_embd, model.config.vocab_size,
                    (cpu_settings + gpu_settings, modules))
        if self._warmup_completed and contract != self._warmup_execution_contract:
            raise StudyFailure('model execution configuration changed since same-owner warmup')
        return contract

    def _model_digest(self):
        try:
            self._check_headroom()
            self._sync()
            digest = hashlib.sha256()
            for group, entries in (('state_dict', self.model.state_dict().items()),
                                   ('all_buffers', self.model.named_buffers(remove_duplicate=False))):
                for name, tensor in entries:
                    if str(tensor.device) != self.device or not tensor.is_contiguous():
                        raise StudyFailure('model state must be contiguous on cuda:0')
                    header = json.dumps({'group': group, 'name': name, 'shape': list(tensor.shape),
                                         'dtype': str(tensor.dtype)}, sort_keys=True,
                                        separators=(',', ':')).encode('utf-8')
                    digest.update(len(header).to_bytes(4, 'big'))
                    digest.update(header)
                    self._check_owner()
                    detached = tensor.detach()
                    self._check_owner()
                    flat = detached.reshape(-1)
                    self._check_owner()
                    raw = flat.view(torch.uint8)
                    self._check_owner()
                    for offset in range(0, raw.numel(), 1024 * 1024):
                        self._check_owner()
                        chunk = raw[offset:offset + 1024 * 1024]
                        self._check_owner()
                        host = chunk.to(device='cpu', copy=True)
                        self._check_owner()
                        digest.update(memoryview(host.numpy()))
                        del host
            self._sync()
            value = digest.hexdigest()
            if (self._warmup_completed and
                    (value != self._warmup_state_sha256
                     or not _same_state_identity(self._warmup_state_identity, self.model))):
                raise StudyFailure('model state or identity changed since same-owner warmup')
            return value
        except BaseException:
            self.quarantined = True
            raise

    def _encode_payload(self, tensor):
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or str(tensor.device) != self.device or sys.byteorder != 'little'):
            raise StudyFailure('expected little-endian float32 output on cuda:0')
        self._sync()
        temporary = None
        try:
            self._check_owner()
            detached = tensor.detach()
            self._check_owner()
            integer_view = detached.view(torch.int32)
            self._check_owner()
            temporary = integer_view.contiguous()
            self._forward_roots['offload'] = temporary
            self._check_owner()
            host = temporary.to(device='cpu', copy=True)
            self._check_owner()
            raw = host.numpy().tobytes()
            decode_f32_le(raw)
            self._forward_roots['offload'] = None
            return raw
        except BaseException:
            if temporary is not None:
                self._forward_roots['offload'] = temporary
            raise

    def _make_inputs(self, case):
        if len(case.input_ids) != 5:
            raise StudyFailure('the measured agreement profile covers exactly five tokens')
        self._check_owner()
        ids, mask = None, None
        try:
            ids = torch.tensor([case.input_ids], dtype=torch.int64, device='cpu').to(self.device, copy=True)
            self._forward_roots['ids'] = ids
            self._check_owner()
            mask = torch.ones_like(ids)
            self._forward_roots['mask'] = mask
            self._check_owner()
            return ids, mask
        except BaseException:
            if ids is not None:
                self._forward_roots['ids'] = ids
            if mask is not None:
                self._forward_roots['mask'] = mask
            raise

    def _prepare_steps(self, steps):
        if steps is None:
            return None
        prepared = []
        self._forward_roots['sources'] = prepared
        device_source = None
        try:
            for step in steps:
                item = dict(step)
                # Publish the container before creating a CUDA source so all
                # completed assignments remain strongly rooted on failure.
                prepared.append(item)
                if 'value' in item:
                    source = item['value']
                    if (not isinstance(source, torch.Tensor) or source.dtype != torch.float32
                            or str(source.device) != 'cpu' or tuple(source.shape) != (768,)):
                        raise StudyFailure('cross-input source must be a 768-element CPU float32 vector')
                    self._check_owner()
                    device_source = source.to(device=self.device, copy=True)
                    item['value'] = device_source
                    self._check_owner()
                    device_source = None
            return prepared
        except BaseException:
            if device_source is not None:
                self._forward_roots['source_fallback'] = device_source
            raise

    def _new_plan(self, steps):
        # Root the object before its initializer can clone a CUDA source.
        plan = None
        try:
            self._check_owner()
            plan = ActivationPlan.__new__(ActivationPlan)
            self._forward_roots['plan'] = plan
            ActivationPlan.__init__(plan, self.model, steps, max_tensor_bytes=1024 * 1024,
                                    max_capture_bytes=512 * 1024, max_source_bytes=128 * 1024,
                                    max_steps=16, max_invocations=1, _parent_lease=self._lease)
            # ActivationPlan clears its source dictionary on exit; retain the
            # exact clones independently until device completion is verified.
            self._forward_roots['plan_sources'] = tuple(plan._sources.values())
            # Allocate the retaining map before entering the plan, so failure
            # cleanup cannot discard already-assigned captures and no copy is
            # needed at the potentially out-of-memory exit boundary.
            plan._captures = _RetainedPlanCaptures()
            self._forward_roots['plan_captures'] = plan._captures
            self._check_owner()
            return plan
        except BaseException:
            if plan is not None:
                self._forward_roots['plan'] = plan
            raise

    def _root_forward(self, name, value):
        if name not in self._forward_roots:
            raise StudyFailure('unexpected CUDA custody slot')
        self._forward_roots[name] = value

    def _finish_forward_roots(self):
        # Called only after a successful synchronization and offload. On every
        # error, retain all known slots through process replacement instead.
        for name in self._forward_roots:
            self._forward_roots[name] = None

    def _before_forward(self):
        self._check_headroom()

    def _after_forward(self):
        self._sync()
        self._check_headroom()

    def _failed_forward(self, error):
        # Frames from a failed plan may retain CUDA tensors; keep all claims and
        # the physical lease until process replacement, even after a good sync.
        self.quarantined = True
        if not self._device_drain_uncertain:
            self._sync(allow_quarantined=True)

    def _record_rejected(self, case, output, captures, metadata, primary_error):
        # Do not even index GPU tensors after a failed or interrupted device
        # operation. Completed prior records remain intact; the controller must
        # treat this forward's device-side observations as unavailable.
        try:
            self.journal.record({'kind': 'rejected_forward', 'case': case.case_id,
                                 'error_type': type(primary_error).__name__,
                                 'unavailable': ['CUDA observations unavailable; known rooted buffers retained until process exit']}, {})
            self._forwards_recorded += 1
        except BaseException as error:
            self._failure_evidence_error = type(error).__name__
            if hasattr(primary_error, 'add_note'):
                primary_error.add_note('Rejected-forward evidence could not be saved: ' + type(error).__name__)

    def _seed_device_rng(self):
        try:
            self._check_headroom()
            # Treat an interrupted RNG call as uncertain before making it, so
            # cleanup cannot restore or inspect device state after a failure.
            self._device_drain_uncertain = True
            self._device_rng_before = torch.cuda.get_rng_state(0)
            self._check_owner(allow_pending=True)
            torch.cuda.default_generators[0].manual_seed(_RNG_SEED)
            self._check_owner(allow_pending=True)
            state = torch.cuda.get_rng_state(0)
            self._check_owner(allow_pending=True)
            result = {'seed': _RNG_SEED, 'seeded_state_sha256': hashlib.sha256(state.numpy().tobytes()).hexdigest()}
            self._device_drain_uncertain = False
            return result
        except BaseException:
            self._device_drain_uncertain = True
            self.quarantined = True
            raise

    def _before_rng_cleanup(self, primary_error):
        if primary_error is not None:
            # Any failed callback leaves completion uncertain. Retain model
            # and device custody without performing RNG restoration on CUDA.
            self._device_drain_uncertain = True
            self.quarantined = True

    def _restore_device_rng(self):
        if self._device_rng_before is not None:
            try:
                self._check_headroom()
                # Keep the unsafe latch set across both device calls and the
                # result check, including interruptions in caller cleanup.
                self._device_drain_uncertain = True
                torch.cuda.set_rng_state(self._device_rng_before, 0)
                self._check_owner(allow_pending=True)
                current = torch.cuda.get_rng_state(0)
                self._check_owner(allow_pending=True)
                if not torch.equal(current, self._device_rng_before):
                    raise StudyFailure('CUDA RNG state restoration failed')
                self._device_drain_uncertain = False
            except BaseException:
                self._device_drain_uncertain = True
                self.quarantined = True
                raise

    def _runtime_metadata(self):
        return {'device': self.device, 'device_uuid': self._expected_key.device_uuid,
                'workload_sha256': self._expected_key.fingerprint(),
                'profile_id': self._profile.profile_id,
                'warmup_profile_id': self._warmup_profile.profile_id,
                'capture_policy_sha256': CAPTURE_POLICY_SHA256,
                'warmup_policy_sha256': WARMUP_POLICY_SHA256,
                'warmup_model_state_sha256': self._warmup_state_sha256}

    def warmup(self):
        """Execute and discard the fixed same-owner clean-capture warmup."""
        if self._warmup_completed or self._used or self.quarantined:
            raise StudyFailure('warmup is single-use and precedes the study')
        lease = ModelLease(self.model)
        self._lease = lease
        initial_state = None
        rng_before = None
        try:
            try:
                lease.acquire()
                self._cancel_check()
                self._baseline_contract = self._execution_contract()
                self._baseline_hooks = _hook_state(self.model)
                if (any(any(registry for registry in module[1:]) for module in self._baseline_hooks[0])
                        or any(self._baseline_hooks[1:])):
                    raise StudyFailure('preexisting local or global hooks are unsupported')
                initial_state = self._model_digest()
                self._baseline_state_identity = _state_identity(self.model)
                rng_before = torch.random.get_rng_state()
                torch.random.default_generator.manual_seed(_RNG_SEED)
                self._seed_device_rng()
                case = AgreementCase('gpu-warmup-cat', 'gpu-warmup', 'cat', 'singular',
                                     CALIBRATION_INPUT_IDS, 1)
                logits, captures, metadata = self._forward(case, _steps_for_capture(case))
                # Do not persist behavioral values in the resource warmup.
                self._pending_forward = None
                del logits, captures
                self._sync()
                if not _same_state_identity(self._baseline_state_identity, self.model):
                    raise StudyFailure('model identity changed during warmup')
                warmup_record = {'sample_id': 'warmup', 'policy_sha256': WARMUP_POLICY_SHA256, 'seed': _RNG_SEED,
                                 'model_state_sha256': initial_state, 'plan_metadata': metadata}
                self.journal.record({'kind': 'gpu_warmup', **warmup_record}, {})
            finally:
                primary_error = sys.exc_info()[1]
                self._before_rng_cleanup(primary_error)
                try:
                    if rng_before is not None:
                        try:
                            try:
                                self._restore_device_rng()
                            finally:
                                torch.random.set_rng_state(rng_before)
                            if not torch.equal(torch.random.get_rng_state(), rng_before):
                                raise StudyFailure('CPU RNG state restoration failed')
                        except BaseException as error:
                            self.quarantined = True
                            if primary_error is not None:
                                if hasattr(primary_error, 'add_note'):
                                    primary_error.add_note('Warmup RNG cleanup failed: ' + type(error).__name__)
                            else:
                                raise
                finally:
                    active_error = sys.exc_info()[1] or primary_error
                    try:
                        self._release_after_work(lease, initial_state, active_error)
                    except BaseException as error:
                        self.quarantined = True
                        try:
                            lease._quarantine()
                        except BaseException:
                            pass
                        if active_error is not None:
                            if hasattr(active_error, 'add_note'):
                                active_error.add_note('Warmup lease cleanup failed: ' + type(error).__name__)
                        else:
                            raise
            # Warmup is documented separately; study forward counts remain exact.
            self._forwards = self._forwards_started = self._model_returns = self._forwards_recorded = 0
            self._started_work = False
            self._warmup_state_sha256 = initial_state
            self._warmup_state_identity = self._baseline_state_identity
            self._warmup_execution_contract = self._baseline_contract
            self._warmup_completed = True
            return warmup_record
        except BaseException as error:
            self.quarantined = True
            try:
                self.journal.finish('failed', {'error_type': type(error).__name__,
                                               'message': _error_message(error), 'stage': 'gpu warmup'})
            except BaseException:
                pass
            raise

    def run(self, cases, *, mode='discovery', frozen_predictions=None, heldout_binding=None):
        try:
            # The committed held-out adapter accepts only the CPU runner. A
            # direct low-level call cannot establish case-to-commitment binding.
            if mode != 'discovery' or frozen_predictions is not None or heldout_binding is not None:
                raise StudyFailure('CUDA held-out execution is not yet supported')
            if not self._warmup_completed:
                raise StudyFailure('a verified same-owner warmup is required before the study')
            _validate_cases(cases, self.expected_quartets, 50257)
            if any(len(case.input_ids) != 5 for case in cases):
                raise StudyFailure('all study cases must match the measured five-token profile')
            result = super().run(cases, mode=mode, frozen_predictions=frozen_predictions,
                                 heldout_binding=heldout_binding)
            self._completed = True
            return result
        except BaseException as error:
            # The owner must keep all commitments and decline attestation; the
            # exception traceback can still retain CUDA allocations.
            self.quarantined = True
            if not self._used:
                try:
                    self.journal.finish('failed', {'error_type': type(error).__name__,
                                                    'message': _error_message(error),
                                                    'stage': 'pre-forward GPU validation'})
                except BaseException:
                    pass
            raise

    def dispose_after_success(self):
        """Drop known runner roots after a successful, synchronized study.

        The owner still must release all other model references, synchronize,
        verify allocator/claim cleanup and attest. The process-lifetime gate
        remains occupied. This method is forbidden after any uncertain result.
        """
        if self.quarantined or not self._completed or self._lease is not None:
            raise StudyFailure('only a completed clean owner can dispose runner roots')
        self._sync()
        if any(value is not None for value in self._forward_roots.values()):
            self.quarantined = True
            raise StudyFailure('forward custody remains outstanding')
        with _owner_gate_lock:
            owner = _owner_gates.get(self._device_lease)
            if owner != (self._memory_lease, self):
                self.quarantined = True
                raise StudyFailure('CUDA owner gate no longer matches')
            self._baseline_state_identity = None
            self._warmup_state_identity = None
            self._warmup_execution_contract = None
            self.model = None
            _owner_gates[self._device_lease] = (self._memory_lease, None)
