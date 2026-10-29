"""Disabled preparation for the fixed five-token supervised GPU trace.

This module does not initialize CUDA or launch an owner.  It defines an exact
resource-key contract for future independently measured phase profiles.  A
trusted live builder and real supervised owner are still required.
"""
from dataclasses import dataclass
import re
from threading import Event, Lock

from .memory_budget import MemoryBudget
from .pretrained import ASSETS, REVISION
from .resource_profile import ProfileRegistry, WorkloadKey, canonical_fingerprint


PROTOCOL = "pretrained-gpu-trace-resource-v1"
TRACE_OWNER_ENABLED = False
TOKENS = (464, 6766, 318, 4171, 13)
LAYERS = (0, 2, 5)
TARGETS = ("attn", "mlp", "block")
POSITIONS = (0, 1)
INPUT_DTYPES = (("attention_mask", "torch.int64"), ("input_ids", "torch.int64"))
OUTPUT_BYTES = 1 * 5 * 50257 * 4
SESSION_TRACE_CAP = 131072
MODEL_LOGICAL_BYTES = 81912576 * 4
RESIDENT_COMMITMENT = 512 * 1024 ** 2
JOB_COMMITMENT = 1024 ** 3
ALLOCATOR_CEILING = RESIDENT_COMMITMENT + JOB_COMMITMENT
PROFILE_MARGIN_BYTES = 32 * 1024 ** 2
PROFILE_MARGIN_BPS = 1000
_QUARANTINED = Event()
_QUARANTINE_HOLDS = []
_LIFETIME_HOLDS = []
_OWNER_LOCK = Lock()
_OWNER_USED = False


@dataclass(frozen=True)
class Phase:
    """One ordered resource-measurement phase, not an observed result."""
    name: str
    operation: str
    capture_mode: str
    warmup_state: str
    forwards: int
    gpu_trace_allowance: int
    host_trace_allowance: int


PHASES = (
    Phase("load", "model_load", "none", "cold", 0, 0, 0),
    Phase("baseline_cold", "forward", "none", "cold", 1, 0, 0),
    Phase("baseline_warm", "forward", "none", "warm", 1, 0, 0),
    Phase("reference", "forward", "direct", "warm", 2, SESSION_TRACE_CAP, SESSION_TRACE_CAP),
    Phase("direct", "forward", "direct", "warm", 2, SESSION_TRACE_CAP, 2 * SESSION_TRACE_CAP),
    Phase("gradient", "forward", "gradient", "warm", 2, SESSION_TRACE_CAP, 3 * SESSION_TRACE_CAP),
    Phase("id", "forward", "id", "warm", 2, 0, 3 * SESSION_TRACE_CAP),
)


def protocol_fingerprint():
    """Fingerprint fixed inputs, capture scope, ordering and comparison rules."""
    return canonical_fingerprint({
        "protocol": PROTOCOL, "tokens": TOKENS, "attention_mask": (1, 1, 1, 1, 1),
        "seed": 0, "layers": LAYERS, "targets": TARGETS, "positions": POSITIONS,
        "forward_count": tuple(phase.forwards for phase in PHASES),
        "phase_order": tuple(phase.name for phase in PHASES),
        "output_limit_bytes": OUTPUT_BYTES, "session_trace_cap": SESSION_TRACE_CAP,
        "logit_and_trace_offload": "cpu", "float32_comparison": "bitwise-int32-view",
        "tf32": "disabled", "runtime_flags": "record-and-match",
        "profile_margin_bytes": PROFILE_MARGIN_BYTES, "profile_margin_bps": PROFILE_MARGIN_BPS,
    })


def phase_policy_fingerprint(index):
    if type(index) is not int or not 0 <= index < len(PHASES):
        raise ValueError("invalid fixed phase index")
    phase = PHASES[index]
    retained = {
        "load": (), "baseline_cold": (), "baseline_warm": (), "reference": (),
        "direct": ("reference",), "gradient": ("reference", "direct"),
        "id": ("reference", "direct", "gradient"),
    }
    return canonical_fingerprint({
        "protocol_sha256": protocol_fingerprint(), "phase": phase.name,
        "preceding_phases": tuple(item.name for item in PHASES[:index]),
        "forwards": phase.forwards, "capture_mode": phase.capture_mode,
        "retained_value_sets_on_entry": retained[phase.name],
        "gpu_trace_allowance": phase.gpu_trace_allowance,
        "host_trace_allowance": phase.host_trace_allowance,
    })


def phase_case_fingerprint(index):
    """Bind each sample's seed and fixed phase inputs to the protocol."""
    if type(index) is not int or not 0 <= index < len(PHASES):
        raise ValueError("invalid fixed phase index")
    return canonical_fingerprint({
        "policy_sha256": phase_policy_fingerprint(index), "seed": 0,
        "input_ids": None if index == 0 else TOKENS,
        "attention_mask": None if index == 0 else (1, 1, 1, 1, 1),
    })


def validate_trace_profiles(registry, *, physical_uuid, verified_model_runtime_id, keys):
    """Resolve seven exact, independently trusted profiles for this workload.

    ``keys`` must be reconstructed by a trusted owner from verified live
    source, model, runtime and device state; profile declarations or clients
    are not authoritative. The independently supplied model/runtime ID must
    be derived from actual verified state, not copied from keys or profiles.
    This comparison does not authenticate a caller's assertion. The production
    live builder is not implemented here. Resolution and size estimates do not
    constitute a safe peak guarantee.
    """
    if type(registry) is not ProfileRegistry or type(keys) is not tuple or len(keys) != len(PHASES):
        raise TypeError("an exact registry and seven immutable live keys are required")
    if type(physical_uuid) is not str:
        raise TypeError("a verified physical device UUID is required")
    if (type(verified_model_runtime_id) is not str
            or re.fullmatch(r"[0-9a-f]{64}", verified_model_runtime_id) is None):
        raise ValueError("a verified model/runtime identity digest is required")
    if any(type(key) is not WorkloadKey for key in keys):
        raise TypeError("each live key must be a WorkloadKey")
    for index, (phase, key) in enumerate(zip(PHASES, keys)):
        if (key.model_id != verified_model_runtime_id or key.device != "cuda:0" or key.device_uuid != physical_uuid
                or key.checkpoint_sha256 != ASSETS["model.safetensors"][1]
                or key.checkpoint_revision != REVISION
                or key.model_config_sha256 != ASSETS["config.json"][1]
                or key.model_dtype != "torch.float32" or key.compute_dtype != "torch.float32"
                or key.attention_backend != "eager" or key.model_mode != "eval"
                or key.grad_mode != "inference" or key.autocast != "disabled"
                or key.use_cache is not False or key.compiled is not False
                or key.warmup_protocol_sha256 != protocol_fingerprint()
                or key.operation != phase.operation or key.capture_mode != phase.capture_mode
                or key.warmup_state != phase.warmup_state
                or key.capture_policy_sha256 != phase_policy_fingerprint(index)
                or key.max_gpu_trace_bytes != phase.gpu_trace_allowance
                or key.max_host_trace_bytes != phase.host_trace_allowance):
            raise ValueError(f"live resource key does not match fixed phase {phase.name}")
        if phase.operation == "model_load":
            if (key.batch_size, key.sequence_length, key.input_dtypes, key.output_limit_bytes) != (0, 0, (), 0):
                raise ValueError("model-load resource key has request fields")
        elif (key.batch_size, key.sequence_length, key.input_dtypes, key.output_limit_bytes) != (
                1, 5, INPUT_DTYPES, OUTPUT_BYTES):
            raise ValueError(f"live resource key has the wrong inputs or output cap for {phase.name}")
    profiles = tuple(registry.resolve(key) for key in keys)
    for index, profile in enumerate(profiles):
        if (profile.margin_bytes != PROFILE_MARGIN_BYTES
                or profile.margin_bps != PROFILE_MARGIN_BPS):
            raise ValueError("profile margin does not match the fixed trace protocol")
        if any(sample.seed != 0 or sample.case_sha256 != phase_case_fingerprint(index)
               for sample in profile.samples):
            raise ValueError(f"sample case or seed does not match fixed phase {PHASES[index].name}")
    if profiles[0].estimated_incremental_bytes > RESIDENT_COMMITMENT:
        raise ValueError("measured load estimate exceeds the proposed resident commitment")
    if any(profile.estimated_incremental_bytes > JOB_COMMITMENT for profile in profiles[1:]):
        raise ValueError("measured phase estimate exceeds the proposed job commitment")
    return profiles


def launch_supervised_trace(*_args, **_kwargs):
    """Refuse production launch until the reviewed owner and profiles exist."""
    raise RuntimeError("five-token supervised GPU trace owner is not ready")


def _quarantine(*objects):
    _QUARANTINED.set()
    try:
        _QUARANTINE_HOLDS.append(objects)
    except BaseException:
        # Even if reference retention fails during OOM, never attest success.
        pass


def _run_guarded_owner(session, device_lease, *, backend, registry):
    """Disabled lifecycle seam; tests enable it only with CPU fakes.

    This function cannot establish that injected callbacks are honest or
    CPU-only, and no production backend or launcher is supplied. It must not
    be enabled for a real owner without the independently reviewed live
    builder, backend, profile evidence, and supervised launch prerequisites.
    """
    if not TRACE_OWNER_ENABLED:
        raise RuntimeError("five-token supervised GPU trace owner is not ready")
    global _OWNER_USED
    with _OWNER_LOCK:
        if _QUARANTINED.is_set():
            raise RuntimeError("trace owner process is quarantined")
        if _OWNER_USED:
            raise RuntimeError("trace owner process is already active or used")
        _OWNER_USED = True
    try:
        return _run_guarded_owner_inner(session, device_lease, backend=backend, registry=registry)
    except BaseException as error:
        _quarantine(session, device_lease, backend, registry, error)
        raise


def _run_guarded_owner_inner(session, device_lease, *, backend, registry):
    if not TRACE_OWNER_ENABLED:
        raise RuntimeError("five-token supervised GPU trace owner is not ready")
    session.check(device_lease)
    if session.identity is None:
        raise RuntimeError("identity-aware supervision is required")

    def verify_binding():
        session.check(device_lease)
        runtime_uuid, count = backend.runtime_binding()
        session.verify_runtime_binding(device_lease, runtime_uuid=runtime_uuid,
                                       device_count=count, cuda_device_index=0)

    verify_binding()
    live_model_runtime_id = backend.verified_model_runtime_id(session.identity)
    verify_binding()
    keys = backend.live_keys(session.identity)
    profiles = validate_trace_profiles(registry, physical_uuid=session.identity.gpu_uuid,
                                       verified_model_runtime_id=live_model_runtime_id, keys=keys)
    verify_binding()
    backend.synchronize()

    def telemetry():
        verify_binding()
        return backend.telemetry()

    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=SESSION_TRACE_CAP,
                          max_host_trace_bytes=3 * SESSION_TRACE_CAP)
    initial = budget.snapshot().telemetry
    resident = budget.acquire(RESIDENT_COMMITMENT, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    job = budget.acquire(JOB_COMMITMENT, max_gpu_trace_bytes=SESSION_TRACE_CAP,
                         max_host_trace_bytes=3 * SESSION_TRACE_CAP)
    job.reserve_trace("gpu", SESSION_TRACE_CAP)
    job.reserve_trace("host", 3 * SESSION_TRACE_CAP)

    def check_headroom():
        snapshot = budget.snapshot()
        memory = snapshot.telemetry
        if (memory.device_total_bytes != initial.device_total_bytes
                or memory.device_free_bytes < snapshot.reserve_bytes + snapshot.committed_incremental_bytes
                or memory.device_total_bytes < memory.process_reserved_bytes + snapshot.reserve_bytes
                + snapshot.committed_incremental_bytes):
            raise RuntimeError("trace owner memory headroom or device total changed")
        return memory

    check_headroom()
    verify_binding()
    if backend.configure_allocator(ALLOCATOR_CEILING, initial.device_total_bytes) is not True:
        raise RuntimeError("allocator ceiling was not confirmed")
    verify_binding()
    if backend.allocator_ceiling_bytes(initial.device_total_bytes) != ALLOCATOR_CEILING:
        raise RuntimeError("allocator ceiling readback differs from commitment")
    if backend.forward_hooks_clear() is not True:
        raise RuntimeError("unexpected local or global forward hooks")
    for phase, profile in zip(PHASES, profiles):
        verify_binding()
        if backend.forward_hooks_clear() is not True:
            raise RuntimeError("unexpected local or global forward hooks")
        verify_binding()
        backend.synchronize()
        before = check_headroom()
        if backend.begin_measurement(phase, before) is not True:
            raise RuntimeError("phase measurement did not start")
        verify_binding()
        check_headroom()
        if backend.allocator_ceiling_bytes(initial.device_total_bytes) != ALLOCATOR_CEILING:
            raise RuntimeError("allocator ceiling changed during phase setup")
        if backend.forward_hooks_clear() is not True:
            raise RuntimeError("forward hooks appeared during phase setup")
        verify_binding()
        if backend.run_phase(phase, profile, job) is not True:
            raise RuntimeError("phase did not complete its fixed checks")
        verify_binding()
        backend.synchronize()
        if backend.forward_hooks_clear() is not True:
            raise RuntimeError("forward hooks remained after a phase")
        after = check_headroom()
        if backend.end_measurement(phase, before, after) is not True:
            raise RuntimeError("phase measurement did not finish")
        verify_binding()
    if backend.dispose() is not True:
        raise RuntimeError("trace owner storage disposal was not confirmed")
    verify_binding()
    backend.synchronize()
    if backend.verify_drained() is not True or backend.quarantined is not False:
        raise RuntimeError("trace owner drain is uncertain")
    verify_binding()
    final = check_headroom()
    if (final.process_allocated_bytes > initial.process_allocated_bytes
            or final.process_reserved_bytes > initial.process_reserved_bytes):
        raise RuntimeError("trace owner retained allocator memory")
    job.release_trace("gpu", SESSION_TRACE_CAP)
    job.release_trace("host", 3 * SESSION_TRACE_CAP)
    job.release()
    resident.release()
    if budget.active_leases():
        raise RuntimeError("trace owner retains memory commitments")
    # Keep the real device lease and all associated state alive until exit;
    # parent reaping is a separate supervisor responsibility.
    _LIFETIME_HOLDS.append((session, device_lease, backend, registry, budget))
    session.attest_clean(device_lease)
    return True
