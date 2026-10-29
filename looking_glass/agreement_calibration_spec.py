"""Logical payload sizes for the proposed five-token agreement calibration.

These numbers are not allocator peaks, VRAM estimates, runtime validation, or
permission to run the currently CPU-only agreement executor on CUDA.
"""

from dataclasses import dataclass


_MODES = frozenset(('baseline', 'clean_capture', 'self_patch', 'cross_patch', 'null'))
_BATCH = 1
_TOKENS = 5
_HIDDEN = 768
_VOCAB = 50257
_F32_BYTES = 4
_I64_BYTES = 8
_CLEAN_SITES = 10
CALIBRATION_SEED = 20261028
TARGET_INPUT_IDS = (464, 3797, 1474, 262, 9168)
DONOR_INPUT_IDS = (464, 11875, 1474, 262, 9168)
SUBJECT_INDEX = 1
LAST_TOKEN_INDEX = 4
MIN_TRACE_ALLOWANCE_BYTES = 2 * 1024 ** 2
MIN_MARGIN_BYTES = 512 * 1024 ** 2
MIN_MARGIN_BPS = 2000
SAMPLE_KINDS = (('clean_capture',)
                + tuple(f'layer{layer}_{component}_cross_patch'
                        for layer in (1, 2, 3, 4) for component in ('attn', 'mlp'))
                + ('null', 'self_patch', 'block_last_cross_patch', 'block_subject_cross_patch'))


@dataclass(frozen=True)
class LogicalPayloads:
    """Selected tensor payloads; different objects can coexist or alias."""
    input_and_mask_bytes: int
    selected_vector_bytes: int
    whole_activation_bytes: int
    full_logits_bytes: int
    plan_capture_bytes: int
    plan_source_clone_bytes: int
    patch_output_clone_bytes: int
    plan_max_tensor_bytes: int
    plan_steps: int


def logical_payloads(mode):
    """Return exact logical sizes for one fixed plan, without measuring memory."""
    if type(mode) is not str or mode not in _MODES:
        raise ValueError('unknown agreement calibration mode')
    vector = _HIDDEN * _F32_BYTES
    whole = _BATCH * _TOKENS * vector
    logits = _BATCH * _TOKENS * _VOCAB * _F32_BYTES
    if mode == 'baseline':
        captures, source, changed, steps = 0, 0, 0, 0
    elif mode == 'clean_capture':
        captures, source, changed, steps = _CLEAN_SITES * vector, 0, 0, _CLEAN_SITES
    else:
        captures = 2 * whole + 2 * vector
        source = vector if mode == 'cross_patch' else 0
        # ActivationPlan clones the complete tensor even for its null action.
        changed, steps = whole, 5
    return LogicalPayloads(
        input_and_mask_bytes=2 * _BATCH * _TOKENS * _I64_BYTES,
        selected_vector_bytes=vector, whole_activation_bytes=whole,
        full_logits_bytes=logits, plan_capture_bytes=captures,
        plan_source_clone_bytes=source, patch_output_clone_bytes=changed,
        plan_max_tensor_bytes=whole if steps else 0, plan_steps=steps)


def validate_plan_limits(mode, *, max_tensor_bytes, max_capture_bytes,
                         max_source_bytes, max_steps, max_invocations,
                         max_output_bytes):
    """Check necessary logical minima, not an aggregate device/host bound.

    Limits mirror the current plan's individual tensor, aggregate capture,
    source, step and invocation checks, plus a separate full-logits limit.
    The eventual adapter must actually enforce these values and account for
    temporaries, storage ownership, allocator overhead and memory pressure.
    """
    payload = logical_payloads(mode)
    given = (max_tensor_bytes, max_capture_bytes, max_source_bytes, max_steps,
             max_invocations, max_output_bytes)
    if any(type(value) is not int or value <= 0 for value in given):
        raise ValueError('limits must be positive integer byte/count values')
    required = (max(1, payload.plan_max_tensor_bytes), max(1, payload.plan_capture_bytes),
                max(1, payload.plan_source_clone_bytes), max(1, payload.plan_steps),
                1, payload.full_logits_bytes)
    if any(value < minimum for value, minimum in zip(given, required)):
        raise ValueError('a plan or output limit is below its logical payload')


def validate_profile_requirements(*, sample_kinds, seed, gpu_trace_bytes,
                                  host_trace_bytes, margin_bytes, margin_bps):
    """Check proposed study-specific declarations, not evidence authenticity."""
    if type(sample_kinds) is not tuple or sample_kinds != SAMPLE_KINDS:
        raise ValueError('the 13 calibration sample kinds must be in protocol order')
    values = (seed, gpu_trace_bytes, host_trace_bytes, margin_bytes, margin_bps)
    if any(type(value) is not int for value in values) or seed != CALIBRATION_SEED:
        raise ValueError('invalid calibration seed or integer allowance')
    if (gpu_trace_bytes < MIN_TRACE_ALLOWANCE_BYTES or host_trace_bytes < MIN_TRACE_ALLOWANCE_BYTES
            or margin_bytes < MIN_MARGIN_BYTES or margin_bps < MIN_MARGIN_BPS):
        raise ValueError('trace allowances or profile margin are below the proposed policy')
