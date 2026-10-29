"""Strict sparse-trace pair comparison for the proposed synthetic v2 protocol.

This is an observation validator, not a site-selection policy. It accepts no
ground-truth injection coordinate, and does not consume held-out cases.
"""
import math

import torch

from .probes import TraceEvent


class SparseTraceInvalid(ValueError):
    """A supplied pair does not satisfy the declared v2 sparse trace policy."""


_NODES = tuple(f"block.{index}.output" for index in range(4))


def _binary32_value(signed_bits):
    """Decode finite IEEE binary32 into exact binary64 without float32 math."""
    bits = signed_bits & 0xffffffff
    exponent = (bits >> 23) & 0xff
    mantissa = bits & 0x7fffff
    if exponent == 0xff:
        raise SparseTraceInvalid("trace payload contains a nonfinite value")
    if exponent == 0:
        value = math.ldexp(float(mantissa), -149)
    else:
        value = math.ldexp(float((1 << 23) | mantissa), exponent - 150)
    return -value if bits >> 31 else value


def _validate_stream(stream):
    if type(stream) not in (list, tuple):
        raise SparseTraceInvalid("trace must be a concrete list or tuple of four events")
    if len(stream) != 4:
        raise SparseTraceInvalid("trace must contain exactly four events")
    run_ids = set()
    payload_bytes = 0
    values = []
    for index, (event, expected) in enumerate(zip(stream, _NODES)):
        if not isinstance(event, TraceEvent):
            raise SparseTraceInvalid("trace members must be probe TraceEvents")
        if (not isinstance(event.run_id, str) or not event.run_id.strip() or
                event.node_id != expected or type(event.sequence) is not int or
                event.sequence != index or type(event.occurrence) is not int or
                event.occurrence != 0 or type(event.shape) is not tuple or len(event.shape) != 2 or
                any(type(dimension) is not int for dimension in event.shape) or
                event.shape != (1, 8) or
                event.dtype != "torch.float32" or event.device != "cpu"):
            raise SparseTraceInvalid("trace event metadata does not match v2 policy")
        run_ids.add(event.run_id)
        value = event.value
        if (type(value) is not torch.Tensor or value.layout != torch.strided or
                value.is_nested or value.is_neg() or value.is_conj() or value.has_names() or
                tuple(value.shape) != (1, 8) or
                value.dtype != torch.float32 or value.device.type != "cpu"):
            raise SparseTraceInvalid("trace payload does not match v2 policy")
        payload_bytes += value.numel() * value.element_size()
        # Reinterpret first, then flatten/copy as integers: no float32 operation
        # can flush a subnormal before it is reconstructed in binary64.
        bits = value.view(torch.int32).reshape(-1).tolist()
        values.append([_binary32_value(bit_pattern) for bit_pattern in bits])
    if len(run_ids) != 1:
        raise SparseTraceInvalid("each stream must have one run ID")
    if payload_bytes != 128:
        raise SparseTraceInvalid("trace payload must contain 128 logical bytes")
    return values, next(iter(run_ids))


def compare_sparse_pair(reference, candidate, *, reference_complete=False,
                        candidate_complete=False, pairing_verified=False):
    """Validate and compare one pair under the exact v2 sparse-tap policy.

    The caller attests that both captures completed under the frozen policy
    and that they share the intended model, input and runtime configuration.
    Flags and run IDs are attestations: this function cannot verify distinct
    executions, prevent IDs from encoding private targets, or establish blindness.
    An active pair's no-difference result is an observation miss. It does not
    by itself imply that the cell's selection policy must abstain.
    """
    if (reference_complete is not True or candidate_complete is not True or
            pairing_verified is not True):
        raise SparseTraceInvalid("complete, externally paired captures are required")
    reference, reference_run = _validate_stream(reference)
    candidate, candidate_run = _validate_stream(candidate)
    if reference_run == candidate_run:
        raise SparseTraceInvalid("paired captures require distinct opaque run IDs")
    norms = []
    first_index = None
    for left, right in zip(reference, candidate):
        if first_index is None and any(a != b for a, b in zip(left, right)):
            first_index = len(norms)
        norms.append(math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right))))
    if first_index is None:
        return {"status": "no_difference_observed", "first_unequal": None,
                "last_equal": _NODES[-1], "l2_by_block": norms}
    first = _NODES[first_index]
    return {"status": "upstream_difference" if first == _NODES[0] else "difference_observed",
            "first_unequal": first, "last_equal": None if first_index == 0 else _NODES[first_index - 1],
            "l2_by_block": norms}
