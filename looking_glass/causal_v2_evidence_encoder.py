"""Encode the synthetic-v2 executor's private result into a cell buffer.

This converts already returned CPU observations; it does not execute the
model, verify a phase gate, validate controls, authenticate identities, or
provide access control. The custodian must keep the input result and snapshots
exclusive and unchanged until this call returns.
"""
import hashlib

import torch

from . import registry
from .causal_v2_evidence_adapter import CellEvidence
from .causal_v2_selection import DiscoveryCondition
from .probes import TraceEvent


_RESULT_KEYS = {"clean_trace", "repeat_trace", "clean_logits", "repeat_logits", "condition", "plans",
                "private_integrity", "comparison", "input_sha256", "model_state_sha256"}
_PLAN_KEYS = {"kind", "site", "metadata", "captures", "source_identity", "comparison_identity", "base_identity"}
_PRIVATE_KEYS = {"target", "delta", "clean_target_bits", "perturbed_target_bits",
                 "clean_capture_metadata", "perturbed_capture_metadata"}
_REFERENCE_KEYS = {"clean_sparse_trace", "perturbed_sparse_trace", "clean", "perturbed"}


def _tensor(value, width):
    if (type(value) is not torch.Tensor or tuple(value.shape) != (1, width)
            or value.dtype != torch.float32 or value.device.type != "cpu" or value.layout != torch.strided
            or value.is_nested or value.is_neg() or value.is_conj()
            or any(name is not None for name in (getattr(value, "names", None) or ()))):
        raise ValueError("snapshot must be a CPU float32 tensor with the expected shape")
    # Reinterpret first, then copy integer bits; a floating-point copy need not
    # preserve subnormals under every ambient denormal mode or implementation.
    bits = [int(bit) & 0xffffffff for bit in value.detach().view(torch.int32).contiguous().reshape(-1).tolist()]
    # Check exponent bits, without a float32 conversion that could flush subnormals.
    if any(((bit >> 23) & 0xff) == 0xff for bit in bits):
        raise ValueError("snapshot must be finite")
    return {"dtype": "torch.float32", "device": "cpu", "shape": [1, width], "bits": bits}


def _bits_tensor(bits):
    if (type(bits) is not list or len(bits) != 8
            or any(type(bit) is not int or not 0 <= bit <= 0xffffffff
                   or ((bit >> 23) & 0xff) == 0xff for bit in bits)):
        raise ValueError("capture must contain eight finite float32 bit patterns")
    return {"dtype": "torch.float32", "device": "cpu", "shape": [1, 8], "bits": bits}


def _trace(events):
    if type(events) not in (tuple, list) or len(events) != 4:
        raise ValueError("a four-event sparse trace is required")
    if any(type(event) is not TraceEvent for event in events):
        raise ValueError("invalid sparse trace event")
    return [{"metadata": event.metadata(), "value": _tensor(event.value, 8)} for event in events]


def _references(value):
    if (type(value) is not dict or set(value) != _REFERENCE_KEYS
            or any(type(ref) is not str or not 0 < len(ref) <= 220 for ref in value.values())
            or len(set(value.values())) != len(value)):
        raise ValueError("four distinct bounded source/base references are required")
    return value


def _plan(plan, refs):
    if (type(plan) is not dict or set(plan) != _PLAN_KEYS or type(plan["kind"]) is not str
            or type(plan["site"]) is not int or not 0 <= plan["site"] <= 3
            or type(plan["metadata"]) is not dict or type(plan["captures"]) is not dict):
        raise ValueError("incomplete executor plan")
    source, comparison, base = plan["source_identity"], plan["comparison_identity"], plan["base_identity"]
    traces = ("clean_sparse_trace", "perturbed_sparse_trace")
    if ((source is not None and source not in (*traces, "same_forward_self_source"))
            or comparison not in traces or base not in ("clean", "perturbed")):
        raise ValueError("invalid executor source/comparison/base role")
    suffix = f"/block.{plan['site']}.output"
    source_id = (None if source is None else "same_forward_self_source" if source == "same_forward_self_source"
                 else refs[source] + suffix)
    return {"kind": plan["kind"], "site": plan["site"], "metadata": plan["metadata"],
            "captures": {name: _bits_tensor(bits) for name, bits in plan["captures"].items()},
            "source_role": source, "source_id": source_id,
            "comparison_role": comparison, "comparison_id": refs[comparison] + suffix,
            "base_role": base, "base_id": refs[base]}


def _record(buffer, result, *, case_id, case_index, condition_id, prediction_ref, references):
    if type(result) is not dict or set(result) != _RESULT_KEYS:
        raise ValueError("executor result fields differ; no fields may be silently omitted")
    refs = _references(references)
    condition = result["condition"]
    if type(condition) is not DiscoveryCondition or type(result["plans"]) is not list:
        raise ValueError("executor condition or plan inventory is invalid")
    private = result["private_integrity"]
    if type(private) is not dict or set(private) != _PRIVATE_KEYS:
        raise ValueError("private target integrity fields differ")
    row = {"schema": "causal-v2-condition-record-v2", "cell_id": buffer.cell_id,
           "phase": buffer.phase, "case_id": case_id, "case_index": case_index,
           "condition_id": condition_id, "model_seed": buffer.model_seed,
           "model_state_sha256": result["model_state_sha256"], "input_sha256": result["input_sha256"],
           "strength": condition.strength, "prediction_ref": prediction_ref,
           "traces": {"clean": _trace(result["clean_trace"]), "repeat_clean": _trace(result["repeat_trace"]),
                      "perturbed": _trace(condition.perturbed_trace)},
           "logits": {"clean": _tensor(result["clean_logits"], 2),
                      "repeat_clean": _tensor(result["repeat_logits"], 2),
                      "perturbed": _tensor(condition.perturbed_logits, 2),
                      "forward": [_tensor(item, 2) for item in condition.clean_to_perturbed_logits],
                      "reverse": [_tensor(item, 2) for item in condition.perturbed_to_clean_logits],
                      "null": [_tensor(item, 2) for item in condition.null_logits],
                      "self": [_tensor(item, 2) for item in condition.identity_logits],
                      "upstream": _tensor(condition.upstream_logits, 2),
                      "post_hook_clean": _tensor(condition.clean_verify_logits, 2)},
           "plans": [_plan(plan, refs) for plan in result["plans"]],
           "private_integrity": {"target": private["target"], "delta": private["delta"],
                                 "clean_target": _bits_tensor(private["clean_target_bits"]),
                                 "perturbed_target": _bits_tensor(private["perturbed_target_bits"]),
                                 "clean_capture_metadata": private["clean_capture_metadata"],
                                 "perturbed_capture_metadata": private["perturbed_capture_metadata"]},
           "comparison": result["comparison"]}
    return registry._json_bytes(row)


def append_executor_result(buffer, result, *, case_id, case_index, condition_id, prediction_ref, references):
    """Encode and append one result; return the record digest, not a publication receipt.

    References bind clean/perturbed sparse-trace and base identities supplied by
    the custodian. The actual input/model/source and reference bindings must be
    verified by the executor and custodian; matching strings do not prove them.
    """
    if type(buffer) is not CellEvidence:
        raise TypeError("a CellEvidence buffer is required")
    try:
        raw = _record(buffer, result, case_id=case_id, case_index=case_index,
                      condition_id=condition_id, prediction_ref=prediction_ref, references=references)
        digest = hashlib.sha256(raw).hexdigest()
        buffer.append(raw)
        return digest
    except BaseException as original:
        try:
            buffer.mark_failed()
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
                raise cleanup from original
            raise original from cleanup
        raise
