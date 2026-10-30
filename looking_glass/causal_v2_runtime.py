"""CPU numeric primitives for the proposed synthetic causal v2 runner.

This module is not a study entry point and does not prepare cases, execute a
model, seal commitments, or establish a source-provenance boundary. Call the
policy setup only in a fresh, isolated execution process after its bootstrap.
"""
from copy import deepcopy
import os
import platform
import threading

import torch


class NumericPolicyError(RuntimeError):
    """The execution thread or a supplied numeric value violates v2 policy."""


_POLICY_TOKEN = object()


def _probe_preservation():
    """Probe both subnormal input preservation and subnormal production."""
    minimum = torch.tensor([1], dtype=torch.int32, device="cpu").view(torch.float32)
    zero = torch.zeros(1, dtype=torch.float32, device="cpu")
    input_preserved = not bool(torch.eq(minimum, zero).all())
    tiny = torch.tensor([0x00800000], dtype=torch.int32, device="cpu").view(torch.float32)
    half = torch.tensor([0.5], dtype=torch.float32, device="cpu")
    result_bits = (tiny * half).view(torch.int32).item()
    return {"input_preserved": input_preserved,
            "output_preserved": result_bits == 0x00400000}


def _require_preservation():
    try:
        result = _probe_preservation()
    except Exception as error:
        raise NumericPolicyError("CPU denormal probes failed") from error
    if not all(result.values()):
        raise NumericPolicyError("CPU denormal preservation is not established")
    return result


class NumericPolicy:
    """An execution-thread policy; recheck it before and after model work."""
    def __init__(self, initial, after_threads, *, _token=None):
        if _token is not _POLICY_TOKEN:
            raise NumericPolicyError("numeric policy must be established by its factory")
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._record = {
            "flush_denormal_requested": False,
            "preservation_request_succeeded": True,
            "initial_probes": dict(initial),
            "single_thread_probes": dict(after_threads),
            "intraop_threads": 1,
            "process_id": self._pid,
            "execution_thread_id": threading.get_ident(),
            "cpu_architecture": platform.machine(),
            "torch_version": str(torch.__version__),
            "torch_build": torch.__config__.show(),
        }

    def verify(self):
        if os.getpid() != self._pid or threading.current_thread() is not self._thread:
            raise NumericPolicyError("numeric policy belongs to a different process or thread")
        if torch.get_num_threads() != 1:
            raise NumericPolicyError("numeric policy requires one intra-op thread")
        return _require_preservation()

    def record(self):
        return deepcopy(self._record)


def establish_numeric_policy():
    """Request and verify v2 preservation in this fresh execution thread.

    This changes the calling thread's denormal mode and the process-wide PyTorch
    intra-op thread count; it must run only inside the isolated study process.
    It does not attest to other threads, code provenance, or prior model work.
    """
    try:
        requested = torch.set_flush_denormal(False)
    except Exception as error:
        raise NumericPolicyError("denormal preservation request failed") from error
    if requested is not True:
        raise NumericPolicyError("denormal preservation is unsupported")
    initial = _require_preservation()
    try:
        torch.set_num_threads(1)
    except Exception as error:
        raise NumericPolicyError("cannot set the single-thread policy") from error
    if torch.get_num_threads() != 1:
        raise NumericPolicyError("single-thread policy was not applied")
    after_threads = _require_preservation()
    return NumericPolicy(initial, after_threads, _token=_POLICY_TOKEN)


def _float32_bits(value):
    if (type(value) is not torch.Tensor or value.layout != torch.strided or
            value.is_nested or value.is_neg() or value.is_conj() or any(name is not None for name in getattr(value, "names", ())) or
            value.device.type != "cpu" or value.dtype != torch.float32 or
            value.numel() == 0 or value.numel() > 8):
        raise NumericPolicyError("expected a nonempty bounded dense CPU float32 tensor")
    bits = value.detach().view(torch.int32)
    if bool(((bits & 0x7f800000) == 0x7f800000).any()):
        raise NumericPolicyError("nonfinite float32 values are invalid")
    return bits


def float32_exact_equal(left, right):
    """Compare retained finite float32 bits, treating both signed zeros alike.

    Shapes must match; no broadcasting or float operation is used. This does
    not validate trace identity, execution conditions, or the model's mode.
    """
    a, b = _float32_bits(left), _float32_bits(right)
    if a.shape != b.shape:
        raise NumericPolicyError("float32 comparison shapes differ")
    equal = (a == b) | (((a & 0x7fffffff) == 0) & ((b & 0x7fffffff) == 0))
    return bool(equal.all())
