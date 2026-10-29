"""CPU runtime and model-identity preflight; this does not authorize a study.

No cases, schedules or model forwards are produced here. A source-verified
launcher must authenticate the code and dependencies separately and invoke this
in the same isolated process and thread as the prospective study.
"""
import hashlib
import os
import platform
import re
import sys

import torch

from .causal_v2_executor import _check_model
from .causal_v2_model_identity import state_sha256
from .causal_v2_preparation import MODEL_SEEDS
from .causal_v2_runtime import establish_numeric_policy
from .model import TinyResidual


class PreflightError(ValueError):
    """The measured process is incompatible with the proposed v2 protocol."""


_THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_DYNAMIC", "MKL_DYNAMIC",
                       "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def inspect_backend_settings():
    """Return bounded CPU backend and thread settings for separate manifest binding."""
    environment = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    if any(value is not None and re.fullmatch(r"[A-Za-z0-9 ,+_.-]{0,128}", value) is None
           for value in environment.values()):
        raise PreflightError("thread environment value is not a bounded safe setting")
    settings = {"interop_threads": torch.get_num_interop_threads(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "mkldnn_enabled": torch.backends.mkldnn.enabled,
                "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "thread_environment": environment}
    if (type(settings["interop_threads"]) is not int or settings["interop_threads"] < 1
            or settings["float32_matmul_precision"] not in ("highest", "high", "medium")
            or type(settings["cpu_capability"]) is not str
            or re.fullmatch(r"[A-Z0-9_]{1,64}", settings["cpu_capability"]) is None
            or any(type(settings[name]) is not bool for name in
                   ("mkldnn_enabled", "mkldnn_deterministic", "deterministic_algorithms",
                    "deterministic_warn_only"))):
        raise PreflightError("CPU backend setting is invalid")
    return settings


def inspect_runtime_and_models():
    """Return a checked numeric policy, runtime record, and ordered state hashes.

    The resulting state hashes are for freshly initialized CPU models and are
    rechecked when the actual callback constructs its models. The policy changes
    this thread's denormal mode and the process-wide intra-op thread count.
    CPU RNG state is restored. These are cooperative runtime measurements, not
    an attestation of executed source, interpreter, or native-library bytes.
    """
    version = str(torch.__version__)
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized()
            or platform.python_version() != "3.11.15" or version != "2.8.0+cu128"
            or torch.get_default_dtype() != torch.float32):
        raise PreflightError("CPU masking, runtime, or floating dtype differs")
    backend_settings = inspect_backend_settings()
    policy = establish_numeric_policy()
    states = []
    for seed in MODEL_SEEDS:
        policy.verify()
        if inspect_backend_settings() != backend_settings:
            raise PreflightError("CPU backend settings changed during preflight")
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = TinyResidual(width=8, depth=4).eval()
        _check_model(model)
        digest = state_sha256(model)
        if state_sha256(model) != digest:
            raise PreflightError("model state changed during preflight")
        states.append({"seed": seed, "sha256": digest})
        del model
    policy.verify()
    if torch.cuda.is_initialized():
        raise PreflightError("CUDA was initialized during CPU preflight")
    runtime = {"python_version": platform.python_version(), "torch_version": version,
               "system": platform.system(), "machine": platform.machine(), "byteorder": sys.byteorder,
               "intraop_threads": torch.get_num_threads(), "denormal_preserved": True,
               "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode("utf-8")).hexdigest()}
    policy.verify()
    if torch.cuda.is_initialized() or inspect_backend_settings() != backend_settings:
        raise PreflightError("CUDA or CPU backend settings changed during runtime recording")
    return policy, runtime, states
