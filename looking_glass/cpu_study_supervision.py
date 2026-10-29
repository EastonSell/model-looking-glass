"""Cooperative supervision for a trusted CPU-only experiment process.

The study root and marker do not coordinate or attest GPU state. See
docs/CPU_STUDY_SUPERVISION.md for the required child and custody contract.
"""

import os
from pathlib import Path
import sys

from .device_lease import DeviceLease
from . import supervisor


CPU_STUDY_SUPERVISION_CONTRACT = "cooperative-cpu-study-v1"
_REQUIRED_LAUNCH_CONTRACT = "gated-rooted-child-v1"
_FORBIDDEN_EARLY_MODULES = frozenset({
    "torch", "transformers", "accelerate", "triton", "torchvision", "torchaudio",
    "cupy", "jax", "tensorflow", "numba", "pycuda",
})


class CpuStudyError(RuntimeError):
    """CPU study preflight could not establish its cooperative contract."""


def _study_paths(study_root):
    root = Path(study_root)
    if not root.is_absolute():
        raise ValueError("study_root must be absolute")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("study_root must be an existing directory")
    return root / "cpu-study.lock", root / "cpu-study.unsafe.json"


def run_cpu_study(command, *, study_root, timeout, lease_timeout=5,
                  terminate_grace=5, cwd=None, env=None, startup_channel=None,
                  inherit_environment=False, process_job=None):
    """Supervise one trusted direct child using its dedicated CPU study root."""
    if getattr(supervisor, "SUPERVISOR_LAUNCH_CUSTODY_CONTRACT", None) != _REQUIRED_LAUNCH_CONTRACT:
        raise CpuStudyError("reviewed supervisor launch custody is unavailable")
    lock, marker = _study_paths(study_root)
    job_option = {} if process_job is None else {"process_job": process_job}
    return supervisor.run_supervised(
        command, device_lease_path=lock, marker_path=marker, identity=None,
        timeout=timeout, lease_timeout=lease_timeout, terminate_grace=terminate_grace,
        cwd=cwd, env=env, startup_channel=startup_channel,
        inherit_environment=inherit_environment, **job_option,
    )


def claim_cpu_study(study_root, *, lease_timeout=5):
    """Pass the launch gate, then acquire the expected CPU-only study lease.

    Call from the fresh direct child before importing model/CUDA-capable
    dependencies. Return the session and lease, both retained until exit.
    A caller must fail closed on errors and attest only after CPU teardown.
    """
    if getattr(supervisor, "SUPERVISOR_LAUNCH_CUSTODY_CONTRACT", None) != _REQUIRED_LAUNCH_CONTRACT:
        raise CpuStudyError("reviewed supervisor launch custody is unavailable")
    lock, marker = _study_paths(study_root)
    session = supervisor.OwnerSession.from_environment()
    if (type(session) is not supervisor.OwnerSession or session._identity_spec is not None
            or session.identity is not None):
        raise CpuStudyError("CPU study requires an identity-less owner session")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CpuStudyError("CPU study requires an empty CUDA visibility mask")
    if any(name.partition(".")[0] in _FORBIDDEN_EARLY_MODULES for name in sys.modules):
        raise CpuStudyError("CPU study model or CUDA-capable modules were imported too early")
    if (os.path.normcase(str(session.device_lease_path)) != os.path.normcase(str(lock))
            or os.path.normcase(str(session.marker_path)) != os.path.normcase(str(marker))):
        raise CpuStudyError("CPU study session belongs to a different custody root")
    lease = DeviceLease.acquire(lock, timeout=lease_timeout)
    try:
        supervisor.OwnerSession.check(session, lease)
        if session.identity is not None:
            raise CpuStudyError("CPU study cannot bind a GPU identity")
    except BaseException:
        lease.release()
        raise
    return session, lease
