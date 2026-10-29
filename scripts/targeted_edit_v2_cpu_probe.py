"""No-model CPU custody/import rehearsal for the prospective v2 study.

The child is launched only by the cooperative CPU supervisor. It does not
load a checkpoint, construct a model, or run any model forward. Its process
audit hook covers CPython-visible process creation only; it cannot prove that
native code created no descendants.
"""

import argparse
import gc
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys


class ProbeError(RuntimeError):
    pass


class _AuditSentinel(Exception):
    pass


_AUDIT_TOKEN = "model-looking-glass.targeted-edit-v2-probe-sentinel"
_PROCESS_EVENTS = frozenset({"subprocess.Popen", "_winapi.CreateProcess", "os.system", "os.spawn",
                             "os.posix_spawn", "os.fork", "os.forkpty", "os.exec",
                             "os.startfile", "os.startfile/2"})
_PROJECT_MODULES = frozenset({"cpu_study_supervision", "supervisor", "device_lease", "device_identity",
    "pretrained", "pretrained_exposures", "pretrained_exposures_v2", "targeted_edit_cpu_v1",
    "targeted_edit_cpu_v2", "targeted_edit_v2_edits"})


class _ProjectSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve only the fixed project modules without adding the checkout to sys.path."""
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "looking_glass":
            return importlib.util.spec_from_file_location(fullname, self.root / "looking_glass/__init__.py",
                                                         submodule_search_locations=[])
        if not fullname.startswith("looking_glass."):
            return None
        name = fullname.removeprefix("looking_glass.")
        if name not in _PROJECT_MODULES:
            raise ImportError("project import is outside the probe source allowlist")
        return importlib.util.spec_from_file_location(fullname, self.root / "looking_glass" / (name + ".py"))


def _audit(event, _args):
    if event == _AUDIT_TOKEN:
        raise _AuditSentinel()
    if event in _PROCESS_EVENTS:
        raise ProbeError("Python-visible descendant process creation was denied")


def _install_process_guard():
    # Existing audit hooks can silently reject a new hook; require a sentinel.
    sys.addaudithook(_audit)
    try:
        sys.audit(_AUDIT_TOKEN)
    except _AuditSentinel:
        return
    raise ProbeError("process audit hook was not installed")


def _prime_spawn_free_windows_platform():
    """Use native Windows version fields where CPython otherwise runs `ver`."""
    if os.name != "nt" or sys.platform != "win32":
        raise ProbeError("this rehearsal currently requires Windows")
    import platform
    if platform._uname_cache is not None or platform._platform_cache:
        raise ProbeError("platform identity was observed before the probe configured it")
    win = sys.getwindowsversion()
    version = f"{win.major}.{win.minor}.{win.build}"
    platform._syscmd_ver = lambda *args, **kwargs: ("Windows", str(win.major), version)
    return version


def _new_json(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
    with open(path, "xb") as handle:
        if handle.write(data) != len(data):
            raise ProbeError("short probe report write")
        handle.flush()
        os.fsync(handle.fileno())


def _load_reference(repo, v1, edits):
    path = repo / "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"
    with open(path, "rb") as handle:
        data = handle.read(65537)
    if len(data) > 65536 or v1._sha(data) != edits.V1_PRECOMMIT_SHA256:
        raise ProbeError("v1 reference precommit is oversized or has the wrong digest")
    return v1._decode_document(data)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve(strict=True)
    site = Path(args.site).resolve(strict=True)
    root = Path(args.study_root).resolve(strict=True)
    report = Path(args.report)
    if (Path(__file__).resolve(strict=True) != repo / "scripts/targeted_edit_v2_cpu_probe.py"
            or not site.is_dir() or not root.is_dir() or not report.is_absolute()
            or report.exists() or not report.parent.is_dir()
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or not Path(sys.pycache_prefix).is_dir() or any(Path(sys.pycache_prefix).iterdir())
            or any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules)):
        raise ProbeError("isolated direct-child startup differs from the probe contract")
    sys.meta_path.insert(0, _ProjectSourceFinder(repo))
    _install_process_guard()
    # A real spawn attempt exercises the process hook before model imports.
    import subprocess
    try:
        subprocess.Popen([sys.executable, "-c", "raise SystemExit(90)"])
    except ProbeError:
        pass
    else:
        raise ProbeError("process creation sentinel was not denied")
    from looking_glass.cpu_study_supervision import claim_cpu_study, _FORBIDDEN_EARLY_MODULES
    def model_dependency_loaded():
        return any(name.partition(".")[0] in _FORBIDDEN_EARLY_MODULES for name in sys.modules)
    if model_dependency_loaded():
        raise ProbeError("a model or CUDA-capable dependency was imported before the claim")
    session, lease = claim_cpu_study(root)
    # The supervisor requires an empty mask for the early claim. The original
    # v1 run used -1, which changes Torch's reported build text on this host.
    # Set the exact v1 mask before importing any model/CUDA-capable dependency.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or model_dependency_loaded():
        raise ProbeError("the exact v1 CPU visibility mask could not be set")
    platform_version = _prime_spawn_free_windows_platform()
    sys.path.append(str(site))
    import torch
    from looking_glass import targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_v2_edits as edits
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized()
            or torch.is_autocast_enabled("cpu")
            or torch.is_inference_mode_enabled() or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise ProbeError("the CPU-only offline runtime preflight failed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise ProbeError("denormal preservation is unavailable")
    before = v1._denormal_probe()
    environment = v1._environment()
    after = v1._denormal_probe()
    reference = _load_reference(repo, v1, edits)
    recorded = {key: reference["environment"][key] for key in edits.ENVIRONMENT_FIELDS}
    observed = {key: environment[key] for key in edits.ENVIRONMENT_FIELDS}
    differences = [key for key in edits.ENVIRONMENT_FIELDS if v1._canonical(recorded[key]) != v1._canonical(observed[key])]
    if differences:
        raise ProbeError("spawn-free environment differs from v1 fields: " + ", ".join(differences))
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ProbeError("CUDA initialized or the CPU mask changed during the rehearsal")
    _new_json(report, {"version": "targeted-edit-v2-cpu-probe-v1", "status": "matching_v1_runtime",
        "owner_pid": os.getpid(), "supervisor_run_id": session.run_id,
        "windows_version": platform_version, "environment": observed,
        "denormal_probe_before": before, "denormal_probe_after": after,
        "model_dependency_absent_before_claim_and_mask": True,
        "cuda_visibility_before_model_imports": "-1", "cuda_visibility_after_imports": "-1",
        "python_process_guard_installed": True, "native_descendant_absence_attested": False,
        "checkpoint_loaded": False, "model_forwarded": False})
    gc.collect()
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ProbeError("CPU runtime state changed before teardown attestation")
    session.attest_clean(lease)
    # Retain the exact lease/session until this direct child exits.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
