"""Small supervised DistilGPT2 GPU memory observation, not a benchmark."""

import gc
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import traceback
import uuid
import weakref
from datetime import datetime, timezone

from .device_identity import normalize_gpu_uuid, parse_nvidia_smi_inventory, resolve_device_identity
from .device_lease import DeviceLease
from .supervisor import OwnerSession, run_supervised

_GIB = 1024 ** 3
_FRACTION = 0.25


def _runtime_uuid(torch, value):
    expected = getattr(getattr(torch, "_C", None), "_CUuuid", None)
    if not isinstance(expected, type) or type(value) is not expected:
        raise ValueError("unsupported CUDA UUID type")
    rendered = str(value)
    try:
        parsed = uuid.UUID(rendered)
    except (ValueError, AttributeError) as exc:
        raise ValueError("unsupported CUDA UUID") from exc
    if rendered != str(parsed):
        raise ValueError("CUDA UUID must be canonical lowercase")
    return normalize_gpu_uuid("GPU-" + rendered)


def _snapshot(session, lease, torch):
    OwnerSession.check(session, lease)
    torch.cuda.synchronize(0)
    free, total = torch.cuda.mem_get_info(0)
    result = {"device_free_bytes": free, "device_total_bytes": total,
              "allocated_bytes": torch.cuda.memory_allocated(0),
              "reserved_bytes": torch.cuda.memory_reserved(0),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
    if (any(type(value) is not int or value < 0 for value in result.values())
            or total == 0 or free > total or result["allocated_bytes"] > result["reserved_bytes"]
            or result["peak_allocated_bytes"] < result["allocated_bytes"]
            or result["peak_reserved_bytes"] < result["reserved_bytes"]
            or result["peak_allocated_bytes"] > result["peak_reserved_bytes"]):
        raise RuntimeError("inconsistent CUDA memory counters")
    return result


def _measure_once(session, lease, torch, load_model, *, record_stage=None):
    """Run exactly one forward, returning only after verified local disposal."""
    if type(session) is not OwnerSession or type(lease) is not DeviceLease:
        raise ValueError("an actual supervised owner and device lease are required")
    OwnerSession.check(session, lease)
    if torch.cuda.is_initialized() is not False:
        raise RuntimeError("CUDA was initialized before the owner check")
    count = torch.cuda.device_count()
    if type(count) is not int or count != 1:
        raise RuntimeError("exactly one CUDA-visible device is required")
    OwnerSession.check(session, lease)
    properties = torch.cuda.get_device_properties(0)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=count)
    if not 7 * _GIB <= properties.total_memory <= 9 * _GIB:
        raise RuntimeError("device total is outside the expected 8 GiB range")
    OwnerSession.check(session, lease)
    torch.cuda.set_per_process_memory_fraction(_FRACTION, 0)
    effective = int(_FRACTION * properties.total_memory)
    readback = torch.cuda.get_per_process_memory_fraction(0)
    if readback != effective / properties.total_memory:
        raise RuntimeError("allocator fraction readback does not match the cap")
    torch.cuda.reset_peak_memory_stats(0)
    def snapshot(stage):
        value = _snapshot(session, lease, torch)
        if record_stage is not None:
            record_stage(stage, value)
        return value
    initial = snapshot("before_model_copy")
    if initial["device_free_bytes"] < 3 * _GIB:
        raise RuntimeError("insufficient device-wide free memory for the pilot")
    loaded = load_model()
    model = loaded.model
    torch.random.default_generator.manual_seed(17)
    torch.cuda.manual_seed(17)
    OwnerSession.check(session, lease)
    model.to("cuda:0")
    after_copy = snapshot("after_model_copy")
    if model.training is not False or model.config.use_cache is not False:
        raise RuntimeError("model must be in eval mode with cache disabled")
    parameter_count = 0
    for parameter in model.parameters():
        if parameter.device.type != "cuda" or parameter.device.index != 0 or parameter.dtype != torch.float32:
            raise RuntimeError("model parameters are not float32 on cuda:0")
        parameter_count += parameter.numel()
    if parameter_count <= 0 or parameter_count != loaded.metadata["parameter_count"]:
        raise RuntimeError("model parameter count does not match the CPU load")
    for buffer in model.buffers():
        if buffer.device.type != "cuda" or buffer.device.index != 0:
            raise RuntimeError("model buffers are not on cuda:0")
    parameter = None
    buffer = None
    input_cpu = torch.tensor([[15496, 995]], dtype=torch.long, device="cpu")
    input_gpu = input_cpu.to("cuda:0")
    if input_gpu.device.type != "cuda" or input_gpu.device.index != 0:
        raise RuntimeError("input was not transferred to cuda:0")
    with torch.inference_mode():
        output = model(input_ids=input_gpu, use_cache=False)
    logits = output.logits
    if (logits.device.type != "cuda" or logits.device.index != 0
            or tuple(logits.shape) != (1, 2, 50257) or logits.dtype != torch.float32
            or logits.numel() * logits.element_size() != 402056):
        raise RuntimeError("forward output differs from the fixed contract")
    after_forward = snapshot("after_forward")
    output_cpu = logits.detach().to(device="cpu", copy=True)
    if output_cpu.device.type != "cpu":
        raise RuntimeError("forward result did not offload to CPU")
    if not bool(torch.isfinite(output_cpu).all()):
        raise RuntimeError("forward produced non-finite logits")
    output_sha256 = hashlib.sha256(output_cpu.detach().view(torch.int32).contiguous().numpy().tobytes()).hexdigest()
    output_observation = {"output_shape": [1, 2, 50257], "output_dtype": "float32",
                          "output_tensor_bytes": 402056, "finite": True,
                          "output_sha256": output_sha256, "output_byte_order": sys.byteorder}
    if record_stage is not None:
        record_stage("output", output_observation)
    references = {"model": weakref.ref(model), "input_gpu": weakref.ref(input_gpu),
                  "logits": weakref.ref(logits)}
    metadata = loaded.metadata
    del output_cpu, output, logits, input_gpu, input_cpu, model, loaded
    gc.collect()
    survivors = {name: ref() is not None for name, ref in references.items()}
    if record_stage is not None:
        record_stage("cleanup_weakrefs", survivors)
    OwnerSession.check(session, lease)
    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()
    cleanup_before_workspaces = snapshot("cleanup_before_workspace_clear")
    # PyTorch 2.8 internally retains cuBLAS workspaces in its allocator.
    # This private API is version-bound and is only called after synchronization.
    if torch.__version__ != "2.8.0+cu128" or not callable(getattr(torch._C, "_cuda_clearCublasWorkspaces", None)):
        raise RuntimeError("unsupported PyTorch cuBLAS workspace cleanup API")
    OwnerSession.check(session, lease)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()
    cleanup = snapshot("after_cleanup")
    if any(survivors.values()):
        raise RuntimeError("model, input or output object still alive after disposal")
    if cleanup["allocated_bytes"] != 0 or cleanup["reserved_bytes"] != 0:
        raise RuntimeError("CUDA allocator still owns bytes after disposal")
    return {"input_ids": [15496, 995], "seed": 17, "output_shape": [1, 2, 50257],
            "output_dtype": "float32", "output_tensor_bytes": 402056,
            "finite": True, "output_sha256": output_sha256, "output_byte_order": sys.byteorder,
            "runtime_uuid_verified": True,
            "execution_device": "cuda:0", "device_name": properties.name,
            "device_total_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "allocator_fraction": _FRACTION, "allocator_limit_bytes": effective,
            "allocator_readback": readback, "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda, "cpu_load_metadata": metadata,
            "cleanup_weakrefs": survivors,
            "workspace_cleanup": "PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces",
            "snapshots": {"before_model_copy": initial, "after_model_copy": after_copy,
                          "after_forward": after_forward,
                          "cleanup_before_workspace_clear": cleanup_before_workspaces,
                          "after_cleanup": cleanup}}


def _write_json(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with open(path, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _import_torch():
    import torch
    return torch


def _owner_main(run_dir, cache_dir, run_id):
    """Dedicated child: retain the device lease until process exit."""
    lease = None
    try:
        if "torch" in sys.modules or "transformers" in sys.modules:
            raise RuntimeError("CUDA-capable dependencies were imported before lease acquisition")
        session = OwnerSession.from_environment()
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        OwnerSession.check(session, lease)
        if session.identity is None:
            raise RuntimeError("physical GPU identity is required")
        _write_json(Path(run_dir) / "owner.json", {"schema": "gpu-oneshot-owner-identity-v2",
                    "run_id": run_id, "supervisor_run_id": session.run_id, "pid": os.getpid()})
        torch = _import_torch()
        from .pretrained import load_distilgpt2
        result = _measure_once(session, lease, torch,
                               lambda: load_distilgpt2(cache_dir=cache_dir, allow_download=False),
                               record_stage=lambda stage, value: _write_json(Path(run_dir) / (stage + ".json"), value))
        candidate = {"schema": "gpu-oneshot-owner-v2", "run_id": run_id,
                     "supervisor_run_id": session.run_id, "pid": os.getpid(), "measurement": result}
        _write_json(Path(run_dir) / "candidate.json", candidate)
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        # An unverified failure never attests clean; the supervisor retains
        # the marker. Keep diagnostics bounded and avoid paths or raw UUIDs.
        try:
            _write_json(Path(run_dir) / "failure.json", {
                "schema": "gpu-oneshot-failure-v2", "run_id": run_id,
                "error_type": type(exc).__name__, "status": "quarantined",
                "private_traceback": traceback.format_exc()[-8192:]})
        except BaseException:
            pass
        return 1


def _smi_snapshot(smi):
    completed = subprocess.run([smi, "--query-gpu=uuid,name,memory.total,memory.free,memory.used,driver_version",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=10, check=True)
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) > 1024:
        raise RuntimeError("pilot requires exactly one nvidia-smi physical GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError("invalid nvidia-smi GPU inventory")
    gpu_uuid = parse_nvidia_smi_inventory(fields[0])[0]
    try:
        total, free, used = (int(value) * 1024 ** 2 for value in fields[2:5])
    except ValueError as exc:
        raise RuntimeError("nvidia-smi memory counters unavailable") from exc
    if not 7 * _GIB <= total <= 9 * _GIB or not 0 <= free <= total or not 0 <= used <= total:
        raise RuntimeError("nvidia-smi memory counters are outside pilot limits")
    return gpu_uuid, {"device_name": fields[1], "total_bytes": total, "free_bytes": free,
                      "used_bytes": used, "driver_version": fields[5],
                      "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def _owner_presence_after_exit(smi, pid):
    try:
        result = subprocess.run([smi, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True)
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if any(not row.isdigit() for row in rows):
            return {"query": "inconclusive", "owner_pid_listed": None}
        return {"query": "ok", "owner_pid_listed": str(pid) in rows}
    except (OSError, subprocess.SubprocessError):
        return {"query": "unavailable", "owner_pid_listed": None}


def _verify_offline_cache(cache_dir):
    from .pretrained import ASSETS, REVISION
    snapshot = Path(cache_dir) / "models--distilbert--distilgpt2" / "snapshots" / REVISION
    for name, (expected_size, expected_hash) in ASSETS.items():
        path = snapshot / name
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(block)
                if count > expected_size:
                    raise RuntimeError("offline checkpoint size mismatch")
                digest.update(block)
        if count != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError("offline checkpoint integrity mismatch")


def _verify_committed_sources(repo, commit):
    """Bind the owner and its local coordination code to the recorded commit."""
    paths = ("looking_glass/gpu_oneshot.py", "looking_glass/pretrained.py",
             "looking_glass/device_identity.py", "looking_glass/device_lease.py",
             "looking_glass/supervisor.py")
    hashes = {}
    for relative in paths:
        stored = subprocess.run(["git", "show", commit + ":" + relative], cwd=repo,
                                capture_output=True, check=True, timeout=10).stdout
        current = (repo / relative).read_bytes()
        if current != stored:
            raise RuntimeError("committed pilot source bytes differ from checkout")
        hashes[relative] = hashlib.sha256(current).hexdigest()
    return hashes


def _verify_controller_import_mode():
    if not sys.dont_write_bytecode or not sys.pycache_prefix:
        raise RuntimeError("controller requires -B and a fresh pycache prefix")
    prefix = Path(sys.pycache_prefix)
    if not prefix.is_absolute() or not prefix.is_dir() or any(prefix.iterdir()):
        raise RuntimeError("controller pycache prefix must be an empty absolute directory")


def run_one_shot(*, workspace_root, cache_dir, output, site_packages, smi):
    """Supervise one short-lived owner and publish its sanitized outcome."""
    _verify_controller_import_mode()
    root = Path(workspace_root).resolve(strict=True)
    cache = Path(cache_dir).resolve(strict=True)
    site = Path(site_packages).resolve(strict=True)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError("output must be an absolute, new file")
    with tempfile.NamedTemporaryFile(prefix=".oneshot-write-test-", dir=destination.parent):
        pass
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                            text=True, check=True, timeout=10).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                           capture_output=True, text=True, check=True, timeout=10).stdout
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or dirty:
        raise RuntimeError("pilot requires a clean committed source checkout")
    source_hashes = _verify_committed_sources(repo, commit)
    _verify_offline_cache(cache)
    gpu_uuid, before = _smi_snapshot(smi)
    environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")
                   if key in os.environ}
    system_root = Path(environment.get("SYSTEMROOT", "C:/Windows"))
    environment.update({"PATH": str(system_root / "System32"), "CUDA_VISIBLE_DEVICES": gpu_uuid,
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "HF_HOME": str(cache.parent), "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "17",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    run_dir = Path(tempfile.mkdtemp(prefix="gpu-oneshot-", dir=root / "work"))
    prefix = Path(tempfile.mkdtemp(prefix="pycache-", dir=run_dir))
    environment["PYTHONPYCACHEPREFIX"] = str(prefix)
    run_id = secrets.token_hex(16)
    _write_json(run_dir / "started.json", {"schema": "gpu-oneshot-start-v2", "run_id": run_id,
                                            "source_commit": commit, "started_utc": before["timestamp_utc"]})
    bootstrap = ("import sys,runpy;sys.path.extend((sys.argv.pop(1),sys.argv.pop(1)));"
                 "runpy.run_module('looking_glass.gpu_oneshot',run_name='__main__',alter_sys=True)")
    command = [sys._base_executable, "-B", "-S", "-P", "-X", "pycache_prefix=" + str(prefix),
               "-c", bootstrap, str(repo), str(site), "--owner", str(run_dir), str(cache), run_id]
    supervision = None
    supervisor_error = None
    try:
        supervision = run_supervised(command, device_lease_path=identity.lock_path, marker_path=identity.marker_path,
                                     identity=identity, timeout=300, lease_timeout=30, cwd=repo,
                                     env=environment, inherit_environment=False)
    except BaseException as exc:
        # The supervisor may have launched an owner. Do not infer an exit or
        # clean teardown from a missing result; its marker remains authoritative.
        supervisor_error = type(exc).__name__
    supervision_evidence_error = None
    if supervision is not None:
        try:
            _write_json(run_dir / "supervision.json", {"run_id": run_id,
                        "supervisor_run_id": supervision.run_id, "pid": supervision.pid,
                        "exit_code": supervision.exit_code, "clean": supervision.clean,
                        "timed_out": supervision.timed_out, "reason": supervision.reason})
        except Exception as exc:
            supervision_evidence_error = type(exc).__name__
    after = None
    telemetry_error = None
    try:
        after_uuid, after = _smi_snapshot(smi)
        if after_uuid != gpu_uuid:
            raise RuntimeError("physical inventory changed after owner exit")
    except Exception as exc:
        after = None
        telemetry_error = type(exc).__name__
    presence = (_owner_presence_after_exit(smi, supervision.pid) if supervision is not None
                else {"query": "not_attempted", "owner_pid_listed": None})
    outcome = {"schema": "gpu-oneshot-result-v2", "run_id": run_id, "source_commit": commit,
               "source_sha256": source_hashes, "controller_fresh_pycache": True,
               "status": "quarantined",
               "supervisor": ({"exit_code": supervision.exit_code,
                   "clean": supervision.clean, "timed_out": supervision.timed_out}
                   if supervision is not None else None), "supervisor_error_type": supervisor_error,
               "supervision_evidence_error_type": supervision_evidence_error,
               "nvidia_smi_before": before, "nvidia_smi_after": after, "owner_presence_after_exit": presence,
               "post_telemetry_error_type": telemetry_error,
               "limitations": ["one observation, not a latency or throughput benchmark",
                   "allocator fraction limits PyTorch's allocator, not all CUDA allocations",
                   "device-wide free memory is a point-in-time value and may include desktop activity",
                   "Windows WDDM may not expose all GPU processes or their memory in nvidia-smi",
                   "the cooperative lock does not control unrelated GPU processes"]}
    if supervision is not None and supervision.clean and supervision.exit_code == 0 and not supervision.timed_out:
        outcome["status"] = "incomplete"
        try:
            raw = (run_dir / "candidate.json").read_bytes()
            if len(raw) > 1024 * 1024:
                raise RuntimeError("owner result exceeds its byte cap")
            candidate = json.loads(raw)
            if (candidate.get("schema") != "gpu-oneshot-owner-v2" or candidate.get("run_id") != run_id
                    or candidate.get("supervisor_run_id") != supervision.run_id
                    or candidate.get("pid") != supervision.pid):
                raise RuntimeError("owner result does not match the reaped child")
            outcome["measurement"] = candidate["measurement"]
            if (telemetry_error is None and supervision_evidence_error is None
                    and presence["owner_pid_listed"] is not True):
                outcome["status"] = "success"
        except Exception as exc:
            outcome["candidate_error_type"] = type(exc).__name__
    else:
        try:
            raw = (run_dir / "failure.json").read_bytes()
            if len(raw) <= 16384:
                failure = json.loads(raw)
                if failure.get("schema") == "gpu-oneshot-failure-v2" and failure.get("run_id") == run_id:
                    outcome["owner_error_type"] = failure.get("error_type")
        except (OSError, ValueError, AttributeError):
            pass
    partial = {}
    for stage in ("before_model_copy", "after_model_copy", "after_forward", "output",
                  "cleanup_weakrefs", "cleanup_before_workspace_clear", "after_cleanup"):
        try:
            raw = (run_dir / (stage + ".json")).read_bytes()
            if len(raw) <= 4096:
                value = json.loads(raw)
                if type(value) is dict:
                    partial[stage] = value
        except (OSError, ValueError):
            pass
    if partial:
        outcome["partial_stage_observations"] = partial
    try:
        _verify_controller_import_mode()
    except Exception as exc:
        outcome["controller_fresh_pycache"] = False
        outcome["controller_import_error_type"] = type(exc).__name__
        if outcome["status"] == "success":
            outcome["status"] = "incomplete"
    _write_json(destination, outcome)
    return outcome


def main(argv=None):
    import argparse
    import shutil
    import sysconfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", nargs=3, metavar=("RUN_DIR", "CACHE_DIR", "RUN_ID"), help=argparse.SUPPRESS)
    parser.add_argument("--workspace-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.owner:
        return _owner_main(*args.owner)
    if not args.workspace_root or not args.cache_dir or not args.output:
        parser.error("--workspace-root, --cache-dir and --output are required")
    result = run_one_shot(workspace_root=args.workspace_root, cache_dir=args.cache_dir, output=args.output,
                          site_packages=sysconfig.get_path("purelib"), smi=shutil.which("nvidia-smi"))
    print(json.dumps({"status": result["status"], "source_commit": result["source_commit"]}))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
