"""Versioned, isolated latency and memory measurement for a pretrained GPT-2.

This is not the shared-service throughput experiment. Run as its own process.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import secrets
import shutil
import site
import statistics
import subprocess
import sys
import threading
import time
import traceback
import zipfile


PROTOCOL = "cuda-isolated-v1"
MODES = ("none", "id", "direct", "gradient")
# The exact eight-token trace workload needs independently reviewed resource
# admission. A two-token untraced service profile cannot authorize this run.
GPU_BENCHMARK_ADMISSION_READY = False
_PROCESS_LEASES = []
_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def _public_evidence(value):
    """Remove the physical device identifier from persisted output."""
    if isinstance(value, str):
        return _GPU_UUID.sub("GPU-<redacted>", value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            safe_key = _public_evidence(key)
            if safe_key in result:
                raise ValueError("device redaction would collide evidence keys")
            if isinstance(key, str) and key.lower() == "cuda_visible_devices":
                result[safe_key] = None if item is None else "<redacted>"
            else:
                result[safe_key] = _public_evidence(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_public_evidence(item) for item in value]
    return value


def direct_python_module(module, arguments, *, project=None):
    """Launch a module as the real Python child, with the caller's packages.

    Windows virtualenv python.exe can be a redirector which creates a second
    PID.  Skip automatic site initialization, then add the caller's known site
    directories as paths without executing arbitrary .pth startup code.
    """
    if not isinstance(module, str) or not module or not isinstance(arguments, (list, tuple)):
        raise ValueError("a module and an argument sequence are required")
    if any(not isinstance(argument, str) for argument in arguments):
        raise ValueError("module arguments must be strings")
    if sys.version_info < (3, 11):
        raise RuntimeError("source-isolated benchmark launch requires Python 3.11 or later")
    project = Path(project or Path(__file__).resolve().parent.parent).resolve(strict=True)
    selected_site = os.environ.get("LOOKING_GLASS_BENCHMARK_SITE_PACKAGES")
    source_only = (sys.flags.no_site and sys.flags.ignore_environment
                   and getattr(sys.flags, "safe_path", False))
    candidates = ([selected_site] if source_only and selected_site else
                  [path for path in site.getsitepackages()
                   if Path(path).name.lower() in ("site-packages", "dist-packages")])
    locations = [str(Path(path).resolve()) for path in candidates if Path(path).is_dir()]
    if not locations:
        raise RuntimeError("cannot find the calling environment's site-packages")
    base = Path(sys._base_executable).resolve(strict=True)
    # The random path is intentionally never created; -B prevents writes and
    # the fresh prefix prevents Python from reading stale source-adjacent pyc.
    cache_prefix = project / (".looking-glass-no-pyc-" + secrets.token_hex(16))
    if os.path.lexists(cache_prefix):
        raise RuntimeError("fresh bytecode-cache prefix unexpectedly exists")
    spec = json.dumps({"module": module, "project": str(project), "site_paths": locations,
                       "empty_cache_prefix": str(cache_prefix)},
                      separators=(",", ":"))
    bootstrap = ("import importlib.machinery,importlib.util,json,os,runpy,sys\n"
                 "s=json.loads(sys.argv.pop(1))\n"
                 "if not sys.flags.isolated or not sys.dont_write_bytecode or not sys.flags.ignore_environment "
                 "or not sys.flags.safe_path or not sys.flags.no_site "
                 "or sys.pycache_prefix != s['empty_cache_prefix'] "
                 "or os.path.lexists(s['empty_cache_prefix']):\n"
                 "    raise RuntimeError('fresh bytecode-cache isolation is unavailable')\n"
                 "sys.path.extend(s['site_paths'])\n"
                 "class Guard:\n"
                 "    @classmethod\n"
                 "    def find_spec(cls,name,path=None,target=None):\n"
                 "        spec=importlib.machinery.PathFinder.find_spec(name,path,target)\n"
                 "        if spec is not None and isinstance(spec.loader,importlib.machinery.SourcelessFileLoader):\n"
                 "            raise ImportError('sourceless Python imports are not permitted')\n"
                 "        if spec is not None and name.startswith('looking_glass.') and isinstance(spec.loader,importlib.machinery.ExtensionFileLoader):\n"
                 "            raise ImportError('native extensions cannot replace project source modules')\n"
                 "        return spec\n"
                 "sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder),Guard)\n"
                 "if s['module'].startswith('looking_glass.'):\n"
                 "    directory=os.path.join(s['project'],'looking_glass')\n"
                 "    p=importlib.util.spec_from_file_location('looking_glass',os.path.join(directory,'__init__.py'),submodule_search_locations=[directory])\n"
                 "    if p is None or not isinstance(p.loader,importlib.machinery.SourceFileLoader):\n"
                 "        raise ImportError('project package must be loaded from source')\n"
                 "    m=importlib.util.module_from_spec(p)\n"
                 "    sys.modules['looking_glass']=m\n"
                 "    p.loader.exec_module(m)\n"
                 "else:\n"
                 "    sys.path.append(s['project'])\n"
                 "runpy.run_module(s['module'],run_name='__main__',alter_sys=True)")
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return [str(base), "-I", "-B", "-X", "pycache_prefix=" + str(cache_prefix), "-S",
            "-c", bootstrap, spec, *arguments], environment


def _require_clean_python_runtime():
    """Require the source-only startup policy before CUDA-facing work."""
    if (not sys.flags.isolated or not sys.dont_write_bytecode or not sys.flags.no_site or not sys.flags.ignore_environment
            or not getattr(sys.flags, "safe_path", False) or not sys.pycache_prefix
            or os.path.lexists(sys.pycache_prefix)):
        raise RuntimeError("source-only Python startup with a fresh empty bytecode prefix is required")


def model_admission(free_bytes, total_bytes, model_bytes, trace_limit):
    """Apply a conservative estimate, not a guarantee against CUDA OOM."""
    if any(type(value) is not int or value <= 0
           for value in (free_bytes, total_bytes, model_bytes, trace_limit)) or free_bytes > total_bytes:
        raise ValueError("invalid CUDA memory admission inputs")
    reserve = max(1024 ** 3, (total_bytes * 15 + 99) // 100)
    increment = 2 * model_bytes + 256 * 1024 ** 2 + trace_limit
    if free_bytes < reserve + increment:
        raise RuntimeError("insufficient free device memory for model, trace headroom and reserve")
    return {"device_free_before_load_bytes": free_bytes, "device_total_bytes": total_bytes,
            "model_parameter_buffer_bytes": model_bytes, "reserve_bytes": reserve,
            "estimated_increment_bytes": increment,
            "limitation": "estimate is not a guarantee against CUDA out of memory"}


def runtime_guard(lease, cuda, device, reserve_bytes, trace_limit, owner_session=None):
    """Check cooperative ownership and current device-wide free headroom."""
    def check():
        lease.check()
        if owner_session is not None:
            owner_session.check(lease)
        free, total = cuda.mem_get_info(device)
        if free < reserve_bytes + trace_limit + 256 * 1024 ** 2:
            raise RuntimeError("device-wide free memory fell below the runtime reserve")
        return {"free_bytes": free, "total_bytes": total}
    return check


def create_source_bundle(project):
    """Snapshot runtime Python sources and protocol with exact bytes."""
    project = Path(project)
    paths = sorted((project / "looking_glass").rglob("*.py"))
    paths += [project / "docs" / "CUDA_BENCHMARK_V1.md", project / "pyproject.toml"]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("benchmark runtime source or protocol is unavailable")
    archive = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            name = path.relative_to(project).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content)
    return archive.getvalue(), hashes


def make_schedule(seed, rounds):
    """Interleave all modes once per round in deterministic shuffled order."""
    if type(seed) is not int or type(rounds) is not int or rounds <= 0:
        raise ValueError("seed must be an integer and rounds a positive integer")
    generator = random.Random(seed)
    rows = []
    for round_number in range(rounds):
        modes = list(MODES)
        generator.shuffle(modes)
        rows.extend({"round": round_number, "position": position, "mode": mode}
                    for position, mode in enumerate(modes))
    return rows


def summarize(rows):
    """Derive descriptive medians directly from the saved raw measurements."""
    if not rows:
        raise ValueError("at least one timing sample is required")
    result = []
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        if subset:
            result.append({"mode": mode, "samples": len(subset),
                           "median_wall_us": statistics.median(row["wall_ns"] for row in subset) / 1000,
                           "median_cuda_ms": statistics.median(row["cuda_ms"] for row in subset)})
    return result


def summarize_paired_ratios(rows):
    """Compare modes within the same seeded and interleaved measurement round."""
    rounds = {}
    for row in rows:
        values = rounds.setdefault((row["seed"], row["round"]), {})
        if row["mode"] in values or row["wall_ns"] <= 0:
            raise ValueError("invalid or duplicate timing sample")
        values[row["mode"]] = row["wall_ns"]
    if not rounds or any(set(values) != set(MODES) for values in rounds.values()):
        raise ValueError("every measured round must contain all four modes")
    return {"paired_rounds": len(rounds),
            "median_id_over_none": statistics.median(v["id"] / v["none"] for v in rounds.values()),
            "median_direct_over_none": statistics.median(v["direct"] / v["none"] for v in rounds.values()),
            "median_gradient_over_direct": statistics.median(v["gradient"] / v["direct"] for v in rounds.values())}


def parse_telemetry_line(line, observed_ns):
    fields = [value.strip() for value in next(csv.reader([line]))]
    if len(fields) != 6 or not fields[1].startswith("GPU-"):
        raise ValueError("invalid nvidia-smi telemetry row")
    try:
        total, used, free, utilization = (int(value) for value in fields[2:])
    except ValueError as exc:
        raise ValueError("unavailable nvidia-smi telemetry value") from exc
    if total <= 0 or min(used, free, utilization) < 0 or used > total or free > total or utilization > 100:
        raise ValueError("invalid nvidia-smi telemetry value")
    return {"observed_monotonic_ns": observed_ns, "gpu_timestamp": fields[0],
            "gpu_uuid": fields[1], "total_bytes": total * 1024 * 1024,
            "used_bytes": used * 1024 * 1024, "free_bytes": free * 1024 * 1024,
            "utilization_percent": utilization}


def classify_isolation(exclusive, observations, started_ns, ended_ns, process_queries,
                       own_pid, gpu_uuid, telemetry_errors):
    """Apply the prespecified idle screen without claiming proof of exclusivity."""
    def valid_metrics(row):
        if not isinstance(row, dict) or not isinstance(row.get("gpu_uuid"), str):
            return False
        total, used, free, utilization, observed = (row.get(name) for name in
            ("total_bytes", "used_bytes", "free_bytes", "utilization_percent", "observed_monotonic_ns"))
        return (all(type(value) is int for value in (total, used, free, utilization, observed))
                and total > 0 and 0 <= used <= total and 0 <= free <= total
                and 0 <= utilization <= 100 and observed >= 0)
    valid_rows = [row for row in observations if valid_metrics(row)]
    selected = sorted((row for row in valid_rows if row["gpu_uuid"] == gpu_uuid),
                      key=lambda row: row["observed_monotonic_ns"])
    before = [row for row in selected if row["observed_monotonic_ns"] < started_ns][-3:]
    after = [row for row in selected
             if type(row.get("query_started_monotonic_ns")) is int
             and row["query_started_monotonic_ns"] >= ended_ns + 1_000_000_000][-3:]
    unavailable, contaminated = [], []
    if len(valid_rows) != len(observations):
        unavailable.append("telemetry contains invalid memory or utilization values")
    if not exclusive:
        unavailable.append("exclusive reservation was not attested")
    if len(before) < 3 or len(after) < 3:
        unavailable.append("fewer than three idle samples before or after workload")
    if any(row["gpu_uuid"] != gpu_uuid for row in valid_rows) or not selected:
        unavailable.append("telemetry GPU identity does not match the selected device")
    if len({row["total_bytes"] for row in selected}) > 1:
        unavailable.append("telemetry reported inconsistent total device memory")
    if telemetry_errors:
        unavailable.append("telemetry query failed")
    if before and after:
        window = [row for row in selected
                  if before[0]["observed_monotonic_ns"] <= row["observed_monotonic_ns"] <= after[-1]["observed_monotonic_ns"]]
        def valid_bounds(row):
            start, end = row.get("query_started_monotonic_ns"), row.get("query_ended_monotonic_ns")
            return (type(start) is int and type(end) is int
                    and start <= end == row["observed_monotonic_ns"])
        if not all(valid_bounds(row) for row in window):
            unavailable.append("telemetry query time bounds are missing or invalid")
        elif any(later["query_ended_monotonic_ns"] - earlier["query_started_monotonic_ns"] > 2_000_000_000
                 for earlier, later in zip(window, window[1:])):
            unavailable.append("telemetry coverage may have a gap longer than two seconds")
    if len(process_queries) != 2 or not all(query.get("ok") for query in process_queries):
        unavailable.append("compute process list unavailable")
    if any(row["utilization_percent"] > 5 for row in before + after):
        contaminated.append("idle utilization exceeded five percent")
    foreign = sorted({pid for query in process_queries for pid in query.get("pids", []) if pid != own_pid})
    if foreign:
        contaminated.append("other compute processes were observed")
    status = "contaminated" if contaminated else ("unverified" if unavailable else "no_contamination_observed")
    return {"status": status, "reasons": contaminated + unavailable,
            "before_idle_samples": before, "after_idle_samples": after,
            "foreign_compute_pids": foreign,
            "limitation": "sampling and endpoint-only WDDM process reporting can miss activity"}


def _command(args, timeout=5):
    """Execute a bounded read-only telemetry command without opening a console."""
    return subprocess.run(args, capture_output=True, text=True, errors="replace",
                          timeout=timeout, check=False,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _gpu_inventory(smi, cuda_count=None):
    """Inspect the sole physical device; verify CUDA count when available."""
    if cuda_count is not None and cuda_count != 1:
        raise RuntimeError("this protocol requires exactly one CUDA-visible device")
    if not smi:
        return {"ok": False, "error": "nvidia-smi unavailable", "uuid": None}
    try:
        command = _command([smi, "--query-gpu=uuid,name,memory.total,driver_version",
                            "--format=csv,noheader,nounits"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "uuid": None}
    if command.returncode:
        return {"ok": False, "error": command.stderr.strip(), "uuid": None}
    rows = list(csv.reader(command.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError("cannot unambiguously map CUDA ordinal to a single physical GPU")
    uuid, name, total, driver = (part.strip() for part in rows[0])
    if not uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("MIG-"):
        raise RuntimeError("unsupported or ambiguous GPU mapping")
    try:
        total_bytes = int(total) * 1024 * 1024
    except ValueError as exc:
        raise RuntimeError("GPU inventory did not report total memory") from exc
    if total_bytes <= 0:
        raise RuntimeError("GPU inventory reported invalid total memory")
    return {"ok": True, "uuid": uuid, "name": name, "total_bytes": total_bytes,
            "driver_version": driver, "mapping": "sole physical GPU; CUDA binding verified separately"}


def resolve_benchmark_identity(workspace_root, smi):
    """Derive the shared physical-GPU lock and marker without touching CUDA."""
    from .device_identity import resolve_device_identity
    inventory = _gpu_inventory(smi)
    if not inventory.get("ok") or not inventory.get("uuid"):
        raise RuntimeError("cannot establish the physical GPU UUID before CUDA")
    identity = resolve_device_identity(workspace_root, inventory["uuid"],
                                       physical_uuids=(inventory["uuid"],))
    return identity, inventory


def _cuda_runtime_uuid(value):
    """Normalize PyTorch's bare CUuuid or a complete NVIDIA GPU UUID."""
    from .device_identity import normalize_gpu_uuid
    if value is None:
        raise ValueError("CUDA runtime did not provide a device UUID")
    value = str(value)
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    value, re.IGNORECASE):
        value = "GPU-" + value
    return normalize_gpu_uuid(value)


def _query_compute_processes(smi, gpu_uuid, phase):
    result = {"phase": phase, "observed_monotonic_ns": time.perf_counter_ns(),
              "ok": False, "pids": [], "processes": []}
    if not smi or not gpu_uuid:
        result["error"] = "nvidia-smi or GPU UUID unavailable"
        return result
    try:
        command = _command([smi, "--query-compute-apps=gpu_uuid,pid,process_name",
                            "--format=csv,noheader,nounits"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
        return result
    result["returncode"] = command.returncode
    result["stderr"] = command.stderr.strip()
    if command.returncode:
        result["error"] = "compute process query failed"
        return result
    try:
        for row in csv.reader(command.stdout.splitlines()):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 3:
                raise ValueError("unexpected compute process row")
            uuid, pid, name = (part.strip() for part in row)
            if uuid == gpu_uuid:
                result["pids"].append(int(pid))
                result["processes"].append({"pid": int(pid), "name": name})
        result["ok"] = True
    except ValueError as exc:
        result["error"] = str(exc)
        result["raw_stdout"] = command.stdout
    return result


class _NvidiaMonitor:
    """Sample device-wide telemetry outside the PyTorch allocator/context."""
    def __init__(self, smi, interval_seconds=0.2):
        self.smi = smi
        self.interval_seconds = interval_seconds
        self.observations, self.errors = [], []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not self.smi:
            self.errors.append({"observed_monotonic_ns": time.perf_counter_ns(),
                                "error": "nvidia-smi unavailable"})
            return
        self._thread = threading.Thread(target=self._poll, name="cuda-benchmark-telemetry", daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            started = time.perf_counter_ns()
            try:
                command = _command([self.smi,
                    "--query-gpu=timestamp,uuid,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits"])
                ended = time.perf_counter_ns()
                if command.returncode:
                    raise RuntimeError(command.stderr.strip() or "telemetry command failed")
                rows = [parse_telemetry_line(line, ended) for line in command.stdout.splitlines() if line.strip()]
                if not rows:
                    raise ValueError("telemetry query returned no GPU rows")
                for row in rows:
                    row["query_started_monotonic_ns"] = started
                    row["query_ended_monotonic_ns"] = ended
                self.observations.extend(rows)
            except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                self.errors.append({"query_started_monotonic_ns": started,
                                    "observed_monotonic_ns": time.perf_counter_ns(),
                                    "error_type": type(exc).__name__, "message": str(exc)})
            self._stop.wait(self.interval_seconds)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                self.errors.append({"observed_monotonic_ns": time.perf_counter_ns(),
                                    "error": "telemetry thread did not stop"})
        return {"configured_interval_seconds": self.interval_seconds,
                "observations": list(self.observations), "errors": list(self.errors),
                "limitations": ["nvidia-smi polling can perturb latency",
                    "query timestamps bound collection rather than identify its exact instant",
                    "WDDM and endpoint-only process reporting can miss activity"]}


def verify_equivalence(invoke, session_factory, identity, completion=None):
    """Require matching outputs and captures under torch.equal before timing.

    This is numeric equality without a tolerance, not stored-bit identity.
    In particular, signed zeros compare equal and subnormal handling can
    depend on the execution backend.
    """
    import torch

    sessions = {}
    with torch.no_grad():
        baseline = invoke()
        outputs = [baseline]
        if not bool(torch.isfinite(baseline).all()):
            raise AssertionError("baseline model output contains non-finite values")
        for mode in ("id", "direct", "gradient"):
            with session_factory(mode) as session:
                output = invoke()
            outputs.append(output)
            if not bool(torch.isfinite(output).all()):
                raise AssertionError(f"model output for mode {mode} contains non-finite values")
            if (baseline.dtype != output.dtype or baseline.device != output.device
                    or baseline.shape != output.shape or not torch.equal(baseline, output)):
                raise AssertionError(f"model outputs differ for mode {mode}")
            sessions[mode] = session

    reference = sessions["direct"].events
    if not reference:
        raise AssertionError("no capture events were observed")
    keys = [identity(event) for event in reference]
    if len(set(keys)) != len(keys):
        raise AssertionError("capture identities must be unique")
    for mode, session in sessions.items():
        if [identity(event) for event in session.events] != keys:
            raise AssertionError(f"capture identities differ for mode {mode}")
        metadata = [(tuple(event.shape), event.dtype, event.device) for event in session.events]
        reference_metadata = [(tuple(event.shape), event.dtype, event.device) for event in reference]
        if metadata != reference_metadata:
            raise AssertionError(f"source metadata differs for mode {mode}")
    if any(event.value is not None for event in sessions["id"].events):
        raise AssertionError("id capture must not save values")
    for direct, gradient in zip(reference, sessions["gradient"].events):
        d, g = direct.value, gradient.value
        if d is None or g is None:
            raise AssertionError("direct and gradient values are missing")
        if d.dtype != g.dtype:
            raise AssertionError("direct and gradient captured dtype differs")
        if d.device != g.device or d.shape != g.shape:
            raise AssertionError("direct and gradient captured device or shape differs")
        if not torch.equal(d, g):
            raise AssertionError("direct and gradient values differ")
    if sessions["direct"].tensor_bytes != sessions["gradient"].tensor_bytes:
        raise AssertionError("direct and gradient retained bytes differ")
    # A caller that uses a device must synchronize while the baseline, final
    # output and both captured sessions are still rooted in this frame.
    if completion is not None:
        completion()
    return {"events_per_mode": len(reference),
            "value_tensor_bytes": sessions["direct"].tensor_bytes,
            "event_identities": keys,
            "value_layouts": [{"shape": list(event.value.shape), "dtype": str(event.value.dtype),
                               "device": str(event.value.device)} for event in reference],
            "comparison": {"operator": "torch.equal", "tolerance": "none",
                           "guarantees_bitwise_identity": False,
                           "subnormal_handling": "backend-dependent"},
            "outputs_finite": True,
            "outputs_exact": True, "values_exact": True}


def _validate_memory_snapshot(free, total, allocated, reserved):
    if (any(type(value) is not int for value in (free, total, allocated, reserved))
            or total <= 0 or not 0 <= free <= total or not 0 <= allocated <= reserved <= total):
        raise ValueError("invalid CUDA memory snapshot")


def validate_measurement(row):
    """Reject internally contradictory timing or process allocator counters."""
    total = row["device_total_bytes"]
    before = (row["device_free_before_bytes"], total, row["allocated_before_bytes"],
              row["reserved_before_bytes"])
    after = (row["device_free_after_bytes"], total, row["allocated_after_bytes"],
             row["reserved_after_bytes"])
    _validate_memory_snapshot(*before)
    _validate_memory_snapshot(*after)
    peak_allocated, peak_reserved = row["peak_allocated_bytes"], row["peak_reserved_bytes"]
    if (type(peak_allocated) is not int or type(peak_reserved) is not int
            or not 0 <= peak_allocated <= peak_reserved <= total
            or peak_allocated < max(before[2], after[2])
            or peak_reserved < max(before[3], after[3])):
        raise ValueError("invalid CUDA process allocator peaks")
    start, end, wall, event_ms = (row[name] for name in
        ("started_monotonic_ns", "ended_monotonic_ns", "wall_ns", "cuda_ms"))
    if (type(start) is not int or type(end) is not int or type(wall) is not int
            or start < 0 or wall <= 0 or end - start != wall
            or type(event_ms) not in (int, float) or not math.isfinite(event_ms) or event_ms < 0):
        raise ValueError("invalid CUDA timing measurement")
    if any(type(row[name]) is not int or row[name] < 0
           for name in ("events", "recorded_tensor_bytes")):
        raise ValueError("invalid CUDA trace measurement")


def measure_sample(invoke, session_factory, mode, device, cuda, start_event, end_event,
                   validate=None, memory_check=None):
    """Measure one operation with the trace alive across its completion boundary."""
    import torch

    if mode not in MODES:
        raise ValueError("unsupported capture mode")
    cuda.synchronize(device)
    stream = cuda.current_stream(device)
    if stream != cuda.default_stream(device):
        raise RuntimeError("benchmark sample requires the selected device's default stream")
    cuda.reset_peak_memory_stats(device)
    free_before, total = cuda.mem_get_info(device)
    allocated_before = cuda.memory_allocated(device)
    reserved_before = cuda.memory_reserved(device)
    _validate_memory_snapshot(free_before, total, allocated_before, reserved_before)
    if memory_check is not None:
        memory_check(free_before, total, allocated_before, reserved_before)
    started_ns = time.perf_counter_ns()
    start_event.record(stream)
    session = None
    with torch.no_grad():
        if mode == "none":
            output = invoke()
            events, value_bytes = 0, 0
        else:
            session = session_factory(mode)
            with session:
                output = invoke()
            events, value_bytes = len(session.events), session.tensor_bytes
    if validate is not None:
        validate(mode, output, session)
    output_metadata = {"output_shape": list(output.shape), "output_dtype": str(output.dtype),
                       "output_device": str(output.device)}
    if cuda.current_stream(device) != stream:
        raise RuntimeError("benchmark sample changed the selected device's current stream")
    end_event.record(stream)
    cuda.synchronize(device)
    cuda_ms = start_event.elapsed_time(end_event)
    del output, session
    cuda.synchronize(device)
    ended_ns = time.perf_counter_ns()
    free_after, total_after = cuda.mem_get_info(device)
    if total != total_after:
        raise RuntimeError("CUDA device total memory changed during a sample")
    allocated_after, reserved_after = cuda.memory_allocated(device), cuda.memory_reserved(device)
    row = {"started_monotonic_ns": started_ns, "ended_monotonic_ns": ended_ns,
            "wall_ns": ended_ns - started_ns, "cuda_ms": cuda_ms,
            "events": events, "recorded_tensor_bytes": value_bytes,
            **output_metadata,
            "cuda_stream_is_default": True,
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": cuda.max_memory_allocated(device),
            "peak_reserved_bytes": cuda.max_memory_reserved(device),
            "allocated_after_bytes": allocated_after,
            "reserved_after_bytes": reserved_after,
            "device_free_before_bytes": free_before,
            "device_free_after_bytes": free_after,
            "device_total_bytes": total}
    validate_measurement(row)
    return row


def collect_trials(invoke, factory, device, cuda, start_event, end_event,
                   seed, warmup_rounds, measured_rounds, progress, guard=None, validate=None):
    """Retain warmup and measured rows separately in their actual order."""
    for kind, schedule_seed, rounds in (("warmup", seed ^ 0x5A17, warmup_rounds),
                                        ("measured", seed, measured_rounds)):
        for row in make_schedule(schedule_seed, rounds):
            if guard is not None:
                guard()
            measurement = measure_sample(invoke, factory, row["mode"], device, cuda,
                                         start_event, end_event, validate=validate)
            sample = {"seed": seed, "schedule_seed": schedule_seed, "sample_kind": kind, **row,
                      "phase": ("forward_only_with_output_validation" if row["mode"] == "none"
                                else "forward_plus_capture_with_validation"),
                      **measurement}
            if kind == "warmup":
                record = progress.get("record_warmup")
                if record is None:
                    progress.setdefault("warmups", []).append(sample)
                else:
                    record(sample)
            else:
                progress.get("record_sample", progress["raw"].append)(sample)


def _git_provenance(project):
    result = {}
    for name, arguments in (("commit", ["rev-parse", "HEAD"]),
                            ("status_porcelain", ["status", "--porcelain=v1", "--untracked-files=all"])):
        try:
            command = _command(["git", "-C", str(project), *arguments], timeout=10)
            if command.returncode:
                result[name + "_error"] = command.stderr.strip()
            else:
                result[name] = command.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name + "_error"] = str(exc)
    try:
        diff = subprocess.run(["git", "-C", str(project), "diff", "--binary", "HEAD"],
                              capture_output=True, timeout=10, check=False,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if diff.returncode:
            result["tracked_diff_error"] = diff.stderr.decode("utf-8", errors="replace")
        else:
            result["tracked_diff_sha256"] = hashlib.sha256(diff.stdout).hexdigest()
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["tracked_diff_error"] = str(exc)
    return result


def _loaded_module_hashes():
    """Record observable Python/native module files, without embedding their paths."""
    result, cached = {}, {}
    for name, module in sorted(list(sys.modules.items())):
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str) or Path(filename).suffix.lower() not in (".py", ".pyd", ".so"):
            continue
        try:
            if filename not in cached:
                digest = hashlib.sha256()
                with open(filename, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                cached[filename] = digest.hexdigest()
            result[name] = cached[filename]
        except OSError as exc:
            result[name] = {"unavailable": type(exc).__name__}
    return result


def _wait_for_observations(monitor, threshold_ns=None, timeout=5.0):
    if not monitor.smi:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = monitor.observations
        selected = rows if threshold_ns is None else [
            row for row in rows
            if type(row.get("query_started_monotonic_ns")) is int
            and row["query_started_monotonic_ns"] >= threshold_ns]
        if len(selected) >= 3 or monitor.errors:
            return
        time.sleep(0.05)


def _transformer_workload(model, input_ids):
    from .transformer_trace import GPT2TraceSession

    def invoke():
        return model(input_ids=input_ids, use_cache=False, return_dict=True).logits

    def factory(mode):
        return GPT2TraceSession(model, run_id=PROTOCOL, mode=mode,
            targets=("block", "attn", "mlp"), layers=None, samples=(0,), tokens=(0, 3, 7),
            capture_device="source", max_events=32, max_trace_bytes=4 * 1024 ** 2)

    def identity(event):
        return (event.module_path, event.module_kind, event.layer_index, event.forward_index,
                event.invocation, tuple(event.output_path), tuple(event.batch_positions),
                tuple(event.token_positions))

    return invoke, factory, identity


def validate_transformer_sample(mode, output, session, *, identity, identities, device):
    """Check the frozen output and every captured location during each trial."""
    import torch

    if mode not in MODES or len(identities) != 18:
        raise AssertionError("sample has an invalid mode or expected event count")
    if (not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 8, 50257)
            or output.dtype != torch.float32 or str(output.device) != str(device)):
        raise AssertionError("sample output does not match the frozen tensor layout")
    if not bool(torch.isfinite(output).all()):
        raise AssertionError("sample output contains non-finite values")
    if mode == "none":
        if session is not None:
            raise AssertionError("uninstrumented sample unexpectedly has a trace session")
        return
    if session is None or session.successful is not True:
        raise AssertionError("sample trace session did not complete successfully")
    events = session.events
    if len(events) != 18 or tuple(identity(event) for event in events) != tuple(identities):
        raise AssertionError("sample trace locations differ from the frozen preflight")
    expected_bytes = 0 if mode == "id" else 18 * 3 * 768 * 4
    if session.tensor_bytes != expected_bytes:
        raise AssertionError("sample trace bytes differ from the frozen capture")
    for index, event in enumerate(events):
        if (event.sequence != index or event.run_id != PROTOCOL or event.capture_mode != mode
                or tuple(event.shape) != (1, 8, 768) or event.dtype != "torch.float32"
                or event.device != str(device)):
            raise AssertionError("sample trace source metadata differs from the frozen capture")
        value = event.value
        if mode == "id":
            if value is not None or event.tensor_bytes != 0:
                raise AssertionError("ID sample unexpectedly retained a value")
        elif (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 3, 768)
              or value.dtype != torch.float32 or str(value.device) != str(device)
              or event.tensor_bytes != 3 * 768 * 4):
            raise AssertionError("sample trace value layout differs from the frozen capture")


def validate_transformer_coverage(identities):
    """Require every frozen layer/role exactly once at the selected positions."""
    expected = {(f"transformer.h.{layer}" + ("" if kind == "block" else f".{kind}"), kind, layer)
                for layer in range(6) for kind in ("block", "attn", "mlp")}
    observed = []
    for entry in identities:
        if len(entry) != 8:
            raise AssertionError("preflight coverage has malformed identity")
        path, kind, layer, forward, invocation, output, samples, tokens = entry
        if forward != 0 or invocation != 0 or samples != (0,) or tokens != (0, 3, 7):
            raise AssertionError("preflight coverage has unexpected invocation or positions")
        if output != ((0,) if kind in ("block", "attn") else ()):
            raise AssertionError("preflight coverage has an unexpected output selector")
        observed.append((path, kind, layer))
    if len(observed) != len(expected) or set(observed) != expected:
        raise AssertionError("preflight coverage does not match every frozen transformer location")


def validate_transformer_layouts(layouts):
    expected = {"shape": [1, 3, 768], "dtype": "torch.float32", "device": "cuda:0"}
    if len(layouts) != 18 or any(layout != expected for layout in layouts):
        raise AssertionError("preflight saved tensor layout does not match the frozen source-device capture")


def stage_model_to_cuda(model, device, stage, progress, prepare=None):
    """Persist model/device identity before an allocation or preflight can fail."""
    import torch
    progress["metadata"]["preload"] = stage
    progress["record_stage"]("preload", stage)
    if prepare is not None:
        prepare()
    model.to(device=device, dtype=torch.float32)


def _acquire_device_lease(path, timeout):
    """Keep the cooperative lease until this dedicated process exits."""
    if path is None or not Path(path).is_absolute():
        raise ValueError("an absolute shared device lease path is required")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout <= 60:
        raise ValueError("lease timeout must be between zero and 60 seconds")
    from .device_lease import DeviceLease
    lease = DeviceLease.acquire(path, timeout=timeout)
    _PROCESS_LEASES.append(lease)
    return lease


def _validate_owner_session(session):
    from .supervisor import OwnerSession
    if not isinstance(session, OwnerSession):
        raise RuntimeError("a real supervised owner session is required")


def _finish_owner(session, lease, cuda):
    """Attest only after the bound owner has drained and released allocations."""
    if lease is None or not cuda.is_initialized():
        raise RuntimeError("the GPU owner did not establish a verifiable CUDA lifecycle")
    lease.check()
    session.check(lease)
    gc.collect()
    cuda.synchronize(0)
    cuda.empty_cache()
    cuda.synchronize(0)
    if cuda.memory_allocated(0) or cuda.memory_reserved(0):
        raise RuntimeError("CUDA allocations remain after owner cleanup")
    session.attest_clean(lease)


def _evaluate(quick, exclusive, progress, cache_dir, lease_path, lease_timeout):
    project = Path(__file__).resolve().parent.parent
    bundle, hashes = create_source_bundle(project)
    progress["source_bundle"] = bundle
    progress["metadata"]["repository_source_sha256"] = hashes
    progress["metadata"]["source_bundle_sha256"] = hashlib.sha256(bundle).hexdigest()
    progress["record_source"](bundle)
    owner_session = progress.get("owner_session")
    if not progress.get("candidate_only") or owner_session is None:
        raise RuntimeError("GPU benchmarks require a supervised owner subprocess")
    _validate_owner_session(owner_session)
    if (progress.get("cpu_preflight") or {}).get("source_bundle_sha256") != hashlib.sha256(bundle).hexdigest():
        raise RuntimeError("CPU preflight and GPU owner source bundles differ")
    if not GPU_BENCHMARK_ADMISSION_READY:
        raise RuntimeError("reviewed GPU admission for the eight-token trace workload is not available")
    _require_clean_python_runtime()
    lease = _acquire_device_lease(lease_path, lease_timeout)
    lease.check()
    owner_session.check(lease)
    if getattr(owner_session, "identity", None) is None:
        raise RuntimeError("supervised GPU owner has no verified device identity")
    if os.path.normcase(str(Path(lease_path).resolve())) != os.path.normcase(str(owner_session.device_lease_path)):
        raise RuntimeError("benchmark lease does not match the supervised device identity")
    def final_guard():
        lease.check()
        owner_session.check(lease)
    progress["final_guard"] = final_guard
    progress["metadata"]["device_lease_path"] = str(lease_path)
    workspace_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace_config not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG is incompatible with deterministic CUDA")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("benchmark must start as a fresh standalone process before CUDA initialization")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the installed PyTorch build")
    if cache_dir is None:
        raise ValueError("an explicit offline checkpoint cache path is required")
    from .pretrained import load_distilgpt2
    import transformers

    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    smi = shutil.which("nvidia-smi")
    device_count = torch.cuda.device_count()
    inventory = _gpu_inventory(smi, device_count)
    properties = torch.cuda.get_device_properties(0)
    runtime_uuid = _cuda_runtime_uuid(getattr(properties, "uuid", None))
    owner_session.verify_runtime_binding(lease, runtime_uuid=runtime_uuid,
                                         device_count=device_count, cuda_device_index=0)
    from .device_identity import normalize_gpu_uuid
    if not inventory.get("ok") or normalize_gpu_uuid(inventory["uuid"]) != owner_session.identity.gpu_uuid:
        raise RuntimeError("nvidia-smi and supervised CUDA device identity disagree")
    loaded = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
    preflight_metadata = (progress.get("cpu_preflight") or {}).get("model_load_metadata")
    if preflight_metadata is None or any(
            preflight_metadata.get(key) != loaded.metadata.get(key)
            for key in ("model_id", "model_revision", "asset_sha256")):
        raise RuntimeError("the CPU preflight and GPU-owner checkpoint identities differ")
    model = loaded.model
    device = torch.device("cuda:0")
    token_ids = [15496, 11, 616, 1438, 318, 257, 1332, 13]
    seeds = [17] if quick else [17, 29, 43, 71]
    warmup_rounds, measured_rounds = (1, 3) if quick else (10, 30)
    token_encoding = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    config = {"model_mode": "eval", "dtype": "float32", "device": "cuda:0",
        "grad_mode": "no_grad", "use_cache": False, "batch_size": 1,
        "token_ids": token_ids, "token_ids_sha256": hashlib.sha256(token_encoding).hexdigest(),
        "targets": ["block", "attn", "mlp"], "layers": list(range(6)),
        "samples": [0], "tokens": [0, 3, 7], "max_events": 32,
        "max_trace_bytes": 4 * 1024 ** 2, "capture_device": "source",
        "seeds": seeds, "warmup_rounds_per_seed": warmup_rounds,
        "torch_rng_seed": 20261028,
        "measured_rounds_per_seed": measured_rounds, "warmup_seed_xor": 0x5A17,
        "cuda_event_stream_policy": "selected-device default stream checked before and after each sample",
        "monitor_interval_seconds": 0.2, "journal_fsync_each_sample": True}
    with torch.cuda.device(device):
        free, total = torch.cuda.mem_get_info(device)
        model_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
        probe = {"protocol": PROTOCOL, "device_free_bytes": free, "device_total_bytes": total,
                 "model_parameter_buffer_bytes": model_bytes, "model_load_metadata": json.loads(json.dumps(loaded.metadata)),
                 "nvidia_inventory": inventory, "config": config,
                 "source_bundle_sha256": progress["metadata"]["source_bundle_sha256"]}
        progress["metadata"]["device_probe"] = probe
        progress["record_stage"]("device_probe", probe)
        admission = model_admission(free, total, model_bytes, 4 * 1024 ** 2)
        fraction = (total - admission["reserve_bytes"]) / total
        admission["pytorch_allocator_fraction_limit"] = fraction
        environment = {
            "python_version": platform.python_version(), "platform": platform.platform(),
            "monotonic_clock": {"name": "perf_counter_ns",
                                "implementation": time.get_clock_info("perf_counter").implementation,
                                "resolution_seconds": time.get_clock_info("perf_counter").resolution,
                                "monotonic": time.get_clock_info("perf_counter").monotonic},
            "python_executable": sys.executable, "python_base_executable": sys._base_executable,
            "python_prefix": sys.prefix, "python_base_prefix": sys.base_prefix,
            "python_search_path": list(sys.path),
            "python_original_argv": list(sys.orig_argv),
            "python_pycache_prefix": sys.pycache_prefix,
            "python_bytecode_writes_disabled": sys.dont_write_bytecode,
            "python_no_site": bool(sys.flags.no_site),
            "python_isolated": bool(sys.flags.isolated),
            "python_ignore_environment": bool(sys.flags.ignore_environment),
            "python_safe_path": bool(sys.flags.safe_path),
            "python_hash_randomization": bool(sys.flags.hash_randomization),
            "torch_module_file": torch.__file__, "transformers_module_file": transformers.__file__,
            "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
            "torch_cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_name": properties.name,
            "cuda_runtime_uuid": runtime_uuid,
            "cuda_compute_capability": [properties.major, properties.minor],
            "cuda_total_memory_bytes": properties.total_memory,
            "visibility_policy": ("default sole physical GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") is None
                                  else "one explicit full GPU UUID"),
            "nvidia_inventory": inventory, "num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": workspace_config,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        stage = {"protocol": PROTOCOL, "protocol_mode": "smoke" if quick else "full",
            "config": config, "model_load_metadata": json.loads(json.dumps(loaded.metadata)),
            "cpu_preflight": progress["cpu_preflight"],
            "environment": environment, "admission": admission,
            "source": progress["metadata"].copy(), "git": _git_provenance(project)}
        if model.training or model.config.n_layer != 6 or model.config.n_embd != 768:
            progress["metadata"]["preload"] = stage
            progress["record_stage"]("preload", stage)
            raise RuntimeError("pretrained checkpoint does not match the frozen workload")
        guard = runtime_guard(lease, torch.cuda, device, admission["reserve_bytes"], 4 * 1024 ** 2, owner_session)
        def prepare_device():
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            guard()
        stage_model_to_cuda(model, device, stage, progress, prepare=prepare_device)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        invoke, factory, identity = _transformer_workload(model, input_ids)
        preflight = verify_equivalence(invoke, factory, identity)
        validate_transformer_coverage(preflight["event_identities"])
        validate_transformer_layouts(preflight["value_layouts"])
        expected_events = 6 * 3
        expected_bytes = expected_events * 3 * 768 * 4
        if preflight["events_per_mode"] != expected_events or preflight["value_tensor_bytes"] != expected_bytes:
            raise AssertionError("preflight trace does not match the frozen event and byte counts")
        def validate_sample(mode, output, session):
            validate_transformer_sample(mode, output, session, identity=identity,
                                        identities=preflight["event_identities"], device=device)
        torch.cuda.synchronize(device)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        prime_stream = torch.cuda.current_stream(device)
        if prime_stream != torch.cuda.default_stream(device):
            raise RuntimeError("benchmark event priming requires the selected device's default stream")
        start_event.record(prime_stream)
        end_event.record(prime_stream)
        torch.cuda.synchronize(device)
        manifest = {**stage,
            "preflight": preflight, "loaded_module_sha256_before_trials": _loaded_module_hashes(),
            "limitations": ["ordering seeds are correlated repeats on one model and one input",
                "allocator peaks exclude CUDA context and other processes",
                "sampled device-wide memory extrema are not true device peaks",
                "native DLL dependencies are not comprehensively hashed, so exact environment replay is not guaranteed"]}
        progress["metadata"]["manifest"] = manifest
        progress["record_manifest"](manifest)
        monitor = _NvidiaMonitor(smi)
        monitor.start()
        before_query = None
        after_query = None
        workload_started = None
        workload_ended = None
        try:
            _wait_for_observations(monitor)
            before_query = _query_compute_processes(smi, inventory.get("uuid"), "before")
            workload_started = time.perf_counter_ns()
            for seed in seeds:
                collect_trials(invoke, factory, device, torch.cuda, start_event, end_event,
                               seed, warmup_rounds, measured_rounds, progress, guard=guard,
                               validate=validate_sample)
            guard()
            workload_ended = time.perf_counter_ns()
            _wait_for_observations(monitor, threshold_ns=workload_ended + 1_000_000_000)
            after_query = _query_compute_processes(smi, inventory.get("uuid"), "after")
        finally:
            progress["telemetry"] = monitor.stop()
            progress["telemetry"]["process_queries"] = [query for query in (before_query, after_query) if query is not None]
            progress["telemetry"]["workload_started_monotonic_ns"] = workload_started
            progress["telemetry"]["workload_ended_monotonic_ns"] = workload_ended

        telemetry = progress["telemetry"]
        isolation = classify_isolation(exclusive, telemetry["observations"], workload_started,
            workload_ended, [before_query, after_query], os.getpid(), inventory.get("uuid"), telemetry["errors"])
        selected = [row for row in telemetry["observations"] if row["gpu_uuid"] == inventory.get("uuid")]
        telemetry_summary = {"samples": len(selected),
            "device_wide_min_sampled_free_bytes": min((row["free_bytes"] for row in selected), default=None),
            "device_wide_max_sampled_used_bytes": max((row["used_bytes"] for row in selected), default=None),
            "process_queries": [before_query, after_query], "isolation": isolation,
            "extrema_scope": "all telemetry observations; sampled, not a true device-wide peak"}
        _, after_hashes = create_source_bundle(project)
        if after_hashes != hashes:
            raise RuntimeError("benchmark repository source changed during the run")
        module_hashes = _loaded_module_hashes()
        final_guard()
        return {"protocol": PROTOCOL, "protocol_mode": "smoke" if quick else "full",
            "manifest": manifest, "raw": progress["raw"], "timing_summary": summarize(progress["raw"]),
            "warmup_samples": len(progress["warmups"]),
            "paired_ratios": summarize_paired_ratios(progress["raw"]),
            "telemetry_summary": telemetry_summary,
            "loaded_module_sha256_after_trials": module_hashes,
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def _write_json(path, value):
    encoded = (json.dumps(_public_evidence(value), indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes(path, encoded)


def _write_all(handle, value):
    """Write all bytes or fail; a successful short write is not a full record."""
    view = memoryview(value)
    while view:
        count = handle.write(view)
        if type(count) is not int or count <= 0 or count > len(view):
            raise OSError("evidence write made no valid progress")
        view = view[count:]


def _write_bytes(path, value):
    temporary = path.with_name(path.name + ".partial")
    created, linked = False, False
    original = None
    try:
        with temporary.open("xb") as handle:
            created = True
            _write_all(handle, value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
    except BaseException as exc:
        original = exc
        # A signal can arrive just after the link succeeded but before Python
        # assigned linked. Once published, the destination is authoritative.
        if created and not linked:
            try:
                linked = os.path.samefile(path, temporary)
            except OSError:
                pass
    if created:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A failed removal leaves unsealed scratch, not a new outcome.
            pass
        except BaseException as cleanup_error:
            if original is None or isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                original = cleanup_error
    if original is not None and not (linked and isinstance(original, OSError)):
        raise original


def _write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_public_evidence(rows))


def _artifact_inventory(output_dir, strict=True, exclude=()):
    inventory = {}
    for path in sorted(Path(output_dir).iterdir()):
        if path.name in ("results.json", "failure.json") or path.name.endswith(".partial") or path.name in exclude:
            continue
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("artifact must be a regular file")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            inventory[path.name] = {"bytes": size, "sha256": digest.hexdigest()}
        except (OSError, ValueError) as exc:
            if strict:
                raise
            inventory[path.name] = {"error_type": type(exc).__name__, "error": str(exc)}
    return inventory


def verify_artifact_inventory(output_dir):
    """Detect changes to the files referenced by a local terminal record."""
    output_dir = Path(output_dir)
    terminals = [output_dir / name for name in ("results.json", "failure.json") if (output_dir / name).is_file()]
    if len(terminals) != 1:
        raise ValueError("run must have exactly one terminal artifact")
    record = json.loads(terminals[0].read_text(encoding="utf-8"))
    if ((terminals[0].name == "results.json" and record.get("status") != "succeeded") or
            (terminals[0].name == "failure.json" and record.get("status") not in ("failed", "cancelled"))):
        raise ValueError("terminal filename and status are inconsistent")
    expected = record.get("artifact_inventory")
    if (not isinstance(expected, dict) or any(
            not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
            or type(value["bytes"]) is not int or value["bytes"] < 0
            or not isinstance(value["sha256"], str) or len(value["sha256"]) != 64
            for value in expected.values()) or expected != _artifact_inventory(output_dir)):
        raise ValueError("artifact inventory does not match the terminal record")
    return True


def publish_supervised_outcome(output_dir, supervision):
    """Publish a terminal outcome only after a clean reaped owner has been verified."""
    output_dir = Path(output_dir)
    if not supervision.clean or supervision.timed_out or type(supervision.exit_code) is not int:
        raise RuntimeError("unsafe or unverified GPU owner exit; leave the run incomplete")
    if any((output_dir / name).exists() for name in ("results.json", "failure.json", "publication_ambiguous.json")):
        raise FileExistsError("a terminal or ambiguous outcome already exists")
    successful = supervision.exit_code == 0
    candidate_name = "candidate.json" if successful else "candidate_failure.json"
    other_name = "candidate_failure.json" if successful else "candidate.json"
    candidate_path = output_dir / candidate_name
    if not candidate_path.is_file() or (output_dir / other_name).exists():
        raise ValueError("owner exit and candidate evidence are inconsistent")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if ((successful and candidate.get("status") != "candidate") or
            (not successful and candidate.get("status") not in ("failed", "cancelled"))):
        raise ValueError("owner exit and candidate status are inconsistent")
    expected_owner = {"run_id": supervision.run_id, "pid": supervision.pid}
    started = json.loads((output_dir / "started.json").read_text(encoding="utf-8"))
    if candidate.get("owner") != expected_owner or started.get("owner") != expected_owner:
        raise ValueError("candidate owner does not match the reaped supervisor owner")
    if candidate.get("artifact_inventory") != _artifact_inventory(output_dir, exclude=(candidate_name,)):
        raise ValueError("candidate artifact inventory does not match")
    terminal = dict(candidate)
    if successful:
        terminal["status"] = "succeeded"
    terminal["supervision"] = {"run_id": supervision.run_id, "owner_pid": supervision.pid,
        "exit_code": supervision.exit_code, "device_teardown_clean": supervision.clean,
        "timed_out": supervision.timed_out, "reason": supervision.reason}
    terminal["artifact_inventory"] = _artifact_inventory(output_dir)
    _write_json(output_dir / ("results.json" if successful else "failure.json"), terminal)
    return terminal


def run_benchmark(output_dir, quick=False, exclusive=False, cache_dir=None,
                  lease_path=None, lease_timeout=30, candidate_only=False, owner_session=None,
                  cpu_preflight=None, parent_run_id=None, preflight_sha256=None):
    """Create a new evidence directory and retain a record if evaluation fails."""
    output_dir = Path(output_dir)
    if parent_run_id is None:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not candidate_only or owner_session is None or output_dir.is_symlink() or not output_dir.is_dir():
            raise RuntimeError("only a supervised owner can use a parent-created run directory")
        if {path.name for path in output_dir.iterdir()} != {"parent_started.json", "cpu_preflight.json"}:
            raise RuntimeError("parent-created run directory has unexpected contents")
        parent = json.loads((output_dir / "parent_started.json").read_text(encoding="utf-8"))
        if (parent.get("protocol") != PROTOCOL or parent.get("status") != "parent_started"
                or parent.get("parent_run_id") != parent_run_id
                or parent.get("pid") != getattr(owner_session, "supervisor_pid", None)):
            raise RuntimeError("parent-created run evidence does not match the supervised owner")
        stored = _read_cpu_preflight(output_dir / "cpu_preflight.json", preflight_sha256)
        if stored != _public_evidence(cpu_preflight):
            raise RuntimeError("stored CPU preflight differs from the owner input")
    owner = {"run_id": getattr(owner_session, "run_id", None), "pid": os.getpid()}
    progress = {"raw": [], "warmups": [], "metadata": {}, "source_bundle": None, "telemetry": None,
                "cpu_preflight": cpu_preflight,
                "owner_session": owner_session, "candidate_only": candidate_only}
    journal = None
    try:
        _write_json(output_dir / "started.json", {"protocol": PROTOCOL, "status": "started",
            "quick": quick, "exclusive_requested": exclusive, "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "started_monotonic_ns": time.perf_counter_ns(),
            "owner": owner,
            "cpu_preflight": cpu_preflight,
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "lease_path": None if lease_path is None else str(lease_path),
            "lease_timeout_seconds": lease_timeout})
        journal = (output_dir / "raw_progress.jsonl").open("x", encoding="utf-8", newline="\n")
        def record(row, bucket):
            journal.write(json.dumps(_public_evidence(row), allow_nan=False, separators=(",", ":")) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            progress[bucket].append(row)
        progress["record_sample"] = lambda row: record(row, "raw")
        progress["record_warmup"] = lambda row: record(row, "warmups")
        progress["record_source"] = lambda bundle: _write_bytes(output_dir / "source_bundle.zip", bundle)
        progress["record_manifest"] = lambda manifest: _write_json(output_dir / "manifest.json", manifest)
        progress["record_stage"] = lambda name, value: _write_json(output_dir / (name + ".json"), value)
        result = _evaluate(quick=quick, exclusive=exclusive, progress=progress,
                           cache_dir=cache_dir, lease_path=lease_path, lease_timeout=lease_timeout)
        try:
            journal.close()
        finally:
            journal = None
        rows = result.pop("raw")
        if progress["source_bundle"] is not None and not (output_dir / "source_bundle.zip").exists():
            _write_bytes(output_dir / "source_bundle.zip", progress["source_bundle"])
        if rows:
            _write_csv(output_dir / "raw_timing.csv", rows)
        if progress["warmups"]:
            _write_csv(output_dir / "warmup_timing.csv", progress["warmups"])
        if progress["telemetry"] is not None:
            _write_json(output_dir / "telemetry.json", progress["telemetry"])
        if "final_guard" in progress:
            progress["final_guard"]()
        result["status"] = "candidate" if candidate_only else "succeeded"
        result["owner"] = owner
        result["artifact_inventory"] = _artifact_inventory(output_dir)
        _write_json(output_dir / ("candidate.json" if candidate_only else "results.json"), result)
        return result
    except BaseException as exc:
        recording_errors = []
        if journal is not None:
            try:
                journal.close()
            except Exception as error:
                recording_errors.append(("raw_progress.jsonl.close", error))
            finally:
                journal = None
        def preserve(name, action):
            try:
                action()
            except Exception as error:
                recording_errors.append((name, error))

        if progress["source_bundle"] is not None and not (output_dir / "source_bundle.zip").exists():
            preserve("source_bundle.zip", lambda: _write_bytes(output_dir / "source_bundle.zip", progress["source_bundle"]))
        if progress["raw"]:
            preserve("partial_raw_timing.csv", lambda: _write_csv(output_dir / "partial_raw_timing.csv", progress["raw"]))
        if progress["warmups"]:
            preserve("partial_warmup_timing.csv", lambda: _write_csv(output_dir / "partial_warmup_timing.csv", progress["warmups"]))
        if progress["telemetry"] is not None:
            preserve("partial_telemetry.json", lambda: _write_json(output_dir / "partial_telemetry.json", progress["telemetry"]))
        if (output_dir / ("candidate.json" if candidate_only else "results.json")).exists():
            # The linked candidate/terminal is already immutable and visible.
            if recording_errors:
                raise exc from recording_errors[0][1]
            raise
        failure = {"protocol": PROTOCOL, "quick": quick, "exclusive_requested": exclusive,
                   "status": "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                   "owner": owner,
                   "error_type": type(exc).__name__, "message": str(exc),
                   "partial_samples": len(progress["raw"]), "partial_warmups": len(progress["warmups"]),
                   "partial_counts_scope": "rows acknowledged after journal fsync; a final complete-looking line after an I/O error may have unknown durability",
                   "metadata": progress["metadata"],
                   "artifact_errors": [{"artifact": name, "error_type": type(error).__name__,
                                        "message": str(error)} for name, error in recording_errors]}
        try:
            failure["artifact_inventory"] = _artifact_inventory(output_dir, strict=False)
        except Exception as inventory_error:
            failure["artifact_errors"].append({"artifact": "inventory", "error_type": type(inventory_error).__name__,
                                               "message": str(inventory_error)})
        try:
            _write_json(output_dir / ("candidate_failure.json" if candidate_only else "failure.json"), failure)
        except Exception as recording_error:
            raise exc from recording_error
        if recording_errors:
            raise exc from recording_errors[0][1]
        raise
    finally:
        if journal is not None:
            journal.close()


def _cpu_preflight(cache_dir):
    """Load and exercise the frozen checkpoint in a separate CPU-only process."""
    _require_clean_python_runtime()
    project = Path(__file__).resolve().parent.parent
    source, hashes = create_source_bundle(project)
    import torch
    import transformers
    from .pretrained import load_distilgpt2

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    bundle = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
    model = bundle.model
    if model.training or model.config.n_layer != 6 or model.config.n_embd != 768:
        raise RuntimeError("CPU checkpoint does not match the frozen model")
    ids = torch.tensor([[15496, 11, 616, 1438, 318, 257, 1332, 13]], dtype=torch.long, device="cpu")
    invoke, factory, identity = _transformer_workload(model, ids)
    checks = verify_equivalence(invoke, factory, identity)
    validate_transformer_coverage(checks["event_identities"])
    expected_layout = {"shape": [1, 3, 768], "dtype": "torch.float32", "device": "cpu"}
    if (checks["events_per_mode"] != 18 or checks["value_tensor_bytes"] != 18 * 3 * 768 * 4
            or any(layout != expected_layout for layout in checks["value_layouts"])):
        raise RuntimeError("CPU checkpoint trace does not match the frozen layout")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    modules = _loaded_module_hashes()
    source_after, after_hashes = create_source_bundle(project)
    if hashes != after_hashes or source != source_after:
        raise RuntimeError("repository source changed during CPU preflight")
    return {"model_load_metadata": bundle.metadata, "trace_preflight": checks,
            "cuda_initialized": False, "source_bundle_sha256": hashlib.sha256(source).hexdigest(),
            "repository_source_sha256": hashes, "loaded_module_file_sha256": modules,
            "runtime": {"python_version": platform.python_version(), "executable": sys.executable,
                        "prefix": sys.prefix, "original_argv": list(sys.orig_argv),
                        "pycache_prefix": sys.pycache_prefix,
                        "bytecode_writes_disabled": sys.dont_write_bytecode,
                        "no_site": bool(sys.flags.no_site),
                        "isolated": bool(sys.flags.isolated),
                        "ignore_environment": bool(sys.flags.ignore_environment),
                        "safe_path": bool(sys.flags.safe_path),
                        "hash_randomization": bool(sys.flags.hash_randomization),
                        "torch_version": str(torch.__version__),
                        "transformers_version": transformers.__version__}}


def _run_cpu_preflight(cache_dir, timeout=180):
    """Fail before reserving a GPU owner when the offline checkpoint is unusable."""
    command, environment = direct_python_module(
        "looking_glass.cuda_benchmark", ["--_cpu-preflight", "--cache-dir", str(cache_dir)])
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True,
            errors="replace", timeout=timeout, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"CPU checkpoint preflight could not complete: {type(exc).__name__}") from exc
    if result.returncode:
        raise RuntimeError("CPU checkpoint preflight failed: " + result.stderr[-2000:])
    try:
        report = json.loads(result.stdout.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("CPU checkpoint preflight returned invalid evidence") from exc
    from .pretrained import ASSETS, MODEL_ID, REVISION
    metadata = report.get("model_load_metadata", {})
    expected = {name: digest for name, (_, digest) in ASSETS.items()}
    if (report.get("cuda_initialized") is not False or metadata.get("model_id") != MODEL_ID
            or metadata.get("model_revision") != REVISION or metadata.get("asset_sha256") != expected):
        raise RuntimeError("CPU checkpoint preflight identity is incomplete")
    return report


def _read_cpu_preflight(path, digest):
    """Read bounded preflight evidence only when its exact bytes match."""
    path = Path(path)
    if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or path.is_symlink() or not path.is_file()):
        raise ValueError("invalid CPU preflight file or hash")
    with path.open("rb") as handle:
        content = handle.read(2 * 1024 * 1024 + 1)
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("CPU preflight evidence exceeds its size limit")
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("CPU preflight evidence hash mismatch")
    try:
        result = json.loads(content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid CPU preflight evidence") from exc
    if not isinstance(result, dict):
        raise ValueError("CPU preflight evidence must be an object")
    return result


def _launch_supervised_owner(output, *, identity, cache_dir, preflight_sha256, parent_run_id,
                             quick, exclusive, lease_timeout, timeout):
    if not GPU_BENCHMARK_ADMISSION_READY:
        raise RuntimeError("reviewed GPU admission for the eight-token trace workload is not available")
    from .supervisor import run_supervised

    arguments = ["--_owner", "--output", str(output), "--cache-dir", str(cache_dir),
                 "--lease-timeout", str(lease_timeout),
                 "--_parent-run-id", parent_run_id,
                 "--_preflight-sha256", preflight_sha256]
    if quick:
        arguments.append("--quick")
    if exclusive:
        arguments.append("--exclusive")
    command, environment = direct_python_module("looking_glass.cuda_benchmark", arguments)
    return run_supervised(command, device_lease_path=identity.lock_path,
        marker_path=identity.marker_path, identity=identity, timeout=timeout,
        lease_timeout=lease_timeout, cwd=str(Path(__file__).resolve().parent.parent), env=environment)


def _record_parent_failure(output, stage, exc):
    _write_json(output / "failure.json", {"protocol": PROTOCOL,
        "status": "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
        "stage": stage, "error_type": type(exc).__name__, "message": str(exc),
        "device_teardown": "unverified" if stage == "owner_launch" else "not started",
        "artifact_inventory": _artifact_inventory(output)})


def run_supervised_benchmark(output, *, workspace_root, cache_dir, quick=False, exclusive=False,
                             lease_timeout=30, timeout=1800):
    """Run the owner through the shared supervisor; accept only reaped outcomes."""
    output = Path(output).resolve()
    cache_dir = Path(cache_dir).resolve()
    workspace_root = Path(workspace_root)
    if not workspace_root.is_absolute():
        raise ValueError("the shared GPU coordination root must be absolute")
    workspace_root = workspace_root.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    parent_run_id = secrets.token_hex(16)
    parent_start = {"protocol": PROTOCOL, "status": "parent_started",
        "parent_run_id": parent_run_id, "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "quick": quick, "exclusive_requested": exclusive, "cache_dir": str(cache_dir)}
    try:
        _write_json(output / "parent_started.json", parent_start)
    except BaseException as exc:
        published = False
        try:
            published = json.loads((output / "parent_started.json").read_text(encoding="utf-8")) == _public_evidence(parent_start)
        except (OSError, ValueError):
            pass
        if published:
            try:
                _record_parent_failure(output, "parent_start", exc)
            except Exception as recording_error:
                raise exc from recording_error
        raise
    if (output / "parent_started.json.partial").exists():
        exc = RuntimeError("parent start publication left scratch; GPU owner was not launched")
        try:
            _record_parent_failure(output, "parent_start_cleanup", exc)
        except Exception as recording_error:
            raise exc from recording_error
        raise exc
    try:
        cpu_preflight = _run_cpu_preflight(cache_dir)
        source, _ = create_source_bundle(Path(__file__).resolve().parent.parent)
        if cpu_preflight.get("source_bundle_sha256") != hashlib.sha256(source).hexdigest():
            raise RuntimeError("CPU preflight source changed before GPU owner launch")
        encoded = (json.dumps(_public_evidence(cpu_preflight), indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            raise RuntimeError("CPU preflight evidence exceeds its size limit")
        preflight_path = output / "cpu_preflight.json"
        _write_bytes(preflight_path, encoded)
        if (output / "cpu_preflight.json.partial").exists():
            raise RuntimeError("CPU preflight publication left scratch; GPU owner was not launched")
        preflight_sha256 = hashlib.sha256(encoded).hexdigest()
        _read_cpu_preflight(preflight_path, preflight_sha256)
    except BaseException as exc:
        try:
            _record_parent_failure(output, "cpu_preflight", exc)
        except Exception as recording_error:
            raise exc from recording_error
        raise
    if not GPU_BENCHMARK_ADMISSION_READY:
        exc = RuntimeError("reviewed GPU admission for the eight-token trace workload is not available")
        try:
            _record_parent_failure(output, "gpu_admission", exc)
        except Exception as recording_error:
            raise exc from recording_error
        raise exc
    try:
        _require_clean_python_runtime()
        identity, _ = resolve_benchmark_identity(workspace_root, shutil.which("nvidia-smi"))
        supervision = _launch_supervised_owner(output, identity=identity, cache_dir=cache_dir,
            preflight_sha256=preflight_sha256, parent_run_id=parent_run_id, quick=quick, exclusive=exclusive,
            lease_timeout=lease_timeout, timeout=timeout)
    except BaseException as exc:
        if not (output / "started.json").exists():
            try:
                _record_parent_failure(output, "owner_launch", exc)
            except Exception as recording_error:
                raise exc from recording_error
        raise
    return publish_supervised_outcome(output, supervision)


def _owner_main(args):
    from .supervisor import OwnerSession

    session = OwnerSession.from_environment()
    exit_code = 0
    try:
        preflight = _read_cpu_preflight(args.output / "cpu_preflight.json", args._preflight_sha256)
        run_benchmark(args.output, quick=args.quick, exclusive=args.exclusive,
            cache_dir=args.cache_dir, lease_path=session.device_lease_path,
            lease_timeout=args.lease_timeout, candidate_only=True, owner_session=session,
            cpu_preflight=preflight, parent_run_id=args._parent_run_id,
            preflight_sha256=args._preflight_sha256)
    except BaseException as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        print("Owner experiment failed: " + _public_evidence(str(exc)), file=sys.stderr)
        traceback.clear_frames(exc.__traceback__)
    try:
        if not _PROCESS_LEASES:
            raise RuntimeError("owner did not acquire the device lease")
        import torch
        _finish_owner(session, _PROCESS_LEASES[-1], torch.cuda)
    except BaseException as exc:
        print("Owner cleanup was not verified: " + _public_evidence(str(exc)), file=sys.stderr)
        return 1
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true", help="small smoke run, not the full protocol")
    parser.add_argument("--exclusive", action="store_true", help="attest a reserved, drained GPU window")
    parser.add_argument("--cache-dir", required=True, type=Path, help="explicit offline checkpoint cache")
    parser.add_argument("--workspace-root", type=Path, help="shared GPU coordination root for all worktrees")
    parser.add_argument("--lease-timeout", type=float, default=30, help="lease wait in seconds, at most 60")
    parser.add_argument("--timeout", type=float, default=1800, help="owner process timeout in seconds")
    parser.add_argument("--_cpu-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_owner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_preflight-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--_parent-run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._cpu_preflight:
        print(json.dumps(_cpu_preflight(args.cache_dir), separators=(",", ":")))
        return 0
    if args._owner:
        if args.output is None or args._preflight_sha256 is None or args._parent_run_id is None:
            parser.error("owner needs output, parent run ID and preflight evidence")
        return _owner_main(args)
    if args.output is None or args.workspace_root is None:
        parser.error("--output and --workspace-root are required")
    try:
        result = run_supervised_benchmark(args.output, workspace_root=args.workspace_root,
            cache_dir=args.cache_dir, quick=args.quick, exclusive=args.exclusive,
            lease_timeout=args.lease_timeout, timeout=args.timeout)
    except Exception as exc:
        print("Benchmark failed or remains incomplete: " + _public_evidence(str(exc)), file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "protocol": result["protocol"], "status": result["status"]}))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
