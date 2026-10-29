"""Versioned, isolated latency and memory measurement for a pretrained GPT-2.

This is not the shared-service throughput experiment. Run as its own process.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile


PROTOCOL = "cuda-isolated-v1"
MODES = ("none", "id", "direct", "gradient")
_PROCESS_LEASES = []


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


def runtime_guard(lease, cuda, device, reserve_bytes, trace_limit):
    """Check cooperative ownership and current device-wide free headroom."""
    def check():
        lease.check()
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
    if min(total, used, free, utilization) < 0 or utilization > 100:
        raise ValueError("invalid nvidia-smi telemetry value")
    return {"observed_monotonic_ns": observed_ns, "gpu_timestamp": fields[0],
            "gpu_uuid": fields[1], "total_bytes": total * 1024 * 1024,
            "used_bytes": used * 1024 * 1024, "free_bytes": free * 1024 * 1024,
            "utilization_percent": utilization}


def classify_isolation(exclusive, observations, started_ns, ended_ns, process_queries,
                       own_pid, gpu_uuid, telemetry_errors):
    """Apply the prespecified idle screen without claiming proof of exclusivity."""
    selected = sorted((row for row in observations if row["gpu_uuid"] == gpu_uuid),
                      key=lambda row: row["observed_monotonic_ns"])
    before = [row for row in selected if row["observed_monotonic_ns"] < started_ns][-3:]
    after = [row for row in selected
             if row["observed_monotonic_ns"] >= ended_ns + 1_000_000_000][-3:]
    unavailable, contaminated = [], []
    if not exclusive:
        unavailable.append("exclusive reservation was not attested")
    if len(before) < 3 or len(after) < 3:
        unavailable.append("fewer than three idle samples before or after workload")
    if any(row["gpu_uuid"] != gpu_uuid for row in observations) or not selected:
        unavailable.append("telemetry GPU identity does not match the selected device")
    if telemetry_errors:
        unavailable.append("telemetry query failed")
    if before and after:
        window = [row["observed_monotonic_ns"] for row in selected
                  if before[-1]["observed_monotonic_ns"] <= row["observed_monotonic_ns"] <= after[0]["observed_monotonic_ns"]]
        if any(later - earlier > 2_000_000_000 for earlier, later in zip(window, window[1:])):
            unavailable.append("telemetry coverage has a gap longer than two seconds")
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
    return {"ok": True, "uuid": uuid, "name": name, "total_bytes": total_bytes,
            "driver_version": driver, "mapping": "only physical GPU and only CUDA-visible GPU"}


def resolve_benchmark_identity(workspace_root, smi):
    """Derive the shared physical-GPU lock and marker without touching CUDA."""
    from .device_identity import resolve_device_identity
    inventory = _gpu_inventory(smi)
    if not inventory.get("ok") or not inventory.get("uuid"):
        raise RuntimeError("cannot establish the physical GPU UUID before CUDA")
    identity = resolve_device_identity(workspace_root, inventory["uuid"],
                                       physical_uuids=(inventory["uuid"],))
    return identity, inventory


def _query_compute_processes(smi, gpu_uuid, phase):
    result = {"phase": phase, "observed_monotonic_ns": time.monotonic_ns(),
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
            self.errors.append({"observed_monotonic_ns": time.monotonic_ns(),
                                "error": "nvidia-smi unavailable"})
            return
        self._thread = threading.Thread(target=self._poll, name="cuda-benchmark-telemetry", daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            started = time.monotonic_ns()
            try:
                command = _command([self.smi,
                    "--query-gpu=timestamp,uuid,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits"])
                ended = time.monotonic_ns()
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
                                    "observed_monotonic_ns": time.monotonic_ns(),
                                    "error_type": type(exc).__name__, "message": str(exc)})
            self._stop.wait(self.interval_seconds)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                self.errors.append({"observed_monotonic_ns": time.monotonic_ns(),
                                    "error": "telemetry thread did not stop"})
        return {"configured_interval_seconds": self.interval_seconds,
                "observations": list(self.observations), "errors": list(self.errors),
                "limitations": ["nvidia-smi polling can perturb latency",
                    "query timestamps bound collection rather than identify its exact instant",
                    "WDDM and endpoint-only process reporting can miss activity"]}


def verify_equivalence(invoke, session_factory, identity):
    """Require identical outputs and matched, exact value snapshots before timing."""
    import torch

    sessions = {}
    with torch.no_grad():
        baseline = invoke()
        for mode in ("id", "direct", "gradient"):
            with session_factory(mode) as session:
                output = invoke()
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
    return {"events_per_mode": len(reference),
            "value_tensor_bytes": sessions["direct"].tensor_bytes,
            "event_identities": keys,
            "value_layouts": [{"shape": list(event.value.shape), "dtype": str(event.value.dtype),
                               "device": str(event.value.device)} for event in reference],
            "outputs_exact": True, "values_exact": True}


def measure_sample(invoke, session_factory, mode, device, cuda, start_event, end_event):
    """Measure one operation with the trace alive across its completion boundary."""
    import torch

    if mode not in MODES:
        raise ValueError("unsupported capture mode")
    cuda.synchronize(device)
    cuda.reset_peak_memory_stats(device)
    free_before, total = cuda.mem_get_info(device)
    allocated_before = cuda.memory_allocated(device)
    reserved_before = cuda.memory_reserved(device)
    started_ns = time.monotonic_ns()
    start_event.record()
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
    end_event.record()
    cuda.synchronize(device)
    cuda_ms = start_event.elapsed_time(end_event)
    del output, session
    cuda.synchronize(device)
    ended_ns = time.monotonic_ns()
    free_after, total_after = cuda.mem_get_info(device)
    if total != total_after:
        raise RuntimeError("CUDA device total memory changed during a sample")
    return {"started_monotonic_ns": started_ns, "ended_monotonic_ns": ended_ns,
            "wall_ns": ended_ns - started_ns, "cuda_ms": cuda_ms,
            "events": events, "recorded_tensor_bytes": value_bytes,
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": cuda.max_memory_allocated(device),
            "peak_reserved_bytes": cuda.max_memory_reserved(device),
            "allocated_after_bytes": cuda.memory_allocated(device),
            "reserved_after_bytes": cuda.memory_reserved(device),
            "device_free_before_bytes": free_before,
            "device_free_after_bytes": free_after,
            "device_total_bytes": total}


def collect_trials(invoke, factory, device, cuda, start_event, end_event,
                   seed, warmup_rounds, measured_rounds, progress, guard=None):
    """Warm each mode, then retain every interleaved measurement as it finishes."""
    for row in make_schedule(seed ^ 0x5A17, warmup_rounds):
        if guard is not None:
            guard()
        measure_sample(invoke, factory, row["mode"], device, cuda, start_event, end_event)
    for row in make_schedule(seed, measured_rounds):
        if guard is not None:
            guard()
        measurement = measure_sample(invoke, factory, row["mode"], device, cuda, start_event, end_event)
        sample = {"seed": seed, **row,
                  "phase": "forward_only" if row["mode"] == "none" else "forward_plus_capture",
                  **measurement}
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
        selected = rows if threshold_ns is None else [row for row in rows if row["observed_monotonic_ns"] >= threshold_ns]
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


def validate_transformer_coverage(identities):
    """Require every frozen layer/role exactly once at the selected positions."""
    expected = {(f"transformer.h.{layer}" + ("" if kind == "block" else f".{kind}"), kind, layer)
                for layer in range(6) for kind in ("block", "attn", "mlp")}
    observed = []
    for entry in identities:
        if len(entry) != 8:
            raise AssertionError("preflight coverage has malformed identity")
        path, kind, layer, forward, invocation, _output, samples, tokens = entry
        if forward != 0 or invocation != 0 or samples != (0,) or tokens != (0, 3, 7):
            raise AssertionError("preflight coverage has unexpected invocation or positions")
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
    lease = _acquire_device_lease(lease_path, lease_timeout)
    lease.check()
    owner_session.check(lease)
    progress["final_guard"] = lease.check
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
    inventory = _gpu_inventory(smi, torch.cuda.device_count())
    loaded = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
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
        properties = torch.cuda.get_device_properties(device)
        environment = {
            "python_version": platform.python_version(), "platform": platform.platform(),
            "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
            "torch_cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_name": properties.name,
            "cuda_compute_capability": [properties.major, properties.minor],
            "cuda_total_memory_bytes": properties.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
            "environment": environment, "admission": admission,
            "source": progress["metadata"].copy(), "git": _git_provenance(project)}
        if model.training or model.config.n_layer != 6 or model.config.n_embd != 768:
            progress["metadata"]["preload"] = stage
            progress["record_stage"]("preload", stage)
            raise RuntimeError("pretrained checkpoint does not match the frozen workload")
        guard = runtime_guard(lease, torch.cuda, device, admission["reserve_bytes"], 4 * 1024 ** 2)
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
        torch.cuda.synchronize(device)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        end_event.record()
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
            workload_started = time.monotonic_ns()
            for seed in seeds:
                collect_trials(invoke, factory, device, torch.cuda, start_event, end_event,
                               seed, warmup_rounds, measured_rounds, progress, guard=guard)
            guard()
            workload_ended = time.monotonic_ns()
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
        lease.check()
        return {"protocol": PROTOCOL, "protocol_mode": "smoke" if quick else "full",
            "manifest": manifest, "raw": progress["raw"], "timing_summary": summarize(progress["raw"]),
            "paired_ratios": summarize_paired_ratios(progress["raw"]),
            "telemetry_summary": telemetry_summary,
            "loaded_module_sha256_after_trials": module_hashes,
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def _write_json(path, value):
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes(path, encoded)


def _write_bytes(path, value):
    temporary = path.with_name(path.name + ".partial")
    created, linked = False, False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
        temporary.unlink()
        created = False
    except BaseException as original:
        if linked:
            try:
                path.unlink()
            except Exception as rollback_error:
                raise RuntimeError("artifact publication is ambiguous; rollback failed") from rollback_error
        raise original
    finally:
        if created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _artifact_inventory(output_dir, strict=True, exclude=()):
    inventory = {}
    for path in sorted(Path(output_dir).iterdir()):
        if path.name in ("results.json", "failure.json") or path.name in exclude:
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
                  lease_path=None, lease_timeout=30, candidate_only=False, owner_session=None):
    """Create a new evidence directory and retain a record if evaluation fails."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    owner = {"run_id": getattr(owner_session, "run_id", None), "pid": os.getpid()}
    progress = {"raw": [], "metadata": {}, "source_bundle": None, "telemetry": None,
                "owner_session": owner_session, "candidate_only": candidate_only}
    journal = None
    try:
        _write_json(output_dir / "started.json", {"protocol": PROTOCOL, "status": "started",
            "quick": quick, "exclusive_requested": exclusive, "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "started_monotonic_ns": time.monotonic_ns(),
            "owner": owner,
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "lease_path": None if lease_path is None else str(lease_path),
            "lease_timeout_seconds": lease_timeout})
        journal = (output_dir / "raw_progress.jsonl").open("x", encoding="utf-8", newline="\n")
        def record_sample(row):
            journal.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            progress["raw"].append(row)
        progress["record_sample"] = record_sample
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
        if progress["telemetry"] is not None:
            preserve("partial_telemetry.json", lambda: _write_json(output_dir / "partial_telemetry.json", progress["telemetry"]))
        if (output_dir / ("candidate.json" if candidate_only else "results.json")).exists():
            preserve("publication_ambiguous.json", lambda: _write_json(output_dir / "publication_ambiguous.json",
                {"status": "ambiguous", "error_type": type(exc).__name__, "message": str(exc)}))
            if recording_errors:
                raise exc from recording_errors[0][1]
            raise
        failure = {"protocol": PROTOCOL, "quick": quick, "exclusive_requested": exclusive,
                   "status": "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                   "owner": owner,
                   "error_type": type(exc).__name__, "message": str(exc),
                   "partial_samples": len(progress["raw"]), "metadata": progress["metadata"],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quick", action="store_true", help="small smoke run, not the full protocol")
    parser.add_argument("--exclusive", action="store_true", help="attest a reserved, drained GPU window")
    parser.add_argument("--cache-dir", required=True, type=Path, help="explicit offline checkpoint cache")
    parser.add_argument("--lease-path", required=True, type=Path, help="absolute cooperative device lock path")
    parser.add_argument("--lease-timeout", type=float, default=30, help="lease wait in seconds, at most 60")
    args = parser.parse_args()
    result = run_benchmark(args.output, quick=args.quick, exclusive=args.exclusive,
                           cache_dir=args.cache_dir, lease_path=args.lease_path,
                           lease_timeout=args.lease_timeout)
    print(json.dumps({"output": str(args.output), "protocol": result["protocol"]}))


if __name__ == "__main__":
    main()
