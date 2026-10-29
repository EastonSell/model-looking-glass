"""Supervised, development-only owner for the two-token tap timing smoke."""

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import weakref

from . import cuda_benchmark as bench
from . import gpu_tap_latency_smoke as smoke

GIB = 1024 ** 3
OUTSIDE_CAP_ALLOWANCE = smoke.OUTSIDE_CAP_ALLOWANCE
TRACE_CLAIM = 2 * smoke.TRACE_LIMIT
# Allocated before any GPU work: lease, session, budget, model bundle, input,
# terminal exception and any conversion result stay rooted until process exit.
_CUSTODY = [None, None, None, None, None, None, None]
_OWNER_LOCK = threading.Lock()
_OWNER_USED = False


def _read_json(path, limit):
    with Path(path).open("rb") as source:
        contents = source.read(limit + 1)
    if len(contents) > limit:
        raise ValueError("evidence file exceeds its byte bound")
    result = json.loads(contents)
    if type(result) is not dict:
        raise ValueError("evidence file must contain an object")
    return result


def _telemetry_projection(report):
    """Remove UUIDs and arbitrary command output from the shareable view."""
    return {"configured_interval_seconds": report["configured_interval_seconds"],
            "observations": bench._public_evidence(report["observations"]),
            "errors": [{key: item[key] for key in ("query_started_monotonic_ns", "observed_monotonic_ns",
                "error_type", "returncode") if key in item} for item in report["errors"]],
            "limitations": report["limitations"]}


def _screen_projection(report):
    """Keep process names, paths and PIDs only in the private evidence."""
    classification = bench._public_evidence(report["classification"])
    classification["foreign_compute_process_count"] = len(classification.pop("foreign_compute_pids", []))
    queries = [{"phase": item.get("phase"), "ok": item.get("ok"),
                "observed_monotonic_ns": item.get("observed_monotonic_ns"),
                "query_started_monotonic_ns": item.get("query_started_monotonic_ns"),
                "query_ended_monotonic_ns": item.get("query_ended_monotonic_ns"),
                "process_count": len(item.get("pids", []))} for item in report["process_queries"]]
    return {"classification": classification, "process_queries": queries,
            "workload_started_monotonic_ns": report["workload_started_monotonic_ns"],
            "workload_ended_monotonic_ns": report["workload_ended_monotonic_ns"], "clock": report["clock"]}


def _activity_interpretation(screen, exploratory_desktop):
    if screen.get("availability") != "complete":
        raise RuntimeError("activity screen is unverified")
    if screen.get("status") == "no_contamination_observed":
        return "screened_no_contamination_observed"
    if screen.get("status") == "contaminated" and exploratory_desktop:
        return "exploratory_contaminated_activity"
    raise RuntimeError("activity screen is " + str(screen.get("status")))


def _valid_process_query(query, phase, gpu_uuid):
    """Check the private process rows and PID projection before classifying."""
    if (type(gpu_uuid) is not str or not gpu_uuid
            or type(query) is not dict or query.get("phase") != phase or query.get("ok") is not True
            or query.get("other_gpu_uuids") != [] or type(query.get("pids")) is not list
            or type(query.get("processes")) is not list):
        return False
    processes = query["processes"]
    if any(type(row) is not dict or set(row) != {"gpu_uuid", "pid", "name"}
           or type(row["gpu_uuid"]) is not str or row["gpu_uuid"].casefold() != gpu_uuid.casefold()
           or type(row["pid"]) is not int or row["pid"] <= 0
           or type(row["name"]) is not str or not row["name"].strip() for row in processes):
        return False
    return (all(type(pid) is int and pid > 0 for pid in query["pids"])
            and [row["pid"] for row in processes] == query["pids"])


def _initial_monitor_issue(observations, errors, process_query, gpu_uuid, now_ns=None):
    """Require available, time-bounded telemetry before beginning a workload."""
    if errors:
        return "telemetry query failed before the workload"
    if not _valid_process_query(process_query, "before", gpu_uuid):
        return "compute process query was unavailable before the workload"
    start, end = (process_query.get(key) for key in
                  ("query_started_monotonic_ns", "query_ended_monotonic_ns"))
    if (type(start) is not int or type(end) is not int or not 0 <= start <= end
            or process_query.get("observed_monotonic_ns") != start
            or (now_ns is not None and (type(now_ns) is not int or end > now_ns
                                        or now_ns - start > 2_000_000_000))):
        return "compute process query was stale or had invalid timestamps before the workload"
    if len(observations) < 3:
        return "fewer than three telemetry observations before the workload"
    if any(type(row) is not dict or row.get("gpu_uuid") != gpu_uuid for row in observations):
        return "telemetry device identity was ambiguous"
    rows = observations
    for row in rows:
        total, used, free, utilization, start, end, observed = (row.get(key) for key in
            ("total_bytes", "used_bytes", "free_bytes", "utilization_percent",
             "query_started_monotonic_ns", "query_ended_monotonic_ns", "observed_monotonic_ns"))
        if (any(type(value) is not int for value in (total, used, free, utilization, start, end, observed))
                or total <= 0 or not 0 <= used <= total or not 0 <= free <= total
                or not 0 <= utilization <= 100 or not 0 <= start <= end == observed):
            return "telemetry values or query timestamps were invalid"
    if (len({row["total_bytes"] for row in rows}) != 1
            or any(later["query_ended_monotonic_ns"] - earlier["query_started_monotonic_ns"] > 2_000_000_000
                   for earlier, later in zip(rows, rows[1:]))):
        return "telemetry coverage or device total was inconsistent before the workload"
    if (now_ns is not None and (type(now_ns) is not int or now_ns < rows[-1]["query_ended_monotonic_ns"]
            or now_ns - rows[-1]["query_started_monotonic_ns"] > 2_000_000_000)):
        return "the latest telemetry observation was stale or in the future"
    return None


def _measure_if_monitored(output, monitor, process_before, gpu_uuid, **sample_args):
    observations, errors = list(monitor.observations), list(monitor.errors)
    issue = _initial_monitor_issue(observations, errors, process_before, gpu_uuid, time.perf_counter_ns())
    if issue is not None:
        return None, issue
    with (output / "observations.jsonl").open("xb") as journal:
        summary = smoke.run_samples(**sample_args, record=lambda row: _record_line(journal, row))
    return summary, None


def _record_line(handle, row):
    encoded = (json.dumps(bench._public_evidence(row), sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > 128 * 1024:
        raise ValueError("evidence row exceeds its bound")
    bench._write_all(handle, encoded)
    handle.flush()
    os.fsync(handle.fileno())


def verify_observations(path, summary, *, expected_total_bytes, expected_cap_bytes, device="cuda:0",
                        workload_started_ns=None, workload_ended_ns=None):
    """Reconstruct accepted paired samples from the durable owner journal."""
    if type(expected_total_bytes) is not int or type(expected_cap_bytes) is not int:
        raise ValueError("expected memory limits must be integers")
    with Path(path).open("rb") as source:
        data = source.read(8 * 1024 ** 2 + 1)
    if len(data) > 8 * 1024 ** 2 or not data.endswith(b"\n"):
        raise ValueError("journal is missing, truncated or exceeds its bound")
    rows = [json.loads(line) for line in data.splitlines()]
    expected = [("preflight", None)]
    for kind, seed, rounds in (("warmup", smoke.SEED ^ 0x5A17, smoke.WARMUP_ROUNDS),
                               ("measured", smoke.SEED, smoke.MEASURED_ROUNDS)):
        expected.extend((kind, item) for item in smoke.schedule(seed, rounds))
    if len(rows) != 2 * len(expected):
        raise ValueError("journal has missing or extra observations")
    measured = []
    clock = summary["clock"]
    resolution = clock["resolution_seconds"]
    if (clock.get("name") != "perf_counter_ns" or clock.get("monotonic") is not True
            or type(resolution) not in (int, float) or not math.isfinite(resolution)
            or not 0 < resolution <= 0.000001
            or type(clock.get("minimum_wall_ns")) is not int
            or clock.get("minimum_wall_ns") != math.ceil(100 * resolution * 1_000_000_000)):
        raise ValueError("journal summary has invalid clock metadata")
    minimum_wall_ns = clock["minimum_wall_ns"]

    def check_memory(state):
        if not isinstance(state, dict):
            raise ValueError("missing memory postcheck")
        free, total, allocated, reserved = (state.get(key) for key in
            ("device_free_bytes", "device_total_bytes", "allocated_bytes", "reserved_bytes"))
        bench._validate_memory_snapshot(free, total, allocated, reserved)
        reserve = max(GIB, (15 * total + 99) // 100)
        required = reserve + smoke.HEADROOM + OUTSIDE_CAP_ALLOWANCE + max(0, expected_cap_bytes - reserved)
        if (total != expected_total_bytes or type(state.get("allocator_cap_bytes")) is not int
                or type(state.get("reserve_bytes")) is not int
                or state.get("allocator_cap_bytes") != expected_cap_bytes
                or state.get("reserve_bytes") != reserve or free < required
                or reserved > expected_cap_bytes - smoke.HEADROOM):
            raise ValueError("memory postcheck does not satisfy the frozen guard")

    for index, (kind, item) in enumerate(expected):
        observation, postcheck = rows[2 * index:2 * index + 2]
        observation_id = "preflight" if item is None else f"{kind}:{item['round']}:{item['position']}"
        if (observation.get("stage") != kind or observation.get("observation_id") != observation_id
                or observation.get("postcheck_status") != "pending" or postcheck.get("stage") != "postcheck"
                or postcheck.get("observation_id") != observation_id or postcheck.get("status") != "passed"):
            raise ValueError("journal contains an unvalidated or mismatched observation")
        if item is None:
            checks = observation.get("checks", {})
            expected_identity = ["transformer.h.0", "block", 0, 0, 0, [0], [0], [1]]
            canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
            if (type(checks.get("events_per_mode")) is not int or checks.get("events_per_mode") != 1
                    or type(checks.get("value_tensor_bytes")) is not int or checks.get("value_tensor_bytes") != 3072
                    or canonical(checks.get("event_identities")) != canonical([expected_identity])
                    or canonical(checks.get("value_layouts")) != canonical([{"shape": [1, 1, 768],
                        "dtype": "torch.float32", "device": str(device)}])
                    or any(checks.get(key) is not True for key in ("outputs_finite", "outputs_exact", "values_exact"))
                    or checks.get("comparison", {}).get("operator") != "torch.equal"):
                raise ValueError("preflight evidence differs from the frozen case")
            before, after = observation.get("memory_before"), postcheck.get("memory_after")
            check_memory(before)
            check_memory(after)
            peak_allocated, peak_reserved = (observation.get("peak_allocated_bytes"),
                                             observation.get("peak_reserved_bytes"))
            if (type(peak_allocated) is not int or type(peak_reserved) is not int
                    or peak_allocated < max(before["allocated_bytes"], after["allocated_bytes"])
                    or peak_reserved < max(before["reserved_bytes"], after["reserved_bytes"])
                    or not 0 <= peak_allocated <= peak_reserved <= expected_cap_bytes - smoke.HEADROOM):
                raise ValueError("invalid preflight memory peaks")
        else:
            seed = smoke.SEED ^ 0x5A17 if kind == "warmup" else smoke.SEED
            if (any(type(observation.get(key)) is not type(value) or observation.get(key) != value
                    for key, value in item.items()) or type(observation.get("seed")) is not int
                    or type(observation.get("schedule_seed")) is not int
                    or observation.get("seed") != smoke.SEED or observation.get("schedule_seed") != seed):
                raise ValueError("journal does not match the frozen schedule")
            bench.validate_measurement(observation)
            if workload_started_ns is not None and (observation["started_monotonic_ns"] < workload_started_ns
                    or observation["ended_monotonic_ns"] > workload_ended_ns):
                raise ValueError("sample falls outside the activity-screen interval")
            for endpoint in ("before", "after"):
                try:
                    smoke.validate_memory(observation[f"device_free_{endpoint}_bytes"],
                        observation["device_total_bytes"], observation[f"allocated_{endpoint}_bytes"],
                        observation[f"reserved_{endpoint}_bytes"], expected_cap_bytes)
                except (ValueError, RuntimeError) as exc:
                    raise ValueError("timed sample failed the live memory guard") from exc
            expected_events = 0 if item["mode"] == "none" else 1
            expected_bytes = 0 if item["mode"] == "none" else 3072
            if (observation["wall_ns"] < minimum_wall_ns or observation["events"] != expected_events
                    or observation["recorded_tensor_bytes"] != expected_bytes
                    or observation.get("output_shape") != [1, 2, 50257]
                    or not all(type(value) is int for value in observation["output_shape"])
                    or observation.get("output_dtype") != "torch.float32"
                    or observation.get("output_device") != str(device)
                    or observation.get("cuda_stream_is_default") is not True
                    or observation["device_total_bytes"] != expected_total_bytes
                    or observation["peak_reserved_bytes"] > expected_cap_bytes - smoke.HEADROOM):
                raise ValueError("journal sample differs from the frozen timing or trace contract")
            check_memory(postcheck.get("memory_before"))
            check_memory(postcheck.get("memory_after"))
            if kind == "measured":
                measured.append(observation)
    groups = {}
    for row in measured:
        groups.setdefault(row["round"], {})[row["mode"]] = row["wall_ns"]
    if len(groups) != smoke.MEASURED_ROUNDS or any(set(value) != set(smoke.MODES) for value in groups.values()):
        raise ValueError("paired rounds are incomplete")
    expected_medians = {mode: statistics.median(row["wall_ns"] for row in measured if row["mode"] == mode)
                        for mode in smoke.MODES}
    gradient_direct = statistics.median(group["gradient"] / group["direct"] for group in groups.values())
    direct_none = statistics.median(group["direct"] / group["none"] for group in groups.values())
    if (summary.get("protocol") != smoke.PROTOCOL or type(summary.get("warmup_samples")) is not int
            or type(summary.get("measured_samples")) is not int or summary.get("warmup_samples") != 6
            or summary.get("measured_samples") != 18 or summary.get("median_wall_ns") != expected_medians
            or not math.isfinite(gradient_direct) or not math.isfinite(direct_none)
            or summary.get("median_paired_gradient_over_direct") != gradient_direct
            or summary.get("median_paired_direct_over_none") != direct_none):
        raise ValueError("summary does not match the validated journal")
    return {"preflight": 1, "warmups": 6, "measured": 18}


def _runtime_uuid(torch, value):
    uuid_type = getattr(getattr(torch, "_C", None), "_CUuuid", None)
    if not isinstance(uuid_type, type) or type(value) is not uuid_type:
        raise RuntimeError("unsupported CUDA UUID representation")
    return bench._cuda_runtime_uuid(value)


def _drain_after_failure(session, lease, cuda):
    """Attempt synchronization without claiming that hooks or storage drained."""
    try:
        session.check(lease)
        if cuda.is_initialized():
            session.check(lease)
            cuda.synchronize(0)
        return {"synchronization": "returned", "clean_teardown": False}
    except BaseException as exc:
        return {"synchronization": "failed", "error_type": type(exc).__name__,
                "clean_teardown": False}


def _verify_cuda_binding(session, lease, torch):
    session.check(lease)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before owner verification")
    session.check(lease)
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError("one CUDA-visible device is required")
    session.check(lease)
    properties = torch.cuda.get_device_properties(0)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    session.verify_runtime_binding(lease, runtime_uuid=runtime_uuid,
                                   device_count=count, cuda_device_index=0)
    return properties, runtime_uuid


def _run_owner(session, lease, output, cache_dir, smi, exclusive, exploratory_desktop):
    """Own all model and trace storage until a verified successful drain."""
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("CUDA-capable packages loaded before the lease")
    session.check(lease)
    if session.identity is None:
        raise RuntimeError("physical device identity is required")
    import torch
    from .memory_budget import CudaMemorySnapshot, MemoryBudget
    from .pretrained import load_distilgpt2

    properties, runtime_uuid = _verify_cuda_binding(session, lease, torch)
    private_identity = {"gpu_uuid": runtime_uuid, "owner": {"run_id": session.run_id, "pid": os.getpid()}}
    bench._write_bytes(output / "owner_identity.private.json",
        (json.dumps(private_identity, sort_keys=True) + "\n").encode("utf-8"))
    session.check(lease)
    if torch.cuda.get_allocator_backend() != "native":
        raise RuntimeError("the native CUDA allocator is required")
    session.check(lease)
    torch.cuda.set_per_process_memory_fraction(0.25, 0)
    cap = int(0.25 * properties.total_memory)
    session.check(lease)
    readback = torch.cuda.get_per_process_memory_fraction(0)
    if readback != cap / properties.total_memory:
        raise RuntimeError("allocator fraction readback differs from the requested cap")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    session.check(lease)
    torch.manual_seed(smoke.SEED)
    session.check(lease)
    torch.cuda.manual_seed(smoke.SEED)
    session.check(lease)
    torch.cuda.synchronize(0)
    session.check(lease)
    initial = smoke.memory_guard(torch.cuda, 0, cap)
    total = initial["device_total_bytes"]
    if total != properties.total_memory or initial["device_free_bytes"] < 3 * GIB:
        raise RuntimeError("CUDA totals differ or less than 3 GiB is free before model transfer")
    reserve = initial["reserve_bytes"]

    def telemetry():
        session.check(lease)
        free, device_total = torch.cuda.mem_get_info(0)
        if device_total != total:
            raise RuntimeError("CUDA device total changed after owner binding")
        return CudaMemorySnapshot(free, device_total, torch.cuda.memory_allocated(0),
                                  torch.cuda.memory_reserved(0))

    # One conservative commitment spans model copy, inputs, simultaneous
    # preflight sessions, output tensors, temporary workspaces and every trial.
    # It is an estimate plus allocator cap, not a physical VRAM reservation.
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=TRACE_CLAIM,
                          max_host_trace_bytes=0, minimum_reserve_bytes=reserve + smoke.HEADROOM)
    _CUSTODY[2] = budget
    commitment = budget.try_acquire(cap + OUTSIDE_CAP_ALLOWANCE,
                                    max_gpu_trace_bytes=TRACE_CLAIM, max_host_trace_bytes=0)
    if commitment is None:
        raise RuntimeError("whole-run provisional memory commitment was not admitted")
    commitment.reserve_trace("gpu", TRACE_CLAIM)

    def guard():
        session.check(lease)
        state = smoke.memory_guard(torch.cuda, 0, cap)
        if state["device_total_bytes"] != total:
            raise RuntimeError("CUDA device total changed")

    bench._write_json(output / "owner_manifest.json", {
        "schema": "gpu-tap-latency-owner-v1", "protocol": smoke.PROTOCOL,
        "owner": {"run_id": session.run_id, "pid": os.getpid()},
        "cooperative_window_attested": exclusive,
        "exploratory_desktop": exploratory_desktop,
        "model_id": "distilbert/distilgpt2", "token_ids": list(smoke.TOKEN_IDS),
        "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        "runtime": {"torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                    "device_name": properties.name, "device_property_total_bytes": properties.total_memory,
                    "cuda_total_bytes": total, "allocator_backend": "native", "allocator_fraction": 0.25,
                    "runtime_uuid_verified_under_lease": True,
                    "allocator_readback": readback, "allocator_limit_bytes": cap,
                    "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
                    "clock": {"name": "perf_counter_ns", **vars(time.get_clock_info("perf_counter"))}},
        "admission": {"type": "unprofiled_development", "initial": initial,
                      "incremental_commitment_bytes": cap + OUTSIDE_CAP_ALLOWANCE,
                      "outside_allocator_cap_allowance_bytes": OUTSIDE_CAP_ALLOWANCE,
                      "trace_claim_bytes": TRACE_CLAIM, "physical_reservation": False,
                      "budget": dataclasses.asdict(budget.snapshot())},
        "limitations": ["the commitment and allocator cap are not a guaranteed peak",
                        "other processes and non-PyTorch CUDA allocations are not controlled"]})

    loaded = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
    _CUSTODY[3] = loaded
    model = loaded.model
    metadata = loaded.metadata
    conversion_flags = {
        "overwrite": torch.__future__.get_overwrite_module_params_on_conversion(),
        "swap": torch.__future__.get_swap_module_params_on_conversion()}
    if any(value is not False for value in conversion_flags.values()) or model.transformer.wte.weight is not model.lm_head.weight:
        raise RuntimeError("pinned model does not have the expected tied-parameter conversion policy")
    guard()
    converted = model.to("cuda:0")
    _CUSTODY[6] = converted
    if converted is not model:
        raise RuntimeError("model transfer returned an unexpected object")
    session.check(lease)
    torch.cuda.synchronize(0)
    guard()
    if (model.transformer.wte.weight is not model.lm_head.weight
            or torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise RuntimeError("model transfer did not preserve tied embedding and output weights")
    if model.training or model.config.use_cache:
        raise RuntimeError("model must use evaluation mode without cache")
    parameter_count = 0
    for parameter in model.parameters():
        if parameter.device.type != "cuda" or parameter.device.index != 0 or parameter.dtype != torch.float32:
            raise RuntimeError("model parameter layout differs from the protocol")
        parameter_count += parameter.numel()
    parameter = None
    if parameter_count != metadata["parameter_count"]:
        raise RuntimeError("model parameter count differs from verified CPU load")
    bench._write_json(output / "model_runtime.json", {"python": sys.version,
        "torch": torch.__version__, "transformers": metadata.get("transformers_version"),
        "attention_implementation": metadata.get("attention_implementation"),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "conversion_flags": conversion_flags, "tied_parameter_preserved": True})
    guard()
    input_ids = torch.tensor([list(smoke.TOKEN_IDS)], dtype=torch.int64, device="cuda:0")
    _CUSTODY[4] = input_ids
    guard()
    invoke, factory, identity = smoke.build_workload(model, input_ids)

    monitor = bench._NvidiaMonitor(smi)
    monitor.start()
    telemetry_result = None
    body_error = None
    summary, initial_issue = None, None
    try:
        bench._wait_for_observations(monitor)
        process_before = bench._query_compute_processes(smi, session.identity.gpu_uuid, "before")
        started = time.perf_counter_ns()
        summary, initial_issue = _measure_if_monitored(output, monitor, process_before, session.identity.gpu_uuid,
            invoke=invoke, factory=factory, identity=identity, cuda=torch.cuda, device="cuda:0",
            cap_bytes=cap, expected_total_bytes=total, check_lease=guard)
        ended = time.perf_counter_ns()
        if initial_issue is None:
            bench._wait_for_observations(monitor, threshold_ns=ended + 1_000_000_000)
        process_after = bench._query_compute_processes(smi, session.identity.gpu_uuid, "after")
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            telemetry_result = monitor.stop()
            # The private copy preserves distinct UUIDs and raw diagnostics.
            # Even the redacted projection requires review before publication.
            private = (json.dumps(telemetry_result, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
            bench._write_bytes(output / "telemetry.private.json", private)
            bench._write_json(output / "telemetry.json", _telemetry_projection(telemetry_result))
        except BaseException as recording_error:
            if body_error is None:
                raise
            if hasattr(body_error, "add_note"):
                body_error.add_note("telemetry recording failed: " + type(recording_error).__name__)
    screen = bench.classify_isolation(exclusive, telemetry_result["observations"], started, ended,
        [process_before, process_after], os.getpid(), session.identity.gpu_uuid, telemetry_result["errors"])
    screen_evidence = {"classification": screen,
        "process_queries": [process_before, process_after],
        "workload_started_monotonic_ns": started, "workload_ended_monotonic_ns": ended,
        "clock": "perf_counter_ns"}
    bench._write_bytes(output / "screen.private.json",
        (json.dumps(screen_evidence, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    bench._write_json(output / "screen.json", _screen_projection(screen_evidence))
    if summary is not None:
        verify_observations(output / "observations.jsonl", summary,
                            expected_total_bytes=total, expected_cap_bytes=cap)
    guard()
    references = {"model": weakref.ref(model), "input": weakref.ref(input_ids)}
    session.check(lease)
    torch.cuda.synchronize(0)
    _CUSTODY[3] = None
    _CUSTODY[4] = None
    _CUSTODY[6] = None
    del invoke, factory, identity, input_ids, model, loaded, converted
    gc.collect()
    survivors = {name: reference() is not None for name, reference in references.items()}
    session.check(lease)
    torch.cuda.synchronize(0)
    session.check(lease)
    torch.cuda.empty_cache()
    session.check(lease)
    before_workspace = smoke.memory_guard(torch.cuda, 0, cap)
    bench._write_json(output / "cleanup_before_workspace.json", before_workspace)
    if torch.__version__ != "2.8.0+cu128" or not callable(getattr(torch._C, "_cuda_clearCublasWorkspaces", None)):
        raise RuntimeError("unsupported cuBLAS workspace cleanup API")
    session.check(lease)
    torch._C._cuda_clearCublasWorkspaces()
    session.check(lease)
    torch.cuda.synchronize(0)
    session.check(lease)
    torch.cuda.empty_cache()
    session.check(lease)
    after_cleanup = smoke.memory_guard(torch.cuda, 0, cap)
    bench._write_json(output / "cleanup_after_workspace.json", after_cleanup)
    if (any(survivors.values()) or after_cleanup["allocated_bytes"] != 0
            or after_cleanup["reserved_bytes"] != 0):
        raise RuntimeError("device storage remained after disposal")
    commitment.release_trace("gpu", TRACE_CLAIM)
    commitment.release()
    session.check(lease)
    return {"summary": summary, "isolation": screen["status"], "initial_screen_issue": initial_issue,
            "experiment_status": "completed" if summary is not None else "not_started",
            "model_load_metadata": metadata,
            "cleanup": {"surviving_references": survivors, "before_workspace_clear": before_workspace,
                        "after_workspace_clear": after_cleanup}}


def owner_main(output, cache_dir, smi, exclusive=False, exploratory_desktop=False):
    global _OWNER_USED
    with _OWNER_LOCK:
        if _OWNER_USED:
            raise RuntimeError("the dedicated GPU owner process is single-use")
        _OWNER_USED = True
    from .device_lease import DeviceLease
    from .supervisor import OwnerSession

    lease = None
    session = None
    try:
        session = OwnerSession.from_environment()
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _CUSTODY[0] = lease
        _CUSTODY[1] = session
        session.check(lease)
        bench._write_json(output / "owner.json", {"owner": {"run_id": session.run_id, "pid": os.getpid()}})
        result = _run_owner(session, lease, output, cache_dir, smi, exclusive, exploratory_desktop)
        bench._write_json(output / "candidate.json", {"status": "candidate",
            "owner": {"run_id": session.run_id, "pid": os.getpid()}, "result": result})
        session.attest_clean(lease)
        return 0
    except BaseException as exc:
        _CUSTODY[5] = exc
        drain = {"synchronization": "not_attempted", "clean_teardown": False}
        torch = sys.modules.get("torch")
        if torch is not None and lease is not None and session is not None:
            drain = _drain_after_failure(session, lease, torch.cuda)
        try:
            bench._write_json(output / "owner_failure.json", {"status": "quarantined",
                "error_type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc()[-8192:], "drain": drain,
                "memory_at_failure": getattr(exc, "snapshot", None)})
        except BaseException:
            pass
        return 1


def _offline_assets(cache):
    from .pretrained import ASSETS, REVISION

    snapshot = Path(cache) / "models--distilbert--distilgpt2" / "snapshots" / REVISION
    hashes = {}
    for name, (expected_size, expected_hash) in ASSETS.items():
        digest, size = hashlib.sha256(), 0
        with (snapshot / name).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                size += len(block)
                if size > expected_size:
                    raise RuntimeError("pinned model asset exceeds its expected size")
                digest.update(block)
        if size != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError("pinned model asset failed verification")
        hashes[name] = digest.hexdigest()
    return hashes


def _probe_telemetry_environment(smi, env, output):
    """Check the exact replacement environment before any GPU owner starts."""
    commands = (("gpu", "--query-gpu=timestamp,uuid,memory.total,memory.used,memory.free,utilization.gpu"),
                ("compute_processes", "--query-compute-apps=gpu_uuid,pid,process_name"))
    records = []
    for label, query in commands:
        try:
            result = subprocess.run([smi, query, "--format=csv,noheader,nounits"], env=env,
                capture_output=True, text=True, errors="replace", timeout=10, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            records.append({"query": label, "returncode": result.returncode,
                "stdout": result.stdout[:65536], "stderr": result.stderr[:4096],
                "truncated": len(result.stdout) > 65536 or len(result.stderr) > 4096})
            if result.returncode == 0 and not records[-1]["truncated"]:
                try:
                    if label == "gpu":
                        lines = [line for line in result.stdout.splitlines() if line.strip()]
                        if len(lines) != 1 or bench.parse_telemetry_line(lines[0], 0)["gpu_uuid"] != env["CUDA_VISIBLE_DEVICES"]:
                            raise ValueError("unmatched telemetry GPU")
                    else:
                        for row in csv.reader(result.stdout.splitlines()):
                            if not row or all(not cell.strip() for cell in row):
                                continue
                            uuid, _, _ = bench._parse_compute_process_row(row)
                            if uuid.casefold() != env["CUDA_VISIBLE_DEVICES"].casefold():
                                raise ValueError("unmatched compute process GPU")
                    records[-1]["valid"] = True
                except (KeyError, ValueError, csv.Error):
                    records[-1]["valid"] = False
        except (OSError, subprocess.SubprocessError) as exc:
            records.append({"query": label, "error_type": type(exc).__name__})
    bench._write_bytes(output / "telemetry_preflight.private.json",
        (json.dumps({"queries": records}, sort_keys=True) + "\n").encode("utf-8"))
    bench._write_json(output / "telemetry_preflight.json", {"queries": [
        {key: row[key] for key in ("query", "returncode", "truncated", "error_type", "valid") if key in row}
        for row in records]})
    if any(row.get("returncode") != 0 or row.get("truncated") or row.get("valid") is not True for row in records):
        raise RuntimeError("NVIDIA telemetry is unavailable under the owner environment")


def _source_identity(repo):
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True,
                              timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    if git("status", "--porcelain=v1", "--untracked-files=all").strip():
        raise RuntimeError("a clean committed source checkout is required")
    paths = git("ls-files", "-z", "looking_glass/*.py", "docs/CUDA_BENCHMARK_V1.md",
                "docs/GPU_TAP_LATENCY_SMOKE_V2.md", "pyproject.toml").split(b"\0")
    hashes = {}
    for path_bytes in paths:
        if not path_bytes:
            continue
        relative = path_bytes.decode("utf-8")
        contents = (repo / relative).read_bytes()
        if contents != git("show", revision + ":" + relative):
            raise RuntimeError("working source bytes differ from the recorded Git commit")
        hashes[relative] = hashlib.sha256(contents).hexdigest()
    required = {"looking_glass/gpu_tap_latency_owner.py", "looking_glass/gpu_tap_latency_smoke.py",
                "looking_glass/gpu_tap_latency_entry.py", "looking_glass/cuda_benchmark.py",
                "docs/GPU_TAP_LATENCY_SMOKE_V2.md"}
    if not required.issubset(hashes):
        raise RuntimeError("source inventory is incomplete")
    if git("status", "--porcelain=v1", "--untracked-files=all").strip():
        raise RuntimeError("source checkout changed during verification")
    return {"commit": revision, "source_sha256": hashes}


def _require_isolated_controller():
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix or not os.path.isabs(sys.pycache_prefix)
            or sys._xoptions.get("pycache_prefix") != sys.pycache_prefix
            or os.path.lexists(sys.pycache_prefix)):
        raise RuntimeError("controller requires isolated/no-site/no-bytecode startup and a fresh cache prefix")


def run_controller(*, output, workspace_root, cache_dir, smi, exclusive=False,
                   exploratory_desktop=False, timeout=600):
    """Run a single owner and publish only after its supervisor reaps it."""
    from .device_identity import resolve_device_identity
    from .supervisor import run_supervised

    if type(exclusive) is not bool or type(exploratory_desktop) is not bool:
        raise TypeError("window and exploratory flags must be boolean")
    _require_isolated_controller()
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("output directory must be absolute")
    output.mkdir(exist_ok=False)
    run_id = secrets.token_hex(16)
    stage = "parent_start"
    supervisor = None
    try:
        bench._write_json(output / "parent_started.json", {"protocol": smoke.PROTOCOL,
            "run_id": run_id, "started_monotonic_ns": time.perf_counter_ns(), "clock": "perf_counter_ns",
            "cooperative_window_attested": exclusive, "exploratory_desktop": exploratory_desktop})
        if not exclusive:
            raise RuntimeError("an explicitly coordinated timing window is required")
        stage = "cpu_preflight"
        if "torch" in sys.modules:
            raise RuntimeError("controller must not import PyTorch before supervision")
        root = Path(workspace_root).resolve(strict=True)
        cache = Path(cache_dir).resolve(strict=True)
        repo = Path(__file__).resolve().parents[1]
        if not smi or not Path(smi).is_file():
            raise RuntimeError("nvidia-smi is required")
        source = _source_identity(repo)
        assets = _offline_assets(cache)
        if "torch" in sys.modules:
            raise RuntimeError("CPU asset verification unexpectedly imported PyTorch")
        bench._write_json(output / "cpu_preflight.json", {"source": source, "asset_sha256": assets,
            "torch_preimported": False, "native_cuda_context": "not_checked",
            "input_ids": list(smoke.TOKEN_IDS), "protocol": smoke.PROTOCOL})
        stage = "device_identity"
        inventory = bench._gpu_inventory(smi)
        if not inventory.get("ok") or not inventory.get("uuid"):
            raise RuntimeError("cannot resolve the sole physical device")
        env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")
               if key in os.environ}
        system_root = Path(env.get("SYSTEMROOT", "C:/Windows"))
        program_files = os.environ.get("PROGRAMFILES")
        if not program_files or not Path(program_files).is_absolute() or not Path(program_files).is_dir():
            raise RuntimeError("the Windows PROGRAMFILES directory is required for NVIDIA telemetry")
        env["PROGRAMFILES"] = str(Path(program_files).resolve(strict=True))
        env.update({"PATH": str(system_root / "System32"), "CUDA_VISIBLE_DEVICES": inventory["uuid"],
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                    "HF_HOME": str(cache.parent), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1"})
        stage = "telemetry_preflight"
        _probe_telemetry_environment(smi, env, output)
        stage = "device_identity"
        identity = resolve_device_identity(root, inventory["uuid"], physical_uuids=(inventory["uuid"],),
                                           environ=env)
        bench._write_bytes(output / "controller_identity.private.json",
            (json.dumps({"gpu_uuid": identity.gpu_uuid, "workspace_root": str(root),
                         "lock_path": str(identity.lock_path)}, sort_keys=True) + "\n").encode("utf-8"))
        arguments = ["--owner", "--output", str(output), "--cache-dir", str(cache), "--smi", str(smi)]
        if exclusive:
            arguments.append("--exclusive")
        if exploratory_desktop:
            arguments.append("--exploratory-desktop")
        command, _ = bench.direct_python_module("looking_glass.gpu_tap_latency_owner", arguments, project=repo)
        bench._write_json(output / "launch.json", {"source_commit": source["commit"], "inventory": inventory,
            "environment_policy": {"cuda_visible_devices": inventory["uuid"], "allocator": "native",
                "cublas_workspace_config": ":4096:8", "cpu_threads": 1},
            "supervision_timeout_seconds": timeout, "cooperative_window_attested": exclusive,
            "exploratory_desktop": exploratory_desktop, "programfiles_present": True})
        stage = "supervised_owner"
        result = run_supervised(command, device_lease_path=identity.lock_path, marker_path=identity.marker_path,
            identity=identity, timeout=timeout, lease_timeout=30, cwd=repo, env=env, inherit_environment=False)
        supervisor = {"run_id": result.run_id, "pid": result.pid, "exit_code": result.exit_code,
                      "clean": result.clean, "timed_out": result.timed_out, "reason": result.reason}
        bench._write_json(output / "supervision.json", supervisor)
        if not result.clean or result.exit_code != 0 or result.timed_out:
            raise RuntimeError("owner exit or device teardown was not verified")
        stage = "evidence_validation"
        owner = _read_json(output / "owner.json", 16 * 1024)
        candidate = _read_json(output / "candidate.json", 2 * 1024 ** 2)
        manifest = _read_json(output / "owner_manifest.json", 2 * 1024 ** 2)
        expected_owner = {"run_id": result.run_id, "pid": result.pid}
        if owner.get("owner") != expected_owner or candidate.get("owner") != expected_owner:
            raise RuntimeError("candidate does not match the reaped owner")
        private_owner = _read_json(output / "owner_identity.private.json", 16 * 1024)
        if private_owner.get("owner") != expected_owner or private_owner.get("gpu_uuid") != identity.gpu_uuid:
            raise RuntimeError("runtime GPU binding differs from the selected physical GPU")
        runtime = manifest["runtime"]
        if (manifest.get("owner") != expected_owner or manifest.get("cooperative_window_attested") is not True
                or manifest.get("exploratory_desktop") is not exploratory_desktop
                or manifest.get("protocol") != smoke.PROTOCOL
                or manifest.get("model_id") != "distilbert/distilgpt2"
                or manifest.get("model_revision") != "2290a62682d06624634c1f46a6ad5be0f47f38aa"
                or manifest.get("token_ids") != list(smoke.TOKEN_IDS)
                or not all(type(token) is int for token in manifest.get("token_ids", []))
                or runtime.get("allocator_backend") != "native" or runtime.get("allocator_fraction") != 0.25
                or runtime.get("runtime_uuid_verified_under_lease") is not True
                or type(runtime.get("device_property_total_bytes")) is not int
                or type(runtime.get("cuda_total_bytes")) is not int
                or type(runtime.get("allocator_limit_bytes")) is not int
                or type(runtime.get("allocator_readback")) is not float
                or runtime.get("device_property_total_bytes") != runtime.get("cuda_total_bytes")
                or runtime.get("allocator_limit_bytes") != int(0.25 * runtime["device_property_total_bytes"])
                or runtime.get("allocator_readback") != runtime["allocator_limit_bytes"] / runtime["device_property_total_bytes"]):
            raise RuntimeError("owner allocator or device evidence is inconsistent")
        measured = candidate["result"]
        model_metadata = measured["model_load_metadata"]
        model_runtime = _read_json(output / "model_runtime.json", 64 * 1024)
        cleanup_before = _read_json(output / "cleanup_before_workspace.json", 16 * 1024)
        cleanup_after = _read_json(output / "cleanup_after_workspace.json", 16 * 1024)
        for snapshot in (cleanup_before, cleanup_after):
            checked = smoke.validate_memory(snapshot.get("device_free_bytes"), snapshot.get("device_total_bytes"),
                snapshot.get("allocated_bytes"), snapshot.get("reserved_bytes"), runtime["allocator_limit_bytes"])
            if (type(snapshot.get("allocator_cap_bytes")) is not int or type(snapshot.get("reserve_bytes")) is not int
                    or snapshot != checked or snapshot["device_total_bytes"] != runtime["cuda_total_bytes"]):
                raise RuntimeError("invalid cleanup memory snapshot")
        surviving = measured["cleanup"]["surviving_references"]
        conversion = model_runtime.get("conversion_flags")
        if (model_metadata.get("model_id") != "distilbert/distilgpt2"
                or model_metadata.get("model_revision") != manifest["model_revision"]
                or model_metadata.get("asset_sha256") != assets
                or type(model_metadata.get("parameter_count")) is not int or model_metadata.get("parameter_count") != 81912576
                or model_metadata.get("dtype") != "float32" or model_metadata.get("use_cache") is not False
                or model_metadata.get("attention_implementation") != "eager"
                or model_runtime.get("tied_parameter_preserved") is not True
                or type(conversion) is not dict or set(conversion) != {"overwrite", "swap"}
                or any(value is not False for value in conversion.values())
                or type(model_runtime.get("threads")) is not int or model_runtime.get("threads") != 1
                or type(model_runtime.get("interop_threads")) is not int or model_runtime.get("interop_threads") != 1
                or model_runtime.get("matmul_tf32") is not False or model_runtime.get("cudnn_tf32") is not False
                or measured["cleanup"]["before_workspace_clear"] != cleanup_before
                or measured["cleanup"]["after_workspace_clear"] != cleanup_after
                or type(surviving) is not dict or set(surviving) != {"model", "input"}
                or any(value is not False for value in surviving.values())
                or cleanup_after.get("allocated_bytes") != 0 or cleanup_after.get("reserved_bytes") != 0
                or cleanup_after.get("device_total_bytes") != runtime["cuda_total_bytes"]):
            raise RuntimeError("model provenance or cleanup evidence is inconsistent")
        if candidate.get("status") != "candidate":
            raise RuntimeError("owner did not provide a complete candidate")
        if measured.get("summary") is None:
            raise RuntimeError("initial monitoring was unavailable; no timing was performed: "
                               + str(measured.get("initial_screen_issue")))
        if _source_identity(repo) != source:
            raise RuntimeError("source changed during the owner run")
        saved_telemetry = _read_json(output / "telemetry.private.json", 8 * 1024 ** 2)
        public_telemetry = _read_json(output / "telemetry.json", 8 * 1024 ** 2)
        saved_screen = _read_json(output / "screen.private.json", 2 * 1024 ** 2)
        public_screen = _read_json(output / "screen.json", 2 * 1024 ** 2)
        started_ns, ended_ns = (saved_screen["workload_started_monotonic_ns"],
                                saved_screen["workload_ended_monotonic_ns"])
        queries = saved_screen["process_queries"]
        if (type(started_ns) is not int or type(ended_ns) is not int or not 0 <= started_ns < ended_ns
                or not isinstance(queries, list) or len(queries) != 2
                or queries[0].get("phase") != "before" or queries[1].get("phase") != "after"
                or any(not _valid_process_query(query, phase, inventory["uuid"])
                    or type(query.get("observed_monotonic_ns")) is not int
                    or type(query.get("query_started_monotonic_ns")) is not int
                    or type(query.get("query_ended_monotonic_ns")) is not int
                    or not 0 <= query["query_started_monotonic_ns"] <= query["query_ended_monotonic_ns"]
                    or query["observed_monotonic_ns"] != query["query_started_monotonic_ns"]
                    for query, phase in zip(queries, ("before", "after")))
                or queries[0]["query_ended_monotonic_ns"] > started_ns
                or started_ns - queries[0]["query_started_monotonic_ns"] > 2_000_000_000
                or queries[1]["query_started_monotonic_ns"] < ended_ns):
            raise RuntimeError("process queries do not cover the workload interval")
        verified = verify_observations(output / "observations.jsonl", candidate["result"]["summary"],
            expected_total_bytes=runtime["cuda_total_bytes"], expected_cap_bytes=runtime["allocator_limit_bytes"],
            workload_started_ns=started_ns, workload_ended_ns=ended_ns)
        if public_telemetry != _telemetry_projection(saved_telemetry):
            raise RuntimeError("public telemetry does not match the private device observations")
        if public_screen != _screen_projection(saved_screen):
            raise RuntimeError("public screen does not match the private process observations")
        recomputed_screen = bench.classify_isolation(exclusive, saved_telemetry["observations"],
            saved_screen["workload_started_monotonic_ns"], saved_screen["workload_ended_monotonic_ns"],
            saved_screen["process_queries"], result.pid, inventory["uuid"],
            saved_telemetry["errors"])
        if (saved_screen.get("clock") != "perf_counter_ns"
                or saved_screen.get("classification") != recomputed_screen
                or candidate["result"]["isolation"] != recomputed_screen["status"]):
            raise RuntimeError("candidate isolation does not match saved telemetry")
        status = recomputed_screen["status"]
        interpretation = _activity_interpretation(recomputed_screen, exploratory_desktop)
        experiment_status = ("completed_screened" if status == "no_contamination_observed"
                             else "completed_exploratory_contaminated")
        bench._write_json(output / "results.json", {"status": "succeeded", "protocol": smoke.PROTOCOL,
            "owner": expected_owner, "supervisor": supervisor, "verified_rows": verified,
            "interpretation": interpretation, "experiment_status": experiment_status,
            "strict_screen": status, "physical_exclusivity_established": False,
            "result": candidate["result"], "artifact_inventory": bench._artifact_inventory(output)})
        return {"status": "succeeded", "strict_screen": status, "interpretation": interpretation,
                "experiment_status": experiment_status,
                "result": candidate["result"]}
    except BaseException as exc:
        if not (output / "results.json").exists() and not (output / "failure.json").exists():
            try:
                owner_failure = None
                try:
                    owner_failure = _read_json(output / "owner_failure.json", 32 * 1024)
                except (OSError, ValueError):
                    pass
                owner_error_type = owner_failure.get("error_type") if owner_failure else None
                cancelled = isinstance(exc, (KeyboardInterrupt, SystemExit)) or owner_error_type in (
                    "KeyboardInterrupt", "SystemExit")
                bench._write_json(output / "failure.json", {"status": "cancelled" if cancelled else "failed",
                    "protocol": smoke.PROTOCOL, "stage": stage, "error_type": type(exc).__name__,
                    "message": str(exc), "owner_error_type": owner_error_type,
                    "device_teardown": ("verified_clean" if supervisor and supervisor.get("clean") is True
                        else "unverified" if stage in ("supervised_owner", "evidence_validation") else "not_started"),
                    "artifact_inventory": bench._artifact_inventory(output, strict=False)})
            except BaseException as recording_error:
                if hasattr(exc, "add_note"):
                    exc.add_note("failure evidence could not be fully written: " + type(recording_error).__name__)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--smi", required=True, type=Path)
    parser.add_argument("--exclusive", action="store_true", help="attest the coordinated timing window")
    parser.add_argument("--exploratory-desktop", action="store_true", help="retain labeled contaminated observations")
    args = parser.parse_args(argv)
    if args.owner:
        return owner_main(args.output, args.cache_dir, str(args.smi), exclusive=args.exclusive,
                          exploratory_desktop=args.exploratory_desktop)
    if args.workspace_root is None:
        parser.error("--workspace-root is required")
    try:
        result = run_controller(output=args.output, workspace_root=args.workspace_root,
                                cache_dir=args.cache_dir, smi=str(args.smi), exclusive=args.exclusive,
                                exploratory_desktop=args.exploratory_desktop)
    except BaseException as exc:
        print("Latency smoke failed or is incomplete: " + bench._public_evidence(str(exc)), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "protocol": smoke.PROTOCOL,
                      "strict_screen": result["strict_screen"], "interpretation": result["interpretation"],
                      "experiment_status": result["experiment_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

