"""Small synchronized direct-versus-gradient tap timing experiment.

The caller must own the supervised GPU lease, enforce the allocator cap, and
dispose the model before attesting clean. This module only implements the
fixed measurement body; it does not launch or authorize a GPU owner.
"""

import math
import random
import statistics
import time

from . import cuda_benchmark as bench

PROTOCOL = "distilgpt2-tap-latency-smoke-v1"
MODES = ("none", "direct", "gradient")
TOKEN_IDS = (15496, 995)
SEED = 17
WARMUP_ROUNDS = 2
MEASURED_ROUNDS = 6
TRACE_LIMIT = 64 * 1024
HEADROOM = 256 * 1024 ** 2
OUTSIDE_CAP_ALLOWANCE = 256 * 1024 ** 2


class MemoryPressureError(RuntimeError):
    """A failed live guard, with the snapshot available for failure evidence."""

    def __init__(self, message, snapshot):
        super().__init__(message)
        self.snapshot = snapshot


def schedule(seed, rounds):
    """Randomize each complete round, retaining paired mode comparisons."""
    if type(seed) is not int or type(rounds) is not int or rounds <= 0:
        raise ValueError("invalid schedule")
    generator = random.Random(seed)
    rows = []
    for number in range(rounds):
        modes = list(MODES)
        generator.shuffle(modes)
        rows.extend({"round": number, "position": position, "mode": mode}
                    for position, mode in enumerate(modes))
    return rows


def build_workload(model, input_ids):
    """Trace one block output at the last token of a fixed two-token input."""
    import torch
    from .transformer_trace import GPT2TraceSession

    if (not isinstance(input_ids, torch.Tensor) or tuple(input_ids.shape) != (1, 2)
            or input_ids.dtype != torch.int64 or input_ids.detach().cpu().tolist() != [list(TOKEN_IDS)]):
        raise ValueError("input differs from the frozen two-token case")

    def invoke():
        return model(input_ids=input_ids, use_cache=False, return_dict=True).logits

    def factory(mode):
        return GPT2TraceSession(model, run_id=PROTOCOL, mode=mode, targets=("block",),
            layers=(0,), samples=(0,), tokens=(1,), capture_device="source",
            max_events=2, max_trace_bytes=TRACE_LIMIT)

    def identity(event):
        return (event.module_path, event.module_kind, event.layer_index, event.forward_index,
                event.invocation, tuple(event.output_path), tuple(event.batch_positions),
                tuple(event.token_positions))

    return invoke, factory, identity


def validate_sample(mode, output, session, *, identity, expected, device):
    """Validate every sample without asserting bitwise equality."""
    import torch

    if mode not in MODES:
        raise ValueError("invalid tap mode")
    if (not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 2, 50257)
            or output.dtype != torch.float32 or str(output.device) != str(device)
            or not bool(torch.isfinite(output).all())):
        raise AssertionError("nonfinite or invalid output layout")
    if mode == "none":
        if session is not None:
            raise AssertionError("baseline unexpectedly retained a trace")
        return
    if session is None or session.successful is not True or len(session.events) != 1:
        raise AssertionError("trace did not complete with one event")
    event = session.events[0]
    if identity(event) != expected or event.sequence != 0 or event.run_id != PROTOCOL or event.capture_mode != mode:
        raise AssertionError("trace identity does not match the preflight")
    value = event.value
    if (tuple(event.shape) != (1, 2, 768) or event.dtype != "torch.float32"
            or event.device != str(device) or not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (1, 1, 768) or value.dtype != torch.float32
            or str(value.device) != str(device) or session.tensor_bytes != 3072
            or event.tensor_bytes != 3072):
        raise AssertionError("trace tensor layout or byte count is invalid")


def validate_memory(free, total, allocated, reserved, cap_bytes):
    """Validate a sampled memory state against the whole-run commitment."""
    bench._validate_memory_snapshot(free, total, allocated, reserved)
    if type(cap_bytes) is not int or not HEADROOM < cap_bytes <= total:
        raise ValueError("invalid allocator cap")
    reserve = max(1024 ** 3, (15 * total + 99) // 100)
    result = {"device_free_bytes": free, "device_total_bytes": total,
            "allocated_bytes": allocated, "reserved_bytes": reserved,
            "allocator_cap_bytes": cap_bytes, "reserve_bytes": reserve}
    required = reserve + HEADROOM + OUTSIDE_CAP_ALLOWANCE + max(0, cap_bytes - reserved)
    if free < required or reserved > cap_bytes - HEADROOM:
        raise MemoryPressureError("insufficient live memory headroom", result)
    return result


def memory_guard(cuda, device, cap_bytes):
    """Check live headroom; the allocator cap cannot control foreign memory."""
    free, total = cuda.mem_get_info(device)
    return validate_memory(free, total, cuda.memory_allocated(device), cuda.memory_reserved(device), cap_bytes)


def run_samples(*, invoke, factory, identity, cuda, device, cap_bytes, expected_total_bytes,
                check_lease, record):
    """Preflight, calibrate and run short paired trials on the leased device.

    ``record`` must persist each observation and its separate postcheck outside
    the timed interval. An observation without a matching passed postcheck is
    not a validated sample. Warmups are excluded from the paired summary.
    """
    if not callable(check_lease) or not callable(record):
        raise TypeError("lease and evidence callbacks are required")
    clock = time.get_clock_info("perf_counter")
    if (not clock.monotonic or not math.isfinite(clock.resolution)
            or not 0 < clock.resolution <= 0.000001):
        raise RuntimeError("a monotonic wall clock with at most 1 microsecond resolution is required")
    minimum_wall_ns = math.ceil(100 * clock.resolution * 1_000_000_000)
    check_lease()
    cuda.synchronize(device)
    before = memory_guard(cuda, device, cap_bytes)
    reference_total = before["device_total_bytes"]
    if type(expected_total_bytes) is not int or expected_total_bytes != reference_total:
        raise RuntimeError("preflight device total differs from the owner's bound CUDA total")
    cuda.reset_peak_memory_stats(device)
    def complete_preflight():
        check_lease()
        cuda.synchronize(device)
    preflight = bench.verify_equivalence(invoke, factory, identity, completion=complete_preflight)
    expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
    if (preflight["events_per_mode"] != 1 or preflight["value_tensor_bytes"] != 3072
            or preflight["event_identities"] != [expected]
            or preflight["value_layouts"] != [{"shape": [1, 1, 768],
                "dtype": "torch.float32", "device": str(device)}]):
        raise AssertionError("preflight does not match the fixed tap")
    cuda.synchronize(device)
    peak_allocated, peak_reserved = cuda.max_memory_allocated(device), cuda.max_memory_reserved(device)
    record({"stage": "preflight", "observation_id": "preflight", "postcheck_status": "pending",
            "checks": preflight, "memory_before": before,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved})
    check_lease()
    after = memory_guard(cuda, device, cap_bytes)
    if (before["device_total_bytes"] != after["device_total_bytes"]
            or type(peak_allocated) is not int or type(peak_reserved) is not int
            or peak_allocated < max(before["allocated_bytes"], after["allocated_bytes"])
            or peak_reserved < max(before["reserved_bytes"], after["reserved_bytes"])
            or not 0 <= peak_allocated <= peak_reserved <= cap_bytes - HEADROOM):
        raise RuntimeError("preflight memory totals, peaks or allocator headroom are invalid")
    record({"stage": "postcheck", "observation_id": "preflight", "status": "passed",
            "memory_after": after})

    current = cuda.current_stream(device)
    if current != cuda.default_stream(device):
        raise RuntimeError("default CUDA stream required")
    start, end = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
    start.record(current)
    end.record(current)
    cuda.synchronize(device)
    def validate(mode, output, session):
        validate_sample(mode, output, session, identity=identity, expected=expected, device=device)
    def validate_before_sample(free, total, allocated, reserved):
        if total != reference_total:
            raise RuntimeError("device total changed before the timed forward")
        validate_memory(free, total, allocated, reserved, cap_bytes)

    warmups, measured = [], []
    for kind, seed, rounds in (("warmup", SEED ^ 0x5A17, WARMUP_ROUNDS),
                               ("measured", SEED, MEASURED_ROUNDS)):
        for item in schedule(seed, rounds):
            check_lease()
            before = memory_guard(cuda, device, cap_bytes)
            if before["device_total_bytes"] != reference_total:
                raise RuntimeError("device total memory changed after preflight")
            result = bench.measure_sample(invoke, factory, item["mode"], device, cuda,
                start, end, validate=validate,
                memory_check=validate_before_sample)
            observation_id = f"{kind}:{item['round']}:{item['position']}"
            row = {"stage": kind, "observation_id": observation_id, "postcheck_status": "pending",
                   "seed": SEED, "schedule_seed": seed, **item, **result}
            record(row)
            if (type(result.get("wall_ns")) is not int or result["wall_ns"] < minimum_wall_ns):
                raise RuntimeError("sample wall interval is too short for clock resolution")
            for endpoint in ("before", "after"):
                validate_memory(result[f"device_free_{endpoint}_bytes"], result["device_total_bytes"],
                    result[f"allocated_{endpoint}_bytes"], result[f"reserved_{endpoint}_bytes"], cap_bytes)
            if result["peak_reserved_bytes"] > cap_bytes - HEADROOM:
                raise RuntimeError("sample exceeded the conservative allocator headroom")
            check_lease()
            after = memory_guard(cuda, device, cap_bytes)
            if (result["device_total_bytes"] != reference_total
                    or after["device_total_bytes"] != reference_total):
                raise RuntimeError("device total memory changed around a sample")
            record({"stage": "postcheck", "observation_id": observation_id, "status": "passed",
                    "memory_before": before, "memory_after": after})
            (warmups if kind == "warmup" else measured).append(row)

    by_round = {}
    for row in measured:
        by_round.setdefault(row["round"], {})[row["mode"]] = row["wall_ns"]
    if len(by_round) != MEASURED_ROUNDS or any(set(values) != set(MODES) for values in by_round.values()):
        raise RuntimeError("paired timing rounds are incomplete")
    ratios = [values["gradient"] / values["direct"] for values in by_round.values()]
    direct_baseline_ratios = [values["direct"] / values["none"] for values in by_round.values()]
    if not all(math.isfinite(value) for value in ratios + direct_baseline_ratios):
        raise RuntimeError("invalid paired timing ratio")
    return {"protocol": PROTOCOL, "warmup_samples": len(warmups), "measured_samples": len(measured),
            "clock": {"name": "perf_counter_ns", "implementation": clock.implementation,
                      "resolution_seconds": clock.resolution, "monotonic": clock.monotonic,
                      "minimum_wall_ns": minimum_wall_ns},
            "median_wall_ns": {mode: statistics.median(row["wall_ns"] for row in measured
                                                        if row["mode"] == mode) for mode in MODES},
            "median_paired_gradient_over_direct": statistics.median(ratios),
            "median_paired_direct_over_none": statistics.median(direct_baseline_ratios),
            "limitations": ["one model and one input; paired rounds are correlated",
                "wall timing includes output validation and synchronization",
                "CUDA events cover the checked default stream, not unjoined streams",
                "sampled memory and allocator peaks are not a guarantee against OOM",
                "finite warmup observations do not establish a general resource bound"]}
