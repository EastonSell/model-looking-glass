"""Development-only CPU resource pilot for the proposed causal v2 schedule."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from .causal_experiment import _state_sha256
from .interventions import ActivationCapture, ActivationIntervention
from .model import TinyResidual
from .probes import TraceSession
from .registry import create_run, sha256_file
from .research_cases import SYNTHETIC_CASES, float32_values


V2_MODEL_SEEDS = frozenset(range(31201, 31207))
PILOT_SEEDS = (49351,)
PILOT_CONFIGURATIONS = ((1, "branch.0"), (2, "activation"), (3, "merge"))
PILOT_STRENGTHS = (0.25, 0.75)
LIMIT = 60000


class ForwardBudgetExceeded(RuntimeError):
    """The pilot would exceed its forward budget."""


def planned_forwards(models, cases_per_model, configurations, strengths):
    dimensions = (models, cases_per_model, configurations, strengths)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("plan dimensions must be positive integers")
    return (models * cases_per_model * 2
            + models * configurations * cases_per_model
            + models * configurations * cases_per_model * strengths * 16)


def generate_pilot_inputs(seeds, cases_per_model):
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 or seed >= 2**64
                   or seed == 2301 or seed in V2_MODEL_SEEDS for seed in seeds)):
        raise ValueError("pilot model seeds must be distinct and separate from v1/v2")
    if type(cases_per_model) is not int or cases_per_model <= 0 or cases_per_model % 2:
        raise ValueError("cases_per_model must be a positive even integer")
    if cases_per_model > 65536:
        raise ValueError("too many pilot cases")
    used = {float32_values(case.values) for case in SYNTHETIC_CASES}
    records = []
    for seed in seeds:
        for index in range(cases_per_model):
            for attempt in range(65536):
                payload = (b"MLG-v2-resource-pilot\0" + seed.to_bytes(8, "big")
                           + index.to_bytes(2, "big") + attempt.to_bytes(2, "big"))
                digest = hashlib.sha256(payload).digest()
                magnitude = ((digest[0] & 15) + 1) / 16
                vector = (magnitude if index % 2 == 0 else -magnitude,) + tuple(
                    ((digest[j] & 31) - 16) / 16 for j in range(1, 8))
                if vector not in used:
                    used.add(vector)
                    records.append({"case_id": f"pilot-m{seed}-i{index:02d}", "values": list(vector)})
                    break
            else:
                raise RuntimeError("unable to generate a distinct pilot vector")
    return sorted(records, key=lambda case: case["case_id"])


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _process_memory():
    """Return current and process-lifetime peak resident working-set bytes."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return {"current_working_set_bytes": int(counters.WorkingSetSize),
                "process_peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "method": "Windows GetProcessMemoryInfo PeakWorkingSetSize"}
    if sys.platform.startswith("linux"):
        import resource
        return {"current_working_set_bytes": None,
                "process_peak_working_set_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "method": "Linux getrusage ru_maxrss (KiB)"}
    raise RuntimeError("peak process working set is unsupported on this platform")


def _verify_denormal_policy():
    """Reject flush-to-zero on this execution thread without changing caller state."""
    minimum_subnormal = torch.tensor([1], dtype=torch.int32, device="cpu").view(torch.float32)
    if torch.equal(minimum_subnormal, torch.zeros_like(minimum_subnormal)):
        raise RuntimeError("CPU denormal preservation is not enabled")
    minimum_normal = torch.tensor([torch.finfo(torch.float32).tiny], dtype=torch.float32, device="cpu")
    half = torch.mul(minimum_normal, 0.5)
    if half.view(torch.int32).item() != 0x00400000:
        raise RuntimeError("CPU denormal output preservation is not enabled")


class _Counter:
    def __init__(self, limit):
        self.limit = limit
        self.total = 0
        self.by_phase = {"clean_trace": 0, "clean_target_capture": 0, "condition": 0}

    def forward(self, model, x, *, phase, perturb=None):
        if self.total >= self.limit:
            raise ForwardBudgetExceeded("pilot forward budget exhausted")
        self.total += 1
        self.by_phase[phase] += 1
        return model(x, route=0, perturb=perturb)


def _sparse_trace(model, x, counter, *, phase, perturb=None, run_id):
    with TraceSession(model, mode="direct", placement="sparse", run_id=run_id,
                      max_events=4, max_tensor_bytes=128) as trace:
        output = counter.forward(model, x, phase=phase, perturb=perturb).detach().clone()
    expected = [f"block.{i}.output" for i in range(4)]
    if len(trace.events) != 4 or trace.tensor_bytes != 128:
        raise RuntimeError("invalid sparse trace coverage or payload")
    for i, event in enumerate(trace.events):
        if (event.node_id != expected[i] or event.sequence != i or event.occurrence != 0
                or event.shape != (1, 8) or event.dtype != "torch.float32" or event.device != "cpu"
                or not torch.isfinite(event.value).all()):
            raise RuntimeError("invalid sparse trace metadata or values")
    if output.shape != (1, 2) or not torch.isfinite(output).all():
        raise RuntimeError("invalid model output")
    return output, trace.events


def _capture(model, x, path, counter, *, phase, perturb=None):
    with ActivationCapture(model, path, invocation=0, max_tensor_bytes=32, max_invocations=1) as capture:
        output = counter.forward(model, x, phase=phase, perturb=perturb).detach().clone()
    return output, capture.value


def _patch(model, x, path, counter, *, source=None, perturb=None, operation="patch"):
    with ActivationIntervention(model, path, operation=operation, value=source, invocation=0,
                                max_tensor_bytes=32, max_invocations=1):
        return counter.forward(model, x, phase="condition", perturb=perturb).detach().clone()


def _bits_equal(actual, expected):
    if (not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor)
            or actual.dtype != torch.float32 or expected.dtype != torch.float32
            or actual.device.type != "cpu" or expected.device.type != "cpu" or actual.shape != expected.shape):
        return False
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise ValueError("exact comparison requires finite float32 tensors")
    def canonical_bits(tensor):
        bits = tensor.contiguous().view(torch.int32)
        zero = torch.bitwise_and(bits, 0x7fffffff) == 0
        return torch.where(zero, torch.zeros_like(bits), bits)
    return bool(torch.equal(canonical_bits(actual), canonical_bits(expected)))


def _equal(actual, expected, description):
    if not _bits_equal(actual, expected):
        raise RuntimeError(f"pilot control mismatch: {description}")


def _case_tensor(case):
    return torch.tensor([case["values"]], dtype=torch.float32, device="cpu")


def _target_path(block, operation):
    return f"blocks.{block}.branch_taps.0" if operation == "branch.0" else f"blocks.{block}.{operation}_tap"


def _run_workload(model, inputs, counter):
    cached = {}
    for case in inputs:
        x = _case_tensor(case)
        clean, first = _sparse_trace(model, x, counter, phase="clean_trace", run_id=f"{case['case_id']}-a")
        repeat, second = _sparse_trace(model, x, counter, phase="clean_trace", run_id=f"{case['case_id']}-b")
        _equal(clean, repeat, "repeated clean output")
        for a, b in zip(first, second):
            _equal(a.value, b.value, "repeated clean sparse trace")
        cached[case["case_id"]] = (clean, {event.node_id: event.value for event in first})

    condition_count = 0
    for block, operation in PILOT_CONFIGURATIONS:
        label = f"block.{block}.{operation}"
        target_path = _target_path(block, operation)
        for case in inputs:
            x = _case_tensor(case)
            clean, clean_values = cached[case["case_id"]]
            capture_output, clean_target = _capture(model, x, target_path, counter, phase="clean_target_capture")
            _equal(capture_output, clean, "clean target capture output")
            for strength in PILOT_STRENGTHS:
                polarity = 1 if case["values"][0] > 0 else -1
                delta = strength * polarity
                perturb = (label, delta)
                corrupt, corrupt_trace = _sparse_trace(model, x, counter, phase="condition", perturb=perturb,
                                                        run_id=f"{case['case_id']}-changed-{counter.total}")
                corrupt_values = {event.node_id: event.value for event in corrupt_trace}
                _equal(corrupt_values["block.0.output"], clean_values["block.0.output"],
                       "unchanged upstream sparse trace")
                capture_output, corrupt_target = _capture(model, x, target_path, counter,
                                                           phase="condition", perturb=perturb)
                _equal(capture_output, corrupt, "changed target capture output")
                expected_target = clean_target.clone()
                expected_target[0, 0] += delta
                _equal(corrupt_target, expected_target, "injected target delta")
                if _bits_equal(corrupt_target, clean_target):
                    raise RuntimeError("pilot injection was numerically masked at target")
                for site in (1, 2, 3):
                    node = f"block.{site}.output"
                    site_path = f"blocks.{site}.output_tap"
                    restored = _patch(model, x, site_path, counter, source=clean_values[node], perturb=perturb)
                    reversed_output = _patch(model, x, site_path, counter, source=corrupt_values[node])
                    null = _patch(model, x, site_path, counter, perturb=perturb, operation="null")
                    identity = _patch(model, x, site_path, counter, source=corrupt_values[node], perturb=perturb)
                    _equal(restored, corrupt if site < block else clean, "forward patch boundary")
                    _equal(reversed_output, clean if site < block else corrupt, "reverse patch boundary")
                    _equal(null, corrupt, "null full logits")
                    _equal(identity, corrupt, "identity full logits")
                upstream = _patch(model, x, "blocks.0.output_tap", counter,
                                  source=clean_values["block.0.output"], perturb=perturb)
                _equal(upstream, corrupt, "upstream full logits")
                after = counter.forward(model, x, phase="condition").detach().clone()
                _equal(after, clean, "post-intervention clean output")
                condition_count += 1
    return condition_count


def _provenance(require_clean):
    root = Path(__file__).resolve().parents[1]
    protocol = root / "docs" / "CAUSAL_V2_PILOT_PROTOCOL.md"
    v2_protocol = root / "docs" / "CAUSAL_CONFIRMATION_V2.md"
    v2_numeric = root / "docs" / "CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                             text=True, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, True
    if require_clean and (not revision or dirty):
        raise RuntimeError("recorded pilot requires a clean committed checkout")
    source_paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        if name == "looking_glass" or name.startswith("looking_glass."):
            module_path = getattr(module, "__file__", None)
            if module_path:
                source_paths.add(Path(module_path).resolve())
    if require_clean and any(not path.is_relative_to(root) for path in source_paths):
        raise RuntimeError("recorded pilot imports project sources outside this checkout")
    if require_clean:
        for path in source_paths | {protocol, v2_protocol, v2_numeric}:
            relative = path.relative_to(root).as_posix()
            try:
                committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root,
                                                    stderr=subprocess.DEVNULL)
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError(f"pilot source is not tracked at HEAD: {relative}") from error
            if hashlib.sha256(committed).hexdigest() != sha256_file(path):
                raise RuntimeError(f"pilot source differs from committed bytes: {relative}")
    source_hashes = {path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path): sha256_file(path)
                     for path in sorted(source_paths)}
    return {"revision": revision, "dirty": dirty,
            "source_sha256": source_hashes, "v2_protocol_sha256": sha256_file(v2_protocol),
            "v2_numeric_clarification_sha256": sha256_file(v2_numeric),
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torch_build": torch.__config__.show(), "platform": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor(),
                        "device": "cpu", "workload_intraop_threads": 1, "denormal_policy": "preserve",
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "float32_matmul_precision": torch.get_float32_matmul_precision()}}, sha256_file(protocol)


def run_pilot(path, *, max_forwards=LIMIT, require_clean=False):
    """Run the small development-only pilot, saving its exclusion inputs first."""
    entry_wall_start, entry_cpu_start = time.perf_counter(), time.process_time()
    if type(max_forwards) is not int or not 0 < max_forwards <= LIMIT:
        raise ValueError("max_forwards must be an integer between 1 and 60000")
    if torch.is_autocast_enabled("cpu"):
        raise RuntimeError("CPU autocast is incompatible with the pilot")
    _verify_denormal_policy()
    provenance, protocol_hash = _provenance(require_clean)
    initialization_threads = torch.get_num_threads()
    provenance["runtime"]["initialization_intraop_threads"] = initialization_threads
    inputs = generate_pilot_inputs(PILOT_SEEDS, 2)
    input_bytes = _json_bytes({"inputs": inputs})
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    plan = {"status": "development-only", "seeds": list(PILOT_SEEDS),
            "configurations": [[block, operation] for block, operation in PILOT_CONFIGURATIONS],
            "strengths": list(PILOT_STRENGTHS), "cases_per_model": 2,
            "expected_forwards": planned_forwards(1, 2, 3, 2),
            "v2_planned_forwards": planned_forwards(6, 32, 9, 2), "max_forwards": max_forwards,
            "pilot_input_sha256": input_sha256, "denormal_policy": "preserve"}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(PILOT_SEEDS[0])
        model = TinyResidual(width=8, depth=4).eval()
    state_sha256 = _state_sha256(model)
    run = create_run(path, protocol={"id": "synthetic-v2-resource-pilot-v1", "sha256": protocol_hash,
                                    "settings": plan},
                     inputs={"pilot_input_sha256": input_sha256, "cases": len(inputs)},
                     model={"kind": "TinyResidual", "width": 8, "depth": 4, "seed": PILOT_SEEDS[0],
                            "state_sha256": state_sha256}, code=provenance,
                     metadata={"status": "development-only", "python": platform.python_version(),
                               "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
                     artifact_limits={"pilot_inputs.json": 1048576, "plan.json": 131072,
                                      "results.json": 131072, "partial.json": 131072},
                     required_artifacts=["pilot_inputs.json", "plan.json", "results.json"],
                     max_total_bytes=1441792)
    counter = _Counter(max_forwards)
    previous_threads = None
    restored = False
    initial_memory = None
    plan_bytes = None
    result_bytes = None
    try:
        plan_bytes = _json_bytes(plan)
        run.write_artifact("pilot_inputs.json", input_bytes)
        run.write_artifact("plan.json", plan_bytes)
        initial_memory = _process_memory()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        _verify_denormal_policy()
        workload_wall_start, workload_cpu_start = time.perf_counter(), time.process_time()
        with torch.no_grad():
            conditions = _run_workload(model, inputs, counter)
        workload_wall_seconds = time.perf_counter() - workload_wall_start
        workload_cpu_seconds = time.process_time() - workload_cpu_start
        entry_to_workload_end_wall_seconds = time.perf_counter() - entry_wall_start
        entry_to_workload_end_cpu_seconds = time.process_time() - entry_cpu_start
        memory = _process_memory()
        if counter.total != plan["expected_forwards"] or conditions != 12:
            raise RuntimeError("observed schedule count differs from pilot plan")
        torch.set_num_threads(previous_threads)
        restored = True
        result = {"status": "development-only", "pilot_input_sha256": input_sha256,
                  "observed_forwards": counter.total, "forwards_by_phase": counter.by_phase,
                  "conditions": conditions,
                  "initialization_intraop_threads": initialization_threads,
                  "entry_to_workload_start_wall_seconds": workload_wall_start - entry_wall_start,
                  "entry_to_workload_start_cpu_seconds": workload_cpu_start - entry_cpu_start,
                  "entry_to_workload_end_wall_seconds": entry_to_workload_end_wall_seconds,
                  "entry_to_workload_end_cpu_seconds": entry_to_workload_end_cpu_seconds,
                  "workload_wall_seconds": workload_wall_seconds, "workload_cpu_seconds": workload_cpu_seconds,
                  "v2_planned_forwards": plan["v2_planned_forwards"],
                  "linear_extrapolated_v2_wall_seconds": workload_wall_seconds * plan["v2_planned_forwards"] / counter.total,
                  "initial_memory": initial_memory, "denormal_policy": "preserve",
                  "denormal_probe_passed": True, **memory,
                  "limitations": ["development inputs only; no v2 held-out data generated",
                                  "linear extrapolation is not a bound or isolated benchmark",
                                  "peak working set is process-lifetime resident memory"]}
        result_bytes = _json_bytes(result)
        run.write_artifact("results.json", result_bytes)
        run.finalize("succeeded", {"observed_forwards": counter.total})
        return result
    except BaseException as error:
        try:
            verified = run.status()
            outcome_state = verified["state"]
        except Exception as verification_error:
            raise RuntimeError("pilot outcome state unknown; registry verification failed") from verification_error
        if outcome_state == "succeeded":
            if plan_bytes is None or result_bytes is None:
                raise RuntimeError("pilot outcome state unknown: success preceded complete evidence") from error
            outcome = verified["outcome"]
            expected_artifacts = {"pilot_inputs.json": hashlib.sha256(input_bytes).hexdigest(),
                                  "plan.json": hashlib.sha256(plan_bytes).hexdigest(),
                                  "results.json": hashlib.sha256(result_bytes).hexdigest()}
            if (outcome["details"] != {"observed_forwards": counter.total}
                    or set(outcome["artifacts"]) != set(expected_artifacts)
                    or any(outcome["artifacts"][name]["sha256"] != digest
                           for name, digest in expected_artifacts.items())):
                raise RuntimeError("pilot outcome state unknown: committed evidence differs") from error
            raise RuntimeError("pilot committed as succeeded but subsequent cleanup failed") from error
        if outcome_state in ("failed", "cancelled"):
            raise
        if outcome_state != "incomplete":
            raise RuntimeError(f"pilot outcome state unknown: {outcome_state}") from error
        restore_error = None
        if not restored and previous_threads is not None:
            try:
                torch.set_num_threads(previous_threads)
            except BaseException as exc:
                restore_error = exc
        partial_error = None
        try:
            run.write_artifact("partial.json", _json_bytes({"status": "incomplete",
                               "observed_forwards": counter.total, "forwards_by_phase": counter.by_phase,
                               "entry_elapsed_wall_seconds": time.perf_counter() - entry_wall_start,
                               "entry_elapsed_cpu_seconds": time.process_time() - entry_cpu_start,
                               "pilot_input_sha256": input_sha256}))
        except Exception as exc:
            partial_error = exc
        try:
            run.finalize("cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                         {"error_type": type(error).__name__, "message": str(error),
                          "partial_error": str(partial_error) if partial_error else None,
                          "restore_error": str(restore_error) if restore_error else None})
        except Exception as exc:
            raise error from exc
        if partial_error is not None:
            raise error from partial_error
        if restore_error is not None:
            raise error from restore_error
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new pilot run directory")
    args = parser.parse_args()
    if not torch.set_flush_denormal(False):
        raise RuntimeError("this CPU does not support the required denormal policy")
    _verify_denormal_policy()
    result = run_pilot(args.output, require_clean=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
