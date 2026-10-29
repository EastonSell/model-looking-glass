"""Versioned, opt-in offline DistilGPT2 tracing correctness smoke.

This is not a benchmark. CUDA execution requires an externally coordinated
exclusive lease and must run in a fresh process.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
import torch

from .transformer_trace import GPT2TraceSession, _note_cleanup
from .model_ownership import ModelLease

PROTOCOL_VERSION = "pretrained-trace-smoke-v1"
TOKENS = (464, 6766, 318, 4171, 13)
LAYERS = (0, 2, 5)
TARGETS = ("attn", "mlp", "block")
POSITIONS = (0, 1)
TRACE_CAP = 131072
_SOURCE_PATHS = (
    "docs/PRETRAINED_SMOKE_V1.md", "looking_glass/pretrained_smoke.py",
    "looking_glass/pretrained.py", "looking_glass/transformer_trace.py",
    "looking_glass/trace_schema.py", "looking_glass/probes.py",
    "looking_glass/model_ownership.py", "looking_glass/__init__.py",
)


class SmokeFailure(RuntimeError):
    """A required correctness or resource check failed."""


def _require(condition, message):
    if not condition:
        raise SmokeFailure(message)


def _hash_float32(tensor):
    _require(tensor.dtype == torch.float32, "hash contract requires float32")
    array = tensor.detach().to("cpu").contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _equal_float32_bits(first, second):
    """Compare float32 encodings; float comparisons may flush subnormals."""
    if first.dtype != torch.float32 or second.dtype != torch.float32 or first.shape != second.shape:
        return False
    first_bits = first.detach().contiguous().view(torch.int32)
    second_bits = second.detach().contiguous().view(torch.int32)
    return bool(torch.equal(first_bits, second_bits))


def _cpu_subnormal_multiply_bits():
    smallest = torch.tensor([1], dtype=torch.int32, device="cpu").view(torch.float32)
    return int((smallest * 1.0).view(torch.int32).item())


def _expected_sites():
    return [(f"transformer.h.{layer}" + ("" if kind == "block" else "." + kind),
             kind, layer)
            for layer in LAYERS for kind in TARGETS]


def check_model(model, input_ids, *, device, after_forward=None, study_id=None):
    """Run fixed comparisons on an already-created eager model.

    Primarily public so an independent replay can execute the identical checks.
    Raises on any missing or nonmatching observation; returns no raw tensors.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if tuple(input_ids.shape) != (1, 5) or tuple(input_ids.cpu().reshape(-1).tolist()) != TOKENS:
        raise ValueError("input_ids do not match the frozen public input")
    if study_id is None:
        study_id = str(uuid.uuid4())
    if not isinstance(study_id, str) or not study_id.strip():
        raise ValueError("study_id must be a nonempty string")
    sites = _expected_sites()
    expected = [(forward, path, kind, layer)
                for forward in range(2) for path, kind, layer in sites]
    reference = []
    forward_index = 0
    direct_values = None
    reference_failed = False
    owner_thread = threading.current_thread()

    def forward():
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                       use_cache=False, return_dict=True)
        if after_forward is not None:
            after_forward()
        _require(output.logits.dtype == torch.float32 and
                 tuple(output.logits.shape) == (1, 5, model.config.vocab_size),
                 "unexpected model logit shape or dtype")
        logits = output.logits.detach().to(device="cpu", copy=True)
        del output
        _require(bool(torch.isfinite(logits).all()), "nonfinite model logits")
        return logits

    def hook_for(path, kind, layer):
        def capture(_module, _args, output):
            nonlocal reference_failed
            try:
                _require(threading.current_thread() is owner_thread,
                         "foreign thread invoked a reference target")
                lease.verify_tree()
                if len(reference) >= 18:
                    raise SmokeFailure("too many independent reference events")
                if isinstance(output, torch.Tensor):
                    source, selector = output, ()
                elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                    source, selector = output[0], (0,)
                else:
                    raise SmokeFailure("unsupported independent reference output")
                _require(source.ndim == 3 and tuple(source.shape[:2]) == (1, 5),
                         "unexpected reference output shape")
                selected = source.detach()[:, list(POSITIONS), :].to("cpu", copy=True)
                _require(bool(torch.isfinite(selected).all()), "nonfinite reference activation")
                reference.append((forward_index, path, kind, layer, selector, selected))
            except BaseException:
                reference_failed = True
                raise
            return None
        return capture

    try:
        with torch.inference_mode():
            # Reference hooks are also model-global, so hold the same exclusive
            # process-local lease used by the adapter and intervention APIs.
            handles = [None] * len(sites)
            lease = ModelLease(model)
            uncertain = False
            try:
                lease.acquire()
                lease.verify_tree()
                baseline = [forward(), forward()]
                lease.verify_tree()
                _require(_equal_float32_bits(baseline[0], baseline[1]), "baseline repeated forwards differ")
                for index, (path, kind, layer) in enumerate(sites):
                    # Any interruption until the handle reaches a preallocated
                    # slot may have left an installed hook without a handle.
                    uncertain = True
                    handle = model.get_submodule(path).register_forward_hook(hook_for(path, kind, layer))
                    if not isinstance(handle, torch.utils.hooks.RemovableHandle):
                        raise SmokeFailure("reference registration returned an invalid handle")
                    handles[index] = handle
                    uncertain = False
                reference_logits = []
                for forward_index in range(2):
                    lease.verify_tree()
                    reference_logits.append(forward())
                    lease.verify_tree()
                    _require(not reference_failed, "reference capture encountered an error")
            finally:
                primary = sys.exc_info()[1]
                cleanup_errors = []
                for handle in handles:
                    if handle is not None:
                        try:
                            handle.remove()
                        except BaseException as error:
                            cleanup_errors.append(error)
                if not uncertain and not cleanup_errors:
                    try:
                        lease.release()
                    except BaseException as error:
                        cleanup_errors.append(error)
                        # Release is token-checked and idempotent; retry once
                        # because interruption may precede or follow its swap.
                        try:
                            lease.release()
                        except BaseException as retry_error:
                            cleanup_errors.append(retry_error)
                if uncertain or cleanup_errors:
                    message = "reference cleanup or lease release was uncertain; do not reuse the model"
                    if cleanup_errors:
                        message += ": " + "; ".join(repr(error) for error in cleanup_errors)
                    if primary is None:
                        if cleanup_errors:
                            raise cleanup_errors[0]
                        raise SmokeFailure(message)
                    _note_cleanup(primary, message)
            _require(len(reference) == 18, "missing independent reference events")
            _require([(f, p, k, l) for f, p, k, l, _, _ in reference] == expected,
                     "independent reference event order mismatch")
            _require(all(_equal_float32_bits(a, b) for a, b in zip(baseline, reference_logits)),
                     "reference hooks changed model output")
            baseline_hashes = [_hash_float32(x) for x in baseline]
            reference_hashes = [_hash_float32(x) for x in reference_logits]
            result = {
                "passed": True,
                "baseline_logit_sha256": baseline_hashes,
                "baseline_logit_shape": [1, 5, model.config.vocab_size],
                "baseline_logit_dtype": "torch.float32",
                "baseline_repeat_equal": True,
                "reference_logit_sha256": reference_hashes,
                "reference_logits_equal": True,
                "reference_value_sha256": [_hash_float32(item[5]) for item in reference],
                "direct_gradient_equal": False,
                "modes": {},
            }
            for mode in ("direct", "gradient", "id"):
                run_id = f"{study_id}-{mode}"
                outputs = []
                with GPT2TraceSession(model, run_id=run_id, layers=LAYERS, targets=TARGETS,
                                      samples=(0,), tokens=POSITIONS, mode=mode,
                                      max_events=18, max_trace_bytes=TRACE_CAP,
                                      capture_device="cpu") as trace:
                    for _ in range(2):
                        outputs.append(forward())
                _require(trace.successful and len(trace.events) == 18,
                         f"{mode} trace incomplete")
                _require(all(_equal_float32_bits(a, b) for a, b in zip(baseline, outputs)),
                         f"{mode} changed model logits")
                event_records = []
                current_values = []
                for event, ref, expect in zip(trace.events, reference, expected):
                    fwd, path, kind, layer = expect
                    _require((event.forward_index, event.module_path, event.module_kind,
                              event.layer_index) == expect, f"{mode} metadata order mismatch")
                    _require(event.sequence == len(event_records) and event.invocation == 0
                             and event.run_id == run_id and event.capture_mode == mode,
                             f"{mode} invocation/sequence mismatch")
                    _require(event.output_path == ref[4] and event.batch_positions == (0,)
                             and event.token_positions == POSITIONS
                             and event.shape == (1, 5, model.config.n_embd)
                             and event.dtype == "torch.float32"
                             and event.device == ("cpu" if device == "cpu" else "cuda:0"),
                             f"{mode} source metadata mismatch")
                    _require(event.output_path == ((0,) if kind in ("attn", "block") else ()),
                             f"{mode} output selector is not the protocol selector")
                    if mode == "id":
                        _require(event.value is None and event.saved_shape is None,
                                 "ID mode unexpectedly retained a tensor")
                        value_hash = None
                    else:
                        _require(event.saved_shape == (1, 2, model.config.n_embd)
                                 and event.saved_device == "cpu"
                                 and _equal_float32_bits(event.value, ref[5]),
                                 f"{mode} saved value mismatch")
                        value_hash = _hash_float32(event.value)
                        current_values.append(event.value)
                    event_records.append({"metadata": event.metadata(), "value_sha256": value_hash})
                expected_bytes = 0 if mode == "id" else 18 * 2 * model.config.n_embd * 4
                _require(trace.tensor_bytes == expected_bytes and expected_bytes <= TRACE_CAP,
                         f"{mode} retained bytes mismatch")
                if mode == "direct":
                    direct_values = current_values
                elif mode == "gradient":
                    _require(all(_equal_float32_bits(a, b) for a, b in zip(direct_values, current_values)),
                             "direct and gradient values differ")
                    result["direct_gradient_equal"] = True
                result["modes"][mode] = {
                    "logits_equal_baseline": True,
                    "logit_sha256": [_hash_float32(x) for x in outputs],
                    "metadata_valid": True,
                    "values_equal_reference": None if mode == "id" else True,
                    "event_count": 18,
                    "tensor_bytes": trace.tensor_bytes,
                    "events": event_records,
                }
            _require(result["direct_gradient_equal"], "channel comparison missing")
            return result
    finally:
        reference.clear()
        direct_values = None


def _source_identity():
    root = Path(__file__).resolve().parent.parent
    # Imports execute only local Python definitions; the loader does not fetch
    # or load weights on import. Verify actual imported files, not lookalikes.
    from . import model_ownership, pretrained, probes, trace_schema, transformer_trace
    modules = {"looking_glass/__init__.py": sys.modules[__package__],
               "looking_glass/pretrained.py": pretrained,
               "looking_glass/transformer_trace.py": transformer_trace,
               "looking_glass/trace_schema.py": trace_schema,
               "looking_glass/probes.py": probes,
               "looking_glass/model_ownership.py": model_ownership}
    for relative, module in modules.items():
        _require(Path(module.__file__).resolve() == (root / relative).resolve(),
                 f"imported module differs from checked-out source: {relative}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    hashes = {}
    for relative in _SOURCE_PATHS:
        current = (root / relative).read_bytes()
        recorded = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                                  check=True, capture_output=True).stdout
        _require(current == recorded, f"source does not match committed blob: {relative}")
        hashes[relative] = hashlib.sha256(current).hexdigest()
    return {"git_commit": commit, "clean": True, "bytecode_attested": False,
            "scope": "executed project files only", "files": hashes}


def _package_version(name):
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def _gpu_identity():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    # One physical and one visible GPU provide an unambiguous ordinal mapping,
    # including CUDA_VISIBLE_DEVICES remapping; reject all other cases.
    _require(len(rows) == 1 and torch.cuda.device_count() == 1,
             "ambiguous physical-to-logical GPU mapping")
    fields = [item.strip() for item in rows[0].split(",")]
    _require(len(fields) == 3, "could not identify CUDA device and driver")
    props = torch.cuda.get_device_properties(0)
    _require(props.name == fields[0], "CUDA and nvidia-smi device names differ")
    return {"name": fields[0], "driver_version": fields[1],
            "reported_total_mib": int(fields[2]), "logical_device": "cuda:0",
            "mapping": "single physical and single visible NVIDIA GPU"}


def _visible_device_policy():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return "unset"
    if not value.strip():
        return "empty"
    if value.strip() == "-1":
        return "disabled"
    if all(part.strip().isdigit() for part in value.split(",")):
        return "ordinal-selection"
    return "identifier-selection-redacted"


def _path_fingerprint(path):
    if not path:
        return None
    normalized = os.path.normcase(os.path.abspath(path))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _environment(device):
    origins = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy"):
        spec = importlib.util.find_spec(name)
        origins[name] = _path_fingerprint(None if spec is None else spec.origin)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "python_executable_path_sha256": _path_fingerprint(sys.executable),
        "python_prefix_path_sha256": _path_fingerprint(sys.prefix),
        "python_base_prefix_path_sha256": _path_fingerprint(sys.base_prefix),
        "package_origin_path_sha256": origins,
        "machine": platform.machine(), "processor": platform.processor(),
        "torch": torch.__version__, "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"), "safetensors": _package_version("safetensors"),
        "huggingface_hub": _package_version("huggingface-hub"), "numpy": np.__version__,
        "device": device, "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device == "cuda" else None,
        "gpu": _gpu_identity() if device == "cuda" else None,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")).hexdigest(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "cpu_subnormal_multiply_bits": _cpu_subnormal_multiply_bits(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices_policy": _visible_device_policy(),
    }


def _fixed_config():
    return {"seed": 0, "tokens": list(TOKENS),
            "token_sha256": hashlib.sha256(np.array(TOKENS, dtype="<i8").tobytes()).hexdigest(),
            "token_hash_format": "contiguous C-order little-endian int64",
            "input_shape": [1, 5], "attention_mask": [1, 1, 1, 1, 1],
            "tokenization": "none; fixed token IDs", "model_mode": "eval",
            "attention_implementation": "eager", "dtype": "float32", "use_cache": False,
            "grad_mode": "inference_mode", "forwards_per_cell": 2,
            "layers": list(LAYERS), "target_order": list(TARGETS),
            "batch_positions": [0], "token_positions": list(POSITIONS),
            "max_events": 18, "max_trace_bytes": TRACE_CAP, "capture_device": "cpu",
            "tensor_hash_format": "contiguous C-order little-endian float32"}


def run_smoke(device, cache_dir, *, lease_id=None, _diagnostics=None, _on_prepared=None):
    """Run v1 offline checks; CUDA requires an externally coordinated lease."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and (not isinstance(lease_id, str) or not lease_id.strip()):
        raise ValueError("CUDA requires an externally coordinated exclusive lease ID")
    if device == "cpu" and lease_id is not None:
        raise ValueError("a CUDA lease ID cannot label a CPU run")
    diagnostics = _diagnostics if _diagnostics is not None else {}
    diagnostics.update({"phase": "source_validation", "device": device,
                        "lease_id": lease_id, "protocol_version": PROTOCOL_VERSION})
    # Verify every project file used by the protocol against the commit first.
    source = _source_identity()
    from .pretrained import ASSETS, MODEL_ID, REVISION, load_distilgpt2
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    study_id = str(uuid.uuid4())
    observations = {"lease_id": lease_id, "study_id": study_id, "memory": []}
    diagnostics.update({"source_hashes": source, "observations": observations})
    config = _fixed_config()
    if _on_prepared is not None:
        _on_prepared({"protocol_version": PROTOCOL_VERSION, "config": config,
                      "source_hashes": source, "device": device, "study_id": study_id,
                      "expected_model_id": MODEL_ID, "expected_model_revision": REVISION,
                      "expected_asset_sha256": {name: digest for name, (_, digest) in ASSETS.items()}})
    model = bundle = ids = None
    reserve = None

    def memory(label, *, enforce=True):
        free, total = torch.cuda.mem_get_info(0)
        observations["memory"].append({
            "label": label, "device_free_bytes": free, "device_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(0),
            "torch_reserved_bytes": torch.cuda.memory_reserved(0)})
        if enforce and free < reserve:
            raise SmokeFailure("device free memory fell below the required reserve")
        return free, total

    try:
        if device == "cuda":
            diagnostics["phase"] = "admission"
            _require(torch.cuda.is_available(), "CUDA is not available")
            torch.cuda.init()
            free, total = memory("admission_initial", enforce=False)
            reserve = max(1024 ** 3, (total * 15 + 99) // 100)
            observations["reserve_bytes"] = reserve
            _require(free >= reserve + 1024 ** 3, "insufficient free memory above reserve")
            allocator_budget = min(1536 * 1024 ** 2, free - reserve)
            torch.cuda.set_per_process_memory_fraction(allocator_budget / total, 0)
            observations["torch_allocator_limit_bytes"] = allocator_budget
            memory("admission")
        diagnostics["phase"] = "checkpoint_load"
        bundle = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
        model = bundle.model
        _require(model.config.n_layer == 6 and model.config.n_embd == 768
                 and model.config.vocab_size == 50257, "unexpected pretrained model configuration")
        diagnostics["model_load_metadata"] = bundle.metadata
        _require(all(not module.training for module in model.modules())
                 and model.config._attn_implementation == "eager"
                 and model.config.use_cache is False,
                 "model is not in the prescribed eval/eager/no-cache configuration")
        model.to(device)
        _require(all(p.dtype == torch.float32 and p.device.type == device
                     for p in model.parameters()), "model parameter dtype/device mismatch")
        ids = torch.tensor([TOKENS], dtype=torch.long, device=device)
        if device == "cuda":
            torch.cuda.synchronize(0)
            memory("after_model_transfer")
        environment = _environment(device)
        diagnostics["phase"] = "model_checks"
        def finish_forward():
            if device == "cuda":
                torch.cuda.synchronize(0)
                memory("after_forward")
        checks = check_model(model, ids, device=device, after_forward=finish_forward,
                             study_id=study_id)
        for mode in ("direct", "gradient"):
            _require(checks["modes"][mode]["tensor_bytes"] == 110592,
                     "unexpected pretrained trace size")
        model_metadata = dict(bundle.metadata)
        model_metadata["load_device"] = model_metadata.pop("device")
        model_metadata["execution_device"] = str(next(model.parameters()).device)
        diagnostics["phase"] = "completed"
        return {"protocol_version": PROTOCOL_VERSION, "config": config,
                "model": model_metadata, "source_hashes": source,
                "environment": environment, "checks": checks, "observations": observations}
    finally:
        primary = sys.exc_info()[1]
        cleanup_errors = []
        cuda_initialized = False
        if device == "cuda":
            try:
                cuda_initialized = torch.cuda.is_initialized()
            except BaseException as error:
                cleanup_errors.append(error)
        if cuda_initialized:
            try:
                torch.cuda.synchronize(0)
            except BaseException as error:
                cleanup_errors.append(error)
            if model is not None:
                try:
                    model.to("cpu")
                except BaseException as error:
                    cleanup_errors.append(error)
        ids = model = bundle = None
        try:
            gc.collect()
        except BaseException as error:
            cleanup_errors.append(error)
        if cuda_initialized:
            try:
                torch.cuda.empty_cache()
                if reserve is not None:
                    memory("after_cleanup", enforce=False)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            diagnostics["cleanup_errors"] = [f"{type(e).__name__}: {e}" for e in cleanup_errors]
            if primary is not None:
                _note_cleanup(primary, "Smoke cleanup also failed: " + "; ".join(diagnostics["cleanup_errors"]))
            else:
                raise SmokeFailure("smoke cleanup failed") from cleanup_errors[0]


def _write_json(path, value):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking a completed same-directory file publishes atomically and
        # cannot replace a pre-existing terminal artifact.
        os.link(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _redact_failure(value, cache_dir):
    """Remove local paths and GPU UUIDs from shareable failure evidence."""
    if isinstance(value, dict):
        return {key: _redact_failure(item, cache_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_failure(item, cache_dir) for item in value]
    if not isinstance(value, str):
        return value
    text = re.sub(r"GPU-[0-9a-fA-F-]+", "<gpu-uuid>", value, flags=re.IGNORECASE)
    paths = [(str(Path(cache_dir).resolve()), "<cache>"),
             (str(Path(__file__).resolve().parent.parent), "<checkout>"),
             (str(Path.home().resolve()), "<home>")]
    for path, label in sorted(paths, key=lambda item: len(item[0]), reverse=True):
        for form in sorted({path, path.replace('\\', '/'), path.replace('\\', '\\\\')},
                           key=len, reverse=True):
            text = re.sub(re.escape(form), lambda _match: label, text, flags=re.IGNORECASE)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cache-dir", required=True, help="explicit offline Hugging Face cache")
    parser.add_argument("--output", required=True, help="new, unique run directory")
    parser.add_argument("--lease-id", help="externally coordinated CUDA-exclusive window ID")
    args = parser.parse_args(argv)
    destination = Path(args.output)
    destination.mkdir(parents=False, exist_ok=False)
    diagnostics = {}
    try:
        _write_json(destination / "started.json", {
            "protocol_version": PROTOCOL_VERSION, "device": args.device,
            "started_utc": datetime.now(timezone.utc).isoformat(), "lease_id": args.lease_id,
        })
        result = run_smoke(args.device, args.cache_dir, lease_id=args.lease_id,
                           _diagnostics=diagnostics,
                           _on_prepared=lambda manifest: _write_json(destination / "manifest.json", manifest))
        _write_json(destination / "result.json", result)
    except BaseException as error:
        try:
            failure = {
                "protocol_version": PROTOCOL_VERSION, "device": args.device,
                "status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "diagnostics": diagnostics}
            _write_json(destination / "failure.json", _redact_failure(failure, args.cache_dir))
        except BaseException as record_error:
            _note_cleanup(error, f"Failure record could not be written: {record_error!r}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
