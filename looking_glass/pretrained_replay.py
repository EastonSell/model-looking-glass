"""Compare a fresh pretrained-smoke execution with a saved record.

Exact hashes do not establish cross-device numerical agreement or causal
mechanisms. Performance and memory observations are deliberately excluded.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from .trace_schema import TraceEvent


class ReplayError(ValueError):
    """A reference or replay record is incomplete or invalid."""


_IDENTITY_FIELDS = ("protocol_version", "config", "model", "source_hashes", "environment")
_ASSET_HASHES = {
    "config.json": "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a",
    "generation_config.json": "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
}
_REQUIRED_SOURCES = {"docs/PRETRAINED_SMOKE_V1.md", "looking_glass/__init__.py",
                     "looking_glass/pretrained.py", "looking_glass/pretrained_smoke.py",
                     "looking_glass/probes.py", "looking_glass/model_ownership.py", "looking_glass/trace_schema.py",
                     "looking_glass/transformer_trace.py"}
_TOKENS = (464, 6766, 318, 4171, 13)
_CONFIG = {"tokens": list(_TOKENS), "seed": 0, "dtype": "float32",
           "token_sha256": hashlib.sha256(struct.pack("<5q", *_TOKENS)).hexdigest(),
           "token_hash_format": "contiguous C-order little-endian int64",
           "input_shape": [1, 5], "attention_mask": [1] * 5,
           "tokenization": "none; fixed token IDs", "model_mode": "eval",
           "attention_implementation": "eager", "use_cache": False,
           "grad_mode": "inference_mode", "forwards_per_cell": 2,
           "layers": [0, 2, 5], "target_order": ["attn", "mlp", "block"],
           "batch_positions": [0], "token_positions": [0, 1], "max_events": 18,
           "max_trace_bytes": 131072, "capture_device": "cpu",
           "tensor_hash_format": "contiguous C-order little-endian float32"}


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayError("record contains non-JSON or nonfinite values") from exc


def _require(condition, message):
    if not condition:
        raise ReplayError(message)


def _digest(value, length=64):
    return (isinstance(value, str) and len(value) == length
            and all(character in "0123456789abcdef" for character in value))


def _validate_checks(checks, source_device):
    _require(checks.get("passed") is True, "smoke checks did not pass")
    baseline = checks.get("baseline_logit_sha256")
    reference_logits = checks.get("reference_logit_sha256")
    values = checks.get("reference_value_sha256")
    _require(isinstance(baseline, list) and len(baseline) == 2 and all(map(_digest, baseline)),
             "expected two baseline logit hashes")
    _require(baseline[0] == baseline[1] and checks.get("baseline_repeat_equal") is True and
             _canonical(checks.get("baseline_logit_shape")) == _canonical([1, 5, 50257]) and
             checks.get("baseline_logit_dtype") == "torch.float32",
             "repeated baseline logits or their shape/dtype disagree with v1")
    _require(reference_logits == baseline and checks.get("reference_logits_equal") is True,
             "reference logits disagree with the baseline")
    _require(isinstance(values, list) and len(values) == 18 and all(map(_digest, values)),
             "expected 18 reference activation hashes")
    _require(checks.get("direct_gradient_equal") is True, "capture channels are not equal")
    modes = checks.get("modes")
    _require(isinstance(modes, dict) and set(modes) == {"direct", "gradient", "id"},
             "missing or unexpected smoke capture modes")
    run_ids = set()
    for mode, record in modes.items():
        _require(isinstance(record, dict), "invalid smoke mode record")
        _require(record.get("logits_equal_baseline") is True and
                 record.get("logit_sha256") == baseline, "mode logits disagree with baseline")
        _require(record.get("metadata_valid") is True and
                 record.get("values_equal_reference") is (None if mode == "id" else True),
                 "mode metadata or reference comparison did not pass")
        _require(type(record.get("event_count")) is int and record["event_count"] == 18 and
                 type(record.get("tensor_bytes")) is int and
                 record["tensor_bytes"] == (0 if mode == "id" else 110592),
                 "mode event or retained-byte count differs from v1")
        events = record.get("events")
        _require(isinstance(events, list) and len(events) == 18, "mode lacks complete event coverage")
        run_id = None
        for index, event in enumerate(events):
            _require(isinstance(event, dict) and isinstance(event.get("metadata"), dict),
                     "invalid event metadata")
            metadata = event["metadata"]
            try:
                TraceEvent.from_dict(metadata)
            except (TypeError, ValueError, KeyError) as exc:
                raise ReplayError("invalid trace schema metadata") from exc
            layer = (0, 2, 5)[(index % 9) // 3]
            kind = ("attn", "mlp", "block")[index % 3]
            path = f"transformer.h.{layer}" + ("" if kind == "block" else f".{kind}")
            expected = {"sequence": index, "forward_index": index // 9, "invocation": 0,
                        "module_path": path, "module_kind": kind, "layer_index": layer,
                        "output_path": [] if kind == "mlp" else [0],
                        "shape": [1, 5, 768], "dtype": "torch.float32", "device": source_device,
                        "capture_mode": mode, "batch_positions": [0], "token_positions": [0, 1],
                        "batch_axis": 0, "token_axis": 1,
                        "generation": {"stage": "unknown", "generation_step": None,
                                       "cache_position": None, "cache_length_before": None,
                                       "cache_length_after": None, "use_cache": None},
                        "saved_shape": None if mode == "id" else [1, 2, 768],
                        "saved_dtype": None if mode == "id" else "torch.float32",
                        "saved_device": None if mode == "id" else "cpu"}
            _require(all(key in metadata and _canonical(metadata[key]) == _canonical(value)
                         for key, value in expected.items()), "event metadata differs from v1")
            identity = metadata.get("identity")
            _require(isinstance(identity, dict) and isinstance(identity.get("run_id"), str)
                     and identity["run_id"].strip(), "event has no run identity")
            if run_id is None:
                run_id = identity["run_id"]
            _require(identity["run_id"] == run_id, "one mode contains multiple run identities")
            _require(event.get("value_sha256") == (None if mode == "id" else values[index]),
                     "event value hash disagrees with its reference")
        _require(run_id not in run_ids, "capture modes must have distinct run identities")
        run_ids.add(run_id)


def _normalized_checks(checks):
    # Each independent study must have distinct session IDs. Preserve the saved
    # records verbatim and normalize only this per-session identifier for replay.
    normalized = copy.deepcopy(checks)
    for mode in normalized["modes"].values():
        for event in mode["events"]:
            event["metadata"]["identity"]["run_id"] = "<session>"
    return normalized


def _validate(report, label):
    if not isinstance(report, dict):
        raise ReplayError(f"{label} must be a JSON object")
    if report.get("protocol_version") != "pretrained-trace-smoke-v1":
        raise ReplayError(f"{label} is not pretrained-trace-smoke-v1")
    for field in ("config", "model", "source_hashes", "environment", "checks"):
        if not isinstance(report.get(field), dict) or not report[field]:
            raise ReplayError(f"{label} has no {field}")
    config, model, source = report["config"], report["model"], report["source_hashes"]
    _require(all(key in config and _canonical(config[key]) == _canonical(value)
                 for key, value in _CONFIG.items()), f"{label} does not have the fixed v1 configuration")
    _require(model.get("model_id") == "distilbert/distilgpt2" and
             model.get("model_revision") == "2290a62682d06624634c1f46a6ad5be0f47f38aa" and
             model.get("tokenizer_id") == "distilbert/distilgpt2" and
             model.get("tokenizer_revision") == "2290a62682d06624634c1f46a6ad5be0f47f38aa" and
             model.get("weights_sha256") == _ASSET_HASHES["model.safetensors"] and
             model.get("asset_sha256") == _ASSET_HASHES, f"{label} has the wrong pinned model assets")
    files = source.get("files")
    _require(_digest(source.get("git_commit"), 40) and source.get("clean") is True and
             source.get("bytecode_attested") is False and
             source.get("scope") == "executed project files only" and
             isinstance(files, dict) and _REQUIRED_SOURCES <= set(files) and
             all(isinstance(key, str) and _digest(value) for key, value in files.items()),
             f"{label} lacks committed source and protocol hashes")
    environment = report["environment"]
    device = environment.get("device")
    _require(device in ("cpu", "cuda"), f"{label} has an unsupported device identity")
    model_details = {"model_class": "GPT2LMHeadModel", "load_device": "cpu",
                     "execution_device": "cpu" if device == "cpu" else "cuda:0",
                     "dtype": "float32", "attention_implementation": "eager", "use_cache": False,
                     "parameter_count": 81912576, "torch_version": environment.get("torch"),
                     "transformers_version": environment.get("transformers")}
    _require(all(key in model and _canonical(model[key]) == _canonical(value)
                 for key, value in model_details.items()) and
             isinstance(model.get("tokenizer_class"), str) and model["tokenizer_class"],
             f"{label} has inconsistent model execution metadata")
    versions = ("python", "platform", "machine", "torch", "transformers", "tokenizers",
                "safetensors", "huggingface_hub", "numpy")
    origins = environment.get("package_origin_path_sha256")
    _require(all(isinstance(environment.get(key), str) and environment[key] for key in versions)
             and "processor" in environment and "cuda_build" in environment
             and "cudnn_version" in environment and "gpu" in environment
             and "cublas_workspace_config" in environment
             and "cuda_visible_devices" not in environment
             and environment.get("cuda_visible_devices_policy") in
                 ("unset", "empty", "disabled", "ordinal-selection", "identifier-selection-redacted")
             and all(_digest(environment.get(key)) for key in
                     ("python_executable_path_sha256", "python_prefix_path_sha256", "python_base_prefix_path_sha256"))
             and isinstance(origins, dict)
             and set(origins) == {"torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy"}
             and all(value is None or _digest(value) for value in origins.values())
             and _digest(environment.get("torch_build_config_sha256"))
             and isinstance(environment.get("torch_cpu_capability"), str)
             and environment["torch_cpu_capability"]
             and type(environment.get("cpu_subnormal_multiply_bits")) is int
             and environment["cpu_subnormal_multiply_bits"] in (0, 1)
             and all(key in environment and (environment[key] is None or isinstance(environment[key], str))
                     for key in ("omp_num_threads", "mkl_num_threads"))
             and type(environment.get("torch_intraop_threads")) is int
             and environment["torch_intraop_threads"] == 1
             and type(environment.get("torch_interop_threads")) is int
             and environment["torch_interop_threads"] > 0
             and type(environment.get("deterministic_algorithms")) is bool
             and type(environment.get("cudnn_deterministic")) is bool
             and type(environment.get("cudnn_benchmark")) is bool
             and environment.get("cuda_matmul_allow_tf32") is False
             and environment.get("cudnn_allow_tf32") is False,
             f"{label} lacks stable runtime and backend identity")
    if device == "cuda":
        gpu = environment["gpu"]
        _require(isinstance(gpu, dict) and
                 set(gpu) == {"name", "driver_version", "reported_total_mib", "logical_device", "mapping"} and
                 all(isinstance(gpu.get(key), str) and gpu[key].strip()
                     for key in ("name", "driver_version")) and
                 type(gpu.get("reported_total_mib")) is int and gpu["reported_total_mib"] > 0 and
                 gpu.get("logical_device") == "cuda:0" and
                 gpu.get("mapping") == "single physical and single visible NVIDIA GPU",
                 f"{label} lacks CUDA device identity")
    else:
        _require(environment["gpu"] is None and environment["cudnn_version"] is None,
                 f"{label} has inconsistent CPU device identity")
    _validate_checks(report["checks"], "cpu" if device == "cpu" else "cuda:0")
    observations = report.get("observations")
    study_id = observations.get("study_id") if isinstance(observations, dict) else None
    _require(isinstance(study_id, str) and study_id.strip(), f"{label} lacks a study identity")
    _require(all(record["events"][0]["metadata"]["identity"]["run_id"] == f"{study_id}-{mode}"
                 for mode, record in report["checks"]["modes"].items()),
             f"{label} study and session identities disagree")
    for field in (*_IDENTITY_FIELDS, "checks"):
        _canonical(report[field])


def compare_reports(reference, replay):
    """Compare provenance first, then exact saved checks, excluding observations.

    ``incompatible`` means exact replay is not a fair comparison. ``different``
    means compatible records disagree in their hashes or event checks; it does
    not diagnose the cause. A malformed or failed record raises ReplayError.
    """
    _validate(reference, "reference")
    _validate(replay, "replay")
    ids = lambda report: {record["events"][0]["metadata"]["identity"]["run_id"]
                          for record in report["checks"]["modes"].values()}
    _require(ids(reference).isdisjoint(ids(replay)), "reference and replay reuse session identities")
    mismatches = [field for field in _IDENTITY_FIELDS
                  if _canonical(reference[field]) != _canonical(replay[field])]
    if mismatches:
        return {"status": "incompatible", "differences": mismatches}
    if _canonical(_normalized_checks(reference["checks"])) != _canonical(_normalized_checks(replay["checks"])):
        return {"status": "different", "differences": ["checks"]}
    return {"status": "reproduced", "differences": []}


def _write_json(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _replay_source_identity():
    """Identify this checker by committed bytes, not just a caller's claim."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL).decode("ascii").strip()
        committed = subprocess.check_output(["git", "show", "HEAD:looking_glass/pretrained_replay.py"],
                                            cwd=root, stderr=subprocess.DEVNULL)
        current = Path(__file__).read_bytes()
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise ReplayError("cannot verify the replay checker against a Git commit") from exc
    _require(_digest(commit, 40) and current == committed,
             "replay checker does not match its committed source bytes")
    return {"git_commit": commit, "sha256": hashlib.sha256(current).hexdigest()}


def replay_to_directory(reference_path, output_dir, *, cache_dir, device="cpu", lease_id=None,
                        run_smoke=None):
    """Execute a fresh smoke and save its comparison in a new directory.

    A CUDA lease ID is an operator attestation, not a cross-process lock.
    Callers must separately coordinate the GPU window. ``run_smoke`` permits
    CPU-only tests; normal operation imports the protocol's real runner.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and (not isinstance(lease_id, str) or not lease_id.strip()):
        raise ValueError("CUDA replay requires a coordinated lease ID")
    if device == "cpu" and lease_id is not None:
        raise ValueError("a lease ID is only applicable to CUDA")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    reference_sha = reference_commit = replay_source = None
    diagnostics = {}
    try:
        _write_json(output_dir / "started.json", {"protocol_version": "pretrained-trace-smoke-v1",
                    "state": "started", "device": device, "lease_id": lease_id})
        reference_raw = Path(reference_path).read_bytes()
        reference_sha = hashlib.sha256(reference_raw).hexdigest()
        reference = json.loads(reference_raw)
        _validate(reference, "reference")
        reference_commit = reference["source_hashes"]["git_commit"]
        replay_source = _replay_source_identity()
        _write_json(output_dir / "identity.json", {"reference_sha256": reference_sha,
                    "reference_source_commit": reference_commit, "replay_source": replay_source,
                    "device": device, "lease_id": lease_id})
        if run_smoke is None:
            from .pretrained_smoke import run_smoke
        prepared = None

        def record_prepared(manifest):
            nonlocal prepared
            source = manifest.get("source_hashes") if isinstance(manifest, dict) else None
            _require(isinstance(manifest, dict) and manifest.get("protocol_version") == "pretrained-trace-smoke-v1"
                     and manifest.get("device") == device and isinstance(source, dict)
                     and source.get("git_commit") == replay_source["git_commit"]
                     and _canonical(manifest.get("config")) == _canonical(reference["config"])
                     and manifest.get("expected_model_id") == "distilbert/distilgpt2"
                     and manifest.get("expected_model_revision") == "2290a62682d06624634c1f46a6ad5be0f47f38aa"
                     and manifest.get("expected_asset_sha256") == _ASSET_HASHES,
                     "runner's prepared manifest does not match the frozen protocol")
            _write_json(output_dir / "prepared.json", manifest)
            prepared = copy.deepcopy(manifest)

        replay = run_smoke(device=device, cache_dir=cache_dir, lease_id=lease_id,
                           _diagnostics=diagnostics, _on_prepared=record_prepared)
        _require(prepared is not None, "runner did not publish its prepared manifest")
        replay_identity = replay.get("source_hashes") if isinstance(replay, dict) else None
        _require(isinstance(replay_identity, dict) and
                 replay_identity.get("git_commit") == replay_source["git_commit"],
                 "replay runner and checker do not share a source commit")
        observations = replay.get("observations") if isinstance(replay, dict) else None
        _require(isinstance(observations, dict) and
                 _canonical(prepared["source_hashes"]) == _canonical(replay_identity) and
                 prepared.get("study_id") == observations.get("study_id"),
                 "prepared source or study identity disagrees with the replay")
        result = {"reference_sha256": reference_sha,
                  "replay_source": replay_source, "comparison": compare_reports(reference, replay), "replay": replay,
                  "limitations": ["exact hashes do not test cross-device tolerances",
                                  "a replay does not prove source provenance or causal explanation"]}
        _write_json(output_dir / "results.json", result)
        return result
    except BaseException as exc:
        # Publication can succeed before cleanup fails. Never publish a second,
        # contradictory terminal in that uncertain-outcome case.
        if (output_dir / "results.json").exists():
            raise
        try:
            _write_json(output_dir / "failure.json", {
                "status": "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                "error_type": type(exc).__name__, "message": str(exc), "device": device,
                "lease_id": lease_id, "reference_sha256": reference_sha,
                "reference_source_commit": reference_commit, "replay_source": replay_source,
                "runner_diagnostics": diagnostics})
        except Exception as recording_error:
            raise exc from recording_error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="saved smoke result.json")
    parser.add_argument("--output", type=Path, required=True, help="new replay output directory")
    parser.add_argument("--cache-dir", type=Path, required=True, help="offline model cache")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lease-id", help="attested coordinated CUDA window")
    args = parser.parse_args(argv)
    try:
        result = replay_to_directory(args.reference, args.output, cache_dir=args.cache_dir,
                                     device=args.device, lease_id=args.lease_id)
    except Exception as exc:
        print(f"replay failed: {exc}", file=sys.stderr)
        return 1
    comparison = result["comparison"]
    print(json.dumps(comparison, sort_keys=True))
    return 0 if comparison["status"] == "reproduced" else 1


if __name__ == "__main__":
    raise SystemExit(main())
