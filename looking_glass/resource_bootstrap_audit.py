"""Read-only CPU audit helpers for private resource-bootstrap-v1 evidence.

Hash consistency and recomputed arithmetic do not authenticate a producer,
prove device isolation, or make a finite measurement an upper bound.
"""

import argparse
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import math
import ntpath
import os
import posixpath
import re
import stat
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .memory_budget import CudaMemorySnapshot
from .resource_profile import load_profile


class BootstrapAuditError(ValueError):
    """A private bootstrap artifact is malformed or internally inconsistent."""


_INPUT_WIDTHS = {
    "torch.bool": 1, "torch.int32": 4, "torch.int64": 8,
    "torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4, "torch.float64": 8,
}
_MODEL_FIELDS = (
    "checkpoint_sha256", "checkpoint_revision", "model_config_sha256",
    "source_sha256", "runtime_settings_sha256", "torch_version",
    "transformers_version", "cuda_runtime_version", "driver_version",
    "python_version", "device", "device_uuid", "device_name",
    "compute_capability", "device_total_bytes", "model_dtype",
    "compute_dtype", "attention_backend", "model_mode", "grad_mode", "autocast",
    "use_cache", "compiled",
)
_GIB = 1024 ** 3
_MIB = 1024 ** 2
_FILES = {
    "started.json", "owner.json", "source.json", "runtime.json", "assets.json", "load-case.json",
    "forward-case.json", "warmup.json", "capture.json", "checks.json", "admissions.jsonl", "observations.jsonl",
    "load.profile.json", "load.evidence.json", "cold.profile.json", "cold.evidence.json",
    "warm.profile.json", "warm.evidence.json",
}
_REQUESTED = {
    "load_commitment_bytes": 2 * _GIB, "job_commitment_bytes": 512 * _MIB,
    "reserve_min_bytes": _GIB, "reserve_percent": 15, "allocator_fraction": "0.20",
    "input_limit_bytes": 64 * 1024, "output_tensor_limit_bytes": 512 * 1024,
    "response_wire_limit_bytes": 4 * _MIB, "max_gpu_trace_bytes": 0,
    "max_host_trace_bytes": 0,
}
_STAGES = (("load", 17), ("cold", 17), ("warm", 29), ("warm", 43), ("warm", 71))
_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
_ASSETS = {
    "config.json": (762, "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a"),
    "generation_config.json": (124, "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805"),
    "merges.txt": (456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "model.safetensors": (352824413, "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"),
    "tokenizer.json": (1355256, "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"),
    "tokenizer_config.json": (26, "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f"),
    "vocab.json": (1042301, "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
}
_FORWARD_CASE = {"schema": "resource-bootstrap-case-v1", "input_ids": [15496, 995],
                 "shape": [1, 2], "dtype": "torch.int64"}
_LOAD_CASE = {"schema": "resource-bootstrap-load-case-v1", "operation": "model_load",
              "checkpoint_sha256": _ASSETS["model.safetensors"][1], "config_sha256": _ASSETS["config.json"][1]}
_WARMUP = {"schema": "resource-bootstrap-warmup-v1", "procedure": "one_completed_cold_forward_same_case",
           "cold_seed": 17, "warm_seeds": [29, 43, 71]}
_CAPTURE = {"schema": "resource-bootstrap-capture-v1", "mode": "none",
            "max_gpu_trace_bytes": 0, "max_host_trace_bytes": 0}
_RUNTIME_FIELDS = {
    "schema", "device", "device_uuid", "device_name", "device_total_bytes", "compute_capability",
    "python_version", "torch_version", "transformers_version", "cuda_runtime_version", "driver_version",
    "cudnn_version", "cuda_visible_devices", "cuda_device_order", "allocator_backend", "allocator_fraction",
    "allocator_effective_limit_bytes", "model_dtype", "compute_dtype", "attention_backend", "model_mode",
    "grad_mode", "autocast", "use_cache", "compiled", "float32_matmul_precision", "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32", "cudnn_enabled", "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms",
    "torch_num_threads", "torch_num_interop_threads", "input_limit_bytes", "output_tensor_limit_bytes",
    "response_wire_limit_bytes", "execution_path", "python_import_paths", "environment",
}
_NO_CANDIDATE = object()


def _require(condition, message):
    if not condition:
        raise BootstrapAuditError(message)


def _digest(value, size=64):
    return isinstance(value, str) and bool(re.fullmatch(f"[0-9a-f]{{{size}}}", value))


def _json_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise BootstrapAuditError("duplicate JSON key")
        result[key] = value
    return result


def _integer_json(value, depth=0):
    if depth > 64:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is list:
        return all(_integer_json(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _integer_json(item, depth + 1) for key, item in value.items())
    return False


def _decode_canonical(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_json_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(BootstrapAuditError("nonfinite JSON")))
        _require(_integer_json(value) and _canonical(value) == raw, "artifact is not canonical JSON")
    except BootstrapAuditError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError, TypeError):
        raise BootstrapAuditError("invalid JSON artifact") from None
    return value


def _unsafe_link(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        raise BootstrapAuditError("cannot inspect artifact path") from None
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _read(root, name):
    path = root / name
    try:
        _require(not _unsafe_link(path) and path.is_file(), "required artifact is unavailable or unsafe")
        _require(path.stat().st_size <= 64 * _MIB, "artifact exceeds the audit size limit")
        return path.read_bytes()
    except OSError:
        raise BootstrapAuditError("cannot read artifact") from None


def _utc(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
    except (ValueError, OverflowError):
        return False


def _inventory(root, terminal):
    records = terminal.get("inventory")
    _require(isinstance(records, list), "terminal inventory is missing")
    names = []
    for record in records:
        _require(isinstance(record, dict) and set(record) == {"path", "size", "sha256"},
                 "invalid inventory entry")
        name = record["path"]
        _require(isinstance(name, str) and name and name not in (".", "..", "terminal.json")
                 and "/" not in name and "\\" not in name and ":" not in name and "\x00" not in name,
                 "inventory path must be a flat filename")
        _require(type(record["size"]) is int and record["size"] >= 0 and _digest(record["sha256"]),
                 "invalid inventory size or digest")
        names.append(name)
    _require(names == sorted(set(names)), "terminal inventory is unsorted or contains duplicates")
    try:
        actual = list(root.iterdir())
    except OSError:
        raise BootstrapAuditError("cannot inspect artifact directory") from None
    _require(all(not _unsafe_link(item) and item.is_file() for item in actual),
             "artifact directory contains a link or non-file")
    actual_names = sorted(item.name for item in actual if item.name != "terminal.json")
    _require(actual_names == names, "inventory does not cover the actual directory")
    for record in records:
        path = root / record["path"]
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            size = path.stat().st_size
        except OSError:
            raise BootstrapAuditError("cannot verify inventory file") from None
        _require(size == record["size"] and digest.hexdigest() == record["sha256"],
                 "inventory bytes do not match the terminal")


def _observations(root, *, require_complete):
    path = root / "observations.jsonl"
    _require(not _unsafe_link(path), "observation journal is a link")
    if not path.exists():
        _require(not require_complete, "success has no observation journal")
        return [], False
    raw = _read(root, "observations.jsonl")
    partial = bool(raw) and not raw.endswith(b"\n")
    lines = raw.split(b"\n")
    lines.pop()
    _require(not require_complete or not partial, "success has a truncated observation line")
    _require(len(lines) <= len(_STAGES), "observation journal has extra lines")
    observations = []
    for index, line in enumerate(lines):
        row = _decode_canonical(line)
        _require(isinstance(row, dict) and set(row) == {"schema", "sequence", "stage", "key", "sample"},
                 "observation has unexpected fields")
        stage, seed = _STAGES[index]
        _require(row["schema"] == "resource-bootstrap-observation-v1"
                 and type(row["sequence"]) is int and row["sequence"] == index + 1
                 and row["stage"] == stage and isinstance(row["key"], dict)
                 and isinstance(row["sample"], dict) and type(row["sample"].get("seed")) is int
                 and row["sample"]["seed"] == seed,
                 "observation sequence, stage or seed differs from the protocol")
        observations.append(row)
    _require(not require_complete or len(observations) == len(_STAGES), "success lacks complete observations")
    return observations, partial


def _admissions(root, *, require_complete):
    path = root / "admissions.jsonl"
    _require(not _unsafe_link(path), "admission journal is a link")
    if not path.exists():
        _require(not require_complete, "success has no admission journal")
        return [], False
    raw = _read(root, "admissions.jsonl")
    partial = bool(raw) and not raw.endswith(b"\n")
    lines = raw.split(b"\n")
    lines.pop()
    _require(not require_complete or not partial, "success has a truncated admission line")
    _require(len(lines) <= len(_STAGES), "admission journal has extra lines")
    required = {"schema", "sequence", "stage", "telemetry", "reserve_bytes", "committed_incremental_bytes",
                "load_commitment_bytes", "job_commitment_bytes", "owner_headroom_check",
                "collector_post_sync_check_attested", "job_lease_released_after"}
    records = []
    for index, line in enumerate(lines):
        row = _decode_canonical(line)
        stage, _ = _STAGES[index]
        _require(isinstance(row, dict) and set(row) == required
                 and row["schema"] == "resource-bootstrap-admission-v1"
                 and type(row["sequence"]) is int and row["sequence"] == index + 1 and row["stage"] == stage,
                 "admission sequence or fields differ from protocol")
        telemetry = row["telemetry"]
        _require(isinstance(telemetry, dict) and set(telemetry) ==
                 {"device_free_bytes", "device_total_bytes", "process_allocated_bytes", "process_reserved_bytes"},
                 "invalid admission telemetry")
        try:
            snapshot = CudaMemorySnapshot(**telemetry)
        except (ValueError, TypeError):
            raise BootstrapAuditError("invalid admission telemetry") from None
        total = snapshot.device_total_bytes
        reserve = max(_GIB, (15 * total + 99) // 100)
        job = 0 if stage == "load" else 512 * _MIB
        commitment = 2 * _GIB + job
        expected = {"reserve_bytes": reserve, "committed_incremental_bytes": commitment,
                    "load_commitment_bytes": 2 * _GIB, "job_commitment_bytes": job,
                    "owner_headroom_check": True, "collector_post_sync_check_attested": True,
                    "job_lease_released_after": None if stage == "load" else True}
        _require(7 * _GIB <= total <= 9 * _GIB
                 and all(_canonical(row[name]) == _canonical(value) for name, value in expected.items())
                 and snapshot.process_reserved_bytes + commitment + reserve <= total
                 and commitment + reserve <= snapshot.device_free_bytes,
                 "admission telemetry or headroom is inconsistent")
        records.append(row)
    _require(not require_complete or len(records) == len(_STAGES), "success lacks complete admissions")
    return records, partial


def _check_execution_checks(checks, runtime, owner):
    fields = {"schema", "device_total_bytes", "allocator_requested_fraction", "allocator_effective_limit_bytes",
              "allocator_readback", "model_aliases_disposed", "device_synchronized_after_disposal",
              "load_lease_disposition", "owner", "import_policy", "forwards"}
    _require(isinstance(checks, dict) and set(checks) == fields and checks["schema"] == "resource-bootstrap-checks-v1",
             "execution checks are incomplete")
    total = runtime["device_total_bytes"]
    _require(type(checks["device_total_bytes"]) is int and checks["device_total_bytes"] == total
             and checks["allocator_requested_fraction"] == "0.20"
             and type(checks["allocator_effective_limit_bytes"]) is int
             and checks["allocator_effective_limit_bytes"] == int(0.20 * total)
             and checks["model_aliases_disposed"] is True and checks["device_synchronized_after_disposal"] is True
             and checks["load_lease_disposition"] in ("released_after_verified_disposal", "held_until_exit")
             and isinstance(checks["allocator_readback"], str), "allocator or teardown checks differ")
    try:
        readback = float(checks["allocator_readback"])
    except ValueError:
        raise BootstrapAuditError("invalid allocator readback") from None
    _require(math.isfinite(readback) and repr(readback) == checks["allocator_readback"]
             and readback == checks["allocator_effective_limit_bytes"] / total,
             "allocator readback does not match the effective limit")
    _require(owner is not None and _canonical(checks["owner"]) ==
             _canonical({"run_id": owner["supervisor_run_id"], "pid": owner["pid"]}),
             "execution checks identify a different owner")
    policy = checks["import_policy"]
    attested = {"parent_created_empty": True, "runtime_prefix_matches": True,
                "cache_empty_after_imports": True, "dont_write_bytecode": True,
                "hash_seed_set_before_exec": True, "module_origins_verified": True,
                "sys_path_verified": True, "safe_path_enabled": True,
                "site_import_disabled": True, "user_site_enabled": False}
    _require(isinstance(policy, dict) and set(policy) == set(attested) | {"private_cache_prefix"}
             and isinstance(policy["private_cache_prefix"], str) and "\x00" not in policy["private_cache_prefix"]
             and (ntpath.isabs(policy["private_cache_prefix"]) or posixpath.isabs(policy["private_cache_prefix"]))
             and all(_canonical(policy[name]) == _canonical(value) for name, value in attested.items()),
             "import policy attestation is incomplete")
    forwards = checks["forwards"]
    _require(isinstance(forwards, list) and len(forwards) == 4, "forward checks are incomplete")
    expected = {"output_shape": [1, 2, 50257], "output_dtype": "torch.float32", "output_tensor_bytes": 402056,
                "finite": True, "offload_complete": True, "disposal_complete": True}
    for index, row in enumerate(forwards, start=2):
        _require(isinstance(row, dict) and set(row) == set(expected) | {"sequence", "framed_response_bytes"}
                 and type(row["sequence"]) is int and row["sequence"] == index
                 and type(row["framed_response_bytes"]) is int and 201031 <= row["framed_response_bytes"] <= 4 * _MIB
                 and all(_canonical(row[name]) == _canonical(value) for name, value in expected.items()),
                 "forward output or disposal checks differ from protocol")


def _check_started(started):
    required = {"schema", "run_id", "started_utc", "protocol_sha256", "git_commit", "git_tree", "requested"}
    _require(isinstance(started, dict) and set(started) == required
             and started["schema"] == "resource-bootstrap-start-v1"
             and _digest(started["run_id"], 32) and _utc(started["started_utc"])
             and _digest(started["protocol_sha256"]) and _digest(started["git_commit"], 40)
             and _digest(started["git_tree"], 40) and _canonical(started["requested"]) == _canonical(_REQUESTED),
             "invalid bootstrap start record")


def _check_terminal(terminal, started, owner):
    fields = {"schema", "run_id", "ended_utc", "status", "child_exit_code", "clean_teardown",
              "failure_stage", "reason", "supervisor", "inventory"}
    _require(isinstance(terminal, dict) and set(terminal) == fields
             and terminal["schema"] == "resource-bootstrap-terminal-v1"
             and _digest(terminal["run_id"], 32) and _utc(terminal["ended_utc"])
             and terminal["status"] in ("success", "failed", "cancelled", "quarantined")
             and (terminal["child_exit_code"] is None or type(terminal["child_exit_code"]) is int)
             and type(terminal["clean_teardown"]) is bool,
             "invalid bootstrap terminal")
    if started is not None:
        _require(terminal["run_id"] == started["run_id"] and
                 datetime.fromisoformat(terminal["ended_utc"].replace("Z", "+00:00")) >=
                 datetime.fromisoformat(started["started_utc"].replace("Z", "+00:00")),
                 "start and terminal identities or times disagree")
    supervisor = terminal["supervisor"]
    if supervisor is None:
        _require(terminal["child_exit_code"] is None and terminal["clean_teardown"] is False
                 and (owner is None or terminal["status"] == "quarantined"),
                 "missing supervisor result cannot establish a clean exit")
    else:
        _require(isinstance(supervisor, dict) and set(supervisor) == {"run_id", "pid", "exit_code", "clean", "timed_out"}
                 and _digest(supervisor["run_id"], 32) and type(supervisor["pid"]) is int and supervisor["pid"] > 0
                 and (supervisor["exit_code"] is None or type(supervisor["exit_code"]) is int)
                 and type(supervisor["clean"]) is bool and type(supervisor["timed_out"]) is bool
                 and supervisor["exit_code"] == terminal["child_exit_code"]
                 and supervisor["clean"] is terminal["clean_teardown"]
                 and (not supervisor["clean"] or not supervisor["timed_out"]),
                 "supervisor result and terminal disagree")
        if owner is not None:
            _require(supervisor["run_id"] == owner["supervisor_run_id"] and supervisor["pid"] == owner["pid"],
                     "supervisor and child identities disagree")
        _require(not supervisor["clean"] or owner is not None, "clean terminal lacks child identity")
    if terminal["status"] == "success":
        _require(terminal["child_exit_code"] == 0 and type(terminal["child_exit_code"]) is int
                 and terminal["clean_teardown"] is True and terminal["failure_stage"] is None
                 and terminal["reason"] is None and supervisor is not None and supervisor["timed_out"] is False,
                 "success has a failed or unclean terminal")
    else:
        _require(all(isinstance(terminal[field], str) and terminal[field].strip()
                     for field in ("failure_stage", "reason"))
                 and (terminal["status"] != "quarantined" or terminal["clean_teardown"] is False),
                 "failure terminal lacks a stage, reason, or quarantine state")


def _check_owner_record(owner, started):
    _require(isinstance(owner, dict) and set(owner) == {"schema", "parent_run_id", "supervisor_run_id", "pid"}
             and owner["schema"] == "resource-bootstrap-owner-v1" and started is not None
             and owner["parent_run_id"] == started["run_id"] and _digest(owner["supervisor_run_id"], 32)
             and type(owner["pid"]) is int and owner["pid"] > 0, "invalid child owner identity")


def _safe_relative(path):
    return (isinstance(path, str) and path and "\\" not in path and ":" not in path
            and "\x00" not in path and not path.startswith("/")
            and all(part not in ("", ".", "..") for part in path.split("/")))


def _file_entries(entries):
    _require(isinstance(entries, list) and entries, "source file manifest is empty")
    names = []
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == {"path", "size", "sha256"}
                 and _safe_relative(entry["path"]) and type(entry["size"]) is int
                 and entry["size"] >= 0 and _digest(entry["sha256"]), "invalid source file entry")
        names.append(entry["path"])
    _require(names == sorted(set(names)), "source file manifest is unsorted or duplicated")
    return names


def _check_source(source, started):
    fields = {"schema", "git_commit", "git_tree", "repository_files", "python_executable", "distributions"}
    _require(isinstance(source, dict) and set(source) == fields
             and source["schema"] == "resource-bootstrap-source-v1"
             and source["git_commit"] == started["git_commit"] and source["git_tree"] == started["git_tree"],
             "source and start revisions disagree")
    names = _file_entries(source["repository_files"])
    protocol = "docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md"
    _require({protocol, "looking_glass/__init__.py", "looking_glass/gpu_service.py",
              "looking_glass/live_profile.py", "looking_glass/pretrained.py"}.issubset(names)
             and all(name.startswith("looking_glass/") or name == protocol for name in names),
             "source manifest lacks required tracked code or protocol")
    recorded_protocol = next(row for row in source["repository_files"] if row["path"] == protocol)
    _require(recorded_protocol["sha256"] == started["protocol_sha256"], "protocol source hash differs")
    executable = source["python_executable"]
    _require(isinstance(executable, dict) and set(executable) == {"size", "sha256"}
             and type(executable["size"]) is int and executable["size"] > 0 and _digest(executable["sha256"]),
             "Python executable identity is incomplete")
    distributions = source["distributions"]
    _require(isinstance(distributions, list), "source manifest lacks the distribution inventory")
    dist_names, all_paths = [], []
    for item in distributions:
        _require(isinstance(item, dict) and set(item) == {"name", "version", "files"}
                 and isinstance(item["name"], str)
                 and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item["name"])
                 and isinstance(item["version"], str)
                 and item["version"].strip(), "invalid distribution identity")
        dist_names.append(item["name"])
        all_paths.extend(_file_entries(item["files"]))
    required = {"torch", "transformers", "safetensors", "tokenizers", "huggingface-hub", "numpy"}
    _require(dist_names == sorted(set(dist_names)) and required.issubset(dist_names)
             and len(all_paths) == len(set(all_paths)), "distribution names or files are duplicated")


def _check_runtime(runtime, source):
    _require(isinstance(runtime, dict) and set(runtime) == _RUNTIME_FIELDS
             and runtime["schema"] == "resource-bootstrap-runtime-v1", "runtime fields differ from protocol")
    uuid = runtime["device_uuid"]
    _require(isinstance(uuid, str) and bool(re.fullmatch(
        r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid)),
        "invalid physical device identity")
    total = runtime["device_total_bytes"]
    _require(type(total) is int and 7 * _GIB <= total <= 9 * _GIB
             and isinstance(runtime["compute_capability"], list) and len(runtime["compute_capability"]) == 2
             and all(type(value) is int and value >= 0 for value in runtime["compute_capability"])
             and type(runtime["cudnn_version"]) is int and runtime["cudnn_version"] > 0,
             "invalid device capacity, capability or cuDNN version")
    fixed = {"device": "cuda:0", "cuda_visible_devices": uuid, "cuda_device_order": "PCI_BUS_ID",
             "allocator_backend": "native", "allocator_fraction": "0.20",
             "allocator_effective_limit_bytes": total // 5,
             "model_dtype": "torch.float32", "compute_dtype": "torch.float32",
             "attention_backend": "eager", "model_mode": "eval", "grad_mode": "inference",
             "autocast": "disabled", "use_cache": False, "compiled": False,
             "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
             "cudnn_allow_tf32": False, "cudnn_enabled": True, "cudnn_benchmark": False,
             "cudnn_deterministic": True, "deterministic_algorithms": True,
             "torch_num_threads": 1, "torch_num_interop_threads": 1,
             "input_limit_bytes": 64 * 1024, "output_tensor_limit_bytes": 512 * 1024,
             "response_wire_limit_bytes": 4 * _MIB,
             "execution_path": "looking_glass.resident_worker.ResidentWorker.execute"}
    _require(all(_canonical(runtime[key]) == _canonical(value) for key, value in fixed.items())
             and all(isinstance(runtime[key], str) and runtime[key].strip()
                     for key in ("python_version", "torch_version", "transformers_version",
                                 "cuda_runtime_version", "driver_version", "device_name")),
             "runtime flags or limits differ from protocol")
    import_paths = runtime["python_import_paths"]
    _require(isinstance(import_paths, list) and import_paths
             and all(_absolute_canonical_path(path) for path in import_paths)
             and len(import_paths) == len(set(import_paths)), "invalid ordered Python import paths")
    actual_environment = runtime["environment"]
    _require(isinstance(actual_environment, dict) and type(actual_environment.get("PATH")) is str
             and actual_environment["PATH"] and "\x00" not in actual_environment["PATH"],
             "controlled native library PATH is missing")
    environment = {"CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "PYTORCH_CUDA_ALLOC_CONF": "backend:native", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "PYTHONHASHSEED": "17", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "PYTORCH_ALLOC_CONF": None, "PYTORCH_NO_CUDA_MEMORY_CACHING": None,
                   "CUDA_LAUNCH_BLOCKING": None, "PYTHONPATH": None, "PYTHONHOME": None,
                   "PATH": actual_environment["PATH"]}
    _require(_canonical(actual_environment) == _canonical(environment), "environment controls differ from protocol")
    versions = {item["name"]: item["version"] for item in source["distributions"]}
    _require(versions["torch"] == runtime["torch_version"] and
             versions["transformers"] == runtime["transformers_version"],
             "distribution and runtime version identities disagree")


def _absolute_canonical_path(path):
    if not isinstance(path, str) or not path or "\x00" in path:
        return False
    if ntpath.splitdrive(path)[0]:
        return ntpath.isabs(path) and ntpath.normpath(path) == path
    return posixpath.isabs(path) and posixpath.normpath(path) == path


def _local_file(root, name, *, permit_links=False):
    _require(_safe_relative(name), "invalid local manifest path")
    try:
        base = Path(root).resolve()
        path = base
        for part in name.split("/"):
            path = path / part
            if not permit_links:
                info = path.lstat()
                _require(not path.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400),
                         "local source file traverses a link or reparse point")
        if not permit_links:
            _require(path.resolve().is_relative_to(base), "local source file escapes its root")
        _require(path.is_file(), "local source file is missing or not regular")
    except (OSError, ValueError):
        raise BootstrapAuditError("cannot inspect a local source file") from None
    return path


def _record_member(root, site_root, relative):
    """Interpret RECORD's relative path without following any intermediate link."""
    _require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative
             and "\x00" not in relative and not relative.startswith("/"), "invalid distribution RECORD path")
    base = Path(root).resolve()
    current = Path(site_root)
    _require(current.is_relative_to(base), "distribution RECORD starts outside the environment")
    try:
        for part in relative.split("/"):
            _require(part not in ("", "."), "invalid distribution RECORD path")
            if part == "..":
                current = current.parent
                _require(current.is_relative_to(base), "distribution RECORD points outside the environment")
                continue
            current = current / part
            info = current.lstat()
            _require(not current.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400),
                     "distribution RECORD traverses a link or reparse point")
        _require(current.is_file(), "distribution RECORD file is unavailable")
    except (OSError, ValueError):
        raise BootstrapAuditError("cannot inspect distribution RECORD path") from None
    return current.relative_to(base).as_posix()


def _match_local(path, size, expected_hash):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_size = path.stat().st_size
    except OSError:
        raise BootstrapAuditError("cannot hash a local file") from None
    _require(actual_size == size and digest.hexdigest() == expected_hash, "local file bytes differ from the manifest")


def _verify_repository(source, directory):
    root = Path(directory).resolve()
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        commit = git("rev-parse", "HEAD").decode("ascii").strip()
        tree = git("show", "-s", "--format=%T", "HEAD").decode("ascii").strip()
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        entries = git("ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", "looking_glass/",
                      "docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md").split(b"\0")
        tracked = []
        for entry in entries:
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            mode, kind, _ = metadata.decode("ascii").split(" ")
            _require(mode in ("100644", "100755") and kind == "blob", "tracked source includes a non-regular file")
            tracked.append(path.decode("utf-8"))
    except BootstrapAuditError:
        raise
    except (OSError, UnicodeError, subprocess.CalledProcessError, ValueError):
        raise BootstrapAuditError("cannot inspect the local Git source revision") from None
    _require(commit == source["git_commit"] and tree == source["git_tree"] and not status,
             "local Git revision, tree, or clean state differs")
    expected = [entry["path"] for entry in source["repository_files"]]
    _require(sorted(tracked) == expected, "local tracked source coverage differs from the manifest")
    for entry in source["repository_files"]:
        _match_local(_local_file(root, entry["path"]), entry["size"], entry["sha256"])


def _verify_environment(source, directory, site_packages):
    """Compare all distributions discovered in an explicitly selected site directory."""
    root = Path(directory).resolve()
    site = Path(site_packages)
    try:
        _require(site.is_absolute(), "site-packages must be an absolute local path")
        relative_site = site.relative_to(root)
        current = root
        for part in relative_site.parts:
            current = current / part
            info = current.lstat()
            _require(stat.S_ISDIR(info.st_mode) and not current.is_symlink()
                     and not (getattr(info, "st_file_attributes", 0) & 0x400),
                     "site-packages traverses a link or reparse point")
        _require(current.is_dir(), "site-packages is unavailable")
        actual = []
        for child in current.iterdir():
            if not child.name.lower().endswith((".dist-info", ".egg-info")):
                continue
            info = child.lstat()
            _require(child.name.lower().endswith(".dist-info") and stat.S_ISDIR(info.st_mode)
                     and not child.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400),
                     "installed distribution metadata is not an ordinary dist-info directory")
            metadata_path = _local_file(root, (child / "METADATA").relative_to(root).as_posix())
            parsed = BytesParser().parsebytes(metadata_path.read_bytes(), headersonly=True)
            _require(len(parsed.get_all("Name", [])) == len(parsed.get_all("Version", [])) == 1,
                     "installed distribution metadata has ambiguous identity")
            name = re.sub(r"[-_.]+", "-", parsed["Name"]).lower()
            version = parsed["Version"]
            _require(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) and version,
                     "installed distribution metadata is incomplete")
            actual.append((name, version))
        _require(len(actual) == len({name for name, _ in actual})
                 and sorted(actual) == sorted((item["name"], item["version"]) for item in source["distributions"]),
                 "selected environment has missing, unlisted or duplicate distributions")
    except BootstrapAuditError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError):
        raise BootstrapAuditError("cannot inspect selected local package environment") from None
    for dist in source["distributions"]:
        entries = dist["files"]
        records = [entry["path"] for entry in entries if re.search(r"(?:^|/)[^/]+\.dist-info/RECORD$", entry["path"])]
        metadata = [entry["path"] for entry in entries if re.search(r"(?:^|/)[^/]+\.dist-info/METADATA$", entry["path"])]
        _require(len(records) == len(metadata) == 1 and
                 records[0].rsplit("/", 1)[0] == metadata[0].rsplit("/", 1)[0],
                 "distribution lacks an unambiguous RECORD and METADATA")
        for entry in entries:
            _match_local(_local_file(root, entry["path"]), entry["size"], entry["sha256"])
        try:
            metadata_bytes = _local_file(root, metadata[0]).read_bytes()
            parsed = BytesParser().parsebytes(metadata_bytes, headersonly=True)
            _require(len(parsed.get_all("Name", [])) == len(parsed.get_all("Version", [])) == 1,
                     "distribution metadata has ambiguous identity")
            normalized = re.sub(r"[-_.]+", "-", parsed["Name"]).lower()
            version = parsed["Version"]
            record_path = _local_file(root, records[0])
            with record_path.open("r", encoding="utf-8", newline="") as stream:
                record_text = stream.read()
            site_root = record_path.parent.parent
            _require(site_root == current, "distribution metadata belongs to a different site-packages")
            listed = []
            for row in csv.reader(io.StringIO(record_text, newline="")):
                _require(len(row) == 3, "invalid distribution RECORD entry")
                listed.append(_record_member(root, site_root, row[0]))
        except BootstrapAuditError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError, csv.Error):
            raise BootstrapAuditError("cannot inspect distribution metadata") from None
        _require(normalized == dist["name"] and version == dist["version"]
                 and len(listed) == len(set(listed))
                 and sorted(listed) == [entry["path"] for entry in entries],
                 "distribution RECORD coverage or identity differs from the manifest")


def _verify_assets(directory):
    for name, (size, digest) in _ASSETS.items():
        # Hugging Face snapshot files may be links to content-addressed blobs;
        # this read-only check hashes the bytes reached by the supplied path.
        _match_local(_local_file(directory, name, permit_links=True), size, digest)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def audit_profile_bytes(document, evidence, source_manifest, approved_id):
    """Verify one bootstrap profile and independently reconstruct its estimate.

    The caller must obtain ``approved_id`` from outside the run directory.
    This function reveals no device UUID, local paths, or manifest content.
    """
    try:
        profile = load_profile(document, evidence=evidence, source_manifest=source_manifest,
                               expected_profile_id=approved_id)
        parsed = json.loads(document)
        key = parsed["key"]
        samples = parsed["samples"]
        if _sha(_canonical(parsed)) != approved_id:
            raise BootstrapAuditError("independently recomputed profile ID differs")
        model_fields = {name: key[name] for name in _MODEL_FIELDS}
        if _sha(_canonical({"schema": "model-runtime-identity-v1", "identity": model_fields})) != key["model_id"]:
            raise BootstrapAuditError("independently recomputed model identity differs")
        workload_id = _sha(_canonical({"schema": "workload-key-v1", "key": key}))
        rebuilt = _canonical({"schema": "resource-profile-evidence-v1", "key": key, "samples": samples})
        if evidence != rebuilt or any(sample["workload_sha256"] != workload_id for sample in samples):
            raise BootstrapAuditError("profile evidence does not match the saved observations")
        source_hash = _sha(source_manifest)
        if source_hash != key["source_sha256"] or source_hash != parsed["provenance"]["source_manifest_sha256"]:
            raise BootstrapAuditError("bootstrap source identities disagree")

        per_sample = []
        for index, sample in enumerate(samples, start=1):
            before, after = sample["before"], sample["after"]
            allocated = sample["peak_allocated_bytes"] - before["process_allocated_bytes"]
            reserved = sample["peak_reserved_bytes"] - before["process_reserved_bytes"]
            free_drop = before["device_free_bytes"] - after["device_free_bytes"]
            per_sample.append({"index": index, "seed": sample["seed"],
                               "allocated_delta_bytes": allocated, "reserved_delta_bytes": reserved,
                               "device_free_endpoint_drop_bytes": free_drop,
                               "observed_incremental_proxy_bytes": max(0, allocated, reserved, free_drop)})
        inputs = key["batch_size"] * key["sequence_length"] * sum(
            _INPUT_WIDTHS[dtype] for _, dtype in key["input_dtypes"])
        observed = max(item["observed_incremental_proxy_bytes"] for item in per_sample)
        allowance = inputs + key["output_limit_bytes"] + key["max_gpu_trace_bytes"]
        basis = observed + allowance
        percentage_margin = (basis * parsed["margin_bps"] + 9999) // 10000
        estimate = basis + parsed["margin_bytes"] + percentage_margin
        if estimate != profile.estimated_incremental_bytes:
            raise BootstrapAuditError("independent resource estimate differs")
        return {
            "profile_id": approved_id, "workload_sha256": workload_id,
            "operation": key["operation"], "warmup_state": key["warmup_state"],
            "samples": per_sample, "input_allowance_bytes": inputs,
            "output_allowance_bytes": key["output_limit_bytes"],
            "gpu_trace_allowance_bytes": key["max_gpu_trace_bytes"],
            "observed_proxy_max_bytes": observed, "basis_bytes": basis,
            "fixed_margin_bytes": parsed["margin_bytes"], "percentage_margin_bytes": percentage_margin,
            "estimated_incremental_bytes": estimate,
        }
    except BootstrapAuditError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError):
        raise BootstrapAuditError("profile, source, or observation evidence is invalid") from None


def _check_key(key, stage, runtime, source_hash, runtime_hash, warmup_hash, capture_hash):
    common = {
        "checkpoint_sha256": _ASSETS["model.safetensors"][1], "checkpoint_revision": _REVISION,
        "model_config_sha256": _ASSETS["config.json"][1], "source_sha256": source_hash,
        "runtime_settings_sha256": runtime_hash, "warmup_protocol_sha256": warmup_hash,
        "capture_policy_sha256": capture_hash, "capture_mode": "none", "max_gpu_trace_bytes": 0,
        "max_host_trace_bytes": 0,
    }
    for field in ("device", "device_uuid", "device_name", "device_total_bytes", "compute_capability",
                  "python_version", "torch_version", "transformers_version", "cuda_runtime_version",
                  "driver_version", "model_dtype", "compute_dtype", "attention_backend", "model_mode",
                  "grad_mode", "autocast", "use_cache", "compiled"):
        common[field] = runtime[field]
    common.update({"operation": "model_load" if stage == "load" else "forward",
                   "warmup_state": "warm" if stage == "warm" else "cold",
                   "batch_size": 0 if stage == "load" else 1,
                   "sequence_length": 0 if stage == "load" else 2,
                   "input_dtypes": [] if stage == "load" else [["input_ids", "torch.int64"]],
                   "output_limit_bytes": 0 if stage == "load" else 512 * 1024})
    _require(all(field in key and _canonical(key[field]) == _canonical(value) for field, value in common.items()),
             "profile workload key differs from the fixed bootstrap")


def audit_run(directory, *, approved_pins=None, repository=None, environment_root=None, site_packages=None,
              python_executable=None, asset_directory=None, _candidate_terminal=_NO_CANDIDATE):
    """Audit a private v1 directory, without CUDA access or filesystem writes.

    Candidate profile IDs can be recomputed without pins, but that does not
    constitute independent approval. A successful terminal is an attestation,
    not proof of its claims. The returned summary excludes UUIDs, path strings
    and failure reasons. Keep caller-generated tracebacks private.
    """
    root = Path(directory)
    _require(not _unsafe_link(root) and root.is_dir(), "bootstrap directory is unavailable or unsafe")
    if approved_pins is not None:
        _require(isinstance(approved_pins, dict) and set(approved_pins) == {"load", "cold", "warm"}
                 and all(_digest(value) for value in approved_pins.values()), "three independent profile pins are required")
    start_path = root / "started.json"
    _require(not _unsafe_link(start_path), "start record is a link")
    started = _decode_canonical(_read(root, "started.json")) if start_path.exists() else None
    if started is not None:
        _check_started(started)
    owner_path = root / "owner.json"
    _require(not _unsafe_link(owner_path), "owner record is a link")
    owner = _decode_canonical(_read(root, "owner.json")) if owner_path.exists() else None
    if owner is not None:
        _check_owner_record(owner, started)
    terminal_path = root / "terminal.json"
    _require(not _unsafe_link(terminal_path), "terminal record is a link")
    candidate = _candidate_terminal is not _NO_CANDIDATE
    if candidate:
        _require(not terminal_path.exists(), "cannot validate a candidate over an existing terminal")
        try:
            terminal = _decode_canonical(_canonical(_candidate_terminal))
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise BootstrapAuditError("invalid candidate terminal") from None
    elif not terminal_path.exists():
        observations, partial = _observations(root, require_complete=False)
        admissions, admission_partial = _admissions(root, require_complete=False)
        return {"status": "incomplete", "publication": "none", "started": started is not None,
                "completed_observations": len(observations), "trailing_partial_observation": partial,
                "completed_admissions": len(admissions), "trailing_partial_admission": admission_partial,
                "pins_verified": False}
    else:
        terminal = _decode_canonical(_read(root, "terminal.json"))
    _check_terminal(terminal, started, owner)
    _inventory(root, terminal)
    success = terminal["status"] == "success"
    observations, partial = _observations(root, require_complete=success)
    admissions, admission_partial = _admissions(root, require_complete=success)
    if not success:
        return {"status": terminal["status"], "publication": "candidate" if candidate else "terminal",
                "started": started is not None,
                "completed_observations": len(observations), "trailing_partial_observation": partial,
                "completed_admissions": len(admissions), "trailing_partial_admission": admission_partial,
                "clean_teardown_attested": terminal["clean_teardown"], "pins_verified": False}

    _require(started is not None and {item.name for item in root.iterdir()} ==
             (_FILES if candidate else _FILES | {"terminal.json"}),
             "success does not contain the exact protocol file set")
    source_bytes, runtime_bytes = _read(root, "source.json"), _read(root, "runtime.json")
    source, runtime = _decode_canonical(source_bytes), _decode_canonical(runtime_bytes)
    _check_source(source, started)
    _check_runtime(runtime, source)
    _check_execution_checks(_decode_canonical(_read(root, "checks.json")), runtime, owner)
    _require(all(row["telemetry"]["device_total_bytes"] == runtime["device_total_bytes"] for row in admissions),
             "admission and runtime device totals disagree")
    assets = _decode_canonical(_read(root, "assets.json"))
    expected_assets = {"schema": "resource-bootstrap-assets-v1", "model_id": "distilbert/distilgpt2",
                       "revision": _REVISION,
                       "assets": [{"name": name, "size": size, "sha256": digest}
                                  for name, (size, digest) in sorted(_ASSETS.items())]}
    _require(_canonical(assets) == _canonical(expected_assets), "asset manifest does not match the pinned checkpoint")
    expected_manifests = {"load-case.json": _LOAD_CASE, "forward-case.json": _FORWARD_CASE,
                          "warmup.json": _WARMUP, "capture.json": _CAPTURE}
    manifest_bytes = {}
    for name, expected in expected_manifests.items():
        raw = _read(root, name)
        _decode_canonical(raw)
        _require(raw == _canonical(expected), "fixed workload manifest differs")
        manifest_bytes[name] = raw
    source_hash, runtime_hash = _sha(source_bytes), _sha(runtime_bytes)
    warmup_hash, capture_hash = _sha(manifest_bytes["warmup.json"]), _sha(manifest_bytes["capture.json"])
    per_profile = {}
    for stage in ("load", "cold", "warm"):
        document = _read(root, stage + ".profile.json")
        profile_data = _decode_canonical(document)
        candidate_profile_id = _sha(_canonical(profile_data))
        pin = candidate_profile_id if approved_pins is None else approved_pins[stage]
        evidence = _read(root, stage + ".evidence.json")
        result = audit_profile_bytes(document, evidence, source_bytes, pin)
        key = profile_data["key"]
        _check_key(key, stage, runtime, source_hash, runtime_hash, warmup_hash, capture_hash)
        selected = [row for row in observations if row["stage"] == stage]
        expected_case = _sha(manifest_bytes["load-case.json" if stage == "load" else "forward-case.json"])
        _require(all(_canonical(row["key"]) == _canonical(key) and row["sample"].get("case_sha256") == expected_case
                     for row in selected)
                 and _canonical([row["sample"] for row in selected]) == _canonical(profile_data["samples"]),
                 "profile samples differ from the ordered raw observations")
        _require(profile_data["margin_bytes"] == (512 * _MIB if stage == "load" else 256 * _MIB)
                 and type(profile_data["margin_bps"]) is int and profile_data["margin_bps"] == 2500,
                 "profile margins differ from the protocol")
        per_profile[stage] = result
    local = {"repository": "not_checked", "distribution_records": "not_checked",
             "python_executable": "not_checked", "checkpoint_assets": "not_checked"}
    if repository is not None:
        _verify_repository(source, repository)
        local["repository"] = "matched"
    if environment_root is not None or site_packages is not None:
        _require(environment_root is not None and site_packages is not None,
                 "environment root and selected site-packages must be supplied together")
        selected = os.path.normcase(os.path.normpath(str(Path(site_packages))))
        _require(selected in {os.path.normcase(os.path.normpath(path)) for path in runtime["python_import_paths"]},
                 "selected site-packages differs from the recorded import path")
        _verify_environment(source, environment_root, site_packages)
        local["distribution_records"] = "matched"
    if python_executable is not None:
        executable = Path(python_executable)
        _require(executable.is_file(), "local Python executable is unavailable")
        _match_local(executable, source["python_executable"]["size"], source["python_executable"]["sha256"])
        local["python_executable"] = "matched"
    if asset_directory is not None:
        _verify_assets(asset_directory)
        local["checkpoint_assets"] = "matched"
    return {"status": "success", "publication": "candidate" if candidate else "terminal",
            "completed_observations": len(observations), "completed_admissions": len(admissions),
            "clean_teardown_attested": True, "pins_verified": approved_pins is not None,
            "source_revision": started["git_commit"], "profiles": per_profile, "local_checks": local,
            "limitations": ["hashes and terminal inventory are not authenticated by this audit",
                            "matched local files do not prove what code, model or runtime was executed",
                            "runtime flags, synchronization and device isolation are recorded attestations",
                            "sampled allocator peaks and device-free endpoints are not guaranteed bounds"]}


def audit_candidate(directory, terminal, *, approved_pins=None, repository=None, environment_root=None, site_packages=None,
                    python_executable=None, asset_directory=None):
    """Validate a prospective terminal in memory after owner exit, without writing it.

    No terminal may already exist. A matching report does not authenticate the
    owner or independently attest clean teardown; the parent must protect the
    files from changes before its own atomic publication.
    """
    return audit_run(directory, approved_pins=approved_pins, repository=repository,
                     environment_root=environment_root, site_packages=site_packages, python_executable=python_executable,
                     asset_directory=asset_directory, _candidate_terminal=terminal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="private bootstrap run directory")
    parser.add_argument("--pins", help="separate canonical JSON object with load/cold/warm approved IDs")
    parser.add_argument("--repository", help="clean checkout at the recorded source commit")
    parser.add_argument("--environment-root", help="environment root containing recorded distribution files")
    parser.add_argument("--site-packages", help="explicit selected site-packages inside the environment root")
    parser.add_argument("--python-executable", help="actual base Python executable to compare")
    parser.add_argument("--asset-directory", help="directory containing the seven pinned checkpoint assets")
    args = parser.parse_args(argv)
    try:
        pins = None
        if args.pins:
            pin_path = Path(args.pins).resolve()
            _require(not pin_path.is_relative_to(Path(args.directory).resolve()),
                     "approved pins must be stored separately from the run")
            pins = _decode_canonical(_read(pin_path.parent, pin_path.name))
        report = audit_run(args.directory, approved_pins=pins, repository=args.repository,
                           environment_root=args.environment_root, site_packages=args.site_packages,
                           python_executable=args.python_executable,
                           asset_directory=args.asset_directory)
    except Exception:
        # Artifact paths, device IDs, and failure reasons are private. The API
        # may be used privately for diagnostics; the CLI never prints a traceback.
        print(json.dumps({"status": "invalid", "error": "artifact verification failed"}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "success" and report["pins_verified"] else 2


if __name__ == "__main__":
    sys.exit(main())
