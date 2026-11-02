"""Bind saved pair/dose evidence to externally supplied public identities.

Checks in this module are consistency checks on supplied bytes and reported
observations. They do not authenticate a process, timestamp, model execution,
custodian, hidden split, or source of externally supplied identities.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import re
import stat
import subprocess

from .copy_pairs_dose_audit import _json_object, _typed_equal, audit_journal, replay_result


EXPECTED_FIELDS = frozenset({
    "source_revision", "source_tree", "source_sha256", "protocol_revision", "protocol_sha256",
    "case_revision", "case_sha256", "clarification_revision", "clarification_sha256",
    "reference_revision", "reference_source_sha256", "exposure_inventories",
    "manifest_exposure_inventories", "weights_sha256", "asset_sha256", "dependencies", "pre_run_sha256",
    "execution_context_sha256", "receipt_sha256",
})
READER_PATHS = frozenset({"looking_glass/copy_pairs_dose_audit.py",
                          "looking_glass/copy_pairs_dose_evidence.py",
                          "scripts/audit_copy_pairs_dose_v1.py"})
ASSET_NAMES = frozenset({"config.json", "generation_config.json", "merges.txt", "model.safetensors",
                         "tokenizer.json", "tokenizer_config.json", "vocab.json"})
DEPENDENCY_NAMES = frozenset({"python", "torch", "numpy", "transformers", "gpt2_source_sha256",
                              "conv1d_source_sha256", "gpt2_module", "conv1d_module", "byteorder", "machine", "system"})
FIXED_DEPENDENCIES = {"python": "3.11.15", "torch": "2.8.0+cu128", "numpy": "2.4.6", "transformers": "4.57.3",
                      "byteorder": "little", "gpt2_source_sha256": "b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80",
                      "conv1d_source_sha256": "e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(value, length):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def _source_map(value):
    if type(value) is not dict or not 1 <= len(value) <= 4096:
        return False
    for path, sha in value.items():
        if (type(path) is not str or not path or len(path) > 512 or path.startswith("/")
                or "\\" in path or ":" in path or any(part in ("", ".", "..") for part in path.split("/"))
                or not _hex(sha, 64)):
            return False
    return True


def _private_status(value, case_sha):
    if (type(value) is not dict or set(value) != {"case_sha256", "natural", "injection"}
            or value["case_sha256"] != case_sha):
        raise ValueError("private custody status is incomplete")
    for lane in ("natural", "injection"):
        item = value[lane]
        required = {"reporter", "status", "scope", "reported_utc"}
        if lane == "injection":
            required.add("target_boundary")
        if (type(item) is not dict or set(item) != required
                or item["status"] not in ("no_protected_rows", "no_overlap")
                or any(type(item.get(key)) is not str or not item[key] or len(item[key]) > 512
                       for key in ("reporter", "scope", "reported_utc"))
                or (lane == "injection" and item["target_boundary"] != "proceed_no_direct_conflict")):
            raise ValueError("private custody status is not an explicit scoped report")
        try:
            observed = datetime.fromisoformat(item["reported_utc"].replace("Z", "+00:00"))
            if observed.tzinfo is None or observed.utcoffset() is None:
                raise ValueError("missing timezone")
        except (ValueError, OverflowError) as error:
            raise ValueError("private custody observation time is invalid") from error


def verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected):
    """Check external identity pins, reported process outcome and file hashes.

    ``expected`` must be supplied independently, including exact public source
    and later reader identities and exact pre-run, execution-context and receipt
    byte hashes. The caller must independently establish their origin. This
    function does not check the reader checkout or claim preregistration. Raw
    payload hashes in the receipt are validated for form and inventory here;
    the caller must read and compare those payloads, reconcile the journal and
    run strict numerical replay before reporting saved-data consistency.
    """
    if type(expected) is not dict or set(expected) != EXPECTED_FIELDS:
        raise ValueError("expected source and evidence identities are incomplete")
    revisions = ("source_revision", "source_tree", "protocol_revision", "case_revision",
                 "clarification_revision", "reference_revision")
    digests = ("protocol_sha256", "case_sha256", "clarification_sha256", "weights_sha256",
               "pre_run_sha256", "execution_context_sha256", "receipt_sha256")
    if (any(not _hex(expected[key], 40) for key in revisions)
            or any(not _hex(expected[key], 64) for key in digests)
            or not _source_map(expected["source_sha256"]) or not _source_map(expected["asset_sha256"])
            or set(expected["asset_sha256"]) != ASSET_NAMES
            or expected["asset_sha256"].get("model.safetensors") != expected["weights_sha256"]
            or not _source_map(expected["reference_source_sha256"])
            or set(expected["reference_source_sha256"]) != READER_PATHS):
        raise ValueError("expected source or digest identity is invalid")
    dependencies = expected["dependencies"]
    if (type(dependencies) is not dict or set(dependencies) != DEPENDENCY_NAMES
            or any(type(key) is not str or not key or len(key) > 128
                   or type(value) is not str or not value or len(value) > 512 for key, value in dependencies.items())
            or not _hex(dependencies["gpt2_source_sha256"], 64)
            or not _hex(dependencies["conv1d_source_sha256"], 64)
            or dependencies["gpt2_module"] != "transformers.models.gpt2.modeling_gpt2"
            or dependencies["conv1d_module"] != "transformers.pytorch_utils"
            or any(dependencies[key] != value for key, value in FIXED_DEPENDENCIES.items())):
        raise ValueError("expected dependency identity is invalid")
    exposure = expected["exposure_inventories"]
    manifest_exposure = expected["manifest_exposure_inventories"]
    if (type(exposure) is not dict or len(exposure) != 2 or "v13" not in exposure
            or any(not _hex(value, 64) for value in exposure.values())
            or type(manifest_exposure) is not dict
            or set(manifest_exposure) != {"v13", "reservation", "reservation_version"}):
        raise ValueError("expected exposure identity is invalid")
    other = next((key for key in exposure if key != "v13"), None)
    if (type(other) is not str or re.fullmatch(r"v[0-9]+", other) is None or int(other[1:]) < 15
            or manifest_exposure["v13"] != exposure["v13"] or manifest_exposure["reservation"] != exposure[other]
            or manifest_exposure["reservation_version"] != "distilgpt2-development-exposures-" + other):
        raise ValueError("expected exposure bindings disagree")
    limits = ((case_raw, 1024 * 1024), (manifest_raw, 2 * 1024 * 1024),
              (result_raw, 16 * 1024 * 1024), (pre_raw, 65536), (context_raw, 65536),
              (receipt_raw, 65536), (journal_raw, 8 * 1024 * 1024))
    if any(type(data) is not bytes or len(data) > maximum for data, maximum in limits):
        raise ValueError("evidence bytes are missing or oversized")
    case, manifest, result, pre, context, receipt = map(_json_object,
        (case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw))
    if any("fabricated_only" in document for document in (case, manifest, result, pre, context, receipt)):
        raise ValueError("fabricated evidence is not a process observation")
    if (_digest(case_raw) != expected["case_sha256"] or _digest(pre_raw) != expected["pre_run_sha256"]
            or _digest(context_raw) != expected["execution_context_sha256"]
            or _digest(receipt_raw) != expected["receipt_sha256"]):
        raise ValueError("case, launch, context or receipt digest differs")
    if (case.get("schema") != "distilgpt2-copy-head-pairs-dose-cases-v1"
            or case.get("tokenizer_sha256") != expected["asset_sha256"]["tokenizer.json"]
            or case.get("model_id") != "distilbert/distilgpt2"
            or case.get("model_revision") != "2290a62682d06624634c1f46a6ad5be0f47f38aa"
            or type(case.get("cases")) is not list or len(case["cases"]) != 32):
        raise ValueError("case inventory is incomplete")
    ids = [f"copy-pairs-{group:02d}-{variant}" for group in range(1, 9) for variant in range(4)]
    if any(type(item) is not dict or item.get("case_id") != identity for item, identity in zip(case["cases"], ids)):
        raise ValueError("case identities differ")
    if (pre.get("schema") != "copying-head-pairs-dose-launch-v1"
            or any(pre.get(key) != expected[key] for key in ("source_revision", "source_tree", "protocol_revision", "protocol_sha256",
                "case_revision", "case_sha256", "clarification_revision", "clarification_sha256"))
            or ("reference_revision" in pre and pre["reference_revision"] != expected["reference_revision"])
            or ("reference_source_sha256" in pre and not _typed_equal(pre["reference_source_sha256"], expected["reference_source_sha256"]))
            or not _typed_equal(pre.get("exposure_inventories"), exposure)
            or type(pre.get("expected_forwards")) is not int or pre["expected_forwards"] != 576
            or type(pre.get("expected_raw_payload_bytes")) is not int or pre["expected_raw_payload_bytes"] != 167696640):
        raise ValueError("launch identity or scope differs")
    environment = pre.get("environment")
    expected_environment = {"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": None, "ATEN_CPU_CAPABILITY": None}
    if (pre.get("device") != "cpu" or pre.get("model_id") != "distilbert/distilgpt2"
            or pre.get("model_revision") != "2290a62682d06624634c1f46a6ad5be0f47f38aa"
            or pre.get("weights_sha256") != expected["weights_sha256"] or type(environment) is not dict
            or not _typed_equal(environment, expected_environment)):
        raise ValueError("launch runtime or checkpoint differs")
    if (context.get("schema") != "copying-head-pairs-dose-execution-context-v1"
            or context.get("source_revision") != expected["source_revision"]
            or context.get("case_sha256") != expected["case_sha256"]
            or not _typed_equal(context.get("exposure_inventories"), exposure)):
        raise ValueError("execution context identity differs")
    _private_status(pre.get("private_split_status"), expected["case_sha256"])
    _private_status(context.get("private_split_status"), expected["case_sha256"])
    source = manifest.get("source")
    if (type(source) is not dict or source.get("revision") != expected["source_revision"]
            or source.get("tree") != expected["source_tree"]
            or not _typed_equal(source.get("sha256"), expected["source_sha256"])):
        raise ValueError("manifest source identity differs")
    if not _typed_equal(manifest.get("dependencies"), dependencies):
        raise ValueError("manifest dependency identity differs")
    if (manifest.get("schema") != "distilgpt2-copy-head-pairs-dose-run-v1"
            or not _typed_equal(manifest.get("cases"), case["cases"])
            or manifest.get("cases_sha256") != expected["case_sha256"]
            or manifest.get("protocol_sha256") != expected["protocol_sha256"]
            or not _typed_equal(manifest.get("numeric_clarification"),
                {"revision": expected["clarification_revision"], "sha256": expected["clarification_sha256"]})
            or not _typed_equal(manifest.get("exposure_inventories"), manifest_exposure)
            or manifest.get("model_id") != "distilbert/distilgpt2"
            or manifest.get("model_revision") != "2290a62682d06624634c1f46a6ad5be0f47f38aa"
            or manifest.get("weights_sha256") != expected["weights_sha256"]
            or type(manifest.get("raw_payload_bytes")) is not int or manifest["raw_payload_bytes"] != 167696640):
        raise ValueError("manifest case, protocol, exposure or checkpoint differs")
    if (result.get("schema") != "distilgpt2-copy-head-pairs-dose-result-v1" or result.get("status") != "complete"
            or result.get("manifest_sha256") != _digest(manifest_raw)
            or type(result.get("cases")) is not list or len(result["cases"]) != 576
            or type(result.get("checkpoint")) is not dict
            or result["checkpoint"].get("weights_sha256") != expected["weights_sha256"]
            or not _typed_equal(result["checkpoint"].get("asset_sha256"), expected["asset_sha256"])
            or result["checkpoint"].get("model_id") != "distilbert/distilgpt2"
            or result["checkpoint"].get("model_revision") != "2290a62682d06624634c1f46a6ad5be0f47f38aa"
            or result["checkpoint"].get("model_class") != "GPT2LMHeadModel"
            or result["checkpoint"].get("tokenizer_class") != "GPT2TokenizerFast"
            or result["checkpoint"].get("tokenizer_id") != "distilbert/distilgpt2"
            or result["checkpoint"].get("tokenizer_revision") != "2290a62682d06624634c1f46a6ad5be0f47f38aa"
            or result["checkpoint"].get("device") != "cpu" or result["checkpoint"].get("dtype") != "float32"
            or result["checkpoint"].get("attention_implementation") != "eager"
            or result["checkpoint"].get("use_cache") is not False
            or type(result["checkpoint"].get("parameter_count")) is not int or result["checkpoint"]["parameter_count"] != 81912576
            or result["checkpoint"].get("torch_version") != dependencies.get("torch")
            or result["checkpoint"].get("transformers_version") != dependencies.get("transformers")
            or type(result.get("raw_payload_bytes")) is not int or result["raw_payload_bytes"] != 167696640):
        raise ValueError("result identity or checkpoint differs")
    for key in ("submitted_forwards", "model_call_returns_observed", "returned_forwards"):
        if type(result.get(key)) is not int or result[key] != 576:
            raise ValueError("result forward counter differs")
    for name in ("model_state_sha256", "model_buffers_sha256"):
        before, after = result.get(name + "_before"), result.get(name + "_after")
        if not _hex(before, 64) or before != after:
            raise ValueError("reported model state or buffer digest differs")
    runtime, intended, launch_intended = result.get("runtime"), manifest.get("intended_runtime"), pre.get("intended_runtime")
    fixed = {"device": "cpu", "dtype": "float32", "model_eval": True,
             "attention_implementation": "eager", "use_cache": False, "intraop_threads": 1,
             "deterministic_algorithms": False, "float32_matmul_precision": "highest", "cpu_rng_seed": 20261029}
    extras = {"cuda_visibility": "-1", "cuda_initialized": False, "cpu_autocast": False, "guard_blocked_calls": []}
    backend = {"MKL_CBWR": {"present": False, "value": None}, "ATEN_CPU_CAPABILITY": {"present": False, "value": None}}
    numeric = {"input_preserved": True, "half_min_normal_bits": 0x00400000, "numpy_cast_preserved": True}
    launch_runtime_keys = ("dtype", "attention_implementation", "use_cache", "intraop_threads",
                           "deterministic_algorithms", "float32_matmul_precision", "cpu_rng_seed")
    if (type(runtime) is not dict or type(intended) is not dict or type(launch_intended) is not dict
            or any(not _typed_equal(launch_intended.get(key), dependencies.get(key))
                   for key in ("python", "torch", "numpy", "transformers"))
            or any(not _typed_equal(launch_intended.get(key), fixed[key]) for key in launch_runtime_keys)
            or any(not _typed_equal(intended.get(key), value) or not _typed_equal(runtime.get(key), value)
                   for key, value in fixed.items())
            or any(not _typed_equal(runtime.get(key), value) for key, value in extras.items())
            or type(intended.get("interop_threads")) is not int or intended["interop_threads"] < 1
            or not _typed_equal(runtime.get("interop_threads"), intended["interop_threads"])
            or type(intended.get("mkldnn_enabled")) is not bool
            or not _typed_equal(runtime.get("mkldnn_enabled"), intended["mkldnn_enabled"])
            or not _typed_equal(manifest.get("cpu_backend_environment_at_torch_import"), backend)
            or not _typed_equal(runtime.get("cpu_backend_environment"), backend)
            or not _typed_equal(manifest.get("initial_numeric_probe"), numeric)
            or not _typed_equal(runtime.get("numeric_probe"), numeric)):
        raise ValueError("reported runtime differs from the CPU protocol")
    if (receipt.get("schema") != "copying-head-pairs-dose-process-observation-v1"
            or receipt.get("source_revision") != expected["source_revision"]
            or receipt.get("pre_run_sha256") != expected["pre_run_sha256"]
            or receipt.get("execution_context_sha256") != expected["execution_context_sha256"]
            or type(receipt.get("process_exit_code")) is not int or receipt["process_exit_code"] != 0
            or "launch_error_type" not in receipt or receipt["launch_error_type"] is not None
            or receipt.get("interrupted") is not False or receipt.get("failure_file_present") is not False):
        raise ValueError("reported process receipt is incomplete")
    for stream in ("stdout", "stderr"):
        count, digest = receipt.get(stream + "_bytes"), receipt.get(stream + "_sha256")
        if type(count) is not int or not 0 <= count <= 4 * 1024 * 1024 or not _hex(digest, 64):
            raise ValueError("reported process receipt stream is incomplete")
    inventory = receipt.get("run_sha256")
    names = {identity + ".f32" for identity in ids} | {"manifest.json", "result.json", "attempt.jsonl"}
    if (type(inventory) is not dict or set(inventory) != names or any(not _hex(value, 64) for value in inventory.values())
            or inventory["manifest.json"] != _digest(manifest_raw) or inventory["result.json"] != _digest(result_raw)
            or inventory["attempt.jsonl"] != _digest(journal_raw)):
        raise ValueError("receipt inventory or metadata digest differs")
    return {"cases": case["cases"], "manifest": manifest, "result": result,
            "pre_run": pre, "execution_context": context, "receipt": receipt}


def _read_regular(path, maximum):
    """Read a bounded ordinary file under the caller's quiescent-directory contract."""
    path = Path(path)
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("evidence is not a regular file")
        if info.st_size > maximum:
            raise ValueError("evidence exceeds its bound")
        with path.open("rb") as stream:
            data = stream.read(maximum + 1)
        if len(data) > maximum or len(data) != info.st_size:
            raise ValueError("evidence exceeds its bound or changed size")
        return data
    except OSError as error:
        raise ValueError("evidence file is unavailable") from error


def _verify_reader_identity(root, revision, hashes):
    """Compare executing files and a locally available Git commit to external pins.

    This is an audit-time consistency check, not proof of remote publication,
    process execution or preregistration. The caller must establish the origin
    of the revision and keep the checkout and Git object store quiescent.
    """
    if not _hex(revision, 40) or not _source_map(hashes) or set(hashes) != READER_PATHS:
        raise ValueError("reader identity is incomplete")
    for path in sorted(READER_PATHS):
        try:
            local = _read_regular(Path(root) / path, 2 * 1024 * 1024)
            prefix = ["git", "--no-replace-objects", "-C", str(root), "cat-file"]
            size = subprocess.run(prefix + ["-s", revision + ":" + path], check=True, timeout=10,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
            if not size.isdigit() or int(size) != len(local):
                raise ValueError("reader commit size differs")
            committed = subprocess.run(prefix + ["blob", revision + ":" + path], check=True, timeout=10,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            if len(committed) != len(local) or _digest(local) != hashes[path] or _digest(committed) != hashes[path]:
                raise ValueError("reader source digest differs")
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ValueError("reader checkout or committed identity differs") from error


def _ordinary_directory(path):
    try:
        info = Path(path).lstat()
    except OSError as error:
        raise ValueError("evidence directory is unavailable") from error
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise ValueError("evidence directory is not ordinary")


def audit_directory(attempt, case_file, *, expected):
    """Audit a complete, quiescent production attempt with bounded reads.

    The caller must independently obtain and bind ``expected`` and ensure the
    files stay quiescent. Directory checks and hashes do not protect against a
    concurrent writer, authenticate the custodian/process, or prove that model
    execution or reported publication chronology occurred. This accepts only
    the fixed 50,257-token study; smaller fabricated cases use lower-level APIs.
    """
    attempt, run = Path(attempt), Path(attempt) / "run"
    _ordinary_directory(attempt)
    _ordinary_directory(run)
    ids = [f"copy-pairs-{group:02d}-{variant}" for group in range(1, 9) for variant in range(4)]
    run_names = {identity + ".f32" for identity in ids} | {"manifest.json", "result.json", "attempt.jsonl"}
    attempt_names = {"run", "pre-run.json", "execution-context.json", "supervisor.json", "stdout.bin", "stderr.bin"}
    if {item.name for item in run.iterdir()} != run_names or {item.name for item in attempt.iterdir()} != attempt_names:
        raise ValueError("evidence directory inventory differs")
    case_raw = _read_regular(case_file, 1024 * 1024)
    manifest_raw = _read_regular(run / "manifest.json", 2 * 1024 * 1024)
    result_raw = _read_regular(run / "result.json", 16 * 1024 * 1024)
    pre_raw = _read_regular(attempt / "pre-run.json", 65536)
    context_raw = _read_regular(attempt / "execution-context.json", 65536)
    receipt_raw = _read_regular(attempt / "supervisor.json", 65536)
    journal_raw = _read_regular(run / "attempt.jsonl", 8 * 1024 * 1024)
    bound = verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected)
    _verify_reader_identity(Path(__file__).resolve().parents[1], expected["reference_revision"],
                            expected["reference_source_sha256"])
    receipt = bound["receipt"]
    for stream in ("stdout", "stderr"):
        raw = _read_regular(attempt / (stream + ".bin"), 4 * 1024 * 1024)
        if len(raw) != receipt[stream + "_bytes"] or _digest(raw) != receipt[stream + "_sha256"]:
            raise ValueError("process receipt stream differs")
    journal_events = audit_journal(bound["result"], journal_raw)
    inventory = receipt["run_sha256"]

    def read_case(identity):
        raw = _read_regular(run / (identity + ".f32"), 5_240_520)
        if len(raw) != 5_240_520 or _digest(raw) != inventory[identity + ".f32"]:
            raise ValueError("case file length or receipt digest differs")
        return raw

    replay = replay_result(bound["cases"], bound["result"], read_case, verify_metadata=True)
    if {item.name for item in run.iterdir()} != run_names or {item.name for item in attempt.iterdir()} != attempt_names:
        raise ValueError("evidence directory inventory changed")
    return {"schema": "copying-head-pairs-dose-independent-replay-v1", "status": "saved_evidence_consistent",
            "source_revision": expected["source_revision"], "source_tree": expected["source_tree"],
            "case_sha256": expected["case_sha256"], "manifest_sha256": _digest(manifest_raw),
            "result_sha256": _digest(result_raw), "receipt_sha256": expected["receipt_sha256"],
            "execution_context_sha256": expected["execution_context_sha256"],
            "audit_reference_revision": expected["reference_revision"],
            "audit_reference_source_sha256": expected["reference_source_sha256"],
            "journal_events": journal_events, "replay": replay,
            "limitations": ["The supplied receipt, source identities, timestamps and custodian reports are not authenticated.",
                "Saved-data consistency does not prove model execution, process-tree lifetime, or actual chronology.",
                "Audit-time reader identity does not establish publication or preregistration before the run.",
                "The public, nonblind cases do not establish unknown-history or pretraining independence.",
                "The directory must remain quiescent; these checks are not protection against a concurrent mutator."]}
