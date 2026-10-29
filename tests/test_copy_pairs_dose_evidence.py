"""Fabricated process-record bindings; no model or checkpoint is used."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _evidence():
    try:
        from looking_glass import copy_pairs_dose_evidence
    except ImportError:
        pytest.fail("bounded pairwise evidence binding is not implemented")
    return copy_pairs_dose_evidence


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _binding_fixture():
    source, tree, protocol_rev, case_rev, clarification_rev, reference_rev = [c * 40 for c in "abcdef"]
    source_hashes = {"looking_glass/a.py": "1" * 64}
    reference_hashes = {path: "2" * 64 for path in (
        "looking_glass/copy_pairs_dose_audit.py", "looking_glass/copy_pairs_dose_evidence.py",
        "scripts/audit_copy_pairs_dose_v1.py")}
    dependencies = {"python": "3.11.15", "torch": "2.8.0+cu128", "numpy": "2.4.6",
                    "transformers": "4.57.3", "gpt2_source_sha256": "b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80",
                    "conv1d_source_sha256": "e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86", "byteorder": "little", "machine": "AMD64", "system": "Windows",
                    "gpt2_module": "transformers.models.gpt2.modeling_gpt2", "conv1d_module": "transformers.pytorch_utils"}
    assets = {name: "a" * 64 for name in ("config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")}
    assets["model.safetensors"] = "7" * 64
    numeric_probe = {"input_preserved": True, "half_min_normal_bits": 0x00400000, "numpy_cast_preserved": True}
    cases = [{"case_id": f"copy-pairs-{group + 1:02d}-{variant}", "group": group + 1,
              "variant": variant, "target_id": 3} for group in range(8) for variant in range(4)]
    case = {"schema": "distilgpt2-copy-head-pairs-dose-cases-v1", "cases": cases, "tokenizer_sha256": assets["tokenizer.json"],
            "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa"}
    case_raw = _bytes(case)
    private = {"case_sha256": _sha(case_raw),
        "natural": {"reporter": "natural custodian", "status": "no_protected_rows", "scope": "synthetic", "reported_utc": "2026-10-29T00:00:00Z"},
        "injection": {"reporter": "injection custodian", "status": "no_overlap", "scope": "synthetic", "reported_utc": "2026-10-29T00:00:00Z", "target_boundary": "proceed_no_direct_conflict"}}
    exposure = {"v13": "3" * 64, "v15": "4" * 64}
    manifest_exposure = {"v13": "3" * 64, "reservation": "4" * 64,
                         "reservation_version": "distilgpt2-development-exposures-v15"}
    pre = {"schema": "copying-head-pairs-dose-launch-v1", "source_revision": source, "source_tree": tree,
           "protocol_revision": protocol_rev, "protocol_sha256": "5" * 64,
           "case_revision": case_rev, "case_sha256": _sha(case_raw),
           "clarification_revision": clarification_rev, "clarification_sha256": "6" * 64,
           "independent_audit": "The saved-data reader will be published and bound to the result.",
           "exposure_inventories": exposure, "private_split_status": private,
           "device": "cpu", "model_id": "distilbert/distilgpt2",
           "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa", "weights_sha256": "7" * 64,
           "environment": {"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": None, "ATEN_CPU_CAPABILITY": None},
           "intended_runtime": {"python": "3.11.15", "torch": "2.8.0+cu128", "numpy": "2.4.6", "transformers": "4.57.3",
               "dtype": "float32", "attention_implementation": "eager", "use_cache": False,
               "intraop_threads": 1, "deterministic_algorithms": False, "float32_matmul_precision": "highest", "cpu_rng_seed": 20261029},
           "expected_forwards": 576, "expected_raw_payload_bytes": 167696640}
    context = {"schema": "copying-head-pairs-dose-execution-context-v1", "source_revision": source,
               "case_sha256": _sha(case_raw), "exposure_inventories": exposure, "private_split_status": private}
    manifest = {"schema": "distilgpt2-copy-head-pairs-dose-run-v1",
                "source": {"revision": source, "tree": tree, "sha256": source_hashes},
                "dependencies": dependencies, "initial_numeric_probe": numeric_probe,
                "cases": cases, "cases_sha256": _sha(case_raw), "protocol_sha256": "5" * 64,
                "numeric_clarification": {"revision": clarification_rev, "sha256": "6" * 64},
                "exposure_inventories": manifest_exposure, "weights_sha256": "7" * 64,
                "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                "raw_payload_bytes": 167696640,
                "intended_runtime": {"device": "cpu", "dtype": "float32", "model_eval": True,
                    "attention_implementation": "eager", "use_cache": False, "intraop_threads": 1,
                    "interop_threads": 1, "deterministic_algorithms": False, "mkldnn_enabled": True,
                    "float32_matmul_precision": "highest", "cpu_rng_seed": 20261029},
                "cpu_backend_environment_at_torch_import": {"MKL_CBWR": {"present": False, "value": None},
                                                             "ATEN_CPU_CAPABILITY": {"present": False, "value": None}}}
    manifest_raw = _bytes(manifest)
    result = {"schema": "distilgpt2-copy-head-pairs-dose-result-v1", "status": "complete",
              "manifest_sha256": _sha(manifest_raw), "submitted_forwards": 576,
              "model_call_returns_observed": 576, "returned_forwards": 576,
              "raw_payload_bytes": 167696640, "cases": [{}] * 576,
              "checkpoint": {"weights_sha256": "7" * 64, "asset_sha256": assets,
                             "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                             "model_class": "GPT2LMHeadModel", "tokenizer_class": "GPT2TokenizerFast",
                             "tokenizer_id": "distilbert/distilgpt2", "tokenizer_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                             "device": "cpu", "dtype": "float32", "attention_implementation": "eager", "use_cache": False,
                             "parameter_count": 81912576, "torch_version": "2.8.0+cu128", "transformers_version": "4.57.3"},
              "runtime": {**manifest["intended_runtime"], "cuda_visibility": "-1", "cuda_initialized": False,
                          "cpu_autocast": False, "guard_blocked_calls": [],
                          "numeric_probe": numeric_probe,
                          "cpu_backend_environment": manifest["cpu_backend_environment_at_torch_import"]},
              "model_state_sha256_before": "8" * 64, "model_state_sha256_after": "8" * 64,
              "model_buffers_sha256_before": "9" * 64, "model_buffers_sha256_after": "9" * 64}
    result_raw, pre_raw, context_raw, journal_raw = _bytes(result), _bytes(pre), _bytes(context), b"{}\n"
    inventory = {case["case_id"] + ".f32": "0" * 64 for case in cases}
    inventory.update({"manifest.json": _sha(manifest_raw), "result.json": _sha(result_raw), "attempt.jsonl": _sha(journal_raw)})
    receipt = {"schema": "copying-head-pairs-dose-process-observation-v1", "source_revision": source,
               "pre_run_sha256": _sha(pre_raw), "execution_context_sha256": _sha(context_raw),
               "process_exit_code": 0, "interrupted": False, "failure_file_present": False, "launch_error_type": None,
               "stdout_bytes": 0, "stdout_sha256": _sha(b""), "stderr_bytes": 0, "stderr_sha256": _sha(b""),
               "run_sha256": inventory}
    receipt_raw = _bytes(receipt)
    expected = {"source_revision": source, "source_tree": tree, "source_sha256": source_hashes,
                "protocol_revision": protocol_rev, "protocol_sha256": "5" * 64,
                "case_revision": case_rev, "case_sha256": _sha(case_raw),
                "clarification_revision": clarification_rev, "clarification_sha256": "6" * 64,
                "reference_revision": reference_rev, "reference_source_sha256": reference_hashes,
                "dependencies": dependencies,
                "exposure_inventories": exposure, "manifest_exposure_inventories": manifest_exposure,
                "weights_sha256": "7" * 64, "asset_sha256": assets,
                "pre_run_sha256": _sha(pre_raw),
                "execution_context_sha256": _sha(context_raw), "receipt_sha256": _sha(receipt_raw)}
    return case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected


def test_binding_requires_external_source_receipt_and_private_scope():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    parsed = evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected)
    assert len(parsed["cases"]) == 32
    assert parsed["receipt"]["process_exit_code"] == 0
    bad = copy.deepcopy(expected)
    bad["source_sha256"]["looking_glass/a.py"] = "f" * 64
    with pytest.raises(ValueError, match="source"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, bad)
    bad_receipt = json.loads(receipt_raw)
    bad_receipt["process_exit_code"] = True
    altered = _bytes(bad_receipt)
    bad = {**expected, "receipt_sha256": _sha(altered)}
    with pytest.raises(ValueError, match="receipt"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, altered, journal_raw, bad)


def test_later_reader_is_not_claimed_as_preregistered():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    pre = json.loads(pre_raw)
    assert "reference_revision" not in pre and "reference_source_sha256" not in pre
    evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected)
    pre["reference_revision"] = "0" * 40
    altered = _bytes(pre)
    receipt = json.loads(receipt_raw)
    receipt["pre_run_sha256"] = _sha(altered)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "pre_run_sha256": _sha(altered), "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="launch|reference"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, altered, context_raw, changed_receipt, journal_raw, pins)


def test_reader_revision_and_executed_sources_match_independent_pins(tmp_path):
    evidence = _evidence()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    files = ("looking_glass/copy_pairs_dose_audit.py", "looking_glass/copy_pairs_dose_evidence.py",
             "scripts/audit_copy_pairs_dose_v1.py")
    hashes = {}
    for path in files:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = ("synthetic " + path + "\n").encode()
        target.write_bytes(data)
        hashes[path] = _sha(data)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", *files], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fabricated reader"], check=True)
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    evidence._verify_reader_identity(tmp_path, revision, hashes)
    (tmp_path / files[0]).write_bytes(b"changed after publication\n")
    with pytest.raises(ValueError, match="reader"):
        evidence._verify_reader_identity(tmp_path, revision, hashes)


def test_binding_rejects_rehashed_unsafe_runtime_and_private_status():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    result = json.loads(result_raw)
    result["runtime"]["cuda_initialized"] = True
    changed_result = _bytes(result)
    receipt = json.loads(receipt_raw)
    receipt["run_sha256"]["result.json"] = _sha(changed_result)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="runtime"):
        evidence.verify_bindings(case_raw, manifest_raw, changed_result, pre_raw, context_raw, changed_receipt, journal_raw, pins)
    context = json.loads(context_raw)
    context["private_split_status"]["injection"]["status"] = "unknown"
    changed_context = _bytes(context)
    receipt = json.loads(receipt_raw)
    receipt["execution_context_sha256"] = _sha(changed_context)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "execution_context_sha256": _sha(changed_context), "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="private custody"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, changed_context, changed_receipt, journal_raw, pins)


def test_binding_rejects_rehashed_dependency_and_numeric_probe_changes():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    bad = copy.deepcopy(expected)
    bad["dependencies"]["torch"] = "other"
    with pytest.raises(ValueError, match="dependenc"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, bad)
    result = json.loads(result_raw)
    result["runtime"]["numeric_probe"]["numpy_cast_preserved"] = False
    changed_result = _bytes(result)
    receipt = json.loads(receipt_raw)
    receipt["run_sha256"]["result.json"] = _sha(changed_result)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="runtime|numeric"):
        evidence.verify_bindings(case_raw, manifest_raw, changed_result, pre_raw, context_raw, changed_receipt, journal_raw, pins)


def test_binding_rejects_rehashed_launch_runtime_drift():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    pre = json.loads(pre_raw)
    pre["intended_runtime"]["python"] = "3.10.0"
    changed_pre = _bytes(pre)
    receipt = json.loads(receipt_raw)
    receipt["pre_run_sha256"] = _sha(changed_pre)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "pre_run_sha256": _sha(changed_pre), "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="runtime"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, changed_pre, context_raw, changed_receipt, journal_raw, pins)


def test_binding_rejects_rehashed_tokenizer_asset_drift():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    result = json.loads(result_raw)
    result["checkpoint"]["asset_sha256"]["tokenizer.json"] = "b" * 64
    changed_result = _bytes(result)
    receipt = json.loads(receipt_raw)
    receipt["run_sha256"]["result.json"] = _sha(changed_result)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="checkpoint"):
        evidence.verify_bindings(case_raw, manifest_raw, changed_result, pre_raw, context_raw, changed_receipt, journal_raw, pins)


@pytest.mark.parametrize("failure", ["missing", "RuntimeError"])
def test_receipt_rejects_missing_or_nonnull_launch_error(failure):
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    receipt = json.loads(receipt_raw)
    if failure == "missing":
        del receipt["launch_error_type"]
    else:
        receipt["launch_error_type"] = failure
    changed = _bytes(receipt)
    pins = {**expected, "receipt_sha256": _sha(changed)}
    with pytest.raises(ValueError, match="receipt"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, changed, journal_raw, pins)


@pytest.mark.parametrize(("field", "value"), [("tokenizer_id", "other/model"), ("tokenizer_revision", "b" * 40),
    ("device", "cuda"), ("dtype", "float16"), ("attention_implementation", "sdpa"), ("use_cache", True),
    ("parameter_count", 1), ("torch_version", "other"), ("transformers_version", "other")])
def test_checkpoint_rejects_rehashed_contradictory_metadata(field, value):
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    result = json.loads(result_raw)
    result["checkpoint"][field] = value
    altered = _bytes(result)
    receipt = json.loads(receipt_raw)
    receipt["run_sha256"]["result.json"] = _sha(altered)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="checkpoint"):
        evidence.verify_bindings(case_raw, manifest_raw, altered, pre_raw, context_raw, changed_receipt, journal_raw, pins)


def test_binding_rejects_a_consistently_omitted_dependency():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    manifest, result, pre, receipt = map(json.loads, (manifest_raw, result_raw, pre_raw, receipt_raw))
    del manifest["dependencies"]["torch"]
    del pre["intended_runtime"]["torch"]
    del result["checkpoint"]["torch_version"]
    changed_manifest, changed_pre = _bytes(manifest), _bytes(pre)
    result["manifest_sha256"] = _sha(changed_manifest)
    changed_result = _bytes(result)
    receipt["pre_run_sha256"] = _sha(changed_pre)
    receipt["run_sha256"]["manifest.json"] = _sha(changed_manifest)
    receipt["run_sha256"]["result.json"] = _sha(changed_result)
    changed_receipt = _bytes(receipt)
    pins = copy.deepcopy(expected)
    del pins["dependencies"]["torch"]
    pins.update(pre_run_sha256=_sha(changed_pre), receipt_sha256=_sha(changed_receipt))
    with pytest.raises(ValueError, match="dependenc"):
        evidence.verify_bindings(case_raw, changed_manifest, changed_result, changed_pre, context_raw, changed_receipt, journal_raw, pins)


@pytest.mark.parametrize("missing", ["tokenizer.json", "gpt2_source_sha256", "conv1d_source_sha256", "gpt2_module", "conv1d_module"])
def test_binding_rejects_consistently_omitted_asset_or_source(missing):
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    manifest, result, receipt = map(json.loads, (manifest_raw, result_raw, receipt_raw))
    pins = copy.deepcopy(expected)
    if missing.endswith(".json"):
        del pins["asset_sha256"][missing]
        del result["checkpoint"]["asset_sha256"][missing]
    else:
        del pins["dependencies"][missing]
        del manifest["dependencies"][missing]
    changed_manifest = _bytes(manifest)
    result["manifest_sha256"] = _sha(changed_manifest)
    changed_result = _bytes(result)
    receipt["run_sha256"]["manifest.json"] = _sha(changed_manifest)
    receipt["run_sha256"]["result.json"] = _sha(changed_result)
    changed_receipt = _bytes(receipt)
    pins["receipt_sha256"] = _sha(changed_receipt)
    with pytest.raises(ValueError):
        evidence.verify_bindings(case_raw, changed_manifest, changed_result, pre_raw, context_raw, changed_receipt, journal_raw, pins)


def test_binding_rejects_rehashed_case_tokenizer_mismatch():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    case, manifest, result, pre, context, receipt = map(json.loads, (case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw))
    case["tokenizer_sha256"] = "b" * 64
    changed_case = _bytes(case)
    case_sha = _sha(changed_case)
    for document in (pre, context):
        document["case_sha256"] = case_sha
        document["private_split_status"]["case_sha256"] = case_sha
    manifest["cases_sha256"] = case_sha
    changed_pre, changed_context, changed_manifest = _bytes(pre), _bytes(context), _bytes(manifest)
    result["manifest_sha256"] = _sha(changed_manifest)
    changed_result = _bytes(result)
    receipt.update(pre_run_sha256=_sha(changed_pre), execution_context_sha256=_sha(changed_context))
    receipt["run_sha256"].update({"manifest.json": _sha(changed_manifest), "result.json": _sha(changed_result)})
    changed_receipt = _bytes(receipt)
    pins = {**expected, "case_sha256": case_sha, "pre_run_sha256": _sha(changed_pre),
            "execution_context_sha256": _sha(changed_context), "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="case|tokenizer"):
        evidence.verify_bindings(changed_case, changed_manifest, changed_result, changed_pre, changed_context, changed_receipt, journal_raw, pins)


@pytest.mark.parametrize("key", ["MKL_CBWR", "ATEN_CPU_CAPABILITY"])
def test_binding_rejects_omitted_launch_environment_observation(key):
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    pre, receipt = map(json.loads, (pre_raw, receipt_raw))
    del pre["environment"][key]
    changed_pre = _bytes(pre)
    receipt["pre_run_sha256"] = _sha(changed_pre)
    changed_receipt = _bytes(receipt)
    pins = {**expected, "pre_run_sha256": _sha(changed_pre), "receipt_sha256": _sha(changed_receipt)}
    with pytest.raises(ValueError, match="runtime"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, changed_pre, context_raw, changed_receipt, journal_raw, pins)


@pytest.mark.parametrize(("key", "value"), [("python", "3.10.0"), ("torch", "2.7.0"), ("numpy", "2.0.0"),
    ("transformers", "4.0.0"), ("byteorder", "big"), ("gpt2_source_sha256", "c" * 64), ("conv1d_source_sha256", "d" * 64)])
def test_binding_rejects_mutually_consistent_nonfrozen_dependencies(key, value):
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    manifest, result, pre, receipt = map(json.loads, (manifest_raw, result_raw, pre_raw, receipt_raw))
    pins = copy.deepcopy(expected)
    pins["dependencies"][key] = value
    manifest["dependencies"][key] = value
    if key in ("python", "torch", "numpy", "transformers"):
        pre["intended_runtime"][key] = value
    if key in ("torch", "transformers"):
        result["checkpoint"][key + "_version"] = value
    changed_manifest, changed_pre = _bytes(manifest), _bytes(pre)
    result["manifest_sha256"] = _sha(changed_manifest)
    changed_result = _bytes(result)
    receipt["pre_run_sha256"] = _sha(changed_pre)
    receipt["run_sha256"].update({"manifest.json": _sha(changed_manifest), "result.json": _sha(changed_result)})
    changed_receipt = _bytes(receipt)
    pins.update(pre_run_sha256=_sha(changed_pre), receipt_sha256=_sha(changed_receipt))
    with pytest.raises(ValueError, match="dependenc"):
        evidence.verify_bindings(case_raw, changed_manifest, changed_result, changed_pre, context_raw, changed_receipt, journal_raw, pins)


def test_fabricated_receipt_cannot_be_mistaken_for_a_process_observation():
    evidence = _evidence()
    case_raw, manifest_raw, result_raw, pre_raw, context_raw, receipt_raw, journal_raw, expected = _binding_fixture()
    receipt = json.loads(receipt_raw)
    receipt["fabricated_only"] = True
    changed = _bytes(receipt)
    pins = {**expected, "receipt_sha256": _sha(changed)}
    with pytest.raises(ValueError, match="fabricated"):
        evidence.verify_bindings(case_raw, manifest_raw, result_raw, pre_raw, context_raw, changed, journal_raw, pins)


def test_directory_audit_rejects_missing_and_extra_files_before_loading_payloads(tmp_path):
    evidence = _evidence()
    run = tmp_path / "run"
    run.mkdir()
    (run / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory"):
        evidence.audit_directory(tmp_path, tmp_path / "cases.json", expected={})


def test_bounded_regular_evidence_reader_rejects_oversized_or_directory(tmp_path):
    evidence = _evidence()
    source = tmp_path / "source"
    source.write_bytes(b"12345")
    assert evidence._read_regular(source, 5) == b"12345"
    with pytest.raises(ValueError, match="bound"):
        evidence._read_regular(source, 4)
    with pytest.raises(ValueError, match="regular"):
        evidence._read_regular(tmp_path, 100)


def test_cli_leaves_no_report_for_incomplete_input(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "audit_copy_pairs_dose_v1.py"
    attempt = tmp_path / "attempt"
    (attempt / "run").mkdir(parents=True)
    case_file, pins, output = tmp_path / "cases.json", tmp_path / "pins.json", tmp_path / "report.json"
    case_file.write_text("{}", encoding="utf-8")
    pins.write_text("{}", encoding="utf-8")
    child = subprocess.run([sys.executable, "-B", str(script), "--attempt", str(attempt),
                            "--cases", str(case_file), "--pins", str(pins), "--output", str(output)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert child.returncode != 0
    assert b"inventory" in child.stderr
    assert not output.exists()
