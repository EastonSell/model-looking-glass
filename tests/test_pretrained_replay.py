"""Contract checks for independently replaying saved pretrained smoke evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from looking_glass.pretrained_replay import ReplayError, compare_reports, replay_to_directory


def example_report():
    values = [f"{index + 1:064x}" for index in range(18)]
    logits = ["d" * 64, "d" * 64]
    modes = {}
    for mode in ("direct", "gradient", "id"):
        events = []
        for index in range(18):
            layer = (0, 2, 5)[(index % 9) // 3]
            kind = ("attn", "mlp", "block")[index % 3]
            path = f"transformer.h.{layer}" + ("" if kind == "block" else f".{kind}")
            events.append({"metadata": {"schema_version": "1.0",
                                        "identity": {"run_id": f"first-study-{mode}",
                                                     "job_id": None, "request_id": None, "sample_id": None},
                                        "generation": {"stage": "unknown", "generation_step": None,
                                                       "cache_position": None, "cache_length_before": None,
                                                       "cache_length_after": None, "use_cache": None},
                                        "sequence": index, "forward_index": index // 9,
                                        "invocation": 0, "module_path": path,
                                        "module_kind": kind, "layer_index": layer,
                                        "output_path": [] if kind == "mlp" else [0],
                                        "shape": [1, 5, 768], "dtype": "torch.float32",
                                        "device": "cpu", "capture_mode": mode,
                                        "batch_positions": [0], "token_positions": [0, 1],
                                        "batch_axis": 0, "token_axis": 1,
                                        "saved_shape": None if mode == "id" else [1, 2, 768],
                                        "saved_dtype": None if mode == "id" else "torch.float32",
                                        "saved_device": None if mode == "id" else "cpu"},
                           "value_sha256": None if mode == "id" else values[index]})
        modes[mode] = {"logits_equal_baseline": True, "logit_sha256": list(logits),
                       "metadata_valid": True,
                       "values_equal_reference": None if mode == "id" else True,
                       "event_count": 18, "tensor_bytes": 0 if mode == "id" else 110592,
                       "events": events}
    return {
        "protocol_version": "pretrained-trace-smoke-v1",
        "config": {"tokens": [464, 6766, 318, 4171, 13], "seed": 0, "dtype": "float32",
                   "token_sha256": hashlib.sha256(struct.pack("<5q", 464, 6766, 318, 4171, 13)).hexdigest(),
                   "token_hash_format": "contiguous C-order little-endian int64",
                   "input_shape": [1, 5], "attention_mask": [1] * 5,
                   "tokenization": "none; fixed token IDs", "model_mode": "eval",
                   "attention_implementation": "eager", "use_cache": False,
                   "grad_mode": "inference_mode", "forwards_per_cell": 2,
                   "layers": [0, 2, 5], "target_order": ["attn", "mlp", "block"],
                   "batch_positions": [0], "token_positions": [0, 1], "max_events": 18,
                   "max_trace_bytes": 131072, "capture_device": "cpu",
                   "tensor_hash_format": "contiguous C-order little-endian float32"},
        "model": {"model_id": "distilbert/distilgpt2",
                  "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                  "tokenizer_id": "distilbert/distilgpt2",
                  "tokenizer_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                  "model_class": "GPT2LMHeadModel", "tokenizer_class": "GPT2TokenizerFast",
                  "load_device": "cpu", "execution_device": "cpu", "dtype": "float32",
                  "attention_implementation": "eager", "use_cache": False,
                  "parameter_count": 81912576, "torch_version": "2.8", "transformers_version": "4.57.3",
                  "weights_sha256": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
                  "asset_sha256": {
                      "config.json": "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a",
                      "generation_config.json": "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805",
                      "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
                      "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
                      "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
                      "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
                      "model.safetensors": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"}},
        "source_hashes": {"git_commit": "f" * 40, "clean": True, "scope": "executed project files only",
                          "files": {"docs/PRETRAINED_SMOKE_V1.md": "b" * 64,
                                    "looking_glass/pretrained_smoke.py": "c" * 64,
                                    "looking_glass/__init__.py": "a" * 64,
                                    "looking_glass/pretrained.py": "a" * 64,
                                    "looking_glass/model_ownership.py": "a" * 64,
                                    "looking_glass/probes.py": "a" * 64,
                                    "looking_glass/trace_schema.py": "a" * 64,
                                    "looking_glass/transformer_trace.py": "a" * 64}},
        "environment": {"device": "cpu", "python": "3.11.15", "platform": "Windows",
                        "python_executable_path_sha256": "a" * 64,
                        "python_prefix_path_sha256": "b" * 64, "python_base_prefix_path_sha256": "c" * 64,
                        "package_origin_path_sha256": {name: "d" * 64 for name in
                            ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy")},
                        "machine": "AMD64", "processor": "", "torch": "2.8",
                        "transformers": "4.57.3", "tokenizers": "0.1", "safetensors": "0.1",
                        "huggingface_hub": "0.1", "numpy": "2.4.6", "cuda_build": "12.8",
                        "cudnn_version": None, "gpu": None, "torch_intraop_threads": 1,
                        "torch_interop_threads": 1, "deterministic_algorithms": False,
                        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                        "cudnn_deterministic": False, "cudnn_benchmark": False,
                        "cublas_workspace_config": None, "cuda_visible_devices_policy": "unset",
                        "torch_build_config_sha256": "a" * 64, "torch_cpu_capability": "AVX2",
                        "omp_num_threads": None, "mkl_num_threads": None},
        "checks": {"passed": True, "baseline_logit_sha256": list(logits), "baseline_repeat_equal": True,
                   "baseline_logit_shape": [1, 5, 50257], "baseline_logit_dtype": "torch.float32",
                   "reference_logit_sha256": list(logits), "reference_logits_equal": True,
                   "reference_value_sha256": values, "direct_gradient_equal": True,
                   "modes": modes},
        "observations": {"elapsed_ns": 100, "study_id": "first-study"},
    }


def independent_report(reference):
    replay = copy.deepcopy(reference)
    replay["observations"]["study_id"] = "new-study"
    for mode, record in replay["checks"]["modes"].items():
        for event in record["events"]:
            event["metadata"]["identity"]["run_id"] = f"new-study-{mode}"
    return replay


def example_prepared():
    report = example_report()
    return {"protocol_version": report["protocol_version"], "config": report["config"],
            "source_hashes": report["source_hashes"], "device": "cpu", "study_id": "new-study",
            "expected_model_id": report["model"]["model_id"],
            "expected_model_revision": report["model"]["model_revision"],
            "expected_asset_sha256": report["model"]["asset_sha256"]}


class ReplayComparisonTests(unittest.TestCase):
    def test_matching_checks_are_reproduced_without_matching_timing(self):
        reference = example_report()
        replay = independent_report(reference)
        replay["observations"]["elapsed_ns"] = 999
        comparison = compare_reports(reference, replay)
        self.assertEqual(comparison["status"], "reproduced")
        self.assertEqual(comparison["differences"], [])

    def test_new_unique_session_ids_do_not_change_replay_identity(self):
        reference = example_report()
        replay = independent_report(reference)
        self.assertEqual(compare_reports(reference, replay)["status"], "reproduced")

    def test_the_same_session_ids_cannot_count_as_a_replay(self):
        reference = example_report()
        with self.assertRaises(ReplayError):
            compare_reports(reference, copy.deepcopy(reference))

    def test_session_ids_must_be_distinct_between_modes(self):
        reference = example_report()
        for record in reference["checks"]["modes"].values():
            for event in record["events"]:
                event["metadata"]["identity"]["run_id"] = "reused-session"
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_other_identity_fields_are_not_normalized(self):
        reference = example_report()
        replay = independent_report(reference)
        for event in replay["checks"]["modes"]["direct"]["events"]:
            event["metadata"]["identity"]["job_id"] = "different-job"
        self.assertEqual(compare_reports(reference, replay)["status"], "different")

    def test_environment_or_source_difference_is_not_a_numerical_failure(self):
        reference = example_report()
        replay = independent_report(reference)
        replay["environment"]["torch"] = "2.9"
        replay["model"]["torch_version"] = "2.9"
        replay["source_hashes"]["files"]["looking_glass/pretrained_smoke.py"] = "0" * 64
        comparison = compare_reports(reference, replay)
        self.assertEqual(comparison["status"], "incompatible")
        self.assertIn("environment", comparison["differences"])
        self.assertIn("source_hashes", comparison["differences"])

    def test_changed_event_value_is_reported_as_a_numerical_difference(self):
        reference = example_report()
        replay = independent_report(reference)
        replay["checks"]["reference_value_sha256"][0] = "f" * 64
        for mode in ("direct", "gradient"):
            replay["checks"]["modes"][mode]["events"][0]["value_sha256"] = "f" * 64
        comparison = compare_reports(reference, replay)
        self.assertEqual(comparison["status"], "different")
        self.assertIn("checks", comparison["differences"])

    def test_two_failed_reports_do_not_count_as_reproduced(self):
        reference = example_report()
        reference["checks"]["passed"] = False
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_bare_passing_flag_is_not_complete_smoke_evidence(self):
        reference = example_report()
        reference["checks"] = {"passed": True}
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_an_unrelated_protocol_is_rejected(self):
        reference = example_report()
        reference["protocol_version"] = "unrelated"
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_token_digest_encoding_is_required(self):
        reference = example_report()
        del reference["config"]["token_hash_format"]
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_missing_identity_is_not_reproducible_evidence(self):
        reference = example_report()
        del reference["source_hashes"]
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_equal_flags_cannot_hide_inconsistent_logits(self):
        reference = example_report()
        reference["checks"]["modes"]["direct"]["logit_sha256"][0] = "0" * 64
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_repeated_baseline_hashes_must_agree(self):
        reference = example_report()
        reference["checks"]["baseline_logit_sha256"][1] = "e" * 64
        reference["checks"]["reference_logit_sha256"][1] = "e" * 64
        for record in reference["checks"]["modes"].values():
            record["logit_sha256"][1] = "e" * 64
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_logits_hashes_need_the_prescribed_shape_and_dtype(self):
        reference = example_report()
        reference["checks"]["baseline_logit_shape"] = [1, 50257, 5]
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_event_coverage_cannot_be_faked_with_a_pass_flag(self):
        reference = example_report()
        reference["checks"]["modes"]["direct"]["events"][0]["metadata"]["invocation"] = 7
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_wrong_tensor_selector_is_rejected(self):
        reference = example_report()
        reference["checks"]["modes"]["direct"]["events"][0]["metadata"]["output_path"] = []
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_unknown_tensor_axes_cannot_justify_token_coordinates(self):
        reference = example_report()
        reference["checks"]["modes"]["direct"]["events"][0]["metadata"]["token_axis"] = None
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_cached_generation_metadata_contradicts_the_fixed_forward(self):
        reference = example_report()
        generation = reference["checks"]["modes"]["direct"]["events"][0]["metadata"]["generation"]
        generation["stage"] = "decode"
        generation["use_cache"] = True
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_value_equality_flag_cannot_hide_a_wrong_reference_hash(self):
        reference = example_report()
        reference["checks"]["modes"]["gradient"]["events"][0]["value_sha256"] = "f" * 64
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_missing_runtime_version_or_dirty_source_is_rejected(self):
        for mutation in ("version", "source"):
            with self.subTest(mutation=mutation):
                reference = example_report()
                if mutation == "version":
                    del reference["environment"]["transformers"]
                else:
                    reference["source_hashes"]["clean"] = False
                with self.assertRaises(ReplayError):
                    compare_reports(reference, independent_report(example_report()))

    def test_missing_backend_identity_or_unexpected_cpu_gpu_is_rejected(self):
        for mutation in ("visible", "determinism", "benchmark", "cpu_gpu"):
            with self.subTest(mutation=mutation):
                reference = example_report()
                if mutation == "visible":
                    del reference["environment"]["cuda_visible_devices_policy"]
                elif mutation == "determinism":
                    reference["environment"]["cudnn_deterministic"] = 1
                elif mutation == "benchmark":
                    del reference["environment"]["cudnn_benchmark"]
                else:
                    reference["environment"]["gpu"] = {"uuid": "GPU-invalid"}
                with self.assertRaises(ReplayError):
                    compare_reports(reference, independent_report(example_report()))

    def test_cuda_identity_requires_single_device_mapping_and_capacity(self):
        reference = example_report()
        reference["environment"]["device"] = "cuda"
        reference["environment"]["gpu"] = {
            "name": "GPU model", "driver_version": "123.45", "reported_total_mib": 8192,
            "logical_device": "cuda:0", "mapping": "single physical and single visible NVIDIA GPU"}
        reference["environment"]["cudnn_version"] = 90000
        reference["model"]["execution_device"] = "cuda:0"
        for record in reference["checks"]["modes"].values():
            for event in record["events"]:
                event["metadata"]["device"] = "cuda:0"
        self.assertEqual(compare_reports(reference, independent_report(reference))["status"], "reproduced")
        for key, invalid in (("name", ""), ("driver_version", ""),
                             ("reported_total_mib", True), ("reported_total_mib", 0),
                             ("logical_device", "cuda:1"), ("mapping", "unknown"),
                             ("uuid", "GPU-private")):
            with self.subTest(key=key, invalid=invalid):
                malformed = copy.deepcopy(reference)
                malformed["environment"]["gpu"][key] = invalid
                with self.assertRaises(ReplayError):
                    compare_reports(malformed, independent_report(reference))

    def test_raw_gpu_identifiers_are_not_accepted_in_records(self):
        reference = example_report()
        reference["environment"]["cuda_visible_devices"] = "GPU-private"
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_cpu_build_and_thread_environment_are_required(self):
        for field in ("torch_build_config_sha256", "torch_cpu_capability", "omp_num_threads", "mkl_num_threads"):
            with self.subTest(field=field):
                reference = example_report()
                del reference["environment"][field]
                with self.assertRaises(ReplayError):
                    compare_reports(reference, independent_report(example_report()))
        for field, invalid in (("torch_build_config_sha256", "not-a-digest"),
                               ("torch_cpu_capability", ""), ("omp_num_threads", 1),
                               ("mkl_num_threads", False)):
            with self.subTest(field=field, invalid=invalid):
                reference = example_report()
                reference["environment"][field] = invalid
                with self.assertRaises(ReplayError):
                    compare_reports(reference, independent_report(example_report()))

    def test_interpreter_and_package_origin_fingerprints_are_required(self):
        for field in ("python_executable_path_sha256", "python_prefix_path_sha256",
                      "python_base_prefix_path_sha256", "package_origin_path_sha256"):
            with self.subTest(field=field):
                reference = example_report()
                del reference["environment"][field]
                with self.assertRaises(ReplayError):
                    compare_reports(reference, independent_report(example_report()))
        for mutation in ("invalid_python", "missing_package", "invalid_package"):
            with self.subTest(mutation=mutation):
                reference = example_report()
                if mutation == "invalid_python":
                    reference["environment"]["python_executable_path_sha256"] = "not-a-digest"
                elif mutation == "missing_package":
                    del reference["environment"]["package_origin_path_sha256"]["torch"]
                else:
                    reference["environment"]["package_origin_path_sha256"]["torch"] = "not-a-digest"
                with self.assertRaises(ReplayError):
                    compare_reports(reference, independent_report(example_report()))

    def test_disabled_device_visibility_policy_is_supported(self):
        reference = example_report()
        reference["environment"]["cuda_visible_devices_policy"] = "disabled"
        self.assertEqual(compare_reports(reference, independent_report(reference))["status"], "reproduced")

    def test_wrong_checkpoint_hash_is_rejected_even_if_both_records_match(self):
        reference = example_report()
        reference["model"]["asset_sha256"]["model.safetensors"] = "0" * 64
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_model_execution_device_must_match_the_environment(self):
        reference = example_report()
        reference["model"]["execution_device"] = "cuda:0"
        with self.assertRaises(ReplayError):
            compare_reports(reference, independent_report(example_report()))

    def test_tokenizer_and_named_weight_identity_must_match_the_pinned_assets(self):
        for field in ("tokenizer_revision", "weights_sha256"):
            with self.subTest(field=field):
                reference = example_report()
                reference["model"][field] = "0" * len(reference["model"][field])
                with self.assertRaises(ReplayError):
                    compare_reports(reference, independent_report(example_report()))


class ReplayExecutionTests(unittest.TestCase):
    def test_checker_source_is_verified_against_its_committed_blob(self):
        import looking_glass.pretrained_replay as replay_module

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "looking_glass"
            package.mkdir()
            source = package / "pretrained_replay.py"
            source.write_bytes(b"committed source\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "add", "looking_glass/pretrained_replay.py"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Replay Test",
                            "-c", "user.email=replay-test@example.invalid", "commit", "-qm", "fixture"],
                           check=True, capture_output=True)
            with patch.object(replay_module, "__file__", str(source)):
                identity = replay_module._replay_source_identity()
                self.assertEqual(identity["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                source.write_bytes(b"modified source\n")
                with self.assertRaisesRegex(ReplayError, "committed source"):
                    replay_module._replay_source_identity()

    def test_new_cpu_replay_saves_evidence_and_excludes_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.json"
            reference.write_text(json.dumps(example_report()), encoding="utf-8")
            observed = independent_report(example_report())
            observed["observations"]["elapsed_ns"] = 1000
            calls = []

            def run_smoke(*, device, cache_dir, lease_id, _diagnostics, _on_prepared):
                calls.append((device, cache_dir, lease_id))
                _on_prepared(example_prepared())
                return observed

            source = {"git_commit": "f" * 40, "sha256": "1" * 64}
            with patch("looking_glass.pretrained_replay._replay_source_identity", return_value=source):
                result = replay_to_directory(reference, root / "replay", cache_dir=root / "cache",
                                             run_smoke=run_smoke)
            self.assertEqual(calls, [("cpu", root / "cache", None)])
            self.assertEqual(result["comparison"]["status"], "reproduced")
            saved = json.loads((root / "replay" / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["comparison"]["status"], "reproduced")
            self.assertEqual(saved["replay"]["checks"], observed["checks"])
            self.assertEqual(saved["replay_source"], source)
            self.assertTrue((root / "replay" / "prepared.json").is_file())
            with self.assertRaises(FileExistsError):
                replay_to_directory(reference, root / "replay", cache_dir=root / "cache",
                                    run_smoke=run_smoke)
            self.assertEqual(len(calls), 1)

    def test_runner_failure_is_retained_without_a_success_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.json"
            reference.write_text(json.dumps(example_report()), encoding="utf-8")

            def fail(*, _diagnostics, _on_prepared, **kwargs):
                _on_prepared(example_prepared())
                _diagnostics.update({"phase": "model_checks", "memory": [123]})
                raise RuntimeError("injected replay failure")

            with patch("looking_glass.pretrained_replay._replay_source_identity",
                       return_value={"git_commit": "f" * 40, "sha256": "1" * 64}):
                with self.assertRaisesRegex(RuntimeError, "injected replay failure"):
                    replay_to_directory(reference, root / "replay", cache_dir=root, run_smoke=fail)
            failure = json.loads((root / "replay" / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertEqual(failure["reference_sha256"], hashlib.sha256(reference.read_bytes()).hexdigest())
            self.assertEqual(failure["replay_source"]["git_commit"], "f" * 40)
            self.assertEqual(failure["device"], "cpu")
            self.assertEqual(failure["runner_diagnostics"]["phase"], "model_checks")
            self.assertEqual(failure["runner_diagnostics"]["memory"], [123])
            self.assertTrue((root / "replay" / "started.json").is_file())
            self.assertTrue((root / "replay" / "identity.json").is_file())
            self.assertTrue((root / "replay" / "prepared.json").is_file())
            self.assertFalse((root / "replay" / "results.json").exists())

    def test_failure_after_success_publication_does_not_add_a_failure_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.json"
            reference.write_text(json.dumps(example_report()), encoding="utf-8")
            original_unlink = Path.unlink

            def fail_cleanup(path, *args, **kwargs):
                if path.name == "results.json.partial":
                    raise OSError("injected cleanup failure")
                return original_unlink(path, *args, **kwargs)

            source = {"git_commit": "f" * 40, "sha256": "1" * 64}
            def run_smoke(*, _on_prepared, **kwargs):
                _on_prepared(example_prepared())
                return independent_report(example_report())
            with patch("looking_glass.pretrained_replay._replay_source_identity", return_value=source), \
                 patch.object(Path, "unlink", new=fail_cleanup):
                with self.assertRaisesRegex(OSError, "injected cleanup failure"):
                    replay_to_directory(reference, root / "replay", cache_dir=root,
                                        run_smoke=run_smoke)
            self.assertTrue((root / "replay" / "results.json").is_file())
            self.assertFalse((root / "replay" / "failure.json").exists())

    def test_runner_must_publish_the_matching_prepared_manifest(self):
        for mutation in ("missing", "wrong_asset", "source_changed"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                reference = root / "reference.json"
                reference.write_text(json.dumps(example_report()), encoding="utf-8")

                def run_smoke(*, _on_prepared, **kwargs):
                    if mutation != "missing":
                        manifest = example_prepared()
                        if mutation == "wrong_asset":
                            manifest["expected_asset_sha256"]["model.safetensors"] = "0" * 64
                        else:
                            manifest["source_hashes"]["files"]["looking_glass/pretrained.py"] = "0" * 64
                        _on_prepared(manifest)
                    return independent_report(example_report())

                source = {"git_commit": "f" * 40, "sha256": "1" * 64}
                with patch("looking_glass.pretrained_replay._replay_source_identity", return_value=source):
                    with self.assertRaises(ReplayError):
                        replay_to_directory(reference, root / "replay", cache_dir=root, run_smoke=run_smoke)
                self.assertTrue((root / "replay" / "failure.json").exists())
                self.assertFalse((root / "replay" / "results.json").exists())

    def test_cuda_requires_an_explicit_coordinated_lease_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "lease"):
                replay_to_directory(root / "not-read.json", root / "replay", cache_dir=root,
                                    device="cuda", run_smoke=lambda **kwargs: None)
            self.assertFalse((root / "replay").exists())

    def test_command_failure_saves_a_record_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            completed = subprocess.run([sys.executable, "-m", "looking_glass.pretrained_replay",
                                        "--reference", str(root / "missing.json"),
                                        "--output", str(root / "replay"),
                                        "--cache-dir", str(root)], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            failure = json.loads((root / "replay" / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["error_type"], "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
