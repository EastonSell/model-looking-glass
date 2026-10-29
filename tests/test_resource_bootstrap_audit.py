"""CPU-only checks of independent bootstrap profile reconstruction."""

import hashlib
import io
import json
import subprocess
import tempfile
import traceback
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_resource_profile as fixtures
from looking_glass import resource_profile as profiles
from looking_glass.memory_budget import CudaMemorySnapshot
from looking_glass import resource_bootstrap_audit as audit
from looking_glass.resource_bootstrap_audit import BootstrapAuditError, audit_profile_bytes


def example_artifacts(*, matching_source=True):
    source = b'{"schema":"synthetic-test-source"}'
    source_id = hashlib.sha256(source).hexdigest() if matching_source else "d" * 64
    key = fixtures.workload(profiles, source_sha256=source_id)
    first = fixtures.sample(profiles, key=key)
    second = fixtures.sample(profiles, key=key, sample_id="sample-2", seed=29,
        before=CudaMemorySnapshot(6000, 10000, 100, 300),
        after=CudaMemorySnapshot(6100, 10000, 300, 500),
        peak_allocated_bytes=600, peak_reserved_bytes=900)
    profile = fixtures.measured_profile(profiles, key, (first, second), margin_bytes=50,
                                        margin_bps=1000, source_manifest=source)
    return profile.to_json().encode("utf-8"), profiles.encode_evidence(key, (first, second)), source, profile.profile_id


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def make_synthetic_run(root):
    """A fully synthetic five-sample fixture; it is not GPU evidence."""
    uuid = "GPU-00000000-0000-0000-0000-000000000001"
    protocol_sha = "1" * 64
    files = [
        {"path": "docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md", "size": 10, "sha256": protocol_sha},
        {"path": "looking_glass/__init__.py", "size": 5, "sha256": "6" * 64},
        {"path": "looking_glass/gpu_bootstrap.py", "size": 8, "sha256": "8" * 64},
        {"path": "looking_glass/gpu_service.py", "size": 20, "sha256": "2" * 64},
        {"path": "looking_glass/live_profile.py", "size": 15, "sha256": "7" * 64},
        {"path": "looking_glass/pretrained.py", "size": 30, "sha256": "3" * 64},
    ]
    versions = {"torch": "2.8.0+cu128", "transformers": "4.57.3", "numpy": "2.4.6",
                "huggingface-hub": "1.0", "safetensors": "1.0", "tokenizers": "1.0", "regex": "1.0"}
    source = {"schema": "resource-bootstrap-source-v1", "git_commit": "a" * 40, "git_tree": "b" * 40,
              "repository_files": files, "python_executable": {"size": 100, "sha256": "4" * 64},
              "distributions": [{"name": name, "version": version,
                                 "files": [{"path": name + "/code.py", "size": 1, "sha256": "5" * 64}]}
                                for name, version in sorted(versions.items())]}
    environment = {"CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "PYTORCH_CUDA_ALLOC_CONF": "backend:native", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "PYTHONHASHSEED": "17", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "PYTORCH_ALLOC_CONF": None, "PYTORCH_NO_CUDA_MEMORY_CACHING": None,
                   "CUDA_LAUNCH_BLOCKING": None, "PYTHONPATH": None, "PYTHONHOME": None,
                   "PATH": "C:/Windows/System32"}
    runtime = {"schema": "resource-bootstrap-runtime-v1", "device": "cuda:0", "device_uuid": uuid,
        "device_name": "Synthetic GPU", "device_total_bytes": 8 * audit._GIB, "compute_capability": [7, 5],
        "python_version": "3.11.15", "torch_version": versions["torch"],
        "transformers_version": versions["transformers"], "cuda_runtime_version": "12.8",
        "driver_version": "570.0", "cudnn_version": 91002, "cuda_visible_devices": uuid,
        "cuda_device_order": "PCI_BUS_ID", "allocator_backend": "native", "allocator_fraction": "0.20",
        "allocator_effective_limit_bytes": 8 * audit._GIB // 5, "model_dtype": "torch.float32",
        "compute_dtype": "torch.float32", "attention_backend": "eager", "model_mode": "eval",
        "grad_mode": "inference", "autocast": "disabled", "use_cache": False, "compiled": False,
        "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_enabled": True, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "deterministic_algorithms": True, "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "input_limit_bytes": 64 * 1024, "output_tensor_limit_bytes": 512 * 1024,
        "response_wire_limit_bytes": 4 * audit._MIB,
        "execution_path": "looking_glass.resident_worker.ResidentWorker.execute",
        "python_import_paths": [r"C:\Python311\Lib", r"C:\private\repo", r"C:\private\venv\Lib\site-packages"],
        "environment": environment}
    source_bytes, runtime_bytes = canonical(source), canonical(runtime)
    write = lambda name, data: (root / name).write_bytes(data)
    started = {"schema": "resource-bootstrap-start-v1", "run_id": "f" * 32,
               "started_utc": "2026-10-28T00:00:00Z", "protocol_sha256": protocol_sha,
               "git_commit": "a" * 40, "git_tree": "b" * 40, "requested": audit._REQUESTED}
    write("started.json", canonical(started))
    write("owner.json", canonical({"schema": "resource-bootstrap-owner-v1", "parent_run_id": "f" * 32,
                                   "supervisor_run_id": "e" * 32, "pid": 1234}))
    write("source.json", source_bytes)
    write("runtime.json", runtime_bytes)
    assets = {"schema": "resource-bootstrap-assets-v1", "model_id": "distilbert/distilgpt2",
              "revision": audit._REVISION,
              "assets": [{"name": name, "size": size, "sha256": digest}
                         for name, (size, digest) in sorted(audit._ASSETS.items())]}
    write("assets.json", canonical(assets))
    for name, manifest in (("load-case.json", audit._LOAD_CASE), ("forward-case.json", audit._FORWARD_CASE),
                           ("warmup.json", audit._WARMUP), ("capture.json", audit._CAPTURE)):
        write(name, canonical(manifest))
    total = 8 * audit._GIB
    checks = {"schema": "resource-bootstrap-checks-v1", "device_total_bytes": total,
              "allocator_requested_fraction": "0.20", "allocator_effective_limit_bytes": int(0.20 * total),
              "allocator_readback": repr(int(0.20 * total) / total), "model_aliases_disposed": True,
              "device_synchronized_after_disposal": True, "load_lease_disposition": "held_until_exit",
              "owner": {"run_id": "e" * 32, "pid": 1234},
              "import_policy": {"private_cache_prefix": "C:/private/synthetic-cache",
                  "parent_created_empty": True, "runtime_prefix_matches": True, "cache_empty_after_imports": True,
                  "dont_write_bytecode": True, "hash_seed_set_before_exec": True, "module_origins_verified": True,
                  "sys_path_verified": True, "safe_path_enabled": True, "site_import_disabled": True,
                  "user_site_enabled": False},
              "forwards": [{"sequence": number, "output_shape": [1, 2, 50257], "output_dtype": "torch.float32",
                            "output_tensor_bytes": 402056, "finite": True, "framed_response_bytes": 1024 * 1024,
                            "offload_complete": True, "disposal_complete": True} for number in range(2, 6)]}
    write("checks.json", canonical(checks))
    keys = {}
    for stage in ("load", "cold", "warm"):
        keys[stage] = fixtures.workload(profiles,
            checkpoint_sha256=audit._ASSETS["model.safetensors"][1], checkpoint_revision=audit._REVISION,
            model_config_sha256=audit._ASSETS["config.json"][1], source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            runtime_settings_sha256=hashlib.sha256(runtime_bytes).hexdigest(), torch_version=versions["torch"],
            transformers_version=versions["transformers"], device_total_bytes=8 * audit._GIB,
            device_name=runtime["device_name"], warmup_protocol_sha256=hashlib.sha256(canonical(audit._WARMUP)).hexdigest(),
            capture_policy_sha256=hashlib.sha256(canonical(audit._CAPTURE)).hexdigest(),
            operation="model_load" if stage == "load" else "forward", warmup_state="warm" if stage == "warm" else "cold",
            batch_size=0 if stage == "load" else 1, sequence_length=0 if stage == "load" else 2,
            input_dtypes=() if stage == "load" else (("input_ids", "torch.int64"),),
            output_limit_bytes=0 if stage == "load" else 512 * 1024)
    by_stage = {"load": [], "cold": [], "warm": []}
    rows, admissions = [], []
    for sequence, (stage, seed) in enumerate(audit._STAGES, start=1):
        case = audit._LOAD_CASE if stage == "load" else audit._FORWARD_CASE
        sample = fixtures.sample(profiles, key=keys[stage], sample_id=f"sample-{sequence}", seed=seed,
            case_sha256=hashlib.sha256(canonical(case)).hexdigest(),
            before=CudaMemorySnapshot(6 * audit._GIB, 8 * audit._GIB, 100, 300),
            after=CudaMemorySnapshot(6 * audit._GIB - 300, 8 * audit._GIB, 300, 400),
            peak_allocated_bytes=400, peak_reserved_bytes=500)
        by_stage[stage].append(sample)
        rows.append({"schema": "resource-bootstrap-observation-v1", "sequence": sequence,
                     "stage": stage, "key": asdict(keys[stage]), "sample": asdict(sample)})
        job = 0 if stage == "load" else 512 * audit._MIB
        admissions.append({"schema": "resource-bootstrap-admission-v1", "sequence": sequence, "stage": stage,
            "telemetry": {"device_free_bytes": 6 * audit._GIB, "device_total_bytes": total,
                          "process_allocated_bytes": 100, "process_reserved_bytes": 300},
            "reserve_bytes": max(audit._GIB, (15 * total + 99) // 100),
            "committed_incremental_bytes": 2 * audit._GIB + job,
            "load_commitment_bytes": 2 * audit._GIB, "job_commitment_bytes": job,
            "owner_headroom_check": True, "collector_post_sync_check_attested": True,
            "job_lease_released_after": None if stage == "load" else True})
    write("observations.jsonl", b"".join(canonical(row) + b"\n" for row in rows))
    write("admissions.jsonl", b"".join(canonical(row) + b"\n" for row in admissions))
    pins = {}
    for stage, samples in by_stage.items():
        measurements = tuple(samples)
        profile = fixtures.measured_profile(profiles, keys[stage], measurements, source_manifest=source_bytes,
            margin_bytes=(512 if stage == "load" else 256) * audit._MIB, margin_bps=2500)
        write(stage + ".profile.json", profile.to_json().encode("utf-8"))
        write(stage + ".evidence.json", profiles.encode_evidence(keys[stage], measurements))
        pins[stage] = profile.profile_id
    publish_terminal(root)
    return pins


def publish_terminal(root, **changes):
    records = [{"path": path.name, "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in sorted(root.iterdir()) if path.name != "terminal.json"]
    terminal = {"schema": "resource-bootstrap-terminal-v1", "run_id": "f" * 32,
                "ended_utc": "2026-10-28T00:01:00Z", "status": "success", "child_exit_code": 0,
                "clean_teardown": True, "failure_stage": None, "reason": None,
                "supervisor": {"run_id": "e" * 32, "pid": 1234, "exit_code": 0, "clean": True,
                               "timed_out": False}, "inventory": records}
    terminal.update(changes)
    if "supervisor" not in changes:
        terminal["supervisor"]["exit_code"] = terminal["child_exit_code"]
        terminal["supervisor"]["clean"] = terminal["clean_teardown"]
    (root / "terminal.json").write_bytes(canonical(terminal))


class BootstrapProfileAuditTests(unittest.TestCase):
    def test_reconstructs_each_raw_delta_and_max_plus_allowances_and_margin(self):
        document, evidence, source, pin = example_artifacts()
        result = audit_profile_bytes(document, evidence, source, pin)
        self.assertEqual(result["samples"][0]["observed_incremental_proxy_bytes"], 300)
        self.assertEqual(result["samples"][1]["device_free_endpoint_drop_bytes"], -100)
        self.assertEqual(result["samples"][1]["observed_incremental_proxy_bytes"], 600)
        self.assertEqual(result["input_allowance_bytes"], 64)
        self.assertEqual(result["output_allowance_bytes"], 100)
        self.assertEqual(result["basis_bytes"], 764)
        self.assertEqual(result["percentage_margin_bytes"], 77)
        self.assertEqual(result["estimated_incremental_bytes"], 891)
        self.assertNotIn("GPU-", json.dumps(result))

    def test_changed_observations_source_or_independent_pin_are_rejected(self):
        document, evidence, source, pin = example_artifacts()
        for field in ("evidence", "source", "pin"):
            with self.subTest(field=field):
                changed = {"evidence": evidence, "source_manifest": source, "approved_id": pin}
                if field == "evidence":
                    changed["evidence"] += b" "
                elif field == "source":
                    changed["source_manifest"] += b" "
                else:
                    changed["approved_id"] = "0" * 64
                with self.assertRaises(BootstrapAuditError):
                    audit_profile_bytes(document, **changed)

    def test_bootstrap_requires_source_key_to_match_manifest(self):
        document, evidence, source, pin = example_artifacts(matching_source=False)
        with self.assertRaisesRegex(BootstrapAuditError, "source identities"):
            audit_profile_bytes(document, evidence, source, pin)


class BootstrapRunAuditTests(unittest.TestCase):
    def test_cli_hides_private_exception_details(self):
        secret = "C:/private-home/GPU-11111111-2222-3333-4444-555555555555"
        output = io.StringIO()
        with patch.object(audit, "audit_run", side_effect=OSError(secret)), redirect_stdout(output):
            self.assertEqual(audit.main(["some-private-directory"]), 1)
        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn("GPU-11111111", output.getvalue())

    def test_local_git_check_covers_every_tracked_project_file_and_detects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "looking_glass").mkdir()
            (root / "docs").mkdir()
            for name, content in (("looking_glass/example.py", b"example\n"),
                                  ("docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md", b"protocol\n")):
                (root / name).write_bytes(content)
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "add", "looking_glass", "docs"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Audit Test", "-c",
                            "user.email=audit-test@example.invalid", "commit", "-qm", "fixture"],
                           check=True, capture_output=True)
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            tree = subprocess.check_output(["git", "-C", str(root), "show", "-s", "--format=%T", "HEAD"],
                                           text=True).strip()
            entries = []
            for name in ("docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md", "looking_glass/example.py"):
                content = (root / name).read_bytes()
                entries.append({"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            source = {"git_commit": commit, "git_tree": tree, "repository_files": entries}
            audit._verify_repository(source, root)
            omitted = dict(source, repository_files=entries[:1])
            with self.assertRaises(BootstrapAuditError):
                audit._verify_repository(omitted, root)
            (root / "looking_glass/example.py").write_bytes(b"changed\n")
            with self.assertRaises(BootstrapAuditError):
                audit._verify_repository(source, root)

    def test_local_environment_checks_metadata_record_coverage_and_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "Lib/site-packages"
            info = site / "example-1.0.dist-info"
            package = site / "example"
            info.mkdir(parents=True)
            package.mkdir()
            (package / "__init__.py").write_bytes(b"package\n")
            (info / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: example\nVersion: 1.0\n\n")
            (info / "RECORD").write_bytes(
                b"example/__init__.py,,\nexample-1.0.dist-info/METADATA,,\nexample-1.0.dist-info/RECORD,,\n")
            entries = []
            for path in sorted(site.rglob("*")):
                if path.is_file():
                    content = path.read_bytes()
                    entries.append({"path": path.relative_to(root).as_posix(), "size": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()})
            entries.sort(key=lambda entry: entry["path"])
            source = {"distributions": [{"name": "example", "version": "1.0", "files": entries}]}
            audit._verify_environment(source, root, site)
            omitted = {"distributions": [{"name": "example", "version": "1.0",
                                          "files": [e for e in entries if not e["path"].endswith("__init__.py")]}]}
            with self.assertRaises(BootstrapAuditError):
                audit._verify_environment(omitted, root, site)
            (package / "__init__.py").write_bytes(b"changed\n")
            with self.assertRaises(BootstrapAuditError):
                audit._verify_environment(source, root, site)

    def test_local_environment_rejects_unlisted_distribution_and_wrong_site(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "Lib/site-packages"
            info = site / "example-1.0.dist-info"
            info.mkdir(parents=True)
            (info / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: example\nVersion: 1.0\n\n")
            (info / "RECORD").write_bytes(b"example-1.0.dist-info/METADATA,,\nexample-1.0.dist-info/RECORD,,\n")
            entries = []
            for path in sorted(info.iterdir()):
                content = path.read_bytes()
                entries.append({"path": path.relative_to(root).as_posix(), "size": len(content),
                                "sha256": hashlib.sha256(content).hexdigest()})
            source = {"distributions": [{"name": "example", "version": "1.0", "files": entries}]}
            audit._verify_environment(source, root, site)
            extra = site / "hidden-1.0.dist-info"
            extra.mkdir()
            (extra / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: hidden\nVersion: 1.0\n\n")
            with self.assertRaises(BootstrapAuditError):
                audit._verify_environment(source, root, site)
            extra.rename(root / "unselected-package")
            with self.assertRaises(BootstrapAuditError):
                audit._verify_environment(source, root, root)

    def test_runtime_requires_complete_import_and_environment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_synthetic_run(root)
            runtime = json.loads((root / "runtime.json").read_bytes())
            source = json.loads((root / "source.json").read_bytes())
            audit._check_runtime(runtime, source)
            for field in ("PYTHONPATH", "PYTHONHOME"):
                for mutation in ("missing", "set"):
                    with self.subTest(field=field, mutation=mutation):
                        altered = json.loads(json.dumps(runtime))
                        if mutation == "missing":
                            del altered["environment"][field]
                        else:
                            altered["environment"][field] = "untrusted"
                        with self.assertRaises(BootstrapAuditError):
                            audit._check_runtime(altered, source)
            for paths in ([], ["relative"], ["C:/trusted", "C:/trusted"], ["C:/trusted", None]):
                with self.subTest(paths=paths):
                    altered = dict(runtime, python_import_paths=paths)
                    with self.assertRaises(BootstrapAuditError):
                        audit._check_runtime(altered, source)
            for path_value in (None, "", 4):
                with self.subTest(path_value=path_value):
                    altered = json.loads(json.dumps(runtime))
                    altered["environment"]["PATH"] = path_value
                    with self.assertRaises(BootstrapAuditError):
                        audit._check_runtime(altered, source)

    def test_source_requires_sorted_normalized_complete_minimum_distribution_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_synthetic_run(root)
            source = json.loads((root / "source.json").read_bytes())
            started = json.loads((root / "started.json").read_bytes())
            audit._check_source(source, started)
            for kind in ("missing", "unnormalized", "duplicate", "unsorted", "empty_files", "missing_launcher"):
                with self.subTest(kind=kind):
                    altered = json.loads(json.dumps(source))
                    packages = altered["distributions"]
                    if kind == "missing_launcher":
                        altered['repository_files'] = [item for item in altered['repository_files']
                                                       if item['path'] != 'looking_glass/gpu_bootstrap.py']
                    elif kind == "missing":
                        packages[:] = [item for item in packages if item["name"] != "numpy"]
                    elif kind == "unnormalized":
                        packages[0]["name"] = "Bad_Name"
                    elif kind == "duplicate":
                        packages.insert(0, packages[0])
                    elif kind == "unsorted":
                        packages.reverse()
                    else:
                        packages[0]["files"] = []
                    with self.assertRaises(BootstrapAuditError):
                        audit._check_source(altered, started)

    def test_record_paths_reject_intermediate_link_before_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "site"
            (site / "alias").mkdir(parents=True)
            (site / "package").mkdir()
            (site / "package" / "module.py").write_bytes(b"data")
            real_is_symlink = Path.is_symlink
            def mocked_link(path):
                return path.name == "alias" or real_is_symlink(path)
            with patch.object(Path, "is_symlink", mocked_link):
                with self.assertRaises(BootstrapAuditError):
                    audit._record_member(root, site, "alias/../package/module.py")

    def test_embedded_null_in_manifest_path_is_rejected_cleanly(self):
        self.assertFalse(audit._safe_relative("site/private\x00file"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(BootstrapAuditError):
                audit._local_file(Path(directory), "site/private\x00file")

    def test_artifact_reparse_points_are_rejected_even_if_not_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_synthetic_run(root)
            real_lstat = Path.lstat
            def marked_reparse(path):
                info = real_lstat(path)
                if path.name == "capture.json":
                    return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
                return info
            with patch.object(Path, "lstat", marked_reparse):
                with self.assertRaises(BootstrapAuditError):
                    audit.audit_run(root, approved_pins=pins)

    def test_local_checkpoint_check_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"tiny synthetic checkpoint"
            (root / "weights.bin").write_bytes(content)
            with patch.object(audit, "_ASSETS", {"weights.bin": (len(content), hashlib.sha256(content).hexdigest())}):
                audit._verify_assets(root)
                (root / "weights.bin").write_bytes(b"altered")
                with self.assertRaises(BootstrapAuditError):
                    audit._verify_assets(root)

    def test_private_os_error_details_are_not_chained_into_audit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.json").write_bytes(b"{}")
            secret = "C:/private-home/GPU-11111111-2222-3333-4444-555555555555"
            with patch.object(Path, "read_bytes", side_effect=OSError(secret)):
                try:
                    audit._read(root, "input.json")
                except BootstrapAuditError as error:
                    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    self.assertNotIn(secret, formatted)
                    self.assertNotIn("GPU-11111111", formatted)
                else:
                    self.fail("missing audit error")

    def test_excessively_nested_and_unpaired_unicode_are_normalized_to_audit_errors(self):
        for raw in (b"[" * 1200 + b"0" + b"]" * 1200, b'"\\ud800"'):
            with self.subTest(raw=raw[:20]), self.assertRaises(BootstrapAuditError):
                audit._decode_canonical(raw)

    def test_synthetic_success_reconstructs_profiles_and_redacts_private_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_synthetic_run(root)
            result = audit.audit_run(root, approved_pins=pins)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["publication"], "terminal")
            self.assertTrue(result["pins_verified"])
            self.assertEqual(result["completed_observations"], 5)
            self.assertEqual(len(result["profiles"]["warm"]["samples"]), 3)
            self.assertEqual(result["profiles"]["cold"]["input_allowance_bytes"], 16)
            self.assertNotIn("GPU-", json.dumps(result))
            self.assertNotIn(str(root), json.dumps(result))
            unpinned = audit.audit_run(root)
            self.assertFalse(unpinned["pins_verified"])
            self.assertEqual(unpinned["profiles"]["warm"]["profile_id"], pins["warm"])

    def test_parent_can_validate_a_candidate_terminal_without_publishing_or_mutating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = make_synthetic_run(root)
            terminal = json.loads((root / "terminal.json").read_bytes())
            (root / "terminal.json").unlink()
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            result = audit.audit_candidate(root, terminal, approved_pins=pins)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["publication"], "candidate")
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})
            self.assertFalse((root / "terminal.json").exists())
            terminal["clean_teardown"] = False
            with self.assertRaises(BootstrapAuditError):
                audit.audit_candidate(root, terminal, approved_pins=pins)

    def test_candidate_cannot_replace_an_existing_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_synthetic_run(root)
            terminal = json.loads((root / "terminal.json").read_bytes())
            with self.assertRaises(BootstrapAuditError):
                audit.audit_candidate(root, terminal)

    def test_started_only_and_truncated_journal_remain_incomplete_and_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_synthetic_run(root)
            (root / "terminal.json").unlink()
            journal = root / "observations.jsonl"
            journal.write_bytes(journal.read_bytes() + b'{"partial":')
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            result = audit.audit_run(root)
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["trailing_partial_observation"])
            self.assertEqual(result["completed_observations"], 5)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_failure_terminal_does_not_promote_partial_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_synthetic_run(root)
            journal = root / "observations.jsonl"
            journal.write_bytes(journal.read_bytes() + b'{"partial":')
            publish_terminal(root, status="failed", child_exit_code=1, clean_teardown=False,
                             failure_stage="warm", reason="test failure")
            result = audit.audit_run(root)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["trailing_partial_observation"])
            self.assertFalse(result["pins_verified"])
            self.assertNotIn("profiles", result)

    def test_unreaped_owner_is_quarantined_without_a_fabricated_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_synthetic_run(root)
            publish_terminal(root, status="quarantined", child_exit_code=None, clean_teardown=False,
                             supervisor=None, failure_stage="unknown", reason="supervisor interrupted")
            self.assertEqual(audit.audit_run(root)["status"], "quarantined")
            publish_terminal(root, status="failed", child_exit_code=None, clean_teardown=False,
                             supervisor=None, failure_stage="unknown", reason="supervisor interrupted")
            with self.assertRaises(BootstrapAuditError):
                audit.audit_run(root)

    def test_clean_supervisor_and_child_owner_must_agree(self):
        for kind in ("run_id", "pid", "missing_owner", "checks_owner", "timeout"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = make_synthetic_run(root)
                if kind == "missing_owner":
                    (root / "owner.json").unlink()
                    publish_terminal(root)
                elif kind == "checks_owner":
                    path = root / "checks.json"
                    checks = json.loads(path.read_bytes())
                    checks["owner"]["pid"] += 1
                    path.write_bytes(canonical(checks))
                    publish_terminal(root)
                else:
                    terminal_path = root / "terminal.json"
                    terminal = json.loads(terminal_path.read_bytes())
                    if kind == "run_id":
                        terminal["supervisor"]["run_id"] = "d" * 32
                    elif kind == "pid":
                        terminal["supervisor"]["pid"] += 1
                    else:
                        terminal["supervisor"]["timed_out"] = True
                    terminal_path.write_bytes(canonical(terminal))
                with self.assertRaises(BootstrapAuditError):
                    audit.audit_run(root, approved_pins=pins)

    def test_success_rejects_truncated_or_extra_journal_lines_even_with_refreshed_inventory(self):
        for filename in ("observations.jsonl", "admissions.jsonl"):
            for kind in ("truncated", "extra"):
                with self.subTest(kind=kind, filename=filename), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    pins = make_synthetic_run(root)
                    journal = root / filename
                    original = journal.read_bytes()
                    journal.write_bytes(original[:-1] if kind == "truncated" else original + original.splitlines()[0] + b"\n")
                    publish_terminal(root)
                    with self.assertRaises(BootstrapAuditError):
                        audit.audit_run(root, approved_pins=pins)

    def test_success_rejects_bad_headroom_and_execution_attestations(self):
        for kind in ("free", "commitment", "readback", "wire", "empty_wire", "imports", "safe_path"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = make_synthetic_run(root)
                if kind in ("free", "commitment"):
                    journal = root / "admissions.jsonl"
                    rows = [json.loads(row) for row in journal.read_bytes().splitlines()]
                    if kind == "free":
                        rows[1]["telemetry"]["device_free_bytes"] = audit._GIB
                    else:
                        rows[1]["committed_incremental_bytes"] = 1
                    journal.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
                else:
                    path = root / "checks.json"
                    checks = json.loads(path.read_bytes())
                    if kind == "readback":
                        checks["allocator_readback"] = "0.1"
                    elif kind == "wire":
                        checks["forwards"][0]["framed_response_bytes"] = 4 * audit._MIB + 1
                    elif kind == "empty_wire":
                        checks["forwards"][0]["framed_response_bytes"] = 0
                    elif kind == "safe_path":
                        checks["import_policy"]["safe_path_enabled"] = False
                    else:
                        checks["import_policy"]["module_origins_verified"] = False
                    path.write_bytes(canonical(checks))
                publish_terminal(root)
                with self.assertRaises(BootstrapAuditError):
                    audit.audit_run(root, approved_pins=pins)

    def test_inventory_must_match_every_file_without_duplicates(self):
        for kind in ("extra_file", "missing_entry", "duplicate_entry", "missing_file"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = make_synthetic_run(root)
                terminal_path = root / "terminal.json"
                terminal = json.loads(terminal_path.read_bytes())
                if kind == "extra_file":
                    (root / "unlisted.partial").write_bytes(b"partial")
                elif kind == "missing_entry":
                    terminal["inventory"].pop()
                    terminal_path.write_bytes(canonical(terminal))
                elif kind == "duplicate_entry":
                    terminal["inventory"].append(terminal["inventory"][-1])
                    terminal_path.write_bytes(canonical(terminal))
                else:
                    (root / "source.json").unlink()
                with self.assertRaises(BootstrapAuditError):
                    audit.audit_run(root, approved_pins=pins)

    def test_success_rejects_raw_sample_or_runtime_tampering(self):
        for kind in ("sample", "runtime", "execution_path", "pin"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = make_synthetic_run(root)
                if kind == "sample":
                    journal = root / "observations.jsonl"
                    rows = [json.loads(row) for row in journal.read_bytes().splitlines()]
                    rows[1]["sample"]["peak_allocated_bytes"] += 1
                    journal.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
                    publish_terminal(root)
                elif kind in ("runtime", "execution_path"):
                    path = root / "runtime.json"
                    data = json.loads(path.read_bytes())
                    if kind == "runtime":
                        data["cudnn_deterministic"] = False
                    else:
                        data["execution_path"] = "direct_model_forward"
                    path.write_bytes(canonical(data))
                    publish_terminal(root)
                else:
                    pins["load"] = "0" * 64
                with self.assertRaises(BootstrapAuditError):
                    audit.audit_run(root, approved_pins=pins)

    def test_boolean_substitution_cannot_match_integer_manifest_or_observation_fields(self):
        for kind in ("case", "capture", "observation_key"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = make_synthetic_run(root)
                if kind == "observation_key":
                    path = root / "observations.jsonl"
                    rows = [json.loads(row) for row in path.read_bytes().splitlines()]
                    rows[1]["key"]["batch_size"] = True
                    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
                else:
                    path = root / ("forward-case.json" if kind == "case" else "capture.json")
                    data = json.loads(path.read_bytes())
                    if kind == "case":
                        data["shape"][0] = True
                    else:
                        data["max_gpu_trace_bytes"] = False
                    path.write_bytes(canonical(data))
                publish_terminal(root)
                with self.assertRaises(BootstrapAuditError):
                    audit.audit_run(root, approved_pins=pins)


if __name__ == "__main__":
    unittest.main()
