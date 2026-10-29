"""CPU-only bootstrap safety tests; no CUDA initialization is permitted."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass.supervisor import SupervisionResult


class BootstrapJournalTests(unittest.TestCase):
    def setUp(self):
        try:
            self.bootstrap = importlib.import_module("looking_glass.gpu_bootstrap")
        except ModuleNotFoundError as exc:
            if exc.name == "looking_glass.gpu_bootstrap":
                self.fail("the bootstrap journal has not been implemented")
            raise
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "run"

    def create(self):
        return self.bootstrap.BootstrapJournal.create(
            self.path, run_id="1" * 32, started_utc="2026-10-29T00:00:00+00:00",
            protocol_sha256="a" * 64, git_commit="b" * 40, git_tree="c" * 40,
        )

    def write_owner(self, journal, *, supervisor_run_id="2" * 32, pid=123):
        journal.write_document("owner.json", {"schema": "resource-bootstrap-owner-v1",
            "parent_run_id": "1" * 32, "supervisor_run_id": supervisor_run_id, "pid": pid})

    def test_start_is_nonoverwriting_canonical_and_records_fixed_caps(self):
        self.create()
        raw = (self.path / "started.json").read_bytes()
        start = json.loads(raw)
        self.assertEqual(raw, json.dumps(start, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8"))
        self.assertEqual(start["schema"], "resource-bootstrap-start-v1")
        self.assertEqual(start["run_id"], "1" * 32)
        self.assertEqual(start["requested"], {
            "load_commitment_bytes": 2147483648, "job_commitment_bytes": 536870912,
            "reserve_min_bytes": 1073741824, "reserve_percent": 15,
            "allocator_fraction": "0.20", "input_limit_bytes": 65536,
            "output_tensor_limit_bytes": 524288, "response_wire_limit_bytes": 4194304,
            "max_gpu_trace_bytes": 0, "max_host_trace_bytes": 0,
        })
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual((self.path / "started.json").read_bytes(), raw)

    def test_documents_are_canonical_nonoverwriting_and_reject_floats(self):
        journal = self.create()
        journal.write_document("checks.json", {"z": True, "label": "é", "a": [3, None]})
        path = self.path / "checks.json"
        expected = '{"a":[3,null],"label":"é","z":true}'.encode("utf-8")
        self.assertEqual(path.read_bytes(), expected)
        with self.assertRaises(FileExistsError):
            journal.write_document("checks.json", {"replaced": True})
        self.assertEqual(path.read_bytes(), expected)
        with self.assertRaises((TypeError, ValueError)):
            journal.write_document("runtime.json", {"fraction": 0.2})
        self.assertFalse((self.path / "runtime.json").exists())

    def test_journal_does_not_append_to_a_truncated_record(self):
        journal = self.create()
        journal.append_record("observations.jsonl", {"sequence": 1, "stage": "load"})
        path = self.path / "observations.jsonl"
        self.assertEqual(path.read_bytes(), b'{"sequence":1,"stage":"load"}\n')
        with path.open("ab") as stream:
            stream.write(b'{"sequence":2')
        damaged = path.read_bytes()
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            journal.append_record("observations.jsonl", {"sequence": 2, "stage": "cold"})
        self.assertEqual(path.read_bytes(), damaged)

    def test_unverified_owner_publishes_quarantine_with_complete_partial_inventory(self):
        journal = self.create()
        (self.path / "evidence.partial").write_bytes(b"partial")
        result = SupervisionResult("2" * 32, 123, 1, False, False, "unverified")
        terminal = journal.finish(result, failure_stage="load", reason="drain uncertain",
                                  ended_utc="2026-10-29T00:01:00+00:00")
        self.assertEqual(terminal["status"], "quarantined")
        self.assertFalse(terminal["clean_teardown"])
        self.assertEqual(terminal["child_exit_code"], 1)
        records = {entry["path"]: entry for entry in terminal["inventory"]}
        self.assertEqual(set(records), {"started.json", "evidence.partial"})
        self.assertEqual(records["evidence.partial"]["size"], 7)
        self.assertEqual(records["evidence.partial"]["sha256"], hashlib.sha256(b"partial").hexdigest())
        self.assertEqual(json.loads((self.path / "terminal.json").read_bytes()), terminal)
        with self.assertRaises(FileExistsError):
            journal.finish(result, failure_stage="load", reason="again")

    def test_clean_workload_failure_remains_failed(self):
        journal = self.create()
        self.write_owner(journal)
        result = SupervisionResult("2" * 32, 123, 17, True, False, "clean teardown")
        terminal = journal.finish(result, failure_stage="forward", reason="invalid output",
                                  ended_utc="2026-10-29T00:01:00+00:00")
        self.assertEqual(terminal["status"], "failed")
        self.assertTrue(terminal["clean_teardown"])
        self.assertEqual(terminal["child_exit_code"], 17)
        self.assertEqual(terminal["supervisor"], {"run_id": "2" * 32, "pid": 123,
                         "exit_code": 17, "clean": True, "timed_out": False})

    def test_clean_result_cannot_publish_without_matching_owner_record(self):
        journal = self.create()
        result = SupervisionResult("2" * 32, 123, 17, True, False, "clean teardown")
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            journal.finish(result, failure_stage="forward", reason="invalid output")
        self.assertFalse((self.path / "terminal.json").exists())
        self.write_owner(journal, supervisor_run_id="3" * 32)
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            journal.finish(result, failure_stage="forward", reason="invalid output")
        self.assertFalse((self.path / "terminal.json").exists())

    def test_unknown_postlaunch_outcome_remains_quarantined_without_inventing_pid(self):
        journal = self.create()
        self.write_owner(journal)
        terminal = journal.finish(None, owner_may_have_started=True, failure_stage="owner",
                                  reason="supervisor interrupted", ended_utc="2026-10-29T00:01:00+00:00")
        self.assertEqual(terminal["status"], "quarantined")
        self.assertIsNone(terminal["supervisor"])
        self.assertIsNone(terminal["child_exit_code"])
        self.assertFalse(terminal["clean_teardown"])

    def test_missing_result_defaults_to_quarantine_without_an_owner_record(self):
        journal = self.create()
        terminal = journal.finish(None, failure_stage="owner", reason="supervisor lost")
        self.assertEqual(terminal["status"], "quarantined")
        self.assertIsNone(terminal["supervisor"])
        self.assertIsNone(terminal["child_exit_code"])
        self.assertFalse(terminal["clean_teardown"])

    def test_confirmed_prelaunch_failure_can_be_recorded_as_failed(self):
        journal = self.create()
        terminal = journal.finish(None, owner_may_have_started=False,
                                  failure_stage="preflight", reason="checkpoint unavailable")
        self.assertEqual(terminal["status"], "failed")
        self.assertIsNone(terminal["supervisor"])
        self.assertFalse(terminal["clean_teardown"])

    def test_terminal_scratch_cleanup_failure_never_invalidates_visible_inventory(self):
        journal = self.create()
        result = SupervisionResult("2" * 32, 123, 1, False, False, "unverified")
        original_unlink = Path.unlink
        def reject_scratch(path, *args, **kwargs):
            if path.name.endswith(".partial"):
                raise PermissionError("simulated scratch cleanup failure")
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", autospec=True, side_effect=reject_scratch):
            try:
                journal.finish(result, failure_stage="load", reason="uncertain",
                               ended_utc="2026-10-29T00:01:00+00:00")
            except PermissionError:
                pass
        path = self.path / "terminal.json"
        if path.exists():
            terminal = json.loads(path.read_bytes())
            listed = {item["path"] for item in terminal["inventory"]}
            actual = {item.name for item in self.path.iterdir()} - {"terminal.json"}
            self.assertEqual(actual, listed)

    def test_inventory_refuses_windows_reparse_files(self):
        self.path.mkdir()
        target = self.path / "partial.bin"
        target.write_bytes(b"partial")
        original = Path.lstat
        info = original(target)
        redirected = SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        def lstat(path, *args, **kwargs):
            return redirected if path == target else original(path, *args, **kwargs)
        with patch.object(Path, "lstat", autospec=True, side_effect=lstat):
            with self.assertRaises(self.bootstrap.BootstrapError):
                self.bootstrap._inventory(self.path)

    def test_interruption_after_terminal_publication_cannot_replace_record(self):
        journal = self.create()
        result = SupervisionResult("2" * 32, 123, 1, False, False, "unverified")
        publish = self.bootstrap._write_terminal_new
        def interrupt_after_publish(path, data):
            publish(path, data)
            raise KeyboardInterrupt("simulated interruption after atomic publication")
        with patch.object(self.bootstrap, "_write_terminal_new", side_effect=interrupt_after_publish):
            with self.assertRaises(KeyboardInterrupt):
                journal.finish(result, failure_stage="load", reason="uncertain",
                               ended_utc="2026-10-29T00:01:00+00:00")
        original = (self.path / "terminal.json").read_bytes()
        terminal = json.loads(original)
        self.assertEqual(terminal["status"], "quarantined")
        self.assertEqual({item["path"] for item in terminal["inventory"]},
                         {item.name for item in self.path.iterdir()} - {"terminal.json"})
        with self.assertRaises(FileExistsError):
            journal.finish(result, failure_stage="load", reason="different")
        self.assertEqual((self.path / "terminal.json").read_bytes(), original)

    def test_clean_exit_cannot_publish_success_without_independent_candidate_audit(self):
        journal = self.create()
        self.write_owner(journal)
        result = SupervisionResult("2" * 32, 123, 0, True, False, "clean teardown")
        with self.assertRaises((ValueError, RuntimeError, self.bootstrap.BootstrapError)):
            journal.finish(result, ended_utc="2026-10-29T00:01:00+00:00")
        self.assertFalse((self.path / "terminal.json").exists())

    def test_complete_synthetic_candidate_is_audited_before_success_publication(self):
        from test_resource_bootstrap_audit import make_synthetic_run
        from looking_glass.resource_bootstrap_audit import audit_run
        self.path.mkdir()
        make_synthetic_run(self.path)
        (self.path / "terminal.json").unlink()
        started = json.loads((self.path / "started.json").read_bytes())
        journal = self.bootstrap.BootstrapJournal(self.path, started)
        result = SupervisionResult("e" * 32, 1234, 0, True, False, "synthetic clean")
        terminal = journal.finish(result, ended_utc="2026-10-28T00:01:00Z")
        self.assertEqual(terminal["status"], "success")
        report = audit_run(self.path)
        self.assertEqual(report["publication"], "terminal")
        self.assertEqual(report["completed_observations"], 5)
        self.assertFalse(report["pins_verified"])

    def test_changed_start_record_prevents_a_terminal_claim(self):
        journal = self.create()
        (self.path / "started.json").write_bytes(b'{"run_id":"different"}')
        result = SupervisionResult("2" * 32, 123, 1, False, False, "unverified")
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            journal.finish(result, failure_stage="load", reason="uncertain",
                           ended_utc="2026-10-29T00:01:00+00:00")
        self.assertFalse((self.path / "terminal.json").exists())

    def test_replaced_run_directory_is_not_used_for_further_writes(self):
        journal = self.create()
        self.path.rename(self.path.with_name("original"))
        self.path.mkdir()
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            journal.write_document("checks.json", {"ok": True})
        self.assertFalse((self.path / "checks.json").exists())

    def test_malformed_supervisor_result_cannot_assert_clean_teardown(self):
        journal = self.create()
        result = SupervisionResult("2" * 32, 123, 0, "true", False, "untrusted")
        with self.assertRaises(TypeError):
            journal.finish(result, ended_utc="2026-10-29T00:01:00+00:00")
        self.assertFalse((self.path / "terminal.json").exists())


class OwnerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.bootstrap = importlib.import_module("looking_glass.gpu_bootstrap")
        if not hasattr(self.bootstrap, "prepare_owner_launch"):
            self.fail("the supervised direct-child launcher has not been implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.private = self.root / "private"
        self.private.mkdir()
        self.environment_root = self.root / "environment"
        self.site = self.environment_root / "Lib" / "site-packages"
        self.site.mkdir(parents=True)
        self.gpu_uuid = "GPU-12345678-1234-5678-9abc-1234567890ab"

    def prepare(self, module="probe", args=()):
        inherited = dict(os.environ, SECRET_API_KEY="do-not-inherit", PYTHONPATH="untrusted",
                         CUDA_LAUNCH_BLOCKING="1", PYTORCH_NO_CUDA_MEMORY_CACHING="1",
                         PATH=str(self.root / "untrusted-bin"))
        def records(root):
            return [{"path": path.relative_to(root).as_posix(), "size": len(path.read_bytes()),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sorted(root.rglob("*"))
                    if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts]
        executable = Path(sys._base_executable).resolve().read_bytes()
        manifest = {"schema": "resource-bootstrap-source-v1", "git_commit": "1" * 40,
                    "git_tree": "2" * 40, "repository_files": records(self.project),
                    "python_executable": {"size": len(executable), "sha256": hashlib.sha256(executable).hexdigest()},
                    "distributions": [{"name": "fixture", "version": "1",
                                       "files": records(self.environment_root)}]}
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")
        return self.bootstrap.prepare_owner_launch(
            project_root=self.project, site_packages=self.site, environment_root=self.environment_root,
            private_parent=self.private, source_manifest=manifest_bytes, trusted_path=str(self.root),
            gpu_uuid=self.gpu_uuid, module=module, arguments=args, inherited_environment=inherited,
        )

    def test_launcher_uses_fresh_empty_cache_and_scrubs_inherited_secrets(self):
        spec = self.prepare()
        self.assertEqual(Path(spec.command[0]).resolve(), Path(sys._base_executable).resolve())
        self.assertIn("-B", spec.command)
        self.assertIn("-S", spec.command)
        self.assertIn("-P", spec.command)
        self.assertTrue(spec.pycache_prefix.is_absolute())
        self.assertEqual(list(spec.pycache_prefix.iterdir()), [])
        self.assertEqual(spec.environment["PYTHONPYCACHEPREFIX"], str(spec.pycache_prefix))
        self.assertEqual(spec.environment["CUDA_VISIBLE_DEVICES"], self.gpu_uuid)
        self.assertEqual(spec.environment["PYTHONHASHSEED"], "17")
        self.assertEqual(spec.environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(spec.environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(spec.environment["PATH"], str(self.root))
        for key in ("SECRET_API_KEY", "PYTHONPATH", "CUDA_LAUNCH_BLOCKING",
                    "PYTORCH_NO_CUDA_MEMORY_CACHING"):
            self.assertNotIn(key, spec.environment)

    def test_child_observes_real_preexec_controls_and_approved_search_path(self):
        (self.project / "probe.py").write_text(
            "import json,os,sys\n"
            "print(json.dumps({'prefix':sys.pycache_prefix,'no_site':sys.flags.no_site,"
            "'no_write':sys.dont_write_bytecode,'seed':os.environ.get('PYTHONHASHSEED'),"
            "'paths':sys.path,'secret':os.environ.get('SECRET_API_KEY')}))\n", encoding="utf-8")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15, check=True)
        report = json.loads(completed.stdout)
        self.assertEqual(Path(report["prefix"]).resolve(), spec.pycache_prefix.resolve())
        self.assertEqual(report["no_site"], 1)
        self.assertTrue(report["no_write"])
        self.assertEqual(report["seed"], "17")
        self.assertIsNone(report["secret"])
        project_index = report["paths"].index(str(self.project.resolve()))
        site_index = report["paths"].index(str(self.site.resolve()))
        self.assertGreater(project_index, 0)
        self.assertEqual(site_index, project_index + 1)
        self.assertNotIn("", report["paths"])
        self.assertNotIn(str(self.root), report["paths"])
        self.assertEqual(list(spec.pycache_prefix.iterdir()), [])

    def test_child_ignores_timestamp_valid_stale_bytecode(self):
        source = self.project / "probe.py"
        source.write_text("print('OLD')\n", encoding="utf-8")
        py_compile.compile(str(source), doraise=True)
        timestamp = source.stat().st_mtime_ns
        source.write_text("print('NEW')\n", encoding="utf-8")
        os.utime(source, ns=(timestamp, timestamp))
        unsafe_environment = dict(os.environ, PYTHONHASHSEED="17")
        unsafe_environment.pop("PYTHONPYCACHEPREFIX", None)
        unsafe_environment.pop("PYTHONPATH", None)
        unsafe = subprocess.run(
            [sys._base_executable, "-B", "-S", "-c",
             "import sys;sys.path.insert(0,sys.argv[1]);import probe", str(self.project)],
            env=unsafe_environment, cwd=self.root, capture_output=True, text=True, timeout=15, check=True,
        )
        self.assertEqual(unsafe.stdout.strip(), "OLD", "fixture must really contain reusable stale bytecode")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15, check=True)
        self.assertEqual(completed.stdout.strip(), "NEW")
        self.assertEqual(list(spec.pycache_prefix.iterdir()), [])

    def test_replaced_or_populated_prefix_is_rejected_before_exec(self):
        spec = self.prepare()
        if not hasattr(spec, "verify_ready"):
            self.fail("launcher cannot recheck its private prefix before exec")
        spec.verify_ready()
        (spec.pycache_prefix / "unexpected.pyc").write_bytes(b"stale")
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            spec.verify_ready()
        (spec.pycache_prefix / "unexpected.pyc").unlink()
        spec.pycache_prefix.rename(spec.pycache_prefix.with_name("original-cache"))
        spec.pycache_prefix.mkdir()
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            spec.verify_ready()

    def test_launcher_environment_is_immutable(self):
        spec = self.prepare()
        with self.assertRaises(TypeError):
            spec.environment["PYTHONHASHSEED"] = "random"

    def test_hostile_current_directory_cannot_shadow_startup_imports(self):
        for name in ("json", "runpy"):
            (self.root / (name + ".py")).write_text(
                "from pathlib import Path\n"
                "Path(__file__).with_suffix('.trap').write_text('executed')\n"
                "raise RuntimeError('hostile cwd imported')\n", encoding="utf-8")
        (self.project / "probe.py").write_text("print('safe')\n", encoding="utf-8")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "safe")
        self.assertFalse((self.root / "json.trap").exists())
        self.assertFalse((self.root / "runpy.trap").exists())

    def test_child_rechecks_cache_and_preexec_flags_before_loading_target(self):
        (self.project / "probe.py").write_text(
            "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('ran')\n", encoding="utf-8")
        spec = self.prepare()
        (spec.pycache_prefix / "unexpected.pyc").write_bytes(b"stale")
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((self.project / "probe.ran").exists())

    def test_child_rejects_changed_hash_seed_before_loading_target(self):
        (self.project / "probe.py").write_text(
            "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('ran')\n", encoding="utf-8")
        spec = self.prepare()
        altered = dict(spec.environment, PYTHONHASHSEED="0")
        completed = subprocess.run(spec.command, env=altered, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((self.project / "probe.ran").exists())

    def test_child_rejects_disabled_safe_path_or_site_guard(self):
        (self.project / "probe.py").write_text(
            "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('ran')\n", encoding="utf-8")
        for flag in ("-P", "-S", "-B"):
            with self.subTest(flag=flag):
                spec = self.prepare()
                command = list(spec.command)
                command.remove(flag)
                completed = subprocess.run(command, env=spec.environment, cwd=self.root,
                                           capture_output=True, text=True, timeout=15)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse((self.project / "probe.ran").exists())

    def test_unlisted_source_is_rejected_before_executing_it(self):
        (self.project / "probe.py").write_text("import intruder\n", encoding="utf-8")
        spec = self.prepare()
        (self.project / "intruder.py").write_text(
            "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('ran')\n", encoding="utf-8")
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((self.project / "intruder.ran").exists())

    def test_sourceless_bytecode_is_rejected_before_executing_it(self):
        source = self.project / "intruder.py"
        source.write_text("from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('ran')\n",
                          encoding="utf-8")
        py_compile.compile(str(source), cfile=str(self.project / "intruder.pyc"), doraise=True)
        source.unlink()
        (self.project / "probe.py").write_text("import intruder\n", encoding="utf-8")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((self.project / "intruder.ran").exists())

    def test_changed_approved_source_is_rejected_before_executing_it(self):
        source = self.project / "probe.py"
        source.write_text("print('original')\n", encoding="utf-8")
        spec = self.prepare()
        source.write_text("from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('ran')\n",
                          encoding="utf-8")
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((self.project / "probe.ran").exists())

    def test_unlisted_module_cannot_enter_through_a_later_meta_path_finder(self):
        outside = self.root / "outside.py"
        outside.write_text("from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('ran')\n",
                           encoding="utf-8")
        (self.project / "probe.py").write_text(
            "import importlib.util,sys\n"
            "class Finder:\n"
            " @classmethod\n"
            " def find_spec(cls, fullname, path=None, target=None):\n"
            "  if fullname == 'unlisted_custom':\n"
            f"   return importlib.util.spec_from_file_location(fullname, {str(outside)!r})\n"
            "sys.meta_path.append(Finder)\n"
            "import unlisted_custom\n", encoding="utf-8")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((self.root / "outside.ran").exists())

    def test_missing_optional_package_find_spec_returns_none(self):
        (self.project / "probe.py").write_text(
            "import importlib.util\n"
            "assert importlib.util.find_spec('mlg_definitely_missing_optional_package') is None\n"
            "print('optional-missing')\n", encoding="utf-8")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "optional-missing")

    def test_installed_distribution_metadata_remains_discoverable(self):
        dist = self.site / "mlg_fixture_dist-1.2.3.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: mlg-fixture-dist\nVersion: 1.2.3\n", encoding="utf-8")
        (self.project / "probe.py").write_text(
            "import importlib.metadata\n"
            "assert importlib.metadata.version('mlg-fixture-dist') == '1.2.3'\n"
            "print('metadata-ok')\n", encoding="utf-8")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "metadata-ok")

    def test_unlisted_distribution_metadata_is_not_trusted(self):
        (self.project / "probe.py").write_text(
            "import importlib.metadata\n"
            "print(importlib.metadata.version('mlg-unlisted-dist'))\n", encoding="utf-8")
        spec = self.prepare()
        dist = self.site / "mlg_unlisted_dist-1.2.3.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: mlg-unlisted-dist\nVersion: 1.2.3\n", encoding="utf-8")
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("1.2.3", completed.stdout)

    def test_real_supervisor_replaces_environment_before_interpreter_startup(self):
        from looking_glass.supervisor import run_supervised
        package = self.project / "looking_glass"
        package.mkdir()
        source = Path(__file__).resolve().parents[1] / "looking_glass"
        for name in ("__init__.py", "device_lease.py", "device_identity.py", "supervisor.py"):
            (package / name).write_bytes((source / name).read_bytes())
        report = self.root / "child-report.json"
        (self.project / "probe.py").write_text(
            "import json,os\n"
            "from looking_glass.supervisor import OwnerSession\n"
            "from looking_glass.device_lease import DeviceLease\n"
            "session=OwnerSession.from_environment()\n"
            "lease=DeviceLease.acquire(session.device_lease_path,timeout=5)\n"
            "OwnerSession.check(session,lease)\n"
            f"open({str(report)!r},'w').write(json.dumps({{'path':os.environ.get('PYTHONPATH'),"
            "'secret':os.environ.get('MLG_TEST_SECRET'),'cuda':os.environ.get('CUDA_LAUNCH_BLOCKING')}))\n"
            "OwnerSession.attest_clean(session,lease)\n", encoding="utf-8")
        hostile = self.root / "hostile"
        (hostile / "encodings").mkdir(parents=True)
        sentinel = self.root / "hostile-ran"
        (hostile / "encodings" / "__init__.py").write_text(
            f"open({str(sentinel)!r},'w').write('executed')\nraise RuntimeError('hostile startup')\n",
            encoding="utf-8")
        spec = self.prepare()
        spec.verify_ready()
        with patch.dict(os.environ, {"PYTHONPATH": str(hostile), "MLG_TEST_SECRET": "secret",
                                     "CUDA_LAUNCH_BLOCKING": "1"}):
            result = run_supervised(spec.command, device_lease_path=self.root / "gpu.lock",
                                    marker_path=self.root / "gpu.unsafe.json", timeout=15,
                                    cwd=self.root, env=spec.environment, inherit_environment=False)
        self.assertTrue(result.clean, result.reason)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(sentinel.exists())
        self.assertEqual(json.loads(report.read_text()), {"path": None, "secret": None, "cuda": None})
        self.assertFalse((self.root / "gpu.unsafe.json").exists())

    def run_fake_supervised_owner(self, *, remove_marker, runtime_uuid=None, initialized=False,
                                  enable_test_owner=True, dll_change=None, preload=False):
        from looking_glass.device_identity import resolve_device_identity
        from looking_glass.supervisor import run_supervised
        package = self.project / "looking_glass"
        package.mkdir()
        source = Path(__file__).resolve().parents[1] / "looking_glass"
        for name in ("__init__.py", "device_lease.py", "device_identity.py", "supervisor.py",
                     "resource_profile.py", "memory_budget.py", "gpu_bootstrap.py"):
            (package / name).write_bytes((source / name).read_bytes())
        observed_uuid = runtime_uuid or self.gpu_uuid
        (self.project / "torch.py").write_text(
            "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('imported')\n"
            "class Uuid:\n"
            f" def __str__(self): return {observed_uuid[4:]!r}\n"
            "class _C:\n _CUuuid=Uuid\n"
            "class cuda:\n"
            " @staticmethod\n def is_initialized():\n"
            "  Path(__file__).with_suffix('.runtime').write_text('called')\n"
            f"  return {initialized!r}\n"
            " @staticmethod\n def device_count(): return 1\n"
            " @staticmethod\n def get_device_properties(index):\n"
            "  class Properties: uuid=Uuid()\n"
            "  return Properties()\n", encoding="utf-8")
        (self.project / "transformers.py").write_text(
            "from pathlib import Path\nPath(__file__).with_suffix('.ran').write_text('imported')\n",
            encoding="utf-8")
        torch_lib = self.site / "torch" / "lib"
        torch_lib.mkdir(parents=True)
        (torch_lib / "approved.dll").write_bytes(b"approved-fake-dll")
        run = self.root / "run"
        journal = self.bootstrap.BootstrapJournal.create(run, protocol_sha256="a" * 64,
                    git_commit="b" * 40, git_tree="c" * 40, run_id="1" * 32)
        start_sha = hashlib.sha256((run / "started.json").read_bytes()).hexdigest()
        sabotage = ("from pathlib import Path\n"
                    "Path(os.environ['MLG_UNSAFE_MARKER_PATH']).unlink()\n") if remove_marker else ""
        (self.project / "probe.py").write_text(
            "import os,sys\nsys.excepthook=lambda *args: None\n"
            + sabotage +
            ("import torch\n" if preload else "") +
            "from looking_glass import gpu_bootstrap\n"
            + ("gpu_bootstrap.GPU_BOOTSTRAP_READY=True\n" if enable_test_owner else "") +
            "from looking_glass.gpu_bootstrap import claim_owner_and_load_dependencies\n"
            "from looking_glass.supervisor import OwnerSession\n"
            f"session,lease,torch,transformers,uuid=claim_owner_and_load_dependencies({str(run)!r},"
            f" parent_run_id={'1' * 32!r}, started_sha256={start_sha!r},"
            " source_sha256=os.environ['MLG_SOURCE_MANIFEST_SHA256'])\n"
            "OwnerSession.attest_clean(session,lease)\n", encoding="utf-8")
        spec = self.prepare()
        if dll_change == "extra":
            (torch_lib / "unlisted.dll").write_bytes(b"unexpected")
        elif dll_change == "changed":
            (torch_lib / "approved.dll").write_bytes(b"modified")
        elif dll_change == "missing":
            (torch_lib / "approved.dll").unlink()
        identity = resolve_device_identity(self.root, self.gpu_uuid, physical_uuids=[self.gpu_uuid],
                    environ={"CUDA_VISIBLE_DEVICES": self.gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        spec.verify_ready()
        result = run_supervised(spec.command, device_lease_path=identity.lock_path,
                    marker_path=identity.marker_path, identity=identity, timeout=15, cwd=self.root,
                    env=spec.environment, inherit_environment=False)
        return journal, identity, result

    def test_owner_claim_precedes_dependency_imports_and_records_supervisor_identity(self):
        journal, identity, result = self.run_fake_supervised_owner(remove_marker=False)
        self.assertTrue(result.clean, result.reason)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue((self.project / "torch.ran").exists())
        self.assertTrue((self.project / "transformers.ran").exists())
        self.assertEqual(json.loads((journal.directory / "owner.json").read_bytes()), {
            "schema": "resource-bootstrap-owner-v1", "parent_run_id": "1" * 32,
            "supervisor_run_id": result.run_id, "pid": result.pid})
        self.assertFalse(identity.marker_path.exists())

    def test_lost_marker_blocks_dependency_imports_and_retains_quarantine(self):
        journal, identity, result = self.run_fake_supervised_owner(remove_marker=True)
        self.assertFalse(result.clean)
        self.assertTrue(identity.marker_path.exists())
        self.assertFalse((journal.directory / "owner.json").exists())
        self.assertFalse((self.project / "torch.ran").exists())
        self.assertFalse((self.project / "transformers.ran").exists())

    def test_owner_is_disabled_by_default_before_dependency_imports(self):
        journal, identity, result = self.run_fake_supervised_owner(remove_marker=False,
                                                                  enable_test_owner=False)
        self.assertFalse(result.clean)
        self.assertTrue(identity.marker_path.exists())
        self.assertFalse((journal.directory / "owner.json").exists())
        self.assertFalse((self.project / "torch.ran").exists())
        self.assertFalse((self.project / "transformers.ran").exists())

    def test_preloaded_torch_forces_quarantine_before_runtime_queries(self):
        journal, identity, result = self.run_fake_supervised_owner(remove_marker=False, preload=True)
        self.assertFalse(result.clean)
        self.assertTrue(identity.marker_path.exists())
        self.assertFalse((journal.directory / "owner.json").exists())
        self.assertTrue((self.project / "torch.ran").exists())
        self.assertFalse((self.project / "torch.runtime").exists())
        self.assertFalse((self.project / "transformers.ran").exists())

    def test_unlisted_or_modified_torch_dlls_block_torch_import(self):
        for change in ("extra", "changed", "missing"):
            with self.subTest(change=change):
                # Each case needs its own workspace and supervisor marker.
                case = OwnerLauncherTests(methodName="runTest")
                case.setUp()
                try:
                    _, identity, result = case.run_fake_supervised_owner(remove_marker=False,
                                                                        dll_change=change)
                    self.assertFalse(result.clean)
                    self.assertTrue(identity.marker_path.exists())
                    self.assertFalse((case.project / "torch.ran").exists())
                    self.assertFalse((case.project / "transformers.ran").exists())
                finally:
                    case.doCleanups()

    def test_wrong_runtime_uuid_blocks_transformers_import_and_retains_quarantine(self):
        _, identity, result = self.run_fake_supervised_owner(
            remove_marker=False, runtime_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        self.assertFalse(result.clean)
        self.assertTrue(identity.marker_path.exists())
        self.assertTrue((self.project / "torch.ran").exists())
        self.assertFalse((self.project / "transformers.ran").exists())

    def test_prior_cuda_initialization_blocks_runtime_work_and_retains_quarantine(self):
        _, identity, result = self.run_fake_supervised_owner(remove_marker=False, initialized=True)
        self.assertFalse(result.clean)
        self.assertTrue(identity.marker_path.exists())
        self.assertTrue((self.project / "torch.ran").exists())
        self.assertFalse((self.project / "transformers.ran").exists())

    def test_guard_binding_detects_removal_and_wrong_manifest(self):
        package = self.project / "looking_glass"
        package.mkdir()
        source = Path(__file__).resolve().parents[1] / "looking_glass"
        for name in ("__init__.py", "device_lease.py", "device_identity.py", "supervisor.py",
                     "resource_profile.py", "memory_budget.py", "gpu_bootstrap.py"):
            (package / name).write_bytes((source / name).read_bytes())
        (self.project / "probe.py").write_text(
            "import os,sys\n"
            "from looking_glass.gpu_bootstrap import assert_import_guard\n"
            "source=os.environ['MLG_SOURCE_MANIFEST_SHA256']\n"
            "assert_import_guard(source)\n"
            "try:\n assert_import_guard('0'*64)\n"
            "except RuntimeError: pass\n"
            "else: raise AssertionError('wrong manifest accepted')\n"
            "sys.meta_path.pop(0)\n"
            "try:\n assert_import_guard(source)\n"
            "except RuntimeError: pass\n"
            "else: raise AssertionError('removed guard accepted')\n"
            "print('guarded')\n", encoding="utf-8")
        spec = self.prepare()
        completed = subprocess.run(spec.command, env=spec.environment, cwd=self.root,
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "guarded")

    def test_rejects_uncontrolled_path(self):
        with self.assertRaises((ValueError, self.bootstrap.BootstrapError)):
            self.bootstrap.prepare_owner_launch(
                project_root=self.project, site_packages=self.site, environment_root=self.environment_root,
                private_parent=self.private, source_manifest=b"{}", trusted_path="relative",
                gpu_uuid=self.gpu_uuid, module="probe", arguments=(), inherited_environment={},
            )

    def test_invalid_uuid_is_rejected_before_launch(self):
        with self.assertRaises(ValueError):
            self.bootstrap.prepare_owner_launch(
                project_root=self.project, site_packages=self.site, environment_root=self.environment_root,
                private_parent=self.private, source_manifest=b"{}", trusted_path=str(self.root),
                gpu_uuid="GPU-1234", module="probe", arguments=(), inherited_environment={},
            )
        self.assertEqual(list(self.private.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
