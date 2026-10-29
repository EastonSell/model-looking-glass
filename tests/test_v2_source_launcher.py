"""Fabricated-repository launcher checks; never generate v2 inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE_PYTHON = Path(sys.base_prefix) / "python.exe"
SCRIPT_FILES = ("v2_source_launcher.py", "v2_source_bootstrap.py", "v2_source_guards.py")


@unittest.skipUnless(os.name == "nt" and shutil.which("git"), "Windows and git required")
class V2SourceLauncherTests(unittest.TestCase):
    def fixture(self, root, *, initializer=None, fail_assignment=False, cancel_assignment=False):
        checkout = root / "checkout"
        scripts = checkout / "scripts"
        package = checkout / "looking_glass"
        scripts.mkdir(parents=True)
        package.mkdir()
        for name in SCRIPT_FILES:
            data = (ROOT / "scripts" / name).read_bytes()
            if name == "v2_source_guards.py" and fail_assignment:
                old = b"if not self._kernel.AssignProcessToJobObject(self._handle, int(process._handle)):"
                self.assertEqual(data.count(old), 1)
                data = data.replace(old, b"if True:")
            if name == "v2_source_guards.py" and cancel_assignment:
                old = b"if not self._kernel.AssignProcessToJobObject(self._handle, int(process._handle)):"
                self.assertEqual(data.count(old), 1)
                data = data.replace(old, b"raise KeyboardInterrupt\n        if True:")
            (scripts / name).write_bytes(data)
        marker = root / "imported.txt"
        if initializer is None:
            initializer = f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        (package / "__init__.py").write_text(initializer, encoding="utf-8")
        (package / "trace_schema.py").write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(checkout)], check=True, capture_output=True)
        for key, value in (("user.name", "Development Test"), ("user.email", "dev@example.invalid")):
            subprocess.run(["git", "-C", str(checkout), "config", key, value],
                           check=True, capture_output=True)
        subprocess.run(["git", "-C", str(checkout), "add", "--", "scripts", "looking_glass"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fabricated fixture"],
                       check=True, capture_output=True)
        venv = root / "venv"
        site = venv / "Lib" / "site-packages"
        site.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            f"home = {sys.base_prefix}\nimplementation = CPython\n"
            f"version_info = {'.'.join(map(str, sys.version_info[:3]))}\n"
            "include-system-site-packages = false\n", encoding="utf-8")
        work = root / "work"
        work.mkdir()
        return checkout, venv, site, work, marker

    def args(self, fixture):
        checkout, venv, site, work, _ = fixture
        return [str(BASE_PYTHON), "-I", "-S", "-B", str(checkout / "scripts" / SCRIPT_FILES[0]),
                "--development-probe", "--checkout", str(checkout), "--venv-root", str(venv),
                "--dependency", str(site), "--work-dir", str(work), "--module", "looking_glass.trace_schema",
                "--timeout", "15"]

    def run_launcher(self, fixture, *, mask=True):
        env = dict(os.environ)
        if mask:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        return subprocess.run(self.args(fixture), capture_output=True, text=True,
                              env=env, cwd=fixture[3], timeout=45)

    def test_guarded_committed_diagnostic_reports_reaped_child(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            result = self.run_launcher(fixture)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "development_only")
            self.assertFalse(report["study_ready"])
            self.assertTrue(report["child_reaped"])
            self.assertEqual(report["source_sha256"]["scripts/v2_source_launcher.py"],
                             hashlib.sha256((fixture[0] / "scripts" / SCRIPT_FILES[0]).read_bytes()).hexdigest())
            self.assertEqual(fixture[4].read_text(), "imported")
            self.assertEqual(list(fixture[3].iterdir()), [])
            self.assertNotIn(str(fixture[0]), result.stdout)

    def test_guarded_torch_import_has_no_untracked_child_process(self):
        torch_spec = importlib.util.find_spec('torch')
        if torch_spec is None or not torch_spec.submodule_search_locations:
            self.skipTest('torch is not installed')
        site = Path(next(iter(torch_spec.submodule_search_locations))).parent.resolve()
        venv = Path(sys.prefix).resolve()
        if not site.is_relative_to(venv):
            self.skipTest('torch is not installed under this interpreter venv')
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            source = fixture[0] / 'looking_glass' / 'trace_schema.py'
            observed = platform.uname()
            values = [observed.system, observed.node, observed.release, observed.version, observed.machine]
            source.write_text("import platform, torch\n"
                              "u=platform.uname()\n"
                              f"assert [u.system,u.node,u.release,u.version,u.machine] == {values!r}\n"
                              "assert not torch.cuda.is_initialized()\n", encoding='utf-8')
            subprocess.run(['git', '-C', str(fixture[0]), 'add', '--', 'looking_glass/trace_schema.py'],
                           check=True, capture_output=True)
            subprocess.run(['git', '-C', str(fixture[0]), 'commit', '-qm', 'torch import fixture'],
                           check=True, capture_output=True)
            args = self.args(fixture)
            args[args.index('--venv-root') + 1] = str(venv)
            args[args.index('--dependency') + 1] = str(site)
            result = subprocess.run(args, capture_output=True, text=True, cwd=fixture[3],
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES':'-1'}, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report['child_reaped'])
            self.assertFalse(report['study_ready'])
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_guarded_child_rejects_python_process_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            marker = root / 'unexpected-child.txt'
            source = fixture[0] / 'looking_glass' / 'trace_schema.py'
            payload = f"from pathlib import Path; Path({str(marker)!r}).touch()"
            source.write_text("import subprocess,sys\n"
                              f"subprocess.run([sys.executable, '-c', {payload!r}], check=True)\n", encoding='utf-8')
            subprocess.run(['git', '-C', str(fixture[0]), 'add', '--', 'looking_glass/trace_schema.py'],
                           check=True, capture_output=True)
            subprocess.run(['git', '-C', str(fixture[0]), 'commit', '-qm', 'process refusal fixture'],
                           check=True, capture_output=True)
            result = self.run_launcher(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(json.loads(result.stderr)['child_reaped'])
            self.assertFalse(marker.exists())
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_dirty_source_and_missing_gpu_mask_reject_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertNotEqual(self.run_launcher(fixture, mask=False).returncode, 0)
            self.assertFalse(fixture[4].exists())
            (fixture[0] / "looking_glass" / "trace_schema.py").write_text("VALUE = 2\n")
            result = self.run_launcher(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture[4].exists())
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_git_replacement_cannot_substitute_a_different_committed_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            checkout = fixture[0]
            original = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            source = checkout / 'looking_glass' / 'trace_schema.py'
            original_bytes = source.read_bytes()
            alternate_bytes = b'VALUE = 42\n'
            source.write_bytes(alternate_bytes)
            subprocess.run(['git', '-C', str(checkout), 'add', '--', 'looking_glass/trace_schema.py'],
                           check=True, capture_output=True)
            subprocess.run(['git', '-C', str(checkout), 'commit', '-qm', 'alternate tree'],
                           check=True, capture_output=True)
            alternate = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            subprocess.run(['git', '-C', str(checkout), 'replace', original, alternate],
                           check=True, capture_output=True)
            subprocess.run(['git', '--no-replace-objects', '-C', str(checkout), 'checkout', '-q', '--detach', original],
                           check=True, capture_output=True)
            source.write_bytes(alternate_bytes)
            result = self.run_launcher(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture[4].exists())
            self.assertEqual(list(fixture[3].iterdir()), [])
            source.write_bytes(original_bytes)
            clean = self.run_launcher(fixture)
            self.assertEqual(clean.returncode, 0, clean.stderr)
            self.assertEqual(json.loads(clean.stdout)['source_revision'], original)
            self.assertTrue(fixture[4].exists())

    def test_inherited_git_environment_cannot_redirect_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intended, other = self.fixture(root / 'intended'), self.fixture(root / 'other')
            revision = subprocess.check_output(['git', '-C', str(intended[0]), 'rev-parse', 'HEAD'],
                                               text=True).strip()
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1',
                   'GIT_DIR': str(other[0] / '.git'), 'GIT_WORK_TREE': str(other[0]),
                   'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'badkey',
                   'GIT_CONFIG_VALUE_0': 'injected',
                   'GIT_GRAFT_FILE': str(other[0] / 'untrusted-grafts')}
            result = subprocess.run(self.args(intended), capture_output=True, text=True,
                                    env=env, cwd=intended[3], timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['source_revision'], revision)
            self.assertTrue(intended[4].exists())
            self.assertFalse(other[4].exists())

    def test_missing_venv_home_does_not_default_to_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            config = fixture[1] / "pyvenv.cfg"
            config.write_text("\n".join(line for line in config.read_text().splitlines()
                                         if not line.startswith("home =")) + "\n")
            result = subprocess.run(self.args(fixture), capture_output=True, text=True,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                                    cwd=Path(sys.base_prefix).resolve(), timeout=45)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture[4].exists())

    def test_existing_writer_blocks_the_guarded_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            with (fixture[0] / "looking_glass" / "trace_schema.py").open("a"):
                result = self.run_launcher(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture[4].exists())
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_concurrent_parent_thread_rejects_before_handle_inheritance(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            args = self.args(fixture)
            code = ("import runpy,sys,threading\n"
                    "p=sys.argv[1]; sys.argv=sys.argv[1:]\n"
                    "stop=threading.Event()\n"
                    "worker=threading.Thread(target=stop.wait,daemon=True); worker.start()\n"
                    "try: runpy.run_path(p)['main']()\n"
                    "finally: stop.set(); worker.join()\n")
            result = subprocess.run([args[0], "-I", "-S", "-B", "-c", code, *args[4:]],
                                    capture_output=True, text=True, cwd=fixture[3],
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=45)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture[4].exists())
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_low_level_thread_does_not_require_parent_handle_inheritance(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            args = self.args(fixture)
            code = ("import _thread,runpy,sys,threading\n"
                    "p=sys.argv[1]; sys.argv=sys.argv[1:]\n"
                    "started=threading.Event(); stop=threading.Event()\n"
                    "def worker(): started.set(); stop.wait()\n"
                    "_thread.start_new_thread(worker,()); started.wait()\n"
                    "try: runpy.run_path(p)['main']()\n"
                    "finally: stop.set()\n")
            result = subprocess.run([args[0], "-I", "-S", "-B", "-c", code, *args[4:]],
                                    capture_output=True, text=True, cwd=fixture[3],
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(fixture[4].exists())

    def test_unapproved_module_is_rejected_before_project_import(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            args = self.args(fixture)
            args[args.index("--module") + 1] = "looking_glass.causal_v2_runtime"
            result = subprocess.run(args, capture_output=True, text=True, cwd=fixture[3],
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=45)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fixture[4].exists())

    def test_assignment_failure_reaps_child_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), fail_assignment=True)
            result = self.run_launcher(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(json.loads(result.stderr)["child_reaped"])
            self.assertFalse(fixture[4].exists())
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_cancellation_is_classified_after_child_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), cancel_assignment=True)
            result = self.run_launcher(fixture)
            self.assertEqual(result.returncode, 130)
            status = json.loads(result.stderr)
            self.assertEqual(status["status"], "cancelled")
            self.assertTrue(status["child_reaped"])
            self.assertFalse(fixture[4].exists())
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_failed_job_drain_still_reaps_direct_child_and_keeps_retry_state(self):
        spec = importlib.util.spec_from_file_location("v2_launcher_test", ROOT / "scripts" / SCRIPT_FILES[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Child:
            stdin = None
            def __init__(self):
                self.killed = False
            def poll(self):
                return 1 if self.killed else None
            def kill(self):
                self.killed = True
            def wait(self, timeout):
                if not self.killed:
                    raise AssertionError("child was not terminated")
                return 1
        class Job:
            def __init__(self):
                self.fail = True
                self.closed = False
            def terminate_and_wait(self):
                if self.fail:
                    raise OSError("injected drain failure")
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "launch"
            temporary.mkdir()
            manifest = temporary / "sources.json"
            manifest.write_text("{}")
            quarantine = module.LaunchQuarantine(temporary, manifest)
            quarantine.job, quarantine.child = Job(), Child()
            with self.assertRaises(module.LaunchError):
                quarantine.retry_cleanup()
            self.assertTrue(quarantine.child.killed)
            self.assertTrue(quarantine.child_reaped)
            self.assertTrue(temporary.exists())
            self.assertFalse(quarantine.job.closed)
            quarantine.job.fail = False
            quarantine.retry_cleanup()
            self.assertTrue(quarantine.complete)
            self.assertFalse(temporary.exists())

    def test_interrupt_during_job_drain_still_reaps_child_and_stays_cancelled(self):
        spec = importlib.util.spec_from_file_location("v2_launcher_cancel_test", ROOT / "scripts" / SCRIPT_FILES[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Child:
            stdin = None
            killed = False
            def poll(self):
                return 1 if self.killed else None
            def kill(self):
                self.killed = True
            def wait(self, timeout):
                self_test.assertTrue(self.killed)
                return 1
        class Job:
            def terminate_and_wait(self):
                raise KeyboardInterrupt
        self_test = self
        child = Child()
        with self.assertRaises(KeyboardInterrupt) as raised:
            module._stop_and_reap(child, Job())
        self.assertTrue(child.killed)
        self.assertTrue(raised.exception.child_reaped)

    def test_unknown_sourceless_module_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "unknown-executed.txt"
            fixture = self.fixture(root, initializer="from . import unknown\n")
            unknown = fixture[0] / "looking_glass" / "unknown.py"
            unknown.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
            py_compile.compile(str(unknown), cfile=str(unknown.with_suffix(".pyc")), doraise=True)
            unknown.unlink()
            result = self.run_launcher(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(json.loads(result.stderr)["child_reaped"])
            self.assertFalse(marker.exists())
            self.assertEqual(list(fixture[3].iterdir()), [])

    def test_owner_death_terminates_actual_interpreter_before_releasing_sources(self):
        import ctypes
        from ctypes import wintypes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "child-ready.txt"
            initializer = ("import os, threading\nfrom pathlib import Path\n"
                           f"Path({str(ready)!r}).write_text(str(os.getpid()))\n"
                           "threading.Event().wait(3600)\n")
            fixture = self.fixture(root, initializer=initializer)
            owner = subprocess.Popen(self.args(fixture), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                                     cwd=fixture[3], creationflags=subprocess.CREATE_NO_WINDOW)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            child_handle = None
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists(), "guarded child did not reach the development fixture")
                child_handle = kernel.OpenProcess(0x00100000 | 1, False, int(ready.read_text()))
                self.assertTrue(child_handle)
                source = fixture[0] / "looking_glass" / "trace_schema.py"
                with self.assertRaises(PermissionError):
                    source.write_text("changed\n")
                owner.kill()
                owner.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(child_handle, 10000), 0)
                source.write_text("released\n")
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=5)
                if child_handle:
                    if kernel.WaitForSingleObject(child_handle, 0) != 0:
                        kernel.TerminateProcess(child_handle, 1)
                        kernel.WaitForSingleObject(child_handle, 10000)
                    kernel.CloseHandle(child_handle)
                owner.stdout.close()
                owner.stderr.close()
                # The owner died before cooperative cleanup. Restore only the
                # disposable test prefix's ACL, then remove its known files.
                for temporary in fixture[3].iterdir():
                    self.assertTrue(temporary.name.startswith("v2-launch-"))
                    items = list(temporary.iterdir())
                    for item in items:
                        if item.is_dir() and item.name.startswith("v2-cache-"):
                            self.assertEqual(list(item.iterdir()), [])
                            icacls = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
                            subprocess.run([str(icacls), str(item), "/remove:d", "*S-1-1-0"],
                                           check=True, capture_output=True)
                            item.rmdir()
                        else:
                            self.assertEqual(item.name, "sources.json")
                            item.unlink()
                    temporary.rmdir()


if __name__ == "__main__":
    unittest.main()
