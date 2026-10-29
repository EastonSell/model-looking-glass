"""Development-only source-loader checks; no v2 cases or schedule."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest


BOOTSTRAP = Path(__file__).resolve().parents[1] / "scripts" / "v2_source_bootstrap.py"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class SourceBootstrapTests(unittest.TestCase):
    def test_guarded_process_audit_hook_rejects_creation_events(self):
        code = r'''
import importlib.util,sys
spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[1]); module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module); module._install_process_guard()
for name in ('subprocess.Popen','_winapi.CreateProcess','os.system','os.spawn','os.posix_spawn',
             'os.fork','os.forkpty','os.exec','os.startfile','os.startfile/2'):
    try: sys.audit(name)
    except module.BootstrapError: pass
    else: raise AssertionError(f'process event {name} was not rejected')
'''
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(BOOTSTRAP)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_guarded_process_audit_hook_detects_suppressed_installation(self):
        code = r'''
import importlib.util,sys
def suppress(event,args):
    if event=='sys.addaudithook': raise RuntimeError('suppress installation')
sys.addaudithook(suppress)
spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[1]); module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try: module._install_process_guard()
except module.BootstrapError: pass
else: raise AssertionError('suppressed hook was accepted')
'''
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(BOOTSTRAP)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def fixture(self, root, initializer, child=b"VALUE = 41\n"):
        checkout = root / "checkout"
        package = checkout / "looking_glass"
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(initializer)
        (package / "child.py").write_bytes(child)
        sources = {"looking_glass/__init__.py": _sha(initializer),
                   "looking_glass/child.py": _sha(child)}
        manifest = root / "sources.json"
        manifest.write_text(json.dumps(sources), encoding="utf-8")
        venv = root / "venv"
        dependency = venv / "Lib" / "site-packages"
        dependency.mkdir(parents=True)
        prefix = root / "private-cache"
        prefix.mkdir()
        return checkout, package, manifest, venv, dependency, prefix

    def run_probe(self, fixture, *, flags=True, module="looking_glass.child"):
        checkout, _, manifest, venv, dependency, prefix = fixture
        options = ["-I", "-S", "-B", "-X", f"pycache_prefix={prefix}"] if flags else []
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        return subprocess.run(
            [sys.executable, *options, str(BOOTSTRAP), "--development-probe",
             "--checkout", str(checkout), "--sources", str(manifest),
             "--venv-root", str(venv), "--dependency", str(dependency),
             "--prefix", str(prefix), "--module", module],
            cwd=checkout, capture_output=True, text=True, env=env,
        )

    def test_vetted_source_import_ignores_checkout_root_shadows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root, b"import fractions\nfrom . import child\n")
            checkout = fixture[0]
            (checkout / "fractions.py").write_text("raise RuntimeError('shadow executed')\n")
            py_compile.compile(str(checkout / "fractions.py"),
                               cfile=str(checkout / "fractions.pyc"), doraise=True)
            (checkout / "fractions.pyd").write_bytes(b"not a library")
            result = self.run_probe(fixture)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "development_only")
            self.assertFalse(report["study_ready"])
            self.assertEqual(set(report["project_sources"]),
                             {"looking_glass/__init__.py", "looking_glass/child.py"})
            self.assertNotIn(str(checkout), result.stdout)

    def test_unknown_sourceless_project_module_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root, b"from . import unknown\n")
            package = fixture[1]
            marker = root / "executed.txt"
            evil = package / "unknown.py"
            evil.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n")
            py_compile.compile(str(evil), cfile=str(package / "unknown.pyc"), doraise=True)
            evil.unlink()
            result = self.run_probe(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

    def test_missing_flags_nonempty_prefix_and_changed_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root, b"from . import child\n")
            self.assertNotEqual(self.run_probe(fixture, flags=False).returncode, 0)
            (fixture[5] / "unexpected").write_bytes(b"x")
            self.assertNotEqual(self.run_probe(fixture).returncode, 0)
            (fixture[5] / "unexpected").unlink()
            (fixture[1] / "child.py").write_text("VALUE = 42\n")
            self.assertNotEqual(self.run_probe(fixture).returncode, 0)

    def test_duplicate_manifest_and_outside_dependency_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root, b"from . import child\n")
            manifest = fixture[2]
            manifest.write_text('{"looking_glass/__init__.py":"' + "0" * 64 +
                                '","looking_glass/__init__.py":"' + "0" * 64 + '"}')
            self.assertNotEqual(self.run_probe(fixture).returncode, 0)
            fixture = self.fixture(root / "other", b"from . import child\n")
            outside = root / "outside"
            outside.mkdir()
            parts = list(fixture)
            parts[4] = outside
            self.assertNotEqual(self.run_probe(tuple(parts)).returncode, 0)

    def test_source_changed_during_initializer_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initializer = (b"from pathlib import Path\n"
                           b"Path(__file__).with_name('child.py').write_text(\"raise RuntimeError('bad')\\n\")\n"
                           b"from . import child\n")
            fixture = self.fixture(root, initializer)
            self.assertNotEqual(self.run_probe(fixture).returncode, 0)

    def test_post_import_source_mutation_and_checkout_path_insertion_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initializer = (b"from . import child\nfrom pathlib import Path\n"
                           b"Path(__file__).with_name('child.py').write_text('VALUE = 99\\n')\n")
            fixture = self.fixture(root, initializer)
            self.assertNotEqual(self.run_probe(fixture).returncode, 0)
            initializer = (b"import sys\nfrom pathlib import Path\n"
                           b"sys.path.append(str(Path(__file__).parent.parent))\nfrom . import child\n")
            fixture = self.fixture(root / "other", initializer)
            self.assertNotEqual(self.run_probe(fixture).returncode, 0)

    def test_builtin_precedence_and_dependency_pth_do_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = b"import _symtable\nassert _symtable.__spec__.origin == 'built-in'\n"
            fixture = self.fixture(root, b"from . import child\n", child)
            dependency = fixture[4]
            marker = root / "pth_executed.txt"
            (dependency / "_symtable.py").write_text("raise RuntimeError('builtin shadow')\n")
            (dependency / "malicious.pth").write_text(
                f"import pathlib; pathlib.Path({str(marker)!r}).write_text('bad')\n")
            result = self.run_probe(fixture)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(marker.exists())

    def test_preimported_project_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, _, manifest, venv, dependency, prefix = self.fixture(root, b"from . import child\n")
            argv = [str(BOOTSTRAP), "--development-probe", "--checkout", str(checkout),
                    "--sources", str(manifest), "--venv-root", str(venv),
                    "--dependency", str(dependency), "--prefix", str(prefix),
                    "--module", "looking_glass.child"]
            code = ("import runpy, sys, types\n"
                    "sys.modules['looking_glass'] = types.ModuleType('looking_glass')\n"
                    f"sys.argv = {argv!r}\n"
                    f"runpy.run_path({str(BOOTSTRAP)!r}, run_name='__main__')\n")
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", f"pycache_prefix={prefix}", "-c", code],
                cwd=checkout, capture_output=True, text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
            )
            self.assertNotEqual(result.returncode, 0)

    def test_cuda_mask_is_required_before_project_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, _, manifest, venv, dependency, prefix = self.fixture(root, b"from . import child\n")
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", f"pycache_prefix={prefix}",
                 str(BOOTSTRAP), "--development-probe", "--checkout", str(checkout),
                 "--sources", str(manifest), "--venv-root", str(venv), "--dependency", str(dependency),
                 "--prefix", str(prefix), "--module", "looking_glass.child"],
                cwd=checkout, capture_output=True, text=True, env=env,
            )
            self.assertNotEqual(result.returncode, 0)

    @unittest.skipUnless(os.name == "nt", "Windows inherited protection handles")
    def test_guarded_start_rejects_eof_and_partial_token_before_import(self):
        import msvcrt
        import platform
        token = b"model-looking-glass-development-start-v2\n"
        observed = platform.uname()
        platform_uname = [observed.system, observed.node, observed.release, observed.version, observed.machine]
        base_python = str(Path(sys.base_prefix) / "python.exe")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard_path = root / "guard.txt"
            guard_path.write_bytes(b"guard")
            marker = root / "project-imported.txt"
            with guard_path.open("rb") as guard:
                handle = msvcrt.get_osfhandle(guard.fileno())
                initial = (f"import os\nfrom pathlib import Path\n"
                           f"assert not os.get_handle_inheritable({handle})\n"
                           f"Path({str(marker)!r}).write_text('imported')\n").encode()
                fixture = self.fixture(root, initial)
                checkout, _, manifest, venv, dependency, prefix = fixture
                args = [base_python, "-I", "-S", "-B", "-X", f"pycache_prefix={prefix}",
                        str(BOOTSTRAP), "--development-probe", "--checkout", str(checkout),
                        "--sources", str(manifest), "--venv-root", str(venv),
                        "--dependency", str(dependency), "--prefix", str(prefix),
                        "--module", "looking_glass.child", "--require-start"]
                valid = token + json.dumps({"guard_handles": [handle], "platform_uname": platform_uname}).encode()
                wrong_platform = token + json.dumps({"guard_handles": [handle],
                                                     "platform_uname": ['Windows','','','','wrong']}).encode()
                for payload in (b"", token[:-1], token + b"x", token + b"{}", valid + b"x", wrong_platform, valid):
                    startup = subprocess.STARTUPINFO()
                    startup.lpAttributeList = {"handle_list": [handle]}
                    os.set_handle_inheritable(handle, True)
                    try:
                        child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE, startupinfo=startup,
                                                 close_fds=True, cwd=checkout,
                                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
                    finally:
                        os.set_handle_inheritable(handle, False)
                    out, err = child.communicate(input=payload, timeout=15)
                    if payload == valid:
                        self.assertEqual(child.returncode, 0, (out, err))
                        self.assertEqual(marker.read_text(), "imported")
                    else:
                        self.assertNotEqual(child.returncode, 0, (out, err))
                        self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "nt", "Windows inherited protection handles")
    def test_guarded_start_requires_live_inherited_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "project-imported.txt"
            fixture = self.fixture(root, f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode())
            checkout, _, manifest, venv, dependency, prefix = fixture
            args = [str(Path(sys.base_prefix) / "python.exe"), "-I", "-S", "-B", "-X",
                    f"pycache_prefix={prefix}", str(BOOTSTRAP), "--development-probe",
                    "--checkout", str(checkout), "--sources", str(manifest),
                    "--venv-root", str(venv), "--dependency", str(dependency),
                    "--prefix", str(prefix), "--module", "looking_glass.child", "--require-start"]
            for handles in ([], [99999999], [True], [1, 1]):
                payload = (b"model-looking-glass-development-start-v2\n" +
                           json.dumps({"guard_handles": handles,
                                       "platform_uname": ['Windows', '', '', '', 'AMD64']}).encode())
                result = subprocess.run(args, input=payload,
                                        capture_output=True, cwd=checkout,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(marker.exists())

    def test_explicit_venv_dependency_path_can_import_cpu_torch(self):
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.submodule_search_locations:
            self.skipTest("torch is not installed")
        site = Path(next(iter(spec.submodule_search_locations))).parent.resolve()
        venv = Path(sys.prefix).resolve()
        if not site.is_relative_to(venv):
            self.skipTest("torch is not installed under this interpreter's venv")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = (b"import torch\n"
                     b"assert not torch.cuda.is_initialized()\n"
                     b"assert torch.tensor([1.], device='cpu').device.type == 'cpu'\n")
            fixture = list(self.fixture(root, b"from . import child\n", child))
            fixture[3], fixture[4] = venv, site
            result = self.run_probe(tuple(fixture))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
