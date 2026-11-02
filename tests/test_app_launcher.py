"""No-network/no-install launcher tests, including a real stdlib venv."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import venv

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mlg launch tests ')
        self.root = Path(self.temporary.name) / 'trusted checkout with spaces'
        self.root.mkdir()
        for name in ('start.py', 'install.py'):
            shutil.copyfile(ROOT / name, self.root / name)
        spec = importlib.util.spec_from_file_location('mlg_start_test', self.root / 'start.py')
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_environment_never_creates_or_installs(self):
        with mock.patch('subprocess.Popen', side_effect=AssertionError('must not spawn')):
            with mock.patch('subprocess.run', side_effect=AssertionError('must not launch')):
                self.assertEqual(self.launcher.main([]), 1)
        self.assertFalse((self.root / '.venv').exists())

    def _real_environment(self, *, fake_pip=True, fake_dependencies=True, editable=True):
        env = self.root / '.venv'
        # No ensurepip, installation, index, network or real ML dependencies.
        venv.EnvBuilder(with_pip=False, symlinks=os.name != 'nt').create(env)
        executable = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        site = Path(subprocess.check_output([str(executable), '-I', '-c',
            "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        if fake_pip: (site / 'pip.py').write_text('# Explicit synthetic discoverable pip fixture; never invoked.\n')
        if fake_dependencies:
            for name in ('torch', 'transformers', 'safetensors', 'numpy', 'PIL'):
                (site / (name + '.py')).write_text('# Explicit synthetic dependency for launcher-only tests.\n')
        package = self.root / 'looking_glass'; package.mkdir()
        (package / '__init__.py').write_text('')
        (package / 'app_runtime.py').write_text('# Synthetic app runtime for launcher tests.\n')
        (package / 'app.py').write_text(
            "import json, pathlib, sys\n"
            "def main():\n"
            "    report = dict(argv=sys.argv[1:], cwd=str(pathlib.Path.cwd()), prefix=sys.prefix, isolated=sys.flags.isolated)\n"
            "    pathlib.Path('launch-report.json').write_text(json.dumps(report))\n"
            "    return 0\n"
            "if __name__ == '__main__': raise SystemExit(main())\n")
        if editable: (site / 'trusted_test_editable.pth').write_text(str(self.root) + '\n')
        return env, executable, site

    def _invoke(self, args=(), **kwargs):
        return subprocess.run([sys.executable, str(self.root / 'start.py'), *args],
            cwd=Path(self.temporary.name), text=True, capture_output=True, timeout=30, **kwargs)

    def test_incomplete_environment_is_preserved_without_child(self):
        env = self.root / '.venv'; env.mkdir()
        (env / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        with mock.patch('subprocess.Popen', side_effect=AssertionError('incomplete environment must not spawn')):
            self.assertEqual(self.launcher.main([]), 1)
        self.assertTrue((env / 'pyvenv.cfg').is_file())

    def test_dry_run_and_help_do_not_probe_or_create(self):
        with mock.patch.object(self.launcher, '_installer_support', side_effect=AssertionError('no probe')):
            with mock.patch('subprocess.run', side_effect=AssertionError('no child')):
                self.assertEqual(self.launcher.main(['--dry-run', '--', '--no-browser', '--port', '8765']), 0)
        result = self._invoke(['--help'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / '.venv').exists())

    def test_real_editable_venv_spaces_and_hostile_pythonpath(self):
        env, executable, site = self._real_environment()
        hostile = Path(self.temporary.name) / 'untrusted import path'; hostile.mkdir()
        (hostile / 'sitecustomize.py').write_text("raise RuntimeError('PYTHONPATH injected')\n")
        (hostile / 'looking_glass').mkdir()
        (hostile / 'looking_glass' / '__init__.py').write_text("raise RuntimeError('wrong app')\n")
        environment = dict(os.environ, PYTHONPATH=str(hostile))
        # Invoke the real launcher parent isolated as well, so the test's injected
        # sitecustomize tests child isolation rather than poisoning test startup.
        result = subprocess.run([sys.executable, '-I', str(self.root / 'start.py'), '--',
            '--no-browser', '--port', '8765', '--models-dir', 'folder with spaces'],
            cwd=hostile, env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / 'launch-report.json').read_text())
        self.assertEqual(report['argv'], ['--no-browser', '--port', '8765', '--models-dir', 'folder with spaces'])
        self.assertEqual(Path(report['cwd']), self.root)
        self.assertEqual(Path(report['prefix']).resolve(), env.resolve())
        self.assertEqual(report['isolated'], 1)
        self.assertNotIn('PYTHONPATH injected', result.stderr)
        self.assertEqual(environment['PYTHONPATH'], str(hostile))

    def test_uninstalled_cwd_package_is_not_silently_used(self):
        self._real_environment(editable=False)
        result = self._invoke(['--', '--no-browser'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('readiness', result.stderr.lower())
        self.assertFalse((self.root / 'launch-report.json').exists())

    def test_wrong_editable_checkout_is_rejected(self):
        _, _, site = self._real_environment(editable=False)
        other = Path(self.temporary.name) / 'another checkout'; other.mkdir()
        shutil.copytree(self.root / 'looking_glass', other / 'looking_glass')
        (site / 'wrong_editable.pth').write_text(str(other) + '\n')
        result = self._invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not resolve to this checkout', result.stderr)
        self.assertFalse((self.root / 'launch-report.json').exists())

    def test_missing_dependency_is_actionable_and_never_launches(self):
        self._real_environment(fake_dependencies=False)
        result = self._invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing app dependencies', result.stderr)
        self.assertIn('--reuse-venv', result.stderr)
        self.assertFalse((self.root / 'launch-report.json').exists())

    def test_broken_dependency_import_is_actionable(self):
        _, _, site = self._real_environment()
        (site / 'torch.py').write_text("raise OSError('synthetic missing binary library')\n")
        result = self._invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('synthetic missing binary library', result.stderr)
        self.assertFalse((self.root / 'launch-report.json').exists())

    def test_no_pip_real_venv_is_preserved_without_ensurepip(self):
        env, _, _ = self._real_environment(fake_pip=False)
        result = self._invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('pip is unavailable', result.stderr)
        self.assertTrue((env / 'pyvenv.cfg').is_file())
        self.assertFalse((self.root / 'launch-report.json').exists())

    def test_system_site_venv_rejected_before_application(self):
        env, _, _ = self._real_environment()
        cfg = env / 'pyvenv.cfg'
        cfg.write_text(cfg.read_text().replace('include-system-site-packages = false', 'include-system-site-packages = true'))
        result = self._invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('disable system-site', result.stderr)
        self.assertFalse((self.root / 'launch-report.json').exists())

    @unittest.skipIf(os.name == 'nt', 'symlink creation permissions vary on Windows')
    def test_redirected_root_rejected_but_interpreter_symlink_is_valid(self):
        env, executable, _ = self._real_environment()
        self.assertTrue(executable.exists())
        self.assertTrue(executable.is_symlink(), 'legitimate POSIX venv interpreter symlink is exercised by positive tests')
        real = self.root / 'moved environment'; env.rename(real); env.symlink_to(real, target_is_directory=True)
        result = self._invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('redirected', result.stderr)
        self.assertTrue((real / 'pyvenv.cfg').is_file())

    def test_reused_probe_rejects_wrong_prefix_and_old_child(self):
        env, executable, _ = self._real_environment()
        support = self.launcher._installer_support()
        valid = dict(version=[3, 12, 0], prefix=str(env), base_prefix=str(env.parent), pip=True)
        for patch, message in ((dict(version=[3, 10, 9]), '3.11'),
                               (dict(prefix=str(env.parent)), 'prefix')):
            with self.subTest(patch=patch), mock.patch.object(support, '_probe_child', return_value={**valid, **patch}):
                with self.assertRaisesRegex(support.InstallError, message):
                    support.probe_environment(env, executable)

    def test_launcher_preserves_failure_code_no_retry_and_interrupt(self):
        self._real_environment()
        support = self.launcher._installer_support()
        with mock.patch.object(self.launcher, '_installer_support', return_value=support), mock.patch.object(support, 'probe_environment'):
            for status in (3, -9):
                with self.subTest(status=status), mock.patch.object(self.launcher.subprocess, 'run',
                        side_effect=[subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], status)]) as run:
                    self.assertEqual(self.launcher.main(['--', '--no-browser']), status if status > 0 else 1)
                    self.assertEqual(run.call_count, 2)
                    command = run.call_args.args[0]
                    self.assertEqual(command[1:], ['-I', '-m', 'looking_glass.app', '--no-browser'])
                    self.assertEqual(run.call_args.kwargs['cwd'], self.root)
            with mock.patch.object(self.launcher.subprocess, 'run', side_effect=KeyboardInterrupt):
                self.assertEqual(self.launcher.main([]), 130)
            with mock.patch.object(self.launcher.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 60)) as run:
                self.assertEqual(self.launcher.main([]), 1)
                self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
