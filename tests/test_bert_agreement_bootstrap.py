"""Isolated CPU study bootstrap starts with explicit GPU masking."""

import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import sysconfig
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'bert_agreement_bootstrap.py'


class BertBootstrapTest(unittest.TestCase):
    def run_check(self, directory, visibility):
        prefix = Path(directory) / 'prefix'
        prefix.mkdir()
        env = os.environ.copy()
        if visibility is None:
            env.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            env['CUDA_VISIBLE_DEVICES'] = visibility
        command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                   str(SCRIPT), '--bootstrap-site', sysconfig.get_path('purelib'), '--check-only']
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)

    def test_masked_cpu_check_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_check(directory, '-1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'status': 'ok', 'torch_imported': False})

    def test_unmasked_cuda_fails_before_project_import(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_check(directory, None)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('torch', result.stdout)

    def test_sourceless_dependency_is_rejected_without_executing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bytecode, sentinel = root / 'isolated_dependency.py', root / 'isolated_dependency.pyc', root / 'executed'
            source.write_text(f'open({str(sentinel)!r}, "w").write("bad")\n', encoding='utf-8')
            py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
            source.unlink()
            code = ('import runpy,sys; m=runpy.run_path(sys.argv[1]); '
                    'm["_install_bytecode_guard"](); sys.path.append(sys.argv[2]); import isolated_dependency')
            result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code,
                                     str(SCRIPT), str(root)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('sourceless', result.stderr.lower())
            self.assertFalse(sentinel.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows junction regression')
    def test_project_package_junction_is_rejected_before_initializer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, link, sentinel = root / 'target', root / 'package', root / 'executed'
            target.mkdir()
            (target / '__init__.py').write_text(f'open({str(sentinel)!r}, "w").write("bad")\n', encoding='utf-8')
            created = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                                     capture_output=True, text=True, timeout=30)
            if created.returncode:
                self.skipTest('junction creation unavailable on this filesystem')
            code = 'import runpy,sys,pathlib; m=runpy.run_path(sys.argv[1]); m["_load_package"](pathlib.Path(sys.argv[2]))'
            result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code,
                                     str(SCRIPT), str(link)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(sentinel.exists())


if __name__ == '__main__':
    unittest.main()
