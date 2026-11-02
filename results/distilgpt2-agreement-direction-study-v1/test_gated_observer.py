"""Model-free integration checks of the gated agreement-study observer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
OBSERVER = HERE / 'gated_observer.py'
SUPPORT = HERE / 'observer_support.py'
SPEC = importlib.util.spec_from_file_location('_test_observer_support', SUPPORT)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
SUPERVISION = ('looking_glass/__init__.py', 'looking_glass/supervisor.py',
               'looking_glass/cpu_study_supervision.py', 'looking_glass/device_lease.py',
               'looking_glass/device_identity.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GatedObserverTests(unittest.TestCase):
    def prepare(self, root, *, validate=False, omit_custody=False, fail=False, bad_mode=False):
        source = Path(os.environ.get('DIRECTION_SUPERVISOR_SOURCE', str(HERE.parents[1]))).resolve()
        repo, study, venv = root / 'repo', root / 'study', root / 'venv'
        (repo / 'looking_glass').mkdir(parents=True)
        (repo / 'tools').mkdir()
        study.mkdir()
        (venv / 'Lib/site-packages').mkdir(parents=True)
        for path in SUPERVISION:
            shutil.copyfile(source / path, repo / path)
        code = '''import argparse,json,os,sys,uuid
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('study-root','venv','child-custody','output-dir','prelaunch-sha256'):
    p.add_argument('--'+name, required=True)
p.add_argument('--validate-only', action='store_true')
p.add_argument('--omit-custody', action='store_true')
p.add_argument('--fail', action='store_true')
p.add_argument('--bad-mode', action='store_true')
a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from looking_glass.cpu_study_supervision import claim_cpu_study
session,lease=claim_cpu_study(a.study_root,lease_timeout=30)
os.environ['CUDA_VISIBLE_DEVICES']='-1'
if not a.omit_custody:
    mode='validate_only' if a.validate_only else 'study'
    if a.bad_mode: mode='incorrect'
    Path(a.child_custody).write_text(json.dumps({'schema':'agreement-direction-supervised-child-v1','run_id':session.run_id,'owner_pid':os.getpid(),'prelaunch_sha256':a.prelaunch_sha256,'mode':mode}))
if a.fail:
    raise RuntimeError('fabricated workload failure')
if not a.validate_only:
    run=Path(a.output_dir); run.mkdir()
    (run/'attempt').mkdir()
    (run/'attempt'/'terminal.json').write_text(json.dumps({'status':'complete_candidate'}))
    (run/'entry-result.json').write_text(json.dumps({'status':'complete_candidate'}))
session.attest_clean(lease)
'''
        entry = repo / 'tools/fake_entry.py'
        entry.write_text(code, encoding='utf-8')
        executor = repo / 'looking_glass/fake_executor.py'
        executor.write_text('# fabricated only\n', encoding='utf-8')
        (repo / '.gitignore').write_text('uuid.py\nuuid.pyc\n', encoding='utf-8')
        def git(*args):
            return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        git('init')
        git('add', '.')
        git('-c', 'user.name=Gated observer test', '-c', 'user.email=test@example.invalid',
            'commit', '-m', 'fabricated owner')
        base_python = Path(sys._base_executable).resolve()
        pre = {'owner': {'source_revision': git('rev-parse', 'HEAD'),
                         'source_tree': git('show', '-s', '--format=%T', 'HEAD'),
                         'executor_path': 'looking_glass/fake_executor.py', 'executor_sha256': digest(executor),
                         'entry_path': 'tools/fake_entry.py', 'entry_script_sha256': digest(entry),
                         'supervision_sha256': {path: digest(repo / path) for path in SUPERVISION}},
               'runtime': {'environment': HELPER.FIXED_ENVIRONMENT,
                           'inherited_environment_sha256': HELPER.inherited_environment_hashes(),
                           'base_python_sha256': digest(base_python), 'claim_cuda_visibility': ''},
               'observer': {'sha256': digest(OBSERVER), 'support_sha256': digest(SUPPORT)}}
        pre_path = root / 'prelaunch.json'
        pre_path.write_text(json.dumps(pre), encoding='utf-8')
        argv = ['--output-dir', '{run}', '--prelaunch-sha256', '{prelaunch_sha256}']
        if validate:
            argv.append('--validate-only')
        if omit_custody:
            argv.append('--omit-custody')
        if fail:
            argv.append('--fail')
        if bad_mode:
            argv.append('--bad-mode')
        argument_path = root / 'arguments.json'
        argument_path.write_text(json.dumps(argv), encoding='utf-8')
        attempt = root / 'attempt'
        command = [sys.executable, '-I', '-S', '-B', str(OBSERVER), '--repository', str(repo),
                   '--base-python', str(base_python), '--venv', str(venv), '--study-root', str(study),
                   '--prelaunch', str(pre_path), '--prelaunch-sha256', digest(pre_path),
                   '--entry', 'tools/fake_entry.py', '--argv-json', str(argument_path),
                   '--attempt', str(attempt), '--timeout-seconds', '20']
        if validate:
            command.append('--validation-only')
        return command, attempt, study

    def run_case(self, root, **kwargs):
        command, attempt, study = self.prepare(root, **kwargs)
        completed = subprocess.run(command, capture_output=True, text=True, timeout=50)
        return completed, attempt, study

    def test_success_uses_gated_child_and_matches_custody(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, attempt, _ = self.run_case(Path(directory))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads((attempt / 'supervisor.json').read_bytes())
            self.assertTrue(receipt['clean'])
            self.assertTrue(receipt['child_binding_matches'])
            self.assertTrue(receipt['supervision_result']['clean'])
            self.assertEqual(receipt['supervision_result']['exit_code'], 0)
            self.assertEqual(receipt['completion_statuses'],
                             {'entry_result': 'complete_candidate', 'terminal': 'complete_candidate'})

    def test_validation_is_clean_without_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, attempt, _ = self.run_case(Path(directory), validate=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads((attempt / 'supervisor.json').read_bytes())
            self.assertTrue(receipt['validation_only'])
            self.assertTrue(receipt['child_binding_matches'])
            self.assertFalse((attempt / 'run').exists())

    def test_missing_child_binding_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, attempt, _ = self.run_case(Path(directory), omit_custody=True)
            self.assertNotEqual(completed.returncode, 0)
            receipt = json.loads((attempt / 'supervisor.json').read_bytes())
            self.assertFalse(receipt['clean'])
            self.assertFalse(receipt['child_binding_matches'])

    def test_wrong_child_mode_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, attempt, _ = self.run_case(Path(directory), bad_mode=True)
            self.assertNotEqual(completed.returncode, 0)
            receipt = json.loads((attempt / 'supervisor.json').read_bytes())
            self.assertFalse(receipt['child_binding_matches'])

    def test_workload_failure_retains_unsafe_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, attempt, study = self.run_case(Path(directory), fail=True)
            self.assertNotEqual(completed.returncode, 0)
            receipt = json.loads((attempt / 'supervisor.json').read_bytes())
            self.assertFalse(receipt['clean'])
            self.assertTrue((study / 'cpu-study.unsafe.json').is_file())

    def test_interruption_at_child_creation_grants_no_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, attempt, study = self.prepare(root)
            driver = '''import importlib.util,subprocess,sys
old=subprocess.Popen.__init__
def interrupted(self,*args,**kwargs):
    old(self,*args,**kwargs)
    if args and isinstance(args[0],list) and '--study-root' in args[0]:
        raise KeyboardInterrupt()
subprocess.Popen.__init__=interrupted
s=importlib.util.spec_from_file_location('_observer',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
try: m.main(sys.argv[2:])
except KeyboardInterrupt: sys.exit(130)
'''
            wrapped = [sys.executable, '-I', '-S', '-B', '-c', driver, str(OBSERVER)] + command[5:]
            completed = subprocess.run(wrapped, capture_output=True, text=True, timeout=50)
            self.assertEqual(completed.returncode, 130, completed.stderr)
            receipt = json.loads((attempt / 'supervisor.json').read_bytes())
            self.assertFalse(receipt['clean'])
            self.assertEqual(receipt['supervision_error_type'], 'KeyboardInterrupt')
            self.assertFalse((attempt / 'child-custody.json').exists())
            self.assertFalse((attempt / 'run').exists())
            self.assertTrue((study / 'cpu-study.unsafe.json').is_file())

    def test_ignored_source_and_sourceless_shadows_do_not_execute(self):
        for suffix in ('.py', '.pyc'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                command, attempt, _ = self.prepare(root, validate=True)
                repo, marker = root / 'repo', root / 'shadow-ran'
                source = repo / 'uuid.py'
                source.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ").write_text('executed')\n", encoding='utf-8')
                if suffix == '.pyc':
                    py_compile.compile(str(source), cfile=str(repo / 'uuid.pyc'), doraise=True)
                    source.unlink()
                completed = subprocess.run(command, capture_output=True, text=True, timeout=50)
                self.assertFalse(marker.exists(), completed.stderr)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(json.loads((attempt / 'supervisor.json').read_bytes())['clean'])

    def test_assume_unchanged_source_must_match_committed_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, attempt, _ = self.prepare(root, validate=True)
            repo = root / 'repo'
            source = repo / 'looking_glass/supervisor.py'
            source.write_text(source.read_text(encoding='utf-8') + '\n# fabricated local modification\n', encoding='utf-8')
            subprocess.run(['git', 'update-index', '--assume-unchanged', 'looking_glass/supervisor.py'],
                           cwd=repo, check=True, capture_output=True)
            pre_path = root / 'prelaunch.json'
            pre = json.loads(pre_path.read_bytes())
            pre['owner']['supervision_sha256']['looking_glass/supervisor.py'] = digest(source)
            pre_path.write_text(json.dumps(pre), encoding='utf-8')
            command[command.index('--prelaunch-sha256') + 1] = digest(pre_path)
            completed = subprocess.run(command, capture_output=True, text=True, timeout=50)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((attempt / 'child-custody.json').exists())


if __name__ == '__main__':
    unittest.main()
