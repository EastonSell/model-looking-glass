"""Model-free observer tests: fabricated markers plus one real harmless child."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import observe_distilbert_color_supervisor_v1 as observer


SOURCE = 'a' * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_synthetic_stage(root, pid, *, marker_pid=None, corrupt=False):
    root = Path(root)
    root.mkdir(exist_ok=True)
    stage = root / 'screen'
    stage.mkdir(exist_ok=True)
    (root / 'run.json').write_bytes(observer._json_bytes({'source_commit': SOURCE}))
    files = {}
    for name in observer.FILES:
        raw = ('synthetic ' + name + '\n').encode()
        (stage / name).write_bytes(raw)
        files[name] = sha(raw)
    if corrupt:
        (stage / 'events.jsonl').write_bytes(b'changed\n')
    marker = {'schema': observer.VALIDATION_SCHEMA, 'phase': 'screen',
              'source_commit': SOURCE, 'supervisor_pid': pid if marker_pid is None else marker_pid,
              'files_sha256': files}
    (stage / 'validated.json').write_bytes(observer._json_bytes(marker))


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.repo = self.base / 'repo'
        self.cache = self.base / 'cache'
        self.site = self.base / 'site'
        self.scratch = self.base / 'scratch'
        self.repo.mkdir()
        self.cache.mkdir()
        self.site.mkdir()
        self.scratch.mkdir()
        self.root = self.base / 'run'
        self.observer_dir = self.base / 'observation'
        self.prelaunch = self.base / 'prelaunch.json'
        self.prelaunch.write_bytes(b'{}\n')
        self.args = dict(python=sys._base_executable, site=self.site, repo=self.repo, cache=self.cache, scratch=self.scratch,
                         root=self.root, phase='screen', source=SOURCE,
                         prelaunch=self.prelaunch, prelaunch_sha256=sha(b'{}\n'),
                         observer_dir=self.observer_dir, stage_timeout=5, observer_timeout=10)

    def tearDown(self):
        self.tmp.cleanup()

    def run_fake(self, *, marker_pid=None, corrupt=False, code=0, interrupt=False,
                 bytecode=False, existing=False):
        test = self

        class FakeChild:
            pid = 4242
            def __init__(self, argv, **kwargs):
                self.argv, self.kwargs = argv, kwargs
                self.killed = False
                self.waits = 0
                self.returned = False
                test.child = self
                write_synthetic_stage(test.root, self.pid, marker_pid=marker_pid, corrupt=corrupt)
                if bytecode:
                    (test.observer_dir / 'bytecode' / 'unexpected').write_text('x')
                if existing:
                    (test.root / 'screen' / 'supervisor-observation.json').write_bytes(b'existing')

            def wait(self, timeout=None):
                self.waits += 1
                if interrupt and self.waits == 1:
                    raise KeyboardInterrupt('fabricated wait interruption')
                self.returned = True
                return -9 if self.killed else code

            def poll(self):
                return (0 if self.returned else None)

            def kill(self):
                self.killed = True

        with patch.object(observer.subprocess, 'Popen', FakeChild):
            return observer.observe(**self.args)

    def failure(self):
        return json.loads((self.observer_dir / 'observer-failure.json').read_bytes())

    def test_success_binds_pid_marker_run_and_command(self):
        receipt = self.run_fake()
        stage = self.root / 'screen'
        self.assertEqual(json.loads((stage / 'supervisor-observation.json').read_bytes()), receipt)
        self.assertEqual(receipt['supervisor_pid'], 4242)
        self.assertEqual(receipt['returncode'], 0)
        self.assertEqual(receipt['validation_sha256'], sha((stage / 'validated.json').read_bytes()))
        self.assertEqual(receipt['run_sha256'], sha((self.root / 'run.json').read_bytes()))
        command_raw = (self.observer_dir / 'command.json').read_bytes()
        self.assertEqual(receipt['command_sha256'], sha(command_raw))
        self.assertEqual(receipt['observer_external_bytes'], len(command_raw) +
                         len((self.observer_dir / 'observer-start.json').read_bytes()))
        self.assertLessEqual(receipt['observer_external_bytes'] +
                             len((stage / 'supervisor-observation.json').read_bytes()), 16 * 1024)
        self.assertFalse((stage / 'supervisor-observation-failure.json').exists())
        self.assertEqual(self.child.argv[1:5], ['-E', '-s', '-B', '-X'])
        self.assertTrue(self.child.argv[5].startswith('pycache_prefix='))
        self.assertEqual(self.child.argv[6:10], ['-c', observer.BOOTSTRAP, str(self.repo), str(self.site)])
        self.assertIn('--site', self.child.argv)
        self.assertEqual(self.child.kwargs['env']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertNotIn('SYSTEMDRIVE', self.child.kwargs['env'])
        self.assertNotIn('PYTHONPATH', self.child.kwargs['env'])
        self.assertTrue(observer._empty(self.observer_dir / 'bytecode'))

    def test_wrong_pid_and_changed_stage_file_fail(self):
        with self.assertRaisesRegex(ValueError, 'marker'):
            self.run_fake(marker_pid=123)
        self.assertFalse((self.root / 'screen' / 'supervisor-observation.json').exists())
        self.assertEqual(self.failure()['direct_process_cleanup'], 'direct_process_reaped')

    def test_changed_stage_file_fails(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.run_fake(corrupt=True)
        self.assertFalse((self.root / 'screen' / 'supervisor-observation.json').exists())

    def test_nonzero_exit_never_publishes_receipt(self):
        with self.assertRaisesRegex(RuntimeError, 'exit zero'):
            self.run_fake(code=1)
        self.assertFalse((self.root / 'screen' / 'supervisor-observation.json').exists())
        self.assertEqual(self.failure()['status'], 'invalid-incomplete')
        self.assertEqual(self.failure()['supervisor_pid'], 4242)
        self.assertEqual(self.failure()['returncode'], 1)

    def test_interrupted_wait_reaps_only_direct_handle(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(interrupt=True)
        self.assertTrue(self.child.killed)
        self.assertEqual(self.child.waits, 2)
        self.assertFalse((self.root / 'screen' / 'supervisor-observation.json').exists())
        self.assertEqual(self.failure()['direct_process_cleanup'], 'direct_process_reaped')
        self.assertEqual(self.failure()['returncode'], -9)
        self.assertIn('does not establish whether its child stopped', self.failure()['scope'])

    def test_unknown_spawn_remains_unknown(self):
        with patch.object(observer.subprocess, 'Popen', side_effect=OSError('unknown spawn')):
            with self.assertRaises(OSError):
                observer.observe(**self.args)
        self.assertEqual(self.failure()['direct_process_cleanup'], 'not_observed')
        self.assertIsNone(self.failure()['supervisor_pid'])
        self.assertIsNone(self.failure()['returncode'])
        self.assertFalse((self.root / 'screen' / 'supervisor-observation.json').exists())

    def test_scratch_overlap_is_rejected_before_observation(self):
        for overlap in (self.base, self.repo, self.cache, self.site):
            with self.subTest(scratch=overlap), self.assertRaisesRegex(ValueError, 'disjoint'):
                observer.observe(**{**self.args, 'scratch': overlap})
        self.assertFalse(self.observer_dir.exists())

    def test_unexpected_bytecode_and_existing_receipt_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'bytecode'):
            self.run_fake(bytecode=True)
        self.assertFalse((self.root / 'screen' / 'supervisor-observation.json').exists())

    def test_existing_receipt_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            self.run_fake(existing=True)
        self.assertEqual((self.root / 'screen' / 'supervisor-observation.json').read_bytes(), b'existing')
        self.assertTrue((self.root / 'screen' / 'supervisor-observation-failure.json').exists())

    def test_interrupt_after_receipt_write_leaves_stage_sentinel(self):
        original = observer._write

        def write_then_interrupt(path, raw):
            original(path, raw)
            if Path(path).name == 'supervisor-observation.json':
                raise KeyboardInterrupt('after receipt write')

        with patch.object(observer, '_write', side_effect=write_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fake()
        stage = self.root / 'screen'
        self.assertTrue((stage / 'supervisor-observation.json').is_file())
        self.assertTrue((stage / 'supervisor-observation-failure.json').is_file())
        failure_path = self.observer_dir / 'observer-failure.json'
        external = sum((self.observer_dir / name).stat().st_size for name in
                       ('command.json', 'observer-start.json', 'observer-failure.json'))
        self.assertEqual(json.loads(failure_path.read_bytes())['observer_external_bytes'], external)

    def test_interrupt_after_sentinel_removal_is_committed_not_failed(self):
        original = Path.unlink

        def unlink_then_interrupt(path, *args, **kwargs):
            original(path, *args, **kwargs)
            if path.name == 'supervisor-observation-failure.json':
                raise KeyboardInterrupt('after observation commit')

        with patch.object(Path, 'unlink', unlink_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fake()
        stage = self.root / 'screen'
        self.assertTrue((stage / 'supervisor-observation.json').is_file())
        self.assertFalse((stage / 'supervisor-observation-failure.json').exists())
        self.assertFalse((self.observer_dir / 'observer-failure.json').exists())

    def test_real_harmless_direct_process_exit_and_pid(self):
        # The patch substitutes only a harmless stdlib process for the model supervisor.
        child_script = self.base / 'synthetic_supervisor.py'
        child_script.write_text('''import hashlib,json,os,pathlib,sys\nroot=pathlib.Path(sys.argv[1]); source=sys.argv[2]\nroot.mkdir(); stage=root/"screen"; stage.mkdir()\ndef enc(x): return (json.dumps(x,sort_keys=True,separators=(",",":"))+"\\n").encode()\n(root/"run.json").write_bytes(enc({"source_commit":source}))\nfiles={}\nfor name in ("claim.json","prelaunch.json","worker-start.json","events.jsonl","result.json","exit.json"):\n raw=("synthetic "+name+"\\n").encode(); (stage/name).write_bytes(raw); files[name]=hashlib.sha256(raw).hexdigest()\n(stage/"validated.json").write_bytes(enc({"schema":"distilbert-color-stage-validation-v1","phase":"screen","source_commit":source,"supervisor_pid":os.getpid(),"files_sha256":files}))\n''', encoding='utf-8')
        real_popen = subprocess.Popen
        captured = []

        def start_harmless(argv, **kwargs):
            process = real_popen([sys._base_executable, '-I', '-S', '-B', str(child_script),
                                  str(self.root), SOURCE], **kwargs)
            captured.append(process)
            return process

        with patch.object(observer.subprocess, 'Popen', start_harmless):
            receipt = observer.observe(**self.args)
        self.assertEqual(receipt['supervisor_pid'], captured[0].pid)
        self.assertEqual(captured[0].poll(), 0)
        self.assertEqual(receipt['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
