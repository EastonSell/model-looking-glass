"""Real-process, CPU-only tests of the bounded number-study outer owner."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from scripts import number_cpu_outer as outer
except ImportError:
    outer = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'nt', 'the owner uses a Windows Job')
class NumberCpuOuterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(outer, 'bounded outer owner is missing')
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TEMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / 'repo'
        (self.repo / 'scripts').mkdir(parents=True)
        shutil.copyfile(ROOT / 'scripts' / 'number_cpu_gate.py', self.repo / 'scripts' / 'number_cpu_gate.py')
        shutil.copyfile(ROOT / 'scripts' / 'number_cpu_outer.py', self.repo / 'scripts' / 'number_cpu_outer.py')
        (self.repo / 'scripts' / 'pretrained_number_cpu_launcher.py').write_text('''
import hashlib, json, os, pathlib, subprocess, sys, time
assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
assert 'FAKE_UNREVIEWED_ENV' not in os.environ
assert sys.flags.ignore_environment and sys.flags.no_site and sys.flags.safe_path and sys.dont_write_bytecode
args = sys.argv[1:]
phase = args[4]
mode = args[args.index('--fake-mode') + 1] if '--fake-mode' in args else 'normal'
def value(name):
    for i, arg in enumerate(args):
        if arg == name: return args[i + 1]
        if arg.startswith(name + '='): return arg[len(name) + 1:]
    raise ValueError('missing option')
if mode == 'overflow':
    sys.stdout.write('x' * (300 * 1024)); sys.stdout.flush()
elif mode == 'deep-json':
    print('[' * 1200 + ']' * 1200)
elif mode == 'tree':
    target = args[args.index('--marker') + 1]
    subprocess.Popen([sys.executable, '-c', 'import pathlib,time,sys; time.sleep(2); pathlib.Path(sys.argv[1]).write_text("BAD")', target])
    time.sleep(10)
elif mode == 'short-tree':
    subprocess.run([sys.executable, '-c', 'pass'], check=True)
    data = b'fake runtime'
    pathlib.Path(value('--runtime-output')).write_bytes(data)
    print(json.dumps({'status': 'prepared', 'runtime_sha256': hashlib.sha256(data).hexdigest()}))
elif mode == 'prefix-write':
    (pathlib.Path(sys.pycache_prefix) / 'unexpected.bin').write_bytes(b'x' * (1024 * 1024 + 1))
    print(json.dumps({'status': 'prepared'}))
else:
    if mode == 'orphan':
        (pathlib.Path(os.environ['HF_HUB_CACHE']) / 'looking-glass-stage-fake').mkdir()
    if phase == 'run' and mode != 'no-records':
        attempt = pathlib.Path(value('--attempt-dir'))
        (attempt / 'private').mkdir(parents=True)
        complete = {'schema': 'mlg-number-discovery-completed-v1', 'record_count': 128,
                    'prediction_sha256': 'c' * 64, 'archive_sha256': 'd' * 64}
        (attempt / 'private' / 'completed.json').write_text(json.dumps(complete, sort_keys=True, separators=(',', ':')))
        terminal = {**complete, 'schema':'mlg-number-injection-launch-terminal-v1','status':'complete-local'}
        if mode == 'mismatched-records': terminal['prediction_sha256'] = 'e' * 64
        attempt.with_name(attempt.name + '.launcher-terminal.json').write_text(json.dumps(terminal, sort_keys=True, separators=(',', ':')))
    status = {'prepare': 'prepared', 'plan': 'plan-created', 'seal': 'commitment-created', 'run': 'complete-local'}[phase]
    result = {'status': status}
    if phase == 'run': result.update(prediction_sha256='c' * 64, archive_sha256='d' * 64)
    if phase == 'prepare':
        data = b'fake runtime'
        pathlib.Path(value('--runtime-output')).write_bytes(data)
        result['runtime_sha256'] = ('a' * 64 if mode == 'bad-output' else hashlib.sha256(data).hexdigest())
    if phase == 'plan':
        directory = pathlib.Path(value('--plan-dir')); directory.mkdir()
        data = b'fake plan'; (directory / 'plan.json').write_bytes(data)
        result['plan_sha256'] = hashlib.sha256(data).hexdigest()
    if phase == 'seal':
        directory = pathlib.Path(value('--custody-dir')); directory.mkdir()
        (directory / 'commitment.txt').write_bytes(('e' * 64 + '\\n').encode('ascii'))
        result['site_commitment'] = 'e' * 64
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))
''', encoding='utf8')
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), 'add', 'scripts'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-q', '-m', 'synthetic'], check=True)
        self.commit = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        self.workspace = self.root / 'workspace'
        self.cache = self.workspace / 'hf' / 'hub'
        self.cache.mkdir(parents=True)
        self.site = self.root / 'site'
        self.site.mkdir()
        self.attempt = self.workspace / 'attempt'
        self.out = self.workspace / 'outer'

    def run_owner(self, command, **kwargs):
        if command[0] == 'prepare' and not outer._option_values(command, '--runtime-output'):
            command = (*command, '--runtime-output', str(self.workspace / 'runtime.json'))
        if command[0] in ('prepare', 'run') and not outer._option_values(command, '--cache-dir'):
            command = (*command, '--cache-dir', str(self.cache))
        return outer.run_once(repo=self.repo, source_commit=self.commit,
            base_python=Path(sys._base_executable).resolve(), site_packages=self.site,
            workspace=self.workspace, hf_home=self.cache.parent, cache_dir=self.cache,
            output_dir=self.out, child_args=command, timeout=kwargs.get('timeout', 20))

    def check_bytes(self):
        total = sum(path.stat().st_size for path in self.out.rglob('*') if path.is_file())
        self.assertLessEqual(total, 1024 * 1024)

    def test_prepare_waits_for_exact_child_and_retains_bounded_receipt(self):
        with patch.dict(os.environ, {'FAKE_UNREVIEWED_ENV': 'must-not-inherit'}):
            result = self.run_owner(('prepare',))
        self.assertEqual((result['status'], result['exit_code'], result['active_processes']), ('succeeded', 0, 0))
        self.assertTrue(result['prefix_empty'])
        self.check_bytes()

    def test_prepare_rejects_mismatched_saved_runtime(self):
        result = self.run_owner(('prepare', '--fake-mode', 'bad-output'))
        self.assertEqual(result['status'], 'incomplete')
        self.check_bytes()

    def test_plan_and_commitment_outputs_are_read_back(self):
        plan = self.workspace / 'plan'
        result = self.run_owner(('plan', '--plan-dir', str(plan)))
        self.assertEqual(result['status'], 'succeeded')
        self.check_bytes()
        self.out = self.workspace / 'second-outer'
        result = self.run_owner(('seal', '--custody-dir', str(self.workspace / 'custody')))
        self.assertEqual(result['status'], 'succeeded')
        self.check_bytes()

    def test_run_requires_matching_independent_completion_records(self):
        result = self.run_owner(('run', '--attempt-dir', str(self.attempt), '--fake-mode', 'no-records'))
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['exit_code'], 0)
        self.check_bytes()

    def test_run_checks_matching_launcher_and_supervisor_completion(self):
        result = self.run_owner(('run', '--attempt-dir', str(self.attempt)))
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual((result['prediction_sha256'], result['archive_sha256']), ('c' * 64, 'd' * 64))
        self.check_bytes()

    def test_run_accepts_inline_attempt_path(self):
        result = self.run_owner(('run', '--attempt-dir=' + str(self.attempt)))
        self.assertEqual(result['status'], 'succeeded')
        self.check_bytes()

    def test_run_requires_cache_staging_free_space_before_spawn(self):
        with patch.object(outer.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()):
            with self.assertRaisesRegex(ValueError, 'staging space'):
                self.run_owner(('run', '--attempt-dir', str(self.attempt)))
        self.assertFalse(self.out.exists())

    def test_run_rejects_child_cache_different_from_checked_cache(self):
        other = self.workspace / 'other-cache'
        other.mkdir()
        with self.assertRaisesRegex(ValueError, 'child cache must match'):
            self.run_owner(('run', '--attempt-dir', str(self.attempt), '--cache-dir', str(other)))
        self.assertFalse(self.out.exists())

    def test_run_rejects_prior_cache_stage_before_spawn(self):
        (self.cache / 'looking-glass-stage-old').mkdir()
        with self.assertRaisesRegex(ValueError, 'prior stages'):
            self.run_owner(('run', '--attempt-dir', str(self.attempt)))
        self.assertFalse(self.out.exists())

    def test_run_retains_new_cache_stage_and_marks_incomplete(self):
        result = self.run_owner(('run', '--attempt-dir', str(self.attempt), '--fake-mode', 'orphan'))
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['cache_stage_clear'])
        self.assertTrue((self.cache / 'looking-glass-stage-fake').is_dir())
        self.check_bytes()

    def test_run_rejects_duplicate_attempt_path_before_spawn(self):
        with self.assertRaisesRegex(ValueError, 'unique run attempt'):
            self.run_owner(('run', '--attempt-dir', str(self.attempt), '--attempt-dir=' + str(self.attempt)))
        self.assertFalse(self.out.exists())

    def test_output_overflow_is_incomplete_and_bounded(self):
        result = self.run_owner(('prepare', '--fake-mode', 'overflow'))
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(result['stdout_truncated'])
        self.check_bytes()

    def test_deep_malformed_completion_retains_output_and_terminal(self):
        result = self.run_owner(('prepare', '--fake-mode', 'deep-json'))
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue((self.out / 'stdout.bin').read_bytes().startswith(b'['))
        self.assertEqual(json.loads((self.out / 'terminal.json').read_bytes())['status'], 'incomplete')
        self.check_bytes()

    def test_run_rejects_different_completion_digests(self):
        result = self.run_owner(('run', '--attempt-dir', str(self.attempt), '--fake-mode', 'mismatched-records'))
        self.assertEqual(result['status'], 'incomplete')
        self.check_bytes()

    def test_rejects_overlapping_child_artifact_directory(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.run_owner(('run', '--attempt-dir', str(self.out / 'attempt')))
        self.assertFalse(self.out.exists())

    def test_rejects_inline_overlapping_child_artifact(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.run_owner(('prepare', '--runtime-output=' + str(self.out / 'runtime.json')))
        self.assertFalse(self.out.exists())

    def test_rejects_abbreviated_child_path_option_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'abbreviated'):
            self.run_owner(('prepare', '--runtime-out=' + str(self.out / 'runtime.json')))
        self.assertFalse(self.out.exists())

    def test_rejects_output_nested_inside_child_artifact_directory(self):
        parent = self.workspace / 'artifact'
        parent.mkdir()
        self.out = parent / 'outer'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.run_owner(('run', '--attempt-dir', str(parent)))
        self.assertFalse(self.out.exists())

    def test_partial_output_write_reports_actual_persisted_bytes(self):
        write = outer._write_new
        def short_write(path, data, maximum):
            if path.name == 'stdout.bin':
                path.write_bytes(data[:5])
                raise OSError('partial evidence write')
            return write(path, data, maximum)
        with patch.object(outer, '_write_new', side_effect=short_write):
            result = self.run_owner(('prepare',))
        saved = (self.out / 'stdout.bin').read_bytes()
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['stdout_bytes_saved'], len(saved))
        self.assertEqual(result['stdout_sha256'], hashlib.sha256(saved).hexdigest())
        self.check_bytes()

    def test_missing_output_has_unknown_saved_length_and_digest(self):
        write = outer._write_new
        def failed_write(path, data, maximum):
            if path.name == 'stdout.bin':
                raise OSError('evidence file unavailable')
            return write(path, data, maximum)
        with patch.object(outer, '_write_new', side_effect=failed_write):
            result = self.run_owner(('prepare',))
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['stdout_bytes_saved'])
        self.assertIsNone(result['stdout_sha256'])
        self.check_bytes()

    def test_unexpected_large_prefix_is_incomplete_and_retained(self):
        result = self.run_owner(('prepare', '--fake-mode', 'prefix-write'))
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['prefix_empty'])
        self.assertEqual((self.out / 'pycache' / 'unexpected.bin').stat().st_size, 1024 * 1024 + 1)
        parent_bytes = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        self.assertLessEqual(parent_bytes, 1024 * 1024)

    def test_short_lived_descendant_does_not_prevent_clean_completion(self):
        result = self.run_owner(('prepare', '--fake-mode', 'short-tree'))
        self.assertEqual((result['status'], result['exit_code'], result['active_processes']), ('succeeded', 0, 0))
        self.check_bytes()

    def test_timeout_terminates_job_descendants_and_waits_for_empty_job(self):
        marker = self.root / 'grandchild-marker'
        result = self.run_owner(('prepare', '--fake-mode', 'tree', '--marker', str(marker)), timeout=0.7)
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(result['timed_out'])
        self.assertTrue(result['child_reaped'])
        self.assertEqual(result['active_processes'], 0)
        self.check_bytes()
        import time
        time.sleep(2.5)
        self.assertFalse(marker.exists())

    def test_job_assignment_failure_never_grants_study_execution(self):
        with patch.object(outer.WindowsJob, 'assign', side_effect=OSError('fake assignment failure')):
            result = self.run_owner(('run', '--attempt-dir', str(self.attempt)))
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('job_assign', result['reason_codes'])
        self.assertFalse((self.out / 'grant').exists())
        self.assertFalse(self.attempt.exists())
        self.check_bytes()

    def test_assignment_cancellation_reaps_child_and_records_fixed_flags(self):
        with patch.object(outer.WindowsJob, 'assign', side_effect=KeyboardInterrupt('stop')):
            with self.assertRaises(KeyboardInterrupt):
                self.run_owner(('run', '--attempt-dir', str(self.attempt)))
        terminal = json.loads((self.out / 'terminal.json').read_bytes())
        self.assertEqual(terminal['status'], 'incomplete')
        self.assertIn('job_assign', terminal['reason_codes'])
        self.assertIn('cancelled', terminal['reason_codes'])
        self.assertTrue(terminal['child_reaped'])
        self.assertIsNone(outer._OWNER_SLOT[0])
        self.check_bytes()

    def test_reader_start_failure_never_grants_study_execution(self):
        self.addCleanup(lambda: outer._OWNER_SLOT.__setitem__(0, None))  # Mock prevented thread creation.
        with patch.object(outer, '_start_capture_thread', side_effect=OSError('reader setup failed')):
            result = self.run_owner(('run', '--attempt-dir', str(self.attempt)))
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse((self.out / 'grant').exists())
        self.assertFalse(self.attempt.exists())
        self.assertIsNotNone(outer._OWNER_SLOT[0])
        self.check_bytes()

    def test_join_cancellation_persists_evidence_and_chains_setup_error(self):
        self.addCleanup(lambda: outer._OWNER_SLOT.__setitem__(0, None))  # Test joins the real reader first.
        start = outer._start_capture_thread
        class InterruptedJoin:
            def __init__(self, thread): self.thread = thread
            def join(self, timeout=None):
                self.thread.join(timeout)
                raise KeyboardInterrupt('reader join interrupted')
            def is_alive(self): return self.thread.is_alive()
        def interrupt_first(stream, state, context, slot):
            thread = start(stream, state, context, slot)
            if not getattr(interrupt_first, 'used', False):
                interrupt_first.used = True
                return InterruptedJoin(thread)
            return thread
        with patch.object(outer, '_start_capture_thread', side_effect=interrupt_first), \
             patch.object(outer.WindowsJob, 'assign', side_effect=OSError('assignment failed')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_owner(('run', '--attempt-dir', str(self.attempt)))
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(json.loads((self.out / 'terminal.json').read_bytes())['status'], 'incomplete')
        self.assertTrue((self.out / 'stdout.bin').exists())
        self.assertIsNotNone(outer._OWNER_SLOT[0])
        self.check_bytes()

    def test_interrupted_job_constructor_still_closes_rooted_handle(self):
        original = outer.WindowsJob.__init__
        instances = []
        def interrupted(instance):
            original(instance)
            instances.append(instance)
            raise OSError('constructor return interrupted')
        with patch.object(outer.WindowsJob, '__init__', interrupted):
            result = self.run_owner(('prepare',))
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(instances), 1)
        self.assertIsNone(instances[0].handle)
        self.assertFalse((self.out / 'grant').exists())
        self.check_bytes()

    def test_job_constructor_preserves_original_cancellation_when_close_fails(self):
        real_close = outer.WindowsJob.close
        jobs = []
        def bad_close(job):
            jobs.append(job)
            raise OSError('close failed')
        try:
            with patch.object(outer.os, 'get_handle_inheritable', side_effect=KeyboardInterrupt('stop')), \
                 patch.object(outer.WindowsJob, 'close', bad_close):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    outer.WindowsJob()
            self.assertIsInstance(caught.exception.__cause__, OSError)
        finally:
            for job in jobs: real_close(job)

    def test_job_constructor_preserves_new_cleanup_cancellation_over_ordinary_failure(self):
        real_close = outer.WindowsJob.close
        jobs = []
        def interrupted_close(job):
            jobs.append(job)
            raise SystemExit(23)
        try:
            with patch.object(outer.os, 'get_handle_inheritable', side_effect=OSError('setup failed')), \
                 patch.object(outer.WindowsJob, 'close', interrupted_close):
                with self.assertRaises(SystemExit) as caught:
                    outer.WindowsJob()
            self.assertEqual(caught.exception.code, 23)
            self.assertIsInstance(caught.exception.__cause__, OSError)
        finally:
            for job in jobs: real_close(job)

    def test_cli_preserves_system_exit_code(self):
        arguments = []
        for name in ('repo', 'source-commit', 'base-python', 'site-packages', 'workspace',
                     'hf-home', 'cache-dir', 'output-dir'):
            arguments.extend(('--' + name, 'synthetic'))
        arguments.extend(('--timeout', '1', '--', 'prepare'))
        with patch.object(outer, 'run_once', side_effect=SystemExit(23)):
            self.assertEqual(outer.main(arguments), 23)

    def test_cli_does_not_print_private_system_exit_payload(self):
        program = '''
from scripts import number_cpu_outer as outer
def fail(**kwargs): raise SystemExit('FABRICATED_PRIVATE_SITE')
outer.run_once = fail
arguments = []
for name in ('repo', 'source-commit', 'base-python', 'site-packages', 'workspace',
             'hf-home', 'cache-dir', 'output-dir'):
    arguments.extend(('--' + name, 'synthetic'))
arguments.extend(('--timeout', '1', '--', 'prepare'))
raise SystemExit(outer.main(arguments))
'''
        completed = subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(completed.returncode, 1)
        self.assertNotIn(b'FABRICATED_PRIVATE_SITE', completed.stdout + completed.stderr)

    def test_cli_does_not_echo_invalid_argument_value(self):
        arguments = []
        for name in ('repo', 'source-commit', 'base-python', 'site-packages', 'workspace',
                     'hf-home', 'cache-dir', 'output-dir'):
            arguments.extend(('--' + name, 'synthetic'))
        arguments.extend(('--timeout', 'FABRICATED_PRIVATE_SITE'))
        completed = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / 'number_cpu_outer.py'),
                                    *arguments], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(completed.returncode, 2)
        self.assertNotIn(b'FABRICATED_PRIVATE_SITE', completed.stdout + completed.stderr)

    def test_uncertain_job_query_blocks_another_owner_in_this_process(self):
        actual_query = outer.WindowsJob.active
        def fail_after_empty(instance):
            self.assertEqual(actual_query(instance), 0)
            raise OSError('job query is unavailable')
        self.addCleanup(lambda: outer._OWNER_SLOT.__setitem__(0, None))  # The real job was observed empty.
        with patch.object(outer.WindowsJob, 'active', fail_after_empty):
            result = self.run_owner(('prepare',))
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['active_processes'])
        self.assertIsNotNone(outer._OWNER_SLOT[0])
        with self.assertRaisesRegex(RuntimeError, 'uncertain cleanup'):
            self.run_owner(('prepare',))
        self.check_bytes()

    def test_interrupted_spawn_return_keeps_attempt_ungranted_and_blocks_retry(self):
        original = outer.subprocess.Popen.__init__
        def interrupted(process, *args, **kwargs):
            original(process, *args, **kwargs)
            if kwargs.get('creationflags', 0) & subprocess.CREATE_NO_WINDOW:
                raise OSError('spawn return interrupted')
        self.addCleanup(lambda: outer._OWNER_SLOT.__setitem__(0, None))  # The captured child is reaped below.
        with patch.object(outer.subprocess.Popen, '__init__', interrupted):
            result = self.run_owner(('prepare',))
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['spawn_returned'])
        self.assertTrue(result['child_reaped'])
        self.assertFalse((self.out / 'grant').exists())
        self.assertIsNotNone(outer._OWNER_SLOT[0])
        with self.assertRaisesRegex(RuntimeError, 'uncertain cleanup'):
            self.run_owner(('prepare',))
        self.check_bytes()

    def test_unreaped_unassigned_child_and_readers_remain_rooted(self):
        init, kill, wait, start = (outer.subprocess.Popen.__init__, outer.subprocess.Popen.kill,
                                   outer.subprocess.Popen.wait, outer._start_capture_thread)
        captured = []
        def track(process, *args, **kwargs):
            init(process, *args, **kwargs)
            if kwargs.get('creationflags', 0) & subprocess.CREATE_NO_WINDOW:
                captured.append(process)
        def no_kill(process, *args, **kwargs):
            if process in captured: raise OSError('synthetic kill failure')
            return kill(process, *args, **kwargs)
        def no_wait(process, *args, **kwargs):
            if process in captured: raise OSError('synthetic wait failure')
            return wait(process, *args, **kwargs)
        class NonblockingReader:
            def __init__(self, thread): self.thread = thread
            def join(self, timeout=None): pass
            def is_alive(self): return self.thread.is_alive()
        def nonblocking(stream, state, context, slot):
            return NonblockingReader(start(stream, state, context, slot))
        try:
            with patch.object(outer.subprocess.Popen, '__init__', track), \
                 patch.object(outer.subprocess.Popen, 'kill', no_kill), \
                 patch.object(outer.subprocess.Popen, 'wait', no_wait), \
                 patch.object(outer, '_start_capture_thread', side_effect=nonblocking), \
                 patch.object(outer.WindowsJob, 'assign', side_effect=OSError('assignment failed')):
                result = self.run_owner(('prepare',))
            self.assertEqual(result['status'], 'incomplete')
            self.assertFalse(result['child_reaped'])
            self.assertEqual(len(captured), 1)
            self.assertIs(outer._OWNER_SLOT[0][1], captured[0])
            self.assertIsNotNone(outer._OWNER_SLOT[0][2])
            self.assertIsNotNone(outer._OWNER_SLOT[0][3])
            self.assertIsNone(captured[0].poll())
            self.assertFalse((self.out / 'grant').exists())
            self.check_bytes()
        finally:
            if captured:
                if captured[0].poll() is None: captured[0].kill()
                captured[0].wait(timeout=5)
            if outer._OWNER_SLOT[0] is not None:
                for thread in outer._OWNER_SLOT[0][2:]:
                    if thread is not None: thread.join(timeout=5)
            outer._OWNER_SLOT[0] = None


if __name__ == '__main__':
    unittest.main()
