"""Real synthetic child processes only; no model, site or study split."""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    outer = importlib.import_module('looking_glass.injection_outer_observer')
except ModuleNotFoundError as error:
    if error.name != 'looking_glass.injection_outer_observer':
        raise
    outer = None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FakeJob:
    active_override = None

    def __init__(self):
        self.process = None
        self.closed = False

    def assign(self, process):
        self.process = process

    def _accounting(self):
        return type('Counts', (), {'active_processes': self.active_override if self.active_override is not None
                                   else int(self.process.poll() is None)})()

    def terminate_and_wait(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)

    def close(self):
        self.closed = True


class OuterObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'config.json'
        self.config.write_bytes(b'{"config":"fabricated"}')
        self.operation = self.root / 'operation.json'
        self.operation.write_bytes(b'{"operation":"fabricated"}')
        self.terminal = self.root / 'supervisor-terminal.json'
        self.output = self.root / 'outer'
        self.source_sha = 'a' * 64
        self.start = b'PRIVATE-FAKE-GATE\n'

    def command(self, exit_code=0, mutate_config=False):
        code = ("import pathlib,sys; assert sys.stdin.buffer.readline()==b'PRIVATE-FAKE-GATE\\n'; "
                "pathlib.Path(sys.argv[1]).write_bytes(b'fabricated-terminal'); "
                + ("pathlib.Path(sys.argv[2]).write_bytes(b'changed'); " if mutate_config else '')
                + f"sys.exit({exit_code})")
        return [sys.executable, '-B', '-c', code, str(self.terminal), str(self.config)]

    def observe(self, command, **options):
        kwargs = dict(purpose='preparation', command=command, output=self.output,
            environment=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'), config_path=self.config, config_sha256=digest(self.config.read_bytes()),
            operation_path=self.operation, operation_sha256=digest(self.operation.read_bytes()),
            command_sha256=digest(outer._canonical(command)), supervisor_source_sha256=self.source_sha,
            terminal_path=self.terminal, start_token=self.start, timeout=10,
            source_check=lambda: True, protect_process=lambda process: True,
            terminal_check=lambda raw, purpose, pid, birth: raw == b'fabricated-terminal' and pid > 0 and birth > 0,
            job_type=FakeJob, configure_job=lambda job: None, birth_time=lambda process: 1)
        kwargs.update(options)
        return outer._observe(**kwargs)

    def test_success_requires_gated_zero_exit_and_binds_terminal(self):
        verifier = mock.Mock(return_value=True)
        result = self.observe(self.command(), terminal_check=verifier)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['purpose'], 'preparation')
        self.assertEqual(result['supervisor_terminal_sha256'], digest(b'fabricated-terminal'))
        self.assertEqual(result['exit_code'], 0)
        self.assertTrue(result['parent_reaped'] and result['parent_handle_closed']
                        and result['outer_job_drained'] and result['outer_job_closed'])
        self.assertEqual(result['active_job_members'], 0)
        verifier.assert_called_once_with(b'fabricated-terminal', 'preparation',
                                         result['parent_pid'], result['parent_birth_100ns'])
        self.assertEqual(json.loads((self.output / 'observation.json').read_bytes()), result)
        with self.assertRaises(ValueError):
            self.observe(self.command())

    def test_nonzero_exit_with_plausible_terminal_never_authorizes(self):
        check = mock.Mock(return_value=True)
        with self.assertRaises(ValueError):
            self.observe(self.command(exit_code=7), terminal_check=check)
        check.assert_not_called()
        self.assertFalse((self.output / 'observation.json').exists())
        self.assertTrue((self.output / 'started.json').exists())

    def test_preexisting_terminal_cannot_authorize_a_child_that_writes_nothing(self):
        self.terminal.write_bytes(b'fabricated-terminal')
        code = "import sys; sys.stdin.buffer.readline(); sys.exit(0)"
        command = [sys.executable, '-B', '-c', code]
        check = mock.Mock(return_value=True)
        with self.assertRaises(ValueError):
            self.observe(command, terminal_check=check)
        check.assert_not_called()
        self.assertFalse((self.output / 'observation.json').exists())

    def test_terminal_created_after_spawn_but_before_start_is_rejected(self):
        calls = 0
        def changed_source():
            nonlocal calls
            calls += 1
            if calls == 2:
                self.terminal.write_bytes(b'fabricated-terminal')
            return True
        check = mock.Mock(return_value=True)
        with self.assertRaises(ValueError):
            self.observe(self.command(), source_check=changed_source, terminal_check=check)
        check.assert_not_called()
        self.assertFalse((self.output / 'observation.json').exists())

    def test_supervisor_may_create_a_fresh_terminal_directory(self):
        self.terminal = self.root / 'fresh-supervisor' / 'supervisor.json'
        code = ("import pathlib,sys; assert sys.stdin.buffer.readline()==b'PRIVATE-FAKE-GATE\\n'; "
                "p=pathlib.Path(sys.argv[1]); p.parent.mkdir(); p.write_bytes(b'fabricated-terminal')")
        result = self.observe([sys.executable, '-B', '-c', code, str(self.terminal)])
        self.assertEqual(result['supervisor_terminal_sha256'], digest(b'fabricated-terminal'))

    def test_changed_configuration_fails_before_terminal_verification(self):
        check = mock.Mock(return_value=True)
        with self.assertRaises(ValueError):
            self.observe(self.command(mutate_config=True), terminal_check=check)
        check.assert_not_called()
        self.assertFalse((self.output / 'observation.json').exists())

    def test_unverified_terminal_or_active_job_fails(self):
        with self.assertRaises(ValueError):
            self.observe(self.command(), terminal_check=lambda raw, purpose, pid, birth: False)
        self.assertFalse((self.output / 'observation.json').exists())
        self.output = self.root / 'second-outer'
        self.terminal.unlink()
        class ActiveJob(FakeJob):
            active_override = 1
        with self.assertRaises(ValueError):
            self.observe(self.command(), job_type=ActiveJob)
        self.assertFalse((self.output / 'observation.json').exists())

    def test_interrupted_wait_keeps_job_and_process_in_quarantine(self):
        command = self.command()
        with mock.patch.object(subprocess.Popen, 'wait', side_effect=KeyboardInterrupt('synthetic')):
            with self.assertRaises(KeyboardInterrupt):
                self.observe(command)
        self.assertFalse((self.output / 'observation.json').exists())
        self.assertTrue(outer._QUARANTINED)
        state = outer._QUARANTINED[-1]
        process = state['process']
        process.wait(timeout=10)
        state['job'].terminate_and_wait()
        state['job'].close()
        outer._close_process_handle(process)
        outer._QUARANTINED.remove(state)

    def test_uncertain_job_close_is_never_retried(self):
        class InterruptCloseJob(FakeJob):
            last = None
            def __init__(self):
                super().__init__()
                self.calls = 0
                type(self).last = self
            def close(self):
                self.calls += 1
                raise KeyboardInterrupt('synthetic uncertain close')
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.observe(self.command(), job_type=InterruptCloseJob)
            self.assertEqual(InterruptCloseJob.last.calls, 1)
            self.assertTrue(outer._QUARANTINED)
            self.assertFalse((self.output / 'observation.json').exists())
        finally:
            if outer._QUARANTINED:
                outer._close_process_handle(outer._QUARANTINED[-1]['process'])
                outer._QUARANTINED.clear()  # Fake Job only; its child was already reaped.

    def test_uncertain_job_drain_is_never_retried_or_closed(self):
        class InterruptDrainJob(FakeJob):
            last = None
            def __init__(self):
                super().__init__()
                self.calls = 0
                type(self).last = self
            def terminate_and_wait(self):
                self.calls += 1
                raise KeyboardInterrupt('synthetic uncertain drain')
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.observe(self.command(), job_type=InterruptDrainJob)
            self.assertEqual(InterruptDrainJob.last.calls, 1)
            self.assertFalse(InterruptDrainJob.last.closed)
            self.assertTrue(outer._QUARANTINED)
            self.assertFalse((self.output / 'observation.json').exists())
        finally:
            if outer._QUARANTINED:
                outer._close_process_handle(outer._QUARANTINED[-1]['process'])
                outer._QUARANTINED.clear()  # Fake Job only; its child was already reaped.

    def test_uncertain_process_handle_close_is_not_retried(self):
        closer = mock.Mock(side_effect=KeyboardInterrupt('synthetic uncertain handle close'))
        try:
            with mock.patch.object(outer, '_close_process_handle', closer):
                with self.assertRaises(KeyboardInterrupt):
                    self.observe(self.command())
            self.assertEqual(closer.call_count, 1)
            self.assertTrue(outer._QUARANTINED)
            self.assertFalse((self.output / 'observation.json').exists())
        finally:
            if outer._QUARANTINED:
                outer._close_process_handle(outer._QUARANTINED[-1]['process'])
                outer._QUARANTINED.clear()

    def test_interrupt_after_cleanup_preserves_cancellation(self):
        lines, start = inspect.getsourcelines(outer._observe)
        return_line = next(start + index for index, line in enumerate(lines) if line.strip() == 'return result')
        def interrupt(frame, event, arg):
            if frame.f_code is outer._observe.__code__ and event == 'line' and frame.f_lineno == return_line:
                raise KeyboardInterrupt('synthetic return-boundary interrupt')
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                self.observe(self.command())
        finally:
            sys.settrace(None)
        self.assertTrue((self.output / 'observation.json').exists())
        self.assertFalse(outer._QUARANTINED)

    @unittest.skipUnless(os.name == 'nt', 'Windows Job integration')
    def test_real_windows_job_reaps_and_closes_a_gated_child(self):
        try:
            source = subprocess.check_output(['git', 'show',
                '479b41fa3ab1b3131644fa14d68d761ce9672f42:scripts/v2_source_guards.py'], stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest('reviewed Job source is unavailable in this archive')
        module = types.ModuleType('_fabricated_reviewed_job')
        exec(compile(source, 'verified-v2-source-guards.py', 'exec', dont_inherit=True), module.__dict__)
        def configure(job):
            limits = module._JobExtendedLimits()
            limits.basic.flags = 0x00002000 | 0x00001000
            if not job._kernel.SetInformationJobObject(job._handle, 9, outer.ctypes.byref(limits),
                                                       outer.ctypes.sizeof(limits)):
                raise RuntimeError('cannot configure synthetic Job')
        result = self.observe(self.command(), job_type=module.WindowsJob,
                              configure_job=configure, birth_time=outer._birth_time)
        self.assertTrue(result['parent_reaped'] and result['parent_handle_closed'])
        self.assertTrue(result['outer_job_drained'] and result['outer_job_closed'])
        self.assertEqual(result['active_job_members'], 0)


if __name__ == '__main__':
    unittest.main()
