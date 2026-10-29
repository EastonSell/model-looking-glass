"""CPU-only integration checks for supervised Windows process jobs."""

import os
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from looking_glass.supervisor import (StartupChannel, SupervisorError, pending_supervised_children,
                                      run_supervised)


OWNER_CLEAN = textwrap.dedent("""
    from looking_glass.supervisor import OwnerSession
    from looking_glass.device_lease import DeviceLease
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
    session.check(lease)
    session.attest_clean(lease)
""")


@unittest.skipUnless(os.name == 'nt', 'Windows process jobs are required')
class SupervisedJobTests(unittest.TestCase):
    def test_clean_owner_is_reaped_and_job_is_drained_before_marker_clears(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            try:
                result = run_supervised(
                    [sys._base_executable, '-c', OWNER_CLEAN],
                    device_lease_path=root / 'gpu.lock',
                    marker_path=root / 'gpu.unsafe.json',
                    timeout=10, lease_timeout=3, terminate_grace=3,
                    process_job=job,
                )
                self.assertTrue(result.clean)
                self.assertEqual(result.exit_code, 0)
                self.assertFalse((root / 'gpu.unsafe.json').exists())
                self.assertIsNone(job._handle)
                self.assertTrue(job._confirmed)
            finally:
                job.close()

    def test_assignment_completes_before_the_child_can_enter_device_work(self):
        from scripts.v2_source_guards import WindowsJob

        code = textwrap.dedent("""
            from pathlib import Path
            import sys
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            Path(sys.argv[1]).touch()
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            Path(sys.argv[2]).touch()
            session.attest_clean(lease)
        """)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entered, active = root / 'entered', root / 'active'
            job = WindowsJob()
            assigning, release = threading.Event(), threading.Event()
            outcomes = []
            assign = job.assign
            def delayed_assignment(child):
                assigning.set()
                if not release.wait(6):
                    raise TimeoutError('test did not release job assignment')
                return assign(child)
            def launch():
                try:
                    outcomes.append(run_supervised(
                        [sys._base_executable, '-c', code, str(entered), str(active)],
                        device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                        timeout=10, lease_timeout=3, terminate_grace=3, process_job=job))
                except BaseException as error:
                    outcomes.append(error)
            try:
                with mock.patch.object(job, 'assign', delayed_assignment):
                    thread = threading.Thread(target=launch, daemon=True)
                    thread.start()
                    try:
                        self.assertTrue(assigning.wait(5))
                        deadline = time.monotonic() + 5
                        while not entered.exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue(entered.exists())
                        self.assertFalse(active.exists(), 'child passed the grant before job assignment')
                    finally:
                        release.set()
                        thread.join(timeout=12)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(outcomes), 1)
                self.assertTrue(getattr(outcomes[0], 'clean', False), repr(outcomes[0]))
                self.assertTrue(active.exists())
            finally:
                job.close()

    def test_failure_after_assignment_stops_owner_and_keeps_marker(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, children = WindowsJob(), []
            assign = job.assign
            def interrupted_assignment(child):
                assign(child)
                children.append(child)
                raise OSError('simulated assignment return failure')
            try:
                with mock.patch.object(job, 'assign', interrupted_assignment):
                    with self.assertRaises(OSError):
                        run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                            device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                            timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertTrue((root / 'gpu.unsafe.json').exists())
                self.assertIsNone(job._handle)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=3)
                job.close()

    def test_fatal_cleanup_interrupt_preserves_custody_and_primary_failure(self):
        from scripts.v2_source_guards import WindowsJob

        # Fatal cancellation wins over an ordinary launch error, while a
        # previously received fatal interruption keeps precedence.
        pairs = ((OSError('assignment failed'), KeyboardInterrupt('drain interrupted')),
                 (OSError('assignment failed'), SystemExit(7)),
                 (KeyboardInterrupt('launch interrupted'), SystemExit(8)))
        for primary, cleanup in pairs:
            with self.subTest(primary=type(primary), cleanup=type(cleanup)):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    job = WindowsJob()
                    assign = job.assign
                    def fail_after_assign(child):
                        assign(child)
                        raise primary
                    def fail_drain(*, timeout):
                        raise cleanup
                    try:
                        with (mock.patch.object(job, 'assign', fail_after_assign),
                              mock.patch.object(job, 'terminate_and_wait', fail_drain)):
                            expected = primary if isinstance(primary, (KeyboardInterrupt, SystemExit)) else cleanup
                            with self.assertRaises(type(expected)) as raised:
                                run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                                    device_lease_path=root / 'gpu.lock',
                                    marker_path=root / 'gpu.unsafe.json',
                                    timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                        self.assertIs(raised.exception, expected)
                        if expected is cleanup:
                            self.assertIs(raised.exception.__cause__, primary)
                        records = [record for record in pending_supervised_children()
                                   if record._process_job is job]
                        self.assertEqual(len(records), 1)
                        self.assertIsNotNone(records[0]._child.poll())
                        self.assertTrue((root / 'gpu.unsafe.json').exists())
                        records[0].recover(terminate_grace=3)
                        self.assertTrue((root / 'gpu.unsafe.json').exists())
                    finally:
                        for record in pending_supervised_children():
                            if record._process_job is job:
                                record.recover(terminate_grace=3)
                        job.close()

    def test_startup_abort_does_not_mask_fatal_supervisor_failure(self):
        from looking_glass import supervisor

        pairs = ((KeyboardInterrupt('supervision interrupted'), OSError('abort failed')),
                 (OSError('supervision failed'), KeyboardInterrupt('abort interrupted')))
        for primary, abort_error in pairs:
            with self.subTest(primary=type(primary), abort=type(abort_error)):
                channel = StartupChannel(max_bytes=64)
                try:
                    with (mock.patch.object(supervisor, '_run_supervised', side_effect=primary),
                          mock.patch.object(channel, 'abort', side_effect=abort_error)):
                        expected = primary if isinstance(primary, (KeyboardInterrupt, SystemExit)) else abort_error
                        with self.assertRaises(type(expected)) as raised:
                            run_supervised([sys._base_executable], device_lease_path='unused.lock',
                                           marker_path='unused.unsafe.json', timeout=1,
                                           startup_channel=channel)
                    self.assertIs(raised.exception, expected)
                    if expected is abort_error:
                        self.assertIs(raised.exception.__cause__, primary)
                finally:
                    channel.abort()
                    self.assertTrue(channel.wait_closed(timeout=3))

    def test_interrupt_at_assignment_return_reaps_child_and_preserves_marker(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, children = WindowsJob(), []
            previous = sys.gettrace()
            def trace(frame, event, arg):
                if frame.f_code is WindowsJob.assign.__code__ and event == 'return':
                    children.append(frame.f_locals['process'])
                    raise KeyboardInterrupt('interrupted after real assignment')
                return trace
            try:
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                            device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                            timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                finally:
                    sys.settrace(previous)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertTrue((root / 'gpu.unsafe.json').exists())
                self.assertIsNone(job._handle)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=3)
                job.close()

    def test_live_descendant_is_drained_but_keeps_the_marker_unsafe(self):
        import ctypes
        from ctypes import wintypes
        from scripts.v2_source_guards import WindowsJob

        code = textwrap.dedent("""
            from pathlib import Path
            import subprocess, sys
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            child = subprocess.Popen([sys._base_executable, '-c', 'import time; time.sleep(60)'],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            Path(sys.argv[1]).write_text(str(child.pid))
            session.attest_clean(lease)
        """)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path, marker = root / 'descendant.pid', root / 'gpu.unsafe.json'
            job = WindowsJob()
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel.CloseHandle.restype = wintypes.BOOL
            observed = []
            drain = job.terminate_and_wait
            def observe_drain(*, timeout):
                self.assertTrue(marker.exists())
                handle = kernel.OpenProcess(0x00100000, False, int(pid_path.read_text()))
                self.assertTrue(handle, 'could not retain the live descendant handle')
                try:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0x102)
                    drain(timeout=timeout)
                    observed.append(kernel.WaitForSingleObject(handle, 0))
                    self.assertTrue(marker.exists(), 'marker cleared before job drain completed')
                finally:
                    kernel.CloseHandle(handle)
            try:
                with mock.patch.object(job, 'terminate_and_wait', observe_drain):
                    result = run_supervised([sys._base_executable, '-c', code, str(pid_path)],
                        device_lease_path=root / 'gpu.lock', marker_path=marker,
                        timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                self.assertFalse(result.clean)
                self.assertEqual(observed, [0])
                self.assertTrue(marker.exists())
            finally:
                job.close()

    def test_same_job_cannot_be_shared_by_concurrent_supervisors(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            assigned, release = threading.Event(), threading.Event()
            outcomes = []
            assign = job.assign
            def pause_after_assignment(child):
                assign(child)
                assigned.set()
                if not release.wait(6):
                    raise TimeoutError('test did not release assignment')
            def first_run():
                try:
                    outcomes.append(run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                        device_lease_path=root / 'first.lock', marker_path=root / 'first.unsafe.json',
                        timeout=10, lease_timeout=3, terminate_grace=3, process_job=job))
                except BaseException as error:
                    outcomes.append(error)
            try:
                with mock.patch.object(job, 'assign', pause_after_assignment):
                    thread = threading.Thread(target=first_run, daemon=True)
                    thread.start()
                    try:
                        self.assertTrue(assigned.wait(5))
                        with self.assertRaises((SupervisorError, ValueError)):
                            run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                                device_lease_path=root / 'second.lock', marker_path=root / 'second.unsafe.json',
                                timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                        self.assertIsNotNone(job._handle)
                        self.assertFalse(job._stopping)
                    finally:
                        release.set()
                        thread.join(timeout=12)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(outcomes), 1)
                self.assertTrue(getattr(outcomes[0], 'clean', False), repr(outcomes[0]))
                self.assertFalse((root / 'second.unsafe.json').exists())
            finally:
                job.close()

    def test_nonempty_job_is_rejected_before_touching_unrelated_process(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            other = subprocess.Popen([sys._base_executable, '-c', 'import time; time.sleep(60)'],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            try:
                job.assign(other)
                with self.assertRaises(ValueError):
                    run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                        device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                        timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                self.assertIsNone(other.poll(), 'unrelated existing job member was terminated')
                self.assertFalse((root / 'gpu.unsafe.json').exists())
            finally:
                if job._handle is not None:
                    job.terminate_and_wait(timeout=3)
                other.wait(timeout=3)
                job.close()

    def test_previously_used_job_is_rejected_even_after_its_member_exits(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            other = subprocess.Popen([sys._base_executable, '-c', 'import time; time.sleep(.3)'],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            try:
                job.assign(other)
                self.assertEqual(other.wait(timeout=3), 0)
                with self.assertRaises(ValueError):
                    run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                        device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                        timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                self.assertFalse((root / 'gpu.unsafe.json').exists())
                self.assertIsNotNone(job._handle)
            finally:
                if job._handle is not None:
                    job.terminate_and_wait(timeout=3)
                other.wait(timeout=3)
                job.close()

    def test_failed_job_drain_is_not_retried_and_remains_in_recovery_custody(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            attempts = []
            def failed_drain(*, timeout):
                attempts.append(timeout)
                raise OSError('simulated uncertain drain')
            try:
                with mock.patch.object(job, 'terminate_and_wait', failed_drain):
                    with self.assertRaises(OSError):
                        run_supervised(
                            [sys._base_executable, '-c', OWNER_CLEAN],
                            device_lease_path=root / 'gpu.lock',
                            marker_path=root / 'gpu.unsafe.json',
                            timeout=10, lease_timeout=3, terminate_grace=3,
                            process_job=job,
                        )
                self.assertEqual(len(attempts), 1)
                self.assertTrue((root / 'gpu.unsafe.json').exists())
                records = [record for record in pending_supervised_children()
                           if record._process_job is job]
                self.assertEqual(len(records), 1)
                self.assertIsNotNone(records[0]._child.poll())
                records[0].recover(terminate_grace=3)
                self.assertTrue((root / 'gpu.unsafe.json').exists())
                self.assertIsNone(job._handle)
            finally:
                for record in pending_supervised_children():
                    if record._process_job is job:
                        record.recover(terminate_grace=3)
                job.close()

    def test_unknown_or_malformed_job_count_keeps_recovery_custody_and_marker(self):
        from scripts.v2_source_guards import WindowsJob

        for value in ('error', -1, True, (0, 0), (1, 2)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                job = WindowsJob()
                calls = []
                def bad_count():
                    calls.append(1)
                    if len(calls) == 1:
                        return (0, 0)
                    if value == 'error':
                        raise OSError('simulated accounting failure')
                    return value
                try:
                    with mock.patch.object(job, 'process_counts', bad_count):
                        with self.assertRaises((OSError, SupervisorError)):
                            run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                                device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                                timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                    self.assertTrue((root / 'gpu.unsafe.json').exists())
                    records = [record for record in pending_supervised_children()
                               if record._process_job is job]
                    self.assertEqual(len(records), 1)
                    self.assertTrue(job._confirmed, 'best-effort job termination was not performed')
                    self.assertIsNotNone(job._handle)
                    records[0].recover(terminate_grace=3)
                    self.assertTrue((root / 'gpu.unsafe.json').exists())
                finally:
                    for record in pending_supervised_children():
                        if record._process_job is job:
                            record.recover(terminate_grace=3)
                    job.close()

    def test_failed_job_close_is_not_retried_and_remains_recoverable(self):
        from scripts.v2_source_guards import WindowsJob

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            calls = []
            def failed_close():
                calls.append(1)
                raise OSError('simulated uncertain job close')
            try:
                with mock.patch.object(job, 'close', failed_close):
                    with self.assertRaises(OSError):
                        run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                            device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                            timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                self.assertEqual(len(calls), 1)
                self.assertTrue((root / 'gpu.unsafe.json').exists())
                records = [record for record in pending_supervised_children()
                           if record._process_job is job]
                self.assertEqual(len(records), 1)
                self.assertTrue(job._confirmed)
                records[0].recover(terminate_grace=3)
                self.assertIsNone(job._handle)
                self.assertTrue((root / 'gpu.unsafe.json').exists())
            finally:
                for record in pending_supervised_children():
                    if record._process_job is job:
                        record.recover(terminate_grace=3)
                job.close()

    def test_interrupt_after_drain_latch_keeps_job_rooted_until_explicit_recovery(self):
        from looking_glass import supervisor
        from scripts.v2_source_guards import WindowsJob

        source, first = inspect.getsourcelines(supervisor._SpawnCustody._finish_job)
        line = first + next(index for index, text in enumerate(source) if 'was_quiescent = False' in text)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            previous = sys.gettrace()
            def trace(frame, event, arg):
                if (frame.f_code is supervisor._SpawnCustody._finish_job.__code__
                        and event == 'line' and frame.f_lineno == line):
                    raise KeyboardInterrupt('interrupted after drain latch')
                return trace
            try:
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        run_supervised([sys._base_executable, '-c', OWNER_CLEAN],
                            device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                            timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                finally:
                    sys.settrace(previous)
                records = [record for record in pending_supervised_children()
                           if record._process_job is job]
                self.assertEqual(len(records), 1)
                self.assertTrue(records[0]._job_drain_uncertain)
                self.assertIsNotNone(records[0]._child.poll())
                self.assertIsNotNone(job._handle)
                self.assertTrue((root / 'gpu.unsafe.json').exists())
                records[0].recover(terminate_grace=3)
                self.assertIsNone(job._handle)
                self.assertTrue((root / 'gpu.unsafe.json').exists())
            finally:
                for record in pending_supervised_children():
                    if record._process_job is job:
                        record.recover(terminate_grace=3)
                job.close()

    def test_timed_out_owner_is_reaped_and_its_job_drained(self):
        from scripts.v2_source_guards import WindowsJob

        code = textwrap.dedent("""
            import time
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            time.sleep(60)
        """)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = WindowsJob()
            try:
                result = run_supervised([sys._base_executable, '-c', code],
                    device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                    timeout=.8, lease_timeout=3, terminate_grace=3, process_job=job)
                self.assertTrue(result.timed_out)
                self.assertFalse(result.clean)
                self.assertIsNotNone(result.exit_code)
                self.assertTrue((root / 'gpu.unsafe.json').exists())
                self.assertTrue(job._confirmed)
                self.assertIsNone(job._handle)
            finally:
                job.close()

    def test_unobserved_exited_descendant_keeps_marker_and_job_custody(self):
        owner = textwrap.dedent("""
            import subprocess, sys
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            subprocess.run([sys._base_executable, '-c', 'pass'], check=True)
            session.attest_clean(lease)
        """)
        probe = textwrap.dedent("""
            from pathlib import Path
            import sys
            from looking_glass.supervisor import pending_supervised_children, run_supervised
            from scripts.v2_source_guards import SourceGuardError, WindowsJob
            root = Path(sys.argv[1])
            job = WindowsJob()
            try:
                try:
                    run_supervised([sys._base_executable, '-c', sys.argv[2]],
                        device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                        timeout=10, lease_timeout=3, terminate_grace=3, process_job=job)
                except SourceGuardError:
                    pass
                else:
                    raise AssertionError('unobserved lifetime member was accepted')
                records = [record for record in pending_supervised_children()
                           if record._process_job is job]
                assert len(records) == 1 and records[0]._child.poll() is not None
                assert job._unconfirmed and job._handle is not None
                assert (root / 'gpu.unsafe.json').exists()
            finally:
                job.close()
        """)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys._base_executable, '-B', '-c', probe, directory, owner],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_private_startup_channel_still_works_with_a_drained_job(self):
        from scripts.v2_source_guards import WindowsJob

        code = textwrap.dedent("""
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            session.send_startup(lease, b'channel-ok')
            session.attest_clean(lease)
        """)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, channel, outcomes = WindowsJob(), StartupChannel(max_bytes=32), []
            def launch():
                try:
                    outcomes.append(run_supervised([sys._base_executable, '-c', code],
                        device_lease_path=root / 'gpu.lock', marker_path=root / 'gpu.unsafe.json',
                        timeout=10, lease_timeout=3, terminate_grace=3, process_job=job,
                        startup_channel=channel))
                except BaseException as error:
                    outcomes.append(error)
            thread = threading.Thread(target=launch, daemon=True)
            try:
                thread.start()
                self.assertEqual(channel.receive(timeout=6), b'channel-ok')
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(outcomes), 1)
                self.assertTrue(getattr(outcomes[0], 'clean', False), repr(outcomes[0]))
                self.assertTrue(job._confirmed)
                self.assertFalse((root / 'gpu.unsafe.json').exists())
            finally:
                channel.abort()
                thread.join(timeout=10)
                job.close()


if __name__ == '__main__':
    unittest.main()
