"""CPU subprocess regressions for supervised child launch custody."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from looking_glass import supervisor as api


OWNER = textwrap.dedent("""
    from pathlib import Path
    import sys, time
    from looking_glass.supervisor import OwnerSession
    from looking_glass.device_lease import DeviceLease
    Path(sys.argv[1]).touch()
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
    session.check(lease)
    Path(sys.argv[2]).touch()
    time.sleep(20)
""")


class SpawnSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.device = self.root / "gpu.lock"
        self.marker = self.root / "gpu.unsafe.json"
        self.entered = self.root / "entered"
        self.ready = self.root / "ready"

    def run_owner(self, channel=None):
        return api.run_supervised(
            [sys._base_executable, "-c", OWNER, str(self.entered), str(self.ready)],
            device_lease_path=self.device, marker_path=self.marker, timeout=8,
            lease_timeout=3, terminate_grace=2, startup_channel=channel,
        )

    def assert_interrupted_return_is_reaped(self, code, channel=None):
        saved = []
        previous = sys.gettrace()

        def trace(frame, event, arg):
            if frame.f_code is code and event == "return":
                if code is api.StartupChannel._spawn.__code__:
                    child = arg
                elif code is api._SpawnCustody._grant.__code__:
                    child = frame.f_locals["self"]._child
                else:
                    child = frame.f_locals["self"]
                saved.append(child)
                raise KeyboardInterrupt("injected return interruption")
            return trace

        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                self.run_owner(channel)
        finally:
            sys.settrace(previous)
            if channel is not None:
                channel.abort()
            for child in saved:
                self.addCleanup(self.stop_if_alive, child)
        self.assertEqual(len(saved), 1)
        self.assertIsNotNone(saved[0].poll(), "the exact direct child was not reaped")
        self.assertTrue(self.marker.exists())
        with self.assertRaises(api.UnsafeOwnerError):
            self.run_owner()

    @staticmethod
    def stop_if_alive(child):
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)

    def test_interrupted_plain_popen_constructor_is_reaped(self):
        self.assert_interrupted_return_is_reaped(subprocess.Popen.__init__.__code__)

    def test_interrupted_channel_spawn_return_is_reaped(self):
        channel = api.StartupChannel(max_bytes=32)
        self.addCleanup(channel.close)
        self.assert_interrupted_return_is_reaped(api.StartupChannel._spawn.__code__, channel)
        self.assertTrue(channel.wait_closed(timeout=3))

    def test_channel_abort_before_authorization_denies_grant(self):
        channel = api.StartupChannel(max_bytes=32)
        self.addCleanup(channel.close)
        reached = threading.Event()
        release = threading.Event()
        failures = []
        original = api.StartupChannel._authorize_launch

        def blocked_authorization(instance):
            reached.set()
            if not release.wait(5):
                raise AssertionError("test did not release authorization")
            return original(instance)

        def launch():
            try:
                self.run_owner(channel)
            except BaseException as exc:
                failures.append(exc)

        with mock.patch.object(api.StartupChannel, "_authorize_launch", blocked_authorization):
            thread = threading.Thread(target=launch, daemon=True)
            thread.start()
            try:
                self.assertTrue(reached.wait(4))
                channel.abort()
            finally:
                release.set()
                thread.join(timeout=6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], api.StartupChannelError)
        self.assertFalse(channel._launch_authorized)
        self.assertFalse(self.ready.exists())
        self.assertTrue(self.marker.exists())
        self.assertTrue(channel.wait_closed(timeout=3))

    def test_channel_abort_after_authorization_does_not_wait_for_grant_write(self):
        channel = api.StartupChannel(max_bytes=32)
        self.addCleanup(channel.close)
        reached = threading.Event()
        release = threading.Event()
        outcomes = []
        original_write = os.write

        def blocked_write(fd, data):
            if data == bytes([1]):
                reached.set()
                if not release.wait(5):
                    raise AssertionError("test did not release grant")
            return original_write(fd, data)

        def launch():
            try:
                outcomes.append(self.run_owner(channel))
            except BaseException as exc:
                outcomes.append(exc)

        with mock.patch.object(api.os, "write", blocked_write):
            thread = threading.Thread(target=launch, daemon=True)
            thread.start()
            try:
                self.assertTrue(reached.wait(4))
                self.assertTrue(channel._launch_authorized)
                records = api.pending_supervised_children()
                self.assertEqual(len(records), 1)
                child = records[0]._child
                self.addCleanup(self.stop_if_alive, child)
                start = time.monotonic()
                channel.abort()
                self.assertLess(time.monotonic() - start, .5)
            finally:
                release.set()
                thread.join(timeout=6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertIsNotNone(child.poll())
        self.assertTrue(self.marker.exists())
        self.assertTrue(channel.wait_closed(timeout=3))

    def test_interrupted_startup_close_does_not_close_reused_descriptor(self):
        channel = api.StartupChannel(max_bytes=32)
        self.addCleanup(channel.close)
        target = channel._write_fd
        replacement = []
        children = []
        original_close = os.close
        original_spawn = api._spawn_with_gate

        def interrupted_close(fd):
            if fd == target and not replacement:
                original_close(fd)
                # A reader thread can release a lower fd at the same time.
                # Keep those replacements open until the target is reused.
                for _ in range(64):
                    replacement.append(os.open(os.devnull, os.O_WRONLY))
                    if replacement[-1] >= target:
                        break
                raise KeyboardInterrupt("close returned before interruption")
            return original_close(fd)

        def capture_child(*args, **kwargs):
            original_spawn(*args, **kwargs)
            children.append(kwargs["custody"]._child)

        try:
            with (mock.patch.object(api.os, "close", interrupted_close),
                  mock.patch.object(api, "_spawn_with_gate", capture_child)):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_owner(channel)
            self.assertTrue(replacement)
            if replacement[-1] != target:
                self.skipTest("could not safely arrange exact descriptor reuse")
            os.fstat(target)
            self.assertIsNone(channel._write_fd)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())
            self.assertTrue(self.marker.exists())
        finally:
            for child in children:
                self.stop_if_alive(child)
            for fd in replacement:
                original_close(fd)

    def test_interrupted_grant_return_stops_exact_child(self):
        self.assert_interrupted_return_is_reaped(api._SpawnCustody._grant.__code__)

    def test_grant_write_that_succeeds_then_raises_reaps_child(self):
        original_write = os.write
        saved = []

        def interrupted_write(fd, data):
            count = original_write(fd, data)
            if data == bytes([1]):
                records = api.pending_supervised_children()
                self.assertEqual(len(records), 1)
                saved.append(records[0]._child)
                raise KeyboardInterrupt("grant write completed")
            return count

        try:
            with mock.patch.object(api.os, "write", interrupted_write):
                with self.assertRaisesRegex(KeyboardInterrupt, "grant write completed"):
                    self.run_owner()
            self.assertEqual(len(saved), 1)
            self.assertIsNotNone(saved[0].poll())
            self.assertTrue(self.marker.exists())
        finally:
            for child in saved:
                self.stop_if_alive(child)

    def test_interrupted_registration_retires_unattempted_record(self):
        previous = sys.gettrace()
        saved = []

        def trace(frame, event, arg):
            if frame.f_code is api._SpawnCustody._register.__code__ and event == "return":
                saved.append(frame.f_locals["self"])
                raise KeyboardInterrupt("registration interrupted")
            return trace

        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                self.run_owner()
        finally:
            sys.settrace(previous)
        self.assertEqual(len(saved), 1)
        self.assertFalse(saved[0]._attempted)
        self.assertNotIn(saved[0], api.pending_supervised_children())
        self.assertTrue(self.marker.exists())
        self.assertFalse(self.entered.exists())

    def test_recovery_retires_unattempted_record_after_launcher_quiescence(self):
        record = api._SpawnCustody("unattempted-test")
        record._register()
        self.addCleanup(record._retire)
        self.assertIsNone(record.recover(terminate_grace=1))
        self.assertNotIn(record, api.pending_supervised_children())

    def test_failed_reap_keeps_recoverable_exact_child(self):
        previous = sys.gettrace()
        saved = []

        def trace(frame, event, arg):
            if frame.f_code is api._SpawnCustody._grant.__code__ and event == "return":
                saved.append(frame.f_locals["self"]._child)
                raise KeyboardInterrupt("lost supervisor")
            return trace

        try:
            with mock.patch.object(api, "_stop_child", side_effect=RuntimeError("stop failed")):
                sys.settrace(trace)
                with self.assertRaisesRegex(KeyboardInterrupt, "lost supervisor"):
                    self.run_owner()
        finally:
            sys.settrace(previous)
        self.assertEqual(len(saved), 1)
        child = saved[0]
        self.addCleanup(self.stop_if_alive, child)
        deadline = time.monotonic() + 5
        while not self.ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(self.ready.exists(), "granted owner did not become active")
        records = [item for item in api.pending_supervised_children() if item._child is child]
        self.assertEqual(len(records), 1)
        records[0].recover(terminate_grace=2)
        self.assertIsNotNone(child.poll())
        self.assertTrue(self.marker.exists())
        self.assertNotIn(records[0], api.pending_supervised_children())
        with self.assertRaises(api.UnsafeOwnerError):
            records[0].recover(terminate_grace=1)

    def test_unknown_spawn_cannot_pass_gate_and_remains_quarantined(self):
        # Run in a separate interpreter because an unknown native outcome must
        # intentionally remain rooted until process-level recovery.
        driver = textwrap.dedent("""
            import os, sys, time
            from pathlib import Path
            from looking_glass import supervisor as api
            root = Path(sys.argv[1])
            saved = []
            original = api.subprocess.Popen

            def interrupted(*args, **kwargs):
                child = original(*args, **kwargs)
                saved.append(child)
                deadline = time.monotonic() + 5
                while not (root / 'entered').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                raise KeyboardInterrupt('constructor lost its return')

            api.subprocess.Popen = interrupted
            try:
                try:
                    api.run_supervised(
                        [sys._base_executable, '-c', sys.argv[2],
                         str(root / 'entered'), str(root / 'ready')],
                        device_lease_path=root / 'gpu.lock',
                        marker_path=root / 'gpu.unsafe.json',
                        timeout=8, lease_timeout=3, terminate_grace=1)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError('injection was not reached')
                assert len(saved) == 1
                assert saved[0].wait(timeout=5) != 0
                assert (root / 'entered').exists()
                assert not (root / 'ready').exists()
                assert (root / 'gpu.unsafe.json').exists()
                records = api.pending_supervised_children()
                assert len(records) == 1 and records[0]._child is None
                try:
                    records[0].recover(terminate_grace=1)
                except api.UnsafeOwnerError:
                    pass
                else:
                    raise AssertionError('unknown process was falsely reported reaped')
            finally:
                api.subprocess.Popen = original
                for child in saved:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=3)
        """)
        result = subprocess.run(
            [sys._base_executable, "-c", driver, str(self.root), OWNER],
            capture_output=True, text=True, timeout=12,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ungranted_child_has_deadline_even_if_writer_stays_open(self):
        code = textwrap.dedent("""
            from pathlib import Path
            import sys
            from looking_glass import supervisor as api
            api._LAUNCH_TIMEOUT = .3
            api.OwnerSession.from_environment()
            Path(sys.argv[1]).touch()
        """)
        env = os.environ.copy()
        env[api._SUPERVISOR_ENV] = str(os.getpid())
        env[api._SUPERVISOR_BIRTH_ENV] = api._supervisor_birth()
        env["CUDA_VISIBLE_DEVICES"] = ""
        custody = api._SpawnCustody("test-deadline")
        custody._open_gate()
        self.addCleanup(custody._close_gate)
        with api._startup_spawn_lock:
            api._spawn_with_gate([sys._base_executable, "-c", code, str(self.ready)],
                                 cwd=None, child_env=env, custody=custody)
        child = custody._child
        self.addCleanup(self.stop_if_alive, child)
        self.assertEqual(child.wait(timeout=6), 70)
        self.assertFalse(self.ready.exists())
        self.assertIsNotNone(custody._write_fd, "parent writer was not held open")

    def test_ungranted_child_rejects_eof(self):
        code = textwrap.dedent("""
            from pathlib import Path
            import sys
            from looking_glass import supervisor as api
            api.OwnerSession.from_environment()
            Path(sys.argv[1]).touch()
        """)
        env = os.environ.copy()
        env[api._SUPERVISOR_ENV] = str(os.getpid())
        env[api._SUPERVISOR_BIRTH_ENV] = api._supervisor_birth()
        env["CUDA_VISIBLE_DEVICES"] = ""
        custody = api._SpawnCustody("test-eof")
        custody._open_gate()
        self.addCleanup(custody._close_gate)
        with api._startup_spawn_lock:
            api._spawn_with_gate([sys._base_executable, "-c", code, str(self.ready)],
                                 cwd=None, child_env=env, custody=custody)
        child = custody._child
        self.addCleanup(self.stop_if_alive, child)
        custody._close_gate()
        self.assertNotEqual(child.wait(timeout=6), 0)
        self.assertFalse(self.ready.exists())

    def test_parent_death_ends_ungranted_child(self):
        if os.name == "posix" and not hasattr(os, "pidfd_open"):
            self.skipTest("an exact child process handle is unavailable")
        child_code = textwrap.dedent("""
            from pathlib import Path
            import sys
            from looking_glass import supervisor as api
            original = api._await_launch_grant
            def at_gate():
                Path(sys.argv[1]).touch()
                original()
            api._await_launch_grant = at_gate
            api.OwnerSession.from_environment()
            Path(sys.argv[2]).touch()
        """)
        driver = textwrap.dedent("""
            import os, sys, time
            from pathlib import Path
            from looking_glass import supervisor as api
            root = Path(sys.argv[1])
            env = os.environ.copy()
            env[api._SUPERVISOR_ENV] = str(os.getpid())
            env[api._SUPERVISOR_BIRTH_ENV] = api._supervisor_birth()
            env['CUDA_VISIBLE_DEVICES'] = ''
            record = api._SpawnCustody('parent-death')
            record._open_gate()
            with api._startup_spawn_lock:
                api._spawn_with_gate([sys._base_executable, '-c', sys.argv[2],
                                      str(root / 'at-gate'), str(root / 'ready')],
                                     cwd=None, child_env=env, custody=record)
            (root / 'child-pid').write_text(str(record._child.pid))
            while True:
                time.sleep(.1)
        """)
        parent = subprocess.Popen([sys._base_executable, "-c", driver, str(self.root), child_code],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.stop_if_alive, parent)
        deadline = time.monotonic() + 8
        while (not (self.root / "at-gate").exists()
               or not (self.root / "child-pid").exists()):
            self.assertIsNone(parent.poll(), "test parent exited before child reached gate")
            self.assertLess(time.monotonic(), deadline, "child did not reach gate")
            time.sleep(.02)
        child_pid = int((self.root / "child-pid").read_text())
        if os.name == "nt":
            kernel = api._windows_kernel()
            handle = kernel.OpenProcess(0x00100000 | 0x1000, False, child_pid)
            self.assertTrue(handle, "could not retain the exact child handle")
            try:
                parent.terminate()
                parent.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
            finally:
                kernel.CloseHandle(handle)
        else:
            import select
            handle = os.pidfd_open(child_pid)
            try:
                parent.terminate()
                parent.wait(timeout=5)
                self.assertTrue(select.select([handle], [], [], 5)[0])
            finally:
                os.close(handle)
        self.assertFalse(self.ready.exists())


if __name__ == "__main__":
    unittest.main()
