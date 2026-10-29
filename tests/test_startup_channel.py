"""CPU subprocess checks for the private startup channel."""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from looking_glass import supervisor as api


class StartupChannelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.cleanup_temp)
        self.root = Path(self.temp.name)
        self.device = self.root / "gpu.lock"
        self.marker = self.root / "gpu.unsafe.json"

    def cleanup_temp(self):
        # Windows can briefly deny deletion of a just-released lock file. Keep
        # the retry bounded so a persistent process/descriptor leak still fails.
        deadline = time.monotonic() + 3
        while True:
            try:
                self.temp.cleanup()
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)

    def launch(self, channel, code, *, executable=None, identity=None, env=None):
        result = {}

        def run():
            try:
                result["value"] = api.run_supervised(
                    [executable or sys._base_executable, "-c", textwrap.dedent(code), str(self.root)],
                    device_lease_path=identity.lock_path if identity else self.device,
                    marker_path=identity.marker_path if identity else self.marker,
                    identity=identity, env=env,
                    timeout=12, lease_timeout=3, terminate_grace=1,
                    startup_channel=channel,
                )
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(lambda: thread.join(timeout=3))
        self.addCleanup(channel.close)
        return thread, result

    def test_direct_owner_delivers_generated_bytes_and_cleanly_stops(self):
        channel = api.StartupChannel(max_bytes=1024)
        thread, result = self.launch(channel, """
            import os, secrets, sys, time
            from pathlib import Path
            from looking_glass.supervisor import OwnerSession, SupervisorError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            token = secrets.token_hex(24)
            assert token not in repr(sys.argv) and token not in repr(dict(os.environ))
            session.send_startup(lease, token.encode('ascii'))
            try:
                session.send_startup(lease, b'again')
                raise AssertionError('duplicate startup frame accepted')
            except SupervisorError:
                pass
            stop = Path(sys.argv[1]) / 'stop'
            deadline = time.monotonic() + 8
            while not stop.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert stop.exists()
            session.attest_clean(lease)
        """)
        token = channel.receive(timeout=6)
        self.assertEqual(len(token), 48)
        self.assertTrue(all(c in b"0123456789abcdef" for c in token))
        (self.root / "stop").touch()
        thread.join(timeout=6)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", result)
        self.assertTrue(result["value"].clean)
        self.assertEqual(result["value"].exit_code, 0)
        self.assertFalse(self.marker.exists())
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=0)

    def test_receive_deadline_stops_owner_and_preserves_marker(self):
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, """
            import time
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            time.sleep(10)
        """)
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=1)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", result)
        self.assertFalse(result["value"].clean)
        self.assertTrue(self.marker.exists())

    def test_invalid_frame_stops_owner_without_echoing_payload(self):
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, """
            import os, struct, time
            value = int(os.environ['MLG_STARTUP_CHANNEL_HANDLE'])
            if os.name == 'nt':
                import msvcrt
                fd = msvcrt.open_osfhandle(value, os.O_WRONLY | os.O_BINARY)
            else:
                fd = value
            os.write(fd, struct.pack('!I', 999))
            time.sleep(10)
        """)
        with self.assertRaises(api.StartupChannelError) as exc:
            channel.receive(timeout=6)
        self.assertNotIn("999", str(exc.exception))
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(result["value"].clean)
        self.assertTrue(self.marker.exists())

    def test_failed_spawn_unblocks_receiver(self):
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, "", executable=str(self.root / 'missing'))
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=6)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(result.get("error"), OSError)
        self.assertTrue(self.marker.exists())

    def test_late_writer_cannot_deliver_after_timeout(self):
        channel = api.StartupChannel(max_bytes=32)
        leaked_writer = os.dup(channel._write_fd)
        try:
            with self.assertRaises(api.StartupChannelError):
                channel.receive(timeout=.05)
            channel.abort()
            # Simulates a second process that inherited a writer. The reader
            # must reject even a complete frame arriving after the deadline.
            os.write(leaked_writer, struct.pack('!I', 8) + b'late-data')
        finally:
            os.close(leaked_writer)
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=1)
        deadline = time.monotonic() + 2
        while channel._payload is not None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertIsNone(channel._payload)

    def test_close_before_receiving_rejects_waiting_payload(self):
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, """
            import time
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            session.send_startup(lease, b'private')
            time.sleep(10)
        """)
        self.assertTrue(channel._ready.wait(6))
        channel.close()
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=1)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(result["value"].clean)
        self.assertTrue(self.marker.exists())

    def test_startup_endpoint_is_noninheritable_before_sending(self):
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, """
            import os, sys
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            assert not os.get_inheritable(session._startup_fd)
            assert 'MLG_STARTUP_CHANNEL_HANDLE' not in os.environ
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.send_startup(lease, b'ok')
            session.attest_clean(lease)
        """)
        self.assertEqual(channel.receive(timeout=6), b'ok')
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["value"].clean)

    @unittest.skipUnless(os.name == 'nt', 'Windows handle inheritance check')
    def test_windows_unrelated_child_does_not_inherit_owner_writer(self):
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, """
            import msvcrt, os, subprocess, sys
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            handle = msvcrt.get_osfhandle(session._startup_fd)
            check = '''
            import ctypes, sys
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            base = ctypes.WinDLL("kernelbase", use_last_error=True)
            kernel.GetHandleInformation.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel.GetHandleInformation.restype = wintypes.BOOL
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.DuplicateHandle.argtypes = (wintypes.HANDLE, wintypes.HANDLE,
                wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE),
                wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.DuplicateHandle.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            base.CompareObjectHandles.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            base.CompareObjectHandles.restype = wintypes.BOOL
            flags = wintypes.DWORD()
            candidate = int(sys.argv[1])
            if not kernel.GetHandleInformation(candidate, ctypes.byref(flags)):
                sys.exit(0)
            # A numeric handle may be reused for an unrelated child object.
            # Compare it with a fresh duplicate of the owner's pipe object.
            source = kernel.OpenProcess(0x40, False, int(sys.argv[2]))
            duplicate = wintypes.HANDLE()
            if not source or not kernel.DuplicateHandle(source, candidate,
                    kernel.GetCurrentProcess(), ctypes.byref(duplicate), 0, False, 2):
                sys.exit(2)
            same = base.CompareObjectHandles(candidate, duplicate)
            kernel.CloseHandle(duplicate)
            kernel.CloseHandle(source)
            sys.exit(1 if same else 0)
            '''
            import textwrap
            def probe():
                return subprocess.run([sys._base_executable, '-c', textwrap.dedent(check),
                                       str(handle), str(os.getpid())],
                                      close_fds=False, timeout=3).returncode
            assert probe() == 0
            # Positive control: the same probe detects actual inheritance.
            os.set_inheritable(session._startup_fd, True)
            try:
                assert probe() == 1
            finally:
                os.set_inheritable(session._startup_fd, False)
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.send_startup(lease, b'ok')
            session.attest_clean(lease)
        """)
        self.assertEqual(channel.receive(timeout=6), b'ok')
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["value"].clean)

    def test_timeout_discards_payload_published_at_deadline(self):
        channel = api.StartupChannel(max_bytes=32)
        actual_ready = channel._ready

        class DeadlineRace:
            def wait(self, timeout):
                assert actual_ready.wait(2)
                return False
            def set(self):
                actual_ready.set()

        channel._ready = DeadlineRace()
        os.write(channel._write_fd, struct.pack('!I', 4) + b'data')
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=1)
        self.assertIsNone(channel._payload)
        channel.close()
        self.assertTrue(channel.wait_closed(timeout=2))

    def test_partial_buffer_is_cleared_while_inherited_writer_remains_open(self):
        channel = api.StartupChannel(max_bytes=32)
        leaked_writer = os.dup(channel._write_fd)
        try:
            os.write(channel._write_fd, struct.pack('!I', 8) + b'secr')
            deadline = time.monotonic() + 2
            while len(channel._partial) != 4 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(bytes(channel._partial), b'secr')
            with self.assertRaises(api.StartupChannelError):
                channel.receive(timeout=.05)
            channel.abort()
            self.assertEqual(channel._partial, bytearray())
            self.assertFalse(channel.wait_closed(timeout=.05))
        finally:
            os.close(leaked_writer)
        self.assertTrue(channel.wait_closed(timeout=2))

    def test_concurrent_owner_senders_only_deliver_one_frame(self):
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, """
            import threading
            from looking_glass.supervisor import OwnerSession, StartupChannelError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            barrier = threading.Barrier(3)
            outcomes = []
            def send(payload):
                barrier.wait()
                try:
                    session.send_startup(lease, payload)
                    outcomes.append('sent')
                except StartupChannelError:
                    outcomes.append('rejected')
            workers = [threading.Thread(target=send, args=(payload,)) for payload in (b'one', b'two')]
            for worker in workers: worker.start()
            barrier.wait()
            for worker in workers: worker.join()
            assert sorted(outcomes) == ['rejected', 'sent']
            session.attest_clean(lease)
        """)
        self.assertIn(channel.receive(timeout=6), (b'one', b'two'))
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["value"].clean)

    def test_failed_send_traceback_does_not_retain_payload(self):
        secret = b'synthetic-sensitive-' + bytes(range(32))
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        session = api.OwnerSession.__new__(api.OwnerSession)
        session._identity_spec = None
        session._startup_fd = write_fd
        session._startup_max = 128
        session._startup_send_lock = threading.Lock()
        with mock.patch.object(api.OwnerSession, 'check', return_value=None):
            try:
                session.send_startup(None, secret)
            except api.StartupChannelError as exc:
                self.assertIsNone(exc.__context__)
                self.assertIsNone(exc.__cause__)
                traceback = exc.__traceback__
                while traceback is not None:
                    if traceback.tb_frame.f_code.co_name == 'send_startup':
                        self.assertFalse(any(value is secret for value in
                                             traceback.tb_frame.f_locals.values()))
                    traceback = traceback.tb_next
            else:
                self.fail('write to a closed pipe did not fail')

    def test_wipe_failure_discards_unpublished_payload_and_releases_reader(self):
        channel = api.StartupChannel(max_bytes=32)
        original = channel._clear_partial
        calls = 0

        def broken_wipe():
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise MemoryError('injected cleanup failure')
            original()

        channel._clear_partial = broken_wipe
        try:
            os.write(channel._write_fd, struct.pack('!I', 4) + b'data')
            with self.assertRaises(api.StartupChannelError):
                channel.receive(timeout=3)
            self.assertTrue(channel.wait_closed(timeout=3))
            self.assertIsNone(channel._payload)
            self.assertEqual(channel._partial, bytearray())
        finally:
            channel.close()

    def test_uncertain_reader_close_is_reported(self):
        original_close = os.close

        def failed_close(fd):
            original_close(fd)
            if threading.current_thread().name == 'looking-glass-startup-reader':
                raise OSError('injected uncertain close')

        channel = api.StartupChannel(max_bytes=32)
        with mock.patch.object(api.os, 'close', side_effect=failed_close):
            os.write(channel._write_fd, struct.pack('!I', 4) + b'data')
            try:
                channel.receive(timeout=3)
            except api.StartupChannelError:
                pass  # Closing may fail before or after delivery.
            self.assertFalse(channel.wait_closed(timeout=3))
            channel.abort()

    def test_abort_does_not_wait_for_a_stuck_process_creation(self):
        channel = api.StartupChannel(max_bytes=32)
        entered = threading.Event()
        release = threading.Event()
        errors = []
        channel._claim()

        def slow_popen(*args, **kwargs):
            entered.set()
            release.wait(5)
            raise OSError('injected spawn failure')

        def spawn():
            try:
                channel._spawn(['unused'], cwd=None, child_env={})
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(api.subprocess, 'Popen', side_effect=slow_popen):
            thread = threading.Thread(target=spawn, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                start = time.monotonic()
                channel.abort()
                self.assertLess(time.monotonic() - start, .5)
                self.assertFalse(channel.wait_closed(timeout=.05))
            finally:
                release.set()
                thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], OSError)
        self.assertTrue(channel.wait_closed(timeout=3))

    def test_constructor_interruptions_do_not_leak_or_deadlock(self):
        # Keep any failure bounded even if the CRT would block closing a pipe
        # whose reader was accidentally left running in another thread.
        code = textwrap.dedent("""
            import inspect, sys, threading, time
            from unittest import mock
            from looking_glass import supervisor as api
            original_start = threading.Thread.start
            original_entry = api.StartupChannel._reader_entry
            def interrupted(thread):
                original_start(thread)
                raise KeyboardInterrupt
            def delayed_entry(self, fd):
                time.sleep(.1)
                original_entry(self, fd)
            for delay in (False, True):
                entry = delayed_entry if delay else original_entry
                with mock.patch.object(api.StartupChannel, '_reader_entry', entry):
                    with mock.patch.object(threading.Thread, 'start', interrupted):
                        try:
                            api.StartupChannel(max_bytes=1)
                        except KeyboardInterrupt:
                            pass
                        else:
                            raise AssertionError('missing interruption')
            def no_lock():
                raise MemoryError('injected allocation failure')
            with mock.patch.object(threading, 'Lock', no_lock):
                try:
                    api.StartupChannel(max_bytes=1)
                except MemoryError:
                    pass
                else:
                    raise AssertionError('missing allocation failure')
            lines, first = inspect.getsourcelines(api.StartupChannel.__init__)
            target = first + next(i for i, line in enumerate(lines) if 'self._write_fd = None' in line)
            def trace(frame, event, arg):
                if (frame.f_code is api.StartupChannel.__init__.__code__
                        and event == 'line' and frame.f_lineno == target):
                    raise KeyboardInterrupt
                return trace
            sys.settrace(trace)
            try:
                api.StartupChannel(max_bytes=1)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('missing handoff interruption')
            finally:
                sys.settrace(None)
            deadline = time.monotonic() + 3
            while any(t.name == 'looking-glass-startup-reader' for t in threading.enumerate()):
                if time.monotonic() >= deadline: raise AssertionError('reader did not exit')
                time.sleep(.01)
            channels = [api.StartupChannel(max_bytes=1) for _ in range(16)]
            for channel in channels: channel.close()
            assert all(channel.wait_closed(timeout=2) for channel in channels)
        """)
        result = subprocess.run([sys._base_executable, '-c', code], capture_output=True,
                                text=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_initial_configuration_failure_unblocks_receiver(self):
        channel = api.StartupChannel(max_bytes=32)
        with self.assertRaises(ValueError):
            api.run_supervised([], device_lease_path=self.device, marker_path=self.marker,
                               timeout=3, startup_channel=channel)
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=2)
        self.assertFalse(self.marker.exists())

    def test_identity_owner_must_verify_runtime_before_sending(self):
        from looking_glass.device_identity import resolve_device_identity
        gpu = 'GPU-12345678-1234-5678-9abc-1234567890ab'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        identity = resolve_device_identity(self.root, gpu, physical_uuids=(gpu,), environ=env)
        channel = api.StartupChannel(max_bytes=32)
        thread, result = self.launch(channel, """
            from looking_glass.supervisor import OwnerSession, StartupChannelError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            try:
                session.send_startup(lease, b'should-not-send')
            except StartupChannelError:
                pass
            else:
                raise AssertionError('unverified identity sent a frame')
        """, identity=identity, env={'CUDA_VISIBLE_DEVICES': gpu})
        with self.assertRaises(api.StartupChannelError):
            channel.receive(timeout=6)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(result['value'].clean)
        self.assertIsNotNone(result['value'].exit_code)
        self.assertTrue(channel.wait_closed(timeout=3))
        from looking_glass import device_lease
        with device_lease._state_lock:
            self.assertFalse(any(lease.path == identity.lock_path
                                 for lease in device_lease._active.values()))
        self.assertTrue(identity.marker_path.exists())

    def test_receiver_capacity_is_bounded(self):
        channels = []
        try:
            for _ in range(16):
                channels.append(api.StartupChannel(max_bytes=1))
            with self.assertRaises(api.StartupChannelError):
                api.StartupChannel(max_bytes=1)
        finally:
            for channel in channels:
                channel.close()


if __name__ == "__main__":
    unittest.main()
