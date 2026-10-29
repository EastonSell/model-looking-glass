"""Cross-process tests for the cooperative GPU device lease; no GPU is used."""

import gc
import importlib
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


def _lease_api():
    return importlib.import_module("looking_glass.device_lease")


def _hold_lease(path, ready, stop, drop_reference=False):
    try:
        api = _lease_api()
        lease = api.DeviceLease.acquire(path, timeout=5)
        if drop_reference:
            del lease
            gc.collect()
        ready.send(("held", os.getpid()))
        ready.close()
        stop.wait(20)
        if not drop_reference:
            lease.release()
    except BaseException as exc:
        try:
            ready.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise


def _race_for_lease(path, start, stop, results):
    api = _lease_api()
    if not start.wait(15):
        results.put("start-timeout")
        return
    try:
        lease = api.DeviceLease.acquire(path, timeout=1)
    except api.LeaseTimeoutError:
        results.put("timeout")
        return
    results.put("held")
    stop.wait(20)
    lease.release()


def _ambiguous_release(path, result):
    api = _lease_api()
    lease = api.DeviceLease.acquire(path, timeout=5)
    outcome = []
    with mock.patch.object(api.os, "close", side_effect=OSError("simulated close failure")):
        try:
            lease.release()
        except api.LeaseError:
            outcome.append("release-failed")
    try:
        lease.check()
    except api.LeaseError:
        outcome.append("check-rejected")
    try:
        lease.release()
    except api.LeaseError:
        outcome.append("retry-rejected")
    result.send(outcome)
    result.close()


def _owner_with_descendant(path, ready_file, stop_file, done_file, result):
    api = _lease_api()
    lease = api.DeviceLease.acquire(path, timeout=5)
    script = (
        "import pathlib,sys,time; "
        "ready=pathlib.Path(sys.argv[1]); stop=pathlib.Path(sys.argv[2]); "
        "ready.write_text('ready'); end=time.monotonic()+15; "
        "exec('while not stop.exists() and time.monotonic() < end:\\n time.sleep(0.02)'); "
        "pathlib.Path(sys.argv[3]).write_text('done')"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, ready_file, stop_file, done_file],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=False,
    )
    deadline = time.monotonic() + 10
    while not Path(ready_file).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    result.send(("ready" if Path(ready_file).exists() else "error", child.pid))
    result.close()
    # Simulate owner process exit while the descendant is still alive. The
    # descendant must not inherit a descriptor which prolongs device ownership.
    os._exit(0)


def _owner_with_fork_child(path, ready_file, stop_file, done_file, result):
    api = _lease_api()
    lease = api.DeviceLease.acquire(path, timeout=5)
    child = os.fork()
    if child == 0:
        try:
            try:
                lease.check()
            except api.LeaseOwnershipError:
                Path(ready_file).write_text("rejected")
            else:
                Path(ready_file).write_text("incorrectly-owned")
            deadline = time.monotonic() + 15
            while not Path(stop_file).exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            Path(done_file).write_text("done")
        finally:
            os._exit(0)
    deadline = time.monotonic() + 10
    while not Path(ready_file).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    result.send(("ready" if Path(ready_file).exists() else "error", child))
    result.close()
    os._exit(0)


class DeviceLeaseTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = _lease_api()
        except ModuleNotFoundError as exc:
            if exc.name == "looking_glass.device_lease":
                self.fail("the device lease module is not implemented")
            raise
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "device.lock"
        self.ctx = multiprocessing.get_context("spawn")

    def _start_owner(self, drop_reference=False):
        recv, send = self.ctx.Pipe(duplex=False)
        stop = self.ctx.Event()
        proc = self.ctx.Process(
            target=_hold_lease,
            args=(str(self.path), send, stop, drop_reference),
        )
        proc.start()
        send.close()

        def cleanup():
            if proc.is_alive():
                stop.set()
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            recv.close()

        self.addCleanup(cleanup)
        self.assertTrue(recv.poll(15), "lease owner did not start")
        status, detail = recv.recv()
        self.assertEqual(status, "held", detail)
        return proc, stop

    def test_acquire_release_persists_lock_file_and_prevents_nesting(self):
        lease = self.api.DeviceLease.acquire(self.path, timeout=1)
        self.assertTrue(self.path.is_file())
        with self.assertRaises(self.api.LeaseTimeoutError):
            self.api.DeviceLease.acquire(self.path, timeout=0)
        lease.release()
        self.assertTrue(self.path.is_file(), "the persistent lock file must not be unlinked")
        lease2 = self.api.DeviceLease.acquire(self.path, timeout=1)
        lease2.release()

    def test_cross_process_timeout_then_release(self):
        proc, stop = self._start_owner()
        start = time.monotonic()
        with self.assertRaises(self.api.LeaseTimeoutError):
            self.api.DeviceLease.acquire(self.path, timeout=0.2)
        self.assertLess(time.monotonic() - start, 5)
        stop.set()
        proc.join(10)
        self.assertEqual(proc.exitcode, 0)
        lease = self.api.DeviceLease.acquire(self.path, timeout=5)
        lease.release()

    def test_wait_can_be_cancelled_and_pre_cancel_does_not_create_file(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(self.api.LeaseCancelledError):
            self.api.DeviceLease.acquire(self.path, timeout=5, cancel_event=cancel)
        self.assertFalse(self.path.exists())
        self._start_owner()
        cancel.clear()
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        start = time.monotonic()
        with self.assertRaises(self.api.LeaseCancelledError):
            self.api.DeviceLease.acquire(self.path, timeout=5, cancel_event=cancel)
        self.assertLess(time.monotonic() - start, 3)

    def test_killed_owner_releases_for_waiters(self):
        proc, _ = self._start_owner()
        proc.terminate()
        proc.join(10)
        self.assertFalse(proc.is_alive())
        lease = self.api.DeviceLease.acquire(self.path, timeout=5)
        lease.release()

    def test_dropped_python_reference_does_not_release_owner_lock(self):
        proc, stop = self._start_owner(drop_reference=True)
        with self.assertRaises(self.api.LeaseTimeoutError):
            self.api.DeviceLease.acquire(self.path, timeout=0.2)
        stop.set()
        proc.join(10)
        self.assertEqual(proc.exitcode, 0)
        lease = self.api.DeviceLease.acquire(self.path, timeout=5)
        lease.release()

    def test_concurrent_first_creation_admits_only_one_owner(self):
        start, stop = self.ctx.Event(), self.ctx.Event()
        results = self.ctx.Queue()
        procs = [self.ctx.Process(target=_race_for_lease, args=(str(self.path), start, stop, results)) for _ in range(2)]
        for proc in procs:
            proc.start()

        def cleanup():
            stop.set()
            for proc in procs:
                proc.join(10)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(10)
            results.close()

        self.addCleanup(cleanup)
        start.set()
        outcomes = [results.get(timeout=20), results.get(timeout=20)]
        self.assertCountEqual(outcomes, ["held", "timeout"])

    def test_hard_link_alias_still_contends_for_the_same_file(self):
        lease = self.api.DeviceLease.acquire(self.path, timeout=1)
        self.addCleanup(lease.release)
        alias = self.path.with_name("alias.lock")
        try:
            os.link(self.path, alias)
        except OSError as exc:
            self.skipTest(f"hard links unavailable: {exc}")
        with self.assertRaises(self.api.LeaseTimeoutError):
            self.api.DeviceLease.acquire(alias, timeout=0)

    def test_invalid_timeouts_and_relative_paths_are_rejected(self):
        for timeout in (-1, math.nan, math.inf):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.api.DeviceLease.acquire(self.path, timeout=timeout)
        with self.assertRaises(ValueError):
            self.api.DeviceLease.acquire("relative.lock", timeout=1)

    def test_ambiguous_release_failure_invalidates_lease(self):
        recv, send = self.ctx.Pipe(duplex=False)
        proc = self.ctx.Process(target=_ambiguous_release, args=(str(self.path), send))
        proc.start()
        send.close()
        try:
            self.assertTrue(recv.poll(15))
            self.assertEqual(recv.recv(), ["release-failed", "check-rejected", "retry-rejected"])
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            recv.close()

    def test_timeout_bounds_waiting_for_internal_mutex(self):
        outcomes = []
        entered = threading.Event()

        def acquire():
            entered.set()
            try:
                lease = self.api.DeviceLease.acquire(self.path, timeout=0.1)
            except self.api.LeaseTimeoutError:
                outcomes.append("timeout")
            else:
                outcomes.append("acquired")
                lease.release()

        self.api._state_lock.acquire()
        thread = threading.Thread(target=acquire)
        try:
            thread.start()
            self.assertTrue(entered.wait(3))
            thread.join(1)
            completed_while_locked = not thread.is_alive()
        finally:
            self.api._state_lock.release()
            thread.join(5)
        self.assertTrue(completed_while_locked, "deadline did not bound mutex wait")
        self.assertEqual(outcomes, ["timeout"])

    def test_large_poll_interval_remains_cancellable(self):
        self._start_owner()
        cancel = threading.Event()
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(self.api.LeaseCancelledError):
            self.api.DeviceLease.acquire(self.path, timeout=1e308, poll_interval=1e308, cancel_event=cancel)

    def test_exec_child_does_not_keep_dead_owners_lease(self):
        recv, send = self.ctx.Pipe(duplex=False)
        ready = Path(self.tmp.name) / "descendant.ready"
        stop = Path(self.tmp.name) / "descendant.stop"
        done = Path(self.tmp.name) / "descendant.done"
        proc = self.ctx.Process(target=_owner_with_descendant, args=(str(self.path), str(ready), str(stop), str(done), send))
        proc.start()
        send.close()
        try:
            self.assertTrue(recv.poll(15))
            status, _ = recv.recv()
            self.assertEqual(status, "ready")
            proc.join(10)
            self.assertEqual(proc.exitcode, 0)
            lease = self.api.DeviceLease.acquire(self.path, timeout=3)
            lease.release()
        finally:
            stop.touch()
            deadline = time.monotonic() + 5
            while not done.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            recv.close()

    @unittest.skipUnless(hasattr(os, "fork"), "fork is unavailable")
    def test_fork_child_cannot_use_or_prolong_parent_lease(self):
        recv, send = self.ctx.Pipe(duplex=False)
        ready = Path(self.tmp.name) / "fork.ready"
        stop = Path(self.tmp.name) / "fork.stop"
        done = Path(self.tmp.name) / "fork.done"
        proc = self.ctx.Process(target=_owner_with_fork_child, args=(str(self.path), str(ready), str(stop), str(done), send))
        proc.start()
        send.close()
        try:
            self.assertTrue(recv.poll(15))
            status, _ = recv.recv()
            self.assertEqual(status, "ready")
            self.assertEqual(ready.read_text(), "rejected")
            proc.join(10)
            self.assertEqual(proc.exitcode, 0)
            lease = self.api.DeviceLease.acquire(self.path, timeout=3)
            lease.release()
        finally:
            stop.touch()
            deadline = time.monotonic() + 5
            while not done.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            recv.close()

    def test_slow_open_cannot_return_a_lease_after_the_deadline(self):
        original_open = self.api.os.open

        def slow_open(*args, **kwargs):
            time.sleep(0.15)
            return original_open(*args, **kwargs)

        with mock.patch.object(self.api.os, "open", side_effect=slow_open):
            with self.assertRaises(self.api.LeaseTimeoutError):
                self.api.DeviceLease.acquire(self.path, timeout=0.05)
        lease = self.api.DeviceLease.acquire(self.path, timeout=1)
        lease.release()

    def test_cancellation_bounds_waiting_for_internal_mutex(self):
        outcomes = []
        cancel = threading.Event()
        entered = threading.Event()

        def acquire():
            entered.set()
            try:
                self.api.DeviceLease.acquire(self.path, timeout=5, cancel_event=cancel)
            except self.api.LeaseCancelledError:
                outcomes.append("cancelled")

        self.api._state_lock.acquire()
        thread = threading.Thread(target=acquire)
        try:
            thread.start()
            self.assertTrue(entered.wait(3))
            cancel.set()
            thread.join(1)
            completed_while_locked = not thread.is_alive()
        finally:
            self.api._state_lock.release()
            thread.join(5)
        self.assertTrue(completed_while_locked, "cancellation did not bound mutex wait")
        self.assertEqual(outcomes, ["cancelled"])


if __name__ == "__main__":
    unittest.main()
