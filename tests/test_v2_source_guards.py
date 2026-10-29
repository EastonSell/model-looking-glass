"""Windows-only development checks for source and empty-cache protections."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


GUARDS = Path(__file__).resolve().parents[1] / "scripts" / "v2_source_guards.py"


def _load_guards():
    spec = importlib.util.spec_from_file_location("v2_source_guards_test", GUARDS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attempt(code, *args):
    return subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, *map(str, args)],
                          capture_output=True, text=True)


@unittest.skipUnless(os.name == "nt", "Windows file-sharing and ACL policy")
class V2SourceGuardTests(unittest.TestCase):
    def test_source_file_cannot_be_rewritten_or_renamed_during_lease(self):
        guards = _load_guards()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "package"
            folder.mkdir()
            source = folder / "module.py"
            source.write_text("original\n")
            with guards.SourceFileLease(root, [source]):
                write = _attempt("import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('changed')", source)
                rename = _attempt("import os,sys; os.rename(sys.argv[1],sys.argv[2])", folder, root / "moved")
                self.assertNotEqual(write.returncode, 0)
                self.assertNotEqual(rename.returncode, 0)
                self.assertEqual(source.read_text(), "original\n")
            source.write_text("after\n")
            self.assertEqual(source.read_text(), "after\n")

    def test_existing_writer_prevents_acquiring_a_source_lease(self):
        guards = _load_guards()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_text("source\n")
            with source.open("a"):
                with self.assertRaises(guards.SourceGuardError):
                    with guards.SourceFileLease(root, [source]):
                        pass

    def test_failed_handle_close_is_retained_for_explicit_retry(self):
        guards = _load_guards()
        original_kernel = guards._kernel()
        class FailingOnce:
            CreateFileW = original_kernel.CreateFileW
            def __init__(self):
                self.fail = True
            def CloseHandle(self, handle):
                if self.fail:
                    self.fail = False
                    return 0
                return original_kernel.CloseHandle(handle)
        proxy = FailingOnce()
        guards._kernel = lambda: proxy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_text("source\n")
            lease = guards.SourceFileLease(root, [source])
            with self.assertRaises(guards.SourceGuardError):
                lease.handles()
            lease.__enter__()
            self.assertEqual(len(lease.handles()), 2)
            with self.assertRaises(guards.SourceGuardError):
                lease.__exit__(None, None, None)
            self.assertEqual(len(lease._handles), 2)
            with self.assertRaises(guards.SourceGuardError):
                lease.handles()
            lease.close()
            self.assertEqual(lease._handles, [])
            with self.assertRaises(guards.SourceGuardError):
                lease.handles()
            source.write_text("released\n")

    def test_private_prefix_denies_child_creation_and_cleans_up(self):
        guards = _load_guards()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            guard = guards.PrivateEmptyCache(parent)
            with self.assertRaises(guards.SourceGuardError):
                guard.handles()
            with guard as prefix:
                self.assertTrue(prefix.is_dir())
                self.assertEqual(len(guard.handles()), 1)
                write = _attempt("import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('bad')", prefix / "bad")
                rename = _attempt("import os,sys; os.rename(sys.argv[1],sys.argv[2])", prefix, parent / "moved")
                self.assertNotEqual(write.returncode, 0)
                self.assertNotEqual(rename.returncode, 0)
                self.assertEqual(list(prefix.iterdir()), [])
            self.assertFalse(prefix.exists())
            with self.assertRaises(guards.SourceGuardError):
                guard.handles()

    def test_job_setup_failure_closes_the_created_handle(self):
        from unittest.mock import patch
        guards = _load_guards()
        kernel = guards._kernel()
        class TrackingKernel:
            CreateJobObjectW = kernel.CreateJobObjectW
            SetInformationJobObject = kernel.SetInformationJobObject
            AssignProcessToJobObject = kernel.AssignProcessToJobObject
            TerminateJobObject = kernel.TerminateJobObject
            QueryInformationJobObject = kernel.QueryInformationJobObject
            def __init__(self):
                self.closed = []
            def CloseHandle(self, handle):
                self.closed.append(handle)
                return kernel.CloseHandle(handle)
        tracking = TrackingKernel()
        guards._kernel = lambda: tracking
        with patch.object(guards.os, "get_handle_inheritable", side_effect=OSError("probe failed")):
            with self.assertRaises(OSError):
                guards.WindowsJob()
        self.assertEqual(len(tracking.closed), 1)

    def test_terminate_and_wait_drains_the_entire_job(self):
        import ctypes
        from ctypes import wintypes
        guards = _load_guards()
        base_python = str(Path(sys.base_prefix) / "python.exe")
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "grandchild.txt"
            code = f'''
import pathlib, subprocess, sys, threading
sys.stdin.readline()
child = subprocess.Popen([{base_python!r}, "-I", "-S", "-B", "-c",
                          "import threading; threading.Event().wait(3600)"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
pathlib.Path({str(ready)!r}).write_text(str(child.pid))
threading.Event().wait(3600)
'''
            child = subprocess.Popen([base_python, "-I", "-S", "-B", "-c", code],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            grandchild = None
            job = guards.WindowsJob()
            try:
                job.assign(child)
                child.stdin.write(b"go\n")
                child.stdin.close()
                child.stdin = None
                deadline = time.monotonic() + 10
                while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists())
                grandchild = kernel.OpenProcess(0x00100000, False, int(ready.read_text()))
                self.assertTrue(grandchild)
                job.terminate_and_wait()
                child.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(grandchild, 0), 0)
            finally:
                job.close()
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                if grandchild:
                    kernel.CloseHandle(grandchild)

    def test_remote_duplicates_protect_source_after_parent_lease_closes(self):
        guards = _load_guards()
        base_python = str(Path(sys.base_prefix) / "python.exe")
        code = ("import json,os,sys\n"
                "handles=json.loads(sys.stdin.readline())\n"
                "assert all(not os.get_handle_inheritable(h) for h in handles)\n"
                "print('ready',flush=True)\n"
                "sys.stdin.readline()\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_text("original\n")
            child = subprocess.Popen([base_python, "-I", "-S", "-B", "-c", code],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, close_fds=True,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                with guards.SourceFileLease(root, [source]) as lease:
                    self.assertTrue(all(not os.get_handle_inheritable(h) for h in lease.handles()))
                    handles = guards.duplicate_to_process(lease.handles(), child)
                    child.stdin.write(__import__("json").dumps(handles) + "\n")
                    child.stdin.flush()
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                with self.assertRaises(PermissionError):
                    source.write_text("changed\n")
                child.stdin.write("stop\n")
                child.stdin.flush()
                self.assertEqual(child.wait(timeout=5), 0)
                source.write_text("released\n")
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                child.stdin.close()
                child.stdout.close()
                child.stderr.close()

    def test_rejected_outside_path_and_exception_cleanup(self):
        guards = _load_guards()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside-does-not-need-to-exist.py"
            with self.assertRaises(guards.SourceGuardError):
                with guards.SourceFileLease(root, [outside]):
                    pass
            try:
                with guards.PrivateEmptyCache(root) as prefix:
                    raise ValueError("deliberate")
            except ValueError:
                pass
            self.assertFalse(prefix.exists())

    def test_job_closes_child_if_owning_process_dies(self):
        import ctypes
        from ctypes import wintypes
        # The venv launcher may be a shim with its own child.  Test the actual
        # process that owns the job, and observe the actual assigned child.
        base_python = str(Path(sys.base_prefix) / "python.exe")
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready.txt"
            script = f'''
import importlib.util, pathlib, subprocess, sys
spec=importlib.util.spec_from_file_location("v2_guards", {str(GUARDS)!r})
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
job=g.WindowsJob()
child=subprocess.Popen([{base_python!r},"-I","-S","-B","-c",
                        "import threading; threading.Event().wait(3600)"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
try:
    job.assign(child)
    pathlib.Path({str(ready)!r}).write_text(str(child.pid))
    sys.stdin.read()
finally:
    job.close()
    child.wait(timeout=10)
'''
            owner = subprocess.Popen([base_python, "-I", "-S", "-B", "-c", script],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            child_handle = None
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not ready.exists():
                    owner.kill()
                    _, error = owner.communicate(timeout=5)
                    self.fail(f"job owner failed to start: {error.decode(errors='replace')}")
                pid = int(ready.read_text())
                child_handle = kernel.OpenProcess(0x00100000 | 0x0001, False, pid)
                self.assertTrue(child_handle)
                owner.kill()
                owner.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(child_handle, 10000), 0)
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=5)
                if child_handle:
                    if kernel.WaitForSingleObject(child_handle, 0) != 0:
                        kernel.TerminateProcess(child_handle, 1)
                    kernel.CloseHandle(child_handle)
                if owner.stdin is not None:
                    owner.stdin.close()
                if owner.stderr is not None:
                    owner.stderr.close()

    def test_inherited_file_guards_outlive_the_owner_until_child_exits(self):
        import ctypes
        from ctypes import wintypes
        base_python = str(Path(sys.base_prefix) / "python.exe")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ready = root / "source.py", root / "ready.txt"
            source.write_text("original\n")
            script = f'''
import importlib.util, pathlib, subprocess, sys
spec=importlib.util.spec_from_file_location("v2_guards", {str(GUARDS)!r})
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
lease=g.SourceFileLease({str(root)!r}, [{str(source)!r}])
with lease:
    with g.InheritableHandles(lease.handles()) as handles:
        info=subprocess.STARTUPINFO(); info.lpAttributeList={{"handle_list":list(handles)}}
        child=subprocess.Popen([{base_python!r},"-I","-S","-B","-c",
            "import threading; threading.Event().wait(3600)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, startupinfo=info, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            pathlib.Path({str(ready)!r}).write_text(str(child.pid))
            sys.stdin.read()
        finally:
            child.terminate(); child.wait(timeout=10)
'''
            owner = subprocess.Popen([base_python, "-I", "-S", "-B", "-c", script],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            child_handle = None
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not ready.exists():
                    owner.kill()
                    _, error = owner.communicate(timeout=5)
                    self.fail(f"guard owner failed to start: {error.decode(errors='replace')}")
                child_handle = kernel.OpenProcess(0x00100000 | 0x0001, False, int(ready.read_text()))
                self.assertTrue(child_handle)
                owner.kill()
                owner.wait(timeout=5)
                with self.assertRaises(PermissionError):
                    source.write_text("changed\n")
                self.assertEqual(source.read_text(), "original\n")
                self.assertTrue(kernel.TerminateProcess(child_handle, 1))
                self.assertEqual(kernel.WaitForSingleObject(child_handle, 10000), 0)
                source.write_text("released\n")
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=5)
                if child_handle:
                    if kernel.WaitForSingleObject(child_handle, 0) != 0:
                        kernel.TerminateProcess(child_handle, 1)
                    kernel.CloseHandle(child_handle)
                if owner.stdin is not None:
                    owner.stdin.close()
                if owner.stderr is not None:
                    owner.stderr.close()


if __name__ == "__main__":
    unittest.main()
