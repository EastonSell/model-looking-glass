"""Development-only Windows source and empty-cache guard primitives.

These are operational protections against ordinary concurrent modification,
not security boundaries against a same-account process. They do not launch or
authorize a v2 study, check committed content, or attest to native dependencies.
"""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time


class SourceGuardError(RuntimeError):
    """A source or disposable-cache protection could not be established."""


def _checked_path(path):
    lexical = Path(os.path.abspath(path))
    for entry in (lexical, *lexical.parents):
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise SourceGuardError("reparse points and symlinks are unsupported")
    return lexical.resolve(strict=True)


def _kernel():
    if os.name != "nt":
        raise SourceGuardError("source guards require Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _acquire(kernel, path, *, directory=False):
    # Share reads only: cooperating readers can import, but new write/delete
    # handles and renames cannot be opened while this handle is alive.
    flags = 0x02000000 if directory else 0x80  # BACKUP_SEMANTICS / NORMAL
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, flags, None)
    if handle == wintypes.HANDLE(-1).value:
        raise SourceGuardError("cannot acquire read-only sharing protection")
    return handle


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority_class", wintypes.DWORD),
                ("scheduling_class", wintypes.DWORD)]


class _JobIoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_operations", "write_operations", "other_operations",
                 "read_bytes", "write_bytes", "other_bytes")]


class _JobExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", _JobBasicLimits), ("io", _JobIoCounters),
                ("process_memory_limit", ctypes.c_size_t), ("job_memory_limit", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _JobAccounting(ctypes.Structure):
    _fields_ = [("total_user", ctypes.c_int64), ("total_kernel", ctypes.c_int64),
                ("period_user", ctypes.c_int64), ("period_kernel", ctypes.c_int64),
                ("page_faults", wintypes.DWORD), ("total_processes", wintypes.DWORD),
                ("active_processes", wintypes.DWORD), ("terminated_processes", wintypes.DWORD)]


class _JobProcessList(ctypes.Structure):
    # A fixed bound makes an oversized or incomplete inventory fail closed.
    _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                ("ids", ctypes.c_size_t * 256)]


class WindowsJob:
    """Kill assigned children when the owning process closes its job handle.

    A future launcher must keep children blocked on an inherited one-way start
    channel until assignment succeeds; this class alone cannot prevent work in
    the interval between process creation and assignment.
    """
    def __init__(self):
        self._kernel = _kernel()
        self._kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        self._kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, wintypes.INT,
                                                         wintypes.LPVOID, wintypes.DWORD)
        self._kernel.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self._kernel.TerminateJobObject.restype = wintypes.BOOL
        self._kernel.QueryInformationJobObject.argtypes = (wintypes.HANDLE, wintypes.INT,
                                                            wintypes.LPVOID, wintypes.DWORD,
                                                            ctypes.POINTER(wintypes.DWORD))
        self._kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self._kernel.OpenProcess.restype = wintypes.HANDLE
        self._kernel.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
        self._kernel.IsProcessInJob.restype = wintypes.BOOL
        self._kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        self._kernel.WaitForSingleObject.restype = wintypes.DWORD
        self._processes = {}
        self._unconfirmed = False
        self._stopping = False
        self._confirmed = False
        self._closing_process = False
        self._closing_job = False
        self._handle = self._kernel.CreateJobObjectW(None, None)
        if not self._handle:
            raise SourceGuardError("cannot create the process job")
        try:
            if os.get_handle_inheritable(self._handle):
                raise SourceGuardError("process job handle must not be inheritable")
            limits = _JobExtendedLimits()
            limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self._kernel.SetInformationJobObject(self._handle, 9, ctypes.byref(limits),
                                                        ctypes.sizeof(limits)):
                raise SourceGuardError("cannot configure the process job")
        except BaseException:
            self.close()
            raise

    def assign(self, process):
        if self._handle is None or self._closing_job or self._stopping or process.poll() is not None:
            raise SourceGuardError("job or child is not active")
        if not self._kernel.AssignProcessToJobObject(self._handle, int(process._handle)):
            raise SourceGuardError("cannot assign the child to the process job")
        try:
            self._remember(process.pid)
        except BaseException:
            # The assignment already happened; later cleanup must terminate it,
            # but cannot claim a complete lifetime inventory.
            self._unconfirmed = True
            raise

    def _remember(self, pid):
        if pid in self._processes:
            raise SourceGuardError("process identifier was reused in the job")
        # Synchronization confirms termination; query rights check membership.
        handle = self._kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            raise SourceGuardError("cannot retain a job process handle")
        self._processes[pid] = handle
        member = wintypes.BOOL()
        if not self._kernel.IsProcessInJob(handle, self._handle, ctypes.byref(member)) or not member.value:
            raise SourceGuardError("job process membership could not be confirmed")

    def _accounting(self):
        info = _JobAccounting()
        if not self._kernel.QueryInformationJobObject(self._handle, 1, ctypes.byref(info),
                                                      ctypes.sizeof(info), None):
            raise SourceGuardError("cannot inspect process job termination")
        return info

    def _inventory(self):
        before = self._accounting()
        listing = _JobProcessList()
        if not self._kernel.QueryInformationJobObject(self._handle, 3, ctypes.byref(listing),
                                                      ctypes.sizeof(listing), None):
            raise SourceGuardError("cannot obtain a bounded process inventory")
        ids = list(listing.ids[:listing.count])
        if (listing.count > 256 or listing.count != listing.assigned or len(set(ids)) != listing.count
                or before.active_processes != listing.count):
            raise SourceGuardError("process inventory is incomplete or changed")
        for pid in ids:
            if pid not in self._processes:
                self._remember(pid)
            else:
                member = wintypes.BOOL()
                if (not self._kernel.IsProcessInJob(self._processes[pid], self._handle, ctypes.byref(member))
                        or not member.value):
                    raise SourceGuardError("retained process is no longer the listed job member")
        after = self._accounting()
        if (after.total_processes != before.total_processes or after.active_processes != listing.count
                or after.total_processes != len(self._processes)):
            raise SourceGuardError("lifetime process inventory could not be established")
        return after.total_processes

    def _close_processes(self):
        if self._closing_process:
            raise SourceGuardError("a process-handle close has an unknown outcome")
        failures = []
        for pid, handle in tuple(self._processes.items()):
            self._closing_process = True
            closed = self._kernel.CloseHandle(handle)
            if closed:
                del self._processes[pid]
            else:
                failures.append(pid)
            self._closing_process = False
        if failures:
            raise SourceGuardError("cannot close all retained process handles")

    def terminate_and_wait(self, timeout=5):
        """Terminate and confirm all inventoried job process handles signal.

        Missing, raced or unobservable lifetime members make confirmation fail;
        termination is still requested, and retained handles remain retryable.
        """
        if self._handle is None or self._closing_job or not 0 < timeout <= 10:
            raise SourceGuardError("invalid job termination or timeout")
        if self._confirmed:
            self._close_processes()
            return
        inventory_error = None
        if not self._stopping:
            self._stopping = True
            try:
                self._inventory_total = self._inventory()
            except BaseException as error:
                self._unconfirmed = True
                inventory_error = error
        try:
            terminated = self._kernel.TerminateJobObject(self._handle, 1)
        except BaseException as error:
            if (isinstance(inventory_error, (KeyboardInterrupt, SystemExit))
                    and not isinstance(error, (KeyboardInterrupt, SystemExit))):
                raise inventory_error from error
            raise
        if not terminated:
            error = SourceGuardError("cannot terminate the process job")
            if isinstance(inventory_error, (KeyboardInterrupt, SystemExit)):
                raise inventory_error from error
            raise error
        if inventory_error is not None:
            raise inventory_error
        if self._unconfirmed:
            raise SourceGuardError("job lifetime process inventory remains unconfirmed")
        deadline = time.monotonic() + timeout
        while True:
            info = self._accounting()
            if info.total_processes != self._inventory_total:
                self._unconfirmed = True
                raise SourceGuardError("a process joined after the inventory")
            if info.active_processes == 0:
                break
            if time.monotonic() >= deadline:
                raise SourceGuardError("process job did not drain")
            time.sleep(0.05)
        for handle in self._processes.values():
            remaining = max(0, min(10000, int((deadline - time.monotonic()) * 1000)))
            result = self._kernel.WaitForSingleObject(handle, remaining)
            if result != 0:  # WAIT_TIMEOUT and WAIT_FAILED both leave cleanup unconfirmed.
                raise SourceGuardError("a job process handle did not signal")
        final = self._accounting()
        if final.total_processes != self._inventory_total or final.active_processes != 0:
            self._unconfirmed = True
            raise SourceGuardError("job membership changed during termination")
        self._confirmed = True
        self._close_processes()

    def close(self):
        self._close_processes()
        if self._closing_job:
            raise SourceGuardError("a job-handle close has an unknown outcome")
        if self._handle is not None:
            self._closing_job = True
            closed = self._kernel.CloseHandle(self._handle)
            if not closed:
                self._closing_job = False
                raise SourceGuardError("cannot close the process job")
            self._handle = None
            self._closing_job = False

    def __enter__(self):
        if self._handle is None:
            raise SourceGuardError("process job is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class SourceFileLease:
    """Hold directory and file share protections for a fixed source list."""
    def __init__(self, root, files):
        try:
            self.root = _checked_path(root)
        except OSError as error:
            raise SourceGuardError("source root is unavailable") from error
        if not self.root.is_dir() or type(files) not in (tuple, list) or not 0 < len(files) <= 4096:
            raise SourceGuardError("a bounded nonempty source list is required")
        paths = []
        for file in files:
            lexical = Path(os.path.abspath(file))
            if not lexical.is_relative_to(self.root):
                raise SourceGuardError("source path lies outside its root")
            try:
                resolved = _checked_path(lexical)
            except OSError as error:
                raise SourceGuardError("source file is unavailable") from error
            if not resolved.is_relative_to(self.root) or not resolved.is_file() or resolved in paths:
                raise SourceGuardError("source path must be a unique file within its root")
            paths.append(resolved)
        self.files = tuple(paths)
        self._handles = []
        self._kernel = None
        self._active = False
        self._closing = False

    def __enter__(self):
        if self._kernel is not None:
            raise SourceGuardError("source lease is single-use")
        self._kernel = _kernel()
        directories = {self.root}
        for path in self.files:
            parent = path.parent
            while parent != self.root:
                directories.add(parent)
                if len(directories) + len(self.files) > 8192:
                    raise SourceGuardError("source protection requires too many handles")
                parent = parent.parent
        try:
            for directory in sorted(directories, key=lambda p: (len(p.parts), str(p))):
                self._handles.append(_acquire(self._kernel, directory, directory=True))
            for file in sorted(self.files, key=str):
                self._handles.append(_acquire(self._kernel, file))
            self._active = True
            return self
        except BaseException:
            self._close()
            raise

    def _close(self):
        # Partial cleanup is quarantine, not an active complete lease.
        self._active = False
        if self._closing:
            raise SourceGuardError("a source-handle close has an unknown outcome")
        while self._handles:
            handle = self._handles[-1]
            self._closing = True
            try:
                closed = self._kernel.CloseHandle(handle)
            except OSError as error:
                raise SourceGuardError("source protection could not be released") from error
            if not closed:
                self._closing = False
                raise SourceGuardError("source protection could not be released")
            # Remove each confirmed close immediately; a failure retains the
            # remaining handles and requires quarantine until cleanup succeeds.
            self._handles.pop()
            self._closing = False

    def close(self):
        """Retry release of any handles retained after a cleanup failure."""
        if self._kernel is None:
            raise SourceGuardError("source lease was never entered")
        self._close()

    def handles(self):
        """Return live protection handles for restricted child inheritance."""
        if not self._active:
            raise SourceGuardError("source lease is not active")
        return tuple(self._handles)

    def __exit__(self, exc_type, exc, tb):
        self._close()
        return False


def _icacls(path, *arguments):
    kernel = _kernel()
    kernel.GetSystemDirectoryW.argtypes = (wintypes.LPWSTR, wintypes.UINT)
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not length or length >= len(buffer):
        raise SourceGuardError("Windows system directory is unavailable")
    executable = _checked_path(Path(buffer.value) / "icacls.exe")
    result = subprocess.run([str(executable), str(path), *arguments],
                            capture_output=True, text=True)
    if result.returncode:
        raise SourceGuardError("cache ACL change failed")


def _probe_denial(prefix):
    code = ("import pathlib,sys\n"
            "try: pathlib.Path(sys.argv[1]).write_bytes(b'x')\n"
            "except PermissionError: raise SystemExit(0)\n"
            "raise SystemExit(1)\n")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code,
                             str(prefix / "creation-probe")], capture_output=True)
    if result.returncode or (prefix / "creation-probe").exists():
        raise SourceGuardError("a separate process could create cache entries")


class PrivateEmptyCache:
    """Protect a fresh, empty disposable cache prefix until context exit."""
    def __init__(self, parent):
        try:
            self.parent = _checked_path(parent)
        except OSError as error:
            raise SourceGuardError("cache parent is unavailable") from error
        if not self.parent.is_dir():
            raise SourceGuardError("cache parent must be a directory")
        self.path = None
        self._kernel = None
        self._handle = None
        self._denied = False
        self._used = False
        self._active = False
        self._closing = False

    def __enter__(self):
        if self._used:
            raise SourceGuardError("cache protection is single-use")
        self._used = True
        self._kernel = _kernel()
        self.path = Path(tempfile.mkdtemp(prefix="v2-cache-", dir=self.parent))
        try:
            # A directory share handle alone does not prevent new children.
            # Deny add-file/add-directory, then independently probe denial.
            # Attempt removal even when the ACL command has an unknown outcome.
            self._denied = True
            _icacls(self.path, "/deny", "*S-1-1-0:(WD,AD)")
            self._handle = _acquire(self._kernel, self.path, directory=True)
            _probe_denial(self.path)
            if any(self.path.iterdir()):
                raise SourceGuardError("cache prefix was not empty")
            self._active = True
            return self.path
        except BaseException as error:
            try:
                self._cleanup()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _cleanup(self):
        self._active = False
        if self._closing:
            raise SourceGuardError("a cache-handle close has an unknown outcome")
        if self._handle is not None:
            self._closing = True
            closed = self._kernel.CloseHandle(self._handle)
            if not closed:
                self._closing = False
                raise SourceGuardError("cache directory protection could not be released")
            self._handle = None
            self._closing = False
        if self._denied:
            _icacls(self.path, "/remove:d", "*S-1-1-0")
            self._denied = False
        if self.path is not None and self.path.exists():
            if any(self.path.iterdir()):
                raise SourceGuardError("nonempty cache prefix retained for inspection")
            self.path.rmdir()

    def __exit__(self, exc_type, exc, tb):
        self._cleanup()
        return False

    def close(self):
        """Retry releasing an interrupted or failed cache cleanup."""
        if not self._used:
            raise SourceGuardError("cache protection was never entered")
        self._cleanup()

    def handles(self):
        if not self._active:
            raise SourceGuardError("cache protection is not active")
        return (self._handle,)


class InheritableHandles:
    """Duplicate live guards for a child restricted to an explicit handle list."""
    def __init__(self, handles):
        if (type(handles) not in (tuple, list) or not 0 < len(handles) <= 8193 or
                any(type(handle) is not int or handle <= 0 for handle in handles) or
                len(set(handles)) != len(handles)):
            raise SourceGuardError("a bounded list of distinct live handles is required")
        self._originals = tuple(handles)
        self._duplicates = []
        self._kernel = None
        self._closing = False

    def __enter__(self):
        if self._kernel is not None:
            raise SourceGuardError("inheritable handle list is single-use")
        self._kernel = _kernel()
        self._kernel.GetCurrentProcess.argtypes = ()
        self._kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self._kernel.DuplicateHandle.argtypes = (wintypes.HANDLE, wintypes.HANDLE,
                                                  wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE),
                                                  wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self._kernel.DuplicateHandle.restype = wintypes.BOOL
        process = self._kernel.GetCurrentProcess()
        try:
            for handle in self._originals:
                duplicate = wintypes.HANDLE()
                if not self._kernel.DuplicateHandle(process, handle, process, ctypes.byref(duplicate),
                                                    0, True, 2):  # DUPLICATE_SAME_ACCESS
                    raise SourceGuardError("cannot duplicate a protection handle")
                self._duplicates.append(duplicate.value)
                if not os.get_handle_inheritable(duplicate.value):
                    raise SourceGuardError("duplicated protection is not inheritable")
            return tuple(self._duplicates)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._kernel is None:
            raise SourceGuardError("inheritable handle list was never entered")
        if self._closing:
            raise SourceGuardError("a duplicate-handle close has an unknown outcome")
        while self._duplicates:
            self._closing = True
            closed = self._kernel.CloseHandle(self._duplicates[-1])
            if not closed:
                self._closing = False
                raise SourceGuardError("cannot release an inherited-handle duplicate")
            self._duplicates.pop()
            self._closing = False

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def duplicate_to_process(handles, process):
    """Duplicate protections directly into a live child, noninheritable there.

    No inheritable parent duplicates are created. On partial failure the child
    owns any handles already duplicated; the caller must terminate and reap it
    before releasing the original protections.
    """
    if (type(handles) not in (tuple, list) or not 0 < len(handles) <= 8193 or
            any(type(handle) is not int or handle <= 0 for handle in handles) or
            len(set(handles)) != len(handles) or process.poll() is not None):
        raise SourceGuardError("a bounded set of live handles and a live child are required")
    kernel = _kernel()
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.DuplicateHandle.argtypes = (wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
                                       ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
                                       wintypes.BOOL, wintypes.DWORD)
    kernel.DuplicateHandle.restype = wintypes.BOOL
    current = kernel.GetCurrentProcess()
    result = []
    for handle in handles:
        if os.get_handle_inheritable(handle):
            raise SourceGuardError("parent protection handles must not be inheritable")
        duplicate = wintypes.HANDLE()
        if not kernel.DuplicateHandle(current, handle, int(process._handle),
                                      ctypes.byref(duplicate), 0, False, 2):
            raise SourceGuardError("cannot duplicate protection into child")
        result.append(duplicate.value)
    return tuple(result)
