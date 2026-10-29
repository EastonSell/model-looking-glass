"""Fail-closed supervision of one cooperative device-owning child process.

This module does not authenticate callers, control unrelated GPU users, inspect
CUDA, or prove that an owner has really drained work. The trusted owner must
attest only after its own service shutdown and synchronization succeed.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import threading
import time
import uuid

from .device_lease import DeviceLease
from .device_identity import DeviceIdentity, DeviceIdentityError, resolve_device_identity


_RUN_ENV = "MLG_OWNER_RUN_ID"
_LEASE_ENV = "MLG_DEVICE_LEASE_PATH"
_MARKER_ENV = "MLG_UNSAFE_MARKER_PATH"
_SUPERVISOR_ENV = "MLG_SUPERVISOR_PID"
_SUPERVISOR_BIRTH_ENV = "MLG_SUPERVISOR_BIRTH"
_IDENTITY_ENV = "MLG_DEVICE_IDENTITY"
_STARTUP_HANDLE_ENV = "MLG_STARTUP_CHANNEL_HANDLE"
_STARTUP_LIMIT_ENV = "MLG_STARTUP_CHANNEL_LIMIT"
_LAUNCH_HANDLE_ENV = "MLG_OWNER_LAUNCH_HANDLE"
_SUPERVISOR_ENV_KEYS = frozenset({
    _RUN_ENV, _LEASE_ENV, _MARKER_ENV, _SUPERVISOR_ENV, _SUPERVISOR_BIRTH_ENV,
    _IDENTITY_ENV, _STARTUP_HANDLE_ENV, _STARTUP_LIMIT_ENV, _LAUNCH_HANDLE_ENV,
})
_VERSION = 1
_watchdog_lock = threading.Lock()
_watchdogs = {}
_startup_spawn_lock = threading.Lock()
_startup_reader_slots = threading.BoundedSemaphore(16)
_MAX_STARTUP_BYTES = 65536
_LAUNCH_TIMEOUT = 30.0
_MAX_SPAWN_CUSTODIES = 64
_spawn_custody_lock = threading.Lock()
_spawn_custodies = {}

# This identifies a cooperative compatibility contract, not a security proof.
SUPERVISOR_LAUNCH_CUSTODY_CONTRACT = "gated-rooted-child-v1"


class SupervisorError(RuntimeError):
    """The owner lifecycle could not be safely coordinated."""


class UnsafeOwnerError(SupervisorError):
    """A prior or ambiguous owner prevents a new owner from starting."""


class StartupChannelError(SupervisorError):
    """A private startup message could not be delivered."""


class StartupChannel:
    """One bounded child-to-parent frame, with fail-closed startup handling.

    A dedicated daemon reads the synchronous pipe. Waiting callers never block
    inside a pipe read; see SUPERVISOR.md for its resource and trust limits.
    """

    def __init__(self, max_bytes=_MAX_STARTUP_BYTES):
        if type(max_bytes) is not int or not 1 <= max_bytes <= _MAX_STARTUP_BYTES:
            raise ValueError("max_bytes must be between 1 and 65536")
        read_fd = None
        owner_lock = None
        acquired = False
        try:
            if not (acquired := _startup_reader_slots.acquire(blocking=False)):
                raise StartupChannelError("startup reader capacity is exhausted")
            self._write_fd = None
            self._lock = threading.Lock()
            self._delivery_lock = threading.Lock()
            self._receive_lock = threading.Lock()
            self._launch_lock = threading.Lock()
            self._launch_authorized = False
            owner_lock = threading.Lock()
            self._reader_owner = "pending"
            self._reader_owner_lock = owner_lock
            self._failed = threading.Event()
            self._ready = threading.Event()
            self._reader_done = threading.Event()
            self._reader_closed = False
            self._max_bytes = max_bytes
            self._payload = None
            self._partial = bytearray()
            self._frame_valid = False
            self._claimed = False
            self._receive_claimed = False
            read_fd, self._write_fd = os.pipe()
            os.set_inheritable(read_fd, False)
            os.set_inheritable(self._write_fd, False)
            thread = threading.Thread(target=self._reader_entry, args=(read_fd,),
                                      name="looking-glass-startup-reader", daemon=True)
            thread.start()
        except BaseException:
            if not acquired:
                raise
            # start() can be interrupted after the OS thread actually starts.
            # Close the writer first, then transfer read-fd/slot cleanup exactly
            # once under the ownership lock; never close an active reader's fd.
            try:
                write_fd = getattr(self, "_write_fd", None)
                if write_fd is not None:
                    self._write_fd = None
                    os.close(write_fd)
            finally:
                if owner_lock is None:
                    try:
                        if read_fd is not None:
                            os.close(read_fd)
                    finally:
                        _startup_reader_slots.release()
                else:
                    with owner_lock:
                        if getattr(self, "_reader_owner", "pending") == "pending":
                            self._reader_owner = "revoked"
                            try:
                                if read_fd is not None:
                                    os.close(read_fd)
                            finally:
                                _startup_reader_slots.release()
            raise

    def _reader_entry(self, fd):
        with self._reader_owner_lock:
            if self._reader_owner != "pending":
                return
            self._reader_owner = "reader"
        self._read_frame(fd)

    def _read_frame(self, fd):
        data = self._partial
        part = None

        def read_exact(size):
            nonlocal part
            while True:
                with self._delivery_lock:
                    if self._failed.is_set():
                        raise StartupChannelError("startup channel was interrupted")
                    remaining = size - len(data)
                if remaining == 0:
                    return
                part = os.read(fd, remaining)
                with self._delivery_lock:
                    if not part or self._failed.is_set():
                        raise StartupChannelError("startup channel was interrupted")
                    data.extend(part)
                # Do not retain the previous immutable read result while the
                # next synchronous read may block.
                part = None

        try:
            read_exact(4)
            with self._delivery_lock:
                size = struct.unpack("!I", data)[0]
                self._clear_partial()
            if not 1 <= size <= self._max_bytes:
                raise StartupChannelError("invalid startup frame")
            read_exact(size)
            with self._delivery_lock:
                if self._failed.is_set():
                    raise StartupChannelError("startup channel was interrupted")
                self._payload = bytes(data)
                self._clear_partial()
                self._frame_valid = True
                self._ready.set()
        except BaseException:
            self._mark_failed()
            with self._delivery_lock:
                self._payload = None
                self._discard_partial()
            self._ready.set()
        finally:
            part = None
            try:
                with self._delivery_lock:
                    self._discard_partial()
            finally:
                try:
                    os.close(fd)
                    self._reader_closed = True
                except BaseException:
                    self._mark_failed()
                    with self._delivery_lock:
                        self._payload = None
                        self._discard_partial()
                    self._ready.set()
                finally:
                    _startup_reader_slots.release()
                    self._reader_done.set()

    def _clear_partial(self):
        # Caller holds _delivery_lock. Python cannot guarantee erasure of
        # immutable allocator copies or values already handed to callers.
        for index in range(len(self._partial)):
            self._partial[index] = 0
        self._partial.clear()

    def _discard_partial(self):
        # A wipe can fail even during memory exhaustion. Always release the
        # buffer reference, reader descriptor, and capacity slot on failure.
        try:
            self._clear_partial()
        except BaseException:
            try:
                self._partial.clear()
            except BaseException:
                pass

    def _mark_failed(self):
        # This lock contains no I/O. Cancellation and launch authorization
        # have a definite order even when a spawn or grant write is stalled.
        with self._launch_lock:
            self._failed.set()

    def _authorize_launch(self):
        with self._launch_lock:
            if self._failed.is_set():
                raise StartupChannelError("startup channel failed before launch authorization")
            self._launch_authorized = True

    def _claim(self):
        with self._lock:
            if self._claimed:
                raise StartupChannelError("startup channel is unavailable")
            if self._failed.is_set():
                if self._write_fd is not None:
                    fd = self._write_fd
                    self._write_fd = None
                    os.close(fd)
                raise StartupChannelError("startup channel is unavailable")
            self._claimed = True

    def _spawn(self, command, *, cwd, child_env, custody):
        # On Windows an inheritable handle is briefly needed for Popen's
        # explicit handle list. Cooperating spawns share this lock; callers
        # must also exclude other broad-inheritance spawns from the launcher.
        with _startup_spawn_lock, self._lock:
            if self._failed.is_set() or self._write_fd is None:
                raise StartupChannelError("startup channel is unavailable")
            fd = self._write_fd
            try:
                endpoint = _pipe_endpoint(fd)
                child_env[_STARTUP_HANDLE_ENV] = str(endpoint)
                child_env[_STARTUP_LIMIT_ENV] = str(self._max_bytes)
                _spawn_with_gate(command, cwd=cwd, child_env=child_env,
                                 custody=custody, startup_fd=fd)
                return custody._child
            finally:
                self._write_fd = None
                os.close(fd)

    def receive(self, *, timeout):
        """Receive one bytes frame, or fail the owner startup on deadline."""
        _finite_timeout(timeout, "timeout")
        with self._receive_lock:
            if self._receive_claimed:
                raise StartupChannelError("startup channel can only be received once")
            self._receive_claimed = True
        if not self._ready.wait(timeout):
            # Never wait on the synchronous reader or the spawn lock here.
            # This short delivery lock contains no I/O, so a frame published
            # concurrently with the deadline cannot retain a stale secret.
            self._mark_failed()
            with self._delivery_lock:
                self._payload = None
                self._discard_partial()
            raise StartupChannelError("startup channel timed out")
        with self._delivery_lock:
            if self._failed.is_set() or not self._frame_valid:
                self._payload = None
                self._discard_partial()
                raise StartupChannelError("startup channel failed")
            payload, self._payload = self._payload, None
            return payload

    def abort(self):
        """Reject startup, including after a caller rejects the received data."""
        self._mark_failed()
        with self._delivery_lock:
            self._payload = None
            self._discard_partial()
        self._ready.set()
        # A concurrent Popen may not return promptly. In that case the spawn
        # owner closes its endpoint in _spawn's finally; abort must not wait.
        if self._lock.acquire(blocking=False):
            try:
                if self._write_fd is not None:
                    fd = self._write_fd
                    self._write_fd = None
                    os.close(fd)
            finally:
                self._lock.release()

    def close(self):
        """Close an unused channel; explicitly abort rejected startup data."""
        with self._delivery_lock:
            should_abort = (not self._frame_valid or not self._receive_claimed
                            or self._payload is not None)
        if should_abort:
            self.abort()

    def wait_closed(self, *, timeout):
        """Return whether the reader exited and closed its pipe endpoint."""
        _finite_timeout(timeout, "timeout")
        return self._reader_done.wait(timeout) and self._reader_closed


@dataclass(frozen=True)
class SupervisionResult:
    run_id: str
    pid: int
    exit_code: int | None
    clean: bool
    timed_out: bool
    reason: str


def _absolute_path(value, name):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path.resolve()


def _finite_timeout(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number")


def _attestation_path(marker_path):
    return marker_path.with_name(marker_path.name + ".attestation")


def _control_path(marker_path):
    return marker_path.with_name(marker_path.name + ".controller.lock")


def _paths(device_lease_path, marker_path):
    device = _absolute_path(device_lease_path, "device_lease_path")
    marker = _absolute_path(marker_path, "marker_path")
    if device.suffix.lower() != ".lock":
        raise ValueError("device lease path must have a .lock suffix")
    expected_marker = device.with_suffix(".unsafe.json")
    if os.path.normcase(str(marker)) != os.path.normcase(str(expected_marker)):
        raise ValueError("marker path must be derived from the device lease path")
    control = _control_path(marker)
    attestation = _attestation_path(marker)
    normalized = {os.path.normcase(str(p)) for p in (device, marker, control, attestation)}
    if len(normalized) != 4:
        raise ValueError("device, marker, attestation, and controller paths must differ")
    if not marker.parent.is_dir() or not device.parent.is_dir():
        raise ValueError("configured parent directories must already exist")
    return device, marker, control, attestation


def _check_lease_path(lease, expected):
    if type(lease) is not DeviceLease:
        raise SupervisorError("a live DeviceLease is required")
    DeviceLease.check(lease)
    if os.path.normcase(str(lease.path)) != os.path.normcase(str(expected)):
        raise SupervisorError("lease does not match the configured device path")
    # A hard-link alias could name the same OS lock while deriving a different
    # marker namespace. Refuse that situation rather than silently splitting it.
    if lease.path.stat().st_nlink != 1:
        raise SupervisorError("device lease file has an ambiguous hard-link alias")


def _write_exclusive(path, data):
    raw = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _read_json(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, ValueError, UnicodeError) as exc:
        raise UnsafeOwnerError(f"owner state is missing or invalid: {path.name}") from exc
    if not isinstance(result, dict):
        raise UnsafeOwnerError(f"owner state is invalid: {path.name}")
    return result


def _marker(run_id, device, identity_spec=None):
    return {"version": _VERSION, "run_id": run_id, "device_lease_path": str(device),
            "device_identity": identity_spec}


def _check_identity(identity, device, marker):
    if type(identity) is not DeviceIdentity:
        raise SupervisorError("an exact DeviceIdentity is required for GPU supervision")
    DeviceIdentity.validate_paths(identity)
    if (os.path.normcase(str(identity.lock_path)) != os.path.normcase(str(device))
            or os.path.normcase(str(identity.marker_path)) != os.path.normcase(str(marker))):
        raise SupervisorError("device identity does not match the supervisor paths")


def _identity_spec(identity):
    return {
        "workspace_root": str(identity.workspace_root),
        "gpu_uuid": identity.gpu_uuid,
        "physical_uuids": list(identity._physical_uuids),
        "cuda_device_index": identity.cuda_device_index,
        "visibility": list(identity._environment_snapshot),
        "root_id": list(identity._root_file_id),
        "state_id": list(identity._state_file_id),
    }


def _resolve_child_identity(spec, device, marker):
    if not isinstance(spec, dict) or set(spec) != {
        "workspace_root", "gpu_uuid", "physical_uuids", "cuda_device_index",
        "visibility", "root_id", "state_id",
    }:
        raise SupervisorError("invalid owner device identity")
    try:
        identity = resolve_device_identity(
            spec["workspace_root"], spec["gpu_uuid"], physical_uuids=spec["physical_uuids"],
            cuda_device_index=spec["cuda_device_index"],
        )
    except (DeviceIdentityError, TypeError, KeyError) as exc:
        raise SupervisorError("cannot revalidate owner device identity") from exc
    _check_identity(identity, device, marker)
    if (_identity_spec(identity) != spec):
        raise SupervisorError("owner identity or environment changed since preflight")
    return identity


def _windows_kernel():
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _windows_birth(kernel, handle):
    import ctypes
    from ctypes import wintypes
    created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        raise SupervisorError("cannot verify supervisor process creation time")
    return str((created.dwHighDateTime << 32) | created.dwLowDateTime)


def _supervisor_birth():
    if os.name == "nt":
        kernel = _windows_kernel()
        return _windows_birth(kernel, kernel.GetCurrentProcess())
    if os.name == "posix":
        return "posix-parent"
    raise SupervisorError("supervisor liveness checking is unsupported on this platform")


def _watch_parent(pid, birth):
    """Start one process-lifetime watchdog before the child can use CUDA."""
    key = (pid, birth)
    with _watchdog_lock:
        guard = _watchdogs.get(key)
        if guard is not None:
            if os.name == "nt" and guard[0].WaitForSingleObject(guard[1], 0) != 0x102:
                raise UnsafeOwnerError("supervisor process is no longer alive")
            return
        if os.name == "nt":
            kernel = _windows_kernel()
            # Wait on a process handle, not a PID that may later be reused.
            handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
            if not handle:
                raise UnsafeOwnerError("cannot open the supervising process")
            try:
                if _windows_birth(kernel, handle) != birth:
                    raise UnsafeOwnerError("supervisor PID was reused")
                if kernel.WaitForSingleObject(handle, 0) != 0x102:
                    raise UnsafeOwnerError("supervisor process is no longer alive")

                def watch():
                    # Even a wait failure means liveness cannot be established.
                    kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
                    os._exit(70)

                thread = threading.Thread(target=watch, name="looking-glass-parent-watchdog", daemon=True)
                thread.start()
                _watchdogs[key] = (kernel, handle, thread)
            except BaseException:
                kernel.CloseHandle(handle)
                raise
        elif os.name == "posix" and birth == "posix-parent":
            def watch():
                while os.getppid() == pid:
                    time.sleep(.1)
                os._exit(70)
            thread = threading.Thread(target=watch, name="looking-glass-parent-watchdog", daemon=True)
            thread.start()
            _watchdogs[key] = (None, None, thread)
        else:
            raise SupervisorError("supervisor liveness checking is unsupported")


class OwnerSession:
    """The run identity provided by a supervising parent to its child."""

    def __init__(self, run_id, device_lease_path, marker_path, supervisor_pid, supervisor_birth,
                 identity_spec=None, startup_fd=None, startup_max=None):
        if (type(run_id) is not str or len(run_id) != 32
                or any(char not in "0123456789abcdef" for char in run_id)):
            raise ValueError("run_id must be a lowercase UUID hex value")
        self.run_id = run_id
        if type(supervisor_pid) is not int or supervisor_pid <= 0:
            raise ValueError("supervisor_pid must be a positive integer")
        self.supervisor_pid = supervisor_pid
        if type(supervisor_birth) is not str or not supervisor_birth:
            raise ValueError("supervisor_birth must identify the supervising process")
        self.supervisor_birth = supervisor_birth
        self.device_lease_path, self.marker_path, _, self.attestation_path = _paths(
            device_lease_path, marker_path)
        self._identity_spec = identity_spec
        self._identity = None
        self._runtime_verified = False
        self._bound_lease = None
        self._startup_fd = startup_fd
        self._startup_max = startup_max
        self._startup_send_lock = threading.Lock()

    @property
    def identity(self):
        """Return the revalidated GPU identity, or None for CPU-only testing."""
        return self._identity

    @classmethod
    def from_environment(cls):
        fd = None
        try:
            parent_pid = int(os.environ[_SUPERVISOR_ENV])
            parent_birth = os.environ[_SUPERVISOR_BIRTH_ENV]
            if os.getppid() != parent_pid:
                raise UnsafeOwnerError("owner must be the supervisor's direct child")
            # This watchdog also covers a parent that dies before granting the
            # launch. A live parent thread abandoning startup is handled by
            # the private launch gate and the parent's rooted child record.
            _watch_parent(parent_pid, parent_birth)
            _await_launch_grant()
            encoded_identity = os.environ.get(_IDENTITY_ENV)
            if encoded_identity is not None and len(encoded_identity) > 32768:
                raise ValueError("device identity is too large")
            spec = json.loads(encoded_identity) if encoded_identity is not None else None
            endpoint = os.environ.pop(_STARTUP_HANDLE_ENV, None)
            limit = os.environ.pop(_STARTUP_LIMIT_ENV, None)
            if (endpoint is None) != (limit is None):
                raise ValueError("incomplete startup channel")
            startup_max = None
            if endpoint is not None:
                endpoint = int(endpoint)
                startup_max = int(limit)
                if endpoint < 0 or not 1 <= startup_max <= _MAX_STARTUP_BYTES:
                    raise ValueError("invalid startup channel")
                if os.name == "nt":
                    import msvcrt
                    fd = msvcrt.open_osfhandle(endpoint, os.O_WRONLY | os.O_BINARY)
                elif os.name == "posix":
                    fd = endpoint
                else:
                    raise ValueError("unsupported startup channel")
                os.set_inheritable(fd, False)
            session = cls(os.environ[_RUN_ENV], os.environ[_LEASE_ENV], os.environ[_MARKER_ENV],
                          parent_pid, parent_birth,
                          spec, fd, startup_max)
            fd = None  # The session owns the endpoint.
            return session
        except (KeyError, ValueError, OSError) as exc:
            raise SupervisorError("supervised owner environment is incomplete") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def check(self, lease):
        """Check the run marker while this child holds the actual device lease."""
        if os.getppid() != self.supervisor_pid:
            raise UnsafeOwnerError("owner must be the supervisor's direct child")
        _watch_parent(self.supervisor_pid, self.supervisor_birth)
        _check_lease_path(lease, self.device_lease_path)
        if self._bound_lease is not None and lease is not self._bound_lease:
            raise SupervisorError("owner session cannot switch device leases")
        if self._identity_spec is not None:
            self._identity = _resolve_child_identity(
                self._identity_spec, self.device_lease_path, self.marker_path)
        if _read_json(self.marker_path) != _marker(
                self.run_id, self.device_lease_path, self._identity_spec):
            raise UnsafeOwnerError("owner marker does not match this run")
        if os.path.lexists(self.attestation_path):
            raise UnsafeOwnerError("an attestation already exists for this marker")
        self._bound_lease = lease

    def verify_runtime_binding(self, lease, *, runtime_uuid, device_count, cuda_device_index=0):
        """Verify the runtime-reported CUDA UUID against the supervised identity."""
        OwnerSession.check(self, lease)
        if self._identity is None:
            raise SupervisorError("GPU runtime verification requires a supervised identity")
        DeviceIdentity.verify_runtime_binding(
            self._identity, lease=lease, runtime_uuid=runtime_uuid,
            device_count=device_count, cuda_device_index=cuda_device_index,
        )
        self._runtime_verified = True

    def attest_clean(self, lease):
        """Attest after verified shutdown, while retaining the lease until exit."""
        OwnerSession.check(self, lease)
        if self._identity_spec is not None and not self._runtime_verified:
            raise SupervisorError("the actual runtime device identity was not verified")
        _write_exclusive(self.attestation_path, {
            "version": _VERSION,
            "run_id": self.run_id,
            "owner_pid": os.getpid(),
            "clean": True,
        })

    def send_startup(self, lease, payload):
        """Send a single bounded bytes frame to the supervising parent."""
        fd = None
        chunk = None
        view = None
        failed = False
        interrupted = None
        try:
            OwnerSession.check(self, lease)
            if self._identity_spec is not None and not self._runtime_verified:
                raise SupervisorError("the actual runtime device identity was not verified")
            if type(payload) is not bytes or not 1 <= len(payload) <= (self._startup_max or 0):
                raise StartupChannelError("invalid startup payload")
            with self._startup_send_lock:
                fd, self._startup_fd = self._startup_fd, None
                if fd is None:
                    raise StartupChannelError("startup channel is unavailable")
            for chunk in (struct.pack("!I", len(payload)), payload):
                view = memoryview(chunk)
                while view:
                    count = os.write(fd, view)
                    if count <= 0:
                        raise OSError("startup write did not progress")
                    view = view[count:]
        except KeyboardInterrupt:
            failed = True
            interrupted = KeyboardInterrupt
        except SystemExit:
            failed = True
            interrupted = SystemExit
        except Exception:
            failed = True
        finally:
            # Avoid retaining the payload in this method's failure traceback.
            # Callers' frames and Python/OS copies remain outside this guarantee.
            payload = chunk = view = None
            if failed and fd is None:
                with self._startup_send_lock:
                    fd, self._startup_fd = self._startup_fd, None
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    failed = True
        if failed:
            # Raise only after leaving the exception handler: neither context
            # nor cause should retain a lower-level traceback with payloads.
            if interrupted is KeyboardInterrupt:
                raise KeyboardInterrupt()
            if interrupted is SystemExit:
                raise SystemExit(1)
            raise StartupChannelError("startup channel could not be sent")


def _stop_child(child, grace):
    """Terminate only the exact child represented by this process handle."""
    if child.poll() is None:
        child.terminate()
    try:
        return child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            return child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            return None


def _pipe_endpoint(fd):
    if os.name == "nt":
        import msvcrt
        return msvcrt.get_osfhandle(fd)
    if os.name == "posix":
        return fd
    raise SupervisorError("launch gates are unsupported on this platform")


def _await_launch_grant():
    """Block a cooperative owner before lease acquisition until authorized."""
    endpoint = int(os.environ.pop(_LAUNCH_HANDLE_ENV))
    if endpoint < 0:
        raise ValueError("invalid launch gate")
    if os.name == "nt":
        import msvcrt
        fd = msvcrt.open_osfhandle(endpoint, os.O_RDONLY | os.O_BINARY)
    elif os.name == "posix":
        fd = endpoint
    else:
        raise SupervisorError("launch gates are unsupported on this platform")
    done = threading.Event()

    def expire():
        if not done.wait(_LAUNCH_TIMEOUT):
            # A leaked parent writer must not leave an ungranted owner alive
            # indefinitely. This child has not been permitted to use the device.
            os._exit(70)

    try:
        os.set_inheritable(fd, False)
        threading.Thread(target=expire, name="looking-glass-launch-deadline",
                         daemon=True).start()
        if os.read(fd, 1) != b"\x01":
            raise UnsafeOwnerError("owner launch was not granted")
    finally:
        done.set()
        os.close(fd)


class _SpawnCustody:
    """Root a direct child before grant; recover only after its launcher stops."""

    def __init__(self, run_id, process_job=None):
        self.run_id = run_id
        self._child = None
        self._attempted = False
        self._read_fd = None
        self._write_fd = None
        self._process_job = process_job
        self._job_drain_uncertain = False
        self._job_assigned = False

    def _register(self):
        with _spawn_custody_lock:
            if len(_spawn_custodies) >= _MAX_SPAWN_CUSTODIES:
                raise SupervisorError("supervised child recovery capacity is exhausted")
            if self.run_id in _spawn_custodies:
                raise SupervisorError("duplicate supervised child recovery identity")
            if (self._process_job is not None
                    and any(record._process_job is self._process_job for record in _spawn_custodies.values())):
                raise SupervisorError("the process job already belongs to another supervised run")
            _spawn_custodies[self.run_id] = self

    def _open_gate(self):
        self._read_fd, self._write_fd = os.pipe()
        os.set_inheritable(self._read_fd, False)
        os.set_inheritable(self._write_fd, False)

    def _start(self, command, **options):
        self._attempted = True
        factory = subprocess.Popen
        if isinstance(factory, type):
            # Store the object before calling its constructor: an interruption
            # in or at the return of __init__ must not lose an already stored
            # process handle. A native spawn can still be interrupted before
            # Popen installs that handle; no grant is sent in that case.
            self._child = factory.__new__(factory)
            factory.__init__(self._child, command, **options)
        else:
            # Trusted test/instrumentation wrappers may replace Popen. Their
            # returned object is still recorded before any grant is sent.
            self._child = factory(command, **options)

    def _known_child(self):
        child = self._child
        if child is None or type(getattr(child, "pid", None)) is not int or child.pid <= 0:
            return False
        if os.name == "nt":
            return getattr(child, "_handle", None) is not None
        return bool(getattr(child, "_child_created", False))

    def _close_gate(self):
        # Forget each fd before close: if close is interrupted, never reuse a
        # stale number that might later refer to an unrelated descriptor.
        write_fd, self._write_fd = self._write_fd, None
        try:
            if write_fd is not None:
                os.close(write_fd)
        finally:
            read_fd, self._read_fd = self._read_fd, None
            if read_fd is not None:
                os.close(read_fd)

    def _grant(self):
        if not self._known_child() or self._write_fd is None:
            raise SupervisorError("launch has no rooted direct child capability")
        if os.write(self._write_fd, b"\x01") != 1:
            raise SupervisorError("owner launch grant did not complete")

    def _retire(self):
        with _spawn_custody_lock:
            if _spawn_custodies.get(self.run_id) is self:
                del _spawn_custodies[self.run_id]

    def _finish_job(self, terminate_grace, *, allow_recovery=False):
        """Confirm process-tree termination before releasing the job capability."""
        job = self._process_job
        if job is not None:
            if self._job_drain_uncertain and not allow_recovery:
                raise UnsafeOwnerError("process-job drain is uncertain; retain custody")
            self._job_drain_uncertain = True
            was_quiescent = False
            if not allow_recovery:
                try:
                    counts = job.process_counts()
                    if (type(counts) is not tuple or len(counts) != 2
                            or any(type(value) is not int or value < 0 for value in counts)
                            or counts[1] > counts[0] or (self._job_assigned and counts[0] < 1)):
                        raise SupervisorError("process-job counts are invalid")
                    active = counts[1]
                    was_quiescent = active == 0
                except BaseException as query_error:
                    # Even if the count is unknown, attempt to stop members;
                    # keep the job rooted and never turn this into a clean run.
                    try:
                        job.terminate_and_wait(timeout=terminate_grace)
                    except BaseException as cleanup_error:
                        if (isinstance(cleanup_error, (KeyboardInterrupt, SystemExit))
                                and not isinstance(query_error, (KeyboardInterrupt, SystemExit))):
                            raise cleanup_error from query_error
                    raise
            job.terminate_and_wait(timeout=terminate_grace)
            job.close()
            self._process_job = None
            self._job_drain_uncertain = False
            return was_quiescent
        return True

    def recover(self, *, terminate_grace=5):
        """Stop/reap this child and its job after the launcher is quiescent.

        Unknown native-spawn outcomes cannot be proved reaped with this API.
        Recovery never removes the sticky device marker.
        """
        _finite_timeout(terminate_grace, "terminate_grace")
        with _spawn_custody_lock:
            if _spawn_custodies.get(self.run_id) is not self:
                raise UnsafeOwnerError("supervised child record is stale")
        self._close_gate()
        if not self._attempted:
            # The launcher is quiescent and never reached Popen: there is no
            # direct child to stop or reap. Quarantine is still untouched.
            self._finish_job(terminate_grace, allow_recovery=True)
            self._retire()
            return None
        if not self._known_child():
            raise UnsafeOwnerError("child creation outcome is unknown; retain quarantine")
        exit_code = _stop_child(self._child, terminate_grace)
        if exit_code is None:
            raise UnsafeOwnerError("direct child exit could not be confirmed")
        self._finish_job(terminate_grace, allow_recovery=True)
        self._retire()
        return exit_code


def pending_supervised_children():
    """Return rooted recovery records; caller must establish launcher quiescence."""
    with _spawn_custody_lock:
        return tuple(_spawn_custodies.values())


def _spawn_with_gate(command, *, cwd, child_env, custody, startup_fd=None):
    """Spawn under _startup_spawn_lock with explicit inherited endpoints."""
    read_fd = custody._read_fd
    if read_fd is None:
        raise SupervisorError("owner launch gate is unavailable")
    child_env[_LAUNCH_HANDLE_ENV] = str(_pipe_endpoint(read_fd))
    fds = (read_fd,) if startup_fd is None else (read_fd, startup_fd)
    if os.name == "nt":
        for fd in fds:
            os.set_inheritable(fd, True)
        info = subprocess.STARTUPINFO()
        info.lpAttributeList = {"handle_list": [_pipe_endpoint(fd) for fd in fds]}
        options = {"startupinfo": info, "close_fds": True}
    elif os.name == "posix":
        options = {"pass_fds": fds, "close_fds": True}
    else:
        raise SupervisorError("launch gates are unsupported on this platform")
    custody._start(command, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL, **options)


def run_supervised(command, *, device_lease_path, marker_path, timeout,
                   lease_timeout=5, terminate_grace=5, cwd=None, env=None, identity=None,
                   startup_channel=None, inherit_environment=True, process_job=None,
                   require_zero_exit=False):
    """Run one trusted owner and clear the sticky marker only after clean exit.

    On Windows an optional trusted, single-run process-job capability provides
    ``assign``, ``process_counts``, ``terminate_and_wait`` and ``close``. The
    caller must supply an independently reviewed job implementation and never share it with other
    work. This function assigns the rooted child before granting startup, and
    keeps the job in recovery custody until its drain and close are confirmed.
    Process-tree guarantees depend on the supplied capability and do not cover
    processes that escape or are outside its job.

    With ``require_zero_exit=True``, a nonzero or unknown owner exit preserves
    the marker even when the owner wrote a clean attestation. This is useful
    when a later owner failure can invalidate that attestation; it does not
    itself establish that all fault-producing threads were quiescent.
    """
    if type(require_zero_exit) is not bool:
        raise ValueError("require_zero_exit must be a boolean")
    if startup_channel is not None:
        if type(startup_channel) is not StartupChannel:
            raise ValueError("startup_channel must be a StartupChannel")
        startup_channel._claim()
    try:
        return _run_supervised(
            command, device_lease_path=device_lease_path, marker_path=marker_path,
            timeout=timeout, lease_timeout=lease_timeout, terminate_grace=terminate_grace,
            cwd=cwd, env=env, identity=identity, startup_channel=startup_channel,
            inherit_environment=inherit_environment, process_job=process_job,
            require_zero_exit=require_zero_exit,
        )
    except BaseException as primary_error:
        if startup_channel is not None:
            try:
                startup_channel.abort()
            except BaseException as abort_error:
                if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                    pass
                elif isinstance(abort_error, (KeyboardInterrupt, SystemExit)):
                    raise abort_error from primary_error
                else:
                    raise
        raise


def _run_supervised(command, *, device_lease_path, marker_path, timeout,
                    lease_timeout, terminate_grace, cwd, env, identity, startup_channel,
                    inherit_environment, process_job, require_zero_exit):
    _finite_timeout(timeout, "timeout")
    _finite_timeout(lease_timeout, "lease_timeout")
    _finite_timeout(terminate_grace, "terminate_grace")
    if process_job is not None:
        if os.name != "nt" or not 0 < terminate_grace <= 10:
            raise ValueError("a Windows process job requires a drain timeout in (0, 10]")
        if any(not callable(getattr(process_job, name, None))
               for name in ("assign", "process_counts", "terminate_and_wait", "close")):
            raise ValueError("an assignable, drainable process-job capability is required")
        counts = process_job.process_counts()
        if (type(counts) is not tuple or len(counts) != 2
                or any(type(value) is not int for value in counts) or counts != (0, 0)):
            raise ValueError("the process job must be fresh and empty before launch")
    if type(inherit_environment) is not bool:
        raise ValueError("inherit_environment must be a boolean")
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a nonempty argument sequence")
    device, marker, control, attestation = _paths(device_lease_path, marker_path)
    if identity is not None:
        _check_identity(identity, device, marker)
    control_lease = DeviceLease.acquire(control, timeout=lease_timeout)
    marker_created = False
    marker_data = None
    keep_marker = True
    try:
        _check_lease_path(control_lease, control)
        device_lease = DeviceLease.acquire(device, timeout=lease_timeout)
        try:
            _check_lease_path(device_lease, device)
            if identity is not None:
                _check_identity(identity, device, marker)
            if os.path.lexists(marker) or os.path.lexists(attestation):
                raise UnsafeOwnerError("a previous owner marker or attestation remains")
            run_id = uuid.uuid4().hex
            marker_data = _marker(run_id, device, _identity_spec(identity) if identity is not None else None)
            _write_exclusive(marker, marker_data)
            marker_created = True
        finally:
            device_lease.release()

        child_env = os.environ.copy() if inherit_environment else {}
        if env is not None:
            child_env.update(env)
        # Windows environment keys are case-insensitive. Remove every spelling
        # of our reserved fields before assigning this run's trusted values.
        for key in list(child_env):
            if isinstance(key, str) and key.upper() in _SUPERVISOR_ENV_KEYS:
                del child_env[key]
        if identity is not None:
            child_env[_IDENTITY_ENV] = json.dumps(_identity_spec(identity), separators=(",", ":"))
        else:
            # The identity-less mode exists only for CPU subprocess tests.
            for key in list(child_env):
                if isinstance(key, str) and key.upper() == "CUDA_VISIBLE_DEVICES":
                    del child_env[key]
            child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env.update({_RUN_ENV: run_id, _LEASE_ENV: str(device), _MARKER_ENV: str(marker),
                          _SUPERVISOR_ENV: str(os.getpid()), _SUPERVISOR_BIRTH_ENV: _supervisor_birth()})
        custody = _SpawnCustody(run_id, process_job)
        timed_out = False
        startup_failed = False
        job_quiescent = True
        try:
            custody._register()
            custody._open_gate()
            if startup_channel is None:
                with _startup_spawn_lock:
                    _spawn_with_gate(command, cwd=cwd, child_env=child_env,
                                     custody=custody)
            else:
                startup_channel._spawn(command, cwd=cwd, child_env=child_env,
                                       custody=custody)
            child = custody._child
            if process_job is not None:
                process_job.assign(child)
                custody._job_assigned = True
            if startup_channel is not None:
                startup_channel._authorize_launch()
            custody._grant()
            custody._close_gate()
            if startup_channel is None:
                try:
                    exit_code = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    exit_code = _stop_child(child, terminate_grace)
            else:
                deadline = time.monotonic() + timeout
                while True:
                    if startup_channel._failed.is_set():
                        startup_failed = True
                        exit_code = _stop_child(child, terminate_grace)
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        exit_code = _stop_child(child, terminate_grace)
                        break
                    try:
                        exit_code = child.wait(timeout=min(remaining, .05))
                        break
                    except subprocess.TimeoutExpired:
                        pass
            if exit_code is not None:
                job_quiescent = custody._finish_job(terminate_grace)
                custody._retire()
        except BaseException as primary_error:
            # Retain the original failure unless cleanup receives a first
            # fatal interruption. In either case the exact record stays rooted
            # when cleanup cannot prove that the child and job are gone.
            cleanup_interrupt = None
            try:
                custody._close_gate()
            except (KeyboardInterrupt, SystemExit) as error:
                cleanup_interrupt = error
            except BaseException:
                pass
            try:
                with _spawn_custody_lock:
                    registered = _spawn_custodies.get(custody.run_id) is custody
                if registered and custody._known_child():
                    if _stop_child(custody._child, terminate_grace) is not None:
                        custody._finish_job(terminate_grace)
                        custody._retire()
                elif registered and not custody._attempted:
                    custody._finish_job(terminate_grace)
                    custody._retire()
            except (KeyboardInterrupt, SystemExit) as error:
                if cleanup_interrupt is None:
                    cleanup_interrupt = error
            except BaseException:
                pass
            if (cleanup_interrupt is not None
                    and not isinstance(primary_error, (KeyboardInterrupt, SystemExit))):
                raise cleanup_interrupt from primary_error
            raise

        if timed_out:
            if startup_channel is not None:
                startup_channel.abort()
            return SupervisionResult(run_id, child.pid, exit_code, False, timed_out,
                                     "timeout")
        if startup_channel is not None:
            # A dead owner must have sent a complete frame. If another process
            # retained the pipe, do not wait indefinitely or clear quarantine.
            startup_channel._ready.wait(terminate_grace)
            if (startup_failed or startup_channel._failed.is_set()
                    or not startup_channel._frame_valid):
                startup_channel.abort()
                return SupervisionResult(run_id, child.pid, exit_code, False, False,
                                         "startup channel failed")

        if not job_quiescent:
            return SupervisionResult(run_id, child.pid, exit_code, False, False,
                                     "process job had live members after owner exit")

        if require_zero_exit and (type(exit_code) is not int or exit_code != 0):
            return SupervisionResult(run_id, child.pid, exit_code, False, False,
                                     "owner exit was not zero")

        cleanup_lease = DeviceLease.acquire(device, timeout=lease_timeout)
        clean = False
        reason = "missing or invalid owner state"
        try:
            _check_lease_path(cleanup_lease, device)
            if identity is not None:
                _check_identity(identity, device, marker)
            expected = {"version": _VERSION, "run_id": run_id,
                        "owner_pid": child.pid, "clean": True}
            if _read_json(marker) != marker_data or _read_json(attestation) != expected:
                reason = "owner marker or attestation mismatch"
            else:
                # Keep the marker if deletion is interrupted or the attestation
                # cannot be removed. Every participant rejects a leftover marker.
                attestation.unlink()
                marker.unlink()
                clean = True
                reason = "clean device teardown" if exit_code == 0 else "clean device teardown after workload failure"
        except UnsafeOwnerError:
            pass
        finally:
            cleanup_lease.release()
        if clean:
            keep_marker = False
        return SupervisionResult(run_id, child.pid, exit_code, clean, False, reason)
    finally:
        # A trusted owner must never remove or replace the marker. If it was
        # lost, restore it before allowing another cooperative controller in.
        # On a filesystem error keep the controller lease held in this process.
        if marker_created and keep_marker and not os.path.lexists(marker):
            try:
                _write_exclusive(marker, marker_data)
            except FileExistsError:
                pass
        control_lease.release()
