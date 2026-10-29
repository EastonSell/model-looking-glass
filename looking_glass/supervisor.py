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
_VERSION = 1
_watchdog_lock = threading.Lock()
_watchdogs = {}


class SupervisorError(RuntimeError):
    """The owner lifecycle could not be safely coordinated."""


class UnsafeOwnerError(SupervisorError):
    """A prior or ambiguous owner prevents a new owner from starting."""


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
                 identity_spec=None):
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

    @property
    def identity(self):
        """Return the revalidated GPU identity, or None for CPU-only testing."""
        return self._identity

    @classmethod
    def from_environment(cls):
        try:
            encoded_identity = os.environ.get(_IDENTITY_ENV)
            if encoded_identity is not None and len(encoded_identity) > 32768:
                raise ValueError("device identity is too large")
            spec = json.loads(encoded_identity) if encoded_identity is not None else None
            return cls(os.environ[_RUN_ENV], os.environ[_LEASE_ENV], os.environ[_MARKER_ENV],
                       int(os.environ[_SUPERVISOR_ENV]), os.environ[_SUPERVISOR_BIRTH_ENV], spec)
        except (KeyError, ValueError) as exc:
            raise SupervisorError("supervised owner environment is incomplete") from exc

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


def run_supervised(command, *, device_lease_path, marker_path, timeout,
                   lease_timeout=5, terminate_grace=5, cwd=None, env=None, identity=None):
    """Run one trusted owner and clear the sticky marker only after clean exit."""
    _finite_timeout(timeout, "timeout")
    _finite_timeout(lease_timeout, "lease_timeout")
    _finite_timeout(terminate_grace, "terminate_grace")
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

        child_env = os.environ.copy()
        if env is not None:
            child_env.update(env)
        child_env.pop(_IDENTITY_ENV, None)
        if identity is not None:
            child_env[_IDENTITY_ENV] = json.dumps(_identity_spec(identity), separators=(",", ":"))
        else:
            # The identity-less mode exists only for CPU subprocess tests.
            child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env.update({_RUN_ENV: run_id, _LEASE_ENV: str(device), _MARKER_ENV: str(marker),
                          _SUPERVISOR_ENV: str(os.getpid()), _SUPERVISOR_BIRTH_ENV: _supervisor_birth()})
        child = subprocess.Popen(command, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL)
        timed_out = False
        try:
            try:
                exit_code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = _stop_child(child, terminate_grace)
        except BaseException:
            _stop_child(child, terminate_grace)
            raise

        if timed_out:
            return SupervisionResult(run_id, child.pid, exit_code, False, timed_out,
                                     "timeout")

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
