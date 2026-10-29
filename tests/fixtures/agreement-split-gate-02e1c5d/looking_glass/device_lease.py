"""Cooperative, process-scoped exclusion for a shared device.

This lease is independent of CUDA.  Every participating GPU owner must use
the same persistent file and retain the lease for its entire device lifetime.
It cannot control software that does not participate in this protocol.
"""

from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import stat
import threading
import time


class LeaseError(RuntimeError):
    """A lease could not be safely acquired or released."""


class LeaseTimeoutError(LeaseError, TimeoutError):
    """The lease was held until the acquisition deadline."""


class LeaseCancelledError(LeaseError):
    """Acquisition was cancelled before ownership was returned."""


class LeaseOwnershipError(LeaseError):
    """A different process attempted to operate on an inherited lease."""


# Track even acquisitions that are waiting. A forked child closes its copies;
# it must not keep the parent's POSIX open-file-description lock alive.
_state_lock = threading.Lock()
_open_fds: set[int] = set()
_active: dict[int, "DeviceLease"] = {}
_WAIT_SLICE = 0.05


def _after_fork_child() -> None:
    try:
        for fd in tuple(_open_fds):
            try:
                os.close(fd)
            except OSError:
                pass
        _open_fds.clear()
        _active.clear()
    finally:
        _state_lock.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_state_lock.acquire,
        after_in_parent=_state_lock.release,
        after_in_child=_after_fork_child,
    )


def _try_lock(fd: int) -> bool:
    """Try the OS lock once; only documented contention means busy."""
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        elif os.name == "posix":
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            raise LeaseError(f"unsupported lock platform: {os.name}")
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise LeaseError("device lock acquisition failed") from exc
    return True


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    elif os.name == "posix":
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    else:
        raise LeaseError(f"unsupported lock platform: {os.name}")


def _check_identity(fd: int, path: Path) -> None:
    """Reject a missing, replaced, non-regular, or unidentifiable file."""
    try:
        opened, named = os.fstat(fd), os.stat(path)
    except OSError as exc:
        raise LeaseError("device lock file cannot be identified") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or not opened.st_ino
        or not named.st_ino
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise LeaseError("device lock file was replaced or has ambiguous identity")


def _cancelled(event: object | None) -> bool:
    return event is not None and bool(event.is_set())  # type: ignore[attr-defined]


def _check_wait(deadline: float, event: object | None, instant: bool) -> None:
    if _cancelled(event):
        raise LeaseCancelledError("device lease acquisition cancelled")
    if not instant and time.monotonic() >= deadline:
        raise LeaseTimeoutError("device lease acquisition timed out")


def _take_state_lock(deadline: float, event: object | None, instant: bool) -> None:
    """Acquire our bookkeeping mutex without defeating the caller's deadline."""
    while True:
        _check_wait(deadline, event, instant)
        remaining = max(0.0, deadline - time.monotonic())
        if _state_lock.acquire(timeout=min(remaining, _WAIT_SLICE)):
            try:
                _check_wait(deadline, event, instant)
            except BaseException:
                _state_lock.release()
                raise
            return
        if instant:
            raise LeaseTimeoutError("device lease acquisition timed out")


class DeviceLease:
    """An exclusive cooperative device lease held by this process.

    Use :meth:`acquire` and call :meth:`release` only once device teardown is
    known to be safe.  No context manager auto-releases on exceptional paths.
    An abandoned lease remains held until explicit release or process exit.
    """

    def __init__(self, fd: int, path: Path) -> None:
        self._fd = fd
        self.path = path
        self._owner_pid = os.getpid()
        self._released = False
        self._invalid = False

    @classmethod
    def acquire(
        cls,
        path: str | os.PathLike[str],
        *,
        timeout: float,
        cancel_event: object | None = None,
        poll_interval: float = 0.05,
    ) -> "DeviceLease":
        """Acquire from an absolute, persistent lock path within a deadline.

        A zero timeout makes one nonblocking attempt. Cancellation is checked
        before opening, between attempts, and immediately after acquisition.
        The containing directory must already exist and be locally managed.
        """
        for name, value, allow_zero in (
            ("timeout", timeout, True),
            ("poll_interval", poll_interval, False),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
        path = Path(path)
        if not path.is_absolute():
            raise ValueError("device lock path must be absolute")
        if cancel_event is not None and (
            not callable(getattr(cancel_event, "is_set", None))
            or not callable(getattr(cancel_event, "wait", None))
        ):
            raise ValueError("cancel_event must provide is_set() and wait()")
        if _cancelled(cancel_event):
            raise LeaseCancelledError("device lease acquisition cancelled")
        deadline = time.monotonic() + timeout
        instant = timeout == 0
        try:
            path = path.resolve()
        except OSError as exc:
            raise LeaseError("device lock path cannot be resolved") from exc
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        _take_state_lock(deadline, cancel_event, instant)
        try:
            try:
                fd = os.open(path, flags, 0o600)
                os.set_inheritable(fd, False)
            except OSError as exc:
                if "fd" in locals():
                    os.close(fd)
                raise LeaseError("device lock file cannot be opened") from exc
            _open_fds.add(fd)
        finally:
            _state_lock.release()

        acquired = False
        returned = False
        try:
            first_attempt = True
            while True:
                _check_wait(deadline, cancel_event, instant and first_attempt)
                first_attempt = False
                _take_state_lock(deadline, cancel_event, instant)
                try:
                    acquired = _try_lock(fd)
                    if acquired:
                        _check_identity(fd, path)
                        _check_wait(deadline, cancel_event, instant)
                        lease = cls(fd, path)
                        _active[fd] = lease
                        returned = True
                        return lease
                finally:
                    _state_lock.release()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LeaseTimeoutError("device lease acquisition timed out")
                delay = min(poll_interval, remaining, _WAIT_SLICE)
                if cancel_event is None:
                    time.sleep(delay)
                else:
                    cancel_event.wait(delay)  # type: ignore[attr-defined]
        finally:
            if not returned:
                if not _state_lock.acquire(timeout=_WAIT_SLICE):
                    raise LeaseError("device lease cleanup blocked; terminate the process")
                try:
                    try:
                        if acquired:
                            _unlock(fd)
                    finally:
                        os.close(fd)
                        _open_fds.discard(fd)
                finally:
                    _state_lock.release()

    def check(self) -> None:
        """Check process ownership and that the named lock file is unchanged."""
        if os.getpid() != self._owner_pid:
            raise LeaseOwnershipError("a lease cannot be used by an inherited process")
        with _state_lock:
            if self._released or self._invalid or self._fd not in _active:
                raise LeaseOwnershipError("the device lease is not held")
            try:
                _check_identity(self._fd, self.path)
            except LeaseError:
                self._invalid = True
                raise

    def release(self) -> None:
        """Release only after the caller has safely relinquished the device."""
        if os.getpid() != self._owner_pid:
            raise LeaseOwnershipError("a lease cannot be released by an inherited process")
        with _state_lock:
            if self._released:
                return
            if self._invalid:
                raise LeaseOwnershipError("device lease ownership is ambiguous; terminate the process")
            # Revoke the Python ownership claim before any operation which can
            # unlock or close the descriptor. An interruption after the OS
            # unlock must never leave check() reporting the old lease as held.
            self._invalid = True
            try:
                _check_identity(self._fd, self.path)
                _unlock(self._fd)
                os.close(self._fd)
            except (OSError, LeaseError) as exc:
                raise LeaseError("device lease release failed; ownership is ambiguous; terminate the process") from exc
            _open_fds.discard(self._fd)
            _active.pop(self._fd, None)
            self._released = True
            self._invalid = False
