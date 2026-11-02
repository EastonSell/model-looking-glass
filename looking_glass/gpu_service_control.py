"""Private control-plane building blocks for a supervised GPU service.

This module does not start a GPU model. The fixed owner and measured profile
must be composed separately; the private generic launcher here is a trusted
internal seam, not an interface for accepting commands from clients.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
from threading import Event, Lock, Thread

from .supervisor import StartupChannel, run_supervised


READY_SCHEMA = "gpu-service-ready-v1"
MODEL_ID = "distilbert/distilgpt2@2290a62682d06624634c1f46a6ad5be0f47f38aa"
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z", re.ASCII)
_HEX32 = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CONTROLLERS = set()  # root a supervisor thread even if startup is interrupted
_CONTROLLERS_LOCK = Lock()


def _teams(teams):
    if (type(teams) is not tuple or not 1 <= len(teams) <= 128
            or any(type(team) is not str or _NAME.fullmatch(team) is None for team in teams)
            or len(set(teams)) != len(teams)):
        raise ValueError("team IDs must be distinct ASCII identifiers")
    return teams


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode_ready(payload, expected_teams):
    """Validate a private startup frame without including its bytes in errors."""
    expected_teams = _teams(expected_teams)
    value = address = tokens = None
    try:
        if type(payload) is not bytes or not 1 <= len(payload) <= 65536:
            raise ValueError("invalid frame length")
        value = json.loads(payload, object_pairs_hook=_unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if type(value) is not dict or set(value) != {
                "schema", "address", "broker_id", "team_tokens", "model_id", "device"}:
            raise ValueError("invalid frame fields")
        address, tokens = value["address"], value["team_tokens"]
        if (value["schema"] != READY_SCHEMA or value["model_id"] != MODEL_ID
                or value["device"] != "cuda:0"
                or type(address) is not list or len(address) != 2
                or address[0] != "127.0.0.1" or type(address[1]) is not int
                or not 1 <= address[1] <= 65535
                or type(value["broker_id"]) is not str
                or _HEX32.fullmatch(value["broker_id"]) is None
                or type(tokens) is not dict or set(tokens) != set(expected_teams)
                or any(type(token) is not str or _HEX64.fullmatch(token) is None
                       for token in tokens.values())
                or len(set(tokens.values())) != len(tokens)):
            raise ValueError("invalid ready metadata")
        return tuple(address), value["broker_id"], dict(tokens)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        pass
    # A JSONDecodeError can itself retain the frame. Raise outside the handler
    # and clear local references so rejected credentials are not in its context.
    payload = value = address = tokens = None
    raise ValueError("invalid private service readiness frame")


def _encode_ready(server, teams):
    """Build the one private readiness frame after the IPC server starts."""
    teams = _teams(teams)
    value = {"schema": READY_SCHEMA, "address": list(server.address),
             "broker_id": server.broker_id, "team_tokens": dict(server.team_tokens),
             "model_id": MODEL_ID, "device": "cuda:0"}
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _decode_ready(payload, teams)
    return payload


def _stop_path(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("stop path must be absolute")
    parent = path.parent.resolve(strict=True)
    info = parent.stat()
    if not stat.S_ISDIR(info.st_mode) or not info.st_ino:
        raise ValueError("stop directory cannot be identified")
    target = parent / path.name
    if os.path.lexists(target):
        raise ValueError("stop path must initially be absent")
    return target, (info.st_dev, info.st_ino)


def _same_parent(path, expected):
    info = path.parent.stat()
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != expected:
        raise RuntimeError("service stop directory changed")


def _monitor_stop(path, stop_event, *, on_failure=None):
    """Start a bounded-poll monitor for a nonsecret private stop marker.

    The preallocated failure flag starts set and is cleared only after the
    complete monitor loop exits normally. Uncaught thread exceptions restore
    it. Diagnostics are best effort; a flagged monitor prevents attestation.
    """
    if on_failure is not None and not callable(on_failure):
        raise TypeError('monitor failure callback must be callable')
    path, parent_id = _stop_path(path)
    errors = []
    failed = [True]

    def signal_failure():
        failed[0] = True
        try:
            if on_failure is not None:
                on_failure()
        finally:
            try:
                if on_failure is not None:
                    on_failure()
            finally:
                stop_event.set()

    def monitor():
        try:
            while not stop_event.wait(.05):
                _same_parent(path, parent_id)
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if (not stat.S_ISREG(info.st_mode)
                        or getattr(info, "st_file_attributes", 0) & 0x400):
                    raise RuntimeError("invalid service stop marker")
                stop_event.set()
            failed[0] = False
        except BaseException as exc:
            signal_failure()
            try:
                errors.append(type(exc).__name__)
            except BaseException:
                pass

    def uncaught_thread_failure(_thread):
        # CPython's Thread bootstrap invokes this per-instance hook after an
        # exception escapes the target, including one at its return boundary.
        # The child joins the exact thread before trusting the resulting flag.
        try:
            signal_failure()
        except BaseException:
            pass

    thread = Thread(target=monitor, name="looking-glass-service-stop", daemon=True)
    thread._invoke_excepthook = uncaught_thread_failure
    thread.start()
    return thread, failed, errors


class _ServiceController:
    """One local supervisor and private team credentials; never repr secrets."""

    def __init__(self, channel, stop_path, parent_id):
        self._channel, self._stop_path, self._parent_id = channel, stop_path, parent_id
        self._done = Event()
        self._lock = Lock()
        self._result = self._error_type = self._thread = None
        self._tokens = {}
        self.address = self.broker_id = None
        self._closed = False
        self._claimed = self._ready = False
        self._run_entered = self._revoked = False

    def __repr__(self):
        return "<private supervised service controller>"

    def client(self, team):
        """Return a team-scoped IPC client to a trusted controller recipient."""
        from .ipc import IPCClient
        with self._lock:
            if self._closed or self._done.is_set() or team not in self._tokens:
                raise RuntimeError("team credential is unavailable")
            return IPCClient(self.address, self._tokens[team])

    def shutdown(self, *, timeout=30):
        """Request shutdown and return the supervisor's actual teardown result."""
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("invalid shutdown timeout")
        with self._lock:
            if self._closed:
                raise RuntimeError("service controller is already closed")
            if not self._claimed:
                raise RuntimeError("service controller has not been started")
            try:
                self._closed = True
                self._tokens.clear()
                if self._ready:
                    _same_parent(self._stop_path, self._parent_id)
                    fd = os.open(self._stop_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    os.close(fd)
                elif self._channel is not None:
                    self._channel.abort()
            except BaseException:
                if self._channel is not None:
                    self._channel.abort()
                raise
        return self.wait(timeout=timeout)

    def wait(self, *, timeout=30):
        """Wait again after a shutdown timeout; this never asserts clean state."""
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("invalid wait timeout")
        if not self._done.wait(timeout):
            if self._channel is not None:
                self._channel.abort()
            raise TimeoutError("service owner did not finish; device state remains uncertain")
        if self._error_type is not None:
            raise RuntimeError("service supervisor failed: " + self._error_type)
        if self._result is None:
            raise RuntimeError("service supervisor did not produce a result")
        return self._result


def _prepare_owner_controller(stop_path):
    """Prepare a caller-held recovery handle before starting any process."""
    path, parent_id = _stop_path(stop_path)
    return _ServiceController(None, path, parent_id)


def _start_owner_controller(controller, command, *, device_lease_path, marker_path, identity, env,
                            cwd, teams, startup_timeout=60, owner_timeout=600,
                            lease_timeout=30, terminate_grace=5):
    """Start a trusted preconfigured owner; not an arbitrary-client API.

    The caller must independently construct the fixed command, runtime and
    profile. A missing/malformed frame aborts the supervisor. A supervisor that
    has not exited at the cleanup deadline remains rooted for process lifetime.
    """
    if type(controller) is not _ServiceController:
        raise TypeError("a caller-held service controller is required")
    teams = _teams(teams)
    for value in (startup_timeout, owner_timeout, lease_timeout, terminate_grace):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 3600:
            raise ValueError("controller timeouts must be finite and positive")
    channel = thread = frame = address = broker_id = tokens = None
    claimed = False
    try:
        with controller._lock:
            if controller._claimed or controller._closed:
                raise RuntimeError("service controller is already used")
            claimed = True
            controller._claimed = True
        channel = StartupChannel(max_bytes=65536)
        with controller._lock:
            if controller._closed:
                raise RuntimeError("service startup was cancelled")
            controller._channel = channel
        with _CONTROLLERS_LOCK:
            if len(_CONTROLLERS) >= 16:
                raise RuntimeError("service supervisor capacity is exhausted")
            _CONTROLLERS.add(controller)

        def run():
            try:
                with controller._lock:
                    if controller._revoked:
                        return
                    controller._run_entered = True
                controller._result = run_supervised(
                    command, device_lease_path=device_lease_path, marker_path=marker_path,
                    identity=identity, startup_channel=channel, timeout=owner_timeout,
                    lease_timeout=lease_timeout, terminate_grace=terminate_grace,
                    cwd=cwd, env=env, inherit_environment=False, require_zero_exit=True)
            except BaseException as exc:
                controller._error_type = type(exc).__name__
            finally:
                with controller._lock:
                    controller._closed = True
                    controller._tokens.clear()
                with _CONTROLLERS_LOCK:
                    _CONTROLLERS.discard(controller)
                controller._done.set()

        thread = Thread(target=run, name="looking-glass-service-supervisor", daemon=True)
        controller._thread = thread
        with controller._lock:
            if controller._closed:
                raise RuntimeError("service startup was cancelled")
            thread.start()
        frame = channel.receive(timeout=startup_timeout)
        address, broker_id, tokens = _decode_ready(frame, teams)
        frame = None
        if not channel.wait_closed(timeout=terminate_grace):
            raise RuntimeError("service owner exited or startup reader did not close")
        with controller._lock:
            if controller._closed or controller._done.is_set():
                raise RuntimeError("service owner exited during startup")
            controller.address, controller.broker_id, controller._tokens = address, broker_id, tokens
            controller._ready = True
        return controller
    except BaseException:
        frame = address = broker_id = tokens = None
        if not claimed:
            raise
        with controller._lock:
            controller._tokens.clear()
            # The target cannot enter run_supervised after this gate is
            # revoked. This also covers start() raising after creating a
            # thread that has not yet acquired the gate.
            revoked = not controller._run_entered
            if revoked:
                controller._revoked = True
                controller._closed = True
                controller._error_type = "StartupCancelled"
                controller._done.set()
        if channel is not None:
            try:
                channel.abort()
            except BaseException:
                pass
        if revoked:
            with _CONTROLLERS_LOCK:
                _CONTROLLERS.discard(controller)
        elif thread is not None and thread.ident is not None:
            thread.join(timeout=min(terminate_grace + lease_timeout, 5))
        raise
