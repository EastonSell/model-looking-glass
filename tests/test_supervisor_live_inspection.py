import json
import ctypes
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from looking_glass import supervisor


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows process handles')
def test_inspection_duplicates_the_rooted_live_child_and_detects_exit(tmp_path):
    lock = tmp_path / 'device.lock'
    marker = lock.with_suffix('.unsafe.json')
    stop = tmp_path / 'stop'
    code = ('import pathlib,sys,time\n'
            'p=pathlib.Path(sys.argv[1])\n'
            'while not p.exists(): time.sleep(0.01)\n')
    result = []
    errors = []

    def supervise():
        try:
            result.append(supervisor.run_supervised(
                [sys._base_executable, '-B', '-c', code, str(stop)],
                device_lease_path=lock, marker_path=marker, timeout=15,
                require_zero_exit=True))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=supervise, daemon=True)
    thread.start()
    capability = None
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if marker.exists():
                try:
                    run_id = json.loads(marker.read_text(encoding='utf-8'))['run_id']
                    capability = supervisor.inspect_live_child(run_id)
                    break
                except (OSError, ValueError, KeyError, supervisor.SupervisorError):
                    pass
            time.sleep(0.02)
        assert capability is not None
        capability.check_live()
        assert capability in supervisor.pending_live_child_inspections()
        assert capability.run_id == run_id and capability.pid > 0 and int(capability.birth) > 0
        with pytest.raises(supervisor.SupervisorError):
            supervisor.inspect_live_child('f' * 32)
        stop.write_text('stop', encoding='ascii')
        thread.join(timeout=8)
        assert not thread.is_alive() and not errors and len(result) == 1
        assert result[0].pid == capability.pid and result[0].run_id == capability.run_id
        assert result[0].clean is False  # The fake owner supplied no clean attestation.
        with pytest.raises(supervisor.SupervisorError):
            capability.check_live()
        capability.close()
        assert capability not in supervisor.pending_live_child_inspections()
        with pytest.raises(supervisor.SupervisorError):
            capability.check_live()
    finally:
        if not stop.exists():
            stop.write_text('stop', encoding='ascii')
        thread.join(timeout=8)
        if capability is not None:
            capability.close()


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows handle layout')
def test_interrupted_duplicate_closes_the_already_returned_handle(monkeypatch):
    from ctypes import wintypes
    class Function:
        def __init__(self, call):
            self.call = call
        def __call__(self, *args):
            return self.call(*args)
    closed = []
    requested = []
    def duplicate(_src, _handle, _dst, out, access, inherit, options):
        requested.append((access, inherit, options))
        ctypes.cast(out, ctypes.POINTER(wintypes.HANDLE)).contents.value = 9876
        raise KeyboardInterrupt()
    kernel = SimpleNamespace(DuplicateHandle=Function(duplicate),
        GetProcessId=Function(lambda _handle: 1234), GetCurrentProcess=Function(lambda: 1),
        CloseHandle=Function(lambda handle: closed.append(handle.value) or True))
    monkeypatch.setattr(supervisor, '_windows_kernel', lambda: kernel)
    run_id = 'c' * 32
    record = supervisor._SpawnCustody(run_id)
    record._child = SimpleNamespace(pid=1234, _handle=456)
    with supervisor._spawn_custody_lock:
        supervisor._spawn_custodies[run_id] = record
    before = set(supervisor.pending_live_child_inspections())
    try:
        with pytest.raises(KeyboardInterrupt):
            supervisor.inspect_live_child(run_id)
        assert closed == [9876]
        assert requested == [(0x00100000 | 0x1000, False, 0)]
        assert set(supervisor.pending_live_child_inspections()) == before
    finally:
        with supervisor._spawn_custody_lock:
            del supervisor._spawn_custodies[run_id]


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows handle layout')
def test_interrupted_handle_close_is_rooted_and_never_retried():
    calls = []
    def close(_handle):
        calls.append('close')
        raise KeyboardInterrupt()
    capability = supervisor._LiveChildInspection(SimpleNamespace(CloseHandle=close), object(), 'd' * 32, 1234)
    capability._handle.value = 9876  # Fake native handle, never handed to the OS.
    with supervisor._live_inspection_lock:
        supervisor._live_inspections.add(capability)
    try:
        with pytest.raises(KeyboardInterrupt):
            capability.close()
        assert capability in supervisor.pending_live_child_inspections()
        with pytest.raises(supervisor.SupervisorError, match='uncertain'):
            capability.close()
        assert calls == ['close']
    finally:
        with supervisor._live_inspection_lock:
            supervisor._live_inspections.discard(capability)


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows handle layout')
def test_interrupted_retirement_does_not_repeat_native_close(monkeypatch):
    class InterruptedOnce:
        def __init__(self):
            self.calls = 0
        def __enter__(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
        def __exit__(self, *unused):
            return False
    calls = []
    capability = supervisor._LiveChildInspection(
        SimpleNamespace(CloseHandle=lambda _handle: calls.append('close') or True),
        object(), 'e' * 32, 1234)
    capability._handle.value = 9876  # Fake native handle, never handed to the OS.
    supervisor._live_inspections.add(capability)
    monkeypatch.setattr(supervisor, '_live_inspection_lock', InterruptedOnce())
    try:
        with pytest.raises(KeyboardInterrupt):
            capability.close()
        assert capability in supervisor._live_inspections
        capability.close()
        assert capability not in supervisor._live_inspections
        assert calls == ['close']
    finally:
        supervisor._live_inspections.discard(capability)
