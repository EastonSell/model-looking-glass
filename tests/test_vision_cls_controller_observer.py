"""CPU-only direct controller wait and partial-receipt tests."""

import importlib.util
import inspect
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools' / 'observe_vision_cls_exposed_gpu.py'
SPEC = importlib.util.spec_from_file_location('vision_cls_controller_observer_test', PATH)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


class FakeProcess:
    pid = 4242

    def __init__(self, interrupt=False):
        self.wait_calls = 0
        self.interrupt = interrupt

    def wait(self, timeout):
        assert timeout == 1
        self.wait_calls += 1
        if self.interrupt and self.wait_calls == 1:
            raise KeyboardInterrupt()
        return 0


def test_start_record_failure_still_waits_for_the_rooted_controller(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(observer.subprocess, 'Popen', lambda *_a, **_k: process)
    original = observer._record
    def fail_start(path, value):
        if path.name == 'start.json':
            raise OSError('fabricated start write failure')
        return original(path, value)
    monkeypatch.setattr(observer, '_record', fail_start)
    observation = tmp_path / 'observation'
    with pytest.raises(OSError, match='start write failure'):
        observer.observe(['fake'], checkout=tmp_path, observation=observation, run=tmp_path / 'run',
                         environment={}, source_commit='a' * 40)
    receipt = json.loads((observation / 'exit.json').read_bytes())
    assert process.wait_calls == 1 and receipt['reaped'] is True and receipt['exit_code'] == 0
    assert receipt['start_recorded'] is False and receipt['wrapper_error_type'] == 'OSError'


def test_cancellation_during_wait_is_preserved_after_the_controller_is_reaped(tmp_path, monkeypatch):
    process = FakeProcess(interrupt=True)
    monkeypatch.setattr(observer.subprocess, 'Popen', lambda *_a, **_k: process)
    observation = tmp_path / 'observation'
    with pytest.raises(KeyboardInterrupt):
        observer.observe(['fake'], checkout=tmp_path, observation=observation, run=tmp_path / 'run',
                         environment={}, source_commit='a' * 40)
    receipt = json.loads((observation / 'exit.json').read_bytes())
    assert process.wait_calls == 2 and receipt['reaped'] is True and receipt['exit_code'] == 0
    assert receipt['start_recorded'] is True and receipt['wrapper_error_type'] == 'KeyboardInterrupt'


def test_one_interruption_at_drain_entry_still_waits_and_records_exit(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(observer.subprocess, 'Popen', lambda *_a, **_k: process)
    source, first_line = inspect.getsourcelines(observer.observe)
    target = first_line + next(i for i, line in enumerate(source) if line.strip() == 'if process is None:')
    fired = []
    def trace(frame, event, _arg):
        if frame.f_code.co_filename == str(PATH) and event == 'line' and frame.f_lineno == target and not fired:
            fired.append(True)
            raise KeyboardInterrupt()
        return trace
    observation = tmp_path / 'observation'
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt):
            observer.observe(['fake'], checkout=tmp_path, observation=observation, run=tmp_path / 'run',
                             environment={}, source_commit='a' * 40)
    finally:
        sys.settrace(previous)
    receipt = json.loads((observation / 'exit.json').read_bytes())
    assert fired == [True] and process.wait_calls == 1 and receipt['reaped'] is True
    assert receipt['wrapper_error_type'] == 'KeyboardInterrupt'
