"""CPU-only direct controller wait and partial-receipt tests."""

import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools' / 'observe_pretrained_gpu_trace_pilot.py'
SPEC = importlib.util.spec_from_file_location('pretrained_trace_controller_observer_test', PATH)
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


def test_controller_temp_is_inside_observation_and_does_not_mutate_parent_env(tmp_path, monkeypatch):
    process = FakeProcess()
    observed = {}
    def spawn(*_args, **kwargs):
        observed.update(kwargs['env'])
        return process
    monkeypatch.setattr(observer.subprocess, 'Popen', spawn)
    monkeypatch.setattr(observer, '_grant', lambda *_a: None)
    monkeypatch.setattr(observer, '_identity_matches', lambda *_a: True)
    environment = {'TEMP': 'old', 'TMP': 'old', 'OTHER': 'retained'}
    observation = tmp_path / 'observation'
    assert observer.observe(['fake'], checkout=tmp_path, observation=observation, run=tmp_path / 'run',
                            environment=environment, source_commit='a' * 40) == 0
    assert observed['TEMP'] == observed['TMP'] == str(observation / 'temp')
    assert (observation / 'temp').is_dir() and observed['OTHER'] == 'retained'
    assert environment['TEMP'] == environment['TMP'] == 'old'


def test_start_record_failure_still_waits_for_the_rooted_controller(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(observer.subprocess, 'Popen', lambda *_a, **_k: process)
    monkeypatch.setattr(observer, '_identity_matches', lambda *_a: True)
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
    monkeypatch.setattr(observer, '_grant', lambda *_a: None)
    monkeypatch.setattr(observer, '_identity_matches', lambda *_a: True)
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
    monkeypatch.setattr(observer, '_grant', lambda *_a: None)
    monkeypatch.setattr(observer, '_identity_matches', lambda *_a: True)
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


def _gated_child_command(marker):
    bootstrap = PATH.parents[1] / 'scripts' / 'pretrained_gpu_trace_pilot_bootstrap.py'
    code = ('import importlib.util,pathlib; '
            f's=importlib.util.spec_from_file_location("trace_bootstrap",{str(bootstrap)!r}); '
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m); '
            'identity=m._await_controller_grant();m._write_controller_identity(identity,"' + 'a' * 40 + '"); '
            f'pathlib.Path({str(marker)!r}).write_text("granted",encoding="ascii")')
    return [sys._base_executable, '-I', '-S', '-B', '-c', code]


def test_direct_interpreter_validation_rejects_venv_shim():
    base = Path(sys._base_executable)
    observer._direct_python(base, base, base)
    if Path(sys.executable).resolve() == base.resolve():
        pytest.skip('test runner is already the direct base interpreter')
    with pytest.raises(ValueError, match='direct base interpreter'):
        observer._direct_python(sys.executable, base, base)


@pytest.mark.skipif(os.name not in ('nt', 'posix'), reason='gate supports Windows and POSIX')
def test_real_child_only_runs_after_the_observer_grant(tmp_path):
    marker, observation = tmp_path / 'child-work', tmp_path / 'observation'
    assert observer.observe(_gated_child_command(marker), checkout=tmp_path, observation=observation,
                            run=tmp_path / 'run', environment=os.environ.copy(),
                            source_commit='a' * 40) == 0
    assert marker.read_text(encoding='ascii') == 'granted'
    receipt = json.loads((observation / 'exit.json').read_bytes())
    identity = json.loads((observation / 'controller-identity.json').read_bytes())
    assert identity['pid'] == receipt['pid'] and identity['parent_pid'] == os.getpid()
    assert receipt['start_recorded'] is True and receipt['reaped'] is True
    assert receipt['controller_identity_verified'] is True


@pytest.mark.skipif(os.name not in ('nt', 'posix'), reason='gate supports Windows and POSIX')
def test_interrupted_spawn_return_revokes_grant_before_real_child_work(tmp_path, monkeypatch):
    marker, observation = tmp_path / 'child-work', tmp_path / 'observation'
    actual_popen, children = subprocess.Popen, []
    def spawn_then_interrupt(*args, **kwargs):
        children.append(actual_popen(*args, **kwargs))
        raise KeyboardInterrupt('interrupted Popen return')
    monkeypatch.setattr(observer.subprocess, 'Popen', spawn_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match='interrupted Popen return'):
        observer.observe(_gated_child_command(marker), checkout=tmp_path, observation=observation,
                         run=tmp_path / 'run', environment=os.environ.copy(), source_commit='a' * 40)
    assert len(children) == 1 and children[0].wait(timeout=35) != 0
    assert not marker.exists() and not (observation / 'exit.json').exists()


@pytest.mark.skipif(os.name not in ('nt', 'posix'), reason='gate supports Windows and POSIX')
def test_interrupted_grant_return_waits_for_the_already_granted_child(tmp_path, monkeypatch):
    marker, observation = tmp_path / 'child-work', tmp_path / 'observation'
    original = observer._grant
    def grant_then_interrupt(*args):
        original(*args)
        raise KeyboardInterrupt('grant return interrupted')
    monkeypatch.setattr(observer, '_grant', grant_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match='grant return interrupted'):
        observer.observe(_gated_child_command(marker), checkout=tmp_path, observation=observation,
                         run=tmp_path / 'run', environment=os.environ.copy(), source_commit='a' * 40)
    assert marker.read_text(encoding='ascii') == 'granted'
    receipt = json.loads((observation / 'exit.json').read_bytes())
    assert receipt['reaped'] is True and receipt['wrapper_error_type'] == 'KeyboardInterrupt'
    assert receipt['controller_identity_verified'] is True


@pytest.mark.skipif(os.name not in ('nt', 'posix'), reason='gate supports Windows and POSIX')
def test_wrong_controller_pid_cannot_reach_real_child_work(tmp_path, monkeypatch):
    marker, observation = tmp_path / 'child-work', tmp_path / 'observation'
    original = observer._grant
    monkeypatch.setattr(observer, '_grant', lambda fd, child, parent: original(fd, child + 1, parent))
    assert observer.observe(_gated_child_command(marker), checkout=tmp_path, observation=observation,
                            run=tmp_path / 'run', environment=os.environ.copy(),
                            source_commit='a' * 40) != 0
    assert not marker.exists() and not (observation / 'controller-identity.json').exists()
    receipt = json.loads((observation / 'exit.json').read_bytes())
    assert receipt['reaped'] is True and receipt['controller_identity_verified'] is False
