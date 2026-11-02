"""Fabricated saved-chain checks for the last no-model process observer."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import injection_runtime_replay_driver as driver
from tools import injection_validation_host as host


def test_root_terminal_requires_exact_nested_observation_and_distinct_processes(tmp_path, monkeypatch):
    run = tmp_path / 'validation-test'
    run.mkdir()
    (run / 'parent-config.json').write_bytes(b'config')
    command = ['fabricated-command']
    command_raw = driver.canonical(command)
    (run / 'operator-command.json').write_bytes(command_raw)
    exit_raw = b'operator-terminal'
    (run / 'validation-exit.json').write_bytes(exit_raw)
    for name, pid in (('parent-observer', 21), ('observer-observer', 22)):
        (run / name).mkdir()
        (run / name / 'observation.json').write_bytes(driver.canonical({'parent_pid': pid}))
    context = {'plan_bytes': b'plan', 'plan': {'sources': {'driver': 'a' * 64}}}
    config = {'run_dir': str(run)}
    owner = SimpleNamespace()
    monkeypatch.setattr(driver, 'verify_parent_command', lambda *a, **k: command)
    monkeypatch.setattr(driver, '_validation_supervisor', lambda *a: (None, {'child_pid': 20}))
    seen = []
    monkeypatch.setattr(driver, 'check_validation_exit_receipt',
                        lambda *args: seen.append(args[-2:]) or args[-2:] == (23, 230))
    observed = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': 23, 'parent_birth_100ns': 230,
        'config_sha256': hashlib.sha256(b'config').hexdigest(),
        'operation_sha256': hashlib.sha256(b'plan').hexdigest(),
        'command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'supervisor_source_sha256': 'a' * 64,
        'supervisor_terminal_sha256': hashlib.sha256(exit_raw).hexdigest()}
    raw = driver.canonical(observed)
    assert host.check_root_terminal(driver, context, config, object(), owner, raw, 24, 240) is True
    assert seen == [(23, 230)]
    for pid in (20, 21, 22, 23):
        with pytest.raises(ValueError):
            host.check_root_terminal(driver, context, config, object(), owner, raw, pid, 240)
    for change in ({'parent_reaped': 1}, {'parent_birth_100ns': 231},
                   {'command_sha256': 'b' * 64}, {'supervisor_terminal_sha256': 'c' * 64}):
        with pytest.raises(ValueError):
            host.check_root_terminal(driver, context, config, object(), owner,
                                     driver.canonical(dict(observed, **change)), 24, 240)


def test_host_start_gate_precedes_driver_loading(monkeypatch):
    calls = []
    monkeypatch.setattr(host, 'run', lambda *args: calls.append(args))
    import io
    assert host.main(['x'] * 6, io.BytesIO(b'')) == 1
    assert calls == []
    assert host.main(['x'] * 6, io.BytesIO(driver.START_TOKEN)) == 0
    assert calls == [tuple(['x'] * 6)]


def test_isolated_host_eof_does_not_access_project_or_start_work(tmp_path):
    source = Path(host.__file__).resolve()
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    script = r"""
import atexit,hashlib,sys
source, expected = sys.argv[1:]
events = []
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.getaddrinfo'):
        events.append(event)
sys.addaudithook(audit)
atexit.register(lambda: print('unexpected-before-gate') if events or 'torch' in sys.modules else None)
raw = open(source, 'rb').read()
assert hashlib.sha256(raw).hexdigest() == expected
sys.argv = [source] + ['absent'] * 6
exec(compile(raw, source, 'exec'), {'__name__': '__main__', '__file__': source})
"""
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                             '-c', script, str(source), hashlib.sha256(source.read_bytes()).hexdigest()],
                            input=b'', capture_output=True, timeout=30)
    assert result.returncode == 1 and result.stdout == b'' and result.stderr == b''
    assert not any(prefix.iterdir())


def test_source_loader_requires_exact_driver_digest(tmp_path):
    source = tmp_path / 'driver.py'
    source.write_bytes(b'VALUE = 7\n')
    with pytest.raises(ValueError):
        host._load_driver(str(source), '0' * 64)
    loaded = host._load_driver(str(source), hashlib.sha256(source.read_bytes()).hexdigest())
    assert loaded.VALUE == 7
    sys.modules.pop(loaded.__name__)


def test_unbound_driver_never_executes_before_plan_authentication(tmp_path):
    source = Path(host.__file__).resolve()
    marker = tmp_path / 'driver-ran'
    driver_source = tmp_path / 'unbound-driver.py'
    driver_source.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("ran")\n',
                             encoding='utf-8')
    bootstrap = r"""
import hashlib,sys
source, expected, *args = sys.argv[1:]
raw = open(source, 'rb').read()
assert hashlib.sha256(raw).hexdigest() == expected
sys.argv = [source] + args
exec(compile(raw, source, 'exec'), {'__name__': '__main__', '__file__': source})
"""
    args = [str(tmp_path / 'absent-config.json'), str(driver_source),
            hashlib.sha256(driver_source.read_bytes()).hexdigest(),
            'https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + '1' * 40
            + '/tools/injection_validation_host.py', '0' * 64, '0' * 64]
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', bootstrap,
                             str(source), hashlib.sha256(source.read_bytes()).hexdigest(), *args],
                            input=driver.START_TOKEN, capture_output=True, timeout=30)
    assert result.returncode == 1 and result.stdout == b'' and result.stderr == b''
    assert not marker.exists()


def test_host_terminal_binds_the_retained_process_and_root_observation(tmp_path, monkeypatch):
    run = tmp_path / 'validation-test'
    run.mkdir()
    config_path = run / 'parent-config.json'
    config_path.write_bytes(b'config')
    driver_path = str(tmp_path / 'driver.py')
    owner = SimpleNamespace(_BOOTSTRAP='bootstrap')
    context = {'plan_bytes': b'plan', 'plan': {'sources': {'driver': 'a' * 64}},
               'source_locations': {'driver': {'path': driver_path}}}
    config = {'run_dir': str(run)}
    command = [str(sys._base_executable), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(run / 'v2-cache-x'),
               '-c', owner._BOOTSTRAP, driver_path, 'a' * 64, '--validation-root', str(config_path)]
    command_raw = driver.canonical(command)
    (run / 'root-command.json').write_bytes(command_raw)
    operator_raw = driver.canonical({'parent_pid': 23})
    (run / 'operator-observer').mkdir()
    (run / 'operator-observer' / 'observation.json').write_bytes(operator_raw)
    for name, pid in (('parent-observer', 21), ('observer-observer', 22)):
        (run / name).mkdir()
        (run / name / 'observation.json').write_bytes(driver.canonical({'parent_pid': pid}))
    monkeypatch.setattr(driver, '_validation_supervisor', lambda *a: (None, {'child_pid': 20}))
    seen = []
    monkeypatch.setattr(host, 'check_root_terminal', lambda *args: seen.append(args[-2:]) or True)
    observation = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': 24, 'parent_birth_100ns': 240,
        'config_sha256': hashlib.sha256(b'config').hexdigest(),
        'operation_sha256': hashlib.sha256(b'plan').hexdigest(),
        'command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'supervisor_source_sha256': 'a' * 64,
        'supervisor_terminal_sha256': hashlib.sha256(operator_raw).hexdigest()}
    obs_raw = driver.canonical(observation)
    (run / 'root-observer').mkdir()
    (run / 'root-observer' / 'observation.json').write_bytes(obs_raw)
    terminal = {'schema': 'mlg-injection-validation-host-terminal-v1', 'status': 'awaiting_host_exit',
        'host_pid': 25, 'host_birth_100ns': 250, 'root_pid': 24, 'root_birth_100ns': 240,
        'host_source_sha256': 'b' * 64, 'driver_source_sha256': 'a' * 64,
        'plan_sha256': hashlib.sha256(b'plan').hexdigest(), 'config_sha256': hashlib.sha256(b'config').hexdigest(),
        'root_observation_sha256': hashlib.sha256(obs_raw).hexdigest(), 'forward_count': 0}
    raw = driver.canonical(terminal)
    assert host.check_host_terminal(driver, context, config, object(), owner, raw, 25, 250, 'b' * 64)
    assert seen == [(24, 240)]
    for pid in (20, 21, 22, 23, 24):
        with pytest.raises(ValueError):
            host.check_host_terminal(driver, context, config, object(), owner, raw, pid, 250, 'b' * 64)
    with pytest.raises(ValueError):
        host.check_host_terminal(driver, context, config, object(), owner, raw, 25, 251, 'b' * 64)
    (run / 'root-observer' / 'observation.json').write_bytes(driver.canonical(dict(observation, parent_reaped=1)))
    with pytest.raises(ValueError):
        host.check_host_terminal(driver, context, config, object(), owner, raw, 25, 250, 'b' * 64)

