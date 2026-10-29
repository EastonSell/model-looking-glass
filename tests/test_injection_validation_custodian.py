"""No-model startup checks for the retained-handle validation custodian."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import injection_validation_custodian as custodian
from tools import injection_runtime_replay_driver as driver


def test_start_gate_precedes_all_custodian_work(monkeypatch):
    called = []
    monkeypatch.setattr(custodian, 'run', lambda *args: called.append(args))
    assert custodian.main(['a', 'b', 'c'], io.BytesIO(b'')) == 1
    assert called == []
    assert custodian.main(['a', 'b', 'c'], io.BytesIO(custodian.START)) == 0
    assert called == [('a', 'b', 'c')]


def test_unbound_source_never_executes(tmp_path):
    marker = tmp_path / 'marker'
    source = tmp_path / 'source.py'
    source.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n', encoding='utf-8')
    with pytest.raises(ValueError):
        custodian.load(str(source), '0' * 64, '_mlg_fake_source')
    assert not marker.exists()


def test_isolated_eof_never_imports_project_or_starts_work(tmp_path):
    source = Path(custodian.__file__).resolve()
    bootstrap = r"""
import atexit,hashlib,sys
source, digest = sys.argv[1:]
events = []
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.getaddrinfo'):
        events.append(event)
sys.addaudithook(audit)
atexit.register(lambda: print('unexpected') if events or 'torch' in sys.modules else None)
raw = open(source, 'rb').read()
assert hashlib.sha256(raw).hexdigest() == digest
sys.argv = [source, 'unused', 'unused', 'unused']
exec(compile(raw, source, 'exec'), {'__name__': '__main__', '__file__': source})
"""
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', bootstrap,
                             str(source), hashlib.sha256(source.read_bytes()).hexdigest()],
                            input=b'', capture_output=True, timeout=30)
    assert result.returncode == 1 and result.stdout == b'' and result.stderr == b''


def test_custodian_terminal_binds_host_command_and_observed_identity(tmp_path):
    run = tmp_path / 'validation'
    run.mkdir()
    config_path = run / 'parent-config.json'
    config_path.write_bytes(b'config')
    host_path, driver_path = str(tmp_path / 'host.py'), str(tmp_path / 'driver.py')
    owner = SimpleNamespace(_BOOTSTRAP='bootstrap')
    context = {'plan_bytes': b'plan', 'source_locations': {'driver': {'path': driver_path}}}
    config = {'run_dir': str(run)}
    host = SimpleNamespace(check_host_terminal=lambda *args: args[-3:] == (24, 240, custodian.HOST_SHA))
    command = [str(sys._base_executable), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(run / 'v2-cache-x'),
               '-c', owner._BOOTSTRAP, host_path, custodian.HOST_SHA, str(config_path), driver_path,
               custodian.DRIVER_SHA, custodian.HOST_URL, custodian.HOST_SHA, custodian.PLAN_SHA]
    command_raw = driver.canonical(command)
    (run / 'custodian-host-command.json').write_bytes(command_raw)
    host_raw = b'host-terminal'
    (run / 'host-terminal.json').write_bytes(host_raw)
    for name, pid in (('root-observer', 23), ('operator-observer', 22),
                      ('observer-observer', 21), ('parent-observer', 20)):
        (run / name).mkdir()
        (run / name / 'observation.json').write_bytes(driver.canonical({'parent_pid': pid}))
    (run / 'child-supervisor').mkdir()
    (run / 'child-supervisor' / 'supervisor.json').write_bytes(driver.canonical({'child_pid': 19}))
    terminal = {'schema': 'mlg-injection-validation-custodian-terminal-v1', 'status': 'awaiting_custodian_exit',
        'custodian_pid': 25, 'custodian_birth_100ns': 250, 'host_pid': 24, 'host_birth_100ns': 240,
        'host_terminal_sha256': hashlib.sha256(host_raw).hexdigest(),
        'host_command_sha256': hashlib.sha256(command_raw).hexdigest(), 'host_source_sha256': custodian.HOST_SHA,
        'driver_source_sha256': custodian.DRIVER_SHA, 'plan_sha256': custodian.PLAN_SHA,
        'config_sha256': hashlib.sha256(b'config').hexdigest(), 'custodian_source_sha256': 'a' * 64,
        'forward_count': 0}
    raw = driver.canonical(terminal)
    assert custodian.check_custodian_terminal(driver, host, context, config, object(), owner,
                                              raw, 25, 250, 'a' * 64, host_path)
    with pytest.raises(ValueError):
        custodian.check_custodian_terminal(driver, host, context, config, object(), owner,
                                            raw, 24, 250, 'a' * 64, host_path)
    bad = list(command)
    bad[5] = 'pycache_prefix=' + str(run / 'unsafe')
    (run / 'custodian-host-command.json').write_bytes(driver.canonical(bad))
    with pytest.raises(ValueError):
        custodian.check_custodian_terminal(driver, host, context, config, object(), owner,
                                            raw, 25, 250, 'a' * 64, host_path)
