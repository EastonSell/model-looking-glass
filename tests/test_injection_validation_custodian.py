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


@pytest.mark.parametrize('fault', ['spawn', 'duplicate', 'short_write', 'write', 'wait',
                                   'bad_exit', 'terminal', 'terminal_check'])
def test_launched_host_failure_retains_process_and_leases(monkeypatch, tmp_path, fault):
    """A failed host launch cannot release custody or publish a terminal."""
    run_synthetic_custodian(monkeypatch, tmp_path, fault)


def test_successful_host_is_verified_before_custody_is_released(monkeypatch, tmp_path):
    run_synthetic_custodian(monkeypatch, tmp_path, None)


def run_synthetic_custodian(monkeypatch, tmp_path, fault):
    import importlib
    from tools import injection_runtime_replay_owner as owner

    assert not owner._QUARANTINED
    workspace = tmp_path / 'workspace'
    run_dir = workspace / 'run'
    run_dir.mkdir(parents=True)
    config_path, host_path, driver_path = (str(run_dir / name) for name in ('config', 'host', 'driver'))
    own_path = str(Path(custodian.__file__).resolve())
    raw = {config_path: b'config', host_path: b'host', own_path: b'custodian',
           str(run_dir / 'host-terminal.json'): b'host-terminal'}
    events, leases = [], []
    plan = b'synthetic plan'
    monkeypatch.setattr(custodian, 'PLAN_SHA', hashlib.sha256(plan).hexdigest())
    flags = SimpleNamespace(isolated=1, ignore_environment=1, no_site=1, no_user_site=1,
                            safe_path=1, optimize=0)
    monkeypatch.setattr(custodian, 'sys', SimpleNamespace(flags=flags, dont_write_bytecode=True,
        executable=sys.executable, _base_executable=sys._base_executable))
    finder = SimpleNamespace(assert_intact=lambda: True)
    claim = SimpleNamespace(__spec__=SimpleNamespace(loader=finder), _diagnostic_bytes=lambda _: (0,))
    monkeypatch.setattr(importlib, 'import_module', lambda name: claim)

    class Lease:
        def __init__(self, *args):
            self.path = run_dir / 'v2-cache-test'
            self.closed = False
            leases.append(self)
        def __enter__(self):
            events.append('lease_enter')
            return self
        def handles(self):
            return (len(leases) + 10,)
        def close(self):
            self.closed = True
            events.append('lease_close')

    class Pipe:
        def write(self, value):
            events.append('write')
            assert value == custodian.START
            if fault == 'write':
                raise OSError('write failed')
            return len(value) - 1 if fault == 'short_write' else len(value)
        def close(self):
            events.append('pipe_close')

    class Process:
        def __init__(self, *args, **kwargs):
            events.append('spawn')
            if fault == 'spawn':
                raise OSError('spawn failed')
            self.stdin = Pipe()
            self.pid = 24
        def wait(self, timeout):
            events.append('wait')
            if fault == 'wait':
                raise TimeoutError('wait uncertain')
            return 1 if fault == 'bad_exit' else 0

    def duplicate(handles, process):
        events.append('duplicate')
        if fault == 'duplicate':
            raise OSError('duplication failed')
        return tuple(range(1, len(handles) + 1))

    def read(path, limit=None):
        if path.endswith('host-terminal.json') and fault == 'terminal':
            raise OSError('terminal unavailable')
        return raw[path]

    def write_once(path, value):
        assert path not in raw
        raw[path] = value
        events.append('write_terminal' if path.endswith('custodian-terminal.json') else 'write_command')

    guard = SimpleNamespace(SourceFileLease=Lease, PrivateEmptyCache=Lease, duplicate_to_process=duplicate)
    observer = SimpleNamespace(_birth_time=lambda p: 240,
                               _close_process_handle=lambda p: events.append('handle_close'))
    owner_proxy = SimpleNamespace(_QUARANTINED=owner._QUARANTINED, _BOOTSTRAP='bootstrap',
        verify_private_parent=lambda *a: True, _current_process_identity=lambda: (25, 250),
        run_guarded=owner.under_protection)
    host = SimpleNamespace(check_host_terminal=lambda *a: fault != 'terminal_check')
    modules = {'owner': owner_proxy, 'child': object(), 'guard': guard, 'outer_observer': observer}
    config = {'workspace': str(workspace), 'run_dir': str(run_dir), 'plan_path': 'plan',
              'plan_url': 'url', 'source_locations': {}, 'public_paths': {}, 'frozen_root': 'frozen',
              'cache_dir': 'cache', 'parent_identity': (1, 2), 'ledger_identity': (3, 4)}
    context = {'plan': {'sources': {'driver': custodian.DRIVER_SHA}}, 'plan_bytes': plan,
               'cache_dir': 'cache', 'public': {'heldout_runtime': b'{}'}, 'graph': object()}
    driver = SimpleNamespace(_ordinary_source=read, _fetch_source=lambda _: b'host',
        parse_validation_parent_config=lambda *a: config, capture_preflight=lambda **kw: context,
        canonical=lambda obj: json.dumps(obj, sort_keys=True, separators=(',', ':')).encode(),
        _execution_modules=lambda _: modules, recheck_preflight=lambda *a: True,
        check_runtime_environment=lambda *a: True, _canonical_object=lambda *a: {},
        check_unclaimed_validation=lambda *a: True, lease_inputs=lambda *a, **kw: ((), (), ()),
        _write_once=write_once, _process_identity=lambda identity: identity,
        run_bound_imports=lambda graph, callback: callback(finder))
    monkeypatch.setattr(custodian, 'load', lambda path, digest, name: driver if path == driver_path else host)
    monkeypatch.setattr(custodian.subprocess, 'Popen', Process)
    try:
        if fault:
            with pytest.raises((OSError, TimeoutError, ValueError)):
                custodian.run(config_path, host_path, driver_path)
            assert len(owner._QUARANTINED) == 1
            custody = owner._QUARANTINED[0]
            assert len(custody['objects']) == 1 and isinstance(custody['objects'][0], Process)
            assert custody['leases'] and all(not lease.closed for lease in custody['leases'])
            assert 'write_terminal' not in events
            assert 'handle_close' not in events
            if fault in ('spawn', 'duplicate'):
                assert 'write' not in events
            if fault in ('short_write', 'write'):
                assert 'wait' not in events
        else:
            result = custodian.run(config_path, host_path, driver_path)
            assert result['status'] == 'awaiting_custodian_exit'
            assert events.index('duplicate') < events.index('write') < events.index('wait')
            assert events.index('wait') < events.index('handle_close') < events.index('write_terminal')
            assert all(lease.closed for lease in leases)
            assert not owner._QUARANTINED
    finally:
        owner._QUARANTINED.clear()
