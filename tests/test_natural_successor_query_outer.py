"""Real nested Windows Jobs with fabricated inputs and synthetic source bindings."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from looking_glass import (injection_outer_observer as observer, natural_successor_accounting as accounting,
                           natural_successor_query_handoff as handoff, natural_successor_query_outer as outer)
from scripts import v2_source_guards as guards
from tools import natural_agreement_heldout_owner as owner
from test_natural_successor_query import canonical, old_snapshot, rebind_config, setup, sha, SyntheticLease


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(setup, tmp_path, request):
    """Replace only the public caller with an invented offline test harness."""
    config = json.loads(setup['config_path'].read_bytes())
    original = Path(config['sources']['caller']).read_text(encoding='utf-8')
    original = original.rsplit("if __name__ == '__main__':", 1)[0]
    wrapper = tmp_path / 'fabricated_query.py'
    wrapper.write_text(original + '''\nif __name__ == '__main__':
    import argparse,base64,json,os
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--config-sha256',required=True); p.add_argument('--operation-sha256',required=True)
    a=p.parse_args(); remote=json.loads(Path(os.environ['FAKE_QUERY_REMOTE']).read_bytes())
    def reader(url): return base64.b64decode(remote[url])
    def observe(op):
        return {'head':'c'*40,'version':op['inventory_version'],
                'source_sha256':op['inventory_source_sha256'],'inventory_sha256':op['inventory_sha256']}
    try:
        value=run_query(Path(a.config),config_sha256=a.config_sha256,operation_sha256=a.operation_sha256,
                        reader=reader,observe_current=observe)
        print(value['status'])
    except QueryCollision:
        print('collision'); raise SystemExit(2)
    except BaseException:
        if _ACTIVE_OWNER:
            for owner,supervisor,_ in _ACTIVE_OWNER:
                if owner._QUARANTINED or supervisor._QUARANTINED:
                    owner._hold_quarantine()
                    while supervisor._QUARANTINED: _HOLD.wait()
        raise SystemExit(1)
''', encoding='utf-8')
    def change(config, operation):
        config['sources']['caller'] = str(wrapper)
        entry = operation['sources']['caller']
        entry['sha256'] = sha(wrapper.read_bytes())
        setup['remote'][entry['url']] = wrapper.read_bytes()
    rebind_config(setup, change)
    remote = tmp_path / 'fabricated-remote.json'
    remote.write_bytes(canonical({key: base64.b64encode(value).decode('ascii')
                                  for key, value in setup['remote'].items()}))
    config = json.loads(setup['config_path'].read_bytes())
    query = load(wrapper, '_fabricated_query_outer')
    acl_path = Path(config['sources']['acl_verifier'])
    acl = load(acl_path, '_fabricated_acl_outer')
    files = [wrapper, setup['config_path'], Path(config['operation']), remote]
    synthetic = request.node.originalname in (
        'test_orphan_stops_before_invocation_or_comparator',
        'test_orphan_after_invocation_marker_stops_before_comparator',
        'test_observer_exception_retains_owner_custody')
    lease_events = []
    leases = ([SyntheticLease(path.parent, [path], lease_events) for path in files] if synthetic else
              [guards.SourceFileLease(path.parent, [path]) for path in files])
    if synthetic:
        def verify_lease_lifecycle():
            entered = [lease for action, lease in lease_events if action == 'enter']
            closed = [lease for action, lease in lease_events if action == 'close']
            assert closed == list(reversed(entered))
        request.addfinalizer(verify_lease_lifecycle)
    interpreter = str(setup['synthetic_interpreter']) if synthetic else sys._base_executable
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '-1', 'FAKE_QUERY_REMOTE': str(remote)})
    return SimpleNamespace(interpreter=interpreter, setup=setup, query=query, acl=acl, leases=leases, environment=env,
                           source_sha256=sha(wrapper.read_bytes()), wrapper=wrapper)


def invoke(value, *, observer_module=observer, leases=None):
    return outer.observe_query(query=value.query, handoff=handoff, observer=observer_module,
        guards=guards, owner=owner, accounting=accounting, acl=value.acl,
        config_path=value.setup['config_path'], config_sha256=value.setup['config_sha256'],
        operation_sha256=value.setup['operation_sha256'], interpreter=value.interpreter,
        interpreter_sha256=sha(Path(value.interpreter).read_bytes()), query_source_sha256=value.source_sha256,
        source_check=lambda: True, leases=value.leases if leases is None else leases,
        environment=value.environment, timeout=60)


@pytest.mark.skipif(os.name != 'nt', reason='Windows nested Job rehearsal')
def test_real_outer_and_inner_jobs_complete_fabricated_query(prepared):
    before = old_snapshot(prepared.setup['old_parent'])
    result = invoke(prepared)
    assert result['status'] == 'clear' and result['parent_pid'] != result['child_pid']
    assert (prepared.setup['root'] / 'pool-query-invocation.json').is_file()
    assert (prepared.setup['root'] / 'pool-query-outer/observation.json').is_file()
    assert not (prepared.setup['root'] / 'phase-claim.json').exists()
    assert not (prepared.setup['root'] / 'nonce.bin').exists()
    assert old_snapshot(prepared.setup['old_parent']) == before
    assert owner._QUARANTINED == [] and observer._QUARANTINED == []
    with pytest.raises((ValueError, RuntimeError, OSError)):
        invoke(prepared, leases=[guards.SourceFileLease(prepared.wrapper.parent, [prepared.wrapper])])


@pytest.mark.parametrize('name', ['nonce.bin', 'heldout-record.json', 'phase-claim.json', 'pool-query-supervisor'])
@pytest.mark.parametrize('during_lease', [False, True])
def test_orphan_stops_before_invocation_or_comparator(prepared, name, during_lease):
    root = prepared.setup['root']
    orphan = root / name
    def create_orphan():
        if name == 'pool-query-supervisor':
            orphan.mkdir()
        else:
            orphan.write_bytes(b'fabricated orphan')
    class InsertingLease:
        def __enter__(self):
            create_orphan()
            return self
        def handles(self): return []
        def close(self): pass
    if not during_lease:
        create_orphan()
    before = old_snapshot(prepared.setup['old_parent'])
    with pytest.raises((ValueError, RuntimeError, OSError)):
        invoke(prepared, leases=[InsertingLease()] if during_lease else prepared.leases)
    assert not (root / 'pool-query-invocation.json').exists()
    assert not (root / 'pool-query-claim.json').exists()
    assert not (root / 'pool-query-receipt.json').exists()
    assert not (root / 'pool-query-outer').exists()
    assert old_snapshot(prepared.setup['old_parent']) == before


@pytest.mark.parametrize('name', ['nonce.bin', 'heldout-record.json', 'pool-query-supervisor'])
def test_orphan_after_invocation_marker_stops_before_comparator(prepared, monkeypatch, name):
    root = prepared.setup['root']
    original = prepared.query._write_new
    def write_then_insert(path, raw):
        original(path, raw)
        if Path(path).name == 'pool-query-invocation.json':
            orphan = root / name
            if name == 'pool-query-supervisor':
                orphan.mkdir()
            else:
                orphan.write_bytes(b'fabricated orphan')
    monkeypatch.setattr(prepared.query, '_write_new', write_then_insert)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        invoke(prepared)
    assert (root / 'pool-query-invocation.json').exists()
    assert not (root / 'pool-query-claim.json').exists()
    assert not (root / 'pool-query-receipt.json').exists()
    assert not (root / 'pool-query-outer').exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows nested Job regression')
def test_saved_outer_birth_must_match_the_retained_observer_return(prepared):
    def mutate_after_return(**kwargs):
        result = observer.observe_supervisor(**kwargs)
        path = Path(kwargs['output']) / 'observation.json'
        saved = json.loads(path.read_bytes())
        saved['parent_birth_100ns'] += 1
        path.write_bytes(canonical(saved))
        return result
    wrapped = SimpleNamespace(_QUARANTINED=observer._QUARANTINED, observe_supervisor=mutate_after_return)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        invoke(prepared, observer_module=wrapped)
    assert (prepared.setup['root'] / 'pool-query-invocation.json').exists()
    assert not (prepared.setup['root'] / 'phase-claim.json').exists()


def test_observer_exception_retains_owner_custody(prepared, monkeypatch):
    class FakeLease:
        closed = False
        def __enter__(self): return self
        def handles(self): return [1]
        def close(self): self.closed = True
    lease = FakeLease()
    fake_observer = SimpleNamespace(_QUARANTINED=[], observe_supervisor=lambda **kw: (_ for _ in ()).throw(
        RuntimeError('fabricated observer failure')))
    seen = []
    monkeypatch.setattr(outer, '_hold_uncertain', lambda own, obs: seen.append(bool(own._QUARANTINED)))
    try:
        with pytest.raises(RuntimeError):
            invoke(prepared, observer_module=fake_observer, leases=[lease])
        assert seen == [True] and owner._QUARANTINED and not lease.closed
        assert (prepared.setup['root'] / 'pool-query-invocation.json').exists()
        assert not (prepared.setup['root'] / 'phase-claim.json').exists()
    finally:
        owner._QUARANTINED.clear()  # This fixture owns only the fabricated lease.
        lease.close()


def test_quarantine_hold_does_not_unwind_in_a_dedicated_process(tmp_path):
    source = Path(outer.__file__).resolve()
    script = tmp_path / 'hold.py'
    script.write_text("import importlib.util,types,sys\n"
                      "s=importlib.util.spec_from_file_location('test_outer',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                      "m._hold_uncertain(types.SimpleNamespace(_QUARANTINED=[1]),types.SimpleNamespace(_QUARANTINED=[]))\n",
                      encoding='ascii')
    process = subprocess.Popen([sys._base_executable, '-I', '-S', '-B', str(script), str(source)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=10)


@pytest.mark.skipif(os.name != 'nt', reason='Windows silent-breakaway regression')
def test_nonzero_parent_with_breakaway_descendant_keeps_source_custody(prepared, tmp_path):
    """A clean outer Job is not proof that a breakaway descendant has exited."""
    import ctypes
    from ctypes import wintypes

    wrapper = prepared.wrapper
    prefix = wrapper.read_text(encoding='utf-8').rsplit("if __name__ == '__main__':", 1)[0]
    wrapper.write_text(prefix + '''\nif __name__ == '__main__':
    import os,subprocess,sys,time
    code="import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(15)"
    subprocess.Popen([sys._base_executable,'-I','-S','-B','-c',code,os.environ['FAKE_CHILD_PID']],
                     stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(0.6)
    raise SystemExit(1)
''', encoding='utf-8')
    def change(config, operation):
        entry = operation['sources']['caller']
        entry['sha256'] = sha(wrapper.read_bytes())
        prepared.setup['remote'][entry['url']] = wrapper.read_bytes()
    rebind_config(prepared.setup, change)
    config = json.loads(prepared.setup['config_path'].read_bytes())
    child_pid_path = tmp_path / 'fabricated-child-pid.txt'
    host_config = tmp_path / 'fabricated-host.json'
    host_config.write_bytes(canonical({'repo': str(Path(outer.__file__).resolve().parents[1]),
        'wrapper': str(wrapper), 'acl': config['sources']['acl_verifier'],
        'config': str(prepared.setup['config_path']), 'config_sha': prepared.setup['config_sha256'],
        'operation': config['operation'], 'operation_sha': prepared.setup['operation_sha256'],
        'source_sha': sha(wrapper.read_bytes()), 'interpreter_sha': sha(Path(sys._base_executable).read_bytes()),
        'child_pid_path': str(child_pid_path)}))
    host = tmp_path / 'fabricated-host.py'
    host.write_text('''import importlib.util,json,os,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_bytes());sys.path.insert(0,c['repo'])
from looking_glass import injection_outer_observer as observer,natural_successor_accounting as accounting,natural_successor_query_handoff as handoff,natural_successor_query_outer as outer
from scripts import v2_source_guards as guards
from tools import natural_agreement_heldout_owner as owner
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
query=load(c['wrapper'],'synthetic_query');acl=load(c['acl'],'synthetic_acl')
paths=[Path(c['wrapper']),Path(c['config']),Path(c['operation'])]
leases=[guards.SourceFileLease(p.parent,[p]) for p in paths]
env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'-1','FAKE_CHILD_PID':c['child_pid_path']})
outer.observe_query(query=query,handoff=handoff,observer=observer,guards=guards,owner=owner,
 accounting=accounting,acl=acl,config_path=c['config'],config_sha256=c['config_sha'],
 operation_sha256=c['operation_sha'],interpreter=sys._base_executable,interpreter_sha256=c['interpreter_sha'],
 query_source_sha256=c['source_sha'],source_check=lambda:True,leases=leases,environment=env,timeout=20)
''', encoding='utf-8')
    process = subprocess.Popen([sys._base_executable, '-I', '-S', '-B', str(host), str(host_config)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    handle = None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel.TerminateProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    try:
        deadline = time.monotonic() + 10
        while not child_pid_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_path.exists(), 'fabricated breakaway process did not start'
        child_pid = int(child_pid_path.read_text(encoding='ascii'))
        handle = kernel.OpenProcess(0x00100000 | 0x1000 | 0x0001, False, child_pid)
        assert handle and kernel.WaitForSingleObject(handle, 0) == 0x102
        time.sleep(1.2)
        assert process.poll() is None
        assert (prepared.setup['root'] / 'pool-query-invocation.json').exists()
        assert not (prepared.setup['root'] / 'pool-query-outer/observation.json').exists()
        with pytest.raises(OSError):
            with wrapper.open('ab'):
                pass
    finally:
        if handle:
            kernel.TerminateProcess(handle, 1)
            kernel.WaitForSingleObject(handle, 5000)
            kernel.CloseHandle(handle)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
