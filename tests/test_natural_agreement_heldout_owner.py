"""Fabricated-only checks for the private held-out owner supervisor."""
from collections import Counter
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tools import natural_agreement_heldout_owner as owner


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def fake_evidence():
    aggregate, inventory, nonce = b'fabricated-aggregate', b'fabricated-inventory', b'n' * 32
    selection = {'selected_site': '1:attn:last', 'earliest_site': '1:attn:last',
                 'random_site': '2:mlp:last', 'magnitude_site': '4:mlp:last'}
    prediction = encoded({'discovery_event_sha256': 'a' * 64, 'independent_audit_sha256': 'b' * 64,
                          'selection': selection})
    families = []
    for family in range(4):
        cases = [{'input_ids': [family, item, 1], 'attention_mask': [1, 1, 1], 'subject_index': 1}
                 for item in range(4)]
        families.append({'family_index': family, 'cases': cases})
    record = encoded({'families': families, 'prediction_freeze_sha256': hashlib.sha256(aggregate).hexdigest(),
                      'exposure_inventory_sha256': hashlib.sha256(inventory).hexdigest()})
    domain = b'fabricated-domain\x00'
    commitment_sha = hashlib.sha256(domain + nonce + b'\x00' + record).hexdigest()
    commitment = encoded({'heldout_commitment': commitment_sha})
    binding = {'runtime': {'fabricated': True}}
    binding_bytes = encoded(binding)
    sites = [f'{layer}:{component}:last' for layer in range(1, 5) for component in ('attn', 'mlp')]
    sites.extend(('5:block:last', '5:block:subject'))
    contract = {'kind': 'run_contract', 'phase': 'heldout', 'prediction_sha256': hashlib.sha256(prediction).hexdigest(),
                'prediction_freeze_sha256': hashlib.sha256(aggregate).hexdigest(),
                'heldout_commitment': commitment_sha, 'exposure_inventory_sha256': hashlib.sha256(inventory).hexdigest(),
                'discovery_event_sha256': 'a' * 64, 'independent_audit_sha256': 'b' * 64,
                'cpu_rng_seed': 20261028, 'cpu_threads': 1, 'deterministic_algorithms': True,
                'sites': [{'id': site} for site in sites], 'cases': owner._expected_cases(json.loads(record))}
    evaluation = {**selection, **{key: contract[key] for key in
                 ('prediction_sha256', 'prediction_freeze_sha256', 'heldout_commitment',
                  'exposure_inventory_sha256', 'discovery_event_sha256', 'independent_audit_sha256')}}
    summary = {'kind': 'summary', 'result': {'status': 'complete', 'forwards': 608,
               'controls': {'null': 16, 'self': 128, 'delivery': 16, 'subject_position': 16,
                            'repeat': 16, 'final_clean': 16},
               'model_state_unchanged': True, 'model_state_sha256': 'c' * 64,
               'heldout_evaluation': evaluation}}
    events = []
    for kind, quartet, target, site, donor in owner._declared_order(contract['cases']):
        event = {'kind': kind}
        if kind in ('clean_capture', 'clean_repeat', 'final_clean'):
            event.update(quartet=quartet, case=target)
        elif kind in ('null', 'delivery', 'subject_position'):
            event.update(quartet=quartet, target=target)
        elif kind == 'patch':
            event.update(quartet=quartet, target=target, site={'id': site}, donor=donor)
        elif kind in ('present_site_metrics', 'secondary_past_site_metrics'):
            event['site'] = site
        events.append(contract if kind == 'run_contract' else summary if kind == 'summary' else event)
    audit = {'events': 626, 'retained_tensor_bytes': 8, 'tensor_sha256': ['d' * 64],
             '_started': {'binding': binding}, '_terminal': {'detail': {'forwards': 608}, 'event_log_bytes': 100},
             '_records': [{'event': event} for event in events]}
    class Control:
        def audit_storage(self, directory):
            return copy.deepcopy(audit)
        def check_start(self, actual, expected):
            assert actual['binding'] == expected['binding']
        def check_execution_configuration(self, actual, runtime):
            assert runtime == binding['runtime']
        @staticmethod
        def _same_json(a, b):
            return encoded(a) == encoded(b)
        @staticmethod
        def _site_metadata(site):
            return {'id': site}
    args = {'control': Control(), 'binding': binding, 'binding_bytes': binding_bytes,
            'prediction_bytes': prediction, 'aggregate_bytes': aggregate, 'inventory_bytes': inventory,
            'record_bytes': record, 'nonce': nonce, 'commitment_bytes': commitment,
            'commitment_domain': domain, 'expected_model_sha256': 'c' * 64}
    return args, audit


def test_structural_audit_binds_committed_rows_and_hides_private_values(tmp_path):
    args, _ = fake_evidence()
    result = owner.audit_saved(tmp_path, 0, True, **args)
    assert result['status'] == 'evidence_complete'
    assert result['inventory']['retained_tensor_bytes'] == 8
    assert 'pending' in result['scientific_validation']
    assert 'fabricated-inventory' not in json.dumps(result)
    assert '_records' not in result['inventory']


@pytest.mark.parametrize('mutation', ['case', 'count', 'summary', 'terminal', 'commitment', 'inventory'])
def test_structural_audit_rejects_inconsistent_evidence(tmp_path, mutation):
    args, audit = fake_evidence()
    if mutation == 'case':
        audit['_records'][0]['event']['cases'][0]['input_ids'][0] = 99
    elif mutation == 'count':
        audit['_records'][1]['event']['kind'] = 'wrong'
    elif mutation == 'summary':
        audit['_records'][-1]['event']['result']['forwards'] = 608.0
    elif mutation == 'terminal':
        audit['_terminal']['detail']['forwards'] = 608.0
    elif mutation == 'commitment':
        args['nonce'] = b'x' * 32
    else:
        args['inventory_bytes'] = b'wrong'
    with pytest.raises(owner.OwnerError):
        owner.audit_saved(tmp_path, 0, True, **args)


@pytest.fixture(autouse=True)
def isolated_quarantine(monkeypatch):
    monkeypatch.setattr(owner, '_QUARANTINED', [])


class Lease:
    def __init__(self, name, events, *, close_fails=False):
        self.name, self.events, self.close_fails, self.active = name, events, close_fails, False
    def __enter__(self):
        self.events.append('enter-' + self.name)
        self.active = True
        return self
    def close(self):
        self.events.append('close-' + self.name)
        if self.close_fails:
            raise OSError('unknown close outcome')
        self.active = False


def test_leases_and_inherited_guard_stay_active_through_verified_cleanup():
    events = []
    frozen, operational, inherited = (Lease(name, events) for name in ('frozen', 'operational', 'inherited'))
    def precheck():
        assert frozen.active and operational.active
    def launch(acquired):
        assert frozen.active and operational.active
        acquired.append(inherited)
        inherited.__enter__()
        return {'cleanup_verified': True}
    def postcheck(result):
        assert frozen.active and operational.active and inherited.active
        return 'audited'
    assert owner.under_source_leases([frozen, operational], precheck, launch, postcheck) == 'audited'
    assert events[-3:] == ['close-inherited', 'close-operational', 'close-frozen']
    assert not owner._QUARANTINED


@pytest.mark.parametrize('mode', ['unreturned_process', 'unknown_drain'])
def test_uncertain_spawn_or_cleanup_quarantines_sources_and_blocks_restart(mode):
    events = []
    frozen, operational, inherited = (Lease(name, events) for name in ('frozen', 'operational', 'inherited'))
    def launch(acquired):
        acquired.append(inherited)
        inherited.__enter__()
        if mode == 'unreturned_process':
            raise OSError('simulated unreturned child')
        return {'cleanup_verified': False}
    with pytest.raises((OSError, owner.OwnerError)):
        owner.under_source_leases([frozen, operational], lambda: None, launch, lambda _: None)
    assert frozen.active and operational.active and inherited.active and len(owner._QUARANTINED) == 1
    with pytest.raises(owner.OwnerError):
        owner.under_source_leases([], lambda: None, launch, lambda _: None)


def test_unknown_duplicate_close_keeps_remaining_protection_rooted():
    events = []
    frozen, operational = Lease('frozen', events), Lease('operational', events)
    inherited = Lease('inherited', events, close_fails=True)
    def launch(acquired):
        acquired.append(inherited)
        inherited.__enter__()
        return {'cleanup_verified': True}
    with pytest.raises(OSError):
        owner.under_source_leases([frozen, operational], lambda: None, launch, lambda _: None)
    assert frozen.active and operational.active and inherited.active and owner._QUARANTINED


def test_primary_precheck_error_survives_failed_lease_close():
    events = []
    frozen = Lease('frozen', events, close_fails=True)
    def precheck():
        raise ValueError('primary failure')
    with pytest.raises(ValueError, match='primary failure') as error:
        owner.under_source_leases([frozen], precheck, lambda _: None, lambda _: None)
    assert isinstance(error.value.__cause__, OSError) and owner._QUARANTINED


def test_fresh_cancellation_during_lease_close_takes_precedence():
    class InterruptedLease(Lease):
        def close(self):
            raise KeyboardInterrupt()
    lease = InterruptedLease('frozen', [])
    def precheck():
        raise ValueError('primary failure')
    with pytest.raises(KeyboardInterrupt):
        owner.under_source_leases([lease], precheck, lambda _: None, lambda _: None)
    assert owner._QUARANTINED


def test_system_exit_zero_during_execution_cannot_report_success(monkeypatch, capsys):
    monkeypatch.setattr(owner, 'execute', lambda _: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(owner.sys, 'argv', ['owner', 'config'])
    with pytest.raises(SystemExit) as error:
        owner.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().err)['status'] == 'incomplete'


def test_quarantine_hold_survives_repeated_cancellation(monkeypatch):
    owner._QUARANTINED.append({'fabricated': True})
    calls = []
    class FakeEvent:
        def wait(self):
            calls.append('wait')
            if len(calls) == 1:
                raise KeyboardInterrupt()
            if len(calls) == 2:
                raise SystemExit(0)
            owner._QUARANTINED.clear()
    monkeypatch.setattr(owner, '_QUARANTINE_EVENT', FakeEvent())
    owner._hold_quarantine()
    assert len(calls) == 3


def test_custody_latch_is_rooted_before_any_lease_or_spawn(monkeypatch):
    class NoAllocation(list):
        def append(self, value):
            raise MemoryError('fabricated prelaunch allocation failure')
    monkeypatch.setattr(owner, '_QUARANTINED', NoAllocation())
    calls = []
    with pytest.raises(MemoryError):
        owner.under_source_leases([Lease('source', calls)], lambda: None,
                                  lambda _: calls.append('spawn'), lambda _: None)
    assert calls == []


def test_console_interrupts_are_ignored_during_execution(monkeypatch):
    observed = []
    def execute(_):
        observed.append(owner.signal.getsignal(owner.signal.SIGINT))
        return {'status': 'evidence_complete', 'scientific_validation': 'pending'}
    monkeypatch.setattr(owner, 'execute', execute)
    monkeypatch.setattr(owner.sys, 'argv', ['owner', 'config'])
    original = owner.signal.getsignal(owner.signal.SIGINT)
    with pytest.raises(SystemExit) as error:
        owner.main()
    assert error.value.code == 0 and observed == [owner.signal.SIG_IGN]
    assert owner.signal.getsignal(owner.signal.SIGINT) == original


@pytest.mark.parametrize('optimization', [None, '-O', '-OO'])
def test_documented_bootstrap_rejects_wrong_source_under_optimization(tmp_path, optimization):
    documentation = (Path(__file__).resolve().parents[1] / 'docs/NATURAL_AGREEMENT_HELDOUT_OWNER.md').read_text()
    bootstrap = documentation.split('```python\n', 1)[1].split('\n```', 1)[0]
    marker = tmp_path / 'executed'
    source = tmp_path / 'untrusted.py'
    source.write_text(f"from pathlib import Path; Path({str(marker)!r}).write_text('executed')")
    command = [sys._base_executable, '-I', '-S', '-B']
    if optimization:
        command.append(optimization)
    command.extend(['-c', bootstrap, str(source), '0' * 64, str(tmp_path / 'config')])
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode != 0 and not marker.exists()


def test_unpublished_commitment_stops_before_owner_spawn(tmp_path, monkeypatch):
    from tools import natural_agreement_heldout_child as child
    phase = tmp_path / 'phase'
    phase.mkdir()
    operation = b'fabricated-operation'
    (phase / 'commitment.json').write_bytes(encoded({'operational_binding_sha256': hashlib.sha256(operation).hexdigest()}))
    config = {'phase_dir': str(phase), 'public_urls': {'operational_binding_url': 'operation',
                                                       'commitment_url': 'commitment'}}
    def published(url, suffix, expected):
        if url == 'commitment':
            raise child.BootstrapError('fabricated unpublished commitment')
    monkeypatch.setattr(child, '_check_published', published)
    calls = []
    with pytest.raises(child.BootstrapError):
        owner.under_source_leases([], lambda: owner._verify_public_commitment(child, config, operation),
                                  lambda _: calls.append('spawn'), lambda _: None)
    assert calls == [] and not owner._QUARANTINED


def test_broken_diagnostic_cannot_skip_the_quarantine_hold(monkeypatch):
    class Broken:
        def write(self, value):
            raise OSError('fabricated broken diagnostic')
        def flush(self):
            raise OSError('fabricated broken diagnostic')
    def execute(_):
        owner._QUARANTINED.append({'fabricated': True})
        raise RuntimeError('unverified cleanup')
    calls = []
    def hold():
        calls.append('holding')
        owner._QUARANTINED.clear()
    monkeypatch.setattr(owner, 'execute', execute)
    monkeypatch.setattr(owner, '_hold_quarantine', hold)
    monkeypatch.setattr(owner.sys, 'argv', ['owner', 'config'])
    monkeypatch.setattr(owner.sys, 'stderr', Broken())
    with pytest.raises(SystemExit) as error:
        owner.main()
    assert error.value.code == 1 and calls == ['holding']


@pytest.mark.skipif(os.name != 'nt', reason='Windows inherited source handles')
def test_inherited_source_protection_survives_parent_death_until_child_exit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source_root = tmp_path / 'protected'
    source_root.mkdir()
    source = source_root / 'source.py'
    source.write_text('original')
    ready, release = tmp_path / 'ready', tmp_path / 'release'
    harness = tmp_path / 'harness.py'
    harness.write_text('''import os,pathlib,subprocess,sys,time
if sys.argv[1] == 'child':
    release=pathlib.Path(sys.argv[2])
    while not release.exists(): time.sleep(0.02)
    raise SystemExit(0)
sys.path.insert(0,sys.argv[2])
from scripts.v2_source_guards import SourceFileLease, InheritableHandles
lease=SourceFileLease(pathlib.Path(sys.argv[3]),[pathlib.Path(sys.argv[4])])
lease.__enter__()
holder=InheritableHandles(lease.handles())
handles=holder.__enter__()
startup=subprocess.STARTUPINFO()
startup.lpAttributeList={'handle_list':list(handles)}
child=subprocess.Popen([sys._base_executable,'-I','-S','-B',__file__,'child',sys.argv[6]],
    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    startupinfo=startup,close_fds=True,creationflags=subprocess.DETACHED_PROCESS)
pathlib.Path(sys.argv[5]).write_text(str(child.pid))
time.sleep(60)
''')
    parent = subprocess.Popen([sys._base_executable, '-I', '-S', '-B', str(harness), 'parent', str(root),
                               str(source_root), str(source), str(ready), str(release)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              creationflags=subprocess.DETACHED_PROCESS)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    child_handle = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), 'fabricated parent did not launch the child'
        child_handle = kernel.OpenProcess(0x00100000 | 0x1000 | 1, False, int(ready.read_text()))
        assert child_handle
        with pytest.raises(PermissionError):
            source.write_text('blocked while parent and child are alive')
        with pytest.raises(PermissionError):
            os.replace(source, source_root / 'renamed.py')
        parent.kill()
        parent.wait(timeout=10)
        with pytest.raises(PermissionError):
            source.write_text('blocked by the surviving child')
        with pytest.raises(PermissionError):
            os.replace(source, source_root / 'renamed.py')
        release.write_text('exit')
        assert kernel.WaitForSingleObject(child_handle, 10000) == 0
        source.write_text('released')
        os.replace(source, source_root / 'renamed.py')
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        if child_handle:
            if kernel.WaitForSingleObject(child_handle, 0) != 0:
                kernel.TerminateProcess(child_handle, 1)
                kernel.WaitForSingleObject(child_handle, 10000)
            kernel.CloseHandle(child_handle)


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job and source handles')
def test_real_gated_supervisor_keeps_both_source_leases_through_audit(tmp_path):
    from scripts.v2_source_guards import InheritableHandles, SourceFileLease
    from looking_glass.natural_heldout_supervisor import supervise_heldout
    roots = [tmp_path / 'core', tmp_path / 'operational']
    for directory in roots:
        directory.mkdir()
        (directory / 'source.py').write_text('original')
    leases = [SourceFileLease(directory, [directory / 'source.py']) for directory in roots]
    output = tmp_path / 'run'
    root = Path(__file__).resolve().parents[1]
    code = ("import pathlib,sys; sys.path.insert(0,sys.argv[1]); "
            "from looking_glass.natural_agreement_controller import await_start; await_start(); "
            "d=pathlib.Path(sys.argv[2])/'attempt'/'tensors'; d.mkdir(parents=True); "
            "(d/'payload').write_bytes(b'partial')")
    command = [sys._base_executable, '-I', '-c', code, str(root), str(output)]
    def protected():
        for directory in roots:
            with pytest.raises(PermissionError):
                (directory / 'source.py').write_text('blocked')
    def callback(attempt, code, reaped):
        protected()
        assert code == 0 and reaped is True and (attempt / 'tensors/payload').read_bytes() == b'partial'
        return {'status': 'evidence_complete', 'inventory': {'retained_tensor_bytes': 7}}
    def launch(acquired):
        duplicates = InheritableHandles(leases[0].handles() + leases[1].handles())
        acquired.append(duplicates)
        handles = duplicates.__enter__()
        return supervise_heldout(command, output, os.environ.copy(), audit_result=callback,
                                 inherited_handles=handles)
    def postcheck(result):
        protected()
        assert result['child_reaped'] and result['job_drained'] and result['job_closed']
        return result
    result = owner.under_source_leases(leases, protected, launch, postcheck)
    assert result['status'] == 'evidence_complete'
    for directory in roots:
        (directory / 'source.py').write_text('released')
