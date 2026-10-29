"""CPU-only checks for the fixed serial two-client host admission boundary."""
import hashlib
import json
import copy
import gc
import os
from pathlib import Path
import sys
from threading import Barrier, Lock, Thread
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest

from looking_glass import supervised_direct_concurrent as subject
from looking_glass.direct_concurrent_clients import ConcurrentClients, _Slot
from looking_glass.gpu_service import ProfileFiles
from looking_glass.trusted_service_profile import TrustedServiceSource
from looking_glass.two_token_service import _PreparedService


GIB = 1024 ** 3
PROFILE_IDS = ('4' * 64, '5' * 64, '6' * 64)
PLAN = (('alpha', 29, False), ('beta', 29, False))
BINDING = {
    'source_base_sha256': '1' * 64,
    'asset_manifest_sha256': '2' * 64,
    'runtime_settings_sha256': '3' * 64,
    'python_executable_sha256': '7' * 64,
    'profile_ids': PROFILE_IDS,
    'plan': PLAN,
}


def observed_document():
    rows = []
    for index, phase in enumerate(('before_start', 'owner_ready', 'clients_active', 'clients_done',
                                   'owner_exiting', 'owner_reaped')):
        live = phase not in ('before_start', 'owner_reaped')
        rows.append({'timestamp_ns': (index + 1) * 1_000_000_000, 'phase': phase,
            'controller_working_set_bytes': 512 * 1024 ** 2,
            'controller_private_commit_bytes': 512 * 1024 ** 2,
            'owner_working_set_bytes': GIB if live else 0,
            'owner_private_commit_bytes': GIB if live else 0,
            'available_physical_bytes': 40 * GIB, 'available_system_commit_bytes': 30 * GIB,
            'available_caller_commit_bytes': 30 * GIB})
    return {'schema': 'serial-two-client-host-evidence-v1',
        'source_base_sha256': BINDING['source_base_sha256'],
        'asset_manifest_sha256': BINDING['asset_manifest_sha256'],
        'runtime_settings_sha256': BINDING['runtime_settings_sha256'],
        'profile_ids': list(PROFILE_IDS),
        'workload': {'input_ids': [15496, 995], 'shape': [1, 2],
            'max_connections': 2, 'max_pending': 2, 'max_pending_per_team': 1,
            'max_request_bytes': 4096, 'max_response_bytes': 4 * 1024 ** 2,
            'trace_bytes': 0, 'plan': [{'team': team, 'seed': seed, 'cancel': cancel}
                                       for team, seed, cancel in PLAN]},
        'measurement': {'kind': 'supervised-gpu-owner', 'controller_exit': 0,
            'owner_exit': 0, 'supervisor_clean': True, 'supervisor_reaped': True,
            'timed_out': False, 'marker_absent': True, 'clients_reconciled': True,
            'client_statuses': ['succeeded', 'succeeded'], 'model_forward_calls': 3,
            'physical_total_bytes': 64 * GIB,
            'sample_caller': 'controller',
            'processes': {'controller': {'pid': 101, 'created_100ns': 116444736000000001,
                                         'executable_sha256': '7' * 64, 'handle_verified': True},
                          'owner': {'pid': 102, 'created_100ns': 116444736000000002,
                                    'executable_sha256': '7' * 64, 'handle_verified': True}},
            'process_scope': 'controller-and-owner-only; observer-and-helpers-excluded',
            'lifetime_peaks': {'controller_working_set_bytes': 600 * 1024 ** 2,
                'owner_working_set_bytes': 1200 * 1024 ** 2,
                'controller_private_commit_bytes': 600 * 1024 ** 2,
                'owner_private_commit_bytes': 1200 * 1024 ** 2},
            'samples': rows},
        'commitment_bytes': 3 * GIB, 'margin_bytes': GIB, 'reserve_bytes': 2 * GIB}


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def test_matching_approved_real_owner_evidence_is_validated_without_a_gpu():
    raw = canonical(observed_document())
    result = subject._validate_host_evidence(raw, hashlib.sha256(raw).hexdigest(), BINDING)
    assert result['commitment_bytes'] == 3 * GIB
    assert result['reserve_bytes'] == 2 * GIB
    assert result['observed_peak_working_set_bytes'] == 1536 * 1024 ** 2
    assert result['observed_peak_private_commit_bytes'] == 1536 * 1024 ** 2
    assert result['lifetime_peak_working_set_sum_bytes'] == 1800 * 1024 ** 2


@pytest.mark.parametrize('mutation', [
    lambda d: d['workload']['shape'].__setitem__(0, True),
    lambda d: d['workload'].__setitem__('trace_bytes', False),
    lambda d: d['workload'].__setitem__('max_response_bytes', 1024),
    lambda d: d.__setitem__('source_base_sha256', '9' * 64),
    lambda d: d.__setitem__('asset_manifest_sha256', '9' * 64),
    lambda d: d.__setitem__('runtime_settings_sha256', '9' * 64),
    lambda d: d['profile_ids'].__setitem__(0, '9' * 64),
    lambda d: d['measurement'].__setitem__('kind', 'cpu-fake'),
    lambda d: d['measurement'].__setitem__('supervisor_reaped', False),
    lambda d: d['measurement'].__setitem__('controller_exit', True),
    lambda d: d['measurement'].__setitem__('client_statuses', ['succeeded', 'unresolved']),
    lambda d: d['measurement'].__setitem__('model_forward_calls', 2),
    lambda d: d['measurement'].__setitem__('sample_caller', 'observer'),
    lambda d: d['measurement']['processes']['owner'].__setitem__('pid', 101),
    lambda d: d['measurement']['processes']['owner'].__setitem__('created_100ns', 0),
    lambda d: d['measurement']['processes']['owner'].__setitem__('executable_sha256', '8' * 64),
    lambda d: d['measurement']['processes']['owner'].__setitem__('handle_verified', False),
    lambda d: d['measurement']['lifetime_peaks'].__setitem__('owner_private_commit_bytes', 1),
    lambda d: d['measurement']['samples'][1].__setitem__('timestamp_ns', 4_000_000_000),
    lambda d: d['measurement']['samples'][2].__setitem__('owner_private_commit_bytes', False),
    lambda d: d['measurement']['samples'][3].__setitem__('available_physical_bytes', GIB),
    lambda d: d['measurement']['samples'][3].__setitem__('available_system_commit_bytes', GIB),
    lambda d: d['measurement']['samples'][3].__setitem__('available_caller_commit_bytes', GIB),
    lambda d: d.__setitem__('commitment_bytes', 2 * GIB),
    lambda d: d.__setitem__('margin_bytes', 0),
    lambda d: d.__setitem__('reserve_bytes', 0),
])
def test_reapproved_but_inconsistent_or_unmeasured_evidence_is_rejected(mutation):
    document = copy.deepcopy(observed_document())
    mutation(document)
    raw = canonical(document)
    with pytest.raises(subject.ConcurrentServiceError):
        subject._validate_host_evidence(raw, hashlib.sha256(raw).hexdigest(), BINDING)


def test_unapproved_or_noncanonical_evidence_is_rejected():
    raw = canonical(observed_document())
    with pytest.raises(subject.ConcurrentServiceError):
        subject._validate_host_evidence(raw, '0' * 64, BINDING)
    raw += b'\n'
    with pytest.raises(subject.ConcurrentServiceError):
        subject._validate_host_evidence(raw, hashlib.sha256(raw).hexdigest(), BINDING)


def test_fresh_host_screen_requires_physical_and_both_commit_headrooms():
    envelope = {'commitment_bytes': 3 * GIB, 'reserve_bytes': 2 * GIB,
                'physical_total_bytes': 64 * GIB}
    snapshot = {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 8 * GIB,
                'available_system_commit_bytes': 8 * GIB, 'available_caller_commit_bytes': 8 * GIB}
    assert subject._check_host_headroom(snapshot, envelope) == 8 * GIB
    for name in ('available_physical_bytes', 'available_system_commit_bytes', 'available_caller_commit_bytes'):
        insufficient = dict(snapshot, **{name: 8 * GIB - 1})
        with pytest.raises(subject.ConcurrentServiceError):
            subject._check_host_headroom(insufficient, envelope)
    with pytest.raises(subject.ConcurrentServiceError):
        subject._check_host_headroom(dict(snapshot, physical_total_bytes=63 * GIB), envelope)
    with pytest.raises(subject.ConcurrentServiceError):
        subject._check_host_headroom(dict(snapshot, available_system_commit_bytes=True), envelope)


def test_host_screen_charges_the_whole_commitment_plus_separate_reserve():
    envelope = {'commitment_bytes': 7 * GIB, 'reserve_bytes': 2 * GIB,
                'physical_total_bytes': 64 * GIB}
    snapshot = {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 9 * GIB,
                'available_system_commit_bytes': 9 * GIB, 'available_caller_commit_bytes': 9 * GIB}
    assert subject._check_host_headroom(snapshot, envelope) == 9 * GIB
    with pytest.raises(subject.ConcurrentServiceError):
        subject._check_host_headroom(dict(snapshot, available_physical_bytes=9 * GIB - 1), envelope)


@pytest.mark.skipif(os.name != 'nt', reason='the host screen uses Windows memory APIs')
def test_windows_host_screen_returns_current_nonnegative_physical_and_commit_values():
    snapshot = subject._windows_host_snapshot()
    assert set(snapshot) == {'physical_total_bytes', 'available_physical_bytes',
                             'available_system_commit_bytes', 'available_caller_commit_bytes'}
    assert all(type(value) is int and value >= 0 for value in snapshot.values())
    assert 0 < snapshot['available_physical_bytes'] <= snapshot['physical_total_bytes']


def test_host_claim_is_rooted_once_and_retained_for_process_lifetime(monkeypatch):
    snapshot = {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 8 * GIB,
                'available_system_commit_bytes': 8 * GIB, 'available_caller_commit_bytes': 8 * GIB}
    envelope = {'commitment_bytes': 3 * GIB, 'reserve_bytes': 2 * GIB,
                'physical_total_bytes': 64 * GIB}
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_windows_host_snapshot', lambda: snapshot)
    first = subject._HostClaim()
    subject._reserve_host_claim(first, envelope)
    assert subject._HOST_CLAIM_SLOT[0] is first
    assert first.commitment_bytes == 3 * GIB
    assert first.required_headroom_bytes == 8 * GIB
    with pytest.raises(subject.ConcurrentServiceError):
        subject._reserve_host_claim(subject._HostClaim(), envelope)
    assert subject._HOST_CLAIM_SLOT[0] is first


def test_interrupted_claim_return_stays_rooted_and_rejects_another_admission(monkeypatch):
    snapshot = {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 8 * GIB,
                'available_system_commit_bytes': 8 * GIB, 'available_caller_commit_bytes': 8 * GIB}
    envelope = {'commitment_bytes': 3 * GIB, 'reserve_bytes': 2 * GIB,
                'physical_total_bytes': 64 * GIB}
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_windows_host_snapshot', lambda: snapshot)
    claim = subject._HostClaim()
    claim_ref = weakref.ref(claim)
    def interrupt_return(frame, event, arg):
        if frame.f_code is subject._reserve_host_claim.__code__ and event == 'return':
            raise KeyboardInterrupt('interrupted handoff')
        return interrupt_return
    try:
        sys.settrace(interrupt_return)
        with pytest.raises(KeyboardInterrupt, match='interrupted handoff'):
            subject._reserve_host_claim(claim, envelope)
    finally:
        sys.settrace(None)
    del claim
    gc.collect()
    assert subject._HOST_CLAIM_SLOT[0] is claim_ref()
    with pytest.raises(subject.ConcurrentServiceError):
        subject._reserve_host_claim(subject._HostClaim(), envelope)


def test_concurrent_claims_admit_only_one(monkeypatch):
    snapshot = {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 8 * GIB,
                'available_system_commit_bytes': 8 * GIB, 'available_caller_commit_bytes': 8 * GIB}
    envelope = {'commitment_bytes': 3 * GIB, 'reserve_bytes': 2 * GIB,
                'physical_total_bytes': 64 * GIB}
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_windows_host_snapshot', lambda: snapshot)
    barrier, outcomes = Barrier(3), []
    def attempt():
        claim = subject._HostClaim()
        barrier.wait()
        try:
            subject._reserve_host_claim(claim, envelope)
        except subject.ConcurrentServiceError:
            outcomes.append('rejected')
        else:
            outcomes.append('admitted')
    threads = [Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcomes) == ['admitted', 'rejected']
    assert subject._HOST_CLAIM_SLOT[0].commitment_bytes == 3 * GIB


def test_unavailable_or_insufficient_host_telemetry_never_creates_a_claim(monkeypatch):
    envelope = {'commitment_bytes': 3 * GIB, 'reserve_bytes': 2 * GIB,
                'physical_total_bytes': 64 * GIB}
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    def unavailable():
        raise subject.ConcurrentServiceError('unavailable')
    monkeypatch.setattr(subject, '_windows_host_snapshot', unavailable)
    with pytest.raises(subject.ConcurrentServiceError):
        subject._reserve_host_claim(subject._HostClaim(), envelope)
    assert subject._HOST_CLAIM_SLOT == [None]


def prepared_gate_fixture(tmp_path):
    service = object.__new__(_PreparedService)
    service._repo = str(tmp_path)
    source = canonical({'schema': 'trusted-service-source-base-v1',
                        'python_executable': {'size': 1, 'sha256': '7' * 64},
                        'repository_files': [], 'assets': []})
    service._source = TrustedServiceSource(str(tmp_path), str(tmp_path), source, b'assets', ())
    service._profiles = tuple(ProfileFiles(Path(tmp_path) / f'p{i}', Path(tmp_path) / f'e{i}',
                                           Path(tmp_path) / f'm{i}', pin)
                              for i, pin in enumerate(PROFILE_IDS))
    service._identity = SimpleNamespace(marker_path=tmp_path / 'unsafe')
    service._controller = SimpleNamespace(_lock=Lock(), _claimed=True, shutdown=Mock(), wait=Mock())
    service._launch = object()
    service._teams = ('alpha', 'beta')
    service.start = Mock(side_effect=RuntimeError('simulated startup failure'))
    clients = object.__new__(ConcurrentClients)
    clients._service, clients._started = service, False
    clients._slots = (_Slot('alpha', 29, False), _Slot('beta', 29, False))
    document = observed_document()
    document['source_base_sha256'] = hashlib.sha256(source).hexdigest()
    document['asset_manifest_sha256'] = hashlib.sha256(b'assets').hexdigest()
    raw = canonical(document)
    return service, clients, raw


def test_admitted_run_retains_host_claim_after_owner_start_failure(monkeypatch, tmp_path):
    service, clients, raw = prepared_gate_fixture(tmp_path)
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_canonical_methods', lambda *_: True)
    keys = tuple(SimpleNamespace(runtime_settings_sha256='3' * 64) for _ in range(3))
    monkeypatch.setattr(subject, '_load_profile_registry', lambda *_: (None, keys), raising=False)
    monkeypatch.setattr(subject, '_windows_host_snapshot', lambda: {
        'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 8 * GIB,
        'available_system_commit_bytes': 8 * GIB, 'available_caller_commit_bytes': 8 * GIB})
    evidence = subject.ApprovedHostEvidence(raw, hashlib.sha256(raw).hexdigest())
    run = subject.SupervisedConcurrentRun(service, clients, host_evidence=evidence)
    with pytest.raises(RuntimeError, match='simulated startup failure'):
        run.run()
    assert subject._HOST_CLAIM_SLOT[0] is run._host_claim
    assert run._host_claim.commitment_bytes == 3 * GIB
    assert service.start.call_count == 1
    retained = (weakref.ref(run), weakref.ref(service), weakref.ref(clients))
    del run, service, clients
    gc.collect()
    assert all(reference() is not None for reference in retained)


def test_wrong_profile_runtime_blocks_owner_before_host_claim(monkeypatch, tmp_path):
    service, clients, raw = prepared_gate_fixture(tmp_path)
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_canonical_methods', lambda *_: True)
    keys = tuple(SimpleNamespace(runtime_settings_sha256='9' * 64) for _ in range(3))
    monkeypatch.setattr(subject, '_load_profile_registry', lambda *_: (None, keys), raising=False)
    evidence = subject.ApprovedHostEvidence(raw, hashlib.sha256(raw).hexdigest())
    run = subject.SupervisedConcurrentRun(service, clients, host_evidence=evidence)
    with pytest.raises(subject.ConcurrentServiceError):
        run.run()
    assert subject._HOST_CLAIM_SLOT == [None]
    service.start.assert_not_called()


def test_evidence_removed_during_host_check_blocks_owner_and_retains_claim(monkeypatch, tmp_path):
    service, clients, raw = prepared_gate_fixture(tmp_path)
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_canonical_methods', lambda *_: True)
    keys = tuple(SimpleNamespace(runtime_settings_sha256='3' * 64) for _ in range(3))
    monkeypatch.setattr(subject, '_load_profile_registry', lambda *_: (None, keys))
    evidence = subject.ApprovedHostEvidence(raw, hashlib.sha256(raw).hexdigest())
    run = subject.SupervisedConcurrentRun(service, clients, host_evidence=evidence)
    def snapshot():
        run._host_evidence = None
        return {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 8 * GIB,
                'available_system_commit_bytes': 8 * GIB, 'available_caller_commit_bytes': 8 * GIB}
    monkeypatch.setattr(subject, '_windows_host_snapshot', snapshot)
    with pytest.raises(subject.ConcurrentServiceError, match='binding'):
        run.run()
    service.start.assert_not_called()
    assert subject._HOST_CLAIM_SLOT[0] is run._host_claim


@pytest.mark.parametrize('mutate', [
    lambda clients: setattr(clients._slots[0], 'seed', 43),
    lambda clients: setattr(clients, '_slots', (_Slot('alpha', 29, False), _Slot('beta', 29, False))),
])
def test_client_plan_change_during_host_check_blocks_owner(monkeypatch, tmp_path, mutate):
    service, clients, raw = prepared_gate_fixture(tmp_path)
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_canonical_methods', lambda *_: True)
    keys = tuple(SimpleNamespace(runtime_settings_sha256='3' * 64) for _ in range(3))
    monkeypatch.setattr(subject, '_load_profile_registry', lambda *_: (None, keys))
    evidence = subject.ApprovedHostEvidence(raw, hashlib.sha256(raw).hexdigest())
    run = subject.SupervisedConcurrentRun(service, clients, host_evidence=evidence)
    def snapshot():
        mutate(clients)
        return {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 8 * GIB,
                'available_system_commit_bytes': 8 * GIB, 'available_caller_commit_bytes': 8 * GIB}
    monkeypatch.setattr(subject, '_windows_host_snapshot', snapshot)
    with pytest.raises(subject.ConcurrentServiceError, match='binding'):
        run.run()
    service.start.assert_not_called()
    assert subject._HOST_CLAIM_SLOT[0] is run._host_claim


def bootstrap_fixture(monkeypatch, tmp_path, *, available=10 * GIB):
    service, clients, _ = prepared_gate_fixture(tmp_path)
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_canonical_methods', lambda *_: True)
    keys = tuple(SimpleNamespace(runtime_settings_sha256='3' * 64) for _ in range(3))
    monkeypatch.setattr(subject, '_load_profile_registry', lambda *_: (None, keys))
    monkeypatch.setattr(subject, '_windows_host_snapshot', lambda: {
        'physical_total_bytes': 64 * GIB, 'available_physical_bytes': available,
        'available_system_commit_bytes': available, 'available_caller_commit_bytes': available})
    return service, clients, subject.SupervisedConcurrentRun(service, clients)


def test_development_bootstrap_has_own_fixed_allowance_and_is_one_shot(monkeypatch, tmp_path):
    service, clients, run = bootstrap_fixture(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match='simulated startup failure'):
        run.run_development_bootstrap()
    assert service.start.call_count == 1
    assert subject._HOST_CLAIM_SLOT[0] is run._host_claim
    assert run._host_claim.commitment_bytes == 8 * GIB
    assert run._host_claim.required_headroom_bytes == 10 * GIB
    assert run._host_claim.admission_mode == 'development-bootstrap'
    with pytest.raises(subject.ConcurrentServiceError, match='one-shot'):
        run.run_development_bootstrap()
    assert service.start.call_count == 1


@pytest.mark.parametrize('problem', ['insufficient', 'missing-profile', 'requested-cancellation'])
def test_development_bootstrap_rejects_before_owner_on_missing_prerequisite(monkeypatch, tmp_path, problem):
    service, clients, run = bootstrap_fixture(monkeypatch, tmp_path,
        available=10 * GIB - 1 if problem == 'insufficient' else 10 * GIB)
    if problem == 'missing-profile':
        def reject_profiles(*_):
            raise ValueError('profiles unavailable')
        monkeypatch.setattr(subject, '_load_profile_registry', reject_profiles)
    if problem == 'requested-cancellation':
        clients._slots[0].cancel = True
    with pytest.raises((subject.ConcurrentServiceError, ValueError)):
        run.run_development_bootstrap()
    assert subject._HOST_CLAIM_SLOT == [None]
    service.start.assert_not_called()


def test_bootstrap_does_not_consume_an_approved_host_record(monkeypatch, tmp_path):
    service, clients, raw = prepared_gate_fixture(tmp_path)
    monkeypatch.setattr(subject, '_HOST_CLAIM_SLOT', [None])
    monkeypatch.setattr(subject, '_canonical_methods', lambda *_: True)
    evidence = subject.ApprovedHostEvidence(raw, hashlib.sha256(raw).hexdigest())
    run = subject.SupervisedConcurrentRun(service, clients, host_evidence=evidence)
    with pytest.raises(subject.ConcurrentServiceError, match='cannot consume'):
        run.run_development_bootstrap()
    assert subject._HOST_CLAIM_SLOT == [None]
    service.start.assert_not_called()


def test_bootstrap_claim_replacement_during_headroom_check_blocks_start(monkeypatch, tmp_path):
    service, clients, run = bootstrap_fixture(monkeypatch, tmp_path)
    original = run._host_claim
    def snapshot():
        run._host_claim = subject._HostClaim(run)
        return {'physical_total_bytes': 64 * GIB, 'available_physical_bytes': 10 * GIB,
                'available_system_commit_bytes': 10 * GIB, 'available_caller_commit_bytes': 10 * GIB}
    monkeypatch.setattr(subject, '_windows_host_snapshot', snapshot)
    with pytest.raises(subject.ConcurrentServiceError, match='binding'):
        run.run_development_bootstrap()
    assert subject._HOST_CLAIM_SLOT[0] is original
    service.start.assert_not_called()


def test_development_bootstrap_success_is_labeled_and_keeps_claim(monkeypatch, tmp_path):
    service, clients, run = bootstrap_fixture(monkeypatch, tmp_path)
    service.start.side_effect = None
    clients.start = Mock()
    clients.wait = Mock(return_value=subject.ConcurrentClientReport(
        (Mock(status='succeeded'), Mock(status='succeeded')), None, None, False))
    run._recovery.shutdown = Mock(return_value=object())
    monkeypatch.setattr(subject, '_clean_supervisor', lambda *_: True)
    monkeypatch.setattr(subject, 'prepare_trusted_source', lambda *_: service._source)
    result = run.run_development_bootstrap()
    assert result.admission_mode == 'development-bootstrap'
    assert result.teardown_verified is True
    assert subject._HOST_CLAIM_SLOT[0] is run._host_claim
