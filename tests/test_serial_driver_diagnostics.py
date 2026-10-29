"""CPU-only failure evidence for the fixed serial profile collector."""
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from looking_glass import live_profile
from looking_glass import trusted_profile_collector as collector
from looking_glass import trusted_service_profile as trusted
from looking_glass.device_lease import DeviceLease
from looking_glass.supervisor import OwnerSession


def test_failed_driver_query_retains_bounded_private_details_and_never_attests(tmp_path, monkeypatch):
    owner = SimpleNamespace(run_id='b' * 32, device_lease_path='test-lease',
                            identity=SimpleNamespace(gpu_uuid='GPU-test'))
    lease = object()
    source = SimpleNamespace(source_base=b'base', asset_manifest=b'assets')
    monkeypatch.setattr(collector, '_PROCESS_CUSTODY', [None] * 4)
    monkeypatch.setattr(collector, '_require_gated_supervisor', lambda: None)
    monkeypatch.setattr(collector, '_validated_hf_home', lambda *_: None)
    monkeypatch.setattr(collector, '_require_workspace_spec', lambda *_: None)
    monkeypatch.setattr(collector, '_require_workspace_binding', lambda *_: None)
    monkeypatch.setattr(OwnerSession, 'from_environment', classmethod(lambda cls: owner))
    monkeypatch.setattr(OwnerSession, 'check', lambda *_: None)
    monkeypatch.setattr(OwnerSession, 'attest_clean', lambda *_: (_ for _ in ()).throw(AssertionError('attested')))
    monkeypatch.setattr(DeviceLease, 'acquire', classmethod(lambda cls, *a, **kw: lease))
    monkeypatch.setattr(trusted, 'prepare_trusted_source', lambda *_: source)
    monkeypatch.setenv('PATH', 'controlled-system-directory')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-private')
    monkeypatch.setenv('SERIAL_DIAGNOSTIC_SECRET', 'must-not-be-saved')
    command = ['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits']
    failed = subprocess.CalledProcessError(255, command, output='out-' + 'x' * 9000,
                                           stderr='driver-error-' + 'y' * 9000)
    monkeypatch.setattr(live_profile.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(failed))
    monkeypatch.setattr(collector, 'collect_owned', lambda *_: live_profile._driver_version('GPU-test'))
    config = {'run_dir': str(tmp_path), 'run_id': 'a' * 32, 'repo_root': 'repo', 'cache_root': 'cache',
              'source_base_sha256': hashlib.sha256(b'base').hexdigest(),
              'asset_sha256': hashlib.sha256(b'assets').hexdigest()}

    assert collector._owner_main(config) == 1
    failure = json.loads((tmp_path / 'failure.json').read_text())
    diagnostic = failure['private_driver_query']
    assert failure['error_type'] == 'ValueError'
    assert diagnostic['command'] == command
    assert diagnostic['returncode'] == 255
    assert diagnostic['stderr']['text'].startswith('driver-error-')
    assert len(diagnostic['stderr']['text'].encode('utf-8')) <= 4096
    assert diagnostic['stderr']['truncated'] is True
    assert diagnostic['stdout']['truncated'] is True
    assert diagnostic['environment']['PATH']['text'] == 'controlled-system-directory'
    assert diagnostic['environment']['CUDA_VISIBLE_DEVICES']['text'] == 'GPU-private'
    assert 'must-not-be-saved' not in json.dumps(diagnostic)
    assert not (tmp_path / 'candidate.json').exists()


def test_driver_timeout_preserves_bounded_binary_output(monkeypatch):
    command = ['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits']
    failed = subprocess.TimeoutExpired(command, 10, output=b'partial', stderr=b'\xff' * 9000)
    monkeypatch.setattr(live_profile.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(failed))
    with pytest.raises(ValueError, match='driver inventory is unavailable') as caught:
        live_profile._driver_version('GPU-test')
    diagnostic = collector._private_driver_failure(caught.value)
    assert diagnostic['failure_type'] == 'TimeoutExpired'
    assert diagnostic['stdout']['text'] == 'partial'
    assert diagnostic['stderr']['truncated'] is True
    assert len(diagnostic['stderr']['text'].encode('utf-8')) <= 4096


def test_unrelated_subprocess_failure_does_not_produce_driver_diagnostic():
    try:
        raise ValueError('unrelated') from subprocess.CalledProcessError(1, ['other'], stderr='private')
    except ValueError as error:
        assert collector._private_driver_failure(error) is None
