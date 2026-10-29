"""CPU-only supervision and saved-observation failure boundaries."""

import json
from types import SimpleNamespace

import pytest

from looking_glass import pretrained_gpu_trace_pilot as owner


def test_raw_directory_must_be_fresh_before_owner_work(tmp_path):
    with pytest.raises(ValueError, match='raw evidence'):
        owner._validate_raw_directory(tmp_path)
    raw = tmp_path / 'raw'
    raw.mkdir()
    owner._validate_raw_directory(tmp_path)
    (raw / 'unexpected.f32').write_bytes(b'partial')
    with pytest.raises(ValueError, match='raw evidence'):
        owner._validate_raw_directory(tmp_path)


@pytest.mark.parametrize('error', [RuntimeError('failed synchronization'), KeyboardInterrupt()])
def test_failed_or_interrupted_synchronization_keeps_sticky_unverified_state(monkeypatch, error):
    roots, calls = {'sync_pending': False}, []
    monkeypatch.setattr(owner, '_ROOT', roots)
    def guard(_callback, _device):
        calls.append('sync')
        raise error
    with pytest.raises(type(error)):
        owner._synchronize(guard, SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda *_a: None)))
    assert roots['sync_pending'] is True and calls == ['sync']


@pytest.mark.parametrize('drift', ['count', 'uuid', 'total'])
def test_runtime_identity_drift_stops_before_attestation(monkeypatch, drift):
    calls = []
    cuda = SimpleNamespace(device_count=lambda: 2 if drift == 'count' else 1,
        get_device_properties=lambda *_a: SimpleNamespace(uuid='other' if drift == 'uuid' else 'fixed',
                                                          total_memory=9 if drift == 'total' else 8))
    monkeypatch.setattr(owner, '_runtime_uuid', lambda _torch, value: value)
    monkeypatch.setattr(owner.OwnerSession, 'verify_runtime_binding', lambda *_a, **_k: calls.append('verified'))
    with pytest.raises(RuntimeError, match='changed'):
        owner._recheck_runtime_binding(lambda fn, *args: fn(*args), SimpleNamespace(cuda=cuda),
                                       object(), object(), 'fixed', 8)
    assert calls == []


@pytest.mark.parametrize('supervisor_result', ['raises', 'unreaped', 'cancel'])
def test_uncertain_supervisor_never_claims_post_exit_or_queries_gpu(tmp_path, monkeypatch, supervisor_result):
    commit = 'a' * 40
    identity = SimpleNamespace(lock_path=tmp_path / 'gpu.lock', marker_path=tmp_path / 'gpu.unsafe.json')
    seen = []
    monkeypatch.setattr(owner, '_startup', lambda: None)
    monkeypatch.setattr(owner, '_pristine', lambda: None)
    monkeypatch.setattr(owner, '_workspace_root', lambda *_a: tmp_path)
    monkeypatch.setattr(owner, '_cache_path', lambda *_a: tmp_path)
    monkeypatch.setattr(owner, '_capacity', lambda *_a: {'disk_free_bytes': 4 * owner.GIB})
    monkeypatch.setattr(owner, '_source_identity', lambda *_a: (commit, {'source': 'fixed'}))
    monkeypatch.setattr(owner, '_verify_offline_cache', lambda *_a: None)
    monkeypatch.setattr(owner, 'resolve_device_identity', lambda *_a, **_k: identity)
    monkeypatch.setattr(owner, '_smi_snapshot', lambda _smi: (seen.append('query') or
        ('GPU-fake', {'timestamp_utc': '2026-10-29T12:00:00Z', 'total_bytes': 8 * owner.GIB})))
    def supervise(*_args, **_kwargs):
        identity.marker_path.write_text('possibly-running', encoding='ascii')
        if supervisor_result == 'raises':
            raise RuntimeError('fabricated interruption after grant')
        if supervisor_result == 'cancel':
            raise KeyboardInterrupt()
        return SimpleNamespace(run_id='fake', pid=123, exit_code=None, clean=False,
                               timed_out=False, reason='unreaped')
    monkeypatch.setattr(owner, 'run_supervised', supervise)
    monkeypatch.setenv('PROGRAMFILES', str(tmp_path))
    monkeypatch.setenv('LOOKING_GLASS_TRACE_PILOT_GIT_EXE', str(tmp_path / 'git.exe'))
    monkeypatch.setenv('LOOKING_GLASS_TRACE_PILOT_SOURCE_COMMIT', commit)
    if supervisor_result == 'cancel':
        with pytest.raises(KeyboardInterrupt):
            owner.controller_main(tmp_path, tmp_path, tmp_path / 'attempt', tmp_path, 'fake-smi')
        result = json.loads((tmp_path / 'attempt' / 'outcome.json').read_bytes())
    else:
        result = owner.controller_main(tmp_path, tmp_path, tmp_path / 'attempt', tmp_path, 'fake-smi')
    assert result['status'] == 'incomplete' and seen == ['query']
    assert result['nvidia_smi_after'] is None and result['owner_presence_after_exit']['query'] == 'not_attempted'
    assert result['marker_absent_at_post_exit_point'] is None
    assert result['attestation_absent_at_post_exit_point'] is None
    assert identity.marker_path.read_text(encoding='ascii') == 'possibly-running'
