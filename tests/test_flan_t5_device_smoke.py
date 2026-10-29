"""CPU-only tests for the frozen FLAN-T5 device-development smoke."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from looking_glass.flan_t5_device_observation import Observation


def subject():
    return importlib.import_module('looking_glass.flan_t5_device_smoke')


class SyntheticRetained:
    def __init__(self, observation):
        self.observation = observation
        self.dropped = False
        self.retired = False

    @property
    def value(self):
        return self.observation

    def drop(self):
        self.dropped = True
        self.observation = None

    def retire(self):
        assert self.dropped
        self.retired = True


class SyntheticOwner:
    def __init__(self, *, corrupt_self=False, hide_donor_effect=False):
        self.calls = []
        self.handles = []
        self.corrupt_self, self.hide_donor_effect = corrupt_self, hide_donor_effect

    def observe(self, source, target, *, action, token_index=None, value=None):
        side = source['input_ids'][-2]
        self.calls.append((side, action, token_index))
        before = torch.zeros((1, 8, 4))
        if side == 629:
            before[0, 6] = 1
        after = before.clone()
        if action == 'patch':
            after[0, 6] = value
        logits = torch.zeros((1, 2, 4))
        if side == 629:
            logits.fill_(1)
        if action == 'patch' and bool(torch.any(after[0, 6] != 0)) and not self.hide_donor_effect:
            logits.fill_(1)
        if action == 'patch' and self.corrupt_self and not bool(torch.any(after[0, 6] != 0)):
            logits.fill_(2)
        observation = Observation(logits, after, before, None if action == 'capture' else after,
            {}, tuple(source['input_ids']), tuple(source['attention_mask']), (0, 4675), (1, 1))
        handle = SyntheticRetained(observation)
        self.handles.append(handle)
        return handle


def rows():
    a = {'input_ids': [30355, 15, 1566, 12, 2968, 10, 484, 1], 'attention_mask': [1] * 8}
    b = {'input_ids': [30355, 15, 1566, 12, 2968, 10, 629, 1], 'attention_mask': [1] * 8}
    return a, b, {'input_ids': [4675, 1]}


def test_fixed_sequence_saves_six_outputs_and_all_controls():
    api = subject()
    owner, saved = SyntheticOwner(), []
    result = api._run_sequence(owner, *rows(), save=lambda name, obs: saved.append(name))
    assert saved == ['capture-a', 'capture-b', 'repeat-a', 'self-patch', 'donor-patch', 'null-a']
    assert owner.calls == [(484, 'capture', None), (629, 'capture', None), (484, 'capture', None),
                           (484, 'patch', 6), (484, 'patch', 6), (484, 'null', 6)]
    assert result == {'forwards': 6, 'repeat_exact': True, 'self_patch_exact': True,
                      'null_exact': True, 'donor_differs': True, 'donor_changes_logits': True}
    assert all(handle.retired for handle in owner.handles)


@pytest.mark.parametrize('owner', [SyntheticOwner(corrupt_self=True), SyntheticOwner(hide_donor_effect=True)])
def test_control_or_nonvacuity_failure_does_not_report_success(owner):
    api = subject()
    with pytest.raises(ValueError):
        api._run_sequence(owner, *rows(), save=lambda name, obs: None)


def test_frozen_case_source_and_raw_observation_record(tmp_path):
    api = subject()
    root = Path(__file__).resolve().parents[1]
    a, b, target = api._fixed_rows(root)
    assert (a['input_ids'], b['input_ids'], target['input_ids']) == tuple(row['input_ids'] for row in rows())
    evidence = api.SmokeEvidence(tmp_path, run_id='1' * 32)
    evidence.begin('capture-a')
    synthetic = Observation(torch.zeros((1, 2, 32128)), torch.zeros((1, 8, 512)),
                            torch.zeros((1, 8, 512)), None, {}, tuple(a['input_ids']),
                            tuple(a['attention_mask']), (0, 4675), (1, 1))
    evidence.save('capture-a', synthetic)
    metadata = json.loads((tmp_path / 'capture-a.json').read_bytes())
    for item in metadata['tensors'].values():
        raw = (tmp_path / item['file']).read_bytes()
        assert len(raw) == item['bytes']
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
    assert metadata['source_ids'] == a['input_ids']
    assert metadata['source_mask'] == a['attention_mask']
    assert metadata['decoder_ids'] == [0, 4675]
    with pytest.raises((FileExistsError, RuntimeError)):
        evidence.begin('capture-a')


def test_bootstrap_without_supervisor_cannot_import_torch_or_create_owner_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    prefix = tmp_path / 'empty-prefix'
    prefix.mkdir()
    run = tmp_path / 'run'
    run.mkdir()
    site = Path(torch.__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('MLG_') and not key.upper().startswith('PYTHON')}
    command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}',
               str(root / 'scripts/flan_t5_gpu_bootstrap.py'), '--bootstrap-site', str(site), '--owner',
               '--run-dir', str(run), '--asset-dir', str(tmp_path), '--source-commit', '0' * 40,
               '--git-executable', str(tmp_path / 'git')]
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, timeout=30)
    assert result.returncode != 0
    summary = json.loads(result.stdout)
    assert summary['status'] == 'incomplete' and summary['error_type'] == 'SupervisorError'
    assert summary['model_dependencies_preloaded'] is False
    assert list(run.iterdir()) == []
    assert not any(prefix.iterdir())


@pytest.mark.parametrize('first,second,expected_code,expected_name', [
    (ValueError('owner failed'), KeyboardInterrupt(), 130, 'KeyboardInterrupt'),
    (KeyboardInterrupt(), SystemExit(), 130, 'KeyboardInterrupt'),
    (ValueError('owner failed'), SystemExit(), 1, 'SystemExit'),
])
def test_diagnostic_interruption_preserves_cancellation_and_both_failures(
        monkeypatch, tmp_path, first, second, expected_code, expected_name):
    api = subject()
    session = type('Session', (), {'run_id': '1' * 32, 'device_lease_path': tmp_path / 'lease'})()
    class SessionAPI:
        @staticmethod
        def from_environment(): return session
        @staticmethod
        def check(*args): return None
    class LeaseAPI:
        @staticmethod
        def acquire(*args, **kwargs): return object()
    class Owner:
        def __init__(self, *args, **kwargs): pass
        def start(self): raise first
    class Evidence:
        def __init__(self, root, *, run_id):
            self.run_id, self.phase, self.forward_count, self.memory_count = run_id, 'startup', 0, 0
        def record_stage(self, *args): return None
        def json(self, name, value):
            if name == 'owner-failure.json':
                raise second
    custody = {'entered': False, 'session': None, 'lease': None, 'owner': None,
               'evidence': None, 'failure': None, 'diagnostic_failure': None}
    monkeypatch.setattr(api, '_CUSTODY', custody)
    monkeypatch.setattr(api, 'OwnerSession', SessionAPI)
    monkeypatch.setattr(api, 'DeviceLease', LeaseAPI)
    monkeypatch.setattr(api, 'FlanT5CudaOwner', Owner)
    monkeypatch.setattr(api, 'SmokeEvidence', Evidence)
    monkeypatch.setattr(api, '_attempt_root', lambda *args: tmp_path / 'attempt')
    (tmp_path / 'attempt').mkdir()
    assets = tmp_path / 'assets'
    assets.mkdir()
    code, name = api.owner_main(tmp_path / 'attempt', assets, '0' * 40, tmp_path / 'git')
    assert (code, name) == (expected_code, expected_name)
    assert custody['failure'] is first and custody['diagnostic_failure'] is second


def test_staged_assets_are_counted_separately_from_observation_records(monkeypatch, tmp_path):
    api = subject()
    from looking_glass import flan_t5_feasibility
    raw = b'synthetic asset manifest'
    (tmp_path / 'asset-manifest.json').write_bytes(raw)
    files = {f'file-{index}': {'bytes': index + 1} for index in range(7)}
    monkeypatch.setattr(flan_t5_feasibility, 'verify_assets', lambda path: {'files': files})
    monkeypatch.setattr(api, '_ASSET_HASH', hashlib.sha256(raw).hexdigest())
    assert api._staged_asset_accounting(tmp_path) == {
        'file_count': 7, 'payload_bytes': 28, 'manifest_bytes': len(raw), 'total_bytes': 28 + len(raw)}
    files['file-0']['bytes'] = True
    with pytest.raises(ValueError, match='accounting'):
        api._staged_asset_accounting(tmp_path)
