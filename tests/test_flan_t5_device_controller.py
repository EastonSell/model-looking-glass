"""CPU-only checks of the disabled FLAN-T5 GPU smoke controller."""

import importlib
import hashlib
import json
import os
import struct
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_disabled_controller_stops_before_inventory_or_process_creation(monkeypatch, tmp_path):
    api = importlib.import_module('looking_glass.flan_t5_device_controller')
    monkeypatch.setattr(api, '_smi_snapshot', lambda *args: pytest.fail('device inventory was queried'))
    monkeypatch.setattr(api, 'run_supervised', lambda *args, **kwargs: pytest.fail('owner was launched'))
    with pytest.raises(RuntimeError, match='disabled'):
        api.run_controller(workspace_root=tmp_path, run_dir=tmp_path / 'attempt',
            asset_dir=tmp_path / 'assets', site_packages=tmp_path / 'site',
            git_executable=tmp_path / 'git', smi=tmp_path / 'smi', source_commit='0' * 40)
    assert not (tmp_path / 'attempt').exists()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n', encoding='ascii')


def _reference_metadata(label):
    action = ('capture' if label in ('capture-a', 'capture-b', 'repeat-a')
              else 'null' if label == 'null-a' else 'patch')
    names = ['before'] if action == 'capture' else ['before', 'action', 'after']
    steps = []
    for index, name in enumerate(names):
        selected = name == 'action'
        steps.append({'name': name, 'operation': action if selected else 'capture', 'order': index,
            'module_path': 'encoder.final_layer_norm', 'module_aliases': ['encoder.final_layer_norm'],
            'call_site_known': False, 'forward_index': 0, 'invocation': 0,
            'batch_index': 0 if selected else None, 'token_index': 6 if selected else None,
            'output_selector': [], 'output_shape': [1, 8, 512],
            'selected_shape': [512] if selected else [1, 8, 512], 'dtype': 'torch.float32',
            'device': 'cuda:0', 'source_capture': None, 'preceding_same_site_actions': names[:index]})
    return {'steps': steps, 'capture_bytes': 8 * 512 * 4 * (1 if action == 'capture' else 2),
            'source_bytes': 512 * 4 if action == 'patch' else 0}


@pytest.fixture
def fabricated_owner(tmp_path, monkeypatch):
    from looking_glass import flan_t5_device_controller as api
    from looking_glass.supervisor import SupervisionResult
    root = tmp_path / 'owner'
    root.mkdir()
    stage = root / 'staged-assets'
    stage.mkdir()
    for name in ('asset-manifest.json', 'config.json', 'generation_config.json', 'model.safetensors',
                 'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json'):
        (stage / name).write_bytes(b'x')
    stage_account = {'file_count': 7, 'payload_bytes': 28, 'manifest_bytes': 2, 'total_bytes': 30}
    monkeypatch.setattr(api, '_staged_asset_accounting', lambda path: stage_account)
    run_id, pid, commit = 'a' * 32, 1234, 'b' * 40
    source = {'source.py': {'size': 4, 'sha256': 'c' * 64}}
    gpu_uuid, device_name = 'GPU-00000000-0000-0000-0000-000000000001', 'Synthetic GPU'
    supervision = SupervisionResult(run_id, pid, 0, True, False, 'clean')
    _write_json(root / 'owner-started.json', {'schema': 'flan-t5-gpu-smoke-owner-v1',
        'status': 'started', 'run_id': run_id, 'pid': pid, 'source_commit': commit,
        'case_manifest_sha256': api._CASE_HASH, 'asset_manifest_sha256': api._ASSET_HASH, 'max_forwards': 6})
    total = 8 * 1024 ** 3
    snapshot = {'device_free_bytes': 7 * 1024 ** 3, 'device_total_bytes': total,
        'allocated_bytes': 0, 'reserved_bytes': 0, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
    stages = [('startup', 'before_model_load'), ('startup', 'after_admission'),
        ('model-load', 'before_model_copy'), ('model-load', 'after_model_copy'),
        ('model-load', 'after_initial_state')]
    for label in api._LABELS:
        stages.extend((label, name) for name in ('before_observation', 'output_live', 'captures_live',
                                                 'device_refs_disposed', 'after_observation'))
    stages.extend([('cleanup', 'cleanup_before_workspace_clear'), ('cleanup', 'after_cleanup')])
    for index, (phase, stage) in enumerate(stages):
        _write_json(root / f'memory-{index:03d}.json', {'schema': 'flan-t5-gpu-memory-v1',
            'run_id': run_id, 'index': index, 'phase': phase, 'stage': stage,
            'memory': snapshot, 'peak_scope': 'cumulative-since-pre-model-load-reset'})
    a_state = bytes(8 * 512 * 4)
    b_state = a_state[:6 * 512 * 4] + struct.pack('<f', 1.0) * 512 + a_state[7 * 512 * 4:]
    a_logits = bytes(2 * 32128 * 4)
    donor_logits = struct.pack('<f', 1.0) + a_logits[4:]
    for index, label in enumerate(api._LABELS):
        _write_json(root / f'forward-{index:03d}-started.json', {'schema': 'flan-t5-gpu-forward-v1',
            'run_id': run_id, 'index': index, 'label': label})
        before = b_state if label == 'capture-b' else a_state
        after = b_state if label == 'donor-patch' else before
        values = {'logits': donor_logits if label == 'donor-patch' else a_logits,
                  'state': after, 'before': before}
        if index >= 3:
            values['after'] = after
        tensor_records = {}
        for name, raw in values.items():
            filename = f'{label}-{name}.bin'
            (root / filename).write_bytes(raw)
            tensor_records[name] = {'file': filename, 'bytes': len(raw),
                'shape': [1, 2, 32128] if name == 'logits' else [1, 8, 512],
                'dtype': 'float32-little-endian-bits', 'sha256': hashlib.sha256(raw).hexdigest()}
        _write_json(root / f'{label}.json', {'schema': 'flan-t5-gpu-observation-v1',
            'run_id': run_id, 'label': label, 'source_ids': api._B if label == 'capture-b' else api._A,
            'source_mask': [1] * 8, 'decoder_ids': [0, 4675], 'decoder_mask': [1, 1],
            'metadata': _reference_metadata(label), 'tensors': tensor_records})
    record_bytes = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
    site = tmp_path / 'site'
    site.mkdir()
    deps = {}
    for name in ('torch', 'transformers', 'numpy', 'tokenizers', 'safetensors', 'huggingface_hub'):
        package = site / name
        package.mkdir()
        (package / '__init__.py').write_bytes(b'x')
        deps[name] = {'path': name + '/__init__.py', 'size': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
    _write_json(root / 'owner-final.json', {'schema': 'flan-t5-gpu-smoke-owner-v1',
        'status': 'owner-clean-pending-supervisor', 'run_id': run_id, 'pid': pid, 'source_commit': commit,
        'source_sha256': source, 'case_manifest_sha256': api._CASE_HASH, 'asset_manifest_sha256': api._ASSET_HASH,
        'revision': api._REVISION, 'model_state_sha256': 'e' * 64,
        'runtime_versions': {'python': '3.11.15', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
            'transformers': '4.57.3', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.4.6'},
        'dependency_entry_files': deps, 'device_uuid': gpu_uuid, 'device_name': device_name,
        'device_total_bytes': total, 'allocator_cap_bytes': total // 4, 'cleanup': snapshot,
        'memory_record_count': 37, 'forward_count': 6,
        'controls': {'forwards': 6, 'repeat_exact': True, 'self_patch_exact': True,
            'null_exact': True, 'donor_differs': True, 'donor_changes_logits': True},
        'record_bytes_before_final': record_bytes, 'staged_assets': stage_account})
    kwargs = {'source_commit': commit, 'source_hashes': source, 'gpu_uuid': gpu_uuid,
              'device_name': device_name, 'site_packages': site}
    return api, root, supervision, kwargs


def _recount(root):
    final_path = root / 'owner-final.json'
    final = json.loads(final_path.read_bytes())
    final['record_bytes_before_final'] = sum(path.stat().st_size for path in root.iterdir()
                                             if path.is_file() and path != final_path)
    _write_json(final_path, final)


def test_closed_synthetic_records_require_all_six_forwards_and_cleanup(fabricated_owner):
    api, root, supervision, kwargs = fabricated_owner
    result = api._validate_records(root, supervision, **kwargs)
    assert result['status'] == 'saved-records-consistent-not-approved'
    assert result['forward_count'] == 6 and result['memory_record_count'] == 37
    assert result['device_total_bytes'] == 8 * 1024 ** 3


@pytest.mark.parametrize('mutation', ['bool_pid', 'runtime', 'missing_receipt', 'extra_forward',
    'boolean_coordinate', 'cleanup', 'wrong_donor', 'duplicate_key', 'stage_order', 'dependency_changed'])
def test_rehashed_or_incomplete_records_do_not_validate(fabricated_owner, mutation):
    api, root, supervision, kwargs = fabricated_owner
    final_path = root / 'owner-final.json'
    final = json.loads(final_path.read_bytes())
    if mutation == 'bool_pid':
        final['pid'] = True
    elif mutation == 'runtime':
        final['runtime_versions']['torch'] = '0'
    elif mutation == 'missing_receipt':
        final_path.unlink()
    elif mutation == 'extra_forward':
        _write_json(root / 'forward-006-started.json', {'index': 6})
    elif mutation == 'boolean_coordinate':
        path = root / 'donor-patch.json'
        value = json.loads(path.read_bytes())
        value['metadata']['steps'][1]['token_index'] = True
        _write_json(path, value)
    elif mutation == 'cleanup':
        final['cleanup']['allocated_bytes'] = 1
    elif mutation == 'wrong_donor':
        path = root / 'donor-patch-after.bin'
        raw = bytes(path.stat().st_size)
        path.write_bytes(raw)
        metadata_path = root / 'donor-patch.json'
        value = json.loads(metadata_path.read_bytes())
        value['tensors']['after']['sha256'] = hashlib.sha256(raw).hexdigest()
        _write_json(metadata_path, value)
    elif mutation == 'duplicate_key':
        path = root / 'capture-a.json'
        path.write_bytes(path.read_bytes().replace(b'{', b'{"label":"other",', 1))
    elif mutation == 'stage_order':
        path = root / 'memory-005.json'
        value = json.loads(path.read_bytes())
        value['stage'] = 'output_live'
        _write_json(path, value)
    elif mutation == 'dependency_changed':
        (kwargs['site_packages'] / 'torch' / '__init__.py').write_bytes(b'y')
    if final_path.exists():
        _write_json(final_path, final)
        _recount(root)
    with pytest.raises((ValueError, OSError, KeyError)):
        api._validate_records(root, supervision, **kwargs)


def test_boolean_exit_cannot_stand_in_for_zero(fabricated_owner):
    from looking_glass.supervisor import SupervisionResult
    api, root, supervision, kwargs = fabricated_owner
    forged = SupervisionResult(supervision.run_id, supervision.pid, False, True, False, 'clean')
    with pytest.raises(ValueError, match='supervisor'):
        api._validate_records(root, forged, **kwargs)


def test_device_inventory_rejects_impossible_or_boolean_memory_counters():
    from looking_glass import flan_t5_device_controller as api
    total = 8 * 1024 ** 3
    snapshot = {'device_name': 'Synthetic GPU', 'total_bytes': total, 'free_bytes': total,
                'used_bytes': total, 'driver_version': 'test', 'timestamp_utc': '2026-10-29T00:00:00+00:00'}
    with pytest.raises(ValueError, match='inventory'):
        api._check_inventory(snapshot)
    snapshot.update(free_bytes=7 * 1024 ** 3, used_bytes=True)
    with pytest.raises(ValueError, match='inventory'):
        api._check_inventory(snapshot)
    snapshot.update(used_bytes=1024 ** 3, timestamp_utc='not-a-timestamp')
    with pytest.raises(ValueError, match='inventory'):
        api._check_inventory(snapshot)


def test_runtime_and_inventory_totals_require_matching_rounded_device():
    from looking_glass import flan_t5_device_controller as api
    total, margin = 8 * 1024 ** 3, 2 * 1024 ** 2
    for runtime in (total - margin, total, total + margin):
        assert api._check_device_totals(total, runtime) is None
    for runtime in (True, 0, total - margin - 1, total + margin + 1):
        with pytest.raises(ValueError, match='totals differ'):
            api._check_device_totals(total, runtime)
    with pytest.raises(ValueError, match='totals differ'):
        api._check_device_totals(True, total)


def test_attempt_inventory_counts_partial_checkpoint_separately(tmp_path):
    from looking_glass import flan_t5_device_controller as api
    stage = tmp_path / 'owner' / 'staged-assets'
    stage.mkdir(parents=True)
    (stage / 'model.safetensors').write_bytes(b'partial')
    (tmp_path / 'owner' / 'memory-000.json').write_bytes(b'abc')
    (tmp_path / 'controller-reserved.json').write_bytes(b'{}')
    assert api._inventory_attempt(tmp_path) == {'staged_asset_bytes': 7, 'owner_record_bytes': 3,
        'controller_and_temporary_bytes': 2, 'file_count': 3}


@pytest.mark.parametrize('target', ['owner', 'attempt'])
def test_file_inventory_stops_at_its_bound(tmp_path, monkeypatch, target):
    from looking_glass import flan_t5_device_controller as api
    root = tmp_path / 'owner' if target == 'owner' else tmp_path / 'attempt'
    root.mkdir()
    if target == 'owner':
        stage = root / 'staged-assets'
        stage.mkdir()
        for name in api._STAGED_FILES:
            (stage / name).write_bytes(b'x')
    for index in range(300):
        (root / f'{index:03d}.bin').write_bytes(b'x')
    original = api.os.scandir
    seen = []
    class Counted:
        def __init__(self, path): self.path, self.iterator = Path(path), None
        def __enter__(self):
            self.iterator = original(self.path)
            return self
        def __exit__(self, *args): self.iterator.close()
        def __iter__(self): return self
        def __next__(self):
            item = next(self.iterator)
            if self.path == root:
                seen.append(item.name)
            return item
    monkeypatch.setattr(api.os, 'scandir', Counted)
    with pytest.raises(ValueError, match='bound'):
        (api._owner_evidence_files if target == 'owner' else api._inventory_attempt)(root)
    assert len(seen) <= 257


def test_evidence_manifest_binds_complete_records_and_staged_assets(fabricated_owner):
    api, root, _, _ = fabricated_owner
    files = api._owner_evidence_files(root)
    manifest = api._evidence_manifest(root, files)
    for path in files:
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        assert manifest[relative] == {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert 'owner-final.json' in manifest and 'staged-assets/model.safetensors' in manifest
    (root / 'staged-assets' / 'unexpected').write_bytes(b'x')
    with pytest.raises(ValueError, match='staged'):
        api._owner_evidence_files(root)


def test_guard_capability_loads_only_the_verified_source_bytes(tmp_path):
    from looking_glass import flan_t5_device_controller as api
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    path = scripts / 'v2_source_guards.py'
    raw = b'class SourceFileLease: pass\nclass WindowsJob: pass\n'
    path.write_bytes(raw)
    hashes = {'scripts/v2_source_guards.py': {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
    guard, job = api._guard_types(tmp_path, hashes)
    assert guard.__name__ == 'SourceFileLease' and job.__name__ == 'WindowsJob'
    path.write_bytes(raw + b'# changed\n')
    with pytest.raises(ValueError, match='guard bytes'):
        api._guard_types(tmp_path, hashes)


@pytest.mark.skipif(os.name != 'nt', reason='the source protection is a Windows capability')
def test_verified_source_guard_holds_and_releases_real_file_protection(tmp_path):
    from looking_glass import flan_t5_device_controller as api
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'scripts/v2_source_guards.py'
    raw = source.read_bytes()
    guard_type, _ = api._guard_types(repo, {'scripts/v2_source_guards.py': {
        'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}})
    target = tmp_path / 'fixed.py'
    target.write_bytes(b'original')
    guard = guard_type(tmp_path, (target,))
    try:
        assert guard.__enter__() is guard and guard.handles()
        with pytest.raises(PermissionError):
            target.write_bytes(b'changed')
    finally:
        guard.close()
    target.write_bytes(b'changed')
    assert target.read_bytes() == b'changed'


@pytest.mark.skipif(os.name != 'nt', reason='the source protection is a Windows capability')
def test_real_evidence_guard_keeps_record_and_staged_file_bytes_stable(fabricated_owner):
    api, root, supervision, kwargs = fabricated_owner
    repo = Path(__file__).resolve().parents[1]
    raw = (repo / 'scripts/v2_source_guards.py').read_bytes()
    guard_type, _ = api._guard_types(repo, {'scripts/v2_source_guards.py': {
        'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}})
    files = api._owner_evidence_files(root)
    guard = guard_type(root, files)
    try:
        assert guard.__enter__() is guard and guard.handles()
        assert api._validate_records(root, supervision, **kwargs)['forward_count'] == 6
        manifest = api._evidence_manifest(root, files)
        assert manifest['staged-assets/model.safetensors']['sha256'] == hashlib.sha256(b'x').hexdigest()
        for target in (root / 'owner-final.json', root / 'staged-assets/model.safetensors'):
            with pytest.raises(PermissionError):
                target.write_bytes(b'changed')
        (root.parent / 'controller-final.json').write_bytes(b'provisional')
        # Windows file sharing pins existing entries, not the set of entries.
        (root / 'new-entry').write_bytes(b'x')
        assert api._owner_evidence_files(root) != files
    finally:
        guard.close()
    (root / 'owner-final.json').write_bytes(b'changed')


def test_isolated_bootstrap_controller_stops_without_importing_model_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    prefix, site = tmp_path / 'prefix', tmp_path / 'site'
    prefix.mkdir()
    site.mkdir()
    marker = tmp_path / 'dependency-imported'
    for name in ('torch', 'transformers', 'numpy'):
        (site / (name + '.py')).write_text(f'open({str(marker)!r}, "w").close()\n', encoding='utf-8')
    attempt = tmp_path / 'attempt'
    command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}',
        str(root / 'scripts/flan_t5_gpu_bootstrap.py'), '--bootstrap-site', str(site), '--controller',
        '--workspace-root', str(tmp_path), '--run-dir', str(attempt), '--asset-dir', str(tmp_path),
        '--git-executable', str(tmp_path / 'git'), '--smi', str(tmp_path / 'smi'), '--source-commit', '0' * 40]
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith('PYTHON') and not key.startswith('MLG_')}
    result = subprocess.run(command, env=environment, cwd=tmp_path, capture_output=True, timeout=30)
    assert result.returncode != 0 and b'smoke is disabled' in result.stderr
    assert not attempt.exists() and not marker.exists() and not any(prefix.iterdir())


def test_attempt_directory_must_not_overlap_the_checkout(tmp_path):
    from looking_glass import flan_t5_device_controller as api
    workspace = tmp_path / 'workspace'
    repo = workspace / 'work' / 'teams' / 'checkout'
    repo.mkdir(parents=True)
    with pytest.raises(ValueError, match='separate'):
        api._new_attempt(workspace, repo, repo / 'nested-attempt')
    target = workspace / 'work' / 'private-attempt'
    assert api._new_attempt(workspace, repo, target) == target
    with pytest.raises(ValueError, match='new'):
        api._new_attempt(workspace, repo, target)


@pytest.mark.parametrize('mode', ['clean', 'timeout', 'interrupted', 'source_changed', 'guard_interrupted',
                                 'inventory_changed', 'inventory_time_reversed', 'tool_changed',
                                 'marker_remains', 'missing_receipt', 'owner_total_mismatch', 'evidence_changed',
                                 'evidence_guard_close', 'closure_write_failure'])
def test_supervised_controller_roots_guards_and_fails_closed(monkeypatch, tmp_path, mode):
    from looking_glass import flan_t5_device_controller as api
    from looking_glass import flan_t5_feasibility
    from looking_glass.supervisor import SupervisionResult
    events = []
    workspace, assets, site = tmp_path / 'workspace', tmp_path / 'assets', tmp_path / 'site'
    for path in (workspace, assets, site):
        path.mkdir()
    (assets / 'asset-manifest.json').write_bytes(b'{}')
    git, smi = tmp_path / 'git.exe', tmp_path / 'nvidia-smi.exe'
    git.write_bytes(b'synthetic git')
    smi.write_bytes(b'synthetic inventory tool')
    attempt = tmp_path / 'attempt'
    monkeypatch.setenv('LOOKING_GLASS_FLAN_T5_BOOTSTRAP_SITE', str(site))
    monkeypatch.setattr(api, 'GPU_SMOKE_ENABLED', True)
    monkeypatch.setattr(api, '_CUSTODY', {'entered': False, 'attempt': None, 'source_guard': None,
        'asset_guard': None, 'evidence_guard': None, 'job': None, 'supervision': None, 'may_have_started': False,
        'failure': None, 'diagnostic_failure': None})
    monkeypatch.setattr(api, '_check_bootstrap', lambda repo: None)
    monkeypatch.setattr(api, '_require_pristine_dependencies', lambda: None)
    def create_attempt(*args):
        attempt.mkdir()
        return attempt
    monkeypatch.setattr(api, '_new_attempt', create_attempt)
    hashes = {'source': {'size': 1, 'sha256': 'a' * 64}}
    source_calls = []
    def verify_source(*args):
        source_calls.append(1)
        if mode == 'source_changed' and len(source_calls) == 3:
            return {'changed': True}
        return hashes
    monkeypatch.setattr(api, '_verify_source_tree', verify_source)
    class Guard:
        def __init__(self, root, files):
            self.root = Path(root)
            events.append(('guard-init', self.root))
        def __enter__(self):
            events.append(('guard-enter', self.root))
            if mode == 'guard_interrupted':
                raise KeyboardInterrupt()
            return self
        def handles(self): return (1,)
        def close(self):
            if self.root == attempt / 'owner':
                assert (attempt / 'controller-final.json').is_file()
                if mode == 'evidence_guard_close':
                    raise OSError('evidence guard close failed')
            events.append(('guard-close', self.root))
    class Job:
        def __init__(self): events.append(('job', None))
    monkeypatch.setattr(api, '_guard_types', lambda *args: (Guard, Job))
    manifest = {'files': {'model.safetensors': {'bytes': 128}}}
    monkeypatch.setattr(flan_t5_feasibility, 'verify_assets', lambda path: manifest)
    monkeypatch.setattr(api, '_fixed_rows', lambda repo: None)
    monkeypatch.setattr(api.shutil, 'disk_usage', lambda path: SimpleNamespace(free=2 * 1024 ** 3))
    gpu = 'GPU-00000000-0000-0000-0000-000000000001'
    snapshot = {'device_name': 'Synthetic GPU', 'total_bytes': 8 * 1024 ** 3,
                'free_bytes': 7 * 1024 ** 3, 'used_bytes': 1024 ** 3,
                'driver_version': 'test', 'timestamp_utc': '2026-10-29T00:00:00+00:00'}
    inventory_calls = []
    def inventory(path):
        inventory_calls.append(1)
        actual_gpu = ('GPU-00000000-0000-0000-0000-000000000002'
                      if mode == 'inventory_changed' and len(inventory_calls) == 2 else gpu)
        current = snapshot.copy()
        if mode == 'inventory_time_reversed' and len(inventory_calls) == 2:
            current['timestamp_utc'] = '2026-10-28T00:00:00+00:00'
        return actual_gpu, current
    monkeypatch.setattr(api, '_smi_snapshot', inventory)
    identity = SimpleNamespace(lock_path=tmp_path / 'device.lock', marker_path=tmp_path / 'device.unsafe.json',
                               attestation_path=tmp_path / 'device.attestation')
    monkeypatch.setattr(api, 'resolve_device_identity', lambda *args, **kwargs: identity)
    failure = KeyboardInterrupt()
    def supervise(command, **options):
        events.append(('launch', None))
        assert api._CUSTODY['source_guard'] is not None and api._CUSTODY['asset_guard'] is not None
        assert api._CUSTODY['job'] is options['process_job']
        assert options['require_zero_exit'] is True and options['inherit_environment'] is False
        assert options['cwd'] == attempt / 'tmp' and '--owner' in command
        assert options['env']['CUDA_VISIBLE_DEVICES'] == gpu and 'SYSTEMDRIVE' not in options['env']
        if mode == 'interrupted':
            raise failure
        if mode == 'tool_changed':
            smi.write_bytes(b'changed inventory tool')
        if mode == 'marker_remains':
            identity.marker_path.write_text('unsafe', encoding='ascii')
        return SupervisionResult('a' * 32, 1234, 1 if mode == 'timeout' else 0,
                                 mode != 'timeout', mode == 'timeout', 'synthetic')
    monkeypatch.setattr(api, 'run_supervised', supervise)
    monkeypatch.setattr(api, '_owner_presence_after_exit',
                        lambda *args: {'query': 'ok', 'owner_pid_listed': False})
    def validate_records(*args, **kwargs):
        if mode == 'missing_receipt':
            raise FileNotFoundError('owner-final.json is absent')
        return {'status': 'saved-records-consistent-not-approved', 'device_total_bytes': (
            snapshot['total_bytes'] + 2 * 1024 ** 2 + 1 if mode == 'owner_total_mismatch'
            else snapshot['total_bytes'])}
    monkeypatch.setattr(api, '_validate_records', validate_records)
    evidence_checks = []
    def evidence_files(path):
        evidence_checks.append(1)
        if mode == 'evidence_changed' and len(evidence_checks) == 3:
            return (path / 'synthetic.json', path / 'new-entry')
        return (path / 'synthetic.json',)
    monkeypatch.setattr(api, '_owner_evidence_files', evidence_files)
    monkeypatch.setattr(api, '_evidence_manifest', lambda *args: {'synthetic.json': {'bytes': 1, 'sha256': 'a' * 64}})
    real_write = api._write_json
    def write_record(path, value):
        digest = real_write(path, value)
        if mode == 'closure_write_failure' and Path(path).name == 'controller-closed.json':
            raise OSError('simulated closure fsync failure')
        return digest
    monkeypatch.setattr(api, '_write_json', write_record)
    result = api.run_controller(workspace_root=workspace, run_dir=attempt, asset_dir=assets,
        site_packages=site, git_executable=git, smi=smi, source_commit='b' * 40)
    if mode == 'clean':
        assert result['status'] == 'awaiting-independent-audit'
        assert (attempt / 'controller-final.json').is_file()
        assert [item[0] for item in events].count('guard-close') == 3
        closure = json.loads((attempt / 'controller-closed.json').read_bytes())
        assert closure['controller_final_sha256'] == hashlib.sha256((attempt / 'controller-final.json').read_bytes()).hexdigest()
        assert closure['requires_observed_controller_exit_zero'] is True
        assert len(source_calls) == 3
    elif mode == 'guard_interrupted':
        assert result['status'] == 'incomplete' and result['error_type'] == 'KeyboardInterrupt'
        assert api._CUSTODY['source_guard'] is not None and not api._CUSTODY['may_have_started']
        assert ('launch', None) not in events
    else:
        assert result['status'] == ('quarantined' if mode in ('timeout', 'interrupted') else 'incomplete')
        assert (attempt / 'controller-failure.json').is_file()
        if mode != 'closure_write_failure':
            assert ('guard-close', assets) not in events
        if mode in ('evidence_guard_close', 'closure_write_failure'):
            assert (attempt / 'controller-final.json').exists()
            assert (attempt / 'controller-closed.json').exists() is (mode == 'closure_write_failure')
        else:
            assert not (attempt / 'controller-final.json').exists()
        if mode == 'interrupted':
            assert result['error_type'] == 'KeyboardInterrupt' and api._CUSTODY['failure'] is failure
