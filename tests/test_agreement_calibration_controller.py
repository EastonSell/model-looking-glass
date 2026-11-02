"""CPU-only tests for the gated supervised calibration controller."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from looking_glass import agreement_calibration_controller as controller
from looking_glass.pretrained import ASSETS, MODEL_ID, REVISION


RUN_ID = '1' * 32
SHA = '2' * 64


@pytest.fixture(autouse=True)
def fresh_controller_attempt(monkeypatch):
    monkeypatch.setattr(controller, '_RUN_ONCE_LOCK', threading.Lock())


def candidate(freeze_sha=SHA):
    return {'measurement': {
        'schema': 'agreement-calibration-candidate-v1', 'run_id': RUN_ID,
        'profile_ids': ['3' * 64, '4' * 64], 'estimates_bytes': [700000000, 800000000],
        'source_sha256': SHA, 'runtime_sha256': SHA, 'asset_sha256': SHA, 'run_manifest_sha256': freeze_sha,
        'sample_count': 14, 'stage_count': 84,
        'workspace_cleanup': {'before': {'allocated_bytes': 5, 'reserved_bytes': 8},
                              'after': {'allocated_bytes': 0, 'reserved_bytes': 0},
                              'api': 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces'},
        'limitations': ['single observations are not worst-case bounds']},
        'cleanup': dict(controller._CLEANUP)}


def freeze(run_dir, cache):
    identity = controller._directory_identity(run_dir)
    value = {'schema': 'agreement-calibration-run-freeze-v1', 'run_id': RUN_ID,
        'source': {'commit': 'a' * 40, 'sha256': SHA,
                   'checkout': str(Path(controller.__file__).resolve().parents[1]),
                   'source_base_sha256': SHA},
        'checkpoint': {'model_id': MODEL_ID, 'revision': REVISION, 'cache_root': str(cache),
                       'assets': [{'name': name, 'size': size, 'sha256': sha}
                                  for name, (size, sha) in sorted(ASSETS.items())]},
        'runtime': {'python_version': '3.11', 'torch_version': '2.8.0+cu128',
                    'transformers_version': '4.57.3', 'numpy_version': '2.4.6',
                    'huggingface_hub_version': '0.36.2', 'safetensors_version': '0.8.0',
                    'tokenizers_version': '0.22.2', 'cuda_runtime_version': '12.8'},
        'device': {'uuid': 'GPU-example', 'name': 'example', 'total_bytes': 8 * 1024**3,
                   'driver_version': '1'}, 'workload': controller._workload(),
        'directory': {'path': str(run_dir), 'device': identity[0], 'inode': identity[1]}}
    digest = controller._write_private(run_dir / 'run-manifest.json', value)
    return value, digest


def test_controller_gate_refuses_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', False)
    with pytest.raises(RuntimeError, match='not yet enabled'):
        controller.run_calibration(workspace_root=tmp_path, cache_root=tmp_path,
                                   output=tmp_path / 'out.json', smi='nvidia-smi')
    assert not (tmp_path / 'out.json').exists()


def test_controller_rejects_a_second_concurrent_attempt(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    errors = []
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    def blocked_preflight():
        entered.set()
        assert release.wait(5)
        raise ValueError('synthetic preflight stop')
    monkeypatch.setattr(controller, '_bootstrap_state', blocked_preflight)
    def first():
        try:
            controller.run_calibration(workspace_root=tmp_path, cache_root=tmp_path,
                                       output=tmp_path / 'first.json', smi='nvidia-smi')
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=first, daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match='one attempt per controller process'):
            controller.run_calibration(workspace_root=tmp_path, cache_root=tmp_path,
                                       output=tmp_path / 'second.json', smi='nvidia-smi')
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], ValueError)


def test_workspace_root_rejects_nested_or_case_aliased_work_ancestors(tmp_path):
    root = tmp_path / 'workspace'
    repo = root / 'work' / 'teams' / 'repo'
    repo.mkdir(parents=True)
    assert controller.shared_workspace_root(repo) == root.resolve()
    for relative in ('other/repo', 'alternative/WORK/repo', 'work/nested-upper/WORK/repo',
                     'work/nested-lower/work/repo'):
        candidate = root / relative
        candidate.mkdir(parents=True)
        with pytest.raises(ValueError, match='canonical work ancestor'):
            controller.shared_workspace_root(candidate)


def test_wrong_workspace_is_rejected_before_source_or_device_preflight(tmp_path, monkeypatch):
    root, cache, other = (tmp_path / name for name in ('root', 'cache', 'other'))
    for path in (root, cache, other):
        path.mkdir()
    (root / 'work').mkdir()
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_CONTROLLER', {'source_guard': None, 'asset_guard': None,
                                                    'process_job': None, 'failure': None,
                                                    'launch_started': False})
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: (
        {'files': {}}, SHA, 'git', root, lambda *a: pytest.fail('source preflight ran')))
    monkeypatch.setattr(controller, 'shared_workspace_root', lambda _: other)
    monkeypatch.setattr(controller, 'resolve_device_identity', lambda *a, **k: pytest.fail('device preflight ran'))
    with pytest.raises(ValueError, match='shared workspace root'):
        controller.run_calibration(workspace_root=root, cache_root=cache,
                                   output=tmp_path / 'result.json', smi='nvidia-smi')
    assert not (tmp_path / 'result.json').exists()


def test_prelease_identity_rejects_another_valid_device_namespace(tmp_path, monkeypatch):
    from looking_glass.device_identity import resolve_device_identity
    from looking_glass.supervisor import _identity_spec
    uuid = 'GPU-00000000-0000-0000-0000-000000000001'
    root, other = tmp_path / 'workspace', tmp_path / 'other'
    repo = root / 'work' / 'teams' / 'repo'
    repo.mkdir(parents=True)
    other.mkdir()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', uuid)
    def session_for(workspace):
        identity = resolve_device_identity(workspace, uuid, physical_uuids=[uuid])
        return SimpleNamespace(_identity_spec=_identity_spec(identity),
                               device_lease_path=identity.lock_path, marker_path=identity.marker_path)
    assert controller._prelease_identity(repo, session_for(root), uuid).workspace_root == root.resolve()
    with pytest.raises(ValueError, match='shared workspace or device'):
        controller._prelease_identity(repo, session_for(other), uuid)


def test_owner_refuses_wrong_serialized_root_before_lease_or_collector(tmp_path, monkeypatch):
    from looking_glass.device_identity import resolve_device_identity
    from looking_glass.supervisor import _identity_spec
    uuid = 'GPU-00000000-0000-0000-0000-000000000001'
    run_dir, cache, other = (tmp_path / name for name in ('run', 'cache', 'other'))
    for path in (run_dir, cache, other):
        path.mkdir()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', uuid)
    wrong = resolve_device_identity(other, uuid, physical_uuids=[uuid])
    session = SimpleNamespace(_identity_spec=_identity_spec(wrong),
                              device_lease_path=wrong.lock_path, marker_path=wrong.marker_path)
    value, _ = freeze(run_dir, cache)
    value['device']['uuid'] = uuid
    (run_dir / 'run-manifest.json').unlink()
    freeze_sha = controller._write_private(run_dir / 'run-manifest.json', value)
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_OWNER', {'entered': False, 'session': None, 'lease': None,
                                              'candidate': None, 'failure': None})
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: ({'commit': 'a' * 40}, SHA, None, None, None))
    monkeypatch.setattr(controller, '_require_pristine_dependencies', lambda: None)
    monkeypatch.setattr(controller.OwnerSession, 'from_environment', lambda: session)
    monkeypatch.setattr(controller.DeviceLease, 'acquire', lambda *a, **k: pytest.fail('lease was acquired'))
    monkeypatch.setattr(controller, '_load_collector', lambda: pytest.fail('collector was imported'))
    directory = ':'.join(map(str, controller._directory_identity(run_dir)))
    assert controller.owner_main(str(run_dir), str(cache), RUN_ID, SHA, directory, freeze_sha) == 1
    assert isinstance(controller._OWNER['failure'], ValueError)
    assert 'shared workspace or device' in str(controller._OWNER['failure'])


@pytest.mark.parametrize(('observation', 'confirmed'), [
    ({'query': 'ok', 'owner_pid_listed': False}, True),
    ({'query': 'ok', 'owner_pid_listed': True}, False),
    ({'query': 'unavailable', 'owner_pid_listed': None}, False),
    ({'query': 'inconclusive', 'owner_pid_listed': None}, False),
    ({'query': 'ok', 'owner_pid_listed': 0}, False),
    ({}, False),
])
def test_owner_absence_requires_an_explicit_successful_observation(observation, confirmed):
    assert controller._owner_absence_confirmed(observation) is confirmed


def test_exact_cleanup_and_candidate_are_checked():
    value = candidate()
    assert controller._validate_candidate(value, RUN_ID, SHA)['sample_count'] == 14
    value['cleanup']['device_synchronized'] = 1
    with pytest.raises(ValueError, match='cleanup'):
        controller._validate_candidate(value, RUN_ID, SHA)
    value = candidate()
    value['measurement']['workspace_cleanup']['after']['allocated_bytes'] = False
    with pytest.raises(ValueError, match='counters'):
        controller._validate_candidate(value, RUN_ID, SHA)
    value = candidate()
    value['measurement']['behavioral_logits'] = [0.5]
    with pytest.raises(ValueError, match='unexpected fields'):
        controller._validate_candidate(value, RUN_ID, SHA)


def test_owner_checks_lease_before_callback_and_attests_only_after_cleanup(tmp_path, monkeypatch):
    run_dir, cache = tmp_path / 'run', tmp_path / 'cache'
    run_dir.mkdir()
    cache.mkdir()
    order = []
    session = SimpleNamespace(device_lease_path='lease', identity=SimpleNamespace(gpu_uuid='GPU-example', workspace_root=tmp_path), run_id='supervisor',
                              _runtime_verified=True)
    lease = object()
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_OWNER', {'entered': False, 'session': None, 'lease': None,
                                              'candidate': None, 'failure': None})
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: ({'commit': 'a' * 40}, SHA, None, None, None))
    monkeypatch.setattr(controller, '_require_pristine_dependencies', lambda: order.append('pristine'))
    monkeypatch.setattr(controller.OwnerSession, 'from_environment', lambda: (order.append('session'), session)[1])
    monkeypatch.setattr(controller, '_prelease_identity', lambda *a: (order.append('prelease'), session.identity)[1])
    monkeypatch.setattr(controller, 'shared_workspace_root', lambda _: tmp_path)
    monkeypatch.setattr(controller.DeviceLease, 'acquire', lambda *a, **k: (order.append('lease'), lease)[1])
    monkeypatch.setattr(controller.OwnerSession, 'check', lambda *a: order.append('check'))
    monkeypatch.setattr(controller.OwnerSession, 'attest_clean', lambda *a: order.append('attest'))
    monkeypatch.setattr(controller, '_validate_private_identity', lambda *a: order.append('evidence'))
    def collect(actual_session, actual_lease, **kwargs):
        assert actual_session is session and actual_lease is lease
        assert kwargs['run_dir_identity'] == controller._directory_identity(run_dir)
        order.append('collect')
        return candidate(kwargs['run_manifest_sha256'])
    monkeypatch.setattr(controller, '_load_collector', lambda: collect)
    _, freeze_sha = freeze(run_dir, cache)
    identity = ':'.join(map(str, controller._directory_identity(run_dir)))
    assert controller.owner_main(str(run_dir), str(cache), RUN_ID, SHA, identity, freeze_sha) == 0
    assert order == ['pristine', 'session', 'prelease', 'lease', 'check', 'collect', 'evidence', 'check', 'attest']
    assert controller._OWNER['lease'] is lease
    record, _ = controller._read_private(run_dir / 'final.json')
    assert record['cleanup'] == controller._CLEANUP


def test_owner_failure_keeps_marker_path_and_never_attests(tmp_path, monkeypatch):
    run_dir, cache = tmp_path / 'run', tmp_path / 'cache'
    run_dir.mkdir()
    cache.mkdir()
    session = SimpleNamespace(device_lease_path='lease', identity=SimpleNamespace(gpu_uuid='GPU-example', workspace_root=tmp_path), run_id='supervisor',
                              _runtime_verified=True)
    error = MemoryError('partial allocation')
    attest = []
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_OWNER', {'entered': False, 'session': None, 'lease': None,
                                              'candidate': None, 'failure': None})
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: ({'commit': 'a' * 40}, SHA, None, None, None))
    monkeypatch.setattr(controller, '_require_pristine_dependencies', lambda: None)
    monkeypatch.setattr(controller.OwnerSession, 'from_environment', lambda: session)
    monkeypatch.setattr(controller, '_prelease_identity', lambda *a: session.identity)
    monkeypatch.setattr(controller, 'shared_workspace_root', lambda _: tmp_path)
    monkeypatch.setattr(controller.DeviceLease, 'acquire', lambda *a, **k: object())
    monkeypatch.setattr(controller.OwnerSession, 'check', lambda *a: None)
    monkeypatch.setattr(controller.OwnerSession, 'attest_clean', lambda *a: attest.append(True))
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(controller, '_load_collector', lambda: fail)
    _, freeze_sha = freeze(run_dir, cache)
    identity = ':'.join(map(str, controller._directory_identity(run_dir)))
    assert controller.owner_main(str(run_dir), str(cache), RUN_ID, SHA, identity, freeze_sha) == 1
    assert controller._OWNER['failure'] is error
    assert not attest
    record, _ = controller._read_private(run_dir / 'failure.json')
    assert record['status'] == 'quarantined' and record['error_type'] == 'MemoryError'


def test_owner_rechecks_identity_after_lease_before_collector(tmp_path, monkeypatch):
    run_dir, cache = tmp_path / 'run', tmp_path / 'cache'
    run_dir.mkdir()
    cache.mkdir()
    original = SimpleNamespace(gpu_uuid='GPU-example', workspace_root=tmp_path)
    session = SimpleNamespace(device_lease_path='lease', identity=original, run_id='supervisor')
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_OWNER', {'entered': False, 'session': None, 'lease': None,
                                              'candidate': None, 'failure': None})
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: ({'commit': 'a' * 40}, SHA, None, None, None))
    monkeypatch.setattr(controller, '_require_pristine_dependencies', lambda: None)
    monkeypatch.setattr(controller.OwnerSession, 'from_environment', lambda: session)
    monkeypatch.setattr(controller, '_prelease_identity', lambda *a: original)
    monkeypatch.setattr(controller, 'shared_workspace_root', lambda _: tmp_path)
    monkeypatch.setattr(controller.DeviceLease, 'acquire', lambda *a, **k: object())
    monkeypatch.setattr(controller.OwnerSession, 'check', lambda *a: setattr(
        session, 'identity', SimpleNamespace(gpu_uuid='GPU-example', workspace_root=tmp_path / 'other')))
    monkeypatch.setattr(controller, '_load_collector', lambda: pytest.fail('collector was imported'))
    _, freeze_sha = freeze(run_dir, cache)
    directory = ':'.join(map(str, controller._directory_identity(run_dir)))
    assert controller.owner_main(str(run_dir), str(cache), RUN_ID, SHA, directory, freeze_sha) == 1
    assert isinstance(controller._OWNER['failure'], RuntimeError)
    assert 'identity changed after lease' in str(controller._OWNER['failure'])


def test_evidence_parser_rejects_duplicate_keys_and_directory_substitution(tmp_path):
    path = tmp_path / 'private.json'
    path.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError, match='duplicate'):
        controller._read_private(path)
    directory = tmp_path / 'run'
    directory.mkdir()
    identity = controller._directory_identity(directory)
    with pytest.raises(ValueError, match='identity changed'):
        controller._check_directory(directory, (identity[0], identity[1] + 1))


def test_private_identity_requires_frozen_sources_and_exact_stage_inventory(tmp_path):
    names = ('looking_glass/__init__.py', 'docs/TRUSTED_SERVICE_PROFILE_V1.md',
             'docs/AGREEMENT_GPU_CALIBRATION_V1.md', 'docs/AGREEMENT_CUDA_ADAPTER.md',
             'docs/AGREEMENT_CALIBRATION_SAMPLING_DRAFT.md')
    manifest = {'files': {name: SHA for name in names}}
    source = {'schema': 'agreement-calibration-source-v1', 'base': {'repository_files': [
        {'path': name, 'size': 1, 'sha256': SHA} for name in names[:2]]}, 'protocols': [
        {'path': name, 'size': 1, 'sha256': SHA} for name in names[2:]]}
    cache = tmp_path / 'cache'
    cache.mkdir()
    run_manifest, _ = freeze(tmp_path, cache)
    site = tmp_path / 'site'
    site.mkdir()
    origins = []
    for name, field in (('torch', 'torch_version'), ('transformers', 'transformers_version'),
                        ('numpy', 'numpy_version'), ('huggingface_hub', 'huggingface_hub_version'),
                        ('safetensors', 'safetensors_version'), ('tokenizers', 'tokenizers_version')):
        path = site / (name + '.py')
        path.write_bytes(b'fixture')
        origins.append({'name': name, 'version': run_manifest['runtime'][field], 'path': str(path),
                        'size': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()})
    source['dependency_origins'] = origins
    base_bytes = json.dumps(source['base'], ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'), allow_nan=False).encode()
    run_manifest['source']['source_base_sha256'] = hashlib.sha256(base_bytes).hexdigest()
    runtime = {'schema': 'agreement-calibration-runtime-v1', 'python_version': '3.11',
               'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
               'numpy_version': '2.4.6', 'huggingface_hub_version': '0.36.2',
               'safetensors_version': '0.8.0', 'tokenizers_version': '0.22.2',
               'cuda_runtime_version': '12.8', 'device_uuid': 'GPU-example', 'device_name': 'example',
               'device_total_bytes': 8 * 1024**3, 'driver_version': '1',
               'driver_version_provenance': 'controller-prelaunch-nvidia-smi',
               'subprocess_policy': 'python-audit-hook-denied'}
    checkpoint = run_manifest['checkpoint']
    assets = {'schema': 'resource-bootstrap-assets-v1', 'model_id': checkpoint['model_id'],
              'revision': checkpoint['revision'], 'assets': checkpoint['assets']}
    for name, value in (('source-manifest.json', source), ('runtime-manifest.json', runtime),
                        ('asset-manifest.json', assets)):
        controller._write_private(tmp_path / name, value)
    for index in range(84):
        controller._write_private(tmp_path / f'stage-{index:03}.json', {'index': index})
    measurement = candidate()['measurement']
    for key, name in (('source_sha256', 'source-manifest.json'), ('runtime_sha256', 'runtime-manifest.json'),
                      ('asset_sha256', 'asset-manifest.json')):
        measurement[key] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    controller._validate_private_identity(tmp_path, manifest, measurement, run_manifest, site)
    runtime['subprocess_policy'] = 'unknown'
    (tmp_path / 'runtime-manifest.json').unlink()
    measurement['runtime_sha256'] = controller._write_private(tmp_path / 'runtime-manifest.json', runtime)
    with pytest.raises(ValueError, match='runtime differs'):
        controller._validate_private_identity(tmp_path, manifest, measurement, run_manifest, site)
    runtime['subprocess_policy'] = 'python-audit-hook-denied'
    (tmp_path / 'runtime-manifest.json').unlink()
    measurement['runtime_sha256'] = controller._write_private(tmp_path / 'runtime-manifest.json', runtime)
    (tmp_path / 'stage-083.json').unlink()
    with pytest.raises(ValueError, match='stage record sequence'):
        controller._validate_private_identity(tmp_path, manifest, measurement, run_manifest, site)
    controller._write_private(tmp_path / 'stage-083.json', {'index': 83})
    manifest['files']['looking_glass/__init__.py'] = 'f' * 64
    with pytest.raises(ValueError, match='package source'):
        controller._validate_private_identity(tmp_path, manifest, measurement, run_manifest, site)


def test_bootstrap_rejects_unapproved_meta_path_finders(monkeypatch):
    script = Path(__file__).parents[1] / 'scripts' / 'agreement_calibration_bootstrap.py'
    spec = importlib.util.spec_from_file_location('agreement_bootstrap_finder_test', script)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    standard = [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                importlib.machinery.PathFinder]
    project, hostile = object(), object()
    monkeypatch.setattr(sys, 'meta_path', [project, *standard])
    bootstrap._check_meta_path(project)
    for finders in ([hostile, project, *standard], [project, hostile, *standard],
                    [project, *standard, hostile], [project, *reversed(standard)],
                    [project, *standard, project]):
        monkeypatch.setattr(sys, 'meta_path', finders)
        with pytest.raises(RuntimeError, match='import finder'):
            bootstrap._check_meta_path(project)


def test_bootstrap_detects_changed_or_untracked_package_source(tmp_path):
    script = Path(__file__).parents[1] / 'scripts' / 'agreement_calibration_bootstrap.py'
    spec = importlib.util.spec_from_file_location('agreement_bootstrap_test', script)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    repo = tmp_path / 'repo'
    (repo / 'looking_glass').mkdir(parents=True)
    (repo / 'docs').mkdir()
    (repo / 'scripts').mkdir()
    (repo / 'looking_glass' / '__init__.py').write_text('value = 1\n')
    (repo / 'scripts' / script.name).write_bytes(script.read_bytes())
    guard = script.parent / 'v2_source_guards.py'
    (repo / 'scripts' / guard.name).write_bytes(guard.read_bytes())
    for name in bootstrap._DOCS:
        (repo / name).write_text('protocol\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=CPU test', '-c', 'user.email=test@example.invalid',
                    'commit', '-q', '-m', 'fixture'], cwd=repo, check=True)
    import shutil
    manifest, digest, _ = bootstrap.capture_source(repo, shutil.which('git'))
    assert len(digest) == 64 and 'looking_glass/__init__.py' in manifest['files']
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    controller._write_private(run_dir / 'controller-source.json', manifest)
    child_manifest, child_digest, child_sources = bootstrap.capture_parent_frozen_source(repo, run_dir, digest)
    assert child_manifest == manifest and child_digest == digest
    assert child_sources['looking_glass/__init__.py'].replace(b'\r\n', b'\n') == b'value = 1\n'
    with pytest.raises(RuntimeError, match='commitment'):
        bootstrap.capture_parent_frozen_source(repo, run_dir, 'f' * 64)
    (repo / 'looking_glass' / 'unexpected.py').write_text('pass\n')
    with pytest.raises(RuntimeError, match='untracked or missing'):
        bootstrap.capture_source(repo, shutil.which('git'))
    (repo / 'looking_glass' / 'unexpected.py').unlink()
    (repo / 'looking_glass' / '__init__.py').write_text('value = 2\n')
    with pytest.raises(RuntimeError, match='clean committed'):
        bootstrap.capture_source(repo, shutil.which('git'))


def test_bootstrap_ignores_git_replace_objects_and_inherited_git_state(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / 'scripts' / 'agreement_calibration_bootstrap.py'
    spec = importlib.util.spec_from_file_location('agreement_bootstrap_replace', script)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    repo = tmp_path / 'repo'
    (repo / 'looking_glass').mkdir(parents=True)
    (repo / 'docs').mkdir()
    (repo / 'scripts').mkdir()
    (repo / 'looking_glass' / '__init__.py').write_text('value = 1\n')
    for name in bootstrap._SCRIPTS:
        (repo / name).write_bytes((script.parents[1] / name).read_bytes())
    for name in bootstrap._DOCS:
        (repo / name).write_text('protocol\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    config = ['git', '-c', 'user.name=CPU test', '-c', 'user.email=test@example.invalid']
    subprocess.run(config + ['commit', '-q', '-m', 'first'], cwd=repo, check=True)
    first = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    (repo / 'looking_glass' / '__init__.py').write_text('value = 2\n')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(config + ['commit', '-q', '-m', 'second'], cwd=repo, check=True)
    second = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    subprocess.run(['git', 'replace', first, second], cwd=repo, check=True)
    subprocess.run(['git', '--no-replace-objects', 'checkout', '-q', '--detach', first], cwd=repo, check=True)
    monkeypatch.setenv('GIT_INDEX_FILE', str(tmp_path / 'missing-index'))
    import shutil
    manifest, _, sources = bootstrap.capture_source(repo, shutil.which('git'))
    assert manifest['commit'] == first
    assert sources['looking_glass/__init__.py'].replace(b'\r\n', b'\n') == b'value = 1\n'


def test_bootstrap_checks_dependency_resolution_before_import(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / 'scripts' / 'agreement_calibration_bootstrap.py'
    spec = importlib.util.spec_from_file_location('agreement_bootstrap_resolution', script)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    site, shadow = tmp_path / 'site', tmp_path / 'shadow'
    site.mkdir()
    shadow.mkdir()
    for name in ('torch', 'transformers', 'huggingface_hub', 'numpy', 'safetensors', 'tokenizers'):
        package = site / name
        package.mkdir()
        (package / '__init__.py').write_text('raise AssertionError("must not be imported")\n')
    monkeypatch.setattr(bootstrap.sys, 'path', [str(site)])
    bootstrap._check_dependency_origins(site)
    (shadow / 'torch').mkdir()
    (shadow / 'torch' / '__init__.py').write_text('pass\n')
    monkeypatch.setattr(bootstrap.sys, 'path', [str(shadow), str(site)])
    with pytest.raises(ImportError, match='configured site'):
        bootstrap._check_dependency_origins(site)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows pre-grant pipe inspection')
def test_bootstrap_waits_for_the_supervisor_token_without_consuming_it(monkeypatch):
    import msvcrt
    script = Path(__file__).parents[1] / 'scripts' / 'agreement_calibration_bootstrap.py'
    spec = importlib.util.spec_from_file_location('agreement_bootstrap_grant', script)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    reader, writer = os.pipe()
    entered, finished, errors = threading.Event(), threading.Event(), []
    monkeypatch.setenv('MLG_OWNER_LAUNCH_HANDLE', str(msvcrt.get_osfhandle(reader)))
    monkeypatch.setenv('MLG_SUPERVISOR_PID', str(os.getppid()))
    def wait():
        entered.set()
        try:
            bootstrap._wait_for_supervisor_grant()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
    thread = threading.Thread(target=wait, daemon=True)
    try:
        thread.start()
        assert entered.wait(2)
        assert not finished.wait(.15)
        assert os.write(writer, b'\x01') == 1
        assert finished.wait(2)
        assert not errors
        assert os.read(reader, 1) == b'\x01'
    finally:
        os.close(writer)
        thread.join(2)
        os.close(reader)


@pytest.mark.parametrize('child_clean', [True, False])
def test_source_and_asset_guards_stay_held_when_child_cleanup_is_unknown(tmp_path, monkeypatch, child_clean):
    root, cache = tmp_path / 'workspace', tmp_path / 'cache'
    (root / 'work').mkdir(parents=True)
    cache.mkdir()
    guards = []
    class Guard:
        def __init__(self, *args):
            self.entered = self.closed = False
            guards.append(self)
        def __enter__(self):
            self.entered = True
            return self
        def close(self):
            assert self.entered
            self.closed = True
    preflight = SimpleNamespace(asset_paths=(('config.json', 'ignored'),), source_base=b'{}',
                                asset_manifest=b'{}')
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_CONTROLLER', {'source_guard': None, 'asset_guard': None,
                                                    'failure': None, 'launch_started': False})
    monkeypatch.setitem(sys.modules, '_agreement_source_guards', SimpleNamespace(SourceFileLease=Guard))
    from looking_glass import trusted_service_profile
    monkeypatch.setattr(trusted_service_profile, 'prepare_trusted_source', lambda *a: preflight)
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: (
        {'files': {'looking_glass/__init__.py': SHA}}, SHA, 'git', root, lambda *a: (None, SHA, None)))
    monkeypatch.setattr(controller, 'shared_workspace_root', lambda _: root)
    def supervised(*args):
        assert all(guard.entered and not guard.closed for guard in guards)
        return {'status': 'awaiting_independent_audit' if child_clean else 'quarantined'}, child_clean, None
    monkeypatch.setattr(controller, '_run_with_protected_source', supervised)
    result = controller.run_calibration(workspace_root=root, cache_root=cache,
                                        output=tmp_path / 'result.json', smi='nvidia-smi')
    assert result['source_protections_retained'] is (not child_clean)
    assert all(guard.closed is child_clean for guard in guards)
    if not child_clean:
        assert controller._CONTROLLER['source_guard'] is guards[0]
        assert controller._CONTROLLER['asset_guard'] is guards[1]
    with pytest.raises(RuntimeError, match='one attempt per controller process'):
        controller.run_calibration(workspace_root=root, cache_root=cache,
                                   output=tmp_path / 'second.json', smi='nvidia-smi')


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_controller_preserves_supervisor_fatal_and_retains_source_guards(tmp_path, monkeypatch, error_type):
    root, cache = tmp_path / 'workspace', tmp_path / 'cache'
    (root / 'work').mkdir(parents=True)
    cache.mkdir()
    guards = []
    class Guard:
        def __init__(self, *args):
            self.closed = False
            guards.append(self)
        def __enter__(self):
            return self
        def close(self):
            self.closed = True
    preflight = SimpleNamespace(asset_paths=(('config.json', 'ignored'),), source_base=b'{}',
                                asset_manifest=b'{}')
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_CONTROLLER', {'source_guard': None, 'asset_guard': None,
                                                    'failure': None, 'launch_started': False})
    monkeypatch.setitem(sys.modules, '_agreement_source_guards', SimpleNamespace(SourceFileLease=Guard))
    from looking_glass import trusted_service_profile
    monkeypatch.setattr(trusted_service_profile, 'prepare_trusted_source', lambda *a: preflight)
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: (
        {'files': {'looking_glass/__init__.py': SHA}}, SHA, 'git', root, lambda *a: (None, SHA, None)))
    monkeypatch.setattr(controller, 'shared_workspace_root', lambda _: root)
    private = error_type('original')
    def supervised(*args):
        controller._CONTROLLER['failure'] = private
        return {'status': 'quarantined'}, False, private
    monkeypatch.setattr(controller, '_run_with_protected_source', supervised)
    with pytest.raises(error_type) as observed:
        controller.run_calibration(workspace_root=root, cache_root=cache,
                                   output=tmp_path / 'result.json', smi='nvidia-smi')
    assert observed.value is not private
    assert observed.value.__context__ is None
    assert controller._CONTROLLER['failure'] is private
    assert all(not guard.closed for guard in guards)
    result = json.loads((tmp_path / 'result.json').read_bytes())
    assert result['status'] == 'quarantined' and result['source_protections_retained'] is True


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('failed_guard', [0, 1])
def test_guard_close_interrupt_remains_fatal_and_roots_guards(tmp_path, monkeypatch, error_type, failed_guard):
    root, cache = tmp_path / 'workspace', tmp_path / 'cache'
    (root / 'work').mkdir(parents=True)
    cache.mkdir()
    guards = []
    private = error_type('injected guard close interruption')
    class Guard:
        def __init__(self, *args):
            self.index, self.closed = len(guards), False
            guards.append(self)
        def __enter__(self):
            return self
        def close(self):
            if self.index == failed_guard:
                raise private
            self.closed = True
    preflight = SimpleNamespace(asset_paths=(('config.json', 'ignored'),), source_base=b'{}',
                                asset_manifest=b'{}')
    monkeypatch.setattr(controller, 'CONTROLLER_EXECUTION_READY', True)
    monkeypatch.setattr(controller, '_CONTROLLER', {'source_guard': None, 'asset_guard': None,
                                                    'process_job': None, 'failure': None,
                                                    'launch_started': False})
    monkeypatch.setitem(sys.modules, '_agreement_source_guards', SimpleNamespace(SourceFileLease=Guard))
    from looking_glass import trusted_service_profile
    monkeypatch.setattr(trusted_service_profile, 'prepare_trusted_source', lambda *a: preflight)
    monkeypatch.setattr(controller, '_bootstrap_state', lambda: (
        {'files': {'looking_glass/__init__.py': SHA}}, SHA, 'git', root, lambda *a: (None, SHA, None)))
    monkeypatch.setattr(controller, 'shared_workspace_root', lambda _: root)
    monkeypatch.setattr(controller, '_run_with_protected_source', lambda *a: (
        {'status': 'awaiting_independent_audit'}, True, None))
    with pytest.raises(error_type) as observed:
        controller.run_calibration(workspace_root=root, cache_root=cache,
                                   output=tmp_path / 'result.json', smi='nvidia-smi')
    assert observed.value is not private and observed.value.__context__ is None
    assert controller._CONTROLLER['failure'] is private
    assert controller._CONTROLLER['source_guard'] is guards[0]
    assert controller._CONTROLLER['asset_guard'] is guards[1]
    result = json.loads((tmp_path / 'result.json').read_bytes())
    assert result['status'] == 'quarantined' and result['source_protections_retained'] is True
    assert result['protection_cleanup_error_type'] == error_type.__name__


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_supervisor_interruption_is_recorded_as_a_fatal_unknown_outcome(tmp_path, monkeypatch, error_type):
    root, cache, repo, site = (tmp_path / name for name in ('root', 'cache', 'repo', 'site'))
    for path in (root, cache, repo, site):
        path.mkdir()
    work = root / 'work'
    work.mkdir()
    preflight = SimpleNamespace(source_base=b'{}', asset_manifest=json.dumps({
        'model_id': MODEL_ID, 'revision': REVISION, 'assets': []}).encode())
    before = {'device_name': 'test', 'total_bytes': 8 * 1024**3, 'driver_version': '1',
              'timestamp_utc': '2026-01-01T00:00:00+00:00'}
    identity = SimpleNamespace(lock_path='lease', marker_path='marker')
    error = error_type('injected')
    monkeypatch.setattr(controller, '_CONTROLLER', {'source_guard': object(), 'asset_guard': object(),
                                                    'failure': None, 'launch_started': False})
    monkeypatch.setattr(controller, '_dependency_versions', lambda *a: {})
    monkeypatch.setattr(controller, '_smi_snapshot', lambda *a: ('GPU-test', before))
    monkeypatch.setattr(controller, 'resolve_device_identity', lambda *a, **k: identity)
    class Job:
        def __init__(self):
            assert controller._CONTROLLER['process_job'] is self
    monkeypatch.setitem(sys.modules, '_agreement_source_guards', SimpleNamespace(WindowsJob=Job))
    def fail(*args, **kwargs):
        assert kwargs['process_job'] is controller._CONTROLLER['process_job']
        raise error
    monkeypatch.setattr(controller, 'run_supervised', fail)
    outcome, clean, fatal = controller._run_with_protected_source(
        root, cache, work, repo, site, 'git', lambda *a: (None, SHA, None),
        {'commit': 'a' * 40, 'files': {}}, SHA, preflight, 'nvidia-smi')
    assert outcome['status'] == 'quarantined' and outcome['supervisor_error_type'] == error_type.__name__
    assert clean is False and fatal is error and controller._CONTROLLER['failure'] is error
    assert type(controller._CONTROLLER['process_job']) is Job


@pytest.mark.parametrize('clean', [True, False])
def test_exact_process_job_is_rooted_before_initialization_and_supervision(tmp_path, monkeypatch, clean):
    root, cache, repo, site = (tmp_path / name for name in ('root', 'cache', 'repo', 'site'))
    for path in (root, cache, repo, site):
        path.mkdir()
    work = root / 'work'
    work.mkdir()
    preflight = SimpleNamespace(source_base=b'{}', asset_manifest=json.dumps({
        'model_id': MODEL_ID, 'revision': REVISION, 'assets': []}).encode())
    before = {'device_name': 'test', 'total_bytes': 8 * 1024**3, 'driver_version': '1',
              'timestamp_utc': '2026-01-01T00:00:00+00:00'}
    identity = SimpleNamespace(lock_path='lease', marker_path='marker')
    monkeypatch.setattr(controller, '_CONTROLLER', {'source_guard': object(), 'asset_guard': object(),
                                                    'process_job': None, 'failure': None,
                                                    'launch_started': False})
    monkeypatch.setattr(controller, '_dependency_versions', lambda *a: {})
    monkeypatch.setattr(controller, '_smi_snapshot', lambda *a: ('GPU-test', before))
    monkeypatch.setattr(controller, 'resolve_device_identity', lambda *a, **k: identity)
    monkeypatch.setattr(controller, '_owner_presence_after_exit', lambda *a: {
        'query': 'ok', 'owner_pid_listed': False})
    jobs = []
    class Job:
        def __init__(self):
            assert controller._CONTROLLER['process_job'] is self
            jobs.append(self)
    monkeypatch.setitem(sys.modules, '_agreement_source_guards', SimpleNamespace(WindowsJob=Job))
    def supervise(*args, **kwargs):
        assert kwargs['process_job'] is jobs[0] is controller._CONTROLLER['process_job']
        assert kwargs['inherit_environment'] is False
        assert kwargs['require_zero_exit'] is True
        return SimpleNamespace(pid=123, run_id='f' * 32, exit_code=0,
                               clean=clean, timed_out=False)
    monkeypatch.setattr(controller, 'run_supervised', supervise)
    outcome, child_clean, fatal = controller._run_with_protected_source(
        root, cache, work, repo, site, 'git', lambda *a: (None, SHA, None),
        {'commit': 'a' * 40, 'files': {}}, SHA, preflight, 'nvidia-smi')
    assert child_clean is clean and fatal is None
    assert outcome['nvidia_smi_before']['uuid'] == 'GPU-test'
    assert outcome['nvidia_smi_after']['uuid'] == 'GPU-test'
    supervision = json.loads((Path(outcome['private_run_dir']) / 'supervision.json').read_bytes())
    assert supervision == {'schema': 'agreement-calibration-supervision-v1',
                           'calibration_run_id': outcome['run_id'], 'supervisor': outcome['supervisor']}
    assert supervision['supervisor']['run_id'] == 'f' * 32
    assert supervision['calibration_run_id'] != supervision['supervisor']['run_id']
    assert controller._CONTROLLER['process_job'] is (None if clean else jobs[0])
    assert outcome['status'] == ('incomplete' if clean else 'quarantined')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows file-share protection')
def test_windows_source_lease_denies_mutation_until_close(tmp_path):
    script = Path(__file__).parents[1] / 'scripts' / 'v2_source_guards.py'
    spec = importlib.util.spec_from_file_location('agreement_source_guard_test', script)
    guards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guards)
    path = tmp_path / 'source.py'
    path.write_text('original')
    lease = guards.SourceFileLease(tmp_path, [path])
    with lease:
        with pytest.raises(PermissionError):
            path.write_text('changed')
        assert path.read_text() == 'original'
    path.write_text('after release')
