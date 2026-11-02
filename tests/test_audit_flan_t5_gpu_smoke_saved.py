"""No-model synthetic controls for the independent FLAN-T5 saved-record reader."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from test_flan_t5_device_controller import fabricated_owner, _recount, _write_json
from tools import audit_flan_t5_gpu_smoke_saved as audit


@pytest.fixture
def bundle(fabricated_owner):
    _, root, supervision, kwargs = fabricated_owner
    stage = root / 'staged-assets'
    files = {}
    for name in audit._FILES:
        raw = (stage / name).read_bytes()
        files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob_sha1': 'a' * 40}
    manifest = {'schema': 'mlg-flant5-small-assets-v1', 'repo': 'google/flan-t5-small',
                'revision': audit._REVISION, 'files': files}
    _write_json(stage / 'asset-manifest.json', manifest)
    manifest_raw = (stage / 'asset-manifest.json').read_bytes()
    asset_hash = hashlib.sha256(manifest_raw).hexdigest()
    for name in ('owner-started.json', 'owner-final.json'):
        path = root / name
        value = json.loads(path.read_bytes())
        value['asset_manifest_sha256'] = asset_hash
        if name == 'owner-final.json':
            payload = sum(item['bytes'] for item in files.values())
            value['staged_assets'] = {'file_count': 7, 'payload_bytes': payload,
                'manifest_bytes': len(manifest_raw), 'total_bytes': payload + len(manifest_raw)}
        _write_json(path, value)
    _recount(root)
    receipt = {'schema': 'flan-t5-gpu-supervision-v1', 'attempt_id': 'd' * 32,
        'run_id': supervision.run_id, 'pid': supervision.pid, 'exit_code': 0,
        'clean': True, 'timed_out': False, 'reason': 'clean device teardown'}
    return root, receipt, kwargs['source_commit'], asset_hash


def run(bundle):
    root, receipt, commit, asset_hash = bundle
    return audit.audit_owner(root, receipt, commit, asset_hash=asset_hash, source_paths={'source.py'})


def edit(path, change):
    value = json.loads(path.read_bytes())
    change(value)
    _write_json(path, value)


def test_complete_synthetic_owner_records_are_checked(bundle):
    result = run(bundle)
    assert result['status'] == 'owner-records-consistent-not-approved'
    assert result['identity_scope'] == 'synthetic-overrides'
    assert result['forwards'] == 6 and result['memory_stages'] == 37
    assert any('controller exit' in item for item in result['limits'])
    assert any('quiescent' in item for item in result['limits'])
    assert any('synchronization' in item for item in result['limits'])


def test_even_equal_explicit_identities_are_reported_as_synthetic():
    assert audit._identities(audit._UNSET, audit._UNSET)[2] == 'pinned'
    assert audit._identities(audit._ASSET, audit._SOURCES)[2] == 'synthetic-overrides'
    with pytest.raises(ValueError):
        audit._identities(True, {'source.py'})
    with pytest.raises(ValueError):
        audit._identities('a' * 64, {'../source.py'})


def test_reader_imports_without_model_dependencies():
    source = Path(audit.__file__).resolve()
    code = ('import importlib.util, sys; '
            'spec=importlib.util.spec_from_file_location("independent_reader", sys.argv[1]); '
            'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
            'assert not any(name in sys.modules for name in ("torch", "transformers", "numpy"))')
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code, str(source)],
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


@pytest.mark.parametrize('mutation', ['bool_pid', 'bool_coordinate', 'source_row', 'missing_forward',
    'extra_forward', 'bad_stage', 'bad_cleanup', 'bad_headroom', 'nonfinite_tensor',
    'changed_patch', 'subnormal_difference', 'changed_asset', 'extra_asset', 'duplicate_json', 'wrong_run', 'receipt_reason'])
def test_rehashed_contradictions_and_incomplete_records_are_rejected(bundle, mutation):
    root, receipt, _, _ = bundle
    final = root / 'owner-final.json'
    if mutation == 'bool_pid':
        edit(final, lambda item: item.update(pid=True))
    elif mutation == 'bool_coordinate':
        edit(root / 'donor-patch.json', lambda item: item['metadata']['steps'][1].update(token_index=True))
    elif mutation == 'source_row':
        edit(root / 'capture-a.json', lambda item: item['source_ids'].__setitem__(6, 629))
    elif mutation == 'missing_forward':
        (root / 'forward-002-started.json').unlink()
    elif mutation == 'extra_forward':
        _write_json(root / 'forward-006-started.json', {'index': 6})
    elif mutation == 'bad_stage':
        edit(root / 'memory-006.json', lambda item: item.update(stage='before_observation'))
    elif mutation == 'bad_cleanup':
        edit(final, lambda item: item['cleanup'].update(allocated_bytes=1))
    elif mutation == 'bad_headroom':
        edit(root / 'memory-006.json', lambda item: item['memory'].update(device_free_bytes=0))
    elif mutation == 'nonfinite_tensor':
        path = root / 'capture-a-logits.bin'
        raw = struct.pack('<I', 0x7f800000) + path.read_bytes()[4:]
        path.write_bytes(raw)
        edit(root / 'capture-a.json', lambda item: item['tensors']['logits'].update(sha256=hashlib.sha256(raw).hexdigest()))
    elif mutation == 'changed_patch':
        path = root / 'donor-patch-after.bin'
        raw = bytes(path.stat().st_size)
        path.write_bytes(raw)
        edit(root / 'donor-patch.json', lambda item: item['tensors']['after'].update(sha256=hashlib.sha256(raw).hexdigest()))
    elif mutation == 'subnormal_difference':
        for name in ('state', 'before'):
            path = root / f'repeat-a-{name}.bin'
            raw = struct.pack('<I', 1) + path.read_bytes()[4:]
            path.write_bytes(raw)
            edit(root / 'repeat-a.json', lambda item: item['tensors'][name].update(sha256=hashlib.sha256(raw).hexdigest()))
    elif mutation == 'changed_asset':
        (root / 'staged-assets' / 'model.safetensors').write_bytes(b'y')
    elif mutation == 'extra_asset':
        (root / 'staged-assets' / 'extra').write_bytes(b'x')
    elif mutation == 'duplicate_json':
        path = root / 'capture-a.json'
        path.write_bytes(path.read_bytes().replace(b'{', b'{"label":"other",', 1))
    elif mutation == 'wrong_run':
        receipt['run_id'] = 'f' * 32
    elif mutation == 'receipt_reason':
        receipt['reason'] = 'timeout'
    if final.exists():
        _recount(root)
    with pytest.raises((ValueError, OSError)):
        run(bundle)
