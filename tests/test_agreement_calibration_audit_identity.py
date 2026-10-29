"""Synthetic source/runtime freeze tests: no model load or CUDA activity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from looking_glass.pretrained import ASSETS, MODEL_ID, REVISION
from looking_glass.resource_profile import model_runtime_fingerprint
from looking_glass import agreement_calibration_saved_audit as audit
from test_agreement_calibration_saved_audit import frozen_fixture


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_workload(source, runtime):
    # Independent resource-profile oracle; fixed policy fields come from the
    # immutable fake producer fixture rather than the audit implementation.
    fields = {'checkpoint_sha256': ASSETS['model.safetensors'][1], 'checkpoint_revision': REVISION,
              'model_config_sha256': ASSETS['config.json'][1], 'source_sha256': digest(source),
              'runtime_settings_sha256': digest(canonical(runtime))}
    names = ('torch_version', 'transformers_version', 'cuda_runtime_version', 'driver_version',
             'python_version', 'device', 'device_uuid', 'device_name', 'device_total_bytes',
             'model_dtype', 'compute_dtype', 'attention_backend', 'model_mode', 'grad_mode',
             'autocast', 'use_cache', 'compiled', 'compute_capability')
    fields.update({name: runtime[name] for name in names})
    return {**frozen_fixture()['warm_key'], **fields, 'model_id': model_runtime_fingerprint(fields)}


@pytest.fixture
def frozen_identity(tmp_path):
    root = tmp_path.resolve()
    cache = root / 'hf' / 'hub'
    site = root / 'site'
    run = root / 'run'
    for path in (cache, site, run): path.mkdir(parents=True)
    uuid = 'GPU-00000000-0000-0000-0000-000000000001'
    version_fields = {'python_version': '3.11.15', 'torch_version': '2.8.0+cu128',
        'transformers_version': '4.57.3', 'numpy_version': '2.4.6', 'huggingface_hub_version': '0.36.2',
        'safetensors_version': '0.8.0', 'tokenizers_version': '0.22.2', 'cuda_runtime_version': '12.8'}
    names = ('looking_glass/__init__.py', 'docs/TRUSTED_SERVICE_PROFILE_V1.md',
             'docs/AGREEMENT_GPU_CALIBRATION_V1.md', 'docs/AGREEMENT_CUDA_ADAPTER.md',
             'docs/AGREEMENT_CALIBRATION_SAMPLING_DRAFT.md',
             'scripts/agreement_calibration_bootstrap.py', 'scripts/v2_source_guards.py')
    files = {name: digest(name.encode()) for name in names}
    controller = {'schema': 'agreement-controller-source-v1', 'commit': 'a' * 40, 'files': files}
    assets = {'schema': 'resource-bootstrap-assets-v1', 'model_id': MODEL_ID, 'revision': REVISION,
              'assets': [{'name': n, 'size': size, 'sha256': sha} for n, (size, sha) in sorted(ASSETS.items())]}
    base = {'schema': 'trusted-service-source-base-v1', 'repository_files': [
        {'path': n, 'size': len(n), 'sha256': files[n]} for n in names[:2]],
        'python_executable': {'size': 10, 'sha256': '1' * 64}, 'assets': assets}
    dependency_versions = [('torch', 'torch_version'), ('transformers', 'transformers_version'),
                           ('numpy', 'numpy_version'), ('huggingface_hub', 'huggingface_hub_version'),
                           ('safetensors', 'safetensors_version'), ('tokenizers', 'tokenizers_version')]
    origins = []
    for n, v in dependency_versions:
        path = site / (n + '.py')
        path.write_bytes(b'fixture')
        origins.append({'name': n, 'version': version_fields[v], 'path': str(path),
                        'size': 7, 'sha256': digest(b'fixture')})
    source = {'schema': 'agreement-calibration-source-v1', 'base': base, 'dependency_origins': origins,
              'protocols': [{'path': n, 'size': len(n), 'sha256': files[n]} for n in names[2:5]]}
    environment = {'CUDA_VISIBLE_DEVICES': uuid, 'HF_HOME': str(cache.parent), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'PYTORCH_CUDA_ALLOC_CONF': 'backend:native', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTORCH_ALLOC_CONF': None,
        'PYTORCH_NO_CUDA_MEMORY_CACHING': None, 'CUDA_LAUNCH_BLOCKING': None}
    runtime = {'schema': 'agreement-calibration-runtime-v1',
        'execution_path': 'looking_glass.agreement_cuda_calibration.CudaAgreementCalibrator',
        'device': 'cuda:0', 'device_uuid': uuid, 'device_name': 'synthetic GPU', 'device_total_bytes': 8 * 1024**3,
        'compute_capability': [7, 5], **version_fields, 'driver_version': '570.0',
        'driver_version_provenance': 'controller-prelaunch-nvidia-smi',
        'subprocess_policy': 'python-audit-hook-denied', 'model_dtype': 'torch.float32',
        'compute_dtype': 'torch.float32', 'attention_backend': 'eager', 'model_mode': 'eval',
        'grad_mode': 'inference', 'autocast': 'disabled', 'use_cache': False, 'compiled': False,
        'allocator_backend': 'native', 'allocator_fraction': '0.25', 'allocator_limit_bytes': 2 * 1024**3,
        'environment': environment,
        'effective_huggingface_settings': {'hf_home': str(cache.parent), 'hf_hub_cache': str(cache),
            'transformers_cache': str(cache), 'hf_modules_cache': str(cache.parent / 'modules'),
            'hub_offline': True, 'transformers_offline': True, 'telemetry_disabled': True},
        'torch_build_sha256': '3' * 64, 'python_import_paths': [str(site)],
        'torch_threads': 1, 'torch_interop_threads': 1, 'deterministic_algorithms': True,
        'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_enabled': True,
        'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'float32_matmul_precision': 'highest', 'deterministic_warn_only': False,
        'calling_thread_denormal_product_bits': 0x00400000, 'cpu_autocast': False}
    freeze = {'schema': 'agreement-calibration-run-freeze-v1', 'run_id': 'b' * 32,
        'source': {'commit': 'a' * 40, 'sha256': digest(canonical(controller)),
            'checkout': str(root / 'source'), 'source_base_sha256': digest(canonical(base))},
        'checkpoint': {'model_id': MODEL_ID, 'revision': REVISION, 'cache_root': str(cache), 'assets': assets['assets']},
        'runtime': version_fields, 'device': {'uuid': uuid, 'name': 'synthetic GPU',
            'total_bytes': 8 * 1024**3, 'driver_version': '570.0'},
        'workload': {'seed': 20261028, 'target_input_ids': [464,3797,1474,262,9168],
            'donor_input_ids': [464,11875,1474,262,9168], 'subject_index': 1, 'last_token_index': 4,
            'batch_size': 1, 'sequence_length': 5},
        'directory': {'path': str(run), 'device': run.stat().st_dev, 'inode': run.stat().st_ino}}
    records = {'run_manifest': canonical(freeze) + b'\n', 'controller_source': canonical(controller) + b'\n',
               'source_base': canonical(base), 'source': canonical(source), 'runtime': canonical(runtime),
               'assets': canonical(assets)}
    return records, 'a' * 40, uuid, expected_workload(records['source'], runtime)


def test_reconstructs_live_workload_key_from_frozen_cpu_records(frozen_identity):
    records, commit, uuid, expected_key = frozen_identity
    assert hasattr(audit, 'audit_identity'), 'independent source/runtime identity audit is missing'
    result = audit.audit_identity(records, expected_commit=commit, expected_device_uuid=uuid,
                                  expected_freeze_sha256=digest(records['run_manifest']))
    assert result['warm_key'] == json.loads(canonical(expected_key))
    assert result['source_sha256'] == digest(records['source'])


def test_accepts_non_ascii_paths_with_controller_and_owner_canonical_encodings(frozen_identity):
    records, commit, uuid, _ = deepcopy(frozen_identity)
    controller = json.loads(records['controller_source'])
    name = 'looking_glass/caf\u00e9.py'
    controller['files'][name] = digest(b'non-ascii filename')
    controller_raw = json.dumps(controller, sort_keys=True, separators=(',', ':')).encode()
    records['controller_source'] = controller_raw + b'\n'
    base = json.loads(records['source_base'])
    base['repository_files'].append({'path': name, 'size': 18, 'sha256': controller['files'][name]})
    records['source_base'] = canonical(base)
    source = json.loads(records['source'])
    source['base'] = base
    records['source'] = canonical(source)
    freeze = json.loads(records['run_manifest'])
    freeze['source']['checkout'] = str(Path(freeze['source']['checkout']) / 'caf\u00e9')
    freeze['source']['sha256'] = digest(controller_raw)
    freeze['source']['source_base_sha256'] = digest(records['source_base'])
    records['run_manifest'] = json.dumps(freeze, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    result = audit.audit_identity(records, expected_commit=commit, expected_device_uuid=uuid,
                                  expected_freeze_sha256=digest(records['run_manifest']))
    assert name in result['controller_source']['files']


def test_accepts_a_committed_empty_python_source_file(frozen_identity):
    records, commit, uuid, _ = deepcopy(frozen_identity)
    controller = json.loads(records['controller_source']); name = 'looking_glass/empty.py'
    controller['files'][name] = digest(b'')
    records['controller_source'] = canonical(controller) + b'\n'
    base = json.loads(records['source_base'])
    base['repository_files'].append({'path': name, 'size': 0, 'sha256': digest(b'')})
    records['source_base'] = canonical(base)
    source = json.loads(records['source']); source['base'] = base; records['source'] = canonical(source)
    freeze = json.loads(records['run_manifest'])
    freeze['source']['sha256'] = digest(canonical(controller))
    freeze['source']['source_base_sha256'] = digest(records['source_base'])
    records['run_manifest'] = canonical(freeze) + b'\n'
    result = audit.audit_identity(records, expected_commit=commit, expected_device_uuid=uuid,
                                  expected_freeze_sha256=digest(records['run_manifest']))
    assert result['controller_source']['files'][name] == digest(b'')


def test_rehashes_each_dependency_origin_inside_the_selected_site(frozen_identity):
    records, _, _, _ = frozen_identity
    source = json.loads(records['source'])
    site = Path(source['dependency_origins'][0]['path']).parent
    assert hasattr(audit, 'verify_dependency_origins'), 'dependency origin verification is missing'
    assert audit.verify_dependency_origins(records['source'], site) == 6


@pytest.mark.parametrize('mutation', ['content', 'size', 'outside_site', 'missing'])
def test_rejects_changed_or_out_of_site_dependency_origin(frozen_identity, mutation, tmp_path):
    records, _, _, _ = deepcopy(frozen_identity)
    source = json.loads(records['source']); item = source['dependency_origins'][0]
    site = Path(item['path']).parent
    if mutation == 'content': Path(item['path']).write_bytes(b'changed')
    elif mutation == 'size': item['size'] += 1
    elif mutation == 'outside_site':
        outside = tmp_path / 'outside.py'; outside.write_bytes(b'fixture'); item['path'] = str(outside)
    elif mutation == 'missing': Path(item['path']).unlink()
    assert hasattr(audit, 'verify_dependency_origins'), 'dependency origin verification is missing'
    with pytest.raises(ValueError): audit.verify_dependency_origins(canonical(source), site)


@pytest.mark.parametrize('mutation', [
    'wrong_freeze_anchor', 'wrong_commit_anchor', 'wrong_device_anchor', 'runtime_uuid', 'runtime_version',
    'allocator_cap', 'numeric_mode', 'false_offline', 'extra_runtime', 'wrong_asset', 'missing_source', 'wrong_source_base',
    'origin_version', 'changed_workload',
])
def test_rejects_frozen_identity_mismatches(frozen_identity, mutation):
    records, commit, uuid, expected_key = deepcopy(frozen_identity)
    assert hasattr(audit, 'audit_identity'), 'independent source/runtime identity audit is missing'
    freeze_hash = digest(records['run_manifest'])
    if mutation == 'wrong_freeze_anchor': freeze_hash = 'f' * 64
    elif mutation == 'wrong_commit_anchor': commit = 'f' * 40
    elif mutation == 'wrong_device_anchor': uuid = 'GPU-00000000-0000-0000-0000-000000000002'
    elif mutation in ('runtime_uuid', 'runtime_version', 'allocator_cap', 'numeric_mode', 'false_offline', 'extra_runtime'):
        value = json.loads(records['runtime'])
        if mutation == 'runtime_uuid': value['device_uuid'] = 'GPU-00000000-0000-0000-0000-000000000002'
        elif mutation == 'runtime_version': value['torch_version'] = 'wrong'
        elif mutation == 'allocator_cap': value['allocator_limit_bytes'] += 1
        elif mutation == 'numeric_mode': value['deterministic_algorithms'] = False
        elif mutation == 'false_offline': value['effective_huggingface_settings']['hub_offline'] = False
        else: value['unreviewed_setting'] = 'unknown'
        records['runtime'] = canonical(value)
    elif mutation == 'wrong_asset':
        value = json.loads(records['assets']); value['assets'][0]['sha256'] = 'f' * 64
        records['assets'] = canonical(value)
    elif mutation == 'missing_source':
        value = json.loads(records['controller_source']); value['files'].pop('docs/AGREEMENT_GPU_CALIBRATION_V1.md')
        records['controller_source'] = canonical(value) + b'\n'
        freeze = json.loads(records['run_manifest']); freeze['source']['sha256'] = digest(canonical(value))
        records['run_manifest'] = canonical(freeze) + b'\n'; freeze_hash = digest(records['run_manifest'])
    elif mutation == 'wrong_source_base': records['source_base'] += b' '
    elif mutation == 'origin_version':
        value = json.loads(records['source']); value['dependency_origins'][0]['version'] = 'bad'
        records['source'] = canonical(value)
    elif mutation == 'changed_workload':
        freeze = json.loads(records['run_manifest']); freeze['workload']['target_input_ids'][1] += 1
        records['run_manifest'] = canonical(freeze) + b'\n'; freeze_hash = digest(records['run_manifest'])
    with pytest.raises(ValueError):
        audit.audit_identity(records, expected_commit=commit, expected_device_uuid=uuid,
                             expected_freeze_sha256=freeze_hash)
