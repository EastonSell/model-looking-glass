"""CPU-only regression checks for the fixed serial owner's CUDA environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from looking_glass import trusted_profile_collector as collector
from looking_glass import trusted_service_profile as profile
from looking_glass.resource_profile import ProfilingSynchronizationError
from looking_glass.trusted_service_launch import prepare_trusted_launch


UUID = 'GPU-00000000-0000-0000-0000-000000000001'
REPO = Path(__file__).resolve().parents[1]


def _fixed_environment(home):
    return {'CUDA_VISIBLE_DEVICES': UUID, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'CUDA_MODULE_LOADING': 'LAZY', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'HF_HOME': str(home), 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}


def test_service_and_collector_launch_pin_lazy_module_loading(tmp_path):
    root = tmp_path.resolve()
    repo, site, private, home = root / 'repo', root / 'env' / 'site', root / 'private', root / 'hf'
    for directory in (repo, site, private, home):
        directory.mkdir(parents=True)
    config = private / 'config.json'
    config.write_text('{}', encoding='utf-8')
    for module in ('looking_glass.gpu_service_process', 'looking_glass.trusted_profile_collector'):
        launch = prepare_trusted_launch(repo_root=repo, site_packages=site, environment_root=site.parent,
            private_parent=private, hf_home=home, gpu_uuid=UUID, module=module, config_path=config,
            trusted_path=str(root), inherited_environment={'CUDA_MODULE_LOADING': 'EAGER'})
        assert launch.environment['CUDA_MODULE_LOADING'] == 'LAZY'


def test_runtime_requires_and_records_lazy_and_private_failure_names_key(tmp_path):
    home, cache = tmp_path.resolve() / 'hf', tmp_path.resolve() / 'hf' / 'hub'
    cache.mkdir(parents=True)
    fixed = _fixed_environment(home)
    with patch.dict(os.environ, fixed, clear=True):
        assert profile._checked_environment(UUID, str(cache))['CUDA_MODULE_LOADING'] == 'LAZY'
    for changed in (None, 'EAGER', 'lazy'):
        environment = dict(fixed)
        if changed is None:
            del environment['CUDA_MODULE_LOADING']
        else:
            environment['CUDA_MODULE_LOADING'] = changed
        with patch.dict(os.environ, environment, clear=True), pytest.raises(ValueError) as failure:
            profile._checked_environment(UUID, str(cache))
        diagnostic = collector._private_runtime_environment_failure(failure.value)
        assert diagnostic['key']['text'] == 'CUDA_MODULE_LOADING'
    with patch.dict(os.environ, {**fixed, 'CUDA_UNEXPECTED_PRIVATE_KEY': 'secret-value'}, clear=True), \
         pytest.raises(ValueError) as failure:
        profile._checked_environment(UUID, str(cache))
    wrapped = ProfilingSynchronizationError('completion unverified', original_error=failure.value)
    diagnostic = collector._private_runtime_environment_failure(wrapped)
    assert diagnostic == {'key': {'text': 'CUDA_UNEXPECTED_PRIVATE_KEY', 'truncated': False}}
    assert 'secret-value' not in json.dumps(diagnostic)
    with patch.dict(os.environ, {**fixed, 'CUDA_' + 'x' * 300: 'secret-value'}, clear=True), \
         pytest.raises(ValueError) as failure:
        profile._checked_environment(UUID, str(cache))
    diagnostic = collector._private_runtime_environment_failure(failure.value)
    assert len(diagnostic['key']['text'].encode('utf-8')) <= 128
    assert diagnostic['key']['truncated'] is True


def test_pinned_torch_lazy_init_keeps_environment_stable_without_cuda(tmp_path):
    home = tmp_path.resolve() / 'hf'
    (home / 'hub').mkdir(parents=True)
    code = r'''
import json, os, sys
from unittest.mock import patch
import torch
from looking_glass import trusted_service_profile as profile
environment = json.loads(sys.argv[1]); uuid = environment['CUDA_VISIBLE_DEVICES']
cache = sys.argv[2]
def stop_before_native(): raise RuntimeError('stopped before CUDA')
with patch.dict(os.environ, environment, clear=True):
    before = profile._checked_environment(uuid, cache)
    if not hasattr(torch._C, '_cuda_getDeviceCount'):
        # A CPU build must reject CUDA, not pretend it reached a native CUDA sentinel.
        try: torch.cuda._lazy_init()
        except AssertionError as error: assert 'not compiled with CUDA' in str(error)
        else: raise AssertionError('CPU-only build unexpectedly initialized CUDA')
        assert not torch.cuda.is_initialized()
    else:
        with patch.object(torch.cuda, 'is_initialized', return_value=False), \
             patch.object(torch.cuda, '_is_in_bad_fork', return_value=False), \
             patch.object(torch.cuda, '_cudart', object()), \
             patch.object(torch._C, '_cuda_init', side_effect=stop_before_native) as native:
            try: torch.cuda._lazy_init()
            except RuntimeError as error: assert str(error) == 'stopped before CUDA'
            else: raise AssertionError('native sentinel not reached')
            assert native.call_count == 1
    assert profile._checked_environment(uuid, cache) == before
    assert os.environ['CUDA_MODULE_LOADING'] == 'LAZY'
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-c', code, json.dumps(_fixed_environment(home)),
                             str(home / 'hub')], cwd=REPO, env=env, capture_output=True,
                            text=True, timeout=40)
    assert result.returncode == 0, result.stderr


def test_private_failure_file_captures_original_key_through_sync_wrapper(tmp_path):
    code = r'''
import json, pathlib, sys
from looking_glass import trusted_profile_collector as collector
from looking_glass import trusted_service_profile as profile
from looking_glass.resource_profile import ProfilingSynchronizationError
key = 'CUDA_' + 'x' * 300
original = profile._runtime_environment_error(key, 'unreviewed CUDA environment')
failure = ProfilingSynchronizationError('completion unverified', original_error=original)
def fail(): raise failure
collector._require_gated_supervisor = fail
assert collector._owner_main({'run_dir': sys.argv[1], 'run_id': 'test'}) == 1
record = json.loads((pathlib.Path(sys.argv[1]) / 'failure.json').read_text())
diagnostic = record['private_runtime_environment']
assert diagnostic['key']['text'] == key[:128] and diagnostic['key']['truncated'] is True
assert record['error_type'] == 'ProfilingSynchronizationError'
assert not (pathlib.Path(sys.argv[1]) / 'candidate.json').exists()
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)], cwd=REPO,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
