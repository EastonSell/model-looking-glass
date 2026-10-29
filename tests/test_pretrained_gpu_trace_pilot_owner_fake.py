"""Exercise the actual owner lifecycle with a tiny CPU model and fake CUDA."""

import gc
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass import pretrained, pretrained_gpu_trace_pilot as owner
from looking_glass import pretrained_gpu_trace_pilot_runtime as runtime


@pytest.mark.parametrize('entry', [False, True])
def test_actual_measurement_lifecycle_with_tiny_cpu_model_and_fake_device(tmp_path, monkeypatch, entry):
    session = SimpleNamespace(identity=SimpleNamespace(workspace_root=str(tmp_path)), run_id='fake-supervisor',
                              device_lease_path=tmp_path / 'fake-device.lock')
    lease = object()
    roots = {'session': session, 'lease': lease, 'gc_disabled': True, 'claims_ready': False,
             'budget': None, 'resident': None, 'job': None, 'pending_return': None,
             'model': None, 'bundle': None, 'executor': None, 'stage': 'entry', 'cuda_pending': False,
             'sync_pending': False, 'entered': False, 'failure': None, 'diagnostic_failure': None}
    monkeypatch.setattr(owner, '_ROOT', roots)
    monkeypatch.setattr(owner.OwnerSession, 'check', lambda *_a: None)
    monkeypatch.setattr(owner.OwnerSession, 'verify_runtime_binding', lambda *_a, **_k: None)
    monkeypatch.setattr(owner, '_runtime_uuid', lambda *_a: 'GPU-fake')
    monkeypatch.setattr(owner, '_verify_model', lambda *_a, **_k: None)
    monkeypatch.setattr(owner, '_host_memory', lambda: {'fake': True})
    monkeypatch.setenv('LOOKING_GLASS_TRACE_PILOT_SMI_TOTAL', str(8 * owner.GIB))
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda *_a: None)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 1)
    for name, value in {'is_initialized': lambda: False, 'device_count': lambda: 1,
            'get_device_properties': lambda *_a: SimpleNamespace(uuid='fake', total_memory=8 * owner.GIB,
                name='fake', major=7, minor=5), 'get_allocator_backend': lambda: 'native',
            'set_per_process_memory_fraction': lambda *_a: None,
            'get_per_process_memory_fraction': lambda *_a: owner.CAP / (8 * owner.GIB),
            'synchronize': lambda *_a: None, 'reset_peak_memory_stats': lambda *_a: None,
            'mem_get_info': lambda *_a: (7 * owner.GIB, 8 * owner.GIB),
            'memory_allocated': lambda *_a: 0, 'memory_reserved': lambda *_a: 0,
            'max_memory_allocated': lambda *_a: 0, 'max_memory_reserved': lambda *_a: 0,
            'manual_seed': lambda *_a: None, 'empty_cache': lambda *_a: None}.items():
        monkeypatch.setattr(torch.cuda, name, value)
    monkeypatch.setattr(torch._C, '_cuda_clearCublasWorkspaces', lambda: None)
    original_tensor_to, original_module_to = torch.Tensor.to, torch.nn.Module.to
    def tensor_to(value, *args, **kwargs):
        if args and args[0] == 'cuda:0':
            return original_tensor_to(value, 'cpu', copy=True)
        return original_tensor_to(value, *args, **kwargs)
    def module_to(value, *args, **kwargs):
        if args and args[0] == 'cuda:0':
            return value
        return original_module_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', tensor_to)
    monkeypatch.setattr(torch.nn.Module, 'to', module_to)
    def load(**_kwargs):
        config = GPT2Config(vocab_size=6767, n_positions=16, n_embd=8, n_layer=6, n_head=2,
                            use_cache=False, _attn_implementation='eager')
        return SimpleNamespace(model=GPT2LMHeadModel(config).eval(), metadata={
            'model_id': pretrained.MODEL_ID, 'model_revision': pretrained.REVISION,
            'tokenizer_id': pretrained.MODEL_ID, 'tokenizer_revision': pretrained.REVISION,
            'parameter_count': 81912576, 'weights_sha256': pretrained.ASSETS['model.safetensors'][1],
            'asset_sha256': {name: digest for name, (_, digest) in pretrained.ASSETS.items()},
            'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
            'dtype': 'float32', 'attention_implementation': 'eager', 'use_cache': False})
    monkeypatch.setattr(pretrained, 'load_distilgpt2', load)
    original_executor = runtime.PilotExecutor
    class CpuExecutor(original_executor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, device='cpu', vocab_size=6767)
    monkeypatch.setattr(runtime, 'PilotExecutor', CpuExecutor)
    attested = []
    if entry:
        from copy import deepcopy
        commit = 'a' * 40
        monkeypatch.setattr(owner, '_startup', lambda: None)
        monkeypatch.setattr(owner, '_pristine', lambda: None)
        monkeypatch.setattr(owner.OwnerSession, 'from_environment', lambda: session)
        monkeypatch.setattr(owner.OwnerSession, 'attest_clean', lambda *_a: attested.append(True))
        monkeypatch.setattr(owner.DeviceLease, 'acquire', lambda *_a, **_k: lease)
        monkeypatch.setattr(owner, '_validate_workspace_spec', lambda *_a: None)
        monkeypatch.setattr(owner, '_validate_workspace_binding', lambda *_a: None)
        monkeypatch.setattr(owner, '_validate_runtime_dirs', lambda *_a: None)
        monkeypatch.setattr(owner, '_workspace_root', lambda *_a: tmp_path)
        monkeypatch.setattr(owner, '_cache_path', lambda *_a: tmp_path)
        monkeypatch.setattr(owner, '_source_identity', lambda *_a: (commit, {}))
        monkeypatch.setattr(owner, '_verify_offline_cache', lambda *_a: None)
        original_validate = owner._validate_saved
        def validate(directory, measurement):
            adjusted = deepcopy(measurement)
            adjusted['runtime']['device'] = 'cpu'
            return original_validate(directory, adjusted, hidden=8, vocab=6767, device='cpu')
        monkeypatch.setattr(owner, '_validate_saved', validate)
    (tmp_path / 'stage').mkdir()
    was_enabled, old_collect = gc.isenabled(), gc.collect
    gc.disable()
    gc.collect = owner._forbid_collection
    try:
        if entry:
            code = owner.main(['--owner', str(tmp_path), str(tmp_path), 'fake-run', commit, 'fake-git'])
            failure = tmp_path / 'failure.json'
            assert code == 0, failure.read_text() if failure.exists() else 'missing candidate'
            result = json.loads((tmp_path / 'candidate.json').read_bytes())['measurement']
            assert attested == [True] and (tmp_path / 'final.json').is_file()
        else:
            result = owner._measure(session, lease, torch, tmp_path, str(tmp_path))
        assert result['forward_count'] == 10 and result['model_input_trace_weakrefs_dead'] is True
        assert result['active_memory_leases'] == 0 and result['cleanup']['reserved_bytes'] == 0
        if not entry:
            result['runtime']['device'] = 'cpu'  # Only this fabricated executor used CPU tensors.
            assert owner._validate_saved(tmp_path, result, hidden=8, vocab=6767, device='cpu') == 10
    finally:
        gc.collect = old_collect
        if was_enabled:
            gc.enable()
