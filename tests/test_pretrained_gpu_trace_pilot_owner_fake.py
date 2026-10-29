"""Exercise the actual owner lifecycle with a tiny CPU model and fake CUDA."""

import gc
from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass import pretrained, pretrained_gpu_trace_pilot as owner
from looking_glass import pretrained_gpu_trace_pilot_runtime as runtime


def test_actual_measurement_lifecycle_with_tiny_cpu_model_and_fake_device(tmp_path, monkeypatch):
    session, lease = object(), object()
    roots = {'session': session, 'lease': lease, 'gc_disabled': True, 'claims_ready': False,
             'budget': None, 'resident': None, 'job': None, 'pending_return': None,
             'model': None, 'bundle': None, 'executor': None, 'stage': 'entry', 'cuda_pending': False}
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
    (tmp_path / 'stage').mkdir()
    was_enabled, old_collect = gc.isenabled(), gc.collect
    gc.disable()
    gc.collect = owner._forbid_collection
    try:
        result = owner._measure(session, lease, torch, tmp_path, tmp_path)
        assert result['forward_count'] == 10 and result['model_input_trace_weakrefs_dead'] is True
        assert result['active_memory_leases'] == 0 and result['cleanup']['reserved_bytes'] == 0
        result['runtime']['device'] = 'cpu'  # Only this fabricated executor used CPU tensors.
        assert owner._validate_saved(tmp_path, result, hidden=8, vocab=6767, device='cpu') == 10
    finally:
        gc.collect = old_collect
        if was_enabled:
            gc.enable()
