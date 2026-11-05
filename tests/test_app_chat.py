"""Cached conversation is deliberately separate from float32 trace capture."""
import json
from functools import wraps
import pytest
import torch
from looking_glass.app_runtime import LocalModelRuntime
from test_app_runtime import local_checkpoint


def loaded_chat(tmp_path, template=True, family="gpt2"):
    path = tmp_path / 'chat-model'
    if family == "llama":
        from test_app_llama import checkpoint
        checkpoint(path)
    else:
        local_checkpoint(path, family)
    runtime = LocalModelRuntime()
    runtime.load(str(path), device='cpu', mode='chat')
    if template:
        runtime.tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    return runtime


@pytest.mark.parametrize("family", ["gpt2", "llama"])
def test_chat_uses_cache_without_trace_hooks_and_matches_greedy(tmp_path, monkeypatch, family):
    runtime = loaded_chat(tmp_path, family=family)
    messages = [{'role':'user', 'content':'hello world'}]
    encoded = runtime.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
    ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    with torch.inference_mode():
        wanted = runtime.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=3, use_cache=False, pad_token_id=0)
    calls = []
    forward = runtime.model.forward
    @wraps(forward)
    def observe(*args, **kwargs):
        calls.append((kwargs.get('use_cache'), kwargs['input_ids'].shape[1]))
        return forward(*args, **kwargs)
    monkeypatch.setattr(runtime.model, 'forward', observe)
    result = runtime.chat(messages, {'max_new_tokens':3})
    assert result['generated_token_ids'] == wanted[0, ids.shape[1]:].tolist()
    assert result['format'] == 'chat_template'
    assert result['output_tokens'] == 3 and result['dropped_messages'] == 0
    assert calls[0][1] == ids.shape[1] and all(n == 1 for _, n in calls[1:])
    assert all(cache is True for cache, _ in calls)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())
    assert 'events' not in result and 'coverage' not in result
    assert runtime.capture_binding() is None
    json.dumps(result)


def test_chat_trims_only_complete_old_turns_and_preserves_system_and_latest(tmp_path):
    runtime = loaded_chat(tmp_path)
    messages = [{'role':'system','content':'the cat'},
        {'role':'user','content':'hello world '*4}, {'role':'assistant','content':'blue red '*4},
        {'role':'user','content':'the dog'}]
    result = runtime.chat(messages, {'max_new_tokens':2, 'context_tokens':16})
    assert result['dropped_messages'] == 2
    assert result['input_tokens'] + 2 <= 16
    assert messages[1]['content'] == 'hello world '*4
    with pytest.raises(ValueError, match='latest user|system'):
        runtime.chat([messages[0], {'role':'user','content':'hello '*30}], {'max_new_tokens':2, 'context_tokens':16})


def test_no_template_is_explicit_single_raw_completion(tmp_path):
    runtime = loaded_chat(tmp_path, template=False)
    result = runtime.chat([{'role':'user','content':'hello world'}], {'max_new_tokens':2})
    assert result['format'] == 'raw_completion'
    with pytest.raises(ValueError, match='chat template'):
        runtime.chat([{'role':'user','content':'hello'}, {'role':'assistant','content':'world'},
                      {'role':'user','content':'blue'}], {'max_new_tokens':2})


def test_mode_switch_requires_reload_and_invalidates_binding(tmp_path):
    runtime = loaded_chat(tmp_path)
    with pytest.raises(ValueError, match='Capture mode'):
        runtime.run('hello', {})
    path = tmp_path / 'chat-model'
    runtime.load(str(path), device='cpu', mode='capture')
    assert runtime.info()['dtype'] == 'float32' and runtime.capture_binding() is not None
    with pytest.raises(ValueError, match='Chat mode'):
        runtime.chat([{'role':'user','content':'hello'}], {})
    assert runtime.run('hello', {'max_new_tokens':1})['events']


@pytest.mark.parametrize('messages,settings', [([],{}),([{'role':'assistant','content':'hi'}],{}),
    ([{'role':'user','content':'hi'}], {'context_tokens':True}),
    ([{'role':'user','content':'hi'}], {'max_new_tokens':1025}),
    ([{'role':'user','content':'hi','extra':True}],{}),
    ([{'role':'user','content':'hi'}], {'unknown':2})])
def test_chat_rejects_invalid_requests_before_forward(tmp_path, monkeypatch, messages, settings):
    runtime = loaded_chat(tmp_path)
    monkeypatch.setattr(runtime.model, 'forward', lambda **kwargs: pytest.fail('forward must not run'))
    with pytest.raises(ValueError): runtime.chat(messages,settings)


def test_auto_device_selects_most_free_gpu(monkeypatch):
    monkeypatch.setattr(LocalModelRuntime, 'capabilities', staticmethod(lambda: {
        'cuda': {'available':True}, 'devices':[{'id':'cpu','available':True},
          {'id':'cuda:0','available':True,'free_bytes':2**30},
          {'id':'cuda:1','available':True,'free_bytes':5*2**30}]}))
    device, _ = LocalModelRuntime._select_device('auto','llama')
    assert str(device) == 'cuda:1'


def test_checkpoint_generation_options_do_not_expand_workload_or_retain_static_cache(tmp_path, monkeypatch):
    runtime = loaded_chat(tmp_path)
    runtime.model.generation_config.num_beams = 16
    runtime.model.generation_config.num_return_sequences = 16
    runtime.model.generation_config.cache_implementation = 'static'
    runtime.model.generation_config.min_new_tokens = 1000000
    runtime.model.generation_config.output_scores = True
    previous = runtime.model.generation_config
    policies = []
    generate = runtime.model.generate
    def observe(**kwargs):
        assert runtime.model.generation_config is kwargs['generation_config']
        policies.append(kwargs['generation_config'])
        return generate(**kwargs)
    monkeypatch.setattr(runtime.model, 'generate', observe)
    runtime.chat([{'role':'user','content':'hello world'}], {'max_new_tokens':2})
    assert runtime.model.generation_config is previous
    policy = policies[0]
    assert policy.num_beams == policy.num_return_sequences == 1
    assert policy.cache_implementation == 'dynamic' and not policy.output_scores
    assert policy.min_new_tokens == 0 and policy.max_new_tokens == 2
    assert getattr(runtime.model, '_cache', None) is None


def test_chat_forward_failure_quarantines_but_preflight_failure_does_not(tmp_path, monkeypatch):
    runtime = loaded_chat(tmp_path)
    with pytest.raises(ValueError): runtime.chat([], {})
    assert runtime.info()['state'] == 'ready'
    previous = runtime.model.generation_config
    def fail(**kwargs): raise RuntimeError('injected chat forward failure')
    monkeypatch.setattr(runtime.model, 'generate', fail)
    with pytest.raises(RuntimeError, match='injected'): runtime.chat([{'role':'user','content':'hello'}], {})
    assert runtime.info()['state'] == 'quarantined'
    assert runtime.model.generation_config is previous


def test_auto_mode_chooses_chat_but_other_families_keep_capture(tmp_path):
    for family in ('gpt2', 'distilbert'):
        path = tmp_path / family
        local_checkpoint(path, family)
        runtime = LocalModelRuntime()
        info = runtime.load(str(path), device='cpu', mode='auto')
        assert info['mode'] == ('chat' if family == 'gpt2' else 'capture')
        assert info['dtype'] == 'float32'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA required; CPU/static checks are not GPU validation')
@pytest.mark.parametrize('family', ['gpt2', 'llama'])
def test_real_cuda_chat_precision_cache_policy_and_repeated_turns(tmp_path, family):
    path = tmp_path / 'gpu-chat'
    if family == 'llama':
        from test_app_llama import checkpoint
        checkpoint(path)
    else:
        local_checkpoint(path, family)
    runtime = LocalModelRuntime()
    info = runtime.load(str(path), device='auto', mode='chat')
    device = next(runtime.model.parameters()).device
    assert device.type == 'cuda' and info['dtype'] in ('float16','bfloat16')
    prior_fraction = torch.cuda.get_per_process_memory_fraction(device)
    allocations = []
    for _ in range(3):
        result = runtime.chat([{'role':'user','content':'hello world'}], {'max_new_tokens':3})
        assert result['output_tokens'] == 3
        assert result['gpu_memory']['synchronized']
        assert all(torch.isfinite(p).all() for p in runtime.model.parameters())
        assert torch.cuda.get_per_process_memory_fraction(device) == prior_fraction
        assert getattr(runtime.model, '_cache', None) is None
        allocations.append(torch.cuda.memory_allocated(device))
    assert max(allocations) - min(allocations) <= 1024**2
    runtime.unload()


def test_empty_assistant_turn_after_eos_can_continue(tmp_path):
    runtime = loaded_chat(tmp_path)
    messages = [{'role':'user','content':'hello'}, {'role':'assistant','content':''},
                {'role':'user','content':'world'}]
    result = runtime.chat(messages, {'max_new_tokens':2})
    assert result['output_tokens'] == 2
    messages[1]['content'] = '   '
    assert runtime.chat(messages, {'max_new_tokens':2})['output_tokens'] == 2


@pytest.mark.parametrize('major,native,expected', [(7,True,torch.float16),(8,True,torch.bfloat16),(8,False,torch.float16)])
def test_cuda_dtype_policy_uses_native_support_on_selected_device(monkeypatch, major, native, expected):
    from contextlib import nullcontext
    from looking_glass.app_runtime import _load_dtype
    requested = []
    monkeypatch.setattr(torch.cuda, 'device', lambda device: requested.append(str(device)) or nullcontext())
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (major,0))
    def bf16(*, including_emulation):
        assert including_emulation is False
        return native
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', bf16)
    assert _load_dtype(torch.device('cuda:1'), 'chat') == expected
    assert requested == ['cuda:1']
    assert _load_dtype(torch.device('cuda:1'), 'capture') == torch.float32
    assert _load_dtype(torch.device('cpu'), 'chat') == torch.float32


def test_automatic_vram_budget_leaves_headroom_and_respects_server_cap(monkeypatch):
    import math
    import looking_glass.app_runtime as module
    snapshot = {'free_bytes':7*1024**3, 'total_bytes':8*1024**3, 'reserved_bytes':256*1024**2}
    monkeypatch.setattr(module, '_cuda_snapshot', lambda device: snapshot)
    expected = 7*1024**3 + 256*1024**2 - math.ceil(8*1024**3*.1)
    assert module._automatic_vram_allowance(torch.device('cuda:0'),64*1024**3) == expected
    assert module._automatic_vram_allowance(torch.device('cuda:0'),1024**3) == 1024**3
    snapshot.update(free_bytes=1,reserved_bytes=0)
    with pytest.raises(ValueError, match='headroom'):
        module._automatic_vram_allowance(torch.device('cuda:0'),64*1024**3)
