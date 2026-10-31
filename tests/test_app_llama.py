import json
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from test_app_runtime import tokenizer
from looking_glass.app_runtime import LocalModelRuntime


def checkpoint(path, kv=1, scaling=None):
    path.mkdir()
    config = LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=kv,
        max_position_embeddings=32,eos_token_id=None,rope_scaling=scaling)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval()
    model.save_pretrained(path,safe_serialization=True)
    tokenizer(path)
    return model


@pytest.mark.parametrize('kv',[1,2])
def test_local_llama_repeated_prefix_full_capture(tmp_path,kv):
    path=tmp_path/'llama'; checkpoint(path,kv)
    runtime=LocalModelRuntime()
    info=runtime.load(str(path),device='cpu')
    assert info['family']=='llama'
    result=runtime.run('hello world',{'max_new_tokens':2})
    assert len(result['events'])==12
    assert [x['input_token_ids'] for x in result['steps']][1]==result['steps'][0]['input_token_ids']+[result['steps'][0]['selected_token_id']]
    assert result['estimates']['capture_bytes_upper_bound']==6*(2+3)*8*4
    assert [tuple(x.value.shape) for x in result['events']]==[(1,2,8)]*6+[(1,3,8)]*6
    assert all(x.value.device.type=='cpu' for x in result['events'])
    assert runtime.run('hello world',{'max_new_tokens':2})['generated_token_ids']==result['generated_token_ids']
    runtime.unload()


def test_advanced_rope_rejected_without_retiring_demo(tmp_path):
    path=tmp_path/'llama'; checkpoint(path)
    file=path/'config.json'; data=json.loads(file.read_text())
    data['rope_scaling']={'rope_type':'linear','factor':2.0}; file.write_text(json.dumps(data))
    runtime=LocalModelRuntime(); runtime.load_demo(); binding=runtime.capture_binding(); old=runtime.model
    with pytest.raises(ValueError,match='RoPE'): runtime.load(str(path),device='cpu')
    assert runtime.model is old and runtime.capture_binding() is binding


def test_budget_before_model_forward_and_owner_preserved(tmp_path,monkeypatch):
    path=tmp_path/'llama'; checkpoint(path)
    runtime=LocalModelRuntime(); runtime.load(str(path),device='cpu'); binding=runtime.capture_binding()
    monkeypatch.setattr(runtime.model,'forward',lambda **kw: pytest.fail('forward must not run'))
    with pytest.raises(ValueError,match='budget'): runtime.run('hello world',{'max_trace_bytes':1})
    assert runtime.capture_binding() is binding
    assert runtime.info()['state']=='ready'


def test_forward_failure_quarantines(tmp_path,monkeypatch):
    import looking_glass.llama_trace as adapter
    path=tmp_path/'llama'; checkpoint(path)
    runtime=LocalModelRuntime(); runtime.load(str(path),device='cpu')
    def fail(*a,**kw): raise RuntimeError('injected execution failure')
    monkeypatch.setattr(adapter,'trace_llama_causal_lm',fail)
    with pytest.raises(RuntimeError,match='injected'): runtime.run('hello world',{'max_new_tokens':1})
    assert runtime.capture_binding() is None
    assert runtime.info()['state']=='quarantined'


def test_gpu_planning_counts_full_retention(tmp_path):
    path=tmp_path/'llama'; checkpoint(path)
    runtime=LocalModelRuntime(); runtime.load(str(path),device='cpu')
    opts=runtime._settings({'max_new_tokens':2})
    estimate=runtime._estimate(opts,2,2)
    assert estimate['estimated_gpu_capture_transient_bytes']==6*3*8*4
    assert estimate['capture_bytes_upper_bound']==6*5*8*4
    assert estimate['capture_events_upper_bound']==12


def test_prefix_keeps_original_mask(tmp_path,monkeypatch):
    path=tmp_path/'llama'; checkpoint(path)
    runtime=LocalModelRuntime(); runtime.load(str(path),device='cpu')
    monkeypatch.setattr(runtime,'_text_inputs',lambda *a:(torch.tensor([[0,5,6]]),torch.tensor([[0,1,1]])))
    result=runtime.run('padded hello world',{'max_new_tokens':2})
    assert result['steps'][0]['input_attention_mask']==[0,1,1]
    assert result['steps'][1]['input_attention_mask']==[0,1,1,1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA device required; CPU checks are not GPU validation')
@pytest.mark.parametrize('scaling',[None,dict(rope_type='llama3',factor=8.0,low_freq_factor=1.0,high_freq_factor=4.0,original_max_position_embeddings=8)])
def test_real_cuda_llama_full_logits_boundaries_offload_export_repeat(tmp_path,scaling):
    import math
    from collections import Counter
    from looking_glass.llama_trace import trace_llama_causal_lm
    from looking_glass.app_runtime import _cuda_operation
    from looking_glass.app_storage import TraceStore
    device=torch.device('cuda:0'); free,total=torch.cuda.mem_get_info(device)
    if free < max(256*1024**2,math.ceil(total*.1))+64*1024**2:
        pytest.skip('real GPU lacks free-memory/headroom prerequisite')
    path=tmp_path/'llama'; checkpoint(path,scaling=scaling)
    runtime=LocalModelRuntime(); previous=torch.cuda.get_per_process_memory_fraction(device)
    handles=[]; expected={}
    try:
        runtime.load(str(path),device=str(device),max_vram_bytes=1024**3)
        with _cuda_operation(device,1024**3),torch.inference_mode():
            encoded=runtime.tokenizer('hello world',return_tensors='pt')
            ids,mask=encoded['input_ids'].to(device),encoded['attention_mask'].to(device)
            with pytest.raises(ValueError,match='CPU is default'):
                trace_llama_causal_lm(runtime.model,ids,mask,run_id='reject')
            baseline=runtime.model(input_ids=ids,attention_mask=mask,use_cache=False).logits.cpu()
            for i in range(2):
                for suffix in ('.self_attn','.mlp',''):
                    name=f'model.layers.{i}{suffix}'
                    def hook(_m,_a,out,name=name):
                        expected[name]=(out[0] if isinstance(out,tuple) else out).detach().cpu().clone()
                    handles.append(runtime.model.get_submodule(name).register_forward_hook(hook))
            hooked=runtime.model(input_ids=ids,attention_mask=mask,use_cache=False).logits.cpu()
            assert torch.equal(baseline.view(torch.int32),hooked.view(torch.int32))
            for handle in handles: handle.remove()
            handles.clear()
            result=trace_llama_causal_lm(runtime.model,ids,mask,run_id='cuda',layers=(0,1),synchronous_cpu=False)
            assert torch.equal(baseline.view(torch.int32),result.output.logits.cpu().view(torch.int32))
            assert len(result.events)==6
            assert Counter(x.event.module_path for x in result.events)==Counter(expected)
            for record in result.events:
                assert torch.equal(record.value.cpu().view(torch.int32),expected[record.event.module_path].view(torch.int32))
                assert record.event.device=='cuda:0'
            del result,hooked
            initial_rotary=runtime.model.model.rotary_emb.inv_freq.view(torch.int32).cpu().clone()
            for length in (2,8,9,16,3):
                varied=torch.arange(length,device=device).remainder(16).unsqueeze(0)
                varied_mask=torch.ones_like(varied)
                plain=runtime.model(input_ids=varied,attention_mask=varied_mask,use_cache=False).logits.cpu()
                captured=trace_llama_causal_lm(runtime.model,varied,varied_mask,run_id=f'varied-{length}',
                    layers=(0,1),synchronous_cpu=False)
                assert torch.equal(plain.view(torch.int32),captured.output.logits.cpu().view(torch.int32))
                assert torch.equal(runtime.model.model.rotary_emb.inv_freq.view(torch.int32).cpu(),initial_rotary)
                assert runtime.model.config.rope_scaling==scaling
                del plain,captured
        result=runtime.run('hello world',{'max_new_tokens':1})
        assert result['generated_token_ids']==[int(baseline[0,-1].argmax())]
        assert result['gpu_memory']['synchronized']
        assert result['gpu_memory']['peak_reserved_bytes']<=result['gpu_memory']['effective_limit_bytes']
        for event in result['events']:
            md=event.metadata()
            assert md['device']=='cuda:0' and md['saved_device']=='cpu'
            assert torch.equal(event.value.view(torch.int32),expected[md['module_path']].view(torch.int32))
        store=TraceStore(tmp_path/'exports');store.publish(result)
        vectors=[[store.tensor(i,vector=j,limit=4096) for j in range(2)] for i in range(6)]
        saved=store.save('llama-cuda');store.clear();store.open(saved['path'])
        assert store.summary()['model']['config_summary']['rope_scaling']==scaling
        for i,rows in enumerate(vectors):
            for j,row in enumerate(rows):
                got=store.tensor(i,vector=j,limit=4096)
                assert got['raw_bits']==row['raw_bits'] and got['coordinate']==row['coordinate']
        again=runtime.run('hello world',{'max_new_tokens':2})
        assert again['coverage']['forward_count']==2
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())
        assert torch.cuda.get_per_process_memory_fraction(device)==previous
        runtime.unload()
        assert torch.cuda.get_per_process_memory_fraction(device)==previous
    finally:
        for h in handles: h.remove()
        if runtime.info() and runtime.info()['state']=='ready': runtime.unload()


def test_llama3_metadata_and_export_roundtrip(tmp_path):
    from looking_glass.app_storage import TraceStore
    scaling=dict(rope_type='llama3',factor=8.0,low_freq_factor=1.0,high_freq_factor=4.0,original_max_position_embeddings=8)
    path=tmp_path/'llama';checkpoint(path,scaling=scaling)
    runtime=LocalModelRuntime();info=runtime.load(str(path),device='cpu')
    assert info['config_summary']['rope_scaling']==scaling
    result=runtime.run('hello world',{'max_new_tokens':2})
    assert result['model']['config_summary']['rope_scaling']==scaling
    store=TraceStore(tmp_path/'exports');store.publish(result)
    saved=store.save('llama3');store.clear();store.open(saved['path'])
    assert store.summary()['model']['config_summary']['rope_scaling']==scaling
    runtime.unload()


def test_static_llama3_tied_sharded_local_checkpoint(tmp_path):
    scaling=dict(rope_type='llama3',factor=8.0,low_freq_factor=1.0,high_freq_factor=4.0,original_max_position_embeddings=8)
    path=tmp_path/'tied-sharded';path.mkdir()
    config=LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32,
        tie_word_embeddings=True,rope_scaling=scaling,eos_token_id=None)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    model.to(dtype=torch.bfloat16)  # Stored reduced precision; inference remains float32.
    expected={name:value.detach().float().clone() for name,value in model.state_dict().items()}
    model.save_pretrained(path,safe_serialization=True,max_shard_size='1KB')
    tokenizer(path)
    assert (path/'model.safetensors.index.json').is_file()
    import hashlib
    hashes={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in path.iterdir() if file.is_file()}
    runtime=LocalModelRuntime();runtime.load(str(path),device='cpu')
    assert runtime.model.lm_head.weight is runtime.model.model.embed_tokens.weight
    for name,value in runtime.model.state_dict().items():
        assert value.dtype==torch.float32
        assert torch.equal(value.view(torch.int32),expected[name].view(torch.int32))
    result=runtime.run('hello world',{'max_new_tokens':2})
    assert len(result['events'])==12
    assert result['model']['config_summary']['rope_scaling']==scaling
    assert hashes=={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in path.iterdir() if file.is_file()}
    runtime.unload()
