import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from looking_glass.llama_trace import trace_llama_causal_lm


def tiny(kv=2):
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=kv,
        max_position_embeddings=32, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    return LlamaForCausalLM(config).eval()


@pytest.mark.parametrize('kv', [1, 2])
def test_full_logits_and_all_boundary_bits(kv):
    model = tiny(kv)
    ids = torch.tensor([[1, 4, 5]])
    mask = torch.ones_like(ids)
    observed, handles = {}, []
    for i, layer in enumerate(model.model.layers):
        for role, module in [('attention', layer.self_attn), ('mlp', layer.mlp), ('block', layer)]:
            def hook(_m, _a, output, key=(i, role)):
                observed[key] = (output[0] if key[1] == 'attention' else output).detach().clone()
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    for h in handles: h.remove()
    result = trace_llama_causal_lm(model, ids, mask, run_id='test', layers=(0, 1))
    assert torch.equal(baseline.view(torch.int32), result.output.logits.view(torch.int32))
    assert len(result.events) == 6
    for record in result.events:
        assert torch.equal(record.value.view(torch.int32), observed[(record.event.layer_index, record.event.module_kind.removeprefix('llama_'))].view(torch.int32))
        assert record.event.saved_shape == (1, 3, 16)
    assert all(not m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('change', ['rope','sdpa','train','hook','root','buffer','geometry','alias'])
def test_execution_contract_rejects_before_forward(change):
    model=tiny(); ids=torch.tensor([[1,4]]); mask=torch.ones_like(ids)
    if change=='rope': model.config.rope_scaling={'rope_type':'linear','factor':2.0}
    elif change=='sdpa': model.config._attn_implementation='sdpa'
    elif change=='train': model.train()
    elif change=='hook': model.register_forward_hook(lambda *a: None)
    elif change=='root': model.forward=lambda **kw: pytest.fail('forward must not run')
    elif change=='buffer': model.register_buffer('bad',torch.ones(1,dtype=torch.float64))
    elif change=='geometry': model.model.layers[0].self_attn.q_proj=torch.nn.Linear(16,8,bias=False).eval()
    elif change=='alias': model.model.layers[1].mlp=model.model.layers[0].mlp
    with pytest.raises((ValueError,TypeError)):
        trace_llama_causal_lm(model,ids,mask,run_id='invalid',layers=(0,1))


@pytest.mark.parametrize('kwargs', [dict(layers=[[]]),dict(layers=(True,)),dict(layers=(0,0)),
    dict(modules=[{}]),dict(synchronous_cpu=1),dict(max_events=1),dict(max_trace_bytes=1),
    dict(max_tensor_bytes=1),dict(max_input_bytes=1),dict(max_logits_bytes=1)])
def test_bad_selectors_and_budgets(kwargs):
    model=tiny(); ids=torch.tensor([[1,4]])
    with pytest.raises(ValueError): trace_llama_causal_lm(model,ids,torch.ones_like(ids),run_id='invalid',**kwargs)
    assert all(not m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('kind',['dtype','device','range','mask','empty_row','shape','too_long'])
def test_input_guards(kind):
    model=tiny(); ids=torch.tensor([[1,4]]); mask=torch.ones_like(ids)
    if kind=='dtype': ids=ids.float()
    elif kind=='device': ids=ids.to('meta')
    elif kind=='range': ids[0,0]=32
    elif kind=='mask': mask[0,0]=2
    elif kind=='empty_row': mask.zero_()
    elif kind=='shape': mask=mask[:,:1]
    elif kind=='too_long': ids=torch.ones((1,33),dtype=torch.long); mask=torch.ones_like(ids)
    with pytest.raises(ValueError): trace_llama_causal_lm(model,ids,mask,run_id='invalid')


def test_autocast_and_integer_buffers():
    model=tiny(); model.register_buffer('ordinary_integer',torch.ones(1,dtype=torch.int64))
    ids=torch.tensor([[1,4]]); mask=torch.ones_like(ids)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='autocast'): trace_llama_causal_lm(model,ids,mask,run_id='invalid')
    result=trace_llama_causal_lm(model,ids,mask,run_id='valid',synchronous_cpu=False)
    assert result.tensor_bytes==3*2*16*4


def test_capture_isolation_and_source_unchanged():
    model=tiny(); ids=torch.tensor([[1,4],[3,6]]); mask=torch.tensor([[1,1],[0,1]])
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    ids_before,mask_before=ids.clone(),mask.clone()
    result=trace_llama_causal_lm(model,ids,mask,run_id='batch',layers=(1,0))
    assert [r.event.layer_index for r in result.events]==[0]*3+[1]*3
    assert result.events[0].event.batch_positions==(0,1)
    with torch.inference_mode(): result.events[0].value.zero_()
    assert torch.equal(ids,ids_before) and torch.equal(mask,mask_before)
    for k,v in model.state_dict().items(): assert torch.equal(v.view(torch.int32),before[k].view(torch.int32))
    repeat=trace_llama_causal_lm(model,ids,mask,run_id='repeat',layers=(0,1))
    assert bool(repeat.events[0].value.any())


def test_existing_lease_rejected():
    from looking_glass.model_ownership import ModelLease
    model=tiny(); ids=torch.tensor([[1,4]])
    with ModelLease(model):
        with pytest.raises(RuntimeError): trace_llama_causal_lm(model,ids,torch.ones_like(ids),run_id='owned')


@pytest.mark.parametrize('head_dim', [4,12])
def test_explicit_standard_head_dimension(head_dim):
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=head_dim,max_position_embeddings=32)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval(); ids=torch.tensor([[1,4]])
    result=trace_llama_causal_lm(model,ids,torch.ones_like(ids),run_id='head')
    assert result.output.logits.shape==(1,2,32)


@pytest.mark.parametrize('kind',['shape','dtype','device','nonfinite','cache'])
def test_rejects_invalid_returned_logits(monkeypatch,kind):
    # Class-local injected fault preserves the standard instance-forward identity
    # gate, so the adapter's post-forward contract is actually exercised.
    original=LlamaForCausalLM.forward
    def faulty(self,*a,**kw):
        result=original(self,*a,**kw)
        if kind=='shape': result.logits=result.logits[:,:1]
        elif kind=='dtype': result.logits=result.logits.double()
        elif kind=='device': result.logits=result.logits.to('meta')
        elif kind=='nonfinite': result.logits.fill_(float('nan'))
        else: result.past_key_values=object()
        return result
    model=tiny(); monkeypatch.setattr(LlamaForCausalLM,'forward',faulty)
    ids=torch.tensor([[1,4]])
    with pytest.raises(RuntimeError,match='logits'):
        trace_llama_causal_lm(model,ids,torch.ones_like(ids),run_id='fault')
    assert all(not m._forward_hooks for m in model.modules())


def test_detects_private_input_mutation_preserves_caller(monkeypatch):
    original=LlamaForCausalLM.forward
    def faulty(self,*a,**kw):
        result=original(self,*a,**kw); kw['input_ids'][0,0]=2; return result
    model=tiny(); monkeypatch.setattr(LlamaForCausalLM,'forward',faulty)
    ids=torch.tensor([[1,4]])
    with pytest.raises(RuntimeError,match='mutated'):
        trace_llama_causal_lm(model,ids,torch.ones_like(ids),run_id='fault')
    assert ids.tolist()==[[1,4]]
    assert all(not m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('key,value',[('rope_scaling',{}),('num_key_value_heads',3),('head_dim',3),
    ('pretraining_tp',True),('rope_theta',float('inf')),('hidden_size',0)])
def test_config_rejection(key,value):
    from looking_glass.llama_trace import validate_llama_config
    cfg=tiny().config.to_dict();cfg[key]=value
    with pytest.raises(ValueError):validate_llama_config(cfg)


def test_nullable_standard_head_defaults():
    from looking_glass.llama_trace import validate_llama_config
    cfg=tiny().config.to_dict();cfg['num_key_value_heads']=None;cfg['head_dim']=None
    validate_llama_config(cfg)
    # Validation is read-only; the actual standard config constructor applies defaults.
    assert cfg['head_dim'] is None and cfg['num_key_value_heads'] is None
