"""Actual tiny Llama fixed-prefix controls, not pretrained/GPU evidence."""
import copy
import pytest
import torch
from looking_glass.app_runtime import LocalModelRuntime
from looking_glass.app_storage import TraceStore
from test_app_llama import checkpoint

SCALING=dict(rope_type='llama3',factor=8.0,low_freq_factor=1.0,high_freq_factor=4.0,original_max_position_embeddings=8)


def fixture(tmp_path,kv=1,scaling=None,event=0,forward=0,device='cpu'):
    path=tmp_path/'model';checkpoint(path,kv=kv,scaling=scaling)
    runtime=LocalModelRuntime();runtime.load(str(path),device=device)
    result=runtime.run('hello world',{'max_new_tokens':2})
    store=TraceStore(tmp_path/'exports');store.publish(result)
    index=event+6*forward
    source=dict(coordinate=copy.deepcopy(store.tensor(index,vector=1)['coordinate']),
        step=copy.deepcopy(result['steps'][forward]),model=copy.deepcopy(result['model']),
        reference_value=result['events'][index].value[0,1].detach().clone())
    return runtime,store,source,path


@pytest.mark.parametrize('kv,scaling',[(1,None),(2,None),(1,SCALING),(2,SCALING)])
@pytest.mark.parametrize('event',[0,1,2])
@pytest.mark.parametrize('operation,feature,value',[('zero',None,None),('add',2,1.),('replace',2,.5),('add',2,0.)])
def test_llama_fixed_forward_controls(tmp_path,kv,scaling,event,operation,feature,value):
    runtime,store,source,_=fixture(tmp_path,kv,scaling,event)
    weights={k:v.detach().clone() for k,v in runtime.model.state_dict().items()}
    before=store.summary();reference=source['reference_value'].clone()
    result=runtime.intervene_fixed_forward(runtime.capture_binding(),source,operation,feature,value)
    assert all(result['experiment']['controls'].values())
    assert len(result['events'])==4
    expected=reference.clone()
    if operation=='zero':expected.zero_()
    elif operation=='add':expected[feature]+=value
    else:expected[feature]=value
    assert torch.equal(result['events'][-1].value.flatten().view(torch.int32),expected.view(torch.int32))
    assert store.summary()==before and torch.equal(source['reference_value'],reference)
    for k,v in runtime.model.state_dict().items():assert torch.equal(v.view(torch.int32),weights[k].view(torch.int32))
    comparison=TraceStore(tmp_path/'experiments');comparison.publish(result)
    assert all(comparison.tensor(i,vector=0)['coordinate']['status']=='known' for i in range(4))
    if value==0:assert result['experiment']['full_logits']['bitwise_equal']


def test_same_value_replace_and_later_prefix(tmp_path):
    runtime,_,source,_=fixture(tmp_path,scaling=SCALING,event=2,forward=1)
    result=runtime.intervene_fixed_forward(runtime.capture_binding(),source,'replace',3,float(source['reference_value'][3]))
    assert result['experiment']['full_logits']['bitwise_equal']
    assert len(result['experiment']['input_prefix'])==3
    assert not result['experiment']['comparison']['selected_vector_changed']


def test_masked_prefix_uses_saved_mask(tmp_path,monkeypatch):
    runtime,_,_,_=fixture(tmp_path)
    monkeypatch.setattr(runtime,'_text_inputs',lambda *a:(torch.tensor([[0,5,6]]),torch.tensor([[0,1,1]])))
    result=runtime.run('masked',{'max_new_tokens':2})
    store=TraceStore(tmp_path/'masked');store.publish(result)
    source=dict(coordinate=store.tensor(6,vector=2)['coordinate'],step=result['steps'][1],model=result['model'],
                reference_value=result['events'][6].value[0,2].clone())
    result=runtime.intervene_fixed_forward(runtime.capture_binding(),source,'add',0,0.)
    assert result['experiment']['input_attention_mask']==[0,1,1,1]
    assert result['experiment']['full_logits']['bitwise_equal']


@pytest.mark.parametrize('mutation',['rope','tree','hook','alias','budget','mask','path','numerics','nan','input_limit'])
def test_preflight_failures_preserve_owner(tmp_path,monkeypatch,mutation):
    runtime,_,source,_=fixture(tmp_path)
    maximum=128*1024**2; value=1.
    if mutation=='rope':runtime.model.config.rope_scaling={'rope_type':'dynamic','factor':2}
    elif mutation=='tree':runtime.model.model.layers[0].self_attn.q_proj=torch.nn.Linear(8,4).eval()
    elif mutation=='hook':runtime.model.register_forward_hook(lambda *a:None)
    elif mutation=='alias':runtime.model.model.layers[1].mlp=runtime.model.model.layers[0].mlp
    elif mutation=='budget':maximum=1
    elif mutation=='mask':source['step'].pop('input_attention_mask')
    elif mutation=='path':source['coordinate']['module_path']='model.embed_tokens'
    elif mutation=='numerics':source['model']['execution_numerics']={}
    elif mutation=='nan':value=float('nan')
    else:source['step']['input_token_ids']=[1]*65;source['step']['input_attention_mask']=[1]*65
    import looking_glass.app_interventions as helper
    monkeypatch.setattr(helper,'execute',lambda *a:pytest.fail('must reject before work'))
    binding=runtime.capture_binding()
    with pytest.raises(ValueError):runtime.intervene_fixed_forward(binding,source,'add',1,value,maximum)
    assert runtime.capture_binding() is binding and runtime.info()['state']=='ready'


def test_stale_or_imported_binding_and_source_drift(tmp_path):
    runtime,store,source,path=fixture(tmp_path)
    binding=runtime.capture_binding()
    for invalid in (None,object()):
        with pytest.raises(ValueError):runtime.intervene_fixed_forward(invalid,source,'zero')
    runtime.load(str(path),device='cpu')
    with pytest.raises(ValueError):runtime.intervene_fixed_forward(binding,source,'zero')
    before=store.summary();source['reference_value'][0]+=1
    with pytest.raises(ValueError,match='original source'):
        runtime.intervene_fixed_forward(runtime.capture_binding(),source,'zero')
    assert runtime.info()['state']=='quarantined' and runtime.capture_binding() is None
    assert store.summary()==before


@pytest.mark.parametrize('trial',[1,2,3,4])
@pytest.mark.parametrize('fault',['rotary_bits','rotary_replace','rotary_config','rotary_callable','input','mask','cache','device'])
def test_each_trial_runtime_fault_quarantines(tmp_path,monkeypatch,trial,fault):
    runtime,store,source,_=fixture(tmp_path,scaling=SCALING)
    original=runtime.model.forward;calls=[];saved=store.summary()
    def changed(*a,**kw):
        output=original(*a,**kw);calls.append(None)
        if len(calls)==trial:
            rotary=runtime.model.model.rotary_emb
            if fault=='rotary_bits':rotary.inv_freq.view(torch.int32)[0]+=1
            elif fault=='rotary_replace':rotary.inv_freq=rotary.inv_freq.clone()
            elif fault=='rotary_config':runtime.model.config.rope_scaling['factor']=2.
            elif fault=='rotary_callable':rotary.rope_init_fn=lambda *a:None
            elif fault=='input':kw['input_ids'][0,0]=2
            elif fault=='mask':kw['attention_mask'][0,0]=0
            elif fault=='cache':output.past_key_values=object()
            else:output.logits=output.logits.to('meta')
        return output
    import looking_glass.app_interventions as helper
    execute=helper.execute
    def after_preflight(owner,request):
        monkeypatch.setattr(runtime.model,'forward',changed)
        return execute(owner,request)
    monkeypatch.setattr(helper,'execute',after_preflight)
    with pytest.raises((ValueError,RuntimeError)):runtime.intervene_fixed_forward(runtime.capture_binding(),source,'zero')
    assert runtime.info()['state']=='quarantined' and runtime.capture_binding() is None
    assert store.summary()==saved
    assert len(calls)==trial


@pytest.mark.parametrize('trial',[2,4])
def test_null_restoration_logits_mismatch(tmp_path,monkeypatch,trial):
    runtime,_,source,_=fixture(tmp_path)
    original=runtime.model.forward;calls=[]
    def changed(*a,**kw):
        output=original(*a,**kw);calls.append(None)
        if len(calls)==trial:output.logits=output.logits+1
        return output
    import looking_glass.app_interventions as helper
    execute=helper.execute
    def after_preflight(owner,request):
        monkeypatch.setattr(runtime.model,'forward',changed);return execute(owner,request)
    monkeypatch.setattr(helper,'execute',after_preflight)
    with pytest.raises(ValueError,match='full logits'):runtime.intervene_fixed_forward(runtime.capture_binding(),source,'zero')
    assert runtime.capture_binding() is None


@pytest.mark.parametrize('operation',['add','replace'])
def test_signed_zero_other_feature_bits_preserved(tmp_path,monkeypatch,operation):
    from transformers.models.llama.modeling_llama import LlamaMLP
    original=LlamaMLP.forward
    def signed_zero(self,*a,**kw):
        output=original(self,*a,**kw).clone()
        output.view(torch.int32)[...,0]=-2147483648
        return output
    # Explicit test-only class-level output seam supplies otherwise rare encodings;
    # actual model/plan/null/restoration forwards and all controls still execute.
    monkeypatch.setattr(LlamaMLP,'forward',signed_zero)
    runtime,_,source,_=fixture(tmp_path,event=1)
    assert int(source['reference_value'].view(torch.int32)[0])==-2147483648
    result=runtime.intervene_fixed_forward(runtime.capture_binding(),source,operation,2,.5)
    assert all(int(event.value.view(torch.int32)[0,0,0])==-2147483648 for event in result['events'])
    assert result['experiment']['controls']['other_features_at_selected_vector_bitwise_equal']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA required; CPU/mocks do not validate GPU interventions')
@pytest.mark.parametrize('scaling',[None,SCALING])
def test_real_cuda_llama_intervention_controls(tmp_path,scaling):
    import math
    device=torch.device('cuda:0');free,total=torch.cuda.mem_get_info(device)
    if free < max(256*1024**2,math.ceil(total*.1))+64*1024**2:
        pytest.skip('GPU free-memory/headroom prerequisite unavailable')
    runtime,store,_,_=fixture(tmp_path,scaling=scaling,device=str(device))
    previous=torch.cuda.get_per_process_memory_fraction(device)
    before=store.summary()
    try:
        for site in range(3):
            source=store.intervention_source(site,1)
            for operation,feature,value in [('zero',None,None),('add',1,.25),('replace',1,.5),('add',1,0.)]:
                result=runtime.intervene_fixed_forward(runtime.capture_binding(),source,operation,feature,value)
                assert all(result['experiment']['controls'].values())
                assert result['gpu_memory']['synchronized']
                assert result['gpu_memory']['peak_reserved_bytes']<=result['gpu_memory']['effective_limit_bytes']
                assert all(e.value.device.type=='cpu' and e.metadata()['device']=='cuda:0'
                    and e.metadata()['saved_device']=='cpu' for e in result['events'])
                assert torch.cuda.get_per_process_memory_fraction(device)==previous
                assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())
                if value==0.:assert result['experiment']['full_logits']['bitwise_equal']
        assert store.summary()==before
        runtime.unload();assert torch.cuda.get_per_process_memory_fraction(device)==previous
    finally:
        if runtime.info() and runtime.info()['state']=='ready':runtime.unload()


def test_zero_allowance_rejects_before_device_validation(tmp_path,monkeypatch):
    runtime,_,source,_=fixture(tmp_path)
    import looking_glass.llama_trace as adapter
    monkeypatch.setattr(adapter,'validate_llama_model',lambda *a,**kw:pytest.fail('no GPU/model validation work with no allowance'))
    with pytest.raises(ValueError,match='positive bounded'):runtime.intervene_fixed_forward(runtime.capture_binding(),source,'zero',max_pair_bytes=0)


def test_saved_prefix_context_limit(tmp_path,monkeypatch):
    runtime,_,source,_=fixture(tmp_path)
    source['step']['input_token_ids']=[1]*33;source['step']['input_attention_mask']=[1]*33
    import looking_glass.app_interventions as helper
    monkeypatch.setattr(helper,'execute',lambda *a:pytest.fail('context rejection must precede forward'))
    with pytest.raises(ValueError,match='exact valid token'):runtime.intervene_fixed_forward(runtime.capture_binding(),source,'zero')


def test_full_logit_sixteen_mib_limit_before_forward(tmp_path,monkeypatch):
    from transformers import LlamaConfig,LlamaForCausalLM
    from test_app_runtime import tokenizer
    path=tmp_path/'large-vocabulary';path.mkdir()
    config=LlamaConfig(vocab_size=65537,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,eos_token_id=None)
    config._attn_implementation='eager'
    LlamaForCausalLM(config).eval().save_pretrained(path,safe_serialization=True);tokenizer(path)
    runtime=LocalModelRuntime();runtime.load(str(path),device='cpu')
    result=runtime.run('hello world',{'max_new_tokens':1});store=TraceStore(tmp_path/'store');store.publish(result)
    source=dict(coordinate=store.tensor(0,vector=1)['coordinate'],step=copy.deepcopy(result['steps'][0]),
        model=result['model'],reference_value=result['events'][0].value[0,1].clone())
    source['step']['input_token_ids']=[1]*64;source['step']['input_attention_mask']=[1]*64;source['coordinate']['token_id']=1
    import looking_glass.app_interventions as helper
    monkeypatch.setattr(helper,'execute',lambda *a:pytest.fail('logit budget must reject before forward'))
    with pytest.raises(ValueError,match='16 MiB'):runtime.intervene_fixed_forward(runtime.capture_binding(),source,'zero')
    assert runtime.info()['state']=='ready'
