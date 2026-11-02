"""Offline CPU probability-view acceptance; exact pinned local assets, no downloads."""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import sys
import threading

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--export-root',type=Path,required=True)
parser.add_argument('--implementation-root',type=Path,default=Path(__file__).resolve().parents[2])
args=parser.parse_args()
if not __debug__:raise RuntimeError('Assertions required; do not use -O')
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
root=args.implementation_root.resolve(strict=True);sys.path.insert(0,str(root))
import torch
import transformers
from looking_glass import app,app_interventions
from looking_glass.model_ownership import ModelLease
assert all(Path(m.__file__).resolve().is_relative_to(root) for m in (app,app_interventions))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
record=Path(__file__).with_name('record.json').read_bytes()
assert hashlib.sha256(record).hexdigest()=='d2425a1bd3848f1378570be7262c138c883be076070dab533ad19b5a279cfe32'
source=json.loads(record)['source'];checkpoint=args.checkpoint.resolve(strict=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def assets():
    names={p.name for p in checkpoint.iterdir() if not(p.name=='.cache' and p.is_dir() and not p.is_symlink())}
    assert names==set(source['asset_manifest'])
    for name,row in source['asset_manifest'].items():
        p=checkpoint/name;assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
def bits(value):return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def weights(model):return {name:bits(value) for name,value in model.named_parameters()}
def distribution(row):
    # Independent scalar Python exp/fsum, not production Torch log_softmax.
    values=row.tolist();maximum=max(values)
    shifted=[value-maximum for value in values]
    exponentials=[math.exp(value) for value in shifted];total=math.fsum(exponentials)
    probabilities=[value/total for value in exponentials]
    log_probabilities=[value-math.log(total) for value in shifted]
    order=sorted(range(len(values)),key=lambda i:(-values[i],i))
    ranks={token:index+1 for index,token in enumerate(order)}
    assert abs(math.fsum(probabilities)-1)<5e-15
    return values,probabilities,log_probabilities,order,ranks
assets()
implementation={name:sha(root/'looking_glass'/name) for name in ('app.py','app_runtime.py','app_storage.py','app_interventions.py','llama_trace.py')}
state=app.AppState(export_root=args.export_root)
assert not state.store.export_root.exists(),'New export root required'
server=app.create_server(state);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
def call(path,body=None):
    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=120)
    headers={'X-MLG-Token':state.token}
    if body is not None:headers['Content-Type']='application/json'
    connection.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body),headers=headers)
    response=connection.getresponse();payload=response.read();code=response.status;connection.close()
    assert code==200,(path,code,payload[:500]);return json.loads(payload)
results=[]
try:
    call('/api/load',{'path':str(checkpoint),'device':'cpu'})
    run=call('/api/run',{'prompt':'Gravity is','settings':{'max_new_tokens':1}})
    assert run['intervention_available'] and run['steps'][0]['input_token_ids']==[22007,6463,314]
    capture=run['capture_id'];original=[bits(t) for t in state.store._values]
    model=state.runtime.model;original_weights=weights(model);rotary=bits(model.model.rotary_emb.inv_freq)
    prefix=run['steps'][0];ids=torch.tensor([prefix['input_token_ids']],dtype=torch.int64)
    mask=torch.tensor([prefix['input_attention_mask']],dtype=torch.int64)
    kwargs=dict(input_ids=ids,attention_mask=mask,use_cache=False,past_key_values=None,logits_to_keep=0,return_dict=True)
    module_path='model.layers.15.self_attn'
    rows=call('/api/events?q='+module_path+'&token_position=2&forward=0&capture_id='+capture)['events']
    event,=[row for row in rows if row['metadata']['module_path']==module_path]
    saved=[]
    for name,operation in [('zero_attention','zero'),('noop_attention','add')]:
        def hook(_module,_args,output):
            assert type(output) is tuple
            altered=output[0].clone()
            if operation=='zero':altered[0,2]=0
            else:altered[0,2,0]=altered[0,2,0]+0.
            return altered,*output[1:]
        with torch.inference_mode(),ModelLease(model):
            baseline=model(**kwargs).logits.detach().clone()
            handle=model.get_submodule(module_path).register_forward_hook(hook)
            try:edited=model(**kwargs).logits.detach().clone()
            finally:handle.remove()
        left=distribution(baseline[0,-1]);right=distribution(edited[0,-1])
        expected_ids=list(dict.fromkeys(left[3][:5]+right[3][:5]))
        expected_tv=.5*math.fsum(abs(p-q) for p,q in zip(left[1],right[1]))
        request=dict(capture_id=capture,event=event['index'],vector=2,operation=operation)
        if operation=='add':request.update(feature_index=0,value=0.)
        comparison=call('/api/intervene',request);actual=comparison['experiment']['next_token_distribution']
        assert actual['normalization']=='full vocabulary' and actual['computation_dtype']=='float64'
        assert actual['vocabulary_size']==49152 and actual['readout_position']==2
        assert [row['token_id'] for row in actual['rows']]==expected_ids
        assert math.isclose(actual['total_variation'],expected_tv,rel_tol=0,abs_tol=5e-15)
        for row in actual['rows']:
            i=row['token_id']
            assert row['baseline_rank']==left[4][i] and row['intervention_rank']==right[4][i]
            assert row['baseline_logit']==left[0][i] and row['intervention_logit']==right[0][i]
            for field,wanted in [('baseline_probability',left[1][i]),('intervention_probability',right[1][i]),
                                 ('probability_delta',right[1][i]-left[1][i])]:
                assert math.isclose(row[field],wanted,rel_tol=0,abs_tol=5e-15),(field,i,row[field],wanted)
            for field,wanted in [('baseline_log_probability',left[2][i]),('intervention_log_probability',right[2][i])]:
                assert math.isclose(row[field],wanted,rel_tol=0,abs_tol=5e-13)
        if operation=='add':
            assert actual['total_variation']==0 and all(row['probability_delta']==0 for row in actual['rows'])
            assert comparison['experiment']['full_logits']['bitwise_equal']
        else:assert actual['total_variation']>0
        assert math.fsum(row['baseline_probability'] for row in actual['rows'])<1
        assert call('/api/status')['run']==run and [bits(t) for t in state.store._values]==original
        assert bits(model.model.rotary_emb.inv_freq)==rotary
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        exported=call('/api/save-experiment',{'name':name,'experiment_id':comparison['experiment_id']})
        saved.append((exported['path'],actual))
        results.append({'name':name,'selected_token_before':int(baseline[0,-1].argmax()),'selected_token_after':int(edited[0,-1].argmax()),
            'independent_scalar_full_vocabulary_normalization_matches':True,'probability_abs_tolerance':5e-15,
            'log_probability_abs_tolerance':5e-13,'next_token_distribution':actual})
    assert weights(model)==original_weights
    call('/api/unload',{})
    for path,wanted in saved:
        reopened=call('/api/open',{'path':path})
        assert not reopened['intervention_available'] and reopened['experiment']['next_token_distribution']==wanted
    assets()
    assert implementation=={name:sha(root/'looking_glass'/name) for name in implementation}
    assert ids.tolist()==[prefix['input_token_ids']] and mask.tolist()==[prefix['input_attention_mask']]
    print(json.dumps({'schema':'mlg-smollm2-probability-view-v1','status':'passed','model':source['repository'],'revision':source['revision'],
        'device':'cpu','dtype':'float32','torch':torch.__version__,'transformers':transformers.__version__,
        'prompt':'Gravity is','input_ids':prefix['input_token_ids'],'attention_mask':prefix['input_attention_mask'],
        'readout_position':2,'target':{'module_path':module_path,'token_position':2,'token_id':314},
        'implementation_sha256':implementation,
        'script_sha256':sha(Path(__file__)),'checks':{'all_original_capture_hashes_unchanged':True,'all_model_parameter_hashes_unchanged':True,
            'rotary_bits_and_assets_unchanged':True,'probability_metadata_exact_after_save_reopen':True},
        'trials':results,'scope':'Two specified edits, one fixed three-token input. Normalized model token probabilities are not calibrated confidence, correctness, semantic explanations or quality results. No GPU execution.'},indent=2))
finally:
    server.shutdown();worker.join(5);server.server_close();assert not worker.is_alive()
