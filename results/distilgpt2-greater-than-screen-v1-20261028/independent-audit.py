"""Independent saved-byte review of the public year screen; never runs a model."""
import collections
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ATTEMPT = Path(__file__).resolve().parent
REPO = Path(sys.argv[1]) if len(sys.argv)>1 else ATTEMPT.parents[1]
REV = '47d42b8137203fa8e5ffdc60942a1adece6c0feb'
TREE = 'a7766a55470650e37c9d1c0b4d69acf1ef953a4a'
CASE = 'experiments/distilgpt2_greater_than_screen_v1.json'
CASE_SHA = '399994789ee4d3eff27dd6c9cce23ae27ab7761e4bc33caf58de4528194b01d2'
PROTOCOL = 'docs/DISTILGPT2_GREATER_THAN_SCREEN_V1.md'
PROTOCOL_SHA = 'b64dcc877c4bcbf7f7cc97c64d691d906304336b9519e78ffa7fb03e6bcc52eb'
SOURCES = {'scripts/greater_than_screen_v1.py','looking_glass/greater_than_screen.py',
           'tests/test_greater_than_screen.py',CASE,PROTOCOL,'scripts/copying_head_eval_v1.py',
           'scripts/copying_dev_v1.py','scripts/distilgpt2_intervention_dev_bootstrap.py',
           'looking_glass/__init__.py','looking_glass/pretrained.py','looking_glass/copy_head_intervention.py'}

def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(path.read_bytes())
def check(test, name):
    if not test: raise AssertionError(name)
def same(a,b,name):
    if type(a) is dict and type(b) is dict:
        check(set(a)==set(b),name+' keys')
        for k in a: same(a[k],b[k],name+'.'+k)
    elif type(a) is list and type(b) is list:
        check(len(a)==len(b),name+' length')
        for i,(x,y) in enumerate(zip(a,b)): same(x,y,name+f'[{i}]')
    elif type(a) is float or type(b) is float:
        check(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
              and math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-14),f'{name}: {a} vs {b}')
    else: check(type(a) is type(b) and a==b,f'{name}: {a} vs {b}')
def f32(n):
    sign=-1 if n>>31 else 1
    e=(n>>23)&255; m=n&0x7fffff
    check(e!=255,'nonfinite retained score')
    return sign*math.ldexp(m if e==0 else m+(1<<23),-149 if e==0 else e-150)

run=ATTEMPT/'run'
receipt=read(ATTEMPT/'supervisor.json'); pre=read(ATTEMPT/'pre-run.json')
manifest=read(run/'manifest.json'); result=read(run/'result.json')
check(sha((ATTEMPT/'supervisor.json').read_bytes())=='469b61cf70916434d2e96779ba842ba1dd9e2c50be9fd511a76ac8d3894bcd48','observed receipt digest')
check(sha((run/'result.json').read_bytes())=='65e48fdd8bfdb2a69455037e5ad059731106793014c366b06896252bb35970ef','observed result digest')
check(receipt['process_exit_code']==0 and receipt['failure_file_present'] is False and not (run/'failure.json').exists(),'exit')
check(receipt['source_revision']==REV and receipt['source_tree']==TREE,'receipt source')
check(receipt['pre_run_sha256']==sha((ATTEMPT/'pre-run.json').read_bytes()),'pre-run digest')
check(pre['source_revision']==REV and pre['source_tree']==TREE,'pre-run source')
check(pre['case_sha256']==CASE_SHA and pre['protocol_sha256']==PROTOCOL_SHA,'pre-run case/protocol')
check(set(receipt['run_sha256'])=={'attempt.jsonl','last_logits.f32','manifest.json','result.json'},'run inventory')
for name,digest in receipt['run_sha256'].items(): check(sha((run/name).read_bytes())==digest,'receipt '+name)
for name in ('stdout','stderr'): check(sha((ATTEMPT/(name+'.txt')).read_bytes())==receipt[name+'_sha256'],name+' receipt')
check(not (ATTEMPT/'stderr.txt').read_bytes(),'stderr')
source=manifest['source']; check(source['revision']==REV and source['tree']==TREE and set(source['sha256'])==SOURCES,'source inventory')
for path,digest in source['sha256'].items():
    blob=subprocess.check_output(['git','show',REV+':'+path],cwd=REPO)
    check(sha(blob)==digest,'source '+path)
frozen_raw=subprocess.check_output(['git','show',REV+':'+CASE],cwd=REPO)
check(sha(frozen_raw)==CASE_SHA and source['sha256'][PROTOCOL]==PROTOCOL_SHA,'frozen hashes')
frozen=json.loads(frozen_raw)
same(manifest['cases'],frozen['cases'],'cases')
check(manifest['cases_sha256']==CASE_SHA and result['manifest_sha256']==sha((run/'manifest.json').read_bytes()),'manifest binding')
check(manifest['expected_forwards']==result['submitted_forwards']==result['model_call_returns_observed']==result['recorded_forwards']==48,'forward count')
check(result['status']=='complete' and result['model_state_sha256_before']==result['model_state_sha256_after']
      and result['model_buffers_sha256_before']==result['model_buffers_sha256_after'],'state')
checkpoint=result['checkpoint']
check(checkpoint['model_revision']=='2290a62682d06624634c1f46a6ad5be0f47f38aa'
      and checkpoint['weights_sha256']=='e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
      and checkpoint['asset_sha256']['tokenizer.json']=='8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6','checkpoint')
for k,v in {'python':'3.11.15','torch':'2.8.0+cu128','numpy':'2.4.6','transformers':'4.57.3','byteorder':'little',
            'gpt2_source_sha256':'b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80',
            'conv1d_source_sha256':'e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86'}.items():
    same(manifest['dependencies'][k],v,'dependency '+k)
for name in ('MKL_CBWR','ATEN_CPU_CAPABILITY'):
    same(manifest['cpu_backend_environment_at_import'][name],{'present':False,'value':None},'backend '+name)
runtime=result['runtime']
for k,v in {'device':'cpu','dtype':'float32','cuda_visibility':'-1','cuda_initialized':False,'cpu_autocast':False,
            'model_eval':True,'attention_implementation':'eager','use_cache':False,'intraop_threads':1,'guard_blocked_calls':[]}.items():
    same(runtime[k],v,'runtime '+k)
events=[json.loads(x) for x in (run/'attempt.jsonl').read_bytes().splitlines()]
check(len(events)==145,'event count'); same(events[0],{'event':'runtime_ready','actual_runtime':runtime},'runtime event')
raw=(run/'last_logits.f32').read_bytes(); check(len(raw)==48*50257*4,'raw size')
check(len(result['records'])==len(manifest['cases'])==48,'record count')
numeric=frozen['numeric_ids']; check(len(numeric)==100 and len(set(numeric))==100,'numeric IDs')
numerics=set(numeric); groups={}
for i,(case,record) in enumerate(zip(manifest['cases'],result['records'])):
    check(record['case_id']==case['case_id'],'case order')
    off=i*50257*4; chunk=raw[off:off+50257*4]
    meta={'offset':off,'bytes':50257*4,'sha256':sha(chunk)}
    same(meta,record['raw'],'raw descriptor')
    same(events[1+3*i],{'event':'start','case_id':case['case_id']},'start event')
    same(events[2+3*i],{'event':'raw_recorded','case_id':case['case_id'],'raw':meta},'raw event')
    same(events[3+3*i],dict(record,event='complete'),'completion event')
    y=case['year_suffix']; lo=case['lower_ids']; hi=case['upper_ids']; equal=case['equal_id']
    check(lo==[numeric[j] for j in range(y-8,y)] and hi==[numeric[j] for j in range(y+1,y+9)] and equal==numeric[y],'case candidate IDs')
    values=[f32(n) for n in struct.unpack('<50257I',chunk)]
    mx=max(values); weights=[math.exp(v-mx) for v in values]; z=math.fsum(weights)
    lower=math.fsum(weights[j] for j in lo); upper=math.fsum(weights[j] for j in hi)
    check(lower>0 and upper>0,'underflow in independent direct-sum method')
    aligned=math.fsum(weights[j] for j in numeric[1:]); zero=weights[numeric[0]]
    other=math.fsum(w for j,w in enumerate(weights) if j not in numerics)
    expected={'log_mass_ratio':math.log(upper/lower),'upper_conditional_fraction':upper/(upper+lower),
              'lower_full_mass':lower/z,'upper_full_mass':upper/z,'equal_full_mass':weights[equal]/z,
              'lower_logits':[values[j] for j in lo],'upper_logits':[values[j] for j in hi],
              'equal_logit':values[equal],'top_token_id':values.index(mx),
              'aligned_01_99_full_mass':aligned/z,'zero_00_next_token_mass':zero/z,'other_vocabulary_mass':other/z}
    same(expected,record['score'],'score '+case['case_id'])
    key=(y,case['event']); check(case['condition'] not in groups.setdefault(key,{}),'duplicate condition')
    groups[key][case['condition']]=expected['log_mass_ratio']
check(len(groups)==16 and all(set(x)=={'forward','reverse','neutral'} for x in groups.values()),'triples')
triples=[]
for y in frozen['years']:
    for event in frozen['events']:
        v=groups[y,event]; f,r,n=v['forward'],v['reverse'],v['neutral']
        triples.append({'year_suffix':y,'event':event,'forward':f,'reverse':r,'neutral':n,
                        'forward_minus_reverse':f-r,'forward_minus_neutral':f-n,'reverse_minus_neutral':r-n})
fields=('forward_minus_reverse','forward_minus_neutral','reverse_minus_neutral')
summary={'triples':triples,**{'mean_'+k:math.fsum(t[k] for t in triples)/16 for k in fields},
         'positive_forward_minus_reverse':sum(t['forward_minus_reverse']>0 for t in triples)}
same(summary,result['summary'],'triple summary')
print('PASS: published sources, direct receipt, 145 events, 48 full-vocabulary score vectors and 16 triples')
print('sign counts:',{condition:{'positive':sum(t[condition]>0 for t in triples),'negative':sum(t[condition]<0 for t in triples),'zero':sum(t[condition]==0 for t in triples)} for condition in ('forward','reverse','neutral','forward_minus_reverse')})
print(json.dumps(summary,indent=2,sort_keys=True))
