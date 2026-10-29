"""Rebuild recipient screen scores and fixed gate from saved bytes, no model."""
import hashlib, json, math, pathlib, struct, subprocess
from collections import defaultdict

ATTEMPT=pathlib.Path(__file__).resolve().parent
RUN=ATTEMPT/'run'; REPO=ATTEMPT.parents[1]
REV='32164cd570dea1f4bc0022c5b7f81fe17f8c2797'
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_bytes())
def check(ok, what):
    if not ok: raise AssertionError(what)
def same(a,b,what):
    if isinstance(a,dict) and isinstance(b,dict):
        check(set(a)==set(b),what+' keys')
        for k in a: same(a[k],b[k],what+'.'+k)
    elif isinstance(a,list) and isinstance(b,list):
        check(len(a)==len(b),what+' length')
        for i,(x,y) in enumerate(zip(a,b)): same(x,y,what+' '+str(i))
    elif isinstance(a,float) or isinstance(b,float):
        check(isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),f'{what}: {a} vs {b}')
    else: check(type(a) is type(b) and a==b,f'{what}: {a} vs {b}')
def f32(bits):
    sign=-1.0 if bits>>31 else 1.0; e=(bits>>23)&255; m=bits&0x7fffff
    check(e!=255,'nonfinite float')
    return sign*(math.ldexp(m,-149) if e==0 else math.ldexp((1<<23)|m,e-150))

receipt=read(ATTEMPT/'supervisor.json'); result=read(RUN/'result.json'); manifest=read(RUN/'manifest.json')
check(sha((ATTEMPT/'supervisor.json').read_bytes())=='fbf7e2f93074c502e13ebb2ceac53547985d6952070c98d6d09a457fbe71b9cf','receipt hash')
check(sha((RUN/'result.json').read_bytes())=='3848582148a04d027012e862004c783152bc18255d783f028325f7c68be9b344','result hash')
check(receipt['process_exit_code']==0 and receipt['failure_file_present'] is False and not (RUN/'failure.json').exists(),'exit')
check(receipt['source_revision']==REV and receipt['source_tree']=='7c74335e9aa96fc716da062477ab8f5f28c8ad9c','receipt source')
check(receipt['pre_run_sha256']==sha((ATTEMPT/'pre-run.json').read_bytes()),'pre-run')
for name,expected in receipt['run_sha256'].items(): check(sha((RUN/name).read_bytes())==expected,'receipt '+name)
for name in ('stdout','stderr'): check(sha((ATTEMPT/(name+'.txt')).read_bytes())==receipt[name+'_sha256'],name+' hash')
check(not (ATTEMPT/'stderr.txt').read_bytes(),'stderr')
source=manifest['source']; check(source['revision']==REV and source['tree']==receipt['source_tree'] and len(source['sha256'])==11,'source identity')
for path,digest in source['sha256'].items():
    blob=subprocess.check_output(['git','show',REV+':'+path],cwd=REPO)
    check(sha(blob)==digest,'source '+path)
frozen=json.loads(subprocess.check_output(['git','show',REV+':experiments/distilgpt2_recipient_screen_v1.json'],cwd=REPO))
same(manifest['cases'],frozen['cases'],'cases')
check(manifest['cases_sha256']=='df9036ece115a42851a43f2d7833865fa1258ab542e322beff16d549df27d345','cases sha')
check(result['manifest_sha256']==sha((RUN/'manifest.json').read_bytes()),'manifest binding')
check(result['status']=='complete' and result['submitted_forwards']==result['model_call_returns_observed']==result['recorded_forwards']==64,'forward count')
check(result['model_state_sha256_before']==result['model_state_sha256_after'] and result['model_buffers_sha256_before']==result['model_buffers_sha256_after'],'state')
check(result['checkpoint']['model_revision']=='2290a62682d06624634c1f46a6ad5be0f47f38aa' and result['checkpoint']['weights_sha256']=='e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b','checkpoint')
runtime=result['runtime']
for k,v in {'device':'cpu','dtype':'float32','cuda_visibility':'-1','cuda_initialized':False,'cpu_autocast':False,'model_eval':True,'attention_implementation':'eager','use_cache':False,'intraop_threads':1,'guard_blocked_calls':[]}.items(): same(runtime[k],v,'runtime '+k)
events=[json.loads(x) for x in (RUN/'attempt.jsonl').read_bytes().splitlines()]; check(len(events)==193,'event count')
same(events[0],{'event':'runtime_ready','actual_runtime':runtime},'runtime event')
raw=(RUN/'last_logits.f32').read_bytes(); check(len(raw)==64*50257*4,'raw size')
observed=[]
for i,(case,record) in enumerate(zip(manifest['cases'],result['records'])):
    check(case['case_id']==record['case_id'],'case order')
    off=i*50257*4; chunk=raw[off:off+50257*4]
    meta={'offset':off,'bytes':50257*4,'sha256':sha(chunk)}; same(meta,record['raw'],'raw descriptor')
    same(events[1+3*i],{'event':'start','case_id':case['case_id']},'start')
    same(events[2+3*i],{'event':'raw_recorded','case_id':case['case_id'],'raw':meta},'raw event')
    same(events[3+3*i],dict(record,event='complete'),'complete')
    values=[f32(x) for x in struct.unpack('<50257I',chunk)]; mx=max(values); z=math.log(math.fsum(math.exp(x-mx) for x in values))
    target=case['recipient_id']; giver=case['giver_id']
    def rank(index): return 1+sum(x>values[index] or (x==values[index] and j<index) for j,x in enumerate(values))
    score={'recipient_logit':values[target],'giver_logit':values[giver],'margin':values[target]-values[giver],
           'recipient_log_probability':(values[target]-mx)-z,'giver_log_probability':(values[giver]-mx)-z,
           'recipient_rank':rank(target),'giver_rank':rank(giver),'top_token_id':values.index(mx)}
    same(score,record['score'],'score '+case['case_id']); observed.append((case,score))
check(len(observed)==64,'case count')
groups=defaultdict(list)
for c,s in observed: groups[(c['pair'],c['template'],c['order'])].append(s['margin'])
check(len(groups)==32 and all(len(x)==2 for x in groups.values()),'pair grouping')
paired={key:all(v>0 for v in vs) for key,vs in groups.items()}
bt=[sum(paired[p,t,o] for p in range(8) for o in (0,1)) for t in (0,1)]
bo=[sum(paired[p,t,o] for p in range(8) for t in (0,1)) for o in (0,1)]
bn=[sum(paired[p,t,o] for t in (0,1) for o in (0,1)) for p in range(8)]
summary={'positive_cases':sum(s['margin']>0 for c,s in observed),'both_positive_pairs':sum(paired.values()),'template_both_positive':bt,'order_both_positive':bo,'name_pair_both_positive':bn,'name_pairs_meeting_requirement':sum(x>=3 for x in bn),'mean_margin':math.fsum(s['margin'] for c,s in observed)/64,'gate_passed':sum(paired.values())>=24 and min(bt)>=12 and sum(x>=3 for x in bn)>=6}
same(summary,result['summary'],'summary')
for t in (0,1):
    rows=[s['margin'] for c,s in observed if c['template']==t]
    print('TEMPLATE',t,'positive',sum(x>0 for x in rows),'mean_margin',math.fsum(rows)/len(rows))
print('TOP1 recipient',sum(s['top_token_id']==c['recipient_id'] for c,s in observed),'giver',sum(s['top_token_id']==c['giver_id'] for c,s in observed),'other',sum(s['top_token_id'] not in (c['recipient_id'],c['giver_id']) for c,s in observed))
print('RANGE margins',min(s['margin'] for c,s in observed),max(s['margin'] for c,s in observed),'ties',sum(s['margin']==0 for c,s in observed))
print('PASS: frozen sources, direct receipt, 193 events, 64 full logits/scores and exact paired gate')
print(json.dumps(summary,sort_keys=True))
