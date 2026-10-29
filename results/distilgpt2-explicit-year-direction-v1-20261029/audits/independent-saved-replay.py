"""Independent model-free replay of explicit-year saved evidence.

Production mode requires independently supplied receipt, pre-run, source,
reservation and current-exposure pins. Synthetic mode asserts no provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess

CASE_SHA='419d74c23333407f9a9dc0b6f7d7b70764ab5d1078e2b0a4c84d37ba194df8a9'
PROTOCOL_SHA='68f028e401fd4c43d1cac89772da10c92cf5e5de7af30c4043c18c21678b902e'
WIDTH,VOCAB=768,50257
ROW_BYTES,LOGITS_BYTES=WIDTH*4,VOCAB*4
CONDITIONS=('after','before','near')
ACTIONS=('self','year_final','year_at_zero','rotated_final')

def need(ok,message):
    if not ok: raise ValueError(message)

def binary64_probe():
    word=bytes.fromhex('0100000000000000')
    x=struct.unpack('<d',word)[0]
    need(x>0 and struct.pack('<d',x*1.0)==word
         and struct.pack('<d',float('5e-324'))==word,
         'auditor cannot preserve binary64 subnormals')

def same(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,dict): return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if type(a) is float: return struct.pack('<d',a)==struct.pack('<d',b)
    return a==b

def sha(b): return hashlib.sha256(b).hexdigest()

def f32(bits):
    sign=-1.0 if bits>>31 else 1.0
    e,f=(bits>>23)&255,bits&0x7fffff
    need(e!=255,'nonfinite float32')
    return sign*math.ldexp(float(f if e==0 else (1<<23)|f),-149 if e==0 else e-150)

def even(n,d):
    q,r=divmod(n,d)
    return q+int(2*r>d or (2*r==d and q%2==1))

def round_f32(value):
    need(math.isfinite(value),'nonfinite planned value')
    sign=0x80000000 if math.copysign(1.0,value)<0 else 0
    n,d=abs(value).as_integer_ratio()
    if not n: return sign
    e=n.bit_length()-d.bit_length()
    e-=int(n < d<<e) if e>=0 else int(n<<-e < d)
    if e < -126:
        f=even(n<<149,d); need(f<=1<<23,'invalid subnormal rounding'); return sign|f
    shift=23-e
    significand=even(n<<shift if shift>=0 else n,d if shift>=0 else d<<-shift)
    if significand==1<<24: significand>>=1; e+=1
    need(e<=127 and (1<<23)<=significand<(1<<24),'float32 overflow')
    return sign|((e+127)<<23)|(significand-(1<<23))

def values(raw):
    need(len(raw)%4==0,'incomplete float32 bytes')
    return [f32(x[0]) for x in struct.iter_unpack('<I',raw)]

def norm(v): return math.sqrt(math.fsum(x*x for x in v))

def score(raw,row,numeric):
    binary64_probe()
    need(len(raw)==LOGITS_BYTES,'wrong logit length')
    vals=values(raw)
    def band(a,b):
        left=[vals[i] for i in a]; right=[vals[i] for i in b]
        u,v=max(left),max(right)
        return ((u-v)+math.log(math.fsum(math.exp(x-u) for x in left)))-math.log(math.fsum(math.exp(x-v) for x in right))
    maximum=max(vals)
    weights=[math.exp(v-maximum) for v in vals]
    total=math.fsum(weights); all_numeric=set(numeric)
    return {'band_score':band(row['middle_ids'],row['above_ids']),
            'local_score':band(row['local_above_ids'],row['local_below_ids']),
            'aligned_01_99_full_mass':math.fsum(weights[i] for i in numeric[1:])/total,
            'zero_00_next_token_mass':weights[numeric[0]]/total,
            'other_vocabulary_mass':math.fsum(w for i,w in enumerate(weights) if i not in all_numeric)/total,
            'top_token_id':vals.index(maximum)}

def provenance(args,run,manifest,result):
    if args.synthetic:
        return 'synthetic fixture only; no execution provenance asserted'
    pins=(args.receipt_sha256,args.pre_run_sha256,args.launch_observation_sha256,args.launch_main,
          args.source_revision,args.source_tree,
          args.source_root,args.reservation_version,args.reservation_sha256,
          args.current_exposure_version,args.current_exposure_sha256)
    need(all(pins),'all external provenance pins are required')
    for value in (args.receipt_sha256,args.pre_run_sha256,args.launch_observation_sha256,args.reservation_sha256,args.current_exposure_sha256):
        need(re.fullmatch(r'[0-9a-f]{64}',value) is not None,'invalid external SHA-256')
    for value in (args.source_revision,args.source_tree,args.launch_main):
        need(re.fullmatch(r'[0-9a-f]{40}',value) is not None,'invalid source revision/tree')
    pattern=r'distilgpt2-development-exposures-v([0-9]+)'
    rm=re.fullmatch(pattern,args.reservation_version); cm=re.fullmatch(pattern,args.current_exposure_version)
    need(rm is not None and cm is not None,'invalid exposure version')
    version=int(rm.group(1)); need(version>=18 and int(cm.group(1))>=version,'unexpected exposure sequence')
    attempt=run.parent
    receipt_raw=(attempt/'supervisor.json').read_bytes()
    need(sha(receipt_raw)==args.receipt_sha256,'receipt hash mismatch')
    receipt=json.loads(receipt_raw)
    need(receipt['schema']=='distilgpt2-explicit-year-direction-process-observation-v1','receipt schema')
    need(type(receipt['process_exit_code']) is int and receipt['process_exit_code']==0
         and receipt['interrupted'] is False and receipt['failure_file_present'] is False
         and not (run/'failure.json').exists(),'incomplete process')
    need(receipt['source_revision']==args.source_revision==manifest['source']['revision']
         and receipt['source_tree']==args.source_tree==manifest['source']['tree'],'receipt source mismatch')
    pre_raw=(attempt/'pre-run.json').read_bytes()
    need(sha(pre_raw)==args.pre_run_sha256==receipt['pre_run_sha256'],'pre-run hash mismatch')
    pre=json.loads(pre_raw)
    need(pre['schema']=='distilgpt2-explicit-year-direction-launch-v1' and pre.get('draft_only') is not True,'pre-run schema or draft')
    for key,value in {'source_revision':args.source_revision,'source_tree':args.source_tree,
                      'protocol_sha256':PROTOCOL_SHA,'case_sha256':CASE_SHA,
                      'reservation_version':args.reservation_version,'reservation_sha256':args.reservation_sha256,
                      'current_exposure_version':args.current_exposure_version,'current_exposure_sha256':args.current_exposure_sha256}.items():
        need(pre[key]==value,'pre-run mismatch: '+key)
    need(pre['protocol_revision']=='e5f71eb66e233a54f1171d1e6f489c0fca587156'
         and manifest['protocol_revision']==pre['protocol_revision'],'protocol revision mismatch')
    observation_raw=(attempt/'launch-observation.json').read_bytes()
    need(sha(observation_raw)==args.launch_observation_sha256,'launch observation hash mismatch')
    observation=json.loads(observation_raw)
    for key,value in {'schema':'distilgpt2-explicit-year-direction-launch-observation-v1',
        'source_revision':args.source_revision,'pre_run_sha256':args.pre_run_sha256,
        'official_main_observed':args.launch_main,'current_exposure_version':args.current_exposure_version,
        'current_exposure_sha256':args.current_exposure_sha256}.items():
        need(observation[key]==value,'launch observation mismatch: '+key)
    update_sha=getattr(args,'launch_update_sha256',None)
    update_main=getattr(args,'launch_update_main',None)
    need(bool(update_sha)==bool(update_main),'both launch update pins are required together')
    if update_sha:
        need(re.fullmatch(r'[0-9a-f]{64}',update_sha) is not None
             and re.fullmatch(r'[0-9a-f]{40}',update_main) is not None,'invalid launch update pins')
        update_raw=(attempt/'launch-observation-update.json').read_bytes()
        need(sha(update_raw)==update_sha,'launch observation update hash mismatch')
        update=json.loads(update_raw)
        for key,value in {'schema':'distilgpt2-explicit-year-direction-launch-observation-update-v1',
            'previous_observation_sha256':args.launch_observation_sha256,
            'pre_run_sha256':args.pre_run_sha256,'official_main_observed':update_main,
            'current_exposure_version':args.current_exposure_version,
            'current_exposure_sha256':args.current_exposure_sha256}.items():
            need(update[key]==value,'launch observation update mismatch: '+key)
    start_sha=getattr(args,'launch_start_sha256',None)
    start_main=getattr(args,'launch_start_main',None)
    need(bool(start_sha)==bool(start_main),'both launch start pins are required together')
    if start_sha:
        need(update_sha is not None and re.fullmatch(r'[0-9a-f]{64}',start_sha) is not None
             and re.fullmatch(r'[0-9a-f]{40}',start_main) is not None,'invalid launch start pins')
        start_raw=(attempt/'launch-observation-start.json').read_bytes()
        need(sha(start_raw)==start_sha,'launch start observation hash mismatch')
        start=json.loads(start_raw)
        for key,value in {'schema':'distilgpt2-explicit-year-direction-launch-observation-start-v1',
            'previous_observation_sha256':update_sha,'pre_run_sha256':args.pre_run_sha256,
            'official_main_observed':start_main,'current_exposure_version':args.current_exposure_version,
            'current_exposure_sha256':args.current_exposure_sha256}.items():
            need(start[key]==value,'launch start observation mismatch: '+key)
    intended=pre['intended_execution']
    for key,value in {'device':'cpu','dtype':'float32','attention':'eager','use_cache':False,
        'cuda_visible_devices':'-1','hf_hub_offline':'1','transformers_offline':'1','threads':1,
        'cpu_rng_seed':20261029,'mkl_cbwr':None,'aten_cpu_capability':None,'automatic_retry':False,
        'deterministic_algorithms':False,'float32_matmul_precision':'highest','planned_forwards':336,
        'expected_raw_payload_bytes':86714688,'model_id':'distilbert/distilgpt2',
        'model_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'tokenizer_sha256':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'weights_sha256':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'}.items():
        need(key in intended and type(intended[key]) is type(value) and intended[key]==value,'pre-run runtime mismatch: '+key)
    need(manifest['exposure_inventories']=={'baseline_version':'distilgpt2-development-exposures-v17',
        'baseline_sha256':'e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a',
        'reservation_version':args.reservation_version,'reservation_sha256':args.reservation_sha256},'manifest exposure binding')
    names={p.name for p in run.iterdir() if p.is_file()}
    need(set(receipt['run_sha256'])==names,'receipt file inventory mismatch')
    for name,digest in receipt['run_sha256'].items():
        need(Path(name).name==name and sha((run/name).read_bytes())==digest,'receipt file hash mismatch')
    need({'manifest.json','result.json','attempt.jsonl'}<=names,'receipt missing files')
    for name,key in (('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')):
        need(sha((attempt/name).read_bytes())==receipt[key],'process output hash mismatch')
    need(type(receipt['stderr_bytes']) is int and receipt['stderr_bytes']==(attempt/'stderr.txt').stat().st_size,'stderr size mismatch')
    source=Path(args.source_root)
    def git(*cmd): return subprocess.check_output(['git',*cmd],cwd=source,timeout=30).decode().strip()
    need(git('rev-parse','HEAD')==args.source_revision and git('show','-s','--format=%T','HEAD')==args.source_tree,'source checkout identity')
    need(not git('status','--porcelain','--untracked-files=normal'),'source checkout not clean')
    need(subprocess.run(['git','merge-base','--is-ancestor',args.source_revision,args.launch_main],cwd=source,timeout=30).returncode==0,
         'source is not an ancestor of the observed launch main')
    if update_main:
        need(subprocess.run(['git','merge-base','--is-ancestor',args.launch_main,update_main],cwd=source,timeout=30).returncode==0,
             'original launch main is not an ancestor of updated launch main')
    if start_main:
        need(subprocess.run(['git','merge-base','--is-ancestor',update_main,start_main],cwd=source,timeout=30).returncode==0,
             'updated launch main is not an ancestor of launch start main')
    expected={'scripts/explicit_year_direction_v1.py','looking_glass/explicit_year_direction.py',
        'looking_glass/year_transfer.py','looking_glass/year_transfer_intervention.py',
        'tests/test_explicit_year_direction.py','tests/test_explicit_year_direction_runner.py',
        'experiments/distilgpt2_explicit_year_direction_v1.json','docs/DISTILGPT2_EXPLICIT_YEAR_DIRECTION_V1.md',
        'docs/EXPLICIT_YEAR_DIRECTION_RUNNER_V1.md','looking_glass/greater_than_screen.py',
        'looking_glass/copy_head_factorial.py','looking_glass/copy_head_intervention.py',
        'scripts/copying_head_eval_v1.py','scripts/copying_dev_v1.py',
        'scripts/distilgpt2_intervention_dev_bootstrap.py','looking_glass/pretrained.py',
        'looking_glass/__init__.py','looking_glass/pretrained_exposures.py'}
    expected.update(f'looking_glass/pretrained_exposures_v{i}.py' for i in range(2,version+1))
    paths=manifest['source']['sha256']; need(set(paths)==expected,'source path inventory mismatch')
    for path in expected:
        committed=subprocess.check_output(['git','show',f'{args.source_revision}:{path}'],cwd=source,timeout=30)
        need(committed==(source/path).read_bytes() and sha(committed)==paths[path],'source bytes mismatch: '+path)
    need(pre['script_sha256']==paths['scripts/explicit_year_direction_v1.py'],'pre-run script mismatch')
    need(paths['experiments/distilgpt2_explicit_year_direction_v1.json']==CASE_SHA
         and paths['docs/DISTILGPT2_EXPLICIT_YEAR_DIRECTION_V1.md']==PROTOCOL_SHA,'source case/protocol mismatch')
    env={k:{'present':False,'value':None} for k in ('MKL_CBWR','ATEN_CPU_CAPABILITY')}
    runtime=result['runtime']
    for key,value in {'device':'cpu','dtype':'float32','model_eval':True,'use_cache':False,
        'attention_implementation':'eager','cuda_visibility':'-1','cuda_initialized':False,
        'cpu_autocast':False,'guard_blocked_calls':[],'intraop_threads':1,'cpu_rng_seed':20261029,
        'deterministic_algorithms':False,'float32_matmul_precision':'highest'}.items():
        need(key in runtime and type(runtime[key]) is type(value) and runtime[key]==value,'observed runtime mismatch: '+key)
    need(manifest['cpu_backend_environment_at_torch_import']==env and runtime['cpu_backend_environment']==env,'backend environment mismatch')
    need(runtime['numeric_probe']=={'half_min_normal_bits':4194304,'input_preserved':True,'numpy_cast_preserved':True}
         and manifest['initial_numeric_probe']==runtime['numeric_probe'],'numeric probe mismatch')
    for name in ('model_config_sha256','cpu_rng_state_sha256'):
        need(type(runtime[name]) is str and re.fullmatch(r'[0-9a-f]{64}',runtime[name]) is not None,'runtime digest malformed: '+name)
    deps=manifest['dependencies']
    for key,value in {'python':'3.11.15','torch':'2.8.0+cu128','numpy':'2.4.6',
        'transformers':'4.57.3','byteorder':'little',
        'gpt2_source_sha256':'b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80',
        'conv1d_source_sha256':'e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86'}.items():
        need(deps.get(key)==value,'dependency mismatch: '+key)
    for a,b in (('model_state_sha256_before','model_state_sha256_after'),('model_buffers_sha256_before','model_buffers_sha256_after')):
        need(re.fullmatch(r'[0-9a-f]{64}',result[a]) is not None and result[a]==result[b],'model or buffer state changed')
    ck=result['checkpoint']
    for key,value in {'weights_sha256':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
        'model_id':'distilbert/distilgpt2','tokenizer_id':'distilbert/distilgpt2',
        'model_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'tokenizer_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'model_class':'GPT2LMHeadModel','tokenizer_class':'GPT2TokenizerFast',
        'parameter_count':81912576,'device':'cpu','dtype':'float32',
        'attention_implementation':'eager','use_cache':False}.items():
        need(key in ck and type(ck[key]) is type(value) and ck[key]==value,'checkpoint mismatch: '+key)
    need(ck['asset_sha256']=={
        'config.json':'4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a',
        'generation_config.json':'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805',
        'merges.txt':'1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
        'model.safetensors':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
        'tokenizer.json':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'tokenizer_config.json':'5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
        'vocab.json':'196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'},'asset inventory mismatch')
    return ('direct-process receipt and externally supplied source/exposure pins checked; '
            'not execution attestation or an independent reconstruction of exposure snapshots or private comparisons')

def main():
    binary64_probe()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--receipt-sha256'); p.add_argument('--pre-run-sha256')
    p.add_argument('--launch-observation-sha256'); p.add_argument('--launch-main')
    p.add_argument('--launch-update-sha256'); p.add_argument('--launch-update-main')
    p.add_argument('--launch-start-sha256'); p.add_argument('--launch-start-main')
    p.add_argument('--source-revision'); p.add_argument('--source-tree'); p.add_argument('--source-root',type=Path)
    p.add_argument('--reservation-version'); p.add_argument('--reservation-sha256')
    p.add_argument('--current-exposure-version'); p.add_argument('--current-exposure-sha256')
    args=p.parse_args()
    run=args.run
    cases_raw=args.cases.read_bytes(); need(sha(cases_raw)==CASE_SHA,'case hash mismatch')
    casefile=json.loads(cases_raw); cases=casefile['cases']; numeric=casefile['numeric_ids']
    need(len(cases)==48 and len({r['case_id'] for r in cases})==48 and len(set(numeric))==100,'case inventory mismatch')
    by_id={r['case_id']:r for r in cases}
    manifest_raw=(run/'manifest.json').read_bytes(); manifest=json.loads(manifest_raw)
    result=json.loads((run/'result.json').read_bytes())
    need(manifest['schema']=='distilgpt2-explicit-year-direction-run-v1' and result['schema']=='distilgpt2-explicit-year-direction-result-v1','schema mismatch')
    need(manifest['cases']==cases and manifest['cases_sha256']==CASE_SHA and manifest['protocol_sha256']==PROTOCOL_SHA,'manifest cases/protocol mismatch')
    need(type(manifest['planned_forwards']) is int and manifest['planned_forwards']==336,'manifest forward plan mismatch')
    need(result['manifest_sha256']==sha(manifest_raw) and result['status']=='complete','result/manifest mismatch')
    scope=provenance(args,run,manifest,result)
    need(len(manifest_raw)<=4*1024*1024 and (run/'result.json').stat().st_size<=4*1024*1024 and (run/'attempt.jsonl').stat().st_size<=8*1024*1024,'metadata size')
    events=[json.loads(x) for x in (run/'attempt.jsonl').read_bytes().splitlines()]
    need(events and events[0]=={'event':'runtime_ready','actual_runtime':result['runtime']},'runtime event mismatch')
    cursor=1; records=[]; offsets={r['case_id']:0 for r in cases}; payload_bytes=0
    cache={r['case_id']:{} for r in cases}
    def event(name):
        nonlocal cursor
        need(cursor<len(events),'missing '+name)
        item=events[cursor]; cursor+=1
        need(item.get('event')==name,'wrong event at '+str(cursor-1))
        return item
    def action(row,name,family=None):
        nonlocal payload_bytes
        ident=row['case_id']; start=event('start')
        need(set(start)=={'event','case_id','action','intervention'} and start['case_id']==ident and start['action']==name,'action start mismatch')
        need((start['intervention'] is None)==(name in ('clean','noop','clean_after')),'plan presence mismatch')
        recorded=event('raw_recorded')
        need(set(recorded)=={'event','case_id','action','raw'} and recorded['case_id']==ident and recorded['action']==name,'raw event mismatch')
        desc=recorded['raw']; expected={'logits'} if name in ('clean','clean_after') else {'logits','block_before','block_after'}
        need(set(desc)==expected,'payload names mismatch')
        payload={}
        with (run/(ident+'.f32')).open('rb') as stream:
            for part,d in sorted(desc.items(),key=lambda kv:kv[1]['offset']):
                length=LOGITS_BYTES if part=='logits' else len(row['input_ids'])*ROW_BYTES
                need(set(d)=={'offset','bytes','sha256'} and d['offset']==offsets[ident] and d['bytes']==length,'segment shape mismatch')
                stream.seek(d['offset']); b=stream.read(length)
                need(len(b)==length and sha(b)==d['sha256'],'segment hash mismatch')
                payload[part]=b; offsets[ident]+=length; payload_bytes+=length
        completed=event('complete')
        need(completed['case_id']==ident and completed['action']==name and completed['raw']==desc,'completion mismatch')
        item=dict(completed); del item['event']; records.append(item)
        if start['intervention'] is not None:
            need(all(item['intervention'].get(k)==v for k,v in start['intervention'].items()),'start/completion plan mismatch')
        actual=score(payload['logits'],row,numeric)
        need(same(item['scores'],actual),'full-vocabulary score mismatch')
        saved=cache[ident]
        if name=='clean':
            saved['clean']=payload['logits']; saved['scores']=actual
        elif name=='noop':
            need(payload['logits']==saved['clean'] and payload['block_before']==payload['block_after'],'no-op differs')
            values(payload['block_before']); saved['before']=payload['block_before']
        elif name=='clean_after':
            need(payload['logits']==saved['clean'],'clean-after differs')
        else:
            need(payload['block_before']==saved['before'],'pre-action tensor differs')
            partner=by_id[row['donor_case_id']]
            target=values(saved['before'][-ROW_BYTES:]); donor=values(cache[partner['case_id']]['before'][-ROW_BYTES:])
            delta=[b-a for a,b in zip(target,donor)]; length=norm(delta); dose=family['dose']
            need(length>0,'zero donor norm')
            displacement=[dose/length*x for x in delta]
            pos=0 if name=='year_at_zero' else row['final_position']; offset=pos*ROW_BYTES
            before=saved['before'][offset:offset+ROW_BYTES]
            if name=='self': after=before; requested=0.0
            else:
                if name=='rotated_final': displacement=[displacement[(i-1)%WIDTH] for i in range(WIDTH)]
                after=struct.pack('<768I',*[round_f32(x+y) for x,y in zip(values(before),displacement)])
                requested=dose
            full=saved['before'][:offset]+after+saved['before'][offset+ROW_BYTES:]
            need(payload['block_after']==full,'landed or untouched bits differ')
            info=item['intervention']
            need(info['position']==pos and info['expected_before_hex']==before.hex() and info['expected_after_hex']==after.hex(),'plan bits differ')
            achieved=norm(x-y for x,y in zip(values(after),values(before)))
            changed=sum(x!=y for x,y in zip(struct.iter_unpack('<I',before),struct.iter_unpack('<I',after)))
            need(info['requested']==requested and info['achieved']==achieved==info['observed_achieved'],'same-call dose mismatch')
            need(abs(achieved-requested)<=1e-5*requested and info['changed_bits']==changed==info['observed_changed_bits'],'dose tolerance or changed bits mismatch')
            need(name!='self' or payload['logits']==saved['clean'],'self logits differ')
            saved.setdefault('action_scores',{})[name]=actual['band_score']
    for row in cases:
        action(row,'clean'); action(row,'noop')
    zero=event('position_zero_check')
    for i in range(0,48,2):
        need(cache[cases[i]['case_id']]['before'][:ROW_BYTES]==cache[cases[i+1]['case_id']]['before'][:ROW_BYTES],'position-zero differs')
    need(zero=={'event':'position_zero_check','pairs_checked':24},'position-zero event mismatch')
    groups=[]
    for i in range(8):
        rows=cases[i*6:(i+1)*6]
        local={c:{role:cache[rows[2*j+k]['case_id']]['scores']['local_score'] for k,role in enumerate(('low','high'))}
               for j,c in enumerate(CONDITIONS)}
        fixed={c:cache[rows[2*j]['case_id']]['scores']['band_score']-cache[rows[2*j+1]['case_id']]['scores']['band_score']
               for j,c in enumerate(CONDITIONS)}
        checks={'after_low':local['after']['low']>0.5,'after_high':local['after']['high']>0.5,
                'before_low':local['before']['low']<-.5,'before_high':local['before']['high']<-.5,
                'D_after':fixed['after']>.5,'D_before':fixed['before']<-.5}
        local_diffs={f'{a}_minus_{b}':{role:local[a][role]-local[b][role] for role in ('low','high')}
                     for a,b in (('after','before'),('after','near'),('before','near'))}
        fixed_diffs={f'{a}_minus_{b}':fixed[a]-fixed[b] for a,b in (('after','before'),('after','near'),('before','near'))}
        groups.append({'pair_index':rows[0]['pair_index'],'event':rows[0]['event'],'local_scores':local,
                       'local_wording_differences':local_diffs,'fixed_band_D':fixed,
                       'fixed_band_wording_differences':fixed_diffs,'checks':checks,'comparator_pass':all(checks.values())})
    passed=[g for g in groups if g['comparator_pass']]
    clean={'pairs':groups,'comparator_gate_count':len(passed),
           'comparator_gate_passed':len(passed)>=6 and len({g['pair_index'] for g in passed})>=3}
    saved_clean=result['clean_summary']
    if 'check_counts' in saved_clean:
        clean['check_counts']={k:sum(g['checks'][k] for g in groups) for k in groups[0]['checks']}
    gates=event('clean_gates')
    need(same(gates,{'event':'clean_gates','summary':clean}) and same(saved_clean,clean),'clean gate mismatch')
    need(result['interventions_performed'] is True,'expected unconditional interventions')
    families=[]
    for i in range(8):
        rows=cases[i*6:(i+1)*6]; norms={}
        for j,c in enumerate(CONDITIONS):
            low=values(cache[rows[2*j]['case_id']]['before'][-ROW_BYTES:]); high=values(cache[rows[2*j+1]['case_id']]['before'][-ROW_BYTES:])
            norms[c]=norm(b-a for a,b in zip(low,high))
        dose=0.5*min(norms.values()); need(math.isfinite(dose) and dose>2**-20,'invalid common dose')
        family={'pair_noun_index':i,'dose':dose,'norms':norms,'fractions':{k:dose/v for k,v in norms.items()}}
        need(same(event('family_dose'),{'event':'family_dose',**family}),'family dose mismatch'); families.append(family)
        for row in rows:
            shift=row['rotation']
            for name in ACTIONS[shift:]+ACTIONS[:shift]: action(row,name,family)
            action(row,'clean_after')
    need(cursor==len(events),'extra events')
    need(records==result['records'] and families==result['family_doses'],'records/families mismatch')
    need(len(records)==336 and all(result[k]==336 for k in ('submitted_forwards','model_call_returns_observed','returned_forwards')),'forward count mismatch')
    need(payload_bytes==result['raw_payload_bytes'],'byte count mismatch')
    for ident,end in offsets.items(): need((run/(ident+'.f32')).stat().st_size==end,'unaccounted bytes')
    need({p.name for p in run.glob('*.f32')}=={x+'.f32' for x in offsets},'unexpected raw file inventory')
    pairs=[]
    for i in range(8):
        rows=cases[i*6:(i+1)*6]; conditions={}
        for j,c in enumerate(CONDITIONS):
            low,high=(cache[rows[2*j+k]['case_id']] for k in (0,1)); acts={}
            for name in ACTIONS:
                le=low['action_scores'][name]-low['scores']['band_score']
                he=high['action_scores'][name]-high['scores']['band_score']
                acts[name]={'low_E':le,'high_E':he,'low_donorward':-le,'high_donorward':he,'donorward_mean':(-le+he)/2}
            conditions[c]=acts
        contrasts={a:conditions['after'][a]['donorward_mean']-conditions['before'][a]['donorward_mean'] for a in ACTIONS}
        generic=all(conditions[c]['year_final']['low_donorward']>0 and conditions[c]['year_final']['high_donorward']>0
                    and conditions[c]['year_final']['donorward_mean']>conditions[c]['year_at_zero']['donorward_mean']+.1
                    and conditions[c]['year_final']['donorward_mean']>conditions[c]['rotated_final']['donorward_mean']+.1 for c in CONDITIONS)
        after,before=(conditions[c]['year_final'] for c in ('after','before'))
        direction=after['low_donorward']>0 and after['high_donorward']>0 and before['low_donorward']<0 and before['high_donorward']<0
        controlled=direction and contrasts['year_final']>contrasts['year_at_zero']+.1 and contrasts['year_final']>contrasts['rotated_final']+.1
        pairs.append({'pair_index':rows[0]['pair_index'],'event':rows[0]['event'],'conditions':conditions,
                      'after_minus_before':contrasts,'generic_pattern':generic,
                      'comparator_direction':direction,'comparator_control':controlled})
    means={c:{a:math.fsum(g['conditions'][c][a]['donorward_mean'] for g in pairs)/8 for a in ACTIONS} for c in CONDITIONS}
    contrast_means={a:math.fsum(g['after_minus_before'][a] for g in pairs)/8 for a in ACTIONS}
    gen=sum(g['generic_pattern'] for g in pairs); direction=sum(g['comparator_direction'] for g in pairs); control=sum(g['comparator_control'] for g in pairs)
    eligible=clean['comparator_gate_passed']
    summary={'pairs':pairs,'mean_donorward_by_condition_and_action':means,'mean_after_minus_before_by_action':contrast_means,
             'generic_pattern_count':gen,'generic_pattern_supported':gen>=6,
             'comparator_direction_count':direction,'comparator_direction_supported':direction>=6 if eligible else None,
             'comparator_control_count':control,'comparator_control_supported':control>=6 if eligible else None}
    need(same(summary,result['intervention_summary']),'intervention summary mismatch')
    binary64_probe()
    print(json.dumps({'status':'pass','scope':scope,
                      'forwards':336,'raw_bytes':payload_bytes,'clean_summary':clean,'intervention_summary':summary},indent=2,allow_nan=False))

if __name__=='__main__': main()
