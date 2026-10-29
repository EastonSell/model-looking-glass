"""Model-free raw-evidence replay for the fixed year-versus-label study.

Synthetic mode checks only a fabricated schedule. Production provenance must
be supplied and is checked separately by this reader before accepting a run.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import subprocess

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('earlier_independent_numeric',here/'explicit_year_saved_replay.py')
numeric=importlib.util.module_from_spec(spec);spec.loader.exec_module(numeric)
need=numeric.need;same=numeric.same;values=numeric.values;norm=numeric.norm;score=numeric.score
round_f32=numeric.round_f32;sha=numeric.sha
WIDTH,VOCAB=768,50257
ROW_BYTES,LOGITS_BYTES=WIDTH*4,VOCAB*4
CASE_SHA='07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'
PROTOCOL_SHA='5da05c4d35708b420e5843b1eb77fda8699c74755233b510a01683ead2081f4a'
CONDITIONS=('after','before','near')
ACTIONS=('self','year_final','label_final','year_at_zero','rotated_year_final')
PROTOCOL_REVISION='d35d685950e36e91fc5cf576c03fe65e8d02aac3'
BASELINE_VERSION='distilgpt2-development-exposures-v24'
BASELINE_SHA='877fa0376925be34ac3de14ff066e5465a67953660bce81011e9c8fd87810bd3'
RESERVATION_VERSION='distilgpt2-development-exposures-v25'
RESERVATION_SHA='32cc4bcbe4b482ded122715c5f96c6c5c765a76a1ef3532360cdf9890827f144'

def provenance(args,run,manifest,result,cases,outcome='complete'):
    """Check saved operator evidence against separately supplied immutable pins."""
    if args.synthetic:
        return 'fabricated data only; no provenance or execution attestation'
    pins=(args.receipt_sha256,args.pre_run_sha256,args.launch_observation_sha256,args.launch_main,
          args.launch_start_sha256,args.start_main,args.source_revision,args.source_tree,
          args.source_root,args.current_exposure_version,args.current_exposure_sha256)
    need(all(pins),'all external provenance pins are required')
    for value in (args.receipt_sha256,args.pre_run_sha256,args.launch_observation_sha256,
                  args.launch_start_sha256,args.current_exposure_sha256):
        need(re.fullmatch(r'[0-9a-f]{64}',value) is not None,'invalid external SHA-256')
    for value in (args.source_revision,args.source_tree,args.launch_main,args.start_main):
        need(re.fullmatch(r'[0-9a-f]{40}',value) is not None,'invalid source revision or tree')
    match=re.fullmatch(r'distilgpt2-development-exposures-v([0-9]+)',args.current_exposure_version)
    need(match is not None and int(match.group(1))>=25,'invalid current exposure version')
    attempt=run.parent
    receipt_raw=(attempt/'supervisor.json').read_bytes();need(sha(receipt_raw)==args.receipt_sha256,'receipt hash')
    receipt=json.loads(receipt_raw)
    need(receipt['schema']=='distilgpt2-year-label-control-process-observation-v1','receipt schema')
    need(outcome in ('complete','incomplete'),'unsupported expected outcome')
    if outcome=='complete':
        need(type(receipt['process_exit_code']) is int and receipt['process_exit_code']==0 and
             receipt['interrupted'] is False and receipt['failure_file_present'] is False and
             not (run/'failure.json').exists(),'incomplete direct process')
    else:
        need(type(receipt['process_exit_code']) is int and receipt['process_exit_code']==1 and
             receipt['interrupted'] is False and receipt['failure_file_present'] is True and
             (run/'failure.json').is_file() and not (run/'result.json').exists(),'not the expected incomplete process')
    need(receipt['source_revision']==args.source_revision==manifest['source']['revision'] and
         receipt['source_tree']==args.source_tree==manifest['source']['tree'],'receipt source mismatch')
    pre_raw=(attempt/'pre-run.json').read_bytes();need(sha(pre_raw)==args.pre_run_sha256==receipt['pre_run_sha256'],'pre-run hash')
    pre=json.loads(pre_raw)
    need(pre['schema']=='distilgpt2-year-label-control-launch-v1' and pre.get('draft_only') is not True,'pre-run schema or draft')
    for key,value in {'source_revision':args.source_revision,'source_tree':args.source_tree,
                      'protocol_revision':PROTOCOL_REVISION,'protocol_sha256':PROTOCOL_SHA,'case_sha256':CASE_SHA,
                      'reservation_version':RESERVATION_VERSION,'reservation_sha256':RESERVATION_SHA,
                      'current_exposure_version':args.current_exposure_version,
                      'current_exposure_sha256':args.current_exposure_sha256}.items():
        need(pre[key]==value,'pre-run mismatch: '+key)
    need(manifest['protocol_revision']==PROTOCOL_REVISION,'manifest protocol revision')
    intended=pre['intended_execution']
    for key,value in {'device':'cpu','dtype':'float32','attention':'eager','use_cache':False,
        'cuda_visible_devices':'-1','hf_hub_offline':'1','transformers_offline':'1','threads':1,
        'cpu_rng_seed':20261029,'mkl_cbwr':None,'aten_cpu_capability':None,'automatic_retry':False,
        'deterministic_algorithms':False,'float32_matmul_precision':'highest','planned_forwards':384,
        'expected_raw_payload_bytes':103736832,'model_id':'distilbert/distilgpt2',
        'model_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'tokenizer_sha256':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'weights_sha256':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'}.items():
        need(key in intended and type(intended[key]) is type(value) and intended[key]==value,'pre-run runtime mismatch: '+key)
    prefix=intended['pycache_prefix_sha256']
    need(type(prefix) is str and re.fullmatch(r'[0-9a-f]{64}',prefix) is not None and
         manifest['empty_pycache_prefix_sha256']==prefix==receipt['pycache_prefix_sha256'] and
         receipt['pycache_prefix_matches_pre_run'] is True and receipt['pycache_prefix_empty_after'] is True,
         'bytecode prefix observation mismatch')
    need(manifest['exposure_inventories']=={'baseline_version':BASELINE_VERSION,'baseline_sha256':BASELINE_SHA,
         'reservation_version':RESERVATION_VERSION,'reservation_sha256':RESERVATION_SHA},'exposure binding')
    paths_in_run=list(run.iterdir())
    need(all(p.is_file() and not p.is_symlink() for p in paths_in_run),'unexpected run entry')
    terminal='result.json' if outcome=='complete' else 'failure.json'
    expected_files={'manifest.json',terminal,'attempt.jsonl'}|{r['case_id']+'.f32' for r in cases}
    need({p.name for p in paths_in_run}==expected_files==set(receipt['run_sha256']),'receipt file inventory')
    for name,digest in receipt['run_sha256'].items():
        need(Path(name).name==name and type(digest) is str and re.fullmatch(r'[0-9a-f]{64}',digest) is not None
             and sha((run/name).read_bytes())==digest,'receipt file hash: '+name)
    need(sha((attempt/'stdout.txt').read_bytes())==receipt['stdout_sha256'],'process output hash: stdout.txt')
    projection=getattr(args,'stderr_projection',None); projection_sha=getattr(args,'stderr_projection_sha256',None)
    need(bool(projection)==bool(projection_sha),'stderr projection and external hash must be supplied together')
    if projection is not None:
        need(outcome=='incomplete' and re.fullmatch(r'[0-9a-f]{64}',projection_sha) is not None and
             sha(Path(projection).read_bytes())==projection_sha and type(receipt['stderr_bytes']) is int and
             receipt['stderr_bytes']>0,'stderr projection or observed size')
        stderr_scope='public redacted projection checked; original stderr hash and projection fidelity are not independently verified'
    else:
        need(sha((attempt/'stderr.txt').read_bytes())==receipt['stderr_sha256'],'process output hash: stderr.txt')
        stderr_size=(attempt/'stderr.txt').stat().st_size
        need(type(receipt['stderr_bytes']) is int and receipt['stderr_bytes']==stderr_size and
             (stderr_size==0 if outcome=='complete' else stderr_size>0),'unexpected process stderr')
        stderr_scope=None
    source=Path(args.source_root)
    def git(*cmd): return subprocess.check_output(['git',*cmd],cwd=source,timeout=30).decode().strip()
    need(git('rev-parse','HEAD')==args.source_revision and git('show','-s','--format=%T','HEAD')==args.source_tree,
         'source checkout identity')
    need(not git('status','--porcelain','--untracked-files=normal'),'source checkout is not clean')
    expected={'scripts/year_label_control_v1.py','looking_glass/year_label_control.py',
        'looking_glass/year_label_intervention.py','looking_glass/explicit_year_direction.py',
        'looking_glass/year_transfer.py','looking_glass/year_transfer_intervention.py',
        'tests/test_year_label_control.py','tests/test_year_label_intervention.py','tests/test_year_label_runner.py',
        'experiments/distilgpt2_year_label_control_v1.json','docs/DISTILGPT2_YEAR_LABEL_CONTROL_V1.md',
        'docs/YEAR_LABEL_CONTROL_RUNNER_V1.md','looking_glass/greater_than_screen.py',
        'looking_glass/copy_head_factorial.py','looking_glass/copy_head_intervention.py',
        'scripts/year_label_receipt_v1.py','tests/test_year_label_receipt.py',
        'scripts/copying_head_eval_v1.py','scripts/copying_dev_v1.py',
        'scripts/distilgpt2_intervention_dev_bootstrap.py','looking_glass/pretrained.py',
        'looking_glass/__init__.py','looking_glass/pretrained_exposures.py'}
    expected.update(f'looking_glass/pretrained_exposures_v{i}.py' for i in range(2,26))
    paths=manifest['source']['sha256'];need(set(paths)==expected,'source path inventory')
    for path in expected:
        committed=subprocess.check_output(['git','show',f'{args.source_revision}:{path}'],cwd=source,timeout=30)
        need(committed==(source/path).read_bytes() and sha(committed)==paths[path],'source bytes: '+path)
    need(pre['script_sha256']==paths['scripts/year_label_control_v1.py'] and
         pre['receipt_writer_sha256']==paths['scripts/year_label_receipt_v1.py'] and
         pre['current_exposure_source_sha256']==paths['looking_glass/pretrained_exposures_v25.py'] and
         paths['experiments/distilgpt2_year_label_control_v1.json']==CASE_SHA and
         paths['docs/DISTILGPT2_YEAR_LABEL_CONTROL_V1.md']==PROTOCOL_SHA,'source script/case/protocol binding')
    observation_raw=(attempt/'launch-observation.json').read_bytes()
    need(sha(observation_raw)==args.launch_observation_sha256,'launch observation hash')
    observation=json.loads(observation_raw)
    for key,value in {'schema':'distilgpt2-year-label-control-launch-observation-v1',
        'pre_run_sha256':args.pre_run_sha256,'source_revision':args.source_revision,'source_tree':args.source_tree,
        'official_main_observed':args.launch_main,'latest_public_exposure_version':args.current_exposure_version,
        'latest_public_exposure_sha256':args.current_exposure_sha256}.items():
        need(observation[key]==value,'launch observation: '+key)
    loader='looking_glass/pretrained.py'
    need(observation['matched_public_file_sha256']=={p:paths[p] for p in expected-{loader}},'matched source observation')
    differing=observation['other_source_differences']
    need(set(differing)=={loader} and differing[loader]['execution_source_sha256']==paths[loader] and
         type(differing[loader]['observed_main_sha256']) is str and
         re.fullmatch(r'[0-9a-f]{64}',differing[loader]['observed_main_sha256']) is not None and
         differing[loader]['observed_main_sha256']!=paths[loader],'loader source-difference observation')
    start_raw=(attempt/'launch-start.json').read_bytes()
    need(sha(start_raw)==args.launch_start_sha256,'launch start hash')
    start=json.loads(start_raw)
    for key,value in {'schema':'distilgpt2-year-label-control-launch-start-v1',
        'pre_run_sha256':args.pre_run_sha256,'prior_observation_sha256':args.launch_observation_sha256,
        'source_revision':args.source_revision,'official_main_observed':args.start_main,
        'latest_public_exposure_version':args.current_exposure_version,
        'latest_public_exposure_sha256':args.current_exposure_sha256,'cache_prefix_empty_observed':True}.items():
        need(key in start and type(start[key]) is type(value) and start[key]==value,'launch start: '+key)
    start_paths={'docs/DISTILGPT2_YEAR_LABEL_CONTROL_V1.md','experiments/distilgpt2_year_label_control_v1.json',
        'looking_glass/pretrained_exposures_v25.py','looking_glass/year_label_control.py',
        'looking_glass/year_label_intervention.py','scripts/year_label_control_v1.py','scripts/year_label_receipt_v1.py'}
    need(start['matched_public_file_sha256']=={p:paths[p] for p in start_paths},'launch start source observation')
    env={k:{'present':False,'value':None} for k in ('MKL_CBWR','ATEN_CPU_CAPABILITY')}
    runtime=result['runtime']
    for key,value in {'device':'cpu','dtype':'float32','model_eval':True,'use_cache':False,
        'attention_implementation':'eager','cuda_visibility':'-1','cuda_initialized':False,
        'cpu_autocast':False,'guard_blocked_calls':[],'intraop_threads':1,'cpu_rng_seed':20261029,
        'deterministic_algorithms':False,'float32_matmul_precision':'highest'}.items():
        need(key in runtime and type(runtime[key]) is type(value) and runtime[key]==value,'observed runtime: '+key)
    need(manifest['cpu_backend_environment_at_torch_import']==env and runtime['cpu_backend_environment']==env,'backend environment')
    need(runtime['numeric_probe']=={'half_min_normal_bits':4194304,'input_preserved':True,'numpy_cast_preserved':True} and
         manifest['initial_numeric_probe']==runtime['numeric_probe'],'numeric probe')
    for name in ('model_config_sha256','cpu_rng_state_sha256'):
        need(type(runtime[name]) is str and re.fullmatch(r'[0-9a-f]{64}',runtime[name]) is not None,'runtime digest: '+name)
    deps=manifest['dependencies']
    for key,value in {'python':'3.11.15','torch':'2.8.0+cu128','numpy':'2.4.6',
        'transformers':'4.57.3','byteorder':'little',
        'gpt2_source_sha256':'b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80',
        'conv1d_source_sha256':'e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86'}.items():
        need(deps.get(key)==value,'dependency: '+key)
    if outcome=='complete':
        for a,b in (('model_state_sha256_before','model_state_sha256_after'),('model_buffers_sha256_before','model_buffers_sha256_after')):
            need(type(result[a]) is str and re.fullmatch(r'[0-9a-f]{64}',result[a]) is not None and result[a]==result[b],
                 'model or buffers changed')
        ck=result['checkpoint']
        for key,value in {'weights_sha256':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
        'model_id':'distilbert/distilgpt2','tokenizer_id':'distilbert/distilgpt2',
        'model_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'tokenizer_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'model_class':'GPT2LMHeadModel','tokenizer_class':'GPT2TokenizerFast',
        'parameter_count':81912576,'device':'cpu','dtype':'float32','attention_implementation':'eager','use_cache':False}.items():
            need(key in ck and type(ck[key]) is type(value) and ck[key]==value,'checkpoint: '+key)
        need(ck['asset_sha256']=={
        'config.json':'4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a',
        'generation_config.json':'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805',
        'merges.txt':'1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
        'model.safetensors':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
        'tokenizer.json':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'tokenizer_config.json':'5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
            'vocab.json':'196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'},'asset inventory')
    scope='direct-process receipt and external source/exposure pins checked; '
    if stderr_scope is not None:scope+=stderr_scope+'; '
    return scope+('this does not independently attest execution, descendants, private exposure comparisons '
                  'or current inventory chronology')

def replay(run,cases_file,args):
    numeric.binary64_probe()
    raw=cases_file.read_bytes();need(sha(raw)==CASE_SHA,'case hash')
    inventory=json.loads(raw);cases=inventory['cases'];digits=inventory['numeric_ids']
    need(len(cases)==48 and len({r['case_id'] for r in cases})==48 and len(set(digits))==100,'case inventory')
    manifest_raw=(run/'manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    result=json.loads((run/'result.json').read_bytes())
    need(manifest['schema']=='distilgpt2-year-label-control-run-v1' and result['schema']=='distilgpt2-year-label-control-result-v1','schemas')
    need(manifest['cases']==cases and manifest['cases_sha256']==CASE_SHA and manifest['protocol_sha256']==PROTOCOL_SHA,'case/source binding')
    need(type(manifest['planned_forwards']) is int and manifest['planned_forwards']==384,'planned forwards')
    need(result['status']=='complete' and result['manifest_sha256']==sha(manifest_raw),'complete result')
    provenance_scope=provenance(args,run,manifest,result,cases)
    need(len(manifest_raw)<=4*1024*1024 and (run/'result.json').stat().st_size<=4*1024*1024 and (run/'attempt.jsonl').stat().st_size<=8*1024*1024,'metadata cap')
    events=[json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
    need(events and events[0]=={'event':'runtime_ready','actual_runtime':result['runtime']},'runtime event')
    cursor=1;records=[];offsets={r['case_id']:0 for r in cases};cache={r['case_id']:{} for r in cases};by_id={r['case_id']:r for r in cases};payload_bytes=0
    def event(kind):
        nonlocal cursor
        need(cursor<len(events),'missing '+kind);item=events[cursor];cursor+=1
        need(item.get('event')==kind,'unexpected event '+str(cursor-1));return item
    def action(row,name,family=None):
        nonlocal payload_bytes
        ident=row['case_id'];start=event('start')
        need(set(start)=={'event','case_id','action','intervention'} and start['case_id']==ident and start['action']==name,'start identity')
        need((start['intervention'] is None)==(name in ('clean','noop','clean_after')),'start plan')
        if start['intervention'] is not None:
            need(type(start['intervention']) is dict and set(start['intervention'])==
                 {'position','expected_before_hex','expected_after_hex','requested','achieved','changed_bits'},
                 'incomplete durable start plan')
        rec=event('raw_recorded');need(rec['case_id']==ident and rec['action']==name and set(rec)=={'event','case_id','action','raw'},'raw event')
        desc=rec['raw'];expected={'logits'} if name in ('clean','clean_after') else {'logits','block_before','block_after'}
        need(set(desc)==expected,'raw payload names');payload={}
        with (run/(ident+'.f32')).open('rb') as stream:
            for part,d in sorted(desc.items(),key=lambda kv:kv[1]['offset']):
                length=LOGITS_BYTES if part=='logits' else len(row['input_ids'])*ROW_BYTES
                need(set(d)=={'offset','bytes','sha256'} and d['offset']==offsets[ident] and d['bytes']==length,'segment offset/size')
                stream.seek(d['offset']);b=stream.read(length);need(len(b)==length and sha(b)==d['sha256'],'segment digest')
                payload[part]=b;offsets[ident]+=length;payload_bytes+=length
        item=event('complete');need(item['case_id']==ident and item['action']==name and item['raw']==desc,'completion identity')
        item=dict(item);del item['event'];records.append(item)
        if start['intervention'] is not None:
            need(type(item['intervention']) is dict and set(item['intervention'])==
                 set(start['intervention'])|{'observed_achieved','observed_changed_bits'} and
                 all(same(item['intervention'][k],v) for k,v in start['intervention'].items()),'start/completion plan')
        else:
            need(item['intervention'] is None,'unexpected completion plan')
        scores=score(payload['logits'],row,digits);need(same(item['scores'],scores),'score mismatch')
        saved=cache[ident]
        if name=='clean':saved['clean']=payload['logits'];saved['scores']=scores
        elif name=='noop':
            need(payload['logits']==saved['clean'] and payload['block_before']==payload['block_after'],'no-op differs')
            values(payload['block_before']);saved['before']=payload['block_before']
        elif name=='clean_after':need(payload['logits']==saved['clean'],'clean-after differs')
        else:
            need(payload['block_before']==saved['before'],'pre-action tensor differs')
            pos=0 if name=='year_at_zero' else row['final_position'];start_offset=pos*ROW_BYTES
            before=saved['before'][start_offset:start_offset+ROW_BYTES]
            if name=='self':after=before;requested=0.
            else:
                field='label_donor_case_id' if name=='label_final' else 'year_donor_case_id'
                target=values(saved['before'][-ROW_BYTES:]);donor=values(cache[by_id[row[field]]['case_id']]['before'][-ROW_BYTES:])
                delta=[b-a for a,b in zip(target,donor)];length=norm(delta);need(length>0,'zero donor distance')
                displacement=[family['dose']/length*x for x in delta]
                if name=='rotated_year_final':displacement=[displacement[(i-1)%WIDTH] for i in range(WIDTH)]
                after=struct.pack('<768I',*[round_f32(x+y) for x,y in zip(values(before),displacement)]);requested=family['dose']
            full=saved['before'][:start_offset]+after+saved['before'][start_offset+ROW_BYTES:]
            need(payload['block_after']==full,'landing or untouched bits')
            info=item['intervention'];need(info['position']==pos and info['expected_before_hex']==before.hex() and info['expected_after_hex']==after.hex(),'plan bits')
            achieved=norm(x-y for x,y in zip(values(after),values(before)))
            changed=sum(x!=y for x,y in zip(struct.iter_unpack('<I',before),struct.iter_unpack('<I',after)))
            need(info['requested']==requested and info['achieved']==achieved==info['observed_achieved'],'achieved dose')
            need(abs(achieved-requested)<=1e-5*requested and info['changed_bits']==changed==info['observed_changed_bits'],'dose tolerance/changed bits')
            need(name!='self' or payload['logits']==saved['clean'],'self logits')
            saved.setdefault('action_scores',{})[name]=scores['band_score']
    for row in cases:action(row,'clean');action(row,'noop')
    zero=event('position_zero_check')
    for i in range(0,48,4):
        snapshots=[cache[r['case_id']]['before'][:ROW_BYTES] for r in cases[i:i+4]]
        need(all(x==snapshots[0] for x in snapshots),'position-zero differs')
    need(zero=={'event':'position_zero_check','pairs_checked':36},'position-zero count')
    clean_families=[];differences=[]
    for i in range(4):
        group=cases[12*i:12*(i+1)];conditions={}
        for condition in CONDITIONS:
            labels={}
            for label in inventory['label_pairs'][i%2]:
                matched={r['year_role']:cache[r['case_id']]['scores'] for r in group
                         if r['condition']==condition and r['label']==label}
                need(set(matched)=={'low','high'},'clean role pair')
                difference=matched['low']['band_score']-matched['high']['band_score'];differences.append(difference)
                labels[label]={'low':matched['low'],'high':matched['high'],'low_minus_high':difference}
            conditions[condition]=labels
        clean_families.append({'family_index':i,'conditions':conditions})
    clean_summary={'families':clean_families,
                   'mean_band_score':math.fsum(cache[r['case_id']]['scores']['band_score'] for r in cases)/48,
                   'mean_local_score':math.fsum(cache[r['case_id']]['scores']['local_score'] for r in cases)/48,
                   'mean_low_minus_high':math.fsum(differences)/24}
    clean_event=event('clean_summary')
    need(same(clean_event,{'event':'clean_summary','summary':clean_summary}) and same(result['clean_summary'],clean_summary),
         'clean summary arithmetic')
    families=[]
    for i in range(4):
        group=cases[12*i:12*(i+1)];labels=inventory['label_pairs'][i%2];norms={}
        by={(r['condition'],r['label'],r['year_role']):r for r in group}
        def final(c,l,r):return values(cache[by[c,l,r]['case_id']]['before'][-ROW_BYTES:])
        for condition in CONDITIONS:
            for label in labels:
                lo,hi=final(condition,label,'low'),final(condition,label,'high')
                norms[condition+':year:'+label]=norm(b-a for a,b in zip(lo,hi))
            for role in ('low','high'):
                left,right=final(condition,labels[0],role),final(condition,labels[1],role)
                norms[condition+':label:'+role]=norm(b-a for a,b in zip(left,right))
        dose=.5*min(norms.values());need(math.isfinite(dose) and dose>2**-20,'invalid family dose')
        family={'family_index':i,'dose':dose,'norms':norms,'fractions':{k:dose/v for k,v in norms.items()}}
        need(same(event('family_dose'),{'event':'family_dose',**family}),'family dose event');families.append(family)
        for row in group:
            shift=row['rotation']
            for name in ACTIONS[shift:]+ACTIONS[:shift]:action(row,name,family)
            action(row,'clean_after')
    need(cursor==len(events) and records==result['records'] and families==result['family_doses'],'extra events or result records')
    need(result['interventions_performed'] is True and len(records)==384 and
         all(type(result[k]) is int and result[k]==384 for k in ('submitted_forwards','model_call_returns_observed','returned_forwards')),'forward counts')
    need(payload_bytes==result['raw_payload_bytes']==103736832,'raw bytes')
    for ident,end in offsets.items():need((run/(ident+'.f32')).stat().st_size==end,'unaccounted bytes')
    need({p.name for p in run.glob('*.f32')}=={x+'.f32' for x in offsets},'raw file set')
    families_out=[]
    for i in range(4):
        targets=[]
        for row in cases[12*i:12*(i+1)]:
            c=cache[row['case_id']];effects={a:c['action_scores'][a]-c['scores']['band_score'] for a in ACTIONS}
            year=-effects['year_final'] if row['year_role']=='low' else effects['year_final']
            controls=('label_final','year_at_zero','rotated_year_final');margins={a:year-abs(effects[a]) for a in controls}
            checks={a:year>abs(effects[a])+.1 for a in controls}
            targets.append({'case_id':row['case_id'],'condition':row['condition'],'label':row['label'],'year_role':row['year_role'],
                            'effects':effects,'year_donorward':year,'margins':margins,'checks':checks,'pass':all(checks.values())})
        families_out.append({'family_index':i,'targets':targets,'pass':all(t['pass'] for t in targets)})
    targets=[t for f in families_out for t in f['targets']];count=sum(f['pass'] for f in families_out)
    means={c:{a:math.fsum(t['effects'][a] for t in targets if t['condition']==c)/16 for a in ACTIONS} for c in CONDITIONS}
    donor={c:math.fsum(t['year_donorward'] for t in targets if t['condition']==c)/16 for c in CONDITIONS}
    summary={'families':families_out,'mean_effect_by_condition':means,'mean_year_donorward_by_condition':donor,'pattern_count':count,'pattern_supported':count>=3}
    need(same(summary,result['intervention_summary']),'signed control/family summary')
    numeric.binary64_probe()
    return {'status':'pass','forwards':len(records),'raw_bytes':payload_bytes,'clean_summary':clean_summary,'family_doses':families,
            'intervention_summary':summary,'provenance_scope':provenance_scope,
            'scope':'saved bits and arithmetic; no model or independent execution attestation'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--receipt-sha256');parser.add_argument('--pre-run-sha256')
    parser.add_argument('--launch-observation-sha256');parser.add_argument('--launch-main')
    parser.add_argument('--launch-start-sha256');parser.add_argument('--start-main')
    parser.add_argument('--source-revision');parser.add_argument('--source-tree');parser.add_argument('--source-root',type=Path)
    parser.add_argument('--current-exposure-version');parser.add_argument('--current-exposure-sha256')
    args=parser.parse_args();print(json.dumps(replay(args.run,args.cases,args),indent=2,allow_nan=False))
