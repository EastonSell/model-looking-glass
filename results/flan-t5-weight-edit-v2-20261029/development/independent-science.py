import hashlib, json, math, pathlib, struct, subprocess, sys, unicodedata
from collections import Counter
from tokenizers import Tokenizer

root=pathlib.Path(r'E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2\fixed-attempt\development')
repo=r'E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2\fixed-attempt\publication.git'
assets=pathlib.Path(r'C:\Users\easto\Documents\Codex\2026-10-28\https-github-com-eastonsell-model-looking\work\t5-cache\flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab')
vocab=32128; doses=(.25,.5,.75,1.,1.25,1.5,2.,3.,4.); edited=1627
labels={'positive':[1465,1], 'negative':[2841,1]}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def close(a,b):
 return isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12)
def must(condition,message):
 if not condition: raise AssertionError(message)
phase_path=root/'phase.json'; exit_path=root/'operator-exit.json'
phase=json.loads(phase_path.read_bytes()); exit_record=json.loads(exit_path.read_bytes())
must(digest(phase_path)=='1cd8e44c5294a2d0074d352f4280eaa365947d572cfd959ae29ab9fc12703a2b','phase pin')
must(digest(exit_path)=='a4ab4a47448ff373ab76673a055a9efe48656ae05aee05ea195ea5336c353b35','exit pin')
for key,path in [('result_sha256',root/'result.json'),('events_sha256',root/'evidence/events.jsonl'),
                 ('logits_sha256',root/'evidence/logits.f32'),('resources_sha256',root/'resources.jsonl')]:
 must(digest(path)==phase[key],key)
must(exit_record['phase_sha256']==digest(phase_path) and exit_record['exit_code']==0 and exit_record['reaped'] is True
 and exit_record['drained'] is True and exit_record['timed_out'] is False and exit_record['error'] is None,'reported process')
result=json.loads((root/'result.json').read_bytes())
events=[json.loads(line) for line in (root/'evidence/events.jsonl').read_bytes().splitlines()]
must(all(e.get('event_index')==i for i,e in enumerate(events)),'event order')
counts=Counter(e['kind'] for e in events)
must(phase['calls']==370 and counts['forward_started']==370 and counts['forward_complete']==370
 and counts['case_complete']==68 and counts['arm_started']==29 and counts['arm_completed']==29,'schedule counts')
case_raw=subprocess.check_output(['git','--git-dir='+repo,'show',
 'b3c7e19a5349cd2fb76ef4908407342233c7e240:docs/FLAN_T5_WEIGHT_EDIT_CASES_V1.json'])
must(hashlib.sha256(case_raw).hexdigest()=='a118dde671351d21c519b6a61957e9478c9501bac52d55aa34e8394772e933d5','case pin')
cases={c['id']:c for c in json.loads(case_raw)['cases']}
tokenizer_path=assets/'tokenizer.json'
manifest=json.loads((assets/'asset-manifest.json').read_bytes())
must(digest(assets/'asset-manifest.json')=='d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
 and digest(tokenizer_path)==manifest['files']['tokenizer.json']['sha256'],'tokenizer pin')
tokenizer=Tokenizer.from_file(str(tokenizer_path))
special={x['id'] for x in json.loads(tokenizer_path.read_bytes())['added_tokens'] if x['special']}
used=set()
stream=(root/'evidence/logits.f32').open('rb')
def raw(span,add=False):
 must(span['bytes']>0 and span['offset']>=0 and span['bytes']% (vocab*4)==0,'span')
 stream.seek(span['offset']); data=stream.read(span['bytes'])
 must(len(data)==span['bytes'] and hashlib.sha256(data).hexdigest()==span['sha256'],'span digest')
 if add:
  must(span['call'] not in used,'duplicate outcome call'); used.add(span['call'])
 return data
def rows(span):
 data=raw(span,True)
 return [struct.unpack('<'+str(vocab)+'f',data[k:k+vocab*4]) for k in range(0,len(data),vocab*4)]
def flat(terms,sign=1):
 for z,m,l in terms:
  yield sign*z; yield -sign*m; yield -sign*l
def score(terms): return math.fsum(flat(terms))
def diff(after,before):return math.fsum((*flat(after),*flat(before,-1)))
def margin(terms):return diff(terms['negative'],terms['positive'])
def effect(after,before):
 a,b=after['terms'],before['terms']
 return {'margin_gain':math.fsum((*flat(a['negative']),*flat(b['negative'],-1),
       *flat(a['positive'],-1),*flat(b['positive']))),
       'target_gain':diff(a['negative'],b['negative']), 'negative':after['label']=='negative'}
def norm(text):return unicodedata.normalize('NFC',text).strip().casefold()
def verify(out):
 case=cases[out['case_id']]
 prefix=[0]; emitted=[]; tops=[]
 for i,span in enumerate(out['greedy_spans']):
  must(span['case_id']==case['id'] and span['stage']==f'greedy-{i}' and span['metadata']=={'arm':out['arm'],'decoder_prefix':prefix},'greedy identity')
  row=rows(span);must(len(row)==1 and all(map(math.isfinite,row[0])),'greedy dimensions')
  maximum=max(row[0]); tied=[j for j,x in enumerate(row[0]) if x==maximum]
  token=tied[0];emitted.append(token);tops.append({'representative':token,'tied_ids':tied,'unique':len(tied)==1})
  prefix=[*prefix,token]
  must(token!=1 or i==len(out['greedy_spans'])-1,'early eos')
 terminated=emitted[-1]==1
 must(terminated or len(emitted)==4,'truncated greedy')
 content=emitted[:-1] if terminated else emitted
 decoded=tokenizer.decode(content,skip_special_tokens=False)
 valid=terminated and all(x['unique'] for x in tops) and not any(x in special for x in content)
 strict_label=norm(decoded) if valid and norm(decoded) in labels else None
 correct=valid and norm(decoded)==norm(case['expected'])
 must(out['greedy_ids']==emitted and out['tops']==tops and out['terminated']==terminated
  and out['decoded']==decoded and out['strict_label']==strict_label and out['strict_correct']==correct,'greedy or label')
 terms={}
 for name,tokens in labels.items():
  span=out['label_spans'][name]
  must(span['case_id']==case['id'] and span['stage']=='teacher-'+name
   and span['metadata']=={'arm':out['arm'],'decoder_prefix':[0,*tokens[:-1]]},'teacher identity')
  matrix=rows(span);must(len(matrix)==len(tokens),'teacher rows')
  terms[name]=[]
  for row,token in zip(matrix,tokens):
   must(all(map(math.isfinite,row)),'nonfinite')
   m=max(row);ls=math.log(math.fsum(math.exp(x-m) for x in row))
   terms[name].append((row[token],m,ls))
  must(all(close(a,b) for aa,bb in zip(terms[name],out['label_terms'][name]) for a,b in zip(aa,bb)), 'teacher score terms')
 must(close(out['margin'],margin(terms)) and close(out['negative_score'],score(terms['negative'])),'score')
 return {'case_id':case['id'],'label':strict_label,'correct':correct,'ids':emitted,'terms':terms,'outcome':out}
outcomes={}
def add(out):
 key=(out['arm'],out['case_id']);must(key not in outcomes,'duplicate case')
 outcomes[key]=verify(out)
for out in result['baseline'].values():add(out)
for out in result['primary'].values():add(out)
for arm in ('broad','random','opposite'):
 must([x['dose'] for x in result['grids'][arm]]==list(doses),'grid')
 for point in result['grids'][arm]:
  for out in point['outcomes']:add(out)
for out in result['same']:add(out)
must(len(outcomes)==68,'outcome count')
event_cases={(e['metadata']['arm'],e['metadata']['outcome']['case_id']):e['metadata']['outcome']
 for e in events if e['kind']=='case_complete'}
must(len(event_cases)==68 and all(event_cases[k]==v['outcome'] for k,v in outcomes.items()),'saved versus journal cases')
baseline_ids=[f'{phase_name}-{n}-{side}' for phase_name in ('discovery','calibration') for n in (1,2) for side in ('target','donor')]
baseline={}
for cid in baseline_ids:
 value=outcomes[('baseline',cid)]; m=margin(value['terms']); positive=cases[cid]['expected']=='positive'
 baseline[cid]={'label':value['label'],'ids':value['ids'],'margin':m,'supported':bool(value['correct'] and (m<0 if positive else m>0))}
must(all(x['supported'] for x in baseline.values()),'clean baseline eligibility')
calib=[f'calibration-{n}-target' for n in (1,2)]
primary_effects=[effect(outcomes[('primary',cid)],outcomes[('baseline',cid)]) for cid in calib]
eligible=all(x['negative'] and x['margin_gain']>=.5 and x['target_gain']>=.5 for x in primary_effects)
must(eligible==result['calibration_eligible'],'calibration eligibility')
for actual,stored in zip(primary_effects,result['primary_effects']):
 must(actual['negative']==stored['negative'] and all(close(actual[k],stored[k]) for k in ('margin_gain','target_gain')),'primary effects')
selected={}
for arm in ('broad','random','opposite'):
 points=[]
 for dose,point in zip(doses,result['grids'][arm]):
  values=[effect(outcomes[(f'{arm}@{dose}',cid)],outcomes[('baseline',cid)]) for cid in calib]
  for actual,stored in zip(values,point['rows']):
   must(actual['negative']==stored['negative'] and all(close(actual[k],stored[k]) for k in ('margin_gain','target_gain')),'grid effects')
  ratios=[abs(math.fsum((v[k],-p[k])))/max(.1,.1*abs(p[k])) for v,p in zip(values,primary_effects) for k in ('margin_gain','target_gain')]
  pointmax=max(ratios); matched=eligible and pointmax<=1 and all(v['negative'] for v in values)
  points.append({'dose':dose,'maximum_ratio':pointmax,'matched':matched,'effects':values})
  recorded=next(x for x in result['selection'][arm]['grid'] if x['dose']==dose)
  must(recorded['matched']==matched and close(recorded['maximum_ratio'],pointmax),'grid selection')
 matches=[p for p in points if p['matched']]
 chosen=min(matches or points,key=lambda x:(x['maximum_ratio'],x['dose']))
 recorded=result['selection'][arm]
 must(recorded['dose']==chosen['dose'] and recorded['matched']==chosen['matched']
  and close(recorded['maximum_ratio'],chosen['maximum_ratio']),'chosen dose')
 selected[arm]={'choice':chosen,'points':points}
same=[effect(outcomes[('same',cid)],outcomes[('baseline',cid)]) for cid in calib]
substitution={}
for name in ('discovery-1','discovery-2','calibration-1','calibration-2'):
 target=outcomes[('primary',name+'-target')]['outcome']
 donor=outcomes[('baseline',name+'-donor')]['outcome']
 checked=0
 for label in ('positive','negative'):
  a,b=target['label_spans'][label],donor['label_spans'][label]
  must(a['metadata']['decoder_prefix']==b['metadata']['decoder_prefix'],'substitution teacher prefix')
  if edited not in a['metadata']['decoder_prefix']:
   must(raw(a)==raw(b),'substitution teacher bytes');checked+=1
 for a,b in zip(target['greedy_spans'],donor['greedy_spans']):
  pa,pb=a['metadata']['decoder_prefix'],b['metadata']['decoder_prefix']
  if pa!=pb or edited in pa: break
  must(raw(a)==raw(b),'substitution greedy bytes');checked+=1
 substitution[name]={'matched_prefix_spans':checked,'identical_generated_ids':target['greedy_ids']==donor['greedy_ids']}
initial=phase['initial_probe']; initial_raw=raw(initial)
starts=[]; installed=[]; completed=[]
for e in events:
 if e['kind']=='arm_started': starts.append(e['metadata'])
 elif e['kind']=='arm_installed': installed.append(e['metadata'])
 elif e['kind']=='arm_completed': completed.append(e['metadata'])
receipt=next(e['metadata']['receipt'] for e in events if e['kind']=='frozen_candidates')
inventory={(x['arm'],x['dose']):x for x in receipt['entries']}
for start,inst,done in zip(starts,installed,completed):
 key=(start['arm'],start['dose']);item=inventory[key]
 must((inst['arm'],inst['dose'])==key==(done['arm'],done['dose'])
   and inst['sha256']==item['sha256']==done['installed_sha256']
   and start['original_sha256']==item['original_sha256']==done['restored_row_sha256']
   and start['state_sha256']==receipt['state_sha256']==done['post_restore_state_sha256']==phase['final_state_sha256']
   and start['buffer_sha256']==done['pre_buffer_sha256']==done['post_probe_buffer_sha256']
   and done['topology_unchanged'] is True and raw(done['probe_span'])==initial_raw,'recorded restoration')
stream.close()
out={'schema':'flan-v2-independent-development-science-review-v1',
 'scope':'closed saved bytes and self-reported process/state; no proof of execution, physical restoration, custody or chronology; no evaluation or locality evidence',
 'phase_sha256':digest(phase_path),'exit_sha256':digest(exit_path),'result_sha256':phase['result_sha256'],
 'events_sha256':phase['events_sha256'],'logits_sha256':phase['logits_sha256'],'calls':phase['calls'],
 'case_sha256':hashlib.sha256(case_raw).hexdigest(),'baseline':baseline,'calibration_eligible':eligible,
 'primary_calibration_effects':primary_effects,'controls':selected,'same_polarity_effects':same,
 'substitution':substitution,'recorded_restorations':len(completed),'outcomes_replayed':len(outcomes)}
path=pathlib.Path(sys.argv[1])/'independent-science.json'
path.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
print('outcomes:',len(outcomes),'calls:',phase['calls'],'restoration records/probe bytes:',len(completed))
print('baseline supported:',sum(x['supported'] for x in baseline.values()),'/8; calibration eligible:',eligible)
print('primary calibration',primary_effects)
for name in ('broad','random','opposite'):
 c=selected[name]['choice'];print(name,'dose:',c['dose'],'matched:',c['matched'],'max_ratio:',c['maximum_ratio'],'effects:',c['effects'])
print('same polarity:',same)
print('substitution:',substitution)
print('report:',path,'sha256:',digest(path))