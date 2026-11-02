"""Independent FLAN v2 saved evaluation and high-precision locality replay."""
import ast,hashlib,importlib.util,json,math,pathlib,struct,subprocess,sys,unicodedata
from collections import Counter
from decimal import Decimal
from tokenizers import Tokenizer
root=pathlib.Path(r'E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2\fixed-attempt\evaluation')
dev=root.parent/'development'
repo=r'E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2\fixed-attempt\publication.git'
review=pathlib.Path(r'E:\Codex\model-looking-glass\work\flan-v2-independent-science-695cd931608548c7bb830f3232bbd86a')
assets=pathlib.Path(r'C:\Users\easto\Documents\Codex\2026-10-28\https-github-com-eastonsell-model-looking\work\t5-cache\flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab')
outdir=pathlib.Path(sys.argv[1]);VOCAB=32128;EDITED=1627
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(x,msg):
 if not x:raise AssertionError(msg)
def read_json(path):return json.loads(path.read_bytes())
phase=read_json(root/'phase.json');exit_record=read_json(root/'operator-exit.json')
check(digest(root/'phase.json')=='192fbb1582cd5bd36a1618dd3795ef1078cd8c2aa0fb3e9ab9d4dbd5b99c160d','phase pin')
check(digest(root/'operator-exit.json')=='235b5c934840462ef5595e2849eed0cb4d974ab7475f60e9de28ed63f805cc5e','exit pin')
check(exit_record['phase_sha256']==digest(root/'phase.json') and exit_record['exit_code']==0
 and exit_record['reaped'] is True and exit_record['drained'] is True and exit_record['timed_out'] is False
 and exit_record['error'] is None,'recorded direct-child exit')
for key,name in [('result_sha256','result.json'),('events_sha256','evidence/events.jsonl'),
                 ('logits_sha256','evidence/logits.f32'),('resources_sha256','resources.jsonl')]:
 check(digest(root/name)==phase[key],key)
devphase=read_json(dev/'phase.json')
check(digest(dev/'phase.json')=='1cd8e44c5294a2d0074d352f4280eaa365947d572cfd959ae29ab9fc12703a2b'
 and digest(review/'independent-science.json')=='250af95f8d9bf418f9d78e04c232d55d69c22074606da1a4c0e18abe790a4b32'
 and digest(review/'audit.py')=='2fa4241d2ad691be8c115dd5743aae939c858001117cd255cef9cb32ac92a104','development pins')
development=read_json(review/'independent-science.json')
result=read_json(root/'result.json')
events=[json.loads(line) for line in (root/'evidence/events.jsonl').read_bytes().splitlines()]
check(all(e.get('event_index')==i for i,e in enumerate(events)),'event indices')
counts=Counter(e['kind'] for e in events)
check(counts['forward_started']==counts['forward_complete']==phase['calls']
 and phase['cumulative_calls']==devphase['calls']+phase['calls']
 and counts['case_complete']==76 and counts['arm_started']==counts['arm_installed']==counts['arm_completed']==5,'event counts')
case_raw=subprocess.check_output(['git','--git-dir='+repo,'show',
 'b3c7e19a5349cd2fb76ef4908407342233c7e240:docs/FLAN_T5_WEIGHT_EDIT_CASES_V1.json'])
check(hashlib.sha256(case_raw).hexdigest()=='a118dde671351d21c519b6a61957e9478c9501bac52d55aa34e8394772e933d5','cases')
cases={c['id']:c for c in json.loads(case_raw)['cases']}
asset_manifest=read_json(assets/'asset-manifest.json')
check(digest(assets/'asset-manifest.json')=='d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
 and digest(assets/'tokenizer.json')==asset_manifest['files']['tokenizer.json']['sha256'],'tokenizer')
tokenizer=Tokenizer.from_file(str(assets/'tokenizer.json'))
special={v['id'] for v in read_json(assets/'tokenizer.json')['added_tokens'] if v['special']}
# Reuse independently authored development primitive functions, not study code.
tree=ast.parse((review/'audit.py').read_text(encoding='utf8'))
names={'close','must','raw','rows','flat','score','diff','margin','effect','norm','verify'}
defs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
check({node.name for node in defs}==names,'development helper set')
stream=(root/'evidence/logits.f32').open('rb')
ns={'hashlib':hashlib,'math':math,'struct':struct,'unicodedata':unicodedata,'vocab':VOCAB,'edited':EDITED,
    'labels':{'positive':[1465,1],'negative':[2841,1]},'used':set(),'stream':stream,
    'cases':cases,'tokenizer':tokenizer,'special':special}
exec(compile(ast.Module(body=defs,type_ignores=[]),'independent_development_primitives','exec'),ns)
def verify(out):
 item=ns['verify'](out);case=cases[out['case_id']]
 item['candidate']=None
 if case['expected'] not in ns['labels']:
  span=out['candidate_span'];tokens=case['expected_ids']
  check(span is not None and span['case_id']==case['id'] and span['stage']=='teacher-candidate'
   and span['metadata']=={'arm':out['arm'],'decoder_prefix':[0,*tokens[:-1]]},'candidate teacher identity')
  matrix=ns['rows'](span);check(len(matrix)==len(tokens),'candidate rows')
  terms=[]
  for row,token in zip(matrix,tokens):
   check(all(map(math.isfinite,row)),'candidate finite')
   m=max(row);terms.append((row[token],m,math.log(math.fsum(math.exp(x-m) for x in row))))
  check(all(ns['close'](a,b) for aa,bb in zip(terms,out['candidate_terms']) for a,b in zip(aa,bb)),'candidate terms')
  item['candidate']=terms
 else: check(out['candidate_span'] is None and out['candidate_terms'] is None,'unexpected candidate')
 return item
verified={}
def add(out):
 key=(out['arm'],out['case_id']);check(key not in verified,'duplicate case')
 verified[key]=verify(out)
for out in result['baseline'].values():add(out)
for arm in ('primary','same','broad','random','opposite'):
 for out in result['arms'][arm].values():add(out)
check(len(verified)==76,'outcome inventory')
event_cases={(e['metadata']['arm'],e['metadata']['outcome']['case_id']):e['metadata']['outcome']
 for e in events if e['kind']=='case_complete'}
check(len(event_cases)==76 and all(event_cases[k]==v['outcome'] for k,v in verified.items()),'case event consistency')
target_ids=[f'evaluation-{n}-target' for n in range(1,5)]
target_rows=[];effects={}
for cid in target_ids:
 base=verified[('baseline',cid)];donor=verified[('baseline',cid.replace('-target','-donor'))]
 primary=verified[('primary',cid)]
 eligible=base['label']=='positive' and donor['label']=='negative'
 target_rows.append({'case_id':cid,'baseline_eligible':eligible,'edited_label':primary['label']})
 effects[cid]={arm:{**ns['effect'](verified[(arm,cid)],base),'strict_label':verified[(arm,cid)]['label']}
               for arm in ('primary','same','broad','random','opposite')}
eligible_rows=[r for r in target_rows if r['baseline_eligible']]
reversed_count=sum(r['edited_label']=='negative' for r in eligible_rows)
status=('ineligible' if not development['calibration_eligible'] or len(eligible_rows)<3
        else 'positive' if reversed_count>=3 else 'negative')
primary={'status':status,'baseline_eligible':len(eligible_rows),'negative_labels':reversed_count}
check(result['targets']==target_rows and result['primary']==primary and phase['scientific_status']==status,'primary classification')
comparisons={}
for arm in ('broad','random','opposite'):
 choice=development['controls'][arm]['choice']
 if not choice['matched']: reason='calibration-unmatched';balanced=False
 elif status!='positive':reason='primary-not-positive';balanced=False
 else:
  balanced=True
  for row in eligible_rows:
   p=effects[row['case_id']]['primary'];c=effects[row['case_id']][arm]
   balanced &= p['strict_label'] in ('positive','negative') and c['strict_label']==p['strict_label']
   balanced &= all(abs(math.fsum((p[k],-c[k])))<=max(.10,.10*abs(p[k])) for k in ('margin_gain','target_gain'))
  reason=None if balanced else 'evaluation-imbalance'
 check(result['comparisons'][arm]['eligible']==balanced and result['comparisons'][arm]['reason']==reason,'comparison classification')
 comparisons[arm]={'eligible':balanced,'reason':reason,'development_dose':choice['dose'],
                   'development_matched':choice['matched']}
# Exact raw substitution identity on comparable teacher/greedy prefixes.
substitution={}
for cid in target_ids:
 a=verified[('primary',cid)]['outcome'];b=verified[('baseline',cid.replace('-target','-donor'))]['outcome'];checks=0
 for label in ('positive','negative'):
  x,y=a['label_spans'][label],b['label_spans'][label]
  check(x['metadata']['decoder_prefix']==y['metadata']['decoder_prefix'],'teacher pair prefix')
  if EDITED not in x['metadata']['decoder_prefix']:check(ns['raw'](x)==ns['raw'](y),'substitution teacher');checks+=1
 for x,y in zip(a['greedy_spans'],b['greedy_spans']):
  if x['metadata']['decoder_prefix']!=y['metadata']['decoder_prefix'] or EDITED in x['metadata']['decoder_prefix']:break
  check(ns['raw'](x)==ns['raw'](y),'substitution greedy');checks+=1
 substitution[cid]={'matching_prefix_spans':checks,'same_generated_ids':a['greedy_ids']==b['greedy_ids']}
# Recorded restored state and the six reference/after-arm probe bytes.
def external_span(path,span):
 with path.open('rb') as f:f.seek(span['offset']);raw=f.read(span['bytes'])
 check(hashlib.sha256(raw).hexdigest()==span['sha256'],'external probe');return raw
initial=ns['raw'](phase['initial_probe'])
check(initial==external_span(dev/'evidence/logits.f32',devphase['initial_probe'])
 and phase['candidate_sha256']==devphase['candidate_sha256']
 and phase['final_state_sha256']==devphase['final_state_sha256'],'cross-phase reference')
receipt=next(e['metadata']['receipt'] for e in events if e['kind']=='frozen_candidates')
candidates={(x['arm'],x['dose']):x for x in receipt['entries']}
starts=[e['metadata'] for e in events if e['kind']=='arm_started']
installs=[e['metadata'] for e in events if e['kind']=='arm_installed']
ends=[e['metadata'] for e in events if e['kind']=='arm_completed']
for a,b,c in zip(starts,installs,ends):
 key=(a['arm'],a['dose']);item=candidates[key]
 check((b['arm'],b['dose'])==key==(c['arm'],c['dose'])
  and b['sha256']==item['sha256']==c['installed_sha256']
  and a['original_sha256']==item['original_sha256']==c['restored_row_sha256']
  and a['state_sha256']==phase['final_state_sha256']==c['post_restore_state_sha256']
  and a['buffer_sha256']==c['pre_buffer_sha256']==c['post_probe_buffer_sha256']
  and c['topology_unchanged'] is True and ns['raw'](c['probe_span'])==initial,'restoration')
# Locality: preserve owner KL/TV, separately recompute exact-f32 Decimal KL/TV.
spec=importlib.util.spec_from_file_location('independent_kl_oracle',outdir/'kl_oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
check(digest(outdir/'kl_oracle.py')=='501b0dcd6866cb9425f068b003404b31ade67265e68d369138990f5e56950f35','oracle pin')
saved_locality={(x['case_id'],x['arm']):x for x in result['locality_distributions']}
check(len(saved_locality)==40,'locality inventory')
locality=[];shared={'primary','same','random','opposite'}
for n in range(1,9):
 cid=f'locality-{n}';case=cases[cid];before=verified[('baseline',cid)]
 first_before=ns['raw'](before['outcome']['greedy_spans'][0])
 av=struct.unpack('<'+str(VOCAB)+'f',first_before)
 for arm in ('primary','same','broad','random','opposite'):
  after=verified[(arm,cid)];first_after=ns['raw'](after['outcome']['greedy_spans'][0])
  bv=struct.unpack('<'+str(VOCAB)+'f',first_after)
  structural=EDITED not in case['input_ids'] and arm in shared
  prefix_checks=0
  if structural:
   aa,bb=before['outcome'],after['outcome']
   for x,y in zip(aa['greedy_spans'],bb['greedy_spans']):
    pa,pb=x['metadata']['decoder_prefix'],y['metadata']['decoder_prefix']
    if pa!=pb or EDITED in pa:break
    check(ns['raw'](x)==ns['raw'](y),'structural greedy identity');prefix_checks+=1
   for name in ('positive','negative'):
    x,y=aa['label_spans'][name],bb['label_spans'][name]
    check(x['metadata']['decoder_prefix']==y['metadata']['decoder_prefix'],'structural teacher prefix')
    if EDITED not in x['metadata']['decoder_prefix']:
     check(ns['raw'](x)==ns['raw'](y),'structural teacher identity');prefix_checks+=1
   x,y=aa.get('candidate_span'),bb.get('candidate_span')
   if x is not None and y is not None and EDITED not in x['metadata']['decoder_prefix']:
    check(x['metadata']['decoder_prefix']==y['metadata']['decoder_prefix'] and ns['raw'](x)==ns['raw'](y),'structural candidate')
    prefix_checks+=1
   check(first_before==first_after,'structural first step')
  orig=before['terms'][case['expected']] if case['expected'] in ns['labels'] else before['candidate']
  mod=after['terms'][case['expected']] if case['expected'] in ns['labels'] else after['candidate']
  owner=saved_locality[(cid,arm)]
  check(owner['baseline_correct']==before['correct'] and owner['changed_correct']==after['correct']
   and owner['baseline_ids']==before['ids'] and owner['changed_ids']==after['ids']
   and owner['input_contains_edited_token']==(EDITED in case['input_ids'])
   and owner['first_step_structural_identity_expected']==structural
   and ns['close'](owner['baseline_expected_log_probability'],ns['score'](orig))
   and ns['close'](owner['changed_expected_log_probability'],ns['score'](mod)),'locality saved scores')
  metric=oracle.kl_tv(av,bv)
  locality.append({'case_id':cid,'arm':arm,'expected':case['expected'],
   'input_contains_edited_token':EDITED in case['input_ids'],'baseline_correct':before['correct'],
   'changed_correct':after['correct'],'baseline_ids':before['ids'],'changed_ids':after['ids'],
   'structural_identity_expected':structural,'structural_prefix_spans_checked':prefix_checks,
   'candidate_teacher_prefix':None if after['outcome']['candidate_span'] is None else after['outcome']['candidate_span']['metadata']['decoder_prefix'],
   'owner_kl':owner['kl'],'owner_tv':owner['tv'],'decimal':metric})
stream.close()
out={'schema':'flan-v2-independent-evaluation-science-review-v1',
 'scope':'saved bytes and recorded process/restoration only; physical execution and custody are cooperative; high-precision locality is a post-hoc numerical supplement',
 'phase_sha256':digest(root/'phase.json'),'exit_sha256':digest(root/'operator-exit.json'),
 'result_sha256':phase['result_sha256'],'events_sha256':phase['events_sha256'],'logits_sha256':phase['logits_sha256'],
 'development_phase_sha256':digest(dev/'phase.json'),'case_sha256':hashlib.sha256(case_raw).hexdigest(),
 'calls':phase['calls'],'outcomes_replayed':len(verified),'primary':primary,'targets':target_rows,'target_effects':effects,
 'comparisons':comparisons,'substitution':substitution,'recorded_restorations':len(ends),'localities':locality}
path=outdir/'independent-evaluation-science.json'
path.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
print('primary',primary,'targets',target_rows)
print('comparisons',comparisons)
print('substitution',substitution,'recorded restorations',len(ends),'calls',phase['calls'])
for x in locality:
 if x['input_contains_edited_token'] or x['arm']=='broad':
  print('locality',x['case_id'],x['arm'],'baseline_correct',x['baseline_correct'],'changed_correct',x['changed_correct'],
   'baseline_ids',x['baseline_ids'],'changed_ids',x['changed_ids'],'owner_KL',x['owner_kl'],'decimal_KL',x['decimal']['kl_decimal'][:35],
   'owner_TV',x['owner_tv'],'decimal_TV',x['decimal']['tv_decimal'][:35])
print('report',path,'sha256',digest(path))