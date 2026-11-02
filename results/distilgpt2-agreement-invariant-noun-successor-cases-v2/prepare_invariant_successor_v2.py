"""Build the fixed v2 invariant-noun successor using the tokenizer only."""
import argparse, hashlib, json, re, sys, types
from pathlib import Path

PROTOCOL_SHA='58e8b4fb26ca04afe727b76938c5ed3ece58a9dd7f37e5a8a18fd7504ef80deb'
PROTOCOL_REVISION='6f241157c7617be04fddf2624ccad6f24388cd85'
V1_CASE_SHA='50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827'
V1_PREPARER_SHA='983eb7cac32f8792fc61cabae3539b63d05e3907768c51c5bac89fb6cc23335e'
SCHEDULE_SHA='bdb9f77cf12b191f2a713324077f6f175b1facdce7989ff79fe0c7390aec21a4'
ENTRY_SHA='3c4607cbf70a1de57b99d5cc568385859c24cce111dc711fd8806c54b9bd1795'
ACTION_SHA='f0625d27ee196eeeb1a85a3577238f0cbcad19017fc6596dfe429dcbc60c610d'
NOUNS=('salmon','trout','cod','fish','moose','elk','tuna','swine','offspring','aircraft','spacecraft')
ATTRACTORS=(('bell','bells'),('door','doors'),('chair','chairs'),('bridge','bridges'),('fence','fences'),('hedge','hedges'),('tower','towers'),('garden','gardens'),('river','rivers'),('stone','stones'),('flower','flowers'),('cloud','clouds'))
FRAMES=('At sunrise,','Before sunset,')

def load_source(path, expected, name):
    raw=Path(path).read_bytes()
    if len(raw)>1024*1024 or hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('source bytes differ: '+name)
    m=types.ModuleType(name); m.__file__=str(path); exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__); return m

def choose(v1,tok,old_subjects,old_attractors,old_cases):
    forbidden=set(old_subjects)
    for row in old_cases:
        if row['role']=='target': forbidden.add(row['token_ids'][row['subject_index']])
        if row['context_index']!=0: forbidden.add(row['token_ids'][-1])
    forms=v1.token_forms(tok,forbidden)
    selected=[]; nd=[]
    for form in NOUNS:
        ids=v1.encode(tok,' '+form); a=v1.encode(tok,' this '+form); b=v1.encode(tok,' these '+form)
        if len(ids)!=1: reason='not_one_token'
        elif a!=[428,*ids] or b!=[777,*ids]: reason='determiner_alignment'
        elif ids[0] in forbidden or form.casefold() in forms: reason='previous_or_selected'
        elif not all(v1.encode(tok,prefix+answer)==seq+[answer_id] for prefix,seq in ((' this '+form,a),(' these '+form,b)) for answer,answer_id in v1.ANSWERS): reason='answer_boundary'
        else:
            reason='selected'; selected.append({'form':form,'token_id':ids[0]}); forbidden.add(ids[0]); forms.add(form.casefold())
        nd.append({'form':form,'ids':ids,'decision':reason})
        if len(selected)==2: break
    if len(selected)!=2: raise ValueError('fixed noun pool exhausted')
    forbidden.update(old_attractors); forms.update(v1.token_forms(tok,old_attractors))
    pairs=[]; ad=[]
    for pair in ATTRACTORS:
        ids=[v1.encode(tok,' '+x) for x in pair]
        if any(len(x)!=1 for x in ids): reason='not_one_token'
        elif ids[0][0]==ids[1][0] or any(x[0] in forbidden for x in ids): reason='previous_or_selected_token'
        elif pair[0].casefold()==pair[1].casefold() or any(x.casefold() in forms for x in pair): reason='previous_or_selected_form'
        else:
            reason='selected'; pairs.append({'forms':list(pair),'token_ids':[x[0] for x in ids]}); forbidden.update(x[0] for x in ids); forms.update(x.casefold() for x in pair)
        ad.append({'forms':list(pair),'ids':ids,'decision':reason})
        if len(pairs)==2: break
    if len(pairs)!=2: raise ValueError('fixed attractor pool exhausted')
    return selected,pairs,nd,ad

def build_discovery(v1,tok,nouns,pairs):
    rows={}
    for f,frame in enumerate(FRAMES):
        for c in range(3):
            suffix=' nearby' if c==0 else ' near the '+pairs[f]['forms'][c-1]
            for n,noun in enumerate(nouns):
                for det in ('this','these'):
                    identity=f'v2_discovery_target_f{f}_c{c}_n{n}_{det}'
                    row=v1.case_row(tok,identity,'discovery','target',f,c,n,det,f'{frame} {det} {noun["form"]}{suffix}')
                    det_id=428 if det=='this' else 777
                    if row['token_ids'].count(det_id)!=1 or row['token_ids'].count(noun['token_id'])!=1: raise ValueError('ambiguous target coordinate')
                    row['determiner_index']=row['token_ids'].index(det_id); row['subject_index']=row['token_ids'].index(noun['token_id']); rows[identity]=row
            identity=f'v2_discovery_fragment_f{f}_c{c}'
            rows[identity]=v1.case_row(tok,identity,'discovery','fragment',f,c,None,None,frame+suffix)
        for det in ('this','these'):
            identity=f'v2_discovery_demonstrative_f{f}_{det}'
            row=v1.case_row(tok,identity,'discovery','demonstrative',f,0,None,det,f'{frame} {det} nearby')
            det_id=428 if det=='this' else 777
            if row['token_ids'].count(det_id)!=1: raise ValueError('ambiguous demonstrative coordinate')
            row['determiner_index']=row['token_ids'].index(det_id); rows[identity]=row
    if len(rows)!=34: raise ValueError('replacement case count differs')
    return rows

def verify_alignment(rows):
    for prefix in ('v2_discovery','evaluation'):
        for f in range(2):
            a=rows[f'{prefix}_demonstrative_f{f}_this']; b=rows[f'{prefix}_demonstrative_f{f}_these']; d=a['determiner_index']
            if b['determiner_index']!=d or len(a['token_ids'])!=len(b['token_ids']) or a['token_ids'][:d]+a['token_ids'][d+1:]!=b['token_ids'][:d]+b['token_ids'][d+1:]: raise ValueError('demonstrative alignment differs')
            for c in range(3):
                group=[rows[f'{prefix}_target_f{f}_c{c}_n{n}_{det}'] for n in range(2) for det in ('this','these')]
                d,n=group[0]['determiner_index'],group[0]['subject_index']
                if d==n or any(len(x['token_ids'])!=len(group[0]['token_ids']) or x['determiner_index']!=d or x['subject_index']!=n for x in group): raise ValueError('target positions differ')
                if any(any(a!=b for i,(a,b) in enumerate(zip(group[0]['token_ids'],row['token_ids'])) if i not in (d,n)) for row in group): raise ValueError('target alignment differs')
            for n in range(2):
                for det in ('this','these'):
                    a=rows[f'{prefix}_target_f{f}_c1_n{n}_{det}']['token_ids']; b=rows[f'{prefix}_target_f{f}_c2_n{n}_{det}']['token_ids']
                    if a[:-1]!=b[:-1] or a[-1]==b[-1]: raise ValueError('attractor alignment differs')
            a=rows[f'{prefix}_fragment_f{f}_c1']['token_ids']; b=rows[f'{prefix}_fragment_f{f}_c2']['token_ids']
            if a[:-1]!=b[:-1] or a[-1]==b[-1]: raise ValueError('fragment alignment differs')

def compare_inventory(entry,records,cases,answers):
    own={('planned-'+r['case_id']):r for r in cases if r['split']=='evaluation'}
    complete=[tuple(r['token_ids']) for r in cases]; complete += [t+(x,) for t in list(complete) for x in answers]
    full=set(complete); prefixes={row[:n] for row in complete for n in range(1,len(row)+1)}; seen=set(); ids=set(); checked=0
    for item in records:
        rid=item.get('record_id')
        if type(rid)!=str or rid in ids: raise ValueError('invalid inventory ID')
        ids.add(rid)
        if rid in own:
            row=own[rid]; expected={'category':'planned','case_role':row['role'],'public_text':row['text'],'token_ids':row['token_ids'],'candidate_token_ids':answers}
            if any(item.get(k)!=v or type(item.get(k)) is not type(v) for k,v in expected.items()) or entry._grouped_candidates(item) or item.get('target_token_id') is not None or any(k in item for k in ('attention_mask','scored_prefix_token_ids','scored_candidate_token_ids')): raise ValueError('own evaluation record differs')
            seen.add(rid); continue
        for seq in entry.record_sequences(item):
            checked+=1
            if seq in prefixes or any(seq[:n] in full for n in range(1,len(seq)+1)): raise ValueError('known input prefix collision')
    if seen!=set(own): raise ValueError('own evaluation reservation missing')
    return checked

def main():
    p=argparse.ArgumentParser()
    for name in ('protocol','v1-cases','v1-preparer','v1-schedule','v1-entry','tokenizer-json','prior-binding','prior-direction-cases','prior-subject-cases','prior-substitution-cases','inventory','inventory-version','inventory-revision','inventory-sha256','natural-support','natural-successor','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',a.inventory_revision) or not re.fullmatch('[0-9a-f]{64}',a.inventory_sha256): raise ValueError('invalid inventory identity')
    if hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest()!=PROTOCOL_SHA: raise ValueError('protocol differs')
    v1=load_source(a.v1_preparer,V1_PREPARER_SHA,'_fixed_v1_preparer'); schedule=load_source(a.v1_schedule,SCHEDULE_SHA,'_fixed_v1_schedule'); entry=load_source(a.v1_entry,ENTRY_SHA,'_fixed_v1_entry')
    old=json.loads(v1.read(a.v1_cases,2*1024*1024,V1_CASE_SHA)); prior=[json.loads(v1.read(x,2*1024*1024,h)) for x,h in zip((a.prior_binding,a.prior_direction_cases,a.prior_subject_cases,a.prior_substitution_cases),v1.PRIOR_SHAS)]
    raw=v1.read(a.inventory,32*1024*1024,a.inventory_sha256); inventory=v1.verify_inventory(raw,a.inventory_version)
    support=json.loads(v1.read(a.natural_support,2*1024*1024,v1.SUPPORT_SHAS[0])); successor=json.loads(v1.read(a.natural_successor,2*1024*1024,v1.SUPPORT_SHAS[1]))
    tokenraw=v1.read(a.tokenizer_json,2*1024*1024,v1.TOKENIZER_SHA)
    if 'torch' in sys.modules or 'transformers' in sys.modules: raise ValueError('model import prohibited')
    from tokenizers import Tokenizer,__version__
    if __version__!='0.22.2' or 'torch' in sys.modules or 'transformers' in sys.modules: raise ValueError('tokenizer runtime differs')
    tok=Tokenizer.from_str(tokenraw.decode('utf-8'))
    if any(v1.encode(tok,answer)!=[value] for answer,value in v1.ANSWERS): raise ValueError('fixed answers differ')
    old_subjects,old_attractors=v1.old_tokens(*prior); nouns,pairs,nd,ad=choose(v1,tok,old_subjects,old_attractors,old['cases'])
    rows=build_discovery(v1,tok,nouns,pairs)
    for row in old['cases']:
        if row['split']=='evaluation':
            if v1.encode(tok,row['text'])!=row['token_ids'] or any(v1.encode(tok,row['text']+a)!=row['token_ids']+[t] for a,t in v1.ANSWERS): raise ValueError('retained evaluation tokenizer differs')
            rows[row['case_id']]=row
    verify_alignment(rows)
    ids=[('v2_'+cid if cid.startswith('discovery_') else cid) for cid,_ in schedule.cases()]
    actions=[('v2_'+cid if cid.startswith('discovery_') else cid,action) for cid,action in schedule.actions()]
    if hashlib.sha256(json.dumps(actions,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()!=ACTION_SHA or len(rows)!=68 or set(rows)!=set(ids): raise ValueError('fixed schedule differs')
    cases=[rows[x] for x in ids]; groups=v1.make_groups(cases); proposed=[x for group in groups for x in group]
    support_rows=[v1.token_row(x) for x in support['rows']]; successor_rows=[v1.token_row(x['input_ids']) for x in successor['cases']]
    if len(support_rows)!=128 or len(successor_rows)!=256 or any(v1.related(x,y) for x in proposed for y in support_rows+successor_rows): raise ValueError('natural support prefix collision')
    checked=compare_inventory(entry,inventory['records'],cases,[t for _,t in v1.ANSWERS])
    rowsha=hashlib.sha256(json.dumps([r['token_ids'] for r in cases],separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()
    result={'schema':'agreement-invariant-noun-successor-cases-v2','protocol_revision':PROTOCOL_REVISION,'protocol_sha256':PROTOCOL_SHA,'previous_case_sha256':V1_CASE_SHA,'tokenizer_sha256':v1.TOKENIZER_SHA,'model_id':v1.MODEL,'model_revision':v1.REVISION,'tokenizers_version':__version__,'inventory_revision':a.inventory_revision,'inventory_version':a.inventory_version,'inventory_sha256':a.inventory_sha256,'inventory_sequences_checked':checked,'public_support_sha256':list(v1.SUPPORT_SHAS),'prior_bundle_sha256':list(v1.PRIOR_SHAS),'selection':{'nouns':nouns,'noun_decisions':nd,'attractors':pairs,'attractor_decisions':ad},'answers':[t for _,t in v1.ANSWERS],'ordered_rows_sha256':rowsha,'actions_sha256':ACTION_SHA,'expected_forwards':612,'cases':cases,'scope':'Tokenizer-only public construction; pending inputs and current custody require separate observation.'}
    data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with Path(a.output).open('xb') as f: f.write(data)
    print(json.dumps({'case_sha256':hashlib.sha256(data).hexdigest(),'ordered_rows_sha256':rowsha,'actions_sha256':ACTION_SHA,'cases':len(cases),'inventory_sequences_checked':checked,'selected_nouns':[x['form'] for x in nouns],'selected_attractors':[x['forms'] for x in pairs],'model_forwards':0},sort_keys=True))
if __name__=='__main__': main()
