"""Check fixed multisuffix BERT inputs against named public manifests only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from tokenizers import Tokenizer

CASE_SHA='bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e'
TOKENIZER_SHA='ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'
PENDING_ANTONYM_SHA='284fc9adc0cc3caa52ea5d122c1f08db50c8dda2d881297e2dcd1f2229cb6386'
PRIOR_AUDIT_COMMIT='8db4866943fbd9ff8a981a07fb2c58a3de364822'
PRIOR_AUDIT_PATH='audits/bert_agreement_suffix_v1/independent-exposure-check.json'
PREVIOUS_SUFFIX=('8db4866943fbd9ff8a981a07fb2c58a3de364822','experiments/bert_agreement_suffix_cases_v1.json','1da4f8e4773009fedc750480c21597fc60ecaa0c990dcff0af15a5c7101c3dca',64)
SIGNED=('5060fc419610ca7f0e2c8ec5095fb5ce93cc3531','experiments/bert_cue_context_signed_cases_v1.json','843649c3f105e26185d96f31d769b0585d219bb7a45ba445cf2c2548598927df',32)
CONTEXT_COMMIT='91d4a3913dacd05436e3ce5248f96aaf2b7aefd9'
CURRENT='2a492669b2ebb16d2612366fbefdf80c4265faca'
EXTRA_NAMES=('bert_color_preference_public_exposures_v1.json','bert_color_preference_public_exposures_v2.json','bert_cue_dose_response_public_exposures_v2.json')


def digest(raw): return hashlib.sha256(raw).hexdigest()

def git_read(repo, rev, path):
    return subprocess.check_output(['git','-C',str(repo),'show',rev+':'+path])

def checked(row):
    ids,mask=row['input_ids'],row['attention_mask']
    if (type(ids) is not list or not 2<=len(ids)<=64 or any(type(x) is not int or not 0<=x<30522 for x in ids)
            or ids[0]!=101 or ids[-1]!=102 or ids.count(103)!=1 or mask != [1]*len(ids)):
        raise ValueError('invalid BERT row')
    return tuple(ids)

def extract(name,document):
    if name=='bert_negation_public_cases_v1.json':
        out=[]
        for case in document['cases']:
            for cue in ('not','never','just','also'):
                ids=list(case['just_ids']); index=document['indices']['cue']
                if ids[index]!=document['cues']['just']: raise ValueError('invalid negation base')
                ids[index]=document['cues'][cue]
                out.append(checked({'input_ids':ids,'attention_mask':document['attention_mask']}))
        return out
    if name=='bert_cue_context_transfer_cases_v1.json' or name=='bert_cue_context_signed_cases_v1.json':
        return [checked(row) for pair in document['pairs'] for context in pair['contexts'] for row in context['rows'].values()]
    if name=='bert_agreement_suffix_cases_v1.json':
        return [checked(row) for case in document['cases'] for row in case['variants'].values()]
    if name.startswith('bert_cue_') and name.endswith('_cases_v1.json'):
        return [checked(row) for case in document['cases'] for row in case['rows'].values()]
    if name in ('bert_color_edit_public_cases_v1.json','bert_color_preference_public_cases_v2.json'):
        return [checked(row) for row in document['cases']]
    if name in ('bert_color_edit_public_exposures_v2.json','bert_color_preference_public_exposures_v1.json','bert_color_preference_public_exposures_v2.json'):
        return [checked(row) for row in document['records']]
    if name=='cross_model_agreement_cases_v1.json': return [checked(row['bert']) for row in document['cases']]
    return [checked(row) for row in document['rows']]

def collision(a,b):
    return any(x[:len(y)]==y or y[:len(x)]==x for x,y in ((a,b),(a[:-1],b[:-1])))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','cases','tokenizer','pending-antonym','output'): parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    raw=a.cases.read_bytes()
    if digest(raw)!=CASE_SHA or digest(a.tokenizer.read_bytes())!=TOKENIZER_SHA: raise ValueError('case or tokenizer bytes differ')
    document=json.loads(raw); tok=Tokenizer.from_file(str(a.tokenizer))
    variants=('terminal','period','present','outside','awake')
    if document['variants']!=list(variants) or len(document['cases'])!=32: raise ValueError('manifest dimensions differ')
    new=[]; labels=[]
    for i,case in enumerate(document['cases']):
        if case['index']!=i or case['same_donor_index'] != (i^2) or case['opposite_donor_index'] != (i^3):
            raise ValueError('case order or donor index differs')
        for variant in variants:
            row=case['variants'][variant]; ids=checked(row)
            encoded=tok.encode(row['text'],add_special_tokens=True)
            if tuple(encoded.ids)!=ids or encoded.attention_mask!=row['attention_mask']: raise ValueError('tokenizer mismatch')
            if row['subject_position']!=6 or row['output_position']!=10 or ids[10]!=103: raise ValueError('coordinate mismatch')
            new.append(ids); labels.append((case['case_id'],variant))
        for donor in (case['same_donor_index'],case['opposite_donor_index']):
            other=document['cases'][donor]
            for variant in variants:
                left=case['variants'][variant]['input_ids']; right=other['variants'][variant]['input_ids']
                if [j for j,(x,y) in enumerate(zip(left,right)) if x!=y]!=[6]: raise ValueError('donor differs outside subject')
        same,opposite=[document['cases'][case[key]] for key in ('same_donor_index','opposite_donor_index')]
        if (same['lemma']!=opposite['lemma'] or same['lemma']==case['lemma'] or same['number']!=case['number']
                or opposite['number']==case['number']): raise ValueError('donor number or lemma differs')
    prior=json.loads(git_read(a.repo,PRIOR_AUDIT_COMMIT,PRIOR_AUDIT_PATH))
    source_specs=[]
    for source in prior['sources']:
        commit=source['commit'] or CONTEXT_COMMIT
        path=source['path'] if source['commit'] else 'experiments/bert_cue_context_transfer_cases_v1.json'
        source_specs.append((commit,path,source['sha256'],source['rows']))
    source_specs.extend([PREVIOUS_SUFFIX,SIGNED])
    for name in EXTRA_NAMES:
        path='experiments/'+name
        content=git_read(a.repo,CURRENT,path)
        source_specs.append((CURRENT,path,digest(content),37 if 'color_preference' in name else 32))
    sources=[]; all_rows=[]; collisions=[]
    for rev,path,sha,expected in source_specs:
        content=git_read(a.repo,rev,path)
        if digest(content)!=sha: raise ValueError('source digest differs: '+path)
        rows=extract(Path(path).name,json.loads(content))
        if len(rows)!=expected: raise ValueError('source row count differs: '+path)
        hits=[(i,j) for i,x in enumerate(new) for j,y in enumerate(rows) if collision(x,y)]
        collisions.extend({'path':path,'proposal_index':i,'source_index':j} for i,j in hits)
        sources.append({'commit':rev,'path':path,'sha256':sha,'rows':len(rows),'collisions':len(hits)})
        all_rows.extend((path,row) for row in rows)
    pending_raw=a.pending_antonym.read_bytes()
    if digest(pending_raw)!=PENDING_ANTONYM_SHA: raise ValueError('pending antonym bytes differ')
    pending_rows=[checked(row) for row in json.loads(pending_raw)['cases']]
    if len(pending_rows)!=55: raise ValueError('pending antonym row count differs')
    path='pending:bert-antonym-preference-cases-v1-corrected-20261029.json'
    hits=[(i,j) for i,x in enumerate(new) for j,y in enumerate(pending_rows) if collision(x,y)]
    collisions.extend({'path':path,'proposal_index':i,'source_index':j} for i,j in hits)
    sources.append({'commit':None,'path':path,'sha256':PENDING_ANTONYM_SHA,'rows':len(pending_rows),'collisions':len(hits)})
    all_rows.extend((path,row) for row in pending_rows)
    within=[(i,j) for (i,x),(j,y) in itertools.combinations(enumerate(new),2) if collision(x,y)]
    expected=[(5*i,5*i+j) for i in range(32) for j in range(1,5)]
    unexpected=[x for x in within if x not in expected]
    if sorted(within)!=expected: raise ValueError('within-proposal full-prefix relations differ')
    nouns=sorted({case[key] for case in document['cases'] for key in ('subject','attractor')})
    noun_overlap={}
    for word in nouns:
        ids=tok.encode(word,add_special_tokens=False).ids
        if len(ids)!=1: raise ValueError('noun not single wordpiece')
        noun_overlap[word]=sorted({path for path,row in all_rows if ids[0] in row})
    frames=('At the park','Beyond the village')
    frame_overlap={}
    for frame in frames:
        ids=tuple(tok.encode(frame,add_special_tokens=False).ids)
        frame_overlap[frame]=sorted({path for path,row in all_rows if any(row[i:i+len(ids)]==ids for i in range(len(row)-len(ids)+1))})
    result={'schema':'bert-multisuffix-public-comparison-v1','case_sha256':CASE_SHA,'proposal_rows':len(new),
            'within_expected':len(expected),'within_unexpected':unexpected,'source_rows':len(all_rows),
            'cross_pairs':len(new)*len(all_rows),'collisions':collisions,'sources':sources,
            'noun_token_sources':noun_overlap,'frame_token_sources':frame_overlap,
            'scope':'Named public sources and one SHA-fixed pending proposal only, including repeated references; no private, unknown or pretraining guarantee.'}
    with a.output.open('x',encoding='utf-8',newline='\n') as f: json.dump(result,f,sort_keys=True,indent=2); f.write('\n')
    print('rows',len(new),'prior_refs',len(all_rows),'cross_pairs',len(new)*len(all_rows),
          'expected_internal',len(expected),'unexpected',len(unexpected),'cross_collisions',len(collisions))
    if collisions or unexpected: raise ValueError('public collision found')

if __name__=='__main__': main()
