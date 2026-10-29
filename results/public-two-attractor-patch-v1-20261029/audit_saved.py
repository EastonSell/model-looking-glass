"""Reconstruct a saved two-attractor patch attempt without importing a model.

The supplied hashes bind the saved journal, result and public cases. This
reader does not attest the process, checkpoint, exposure or execution history.
Only the four saved selected logits are replayable; full-vocabulary finiteness
is a runner-time check and cannot be independently reconstructed from them.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

SITES=('transformer.h.1.attn','transformer.h.3.attn')
labels=('is','are','was','were');width=768*4
CASE_SHA='6d0cbba21d15f9ab05df106617141c25a460f07be7ee57c53470f969eb79e518'


def need(ok,reason):
    if not ok:raise ValueError(reason)


def require_f32_subnormals():
    smallest=2.0**-149
    need(struct.unpack('<f',b'\x01\x00\x00\x00')[0]==smallest and
         struct.unpack('<f',b'\x01\x00\x00\x80')[0]==-smallest and
         struct.pack('<f',smallest)==b'\x01\x00\x00\x00','float32 subnormals are not preserved')


def readref(tensor_dir,ref,referenced,shape):
    digest=ref['sha256']
    need(isinstance(digest,str) and len(digest)==64 and
         all(c in '0123456789abcdef' for c in digest),'invalid tensor digest')
    need(ref['path']=='tensors/'+digest+'.f32le','unsafe or inconsistent tensor reference')
    need(ref['shape']==shape and type(ref['bytes']) is int,'tensor reference shape')
    path=tensor_dir/(digest+'.f32le')
    raw=path.read_bytes();referenced.add(path.name)
    need(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'raw tensor hash')
    need(len(raw)%4==0 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',raw)),'nonfinite tensor')
    return raw


def scores(record):
    raw=bytes.fromhex(record['scores']['float32_le_hex'])
    need(len(raw)==16,'logit byte length')
    vals=dict(zip(labels,struct.unpack('<4f',raw)))
    need(struct.pack('<4f',*(record['scores']['values'][x] for x in labels))==raw and
         all(math.isfinite(v) for v in vals.values()),'selected logits')
    return vals


def audit(journal_path,result_path,tensor_dir,cases):
    require_f32_subnormals()
    records=[json.loads(line) for line in journal_path.read_bytes().splitlines()]
    result_bytes=result_path.read_bytes();result=json.loads(result_bytes);need(records[0]['event']=='begin','start')
    need(records[0]['case_sha256']==CASE_SHA and result['case_sha256']==CASE_SHA and
         result['schema']=='public-two-attractor-patch-result-v1','case or result identity')
    need(records[0]['metadata']==result['model_metadata'],'model metadata')
    need(records[-1]['event']=='complete' and records[-1]['result_sha256']==hashlib.sha256(result_bytes).hexdigest()
         and records[-1]['forwards']==result['forwards'] and
         records[-1]['raw_tensor_bytes']==result['raw_tensor_bytes'],'complete')
    starts=[r for r in records if r['event']=='forward_start'];finished=[r for r in records if r['event']=='forward_result']
    need(len(starts)==len(finished)==result['forwards']==112,'forward count')
    need([r['event'] for r in records]==['begin']+['forward_start','model_returned','forward_result']*112+['complete'],'event schedule')
    returned=[r for r in records if r['event']=='model_returned']
    need([{k:v for k,v in r.items() if k!='event'} for r in starts]==
         [{k:v for k,v in r.items() if k!='event'} for r in returned],'model return order')
    need([(r['kind'],r.get('case_id'),r.get('target'),r.get('donor'),r.get('site')) for r in starts]==
         [(r['kind'],r.get('case_id'),r.get('target'),r.get('donor'),r.get('site')) for r in finished], 'ordered actions')
    clean={r['case_id']:r for r in finished if r['kind']=='clean'}
    need(len(clean)==32 and list(clean)==[c['case_id'] for c in cases],'clean schedule')
    need(all(set(r['vectors'])==set(SITES) for r in clean.values()),'clean vector sites')
    bykey={}
    for c in cases:
        key=(c['subject_pair'],c['order'],c['subject_number'])
        cell=(int(c['near_number']!=c['subject_number']),int(c['beside_number']!=c['subject_number']))
        bykey.setdefault(key,{})[cell]=c
    need(len(bykey)==8 and len(result['units'])==16,'unit count')
    rebuilt=[];index=32;referenced=set()
    for key in sorted(bykey):
        group=bykey[key];target=group[(0,0)];sign=1 if key[2]=='singular' else -1
        m={}
        for cell,c in group.items():
            values=scores(clean[c['case_id']]);m[cell]={tense:sign*(values[a]-values[b]) for tense,a,b in
                [('present','is','are'),('past','was','were')]}
        for site in SITES:
            effects={};patched={}
            for label,cell in [('null',(0,0)),('self',(0,0)),('near',(1,0)),('beside',(0,1)),('both',(1,1))]:
                r=finished[index];index+=1;donor=group[cell]
                need((r['kind'],r['site'],r['target'],r['donor'])==(label,site,target['case_id'],donor['case_id']),'donor or site')
                need(set(r['tensors'])=={'whole_before','whole_after','donor'},'action tensor set')
                expected_checks={'control_logits_equal_clean','landing_matches_donor','selected_slice_consistent',
                                 'target_matches_clean','untouched_positions_equal'}
                need(set(r['checks'])==expected_checks and all(v is True for v in r['checks'].values()),'action check flags')
                before=readref(tensor_dir,r['tensors']['whole_before'],referenced,[1,8,768])
                after=readref(tensor_dir,r['tensors']['whole_after'],referenced,[1,8,768])
                expected=readref(tensor_dir,clean[donor['case_id']]['vectors'][site],referenced,[768])
                stored=readref(tensor_dir,r['tensors']['donor'],referenced,[768])
                target_clean=readref(tensor_dir,clean[target['case_id']]['vectors'][site],referenced,[768]);offset=7*width
                need(len(before)==len(after)==8*width and len(expected)==width,'witness shape')
                need(before[offset:offset+width]==target_clean and after[offset:offset+width]==expected==stored,'landing')
                need(before[:offset]==after[:offset] and before[offset+width:]==after[offset+width:],'untouched positions')
                vals=scores(r)
                if label in ('null','self'):need(r['scores']['float32_le_hex']==clean[target['case_id']]['scores']['float32_le_hex'],'control logits')
                distance=math.sqrt(math.fsum((x-y)**2 for x,y in zip(struct.unpack('<768f',expected),struct.unpack('<768f',target_clean))))
                need(distance==r['displacement_l2'],'displacement')
                if label not in ('null','self'):
                    patched[label]={tense:sign*(vals[a]-vals[b]) for tense,a,b in
                                    [('present','is','are'),('past','was','were')]}
                    effects[label]={tense:patched[label][tense]-m[(0,0)][tense] for tense in ('present','past')}
            matching=[u for u in result['units'] if (u['subject_pair'],u['order'],u['subject_number'],u['site'])==(*key,site)]
            need(len(matching)==1,'result unit identity');unit=matching[0]
            need(unit['target']==target['case_id'],'result target identity')
            for tense in ('present','past'):
                clean_i=m[(1,1)][tense]-m[(1,0)][tense]-m[(0,1)][tense]+m[(0,0)][tense]
                patch_i=effects['both'][tense]-effects['near'][tense]-effects['beside'][tense]
                saved=unit['contrasts'][tense]
                need(saved['clean_interaction']==clean_i and saved['patch_interaction']==patch_i and
                     saved['patch_minus_clean']==patch_i-clean_i,'interaction')
                need(saved['patch_shifts']=={name:effects[name][tense] for name in effects},'individual shifts')
                for cell,value in m.items():
                    margin=value[tense];stored=saved['clean_accuracy'][str(cell[0])+str(cell[1])]
                    name=str(cell[0])+str(cell[1])
                    need(saved['clean_margins'][name]==margin and stored=={'correct':margin>0,'tie':margin==0},'accuracy/tie')
                for name in effects:
                    margin=patched[name][tense]
                    need(saved['patched_margins'][name]==margin and
                         saved['patched_accuracy'][name]=={'correct':margin>0,'tie':margin==0},'patched accuracy/tie')
            rebuilt.append(unit)
    need(index==112,'action tail')
    payloads=list(tensor_dir.glob('*.f32le'))
    need({p.name for p in payloads}==referenced,'unreferenced or missing tensors')
    need(sum(p.stat().st_size for p in payloads)==result['raw_tensor_bytes']<=16*1024*1024,'retained byte count')
    groups={'all':rebuilt}
    for field in ('site','subject_pair','order','subject_number'):
        for unit in rebuilt:groups.setdefault(f'{field}={unit[field]}',[]).append(unit)
    means={}
    for name,members in groups.items():
        means[name]={'units':len(members)}
        for tense in ('present','past'):
            means[name][tense]={metric:math.fsum(u['contrasts'][tense][metric] for u in members)/len(members)
                for metric in ('clean_interaction','patch_interaction','patch_minus_clean')}
            means[name][tense]['patch_shifts']={label:math.fsum(u['contrasts'][tense]['patch_shifts'][label] for u in members)/len(members)
                for label in ('near','beside','both')}
    need(means==result['means'],'group means')
    return {'status':'pass','forwards':112,'unit_site_pairs':len(rebuilt),'payload_files':len(payloads),
            'raw_bytes':result['raw_tensor_bytes'],'units':rebuilt,'means':means,
            'scope':'saved-byte and arithmetic consistency only; no execution, source or exposure attestation',
            'limitations':['Only four selected logits are saved; full-vocabulary finiteness cannot be independently replayed.']}


if __name__=='__main__':
    require_f32_subnormals()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--tensors',type=Path,required=True)
    parser.add_argument('--case-sha256',required=True);parser.add_argument('--journal-sha256',required=True)
    parser.add_argument('--result-sha256',required=True)
    args=parser.parse_args();case_bytes=args.cases.read_bytes()
    need(args.case_sha256==CASE_SHA and hashlib.sha256(case_bytes).hexdigest()==args.case_sha256,'case hash')
    need(hashlib.sha256(args.journal.read_bytes()).hexdigest()==args.journal_sha256,'journal hash')
    need(hashlib.sha256(args.result.read_bytes()).hexdigest()==args.result_sha256,'result hash')
    cases=json.loads(case_bytes)['cases'];need(len(cases)==32 and len({c['case_id'] for c in cases})==32,'case inventory')
    print(json.dumps(audit(args.journal,args.result,args.tensors,cases),sort_keys=True,allow_nan=False))
