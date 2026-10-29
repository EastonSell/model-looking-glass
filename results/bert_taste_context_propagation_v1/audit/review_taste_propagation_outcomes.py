"""Second saved-byte check of target, full-vocabulary and propagation effects.

No model or study code is imported. The operator/source and untouched whole
activation tensors require separate verification.
"""
import argparse
import base64
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',required=True,type=Path)
parser.add_argument('--manifest',required=True,type=Path)
parser.add_argument('--tokenizer-json',required=True,type=Path)
parser.add_argument('--result-sha256',required=True)
parser.add_argument('--manifest-sha256',required=True)
args=parser.parse_args()
run=args.run
manifest_path=args.manifest
vocab_path=args.tokenizer_json
def digest(data):return hashlib.sha256(data).hexdigest()
def check(condition,message):
    if not condition:raise ValueError(message)
def probe():
    raw=bytes.fromhex('0100000000000000');v=struct.unpack('<d',raw)[0]
    check(v>0 and struct.pack('<d',v*1.)==raw and struct.pack('<d',float('5e-324'))==raw,'binary64 denormals')
def close(a,b,message):
    if a is None or b is None:check(a is None and b is None,message);return
    check(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),message)
    check(math.isclose(a,b,rel_tol=3e-13,abs_tol=5e-324),f'{message}: {a} != {b}')
def f32(encoded,width):
    raw=base64.b64decode(encoded,validate=True)
    check(len(raw)==4*width and base64.b64encode(raw).decode()==encoded,'float32 bytes')
    vals=struct.unpack(f'<{width}f',raw);check(all(math.isfinite(x) for x in vals),'nonfinite float32')
    return raw,vals
def norm(v):return math.sqrt(math.fsum(x*x for x in v))
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def centered(v):
    mean=math.fsum(v)/len(v)
    return tuple(math.fsum((x,-mean)) for x in v)
def projection(v,d):
    vn,dn=norm(v),norm(d)
    if dn<1e-12:return {'coefficient':None,'cosine':None,'vector_norm':vn,'direction_norm':dn,'residual_norm':None}
    numerator=dot(v,d);coefficient=numerator/dot(d,d)
    return {'coefficient':coefficient,'cosine':None if vn<1e-12 else numerator/(vn*dn),
            'vector_norm':vn,'direction_norm':dn,
            'residual_norm':norm(math.fsum((a,-coefficient*b)) for a,b in zip(v,d))}
def probabilities(v):
    m=max(v);a=[math.exp(x-m) for x in v];z=math.fsum(a)
    return tuple(x/z for x in a)
def verify_logits(record,width,tokens):
    raw,vals=f32(record['mask_logits_float32_le_base64'],width);s=record['summary']
    check(digest(raw)==s['mask_logits_sha256'],'mask logit hash')
    check(struct.unpack_from('<I',raw,4086*4)[0]==s['sweet_bits'] and struct.unpack_from('<I',raw,14768*4)[0]==s['sour_bits'],'target bits')
    close(s['sweet_logit'],vals[4086],'sweet logit');close(s['sour_logit'],vals[14768],'sour logit')
    close(s['sweet_minus_sour'],math.fsum((vals[4086],-vals[14768])),'taste margin')
    top=heapq.nsmallest(5,range(width),key=lambda i:(-vals[i],i))
    check(s['top_five_ids']==top and s['top_five_tokens']==[tokens[i] for i in top],'top token identity')
    p=probabilities(vals);close(s['sweet_probability'],p[4086],'sweet probability');close(s['sour_probability'],p[14768],'sour probability')
    return vals,p
def check_projection(saved,actual,where):
    for k,v in actual.items():close(saved[k],v,where+' '+k)
def check_aggregates(cells,stems,stored,groups,fields):
    for group in groups:
      for field in fields:
        ref=stored[group][field];values={}
        for stem in stems:
          entry=ref['stems'][stem];values[stem]={}
          for template in ('after','with'):
            x=[c[field] for c in cells if c['group']==group and c['stem']==stem and c['template']==template]
            check(len(x)==2,'recipient aggregate count')
            value=None if any(y is None for y in x) else math.fsum(x)/2
            close(entry[template]['value'],value,'recipient aggregate '+field)
            check(entry[template]['missing_count']==sum(y is None for y in x),'recipient missing count')
            values[stem][template]=value
          a,b=values[stem]['after'],values[stem]['with'];difference=None if a is None or b is None else math.fsum((a,-b))
          close(entry['difference'],difference,'stem difference '+field);values[stem]['difference']=difference
        for key in ('after','with','difference'):
          x=[values[s][key] for s in stems];value=None if any(y is None for y in x) else math.fsum(x)/len(x)
          close(ref['means'][key]['value'],value,'six-stem mean '+field)
          check(ref['means'][key]['missing_count']==sum(y is None for y in x),'stem missing count')

probe()
result_raw=(run/'result.json').read_bytes();check(digest(result_raw)==args.result_sha256,'result hash')
result=json.loads(result_raw);manifest_raw=manifest_path.read_bytes()
check(digest(manifest_raw)==args.manifest_sha256,'manifest hash')
manifest=json.loads(manifest_raw);stems=manifest['selected_stems'];rows=manifest['rows']
check(len(stems)==6 and len(rows)==24 and result['status']=='complete' and result['completed_forwards']==960,'run completion')
vocab_raw=vocab_path.read_bytes();check(digest(vocab_raw)=='ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98','vocabulary hash')
vocab=json.loads(vocab_raw)['model']['vocab'];tokens={v:k for k,v in vocab.items()};check(len(tokens)==30522,'vocabulary size')
sites={'early_ingredient':(0,7),'late_ingredient':(3,7),'late_the':(3,1),'late_tasted':(3,3)}
late=('late_ingredient','late_the','late_tasted','late_mask');axes=('lex','context','orth1','orth2')
by_row={r['id']:r for r in rows}
cells={(c['recipient'],c['group']):c for c in result['comparisons']}
prop={(c['recipient'],c['group']):c for c in result['propagation']}
effects={(e['recipient'],e['site'],e['mode']):e for e in result['effects']}
check(len(cells)==96 and len(prop)==96 and len(effects)==864,'result cell counts')
verified_logits=0;verified_later_vectors=0;calculated_cells=[];calculated_prop=[]
for stem in stems:
    raw=(run/(stem+'.jsonl')).read_bytes();item=result['journals'][stem+'.jsonl']
    check(item=={'size':len(raw),'sha256':digest(raw)} and raw.endswith(b'\n'),'journal hash')
    events=[json.loads(line) for line in raw.splitlines()]
    complete=[e for e in events if e.get('event')=='forward_complete']
    check(len(complete)==160,'stem forward count')
    records={(e['row_id'],e['layer'],e['position'],e['mode']):e for e in complete}
    check(len(records)==160,'duplicate action identity')
    group=[r for r in rows if r['stem']==stem];check(len(group)==4,'stem rows')
    clean={}
    for row in group:
      event=records[row['id'],0,7,'clean_capture'];check(event['model_entry']=={'input_ids':row['input_ids'],'attention_mask':row['attention_mask']},'clean model entry')
      vals,prob=verify_logits(event,30522,tokens);verified_logits+=1
      check(event['summary']==result['baselines'][row['id']],'saved baseline')
      later={k:f32(event['later_float32_le_base64'][k],768)[1] for k in late}
      clean[row['id']]={'event':event,'vals':vals,'prob':prob,'later':later}
    for template in ('after','with'):
      s=clean[stem+'-'+template+'-sugar']['event']['summary'];l=clean[stem+'-'+template+'-lemon']['event']['summary']
      eligible=(s['sweet_minus_sour']>0 and l['sweet_minus_sour']<0 and bool({4086,14768}.intersection(s['top_five_ids'])) and bool({4086,14768}.intersection(l['top_five_ids'])))
      check(result['eligibility_descriptive'][stem+'-'+template] is eligible,'descriptive eligibility')
    for row in group:
      ident=row['id'];base=clean[ident];other='sugar' if row['ingredient']=='lemon' else 'lemon'
      donor=clean[stem+'-'+row['template']+'-'+other];sgn=1 if other=='sugar' else -1
      direction=centered(tuple(a-b for a,b in zip(donor['vals'],base['vals'])))
      for site,(layer,pos) in sites.items():
        saved=cells[ident,site];calculated={'stem':stem,'template':row['template'],'group':site};paired={}
        for axis in axes:
          pair=[]
          for suffix in ('plus','minus'):
            rec=records[ident,layer,pos,axis+'_'+suffix]
            check(rec['model_entry']=={'input_ids':row['input_ids'],'attention_mask':row['attention_mask']},'model entry')
            vals,prob=verify_logits(rec,30522,tokens);verified_logits+=1
            e=math.fsum((math.fsum((vals[4086],-vals[14768])),-base['event']['summary']['sweet_minus_sour']))
            close(effects[ident,site,axis+'_'+suffix]['effect'],e,'target effect')
            t=math.fsum(abs(a-b) for a,b in zip(prob,base['prob']))/2
            pair.append((rec,vals,e,t))
          plus,minus=pair;odd=math.fsum((plus[2],-minus[2]))/2;even=math.fsum((plus[2],minus[2]))/2
          a=saved['axes'][axis];close(a['odd'],odd,'target odd');close(a['even'],even,'target even');close(a['signed_odd'],sgn*odd,'signed odd')
          close(a['tv_plus'],plus[3],'TV plus');close(a['tv_minus'],minus[3],'TV minus')
          output=centered(tuple(math.fsum((p,-m))/2 for p,m in zip(plus[1],minus[1])))
          check_projection(a['output_projection'],projection(output,direction),'output projection')
          paired[axis]=odd
          if site=='early_ingredient':
            for name in late:
              up=f32(plus[0]['later_float32_le_base64'][name],768)[1];down=f32(minus[0]['later_float32_le_base64'][name],768)[1];z=base['later'][name]
              od=tuple(math.fsum((p,-m))/2 for p,m in zip(up,down))
              ev=tuple(math.fsum((p,m,-2*x))/2 for p,m,x in zip(up,down,z))
              ld=tuple(a-b for a,b in zip(donor['later'][name],z));saved_late=a['later'][name]
              check(base64.b64decode(saved_late['odd_float64_le_base64'],validate=True)==b''.join(struct.pack('<d',v) for v in od),'later odd bits')
              check(base64.b64decode(saved_late['even_float64_le_base64'],validate=True)==b''.join(struct.pack('<d',v) for v in ev),'later even bits')
              check_projection(saved_late['projection'],projection(od,ld),'later projection')
              verified_later_vectors+=2
        calculated['taste_specificity']=math.fsum((sgn*paired['lex'],-math.fsum((abs(paired['orth1']),abs(paired['orth2'])))/2))
        calculated['context_specificity']=math.fsum((sgn*paired['lex'],-abs(paired['context'])))
        calculated['lex_signed_odd']=sgn*paired['lex']
        for k in ('taste_specificity','context_specificity','lex_signed_odd'):close(saved[k],calculated[k],k)
        calculated_cells.append(calculated)
        if site=='early_ingredient':
          for name in late:
            record=prop[ident,name];pc={'stem':stem,'template':row['template'],'group':name}
            for axis in axes:
              x=saved['axes'][axis]['later'][name]['projection']
              close(record[axis+'_projection'],x['coefficient'],'propagation projection')
              pc[axis+'_projection']=x['coefficient']
            terms=[pc[axis+'_projection'] for axis in ('lex','orth1','orth2')]
            pc['propagation_specificity']=None if any(x is None for x in terms) else math.fsum((terms[0],-math.fsum((abs(terms[1]),abs(terms[2])))/2))
            close(record['propagation_specificity'],pc['propagation_specificity'],'propagation specificity')
            calculated_prop.append(pc)
check(verified_logits==792 and verified_later_vectors==768,'full output/later count')
check_aggregates(calculated_cells,stems,result['taste_aggregates'],tuple(sites),('taste_specificity','context_specificity','lex_signed_odd'))
check_aggregates(calculated_prop,stems,result['propagation_aggregates'],late,('propagation_specificity','lex_projection','context_projection','orth1_projection','orth2_projection'))
probe()
print(json.dumps({'status':'pass','full_vocab_vectors':verified_logits,'later_vectors':verified_later_vectors,
  'taste_cells':len(calculated_cells),'propagation_cells':len(calculated_prop),
  'eligibility_descriptive':result['eligibility_descriptive'],
  'taste_means':{s:{k:result['taste_aggregates'][s][k]['means'] for k in ('taste_specificity','context_specificity','lex_signed_odd')} for s in sites},
  'propagation_means':{s:{k:result['propagation_aggregates'][s][k]['means'] for k in ('propagation_specificity','lex_projection')} for s in late},
  'scope':'closed saved vectors and summaries; no model, execution attestation or independently retained whole activation tensors'},indent=2,allow_nan=False))
