"""Independently replay saved cue-context scores, TVs and fixed decisions.

No model is loaded. This checks recorded logits, schedule, recorded distance
labels, target distributions and arithmetic; it does not verify activation
geometry, full-state landing, execution, or pretraining exposure.
"""
import argparse
import base64
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct

CUES=('never','rarely','always','often')
SITES=('cue2','object2','mask2','cue5')
PAIR_IDS=('discovery-01','evaluation-01','evaluation-02')
MANIFEST_SHA='4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7'
MANIFEST_PATH='experiments/bert_cue_context_transfer_cases_v1.json'

def same(left,right):
    if type(left) is not type(right): return False
    if type(left) is dict:
        return left.keys()==right.keys() and all(same(left[k],right[k]) for k in left)
    if type(left) is list:
        return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right

def numeric_mode():
    tiny64=struct.unpack('<d',bytes.fromhex('0100000000000000'))[0]
    neg64=struct.unpack('<d',bytes.fromhex('0100000000000080'))[0]
    tiny32=struct.unpack('<f',bytes.fromhex('01000000'))[0]
    neg32=struct.unpack('<f',bytes.fromhex('01000080'))[0]
    if (struct.pack('<d',tiny64*2)!=bytes.fromhex('0200000000000000') or
        struct.pack('<d',neg64*2)!=bytes.fromhex('0200000000000080') or
        struct.pack('<d',tiny32)!=struct.pack('<d',math.ldexp(1.,-149)) or
        struct.pack('<d',neg32)!=struct.pack('<d',-math.ldexp(1.,-149)) or
        struct.pack('<f',1.+2.**-24)!=bytes.fromhex('0000803f') or
        struct.pack('<f',1.+3.*2.**-24)!=bytes.fromhex('0200803f')):
        raise RuntimeError('numeric mode must preserve subnormals and use nearest-even f32 rounding')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def strict(data):
    def unique(items):
        d={}
        for key,value in items:
            if key in d: raise ValueError('duplicate JSON key')
            d[key]=value
        return d
    return json.loads(data,object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def logits(event,vocab):
    raw=base64.b64decode(event['mask_logits_f32le'],validate=True)
    if len(raw)!=4*vocab: raise ValueError('vocabulary size differs')
    values=tuple(x[0] for x in struct.iter_unpack('<f',raw))
    if any(not math.isfinite(x) for x in values): raise ValueError('nonfinite logits')
    return raw,values

def distribution(values,excluded=()):
    kept=[v for i,v in enumerate(values) if i not in excluded]
    peak=max(kept)
    weights=[math.exp(v-peak) for v in kept]
    denominator=math.fsum(weights)
    return [x/denominator for x in weights],peak,denominator

def baseline(values,any_id,some_id):
    p,peak,denominator=distribution(values)
    logden=math.log(denominator)
    return {'any_minus_some':math.fsum((values[any_id],-values[some_id])),
            'any_probability':p[any_id], 'some_probability':p[some_id],
            'any_logit':values[any_id], 'some_logit':values[some_id],
            'any_log_probability':(values[any_id]-peak)-logden,
            'some_log_probability':(values[some_id]-peak)-logden,
            'target_pair_probability':math.fsum((p[any_id],p[some_id])),
            'top_five':[{'token_id':i,'logit':values[i]} for i in
                        sorted(range(len(values)),key=lambda i:(-values[i],i))[:5]]}

def tv(left,right):
    return math.fsum(abs(a-b) for a,b in zip(left,right))/2.0

def donor_order(context,cue):
    return [(1-context,c) for c in CUES]+[(context,c) for c in CUES if c!=cue]

def buddy(cue):
    index=CUES.index(cue)
    return CUES[(index//2)*2+1-index%2]

def decision(effects,baseline_ok,dose_ok):
    if not dose_ok:
        return {'status':'dose_ineligible','informative':False,'recipients':[],
                'same_cue_mean':None,'same_class_mean':None,'half_difference':None}
    rows=[]
    for context in range(2):
        for cue in CUES:
            rows.append({'target_context':context,'target_cue':cue,
                         'same_cue_delta':effects['cue2',context,cue,1-context,cue]['delta'],
                         'same_class_delta':effects['cue2',context,cue,1-context,buddy(cue)]['delta']})
    a=math.fsum(abs(x['same_cue_delta']) for x in rows)/8
    b=math.fsum(abs(x['same_class_delta']) for x in rows)/8
    h=math.fsum((a,-b/2))
    status=('baseline_ineligible' if not baseline_ok else 'uninformative' if b<.01
            else 'at_least_half' if h>=0 else 'below_half')
    return {'status':status,'informative':baseline_ok and b>=.01,'recipients':rows,
            'same_cue_mean':a,'same_class_mean':b,'half_difference':h}

def aggregate(pairs):
    selected=pairs[1:]
    eligible=[x for x in selected if x['informative']]
    if len(eligible)!=2:
        return {'status':'unevaluable','informative_pairs':len(eligible),'recipients':8*len(eligible),
                'same_cue_mean':None,'same_class_mean':None,'half_difference':None}
    rows=[r for p in selected for r in p['recipients']]
    if len(rows)!=16: raise ValueError('evaluation recipient count differs')
    a=math.fsum(abs(x['same_cue_delta']) for x in rows)/16
    b=math.fsum(abs(x['same_class_delta']) for x in rows)/16
    h=math.fsum((a,-b/2))
    return {'status':'at_least_half' if h>=0 else 'below_half','informative_pairs':2,
            'recipients':16,'same_cue_mean':a,'same_class_mean':b,'half_difference':h}

def replay(manifest,result,stream,vocab):
    numeric_mode()
    cases=manifest['pairs']
    if (result['status']!='complete' or len(result['pairs'])!=3 or
        tuple(p['id'] for p in cases)!=PAIR_IDS or
        tuple(p['split'] for p in cases)!=('discovery','evaluation','evaluation')):
        raise ValueError('fixed pair order differs')
    any_id,some_id=(manifest['scored_tokens'][x] for x in ('any','some'))
    if type(any_id) is not int or type(some_id) is not int or any_id==some_id or min(any_id,some_id)<0 or max(any_id,some_id)>=vocab:
        raise ValueError('score token identities differ')
    count=0
    def take(kind,pair,**identity):
        nonlocal count
        line=stream.readline(2*1024*1024+1)
        if not line or len(line)>2*1024*1024 or not line.endswith(b'\n'):
            raise ValueError('missing or oversized event')
        event=strict(line)
        if event.get('kind')!=kind or event.get('pair_id')!=pair or any(not same(event.get(k),v) for k,v in identity.items()):
            raise ValueError('event identity or order differs')
        count+=1
        return event
    summaries=[]
    for case,saved in zip(cases,result['pairs']):
        ident=case['id']
        if saved['id']!=ident or saved['split']!=case['split'] or len(case['contexts'])!=2:
            raise ValueError('case identity differs')
        contexts=case['contexts']
        ids={(i,c):contexts[i]['rows'][c]['input_ids'] for i in range(2) for c in CUES}
        if any(not same(contexts[i]['rows'][c]['attention_mask'],[1]*len(ids[i,c])) for i in range(2) for c in CUES):
            raise ValueError('attention mask differs')
        clean={}; clean_prob={}; clean_excluded={}; baseline_rows=[]; margins={}
        for i in range(2):
            for cue in CUES:
                event=take('clean',ident,context=i,cue=cue,input_ids=ids[i,cue])
                raw,values=logits(event,vocab)
                clean[i,cue]=raw
                clean_prob[i,cue]=distribution(values)[0]
                clean_excluded[i,cue]=distribution(values,(any_id,some_id))[0]
                b=baseline(values,any_id,some_id)
                baseline_rows.append({'context':i,'cue':cue,**b})
                margins[i,cue]=b['any_minus_some']
        if not same(saved['baselines'],baseline_rows): raise ValueError('baseline scores or probabilities differ')
        screens=[]
        for i in range(2):
            gaps=[{'negative':n,'affirmative':a,'gap':math.fsum((margins[i,n],-margins[i,a]))}
                  for n in CUES[:2] for a in CUES[2:]]
            screens.append({'eligible':all(x['gap']>.5 for x in gaps),'gaps':gaps})
        baseline_ok=all(x['eligible'] for x in screens)
        event=take('baseline',ident)
        if (not same(event['contexts'],screens) or event['eligible'] is not baseline_ok or
            not same(saved['baseline_screens'],screens) or saved['baseline_eligible'] is not baseline_ok):
            raise ValueError('baseline eligibility differs')
        event=take('radius',ident)
        distances=event['distances']
        identities=[(site,i,c,j,d) for site in SITES for i in range(2) for c in CUES
                    for j,d in donor_order(i,c)]
        if (len(distances)!=224 or not same([[x['site'],x['target_context'],x['target_cue'],x['donor_context'],x['donor_cue']]
                                    for x in distances],[list(x) for x in identities]) or
            any(type(x['distance']) not in (int,float) or not math.isfinite(x['distance']) or x['distance']<0 for x in distances)):
            raise ValueError('recorded distance schedule differs')
        minimum=min(x['distance'] for x in distances)
        q=min(.5,minimum); dose_ok=q>=.05
        if (not same(event['q'],q) or not same(event['minimum_distance'],minimum) or event['dose_eligible'] is not dose_ok or
            not same(event['binding_edges'],[x for x in distances if x['distance']==minimum]) or
            not same(saved['q'],q) or not same(saved['minimum_distance'],minimum) or saved['dose_eligible'] is not dose_ok or
            not same(saved['distances'],distances)):
            raise ValueError('recorded q or dose eligibility differs')
        if not dose_ok: take('dose_ineligible',ident,q=q)
        effects={}; effect_rows=[]
        if dose_ok:
            for site in SITES:
                for i in range(2):
                    for cue in CUES:
                        event=take('self',ident,site=site,target_context=i,target_cue=cue,input_ids=ids[i,cue])
                        raw,_=logits(event,vocab)
                        if raw!=clean[i,cue]: raise ValueError('self full logits differ')
                for i in range(2):
                    for cue in CUES:
                        for j,donor in donor_order(i,cue):
                            event=take('transfer',ident,site=site,target_context=i,target_cue=cue,
                                       donor_context=j,donor_cue=donor,input_ids=ids[i,cue],requested_radius=q)
                            raw,values=logits(event,vocab)
                            if site=='cue5' and raw!=clean[i,cue]: raise ValueError('final cue logits differ')
                            full=distribution(values)[0]; excluded=distribution(values,(any_id,some_id))[0]
                            data={'delta':math.fsum((math.fsum((values[any_id],-values[some_id])),-margins[i,cue])),
                                  'full_tv':tv(clean_prob[i,cue],full),
                                  'excluded_tv':tv(clean_excluded[i,cue],excluded),
                                  'clean_pair_probability':baseline_rows[i*4+CUES.index(cue)]['target_pair_probability'],
                                  'patched_pair_probability':math.fsum((full[any_id],full[some_id]))}
                            effects[site,i,cue,j,donor]=data
                            effect_rows.append({'site':site,'target_context':i,'target_cue':cue,
                                                'donor_context':j,'donor_cue':donor,**data})
        if not same(saved['effects'],effect_rows): raise ValueError('saved transfer scores or TV differ')
        for i in range(2):
            for cue in CUES:
                event=take('repeat',ident,context=i,cue=cue,input_ids=ids[i,cue])
                raw,_=logits(event,vocab)
                if raw!=clean[i,cue]: raise ValueError('repeat full logits differ')
        tv_rows=[]
        if dose_ok:
            for site in SITES:
                for metric in ('full_tv','excluded_tv'):
                    tv_rows.append({'site':site,'metric':metric,
                      'same_cue_mean':math.fsum(effects[site,i,c,1-i,c][metric] for i in range(2) for c in CUES)/8,
                      'same_class_mean':math.fsum(effects[site,i,c,1-i,buddy(c)][metric] for i in range(2) for c in CUES)/8})
        if not same(saved['tv_summary'],tv_rows): raise ValueError('saved TV averages differ')
        comp=decision(effects,baseline_ok,dose_ok)
        if not same(saved['comparison'],comp) or not same(saved['forwards'],272 if dose_ok else 16):
            raise ValueError('pair decision or forward count differs')
        summaries.append({'id':ident,'split':case['split'],'forwards':saved['forwards'],
                          'baseline_eligible':baseline_ok,'dose_eligible':dose_ok,'q':q,
                          'minimum_recorded_distance':minimum,'comparison':comp,'tv_summary':tv_rows})
    if stream.readline(1): raise ValueError('extra journal events')
    outcome=aggregate([p['comparison'] for p in summaries])
    if not same(result['evaluation'],outcome) or not same(result['forwards'],sum(p['forwards'] for p in summaries)):
        raise ValueError('aggregate outcome or forward count differs')
    numeric_mode()
    return {'status':'passed','events':count,'forwards':result['forwards'],'pairs':summaries,'evaluation':outcome,
            'scope':'full mask logits, scores, TV and recorded distance labels; not activation geometry, '
                    'landing, execution or pretraining attestation'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path); parser.add_argument('events',type=Path); parser.add_argument('result',type=Path)
    parser.add_argument('--started',type=Path); parser.add_argument('--expected-source')
    parser.add_argument('--expected-started-sha256'); parser.add_argument('--expected-result-sha256')
    parser.add_argument('--synthetic',action='store_true'); parser.add_argument('--vocab',type=int,default=30522)
    a=parser.parse_args(); paths=(a.manifest,a.events,a.result); before=tuple(sha(p) for p in paths)
    manifest,result=strict(a.manifest.read_bytes()),strict(a.result.read_bytes())
    if not a.synthetic:
        if (a.vocab!=30522 or before[0]!=MANIFEST_SHA or not re.fullmatch('[0-9a-f]{40}',a.expected_source or '') or
            not re.fullmatch('[0-9a-f]{64}',a.expected_result_sha256 or '') or before[2]!=a.expected_result_sha256 or
            not re.fullmatch('[0-9a-f]{64}',a.expected_started_sha256 or '') or not a.started or
            sha(a.started)!=a.expected_started_sha256 or result.get('source_commit')!=a.expected_source or
            result.get('events_sha256')!=before[1] or
            result.get('source_sha256',{}).get(MANIFEST_PATH)!=MANIFEST_SHA or a.events.suffix!='.gz'):
            raise ValueError('operator-supplied source or saved-byte identity differs')
        started=strict(a.started.read_bytes())
        if any(not same(started.get(k),result.get(k)) for k in ('source_commit','source_sha256','assets','versions','seed','cpu_environment','tokenizer')):
            raise ValueError('started/result metadata differs')
    opener=gzip.open if a.events.suffix=='.gz' else open
    with opener(a.events,'rb') as stream:
        report=replay(manifest,result,stream,a.vocab)
    if tuple(sha(p) for p in paths)!=before or (not a.synthetic and sha(a.started)!=a.expected_started_sha256):
        raise ValueError('an input changed during replay')
    report.update({'synthetic':a.synthetic,'manifest_sha256':before[0],'events_sha256':before[1],
                   'result_sha256':before[2],'source_commit':result.get('source_commit'),
                   'identity_scope':'operator-supplied expected source and start/result consistency; '
                                    'not independent source/asset pin or execution attestation'})
    print(json.dumps(report,sort_keys=True,allow_nan=False))

if __name__=='__main__': main()
