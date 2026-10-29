"""Replay an operator-derived selected-logit/TV projection without a model.

Selected-logit effects, schedule, gates and summary arithmetic are checked.
TV and probability values are retained reports, not reconstructible from two
logits. This subset does not authenticate its origin, full vectors, geometry,
landing, dose, execution, or model behavior beyond the recorded projections.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
SCALAR = HERE / 'review_bert_cue_context_scores.py'
SCALAR_SHA = '50837eaa286f33b05160e6dac51fa98cb82decdd860bbfeea7bb623a864f3771'
if hashlib.sha256(SCALAR.read_bytes()).hexdigest() != SCALAR_SHA:
    raise ValueError('score reader dependency changed')
spec = importlib.util.spec_from_file_location('fixed_context_score_reader', SCALAR)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def pair(event):
    data = base64.b64decode(event['mask_logits_pair_f32le'],validate=True)
    if len(data) != 8:
        raise ValueError('invalid selected-logit pair')
    a,b = struct.unpack('<ff',data)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('nonfinite selected logits')
    return data,a,b


def probability(value):
    return type(value) in (float,int) and math.isfinite(value) and 0 <= value <= 1


def replay(manifest,result,stream):
    score.numeric_mode()
    cases = manifest['pairs']
    if (result['status'] != 'complete' or len(result['pairs']) != 3 or
            tuple(x['id'] for x in cases) != score.PAIR_IDS or
            tuple(x['split'] for x in cases) != ('discovery','evaluation','evaluation')):
        raise ValueError('fixed pair order differs')
    count = 0
    def take(kind,pair_id,**identity):
        nonlocal count
        line = stream.readline(2*1024*1024+1)
        if not line or len(line)>2*1024*1024 or not line.endswith(b'\n'):
            raise ValueError('missing or oversized projection event')
        event = score.strict(line)
        if (event.get('kind') != kind or event.get('pair_id') != pair_id or
                any(not score.same(event.get(k),v) for k,v in identity.items())):
            raise ValueError('projection event identity or order differs')
        count += 1
        return event
    summaries=[]
    for case,saved in zip(cases,result['pairs'],strict=True):
        ident=case['id']
        if saved['id'] != ident or saved['split'] != case['split'] or len(case['contexts']) != 2:
            raise ValueError('case identity differs')
        contexts=case['contexts']
        ids={(i,c):contexts[i]['rows'][c]['input_ids'] for i in range(2) for c in score.CUES}
        clean={}; margins={}; mass={}
        for i in range(2):
            for cue in score.CUES:
                if not score.same(contexts[i]['rows'][cue]['attention_mask'],[1]*len(ids[i,cue])):
                    raise ValueError('attention mask differs')
                event=take('clean',ident,context=i,cue=cue,input_ids=ids[i,cue])
                raw,any_logit,some_logit=pair(event)
                b=saved['baselines'][i*4+score.CUES.index(cue)]
                margin=math.fsum((any_logit,-some_logit))
                if (not score.same(b['context'],i) or b['cue']!=cue or
                        not score.same(b['any_logit'],any_logit) or not score.same(b['some_logit'],some_logit) or
                        not score.same(b['any_minus_some'],margin) or
                        not all(probability(b[k]) for k in ('any_probability','some_probability','target_pair_probability')) or
                        not score.same(b['target_pair_probability'],math.fsum((b['any_probability'],b['some_probability'])))):
                    raise ValueError('projected baseline score or reported mass differs')
                clean[i,cue],margins[i,cue],mass[i,cue]=raw,margin,b['target_pair_probability']
        screens=[]
        for i in range(2):
            gaps=[{'negative':n,'affirmative':a,'gap':math.fsum((margins[i,n],-margins[i,a]))}
                  for n in score.CUES[:2] for a in score.CUES[2:]]
            screens.append({'eligible':all(x['gap']>.5 for x in gaps),'gaps':gaps})
        baseline_ok=all(x['eligible'] for x in screens)
        event=take('baseline',ident)
        if (not score.same(event['contexts'],screens) or event['eligible'] is not baseline_ok or
                not score.same(saved['baseline_screens'],screens) or saved['baseline_eligible'] is not baseline_ok):
            raise ValueError('baseline gate differs')
        event=take('radius',ident)
        distances=event['distances']
        identities=[(site,i,c,j,d) for site in score.SITES for i in range(2) for c in score.CUES
                    for j,d in score.donor_order(i,c)]
        if (len(distances)!=224 or not score.same(
                [[x['site'],x['target_context'],x['target_cue'],x['donor_context'],x['donor_cue']] for x in distances],
                [list(x) for x in identities]) or
                any(type(x['distance']) not in (int,float) or not math.isfinite(x['distance']) or x['distance']<0 for x in distances)):
            raise ValueError('recorded distance labels differ')
        minimum=min(x['distance'] for x in distances)
        q=min(.5,minimum); dose_ok=q>=.05
        if (not score.same(event['q'],q) or not score.same(event['minimum_distance'],minimum) or
                event['dose_eligible'] is not dose_ok or
                not score.same(event['binding_edges'],[x for x in distances if x['distance']==minimum]) or
                not score.same(saved['q'],q) or not score.same(saved['minimum_distance'],minimum) or
                saved['dose_eligible'] is not dose_ok or not score.same(saved['distances'],distances)):
            raise ValueError('recorded radius or gate differs')
        if not dose_ok:
            take('dose_ineligible',ident,q=q)
        effects={}; rows=[]
        if dose_ok:
            for site in score.SITES:
                for i in range(2):
                    for cue in score.CUES:
                        event=take('self',ident,site=site,target_context=i,target_cue=cue,input_ids=ids[i,cue])
                        if pair(event)[0]!=clean[i,cue]:
                            raise ValueError('self selected-logit bits differ')
                for i in range(2):
                    for cue in score.CUES:
                        for j,donor in score.donor_order(i,cue):
                            event=take('transfer',ident,site=site,target_context=i,target_cue=cue,
                                       donor_context=j,donor_cue=donor,input_ids=ids[i,cue],requested_radius=q)
                            raw,a,b=pair(event)
                            data={'delta':math.fsum((math.fsum((a,-b)),-margins[i,cue])),
                                  **{k:event[k] for k in ('full_tv','excluded_tv','clean_pair_probability','patched_pair_probability')}}
                            if (not all(probability(data[k]) for k in ('full_tv','excluded_tv','clean_pair_probability','patched_pair_probability')) or
                                    not score.same(data['clean_pair_probability'],mass[i,cue])):
                                raise ValueError('reported transfer probability or TV differs')
                            if site=='cue5' and (raw!=clean[i,cue] or data['full_tv']!=0 or data['excluded_tv']!=0 or
                                                 not score.same(data['patched_pair_probability'],mass[i,cue])):
                                raise ValueError('final-cue projected scores or reported metrics differ')
                            effects[site,i,cue,j,donor]=data
                            rows.append({'site':site,'target_context':i,'target_cue':cue,
                                         'donor_context':j,'donor_cue':donor,**data})
        if not score.same(saved['effects'],rows):
            raise ValueError('selected effects or retained reported metrics differ')
        for i in range(2):
            for cue in score.CUES:
                event=take('repeat',ident,context=i,cue=cue,input_ids=ids[i,cue])
                if pair(event)[0]!=clean[i,cue]:
                    raise ValueError('repeat selected-logit bits differ')
        tv_rows=[]
        if dose_ok:
            for site in score.SITES:
                for metric in ('full_tv','excluded_tv'):
                    tv_rows.append({'site':site,'metric':metric,
                        'same_cue_mean':math.fsum(effects[site,i,c,1-i,c][metric] for i in range(2) for c in score.CUES)/8,
                        'same_class_mean':math.fsum(effects[site,i,c,1-i,score.buddy(c)][metric] for i in range(2) for c in score.CUES)/8})
        if not score.same(saved['tv_summary'],tv_rows):
            raise ValueError('retained TV summary arithmetic differs')
        comp=score.decision(effects,baseline_ok,dose_ok)
        if not score.same(saved['comparison'],comp) or not score.same(saved['forwards'],272 if dose_ok else 16):
            raise ValueError('pair decision differs')
        summaries.append({'id':ident,'split':case['split'],'forwards':saved['forwards'],
                          'baseline_eligible':baseline_ok,'dose_eligible':dose_ok,'q':q,
                          'minimum_recorded_distance':minimum,'comparison':comp,'tv_summary':tv_rows})
    if stream.readline(1):
        raise ValueError('extra projection event')
    evaluation=score.aggregate([x['comparison'] for x in summaries])
    if not score.same(result['evaluation'],evaluation) or not score.same(result['forwards'],sum(x['forwards'] for x in summaries)):
        raise ValueError('aggregate outcome or count differs')
    score.numeric_mode()
    return {'status':'passed','events':count,'forwards':result['forwards'],'pairs':summaries,'evaluation':evaluation}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path); p.add_argument('subset',type=Path); p.add_argument('result',type=Path)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--expected-subset-sha256',required=True)
    p.add_argument('--expected-result-sha256',required=True)
    p.add_argument('--expected-original-journal-sha256',required=True)
    p.add_argument('--expected-source'); p.add_argument('--expected-started-sha256')
    p.add_argument('--synthetic',action='store_true')
    a=p.parse_args()
    paths=(a.manifest,a.subset,a.result); before=tuple(score.sha(x) for x in paths)
    if before!=(a.expected_manifest_sha256,a.expected_subset_sha256,a.expected_result_sha256):
        raise ValueError('projection input hashes differ')
    manifest,result=score.strict(a.manifest.read_bytes()),score.strict(a.result.read_bytes())
    if not a.synthetic:
        if (before[0]!=score.MANIFEST_SHA or not re.fullmatch('[0-9a-f]{40}',a.expected_source or '') or
                not re.fullmatch('[0-9a-f]{64}',a.expected_started_sha256 or '') or
                result.get('source_commit')!=a.expected_source or result.get('events_sha256')!=a.expected_original_journal_sha256 or
                result.get('source_sha256',{}).get(score.MANIFEST_PATH)!=before[0]):
            raise ValueError('source/result identity differs')
    with gzip.open(a.subset,'rb') as stream:
        header=score.strict(stream.readline(2*1024*1024+1))
        if (header.get('schema')!='bert-cue-context-projection-v1' or header.get('synthetic') is not a.synthetic or
                header.get('manifest_sha256')!=before[0] or header.get('result_sha256')!=before[2] or
                header.get('original_journal_sha256')!=a.expected_original_journal_sha256 or
                header.get('source_commit')!=(None if a.synthetic else a.expected_source) or
                header.get('started_sha256')!=(None if a.synthetic else a.expected_started_sha256) or
                header.get('full_score_reader_sha256')!=SCALAR_SHA):
            raise ValueError('projection header differs')
        report=replay(manifest,result,stream)
    if tuple(score.sha(x) for x in paths)!=before or hashlib.sha256(SCALAR.read_bytes()).hexdigest()!=SCALAR_SHA:
        raise ValueError('projection inputs or score dependency changed')
    report.update({'synthetic':a.synthetic,'manifest_sha256':before[0],'subset_sha256':before[1],
                   'result_sha256':before[2],'original_journal_sha256':a.expected_original_journal_sha256,
                   'source_commit':a.expected_source if not a.synthetic else None,'score_reader_sha256':SCALAR_SHA,
                   'scope':'operator-derived selected-logit scores, gate/decision and reported TV summary arithmetic; '
                           'not original-journal authentication, full TV reconstruction, geometry, landing, dose or execution'})
    print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
