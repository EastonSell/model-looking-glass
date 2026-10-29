"""Independently reconstruct the closed, incomplete year/label CPU attempt.

Uses retained bytes and operator records only; it does not execute a model or
infer a four-family study decision from the first family's completed actions.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('year_label_complete_reader',here/'year_label_saved_replay.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
need,same,sha,values,norm,score,round_f32=(base.need,base.same,base.sha,base.values,base.norm,base.score,base.round_f32)
ROW_BYTES,LOGITS_BYTES,WIDTH=base.ROW_BYTES,base.LOGITS_BYTES,base.WIDTH
CONDITIONS,ACTIONS=base.CONDITIONS,base.ACTIONS

def audit(args):
    base.numeric.binary64_probe()
    run=args.run;case_raw=args.cases.read_bytes();need(sha(case_raw)==base.CASE_SHA,'case hash')
    inventory=json.loads(case_raw);cases=inventory['cases'];digits=inventory['numeric_ids']
    need(len(cases)==48 and len({r['case_id'] for r in cases})==48,'fixed case identities')
    manifest_raw=(run/'manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    failure=json.loads((run/'failure.json').read_bytes())
    need(manifest['schema']=='distilgpt2-year-label-control-run-v1' and
         manifest['cases']==cases and manifest['cases_sha256']==base.CASE_SHA and
         manifest['protocol_sha256']==base.PROTOCOL_SHA and manifest['planned_forwards']==384,'manifest identity')
    need(failure['schema']=='distilgpt2-year-label-control-failure-v1' and failure['status']=='incomplete' and
         failure['manifest_sha256']==sha(manifest_raw) and failure['error_type']=='ValueError' and
         failure['phase']=='family_dose' and failure['case_id'] is None and failure['action'] is None,'failure identity')
    events=[json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
    need(events and set(events[0])=={'event','actual_runtime'} and events[0]['event']=='runtime_ready','runtime event')
    provenance=base.provenance(args,run,manifest,{'runtime':events[0]['actual_runtime']},cases,outcome='incomplete')
    need(len(manifest_raw)<=4*1024*1024 and (run/'failure.json').stat().st_size<=4*1024*1024 and
         (run/'attempt.jsonl').stat().st_size<=8*1024*1024,'metadata caps')
    offsets={r['case_id']:0 for r in cases};by_id={r['case_id']:r for r in cases}
    cache={r['case_id']:{} for r in cases};cursor=1;payload_bytes=0;completed=0
    def event(kind):
        nonlocal cursor
        need(cursor<len(events),'missing '+kind);item=events[cursor];cursor+=1
        need(item.get('event')==kind,'unexpected event '+str(cursor-1));return item
    def action(row,name,family=None):
        nonlocal payload_bytes,completed
        ident=row['case_id'];start=event('start')
        need(set(start)=={'event','case_id','action','intervention'} and
             start['case_id']==ident and start['action']==name,'start identity/order')
        planned=start['intervention'];hooked=name not in ('clean','noop','clean_after')
        need((planned is not None)==hooked,'start plan presence')
        plan_keys={'position','expected_before_hex','expected_after_hex','requested','achieved','changed_bits'}
        if hooked:need(type(planned) is dict and set(planned)==plan_keys,'durable start plan')
        recorded=event('raw_recorded');need(set(recorded)=={'event','case_id','action','raw'} and
             recorded['case_id']==ident and recorded['action']==name,'raw event identity')
        desc=recorded['raw'];need(set(desc)==({'logits'} if name in ('clean','clean_after') else
             {'logits','block_before','block_after'}),'raw payload inventory')
        payload={}
        with (run/(ident+'.f32')).open('rb') as stream:
            for part,d in sorted(desc.items(),key=lambda kv:kv[1]['offset']):
                length=LOGITS_BYTES if part=='logits' else len(row['input_ids'])*ROW_BYTES
                need(set(d)=={'offset','bytes','sha256'} and d['offset']==offsets[ident] and
                     d['bytes']==length,'raw offset/size')
                stream.seek(d['offset']);raw=stream.read(length)
                need(len(raw)==length and sha(raw)==d['sha256'],'raw segment hash')
                payload[part]=raw;offsets[ident]+=length;payload_bytes+=length
        done=event('complete');need(done['case_id']==ident and done['action']==name and
             done['raw']==desc and set(done)=={'event','case_id','action','raw','scores','intervention'},'complete identity')
        if hooked:
            info=done['intervention'];need(type(info) is dict and set(info)==plan_keys|{'observed_achieved','observed_changed_bits'} and
                 all(same(planned[k],info[k]) for k in plan_keys),'start/completion intervention')
        else:need(done['intervention'] is None,'unexpected completion plan')
        scores=score(payload['logits'],row,digits);need(same(done['scores'],scores),'raw score')
        saved=cache[ident]
        if name=='clean':saved['clean']=payload['logits'];saved['scores']=scores
        elif name=='noop':
            need(payload['logits']==saved['clean'] and payload['block_before']==payload['block_after'],'noop mismatch')
            values(payload['block_before']);saved['before']=payload['block_before']
        elif name=='clean_after':need(payload['logits']==saved['clean'],'clean-after mismatch')
        else:
            need(payload['block_before']==saved['before'],'pre-action block changed')
            pos=0 if name=='year_at_zero' else row['final_position'];start_offset=pos*ROW_BYTES
            before=saved['before'][start_offset:start_offset+ROW_BYTES]
            if name=='self':after=before;requested=0.
            else:
                field='label_donor_case_id' if name=='label_final' else 'year_donor_case_id'
                source=values(saved['before'][-ROW_BYTES:]);donor=values(cache[row[field]]['before'][-ROW_BYTES:])
                delta=[b-a for a,b in zip(source,donor)];distance=norm(delta);need(distance>0,'zero donor')
                d=[family['dose']/distance*x for x in delta]
                if name=='rotated_year_final':d=[d[(i-1)%WIDTH] for i in range(WIDTH)]
                after=struct.pack('<768I',*[round_f32(x+y) for x,y in zip(values(before),d)]);requested=family['dose']
            full=saved['before'][:start_offset]+after+saved['before'][start_offset+ROW_BYTES:]
            need(payload['block_after']==full,'landing or untouched block bits')
            achieved=norm(x-y for x,y in zip(values(after),values(before)))
            changed=sum(x!=y for x,y in zip(struct.iter_unpack('<I',before),struct.iter_unpack('<I',after)))
            need(info['position']==pos and info['expected_before_hex']==before.hex() and
                 info['expected_after_hex']==after.hex() and info['requested']==requested and
                 info['achieved']==achieved==info['observed_achieved'] and
                 info['changed_bits']==changed==info['observed_changed_bits'] and
                 abs(achieved-requested)<=1e-5*requested,'planned/realized intervention')
            need(name!='self' or payload['logits']==saved['clean'],'self logits')
            saved.setdefault('actions',{})[name]=scores['band_score']
        completed+=1
    for row in cases:action(row,'clean');action(row,'noop')
    for i in range(0,48,4):
        zeros=[cache[r['case_id']]['before'][:ROW_BYTES] for r in cases[i:i+4]]
        need(all(x==zeros[0] for x in zeros),'position-zero pairing')
    need(event('position_zero_check')=={'event':'position_zero_check','pairs_checked':36},'position-zero event')
    clean_families=[];differences=[]
    for i in range(4):
        group=cases[12*i:12*(i+1)];conditions={}
        for condition in CONDITIONS:
            labels={}
            for label in inventory['label_pairs'][i%2]:
                matched={r['year_role']:cache[r['case_id']]['scores'] for r in group
                         if r['condition']==condition and r['label']==label}
                need(set(matched)=={'low','high'},'clean role pair')
                diff=matched['low']['band_score']-matched['high']['band_score'];differences.append(diff)
                labels[label]={'low':matched['low'],'high':matched['high'],'low_minus_high':diff}
            conditions[condition]=labels
        clean_families.append({'family_index':i,'conditions':conditions})
    clean={'families':clean_families,'mean_band_score':math.fsum(cache[r['case_id']]['scores']['band_score'] for r in cases)/48,
           'mean_local_score':math.fsum(cache[r['case_id']]['scores']['local_score'] for r in cases)/48,
           'mean_low_minus_high':math.fsum(differences)/24}
    need(same(event('clean_summary'),{'event':'clean_summary','summary':clean}),'clean summary arithmetic')
    def family_info(i):
        group=cases[12*i:12*(i+1)];labels=inventory['label_pairs'][i%2]
        by={(r['condition'],r['label'],r['year_role']):r for r in group};norms={}
        def final(c,l,r):return values(cache[by[c,l,r]['case_id']]['before'][-ROW_BYTES:])
        for c in CONDITIONS:
            for l in labels:
                low,high=final(c,l,'low'),final(c,l,'high');norms[c+':year:'+l]=norm(b-a for a,b in zip(low,high))
            for role in ('low','high'):
                left,right=final(c,labels[0],role),final(c,labels[1],role)
                norms[c+':label:'+role]=norm(b-a for a,b in zip(left,right))
        dose=.5*min(norms.values());need(math.isfinite(dose) and dose>2**-20,'small common dose')
        return {'family_index':i,'dose':dose,'norms':norms,'fractions':{k:dose/v for k,v in norms.items()}}
    family0=family_info(0);need(same(event('family_dose'),{'event':'family_dose',**family0}),'family-zero dose')
    for row in cases[:12]:
        shift=row['rotation']
        for name in ACTIONS[shift:]+ACTIONS[:shift]:action(row,name,family0)
        action(row,'clean_after')
    failed=event('failed');need(failed=={'event':'failed','action':None,'case_id':None,'error_type':'ValueError',
        'phase':'family_dose','model_call_return_observed':False,'partial_payload_recorded':False,
        'partial_save_error':None,'payload_recorded':False,'payload_returned':False},'failure event')
    need(cursor==len(events) and completed==168 and payload_bytes==43725984,'incomplete schedule/counts')
    for key in ('submitted_forwards','returned_forwards','model_call_returns_observed','completed_records'):
        need(type(failure[key]) is int and failure[key]==168,'failure count '+key)
    need(failure['admitted_payload_bytes']==payload_bytes and failure['partial_files']==[] and
         failure['partial_save_error'] is None and failure['partial_payload_recorded'] is False and
         failure['payload_recorded'] is False and failure['payload_returned'] is False and
         failure['model_call_return_observed'] is False,'failure payload state')
    for row,entry in zip(cases,failure['raw_files']):
        ident=row['case_id'];raw=(run/(ident+'.f32')).read_bytes()
        need(entry=={'case_id':ident,'bytes':len(raw),'sha256':sha(raw)} and len(raw)==offsets[ident],
             'failure raw file inventory')
    need(len(failure['raw_files'])==48,'failure raw file count')
    group=cases[12:24];family1=family_info(1);by={(r['condition'],r['label'],r['year_role']):r for r in group}
    first_bad=None;plans_checked=0
    for c in CONDITIONS:
        for label in inventory['label_pairs'][1]:
            for role in ('low','high'):
                row=by[c,label,role];source=values(cache[row['case_id']]['before'][-ROW_BYTES:])
                year_donor=values(cache[row['year_donor_case_id']]['before'][-ROW_BYTES:])
                label_donor=values(cache[row['label_donor_case_id']]['before'][-ROW_BYTES:])
                year=[family1['dose']/family1['norms'][c+':year:'+label]*(b-a) for a,b in zip(source,year_donor)]
                lexical=[family1['dose']/family1['norms'][c+':label:'+role]*(b-a) for a,b in zip(source,label_donor)]
                rotated=[year[(i-1)%WIDTH] for i in range(WIDTH)]
                for action_name,position,displacement in [('year_final',row['final_position'],year),
                    ('label_final',row['final_position'],lexical),('year_at_zero',0,year),
                    ('rotated_year_final',row['final_position'],rotated)]:
                    raw=cache[row['case_id']]['before'][position*ROW_BYTES:(position+1)*ROW_BYTES];before=values(raw)
                    after=struct.pack('<768I',*[round_f32(x+y) for x,y in zip(before,displacement)])
                    achieved=norm(x-y for x,y in zip(values(after),before));q=family1['dose'];relative=abs(achieved-q)/q
                    plans_checked+=1
                    if abs(achieved-q)>1e-5*q:
                        first_bad={'family_index':1,'case_id':row['case_id'],'condition':c,'label':label,'year_role':role,
                                   'action':action_name,'requested':q,'achieved':achieved,'relative_error':relative,
                                   'tolerance':1e-5,'plans_checked_before_failure':plans_checked-1}
                        break
                if first_bad:break
            if first_bad:break
        if first_bad:break
    need(first_bad is not None and first_bad['action']=='year_at_zero','failed dose not reproduced')
    targets=[]
    for row in cases[:12]:
        saved=cache[row['case_id']];effects={a:saved['actions'][a]-saved['scores']['band_score'] for a in ACTIONS}
        year=-effects['year_final'] if row['year_role']=='low' else effects['year_final']
        checks={a:year>abs(effects[a])+.1 for a in ('label_final','year_at_zero','rotated_year_final')}
        targets.append({'case_id':row['case_id'],'effects':effects,'year_donorward':year,'checks':checks,'pass':all(checks.values())})
    base.numeric.binary64_probe()
    return {'status':'incomplete-replay-pass','completed_forwards':completed,'raw_bytes':payload_bytes,
            'clean_summary':clean,'completed_family0_dose':family0,
            'completed_family0_targets':targets,'completed_family0_passed_targets':sum(t['pass'] for t in targets),
            'failed_family1_dose':family1,'first_failed_plan_reconstructed':first_bad,'study_pattern_decision':None,
            'provenance_scope':provenance,'scope':'closed saved bytes; no model replay, execution attestation, four-family conclusion, or post-failure model/buffer-state verification'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True);p.add_argument('--pre-run-sha256',required=True)
    p.add_argument('--launch-observation-sha256',required=True);p.add_argument('--launch-main',required=True)
    p.add_argument('--launch-start-sha256',required=True);p.add_argument('--start-main',required=True)
    p.add_argument('--source-revision',required=True);p.add_argument('--source-tree',required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--current-exposure-version',required=True);p.add_argument('--current-exposure-sha256',required=True)
    p.add_argument('--stderr-projection',type=Path);p.add_argument('--stderr-projection-sha256')
    args=p.parse_args();args.synthetic=False
    print(json.dumps(audit(args),indent=2,allow_nan=False))
