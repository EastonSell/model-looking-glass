"""Independent saved-byte replay of year/label v2, with a synthetic test mode.

The direct-process receipt is an operator observation, not an independent
execution or descendant-lifetime attestation. No model is loaded here.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

here=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
n=load('numeric',here/'explicit_year_saved_replay.py')
oracle=load('oracle',here/'year_label_v2_numeric_oracle.py')
need=n.need;sha=lambda b:hashlib.sha256(b).hexdigest()
ACTIONS=('self','year_final','label_final','year_at_zero','rotated_year_final')
ROW=3072;LOGITS=50257*4

def audit(run,casefile,*,source_identity=None,prior=None):
    n.binary64_probe()
    source=casefile.read_bytes()
    need(sha(source)=='07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08','case pin')
    inventory=json.loads(source);cases=inventory['cases'][12:];digits=inventory['numeric_ids']
    manifest_raw=(run/'manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    result=json.loads((run/'result.json').read_bytes())
    if source_identity is None:
        need(manifest['source']=={'revision':'fabricated'},'synthetic-only fixture')
    else:
        need(manifest['source']==source_identity,'source identity')
    need(manifest['cases']==cases and len(cases)==36,'fixed cases')
    need(manifest['schema']=='distilgpt2-year-label-control-run-v2' and
         result['schema']=='distilgpt2-year-label-control-result-v2' and
         result['status']=='complete' and result['manifest_sha256']==sha(manifest_raw),'manifest/result')
    need(result['submitted_forwards']==result['model_call_returns_observed']==
         result['returned_forwards']==288,'forward counters')
    need(result['model_state_sha256_before']==result['model_state_sha256_after'] and
         result['model_buffers_sha256_before']==result['model_buffers_sha256_after'], 'model state')
    events=[json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
    need(events[0]=={'event':'runtime_ready','actual_runtime':result['runtime']},'runtime event')
    cache={r['case_id']:{} for r in cases};offsets={r['case_id']:0 for r in cases}
    cursor=1;total=0;records=[];certificates=0
    def event(kind):
        nonlocal cursor
        need(cursor<len(events) and events[cursor].get('event')==kind,'journal event: '+kind)
        value=events[cursor];cursor+=1;return value
    def forward(row,action,family=None):
        nonlocal total,certificates
        key=row['case_id'];start=event('start')
        need(start['case_id']==key and start['action']==action,'start order')
        plan=start['intervention'];hooked=action in ACTIONS
        need((plan is not None)==hooked,'plan presence')
        parts=event('raw_recorded')
        need(parts['case_id']==key and parts['action']==action,'raw identity')
        descriptors=parts['raw'];expected={'logits'} if action in ('clean','clean_after') else {'logits','block_before','block_after'}
        need(set(descriptors)==expected,'segment inventory')
        payload={}
        with (run/(key+'.f32')).open('rb') as f:
            for label,d in sorted(descriptors.items(),key=lambda x:x[1]['offset']):
                length=LOGITS if label=='logits' else len(row['input_ids'])*ROW
                need(d['offset']==offsets[key] and d['bytes']==length,'segment extent')
                f.seek(d['offset']);v=f.read(length)
                need(len(v)==length and sha(v)==d['sha256'],'segment hash')
                payload[label]=v;offsets[key]+=length;total+=length
        done=event('complete')
        need(done['case_id']==key and done['action']==action and done['raw']==descriptors,'completed identity')
        record=dict(done);del record['event'];records.append(record)
        scores=n.score(payload['logits'],row,digits)
        need(n.same(scores,record['scores']),'full-logit scoring')
        c=cache[key]
        if action=='clean':c['clean']=payload['logits'];c['scores']=scores;return
        if action=='noop':
            need(payload['logits']==c['clean'] and payload['block_before']==payload['block_after'],'no-op')
            c['before']=payload['block_before'];return
        if action=='clean_after':need(payload['logits']==c['clean'],'clean-after');return
        need(payload['block_before']==c['before'],'preaction baseline')
        pos=0 if action=='year_at_zero' else row['final_position']
        before=c['before'][pos*ROW:(pos+1)*ROW]
        if action=='self':after=before;displacement=None;dose=0.
        else:
            donor=row['label_donor_case_id'] if action=='label_final' else row['year_donor_case_id']
            target=n.values(c['before'][-ROW:]);source=n.values(cache[donor]['before'][-ROW:])
            difference=[b-a for a,b in zip(target,source)];distance=n.norm(difference)
            need(distance>0,'zero donor distance')
            factor=family['dose']/distance;displacement=[factor*x for x in difference]
            if action=='rotated_year_final':displacement=[displacement[(i-1)%768] for i in range(768)]
            after=struct.pack('<768I',*[n.round_f32(a+b) for a,b in zip(n.values(before),displacement)])
            dose=family['dose']
        expected_block=c['before'][:pos*ROW]+after+c['before'][(pos+1)*ROW:]
        need(payload['block_after']==expected_block,'landing or untouched coordinates')
        need(plan['position']==pos and plan['requested']==dose and
             plan['expected_before_hex']==before.hex() and plan['expected_after_hex']==after.hex(),'plan fields')
        if displacement is None:
            need(plan['displacement_f64_hex'] is None and plan['certificate'] is None and payload['logits']==c['clean'],'self')
        else:
            need(plan['displacement_f64_hex']==struct.pack('<768d',*displacement).hex(),'donor displacement bits')
            truth=oracle.certificate(dose,before,after,displacement)
            need(truth['pass'],'oracle fidelity')
            cert=plan['certificate']
            for field in ('D2','R2','E2'):
                ratio=cert[field]
                need(int(ratio['numerator'])==truth[field].numerator and int(ratio['denominator'])==truth[field].denominator,'rational '+field)
            q=Fraction.from_float(dose)
            limits={'D2_min':(q*Fraction(10**12-1,10**12))**2,
                    'D2_max':(q*Fraction(10**12+1,10**12))**2,
                    'R2_min':(q*Fraction(4999,5000))**2,
                    'R2_max':(q*Fraction(5001,5000))**2,'E2_max':(q/5000)**2}
            need(set(cert['limits'])==set(limits),'certificate limit inventory')
            for name,value in limits.items():
                pair=cert['limits'][name]
                need(int(pair['numerator'])==value.numerator and int(pair['denominator'])==value.denominator,'rational limit '+name)
            need(done['intervention']['observed_certificate']==cert,'same-forward certificate')
            certificates+=1
        need(all(n.same(v,done['intervention'][k]) for k,v in plan.items()),'start/completion plan')
        c.setdefault('actions',{})[action]=scores['band_score']
    for row in cases:forward(row,'clean');forward(row,'noop')
    for i in range(0,36,4):
        z=[cache[r['case_id']]['before'][:ROW] for r in cases[i:i+4]]
        need(all(v==z[0] for v in z),'position-zero equality')
    need(event('position_zero_check')=={'event':'position_zero_check','pairs_checked':27},'position-zero count')
    cross=event('cross_run_baseline_comparison')
    expected=[{'case_id':r['case_id'],
               'clean_logits_match_v1':cache[r['case_id']]['clean']==(bytes(LOGITS) if prior is None else prior[r['case_id']]['clean']),
               'noop_block_output_matches_v1':cache[r['case_id']]['before']==(bytes(len(r['input_ids'])*ROW) if prior is None else prior[r['case_id']]['before'])} for r in cases]
    need(cross['cases']==expected==result['cross_run_baseline_comparison'],'prior comparisons')
    clean=event('clean_summary');need(n.same(clean['summary'],result['clean_summary']),'clean summary event')
    for offset,index in enumerate((1,2,3)):
        group=cases[12*offset:12*(offset+1)];dose_event=event('family_dose')
        need(dose_event['family_index']==index,'family index')
        family=result['family_doses'][offset]
        need(n.same({'event':'family_dose',**family},dose_event),'family event')
        labels=list(dict.fromkeys(row['label'] for row in group));need(len(labels)==2,'labels')
        norms={}
        def site(condition,label,role):
            matches=[r for r in group if (r['condition'],r['label'],r['year_role'])==(condition,label,role)]
            need(len(matches)==1,'family edge mapping')
            return n.values(cache[matches[0]['case_id']]['before'][-ROW:])
        for condition in ('after','before','near'):
            for label in labels:
                low,high=site(condition,label,'low'),site(condition,label,'high')
                norms[f'{condition}:year:{label}']=n.norm([b-a for a,b in zip(low,high)])
            for role in ('low','high'):
                a,b=site(condition,labels[0],role),site(condition,labels[1],role)
                norms[f'{condition}:label:{role}']=n.norm([y-x for x,y in zip(a,b)])
        dose=min(norms.values())/2
        independent={'family_index':index,'dose':dose,'norms':norms,
                     'fractions':{key:dose/value for key,value in norms.items()}}
        need(n.same(independent,family),'reconstructed 12-edge family dose')
        for row in group:
            shift=row['rotation']
            for name in ACTIONS[shift:]+ACTIONS[:shift]:forward(row,name,family)
            forward(row,'clean_after')
    need(cursor==len(events)==871 and records==result['records'] and len(records)==288 and certificates==144,'journal tail')
    need(total==result['raw_payload_bytes']==77802624,'raw byte total')
    for key,end in offsets.items():need((run/(key+'.f32')).stat().st_size==end,'unclaimed raw tail')
    need(result['interventions_performed'] is True,'interventions not complete')
    families=result['intervention_summary']['families'];need(len(families)==3,'family summaries')
    independent_families=[]
    for offset,index in enumerate((1,2,3)):
        group=cases[12*offset:12*(offset+1)];family=families[offset]
        need(family['family_index']==index and len(family['targets'])==12,'summary mapping')
        passed=[];targets=[]
        for row,target in zip(group,family['targets']):
            key=row['case_id'];c=cache[key]
            effects={name:c['actions'][name]-c['scores']['band_score'] for name in ACTIONS}
            year=effects['year_final']*(-1 if row['year_role']=='low' else 1)
            controls=('label_final','year_at_zero','rotated_year_final')
            margins={name:year-abs(effects[name]) for name in controls}
            checks={name:year>abs(effects[name])+0.1 for name in controls}
            need(target['case_id']==key and n.same(target['effects'],effects) and
                 n.same(target['year_donorward'],year) and n.same(target['margins'],margins) and
                 target['checks']==checks and target['pass']==all(checks.values()),'target decision')
            passed.append(target['pass'])
            targets.append({'case_id':key,'condition':row['condition'],'label':row['label'],
                            'year_role':row['year_role'],'effects':effects,'year_donorward':year,
                            'margins':margins,'checks':checks,'pass':all(checks.values())})
        need(family['pass']==all(passed),'family decision')
        independent_families.append({'family_index':index,'pass':all(passed),'targets':targets})
    count=sum(family['pass'] for family in families)
    need(result['intervention_summary']['pattern_count']==count and
         result['intervention_summary']['pattern_supported']==(count==3),'three-family decision')
    n.binary64_probe()
    return {'status':'pass','forwards':len(records),'certificates':certificates,'raw_bytes':total,
            'family_pattern_count':count,'family_patterns':[family['pass'] for family in families],
            'families':independent_families,
            'cross_run_baseline_comparison':expected,
            'scope':'saved-byte consistency only; no model, execution or descendant attestation'}


def checked(path, expected):
    raw=path.read_bytes();need(sha(raw)==expected,'hash differs: '+path.name);return raw


def prior_data(attempt,cases):
    receipt=json.loads(checked(attempt/'supervisor.json','f3b418c7760ed23e2f3c3f90bd39b33538b2b9913ffad2dcf0dd7546742f55bf'))
    need(receipt['process_exit_code']==1 and receipt['failure_file_present'] is True,'prior stopped outcome')
    run=attempt/'run';files=receipt['run_sha256']
    all_ids=[r['case_id'] for r in json.loads((run/'manifest.json').read_bytes())['cases']]
    need(set(files)=={'manifest.json','attempt.jsonl','failure.json'}|{name+'.f32' for name in all_ids},'prior file set')
    for name,digest in files.items():checked(run/name,digest)
    events=[json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
    records={(e['case_id'],e['action']):e for e in events if e.get('event')=='complete' and e.get('action') in ('clean','noop')}
    result={}
    for row in cases:
        key=row['case_id'];chunks={}
        for action,name,target,size in (('clean','logits','clean',LOGITS),('noop','block_before','before',len(row['input_ids'])*ROW)):
            part=records[key,action]['raw'][name];need(part['bytes']==size,'prior segment length')
            with (run/(key+'.f32')).open('rb') as stream:stream.seek(part['offset']);raw=stream.read(size)
            need(len(raw)==size and sha(raw)==part['sha256'],'prior segment hash');chunks[target]=raw
        result[key]=chunks
    return result


def audit_attempt(args):
    attempt=args.attempt;run=attempt/'run';pre=json.loads(checked(attempt/'pre-run.json',args.pre_run_sha256))
    receipt=json.loads(checked(attempt/'supervisor.json',args.receipt_sha256))
    need(pre['schema']=='distilgpt2-year-label-control-launch-v2' and
         pre['source_revision']=='b256a6846ad2386d0c0c306003026bbd9b2893a8' and
         pre['source_tree']=='f23aa6799a145b9168677563da91d5ec78abc9b7','public pre-run identity')
    need(receipt['schema']=='distilgpt2-year-label-control-process-observation-v2' and
         receipt['pre_run_sha256']==args.pre_run_sha256 and
         receipt['source_revision']==pre['source_revision'] and receipt['source_tree']==pre['source_tree'] and
         receipt['process_exit_code']==0 and receipt['interrupted'] is False and
         receipt['failure_file_present'] is False and
         receipt['pycache_prefix_matches_pre_run'] is True and receipt['pycache_prefix_empty_after'] is True and
         receipt['pycache_prefix_sha256']==pre['intended_execution']['pycache_prefix_sha256'],'process receipt')
    checked(attempt/'stdout.txt',receipt['stdout_sha256']);stderr=checked(attempt/'stderr.txt',receipt['stderr_sha256'])
    need(len(stderr)==receipt['stderr_bytes'],'stderr size')
    files=receipt['run_sha256'];cases=json.loads(args.cases.read_bytes())['cases'][12:]
    expected={'manifest.json','attempt.jsonl','result.json'}|{r['case_id']+'.f32' for r in cases}
    need(set(files)==expected and {p.name for p in run.iterdir() if p.is_file()}==expected,'run file set')
    for name,digest in files.items():checked(run/name,digest)
    need(len(pre['source_sha256'])==51,'source inventory size')
    for name,digest in pre['source_sha256'].items():checked(args.source/name,digest)
    source={'revision':pre['source_revision'],'tree':pre['source_tree'],'sha256':pre['source_sha256']}
    manifest=json.loads((run/'manifest.json').read_bytes())
    need(manifest['empty_pycache_prefix_sha256']==receipt['pycache_prefix_sha256'] and
         manifest['prior_v1_receipt_sha256']==pre['prior_v1_receipt_sha256'] and
         manifest['cases_sha256']==pre['case_sha256'] and manifest['protocol_sha256']==pre['protocol_sha256'],
         'manifest pre-run binding')
    prior=prior_data(args.prior_attempt,cases)
    report=audit(run,args.cases,source_identity=source,prior=prior)
    report.update({'pre_run_sha256':args.pre_run_sha256,'receipt_sha256':args.receipt_sha256,
                   'stderr_bytes':len(stderr),'scope':'receipt and saved-byte consistency; operator-observed direct exit, not independent execution or descendant attestation; prior stderr is not checked'})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',type=Path,help='fabricated fixture only');mode.add_argument('--attempt',type=Path)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--prior-attempt',type=Path);p.add_argument('--pre-run-sha256');p.add_argument('--receipt-sha256')
    a=p.parse_args()
    if a.attempt is not None:
        need(all((a.source,a.prior_attempt,a.pre_run_sha256,a.receipt_sha256)),'production mode requires source, prior and external hashes')
        report=audit_attempt(a)
    else:report=audit(a.run,a.cases)
    print(json.dumps(report,sort_keys=True,allow_nan=False))
