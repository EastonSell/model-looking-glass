"""Replay a stopped fixed year/label run from saved bytes; no model imports.

The score/bit helpers are adapted from the author's separate pre-run fabricated
reader. This is a saved-evidence consistency check, not an execution attestation.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

if sys.byteorder!='little' or not __debug__:
    raise RuntimeError('requires little-endian CPU and assertions enabled')

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--cases',type=Path,required=True)
parser.add_argument('--failure-sha256',required=True)
parser.add_argument('--journal-sha256',required=True)
args=parser.parse_args()

DIGITS=(405,486,2999,3070,3023,2713,3312,2998,2919,2931,940,1157,1065,1485,1415,1314,1433,1558,
1507,1129,1238,2481,1828,1954,1731,1495,2075,1983,2078,1959,1270,3132,2624,2091,2682,2327,
2623,2718,2548,2670,1821,3901,3682,3559,2598,2231,3510,2857,2780,2920,1120,4349,4309,4310,
4051,2816,3980,3553,3365,3270,1899,5333,5237,5066,2414,2996,2791,3134,3104,3388,2154,4869,
4761,4790,4524,2425,4304,3324,3695,3720,1795,6659,6469,5999,5705,5332,4521,5774,3459,4531,
3829,6420,5892,6052,5824,3865,4846,5607,4089,2079)
NUMERIC=set(DIGITS)
ACTIONS=('self','year_final','label_final','year_at_zero','rotated_year_final')
CONDITIONS=('after','before','near')
LABELS=(('alpha','beta'),('red','blue'))
ROW=768*4

def digest(x): return hashlib.sha256(x).hexdigest()
def floats(raw):
    assert len(raw)%4==0
    result=array.array('f'); assert result.itemsize==4; result.frombytes(raw)
    assert all(math.isfinite(x) for x in result)
    return result
def norm(values): return math.sqrt(math.fsum(x*x for x in values))
def band(values,left,right):
    assert len(left)==len(right)==3 and len(set(left+right))==6
    a,b=[values[i] for i in left],[values[i] for i in right]
    u,v=max(a),max(b)
    return ((u-v)+math.log(math.fsum(math.exp(x-u) for x in a)))-math.log(math.fsum(math.exp(x-v) for x in b))
def score(raw,row):
    values=floats(raw); assert len(values)==50257
    m=max(values); weights=[math.exp(float(x)-m) for x in values]; total=math.fsum(weights)
    return {'band_score':band(values,row['middle_ids'],row['above_ids']),
            'local_score':band(values,row['local_above_ids'],row['local_below_ids']),
            'aligned_01_99_full_mass':math.fsum(weights[i] for i in DIGITS[1:])/total,
            'zero_00_next_token_mass':weights[DIGITS[0]]/total,
            'other_vocabulary_mass':math.fsum(w for i,w in enumerate(weights) if i not in NUMERIC)/total,
            'top_token_id':values.index(m)}

cases_raw=args.cases.read_bytes()
assert digest(cases_raw)=='07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'
cases=json.loads(cases_raw)['cases']; assert len(cases)==48
manifest_raw=(args.run/'manifest.json').read_bytes(); manifest=json.loads(manifest_raw)
failure_raw=(args.run/'failure.json').read_bytes(); assert digest(failure_raw)==args.failure_sha256
failure=json.loads(failure_raw)
journal_raw=(args.run/'attempt.jsonl').read_bytes(); assert digest(journal_raw)==args.journal_sha256
events=[json.loads(line) for line in journal_raw.splitlines()]
assert manifest['cases']==cases and failure['manifest_sha256']==digest(manifest_raw)
assert failure['status']=='incomplete' and failure['phase']=='family_dose' and failure['error_type']=='ValueError'
assert all(failure[x]==168 for x in ('submitted_forwards','model_call_returns_observed','returned_forwards','completed_records'))
assert failure['partial_files']==[] and failure['partial_payload_recorded'] is False
expected=[(r['case_id'],a) for r in cases for a in ('clean','noop')]
for row in cases[:12]:
    shift=(row['family_index']+2*row['condition_index']+row['label_index']+3*row['anchor_index'])%5
    assert shift==row['rotation']
    expected.extend((row['case_id'],a) for a in ACTIONS[shift:]+ACTIONS[:shift]+('clean_after',))
assert len(expected)==168
assert [(e['event'],e['case_id'],e['action']) for e in events if e['event'] in ('start','raw_recorded','complete')]==[
    (stage,case,action) for case,action in expected for stage in ('start','raw_recorded','complete')]
completed=[{k:v for k,v in e.items() if k!='event'} for e in events if e['event']=='complete']
assert [(r['case_id'],r['action']) for r in completed]==expected
assert [e['raw'] for e in events if e['event']=='raw_recorded']==[r['raw'] for r in completed]
assert events[-1]['event']=='failed' and events[-1]['phase']=='family_dose'
byid={row['case_id']:row for row in cases}; seen={}; offsets={key:0 for key in byid}
for record in completed:
    case,action=byid[record['case_id']],record['action']; payload={}
    with (args.run/(case['case_id']+'.f32')).open('rb') as stream:
        for key,part in sorted(record['raw'].items(),key=lambda item:item[1]['offset']):
            assert part['offset']==offsets[case['case_id']]
            stream.seek(part['offset']); raw=stream.read(part['bytes'])
            assert len(raw)==part['bytes'] and digest(raw)==part['sha256']
            offsets[case['case_id']]+=len(raw); payload[key]=raw
    assert set(payload)==({'logits'} if action in ('clean','clean_after') else {'logits','block_before','block_after'})
    actual=score(payload['logits'],case); assert actual==record['scores']
    seen[case['case_id'],action]=(record,payload,actual)
assert sum(offsets.values())==failure['admitted_payload_bytes']==43725984
assert {p.name for p in args.run.glob('*.f32')}=={key+'.f32' for key in byid}
inventory={item['case_id']:item for item in failure['raw_files']}; assert set(inventory)==set(byid)
for key,size in offsets.items():
    data=(args.run/(key+'.f32')).read_bytes()
    assert len(data)==size==inventory[key]['bytes'] and digest(data)==inventory[key]['sha256']

clean={row['case_id']:seen[row['case_id'],'clean'][2] for row in cases}
for row in cases:
    key=row['case_id']; a=seen[key,'noop'][1]
    assert seen[key,'clean'][1]['logits']==a['logits']
    assert a['block_before']==a['block_after'] and len(a['block_before'])==len(row['input_ids'])*ROW
for row in cases[:12]:
    key=row['case_id']; assert all(seen[key,arm][1]['logits']==seen[key,'clean'][1]['logits'] for arm in ('self','clean_after'))

clean_families=[]; contrasts=[]
for fi in range(4):
    conditions={}
    for condition in CONDITIONS:
        labels={}
        for label in LABELS[fi%2]:
            pair=[row for row in cases[12*fi:12*fi+12] if row['condition']==condition and row['label']==label]
            assert [row['year_role'] for row in pair]==['low','high']
            low,high=(clean[row['case_id']] for row in pair); gap=low['band_score']-high['band_score']
            contrasts.append(gap); labels[label]={'low':low,'high':high,'low_minus_high':gap}
        conditions[condition]=labels
    clean_families.append({'family_index':fi,'conditions':conditions})
clean_summary={'families':clean_families,
               'mean_band_score':math.fsum(clean[row['case_id']]['band_score'] for row in cases)/48,
               'mean_local_score':math.fsum(clean[row['case_id']]['local_score'] for row in cases)/48,
               'mean_low_minus_high':math.fsum(contrasts)/24}
assert [e['summary'] for e in events if e['event']=='clean_summary']==[clean_summary]
assert [e['pairs_checked'] for e in events if e['event']=='position_zero_check']==[36]

family=cases[:12]; labels=LABELS[0]
def find(c,l,r):
    x=[row for row in family if (row['condition'],row['label'],row['year_role'])==(c,l,r)]
    assert len(x)==1
    return x[0]
raw={(c,l,r):seen[find(c,l,r)['case_id'],'noop'][1]['block_before']
     for c in CONDITIONS for l in labels for r in ('low','high')}
vec={key:floats(value[-ROW:]) for key,value in raw.items()}; norms={}
for c in CONDITIONS:
    assert len({raw[c,l,r][:ROW] for l in labels for r in ('low','high')})==1
    for l in labels: norms[f'{c}:year:{l}']=norm(b-a for a,b in zip(vec[c,l,'low'],vec[c,l,'high']))
    for r in ('low','high'): norms[f'{c}:label:{r}']=norm(b-a for a,b in zip(vec[c,labels[0],r],vec[c,labels[1],r]))
q=min(norms.values())/2; assert q>2**-20
family_dose={'family_index':0,'dose':q,'norms':norms,'fractions':{k:q/v for k,v in norms.items()}}
assert [{k:v for k,v in e.items() if k!='event'} for e in events if e['event']=='family_dose']==[family_dose]
targets=[]
for case in family:
    c,l,r=case['condition'],case['label'],case['year_role']; li=labels.index(l); ri=('low','high').index(r)
    source=vec[c,l,r]; year=vec[c,l,('low','high')[1-ri]]; lexical=vec[c,labels[1-li],r]
    yd=[q/norms[f'{c}:year:{l}']*(b-a) for a,b in zip(source,year)]
    ld=[q/norms[f'{c}:label:{r}']*(b-a) for a,b in zip(source,lexical)]
    rot=[yd[(i-1)%768] for i in range(768)]
    before=raw[c,l,r]; end=len(case['input_ids'])-1
    choices={'self':(end,before[-ROW:]),
             'year_final':(end,struct.pack('<768f',*(a+b for a,b in zip(source,yd)))),
             'label_final':(end,struct.pack('<768f',*(a+b for a,b in zip(source,ld)))),
             'year_at_zero':(0,struct.pack('<768f',*(a+b for a,b in zip(floats(before[:ROW]),yd)))),
             'rotated_year_final':(end,struct.pack('<768f',*(a+b for a,b in zip(source,rot))))}
    for action,(pos,after_site) in choices.items():
        record,payload,_=seen[case['case_id'],action]; start=pos*ROW; before_site=before[start:start+ROW]
        assert payload['block_before']==before and payload['block_after']==before[:start]+after_site+before[start+ROW:]
        detail=record['intervention']; achieved=norm(a-b for a,b in zip(floats(after_site),floats(before_site)))
        changed=sum(a!=b for a,b in zip(struct.iter_unpack('<I',before_site),struct.iter_unpack('<I',after_site)))
        target=0.0 if action=='self' else q
        assert detail['position']==pos and detail['expected_before_hex']==before_site.hex() and detail['expected_after_hex']==after_site.hex()
        assert detail['requested']==target and detail['achieved']==detail['observed_achieved']==achieved
        assert detail['changed_bits']==detail['observed_changed_bits']==changed and abs(achieved-target)<=1e-5*q
    key=case['case_id']; base=clean[key]['band_score']
    effects={a:seen[key,a][2]['band_score']-base for a in ACTIONS}
    year=-effects['year_final'] if r=='low' else effects['year_final']
    controls=('label_final','year_at_zero','rotated_year_final')
    checks={a:year>abs(effects[a])+0.1 for a in controls}
    targets.append({'case_id':key,'condition':c,'label':l,'year_role':r,'effects':effects,
                    'year_donorward':year,'margins':{a:year-abs(effects[a]) for a in controls},
                    'checks':checks,'pass':all(checks.values())})

report={'status':'partial_replay_passed','case_sha256':digest(cases_raw),'manifest_sha256':digest(manifest_raw),
        'failure_sha256':digest(failure_raw),'journal_sha256':digest(journal_raw),'completed_forwards':168,
        'raw_bytes':sum(offsets.values()),'baseline_cases':48,'first_family_dose':family_dose,
        'clean_summary':clean_summary,'first_family_targets':targets,
        'first_family_targets_passing':sum(t['pass'] for t in targets),
        'study_criterion':'incomplete: three families did not run',
        'scope':'Saved-bit consistency and first-family checks; no execution or outcome attestation.'}
print(json.dumps(report,sort_keys=True,allow_nan=False))
