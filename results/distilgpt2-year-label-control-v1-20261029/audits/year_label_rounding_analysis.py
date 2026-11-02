"""Post-hoc, model-free delivery-error bound analysis from saved no-op bytes."""
import argparse
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path
import struct

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved_numeric',here/'year_label_saved_replay.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
need,values,norm,round_f32=base.need,base.values,base.norm,base.round_f32
base.numeric.binary64_probe()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True);parser.add_argument('--cases',type=Path,required=True)
args=parser.parse_args();run=args.run;case_raw=args.cases.read_bytes()
need(hashlib.sha256(case_raw).hexdigest()==base.CASE_SHA,'case hash')
inventory=json.loads(case_raw);cases=inventory['cases'];noops={}
for line in (run/'attempt.jsonl').read_bytes().splitlines():
    event=json.loads(line)
    if event.get('event')=='complete' and event.get('action')=='noop':
        desc=event['raw']['block_before'];path=run/(event['case_id']+'.f32')
        with path.open('rb') as f:f.seek(desc['offset']);raw=f.read(desc['bytes'])
        need(len(raw)==desc['bytes'] and hashlib.sha256(raw).hexdigest()==desc['sha256'],'noop bytes')
        need(event['case_id'] not in noops,'duplicate noop case')
        noops[event['case_id']]=raw
need(set(noops)=={r['case_id'] for r in cases} and len(cases)==48,'noop inventory')
rows=[]
for i in (1,2,3):
    group=cases[12*i:12*i+12];labels=inventory['label_pairs'][i%2]
    by={(r['condition'],r['label'],r['year_role']):r for r in group};norms={}
    def final(c,l,r):return values(noops[by[c,l,r]['case_id']][-base.ROW_BYTES:])
    for c in base.CONDITIONS:
        for label in labels:
            lo,hi=final(c,label,'low'),final(c,label,'high')
            norms[c+':year:'+label]=norm(b-a for a,b in zip(lo,hi))
        for role in ('low','high'):
            a,b=final(c,labels[0],role),final(c,labels[1],role)
            norms[c+':label:'+role]=norm(y-x for x,y in zip(a,b))
    q=.5*min(norms.values());need(q>2**-20,'small dose');qf=Fraction.from_float(q)
    for row in group:
        source=values(noops[row['case_id']][-base.ROW_BYTES:])
        year_donor=values(noops[row['year_donor_case_id']][-base.ROW_BYTES:])
        label_donor=values(noops[row['label_donor_case_id']][-base.ROW_BYTES:])
        y=[q/norms[row['condition']+':year:'+row['label']]*(b-a) for a,b in zip(source,year_donor)]
        l=[q/norms[row['condition']+':label:'+row['year_role']]*(b-a) for a,b in zip(source,label_donor)]
        rot=[y[(j-1)%base.WIDTH] for j in range(base.WIDTH)]
        for action,position,d in [('year_final',row['final_position'],y),('label_final',row['final_position'],l),
                                  ('year_at_zero',0,y),('rotated_year_final',row['final_position'],rot)]:
            raw=noops[row['case_id']][position*base.ROW_BYTES:(position+1)*base.ROW_BYTES];b=values(raw)
            delivered=values(struct.pack('<768I',*[round_f32(x+t) for x,t in zip(b,d)]))
            f=Fraction.from_float
            d2=sum((f(v)*f(v) for v in d),Fraction())
            e2=sum(((f(z)-f(x)-f(v))**2 for x,z,v in zip(b,delivered,d)),Fraction())
            r2=sum(((f(z)-f(x))**2 for x,z in zip(b,delivered)),Fraction())
            bounds=[]
            for x,v,z in zip(b,d,delivered):
                y=x+v
                exponent=(-150 if abs(y)<2**-126 else math.frexp(abs(y))[1]-25)
                half_ulp=Fraction(1<<exponent,1) if exponent>=0 else Fraction(1,1<<(-exponent))
                need(abs(f(z)-f(y))<=half_ulp,'float32 rounding exceeds half ulp')
                bounds.append(abs(f(y)-f(x)-f(v))+half_ulp)
            bound2=sum((x*x for x in bounds),Fraction())
            need(e2<=bound2,'per-vector mathematical bound failed')
            norm_relative=abs(math.sqrt(float(r2))-q)/q
            error_relative=math.sqrt(float(e2))/q
            bound_relative=math.sqrt(float(bound2))/q
            direction_relative=abs(math.sqrt(float(d2))-q)/q
            rows.append({'family_index':i,'case_id':row['case_id'],'action':action,'q':q,
                'realized':math.sqrt(float(r2)),'norm_relative_error':norm_relative,
                'vector_relative_error':error_relative,'direction_norm_relative_error':direction_relative,
                'rounding_bound_relative':bound_relative,
                'passes_exact_vector_1e_4':e2*10000**2<=qf*qf,
                'passes_exact_bound_2e_4':bound2*5000**2<=qf*qf,
                'passes_exact_direction_1e_12':(qf*Fraction(10**12-1,10**12))**2<=d2<=
                    (qf*Fraction(10**12+1,10**12))**2})
summary={'schema':'year-label-rounding-posthoc-v1','scope':'saved no-op values only; no model or fresh outcome',
         'rows':rows,'plans':len(rows),'max_vector_relative_error':max(x['vector_relative_error'] for x in rows),
         'max_norm_relative_error':max(x['norm_relative_error'] for x in rows),
         'max_rounding_bound_relative':max(x['rounding_bound_relative'] for x in rows),
         'max_direction_norm_relative_error':max(x['direction_norm_relative_error'] for x in rows),
         'old_norm_tolerance_failures':sum(x['norm_relative_error']>1e-5 for x in rows),
         'proposed_vector_bound_failures':sum(not x['passes_exact_vector_1e_4'] for x in rows),
         'proposed_rounding_bound_failures':sum(not x['passes_exact_bound_2e_4'] for x in rows),
         'proposed_direction_bound_failures':sum(not x['passes_exact_direction_1e_12'] for x in rows)}
base.numeric.binary64_probe()
print(json.dumps(summary,indent=2,allow_nan=False))
