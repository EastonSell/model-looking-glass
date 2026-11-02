"""Build fabricated 12-family intervention records; never load a model."""
import argparse,hashlib,json,math,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import copy_behavior_intervention_v1 as owner
from looking_glass.copy_head_factorial import _values
from looking_glass.copy_behavior_output_dose_v1 import HEADS,project,weights


def pack(values): return struct.pack('<%df'%len(values),*values)
def sha(raw): return hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--underfill',action='store_true')
    parser.add_argument('--one-family',action='store_true')
    parser.add_argument('--actual-mismatch',action='store_true')
    args=parser.parse_args()
    if not args.cases.is_absolute() or not args.output.is_absolute() or args.output.exists():
        raise ValueError('absolute new paths required')
    raw_cases=args.cases.read_bytes()
    if sha(raw_cases)!=owner.CASE_SHA: raise ValueError('wrong cases')
    cases=json.loads(raw_cases)['cases']; all_held=[c for c in cases if c['split']=='held-out']
    held=all_held[:8] if args.one_family else all_held
    by_input={tuple(c['token_ids']):c for c in held}
    if len(cases)!=192 or len(by_input)!=(8 if args.one_family else 96): raise ValueError('wrong case counts')
    w=[0.0]*(64*768); w[0],w[1],w[768]=1.0,-2.0,17.0
    one=pack(w); weight_raw={h:one for h in HEADS}; combined=b''.join(weight_raw[h] for h in HEADS)
    matrix=weights(weight_raw)
    def baseline(case):
        row=bytearray(owner.base.LOGITS_BYTES)
        struct.pack_into('<f',row,4*case['target_id'],1.0)
        return bytes(row)
    screen_raw=b''.join(baseline(c) for c in cases)
    support={g:('supported' if g in (2,10,12,16,20) else 'unsupported') for g in range(2,25,2)}
    args.output.mkdir(); (args.output/'projection-weights.f32').write_bytes(combined)
    manifest={'schema':'distilgpt2-copy-behavior-intervention-run-v1','synthetic_fixture':True,
              'cases_sha256':owner.CASE_SHA,'baseline_raw_sha256':sha(screen_raw)}
    msha=owner.base.durable_json(args.output/'manifest.json',manifest)
    counts={'submitted':0,'model_returns':0,'returned':0,'retained':0}; activity={}; completed=[]; families=[]
    runtime={'fabricated':True}
    def fake_forward(_model,ids,action,*,expected_before=None,deliveries=None,observation=None):
        case=by_input[tuple(ids.tolist()[0])]
        source=[0.0]*(11*768)
        for h,value in ((2,1.0),(10,2.0),(7,4.0)):
            index=2 if args.underfill and case['group']<=10 and h==2 else 0
            source[10*768+64*h+index]=value
        before=pack(source); after=before; effect=0.0
        if action.startswith('h'):
            head=int(action[1:])
            if expected_before!=before or set(deliveries)!={head}: raise ValueError('wrong fake delivery')
            a=bytearray(before); start=(10*768+64*head)*4; a[start:start+256]=deliveries[head]; after=bytes(a)
            effect={2:-1.0,10:-0.5,7:-0.1}[head]*(1.0 if case['variant']%4==0 else 0.1)
        logits=bytearray(owner.base.LOGITS_BYTES)
        struct.pack_into('<f',logits,case['target_id']*4,1.0+effect)
        payload={'logits':bytes(logits)}
        if action not in ('clean','clean_after'):
            a=_values(after)
            contributions=[project(a[10*768+64*h:10*768+64*(h+1)],matrix[h]) for h in HEADS]
            final=[math.fsum(row[j] for row in contributions) for j in range(768)]
            if args.actual_mismatch and case['group']==2 and case['variant']==0 and action=='h2':
                final[0]+=0.1
            payload.update({'cproj_before':before,'cproj_after':after,'cproj_output':pack([0.0]*(10*768)+final)})
        observation.update({'model_call_return_observed':True,'payload':payload})
        return payload
    owner.prior.runtime=lambda _model:runtime
    owner.prior.run_forward=fake_forward
    with (args.output/'attempt.jsonl').open('xb') as stream:
        log=owner.old.BoundedJournal(stream)
        owner.base._append_event(log,{'event':'runtime_ready','actual_runtime':runtime,'weight_order':list(HEADS),
            'weight_sha256':sha(combined),'weight_head_sha256':{str(h):sha(weight_raw[h]) for h in HEADS}})
        def perform(case,kind,cache,head=None,delivery=None):
            return owner.old.perform_action(case,kind,cache,head,delivery,model=None,runtime_before=runtime,
                matrix=matrix,output=args.output,log=log,counts=counts,activity=activity,completed=completed)
        for i in range(1 if args.one_family else 12):
            batch=held[8*i:8*i+8]
            families.append(owner.execute_family(batch,support[batch[0]['group']],screen_raw,perform,lambda:None,log,weight_raw))
    dummy='0'*64
    result={'schema':'distilgpt2-copy-behavior-intervention-result-v1','status':'complete','synthetic_fixture':True,
        'manifest_sha256':msha,'checkpoint':{'synthetic_fixture':True},'weight_sha256':sha(combined),
        'weight_head_sha256':{str(h):sha(weight_raw[h]) for h in HEADS},'state_sha256_before':dummy,
        'state_sha256_after':dummy,'buffers_sha256_before':dummy,'buffers_sha256_after':dummy,
        'runtime':runtime,'counts':counts,'records':completed,'families':families,
        'partial_fixture':args.one_family,
        'decision':None if args.one_family else owner.decision([f['summary'] for f in families])}
    owner.old.bounded_json(args.output/'result.json',result,owner.old.MAX_RESULT_BYTES)
    print(json.dumps({'synthetic_fixture':True,'counts':counts,'decision':result['decision'],
        'journal_sha256':sha((args.output/'attempt.jsonl').read_bytes()),
        'result_sha256':sha((args.output/'result.json').read_bytes()),'weight_sha256':sha(combined)}))


if __name__=='__main__': main()