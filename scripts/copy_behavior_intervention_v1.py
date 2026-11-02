"""Run the fixed twelve-family, eight-prompt copying intervention on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

if __name__ == "__main__" and (
        not sys.dont_write_bytecode or not sys.pycache_prefix
        or not Path(sys.pycache_prefix).is_absolute() or not Path(sys.pycache_prefix).is_dir()
        or any(Path(sys.pycache_prefix).iterdir()) or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or any(k in os.environ for k in ("MKL_CBWR", "ATEN_CPU_CAPABILITY"))
        or not os.environ.get("COPY_BEHAVIOR_INTERVENTION_BOOTSTRAP_SHA256") or "torch" in sys.modules):
    raise RuntimeError("use the reviewed CPU/offline bootstrap before project imports")

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network
GUARD_EVENTS=[]
SAFE_QUERIES={"cuda_is_available":0,"cuda_device_count":0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch,GUARD_EVENTS,SAFE_QUERIES)
from scripts import copying_head_output_dose_v1 as old
from scripts import copy_behavior_screen_v1 as screen_owner
from scripts.copy_behavior_intervention_bootstrap_v1 import verify_manifest
from looking_glass.copy_behavior_output_dose_v1 import HEADS, WEIGHT_BYTES, plan_family, inspect_actual, weights
from looking_glass.copy_behavior_effects_v1 import summarize_family, decision
from looking_glass.copy_head_intervention import assert_no_forward_hooks
from looking_glass.copy_head_factorial import binary64_rounding_probe
base, prior = old.base, old.prior
base.GUARD_EVENTS=GUARD_EVENTS

CASES='experiments/distilgpt2_copy_behavior_screen_cases_v1.json'
PROTOCOL='docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md'
AMENDMENT='docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_RESOURCE_AMENDMENT_V1.md'
MANIFEST='experiments/distilgpt2_copy_behavior_intervention_manifest_v1.json'
CASE_SHA='96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734'
PROTOCOL_SHA='78907b74bf3edb44a47c794508a9385da23b2bf2f26d07df4fefe74542b58875'
AMENDMENT_SHA='832d115a81dc58359f5ba5189222253ea002036fa46a37f99b1813b5f51a8f01'
SCREEN_HASHES={'screen.f32':'7d54f4f40b9bcdf56dd6a34823ceb4da61098c0f2c6ed84c90396272bc7328aa',
 'manifest.json':'4b920bd5f29b54c77914ec8fc402784ceb4b1219da386d900926727547442549',
 'attempt.jsonl':'f5a7857dec21cdc21d855ed4c07686c8b976162e41beae8256c6ab4f0877ade0',
 'result.json':'fcee5b994d9f8d70c0d9c35eae62da8827f5cb92008c9db709c8beb9a733aa99'}
AUDIT_SHA='584c0ccf1ad11c5ad4d857bfe73c3a5d70b138fd1c85b5bee6aac6152ba6b186'
BASELINE_COMMIT='c2b88d39e2826ec82253818d3778a59c0b4c7350'
RAW_LIMIT=154_720_512
WEIGHT_LIMIT=589_824
STAGING_BYTES=355_679_200
META_RESERVE=32*1024**2
FAILURE_RESERVE=1024**2
WORK_ROOT=Path('E:/Codex/model-looking-glass/work')
ORDER=('out_h2','out_h10','out_h7')
old.MAX_ACTION_BYTES=RAW_LIMIT


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_screen(directory,audit_path,cases):
    if not directory.is_absolute() or not audit_path.is_absolute():
        raise ValueError('baseline and audit paths must be absolute')
    records={}
    for name,expected in SCREEN_HASHES.items():
        raw=(directory/name).read_bytes()
        if sha(raw)!=expected: raise RuntimeError('baseline source digest differs: '+name)
        records[name]=raw
    audit_raw=audit_path.read_bytes()
    if sha(audit_raw)!=AUDIT_SHA: raise RuntimeError('baseline independent replay differs')
    result=json.loads(records['result.json']); audit=json.loads(audit_raw)
    if (result.get('status')!='complete' or result.get('counts')!={'submitted':192,'model_returns':192,'returned':192,'retained':38_597_376}
            or audit.get('cases_sha256')!=CASE_SHA or audit.get('raw_sha256')!=SCREEN_HASHES['screen.f32']
            or audit.get('row_count')!=192 or audit.get('synthetic_fixture') is not False):
        raise RuntimeError('baseline result or independent replay identity differs')
    families=result['summary']['families']; audited=audit['summary']['families']
    if (len(families)!=24 or len(audited)!=24 or
        [(f['group'],f['baseline_status']) for f in families] != [(f['group'],f['baseline_status']) for f in audited]):
        raise RuntimeError('baseline family support differs from independent replay')
    if len(records['screen.f32']) != 192*base.LOGITS_BYTES:
        raise RuntimeError('baseline matrix geometry differs')
    return {f['group']:f['baseline_status'] for f in families}, records['screen.f32']


def action_order(group,variant):
    shift=(8*(group-1)+variant)%3
    return ORDER[shift:]+ORDER[:shift]


def execute_family(batch,baseline_status,screen_raw,perform,plan_start,log,weight_raw):
    if (len(batch)!=8 or batch[0]['group'] not in range(2,25,2)
            or any(c['group']!=batch[0]['group'] or c['variant']!=v for v,c in enumerate(batch))):
        raise ValueError('held-out family identities or order differ')
    group=batch[0]['group']; caches=[{} for _ in batch]
    for case,cache in zip(batch,caches):
        perform(case,'clean',cache)
        index=(group-1)*8+case['variant']
        if cache['clean'] != screen_raw[index*base.LOGITS_BYTES:(index+1)*base.LOGITS_BYTES]:
            raise RuntimeError('intervention clean row differs from published baseline')
        perform(case,'noop',cache)
    plan_start()
    plan=plan_family(tuple(c['before'] for c in caches),weight_raw)
    old.base._append_event(log,{'event':'family_plan','group':group,'plan':old.output_description(plan)})
    for case,cache in zip(batch,caches):
        if plan['status']=='eligible':
            for kind in action_order(group,case['variant']):
                head=int(kind.split('h')[1])
                perform(case,kind,cache,head,plan['plans'][case['variant']][head]['delivered'])
        perform(case,'clean_after',cache)
    if plan['status']=='eligible':
        assessment=inspect_actual(tuple(c['noop_output'] for c in caches),
            tuple({h:c['arm_outputs']['out_h'+str(h)] for h in HEADS} for c in caches),plan)
    else:
        assessment={'status':'dose-ineligible','reason':plan['reason'],'q_out':plan['q_out']}
    comparisons=[{'h'+str(h):c.get('comparisons',{}).get('out_h'+str(h)) for h in HEADS} for c in caches]
    summary=summarize_family(group,baseline_status,assessment['status'],plan['q_out'],comparisons)
    family={'group':group,'baseline_status':baseline_status,'output_plan':old.output_description(plan),
            'output_assessment':assessment,'summary':summary,
            'clean_target_ranks':[c['clean_score']['target_rank'] for c in caches]}
    old.base._append_event(log,{'event':'family_complete',**family})
    return family


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir','stage-dir','output-dir','manifest','screen-dir','audit-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if any(not p.is_absolute() for p in (args.cache_dir,args.stage_dir,args.output_dir,args.manifest,args.screen_dir,args.audit_file)):
        raise ValueError('all paths must be absolute')
    digest=verify_manifest(args.manifest)
    if os.environ.get('COPY_BEHAVIOR_INTERVENTION_BOOTSTRAP_SHA256')!=digest:
        raise RuntimeError('bootstrap and manifest differ')
    manifest=json.loads(args.manifest.read_bytes())
    raw_cases=(ROOT/CASES).read_bytes()
    if (sha(raw_cases)!=CASE_SHA or sha((ROOT/PROTOCOL).read_bytes())!=PROTOCOL_SHA
            or sha((ROOT/AMENDMENT).read_bytes())!=AMENDMENT_SHA):
        raise RuntimeError('scientific rules or case bytes differ')
    data=json.loads(raw_cases); cases=data['cases']
    exposure=screen_owner.verify_reservation(cases,manifest)
    support,screen_raw=verify_screen(args.screen_dir,args.audit_file,cases)
    root=WORK_ROOT.resolve(strict=True)
    parent=args.output_dir.parent.resolve(strict=True); stage=args.stage_dir.resolve(strict=True)
    if (args.output_dir.exists() or not parent.is_relative_to(root) or not stage.is_relative_to(root)
            or parent.is_relative_to(ROOT) or stage.is_relative_to(ROOT)):
        raise ValueError('output and staging paths must be under E work outside checkout')
    from looking_glass.pretrained import ASSETS
    if sum(size for size,_ in ASSETS.values())!=STAGING_BYTES:
        raise RuntimeError('pinned stage size differs')
    needed=RAW_LIMIT+WEIGHT_LIMIT+STAGING_BYTES+META_RESERVE+FAILURE_RESERVE
    if shutil.disk_usage(parent).free<needed or shutil.disk_usage('C:/').free<1024**3:
        raise OSError('insufficient E or C headroom at phase start')
    torch.set_num_interop_threads(1)
    initial=base.numeric_policy(initialize=True); binary64_rounding_probe()
    torch.use_deterministic_algorithms(False); torch.set_float32_matmul_precision('highest'); torch.set_default_dtype(torch.float32)
    output=args.output_dir; output.mkdir()
    run_manifest={'schema':'distilgpt2-copy-behavior-intervention-run-v1','source_revision':manifest['source_revision'],
        'submitted_manifest_sha256':digest,'cases_sha256':CASE_SHA,'protocol_sha256':PROTOCOL_SHA,
        'amendment_sha256':AMENDMENT_SHA,'baseline_commit':BASELINE_COMMIT,'baseline_hashes':SCREEN_HASHES,
        'baseline_independent_audit_sha256':AUDIT_SHA,'exposure':exposure,'dependencies':base.dependency_identity(),
        'model_id':base.MODEL_ID,'model_revision':base.MODEL_REVISION,'weights_sha256':base.WEIGHTS_SHA256,
        'maximum_action_bytes':RAW_LIMIT,'maximum_weight_bytes':WEIGHT_LIMIT,'staging_bytes':STAGING_BYTES,
        'required_e_free':needed,'initial_numeric_probe':initial,'stage_directory':'E work',
        'scope':'twelve fixed held-out groups; supported primary and all-family descriptive strata'}
    run_sha=old.bounded_json(output/'manifest.json',run_manifest,old.MAX_MANIFEST_BYTES)
    counts={'submitted':0,'model_returns':0,'returned':0,'retained':0}
    activity={'phase':'model_load','case_id':None,'action':None,'observation':{},
              'marker':{'recorded':False,'partial':False,'partial_save_error':None}}
    completed=[]; families=[]
    held=[c for c in cases if c['split']=='held-out']
    try:
        bundle=base.load_distilgpt2(cache_dir=args.cache_dir,allow_download=False,stage_dir=stage)
        if bundle.metadata['weights_sha256']!=base.WEIGHTS_SHA256: raise RuntimeError('checkpoint differs')
        model=bundle.model; screen_owner.verify_tokenization(bundle.tokenizer,data)
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        state_before,buffers_before,runtime_before=base.state_hash(model),base.buffer_hash(model),prior.runtime(model)
        assert_no_forward_hooks(model)
        w=model.transformer.h[3].attn.c_proj.weight
        if tuple(w.shape)!=(768,768) or w.dtype!=torch.float32 or w.device.type!='cpu':
            raise RuntimeError('checkpoint projection geometry differs')
        weight_raw={h:base._raw(w[64*h:64*(h+1)].detach().contiguous(),WEIGHT_BYTES) for h in HEADS}
        matrix=weights(weight_raw); combined=b''.join(weight_raw[h] for h in HEADS)
        with (output/'projection-weights.f32').open('xb') as stream:
            if len(combined)!=WEIGHT_LIMIT or stream.write(combined)!=len(combined): raise OSError('weight slice write failed')
            stream.flush(); os.fsync(stream.fileno())
        with (output/'attempt.jsonl').open('xb') as stream:
            log=old.BoundedJournal(stream)
            base._append_event(log,{'event':'runtime_ready','actual_runtime':runtime_before,'weight_order':list(HEADS),
                'weight_sha256':sha(combined),'weight_head_sha256':{str(h):sha(weight_raw[h]) for h in HEADS}})
            def perform(case,kind,cache,head=None,delivery=None):
                return old.perform_action(case,kind,cache,head,delivery,model=model,runtime_before=runtime_before,
                    matrix=matrix,output=output,log=log,counts=counts,activity=activity,completed=completed)
            def plan_start():
                activity.update({'phase':'dose_plan','case_id':None,'action':None,'observation':{},
                                 'marker':{'recorded':False,'partial':False,'partial_save_error':None}})
            if len(held)!=96: raise RuntimeError('held-out case count differs')
            for i in range(12):
                batch=held[8*i:8*i+8]
                families.append(execute_family(batch,support[batch[0]['group']],screen_raw,perform,plan_start,log,weight_raw))
        activity['phase']='verification'
        eligible=sum(f['output_plan']['status']=='eligible' for f in families)
        forwards=96*3+eligible*24
        raw_bytes=96*(3*base.LOGITS_BYTES+3*base.WITNESS_BYTES)+eligible*24*(base.LOGITS_BYTES+3*base.WITNESS_BYTES)
        if (len(families)!=12 or len(completed)!=forwards or counts!={'submitted':forwards,'model_returns':forwards,
                'returned':forwards,'retained':raw_bytes} or base.state_hash(model)!=state_before
                or base.buffer_hash(model)!=buffers_before or prior.runtime(model)!=runtime_before
                or verify_manifest(args.manifest)!=digest or GUARD_EVENTS or torch.cuda.is_initialized()):
            raise RuntimeError('final counts, state, runtime or source differ')
        assert_no_forward_hooks(model)
        outcome=decision([f['summary'] for f in families])
        result={'schema':'distilgpt2-copy-behavior-intervention-result-v1','status':'complete','manifest_sha256':run_sha,
            'checkpoint':bundle.metadata,'weight_sha256':sha(combined),'weight_head_sha256':{str(h):sha(weight_raw[h]) for h in HEADS},
            'state_sha256_before':state_before,'state_sha256_after':base.state_hash(model),
            'buffers_sha256_before':buffers_before,'buffers_sha256_after':base.buffer_hash(model),
            'runtime':runtime_before,'counts':counts,'records':completed,'families':families,'decision':outcome}
        old.bounded_json(output/'result.json',result,old.MAX_RESULT_BYTES)
    except BaseException as error:
        try:
            old.bounded_json(output/'failure.json',{'schema':'distilgpt2-copy-behavior-intervention-failure-v1',
                'status':'incomplete','manifest_sha256':run_sha,'phase':activity['phase'],'case_id':activity['case_id'],
                'action':activity['action'],'marker':activity['marker'],
                'model_call_return_observed':activity['observation'].get('model_call_return_observed',False),
                'helper_return_observed':activity['observation'].get('helper_return_observed',False),
                'helper_return_unbound_uncertain':activity['observation'].get('helper_return_unbound_uncertain',False),
                'counts':counts,'completed_records':len(completed),'raw_files':base.raw_file_inventory(output,held),
                'error_type':type(error).__name__,'traceback_frames':[{'file':Path(f.filename).name,'line':f.lineno,
                    'function':f.name} for f in traceback.extract_tb(error.__traceback__)]},old.MAX_FAILURE_BYTES)
        finally:
            raise
    print(json.dumps({'status':'complete','forwards':forwards,'decision':outcome},allow_nan=False))


if __name__ == '__main__':
    main()