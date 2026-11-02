"""One fixed offline phase; no protected inputs are read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

CASE_SHA='811a1f0251ff21ddaa1869ee96c80dfa47beb87a6b6c2f1e3b6505728fee1825'
PROTOCOL_SHA='6d87a579eb27f214db3e8769232b5940d263da59490480765118174ef99b5d70'
RESERVATION_SHA='f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9'
STUDY=('run_phase.py','distractor_core.py','distractor_executor.py','distractor_journal.py',
       'distractor_analysis.py','distractor_saved.py','distractor_admission.py',
       'direct_launch.py','distractor_report.py')
DEPENDENCIES=('looking_glass/__init__.py','looking_glass/pretrained.py',
 'looking_glass/intervention_plan.py','looking_glass/interventions.py','looking_glass/model_ownership.py',
 'looking_glass/transformer_trace.py','looking_glass/trace_schema.py','looking_glass/probes.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def path_hash(path):return hashlib.sha256(os.path.normcase(str(path.resolve(strict=True))).encode()).hexdigest()
def checked(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('fixed file digest differs')
    return raw

def validate_prefix(expected):
    if not sys.dont_write_bytecode or not sys.pycache_prefix:raise ValueError('requires -B and fresh pycache prefix')
    prefix=Path(sys.pycache_prefix)
    if (not prefix.is_absolute() or prefix.is_symlink() or not prefix.is_dir()
        or any(prefix.iterdir()) or path_hash(prefix)!=expected):
        raise ValueError('bytecode prefix is not the fresh pinned directory')
    return prefix

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('discovery','held-out'),required=True)
    for name in ('study-root','source-root','cases','protocol','inventory','cache-dir','stage-dir','pre-run','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pre-run-sha256',required=True)
    parser.add_argument('--discovery-dir',type=Path)
    parser.add_argument('--discovery-receipt',type=Path)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    pre=json.loads(checked(args.pre_run,args.pre_run_sha256))
    if (pre.get('schema')!='public-distractor-repair-pre-run-v1' or pre.get('case_sha256')!=CASE_SHA
        or pre.get('protocol_sha256')!=PROTOCOL_SHA or pre.get('reservation_sha256')!=RESERVATION_SHA):
        raise ValueError('pre-run fixed pins differ')
    prefix=validate_prefix(pre['pycache_prefix_sha256'][args.phase])
    study=args.study_root.resolve(strict=True);source=args.source_root.resolve(strict=True)
    if Path(__file__).resolve()!=study/'run_phase.py':raise ValueError('runner path differs')
    if set(pre['study_sha256'])!=set(STUDY) or set(pre['dependency_sha256'])!=set(DEPENDENCIES):
        raise ValueError('source inventory differs')
    for name in STUDY:
        if sha(study/name)!=pre['study_sha256'][name]:raise ValueError('study source digest differs')
    for name in DEPENDENCIES:
        if sha(source/name)!=pre['dependency_sha256'][name]:raise ValueError('dependency source digest differs')
    cases=json.loads(checked(args.cases,CASE_SHA))['cases']
    checked(args.protocol,PROTOCOL_SHA)
    if (len(cases)!=16 or [c['case_id'] for c in cases]!=[f'distractor-repair-{i:02d}' for i in range(1,17)]
        or any(c['split']!=('discovery' if i<8 else 'held-out') or c['candidate_token_ids']!=[318,389]
               or c['token_ids']!=c['scored_prefix_token_ids']+[13]
               or c['scored_position']!=len(c['token_ids'])-2 or c['future_position']!=len(c['token_ids'])-1
               for i,c in enumerate(cases))):raise ValueError('fixed cases or positions differ')
    stage=args.stage_dir
    if (args.output_dir.exists() or not args.output_dir.parent.is_dir() or not stage.is_absolute()
        or stage.is_symlink() or not stage.is_dir() or any(stage.iterdir())
        or path_hash(stage)!=pre['stage_dir_sha256']):raise ValueError('stage or output differs')
    if any(os.path.splitdrive(str(x.resolve()))[0].upper()!='E:' for x in (prefix,stage,args.output_dir.parent)):
        raise ValueError('output, stage, and bytecode prefix must use E:')
    if (shutil.disk_usage(args.output_dir.parent).free<2*1024**3 or shutil.disk_usage(stage).free<2*1024**3
        or any(shutil.disk_usage(x).free<256*1024**2 for x in (source,study,args.cache_dir))):
        raise ValueError('insufficient source, cache or output reserve')
    sys.path.insert(0,str(source));sys.path.insert(0,str(study))
    import distractor_admission as admission
    if Path(admission.__file__).resolve()!=study/'distractor_admission.py':raise ValueError('admission import differs')
    admission_report=admission.verify(args.inventory,cases)
    if args.phase=='held-out':
        if not all((args.discovery_dir,args.discovery_receipt,args.freeze,args.freeze_sha256)):
            raise ValueError('held-out needs the published discovery freeze and receipt')
        frozen=json.loads(checked(args.freeze,args.freeze_sha256))
        receipt_hash=sha(args.discovery_receipt)
        import distractor_saved as saved
        records,evidence=saved.load_phase(args.discovery_dir,cases[:8])
        if (frozen.get('schema')!='public-distractor-discovery-freeze-v1'
            or frozen.get('case_sha256')!=CASE_SHA or frozen.get('pre_run_sha256')!=args.pre_run_sha256
            or frozen.get('reservation_sha256')!=RESERVATION_SHA
            or frozen.get('discovery_receipt_sha256')!=receipt_hash
            or frozen.get('evidence')!=evidence or frozen.get('discovery')!=saved.discovery(records,cases[:8])):
            raise ValueError('published freeze does not match discovery evidence')
    elif any((args.discovery_dir,args.discovery_receipt,args.freeze,args.freeze_sha256)):
        raise ValueError('discovery cannot consume held-out freeze')
    import torch
    import looking_glass.pretrained as pretrained
    import distractor_executor as execution
    import distractor_core as core
    for name in STUDY[1:]:
        mod=sys.modules.get(Path(name).stem)
        if mod is not None and Path(mod.__file__).resolve()!=study/name:raise ValueError('study import differs')
    for name in DEPENDENCIES:
        key='looking_glass' if name.endswith('/__init__.py') else 'looking_glass.'+Path(name).stem
        mod=sys.modules.get(key)
        if mod is not None and Path(mod.__file__).resolve()!=source/name:raise ValueError('dependency import differs')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):raise RuntimeError('unable to disable subnormal flushing')
    core.ensure_round_to_nearest();torch.manual_seed(0)
    loaded=pretrained.load_distilgpt2(cache_dir=args.cache_dir,allow_download=False,stage_dir=stage)
    if any(stage.iterdir()):raise RuntimeError('checkpoint stage is not empty after load')
    if (loaded.metadata['model_revision']!='2290a62682d06624634c1f46a6ad5be0f47f38aa'
        or loaded.metadata['weights_sha256']!='e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
        or loaded.metadata['asset_sha256']['tokenizer.json']!='8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'):
        raise ValueError('checkpoint identity differs')
    for case in cases:
        text=case['public_text'];prefix_ids=case['scored_prefix_token_ids']
        if (not text.endswith('.') or loaded.tokenizer.encode(text,add_special_tokens=False)!=case['token_ids']
            or loaded.tokenizer.encode(text[:-1],add_special_tokens=False)!=prefix_ids
            or any(loaded.tokenizer.encode(text[:-1]+suffix,add_special_tokens=False)!=prefix_ids+[token]
                   for token,suffix in ((318,' is'),(389,' are')))):
            raise ValueError('loaded tokenizer disagrees with fixed case')
    chosen=cases[:8] if args.phase=='discovery' else cases[8:]
    result=execution.execute_phase(loaded.model,chosen,args.output_dir,phase=args.phase,
        additional_space_checks=((source,256*1024**2),(study,256*1024**2),(args.cache_dir,256*1024**2)))
    if result['completed_actions']!=224 or any(prefix.iterdir()) or any(stage.iterdir()):
        raise RuntimeError('phase count or runtime directories differ')
    print(json.dumps({'event':'phase-complete','phase':args.phase,'actions':224,
                      'raw_bytes':result['raw_bytes'],'admission':admission_report},sort_keys=True),flush=True)

if __name__=='__main__':main()
