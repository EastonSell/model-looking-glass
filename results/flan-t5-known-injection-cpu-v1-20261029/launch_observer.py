"""Run one source-pinned FLAN study child or its no-model --help rehearsal.

The saved observation covers the directly waited base interpreter only.
It does not attest descendant lifetimes, independent clocks, or open handles.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path('E:/Codex/model-looking-glass/work')
REPO = ROOT / 'work/teams/flan_t5_sync_merged_check'
PYTHON = Path(sys._base_executable)
VENV = ROOT / 'work/.venv'
ASSETS = ROOT / 'work/t5-cache/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
STAGE = ROOT / 'work/flan-injection-public-v1'
RUNNER = STAGE / 'run_known_injection.py'
CASES = STAGE / 'cases-v1.json'
RUNNER_SHA = '84ede689c01df925ded432823afa81a34e5e74b2f9621480fba746e854be48ac'
FILES = {
    'runner': (RUNNER, RUNNER_SHA),
    'cases': (CASES, '1b21870b8d2c83a7c144ecbf2911728fcdc14a828bf2e8078fca43780df64c11'),
    'init': (REPO/'looking_glass/__init__.py', '9e937f02ec3e4ce0908849aa1741d8afafe378e64212362dbf67af73e11a4fe0'),
    'trace': (REPO/'looking_glass/flan_t5_trace.py', '499ecb4bd3cc4eab253d525962a8b616a5c87421e5acaaa32ffb198484c93d4c'),
    'assets': (REPO/'looking_glass/flan_t5_feasibility.py', '67eace7da47a644a78ec39c802472ab296c65a1fcb41136ee6dd361e0ff1b63c'),
    'schema': (REPO/'looking_glass/trace_schema.py', '38e05eb52a59a10ad9f93d19763d60df4c27f9eb626fdefdcba39fc4af3eb92c'),
    'numeric': (REPO/'looking_glass/agreement_numeric.py', 'f9a5a9ee3627162d843c0dc0d8c6ea9d98c000a4b6d472fb55254fb3395d07dc'),
    'plan': (REPO/'looking_glass/intervention_plan.py', '5bb643c538a1d7dcd97214238c2a996967d4b6ae2e965d3776b3699758491c0a'),
    'interventions': (REPO/'looking_glass/interventions.py', '71cbcf4776660d2dd883b82815f710b0c909228fd1762504f9e3f984f7f6c60f'),
    'ownership': (REPO/'looking_glass/model_ownership.py', '99fee108f56ff8b7b7361e39de982c242cff590cc06c5d7a640cc12123cfe434'),
}
BOOTSTRAP = '''import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); repo=sys.argv[2]; expected=sys.argv[3]
venv=Path(sys.argv[4]); observer=Path(sys.argv[5])
if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode: raise RuntimeError("interpreter isolation flags differ")
if Path(sys.executable).resolve()!=Path(sys._base_executable).resolve(): raise RuntimeError("not the direct base interpreter")
if not venv.is_absolute() or not venv.is_dir() or venv.is_symlink() or getattr(venv.lstat(),"st_file_attributes",0)&0x400: raise RuntimeError("invalid venv")
config=venv/"pyvenv.cfg"; cfg=config.read_bytes()
if len(cfg)>16384 or config.is_symlink(): raise RuntimeError("invalid venv config")
entries={}
for line in cfg.decode("utf8").splitlines():
 if "=" in line:
  key,value=(part.strip() for part in line.split("=",1)); key=key.lower()
  if key in entries: raise RuntimeError("duplicate venv entry")
  entries[key]=value
if Path(entries.get("home","")).resolve()!=Path(sys.base_prefix).resolve() or entries.get("implementation")!="CPython" or entries.get("version_info")!=".".join(map(str,sys.version_info[:3])) or entries.get("include-system-site-packages","").lower()!="false": raise RuntimeError("venv/base mismatch")
site=venv/"Lib/site-packages"
if not site.is_dir() or site.is_symlink() or getattr(site.lstat(),"st_file_attributes",0)&0x400: raise RuntimeError("invalid dependency site")
sys.path.append(str(site))
raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=expected: raise RuntimeError("runner source changed")
sys.path.insert(0,repo)
identity={"pid":os.getpid(),"parent_pid":os.getppid(),"base_executable_sha256":hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),"venv_config_sha256":hashlib.sha256(cfg).hexdigest(),"python_version":sys.version.split()[0],"isolated":bool(sys.flags.isolated),"no_site":bool(sys.flags.no_site),"no_bytecode":bool(sys.dont_write_bytecode)}
data=(json.dumps(identity,sort_keys=True,separators=(",",":"))+"\\n").encode()
with (observer/"child-identity.json").open("xb") as f:
 if f.write(data)!=len(data): raise OSError("short child identity")
 f.flush();os.fsync(f.fileno())
sys.argv=[str(p)]+sys.argv[6:]
exec(compile(raw,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('rehearsal','study'),required=True)
    parser.add_argument('--expected-main',required=True)
    parser.add_argument('--runner-merge',required=True)
    args=parser.parse_args()
    require(len(args.expected_main)==40 and all(c in '0123456789abcdef' for c in args.expected_main), 'invalid main identity')
    require(len(args.runner_merge)==40 and all(c in '0123456789abcdef' for c in args.runner_merge), 'invalid runner merge')
    check=subprocess.run(['git','ls-remote','origin','refs/heads/main'],cwd=REPO,capture_output=True,timeout=15,check=True)
    require(check.stdout.strip()==(args.expected_main+'\trefs/heads/main').encode(),'official main changed')
    published=subprocess.check_output(['git','show',args.runner_merge+':tools/flan_t5_known_injection_v1.py'],cwd=REPO)
    require(published==RUNNER.read_bytes(),'published runner differs')
    for name,(path,expected) in FILES.items():
        require(sha(path)==expected,'source or case differs: '+name)
    require(OUTPUT_ROOT.is_dir() and not OUTPUT_ROOT.is_symlink() and not (getattr(OUTPUT_ROOT.lstat(),'st_file_attributes',0)&0x400),'invalid output root')
    work_free=shutil.disk_usage(OUTPUT_ROOT).free
    asset_free=shutil.disk_usage(ASSETS).free
    require(min(work_free,asset_free)>=2*1024**3,'insufficient fresh disk space')
    output=OUTPUT_ROOT/('flan-known-injection-'+args.mode+'-output-v3')
    observer=OUTPUT_ROOT/('flan-known-injection-'+args.mode+'-observer-v3')
    require(not output.exists() and not observer.exists(),'output or observer already exists')
    observer.mkdir(parents=True,exist_ok=False)
    pycache=observer/'empty-pycache'
    pycache.mkdir(exist_ok=False)
    st=pycache.lstat()
    require(not pycache.is_symlink() and not (getattr(st,'st_file_attributes',0)&0x400) and not any(pycache.iterdir()),'invalid pycache')
    suffix=(['--help'] if args.mode=='rehearsal' else ['--assets',str(ASSETS),'--cases',str(CASES),'--output',str(output)])
    cmd=[str(PYTHON),'-I','-S','-B','-X','pycache_prefix='+str(pycache),'-c',BOOTSTRAP,str(RUNNER),str(REPO),RUNNER_SHA,str(VENV),str(observer),*suffix]
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONPYCACHEPREFIX','PYTHONOPTIMIZE'):
        env.pop(key,None)
    env.update({'CUDA_VISIBLE_DEVICES':'-1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    start=now(); proc=None; pid=None; code=None; waited=False; failure=None; cleanup='not_needed'
    with (observer/'stdout.txt').open('xb') as out,(observer/'stderr.txt').open('xb') as err:
        try:
            proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            pid=proc.pid
            code=proc.wait()
            waited=True
        except BaseException as exc:
            failure=type(exc).__name__
            if proc is None:
                cleanup='spawn_return_uncertain'
            else:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                    code=proc.wait(timeout=15)
                    waited=True
                    cleanup='direct_child_reaped_after_interruption'
                except BaseException:
                    cleanup='direct_child_cleanup_uncertain'
    finish=now()
    pycache_empty=None
    after=None; child_identity=None
    if waited:
        try:
            st=pycache.lstat()
            pycache_empty=not pycache.is_symlink() and not (getattr(st,'st_file_attributes',0)&0x400) and not any(pycache.iterdir())
            after={name:sha(path) for name,(path,_) in FILES.items()}
            child_identity=json.loads((observer/'child-identity.json').read_bytes())
        except (OSError,ValueError):
            pycache_empty=False
    def saved(name):
        path=output/name
        return sha(path) if waited and path.is_file() else None
    record={'schema':'flan-known-injection-direct-launch-observation-v1', 'mode':args.mode,
      'scope':'Cooperative directly waited base interpreter only; descendants, open output handles, and clock synchronization are not independently attested.',
      'child_pid':pid,'child_identity_after_wait':child_identity,'start_utc':start,'end_utc':finish,'wait_returned':waited,'exit_code':code,
      'observer_failure':failure,'cleanup':cleanup,'pycache_empty_after_wait':pycache_empty,
      'official_main_observed':args.expected_main,'runner_merge':args.runner_merge,
      'cases_merge':'a96bba67ca8067aa7bd6408455bfda81d9bb03b6',
      'source_hashes_before':{name:expected for name,(_,expected) in FILES.items()},'source_hashes_after_wait':after,
      'bootstrap_sha256':hashlib.sha256(BOOTSTRAP.encode()).hexdigest(),
      'command':'base-python -I -S -B -X pycache_prefix=<new-empty-dir> -c <source-bound bootstrap> <published-runner> <repo> <runner-sha> <venv> <observer> '+('--help' if args.mode=='rehearsal' else '--assets <offline-assets> --cases <published-cases> --output <new-output>'),
      'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONNOUSERSITE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
      'free_work_bytes':work_free,'free_asset_volume_bytes':asset_free,
      'stdout_sha256':sha(observer/'stdout.txt') if waited else None,'stderr_sha256':sha(observer/'stderr.txt') if waited else None,
      'journal_sha256':saved('attempt.jsonl'),'result_sha256':saved('result.json')}
    data=(json.dumps(record,sort_keys=True,separators=(',',':'))+'\n').encode()
    with (observer/'observation.json').open('xb') as stream:
        require(stream.write(data)==len(data),'short observation write')
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'mode':args.mode,'child_pid':pid,'exit_code':code,'wait_returned':waited,
      'failure':failure,'cleanup':cleanup,'pycache_empty_after_wait':pycache_empty,
      'journal_sha256':record['journal_sha256'],'result_sha256':record['result_sha256']}))
    if (failure is not None or not waited or code!=0 or not pycache_empty
            or after!={name:expected for name,(_,expected) in FILES.items()}
            or not child_identity or child_identity.get('pid')!=pid or child_identity.get('parent_pid')!=os.getpid()
            or (args.mode=='study' and (not record['journal_sha256'] or not record['result_sha256']))):
        raise SystemExit(1)


if __name__=='__main__':
    main()
