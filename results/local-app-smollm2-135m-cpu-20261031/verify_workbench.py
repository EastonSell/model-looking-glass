"""Explicit local-only installed-wheel pretrained workbench verification."""
import sys
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description="Offline verification of one exact public SmolLM2 revision; no downloads")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--implementation-root", type=Path, default=Path(__file__).resolve().parents[2], help="Checkout or separately installed package directory")
parser.add_argument("--export-root", type=Path, required=True, help="New output directory; never overwritten")
args=parser.parse_args()
if not __debug__: raise RuntimeError("Run without -O; verification assertions are required")
INSTALLED=args.implementation_root.resolve(strict=True)
import os
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[name]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["TOKENIZERS_PARALLELISM"]="false"
sys.path.insert(0,str(INSTALLED))
import hashlib,http.client,json,threading,time,platform
import torch,transformers
from looking_glass import app,app_runtime,app_storage
assert all(Path(m.__file__).is_relative_to(INSTALLED) for m in (app,app_runtime,app_storage))
CHECKPOINT=args.checkpoint.resolve(strict=True)
REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'

EXPECTED_ASSETS = {'README.md': {'bytes': 6340, 'sha256': 'd1ba68cae64a89b6b434b11526e6e2271ee5ffd2c914ec35ed515f9d84c6085c'}, 'config.json': {'bytes': 704, 'sha256': '1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843'}, 'generation_config.json': {'bytes': 111, 'sha256': '2056c988e990b0d13670f63f2f3b87b3b6d07edaf7a3416998ba27dab2d8a059'}, 'merges.txt': {'bytes': 466391, 'sha256': '0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510'}, 'model.safetensors': {'bytes': 269060552, 'sha256': '80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'}, 'special_tokens_map.json': {'bytes': 831, 'sha256': 'e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3'}, 'tokenizer.json': {'bytes': 2104556, 'sha256': '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'}, 'tokenizer_config.json': {'bytes': 3658, 'sha256': '4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424'}, 'vocab.json': {'bytes': 800662, 'sha256': '82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79'}}

def check_assets(folder):
    # The Hub's local-download bookkeeping is not a model/tokenizer input.
    names={p.name for p in folder.iterdir() if not (p.name==".cache" and p.is_dir() and not p.is_symlink())}
    if names != set(EXPECTED_ASSETS):
        raise RuntimeError("Checkpoint inventory differs from the exact pinned assets")
    for name, expected in EXPECTED_ASSETS.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise RuntimeError("Missing or wrong-sized pinned asset: " + name)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected["sha256"]:
            raise RuntimeError("Pinned asset hash mismatch: " + name)

check_assets(CHECKPOINT)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hashes():
    result={}
    for name in sorted(EXPECTED_ASSETS):
        p=CHECKPOINT/name
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        result[p.name]=h.hexdigest()
    return result
original_files=file_hashes()
state=app.AppState(export_root=args.export_root)
assert not state.store.export_root.exists(),'new export directory required'
server=app.create_server(state);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
def call(path,body=None,authenticated=True,expected=200):
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=120)
    headers={'X-MLG-Token':state.token} if authenticated else {}
    if body is not None:headers['Content-Type']='application/json'
    conn.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body),headers=headers)
    response=conn.getresponse();data=response.read();status=response.status;conn.close()
    assert status==expected,(path,status,data[:1000])
    return json.loads(data)
start=time.monotonic()
try:
    call('/api/status',authenticated=False,expected=403)
    inspect=call('/api/inspect',{'path':str(CHECKPOINT)})
    assert inspect['family']=='llama' and not inspect['inspection']['weights_loaded']
    info=call('/api/load',{'path':str(CHECKPOINT),'device':'cpu'})
    assert info['device']=='cpu' and info['architecture']=='LlamaForCausalLM'
    model=state.runtime.model
    assert model.lm_head.weight is model.model.embed_tokens.weight
    rotary=model.model.rotary_emb.inv_freq.view(torch.int32).clone()
    run=call('/api/run',{'prompt':'Gravity is','settings':{'max_new_tokens':4}})
    assert run['complete'] and run['model']['family']=='llama' and not run['intervention_available']
    steps=run['steps'];assert 1<=len(steps)<=4 and run['event_count']==90*len(steps)
    for i,step in enumerate(steps):
        assert len(step['input_attention_mask'])==len(step['input_token_ids'])
        if i:
            assert step['input_token_ids']==steps[i-1]['input_token_ids']+[steps[i-1]['selected_token_id']]
            assert step['input_attention_mask']==steps[i-1]['input_attention_mask']+[1]
    assert torch.equal(rotary,model.model.rotary_emb.inv_freq.view(torch.int32))
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    capture=run['capture_id'];events=[]
    for offset in range(0,run['event_count'],100):events.extend(call(f'/api/events?offset={offset}&limit=100&capture_id={capture}')['events'])
    assert len(events)==run['event_count']
    seen=set();vectors=[];total_bytes=0
    for row in events:
        md=row['metadata'];f=md['app_forward_index'];ids=steps[f]['input_token_ids'];n=len(ids)
        assert row['shape']==[1,n,576] and row['vector_count']==n
        path=md['module_path'];selector=[0] if path.endswith('.self_attn') else []
        assert md['output_path']==selector
        key=(f,path);assert key not in seen;seen.add(key)
        for vector,token_id in enumerate(ids):
            value=call(f"/api/tensor?event={row['index']}&vector={vector}&limit=4096&capture_id={capture}")
            c=value['coordinate'];assert c['status']=='known' and c['stack']=='input'
            assert c['token_position']==vector and c['token_id']==token_id and c['app_forward_index']==f
            assert len(value['raw_bits'])==576
            vectors.append((row['index'],vector,digest(value)))
        total_bytes+=n*576*4
    expected={(f,f'model.layers.{layer}'+suffix) for f in range(len(steps)) for layer in range(30) for suffix in ('.self_attn','.mlp','')}
    assert seen==expected
    assert total_bytes==run['total_bytes']
    assert not state.store.export_root.exists(),'RAM-only capture wrote export before Save'
    saved=call('/api/save',{'name':'smollm2-135m-cpu','capture_id':capture})
    call('/api/unload',{});call('/api/clear',{})
    reopened=call('/api/open',{'path':saved['path']});new_id=reopened['capture_id']
    assert reopened['event_count']==run['event_count'] and reopened['model']==run['model']
    assert reopened['steps']==steps and not reopened['intervention_available']
    for event,vector,expected_hash in vectors:
        value=call(f'/api/tensor?event={event}&vector={vector}&limit=4096&capture_id={new_id}')
        assert digest(value)==expected_hash
    call('/api/tensor?event=0&capture_id='+capture,expected=400)
    check_assets(CHECKPOINT)
    assert file_hashes()==original_files
    report={'status':'passed','model':'HuggingFaceTB/SmolLM2-135M','revision':REVISION,
            'installed_wheel':bool(list(INSTALLED.glob('model_looking_glass-*.dist-info'))),'device':'cpu','dtype':'float32','torch':torch.__version__,
            'transformers':transformers.__version__,'python':platform.python_version(),
            'prompt':'Gravity is','answer':run.get('answer'),'steps':steps,
            'forward_count':len(steps),'event_count':len(events),'vector_count':len(vectors),
            'captured_tensor_bytes':total_bytes,'all_boundary_inventory_exact':True,
            'all_vector_values_bits_coordinates_equal_after_reopen':True,
            'vector_response_sha256':digest(vectors),'ram_only_until_explicit_save':True,
            'source_files_unchanged':True,'rotary_bits_unchanged':True,'hooks_removed':True,
            'tied_weights_preserved':True,'auth_and_stale_capture_guards':True,
            'elapsed_seconds_not_benchmark':round(time.monotonic()-start,3),
            'source_files':original_files,
            'limits':'Single short CPU workbench verification; not model-quality, GPU, all-checkpoint or maximum-context validation.'}
    print(json.dumps(report,indent=2))
finally:
    server.shutdown();worker.join(5);server.server_close();assert not worker.is_alive()
