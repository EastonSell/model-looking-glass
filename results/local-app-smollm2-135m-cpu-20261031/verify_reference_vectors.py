"""Offline, explicit-checkpoint HTTP reference-vector reproduction.

Requires the app CPU test/runtime dependencies already installed. This script
never downloads weights, installs packages, or overwrites evidence. It starts an
authenticated loopback server for its own process and deletes temporary exports.
"""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode

REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
HISTORICAL_RECORD=Path(__file__).resolve().with_name('record.json')
HISTORICAL_RECORD_SHA256='d2425a1bd3848f1378570be7262c138c883be076070dab533ad19b5a279cfe32'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def verify_checkpoint(checkpoint, manifest):
    if len(manifest)!=9:raise RuntimeError('historical manifest must contain exactly nine assets')
    for entry in checkpoint.iterdir():
        if entry.name=='.cache' and entry.is_dir() and not entry.is_symlink():continue
        if entry.name not in manifest or entry.is_symlink() or not entry.is_file():
            raise RuntimeError('unverified extra or redirected checkpoint asset')
    hashes={p.name:file_hash(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}
    if set(hashes)!=set(manifest) or any(hashes[name]!=record['sha256']
            or (checkpoint/name).stat().st_size!=record['bytes'] for name,record in manifest.items()):
        raise RuntimeError('local assets differ from exact pinned nine-file manifest')
    return hashes


def implementation_hashes(implementation):
    paths=set((implementation/'looking_glass').glob('app*.py'))
    paths.update(implementation/'looking_glass'/name for name in
        ('llama_trace.py','intervention_plan.py','model_ownership.py','trace_schema.py','agreement_numeric.py'))
    return {str(path.relative_to(implementation)):file_hash(path) for path in sorted(paths)}


def state_hashes(model):
    # The actual CPU float32 checkpoint is contiguous. Hash views, not cloned weights.
    import torch
    result={}
    for kind,iterator in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
        for name,value in iterator:
            if value.device.type!='cpu' or not value.is_contiguous():
                raise RuntimeError('zero-copy CPU contiguous hashing precondition failed')
            view=value.detach().view(torch.uint8).numpy()
            result[f'{kind}:{name}']=hashlib.sha256(memoryview(view).cast('B')).hexdigest()
    return result


def raw_bits(value):
    return f"0x{struct.unpack('<I',struct.pack('<f',value))[0]:08x}"


def check_metrics(pair,reference,current):
    if len(reference)!=576 or len(current)!=576:raise RuntimeError('unexpected vector width')
    deltas=[float(b)-float(a) for a,b in zip(reference,current)]
    l1=math.fsum(abs(x) for x in deltas)
    l2=math.sqrt(math.fsum(x*x for x in deltas))
    norm_a=math.sqrt(math.fsum(x*x for x in reference));norm_b=math.sqrt(math.fsum(x*x for x in current))
    cosine=math.fsum(a*b for a,b in zip(reference,current))/(norm_a*norm_b) if norm_a and norm_b else None
    expected=dict(max_abs_delta=max(map(abs,deltas)),l1=l1,l2=l2,cosine=cosine,
        raw_bit_changed_count=sum(raw_bits(a)!=raw_bits(b) for a,b in zip(reference,current)),
        numeric_nonzero_count=sum(x!=0 for x in deltas))
    metrics=pair['metrics']
    for key in ('max_abs_delta','l1','l2'):
        if not math.isclose(metrics[key],expected[key],rel_tol=1e-12,abs_tol=1e-14):
            raise RuntimeError(f'independent scalar metric mismatch: {key}')
    if cosine is None:
        if metrics['cosine'] is not None or not metrics['cosine_reason']:raise RuntimeError('missing zero-norm explanation')
    elif not math.isclose(metrics['cosine'],max(-1.,min(1.,cosine)),rel_tol=1e-12,abs_tol=1e-14):
        raise RuntimeError('independent cosine mismatch')
    for key in ('raw_bit_changed_count','numeric_nonzero_count'):
        if metrics[key]!=expected[key]:raise RuntimeError(f'independent count mismatch: {key}')
    return expected,deltas


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True,help='Existing exact pinned local checkpoint directory; no downloads')
    parser.add_argument('--output',required=True,help='New JSON evidence file; existing files are never overwritten')
    parser.add_argument('--implementation-root',default=str(Path(__file__).resolve().parents[2]),
                        help='Repository or installed-package root to verify')
    args=parser.parse_args()
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    implementation=Path(args.implementation_root).expanduser().resolve()
    output=Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('output must be a new file')
    if not output.parent.is_dir():raise ValueError('output parent directory must already exist')
    if not checkpoint.is_dir():raise ValueError('checkpoint must be an existing local directory')
    if output.resolve().is_relative_to(checkpoint):raise ValueError('output must not be inside the checkpoint')
    if Path(tempfile.gettempdir()).resolve().is_relative_to(checkpoint):
        raise ValueError('temporary directory must not be inside the checkpoint')
    if file_hash(HISTORICAL_RECORD)!=HISTORICAL_RECORD_SHA256:
        raise RuntimeError('historical evidence record differs from its pinned SHA256')
    script_sha256=file_hash(Path(__file__).resolve())
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,str(implementation))
    import torch,transformers
    from looking_glass import app,app_runtime,app_storage
    for module in (app,app_runtime,app_storage):
        if not Path(module.__file__).resolve().is_relative_to(implementation):raise RuntimeError('wrong implementation import origin')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    historical=json.loads(HISTORICAL_RECORD.read_text())['source']
    if historical['repository']!='HuggingFaceTB/SmolLM2-135M' or historical['revision']!=REVISION:
        raise RuntimeError('historical manifest identity mismatch')
    manifest=historical['asset_manifest']
    initial_files=verify_checkpoint(checkpoint,manifest)
    sources=implementation_hashes(implementation)
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='mlg-reference-verification-') as temporary:
        state=app.AppState(export_root=Path(temporary)/'exports')
        server=app.create_server(state);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        def call(path,body=None,expected=200,authenticated=True):
            connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=120)
            headers={'X-MLG-Token':state.token} if authenticated else {}
            if body is not None:headers['Content-Type']='application/json'
            connection.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body),headers=headers)
            response=connection.getresponse();data=response.read();status=response.status;connection.close()
            if status!=expected:raise RuntimeError(f'HTTP {path.split("?")[0]} status {status}, expected {expected}: {data[:300]!r}')
            return json.loads(data)
        def get(path,**query):return call(path+'?'+urlencode(query))
        def tensor(capture,event,vector):return get('/api/tensor',capture_id=capture,event=event,vector=vector,limit=4096)
        def events(run):
            return get('/api/events',capture_id=run['capture_id'],offset=0,limit=100)['events']
        def selected(run):
            candidates=[row for row in events(run) if row['metadata']['module_path']=='model.layers.15.self_attn'
                        and row['metadata']['app_forward_index']==0]
            if len(candidates)!=1:raise RuntimeError('selected boundary is ambiguous or missing')
            return candidates[0]['index'],len(run['steps'][0]['input_token_ids'])-1
        def trace_hashes(run):
            result=[]
            for row in events(run):
                for vector in range(row['vector_count']):
                    result.append((row['index'],vector,digest(tensor(run['capture_id'],row['index'],vector))))
            return digest(result),len(result)
        try:
            call('/api/status',authenticated=False,expected=403)
            call('/api/load',{'path':str(checkpoint),'device':'cpu'})
            model=state.runtime.model;parameters_before=state_hashes(model)
            first=call('/api/run',{'prompt':'Gravity is','settings':{'max_new_tokens':1}})
            event,vector=selected(first);reference_page=tensor(first['capture_id'],event,vector)
            original_hash,original_vectors=trace_hashes(first)
            pin=call('/api/pin-reference',{'capture_id':first['capture_id'],'event':event,'vector':vector})
            if state.reference._value.untyped_storage().data_ptr()==state.store._values[event].untyped_storage().data_ptr():
                raise RuntimeError('pinned reference aliases original trace storage')
            rid=pin['reference_id']
            if call('/api/status')['reference']['reference_id']!=rid:raise RuntimeError('status reference identity mismatch')
            if pin['width']!=576 or pin['dtype']!='float32':raise RuntimeError('pinned summary dtype/width mismatch')
            if trace_hashes(first)[0]!=original_hash:raise RuntimeError('pin changed original trace')
            pinned=get('/api/reference-tensor',reference_id=rid,offset=0,limit=4096)
            if pinned['raw_bits']!=reference_page['raw_bits'] or pinned['values']!=reference_page['values']:
                raise RuntimeError('reference copy differs from captured source')
            pinned_digest=digest(pinned)
            second=call('/api/run',{'prompt':'Electricity is','settings':{'max_new_tokens':1}})
            current_event,current_vector=selected(second);capture=second['capture_id']
            current_page=tensor(capture,current_event,current_vector)
            call('/api/pin-reference',{'capture_id':capture,'event':current_event,'vector':current_vector,
                 'expected_reference_id':'stale-reference'},expected=400)
            current_hash,current_vectors=trace_hashes(second)
            if digest(get('/api/reference-tensor',reference_id=rid,offset=0,limit=4096))!=pinned_digest:
                raise RuntimeError('reference failed to survive new prompt')
            query=dict(reference_id=rid,capture_id=capture,event=current_event,vector=current_vector,offset=0,limit=4096)
            pair=get('/api/compare-vector',**query)
            if pair['reference_id']!=rid or pair['capture_id']!=capture:raise RuntimeError('comparison identities mismatch')
            if pair['reference']['coordinate']!=reference_page['coordinate'] or pair['current']['coordinate']!=current_page['coordinate']:
                raise RuntimeError('comparison does not retain explicit independent coordinates')
            if pair['total']!=576 or pair['offset']!=0:raise RuntimeError('pair page shape mismatch')
            if pair['reference_raw_bits']!=reference_page['raw_bits'] or pair['current_raw_bits']!=current_page['raw_bits']:
                raise RuntimeError('pair raw bits mismatch')
            if pair['reference_values']!=reference_page['values'] or pair['current_values']!=current_page['values']:
                raise RuntimeError('pair float32 values mismatch')
            for values,bits in ((pair['reference_values'],pair['reference_raw_bits']),(pair['current_values'],pair['current_raw_bits'])):
                if [raw_bits(value) for value in values]!=bits:raise RuntimeError('float values and bit encodings disagree')
            metrics,deltas=check_metrics(pair,pair['reference_values'],pair['current_values'])
            if pair['deltas']!=deltas:raise RuntimeError('full float64 coordinate deltas differ')
            # Independently verify paginated coordinate coverage; no page-local normalization.
            for offset in range(0,576,73):
                page=get('/api/compare-vector',**{**query,'offset':offset,'limit':73})
                count=min(73,576-offset)
                if page['deltas']!=deltas[offset:offset+count] or page['metrics']!=pair['metrics']:
                    raise RuntimeError('comparison pagination changed coordinates/full-width metrics')
            for invalid in ({**query,'event':current_event+1},{**query,'capture_id':first['capture_id']},{**query,'reference_id':'stale-reference'}):
                call('/api/compare-vector?'+urlencode(invalid),expected=400)
                if digest(get('/api/reference-tensor',reference_id=rid,offset=0,limit=4096))!=pinned_digest:
                    raise RuntimeError('failed comparison altered reference')
            if trace_hashes(second)[0]!=current_hash:raise RuntimeError('comparison altered current trace')
            if state_hashes(model)!=parameters_before:raise RuntimeError('parameters/buffers changed during comparisons')
            if state.store.export_root.exists():raise RuntimeError('RAM-only flow wrote export before explicit Save')
            saved=call('/api/save',{'name':'current-only','capture_id':capture})
            del model  # Do not retain a second weight owner across unload/reload.
            call('/api/unload',{})
            if digest(get('/api/reference-tensor',reference_id=rid,offset=0,limit=4096))!=pinned_digest:
                raise RuntimeError('unload destroyed pinned reference')
            if call('/api/status')['reference']['ready_for_compare']:raise RuntimeError('unloaded reference incorrectly ready for comparison')
            call('/api/compare-vector?'+urlencode(query),expected=400)
            reopened=call('/api/open',{'path':saved['path']})
            imported=reopened['capture_id']
            call('/api/compare-vector?'+urlencode({**query,'capture_id':imported}),expected=400)
            call('/api/load',{'path':str(checkpoint),'device':'cpu'})
            new_run=call('/api/run',{'prompt':'Electricity is','settings':{'max_new_tokens':1}})
            new_event,new_vector=selected(new_run)
            call('/api/compare-vector?'+urlencode({**query,'capture_id':new_run['capture_id'],'event':new_event,'vector':new_vector}),expected=400)
            if digest(get('/api/reference-tensor',reference_id=rid,offset=0,limit=4096))!=pinned_digest:
                raise RuntimeError('reload invalidation destroyed readable reference')
            if state_hashes(state.runtime.model)!=parameters_before:raise RuntimeError('reloaded weights differ')
            call('/api/clear-reference',{'reference_id':'stale-reference'},expected=400)
            call('/api/clear-reference',{'reference_id':rid})
            # A saved current trace cannot restore a separately pinned RAM reference.
            call('/api/open',{'path':saved['path']})
            if call('/api/status')['reference'] is not None:raise RuntimeError('current export resurrected pinned reference')
            call('/api/unload',{})
            if verify_checkpoint(checkpoint,manifest)!=initial_files:raise RuntimeError('source checkpoint files changed')
            if sources!=implementation_hashes(implementation):
                raise RuntimeError('implementation changed during validation')
            report=dict(status='passed',model='HuggingFaceTB/SmolLM2-135M',revision=REVISION,
                device='cpu',dtype='float32',python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                reference_prompt='Gravity is',current_prompt='Electricity is',reference_coordinate=reference_page['coordinate'],
                current_coordinate=current_page['coordinate'],reference_input_token_ids=first['steps'][0]['input_token_ids'],
                current_input_token_ids=second['steps'][0]['input_token_ids'],width=576,independent_scalar_metrics=metrics,
                all576_float32_bits_and_float64_deltas_verified=True,pagination_preserves_full_width_metrics=True,
                original_trace_vectors=original_vectors,current_trace_vectors=current_vectors,
                original_trace_unchanged_by_pin=True,current_trace_unchanged_by_comparison=True,
                reference_survives_new_prompt_bad_requests_unload_reload=True,stale_imported_reloaded_bindings_rejected=True,
                explicit_clear_required=True,current_save_does_not_persist_reference=True,
                parameters_and_buffers_unchanged=True,checkpoint_files_unchanged=True,
                pinned_tensor_has_distinct_storage=True,stale_replacement_rejected=True,
                checkpoint_file_sha256=initial_files,exact_nine_asset_manifest_verified=True,
                historical_manifest_record_sha256=file_hash(HISTORICAL_RECORD),implementation_sha256=sources,
                reference_http_response_sha256=digest(reference_page),current_http_response_sha256=digest(current_page),
                http_response_hash_note='Response hashes include session/run identities and are not portable vector identities.',
                reference_vector_bits_sha256=hashlib.sha256(b''.join(struct.pack('<I',int(bits,16)) for bits in reference_page['raw_bits'])).hexdigest(),
                current_vector_bits_sha256=hashlib.sha256(b''.join(struct.pack('<I',int(bits,16)) for bits in current_page['raw_bits'])).hexdigest(),
                vector_hash_encoding='Concatenated little-endian uint32 raw float32 encodings in feature order',
                verification_script_sha256=script_sha256,
                elapsed_seconds_not_benchmark=round(time.monotonic()-started,3),
                scope='Actual short CPU local-checkpoint comparison at an explicitly selected common hook; no automatic token/semantic alignment, GPU evidence, causal explanation, or model-quality claim.')
        finally:
            server.shutdown();worker.join(5);server.server_close()
            if worker.is_alive():raise RuntimeError('HTTP server thread did not stop')

    if file_hash(HISTORICAL_RECORD)!=HISTORICAL_RECORD_SHA256:raise RuntimeError('historical record changed during execution')
    if file_hash(Path(__file__).resolve())!=script_sha256:raise RuntimeError('verification script changed during execution')
    with output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoint_file_sha256','implementation_sha256')},indent=2,allow_nan=False))


if __name__=='__main__':main()
