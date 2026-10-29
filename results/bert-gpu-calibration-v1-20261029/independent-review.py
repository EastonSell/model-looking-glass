"""Audit a DistilBERT GPU result, private evidence, source, and optional cache.

Usage: python review.py RESULT EVIDENCE_DIR SOURCE_CHECKOUT [HF_CACHE_DIR]
The optional cache check hashes all five pinned files; without it this script
checks their reported metadata, not the original bytes read during the run.
"""
import base64, hashlib, json, math, pathlib, re, struct, subprocess, sys

if not __debug__:
    raise SystemExit('Run without Python -O: this audit uses assertions')
result_path, run_dir, checkout = map(lambda p: pathlib.Path(p).resolve(strict=True), sys.argv[1:4])
result = json.loads(result_path.read_bytes())
run_dir = run_dir.resolve(strict=True)
commit = 'c2bff8183003ac25ec28475d23ddd20e7026f370'
assert result['schema'] == 'bert-gpu-calibration-result-v1' and result['status'] == 'success'
assert result['source_commit'] == commit
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == commit
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=checkout, text=True).strip()
source_paths = {'looking_glass/__init__.py', 'looking_glass/bert_gpu_calibration.py',
    'looking_glass/gpu_oneshot.py', 'looking_glass/device_identity.py',
    'looking_glass/device_lease.py', 'looking_glass/supervisor.py',
    'looking_glass/memory_budget.py', 'looking_glass/intervention_plan.py',
    'looking_glass/interventions.py', 'looking_glass/model_ownership.py',
    'docs/BERT_GPU_CALIBRATION_V1.md', 'scripts/bert_gpu_bootstrap.py'}
assert set(result['source_sha256']) == source_paths
for path, expected in result['source_sha256'].items():
    working = (checkout / path).read_bytes()
    committed = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=checkout)
    assert working == committed and hashlib.sha256(working).hexdigest() == expected, path
assert len(result['source_sha256']) == 12
assert result['source_sha256']['looking_glass/supervisor.py'] == '87603aad90ff541e14e5a8f7c847f0237995be8e578fe1fb4f7e6c59aef94d2a'
assert result['source_sha256']['looking_glass/bert_gpu_calibration.py'] == 'bea625bc255949905e0c5d106d50d4bc0770fe65fef8f8ba2c69803fb30d4216'

def read(name): return json.loads((run_dir / name).read_bytes())
start, owner, candidate, final, supervisor = [read(x) for x in ('started.json','owner.json','candidate.json','final.json','supervision.json')]
run_id = result['run_id']
assert start['run_id'] == run_id and start['source_commit'] == commit
assert supervisor['run_id'] == run_id and supervisor['exit_code'] == 0 and supervisor['clean'] is True and supervisor['timed_out'] is False
assert result['supervisor']['exit_code'] == 0 and result['supervisor']['clean'] is True and result['supervisor']['timed_out'] is False
assert type(supervisor['pid']) is int and supervisor['pid'] > 0
for record in (owner,candidate,final):
    assert record['run_id'] == run_id and record['supervisor_run_id'] == supervisor['supervisor_run_id'] and record['pid'] == supervisor['pid']
assert owner['schema'] == 'bert-gpu-calibration-owner-v1'
assert candidate['schema'] == 'bert-gpu-calibration-candidate-v1'
assert final['schema'] == 'bert-gpu-calibration-final-v1'
assert final['candidate_sha256'] == hashlib.sha256((run_dir / 'candidate.json').read_bytes()).hexdigest()
assert final['active_memory_leases'] == 0
m = candidate['measurement']
assert m == result['measurement']
assert m['target_ids'] == [101,1996,7500,2379,1996,11500,103,3201,1012,102]
assert m['donor_ids'] == [101,1996,6617,2379,1996,11500,103,3201,1012,102]
assert m['output_token_ids'] == {'is':2003,'are':2024}
assert m['runtime_uuid_verified'] is True and m['device_name'] == 'NVIDIA GeForce RTX 2070 SUPER'
assert m['compute_capability'] == [7,5] and m['allocator_backend'] == 'native'
assert m['torch_version'] == '2.8.0+cu128' and m['cuda_build'] == '12.8' and m['python_version']
assert m['seed'] == 17 and m['allocator_fraction'] == .25
assert m['gpu_trace_claim_bytes'] == 65536 and m['host_trace_claim_bytes'] == 1048576
assert m['budget_active_before_candidate_write'] == 1
meta = m['model_metadata']
assets = {
'config.json':(483,'69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83'),
'model.safetensors':(267954768,'5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'),
'tokenizer.json':(466062,'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'),
'tokenizer_config.json':(48,'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'),
'vocab.txt':(231508,'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3')}
assert meta['model'] == 'distilbert/distilbert-base-uncased' and meta['revision'] == '12040accade4e8a0f71eabdb258fecc2e7e948be'
assert meta['assets'] == {name:{'size':size,'sha256':sha} for name,(size,sha) in assets.items()}
if len(sys.argv) == 5:
    snapshot = pathlib.Path(sys.argv[4]).resolve(strict=True) / 'models--distilbert--distilbert-base-uncased' / 'snapshots' / meta['revision']
    for name, (size, expected) in assets.items():
        count, digest = 0, hashlib.sha256()
        with (snapshot / name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(block)
                digest.update(block)
        assert (count, digest.hexdigest()) == (size, expected), name
elif len(sys.argv) != 4:
    raise SystemExit('usage: review.py RESULT EVIDENCE_DIR SOURCE_CHECKOUT [HF_CACHE_DIR]')
assert meta['model_class'] == 'DistilBertForMaskedLM' and meta['tokenizer_class'] == 'DistilBertTokenizerFast'
assert meta['transformers_version'] == '4.57.3' and meta['tokenizers_version'] and meta['safetensors_version']
assert meta['attention_implementation'] == 'eager' and meta['parameter_count'] == 66985530 and meta['cpu_tied_weight_verified'] is True
expected_config={'device':'cuda:0','threads':1,'interop_threads':1,'deterministic_algorithms':True,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True,'overwrite_module_params_on_conversion':False,'swap_module_params_on_conversion':False}
assert m['execution_config'] == expected_config
assert m['cleanup_weakrefs'] == {'model':False,'target':False,'donor':False}
assert m['cleanup']['allocated_bytes'] == 0 and m['cleanup']['reserved_bytes'] == 0
stages = [('before_model_load',0),('before_model_copy',None),('after_model_copy',None),('before_target',0),('after_target',1),('before_donor',1),('after_donor',2),('before_self',2),('after_self',3),('before_cross',3),('after_cross',4),('cleanup_before_workspace_clear',None),('after_cleanup',None)]
records=result['partial_stage_observations']
assert records['cleanup_weakrefs'] == m['cleanup_weakrefs'] == read('cleanup_weakrefs.json')
total=m['device_total_bytes']; cap=m['allocator_cap_bytes']; commitment=m['whole_run_commitment_bytes']
assert 7*1024**3 <= total <= 9*1024**3 and cap == int(total*.25) and commitment == cap+256*1024**2
reserve=max(1024**3,(15*total+99)//100)
previous=(0,0)
for name,count in stages:
    item=records[name]
    assert item == read(name+'.json') and item['observations_live'] == count and item['peak_scope']=='cumulative_since_pre_model_load_reset'
    v=item['memory']; free=v['device_free_bytes']; total2=v['device_total_bytes']; alloc=v['allocated_bytes']; reserved=v['reserved_bytes']; peak=v['peak_allocated_bytes']; peak_reserved=v['peak_reserved_bytes']
    assert total2==total and 0<=free<=total and 0<=alloc<=reserved<=total
    assert alloc<=peak<=peak_reserved<=total and reserved<=peak_reserved and (peak>=previous[0] and peak_reserved>=previous[1])
    assert free>=reserve+commitment and reserved+reserve+commitment<=total and free>=reserve+256*1024**2 and peak_reserved<=cap-256*1024**2
    previous=(peak,peak_reserved)
assert records['before_model_load']['memory']['device_free_bytes']>=3*1024**3
assert records['after_cleanup']['memory']==m['cleanup']
assert result['nvidia_smi_before']['device_name'] == result['nvidia_smi_after']['device_name'] == m['device_name']
assert result['post_telemetry_error_type'] is None and result['supervision_evidence_error_type'] is None
assert result['owner_presence_after_exit']['owner_pid_listed'] is not True
sequence=m['sequence']; rows=sequence['observations']
assert sequence['forward_count']==4 and [r['stage'] for r in rows]==['target','donor','self','cross']
assert all(sequence[k] is True for k in ('self_mask_logits_bitwise_equal','cross_source_landed','source_differs_from_target'))
before=[]; after=[]; margins=[]
for row in rows:
    pair=[]
    for token in ('is','are'):
        bits=row[token+'_bits']; value=row[token+'_logit']
        assert type(bits) is int and 0<=bits<=0xffffffff and type(value) is float and math.isfinite(value)
        assert struct.pack('<f',value)==bits.to_bytes(4,'little')
        pair.append(value)
    margins.append(pair[0]-pair[1])
    assert re.fullmatch('[0-9a-f]{64}',row['mask_logits_sha256'])
    for name,dest in [('before_float32_le_base64',before),('after_float32_le_base64',after)]:
        s=row[name]
        if s is None: dest.append(None); continue
        raw=base64.b64decode(s,validate=True)
        assert len(raw)==3072 and base64.b64encode(raw).decode()==s
        assert all(math.isfinite(v) for v in struct.unpack('<768f',raw))
        dest.append(raw)
    ops=['capture'] if row['stage'] in ('target','donor') else ['capture','patch','capture']
    names=['before'] if len(ops)==1 else ['before','patch','after']
    assert [e['operation'] for e in row['events']]==ops
    for index,event in enumerate(row['events']):
        assert event['order']==index and event['module_path']=='distilbert.transformer.layer.2'
        assert event['name']==names[index] and event['module_aliases']==[event['module_path']]
        assert event['call_site_known'] is False and event['forward_index']==0
        assert event['source_capture']==('before' if row['stage']=='self' and index==1 else None)
        assert event['preceding_same_site_actions']==names[:index]
        assert event['invocation']==0 and event['batch_index']==0 and event['token_index']==2
        assert event['output_selector']==[0] and event['output_shape']==[1,10,768] and event['selected_shape']==[768]
        assert event['dtype']=='torch.float32' and event['device']=='cuda:0'
assert after[0] is None and after[1] is None and before[0]!=before[1]
assert before[2]==after[2]==before[0] and before[3]==before[0] and after[3]==before[1]
assert rows[0]['mask_logits_sha256']==rows[2]['mask_logits_sha256']
assert all(rows[0][t+'_bits']==rows[2][t+'_bits'] for t in ('is','are'))
raw_result=result_path.read_text(encoding='utf-8')
assert not re.search(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',raw_result,re.I)
normalized_result=raw_result.lower().replace('\\\\','/').replace('\\','/')
assert str(pathlib.Path.home()).lower().replace('\\','/') not in normalized_result
assert not re.search(r'(?:[a-z]:/users/|/home/|/users/)',normalized_result)
print('AUDIT PASS: source, pinned asset metadata, process/hash chain, 13 synchronized stages, controls, finite 768-float vectors, selected logits, privacy')
if len(sys.argv) == 5:
    print('OPTIONAL CACHE PASS: five pinned cached files matched size and SHA-256 at audit time')
print('Observed peak allocated/reserved MiB:',previous[0]/1024**2,previous[1]/1024**2)
print('is-minus-are logits target/donor/self/cross:',margins)
print('cross minus target margin:',margins[3]-margins[0])
print('Limits: one exposed development pair; full mask logits are unsaved; runtime UUID binding is an owner attestation.')
