"""Check only the public redacted worker-failure package."""
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent
def read(name): return json.loads((root/name).read_bytes())
for line in (root/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1); path=(root/name).resolve()
    assert path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
manifest=read('source-manifest.json')
for name,entry in manifest['files'].items():
    raw=(root/'source-snapshot'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['git_blob_sha1']
assert len(manifest['declared_source_paths'])==26 and len(manifest['cpu_test_paths'])==6
rows=[json.loads(line) for line in (root/'stages-redacted.jsonl').read_text().splitlines()]
names=('admission','after_model_transfer','trace_0','after_trace_0','trace_1','after_trace_1','trace_2','after_trace_2','before_worker','baseline','after_baseline','cancellation','after_cancellation','post_cancel','after_post_cancel','before_simulated_oom','simulated_oom')
assert [r['stage'] for r in rows]==list(names)
v={r['stage']:r['value'] for r in rows}
for name,outcome in (('trace_0','success'),('trace_1','synthetic_trace_cancellation'),('trace_2','success')):
    x=v[name]; assert x['outcome']==outcome and x['event_bytes']==64
    assert all(x[k] is True for k in ('capture_retained_until_sync','capture_released','trace_resources_retained_until_sync','trace_resources_released','trace_claim_released'))
assert v['baseline']['outcome']==v['post_cancel']['outcome']=='succeeded'
assert v['cancellation']=={'accepted':True,'outcome':'cancelled'}
oom=v['simulated_oom']; assert oom['injection_reached'] and oom['private_forward_error_type']=='OutOfMemoryError'
assert oom['public_error_type']=='WorkerSynchronizationError' and oom['worker_unsafe'] and oom['active_claim_count']==2 and oom['subsequent_work_refused'] and oom['no_explicit_checkpoint_after_unsafe']
memory_names=('admission','after_model_transfer','after_trace_0','after_trace_1','after_trace_2','before_worker','after_baseline','after_cancellation','after_post_cancel','before_simulated_oom')
points=[v[n] for n in memory_names]
assert all(0<=x['allocated_bytes']<=x['reserved_bytes']<=x['total_bytes'] and 2*1024**3<=x['free_bytes']<=x['total_bytes'] for x in points)
assert max(x['allocated_bytes'] for x in points)==67131904
assert max(x['reserved_bytes'] for x in points)==69206016
assert min(x['free_bytes'] for x in points)==7322206208
memory=read('independent-memory-audit.json'); assert memory['maximum_sampled_allocated_bytes']==67131904 and memory['maximum_sampled_reserved_bytes']==69206016 and memory['minimum_sampled_device_free_bytes']==7322206208
saved=read('independent-saved-audit.json'); assert saved['stage_count']==17 and saved['source_files_checked']==26
result=read('result-redacted.json'); assert result['controller_result']['status']=='incomplete' and result['controller_result']['terminal_record_present'] and result['controller_observation']['reaped']
assert result['supervision']['job_counts']==[[0,0],[1,0]] and result['supervision']['owner_exit']==1
recovery=read('qualified-recovery-record.json'); release=read('scheduler-release.json')
assert recovery['cuda_cleanup_verified'] is False and recovery['clean_attestation_created'] is False and release['scheduler_hold_released'] is True and release['cuda_cleanup_verified'] is False
assert release['qualified_recovery_sha256']==hashlib.sha256((root/'qualified-recovery-record.json').read_bytes()).hexdigest()
print('PASS: public hashes, 32 source/test files, 17 projected stages and 10 safe memory points')
