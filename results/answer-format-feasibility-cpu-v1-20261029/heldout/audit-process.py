"""Independent CPU-only saved-row, provenance and process-consistency review."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np
from tokenizers import Tokenizer

root = Path(__file__).resolve().parents[2]
checkout = root/'work/full_top_reservation_checkout'
attempt = root/'work/answer_format_real/answer-format-heldout-308c30b5e74440f6b53e177a3f1e59f6'
bare = root/'work/reviews/answer_format_remote_readback.git'
source='266b8c72a4fb1065a078ffbf033a65216c5f79ef'
head='a8acc76aa64332dfb96964284c7d78ccf2c51b5e'
manifest_path='experiments/answer_format_feasibility_manifest_v1.json'
phase_sha='d6e1cdb4fb724d38105c6bd80cd73cab50cee513515783a2ad316f9421742a69'
manifest_sha='6767fc7c95ffc89809d45df0991dce526d23aadd4c93536cc71e316d5c08bd78'
case_sha='cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a'
inventory_sha='921025a7f9c1246c5f10e97f2b987ca4971462241b2d05ebbc525d4aa0736c79'
sha=lambda b:hashlib.sha256(b).hexdigest()
blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def strict(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise AssertionError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(AssertionError('nonfinite JSON')))
def git(*args):return subprocess.check_output(['git','-C',str(checkout),*args])
def tree(rev):
    out={}
    for line in git('ls-tree','-rz','--full-tree',rev).split(b'\0'):
        if line:
            meta,name=line.split(b'\t',1);mode,typ,oid=meta.decode().split(' ')
            out[name.decode()]=(mode,typ,oid)
    return out

assert git('rev-parse','HEAD').decode().strip()==head and not git('status','--porcelain').strip()
assert git('show','-s','--format=%P',head).decode().split()==['e56a9cd43f9401c760053d6a99cbcd75714f1e80','3627367a9ac0e3f1558988b76cf9f0cbf295d42f']
manifest_raw=(checkout/manifest_path).read_bytes()
assert sha(manifest_raw)==manifest_sha
assert subprocess.check_output(['git','--git-dir',str(bare),'show',f'{head}:{manifest_path}'])==manifest_raw
m=strict(manifest_raw); assert m['source_commit']==source and m['case_sha256']==case_sha
st,ht=tree(source),tree(head)
sources=m['source_sha256']
for name,digest in sources.items():
    data=(checkout/name).read_bytes()
    assert sha(data)==digest and st[name]==ht[name]==('100644','blob',blob(data)),name
assert {p for p in st if p.startswith('looking_glass/') and p.endswith('.py')} == {p for p in sources if p.startswith('looking_glass/') and p.endswith('.py')}
phase_raw=(attempt/'phase-record.json').read_bytes();assert sha(phase_raw)==phase_sha
phase=strict(phase_raw)
assert phase['version']=='answer-format-phase-v1' and phase['status']=='complete' and phase['phase']=='heldout'
assert phase['source_commit']==source and phase['checkout_commit']==head and phase['manifest_sha256']==manifest_sha
expected={'child-result.json','custody-observation.json','development-review-attestation.json','heldout/journal.jsonl','heldout/matrix.f32',
          'heldout/results.json','heldout/state-original.json','heldout/top-tokens.json',
          'manifest-review-attestation.json','public-exposure-attestation.json','public-exposure.json','started.json'}
assert set(phase['artifacts'])==expected
assert {p.relative_to(attempt).as_posix() for p in attempt.rglob('*') if p.is_file()} == expected|{'phase-record.json'}
assert (attempt/'pycache').is_dir() and not any((attempt/'pycache').iterdir())
for name,item in phase['artifacts'].items():
    p=attempt/name;assert p.is_file() and not p.is_symlink(),name
    data=p.read_bytes();assert item=={'bytes':len(data),'sha256':sha(data)},name
child=strict((attempt/'child-result.json').read_bytes());start=strict((attempt/'started.json').read_bytes())
review=strict((attempt/'manifest-review-attestation.json').read_bytes())
attest=strict((attempt/'public-exposure-attestation.json').read_bytes())
custody=strict((attempt/'custody-observation.json').read_bytes())
assert child==phase['child'] and phase['child_result_sha256']==sha((attempt/'child-result.json').read_bytes())
assert phase['started_sha256']==sha((attempt/'started.json').read_bytes())
sup=phase['supervisor']
assert sup['clean'] is True and type(sup['exit_code']) is int and sup['exit_code']==0 and sup['timed_out'] is False
assert phase['owner_reaped'] is True and phase['job_drained'] is True
assert type(sup['pid']) is int and sup['pid']>0 and child['owner_pid']==sup['pid']
assert type(sup['run_id']) is str and len(sup['run_id'])==32 and child['owner_run_id']==sup['run_id']
assert child['phase']=='heldout' and child['manifest_sha256']==manifest_sha
assert child['model_disposed'] is True and child['no_model_loaded'] is False
assert child['model_state_sha256']==m['original_model_state_sha256']
assert child['weight_sha256']==m['original_weight_sha256'] and child['buffer_state_sha256']==m['baseline_buffer_state_sha256']
assert child['runtime_settings']==m['baseline_runtime_settings']
assert child['runtime_settings']['cuda_visibility']=='-1' and child['runtime_settings']['cuda_initialized'] is False
assert child['runtime_settings']['intraop_threads']==child['runtime_settings']['interop_threads']==1
assert child['post_work_checkpoint_assets']==m['checkpoint']['asset_sha256']
assert all(sources.get(k)==v for k,v in child['imported_sources'].items())
assert {'looking_glass/answer_format_feasibility.py','looking_glass/answer_format_feasibility_worker.py',
        'looking_glass/cpu_study_supervision.py','looking_glass/pretrained.py'} <= set(child['imported_sources'])
for name in ('custody-observation.json','manifest-review-attestation.json','public-exposure-attestation.json','public-exposure.json'):
    key={'custody-observation.json':'custody_observation_sha256','manifest-review-attestation.json':'manifest_review_sha256',
         'public-exposure-attestation.json':'public_exposure_attestation_sha256','public-exposure.json':'public_exposure_file_sha256'}[name]
    assert start[key]==sha((attempt/name).read_bytes())
assert start['custodian_observations']['observations']==custody['observations']
assert start['development_review_sha256']==sha((attempt/'development-review-attestation.json').read_bytes())
assert start['development_record_sha256']=='643ee3b5c44c8901d4e443ab25ded57e83d8e1d0bf8b93ff1ca003b646e4c659'
development_review=strict((attempt/'development-review-attestation.json').read_bytes())
assert development_review['status']=='passed' and development_review['reviewed_commit']=='e56a9cd43f9401c760053d6a99cbcd75714f1e80'
assert development_review['case_sha256']==case_sha and development_review['manifest_sha256']==manifest_sha
development_root='results/answer-format-feasibility-cpu-v1-20261029/development/'
development_files={'attempt/phase-record.json':'643ee3b5c44c8901d4e443ab25ded57e83d8e1d0bf8b93ff1ca003b646e4c659',
 'attempt/development/matrix.f32':'2cbe55c3abf6bccb612c39403f423668ed2b38cde4452d4df22e0a56b85056fa',
 'attempt/development/results.json':'f72ec335c563c6810280a1e08a998e08bf81c61170bd22d0b2c3e69201660d0a',
 'independent-audit.json':'07573eb58b2ee90fd93daec4e23c570ddc0762c19a4fe235becdb04e11c4fcba',
 'independent-process-audit.json':'2702cdb8e1b828c49a4e81c71d479b7e6ee81f683d95d32a87295bb274ff8345'}
for name,digest in development_files.items():
    data=(checkout/development_root/name).read_bytes()
    assert sha(data)==digest and subprocess.check_output(['git','--git-dir',str(bare),'show',f"{development_review['reviewed_commit']}:{development_root+name}"])==data,name
assert development_review['phase_sha256']==development_files['attempt/phase-record.json']
assert development_review['matrix_sha256']==development_files['attempt/development/matrix.f32']
assert development_review['result_sha256']==development_files['attempt/development/results.json']
assert development_review['audit_sha256']==development_files['independent-audit.json']
assert datetime.fromisoformat(development_review['observed_utc']) <= datetime.fromisoformat(start['started_utc'])
dev_result=strict((checkout/development_root/'attempt/development/results.json').read_bytes())
assert dev_result['score']['status']=='ineligible' and dev_result['score']['selected_family'] is None
assert start['phase']=='heldout' and start['source_commit']==source and start['checkout_commit']==head
assert start['manifest_sha256']==review['manifest_sha256']==manifest_sha and review['reviewed_commit']=='8dda000371337d8718d0c97960ffd68bdf1ba408'
assert review['reviewer']=='review_trace_budget' and review['observed_utc']=='2026-10-29T10:48:19+00:00'
assert sha((attempt/'public-exposure.json').read_bytes())==inventory_sha
assert (attempt/'public-exposure.json').read_bytes()==(root/'work/year-label-v30-draft/inventory.canonical.json').read_bytes()
assert attest['reviewer']=='review_trace_budget' and attest['reviewed_commit']=='3627367a9ac0e3f1558988b76cf9f0cbf295d42f'
assert attest['latest_main_commit']=='afab9f093f68be0636042b9098ec579e167400f5'
assert attest['latest_inventory_version']=='distilgpt2-development-exposures-v30'
assert attest['inventory_sha256']==attest['latest_inventory_sha256']==inventory_sha
assert attest['source_sha256']=='944a81a00e0ce57c691fdfda32b6ef8b2089e85168281dbfd18255c62e845682'
assert attest['no_unreserved_relevant_inputs_reported'] is True
assert 'no complete unknown-history guarantee' in attest['currentness_scope']
custodians={x['custodian']:x for x in custody['observations']}
assert set(custodians)=={'natural','injection'}
assert custodians['natural']['reported_by']=='review_localization' and custodians['natural']['case_sha256']==case_sha
assert custodians['natural']['status']=='public_frame_clear' and custodians['natural']['original_split_unchanged'] is True
assert custodians['natural']['successor_status']=='no_successor_split_reported'
assert custodians['injection']['reported_by']=='fixtures' and custodians['injection']['status']=='no_protected_rows_reported'
assert all('no-later-than' in x['time_scope'].lower() for x in custodians.values())
timestamps=[m['created_utc'],review['observed_utc'],start['started_utc'],child['finished_utc'],phase['finished_utc']]
times=[datetime.fromisoformat(x) for x in timestamps]
assert all(x.tzinfo is not None for x in times) and all(a<=b for a,b in zip(times,times[1:]))
for row in custody['observations']:
    assert 0<=(times[2]-datetime.fromisoformat(row['observed_utc'])).total_seconds()<=870
assert 0<=(times[2]-datetime.fromisoformat(attest['observed_utc'])).total_seconds()<=870
state=strict((attempt/'heldout/state-original.json').read_bytes())
probe={'normal_times_half_bits':4194304,'numpy_cast_preserves_min_subnormal':True,'subnormal_times_one_bits':1}
assert state['cleanup']=='verified' and state['condition']=='original'
for field,expected_value in [('model_state_sha256',child['model_state_sha256']),('weight_sha256',child['weight_sha256']),
 ('buffer_state_sha256',child['buffer_state_sha256']),('runtime_settings',child['runtime_settings'])]:assert state[field]==expected_value
assert state['applied_weight_sha256']==child['weight_sha256']
assert state['addition_denormal_probe_before']==state['addition_denormal_probe_after']==probe
result=strict((attempt/'heldout/results.json').read_bytes());score=result['score']
assert result['stage']=='heldout' and result['case_sha256']==case_sha
assert result['identity']=={'source_commit':source,'manifest_sha256':manifest_sha}
assert result['score_denormal_probe_before']==result['score_denormal_probe_after']==probe
assert result['development_matrix_sha256']=='2cbe55c3abf6bccb612c39403f423668ed2b38cde4452d4df22e0a56b85056fa'
assert result['development_result_sha256']=='f72ec335c563c6810280a1e08a998e08bf81c61170bd22d0b2c3e69201660d0a'
assert result['development_replay_denormal_probe_before']==result['development_replay_denormal_probe_after']==probe
assert child['result_sha256']==sha((attempt/'heldout/results.json').read_bytes())
for name,item in result['artifacts'].items():assert item==phase['artifacts']['heldout/'+name]
assert set(result['artifacts'])=={'matrix.f32','journal.jsonl','state-original.json'}
matrix=(attempt/'heldout/matrix.f32').read_bytes(); row_size=50257*4
assert len(matrix)==32*row_size
journal=[strict(line) for line in (attempt/'heldout/journal.jsonl').read_bytes().splitlines()]
decoded=strict((attempt/'heldout/top-tokens.json').read_bytes())
cases=strict((checkout/'experiments/answer_format_feasibility_cases_v1.json').read_bytes())
assert sha((checkout/'experiments/answer_format_feasibility_cases_v1.json').read_bytes())==case_sha
rows=[r for r in cases['cases'] if r['role']=='heldout']
order=list(range(32));random.Random(20261101).shuffle(order)
assert score['execution_order']==order and len(rows)==len(journal)==len(decoded)==len(score['observations'])==32
cache=root/'work/hf-cache/hub/models--distilbert--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa'
for name,expected_sha in m['checkpoint']['asset_sha256'].items():
    h=hashlib.sha256()
    with (cache/name).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==expected_sha,name
tokenizer=Tokenizer.from_file(str(cache/'tokenizer.json'))
observed={}
for index,(j,s,t) in enumerate(zip(journal,score['observations'],decoded)):
    case=rows[order[index]];offset=index*row_size;raw=matrix[offset:offset+row_size]
    assert j['case_id']==s['case_id']==t['case_id']==case['case_id']
    assert j['execution_index']==s['execution_index']==index and j['canonical_index']==s['canonical_index']==order[index]
    assert j['offset']==s['logit_byte_offset']==offset and j['bytes']==s['logit_bytes']==row_size
    assert j['sha256']==s['logit_sha256']==sha(raw)
    assert j['forward_denormal_probe_before']==j['forward_denormal_probe_after']==probe
    bits=np.frombuffer(raw,dtype='<u4');exponent=((bits>>23)&255).astype(np.int64)
    assert not np.any(exponent==255)
    mantissa=(bits&0x7fffff).astype(np.float64)
    promoted=np.ldexp(np.where(exponent==0,mantissa,mantissa+2**23),np.where(exponent==0,-149,exponent-150))
    promoted=np.where((bits&0x80000000)!=0,-promoted,promoted)
    maximum=float(promoted.max());tie=np.flatnonzero(promoted==maximum);rep=int(tie[0]);unique=rep if len(tie)==1 else None
    assert s['unique_top_id']==t['unique_top_id']==unique and s['top_representative_id']==t['top_representative_id']==rep
    assert s['top_tie_count']==t['top_tie_count']==len(tie)
    assert t['unique_top_text']==(None if unique is None else tokenizer.decode([unique],skip_special_tokens=False))
    assert t['unique_top_raw_token']==(None if unique is None else tokenizer.id_to_token(unique))
    assert t['top_representative_text']==tokenizer.decode([rep],skip_special_tokens=False)
    assert t['top_representative_raw_token']==tokenizer.id_to_token(rep)
    denom=math.log(math.fsum(math.exp(float(v)-maximum) for v in promoted))
    for label in ('factual','counterfactual','madrid','stockholm'):
        token=case['labels'][label]['token_id'];value=float(promoted[token]);entry=s['scores'][label]
        assert entry['token_id']==token and entry['logit']==value
        assert math.isclose(entry['logp'],value-maximum-denom,rel_tol=1e-12,abs_tol=1e-12)
    assert s['factual_top'] is (unique==case['labels']['factual']['token_id'])
    observed[(case['fact_id'],case['family'],case['prefix'])]=s['factual_top']
facts=('france','japan','italy','uk');families=('statement','question');prefixes=('none','neutral','spain','sweden')
robust={family:[fact for fact in facts if all(observed[(fact,family,p)] for p in prefixes)] for family in families}
for family in score['families']:
    assert family['robust_facts']==robust[family['family']] and family['eligible'] is (len(robust[family['family']])>=3)
chosen=next((f for f in families if len(robust[f])>=3),None)
assert score['selected_family']==child['selected_family'] is None
assert score['status']==child['selection_status']=='ineligible'
assert score['reproduced_development_facts']==[]
chosen=None
report={'schema':'independent-answer-format-heldout-process-v1','status':'passed','phase_sha256':phase_sha,
 'manifest_sha256':manifest_sha,'source_commit':source,'checkout_commit':head,'source_files_verified':len(sources),
 'matrix_sha256':sha(matrix),'result_sha256':child['result_sha256'],'journal_rows':32,'finite_full_vocabulary_rows':32,
 'selected_family':chosen,'selection_status':score['status'],'robust_facts':robust,'asset_count':len(m['checkpoint']['asset_sha256']),
 'review_completed_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Saved-byte, row and cooperative process/report consistency; not authenticated execution, timing, private disjointness or unknown history.'}
out=root/'work/reviews/answer_format_heldout_process_audit.json'
out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,sort_keys=True,indent=2))
