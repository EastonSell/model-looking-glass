"""Independent stdlib replay of saved year-comparison logits; no model call."""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
repo = Path(sys.argv[2]) if len(sys.argv) > 2 else root.parents[1]
run = root / 'run'
sha = lambda data: hashlib.sha256(data).hexdigest()
receipt = json.loads((root/'supervisor.json').read_bytes())
assert receipt['source_revision'] == '47d42b8137203fa8e5ffdc60942a1adece6c0feb'
assert receipt['source_tree'] == 'a7766a55470650e37c9d1c0b4d69acf1ef953a4a'
assert receipt['process_exit_code'] == 0 and receipt['failure_file_present'] is False
assert not (run/'failure.json').exists() and (root/'stderr.txt').stat().st_size == 0
assert sha((root/'pre-run.json').read_bytes()) == receipt['pre_run_sha256']
for name, digest in receipt['run_sha256'].items():
    assert sha((run/name).read_bytes()) == digest
assert sha((root/'stdout.txt').read_bytes()) == receipt['stdout_sha256']
assert sha((root/'stderr.txt').read_bytes()) == receipt['stderr_sha256']
manifest = json.loads((run/'manifest.json').read_bytes())
result = json.loads((run/'result.json').read_bytes())
source = manifest['source']
assert source['revision'] == receipt['source_revision'] and source['tree'] == receipt['source_tree']
for path, digest in source['sha256'].items():
    assert sha(subprocess.check_output(['git', 'show', source['revision'] + ':' + path], cwd=repo)) == digest
case_bytes = subprocess.check_output(['git', 'show', source['revision'] + ':experiments/distilgpt2_greater_than_screen_v1.json'], cwd=repo)
assert sha(case_bytes) == manifest['cases_sha256'] == '399994789ee4d3eff27dd6c9cce23ae27ab7761e4bc33caf58de4528194b01d2'
cases_data = json.loads(case_bytes)
cases, numeric = manifest['cases'], cases_data['numeric_ids']
assert cases == cases_data['cases'] and len(cases) == 48 and len(set(numeric)) == 100
assert result['manifest_sha256'] == sha((run/'manifest.json').read_bytes())
assert result['status'] == 'complete'
assert result['submitted_forwards'] == result['model_call_returns_observed'] == result['recorded_forwards'] == 48
assert result['model_state_sha256_before'] == result['model_state_sha256_after']
assert result['model_buffers_sha256_before'] == result['model_buffers_sha256_after']
events = [json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
assert len(events) == 145 and events[0]['event'] == 'runtime_ready'
raw = (run/'last_logits.f32').read_bytes()
size = 50257*4
assert len(raw) == 48*size
rows = []
for i, (case, record) in enumerate(zip(cases, result['records'])):
    payload = raw[i*size:(i+1)*size]
    meta = {'offset': i*size, 'bytes': size, 'sha256': sha(payload)}
    start, saved, complete = events[1+3*i:4+3*i]
    assert [x['event'] for x in (start,saved,complete)] == ['start','raw_recorded','complete']
    assert all(x['case_id'] == case['case_id'] for x in (start,saved,complete))
    assert record['case_id'] == case['case_id'] and record['raw'] == saved['raw'] == complete['raw'] == meta
    assert complete['score'] == record['score']
    vals = struct.unpack('<50257f', payload)
    assert all(math.isfinite(x) for x in vals)
    maximum = max(vals)
    weights = [math.exp(x-maximum) for x in vals]
    denom = math.fsum(weights)
    lo = math.fsum(weights[j] for j in case['lower_ids'])
    hi = math.fsum(weights[j] for j in case['upper_ids'])
    aligned = math.fsum(weights[j] for j in numeric[1:]) / denom
    zero = weights[numeric[0]] / denom
    scored = record['score']
    expected = {'log_mass_ratio': math.log(hi/lo), 'upper_conditional_fraction': hi/(hi+lo),
                'lower_full_mass': lo/denom, 'upper_full_mass': hi/denom,
                'equal_full_mass': weights[case['equal_id']]/denom,
                'aligned_01_99_full_mass': aligned, 'zero_00_next_token_mass': zero,
                'other_vocabulary_mass': max(0.0,1.0-aligned-zero), 'top_token_id': vals.index(maximum),
                'lower_logits': [vals[j] for j in case['lower_ids']],
                'upper_logits': [vals[j] for j in case['upper_ids']], 'equal_logit': vals[case['equal_id']]}
    assert set(scored) == set(expected)
    for key, value in expected.items():
        if isinstance(value, float):
            assert math.isclose(scored[key], value, rel_tol=1e-12, abs_tol=1e-12), (case['case_id'], key)
        else:
            assert scored[key] == value
    rows.append({'case_id': case['case_id'], 'year_suffix': case['year_suffix'], 'event': case['event'],
                 'condition': case['condition'], **expected})
groups = {}
for row in rows:
    groups.setdefault((row['year_suffix'],row['event']),{})[row['condition']] = row['log_mass_ratio']
triples = result['summary']['triples']
assert len(groups) == len(triples) == 16
for t in triples:
    g=groups[(t['year_suffix'],t['event'])]
    assert all(math.isclose(t[k],g[k],abs_tol=1e-12) for k in ('forward','reverse','neutral'))
    for key, value in {'forward_minus_reverse':g['forward']-g['reverse'],
                       'forward_minus_neutral':g['forward']-g['neutral'],
                       'reverse_minus_neutral':g['reverse']-g['neutral']}.items():
        assert math.isclose(t[key],value,abs_tol=1e-12)
for key in ('forward_minus_reverse','forward_minus_neutral','reverse_minus_neutral'):
    assert math.isclose(result['summary']['mean_'+key],math.fsum(t[key] for t in triples)/16,abs_tol=1e-12)
assert result['summary']['positive_forward_minus_reverse'] == sum(t['forward_minus_reverse']>0 for t in triples)
conditions = {}
for name in ('forward','reverse','neutral'):
    subset = [r for r in rows if r['condition']==name]
    conditions[name] = {'mean_M': math.fsum(r['log_mass_ratio'] for r in subset)/16,
                        'positive_M': sum(r['log_mass_ratio']>0 for r in subset),
                        'mean_aligned_01_99_mass': math.fsum(r['aligned_01_99_full_mass'] for r in subset)/16,
                        'mean_zero_00_mass': math.fsum(r['zero_00_next_token_mass'] for r in subset)/16,
                        'mean_other_mass': math.fsum(r['other_vocabulary_mass'] for r in subset)/16}
print(json.dumps({'schema':'distilgpt2-greater-than-independent-replay-v1',
                  'scope':'saved bytes only; no model forward', 'records_checked':48,
                  'journal_events_checked':145, 'receipt_sha256':sha((root/'supervisor.json').read_bytes()),
                  'result_sha256':sha((run/'result.json').read_bytes()), 'conditions':conditions,
                  'summary':result['summary'], 'rows':rows},sort_keys=True,indent=2,allow_nan=False))
