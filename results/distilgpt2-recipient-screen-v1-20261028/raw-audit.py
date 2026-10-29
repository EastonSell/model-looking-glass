"""Independent stdlib replay of one saved recipient-screen attempt; no model call."""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
run = root / 'run'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
receipt = json.loads((root / 'supervisor.json').read_bytes())
assert receipt['source_revision'] == '32164cd570dea1f4bc0022c5b7f81fe17f8c2797'
assert receipt['process_exit_code'] == 0 and receipt['failure_file_present'] is False
assert sha((root / 'pre-run.json').read_bytes()) == receipt['pre_run_sha256']
assert not (run / 'failure.json').exists() and (root / 'stderr.txt').stat().st_size == 0
for name, digest in receipt['run_sha256'].items():
    assert sha((run / name).read_bytes()) == digest
manifest = json.loads((run / 'manifest.json').read_bytes())
result = json.loads((run / 'result.json').read_bytes())
assert result['manifest_sha256'] == receipt['run_sha256']['manifest.json']
assert manifest['source']['revision'] == receipt['source_revision']
assert manifest['source']['tree'] == receipt['source_tree']
assert manifest['cases_sha256'] == 'df9036ece115a42851a43f2d7833865fa1258ab542e322beff16d549df27d345'
assert result['submitted_forwards'] == result['model_call_returns_observed'] == result['recorded_forwards'] == 64
assert result['model_state_sha256_before'] == result['model_state_sha256_after']
assert result['model_buffers_sha256_before'] == result['model_buffers_sha256_after']
assert manifest['expected_forwards'] == 64
cases, records = manifest['cases'], result['records']
events = [json.loads(line) for line in (run / 'attempt.jsonl').read_bytes().splitlines()]
assert len(cases) == len(records) == 64 and len(events) == 193
assert events[0]['event'] == 'runtime_ready'
raw = (run / 'last_logits.f32').read_bytes()
size = 50257 * 4
assert len(raw) == 64 * size
rows = []
for i, (case, record) in enumerate(zip(cases, records)):
    assert record['case_id'] == case['case_id']
    start, saved, complete = events[1+3*i:4+3*i]
    assert [x['event'] for x in (start, saved, complete)] == ['start', 'raw_recorded', 'complete']
    assert all(x['case_id'] == case['case_id'] for x in (start, saved, complete))
    expected = {'offset': i * size, 'bytes': size, 'sha256': sha(raw[i*size:(i+1)*size])}
    assert record['raw'] == saved['raw'] == complete['raw'] == expected
    assert record['score'] == complete['score']
    logits = struct.unpack('<50257f', raw[i*size:(i+1)*size])
    assert all(math.isfinite(value) for value in logits)
    rid, gid = case['recipient_id'], case['giver_id']
    maximum = max(logits)
    logsum = math.log(math.fsum(math.exp(x - maximum) for x in logits))
    def rank(index):
        return 1 + sum(x > logits[index] or (x == logits[index] and k < index) for k, x in enumerate(logits))
    expected_score = {'recipient_logit': logits[rid], 'giver_logit': logits[gid],
                      'margin': logits[rid] - logits[gid],
                      'recipient_log_probability': (logits[rid] - maximum) - logsum,
                      'giver_log_probability': (logits[gid] - maximum) - logsum,
                      'recipient_rank': rank(rid), 'giver_rank': rank(gid),
                      'top_token_id': logits.index(maximum)}
    assert record['score'] == expected_score
    rows.append({'case_id': case['case_id'], 'pair': case['pair'], 'template': case['template'],
                 'order': case['order'], 'giver_index': case['giver_index'], **expected_score})
matched = {}
for row in rows:
    matched.setdefault((row['pair'], row['template'], row['order']), []).append(row['margin'])
assert len(matched) == 32 and all(len(group) == 2 for group in matched.values())
paired = {key: all(value > 0 for value in vals) for key, vals in matched.items()}
by_template = [sum(paired[(p, t, o)] for p in range(8) for o in range(2)) for t in range(2)]
by_order = [sum(paired[(p, t, o)] for p in range(8) for t in range(2)) for o in range(2)]
by_pair = [sum(paired[(p, t, o)] for t in range(2) for o in range(2)) for p in range(8)]
gate = sum(paired.values()) >= 24 and min(by_template) >= 12 and sum(x >= 3 for x in by_pair) >= 6
summary = result['summary']
assert summary['gate_passed'] == gate and summary['both_positive_pairs'] == sum(paired.values())
assert summary['template_both_positive'] == by_template and summary['order_both_positive'] == by_order
assert summary['name_pair_both_positive'] == by_pair
assert summary['positive_cases'] == sum(row['margin'] > 0 for row in rows)
assert summary['mean_margin'] == math.fsum(row['margin'] for row in rows) / 64
condition = lambda subset: {'count': len(subset), 'positive': sum(r['margin'] > 0 for r in subset),
                            'mean_margin': math.fsum(r['margin'] for r in subset) / len(subset),
                            'recipient_top1': sum(r['recipient_rank'] == 1 for r in subset),
                            'giver_top1': sum(r['giver_rank'] == 1 for r in subset)}
analysis = {'schema': 'distilgpt2-recipient-independent-replay-v1', 'scope': 'saved bytes only; no model forward',
            'receipt_sha256': sha((root / 'supervisor.json').read_bytes()),
            'result_sha256': sha((run / 'result.json').read_bytes()),
            'records_checked': 64, 'journal_events_checked': len(events), 'summary': summary,
            'overall': condition(rows),
            'by_template': [condition([r for r in rows if r['template'] == t]) for t in range(2)],
            'by_order': [condition([r for r in rows if r['order'] == o]) for o in range(2)],
            'by_name_pair': [condition([r for r in rows if r['pair'] == p]) for p in range(8)],
            'rows': rows}
print(json.dumps(analysis, sort_keys=True, indent=2, allow_nan=False))
