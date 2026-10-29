"""Recompute the public timing summary, without asserting device custody."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPECTED = {
    'journal.jsonl': '23fa2897658792e739db1df00ca6e1e470427523f8f4e3e322252d164ea21236',
    'schedule.json': 'ce2f990a6937da86565a21c52f13ad54f11b1e683421a8a2060e544694fe6bf9',
    'session-result.json': '50bdfb869ab16218e42a0c51b69c5f13d6b131cebf6c5ae5cd4a8e8282de1582',
}
for name, digest in EXPECTED.items():
    if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f'{name} digest differs')
source = REPO / 'looking_glass/tap_repeated_protocol.py'
if hashlib.sha256(source.read_bytes()).hexdigest() != '86b475a31d60a2158ee0de3ce0f4fe65068443335927bc0307d0daa8c1bf7503':
    raise SystemExit('protocol module differs from the recorded source')
spec = importlib.util.spec_from_file_location('_tap_v13_public_protocol', source)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
schedule = protocol.load_schedule(HERE / 'schedule.json')
result = json.loads((HERE / 'session-result.json').read_bytes())
context = json.loads((HERE / 'host-context.json').read_bytes())
rows = [json.loads(line) for line in (HERE / 'journal.jsonl').read_bytes().splitlines()]
total = result['slot']['device_totals']['cuda_bytes']
summary = protocol.summarize_session(schedule, 0, rows, clock=context['clock'],
    total_bytes=total, cap_bytes=total // 4, device='cuda:0', run_protocol=protocol.EXPERIMENT_PROTOCOL)
if summary != result['slot']['summary']:
    raise SystemExit('recomputed timing summary differs')
if (result['protocol'] != 'gpu-tap-latency-repeated-v1.3'
        or result['status'] != 'completed_exploratory_contaminated'
        or result['slot']['screen']['status'] != 'contaminated'):
    raise SystemExit('protocol or contamination classification differs')
print(json.dumps({'status': 'passed', 'rows': len(rows), 'measured_samples': summary['measured_samples'],
    'paired_rounds': len(summary['paired_rounds']), 'screen_status_as_reported': 'contaminated',
    'scope': 'public timing arithmetic only; private screen and process closure are not replayed'}, sort_keys=True))
