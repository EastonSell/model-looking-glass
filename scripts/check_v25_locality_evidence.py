"""Read-only verification of public locality rows in the merged evaluation."""
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if not __debug__:
    raise RuntimeError('the evidence checker requires assertions enabled')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looking_glass import pretrained_exposures_v25 as exposure

commit = '475228289efa7840fa577833d17573a6a73e001e'
root = 'results/targeted-edit-screened-matched-v1-20261029/evaluation/'
def read(path):
    return subprocess.check_output(['git', 'show', commit + ':' + path])

case_bytes = read('experiments/targeted_edit_screened_public_cases_v1.json')
assert hashlib.sha256(case_bytes).hexdigest() == 'fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3'
cases = {r['case_id']: r for r in json.loads(case_bytes)['cases'] if r['role'] == 'locality'}
assert len(cases) == 12
assert tuple((r['case_id'], r['prompt'], tuple(r['token_ids'])) for r in cases.values()) == exposure.LOCALITY_INPUTS
journal = read(root + 'evaluation/observations.jsonl')
assert hashlib.sha256(journal).hexdigest() == '76477d15ec279b28827591b3d643767b51321e68b54196eee49134c35dd4c809'
phase_bytes = read(root + 'phase-record.json')
assert hashlib.sha256(phase_bytes).hexdigest() == '79faacd17bac96039729e241e0ef92b853f70b8b8a9fb326b09ff1e4fbc4bd78'
phase = json.loads(phase_bytes)
assert phase['status'] == 'complete' and phase['owner_reaped'] is True and phase['job_drained'] is True
events = [json.loads(line) for line in journal.splitlines()]
conditions = ('original', 'projected', 'unprojected', 'projected_random', 'opposite', 'nuisance', 'matched')
counts = collections.Counter()
for condition in conditions:
    raw = read(root + 'evaluation/' + condition + '.f32')
    identity = phase['artifacts']['evaluation/' + condition + '.f32']
    assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256']
    selected = [event for event in events if event['condition'] == condition and event['case_id'] in cases]
    assert len(selected) == 12
    assert {event['case_id'] for event in selected} == set(cases)
    for event in selected:
        assert event['source'] == 'forward' and event['bytes'] == 201028
        payload = raw[event['offset']:event['offset'] + event['bytes']]
        assert len(payload) == event['bytes'] and hashlib.sha256(payload).hexdigest() == event['sha256']
        counts[event['case_id']] += 1
assert set(counts) == set(cases) and set(counts.values()) == {7}
print('PASS: 12 locality case IDs, 7 direct-forward saved rows each, all 84 slice hashes; cooperative execution scope')
