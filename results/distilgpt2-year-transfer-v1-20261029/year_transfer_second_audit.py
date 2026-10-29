"""Replay the closed public year-transfer attempt from saved CPU evidence only."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

if not __debug__:
    raise RuntimeError('do not run this auditor with Python optimization enabled')
if len(sys.argv) > 2:
    raise SystemExit('usage: python audit_year_transfer.py [attempt-directory]')
attempt = Path(sys.argv[1]).resolve(strict=True) if len(sys.argv) == 2 else Path(__file__).resolve().parent
run = attempt/'run'
receipt_raw = (attempt/'supervisor.json').read_bytes()
assert hashlib.sha256(receipt_raw).hexdigest() == '1bd457a772a4985bf223624ea5248e8343771bde97495b240d9717748d997a42'
receipt = json.loads(receipt_raw)
assert receipt['process_exit_code'] == 0 and receipt['interrupted'] is False
assert receipt['failure_file_present'] is False and not (run/'failure.json').exists()
for name, expected in receipt['run_sha256'].items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest() == expected, name
assert {p.name for p in run.iterdir() if p.is_file()} == set(receipt['run_sha256'])
for name in ('stdout', 'stderr'):
    assert hashlib.sha256((attempt/(name+'.txt')).read_bytes()).hexdigest() == receipt[name+'_sha256']
pre_raw = (attempt/'pre-run.json').read_bytes()
assert hashlib.sha256(pre_raw).hexdigest() == receipt['pre_run_sha256']
pre = json.loads(pre_raw)
manifest_raw = (run/'manifest.json').read_bytes()
result_raw = (run/'result.json').read_bytes()
result = json.loads(result_raw)
manifest = json.loads(manifest_raw)
assert hashlib.sha256(manifest_raw).hexdigest() == result['manifest_sha256']
assert result['status'] == 'complete' and result['interventions_performed'] is True
assert result['submitted_forwards'] == result['model_call_returns_observed'] == result['returned_forwards'] == 336
assert pre['source_revision'] == receipt['source_revision'] == manifest['source']['revision']
assert pre['source_tree'] == receipt['source_tree'] == manifest['source']['tree']
assert pre['script_sha256'] == manifest['source']['sha256']['scripts/year_transfer_v1.py']
assert pre['case_sha256'] == manifest['cases_sha256']
assert pre['protocol_sha256'] == manifest['protocol_sha256']
assert result['checkpoint']['model_id'] == pre['intended_execution']['model_id']
assert result['checkpoint']['model_revision'] == pre['intended_execution']['model_revision']
assert result['checkpoint']['weights_sha256'] == pre['intended_execution']['weights_sha256']
assert result['model_state_sha256_before'] == result['model_state_sha256_after']
assert result['model_buffers_sha256_before'] == result['model_buffers_sha256_after']

rows = manifest['cases']
assert len(rows) == 48 and len(result['records']) == 336
by_id = {row['case_id']: row for row in rows}
assert len(by_id) == 48
records = result['records']
expected_order = [(r['case_id'], action) for r in rows for action in ('clean', 'noop')]
for row in rows:
    actions = ('self', 'year_final', 'year_at_zero', 'rotated_final')
    shift = row['rotation']
    expected_order.extend((row['case_id'], action) for action in actions[shift:]+actions[:shift]+('clean_after',))
assert [(rec['case_id'], rec['action']) for rec in records] == expected_order
grouped = {key: {} for key in by_id}
for record in records:
    assert record['action'] not in grouped[record['case_id']]
    grouped[record['case_id']][record['action']] = record

data_by_id = {}
score_by_id = {}
total = logit_bytes = segments = 0
for row in rows:
    case_id = row['case_id']
    data = (run/(case_id+'.f32')).read_bytes()
    total += len(data)
    item, ranges, scores = {}, [], {}
    for action, record in grouped[case_id].items():
        payload = {}
        expected_parts = {'logits'} if action in ('clean', 'clean_after') else {'logits', 'block_before', 'block_after'}
        assert set(record['raw']) == expected_parts
        for name, ref in record['raw'].items():
            start, end = ref['offset'], ref['offset']+ref['bytes']
            raw = data[start:end]
            assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256']
            ranges.append((start, end))
            payload[name] = raw
            segments += 1
        logits = payload['logits']
        assert len(logits) == 50257*4
        assert all(bits & 0x7f800000 != 0x7f800000 for (bits,) in struct.iter_unpack('<I', logits))
        logit_bytes += len(logits)
        mid = [struct.unpack_from('<f', logits, index*4)[0] for index in row['middle_ids']]
        above = [struct.unpack_from('<f', logits, index*4)[0] for index in row['above_ids']]
        u, v = max(mid), max(above)
        score = ((u-v)+math.log(math.fsum(math.exp(x-u) for x in mid))) - math.log(
            math.fsum(math.exp(x-v) for x in above))
        assert score == record['band_score']
        scores[action] = score
        item[action] = payload
    cursor = 0
    for first, last in sorted(ranges):
        assert first == cursor and last > first
        cursor = last
    assert cursor == len(data)
    assert all(item[a]['logits'] == item['clean']['logits'] for a in ('noop', 'self', 'clean_after'))
    assert item['noop']['block_before'] == item['noop']['block_after']
    assert len(item['noop']['block_before']) == len(row['input_ids'])*3072
    data_by_id[case_id], score_by_id[case_id] = item, scores
assert total == result['raw_payload_bytes'] == 87206208
assert logit_bytes == 336*50257*4 and segments == 816

def vector(raw):
    assert len(raw) == 3072
    values = struct.unpack('<768f', raw)
    assert all(math.isfinite(x) for x in values)
    return values


def norm(values):
    return math.sqrt(math.fsum(x*x for x in values))


def group_row(group, condition, role):
    return next(row for row in group if row['condition'] == condition and row['year_role'] == role)


conditions = ('forward', 'reverse', 'neutral')
actions = ('self', 'year_final', 'year_at_zero', 'rotated_final')
gate_comp = gate_pref = pref_count = comparator_count = 0
comp_pairs, pref_pairs = set(), set()
effect_values = {condition: {action: [] for action in actions} for condition in conditions}
for family in range(8):
    group = rows[family*6:(family+1)*6]
    pair = group[0]['pair_index']
    event = group[0]['event']
    assert all(row['pair_index'] == pair and row['event'] == event for row in group)
    rawpairs = {}
    norms = {}
    clean_delta = {}
    for condition in conditions:
        low = group_row(group, condition, 'low')
        high = group_row(group, condition, 'high')
        lo = data_by_id[low['case_id']]['noop']['block_before']
        hi = data_by_id[high['case_id']]['noop']['block_before']
        assert lo[:3072] == hi[:3072]
        lv, hv = vector(lo[-3072:]), vector(hi[-3072:])
        norms[condition] = norm(b-a for a, b in zip(lv, hv))
        rawpairs[condition] = (lo, hi, lv, hv)
        clean_delta[condition] = score_by_id[low['case_id']]['clean'] - score_by_id[high['case_id']]['clean']
    dose = 0.5*min(norms.values())
    saved_dose = result['family_doses'][family]
    assert saved_dose['pair_noun_index'] == family and saved_dose['dose'] == dose and saved_dose['norms'] == norms
    assert all(saved_dose['fractions'][c] == dose/norms[c] for c in conditions)
    assert dose > 2**-20
    f, r, n = (clean_delta[c] for c in conditions)
    comp = f > .5 and r < -.5
    pref = f > .5 and n > .5 and f-n > .1
    gate_comp += comp
    gate_pref += pref
    if comp: comp_pairs.add(pair)
    if pref: pref_pairs.add(pair)
    saved_clean = result['clean_summary']['pairs'][family]
    assert saved_clean['pair_index'] == pair and saved_clean['event'] == event
    assert all(saved_clean[c] == clean_delta[c] for c in conditions)
    assert saved_clean['comparator_pass'] == comp and saved_clean['preference_pass'] == pref
    assert saved_clean['forward_minus_neutral'] == f-n
    computed_effects = {}
    for condition in conditions:
        computed_effects[condition] = {}
        lo, hi, lv, hv = rawpairs[condition]
        for role, before, own, donor in (('low', lo, lv, hv), ('high', hi, hv, lv)):
            row = group_row(group, condition, role)
            case_id = row['case_id']
            assert row['donor_case_id'] == group_row(group, condition, 'high' if role == 'low' else 'low')['case_id']
            displacement = tuple((dose/norms[condition])*(b-a) for a, b in zip(own, donor))
            for action in actions:
                payload = data_by_id[case_id][action]
                meta = grouped[case_id][action]['intervention']
                assert payload['block_before'] == before
                position = 0 if action == 'year_at_zero' else len(row['input_ids'])-1
                start = position*3072
                source = before[start:start+3072]
                source_values = vector(source)
                delta = (tuple(displacement[(i-1) % 768] for i in range(768))
                         if action == 'rotated_final' else displacement)
                expected = source if action == 'self' else struct.pack('<768f', *(a+b for a, b in zip(source_values, delta)))
                actual = payload['block_after']
                assert actual[:start] == before[:start] and actual[start+3072:] == before[start+3072:]
                assert actual[start:start+3072] == expected
                assert meta['position'] == position and bytes.fromhex(meta['expected_before_hex']) == source
                assert bytes.fromhex(meta['expected_after_hex']) == expected
                achieved = norm(a-b for a, b in zip(vector(expected), source_values))
                changed = sum(a != b for a, b in zip(struct.iter_unpack('<I', source), struct.iter_unpack('<I', expected)))
                requested = 0.0 if action == 'self' else dose
                assert abs(achieved-requested) <= 1e-5*requested
                assert meta['requested'] == requested and meta['achieved'] == meta['observed_achieved'] == achieved
                assert meta['changed_bits'] == meta['observed_changed_bits'] == changed
        for action in actions:
            low, high = (group_row(group, condition, role) for role in ('low', 'high'))
            low_effect = score_by_id[low['case_id']][action] - score_by_id[low['case_id']]['clean']
            high_effect = score_by_id[high['case_id']][action] - score_by_id[high['case_id']]['clean']
            value = {'low_E': low_effect, 'high_E': high_effect, 'low_donorward': -low_effect,
                     'high_donorward': high_effect, 'donorward_mean': (-low_effect+high_effect)/2}
            computed_effects[condition][action] = value
            effect_values[condition][action].append(value['donorward_mean'])
    fe, ne, re = (computed_effects[c]['year_final'] for c in ('forward', 'neutral', 'reverse'))
    preference_pattern = (fe['low_donorward'] > 0 and fe['high_donorward'] > 0
                          and ne['low_donorward'] > 0 and ne['high_donorward'] > 0
                          and fe['donorward_mean']-ne['donorward_mean'] > .1
                          and fe['donorward_mean'] > computed_effects['forward']['year_at_zero']['donorward_mean']
                          and fe['donorward_mean'] > computed_effects['forward']['rotated_final']['donorward_mean'])
    comparator_pattern = (fe['low_donorward'] > 0 and fe['high_donorward'] > 0
                          and re['low_donorward'] < 0 and re['high_donorward'] < 0)
    pref_count += preference_pattern
    comparator_count += comparator_pattern
    saved_effect = result['intervention_summary']['pairs'][family]
    assert saved_effect['conditions'] == computed_effects
    assert saved_effect['preference_pattern'] == preference_pattern
    assert saved_effect['comparator_pattern'] == comparator_pattern
    assert saved_effect['forward_minus_neutral'] == fe['donorward_mean']-ne['donorward_mean']
clean = result['clean_summary']
assert clean['preference_gate_count'] == gate_pref and clean['comparator_gate_count'] == gate_comp
assert clean['preference_gate_passed'] == (gate_pref >= 6 and len(pref_pairs) >= 3)
assert clean['comparator_gate_passed'] == (gate_comp >= 6 and len(comp_pairs) >= 3)
effects = result['intervention_summary']
assert effects['preference_pattern_count'] == pref_count and effects['comparator_pattern_count'] == comparator_count
assert effects['preference_pattern_supported'] == (pref_count >= 6 if clean['preference_gate_passed'] else None)
assert effects['comparator_pattern_supported'] == (comparator_count >= 6 if clean['comparator_gate_passed'] else None)
means = {condition: {action: math.fsum(effect_values[condition][action])/8 for action in actions}
         for condition in conditions}
assert effects['mean_donorward_by_condition_and_action'] == means

journal = [json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
for kind in ('start', 'raw_recorded', 'complete'):
    selected = [item for item in journal if item['event'] == kind]
    assert [(item['case_id'], item['action']) for item in selected] == expected_order
    if kind == 'complete':
        assert [{k: v for k, v in item.items() if k != 'event'} for item in selected] == records
assert sum(item['event'] == 'position_zero_check' for item in journal) == 1
assert sum(item['event'] == 'clean_gates' for item in journal) == 1
assert sum(item['event'] == 'family_dose' for item in journal) == 8
assert not any(item['event'] == 'failed' for item in journal)
report = {'schema': 'independent-year-transfer-saved-evidence-review-v1',
          'scope': 'saved evidence and direct-process receipt consistency; not an independent execution attestation',
          'supervisor_sha256': hashlib.sha256(receipt_raw).hexdigest(),
          'result_sha256': hashlib.sha256(result_raw).hexdigest(),
          'pre_run_sha256': hashlib.sha256(pre_raw).hexdigest(),
          'source_revision': pre['source_revision'], 'forwards': 336, 'payload_bytes': total,
          'logit_bytes': logit_bytes, 'raw_segments': segments,
          'preference_gate_count': gate_pref, 'preference_gate_passed': clean['preference_gate_passed'],
          'comparator_gate_count': gate_comp, 'comparator_gate_passed': clean['comparator_gate_passed'],
          'preference_pattern_count': pref_count, 'preference_pattern_supported': effects['preference_pattern_supported'],
          'comparator_pattern_count': comparator_count, 'comparator_pattern_supported': effects['comparator_pattern_supported'],
          'mean_donorward': means}
raw = (json.dumps(report, sort_keys=True, separators=(',', ':'))+'\n').encode()
sys.stdout.buffer.write(raw)
