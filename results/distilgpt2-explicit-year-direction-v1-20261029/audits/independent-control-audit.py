"""Independent stdlib replay of fixed explicit-year study evidence.

Usage: python -B explicit_year_direction_saved_audit.py --run DIR --cases JSON
       --terminal-sha SHA256 [--source-commit COMMIT | --synthetic]

Read saved evidence only; this never executes a model or imports project code.
The output is a consistency check, not an attestation of model execution,
publication chronology, private exposure, or a unique mechanism.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

if not __debug__ or sys.byteorder != 'little':
    raise RuntimeError('requires assertions and a little-endian host')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', required=True, type=Path)
parser.add_argument('--cases', required=True, type=Path)
parser.add_argument('--terminal-sha', required=True)
parser.add_argument('--source-commit')
parser.add_argument('--synthetic', action='store_true')
args = parser.parse_args()
assert bool(args.source_commit) != args.synthetic

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def vec(raw):
    assert len(raw) % 4 == 0
    values = struct.unpack(f'<{len(raw)//4}f', raw)
    assert all(math.isfinite(v) for v in values)
    return values

def norm(delta):
    return math.sqrt(math.fsum(x*x for x in delta))

def lse_difference(values, left, right):
    a, b = [values[i] for i in left], [values[i] for i in right]
    assert len(a) == len(b) == 3 and len(set(left + right)) == 6
    u, v = max(a), max(b)
    return ((u-v) + math.log(math.fsum(math.exp(z-u) for z in a))) - math.log(math.fsum(math.exp(z-v) for z in b))

def scores_from_logits(raw, row, numeric_ids):
    assert len(raw) == 50257 * 4
    z = vec(raw)
    middle = lse_difference(z, row['middle_ids'], row['above_ids'])
    local = lse_difference(z, row['local_above_ids'], row['local_below_ids'])
    maximum = max(z)
    weights = [math.exp(x-maximum) for x in z]
    total = math.fsum(weights)
    numeric = set(numeric_ids)
    return {'band_score': middle, 'local_score': local,
            'aligned_01_99_full_mass': math.fsum(weights[i] for i in numeric_ids[1:]) / total,
            'zero_00_next_token_mass': weights[numeric_ids[0]] / total,
            'other_vocabulary_mass': math.fsum(w for i, w in enumerate(weights) if i not in numeric) / total,
            'top_token_id': z.index(maximum)}

cases_raw = args.cases.read_bytes()
assert digest(cases_raw) == '419d74c23333407f9a9dc0b6f7d7b70764ab5d1078e2b0a4c84d37ba194df8a9'
fixed = json.loads(cases_raw)
assert fixed['schema'] == 'distilgpt2-explicit-year-direction-cases-v1'
assert fixed['protocol_sha256'] == '68f028e401fd4c43d1cac89772da10c92cf5e5de7af30c4043c18c21678b902e'
rows = fixed['cases']
assert len(rows) == 48 and len({r['case_id'] for r in rows}) == 48
assert len(fixed['numeric_ids']) == len(set(fixed['numeric_ids'])) == 100
by_id = {r['case_id']: r for r in rows}
assert [(r['pair_noun_index'], r['condition'], r['year_role']) for r in rows] == [
    (p, c, y) for p in range(8) for c in ('after', 'before', 'near') for y in ('low', 'high')]
manifest_raw = (args.run/'manifest.json').read_bytes()
manifest = json.loads(manifest_raw)
terminal_raw = (args.run/'result.json').read_bytes()
assert digest(terminal_raw) == args.terminal_sha
result = json.loads(terminal_raw)
assert manifest['schema'] == 'distilgpt2-explicit-year-direction-run-v1'
assert manifest['cases'] == rows and manifest['cases_sha256'] == digest(cases_raw)
assert manifest['protocol_sha256'] == fixed['protocol_sha256']
assert result['schema'] == 'distilgpt2-explicit-year-direction-result-v1' and result['status'] == 'complete'
assert result['manifest_sha256'] == digest(manifest_raw)
if not args.synthetic:
    assert manifest['source']['revision'] == args.source_commit
    assert result['model_state_sha256_before'] == result['model_state_sha256_after']
    assert result['model_buffers_sha256_before'] == result['model_buffers_sha256_after']
assert result['interventions_performed'] is True
assert all(result[k] == 336 for k in ('submitted_forwards', 'model_call_returns_observed', 'returned_forwards'))
assert len(result['records']) == 336 and not (args.run/'failure.json').exists()

actions = ('self', 'year_final', 'year_at_zero', 'rotated_final')
expected_order = [(r['case_id'], a) for r in rows for a in ('clean', 'noop')]
for row in rows:
    shift = row['rotation']
    assert type(shift) is int and shift == (row['pair_noun_index'] + 2*row['condition_index'] + row['year_index']) % 4
    expected_order.extend((row['case_id'], a) for a in actions[shift:] + actions[:shift] + ('clean_after',))
assert [(r['case_id'], r['action']) for r in result['records']] == expected_order

# All saved segments are contiguous, hashed, finite, and linked to exactly one
# known case and scheduled action. Each full vocabulary is rescored from bits.
assert {p.name for p in args.run.glob('*.f32')} == {r['case_id']+'.f32' for r in rows}
case_files = {r['case_id']: (args.run/(r['case_id']+'.f32')).read_bytes() for r in rows}
cursor = {r['case_id']: 0 for r in rows}
records = {}
raw_total = 0
for record in result['records']:
    case_id, action = record['case_id'], record['action']
    row = by_id[case_id]
    pieces = record['raw']
    expected_names = ('logits',) if action in ('clean', 'clean_after') else ('logits', 'block_before', 'block_after')
    assert set(pieces) == set(expected_names)
    saved = {}
    for name in expected_names:
        part = pieces[name]
        assert part['offset'] == cursor[case_id]
        assert part['bytes'] == (50257*4 if name == 'logits' else len(row['input_ids'])*768*4)
        raw = case_files[case_id][part['offset']:part['offset']+part['bytes']]
        assert len(raw) == part['bytes'] and digest(raw) == part['sha256']
        saved[name] = raw
        cursor[case_id] += len(raw)
        raw_total += len(raw)
    recomputed = scores_from_logits(saved['logits'], row, fixed['numeric_ids'])
    assert recomputed == record['scores']
    assert (case_id, action) not in records
    records[case_id, action] = {'record': record, 'bytes': saved, 'scores': recomputed}
assert all(cursor[k] == len(v) for k, v in case_files.items())
assert raw_total == result['raw_payload_bytes'] == 86714688

def rec(case, action):
    return records[case['case_id'], action]

row_bytes = 768*4
for case in rows:
    clean, noop, after, self_action = (rec(case, a) for a in ('clean', 'noop', 'clean_after', 'self'))
    assert clean['bytes']['logits'] == noop['bytes']['logits'] == after['bytes']['logits'] == self_action['bytes']['logits']
    assert noop['bytes']['block_before'] == noop['bytes']['block_after']
    vec(noop['bytes']['block_before'])
    assert clean['record']['intervention'] is None and noop['record']['intervention'] is None
    assert after['record']['intervention'] is None

doses = []
for group in range(8):
    block = rows[6*group:6*group+6]
    pairs = {condition: {role: next(r for r in block if r['condition'] == condition and r['year_role'] == role)
                         for role in ('low', 'high')} for condition in ('after', 'before', 'near')}
    distances = {}
    for condition, pair in pairs.items():
        lo = rec(pair['low'], 'noop')['bytes']['block_before']
        hi = rec(pair['high'], 'noop')['bytes']['block_before']
        assert lo[:row_bytes] == hi[:row_bytes]
        distances[condition] = norm([b-a for a, b in zip(vec(lo[-row_bytes:]), vec(hi[-row_bytes:]))])
    q = .5 * min(distances.values())
    assert q > 2**-20
    family = {'pair_noun_index': group, 'dose': q, 'norms': distances,
              'fractions': {name: q/value for name, value in distances.items()}}
    assert result['family_doses'][group] == family
    doses.append(family)
    for condition, pair in pairs.items():
        for role in ('low', 'high'):
            row, donor = pair[role], pair['high' if role == 'low' else 'low']
            own = rec(row, 'noop')['bytes']['block_before']
            other = rec(donor, 'noop')['bytes']['block_before']
            displacement = [(b-a) * (q/distances[condition])
                            for a, b in zip(vec(own[-row_bytes:]), vec(other[-row_bytes:]))]
            for action in actions:
                entry = rec(row, action)
                before, after = entry['bytes']['block_before'], entry['bytes']['block_after']
                assert before == own
                pos = 0 if action == 'year_at_zero' else len(row['input_ids'])-1
                original = before[pos*row_bytes:(pos+1)*row_bytes]
                if action == 'self':
                    delivered = original
                    requested = 0.
                else:
                    values = vec(original)
                    d = ([displacement[(i-1) % 768] for i in range(768)]
                         if action == 'rotated_final' else displacement)
                    delivered = struct.pack('<768f', *(a+b for a, b in zip(values, d)))
                    requested = q
                expected = before[:pos*row_bytes] + delivered + before[(pos+1)*row_bytes:]
                assert after == expected
                actual = norm([a-b for a, b in zip(vec(delivered), vec(original))])
                changed = sum(a != b for a, b in zip(struct.unpack('<768I', original), struct.unpack('<768I', delivered)))
                if action != 'self':
                    assert abs(actual-q) <= 1e-5*q
                info = entry['record']['intervention']
                assert info == {'position': pos, 'requested': requested, 'achieved': actual,
                                'changed_bits': changed, 'expected_before_hex': original.hex(),
                                'expected_after_hex': delivered.hex(), 'observed_achieved': actual,
                                'observed_changed_bits': changed}

# Reconstruct the six strict comparator inequalities and the distinct
# generic-shift and comparator-pattern decisions directly from saved scores.
clean_groups, intervention_groups = [], []
for group in range(8):
    subset = rows[6*group:6*group+6]
    case = {c: {y: next(r for r in subset if r['condition'] == c and r['year_role'] == y)
                for y in ('low', 'high')} for c in ('after', 'before', 'near')}
    local = {c: {y: rec(case[c][y], 'clean')['scores']['local_score'] for y in ('low', 'high')}
             for c in case}
    fixed_d = {c: rec(case[c]['low'], 'clean')['scores']['band_score'] -
               rec(case[c]['high'], 'clean')['scores']['band_score'] for c in case}
    checks = {'after_low': local['after']['low'] > .5, 'after_high': local['after']['high'] > .5,
              'before_low': local['before']['low'] < -.5, 'before_high': local['before']['high'] < -.5,
              'D_after': fixed_d['after'] > .5, 'D_before': fixed_d['before'] < -.5}
    clean_groups.append({'pair_index': group//2, 'event': subset[0]['event'], 'local_scores': local,
                         'local_wording_differences': {f'{a}_minus_{b}': {y: local[a][y]-local[b][y]
                            for y in ('low', 'high')} for a,b in (('after','before'),('after','near'),('before','near'))},
                         'fixed_band_D': fixed_d, 'fixed_band_wording_differences': {
                            f'{a}_minus_{b}': fixed_d[a]-fixed_d[b]
                            for a,b in (('after','before'),('after','near'),('before','near'))},
                         'checks': checks, 'comparator_pass': all(checks.values())})
    effects = {}
    for condition in case:
        effects[condition] = {}
        for action in actions:
            lo, hi = case[condition]['low'], case[condition]['high']
            le = rec(lo, action)['scores']['band_score'] - rec(lo, 'clean')['scores']['band_score']
            he = rec(hi, action)['scores']['band_score'] - rec(hi, 'clean')['scores']['band_score']
            effects[condition][action] = {'low_E': le, 'high_E': he, 'low_donorward': -le,
                                          'high_donorward': he, 'donorward_mean': (-le+he)/2}
    contrasts = {a: effects['after'][a]['donorward_mean'] - effects['before'][a]['donorward_mean'] for a in actions}
    generic = all(effects[c]['year_final']['low_donorward'] > 0 and effects[c]['year_final']['high_donorward'] > 0
                  and all(effects[c]['year_final']['donorward_mean'] > effects[c][a]['donorward_mean']+.1
                          for a in ('year_at_zero', 'rotated_final')) for c in case)
    directional = (all(effects['after']['year_final'][f'{y}_donorward'] > 0 for y in ('low','high')) and
                   all(effects['before']['year_final'][f'{y}_donorward'] < 0 for y in ('low','high')))
    controlled = directional and all(contrasts['year_final'] > contrasts[a]+.1 for a in ('year_at_zero','rotated_final'))
    intervention_groups.append({'pair_index': group//2, 'event': subset[0]['event'], 'conditions': effects,
                                'after_minus_before': contrasts, 'generic_pattern': generic,
                                'comparator_direction': directional, 'comparator_control': controlled})
passed = [x for x in clean_groups if x['comparator_pass']]
gate = len(passed) >= 6 and len({x['pair_index'] for x in passed}) >= 3
expected_clean = {'pairs': clean_groups, 'comparator_gate_count': len(passed), 'comparator_gate_passed': gate}
if not args.synthetic or 'check_counts' in result['clean_summary']:
    expected_clean['check_counts'] = {key: sum(g['checks'][key] for g in clean_groups) for key in clean_groups[0]['checks']}
assert result['clean_summary'] == expected_clean
count = {key: sum(g[key] for g in intervention_groups) for key in ('generic_pattern', 'comparator_direction','comparator_control')}
expected_effects = {'pairs': intervention_groups,
    'mean_donorward_by_condition_and_action': {c: {a: math.fsum(g['conditions'][c][a]['donorward_mean']
       for g in intervention_groups)/8 for a in actions} for c in ('after','before','near')},
    'mean_after_minus_before_by_action': {a: math.fsum(g['after_minus_before'][a] for g in intervention_groups)/8 for a in actions},
    'generic_pattern_count': count['generic_pattern'], 'generic_pattern_supported': count['generic_pattern'] >= 6,
    'comparator_direction_count': count['comparator_direction'],
    'comparator_direction_supported': count['comparator_direction'] >= 6 if gate else None,
    'comparator_control_count': count['comparator_control'],
    'comparator_control_supported': count['comparator_control'] >= 6 if gate else None}
assert result['intervention_summary'] == expected_effects

# Journal events bind every scored record and all intermediate gate/dose entries.
events = [json.loads(line) for line in (args.run/'attempt.jsonl').read_bytes().splitlines()]
assert len(events) == 1019
assert events[0] == {'event': 'runtime_ready', 'actual_runtime': result['runtime']}
at, nrecord = 1, 0
def check_events():
    global at, nrecord
    record = result['records'][nrecord]
    start, saved, complete = events[at:at+3]
    assert start['event'] == 'start' and saved['event'] == 'raw_recorded' and complete['event'] == 'complete'
    assert all(e['case_id'] == record['case_id'] and e['action'] == record['action'] for e in (start,saved,complete))
    assert saved['raw'] == record['raw'] and {k:v for k,v in complete.items() if k != 'event'} == record
    if record['intervention'] is None:
        assert start['intervention'] is None
    else:
        assert set(start['intervention']) == {'position', 'requested', 'achieved', 'changed_bits',
                                              'expected_before_hex', 'expected_after_hex'}
        assert all(start['intervention'][k] == record['intervention'][k] for k in start['intervention'])
    at += 3
    nrecord += 1
for _ in range(96):
    check_events()
assert events[at] == {'event': 'position_zero_check', 'pairs_checked': 24}
assert events[at+1] == {'event': 'clean_gates', 'summary': expected_clean}
at += 2
for family in doses:
    assert events[at] == {'event': 'family_dose', **family}
    at += 1
    for _ in range(30):
        check_events()
assert at == len(events) and nrecord == 336
report = {'status': 'passed', 'terminal_sha256': digest(terminal_raw), 'manifest_sha256': digest(manifest_raw),
          'cases_sha256': digest(cases_raw), 'source_commit': manifest['source']['revision'],
          'forwards': 336, 'raw_bytes': raw_total, 'comparator_gate': expected_clean,
          'intervention_summary': expected_effects, 'family_doses': doses,
          'scope': 'saved-byte consistency only; no model execution or exposure/chronology attestation'}
print(json.dumps(report, sort_keys=True, allow_nan=False))
